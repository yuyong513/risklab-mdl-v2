"""CSV 加载层：编码探测、列清单 / dtype 推断、角色推断（**推断不阻断**）。

设计要点（对应 ``docs/tool-contract.md`` §3.1 与 ``docs/data-feature.md``）
----------------------------------------------------------------------------
1. **编码**：一律走 :func:`risklab_mdl.platform.fs.detect_encoding`（BOM 优先 →
   UTF-8 → gb18030），不自己写探测逻辑。探测到的编码读不动时按
   ``utf-8 → gb18030 → utf-8-sig → utf-16`` 顺序重试，全失败抛
   ``encoding_undecodable``（**不静默返回空表**）。
2. **dtype 推断**只给"语义类型"，用于决策（能不能分箱 / 要不要先编码），
   不改变原始存储。语义类型取值见 :data:`SEMANTIC_DTYPES`。
3. **角色推断不阻断**：``id / time / split / target_candidates`` 只是候选，
   每个都带理由；目标字段由调用方通过 ``config.target`` 显式确认。
   推断结果永远附带 ``"角色为推断，需用户确认"`` 警告。
4. 读表只做一次：宽表下 ``usecols`` 让下游步骤只读需要的列（性能边界）。

★ 为什么角色推断不能阻断
    同一列在不同业务里含义不同（``flag`` 可能是标签、也可能是渠道标记），
    静态推断必然有错判。早期版本的弹窗式确认被废弃后，语义变成
    「给候选 + 给理由 + 让调用方显式传 target」，因此推断只提供线索。
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..platform.fs import (
    EncodingUndecodable,
    PlatformError,
    atomic_write_text,
    detect_encoding,
    require_readable,
)
from ..core.utils import safe_ratio, same_value_rate

__all__ = [
    "NUMERIC_SEMANTIC_TYPES",
    "ROLE_CONFIRM_WARNING",
    "SEMANTIC_DTYPES",
    "CsvRead",
    "as_numeric",
    "column_infos",
    "infer_roles",
    "infer_semantic_dtype",
    "load_dataset",
    "read_csv_auto",
    "to_jsonable",
]

#: 语义 dtype 取值（推断结果，不是 pandas 存储类型）。
SEMANTIC_DTYPES: tuple[str, ...] = (
    "bool",
    "integer",
    "float",
    "datetime",
    "categorical",
    "text",
    "empty",
)

#: 可参与分箱/描述统计的语义类型。
NUMERIC_SEMANTIC_TYPES: tuple[str, ...] = ("bool", "integer", "float")

#: 角色推断必带的警告（契约要求：推断不阻断，必须让用户知道要确认）。
ROLE_CONFIRM_WARNING = "角色为推断，需用户确认（id/time/split/target_candidates 仅为候选）"

#: 判定「像数值」的阈值：非空值能被 to_numeric 解析的比例。
_NUMERIC_PARSE_RATIO = 0.95

#: 类别型 vs 文本型的分界（唯一值个数）。
_CATEGORICAL_MAX_UNIQUE = 20

#: 高基数的绝对下限（唯一值个数）。
_HIGH_CARDINALITY_MIN = 50

#: 编码尝试链（探测结果排第一）。
_ENCODING_RETRY_CHAIN: tuple[str, ...] = ("utf-8", "gb18030", "utf-8-sig", "utf-16")

_ROLE_CONFIRM = ROLE_CONFIRM_WARNING


# --------------------------------------------------------------------------- #
# JSON 友好化
# --------------------------------------------------------------------------- #
def to_jsonable(value: Any) -> Any:
    """把 numpy / pandas / Path / datetime 标量转成可 JSON 序列化的 Python 值。

    递归处理 dict / list / tuple；``NaN`` / ``inf`` 统一转 ``None``
    （JSON 没有 NaN，直接 dumps 会产出非法 JSON）。
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (int,)):
        return value
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return None if (math.isnan(out) or math.isinf(out)) else out
    if isinstance(value, (np.datetime64, pd.Timestamp, datetime)):
        return str(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    return str(value)


# --------------------------------------------------------------------------- #
# 读表
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CsvRead:
    """一次读表的结果（含实际使用的编码）。"""

    path: Path
    encoding: str
    frame: pd.DataFrame
    warnings: tuple[str, ...] = ()

    @property
    def rows(self) -> int:
        return int(self.frame.shape[0])

    @property
    def columns(self) -> List[str]:
        return [str(c) for c in self.frame.columns]


def read_csv_auto(
    csv_path: Any,
    *,
    usecols: Optional[Sequence[str]] = None,
    nrows: Optional[int] = None,
) -> CsvRead:
    """用平台层探测编码后读 CSV。

    - 探测到的编码解不开时按 :data:`_ENCODING_RETRY_CHAIN` 重试；
    - 全部失败 ⇒ :class:`~risklab_mdl.platform.fs.EncodingUndecodable`
      （``error.code=encoding_undecodable``），不返回空表；
    - 空文件（无任何数据行）不报错，但 ``warnings`` 显式说明。
    """
    path = require_readable(csv_path, label="CSV 文件")
    detected = detect_encoding(path)
    order: List[str] = [detected] + [e for e in _ENCODING_RETRY_CHAIN if e != detected]

    warnings: List[str] = []
    errors: List[str] = []
    for encoding in order:
        try:
            frame = pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
                usecols=list(usecols) if usecols is not None else None,
                nrows=nrows,
            )
        except UnicodeError as exc:
            # 注意 catch UnicodeError 而不是 UnicodeDecodeError：utf-16 分支在
            # 没有 BOM 时抛的是 UnicodeError("UTF-16 stream does not start with BOM")
            reason = getattr(exc, "reason", None) or str(exc)
            errors.append(f"{encoding}: {reason}")
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            if isinstance(exc, pd.errors.EmptyDataError):
                empty = pd.DataFrame()
                warnings.append("CSV 解析为空（没有表头或没有数据行），请确认文件内容")
                return CsvRead(path=path, encoding=encoding, frame=empty, warnings=tuple(warnings))
            raise PlatformError(
                f"CSV 结构无法解析：{path}（{exc}）",
                code="csv_parse_error",
                hint="检查分隔符/引号是否配对；Excel 导出的 CSV 请另存为「CSV UTF-8」后重试",
            ) from exc
        if encoding != detected:
            warnings.append(
                f"按探测编码 {detected} 读取失败，已退回 {encoding} 重试成功"
                f"（文件可能混合编码，建议统一另存为 UTF-8）"
            )
        return CsvRead(path=path, encoding=encoding, frame=frame, warnings=tuple(warnings))

    raise EncodingUndecodable(
        f"CSV 无法按任何已知编码解码：{path}；尝试过 {order}；失败原因：" + "；".join(errors),
        hint=(
            "该文件可能不是文本（二进制/加密/压缩），或编码不在支持范围。"
            "请用 Excel/编辑器另存为 UTF-8（或 UTF-16、GB18030）后重试。"
        ),
    )


# --------------------------------------------------------------------------- #
# dtype 推断
# --------------------------------------------------------------------------- #
def write_csv_frame(frame: pd.DataFrame, path: Any) -> Path:
    """原子写 CSV（UTF-8 无 BOM，``\\n`` 换行），返回写入路径。

    与 :func:`read_csv_auto` 对称：所有会写 CSV 产物的模块共用这一份，
    避免有的地方带索引、有的地方用系统默认换行。
    """
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    target = Path(path)
    atomic_write_text(target, buffer.getvalue())
    return target


def _non_null_sample(series: pd.Series, *, limit: int = 500) -> pd.Series:
    """取非空值样本（用于解析判定，避免宽表下全量转换）。"""
    non_null = series.dropna()
    if len(non_null) <= limit:
        return non_null
    return non_null.iloc[:limit]


def as_numeric(series: pd.Series) -> Tuple[pd.Series, float]:
    """尝试把一列转成数值，返回 ``(转换结果, 解析成功率)``。

    解析成功率 = 可解析的非空值 / 非空值总数。全空列记 0.0。
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return pd.to_numeric(series, errors="coerce"), 0.0
    converted = pd.to_numeric(series, errors="coerce")
    ok = int(converted.notna().sum())
    return converted, float(ok / len(non_null))


def is_numeric_series(series: pd.Series) -> bool:
    """该列是否可作为数值特征（原生数值，或可解析为数值的对象/字符串列）。"""
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return True
    _, ratio = as_numeric(series)
    return ratio >= _NUMERIC_PARSE_RATIO


def infer_semantic_dtype(series: pd.Series) -> str:
    """推断语义 dtype（见 :data:`SEMANTIC_DTYPES`）。

    对象/字符串列先试数值解析：整个 CSV 里「数字被读成字符串」很常见
    （混入一个 ``"-"`` 就会整列退化），这一步把它们救回数值语义。
    """
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if not len(series.dropna()):
        return "empty"

    converted, ratio = as_numeric(series)
    if ratio >= _NUMERIC_PARSE_RATIO:
        sample = converted.dropna()
        if len(sample) and bool((sample % 1 == 0).all()):
            return "integer"
        return "float"

    nunique = int(series.nunique(dropna=True))
    return "categorical" if nunique <= _CATEGORICAL_MAX_UNIQUE else "text"


def _is_high_cardinality(nunique: int, n_rows: int, dtype: str) -> bool:
    if dtype not in ("categorical", "text"):
        return False
    if nunique < _HIGH_CARDINALITY_MIN:
        return False
    return nunique > max(_HIGH_CARDINALITY_MIN, int(0.5 * max(n_rows, 1)))


# --------------------------------------------------------------------------- #
# 角色推断
# --------------------------------------------------------------------------- #
_ID_NAMES = {
    "id",
    "pid",
    "uid",
    "uuid",
    "guid",
    "key",
    "index",
    "rowid",
    "row_id",
    "sample_id",
    "user_id",
    "cust_id",
    "customer_id",
    "order_id",
    "acct_id",
    "account_id",
    "card_id",
    "loan_id",
    "apply_id",
}
_ID_SUFFIXES = ("_id", "_uuid", "_guid", "_key", "_no")
_ID_PREFIXES = ("id_",)

_TIME_NAMES = {
    "month",
    "months",
    "date",
    "dt",
    "time",
    "day",
    "week",
    "period",
    "observe_month",
    "data_date",
    "stat_date",
    "year",
    "ym",
    "y_m",
    "yrmth",
    "月份",
    "月",
    "年月",
    "日期",
    "时间",
    "账期",
    "数据日期",
    "观察月",
}
_TIME_SUFFIXES = ("_month", "_date", "_dt", "_time", "_day", "_period")
_TIME_SUFFIXES_CN = ("月份", "日期", "年月")

_SPLIT_NAMES = {
    "split",
    "split_set",
    "dataset",
    "data_set",
    "sample_set",
    "set",
    "part",
    "phase",
    "subset",
    "fold",
}
_SPLIT_VALUES = {
    "train",
    "training",
    "train_set",
    "val",
    "valid",
    "validation",
    "dev",
    "test",
    "testing",
    "test_set",
    "oot",
    "holdout",
    "hold_out",
}

_TARGET_NAMES = {
    "y",
    "Y",
    "target",
    "label",
    "labels",
    "label_bad",
    "is_bad",
    "bad",
    "bad_flag",
    "flag",
    "dpd",
    "overdue",
    "overdue_flag",
    "default",
    "default_flag",
    "result",
    "outcome",
    "response",
    "target_flag",
    "y_flag",
    "违约",
    "好坏",
    "标签",
    "目标",
}
_TARGET_NAME_HINTS = ("target", "label", "bad", "default", "overdue", "dpd", "flag", "y")

#: 日期样式的取值正则（``2026-01`` / ``202601`` / ``2026-01-31`` / ``2026年1月`` / ``2026/1``）。
_DATE_LIKE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\d{4}-\d{2}$"),
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    re.compile(r"^\d{4}/\d{1,2}$"),
    re.compile(r"^\d{4}/\d{1,2}/\d{1,2}$"),
    re.compile(r"^\d{6}$"),
    re.compile(r"^\d{8}$"),
    re.compile(r"^\d{4}年\d{1,2}月$"),
    re.compile(r"^\d{4}年\d{1,2}月\d{1,2}日$"),
    re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}"),
)

_MONTHLY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(\d{4})-(\d{1,2})$"),
    re.compile(r"^(\d{4})/(\d{1,2})$"),
    re.compile(r"^(\d{4})年(\d{1,2})月$"),
    re.compile(r"^(\d{4})(\d{2})$"),
    re.compile(r"^(\d{4})-(\d{2})-\d{2}$"),
)


def _looks_like_dates(series: pd.Series, *, sample_limit: int = 200, ratio: float = 0.9) -> bool:
    """抽样判定一列是否「像日期」（不真的解析成 datetime，避免污染原值）。"""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    sample = non_null.iloc[:sample_limit].astype(str).str.strip()
    hits = sum(1 for value in sample if any(p.match(value) for p in _DATE_LIKE_PATTERNS))
    return safe_ratio(hits, len(sample)) >= ratio


def _is_split_column(series: pd.Series, name_lower: str) -> Tuple[bool, str]:
    if name_lower in _SPLIT_NAMES:
        return True, f"列名 {name_lower!r} 命中划分字段命名（split/dataset/sample_set …）"
    non_null = series.dropna()
    if len(non_null) == 0:
        return False, ""
    values = {str(v).strip().lower() for v in non_null.iloc[:200].unique()}
    if values and values <= _SPLIT_VALUES:
        return True, f"取值全部落在 {sorted(_SPLIT_VALUES)} 内（train/val/test …）"
    return False, ""


def _is_id_column(
    series: pd.Series, name_lower: str, *, n_rows: int, nunique: int
) -> Tuple[bool, str]:
    if name_lower in _ID_NAMES:
        return True, f"列名 {name_lower!r} 命中标识列命名"
    for prefix in _ID_PREFIXES:
        if name_lower.startswith(prefix):
            return True, f"列名以 {prefix!r} 开头（标识列命名习惯）"
    for suffix in _ID_SUFFIXES:
        if name_lower.endswith(suffix):
            return True, f"列名以 {suffix!r} 结尾（标识列命名习惯）"
    if n_rows > 1 and nunique == n_rows:
        return True, f"取值唯一（{nunique}/{n_rows}），且无重复 ⇒ 疑似唯一标识"
    return False, ""


def _is_time_column(
    series: pd.Series, name_lower: str, semantic_dtype: str
) -> Tuple[bool, str]:
    if name_lower in _TIME_NAMES:
        return True, f"列名 {name_lower!r} 命中时间列命名"
    for suffix in _TIME_SUFFIXES:
        if name_lower.endswith(suffix):
            return True, f"列名以 {suffix!r} 结尾（时间列命名习惯）"
    for suffix in _TIME_SUFFIXES_CN:
        if name_lower.endswith(suffix):
            return True, f"列名以 {suffix!r} 结尾（中文时间列命名习惯）"
    if semantic_dtype == "datetime":
        return True, "pandas 已解析为 datetime64 类型"
    if _looks_like_dates(series):
        return True, "抽样取值形如日期/月份（如 2026-01、202601、2026年1月）"
    return False, ""


def _target_priority(
    series: pd.Series, name_lower: str
) -> Tuple[int, str]:
    """给目标候选打分（数值越大越靠前），返回 ``(分数, 理由)``。"""
    nunique = int(series.nunique(dropna=True))
    if nunique < 2:
        return 0, ""
    score = 0
    reasons: List[str] = []
    if name_lower in {n.lower() for n in _TARGET_NAMES}:
        score += 60
        reasons.append(f"列名 {name_lower!r} 命中标签列命名（y/target/label/flag …）")
    else:
        for hint in _TARGET_NAME_HINTS:
            if hint in name_lower:
                score += 25
                reasons.append(f"列名包含 {hint!r}，疑似标签")
                break
    numeric_like = is_numeric_series(series)
    if numeric_like and nunique == 2:
        score += 30
        reasons.append(f"数值列且只有 {nunique} 个取值（典型二分类标签）")
    elif numeric_like and nunique <= 10:
        score += 10
        reasons.append(f"数值列且取值数 {nunique} ≤ 10（可能是多分类标签）")
    if not numeric_like:
        score = min(score, 20)  # 文本列不当目标首选
    return score, "；".join(reasons)


def infer_roles(frame: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """推断 ``id / time / split / target_candidates``，返回 ``(roles, reasons)``。

    ★ 只推断、不阻断：目标字段仍需调用方通过 ``config.target`` 显式确认。
    """
    rows = int(frame.shape[0])
    roles: Dict[str, Any] = {"id": None, "time": None, "split": None, "target_candidates": []}
    reasons: Dict[str, Any] = {
        "id": None,
        "time": None,
        "split": None,
        "target_candidates": {},
    }

    stats: Dict[str, Tuple[int, str]] = {}
    for column in frame.columns:
        name = str(column)
        series = frame[column]
        stats[name] = (int(series.nunique(dropna=True)), infer_semantic_dtype(series))

    taken: set[str] = set()

    # ① split：先占位（它的取值集合非常特征化，误判率最低）
    for column in frame.columns:
        name = str(column)
        hit, why = _is_split_column(frame[column], name.lower())
        if hit:
            roles["split"] = name
            reasons["split"] = why
            taken.add(name)
            break

    # ② time
    for column in frame.columns:
        name = str(column)
        if name in taken:
            continue
        hit, why = _is_time_column(frame[column], name.lower(), stats[name][1])
        if hit:
            roles["time"] = name
            reasons["time"] = why
            taken.add(name)
            break

    # ③ id
    for column in frame.columns:
        name = str(column)
        if name in taken:
            continue
        nunique = stats[name][0]
        hit, why = _is_id_column(frame[column], name.lower(), n_rows=rows, nunique=nunique)
        if hit:
            roles["id"] = name
            reasons["id"] = why
            taken.add(name)
            break

    # ④ target 候选（按分数排序，最多 5 个）
    scored: List[Tuple[int, str, str]] = []
    for column in frame.columns:
        name = str(column)
        if name in taken:
            continue
        score, why = _target_priority(frame[column], name.lower())
        if score > 0 and why:
            scored.append((score, name, why))
    scored.sort(key=lambda item: (-item[0], item[1]))
    roles["target_candidates"] = [name for _, name, _ in scored[:5]]
    reasons["target_candidates"] = {name: why for _, name, why in scored[:5]}

    reasons["_note"] = _ROLE_CONFIRM
    return roles, reasons


# --------------------------------------------------------------------------- #
# 列清单
# --------------------------------------------------------------------------- #
def column_infos(
    frame: pd.DataFrame,
    roles: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """逐列画像：``{name, dtype, role, nunique, missing_rate, same_value_rate}``。"""
    rows = int(frame.shape[0])
    role_of: Dict[str, str] = {}
    target_candidates: List[str] = []
    if roles:
        for key in ("id", "time", "split"):
            value = roles.get(key)
            if value:
                role_of[str(value)] = key
        target_candidates = [str(c) for c in (roles.get("target_candidates") or [])]
        for column in target_candidates:
            role_of.setdefault(column, "target")

    out: List[Dict[str, Any]] = []
    for column in frame.columns:
        name = str(column)
        series = frame[column]
        semantic = infer_semantic_dtype(series)
        nunique = int(series.nunique(dropna=True))
        missing_rate = safe_ratio(float(series.isna().sum()), float(rows))
        out.append(
            {
                "name": name,
                "dtype": semantic,
                "pandas_dtype": str(series.dtype),
                "role": role_of.get(name, "feature"),
                "nunique": nunique,
                "missing_rate": round(missing_rate, 6),
                "same_value_rate": round(same_value_rate(series), 6),
                "is_high_cardinality": _is_high_cardinality(nunique, rows, semantic),
            }
        )
    return out


def load_dataset(csv_path: Any) -> Dict[str, Any]:
    """``dataset_load`` 的数据载荷：编码 / 行数列数 / 角色 / dtype。

    返回结构（不含三态外壳，由工具层包装）::

        {"csv_path", "encoding", "rows", "columns", "roles", "role_reasons",
         "dtypes", "column_infos", "warnings"}
    """
    read = read_csv_auto(csv_path)
    frame = read.frame
    roles, reasons = infer_roles(frame)
    infos = column_infos(frame, roles)
    warnings: List[str] = list(read.warnings)
    warnings.append(_ROLE_CONFIRM)
    if not roles["target_candidates"]:
        warnings.append(
            "未推断出目标字段候选，请在 config.target.column 显式指定标签列"
        )
    if roles["time"] is None:
        warnings.append("未推断出时间列：与时间相关的分析与图表将被跳过（不伪造）")
    return {
        "csv_path": read.path.as_posix(),
        "encoding": read.encoding,
        "rows": read.rows,
        "columns": read.columns,
        "roles": roles,
        "role_reasons": reasons,
        "dtypes": {info["name"]: info["dtype"] for info in infos},
        "column_infos": infos,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# 目标配置
# --------------------------------------------------------------------------- #
def normalize_target(
    target: Optional[Dict[str, Any]],
    *,
    fallback_candidates: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """规范化 ``config.target``；缺列或列不存在时抛 ``ValueError``。

    返回 ``{column, task, labels}``：``task`` 缺省时按标签取值个数推断
    （2 ⇒ binary，其余 ⇒ multiclass）；``labels`` 统一成 ``{str: str}``。
    """
    if not target:
        return None
    column = target.get("column")
    if not isinstance(column, str) or not column.strip():
        raise ValueError("config.target.column 必须是非空字符串（目标字段列名）")
    task = target.get("task")
    if task is not None and task not in ("binary", "multiclass"):
        raise ValueError(f"config.target.task 只能是 binary / multiclass，收到 {task!r}")
    labels = target.get("labels")
    if labels is not None:
        if not isinstance(labels, dict):
            raise ValueError("config.target.labels 必须是对象（如 {\"1\": \"bad\", \"0\": \"good\"}）")
        labels = {str(k): str(v) for k, v in labels.items()}
    return {"column": column, "task": task, "labels": labels}


def resolve_task(series: pd.Series, task: Optional[str]) -> str:
    """按标签取值个数补齐任务类型（2 ⇒ binary，其余 ⇒ multiclass）。"""
    if task in ("binary", "multiclass"):
        return task
    nunique = int(series.nunique(dropna=True))
    return "binary" if nunique <= 2 else "multiclass"


def binary_bad_label(series: pd.Series, labels: Optional[Dict[str, str]]) -> str:
    """二分类里「坏」标签的字符串形式。

    优先用 ``labels`` 中值为 ``bad`` 的那个键；否则取较大取值（约定 1=坏）。
    """
    values = sorted({str(v) for v in series.dropna().unique()})
    if labels:
        for key, name in labels.items():
            if str(name).strip().lower() in ("bad", "1", "坏", "违约", "正例", "positive"):
                return str(key)
    return values[-1] if values else "1"


def monthly_key(series: pd.Series) -> pd.Series:
    """把时间列规范成 ``YYYY-MM`` 字符串（无法识别时返回空 Series）。"""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.strftime("%Y-%m")
    text = series.dropna().astype(str).str.strip()
    if len(text) == 0:
        return pd.Series([], dtype="object")
    for pattern in _MONTHLY_PATTERNS:
        match = text.str.match(pattern)
        if bool(match.all()):
            groups = text.str.extract(pattern)
            return groups[0].str.zfill(4) + "-" + groups[1].str.zfill(2)
    return pd.Series([], dtype="object")
