"""预处理规格执行（契约 §3.3 ``preprocess_apply``）。

规格结构（完整说明与示例见 ``docs/preprocess-split.md``）
--------------------------------------------------------
::

    spec = {
      "missing": [{"column": "amount", "strategy": "median"}],          # 缺失填充
      "encode":  [{"column": "channel", "strategy": "onehot"}],          # 类别编码
      "cap":     [{"column": "score", "strategy": "quantile",
                   "lower_quantile": 0.01, "upper_quantile": 0.99}],     # 盖帽
      "binning": {"amount": {"edges": [-inf, 0, 100, 500, inf]}},        # 手动分箱
    }

四段都可以写成「列表形态」或「``{列: 策略/参数}`` 映射形态」，由 :func:`normalize_spec`
统一收口（只在这一处做形态转换，避免两套解析各自漂移）。**未知列 / 未知策略 /
未知键 / 非法区间一律 ``spec_invalid``**，不静默跳过。

★ 执行顺序固定：缺失填充 → 编码 → 盖帽 → 手动分箱
    ① 填充先做：编码与盖帽都需要拿到确定的取值域；
    ② 盖帽在编码之后：只对数值列生效，编码后的列已是数值；
    ③ 分箱最后：切点作用在「最终数值」上（先盖帽再分箱，箱内极值才不会被切碎）。
    这个顺序写进 ``preprocess/summary.json`` 的 ``order`` 字段，可复核。

★ 手动分箱会**替换**原列（值为箱编码 0..k-1，缺失为 -1）
    入模矩阵必须是纯数值且无 NaN（LR 的硬要求）。若同时保留原始列，
    「原始列仍有缺失」会让「入模矩阵是否可用」永远无法判定。
    原始取值在 CSV 里永远可回溯，分箱映射完整记录在 ``manual_bins.json``。

★ WOE 表只在训练期拟合（标签泄漏红线）
    ``split/split.json`` 存在 ⇒ 只用 train 行拟合 WOE 表；
    不存在 ⇒ 用全量并**显式告警**（说明该表尚不可用于最终交付，需划分后重跑）。
    ordinal / onehot 与标签无关，不受此约束（映射按取值字典序派生，天然可复现）。

★ 失败必须可见
    任何降级（parquet 不可用落 CSV、缺失未成箱、编码后仍有缺失、入模矩阵仍不可用、
    划分存在时用 drop 会破坏 mask 对齐）都写进 ``warnings`` 与 ``summary.json``。
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import __version__
from ..artifacts.layout import resolve_project
from ..artifacts.manifest import now_iso, read_manifest, stage_done, update_manifest
from ..core.binning import MISSING_BIN, BinSpec, transform_bins
from ..core.encode import (
    apply_onehot_encode,
    apply_woe_encode,
    derive_label_mapping,
    derive_onehot_values,
    fit_woe_table,
)
from ..core.errors import ConfigError, DataError, RiskLabError
from ..core.split import SplitSpec, split_frame
from ..core.transforms import (
    SENTINEL_FILL_VALUE,
    EncodeSpec,
    FillSpec,
    apply_encode,
    apply_fill,
)
from ..data.loader import (
    binary_bad_label,
    is_numeric_series,
    normalize_target,
    read_csv_auto,
    resolve_task,
    to_jsonable,
    write_csv_frame,
)
from ..platform.fs import atomic_write_bytes, atomic_write_json
from ..project import resolve_csv_path, roles_for_analysis
from ..protocol import (
    relative_paths,
    result_error,
    result_from_exception,
    result_needs_input,
    result_ok,
    target_question,
)
from ..core.utils import bin_interval_label

__all__ = [
    "APPLY_ORDER",
    "CAP_STRATEGIES",
    "ENCODE_STRATEGIES",
    "MISSING_STRATEGIES",
    "SPEC_SECTIONS",
    "SpecError",
    "apply_plan",
    "normalize_spec",
    "preprocess_apply",
    "validate_spec",
]

#: ``preprocess/summary.json`` 的 schema 标识。
SUMMARY_SCHEMA = "risklab-mdl.preprocess-summary/1"

#: 固定的执行顺序（落盘可复核）。
APPLY_ORDER: Tuple[str, ...] = ("missing", "encode", "cap", "binning")

#: ``spec`` 允许的顶层键。
SPEC_SECTIONS: Tuple[str, ...] = APPLY_ORDER

MISSING_STRATEGIES: Tuple[str, ...] = ("mean", "median", "mode", "sentinel", "as_bin", "drop")
ENCODE_STRATEGIES: Tuple[str, ...] = ("onehot", "ordinal", "woe", "drop")
CAP_STRATEGIES: Tuple[str, ...] = ("absolute", "quantile")

#: 缺失独立箱的标签（与 ``docs/preprocess-split.md`` 约定一致）。
MISSING_BIN_LABEL = "__MISSING__"

_ALLOWED_KEYS: Dict[str, Tuple[str, ...]] = {
    "missing": ("column", "strategy", "value", "note"),
    "encode": ("column", "strategy", "note"),
    "cap": ("column", "strategy", "min", "max", "lower_quantile", "upper_quantile", "note"),
    "binning": ("column", "edges", "missing_as_bin", "note"),
}


class SpecError(RiskLabError):
    """``spec`` 结构/取值非法 —— 工具层映射成 ``error.code=spec_invalid``。"""

    code = "SPEC_INVALID"

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


# --------------------------------------------------------------------------- #
# 规格归一化
# --------------------------------------------------------------------------- #
def _entry(column: Any, value: Any, section: str) -> Dict[str, Any]:
    col = str(column)
    if section == "binning" and isinstance(value, (list, tuple)):
        return {"column": col, "edges": list(value)}
    if isinstance(value, Mapping):
        payload: Dict[str, Any] = {"column": col}
        payload.update({str(k): v for k, v in value.items()})
        return payload
    if value is None:
        raise SpecError(
            f"spec.{section}[{col!r}] 没有任何参数",
            hint=f"{section} 需要 {{\"column\": ..., \"strategy\": ...}} 或 {{列: 策略}} 形态",
        )
    return {"column": col, "strategy": value}


def _section_entries(raw: Any, section: str) -> List[Dict[str, Any]]:
    """把一段规格统一成 ``[{column, ...}]``（列表 / 映射两种形态）。"""
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        return [_entry(col, value, section) for col, value in raw.items()]
    if isinstance(raw, (list, tuple)):
        out: List[Dict[str, Any]] = []
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise SpecError(
                    f"spec.{section}[{index}] 必须是对象，收到 {type(item).__name__}",
                    hint=f"形如 {{\"column\": \"amount\", \"strategy\": \"median\"}}",
                )
            payload = {str(k): v for k, v in item.items()}
            column = payload.pop("column", None)
            if column is None or not str(column).strip():
                raise SpecError(f"spec.{section}[{index}] 缺少 column")
            payload["column"] = str(column)
            out.append(payload)
        return out
    raise SpecError(
        f"spec.{section} 必须是列表或 {{列: 参数}} 映射，收到 {type(raw).__name__}"
    )


def normalize_spec(spec: Optional[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """把任意合法形态的 ``spec`` 归一成四段有序列表。"""
    if not isinstance(spec, Mapping):
        raise SpecError(f"spec 必须是对象，收到 {type(spec).__name__}")
    unknown = sorted(set(spec) - set(SPEC_SECTIONS), key=str)
    if unknown:
        raise SpecError(
            f"spec 含未知键：{[str(u) for u in unknown]}",
            hint=f"只支持 {', '.join(SPEC_SECTIONS)}；字段名见 docs/preprocess-split.md",
        )
    return {section: _section_entries(spec.get(section), section) for section in SPEC_SECTIONS}


def _check_keys(entry: Mapping[str, Any], section: str) -> None:
    unknown = [str(k) for k in entry if k not in _ALLOWED_KEYS[section]]
    if unknown:
        raise SpecError(
            f"spec.{section}[{entry.get('column')!r}] 含未知字段：{unknown}",
            hint=f"{section} 允许的字段：{list(_ALLOWED_KEYS[section])}",
        )


def _as_float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SpecError(f"{label} 必须是数字，收到 {value!r}") from exc


def _require_column(
    frame: pd.DataFrame,
    entry: Mapping[str, Any],
    section: str,
    *,
    target_column: Optional[str],
    forbidden: Sequence[str] = (),
) -> str:
    column = str(entry.get("column", ""))
    if not column:
        raise SpecError(f"spec.{section} 中存在空的 column")
    if column not in set(map(str, frame.columns)):
        raise SpecError(
            f"spec.{section} 指定的列不存在：{column!r}",
            hint=f"数据实际列名（前 20 个）：{list(map(str, frame.columns))[:20]}",
        )
    if target_column and column == str(target_column):
        raise SpecError(
            f"spec.{section} 不能作用于目标列 {column!r}",
            hint="目标列不参与预处理；它由模型训练阶段单独读取",
        )
    if column in forbidden:
        raise SpecError(
            f"spec.{section} 作用于角色列 {column!r}（id/time/split）",
            hint="角色列既不入模也不应被改写；需要剔除请在特征清单里排除",
        )
    return column


def validate_spec(
    spec: Optional[Mapping[str, Any]],
    frame: pd.DataFrame,
    *,
    target: Optional[Mapping[str, Any]] = None,
    roles: Optional[Mapping[str, Any]] = None,
    has_split: bool = False,
) -> Dict[str, Any]:
    """校验并归一化 ``spec``，返回可执行计划（含逐条已校验的参数）。

    Raises
    ------
    SpecError
        未知列 / 未知策略 / 未知字段 / 非法区间 / 策略与列类型不匹配。
    """
    plan = normalize_spec(spec)
    if not any(plan[section] for section in SPEC_SECTIONS):
        raise SpecError(
            "spec 为空：missing / encode / cap / binning 至少要有一项",
            hint='示例：{"missing": {"amount": "median"}, "encode": {"channel": "onehot"}}',
        )

    target_column = (target or {}).get("column")
    role_columns = [
        str((roles or {}).get(key)) for key in ("id", "time", "split") if (roles or {}).get(key)
    ]
    checks: Dict[str, Any] = {
        "missing": [],
        "encode": [],
        "cap": [],
        "binning": [],
        "warnings": [],
    }
    seen: Dict[str, set[str]] = {section: set() for section in SPEC_SECTIONS}

    # ---------------- ① 缺失填充 ----------------
    for entry in plan["missing"]:
        _check_keys(entry, "missing")
        column = _require_column(
            frame, entry, "missing", target_column=target_column, forbidden=role_columns
        )
        if column in seen["missing"]:
            raise SpecError(f"spec.missing 中列 {column!r} 重复出现")
        seen["missing"].add(column)
        strategy = str(entry.get("strategy", "")).strip().lower()
        if strategy not in MISSING_STRATEGIES:
            raise SpecError(
                f"未知的缺失策略：{strategy!r}（列 {column}）",
                hint=f"可用：{', '.join(MISSING_STRATEGIES)}",
            )
        numeric = is_numeric_series(frame[column])
        if strategy in ("mean", "median") and not numeric:
            raise SpecError(
                f"列 {column!r} 是非数值列，不能用 {strategy} 填充",
                hint="用 mode / sentinel / as_bin / drop；或先用 encode 编码",
            )
        if strategy == "sentinel":
            if not numeric:
                raise SpecError(
                    f"列 {column!r} 是非数值列，不能用 sentinel 填充",
                    hint="用 mode / as_bin / drop",
                )
            value = entry.get("value", SENTINEL_FILL_VALUE)
            value = _as_float(value, f"spec.missing[{column}].value")
            checks["missing"].append(
                {"column": column, "strategy": "sentinel", "value": value}
            )
            continue
        if strategy == "as_bin":
            # 缺失保留为 NaN，是否真的落成独立箱取决于有没有配套的 binning 规格，
            # 因此这条告警在看完分箱段之后再决定（见下方「跨段冲突」）。
            pass
        checks["missing"].append({"column": column, "strategy": strategy})

    # ---------------- ② 编码 ----------------
    for entry in plan["encode"]:
        _check_keys(entry, "encode")
        column = _require_column(
            frame, entry, "encode", target_column=target_column, forbidden=role_columns
        )
        if column in seen["encode"]:
            raise SpecError(f"spec.encode 中列 {column!r} 重复出现")
        seen["encode"].add(column)
        strategy = str(entry.get("strategy", "")).strip().lower()
        if strategy not in ENCODE_STRATEGIES:
            raise SpecError(
                f"未知的编码策略：{strategy!r}（列 {column}）",
                hint=f"可用：{', '.join(ENCODE_STRATEGIES)}；woe 仅支持二分类",
            )
        if strategy == "woe":
            if target_column is None:
                raise SpecError(
                    f"列 {column!r} 使用 woe 编码，但项目没有确认目标字段",
                    hint="先确认 config.target（woe 表要用标签计算坏样本率）",
                )
            task = resolve_task(frame[target_column], (target or {}).get("task"))
            if task != "binary":
                raise SpecError(
                    f"woe 编码仅支持二分类目标，当前任务 {task!r}",
                    hint="多分类请用 ordinal / onehot；WOE 的坏率在多分类下没有唯一含义",
                )
        if is_numeric_series(frame[column]) and strategy in ("onehot", "ordinal", "woe"):
            checks["warnings"].append(
                f"列 {column!r} 是数值列却要求 {strategy} 编码：会按字符取值映射，"
                f"丢失数值顺序（确认这是有意为之）"
            )
        checks["encode"].append({"column": column, "strategy": strategy})

    # ---------------- ③ 盖帽 ----------------
    for entry in plan["cap"]:
        _check_keys(entry, "cap")
        column = _require_column(
            frame, entry, "cap", target_column=target_column, forbidden=role_columns
        )
        if column in seen["cap"]:
            raise SpecError(f"spec.cap 中列 {column!r} 重复出现")
        seen["cap"].add(column)
        if not is_numeric_series(frame[column]):
            raise SpecError(
                f"列 {column!r} 不是数值列，无法盖帽",
                hint="先 encode 成数值，或从 spec.cap 中移除该列",
            )
        strategy = str(entry.get("strategy") or "").strip().lower()
        has_abs = "min" in entry or "max" in entry
        has_quant = "lower_quantile" in entry or "upper_quantile" in entry
        if not strategy:
            strategy = "absolute" if has_abs else ("quantile" if has_quant else "")
        if strategy not in CAP_STRATEGIES:
            raise SpecError(
                f"未知的盖帽策略：{strategy!r}（列 {column}）",
                hint=f"可用：{', '.join(CAP_STRATEGIES)}；absolute 给 min/max，quantile 给分位",
            )
        if strategy == "absolute":
            if not has_abs:
                raise SpecError(
                    f"spec.cap[{column}] strategy=absolute 但没给 min/max",
                    hint='形如 {"column": "score", "strategy": "absolute", "max": 100}',
                )
            lo = _as_float(entry["min"], f"spec.cap[{column}].min") if "min" in entry else None
            hi = _as_float(entry["max"], f"spec.cap[{column}].max") if "max" in entry else None
            if lo is not None and hi is not None and lo >= hi:
                raise SpecError(f"spec.cap[{column}] 的 min={lo} 必须小于 max={hi}")
            checks["cap"].append(
                {"column": column, "strategy": "absolute", "min": lo, "max": hi}
            )
            continue
        if not has_quant:
            raise SpecError(
                f"spec.cap[{column}] strategy=quantile 但没给 lower_quantile/upper_quantile",
                hint='形如 {"column": "score", "strategy": "quantile", "upper_quantile": 0.99}',
            )
        lo_q = (
            _as_float(entry["lower_quantile"], f"spec.cap[{column}].lower_quantile")
            if "lower_quantile" in entry
            else None
        )
        hi_q = (
            _as_float(entry["upper_quantile"], f"spec.cap[{column}].upper_quantile")
            if "upper_quantile" in entry
            else None
        )
        for label, value in (("lower_quantile", lo_q), ("upper_quantile", hi_q)):
            if value is not None and not (0.0 <= value <= 1.0):
                raise SpecError(f"spec.cap[{column}].{label}={value} 必须落在 [0, 1]")
        if lo_q is not None and hi_q is not None and lo_q >= hi_q:
            raise SpecError(f"spec.cap[{column}] 的 lower_quantile 必须小于 upper_quantile")
        checks["cap"].append(
            {
                "column": column,
                "strategy": "quantile",
                "lower_quantile": lo_q,
                "upper_quantile": hi_q,
            }
        )

    # ---------------- ④ 手动分箱 ----------------
    for entry in plan["binning"]:
        _check_keys(entry, "binning")
        column = _require_column(
            frame, entry, "binning", target_column=target_column, forbidden=role_columns
        )
        if column in seen["binning"]:
            raise SpecError(f"spec.binning 中列 {column!r} 重复出现")
        seen["binning"].add(column)
        if not is_numeric_series(frame[column]):
            raise SpecError(
                f"列 {column!r} 不是数值列，无法按切点分箱",
                hint="先 encode 成数值再分箱",
            )
        raw_edges = entry.get("edges")
        if not isinstance(raw_edges, (list, tuple)) or len(raw_edges) == 0:
            raise SpecError(
                f"spec.binning[{column}].edges 必须是非空切点列表",
                hint="形如 [-inf, 0, 100, 500, inf]；首尾 -inf/inf 可写可不写",
            )
        values = [_as_float(v, f"spec.binning[{column}].edges") for v in raw_edges]
        interior = list(values)
        if interior and np.isneginf(interior[0]):
            interior = interior[1:]
        if interior and np.isposinf(interior[-1]):
            interior = interior[:-1]
        if not interior:
            raise SpecError(
                f"spec.binning[{column}] 至少需要一个有限内部切点（否则只有单箱，没有意义）"
            )
        if any(not np.isfinite(v) for v in interior):
            raise SpecError(
                f"spec.binning[{column}] 的切点里 -inf/inf 只能出现在首尾，收到 {values}"
            )
        for prev, cur in zip(interior, interior[1:]):
            if not cur > prev:
                raise SpecError(
                    f"spec.binning[{column}] 的切点必须严格递增，"
                    f"发现 {prev} → {cur}（收到 {values}）"
                )
        missing_as_bin = bool(entry.get("missing_as_bin", True))
        n_missing = int(frame[column].isna().sum())
        if missing_as_bin and n_missing == 0:
            checks["warnings"].append(
                f"列 {column!r} 无缺失，{MISSING_BIN_LABEL} 箱为空（不影响结果）"
            )
        checks["binning"].append(
            {
                "column": column,
                "edges": values,
                "interior_edges": [float(v) for v in interior],
                "missing_as_bin": missing_as_bin,
            }
        )

    # ---------------- 跨段冲突 ----------------
    dropped = {e["column"] for e in checks["encode"] if e["strategy"] == "drop"}
    for section in ("cap", "binning"):
        clash = [e["column"] for e in checks[section] if e["column"] in dropped]
        if clash:
            raise SpecError(
                f"列 {clash} 同时出现在 encode(drop) 与 {section} 中，执行顺序会互相矛盾",
                hint="被剔除的列不需要再盖帽/分箱，请从 spec 中移除其中一段",
            )
    for entry in checks["missing"]:
        if entry["column"] in dropped:
            checks["warnings"].append(
                f"列 {entry['column']!r} 先填充后又被 encode drop：填充不会生效"
            )
    binned_columns = {e["column"] for e in checks["binning"]}
    for entry in checks["missing"]:
        if entry["strategy"] == "as_bin" and entry["column"] not in binned_columns:
            checks["warnings"].append(
                f"列 {entry['column']!r} 使用 as_bin 但没有对应的 spec.binning："
                f"缺失会保留为 NaN，processed 中该列仍有缺失（入模矩阵不可用）"
            )
    if has_split and any(e["strategy"] == "drop" for e in checks["missing"]):
        checks["warnings"].append(
            "spec.missing 使用了 drop（会删行）：项目已存在 split/split.json，"
            "行数变化后该 mask 无法对齐；建议改用填充，或先划分再对各子集分别 apply"
        )
    return checks


# --------------------------------------------------------------------------- #
# 执行
# --------------------------------------------------------------------------- #
@dataclass
class ApplyOutcome:
    """执行结果（处理后数据 + 四段明细）。"""

    frame: pd.DataFrame
    fill: List[Dict[str, Any]] = field(default_factory=list)
    encode: List[Dict[str, Any]] = field(default_factory=list)
    cap: List[Dict[str, Any]] = field(default_factory=list)
    binning: List[Dict[str, Any]] = field(default_factory=list)
    rows_in: int = 0
    rows_out: int = 0
    columns_in: int = 0
    warnings: List[str] = field(default_factory=list)


def _json_edges(values: Sequence[float]) -> List[Any]:
    """切点转 JSON：``±inf`` 保持 ``"-inf"``/``"inf"`` 字面量（不被 to_jsonable 抹成 null）。"""
    out: List[Any] = []
    for value in values:
        number = float(value)
        if np.isneginf(number):
            out.append("-inf")
        elif np.isposinf(number):
            out.append("inf")
        else:
            out.append(number)
    return out


def _split_spec_from_project(project_dir) -> Optional[Dict[str, Any]]:
    """读 ``split/split.json`` 的 spec（不存在 ⇒ None）。"""
    from ..platform.fs import read_json

    path = project_dir / "split" / "split.json"
    if not path.is_file():
        return None
    try:
        doc = read_json(path)
    except Exception:  # noqa: BLE001 - 划分文件坏了不阻断预处理，但要告警
        return {"__error__": True}
    spec = doc.get("spec") if isinstance(doc, dict) else None
    return spec if isinstance(spec, Mapping) else None


def apply_plan(
    frame: pd.DataFrame,
    plan: Mapping[str, Any],
    *,
    target: Optional[Mapping[str, Any]] = None,
    train_mask: Optional[pd.Series] = None,
    split_present: bool = False,
) -> ApplyOutcome:
    """按已校验的计划执行四段预处理（纯计算，不落盘）。"""
    out = frame.copy()
    outcome = ApplyOutcome(
        frame=out,
        rows_in=int(out.shape[0]),
        columns_in=int(out.shape[1]),
    )
    warnings: List[str] = list(plan.get("warnings") or [])
    target_column = (target or {}).get("column")

    # ---------------- ① 缺失填充 ----------------
    for entry in plan.get("missing") or []:
        column = entry["column"]
        strategy = entry["strategy"]
        before = int(out[column].isna().sum())
        rows_before = int(out.shape[0])
        fill_spec = FillSpec(
            column=column,
            strategy=strategy,
            value=float(entry.get("value", SENTINEL_FILL_VALUE)),
        )
        try:
            out = apply_fill(out, {column: fill_spec})
        except ConfigError as exc:
            raise SpecError(
                f"列 {column!r} 填充失败（{strategy}）：{exc.message}",
                hint="换填充策略（mean/median/mode/as_bin）或检查该列取值域",
            ) from exc
        after = int(out[column].isna().sum()) if column in out.columns else 0
        outcome.fill.append(
            {
                "column": column,
                "strategy": strategy,
                "value": float(entry["value"]) if strategy == "sentinel" else None,
                "missing_before": before,
                "missing_after": after,
                "rows_removed": rows_before - int(out.shape[0]),
            }
        )
        if strategy == "drop":
            warnings.append(
                f"列 {column!r} 使用 drop：删除了 {rows_before - int(out.shape[0])} 行"
            )
        elif strategy != "as_bin" and after > 0:
            warnings.append(
                f"列 {column!r} 填充后仍有 {after} 个缺失（策略 {strategy}）"
            )
    outcome.frame = out

    # ---------------- ② 编码 ----------------
    for entry in plan.get("encode") or []:
        column = entry["column"]
        strategy = entry["strategy"]
        if column not in out.columns:
            raise SpecError(f"列 {column!r} 已在前面的步骤中被删除，无法编码")
        if strategy == "drop":
            out = out.drop(columns=[column])
            outcome.encode.append({"column": column, "strategy": "drop", "output_columns": []})
            continue
        if strategy == "ordinal":
            mapping = derive_label_mapping(out[column])
            out = apply_encode(
                out,
                {column: EncodeSpec(column=column, strategy="ordinal", mapping=mapping)},
            )
            residual = int(out[column].isna().sum())
            if residual:
                warnings.append(
                    f"列 {column!r} 用 ordinal 编码后仍有 {residual} 个缺失（编码保留 NaN）："
                    f"需在 spec.missing 中先填充"
                )
            outcome.encode.append(
                {
                    "column": column,
                    "strategy": "ordinal",
                    "output_columns": [column],
                    "mapping": {str(k): float(v) for k, v in mapping.items()},
                    "missing_preserved": residual,
                }
            )
            continue
        if strategy == "onehot":
            values = derive_onehot_values(out[column])
            if len(values) > 20:
                warnings.append(
                    f"列 {column!r} 用 onehot 编码产生 {len(values)} 列（高基数 ⇒ 稀疏共线），"
                    f"建议改用 ordinal/woe"
                )
            out = apply_onehot_encode(out, [column], values={column: values})
            outcome.encode.append(
                {
                    "column": column,
                    "strategy": "onehot",
                    "output_columns": [f"{column}_{value}" for value in values],
                    "onehot_values": [str(value) for value in values],
                }
            )
            continue
        if strategy == "woe":
            if target is None or target_column is None:
                raise SpecError(f"列 {column!r} 需要 woe 编码，但没有确认目标字段")
            if split_present and train_mask is not None:
                fit_scope = "train（split/split.json）"
            else:
                fit_scope = "全量（未划分）"
                warnings.append(
                    f"列 {column!r} 的 WOE 表在全量数据上拟合（未找到 split/split.json）："
                    f"坏样本率用到了全部样本，存在标签泄漏风险；划分后需重跑 preprocess_apply"
                )
            y_all = pd.Series(
                (
                    out[target_column].astype(str)
                    == binary_bad_label(out[target_column].dropna(), (target or {}).get("labels"))
                ).astype(int),
                index=out.index,
            )
            keep = out[target_column].notna()
            if split_present and train_mask is not None:
                keep = keep & train_mask.reindex(out.index).fillna(False).astype(bool)
            try:
                table = fit_woe_table(
                    out.loc[keep, column], y_all.loc[keep], column=column
                )
            except DataError as exc:
                raise SpecError(
                    f"列 {column!r} 的 WOE 表拟合失败：{exc.message}",
                    hint="检查训练集里是否同时存在好/坏样本，以及该列是否有足够取值",
                ) from exc
            out = apply_woe_encode(out, [column], {column: table})
            outcome.encode.append(
                {
                    "column": column,
                    "strategy": "woe",
                    "output_columns": [column],
                    "fit_scope": fit_scope,
                    "woe_table": table.to_dict(),
                }
            )
            continue
        raise SpecError(f"未知的编码策略：{strategy!r}")  # pragma: no cover - 已校验

    # ---------------- ③ 盖帽 ----------------
    for entry in plan.get("cap") or []:
        column = entry["column"]
        if column not in out.columns:
            raise SpecError(f"列 {column!r} 已在前面的步骤中被删除，无法盖帽")
        values = pd.to_numeric(out[column], errors="coerce")
        valid = values.notna()
        lower: Optional[float] = None
        upper: Optional[float] = None
        if entry["strategy"] == "absolute":
            lower = entry.get("min")
            upper = entry.get("max")
            fit_scope = "绝对值"
        else:
            if entry.get("lower_quantile") is not None and bool(valid.any()):
                lower = float(values[valid].quantile(float(entry["lower_quantile"])))
            if entry.get("upper_quantile") is not None and bool(valid.any()):
                upper = float(values[valid].quantile(float(entry["upper_quantile"])))
            fit_scope = "分位数（按传入数据拟合）"
        n_capped = 0
        if bool(valid.any()):
            mask = pd.Series(False, index=out.index)
            if lower is not None:
                mask = mask | (values < lower)
            if upper is not None:
                mask = mask | (values > upper)
            n_capped = int((mask & valid).sum())
        min_before = float(values[valid].min()) if bool(valid.any()) else None
        max_before = float(values[valid].max()) if bool(valid.any()) else None
        capped = values.clip(lower=lower, upper=upper)
        out[column] = capped
        valid_after = capped.notna()
        outcome.cap.append(
            {
                "column": column,
                "strategy": entry["strategy"],
                "lower": lower,
                "upper": upper,
                "fit_scope": fit_scope,
                "n_capped": n_capped,
                "min_before": min_before,
                "max_before": max_before,
                "min_after": float(capped[valid_after].min()) if bool(valid_after.any()) else None,
                "max_after": float(capped[valid_after].max()) if bool(valid_after.any()) else None,
            }
        )
    outcome.frame = out

    # ---------------- ④ 手动分箱 ----------------
    for entry in plan.get("binning") or []:
        column = entry["column"]
        if column not in out.columns:
            raise SpecError(f"列 {column!r} 已在前面的步骤中被删除，无法分箱")
        interior = [float(v) for v in entry["interior_edges"]]
        bin_spec = BinSpec(
            method="business_cut",
            n_bins=len(interior) + 1,
            edges=tuple(interior),
            missing_as_bin=bool(entry.get("missing_as_bin", True)),
        )
        codes = transform_bins(out[column], bin_spec)
        n_bins = len(interior) + 1
        counts = codes.value_counts(dropna=False)
        bin_counts: Dict[str, int] = {}
        for code in range(n_bins):
            label = bin_interval_label(code, interior, missing_label=MISSING_BIN_LABEL)
            bin_counts[label] = int(counts.get(code, 0))
        if bin_spec.missing_as_bin:
            missing_count = int(counts.get(MISSING_BIN, 0))
            bin_counts[MISSING_BIN_LABEL] = missing_count
        else:
            missing_count = int(codes.isna().sum())
            warnings.append(
                f"列 {column!r} 声明 missing_as_bin=False：缺失不会成箱，"
                f"processed 中该列仍有 {missing_count} 个 NaN（入模矩阵不可用）"
            )
        out[column] = codes
        outcome.binning.append(
            {
                "column": column,
                "edges": _json_edges(entry["edges"]),
                "interior_edges": interior,
                "n_bins": n_bins,
                "missing_as_bin": bool(bin_spec.missing_as_bin),
                "missing_bin_label": MISSING_BIN_LABEL,
                "missing_count": missing_count,
                "bin_counts": bin_counts,
                "bin_labels": [
                    bin_interval_label(
                        code, interior, missing_label=MISSING_BIN_LABEL
                    )
                    for code in range(n_bins)
                ]
                + ([MISSING_BIN_LABEL] if bin_spec.missing_as_bin else []),
                "values_are": "bin_code",
                "row_count": int(out.shape[0]),
            }
        )
    outcome.frame = out
    outcome.rows_out = int(out.shape[0])
    outcome.warnings = warnings
    return outcome


def _model_readiness(
    frame: pd.DataFrame,
    *,
    target_column: Optional[str],
    roles: Mapping[str, Any],
) -> Dict[str, Any]:
    """入模矩阵可用性：是否还有非数值列 / 缺失（LR 的硬门槛）。"""
    excluded = set()
    if target_column:
        excluded.add(str(target_column))
    for key in ("id", "time", "split"):
        value = roles.get(key)
        if value:
            excluded.add(str(value))
    model_columns = [str(c) for c in frame.columns if str(c) not in excluded]
    non_numeric = [c for c in model_columns if not is_numeric_series(frame[c])]
    remaining = {
        c: int(frame[c].isna().sum()) for c in model_columns if int(frame[c].isna().sum()) > 0
    }
    return {
        "model_columns": model_columns,
        "model_column_count": len(model_columns),
        "non_numeric_columns": non_numeric,
        "remaining_missing": remaining,
        "lr_ready": not non_numeric and not remaining,
    }


# --------------------------------------------------------------------------- #
# 工具入口
# --------------------------------------------------------------------------- #
def _write_parquet(frame: pd.DataFrame, path) -> Optional[str]:
    """尝试写 parquet；成功返回 None，失败返回降级原因。"""
    import importlib.util

    if importlib.util.find_spec("pyarrow") is None:
        return "pyarrow 未安装：processed 数据以 CSV 落盘（可用 pip install pyarrow 换成 parquet）"
    try:
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        atomic_write_bytes(path, buffer.getvalue())
        return None
    except Exception as exc:  # noqa: BLE001 - parquet 失败一律降级 CSV，不静默
        return f"parquet 写入失败（{type(exc).__name__}: {exc}）：processed 数据以 CSV 落盘"


def preprocess_apply(
    project_dir: str, spec: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """落实缺失/编码/盖帽/手动分箱规格，产出四份规格产物 + 处理后的数据。"""
    try:
        layout = resolve_project(project_dir)
        manifest = read_manifest(layout.project_dir)
        csv = resolve_csv_path(layout, manifest)

        target = normalize_target((manifest.get("project") or {}).get("target"))
        frame_head = read_csv_auto(csv, nrows=200).frame
        all_columns = [str(c) for c in frame_head.columns]
        if target is None or target["column"] not in all_columns:
            reason = (
                "尚未确认目标字段。"
                if target is None
                else f"指定的目标列 {target['column']!r} 不在数据中。"
            )
            from ..data.loader import infer_roles

            inferred, _ = infer_roles(frame_head)
            question, context = target_question(
                frame_head, inferred["target_candidates"], reason=reason
            )
            return result_needs_input(question, context)

        read = read_csv_auto(csv)
        frame = read.frame
        warnings: List[str] = list(read.warnings)
        roles = roles_for_analysis(manifest, frame)

        split_spec = _split_spec_from_project(layout.project_dir)
        split_present = split_spec is not None
        train_mask: Optional[pd.Series] = None
        if split_spec:
            if split_spec.get("__error__"):
                warnings.append("split/split.json 无法解析：WOE 拟合范围退回全量（划分口径不可用）")
            else:
                try:
                    split_result = split_frame(
                        frame, SplitSpec.from_config(split_spec), target=target["column"]
                    )
                    n_train = int(split_result.counts.get("train", 0))
                    if n_train:
                        train_mask = split_result.train_mask.reindex(frame.index).fillna(False)
                        warnings.append(
                            f"WOE 拟合使用 train 子集（{n_train} 行，"
                            f"split 指纹 {split_result.fingerprint}）"
                        )
                except Exception as exc:  # noqa: BLE001 - 划分复算失败不阻断，但显式告警
                    warnings.append(
                        f"split/split.json 无法复算（{type(exc).__name__}: {exc}）："
                        f"WOE 拟合范围退回全量"
                    )

        try:
            plan = validate_spec(
                spec,
                frame,
                target=target,
                roles=roles,
                has_split=split_present,
            )
        except SpecError as exc:
            return result_error(
                "spec_invalid", exc.message, getattr(exc, "hint", "") or
                "按 docs/preprocess-split.md 的规格格式修正 spec 后重试"
            )

        outcome = apply_plan(
            frame,
            plan,
            target=target,
            train_mask=train_mask,
            split_present=split_present,
        )
        processed = outcome.frame
        warnings.extend(outcome.warnings)

        # ---------- 落盘：四份规格产物 ----------
        preprocess_dir = layout.subdir("preprocess")
        written = []
        fill_path = preprocess_dir / "fill.json"
        encode_path = preprocess_dir / "encode.json"
        cap_path = preprocess_dir / "outlier.json"
        bins_path = preprocess_dir / "manual_bins.json"
        summary_path = preprocess_dir / "summary.json"

        atomic_write_json(
            fill_path,
            to_jsonable(
                {
                    "schema": "risklab-mdl.preprocess-fill/1",
                    "version": __version__,
                    "generated_at": now_iso(),
                    "target": target,
                    "order": list(APPLY_ORDER),
                    "entries": outcome.fill,
                }
            ),
        )
        atomic_write_json(
            encode_path,
            to_jsonable(
                {
                    "schema": "risklab-mdl.preprocess-encode/1",
                    "version": __version__,
                    "generated_at": now_iso(),
                    "entries": outcome.encode,
                    "notes": [
                        "WOE 表只在训练期拟合（有 split/split.json 时用 train 行）",
                        "ordinal 按取值字典序映射，可复现（不依赖数据行序）",
                        "onehot 缺失单独成列（__MISSING__）",
                    ],
                }
            ),
        )
        atomic_write_json(
            cap_path,
            to_jsonable(
                {
                    "schema": "risklab-mdl.preprocess-outlier/1",
                    "version": __version__,
                    "generated_at": now_iso(),
                    "entries": outcome.cap,
                }
            ),
        )
        atomic_write_json(
            bins_path,
            to_jsonable(
                {
                    "schema": "risklab-mdl.preprocess-bins/1",
                    "version": __version__,
                    "generated_at": now_iso(),
                    "entries": outcome.binning,
                    "note": "分箱后原列被替换为箱编码（0..k-1，缺失 -1）；原始取值可从 CSV 回溯",
                }
            ),
        )
        written.extend([fill_path, encode_path, cap_path, bins_path])

        # ---------- 处理后的数据（模型层直接读取） ----------
        processed_path = preprocess_dir / "processed.parquet"
        fallback = _write_parquet(processed, processed_path)
        processed_format = "parquet"
        if fallback:
            warnings.append(fallback)
            processed_path = preprocess_dir / "processed.csv"
            write_csv_frame(processed, processed_path)
            processed_format = "csv"
        written.append(processed_path)

        # ---------- 处理摘要 ----------
        readiness = _model_readiness(
            processed, target_column=target["column"], roles=roles
        )
        if not readiness["lr_ready"]:
            if readiness["non_numeric_columns"]:
                warnings.append(
                    f"入模矩阵仍含非数值列：{readiness['non_numeric_columns'][:10]}"
                    f"（共 {len(readiness['non_numeric_columns'])} 列）—— LR 不可用"
                )
            if readiness["remaining_missing"]:
                preview = list(readiness["remaining_missing"].items())[:10]
                warnings.append(f"入模矩阵仍有缺失（列: 行数）：{preview} —— LR 不可用")

        summary_payload = {
            "schema": SUMMARY_SCHEMA,
            "version": __version__,
            "generated_at": now_iso(),
            "order": list(APPLY_ORDER),
            "target": target,
            "rows_in": outcome.rows_in,
            "rows_out": outcome.rows_out,
            "columns_in": outcome.columns_in,
            "columns_out": int(processed.shape[1]),
            "processed_path": layout.rel(processed_path),
            "processed_format": processed_format,
            "missing": outcome.fill,
            "encode": [
                {
                    "column": e["column"],
                    "strategy": e["strategy"],
                    "output_columns": e.get("output_columns", []),
                }
                for e in outcome.encode
            ],
            "cap": outcome.cap,
            "binning": [
                {
                    "column": b["column"],
                    "edges": b["edges"],
                    "n_bins": b["n_bins"],
                    "missing_count": b["missing_count"],
                }
                for b in outcome.binning
            ],
            "model_readiness": {
                "lr_ready": readiness["lr_ready"],
                "non_numeric_columns": readiness["non_numeric_columns"],
                "remaining_missing": readiness["remaining_missing"],
                "model_column_count": readiness["model_column_count"],
                "model_columns": readiness["model_columns"],
            },
            "warnings": warnings,
        }
        atomic_write_json(summary_path, to_jsonable(summary_payload))
        written.append(summary_path)

        summary_text = (
            f"填充 {len(outcome.fill)} 列 / 编码 {len(outcome.encode)} 列 / "
            f"盖帽 {len(outcome.cap)} 列 / 分箱 {len(outcome.binning)} 列；"
            f"{outcome.rows_in}→{outcome.rows_out} 行"
        )

        def _mutate(manifest_doc: Dict[str, Any]) -> None:
            stage_done(
                manifest_doc,
                "preprocess",
                done=True,
                path="preprocess/",
                summary=summary_text,
            )

        update_manifest(layout.project_dir, _mutate)

        data = {
            "project_dir": layout.project_dir.as_posix(),
            "processed_path": layout.rel(processed_path),
            "processed_format": processed_format,
            "rows_in": outcome.rows_in,
            "rows_out": outcome.rows_out,
            "columns_in": outcome.columns_in,
            "columns_out": int(processed.shape[1]),
            "fill": outcome.fill,
            "encode": [
                {
                    "column": e["column"],
                    "strategy": e["strategy"],
                    "output_columns": e.get("output_columns", []),
                }
                for e in outcome.encode
            ],
            "cap": outcome.cap,
            "binning": [
                {
                    "column": b["column"],
                    "edges": b["edges"],
                    "n_bins": b["n_bins"],
                    "missing_count": b["missing_count"],
                    "bin_counts": b["bin_counts"],
                }
                for b in outcome.binning
            ],
            "model_readiness": {
                "lr_ready": readiness["lr_ready"],
                "non_numeric_columns": readiness["non_numeric_columns"],
                "remaining_missing": readiness["remaining_missing"],
                "model_column_count": readiness["model_column_count"],
            },
            "summary_path": layout.rel(summary_path),
            "summary": summary_text,
        }
        return result_ok(
            data,
            artifacts=relative_paths(layout, written),
            warnings=warnings + [f"已写入 {len(written)} 个文件到 preprocess/"],
        )
    except SpecError as exc:  # 兜底：任何遗漏的 SpecError 都映射成 spec_invalid
        return result_error(
            "spec_invalid",
            exc.message,
            getattr(exc, "hint", "") or "按 docs/preprocess-split.md 修正 spec",
        )
    except Exception as exc:  # noqa: BLE001
        return result_from_exception(exc)
