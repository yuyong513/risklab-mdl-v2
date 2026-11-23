"""样本画像（纯计算，不落盘、不画图）。

分工：本模块**只算数据**，落盘与出图由 ``tools/dataset.py`` + ``charts/`` 负责。
好处是画像逻辑能被单测直接验证数值，不必先起 matplotlib。

口径（与 ``docs/口径-metrics.md`` 一致）
----------------------------------------
* 坏率 = 坏样本数 / 有效标签样本数（**标签缺失的行不计入分母**，且单独给出
  ``label_missing`` 计数，避免分母口径不明）；
* 多分类没有"坏率"概念，给出每类占比 + 最大类占比（``majority_share``），
  不把多分类硬压成二分类的坏率（那会伪造出一个不存在的指标）；
* 缺失率、同值率的分母都是总行数 / 非空样本数，逐列注明；
* 月份 × 标签交叉需要时间列：没有时间列时返回 ``None`` 并在 ``warnings``
  说明「已跳过」，**不伪造**时间维度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from ..core.utils import safe_ratio
from .loader import (
    NUMERIC_SEMANTIC_TYPES,
    as_numeric,
    binary_bad_label,
    column_infos,
    is_numeric_series,
    monthly_key,
    resolve_task,
    to_jsonable,
)

__all__ = [
    "DEFAULT_MISSING_TOP",
    "ProfileResult",
    "compute_profile",
    "missing_summary_frame",
]

#: ``data.missing_summary`` 默认只回传缺失率最高的前 N 列（profile.json 里有全量）。
DEFAULT_MISSING_TOP = 20

#: 坏率偏低告警阈值（低于它时 IV / KS 读数不稳定）。
LOW_BAD_RATE = 0.05

#: 样本量偏小告警阈值。
SMALL_SAMPLE = 500

#: 高缺失列的告警阈值。
HIGH_MISSING = 0.5


@dataclass
class ProfileResult:
    """画像结果。

    ``describe_frame`` / ``target_distribution_frame`` / ``monthly_frame``
    是待落盘的表格，其余字段直接进 ``profile.json`` 与工具 ``data`` 载荷。
    """

    target: Dict[str, Any]
    shape: Dict[str, int]
    target_distribution: Dict[str, Any]
    target_distribution_frame: pd.DataFrame
    missing_summary: List[Dict[str, Any]]
    columns: List[Dict[str, Any]]
    describe_frame: pd.DataFrame
    feature_types: Dict[str, Any]
    monthly_frame: Optional[pd.DataFrame]
    monthly: Optional[List[Dict[str, Any]]]
    warnings: List[str] = field(default_factory=list)

    # ---------- 派生 ----------

    @property
    def target_column(self) -> str:
        return str(self.target["column"])

    def missing_top(self, limit: int = DEFAULT_MISSING_TOP) -> List[Dict[str, Any]]:
        return self.missing_summary[:limit]

    def to_payload(self) -> Dict[str, Any]:
        """写进 ``profile.json`` 的结构（全量，不加截断）。"""
        return {
            "shape": self.shape,
            "target": to_jsonable(self.target),
            "target_distribution": to_jsonable(self.target_distribution),
            "feature_types": to_jsonable(self.feature_types),
            "missing_summary": to_jsonable(self.missing_summary),
            "columns": to_jsonable(self.columns),
            "monthly": to_jsonable(self.monthly),
            "warnings": list(self.warnings),
        }

    def data_block(self, *, missing_top: int = DEFAULT_MISSING_TOP) -> Dict[str, Any]:
        """工具返回的 ``data``：契约要求的三个字段 + 少量定位信息。"""
        return {
            "shape": to_jsonable(self.shape),
            "target": to_jsonable(self.target),
            "target_distribution": to_jsonable(self.target_distribution),
            "missing_summary": to_jsonable(self.missing_top(missing_top)),
            "feature_types": to_jsonable(
                {k: v for k, v in self.feature_types.items() if k != "columns"}
            ),
            "has_time_column": self.monthly is not None,
            "chart_count": 0,  # 由工具层在出图后回填
        }


def missing_summary_frame(columns: List[Dict[str, Any]]) -> pd.DataFrame:
    """缺失汇总表（``describe.csv`` 之外单独存一份，便于快速筛列）。"""
    rows = [
        {
            "column": info["name"],
            "role": info["role"],
            "dtype": info["dtype"],
            "nunique": info["nunique"],
            "missing_rate": info["missing_rate"],
            "same_value_rate": info["same_value_rate"],
        }
        for info in columns
    ]
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.sort_values(
            ["missing_rate", "same_value_rate"], ascending=[False, False]
        ).reset_index(drop=True)
    return frame


def _describe(frame: pd.DataFrame, roles: Dict[str, Any], target_column: str) -> pd.DataFrame:
    """数值列描述统计（含分位数）；非数值列不参与（由 ``feature_types`` 说明）。"""
    numeric_columns = [
        str(c)
        for c in frame.columns
        if is_numeric_series(frame[c]) and int(frame[c].dropna().shape[0]) > 0
    ]
    if not numeric_columns:
        return pd.DataFrame(
            columns=["count", "mean", "std", "min", "25%", "50%", "75%", "max", "missing_rate"]
        )
    numeric_frame = pd.DataFrame(
        {c: as_numeric(frame[c])[0] for c in numeric_columns}, index=frame.index
    )
    described = numeric_frame.describe(percentiles=[0.25, 0.5, 0.75]).T
    described = described.astype(float)
    described["missing_rate"] = [
        round(safe_ratio(float(frame[c].isna().sum()), float(len(frame))), 6)
        for c in numeric_columns
    ]
    role_of = {}
    for key in ("id", "time", "split"):
        value = (roles or {}).get(key)
        if value:
            role_of[str(value)] = key
    role_of[str(target_column)] = "target"
    described["role"] = [role_of.get(c, "feature") for c in numeric_columns]
    described.index.name = "column"
    return described


def _feature_types(frame: pd.DataFrame, columns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """特征类型构成（数值 / 类别 / 文本 / 高基数 / 时间）。"""
    buckets: Dict[str, List[str]] = {
        "numeric": [],
        "categorical": [],
        "text": [],
        "datetime": [],
        "high_cardinality": [],
    }
    for info in columns:
        if info["role"] in ("id", "time", "split"):
            continue
        dtype = info["dtype"]
        if dtype in NUMERIC_SEMANTIC_TYPES:
            buckets["numeric"].append(info["name"])
        elif dtype == "categorical":
            buckets["categorical"].append(info["name"])
        elif dtype == "datetime":
            buckets["datetime"].append(info["name"])
        else:
            buckets["text"].append(info["name"])
        if info.get("is_high_cardinality"):
            buckets["high_cardinality"].append(info["name"])
    counts = {key: len(value) for key, value in buckets.items()}
    return {"counts": counts, "columns": buckets}


def _target_distribution(
    frame: pd.DataFrame, target_column: str, task: str, labels: Optional[Dict[str, str]]
) -> Dict[str, Any]:
    """目标分布载荷 + 落盘表。"""
    series = frame[target_column]
    total_rows = int(len(frame))
    missing = int(series.isna().sum())
    valid = series.dropna()
    counts = {str(k): int(v) for k, v in valid.value_counts().sort_index().items()}
    valid_n = int(len(valid))
    shares = {label: round(safe_ratio(count, valid_n), 6) for label, count in counts.items()}
    label_names = dict(labels or {})

    frame_rows = [
        {
            "label": label,
            "label_name": label_names.get(label, label),
            "count": count,
            "share": shares[label],
        }
        for label, count in counts.items()
    ]
    payload: Dict[str, Any] = {
        "column": target_column,
        "task": task,
        "n_classes": len(counts),
        "labels": label_names,
        "counts": counts,
        "shares": shares,
        "valid": valid_n,
        "missing": missing,
        "missing_rate": round(safe_ratio(float(missing), float(total_rows)), 6),
    }
    if task == "binary" and len(counts) >= 2:
        bad_label = binary_bad_label(valid, labels)
        bad_count = counts.get(bad_label, 0)
        payload.update(
            {
                "bad_label": bad_label,
                "bad_count": bad_count,
                "good_count": valid_n - bad_count,
                "bad_rate": round(safe_ratio(float(bad_count), float(valid_n)), 6),
            }
        )
    else:
        majority = max(shares.values()) if shares else 0.0
        payload["majority_share"] = round(float(majority), 6)
        payload["note"] = "多分类：不给二分类坏率（避免伪造指标），只看各类占比"
    return {"payload": payload, "frame": pd.DataFrame(frame_rows)}


def _monthly_cross(
    frame: pd.DataFrame,
    target_column: str,
    task: str,
    labels: Optional[Dict[str, str]],
    time_column: str,
) -> Optional[tuple[pd.DataFrame, List[Dict[str, Any]], List[str]]]:
    """月份 × 标签交叉。时间列不可解析时返回 ``(None, None, warnings)``。"""
    warnings: List[str] = []
    month = monthly_key(frame[time_column])
    if len(month) == 0:
        warnings.append(
            f"时间列 {time_column!r} 的取值无法解析为月份（支持 YYYY-MM / YYYYMM / "
            "YYYY年M月 / YYYY-MM-DD），已跳过月份交叉与趋势图（不伪造）"
        )
        return None, None, warnings
    # monthly_key 在文本分支只返回非空行，这里按原索引对齐（避免长度不一致）
    month_full = month.reindex(frame.index)
    tmp = pd.DataFrame(
        {"month": month_full.values, "y": frame[target_column].values}, index=frame.index
    )
    tmp = tmp[tmp["month"].notna() & tmp["y"].notna()]
    if not len(tmp):
        warnings.append("目标列全为缺失，无法做月份 × 标签交叉")
        return None, None, warnings

    if task == "binary":
        bad_label = binary_bad_label(frame[target_column].dropna(), labels)
        tmp["is_bad"] = (tmp["y"].astype(str) == bad_label).astype(int)
        grouped = (
            tmp.groupby("month", dropna=False)
            .agg(count=("is_bad", "size"), bad_count=("is_bad", "sum"))
            .reset_index()
            .sort_values("month")
        )
        grouped["bad_rate"] = (grouped["bad_count"] / grouped["count"]).astype(float)
        frame_rows = [
            {
                "month": str(row.month),
                "count": int(row.count),
                "bad_count": int(row.bad_count),
                "bad_rate": round(float(row.bad_rate), 6),
            }
            for row in grouped.itertuples()
        ]
        table = pd.DataFrame(frame_rows)
        return table, frame_rows, warnings

    grouped = (
        tmp.groupby(["month", "y"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["month", "y"])
    )
    grouped["share_in_month"] = grouped["count"] / grouped.groupby("month")["count"].transform(
        "sum"
    )
    table = pd.DataFrame(
        {
            "month": grouped["month"].astype(str),
            "label": grouped["y"].astype(str),
            "count": grouped["count"].astype(int),
            "share_in_month": grouped["share_in_month"].astype(float).round(6),
        }
    )
    frame_rows = table.to_dict(orient="records")
    return table, frame_rows, warnings


def compute_profile(
    frame: pd.DataFrame,
    *,
    target: Dict[str, Any],
    roles: Optional[Dict[str, Any]] = None,
    max_features_for_charts: int = 30,
    sample_rows: Optional[int] = None,
) -> ProfileResult:
    """计算样本画像（纯计算）。

    Parameters
    ----------
    frame
        已读入的数据（原始值，不做编码/填充）。
    target
        ``{column, task, labels}``（``normalize_target`` 的输出）。
    roles
        角色推断结果（只用于标注 ``role`` 列，不影响计算）。
    max_features_for_charts
        图表阶段可用特征数上限（画像本身全量算，不截断）。
    sample_rows
        可选抽样行数（宽表性能兜底）。抽样时 ``warnings`` 会显式说明。
    """
    warnings: List[str] = []
    if sample_rows and sample_rows > 0 and len(frame) > sample_rows:
        frame = frame.sample(n=int(sample_rows), random_state=42).reset_index(drop=True)
        warnings.append(
            f"已按 sample_rows={sample_rows} 抽样计算画像（原 {len(frame)} 行被抽样），"
            "统计量与全量口径不同，报表需注明"
        )

    target_column = str(target["column"])
    if target_column not in frame.columns:
        raise ValueError(
            f"目标列 {target_column!r} 不在数据中；可用列：{list(frame.columns)[:20]}"
        )
    series = frame[target_column]
    task = resolve_task(series, target.get("task"))
    labels = target.get("labels")

    columns = column_infos(
        frame, {**(roles or {}), "target_candidates": [target_column]}
    )
    shape = {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])}
    distribution = _target_distribution(frame, target_column, task, labels)
    payload = distribution["payload"]
    frame_rows = distribution["frame"]
    if frame_rows.empty:
        frame_rows = pd.DataFrame(columns=["label", "label_name", "count", "share"])

    describe_frame = _describe(frame, roles or {}, target_column)
    missing_frame = missing_summary_frame(columns)
    missing_summary = missing_frame.to_dict(orient="records")
    feature_types = _feature_types(frame, columns)

    # ---------- 时间维度 ----------
    time_column = str(roles.get("time") or "") if roles else ""
    if not time_column or time_column not in frame.columns:
        warnings.append(
            "无时间列：已跳过月份 × 标签交叉与按月坏率趋势图（不伪造时间维度）"
        )
        monthly_frame: Optional[pd.DataFrame] = None
        monthly: Optional[List[Dict[str, Any]]] = None
    else:
        monthly_frame, monthly, month_warnings = _monthly_cross(
            frame, target_column, task, labels, time_column
        )
        warnings.extend(month_warnings)

    # ---------- 口径与样本量告警（失败必须可见） ----------
    if task == "binary":
        bad_rate = float(payload.get("bad_rate") or 0.0)
        if 0.0 < bad_rate < LOW_BAD_RATE:
            warnings.append(
                f"坏样本占比仅 {bad_rate:.2%}（低于 {LOW_BAD_RATE:.0%}）：IV/KS 等读数对抽样噪声敏感，"
                "解释时需显式提示小样本与不平衡偏差"
            )
        if payload.get("bad_count", 0) == 0 or payload.get("good_count", 0) == 0:
            warnings.append("目标列只有单一类别：无法进行二分类建模（后续 WOE/IV 会失败）")
    if int(shape["rows"]) < SMALL_SAMPLE:
        warnings.append(
            f"样本量仅 {shape['rows']} 行（低于 {SMALL_SAMPLE}）：分箱/IV 建议降低箱数并标注不可靠"
        )
    if payload.get("missing", 0) > 0:
        warnings.append(
            f"目标列有 {payload['missing']} 行缺失标签（占 {payload['missing_rate']:.2%}）："
            "建模时需先处理（剔除或单独成类），本画像的分母为有效标签数"
        )
    high_missing = [row for row in missing_summary if float(row["missing_rate"]) >= HIGH_MISSING]
    if high_missing:
        warnings.append(
            f"{len(high_missing)} 列缺失率 ≥ {HIGH_MISSING:.0%}（如 "
            + "、".join(row["column"] for row in high_missing[:5])
            + "）：特征分析仍会计算，但需标注"
        )
    numeric_count = int(feature_types["counts"]["numeric"])
    if numeric_count > max_features_for_charts:
        warnings.append(
            f"数值特征 {numeric_count} 个，超过 max_features_for_charts={max_features_for_charts}："
            "图表只画前若干个，其余以表格呈现（宽表性能策略）"
        )

    return ProfileResult(
        target={"column": target_column, "task": task, "labels": dict(labels or {})},
        shape=shape,
        target_distribution=payload,
        target_distribution_frame=frame_rows,
        missing_summary=missing_summary,
        columns=columns,
        describe_frame=describe_frame,
        feature_types=feature_types,
        monthly_frame=monthly_frame,
        monthly=monthly,
        warnings=warnings,
    )
