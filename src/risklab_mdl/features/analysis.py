"""特征分析：逐特征分箱 + WOE/IV 明细（二分类 / 多分类 OVR）。

分箱方法（``config.method``）
-----------------------------
===============  ==========================================================
``quantile``     ``core.binning.fit_bins(BinSpec(method="quantile"))`` 等频
``equal_width``  ``core.binning.fit_bins(BinSpec(method="equal_width"))`` 等宽
``dt``           ``core.binning_dt.fit_bins_dt(DTBinningSpec(max_bins=bins))``
``custom``       调用方给 ``edges``（``BinSpec(method="business_cut")``）
===============  ==========================================================

★ 只在 train 上拟合的语义
    ``core.binning.fit_bins`` / ``core.binning_dt.fit_bins_dt`` **都要求在 train 上拟合**
    （防穿越），它们自己不做数据切分 —— 切分是编排层的责任：本层把**传入的
    整份数据当作训练集**拟合，并把这句话写进 ``warnings``；划分之后再重跑一次
    即可得到无穿越的分箱。

多分类口径
----------
按类 one-vs-rest：对第 k 类构造 ``y_k = (y == k)``，复用二分类同一套
``fit_bins/transform_bins/bin_table_processed/iv_of``，得到逐类 IV；
``iv_macro`` = 各类 IV 的算术平均。**不把多分类压成二分类的坏率**
（那会伪造一个不存在的指标）。

失败必须可见
------------
非数值 / 整列缺失 / 取值恒定（等宽无法切）等情形**不静默跳过**：
逐条写进 ``skipped: [{feature, reason}]`` 与 ``iv_table.skipped/reason``。
高缺失列照算，但 ``warnings`` 与 ``iv_table`` 的 ``missing_rate`` 上标注。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.binning import BinSpec, bin_table_processed, fit_bins, iv_of, transform_bins
from ..core.binning_dt import DTBinningSpec, fit_bins_dt
from ..core.errors import RiskLabError

from ..data.loader import (
    as_numeric,
    binary_bad_label,
    infer_semantic_dtype,
    is_numeric_series,
    resolve_task,
    to_jsonable,
)
from ..core.utils import bin_interval_label, safe_ratio, same_value_rate

__all__ = [
    "BINNING_METHODS",
    "HIGH_MISSING_WARN",
    "IV_TABLE_COLUMNS",
    "WOE_DETAIL_COLUMNS",
    "FeatureAnalysisResult",
    "analyze_features",
    "describe_bins",
]

#: 支持的分箱方法（契约 §3.2）。
BINNING_METHODS: tuple[str, ...] = ("quantile", "equal_width", "dt", "custom")

#: 高缺失列的标注阈值（照算，但要标注）。
HIGH_MISSING_WARN = 0.5

#: ``iv_table.csv`` 的固定列（多分类会追加 ``iv_<label>`` 动态列）。
IV_TABLE_COLUMNS: tuple[str, ...] = (
    "feature",
    "iv",
    "n_bins",
    "n_bins_fitted",
    "missing_rate",
    "same_value_rate",
    "method",
    "task",
    "skipped",
    "reason",
)

#: ``woe_detail.csv`` 的固定列。
WOE_DETAIL_COLUMNS: tuple[str, ...] = (
    "feature",
    "class",
    "bin",
    "bin_index",
    "bin_label",
    "total",
    "bad",
    "good",
    "bad_rate",
    "share",
    "woe",
    "iv_contribution",
    "smoothed",
    "note",
)

#: 非数值列的跳过理由（统一文案，便于下游按前缀匹配）。
SKIP_NON_NUMERIC = "非数值，需先编码"
SKIP_ALL_MISSING = "整列缺失，无有效取值"
SKIP_CONSTANT = "取值恒定（单值），无法分箱"
SKIP_TARGET_MISSING = "目标列在该特征有效样本上无好坏两类"


def describe_bins(spec: BinSpec) -> Dict[str, Any]:
    """把 ``BinSpec`` 转成可落盘的 dict（``binning/<feature>.json`` 用）。"""
    return {
        "method": spec.method,
        "n_bins": int(spec.n_bins),
        "n_valid_bins": int(spec.n_valid_bins),
        "requested_bins": int(spec.requested_bins),
        "bins_downgraded": bool(spec.bins_downgraded),
        "edges": [float(e) for e in (spec.edges or ())],
        "missing_as_bin": bool(spec.missing_as_bin),
        "min_bin_count": int(spec.min_bin_count),
        "min_bin_pct": float(spec.min_bin_pct),
        "sparse_strategy": spec.sparse_strategy,
        "woe_smoothing": float(spec.woe_smoothing),
        "fit_on_months": list(spec.fit_on_months or ()),
    }


def _fit_spec(
    x_numeric: pd.Series,
    y_binary: pd.Series,
    *,
    method: str,
    bins: int,
    edges: Optional[Sequence[float]],
) -> Tuple[BinSpec, List[str]]:
    """拟合一层分箱，返回 ``(BinSpec, notes)``。异常向上抛，由调用方归类为 skipped。"""
    if method == "dt":
        result = fit_bins_dt(x_numeric, y_binary, DTBinningSpec(max_bins=max(2, int(bins))))
        return result.spec, list(result.notes)
    if method == "custom":
        if not edges:
            raise ValueError("method='custom' 需要在 edges 中给出切点（至少一个）")
        spec = BinSpec(
            method="business_cut",
            n_bins=len(edges) + 1,
            edges=tuple(float(e) for e in edges),
        )
        return spec, []
    if method in ("quantile", "equal_width"):
        spec = BinSpec(method=method, n_bins=max(2, int(bins)))
        return fit_bins(x_numeric, spec), []
    raise ValueError(
        f"不支持的分箱方法：{method!r}（可用：{', '.join(BINNING_METHODS)}）"
    )


@dataclass
class FeatureAnalysisResult:
    """特征分析结果（表格 + 可复用分箱规格 + 跳过清单）。"""

    iv_table: pd.DataFrame
    woe_detail: pd.DataFrame
    bin_specs: Dict[str, Dict[str, Any]]
    skipped: List[Dict[str, str]]
    warnings: List[str]
    feature_count: int
    target: Dict[str, Any]

    def iv_top(self, limit: int = 20) -> List[Dict[str, Any]]:
        """按 IV 降序的 TOP 特征（跳过的特征不参与排序）。"""
        if not len(self.iv_table):
            return []
        usable = self.iv_table[~self.iv_table["skipped"].astype(bool)].copy()
        if not len(usable):
            return []
        usable = usable.sort_values("iv", ascending=False).head(limit)
        return [
            {"feature": str(row.feature), "iv": round(float(row.iv), 6)}
            for row in usable.itertuples()
        ]

    def data_block(self, *, top: int = 20) -> Dict[str, Any]:
        """工具返回的 ``data``（契约 §3.2：feature_count / iv_top / skipped）。"""
        return {
            "feature_count": int(self.feature_count),
            "analyzed_count": int(self.iv_table["skipped"].eq(False).sum())
            if len(self.iv_table)
            else 0,
            "iv_top": self.iv_top(top),
            "skipped": to_jsonable(self.skipped),
            "target": to_jsonable(self.target),
        }


def _select_features(
    frame: pd.DataFrame,
    roles: Optional[Dict[str, Any]],
    target_column: str,
    features: Optional[Sequence[str]],
) -> List[str]:
    """确定待分析特征：显式列表优先，否则按角色排除 id/time/split/target。"""
    all_columns = [str(c) for c in frame.columns]
    if features is not None:
        missing = [name for name in features if name not in all_columns]
        if missing:
            raise ValueError(
                f"config.features 中的列不存在：{missing[:10]}；可用列示例：{all_columns[:10]}"
            )
        return [str(name) for name in features]

    excluded: set[str] = {target_column}
    if roles:
        for key in ("id", "time", "split"):
            value = roles.get(key)
            if value:
                excluded.add(str(value))
    return [name for name in all_columns if name not in excluded]


def _class_masks(
    target_series: pd.Series, task: str, labels: Optional[Dict[str, str]]
) -> List[Tuple[str, pd.Series]]:
    """构造 ``[(class_key, y_binary)]``；二分类只有一个整体层。"""
    if task == "binary":
        bad_label = binary_bad_label(target_series.dropna(), labels)
        y_binary = pd.Series(
            (target_series.astype(str) == bad_label).astype(int), index=target_series.index
        )
        y_binary = y_binary.where(target_series.notna())
        return [("binary", y_binary)]
    values = sorted({str(v) for v in target_series.dropna().unique()})
    out: List[Tuple[str, pd.Series]] = []
    for value in values:
        y_binary = pd.Series(
            (target_series.astype(str) == value).astype(int), index=target_series.index
        )
        y_binary = y_binary.where(target_series.notna())
        out.append((value, y_binary))
    return out


def analyze_features(
    frame: pd.DataFrame,
    *,
    target: Dict[str, Any],
    method: str = "quantile",
    bins: int = 8,
    features: Optional[Sequence[str]] = None,
    roles: Optional[Dict[str, Any]] = None,
    custom_edges: Optional[Dict[str, Sequence[float]]] = None,
    max_features_for_charts: int = 30,
) -> FeatureAnalysisResult:
    """逐特征分箱 + WOE/IV（纯计算，不落盘、不画图）。

    Parameters
    ----------
    target
        ``normalize_target`` 的输出（``column`` / ``task`` / ``labels``）。
    method
        ``quantile`` / ``equal_width`` / ``dt`` / ``custom``。
    bins
        请求的箱数（``dt`` 时为 ``max_bins`` 上限）。样本量不足时会由 core 自动下调，
        下调事实写进 ``iv_table.n_bins`` 与 ``warnings``。
    features
        待分析特征列表；``None`` ⇒ 除 id/time/split/target 外的全部列
        （**含非数值列**，它们会出现在 ``skipped`` 里，便于用户一次看全）。
    custom_edges
        ``method='custom'`` 时逐特征的切点映射。
    """
    if method not in BINNING_METHODS:
        raise ValueError(f"不支持的分箱方法：{method!r}（可用：{', '.join(BINNING_METHODS)}）")

    target_column = str(target["column"])
    if target_column not in frame.columns:
        raise ValueError(f"目标列 {target_column!r} 不在数据中")
    target_series = frame[target_column]
    task = resolve_task(target_series, target.get("task"))
    labels = target.get("labels")
    if int(target_series.dropna().shape[0]) == 0:
        raise ValueError(f"目标列 {target_column!r} 全为缺失，无法计算 WOE/IV")

    selected = _select_features(frame, roles, target_column, features)
    class_masks = _class_masks(target_series, task, labels)
    bins = max(2, int(bins))

    warnings: List[str] = [
        "分箱在传入数据的全量样本上拟合；dataset_split 尚未执行时全量≈训练集，"
        "划分完成后需重跑以得到无穿越的分箱",
    ]
    skipped: List[Dict[str, str]] = []
    iv_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    bin_specs: Dict[str, Dict[str, Any]] = {}
    downgraded: List[str] = []
    high_missing: List[str] = []
    constant_features: List[str] = []

    for name in selected:
        series = frame[name]
        missing_rate = safe_ratio(float(series.isna().sum()), float(len(frame)))
        same_value = same_value_rate(series)
        base_row: Dict[str, Any] = {
            "feature": name,
            "iv": float("nan"),
            "n_bins": 0,
            "n_bins_fitted": 0,
            "missing_rate": round(missing_rate, 6),
            "same_value_rate": round(same_value, 6),
            "method": method,
            "task": task,
            "skipped": True,
            "reason": "",
        }

        # ---------- 可分析性判定（失败必须可见） ----------
        if int(series.dropna().shape[0]) == 0:
            base_row["reason"] = SKIP_ALL_MISSING
            skipped.append({"feature": name, "reason": SKIP_ALL_MISSING})
            iv_rows.append(base_row)
            continue
        if not is_numeric_series(series):
            dtype = infer_semantic_dtype(series)
            reason = f"{SKIP_NON_NUMERIC}（推断类型 {dtype}）"
            base_row["reason"] = reason
            skipped.append({"feature": name, "reason": reason})
            iv_rows.append(base_row)
            continue

        numeric = as_numeric(series)[0]
        if missing_rate >= HIGH_MISSING_WARN:
            high_missing.append(f"{name}（缺失 {missing_rate:.0%}）")

        # ---------- 逐层分箱（二分类一层 / 多分类 OVR 多层） ----------
        per_class_specs: Dict[str, Any] = {}
        class_ivs: Dict[str, float] = {}
        notes: List[str] = []
        failure: Optional[str] = None
        merged_note: Optional[str] = None
        kept_bins = 0

        for class_key, y_binary in class_masks:
            # ★ 只按「标签缺失」筛行：特征缺失必须保留，否则缺失箱(-1)永远出不来。
            #   fit_bins / fit_bins_dt 内部会自行 dropna 拟合切点，transform_bins 把
            #   缺失映射成 MISSING_BIN，bin_table 单独统计该箱 —— 口径才和报表一致。
            keep = y_binary.notna()
            pair = pd.DataFrame(
                {"x": numeric[keep], "y": y_binary[keep]}, index=numeric.index[keep]
            )
            if not len(pair):
                failure = SKIP_TARGET_MISSING
                break
            y = pair["y"].astype(int)
            if int(y.sum()) == 0 or int((1 - y).sum()) == 0:
                failure = f"{SKIP_TARGET_MISSING}（类 {class_key} 在全样本上无正/负例）"
                break
            if int(pair["x"].nunique(dropna=True)) < 2:
                failure = SKIP_CONSTANT
                break
            try:
                spec, fit_notes = _fit_spec(
                    pair["x"],
                    y,
                    method=method,
                    bins=bins,
                    edges=(custom_edges or {}).get(name),
                )
                codes = transform_bins(pair["x"], spec)
                table = bin_table_processed(codes, y, spec)
                iv = iv_of(table)
            except RiskLabError as exc:  # core 的 DataError / ConfigError
                failure = f"分箱失败：{exc.message}"
                break
            except ValueError as exc:
                failure = f"分箱失败：{exc}"
                break

            notes.extend(fit_notes)
            per_class_specs[class_key] = describe_bins(spec)
            class_ivs[class_key] = float(iv)
            fitted_valid = int(spec.n_valid_bins)
            kept_now = int((table["bin"] >= 0).sum())
            if class_key == class_masks[0][0]:
                kept_bins = kept_now
            note = ""
            if kept_now < fitted_valid:
                note = "稀疏箱已合并（箱编码保留原始切点区间，覆盖范围可能跨多个原始箱）"
                merged_note = note
            if spec.bins_downgraded:
                downgraded.append(
                    f"{name}：请求 {spec.requested_bins} 箱，实际 {spec.n_valid_bins} 箱（样本量不足）"
                )

            for idx, row in enumerate(table.to_dict(orient="records")):
                code = int(row["bin"])
                bad = float(row["坏用户"])
                good = float(row["好用户"])
                total = float(row["总用户"])
                detail_rows.append(
                    {
                        "feature": name,
                        "class": class_key,
                        "bin": code,
                        "bin_index": idx,
                        "bin_label": bin_interval_label(code, spec.edges or ()),
                        "total": int(total),
                        "bad": int(bad),
                        "good": int(good),
                        "bad_rate": round(safe_ratio(bad, total), 6),
                        "share": round(float(row["占比"]), 6),
                        "woe": round(float(row["WOE"]), 6),
                        "iv_contribution": round(float(row["IV贡献"]), 6),
                        "smoothed": bool(row["是否平滑"]),
                        "note": note,
                    }
                )

        if failure:
            base_row["reason"] = failure
            if failure == SKIP_CONSTANT:
                constant_features.append(name)
            skipped.append({"feature": name, "reason": failure})
            iv_rows.append(base_row)
            continue

        primary = per_class_specs.get("binary") or next(iter(per_class_specs.values()), {})
        iv_values = list(class_ivs.values())
        base_row["iv"] = round(float(np.mean(iv_values)), 6) if task != "binary" else round(
            float(iv_values[0]), 6
        )
        base_row["n_bins"] = int(kept_bins)
        base_row["n_bins_fitted"] = int(primary.get("n_valid_bins", 0))
        base_row["skipped"] = False
        base_row["reason"] = ""
        for class_key, value in class_ivs.items():
            if task == "multiclass":
                base_row[f"iv_{class_key}"] = round(float(value), 6)
        iv_rows.append(base_row)

        bin_specs[name] = {
            "feature": name,
            "method": method,
            "task": task,
            "target_column": target_column,
            "labels": dict(labels or {}),
            "iv": round(float(base_row["iv"]), 6),
            "per_class": per_class_specs,
            "notes": notes,
        }

    iv_table = pd.DataFrame(iv_rows)
    if len(iv_table):
        dynamic = [c for c in iv_table.columns if str(c).startswith("iv_")]
        ordered = [c for c in IV_TABLE_COLUMNS if c in iv_table.columns] + dynamic
        iv_table = iv_table[ordered]
        iv_table = iv_table.sort_values(
            ["skipped", "iv"], ascending=[True, False], na_position="last"
        ).reset_index(drop=True)
    else:
        iv_table = pd.DataFrame(columns=list(IV_TABLE_COLUMNS))
    woe_detail = pd.DataFrame(detail_rows) if detail_rows else pd.DataFrame(columns=list(WOE_DETAIL_COLUMNS))
    if len(woe_detail):
        woe_detail = woe_detail[list(WOE_DETAIL_COLUMNS)]

    # ---------- 汇总告警（不静默） ----------
    if skipped:
        reasons: Dict[str, int] = {}
        for row in skipped:
            key = row["reason"].split("（")[0]
            reasons[key] = reasons.get(key, 0) + 1
        warnings.append(
            f"{len(skipped)} 个特征被跳过：" + "；".join(f"{k} × {v}" for k, v in reasons.items())
        )
    if downgraded:
        warnings.append("箱数因样本量不足被下调：" + "；".join(downgraded[:10]))
    if high_missing:
        warnings.append(
            f"{len(high_missing)} 个特征缺失率 ≥ {HIGH_MISSING_WARN:.0%}（已照算，需在报表标注）："
            + "、".join(high_missing[:10])
        )
    if constant_features:
        warnings.append(
            f"{len(constant_features)} 个特征取值恒定，IV 无意义（已在 skipped 中列出）"
        )
    if task == "multiclass":
        warnings.append(
            "多分类按类 one-vs-rest 计算逐类 IV，iv 列为各类 IV 的 macro 平均（不是压成二分类的坏率）"
        )
    if len(selected) > max_features_for_charts:
        warnings.append(
            f"特征 {len(selected)} 个超过 max_features_for_charts={max_features_for_charts}："
            "只有前若干个出箱图，其余仅出表（宽表性能策略）"
        )
    if merged_note:
        warnings.append("部分特征的稀疏箱已由 core.binning 合并：" + merged_note)

    return FeatureAnalysisResult(
        iv_table=iv_table.reset_index(drop=True),
        woe_detail=woe_detail.reset_index(drop=True),
        bin_specs=bin_specs,
        skipped=skipped,
        warnings=warnings,
        feature_count=len(selected),
        target={"column": target_column, "task": task, "labels": dict(labels or {})},
    )


def woe_matrix(woe_detail: pd.DataFrame, feature: str, class_key: str = "binary") -> pd.DataFrame:
    """取单个特征单层的 WOE 明细（出图用；不存在时返回空表）。"""
    if not len(woe_detail):
        return woe_detail
    mask = (woe_detail["feature"] == feature) & (woe_detail["class"] == class_key)
    return woe_detail[mask].reset_index(drop=True)
