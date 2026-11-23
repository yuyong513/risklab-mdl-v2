"""特征筛选家族工具（契约 §3.2）：``feature_prescreen`` / ``feature_refine``。

两个阶段的口径来自 ``core.split``（**全项目唯一来源**）：

====================  ==========================================
阶段                    用哪些数据
====================  ==========================================
特征初筛（prescreen）   只有 train（IV/PSI 都要用「训练期学到的分布」）
特征精筛（refine）      train + val（train 训练、val 早停/记录 AUC）
====================  ==========================================

``feature_prescreen`` 跑 ``core.prescreen`` 的五道筛子（固定顺序：缺失率 → 同值率 →
IV → PSI → 相关性），阈值全部显式注入并回显在产物里。二分类直接复用
``run_prescreen``；多分类按类 OVR 逐类算 IV（**取各类 IV 的最大值**判定，逐类明细落表），
缺失/同值/PSI 三道与标签无关的筛子只跑一遍。

``feature_refine`` 跑 ``core.refine.select_features_by_gain``（gain 迭代），
候选池必须来自初筛存活（或显式 candidates）—— **绝不默认取所有列**（那会把标签列
送进模型：AUC/KS 直接 1.0 且线上不可复现）。

失败必须可见：缺 lightgbm ⇒ ``dependency_missing``；无划分且无法按时点兜底 ⇒
``config_invalid`` 并给处置建议；PSI 因缺时间列/单月而跳过 ⇒ ``warnings`` 明示，不伪造数值。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .. import __version__
from ..artifacts.layout import resolve_project
from ..artifacts.manifest import now_iso, read_manifest, stage_done, update_manifest
from ..core.errors import ConfigError, DataError, ModelError
from ..core.prescreen import (
    PrescreenConfig,
    run_prescreen,
    screen_correlation,
    screen_iv,
    screen_missing_rate,
    screen_same_value_rate,
)
from ..core.psi import PsiConfig, screen_psi
from ..core.refine import RefineConfig, select_features_by_gain
from ..core.split import (
    SPLIT_STAGE_REFINE,
    SplitSpec,
    resolve_split_subset,
    split_frame,
)
from ..data.loader import (
    binary_bad_label,
    is_numeric_series,
    normalize_target,
    read_csv_auto,
    resolve_task,
    to_jsonable,
)
from ..platform.fs import atomic_write_json, atomic_write_text, read_json
from ..preprocess.check import select_feature_columns
from ..project import resolve_csv_path, roles_for_analysis
from ..protocol import (
    relative_paths,
    result_error,
    result_from_exception,
    result_needs_input,
    result_ok,
    target_question,
)

__all__ = ["DEFAULT_THRESHOLDS", "feature_prescreen", "feature_refine"]

#: 初筛默认阈值（工具层默认；core 层按「零兜底」原则不设默认值）。
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "missing_rate_max": 0.5,
    "iv_min": 0.02,
    "psi_max": 0.10,
    "corr_max": 0.8,
    "same_value_rate_max": 0.95,
}

#: 五道筛子的固定顺序（与 ``core.prescreen`` 一致）。
STAGE_ORDER: Tuple[str, ...] = (
    "missing_rate",
    "same_value_rate",
    "iv",
    "psi",
    "correlation",
)

#: 初筛产物 schema。
PRESCREEN_SCHEMA = "risklab-mdl.prescreen/1"

#: 精筛产物 schema。
REFINE_SCHEMA = "risklab-mdl.refine/1"


# --------------------------------------------------------------------------- #
# 公共：阈值 / 取数口径 / 候选池
# --------------------------------------------------------------------------- #
def _prescreen_thresholds(cfg: Mapping[str, Any]) -> Dict[str, float]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    for key in DEFAULT_THRESHOLDS:
        if cfg.get(key) is not None:
            try:
                thresholds[key] = float(cfg[key])
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"config.{key} 必须是数字，收到 {cfg[key]!r}") from exc
    for key, value in thresholds.items():
        if not 0.0 < value < 1.0:
            raise ConfigError(f"config.{key}={value} 必须落在 (0, 1) 区间")
    return thresholds


def _read_split_doc(layout) -> Optional[Dict[str, Any]]:
    """读 ``split/split.json``（不存在/损坏返回 None，由调用方决定是否告警）。"""
    path = layout.subdir("split") / "split.json"
    if not path.is_file():
        return None
    try:
        doc = read_json(path)
    except Exception:  # noqa: BLE001 - 损坏的文件不能用于取数，但上层会给出替代口径
        return None
    return doc if isinstance(doc, dict) else None


def _train_mask_from_split(layout, frame: pd.DataFrame, target_column: str):
    """按 ``split/split.json`` 复算 train mask；返回 ``(mask, info, warnings)``。"""
    warnings: List[str] = []
    doc = _read_split_doc(layout)
    if not doc:
        return None, {"source": "未划分（全量）"}, warnings
    spec = doc.get("spec")
    if not isinstance(spec, Mapping):
        warnings.append("split/split.json 缺少 spec：初筛退回全量数据口径")
        return None, {"source": "未划分（全量）"}, warnings
    try:
        result = split_frame(
            frame, SplitSpec.from_config(spec), target=target_column
        )
    except Exception as exc:  # noqa: BLE001 - 复算失败必须可见，但不阻断初筛
        warnings.append(
            f"split/split.json 无法复算（{type(exc).__name__}: {exc}）：初筛退回全量数据口径"
        )
        return None, {"source": "未划分（全量）"}, warnings
    mask = result.train_mask.reindex(frame.index).fillna(False).astype(bool)
    if not bool(mask.any()):
        warnings.append("split/split.json 的 train 子集为空：初筛退回全量数据口径")
        return None, {"source": "未划分（全量）"}, warnings
    info = {
        "source": "split/split.json（train 子集）",
        "mode": result.spec.mode,
        "fingerprint": result.fingerprint,
        "train_rows": int(mask.sum()),
        "total_rows": int(frame.shape[0]),
    }
    return mask, info, warnings


def _prescreen_candidates(
    frame: pd.DataFrame,
    *,
    roles: Mapping[str, Any],
    target_column: str,
    features: Optional[Sequence[str]],
) -> Tuple[List[str], List[str]]:
    """返回 ``(数值候选, 非数值列)``；非数值列不静默丢弃，由调用方写进产物。"""
    names = select_feature_columns(
        frame,
        roles=roles,
        target_column=target_column,
        features=features,
    )
    numeric: List[str] = []
    non_numeric: List[str] = []
    for name in names:
        (numeric if is_numeric_series(frame[name]) else non_numeric).append(name)
    return numeric, non_numeric


def _target_binary(
    series: pd.Series, labels: Optional[Mapping[str, str]]
) -> pd.Series:
    """二分类标签 → 0/1（坏=1）；目标缺失行保留 NaN，由调用方剔除。"""
    bad = binary_bad_label(series.dropna(), dict(labels or {}))
    y = pd.Series((series.astype(str) == bad).astype(int), index=series.index)
    return y.where(series.notna())


# --------------------------------------------------------------------------- #
# 多分类 OVR 初筛（组合 core 的五道筛子，逐类 IV 取最大值判定）
# --------------------------------------------------------------------------- #
def _advance(
    alive: List[str],
    dropped: List[str],
    reason: Mapping[str, str],
    metrics: Mapping[str, float],
    agg: Dict[str, Any],
    stage: str,
) -> List[str]:
    dropped_set = set(dropped)
    agg["stage_dropped"][stage] = list(dropped)
    agg["stage_reason"][stage] = dict(reason)
    agg["stage_metrics"][stage] = dict(metrics)
    return [c for c in alive if c not in dropped_set]


def _multiclass_prescreen(
    train: pd.DataFrame,
    candidates: Sequence[str],
    target: pd.Series,
    thresholds: Mapping[str, float],
    *,
    month_column: str,
    enable_psi: bool,
) -> Dict[str, Any]:
    """多分类初筛：三道与标签无关的筛子各跑一遍，IV 按类 OVR 逐类算。"""
    agg: Dict[str, Any] = {
        "survived": [],
        "stage_dropped": {},
        "stage_reason": {},
        "stage_metrics": {},
        "iv_map": {},
        "iv_by_class": {},
        "iv_bins": {},
        "notes": [],
        "psi_detail": None,
        "psi_base_used": "",
        "high_iv_columns": [],
        "high_psi_columns": [],
    }
    alive = list(candidates)
    agg["computed_counts"] = {"missing_rate": len(alive)}

    r = screen_missing_rate(train, alive, threshold=thresholds["missing_rate_max"])
    alive = _advance(alive, r.dropped, r.reason, r.metrics, agg, "missing_rate")

    r = screen_same_value_rate(train, alive, threshold=thresholds["same_value_rate_max"])
    alive = _advance(alive, r.dropped, r.reason, r.metrics, agg, "same_value_rate")

    class_values = sorted({str(v) for v in target.dropna().unique()})
    if len(class_values) < 3:
        agg["notes"].append(
            f"目标只有 {len(class_values)} 类，按二分类处理更合适（本次仍走 OVR 口径）"
        )
    per_class_reasons: Dict[str, Dict[str, str]] = {}
    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for class_value in class_values:
        y_k = pd.Series(
            (target.astype(str) == class_value).astype(int), index=target.index
        )
        iv_result = screen_iv(
            train,
            alive,
            y_k,
            threshold=thresholds["iv_min"],
            high_iv_alert=0.5,
        )
        per_class_reasons[class_value] = dict(iv_result.reason)
        per_class_metrics[class_value] = dict(iv_result.metrics)
        agg["high_iv_columns"] = sorted(
            set(agg["high_iv_columns"]) | set(iv_result.high_iv_columns)
        )

    dropped: List[str] = []
    reasons: Dict[str, str] = {}
    metrics: Dict[str, float] = {}
    for col in alive:
        values = {
            cls: per_class_metrics[cls][col]
            for cls in class_values
            if col in per_class_metrics[cls]
        }
        agg["iv_by_class"][col] = dict(values)
        if not values:
            # 所有类都算不出 IV ⇒ 记录第一类的原因（可读）
            first = next(iter(per_class_reasons.values()), {})
            dropped.append(col)
            reasons[col] = first.get(col, "各类 OVR IV 均无法计算")
            continue
        best = max(values.values())
        agg["iv_map"][col] = round(float(best), 6)
        metrics[col] = round(float(best), 6)
        if best < thresholds["iv_min"]:
            detail = "、".join(f"{cls}:{v:.4f}" for cls, v in values.items())
            dropped.append(col)
            reasons[col] = (
                f"各类 OVR IV 均低于阈值 {thresholds['iv_min']}（{detail}）"
            )
    alive = _advance(alive, dropped, reasons, metrics, agg, "iv")

    if enable_psi and month_column:
        base_months = tuple(
            sorted(train[month_column].dropna().astype(str).unique())[:1]
        )
        psi_cfg = PsiConfig(
            threshold=thresholds["psi_max"],
            base_mode="specified",
            base_months=base_months,
            rolling_base_months=base_months[:1],
        )
        psi = screen_psi(train, alive, month_column, psi_cfg)
        agg["psi_detail"] = psi.detail
        agg["psi_base_used"] = psi.base_used
        agg["high_psi_columns"] = list(psi.high_psi_columns)
        if psi.note:
            agg["notes"].append(psi.note)
        alive = _advance(alive, psi.dropped, psi.reason, psi.metrics, agg, "psi")

    importance = {c: agg["iv_map"].get(c, 0.0) for c in alive}
    corr = screen_correlation(
        train, alive, importance, threshold=thresholds["corr_max"], strategy="greedy"
    )
    alive = _advance(alive, corr.dropped, corr.reason, corr.metrics, agg, "correlation")

    agg["survived"] = alive
    agg["computed_counts"]["iv"] = len(candidates)
    return agg


# --------------------------------------------------------------------------- #
# feature_prescreen
# --------------------------------------------------------------------------- #
def _feature_records(
    candidates: Sequence[str],
    prescreen: Mapping[str, Any],
    *,
    non_numeric: Sequence[str],
) -> List[Dict[str, Any]]:
    """逐特征判定表（初筛结论的所有依据都在这里，可追查）。"""
    survived = set(prescreen.get("survived") or [])
    records: List[Dict[str, Any]] = []
    for col in candidates:
        dropped_stage = None
        reason = ""
        for stage in STAGE_ORDER:
            stage_reason = prescreen.get("stage_reason", {}).get(stage) or {}
            if col in stage_reason:
                dropped_stage = stage
                reason = stage_reason[col]
                break
        metrics: Dict[str, Any] = {}
        stage_metrics = prescreen.get("stage_metrics") or {}
        for stage in STAGE_ORDER:
            value = (stage_metrics.get(stage) or {}).get(col)
            if value is not None:
                metrics[stage] = round(float(value), 6)
        record: Dict[str, Any] = {
            "feature": col,
            "survived": col in survived,
            "dropped_stage": dropped_stage if col not in survived else None,
            "reason": reason if col not in survived else "",
            "metrics": {
                "missing_rate": metrics.get("missing_rate"),
                "same_value_rate": metrics.get("same_value_rate"),
                "iv": metrics.get("iv") if metrics.get("iv") is not None
                else (prescreen.get("iv_map") or {}).get(col),
                "psi": metrics.get("psi"),
            },
        }
        if (prescreen.get("iv_by_class") or {}).get(col):
            record["iv_by_class"] = prescreen["iv_by_class"][col]
        records.append(record)

    for col in non_numeric:
        records.append(
            {
                "feature": col,
                "survived": False,
                "dropped_stage": "type_check",
                "reason": "非数值列：初筛只处理数值特征，需先 preprocess_apply 编码",
                "metrics": {},
            }
        )
    return records


def feature_prescreen(
    project_dir: str, config: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """初步筛选：缺失率 / 同值率 / IV / PSI / 相关性五道筛子。"""
    try:
        cfg = dict(config or {})
        layout = resolve_project(project_dir)
        manifest = read_manifest(layout.project_dir)
        csv = resolve_csv_path(layout, manifest)

        target = normalize_target(cfg.get("target")) or normalize_target(
            (manifest.get("project") or {}).get("target")
        )
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

        try:
            thresholds = _prescreen_thresholds(cfg)
        except ConfigError as exc:
            return result_error(
                "config_invalid",
                exc.message,
                "阈值都是占比，必须落在 (0, 1)；见 docs/preprocess-split.md",
            )

        read = read_csv_auto(csv)
        frame = read.frame
        warnings: List[str] = list(read.warnings)
        roles = roles_for_analysis(manifest, frame)

        try:
            numeric, non_numeric = _prescreen_candidates(
                frame,
                roles=roles,
                target_column=target["column"],
                features=cfg.get("features"),
            )
        except ValueError as exc:
            return result_error(
                "config_invalid", str(exc), "检查 config.features 是否写错"
            )
        if not numeric:
            return result_error(
                "features_empty",
                "没有可初筛的数值特征（全是非数值列或被角色排除）",
                "先 preprocess_apply 把类别列编码成数值，或用 config.features 指定数值列",
            )

        # ---------- 取数口径：初筛只用 train ----------
        train_mask, scope, scope_warnings = _train_mask_from_split(
            layout, frame, target["column"]
        )
        scope.setdefault("rows_total", int(frame.shape[0]))
        warnings.extend(scope_warnings)
        if train_mask is None:
            train_frame = frame
            scope["note"] = (
                "未找到可用的 split/split.json ⇒ 初筛使用全量数据"
                "（无切分信息时无从界定训练期，这是 core.split 的既定兜底口径）"
            )
            warnings.append(scope["note"])
        else:
            train_frame = frame.loc[train_mask]

        target_series = train_frame[target["column"]]
        y = _target_binary(target_series, (target or {}).get("labels"))
        kept = y.notna()
        dropped_target = int((~kept).sum())
        if dropped_target:
            warnings.append(
                f"train 中有 {dropped_target} 行目标缺失，已从初筛样本中剔除（分母随之变化）"
            )
        train = train_frame.loc[kept]
        y = y.loc[kept]
        scope["rows_used"] = int(train.shape[0])
        scope["target_missing_rows"] = dropped_target

        task = resolve_task(frame[target["column"]], (target or {}).get("task"))

        # ---------- PSI 可用性（无时间列就跳过并告警，不伪造） ----------
        time_column = (
            cfg.get("month_column")
            or cfg.get("time_column")
            or roles.get("time")
            or ""
        )
        month_column = ""
        enable_psi = False
        psi_note = ""
        if not time_column:
            psi_note = "未识别到时间列：PSI 跳过（不伪造数值）"
        elif str(time_column) not in train.columns:
            psi_note = f"时间列 {time_column!r} 不在数据中：PSI 跳过"
        else:
            n_months = int(train[str(time_column)].nunique(dropna=True))
            if n_months < 2:
                psi_note = (
                    f"时间列 {time_column!r} 在 train 中只有 {n_months} 个月："
                    f"无跨期可比，PSI 跳过（不伪造数值）"
                )
            else:
                month_column = str(time_column)
                enable_psi = True
                psi_note = f"PSI 基期取最早月份；按月对比 {n_months} 个月"
        if psi_note:
            warnings.append(psi_note)

        # ---------- 五道筛子 ----------
        try:
            if task == "binary":
                base_months: Tuple[str, ...] = ()
                if enable_psi and month_column:
                    # 基期显式给「最早月份」：core.prescreen 的零兜底策略要求
                    # 基期由调用方注入，不能让它默认成「全部月份」（等于没做基期对比）。
                    base_months = tuple(
                        sorted(train[month_column].dropna().astype(str).unique())[:1]
                    )
                prescreen_cfg = PrescreenConfig(
                    missing_threshold=thresholds["missing_rate_max"],
                    same_value_threshold=thresholds["same_value_rate_max"],
                    iv_threshold=thresholds["iv_min"],
                    psi_threshold=thresholds["psi_max"],
                    corr_threshold=thresholds["corr_max"],
                    month_column=month_column,
                    enable_psi=enable_psi,
                    psi_base_months=base_months,
                    psi_rolling_base_months=base_months[:1],
                )
                result = run_prescreen(train, numeric, y.astype(int), prescreen_cfg)
                prescreen: Dict[str, Any] = {
                    "survived": list(result.survived),
                    "stage_dropped": dict(result.stage_dropped),
                    "stage_reason": dict(result.stage_reason),
                    "stage_metrics": dict(result.stage_metrics),
                    "iv_map": dict(result.iv_map),
                    "notes": list(result.notes),
                    "psi_base_used": result.psi_base_used,
                    "high_iv_columns": list(result.high_iv_columns),
                    "high_psi_columns": list(result.high_psi_columns),
                    "computed_counts": dict(result.computed_counts),
                }
            else:
                prescreen = _multiclass_prescreen(
                    train,
                    numeric,
                    train[target["column"]],
                    thresholds,
                    month_column=month_column,
                    enable_psi=enable_psi,
                )
                warnings.append(
                    "多分类按类 one-vs-rest 计算 IV：某类 IV 达标即保留，"
                    "判定值取各类 IV 的最大值；逐类 IV 见 prescreen.json 的 iv_by_class"
                )
        except DataError as exc:
            return result_error(
                "analysis_failed",
                exc.message,
                "检查训练样本是否同时含好坏样本、特征是否可数值化",
            )

        warnings.extend(prescreen.get("notes") or [])
        records = _feature_records(numeric, prescreen, non_numeric=non_numeric)
        survived = list(prescreen.get("survived") or [])
        if not survived:
            warnings.append(
                "初筛后无特征存活：可放宽容差（missing_rate_max/iv_min/corr_max）后重跑"
            )
        if non_numeric:
            warnings.append(
                f"{len(non_numeric)} 个非数值特征未参与初筛：{non_numeric[:10]}"
                "（需先 preprocess_apply 编码）"
            )

        # ---------- 落盘 ----------
        selection_dir = layout.subdir("feature_selection")
        prescreen_path = selection_dir / "prescreen.json"
        iv_top = sorted(
            [(col, float(prescreen.get("iv_map", {}).get(col, 0.0))) for col in survived],
            key=lambda item: -item[1],
        )[:20]
        payload = {
            "schema": PRESCREEN_SCHEMA,
            "version": __version__,
            "generated_at": now_iso(),
            "target": target,
            "task": task,
            "thresholds": thresholds,
            "data_scope": scope,
            "time_column": time_column or None,
            "psi": {
                "enabled": bool(enable_psi),
                "base_used": prescreen.get("psi_base_used") or "",
                "note": psi_note,
            },
            "stages": list(STAGE_ORDER),
            "summary": {
                "total_input": len(numeric),
                "n_survived": len(survived),
                "stage_dropped": {
                    stage: len(prescreen.get("stage_dropped", {}).get(stage) or [])
                    for stage in STAGE_ORDER
                },
                "computed_counts": prescreen.get("computed_counts", {}),
                "iv_top": [{"feature": col, "iv": round(iv, 6)} for col, iv in iv_top],
                "high_iv_columns": prescreen.get("high_iv_columns", []),
                "high_psi_columns": prescreen.get("high_psi_columns", []),
            },
            "survived": survived,
            "features": records,
            "non_numeric": list(non_numeric),
            "warnings": warnings,
        }
        atomic_write_json(prescreen_path, to_jsonable(payload))

        summary_text = (
            f"初筛 {len(numeric)} → {len(survived)} 个特征"
            f"（{scope.get('source', '')}）"
        )

        def _mutate(manifest_doc: Dict[str, Any]) -> None:
            stage_done(
                manifest_doc,
                "feature_selection",
                done=False,
                path="feature_selection/",
                summary=f"{summary_text}（待精筛）",
            )

        update_manifest(layout.project_dir, _mutate)

        dropped = [
            {"feature": r["feature"], "stage": r["dropped_stage"], "reason": r["reason"]}
            for r in records
            if not r["survived"]
        ]
        data = {
            "project_dir": layout.project_dir.as_posix(),
            "task": task,
            "thresholds": thresholds,
            "data_scope": scope,
            "psi": payload["psi"],
            "n_input": len(numeric),
            "n_survived": len(survived),
            "survived": survived,
            "dropped": dropped,
            "non_numeric": list(non_numeric),
            "iv_top": payload["summary"]["iv_top"],
        }
        return result_ok(
            data,
            artifacts=relative_paths(layout, [prescreen_path]),
            warnings=warnings + [f"已写入 {layout.rel(prescreen_path)}"],
        )
    except ConfigError as exc:
        return result_error(
            "config_invalid",
            exc.message,
            "按 docs/preprocess-split.md 检查 config（阈值都在 (0,1) 内）",
        )
    except Exception as exc:  # noqa: BLE001
        return result_from_exception(exc)


# --------------------------------------------------------------------------- #
# feature_refine
# --------------------------------------------------------------------------- #
def _refine_candidates(
    layout,
    frame: pd.DataFrame,
    *,
    cfg: Mapping[str, Any],
    prescreen_doc: Optional[Mapping[str, Any]],
    target_column: str,
) -> Tuple[List[str], str, List[str]]:
    """候选池：显式 candidates > 初筛存活 > 特征分析数值列。

    Returns
    -------
    (candidates, source, warnings)
    """
    warnings: List[str] = []
    explicit = cfg.get("candidates")
    if explicit is not None:
        candidates = [str(c) for c in explicit]
        if not candidates:
            raise ConfigError("config.candidates 是空列表")
        unknown = [c for c in candidates if c not in set(map(str, frame.columns))]
        if unknown:
            raise ConfigError(
                f"config.candidates 中的列不存在：{unknown[:10]}；"
                f"可用列示例：{list(map(str, frame.columns))[:20]}"
            )
        return candidates, "config.candidates", warnings

    if prescreen_doc:
        candidates = [str(c) for c in (prescreen_doc.get("survived") or [])]
        if candidates:
            present = [c for c in candidates if c in set(map(str, frame.columns))]
            missing = [c for c in candidates if c not in set(map(str, frame.columns))]
            if missing:
                warnings.append(
                    f"{len(missing)} 个初筛存活特征在数据中不存在，已略过：{missing[:10]}"
                )
            if present:
                return present, "feature_selection/prescreen.json（初筛存活）", warnings
        warnings.append("prescreen.json 里没有存活特征：退回特征分析的数值列")

    iv_path = layout.subdir("feature_analysis") / "iv_table.csv"
    if iv_path.is_file():
        table = pd.read_csv(iv_path, encoding="utf-8", low_memory=False)
        if len(table) and "feature" in table.columns:
            usable = table[~table["skipped"].astype(bool)] if "skipped" in table.columns else table
            candidates = [
                str(c) for c in usable["feature"].tolist() if str(c) in set(map(str, frame.columns))
            ]
            if candidates:
                return candidates, "feature_analysis/iv_table.csv（可分析特征）", warnings
        warnings.append("iv_table.csv 里没有可分析特征：退回数据中的全部数值列")
    else:
        warnings.append("缺少 feature_selection/prescreen.json 与 feature_analysis/iv_table.csv："
                        "候选池退回数据中的全部数值列（建议先跑初筛）")

    roles = roles_for_analysis({}, frame)
    numeric, _non = _prescreen_candidates(
        frame, roles=roles, target_column=target_column, features=None
    )
    if not numeric:
        raise ConfigError("无法确定候选特征池：数据里没有数值列")
    return numeric, "数据中的全部数值列（无初筛产物）", warnings


def _refine_split(
    layout,
    frame: pd.DataFrame,
    *,
    target_column: str,
    roles: Mapping[str, Any],
) -> Tuple[pd.Series, pd.Series, Dict[str, Any], List[str]]:
    """精筛口径：train + val（优先 split.json，其次 core.split 阶段兜底）。"""
    warnings: List[str] = []
    doc = _read_split_doc(layout)
    if doc and isinstance(doc.get("spec"), Mapping):
        try:
            result = split_frame(
                frame, SplitSpec.from_config(doc["spec"]), target=target_column
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"split/split.json 无法复算（{type(exc).__name__}: {exc}）："
                f"退回 core.split 的阶段口径"
            )
        else:
            train_mask = result.train_mask.reindex(frame.index).fillna(False).astype(bool)
            val_mask = result.val_mask.reindex(frame.index).fillna(False).astype(bool)
            info = {
                "source": "split/split.json",
                "mode": result.spec.mode,
                "fingerprint": result.fingerprint,
                "counts": dict(result.counts),
            }
            if not bool(val_mask.any()):
                raise ConfigError(
                    "split/split.json 的 val 子集为空：gain 精筛需要验证集做早停与 AUC 记录。"
                    "请重新 dataset_split 生成 val（如 ratios=[0.7,0.2,0.1]），"
                    "或改用 mode=field 指定 val 条件"
                )
            return train_mask, val_mask, info, warnings

    try:
        subset = resolve_split_subset(
            frame,
            {},
            SPLIT_STAGE_REFINE,
            split_col=str(roles.get("split") or "") or None,
            date_col=str(roles.get("time") or "") or None,
        )
    except DataError as exc:
        raise ConfigError(
            f"精筛需要 train/val 切分，但当前数据无法切分：{exc.message}。"
            "请先 dataset_split（mode=random/stratified/field）生成 split/split.json；"
            "或确认数据里有 split 列（train/val）或时间列"
        ) from exc
    train_mask = subset.train_mask
    val_mask = subset.val_mask
    info = {
        "source": f"core.split 阶段口径（{subset.description}）",
        "mode": None,
        "fingerprint": None,
        "counts": subset.counts(),
    }
    if subset.used_fallback:
        warnings.append(f"无 split/split.json ⇒ 精筛按 core.split 兜底口径取数：{subset.description}")
    return train_mask, val_mask, info, warnings


def feature_refine(
    project_dir: str, config: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """精细筛选（gain 迭代）：产物 ``refine.json`` + ``selected_features.txt``。"""
    try:
        cfg = dict(config or {})
        layout = resolve_project(project_dir)
        manifest = read_manifest(layout.project_dir)
        csv = resolve_csv_path(layout, manifest)

        target = normalize_target(cfg.get("target")) or normalize_target(
            (manifest.get("project") or {}).get("target")
        )
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

        scheme = str(cfg.get("scheme") or cfg.get("method") or "gain").strip().lower()
        if scheme not in ("gain", "gain_iterative", "gain-iterative"):
            return result_error(
                "config_invalid",
                f"不支持的精筛方案：{scheme!r}",
                "当前仅支持 scheme='gain'（gain 迭代）；shap/boruta 未实现，不静默降级",
            )

        read = read_csv_auto(csv)
        frame = read.frame
        warnings: List[str] = list(read.warnings)
        roles = roles_for_analysis(manifest, frame)

        prescreen_path = layout.subdir("feature_selection") / "prescreen.json"
        prescreen_doc: Optional[Mapping[str, Any]] = None
        if prescreen_path.is_file():
            try:
                prescreen_doc = read_json(prescreen_path)
            except Exception as exc:  # noqa: BLE001
                warnings.append(
                    f"prescreen.json 无法解析（{type(exc).__name__}: {exc}）："
                    f"候选池退回特征分析数值列"
                )
                prescreen_doc = None

        try:
            candidates, candidate_source, candidate_warnings = _refine_candidates(
                layout,
                frame,
                cfg=cfg,
                prescreen_doc=prescreen_doc,
                target_column=target["column"],
            )
        except ConfigError as exc:
            return result_error(
                "config_invalid",
                exc.message,
                "用 config.candidates 显式指定，或先跑 feature_prescreen / feature_analyze",
            )
        warnings.extend(candidate_warnings)

        non_numeric = [c for c in candidates if not is_numeric_series(frame[c])]
        if non_numeric:
            return result_error(
                "config_invalid",
                f"候选特征含非数值列：{non_numeric[:10]}",
                "gain 迭代需要数值矩阵：先 preprocess_apply 编码，或从 candidates 中排除",
            )
        if not candidates:
            return result_error(
                "features_empty",
                "候选特征为空",
                "放宽容差重跑 feature_prescreen，或用 config.candidates 指定数值列",
            )

        try:
            train_mask, val_mask, split_info, split_warnings = _refine_split(
                layout, frame, target_column=target["column"], roles=roles
            )
        except ConfigError as exc:
            return result_error(
                "config_invalid", exc.message, getattr(exc, "hint", "") or "先 dataset_split"
            )
        warnings.extend(split_warnings)

        target_series = frame[target["column"]]
        task = resolve_task(target_series, (target or {}).get("task"))
        if task == "binary":
            y_all = _target_binary(target_series, (target or {}).get("labels"))
        else:
            classes = sorted({str(v) for v in target_series.dropna().unique()})
            mapping = {value: index for index, value in enumerate(classes)}
            y_all = target_series.astype(str).map(mapping).astype("float64")
            y_all = y_all.where(target_series.notna())
        labelled = y_all.notna()
        train_mask = train_mask & labelled
        val_mask = val_mask & labelled
        n_train = int(train_mask.sum())
        n_val = int(val_mask.sum())
        if n_train == 0 or n_val == 0:
            return result_error(
                "config_invalid",
                f"精筛需要非空的 train 与 val：当前 train {n_train} 行 / val {n_val} 行",
                "调整 split.json 的比例或改用 mode=random（如 ratios=[0.7,0.2,0.1]）",
            )

        X = frame.loc[train_mask, candidates].astype("float64")
        X_val = frame.loc[val_mask, candidates].astype("float64")
        y_train = y_all.loc[train_mask].astype(int)
        y_val = y_all.loc[val_mask].astype(int)

        # ---------- 精筛配置（区间默认值随候选数自适应，避免小数据必然落区间外） ----------
        try:
            n_candidates = len(candidates)
            lo_default = max(1, min(5, n_candidates))
            hi_default = max(lo_default, min(60, n_candidates))
            target_range = cfg.get("target_range") or [lo_default, hi_default]
            if len(target_range) != 2:
                raise ConfigError("config.target_range 必须是 [下限, 上限] 两个整数")
            lo, hi = int(target_range[0]), int(target_range[1])
            max_features = cfg.get("max_features")
            max_features = int(max_features) if max_features is not None else min(60, n_candidates)
            threshold = float(cfg.get("threshold") or 0.05)
            ladder = tuple(t for t in (0.03, 0.01) if t <= threshold)
            refine_cfg = RefineConfig(
                method="gain_iterative",
                threshold=threshold,
                max_rounds=int(cfg.get("rounds") or 10),
                num_boost_round=int(cfg.get("num_boost_round") or 1000),
                early_stopping_rounds=int(cfg.get("early_stopping_rounds") or 50),
                target_range=(lo, hi),
                max_features=max_features,
                threshold_relax_ladder=ladder,
                force_keep=tuple(str(c) for c in (cfg.get("keep") or [])),
                force_drop=tuple(str(c) for c in (cfg.get("drop") or [])),
                seed=int(cfg.get("seed") or 42),
            )
        except ConfigError as exc:
            return result_error(
                "config_invalid",
                exc.message,
                "target_range=[下限,上限]、max_features、threshold、rounds、keep/drop 见 docs/preprocess-split.md",
            )
        except (TypeError, ValueError) as exc:
            return result_error(
                "config_invalid",
                f"精筛参数类型非法：{exc}",
                "target_range / rounds / threshold / seed 必须是数字",
            )

        if task == "binary":
            params: Dict[str, Any] = {
                "objective": "binary",
                "metric": "auc",
                "verbosity": -1,
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_pre_filter": False,
            }
        else:
            params = {
                "objective": "multiclass",
                "num_class": int(y_all.max()) + 1,
                "metric": "multi_logloss",
                "verbosity": -1,
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_pre_filter": False,
            }
        params.update({str(k): v for k, v in (cfg.get("params") or {}).items()})

        forbidden = [
            str(target["column"]),
            *[str(roles.get(key)) for key in ("id", "time", "split") if roles.get(key)],
        ]

        try:
            result = select_features_by_gain(
                X,
                y_train,
                X_val,
                y_val,
                params,
                refine_cfg,
                candidates=list(candidates),
                forbidden=forbidden,
            )
        except ModelError as exc:
            detail = getattr(exc, "detail", None) or {}
            if isinstance(detail, Mapping) and detail.get("dependency"):
                return result_error(
                    "dependency_missing",
                    exc.message,
                    f"执行：{detail.get('install', 'pip install lightgbm')} 后重试",
                )
            return result_error(
                "analysis_failed",
                exc.message,
                "gain 迭代失败：检查候选特征是否全为常量、训练集是否好坏样本齐全",
            )
        except ConfigError as exc:
            return result_error(
                "config_invalid",
                exc.message,
                "检查 keep/drop/candidates 与 target_range 配置",
            )
        except DataError as exc:
            return result_error(
                "analysis_failed",
                exc.message,
                "检查数据（空表/标签取值/列类型）后重试",
            )

        # ---------- 落盘 ----------
        selection_dir = layout.subdir("feature_selection")
        refine_path = selection_dir / "refine.json"
        selected_path = selection_dir / "selected_features.txt"

        summary = result.summary()
        gain_top: List[Dict[str, Any]] = []
        if result.gain_table is not None and len(result.gain_table):
            top = result.gain_table.sort_values("total_gain", ascending=False).head(100)
            gain_top = [
                {
                    "feature": str(index),
                    "total_gain": float(row["total_gain"]),
                    "retained": bool(row.get("retained", False)),
                }
                for index, row in top.iterrows()
            ]
        payload = {
            "schema": REFINE_SCHEMA,
            "version": __version__,
            "generated_at": now_iso(),
            "method": result.method,
            "target": target,
            "task": task,
            "candidate_source": candidate_source,
            "candidates": list(candidates),
            "n_candidates": len(candidates),
            "params": params,
            "refine_config": {
                "threshold": refine_cfg.threshold,
                "max_rounds": refine_cfg.max_rounds,
                "target_range": list(refine_cfg.target_range),
                "max_features": refine_cfg.max_features,
                "threshold_relax_ladder": list(refine_cfg.threshold_relax_ladder),
                "force_keep": list(refine_cfg.force_keep),
                "force_drop": list(refine_cfg.force_drop),
                "seed": refine_cfg.seed,
            },
            "split": {
                **split_info,
                "train_rows": n_train,
                "val_rows": n_val,
            },
            "summary": summary,
            "features": list(result.features),
            "n_selected": result.n_features,
            "gain_top": gain_top,
            "warnings": warnings,
        }
        atomic_write_json(refine_path, to_jsonable(payload))
        atomic_write_text(selected_path, "\n".join(result.features) + "\n")

        if result.disposition != "converged":
            warnings.append(f"精筛处置：{result.disposition}；{result.message}")

        def _mutate(manifest_doc: Dict[str, Any]) -> None:
            stage_done(
                manifest_doc,
                "feature_selection",
                done=True,
                path="feature_selection/",
                summary=f"精筛 {len(candidates)} → {result.n_features} 个特征（{result.disposition}）",
            )

        update_manifest(layout.project_dir, _mutate)

        data = {
            "project_dir": layout.project_dir.as_posix(),
            "method": result.method,
            "candidate_source": candidate_source,
            "n_candidates": len(candidates),
            "n_selected": result.n_features,
            "features": list(result.features),
            "disposition": result.disposition,
            "message": result.message,
            "threshold": result.threshold,
            "target_range": list(refine_cfg.target_range),
            "attempts": result.attempts,
            "split": payload["split"],
            "gain_shape": summary.get("gain_shape"),
            "selected_features_path": layout.rel(selected_path),
        }
        return result_ok(
            data,
            artifacts=relative_paths(layout, [refine_path, selected_path]),
            warnings=warnings
            + [
                f"已写入 {layout.rel(refine_path)} 与 {layout.rel(selected_path)}",
            ],
        )
    except ConfigError as exc:
        return result_error(
            "config_invalid",
            exc.message,
            "按 docs/preprocess-split.md 检查 config；或先跑 feature_prescreen",
        )
    except Exception as exc:  # noqa: BLE001
        return result_from_exception(exc)
