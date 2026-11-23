"""模型家族工具（契约 §3.4）：``model_catalog`` / ``model_train`` /
``model_tune`` / ``model_evaluate`` / ``model_confirm``。

★ 五个工具共享同一条装配链
--------------------------
``_prepare()``：解析项目 → 取数（processed → 原始 CSV）→ 定特征 → 标签编码 →
定划分 → 产出 ``_Prepared``。``model_train`` 与 ``model_tune`` 都从它出发，
``model_evaluate`` / ``model_confirm`` 则在**已落盘的 run** 上复现同一批特征与划分。

这样「训练时用哪些行/哪些列」与「评价时用哪些行/哪些列」只有一处定义 ——
两处各算一次是「离线指标很漂亮、复核时对不上」的经典成因。

★ 失败必须可见
--------------
* 算法×任务不匹配 → ``task_algorithm_mismatch``（不降级到别的算法）；
* 缺 xgboost / lightgbm → ``dependency_missing``（带 pip 命令）；
* 非数值列 / 缺失值 → 要么显式报错（用户点名要的列），要么进 ``warnings``
  （自动选择时排除该列、LR 系用 train 中位数填充），**绝不静默丢弃或补 0**；
* 指标算不出来（子集只有一类）→ ``status="skipped"`` + 原因，不给 0 冒充满分。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import __version__
from ..artifacts.layout import resolve_project
from ..artifacts.manifest import ManifestError, now_iso, read_manifest, update_manifest
from ..core.errors import ConfigError, DataError, ModelError
from ..core.package import ModelPackage, predict_proba_matrix
from ..core.tuning import (
    ObjectiveWeights,
    SearchSpace,
    TuningConfig,
    tune_model,
    validate_tune_on,
    weights_from_metrics,
)
from ..data.loader import (
    normalize_target,
    read_csv_auto,
    resolve_task,
    to_jsonable,
)
from ..models import catalog as cat
from ..models.data import (
    ModelFrame,
    available_time_column,
    load_model_frame,
    resolve_features,
    resolve_masks,
)
from ..models.evaluate import (
    DEFAULT_LIFT_PCT,
    evaluate_subsets,
    ks_table,
    lift_table,
    monthly_psi,
)
from ..models.labels import LabelEncoding, encode_labels
from ..models.train import (
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_NUM_BOOST_ROUND,
    FitOutcome,
    fit_model,
)
from ..platform.fs import atomic_write_json, atomic_write_text
from ..runs import (
    read_run,
    load_run_package,
    run_file,
    run_metrics_summary,
    update_run,
    write_evaluation,
    write_run,
)
from ..protocol import (
    result_error,
    result_from_exception,
    result_needs_input,
    result_ok,
    target_question,
)

__all__ = [
    "DEFAULT_THRESHOLDS",
    "model_catalog",
    "model_confirm",
    "model_evaluate",
    "model_train",
    "model_tune",
]

METRICS_SCHEMA = "risklab-mdl.metrics/1"
CONFIRM_SCHEMA = "risklab-mdl.confirm/1"
TUNING_SCHEMA = "risklab-mdl.tuning/1"

#: 确认闸门的默认阈值（契约 §3.4：KS≥0.25、PSI≤0.10）。
DEFAULT_THRESHOLDS: Dict[str, float] = {"ks_min": 0.25, "psi_max": 0.10}

#: 调参目标函数的「衰减（train − val）」惩罚阈值。
#: 与 ``core.tuning.TuningConfig.gap_threshold`` 同一口径；预览文案里的数值由它生成，
#: 避免「文档写 0.05、实现按 0 罚」这类文案与实现漂移。
DEFAULT_GAP_THRESHOLD: float = 0.05


# --------------------------------------------------------------------------- #
# 错误映射 / 小工具
# --------------------------------------------------------------------------- #
def _model_error_payload(exc: ModelError, *, hint: str = "") -> Dict[str, Any]:
    """``ModelError`` → 契约错误码（``task_algorithm_mismatch`` / ``dependency_missing``）。"""
    detail = getattr(exc, "detail", None) or {}
    code = None
    if isinstance(detail, Mapping):
        code = detail.get("tool_code")
    if code == "task_algorithm_mismatch":
        return result_error(
            "task_algorithm_mismatch",
            exc.message,
            hint or "按 model_catalog 选择算法×任务组合（5 算法 × 2 任务共 10 格）",
        )
    if isinstance(detail, Mapping) and detail.get("dependency"):
        return result_error(
            "dependency_missing",
            exc.message,
            f"执行：{detail.get('install', 'pip install ...')} 后重试",
        )
    return result_error(
        "analysis_failed",
        exc.message,
        hint or "检查算法参数、数据口径与依赖安装情况",
    )


def _normalize_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise ConfigError(f"config 必须是映射，收到 {type(config).__name__}")
    return dict(config)


def _target_of(manifest: Mapping[str, Any], cfg: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    return normalize_target(cfg.get("target")) or normalize_target(
        (manifest.get("project") or {}).get("target")
    )


def _csv_path(layout, manifest: Mapping[str, Any]) -> Optional[Path]:
    if layout.csv_path is not None:
        return Path(layout.csv_path)
    raw = (manifest.get("project") or {}).get("csv_path")
    return Path(str(raw)) if raw else None


def _head_frame(layout, manifest: Mapping[str, Any]) -> pd.DataFrame:
    csv = _csv_path(layout, manifest)
    if csv is None or not csv.is_file():
        return pd.DataFrame()
    return read_csv_auto(csv, nrows=200).frame


def _needs_target_payload(layout, manifest: Mapping[str, Any], reason: str) -> Dict[str, Any]:
    from ..data.loader import infer_roles

    head = _head_frame(layout, manifest)
    if head.empty:
        return result_error(
            "manifest_schema_invalid",
            "无法读取原始 CSV，无法确认目标字段",
            "重新执行 project_init / dataset_load",
        )
    inferred, _ = infer_roles(head)
    question, context = target_question(
        head, inferred["target_candidates"], reason=reason
    )
    return result_needs_input(question, context)


# --------------------------------------------------------------------------- #
# 共享装配链
# --------------------------------------------------------------------------- #
@dataclass
class _Prepared:
    layout: Any
    manifest: Dict[str, Any]
    target: Dict[str, Any]
    algorithm: str
    task: str
    model_frame: ModelFrame
    frame: pd.DataFrame
    features: List[str]
    feature_source: str
    encoding: LabelEncoding
    codes: pd.Series
    masks: Dict[str, pd.Series]
    split_info: Dict[str, Any]
    seed: int
    num_boost_round: int
    early_stopping_rounds: int
    lift_pct: float
    warnings: List[str] = field(default_factory=list)

    def subset(self, name: str) -> Tuple[pd.DataFrame, np.ndarray]:
        mask = self.masks[name]
        X = self.frame.loc[mask, self.features]
        y = self.codes.loc[mask].astype(int).to_numpy()
        return X, y

    def counts(self) -> Dict[str, int]:
        return {name: int(mask.sum()) for name, mask in self.masks.items()}


def _prepare(
    project_dir: str, cfg: Mapping[str, Any], *, require_algorithm: bool = True
) -> Tuple[Optional[_Prepared], Optional[Dict[str, Any]]]:
    """装配数据/特征/标签/划分。返回 ``(prepared, error_payload)`` 二者之一。"""
    layout = resolve_project(project_dir)
    manifest = read_manifest(layout.project_dir)
    target = _target_of(manifest, cfg)
    if target is None:
        return None, _needs_target_payload(
            layout, manifest, "尚未确认目标字段（config.target.column）"
        )

    algorithm = str(cfg.get("algorithm") or "").strip().lower()
    if require_algorithm and not algorithm:
        return None, result_error(
            "config_invalid",
            "config.algorithm 必填（dt / lr / gbdt / xgb / lgb）",
            "用 model_catalog 查 10 格能力表与默认参数",
        )

    model_frame = load_model_frame(layout, manifest, cfg)
    frame = model_frame.frame
    warnings: List[str] = list(model_frame.warnings)

    if target["column"] not in frame.columns:
        return None, result_error(
            "config_invalid",
            f"目标列 {target['column']!r} 不在当前数据中"
            f"（数据来源：{model_frame.source}）",
            "核对 config.target.column，或重跑 preprocess_apply 让目标列保留在产物里",
        )

    requested_task = str(cfg.get("task") or target.get("task") or "").strip().lower()
    if requested_task and requested_task not in cat.TASKS:
        return None, result_error(
            "task_algorithm_mismatch",
            f"任务类型不可用：task={requested_task!r}（只支持 {list(cat.TASKS)}）。"
            f"不回退到按标签自动推断 —— 那会把「配置写错」变成「静默换口径」",
            "把 config.task / project.target.task 设为 binary 或 multiclass",
            data={"task": requested_task, "supported_tasks": list(cat.TASKS)},
        )
    task = resolve_task(frame[target["column"]], requested_task or None)
    target_record = dict(target)
    target_record["task"] = task

    if algorithm and not cat.supports(algorithm, task):
        payload = result_error(
            "task_algorithm_mismatch",
            f"算法与任务组合不可用：algorithm={algorithm!r}, task={task!r}。"
            f"可用算法 {list(cat.ALGORITHMS)}；可用任务 {list(cat.TASKS)}",
            "按 model_catalog 换算法或换任务类型（5 算法 × 2 任务共 10 格）",
        )
        payload["data"] = {
            "algorithm": algorithm,
            "task": task,
            "supported_algorithms": list(cat.ALGORITHMS),
        }
        return None, payload

    try:
        features, feature_source, feature_warnings, non_numeric = resolve_features(
            frame, layout, manifest, cfg
        )
    except ConfigError as exc:
        return None, result_error(
            "config_invalid", exc.message, "按 docs/model-layer.md 的取数口径检查 config"
        )
    warnings.extend(feature_warnings)

    try:
        codes, encoding = encode_labels(
            frame[target["column"]], task=task, labels=target.get("labels")
        )
    except DataError as exc:
        return None, result_error(
            "analysis_failed", exc.message, "检查 config.target（列/任务/标签映射）"
        )
    warnings.extend(encoding.warnings)
    labelled = codes.notna()
    if int(labelled.sum()) < 30:
        return None, result_error(
            "features_empty",
            f"有效标注样本只有 {int(labelled.sum())} 行，不足以训练模型",
            "检查目标列是否大多缺失，或换一份数据",
        )
    if int((~labelled).sum()):
        warnings.append(
            f"{int((~labelled).sum())} 行因标签缺失/未知被剔除（不入模、不进指标）"
        )

    seed = int(cfg.get("seed") or 42)
    masks, split_info, split_warnings = resolve_masks(
        layout,
        frame,
        y_codes=codes,
        target_column=target["column"],
        seed=seed,
        config=cfg,
    )
    warnings.extend(split_warnings)
    masks = {name: (mask & labelled) for name, mask in masks.items()}

    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    if counts["train"] == 0:
        return None, result_error(
            "config_invalid",
            "训练子集为空（0 行）—— 划分与标签对齐后没有任何训练样本",
            "检查 split/split.json 的划分条件",
        )
    train_codes = codes.loc[masks["train"]]
    if train_codes.nunique() < 2:
        return None, result_error(
            "analysis_failed",
            f"训练子集只有 {int(train_codes.nunique())} 个类别，无法训练分类模型",
            "检查划分是否把某一类全分到了 val/test",
        )

    prepared = _Prepared(
        layout=layout,
        manifest=manifest,
        target=target_record,
        algorithm=algorithm,
        task=task,
        model_frame=model_frame,
        frame=frame,
        features=features,
        feature_source=feature_source,
        encoding=encoding,
        codes=codes,
        masks=masks,
        split_info=split_info,
        seed=seed,
        num_boost_round=int(cfg.get("num_boost_round") or DEFAULT_NUM_BOOST_ROUND),
        early_stopping_rounds=int(
            cfg.get("early_stopping_rounds", DEFAULT_EARLY_STOPPING_ROUNDS)
        ),
        lift_pct=float(cfg.get("lift_pct") or DEFAULT_LIFT_PCT),
        warnings=warnings,
    )
    return prepared, None


def _matrix_for(package: ModelPackage, X: pd.DataFrame) -> Tuple[np.ndarray, List[Any]]:
    return predict_proba_matrix(X, package)


def _fit_and_measure(
    prepared: _Prepared,
    *,
    params: Optional[Mapping[str, Any]] = None,
    extra_warnings: Optional[Sequence[str]] = None,
) -> Tuple[FitOutcome, Dict[str, Any], List[str]]:
    """训练 + 产出 train/val/test 指标（训练与调参共用）。"""
    X_train, y_train = prepared.subset("train")
    X_val, y_val = prepared.subset("val")
    warnings: List[str] = list(prepared.warnings) + list(extra_warnings or [])
    outcome = fit_model(
        algorithm=prepared.algorithm,
        task=prepared.task,
        features=prepared.features,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val if len(X_val) else None,
        y_val=y_val if len(y_val) else None,
        num_class=prepared.encoding.num_class,
        classes=list(prepared.encoding.classes),
        bad_class=prepared.encoding.bad_class,
        params=params,
        seed=prepared.seed,
        num_boost_round=prepared.num_boost_round,
        early_stopping_rounds=prepared.early_stopping_rounds,
        preprocess_spec={
            "source": prepared.model_frame.source,
            "processed_path": prepared.model_frame.processed_path,
        },
        label_spec_digest=_label_digest(prepared.encoding),
    )
    metrics, metric_warnings = _measure_subsets(prepared, outcome.package)
    warnings.extend(outcome.warnings)
    warnings.extend(metric_warnings)
    outcome.package.metrics = run_metrics_summary(metrics)
    return outcome, metrics, warnings


def _label_digest(encoding: LabelEncoding) -> str:
    import hashlib

    payload = repr(
        (encoding.task, list(encoding.classes), str(encoding.bad_class))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _measure_subsets(
    prepared: _Prepared, package: ModelPackage
) -> Tuple[Dict[str, Any], List[str]]:
    subsets: Dict[str, Tuple[Any, Any]] = {}
    for name in ("train", "val", "test"):
        X, y = prepared.subset(name)
        if len(X) == 0:
            subsets[name] = (None, None)
            continue
        matrix, _classes = _matrix_for(package, X)
        subsets[name] = (y, matrix)
    metrics = evaluate_subsets(
        task=prepared.task,
        subsets={k: v for k, v in subsets.items()},
        classes=list(prepared.encoding.classes),
        bad_index=prepared.encoding.bad_index,
        lift_pct=prepared.lift_pct,
    )
    warnings: List[str] = []
    for name, block in metrics.items():
        if block.get("status") == "skipped":
            warnings.append(f"{name} 子集指标跳过：{block.get('reason')}")
        if prepared.task == "multiclass":
            for note in block.get("notes") or []:
                warnings.append(f"{name}: {note}")
    return metrics, warnings


def _metrics_document(
    prepared: _Prepared,
    metrics: Mapping[str, Any],
    *,
    run_id: str = "",
    psi: Optional[Mapping[str, Any]] = None,
    warnings: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    doc: Dict[str, Any] = {
        "schema": METRICS_SCHEMA,
        "version": __version__,
        "run_id": run_id,
        "generated_at": now_iso(),
        "algorithm": prepared.algorithm,
        "task": prepared.task,
        "classes": list(prepared.encoding.classes),
        "bad_class": prepared.encoding.bad_class,
        "bad_index": int(prepared.encoding.bad_index),
        "n_features": len(prepared.features),
        "split": prepared.split_info,
        "sample_sizes": prepared.counts(),
        "lift_pct": float(prepared.lift_pct),
        "psi": dict(psi)
        if psi is not None
        else {"enabled": False, "reason": "未运行 model_evaluate：PSI 未计算"},
        "warnings": list(warnings or []),
    }
    for name in ("train", "val", "test"):
        doc[name] = metrics.get(name, {"status": "skipped", "reason": "子集为空"})
    if prepared.task == "multiclass":
        doc["multi"] = {
            name: {
                "per_class": doc[name].get("per_class", []),
                "macro": doc[name].get("macro", {}),
                "weighted": doc[name].get("weighted", {}),
                "bad_class": doc[name].get("bad_class", {}),
            }
            for name in ("train", "val", "test")
        }
    return doc


def _flat_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in ("train", "val", "test"):
        block = metrics.get(name) or {}
        out[name] = {
            key: block.get(key)
            for key in ("ks", "auc", "gini", "lift", "n", "bad", "good", "status", "reason")
            if key in block
        }
    return out


# --------------------------------------------------------------------------- #
# model_catalog
# --------------------------------------------------------------------------- #
def model_catalog() -> Dict[str, Any]:
    """算法 × 任务能力表（5 × 2 = 10 格）+ 默认参数 + 默认调参空间 + 依赖状态。"""
    payload = cat.catalog_payload()
    available: Dict[str, Any] = {}
    warnings: List[str] = []
    import importlib.util

    for algorithm in cat.ALGORITHMS:
        dependency = cat.DEPENDENCIES.get(algorithm)
        if not dependency:
            available[algorithm] = {"installed": True, "dependency": None}
            continue
        installed = importlib.util.find_spec(dependency) is not None
        available[algorithm] = {"installed": installed, "dependency": dependency}
        if not installed:
            warnings.append(
                f"算法 {algorithm} 需要 {dependency}（未安装）："
                f"执行 pip install {dependency} 后再用，否则 model_train 会返回 dependency_missing"
            )
    return result_ok(
        {"catalog": payload, "available": available},
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# model_train
# --------------------------------------------------------------------------- #
def model_train(
    project_dir: str, config: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """训练一个 run（DT/LR/GBDT/XGB/LGB × binary/multiclass）。"""
    try:
        cfg = _normalize_config(config)
        prepared, error = _prepare(project_dir, cfg)
        if error is not None:
            return error
        assert prepared is not None

        try:
            outcome, metrics, warnings = _fit_and_measure(
                prepared, params=cfg.get("params")
            )
        except ModelError as exc:
            return _model_error_payload(exc)

        run_config = {
            "target": prepared.target,
            "algorithm": prepared.algorithm,
            "task": prepared.task,
            "features": list(prepared.features),
            "feature_source": prepared.feature_source,
            "data_source": prepared.model_frame.source,
            "processed_path": prepared.model_frame.processed_path,
            "split": prepared.split_info,
            "seed": prepared.seed,
            "num_boost_round": prepared.num_boost_round,
            "early_stopping_rounds": prepared.early_stopping_rounds,
            "lift_pct": prepared.lift_pct,
            "classes": list(prepared.encoding.classes),
            "bad_class": prepared.encoding.bad_class,
            "sample_sizes": prepared.counts(),
            "input_config": to_jsonable(cfg),
        }
        run = write_run(
            prepared.layout,
            algorithm=prepared.algorithm,
            task=prepared.task,
            config=run_config,
            params=outcome.params,
            metrics=_metrics_document(
                prepared, metrics, warnings=warnings
            ),
            package=outcome.package,
            importance=outcome.importance,
            status="trained",
            summary_text=(
                f"{prepared.algorithm}/{prepared.task}："
                f"train {prepared.counts()['train']} 行 / val {prepared.counts()['val']} 行 / "
                f"{len(prepared.features)} 特征"
            ),
        )
        run_id = run["run_id"]
        # metrics.json 需要回填 run_id（先写后补：run_id 由目录分配决定）
        metrics_doc = _metrics_document(prepared, metrics, run_id=run_id, warnings=warnings)
        atomic_write_json(run["run_dir"] / "metrics.json", to_jsonable(metrics_doc))

        warnings.append(f"已写入 run {run_id}（{prepared.layout.rel(run['run_dir'])}）")
        data = {
            "run_id": run_id,
            "algorithm": prepared.algorithm,
            "task": prepared.task,
            "metrics": _flat_metrics(metrics),
            "multi": metrics_doc.get("multi"),
            "classes": list(prepared.encoding.classes),
            "bad_class": prepared.encoding.bad_class,
            "n_features": len(prepared.features),
            "features": list(prepared.features)[:50],
            "feature_source": prepared.feature_source,
            "data_source": prepared.model_frame.source,
            "sample_sizes": prepared.counts(),
            "split": prepared.split_info,
            "params": to_jsonable(outcome.params),
            "best_iteration": outcome.best_iteration,
            "used_early_stopping": outcome.used_early_stopping,
            "fit_seconds": outcome.fit_seconds,
            "manifest_entry": to_jsonable(run["entry"]),
        }
        return result_ok(
            data, artifacts=run["files"], warnings=warnings
        )
    except ConfigError as exc:
        return result_error(
            "config_invalid", exc.message, "按 docs/model-layer.md 检查 config"
        )
    except ManifestError as exc:
        return result_from_exception(exc)
    except ModelError as exc:
        return _model_error_payload(exc)
    except Exception as exc:  # noqa: BLE001
        return result_from_exception(exc)


# --------------------------------------------------------------------------- #
# model_tune
# --------------------------------------------------------------------------- #
def _resolve_space(
    prepared: _Prepared, cfg: Mapping[str, Any]
) -> Tuple[Dict[str, Dict[str, Any]], int, ObjectiveWeights, List[str]]:
    warnings: List[str] = []
    space = cat.resolve_space(prepared.algorithm, prepared.task, cfg.get("space"))
    trials_raw = cfg.get("trials")
    trials = int(trials_raw) if trials_raw is not None else 20
    if trials < 1 or trials > 500:
        raise ConfigError(f"config.trials 必须落在 [1, 500]，当前 {trials}")

    objective = cfg.get("objective")
    if objective is None:
        weights = ObjectiveWeights()
    elif isinstance(objective, Mapping):
        keys = {str(k) for k in objective}
        if keys & {"cv_ks", "cv_auc", "cv_lift", "negative_train_cv_gap"}:
            weights = weights_from_metrics(objective)
        else:
            allowed = {"ks", "auc", "lift", "gap"}
            unknown = keys - allowed
            if unknown:
                raise ConfigError(
                    f"config.objective 含未知权重项：{sorted(unknown)}。"
                    f"可用：{sorted(allowed)}（或指标全名 cv_ks/cv_auc/cv_lift/negative_train_cv_gap）"
                )
            weights = ObjectiveWeights(**{k: float(v) for k, v in objective.items()})
    else:
        raise ConfigError("config.objective 必须是映射（{ks, auc, lift, gap}）")

    if weights.lift > 0 and prepared.task == "multiclass":
        warnings.append(
            "多分类的 LIFT 取 macro（各类 OVR LIFT 的算术平均）—— 与二分类 LIFT 不同口径"
        )
    return space, trials, weights, warnings


def model_tune(
    project_dir: str,
    config: Optional[Mapping[str, Any]] = None,
    mode: str = "preview",
) -> Dict[str, Any]:
    """调参：``mode=preview`` 只回显空间；``mode=execute`` 落一个 run。"""
    try:
        cfg = _normalize_config(config)
        mode_raw = str(mode or "preview").strip().lower()
        if mode_raw not in ("preview", "execute"):
            return result_error(
                "config_invalid",
                f"mode 只允许 preview / execute，收到 {mode!r}",
                "mode=preview 看空间说明；mode=execute 真正调参",
            )

        tune_on = str(cfg.get("tune_on") or "val").strip().lower()
        try:
            validate_tune_on(tune_on)
        except ConfigError as exc:
            return result_error(
                "config_invalid", exc.message, "tune_on 只允许 'val'（禁 OOT）"
            )
        if tune_on != "val":
            return result_error(
                "config_invalid",
                f"tune_on={tune_on!r} 不被允许：本工具只支持 'val'。"
                f"用 train 调参与用 OOT 调参同属「让被优化对象参与选择」——"
                f"val 是唯一「用完早停后仍未被选择过」的样本",
                "把 config.tune_on 设为 'val'（默认值）",
            )

        prepared, error = _prepare(project_dir, cfg)
        if error is not None:
            return error
        assert prepared is not None

        space, trials, weights, tune_warnings = _resolve_space(prepared, cfg)
        warnings = list(prepared.warnings) + tune_warnings

        space_rows = SearchSpace(space).to_frame()
        space_table = space_rows.to_dict(orient="records")
        base_payload = {
            "mode": mode_raw,
            "algorithm": prepared.algorithm,
            "task": prepared.task,
            "classes": list(prepared.encoding.classes),
            "bad_class": prepared.encoding.bad_class,
            "space": to_jsonable(space),
            "space_table": to_jsonable(space_table),
            "expected_trials": int(trials),
            "objective": {
                "weights": weights.to_dict(),
                "gap_threshold": DEFAULT_GAP_THRESHOLD,
                "formula": (
                    "ks*w_ks + auc*w_auc + lift*w_lift − gap*w_gap"
                    f"（gap = max(0, train_auc − val_auc)；只有超过 "
                    f"{DEFAULT_GAP_THRESHOLD:g} 才惩罚，阈值以内不扣分）"
                ),
            },
            "tune_on": tune_on,
            "seed": prepared.seed,
            "sample_sizes": prepared.counts(),
            "n_features": len(prepared.features),
            "notes": [
                "preview 与 execute 使用同一份空间解析（catalog.resolve_space）",
                "tune_on 只允许 val；val 只承担早停与参数选择，test 不参与",
            ],
        }

        if mode_raw == "preview":
            warnings.append("mode=preview：未训练任何模型（只回显搜索空间与目标函数）")
            return result_ok(base_payload, warnings=warnings)

        # ---------- execute ----------
        tuning_cfg = TuningConfig(
            space=SearchSpace(space, seed=prepared.seed),
            weights=weights,
            n_iters=trials,
            tune_on="val",
            gap_threshold=DEFAULT_GAP_THRESHOLD,
            num_boost_round=prepared.num_boost_round,
            early_stopping_rounds=prepared.early_stopping_rounds,
            lift_pct=prepared.lift_pct,
            seed=prepared.seed,
        )

        def _trial_fn(params: Dict[str, Any], trial_index: int):
            outcome, metrics, _warn = _fit_and_measure(prepared, params=params)
            val = metrics.get("val") or {}
            train = metrics.get("train") or {}
            cv_auc = val.get("auc")
            cv_ks = val.get("ks")
            if cv_auc is None or cv_ks is None:
                raise ModelError(
                    f"试验 {trial_index}: val 子集的 KS/AUC 算不出来（{val.get('reason')}）"
                )
            train_auc = train.get("auc")
            metrics_out = {
                "cv_ks": float(cv_ks),
                "cv_auc": float(cv_auc),
                "train_auc": float(train_auc) if train_auc is not None else float("nan"),
                "train_cv_gap": (
                    float(train_auc) - float(cv_auc)
                    if train_auc is not None
                    else 0.0
                ),
            }
            if val.get("lift") is not None:
                metrics_out["cv_lift"] = float(val["lift"])
            return metrics_out, outcome.best_iteration

        try:
            result = tune_model(tuning_cfg, _trial_fn)
        except ModelError as exc:
            return _model_error_payload(exc, hint="减少 trials 或放宽搜索空间后重试")
        except ConfigError as exc:
            return result_error(
                "config_invalid", exc.message, "检查 config.space / config.objective"
            )

        outcome, metrics, fit_warnings = _fit_and_measure(
            prepared, params=result.best_params
        )
        warnings = fit_warnings + tune_warnings

        run_config = {
            "target": prepared.target,
            "algorithm": prepared.algorithm,
            "task": prepared.task,
            "features": list(prepared.features),
            "feature_source": prepared.feature_source,
            "data_source": prepared.model_frame.source,
            "processed_path": prepared.model_frame.processed_path,
            "split": prepared.split_info,
            "seed": prepared.seed,
            "num_boost_round": prepared.num_boost_round,
            "early_stopping_rounds": prepared.early_stopping_rounds,
            "lift_pct": prepared.lift_pct,
            "classes": list(prepared.encoding.classes),
            "bad_class": prepared.encoding.bad_class,
            "sample_sizes": prepared.counts(),
            "input_config": to_jsonable(cfg),
            "tuning": {
                "mode": "execute",
                "tune_on": "val",
                "n_iters": trials,
                "space": to_jsonable(space),
                "objective": weights.to_dict(),
                "best_params": to_jsonable(dict(result.best_params)),
                "best_target": None
                if np.isnan(result.best_target)
                else float(result.best_target),
                "best_metrics": to_jsonable(dict(result.best_metrics)),
                "n_trials": result.n_trials,
                "n_failed": result.n_failed,
                "aborted": result.aborted,
                "abort_reason": result.abort_reason,
            },
        }
        run = write_run(
            prepared.layout,
            algorithm=prepared.algorithm,
            task=prepared.task,
            config=run_config,
            params=outcome.params,
            metrics=_metrics_document(prepared, metrics, warnings=warnings),
            package=outcome.package,
            importance=outcome.importance,
            status="tuned",
            summary_text=(
                f"{prepared.algorithm}/{prepared.task} 调参 {result.n_trials} 次"
                f"（best target {result.best_target:.4f}）"
            ),
        )
        run_id = run["run_id"]
        files = run_file(prepared.layout, run_id)
        tuning_dir = files.ensure_tuning_dir()
        trials_frame = result.trials_frame()
        atomic_write_text(
            tuning_dir / "trials.csv",
            trials_frame.to_csv(index=False, lineterminator="\n"),
        )
        atomic_write_json(
            tuning_dir / "space.json",
            to_jsonable(
                {
                    "schema": TUNING_SCHEMA,
                    "algorithm": prepared.algorithm,
                    "task": prepared.task,
                    "space": space,
                    "objective": weights.to_dict(),
                    "trials": trials,
                    "tune_on": "val",
                }
            ),
        )
        metrics_doc = _metrics_document(
            prepared, metrics, run_id=run_id, warnings=warnings
        )
        atomic_write_json(files.metrics_path, to_jsonable(metrics_doc))

        warnings.append(f"已写入 run {run_id}（{prepared.layout.rel(run['run_dir'])}）")
        if result.aborted:
            warnings.append(f"调参提前中止：{result.abort_reason}")
        data = dict(base_payload)
        data.update(
            {
                "mode": "execute",
                "run_id": run_id,
                "best_params": to_jsonable(dict(result.best_params)),
                "best_target": None
                if np.isnan(result.best_target)
                else float(result.best_target),
                "best_metrics": to_jsonable(dict(result.best_metrics)),
                "n_trials": result.n_trials,
                "n_failed": result.n_failed,
                "aborted": result.aborted,
                "metrics": _flat_metrics(metrics),
                "fit_seconds": outcome.fit_seconds,
                "params": to_jsonable(outcome.params),
            }
        )
        artifacts = list(run["files"]) + [
            prepared.layout.rel(tuning_dir / "trials.csv"),
            prepared.layout.rel(tuning_dir / "space.json"),
        ]
        return result_ok(data, artifacts=artifacts, warnings=warnings)
    except ConfigError as exc:
        return result_error(
            "config_invalid", exc.message, "按 docs/model-layer.md 检查调参 config"
        )
    except ManifestError as exc:
        return result_from_exception(exc)
    except ModelError as exc:
        return _model_error_payload(exc)
    except Exception as exc:  # noqa: BLE001
        return result_from_exception(exc)


# --------------------------------------------------------------------------- #
# run 定位 / 评价
# --------------------------------------------------------------------------- #
def _latest_run_id(manifest: Mapping[str, Any]) -> Optional[str]:
    runs = manifest.get("runs") or []
    if not runs:
        return None
    return max(str(entry.get("run_id")) for entry in runs if entry.get("run_id"))


def _run_or_error(layout, manifest: Mapping[str, Any], run_id: Optional[str]):
    rid = str(run_id or "").strip()
    if not rid:
        rid = _latest_run_id(manifest) or ""
    if not rid:
        return None, result_error(
            "run_not_found",
            "项目里还没有任何 run：先 model_train 训练一个模型",
            "用 run_list 查看已有 run（当前为空）",
        )
    try:
        return read_run(layout, rid), None
    except ModelError as exc:
        return None, _model_error_payload(exc, hint="用 run_list 查可用 run")


def _rebuild_for_run(layout, manifest: Mapping[str, Any], run: Mapping[str, Any], cfg: Mapping[str, Any]):
    """按已落盘 run 的特征/标签口径重建数据与划分。"""
    config = run.get("config") or {}
    target = config.get("target") or (manifest.get("project") or {}).get("target")
    if not target or not target.get("column"):
        raise ConfigError("run 的 config.json 里没有目标列信息，无法重算评价指标")
    package = load_run_package(layout, run["run_id"])
    model_frame = load_model_frame(layout, manifest, cfg)
    frame = model_frame.frame
    if target["column"] not in frame.columns:
        raise ConfigError(
            f"目标列 {target['column']!r} 不在当前数据中（数据来源：{model_frame.source}）"
        )
    task = str(config.get("task") or package.task)
    codes, encoding = encode_labels(
        frame[target["column"]], task=task, labels=target.get("labels")
    )
    # 以模型包为准的类别顺序（保证概率列与训练期一致）
    if package.classes:
        mapping = {str(c): i for i, c in enumerate(package.classes)}
        codes = frame[target["column"]].map(lambda v: mapping.get(str(v), np.nan)).astype("float64")
        encoding = LabelEncoding(
            task=task,
            classes=list(package.classes),
            class_map=mapping,
            bad_class=package.bad_class,
            bad_index=package.bad_index(),
            n_rows=int(frame.shape[0]),
        )
    masks, split_info, split_warnings = resolve_masks(
        layout,
        frame,
        y_codes=codes,
        target_column=target["column"],
        seed=int(config.get("seed") or 42),
        config=cfg,
    )
    labelled = codes.notna()
    masks = {name: (mask & labelled) for name, mask in masks.items()}
    prepared = _Prepared(
        layout=layout,
        manifest=dict(manifest),
        target=dict(target),
        algorithm=str(config.get("algorithm") or package.algorithm),
        task=task,
        model_frame=model_frame,
        frame=frame,
        features=list(package.features),
        feature_source=str(config.get("feature_source") or "run config"),
        encoding=encoding,
        codes=codes,
        masks=masks,
        split_info=split_info,
        seed=int(config.get("seed") or 42),
        num_boost_round=int(config.get("num_boost_round") or DEFAULT_NUM_BOOST_ROUND),
        early_stopping_rounds=int(
            config.get("early_stopping_rounds", DEFAULT_EARLY_STOPPING_ROUNDS)
        ),
        lift_pct=float(config.get("lift_pct") or DEFAULT_LIFT_PCT),
        warnings=list(model_frame.warnings) + split_warnings,
    )
    return prepared, package


def _prob_column(matrix: np.ndarray, prepared: "_Prepared") -> np.ndarray:
    """取「坏类」概率列。

    ★ 不硬编码 ``matrix[:, 1]``：二分类的列顺序由 ``classes`` 决定，
      ``bad_index`` 才是唯一口径（见 models.labels / core.package.bad_index）。
    """
    mat = np.asarray(matrix, dtype=float)
    column = int(prepared.encoding.bad_index)
    if mat.ndim == 2 and mat.shape[1] > column:
        return mat[:, column]
    return mat.ravel()


def _binary_target(y: Any, bad_index: int) -> np.ndarray:
    """0/1 二值标签（``1`` = 坏类）。

    KS/LIFT 表与图**只吃这个口径**：多分类若直接把 ``0..k-1`` 编码传进去，
    只有「编码 1」的那类会被当坏样本，真正的 bad_class 整批算成好样本 ——
    表/图看着有数，但与 metrics.json 的 per_class/macro 完全错位。
    """
    return (np.asarray(y).ravel() == int(bad_index)).astype(int)


def _write_evaluation_charts(
    evaluation_dir: Path, y: np.ndarray, prob: np.ndarray, *, title: str
) -> Tuple[List[Path], List[str]]:
    """KS 曲线 + LIFT 图（Agg）。缺 matplotlib / 字体时**降级为仅出表**并告警。"""
    warnings: List[str] = []
    paths: List[Path] = []
    try:
        from ..charts.figures import setup_matplotlib

        pyplot, _font, font_warnings = setup_matplotlib()
        warnings.extend(font_warnings)
    except Exception as exc:  # noqa: BLE001 - 出图失败不阻断评价
        warnings.append(f"评价图未生成（{type(exc).__name__}: {exc}）：表已写出")
        return paths, warnings

    plt = pyplot

    try:
        table = ks_table(y, prob)
        if len(table):
            cum_bad = table["cum_bad_rate"].to_numpy(dtype=float)
            cum_good = table["cum_good_rate"].to_numpy(dtype=float)
            xs = np.arange(1, len(table) + 1)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(xs, cum_bad, marker="o", label="累计坏样本占比")
            ax.plot(xs, cum_good, marker="s", label="累计好样本占比")
            ax.set_xlabel("分箱（分数降序）")
            ax.set_ylabel("累计占比")
            ax.set_title(title)
            ax.legend()
            fig.tight_layout()
            path = evaluation_dir / "ks_curve.png"
            fig.savefig(path, dpi=120)
            plt.close(fig)
            paths.append(path)

        lift = lift_table(y, prob)
        if len(lift):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(lift["bin"].astype(str), lift["lift"].astype(float))
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
            ax.set_xlabel("分箱（分数降序）")
            ax.set_ylabel("LIFT")
            ax.set_title(f"{title} LIFT")
            fig.tight_layout()
            path = evaluation_dir / "lift_chart.png"
            fig.savefig(path, dpi=120)
            plt.close(fig)
            paths.append(path)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"评价图绘制失败（{type(exc).__name__}: {exc}）：表已写出")
    return paths, warnings


def _evaluate_run(
    layout,
    manifest: Mapping[str, Any],
    run_id: str,
    cfg: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """评价一个 run（model_evaluate 与 model_confirm 共用）。"""
    run, error = _run_or_error(layout, manifest, run_id)
    if error is not None:
        return None, error
    assert run is not None
    prepared, package = _rebuild_for_run(layout, manifest, run, cfg)
    warnings = list(prepared.warnings)

    subsets: Dict[str, Tuple[Any, Any]] = {}
    for name in ("train", "val", "test"):
        X, y = prepared.subset(name)
        if len(X) == 0:
            subsets[name] = (None, None)
            continue
        matrix, _classes = _matrix_for(package, X)
        subsets[name] = (y, matrix)
    metrics = evaluate_subsets(
        task=prepared.task,
        subsets=subsets,
        classes=list(prepared.encoding.classes),
        bad_index=prepared.encoding.bad_index,
        lift_pct=prepared.lift_pct,
    )
    for name, block in metrics.items():
        if block.get("status") == "skipped":
            warnings.append(f"{name} 子集指标跳过：{block.get('reason')}")

    # ---------- PSI ----------
    time_column, psi_reason = available_time_column(prepared.frame, manifest, cfg)
    psi_block: Dict[str, Any]
    tables: Dict[str, pd.DataFrame] = {}
    if not time_column:
        psi_block = {"enabled": False, "reason": psi_reason, "max_psi": None}
        warnings.append(psi_reason or "PSI 跳过")
    else:
        valid = prepared.codes.notna()
        X_all = prepared.frame.loc[valid, prepared.features]
        matrix, _ = _matrix_for(package, X_all)
        prob = _prob_column(matrix, prepared)
        psi_block = monthly_psi(
            prob,
            prepared.frame.loc[valid, time_column],
            base_months=cfg.get("psi_base_months"),
        )
        psi_block["time_column"] = time_column
        if psi_block.get("enabled"):
            tables["psi_by_month"] = pd.DataFrame(psi_block.get("by_month") or [])
            warnings.append(
                f"分数 PSI：基期 {psi_block.get('base_label')}，"
                f"最大 PSI {psi_block.get('max_psi')}（{psi_block.get('max_month')}）"
            )
        else:
            warnings.append(str(psi_block.get("reason") or "PSI 跳过"))

    # ---------- 评价表 ----------
    rows = []
    for name in ("train", "val", "test"):
        block = metrics.get(name) or {}
        rows.append(
            {
                "subset": name,
                "n": block.get("n"),
                "bad": block.get("bad"),
                "good": block.get("good"),
                "ks": block.get("ks"),
                "auc": block.get("auc"),
                "gini": block.get("gini"),
                "lift": block.get("lift"),
                "status": block.get("status"),
                "reason": block.get("reason"),
            }
        )
    tables["metrics_by_subset"] = pd.DataFrame(rows)
    if prepared.task == "multiclass":
        per_class_rows = []
        for name in ("train", "val", "test"):
            for record in (metrics.get(name) or {}).get("per_class") or []:
                per_class_rows.append({"subset": name, **record})
        tables["per_class_metrics"] = pd.DataFrame(per_class_rows)

    primary = "val" if (metrics.get("val") or {}).get("n") else "train"
    y_primary, matrix_primary = subsets.get(primary, (None, None))
    y_primary_bin: Optional[np.ndarray] = None
    prob_primary: Optional[np.ndarray] = None
    table_bad_label: Any = None
    if y_primary is not None and len(y_primary):
        prob_primary = _prob_column(matrix_primary, prepared)
        y_primary_bin = _binary_target(y_primary, prepared.encoding.bad_index)
        # 多分类要在表里注明「哪一类算坏」，否则表本身看不出 1 是谁
        table_bad_label = (
            prepared.encoding.bad_class if prepared.task == "multiclass" else None
        )
        tables["ks_table"] = ks_table(
            y_primary_bin, prob_primary, bad_label=table_bad_label
        )
        tables["lift_table"] = lift_table(
            y_primary_bin, prob_primary, bad_label=table_bad_label
        )
        if prepared.task == "multiclass":
            warnings.append(
                f"多分类 KS/LIFT 表与图以 bad_class={prepared.encoding.bad_class}"
                f"（编码 {int(prepared.encoding.bad_index)}）为坏样本、其余类合并为好样本"
                f" —— 与 metrics.json 的 per_class[{prepared.encoding.bad_class}] 同口径"
            )

    files = run_file(layout, run_id)
    written = write_evaluation(run_id, files, tables)
    chart_warnings: List[str] = []
    chart_paths: List[Path] = []
    if y_primary_bin is not None and prob_primary is not None:
        if np.unique(y_primary_bin).size >= 2:
            chart_title = f"{prepared.algorithm}/{prepared.task}（{primary}）"
            if table_bad_label is not None:
                chart_title = f"{chart_title}，坏类={table_bad_label}"
            chart_paths, chart_warnings = _write_evaluation_charts(
                files.evaluation_dir,
                y_primary_bin,
                prob_primary,
                title=chart_title,
            )
        else:
            chart_warnings.append(
                f"{primary} 子集里 bad_class="
                f"{prepared.encoding.bad_class} 的样本为 0：KS/LIFT 图跳过（表已写出）"
            )
    warnings.extend(chart_warnings)

    metrics_doc = _metrics_document(
        prepared,
        metrics,
        run_id=run_id,
        psi=psi_block,
        warnings=warnings,
    )
    if y_primary_bin is not None:
        metrics_doc["ks_lift_tables"] = {
            "subset": primary,
            "bad_class": table_bad_label,
            "bad_index": int(prepared.encoding.bad_index),
            "label_scheme": "y_bin = (y == bad_index)，1=坏 / 0=好",
        }
    atomic_write_json(files.metrics_path, to_jsonable(metrics_doc))
    update_run(
        layout,
        run_id,
        status="evaluated",
        metrics_patch=run_metrics_summary(metrics_doc),
        paths_patch={"evaluation": layout.rel(files.evaluation_dir)},
    )
    artifacts = [layout.rel(files.metrics_path)] + [layout.rel(p) for p in written]
    artifacts += [layout.rel(p) for p in chart_paths]
    return (
        {
            "run_id": run_id,
            "algorithm": prepared.algorithm,
            "task": prepared.task,
            "metrics": _flat_metrics(metrics),
            "multi": metrics_doc.get("multi"),
            "psi": psi_block,
            "classes": list(prepared.encoding.classes),
            "bad_class": prepared.encoding.bad_class,
            "sample_sizes": prepared.counts(),
            "split": prepared.split_info,
            "tables": {name: len(frame) for name, frame in tables.items()},
            "charts": [prepared.layout.rel(p) for p in chart_paths],
            "warnings": warnings,
            "artifacts": artifacts,
            "metrics_document": metrics_doc,
        },
        None,
    )


def model_evaluate(
    project_dir: str,
    run_id: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """train/val/test 指标（KS/AUC/Gini/LIFT；多分类 OVR+macro）+ 月度 PSI。"""
    try:
        cfg = _normalize_config(config)
        layout = resolve_project(project_dir)
        manifest = read_manifest(layout.project_dir)
        result, error = _evaluate_run(layout, manifest, str(run_id or ""), cfg)
        if error is not None:
            return error
        assert result is not None
        warnings = list(result.pop("warnings"))
        artifacts = list(result.pop("artifacts"))
        result.pop("metrics_document", None)
        warnings.append(
            f"已写入 {layout.rel(run_file(layout, result['run_id']).metrics_path)} 与 evaluation/ 表"
        )
        return result_ok(result, artifacts=artifacts, warnings=warnings)
    except ConfigError as exc:
        return result_error(
            "config_invalid", exc.message, "检查 run 产物与 config（time_column 等）"
        )
    except ManifestError as exc:
        return result_from_exception(exc)
    except ModelError as exc:
        return _model_error_payload(exc)
    except Exception as exc:  # noqa: BLE001
        return result_from_exception(exc)


# --------------------------------------------------------------------------- #
# model_confirm
# --------------------------------------------------------------------------- #
def _confirmation_thresholds(raw: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    for key in thresholds:
        if raw and raw.get(key) is not None:
            try:
                thresholds[key] = float(raw[key])
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"thresholds.{key} 必须是数字，收到 {raw[key]!r}"
                ) from exc
    if thresholds["ks_min"] < 0 or thresholds["ks_min"] > 1:
        raise ConfigError(f"thresholds.ks_min 必须落在 [0, 1]，当前 {thresholds['ks_min']}")
    if thresholds["psi_max"] < 0:
        raise ConfigError(f"thresholds.psi_max 不能为负，当前 {thresholds['psi_max']}")
    return thresholds


def model_confirm(
    project_dir: str,
    run_id: Optional[str] = None,
    thresholds: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """阈值校验通过后锁定最终模型（``final/`` + ``manifest.confirmed_run``）。"""
    try:
        layout = resolve_project(project_dir)
        manifest = read_manifest(layout.project_dir)
        try:
            th = _confirmation_thresholds(thresholds)
        except ConfigError as exc:
            return result_error(
                "config_invalid", exc.message, "thresholds={ks_min, psi_max}（默认 0.25 / 0.10）"
            )

        run, error = _run_or_error(layout, manifest, run_id)
        if error is not None:
            return error
        assert run is not None
        rid = run["run_id"]
        metrics_doc = run.get("metrics")
        warnings: List[str] = []
        artifacts: List[str] = []

        psi_block = (metrics_doc or {}).get("psi") if isinstance(metrics_doc, Mapping) else None
        need_eval = (
            not isinstance(metrics_doc, Mapping)
            or not isinstance(psi_block, Mapping)
            or (not psi_block.get("enabled") and "PSI 未计算" in str(psi_block.get("reason", "")))
        )
        if need_eval:
            evaluated, eval_error = _evaluate_run(layout, manifest, rid, {})
            if eval_error is not None:
                return eval_error
            assert evaluated is not None
            warnings.extend(evaluated["warnings"])
            artifacts.extend(evaluated["artifacts"])
            metrics_doc = evaluated["metrics_document"]
            warnings.append(
                "确认前已自动重算评价指标（原 metrics.json 缺少 PSI 结果）—— "
                "PSI 结果是闸门的一部分，不能跳过"
            )

        val = (metrics_doc or {}).get("val") or {}
        train = (metrics_doc or {}).get("train") or {}
        ks = val.get("ks")
        ks_source = "val"
        if ks is None:
            ks = train.get("ks")
            ks_source = "train" if ks is not None else None
        psi = None
        if isinstance(metrics_doc.get("psi"), Mapping) and metrics_doc["psi"].get("enabled"):
            psi = metrics_doc["psi"].get("max_psi")

        ks_min = th["ks_min"]
        psi_max = th["psi_max"]
        checks: List[str] = []
        failed: List[str] = []
        data: Dict[str, Any] = {
            "run_id": rid,
            "thresholds": th,
            "ks": ks,
            "ks_source": ks_source,
            "psi": psi,
            "ks_gap": None if ks is None else round(float(ks_min) - float(ks), 6),
            "psi_gap": None if psi is None else round(float(psi) - float(psi_max), 6),
            "sample_sizes": (metrics_doc or {}).get("sample_sizes"),
            "algorithm": (
                run.get("config", {}).get("algorithm")
                if isinstance(run.get("config"), Mapping)
                else None
            ),
            "task": (metrics_doc or {}).get("task"),
        }
        if ks is None:
            failed.append("ks")
            checks.append("KS 无法计算（该子集只有一类标签）：无法通过确认闸门")
        elif float(ks) < ks_min:
            failed.append("ks")
            checks.append(
                f"KS {float(ks):.4f} < 阈值 {ks_min:.4f}（差距 {float(ks_min) - float(ks):.4f}）"
            )
        else:
            checks.append(f"KS {float(ks):.4f} ≥ 阈值 {ks_min:.4f}（{ks_source}）")

        if psi is None:
            warnings.append(
                "PSI 未计算（无时间列或多于 1 个月不足）：本次只校验 KS —— "
                "不代表跨期稳定，上线前建议补齐时间列后重跑 model_evaluate"
            )
            checks.append("PSI 未计算：跳过该项校验")
        elif float(psi) > psi_max:
            failed.append("psi")
            checks.append(
                f"PSI {float(psi):.4f} > 阈值 {psi_max:.4f}"
                f"（超出 {float(psi) - float(psi_max):.4f}）"
            )
        else:
            checks.append(f"PSI {float(psi):.4f} ≤ 阈值 {psi_max:.4f}")

        n_val = int(val.get("n") or 0)
        if n_val and n_val < 100:
            warnings.append(
                f"val 样本量只有 {n_val} 行：确认结论的可靠性有限（阈值判断的统计波动大）"
            )

        data["checks"] = checks
        data["failed_checks"] = failed
        if failed:
            return result_error(
                "threshold_not_met",
                "确认闸门未通过：" + "；".join(checks),
                "调强模型（换算法/调参/加特征）后重跑 model_train，或按业务口径调整 thresholds",
                data=data,
                artifacts=artifacts,
                warnings=warnings,
            )

        # ---------- 写 final/ ----------
        files = run_file(layout, rid)
        if not files.model_path.is_file():
            return result_error(
                "run_not_found",
                f"run {rid} 缺少 model.pkl，无法确认为最终模型",
                "重跑 model_train",
            )
        final_dir = layout.final_dir
        final_dir.mkdir(parents=True, exist_ok=True)
        final_model = final_dir / "model.pkl"
        shutil.copyfile(files.model_path, final_model)
        confirm_doc = {
            "schema": CONFIRM_SCHEMA,
            "version": __version__,
            "confirmed_at": now_iso(),
            "run_id": rid,
            "algorithm": data["algorithm"],
            "task": data["task"],
            "thresholds": th,
            "metrics": {
                "ks": ks,
                "ks_source": ks_source,
                "psi": psi,
                **{
                    key: val.get(key)
                    for key in ("auc", "gini", "lift", "n", "bad", "good")
                    if key in val
                },
            },
            "checks": checks,
            "sample_sizes": data["sample_sizes"],
            "model_path": layout.rel(final_model),
            "source_run_model": layout.rel(files.model_path),
            "warnings": list(warnings),
        }
        confirm_path = final_dir / "confirm.json"
        atomic_write_json(confirm_path, to_jsonable(confirm_doc))

        def _mutate(manifest_doc: Dict[str, Any]) -> None:
            for entry in manifest_doc.get("runs") or []:
                if entry.get("run_id") == rid:
                    entry["status"] = "confirmed"
            manifest_doc["confirmed_run"] = rid

        update_manifest(layout.project_dir, _mutate)

        artifacts.extend([layout.rel(final_model), layout.rel(confirm_path)])
        warnings.append(f"已确认 run {rid}：final/model.pkl + final/confirm.json 已写入")
        data.update(
            {
                "confirmed": True,
                "confirmed_at": confirm_doc["confirmed_at"],
                "final_model": layout.rel(final_model),
                "confirm_path": layout.rel(confirm_path),
                "metrics": confirm_doc["metrics"],
            }
        )
        return result_ok(data, artifacts=artifacts, warnings=warnings)
    except ConfigError as exc:
        return result_error(
            "config_invalid", exc.message, "thresholds={ks_min, psi_max}"
        )
    except ManifestError as exc:
        return result_from_exception(exc)
    except ModelError as exc:
        return _model_error_payload(exc)
    except Exception as exc:  # noqa: BLE001
        return result_from_exception(exc)
