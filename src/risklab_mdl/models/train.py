"""模型层训练装配：缺失填充 → 适配器 fit → 模型包（schema 2.0）。

★ 与 ``core.train`` 的关系
-------------------------
``core.train`` 是**旧版单算法**（lgb）的既有实现，输出 schema 1.0 模型包。
本模块是「5 算法统一装配」，输出 schema 2.0 模型包（含 task / classes /
bad_class / preprocess_spec 引用）。两者共用 ``core.package`` 的打分链路 ——
1.0 包仍可加载与打分（兼容性由 ``tests/test_package_v2.py`` 覆盖）。

★ 缺失值口径（必须可见）
-----------------------
``xgb`` / ``lgb`` 原生支持 NaN ⇒ **不填**（填了反而抹掉缺失信息）。
``dt`` / ``gbdt`` / ``lr`` 不接受 NaN ⇒ 用 **train 的中位数**填充，
并：① 把填充值写进模型包 ``fill_values``（推理期用同一批值，绝不用推理数据的中位数）；
② 把「填了哪几列、各填多少个」写进 ``warnings``。

⚠ 为什么不能用全量中位数：那会把 val/test 的分布信息泄漏进训练链路，
且推理期根本拿不到同样的中位数 —— 是「线上分数与离线指标不一致」的经典来源。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.errors import ModelError
from ..core.metrics import ScoreScale
from ..core.package import ModelPackage
from .adapters import build_adapter
from .adapters.base import BaseAdapter
from .catalog import default_params

__all__ = [
    "FitOutcome",
    "DEFAULT_NUM_BOOST_ROUND",
    "DEFAULT_EARLY_STOPPING_ROUNDS",
    "fill_for_estimator",
    "fit_model",
]

#: 固定轮数兜底（各算法在 catalog 里有自己的 n_estimators 默认值）。
DEFAULT_NUM_BOOST_ROUND = 300
DEFAULT_EARLY_STOPPING_ROUNDS = 50


@dataclass
class FitOutcome:
    """一次训练的全部结果（模型包 + 重要性 + 诊断信息）。"""

    adapter: BaseAdapter
    package: ModelPackage
    importance: Optional[pd.DataFrame]
    params: Dict[str, Any]
    n_train: int
    n_val: int
    n_features: int
    best_iteration: Optional[int]
    used_early_stopping: bool
    fit_seconds: float
    warnings: List[str] = field(default_factory=list)
    fill_values: Dict[str, float] = field(default_factory=dict)


def fill_for_estimator(
    X_train: pd.DataFrame,
    others: Mapping[str, pd.DataFrame],
    *,
    native_nan: bool,
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, float], int]:
    """按需填充缺失（只用 **train** 的中位数）。

    Returns
    -------
    (X_train, others, fill_values, n_filled_cells)
    """
    if native_nan:
        return X_train, dict(others), {}, 0
    medians = X_train.median(numeric_only=True)
    fill_values = {
        str(col): float(value)
        for col, value in medians.items()
        if pd.notna(value) and bool(X_train[col].isna().any())
    }
    if not fill_values:
        # 仍然返回：即使当前无缺失，也把「无填充」这个事实留给调用方记录
        return X_train, dict(others), {}, 0
    n_filled = 0
    train = X_train.copy()
    for col, value in fill_values.items():
        n_filled += int(train[col].isna().sum())
        train[col] = train[col].fillna(value)
    filled_others: Dict[str, pd.DataFrame] = {}
    for name, frame in others.items():
        block = frame.copy()
        for col, value in fill_values.items():
            if col in block.columns:
                block[col] = block[col].fillna(value)
        filled_others[name] = block
    return train, filled_others, fill_values, n_filled


def fit_model(
    *,
    algorithm: str,
    task: str,
    features: Sequence[str],
    X_train: pd.DataFrame,
    y_train: Sequence[int],
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[Sequence[int]] = None,
    num_class: int = 2,
    classes: Optional[Sequence[Any]] = None,
    bad_class: Any = None,
    params: Optional[Mapping[str, Any]] = None,
    seed: int = 42,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    score_scale: Optional[ScoreScale] = None,
    preprocess_spec: Optional[Mapping[str, Any]] = None,
    label_spec_digest: str = "",
    extra_warnings: Optional[Sequence[str]] = None,
) -> FitOutcome:
    """训练一个模型并打包（**不落盘**，落盘见 :mod:`risklab_mdl.runs.store`）。"""
    warnings: List[str] = list(extra_warnings or [])
    feature_list = [str(c) for c in features]
    if not feature_list:
        raise ModelError("入模特征为空，无法训练")

    adapter = build_adapter(
        algorithm,
        task,
        params,
        seed=seed,
        num_class=num_class,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )

    X_train = X_train.loc[:, feature_list]
    use_val = X_val is not None and y_val is not None and len(X_val) > 0
    X_val_used: Optional[pd.DataFrame] = None
    if use_val:
        X_val_used = X_val.loc[:, feature_list]

    X_train_filled, others, fill_values, n_filled = fill_for_estimator(
        X_train,
        {"val": X_val_used} if X_val_used is not None else {},
        native_nan=adapter.native_nan,
    )
    if X_val_used is not None:
        X_val_used = others["val"]
    if fill_values:
        warnings.append(
            f"{algorithm} 不接受缺失值：已用 **train 中位数**填充 "
            f"{len(fill_values)} 列（共 {n_filled} 个缺失单元格）；填充值随模型包落盘，"
            f"推理期用同一批值（不用推理数据的中位数）"
        )

    started = time.perf_counter()
    adapter.fit(X_train_filled, np.asarray(y_train).astype(int), X_val_used, y_val)
    fit_seconds = time.perf_counter() - started
    for item in adapter.warnings:
        warnings.append(item.message)
    if use_val and adapter.used_early_stopping:
        if adapter.best_iteration is None:
            warnings.append(
                f"{algorithm} 启用了早停但拿不到 best_iteration（该版本未暴露），"
                f"已按最终轮数训练"
            )
        elif early_stopping_rounds and adapter.best_iteration >= _max_round(adapter) - 1:
            warnings.append(
                f"{algorithm} 的 best_iteration={adapter.best_iteration} 已达上限 "
                f"{_max_round(adapter)} —— 早停可能没有真正生效（模型可能欠拟合）"
            )
    elif not use_val:
        warnings.append(
            f"{algorithm} 没有 val 子集：按固定轮数训练（早停不启用）"
        )

    importance: Optional[pd.DataFrame] = None
    try:
        importance = adapter.feature_importance()
    except ModelError as exc:  # 重要性算不出来不阻断训练，但必须可见
        warnings.append(f"特征重要性不可用：{exc.message}")

    package = ModelPackage(
        model=adapter,
        algorithm=algorithm,  # type: ignore[arg-type]
        features=list(feature_list),
        params=dict(adapter.params),
        best_iteration=adapter.best_iteration,
        fill_values=fill_values,
        score_scale=score_scale or ScoreScale(),
        task=task,
        num_class=int(num_class),
        classes=list(classes or list(range(num_class))),
        bad_class=bad_class,
        algorithm_family=_family(algorithm),
        preprocess_spec=dict(preprocess_spec or {}),
        label_spec_digest=label_spec_digest,
        training={"fit_seconds": round(fit_seconds, 3), "n_train": int(len(X_train_filled)),
                  "n_val": int(len(X_val_used)) if X_val_used is not None else 0},
    )
    return FitOutcome(
        adapter=adapter,
        package=package,
        importance=importance,
        params=dict(adapter.params),
        n_train=int(len(X_train_filled)),
        n_val=int(len(X_val_used)) if X_val_used is not None else 0,
        n_features=len(feature_list),
        best_iteration=adapter.best_iteration,
        used_early_stopping=bool(adapter.used_early_stopping),
        fit_seconds=round(fit_seconds, 3),
        warnings=warnings,
        fill_values=fill_values,
    )


def _max_round(adapter: BaseAdapter) -> int:
    for key in ("n_estimators", "num_boost_round"):
        value = adapter.params.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return int(adapter.num_boost_round)


def _family(algorithm: str) -> str:
    return {
        "dt": "sklearn",
        "lr": "sklearn",
        "gbdt": "sklearn",
        "xgb": "xgboost",
        "lgb": "lightgbm",
    }.get(algorithm, "unknown")


def merged_default_params(algorithm: str, task: str) -> Dict[str, Any]:
    """catalog 默认参数（工具层回显用）。"""
    return default_params(algorithm, task)
