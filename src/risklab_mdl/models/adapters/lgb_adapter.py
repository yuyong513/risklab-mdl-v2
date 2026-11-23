"""LightGBM 适配器（``lgb``）。

与 :mod:`risklab_mdl.models.adapters.xgb_adapter` 同构：早停只用调用方给的 val，
objective 由任务决定，``predict_proba`` 恒返回 ``(n, num_class)`` 概率矩阵。

★ 为什么不复用 ``core.train`` 的 LGB 训练路径
--------------------------------------------
``core.train`` 是**旧版单算法**的既有实现（模型包 schema 1.0 口径、只支持 lgb），
它的返回值与「5 算法统一适配器」的接口不同。本模块按统一接口重写，
但概率取法与早停口径完全一致（都用 sklearn API 的 ``predict_proba``）；
两条路径的一致性由 ``tests/test_package_v2.py``（1.0 兼容）与实际指标对照守护。

★ ``num_class`` 不显式传
-----------------------
LightGBM 的 sklearn 接口会从 ``y`` 推断类别数；显式传 ``num_class`` 在部分版本会
触发「Cannot use num_class with sklearn interface」告警。本层不传，改由
:func:`risklab_mdl.models.adapters.base.check_proba` 在预测时校验列数 ——
**校验点在后、不变量在前**，两边都不会静默。
"""

from __future__ import annotations

import inspect
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from ...core.deps import require_lightgbm
from ...core.errors import ModelError
from .base import BaseAdapter

__all__ = ["LgbAdapter"]


class LgbAdapter(BaseAdapter):
    algorithm = "lgb"
    native_nan = True
    uses_val_early_stopping = True

    def fit(
        self,
        X: pd.DataFrame,
        y: Any,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Any = None,
    ) -> "LgbAdapter":
        if X is None or len(X) == 0:
            raise ModelError("lgb: 训练集为空")
        lgb = require_lightgbm()
        params = self._strip_objective_params()
        params.setdefault("random_state", self.seed)
        params.setdefault("verbosity", -1)
        params.setdefault("n_estimators", self.num_boost_round)
        params["objective"] = "binary" if self.task == "binary" else "multiclass"
        params["metric"] = "auc" if self.task == "binary" else "multi_logloss"

        use_val = X_val is not None and y_val is not None and len(X_val) > 0
        self._feature_names = [str(c) for c in X.columns]
        model = lgb.LGBMClassifier(**params)

        started = time.perf_counter()
        if use_val:
            callbacks = [
                lgb.early_stopping(int(self.early_stopping_rounds), verbose=False),
                lgb.log_evaluation(0),
            ]
            # lightgbm 4.7 起 eval_set 被 eval_X / eval_y 取代（旧参数仍可用但会发
            # 弃用告警）。按签名选择，避免测试输出被 DeprecationWarning 淹没。
            fit_params = inspect.signature(model.fit).parameters
            if "eval_X" in fit_params:
                # lightgbm 4.7 的 eval_X / eval_y 接受「单个 DataFrame/Series」
                # （与 eval_set 的 [(X, y)] 形态不同）—— 传 list 会报
                # "Data list can only be of ndarray or Sequence"
                model.fit(
                    X,
                    np.asarray(y).astype(int),
                    eval_X=X_val,
                    eval_y=np.asarray(y_val).astype(int),
                    callbacks=callbacks,
                )
            else:  # pragma: no cover - 旧版本 lightgbm
                model.fit(
                    X,
                    np.asarray(y).astype(int),
                    eval_set=[(X_val, np.asarray(y_val).astype(int))],
                    callbacks=callbacks,
                )
        else:
            model.fit(X, np.asarray(y).astype(int))
        self.train_seconds = time.perf_counter() - started
        self.model = model
        self.used_early_stopping = bool(use_val)
        self.best_iteration = _best_iteration_or_none(model)
        classes = [int(c) for c in getattr(model, "classes_", [])]
        expected = list(range(self.num_class))
        if classes and classes != expected:
            raise ModelError(
                f"lgb: 模型 classes_={classes} 与期望的 {expected} 不一致"
            )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ModelError("lgb: 尚未 fit，无法预测")
        frame = X.loc[:, [str(c) for c in self._feature_names]]
        return np.asarray(self.model.predict_proba(frame), dtype=float)

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ModelError("lgb: 尚未 fit，无法给出特征重要性")
        booster = getattr(self.model, "booster_", None)
        if booster is not None:
            values = booster.feature_importance(importance_type="gain")
            names = list(booster.feature_name())
            if len(names) == len(values):
                frame = pd.DataFrame({"feature": names, "gain": np.asarray(values, float)})
                total = float(frame["gain"].sum())
                frame["share"] = frame["gain"] / total if total > 0 else 0.0
                return frame.sort_values("gain", ascending=False).reset_index(drop=True)
        values = getattr(self.model, "feature_importances_", None)
        if values is None:
            raise ModelError("lgb: 无法取得特征重要性（booster 不可用）")
        return self._importance_frame(values, name="gain")


def _best_iteration_or_none(model: Any) -> Optional[int]:
    for name in ("best_iteration_", "best_iteration"):
        value = getattr(model, name, None)
        if isinstance(value, (int, np.integer)) and int(value) >= 0:
            return int(value)
    return None
