"""XGBoost 适配器（``xgb``）。

★ 早停只认调用方给的 val
----------------------
``XGBClassifier`` 的 ``early_stopping_rounds`` 必须配合 ``eval_set`` 才有意义。
没有 val 时**显式关掉早停**（不传该参数）并固定 ``n_estimators`` 轮数 ——
让 xgboost 在内部切一块训练集当验证集，等于把「训练集规模」和「早停口径」
同时变成不可见的事实。

★ objective 由任务决定
--------------------
``binary`` → ``binary:logistic``；``multiclass`` → ``multi:softprob``（+ ``num_class``）。
用户传的 ``objective`` / ``num_class`` 会被丢弃并记录 warning（见 base.py）。
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from ...core.deps import require_xgboost
from ...core.errors import ModelError
from .base import BaseAdapter

__all__ = ["XgbAdapter"]


class XgbAdapter(BaseAdapter):
    algorithm = "xgb"
    native_nan = True
    uses_val_early_stopping = True

    def fit(
        self,
        X: pd.DataFrame,
        y: Any,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Any = None,
    ) -> "XgbAdapter":
        if X is None or len(X) == 0:
            raise ModelError("xgb: 训练集为空")
        xgb = require_xgboost()  # 缺库 ⇒ ModelError(detail.dependency="xgboost")
        params = self._strip_objective_params()
        params.setdefault("random_state", self.seed)
        params.setdefault("tree_method", "hist")
        params.setdefault("n_estimators", self.num_boost_round)
        params["objective"] = (
            "binary:logistic" if self.task == "binary" else "multi:softprob"
        )
        params["eval_metric"] = "auc" if self.task == "binary" else "mlogloss"
        if self.task == "multiclass":
            params["num_class"] = int(self.num_class)

        use_val = X_val is not None and y_val is not None and len(X_val) > 0
        if use_val:
            params["early_stopping_rounds"] = int(self.early_stopping_rounds)

        self._feature_names = [str(c) for c in X.columns]
        try:
            model = xgb.XGBClassifier(**params)
        except TypeError as exc:
            raise ModelError(
                f"xgb: 构造 XGBClassifier 失败（{exc}）。"
                f"常见原因：当前 xgboost 版本不接受某个参数（如 early_stopping_rounds "
                f"在不同大版本里位置不同）"
            ) from exc

        started = time.perf_counter()
        try:
            if use_val:
                model.fit(
                    X,
                    np.asarray(y).astype(int),
                    eval_set=[(X_val, np.asarray(y_val).astype(int))],
                    verbose=False,
                )
            else:
                model.fit(X, np.asarray(y).astype(int), verbose=False)
        except TypeError as exc:
            # 新旧版本对 early_stopping_rounds 的落点不同：回退到 fit 前 set_params
            raise ModelError(
                f"xgb: 训练失败（{exc}）。当前 xgboost 版本对 early_stopping_rounds "
                f"的接受位置与本适配器不一致，请升级 xgboost 或关掉早停"
                f"（config.early_stopping_rounds=0）"
            ) from exc
        self.train_seconds = time.perf_counter() - started
        self.model = model
        self.used_early_stopping = bool(use_val)
        self.best_iteration = _best_iteration_or_none(model)
        classes = [int(c) for c in getattr(model, "classes_", [])]
        expected = list(range(self.num_class))
        if classes and classes != expected:
            raise ModelError(
                f"xgb: 模型 classes_={classes} 与期望的 {expected} 不一致"
            )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ModelError("xgb: 尚未 fit，无法预测")
        frame = X.loc[:, [str(c) for c in self._feature_names]]
        return np.asarray(self.model.predict_proba(frame), dtype=float)

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ModelError("xgb: 尚未 fit，无法给出特征重要性")
        values = getattr(self.model, "feature_importances_", None)
        if values is None:
            raise ModelError("xgb: 模型没有 feature_importances_（增益重要性不可用）")
        return self._importance_frame(values, name="gain")


def _best_iteration_or_none(model: Any) -> Optional[int]:
    for name in ("best_iteration", "best_iteration_"):
        value = getattr(model, name, None)
        if isinstance(value, (int, np.integer)) and int(value) >= 0:
            return int(value)
    return None
