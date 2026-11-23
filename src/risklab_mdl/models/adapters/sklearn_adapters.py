"""scikit-learn 家族适配器：``dt`` / ``lr`` / ``gbdt``。

三者都是**固定迭代**（无早停）：sklearn 的 ``DecisionTreeClassifier`` /
``LogisticRegression`` 没有「按验证集提前停」的概念，
``GradientBoostingClassifier`` 的 ``n_iter_no_change`` 只在内部验证集上生效，
用它做早停等于**偷偷拿一部分训练数据当验证集**（口径不可见）。

⇒ 本层统一按固定轮数训练；需要早停的算法（xgb / lgb）在各自适配器里实现，
且只用调用方显式传入的 val 子集。

★ NaN 的硬门槛
--------------
``lr`` / ``dt`` / ``gbdt`` 都**不接受** NaN：sklearn 会直接抛
``ValueError: Input contains NaN``。本层不自己填（填充策略属于模型层
:mod:`risklab_mdl.models.train`，它知道 train 的中位数并会写 warnings）——
这里只在真收到 NaN 时给出可定位的报错（而不是让底层抛一条没有列名的异常）。
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from ...core.errors import ModelError
from .base import AdapterWarning, BaseAdapter

__all__ = ["DTAdapter", "GBDTAdapter", "LRAdapter"]


class _SklearnAdapter(BaseAdapter):
    """sklearn 家族公共部分：固定迭代、feature_importances_ 取重要性。"""

    native_nan = False
    uses_val_early_stopping = False
    _estimator: Any = None

    def _build_estimator(self, params: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def fit(
        self,
        X: pd.DataFrame,
        y: Any,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Any = None,
    ) -> "_SklearnAdapter":
        if X is None or len(X) == 0:
            raise ModelError(f"{self.algorithm}: 训练集为空")
        nan_columns = [c for c in X.columns if X[c].isna().any()]
        if nan_columns:
            raise ModelError(
                f"{self.algorithm}: 训练矩阵仍有缺失值（{len(nan_columns)} 列，"
                f"如 {nan_columns[:5]}）。sklearn 不接受 NaN —— "
                f"模型层应先用 train 中位数填充并告警；"
                f"直接交给 sklearn 会抛一条不带列名的 ValueError。"
            )
        params = self._strip_objective_params()
        params.setdefault("random_state", self.seed)
        self.params = params
        self._feature_names = [str(c) for c in X.columns]
        estimator = self._build_estimator(params)
        started = time.perf_counter()
        # ★ 收敛告警必须可见：sklearn 只在 stderr 打一条 ConvergenceWarning，
        #   工具层看不到 ⇒ 用户会拿一个「没收敛的 LR」当结论。
        from sklearn.exceptions import ConvergenceWarning  # 惰性导入（见模块说明）

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(X.to_numpy(dtype=float), np.asarray(y).astype(int))
        self.train_seconds = time.perf_counter() - started
        for item in caught:
            if issubclass(item.category, ConvergenceWarning):
                self.warnings.append(
                    AdapterWarning(
                        "not_converged",
                        f"{self.algorithm}: 训练未收敛（{item.category.__name__}: "
                        f"{str(item.message).splitlines()[0]}）—— "
                        f"系数不可信，请调小 C 或先做特征缩放/筛选",
                    )
                )
        self.model = estimator
        self._estimator = estimator
        self._class_check()
        return self

    def _class_check(self) -> None:
        classes = [int(c) for c in getattr(self.model, "classes_", [])]
        expected = list(range(self.num_class))
        if classes and classes != expected:
            raise ModelError(
                f"{self.algorithm}: 模型 classes_={classes} 与期望的 {expected} 不一致 —— "
                f"标签编码与训练矩阵不匹配"
            )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ModelError(f"{self.algorithm}: 尚未 fit，无法预测")
        frame = X.loc[:, [str(c) for c in self._feature_names]]
        raw = self.model.predict_proba(frame.to_numpy(dtype=float))
        return np.asarray(raw, dtype=float)

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ModelError(f"{self.algorithm}: 尚未 fit，无法给出特征重要性")
        values = getattr(self.model, "feature_importances_", None)
        if values is None:
            raise ModelError(
                f"{self.algorithm}: 该模型没有 feature_importances_，无法输出重要性"
            )
        return self._importance_frame(values)


class DTAdapter(_SklearnAdapter):
    algorithm = "dt"

    def _build_estimator(self, params: Mapping[str, Any]) -> Any:
        from sklearn.tree import DecisionTreeClassifier

        return DecisionTreeClassifier(**dict(params))


class LRAdapter(_SklearnAdapter):
    algorithm = "lr"

    def _build_estimator(self, params: Mapping[str, Any]) -> Any:
        from sklearn.linear_model import LogisticRegression

        # solver/penalty 组合由 catalog 默认值保证合法；用户覆盖成非法组合时
        # sklearn 自己会抛 ValueError（工具层转成 config_invalid，不静默换 solver）
        return LogisticRegression(**dict(params))

    def feature_importance(self) -> pd.DataFrame:
        """LR 的「重要性」= 系数绝对值（带符号的系数另列一栏）。

        ★ 为什么不直接返回系数
        --------------------
        报表里排 top 特征用的是「贡献大小」，符号是第二层信息。
        两者都给出：``importance``（|coef|）+ ``coef``（带符号），
        避免使用方按系数排序时把最强的负向特征排到最后。
        """
        if self.model is None:
            raise ModelError("lr: 尚未 fit，无法给出特征重要性")
        coef = np.asarray(self.model.coef_, dtype=float)
        if coef.ndim == 2 and coef.shape[0] > 1:
            # 多分类：取各类系数绝对值的均值（OVR 意义上的整体重要性）
            values = np.abs(coef).mean(axis=0)
        else:
            values = np.abs(np.ravel(coef))
        frame = self._importance_frame(values)
        if coef.ndim == 2:
            signed = coef.mean(axis=0) if coef.shape[0] > 1 else coef[0]
        else:
            signed = coef
        sign_map = dict(zip([str(c) for c in self._feature_names], np.ravel(signed)))
        frame["coef"] = [float(sign_map.get(f, np.nan)) for f in frame["feature"]]
        return frame


class GBDTAdapter(_SklearnAdapter):
    algorithm = "gbdt"

    def _build_estimator(self, params: Mapping[str, Any]) -> Any:
        from sklearn.ensemble import GradientBoostingClassifier

        params = dict(params)
        # n_estimators 是「固定轮数」的落点：与 catalog 默认值 / config 对齐
        if "n_estimators" not in params:
            params["n_estimators"] = self.num_boost_round
        return GradientBoostingClassifier(**params)
