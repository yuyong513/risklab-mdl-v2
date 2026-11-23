"""统一适配器接口：5 算法 × 2 任务的**唯一调用形态**。

★ 为什么要适配器层，而不是在工具里 ``if algorithm == "xgb": ...``
-----------------------------------------------------------------
5 个库的 ``fit`` 签名各不相同（早停怎么传、验证集叫什么、概率怎么取），
若写在工具层，``model_train`` / ``model_tune`` / ``model_evaluate`` 会各写一份，
三份之间必然漂移。适配器把这套差异收口在**一个类**里，上层只认四个方法：

============================  =====================================
方法                           语义
============================  =====================================
``fit(X, y, X_val, y_val)``    训练（有 val 则启用早停，无 val 固定轮数）
``predict_proba(X)``           ``(n, num_class)`` 概率矩阵，列序 = ``0..k-1``
``feature_importance()``       ``DataFrame[feature, importance]``（已排序）
``classes_``                   ``0..k-1``（**不是**原始标签，见 models.labels）
============================  =====================================

★ 惰性导入（硬约束）
-------------------
``dt/lr/gbdt`` 走 scikit-learn（本就随 pandas 栈安装），``xgb``/``lgb`` 必须
**在方法体内** import —— 模块级 import 会让「装了 scikit-learn 但没装 xgboost」
的机器连 ``model_catalog`` 都调不了。缺库时抛 ``ModelError``（``detail`` 带
``dependency`` / ``install``），由工具层转成契约错误码 ``dependency_missing``。

★ objective 由 task 决定，不由用户拍板
-------------------------------------
用户传进来的 ``objective`` / ``metric`` 会被**丢弃并记录到 warnings**：
``binary`` 用了 ``multi:softprob`` 这类错配会让概率列数不对，
而「能跑出数」会让这种错误一路带到报表。见 docs/model-layer.md。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ...core.errors import ModelError

__all__ = [
    "AdapterWarning",
    "BaseAdapter",
    "as_dataframe",
    "check_proba",
]


@dataclass
class AdapterWarning:
    """适配器侧的可见提示（工具层汇总进 ``warnings``）。"""

    kind: str
    message: str


def as_dataframe(X: Any, features: Sequence[str]) -> pd.DataFrame:
    """把入参规整成以 ``features`` 为列序的 DataFrame（列名统一成 str）。"""
    if isinstance(X, pd.DataFrame):
        frame = X
    else:
        frame = pd.DataFrame(np.asarray(X), columns=list(features))
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise ModelError(
            f"训练/打分矩阵缺少特征列：{missing[:10]}"
            f"{' ...' if len(missing) > 10 else ''}"
        )
    out = frame.loc[:, list(features)].copy()
    out.columns = [str(c) for c in out.columns]
    return out


def check_proba(arr: Any, n_rows: int, num_class: int, *, context: str) -> np.ndarray:
    """校验概率矩阵形状/有限性 —— 概率串位是「指标看着正常但结论全错」的根源。"""
    mat = np.asarray(arr, dtype=float)
    if mat.ndim == 1:
        if num_class != 2:
            raise ModelError(
                f"{context}：多分类模型返回了一维概率（{mat.shape}），"
                f"无法确定列顺序 ⇒ 拒绝使用"
            )
        mat = np.column_stack([1.0 - mat, mat])
    if mat.ndim != 2:
        raise ModelError(f"{context}：概率矩阵维度异常 {mat.shape}")
    if mat.shape[1] != num_class:
        raise ModelError(
            f"{context}：概率列数 {mat.shape[1]} 与类别数 {num_class} 不一致 —— "
            f"模型与标签编码不匹配（最常见：num_class 配错或复用了别的模型包）"
        )
    if mat.shape[0] != n_rows:
        raise ModelError(
            f"{context}：预测行数 {mat.shape[0]} 与输入行数 {n_rows} 不一致"
        )
    if not np.isfinite(mat).all():
        raise ModelError(f"{context}：概率中存在 NaN/Inf（请检查特征值域）")
    return mat


class BaseAdapter:
    """适配器基类：**只定义契约，不定义算法**。"""

    algorithm: str = ""
    task: str = "binary"
    num_class: int = 2
    #: 是否使用 val 早停（``False`` = 固定轮数训练）。
    uses_val_early_stopping: bool = False
    #: 该算法是否原生接受 NaN（``False`` ⇒ 模型层负责填充并告警）。
    native_nan: bool = False

    def __init__(
        self,
        *,
        task: str = "binary",
        params: Optional[Mapping[str, Any]] = None,
        seed: int = 42,
        num_class: int = 2,
        num_boost_round: int = 300,
        early_stopping_rounds: int = 50,
    ) -> None:
        self.task = str(task)
        self.params: Dict[str, Any] = dict(params or {})
        self.seed = int(seed)
        self.num_class = int(num_class)
        self.num_boost_round = int(num_boost_round)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.model: Any = None
        self._feature_names: List[str] = []
        self.warnings: List[AdapterWarning] = []
        self.best_iteration: Optional[int] = None
        self.used_early_stopping: bool = False
        self.train_seconds: float = 0.0

    # -- 生命周期 ------------------------------------------------------ #
    def fit(
        self,
        X: pd.DataFrame,
        y: Any,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Any = None,
    ) -> "BaseAdapter":  # pragma: no cover - 抽象方法
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def feature_importance(self) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError

    # -- 公共工具 ------------------------------------------------------ #
    @property
    def classes_(self) -> List[int]:
        return list(range(self.num_class))

    def _importance_frame(
        self, values: Sequence[float], *, name: str = "importance"
    ) -> pd.DataFrame:
        names = list(self._feature_names)
        values = np.asarray(values, dtype=float).ravel()
        if values.size != len(names):
            raise ModelError(
                f"特征重要性长度 {values.size} 与特征数 {len(names)} 不一致"
            )
        frame = pd.DataFrame({"feature": names, name: values})
        total = float(np.abs(values).sum())
        frame["share"] = (values / total) if total > 0 else 0.0
        return frame.sort_values(name, ascending=False).reset_index(drop=True)

    def _strip_objective_params(self) -> Dict[str, Any]:
        """丢弃用户传的 objective/metric/num_class（口径由 task 决定）。"""
        params = dict(self.params)
        dropped = {}
        for key in ("objective", "metric", "num_class", "eval_metric"):
            if key in params:
                dropped[key] = params.pop(key)
        if dropped:
            self.warnings.append(
                AdapterWarning(
                    "objective_ignored",
                    f"{self.algorithm}: 忽略 config.params 里的 {sorted(dropped)} —— "
                    f"objective/metric 由任务类型 {self.task} 决定（口径不允许由入参改写）",
                )
            )
        return params

    def predict_proba_checked(self, X: pd.DataFrame) -> np.ndarray:
        mat = self.predict_proba(X)
        return check_proba(
            mat, len(X), self.num_class, context=f"{self.algorithm}.predict_proba"
        )
