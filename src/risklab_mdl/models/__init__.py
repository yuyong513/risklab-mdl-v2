"""模型层：能力表（catalog）、适配器（adapters）、标签编码（labels）、
数据解析（data）、评价口径（evaluate）、训练装配（train）。

★ 导入约束
----------
本包 ``__init__`` **只导出模块对象**，不在导入期拉起 sklearn / xgboost / lightgbm：
``model_catalog`` 这类只读工具必须在「没装 xgboost」的机器上也能用
（理由同 :mod:`risklab_mdl.core.deps`）。
"""

from __future__ import annotations

from . import adapters, catalog, data, evaluate, labels, train

__all__ = [
    "adapters",
    "catalog",
    "data",
    "evaluate",
    "labels",
    "train",
]
