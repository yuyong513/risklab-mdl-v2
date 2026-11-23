"""risklab_mdl.core —— 纯计算层。

分箱 / WOE / IV、PSI、相关性、指标、编码、初筛、精筛、划分、变换、训练、调参、
模型包、模型文档。

■ 本层的三条硬约束
------------------
1. **纯计算**：不读 manifest、不读流程配置、不落盘（唯一例外是
   :func:`~risklab_mdl.core.package.save_package` / ``load_package``，
   它们由调用方显式传入路径）；
2. **导入期不拉重依赖**：lightgbm / xgboost / matplotlib 一律**函数内惰性导入**
   （统一入口 :mod:`risklab_mdl.core.deps`），因此
   ``import risklab_mdl.core`` 在没装这三个库的机器上也能成功；
3. **口径唯一**：KS/AUC/Gini/LIFT 只在 :mod:`~risklab_mdl.core.metrics` 定义，
   分箱/IV 只在 :mod:`~risklab_mdl.core.binning` 定义，样本划分只在
   :mod:`~risklab_mdl.core.split` 定义 —— 其它模块不得各自实现一份。

■ 模块速查
----------
===================================  ==================================================
模块                                 职责
===================================  ==================================================
``errors``                            异常体系（RiskLabError 及其三个子类）
``deps``                              可选重依赖的惰性入口（缺库抛 ModelError）
``binning``                           分箱 / WOE / IV / 分箱表 PSI（等频/等宽/自定义）
``binning_dt``                        决策树自动分箱（箱数上限/最小箱/单调/缺失独立）
``psi``                               跨期稳定性 PSI（单基期/滚动/多基期合并池）
``correlation``                       高相关特征去重（greedy / component）
``encode``                            标签编码 / onehot / WOE 编码与复现
``metrics``                           AUC / KS / Gini / LIFT / ScoreScale + 多分类 OVR
``expr``                              字段表达式受限解析器（禁 eval）
``split``                             样本划分（SplitSpec + split_frame，纯函数）
``prescreen``                         初筛五道筛子（缺失/同值/IV/PSI/相关性）
``refine``                            精筛（gain 迭代，需 lightgbm）
``tuning``                            调参空间与搜索（hyperopt）
``train``                             LGB 训练编排与指标（需 lightgbm）
``package``                           模型包（打分链路 / 落盘 / 加载）
``model_doc``                         模型文档 sheet 组装
``transforms``                        缺失填充 / 编码 / 异常值变换（预处理层直接调用）
``utils``                             跨层共享的小工具（数值归一 / 比率 / 箱标签）
===================================  ==================================================
"""

from __future__ import annotations

from . import (
    binning,
    binning_dt,
    correlation,
    deps,
    encode,
    errors,
    expr,
    metrics,
    model_doc,
    package,
    prescreen,
    psi,
    refine,
    split,
    train,
    transforms,
    tuning,
    utils,
)

__all__ = [
    "binning",
    "binning_dt",
    "correlation",
    "deps",
    "encode",
    "errors",
    "expr",
    "metrics",
    "model_doc",
    "package",
    "prescreen",
    "psi",
    "refine",
    "split",
    "train",
    "transforms",
    "tuning",
    "utils",
]
