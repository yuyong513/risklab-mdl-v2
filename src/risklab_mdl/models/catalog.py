"""算法 × 任务能力表（5 算法 × 2 任务 = 10 格）。

★ 为什么能力表是**数据**而不是散落在各工具的 if/else
----------------------------------------------------
「哪种算法支持哪种任务」「默认超参是什么」「默认调参搜什么范围」这三件事
在 ``model_catalog`` / ``model_train`` / ``model_tune`` 三个工具里都要用。
如果各写一份，调参预览与真正执行的搜索空间就会漂移 ——
用户在预览里看到 ``max_depth ∈ [3, 10]``，实际执行却搜 ``[2, 5]``，
而**两边都不报错**（这是「预览与执行不一致」最典型的形态）。

⇒ 本模块是**唯一真源**：三个工具都从这里取，测试对预览/执行做一致性断言。

★ 10 格一个不少
----------------
``dt`` / ``lr`` / ``gbdt``（scikit-learn）、``xgb``（xgboost）、``lgb``（lightgbm）
× ``binary`` / ``multiclass``。每格都有 ``default_params`` 与 ``default_space``；
不存在「某格留空 ⇒ 运行时降级用另一格」的写法（静默降级比报错糟得多）。

★ 默认参数的取舍
----------------
* 树模型默认深度浅、叶子小 —— 信贷样本量下深树必然过拟合，且 696 列宽表上
  深树的训练时间会失控（这是本项目的真实样例数据规模）；
* ``gbdt``（sklearn 的 ``GradientBoostingClassifier``）默认 ``n_estimators=100``：
  它是单线程逐树串行，默认 100 棵是「能出可用结果」与「宽表不超时」的折中；
* ``lr`` 必须无缺失（见 docs/model-layer.md「LR 的硬门槛」），缺失由模型层显式
  填充并告警，**不静默当 0**。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from ..core.errors import ConfigError

__all__ = [
    "ALGORITHMS",
    "ALGORITHM_LABELS",
    "CATALOG_SCHEMA",
    "DEPENDENCIES",
    "DEFAULT_PARAMS",
    "DEFAULT_SPACES",
    "TASKS",
    "algorithm_tasks",
    "catalog_payload",
    "default_params",
    "default_space",
    "describe_algorithm",
    "supports",
]

CATALOG_SCHEMA = "risklab-mdl.model-catalog/1"

#: 算法顺序（也是 UI 展示顺序）。
ALGORITHMS: tuple[str, ...] = ("dt", "lr", "gbdt", "xgb", "lgb")

#: 任务类型（与 manifest 的 ``TASK_TYPES`` 一致）。
TASKS: tuple[str, ...] = ("binary", "multiclass")

ALGORITHM_LABELS: Dict[str, str] = {
    "dt": "决策树（scikit-learn DecisionTreeClassifier）",
    "lr": "逻辑回归（scikit-learn LogisticRegression）",
    "gbdt": "梯度提升树（scikit-learn GradientBoostingClassifier）",
    "xgb": "XGBoost（XGBClassifier）",
    "lgb": "LightGBM（LGBMClassifier）",
}

#: 该算法依赖的可选库（``None`` 表示只依赖 scikit-learn，无额外可选依赖）。
DEPENDENCIES: Dict[str, str | None] = {
    "dt": None,
    "lr": None,
    "gbdt": None,
    "xgb": "xgboost",
    "lgb": "lightgbm",
}

_NOTE_COMMON = (
    "早停只在提供 val 子集时启用；无 val 时按固定轮数训练（不静默少训/多训）。"
    "objective / metric 由 task 决定，config.params 里的同名项会被忽略（口径优先）。"
)

#: 10 格的默认参数。
DEFAULT_PARAMS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "dt": {
        "binary": {
            "criterion": "gini",
            "max_depth": 6,
            "min_samples_leaf": 20,
            "min_samples_split": 10,
            "random_state": 42,
        },
        "multiclass": {
            "criterion": "gini",
            "max_depth": 6,
            "min_samples_leaf": 20,
            "min_samples_split": 10,
            "random_state": 42,
        },
    },
    "lr": {
        "binary": {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 2000,
            "tol": 0.0001,
            "fit_intercept": True,
        },
        "multiclass": {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 2000,
            "tol": 0.0001,
            "fit_intercept": True,
        },
    },
    "gbdt": {
        "binary": {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.8,
            "min_samples_leaf": 20,
            "random_state": 42,
        },
        "multiclass": {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.8,
            "min_samples_leaf": 20,
            "random_state": 42,
        },
    },
    "xgb": {
        "binary": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "random_state": 42,
        },
        "multiclass": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "random_state": 42,
        },
    },
    "lgb": {
        "binary": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "verbosity": -1,
            "random_state": 42,
        },
        "multiclass": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "verbosity": -1,
            "random_state": 42,
        },
    },
}

#: 10 格的默认调参空间（类型口径与 ``core.tuning.SearchSpace`` 一致）。
DEFAULT_SPACES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "dt": {
        "binary": {
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "min_samples_leaf": {"type": "int", "low": 5, "high": 100},
            "criterion": {"type": "choice", "options": ["gini", "entropy"]},
        },
        "multiclass": {
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "min_samples_leaf": {"type": "int", "low": 5, "high": 100},
            "criterion": {"type": "choice", "options": ["gini", "entropy"]},
        },
    },
    "lr": {
        "binary": {
            "C": {"type": "loguniform", "low": 0.01, "high": 10.0},
            "fit_intercept": {"type": "choice", "options": [True, False]},
        },
        "multiclass": {
            "C": {"type": "loguniform", "low": 0.01, "high": 10.0},
            "fit_intercept": {"type": "choice", "options": [True, False]},
        },
    },
    "gbdt": {
        "binary": {
            "n_estimators": {"type": "int", "low": 50, "high": 200},
            "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.2},
            "max_depth": {"type": "int", "low": 2, "high": 5},
            "subsample": {"type": "uniform", "low": 0.6, "high": 1.0},
            "min_samples_leaf": {"type": "int", "low": 5, "high": 50},
        },
        "multiclass": {
            "n_estimators": {"type": "int", "low": 50, "high": 200},
            "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.2},
            "max_depth": {"type": "int", "low": 2, "high": 5},
            "subsample": {"type": "uniform", "low": 0.6, "high": 1.0},
            "min_samples_leaf": {"type": "int", "low": 5, "high": 50},
        },
    },
    "xgb": {
        "binary": {
            "n_estimators": {"type": "int", "low": 100, "high": 500},
            "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.3},
            "max_depth": {"type": "int", "low": 3, "high": 8},
            "subsample": {"type": "uniform", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "uniform", "low": 0.6, "high": 1.0},
            "reg_lambda": {"type": "loguniform", "low": 0.1, "high": 10.0},
        },
        "multiclass": {
            "n_estimators": {"type": "int", "low": 100, "high": 500},
            "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.3},
            "max_depth": {"type": "int", "low": 3, "high": 8},
            "subsample": {"type": "uniform", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "uniform", "low": 0.6, "high": 1.0},
            "reg_lambda": {"type": "loguniform", "low": 0.1, "high": 10.0},
        },
    },
    "lgb": {
        "binary": {
            "num_leaves": {"type": "int", "low": 15, "high": 127},
            "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.3},
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "min_child_samples": {"type": "int", "low": 5, "high": 50},
            "subsample": {"type": "uniform", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "uniform", "low": 0.6, "high": 1.0},
            "reg_lambda": {"type": "loguniform", "low": 0.01, "high": 10.0},
        },
        "multiclass": {
            "num_leaves": {"type": "int", "low": 15, "high": 127},
            "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.3},
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "min_child_samples": {"type": "int", "low": 5, "high": 50},
            "subsample": {"type": "uniform", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "uniform", "low": 0.6, "high": 1.0},
            "reg_lambda": {"type": "loguniform", "low": 0.01, "high": 10.0},
        },
    },
}

#: 逐算法的说明（供 agent / 报表直接呈现，不必再解释）。
_NOTES: Dict[str, str] = {
    "dt": (
        "单棵决策树：可解释性最好（规则可直接读出），但方差大、易过拟合。"
        "常用作「规则基线」，不建议直接作为上线模型。" + _NOTE_COMMON
    ),
    "lr": (
        "逻辑回归：系数可解释、可做 WOE 入模。★ 硬门槛：不接受缺失值，"
        "模型层会用 train 中位数填充并显式告警（不静默当 0）；"
        "宽表下建议先做特征筛选，否则系数不稳定。" + _NOTE_COMMON
    ),
    "gbdt": (
        "sklearn 梯度提升树：无需额外依赖，但**单线程逐树串行**，"
        "宽表（数百列）上明显慢于 xgb/lgb。默认 n_estimators=100 是速度折中。"
        + _NOTE_COMMON
    ),
    "xgb": (
        "XGBoost：正则化完整、宽表表现稳定。需要 ``pip install xgboost``。"
        "早停用 val 的 AUC（binary）/ mlogloss（multiclass）。" + _NOTE_COMMON
    ),
    "lgb": (
        "LightGBM：最快、内存占用低，风控主力。需要 ``pip install lightgbm``。"
        "早停用 val 的 AUC（binary）/ multi_logloss（multiclass）。" + _NOTE_COMMON
    ),
}


def supports(algorithm: str, task: str) -> bool:
    """该 ``(算法, 任务)`` 组合是否在 10 格能力表内。"""
    return algorithm in ALGORITHMS and task in TASKS


def algorithm_tasks(algorithm: str) -> List[str]:
    """某算法支持的任务列表（未知算法返回空列表，由调用方报错）。"""
    return list(TASKS) if algorithm in ALGORITHMS else []


def default_params(algorithm: str, task: str) -> Dict[str, Any]:
    """该格的默认参数（**深拷贝**：调用方可随意改，不会污染能力表）。"""
    return dict(DEFAULT_PARAMS.get(algorithm, {}).get(task, {}) or {})


def default_space(algorithm: str, task: str) -> Dict[str, Dict[str, Any]]:
    """该格的默认调参空间（深拷贝两层，理由同上）。"""
    space = DEFAULT_SPACES.get(algorithm, {}).get(task, {}) or {}
    return {name: dict(spec) for name, spec in space.items()}


def describe_algorithm(algorithm: str, task: str) -> Dict[str, Any]:
    """单格说明（catalog 与 model_train 回显共用）。"""
    return {
        "algorithm": algorithm,
        "label": ALGORITHM_LABELS.get(algorithm, algorithm),
        "task": task,
        "supported": supports(algorithm, task),
        "dependency": DEPENDENCIES.get(algorithm),
        "install": (
            f"pip install {DEPENDENCIES[algorithm]}"
            if DEPENDENCIES.get(algorithm)
            else ""
        ),
        "default_params": default_params(algorithm, task),
        "default_space": default_space(algorithm, task),
        "notes": _NOTES.get(algorithm, ""),
    }


def catalog_payload() -> Dict[str, Any]:
    """``model_catalog`` 工具的 ``data`` 载荷（契约 §3.4）。

    ``data.catalog.algorithms`` 的形状按契约：每个算法含 ``tasks``（支持的任务
    列表）、``default_params``、``default_space``、``notes``。

    ★ 额外给出 ``by_task``
    ----------------------
    二分类与多分类的默认参数/空间**确实不同**（objective、num_class 一类）。
    契约的扁平字段保留（取 binary 那格，便于「只想看一个数」的调用方），
    逐格的完整取值放 ``by_task``，两处都由本模块同一份数据生成，不会漂移。
    """
    algorithms: Dict[str, Any] = {}
    for algorithm in ALGORITHMS:
        by_task = {
            task: {
                "default_params": default_params(algorithm, task),
                "default_space": default_space(algorithm, task),
            }
            for task in TASKS
        }
        algorithms[algorithm] = {
            "label": ALGORITHM_LABELS[algorithm],
            "tasks": algorithm_tasks(algorithm),
            "dependency": DEPENDENCIES.get(algorithm),
            "install": (
                f"pip install {DEPENDENCIES[algorithm]}"
                if DEPENDENCIES.get(algorithm)
                else ""
            ),
            "default_params": default_params(algorithm, "binary"),
            "default_space": default_space(algorithm, "binary"),
            "by_task": by_task,
            "notes": _NOTES.get(algorithm, ""),
        }
    matrix = [
        {
            "algorithm": algorithm,
            "task": task,
            "supported": True,
            "default_params": default_params(algorithm, task),
            "default_space": default_space(algorithm, task),
        }
        for algorithm in ALGORITHMS
        for task in TASKS
    ]
    return {
        "schema": CATALOG_SCHEMA,
        "algorithms": algorithms,
        "algorithm_order": list(ALGORITHMS),
        "tasks": list(TASKS),
        "matrix": matrix,
        "matrix_size": len(matrix),
        "label_rules": {
            "encoding": "原始类别 → 0..k-1（按类别排序），编码映射随模型包落盘",
            "bad_class_binary": (
                "labels 映射里标 'bad' 的类；缺省取原始取值 1（其次取排序后第二类）"
            ),
            "bad_class_multiclass": "labels 映射里标 'bad' 的类；缺省取排序后最后一类",
        },
    }


def resolve_space(
    algorithm: str, task: str, custom: Mapping[str, Any] | None = None
) -> Dict[str, Dict[str, Any]]:
    """合并默认空间与调用方覆盖项。

    ``custom`` 中的参数名必须出现在默认空间里 —— **不允许凭空新增**
    （那会让预览里看不到、执行时却生效，属「配置漂移」）。
    """
    space = default_space(algorithm, task)
    for name, spec in (custom or {}).items():
        if name not in space:
            raise ConfigError(
                f"调参空间参数 {name!r} 不在 {algorithm}/{task} 的默认空间内"
                f"（可用：{sorted(space)}）"
            )
        if not isinstance(spec, Mapping):
            raise ConfigError(f"调参空间参数 {name!r} 的声明必须是映射")
        space[name] = dict(spec)
    return space
