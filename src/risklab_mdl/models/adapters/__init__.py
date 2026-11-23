"""适配器工厂：``(algorithm, task)`` → 适配器实例。

★ 唯一的「能不能跑」判定点
--------------------------
未知算法 / 未知任务 / 组合不在 10 格能力表内 ⇒ ``ModelError``，
``detail`` 里带 ``tool_code="task_algorithm_mismatch"``（工具层据此映射契约错误码）。
缺库 ⇒ ``ModelError``，``detail`` 里带 ``dependency`` / ``install``
（工具层映射 ``dependency_missing``）。

★ 惰性导入
----------
本模块**顶层不 import** sklearn / xgboost / lightgbm（xgboost / lightgbm 在各自
适配器的 ``fit`` 里才 import）。因此 ``import risklab_mdl.models`` 不会拉起重依赖，
``model_catalog`` 这类只读工具在没有 xgboost 的机器上照样可用。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from ...core.errors import ConfigError, ModelError
from ..catalog import ALGORITHMS, TASKS, default_params, default_space, supports
from .base import BaseAdapter
from .sklearn_adapters import DTAdapter, GBDTAdapter, LRAdapter

__all__ = [
    "ADAPTER_CLASSES",
    "build_adapter",
    "describe_support",
    "merge_params",
]

#: 允许出现但会由适配器**丢弃并告警**的参数名（口径由 task 决定，见 base.py）。
OBJECTIVE_PASSTHROUGH: tuple[str, ...] = (
    "objective",
    "metric",
    "num_class",
    "eval_metric",
)

#: 算法 → 适配器类（xgb / lgb 的模块导入本身不拉库，所以可以静态登记）。
ADAPTER_CLASSES: Dict[str, type] = {
    "dt": DTAdapter,
    "lr": LRAdapter,
    "gbdt": GBDTAdapter,
}


def _mismatch(algorithm: str, task: str, detail: str = "") -> ModelError:
    message = (
        f"算法与任务组合不可用：algorithm={algorithm!r}, task={task!r}。"
        f"可用算法：{list(ALGORITHMS)}；可用任务：{list(TASKS)}。"
    )
    if detail:
        message = f"{message} {detail}"
    return ModelError(
        message,
        detail={
            "tool_code": "task_algorithm_mismatch",
            "algorithm": algorithm,
            "task": task,
            "supported_algorithms": list(ALGORITHMS),
            "supported_tasks": list(TASKS),
        },
    )


def describe_support(algorithm: str, task: str) -> Dict[str, Any]:
    """能力查询（不构造适配器，不 import 重依赖）。"""
    return {
        "algorithm": algorithm,
        "task": task,
        "supported": supports(algorithm, task),
        "default_params": default_params(algorithm, task),
        "default_space": default_space(algorithm, task),
    }


def merge_params(
    algorithm: str,
    task: str,
    params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """默认参数 ← ``config.params`` 覆盖。

    ★ 未知键名在**这里**就拒绝（``ConfigError`` → 契约 ``config_invalid``）
    ---------------------------------------------------------------------
    以前不校验：拼错的键（如 ``max_depht``）会一路进到
    ``DecisionTreeClassifier(**params)`` 抛 ``TypeError: unexpected keyword
    argument``，外层把它当「非预期失败」映射成 ``analysis_failed`` ——
    用户被告知「程序出错了」，而真正的原因是参数名拼错。
    合法键集合 = 该 ``(algorithm, task)`` 格的 ``default_params``（catalog 是声明口径），
    外加会被丢弃并告警的 ``objective/metric/num_class/eval_metric``。
    """
    merged = default_params(algorithm, task)
    incoming = {str(k): v for k, v in (params or {}).items()}
    allowed = set(merged) | set(OBJECTIVE_PASSTHROUGH)
    unknown = sorted(key for key in incoming if key not in allowed)
    if unknown:
        raise ConfigError(
            f"config.params 含未知参数：{unknown}。"
            f"{algorithm}/{task} 可用参数：{sorted(merged)}"
            f"（objective/metric/num_class/eval_metric 会被忽略并告警）。"
            f"参数名拼错不会静默忽略 —— 拼错的参数等于没设，模型会用默认值跑"
        )
    merged.update(incoming)
    return merged


def _adapter_class(algorithm: str) -> type:
    """取适配器类（xgb / lgb 的模块在此处才 import，仍然不拉第三方库）。"""
    if algorithm in ADAPTER_CLASSES:
        return ADAPTER_CLASSES[algorithm]
    if algorithm == "xgb":
        from .xgb_adapter import XgbAdapter

        return XgbAdapter
    if algorithm == "lgb":
        from .lgb_adapter import LgbAdapter

        return LgbAdapter
    raise _mismatch(algorithm, "", detail="未知算法。")


def build_adapter(
    algorithm: str,
    task: str,
    params: Optional[Mapping[str, Any]] = None,
    *,
    seed: int = 42,
    num_class: int = 2,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 50,
) -> BaseAdapter:
    """构造适配器（**不训练**）。组合非法时抛 ``task_algorithm_mismatch`` 载荷。"""
    algorithm = str(algorithm or "").strip().lower()
    task = str(task or "").strip().lower()
    if algorithm not in ALGORITHMS:
        raise _mismatch(algorithm, task, detail="未知算法（从 model_catalog 取）。")
    if task not in TASKS:
        raise _mismatch(
            algorithm, task, detail="未知任务类型（只支持 binary / multiclass）。"
        )
    if not supports(algorithm, task):
        raise _mismatch(algorithm, task)

    cls = _adapter_class(algorithm)
    return cls(
        task=task,
        params=merge_params(algorithm, task, params),
        seed=seed,
        num_class=num_class,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )
