"""跨模块共享的小工具：数值归一、比率、类别相等与箱标签。

放在 core 最底层：只依赖标准库 + numpy / pandas，不 import 其它 core 模块。
``core`` / ``data`` / ``features`` / ``preprocess`` / ``models`` 共用同一份实现，
避免同一个辅助函数在各层各写一遍、口径悄悄分叉。
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "bin_interval_label",
    "round_finite",
    "round_value",
    "safe_ratio",
    "same_label",
    "same_value_rate",
]


def round_finite(value: Any, digits: int = 6) -> Optional[float]:
    """数值 → 保留 ``digits`` 位的 float；``None`` / 非数值 / NaN / ±inf 一律 ``None``。

    用于指标等「必须能进 JSON」的场景：JSON 没有 NaN / inf，直接落盘会产出非法文档。
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if np.isfinite(number) else None


def round_value(value: Any, digits: int = 6) -> Any:
    """数值 → 保留 ``digits`` 位；非数值原样返回（配置 / 模型参数透传用）。

    与 :func:`round_finite` 的区别：这里允许字符串等非数值原样通过 ——
    调用方在给「模型参数 / 配置」做 JSON 归一，不应把拼错的字符串悄悄变成 ``None``。
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return round(number, digits) if np.isfinite(number) else None


def safe_ratio(numerator: float, denominator: float) -> float:
    """安全除法：``denominator <= 0`` 时返回 ``0.0``（不抛 ``ZeroDivisionError``）。"""
    return float(numerator / denominator) if denominator > 0 else 0.0


def same_value_rate(series: pd.Series) -> float:
    """同值率 = 众数出现次数 / 非空样本数（全空列记 ``0.0``）。"""
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    counts = non_null.value_counts(dropna=True)
    return safe_ratio(float(counts.iloc[0]), float(len(non_null)))


def same_label(left: Any, right: Any) -> bool:
    """类别相等（数值 / 文本混用时按字符串兜底）。"""
    if left == right:
        return True
    return str(left) == str(right)


def bin_interval_label(
    code: int, edges: Sequence[float], *, missing_label: str = "[MISSING]"
) -> str:
    """箱编码 → 可读区间 ``(left, right]``；``code < 0`` 为缺失箱。"""
    if code < 0:
        return missing_label
    left = "-inf" if code == 0 else f"{float(edges[code - 1]):g}"
    right = "inf" if code >= len(edges) else f"{float(edges[code]):g}"
    return f"({left}, {right}]"
