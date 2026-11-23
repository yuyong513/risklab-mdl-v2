"""标签编码与 ``bad_class`` 判定（**全项目唯一入口**）。

★ 为什么必须有这一层
--------------------
5 个算法 × 2 个任务都以「0..k-1 的整数标签 + ``predict_proba`` 的列顺序」为接口
（sklearn / xgboost / lightgbm 全一样）。而业务标签可能是 ``0/1``、``good/bad``、
``M0/M1/M2``。中间必须有一处**确定的、可落盘、可复现**的映射，否则：

* 训练时 ``classes_`` 与推理时列顺序不一致 ⇒ 概率串位（AUC 看着还行，KS 崩）；
* 二分类「哪一类算坏」没有落盘 ⇒ 换台机器重算分数方向可能反过来。

⇒ 本模块产出 :class:`LabelEncoding`，并随模型包（schema 2.0 的
``classes`` / ``bad_class``）一起落盘，见 :mod:`risklab_mdl.core.package`。

★ ``bad_class`` 规则（写死在这里，文档见 docs/model-layer.md）

1. **显式指定**：``target.labels`` 映射里值为 ``"bad"``（大小写不敏感）的原始类别；
2. **二分类缺省**：原始取值 ``1``（按字符串比较，兼容 ``"1"``），
   若没有则取排序后的第二类；
3. **多分类缺省**：排序后的最后一类（风险等级通常按 ``0 < 1 < 2`` 递增）。

二分类的编码顺序固定为 ``[good, bad]`` ⇒ ``predict_proba[:, 1]`` 恒为「坏」的概率。
多分类按类别排序编码，``bad_class`` 只影响「分数用哪一列」与 PSI 口径。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.errors import DataError
from ..core.utils import same_label

__all__ = [
    "BAD_TOKEN",
    "LabelEncoding",
    "class_sort_key",
    "encode_labels",
    "resolve_bad_class",
    "sorted_classes",
]

#: ``labels`` 映射里表示「坏」的取值（大小写不敏感，与 ``data.loader.binary_bad_label``
#: 同一套 token —— 两处口径不一致会让「初筛认定的坏类」与「模型认定的坏类」不同）。
BAD_TOKENS: tuple[str, ...] = ("bad", "1", "坏", "违约", "正例", "positive")

#: 兼容旧名（单数 token 常量，仅供阅读用）。
BAD_TOKEN = "bad"


def class_sort_key(value: Any) -> tuple:
    """类别的自然排序键：数字类按数值，文本类按字符串。

    ★ 为什么不能直接 ``sorted(unique)``
    ----------------------------------
    ``{2, 10}`` 与 ``{"0", "1", "10", "2"}`` 的默认排序结果分别是 ``[2, 10]``
    与 ``["0", "1", "10", "2"]`` —— 后者把 10 排在 2 前面，
    多分类的「最后一类 = 最高风险」这条缺省规则会指到一个奇怪的类上。
    """
    if isinstance(value, (bool, np.bool_)):
        return (0, int(value), "")
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if np.isfinite(number):
            return (0, number, "")
    text = str(value)
    match = re.fullmatch(r"-?\d+(?:\.\d+)?", text.strip())
    if match:
        try:
            return (0, float(text), text)
        except ValueError:  # pragma: no cover - 正则已限定形态
            return (1, 0.0, text)
    return (1, 0.0, text)


def sorted_classes(values: Sequence[Any]) -> List[Any]:
    """去重 + 自然排序（保留 Python 原生类型，便于 JSON 落盘）。"""
    unique: List[Any] = []
    for value in values:
        native = value.item() if isinstance(value, np.generic) else value
        if not any(same_label(native, existing) for existing in unique):
            unique.append(native)
    return sorted(unique, key=class_sort_key)


def _label_token_map(labels: Mapping[str, Any] | None) -> Dict[str, str]:
    return {str(k): str(v) for k, v in (labels or {}).items()}


def resolve_bad_class(
    classes: Sequence[Any],
    *,
    task: str,
    labels: Mapping[str, Any] | None = None,
) -> Any:
    """按模块说明里的三条规则判定 ``bad_class``。"""
    if not classes:
        raise DataError("标签为空，无法判定 bad_class")
    tokens = _label_token_map(labels)
    explicit = [
        c
        for c in classes
        if tokens.get(str(c), "").strip().lower() in BAD_TOKENS
    ]
    if len(explicit) > 1:
        raise DataError(
            f"target.labels 把多个类别标成了 'bad'：{explicit}。"
            f"二分类只能有一个坏类；多分类的 bad_class 只用于「分数取哪一列」，"
            f"也必须是唯一一个 —— 多个坏类会让分数口径无定义。"
        )
    if explicit:
        return explicit[0]
    if task == "binary":
        for candidate in classes:
            if str(candidate).strip() == "1":
                return candidate
        return classes[-1] if len(classes) >= 2 else classes[0]
    return classes[-1]


@dataclass
class LabelEncoding:
    """标签编码结果（可序列化部分随模型包落盘）。"""

    task: str
    classes: List[Any]
    """原始类别，顺序即编码（``classes[i]`` 的编码是 ``i``）。"""
    class_map: Dict[str, int]
    bad_class: Any
    bad_index: int
    n_rows: int
    dropped_rows: int = 0
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "classes": list(self.classes),
            "class_map": dict(self.class_map),
            "bad_class": self.bad_class,
            "bad_index": int(self.bad_index),
            "n_rows": int(self.n_rows),
            "dropped_rows": int(self.dropped_rows),
            "warnings": list(self.warnings),
        }

    @property
    def num_class(self) -> int:
        return len(self.classes)

    def encode(self, series: pd.Series) -> pd.Series:
        """按本编码把原始标签列映射成整数（未知取值 → NaN，由调用方剔除）。"""
        mapping = {str(k): int(v) for k, v in self.class_map.items()}
        out = series.map(lambda v: mapping.get(str(v), np.nan))
        return out.astype("float64")

    def decode(self, codes: Sequence[int]) -> List[Any]:
        return [self.classes[int(code)] for code in codes]


def encode_labels(
    series: pd.Series,
    *,
    task: str,
    labels: Mapping[str, Any] | None = None,
) -> tuple[pd.Series, LabelEncoding]:
    """原始标签列 → 整数编码 + :class:`LabelEncoding`。

    Returns
    -------
    (codes, encoding)
        ``codes`` 与 ``series`` 同索引，缺失/未知标签为 ``NaN``（调用方负责剔除
        并告警 —— 本函数不静默补 0：补 0 等于把缺失标签判成「好客户」）。
    """
    task = str(task or "").strip().lower()
    if task not in ("binary", "multiclass"):
        raise DataError(f"未知任务类型 {task!r}（只允许 binary / multiclass）")

    observed = series.dropna().tolist()
    classes = sorted_classes(observed)
    warnings: List[str] = []
    if len(classes) < 2:
        raise DataError(
            f"目标列只有 {len(classes)} 个有效取值（{classes}）：至少需要 2 类。"
            f"检查 config.target.column 是否指对了标签列。"
        )

    if task == "binary":
        if len(classes) != 2:
            raise DataError(
                f"任务类型是 binary，但目标列有 {len(classes)} 个取值：{classes}。"
                f"请把 target.task 改成 multiclass，或确认标签列是否指错。"
            )
        bad = resolve_bad_class(classes, task=task, labels=labels)
        good = next(c for c in classes if not same_label(c, bad))
        ordered = [good, bad]
    else:
        ordered = list(classes)
        bad = resolve_bad_class(ordered, task=task, labels=labels)
        if len(ordered) == 2:
            warnings.append(
                f"任务类型声明为 multiclass，但目标列只有 2 类 {ordered}："
                f"仍按多分类 OVR 口径计算（macro 与二分类指标不等价）"
            )

    class_map = {str(label): index for index, label in enumerate(ordered)}
    codes = series.map(lambda v: class_map.get(str(v), np.nan)).astype("float64")
    dropped = int(codes.isna().sum() - series.isna().sum())
    missing = int(series.isna().sum())
    if missing:
        warnings.append(f"目标列有 {missing} 行缺失标签，已剔除（不补 0）")
    if dropped > 0:
        unknown = sorted({str(v) for v in series[codes.isna() & series.notna()].tolist()})
        warnings.append(
            f"目标列有 {dropped} 行取值不在已识别类别内（{unknown[:5]}），已剔除"
        )

    encoding = LabelEncoding(
        task=task,
        classes=ordered,
        class_map=class_map,
        bad_class=bad,
        bad_index=class_map[str(bad)],
        n_rows=int(series.shape[0]),
        dropped_rows=missing + max(dropped, 0),
        warnings=warnings,
    )
    return codes, encoding
