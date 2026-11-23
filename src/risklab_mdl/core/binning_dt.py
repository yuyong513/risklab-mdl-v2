"""决策树自动分箱（DT binning）—— 输出与 :class:`~risklab_mdl.core.binning.BinSpec` 兼容的切点。

设计目标（与 PM/Q11 口径对齐）
------------------------------
1. **最大箱数** ``max_bins``：切点数量上限（有效箱数 ≤ ``max_bins``，不含缺失箱）；
2. **最小箱占比 / 最小样本数** ``min_bin_pct`` / ``min_bin_count``：两者取严，
   违规的小箱向相邻箱合并（合并后仍违规就继续合并）；
3. **切点单调**：切点严格升序、无重复；``monotonic=True`` 时额外保证
   **各箱坏样本率单调**（不增或不减），违规的相邻箱对会被合并；
4. **缺失独立成箱**：缺失不参与切点拟合，变换时落 ``MISSING_BIN``(-1)
   —— 与 :func:`~risklab_mdl.core.binning.transform_bins` 完全一致，
   因此 ``result.spec`` 可直接喂给 ``transform_bins`` / ``bin_table_processed``。

★ 为什么切点用「树」而不是等频
------------------------------
等频只看边缘分布，会在「取值集中但风险差异大」的区间里切不出区分度；
树的切点直接由**坏样本率的变化**驱动，天然对齐 IV。
代价是切点依赖训练期样本，因此**必须在 train 上拟合**（防穿越，同 ``fit_bins``）。

★ 算法
------
① ``sklearn.tree.DecisionTreeClassifier(max_leaf_nodes=max_bins, min_samples_leaf=...)``
   在单列特征上拟合，取 ``tree_.threshold`` 作为初始切点；
② 用切点把样本装箱（缺失行不参与），得到逐箱样本数与坏样本率；
③ 依次执行：箱数上限收敛 → 最小占比/样本数合并 → 坏样本率单调合并；
④ 输出 ``BinSpec(method="business_cut", edges=...)`` + 逐箱诊断。

★ 失败必须可见
--------------
* 标签不同时含 0/1 ⇒ ``DataError``（算不出坏样本率，分箱没有意义）；
* 全列取值恒定 / 全部缺失 ⇒ 不报错，但 ``notes`` 显式说明「退化为单箱」；
* ``max_bins < 2`` / 占比下限 × 箱数 > 1 之类**自相矛盾**的配置 ⇒ ``ConfigError``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .binning import BinSpec
from .errors import ConfigError, DataError

__all__ = [
    "DTBinningSpec",
    "DTBinResult",
    "dt_bin_table",
    "fit_bins_dt",
]

_EPS = 1e-12


@dataclass(frozen=True)
class DTBinningSpec:
    """DT 分箱参数。

    ``max_bins``
        有效箱（不含缺失箱）上限。``DecisionTreeClassifier`` 的 ``max_leaf_nodes``
        也取该值，因此树本身不会切出多余的箱。
    ``min_bin_pct`` / ``min_bin_count``
        每箱最小占比 / 最小样本数，**两者同时生效（取严）**。
        ``min_bin_count=0`` 表示只看占比。
    ``max_depth``
        树深上限（``max_leaf_nodes`` 更严格时由它兜底）。
    ``min_samples_leaf_override``
        非 0 时直接指定树的叶节点最小样本数（默认由 ``min_bin_pct``/``min_bin_count`` 推导）。
    ``criterion``
        ``"entropy"``（默认，等价 Gini/信息增益权衡下的风险区分）或 ``"gini"``。
    ``monotonic``
        是否强制各箱坏样本率单调。
    ``missing_as_bin``
        缺失是否单独成箱（默认 True，落 ``MISSING_BIN``）。
    ``random_state``
        树是确定性的，但显式传参便于复现记录。
    """

    max_bins: int = 5
    min_bin_pct: float = 0.05
    min_bin_count: int = 50
    max_depth: int = 4
    min_samples_leaf_override: int = 0
    criterion: str = "entropy"
    monotonic: bool = True
    missing_as_bin: bool = True
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.max_bins < 2:
            raise ConfigError(f"DT 分箱的 max_bins 至少为 2，当前 {self.max_bins}")
        if not 0.0 <= self.min_bin_pct < 1.0:
            raise ConfigError(
                f"min_bin_pct 必须落在 [0, 1) 区间，当前 {self.min_bin_pct}"
            )
        if self.min_bin_count < 0:
            raise ConfigError(
                f"min_bin_count 不能为负，当前 {self.min_bin_count}"
            )
        if self.min_bin_pct * self.max_bins > 1.0 + _EPS:
            raise ConfigError(
                f"配置自相矛盾：min_bin_pct({self.min_bin_pct}) × "
                f"max_bins({self.max_bins}) > 1 —— 再多箱也不可能同时满足占比下限。"
                f"请下调 min_bin_pct 或 max_bins。"
            )
        if self.max_depth < 1:
            raise ConfigError(f"max_depth 至少为 1，当前 {self.max_depth}")
        if self.criterion not in ("gini", "entropy"):
            raise ConfigError(
                f"criterion 只支持 gini / entropy，当前 {self.criterion!r}"
            )


@dataclass(frozen=True)
class DTBinResult:
    """DT 分箱结果。

    ``spec``
        与既有 ``BinSpec`` 完全兼容 —— 可直接交给
        :func:`~risklab_mdl.core.binning.transform_bins` /
        :func:`~risklab_mdl.core.binning.bin_table_processed`。
    ``edges``
        与 ``spec.edges`` 相同的切点（升序、无重复）。
    ``bin_counts`` / ``bin_bad_rates``
        逐有效箱（不含缺失箱）的样本数与坏样本率，用于报表核对。
    ``missing_count``
        缺失样本数（``missing_as_bin=False`` 时该批样本不落箱）。
    ``is_monotonic``
        最终各箱坏样本率是否单调。
    ``notes``
        退化/下调等需要显式告知的事实（**不静默**）。
    """

    spec: BinSpec
    edges: tuple[float, ...]
    bin_counts: tuple[int, ...]
    bin_bad_rates: tuple[float, ...]
    missing_count: int
    total_count: int
    is_monotonic: bool
    monotonic_direction: str
    notes: tuple[str, ...]

    @property
    def n_bins(self) -> int:
        return len(self.bin_counts)

    def to_dict(self) -> dict:
        """便于落盘/断言的结构化摘要。"""
        return {
            "edges": list(self.edges),
            "n_bins": self.n_bins,
            "bin_counts": list(self.bin_counts),
            "bin_bad_rates": [round(v, 6) for v in self.bin_bad_rates],
            "missing_count": self.missing_count,
            "total_count": self.total_count,
            "is_monotonic": self.is_monotonic,
            "monotonic_direction": self.monotonic_direction,
            "missing_as_bin": self.spec.missing_as_bin,
            "min_bin_count": self.spec.min_bin_count,
            "min_bin_pct": self.spec.min_bin_pct,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _as_numeric(x: pd.Series) -> np.ndarray:
    if not isinstance(x, pd.Series):
        raise DataError(f"DT 分箱需要 pd.Series，收到 {type(x).__name__}")
    return pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)


def _as_binary_y(y: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=float).ravel()
    if arr.size == 0:
        raise DataError("标签为空，无法进行 DT 分箱")
    if np.isnan(arr).any():
        raise DataError("标签中存在缺失值，请先处理标签后再进行 DT 分箱")
    uniq = np.unique(arr)
    if uniq.size != 2 or not set(uniq.tolist()) <= {0.0, 1.0}:
        raise DataError(
            f"DT 分箱只接受二分类 0/1 标签，当前取值 {uniq.tolist()[:5]}"
            f"（共 {uniq.size} 种）。多分类请按类 one-vs-rest 逐类分箱。"
        )
    return arr


def _bin_counts(values: np.ndarray, y: np.ndarray, edges: Sequence[float]):
    """按切点装箱，返回逐箱 (样本数, 坏样本数)。"""
    e = np.asarray(list(edges), dtype=float)
    codes = np.searchsorted(e, values, side="left")
    k = len(e) + 1
    counts = np.bincount(codes, minlength=k)[:k]
    bad = np.bincount(codes, weights=y, minlength=k)[:k]
    return counts.astype(int), bad.astype(float)


def _merge(bins: list[dict], i: int) -> None:
    """把第 i 箱与第 i+1 箱合并（就地）。

    ★ 合并只「吃掉」两箱之间的那条切点（右箱的边界由上界覆盖），
      因此最终切点始终是**初始切点的子集** —— 与拟合时的装箱结果逐行一致。
    """
    left, right = bins[i], bins[i + 1]
    bins[i] = {
        "count": left["count"] + right["count"],
        "bad": left["bad"] + right["bad"],
        "hi": right["hi"],
    }
    del bins[i + 1]


def _bad_rate(b: dict) -> float:
    return float(b["bad"] / b["count"]) if b["count"] else 0.0


def _enforce_min_share(
    bins: list[dict], n_total: int, min_count: int, min_pct: float, notes: list[str]
) -> None:
    """把违规的小箱并入**样本更少的那个邻箱**，直到全部达标或只剩 1 箱。"""
    threshold = max(int(min_count), int(np.ceil(min_pct * n_total)))
    if threshold <= 0:
        return
    while len(bins) > 1:
        small = [i for i, b in enumerate(bins) if b["count"] < threshold]
        if not small:
            return
        # 最违规的箱优先（样本数最少）
        i = min(small, key=lambda j: bins[j]["count"])
        if i == 0:
            j = 0
        elif i == len(bins) - 1:
            j = len(bins) - 2
        else:
            j = i - 1 if bins[i - 1]["count"] <= bins[i + 1]["count"] else i
        _merge(bins, j)
        notes.append(
            f"最小箱约束：样本数 < {threshold} 的箱已合并到相邻箱，"
            f"当前剩 {len(bins)} 箱"
        )


def _is_monotonic_rates(rates: Sequence[float], direction: str) -> bool:
    if direction == "increasing":
        return all(rates[i] <= rates[i + 1] + 1e-12 for i in range(len(rates) - 1))
    if direction == "decreasing":
        return all(rates[i] >= rates[i + 1] - 1e-12 for i in range(len(rates) - 1))
    return True


def _enforce_monotonic(
    bins: list[dict], direction: str, notes: list[str]
) -> None:
    """合并坏样本率违反单调方向的相邻箱对（每次都合并最小的一对，确定性）。"""
    if direction == "flat" or len(bins) < 3:
        return
    guard = 0
    while len(bins) >= 2:
        rates = [_bad_rate(b) for b in bins]
        violators = [
            i
            for i in range(len(rates) - 1)
            if (direction == "increasing" and rates[i] > rates[i + 1] + 1e-12)
            or (direction == "decreasing" and rates[i] < rates[i + 1] - 1e-12)
        ]
        if not violators:
            return
        guard += 1
        if guard > len(bins) + 5:  # 兜底，避免死循环（数学上不会发生）
            return
        i = min(violators, key=lambda j: bins[j]["count"] + bins[j + 1]["count"])
        _merge(bins, i)
        notes.append(f"单调性约束：坏样本率违反「{direction}」，第 {i}、{i + 1} 箱已合并")


def _tree_edges(
    values: np.ndarray, y: np.ndarray, spec: DTBinningSpec, min_leaf: int
) -> np.ndarray:
    """用决策树取初始切点；无有效切分时返回空数组。"""
    from sklearn.tree import DecisionTreeClassifier  # 局部 import：core 导入期不拉起 sklearn 树模块

    if values.size < 2 or np.unique(values).size < 2:
        return np.array([], dtype=float)
    tree = DecisionTreeClassifier(
        criterion=spec.criterion,
        max_depth=spec.max_depth,
        max_leaf_nodes=spec.max_bins,
        min_samples_leaf=max(1, min_leaf),
        random_state=spec.random_state,
    )
    tree.fit(values.reshape(-1, 1), y.astype(int))
    thresholds = tree.tree_.threshold
    edges = np.unique(thresholds[thresholds != -2.0])
    return np.sort(edges.astype(float))


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def fit_bins_dt(
    x: pd.Series,
    y: pd.Series | np.ndarray,
    spec: DTBinningSpec | None = None,
) -> DTBinResult:
    """在 ``(x, y)`` 上拟合 DT 分箱。

    Parameters
    ----------
    x
        待分箱特征（``pd.Series``）。非数值取值按 ``NaN`` 处理（计入缺失箱）。
    y
        二分类标签（0/1），与 ``x`` 等长同序。
    spec
        :class:`DTBinningSpec`；``None`` 用默认值（5 箱 / 5% / 50 样本 / 单调）。

    Returns
    -------
    DTBinResult
        ``result.spec`` 可直接用于 ``transform_bins`` / ``bin_table_processed``。

    Raises
    ------
    DataError
        标签非 0/1、长度不一致、全部缺失等。
    ConfigError
        参数自相矛盾（见 :class:`DTBinningSpec`）。

    Notes
    -----
    ★ ``x`` 必须是 **train 集**（防穿越）。本函数不做切分 —— 与 ``fit_bins`` 同一约定。
    """
    spec = spec or DTBinningSpec()
    values = _as_numeric(x)
    yv = _as_binary_y(y)
    if values.size != yv.size:
        raise DataError(
            f"特征与标签长度不一致：{values.size} vs {yv.size}"
        )

    notes: list[str] = []
    finite = np.isfinite(values)
    missing_count = int((~finite).sum())
    xv, yvm = values[finite], yv[finite]
    n_total = int(xv.size)

    if n_total == 0:
        raise DataError(f"特征 {getattr(x, 'name', '')!r} 全部缺失，无法进行 DT 分箱")
    bad_total = float(yvm.sum())
    good_total = float(n_total - bad_total)
    if bad_total <= 0 or good_total <= 0:
        raise DataError(
            f"DT 分箱要求好坏样本同时存在，当前 bad={int(bad_total)}, good={int(good_total)}"
        )

    # 树的叶节点最小样本：与后处理同一个尺子（取严）
    min_leaf = spec.min_samples_leaf_override or max(
        int(spec.min_bin_count), int(np.ceil(spec.min_bin_pct * n_total)), 1
    )
    min_leaf = min(min_leaf, max(1, n_total // 2))

    edges = _tree_edges(xv, yvm, spec, min_leaf)
    if edges.size == 0:
        notes.append(
            "树未能切出任何切点（取值恒定或样本量不足）⇒ 退化为单箱，"
            "IV 在该特征上不可用"
        )

    counts, bads = _bin_counts(xv, yvm, edges)
    # ★ 保留空箱（count=0）：它在「最小箱约束」里会被立即合并掉，
    #   而先删箱再重建切点会让「切点 ↔ 箱」的对应关系更难核对。
    # ``hi`` 记录每箱的右边界（= 初始切点或 +inf），合并时只吃掉中间那条切点。
    bounds = np.concatenate([edges, [np.inf]]).astype(float) if edges.size else (
        np.array([np.inf], dtype=float)
    )
    bins = [
        {"count": int(c), "bad": float(b), "hi": float(bounds[i])}
        for i, (c, b) in enumerate(zip(counts, bads))
    ]

    # ① 箱数上限
    if len(bins) > spec.max_bins:
        while len(bins) > spec.max_bins:
            i = min(
                range(len(bins) - 1),
                key=lambda j: bins[j]["count"] + bins[j + 1]["count"],
            )
            _merge(bins, i)
        notes.append(f"箱数收敛到上限 {spec.max_bins}")

    # ② 最小占比/样本数
    _enforce_min_share(bins, n_total, spec.min_bin_count, spec.min_bin_pct, notes)

    # ③ 单调性
    rates = [_bad_rate(b) for b in bins]
    direction = "flat"
    if len(rates) >= 2:
        if rates[-1] > rates[0] + 1e-12:
            direction = "increasing"
        elif rates[-1] < rates[0] - 1e-12:
            direction = "decreasing"
    if spec.monotonic:
        _enforce_monotonic(bins, direction, notes)

    final_counts = tuple(int(b["count"]) for b in bins)
    final_rates = tuple(_bad_rate(b) for b in bins)
    is_mono = _is_monotonic_rates(final_rates, direction) if direction != "flat" else True

    # 切点 = 剩余箱的右边界（最后一箱是 +inf，不作为切点）
    final_edges = tuple(float(b["hi"]) for b in bins[:-1])

    if len(final_edges) + 1 != len(final_counts):
        # 理论上不会发生；发生了说明切点与箱不对应 ⇒ 显式报错而不是产出错口径
        raise DataError(
            f"DT 分箱内部不一致：切点 {len(final_edges)} 个但箱 {len(final_counts)} 个"
        )

    bin_spec = BinSpec(
        method="business_cut",
        n_bins=len(final_counts),
        edges=tuple(float(e) for e in final_edges),
        missing_as_bin=spec.missing_as_bin,
        min_bin_count=spec.min_bin_count,
        min_bin_pct=spec.min_bin_pct,
        requested_bins=spec.max_bins,
        adaptive_bins=False,
        sparse_strategy="merge_adjacent",
    )

    return DTBinResult(
        spec=bin_spec,
        edges=tuple(float(e) for e in final_edges),
        bin_counts=final_counts,
        bin_bad_rates=final_rates,
        missing_count=missing_count if spec.missing_as_bin else 0,
        total_count=n_total + (missing_count if spec.missing_as_bin else 0),
        is_monotonic=is_mono,
        monotonic_direction=direction,
        notes=tuple(notes),
    )


def dt_bin_table(result: DTBinResult, x: pd.Series, y: pd.Series) -> pd.DataFrame:
    """把 DT 分箱结果落成「箱 → 样本数/坏样本率」表（用于报表与核对）。

    与 :func:`~risklab_mdl.core.binning.bin_table_processed` 的区别：
    本函数**不重算 WOE/IV**，只用于核对「拟合时的箱」与「变换后的箱」是否一致。
    """
    from .binning import bin_table, transform_bins  # 局部 import 避免循环依赖

    codes = transform_bins(x, result.spec)
    table = bin_table(codes, y)
    return table
