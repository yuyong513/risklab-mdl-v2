"""分箱 / WOE / IV / 分箱表 PSI。

★ 稀疏箱先合并、再做归一化平滑；IV 读数依赖箱数，
   因此逐列透出（请求箱数, 实际箱数）作为参照系。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

from .errors import DataError

__all__ = [
    "BinMethod",
    "BinSpec",
    "SparseStrategy",
    "bin_psi",
    "bin_table",
    "bin_table_processed",
    "fit_bins",
    "iv_of",
    "transform_bins",
]

BinMethod = Literal["quantile", "equal_width", "business_cut", "unsupervised", "none"]
SparseStrategy = Literal["merge_adjacent", "smooth", "keep"]

MISSING_BIN: int = -1
"""缺失箱的统一箱编码。全项目唯一约定。"""

TOTAL_BINS_FALLBACK: int = 2
"""退化后的最小箱数。取值过于集中时实际箱数会少于请求值，属预期行为。"""


@dataclass(frozen=True)
class BinSpec:
    """分箱规格。可作为策略文件的一部分由使用方声明。

    典型取值组合：等频 8 箱 / 切点仅在 train 集拟合并缓存 / 最小箱 50 或 0.5% /
    稀疏箱合并相邻 / 平滑系数 0.5。

    F-07 相关字段
    --------------
    ``adaptive_bins``
        是否按样本量自动下调箱数（F-07 修复 A）。默认开启 —— 样本量撑不起
        请求的箱数时，下调比「全箱判稀疏 → 塌成 1 箱 → IV 恒为 0」正确得多。
    ``requested_bins``
        :func:`fit_bins` 实际使用的箱数。**当它小于请求值时，说明发生了下调**
        （样本量不足），报表应显式提示 —— 否则使用者会以为 IV 是在
        请求的箱数上算的，与历史口径不可比。
    """

    method: BinMethod = "quantile"
    n_bins: int = 8
    edges: tuple[float, ...] | None = None          # method="business_cut" 时必填
    fit_on_months: tuple[str, ...] | None = None    # 仅在这些月份上拟合（防穿越）
    missing_as_bin: bool = True                     # 缺失是否单独成箱
    min_bin_count: int = 50                         # 稀疏箱合并的样本数下限
    min_bin_pct: float = 0.005                      # 稀疏箱合并的占比下限
    sparse_strategy: SparseStrategy = "merge_adjacent"
    woe_smoothing: float = 0.5                      # L-02：归一化 0.5 平滑
    clip_extreme: bool = False                      # 是否先截尾再分箱
    extreme_quantile: float = 0.001
    adaptive_bins: bool = True                      # F-07-A：按样本量下调箱数
    requested_bins: int = 0                         # F-07：实际使用的箱数（0=未拟合）

    @property
    def n_effective_bins(self) -> int:
        """实际箱数（含缺失箱）。重复值多时会少于 ``n_bins``。"""
        if self.method == "none":
            return 2
        return len(self.edges) + 2 if self.edges else 2

    @property
    def n_valid_bins(self) -> int:
        """实际有效箱数（不含缺失箱）。"""
        if self.method == "none":
            return 1
        return len(self.edges) + 1 if self.edges else 1

    @property
    def bins_downgraded(self) -> bool:
        """F-07：箱数是否因样本量不足而被下调。"""
        return bool(self.requested_bins) and self.n_valid_bins < self.requested_bins


# --------------------------------------------------------------------------- #
# 切点拟合
# --------------------------------------------------------------------------- #
def _business_edges(spec: BinSpec) -> np.ndarray:
    if not spec.edges:
        raise DataError("method='business_cut' 时必须提供 edges")
    edges = np.asarray(sorted(float(e) for e in spec.edges), dtype=float)
    if edges.size < 1:
        raise DataError("edges 至少需要一个切点")
    return edges


def _quantile_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    """等频切点。

    重复值多时 ``np.unique`` 会自动收敛切点数量 —— 这是**正确行为**：
    强行凑够 ``n_bins`` 会产生多个空箱，而空箱正是 L-02 的问题源头。
    真实箱数由收敛后的切点数决定，并在 ``BinSpec.n_effective_bins`` 中体现。
    """
    if n_bins < 2:
        raise DataError(f"n_bins 至少为 2，当前 {n_bins}")
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.unique(np.quantile(x, qs))


def _equal_width_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins < 2:
        raise DataError(f"n_bins 至少为 2，当前 {n_bins}")
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:
        raise DataError("特征取值恒定，无法等宽分箱")
    return np.linspace(lo, hi, n_bins + 1)[1:-1]


def _affordable_bins(n_samples: int, requested: int, min_bin_count: int) -> int:
    """F-07-A：样本量能支撑的最大箱数。

    稀疏箱合并的下限是「每箱至少 ``min_bin_count`` 个样本」，
    因此 ``k`` 个箱至少需要 ``k × min_bin_count`` 个样本::

        k_max = n_samples // min_bin_count

    再与请求箱数取小。返回至少 2 —— 退化成单箱会让 IV 恒为 0（无区分度），
    宁可箱少也要保留「能算出分布差异」的能力。
    """
    if min_bin_count <= 0:
        return max(2, requested)
    affordable = max(2, n_samples // min_bin_count)
    return max(2, min(requested, affordable))


def fit_bins(x: pd.Series, spec: BinSpec) -> BinSpec:
    """在给定数据上拟合切点，返回补齐 ``edges`` 的 BinSpec。

    防穿越约束（对应需求 I-15）
    -------------------------
    调用方**必须**保证传入的 ``x`` 是 train 集（或 ``fit_on_months`` 指定的月份）
    上的取值。本函数不做数据切分，只做切点拟合 —— 它无法知道自己拿到的是
    全量还是 train，这是编排层的责任。

    注意 ``None`` 入参会被静默忽略。``fit_on_months`` 的落地由编排层负责：
    在调用前用月份掩码过滤 ``x``。

    F-07：箱数自适应
    ----------------
    请求的箱数超过样本量能支撑的箱数时**主动下调**（``adaptive_bins=True``），
    下调后的事实记录在 ``requested_bins`` 与 ``bins_downgraded`` 里。

    为什么必须在这里处理
    --------------------
    ``min_bin_count``（每箱最少样本数）与 ``n_bins``（切几份）在样本量小时
    不可同时满足。若不在这里下调，后续稀疏合并会把**所有**箱判为稀疏并
    一路合并到只剩 1 箱 —— 此时 IV 在数学上恒等于 0（单箱内好/坏分布相同），
    初筛会把所有特征误判为「无区分度」。这是静默失效：既不报错，
    产出的「IV=0.0000」看起来也像合法读数。
    """
    if not isinstance(x, pd.Series):
        raise DataError(f"fit_bins 需要 pd.Series，收到 {type(x).__name__}")

    values = x.dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise DataError(f"特征 {x.name} 无有效取值，无法分箱")

    if spec.clip_extreme:
        lo, hi = np.quantile(values, [spec.extreme_quantile, 1 - spec.extreme_quantile])
        values = np.clip(values, lo, hi)

    # F-07-A：按样本量收敛箱数
    requested = spec.n_bins
    n_bins = requested
    if spec.adaptive_bins and spec.method in ("quantile", "equal_width"):
        n_bins = _affordable_bins(values.size, requested, spec.min_bin_count)

    if spec.method == "business_cut":
        edges = _business_edges(spec)
    elif spec.method == "quantile":
        edges = _quantile_edges(values, n_bins)
    elif spec.method == "equal_width":
        edges = _equal_width_edges(values, n_bins)
    elif spec.method in ("unsupervised", "none"):
        edges = np.array([], dtype=float)
    else:
        raise DataError(f"不支持的分箱方式：{spec.method}")

    if edges.size + 1 < TOTAL_BINS_FALLBACK:
        # 取值几乎恒定：退化为单箱 + 缺失箱，由上层在报告中提示
        edges = np.array([], dtype=float)

    # F-07：记录「请求的箱数」与「实际使用的箱数」。
    # requested_bins 记的是**请求值**（不是实际值）——
    # 因为 edges 数量还受重复值收敛影响，把它当成「请求值」会掩盖下调事实。
    return replace(
        spec,
        edges=tuple(float(e) for e in edges),
        requested_bins=requested,
    )


# --------------------------------------------------------------------------- #
# 变换
# --------------------------------------------------------------------------- #
def transform_bins(x: pd.Series, spec: BinSpec) -> pd.Series:
    """按 BinSpec 把原始取值映射为箱编码。

    箱编码约定：``0..k-1`` 为有效箱（按取值升序），``MISSING_BIN``(-1) 为缺失箱。

    缺失值的处理
    ------------
    * ``missing_as_bin=True`` → 缺失映射为 ``-1``，参与后续 WOE/IV 计算
      （缺失本身携带信息：「无还款记录」与「还款 0 元」是两回事）；
    * ``missing_as_bin=False`` → 缺失映射为 ``NA``，由调用方另行处置
      （填充 / 剔除）。
    """
    if spec.method == "none":
        return pd.Series(0, index=x.index, name=f"{x.name}__bin", dtype="int64")

    edges = np.asarray(spec.edges if spec.edges is not None else (), dtype=float)
    if edges.size:
        full = np.concatenate([[-np.inf], edges, [np.inf]])
    else:
        full = np.array([-np.inf, np.inf])

    codes = pd.cut(x, bins=full, labels=False, include_lowest=True, right=True)
    out = pd.Series(codes, index=x.index, name=f"{x.name}__bin", dtype="float64")
    if spec.missing_as_bin:
        return out.fillna(MISSING_BIN).astype("int64")
    # 缺失保留为 NA —— 用 pandas 可空整数，不能 astype("int64")
    return out.astype("Int64")


# --------------------------------------------------------------------------- #
# 分箱表 / 稀疏箱合并 / WOE-IV
# --------------------------------------------------------------------------- #
def bin_table(codes: pd.Series, y: pd.Series) -> pd.DataFrame:
    """生成原始分箱表（尚未做稀疏合并与 WOE）。

    输出列：``bin`` / 总用户 / 坏用户 / 好用户 / 坏样本率 / 占比
    """
    if codes.index.size != y.index.size:
        raise DataError("箱编码与标签长度不一致")
    df = pd.DataFrame({"bin": codes.to_numpy(), "y": np.asarray(y, dtype=float)})
    df = df[df["bin"].notna()]
    df["bin"] = df["bin"].astype("int64")

    grouped = (
        df.groupby("bin", as_index=False)
        .agg(总用户=("y", "size"), 坏用户=("y", "sum"))
        .sort_values("bin")
        .reset_index(drop=True)
    )
    grouped["好用户"] = grouped["总用户"] - grouped["坏用户"]
    grouped["坏样本率"] = grouped["坏用户"] / grouped["总用户"]
    grouped["占比"] = grouped["总用户"] / grouped["总用户"].sum()
    return grouped


def _merge_sparse_bins(
    table: pd.DataFrame, min_count: int, min_pct: float
) -> pd.DataFrame:
    """把样本量过小的**有效箱**合并到相邻箱。

    合并只在有效箱（``bin >= 0``）之间进行；**缺失箱不参与合并** ——
    「无还款记录」与「还款 0 元」业务含义不同，不能因为样本少就并进去。

    F-07：不得塌成单箱
    ------------------
    合并逻辑是「只要有稀疏箱就继续合并」，在样本量偏小时会一路合并到
    只剩 1 箱 —— 此时 IV 在数学上**恒等于 0**（单箱内好/坏分布完全一致），
    特征会被静默判为「无区分度」。

    因此这里加一条兜底保护：**合并后的有效箱数不得少于
    ``max(2, total // min_count)``**。

    这个判据与 :func:`_affordable_bins`（F-07-A）**保持一致** ——
    「样本量能支撑几个箱」在切点拟合与稀疏合并两处必须用同一把尺子，
    否则会出现「A 处说能切 2 箱、B 处却合并成 1 箱」的矛盾。

    * 样本量够支撑 2 箱 → 至少留 2 箱（IV 才算得出分布差异）；
    * 样本量连 2 箱都撑不起（真的极小，如 n=60）→ 也留 2 箱，
      但此时箱内样本极少、IV 读数不稳定 —— 由 ``fit_bins`` 记录箱数下调事实，
      再由 ``screen_iv`` 的 notes 显式告警（**告警，不假装读数可靠**）。
    """
    df = table.copy().reset_index(drop=True)
    total = int(df["总用户"].sum())
    if total <= 0:
        return df

    # F-07-B：塌陷下限。与 _affordable_bins 同一判据。
    min_valid_bins = max(2, total // max(min_count, 1))

    while True:
        valid_mask = df["bin"] >= 0
        n_valid = int(valid_mask.sum())
        if n_valid <= min_valid_bins:
            break
        small = df[valid_mask & (
            (df["总用户"] < min_count) | (df["总用户"] / total < min_pct)
        )]
        if small.empty:
            break

        pos = int(small.index[0])
        left_idx = pos - 1 if pos - 1 >= 0 and df.at[pos - 1, "bin"] >= 0 else None
        right_idx = pos + 1 if pos + 1 < len(df) and df.at[pos + 1, "bin"] >= 0 else None
        if left_idx is None and right_idx is None:
            break
        if left_idx is None:
            target = right_idx
        elif right_idx is None:
            target = left_idx
        else:
            target = (
                left_idx
                if df.at[left_idx, "总用户"] >= df.at[right_idx, "总用户"]
                else right_idx
            )
        assert target is not None
        df.at[target, "总用户"] += df.at[pos, "总用户"]
        df.at[target, "坏用户"] += df.at[pos, "坏用户"]
        df = df.drop(index=pos).reset_index(drop=True)

    df["好用户"] = df["总用户"] - df["坏用户"]
    df["坏样本率"] = df["坏用户"] / df["总用户"]
    df["占比"] = df["总用户"] / df["总用户"].sum()
    return df.reset_index(drop=True)


def bin_table_processed(
    codes: pd.Series, y: pd.Series, spec: BinSpec
) -> pd.DataFrame:
    """完整链路：原始分箱表 → 稀疏箱合并 → WOE / IV。

    输出列：``bin`` / 总用户 / 坏用户 / 好用户 / 坏样本率 / 占比 / WOE / IV贡献 / 是否平滑

    L-02 的修复就在这里：先合并稀疏箱，再做**归一化**平滑。
    """
    table = bin_table(codes, y)
    if spec.sparse_strategy == "merge_adjacent":
        table = _merge_sparse_bins(table, spec.min_bin_count, spec.min_bin_pct)

    total_bad = float(table["坏用户"].sum())
    total_good = float(table["好用户"].sum())
    if total_bad <= 0 or total_good <= 0:
        raise DataError(
            "分箱结果中好样本或坏样本为 0，无法计算 WOE/IV"
            "（该特征在全样本或当前分箱下无区分度）"
        )

    n_bins = len(table)
    # L-02：归一化平滑。分母 δ * n_bins 保证各箱占比之和仍为 1。
    # sparse_strategy="keep" 时不平滑（δ=0），用于复现旧口径做对比。
    delta = spec.woe_smoothing if spec.sparse_strategy != "keep" else 0.0
    bad_pct = (table["坏用户"] + delta) / (total_bad + delta * n_bins)
    good_pct = (table["好用户"] + delta) / (total_good + delta * n_bins)

    out = table.copy()
    # delta=0（sparse_strategy="keep"）且存在全好/全坏箱时占比可能为 0，
    # 直接 log 会产生 -inf 与 RuntimeWarning。用 EPS 兜底并在「是否平滑」中标记，
    # 让报表能识别出这批箱的 WOE 不可信。
    with np.errstate(divide="ignore", invalid="ignore"):
        woe = np.log(bad_pct / good_pct)
    out["WOE"] = np.nan_to_num(woe, nan=0.0, posinf=0.0, neginf=0.0)
    out["WOE是否截断"] = ~np.isfinite(woe)
    out["IV贡献"] = (bad_pct - good_pct) * out["WOE"]
    out["是否平滑"] = (out["坏用户"] == 0) | (out["好用户"] == 0)
    return out.reset_index(drop=True)


def iv_of(table_with_woe: pd.DataFrame) -> float:
    """IV = Σ (坏占比 - 好占比) × WOE。"""
    if "IV贡献" not in table_with_woe.columns:
        raise DataError("分箱表缺少 IV贡献 列，请先调用 bin_table_processed")
    return float(table_with_woe["IV贡献"].sum())


def bin_psi(
    base_table: pd.DataFrame, test_table: pd.DataFrame, eps: float = 1e-6
) -> float:
    """基于**同一套箱**的两表 PSI。

    相对旧实现修正了三点：

    1. 缺失箱参与对比（旧实现用 -99 / -9999 哨兵值，缺失可能被当成正常取值）；
    2. test 中不存在的箱按占比 0 处理（旧实现直接跳过 → PSI 被低估）；
    3. 分母统一加 ``eps`` 防除零。
    """
    for name, table in (("base_table", base_table), ("test_table", test_table)):
        if not {"bin", "占比"} <= set(table.columns):
            raise DataError(f"{name} 需包含 bin 与 占比 列")

    base = base_table.set_index("bin")["占比"]
    test = test_table.set_index("bin")["占比"]
    idx = base.index.union(test.index)
    base = base.reindex(idx, fill_value=0.0).astype(float)
    test = test.reindex(idx, fill_value=0.0).astype(float)

    return float(((test - base) * np.log((test + eps) / (base + eps))).sum())