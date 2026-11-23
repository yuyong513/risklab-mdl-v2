"""相关性去重：贪心（默认）与连通分量两种策略。

★ 默认贪心：按重要度降序保留，与已保留特征 |corr| 超阈值即剔除。
★ 连通分量策略只在星型相关结构下可用，链型结构会大规模误杀。
"""

from __future__ import annotations

from typing import Literal, Mapping, Sequence

import pandas as pd

from .errors import DataError

__all__ = [
    "CorrStrategy",
    "DecorrelateResult",
    "component_decorrelate",
    "decorrelate",
    "greedy_decorrelate",
]

CorrStrategy = Literal["greedy", "component"]


class DecorrelateResult:
    """去相关结果。``dropped_reason`` 逐条给出剔除原因，供过程报表留痕。"""

    __slots__ = ("kept", "dropped_reason", "strategy")

    def __init__(
        self,
        kept: list[str],
        dropped_reason: dict[str, str],
        strategy: CorrStrategy,
    ) -> None:
        self.kept = kept
        self.dropped_reason = dropped_reason
        self.strategy = strategy

    @property
    def dropped(self) -> list[str]:
        return list(self.dropped_reason)

    def __iter__(self):
        """兼容旧接口的 ``kept, dropped = decorrelate(...)`` 用法。"""
        return iter((self.kept, self.dropped_reason))

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"DecorrelateResult(strategy={self.strategy!r}, "
            f"kept={len(self.kept)}, dropped={len(self.dropped_reason)})"
        )


def _abs_corr(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """取指定列的绝对相关矩阵；对角线置 1，NaN 置 0。"""
    if not features:
        raise DataError("去重特征列表为空")
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise DataError(f"去重特征在数据中不存在：{missing[:10]}")
    cols = list(features)
    # 不用 np.fill_diagonal(corr.values, ...)：pandas 3.x 下 .values 可能是只读视图
    corr = df[cols].corr().abs()
    for c in cols:
        corr.loc[c, c] = 1.0
    return corr.fillna(0.0)


def _order_by_importance(
    features: Sequence[str], importance: Mapping[str, float]
) -> list[str]:
    """按重要度降序；重要度缺失的特征排在最后（内部按列名稳定排序）。"""
    return sorted(features, key=lambda f: (-float(importance.get(f, 0.0)), str(f)))


def greedy_decorrelate(
    corr_abs: pd.DataFrame,
    importance: Mapping[str, float],
    threshold: float,
) -> DecorrelateResult:
    """贪心去相关：与已保留特征的最大 |corr| 超阈值即剔除。"""
    kept: list[str] = []
    dropped_reason: dict[str, str] = {}
    for feat in _order_by_importance(corr_abs.columns, importance):
        if not kept:
            kept.append(feat)
            continue
        best_partner = corr_abs.loc[feat, kept].idxmax()
        best_val = float(corr_abs.loc[feat, best_partner])
        if best_val > threshold:
            dropped_reason[feat] = (
                f"与已保留特征 {best_partner} 的 |corr|={best_val:.3f} "
                f"超过阈值 {threshold:g}"
            )
            continue
        kept.append(feat)
    return DecorrelateResult(kept, dropped_reason, "greedy")


def component_decorrelate(
    corr_abs: pd.DataFrame,
    importance: Mapping[str, float],
    threshold: float,
) -> DecorrelateResult:
    """连通分量策略（旧行为，作为可选开关保留，**默认不启用**）。

    纯并查集实现，不引入额外依赖。每个连通分量保留重要度最高的 1 个。

    启用前请确认数据的相关结构确实是**星型**而非**链型** ——
    链型下该策略会大规模误杀（这正是 L-01）。
    """
    parent: dict[str, str] = {c: c for c in corr_abs.columns}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # 路径压缩
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    cols = list(corr_abs.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if float(corr_abs.at[a, b]) > threshold:
                union(a, b)

    groups: dict[str, list[str]] = {}
    for c in cols:
        groups.setdefault(find(c), []).append(c)

    kept: list[str] = []
    dropped_reason: dict[str, str] = {}
    for root, members in groups.items():
        ordered = _order_by_importance(members, importance)
        best = ordered[0]
        kept.append(best)
        for f in ordered[1:]:
            dropped_reason[f] = (
                f"与 {best} 同属一个相关性连通分量（本分量共 {len(members)} 个特征）"
            )

    return DecorrelateResult(_order_by_importance(kept, importance),
                             dropped_reason, "component")


def decorrelate(
    df: pd.DataFrame,
    features: Sequence[str],
    importance: Mapping[str, float],
    *,
    strategy: CorrStrategy = "greedy",
    threshold: float = 0.90,
) -> DecorrelateResult:
    """统一去相关入口。

    参数
    ----
    df           数据框，须包含全部 ``features``
    features     待去重的特征列表（一般传初筛存活集）
    importance   特征重要度映射（通常传 IV；LR 场景可传 WOE 后的信息量）
    strategy     ``greedy``（默认） | ``component``（旧行为，星型结构下用）
    threshold    |corr| 阈值；数值越大保留越多

    Returns
    -------
    DecorrelateResult
        ``.kept`` / ``.dropped_reason``；也可直接解包为 ``kept, dropped``
        以兼容旧调用方式。
    """
    if not 0.0 < threshold <= 1.0:
        raise DataError(f"threshold 必须落在 (0, 1] 区间，当前 {threshold}")
    if strategy not in ("greedy", "component"):
        raise DataError(f"不支持的去相关策略：{strategy}")
    if not features:
        return DecorrelateResult([], {}, strategy)

    corr_abs = _abs_corr(df, features)
    fn = greedy_decorrelate if strategy == "greedy" else component_decorrelate
    return fn(corr_abs, importance, threshold)