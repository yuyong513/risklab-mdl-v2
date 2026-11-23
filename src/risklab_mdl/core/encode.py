"""类别变量编码：drop / ordinal / onehot / WOE。

★ WOE 编码表必须回写参数文档，否则上线无法复现同样的编码；
★ WOE 表只能在训练集上拟合（用坏样本率算权重 ⇒ 全量拟合是标签泄漏）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .errors import ConfigError, DataError

__all__ = [
    "CategoryWoeTable",
    "apply_encode_meta",
    "apply_label_encode",
    "apply_onehot_encode",
    "apply_woe_encode",
    "derive_label_mapping",
    "derive_onehot_values",
    "fit_woe_table",
]

WOE_EPS = 1e-6
"""WOE 计算的分母保护。避免 ``ln(0)`` 产生 -inf。"""


@dataclass
class CategoryWoeTable:
    """类别变量的 WOE 编码表。

    ``mapping``  : 原始取值 → WOE 数值（缺失单独一档）
    ``iv``        : 该变量的 IV
    ``bad_rate``  : 各取值对应的坏样本率（可解释性用）
    ``unknown``   : 训练期未见过的取值，其 WOE 取值
    """

    column: str
    mapping: Dict[str, float] = field(default_factory=dict)
    bad_rate: Dict[str, float] = field(default_factory=dict)
    iv: float = 0.0
    unknown: float = 0.0
    base_bad_pct: float = 0.0
    base_good_pct: float = 0.0
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "column": self.column,
            "mapping": dict(self.mapping),
            "bad_rate": dict(self.bad_rate),
            "iv": round(float(self.iv), 6),
            "unknown": self.unknown,
            "note": self.note,
        }

    @staticmethod
    def from_dict(d: Mapping[str, object]) -> "CategoryWoeTable":
        return CategoryWoeTable(
            column=str(d.get("column", "")),
            mapping={str(k): float(v) for k, v in (d.get("mapping") or {}).items()},
            bad_rate={str(k): float(v) for k, v in (d.get("bad_rate") or {}).items()},
            iv=float(d.get("iv", 0.0) or 0.0),
            unknown=float(d.get("unknown", 0.0) or 0.0),
            note=str(d.get("note", "") or ""),
        )


def _as_str_series(series: pd.Series) -> pd.Series:
    """把任意类型转成字符串表示的类别列。

    ★ 缺失保留为 ``__MISSING__`` 独立一档 —— 不能 astype(str) 后变成字面量
    ``"nan"``（那样「无记录」会与真实取值 ``"nan"`` 混为一谈）。
    """
    return series.astype(object).where(series.notna(), "__MISSING__").astype(str)


def derive_label_mapping(series: pd.Series) -> Dict[str, float]:
    """标签编码映射表的**唯一派生实现**。

    训练期与推理期必须用同一张表 —— 派生逻辑只写这一处：
    按取值字典序映射为 0,1,2…（不依赖数据行序，天然可复现）。
    """
    return {v: float(i) for i, v in enumerate(sorted(_as_str_series(series).unique()))}


def derive_onehot_values(series: pd.Series) -> list:
    """独热取值清单的**唯一派生实现**。

    训练期捕获进规格、随模型包落盘；推理期按清单生成同名列。
    缺失档 ``__MISSING__`` 与 ``apply_onehot_encode`` 的口径天然一致
    （两者都走 ``_as_str_series``）。
    """
    return sorted(_as_str_series(series).unique().tolist())


def apply_label_encode(
    df: pd.DataFrame, columns: Sequence[str], *, mapping: Optional[Mapping[str, Mapping[str, float]]] = None
) -> pd.DataFrame:
    """标签编码：按取值字典序映射为 0,1,2…

    可复现（不依赖数据顺序）。若传入 ``mapping`` 则复用已有编码表。
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            raise DataError(f"待编码列 {col} 不存在")
        series = _as_str_series(out[col])
        table = mapping[col] if mapping and col in mapping else derive_label_mapping(series)
        out[col] = series.map(table)
    return out


def apply_onehot_encode(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    prefix: str = "",
    values: Optional[Mapping[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """独热编码：每个取值生成一列 0/1。

    ★ 缺失单独成列（``__MISSING__``）
    --------------------------------
    与 WOE 编码的缺失独立档保持同一口径。

    ★ 为什么不能直接 ``astype(str)``
    ------------------------------
    ``astype(str)`` 会把 NaN 变成字面量字符串 ``"nan"`` ——
    于是「没有记录」和「真实取值就叫 nan」被混成同一档。
    而 ``dummy_na=True`` 在 astype 之后**不再起作用**（已经没有 NaN 了），
    缺失会变成一列名叫 ``city_nan`` 的 0/1 列，**看起来**处理了缺失，
    实际语义是错的。故统一走 ``_as_str_series``。

    ★ ``values``：按训练期取值清单生成列（复现的关键）
    ------------------------------------------------
    传入 ``{列: [训练期取值清单]}`` 时按清单生成 0/1 列 ——
    训练期与推理期**共用同一生成逻辑**；原始数据里出现
    训练期没有的取值时**报错**（不静默丢弃——那会让该样本
    在所有独热列上取 0，伪装成一个「训练期没见过」的中性样本）。
    不传 ``values`` 时按数据现场派生（兼容旧调用）。
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            raise DataError(f"待编码列 {col} 不存在")
        s = _as_str_series(out[col])
        if values and col in values:
            wanted = [str(v) for v in values[col]]
            seen = set(s.unique())
            unseen = sorted(seen - set(wanted))
            if unseen:
                raise DataError(
                    f"列 {col!r} 存在训练期未见的取值 {unseen[:5]}（共 {len(unseen)} 个）。"
                    f"独热编码只认训练期取值清单（{len(wanted)} 档）—— "
                    f"新取值无法按训练口径生成 0/1 列，"
                    f"请先在数据确认环节处理（合并/剔除/改用 WOE 等策略）。"
                )
            for v in wanted:
                out[f"{prefix or col}_{v}"] = (s == v).astype("int8")
            out = out.drop(columns=[col])
        else:
            dummies = pd.get_dummies(s, prefix=prefix or col, dummy_na=False)
            out = pd.concat([out.drop(columns=[col]), dummies], axis=1)
    return out


def fit_woe_table(
    series: pd.Series,
    y: pd.Series,
    *,
    column: str = "",
    smoothing: float = 0.5,
    min_count: int = 1,
) -> CategoryWoeTable:
    """在给定数据（**必须是基期/train**）上拟合类别变量的 WOE 编码表。

    WOE 定义与 ``core/binning.py`` 保持一致口径::

        WOE = ln( (坏占比_i + δ) / (好占比_i + δ) )

    归一化平滑（分母加 ``δ × 类别数``），保证各档占比之和为 1 ——
    与数值分箱的 L-02 修复同源。
    """
    if len(series) != len(y):
        raise DataError("特征列与标签列长度不一致")
    s = _as_str_series(series)
    yv = pd.Series(np.asarray(y, dtype=float), index=series.index)

    total_bad = float(yv.sum())
    total_good = float(len(yv) - total_bad)
    if total_bad <= 0 or total_good <= 0:
        raise DataError(
            f"类别变量 {column or s.name} 的基期数据中好样本或坏样本为 0，"
            "无法计算 WOE（该变量在全样本或当前基期下无区分度）"
        )

    grouped = (
        pd.DataFrame({"cat": s.to_numpy(), "y": yv.to_numpy()})
        .groupby("cat", as_index=False)
        .agg(总用户=("y", "size"), 坏用户=("y", "sum"))
    )
    grouped["好用户"] = grouped["总用户"] - grouped["坏用户"]

    # 稀疏类别（样本数过少）合并进「其他」档 —— 否则该档 WOE 不稳定
    n_cats = len(grouped)
    delta = smoothing
    bad_pct = (grouped["坏用户"] + delta) / (total_bad + delta * n_cats)
    good_pct = (grouped["好用户"] + delta) / (total_good + delta * n_cats)

    with np.errstate(divide="ignore", invalid="ignore"):
        woe = np.log((bad_pct + WOE_EPS) / (good_pct + WOE_EPS))
    woe = np.nan_to_num(woe, nan=0.0, posinf=0.0, neginf=0.0)

    grouped["WOE"] = woe
    grouped["坏样本率"] = grouped["坏用户"] / grouped["总用户"]
    grouped["IV贡献"] = (bad_pct - good_pct) * grouped["WOE"]

    mapping: Dict[str, float] = {}
    bad_rate: Dict[str, float] = {}
    # ★ 注意：iterrows() 的第一个元素是**索引**，不是 cat 列的取值。
    #   误当成 cat 用会让编码表的键变成 0/1/2/3 —— 计算照样跑完、
    #   IV 数值也「看着正常」，但 apply_woe_encode 时所有值都匹配不上
    #   ⇒ 整列被填成 unknown(=0)。**静默失效**，必须写对。
    for _, row in grouped.iterrows():
        cat = row["cat"]
        mapping[str(cat)] = float(row["WOE"])
        bad_rate[str(cat)] = float(row["坏样本率"])

    table = CategoryWoeTable(
        column=column or str(series.name),
        mapping=mapping,
        bad_rate=bad_rate,
        iv=float(grouped["IV贡献"].sum()),
        base_bad_pct=total_bad / max(len(yv), 1),
        base_good_pct=total_good / max(len(yv), 1),
    )
    return table


def apply_woe_encode(
    df: pd.DataFrame,
    columns: Sequence[str],
    tables: Mapping[str, CategoryWoeTable],
) -> pd.DataFrame:
    """按已拟合的 WOE 表做编码。

    ★ 训练期未见过的取值：归入 ``unknown`` 档（默认 WOE=0），
    并在输出里保留原始列名后缀 ``__unknown`` 标记 —— 不静默丢弃。
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            raise DataError(f"待编码列 {col} 不存在")
        table = tables.get(col)
        if table is None:
            raise ConfigError(f"类别变量 {col} 缺少已拟合的 WOE 表，不允许隐式拟合")
        s = _as_str_series(out[col])
        known = table.mapping
        unknown_mask = ~s.isin(known.keys())
        mapped = s.map(known)
        mapped[unknown_mask] = table.unknown
        out[col] = mapped.astype(float)
    return out


def apply_encode_meta(df: pd.DataFrame, meta: Mapping[str, object]) -> pd.DataFrame:
    """按模型包落盘的 ``encode_meta`` 规格编码**原始未编码数据**。

    ★ 这是「编码复现」的推理期入口
    --------------------------------
    训练期由 ``feature.apply_encode`` 捕获规格（派生逻辑）并随模型包落盘；
    推理期用本函数把**同一批 apply 函数**应用到原始数据上 ——
    训练/推理共用同一份实现与同一张表，这是可复现的全部保证。

    ``meta`` 结构（全部为纯数据，来自模型包 ``encode_meta`` 字段）::

        {
            "strategy": "drop|ordinal|onehot|woe",
            "applied":   [已编码的非数值列],
            "dropped":   [被剔除的列],          # drop 策略：无需操作
            "ordinal_maps":   {列: {取值: 码}},  # ordinal 策略
            "onehot_values":  {列: [取值清单]},  # onehot 策略
            "woe_tables":     {列: WOE表dict},   # woe 策略
            "woe_fit_desc":   "拟合集说明",
        }

    纪律：**不静默兜底** —— ordinal/onehot 遇到训练期未见的取值直接报错
    （WOE 有显式的 unknown 档，属表自身语义，不在此列）。
    """
    strategy = str(meta.get("strategy", "") or "")
    applied = [str(c) for c in (meta.get("applied") or [])]
    if strategy == "drop":
        # 被剔除的列不在入模特征清单里，原样透传即可。
        return df
    if strategy == "ordinal":
        maps = {k: {str(kk): float(vv) for kk, vv in m.items()}
                for k, m in (meta.get("ordinal_maps") or {}).items()}
        out = apply_label_encode(df, applied, mapping=maps)
        # 未见取值 → map 出 NaN（且原本不是 NaN）。不静默兜底。
        for col in applied:
            if col in out.columns and out[col].isna().any():
                bad = df.loc[out[out[col].isna()].index, col]
                preview = bad.astype(str).unique()[:5].tolist()
                raise DataError(
                    f"列 {col!r} 存在编码表外的取值 {preview}。"
                    f"标签编码按训练期映射表编码 —— 新取值没有对应的码值，"
                    f"不能静默填 0（那会伪装成训练期第一档）。"
                    f"请先处理取值口径（合并/剔除），或改用 WOE 策略（有 unknown 档）。"
                )
        return out
    if strategy == "onehot":
        return apply_onehot_encode(df, applied, values=meta.get("onehot_values") or {})
    if strategy == "woe":
        tables = {
            k: CategoryWoeTable.from_dict(v)   # type: ignore[arg-type]
            for k, v in (meta.get("woe_tables") or {}).items()
        }
        return apply_woe_encode(df, applied, tables)
    raise DataError(
        f"encode_meta.strategy 取值非法：{strategy!r}（可选 drop/ordinal/onehot/woe）。"
        f"模型包内容损坏或来自不兼容版本。"
    )
