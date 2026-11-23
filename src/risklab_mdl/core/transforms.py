"""特征画像与编码 / 填充 / 异常值规格。

★ 规格只在训练期拟合，禁止在全量数据上隐式拟合（标签泄漏）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .errors import ConfigError, DataError

__all__ = [
    "EncodeSpec",
    "EncodeStrategy",
    "FeatureProfile",
    "FeatureProfileTable",
    "FillSpec",
    "FillStrategy",
    "OutlierSpec",
    "OutlierStrategy",
    "PrepareAction",
    "PrepareReport",
    "SENTINEL_FILL_VALUE",
    "apply_encode",
    "apply_fill",
    "apply_outlier",
    "profile_features",
]

SENTINEL_FILL_VALUE: float = 99999.0
"""缺失默认哨兵值。业务上不可能出现的极大值。"""

EncodeStrategy = Literal["ordinal", "onehot", "target", "binned", "drop"]
FillStrategy = Literal[
    "sentinel", "mean", "median", "mode", "ffill", "group", "as_bin", "drop"
]
OutlierStrategy = Literal[
    "mark", "winsorize", "clip", "to_missing", "drop_column", "drop_rows"
]

# 需要「已拟合编码表」的策略 —— 不允许在预处理阶段隐式拟合（防泄漏 + 可复现）
_NEEDS_FITTED_TABLE: frozenset[str] = frozenset({"target", "binned"})


@dataclass(frozen=True)
class EncodeSpec:
    """非数值编码策略。"""

    column: str
    strategy: EncodeStrategy = "ordinal"
    mapping: Mapping[str, float] | None = None    # ordinal 用的 取值→编码
    note: str = ""


@dataclass(frozen=True)
class FillSpec:
    """缺失填充策略。"""

    column: str
    strategy: FillStrategy = "sentinel"
    value: float = SENTINEL_FILL_VALUE
    group_by: str = ""          # strategy="group" 时的分组列
    note: str = ""


@dataclass(frozen=True)
class OutlierSpec:
    """异常值处置策略。"""

    column: str
    strategy: OutlierStrategy = "mark"
    sigma_multiple: float = 3.0
    iqr_factor: float = 1.5
    iqr_ratio: float = 0.01
    note: str = ""


@dataclass(frozen=True)
class FeatureProfile:
    """单列的预处理画像。"""

    column: str
    is_numeric: bool
    n_unique: int
    missing_rate: float
    top_value_share: float
    value_samples: tuple[str, ...] = ()
    sigma_deviation: float | None = None
    iqr_outlier_ratio: float | None = None
    has_conflicting_sentinel: bool = False
    flag: str = ""

    @property
    def needs_encode(self) -> bool:
        """非数值列才需要编码。"""
        return not self.is_numeric

    @property
    def needs_fill(self) -> bool:
        return self.missing_rate > 0


@dataclass
class FeatureProfileTable:
    """待预处理特征统计表。呈现给使用方做策略选择，并随报告留痕。"""

    frame: pd.DataFrame
    profiles: Mapping[str, FeatureProfile] = field(default_factory=dict)
    encode_specs: Mapping[str, EncodeSpec] = field(default_factory=dict)
    fill_specs: Mapping[str, FillSpec] = field(default_factory=dict)
    outlier_specs: Mapping[str, OutlierSpec] = field(default_factory=dict)

    def restricted_columns(self, keywords: Sequence[str]) -> list[str]:
        """返回命中合规敏感词清单的列。服务只做命中提示，不自建合规标准。"""
        hits: list[str] = []
        for col in self.frame["列名"].astype(str):
            low = col.lower()
            if any(k.lower() in low for k in keywords):
                hits.append(col)
        return hits

    def with_strategy_overrides(
        self,
        *,
        encode: Mapping[str, EncodeSpec] | None = None,
        fill: Mapping[str, FillSpec] | None = None,
        outlier: Mapping[str, OutlierSpec] | None = None,
    ) -> "FeatureProfileTable":
        """返回策略覆盖后的新表（不修改自身）。"""
        return FeatureProfileTable(
            frame=self.frame,
            profiles=self.profiles,
            encode_specs={**self.encode_specs, **(encode or {})},
            fill_specs={**self.fill_specs, **(fill or {})},
            outlier_specs={**self.outlier_specs, **(outlier or {})},
        )


def profile_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    value_sample_n: int = 10,
    sigma_multiple: float = 3.0,
    iqr_factor: float = 1.5,
    iqr_ratio: float = 0.01,
    sentinel_value: float = SENTINEL_FILL_VALUE,
) -> FeatureProfileTable:
    """生成待预处理特征统计表，并为每类问题填入默认策略。

    输出列
    ------
    列名 / 类型 / 基数 / 缺失率 / 最高频取值占比 / 非数值取值样例 /
    最大值 / 3σ偏离倍数 / IQR离群比例 / 哨兵值冲突 / 异常标记
    """
    rows: list[dict[str, object]] = []
    profiles: dict[str, FeatureProfile] = {}
    enc: dict[str, EncodeSpec] = {}
    fil: dict[str, FillSpec] = {}
    out: dict[str, OutlierSpec] = {}

    for col in feature_cols:
        if col not in df.columns:
            raise DataError(f"特征列 {col} 不存在于数据中")
        series = df[col]
        numeric = pd.api.types.is_numeric_dtype(series)
        miss = float(series.isna().mean())
        top_share = _top_share(series)

        sigma_dev: float | None = None
        iqr_ratio_v: float | None = None
        conflict = False
        flag = ""

        if numeric:
            values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
            if values.size:
                mu, sd = float(values.mean()), float(values.std())
                if sd > 0:
                    sigma_dev = round(float(np.abs(values - mu).max()) / sd, 2)
                q1, q3 = np.quantile(values, [0.25, 0.75])
                iqr = float(q3 - q1)
                if iqr > 0:
                    iqr_ratio_v = round(
                        float(
                            ((values < q1 - iqr_factor * iqr) | (values > q3 + iqr_factor * iqr))
                            .mean()
                        ),
                        4,
                    )
                if np.isclose(values, sentinel_value).any():
                    conflict = True
            if (sigma_dev is not None and sigma_dev > sigma_multiple) or (
                iqr_ratio_v is not None and iqr_ratio_v > iqr_ratio
            ):
                flag = "显著"

        samples: tuple[str, ...] = ()
        if not numeric:
            samples = tuple(map(str, series.dropna().unique()[:value_sample_n]))

        profile = FeatureProfile(
            column=col,
            is_numeric=numeric,
            n_unique=int(series.nunique(dropna=True)),
            missing_rate=round(miss, 6),
            top_value_share=round(top_share, 6),
            value_samples=samples,
            sigma_deviation=sigma_dev,
            iqr_outlier_ratio=iqr_ratio_v,
            has_conflicting_sentinel=conflict,
            flag=flag,
        )
        profiles[col] = profile

        if not numeric:
            enc[col] = EncodeSpec(column=col, strategy="ordinal")
        if miss > 0:
            fil[col] = FillSpec(column=col, strategy="sentinel", value=sentinel_value)
        if numeric:
            out[col] = OutlierSpec(
                column=col, strategy="mark",
                sigma_multiple=sigma_multiple, iqr_factor=iqr_factor, iqr_ratio=iqr_ratio,
            )

        rows.append(
            {
                "列名": col,
                "类型": "数值" if numeric else "非数值",
                "基数": profile.n_unique,
                "缺失率": profile.missing_rate,
                "最高频取值占比": profile.top_value_share,
                "非数值取值样例": ", ".join(samples),
                "最大值": (
                    float(pd.to_numeric(series, errors="coerce").max()) if numeric else None
                ),
                "3σ偏离倍数": sigma_dev,
                "IQR离群比例": iqr_ratio_v,
                "哨兵值冲突": "★是" if conflict else "",
                "异常标记": flag,
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "列名", "类型", "基数", "缺失率", "最高频取值占比", "非数值取值样例",
            "最大值", "3σ偏离倍数", "IQR离群比例", "哨兵值冲突", "异常标记",
        ],
    )
    return FeatureProfileTable(
        frame=frame, profiles=profiles, encode_specs=enc, fill_specs=fil, outlier_specs=out
    )


def _top_share(series: pd.Series) -> float:
    """最高频取值占比。单值列返回 1.0，不抛异常。"""
    try:
        counts = series.value_counts(normalize=True, dropna=False)
    except (ValueError, TypeError):       # pragma: no cover - 极端 dtype
        return 0.0
    return float(counts.iloc[0]) if len(counts) else 0.0


# --------------------------------------------------------------------------- #
# 编码
# --------------------------------------------------------------------------- #
def apply_encode(df: pd.DataFrame, specs: Mapping[str, EncodeSpec]) -> pd.DataFrame:
    """执行非数值编码。

    ``ordinal`` 按取值字典序映射，保证同一列的映射可复现（不依赖数据顺序）。

    ``target`` / ``binned`` **必须提供已拟合的编码表**，本函数不自行拟合 ——
    隐式在全量数据上拟合编码表会造成标签泄漏，且无法复现。

    Raises
    ------
    ConfigError
        列不存在、策略未知、或缺少已拟合编码表。
    """
    out = df.copy()
    for col, spec in specs.items():
        if col not in out.columns:
            raise ConfigError(f"待编码列 {col} 不存在")
        if spec.strategy == "drop":
            out = out.drop(columns=[col])
            continue

        if spec.strategy == "ordinal":
            mapping = spec.mapping
            if mapping is None:
                values = sorted(map(str, out[col].dropna().unique()))
                mapping = {v: float(i) for i, v in enumerate(values)}
            # 缺失保留为 NA —— 若先 astype(str)，NaN 会变成字面量 "nan"，
            # 既被当成一个真实类别参与映射，也丢掉了「这是缺失」的信息。
            out[col] = out[col].map(
                lambda v: np.nan if pd.isna(v) else mapping.get(str(v), np.nan)
            )
        elif spec.strategy == "onehot":
            dummies = pd.get_dummies(out[col].astype(str), prefix=col, dummy_na=True)
            out = pd.concat([out.drop(columns=[col]), dummies], axis=1)
        elif spec.strategy in _NEEDS_FITTED_TABLE:
            raise ConfigError(
                f"{spec.strategy} 编码必须提供已拟合的编码表（fit_mask=train），"
                f"不允许在预处理阶段隐式拟合 —— 那会造成标签泄漏且无法复现。"
                f"（当前仅开放 ordinal / onehot / drop）"
            )
        else:
            raise ConfigError(f"未知的编码策略：{spec.strategy}")
    return out


# --------------------------------------------------------------------------- #
# 缺失填充
# --------------------------------------------------------------------------- #
def apply_fill(df: pd.DataFrame, specs: Mapping[str, FillSpec]) -> pd.DataFrame:
    """执行缺失填充。

    哨兵值填充前强制校验取值域 —— 冲突即抛 :class:`ConfigError`，
    **不做静默继续**（详见模块 docstring）。

    ``as_bin`` 策略保留 NaN，由下游 binning 以 ``MISSING_BIN`` 处理。

    Raises
    ------
    ConfigError
        取值域与哨兵值冲突，或策略未知。
    DataError
        ``group`` 策略的分组列不存在。
    """
    out = df.copy()
    for col, spec in specs.items():
        if col not in out.columns:
            raise ConfigError(f"待填充列 {col} 不存在")
        n_missing = int(out[col].isna().sum())
        if n_missing == 0:
            continue

        if spec.strategy == "sentinel":
            existing = pd.to_numeric(out[col], errors="coerce").dropna()
            if len(existing) and bool(np.isclose(existing, spec.value).any()):
                raise ConfigError(
                    f"特征 {col} 的实际取值域中已包含哨兵值 {spec.value}，"
                    f"填充后真实值与缺失不可区分，模型会学到错误模式。"
                    f"请改用其他填充策略（mean / median / as_bin / drop）。"
                )
            out[col] = out[col].fillna(spec.value)
        elif spec.strategy == "mean":
            out[col] = out[col].fillna(out[col].mean())
        elif spec.strategy == "median":
            out[col] = out[col].fillna(out[col].median())
        elif spec.strategy == "mode":
            modes = out[col].mode()
            out[col] = out[col].fillna(modes.iloc[0] if len(modes) else np.nan)
        elif spec.strategy == "ffill":
            out[col] = out[col].ffill()
        elif spec.strategy == "group":
            if not spec.group_by:
                raise ConfigError(f"特征 {col} 使用 group 填充但未指定 group_by 列")
            if spec.group_by not in out.columns:
                raise DataError(f"分组列 {spec.group_by} 不存在")
            out[col] = out[col].fillna(out.groupby(spec.group_by)[col].transform("mean"))
        elif spec.strategy == "as_bin":
            pass                      # 保留 NaN，交给 binning 的 MISSING_BIN
        elif spec.strategy == "drop":
            out = out.loc[out[col].notna()]
        else:
            raise ConfigError(f"未知的填充策略：{spec.strategy}")
    return out


# --------------------------------------------------------------------------- #
# 异常值
# --------------------------------------------------------------------------- #
def apply_outlier(
    df: pd.DataFrame,
    specs: Mapping[str, OutlierSpec],
    *,
    flag_column_suffix: str = "__outlier_flag",
    flags: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """执行异常值处置。

    默认策略 ``mark`` 只在结果上打标记列、**不改数值**；若调用方传了 ``flags``
    则额外把布尔标记写进该字典（供报告或下游使用，不污染主表）。

    Raises
    ------
    ConfigError
        策略未知。
    """
    out = df.copy()
    for col, spec in specs.items():
        if col not in out.columns:
            raise ConfigError(f"待处置列 {col} 不存在")
        values = pd.to_numeric(out[col], errors="coerce")
        valid = values.notna()
        if not bool(valid.any()):
            continue

        mask = _outlier_mask(values, spec, valid)
        if flags is not None:
            flags[col] = mask
        if spec.strategy == "mark":
            out[f"{col}{flag_column_suffix}"] = mask
        elif spec.strategy == "winsorize":
            lo, hi = _bounds(values, spec, valid)
            out.loc[mask, col] = np.where(values[mask] < lo, lo, hi)
        elif spec.strategy == "clip":
            lo, hi = _bounds(values, spec, valid)
            out.loc[valid, col] = values[valid].clip(lo, hi)
        elif spec.strategy == "to_missing":
            out.loc[mask, col] = np.nan
        elif spec.strategy == "drop_column":
            out = out.drop(columns=[col])
        elif spec.strategy == "drop_rows":
            out = out.loc[~mask]
        else:
            raise ConfigError(f"未知的异常值处置策略：{spec.strategy}")
    return out


def _outlier_mask(
    values: pd.Series, spec: OutlierSpec, valid: pd.Series
) -> pd.Series:
    """IQR 口径的离群掩码。3σ 只用于**标记**是否显著，不作为截断依据。

    为什么截断只用 IQR 而不用 3σ：3σ 对长尾特征过于敏感
    （如「额度」这种右偏极重的字段，正态假设根本不成立），
    用它做截断会砍掉大量合法的头部客户。
    """
    q1, q3 = values[valid].quantile([0.25, 0.75])
    iqr = float(q3 - q1)
    mask = pd.Series(False, index=values.index)
    if iqr > 0:
        mask = valid & ((values < q1 - spec.iqr_factor * iqr) | (values > q3 + spec.iqr_factor * iqr))
    return mask


def _bounds(
    values: pd.Series, spec: OutlierSpec, valid: pd.Series
) -> tuple[float, float]:
    q1, q3 = values[valid].quantile([0.25, 0.75])
    iqr = float(q3 - q1)
    return float(q1 - spec.iqr_factor * iqr), float(q3 + spec.iqr_factor * iqr)


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PrepareAction:
    """单条预处理动作，用于写入报告。"""

    column: str
    kind: Literal["encode", "fill", "outlier"]
    strategy: str
    detail: str = ""


@dataclass
class PrepareReport:
    """列级预处理报告。落盘为「预处理报告」sheet。"""

    actions: list[PrepareAction] = field(default_factory=list)
    n_rows_in: int = 0
    n_rows_out: int = 0
    columns_dropped: list[str] = field(default_factory=list)

    @property
    def n_rows_removed(self) -> int:
        return self.n_rows_in - self.n_rows_out

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [asdict(a) for a in self.actions],
            columns=["column", "kind", "strategy", "detail"],
        ).rename(
            columns={
                "column": "列名", "kind": "处理类别",
                "strategy": "策略", "detail": "说明",
            }
        )

    def record_encode(self, specs: Mapping[str, EncodeSpec]) -> None:
        for col, spec in specs.items():
            detail = ""
            if spec.strategy == "ordinal" and spec.mapping is None:
                detail = "按取值字典序映射为 0,1,2…（可复现，不依赖数据顺序）"
            self.actions.append(PrepareAction(col, "encode", spec.strategy, detail or spec.note))

    def record_fill(self, specs: Mapping[str, FillSpec]) -> None:
        for col, spec in specs.items():
            detail = (
                f"填充为 {spec.value}" if spec.strategy == "sentinel"
                else (f"按 {spec.group_by} 分组均值" if spec.group_by else "")
            )
            self.actions.append(PrepareAction(col, "fill", spec.strategy, detail or spec.note))

    def record_outlier(self, specs: Mapping[str, OutlierSpec]) -> None:
        for col, spec in specs.items():
            detail = "仅打标记，不改数值" if spec.strategy == "mark" else spec.note
            self.actions.append(PrepareAction(col, "outlier", spec.strategy, detail))
