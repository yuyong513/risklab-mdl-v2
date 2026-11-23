"""级联初筛：五道筛子按固定顺序执行，纯包内实现。

执行顺序（固定）
----------------
::

    ① 缺失率 → ② 同值率 → ③ IV → ④ PSI → ⑤ 相关性

★ 为什么必须级联
----------------
缺失率与同值率只需列级聚合；IV / PSI / 相关性要分箱、排序或建相关矩阵，
成本高 1~2 个数量级。每一步只在上一步的存活集上计算，前两道先剔掉一批，
后续指标的计算量同比下降。级联不改变筛选结论 —— 被前道剔掉的特征
无论后面用多严格的阈值都进不了模型，因此是纯粹的性能优化。

★ 阈值零兜底
------------
所有阈值由 ``PrescreenConfig`` 显式注入，本层不读配置文件、不设默认值。
设了默认值就分不清「使用方主动选了 0.9」和「忘了配、代码替我选了一个」。

★ 取数口径
----------
IV 的分箱切点只在传入的 ``train`` 上拟合；PSI 用同一份数据的月份列做跨期对比。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .binning import BinSpec, bin_table_processed, fit_bins, iv_of, transform_bins
from .correlation import decorrelate
from .errors import ConfigError, DataError
from .psi import PsiConfig, screen_psi

__all__ = [
    "PrescreenConfig",
    "PrescreenResult",
    "ScreenResult",
    "read_table",
    "run_prescreen",
    "screen_correlation",
    "screen_iv",
    "screen_missing_rate",
    "screen_same_value_rate",
]

_STAGE_ORDER = ("missing_rate", "same_value_rate", "iv", "psi", "correlation")

_STAGE_LABEL = {
    "missing_rate": "缺失率",
    "same_value_rate": "同值率",
    "iv": "IV",
    "psi": "PSI",
    "correlation": "相关性",
}


# ============ 数据读取 ============


def read_table(path: str | Path) -> pd.DataFrame:
    """读取表格数据。

    ★ 先探测编码再读
    ----------------
    用 utf-8 + ``errors="replace"`` 读 GBK 文件会「静默成功但内容全错」：
    中文乱码，字面 ``\\N`` 被破坏后空值计数变 0 ⇒ 缺失率算错且不报错。

    ★ 空值双表示统一
    ----------------
    CSV 里空值有真空字段与字面 ``\\N`` 两种写法，不统一会污染分母。
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise DataError(f"数据文件不存在: {p}")

    raw = p.read_bytes()[:200000]
    encoding = "latin-1"
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin-1"):
        try:
            raw.decode(enc)
            encoding = enc
            break
        except UnicodeDecodeError:
            continue

    return pd.read_csv(
        p,
        encoding=encoding,
        low_memory=False,
        na_values=["\\N", ""],
        keep_default_na=True,
    )


# ============ 配置与结果 ============


@dataclass(frozen=True)
class PrescreenConfig:
    """初筛配置。阈值全部显式注入 —— 本层零配置依赖。"""

    missing_threshold: float
    same_value_threshold: float
    iv_threshold: float
    psi_threshold: float
    corr_threshold: float

    corr_strategy: str = "greedy"
    bin_spec: BinSpec | None = None
    month_column: str = ""
    psi_base_mode: str = "specified"
    psi_base_months: tuple[str, ...] = ()
    psi_rolling_base_months: tuple[str, ...] = ()
    psi_multi_base_tolerance: int = 1
    high_iv_alert: float = 0.50
    high_psi_alert: float = 0.25
    enable_psi: bool = True

    def __post_init__(self) -> None:
        for name in (
            "missing_threshold", "same_value_threshold", "iv_threshold",
            "psi_threshold", "corr_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ConfigError(f"{name} 必须落在 (0, 1) 区间，当前 {value}")
        if self.enable_psi and not self.month_column:
            raise ConfigError(
                "启用 PSI 时必须提供 month_column，否则没有时间维度可比；"
                "若确实无月份列，请显式设 enable_psi=False 并在报告中说明未做跨期验证。"
            )
        if self.enable_psi and self.psi_base_mode == "specified" and not self.psi_base_months:
            raise ConfigError(
                "psi_base_mode='specified' 时必须提供 psi_base_months。"
                "留空会让基期默认为「全部月份」，等于没做基期对比。"
            )


@dataclass
class ScreenResult:
    """单道筛子的结果。``metrics`` 是逐列指标值，供报告显示。"""

    dropped: list[str] = field(default_factory=list)
    reason: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class IvScreenResult(ScreenResult):
    """IV 环节额外透出箱数参照系与高 IV 告警。"""

    iv_bins: dict[str, list[int]] = field(default_factory=dict)
    high_iv_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class PrescreenResult:
    """初筛汇总结果。"""

    survived: list[str] = field(default_factory=list)
    stage_dropped: dict[str, list[str]] = field(default_factory=dict)
    stage_reason: dict[str, dict[str, str]] = field(default_factory=dict)
    stage_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    iv_map: dict[str, float] = field(default_factory=dict)
    iv_bins: dict[str, list[int]] = field(default_factory=dict)
    """逐列的（请求箱数, 实际箱数）。IV 读数依赖箱数，不透出就无法判断参照系。"""
    high_iv_columns: list[str] = field(default_factory=list)
    high_psi_columns: list[str] = field(default_factory=list)
    total_input: int = 0
    computed_counts: dict[str, int] = field(default_factory=dict)
    psi_detail: pd.DataFrame | None = None
    psi_base_used: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def total_dropped(self) -> int:
        return sum(len(v) for v in self.stage_dropped.values())

    @property
    def saved_ratio(self) -> float:
        """级联剪枝节省的后续计算量（相对全量并行）。"""
        if self.total_input <= 0:
            return 0.0
        later = self.computed_counts.get("iv", self.total_input)
        return 1.0 - later / self.total_input

    def as_frame(self) -> pd.DataFrame:
        """逐条剔除明细。"""
        rows: list[dict[str, Any]] = []
        for stage in _STAGE_ORDER:
            for col, reason in self.stage_reason.get(stage, {}).items():
                rows.append({
                    "筛子": _STAGE_LABEL[stage],
                    "列名": col,
                    "剔除原因": reason,
                    "指标值": self.stage_metrics.get(stage, {}).get(col, float("nan")),
                })
        return pd.DataFrame(rows, columns=["筛子", "列名", "剔除原因", "指标值"])

    def iv_frame(self) -> pd.DataFrame:
        """存活特征的 IV 排序表。"""
        items = [(c, self.iv_map[c]) for c in self.survived if c in self.iv_map]
        items.sort(key=lambda t: -t[1])
        return pd.DataFrame(items, columns=["列名", "IV"])

    def summary(self) -> dict[str, Any]:
        return {
            "total_input": self.total_input,
            "n_survived": len(self.survived),
            "total_dropped": self.total_dropped,
            "stage_dropped": {k: len(v) for k, v in self.stage_dropped.items()},
            "computed_counts": dict(self.computed_counts),
            "saved_ratio": round(self.saved_ratio, 4),
            "high_iv_columns": self.high_iv_columns,
            "high_psi_columns": self.high_psi_columns,
            "psi_base_used": self.psi_base_used,
            "notes": list(self.notes),
        }


# ============ 五道筛子 ============


def screen_missing_rate(
    df: pd.DataFrame, cols: Sequence[str], *, threshold: float
) -> ScreenResult:
    """① 缺失率超阈值即剔除。"""
    rates = df[list(cols)].isna().mean()
    dropped, reason, metrics = [], {}, {}
    for col in cols:
        rate = float(rates[col])
        metrics[col] = round(rate, 6)
        if rate > float(threshold):
            dropped.append(col)
            reason[col] = f"缺失率 {rate:.2%} 超过阈值 {float(threshold):.0%}"
    return ScreenResult(dropped, reason, metrics)


def screen_same_value_rate(
    df: pd.DataFrame, cols: Sequence[str], *, threshold: float
) -> ScreenResult:
    """② 最高频取值占比超阈值即剔除（含 NaN 作为一个取值）。"""
    dropped, reason, metrics = [], {}, {}
    for col in cols:
        counts = df[col].value_counts(normalize=True, dropna=False)
        share = float(counts.iloc[0]) if len(counts) else 0.0
        metrics[col] = round(share, 6)
        if share > float(threshold):
            dropped.append(col)
            reason[col] = f"最高频取值占比 {share:.2%} 超过阈值 {float(threshold):.0%}"
    return ScreenResult(dropped, reason, metrics)


def screen_iv(
    df: pd.DataFrame,
    cols: Sequence[str],
    y: pd.Series,
    *,
    threshold: float,
    bin_spec: BinSpec | None = None,
    high_iv_alert: float = 0.50,
) -> IvScreenResult:
    """③ 分箱算 IV，低于阈值即剔除；无法分箱的列同样剔除并记原因。

    ★ 切点只在 ``df`` 上拟合，调用方须保证传入的是训练期数据。
    ★ IV 过高只告警不剔除 —— 通常意味着含未来信息，要人工排查。
    """
    spec = bin_spec or BinSpec()
    out = IvScreenResult()
    iv_values: dict[str, float] = {}

    for col in cols:
        try:
            numeric = pd.to_numeric(df[col], errors="coerce")
            fitted = fit_bins(numeric, spec)
            codes = transform_bins(numeric, fitted)
            table = bin_table_processed(codes, y, fitted)
            value = float(iv_of(table))
        except (ValueError, TypeError, DataError) as exc:
            out.dropped.append(col)
            out.reason[col] = f"无法计算 IV：{type(exc).__name__}: {exc}"
            continue
        iv_values[col] = round(value, 6)
        out.metrics[col] = round(value, 6)
        out.iv_bins[col] = [int(fitted.requested_bins), int(fitted.n_valid_bins)]
        if value < float(threshold):
            out.dropped.append(col)
            out.reason[col] = f"IV={value:.4f} 低于阈值 {float(threshold)}"

    out.high_iv_columns = sorted(
        [c for c, v in iv_values.items() if v >= float(high_iv_alert)],
        key=lambda c: -iv_values[c],
    )
    downgraded = {k: v for k, v in out.iv_bins.items() if v[1] < v[0]}
    if downgraded:
        out.notes.append(
            f"⚠ {len(downgraded)} 个特征的箱数因样本量不足被下调，"
            "IV 读数稳定性下降，与按请求箱数算出的口径不可直接比较。"
        )
    return out


def screen_correlation(
    df: pd.DataFrame,
    cols: Sequence[str],
    importance: dict[str, float],
    *,
    threshold: float,
    strategy: str = "greedy",
) -> ScreenResult:
    """⑤ 去相关。按重要度（通常为 IV）降序保留，超阈值的高相关列剔除。

    直接复用 ``correlation.decorrelate``，保证与流水线主流程同一口径。
    """
    if not cols:
        return ScreenResult()
    result = decorrelate(
        df, list(cols), importance,
        strategy=strategy,  # type: ignore[arg-type]
        threshold=float(threshold),
    )
    return ScreenResult(
        dropped=list(result.dropped),
        reason=dict(result.dropped_reason),
        metrics={},
    )


# ============ 级联编排 ============


def run_prescreen(
    train: pd.DataFrame,
    feature_cols: Sequence[str],
    y: pd.Series,
    cfg: PrescreenConfig,
) -> PrescreenResult:
    """按固定顺序级联执行五道筛子，每步只在上一步的存活集上计算。"""
    alive = list(feature_cols)
    result = PrescreenResult(total_input=len(alive))
    if not alive:
        raise DataError("待筛选特征列表为空")
    y = pd.Series(y, index=train.index) if not isinstance(y, pd.Series) else y

    # ① 缺失率（最便宜）
    r = screen_missing_rate(train, alive, threshold=cfg.missing_threshold)
    alive = _advance(alive, r, result, "missing_rate")
    result.computed_counts["missing_rate"] = len(alive)

    # ② 同值率（同样便宜）
    r = screen_same_value_rate(train, alive, threshold=cfg.same_value_threshold)
    alive = _advance(alive, r, result, "same_value_rate")

    # ③ IV（从这里开始才是昂贵环节）
    result.computed_counts["iv"] = len(alive)
    iv = screen_iv(
        train, alive, y,
        threshold=cfg.iv_threshold,
        bin_spec=cfg.bin_spec,
        high_iv_alert=cfg.high_iv_alert,
    )
    result.iv_map = dict(iv.metrics)
    result.iv_bins = dict(iv.iv_bins)
    result.high_iv_columns = list(iv.high_iv_columns)
    result.notes.extend(iv.notes)
    alive = _advance(alive, iv, result, "iv")

    # ④ PSI
    if cfg.enable_psi:
        result.computed_counts["psi"] = len(alive)
        base_months = cfg.psi_base_months or _earliest_months(train, cfg.month_column)
        psi_cfg = PsiConfig(
            threshold=cfg.psi_threshold,
            base_mode=cfg.psi_base_mode,  # type: ignore[arg-type]
            base_months=tuple(base_months),
            rolling_base_months=tuple(cfg.psi_rolling_base_months or base_months[:1]),
            multi_base_tolerance=cfg.psi_multi_base_tolerance,
            bin_spec=cfg.bin_spec,
            high_psi_alert=cfg.high_psi_alert,
        )
        psi = screen_psi(train, alive, cfg.month_column, psi_cfg)
        result.high_psi_columns = list(psi.high_psi_columns)
        result.psi_detail = psi.detail
        result.psi_base_used = psi.base_used
        if psi.note:
            result.notes.append(psi.note)
        alive = _advance(alive, psi, result, "psi")

    # ⑤ 相关性
    result.computed_counts["correlation"] = len(alive)
    importance = {c: result.iv_map.get(c, 0.0) for c in alive}
    corr = screen_correlation(
        train, alive, importance,
        threshold=cfg.corr_threshold, strategy=cfg.corr_strategy,
    )
    alive = _advance(alive, corr, result, "correlation")

    result.survived = alive
    if result.high_iv_columns:
        result.notes.append(
            f"⚠ 以下 {len(result.high_iv_columns)} 个特征 IV ≥ "
            f"{cfg.high_iv_alert}：{result.high_iv_columns[:10]}"
            f"{' ...' if len(result.high_iv_columns) > 10 else ''}。"
            "正常变量 IV 在 0.02~0.5，超线通常意味着含未来信息，建议排查。"
        )
    return result


def _advance(
    alive: list[str], screen: ScreenResult, agg: PrescreenResult, stage: str
) -> list[str]:
    dropped = set(screen.dropped)
    agg.stage_dropped[stage] = list(screen.dropped)
    agg.stage_reason[stage] = dict(screen.reason)
    agg.stage_metrics[stage] = dict(screen.metrics)
    return [c for c in alive if c not in dropped]


def _earliest_months(df: pd.DataFrame, month_column: str) -> tuple[str, ...]:
    if not month_column or month_column not in df.columns:
        raise ConfigError(
            "启用 PSI 时必须提供有效的 month_column（否则没有时间维度可比）。"
        )
    return tuple(sorted(df[month_column].dropna().astype(str).unique())[:1])
