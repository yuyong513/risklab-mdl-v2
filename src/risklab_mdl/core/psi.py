"""逐月 PSI 稳定性筛选。

★ 切点只在基期拟合；逐月 PSI 取**最大值**作为该特征的判定值。
★ 多基期的「标签 vs 数值」口径必须一致 —— 标签写取中位数而实际取
   最后一个基期，会让报告与结论相反，属典型的失败不可见。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .binning import BinSpec, bin_psi, bin_table, fit_bins, transform_bins
from .errors import ConfigError, DataError

__all__ = [
    "PsiBaseMode",
    "PsiConfig",
    "PsiResult",
    "screen_psi",
]

PsiBaseMode = Literal["specified", "rolling", "multi_base"]

_BASE_LABEL = "__BASE__"
"""基期月在输出表中的占位标签 —— 月份列表可能不含基期本身。"""


@dataclass(frozen=True)
class PsiConfig:
    """PSI 配置。阈值由 config 注入（默认 0.10）。"""

    threshold: float
    base_mode: PsiBaseMode = "specified"
    base_months: tuple[str, ...] = ()
    rolling_base_months: tuple[str, ...] = ()
    multi_base_tolerance: int = 1
    bin_spec: BinSpec | None = None
    high_psi_alert: float = 0.25
    """业界通用的「显著漂移」线。超过此值的列会额外标记，供人工判断。"""


@dataclass
class PsiResult:
    """PSI 筛选结果。"""

    screen: str = "PSI"
    dropped: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    reason: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    detail: pd.DataFrame | None = None
    """逐列逐月的 PSI 明细，供报告与追查。"""
    high_psi_columns: list[str] = field(default_factory=list)
    base_used: str = ""
    note: str = ""

    @property
    def n_dropped(self) -> int:
        return len(self.dropped)

    def as_frame(self) -> pd.DataFrame:
        rows = [
            {"筛子": self.screen, "列名": col, "剔除原因": r, "指标值": self.metrics.get(col, np.nan)}
            for col, r in self.reason.items()
        ]
        return pd.DataFrame(rows, columns=["筛子", "列名", "剔除原因", "指标值"])


def screen_psi(
    df: pd.DataFrame,
    cols: Sequence[str],
    month_col: str,
    cfg: PsiConfig,
) -> PsiResult:
    """按 PSI 剔除漂移特征。

    Parameters
    ----------
    df
        含``month_col`` 的数据集。
    cols
        待筛选的列。
    month_col
        月粒度列。
    cfg
        阈值 + 基期模式 + 分箱规格。

    Raises
    ------
    ConfigError
        基期月份为空、``multi_base`` 模式下月份不足、或指定月份不在数据中。
    DataError
        列或月份列不存在。
    """
    if month_col not in df.columns:
        raise DataError(f"月份列 {month_col} 不存在")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataError(f"待筛选列不存在：{missing[:10]}")

    months = sorted(df[month_col].dropna().astype(str).unique())
    if not months:
        raise DataError("月份列全部缺失，无法计算 PSI")

    pairs, base_months, note = _resolve_pairs(months, cfg)
    # ★★ 多基期 ⇒ 多个基期月被**合并成一个池**（base_mask 用 isin）
    #    ⇒ 标签必须表达「池」，不能逐个写成单月（那是假标签）。
    #    单基期时池 == 该月，标签本就正确 ⇒ 逐字不变。
    base_pool = len(base_months) > 1
    base_label = _base_pool_label(base_months)
    spec_template = cfg.bin_spec or BinSpec()

    dropped: list[str] = []
    reason: dict[str, str] = {}
    metrics: dict[str, float] = {}
    kept: list[str] = []
    high_psi: list[str] = []
    detail_rows: list[dict[str, object]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for col in cols:
        # 切点统一在**基期**上拟合并复用到各对比月 —— 保证 PSI 是同一把尺子量的
        base_mask = df[month_col].astype(str).isin(list(base_months))
        base_series = df.loc[base_mask, col]
        if base_series.dropna().empty:
            continue
        try:
            fitted = fit_bins(base_series.astype(float), spec_template)
            base_table = bin_table(
                transform_bins(base_series.astype(float), fitted),
                _safe_y(df, base_mask, col),
            )
        except (ValueError, TypeError, DataError) as exc:
            dropped.append(col)
            reason[col] = f"无法计算 PSI：{exc}"
            continue

        per_month: dict[str, float] = {}
        failed_months: list[str] = []
        empty_months: list[str] = []
        for base_month, test_month in pairs:
            test_mask = df[month_col].astype(str) == test_month
            test_series = df.loc[test_mask, col]
            if test_series.dropna().empty:
                empty_months.append(test_month)
                continue
            try:
                test_table = bin_table(
                    transform_bins(test_series.astype(float), fitted),
                    _safe_y(df, test_mask, col),
                )
                value = bin_psi(base_table, test_table)
            except (ValueError, TypeError, DataError) as exc:
                failed_months.append(f"{test_month}({exc})")
                continue
            per_month[test_month] = round(value, 6)
            # ★★ 合并池下，同一对比月的多行数值**必然完全相同**
            #    （base_table 是同一个池算出来的）⇒ 只留第一行，避免冗余。
            #    注意：这**不改变**任何数值 —— per_month / metrics 只依赖
            #    「对比月 → PSI」的映射，与这里写几行无关。
            key = (col, test_month)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            # ★ 标签用真标签：多基期时写「基期池(...)」，不再逐个写成单月。
            detail_rows.append({"列名": col, "基期": base_label, "对比月": test_month, "PSI": value})

        if not per_month:
            dropped.append(col)
            # ★ F-08：区分「没配对」与「配对了但算不出来」。
            #   旧文案恒为「计算失败：[]」—— 空列表看着像「没失败」，
            #   使用者会以为是自己数据的问题，实际是**根本没配对**（F-08）。
            if not pairs:
                reason[col] = (
                    f"PSI 无可对比的配对：数据里只有基期月 {list(base_months)}，"
                    f"没有可与之对比的其他月份。"
                    f"★ 这不是特征本身的问题 —— 请检查参与 PSI 计算的数据是否包含对比月。"
                )
            elif not failed_months and empty_months:
                reason[col] = (
                    f"PSI 对比月无有效数据：{empty_months[:3]}。"
                    f"★ 基期月 {list(base_months)} 与这些月份虽有配对，"
                    f"但对比月该列全为空 —— 请检查数据覆盖范围。"
                )
            else:
                reason[col] = f"PSI 全部对比月计算失败：{failed_months[:3]}"
            continue

        stat = _reduce(per_month, cfg.base_mode)
        metrics[col] = round(stat, 6)
        if stat > cfg.high_psi_alert:
            high_psi.append(col)

        if stat > cfg.threshold:
            dropped.append(col)
            worst = max(per_month, key=per_month.get)
            reason[col] = (
                f"{_label(cfg.base_mode, base_pool=base_pool)}={stat:.4f} 超过阈值 {cfg.threshold}"
                f"（最大月 {worst}: {per_month[worst]:.4f}）"
                + (f"；失败月 {failed_months[:2]}" if failed_months else "")
            )
        else:
            kept.append(col)

    detail = pd.DataFrame(detail_rows, columns=["列名", "基期", "对比月", "PSI"])
    return PsiResult(
        dropped=dropped,
        kept=kept,
        reason=reason,
        metrics=metrics,
        detail=detail,
        high_psi_columns=sorted(high_psi),
        base_used=base_label,
        note=note,
    )


def _base_pool_label(base_months: Sequence[str]) -> str:
    """基期标签。

    ★ 单基期 ⇒ 直接就是那个月（合并池 == 该月，标签本就正确）。
    ★ 多基期 ⇒ 必须表达「**池**」，不能逐个写成单月 ——
      ``base_mask`` 用 ``isin`` ⇒ 多个基期月被合并成一个池，
      切点在池上拟合，各对比月与**池**对比。
      写成 ``"2026年1月→2026年2月"`` 会让人误以为是「先后两月」，
      而 detail 里逐行标单月则是**假标签**（数值其实是池口径）。
    """
    months = list(base_months)
    if len(months) <= 1:
        return months[0] if months else ""
    return "基期池(" + "+".join(months) + ")"


def _safe_y(df: pd.DataFrame, mask: pd.Series, col: str) -> pd.Series:
    """取与 ``mask`` 对齐的标签列；标签列缺失时用全1 占位。

    PSI 只看分布不看标签，占位即可 —— 但为保持与 IV 口径一致，
    若存在 ``y`` 列则优先用它。
    """
    if "y" in df.columns:
        return df.loc[mask, "y"]
    return pd.Series(1, index=df.index)[mask]


def _resolve_pairs(
    months: Sequence[str], cfg: PsiConfig
) -> tuple[list[tuple[str, str]], tuple[str, ...], str]:
    """确定「基期月 → 对比月」的配对。

    Returns
    -------
    ``(配对列表, 基期月份元组, 提示语)``

    ⚠ **第二个返回值必须是月份元组，不能是拼好的字符串。**
    调用方要用它做 ``df[month].isin(base_months)`` ——
    若传字符串再 ``list()``，会被拆成**单个字符**（``"202601"`` → ``['2','0','2',...]``），
    于是一个月份都匹配不上，PSI 静默返回空结果。
    """
    months = list(months)
    n = len(months)

    if cfg.base_mode == "specified":
        base = tuple(cfg.base_months)
        if not base:
            raise ConfigError(
                "base_mode='specified' 时必须提供 base_months。"
                "留空会让基期被默认为「全部月份」，等于没做基期对比。"
            )
        absent = [m for m in base if m not in months]
        if absent:
            raise ConfigError(f"指定的基期月份不在数据中：{absent}")
        pairs = [(b, m) for m in months for b in base if m != b]
        if len(base) > 1:
            # ★★ 多基期 = **合并池**口径（base_mask 用 isin）
            #    切点在池上拟合，各对比月与**池**对比 —— 不是逐个单月对比。
            #    这条 note 必须说清，否则使用者会按「多次单月对比」去解读数值。
            return pairs, tuple(base), (
                f"多基期合并池口径：基期池 {list(base)}（{len(base)} 个月**合并为一个池**，"
                f"切点在池上拟合），其余月份全部与**池**对比。"
                f"★ 不是逐个单月对比 —— detail 里同一对比月只有一行。"
            )
        return pairs, tuple(base), (
            f"基期为指定月份 {list(base)}，其余月份全部与它对比。"
        )

    if cfg.base_mode == "multi_base":
        base = tuple(cfg.base_months)
        if len(base) < 2:
            raise ConfigError(
                "base_mode='multi_base' 时至少指定 2 个基期月份"
                "（单基期请用 specified）。"
            )
        absent = [m for m in base if m not in months]
        if absent:
            raise ConfigError(f"指定的基期月份不在数据中：{absent}")
        pairs = [(b, m) for m in months for b in base if m != b]
        # ★★ 与 specified 多基期**同口径**：多个基期月合并成一个池。
        #   原 note 写「取中位数」是**假说明** —— 实现里从未取中位数
        #   （risklab-mcp 原版的 ``_aggregate_multi_base`` 定义了但从未调用，
        #     本实现已把那段死代码删除，避免继续误导）。
        return pairs, tuple(base), (
            f"多基期合并池口径：基期池 {list(base)}（{len(base)} 个月**合并为一个池**，"
            f"切点在池上拟合），其余月份全部与**池**对比，最终对各月取最大值。"
        )

    # rolling：所有特征**共用同一个**基期序列，逐月推进 —— L-04 的关键修正
    base_seq = tuple(cfg.rolling_base_months) or (months[-1],)
    absent = [m for m in base_seq if m not in months]
    if absent:
        raise ConfigError(f"指定的轮转基期月份不在数据中：{absent}")
    pairs: list[tuple[str, str]] = []
    for i, month in enumerate(months):
        if i >= len(base_seq):
            break
        pairs.append((base_seq[i], month))
    return pairs, tuple(base_seq), (
        f"轮转基期 {list(base_seq)}（**所有特征共用同一序列**，这是L-04 的修正点："
        f"不同特征用不同基期会让 PSI 不可比，汇总决策时被放大）。"
    )


def _reduce(per_month: Mapping[str, float], mode: PsiBaseMode) -> float:
    """把逐月 PSI 汇总成单值。

    ★ 所有模式都取**最大值**，不取中位数
    ------------------------------------
    中位数在这里是错的。设想 6 个月里有 1 个月漂移（scale 从 1 跳到 5）：

        逐月 PSI = [0.01, 0.01, 0.01, 0.01, 1.42, 1.63]
                    └── 4 个正常 ──┘└─ 2 个漂移 ─┘
        中位数 = 0.0116   ← 漂移被完全淹没，特征不会被剔除
        最大值 = 1.63     ← 正确识别

    中位数适合「基期选择有不确定性」的场景（那里要的是**典型**水平），
    但 PSI 评估的是「**有没有**月漂移」—— 一次显著漂移就该暴露出来。
    用中位数会让偶发但严重的漂移静默通过，而那恰恰是最该拦住的。

    ★ 多基期下「同一对比月 vs 多个基期」的中位数已**移除**
    ----------------------------------------------
    risklab-mcp 原版留了一个 ``_aggregate_multi_base``（取中位数），
    但它**定义了从未调用** ⇒ 实际口径一直是「合并池 vs 对比月」。
    保留那段死代码只会让人以为在用中位数 —— 本实现已删除，
    并把 ``note``/标签改成如实描述「合并池」。
    """
    values = list(per_month.values())
    if not values:
        return 0.0
    return float(max(values))


def _label(mode: PsiBaseMode, *, base_pool: bool = False) -> str:
    """剔除原因里的指标名。

    ★ 多基期时标明「基期池」—— 否则使用者会以为是单月 PSI，
      而实际数值是「对比月 vs 合并池」。
    """
    suffix = ",基期池" if base_pool else ""
    return {
        "specified": f"PSI(最大月{suffix})",
        "rolling": f"PSI(最大月{suffix})",
        "multi_base": f"PSI(最大月{suffix})",
    }[mode]

# --------------------------------------------------------------------------- #
# ★ 分数分箱的月度 PSI（2026-10-20 新增）
# --------------------------------------------------------------------------- #
def psi_by_month(
    scores: pd.Series,
    months: pd.Series,
    *,
    n_bins: int = 10,
    base_month: str | None = None,
    bin_spec: "BinSpec | None" = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """分数分箱的**月度 PSI**（基期 vs 各对比月），含箱级明细。

    ==================  ===============================================
    返回                内容
    ==================  ===============================================
    ``summary``         每个对比月一行：总 PSI
    ``detail``          **箱级明细**：每个箱在各月的占比与箱级 PSI
    ==================  ===============================================

    ★ 切点必须在**基期**上拟合，且各月共用
    ------------------------------------------
    这是 PSI 可比的前提。若各月各自切点，
    每箱占比都接近 1/n，PSI 恒为 0 ——
    看起来「很稳定」，实际什么都没测到。

    ★ 箱级明细为什么重要
    ------------------
    只有总 PSI 的话，只能知道「漂了多少」，不知道「哪几个箱漂的」。
    箱级明细让使用方能定位到具体分数段。

    ★ 与「不同箱数方案互比」是两回事
    ------------------------------
    本函数比的是**同一个分数变量在不同月份**的分布稳定性；
    ``新分数评价.xlsx`` 里的「分箱PSI矩阵」比的是**不同箱数方案**。
    两者回答不同问题，不能互相替代。
    """
    from .binning import BinSpec, fit_bins, transform_bins

    if len(scores) == 0:
        empty_s = pd.DataFrame(columns=["对比月份", "PSI", "基期月份"])
        empty_d = pd.DataFrame(columns=["箱", "基期占比", "PSI贡献"])
        return empty_s, empty_d

    months = months.astype(str)
    if base_month is None:
        base_month = sorted(months.unique())[0]

    base_mask = months == base_month
    if not base_mask.any():
        raise DataError(f"基期月份 {base_month!r} 在数据中不存在")

    # ★ 切点只在基期上拟合
    spec = bin_spec or BinSpec(n_bins=n_bins)
    fitted = fit_bins(scores[base_mask], spec)
    codes_all = transform_bins(scores, fitted)

    n = int(codes_all.max()) + 1 if len(codes_all) else n_bins
    base_codes = codes_all[base_mask]

    base_cnt = np.bincount(base_codes.to_numpy(), minlength=n).astype(float)
    base_pct = base_cnt / base_cnt.sum() if base_cnt.sum() else base_cnt

    summary_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []

    for m in sorted(months.unique()):
        if m == base_month:
            continue
        sel = months == m
        cnt = np.bincount(codes_all[sel].to_numpy(), minlength=n).astype(float)
        pct = cnt / cnt.sum() if cnt.sum() else cnt

        eps = 1e-6
        b = np.clip(base_pct, eps, None)
        t = np.clip(pct, eps, None)
        box_psi = (t - b) * np.log(t / b)

        summary_rows.append({
            "基期月份": base_month,
            "对比月份": m,
            "PSI": float(box_psi.sum()),
            "样本数": int(sel.sum()),
        })

        detail_frames.append(pd.DataFrame({
            "对比月份": m,
            "箱": np.arange(n),
            "基期占比": base_pct,
            "对比月占比": pct,
            "基期样本数": base_cnt.astype(int),
            "对比月样本数": cnt.astype(int),
            "箱级PSI": box_psi,
        }))

    summary = pd.DataFrame(summary_rows)
    detail = (
        pd.concat(detail_frames, ignore_index=True)
        if detail_frames else pd.DataFrame()
    )
    return summary, detail
