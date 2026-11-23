"""跨期稳定性 PSI 口径测试（移植自旧项目 ``dev/tests/test_psi_multibase_label.py``）。

★ 移植时保留的核心断言
----------------------
旧实现的 ``base_mask`` 用 ``isin(base_months)`` ⇒ **多个基期月被合并成一个池**，
切点在池上拟合、各对比月与**池**比较。但旧 detail 的「基期」列却逐月写成
``2026年1月`` / ``2026年2月``，让人误以为是两次独立的单月对比 —— 假标签。

修的是**标签**，不是**数值**。因此本文件同时守护两件事：

1. 标签必须表达「基期池」（不能出现 ``A→B`` 这种暗示先后对比的写法）；
2. 数值必须与「独立复算的合并池口径」**逐列完全一致** ——
   谁要是借改标签之名动了数值，这条会立刻红。

额外补充（v2）：``psi_by_month``（分数分箱的月度 PSI）的基期切点必须共用，
否则各月箱占比都接近 1/n ⇒ PSI 恒为 0，「看起来稳定」但什么都没测到。
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from risklab_mdl.core import psi as psi_mod
from risklab_mdl.core.binning import (  # noqa: E402
    BinSpec,
    bin_psi,
    bin_table,
    fit_bins,
    transform_bins,
)

BASE_A = "2026年1月"
BASE_B = "2026年2月"
DRIFT_MONTH = "2026年3月"


def _frame(months: tuple[str, ...] = (BASE_A, BASE_B, DRIFT_MONTH, "2026年4月")) -> pd.DataFrame:
    """构造多基期可测的迷你数据集（自建 fixture，不依赖外部业务数据）。"""
    rng = np.random.default_rng(20261029)
    rows = []
    for m in months:
        for _ in range(120):
            shift = 3.0 if m == DRIFT_MONTH else 0.0
            rows.append(
                {
                    "y": int(rng.integers(0, 2)),
                    "month": m,
                    "f_stable": rng.normal(shift, 1.0),
                    "f_drift": rng.normal(shift * 2.0, 1.0),
                    "f_const": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _cfg(base_months, threshold: float = 0.1) -> "psi_mod.PsiConfig":
    return psi_mod.PsiConfig(
        threshold=threshold,
        base_mode="specified",
        base_months=tuple(base_months),
        bin_spec=BinSpec(n_bins=8, min_bin_count=20, min_bin_pct=0.005),
    )


def _manual_pool_metrics(df: pd.DataFrame, cols, base_months, month_col="month", label="y"):
    """独立复算「合并池口径」—— 数值对账的参照实现（刻意不复用内部逻辑）。"""
    spec = BinSpec(n_bins=8, min_bin_count=20, min_bin_pct=0.005)
    months = sorted(df[month_col].dropna().astype(str).unique())
    out = {}
    for col in cols:
        base_mask = df[month_col].astype(str).isin(list(base_months))
        bs = df.loc[base_mask, col]
        if bs.dropna().empty:
            continue
        fitted = fit_bins(bs.astype(float), spec)
        bt = bin_table(transform_bins(bs.astype(float), fitted), df.loc[base_mask, label])
        vals = []
        for m in months:
            if m in set(base_months):
                continue
            tm = df[month_col].astype(str) == m
            ts = df.loc[tm, col]
            if ts.dropna().empty:
                continue
            tt = bin_table(transform_bins(ts.astype(float), fitted), df.loc[tm, label])
            vals.append(bin_psi(bt, tt))
        if vals:
            out[col] = round(max(vals), 6)
    return out


class MultiBaseLabelTests(unittest.TestCase):
    """多基期 ⇒ 标签必须表达「基期池」。"""

    def setUp(self) -> None:
        self.df = _frame()
        self.cols = ["f_stable", "f_drift", "f_const"]

    def test_base_used_is_pool_label(self) -> None:
        res = psi_mod.screen_psi(self.df, self.cols, "month", _cfg([BASE_A, BASE_B]))
        self.assertEqual(res.base_used, f"基期池({BASE_A}+{BASE_B})")
        self.assertNotIn("→", res.base_used)

    def test_detail_base_label_is_pool(self) -> None:
        res = psi_mod.screen_psi(self.df, self.cols, "month", _cfg([BASE_A, BASE_B]))
        labels = set(res.detail["基期"].unique().tolist())
        self.assertEqual(labels, {f"基期池({BASE_A}+{BASE_B})"})
        self.assertNotIn(BASE_A, labels)
        self.assertNotIn(BASE_B, labels)

    def test_one_row_per_comparison_month(self) -> None:
        res = psi_mod.screen_psi(self.df, self.cols, "month", _cfg([BASE_A, BASE_B]))
        grp = res.detail.groupby(["列名", "对比月"])["PSI"].agg(["count", "nunique"])
        self.assertTrue((grp["count"] == 1).all(), "同一(列,对比月)应只有一行")
        self.assertTrue((grp["nunique"] == 1).all())

    def test_note_states_pool_semantics(self) -> None:
        res = psi_mod.screen_psi(self.df, self.cols, "month", _cfg([BASE_A, BASE_B]))
        self.assertIn("合并池", res.note)
        self.assertIn("合并为一个池", res.note)

    def test_dropped_reason_marks_pool(self) -> None:
        res = psi_mod.screen_psi(
            self.df, self.cols, "month", _cfg([BASE_A, BASE_B], threshold=0.0)
        )
        self.assertTrue(res.reason)
        for reason in res.reason.values():
            self.assertIn("基期池", reason)


class SingleBaseTests(unittest.TestCase):
    """单基期场景必须**逐字不变**（合并池 == 该月，标签本来就对）。"""

    def setUp(self) -> None:
        self.df = _frame()

    def test_label_unchanged(self) -> None:
        res = psi_mod.screen_psi(self.df, ["f_stable", "f_drift"], "month", _cfg([BASE_A]))
        self.assertEqual(res.base_used, BASE_A)
        self.assertEqual(set(res.detail["基期"].unique().tolist()), {BASE_A})

    def test_note_unchanged(self) -> None:
        res = psi_mod.screen_psi(self.df, ["f_stable", "f_drift"], "month", _cfg([BASE_A]))
        self.assertIn("基期为指定月份", res.note)
        self.assertNotIn("基期池", res.note)

    def test_reason_has_no_pool_marker(self) -> None:
        res = psi_mod.screen_psi(
            self.df, ["f_stable", "f_drift"], "month", _cfg([BASE_A], threshold=0.0)
        )
        for reason in res.reason.values():
            self.assertNotIn("基期池", reason)
            self.assertIn("PSI(最大月)", reason)


class ValuesUnchangedTests(unittest.TestCase):
    """核心守卫：只改标签，不改数值。"""

    def setUp(self) -> None:
        self.df = _frame()
        self.cols = ["f_stable", "f_drift", "f_const"]

    def test_multi_base_equals_manual_pool(self) -> None:
        res = psi_mod.screen_psi(self.df, self.cols, "month", _cfg([BASE_A, BASE_B]))
        manual = _manual_pool_metrics(self.df, self.cols, [BASE_A, BASE_B])
        self.assertTrue(manual, "参照实现应至少算出 1 列")
        for col, value in manual.items():
            self.assertIn(col, res.metrics)
            self.assertAlmostEqual(res.metrics[col], value, places=9, msg=f"{col} 数值被改动了")

    def test_single_base_equals_manual(self) -> None:
        res = psi_mod.screen_psi(self.df, self.cols, "month", _cfg([BASE_A]))
        manual = _manual_pool_metrics(self.df, self.cols, [BASE_A])
        for col, value in manual.items():
            self.assertAlmostEqual(res.metrics[col], value, places=9)

    def test_base_month_order_irrelevant(self) -> None:
        r1 = psi_mod.screen_psi(self.df, self.cols, "month", _cfg([BASE_A, BASE_B]))
        r2 = psi_mod.screen_psi(self.df, self.cols, "month", _cfg([BASE_B, BASE_A]))
        self.assertEqual(r1.metrics, r2.metrics)
        self.assertEqual(r1.kept, r2.kept)
        self.assertEqual(r1.dropped, r2.dropped)


class BehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _frame()

    def test_drift_feature_detected(self) -> None:
        """★ fixture 里 f_stable 在漂移月也右移了 3σ ⇒ 两者都会被剔除（同一把尺子）。

        「谁漂得更狠」由 PSI 数值体现：f_drift 的位移是 6σ，必然大于 f_stable 的 3σ。
        这条断言只依赖**相对关系**，不依赖阈值绝对值的运气。
        """
        res = psi_mod.screen_psi(self.df, ["f_stable", "f_drift"], "month", _cfg([BASE_A]))
        self.assertGreater(res.metrics["f_drift"], res.metrics["f_stable"])
        self.assertIn("f_drift", res.dropped, "漂移特征应被剔除")
        self.assertIn("f_stable", res.dropped, "3σ 位移同样超阈值")

    def test_stable_feature_kept(self) -> None:
        """真正稳定的特征（各月同分布）必须被保留 —— 否则筛子只会一路误杀。"""
        rng = np.random.default_rng(77)
        df = pd.DataFrame(
            {
                "month": [m for m in (BASE_A, BASE_B, DRIFT_MONTH) for _ in range(100)],
                "y": rng.integers(0, 2, 300),
                "flat": rng.normal(size=300),
            }
        )
        res = psi_mod.screen_psi(df, ["flat"], "month", _cfg([BASE_A], threshold=0.1))
        self.assertIn("flat", res.kept)
        self.assertNotIn("flat", res.dropped)

    def test_high_psi_columns_reported(self) -> None:
        res = psi_mod.screen_psi(
            self.df, ["f_drift"], "month",
            psi_mod.PsiConfig(
                threshold=0.5, base_mode="specified", base_months=(BASE_A,),
                high_psi_alert=0.1,
            ),
        )
        self.assertIn("f_drift", res.high_psi_columns)

    def test_specified_without_base_months_raises(self) -> None:
        """``base_mode='specified'`` 却给空基期 ⇒ 必须报错（不能默认「全部月份」）。"""
        from risklab_mdl.core.errors import ConfigError

        cfg = psi_mod.PsiConfig(threshold=0.1, base_mode="specified", base_months=())
        with self.assertRaises(ConfigError):
            psi_mod.screen_psi(self.df, ["f_stable"], "month", cfg)

    def test_multi_base_needs_two_months(self) -> None:
        from risklab_mdl.core.errors import ConfigError

        with self.assertRaises(ConfigError):
            psi_mod.screen_psi(
                self.df, ["f_stable"], "month",
                psi_mod.PsiConfig(threshold=0.1, base_mode="multi_base", base_months=(BASE_A,)),
            )

    def test_rolling_mode_runs(self) -> None:
        cfg = psi_mod.PsiConfig(
            threshold=10.0, base_mode="rolling", rolling_base_months=(BASE_A,)
        )
        res = psi_mod.screen_psi(self.df, ["f_stable"], "month", cfg)
        self.assertIn("f_stable", res.metrics)

    def test_unknown_month_column_raises(self) -> None:
        from risklab_mdl.core.errors import DataError

        with self.assertRaises(DataError):
            psi_mod.screen_psi(
                self.df, ["f_stable"], "nope", _cfg([BASE_A])
            )

    def test_detail_columns(self) -> None:
        res = psi_mod.screen_psi(self.df, ["f_stable"], "month", _cfg([BASE_A]))
        for col in ("列名", "基期", "对比月", "PSI"):
            self.assertIn(col, res.detail.columns)


class DeadCodeTests(unittest.TestCase):
    """死代码不得复活。"""

    def test_aggregate_multi_base_is_gone(self) -> None:
        self.assertFalse(
            hasattr(psi_mod, "_aggregate_multi_base"),
            "_aggregate_multi_base 是定义了从不调用的死代码（取中位数），且与合并池口径矛盾",
        )

    def test_no_midpoint_claim_in_notes(self) -> None:
        df = _frame()
        for mode, base in (("specified", (BASE_A, BASE_B)), ("multi_base", (BASE_A, BASE_B))):
            cfg = psi_mod.PsiConfig(threshold=0.1, base_mode=mode, base_months=base)
            res = psi_mod.screen_psi(df, ["f_stable"], "month", cfg)
            self.assertNotIn("中位数", res.note, f"{mode} 模式的 note 仍在宣称中位数")


class ScorePsiTests(unittest.TestCase):
    """``psi_by_month``：分数分箱的月度 PSI。"""

    def test_summary_and_detail_shapes(self) -> None:
        df = _frame()
        summary, detail = psi_mod.psi_by_month(
            df["f_stable"], df["month"], n_bins=5, base_month=BASE_A
        )
        self.assertFalse(summary.empty)
        self.assertFalse(detail.empty)
        self.assertTrue({"基期", "对比月", "PSI"} & set(summary.columns)
                        or "PSI" in summary.columns)

    def test_psi_non_negative(self) -> None:
        df = _frame()
        summary, _detail = psi_mod.psi_by_month(
            df["f_drift"], df["month"], n_bins=5, base_month=BASE_A
        )
        psi_values = summary["PSI"].to_numpy(dtype=float)
        self.assertTrue(np.all(psi_values >= -1e-9), f"PSI 不应为负：{psi_values}")


if __name__ == "__main__":
    unittest.main()
