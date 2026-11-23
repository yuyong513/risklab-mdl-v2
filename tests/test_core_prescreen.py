"""初筛五道筛子（缺失率 / 同值率 / IV / PSI / 相关性）的判定测试。

★ 守护点
--------
筛子是**串联**的：前一道把列剔掉，后一道就看不到它。
最容易出的错是「各道筛子各自算、结果拼不起来」——
``survived`` 与 ``stage_dropped`` 必须能互相印证（
``输入 = survived + Σ stage_dropped``，且每个掉落列都有原因）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures.make_fixtures import binary_frame  # noqa: E402

from risklab_mdl.core.binning import BinSpec  # noqa: E402
from risklab_mdl.core.errors import ConfigError  # noqa: E402
from risklab_mdl.core.prescreen import (  # noqa: E402
    PrescreenConfig,
    run_prescreen,
    screen_correlation,
    screen_iv,
    screen_missing_rate,
    screen_same_value_rate,
)

FEATURES = ["f_signal", "f_weak", "f_noise", "f_dup", "f_missing", "f_const"]


def _cfg(**overrides) -> PrescreenConfig:
    base = dict(
        missing_threshold=0.5,
        same_value_threshold=0.99,
        iv_threshold=0.02,
        psi_threshold=0.1,
        corr_threshold=0.8,
        bin_spec=BinSpec(n_bins=5, min_bin_count=20),
        enable_psi=False,
    )
    base.update(overrides)
    return PrescreenConfig(**base)


class ScreenFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = binary_frame()
        self.y = self.df["y"]

    def test_missing_rate_screen(self) -> None:
        df = self.df.copy()
        df["f_mostly_missing"] = np.nan
        df.loc[df.index[:20], "f_mostly_missing"] = 1.0
        result = screen_missing_rate(
            df, list(FEATURES) + ["f_mostly_missing"], threshold=0.5
        )
        self.assertIn("f_mostly_missing", result.dropped)
        self.assertIn("f_signal", result.metrics)
        self.assertLess(result.metrics["f_signal"], 0.5)
        self.assertNotIn("f_signal", result.dropped)
        self.assertIn("缺失率", result.reason["f_mostly_missing"])

    def test_same_value_screen(self) -> None:
        result = screen_same_value_rate(self.df, FEATURES, threshold=0.99)
        self.assertIn("f_const", result.dropped)
        self.assertNotIn("f_signal", result.dropped)
        self.assertAlmostEqual(result.metrics["f_const"], 1.0, places=9)

    def test_iv_screen_keeps_signal(self) -> None:
        result = screen_iv(
            self.df, FEATURES, self.y, threshold=0.02,
            bin_spec=BinSpec(n_bins=5, min_bin_count=20),
        )
        self.assertIn("f_signal", result.metrics)
        self.assertGreater(result.metrics["f_signal"], 0.02)
        self.assertNotIn("f_signal", result.dropped)

    def test_iv_screen_drops_constant_column(self) -> None:
        """常量列 IV 恒为 0 ⇒ 必须被 IV 筛子剔除（不能「算出 0 就放行」）。"""
        result = screen_iv(
            self.df, ["f_const"], self.y, threshold=0.02,
            bin_spec=BinSpec(n_bins=5, min_bin_count=20),
        )
        self.assertIn("f_const", result.dropped)

    def test_correlation_screen_drops_duplicate_keeping_stronger(self) -> None:
        importance = {"f_signal": 1.0, "f_dup": 0.5, "f_weak": 0.2, "f_noise": 0.1}
        result = screen_correlation(
            self.df, ["f_signal", "f_dup", "f_weak", "f_noise"], importance,
            threshold=0.8, strategy="greedy",
        )
        self.assertNotIn("f_signal", result.dropped, "重要性最高的列必须保留")
        self.assertIn("f_dup", result.dropped, "高相关副本应被剔除")
        self.assertIn("f_signal", result.reason["f_dup"])


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = binary_frame()
        self.y = self.df["y"]

    def test_survived_and_dropped_partition_input(self) -> None:
        result = run_prescreen(self.df, FEATURES, self.y, _cfg())
        dropped_all = [c for cols in result.stage_dropped.values() for c in cols]
        self.assertEqual(
            sorted(result.survived + dropped_all), sorted(FEATURES),
            "存活列 + 各阶段剔除列 必须恰好等于输入列（不重不漏）",
        )
        self.assertEqual(result.total_input, len(FEATURES))

    def test_every_dropped_column_has_reason(self) -> None:
        result = run_prescreen(self.df, FEATURES, self.y, _cfg())
        dropped_all = [c for cols in result.stage_dropped.values() for c in cols]
        for col in dropped_all:
            found = any(col in stage_reasons for stage_reasons in result.stage_reason.values())
            self.assertTrue(found, f"{col} 被剔除但没有原因说明")

    def test_signal_survives_full_pipeline(self) -> None:
        result = run_prescreen(self.df, FEATURES, self.y, _cfg())
        self.assertIn("f_signal", result.survived)
        self.assertNotIn("f_const", result.survived)
        self.assertGreater(result.iv_map["f_signal"], 0.02)

    def test_psi_stage_runs_with_month_column(self) -> None:
        cfg = _cfg(
            enable_psi=True,
            month_column="month",
            psi_base_mode="specified",
            psi_base_months=("2026-01",),
            psi_threshold=0.3,  # 放宽 ⇒ 不应因 PSI 剔除
        )
        result = run_prescreen(self.df, FEATURES, self.y, cfg)
        self.assertIn("psi", result.stage_metrics)
        self.assertIn("f_signal", result.survived)

    def test_config_validation(self) -> None:
        with self.assertRaises(ConfigError):
            _cfg(missing_threshold=1.5)
        with self.assertRaises(ConfigError):
            _cfg(iv_threshold=0.0)
        with self.assertRaises(ConfigError):
            _cfg(enable_psi=True, month_column="")


if __name__ == "__main__":
    unittest.main()
