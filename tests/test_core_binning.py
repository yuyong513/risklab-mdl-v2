"""分箱 / WOE / IV / 分箱表 PSI / 决策树分箱 的口径测试。

守护点（每条都对应一个真实会静默出错的地方）：

* ``transform_bins`` 的缺失箱编码恒为 ``MISSING_BIN(-1)``；
* 稀疏箱合并**不得塌成单箱**（塌成单箱 ⇒ IV 恒为 0 ⇒ 初筛把所有特征判成无区分度）；
* ``bin_psi`` 必须让缺失箱参与对比、test 缺失的箱按 0 计；
* DT 分箱：箱数上限、最小箱占比/样本数、缺失独立成箱、切点严格单调、坏样本率单调；
* DT 的 ``edges`` 与 ``transform_bins`` 的装箱结果**逐行一致**（口径不漂移）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures.make_fixtures import binary_frame  # noqa: E402

from risklab_mdl.core.binning import (  # noqa: E402
    MISSING_BIN,
    BinSpec,
    bin_psi,
    bin_table,
    bin_table_processed,
    fit_bins,
    iv_of,
    transform_bins,
)
from risklab_mdl.core.binning_dt import (  # noqa: E402
    DTBinningSpec,
    dt_bin_table,
    fit_bins_dt,
)
from risklab_mdl.core.errors import ConfigError, DataError  # noqa: E402


def _frame() -> pd.DataFrame:
    df = binary_frame()
    return df


class BinSpecTests(unittest.TestCase):
    def test_defaults_and_derived_counts(self) -> None:
        spec = BinSpec()
        self.assertEqual(spec.method, "quantile")
        self.assertEqual(spec.n_bins, 8)
        self.assertTrue(spec.missing_as_bin)

    def test_effective_bins_follow_edges(self) -> None:
        spec = BinSpec(edges=(1.0, 2.0, 3.0))
        self.assertEqual(spec.n_valid_bins, 4)   # 3 切点 ⇒ 4 个有效箱
        self.assertEqual(spec.n_effective_bins, 5)  # 含缺失箱

    def test_method_none_is_single_bin(self) -> None:
        spec = BinSpec(method="none")
        self.assertEqual(spec.n_valid_bins, 1)
        self.assertEqual(spec.n_effective_bins, 2)

    def test_bins_downgraded_flag(self) -> None:
        self.assertFalse(BinSpec(requested_bins=8, edges=(1.0,) * 7).bins_downgraded)
        self.assertTrue(BinSpec(requested_bins=8, edges=(1.0,)).bins_downgraded)


class FitBinsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _frame()
        self.x = self.df["f_signal"]

    def test_quantile_edges_strictly_increasing(self) -> None:
        fitted = fit_bins(self.x, BinSpec(method="quantile", n_bins=5, min_bin_count=20))
        edges = list(fitted.edges)
        self.assertEqual(edges, sorted(edges))
        self.assertEqual(len(edges), len(set(edges)), "切点出现重复值")
        self.assertEqual(fitted.requested_bins, 5)

    def test_quantile_edges_within_range(self) -> None:
        fitted = fit_bins(self.x, BinSpec(n_bins=4, min_bin_count=20))
        lo, hi = float(self.x.min()), float(self.x.max())
        for edge in fitted.edges:
            self.assertGreaterEqual(edge, lo)
            self.assertLessEqual(edge, hi)

    def test_equal_width_edges_uniform(self) -> None:
        fitted = fit_bins(self.x, BinSpec(method="equal_width", n_bins=4, min_bin_count=20))
        edges = np.asarray(fitted.edges, dtype=float)
        self.assertEqual(len(edges), 3)
        diffs = np.diff(np.concatenate([[float(self.x.min())], edges, [float(self.x.max())]]))
        self.assertTrue(np.allclose(diffs, diffs[0], atol=1e-9), f"非等宽：{diffs}")

    def test_business_cut_sorts_input_edges(self) -> None:
        spec = BinSpec(method="business_cut", edges=(2.0, 0.0, 1.0))
        fitted = fit_bins(self.x, spec)
        self.assertEqual(list(fitted.edges), [0.0, 1.0, 2.0])

    def test_business_cut_without_edges_raises(self) -> None:
        with self.assertRaises(DataError):
            fit_bins(self.x, BinSpec(method="business_cut", edges=None))

    def test_requires_series(self) -> None:
        with self.assertRaises(DataError):
            fit_bins(self.x.to_numpy(), BinSpec())  # type: ignore[arg-type]

    def test_all_missing_raises(self) -> None:
        with self.assertRaises(DataError):
            fit_bins(pd.Series([np.nan, np.nan]), BinSpec())

    def test_adaptive_downgrade_on_small_sample(self) -> None:
        """样本量撑不起请求箱数时**主动下调**，并在 requested_bins 留痕。"""
        small = self.x.head(100)
        fitted = fit_bins(small, BinSpec(n_bins=8, min_bin_count=50))
        self.assertEqual(fitted.requested_bins, 8, "requested_bins 记的是请求值")
        self.assertLess(fitted.n_valid_bins, 8)
        self.assertTrue(fitted.bins_downgraded, "箱数下调必须可被报表识别")

    def test_constant_feature_degrades_to_no_discrimination(self) -> None:
        """常量特征：切点收敛到 1 个（右箱为空箱）⇒ IV 必须为 0，不得虚高。

        ★ 这是**旧口径的实际行为**（``np.unique`` 把重复切点收敛成 1 个，
          而不是退化为 0 个）：留下 `(1, +inf)` 这个空箱，IV 仍为 0。
          测试记录事实，避免有人以为「常量特征会得到 0 个切点」。
        """
        x = pd.Series(np.ones(50))
        y = pd.Series([0, 1] * 25)
        fitted = fit_bins(x, BinSpec(n_bins=5, min_bin_count=10))
        self.assertLessEqual(len(fitted.edges), 1)
        self.assertLessEqual(fitted.n_valid_bins, 2)
        codes = transform_bins(x, fitted)
        self.assertEqual(set(codes.unique().tolist()), {0})
        table = bin_table_processed(codes, y, fitted)
        self.assertAlmostEqual(iv_of(table), 0.0, places=9)


class TransformBinsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _frame()
        self.x = self.df["f_missing"]  # 含约 17~20% 缺失
        self.spec = fit_bins(self.x, BinSpec(n_bins=4, min_bin_count=20))

    def test_missing_goes_to_missing_bin(self) -> None:
        codes = transform_bins(self.x, self.spec)
        self.assertEqual(codes.dtype, np.dtype("int64"))
        na_rows = self.x.isna()
        self.assertTrue((codes[na_rows] == MISSING_BIN).all())
        self.assertTrue((codes[~na_rows] >= 0).all())

    def test_missing_kept_as_na_when_disabled(self) -> None:
        spec = BinSpec(
            method="business_cut", edges=self.spec.edges, missing_as_bin=False
        )
        codes = transform_bins(self.x, spec)
        self.assertEqual(str(codes.dtype), "Int64")
        self.assertTrue(codes[self.x.isna()].isna().all())

    def test_codes_are_ascending_in_value(self) -> None:
        codes = transform_bins(self.x, self.spec)
        frame = pd.DataFrame({"x": self.x, "c": codes}).dropna()
        for code, part in frame.groupby("c"):
            self.assertGreaterEqual(part["x"].min(), float(self.spec.edges[code - 1])
                                    if code > 0 else -np.inf)
            if code < len(self.spec.edges):
                self.assertLessEqual(part["x"].max(), float(self.spec.edges[code]))

    def test_method_none_gives_zero(self) -> None:
        codes = transform_bins(self.x, BinSpec(method="none"))
        self.assertTrue((codes == 0).all())

    def test_transform_keeps_index(self) -> None:
        codes = transform_bins(self.x, self.spec)
        self.assertTrue(codes.index.equals(self.x.index))


class BinTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _frame()
        self.spec = fit_bins(self.df["f_signal"], BinSpec(n_bins=5, min_bin_count=20))
        self.codes = transform_bins(self.df["f_signal"], self.spec)

    def test_bin_table_columns_and_totals(self) -> None:
        table = bin_table(self.codes, self.df["y"])
        for col in ("bin", "总用户", "坏用户", "好用户", "坏样本率", "占比"):
            self.assertIn(col, table.columns)
        self.assertEqual(int(table["总用户"].sum()), len(self.df))
        self.assertAlmostEqual(float(table["占比"].sum()), 1.0, places=9)
        self.assertTrue((table["好用户"] >= 0).all())

    def test_bin_table_length_mismatch_raises(self) -> None:
        with self.assertRaises(DataError):
            bin_table(self.codes, self.df["y"].head(10))

    def test_processed_table_has_woe_and_iv(self) -> None:
        table = bin_table_processed(self.codes, self.df["y"], self.spec)
        for col in ("WOE", "IV贡献", "是否平滑", "WOE是否截断"):
            self.assertIn(col, table.columns)
        iv = iv_of(table)
        self.assertGreater(iv, 0.1, f"强信号特征的 IV 应明显大于 0，实际 {iv}")
        self.assertAlmostEqual(iv, float(table["IV贡献"].sum()), places=12)

    def test_iv_of_requires_processed_table(self) -> None:
        raw = bin_table(self.codes, self.df["y"])
        with self.assertRaises(DataError):
            iv_of(raw)

    def test_noise_feature_has_lower_iv_than_signal(self) -> None:
        def _iv(col: str) -> float:
            spec = fit_bins(self.df[col], BinSpec(n_bins=5, min_bin_count=20))
            codes = transform_bins(self.df[col], spec)
            return iv_of(bin_table_processed(codes, self.df["y"], spec))

        self.assertLess(_iv("f_noise"), _iv("f_signal"))

    def test_noise_iv_not_zero_bins_collapse(self) -> None:
        """★ 稀疏合并的底线：噪声特征也必须留下 ≥2 个有效箱，否则 IV 恒为 0。"""
        spec = BinSpec(n_bins=8, min_bin_count=50)
        fitted = fit_bins(self.df["f_noise"], spec)
        codes = transform_bins(self.df["f_noise"], fitted)
        table = bin_table_processed(codes, self.df["y"], fitted)
        n_valid = int((table["bin"] >= 0).sum())
        self.assertGreaterEqual(n_valid, 2, f"箱被合并成 {n_valid} 个 ⇒ IV 将恒为 0")

    def test_bad_sample_rate_monotone_on_signal(self) -> None:
        """强信号特征在 5 箱下坏样本率应基本单调递增（口径正确性的直观校验）。"""
        table = bin_table(self.codes, self.df["y"])
        rates = table.loc[table["bin"] >= 0, "坏样本率"].to_numpy()
        self.assertGreater(rates[-1], rates[0] * 3, f"首末箱坏样本率差异过小：{rates}")


class BinPsiTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(5)
        self.base = pd.Series(rng.normal(size=500))
        self.same = pd.Series(rng.normal(size=500))
        self.shifted = pd.Series(rng.normal(loc=2.0, size=500))
        self.spec = fit_bins(self.base, BinSpec(n_bins=5, min_bin_count=30))

    def _table(self, series: pd.Series) -> pd.DataFrame:
        return bin_table(transform_bins(series, self.spec), pd.Series(np.zeros(len(series))))

    def test_identical_distribution_gives_near_zero(self) -> None:
        psi = bin_psi(self._table(self.base), self._table(self.base))
        self.assertAlmostEqual(psi, 0.0, places=9)

    def test_shifted_distribution_gives_positive(self) -> None:
        psi = bin_psi(self._table(self.base), self._table(self.shifted))
        self.assertGreater(psi, 0.25, f"分布右移 2σ 的 PSI 应显著，实际 {psi}")

    def test_missing_bin_participates(self) -> None:
        """★ 缺失箱必须参与对比（旧实现用 -99 哨兵会把缺失混进正常取值）。"""
        base = self.base.copy()
        base.iloc[:50] = np.nan
        test = self.shifted.copy()
        test.iloc[:200] = np.nan
        psi_with_missing = bin_psi(self._table(base), self._table(test))
        psi_no_missing = bin_psi(self._table(self.base), self._table(self.shifted))
        self.assertNotAlmostEqual(psi_with_missing, psi_no_missing, places=6)

    def test_empty_test_bin_treated_as_zero(self) -> None:
        """test 里不存在的箱按占比 0 计（跳过会让 PSI 被低估）。"""
        base_table = self._table(self.base)
        test_table = self._table(self.shifted).head(2)  # 人为丢箱
        psi = bin_psi(base_table, test_table)
        self.assertGreater(psi, 0.0)

    def test_requires_bin_and_share_columns(self) -> None:
        with self.assertRaises(DataError):
            bin_psi(pd.DataFrame({"bin": [0]}), pd.DataFrame({"bin": [0]}))


class DTBinningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _frame()
        self.x = self.df["f_signal"]
        self.y = self.df["y"]

    def test_bin_count_within_limit(self) -> None:
        for max_bins in (2, 3, 5, 8):
            result = fit_bins_dt(
                self.x, self.y, DTBinningSpec(max_bins=max_bins, min_bin_count=20)
            )
            self.assertLessEqual(result.n_bins, max_bins, f"max_bins={max_bins}")
            self.assertLessEqual(len(result.edges), max_bins - 1)

    def test_edges_strictly_increasing(self) -> None:
        result = fit_bins_dt(self.x, self.y, DTBinningSpec(max_bins=6, min_bin_count=20))
        edges = list(result.edges)
        self.assertEqual(edges, sorted(edges))
        self.assertEqual(len(edges), len(set(edges)), f"切点重复：{edges}")

    def test_min_bin_count_respected(self) -> None:
        spec = DTBinningSpec(max_bins=8, min_bin_count=40, min_bin_pct=0.0)
        result = fit_bins_dt(self.x, self.y, spec)
        for count in result.bin_counts:
            self.assertGreaterEqual(count, 40, f"箱样本数不足：{result.bin_counts}")

    def test_min_bin_pct_respected(self) -> None:
        spec = DTBinningSpec(max_bins=6, min_bin_count=0, min_bin_pct=0.10)
        result = fit_bins_dt(self.x, self.y, spec)
        n = int(self.x.notna().sum())
        for count in result.bin_counts:
            self.assertGreaterEqual(count, int(np.ceil(0.10 * n)) - 1,
                                    f"箱占比不足：{result.bin_counts}")

    def test_missing_is_separate_bin(self) -> None:
        x = self.x.copy()
        x.iloc[:40] = np.nan
        result = fit_bins_dt(x, self.y, DTBinningSpec(max_bins=5, min_bin_count=20))
        self.assertEqual(result.missing_count, 40)
        self.assertTrue(result.spec.missing_as_bin)
        codes = transform_bins(x, result.spec)
        self.assertEqual(int((codes == MISSING_BIN).sum()), 40)
        self.assertGreaterEqual(int((codes >= 0).sum()), len(x) - 40)

    def test_bad_rates_monotone_by_default(self) -> None:
        result = fit_bins_dt(self.x, self.y, DTBinningSpec(max_bins=6, min_bin_count=20))
        self.assertTrue(result.is_monotonic, f"坏样本率不单调：{result.bin_bad_rates}")
        rates = result.bin_bad_rates
        self.assertTrue(
            all(rates[i] <= rates[i + 1] + 1e-9 for i in range(len(rates) - 1))
            or all(rates[i] >= rates[i + 1] - 1e-9 for i in range(len(rates) - 1))
        )

    def test_edges_reproduce_transform_bins(self) -> None:
        """★ 口径一致性：DT 输出的切点交给 transform_bins 装箱，结果与拟合时逐箱相同。"""
        result = fit_bins_dt(self.x, self.y, DTBinningSpec(max_bins=6, min_bin_count=20))
        codes = transform_bins(self.x, result.spec)
        counts = codes.value_counts().sort_index()
        positive = [int(counts.get(i, 0)) for i in range(result.n_bins)]
        self.assertEqual(positive, list(result.bin_counts))

    def test_dt_bin_table_matches(self) -> None:
        result = fit_bins_dt(self.x, self.y, DTBinningSpec(max_bins=5, min_bin_count=20))
        table = dt_bin_table(result, self.x, self.y)
        self.assertEqual(int(table["总用户"].sum()), len(self.x))

    def test_iv_of_dt_bins_is_positive(self) -> None:
        result = fit_bins_dt(self.x, self.y, DTBinningSpec(max_bins=6, min_bin_count=20))
        codes = transform_bins(self.x, result.spec)
        table = bin_table_processed(codes, self.y, result.spec)
        self.assertGreater(iv_of(table), 0.1)

    def test_spec_is_binspec_compatible(self) -> None:
        result = fit_bins_dt(self.x, self.y)
        self.assertIsInstance(result.spec, BinSpec)
        self.assertEqual(result.spec.method, "business_cut")
        self.assertEqual(result.spec.edges, result.edges)

    def test_near_constant_feature_degrades_with_note(self) -> None:
        x = pd.Series(np.round(self.x.to_numpy() * 0.0) + 1.0)
        x.iloc[0] = 2.0
        result = fit_bins_dt(x, self.y, DTBinningSpec(max_bins=4, min_bin_count=5))
        self.assertLessEqual(result.n_bins, 2)
        self.assertTrue(result.notes, "退化必须留 note，不能静默")

    def test_rejects_non_binary_label(self) -> None:
        y = self.y.copy()
        y.iloc[:5] = 2
        with self.assertRaises(DataError):
            fit_bins_dt(self.x, y)

    def test_rejects_single_class(self) -> None:
        with self.assertRaises(DataError):
            fit_bins_dt(self.x, pd.Series(np.zeros(len(self.x))))

    def test_rejects_all_missing(self) -> None:
        with self.assertRaises(DataError):
            fit_bins_dt(pd.Series([np.nan] * 50), pd.Series([0, 1] * 25))

    def test_rejects_bad_config(self) -> None:
        with self.assertRaises(ConfigError):
            DTBinningSpec(max_bins=1)
        with self.assertRaises(ConfigError):
            DTBinningSpec(max_bins=10, min_bin_pct=0.5)
        with self.assertRaises(ConfigError):
            DTBinningSpec(min_bin_pct=1.5)
        with self.assertRaises(ConfigError):
            DTBinningSpec(criterion="mae")

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(DataError):
            fit_bins_dt(self.x, self.y.head(10))

    def test_result_to_dict_is_json_safe(self) -> None:
        import json

        result = fit_bins_dt(self.x, self.y)
        payload = json.dumps(result.to_dict(), ensure_ascii=False)
        self.assertIn("edges", json.loads(payload))

    def test_multiclass_label_message_mentions_ovr(self) -> None:
        y = pd.Series(np.tile([0, 1, 2], 20))
        x = pd.Series(np.arange(60, dtype=float))
        with self.assertRaises(DataError) as ctx:
            fit_bins_dt(x, y)
        self.assertIn("one-vs-rest", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
