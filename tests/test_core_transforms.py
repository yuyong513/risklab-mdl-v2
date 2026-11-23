"""预处理变换（缺失填充 / 编码 / 异常值）的直接单测。

★ 为什么 core 层要有这些测试
--------------------------
``transforms.py`` 是**唯一**会被 T4（``preprocess_apply``）直接接线的模块。
旧项目里它「写了但从未接线」，因此长期没有任何测试覆盖 —— 一旦接线才发现
``apply_fill`` 的哨兵值与画像里的哨兵冲突判定不一致。

⇒ 本文件按「编码 / 填充 / 异常值各一例」直接调用（不经过任何流程层），
   并额外验证：变换**不改原表**、缺列报错可读、策略非法报 ConfigError。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures.make_fixtures import missing_text_frame  # noqa: E402

from risklab_mdl.core.errors import ConfigError, DataError  # noqa: E402
from risklab_mdl.core.transforms import (  # noqa: E402
    SENTINEL_FILL_VALUE,
    EncodeSpec,
    FillSpec,
    OutlierSpec,
    apply_encode,
    apply_fill,
    apply_outlier,
    profile_features,
)


def _frame() -> pd.DataFrame:
    return missing_text_frame()


class FillTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _frame()

    def test_sentinel_fill(self) -> None:
        out = apply_fill(self.df, {"amount": FillSpec("amount", "sentinel", 88888.0)})
        self.assertEqual(int(out["amount"].isna().sum()), 0)
        self.assertEqual(
            int((out["amount"] == 88888.0).sum()),
            int(self.df["amount"].isna().sum()),
            "哨兵值个数必须等于原缺失个数",
        )
        self.assertEqual(len(out), len(self.df))

    def test_sentinel_conflict_rejected(self) -> None:
        """★ 哨兵值冲突必须报错：真实取值里已有 99999，再用它填缺失 ⇒
        模型会把真实值与缺失混为一谈。"""
        with self.assertRaises(ConfigError) as ctx:
            apply_fill(self.df, {"amount": FillSpec("amount", "sentinel", 99999.0)})
        self.assertIn("99999", str(ctx.exception))

    def test_median_fill(self) -> None:
        out = apply_fill(self.df, {"amount": FillSpec("amount", "median")})
        expected = float(self.df["amount"].median())
        filled = out.loc[self.df["amount"].isna(), "amount"]
        self.assertTrue(np.allclose(filled.to_numpy(), expected))
        self.assertTrue(np.allclose(
            out.loc[~self.df["amount"].isna(), "amount"],
            self.df.loc[~self.df["amount"].isna(), "amount"],
        ), "非缺失行不得被改动")

    def test_mode_fill_for_text(self) -> None:
        out = apply_fill(self.df, {"channel": FillSpec("channel", "mode")})
        self.assertEqual(int(out["channel"].isna().sum()), 0)

    def test_input_frame_not_mutated(self) -> None:
        before = self.df.copy(deep=True)
        apply_fill(self.df, {"amount": FillSpec("amount", "mean")})
        pd.testing.assert_frame_equal(self.df, before)

    def test_missing_column_raises(self) -> None:
        with self.assertRaises((DataError, ConfigError)):
            apply_fill(self.df, {"nope": FillSpec("nope", "mean")})

    def test_default_sentinel_constant(self) -> None:
        self.assertEqual(SENTINEL_FILL_VALUE, 99999.0)
        spec = FillSpec("amount")
        self.assertEqual(spec.value, SENTINEL_FILL_VALUE)


class EncodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _frame()

    def test_ordinal_with_explicit_mapping(self) -> None:
        mapping = {"app": 0.0, "web": 1.0, "store": 2.0}
        out = apply_encode(self.df, {"channel": EncodeSpec("channel", "ordinal", mapping)})
        encoded = out.loc[self.df["channel"] == "web", "channel"]
        self.assertTrue((encoded == 1.0).all())
        self.assertNotIn("channel", [c for c in out.columns if out[c].dtype == object])

    def test_onehot(self) -> None:
        out = apply_encode(self.df, {"channel": EncodeSpec("channel", "onehot")})
        new_cols = [c for c in out.columns if c.startswith("channel")]
        self.assertGreaterEqual(len(new_cols), 3, f"onehot 列：{new_cols}")
        for col in new_cols:
            self.assertTrue(pd.api.types.is_numeric_dtype(out[col]))

    def test_drop_strategy(self) -> None:
        out = apply_encode(self.df, {"channel": EncodeSpec("channel", "drop")})
        self.assertNotIn("channel", out.columns)
        self.assertEqual(len(out), len(self.df))

    def test_unknown_strategy_rejected(self) -> None:
        with self.assertRaises((ConfigError, DataError)):
            apply_encode(self.df, {"channel": EncodeSpec("channel", "magic")})  # type: ignore[arg-type]

    def test_needs_fitted_table_rejected(self) -> None:
        """target / binned 编码必须带已拟合表 —— 预处理阶段不得隐式拟合（防泄漏）。"""
        with self.assertRaises(ConfigError):
            apply_encode(self.df, {"channel": EncodeSpec("channel", "target")})

    def test_input_frame_not_mutated(self) -> None:
        before = self.df.copy(deep=True)
        apply_encode(self.df, {"channel": EncodeSpec("channel", "onehot")})
        pd.testing.assert_frame_equal(self.df, before)


class OutlierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _frame()

    def test_mark_adds_flag_column(self) -> None:
        out = apply_outlier(self.df, {"score": OutlierSpec("score", "mark")})
        flag = "score__outlier_flag"
        self.assertIn(flag, out.columns)
        self.assertGreater(int(out[flag].sum()), 0)
        # mark 只打标，不改取值
        self.assertTrue(np.allclose(out["score"], self.df["score"], equal_nan=True))

    def test_clip_winsorize_bounds_values(self) -> None:
        out = apply_outlier(
            self.df, {"score": OutlierSpec("score", "winsorize", sigma_multiple=2.0)}
        )
        self.assertLess(float(out["score"].max()), float(self.df["score"].max()))
        self.assertGreaterEqual(float(out["score"].min()), float(self.df["score"].min()))

    def test_to_missing(self) -> None:
        out = apply_outlier(self.df, {"score": OutlierSpec("score", "to_missing")})
        self.assertGreater(int(out["score"].isna().sum()), int(self.df["score"].isna().sum()))

    def test_flags_out_parameter(self) -> None:
        flags: dict[str, pd.Series] = {}
        apply_outlier(self.df, {"score": OutlierSpec("score", "mark")}, flags=flags)
        self.assertTrue(flags, "flags 输出参数应带回逐列标记（T4 要落盘）")


class ProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _frame()

    def test_profile_columns_and_flags(self) -> None:
        table = profile_features(self.df, ["amount", "channel", "score", "const"])
        for col in ("列名", "类型", "基数", "缺失率", "最高频取值占比"):
            self.assertIn(col, table.frame.columns)
        self.assertEqual(len(table.frame), 4)
        self.assertGreater(table.profiles["amount"].missing_rate, 0.1)
        self.assertFalse(table.profiles["amount"].needs_encode)
        self.assertTrue(table.profiles["channel"].needs_encode)
        self.assertAlmostEqual(table.profiles["const"].top_value_share, 1.0, places=9)

    def test_default_specs_generated(self) -> None:
        table = profile_features(self.df, ["amount", "channel", "score", "const"])
        self.assertIn("channel", table.encode_specs)
        self.assertIn("amount", table.fill_specs)

    def test_unknown_column_raises(self) -> None:
        with self.assertRaises(DataError):
            profile_features(self.df, ["nope"])

    def test_restricted_columns_keyword_hit(self) -> None:
        df = self.df.rename(columns={"score": "id_card_score"})
        table = profile_features(df, ["id_card_score", "amount"])
        self.assertIn("id_card_score", table.restricted_columns(["id_card"]))


if __name__ == "__main__":
    unittest.main()
