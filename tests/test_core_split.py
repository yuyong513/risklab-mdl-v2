"""样本划分测试：SplitSpec / split_frame（纯函数）+ 字段表达式受限解析器。

守护点（每条都对着一个会**静默改变口径**的坑）：

* 同 seed 两次划分的掩码**逐行一致**（指纹相同）—— 否则复现性无从谈起；
* 随机比例误差 < 1%，分层各类占比偏差 < 2%；
* 字段表达式只认白名单语法：``__import__`` / 属性访问 / 赋值 一律 ConfigError；
* 三个子集互斥（重叠 ⇒ 同一行既在 train 又在 test ⇒ 指标虚高且不可见）；
* ``split_frame`` 是纯函数：不修改输入 df、不增加列。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures.make_fixtures import binary_frame  # noqa: E402

from risklab_mdl.core.errors import ConfigError, DataError  # noqa: E402
from risklab_mdl.core.expr import (  # noqa: E402
    compile_expression,
    evaluate_expression,
    expression_columns,
)
from risklab_mdl.core.split import (  # noqa: E402
    SPLIT_STAGE_REFINE,
    SplitSpec,
    earliest_month_masks,
    resolve_split_subset,
    split_frame,
)


def _frame() -> pd.DataFrame:
    return binary_frame()


class SplitSpecTests(unittest.TestCase):
    def test_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ConfigError):
            SplitSpec(mode="kfold")  # type: ignore[arg-type]

    def test_rejects_bad_ratios(self) -> None:
        with self.assertRaises(ConfigError):
            SplitSpec(mode="random", ratios=(0.5, 0.2, 0.1))
        with self.assertRaises(ConfigError):
            SplitSpec(mode="random", ratios=(-0.1, 0.6, 0.5))
        with self.assertRaises(ConfigError):
            SplitSpec(mode="random", ratios=(0.7, 0.3))

    def test_field_requires_expression(self) -> None:
        with self.assertRaises(ConfigError):
            SplitSpec(mode="field")

    def test_from_config_accepts_contract_shapes(self) -> None:
        a = SplitSpec.from_config({"mode": "random", "ratios": [0.7, 0.2, 0.1], "seed": 1})
        b = SplitSpec.from_config({"mode": "random", "train": 0.7, "val": 0.2, "test": 0.1, "seed": 1})
        c = SplitSpec.from_config(
            {"mode": "random", "train": {"ratio": 0.7, "seed": 1}, "val": 0.2, "test": 0.1}
        )
        for spec in (a, b, c):
            self.assertEqual(spec.ratios, (0.7, 0.2, 0.1))
            self.assertEqual(spec.seed, 1)

    def test_from_config_field_block(self) -> None:
        spec = SplitSpec.from_config(
            {
                "mode": "field",
                "field": {
                    "column": "month",
                    "expr_train": "month == '2026-01'",
                    "expr_val": "month == '2026-02'",
                    "expr_test": "month == '2026-03'",
                },
            }
        )
        self.assertEqual(spec.column, "month")
        self.assertEqual(set(spec.expression), {"train", "val", "test"})

    def test_to_dict_roundtrip(self) -> None:
        spec = SplitSpec.from_config({"mode": "stratified", "ratios": [0.6, 0.2, 0.2], "seed": 9,
                                      "stratify_by": "y"})
        payload = spec.to_dict()
        self.assertEqual(payload["mode"], "stratified")
        self.assertEqual(payload["seed"], 9)
        self.assertEqual(SplitSpec.from_config(payload).ratios, spec.ratios)


class NoneModeTests(unittest.TestCase):
    def test_everything_is_train(self) -> None:
        df = _frame()
        result = split_frame(df, SplitSpec(mode="none"))
        self.assertEqual(result.counts["train"], len(df))
        self.assertEqual(result.counts["val"], 0)
        self.assertEqual(result.counts["test"], 0)
        self.assertTrue(result.warnings, "不划分必须显式提示「指标非样本外」")

    def test_subset_frames(self) -> None:
        df = _frame()
        result = split_frame(df, SplitSpec())
        train, val, test = result.frames(df)
        self.assertEqual(len(train), len(df))
        self.assertEqual(len(val), 0)
        self.assertEqual(len(test), 0)


class RandomModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _frame()

    def test_same_seed_same_masks(self) -> None:
        a = split_frame(self.df, SplitSpec(mode="random", seed=7))
        b = split_frame(self.df, SplitSpec(mode="random", seed=7))
        for name in ("train", "val", "test"):
            self.assertTrue(
                a.mask(name).equals(b.mask(name)), f"{name} 掩码不一致 ⇒ 不可复现"
            )
        self.assertEqual(a.fingerprint, b.fingerprint)

    def test_different_seed_differs(self) -> None:
        a = split_frame(self.df, SplitSpec(mode="random", seed=7))
        b = split_frame(self.df, SplitSpec(mode="random", seed=8))
        self.assertNotEqual(a.fingerprint, b.fingerprint)

    def test_ratio_error_below_one_percent(self) -> None:
        df = pd.concat([self.df] * 3, ignore_index=True)  # 1200 行
        result = split_frame(df, SplitSpec(mode="random", seed=42))
        for name, expected in zip(("train", "val", "test"), (0.7, 0.2, 0.1)):
            actual = result.counts[name] / len(df)
            self.assertLess(abs(actual - expected), 0.01, f"{name}: {actual} vs {expected}")

    def test_masks_are_disjoint_and_complete(self) -> None:
        result = split_frame(self.df, SplitSpec(mode="random", seed=42))
        masks = result.masks()
        total = sum(m.to_numpy().astype(int) for m in masks.values())
        self.assertTrue(np.all(total == 1), "有行未被划分或被重复划分")
        self.assertFalse((masks["train"] & masks["test"]).any())

    def test_index_preserved(self) -> None:
        result = split_frame(self.df, SplitSpec(mode="random"))
        self.assertTrue(result.train_mask.index.equals(self.df.index))
        self.assertEqual(result.train_mask.dtype, np.dtype("bool"))


class StratifiedModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _frame()

    def test_class_shares_within_two_percent(self) -> None:
        result = split_frame(self.df, SplitSpec(mode="stratified", seed=42), target="y")
        overall = self.df["y"].value_counts(normalize=True).to_dict()
        for name in ("train", "val", "test"):
            part = self.df.loc[result.mask(name), "y"]
            shares = part.value_counts(normalize=True).to_dict()
            for label, expected in overall.items():
                self.assertLess(
                    abs(shares.get(label, 0.0) - expected),
                    0.02,
                    f"{name} 类 {label} 占比偏差过大：{shares} vs {overall}",
                )

    def test_reproducible(self) -> None:
        a = split_frame(self.df, SplitSpec(mode="stratified", seed=5), target="y")
        b = split_frame(self.df, SplitSpec(mode="stratified", seed=5), target="y")
        self.assertEqual(a.fingerprint, b.fingerprint)

    def test_requires_stratify_column(self) -> None:
        with self.assertRaises(ConfigError):
            split_frame(self.df, SplitSpec(mode="stratified"))

    def test_missing_column_raises(self) -> None:
        with self.assertRaises(ConfigError):
            split_frame(self.df, SplitSpec(mode="stratified", stratify_by="nope"), target="y")

    def test_high_cardinality_warns(self) -> None:
        df = self.df.copy()
        df["uid_stratum"] = [f"s{i}" for i in range(len(df))]
        result = split_frame(
            df, SplitSpec(mode="stratified", stratify_by="uid_stratum", seed=1)
        )
        self.assertTrue(any("分层过细" in w for w in result.warnings))


class FieldModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _frame()

    def test_in_and_comparison_counts(self) -> None:
        spec = SplitSpec(
            mode="field",
            column="month",
            expression={
                "train": "month in ('2026-01','2026-02')",
                "val": "month == '2026-03'",
            },
        )
        result = split_frame(self.df, spec)
        manual_train = int(self.df["month"].isin(["2026-01", "2026-02"]).sum())
        manual_val = int((self.df["month"] == "2026-03").sum())
        self.assertEqual(result.counts["train"], manual_train)
        self.assertEqual(result.counts["val"], manual_val)
        self.assertEqual(result.counts["test"], 0)

    def test_and_or_parentheses(self) -> None:
        spec = SplitSpec(
            mode="field",
            expression={
                "train": "(month == '2026-01' | month == '2026-02') & f_signal > 0",
                "val": "month == '2026-03' & f_signal <= 0",
            },
        )
        result = split_frame(self.df, spec)
        manual = self.df[
            self.df["month"].isin(["2026-01", "2026-02"]) & (self.df["f_signal"] > 0)
        ]
        self.assertEqual(result.counts["train"], len(manual))

    def test_fields_on_left_side(self) -> None:
        spec = SplitSpec(mode="field", expression={"train": "0 <= f_weak"})
        result = split_frame(self.df, spec)
        self.assertEqual(result.counts["train"], int((self.df["f_weak"] >= 0).sum()))

    def test_not_in(self) -> None:
        spec = SplitSpec(mode="field", expression={"train": "month not in ('2026-01')"})
        result = split_frame(self.df, spec)
        self.assertEqual(result.counts["train"], int((self.df["month"] != "2026-01").sum()))

    def test_overlap_rejected(self) -> None:
        spec = SplitSpec(
            mode="field",
            expression={"train": "month in ('2026-01','2026-02')", "val": "f_signal > 0"},
        )
        with self.assertRaises(ConfigError):
            split_frame(self.df, spec)

    def test_uncovered_rows_warn(self) -> None:
        spec = SplitSpec(mode="field", expression={"train": "month == '2026-01'"})
        result = split_frame(self.df, spec)
        self.assertTrue(any("未覆盖" in w for w in result.warnings))

    def test_backtick_column(self) -> None:
        df = self.df.rename(columns={"f_signal": "信号 值"})
        mask = evaluate_expression("`信号 值` > 0", df)
        self.assertEqual(int(mask.sum()), int((df["信号 值"] > 0).sum()))


class ExpressionSafetyTests(unittest.TestCase):
    """★ 安全红线：危险表达式必须被拒，且报错可读。"""

    def setUp(self) -> None:
        self.df = _frame()

    def test_rejects_import_call(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            compile_expression("__import__('os').system('echo hi')")
        self.assertIn("表达式非法", str(ctx.exception))

    def test_rejects_attribute_access(self) -> None:
        for text in ("f_signal.x == 1", "df.__class__ == 1"):
            with self.assertRaises(ConfigError):
                compile_expression(text)

    def test_rejects_assignment(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            compile_expression("f_signal = 1")
        self.assertIn("==", str(ctx.exception))

    def test_rejects_function_call(self) -> None:
        with self.assertRaises(ConfigError):
            compile_expression("f_signal > len(y)")

    def test_rejects_empty_and_semicolon(self) -> None:
        for text in ("", "f_signal > 1; import os"):
            with self.assertRaises(ConfigError):
                compile_expression(text)

    def test_rejects_unknown_column(self) -> None:
        expr = compile_expression("nope > 1")
        with self.assertRaises(ConfigError) as ctx:
            expr.evaluate(self.df)
        self.assertIn("nope", str(ctx.exception))

    def test_rejects_numeric_compare_on_text(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            evaluate_expression("f_text > 1", self.df)
        self.assertIn("非数值", str(ctx.exception))

    def test_rejects_too_long(self) -> None:
        with self.assertRaises(ConfigError):
            compile_expression("f_signal > 0 " + "| f_weak > 0" * 500)

    def test_string_literals_and_null(self) -> None:
        mask = evaluate_expression("f_text == 'A'", self.df)
        self.assertEqual(int(mask.sum()), int((self.df["f_text"] == "A").sum()))
        mask_null = evaluate_expression("f_missing == null", self.df)
        self.assertEqual(int(mask_null.sum()), int(self.df["f_missing"].isna().sum()))

    def test_expression_columns(self) -> None:
        self.assertEqual(
            expression_columns("month in ('2026-01') & f_signal > 0"),
            ("month", "f_signal"),
        )


class PurityAndJsonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _frame()

    def test_input_frame_unchanged(self) -> None:
        before = self.df.copy(deep=True)
        split_frame(self.df, SplitSpec(mode="random", seed=1))
        pd.testing.assert_frame_equal(self.df, before)
        self.assertEqual(list(self.df.columns), list(before.columns))

    def test_result_to_dict_is_json_safe(self) -> None:
        result = split_frame(self.df, SplitSpec(mode="stratified", seed=3), target="y")
        payload = json.loads(json.dumps(result.to_dict(), ensure_ascii=False))
        self.assertEqual(payload["counts"]["train"] + payload["counts"]["val"]
                         + payload["counts"]["test"], len(self.df))
        self.assertTrue(payload["fingerprint"])
        self.assertIn("description", payload)

    def test_subset_returns_new_object(self) -> None:
        result = split_frame(self.df, SplitSpec(mode="random", seed=2))
        part = result.subset(self.df, "train")
        part["y"] = 99
        self.assertFalse((self.df["y"] == 99).any(), "子集必须是副本，不能改到原表")

    def test_empty_frame_raises_for_random(self) -> None:
        empty = self.df.iloc[0:0]
        with self.assertRaises(DataError):
            split_frame(empty, SplitSpec(mode="random"))


class LegacyHelperTests(unittest.TestCase):
    """旧口径 helper 必须可复用（T4 接线用）。"""

    def test_earliest_month_masks(self) -> None:
        df = _frame()
        train_mask, val_mask, earliest = earliest_month_masks(df, "month")
        self.assertEqual(earliest, "2026-01")
        self.assertTrue((df.loc[train_mask, "month"] == "2026-01").all())
        self.assertFalse((df.loc[val_mask, "month"] == "2026-01").any())
        self.assertTrue(train_mask.any() and val_mask.any())

    def test_earliest_month_missing_column(self) -> None:
        with self.assertRaises(DataError):
            earliest_month_masks(_frame(), "nope")

    def test_resolve_split_subset_still_works(self) -> None:
        df = _frame()
        subset = resolve_split_subset(df, {}, SPLIT_STAGE_REFINE, split_col="split")
        self.assertEqual(subset.subsets, ("train", "val"))
        self.assertIn("train", subset.description)
        self.assertIsNone(subset.test_mask)

    def test_resolve_split_subset_missing_subset_raises(self) -> None:
        df = _frame()
        df = df[df["split"] != "val"]
        with self.assertRaises(DataError):
            resolve_split_subset(df, {}, SPLIT_STAGE_REFINE, split_col="split")


if __name__ == "__main__":
    unittest.main()
