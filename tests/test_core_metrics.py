"""指标口径测试：AUC / KS / Gini / LIFT / ScoreScale / 多分类 OVR+macro。

守护点：

* KS 必须**按分数整组推进**（并列值不能被拆开），否则同分客户被切开 ⇒ KS 偏大；
* LIFT 的方向判定统一走 ``score_direction``（不各算各的）；
* 多分类 OVR 的 macro = 有效类算术平均，skipped 类**显式记录**、不参与汇总；
* 多分类打分矩阵形态错误必须报错（不能把一维数组当成某一类的概率）。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures.make_fixtures import multiclass_frame  # noqa: E402

from risklab_mdl.core.errors import DataError  # noqa: E402
from risklab_mdl.core.metrics import (  # noqa: E402
    ScoreScale,
    auc_significance,
    calc_auc,
    calc_auc_ovr,
    calc_gini,
    calc_ks,
    calc_lift,
    calc_multiclass_metrics,
    score_direction,
)


def _binary(n: int = 300, seed: int = 3):
    rng = np.random.default_rng(seed)
    y = np.array([1, 0] * (n // 2))
    rng.shuffle(y)
    score = rng.normal(size=n) + 1.4 * y
    return y, score


class BinaryMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.y, self.score = _binary()

    def test_auc_perfect_separation(self) -> None:
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(calc_auc(y, s), 1.0, places=12)

    def test_auc_reversed_is_zero(self) -> None:
        y = np.array([0, 0, 1, 1])
        s = np.array([0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(calc_auc(y, s), 0.0, places=12)

    def test_auc_matches_sklearn(self) -> None:
        from sklearn.metrics import roc_auc_score

        self.assertAlmostEqual(
            calc_auc(self.y, self.score),
            roc_auc_score(self.y, self.score),
            places=12,
        )

    def test_auc_handles_ties(self) -> None:
        """并列值用平均秩 ⇒ 与 sklearn 一致（相差 < 1e-12）。"""
        from sklearn.metrics import roc_auc_score

        y = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        s = np.array([0.5] * 8)
        self.assertAlmostEqual(calc_auc(y, s), roc_auc_score(y, s), places=12)
        self.assertAlmostEqual(calc_auc(y, s), 0.5, places=12)

    def test_auc_respects_nan_scores(self) -> None:
        s = self.score.copy()
        s[:10] = np.nan
        auc = calc_auc(self.y, s)
        self.assertAlmostEqual(auc, calc_auc(self.y[10:], self.score[10:]), places=12)

    def test_gini_identity(self) -> None:
        auc = calc_auc(self.y, self.score)
        self.assertAlmostEqual(calc_gini(self.y, self.score), 2 * auc - 1, places=12)
        self.assertAlmostEqual(
            calc_gini(self.y, -self.score), -calc_gini(self.y, self.score), places=12
        )

    def test_ks_perfect_separation(self) -> None:
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(calc_ks(y, s), 1.0, places=12)

    def test_ks_is_absolute_under_reversal(self) -> None:
        self.assertAlmostEqual(
            calc_ks(self.y, self.score), calc_ks(self.y, -self.score), places=12
        )

    def test_ks_groups_ties(self) -> None:
        """★ 并列分数必须整组推进：同分客户拆开会让 KS 虚高。"""
        y = np.array([0, 1, 0, 1] * 25)
        s = np.concatenate([np.zeros(50), np.ones(50)])
        # 前 50 个全 0 分：坏样本 50%，后 50 全 1 分：坏样本 50% ⇒ KS 应为 0
        self.assertAlmostEqual(calc_ks(y, s), 0.0, places=12)

    def test_ks_bounds(self) -> None:
        ks = calc_ks(self.y, self.score)
        self.assertGreaterEqual(ks, 0.0)
        self.assertLessEqual(ks, 1.0)

    def test_score_direction(self) -> None:
        self.assertEqual(score_direction(self.y, self.score), "higher_score_higher_risk")
        self.assertEqual(score_direction(self.y, -self.score), "higher_score_lower_risk")

    def test_auc_significance_small_sample_is_small(self) -> None:
        z = auc_significance(0.4833, 30, 90)
        self.assertLess(z, 1.0, "小样本 AUC=0.4833 的偏离不显著，不应告警")

    def test_auc_significance_perfect_is_extreme(self) -> None:
        self.assertEqual(auc_significance(1.0, 30, 90), float("inf"))

    def test_lift_above_one_for_risky_tail(self) -> None:
        result = calc_lift(self.y, self.score, pct=0.2, min_seg_size=10)
        self.assertEqual(result.direction, "higher_score_higher_risk")
        self.assertGreater(result.lift, 1.5, f"尾部 20% 的 LIFT 应明显 >1：{result.lift}")
        self.assertTrue(result.is_reliable)
        self.assertEqual(result.seg_size, int(round(0.2 * len(self.y))))

    def test_lift_best_tail_below_one(self) -> None:
        result = calc_lift(self.y, self.score, pct=0.2, worst=False, min_seg_size=10)
        self.assertLess(result.lift, 1.0)

    def test_lift_unreliable_small_segment(self) -> None:
        result = calc_lift(self.y, self.score, pct=0.01, min_seg_size=30)
        self.assertFalse(result.is_reliable, "段内样本不足必须标记不可靠，不得静默给结论")

    def test_lift_rejects_bad_pct(self) -> None:
        with self.assertRaises(DataError):
            calc_lift(self.y, self.score, pct=0.0)
        with self.assertRaises(DataError):
            calc_lift(self.y, self.score, pct=1.5)

    def test_lift_weight_length_mismatch(self) -> None:
        with self.assertRaises(DataError):
            calc_lift(self.y, self.score, weight=np.ones(5))

    def test_single_class_raises(self) -> None:
        with self.assertRaises(DataError):
            calc_auc(np.zeros(10), np.arange(10.0))

    def test_nan_label_raises(self) -> None:
        y = self.y.copy().astype(float)
        y[0] = np.nan
        with self.assertRaises(DataError):
            calc_auc(y, self.score)

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(DataError):
            calc_ks(self.y, self.score[:-1])


class ScoreScaleTests(unittest.TestCase):
    def test_prob_score_roundtrip(self) -> None:
        scale = ScoreScale()
        prob = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
        score = scale.prob_to_score(prob)
        back = scale.score_to_prob(score)
        self.assertTrue(np.allclose(prob, back, atol=1e-9), f"{prob} -> {back}")
        self.assertTrue(np.all(np.diff(score) < 0), "概率越高分数应越低（默认方向）")

    def test_score_clipped_to_range(self) -> None:
        scale = ScoreScale()
        score = scale.prob_to_score([1e-12, 1 - 1e-12])
        self.assertGreaterEqual(float(score.min()), scale.min_score)
        self.assertLessEqual(float(score.max()), scale.max_score)

    def test_higher_score_higher_risk_mirrors(self) -> None:
        normal = ScoreScale()
        mirrored = ScoreScale(direction="higher_score_higher_risk")
        prob = np.array([0.1, 0.5, 0.9])
        self.assertTrue(
            np.allclose(
                normal.prob_to_score(prob),
                (normal.min_score + normal.max_score) - mirrored.prob_to_score(prob),
                atol=1e-9,
            )
        )

    def test_spec_roundtrip(self) -> None:
        scale = ScoreScale(base_point=650.0, pdo=50.0)
        rebuilt = ScoreScale.from_spec(scale.to_spec())
        self.assertEqual(rebuilt.to_spec(), scale.to_spec())

    def test_from_user_input_text(self) -> None:
        scale = ScoreScale.from_user_input("620,45,320,980")
        self.assertEqual((scale.base_point, scale.pdo), (620.0, 45.0))
        self.assertEqual((scale.min_score, scale.max_score), (320.0, 980.0))

    def test_from_user_input_partial_keeps_base(self) -> None:
        base = ScoreScale(base_point=600.0, pdo=40.0)
        scale = ScoreScale.from_user_input({"base_point": 700.0}, base=base)
        self.assertEqual(scale.base_point, 700.0)
        self.assertEqual(scale.pdo, base.pdo, "未填字段必须沿用 base，不得悄悄重置")

    def test_invalid_params_rejected(self) -> None:
        with self.assertRaises(DataError):
            ScoreScale(pdo=0)
        with self.assertRaises(DataError):
            ScoreScale(min_score=1000, max_score=300)


class MultiClassMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = multiclass_frame()
        self.y = self.df["y"].to_numpy()
        # 用一个「知道正确答案」的打分：目标类概率最高
        rng = np.random.default_rng(21)
        logits = np.eye(3)[self.y] * 2.2 + rng.normal(size=(len(self.y), 3))
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        self.proba = exp / exp.sum(axis=1, keepdims=True)

    def test_per_class_metrics_finite(self) -> None:
        result = calc_multiclass_metrics(self.y, self.proba)
        self.assertEqual(len(result.per_class), 3)
        for cm in result.per_class:
            self.assertEqual(cm.status, "ok")
            for name in ("auc", "ks", "gini", "lift"):
                value = getattr(cm, name)
                self.assertTrue(np.isfinite(value), f"{cm.label}.{name}={value}")
            self.assertGreater(cm.auc, 0.5)
            self.assertGreaterEqual(cm.ks, 0.0)
            self.assertLessEqual(cm.ks, 1.0)

    def test_macro_is_mean_of_valid_classes(self) -> None:
        result = calc_multiclass_metrics(self.y, self.proba)
        manual = float(np.mean([c.auc for c in result.per_class]))
        self.assertAlmostEqual(result.macro["auc"], round(manual, 8), places=8)

    def test_weighted_is_support_weighted(self) -> None:
        result = calc_multiclass_metrics(self.y, self.proba)
        total = sum(c.support for c in result.per_class)
        manual = sum(c.auc * c.support for c in result.per_class) / total
        self.assertAlmostEqual(result.weighted["auc"], round(manual, 8), places=8)

    def test_perfect_predictor_gives_auc_one(self) -> None:
        proba = np.eye(3)[self.y]
        result = calc_multiclass_metrics(self.y, proba, labels=[0, 1, 2])
        for cm in result.per_class:
            self.assertAlmostEqual(cm.auc, 1.0, places=10)

    def test_mapping_input_equivalent_to_matrix(self) -> None:
        mapping = {label: self.proba[:, i] for i, label in enumerate([0, 1, 2])}
        a = calc_multiclass_metrics(self.y, self.proba, labels=[0, 1, 2])
        b = calc_multiclass_metrics(self.y, mapping, labels=[0, 1, 2])
        self.assertEqual(a.to_dict()["per_class"], b.to_dict()["per_class"])

    def test_labels_order_controls_columns(self) -> None:
        swapped = self.proba[:, [2, 1, 0]]
        result = calc_multiclass_metrics(self.y, swapped, labels=[2, 1, 0])
        for cm in result.per_class:
            self.assertGreater(cm.auc, 0.5, f"{cm.label} 的列序映射错了")

    def test_skipped_class_is_recorded(self) -> None:
        """某类在某子集没有样本 ⇒ 该类的 OVR 无定义，必须 skipped 且有 note。"""
        mask = self.y != 2
        y = self.y[mask]
        proba = self.proba[mask]
        result = calc_multiclass_metrics(y, proba, labels=[0, 1, 2])
        class2 = [c for c in result.per_class if c.label == 2][0]
        self.assertEqual(class2.status, "skipped")
        self.assertIn("无正例", class2.reason or "")
        self.assertEqual(result.n_valid_classes, 2)
        self.assertTrue(any("跳过" in note for note in result.notes))
        # 被跳过的类不得污染 macro
        manual = float(np.mean([c.auc for c in result.per_class if c.status == "ok"]))
        self.assertAlmostEqual(result.macro["auc"], round(manual, 8), places=8)

    def test_small_class_marked_unreliable(self) -> None:
        rng = np.random.default_rng(9)
        y = np.array([0] * 50 + [1] * 50 + [2] * 5)
        proba = rng.dirichlet([1, 1, 1], size=len(y))
        result = calc_multiclass_metrics(y, proba, labels=[0, 1, 2], min_class_support=30)
        class2 = [c for c in result.per_class if c.label == 2][0]
        self.assertEqual(class2.status, "ok")
        self.assertFalse(class2.is_reliable)
        self.assertTrue(any("不可靠" in note for note in result.notes))

    def test_auc_ovr_convenience(self) -> None:
        payload = calc_auc_ovr(self.y, self.proba, labels=[0, 1, 2])
        self.assertEqual(set(payload["per_class"]), {0, 1, 2})
        self.assertAlmostEqual(
            payload["macro"],
            float(np.mean(list(payload["per_class"].values()))),
            places=6,
        )

    def test_rejects_1d_score(self) -> None:
        with self.assertRaises(DataError):
            calc_multiclass_metrics(self.y, self.proba[:, 0])

    def test_rejects_wrong_shape(self) -> None:
        with self.assertRaises(DataError):
            calc_multiclass_metrics(self.y, self.proba[:-5])

    def test_rejects_nan_score(self) -> None:
        proba = self.proba.copy()
        proba[0, 0] = np.nan
        with self.assertRaises(DataError):
            calc_multiclass_metrics(self.y, proba)

    def test_rejects_single_class_labels(self) -> None:
        with self.assertRaises(DataError):
            calc_multiclass_metrics(np.zeros(10), np.ones((10, 1)), labels=[0])

    def test_rejects_nan_label(self) -> None:
        y = self.y.astype(float).copy()
        y[0] = np.nan
        with self.assertRaises(DataError):
            calc_multiclass_metrics(y, self.proba)

    def test_to_dict_is_json_safe(self) -> None:
        result = calc_multiclass_metrics(self.y, self.proba)
        text = json.dumps(result.to_dict(), ensure_ascii=False)
        payload = json.loads(text)
        self.assertNotIn("NaN", text)
        self.assertEqual(payload["task"], "multiclass")
        self.assertEqual(payload["n_classes"], 3)

    def test_frame_has_one_row_per_class(self) -> None:
        frame = calc_multiclass_metrics(self.y, self.proba).frame()
        self.assertIsInstance(frame, pd.DataFrame)
        self.assertEqual(len(frame), 3)
        for col in ("label", "auc", "ks", "gini", "lift", "status"):
            self.assertIn(col, frame.columns)

    def test_string_labels(self) -> None:
        y = np.array(["good", "bad", "mid"] * 60)
        rng = np.random.default_rng(4)
        proba = rng.dirichlet([1, 1, 1], size=len(y))
        result = calc_multiclass_metrics(y, proba)
        self.assertEqual(sorted(result.labels), ["bad", "good", "mid"])
        self.assertEqual(len(result.per_class), 3)


if __name__ == "__main__":
    unittest.main()
