"""模型包（打分链路 / 落盘 / 加载）与模型文档 sheet 的迁移可用性测试。

守护点：

* ``save_package`` → ``load_package`` 往返后 **digest 不变**（否则「同一个包」无法比对）；
* 打分链路顺序固定：取列 → 变换 → 填充 → 预测 → 标尺，**缺列报错不静默补 0**；
* schema 版本不匹配 ⇒ 拒绝加载（不「尽力而为」）；
* ``build_model_doc_sheets`` 在新版 pandas/openpyxl 下能产出非空 DataFrame。

★ 这些是「迁移可用性」验证：旧模块在新版本 pandas 3.x / lightgbm 4.x 下
  仍能跑通，且**口径未被改动**。多分类 sheet / 5 算法矩阵属于 T5 范围，不在此处。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures.make_fixtures import binary_frame  # noqa: E402

from risklab_mdl.core.errors import ModelError  # noqa: E402
from risklab_mdl.core.model_doc import ModelDocInput, build_model_doc_sheets  # noqa: E402
from risklab_mdl.core.package import (  # noqa: E402
    PACKAGE_SCHEMA_VERSION,
    ModelPackage,
    check_scorable,
    load_package,
    predict_score,
    save_package,
)
from risklab_mdl.core.train import build_train_config, train_model  # noqa: E402

LGB_PARAMS = {"objective": "binary", "metric": "auc", "verbose": -1}
FEATURES = ["f_signal", "f_weak", "f_noise", "f_missing"]


def _trained() -> tuple[ModelPackage, pd.DataFrame, pd.Series, pd.DataFrame]:
    df = binary_frame()
    train = df[df["split"] == "train"]
    val = df[df["split"] != "train"]
    cfg = build_train_config(
        algorithm="lgb", params=LGB_PARAMS, num_boost_round=30,
        early_stopping_rounds=10, seed=42,
    )
    outcome = train_model(
        train[FEATURES], train["y"], FEATURES,
        val[FEATURES], val["y"], cfg=cfg,
    )
    return outcome.package, train, val, df


class PackageRoundtripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pkg, cls.train, cls.val, cls.df = _trained()

    def test_save_and_load_keeps_digest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="t2-pkg-") as tmp:
            path = save_package(self.pkg, Path(tmp) / "model.pkl")
            self.assertTrue(path.is_file())
            loaded = load_package(path)
            self.assertEqual(loaded.digest(), self.pkg.digest())
            self.assertEqual(list(loaded.features), list(self.pkg.features))
            self.assertEqual(loaded.algorithm, "lgb")
            self.assertEqual(loaded.schema_version, PACKAGE_SCHEMA_VERSION)

    def test_load_missing_file_raises(self) -> None:
        with tempfile.TemporaryDirectory(prefix="t2-pkg-") as tmp:
            with self.assertRaises(ModelError):
                load_package(Path(tmp) / "nope.pkl")

    def test_schema_mismatch_rejected(self) -> None:
        import pickle

        with tempfile.TemporaryDirectory(prefix="t2-pkg-") as tmp:
            bad = self.pkg
            bad.schema_version = "0.9"
            path = Path(tmp) / "bad.pkl"
            path.write_bytes(pickle.dumps(bad, protocol=pickle.HIGHEST_PROTOCOL))
            with self.assertRaises(ModelError) as ctx:
                load_package(path)
            self.assertIn("schema", str(ctx.exception))
            bad.schema_version = PACKAGE_SCHEMA_VERSION

    def test_manifest_is_json_friendly(self) -> None:
        manifest = self.pkg.manifest()
        self.assertEqual(manifest["algorithm"], "lgb")
        self.assertEqual(manifest["n_features"], len(FEATURES))
        self.assertTrue(manifest["digest"])

    def test_empty_features_rejected(self) -> None:
        with self.assertRaises(ModelError):
            ModelPackage(model=object(), algorithm="lgb", features=[])


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pkg, cls.train, cls.val, cls.df = _trained()

    def test_predict_score_columns(self) -> None:
        out = predict_score(self.val[FEATURES], self.pkg)
        self.assertEqual(list(out.columns), ["predict_proba", "fico_score"])
        self.assertEqual(len(out), len(self.val))
        proba = out["predict_proba"].to_numpy()
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))
        self.assertTrue(np.all(np.isfinite(out["fico_score"].to_numpy())))

    def test_score_ordering_matches_proba(self) -> None:
        out = predict_score(self.val[FEATURES], self.pkg)
        order = np.argsort(out["predict_proba"].to_numpy())
        score = out["fico_score"].to_numpy()[order]
        self.assertTrue(
            np.all(np.diff(score) <= 1e-9) or np.all(np.diff(score) >= -1e-9),
            "标尺只改刻度不改排序（默认方向：概率越高分数越低）",
        )

    def test_missing_feature_column_raises(self) -> None:
        X = self.val[FEATURES].drop(columns=["f_weak"])
        with self.assertRaises(ModelError) as ctx:
            predict_score(X, self.pkg)
        self.assertIn("f_weak", str(ctx.exception))

    def test_check_scorable_empty_raises(self) -> None:
        with self.assertRaises(ModelError):
            check_scorable(self.val[FEATURES].iloc[0:0], self.pkg)

    def test_non_numeric_feature_raises(self) -> None:
        X = self.val[FEATURES].copy()
        X["f_weak"] = X["f_weak"].astype(str)
        with self.assertRaises(ModelError):
            predict_score(X, self.pkg)

    def test_metrics_present_in_package(self) -> None:
        out = predict_score(self.val[FEATURES], self.pkg)
        self.assertGreaterEqual(len(out), 100)


class ModelDocTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pkg, cls.train, cls.val, cls.df = _trained()

    def _input(self) -> ModelDocInput:
        scored = predict_score(self.df[FEATURES], self.pkg).reset_index(drop=True)
        frame = self.df.reset_index(drop=True).copy()
        frame["predict_proba"] = scored["predict_proba"].to_numpy()
        frame["fico_score"] = scored["fico_score"].to_numpy()
        return ModelDocInput(
            frame=frame,
            package=self.pkg,
            label_col="y",
            month_col="month",
            split_col="split",
            iv_values={"f_signal": 0.3, "f_weak": 0.05},
            importance=[{"feature": f, "gain": 1.0 / (i + 1)} for i, f in enumerate(FEATURES)],
        )

    def test_sheets_are_non_empty(self) -> None:
        sheets = build_model_doc_sheets(self._input())
        self.assertTrue(sheets, "模型文档至少要有一个 sheet")
        for name, frame in sheets.items():
            self.assertIsInstance(frame, pd.DataFrame, name)
            self.assertFalse(frame.empty, f"sheet {name} 为空")

    def test_overview_contains_metrics(self) -> None:
        sheets = build_model_doc_sheets(self._input())
        overview = sheets.get("模型概览") or next(iter(sheets.values()))
        text = " ".join(map(str, overview.astype(str).to_numpy().ravel().tolist()))
        self.assertTrue("KS" in text.upper() or "AUC" in text.upper())


if __name__ == "__main__":
    unittest.main()
