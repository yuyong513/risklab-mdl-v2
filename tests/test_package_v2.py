"""T5 验收 ①(续)：模型包 schema 2.0 与 **schema 1.0 向后兼容**。

* 2.0：``task`` / ``num_class`` / ``classes`` / ``bad_class`` 随包落盘，
  ``predict_score`` 二分类取正类、多分类给逐类概率 + ``bad_class`` 分数；
* 1.0：一期 ``core.train`` 产出的旧包（只有 lgb + 二分类）**仍可 load 并打分**，
  列名与旧口径完全一致（``predict_proba`` / ``fico_score``）——
  升级不能让旧项目训练好的模型失效；
* 版本闸门：不支持的版本（如 ``0.9``）仍拒绝加载。
"""

from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from fixtures.make_fixtures import binary_frame, multiclass_frame  # noqa: E402
from risklab_mdl.core.errors import ModelError  # noqa: E402
from risklab_mdl.core.metrics import ScoreScale  # noqa: E402
from risklab_mdl.core.package import (  # noqa: E402
    PACKAGE_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    ModelPackage,
    load_package,
    predict_proba_matrix,
    predict_score,
    save_package,
)
from risklab_mdl.core.train import build_train_config, train_model  # noqa: E402
from risklab_mdl.models.labels import encode_labels  # noqa: E402
from risklab_mdl.models.train import fit_model  # noqa: E402

LGB_PARAMS = {"objective": "binary", "metric": "auc", "verbose": -1}
FEATURES = ["f_signal", "f_weak", "f_noise", "f_missing"]
MULTI_FEATURES = ["f1", "f2", "f3", "f_noise"]


def _fit_binary() -> ModelPackage:
    frame = binary_frame()
    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] != "train"]
    codes, encoding = encode_labels(train["y"], task="binary")
    X = train[FEATURES]
    X_val = val[FEATURES]
    outcome = fit_model(
        algorithm="dt",
        task="binary",
        features=FEATURES,
        X_train=X,
        y_train=codes.astype(int).to_numpy(),
        X_val=X_val,
        y_val=encode_labels(val["y"], task="binary")[0].astype(int).to_numpy(),
        num_class=2,
        classes=list(encoding.classes),
        bad_class=encoding.bad_class,
        params={"max_depth": 4, "min_samples_leaf": 10},
        seed=42,
    )
    return outcome.package


def _fit_multiclass() -> tuple[ModelPackage, list]:
    frame = multiclass_frame()
    codes, encoding = encode_labels(frame["y"], task="multiclass")
    valid = codes.notna()
    X = frame.loc[valid, MULTI_FEATURES]
    y = codes[valid].astype(int).to_numpy()
    split = int(len(X) * 0.75)
    outcome = fit_model(
        algorithm="dt",
        task="multiclass",
        features=MULTI_FEATURES,
        X_train=X.iloc[:split],
        y_train=y[:split],
        X_val=X.iloc[split:],
        y_val=y[split:],
        num_class=3,
        classes=list(encoding.classes),
        bad_class=encoding.bad_class,
        params={"max_depth": 4, "min_samples_leaf": 10},
        seed=42,
    )
    return outcome.package, list(encoding.classes)


class PackageV2Tests(unittest.TestCase):
    def test_schema_version_is_2(self) -> None:
        self.assertEqual(PACKAGE_SCHEMA_VERSION, "2.0")
        self.assertIn("1.0", SUPPORTED_SCHEMA_VERSIONS)
        self.assertIn("2.0", SUPPORTED_SCHEMA_VERSIONS)

    def test_binary_package_roundtrip(self) -> None:
        package = _fit_binary()
        self.assertEqual(package.task, "binary")
        self.assertEqual(package.num_class, 2)
        self.assertEqual(list(package.classes), [0, 1])
        self.assertEqual(package.bad_index(), 1)
        with tempfile.TemporaryDirectory(prefix="t5-pkg-") as tmp:
            path = save_package(package, Path(tmp) / "model.pkl")
            loaded = load_package(path)
            self.assertEqual(loaded.digest(), package.digest())
            self.assertEqual(loaded.task, "binary")
            self.assertEqual(list(loaded.classes), list(package.classes))
            self.assertEqual(loaded.bad_index(), package.bad_index())
            scored = predict_score(binary_frame()[FEATURES], loaded)
            self.assertEqual(list(scored.columns), ["predict_proba", "fico_score"])
            proba = scored["predict_proba"].to_numpy()
            self.assertTrue(np.all((proba >= 0) & (proba <= 1)))
            self.assertTrue(np.all(np.isfinite(scored["fico_score"].to_numpy())))

    def test_binary_score_uses_bad_index_not_second_column(self) -> None:
        """二分类分数取 bad_index() 那一列，不硬编码第 2 列。

        ``classes`` 顺序被改成 ``[bad, good]`` 时，硬编码 ``matrix[:, 1]``
        会让分数方向反转（KS/AUC 照样算得出来，只是把好客户排到最差）。
        """
        package = _fit_binary()
        frame = binary_frame()[FEATURES]
        matrix, _classes = predict_proba_matrix(frame, package)
        default_scored = predict_score(frame, package)
        self.assertEqual(package.bad_index(), 1)
        np.testing.assert_allclose(
            default_scored["predict_proba"].to_numpy(), matrix[:, 1],
            rtol=1e-12, err_msg="默认 [good, bad] 顺序下应取第 2 列",
        )

        package.classes = ["bad", "good"]
        package.bad_class = "bad"
        self.assertEqual(package.bad_index(), 0)
        flipped = predict_score(frame, package)
        flipped_matrix, _ = predict_proba_matrix(frame, package)
        np.testing.assert_allclose(
            flipped["predict_proba"].to_numpy(), flipped_matrix[:, 0],
            rtol=1e-12,
            err_msg="bad_class 在第 1 列时，predict_proba 必须是该列（不得硬编码第 2 列）",
        )
        # 方向确实被翻了：两列不同 ⇒ 用第 2 列会得到完全不同的排序
        self.assertFalse(
            np.allclose(flipped_matrix[:, 0], flipped_matrix[:, 1]),
            "两列概率相同 ⇒ 该断言无法证明列选择，换 fixture",
        )

    def test_multiclass_package_roundtrip(self) -> None:
        package, classes = _fit_multiclass()
        self.assertEqual(package.task, "multiclass")
        self.assertEqual(package.num_class, 3)
        self.assertEqual(list(package.classes), classes)
        self.assertEqual(package.bad_index(), len(classes) - 1)
        with tempfile.TemporaryDirectory(prefix="t5-pkg-") as tmp:
            path = save_package(package, Path(tmp) / "model.pkl")
            loaded = load_package(path)
            frame = multiclass_frame()[MULTI_FEATURES]
            scored = predict_score(frame, loaded)
            expected = ["predict_proba", "fico_score"] + [f"proba_{c}" for c in classes]
            self.assertEqual(list(scored.columns), expected)
            matrix, matrix_classes = predict_proba_matrix(frame, loaded)
            self.assertEqual(matrix.shape, (len(frame), 3))
            self.assertEqual(matrix_classes, classes)
            np.testing.assert_allclose(
                scored[f"proba_{loaded.bad_class}"].to_numpy(),
                scored["predict_proba"].to_numpy(),
            )
            np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=1e-6)

    def test_digest_changes_with_labels(self) -> None:
        package = _fit_binary()
        baseline = package.digest()
        package.classes = ["good", "bad"]
        package.bad_class = "bad"
        self.assertNotEqual(package.digest(), baseline, "换标签编码必须改变摘要")

    def test_invalid_package_fields_rejected(self) -> None:
        with self.assertRaises(ModelError):
            ModelPackage(
                model=object(), algorithm="dt", features=["a"], task="binary", num_class=3
            )
        with self.assertRaises(ModelError):
            ModelPackage(
                model=object(),
                algorithm="dt",
                features=["a"],
                task="multiclass",
                num_class=3,
                classes=[0, 1, 2],
                bad_class=9,
            )
        with self.assertRaises(ModelError):
            ModelPackage(
                model=object(), algorithm="dt", features=["a"], task="regression"
            )

    def test_unknown_schema_rejected(self) -> None:
        package = _fit_binary()
        package.schema_version = "0.9"
        with tempfile.TemporaryDirectory(prefix="t5-pkg-") as tmp:
            path = Path(tmp) / "bad.pkl"
            path.write_bytes(pickle.dumps(package, protocol=pickle.HIGHEST_PROTOCOL))
            with self.assertRaises(ModelError):
                load_package(path)


class PackageV1CompatibilityTests(unittest.TestCase):
    """schema 1.0 旧包必须仍可加载、仍可打分。"""

    def _legacy_package(self) -> ModelPackage:
        frame = binary_frame()
        train = frame[frame["split"] == "train"]
        val = frame[frame["split"] != "train"]
        cfg = build_train_config(
            algorithm="lgb",
            params=LGB_PARAMS,
            num_boost_round=30,
            early_stopping_rounds=10,
            seed=42,
        )
        outcome = train_model(
            train[FEATURES], train["y"], FEATURES, val[FEATURES], val["y"], cfg=cfg
        )
        package = outcome.package
        # 模拟一期落盘的旧包：schema 1.0，没有 task/classes/bad_class 语义
        package.schema_version = "1.0"
        package.task = "binary"
        package.classes = []
        package.bad_class = None
        return package

    def test_v1_package_loads_and_scores(self) -> None:
        package = self._legacy_package()
        with tempfile.TemporaryDirectory(prefix="t5-pkgv1-") as tmp:
            path = save_package(package, Path(tmp) / "legacy.pkl")
            loaded = load_package(path)
            self.assertEqual(loaded.schema_version, "1.0")
            self.assertFalse(loaded.is_v2)
            scored = predict_score(binary_frame()[FEATURES], loaded)
            self.assertEqual(list(scored.columns), ["predict_proba", "fico_score"])
            self.assertEqual(len(scored), len(binary_frame()))
            matrix, classes = predict_proba_matrix(binary_frame()[FEATURES], loaded)
            self.assertEqual(matrix.shape[1], 2)
            self.assertEqual(classes, [0, 1])
            np.testing.assert_allclose(matrix[:, 1], scored["predict_proba"].to_numpy())

    def test_v1_and_v2_agree_on_same_model(self) -> None:
        """同一个 lgb 模型：1.0 兼容口径与 2.0 口径的概率必须逐行一致。"""
        legacy = self._legacy_package()
        v2 = self._legacy_package()
        v2.schema_version = PACKAGE_SCHEMA_VERSION
        frame = binary_frame()[FEATURES]
        old_scored = predict_score(frame, legacy)
        new_scored = predict_score(frame, v2)
        np.testing.assert_allclose(
            old_scored["predict_proba"].to_numpy(),
            new_scored["predict_proba"].to_numpy(),
            rtol=1e-12,
            err_msg="1.0 兼容路径与 2.0 路径的概率不一致",
        )
        np.testing.assert_allclose(
            old_scored["fico_score"].to_numpy(),
            new_scored["fico_score"].to_numpy(),
            rtol=1e-12,
        )

    def test_legacy_pickle_without_new_fields_loads(self) -> None:
        """真实旧包：pickle 里根本没有 2.0 新增属性（只剩 1.0 字段）。"""
        package = self._legacy_package()
        for name in ("task", "num_class", "classes", "bad_class",
                     "algorithm_family", "preprocess_spec", "training"):
            if hasattr(package, name):
                delattr(package, name)
        with tempfile.TemporaryDirectory(prefix="t5-pkgv1-") as tmp:
            path = Path(tmp) / "legacy_no_fields.pkl"
            path.write_bytes(pickle.dumps(package, protocol=pickle.HIGHEST_PROTOCOL))
            loaded = load_package(path)
            self.assertEqual(loaded.task, "binary")
            self.assertEqual(int(loaded.num_class), 2)
            self.assertEqual(list(loaded.classes), [])
            scored = predict_score(binary_frame()[FEATURES], loaded)
            self.assertEqual(list(scored.columns), ["predict_proba", "fico_score"])

    def test_score_scale_still_applied(self) -> None:
        package = self._legacy_package()
        package.score_scale = ScoreScale(base_point=700.0, pdo=50.0)
        scored = predict_score(binary_frame()[FEATURES], package)
        self.assertLessEqual(scored["fico_score"].max(), 1000.0)
        self.assertGreaterEqual(scored["fico_score"].min(), 300.0)
        self.assertFalse(pd.isna(scored["fico_score"]).any())


if __name__ == "__main__":
    unittest.main()
