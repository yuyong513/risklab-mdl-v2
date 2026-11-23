"""T5 验收 ⑥：调参 —— ``preview`` 与 ``execute`` 的空间必须一致。

口径要点：
* ``mode=preview`` **不训练**（不新增 run），只回显空间与预计试验数；
* ``mode=execute`` 用 hyperopt 落一个 run（``runs/<run_id>/tuning/``）；
* ``tune_on`` 只允许 ``val`` —— ``oot`` / ``train`` / ``cv`` 一律 ``config_invalid``。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from fixtures.make_fixtures import binary_frame, prepare_project  # noqa: E402
from risklab_mdl.core.tuning import (  # noqa: E402
    ObjectiveWeights,
    _objective_value,
    apply_gap_threshold,
)
from risklab_mdl.tools.model import model_catalog, model_tune  # noqa: E402
from risklab_mdl.tools.split import dataset_split  # noqa: E402

FEATURES = ["f_signal", "f_weak", "f_noise"]
SPLIT = {"mode": "stratified", "ratios": [0.7, 0.15, 0.15], "seed": 42}


class TuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="t5-tune-")
        _, cls.project_dir = prepare_project(
            cls._tmp.name, binary_frame(), name="binary_small"
        )
        assert dataset_split(cls.project_dir, SPLIT)["status"] == "ok"

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _config(self, **extra):
        cfg = {"algorithm": "lgb", "task": "binary", "features": FEATURES, "trials": 2}
        cfg.update(extra)
        return cfg

    # ---------- preview ----------
    def test_preview_returns_space_without_training(self) -> None:
        before = self._manifest_runs()
        payload = model_tune(self.project_dir, self._config(), "preview")
        self.assertEqual(payload["status"], "ok", payload.get("error"))
        data = payload["data"]
        self.assertEqual(data["mode"], "preview")
        self.assertEqual(data["tune_on"], "val")
        self.assertEqual(data["expected_trials"], 2)
        self.assertTrue(data["space"])
        self.assertTrue(data["space_table"])
        self.assertEqual(self._manifest_runs(), before, "preview 不允许落 run")

    def test_preview_space_matches_catalog(self) -> None:
        payload = model_tune(self.project_dir, self._config(), "preview")
        catalog = model_catalog()["data"]["catalog"]
        expected = catalog["algorithms"]["lgb"]["by_task"]["binary"]["default_space"]
        self.assertEqual(payload["data"]["space"], expected)
        for name, spec in expected.items():
            preview_spec = payload["data"]["space"][name]
            self.assertEqual(preview_spec["type"], spec["type"])
            if "low" in spec:
                self.assertEqual(preview_spec["low"], spec["low"])
                self.assertEqual(preview_spec["high"], spec["high"])

    # ---------- execute ----------
    def test_execute_writes_run_and_matches_preview_space(self) -> None:
        preview = model_tune(self.project_dir, self._config(), "preview")
        before = self._manifest_runs()
        executed = model_tune(self.project_dir, self._config(), "execute")
        self.assertEqual(executed["status"], "ok", executed.get("error"))
        data = executed["data"]
        self.assertEqual(data["mode"], "execute")
        self.assertEqual(data["space"], preview["data"]["space"], "预览与执行的空间不一致")
        self.assertEqual(data["expected_trials"], preview["data"]["expected_trials"])
        self.assertEqual(self._manifest_runs(), before + 1, "execute 必须落一个 run")
        self.assertEqual(data["n_trials"], 2)

        run_id = data["run_id"]
        run_dir = Path(self.project_dir) / "runs" / run_id
        self.assertTrue((run_dir / "tuning" / "trials.csv").is_file())
        self.assertTrue((run_dir / "tuning" / "space.json").is_file())
        space_doc = json.loads(
            (run_dir / "tuning" / "space.json").read_text(encoding="utf-8")
        )
        self.assertEqual(space_doc["space"], preview["data"]["space"])
        self.assertEqual(space_doc["tune_on"], "val")
        config_doc = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config_doc["tuning"]["mode"], "execute")
        self.assertEqual(config_doc["tuning"]["tune_on"], "val")
        self.assertEqual(config_doc["tuning"]["n_trials"], 2)

    def test_best_params_are_applied_to_final_model(self) -> None:
        executed = model_tune(self.project_dir, self._config(), "execute")
        self.assertEqual(executed["status"], "ok", executed.get("error"))
        params_path = (
            Path(self.project_dir) / "runs" / executed["data"]["run_id"] / "params.json"
        )
        final_params = json.loads(params_path.read_text(encoding="utf-8"))["params"]
        for key, value in executed["data"]["best_params"].items():
            self.assertIn(key, final_params)
            self.assertEqual(final_params[key], value, f"{key} 未按最优参数落盘")

    # ---------- 禁 OOT ----------
    def test_tune_on_oot_rejected(self) -> None:
        payload = model_tune(self.project_dir, self._config(tune_on="oot"), "execute")
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "config_invalid")
        self.assertIn("oot", payload["error"]["message"].lower())

    def test_tune_on_train_rejected(self) -> None:
        payload = model_tune(self.project_dir, self._config(tune_on="train"), "preview")
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "config_invalid")

    def test_trials_bounds(self) -> None:
        payload = model_tune(self.project_dir, self._config(trials=0), "preview")
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "config_invalid")

    def test_unknown_space_param_rejected(self) -> None:
        payload = model_tune(
            self.project_dir,
            self._config(space={"not_a_param": {"type": "int", "low": 1, "high": 2}}),
            "preview",
        )
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "config_invalid")
        self.assertIn("not_a_param", payload["error"]["message"])

    def test_bad_mode_rejected(self) -> None:
        payload = model_tune(self.project_dir, self._config(), "quick")
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "config_invalid")

    # ---------- gap 阈值（承诺必须真生效） ----------
    def test_preview_formula_matches_gap_threshold(self) -> None:
        """预览文案里的阈值数字必须与实现同源（不许再出现「文档 0.05、实现按 0 罚」）。"""
        payload = model_tune(self.project_dir, self._config(), "preview")
        objective = payload["data"]["objective"]
        self.assertEqual(objective["gap_threshold"], 0.05)
        self.assertIn("0.05", objective["formula"])
        self.assertIn("不扣分", objective["formula"])

    def test_gap_below_threshold_is_not_penalized(self) -> None:
        weights = ObjectiveWeights(ks=0.5, auc=0.4, lift=0.0, gap=0.1)
        metrics = {"cv_ks": 0.40, "cv_auc": 0.80, "train_cv_gap": 0.03}
        expected = weights.ks * 0.40 + weights.auc * 0.80
        trimmed = apply_gap_threshold(metrics, 0.05)
        self.assertEqual(trimmed["train_cv_gap_penalized"], 0.0)
        self.assertAlmostEqual(
            _objective_value(trimmed, weights, gap_threshold=0.05), expected, places=9
        )
        # 不先裁剪直接调用也必须按阈值扣分（阈值承诺不能只在某一个入口生效）
        self.assertAlmostEqual(
            _objective_value(metrics, weights, gap_threshold=0.05), expected, places=9
        )

    def test_gap_above_threshold_is_penalized(self) -> None:
        weights = ObjectiveWeights(ks=0.5, auc=0.4, lift=0.0, gap=0.1)
        metrics = {"cv_ks": 0.40, "cv_auc": 0.80, "train_cv_gap": 0.08}
        base = weights.ks * 0.40 + weights.auc * 0.80
        trimmed = apply_gap_threshold(metrics, 0.05)
        self.assertEqual(trimmed["train_cv_gap_penalized"], 0.08)
        self.assertAlmostEqual(
            _objective_value(trimmed, weights, gap_threshold=0.05),
            base - weights.gap * 0.08,
            places=9,
        )
        self.assertAlmostEqual(
            _objective_value(metrics, weights, gap_threshold=0.05),
            base - weights.gap * 0.08,
            places=9,
        )

    def test_negative_gap_never_penalized(self) -> None:
        """val 比 train 还好（gap<0）不算过拟合，不允许倒扣分。"""
        weights = ObjectiveWeights(ks=0.5, auc=0.4, lift=0.0, gap=0.1)
        metrics = {"cv_ks": 0.40, "cv_auc": 0.80, "train_cv_gap": -0.05}
        trimmed = apply_gap_threshold(metrics, 0.05)
        self.assertEqual(trimmed["train_cv_gap_penalized"], 0.0)
        self.assertAlmostEqual(
            _objective_value(trimmed, weights, gap_threshold=0.05),
            weights.ks * 0.40 + weights.auc * 0.80,
            places=9,
        )

    # ---------- helpers ----------
    def _manifest_runs(self) -> int:
        from risklab_mdl.artifacts.manifest import read_manifest

        manifest = read_manifest(Path(self.project_dir))
        return len(manifest.get("runs") or [])


if __name__ == "__main__":
    unittest.main()
