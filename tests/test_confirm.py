"""T5 验收 ⑤：确认闸门 —— 不达标必须 ``threshold_not_met`` 且给差距值。

* 不达标 → ``error.code=threshold_not_met``，``data`` 里给 ``ks_gap`` / ``psi_gap``
  与失败项清单（不给「差不多就行」的模糊结论）；
* 达标 → ``final/model.pkl`` + ``final/confirm.json`` + ``manifest.confirmed_run``，
  且 run 状态变 ``confirmed``；
* 无时间列时 PSI 未计算 → **只校验 KS 并在 warnings 明示**（不伪造 PSI）。
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

from fixtures.make_fixtures import (  # noqa: E402
    binary_frame,
    no_time_frame,
    prepare_project,
)
from risklab_mdl.artifacts.manifest import read_manifest  # noqa: E402
from risklab_mdl.core.package import load_package, predict_score  # noqa: E402
from risklab_mdl.tools.model import model_confirm, model_evaluate, model_train  # noqa: E402
from risklab_mdl.tools.split import dataset_split  # noqa: E402

FEATURES = ["f_signal", "f_weak", "f_noise"]
SPLIT = {"mode": "stratified", "ratios": [0.7, 0.15, 0.15], "seed": 42}


class ConfirmGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="t5-confirm-")
        _, cls.project_dir = prepare_project(
            cls._tmp.name, binary_frame(), name="binary_small"
        )
        assert dataset_split(cls.project_dir, SPLIT)["status"] == "ok"
        trained = model_train(
            cls.project_dir,
            {"algorithm": "lgb", "task": "binary", "features": FEATURES},
        )
        assert trained["status"] == "ok", trained
        cls.run_id = trained["data"]["run_id"]
        evaluated = model_evaluate(cls.project_dir, cls.run_id)
        assert evaluated["status"] == "ok", evaluated
        cls.metrics = json.loads(
            (Path(cls.project_dir) / "runs" / cls.run_id / "metrics.json").read_text(
                encoding="utf-8"
            )
        )

        # 无时间列项目（PSI 必须跳过并告警）
        _, cls.no_time_dir = prepare_project(
            cls._tmp.name, no_time_frame(), name="no_time_small"
        )
        assert dataset_split(
            cls.no_time_dir, {"mode": "stratified", "ratios": [0.8, 0.2, 0.0], "seed": 42}
        )["status"] == "ok"
        trained_no_time = model_train(
            cls.no_time_dir,
            {"algorithm": "lgb", "task": "binary", "features": ["f_signal", "f_sparse"]},
        )
        assert trained_no_time["status"] == "ok", trained_no_time
        cls.no_time_run = trained_no_time["data"]["run_id"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # ---------- 不达标 ----------
    def test_threshold_not_met_returns_gaps(self) -> None:
        payload = model_confirm(
            self.project_dir, self.run_id, {"ks_min": 0.99, "psi_max": 0.0000001}
        )
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "threshold_not_met")
        data = payload["data"]
        self.assertIn("failed_checks", data)
        self.assertIn("ks", data["failed_checks"])
        self.assertIsNotNone(data["ks_gap"])
        self.assertGreater(data["ks_gap"], 0.0)
        self.assertEqual(data["thresholds"]["ks_min"], 0.99)
        self.assertTrue(data["checks"])

    def test_threshold_not_met_writes_no_final(self) -> None:
        """不达标时连 final/ 目录都不该被写（用独立项目，避免与通过用例互相污染）。"""
        with tempfile.TemporaryDirectory(prefix="t5-confirm-neg-") as tmp:
            _, project_dir = prepare_project(
                Path(tmp), binary_frame(seed=31), name="binary_small"
            )
            assert dataset_split(project_dir, SPLIT)["status"] == "ok"
            trained = model_train(
                project_dir,
                {"algorithm": "lgb", "task": "binary", "features": FEATURES},
            )
            self.assertEqual(trained["status"], "ok", trained)
            payload = model_confirm(
                project_dir, trained["data"]["run_id"], {"ks_min": 0.99}
            )
            self.assertEqual(payload["status"], "error")
            self.assertEqual(payload["error"]["code"], "threshold_not_met")
            self.assertFalse((Path(project_dir) / "final" / "model.pkl").is_file())
            manifest = read_manifest(Path(project_dir))
            self.assertIsNone(manifest["confirmed_run"])

    # ---------- 达标 ----------
    def test_confirm_pass_writes_final_and_manifest(self) -> None:
        payload = model_confirm(
            self.project_dir, self.run_id, {"ks_min": 0.0, "psi_max": 10.0}
        )
        self.assertEqual(payload["status"], "ok", payload.get("error"))
        final_dir = Path(self.project_dir) / "final"
        self.assertTrue((final_dir / "model.pkl").is_file())
        confirm_path = final_dir / "confirm.json"
        self.assertTrue(confirm_path.is_file())
        document = json.loads(confirm_path.read_text(encoding="utf-8"))
        self.assertEqual(document["run_id"], self.run_id)
        self.assertEqual(document["thresholds"]["ks_min"], 0.0)
        self.assertIsNotNone(document["metrics"]["ks"])
        self.assertEqual(document["schema"], "risklab-mdl.confirm/1")

        manifest = read_manifest(Path(self.project_dir))
        self.assertEqual(manifest["confirmed_run"], self.run_id)
        statuses = {entry["run_id"]: entry["status"] for entry in manifest["runs"]}
        self.assertEqual(statuses[self.run_id], "confirmed")

        # 最终模型仍可加载并打分
        package = load_package(final_dir / "model.pkl")
        scored = predict_score(binary_frame()[FEATURES], package)
        self.assertEqual(list(scored.columns), ["predict_proba", "fico_score"])

    def test_confirm_psi_threshold_blocks(self) -> None:
        """fixture 有 3 个月 ⇒ PSI 必须算出来；把阈值压在实测值之下必须拦住（不再条件跳过）。"""
        psi = self.metrics.get("psi") or {}
        self.assertTrue(psi.get("enabled"), f"fixture 有 3 个月，PSI 必须可用：{psi}")
        max_psi = psi.get("max_psi")
        self.assertIsNotNone(max_psi, psi)
        max_psi = float(max_psi)
        if max_psi > 0:
            psi_max = round(max_psi / 2.0, 6)
            payload = model_confirm(
                self.project_dir, self.run_id, {"ks_min": 0.0, "psi_max": psi_max}
            )
            self.assertEqual(payload["status"], "error", payload.get("error"))
            self.assertEqual(payload["error"]["code"], "threshold_not_met")
            self.assertIn("psi", payload["data"]["failed_checks"])
            self.assertGreater(payload["data"]["psi_gap"], 0.0)
        else:
            # 实测 PSI 恰为 0：阈值 0 时 0 > 0 不成立 ⇒ 闸门应放行（口径：psi_max 是上界）
            payload = model_confirm(
                self.project_dir, self.run_id, {"ks_min": 0.0, "psi_max": 0.0}
            )
            self.assertEqual(payload["status"], "ok", payload.get("error"))
            self.assertNotIn("psi", payload["data"]["failed_checks"])

    # ---------- PSI 缺失 ----------
    def test_confirm_without_time_column_only_checks_ks(self) -> None:
        payload = model_confirm(
            self.no_time_dir, self.no_time_run, {"ks_min": 0.0, "psi_max": 0.0}
        )
        self.assertEqual(payload["status"], "ok", payload.get("error"))
        text = " ".join(payload["warnings"])
        self.assertIn("PSI 未计算", text)
        confirm = json.loads(
            (Path(self.no_time_dir) / "final" / "confirm.json").read_text(encoding="utf-8")
        )
        self.assertIsNone(confirm["metrics"]["psi"])

    # ---------- 错误参数 ----------
    def test_bad_thresholds_rejected(self) -> None:
        payload = model_confirm(self.project_dir, self.run_id, {"ks_min": 2.0})
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "config_invalid")

    def test_confirm_unknown_run(self) -> None:
        payload = model_confirm(self.project_dir, "20200101T000000")
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "run_not_found")


if __name__ == "__main__":
    unittest.main()
