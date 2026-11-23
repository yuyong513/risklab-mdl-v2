"""T5 验收 ③：错误路径必须可见（不静默降级）。

覆盖：
* 未知算法 / 未知任务 → ``task_algorithm_mismatch``；
* 依赖缺失（monkeypatch ``require_xgboost`` / ``require_lightgbm`` 抛错）
  → ``dependency_missing``，且 hint 里带 pip 命令；
* 未处理文本 / 缺失 → 显式 ``warnings``（自动排除非数值列、LR 系用 train 中位数填充）
  或显式 ``config_invalid``（用户点名要一个非数值列）；
* 目标字段未确认 → ``needs_user_input``；
* 没有 run 就评价/确认 → ``run_not_found``。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from fixtures.make_fixtures import (  # noqa: E402
    MISSING_TEXT_CSV,
    binary_frame,
    missing_text_frame,
    prepare_project,
)
from risklab_mdl.core.errors import ModelError  # noqa: E402
from risklab_mdl.tools.dataset import project_init  # noqa: E402
from risklab_mdl.tools.model import model_confirm, model_evaluate, model_train  # noqa: E402

FEATURES = ["f_signal", "f_weak", "f_noise"]


def _missing_dependency(name: str) -> ModelError:
    return ModelError(
        f"缺少可选依赖 {name}",
        detail={"dependency": name, "install": f"pip install {name}"},
    )


class ModelErrorPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="t5-errors-")
        _, cls.project_dir = prepare_project(
            cls._tmp.name, binary_frame(), name="binary_small"
        )
        _, cls.text_project = prepare_project(
            cls._tmp.name,
            missing_text_frame(),
            name=Path(MISSING_TEXT_CSV).stem,
        )
        # 专供「还没有 run」的断言：别的用例会往主项目里写 run，不能共用
        _, cls.empty_project = prepare_project(
            cls._tmp.name, binary_frame(seed=99), name="empty_binary"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # ---------- 算法 / 任务 ----------
    def test_unknown_algorithm(self) -> None:
        payload = model_train(
            self.project_dir, {"algorithm": "svm", "task": "binary", "features": FEATURES}
        )
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "task_algorithm_mismatch")
        self.assertIn("svm", payload["error"]["message"])

    def test_unknown_task(self) -> None:
        payload = model_train(
            self.project_dir,
            {"algorithm": "lgb", "task": "regression", "features": FEATURES},
        )
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "task_algorithm_mismatch")

    def test_missing_algorithm(self) -> None:
        payload = model_train(self.project_dir, {"task": "binary", "features": FEATURES})
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "config_invalid")

    # ---------- 依赖缺失 ----------
    def test_xgboost_missing(self) -> None:
        with mock.patch(
            "risklab_mdl.models.adapters.xgb_adapter.require_xgboost",
            side_effect=_missing_dependency("xgboost"),
        ):
            payload = model_train(
                self.project_dir,
                {"algorithm": "xgb", "task": "binary", "features": FEATURES},
            )
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "dependency_missing")
        self.assertIn("pip install xgboost", payload["error"]["hint"])

    def test_lightgbm_missing(self) -> None:
        with mock.patch(
            "risklab_mdl.models.adapters.lgb_adapter.require_lightgbm",
            side_effect=_missing_dependency("lightgbm"),
        ):
            payload = model_train(
                self.project_dir,
                {"algorithm": "lgb", "task": "binary", "features": FEATURES},
            )
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "dependency_missing")
        self.assertIn("pip install lightgbm", payload["error"]["hint"])

    # ---------- 未处理文本 / 缺失 ----------
    def test_auto_features_exclude_text_with_warnings(self) -> None:
        """自动选特征：非数值列被排除，但必须**告警**（不静默丢）。"""
        payload = model_train(
            self.project_dir, {"algorithm": "dt", "task": "binary"}
        )
        self.assertEqual(payload["status"], "ok", payload.get("error"))
        warnings = " ".join(payload["warnings"])
        self.assertIn("非数值列", warnings)
        self.assertIn("f_text", warnings)

    def test_lr_fills_missing_with_train_median_and_warns(self) -> None:
        payload = model_train(
            self.project_dir,
            {"algorithm": "lr", "task": "binary", "features": FEATURES + ["f_missing"]},
        )
        self.assertEqual(payload["status"], "ok", payload.get("error"))
        warnings = " ".join(payload["warnings"])
        self.assertIn("train 中位数", warnings)
        package_path = (
            Path(self.project_dir) / "runs" / payload["data"]["run_id"] / "model.pkl"
        )
        self.assertTrue(package_path.is_file())

    def test_explicit_text_feature_rejected(self) -> None:
        payload = model_train(
            self.project_dir,
            {"algorithm": "lgb", "task": "binary", "features": FEATURES + ["f_text"]},
        )
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "config_invalid")
        self.assertIn("f_text", payload["error"]["message"])

    def test_unknown_feature_rejected(self) -> None:
        payload = model_train(
            self.project_dir,
            {"algorithm": "dt", "task": "binary", "features": ["no_such_column"]},
        )
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "config_invalid")

    # ---------- 未知参数名（不能变成 analysis_failed） ----------
    def test_unknown_param_key_is_config_invalid(self) -> None:
        payload = model_train(
            self.project_dir,
            {
                "algorithm": "dt",
                "task": "binary",
                "features": FEATURES,
                "params": {"max_depht": 4},
            },
        )
        self.assertEqual(payload["status"], "error", payload)
        self.assertEqual(payload["error"]["code"], "config_invalid")
        message = payload["error"]["message"]
        self.assertIn("max_depht", message)
        self.assertIn("max_depth", message, "报错要给出可用参数名（便于改正拼写）")

    def test_objective_param_still_dropped_with_warning(self) -> None:
        """objective/metric 等由 task 决定：允许传入，但必须丢弃并告警（不算未知键）。"""
        payload = model_train(
            self.project_dir,
            {
                "algorithm": "dt",
                "task": "binary",
                "features": FEATURES,
                "params": {"objective": "multiclass", "metric": "mlogloss"},
            },
        )
        self.assertEqual(payload["status"], "ok", payload.get("error"))
        text = " ".join(payload["warnings"])
        self.assertIn("objective", text)

    def test_raw_text_project_warns_instead_of_silent(self) -> None:
        """有文本/缺失、无预处理产物：要么报错要么 warnings —— 不得静默。"""
        payload = model_train(
            self.text_project, {"algorithm": "lgb", "task": "binary"}
        )
        self.assertIn(payload["status"], ("ok", "error"))
        if payload["status"] == "ok":
            text = " ".join(payload["warnings"])
            self.assertTrue(
                "非数值列" in text or "缺失" in text,
                f"原始数据入口没有任何可见告警：{payload['warnings']}",
            )

    # ---------- 目标字段 / run ----------
    def test_missing_target_asks_user(self) -> None:
        import pandas as pd

        with tempfile.TemporaryDirectory(prefix="t5-target-") as tmp:
            csv = Path(tmp) / "no_target.csv"
            pd.DataFrame(binary_frame()).to_csv(csv, index=False, encoding="utf-8")
            fresh = project_init(str(csv))
            project_dir = fresh["data"]["project_dir"]
            payload = model_train(project_dir, {"algorithm": "lgb", "task": "binary"})
        self.assertEqual(payload["status"], "needs_user_input")
        self.assertIn("question", payload["data"])

    def test_evaluate_without_run(self) -> None:
        payload = model_evaluate(self.empty_project)
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "run_not_found")

    def test_confirm_without_run(self) -> None:
        payload = model_confirm(self.empty_project)
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "run_not_found")

    def test_evaluate_unknown_run(self) -> None:
        payload = model_evaluate(self.empty_project, "20200101T000000")
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "run_not_found")


if __name__ == "__main__":
    unittest.main()
