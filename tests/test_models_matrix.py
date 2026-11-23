"""T5 验收 ①：5 算法 × 2 任务 = **10 格矩阵**全链路。

每格都要走完：
``model_train``（落 runs/<run_id>/）→ ``load_package`` + ``predict_score``
→ ``model_evaluate``（写 metrics.json + evaluation/）。

断言口径：
* 指标数值**有限**（不给 NaN 冒充满分）；
* 二分类 ``predict_score`` 的列名与 schema 1.0 一致（``predict_proba`` 取正类）；
* 多分类 ``predict_score`` 输出**逐类概率** + 按 ``bad_class`` 的分数，
  且 ``predict_proba`` 恰等于 ``bad_class`` 那一列（口径自洽）；
* 多分类 ``metrics.json`` 含**逐类 OVR 指标**与 **macro**。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from fixtures.make_fixtures import (  # noqa: E402
    binary_frame,
    multiclass_frame,
    prepare_project,
)
from risklab_mdl.core.package import load_package, predict_score  # noqa: E402
from risklab_mdl.tools.model import model_evaluate, model_train  # noqa: E402
from risklab_mdl.tools.split import dataset_split  # noqa: E402

ALGORITHMS = ("dt", "lr", "gbdt", "xgb", "lgb")
TASKS = ("binary", "multiclass")
BINARY_FEATURES = ["f_signal", "f_weak", "f_noise", "f_missing", "f_dup"]
MULTI_FEATURES = ["f1", "f2", "f3", "f_noise"]
SPLIT = {"mode": "stratified", "ratios": [0.7, 0.15, 0.15], "seed": 42}


class ModelMatrixTests(unittest.TestCase):
    """10 格矩阵。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="t5-matrix-")
        _, cls.binary_dir = prepare_project(
            cls._tmp.name, binary_frame(), name="binary_small"
        )
        split = dataset_split(cls.binary_dir, SPLIT)
        assert split["status"] == "ok", split
        _, cls.multi_dir = prepare_project(
            cls._tmp.name, multiclass_frame(), name="multiclass_small", task="multiclass"
        )
        split = dataset_split(cls.multi_dir, SPLIT)
        assert split["status"] == "ok", split

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # ---------- 公共 ----------
    def _project(self, task: str) -> str:
        return self.binary_dir if task == "binary" else self.multi_dir

    def _features(self, task: str):
        return BINARY_FEATURES if task == "binary" else MULTI_FEATURES

    def _frame(self, task: str):
        frame = binary_frame() if task == "binary" else multiclass_frame()
        return frame[self._features(task)]

    def test_matrix_has_ten_cells(self) -> None:
        from risklab_mdl.tools.model import model_catalog

        payload = model_catalog()
        self.assertEqual(payload["status"], "ok")
        catalog = payload["data"]["catalog"]
        self.assertEqual(catalog["matrix_size"], 10)
        self.assertEqual(len(catalog["matrix"]), 10)
        for algorithm in ALGORITHMS:
            entry = catalog["algorithms"][algorithm]
            self.assertEqual(entry["tasks"], list(TASKS))
            for task in TASKS:
                self.assertIn(task, entry["by_task"])
                self.assertTrue(entry["by_task"][task]["default_space"])

    def test_all_cells_train_evaluate(self) -> None:
        for task in TASKS:
            for algorithm in ALGORITHMS:
                with self.subTest(algorithm=algorithm, task=task):
                    self._run_cell(algorithm, task)

    # ---------- 单格 ----------
    def _run_cell(self, algorithm: str, task: str) -> None:
        project_dir = self._project(task)
        train = model_train(
            project_dir,
            {"algorithm": algorithm, "task": task, "features": self._features(task)},
        )
        self.assertEqual(train["status"], "ok", f"{algorithm}/{task}: {train}")
        data = train["data"]
        run_id = data["run_id"]
        self.assertEqual(data["algorithm"], algorithm)
        self.assertEqual(data["task"], task)

        # ---- 指标有限 ----
        for subset in ("train", "val"):
            block = data["metrics"][subset]
            if block.get("status") != "ok":
                self.fail(f"{algorithm}/{task}/{subset} 指标未计算：{block}")
            for field in ("ks", "auc", "gini", "lift"):
                value = block[field]
                self.assertIsNotNone(value, f"{algorithm}/{task}/{subset}.{field} 为 None")
                self.assertTrue(np.isfinite(float(value)))

        # ---- 产物齐全 ----
        run_dir = Path(project_dir) / "runs" / run_id
        for name in ("config.json", "params.json", "metrics.json", "model.pkl", "importance.csv"):
            self.assertTrue((run_dir / name).is_file(), f"{algorithm}/{task} 缺 {name}")

        # ---- 模型包可加载、可打分 ----
        package = load_package(run_dir / "model.pkl")
        self.assertEqual(package.task, task)
        self.assertEqual(package.num_class, 2 if task == "binary" else 3)
        scored = predict_score(self._frame(task), package)
        if task == "binary":
            self.assertEqual(list(scored.columns), ["predict_proba", "fico_score"])
            proba = scored["predict_proba"].to_numpy()
            self.assertTrue(np.all((proba >= 0) & (proba <= 1)))
        else:
            classes = list(package.classes)
            expected = ["predict_proba", "fico_score"] + [f"proba_{c}" for c in classes]
            self.assertEqual(list(scored.columns), expected)
            bad_column = f"proba_{package.bad_class}"
            self.assertIn(bad_column, scored.columns)
            np.testing.assert_allclose(
                scored["predict_proba"].to_numpy(),
                scored[bad_column].to_numpy(),
                rtol=1e-9,
                err_msg="predict_proba 必须等于 bad_class 那一列的概率",
            )
            per_class = scored[[f"proba_{c}" for c in classes]].sum(axis=1).to_numpy()
            np.testing.assert_allclose(per_class, 1.0, rtol=1e-6, atol=1e-6)
        self.assertTrue(np.all(np.isfinite(scored["fico_score"].to_numpy())))

        # ---- 评价（写 metrics.json + evaluation/） ----
        evaluated = model_evaluate(project_dir, run_id)
        self.assertEqual(evaluated["status"], "ok", f"{algorithm}/{task}: {evaluated}")
        self.assertTrue((run_dir / "evaluation").is_dir())
        self.assertTrue((run_dir / "evaluation" / "metrics_by_subset.csv").is_file())

        doc = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        if task == "multiclass":
            for subset in ("train", "val"):
                block = doc[subset]
                self.assertEqual(block["task"], "multiclass")
                self.assertEqual(len(block["per_class"]), 3, block)
                self.assertIn("macro", block)
                self.assertIsNotNone(block["macro"]["auc"])
                self.assertEqual(
                    {str(item["label"]) for item in block["per_class"]},
                    {str(c) for c in package.classes},
                )
            self.assertIn("multi", doc)
            self.assertIn("per_class", doc["multi"]["train"])
            self.assertIn("macro", doc["multi"]["train"])
            # 坏类分数口径：bad_class 是排序后最后一类（fixture 的 y=2）
            self.assertEqual(str(doc["bad_class"]), str(package.classes[-1]))
        else:
            for subset in ("train", "val"):
                self.assertIsNotNone(doc[subset]["ks"])
                self.assertIsInstance(doc["classes"], list)
                self.assertEqual(len(doc["classes"]), 2)

    # ---------- 多分类 KS/LIFT 表口径（bad_class，不是「编码 1」） ----------
    def test_multiclass_ks_lift_tables_use_bad_class(self) -> None:
        project_dir = self.multi_dir
        train = model_train(
            project_dir,
            {"algorithm": "dt", "task": "multiclass", "features": MULTI_FEATURES},
        )
        self.assertEqual(train["status"], "ok", train)
        run_id = train["data"]["run_id"]
        evaluated = model_evaluate(project_dir, run_id)
        self.assertEqual(evaluated["status"], "ok", evaluated)

        run_dir = Path(project_dir) / "runs" / run_id
        doc = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        bad_label = doc["bad_class"]
        bad_index = int(doc["bad_index"])
        self.assertEqual(str(bad_label), str(doc["classes"][-1]))

        subset = "val" if (doc.get("val") or {}).get("n") else "train"
        block = doc[subset]
        self.assertEqual(block["task"], "multiclass")
        expected_bad = int(block["class_counts"][str(bad_label)])
        self.assertGreater(expected_bad, 0)

        ks_df = pd.read_csv(run_dir / "evaluation" / "ks_table.csv")
        lift_df = pd.read_csv(run_dir / "evaluation" / "lift_table.csv")
        for name, frame in (("ks_table", ks_df), ("lift_table", lift_df)):
            self.assertIn("bad_class", frame.columns, f"{name} 必须注明坏类")
            self.assertEqual(set(frame["bad_class"].astype(str)), {str(bad_label)})
            self.assertEqual(
                int(frame["bad"].sum()),
                expected_bad,
                f"{name} 的 bad 计数必须等于 bad_class 样本数"
                f"（不是「编码 == 1」的那一类：{block['class_counts']}）",
            )
            self.assertEqual(int(frame["n"].sum()), int(block["n"]))

        # 与 per_class[bad_class]（OVR）同口径：表只有 10 个阈值点 ⇒ max KS ≤ OVR KS
        bad_record = next(
            item for item in block["per_class"] if int(item["class_index"]) == bad_index
        )
        self.assertIsNotNone(bad_record["ks"])
        table_ks = float(ks_df["ks"].max())
        self.assertGreater(table_ks, 0.0)
        self.assertLessEqual(table_ks, float(bad_record["ks"]) + 1e-9)

        # 顶层 ks 取 macro（一个 run 一个主指标），与 macro 块自洽
        self.assertEqual(block["ks"], block["macro"]["ks"])

        # metrics.json 显式记录表的坏类口径（表本身看不出「1 是谁」）
        note = doc["ks_lift_tables"]
        self.assertEqual(note["subset"], subset)
        self.assertEqual(str(note["bad_class"]), str(bad_label))
        self.assertEqual(int(note["bad_index"]), bad_index)
        self.assertIn("bad_index", note["label_scheme"])

    def test_ks_lift_tables_reject_non_binary_labels(self) -> None:
        """把多分类原始编码直接喂进表 ⇒ 必须报错（否则坏类被当好样本且看不出来）。"""
        from risklab_mdl.core.errors import DataError
        from risklab_mdl.models.evaluate import ks_table, lift_table

        for func in (ks_table, lift_table):
            with self.assertRaises(DataError):
                func([0, 1, 2], [0.1, 0.2, 0.3])


class EvaluatePsiEdgeTests(unittest.TestCase):
    """PSI 基期边界：不抛 ValueError、不静默回落（口径错位比报错更难查）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="t5-psi-edge-")
        _, cls.project_dir = prepare_project(
            cls._tmp.name, binary_frame(), name="binary_small"
        )
        split = dataset_split(cls.project_dir, SPLIT)
        assert split["status"] == "ok", split
        trained = model_train(
            cls.project_dir,
            {"algorithm": "dt", "task": "binary", "features": BINARY_FEATURES},
        )
        assert trained["status"] == "ok", trained
        cls.run_id = trained["data"]["run_id"]
        cls.months = sorted({str(m) for m in binary_frame()["month"].tolist()})

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_base_months_cover_all_months_disables_psi(self) -> None:
        """I2：基期覆盖全部月份 ⇒ enabled=False + reason，而不是 max() on empty sequence。"""
        self.assertGreaterEqual(len(self.months), 2, self.months)
        payload = model_evaluate(
            self.project_dir, self.run_id, {"psi_base_months": self.months}
        )
        self.assertEqual(payload["status"], "ok", payload.get("error"))
        psi = payload["data"]["psi"]
        self.assertFalse(psi.get("enabled"), psi)
        self.assertIn("没有对比月", str(psi.get("reason")))
        self.assertIsNone(psi.get("max_psi"))
        self.assertNotIn("psi_by_month", payload["data"]["tables"])

    def test_unknown_base_month_is_config_invalid(self) -> None:
        """I3：指定基期月不存在 ⇒ config_invalid（不静默回落到最早月）。"""
        payload = model_evaluate(
            self.project_dir, self.run_id, {"psi_base_months": ["1999-01"]}
        )
        self.assertEqual(payload["status"], "error", payload)
        self.assertEqual(payload["error"]["code"], "config_invalid")
        message = payload["error"]["message"]
        self.assertIn("1999-01", message)
        self.assertIn(self.months[0], message, "报错必须列出可用月份，便于改 config")


if __name__ == "__main__":
    unittest.main()
