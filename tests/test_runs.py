"""T5 验收 ④：run 追踪 —— ``run_list`` / ``run_compare`` / 目录并存 / manifest 记录。

断言：
* 两次训练 → ``run_list`` 两条、两个 run 目录并存；
* ``run_compare`` 对比全部 run 并标出 best（主指标 KS）；
* manifest.runs 是**追加**而非覆盖（两次训练都在）；
* ``report_build`` 产出多 sheet 非空 xlsx + md（只读产物，不重算）。
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
from risklab_mdl.artifacts.manifest import read_manifest  # noqa: E402
from risklab_mdl.tools.artifact import report_build, run_compare, run_list  # noqa: E402
from risklab_mdl.tools.model import model_evaluate, model_train  # noqa: E402
from risklab_mdl.tools.split import dataset_split  # noqa: E402

FEATURES = ["f_signal", "f_weak", "f_noise", "f_missing"]
SPLIT = {"mode": "stratified", "ratios": [0.7, 0.15, 0.15], "seed": 42}


class RunTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="t5-runs-")
        _, cls.project_dir = prepare_project(
            cls._tmp.name, binary_frame(), name="binary_small"
        )
        assert dataset_split(cls.project_dir, SPLIT)["status"] == "ok"
        cls.runs = []
        for algorithm in ("lgb", "dt"):
            payload = model_train(
                cls.project_dir,
                {"algorithm": algorithm, "task": "binary", "features": FEATURES},
            )
            assert payload["status"] == "ok", payload
            cls.runs.append(payload["data"]["run_id"])
            evaluated = model_evaluate(cls.project_dir, payload["data"]["run_id"])
            assert evaluated["status"] == "ok", evaluated

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_run_list_has_two_entries_ascending(self) -> None:
        payload = run_list(self.project_dir)
        self.assertEqual(payload["status"], "ok", payload.get("error"))
        runs = payload["data"]["runs"]
        self.assertEqual(len(runs), 2)
        self.assertEqual([entry["run_id"] for entry in runs], sorted(self.runs))
        self.assertEqual(payload["data"]["n_runs"], 2)
        for entry in runs:
            self.assertIn(entry["algorithm"], ("lgb", "dt"))
            self.assertEqual(entry["task"], "binary")
            self.assertEqual(entry["status"], "evaluated")
            self.assertTrue(entry["has_model"])
            self.assertIsNotNone(entry["primary_ks"])

    def test_both_run_dirs_exist(self) -> None:
        runs_dir = Path(self.project_dir) / "runs"
        for run_id in self.runs:
            run_dir = runs_dir / run_id
            self.assertTrue(run_dir.is_dir(), f"run 目录缺失：{run_dir}")
            for name in ("config.json", "params.json", "metrics.json", "model.pkl"):
                self.assertTrue((run_dir / name).is_file(), f"{run_id} 缺 {name}")

    def test_manifest_appends_runs(self) -> None:
        manifest = read_manifest(Path(self.project_dir))
        run_ids = [entry["run_id"] for entry in manifest["runs"]]
        self.assertEqual(sorted(run_ids), sorted(self.runs))
        self.assertEqual(manifest["stages"]["train"]["done"], True)

    def test_run_compare_marks_best(self) -> None:
        payload = run_compare(self.project_dir)
        self.assertEqual(payload["status"], "ok", payload.get("error"))
        data = payload["data"]
        self.assertEqual(data["primary_metric"], "ks")
        self.assertIn(data["best"], self.runs)
        self.assertEqual(data["n_runs"], 2)
        csv_path = Path(self.project_dir) / "reports" / "run_compare.csv"
        self.assertTrue(csv_path.is_file())
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3, "表头 + 2 个 run")
        best_values = {}
        for row in data["rows"]:
            best_values[row["run_id"]] = row["primary_ks"]
        expected_best = max(best_values, key=lambda rid: best_values[rid])
        self.assertEqual(data["best"], expected_best)

    def test_run_compare_subset(self) -> None:
        payload = run_compare(self.project_dir, [self.runs[0]])
        self.assertEqual(payload["status"], "ok", payload.get("error"))
        self.assertEqual(payload["data"]["n_runs"], 1)
        self.assertEqual(payload["data"]["best"], self.runs[0])

    def test_run_compare_unknown_run(self) -> None:
        payload = run_compare(self.project_dir, ["20200101T000000"])
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "run_not_found")

    def test_report_build_sheets_and_markdown(self) -> None:
        payload = report_build(self.project_dir)
        self.assertEqual(payload["status"], "ok", payload.get("error"))
        data = payload["data"]
        xlsx_path = Path(self.project_dir) / "reports" / "结果汇总.xlsx"
        md_path = Path(self.project_dir) / "reports" / "过程报告.md"
        self.assertTrue(xlsx_path.is_file())
        self.assertTrue(md_path.is_file())

        from openpyxl import load_workbook

        workbook = load_workbook(xlsx_path, read_only=True)
        try:
            self.assertEqual(
                set(workbook.sheetnames),
                {"概览", "模型指标", "run对比", "特征重要性"},
            )
            for name in workbook.sheetnames:
                sheet = workbook[name]
                self.assertGreaterEqual(sheet.max_row, 2, f"sheet {name} 只有表头")
                self.assertGreaterEqual(sheet.max_column, 2, f"sheet {name} 列数不足")
        finally:
            workbook.close()

        text = md_path.read_text(encoding="utf-8")
        for section in ("建模过程报告", "流程阶段", "模型与指标", "确认状态"):
            self.assertIn(section, text)
        self.assertIn(self.runs[0], text)
        self.assertEqual(data["n_runs"], 2)
        self.assertIn(data["best_run"], self.runs)

    def test_report_build_does_not_retrain(self) -> None:
        before = len(read_manifest(Path(self.project_dir))["runs"])
        report_build(self.project_dir)
        after = len(read_manifest(Path(self.project_dir))["runs"])
        self.assertEqual(before, after, "report_build 不得新增 run（只读产物）")

    def test_status_only_moves_forward(self) -> None:
        """状态只前进：重跑评价不得把 confirmed/evaluated 打回更早的状态。

        用独立项目（避免改动本类其它用例的 run 状态）。
        """
        from risklab_mdl.artifacts.layout import resolve_project
        from risklab_mdl.runs.store import update_run

        with tempfile.TemporaryDirectory(prefix="t5-runs-status-") as tmp:
            _, project_dir = prepare_project(
                Path(tmp), binary_frame(seed=17), name="binary_small"
            )
            self.assertEqual(dataset_split(project_dir, SPLIT)["status"], "ok")
            trained = model_train(
                project_dir,
                {"algorithm": "dt", "task": "binary", "features": ["f_signal", "f_weak"]},
            )
            self.assertEqual(trained["status"], "ok", trained)
            run_id = trained["data"]["run_id"]
            layout = resolve_project(project_dir)

            def entry() -> dict:
                manifest = read_manifest(Path(project_dir))
                return {item["run_id"]: item for item in manifest["runs"]}[run_id]

            update_run(layout, run_id, status="evaluated")
            self.assertEqual(entry()["status"], "evaluated")

            update_run(layout, run_id, status="trained")
            self.assertEqual(
                entry()["status"], "evaluated", "evaluated → trained 的状态回退必须被拦住"
            )

            update_run(layout, run_id, status="confirmed")
            self.assertEqual(entry()["status"], "confirmed")

            update_run(
                layout, run_id, status="evaluated", metrics_patch={"ks": 0.5}
            )
            after = entry()
            self.assertEqual(
                after["status"], "confirmed", "confirmed 之后重跑评价不得把状态打回 evaluated"
            )
            self.assertEqual(after["metrics"].get("ks"), 0.5, "指标摘要仍要更新（只拦状态）")


if __name__ == "__main__":
    unittest.main()
