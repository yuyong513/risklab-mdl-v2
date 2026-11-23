"""快速端到端：小型合成数据走完整条建模链路。

为什么不在这里跑 ``sample_2000.csv``
-----------------------------------
2000 行 × 696 列的全流程约 1 分钟（feature_analyze / feature_prescreen 占大头），
放进每次 ``pytest`` 会拖慢整个套件。所以：

* 本文件用 ``tests/fixtures`` 的 400 行平衡二分类数据（5 个数值特征 + 1 个文本列），
  跑 ``project_init → ... → report_build``，验证**链路与产物契约**；
* 真实宽表（如 2000 行 × 696 列）的耗时验证请自行用真实 CSV 跑一遍。

覆盖点
------
1. 9 环节 15 次工具调用全部 ``status=ok``（含 ``model_tune(preview)`` /
   ``run_list`` / ``run_compare`` 三个可选调用）；
2. 产物目录与关键文件落盘；``final/confirm.json`` 可解析且与 manifest 一致；
3. 确认闸门：严阈值必须被拦下（``threshold_not_met`` + 差距值、不写 ``final/``），
   宽松阈值才通过；
4. ``reports/结果汇总.xlsx`` 可被 openpyxl 打开、每个 sheet 非空；
5. 整条链路耗时 ≤ 120 s（本机实测约 5 s，远低于预算）。
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from fixtures.make_fixtures import binary_frame  # noqa: E402
from risklab_mdl.artifacts.manifest import STAGES, read_manifest  # noqa: E402
from risklab_mdl.artifacts.store import RUN_ID_PATTERN  # noqa: E402
from risklab_mdl.tools.artifact import report_build, run_compare, run_list  # noqa: E402
from risklab_mdl.tools.dataset import (  # noqa: E402
    dataset_columns,
    dataset_load,
    dataset_profile,
    project_init,
)
from risklab_mdl.tools.feature import feature_analyze  # noqa: E402
from risklab_mdl.tools.model import (  # noqa: E402
    model_catalog,
    model_confirm,
    model_evaluate,
    model_train,
    model_tune,
)
from risklab_mdl.tools.preprocess import preprocess_apply, preprocess_check  # noqa: E402
from risklab_mdl.tools.selection import feature_prescreen, feature_refine  # noqa: E402
from risklab_mdl.tools.split import dataset_split  # noqa: E402

TARGET = {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}}
#: 宽松阈值：合成数据平衡且信号强，PSI 很小；宽松值只为让通过路径可断言。
LOOSE_THRESHOLDS = {"ks_min": 0.05, "psi_max": 10.0}
#: 严阈值：只用来验证闸门确实会拦下，并给出差距值。
STRICT_THRESHOLDS = {"ks_min": 0.99, "psi_max": 0.000001}

TIME_BUDGET_SEC = 120.0


def _ok(payload: Dict[str, Any], step: str) -> Dict[str, Any]:
    """断言三态自洽 + ``status=ok``；失败时把错误原文抛出来。"""
    assert isinstance(payload, dict), f"{step}: 工具必须返回 dict"
    assert payload.get("status") == "ok" and payload.get("ok") is True, (
        f"{step}: 期望 status=ok，实际 status={payload.get('status')} "
        f"/ error={payload.get('error')} / warnings={payload.get('warnings')}"
    )
    assert payload.get("error") is None, f"{step}: status=ok 时 error 必须为 None"
    return payload["data"]


class FastEndToEndTests(unittest.TestCase):
    """小型合成数据上的全链路（约 5 s）。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="t7-e2e-small-")
        self.tmp = Path(self._tmp.name)
        self.timings: Dict[str, float] = {}
        self.started = time.perf_counter()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, name: str, fn, *args, **kwargs) -> Dict[str, Any]:
        t0 = time.perf_counter()
        payload = fn(*args, **kwargs)
        self.timings[name] = round(time.perf_counter() - t0, 2)
        return _ok(payload, name)

    def test_full_chain_on_small_synthetic_data(self) -> None:
        frame = binary_frame()
        csv_path = self.tmp / "e2e_small.csv"
        frame.to_csv(csv_path, index=False, encoding="utf-8")

        # ---------- 环节 1：项目引导 / 读表 / 列清单 ----------
        init = self._run("project_init", project_init, str(csv_path))
        project_dir = Path(init["project_dir"])
        self.assertEqual(project_dir.name, "mdl-e2e_small")
        self.assertTrue(init["created"] is True)
        self.assertTrue((project_dir / "manifest.json").is_file())

        load = self._run("dataset_load", dataset_load, str(csv_path))
        self.assertEqual(load["rows"], len(frame))
        self.assertEqual(load["column_count"], len(frame.columns))
        self.assertEqual(load["roles"]["id"], "id")
        self.assertEqual(load["roles"]["time"], "month")
        self.assertIn("y", load["roles"]["target_candidates"])

        columns = self._run("dataset_columns", dataset_columns, str(project_dir))
        self.assertIsNone(columns["confirmed_target"], "列清单阶段目标字段还没确认")
        by_name = {item["name"]: item for item in columns["columns"]}
        self.assertEqual(by_name["y"]["role"], "target")
        self.assertEqual(by_name["month"]["role"], "time")

        # ---------- 环节 2：样本画像（目标字段在这里确认） ----------
        profile = self._run(
            "dataset_profile",
            dataset_profile,
            str(project_dir),
            {"target": TARGET, "max_features_for_charts": 10},
        )
        self.assertEqual(profile["shape"], {"rows": len(frame), "columns": len(frame.columns)})
        self.assertEqual(profile["target_distribution"]["counts"], {"0": 200, "1": 200})
        self.assertTrue(profile["has_time_column"])
        for name in ("profile.json", "describe.csv", "target_distribution.csv", "index.html"):
            self.assertTrue((project_dir / "data_analysis" / name).is_file(), name)

        # ---------- 环节 3：特征分析 ----------
        analysis = self._run(
            "feature_analyze",
            feature_analyze,
            str(project_dir),
            {"method": "quantile", "bins": 8},
        )
        self.assertGreaterEqual(analysis["feature_count"], 5)
        self.assertGreaterEqual(analysis["analyzed_count"], 4)
        skipped = {item["feature"]: item["reason"] for item in analysis["skipped"]}
        self.assertIn("f_const", skipped, "常量列必须被显式跳过并给原因")
        self.assertTrue((project_dir / "feature_analysis" / "iv_table.csv").is_file())
        self.assertTrue((project_dir / "feature_analysis" / "woe_detail.csv").is_file())
        self.assertTrue((project_dir / "feature_analysis" / "index.html").is_file())

        # ---------- 环节 5：特征筛选（尚未划分 ⇒ 全量口径，工具会告警） ----------
        prescreen = self._run(
            "feature_prescreen",
            feature_prescreen,
            str(project_dir),
            {"iv_min": 0.02, "psi_max": 0.25, "corr_max": 0.8},
        )
        self.assertGreater(prescreen["n_input"], 0)
        self.assertGreaterEqual(prescreen["n_survived"], 1)
        self.assertLess(prescreen["n_survived"], prescreen["n_input"])
        self.assertTrue(prescreen["data_scope"]["source"].startswith("未划分"))
        self.assertTrue((project_dir / "feature_selection" / "prescreen.json").is_file())

        # ---------- 环节 6：入模前检查 + 精调 + 预处理 ----------
        refine = self._run(
            "feature_refine",
            feature_refine,
            str(project_dir),
            {"scheme": "gain", "rounds": 3, "threshold": 0.05},
        )
        self.assertGreaterEqual(refine["n_selected"], 1)
        selected_text = (project_dir / "feature_selection" / "selected_features.txt").read_text(
            encoding="utf-8"
        )
        self.assertEqual(selected_text.split(), refine["features"])

        check = self._run("preprocess_check", preprocess_check, str(project_dir))
        self.assertGreater(check["must_fix_count"], 0)
        self.assertTrue((project_dir / "preprocess" / "check.json").is_file())

        applied = self._run(
            "preprocess_apply",
            preprocess_apply,
            str(project_dir),
            {"missing": {"f_missing": "median"}, "encode": {"f_text": "onehot"}},
        )
        self.assertEqual(applied["rows_in"], len(frame))
        self.assertEqual(applied["rows_out"], len(frame), "本 spec 不删行，掩码不会错位")
        self.assertGreaterEqual(applied["columns_out"], applied["columns_in"])
        self.assertTrue(applied["model_readiness"]["lr_ready"])
        processed = project_dir / applied["processed_path"]
        self.assertIsInstance(processed, Path)
        self.assertTrue(processed.is_file())
        self.assertIn(processed.suffix, {".parquet", ".csv"})

        # ---------- 环节 7：样本划分 ----------
        split = self._run(
            "dataset_split",
            dataset_split,
            str(project_dir),
            {
                "mode": "stratified",
                "ratios": [0.6, 0.2, 0.2],
                "seed": 42,
                "stratify_by": "y",
                "time_column": "month",
            },
        )
        self.assertEqual(split["counts"], {"train": 240, "val": 80, "test": 80})
        self.assertEqual(len(split["fingerprint"]), 16)
        split_doc = json.loads((project_dir / "split" / "split.json").read_text(encoding="utf-8"))
        self.assertEqual(split_doc["fingerprint"], split["fingerprint"])

        # ---------- 环节 4：模型选型 ----------
        catalog = self._run("model_catalog", model_catalog)
        algorithms = catalog["catalog"]["algorithms"]
        self.assertEqual(sorted(algorithms), ["dt", "gbdt", "lgb", "lr", "xgb"])
        self.assertEqual(catalog["catalog"]["matrix_size"], 10)

        # ---------- 环节 8：训练（先预览调参空间） ----------
        preview = self._run(
            "model_tune",
            model_tune,
            str(project_dir),
            {"algorithm": "lgb", "task": "binary"},
            "preview",
        )
        self.assertEqual(preview["mode"], "preview")
        self.assertEqual(preview["tune_on"], "val")
        self.assertEqual(preview["sample_sizes"], split["counts"])

        train = self._run(
            "model_train",
            model_train,
            str(project_dir),
            {"algorithm": "lgb", "task": "binary", "seed": 42},
        )
        run_id = train["run_id"]
        self.assertTrue(RUN_ID_PATTERN.match(run_id), run_id)
        self.assertEqual(train["task"], "binary")
        self.assertTrue(train["data_source"].startswith("preprocess/"))
        for subset in ("train", "val", "test"):
            block = train["metrics"][subset]
            self.assertEqual(block["status"], "ok", (subset, block))
            self.assertGreaterEqual(block["ks"], 0.0)
            self.assertLessEqual(block["ks"], 1.0)
            self.assertEqual(block["n"], split["counts"][subset])
        self.assertGreater(train["metrics"]["val"]["ks"], 0.2)
        for name in ("config.json", "params.json", "metrics.json", "model.pkl", "importance.csv"):
            self.assertTrue((project_dir / "runs" / run_id / name).is_file(), name)

        # ---------- 环节 9：评估 → 闸门 → 报告 ----------
        evaluate = self._run("model_evaluate", model_evaluate, str(project_dir), run_id)
        psi = evaluate["psi"]
        self.assertTrue(psi["enabled"], "fixture 有 3 个月，PSI 必须可用")
        self.assertIsNotNone(psi["max_psi"])
        self.assertTrue(psi["by_month"])
        self.assertGreater(evaluate["tables"]["ks_table"], 0)
        self.assertGreater(evaluate["tables"]["lift_table"], 0)

        listed = self._run("run_list", run_list, str(project_dir))
        self.assertEqual([item["run_id"] for item in listed["runs"]], [run_id])
        self.assertEqual(listed["runs"][0]["status"], "evaluated")

        # 9.1 严阈值必被拦下：给差距值、不写 final/
        strict = model_confirm(str(project_dir), run_id, STRICT_THRESHOLDS)
        self.assertEqual(strict["status"], "error", strict)
        self.assertEqual(strict["error"]["code"], "threshold_not_met")
        self.assertTrue(strict["data"]["failed_checks"])
        self.assertGreater(strict["data"]["ks_gap"], 0.0)
        self.assertFalse((project_dir / "final" / "confirm.json").exists())
        self.assertFalse((project_dir / "final" / "model.pkl").exists())

        # 9.2 宽松阈值通过
        confirm = self._run(
            "model_confirm", model_confirm, str(project_dir), run_id, LOOSE_THRESHOLDS
        )
        self.assertTrue(confirm["confirmed"])
        self.assertEqual(confirm["failed_checks"], [])
        self.assertEqual(confirm["thresholds"], LOOSE_THRESHOLDS)
        confirm_doc = json.loads(
            (project_dir / "final" / "confirm.json").read_text(encoding="utf-8")
        )
        self.assertEqual(confirm_doc["run_id"], run_id)
        self.assertEqual(confirm_doc["thresholds"], LOOSE_THRESHOLDS)
        self.assertTrue((project_dir / "final" / "model.pkl").is_file())

        manifest = read_manifest(project_dir)
        self.assertEqual(manifest["confirmed_run"], run_id)
        self.assertIsNotNone(manifest["project"]["target"])
        self.assertEqual(manifest["project"]["target"]["column"], "y")
        self.assertTrue(all(manifest["stages"][stage]["done"] for stage in STAGES))

        # 9.3 汇总报告 + run 对比
        report = self._run("report_build", report_build, str(project_dir))
        self.assertEqual(report["n_runs"], 1)
        self.assertEqual(report["best_run"], run_id)
        self.assertEqual(
            set(report["sheets"]), {"概览", "模型指标", "run对比", "特征重要性"}
        )
        self.assertTrue(all(count > 1 for count in report["sheets"].values()), report["sheets"])
        xlsx_path = project_dir / "reports" / "结果汇总.xlsx"
        self.assertTrue(xlsx_path.is_file())
        self.assertTrue((project_dir / "reports" / "过程报告.md").is_file())
        from openpyxl import load_workbook

        workbook = load_workbook(xlsx_path, read_only=True)
        try:
            self.assertTrue(workbook.sheetnames)
            for name in workbook.sheetnames:
                sheet = workbook[name]
                self.assertGreater(sheet.max_row, 1, f"sheet「{name}」只有表头")
        finally:
            workbook.close()

        compare = self._run("run_compare", run_compare, str(project_dir))
        self.assertEqual(compare["best"], run_id)
        self.assertTrue((project_dir / "reports" / "run_compare.csv").is_file())

        # ---------- 全局：目录齐全 + 耗时 ----------
        for name in (
            "data_analysis",
            "feature_analysis",
            "feature_selection",
            "preprocess",
            "split",
            "runs",
            "final",
            "reports",
        ):
            directory = project_dir / name
            self.assertTrue(directory.is_dir(), f"缺少产物目录：{name}")
            self.assertTrue(any(directory.rglob("*")), f"产物目录为空：{name}")

        elapsed = time.perf_counter() - self.started
        self.assertLess(elapsed, TIME_BUDGET_SEC, f"快速 e2e 超预算：{elapsed:.1f}s")
        print(
            "\n[fast-e2e] 9 环节 / %d 次工具调用全部 status=ok；总耗时 %.2fs；逐步骤耗时：%s"
            % (len(self.timings), elapsed, json.dumps(self.timings, ensure_ascii=False))
        )


if __name__ == "__main__":
    unittest.main()
