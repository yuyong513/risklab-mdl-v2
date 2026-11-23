"""T5 验收 ②：``sample_2000.csv`` 上的二分类 5 算法端到端（宽表 691 列）。

流程与真实使用一致：
复制 CSV 到临时目录（**旧项目目录严格只读**）→ ``project_init`` → ``preprocess_apply``
（缺失填充 + 文本编码）→ ``dataset_split`` → 5 算法 ``model_train`` + ``model_evaluate``。

断言：指标数值有限；总耗时 ≤ 600s（本机实测约 1 分钟）。
样例数据不存在时 skip（不把「缺样例」当成测试失败）。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from fixtures.make_fixtures import prepare_project  # noqa: E402
from risklab_mdl.tools.model import model_evaluate, model_train  # noqa: E402
from risklab_mdl.tools.preprocess import preprocess_apply  # noqa: E402
from risklab_mdl.tools.split import dataset_split  # noqa: E402

SAMPLE_CSV = ROOT.parent / "risklab-mcp-main" / "examples" / "sample_2000.csv"

ALGORITHMS = ("dt", "lr", "gbdt", "xgb", "lgb")
TIME_BUDGET_SEC = 600.0


@unittest.skipUnless(SAMPLE_CSV.is_file(), "缺少样例数据 sample_2000.csv（跳过）")
class SampleFiveAlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import pandas as pd

        cls._tmp = tempfile.TemporaryDirectory(prefix="t5-sample-")
        tmp_dir = Path(cls._tmp.name)
        copied = tmp_dir / "sample_2000.csv"
        shutil.copyfile(SAMPLE_CSV, copied)  # 只读旧项目：先复制再动
        cls.csv_path = copied
        frame = pd.read_csv(copied)
        _, cls.project_dir = prepare_project(
            tmp_dir / "project", frame, name="sample_2000"
        )

        spec = {
            "missing": [
                {"column": column, "strategy": "median"}
                for column in frame.columns
                if column not in ("y", "sample_id", "month", "split", "f_0423")
                and int(frame[column].isna().sum()) > 0
            ],
            "encode": [{"column": "f_0423", "strategy": "onehot"}],
        }
        cls.preprocess = preprocess_apply(cls.project_dir, spec)
        assert cls.preprocess["status"] == "ok", cls.preprocess
        cls.split = dataset_split(
            cls.project_dir,
            {"mode": "stratified", "ratios": [0.7, 0.2, 0.1], "seed": 42},
        )
        assert cls.split["status"] == "ok", cls.split
        summary = cls.preprocess["data"]
        # 入模列清单落在 preprocess/summary.json 的 model_readiness.model_columns
        import json

        summary_doc = json.loads(
            (Path(cls.project_dir) / "preprocess" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        cls.features = list(summary_doc["model_readiness"]["model_columns"])
        cls.timings: dict[str, float] = {}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_preprocess_and_split_are_ready(self) -> None:
        data = self.preprocess["data"]
        self.assertLessEqual(data["rows_out"], data["rows_in"])
        self.assertGreater(len(self.features), 100, "宽表应保留上百个入模特征")
        self.assertTrue(data["model_readiness"]["lr_ready"])
        counts = self.split["data"]["counts"]
        self.assertEqual(sum(counts.values()), data["rows_out"])

    def test_five_algorithms_train_and_evaluate(self) -> None:
        started = time.perf_counter()
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                t0 = time.perf_counter()
                trained = model_train(
                    self.project_dir,
                    {
                        "algorithm": algorithm,
                        "task": "binary",
                        "features": self.features,
                        "early_stopping_rounds": 30,
                        "num_boost_round": 200,
                    },
                )
                self.assertEqual(trained["status"], "ok", f"{algorithm}: {trained}")
                data = trained["data"]
                for subset in ("train", "val"):
                    block = data["metrics"][subset]
                    self.assertEqual(block.get("status"), "ok", f"{algorithm}/{subset}: {block}")
                    for field in ("ks", "auc"):
                        value = block[field]
                        self.assertIsNotNone(value)
                        self.assertGreaterEqual(float(value), 0.0)
                        self.assertLessEqual(float(value), 1.0 + 1e-9)
                    # LIFT 上限是 1/坏率（不是 1）；坏样本极少时尾部可能一个坏都没抓到 ⇒ LIFT=0 合法
                    lift = block["lift"]
                    self.assertIsNotNone(lift)
                    self.assertGreaterEqual(float(lift), 0.0)
                    self.assertTrue(float(np.isfinite(lift)))
                # test 子集极小（0.1 比例）可能只剩一类 ⇒ status 允许 skipped，
                # 但 skipped 必须给原因、ok 必须给数值 —— 不接受「status 非空」这种恒真断言
                test_block = data["metrics"]["test"]
                self.assertIn(
                    test_block.get("status"), ("ok", "skipped"), f"{algorithm}: {test_block}"
                )
                if test_block["status"] == "ok":
                    self.assertIsNotNone(test_block.get("ks"), f"{algorithm}: {test_block}")
                    self.assertIsNotNone(test_block.get("auc"), f"{algorithm}: {test_block}")
                else:
                    self.assertTrue(
                        test_block.get("reason"),
                        f"{algorithm}: skipped 必须带原因：{test_block}",
                    )

                evaluated = model_evaluate(self.project_dir, data["run_id"])
                self.assertEqual(evaluated["status"], "ok", f"{algorithm}: {evaluated}")
                psi = evaluated["data"]["psi"]
                self.assertTrue(psi.get("enabled"), f"{algorithm}: PSI 应可用（有 month 列）")
                self.assertIsNotNone(psi.get("max_psi"))
                self.timings[algorithm] = round(time.perf_counter() - t0, 2)
                print(f"[sample] {algorithm}: {self.timings[algorithm]}s, "
                      f"val KS={data['metrics']['val']['ks']}, "
                      f"val AUC={data['metrics']['val']['auc']}, "
                      f"max PSI={psi.get('max_psi')}")

        elapsed = time.perf_counter() - started
        print(f"[sample] 总耗时 {elapsed:.1f}s；逐算法 {self.timings}")
        self.assertLess(
            elapsed, TIME_BUDGET_SEC, f"5 算法端到端耗时 {elapsed:.1f}s 超出预算"
        )

    def test_lgb_metrics_are_comparable_across_repeats(self) -> None:
        """同参同 seed 两次训练：val KS 必须一致（可复现，不是随机数）。"""
        config = {
            "algorithm": "lgb",
            "task": "binary",
            "features": self.features[:80],
            "seed": 7,
            "num_boost_round": 60,
            "early_stopping_rounds": 15,
        }
        first = model_train(self.project_dir, config)
        self.assertEqual(first["status"], "ok", first)
        second = model_train(self.project_dir, config)
        self.assertEqual(second["status"], "ok", second)
        self.assertEqual(
            first["data"]["metrics"]["val"]["ks"],
            second["data"]["metrics"]["val"]["ks"],
            "同参同 seed 的 val KS 不一致 ⇒ 训练链路有未固定的随机源",
        )


if __name__ == "__main__":
    unittest.main()
