"""精筛（gain 迭代）与训练编排测试 —— 需要 LightGBM（本机已安装）。

★ 这些测试守护的是**迁移后的可用性**，不是算法优劣
--------------------------------------------------
旧 ``refine.py`` / ``train.py`` 在模块顶层
``from ..adapters.lgbm_env import ensure_lightgbm`` —— 这既是包耦合，
也意味着「lightgbm 没装 ⇒ core 整包导入失败」。
v2 改成函数内惰性 import（:func:`risklab_mdl.core.deps.require_lightgbm`），
缺库时抛 ``ModelError`` 且带 ``pip install lightgbm`` 提示。

⇒ 本文件同时验证：

1. **装了库**：精筛/训练能跑出合理结构（箱数/特征数/指标齐全）；
2. **没装库**：错误是 ``ModelError``（不是 ImportError / NameError），
   且消息里有安装命令 —— 这条用 ``sys.modules`` 注入 ``None`` 模拟缺库，
   不会真的卸载环境里的 lightgbm。

★ 小样本 + 少轮数
  自造 400 行、30 轮、早停 10 —— 单次运行约 1 秒，避免测试慢到没人跑。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures.make_fixtures import binary_frame  # noqa: E402

from risklab_mdl.core import deps  # noqa: E402
from risklab_mdl.core.errors import ConfigError, ModelError  # noqa: E402
from risklab_mdl.core.refine import (  # noqa: E402
    RefineConfig,
    select_features_by_gain,
)
from risklab_mdl.core.train import build_train_config, train_model  # noqa: E402
from risklab_mdl.core.tuning import (  # noqa: E402
    ObjectiveWeights,
    TuningConfig,
    build_search_space,
    tune_model,
)

LGB_PARAMS = {"objective": "binary", "metric": "auc", "verbose": -1}
FEATURES = ["f_signal", "f_weak", "f_noise", "f_dup", "f_missing"]


def _split(df: pd.DataFrame):
    train = df[df["split"] == "train"]
    val = df[df["split"] != "train"]
    return train, val


class LazyDependencyTests(unittest.TestCase):
    """缺库时必须报 ModelError（不是裸 ImportError）。"""

    def test_require_lightgbm_missing_raises_model_error(self) -> None:
        with mock.patch.dict(sys.modules, {"lightgbm": None}):
            with self.assertRaises(ModelError) as ctx:
                deps.require_lightgbm()
        message = str(ctx.exception)
        self.assertIn("pip install lightgbm", message)
        self.assertIn("MODEL_ERROR", message)

    def test_require_xgboost_missing_raises_model_error(self) -> None:
        with mock.patch.dict(sys.modules, {"xgboost": None}):
            with self.assertRaises(ModelError) as ctx:
                deps.require_xgboost()
        self.assertIn("pip install xgboost", str(ctx.exception))

    def test_require_matplotlib_missing_raises_model_error(self) -> None:
        with mock.patch.dict(sys.modules, {"matplotlib": None}):
            with self.assertRaises(ModelError) as ctx:
                deps.require_matplotlib()
        self.assertIn("pip install matplotlib", str(ctx.exception))

    def test_detail_carries_dependency_and_install(self) -> None:
        with mock.patch.dict(sys.modules, {"lightgbm": None}):
            with self.assertRaises(ModelError) as ctx:
                deps.require_lightgbm()
        self.assertEqual(ctx.exception.detail["dependency"], "lightgbm")
        self.assertEqual(ctx.exception.detail["install"], "pip install lightgbm")


class RefineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = binary_frame()
        self.train, self.val = _split(self.df)

    def test_gain_iteration_returns_features_in_range(self) -> None:
        cfg = RefineConfig(
            target_range=(3, 4),
            max_rounds=3,
            num_boost_round=30,
            early_stopping_rounds=10,
            min_features=2,
            seed=42,
        )
        result = select_features_by_gain(
            self.train[FEATURES], self.train["y"],
            self.val[FEATURES], self.val["y"],
            LGB_PARAMS, cfg, candidates=FEATURES,
        )
        low, high = cfg.target_range
        self.assertGreaterEqual(len(result.features), min(low, len(FEATURES)))
        self.assertLessEqual(len(result.features), high)
        self.assertTrue(set(result.features) <= set(FEATURES))
        self.assertIn("f_signal", result.features, "最强信号不应被精筛剔掉")

    def test_gain_table_and_rounds(self) -> None:
        cfg = RefineConfig(
            target_range=(3, 4), num_boost_round=30, early_stopping_rounds=10,
            max_rounds=3, min_features=2,
        )
        result = select_features_by_gain(
            self.train[FEATURES], self.train["y"],
            self.val[FEATURES], self.val["y"],
            LGB_PARAMS, cfg, candidates=FEATURES,
        )
        self.assertIsNotNone(result.gain_table)
        self.assertFalse(result.gain_table.empty)
        self.assertTrue(result.rounds, "至少应有一轮记录（口径留痕）")
        self.assertIn("gain", " ".join(result.gain_table.columns).lower())

    def test_force_keep_and_force_drop(self) -> None:
        cfg = RefineConfig(
            target_range=(2, 5), num_boost_round=30, early_stopping_rounds=10,
            max_rounds=2, min_features=1,
            force_keep=("f_noise",), force_drop=("f_weak",),
        )
        result = select_features_by_gain(
            self.train[FEATURES], self.train["y"],
            self.val[FEATURES], self.val["y"],
            LGB_PARAMS, cfg, candidates=FEATURES,
        )
        self.assertIn("f_noise", result.features, "强保留必须生效")
        self.assertNotIn("f_weak", result.features, "强剔除必须生效")
        self.assertIn("f_noise", result.force_kept)
        self.assertIn("f_weak", result.force_dropped)

    def test_unsupported_method_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            RefineConfig(method="shap")  # 备选方案未实现，必须显式报错
        with self.assertRaises(ConfigError):
            RefineConfig(target_range=(5, 3))


class TrainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = binary_frame()
        self.train, self.val = _split(self.df)

    def test_train_model_metrics_and_package(self) -> None:
        cfg = build_train_config(
            algorithm="lgb", params=LGB_PARAMS, num_boost_round=40,
            early_stopping_rounds=10, seed=42,
        )
        outcome = train_model(
            self.train[FEATURES], self.train["y"],
            FEATURES,
            self.val[FEATURES], self.val["y"],
            cfg=cfg,
        )
        for key in ("train_ks", "train_auc", "val_ks", "val_auc"):
            self.assertIn(key, outcome.metrics, f"metrics 缺少 {key}")
        self.assertTrue(all(np.isfinite(outcome.metrics[k]) for k in outcome.metrics))
        self.assertEqual(outcome.package.algorithm, "lgb")
        self.assertEqual(list(outcome.package.features), FEATURES)
        self.assertIsNotNone(outcome.feature_importance)
        self.assertEqual(int(outcome.n_train), len(self.train))
        self.assertEqual(int(outcome.n_val), len(self.val))

    def test_unsupported_algorithm_rejected(self) -> None:
        """不支持的算法必须报错，绝不静默用 LGB 顶替。

        ★ 报错点在 ``build_train_config``（构造期就拦），而不是训练到一半 ——
          测试把两步都包进 ``assertRaises``，避免「换个版本就假红」。
        """
        with self.assertRaises(ConfigError):
            cfg = build_train_config(algorithm="lr", params={})
            train_model(
                self.train[FEATURES], self.train["y"], FEATURES,
                self.val[FEATURES], self.val["y"], cfg=cfg,
            )


class TuningTests(unittest.TestCase):
    def test_build_search_space(self) -> None:
        space = build_search_space(
            {
                "num_leaves": {"type": "int", "low": 4, "high": 16},
                "learning_rate": {"type": "uniform", "low": 0.01, "high": 0.3},
            },
            seed=42,
        )
        self.assertEqual(set(space.params), {"num_leaves", "learning_rate"})

    def test_tune_model_with_trivial_objective(self) -> None:
        """调参框架本身可跑（trial_fn 用假目标，不训练模型 ⇒ 秒级）。"""
        space = build_search_space(
            {"x": {"type": "uniform", "low": 0.0, "high": 1.0}}, seed=42
        )
        cfg = TuningConfig(space=space, weights=ObjectiveWeights(), n_iters=2, seed=42)

        calls: list[int] = []

        def trial_fn(params: dict, index: int):
            calls.append(index)
            value = 0.5 + 0.4 * float(params["x"])
            # ★ 目标函数只认 cv_* 键（cv_ks / cv_auc / cv_lift / train_cv_gap）：
            #   权重为 0 的指标缺失不影响；权重 >0 而指标缺失 ⇒ 该次试验判失败。
            return {"cv_ks": value, "cv_auc": value, "train_cv_gap": 0.0}, None

        result = tune_model(cfg, trial_fn)
        self.assertEqual(len(calls), 2)
        self.assertTrue(result.trials)
        self.assertFalse(result.aborted, result.abort_reason)
        self.assertGreater(float(result.best_target), -900.0)
        self.assertIn("x", result.best_params)


if __name__ == "__main__":
    unittest.main()
