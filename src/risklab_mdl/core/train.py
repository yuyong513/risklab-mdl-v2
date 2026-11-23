"""模型训练与评价：TrainConfig / TrainOutcome。

★ 训练标尺恒为流程确认的刻度，不被 AUC 结果反向改写。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .deps import require_lightgbm
from .errors import ConfigError, DataError, ModelError
from .metrics import (
    ScoreScale,
    calc_auc,
    calc_gini,
    calc_ks,
    calc_lift,
    score_direction,
)
from .package import ModelPackage
from .utils import round_value

__all__ = [
    "SUPPORTED_ALGORITHMS",
    "TrainConfig",
    "TrainOutcome",
    "build_train_config",
    "train_model",
]

Algorithm = Literal["lgb", "lr", "xgb"]

SUPPORTED_ALGORITHMS: tuple = ("lgb",)
"""本模块已实现的算法；``lr`` / ``xgb`` 见 :mod:`risklab_mdl.models.adapters`。

★ 不做「自动降级」：传 ``lr`` 却悄悄跑出 LGB 结果，比直接报错糟得多 ——
使用方会按 LR 的口径解读一个树模型的系数，结论完全错位。
"""


@dataclass(frozen=True)
class TrainConfig:
    """训练配置。"""

    algorithm: str = "lgb"
    params: Mapping[str, Any] = field(default_factory=dict)
    """完整参数集（固定参数 + 调参产出的最优搜索参数）。"""

    num_boost_round: int = 1000
    early_stopping_rounds: int | None = 200
    """早停轮数。``None`` 表示不早停（固定轮数训练）。"""

    use_val_for_early_stopping: bool = True
    """★ 是否用 val 早停。

    默认 True —— val 只承担早停职责，**不参与超参选择**。
    用完早停后它仍然是一份「未被选择过」的样本。

    ⚠ 关闭早停（``False``）时轮数必须由 ``params['num_boost_round']`` 决定，
    且此时**没有任何样本可以告诉我们何时该停**。
    """

    score_scale: ScoreScale = field(default_factory=ScoreScale)
    seed: int = 42
    categorical_features: tuple = ()

    def __post_init__(self) -> None:
        if self.algorithm not in SUPPORTED_ALGORITHMS:
            raise ConfigError(
                f"算法 {self.algorithm!r} 未实现。"
                f"仅支持 {list(SUPPORTED_ALGORITHMS)}（LightGBM）；"
                f"lr / xgboost 在后续版本。**服务不会静默用 LGB 顶替** —— "
                f"两者的系数解释与特征重要性语义完全不同，"
                f"评委会按错误的口径解读模型。"
            )
        if not self.params:
            raise ConfigError(
                "训练参数为空。至少需要 objective / metric 等固定参数。"
                "服务不内置默认参数 —— 各项目的口径差异太大。"
            )
        if "objective" not in self.params:
            raise ConfigError(
                "params 缺少 objective。LightGBM 的 objective 决定 "
                "predict 返回的是概率还是 raw score —— "
                "缺失时分数量纲完全不同，标尺转换会得出无意义的分数。"
            )
        if self.use_val_for_early_stopping and self.early_stopping_rounds is None:
            raise ConfigError(
                "启用早停必须提供 early_stopping_rounds。"
                "若无验证集，请把 use_val_for_early_stopping 设为 False。"
            )


@dataclass
class TrainOutcome:
    """训练产出。"""

    package: ModelPackage
    metrics: Mapping[str, float]
    n_train: int
    n_val: int
    best_iteration: int | None
    feature_importance: pd.DataFrame | None = None

    def as_context(self) -> dict:
        return {
            "n_train": self.n_train,
            "n_val": self.n_val,
            "best_iteration": self.best_iteration,
            "metrics": {k: round_value(v) for k, v in self.metrics.items()},
            "package": self.package.manifest(),
        }

    def _warnings(self) -> list:
        """需要使用方特别注意的情形。

        ★ 「早停没触发」的判定为什么要容许 1 的误差
        -------------------------------------------
        LightGBM 在「验证集指标最后一轮仍是历史最优」时会停在最后一轮，
        此时 ``best_iteration == num_boost_round - 1``（0-based 计数）。

        若判定写成严格相等 ``best_iteration == num_boost_round``，
        这种「早停形同虚设」的情形就**不会告警** ——
        而它恰恰是最该告警的（早停白配了，模型很可能欠拟合，或验证集已失效）。
        """
        out = []
        m = self.metrics
        if "train_ks" in m and "val_ks" in m and m["train_ks"] > 0:
            drop = (m["train_ks"] - m["val_ks"]) / m["train_ks"]
            if drop > 0.3:
                out.append(
                    f"train KS {m['train_ks']:.3f} → val KS {m['val_ks']:.3f}，"
                    f"掉幅 {drop:.1%} 超过 30%。上线前请确认不是过拟合。"
                )
        if self.best_iteration is not None and self.best_iteration >= self._max_round() - 1:
            out.append(
                f"best_iteration 为 {self.best_iteration}，已达上限 "
                f"{self._max_round()} —— 说明早停没有真正起作用，模型可能欠拟合。"
                f"可增大 num_boost_round 或检查验证集是否还有区分度。"
            )
        return out

    def _max_round(self) -> int:
        return int(self.package.params.get("num_boost_round", self.package.best_iteration or 0))


def build_train_config(
    *,
    algorithm: str = "lgb",
    params: Mapping[str, Any] | None = None,
    fixed_params: Mapping[str, Any] | None = None,
    best_params: Mapping[str, Any] | None = None,
    num_boost_round: int = 1000,
    early_stopping_rounds: int | None = 200,
    use_val_for_early_stopping: bool = True,
    score_scale: ScoreScale | None = None,
    seed: int = 42,
    categorical_features: Sequence[str] = (),
) -> TrainConfig:
    """由调参产出拼装训练配置。

    ★ 参数合并只有这一处
    -------------------
    ``fixed_params``（不参与搜索的）与 ``best_params``（搜索产出的）在这里
    合并。训练流程内部**不再做任何参数拼装** —— 两处分别拼装正是
    「调参用的参数 ≠ 训练用的参数」这类问题的根源。
    """
    merged: dict = dict(fixed_params or {})
    merged.update(dict(best_params or {}))
    if params:
        merged.update(dict(params))       # 显式传入优先级最高
    return TrainConfig(
        algorithm=algorithm,
        params=merged,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        use_val_for_early_stopping=use_val_for_early_stopping,
        score_scale=score_scale or ScoreScale(),
        seed=seed,
        categorical_features=tuple(categorical_features),
    )


def train_model(
    train: pd.DataFrame,
    y_train: pd.Series,
    features: Sequence[str],
    val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    *,
    cfg: TrainConfig | None = None,
    label_spec_digest: str = "",
    feature_iv: Mapping[str, float] | None = None,
    feature_iv_bins: Mapping[str, Any] | None = None,
) -> TrainOutcome:
    """训练并产出模型包。

    Parameters
    ----------
    train / y_train / features
        训练集与入模特征。**精筛产出的特征集**。
    val / y_val
        验证集，**仅用于早停**。不参与超参选择。

    ★★ train 与 val 的职责边界不可混淆
    ----------------------------------
    ==========  ==========================================
    集合         唯一职责
    ==========  ==========================================
    train        拟合参数
    val          早停（决定训练几轮）
    ==========  ==========================================

    ★★ test **不进本函数**
    ---------------------
    test 只在评价阶段被读取与打分，**绝不参与训练、早停、调参**。
    """
    cfg = cfg or TrainConfig(
        algorithm="lgb", params={"objective": "binary", "metric": "auc", "verbose": -1}
    )
    lgb = require_lightgbm()

    if not features:
        raise ConfigError("入模特征为空，无法训练")
    missing = [c for c in features if c not in train.columns]
    if missing:
        raise DataError(f"入模特征在训练集中不存在：{missing[:10]}")

    if cfg.use_val_for_early_stopping:
        if val is None or y_val is None:
            raise ConfigError(
                "启用了 val 早停但未提供验证集。"
                "★ 训练必须有一份没被用于选参的样本来决定训练轮数 —— "
                "直接用固定轮数会让模型因过拟合而失效，"
                "而这个失效在训练报表里完全看不出来。"
            )

    params = dict(cfg.params)
    params["random_state"] = cfg.seed
    params.pop("num_boost_round", None)      # 轮数不是 LGB 的 params

    cats = [c for c in cfg.categorical_features if c in set(features)] or None
    train_data = lgb.Dataset(train[list(features)], y_train, categorical_feature=cats)

    valid_sets = []
    valid_names = []
    if cfg.use_val_for_early_stopping:
        val_data = lgb.Dataset(
            val[list(features)], y_val, reference=train_data, categorical_feature=cats
        )
        valid_sets.append(val_data)
        valid_names.append("valid_0")

    callbacks = [lgb.log_evaluation(0)]
    if valid_sets and cfg.early_stopping_rounds:
        callbacks.append(lgb.early_stopping(cfg.early_stopping_rounds, verbose=False))

    try:
        model = lgb.train(
            params,
            train_data,
            valid_sets=valid_sets or None,
            valid_names=valid_names or None,
            num_boost_round=cfg.num_boost_round,
            callbacks=callbacks,
        )
    except (ValueError, RuntimeError) as exc:
        raise ModelError(
            f"LightGBM 训练失败：{exc}\n"
            f"常见原因：① min_data_in_leaf 远大于样本量；"
            f"② 特征中存在全常量列；③ 标签只有单一类别。"
            f"当前参数：{ {k: v for k, v in params.items() if k != 'verbose'} }"
        ) from exc

    best_iter = _best_iteration(model, cfg.num_boost_round)

    metrics: dict = {}
    train_scores = _predict(model, train[list(features)])
    metrics.update(_metrics("train", y_train, train_scores))

    if val is not None and y_val is not None and len(val):
        metrics.update(_metrics("val", y_val, _predict(model, val[list(features)])))

    pkg_params = dict(params)
    pkg_params["num_boost_round"] = cfg.num_boost_round
    pkg_params["seed"] = cfg.seed

    package = ModelPackage(
        model=model,
        algorithm="lgb",                       # type: ignore[arg-type]
        features=list(features),
        params=pkg_params,
        best_iteration=best_iter,
        transforms=[],                         # 旧版 LGB 无变换，字段先就位
        fill_values={},                        # LGB 原生支持 NaN
        score_scale=cfg.score_scale,
        bin_specs={},
        label_spec_digest=label_spec_digest,
        metrics=metrics,
        feature_iv=dict(feature_iv or {}),
        feature_iv_bins=dict(feature_iv_bins or {}),
    )

    return TrainOutcome(
        package=package,
        metrics=metrics,
        n_train=int(len(train)),
        n_val=int(len(val)) if val is not None else 0,
        best_iteration=best_iter,
        feature_importance=_importance_frame(model, list(features)),
    )


def _predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """LightGBM Booster 只有 ``predict``。objective=binary 时返回的就是概率。

    ⚠ 不加 sigmoid —— LGB 已经做了。再加一次会让概率全部映射到 0.73以上，
    分数看起来正常但区分度全丢。
    """
    return np.asarray(model.predict(X), dtype=float).ravel()


def _best_iteration(model: Any, trained_rounds: int) -> int | None:
    """★ 实际使用的树数。**必须落盘**。

    不落盘 → 增量重打分时不知道该用多少棵树 → 分数与训练时不同。

    ★ 关闭早停时 LGB 的 ``best_iteration`` 为 None
    --------------------------------------------
    但此时模型**确实**训练了 ``trained_rounds`` 轮，这个值必须填上。
    填 None 会让「增量重打分」退化为「用默认轮数」——
    而默认轮数往往比实际训练轮数多，模型会继续分裂、分数偏移。
    """
    value = getattr(model, "best_iteration", None)
    return int(value) if value else int(trained_rounds)


def _metrics(prefix: str, y: pd.Series, scores: np.ndarray) -> dict:
    """算一组指标。单一类别时跳过 —— 不能让 NaN 混进报表。"""
    y_arr = np.asarray(y, dtype=float).ravel()
    if (y_arr == 1).sum() == 0 or (y_arr == 0).sum() == 0:
        return {}
    out = {
        f"{prefix}_ks": calc_ks(y_arr, scores),
        f"{prefix}_auc": calc_auc(y_arr, scores),
        f"{prefix}_gini": calc_gini(y_arr, scores),
    }
    direction = score_direction(y_arr, scores)
    out[f"{prefix}_direction_is_high_risk"] = float(
        direction == "higher_score_higher_risk"
    )
    lift = calc_lift(y_arr, scores, direction=direction)
    out[f"{prefix}_lift_at_10"] = lift.lift
    return out


def _importance_frame(model: Any, features: list) -> pd.DataFrame:
    """特征重要性（gain / split 次数）。仅供报告展示。"""
    try:
        gain = model.feature_importance(importance_type="gain")
        split = model.feature_importance(importance_type="split")
    except (AttributeError, ValueError):      # pragma: no cover - 依赖库差异
        return pd.DataFrame(columns=["特征", "gain", "split次数", "gain占比"])
    total = float(np.sum(gain)) or 1.0
    return pd.DataFrame(
        {
            "特征": features,
            "gain": np.asarray(gain, dtype=float),
            "split次数": np.asarray(split, dtype=int),
            "gain占比": np.asarray(gain, dtype=float) / total,
        }
    ).sort_values("gain", ascending=False).reset_index(drop=True)
