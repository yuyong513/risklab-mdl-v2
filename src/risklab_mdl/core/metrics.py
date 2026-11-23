"""评分指标与分数量尺：AUC / KS / LIFT / ScoreScale。

★ LIFT 取尾部最差 10%，按权重加权，score 方向自适应。
★ ScoreScale 的 direction 是刻度的固定语义，不由 AUC 改写。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping, Sequence

import numpy as np

from .errors import DataError
from .utils import round_finite

__all__ = [
    "ClassMetrics",
    "Direction",
    "LiftResult",
    "MultiClassMetrics",
    "ScoreScale",
    "auc_significance",
    "calc_auc",
    "calc_auc_ovr",
    "calc_gini",
    "calc_ks",
    "calc_lift",
    "calc_multiclass_metrics",
    "score_direction",
]

EPS = 1e-12

Direction = Literal["higher_score_higher_risk", "higher_score_lower_risk"]


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _as_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    """转数组、剔除非有限值、校验好坏样本同时存在。"""
    y = np.asarray(y_true, dtype=float).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    if y.shape != s.shape:
        raise DataError(f"标签与分数长度不一致：{y.shape} vs {s.shape}")
    if np.isnan(y).any():
        raise DataError(
            "标签中存在缺失值，请在标签确认环节处理（单独留出 / 剔除）后再调用指标"
        )
    valid = np.isfinite(s)
    y, s = y[valid], s[valid]
    bad = int((y == 1).sum())
    good = int((y == 0).sum())
    if bad == 0 or good == 0:
        raise DataError(f"标签必须同时包含好/坏样本，当前 bad={bad}, good={good}")
    return y, s


def _average_rank(x: np.ndarray) -> np.ndarray:
    """返回 1-based **平均秩**，正确处理并列值（ties）。"""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(sorted_x):
        j = i
        while j + 1 < len(sorted_x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


# --------------------------------------------------------------------------- #
# AUC / GINI  —— 修复 F-01
# --------------------------------------------------------------------------- #
def calc_auc(y_true, y_score) -> float:
    """AUC（Mann-Whitney U / 平均秩法）。取值 ``[0, 1]``，越大越好。

    与 ``sklearn.metrics.roc_auc_score`` 数值一致（偏差 < 1e-12）。
    不做方向翻转；``AUC < 0.5`` 说明分数方向与标签相反。
    """
    y, s = _as_arrays(y_true, y_score)
    bad = float((y == 1).sum())
    good = float((y == 0).sum())
    ranks = _average_rank(s)
    return float((ranks[y == 1].sum() - bad * (bad + 1) / 2.0) / (bad * good))


def calc_gini(y_true, y_score) -> float:
    """GINI = 2 * AUC - 1，取值 ``[-1, 1]``。

    为负说明分数方向与标签相反（``calc_gini(-s) == -calc_gini(s)``）。
    """
    return 2.0 * calc_auc(y_true, y_score) - 1.0


def score_direction(y_true, y_score) -> Direction:
    """按 ``AUC >= 0.5`` 判定分数方向。**全项目方向判定的唯一入口。**"""
    return (
        "higher_score_higher_risk"
        if calc_auc(y_true, y_score) >= 0.5
        else "higher_score_lower_risk"
    )


def auc_significance(auc: float, n_bad: int, n_good: int) -> float:
    """AUC 与 0.5 的偏离的 **z 值**（Hanley-McNeil 标准误）。

    ★★ 为什么需要它
    ---------------
    「AUC < 0.5 ⇒ 模型学反了」这个判据**太粗**：

    * 小样本下 AUC 天然在 0.5 附近波动 —— 实测 30 坏 / 90 好时，
      AUC=0.4833 的 z 值只有 **0.27** ⇒ 纯属噪声，不是学反；
    * 若拿它告警 ⇒ 每个小样本模型都被误报「方向相反」，
      **告警本身就成了噪音**，真正学反的那次反而被淹没。

    ⇒ 必须问「这个偏离在统计上显著吗」。

    Returns
    -------
    z
        ``|AUC - 0.5| / SE(AUC)``；样本不足或 AUC 越界时返回 ``0.0``
        （⇒ 不显著，不告警）。
    """
    if n_bad <= 0 or n_good <= 0:
        return 0.0
    if not (0.0 < auc < 1.0):
        # AUC 恰为 0 或 1 ⇒ 方向完全反转，是极端情形 ⇒ 视为显著
        return float("inf")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (
        auc * (1.0 - auc)
        + (n_bad - 1) * (q1 - auc * auc)
        + (n_good - 1) * (q2 - auc * auc)
    ) / (n_bad * n_good)
    se = float(np.sqrt(max(var, 0.0)))
    if se <= 0:
        return 0.0
    return abs(auc - 0.5) / se


# --------------------------------------------------------------------------- #
# KS  —— 修复 F-02 / F-06
# --------------------------------------------------------------------------- #
def calc_ks(y_true, y_score) -> float:
    """KS = ``max |累计坏样本分布 - 累计好样本分布|``，取值 ``[0, 1]``。

    **取绝对值**，因此分数方向反转时仍返回正确量级（不会退回 0）。
    方向信息请用 :func:`score_direction` 获取。

    ★★ 并列值必须**整组推进**（F-06）—— 详见模块 docstring。
    """
    y, s = _as_arrays(y_true, y_score)
    bad = float((y == 1).sum())
    good = float((y == 0).sum())

    # ★ 按分数值去重后降序推进 —— 同分整组，不拆开
    levels, inverse = np.unique(s, return_inverse=True)
    order = np.argsort(-levels, kind="mergesort")
    rank = np.empty(levels.size, dtype=np.int64)
    rank[order] = np.arange(levels.size)

    bad_by_level = np.bincount(rank[inverse], weights=y, minlength=levels.size)
    good_by_level = np.bincount(
        rank[inverse], weights=1.0 - y, minlength=levels.size
    )

    cum_bad = np.cumsum(bad_by_level) / bad
    cum_good = np.cumsum(good_by_level) / good
    return float(np.abs(cum_bad - cum_good).max())


# --------------------------------------------------------------------------- #
# LIFT
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LiftResult:
    """LIFT 的结构化结果。

    业务定义：``LIFT = 段内坏样本率 / 整体坏样本率``，默认取「尾部最差 pct」
    的客户，按权重加权，分数方向自适应。
    """

    lift: float
    seg_bad_rate: float
    overall_bad_rate: float
    seg_share: float          # 段内权重占全量比例（无权重时即样本占比）
    seg_size: int             # 段内实际样本数
    direction: Direction
    is_worst_tail: bool
    is_reliable: bool         # 段内样本量是否达到 min_seg_size


def calc_lift(
    y_true,
    y_score,
    *,
    weight=None,
    pct: float = 0.10,
    worst: bool = True,
    direction: Direction | Literal["auto"] = "auto",
    min_seg_size: int = 30,
) -> LiftResult:
    """``LIFT@pct``。

    与 risklab-mcp 的差异（口径一致）：方向判定统一走 :func:`score_direction`；
    返回结构化结果，含方向、段内坏样本率、可靠性标记。

    ★ 段内样本量低于 ``min_seg_size`` 时**不做静默修正**，只把 ``is_reliable``
      置 ``False`` ⇒ 由报表显式提示 —— 防止小样本 LIFT 被当成有效结论。
    """
    if not 0.0 < pct <= 1.0:
        raise DataError(f"pct 必须落在 (0, 1] 区间，当前 {pct}")

    y_raw = np.asarray(y_true, dtype=float).ravel()
    s_raw = np.asarray(y_score, dtype=float).ravel()
    if y_raw.shape != s_raw.shape:
        raise DataError(f"标签与分数长度不一致：{y_raw.shape} vs {s_raw.shape}")

    if weight is None:
        w_raw = np.ones_like(s_raw)
    else:
        w_raw = np.asarray(weight, dtype=float).ravel()
        if w_raw.shape != s_raw.shape:
            raise DataError(f"weight 与样本长度不一致：{w_raw.shape} vs {s_raw.shape}")
        if np.isnan(w_raw).any():
            raise DataError("weight 中不允许出现缺失值")
        if (w_raw < 0).any():
            raise DataError("weight 中不允许出现负值")

    if np.isnan(y_raw).any():
        raise DataError("标签中存在缺失值，请在标签确认环节处理后再计算 LIFT")

    # 有限性掩码需要作用到三个数组上，长度必须一致
    valid = np.isfinite(s_raw) & np.isfinite(w_raw)
    y, s, w = y_raw[valid], s_raw[valid], w_raw[valid]

    bad = int((y == 1).sum())
    good = int((y == 0).sum())
    if bad == 0 or good == 0:
        raise DataError(f"标签必须同时包含好/坏样本，当前 bad={bad}, good={good}")

    d: Direction = score_direction(y, s) if direction == "auto" else direction
    risk = s if d == "higher_score_higher_risk" else -s
    order = np.argsort(-risk if worst else risk, kind="mergesort")
    y, s, w = y[order], s[order], w[order]

    total_w = float(w.sum())
    if total_w <= 0:
        raise DataError("权重合计为 0，无法计算 LIFT")

    # 取累计权重首次达到 pct 的位置，保证段内权重占比 >= pct
    cum_w = np.cumsum(w)
    n_take = int(np.searchsorted(cum_w, total_w * pct, side="left") + 1)
    n_take = max(1, min(n_take, y.size))

    seg_w = w[:n_take]
    seg_w_sum = float(seg_w.sum())
    seg_bad_rate = float((y[:n_take] * seg_w).sum() / seg_w_sum)
    overall_bad_rate = float((y * w).sum() / total_w)

    return LiftResult(
        lift=seg_bad_rate / overall_bad_rate if overall_bad_rate > 0 else float("nan"),
        seg_bad_rate=seg_bad_rate,
        overall_bad_rate=overall_bad_rate,
        seg_share=seg_w_sum / total_w,
        seg_size=n_take,
        direction=d,
        is_worst_tail=worst,
        is_reliable=n_take >= min_seg_size,
    )


# --------------------------------------------------------------------------- #
# 评分标尺（FICO 风格）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScoreScale:
    """FICO 风格评分标尺 —— **全项目唯一的概率↔分数转换**。

    ★ 实现来源
    -----------
    本包对应模块 的 ``ScoreScale``。
    口径说明：只把异常类型换成 risklab-mcp 自己的 ``DataError``。

    数学形式（先直觉后公式）
    ----------------------
    分数每增加 ``pdo`` 分，赔率（坏/好）翻一倍。因此分数与赔率的对数成线性::

        B = pdo / ln(2)
        A = base_point + B * ln(base_odds)
        score = A - B * ln(p / (1 - p))

    ``base_odds`` 是基准分处的赔率，默认 1（即基准分处好坏各半）。

    ★★ 为什么必须「唯一」而不是各报表自己换算
    ------------------------------------------
    同一概率换出两个分数，业务方拿到两个数无法解释，
    且**分数跨期不可比**（增量重打分时两边用的标尺不同）。
    ⇒ 标尺随模型包一起落盘，打分时用它换算，不重新猜。

    ★ 为什么标尺是**呈现层**、不能放在预测之前
    -------------------------------------------
    标尺只改变分数的刻度，不改变排序。若把标尺放在预测前，
    一旦标尺参数配错，模型的**排序**也会被扭曲 ——
    指标就再也不能反映模型的真实能力了。
    """

    base_point: float = 600.0
    base_odds: float = 1.0
    pdo: float = 40.0
    min_score: float = 300.0
    max_score: float = 1000.0
    direction: Direction = "higher_score_lower_risk"

    def __post_init__(self) -> None:
        if self.pdo <= 0:
            raise DataError(f"pdo 必须为正数，当前 {self.pdo}")
        if self.base_odds <= 0:
            raise DataError(f"base_odds 必须为正数，当前 {self.base_odds}")
        if self.min_score >= self.max_score:
            raise DataError(
                f"min_score({self.min_score}) 必须小于 max_score({self.max_score})"
            )

    @property
    def b(self) -> float:
        """比例系数 B = pdo / ln(2)。"""
        return self.pdo / np.log(2.0)

    @property
    def a(self) -> float:
        """截距 A = base_point + B * ln(base_odds)。"""
        return self.base_point + self.b * np.log(self.base_odds)

    def prob_to_score(self, prob) -> np.ndarray:
        """概率 → 分数。``prob`` 会被裁剪到 ``(0, 1)``。"""
        p = np.clip(np.asarray(prob, dtype=float), EPS, 1.0 - EPS)
        score = self.a - self.b * np.log(p / (1.0 - p))
        if self.direction == "higher_score_higher_risk":
            score = (self.min_score + self.max_score) - score
        return np.clip(score, self.min_score, self.max_score)

    def score_to_prob(self, score) -> np.ndarray:
        """分数 → 概率。与 :meth:`prob_to_score` 可逆（裁剪边界除外）。"""
        s = np.asarray(score, dtype=float)
        if self.direction == "higher_score_higher_risk":
            s = (self.min_score + self.max_score) - s
        odds = np.exp((self.a - s) / self.b)
        return odds / (1.0 + odds)

    def to_spec(self) -> dict:
        """序列化，随模型包一起落盘（增量重打分依赖此字段）。"""
        return {
            "base_point": self.base_point,
            "base_odds": self.base_odds,
            "pdo": self.pdo,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "direction": self.direction,
        }

    @classmethod
    def from_spec(cls, spec) -> "ScoreScale":
        """从 :meth:`to_spec` 的产物重建；缺失字段回落到默认值。"""
        allowed = ("base_point", "base_odds", "pdo", "min_score",
                   "max_score", "direction")
        return cls(**{k: spec[k] for k in allowed if k in spec})

    @classmethod
    def from_user_input(cls, value: Any, *, base: "ScoreScale | None" = None) -> "ScoreScale":
        """把「用户填的标尺取值」规整成 :class:`ScoreScale`。

        ★★ 为什么需要这个函数
        ----------------------
        「打分参数」弹窗 的 A/B 两个选项给用户的**形态不一样**：

        * A「展示默认参数」 ⇒ 执行层直接用模型包里已有的标尺；
        * B「在下方输入修改参数」 ⇒ 用户填的是**文本**（如 ``600,40,300,1000``）。

        而 :meth:`from_spec` 只接受**映射** ⇒ 走 B 路径时必然拿不到标尺，
        静默回落到默认值 ⇒ **用户填了但没生效，且全程不报错**
        （典型的「失败不可见」）。

        ⇒ 形态转换**收口在这里**：弹窗写什么、执行层怎么解析，只此一处定义。

        Parameters
        ----------
        value
            三种形态都接受：

            * ``None`` / 空 ⇒ 返回 ``base``（默认标尺）；
            * **映射** ⇒ 取其中已知键，缺失的回落到 ``base``；
            * **文本** ⇒ 按逗号或 ``｜`` 分隔，依次为
              基准分 / PDO / 分数下限 / 分数上限。
        base
            缺省时的回落标尺（通常是模型包里已有的那个）。

        ★ ``direction`` **不在用户可配范围**内
        --------------------------------------
        方向**不归用户拍板** —— 它必须与「模型实际输出的方向」保持一致，
        否则「分数高 = 风险低」这句话就是假的。

        ★★ 方向到底是**数据驱动**还是**固定默认**？（实测结论）
        ------------------------------------------------------
        * ``core.metrics.score_direction(y, scores)`` 确实是数据驱动的
          （按 AUC ≥ 0.5 判定）—— 但它算的是**指标层的方向**；
        * 而标尺的 ``direction`` 是**刻度层的固定默认值**
          （``higher_score_lower_risk``），训练时不会被 AUC 改写
          （见 ``core.train.build_train_config``：``score_scale`` 原样透传）。

        ⇒ 二者的**关系**是：标尺把概率 p 映射成分数 s，
          分数的 AUC 恒等于 ``1 - 概率的 AUC``（镜像），
          所以「分数方向」是**由标尺的 direction 直接决定**的，
          不是由 AUC 现场判定的。

        ⇒ 因此：
          - 用户**不能**拍方向（配反了 ⇒ 分数语义与模型相反，
            而 KS 取绝对值仍然「正常」，只有按分数分段看坏样本率时才显形）；
          - 但**模型学反了**（AUC < 0.5）这件事必须有人告警 ——
            由 ``exec.handlers.feature_model.feature_evaluate_model``
            产出告警（分数的 AUC < 0.5 ⇒ 分数语义与标尺矛盾）。
        """
        base = base or cls()
        if value is None:
            return base
        if isinstance(value, ScoreScale):
            return value
        if isinstance(value, Mapping):
            spec = value
        else:
            text = str(value).strip()
            if not text:
                return base
            parts = [p.strip() for p in text.replace("｜", ",").split(",")]
            parts = [p for p in parts if p != ""]
            if not parts:
                return base
            if len(parts) > 4:
                raise DataError(
                    f"评分标尺取值格式错误：{text!r}。"
                    f"需要 4 项（基准分 / PDO / 分数下限 / 分数上限），"
                    f"实际读到 {len(parts)} 项。"
                )
            keys = ("base_point", "pdo", "min_score", "max_score")
            spec = {}
            for key, raw in zip(keys, parts):
                try:
                    spec[key] = float(raw)
                except (TypeError, ValueError) as exc:
                    raise DataError(
                        f"评分标尺的 {key} 取值 {raw!r} 不是数字 —— "
                        f"请按「基准分,PDO,分数下限,分数上限」填写。"
                    ) from exc

        allowed = ("base_point", "base_odds", "pdo", "min_score", "max_score")
        kwargs: Dict[str, Any] = {}
        for key in allowed:
            # ★ 缺失/None ⇒ 回落到 base 的对应项（docstring 承诺的语义）
            #   漏了这一步 ⇒ 「用户只改了基准分」会把 PDO/区间悄悄重置成
            #   类默认值 —— 用户没碰过的项被改掉，且**不报错**。
            raw = spec.get(key)
            if raw is None:
                raw = getattr(base, key)
            kwargs[key] = float(raw)
        # ★ 方向永远沿用回落标尺（数据驱动，不让用户拍板）
        kwargs["direction"] = base.direction
        return cls(**kwargs)


# =========================================================================== #
# 多分类指标（OVR + macro）—— v2 新增
# =========================================================================== #
# 口径（正式说明见 docs/口径-metrics.md）：
#
#   对每个类别 k：把 y == k 视为「坏」，其余视为「好」⇒ 二分类 OVR 指标
#   AUC_k / KS_k / Gini_k / LIFT_k；macro = **各类的算术平均**（只统计有效类），
#   weighted = 按各类样本数加权的平均（供参考，不进主结论）。
#
# ★ 为什么是 OVR 而不是 multi-class AUC 的现成实现
#   风控落地看的是「每一类的区分度」（哪一类难分、哪一类靠运气），
#   单一 multiclass AUC 会把 3 类压成一个数，看不出是哪类拖后腿。
#   OVR + macro 与 sklearn 的 roc_auc_score(multi_class="ovr", average="macro")
#   同口径，可与外部复核对齐。
#
# ★ 样本量不足必须可见
#   某类在某子集里只有 1~2 个样本时，OVR 的 AUC/KS 是噪声。
#   本实现**不静默给个数**：该类的 ``status="skipped"`` 带原因，且
#   ``is_reliable=False``（样本数 < ``min_class_support``）；macro 只汇总有效类，
#   并在 ``notes`` 里写明「有几个类被跳过」。


@dataclass(frozen=True)
class ClassMetrics:
    """单个类别的 OVR 指标。"""

    label: Any
    support: int
    positives: int
    negatives: int
    auc: float
    ks: float
    gini: float
    lift: float
    lift_seg_size: int
    status: str
    """``"ok"`` 或 ``"skipped"``（无正例/无负例）。"""
    reason: str | None = None
    is_reliable: bool = True
    """该类样本数是否达到 ``min_class_support``（不足 ⇒ 指标不可靠，仅参考）。"""

    def to_dict(self) -> dict:
        return {
            "label": _native_label(self.label),
            "support": int(self.support),
            "positives": int(self.positives),
            "negatives": int(self.negatives),
            "auc": round_finite(self.auc, 8),
            "ks": round_finite(self.ks, 8),
            "gini": round_finite(self.gini, 8),
            "lift": round_finite(self.lift, 8),
            "lift_seg_size": int(self.lift_seg_size),
            "status": self.status,
            "reason": self.reason,
            "is_reliable": bool(self.is_reliable),
        }


@dataclass(frozen=True)
class MultiClassMetrics:
    """多分类 OVR + macro 汇总结果。"""

    per_class: tuple[ClassMetrics, ...]
    macro: dict
    weighted: dict
    labels: tuple
    n_samples: int
    n_valid_classes: int
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "task": "multiclass",
            "n_classes": len(self.labels),
            "n_valid_classes": int(self.n_valid_classes),
            "n_samples": int(self.n_samples),
            "labels": [_native_label(lab) for lab in self.labels],
            "per_class": [c.to_dict() for c in self.per_class],
            "macro": dict(self.macro),
            "weighted": dict(self.weighted),
            "notes": list(self.notes),
        }

    def frame(self):
        """逐类指标转 DataFrame（报表直接用）。"""
        import pandas as pd

        return pd.DataFrame([c.to_dict() for c in self.per_class])


def _native_label(label: Any) -> Any:
    """numpy 标量 → Python 原生（JSON 可序列化）。"""
    if isinstance(label, np.generic):
        return label.item()
    return label


def _as_score_matrix(y: np.ndarray, y_score, labels: Sequence) -> np.ndarray:
    """把打分统一成 ``(n, k)`` 概率矩阵；列顺序与 ``labels`` 一致。"""
    if isinstance(y_score, Mapping):
        cols = []
        for label in labels:
            if label not in y_score:
                raise DataError(
                    f"打分映射缺少类别 {label!r}（需要每个类别一列概率）"
                )
            cols.append(np.asarray(y_score[label], dtype=float).ravel())
        mat = np.column_stack(cols) if cols else np.empty((y.size, 0))
    else:
        mat = np.asarray(y_score, dtype=float)
        if mat.ndim == 1:
            raise DataError(
                "多分类打分需要 (n, k) 概率矩阵（或 {类别: 分数列} 映射），"
                f"当前是一维数组（长度 {mat.size}）"
            )
        if mat.ndim != 2:
            raise DataError(f"多分类打分维度错误：{mat.shape}")
    if mat.shape[0] != y.size:
        raise DataError(f"标签与打分行数不一致：{y.size} vs {mat.shape[0]}")
    if mat.shape[1] != len(labels):
        raise DataError(
            f"打分类别列数({mat.shape[1]}) 与类别数({len(labels)}) 不一致"
        )
    if not np.isfinite(mat).all():
        raise DataError("多分类打分中存在 NaN/Inf —— 请先检查模型输出")
    return mat


def calc_multiclass_metrics(
    y_true,
    y_score,
    *,
    labels: Sequence | None = None,
    lift_pct: float = 0.10,
    min_seg_size: int = 30,
    min_class_support: int = 30,
) -> MultiClassMetrics:
    """多分类 OVR + macro 指标（AUC / KS / Gini / LIFT）。

    Parameters
    ----------
    y_true
        真实标签（可为字符串/整数；不得含缺失）。
    y_score
        ``(n, k)`` 概率矩阵（列顺序与 ``labels`` 一致），或
        ``{label: 一维分数}`` 映射。OVR 对**每一类**只用该类对应那一列。
    labels
        类别顺序；缺省为 ``np.unique(y_true)``（升序）。
    lift_pct / min_seg_size
        透传给 :func:`calc_lift`（默认尾部最差 10%、段内最小 30 样本）。
    min_class_support
        类样本数低于该值时标记 ``is_reliable=False``（**不剔除**，仍给指标）。

    Returns
    -------
    MultiClassMetrics
        ``per_class`` 逐类指标 + ``macro``（有效类算术平均）+ ``weighted``（样本加权）。

    Raises
    ------
    DataError
        标签含缺失、行数不一致、类别数 < 2、打分含 NaN 等。

    Notes
    -----
    ★ 口径与 sklearn ``roc_auc_score(multi_class="ovr", average="macro")`` 一致：
      某类若无正例或无负例，**该类的 AUC 无定义** ⇒ 跳过并记入 ``notes``，
      不参与 macro（sklearn 的默认做法是抛错；此处选择显式跳过 + 告警，
      因为筛选阶段确实可能出现「某类在当前子集为 0」）。
    """
    yv = np.asarray(y_true).ravel()
    if yv.size == 0:
        raise DataError("标签为空，无法计算多分类指标")
    try:
        y_isnan = np.isnan(yv.astype(float))
    except (TypeError, ValueError):
        y_isnan = np.zeros(yv.size, dtype=bool)
    if y_isnan.any():
        raise DataError("标签中存在缺失值，请先处理标签后再计算多分类指标")

    label_list = list(labels) if labels is not None else list(np.unique(yv))
    if len(label_list) < 2:
        raise DataError(
            f"多分类指标至少需要 2 个类别，当前 {len(label_list)} 个：{label_list}"
        )
    mat = _as_score_matrix(yv, y_score, label_list)

    notes: list[str] = []
    per_class: list[ClassMetrics] = []
    skipped: list[str] = []

    for col, label in enumerate(label_list):
        y_bin = (yv == label).astype(float)
        pos = int(y_bin.sum())
        neg = int(y_bin.size - pos)
        support = pos
        scores = mat[:, col]
        if pos == 0 or neg == 0:
            reason = (
                "该类在当前样本中无正例" if pos == 0 else "该类在当前样本中无负例"
            )
            skipped.append(f"{_native_label(label)}({reason})")
            per_class.append(
                ClassMetrics(
                    label=label,
                    support=support,
                    positives=pos,
                    negatives=neg,
                    auc=float("nan"),
                    ks=float("nan"),
                    gini=float("nan"),
                    lift=float("nan"),
                    lift_seg_size=0,
                    status="skipped",
                    reason=reason,
                    is_reliable=False,
                )
            )
            continue

        auc = calc_auc(y_bin, scores)
        ks = calc_ks(y_bin, scores)
        gini = calc_gini(y_bin, scores)
        lift = calc_lift(
            y_bin, scores, pct=lift_pct, min_seg_size=min_seg_size
        )
        reliable = support >= int(min_class_support)
        if not reliable:
            notes.append(
                f"类别 {_native_label(label)} 仅 {support} 个样本"
                f"（下限 {min_class_support}）⇒ 该类指标不可靠，仅供参照"
            )
        per_class.append(
            ClassMetrics(
                label=label,
                support=support,
                positives=pos,
                negatives=neg,
                auc=float(auc),
                ks=float(ks),
                gini=float(gini),
                lift=float(lift.lift),
                lift_seg_size=int(lift.seg_size),
                status="ok",
                is_reliable=reliable,
            )
        )

    valid = [c for c in per_class if c.status == "ok"]
    macro: dict[str, float] = {}
    weighted: dict[str, float] = {}
    for key in ("auc", "ks", "gini", "lift"):
        values = [getattr(c, key) for c in valid]
        finite = [v for v in values if v is not None and np.isfinite(v)]
        macro[key] = float(np.mean(finite)) if finite else float("nan")
        if finite:
            total = sum(c.support for c in valid)
            weighted[key] = (
                float(
                    sum(
                        getattr(c, key) * c.support
                        for c in valid
                        if getattr(c, key) is not None
                        and np.isfinite(getattr(c, key))
                    )
                    / total
                )
                if total
                else float("nan")
            )
        else:
            weighted[key] = float("nan")

    if skipped:
        notes.append(
            f"{len(skipped)} 个类别因无正例/负例被跳过（不计入 macro）："
            + "、".join(skipped)
        )
    if not valid:
        notes.append("没有任何类别可计算 OVR 指标 ⇒ macro 为空（返回 NaN）")

    macro_out = {k: round_finite(v, 8) for k, v in macro.items()}
    weighted_out = {k: round_finite(v, 8) for k, v in weighted.items()}
    return MultiClassMetrics(
        per_class=tuple(per_class),
        macro=macro_out,
        weighted=weighted_out,
        labels=tuple(label_list),
        n_samples=int(yv.size),
        n_valid_classes=len(valid),
        notes=tuple(notes),
    )


def calc_auc_ovr(y_true, y_score, *, labels: Sequence | None = None) -> dict:
    """便捷入口：只返回逐类 AUC 与 macro AUC（其余指标见 :func:`calc_multiclass_metrics`）。"""
    result = calc_multiclass_metrics(y_true, y_score, labels=labels)
    return {
        "per_class": {
            _native_label(c.label): round_finite(c.auc, 8) for c in result.per_class
        },
        "macro": result.macro["auc"],
        "weighted": result.weighted["auc"],
    }
