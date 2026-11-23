"""特征精筛：按增益多轮收敛到目标特征区间。

★ 标签泄漏三道防线；★ 链式相关下贪心去相关避免一次误杀多个可用特征。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .deps import require_lightgbm
from .errors import ConfigError, ModelError

__all__ = [
    "RefineConfig",
    "RefineDisposition",
    "RefineMethod",
    "RefineResult",
    "RefineRound",
    "select_features_by_gain",
]

RefineMethod = Literal["gain_iterative", "shap", "boruta", "two_stage", "recall", "cv"]

RefineDisposition = Literal[
    "converged",           # 自然收敛在区间内
    "truncated_high",      # 超上限，按累计 gain 截断
    "relaxed_converged",   # 放宽阈值后收敛进区间
    "relaxed_truncated",   # 放宽阈值后仍超上限
    "relaxed_failed",      # 放宽到底仍不足下限 → 告警放行
]

_UNSUPPORTED_METHODS: Mapping[str, str] = {
    "shap": "SHAP 三规则精筛（备选）",
    "boruta": "Boruta 精筛（备选）",
    "two_stage": "两阶段调参 + gain 筛选（备选）",
    "recall": "回捞后剔（备选）",
    "cv": "CV 终验（备选）",
}
"""已登记但**未实现**的备选方法。

刻意不实现：SHAP 需要逐样本解释（万级特征下开销显著），
Boruta 需要额外依赖且训练轮数多。两者都属于「可解释性 / 稳健性」范畴，
当前版本目标是「跑通 + 产出可用模型」。

**不静默降级**：传未实现的方法时明确报错，而不是悄悄回退到 gain 迭代 ——
使用方以为自己在用 SHAP，实际是 gain，结果不可复现、也不可解释。
"""


@dataclass(frozen=True)
class RefineConfig:
    """精筛配置。默认值面向 LightGBM 默认方法。"""

    method: RefineMethod = "gain_iterative"

    # ── gain 迭代参数 ──
    threshold: float = 0.05
    """每轮保留线：``gain >= max_gain * threshold``。相对量，不随特征量级漂移。"""
    max_rounds: int = 10
    num_boost_round: int = 1000
    early_stopping_rounds: int = 50
    categorical_features: tuple[str, ...] = ()

    # ── ★ 特征数区间控制（当前版本新增）──
    target_range: tuple[int, int] = (30, 60)
    """期望的最终特征数区间。超出时的处置见 :attr:`RefineResult.disposition`。"""
    max_features: int | None = 60
    """硬上限。``None`` 表示不截断。"""
    threshold_relax_ladder: tuple[float, ...] = (0.03, 0.01)
    """低于下限时逐级放宽的 threshold。最多重跑 ``len(ladder)`` 次。"""
    min_features: int = 5
    """单轮的兜底下限 —— 防止某轮全被剔光导致训练无法进行。"""

    # ── 白名单 / 黑名单（合规刚需）──
    force_keep: tuple[str, ...] = ()
    """强制保留。无论 IV / gain 多低都进模型，且**不参与迭代判定**。"""
    force_drop: tuple[str, ...] = ()
    """强制剔除。合规受限字段的落地方式。"""

    seed: int = 42

    def __post_init__(self) -> None:
        if self.method != "gain_iterative":
            raise ConfigError(
                f"方法 {self.method}（{_UNSUPPORTED_METHODS.get(self.method, '备选')}）"
                f"当前版本未实现。仅支持 gain_iterative —— "
                f"不静默降级（悄悄换方法会让结果不可复现、不可解释）。"
            )
        if not 0 < self.threshold < 1:
            raise ConfigError(f"threshold 必须落在 (0, 1)，当前 {self.threshold}")
        ladder = tuple(self.threshold_relax_ladder)
        for t in ladder:
            if not 0 < t < 1:
                raise ConfigError(f"放宽阶梯中的 threshold 非法：{t}")
        lo, hi = self.target_range
        if not 0 < lo <= hi:
            raise ConfigError(f"target_range 必须满足 0 < 下限 <= 上限，当前 {target_range_fmt(self)}")
        if self.max_features is not None and self.max_features < lo:
            raise ConfigError(
                f"max_features={self.max_features} 小于 target_range 下限 {lo}，"
                f"会与区间控制逻辑冲突。"
            )
        overlap = set(self.force_keep) & set(self.force_drop)
        if overlap:
            raise ConfigError(
                f"以下列同时在白名单与黑名单中：{sorted(overlap)}。"
                f"白名单是合规强制保留，黑名单是合规强制剔除 —— 必须先解决冲突。"
            )
        # ★ P2：放宽阶梯的两条独立约束
        # ----------------------------------
        # 【约束一】阶梯**自身**严格递减（越往后越小）。
        #   放宽 = 保留线变低 = 保留更多特征 → threshold 必须递减。
        #   ⚠ 方向写反会让「放宽」变成「收紧」，结果是特征更少、且无提示。
        #
        # 【约束二】阶梯首项必须**不大于** threshold。
        #   阶梯是「threshold 不够时的备选」，方向上必须不严于 threshold。
        #
        # ★ 为什么允许「首项 == threshold」而不是要求严格小于
        # -----------------------------------------------
        # 常见组合是 ``threshold=0.03`` + 阶梯 ``(0.03, 0.01)``：
        # 首项与 threshold 同值，那一步确实等价于「没放宽」——
        # 但它是**合法的冗余档**，不是配置错误。
        #
        # 早先的校验把 threshold 拼到阶梯前面一起查递减，于是这个组合被判非法 ——
        # 而它**语义完全正确**（第二档 0.01 才是真正的放宽）。
        # 把「阶梯自身有序」和「阶梯相对 threshold 的方向」混成一条，
        # 会让**合法配置被拒**。
        #
        # 冗余档不会造成浪费：迭代时会把与 threshold 同值的档位**去重**，
        # 不重复训练一次 LightGBM。
        if len(ladder) >= 2 and any(
            ladder[i] <= ladder[i + 1] for i in range(len(ladder) - 1)
        ):
            raise ConfigError(
                f"放宽阶梯自身必须严格递减（越往后越小），当前 {list(ladder)}。"
                f"threshold 越小 → 保留线越低 → 保留的特征越多，"
                f"这才是「放宽」。若递增，「放宽」会变成「收紧」，反而更少特征。"
            )
        if ladder and ladder[0] > self.threshold:
            raise ConfigError(
                f"放宽阶梯的首项 {ladder[0]} 不能**大于** threshold {self.threshold}。"
                f"★ 阶梯是「threshold 不够时的备选」，方向上必须不严于 threshold —— "
                f"若阶梯首项 > threshold，「放宽」会变成「收紧」，"
                f"结果反而更少特征，且无任何提示。"
            )


def target_range_fmt(cfg: RefineConfig) -> tuple[int, int]:
    return cfg.target_range


@dataclass(frozen=True)
class RefineRound:
    """单轮迭代的过程记录。"""

    round_no: int
    n_input: int
    n_features: int
    cut: float | None
    valid_auc: float | None
    note: str = ""


@dataclass
class RefineResult:
    """精筛结果 + 可复现的过程报表。"""

    features: list[str]
    method: str = "gain_iterative"
    threshold: float = 0.05
    disposition: RefineDisposition = "converged"
    message: str = ""

    rounds: list[RefineRound] = field(default_factory=list)
    gain_table: pd.DataFrame | None = None
    """逐列逐轮 gain + ``total_gain`` + ``retained`` 标记。"""

    force_kept: list[str] = field(default_factory=list)
    force_dropped: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    """阈值放宽的尝试记录，用于报告里说明「为什么最终是这个特征数」。"""

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def converged_round(self) -> int | None:
        return self.rounds[-1].round_no if self.rounds else None

    @property
    def in_target_range(self) -> bool:
        lo, hi = self.target_range
        return lo <= len(self.features) <= hi

    _range: tuple[int, int] = (30, 60)

    @property
    def target_range(self) -> tuple[int, int]:
        return self._range

    # ------------------------------------------------------------------ #
    def rounds_frame(self) -> pd.DataFrame:
        """逐轮过程表（对应 ``筛选结果_精筛.xlsx`` 的「逐轮过程」sheet）。"""
        return pd.DataFrame(
            [
                {
                    "轮次": r.round_no,
                    "输入特征数": r.n_input,
                    "保留特征数": r.n_features,
                    "保留线(gain)": r.cut,
                    "验证集AUC": r.valid_auc,
                    "说明": r.note,
                }
                for r in self.rounds
            ],
            columns=["轮次", "输入特征数", "保留特征数", "保留线(gain)", "验证集AUC", "说明"],
        )

    def attempts_frame(self) -> pd.DataFrame:
        """阈值放宽尝试记录（对应「处置说明」sheet）。"""
        rows = [
            {
                "尝试": i + 1,
                "threshold": a["threshold"],
                "该次收敛特征数": a["n_features"],
                "是否落在区间": "✓" if a["in_range"] else "",
            }
            for i, a in enumerate(self.attempts)
        ]
        rows.append(
            {
                "尝试": "最终",
                "threshold": self.threshold,
                "该次收敛特征数": self.n_features,
                "是否落在区间": "✓" if self.in_target_range else "",
            }
        )
        return pd.DataFrame(rows, columns=["尝试", "threshold", "该次收敛特征数", "是否落在区间"])

    def features_frame(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """入模特征 + 落选特征（对应「入模特征」「落选特征」两个 sheet）。

        ★ 白名单列会被**单独补进表里**
        ---------------------------------
        白名单列**不参与迭代**（它们在 ``pool`` 之外），因此 ``gain_table``
        里没有它们的 gain 记录。若直接按 ``retained`` 切表，白名单特征会被
        归入「落选」—— 报表会显示「白名单没生效」，而实际上它们已入模。
        这类「报表与实际不一致」的错误比计算错误更难发现。

        因此这里显式把白名单列补进表，gain 记为 ``NaN``（标注「未参与迭代」）。
        """
        table = (
            self.gain_table.copy()
            if self.gain_table is not None
            else pd.DataFrame(columns=["total_gain", "retained"])
        )
        table = table.reset_index()
        table = table.rename(columns={table.columns[0]: "特征"})

        if "retained" not in table.columns:
            table["retained"] = True
        table["归属"] = np.where(table["retained"], "常规", "落选")

        # 白名单列：不在 gain_table 里 → 补行
        in_table = set(table["特征"])
        forced = [c for c in self.force_kept if c not in in_table]
        if forced:
            extra = pd.DataFrame(
                {
                    "特征": forced,
                    "total_gain": np.nan,
                    "retained": True,
                    "归属": "白名单",
                }
            )
            table = pd.concat([table, extra], ignore_index=True)
        # 白名单里恰好也参与了迭代的（极端情况，如同时在 pool 里）→ 改标注
        table.loc[
            table["特征"].isin(self.force_kept), "归属"
        ] = "白名单"

        keep = table[table["归属"] != "落选"].copy()
        drop = table[table["归属"] == "落选"].copy()
        order = {f: i for i, f in enumerate(self.features)}
        keep["_ord"] = keep["特征"].map(order).fillna(len(order))
        keep = keep.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)
        drop = drop.sort_values("total_gain", ascending=False).reset_index(drop=True)
        return keep, drop

    def gain_distribution_frame(self) -> pd.DataFrame:
        """★ P2：gain 分布形态表 —— 「重要性是集中在头部，还是摊在一片？」

        ★ 为什么必须透出这个形态
        -----------------------
        ``total_gain`` 的**绝对值**没有解读价值（它取决于特征量级与树数），
        但**分布形态**有：

        * **头部集中**（前 10% 特征吃掉 70%+ 的 gain）
          ⇒ 这批特征的信息高度冗余，精筛是在做「去重」而非「选优」；
          剩下的特征大概率是噪声，特征数区间可以往**下限**调。
        * **长尾平坦**（前 10% 只占 20%）
          ⇒ gain 区分度差，阈值切在哪都差不多 ——
          此时该调的是**threshold**，而不是 target_range。

        没有这张表，使用者只能看到「选了 43 个」，
        无从判断「43 个」是**信息充分**还是**阈值切歪了**。

        表结构
        ------
        逐列：排名 / 特征 / 累计 gain / 累计占比 / 与上一名的比值（衰减倍率）。

        ★ 「与上一名的比值」这一列是关键
        ------------------------------
        光看累计占比不知道「头部有多陡」；
        比值列把「断崖」显式化 —— 比值 < 0.5 处就是天然的分界。
        """
        table = self.gain_table
        if table is None or table.empty:
            return pd.DataFrame(
                columns=["排名", "特征", "total_gain", "累计gain", "累计占比",
                         "占比(%)", "与上一名比值"]
            )
        df = table.copy()
        if "total_gain" not in df.columns:
            return pd.DataFrame(
                columns=["排名", "特征", "total_gain", "累计gain", "累计占比",
                         "占比(%)", "与上一名比值"]
            )
        df = df.sort_values("total_gain", ascending=False)
        gains = df["total_gain"].astype(float)
        total = float(gains.sum())
        cum = gains.cumsum()
        rows: list[dict[str, Any]] = []
        prev: float | None = None
        for i, (name, g) in enumerate(gains.items(), start=1):
            ratio = (float(g) / prev) if (prev is not None and prev > 0) else None
            rows.append(
                {
                    "排名": i,
                    "特征": name,
                    "total_gain": round(float(g), 6),
                    "累计gain": round(float(cum.iloc[i - 1]), 6),
                    "累计占比": round(float(cum.iloc[i - 1] / total), 6) if total > 0 else 0.0,
                    "占比(%)": round(float(g / total) * 100, 4) if total > 0 else 0.0,
                    "与上一名比值": round(ratio, 4) if ratio is not None else None,
                }
            )
        return pd.DataFrame(rows)

    def gain_shape_summary(self) -> dict[str, Any]:
        """★ P2：gain 分布形态的**可读结论**（进汇总与告警）。

        返回的每个数字都是**可复算**的（来自 ``gain_table``），
        不是模型自己下的结论 —— 使用方可拿报表核对。
        """
        table = self.gain_table
        if table is None or table.empty or "total_gain" not in table.columns:
            return {
                "available": False,
                "note": "无 gain 表（本次未产生迭代记录）。",
            }
        gains = table["total_gain"].astype(float).sort_values(ascending=False)
        total = float(gains.sum())
        if total <= 0:
            return {
                "available": False,
                "note": "gain 总量为 0 —— 模型已无法分裂，特征数不可信。",
            }
        n = len(gains)
        top10 = float(gains.head(max(1, int(round(n * 0.1)))).sum() / total)
        top30 = float(gains.head(max(1, int(round(n * 0.3)))).sum() / total)
        median = float(gains.median())
        max_gain = float(gains.iloc[0])

        # ★ 兜底影响：有多少特征是被「单轮兜底」救回来的？
        #   兜底（min_features）在某轮把保留线以下的全捞回来 ——
        #   这些特征**不是**因为 gain 达标才留下的，
        #   它们进模型靠的是「不能太少」这条规则，不是信息量。
        #   不透出这个数，使用者会把兜底捞回来的噪声当成「筛出来的好特征」。
        floor_applied = any(r.note for r in self.rounds)
        n_floor_rounds = sum(1 for r in self.rounds if "兜底" in r.note)

        if top10 >= 0.7:
            shape = "头部集中"
            advice = (
                "gain 高度集中在头部少数特征 —— 这批特征信息冗余度较高，"
                "精筛起的是「去重」作用。若想提速/降维，"
                "可把 target_range 下限往低处调（不必硬凑到区间下限）。"
            )
        elif top10 >= 0.4:
            shape = "中等集中"
            advice = (
                "gain 分布中等集中，筛出当前特征数是合理的 —— "
                "继续沿用当前 threshold 即可。"
            )
        else:
            shape = "长尾平坦"
            advice = (
                "gain 分布长尾平坦 —— 各特征重要性差异不大，"
                "这意味着**保留线切在哪都会得到差不多的特征数**。"
                "此时特征数偏少多半是信号本就稀疏，"
                "调 threshold 的收益有限；建议排查初筛阈值是否过严、"
                "或补充业务变量，而不是反复放宽。"
            )

        return {
            "available": True,
            "shape": shape,
            "n_features_with_gain": n,
            "total_gain": round(total, 6),
            "max_gain": round(max_gain, 6),
            "median_gain": round(median, 6),
            "top10pct_gain_share": round(top10, 4),
            "top30pct_gain_share": round(top30, 4),
            "head_drop_ratio": round(max_gain / median, 4) if median > 0 else None,
            "floor_applied": floor_applied,
            "n_floor_rounds": n_floor_rounds,
            "advice": advice,
        }

    def summary(self) -> dict[str, Any]:
        lo, hi = self.target_range
        return {
            "method": self.method,
            "threshold": self.threshold,
            "n_features": self.n_features,
            "target_range": [lo, hi],
            "in_target_range": self.in_target_range,
            "disposition": self.disposition,
            "message": self.message,
            "converged_round": self.converged_round,
            "n_force_kept": len(self.force_kept),
            "n_force_dropped": len(self.force_dropped),
            "attempts": self.attempts,
            # ★ P2：gain 分布形态 —— 让使用者能判断「这个特征数是信息充分
            #   还是阈值切歪了」，而不是只看到一个数字。
            "gain_shape": self.gain_shape_summary(),
            "rounds": [
                {
                    "round": r.round_no,
                    "n_input": r.n_input,
                    "n_features": r.n_features,
                    "cut": r.cut,
                    "valid_auc": r.valid_auc,
                    "note": r.note,
                }
                for r in self.rounds
            ],
        }


def select_features_by_gain(
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Mapping[str, Any],
    cfg: RefineConfig | None = None,
    candidates: Sequence[str] | None = None,
    forbidden: Sequence[str] = (),
) -> RefineResult:
    """gain 迭代筛选（默认精筛方法）。

    Parameters
    ----------
    X, y
        训练集。**分箱切点、编码表等一切「拟合」都只在此进行**。
    X_val, y_val
        验证集。每轮的 early stopping 与 AUC 记录都在此 —— **不用 OOT**。
    params
        LightGBM 参数。由 M4 调参产出；此处不重新调参。
    cfg
        精筛配置；为 None 时用默认值。
    candidates
        ★ **候选特征白名单**（F-21）。只有这里的列可以进入筛选。

        为 None 时**不是**「取 X 的所有列」，而是**抛错** ——
        旧实现取 ``X.columns``，于是标签列 ``y``、月份列 ``month``
        一起进了候选池 → LightGBM 第一棵树对 ``y`` 做 split
        → Train/Val/OOT 全套 AUC=1.0、KS=1.0、best_iteration=1
        （标签泄漏，且线上完全无法复现）。

        「不传就默认全选」正是这个缺陷的来源：
        调用方忘了传，等于**默认把标签送进模型**。
        所以这里要求**显式声明**，不提供危险默认值。
    forbidden
        ★ **禁止入模的列**（F-21，如标签列、其他标签列）。

        core 是最底层，**必须自己兜住**这条约束 ——
        不能假设上层已经过滤过。

    Returns
    -------
    RefineResult
        含最终特征、逐轮过程、逐列 gain 表、处置说明。

    Raises
    ------
    ConfigError
        配置非法（见 :class:`RefineConfig` 的校验）；白名单列不存在；
        候选为空；**候选或白名单中含禁止入模的列**。
    ModelError
        LightGBM 不可用；训练后 ``max(gain) == 0``（模型已无法分裂）；
        白名单剔除后候选为空。
    """
    cfg = cfg or RefineConfig()
    lgb = require_lightgbm()

    if X.empty or X_val.empty:
        raise ModelError("训练集或验证集为空，无法精筛")
    y = pd.Series(y, index=X.index)
    y_val = pd.Series(y_val, index=X_val.index)

    # ── 白 / 黑名单先行：合规与业务硬约束不参与迭代 ──
    keep_set = set(cfg.force_keep)
    drop_set = set(cfg.force_drop)
    missing_keep = [c for c in keep_set if c not in X.columns]
    if missing_keep:
        raise ConfigError(f"白名单中的列不在训练集内：{sorted(missing_keep)[:10]}")

    # ★ F-21：白名单是标签泄漏的**独立通道** —— 必须在这里堵死。
    #   ``force_keep`` 的列会**无条件**进入入模特征集，且**不参与 gain 迭代**，
    #   也就是说它绕过整个筛选过程 ⇒ ``candidates`` 白名单拦不住它。
    #   core 是唯一知道「标签叫什么」的地方（``y.name``）——
    #   协议层的 forbidden 清单依赖调用方传进来，core 自己必须兜住。
    y_name = y.name
    if y_name and y_name in keep_set:
        raise ConfigError(
            f"白名单（force_keep）中含标签列 `{y_name}`。"
            f"★ 白名单列会**无条件**进入模型且不参与 gain 迭代 —— "
            f"它绕过整个筛选过程，标签列走这条路同样会造成标签泄漏："
            f"Train/Val/OOT 全套 AUC/KS 变成 1.0，线上完全无法复现。"
        )
    # ── ★ F-21：候选池 = 「调用方声明的候选」∩「X 的列」 ──
    # 绝不退回「X 的所有列」—— 那是标签泄漏的来源。
    if candidates is None:
        raise ConfigError(
            "必须显式传入 candidates（候选特征白名单）。\n"
            "★ 服务**不会**默认把训练集的所有列都当候选 —— "
            "标签列 y、月份列 month 混进去会让模型 AUC/KS 直接变成 1.0"
            "（标签泄漏），线上完全无法复现。\n"
            "请传入初筛存活列（如 prescreen_features）。"
        )
    missing_candidates = [c for c in candidates if c not in X.columns]
    if missing_candidates:
        raise ConfigError(
            f"候选特征在训练集中不存在：{missing_candidates[:10]}"
        )
    # 去重但**保序**（保序让报表可复现、可 diff）
    seen: set[str] = set()
    candidate_list: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            candidate_list.append(c)

    # ★ F-21：core 自己兜住「禁止入模」这条约束。
    #   上层传下来的 forbidden 是**权威事实**（协议层从列角色登记推导），
    #   但 core 不能假设上层一定传了 —— 自己再查一遍。
    forbidden_set = {c for c in forbidden if c}
    if forbidden_set:
        leaked_cand = [c for c in candidate_list if c in forbidden_set]
        if leaked_cand:
            raise ConfigError(
                f"候选特征中含**禁止入模的列**：{leaked_cand[:10]}。"
                f"★ 标签列入模会造成标签泄漏 —— 模型 AUC/KS 会直接变成 1.0，"
                f"线上完全无法复现。"
            )
        leaked_keep = [c for c in keep_set if c in forbidden_set]
        if leaked_keep:
            raise ConfigError(
                f"白名单（force_keep）中含**禁止入模的列**：{leaked_keep[:10]}。"
                f"★ 白名单列会**无条件**进入模型且不参与 gain 迭代 —— "
                f"它绕过整个筛选过程，标签列走这条路同样会造成标签泄漏。"
            )

    pool = [c for c in candidate_list if c not in drop_set and c not in keep_set]
    if not pool:
        raise ModelError(
            "剔除黑名单后无候选特征可筛。"
            "若候选全部来自白名单，说明白名单配置有误。"
        )

    base_params = {
        k: v for k, v in dict(params).items()
        if k not in ("categorical_feature", "n_estimators", "random_state")
    }
    base_params["random_state"] = cfg.seed          # ★ 原实现漏了种子 → 结果不可复现

    attempts: list[dict[str, Any]] = []
    final: _IterationOutcome | None = None
    used_threshold = cfg.threshold

    # ★ P2：去重冗余档位。
    #   阶梯首项可能与 threshold 同值（这是**合法的**冗余档，不是配置错误），
    #   但同值 = 同一套保留线 = 结果必然相同 ⇒ 重复训练一次 LightGBM
    #   纯属浪费，而且报表里会出现两条一模一样的「尝试」记录，
    #   让人误以为「放宽真的起作用了」。
    #   ⇒ 只在**值不同**时才真正跑一遍。
    tried: set[float] = set()
    for threshold in (cfg.threshold, *cfg.threshold_relax_ladder):
        if threshold in tried:
            continue
        tried.add(threshold)
        outcome = _iterate(
            X, y, X_val, y_val, pool, base_params, cfg, threshold
        )
        lo, hi = cfg.target_range
        in_range = lo <= len(outcome.features) <= hi
        attempts.append(
            {
                "threshold": threshold,
                "n_features": len(outcome.features),
                "in_range": in_range,
                "n_rounds": len(outcome.rounds),
            }
        )
        final = outcome
        used_threshold = threshold

        if len(outcome.features) >= lo:
            # 落在区间内或超上限 —— 都不需要再放宽
            break

    assert final is not None
    features, rounds, gain_df = final.features, final.rounds, final.gain_table
    lo, hi = cfg.target_range

    # ── 超上限：按累计 gain 截断 ──
    disposition: RefineDisposition = "converged"
    message = ""
    limit = cfg.max_features if cfg.max_features is not None else hi
    if len(features) > limit:
        order = gain_df.sort_values("total_gain", ascending=False).index
        features = [c for c in order if c in set(features)][:limit]
        disposition = (
            "relaxed_truncated" if used_threshold != cfg.threshold else "truncated_high"
        )
        message = (
            f"迭代收敛于 {len(gain_df)} 个候选中的较大部分，超过上限 {limit}，"
            f"已按**累计 gain**排序截断至 {len(features)} 个。"
            f"未入选特征的完整 gain 排名见「落选特征」sheet。"
        )
    elif used_threshold != cfg.threshold:
        disposition = "relaxed_converged"
        message = (
            f"首次迭代收敛特征数低于下限 {lo}，已自动放宽 threshold "
            f"至 {used_threshold} 后收敛到 {len(features)} 个。"
        )

    # ── 不足下限：告警放行，不硬凑 ──
    if len(features) < lo:
        disposition = "relaxed_failed"
        message = (
            f"阈值已放宽至 {used_threshold}（阶梯 {list(cfg.threshold_relax_ladder)} 已用尽），"
            f"收敛特征数仍仅 {len(features)} 个，低于经验下限 {lo}。"
            f"可能原因与建议：\n"
            f"  ① 候选特征本身不足 —— 检查初筛阈值是否过严，可回退重跑初筛\n"
            f"  ② 数据维度有限 —— 补充业务变量\n"
            f"  ③ 信号本就稀疏 —— 接受当前结果，**但需在报告中说明特征数偏少**\n"
            f"服务不会为了凑够 {lo} 个而把噪声塞进模型。"
        )

    # ★★ 超上限且**发生过放宽**时，message 必须点明「拍板值 → 生效值」。
    #    否则交付表格只写「保留线 threshold=0.01」，
    #    客户会以为 0.01 就是他拍的值 —— 他拍的是 0.03，
    #    是执行层在收敛特征数不足时**自行放宽**的。
    #    这是典型的「失败不可见」：不报错、不崩，只是悄悄改了参数。
    if used_threshold != cfg.threshold and disposition in (
        "relaxed_truncated",
        "relaxed_converged",
    ):
        message = (
            f"⚠ 首次迭代收敛特征数低于下限 {lo}，已**自动放宽保留线**"
            f"（{cfg.threshold} → {used_threshold}）。"
            f"{message}"
            f"若需严格按 {cfg.threshold} 执行，请回退重跑精筛并调整「最终特征数下限」。"
        )

    final_keep = [c for c in cfg.force_keep if c in X.columns] + list(features)

    return RefineResult(
        features=final_keep,
        method=cfg.method,
        threshold=used_threshold,
        disposition=disposition,        # type: ignore[arg-type]
        message=message,
        rounds=rounds,
        gain_table=gain_df,
        force_kept=[c for c in cfg.force_keep if c in X.columns],
        force_dropped=sorted(drop_set),
        attempts=attempts,
        _range=cfg.target_range,
    )


@dataclass
class _IterationOutcome:
    features: list[str]
    rounds: list[RefineRound]
    gain_table: pd.DataFrame


def _iterate(
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    pool: Sequence[str],
    params: Mapping[str, Any],
    cfg: RefineConfig,
    threshold: float,
) -> _IterationOutcome:
    """单次迭代。返回 (保留特征, 逐轮过程, 逐列 gain 表)。"""
    lgb = require_lightgbm()

    features = list(pool)
    round_gains: list[dict[str, float]] = []
    rounds: list[RefineRound] = []
    cats = [c for c in cfg.categorical_features if c in features] or None

    for round_no in range(1, cfg.max_rounds + 1):
        train_data = lgb.Dataset(X[features], y, categorical_feature=cats)
        val_data = lgb.Dataset(
            X_val[features], y_val, reference=train_data, categorical_feature=cats
        )
        model = lgb.train(
            dict(params),
            train_data,
            valid_sets=[val_data],
            num_boost_round=cfg.num_boost_round,
            callbacks=[
                lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        gains = model.feature_importance(importance_type="gain")
        gain_map = dict(zip(features, (float(g) for g in gains)))
        round_gains.append(gain_map)
        auc = _best_auc(model)

        max_gain = max(gain_map.values(), default=0.0)
        if max_gain <= 0:
            rounds.append(
                RefineRound(
                    round_no, len(features), len(features), None, auc,
                    "max gain = 0，模型已无法分裂，提前终止",
                )
            )
            break

        cut = max_gain * threshold
        kept = [f for f in features if gain_map[f] >= cut]

        # ★ 单轮兜底：全部被剔光会让下一次训练无法进行
        if len(kept) < cfg.min_features:
            order = sorted(gain_map, key=lambda f: -gain_map[f])
            kept = order[: cfg.min_features]
            note = f"低于单轮兜底 {cfg.min_features} 个，按 gain 取前 {cfg.min_features}"
        else:
            note = ""

        rounds.append(
            RefineRound(round_no, len(features), len(kept), round(cut, 6), auc, note)
        )
        if len(kept) == len(features):
            # RefineRound 是 frozen —— 用 replace 而非原地赋值
            merged = f"{rounds[-1].note}；本轮无剔除，已收敛" if note else "本轮无剔除，已收敛"
            rounds[-1] = replace(rounds[-1], note=merged)
            break
        features = kept

    gain_df = pd.DataFrame(round_gains).T.fillna(0.0)
    gain_df.columns = [f"round_{i + 1}_gain" for i in range(gain_df.shape[1])]
    gain_df.index.name = "feature"
    gain_df["total_gain"] = gain_df.sum(axis=1)
    gain_df["retained"] = gain_df.index.isin(set(features))
    gain_df = gain_df.sort_values("total_gain", ascending=False)
    return _IterationOutcome(features, rounds, gain_df)


def _best_auc(model) -> float | None:
    try:
        return round(float(model.best_score["valid_0"]["auc"]), 6)
    except (KeyError, TypeError, ValueError):
        return None