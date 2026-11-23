"""按阶段的样本切分口径 —— **各阶段用哪部分数据，全项目唯一来源**。

★ 为什么必须集中收口
--------------------
同一份数据在「初筛 / 精筛 / 调参」三个阶段用的是**不同子集**：

============  ==============================================  ====================
阶段          用哪些数据                                     理由
============  ==============================================  ====================
特征初筛      **只有 train**                                初筛的 IV / PSI 是
                                                            「用训练期学到的分布」
                                                            去评判特征，掺入 val/OOT
                                                            会让筛选结果带上未来信息
特征精筛      **train + val**（train 训练，val 做早停/评估）  精筛要按「线上能否复现」
                                                            训练与早停
调参          **train + val**                                调参目标是候选参数的评价，
                                                            与精筛同一套数据口径
============  ==============================================  ====================

★★ 口径不符是最隐蔽的一类缺陷
----------------------------
「用错了数据子集」**不报错、不崩、结果看着合理** ——
初筛在 train+val+OOT 上算 IV，得到的 IV 偏高；精筛在 train+val+OOT 上训练，
得到的特征重要性掺了未来信息。**整条链路的结论全错，但没有任何异常信号。**

⇒ 本模块把「阶段 → 取哪个子集」写成**可执行的函数**，
  并在**缺失时抛错**而不是静默兜底。

★★ 为什么缺失即报错（不退回最早账龄月）
------------------------------------
旧实现（``_split_masks``）在切分列没有 ``train`` 时会**退回最早账龄月**。
这在「只有一个数据集、要自己造 train/val」的场景是合理的；
但在**用户已经明确给了 split 列**的场景下，它是危险的 ——
用户配了 split 列却发现它没生效，交付物里也看不出来。

⇒ 决策（用户明确拍板）：**各阶段严格按阶段口径，缺失即报错**。
  只有在**根本没有 split 列**时，才退回最早账龄月（并**显式标注**口径）。

★ 切分列的取值别名
------------------
``train`` / ``training`` / ``fit`` / ``train_set`` 都视为训练集；
``val`` / ``valid`` / ``validation`` / ``dev`` 都视为验证集。
大小写不敏感、前后空白忽略。

★★ 三种「取不到口径」的情形，三种不同处理
------------------------------------------
======================================  =====================  =====================
情形                                    处理                   理由
======================================  =====================  =====================
切分列**已指定**但数据里没这一列        **报错**               配置与数据不符 ⇒
                                                                静默换口径 =
                                                                违背用户已确认的决定
切分列**已指定**但缺本阶段必需的子集    **报错**               同上
**根本没有**切分列                      按阶段兜底（见下）    数据本身无切分信息
======================================  =====================  =====================

★★ 无切分列时的兜底是**分阶段**的
--------------------------------
- **初筛**：用**全量数据**。
  初筛只需要「训练期学到的分布」，而没有切分信息时**无从界定训练期** ——
  此时用全量是唯一不凭空发明口径的做法；更重要的是，
  若也退回「最早账龄月」，初筛就只剩**一个月** ⇒ PSI 没有任何可比对的配对
  ⇒ **所有特征都会被剔除**（指标值为空），整轮初筛白跑。
- **精筛 / 调参**：用**最早账龄月**作训练集、其余作验证集。
  这两个阶段**必须有验证集**（早停 / 目标函数），而「用过去预测未来」
  是唯一不需要额外假设的切法。

⇒ 兜底口径必须在产出里**显式标注**（``used_fallback=True`` + 描述串）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping, Optional, Sequence

import numpy as np

from .errors import ConfigError, DataError

__all__ = [
    "SPLIT_MODES",
    "SPLIT_STAGE_EVAL",
    "SPLIT_STAGE_PRESCREEN",
    "SPLIT_STAGE_REFINE",
    "SPLIT_STAGE_TUNING",
    "SplitMode",
    "SplitResult",
    "SplitSpec",
    "SplitStage",
    "SplitSubset",
    "TEST_SPLIT_VALUES",
    "earliest_month_masks",
    "resolve_split_subset",
    "split_frame",
]

#: 切分列里代表「训练集」的取值（大小写不敏感）
TRAIN_SPLIT_VALUES: frozenset[str] = frozenset(
    {"train", "training", "fit", "train_set"}
)

#: 切分列里代表「验证集」的取值（大小写不敏感）
VAL_SPLIT_VALUES: frozenset[str] = frozenset(
    {"val", "valid", "validation", "valid_set", "dev"}
)

#: 切分列里代表「评价样本(test)」的取值（大小写不敏感）
#:
#: ★★ 2026-10-30 新增 —— 配合「模型评价阶段」导入 test
#: --------------------------------------------------
#: test 与 train/val 的**根本区别**：
#:
#: * train / val **参与**特征筛选与调参（它们决定模型长什么样）；
#: * test **不参与**任何筛选决策，只在评价阶段被读取与打分
#:   ⇒ 它的指标是「未参与选择过的样本上的表现」，有独立参考价值。
#:
#: ⇒ 因此 test 的取值别名**独立维护**，不与 train/val 混用。
TEST_SPLIT_VALUES: frozenset[str] = frozenset(
    {"test", "testing", "test_set", "oot", "out_of_time", "holdout", "hold_out"}
)

SplitStage = Literal["prescreen", "refine", "tuning", "eval"]
"""阶段标识。"""

SPLIT_STAGE_PRESCREEN: SplitStage = "prescreen"
SPLIT_STAGE_REFINE: SplitStage = "refine"
SPLIT_STAGE_TUNING: SplitStage = "tuning"
SPLIT_STAGE_EVAL: SplitStage = "eval"
"""模型评价阶段。

★ 与前三个阶段的**根本差异**
------------------------
``prescreen`` / ``refine`` / ``tuning`` 三个阶段的口径是**筛选口径** ——
决定「哪些特征能进模型」。

``eval`` 是**评价口径** —— 决定「拿哪些样本算模型表现」。
它**不是**新的筛选口径：筛选口径完全不变（初筛 train；精筛/调参 train+val），
test 只是在这里额外参与**指标计算**。

★ 「是否含 test」不是阶段属性，而是**本次运行的选择**
-------------------------------------------------
⇒ 不写死成 ``("train", "val", "test")``，
  而是让调用方传 ``include_test`` 开关（见 :func:`resolve_split_subset`），
  这样「不导入 test」与「导入 test」两条路径走的是**同一份口径代码**，
  不会各自实现一遍而出现口径漂移。
"""

#: 每个阶段用哪些子集
#:
#: ★ 注意：``eval`` **不在这里** —— 它的子集是「train+val [+test]」，
#:   由 ``include_test`` 开关决定，故走单独的分支。
_STAGE_SUBSETS: dict[str, tuple[str, ...]] = {
    SPLIT_STAGE_PRESCREEN: ("train",),
    SPLIT_STAGE_REFINE: ("train", "val"),
    SPLIT_STAGE_TUNING: ("train", "val"),
}

#: 无切分列时，各阶段的兜底方式
#:
#: ``"all"``  —— 用全量数据（初筛：没有切分信息时无从界定训练期）
#: ``"earliest_month"`` —— 最早账龄月作训练集、其余作验证集（精筛/调参：需要验证集）
#: ``"train_val_only"`` —— 无切分列时评价阶段只取全量（无 test 概念）
_STAGE_FALLBACK: dict[str, str] = {
    SPLIT_STAGE_PRESCREEN: "all",
    SPLIT_STAGE_REFINE: "earliest_month",
    SPLIT_STAGE_TUNING: "earliest_month",
    SPLIT_STAGE_EVAL: "all",
}

_STAGE_LABELS: dict[str, str] = {
    SPLIT_STAGE_PRESCREEN: "特征初筛",
    SPLIT_STAGE_REFINE: "特征精筛",
    SPLIT_STAGE_TUNING: "调参",
    SPLIT_STAGE_EVAL: "模型评价",
}


@dataclass(frozen=True)
class SplitSubset:
    """一个阶段的取数子集。"""

    stage: SplitStage
    subsets: tuple[str, ...]
    """本阶段取哪几个子集（``train`` / ``val`` / ``test``）。"""
    train_mask: Any
    """训练集掩码（bool Series，与输入 df 同索引）。"""
    val_mask: Any
    """验证集掩码（**真实** val 子集，不随阶段裁剪）。

    ★★ 阶段口径由 ``subsets`` + :meth:`mask_for` 决定，**不是**由本字段决定
    ------------------------------------------------------------------
    本字段始终是「数据里属于 val 的行」。初筛阶段 ``subsets=("train",)`` ⇒
    实际参与计算的样本只有 train，但 ``val_mask`` 仍然非空 ——
    这是**故意**的：掩码是数据的客观属性，阶段口径是流程的决策，两者分开。

    ⇒ 阶段口径一律用 :meth:`mask_for` 取，别直接用 ``val_mask`` 做计算。
    """
    description: str
    """人可读的口径描述 —— 必须出现在交付物里。"""
    used_fallback: bool = False
    """是否走了「无 split 列 ⇒ 退回最早账龄月」的兜底。"""
    test_mask: Any = None
    """评价样本(test)掩码。**仅评价阶段使用**。

    ★★ test 在其他阶段永远是 None
    ------------------------------
    test **绝不参与**初筛/精筛/调参 —— 那是「筛选口径」，
    而 test 的价值恰恰在于「未参与选择过的样本」。
    ⇒ 除评价阶段外，本字段恒为 None，从结构上杜绝 test 混入筛选。

    ★ 为什么放在末尾
    ----------------
    dataclass 要求「有默认值的字段必须在无默认值的字段之后」。
    本字段是唯一一个可选字段 ⇒ 放最后，不破坏既有构造调用。
    """

    @property
    def stage_label(self) -> str:
        return _STAGE_LABELS.get(self.stage, self.stage)

    def mask_for(self) -> Any:
        """本阶段实际参与计算的样本掩码。"""
        if len(self.subsets) == 1:
            return self.train_mask
        masks = [self.train_mask, self.val_mask]
        if "test" in self.subsets and self.test_mask is not None:
            masks.append(self.test_mask)
        out = masks[0]
        for m in masks[1:]:
            out = out | m
        return out

    def counts(self) -> dict[str, int]:
        out = {
            "train": int(self.train_mask.sum()),
            "val": int(self.val_mask.sum()),
        }
        if self.test_mask is not None:
            out["test"] = int(self.test_mask.sum())
        return out


def _month_sort_key(value: str) -> tuple[int, int, str]:
    """把月份字符串解析成可比较的 (年, 月, 原串)。"""
    import re

    s = str(value).strip()
    nums = re.findall(r"\d+", s)
    if len(nums) >= 2:
        return (int(nums[0]), int(nums[1]), s)
    if len(nums) == 1:
        return (0, int(nums[0]), s)
    return (0, 0, s)


def _earliest_month_mask(df, date_col: str):
    """按**年月数字**取最早账龄月（不能按字符串排！）。

    ★ 中文月份按字符串排会得到 ``2026年12月 < 2026年2月``，
      于是「最晚的月」被当成「最早的月」⇒ 训练集选错 ⇒ 结论全错且不报错。
    """
    months = df[date_col].astype(str)
    earliest = min(months, key=_month_sort_key)
    mask = months == earliest
    return mask, (~mask), earliest


def _all_mask(df):
    """全选掩码（与 df 同索引的布尔 Series）。"""
    import pandas as pd

    return pd.Series(True, index=df.index)


def resolve_split_subset(
    df,
    params: Dict[str, Any],
    stage: SplitStage,
    *,
    split_col: Optional[str] = None,
    date_col: Optional[str] = None,
    include_test: bool = False,
    test_from_external: bool = False,
) -> SplitSubset:
    """按阶段口径切出样本子集。

    Parameters
    ----------
    df
        已读入的数据框（**必须含切分列与日期列**）。
    params
        参数字典（用于取 ``columns.split`` / ``columns.date``）。
    stage
        阶段标识 —— 决定取哪几个子集。
    include_test
        ★ **仅评价阶段（``SPLIT_STAGE_EVAL``）有效**：是否把 test 纳入本阶段。

        ★ 为什么做成「开关」而不是独立的阶段
        --------------------------------------
        「导入 test ⇒ 评价基于 train-val-test」与「不导入 ⇒ 评价基于 train-val」
        两条路径**只差一个 test**，其余口径完全相同。
        ⇒ 若写成两个阶段，两份口径代码各自演化 ⇒ 迟早漂移。
        ⇒ 现在是**同一份代码 + 一个开关**，两条路径不可能分叉。

        ★★ test 不参与筛选（结构性保证）
        --------------------------------
        本参数**只对 eval 阶段生效** —— 传入 ``prescreen`` / ``refine`` / ``tuning``
        时一律忽略（且 test_mask 恒为 None），
        从结构上保证 test 不会混进初筛/精筛/调参。
    test_from_external
        ★★ test 是否来自**独立文件**（``eval.test_path``）。

        ★★★ 为什么需要这个开关（本轮修掉的一个真缺陷）
        -------------------------------------------------
        「导入 test」有**两种来源**（用户 2026-10-30 拍板：两者都支持）：

        ① **独立文件**（``eval.test_path``）—— test 在另一个文件里；
        ② **主数据切分列的 test 行** —— test 就在主数据的 ``split`` 列里。

        ★ 问题：① 的情况下，主数据**本来就没有** test 行
          （实测 ``样本3.csv`` 只有 train 2273 / val 726）。
          而本函数原先在 ``include_test=True`` 时**无条件**要求主数据含 test 子集
          ⇒ **独立文件这条路直接走不通**，评价阶段直接报错。

        ★ 这是「口径校验」与「数据来源」两件事被混为一谈：
          口径要的是「本阶段用 train+val+test」，
          但 **test 从哪来**是另一回事 —— 它可以来自另一个文件。

        ⇒ 本开关为 True 时：主数据**只要求** train + val，
          test 子集由调用方从独立文件取（本函数返回 ``test_mask=None``）。

    Raises
    ------
    DataError
        用户**已指定切分列**但该列：
        - 不在数据里（配置与数据不符），或
        - 缺少本阶段必需的子集（如初筛要求 train，却没有 train 行）

        ⇒ **报错**，绝不静默退回最早账龄月。

    ★★ 为什么「已指定 split 列」和「没指定」要区别对待
    ---------------------------------------------------
    * **没指定切分列** → 数据本身没有切分信息 ⇒ 按阶段兜底（见下），
      但必须在交付物里**显式标注**；
    * **已指定切分列** → 用户明确表达了切分意图 ⇒ 静默换口径等于
      **违背用户已确认的决定**，且不可见 ⇒ 必须报错。
    """
    from .errors import DataError

    if split_col is None:
        split_col = params.get("columns.split") or ""
    if date_col is None:
        date_col = params.get("columns.date") or ""

    # ★★ eval 阶段的口径是「train+val [+test]」—— 由开关决定，不写死。
    if stage == SPLIT_STAGE_EVAL:
        wanted = ("train", "val", "test") if include_test else ("train", "val")
    else:
        wanted = _STAGE_SUBSETS.get(stage)
        if wanted is None:
            raise DataError(f"未知阶段 {stage!r}，可选：{sorted(_STAGE_SUBSETS)}")
        # ★ 非评价阶段一律**不认 test** —— 筛选口径不因导入 test 而改变。
        include_test = False

    # ★★ test 来自独立文件 ⇒ 主数据**不要求**含 test 行。
    #   （test 行在另一个文件里，主数据本来就没有 —— 见上文的缺陷说明）
    required_from_main = wanted
    if test_from_external and "test" in wanted:
        required_from_main = ("train", "val")


    # ── 情形 A：有切分列 ──
    if split_col:
        if split_col not in df.columns:
            # ★★ 已指定切分列但数据里没有 ⇒ 配置与数据不符 ⇒ 报错。
            #   静默退回兜底 = 违背用户已确认的决定，且交付物里看不出来。
            raise DataError(
                f"{_STAGE_LABELS.get(stage, stage)}阶段要求按切分列 {split_col!r} 取数，"
                f"但数据里**没有这一列**。\n"
                f"★ 静默退回兜底口径等于违背你已确认的切分决定，"
                f"且交付物里看不出来 ⇒ 这里直接报错。\n"
                f"★ 数据实际列名（前 15 个）：{list(df.columns)[:15]}\n"
                f"★ 请检查：① 切分列名是否写错；② 是否该改用「无切分列」模式。"
            )

        vals = df[split_col].astype(str).str.strip().str.lower()
        train_mask = vals.isin(TRAIN_SPLIT_VALUES)
        val_mask = vals.isin(VAL_SPLIT_VALUES)
        test_mask = vals.isin(TEST_SPLIT_VALUES)

        # ★ 其它取值一律不参与 —— 它们既不是 train/val，也不是已声明的 test。

        def _has(sub: str) -> bool:
            if sub == "train":
                return bool(train_mask.any())
            if sub == "val":
                return bool(val_mask.any())
            return bool(test_mask.any())

        missing = [s for s in required_from_main if not _has(s)]
        if missing:
            present = sorted(set(vals.unique()))
            raise DataError(
                f"{_STAGE_LABELS.get(stage, stage)}阶段要求切分列 "
                f"{split_col} 含 {missing} 子集，但实际没有。\n"
                f"★ 该阶段的口径是「只用 {' + '.join(wanted)}」—— "
                f"口径不符会静默改变结论（IV 偏高 / 特征重要性掺未来信息），"
                f"且没有任何报错信号。\n"
                f"★ 切分列实际取值：{present}\n"
                f"★ 请检查：① 切分列是否选错；② 数据里是否真的做了切分；"
                f"③ 或改用「无切分列」模式（系统会按阶段兜底并在结果中标注）。"
            )

        desc = (
            f"切分列 {split_col}："
            f"train {int(train_mask.sum())} 行"
            + (f"、val {int(val_mask.sum())} 行" if "val" in wanted else "")
            + (f"、test {int(test_mask.sum())} 行" if "test" in wanted else "")
            + f"（本阶段只用 {' + '.join(wanted)}）"
        )
        return SplitSubset(
            stage=stage,
            subsets=wanted,
            train_mask=train_mask,
            val_mask=val_mask,
            # ★ test 来自独立文件 ⇒ 主数据里没有 test 行 ⇒ test_mask 恒为 None
            test_mask=(
                test_mask
                if ("test" in wanted and not test_from_external)
                else None
            ),
            description=desc,
            used_fallback=False,
        )

    # ── 情形 B：没有切分列 ──
    #   数据本身没有切分信息 ⇒ 按阶段兜底。
    #   ★ 但必须显式标注口径 —— 让人知道这不是用户配的切分。
    fallback = _STAGE_FALLBACK.get(stage, "earliest_month")

    if fallback == "all":
        all_mask = _all_mask(df)
        if stage == SPLIT_STAGE_EVAL:
            desc = (
                f"★ 无切分列 ⇒ 评价阶段用**全量数据**（{int(all_mask.sum())} 行）。"
                f"（无切分信息时无从界定 train/val/test 的边界；"
                f"此时所有指标都是「全量口径」，不代表样本外表现。）"
            )
        else:
            desc = (
                f"★ 无切分列 ⇒ 本阶段用**全量数据**（{int(all_mask.sum())} 行）。"
                f"（初筛只需要「训练期学到的分布」，而无切分信息时无从界定训练期；"
                f"若改用「最早账龄月」则只剩一个月 ⇒ PSI 无配对 ⇒ 全部特征都会被剔除。）"
            )
        return SplitSubset(
            stage=stage,
            subsets=wanted,
            train_mask=all_mask,
            val_mask=~all_mask,
            test_mask=None,
            description=desc,
            used_fallback=True,
        )

    # fallback == "earliest_month"
    if date_col and date_col in df.columns:
        train_mask, val_mask, earliest = _earliest_month_mask(df, date_col)
        if not train_mask.any():
            raise DataError(
                f"切分列 {split_col or '(未指定)'} 不可用，日期列 {date_col} "
                f"也无法取到最早账龄月 —— 无法切出训练集。"
            )
        desc = (
            f"★ 无切分列 ⇒ 退回最早账龄月 {earliest} 作训练集"
            f"（{int(train_mask.sum())} 行），其余作验证集"
            f"（{int(val_mask.sum())} 行）。本阶段只用 {' + '.join(wanted)}。"
        )
        return SplitSubset(
            stage=stage,
            subsets=wanted,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=None,
            description=desc,
            used_fallback=True,
        )

    raise DataError(
        f"{_STAGE_LABELS.get(stage, stage)}阶段无法切分样本："
        f"既没有可用的切分列（{split_col or '未指定'}），"
        f"也没有日期列（{date_col or '未指定'}）可退回。"
    )


# =========================================================================== #
# 纯函数式样本划分（v2 新增）—— SplitSpec / split_frame
# =========================================================================== #
# 旧 ``resolve_split_subset`` 绑定「流程阶段 + 切分列取值」，是流程语义；
# v2 的 ``dataset_split`` 工具需要的是与流程无关、可复现、可落盘的划分结果
# （见 docs/tool-contract.md §3.3）。两者口径不同、不能混用，因此：
#
# * SplitSpec     —— 声明式划分规格（不划分 / 随机 / 分层 / 字段表达式）；
# * split_frame() —— 纯函数：(df, spec) -> SplitResult，
#                    不读文件、不落盘、不改 df，只返回掩码 + 计数 + 描述 + 指纹；
# * earliest_month_masks() —— 旧的按月兜底逻辑，保留为可复用 helper。
#
# ★ 为什么必须纯函数
#   同一份数据在「初筛 / 精筛 / 调参 / 评价」被反复取子集。若划分逻辑带状态
#   （缓存、随机种子藏在全局、写回 df 的列），两次调用就可能得到不同掩码，
#   而「同一个 seed 两次划分不一致」在报表里几乎看不出来 ——
#   直到有人复核指标时才发现 train 混了 test。
#   ⇒ 划分只由 (df, spec) 决定：同 seed 同输入 ⇒ 逐行相同的掩码。


SPLIT_MODES: tuple[str, ...] = ("none", "random", "stratified", "field")
"""四种划分方式（与 docs/tool-contract.md §3.3 的 config.mode 一致）。"""

SplitMode = Literal["none", "random", "stratified", "field"]

_SUBSET_NAMES: tuple[str, str, str] = ("train", "val", "test")


def _coerce_ratio(value: Any, label: str) -> float:
    """把 ``0.7`` / ``{"ratio": 0.7}`` / ``"0.7"`` 统一成 float。"""
    if isinstance(value, Mapping):
        if "ratio" not in value:
            return 0.0
        value = value["ratio"]
    if value is None:
        return 0.0
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise DataError(f"{label} 的比例必须是数字，收到 {value!r}") from exc
    if ratio < 0 or ratio > 1:
        raise DataError(f"{label} 的比例必须落在 [0, 1]，当前 {ratio}")
    return ratio


@dataclass(frozen=True)
class SplitSpec:
    """样本划分规格（声明式、可序列化、可复现）。

    Parameters
    ----------
    mode
        ``none`` —— 不划分（全部样本算训练/评估，报告标注）；
        ``random`` —— 随机抽样；
        ``stratified`` —— 按 ``stratify_by``（缺省用目标列）分层抽样；
        ``field`` —— 按字段表达式划分（``expression`` 提供各子集条件）。
    ratios
        ``(train, val, test)`` 比例，必须合计为 1（允许某段为 0）。
    seed
        随机种子；``random`` / ``stratified`` 的结果只由它和行数决定。
    stratify_by
        分层列（``stratified`` 模式）；``None`` 时用调用方传入的目标列。
    column
        字段表达式的参照列名（``field`` 模式；仅用于描述与校验，条件里可跨列）。
    expression
        ``{"train": "...", "val": "...", "test": "..."}``；用受限解析器求值
        （:mod:`risklab_mdl.core.expr`），**禁止 eval**，非法输入抛 ``ConfigError``。
    time_column
        时间列（可选）。存在时写入描述/落盘，供 PSI 分月使用；本身不参与划分。

    ★ 字段表达式的键
      同时接受 ``expr_train`` / ``expr_val`` / ``expr_test``（tool-contract 的
      ``field.expr_*`` 形态）与 ``train`` / ``val`` / ``test``，
      由 :meth:`from_config` 统一收口 —— 只有这一处做形态转换。
    """

    mode: SplitMode = "none"
    ratios: tuple[float, float, float] = (0.7, 0.2, 0.1)
    seed: int = 42
    stratify_by: Optional[str] = None
    column: Optional[str] = None
    expression: Optional[Mapping[str, str]] = None
    time_column: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mode not in SPLIT_MODES:
            raise ConfigError(
                f"未知划分方式 {self.mode!r}，可选：{list(SPLIT_MODES)}"
            )
        ratios = tuple(float(r) for r in self.ratios)
        if len(ratios) != 3:
            raise ConfigError(
                f"ratios 必须是 (train, val, test) 三项，当前 {len(ratios)} 项"
            )
        if any(r < 0 for r in ratios):
            raise ConfigError(f"ratios 不允许负数，当前 {ratios}")
        total = sum(ratios)
        if abs(total - 1.0) > 1e-3:
            raise ConfigError(
                f"ratios 合计必须为 1，当前合计 {total:.6f}（{ratios}）。"
                f"若要表示「不划分」请用 mode='none'。"
            )
        object.__setattr__(self, "ratios", ratios)
        if self.mode == "field" and not self.expression:
            raise ConfigError(
                "mode='field' 必须提供 expression="
                "{'train': ..., 'val': ..., 'test': ...}"
            )
        if self.mode == "stratified" and self.stratify_by is not None:
            if not str(self.stratify_by).strip():
                raise ConfigError("stratify_by 不能是空白字符串（留空请传 None）")

    # -- 形态转换 -------------------------------------------------------- #
    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "SplitSpec":
        """从工具入参 ``config``（docs/tool-contract.md §3.3）构造规格。

        兼容三种写法（契约与历史配置里都出现过，统一在此收口）：

        * ``{"mode": "random", "ratios": [0.7,0.2,0.1], "seed": 42}``
        * ``{"mode": "random", "train": 0.7, "val": 0.2, "test": 0.1}``
        * ``{"mode": "random", "train": {"ratio": 0.7, "seed": 42}, ...}``
        * ``{"mode": "field", "field": {"column": "month", "expr_train": "..."}}``
        """
        cfg: Mapping[str, Any] = config or {}
        if not isinstance(cfg, Mapping):
            raise ConfigError(f"划分 config 必须是映射，收到 {type(cfg).__name__}")
        mode = str(cfg.get("mode") or "none").strip().lower()
        if mode not in SPLIT_MODES:
            raise ConfigError(
                f"未知划分方式 {mode!r}，可选：{list(SPLIT_MODES)}"
            )

        raw_ratios = cfg.get("ratios")
        if raw_ratios is not None:
            try:
                ratios = tuple(float(r) for r in raw_ratios)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"ratios 必须是数字序列，收到 {raw_ratios!r}") from exc
        else:
            ratios = tuple(
                _coerce_ratio(cfg.get(name), name) for name in _SUBSET_NAMES
            )
            if all(r == 0.0 for r in ratios):
                ratios = (1.0, 0.0, 0.0) if mode == "none" else (0.7, 0.2, 0.1)

        seed_raw = cfg.get("seed")
        if seed_raw is None:
            train_cfg = cfg.get("train")
            if isinstance(train_cfg, Mapping) and "seed" in train_cfg:
                seed_raw = train_cfg["seed"]
            else:
                seed_raw = 42
        try:
            seed = int(seed_raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"seed 必须是整数，收到 {seed_raw!r}") from exc

        expression = cfg.get("expression")
        if expression is not None and not isinstance(expression, Mapping):
            raise ConfigError(
                "expression 必须是 {'train': ..., 'val': ..., 'test': ...} 映射"
            )
        field_cfg = cfg.get("field")
        column = cfg.get("column")
        expr_map: dict[str, str] = {}
        if isinstance(field_cfg, Mapping):
            column = field_cfg.get("column", column)
            for name in _SUBSET_NAMES:
                value = field_cfg.get(f"expr_{name}", field_cfg.get(name))
                if value:
                    expr_map[name] = str(value)
        if isinstance(expression, Mapping):
            for name in _SUBSET_NAMES:
                value = expression.get(name)
                if value:
                    expr_map[name] = str(value)

        stratify_by = cfg.get("stratify_by")
        time_column = cfg.get("time_column")
        return cls(
            mode=mode,
            ratios=ratios,
            seed=seed,
            stratify_by=None if stratify_by in (None, "") else str(stratify_by),
            column=None if column in (None, "") else str(column),
            expression=expr_map or None,
            time_column=None if time_column in (None, "") else str(time_column),
        )

    # -- 序列化 ---------------------------------------------------------- #
    def to_dict(self) -> dict:
        """落盘用（``split/split.json``）；字段名与 tool-contract 对齐。"""
        return {
            "mode": self.mode,
            "ratios": [float(r) for r in self.ratios],
            "train": {"ratio": float(self.ratios[0]), "seed": self.seed},
            "val": {"ratio": float(self.ratios[1])},
            "test": {"ratio": float(self.ratios[2])},
            "seed": self.seed,
            "stratify_by": self.stratify_by,
            "column": self.column,
            "expression": dict(self.expression) if self.expression else None,
            "time_column": self.time_column,
        }


@dataclass(frozen=True)
class SplitResult:
    """一次划分的结果（掩码 + 计数 + 描述 + 指纹）。

    掩码与输入 ``df`` 同索引、``dtype=bool``；三个子集互斥（不重叠）。
    """

    spec: SplitSpec
    train_mask: Any
    val_mask: Any
    test_mask: Any
    counts: Mapping[str, int]
    ratios: Mapping[str, float]
    description: str
    fingerprint: str
    warnings: tuple[str, ...] = ()

    def mask(self, subset: str) -> Any:
        """按名取掩码（``train`` / ``val`` / ``test``）。"""
        if subset not in _SUBSET_NAMES:
            raise ConfigError(f"未知子集 {subset!r}，可选：{list(_SUBSET_NAMES)}")
        return getattr(self, f"{subset}_mask")

    def masks(self) -> dict[str, Any]:
        return {name: self.mask(name) for name in _SUBSET_NAMES}

    def subset(self, df, subset: str):
        """返回 ``df`` 中属于该子集的行（新对象，不改原 df）。"""
        return df.loc[self.mask(subset)]

    def frames(self, df) -> tuple[Any, Any, Any]:
        """返回 ``(train, val, test)`` 三个子集（评价阶段直接用）。"""
        return tuple(self.subset(df, name) for name in _SUBSET_NAMES)  # type: ignore[return-value]

    def to_dict(self) -> dict:
        """落盘用（``split/split.json``，不含逐行掩码，只含指纹与统计）。"""
        return {
            "mode": self.spec.mode,
            "spec": self.spec.to_dict(),
            "counts": dict(self.counts),
            "ratios": {k: round(float(v), 6) for k, v in self.ratios.items()},
            "fingerprint": self.fingerprint,
            "description": self.description,
            "warnings": list(self.warnings),
        }


def _allocate(n_rows: int, ratios: Sequence[float]) -> tuple[int, int, int]:
    """按比例分配行数；保证三段之和恰为 ``n_rows``（不丢行、不重复）。

    ★ 用 ``floor(x + 0.5)`` 而不是 ``round``：Python 的 ``round`` 是银行家舍入
      （``round(0.5) == 0``），在极小样本下会让 train 段凭空少一行。
    """
    if n_rows <= 0:
        return (0, 0, 0)
    cum1 = int(np.floor(n_rows * float(ratios[0]) + 0.5))
    cum2 = int(np.floor(n_rows * (float(ratios[0]) + float(ratios[1])) + 0.5))
    cum1 = max(0, min(cum1, n_rows))
    cum2 = max(cum1, min(cum2, n_rows))
    return (cum1, cum2 - cum1, n_rows - cum2)


def _fingerprint(*masks) -> str:
    """掩码指纹：三段掩码按位打包后取 SHA256 前 16 位。"""
    import hashlib

    h = hashlib.sha256()
    for mask in masks:
        h.update(np.asarray(mask, dtype=bool).tobytes())
        h.update(b"|")
    return h.hexdigest()[:16]


def _build_result(
    spec: SplitSpec,
    df,
    masks: Mapping[str, Any],
    *,
    description: str,
    warnings: Sequence[str] = (),
) -> SplitResult:
    n = int(df.shape[0])
    counts = {name: int(masks[name].sum()) for name in _SUBSET_NAMES}
    ratios = {name: (counts[name] / n if n else 0.0) for name in _SUBSET_NAMES}
    return SplitResult(
        spec=spec,
        train_mask=masks["train"],
        val_mask=masks["val"],
        test_mask=masks["test"],
        counts=counts,
        ratios=ratios,
        description=description,
        fingerprint=_fingerprint(masks["train"], masks["val"], masks["test"]),
        warnings=tuple(warnings),
    )


def earliest_month_masks(df, date_col: str):
    """按最早账龄月切 train/val 的**可复用 helper**（旧口径，原样保留）。

    Returns
    -------
    (train_mask, val_mask, earliest_month)
        最早账龄月 ⇒ train，其余 ⇒ val（不产生 test）。
    """
    if date_col not in df.columns:
        raise DataError(f"日期列 {date_col!r} 不在数据里")
    return _earliest_month_mask(df, date_col)


def split_frame(
    df,
    spec: SplitSpec | Mapping[str, Any] | None = None,
    *,
    target: Optional[str] = None,
) -> SplitResult:
    """按 :class:`SplitSpec` 划分样本（**纯函数**：不改 df、不落盘）。

    Parameters
    ----------
    df
        待划分数据框（``stratified`` 模式须含 ``stratify_by`` 列）。
    spec
        :class:`SplitSpec`，或直接给 tool-contract 的 ``config`` 映射
        （内部走 :meth:`SplitSpec.from_config`）。
    target
        目标列名 —— ``stratified`` 模式未指定 ``stratify_by`` 时用它分层。

    Returns
    -------
    SplitResult
        掩码 / 计数 / 实际占比 / 描述 / 指纹（同 seed 同输入 ⇒ 指纹相同）。

    Raises
    ------
    ConfigError
        规格非法、分层列缺失、字段表达式非法或子集重叠。
    DataError
        数据为空、缺少必要列。
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise DataError(f"split_frame 需要 DataFrame，收到 {type(df).__name__}")
    if isinstance(spec, Mapping) or spec is None:
        spec = SplitSpec.from_config(spec)
    if not isinstance(spec, SplitSpec):  # pragma: no cover - 防御性
        raise ConfigError(f"spec 必须是 SplitSpec 或映射，收到 {type(spec).__name__}")

    n = int(df.shape[0])
    index = df.index
    warn: list[str] = []

    # ── mode=none ──
    if spec.mode == "none":
        all_mask = pd.Series(True, index=index, dtype=bool)
        empty = pd.Series(False, index=index, dtype=bool)
        note = "未划分：全部样本用于训练与评估（指标不代表样本外表现）"
        return _build_result(
            spec,
            df,
            {"train": all_mask, "val": empty, "test": empty},
            description=note,
            warnings=(note,),
        )

    if n == 0:
        raise DataError("数据为空（0 行），无法划分样本")

    # ── mode=field ──
    if spec.mode == "field":
        from .expr import compile_expression

        expr_map = dict(spec.expression or {})
        masks: dict[str, Any] = {}
        for name in _SUBSET_NAMES:
            text = expr_map.get(name)
            if not text:
                masks[name] = pd.Series(False, index=index, dtype=bool)
                continue
            masks[name] = compile_expression(text).evaluate(df)

        # 子集必须互斥：重叠会让同一行同时进 train 与 test ⇒ 指标虚高且不可见
        for i, a in enumerate(_SUBSET_NAMES):
            for b in _SUBSET_NAMES[i + 1:]:
                overlap = int((masks[a] & masks[b]).sum())
                if overlap:
                    raise ConfigError(
                        f"字段划分的 {a} 与 {b} 条件重叠 {overlap} 行 —— "
                        f"子集必须互斥，否则同一行会同时出现在两边（口径错误）。\n"
                        f"★ {a} = {expr_map.get(a)!r}\n★ {b} = {expr_map.get(b)!r}"
                    )

        covered = masks["train"] | masks["val"] | masks["test"]
        uncovered = int((~covered).sum())
        if uncovered:
            warn.append(
                f"字段划分未覆盖 {uncovered} 行（不落入 train/val/test 任一条件）⇒ "
                f"这些样本被排除，不参与任何计算"
            )
        desc = (
            f"字段表达式划分（列 {spec.column or '(多列)'}）："
            + "、".join(
                f"{name} {int(masks[name].sum())} 行" for name in _SUBSET_NAMES
            )
        )
        return _build_result(spec, df, masks, description=desc, warnings=warn)

    # ── mode=random / stratified ──
    rng = np.random.default_rng(int(spec.seed))
    masks = {name: np.zeros(n, dtype=bool) for name in _SUBSET_NAMES}

    if spec.mode == "random":
        order = rng.permutation(n)
        n_train, n_val, _n_test = _allocate(n, spec.ratios)
        masks["train"][order[:n_train]] = True
        masks["val"][order[n_train:n_train + n_val]] = True
        masks["test"][order[n_train + n_val:]] = True
        desc = (
            f"随机划分（seed={spec.seed}）：train {int(masks['train'].sum())} / "
            f"val {int(masks['val'].sum())} / test {int(masks['test'].sum())} 行"
        )
    else:  # stratified
        strat_col = spec.stratify_by or target
        if not strat_col:
            raise ConfigError(
                "stratified 划分需要分层列：请传 stratify_by 或目标列名（target）。"
            )
        if strat_col not in df.columns:
            raise ConfigError(
                f"分层列 {strat_col!r} 不在数据里。数据实际列名（前 20 个）："
                f"{list(df.columns)[:20]}"
            )
        # 每个分层独立按比例分配；分层顺序由 factorize 固定 ⇒ 结果可复现
        codes, uniques = pd.factorize(df[strat_col], use_na_sentinel=False)
        codes = np.asarray(codes)
        for code in range(len(uniques)):
            pos = np.flatnonzero(codes == code)
            m = int(pos.size)
            a, b, _c = _allocate(m, spec.ratios)
            shuffled = rng.permutation(m)
            masks["train"][pos[shuffled[:a]]] = True
            masks["val"][pos[shuffled[a:a + b]]] = True
            masks["test"][pos[shuffled[a + b:]]] = True
        desc = (
            f"分层划分（按 {strat_col}，seed={spec.seed}）："
            f"train {int(masks['train'].sum())} / val {int(masks['val'].sum())} / "
            f"test {int(masks['test'].sum())} 行"
        )
        n_levels = int(len(uniques))
        if n_levels > max(2, n // 10):
            warn.append(
                f"分层列 {strat_col} 有 {n_levels} 个取值，分层过细 ⇒ 各段样本量小、"
                f"指标波动大，建议改用 random 或换低基数列"
            )

    masks = {
        name: pd.Series(masks[name], index=index, dtype=bool)
        for name in _SUBSET_NAMES
    }
    return _build_result(spec, df, masks, description=desc, warnings=warn)
