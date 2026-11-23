"""模型文档（固定 6 sheet）的计算层 —— 纯函数，不落盘、不读 MCP 参数。

★ 唯一数据源：编码后的数据帧 + 模型包
--------------------------------------
所有指标都由 ``predict_score`` 产出的 ``predict_proba``（模型概率分）自算，
与「是否调用过打分工具」彻底解耦 —— 同一份会话不会因为打分分支不同
就产出两份内容不同的模型文档。

★ 固定 6 sheet，恒产出
-----------------------
``概览 / 分月表现 / 分箱WOE / 分组评估 / PSI / 特征重要性``。
「分组评估」在用户没声明维度列时保留为**带说明的空表**，
而不是让 sheet 整个消失 —— 「没算出来」与「不存在」不能混淆。

★ 三条口径纪律
---------------
1. train / val / test 严格分开，从不混算；
2. 样本量 < ``MIN_SEG_SIZE`` 照算 + ``可靠=False`` + 说明，不静默丢弃；
   单一类别样本 KS/AUC 返回**空**（非 0）+ 说明；
3. 分箱按样本集各自拟合切点；PSI 切点只在基期池上拟合、各月共用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .binning import BinSpec, bin_table_processed, fit_bins, transform_bins
from .errors import DataError
from .metrics import calc_auc, calc_gini, calc_ks, calc_lift, score_direction
from .package import ModelPackage, describe_scale, predict_score

__all__ = [
    "BIN_NUMS",
    "LIFT_PCT",
    "MIN_SEG_SIZE",
    "SHEET_NAMES",
    "ModelDocInput",
    "build_model_doc_sheets",
]

#: 分箱 WOE 的三个箱数方案；LIFT 的头尾比例；分组样本量下限。
BIN_NUMS: tuple[int, ...] = (5, 10, 20)
LIFT_PCT: float = 0.10
MIN_SEG_SIZE: int = 30
_EPS: float = 1e-6

SHEET_NAMES: tuple[str, ...] = (
    "概览",
    "分月表现",
    "分箱WOE",
    "分组评估",
    "PSI",
    "特征重要性",
)

_METRIC_COLS: tuple[str, ...] = (
    "样本量", "坏样本数", "坏样本率", "KS", "AUC", "GINI",
    "LIFT@头部(最差10%)", "LIFT@尾部(最优10%)",
    "头部可靠", "尾部可靠", "方向", "可靠", "说明",
)

_PROFILE_COLS: tuple[str, ...] = (
    "查得数", "查得率", "缺失率", "取值数", "均值", "标准差",
    "最小值", "最大值", "变异系数",
    "P1", "P5", "P25", "P50", "P75", "P95", "P99",
)


@dataclass
class ModelDocInput:
    """构建模型文档所需的全部上下文（由执行层从参数文档收齐后传入）。"""

    frame: pd.DataFrame
    package: ModelPackage
    label_col: str = "y"
    month_col: str = "month"
    split_col: str = "split"
    #: ★ 只认弹窗拍板的维度类列；为空时「分组评估」出带说明的空表。
    dimensions: Sequence[str] = ()
    #: ★ PSI 基期月（弹窗拍板，可多月合并为一个池）。
    base_months: Sequence[str] = ()
    base_label: str = ""
    iv_values: Mapping[str, float] = field(default_factory=dict)
    iv_bins: Mapping[str, Sequence[int]] = field(default_factory=dict)
    psi_metrics: Mapping[str, float] = field(default_factory=dict)
    importance: Sequence[Mapping[str, Any]] = ()
    split_desc: str = ""
    #: 仅用于概览「评分标尺」区块点破「有标尺 ≠ 有分数」，不参与任何指标计算。
    score_files: Sequence[str] = ()
    score_n_rows: int = 0


# --------------------------------------------------------------------------- #
# 单组指标
# --------------------------------------------------------------------------- #
def _summarize(y: np.ndarray, s: np.ndarray) -> dict[str, Any]:
    """一组样本的指标。★ 单一类别时 KS/AUC 置 None，绝不写 0。

    写进 DataFrame 数值列后 pandas 会把 None 转成 NaN，落到 xlsx 仍是空格；
    「留空」指的是**单元格空**，不是要求内存里保持 None（dtype 不是报告口径）。
    """
    n = int(len(y))
    n_bad = int((y == 1).sum())
    row: dict[str, Any] = {
        "样本量": n,
        "坏样本数": n_bad,
        "坏样本率": round(n_bad / n, 6) if n else None,
        "KS": None,
        "AUC": None,
        "GINI": None,
        "LIFT@头部(最差10%)": None,
        "LIFT@尾部(最优10%)": None,
        "头部可靠": None,
        "尾部可靠": None,
        "方向": "",
        "可靠": n >= MIN_SEG_SIZE,
        "说明": "",
    }
    notes: list[str] = []
    if n == 0:
        row["可靠"] = False
        notes.append("该组没有样本")
    elif n_bad == 0 or n_bad == n:
        row["可靠"] = False
        notes.append(
            "该组只有单一类别样本，KS/AUC 无定义（返回空而非 0；"
            "0 会被误读为「模型毫无区分度」）"
        )
    else:
        row["KS"] = round(float(calc_ks(y, s)), 6)
        row["AUC"] = round(float(calc_auc(y, s)), 6)
        row["GINI"] = round(float(calc_gini(y, s)), 6)
        direction = score_direction(y, s)
        row["方向"] = (
            "分数越高风险越高"
            if direction == "higher_score_higher_risk"
            else "分数越高风险越低"
        )
        for worst, val_key, rel_key in (
            (True, "LIFT@头部(最差10%)", "头部可靠"),
            (False, "LIFT@尾部(最优10%)", "尾部可靠"),
        ):
            res = calc_lift(
                y, s, pct=LIFT_PCT, worst=worst,
                direction=direction, min_seg_size=MIN_SEG_SIZE,
            )
            row[val_key] = round(float(res.lift), 6)
            row[rel_key] = bool(res.is_reliable)
    if 0 < n < MIN_SEG_SIZE:
        row["可靠"] = False
        notes.append(f"样本量 {n} < 下限 {MIN_SEG_SIZE}，结论仅供参考")
    row["说明"] = "；".join(notes)
    return row


def _year_month_key(text: Any) -> tuple[int, int]:
    """'2026年12月' ⇒ (2026, 12) —— 按数字排序，不按字符串排。"""
    mt = re.search(r"(\d{4})\D*(\d{1,2})", str(text))
    return (int(mt.group(1)), int(mt.group(2))) if mt else (9999, 99)


def _work_frame(data: ModelDocInput) -> pd.DataFrame:
    """把编码数据 + 概率分收成一张工作表，并校验必需列。"""
    frame = data.frame
    needed = [data.label_col, data.month_col, data.split_col, *data.dimensions]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise DataError(
            f"模型文档缺少分组/口径列：{missing}。"
            f"★ 维度列来自「维度列确认」弹窗；缺列时**不做静默跳过** —— "
            f"那会让报告少一个维度却没有任何异常信号。"
        )
    proba = predict_score(frame, data.package)["predict_proba"]
    work = frame[needed].copy()
    work["_proba"] = proba.reindex(frame.index).to_numpy(dtype=float)
    y = pd.to_numeric(work[data.label_col], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(y) & np.isfinite(work["_proba"].to_numpy(dtype=float))
    work = work.loc[keep].reset_index(drop=True)
    work[data.label_col] = y[keep]
    work["_proba"] = work["_proba"].astype(float).to_numpy()
    return work


def _y_s(part: pd.DataFrame, label_col: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        part[label_col].to_numpy(dtype=float),
        part["_proba"].to_numpy(dtype=float),
    )


# --------------------------------------------------------------------------- #
# sheet 1：概览（5 个区块）
# --------------------------------------------------------------------------- #
def _overview_sheet(data: ModelDocInput, work: pd.DataFrame) -> pd.DataFrame:
    pkg = data.package
    sets = [str(v) for v in work[data.split_col].dropna().unique()]
    by_set = {
        name: _summarize(*_y_s(work[work[data.split_col] == name], data.label_col))
        for name in sets
    }
    rows: list[dict[str, Any]] = []

    def block(title: str) -> None:
        rows.append({"项目": title, "值": ""})

    def item(key: str, value: Any) -> None:
        rows.append({"项目": "  " + key, "值": "" if value is None else str(value)})

    # ① 指标（按样本集）—— 行=指标、列=样本集，横排对比，绝不混算
    block("指标（按样本集）")
    head: dict[str, Any] = {"项目": "  样本集", "值": ""}
    head.update({s: s for s in sets})
    rows.append(head)
    for metric in (
        "样本量", "坏样本数", "坏样本率", "KS", "AUC", "GINI",
        "LIFT@头部(最差10%)", "LIFT@尾部(最优10%)",
        "头部可靠", "尾部可靠", "方向", "可靠",
    ):
        line: dict[str, Any] = {"项目": "  " + metric, "值": ""}
        line.update({s: by_set[s].get(metric) for s in sets})
        rows.append(line)

    # ② 模型
    block("模型")
    item("算法", pkg.algorithm)
    item("入模特征数", len(pkg.features))
    item("最佳迭代轮数", pkg.best_iteration)
    item("模型参数", str(dict(pkg.params or {}))[:600])
    item("推理期变换", "、".join(pkg.transform_names()) or "（旧版 LGB 无变换）")
    item("模型包摘要", pkg.digest())
    item("模型包 schema", getattr(pkg, "schema_version", ""))
    item("创建时间", pkg.created_at)
    item("标签口径摘要", pkg.label_spec_digest or "（空）")
    item("评价口径", data.split_desc or "（未记录）")

    # ③ 评分标尺（★ 必须点破「有标尺 ≠ 有分数」）
    block("评分标尺")
    for k, v in describe_scale(pkg).items():
        item(k, v)
    item(
        "本次是否产出打分明细",
        f"是（{data.score_n_rows} 行，{len(data.score_files)} 个文件）"
        if data.score_files else "否（未产出任何打分明细文件）",
    )
    item(
        "标尺与分数的关系",
        "评分标尺是模型包自带的刻度定义，**不代表已经产出分数**；"
        "本文档全部指标由模型概率分自算，与是否打分无关。",
    )

    # ④ 缺失处理
    block("缺失处理")
    item("填充策略", "不填充（LightGBM 原生支持 NaN，按默认分支处理）")
    item(
        "为何不用哨兵值 99999",
        "哨兵值会让缺失变成一个「数值上极大」的合法取值，可能被误当作极端值参与分裂；"
        "LGB 的缺失分支语义更明确",
    )
    item("声明了填充值的列数", len(pkg.fill_values or {}))
    item(
        "三期 LR 说明",
        "LR 不支持 NaN，届时必须逐列声明 fill_values，模型包已预留该字段",
    )

    # ⑤ 标签口径 + 指标说明
    block("标签口径 + 指标说明")
    item("Y 列名", data.label_col)
    item("好/坏取值", "坏=1、好=0（标签列按位置识别，取值须为 0/1）")
    item("特殊值", "标签缺失的行已在评价前剔除，不参与任何指标")
    item(
        "KS",
        "累积好坏分布的最大垂直距离，取值 0~1，越大越好；不区分方向。",
    )
    item(
        "AUC",
        "ROC 曲线下面积，0.5 表示无区分度；本服务不翻转方向，0.5 以下表示分数与风险反向。",
    )
    item("GINI", "2×AUC−1，可正可负；负值表示分数与风险反向。")
    item(
        "LIFT@头部(最差10%)",
        "风险最高的那 10% 客户，其坏样本率是整体坏样本率的多少倍；"
        "＞1 表示坏样本确实集中在头部，风控最关心的口径。",
    )
    item(
        "LIFT@尾部(最优10%)",
        "风险最低的那 10% 客户的同口径倍数（＜1 为正常）；"
        "作为对照，用于判断分数是否单调对应风险。",
    )
    item(
        "头部可靠 / 尾部可靠 / 可靠",
        f"样本量是否达到 {MIN_SEG_SIZE} 条下限。False = 未达到，该值仅供参考，"
        f"不可作为结论；单一类别组的 KS/AUC 直接留空并写明原因。",
    )
    item(
        "方向",
        "「分数越高风险越高」表示分数可直接当风险分；"
        "「分数越高风险越低」表示分数是信用分语义，用时须注意方向转换。"
        "★ 指标表里的方向针对**模型概率分 predict_proba**；"
        "评分标尺的方向是刻度固定语义（见上方「评分标尺」区块），两者相反属正常。",
    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# sheet 2：分月表现（样本集 × 月份交叉）
# --------------------------------------------------------------------------- #
def _month_sheet(data: ModelDocInput, work: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = (
        work.groupby([data.split_col, data.month_col], dropna=False, sort=False)
        .groups.keys()
    )
    for split_name, month in sorted(pairs, key=lambda p: (str(p[0]), _year_month_key(p[1]))):
        part = work[
            (work[data.split_col] == split_name) & (work[data.month_col] == month)
        ]
        row: dict[str, Any] = {"样本集": split_name, "月份": month}
        row.update(_summarize(*_y_s(part, data.label_col)))
        rows.append(row)
    return pd.DataFrame(rows, columns=["样本集", "月份", *_METRIC_COLS])


# --------------------------------------------------------------------------- #
# sheet 3：分箱 WOE（5/10/20 箱，各样本集独立拟合切点）
# --------------------------------------------------------------------------- #
def _bin_spec(n_bins: int) -> BinSpec:
    """展示用途的分箱规格：严格按请求箱数拟合，WOE 走归一化平滑。

    ★ 为什么不用 ``eps=1e-6`` 下限裁剪：全好/全坏箱裁剪后会得到
      WOE≈-12、IV≈2.5 这种**假极端值**（看着像超强特征，其实是分母被兜底）。
      项目自带的 ``bin_table_processed`` 用 δ 归一化平滑，且显式标出
      「是否平滑」，让这类箱在表里可见 —— 属于「失败不可见」的反面。

    ★ 实际箱数仍可能少于请求值（重复值多 ⇒ 切点退化），
      所以表里单列「实际箱数」如实记录，不把请求值当事实写。
    """
    return BinSpec(
        n_bins=n_bins,
        min_bin_count=1,
        min_bin_pct=0.0,
        adaptive_bins=False,
        missing_as_bin=False,
        woe_smoothing=0.5,
    )


def _bin_sheet(data: ModelDocInput, work: pd.DataFrame) -> pd.DataFrame:
    cols = (
        "样本集", "箱数方案", "实际箱数", "箱", "总用户", "坏用户", "好用户",
        "坏样本率", "占比", "坏用户占比", "累计坏样本占比", "累计样本占比",
        "WOE", "IV", "逐箱KS", "逐箱LIFT", "是否平滑", "说明",
    )
    rows: list[dict[str, Any]] = []
    for set_name in [str(v) for v in work[data.split_col].dropna().unique()]:
        part = work[work[data.split_col] == set_name]
        scores = part["_proba"].reset_index(drop=True)
        y = pd.Series(part[data.label_col].to_numpy(dtype=float), name="y")
        n_bad = float(y.sum())
        n_good = float(len(y) - n_bad)
        overall_bad_rate = n_bad / len(y) if len(y) else 0.0
        for n_bins in BIN_NUMS:
            spec = _bin_spec(n_bins)
            fitted = fit_bins(scores, spec)
            codes = transform_bins(scores, fitted).reset_index(drop=True)
            try:
                table = bin_table_processed(codes, y, spec)
            except DataError as exc:
                # ★ 不静默少行：好坏样本不足时写一行说明，行数仍在
                blank = {c: None for c in cols}
                blank.update({
                    "样本集": set_name, "箱数方案": n_bins,
                    "说明": f"该样本集无法计算 WOE/IV：{exc}",
                })
                rows.append(blank)
                continue
            if table.empty:
                continue
            cum_bad = np.cumsum(table["坏用户"].to_numpy(dtype=float)) / n_bad if n_bad else 0.0
            cum_good = np.cumsum(table["好用户"].to_numpy(dtype=float)) / n_good if n_good else 0.0
            cum_n = np.cumsum(table["总用户"].to_numpy(dtype=float)) / len(part)
            for i, r in table.iterrows():
                rows.append({
                    "样本集": set_name,
                    "箱数方案": n_bins,
                    "实际箱数": len(table),
                    "箱": int(r["bin"]),
                    "总用户": int(r["总用户"]),
                    "坏用户": int(r["坏用户"]),
                    "好用户": int(r["好用户"]),
                    "坏样本率": round(float(r["坏样本率"]), 6),
                    "占比": round(float(r["占比"]), 6),
                    "坏用户占比": round(float(r["坏用户"]) / n_bad, 6) if n_bad else None,
                    "累计坏样本占比": round(float(cum_bad[i]), 6),
                    "累计样本占比": round(float(cum_n[i]), 6),
                    "WOE": round(float(r["WOE"]), 6),
                    "IV": round(float(r["IV贡献"]), 6),
                    "逐箱KS": round(float(abs(cum_bad[i] - cum_good[i])), 6),
                    "逐箱LIFT": (
                        round(float(r["坏样本率"]) / overall_bad_rate, 6)
                        if overall_bad_rate > 0 else None
                    ),
                    "是否平滑": bool(r["是否平滑"]),
                    "说明": "全好/全坏箱已做归一化平滑，WOE 参考值" if r["是否平滑"] else "",
                })
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# sheet 4：分组评估（只认弹窗拍板的维度列）
# --------------------------------------------------------------------------- #
def _group_sheet(data: ModelDocInput, work: pd.DataFrame) -> pd.DataFrame:
    cols = ("维度", "维度列", "分组", *_METRIC_COLS, "是否最差组")
    dims = [str(c) for c in data.dimensions if c]
    if not dims:
        # ★ sheet 必须存在；空也要写明「为什么空」，不能静默消失
        blank = {c: None for c in cols}
        blank.update({
            "维度": "（无）",
            "分组": "（无）",
            "说明": "「维度列确认」弹窗未声明维度类列 —— 分组评估无维度可分组。"
                    "如需逐组指标，请回退该弹窗填列名后重跑。",
        })
        return pd.DataFrame([blank], columns=cols)

    missing = [c for c in dims if c not in work.columns]
    if missing:
        raise DataError(f"维度列在编码后数据中不存在：{missing}（不做静默跳过）")

    rows: list[dict[str, Any]] = []
    for dim in dims:
        part_rows: list[dict[str, Any]] = []
        for value, part in work.groupby(dim, dropna=False, sort=False):
            row: dict[str, Any] = {"维度": dim, "维度列": dim, "分组": str(value)}
            row.update(_summarize(*_y_s(part, data.label_col)))
            part_rows.append(row)
        reliable = [r for r in part_rows if r.get("可靠") and r.get("KS") is not None]
        worst = min(reliable, key=lambda r: r["KS"])["分组"] if reliable else None
        for r in part_rows:
            r["是否最差组"] = worst is not None and r["分组"] == worst
        rows.extend(part_rows)
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# sheet 5：PSI（基期合并池 vs 逐月总 PSI，对模型概率分）
# --------------------------------------------------------------------------- #
def _psi_sheet(data: ModelDocInput, work: pd.DataFrame) -> pd.DataFrame:
    cols = ("基期", "对比月份", "PSI", "样本数", "说明")
    months_all = work[data.month_col].astype(str)
    base = [str(m) for m in data.base_months if m]
    base_label = data.base_label or (
        "基期池(" + "+".join(sorted(base, key=_year_month_key)) + ")" if base else ""
    )
    if not base:
        blank = {c: None for c in cols}
        blank["说明"] = "未拍板 PSI 基期月 —— 无法计算月度 PSI（不做静默兜底）。"
        return pd.DataFrame([blank], columns=cols)

    pool_mask = months_all.isin(base).to_numpy()
    if not pool_mask.any():
        blank = {c: None for c in cols}
        blank["说明"] = f"拍板的基期月 {base} 在数据中不存在，无法计算 PSI。"
        return pd.DataFrame([blank], columns=cols)

    # ★★ 切点只在基期池上拟合，各月共用 —— 各月各自切点会让 PSI 恒为 0
    all_scores = work["_proba"].reset_index(drop=True)
    fitted = fit_bins(all_scores[pool_mask].reset_index(drop=True), _bin_spec(10))
    codes = transform_bins(all_scores, fitted).reset_index(drop=True)
    codes = codes[codes.notna()].astype("int64")
    base_codes = codes[pool_mask]
    if base_codes.empty:
        blank = {c: None for c in cols}
        blank["说明"] = "基期池没有可分箱的分数，无法计算 PSI。"
        return pd.DataFrame([blank], columns=cols)

    idx = sorted(set(base_codes.unique().tolist()))
    b = (
        base_codes.value_counts(normalize=True)
        .reindex(idx, fill_value=0.0)
        .to_numpy(dtype=float)
    )
    b = np.clip(b, _EPS, None)
    b = b / b.sum()

    rows: list[dict[str, Any]] = []
    for month in sorted(months_all.unique().tolist(), key=_year_month_key):
        mask = (months_all == month).to_numpy()
        sub = codes[mask]
        if sub.empty:
            continue
        t = (
            sub.value_counts(normalize=True)
            .reindex(idx, fill_value=0.0)
            .to_numpy(dtype=float)
        )
        t = np.clip(t, _EPS, None)
        t = t / t.sum()
        psi = float(((t - b) * np.log(t / b)).sum())
        rows.append({
            "基期": base_label,
            "对比月份": month,
            "PSI": round(psi, 6),
            "样本数": int(len(sub)),
            "说明": (
                "基期月之一（已并入基期池；是与**池**对比，不是与自身对比，"
                "故 PSI 不一定为 0）" if month in base
                else "与基期池对比；切点只在池上拟合、各月共用"
            ),
        })
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# sheet 6：特征重要性（gain/split/占比 + IV + PSI + 特征画像）
# --------------------------------------------------------------------------- #
def _feature_profile(frame: pd.DataFrame, feature: str) -> dict[str, Any]:
    if feature not in frame.columns:
        return {c: None for c in _PROFILE_COLS}
    col = frame[feature]
    valid = pd.to_numeric(col, errors="coerce")
    n = len(col)
    found = int(valid.notna().sum())
    out: dict[str, Any] = {c: None for c in _PROFILE_COLS}
    out.update({
        "查得数": found,
        "查得率": round(found / n, 6) if n else None,
        "缺失率": round(1 - found / n, 6) if n else None,
        "取值数": int(col.nunique(dropna=True)),
    })
    if found == 0:
        return out
    v = valid.dropna().to_numpy(dtype=float)
    mean, std = float(v.mean()), float(v.std(ddof=0))
    out.update({
        "均值": round(mean, 6),
        "标准差": round(std, 6),
        "最小值": round(float(v.min()), 6),
        "最大值": round(float(v.max()), 6),
        "变异系数": round(std / mean, 6) if abs(mean) > _EPS else None,
    })
    for name, q in (("P1", .01), ("P5", .05), ("P25", .25), ("P50", .5),
                    ("P75", .75), ("P95", .95), ("P99", .99)):
        out[name] = round(float(np.quantile(v, q)), 6)
    return out


def _importance_sheet(data: ModelDocInput, frame: pd.DataFrame) -> pd.DataFrame:
    cols = (
        "特征", "IV", "请求箱数", "实际箱数", "箱数下调", "PSI",
        "gain", "split次数", "gain占比", *_PROFILE_COLS,
    )
    rows: list[dict[str, Any]] = []
    for item in data.importance or []:
        name = str(item.get("特征") or item.get("feature") or "")
        if not name:
            continue
        bins = data.iv_bins.get(name) or []
        requested = bins[0] if len(bins) > 0 else None
        actual = bins[1] if len(bins) > 1 else None
        row: dict[str, Any] = {
            "特征": name,
            "IV": data.iv_values.get(name),
            "请求箱数": requested,
            "实际箱数": actual,
            "箱数下调": (
                bool(requested and actual and actual < requested)
                if requested is not None and actual is not None else None
            ),
            "PSI": data.psi_metrics.get(name),
            "gain": item.get("gain"),
            "split次数": item.get("split次数"),
            "gain占比": item.get("gain占比"),
        }
        row.update(_feature_profile(frame, name))
        rows.append(row)
    if rows:
        rows.sort(key=lambda r: (-(r.get("gain占比") or 0), str(r["特征"])))
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def build_model_doc_sheets(data: ModelDocInput) -> dict[str, pd.DataFrame]:
    """构建固定 6 个 sheet。★ 任何分支下都返回 6 个 sheet（顺序即写出顺序）。"""
    work = _work_frame(data)
    sheets: dict[str, pd.DataFrame] = {
        "概览": _overview_sheet(data, work),
        "分月表现": _month_sheet(data, work),
        "分箱WOE": _bin_sheet(data, work),
        "分组评估": _group_sheet(data, work),
        "PSI": _psi_sheet(data, work),
        "特征重要性": _importance_sheet(data, data.frame),
    }
    assert tuple(sheets) == SHEET_NAMES  # noqa: S101 —— 结构契约，别调换/漏掉
    return sheets
