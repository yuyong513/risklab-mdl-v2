"""模型评价口径：二分类 / 多分类 OVR + macro、月度分数 PSI、KS/LIFT 分箱表。

★ 口径来源
----------
* 二分类单指标一律走 :mod:`risklab_mdl.core.metrics`（``calc_ks`` / ``calc_auc`` /
  ``calc_gini`` / ``calc_lift``）—— **不在本模块重算**：KS 的并列值整组推进、
  LIFT 的权重与尾部方向、AUC 的平均秩法都在那里定义，两处各写一份必然漂移；
* 多分类走 ``calc_multiclass_metrics``（OVR 逐类 + macro 算术平均），
  本模块只做「把编码标签翻译回原始类别名」和「子集口径」这两件事；
* 分数 PSI 用 ``core.binning`` 在**基期合并池**上拟合切点（各月共用同一套切点），
  理由见 :func:`monthly_psi` 的 docstring。

★ 子集指标为什么要带 status
--------------------------
val / test 很小的时候，某个子集可能只含一类标签，此时 KS/AUC **无定义**。
本模块**不返回 NaN 冒充满分值**，而是给 ``status="skipped"`` + 原因，
让报表与 confirm 闸门都能看到「这个数没算出来」。

★ KS/LIFT 表只吃 0/1 标签
------------------------
``ks_table`` / ``lift_table`` 的口径是「1=坏、0=好」。多分类的原始编码
（``0..k-1``）直接传进来时，``y==1`` 只认第二类 ⇒ 坏类整批被算成好样本，
表/图与 ``metrics.json`` 的 ``per_class/macro`` 口径错位（且错得看不出来）。
因此这里**显式校验**：非 0/1 标签直接 ``DataError``，调用方必须先
``y_bin = (y == bad_index).astype(int)``，并用 ``bad_label`` 把坏类写进表。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.errors import ConfigError, DataError
from ..core.metrics import calc_auc, calc_gini, calc_ks, calc_lift, calc_multiclass_metrics
from ..core.utils import round_finite

__all__ = [
    "DEFAULT_LIFT_PCT",
    "binary_metrics",
    "evaluate_subsets",
    "ks_table",
    "lift_table",
    "monthly_psi",
    "multiclass_metrics",
]

DEFAULT_LIFT_PCT = 0.10

#: PSI 里 0 占比的安全下限（与 core.psi 同口径：不把 0 当极小值再放大到 inf）。
_PSI_EPS = 1e-6


def _binary_labels(y_true: Any, *, context: str) -> np.ndarray:
    """校验 KS/LIFT 表的标签是 0/1（多分类必须先二值化，否则坏类被当好人）。"""
    y = np.asarray(y_true, dtype=float).ravel()
    if y.size and not np.isin(y, (0.0, 1.0)).all():
        raise DataError(
            f"{context} 只接受 0/1 二值标签（1=坏、0=好），收到取值 "
            f"{np.unique(y)[:5].tolist()}。多分类请先二值化："
            f"y_bin = (y == bad_index).astype(int) —— "
            f"否则坏类（编码 != 1）会被当成好样本，表/图与 per_class 口径错位"
        )
    return y


def binary_metrics(
    y_true: Any,
    prob: Any,
    *,
    lift_pct: float = DEFAULT_LIFT_PCT,
    min_seg_size: int = 30,
) -> Dict[str, Any]:
    """二分类指标（``y_true`` 为 0/1，``prob`` 为「坏」的概率）。"""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(prob, dtype=float).ravel()
    n_bad = int((y == 1).sum())
    n_good = int((y == 0).sum())
    base = {
        "n": int(y.size),
        "bad": n_bad,
        "good": n_good,
        "bad_rate": round_finite(n_bad / y.size) if y.size else None,
    }
    if n_bad == 0 or n_good == 0:
        return {
            **base,
            "status": "skipped",
            "reason": f"该子集标签只有一类（bad={n_bad}, good={n_good}），KS/AUC 无定义",
            "ks": None,
            "auc": None,
            "gini": None,
            "lift": None,
        }
    lift = calc_lift(y, p, pct=lift_pct, min_seg_size=min_seg_size)
    return {
        **base,
        "status": "ok",
        "reason": None,
        "ks": round_finite(calc_ks(y, p)),
        "auc": round_finite(calc_auc(y, p)),
        "gini": round_finite(calc_gini(y, p)),
        "lift": round_finite(lift.lift),
        "lift_pct": float(lift_pct),
        "lift_seg_size": int(lift.seg_size),
        "lift_reliable": bool(lift.is_reliable),
        "direction": lift.direction,
    }


def multiclass_metrics(
    y_true: Any,
    proba: Any,
    *,
    classes: Sequence[Any],
    bad_index: int,
    lift_pct: float = DEFAULT_LIFT_PCT,
    min_seg_size: int = 30,
) -> Dict[str, Any]:
    """多分类 OVR + macro（逐类明细按**原始类别名**呈现）。"""
    y = np.asarray(y_true).ravel()
    mat = np.asarray(proba, dtype=float)
    if mat.ndim != 2:
        raise DataError(f"多分类需要 (n, k) 概率矩阵，当前 {mat.shape}")
    if mat.shape[1] != len(classes):
        raise DataError(
            f"概率列数 {mat.shape[1]} 与类别数 {len(classes)} 不一致"
        )
    codes = list(range(len(classes)))
    result = calc_multiclass_metrics(
        y, mat, labels=codes, lift_pct=lift_pct, min_seg_size=min_seg_size
    )
    per_class: List[Dict[str, Any]] = []
    for item in result.per_class:
        record = item.to_dict()
        index = int(record["label"])
        record["class_index"] = index
        record["label"] = classes[index] if 0 <= index < len(classes) else record["label"]
        per_class.append(record)

    counts = {str(classes[i]): int((y == i).sum()) for i in range(len(classes))}
    macro = dict(result.macro)
    bad_class = classes[bad_index] if 0 <= bad_index < len(classes) else None
    bad_metrics = next(
        (r for r in per_class if r.get("class_index") == bad_index), None
    )
    return {
        "task": "multiclass",
        "status": "ok" if result.n_valid_classes else "skipped",
        "reason": None if result.n_valid_classes else "没有任何类别可计算 OVR 指标",
        "n": int(y.size),
        "n_classes": len(classes),
        "n_valid_classes": int(result.n_valid_classes),
        "class_counts": counts,
        # 顶层 ks/auc/gini/lift 取 macro（通用口径：一个 run 一个主指标）
        "ks": macro.get("ks"),
        "auc": macro.get("auc"),
        "gini": macro.get("gini"),
        "lift": macro.get("lift"),
        "macro": macro,
        "weighted": dict(result.weighted),
        "per_class": per_class,
        "bad_class": {
            "label": bad_class,
            "class_index": int(bad_index),
            "ks": (bad_metrics or {}).get("ks"),
            "auc": (bad_metrics or {}).get("auc"),
            "gini": (bad_metrics or {}).get("gini"),
            "lift": (bad_metrics or {}).get("lift"),
            "status": (bad_metrics or {}).get("status", "skipped"),
        },
        "notes": list(result.notes),
    }


def ks_table(
    y_true: Any,
    score: Any,
    *,
    n_bins: int = 10,
    bad_label: Any = None,
) -> pd.DataFrame:
    """KS 分箱表（按分数降序等频切，含累计好坏占比与 KS 列）。

    ``y_true`` 必须是 **0/1 二值标签**（1=坏）；多分类请先二值化，
    并把坏类名传进 ``bad_label``（会多一列 ``bad_class``，让表自己说明口径）。
    """
    y = _binary_labels(y_true, context="ks_table")
    s = np.asarray(score, dtype=float).ravel()
    columns = ["bin", "n", "bad", "good", "bad_rate", "cum_bad_rate",
               "cum_good_rate", "ks"]
    if bad_label is not None:
        columns.append("bad_class")
    if y.size != s.size:
        raise DataError(f"标签与分数长度不一致：{y.size} vs {s.size}")
    if y.size == 0:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame({"y": y, "s": s}).sort_values("s", ascending=False)
    n_bins = max(1, min(int(n_bins), len(frame)))
    frame["bin"] = np.floor(np.arange(len(frame)) * n_bins / len(frame)).astype(int) + 1
    grouped = frame.groupby("bin", sort=True)
    rows = []
    cum_bad = 0.0
    cum_good = 0.0
    total_bad = max(float((y == 1).sum()), 1.0)
    total_good = max(float((y == 0).sum()), 1.0)
    for bin_id, chunk in grouped:
        bad = float((chunk["y"] == 1).sum())
        good = float((chunk["y"] == 0).sum())
        cum_bad += bad
        cum_good += good
        row = {
            "bin": int(bin_id),
            "n": int(len(chunk)),
            "bad": int(bad),
            "good": int(good),
            "bad_rate": round_finite(bad / len(chunk)) if len(chunk) else None,
            "cum_bad_rate": round_finite(cum_bad / total_bad),
            "cum_good_rate": round_finite(cum_good / total_good),
            "ks": round_finite(abs(cum_bad / total_bad - cum_good / total_good)),
        }
        if bad_label is not None:
            row["bad_class"] = str(bad_label)
        rows.append(row)
    return pd.DataFrame(rows)


def lift_table(
    y_true: Any,
    score: Any,
    *,
    n_bins: int = 10,
    bad_label: Any = None,
) -> pd.DataFrame:
    """LIFT 分箱表（每个分箱的坏样本率与相对整体的提升倍数）。

    ``y_true`` 必须是 **0/1 二值标签**（1=坏），口径同 :func:`ks_table`。
    """
    y = _binary_labels(y_true, context="lift_table")
    s = np.asarray(score, dtype=float).ravel()
    columns = ["bin", "n", "bad", "bad_rate", "lift", "cum_lift"]
    if bad_label is not None:
        columns.append("bad_class")
    if y.size != s.size:
        raise DataError(f"标签与分数长度不一致：{y.size} vs {s.size}")
    if y.size == 0 or (y == 1).sum() == 0:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame({"y": y, "s": s}).sort_values("s", ascending=False)
    n_bins = max(1, min(int(n_bins), len(frame)))
    frame["bin"] = np.floor(np.arange(len(frame)) * n_bins / len(frame)).astype(int) + 1
    overall = float((y == 1).mean())
    rows = []
    cum_n = 0
    cum_bad = 0
    for bin_id, chunk in frame.groupby("bin", sort=True):
        bad = float((chunk["y"] == 1).sum())
        cum_n += len(chunk)
        cum_bad += bad
        row = {
            "bin": int(bin_id),
            "n": int(len(chunk)),
            "bad": int(bad),
            "bad_rate": round_finite(bad / len(chunk)) if len(chunk) else None,
            "lift": round_finite((bad / len(chunk)) / overall) if len(chunk) else None,
            "cum_lift": round_finite((cum_bad / cum_n) / overall) if cum_n else None,
        }
        if bad_label is not None:
            row["bad_class"] = str(bad_label)
        rows.append(row)
    return pd.DataFrame(rows)


def monthly_psi(
    scores: Any,
    months: Any,
    *,
    base_months: Optional[Sequence[str]] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """按月分数 PSI（**基期合并池**口径）。

    ★ 三个口径决定（缺一个 PSI 就不可比）
    ------------------------------------
    1. **切点在基期池上拟合，各月共用** —— 各月各自切点会让每箱占比都接近 1/n，
       PSI 恒为 0（看起来「很稳定」，实际什么都没测到）；
    2. **基期是「池」而不是单月** —— ``base_months`` 可以是多个月（默认最早一个月）：
       把基期样本合并起来估分布，基期的箱占比更稳；
    3. **零占比不放大** —— 某箱在某月为 0 时占比取 ``_PSI_EPS``（1e-6），
       与 ``core.psi`` 同口径；否则 ln(0) 会让 PSI 变成 inf。

    Returns
    -------
    dict
        ``{enabled, base_months, n_bins, by_month: [{month, psi, n}], max_psi,
        max_month, overall_psi, n_base, reason}``；以下情形一律
        ``enabled=False`` + 给 ``reason``（**不返回 0.0 冒充「很稳定」**）：

        * 月份不足 2 个（无跨期可比）；
        * 基期覆盖了**全部**月份（没有对比月，PSI 无定义 —— 不是「PSI=0」）；
        * ``base_months`` 里指定的月份在数据中不存在 ⇒ ``ConfigError``
          （**不静默回落到最早月**：那会让用户以为用的是他指定的基期）。
    """
    from ..core.binning import BinSpec, fit_bins, transform_bins

    s = pd.Series(np.asarray(scores, dtype=float).ravel())
    m = pd.Series(months).astype(str).reset_index(drop=True)
    if len(s) != len(m):
        raise DataError(f"分数与月份长度不一致：{len(s)} vs {len(m)}")
    available = sorted(m.dropna().unique().tolist())
    if len(available) < 2:
        return {
            "enabled": False,
            "reason": f"月份只有 {len(available)} 个（{available}）：无跨期可比，PSI 跳过",
            "base_months": [],
            "by_month": [],
            "max_psi": None,
            "max_month": None,
            "overall_psi": None,
            "n_base": 0,
            "n_bins": 0,
        }
    requested = [str(x) for x in base_months] if base_months else []
    if requested:
        missing = [month for month in requested if month not in available]
        if missing:
            raise ConfigError(
                f"指定的基期月份 {missing} 不在数据中（可用月份：{available}）。"
                f"不会静默回落到最早月 —— 那样报表会显示一个你没指定的基期。"
                f"检查 config.psi_base_months"
            )
        base = requested
    else:
        base = available[:1]
    base_mask = m.isin(base).to_numpy()
    if int(base_mask.sum()) == 0:  # pragma: no cover - requested 已在上面校验
        raise ConfigError(f"基期月份 {base} 在数据中不存在（可用月份：{available}）")

    spec = BinSpec(n_bins=int(n_bins))
    fitted = fit_bins(s[base_mask], spec)
    codes = np.asarray(transform_bins(s, fitted))
    n_bin = int(codes.max()) + 1 if codes.size else int(n_bins)
    base_counts = np.bincount(codes[base_mask], minlength=n_bin).astype(float)
    base_pct = np.where(
        base_counts / base_counts.sum() > 0,
        base_counts / base_counts.sum(),
        _PSI_EPS,
    )
    denom = int(base_mask.sum())

    rows: List[Dict[str, Any]] = []
    for month in available:
        if month in base:
            continue
        mask = (m == month).to_numpy()
        counts = np.bincount(codes[mask], minlength=n_bin).astype(float)
        pct = np.where(
            counts / max(counts.sum(), 1.0) > 0,
            counts / max(counts.sum(), 1.0),
            _PSI_EPS,
        )
        psi = float(np.sum((pct - base_pct) * np.log(pct / base_pct)))
        rows.append({"month": month, "psi": round_finite(psi), "n": int(mask.sum())})

    if not rows:
        # 基期覆盖全部月份：没有对比月 ⇒ PSI 无定义（不是「很稳定」）——
        # 这里必须返回 enabled=False，而不是让 max(rows) 抛 ValueError 变 config_invalid
        return {
            "enabled": False,
            "reason": (
                f"基期覆盖了全部 {len(available)} 个月份（{'、'.join(base)}）："
                f"没有对比月，PSI 无定义（不是 PSI=0）"
            ),
            "base_months": base,
            "by_month": [],
            "max_psi": None,
            "max_month": None,
            "overall_psi": None,
            "n_base": int(base_mask.sum()),
            "n_bins": int(n_bin),
        }

    compare_mask = ~base_mask
    counts_all = np.bincount(codes[compare_mask], minlength=n_bin).astype(float)
    pct_all = np.where(
        counts_all / max(counts_all.sum(), 1.0) > 0,
        counts_all / max(counts_all.sum(), 1.0),
        _PSI_EPS,
    )
    overall = float(np.sum((pct_all - base_pct) * np.log(pct_all / base_pct)))
    best = max(rows, key=lambda r: (r["psi"] if r["psi"] is not None else -1))
    return {
        "enabled": True,
        "reason": None,
        "base_months": base,
        "base_label": f"基期合并池（{'、'.join(base)}）",
        "n_base": int(denom),
        "n_bins": int(n_bin),
        "by_month": rows,
        "max_psi": best["psi"],
        "max_month": best["month"],
        "overall_psi": round_finite(overall),
    }


def evaluate_subsets(
    *,
    task: str,
    subsets: Mapping[str, tuple],
    classes: Sequence[Any],
    bad_index: int,
    lift_pct: float = DEFAULT_LIFT_PCT,
) -> Dict[str, Any]:
    """对多个子集逐一算指标。

    ``subsets`` 形如 ``{"train": (y_codes, proba_matrix), ...}``；
    返回 ``{"train": {...}, "val": {...}, "test": {...}}``（缺失的子集不出现）。
    """
    out: Dict[str, Any] = {}
    for name, payload in subsets.items():
        y, proba = payload
        if y is None or len(y) == 0:
            out[name] = {
                "status": "skipped",
                "reason": "该子集为空（0 行）",
                "n": 0,
                "ks": None,
                "auc": None,
                "gini": None,
                "lift": None,
            }
            continue
        if task == "binary":
            mat = np.asarray(proba, dtype=float)
            # 取「坏类」那一列（不是硬编码第 2 列）：classes 顺序万一不是
            # [good, bad]，硬编码会让分数方向反转（AUC 看着 0.5 左右、KS 失真）
            column = int(bad_index)
            if mat.ndim == 2 and mat.shape[1] > column:
                prob = mat[:, column]
            else:
                prob = mat.ravel()
            # 标签同样按 bad_index 二值化：二分类编码通常已是 [good, bad]，
            # 此时 bad_index=1 ⇒ y 原样；顺序被换过时不至于把好样本当坏样本
            y_bin = (np.asarray(y, dtype=float).ravel() == column).astype(int)
            out[name] = binary_metrics(y_bin, prob, lift_pct=lift_pct)
        else:
            out[name] = multiclass_metrics(
                y, proba, classes=classes, bad_index=bad_index, lift_pct=lift_pct
            )
    return out
