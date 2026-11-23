"""matplotlib Agg 图表函数（PNG / 120 dpi / 中文字体 fallback）。

三条硬约束
----------
1. **Agg backend**：`matplotlib.use("Agg", force=True)`，无 GUI 环境也能出图；
   强制后仍不是 Agg ⇒ 抛错（``doctor`` 的 matplotlib 探针同口径，不静默降级）。
2. **中文字体 fallback**：与 ``doctor.CHINESE_FONT_CANDIDATES`` 同一条链
   （Microsoft YaHei → SimHei → SimSun → DengXian → …）。缺字体只
   ``warnings`` 告警 + 退回 DejaVu Sans，**不报错**（R5/Q10）。
3. **宽表性能**：每个图表函数都收 ``max_*`` 上限，只画前若干个特征；
   调用方负责把「只画了前 N 个」写进 ``warnings``。

所有函数返回落盘路径（``Path``）；无数据可画时返回 ``None`` 由调用方决定
是否告警 —— 图表层不负责编造一张空图。
"""

from __future__ import annotations

import logging
import re
import warnings as _warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "PNG_DPI",
    "ChartRenderer",
    "sanitize_filename",
    "setup_matplotlib",
]

#: PNG 输出分辨率（契约 §3.2 冻结值）。
PNG_DPI = 120

#: 图表默认尺寸（英寸）；120 dpi ⇒ 约 960×600 px。
_FIG_SIZE = (8.0, 5.0)
_WIDE_FIG_SIZE = (10.0, 5.5)

#: 退化/缺字体等提示统一放这里（每次 ``setup_matplotlib`` 重置）。
_LAST_WARNINGS: List[str] = []


def _font_candidates() -> Sequence[str]:
    """与 doctor 共用同一条中文字体 fallback 链（导入失败时内置兜底）。"""
    try:
        from ..doctor import CHINESE_FONT_CANDIDATES

        return CHINESE_FONT_CANDIDATES
    except Exception:  # noqa: BLE001 - 字体链拿不到也不该让出图失败
        return (
            "Microsoft YaHei",
            "SimHei",
            "SimSun",
            "DengXian",
            "Noto Sans SC",
            "Arial Unicode MS",
        )


def sanitize_filename(name: str, *, fallback: str = "chart") -> str:
    """把特征名/文件名规范成安全文件名（保留字母数字下划线，其余换 ``_``）。"""
    cleaned = re.sub(r"[^0-9A-Za-z_\-]+", "_", str(name)).strip("_")
    return cleaned or fallback


def setup_matplotlib() -> Tuple[Any, Optional[str], List[str]]:
    """配置 Agg + 中文字体 fallback，返回 ``(pyplot, 命中字体名, warnings)``。"""
    del _LAST_WARNINGS[:]
    import matplotlib

    matplotlib.use("Agg", force=True)
    backend = str(matplotlib.get_backend())
    if backend.lower() != "agg":  # pragma: no cover - 环境异常才会走到
        raise RuntimeError(
            f"matplotlib backend 强制 Agg 后仍是 {backend!r}，无法在无 GUI 环境下出图"
        )
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    families = {f.name for f in font_manager.fontManager.ttflist}
    found = [name for name in _font_candidates() if name in families]
    if found:
        matplotlib.rcParams["font.sans-serif"] = list(found) + ["DejaVu Sans"]
        font: Optional[str] = found[0]
    else:
        font = None
        _LAST_WARNINGS.append(
            "未找到候选中文字体（Microsoft YaHei / SimHei / Noto Sans SC 等）："
            "图表中文可能显示为方块；安装字体（如 fonts-noto-cjk）后重跑即可，不影响数值结果"
        )
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["figure.max_open_warning"] = 0
    return plt, font, list(_LAST_WARNINGS)


class ChartRenderer:
    """一次分析会话复用的图表渲染器（持有 pyplot / 字体 / 告警）。"""

    def __init__(self, *, dpi: int = PNG_DPI) -> None:
        self.plt, self.font, self.warnings = setup_matplotlib()
        self.dpi = int(dpi)
        self.rendered: List[Path] = []

    # ---------- 基础设施 ----------

    def save(self, fig: Any, out_path: Path) -> Path:
        """保存 PNG（含缺字形告警捕获）并关闭 figure。"""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            # matplotlib 的缺字形提示走 logging，不是 warnings —— 两路都收
            logger = logging.getLogger("matplotlib.font_manager")
            handler_records: List[str] = []

            class _Collect(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
                    handler_records.append(record.getMessage())

            handler = _Collect(level=logging.WARNING)
            logger.addHandler(handler)
            try:
                fig.savefig(out_path, format="png", dpi=self.dpi, bbox_inches="tight")
            finally:
                logger.removeHandler(handler)
                self.plt.close(fig)
        missing_glyph = [str(w.message) for w in caught if "missing from font" in str(w.message)]
        missing_glyph += [m for m in handler_records if "missing from font" in m]
        if missing_glyph and not self.font:
            self._note_once("图表渲染时出现缺字形（中文字体不可用），图表仍已生成")
        self.rendered.append(out_path)
        return out_path

    def _note_once(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def _new_figure(self, size: Tuple[float, float] = _FIG_SIZE) -> Any:
        fig = self.plt.figure(figsize=size)
        return fig

    # ---------- 画像类图表 ----------

    def target_distribution(self, payload: Dict[str, Any], out_path: Path) -> Optional[Path]:
        """目标分布柱状图（二分类标注坏率，多分类标注各类占比）。"""
        counts = payload.get("counts") or {}
        shares = payload.get("shares") or {}
        if not counts:
            return None
        labels = [str(k) for k in counts]
        values = [int(counts[k]) for k in labels]
        names = payload.get("labels") or {}
        ticks = [f"{k}\n{names.get(k, '')}" if names.get(k) else str(k) for k in labels]

        fig = self._new_figure()
        ax = fig.add_subplot(1, 1, 1)
        bars = ax.bar(ticks, values, color="#4C78A8")
        top = max(values) if values else 1
        for bar, label in zip(bars, labels):
            ax.annotate(
                f"{int(bar.get_height())}\n（{float(shares.get(label, 0.0)):.2%}）",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_ylim(0, top * 1.18 if top else 1)
        task = payload.get("task")
        title = f"目标分布：{payload.get('column')}（任务 {task}）"
        if task == "binary" and payload.get("bad_rate") is not None:
            title += f"  坏率 {float(payload['bad_rate']):.2%}"
            ax.set_title(title + f"\n（坏标签 = {payload.get('bad_label')}）", fontsize=11)
        else:
            ax.set_title(title, fontsize=11)
        ax.set_ylabel("样本数")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        return self.save(fig, out_path)

    def missing_rate(
        self, missing_rows: Sequence[Dict[str, Any]], out_path: Path, *, top: int = 20
    ) -> Optional[Path]:
        """缺失率 TOP N 横向柱状图。"""
        rows = [r for r in missing_rows if float(r.get("missing_rate") or 0.0) > 0]
        if not rows:
            return None
        rows = sorted(rows, key=lambda r: float(r["missing_rate"]), reverse=True)[:top]
        rows = list(reversed(rows))
        names = [str(r["column"]) for r in rows]
        values = [float(r["missing_rate"]) for r in rows]

        fig = self._new_figure((8.0, max(3.0, 0.32 * len(rows) + 1.6)))
        ax = fig.add_subplot(1, 1, 1)
        ax.barh(names, values, color="#E45756")
        for idx, value in enumerate(values):
            ax.annotate(
                f"{value:.1%}",
                xy=(value, idx),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
        ax.set_xlim(0, min(1.0, max(values) * 1.25) if values else 1.0)
        ax.set_xlabel("缺失率")
        ax.set_title(f"缺失率 TOP {len(rows)}", fontsize=11)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        return self.save(fig, out_path)

    def feature_distributions(
        self,
        frame: pd.DataFrame,
        features: Sequence[str],
        out_path: Path,
        *,
        max_features: int = 6,
        random_state: int = 42,
    ) -> Optional[Path]:
        """抽样特征的取值分布（数值直方图 / 类别柱状图），最多 ``max_features`` 个。"""
        chosen = [f for f in list(features)[:max_features] if f in frame.columns]
        if not chosen:
            return None
        rows = int(np.ceil(len(chosen) / 2))
        fig = self._new_figure((10.0, 2.6 * rows + 0.8))
        for idx, name in enumerate(chosen, start=1):
            ax = fig.add_subplot(rows, 2, idx)
            series = frame[name]
            numeric = pd.to_numeric(series, errors="coerce")
            if float(numeric.notna().mean()) >= 0.95 and int(numeric.dropna().shape[0]) > 0:
                values = numeric.dropna()
                if int(values.nunique()) > 10:
                    ax.hist(values, bins=min(30, max(5, int(values.nunique()))), color="#4C78A8")
                else:
                    counts = values.value_counts().sort_index()
                    ax.bar([str(v) for v in counts.index], counts.values, color="#4C78A8")
                ax.set_ylabel("样本数", fontsize=8)
            else:
                counts = series.astype(str).value_counts().head(12)
                ax.bar([str(v) for v in counts.index], counts.values, color="#72B7B2")
                ax.tick_params(axis="x", rotation=45, labelsize=7)
                ax.set_ylabel("样本数", fontsize=8)
            ax.set_title(str(name), fontsize=9)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
        fig.suptitle("特征取值分布（抽样展示）", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return self.save(fig, out_path)

    def monthly_bad_rate(
        self, monthly: Sequence[Dict[str, Any]], out_path: Path
    ) -> Optional[Path]:
        """按月坏率趋势（柱=样本数，线=坏率）。"""
        rows = list(monthly or [])
        if not rows:
            return None
        months = [str(r["month"]) for r in rows]
        rates = [float(r.get("bad_rate") or 0.0) for r in rows]
        counts = [int(r.get("count") or 0) for r in rows]

        fig = self._new_figure(_WIDE_FIG_SIZE)
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(months, counts, color="#B0C4DE", label="样本数")
        ax.set_ylabel("样本数")
        ax.tick_params(axis="x", rotation=30)
        ax2 = ax.twinx()
        ax2.plot(months, rates, color="#E45756", marker="o", label="坏率")
        for month, rate in zip(months, rates):
            ax2.annotate(f"{rate:.2%}", xy=(month, rate), xytext=(0, 6), textcoords="offset points",
                         ha="center", fontsize=8, color="#E45756")
        ax2.set_ylabel("坏率")
        ax.set_title("按月坏率趋势", fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        return self.save(fig, out_path)

    # ---------- 特征分析类图表 ----------

    def iv_distribution(
        self,
        rows: Sequence[Dict[str, Any]],
        out_path: Path,
        *,
        top: int = 30,
        thresholds: Tuple[float, ...] = (0.02, 0.1, 0.3),
    ) -> Optional[Path]:
        """IV 分布（按 IV 降序的横向柱 + 阈值线）。"""
        usable = [
            (str(r["feature"]), float(r["iv"]))
            for r in rows
            if r.get("iv") is not None and not bool(r.get("skipped")) and np.isfinite(float(r["iv"]))
        ]
        if not usable:
            return None
        usable.sort(key=lambda item: item[1], reverse=True)
        shown = usable[:top]
        names = [name for name, _ in reversed(shown)]
        values = [value for _, value in reversed(shown)]

        fig = self._new_figure((8.0, max(3.0, 0.28 * len(shown) + 1.8)))
        ax = fig.add_subplot(1, 1, 1)
        ax.barh(names, values, color="#54A24B")
        for threshold in thresholds:
            ax.axvline(threshold, color="#E45756", linestyle="--", linewidth=0.9)
            ax.annotate(
                f"IV={threshold:g}",
                xy=(threshold, len(shown) - 0.5),
                xytext=(2, 0),
                textcoords="offset points",
                fontsize=7,
                color="#E45756",
                rotation=90,
                va="top",
            )
        ax.set_xlabel("IV")
        ax.set_title(f"特征 IV 分布（TOP {len(shown)} / 共 {len(usable)} 个可分析特征）", fontsize=11)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        return self.save(fig, out_path)

    def correlation_heatmap(
        self,
        frame: pd.DataFrame,
        features: Sequence[str],
        out_path: Path,
        *,
        max_features: int = 15,
    ) -> Optional[Path]:
        """相关性热力图（抽样特征；不足 2 个数值特征时不出图）。"""
        chosen = [f for f in list(features)[:max_features] if f in frame.columns]
        numeric = {}
        for name in chosen:
            values = pd.to_numeric(frame[name], errors="coerce")
            if float(values.notna().mean()) >= 0.5 and int(values.dropna().shape[0]) > 1:
                numeric[name] = values
        if len(numeric) < 2:
            return None
        corr = pd.DataFrame(numeric).corr()
        fig = self._new_figure((max(6.0, 0.5 * len(numeric) + 3.0),) * 2)
        ax = fig.add_subplot(1, 1, 1)
        image = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                value = float(corr.values[i, j])
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(value) > 0.6 else "black",
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"特征相关性热力图（抽样 {len(corr.columns)} 个特征）", fontsize=11)
        fig.tight_layout()
        return self.save(fig, out_path)

    def woe_bins(
        self,
        detail: pd.DataFrame,
        feature: str,
        out_path: Path,
        *,
        class_key: str = "binary",
    ) -> Optional[Path]:
        """单特征 WOE 箱图（柱=WOE，线=箱内坏样本率）。"""
        if detail is None or not len(detail):
            return None
        rows = detail[(detail["feature"] == feature) & (detail["class"] == class_key)]
        if not len(rows):
            return None
        labels = [str(v) for v in rows["bin_label"]]
        woe = [float(v) for v in rows["woe"]]
        rates = [float(v) for v in rows["bad_rate"]]

        fig = self._new_figure(_WIDE_FIG_SIZE)
        ax = fig.add_subplot(1, 1, 1)
        positions = list(range(len(labels)))
        ax.bar(positions, woe, color=["#E45756" if w > 0 else "#4C78A8" for w in woe])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("WOE")
        ax2 = ax.twinx()
        ax2.plot(positions, rates, color="#F58518", marker="o", linewidth=1.2, label="坏样本率")
        ax2.set_ylabel("坏样本率")
        title = f"{feature} 的 WOE 分箱（{class_key}）"
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        return self.save(fig, out_path)


def figure_paths(directory: Path, names: Iterable[str]) -> Dict[str, Path]:
    """把图名映射成 ``<directory>/<sanitize(name)>.png``。"""
    directory = Path(directory)
    return {name: directory / f"{sanitize_filename(name)}.png" for name in names}
