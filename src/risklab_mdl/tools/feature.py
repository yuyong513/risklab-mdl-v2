"""特征家族工具（契约 §3.2）：``feature_analyze`` / ``feature_charts``。

* ``feature_analyze`` —— 逐特征分箱 + WOE/IV 明细，落
  ``feature_analysis/{iv_table.csv, woe_detail.csv, binning/<feature>.json}``，
  并出 ``charts/*.png`` 与 ``feature_analysis/index.html``（单文件、无外链）。
* ``feature_charts`` —— 只重画图（IV 分布 / TOP 特征箱图 / 相关性热力图），
  读已有 ``iv_table.csv`` + ``woe_detail.csv``，**不重算** IV。

三态载荷构造器复用 ``tools/dataset.py`` 的实现（同一份协议，避免两处漂移）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .. import __version__
from ..artifacts.layout import resolve_project
from ..artifacts.manifest import now_iso, read_manifest, stage_done, update_manifest
from ..charts import ChartRenderer, build_report_html, sanitize_filename
from ..data.loader import normalize_target, read_csv_auto, to_jsonable, write_csv_frame
from ..features.analysis import BINNING_METHODS, analyze_features
from ..platform.fs import atomic_write_json
from ..project import resolve_csv_path, roles_for_analysis
from ..protocol import (
    relative_paths,
    result_error,
    result_from_exception,
    result_needs_input,
    result_ok,
    target_question,
)

__all__ = ["feature_analyze", "feature_charts"]

#: ``feature_analyze`` 出箱图的特征数上限（宽表性能策略）。
DEFAULT_WOE_CHARTS = 10

#: ``feature_charts`` 默认图数上限。
DEFAULT_CHARTS_MAX_FEATURES = 15


def _fail_missing_analysis(path: Path) -> Dict[str, Any]:
    return result_error(
        "analysis_failed",
        f"缺少特征分析产物：{path}",
        "先执行 feature_analyze（project_dir + config{method, bins}）再调用本工具",
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8", low_memory=False)


def _custom_edges(
    cfg: Dict[str, Any], features: Optional[Sequence[str]]
) -> Optional[Dict[str, Sequence[float]]]:
    """``config.edges``：``{feature: [切点…]}``；单特征时也接受裸列表。"""
    raw = cfg.get("edges")
    if raw is None:
        return None
    if isinstance(raw, dict):
        out: Dict[str, Sequence[float]] = {}
        for key, value in raw.items():
            if not isinstance(value, (list, tuple)) or not len(value):
                raise ValueError(f"config.edges[{key!r}] 必须是非空的切点列表")
            out[str(key)] = [float(v) for v in value]
        return out
    if isinstance(raw, (list, tuple)):
        values = [float(v) for v in raw]
        if not values:
            raise ValueError("config.edges 为空：method='custom' 至少需要一个切点")
        if features is not None and len(features) == 1:
            return {str(features[0]): values}
        raise ValueError(
            "config.edges 传裸列表时只能分析单个特征；多特征请用 {feature: [切点…]} 形式"
        )
    raise ValueError("config.edges 必须是 {feature: [切点…]} 或单个切点列表")


# --------------------------------------------------------------------------- #
# 1. feature_analyze
# --------------------------------------------------------------------------- #
def feature_analyze(
    project_dir: str, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """逐特征分箱 + WOE/IV（等频 / 等宽 / DT / 自定义）。"""
    try:
        cfg = dict(config or {})
        layout = resolve_project(project_dir)
        manifest = read_manifest(layout.project_dir)
        csv = resolve_csv_path(layout, manifest)

        method = str(cfg.get("method") or "quantile")
        if method not in BINNING_METHODS:
            return result_error(
                "config_invalid",
                f"不支持的分箱方法：{method!r}",
                f"可用方法：{', '.join(BINNING_METHODS)}（见 docs/data-feature.md）",
            )
        bins = int(cfg.get("bins") or 8)
        max_features_for_charts = int(cfg.get("max_features_for_charts") or 30)

        target = normalize_target(cfg.get("target")) or normalize_target(
            (manifest.get("project") or {}).get("target")
        )

        # 先读表头做列名校验（宽表下避免整表读入再报错）
        header = read_csv_auto(csv, nrows=0).frame
        all_columns = [str(c) for c in header.columns]
        features = cfg.get("features")
        if features is not None:
            features = [str(f) for f in features]
            missing = [f for f in features if f not in all_columns]
            if missing:
                return result_error(
                    "config_invalid",
                    f"config.features 中的列不存在：{missing[:10]}",
                    f"可用列示例：{all_columns[:10]}（共 {len(all_columns)} 列）",
                )
            if not features:
                return result_error(
                    "features_empty", "config.features 是空列表", "留空（不传）表示分析全部特征"
                )

        if target is None or target["column"] not in all_columns:
            reason = (
                "尚未确认目标字段。"
                if target is None
                else f"指定的目标列 {target['column']!r} 不在数据中。"
            )
            frame_head = read_csv_auto(csv, nrows=200).frame
            from ..data.loader import infer_roles

            inferred, _ = infer_roles(frame_head)
            question, context = target_question(frame_head, inferred["target_candidates"], reason=reason)
            return result_needs_input(question, context)

        usecols: Optional[List[str]] = None
        if features is not None:
            usecols = sorted({target["column"], *features})
        read = read_csv_auto(csv, usecols=usecols)
        frame = read.frame
        warnings: List[str] = list(read.warnings)

        roles = roles_for_analysis(manifest, frame)
        edges_map = _custom_edges(cfg, features)
        if method == "custom" and not edges_map:
            return result_error(
                "config_invalid",
                "method='custom' 需要在 config.edges 中给出切点",
                '形如 {"f_0001": [0.5, 1.2]}；单个特征时也可传 [0.5, 1.2]',
            )

        result = analyze_features(
            frame,
            target=target,
            method=method,
            bins=bins,
            features=features,
            roles=roles,
            custom_edges=edges_map,
            max_features_for_charts=max_features_for_charts,
        )
        warnings.extend(result.warnings)

        analysis_dir = layout.subdir("feature_analysis")
        charts_dir = layout.charts_dir("feature_analysis")
        binning_dir = analysis_dir / "binning"
        binning_dir.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []

        written.append(write_csv_frame(result.iv_table, analysis_dir / "iv_table.csv"))
        written.append(write_csv_frame(result.woe_detail, analysis_dir / "woe_detail.csv"))
        for name, spec in result.bin_specs.items():
            path = binning_dir / f"{sanitize_filename(name, fallback='feature')}.json"
            atomic_write_json(
                path,
                to_jsonable(
                    {
                        "schema": "risklab-mdl.binspec/1",
                        "generated_at": now_iso(),
                        **spec,
                    }
                ),
            )
            written.append(path)

        # ---------- 图表：IV 分布 + TOP N 单特征箱图 ----------
        renderer = ChartRenderer()
        chart_files: List[Path] = []
        chart = renderer.iv_distribution(
            result.iv_table.to_dict(orient="records"), charts_dir / "iv_distribution.png"
        )
        if chart:
            chart_files.append(chart)
        top_features = [row["feature"] for row in result.iv_top(DEFAULT_WOE_CHARTS)]
        for name in top_features:
            chart = renderer.woe_bins(
                result.woe_detail,
                name,
                charts_dir / f"woe_{sanitize_filename(name, fallback='feature')}.png",
            )
            if chart:
                chart_files.append(chart)
        warnings.extend(renderer.warnings)
        if result.iv_table["skipped"].eq(False).sum() > len(top_features):
            warnings.append(
                f"只对 IV 最高的 {len(top_features)} 个特征出了箱图，其余仅出表"
                "（max_features_for_charts / 宽表性能策略）"
            )

        # ---------- 单文件 HTML ----------
        usable = result.iv_table[~result.iv_table["skipped"].astype(bool)]
        sections: List[Dict[str, Any]] = [
            {
                "title": "概览",
                "metrics": [
                    {"label": "特征总数", "value": f"{result.feature_count}"},
                    {"label": "可分析", "value": f"{int(len(usable))}"},
                    {"label": "跳过", "value": f"{len(result.skipped)}"},
                    {"label": "图表数", "value": f"{len(chart_files)}"},
                ],
                "text": [
                    f"目标字段：{result.target['column']}（任务 {result.target['task']}）",
                    f"分箱方法：{method}，请求箱数：{bins}",
                ],
            },
            {
                "title": "IV TOP 20",
                "table": {
                    "columns": ["特征", "IV", "有效箱数", "拟合箱数", "缺失率"],
                    "rows": [
                        [
                            str(row["feature"]),
                            f"{float(row['iv']):.4f}",
                            int(row["n_bins"]),
                            int(row["n_bins_fitted"]),
                            f"{float(row['missing_rate']):.2%}",
                        ]
                        for row in usable.head(20).to_dict(orient="records")
                    ],
                },
            },
        ]
        if chart_files:
            sections.append(
                {
                    "title": "图表",
                    "charts": [{"file": p.name, "caption": p.stem} for p in chart_files],
                }
            )
        if result.skipped:
            sections.append(
                {
                    "title": "跳过的特征",
                    "table": {
                        "columns": ["特征", "原因"],
                        "rows": [[row["feature"], row["reason"]] for row in result.skipped],
                    },
                }
            )
        if warnings:
            sections.append({"title": "告警与口径说明", "text": list(warnings)})

        report = build_report_html(
            title=f"特征分析：{layout.stem}",
            subtitle=f"risklab-mdl-v2 v{__version__} ｜ method={method} ｜ bins={bins}",
            out_path=analysis_dir / "index.html",
            sections=sections,
            charts_rel="charts",
        )
        warnings.extend(report["warnings"])
        written.append(report["path"])
        written.extend(chart_files)

        summary = (
            f"{result.feature_count} 个特征（可分析 {int(len(usable))}），"
            f"method={method} bins={bins}，图表 {len(chart_files)} 张"
        )

        def _mutate(manifest_doc: Dict[str, Any]) -> None:
            project = manifest_doc.setdefault("project", {})
            project["target"] = {
                "column": result.target["column"],
                "task": result.target["task"],
                "labels": dict(result.target.get("labels") or {}),
            }
            stage_done(
                manifest_doc,
                "feature_analysis",
                done=True,
                path="feature_analysis/",
                summary=summary,
            )

        update_manifest(layout.project_dir, _mutate)

        data = result.data_block(top=20)
        data["method"] = method
        data["bins"] = bins
        data["chart_count"] = len(chart_files)
        data["project_dir"] = layout.project_dir.as_posix()
        return result_ok(data, artifacts=relative_paths(layout, written), warnings=warnings)
    except Exception as exc:  # noqa: BLE001
        return result_from_exception(exc)


# --------------------------------------------------------------------------- #
# 2. feature_charts
# --------------------------------------------------------------------------- #
def feature_charts(
    project_dir: str, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """特征图表：IV 分布（含阈值线）/ TOP 特征 WOE 箱图 / 相关性热力图。"""
    try:
        cfg = dict(config or {})
        layout = resolve_project(project_dir)
        manifest = read_manifest(layout.project_dir)
        analysis_dir = layout.subdir("feature_analysis")
        iv_path = analysis_dir / "iv_table.csv"
        woe_path = analysis_dir / "woe_detail.csv"
        for path in (iv_path, woe_path):
            if not path.is_file():
                return _fail_missing_analysis(path)

        iv_table = _read_csv(iv_path)
        woe_detail = _read_csv(woe_path)
        if not len(iv_table):
            return result_error(
                "features_empty", "iv_table.csv 为空", "确认 feature_analyze 的 features 入参"
            )

        requested = cfg.get("features")
        if requested:
            requested = [str(f) for f in requested]
            unknown = [f for f in requested if f not in set(iv_table["feature"].astype(str))]
            if unknown:
                return result_error(
                    "config_invalid",
                    f"config.features 不在已分析的特征里：{unknown[:10]}",
                    "先对这批特征跑 feature_analyze，再调用 feature_charts",
                )
        max_features = int(cfg.get("max_features") or DEFAULT_CHARTS_MAX_FEATURES)

        usable = iv_table[~iv_table["skipped"].astype(bool)].copy()
        usable = usable.sort_values("iv", ascending=False)
        pool = requested or [str(f) for f in usable["feature"].head(max_features)]
        pool = [f for f in pool if f in set(usable["feature"].astype(str))][:max_features]

        csv = resolve_csv_path(layout, manifest)
        frame_for_corr: Optional[pd.DataFrame] = None
        if pool:
            try:
                frame_for_corr = read_csv_auto(csv, usecols=pool).frame
            except Exception:  # noqa: BLE001 - 相关性图是增强项，读不到不该整步失败
                frame_for_corr = None

        charts_dir = layout.charts_dir("feature_analysis")
        renderer = ChartRenderer()
        warnings: List[str] = list(renderer.warnings)
        chart_files: List[Path] = []
        chart = renderer.iv_distribution(
            iv_table.to_dict(orient="records"), charts_dir / "iv_distribution.png"
        )
        if chart:
            chart_files.append(chart)
        if frame_for_corr is not None:
            chart = renderer.correlation_heatmap(
                frame_for_corr, pool, charts_dir / "correlation_heatmap.png",
                max_features=max_features,
            )
            if chart:
                chart_files.append(chart)
            else:
                warnings.append("可用数值特征不足 2 个，已跳过相关性热力图（不伪造）")
        else:
            warnings.append("相关数据读不到，已跳过相关性热力图")
        for name in pool:
            chart = renderer.woe_bins(
                woe_detail,
                name,
                charts_dir / f"woe_{sanitize_filename(name, fallback='feature')}.png",
            )
            if chart:
                chart_files.append(chart)
        if len(usable) > len(pool):
            warnings.append(
                f"共 {len(usable)} 个可分析特征，本次只画了 {len(pool)} 个"
                f"（max_features={max_features}）；可用 config.features 指定要画的特征"
            )

        sections: List[Dict[str, Any]] = [
            {
                "title": "概览",
                "metrics": [
                    {"label": "可分析特征", "value": f"{len(usable)}"},
                    {"label": "本次出图特征", "value": f"{len(pool)}"},
                    {"label": "图表数", "value": f"{len(chart_files)}"},
                ],
            }
        ]
        if chart_files:
            sections.append(
                {
                    "title": "图表",
                    "charts": [{"file": p.name, "caption": p.stem} for p in chart_files],
                }
            )
        if warnings:
            sections.append({"title": "告警", "text": list(warnings)})
        report = build_report_html(
            title=f"特征图表：{layout.stem}",
            subtitle=f"risklab-mdl-v2 v{__version__}",
            out_path=analysis_dir / "index.html",
            sections=sections,
            charts_rel="charts",
        )
        warnings.extend(report["warnings"])

        written: List[Path] = [report["path"], *chart_files]
        return result_ok(
            {
                "project_dir": layout.project_dir.as_posix(),
                "chart_count": len(chart_files),
                "features": pool,
                "charts": [p.name for p in chart_files],
            },
            artifacts=relative_paths(layout, written),
            warnings=warnings,
        )
    except Exception as exc:  # noqa: BLE001
        return result_from_exception(exc)
