"""数据家族工具（契约 §3.1）：``project_init`` / ``dataset_load`` /
``dataset_columns`` / ``dataset_profile``。

三态返回协议（契约 §1）
-----------------------
每个工具都返回**一个** JSON 对象：``{ok, status, data, artifacts, warnings, error}``。
三态载荷由 :mod:`risklab_mdl.protocol` 统一构造；本模块只负责取数、落盘与结果组装。

★ 错误码补充
    ``tool-contract.md`` §1.2 的表格列的是主要场景，本实现补充了两个
    代码，都在 ``docs/data-feature.md`` 记录：
    * ``config_invalid`` —— 入参/config 结构或取值非法（如 features 里的列不存在）；
    * ``analysis_failed`` —— 分析过程中的非预期失败（兜底，绝不静默）。
    其余一律沿用契约既有代码（``path_not_readable`` / ``project_dir_not_found`` /
    ``manifest_not_found`` / ``encoding_undecodable`` / ``dependency_missing`` …）。

★ 失败必须可见
    任何降级（跳图、跳月份、抽样、箱数下调、字体缺失）都进 ``warnings``；
    拿不到目标字段就返回 ``needs_user_input``，不用猜测的目标硬算。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


from .. import __version__
from ..artifacts.layout import open_project, project_dir_for, resolve_project
from ..artifacts.manifest import (
    ensure_manifest,
    now_iso,
    read_manifest,
    stage_done,
    update_manifest,
)
from ..charts import ChartRenderer, build_report_html
from ..data.loader import (
    ROLE_CONFIRM_WARNING,
    column_infos,
    infer_roles,
    normalize_target,
    read_csv_auto,
    to_jsonable,
    write_csv_frame,
)
from ..data.profile import DEFAULT_MISSING_TOP, compute_profile, missing_summary_frame
from ..platform.fs import atomic_write_json, normalize_path
from ..project import manifest_roles, resolve_csv_path
from ..protocol import (
    relative_paths,
    result_from_exception,
    result_needs_input,
    result_ok,
    target_question,
)

__all__ = [
    "dataset_columns",
    "dataset_load",
    "dataset_profile",
    "project_init",
]

#: 画像图表默认上限（宽表策略）。
DEFAULT_MAX_FEATURES_FOR_CHARTS = 30


def _write_json(obj: Any, path: Path) -> Path:
    """原子写 JSON（经 :func:`to_jsonable` 归一，NaN/inf → null）。"""
    atomic_write_json(path, to_jsonable(obj))
    return path


# --------------------------------------------------------------------------- #
# 1. project_init
# --------------------------------------------------------------------------- #
def project_init(csv_path: str) -> Dict[str, Any]:
    """由 CSV 路径建/复用 ``mdl-<stem>/`` 与 manifest。"""
    try:
        target_dir = project_dir_for(csv_path)
        dir_existed = target_dir.is_dir()
        manifest_existed = (target_dir / "manifest.json").is_file()
        layout = open_project(csv_path)
        manifest = ensure_manifest(layout.project_dir, layout.csv_path or csv_path)
        warnings: List[str] = []
        if dir_existed:
            warnings.append(f"项目目录已存在，复用而不覆盖：{layout.project_dir}")
        else:
            warnings.append(f"已创建项目目录：{layout.project_dir}")
        if manifest_existed:
            warnings.append("已存在 manifest.json，保留原有状态（不重置已完成阶段与 runs）")
        artifacts = [] if manifest_existed else ["manifest.json"]
        return result_ok(
            {
                "project_dir": layout.project_dir.as_posix(),
                "csv_path": (layout.csv_path or normalize_path(csv_path)).as_posix(),
                "created": not dir_existed,
                "manifest_path": layout.manifest_path.as_posix(),
                "manifest": manifest,
            },
            artifacts=artifacts,
            warnings=warnings,
        )
    except Exception as exc:  # noqa: BLE001 - 统一走三态错误载荷
        return result_from_exception(exc)


# --------------------------------------------------------------------------- #
# 2. dataset_load
# --------------------------------------------------------------------------- #
def dataset_load(csv_path: str) -> Dict[str, Any]:
    """读 CSV：编码探测、列清单、dtype / 角色推断（不阻断）。"""
    try:
        layout = open_project(csv_path)
        # dataset_load 也能从零起项目（契约 §3.1：入参是 csv_path，出参带 project_dir），
        # 因此这里补 ensure_manifest —— 否则「没先调 project_init」会变成 manifest_not_found。
        ensure_manifest(layout.project_dir, layout.csv_path or csv_path)
        read = read_csv_auto(layout.csv_path or csv_path)
        frame = read.frame
        roles, reasons = infer_roles(frame)
        infos = column_infos(frame, roles)
        dtypes = {info["name"]: info["dtype"] for info in infos}
        numeric_features = [
            info["name"]
            for info in infos
            if info["role"] == "feature" and info["dtype"] in ("bool", "integer", "float")
        ]

        def _mutate(manifest: Dict[str, Any]) -> None:
            project = manifest.setdefault("project", {})
            project["csv_path"] = (layout.csv_path or normalize_path(csv_path)).as_posix()
            project["roles"] = {
                "id": roles.get("id"),
                "time": roles.get("time"),
                "split": roles.get("split"),
                "features": numeric_features,
            }

        manifest = update_manifest(layout.project_dir, _mutate)

        warnings: List[str] = list(read.warnings)
        warnings.append(ROLE_CONFIRM_WARNING)
        if not roles["target_candidates"]:
            warnings.append("未推断出目标字段候选：请在 config.target.column 显式指定标签列")
        if roles["time"] is None:
            warnings.append("未推断出时间列：与时间相关的分析与图表将被跳过（不伪造）")
        if read.rows == 0:
            warnings.append("CSV 没有数据行：后续画像/特征分析会因无样本而失败（已显式记录）")

        return result_ok(
            {
                "project_dir": layout.project_dir.as_posix(),
                "csv_path": read.path.as_posix(),
                "encoding": read.encoding,
                "rows": read.rows,
                "columns": read.columns,
                "column_count": len(read.columns),
                "roles": roles,
                "role_reasons": reasons,
                "dtypes": dtypes,
                "numeric_feature_count": len(numeric_features),
            },
            artifacts=["manifest.json"],
            warnings=warnings,
        )
    except Exception as exc:  # noqa: BLE001
        return result_from_exception(exc)


# --------------------------------------------------------------------------- #
# 3. dataset_columns
# --------------------------------------------------------------------------- #
def dataset_columns(project_dir: str) -> Dict[str, Any]:
    """列出列与角色（含 nunique / missing_rate），供用户确认或改角色。"""
    try:
        layout = resolve_project(project_dir)
        manifest = read_manifest(layout.project_dir)
        csv = resolve_csv_path(layout, manifest)
        read = read_csv_auto(csv)
        frame = read.frame
        roles = manifest_roles(manifest)
        if not any(roles.get(k) for k in ("id", "time", "split")):
            inferred, reasons = infer_roles(frame)
            roles = {key: inferred.get(key) for key in ("id", "time", "split")}
            roles["target_candidates"] = inferred.get("target_candidates", [])
        else:
            inferred, reasons = infer_roles(frame)
            roles["target_candidates"] = inferred.get("target_candidates", [])
        infos = column_infos(frame, roles)

        warnings: List[str] = list(read.warnings)
        confirmed_target = ((manifest.get("project") or {}).get("target") or {}).get("column")
        if not confirmed_target:
            warnings.append(ROLE_CONFIRM_WARNING)
        return result_ok(
            {
                "project_dir": layout.project_dir.as_posix(),
                "csv_path": csv.as_posix(),
                "encoding": read.encoding,
                "rows": read.rows,
                "columns": infos,
                "roles": roles,
                "role_reasons": reasons,
                "confirmed_target": confirmed_target,
            },
            warnings=warnings,
        )
    except Exception as exc:  # noqa: BLE001
        return result_from_exception(exc)


# --------------------------------------------------------------------------- #
# 4. dataset_profile
# --------------------------------------------------------------------------- #
def dataset_profile(
    project_dir: str, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """样本画像：目标分布 / 缺失汇总 / 描述统计 / 特征类型 / 月份交叉 + 图表。"""
    try:
        cfg = dict(config or {})
        layout = resolve_project(project_dir)
        manifest = read_manifest(layout.project_dir)
        csv = resolve_csv_path(layout, manifest)
        read = read_csv_auto(csv)
        frame = read.frame
        warnings: List[str] = list(read.warnings)

        target = normalize_target(cfg.get("target")) or normalize_target(
            (manifest.get("project") or {}).get("target")
        )
        if target is None or target["column"] not in frame.columns:
            reason = (
                "尚未确认目标字段。"
                if target is None
                else f"指定的目标列 {target['column']!r} 不在数据中。"
            )
            inferred, _ = infer_roles(frame)
            question, context = target_question(frame, inferred["target_candidates"], reason=reason)
            return result_needs_input(question, context, warnings=warnings)

        roles = manifest_roles(manifest)
        roles_from_config = bool(cfg.get("roles"))
        if roles_from_config:
            roles.update({k: v for k, v in (cfg["roles"] or {}).items() if k in ("id", "time", "split")})
        else:
            # 角色（id/time/split）来自推断或历史 manifest，不是用户当面确认的：
            # 必须显式告警，否则「按月交叉用了错的日期列」这类错误会静默发生。
            warnings.append(
                ROLE_CONFIRM_WARNING
                + f"；本次使用的角色 id={roles.get('id')!r} time={roles.get('time')!r} "
                f"split={roles.get('split')!r}，可用 config.roles 覆盖"
            )
        max_features = int(cfg.get("max_features_for_charts") or DEFAULT_MAX_FEATURES_FOR_CHARTS)
        sample_rows = cfg.get("sample_rows")
        sample_rows = int(sample_rows) if sample_rows else None

        result = compute_profile(
            frame,
            target=target,
            roles=roles,
            max_features_for_charts=max_features,
            sample_rows=sample_rows,
        )
        warnings.extend(result.warnings)

        analysis_dir = layout.subdir("data_analysis")
        charts_dir = layout.charts_dir("data_analysis")
        written: List[Path] = []

        profile_payload = {
            "schema": "risklab-mdl.profile/1",
            "version": __version__,
            "generated_at": now_iso(),
            "csv_path": csv.as_posix(),
            "encoding": read.encoding,
            **result.to_payload(),
        }
        written.append(_write_json(profile_payload, analysis_dir / "profile.json"))
        written.append(write_csv_frame(result.describe_frame.reset_index(), analysis_dir / "describe.csv"))
        written.append(
            write_csv_frame(
                result.target_distribution_frame,
                analysis_dir / "target_distribution.csv",
            )
        )

        # ---------- 图表（Agg + 中文字体 fallback，缺字体只告警） ----------
        renderer = ChartRenderer()
        chart_files: List[Path] = []
        chart = renderer.target_distribution(
            result.target_distribution, charts_dir / "target_distribution.png"
        )
        if chart:
            chart_files.append(chart)
        chart = renderer.missing_rate(
            result.missing_summary, charts_dir / "missing_rate_top.png", top=20
        )
        if chart:
            chart_files.append(chart)
        numeric_features = result.feature_types["columns"]["numeric"]
        if numeric_features:
            chart = renderer.feature_distributions(
                frame,
                numeric_features,
                charts_dir / "feature_distributions.png",
                max_features=6,
            )
            if chart:
                chart_files.append(chart)
        if result.monthly:
            chart = renderer.monthly_bad_rate(result.monthly, charts_dir / "monthly_bad_rate.png")
            if chart:
                chart_files.append(chart)
        warnings.extend(renderer.warnings)

        # ---------- 单文件 HTML（无外链） ----------
        sections: List[Dict[str, Any]] = []
        target_metrics = [
            {"label": "样本数", "value": f"{result.shape['rows']}"},
            {"label": "列数", "value": f"{result.shape['columns']}"},
        ]
        if result.target_distribution.get("bad_rate") is not None:
            target_metrics.append(
                {"label": "坏率", "value": f"{float(result.target_distribution['bad_rate']):.2%}"}
            )
        else:
            target_metrics.append(
                {
                    "label": "最大类占比",
                    "value": f"{float(result.target_distribution.get('majority_share') or 0):.2%}",
                }
            )
        target_metrics.append({"label": "图表数", "value": f"{len(chart_files)}"})
        sections.append(
            {
                "title": "样本概况",
                "metrics": target_metrics,
                "text": [
                    f"数据文件：{csv.name}（编码 {read.encoding}）",
                    f"目标字段：{result.target['column']}（任务 {result.target['task']}）",
                ],
            }
        )
        distribution_rows = [
            [
                row["label"],
                row.get("label_name", ""),
                row["count"],
                f"{float(row['share']):.2%}",
            ]
            for row in result.target_distribution_frame.to_dict(orient="records")
        ]
        sections.append(
            {
                "title": "目标分布",
                "table": {
                    "columns": ["标签", "含义", "样本数", "占比"],
                    "rows": distribution_rows,
                },
            }
        )
        if chart_files:
            sections.append(
                {
                    "title": "图表",
                    "charts": [
                        {"file": path.name, "caption": path.stem} for path in chart_files
                    ],
                }
            )
        missing_top = missing_summary_frame(result.columns).head(DEFAULT_MISSING_TOP)
        sections.append(
            {
                "title": f"缺失率 TOP {DEFAULT_MISSING_TOP}",
                "table": {
                    "columns": ["列名", "角色", "类型", "缺失率", "同值率", "取值数"],
                    "rows": [
                        [
                            row["column"],
                            row["role"],
                            row["dtype"],
                            f"{float(row['missing_rate']):.2%}",
                            f"{float(row['same_value_rate']):.2%}",
                            row["nunique"],
                        ]
                        for row in missing_top.to_dict(orient="records")
                    ],
                },
            }
        )
        counts = result.feature_types["counts"]
        sections.append(
            {
                "title": "特征类型构成",
                "table": {
                    "columns": ["类型", "数量"],
                    "rows": [[key, value] for key, value in counts.items()],
                },
            }
        )
        if result.monthly:
            sections.append(
                {
                    "title": "月份 × 标签",
                    "table": {
                        "columns": (
                            list(result.monthly_frame.columns)
                            if result.monthly_frame is not None
                            else []
                        ),
                        "rows": [
                            list(row.values())
                            for row in (result.monthly or [])
                        ],
                    },
                }
            )
        if warnings:
            sections.append({"title": "告警与口径说明", "text": list(warnings)})

        report = build_report_html(
            title=f"样本画像：{layout.stem}",
            subtitle=f"risklab-mdl-v2 v{__version__}",
            out_path=analysis_dir / "index.html",
            sections=sections,
            charts_rel="charts",
        )
        warnings.extend(report["warnings"])
        written.append(report["path"])
        if charts_dir.is_dir():
            written.extend(sorted(charts_dir.glob("*.png")))

        # ---------- manifest：记录确认的目标 + 阶段状态 ----------
        summary = (
            f"{result.shape['rows']} 行 × {result.shape['columns']} 列"
            + (
                f"，坏率 {float(result.target_distribution['bad_rate']):.2%}"
                if result.target_distribution.get("bad_rate") is not None
                else ""
            )
        )

        def _mutate(manifest_doc: Dict[str, Any]) -> None:
            project = manifest_doc.setdefault("project", {})
            project["target"] = {
                "column": result.target["column"],
                "task": result.target["task"],
                "labels": dict(result.target.get("labels") or {}),
            }
            roles_doc = project.setdefault("roles", {})
            for key in ("id", "time", "split"):
                if roles.get(key):
                    roles_doc[key] = roles[key]
            stage_done(
                manifest_doc,
                "data_analysis",
                done=True,
                path="data_analysis/",
                summary=summary,
            )

        update_manifest(layout.project_dir, _mutate)

        data = result.data_block(missing_top=DEFAULT_MISSING_TOP)
        data["chart_count"] = len(chart_files)
        data["project_dir"] = layout.project_dir.as_posix()
        return result_ok(
            data,
            artifacts=relative_paths(layout, written),
            warnings=warnings + [f"已写入 data_analysis/：{len(written)} 个文件"],
        )
    except Exception as exc:  # noqa: BLE001
        return result_from_exception(exc)
