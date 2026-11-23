"""过程家族工具（契约 §3.4）：``run_list`` / ``run_compare`` / ``report_build``。

★ 只读产物、不重算
------------------
三个工具都只读已有产物（``manifest.json`` / ``runs/*/metrics.json`` /
``runs/*/importance.csv`` / 各阶段 JSON），**不重新训练、不重算指标**。
理由：报告数字与确认闸门必须来自同一份 ``metrics.json`` ——
若报告自己再算一遍，两处口径稍有差异就会出现「报告说 KS 0.31、确认时 0.24」
的对不上，且谁也说不清哪份是对的。

★ 排序与「最优」
----------------
``run_list`` 按 ``run_id`` 升序（时间顺序）；``run_compare`` 用**主指标**
（默认 val KS，P2 未落盘时回落 train KS）标出 best。
KS 是风控确认闸门的主指标（``model_confirm`` 默认 ks_min=0.25），
所以「最优」与「能过闸门」用同一把尺子。
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from ..artifacts.layout import resolve_project
from ..artifacts.manifest import ManifestError, now_iso, read_manifest
from ..core.errors import ConfigError, ModelError
from ..data.loader import to_jsonable
from ..platform.fs import atomic_write_bytes, atomic_write_text, read_json
from ..protocol import (
    result_error,
    result_from_exception,
    result_ok,
)

__all__ = ["report_build", "run_compare", "run_list"]

REPORT_SCHEMA = "risklab-mdl.report/1"

#: 主指标（"最优" 与确认闸门共用同一把尺子）。
PRIMARY_METRIC = "ks"

#: 逐 run 对比表的指标列（顺序即 Excel 列顺序）。
COMPARE_FIELDS: tuple[str, ...] = ("ks", "auc", "gini", "lift")


def _runs_of(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    runs = [dict(entry) for entry in (manifest.get("runs") or []) if entry.get("run_id")]
    return sorted(runs, key=lambda entry: str(entry["run_id"]))


def _metrics_doc(layout, run_id: str) -> Optional[Mapping[str, Any]]:
    path = layout.run_dir(run_id) / "metrics.json"
    if not path.is_file():
        return None
    try:
        doc = read_json(path)
    except Exception:  # noqa: BLE001 - 损坏的产物必须可见，但不阻断列表
        return {"__error__": True}
    return doc if isinstance(doc, Mapping) else None


def _primary_value(metrics: Optional[Mapping[str, Any]]) -> Optional[float]:
    """主指标取值：val → train → manifest 摘要（找不到就返回 ``None``）。"""
    if not isinstance(metrics, Mapping):
        return None
    for subset in ("val", "train"):
        block = metrics.get(subset)
        if isinstance(block, Mapping) and block.get(PRIMARY_METRIC) is not None:
            return float(block[PRIMARY_METRIC])
    return None


# --------------------------------------------------------------------------- #
# run_list
# --------------------------------------------------------------------------- #
def run_list(project_dir: str) -> Dict[str, Any]:
    """列出 run 与指标摘要（按 run_id 升序）。"""
    try:
        layout = resolve_project(project_dir)
        manifest = read_manifest(layout.project_dir)
        runs = _runs_of(manifest)
        confirmed = manifest.get("confirmed_run")
        records: List[Dict[str, Any]] = []
        warnings: List[str] = []
        for entry in runs:
            run_id = str(entry["run_id"])
            run_dir = layout.run_dir(run_id)
            metrics = _metrics_doc(layout, run_id)
            if isinstance(metrics, Mapping) and metrics.get("__error__"):
                warnings.append(f"run {run_id} 的 metrics.json 损坏（无法解析）：重跑 model_evaluate")
                metrics = None
            summary = dict(entry.get("metrics") or {})
            block = {
                "run_id": run_id,
                "algorithm": entry.get("algorithm"),
                "task": entry.get("task"),
                "status": entry.get("status"),
                "metrics": summary,
                "created_at": entry.get("created_at"),
                "confirmed": confirmed == run_id,
                "has_model": (run_dir / "model.pkl").is_file(),
                "has_evaluation": (run_dir / "evaluation").is_dir(),
                "primary_ks": summary.get(PRIMARY_METRIC),
                "metrics_json": (
                    None
                    if not isinstance(metrics, Mapping)
                    else {
                        name: {
                            key: (metrics.get(name) or {}).get(key)
                            for key in COMPARE_FIELDS
                        }
                        for name in ("train", "val", "test")
                        if isinstance(metrics.get(name), Mapping)
                    }
                ),
            }
            records.append(block)
        if not records:
            warnings.append("项目里还没有 run：先 model_train 训练一个模型")
        return result_ok(
            {
                "project_dir": layout.project_dir.as_posix(),
                "runs": to_jsonable(records),
                "confirmed_run": confirmed,
                "n_runs": len(records),
            },
            warnings=warnings,
        )
    except ManifestError as exc:
        return result_from_exception(exc)
    except ConfigError as exc:
        return result_error("config_invalid", exc.message, "检查 project_dir")
    except Exception as exc:  # noqa: BLE001
        return result_from_exception(exc)


# --------------------------------------------------------------------------- #
# run_compare
# --------------------------------------------------------------------------- #
def _compare_rows(
    layout, runs: Sequence[Mapping[str, Any]]
) -> tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for entry in runs:
        run_id = str(entry["run_id"])
        metrics = _metrics_doc(layout, run_id)
        if isinstance(metrics, Mapping) and metrics.get("__error__"):
            warnings.append(f"run {run_id} 的 metrics.json 损坏：该行指标留空")
            metrics = None
        row: Dict[str, Any] = {
            "run_id": run_id,
            "algorithm": entry.get("algorithm"),
            "task": entry.get("task"),
            "status": entry.get("status"),
            "created_at": entry.get("created_at"),
        }
        for subset in ("train", "val", "test"):
            block = (metrics or {}).get(subset) if isinstance(metrics, Mapping) else None
            for field in COMPARE_FIELDS:
                row[f"{subset}_{field}"] = (
                    block.get(field) if isinstance(block, Mapping) else None
                )
            row[f"{subset}_n"] = (
                block.get("n") if isinstance(block, Mapping) else None
            )
        psi = (metrics or {}).get("psi") if isinstance(metrics, Mapping) else None
        row["psi"] = psi.get("max_psi") if isinstance(psi, Mapping) and psi.get("enabled") else None
        row["primary_ks"] = _primary_value(metrics)
        rows.append(row)
    return rows, warnings


def run_compare(
    project_dir: str, run_ids: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """对比多个 run 的指标并标出最优（产物 ``reports/run_compare.csv``）。"""
    try:
        layout = resolve_project(project_dir)
        manifest = read_manifest(layout.project_dir)
        all_runs = _runs_of(manifest)
        if not all_runs:
            return result_error(
                "run_not_found",
                "项目里还没有任何 run，无法对比",
                "先 model_train 训练至少一个模型",
            )
        if run_ids:
            wanted = [str(r) for r in run_ids]
            known = {str(entry["run_id"]) for entry in all_runs}
            unknown = [r for r in wanted if r not in known]
            if unknown:
                return result_error(
                    "run_not_found",
                    f"manifest 里没有这些 run：{unknown}",
                    f"可用 run：{sorted(known)}（用 run_list 查看）",
                )
            runs = [entry for entry in all_runs if str(entry["run_id"]) in set(wanted)]
            runs.sort(key=lambda entry: str(entry["run_id"]))
        else:
            runs = all_runs

        rows, warnings = _compare_rows(layout, runs)
        candidates = [row for row in rows if row.get("primary_ks") is not None]
        best_id = (
            max(candidates, key=lambda row: float(row["primary_ks"]))["run_id"]
            if candidates
            else None
        )
        if best_id is None:
            warnings.append(
                f"没有任何 run 有可用的 {PRIMARY_METRIC.upper()}（metrics.json 缺失或指标被跳过）："
                f"best 留空，不默认挑第一个"
            )
        for row in rows:
            row["best"] = row["run_id"] == best_id

        frame = pd.DataFrame(rows)
        reports_dir = layout.reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)
        csv_path = reports_dir / "run_compare.csv"
        atomic_write_text(
            csv_path, frame.to_csv(index=False, lineterminator="\n")
        )
        return result_ok(
            {
                "project_dir": layout.project_dir.as_posix(),
                "primary_metric": PRIMARY_METRIC,
                "best": best_id,
                "n_runs": len(rows),
                "rows": to_jsonable(rows),
                "compare_csv": layout.rel(csv_path),
            },
            artifacts=[layout.rel(csv_path)],
            warnings=warnings + [f"已写入 {layout.rel(csv_path)}"],
        )
    except ManifestError as exc:
        return result_from_exception(exc)
    except Exception as exc:  # noqa: BLE001
        return result_from_exception(exc)


# --------------------------------------------------------------------------- #
# report_build
# --------------------------------------------------------------------------- #
def _safe_read(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except Exception:  # noqa: BLE001 - 报告要能出，坏产物在 warnings 里说
        return None


def _overview_rows(layout, manifest: Mapping[str, Any], runs: Sequence[Mapping[str, Any]]) -> List[List[Any]]:
    project = manifest.get("project") or {}
    target = project.get("target") or {}
    roles = project.get("roles") or {}
    stages = manifest.get("stages") or {}
    rows: List[List[Any]] = [
        ["项目目录", layout.project_dir.as_posix()],
        ["数据文件", str(project.get("csv_path") or "")],
        ["项目标识", str(project.get("stem") or "")],
        ["创建时间", str(project.get("created_at") or "")],
        ["更新时间", str(project.get("updated_at") or "")],
        ["目标列", str(target.get("column") or "（未确认）")],
        ["任务类型", str(target.get("task") or "")],
        ["标签映射", str(target.get("labels") or {})],
        ["ID 列", str(roles.get("id") or "")],
        ["时间列", str(roles.get("time") or "")],
        ["划分列", str(roles.get("split") or "")],
        ["run 总数", len(runs)],
        ["确认 run", str(manifest.get("confirmed_run") or "（未确认）")],
        ["报告生成时间", now_iso()],
    ]
    for name, entry in stages.items():
        summary = (entry or {}).get("summary") if isinstance(entry, Mapping) else ""
        rows.append(
            [
                f"阶段 {name}",
                f"完成={bool((entry or {}).get('done'))}"
                + (f"；{summary}" if summary else ""),
            ]
        )
    return rows


def _metrics_rows(
    layout, runs: Sequence[Mapping[str, Any]]
) -> List[List[Any]]:
    header = [
        "run_id", "算法", "任务", "状态", "样本(子集)",
        "KS", "AUC", "Gini", "LIFT", "n", "bad", "bad_rate",
    ]
    rows: List[List[Any]] = [header]
    for entry in runs:
        run_id = str(entry["run_id"])
        metrics = _metrics_doc(layout, run_id)
        if not isinstance(metrics, Mapping):
            rows.append([run_id, entry.get("algorithm"), entry.get("task"),
                         entry.get("status"), "（缺 metrics.json）", None, None,
                         None, None, None, None, None])
            continue
        for subset in ("train", "val", "test"):
            block = metrics.get(subset)
            if not isinstance(block, Mapping):
                continue
            rows.append(
                [
                    run_id,
                    metrics.get("algorithm"),
                    metrics.get("task"),
                    entry.get("status"),
                    subset,
                    block.get("ks"),
                    block.get("auc"),
                    block.get("gini"),
                    block.get("lift"),
                    block.get("n"),
                    block.get("bad"),
                    block.get("bad_rate"),
                ]
            )
        multi = metrics.get("multi")
        if isinstance(multi, Mapping):
            for subset, payload in multi.items():
                macro = (payload or {}).get("macro") or {}
                rows.append(
                    [
                        run_id,
                        metrics.get("algorithm"),
                        f"{metrics.get('task')}:{subset}:macro",
                        entry.get("status"),
                        subset,
                        macro.get("ks"),
                        macro.get("auc"),
                        macro.get("gini"),
                        macro.get("lift"),
                        None,
                        None,
                        None,
                    ]
                )
    if len(rows) == 1:
        rows.append(["（暂无 run）", "", "", "", "", None, None, None, None, None, None, None])
    return rows


def _importance_rows(layout, runs: Sequence[Mapping[str, Any]], best_id: Optional[str]) -> List[List[Any]]:
    header = ["run_id", "特征", "重要性", "占比", "来源列"]
    rows: List[List[Any]] = [header]
    target_run = best_id or (str(runs[-1]["run_id"]) if runs else None)
    if target_run is None:
        rows.append(["（暂无 run）", "", None, None, ""])
        return rows
    path = layout.run_dir(target_run) / "importance.csv"
    if not path.is_file():
        rows.append([target_run, "（无 importance.csv）", None, None, ""])
        return rows
    try:
        frame = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        rows.append([target_run, "（importance.csv 无法解析）", None, None, ""])
        return rows
    value_column = next(
        (c for c in ("gain", "importance", "share") if c in frame.columns), None
    )
    if value_column is None:
        rows.append([target_run, "（importances 列缺失）", None, None, ""])
        return rows
    top = frame.sort_values(value_column, ascending=False).head(20)
    for _, record in top.iterrows():
        rows.append(
            [
                target_run,
                str(record.get("feature")),
                float(record[value_column]),
                float(record.get("share")) if "share" in frame.columns else None,
                value_column,
            ]
        )
    return rows


def _write_xlsx(path: Path, sheets: Mapping[str, List[List[Any]]]) -> None:
    """用 openpyxl 写多 sheet 工作簿（原子写：先写内存再落盘）。"""
    from openpyxl import Workbook
    from openpyxl.styles import Font

    workbook = Workbook()
    workbook.remove(workbook.active)
    for name, rows in sheets.items():
        sheet = workbook.create_sheet(title=name[:31])
        for row_index, row in enumerate(rows):
            sheet.append(list(row))
            if row_index == 0 and row:
                for cell in sheet[1]:
                    cell.font = Font(bold=True)
        if len(rows) <= 1:  # 只有表头 ⇒ 补一行说明，sheet 不得为空
            sheet.append(["（无数据）"] + [""] * (len(rows[0]) - 1 if rows else 0))
        sheet.freeze_panes = "A2"
    buffer = io.BytesIO()
    workbook.save(buffer)
    atomic_write_bytes(path, buffer.getvalue())


def _markdown_report(
    layout,
    manifest: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    *,
    best_id: Optional[str],
    warnings: Sequence[str],
) -> str:
    project = manifest.get("project") or {}
    target = project.get("target") or {}
    lines: List[str] = []
    lines.append("# 建模过程报告")
    lines.append("")
    lines.append(f"- 项目目录：`{layout.project_dir.as_posix()}`")
    lines.append(f"- 数据文件：`{project.get('csv_path')}`")
    lines.append(f"- 目标列：`{target.get('column')}`（任务：{target.get('task')}）")
    lines.append(f"- 报告生成时间：{now_iso()}")
    lines.append("")

    lines.append("## 一、流程阶段")
    lines.append("")
    lines.append("| 阶段 | 完成 | 摘要 |")
    lines.append("|---|---|---|")
    for name, entry in (manifest.get("stages") or {}).items():
        done = "✅" if (entry or {}).get("done") else "—"
        summary = (entry or {}).get("summary") or ""
        lines.append(f"| {name} | {done} | {summary} |")
    lines.append("")

    pre = _safe_read(layout.subdir("preprocess") / "summary.json")
    if isinstance(pre, Mapping):
        lines.append("## 二、预处理")
        lines.append("")
        lines.append(
            f"- 行数：{pre.get('rows_in')} → {pre.get('rows_out')}；"
            f"列数：{pre.get('columns_in')} → {pre.get('columns_out')}"
        )
        lines.append(f"- 处理数据：`{pre.get('processed_path')}`（{pre.get('processed_format')}）")
        lines.append(
            f"- 填充 {len(pre.get('missing') or [])} 列 / 编码 {len(pre.get('encode') or [])} 列 / "
            f"盖帽 {len(pre.get('cap') or [])} 列 / 分箱 {len(pre.get('binning') or [])} 列"
        )
        readiness = pre.get("model_readiness") or {}
        lines.append(
            f"- 入模矩阵就绪（LR 口径）：{readiness.get('lr_ready')}；"
            f"残留非数值列 {len(readiness.get('non_numeric_columns') or [])} 个"
        )
        lines.append("")

    split = _safe_read(layout.subdir("split") / "split.json")
    if isinstance(split, Mapping):
        lines.append("## 三、样本划分")
        lines.append("")
        spec = split.get("spec") or {}
        lines.append(f"- 方式：{spec.get('mode')}；比例 {spec.get('ratios')}；seed {spec.get('seed')}")
        lines.append(f"- 计数：{split.get('counts')}")
        lines.append(f"- 指纹：`{split.get('fingerprint')}`")
        lines.append("")

    refine = _safe_read(layout.subdir("feature_selection") / "refine.json")
    prescreen = _safe_read(layout.subdir("feature_selection") / "prescreen.json")
    if isinstance(refine, Mapping) or isinstance(prescreen, Mapping):
        lines.append("## 四、特征筛选")
        lines.append("")
        if isinstance(prescreen, Mapping):
            summary = prescreen.get("summary") or {}
            lines.append(
                f"- 初筛：{summary.get('total_input')} → {summary.get('n_survived')} 个特征"
            )
        if isinstance(refine, Mapping):
            lines.append(
                f"- 精筛：{refine.get('n_candidates')} 个候选 → {refine.get('n_selected')} 个保留"
                f"（{refine.get('disposition')}）"
            )
        lines.append("")

    lines.append("## 五、模型与指标")
    lines.append("")
    if not runs:
        lines.append("- 暂无训练记录。")
    else:
        lines.append("| run_id | 算法 | 任务 | 状态 | 子集 | KS | AUC | Gini | LIFT |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for entry in runs:
            run_id = str(entry["run_id"])
            metrics = _metrics_doc(layout, run_id)
            if not isinstance(metrics, Mapping):
                lines.append(
                    f"| {run_id} | {entry.get('algorithm')} | {entry.get('task')} | "
                    f"{entry.get('status')} | （缺 metrics.json） | | | | |"
                )
                continue
            for subset in ("train", "val", "test"):
                block = metrics.get(subset)
                if not isinstance(block, Mapping):
                    continue
                lines.append(
                    f"| {run_id} | {metrics.get('algorithm')} | {metrics.get('task')} | "
                    f"{entry.get('status')} | {subset} | {block.get('ks')} | "
                    f"{block.get('auc')} | {block.get('gini')} | {block.get('lift')} |"
                )
        if best_id:
            lines.append("")
            lines.append(f"- 主指标最优 run：`{best_id}`")
    lines.append("")

    lines.append("## 六、确认状态")
    lines.append("")
    confirmed = manifest.get("confirmed_run")
    lines.append(f"- confirmed_run：`{confirmed}`")
    confirm_doc = _safe_read(layout.final_dir / "confirm.json")
    if isinstance(confirm_doc, Mapping):
        lines.append(f"- 确认时间：{confirm_doc.get('confirmed_at')}")
        lines.append(f"- 阈值：{confirm_doc.get('thresholds')}")
        lines.append(f"- 指标：{confirm_doc.get('metrics')}")
    else:
        lines.append("- 尚未生成 `final/confirm.json`（未确认）。")
    lines.append("")

    if warnings:
        lines.append("## 七、构建提示")
        lines.append("")
        for item in warnings:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines) + "\n"


def report_build(project_dir: str) -> Dict[str, Any]:
    """汇总报告：``reports/结果汇总.xlsx``（多 sheet）+ ``reports/过程报告.md``。"""
    try:
        layout = resolve_project(project_dir)
        manifest = read_manifest(layout.project_dir)
        runs = _runs_of(manifest)
        warnings: List[str] = []
        if not runs:
            warnings.append("项目里还没有 run：报告的模型指标/特征重要性 sheet 只有占位说明")

        rows, compare_warnings = _compare_rows(layout, runs)
        warnings.extend(compare_warnings)
        candidates = [row for row in rows if row.get("primary_ks") is not None]
        best_id = (
            max(candidates, key=lambda row: float(row["primary_ks"]))["run_id"]
            if candidates
            else None
        )
        if manifest.get("confirmed_run"):
            best_id = str(manifest["confirmed_run"])

        sheets: Dict[str, List[List[Any]]] = {
            "概览": [["字段", "取值"]] + _overview_rows(layout, manifest, runs),
            "模型指标": _metrics_rows(layout, runs),
            "run对比": [list(rows[0].keys())] + [list(row.values()) for row in rows]
            if rows
            else [["run_id"], ["（暂无 run）"]],
            "特征重要性": _importance_rows(layout, runs, best_id),
        }
        reports_dir = layout.reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)
        xlsx_path = reports_dir / "结果汇总.xlsx"
        md_path = reports_dir / "过程报告.md"
        _write_xlsx(xlsx_path, sheets)
        atomic_write_text(
            md_path,
            _markdown_report(
                layout, manifest, runs, best_id=best_id, warnings=warnings
            ),
        )
        for name, rows_ in sheets.items():
            if len(rows_) <= 1:
                warnings.append(f"sheet「{name}」只有表头（对应产物缺失）")
        return result_ok(
            {
                "project_dir": layout.project_dir.as_posix(),
                "xlsx": layout.rel(xlsx_path),
                "markdown": layout.rel(md_path),
                "sheets": {name: len(rows_) for name, rows_ in sheets.items()},
                "n_runs": len(runs),
                "best_run": best_id,
                "schema": REPORT_SCHEMA,
            },
            artifacts=[layout.rel(xlsx_path), layout.rel(md_path)],
            warnings=warnings,
        )
    except ManifestError as exc:
        return result_from_exception(exc)
    except ModelError as exc:
        return result_error("analysis_failed", exc.message, "检查 run 产物完整性")
    except Exception as exc:  # noqa: BLE001
        return result_from_exception(exc)
