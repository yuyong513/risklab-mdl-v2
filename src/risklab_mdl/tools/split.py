"""划分家族工具（契约 §3.3）：``dataset_split``。

四种划分方式（``mode``）::

    none        不划分（全部算训练/评估，报告标注「指标不代表样本外表现」）
    random      随机抽样
    stratified  按 stratify_by（缺省用目标列）分层抽样
    field       按字段表达式划分（expr_train / expr_val / expr_test）

★ 口径与可复现
    划分是**纯函数**：结果只由 ``(数据, spec)`` 决定，同 seed 同输入 ⇒ 逐行相同的
    mask、相同的指纹（``core.split.split_frame``）。``split/split.json`` 落盘
    spec + 计数 + 实际占比 + 指纹 + seed，模型层可直接用 ``SplitSpec.from_config``
    复算同一份掩码，不需要再猜一遍参数。

★ 表达式是**受限解析器**（``core.expr``），不使用 ``eval``
    ``__import__`` / 属性访问（``a.b``）/ 赋值（``a = 1``）/ 函数调用 / ``;`` 等
    一律 ``ConfigError``，本层映射成 ``error.code=expression_invalid`` 并给示例。
    子集重叠、引用了不存在的列同样按表达式问题报错（它们无法算出合法划分）。

★ 失败必须可见
    模式/比例/种子非法 ⇒ ``config_invalid``；表达式非法 ⇒ ``expression_invalid``；
    被表达式排除的行、分层过细、mode=none 的口径局限一律进 ``warnings``。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from .. import __version__
from ..artifacts.layout import resolve_project
from ..artifacts.manifest import now_iso, read_manifest, stage_done, update_manifest
from ..core.errors import ConfigError, DataError
from ..core.expr import compile_expression
from ..core.split import SplitSpec, split_frame
from ..data.loader import normalize_target, read_csv_auto, to_jsonable
from ..platform.fs import atomic_write_json
from ..project import resolve_csv_path
from ..protocol import (
    relative_paths,
    result_error,
    result_from_exception,
    result_ok,
)

__all__ = ["dataset_split"]

#: 表达式非法时的修复提示（与 ``core.expr`` 的支持语法一致）。
EXPRESSION_HINT = (
    "示例：month in ('2026-01','2026-02') / amount >= 1000 / "
    "(channel == 'app' | channel == 'web') & score != null；"
    "只支持 == != > >= < <= in / not in / & / | / 括号 / 数值与字符串字面量，"
    "禁止函数调用、属性访问（.）、赋值（=）、分号与下标"
)

#: ``split/split.json`` 的 schema 标识。
SPLIT_SCHEMA = "risklab-mdl.split/1"


def _label_distribution(
    frame: pd.DataFrame, masks: Mapping[str, Any], target_column: Optional[str]
) -> Optional[Dict[str, Any]]:
    """各子集的标签分布（验收「分层后各类占比偏差」用）。"""
    if not target_column or target_column not in frame.columns:
        return None
    out: Dict[str, Any] = {}
    for name in ("train", "val", "test"):
        mask = masks.get(name)
        if mask is None:
            continue
        labels = frame.loc[mask, target_column].astype(str)
        total = int(len(labels))
        counts = {str(k): int(v) for k, v in labels.value_counts(dropna=False).items()}
        out[name] = {
            "count": total,
            "counts": counts,
            "shares": {
                label: round(count / total, 6) if total else 0.0
                for label, count in counts.items()
            },
        }
    return out


def dataset_split(
    project_dir: str, config: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """样本划分：mode ∈ {none, random, stratified, field}，落 ``split/split.json``。"""
    try:
        cfg = dict(config or {})
        layout = resolve_project(project_dir)
        manifest = read_manifest(layout.project_dir)
        csv = resolve_csv_path(layout, manifest)

        target = normalize_target(cfg.get("target")) or normalize_target(
            (manifest.get("project") or {}).get("target")
        )

        # ---------- ① 规格构造（mode / ratios / seed / expression 形态） ----------
        try:
            spec = SplitSpec.from_config(cfg)
        except ConfigError as exc:
            return result_error(
                "config_invalid",
                exc.message,
                "mode ∈ none/random/stratified/field；ratios 合计必须为 1；"
                "见 docs/preprocess-split.md",
            )

        header = read_csv_auto(csv, nrows=0).frame
        all_columns = [str(c) for c in header.columns]
        target_column = target["column"] if target else None
        if target_column and target_column not in all_columns:
            return result_error(
                "config_invalid",
                f"目标列 {target_column!r} 不在数据中",
                f"可用列示例：{all_columns[:10]}（共 {len(all_columns)} 列）",
            )

        needed: set[str] = set()
        if target_column:
            needed.add(target_column)
        if spec.stratify_by:
            if spec.stratify_by not in all_columns:
                return result_error(
                    "config_invalid",
                    f"config.stratify_by 指定的分层列不存在：{spec.stratify_by!r}",
                    f"可用列示例：{all_columns[:20]}",
                )
            needed.add(spec.stratify_by)
        if spec.time_column:
            if spec.time_column not in all_columns:
                return result_error(
                    "config_invalid",
                    f"config.time_column 指定的时间列不存在：{spec.time_column!r}",
                    f"可用列示例：{all_columns[:20]}",
                )
            needed.add(spec.time_column)

        if spec.mode == "field":
            if not spec.expression:
                return result_error(
                    "config_invalid",
                    "mode='field' 需要 field 里的 expr_train / expr_val / expr_test",
                    "形如 {\"field\": {\"column\": \"month\", \"expr_train\": \"month in ('2026-01')\"}}",
                )
            for name, text in spec.expression.items():
                try:
                    expr = compile_expression(str(text))
                except ConfigError as exc:
                    return result_error("expression_invalid", exc.message, EXPRESSION_HINT)
                unknown = [c for c in expr.columns if c not in all_columns]
                if unknown:
                    return result_error(
                        "expression_invalid",
                        f"{name} 表达式引用了不存在的列：{unknown[:10]}",
                        f"可用列示例：{all_columns[:20]}；{EXPRESSION_HINT}",
                    )
                needed |= set(expr.columns)

        usecols: Optional[List[str]] = sorted(needed) if needed else None
        read = read_csv_auto(csv, usecols=usecols)
        frame = read.frame
        warnings: List[str] = list(read.warnings)

        # ---------- ② 划分（纯函数） ----------
        try:
            result = split_frame(frame, spec, target=target_column)
        except ConfigError as exc:
            if spec.mode == "field":
                return result_error("expression_invalid", exc.message, EXPRESSION_HINT)
            return result_error(
                "config_invalid",
                exc.message,
                "检查 mode / ratios / stratify_by / seed；见 docs/preprocess-split.md",
            )
        except DataError as exc:
            return result_error(
                "analysis_failed",
                exc.message,
                "数据为空或列不满足划分要求；先确认 dataset_load 读到了数据行",
            )
        warnings.extend(result.warnings)

        masks = result.masks()
        label_distribution = _label_distribution(frame, masks, target_column)

        # ---------- ③ 落盘 split/split.json ----------
        split_dir = layout.subdir("split")
        split_path = split_dir / "split.json"
        payload = {
            "schema": SPLIT_SCHEMA,
            "version": __version__,
            "generated_at": now_iso(),
            "mode": spec.mode,
            "target": target,
            "rows": int(frame.shape[0]),
            "spec": spec.to_dict(),
            "counts": dict(result.counts),
            "ratios": {k: round(float(v), 6) for k, v in result.ratios.items()},
            "seed": spec.seed,
            "fingerprint": result.fingerprint,
            "description": result.description,
            "label_distribution": label_distribution,
            "warnings": warnings,
        }
        atomic_write_json(split_path, to_jsonable(payload))

        summary = (
            f"mode={spec.mode}，train {result.counts['train']} / "
            f"val {result.counts['val']} / test {result.counts['test']} 行，"
            f"指纹 {result.fingerprint}"
        )

        def _mutate(manifest_doc: Dict[str, Any]) -> None:
            stage_done(
                manifest_doc,
                "split",
                done=True,
                path="split/",
                summary=summary,
            )

        update_manifest(layout.project_dir, _mutate)

        data = {
            "project_dir": layout.project_dir.as_posix(),
            "mode": spec.mode,
            "rows": int(frame.shape[0]),
            "counts": dict(result.counts),
            "ratios": {k: round(float(v), 6) for k, v in result.ratios.items()},
            "seed": spec.seed,
            "fingerprint": result.fingerprint,
            "description": result.description,
            "label_distribution": label_distribution,
            "spec": spec.to_dict(),
        }
        if spec.mode == "none":
            warnings.append(
                "mode=none：全部样本用于训练与评估，指标是「全量口径」，"
                "不代表样本外表现（已在 split.json.description 标注）"
            )
        return result_ok(
            data,
            artifacts=relative_paths(layout, [split_path]),
            warnings=warnings + [f"已写入 {layout.rel(split_path)}"],
        )
    except Exception as exc:  # noqa: BLE001 - 统一三态错误载荷
        return result_from_exception(exc)
