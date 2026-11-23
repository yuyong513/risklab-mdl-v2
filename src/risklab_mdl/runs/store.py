"""run 目录与产物写入：``runs/<run_id>/`` 的唯一写入入口。
★ 产物布局（工具契约 §4 冻结）
------------------------------
    runs/<run_id>/
      config.json     本次训练的全部入参（算法/任务/特征/划分/seed/数据来源）
      params.json     最终超参（默认值 + 覆盖 + 调参结果）
      metrics.json    train/val/test 指标（+ 多分类 per_class/macro + PSI）
      model.pkl       模型包（schema 2.0，可 load_package 并打分）
      importance.csv  特征重要性（已按重要性降序）
      evaluation/     评价表与图（model_evaluate 写）
      tuning/         调参轨迹与空间（model_tune 写）

★ 为什么 manifest 只走 ``update_manifest``
-----------------------------------------
``manifest.runs`` 是**增量**事实（每跑一次训练追加一条）。若用
「read → append → write_manifest」，两次并发训练会互相覆盖（lost update）。
``update_manifest`` 在锁内读改写，并且 run_id 由 ``allocate_run_dir`` 原子占位
（mkdir exist_ok=False），因此既不会撞号也不会丢记录。

★ run 状态机
------------
``trained`` →（``model_evaluate``）``evaluated`` →（``model_confirm``）``confirmed``；
``model_tune`` 产出的 run 记为 ``tuned``。任何一步失败都不会把状态改回 ``trained``
以外的值（状态只能前进，除了显式失败标记）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from ..artifacts.manifest import now_iso, stage_done, update_manifest
from ..artifacts.store import allocate_run_dir, resolve_run_dir, validate_run_id
from ..core.errors import ModelError
from ..core.package import ModelPackage, load_package, save_package
from ..data.loader import to_jsonable
from ..platform.fs import atomic_write_json, atomic_write_text, read_json

__all__ = [
    "RUN_SCHEMA",
    "RunFiles",
    "append_run",
    "read_run",
    "rel_paths",
    "run_metrics_summary",
    "run_file",
    "update_run",
    "write_evaluation",
    "write_run",
]

RUN_SCHEMA = "risklab-mdl.run/1"

#: run 状态的推进次序（``update_run`` 用它拦「状态回退」）。
#: ``trained`` 与 ``tuned`` 同为第 0 档：调参产物也是「已训练但未评价」。
_STATUS_ORDER: Dict[str, int] = {
    "trained": 0,
    "tuned": 0,
    "evaluated": 1,
    "confirmed": 2,
}

#: run 目录内的产物文件名（相对 run_dir）。
CONFIG_FILE = "config.json"
PARAMS_FILE = "params.json"
METRICS_FILE = "metrics.json"
MODEL_FILE = "model.pkl"
IMPORTANCE_FILE = "importance.csv"


@dataclass
class RunFiles:
    """一个 run 的目录与文件视图。"""

    run_id: str
    run_dir: Path

    @property
    def config_path(self) -> Path:
        return self.run_dir / CONFIG_FILE

    @property
    def params_path(self) -> Path:
        return self.run_dir / PARAMS_FILE

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / METRICS_FILE

    @property
    def model_path(self) -> Path:
        return self.run_dir / MODEL_FILE

    @property
    def importance_path(self) -> Path:
        return self.run_dir / IMPORTANCE_FILE

    @property
    def evaluation_dir(self) -> Path:
        return self.run_dir / "evaluation"

    @property
    def tuning_dir(self) -> Path:
        return self.run_dir / "tuning"

    def ensure_evaluation_dir(self) -> Path:
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        return self.evaluation_dir

    def ensure_tuning_dir(self) -> Path:
        self.tuning_dir.mkdir(parents=True, exist_ok=True)
        return self.tuning_dir

    def files(self) -> Dict[str, Optional[str]]:
        """已存在的产物（相对 ``runs/<run_id>/`` 的 POSIX 路径）。"""
        out: Dict[str, Optional[str]] = {}
        for name, path in (
            ("config", self.config_path),
            ("params", self.params_path),
            ("metrics", self.metrics_path),
            ("model", self.model_path),
            ("importance", self.importance_path),
        ):
            out[name] = path.name if path.is_file() else None
        out["evaluation"] = "evaluation" if self.evaluation_dir.is_dir() else None
        out["tuning"] = "tuning" if self.tuning_dir.is_dir() else None
        return out


def run_file(layout, run_id: str) -> RunFiles:
    """定位已有 run（不存在 ⇒ ``run_not_found``）。"""
    run_id = validate_run_id(run_id)
    return RunFiles(run_id=run_id, run_dir=resolve_run_dir(layout.runs_dir, run_id))


def rel_paths(layout, files: RunFiles) -> List[str]:
    """产物相对项目目录的 POSIX 路径（用于载荷的 ``artifacts``）。"""
    paths: List[str] = []
    for name in (CONFIG_FILE, PARAMS_FILE, METRICS_FILE, MODEL_FILE, IMPORTANCE_FILE):
        path = files.run_dir / name
        if path.is_file():
            paths.append(layout.rel(path))
    for sub in ("evaluation", "tuning"):
        directory = files.run_dir / sub
        if directory.is_dir():
            for path in sorted(directory.iterdir()):
                if path.is_file():
                    paths.append(layout.rel(path))
    return paths


def _write_importance(frame: Optional[pd.DataFrame], path: Path) -> None:
    if frame is None or len(frame) == 0:
        return
    buffer = frame.to_csv(index=False, lineterminator="\n")
    atomic_write_text(path, buffer)


def write_run(
    layout,
    *,
    algorithm: str,
    task: str,
    config: Mapping[str, Any],
    params: Mapping[str, Any],
    metrics: Mapping[str, Any],
    package: ModelPackage,
    importance: Optional[pd.DataFrame] = None,
    status: str = "trained",
    summary_text: str = "",
    extra_files: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """新建一个 run 目录并写入全部产物，同时把记录追加进 ``manifest.runs``。

    Returns
    -------
    dict
        ``{run_id, run_dir, files(rel paths), entry(manifest 里那条记录)}``
    """
    run_id, run_dir = allocate_run_dir(layout.runs_dir)
    files = RunFiles(run_id=run_id, run_dir=run_dir)

    created_at = now_iso()
    config_payload = {
        "schema": RUN_SCHEMA,
        "run_id": run_id,
        "created_at": created_at,
        "algorithm": algorithm,
        "task": task,
        **to_jsonable(dict(config)),
    }
    atomic_write_json(files.config_path, config_payload)
    atomic_write_json(
        files.params_path,
        to_jsonable({"run_id": run_id, "algorithm": algorithm, "params": dict(params)}),
    )
    atomic_write_json(files.metrics_path, to_jsonable(metrics))
    save_package(package, files.model_path)
    _write_importance(importance, files.importance_path)

    for name, text in (extra_files or {}).items():
        atomic_write_text(run_dir / name, text)

    summary = _metrics_summary(metrics)
    entry = {
        "run_id": run_id,
        "algorithm": algorithm,
        "task": task,
        "status": status,
        "metrics": summary,
        "paths": {
            "config": layout.rel(files.config_path),
            "params": layout.rel(files.params_path),
            "metrics": layout.rel(files.metrics_path),
            "model": layout.rel(files.model_path),
            "importance": layout.rel(files.importance_path),
        },
        "created_at": created_at,
    }
    append_run(layout, entry, summary_text=summary_text or _summary_text(algorithm, task, run_id))

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "files": rel_paths(layout, files),
        "entry": entry,
    }


def append_run(layout, entry: Mapping[str, Any], *, summary_text: str = "") -> Dict[str, Any]:
    """把一条 run 记录追加进 manifest（锁内读改写，不改历史记录）。"""

    def _mutate(manifest: Dict[str, Any]) -> None:
        runs = manifest.setdefault("runs", [])
        runs[:] = [r for r in runs if r.get("run_id") != entry.get("run_id")]
        runs.append(dict(entry))
        stage_done(
            manifest,
            "train",
            done=True,
            path="runs/",
            summary=summary_text or f"共 {len(runs)} 个 run",
        )

    return update_manifest(layout.project_dir, _mutate)


def update_run(
    layout,
    run_id: str,
    *,
    status: Optional[str] = None,
    metrics_patch: Optional[Mapping[str, Any]] = None,
    paths_patch: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """更新已有 run 的状态/指标摘要（只改传入的字段，不整条重写）。

    ★ 状态只前进（不因重跑评价而回退）
    ---------------------------------
    ``confirmed`` 之后重跑 ``model_evaluate`` 只更新指标摘要，
    **不会**把状态打回 ``evaluated`` —— 否则 manifest 里「已确认的 run」
    会看起来像没确认过，下游报告/闸门都会读错。``evaluated`` 同样不会被
    ``trained``/``tuned`` 覆盖。未知状态值不受此约束（原样写入）。
    """
    run_id = validate_run_id(run_id)

    def _mutate(manifest: Dict[str, Any]) -> None:
        runs = manifest.setdefault("runs", [])
        for entry in runs:
            if entry.get("run_id") != run_id:
                continue
            if status is not None:
                current = str(entry.get("status") or "")
                new_rank = _STATUS_ORDER.get(str(status))
                current_rank = _STATUS_ORDER.get(current)
                if new_rank is None or current_rank is None or new_rank >= current_rank:
                    entry["status"] = status
                # 否则静默保留更靠后的状态：这正是「状态只前进」的落点
            if metrics_patch:
                merged = dict(entry.get("metrics") or {})
                merged.update(dict(metrics_patch))
                entry["metrics"] = merged
            if paths_patch:
                merged_paths = dict(entry.get("paths") or {})
                merged_paths.update(dict(paths_patch))
                entry["paths"] = merged_paths
            return
        raise ModelError(
            f"manifest 里没有 run_id={run_id!r} 的记录（runs 目录与 manifest 不一致）。"
            f"请用 run_list 查看可用 run"
        )

    return update_manifest(layout.project_dir, _mutate)


def read_run(layout, run_id: str) -> Dict[str, Any]:
    """读一个 run 的 config / params / metrics（缺失的产物返回 ``None``）。"""
    files = run_file(layout, run_id)

    def _maybe(path: Path) -> Any:
        if not path.is_file():
            return None
        try:
            return read_json(path)
        except Exception as exc:  # noqa: BLE001 - 损坏的 JSON 必须可见
            raise ModelError(
                f"{layout.rel(path)} 无法解析（{type(exc).__name__}: {exc}）—— "
                f"重跑产生该产物的步骤"
            ) from exc

    return {
        "run_id": files.run_id,
        "run_dir": files.run_dir,
        "files": files,
        "config": _maybe(files.config_path),
        "params": _maybe(files.params_path),
        "metrics": _maybe(files.metrics_path),
        "model_path": files.model_path if files.model_path.is_file() else None,
        "importance_path": files.importance_path
        if files.importance_path.is_file()
        else None,
    }


def load_run_package(layout, run_id: str) -> ModelPackage:
    """加载某 run 的模型包（缺失 ⇒ 可定位的 ModelError）。"""
    files = run_file(layout, run_id)
    if not files.model_path.is_file():
        raise ModelError(
            f"run {run_id} 缺少 model.pkl：{layout.rel(files.model_path)}。"
            f"该 run 可能只记录了配置（训练未完成）"
        )
    return load_package(files.model_path)


def write_evaluation(run_id: str, files: RunFiles, tables: Mapping[str, pd.DataFrame]) -> List[Path]:
    """写 ``evaluation/*.csv``（表名 → DataFrame）。"""
    directory = files.ensure_evaluation_dir()
    written: List[Path] = []
    for name, frame in tables.items():
        if frame is None:
            continue
        path = directory / f"{name}.csv"
        buffer = frame.to_csv(index=False, lineterminator="\n")
        atomic_write_text(path, buffer)
        written.append(path)
    return written


def run_metrics_summary(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """把完整 metrics 压成 manifest 里的一行摘要（val 优先，其次 train）。"""
    return _metrics_summary(metrics)


def _pick(metrics: Mapping[str, Any], key: str, field: str = "ks") -> Any:
    for subset in (key, "train"):
        block = metrics.get(subset)
        if isinstance(block, Mapping):
            value = block.get(field)
            if value is not None:
                return value
    return None


def _metrics_summary(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for field in ("ks", "auc", "gini", "lift"):
        value = _pick(metrics, "val", field)
        if value is not None:
            summary[field] = value
    psi = metrics.get("psi")
    if isinstance(psi, Mapping) and psi.get("enabled") and psi.get("max_psi") is not None:
        summary["psi"] = psi.get("max_psi")
    if not summary:
        # 连 train 都没有 ⇒ 如实记录空摘要（不编 0）
        return {}
    summary["subset"] = "val" if _pick(metrics, "val") is not None else "train"
    return summary


def _summary_text(algorithm: str, task: str, run_id: str) -> str:
    return f"{algorithm}/{task} run {run_id}"
