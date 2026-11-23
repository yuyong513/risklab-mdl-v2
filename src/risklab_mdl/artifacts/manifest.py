"""``manifest.json`` 读写与 schema 校验（schema = ``risklab-mdl.manifest/1``）。

★ 为什么 manifest 要严格校验
---------------------------
manifest 是项目状态的**唯一事实来源**：哪个阶段做完了、有几个 run、最终确认了哪个。
它一旦被写坏（手工编辑、并发覆盖、中断写），后续所有工具都会读到错的状态。
因此三条规矩：

1. 写之前必须过 :func:`validate_manifest`（非法结构写不进去）；
2. 写必须是原子替换（临时文件 + ``os.replace``）且持项目级锁；
3. 读之前也校验（被外部改坏时**当场**报 ``manifest_schema_invalid``，不静默当空文档）。

★ 并发写不静默覆盖（lost update）
--------------------------------
「加锁 + 原子替换」只保证**单次**写不会写坏文件，不能阻止两个调用方各自
「先读 → 改 → 写」时后写的一方覆盖前一方（经典 lost update）。因此：

- 所有增量变更走 :func:`update_manifest(mutator)`：在锁内读 → 改 → 写，
  调用方不可能基于过期快照；
- :func:`write_manifest` 只用于初始化 / 全量重写；若调用方是「先读过、再写回」，
  必须传 ``expected_updated_at=``（读到的 ``project.updated_at``）。函数在锁内
  重读并比对，不一致 ⇒ ``manifest_conflict``（``hint`` 指向 update_manifest），
  过期快照绝不会落盘；
- ``updated_at`` 是严格递增的逻辑时钟（同一秒内连续写按 +1s 递推），保证秒级
  精度下仍能区分两次写。

schema v1 结构见 ``docs/tool-contract.md`` 与 PM 计划 §2.3。
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ..platform.fs import (
    PlatformError,
    atomic_write_json,
    check_writable,
    file_lock,
    normalize_path,
    read_json,
)
from .store import RUN_ID_PATTERN

PathLike = Union[str, os.PathLike]

#: manifest 文件名（项目根下）。
MANIFEST_FILENAME = "manifest.json"

#: manifest 锁文件名（与 manifest 分离，替换 manifest 不影响锁）。
MANIFEST_LOCK_NAME = ".manifest.lock"

#: 冻结的 schema 标识。
MANIFEST_SCHEMA = "risklab-mdl.manifest/1"

#: 固定阶段键（顺序即建模流程顺序）。
STAGES: tuple[str, ...] = (
    "data_analysis",
    "feature_analysis",
    "feature_selection",
    "preprocess",
    "split",
    "train",
)

#: run 状态机取值。
RUN_STATUSES: tuple[str, ...] = ("trained", "tuned", "evaluated", "confirmed", "failed")

#: 任务类型取值。
TASK_TYPES: tuple[str, ...] = ("binary", "multiclass")


class ManifestError(PlatformError):
    """manifest 缺失、损坏或结构非法。"""

    code = "manifest_invalid"


def now_iso() -> str:
    """当前本地时间（ISO 8601，含时区偏移，精确到秒）。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def manifest_path(project_dir: PathLike) -> Path:
    return normalize_path(project_dir) / MANIFEST_FILENAME


def new_manifest(csv_path: PathLike, *, stem: Optional[str] = None) -> Dict[str, Any]:
    """构造一个空 manifest（所有阶段未完成、无 run、未确认）。"""
    csv = normalize_path(csv_path)
    ts = now_iso()
    return {
        "schema": MANIFEST_SCHEMA,
        "project": {
            "csv_path": csv.as_posix(),
            "stem": stem or csv.stem,
            "created_at": ts,
            "updated_at": ts,
            "target": None,
            "roles": {"id": None, "time": None, "split": None, "features": []},
        },
        "stages": {name: {"done": False} for name in STAGES},
        "runs": [],
        "confirmed_run": None,
    }


def _problems(manifest: Any) -> List[str]:
    """收集结构问题（一次列全，不第一处就抛，便于用户一次改完）。"""
    out: List[str] = []
    if not isinstance(manifest, dict):
        return [f"顶层必须是 JSON 对象，实际是 {type(manifest).__name__}"]

    if manifest.get("schema") != MANIFEST_SCHEMA:
        out.append(f"schema 必须是 {MANIFEST_SCHEMA!r}，实际是 {manifest.get('schema')!r}")

    project = manifest.get("project")
    if not isinstance(project, dict):
        out.append("缺少 project 对象")
        project = {}
    else:
        for key in ("csv_path", "stem", "created_at", "updated_at"):
            value = project.get(key)
            if not isinstance(value, str) or not value.strip():
                out.append(f"project.{key} 必须是非空字符串")
        target = project.get("target")
        if target is not None:
            if not isinstance(target, dict):
                out.append("project.target 必须是对象或 null")
            else:
                if not isinstance(target.get("column"), str) or not target["column"].strip():
                    out.append("project.target.column 必须是非空字符串")
                if target.get("task") not in TASK_TYPES:
                    out.append(f"project.target.task 必须是 {TASK_TYPES} 之一")
                labels = target.get("labels")
                if labels is not None and not isinstance(labels, dict):
                    out.append("project.target.labels 必须是对象或 null")
        roles = project.get("roles")
        if roles is not None and not isinstance(roles, dict):
            out.append("project.roles 必须是对象或 null")

    stages = manifest.get("stages")
    if not isinstance(stages, dict):
        out.append("缺少 stages 对象")
        stages = {}
    else:
        unknown = sorted(set(stages) - set(STAGES))
        if unknown:
            out.append(f"stages 含未知阶段：{unknown}（合法：{list(STAGES)}）")
        for name in STAGES:
            entry = stages.get(name)
            if not isinstance(entry, dict):
                out.append(f"stages.{name} 必须是对象")
                continue
            if not isinstance(entry.get("done"), bool):
                out.append(f"stages.{name}.done 必须是布尔值")
            # 阶段里的附加字段也做类型校验：写坏了同样会让下游读错状态
            for key in ("path", "summary"):
                if key in entry and not isinstance(entry[key], str):
                    out.append(f"stages.{name}.{key} 必须是字符串")

    runs = manifest.get("runs")
    run_ids: List[str] = []
    seen_run_ids: Dict[str, int] = {}
    if not isinstance(runs, list):
        out.append("runs 必须是数组")
        runs = []
    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            out.append(f"runs[{idx}] 必须是对象")
            continue
        run_id = run.get("run_id")
        if not isinstance(run_id, str) or not RUN_ID_PATTERN.match(run_id):
            out.append(
                f"runs[{idx}].run_id 必须形如 YYYYmmddTHHMMSS（正则 {RUN_ID_PATTERN.pattern}）"
            )
        elif run_id in seen_run_ids:
            out.append(
                f"runs[{idx}].run_id={run_id!r} 与 runs[{seen_run_ids[run_id]}] 重复"
                "（run_id 必须全局唯一）"
            )
        else:
            seen_run_ids[run_id] = idx
            run_ids.append(run_id)
        if run.get("status") not in RUN_STATUSES:
            out.append(f"runs[{idx}].status 必须是 {RUN_STATUSES} 之一")
        for key in ("algorithm", "created_at"):
            if key in run and not isinstance(run[key], str):
                out.append(f"runs[{idx}].{key} 必须是字符串")
        # task 可省略（兼容只记 run_id/status 的早期产物）；一旦给出就必须是合法任务类型
        if "task" in run and run["task"] not in TASK_TYPES:
            out.append(
                f"runs[{idx}].task 必须是 {TASK_TYPES} 之一（或省略该字段）"
            )
        for key in ("metrics", "paths"):
            if key in run and not isinstance(run[key], dict):
                out.append(f"runs[{idx}].{key} 必须是对象")

    confirmed = manifest.get("confirmed_run")
    if confirmed is not None:
        if not isinstance(confirmed, str) or not confirmed.strip():
            out.append("confirmed_run 必须是 run_id 字符串或 null")
        elif confirmed not in run_ids:
            out.append(
                f"confirmed_run={confirmed!r} 不在 runs 列表中"
                "（confirmed_run 必须指向已存在的 run）"
            )

    return out


def validate_manifest(manifest: Any) -> Dict[str, Any]:
    """校验结构；不合法抛 :class:`ManifestError`（``manifest_schema_invalid``）。"""
    problems = _problems(manifest)
    if problems:
        raise ManifestError(
            "manifest.json 结构不合法：" + "；".join(problems),
            code="manifest_schema_invalid",
            hint=(
                f"按 schema {MANIFEST_SCHEMA} 修正；若不想要旧状态，删除 manifest.json "
                "后由 project_init 重新生成"
            ),
        )
    return manifest  # type: ignore[return-value]


def read_manifest(project_dir: PathLike) -> Dict[str, Any]:
    """读并校验 manifest；缺失抛 ``manifest_not_found``。"""
    path = manifest_path(project_dir)
    if not path.is_file():
        raise ManifestError(
            f"项目缺少 manifest.json：{path}",
            code="manifest_not_found",
            hint="先调用 project_init / dataset_load 创建项目（会生成 manifest.json）",
        )
    data = read_json(path)
    return validate_manifest(data)


def _lock_file(project_dir: Path) -> Path:
    return project_dir / MANIFEST_LOCK_NAME


def _project_dir_for_write(project_dir: PathLike) -> Path:
    """写 manifest 前的目录前置条件：**必须已存在**且可写。

    ★ 为什么不顺手 mkdir
      项目目录的创建是 ``artifacts.layout.open_project`` 的职责（它负责建齐 8 个
      固定子目录）。这里若隐式建一个只有 manifest 的空目录，后续步骤会以为
      「项目已初始化」，实际缺子目录 —— 又是一种静默的坏状态。
    """
    target = normalize_path(project_dir)
    if not target.is_dir():
        raise ManifestError(
            f"项目目录不存在：{target}",
            code="project_dir_not_found",
            hint="先调用 project_init / dataset_load 由 CSV 路径创建项目目录（mdl-<CSV 文件名>）",
        )
    return check_writable(target, label="项目目录")


def _read_current_updated_at(path: Path) -> Optional[str]:
    """锁内重读磁盘上的 ``project.updated_at``（文件不存在/读不出时返回 ``None``）。"""
    if not path.is_file():
        return None
    data = read_json(path)
    if not isinstance(data, dict):
        return None
    project = data.get("project")
    if not isinstance(project, dict):
        return None
    stamp = project.get("updated_at")
    return stamp if isinstance(stamp, str) else None


def _next_updated_at(current: Optional[str]) -> str:
    """生成**严格递增**的 ``updated_at``（逻辑时钟）。

    ``now_iso()`` 只精确到秒：同一秒内连写两次会得到相同字符串，届时
    ``expected_updated_at`` 乐观校验就失去意义。这里若新时间戳不晚于磁盘值，
    就在磁盘值上 +1s 递推，保证「每次成功写入的 updated_at 都不同且递增」。
    """
    stamp = now_iso()
    if current and stamp <= current:
        try:
            stamp = (datetime.fromisoformat(current) + timedelta(seconds=1)).isoformat(
                timespec="seconds"
            )
        except ValueError:  # 磁盘上的时间戳被手工改成了非 ISO 文本
            pass
    return stamp


def write_manifest(
    project_dir: PathLike,
    manifest: Dict[str, Any],
    *,
    timeout: float = 30.0,
    expected_updated_at: Optional[str] = None,
) -> Path:
    """校验 + 加锁 + 原子写，``updated_at`` 由本函数统一刷新。

    **本函数只用于初始化 / 全量重写**。增量变更请用 :func:`update_manifest`
    （锁内读 → 改 → 写，天然免疫 lost update）。

    若调用方确实需要「先读过 → 再写回」，必须把读到的 ``project.updated_at``
    作为 ``expected_updated_at`` 传入：函数在锁内重读并比对，不一致时抛
    :class:`ManifestError`（``code=manifest_conflict``），过期快照绝不落盘。
    """
    validate_manifest(manifest)
    target_dir = _project_dir_for_write(project_dir)
    path = target_dir / MANIFEST_FILENAME
    with file_lock(_lock_file(target_dir), timeout=timeout):
        try:
            current = _read_current_updated_at(path)
        except PlatformError as exc:
            if expected_updated_at is not None:
                raise ManifestError(
                    f"无法确认 manifest 当前版本（{exc.message}），已中止写入以免覆盖别人的改动",
                    code="manifest_conflict",
                    hint="manifest.json 已损坏；修复或删除后由 project_init 重新生成，再重试",
                ) from exc
            current = None
        if expected_updated_at is not None and current != expected_updated_at:
            raise ManifestError(
                "manifest 已被其他写入更新（期望 updated_at="
                f"{expected_updated_at!r}，当前 {current!r}），"
                "本次写入基于过期快照，已中止（不静默覆盖）",
                code="manifest_conflict",
                hint=(
                    "重新读取 manifest 后再合并本次修改并重试；"
                    "批量/增量变更请统一走 update_manifest(mutator)，它在锁内读改写"
                ),
            )
        manifest["project"]["updated_at"] = _next_updated_at(current)
        return atomic_write_json(path, manifest)


def update_manifest(
    project_dir: PathLike,
    mutator: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
    *,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """在锁内「读 → 改 → 写」，返回落盘后的 manifest。

    ``mutator`` 可就地修改并返回 ``None``，或返回新的字典（返回即替换）。
    这样调用方不必自己管锁，也不会出现「读旧写旧」的 lost update。
    并发调用方各自在锁内读到最新状态，不存在静默覆盖。
    """
    target_dir = _project_dir_for_write(project_dir)
    path = target_dir / MANIFEST_FILENAME
    with file_lock(_lock_file(target_dir), timeout=timeout):
        if not path.is_file():
            raise ManifestError(
                f"项目缺少 manifest.json：{path}",
                code="manifest_not_found",
                hint="先调用 project_init / dataset_load 创建项目",
            )
        manifest = validate_manifest(read_json(path))
        replaced = mutator(manifest)
        final = replaced if replaced is not None else manifest
        validate_manifest(final)
        final["project"]["updated_at"] = _next_updated_at(manifest["project"].get("updated_at"))
        atomic_write_json(path, final)
    return final


def ensure_manifest(
    project_dir: PathLike,
    csv_path: PathLike,
    *,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """读已有 manifest，缺失则创建（在锁内判断，避免两个进程各建一份）。"""
    target_dir = _project_dir_for_write(project_dir)
    path = target_dir / MANIFEST_FILENAME
    with file_lock(_lock_file(target_dir), timeout=timeout):
        if path.is_file():
            return validate_manifest(read_json(path))
        manifest = new_manifest(csv_path)
        validate_manifest(manifest)
        atomic_write_json(path, manifest)
        return manifest


def stage_done(
    manifest: Dict[str, Any],
    stage: str,
    *,
    done: bool = True,
    path: Optional[str] = None,
    summary: Optional[str] = None,
) -> None:
    """就地标记某阶段完成度（调用方随后 ``write_manifest`` / ``update_manifest``）。"""
    if stage not in STAGES:
        raise ManifestError(
            f"未知阶段：{stage}",
            code="stage_unknown",
            hint=f"合法阶段：{', '.join(STAGES)}",
        )
    entry: Dict[str, Any] = {"done": done}
    if path is not None:
        entry["path"] = path
    if summary is not None:
        entry["summary"] = summary
    manifest.setdefault("stages", {})[stage] = entry


__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_LOCK_NAME",
    "MANIFEST_SCHEMA",
    "RUN_STATUSES",
    "STAGES",
    "TASK_TYPES",
    "ManifestError",
    "ensure_manifest",
    "manifest_path",
    "new_manifest",
    "now_iso",
    "read_manifest",
    "stage_done",
    "update_manifest",
    "validate_manifest",
    "write_manifest",
]
