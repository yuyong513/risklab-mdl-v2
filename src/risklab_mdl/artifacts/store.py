"""run 目录分配与命名校验：``runs/<YYYYmmddTHHMMSS>/``。

★ 同一秒内的冲突怎么处理
----------------------
两次训练可能落在同一秒。这里不用随机后缀（会破坏契约正则），而是**按秒递推**：
``20260101T120000`` 被占就用 ``20260101T120001``、``120002`` ……
占用判定用 ``mkdir(exist_ok=False)`` —— 它是原子的，两个进程同时抢同一秒时
只有一个能建成，另一个自然落到下一秒，不存在「先查后建」的竞态。

★ 为什么不用随机 4 位后缀
--------------------------
PM 计划 §1.2 已冻结 ``run_id`` 形如 ``YYYYmmddTHHMMSS``（正则可校验、可按时间排序、
人眼可读）。随机后缀会破坏这三点。
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

from ..platform.fs import (
    PathNotWritable,
    PlatformError,
    ensure_dir,
    normalize_path,
)

PathLike = Union[str, os.PathLike]

#: run_id 格式：``YYYYmmddTHHMMSS``（8 位日期 + ``T`` + 6 位时间）。
RUN_ID_FORMAT = "%Y%m%dT%H%M%S"

#: run_id 校验正则（冻结契约，测试按此断言）。
RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}$")

#: 同秒冲突最多往后递推多少秒（1 小时），超过则判定为异常情况。
MAX_RUN_ID_OFFSET = 3600


class StoreError(PlatformError):
    """run 存储层错误。"""

    code = "run_store_error"


def validate_run_id(run_id: str) -> str:
    """校验 run_id 形式（同时防路径穿越：只允许数字与 ``T``）。"""
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.match(run_id):
        raise StoreError(
            f"run_id 不合法：{run_id!r}",
            code="run_id_invalid",
            hint=(
                f"run_id 必须是 {RUN_ID_FORMAT} 格式的时间戳（正则 {RUN_ID_PATTERN.pattern}），"
                "例如 20260101T120000；从 run_list 返回的列表里取，不要手写"
            ),
        )
    return run_id


def allocate_run_dir(
    runs_dir: PathLike,
    *,
    now: Optional[datetime] = None,
    max_attempts: int = MAX_RUN_ID_OFFSET,
) -> Tuple[str, Path]:
    """在 ``runs/`` 下原子分配一个新 run 目录，返回 ``(run_id, run_dir)``。

    ``now`` 可注入（测试同秒冲突用）。分配即建目录 —— 目录存在就代表这个 run
    已被本次调用占用，后续步骤往里写 config/metrics。
    """
    target = ensure_dir(runs_dir)
    if not target.is_dir():
        raise StoreError(
            f"runs 路径不是目录：{target}",
            code="runs_dir_invalid",
            hint="项目目录结构被破坏；删除该同名文件或重建项目目录",
        )
    base = now or datetime.now()
    if base.tzinfo is not None:
        base = base.astimezone().replace(tzinfo=None)
    last_ts = ""
    for offset in range(max_attempts):
        run_id = (base + timedelta(seconds=offset)).strftime(RUN_ID_FORMAT)
        last_ts = run_id
        path = target / run_id
        try:
            path.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        except OSError as exc:
            raise PathNotWritable(
                f"无法创建 run 目录：{path}（{type(exc).__name__}: {exc}）",
                code="runs_dir_not_writable",
                hint="检查项目目录写权限与磁盘空间",
            ) from exc
        return run_id, path
    raise StoreError(
        f"连续 {max_attempts} 个时间戳（{last_ts} 及之后）的 run 目录都已存在",
        code="run_id_exhausted",
        hint=(
            "系统时钟可能被回拨或错误设置；确认时钟正确，"
            "或清理 runs/ 下多余的空目录后重试"
        ),
    )


def allocate_run_id(
    runs_dir: PathLike,
    *,
    now: Optional[datetime] = None,
    max_attempts: int = MAX_RUN_ID_OFFSET,
) -> str:
    """:func:`allocate_run_dir` 的薄包装：只要 run_id（目录已被占位）。"""
    run_id, _ = allocate_run_dir(runs_dir, now=now, max_attempts=max_attempts)
    return run_id


def list_run_ids(runs_dir: PathLike) -> List[str]:
    """按时间升序列出已有 run_id（忽略非时间戳目录，如 ``.lock``）。"""
    target = normalize_path(runs_dir)
    if not target.is_dir():
        return []
    return sorted(p.name for p in target.iterdir() if p.is_dir() and RUN_ID_PATTERN.match(p.name))


def latest_run_id(runs_dir: PathLike) -> Optional[str]:
    """最新的 run_id（无 run 时返回 ``None``）。"""
    ids = list_run_ids(runs_dir)
    return ids[-1] if ids else None


def resolve_run_dir(runs_dir: PathLike, run_id: str) -> Path:
    """按 run_id 定位已有 run 目录；不存在则报 ``run_not_found``。"""
    validate_run_id(run_id)
    path = normalize_path(runs_dir) / run_id
    if not path.is_dir():
        known = list_run_ids(runs_dir)
        raise StoreError(
            f"run 目录不存在：{path}",
            code="run_not_found",
            hint=(
                f"已有 run：{known if known else '（无）'}；"
                "用 run_list 查看可用 run_id 后再调用"
            ),
        )
    return path


__all__ = [
    "MAX_RUN_ID_OFFSET",
    "RUN_ID_FORMAT",
    "RUN_ID_PATTERN",
    "StoreError",
    "allocate_run_dir",
    "allocate_run_id",
    "latest_run_id",
    "list_run_ids",
    "resolve_run_dir",
    "validate_run_id",
]
