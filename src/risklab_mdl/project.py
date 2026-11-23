"""项目状态读取：manifest 里的角色、CSV 定位。

工具层与预处理层都需要「从 manifest 恢复项目状态」，但各自不应再实现一遍
（否则同一个 csv_path 回退顺序或角色优先级会在两个地方分叉）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .artifacts.manifest import ManifestError
from .data.loader import infer_roles
from .platform.fs import PathNotReadable, normalize_path

__all__ = ["manifest_roles", "resolve_csv_path", "roles_for_analysis"]


def manifest_roles(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """从 manifest 取角色（只留 id/time/split，``features`` 在此无意义）。"""
    roles = (manifest.get("project") or {}).get("roles") or {}
    return {key: roles.get(key) for key in ("id", "time", "split")}


def roles_for_analysis(manifest: Dict[str, Any], frame: pd.DataFrame) -> Dict[str, Any]:
    """优先用 manifest 里已确认/已记录的角色，缺失则现场推断一次。"""
    roles = manifest_roles(manifest)
    if not any(roles.get(key) for key in ("id", "time", "split")):
        inferred, _ = infer_roles(frame)
        roles = {key: inferred.get(key) for key in ("id", "time", "split")}
    return roles


def resolve_csv_path(layout: Any, manifest: Dict[str, Any]) -> Path:
    """定位 CSV：优先布局里解析到的路径，其次 manifest 记录的路径。"""
    if layout.csv_path is not None:
        return Path(layout.csv_path)
    raw = (manifest.get("project") or {}).get("csv_path")
    if not raw:
        raise ManifestError(
            f"manifest 里没有 csv_path，无法定位数据文件：{layout.project_dir}",
            code="manifest_schema_invalid",
            hint="重新执行 dataset_load 以写入 csv_path",
        )
    path = normalize_path(raw)
    if not path.is_file():
        raise PathNotReadable(
            f"manifest 记录的 CSV 不存在：{path}",
            hint="文件被移动/删除时，把 CSV 放回原位置或重新执行 project_init / dataset_load",
        )
    return path
