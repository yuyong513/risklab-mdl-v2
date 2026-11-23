"""``mdl-<csv stem>/`` 项目目录布局：解析、创建/复用、可写性错误码。

契约（PM 计划 §1.2，冻结）::

    <CSV 同目录>/mdl-<CSV 文件名去扩展名>/
      manifest.json
      data_analysis/      charts/
      feature_analysis/   charts/
      feature_selection/
      preprocess/
      split/
      runs/<YYYYmmddTHHMMSS>/
      final/
      reports/

★ 三条不可协商的行为
--------------------
1. **位置唯一**：项目目录只由 CSV 路径决定，没有全局工作区、没有环境变量兜底。
   CSV 在 ``P/x.csv`` ⇒ 项目必在 ``P/mdl-x``。
2. **复用不覆盖**：目录已存在即复用；历史 run 靠时间戳目录名区分，绝不清理。
3. **不可写就报错**：``project_dir_not_writable``，并给可执行的处置建议；
   不静默改写到临时目录（否则用户找不到产物）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from ..platform.fs import (
    PathNotWritable,
    PlatformError,
    check_writable,
    ensure_dir,
    normalize_path,
    require_readable,
)

PathLike = Union[str, os.PathLike]

#: 项目目录前缀。``x.csv`` ⇒ ``mdl-x``。
MODEL_DIR_PREFIX = "mdl-"

#: 项目目录下的固定子目录（创建项目时一次性建好，避免后续步骤各建一份）。
PROJECT_SUBDIRS: tuple[str, ...] = (
    "data_analysis",
    "feature_analysis",
    "feature_selection",
    "preprocess",
    "split",
    "runs",
    "final",
    "reports",
)

#: 需要预建 ``charts/`` 的子目录。
CHART_SUBDIRS: tuple[str, ...] = ("data_analysis", "feature_analysis")


class LayoutError(PlatformError):
    """项目目录布局相关错误（含不可写、冲突）。"""

    code = "layout_error"


def _not_writable(target: Path, detail: str = "") -> PathNotWritable:
    """统一把「项目目录不可写」映射成契约错误码 ``project_dir_not_writable``。"""
    msg = f"CSV 所在目录不可写，无法创建建模项目目录：{target}"
    if detail:
        msg = f"{msg}（{detail}）"
    return PathNotWritable(
        msg,
        code="project_dir_not_writable",
        hint=(
            "把 CSV 复制到可写目录（如用户目录下的工作目录）后重试，"
            "或为该目录授予写权限；工具不会静默改写到其他位置。"
        ),
    )


def _csv_stem(csv_path: Path) -> str:
    stem = csv_path.stem
    if not stem or stem in (".", ".."):
        raise LayoutError(
            f"无法从 CSV 文件名推导项目目录名：{csv_path}",
            code="csv_name_invalid",
            hint="CSV 文件名不能为空或只有扩展名（例如 .csv），请重命名后再试",
        )
    return stem


def project_dir_for(csv_path: PathLike) -> Path:
    """解析项目目录路径（不创建、不校验可写）。"""
    p = normalize_path(csv_path)
    return p.parent / f"{MODEL_DIR_PREFIX}{_csv_stem(p)}"


@dataclass(frozen=True)
class ProjectLayout:
    """一个建模项目的目录视图。``csv_path`` 可为 ``None``（只给 ``project_dir`` 的场景）。"""

    project_dir: Path
    csv_path: Optional[Path] = None

    # ---------- 属性 ----------

    @property
    def stem(self) -> str:
        name = self.project_dir.name
        return name[len(MODEL_DIR_PREFIX):] if name.startswith(MODEL_DIR_PREFIX) else name

    @property
    def manifest_path(self) -> Path:
        return self.project_dir / "manifest.json"

    @property
    def runs_dir(self) -> Path:
        return self.project_dir / "runs"

    @property
    def final_dir(self) -> Path:
        return self.project_dir / "final"

    @property
    def reports_dir(self) -> Path:
        return self.project_dir / "reports"

    # ---------- 子路径 ----------

    def subdir(self, name: str) -> Path:
        """取固定子目录的绝对路径；``name`` 不在契约内时直接报错。"""
        if name not in PROJECT_SUBDIRS:
            raise LayoutError(
                f"未知的项目子目录：{name}",
                code="subdir_unknown",
                hint=f"可用子目录：{', '.join(PROJECT_SUBDIRS)}",
            )
        return self.project_dir / name

    def charts_dir(self, parent: str) -> Path:
        """``data_analysis/charts`` / ``feature_analysis/charts``。"""
        if parent not in CHART_SUBDIRS:
            raise LayoutError(
                f"该子目录不含 charts：{parent}",
                code="subdir_unknown",
                hint=f"可含 charts 的子目录：{', '.join(CHART_SUBDIRS)}",
            )
        return self.subdir(parent) / "charts"

    def run_dir(self, run_id: str) -> Path:
        """某个 run 的目录（不创建、不校验存在）。"""
        from .store import validate_run_id  # 延迟导入避免循环

        return self.runs_dir / validate_run_id(run_id)

    def rel(self, path: PathLike) -> str:
        """把项目内绝对路径转成 manifest 里记录的 POSIX 相对路径。"""
        p = normalize_path(path)
        try:
            return p.relative_to(self.project_dir).as_posix()
        except ValueError:
            # 项目外路径（如 CSV 本身）原样返回绝对路径字符串，不做静默截断
            return p.as_posix()

    def exists(self) -> bool:
        return self.project_dir.is_dir()


def _make_project_dirs(layout: ProjectLayout) -> ProjectLayout:
    """建齐固定子目录。任何一步失败都映射成 ``project_dir_not_writable``。"""
    try:
        ensure_dir(layout.project_dir)
        for name in PROJECT_SUBDIRS:
            ensure_dir(layout.project_dir / name)
        for name in CHART_SUBDIRS:
            ensure_dir(layout.project_dir / name / "charts")
    except PathNotWritable as exc:
        raise _not_writable(layout.project_dir, exc.message) from exc
    except OSError as exc:
        raise _not_writable(layout.project_dir, f"{type(exc).__name__}: {exc}") from exc
    return layout


def open_project(csv_path: PathLike, *, create: bool = True) -> ProjectLayout:
    """由 CSV 路径解析/创建项目目录（已存在则复用）。

    - CSV 不存在或不可读 ⇒ ``path_not_readable``；
    - 同名路径存在但不是目录 ⇒ ``project_dir_conflict``；
    - CSV 所在目录不可写 ⇒ ``project_dir_not_writable``。
    """
    csv = require_readable(csv_path, label="CSV 文件")
    target = project_dir_for(csv)
    if target.exists() and not target.is_dir():
        raise LayoutError(
            f"项目目录位置被同名文件占用：{target}",
            code="project_dir_conflict",
            hint="删除或重命名该文件；项目目录必须是文件夹（mdl-<CSV 文件名>）",
        )
    layout = ProjectLayout(project_dir=target, csv_path=csv)
    if create:
        try:
            check_writable(target.parent, label=f"CSV 所在目录 {target.parent}")
        except PathNotWritable as exc:
            # 契约要求统一错误码：不可写一律 project_dir_not_writable
            raise _not_writable(target, exc.message) from exc
        _make_project_dirs(layout)
    return layout


def resolve_project(
    project_dir: PathLike,
    *,
    csv_path: Optional[PathLike] = None,
    create: bool = False,
) -> ProjectLayout:
    """由**项目目录**解析布局（工具以 ``project_dir`` 为入参）。

    ``csv_path`` 未显式给出时，从 ``manifest.json`` 的 ``project.csv_path`` 读取；
    读不到也不报错（部分步骤只需要目录本身），``csv_path`` 保持 ``None``。
    """
    target = normalize_path(project_dir)
    if target.exists() and not target.is_dir():
        raise LayoutError(
            f"project_dir 指向的是文件而不是目录：{target}",
            code="project_dir_conflict",
            hint="传入 mdl-<CSV 文件名> 目录本身",
        )
    csv: Optional[Path]
    if csv_path is not None:
        csv = require_readable(csv_path, label="CSV 文件")
    else:
        csv = None
        manifest_file = target / "manifest.json"
        if manifest_file.is_file():
            from ..platform.fs import read_json

            try:
                data = read_json(manifest_file, default={}, has_default=True)
            except PlatformError:
                data = {}
            raw = ((data or {}).get("project") or {}).get("csv_path")
            if raw:
                csv = normalize_path(raw)
    if not target.exists():
        if not create:
            raise LayoutError(
                f"项目目录不存在：{target}",
                code="project_dir_not_found",
                hint="先调用 project_init / dataset_load 创建项目目录（mdl-<CSV 文件名>）",
            )
        return _make_project_dirs(ProjectLayout(project_dir=target, csv_path=csv))
    layout = ProjectLayout(project_dir=target, csv_path=csv)
    if create:
        check_writable(target, label=f"项目目录 {target}")
        _make_project_dirs(layout)
    return layout
