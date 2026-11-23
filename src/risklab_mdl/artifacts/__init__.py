"""``risklab_mdl.artifacts`` —— 产物目录契约（``mdl-<csv stem>/``）。

冻结契约（后续模块共用）：

- 项目目录 = ``<CSV 所在目录>/mdl-<CSV 文件名去扩展名>/``，已存在则**复用**；
- CSV 所在目录不可写 ⇒ 报错 ``project_dir_not_writable``，不静默换位置；
- ``runs/<run_id>/``，``run_id`` 形如 ``20260101T120000``；
- ``manifest.json`` 是项目状态唯一事实来源，schema = ``risklab-mdl.manifest/1``。
"""

from .layout import (
    CHART_SUBDIRS,
    MODEL_DIR_PREFIX,
    PROJECT_SUBDIRS,
    LayoutError,
    ProjectLayout,
    open_project,
    project_dir_for,
    resolve_project,
)
from .manifest import (
    MANIFEST_FILENAME,
    MANIFEST_LOCK_NAME,
    MANIFEST_SCHEMA,
    RUN_STATUSES,
    STAGES,
    TASK_TYPES,
    ManifestError,
    ensure_manifest,
    manifest_path,
    new_manifest,
    now_iso,
    read_manifest,
    stage_done,
    update_manifest,
    validate_manifest,
    write_manifest,
)
from .store import (
    RUN_ID_FORMAT,
    RUN_ID_PATTERN,
    StoreError,
    allocate_run_dir,
    allocate_run_id,
    latest_run_id,
    list_run_ids,
    resolve_run_dir,
    validate_run_id,
)

__all__ = [
    "CHART_SUBDIRS",
    "MANIFEST_FILENAME",
    "MANIFEST_LOCK_NAME",
    "MANIFEST_SCHEMA",
    "MODEL_DIR_PREFIX",
    "PROJECT_SUBDIRS",
    "RUN_ID_FORMAT",
    "RUN_ID_PATTERN",
    "RUN_STATUSES",
    "STAGES",
    "TASK_TYPES",
    "LayoutError",
    "ManifestError",
    "ProjectLayout",
    "StoreError",
    "allocate_run_dir",
    "allocate_run_id",
    "ensure_manifest",
    "latest_run_id",
    "list_run_ids",
    "manifest_path",
    "new_manifest",
    "now_iso",
    "open_project",
    "project_dir_for",
    "read_manifest",
    "resolve_project",
    "resolve_run_dir",
    "stage_done",
    "update_manifest",
    "validate_manifest",
    "validate_run_id",
    "write_manifest",
]
