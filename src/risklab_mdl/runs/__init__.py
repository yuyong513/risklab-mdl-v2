"""run 目录与产物写入（实现在 :mod:`risklab_mdl.runs.store`）。

★ 为什么分成 package + store
---------------------------
``runs/`` 是产物目录概念（``runs/<run_id>/``），实现集中在 ``store.py`` 一处；
本 ``__init__`` 只做导出，方便工具层 ``from ..runs import write_run`` 一眼看到
「run 相关的全部操作」。**写入入口只有 store 里的函数** —— 工具层不允许自己
拼 ``runs/<run_id>/xxx.json`` 路径再写，否则产物布局会随工具漂移。
"""

from __future__ import annotations

from .store import (
    CONFIG_FILE,
    IMPORTANCE_FILE,
    METRICS_FILE,
    MODEL_FILE,
    PARAMS_FILE,
    RUN_SCHEMA,
    RunFiles,
    append_run,
    load_run_package,
    read_run,
    rel_paths,
    run_file,
    run_metrics_summary,
    update_run,
    write_evaluation,
    write_run,
)

__all__ = [
    "CONFIG_FILE",
    "IMPORTANCE_FILE",
    "METRICS_FILE",
    "MODEL_FILE",
    "PARAMS_FILE",
    "RUN_SCHEMA",
    "RunFiles",
    "append_run",
    "load_run_package",
    "read_run",
    "rel_paths",
    "run_file",
    "run_metrics_summary",
    "update_run",
    "write_evaluation",
    "write_run",
]
