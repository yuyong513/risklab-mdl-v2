"""MCP 三态响应载荷构造（``ok`` / ``needs_user_input`` / ``error``）。

服务端不渲染弹窗、不维护会话状态：每个工具只返回一个结构化 JSON，
由 skill / agent 负责呈现。所有工具模块共用这里的构造器，让
``{ok, status, data, artifacts, warnings, error}`` 的形状与错误码映射只定义一次。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .core.errors import ConfigError, DataError, RiskLabError
from .data.loader import to_jsonable
from .platform.fs import PlatformError

__all__ = [
    "relative_paths",
    "result_error",
    "result_from_exception",
    "result_needs_input",
    "result_ok",
    "target_question",
]


def result_ok(
    data: Optional[Dict[str, Any]] = None,
    *,
    artifacts: Optional[Sequence[str]] = None,
    warnings: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """``status="ok"`` 的载荷。"""
    return {
        "ok": True,
        "status": "ok",
        "data": to_jsonable(data or {}),
        "artifacts": [str(a) for a in (artifacts or [])],
        "warnings": [str(w) for w in (warnings or [])],
        "error": None,
    }


def result_error(
    code: str,
    message: str,
    hint: str = "",
    *,
    data: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Sequence[str]] = None,
    warnings: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """``status="error"`` 的载荷（``ok=False``，``error`` 非空）。"""
    return {
        "ok": False,
        "status": "error",
        "data": to_jsonable(data or {}),
        "artifacts": [str(a) for a in (artifacts or [])],
        "warnings": [str(w) for w in (warnings or [])],
        "error": {"code": str(code), "message": str(message), "hint": str(hint)},
    }


def result_needs_input(
    question: Dict[str, Any],
    context: Dict[str, Any],
    *,
    warnings: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """``status="needs_user_input"`` 的载荷（服务端只给数据，不渲染弹窗）。"""
    return {
        "ok": False,
        "status": "needs_user_input",
        "data": to_jsonable({"question": question, "context": context}),
        "artifacts": [],
        "warnings": [str(w) for w in (warnings or [])],
        "error": None,
    }


def result_from_exception(
    exc: Exception,
    *,
    data: Optional[Dict[str, Any]] = None,
    warnings: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """把内部异常映射成契约错误码（平台层直接沿用其 ``code``）。"""
    if isinstance(exc, PlatformError):
        payload = exc.to_payload()
        return result_error(
            payload["code"],
            payload["message"],
            payload.get("hint", ""),
            data=data,
            warnings=warnings,
        )
    if isinstance(exc, ConfigError):
        return result_error(
            "config_invalid",
            exc.message,
            "检查 config 字段名与取值（见 docs/data-feature.md）",
            data=data,
            warnings=warnings,
        )
    if isinstance(exc, DataError):
        return result_error(
            "analysis_failed",
            exc.message,
            "检查数据本身（缺失/类型/标签取值）后重试",
            data=data,
            warnings=warnings,
        )
    if isinstance(exc, RiskLabError):
        return result_error(
            "analysis_failed", exc.message, "", data=data, warnings=warnings
        )
    if isinstance(exc, ImportError):
        return result_error(
            "dependency_missing",
            f"缺少运行期依赖：{exc}",
            "python -m pip install matplotlib",
            data=data,
            warnings=warnings,
        )
    if isinstance(exc, (ValueError, KeyError)):
        return result_error(
            "config_invalid",
            str(exc),
            "按 docs/data-feature.md 的入参说明检查字段名与取值",
            data=data,
            warnings=warnings,
        )
    return result_error(
        "analysis_failed",
        f"{type(exc).__name__}: {exc}",
        "这是非预期失败；请把该消息与复现步骤反馈给维护者（不要当作正常结果使用）",
        data=data,
        warnings=warnings,
    )


def relative_paths(layout: Any, paths: Sequence[Any]) -> List[str]:
    """把产物路径转成相对项目目录的 POSIX 路径（``artifacts`` 字段用）。"""
    return [layout.rel(path) for path in paths]


def target_question(
    frame: pd.DataFrame, candidates: Sequence[str], *, reason: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """构造 ``dataset_confirm_target`` 问题载荷（契约 §1.1）。"""
    label_counts: Dict[str, int] = {}
    best: Optional[str] = None
    for name in candidates:
        if name in frame.columns:
            counts = frame[name].value_counts(dropna=True)
            if 2 <= len(counts) <= 10:
                best = name
                label_counts = {str(k): int(v) for k, v in counts.items()}
                break
    question = {
        "id": "dataset_confirm_target",
        "text": f"{reason}请确认目标字段与任务类型（config.target={{column, task, labels?}}）",
        "options": [
            {"value": "binary", "label": "二分类（如 0/1 好坏）"},
            {"value": "multiclass", "label": "多分类（3 类及以上）"},
        ],
        "default": "binary",
        "fields": [
            {"name": "target_column", "type": "string", "required": True},
            {"name": "task", "type": "string", "required": False},
            {"name": "labels", "type": "object", "required": False},
        ],
    }
    context = {
        "columns": [str(c) for c in frame.columns][:50],
        "column_count": int(frame.shape[1]),
        "rows": int(frame.shape[0]),
        "candidates": list(candidates),
        "suggested": best,
        "label_counts": label_counts,
        "example": {
            "column": best or "y",
            "task": "binary",
            "labels": {"1": "bad", "0": "good"},
        },
    }
    return question, context
