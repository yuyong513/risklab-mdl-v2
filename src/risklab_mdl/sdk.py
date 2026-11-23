"""MCP SDK 适配层：屏蔽 ``mcp`` 2.x ``MCPServer`` 与 1.x ``FastMCP`` 的差异。

★ 为什么要有这一层
------------------
早期版本写死 ``from mcp.server.mcpserver import MCPServer``（mcp 2.x 风格）。
``MCPServer`` 是 2.x 相对 1.x ``FastMCP`` 的改名，用户环境里装到哪个大版本
不应导致「导入即崩」。这里把「候选导入路径 → 第一个可用的类」收敛成唯一实现，
``server.py`` 与 ``doctor.py`` 共用同一份探测结果，避免两处各写一套。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

#: 候选 ``(模块路径, 类名)``，按优先级排列：2.x 优先，1.x 兜底。
SERVER_CLASS_CANDIDATES: Tuple[Tuple[str, str], ...] = (
    ("mcp.server.mcpserver", "MCPServer"),
    ("mcp.server.fastmcp", "FastMCP"),
)


class McpSdkError(RuntimeError):
    """mcp SDK 不可用（未安装，或候选类都不存在）。"""


@dataclass(frozen=True)
class ServerClass:
    """解析结果：用哪个模块的哪个类。"""

    module: str
    name: str
    cls: Any

    @property
    def qualname(self) -> str:
        return f"{self.module}.{self.name}"


def resolve_server_class() -> ServerClass:
    """返回第一个可导入的服务器类；都不可用则抛 :class:`McpSdkError`。"""
    problems: List[str] = []
    for module_path, class_name in SERVER_CLASS_CANDIDATES:
        try:
            module = __import__(module_path, fromlist=[class_name])
        except Exception as exc:  # noqa: BLE001 - 逐个候选试，最后统一报
            problems.append(f"{module_path}（{type(exc).__name__}: {exc}）")
            continue
        cls = getattr(module, class_name, None)
        if cls is None:
            problems.append(f"{module_path} 中没有 {class_name}")
            continue
        return ServerClass(module=module_path, name=class_name, cls=cls)
    raise McpSdkError(
        "找不到可用的 MCP 服务器类，尝试过：" + "；".join(problems),
    )


def make_server(name: str, version: str) -> Any:
    """按解析结果构造服务器实例。

    ★ 1.x ``FastMCP`` 不接受 ``version``
      2.x ``MCPServer`` 的签名是 ``(name, version=...)``，而 1.x ``FastMCP``
      只有 ``name`` —— 直接传 ``version`` 会 ``TypeError: unexpected keyword
      argument``，表现为「换了 SDK 大版本就起不来」。这里先按 2.x 签名构造，
      遇到 ``TypeError`` 回退为 ``cls(name=name)``：能接受 version 的环境不受
      影响，1.x 环境也能起来（版本号只用于 serverInfo 展示）。
    """
    resolved = resolve_server_class()
    try:
        return resolved.cls(name=name, version=version)
    except TypeError:
        # 1.x FastMCP 不接受 version 关键字；回退到最小签名
        return resolved.cls(name=name)


def describe_api() -> Dict[str, Any]:
    """给 doctor 用：报告 mcp 版本、命中的导入路径、每个候选的可导入性。"""
    sdk_version: Any
    try:
        import importlib.metadata as _md

        sdk_version = _md.version("mcp")
    except Exception:  # noqa: BLE001
        sdk_version = None
    candidates: List[Dict[str, Any]] = []
    resolved: Any = None
    for module_path, class_name in SERVER_CLASS_CANDIDATES:
        entry: Dict[str, Any] = {"module": module_path, "class": class_name, "importable": False}
        try:
            module = __import__(module_path, fromlist=[class_name])
            entry["importable"] = getattr(module, class_name, None) is not None
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
        candidates.append(entry)
        if resolved is None and entry["importable"]:
            resolved = f"{module_path}.{class_name}"
    return {
        "sdk_version": sdk_version,
        "resolved_api": resolved,
        "candidates": candidates,
    }
