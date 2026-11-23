"""risklab_mdl.tools —— MCP 工具薄封装。

职责：参数校验 → 调下层计算 → 产物落盘 → 返回结构化 JSON（三态协议）。
每个模块只暴露 MCP 工具函数；公共响应构造在 :mod:`risklab_mdl.protocol`，
项目状态读取在 :mod:`risklab_mdl.project`。
"""

from __future__ import annotations
