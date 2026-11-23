"""预处理家族工具（契约 §3.3）：``preprocess_check`` / ``preprocess_apply``。

* ``preprocess_check`` —— 入模前检查，落 ``preprocess/check.json``（逻辑见
  :mod:`risklab_mdl.preprocess.check`）；
* ``preprocess_apply`` —— 落实缺失/编码/盖帽/手动分箱规格，落四份规格产物 +
  处理后的数据 + 处理摘要（逻辑见 :mod:`risklab_mdl.preprocess.apply`）。

本模块只做工具注册用的一层封装（签名与 MCP 注册严格对齐），不重复实现业务逻辑。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from ..preprocess.apply import preprocess_apply as _preprocess_apply
from ..preprocess.check import preprocess_check as _preprocess_check

__all__ = ["preprocess_apply", "preprocess_check"]


def preprocess_check(
    project_dir: str, config: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """入模前检查：缺失、文本列、类别基数、编码/盖帽建议（must_fix / suggested）。"""
    return _preprocess_check(project_dir, config)


def preprocess_apply(
    project_dir: str, spec: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """执行预处理规格：``{missing, encode, cap, binning}`` → 规格产物 + 处理数据。"""
    return _preprocess_apply(project_dir, spec)
