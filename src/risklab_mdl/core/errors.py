"""包内异常体系。

DataError / ConfigError / ModelError 一律继承 RiskLabError，
执行层只需捕获一个基类，不依赖任何外部异常类型。
"""

from __future__ import annotations

__all__ = [
    "RiskLabError",
    "DataError",
    "ConfigError",
    "ModelError",
]


class RiskLabError(Exception):
    """所有业务异常的基类。"""

    code: str = "RISKLAB_ERROR"

    def __init__(self, message: str, *, detail: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class DataError(RiskLabError):
    """数据本身的问题：缺列、类型不符、取值域冲突、标签缺失等。"""

    code = "DATA_ERROR"


class ConfigError(RiskLabError):
    """配置缺失或非法。"""

    code = "CONFIG_ERROR"


class ModelError(RiskLabError):
    """训练 / 推理异常：依赖缺失、模型不可用、打分失败等。

    ★ 由本包实现 的 ``risklab-mcp/errors.py``。
      精筛（gain 迭代）需要 LightGBM，依赖缺失/训练失败走这个异常 ——
      与配置错误（ConfigError）区分开：前者是**环境问题**，
      后者是**参数问题**，处理路径不同。
    """

    code = "MODEL_ERROR"
