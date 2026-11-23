"""可选重依赖的惰性引导（lightgbm / xgboost / matplotlib）。

★ 为什么单独一个模块
--------------------
旧版 ``core/`` 在 ``train.py`` / ``refine.py`` 顶部写 ``from ..adapters.lgbm_env
import ensure_lightgbm``，把「重依赖是否安装」变成了**包级导入期**的事实：
只要 import 了 core，就必须先装 lightgbm —— 而 core 里绝大多数模块（分箱、
指标、PSI、相关性）根本不需要它。MCP 服务端启动时若因一条路径缺库就整包起不来，
是典型的「失败不可见」反例（缺库信息埋在 import traceback 里，工具层拿不到 error 载荷）。

⇒ 本模块给出统一的惰性入口：

* ``require_lightgbm()`` / ``require_xgboost()`` / ``require_matplotlib()``
  在**函数体内** import，缺库时抛 :class:`~risklab_mdl.core.errors.ModelError`
  （``code="MODEL_ERROR"``，``detail`` 里带 ``pip install ...`` 提示）；
* **本模块顶层不 import 任何第三方重依赖**（连 numpy 都不需要）——
  因此 ``import risklab_mdl.core.deps`` 不会拉起 lightgbm / xgboost / matplotlib，
  ``tests/test_core_imports.py`` 对此有 ``sys.modules`` 断言。

★ 与旧实现的差异
----------------
旧 ``adapters/lgbm_env.py`` 还负责 macOS ARM 的 ``libomp`` 注入与 ``execve`` 重启进程。
新项目跨平台优先、且不再做「重启自己」这种副作用极强的事 ——
libomp 缺失同样以 ``ModelError`` 报出，由调用方（工具层）转成
``dependency_missing`` 载荷，提示用户装 scikit-learn / libomp。
"""

from __future__ import annotations

from typing import Any

from .errors import ModelError

__all__ = [
    "DEPENDENCY_HINTS",
    "require_lightgbm",
    "require_matplotlib",
    "require_xgboost",
]

#: 缺库时的安装提示（错误码与提示文案见 ``docs/tool-contract.md`` §6）
DEPENDENCY_HINTS: dict[str, str] = {
    "lightgbm": "pip install lightgbm",
    "xgboost": "pip install xgboost",
    "matplotlib": "pip install matplotlib",
}


def _missing(name: str, exc: BaseException) -> ModelError:
    hint = DEPENDENCY_HINTS.get(name, f"pip install {name}")
    return ModelError(
        f"缺少可选依赖 {name}（import 失败：{type(exc).__name__}: {exc}）。"
        f"请执行：{hint}。"
        f"★ 本依赖按需加载：只有真正用到 {name} 的步骤才需要它，"
        f"其余 core 计算（分箱/指标/PSI/相关性/划分）不受影响。",
        detail={"dependency": name, "install": hint, "cause": repr(exc)},
    )


def require_lightgbm() -> Any:
    """返回 ``lightgbm`` 模块；缺失时抛 :class:`ModelError`。

    ★ 只在**函数体**里 import —— 模块级 import 会让整个 core 包依赖 lightgbm。
    """
    try:
        import lightgbm  # noqa: PLC0415  (惰性导入是本函数的语义)
    except ImportError as exc:
        raise _missing("lightgbm", exc) from exc
    except OSError as exc:  # macOS ARM 缺 libomp 等运行期链接错误
        raise _missing("lightgbm", exc) from exc
    return lightgbm


def require_xgboost() -> Any:
    """返回 ``xgboost`` 模块；缺失时抛 :class:`ModelError`。"""
    try:
        import xgboost  # noqa: PLC0415
    except ImportError as exc:
        raise _missing("xgboost", exc) from exc
    except OSError as exc:
        raise _missing("xgboost", exc) from exc
    return xgboost


def require_matplotlib() -> Any:
    """返回 ``matplotlib`` 模块；缺失时抛 :class:`ModelError`。

    ★ core 层不出图（图表在 tools/charts 层），这里只是给 tools 层一个统一入口，
      保证「缺 matplotlib 时报错文案一致」。
    """
    try:
        import matplotlib  # noqa: PLC0415
    except ImportError as exc:
        raise _missing("matplotlib", exc) from exc
    except OSError as exc:
        raise _missing("matplotlib", exc) from exc
    return matplotlib
