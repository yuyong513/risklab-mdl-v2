"""risklab-mdl-v2 —— 风控建模计算层（MCP Server，跨平台）。

分层：
- ``platform/``  跨平台基础设施：原子写、文件锁、路径校验、编码安全 JSON
- ``artifacts/`` 产物目录契约：``mdl-<csv stem>/`` 布局、manifest、run 目录分配
- ``core/``      纯计算层（分箱 / WOE / IV / 指标 / 划分 / 训练 / 调参）
- ``data/`` ``features/`` ``preprocess/`` ``models/`` ``runs/`` ``charts/``
                 读数据、画像、特征分析、预处理、模型适配与 run 存储
- ``protocol.py``  MCP 三态响应（ok / needs_user_input / error）构造
- ``project.py``   项目状态读取（manifest 角色、CSV 定位）
- ``tools/``     MCP 工具薄封装（20 个工具）
- ``server.py``  MCP stdio 服务端（工具注册 + 启动）
- ``doctor.py``  环境自检

产品版本唯一真源：本文件的 ``__version__`` 与 ``pyproject.toml`` 的
``project.version`` 必须一致（由 ``tests/test_doctor.py`` 守护）。
"""

#: 产品版本。✱ 与 pyproject.toml 的 project.version 同步（test_doctor.py 守护）。
__version__ = "2.0.0"

__all__ = ["__version__"]
