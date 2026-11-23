"""``python -m risklab_mdl`` —— 等价于 ``risklab-mdl`` 控制台入口（启动 MCP stdio 服务）。"""

from .server import main

if __name__ == "__main__":
    raise SystemExit(main())
