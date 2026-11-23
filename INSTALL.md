# 安装与使用（risklab-mdl-v2 2.0.0）

覆盖四件事：**装依赖**、**跑自检**、**接客户端**、**验收/排障**。
编排流程怎么用见 [docs/skill使用指南.md](docs/skill使用指南.md)；工具契约见
[docs/tool-contract.md](docs/tool-contract.md)。

## 1. 环境要求

| 项 | 要求 | 说明 |
|---|---|---|
| Python | **3.11+** | 本机实测 3.11.9；`tomllib`、类型语法依赖 3.11 |
| 操作系统 | Windows / macOS / Linux | 跨平台，无 `fcntl` 等 POSIX-only 依赖 |
| 网络 | 首次安装需要 | 依赖从 PyPI 拉取（xgboost 约 100 MB） |
| 磁盘 | ≥ 1.5 GB 空闲 | xgboost + scipy + matplotlib 等 wheel |

运行时依赖（`pyproject.toml`）：`mcp>=2.0.0`、`pandas>=2.0`、`numpy>=1.24`、
`scikit-learn>=1.3`、`lightgbm>=4.0`、`xgboost>=2.0`、`matplotlib>=3.7`、
`hyperopt>=0.2.7`、`openpyxl>=3.1`、`PyYAML>=6.0`、`filelock>=3.12`。

可选：`pyarrow`（让预处理产物落 `processed.parquet`；不装则降级 `processed.csv`，
功能与读数不受影响，`warnings` 会明示）。

## 2. 安装

```powershell
cd risklab-mdl-v2

# 建议先建虚拟环境（也可装进系统 Python；关键是客户端要指向同一个解释器）
python -m venv .venv
.\.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux

# 可编辑安装（含 pytest 开发依赖）
python -m pip install -e ".[dev]"
```

### 网络慢时

卡在 xgboost / scipy 下载时改用镜像：

```powershell
python -m pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 60 --retries 3
```

### 安装后确认

```powershell
python -c "import risklab_mdl, sys; print(risklab_mdl.__version__, sys.executable)"
python -m risklab_mdl.doctor            # 六项自检，退出码 0 = 可用
```

两个 console script（装包后可用）：

| 命令 | 作用 |
|---|---|
| `risklab-mdl` / `python -m risklab_mdl` | 启动 MCP stdio 服务（客户端自动拉起，无需常驻） |
| `risklab-mdl-doctor` / `python -m risklab_mdl.doctor` | 六项环境自检，退出码 0=可用 / 1=有 FAIL |

## 3. 环境自检（doctor）

```powershell
python -m risklab_mdl.doctor            # 中文摘要
python -m risklab_mdl.doctor --json     # 结构化输出（部署脚本 / 工单用）
```

六项探针：

| # | 探针 | 检查什么 | FAIL 含义 |
|---|---|---|---|
| 1 | `python` | Python ≥ 3.11；pyproject 与包内版本号一致 | 解释器过低（版本号不一致只 WARN） |
| 2 | `dependencies` | 11 个运行时依赖逐个 import 并取版本 | 有依赖没装上（给出 pip 命令） |
| 3 | `mcp_sdk` | MCP SDK 服务器类可导入（`MCPServer` / 1.x `FastMCP`） | mcp 版本不匹配 |
| 4 | `matplotlib` | backend = **Agg** 且能真的导出 PNG | 无头环境下画不了图 |
| 5 | `chinese_font` | 中文字体 fallback 链 + 中文渲染无缺字形告警 | **只告警不阻断**（图中中文可能方块） |
| 6 | `artifact_dir` | 临时目录实地跑 `mdl-<stem>/` + manifest 往返 + run 目录命名 | 写盘链路坏了（平台层缺陷） |

退出码：**任一 FAIL ⇒ 1**（打印可复制的修复命令）；只有 WARN ⇒ **0**（能跑但有降级）。

> 注意：`doctor` 是 **CLI**，不是 MCP 工具；MCP 侧的环境探针是 `server_probe`（见 §4）。

## 4. 启动 MCP 服务

```powershell
python -m risklab_mdl                 # 默认 stdio（MCP 客户端标准方式）
python -m risklab_mdl --transport sse # 需要 HTTP 传输时
```

服务端**只往 stdout 写 JSON-RPC 协议帧**；日志与错误一律走 stderr。
MCP 会话第一条建议调 `server_probe`（无入参）：返回 `version / platform / deps /
artifacts_check`；依赖缺失或产物链路不可用 ⇒ `error.code=environment_not_ready`。

## 5. 客户端接入（三步）

三份样例 `mcp/mcp.claude.json`、`mcp/mcp.dsh.json`、`mcp/mcp.workbuddy.json` 内容一致，
因为三家 stdio 接入形状相同：

```json
{
  "mcpServers": {
    "risklab-mdl": {
      "command": "python",
      "args": ["-m", "risklab_mdl"],
      "env": {}
    }
  }
}
```

**① 装计算层**（§2，确认 `python -m risklab_mdl.doctor` 退出码 0）。
**② 装 skill**：把 `skill/` 目录复制到客户端的 skills 目录，目录名建议 `risklab-mdl`
（与 `SKILL.md` 的 `name: risklab-mdl` 一致）。
**③ 配 MCP**：把对应 `mcp.<client>.json` 合并进客户端 MCP 配置；`command` 必须是
**装了本包的那个解释器**的绝对路径（GUI 客户端常常继承不到终端 PATH）。

| 客户端 | skill 位置（常规） | MCP 配置位置 | 本机实测 |
|---|---|---|---|
| Claude Code | `~/.claude/skills/risklab-mdl/` 或项目 `.claude/skills/risklab-mdl/` | 项目 `.mcp.json`；或 `claude mcp add risklab-mdl -- python -m risklab_mdl` | 协议层实测，客户端界面**未实测** |
| DSH | 工作区 `.dsh/skills/risklab-mdl/`（或全局技能目录，按版本） | 并入 DSH 的 MCP 配置（`mcpServers` 段） | 协议层实测，落盘位置**按版本为准** |
| WorkBuddy | 客户端技能目录，目录名 `risklab-mdl` | 工作区/用户级 `mcp.json` | **未实测**（本机未安装） |

> 三份配置的**协议层**已实测：用配置里的 `command`/`args` 起真进程做
> `initialize` + `tools/list`，返回 20 个工具（[tests/test_skill_contract.py](tests/test_skill_contract.py)）。
> **客户端界面能否加载、skill 放在哪个目录**未在本机实测，装完请按
> [docs/skill使用指南.md](docs/skill使用指南.md) §6.3 核对。
>
> 若 `python` 不在客户端进程的 PATH 里，把 `command` 换成
> `python -c "import sys;print(sys.executable)"` 打印的绝对路径。

## 6. 验收命令

```powershell
cd risklab-mdl-v2

# ① 快速 e2e：合成小样本走完整条链路（约 7 s）
python -m pytest tests/test_e2e_sample.py -q

# ② 全量测试（约 4–9 分钟，取决于机器与是否装了 xgboost/lightgbm）
python -m pytest tests -q

# ③ 产物树契约校验（传入 project_dir；完整流程跑完后才有全部产物）
python scripts\check_layout.py "<...>\mdl-<CSV 文件名>"
```

`check_layout.py` 一次列全所有缺失项，失败退出码 1。

## 7. 常见问题

| 现象 | 原因 | 处置 |
|---|---|---|
| `doctor` 报 `dependencies` FAIL | 依赖没装齐（或装到别的解释器） | 按打印的 pip 命令重装；核对 `python -c "import sys;print(sys.executable)"` 与客户端 `command` 一致 |
| `doctor` 报 `mcp_sdk` FAIL | mcp 版本不提供 `MCPServer`/`FastMCP` | `python -m pip install "mcp>=2.0.0"` |
| `doctor` 报 `artifact_dir` FAIL | 系统临时目录不可写 | 检查 `TEMP`/`TMP`；`python -c "import tempfile;print(tempfile.gettempdir())"` |
| `chinese_font` WARN / 图里中文方块 | 缺 CJK 字形 | Windows 启用微软雅黑；Linux 装 `fonts-noto-cjk`；不阻断建模 |
| 客户端报"命令不存在" | 客户端进程 PATH 没有 python | `command` 用绝对路径（§5） |
| 客户端工具列表为空 | 读的是另一份配置文件 / 未重启 | 确认 `mcpServers` 键名；重启客户端 |
| `project_dir_not_writable` | CSV 在只读/受保护目录 | 把 CSV 复制到可写目录再给路径（工具**不会**静默换位置） |
| `project_dir_not_found` | 把 CSV 路径当成了 `project_dir` | `project_dir` = `…/mdl-<CSV 文件名>`，由 `project_init` 返回 |
| `path_not_readable` / `encoding_undecodable` | 路径拼错 / CSV 是 ANSI/GBK/二进制 | 用绝对路径；另存 UTF-8 或 GB18030 |
| 训练 train 0.99 / val 0.65 | 宽表 + 少坏样本的过拟合，不是程序错误 | 加强 `feature_refine`、降 `num_leaves`/`max_depth`、补坏样本 |
| PSI 没算 | 无时间列 / 只有一个月 / 基期覆盖全部月 | 看 `psi.reason`；不能声称跨期稳定 |
| `threshold_not_met` | KS/PSI 未达确认阈值 | 看 `ks_gap`/`psi_gap` 差距值，由用户决定放宽还是重训 |
| 安装卡在 xgboost/scipy | PyPI 直连慢 | 用清华镜像重试（§2） |

## 8. 卸载

```powershell
python -m pip uninstall risklab-mdl-v2
```

建模产物不在包内，而是落在 `CSV 同目录/mdl-<CSV 文件名>/`；skill 目录与 MCP 配置
按安装位置手工删除。

## 9. 许可

[PolyForm Noncommercial License 1.0.0](LICENSE)：允许个人学习、研究、实验、教学等
**非商业用途**；**禁止商业使用**。修改与再分发需保留许可与 `Required Notice`。
商业授权请联系作者（GitHub Issue）。
