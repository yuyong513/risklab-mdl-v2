# skill 使用指南（安装 · 配置 · 首次会话 · 排障）

> 适用版本：`risklab-mdl-v2` 2.0.0（Python ≥ 3.11）。
> 本指南覆盖两件事：**把 skill 装到 agent 客户端**、**把 MCP 服务端接上客户端**。
> skill 本身见 [../skill/SKILL.md](../skill/SKILL.md)，工具契约见 [tool-contract.md](tool-contract.md)。

## 0. 三条腿，缺一不可

| 层 | 装什么 | 装在哪 |
|---|---|---|
| L2 计算层 | Python 包 `risklab-mdl-v2`（提供 `python -m risklab_mdl`） | 任意 Python 3.11+ 环境（建议 venv） |
| L1 编排层 | `skill/` 目录（`SKILL.md` + `references/`） | 客户端的 skills 目录 |
| L3 交互层 | MCP 配置（`mcp/mcp.<client>.json`） | 客户端的 MCP 配置文件 |

只装 skill 不接 MCP：agent 会"知道流程但调不动工具"。只接 MCP 不装 skill：工具能调，
但没人告诉 agent 该先问目标字段、该怎么呈现结果。

## 1. 安装计算层

```powershell
cd risklab-mdl-v2
python -m venv .venv
.\.venv\Scripts\activate                    # macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

网络卡时换镜像：

```powershell
python -m pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 60 --retries 3
```

**可选但推荐**：`python -m pip install pyarrow`（让预处理产物落 `processed.parquet` 而不是降级的
`processed.csv`，`warnings` 里那条"pyarrow 未安装"会消失，读数一样但更快）。

装完确认两个入口都在：

```powershell
python -m risklab_mdl.doctor          # 六项自检，退出码 0 = 可用
python -c "import risklab_mdl, sys; print(risklab_mdl.__version__, sys.executable)"
```

## 2. 安装 skill

skill 目录结构：

```text
skill/
  SKILL.md                     # 触发条件 + 总原则 + 9 环节流程 + 交互协议 + 边界
  references/
    tools.md                   # 20 个工具：何时调用 / 入参示例 / 结果解读
    workflow.md                # 9 环节话术与输入示例
    models.md                  # 5 算法场景、默认参数、默认调参空间
    metrics.md                 # KS/AUC/Gini/LIFT/PSI/IV 口径与确认阈值
    errors.md                  # 错误码与修复指引（含 doctor CLI）
    artifacts.md               # mdl-<csv名>/ 目录与产物解读
```

### 2.1 Claude Code

个人级（所有项目可用）：

```powershell
# Windows
xcopy /E /I /Y "skill" "%USERPROFILE%\.claude\skills\risklab-mdl"
# macOS / Linux
cp -R risklab-mdl-v2/skill ~/.claude/skills/risklab-mdl
```

项目级（只在某个项目可用，推荐）：

```powershell
mkdir .claude\skills -Force
xcopy /E /I /Y risklab-mdl-v2\skill .claude\skills\risklab-mdl
```

目录名建议就叫 `risklab-mdl`，与 `SKILL.md` 里的 `name: risklab-mdl` 一致。

### 2.2 DSH

DSH 读取技能目录（工作区级 `.dsh/skills/` 或全局技能目录，按客户端版本为准）。
把自己的 skill 放进去即可：

```powershell
mkdir .dsh\skills -Force
xcopy /E /I /Y risklab-mdl-v2\skill .dsh\skills\risklab-mdl
```

DSH 已经支持 MCP（本会话就挂着一个 MCP 资源服务器），把 `mcp/mcp.dsh.json` 的内容并入
DSH 的 MCP 配置即可（见 §3.2）。

### 2.3 WorkBuddy

WorkBuddy 走 `mcp.json`（工作区或用户级，按客户端版本为准）：

```powershell
# 工作区级
copy risklab-mdl-v2\mcp\mcp.workbuddy.json .\mcp.json
```

skill 侧的安装方式与 Claude Code 类似，把 `skill/` 目录放到 WorkBuddy 的技能目录，
目录名 `risklab-mdl`。

> **实测范围坦白**：本机只装了 DSH（Claude Code 与 WorkBuddy 均未安装）。
> 三份 MCP 配置的**协议层已实测**（用配置里的 `command`/`args` 起真进程做
> `initialize` + `tools/list`，返回 20 个工具，见 §5），
> 但**三家客户端的界面/落盘位置未在本机实测**——上表里的路径是各客户端的常规位置，
> 装完请按 §6.3 的步骤验证。

## 3. 配置 MCP

三份样例（`mcp/mcp.claude.json`、`mcp/mcp.dsh.json`、`mcp/mcp.workbuddy.json`）内容一致，
因为三家客户端的 stdio 接入形状相同：`mcpServers → <名字> → command/args/env`。差异只在**放哪里**和
**command 怎么解析**。

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

### 3.1 相对 vs 绝对路径（最常见的坑）

| 写法 | 什么时候用 | 注意 |
|---|---|---|
| `"command": "python"` | Python 在客户端进程的 `PATH` 里 | **GUI 客户端常常继承不到你终端的 PATH**；报"命令找不到"就换绝对路径 |
| `"command": "C:/path/to/python.exe"` | 客户端起不来 python 时 | 用 `python -c "import sys;print(sys.executable)"` 的输出；JSON 里用 `/` 或 `\\` |
| `"command": "<项目根>/.venv/Scripts/python.exe"` | 装在 venv 里 | **venv 路径必须绝对**，且必须是装了 `risklab-mdl-v2` 的那个解释器 |
| `"args": ["-m", "risklab_mdl"]` | 包已 `pip install -e` 到该解释器 | 若没装包，等价写法 `["<项目根>/src/risklab_mdl/__main__.py"]` 不可靠——请先装包 |
| `"command": "risklab-mdl"` | 用 console script | 只有 `Scripts` 目录在 PATH 里才有效 |

自检一行：

```powershell
& "<上面打印的解释器绝对路径>" -m risklab_mdl.doctor
```

能跑通，就把这个解释器路径写进 `command`。

### 3.2 各客户端放置位置

| 客户端 | 文件 | 作用域 | 备注 |
|---|---|---|---|
| Claude Code | `<项目根>/.mcp.json` | 项目 | 也支持用户级配置；`mcpServers` 同名覆盖 |
| Claude Code | 或用 `claude mcp add risklab-mdl -- python -m risklab_mdl` | — | CLI 会写进自己的配置 |
| DSH | 并入 DSH 的 MCP 配置（把 `mcpServers` 段整体合并） | 用户/工作区 | 字段形状同 Claude Code；具体路径以客户端版本为准（**未实测**） |
| WorkBuddy | `mcp.json`（工作区根或用户目录） | 工作区/用户 | 同上（**未实测**） |

JSON 是**严格格式**：不能写注释、不能有多余逗号。想保留说明就写在本文档里。

### 3.3 服务端启动方式

客户端会自己起进程，不需要手工常驻；想单独验证服务端：

```powershell
python -m risklab_mdl                  # stdio（客户端标准方式），Ctrl+C 退出
python -m risklab_mdl --transport sse  # 需要 HTTP 传输时
```

服务端**只往 stdout 写 JSON-RPC 协议帧**，日志与错误一律走 stderr——所以终端里"看起来没输出"是正常的。

## 4. 首次会话示例

用户说：

> 帮我用 `D:/work/sample_2000.csv` 建个信用风险模型。

agent（按 skill 执行）应当：

1. **环境探针**：调 `server_probe`。`ok=false` ⇒ 停，报依赖缺失 + `doctor` 修复命令。
2. **建项目**：`project_init({"csv_path": "D:/work/sample_2000.csv"})` →
   项目目录 `D:/work/mdl-sample_2000`（已存在则复用，不覆盖历史产物）。
3. **读表 + 列清单**：`dataset_load` → `dataset_columns`，然后**先问目标字段**：

   > 数据 2000 行 × 696 列，推断目标字段是 `y`（0/1，坏率 1.75%）。确认用 `y` 做二分类，
   > 且 `1` = 坏（违约）吗？

4. 用户确认后：`dataset_profile({"target": {"column": "y", "task": "binary",
   "labels": {"1": "bad", "0": "good"}}})` → 报告坏率、缺失 TOP、图表路径。
5. `feature_analyze({"method": "quantile", "bins": 8})` → IV TOP10 + 跳过原因。
6. `model_catalog({})` → 5 算法能力表，推荐 `lgb`；`feature_prescreen` → 存活数 + 淘汰原因。
7. `preprocess_check` → 给"推荐 spec 草案"请用户确认 → `preprocess_apply` → `feature_refine`（可选）。
8. `dataset_split({"mode": "stratified", "ratios": [0.7, 0.2, 0.1], "seed": 42})`。
9. `model_train({"algorithm": "lgb", "task": "binary"})` → `model_evaluate` →
   `model_confirm`（默认 KS ≥ 0.25、PSI ≤ 0.10）→ `report_build`。

一段典型的真实回复（本机 `sample_2000` 实测数字）：

> 训练完成（run `20261116T213542`，60 个特征）：
> | 子集 | KS | AUC | Gini | LIFT@10% | n / bad |
> |---|---|---|---|---|---|
> | train | 0.9992 | 0.99996 | 0.9999 | 10.00 | 1200 / 21 |
> | val | 0.4020 | 0.6579 | 0.3159 | 1.43 | 400 / 7 |
> | test | 0.4064 | 0.7023 | 0.4046 | 2.86 | 400 / 7 |
>
> ⚠ train 与 val 差距 0.34，属明显过拟合（691 列 + 35 个坏样本的必然结果），建议加强筛选或降复杂度。
> 分数 PSI：基期 2026-01，最差月 2026-04 = 0.115，超过 0.10 常规线。
> 确认闸门：KS 0.402 ≥ 0.25 ✅ / PSI 0.115 > 0.10 ❌ —— 是否放宽 PSI 阈值（需你确认风险）或先重训？

**注意 skill 的硬规则**：不伪造数值、失败必须可见、不擅自改阈值。若 agent 试图"为了跑通"把阈值
改小再调一次，把 `skill/SKILL.md` §2 的硬规则贴给它。

## 5. 验证：MCP 握手（不用客户端也能验）

```powershell
cd risklab-mdl-v2
python -m pytest tests/test_skill_contract.py -q
```

其中 `test_mcp_config_samples_handshake` 会读三份 `mcp/*.json`，用配置里的 `command`/`args`
起真进程，做 `initialize` + `tools/list`，断言拿到 **20 个工具**。
这证明"配置内容是可用的"，但不证明"某个客户端的界面能加载它"。

手工验一遍同样的东西：

```powershell
python -m pytest tests/test_server_smoke.py -q     # 同一套协议的 stdio 冒烟
```

## 6. 故障排查

### 6.1 依赖缺失 / 环境不可用

症状：`server_probe` 返回 `environment_not_ready`；或客户端连上但一调工具就报
`dependency_missing`。

```powershell
python -m risklab_mdl.doctor            # 中文摘要
python -m risklab_mdl.doctor --json     # 结构化（给工单用）
```

| doctor 探针 | FAIL 含义 | 处置 |
|---|---|---|
| `python` | 解释器 < 3.11 | 装 3.11+，并把 `command` 指向它 |
| `dependencies` | 有包没装（或装到别的解释器） | 按打印的 pip 命令装；核对 `sys.executable` 与配置里的 `command` 是同一个 |
| `mcp_sdk` | mcp 版本不提供服务器类 | `python -m pip install "mcp>=2.0.0"` |
| `matplotlib` | 强制 `Agg` 后仍不可用 | 重装 matplotlib；无头环境必须能用 Agg |
| `chinese_font` | 只 WARN，不阻断 | Windows 装/启用微软雅黑；Linux 装 `fonts-noto-cjk`；图里中文可能变方块 |
| `artifact_dir` | 临时目录不可写 | 查 `TEMP`/`TMP`：`python -c "import tempfile;print(tempfile.gettempdir())"` |

**任何 FAIL 都不要继续建模流程**——失败必须让用户看见。

### 6.2 路径与编码

| 现象 | 原因 | 处置 |
|---|---|---|
| `path_not_readable` 但文件确实在 | 把目录/CSV 搞混，或路径里有引号/空格未处理 | 用绝对路径；JSON 里不用 `"` 包路径，用 `/` 或 `\\` |
| `project_dir_not_found` | 把 CSV 路径传给了 `project_dir` | `project_dir` = `…/mdl-<stem>`，由 `project_init` 返回 |
| `project_dir_not_writable` | CSV 在只读/受保护目录 | 把 CSV 复制到可写目录（桌面/临时目录）；工具**不会**自动换位置 |
| `encoding_undecodable` | CSV 是 ANSI/GBK/二进制 | 另存 UTF-8 或 GB18030；Excel 另存为"CSV UTF-8" |
| 列名/中文月份显示成乱码 | 终端编码不是 UTF-8（这是**显示**问题，不是数据问题） | `chcp 65001` 或用 UTF-8 终端；数据本身按 `data.encoding` 判定 |
| `chinese_font` WARN + 图里方块 | 缺 CJK 字形 | 装字体；不影响数值 |

### 6.3 客户端连上了但 agent 不按流程走

| 现象 | 原因 | 处置 |
|---|---|---|
| agent 不认识 `risklab-mdl` skill | skill 没装到正确目录，或目录名/`name` 不一致 | 确认 `…/skills/risklab-mdl/SKILL.md` 存在，`name: risklab-mdl` |
| agent 拿到工具但乱调参数 | 没读 `references/tools.md` | 让 agent 先读 `references/tools.md`；本文件 §4 有标准流程 |
| 一调工具就"命令不存在" | 客户端进程 PATH 里没有 python | 用绝对路径（§3.1） |
| 工具列表为空 | 客户端读的是另一份配置文件 | 确认 `mcpServers` 键名与文件名；重启客户端 |
| 工具调用超时 | 首次 import lightgbm/xgboost 较慢；宽表分析本身要几十秒 | 放宽客户端超时；宽表先抽样或用 `config.features` 收窄 |

### 6.4 结果相关

| 现象 | 说明 | 处置 |
|---|---|---|
| train AUC 0.99+，val 0.65 | 宽表 + 少坏样本的过拟合，**不是程序错误** | 加强 `feature_refine`、降 `num_leaves`/`max_depth`、补坏样本 |
| LIFT = 0 | 尾部 10% 没抓到坏样本，真实值 | 看 `lift_seg_size` / `is_reliable` |
| PSI 没算 | 无时间列 / 只有一个月 / 基期覆盖全部月 | 看 `psi.reason`；**不能**声称跨期稳定 |
| `threshold_not_met` | KS/PSI 未达阈值 | 看 `ks_gap`/`psi_gap` 差距值，由用户决定下一步 |
| `feature_*` 报 `dependency_missing` | 缺 lightgbm（精筛/训练 lgb） | 装 lightgbm，或换 `gbdt`/`lr`/`dt` |

## 7. 卸载

```powershell
python -m pip uninstall risklab-mdl-v2
# skill 目录与 MCP 配置按装的位置手工删除
```

建模产物**不在包内**：它们落在 `CSV 同目录/mdl-<csv名>/`，需要手工清理。

## 8. 复现验收命令

```powershell
cd risklab-mdl-v2
python -m pytest tests/test_skill_contract.py -q      # 20/20 工具覆盖 + 无旧交互关键词 + 三份配置握手
python -m pytest tests/test_e2e_flow.py -q            # 模拟 agent 会话跑通 9 环节（约 60 s）
python -m pytest tests -q                             # 全量
```
