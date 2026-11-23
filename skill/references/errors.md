# 错误码与修复指引

> 错误码来源：`docs/tool-contract.md` §1.2（冻结）+ §6（模型层追加）+ `docs/data-feature.md` §5 +
> `docs/preprocess-split.md` §7。**本文只描述、不新增语义**；卡上遇到未登记的码，按 `error.message` /
> `error.hint` 处理并上报。
> 失败必须可见：`status="error"` 时 `error={code, message, hint}` 必须出现在回复里，
> **禁止**把失败当"没有结果"跳过，也禁止用上一次的结果顶替。

## 1. 错误载荷形态

```json
{"ok": false, "status": "error", "data": {...},
 "artifacts": [], "warnings": ["…"],
 "error": {"code": "project_dir_not_writable",
           "message": "CSV 所在目录不可写：D:/readonly",
           "hint": "把 CSV 复制到可写目录后重试（不会自动换位置）"}}
```

agent 侧固定三段式回复：

> ✗ 失败：`<code>` —— `<message 的人话版本>`
> 原因：…
> 下一步：`<按 hint 给一个具体动作，或给 2–3 个选项让用户选>`

## 2. 错误码表

### 2.1 路径与项目目录

| 错误码 | 触发场景 | 修复指引（agent 要做的） |
|---|---|---|
| `path_not_readable` | 文件不存在 / 不可读 / 是目录 | 让用户核对路径拼写；Windows 注意反斜杠、引号、中文路径；**不要自己猜别的路径** |
| `project_dir_not_found` | 传入的 `project_dir` 不存在 | 先 `project_init(csv_path)`；或确认 `project_dir` 拼的是 `mdl-<stem>` 而不是 CSV 路径 |
| `project_dir_not_writable` | CSV 所在目录不可写，建不了 `mdl-*` | 让用户把 CSV 复制到可写目录（桌面/临时目录）；工具**不会静默换位置** |
| `project_dir_conflict` | `mdl-*` 位置被同名**文件**占用 | 让用户删除或重命名那个文件 |
| `path_not_writable` | 目标文件无法写入（被占用/权限） | 关闭占用该文件的程序（Excel 打开 xlsx 是常见原因）后重试 |

### 2.2 manifest / 产物一致性

| 错误码 | 触发场景 | 修复指引 |
|---|---|---|
| `manifest_not_found` | 项目缺 `manifest.json` | 先 `project_init`（会补建 manifest） |
| `manifest_schema_invalid` | manifest 结构非法（含手工改坏、`run_id` 重复、`confirmed_run` 悬空） | 按 schema `risklab-mdl.manifest/1` 修；不确定就让用户重建项目（换 CSV 名或删掉 `mdl-*` 目录后 `project_init`） |
| `manifest_conflict` | 基于过期 manifest 快照写入（并发 lost update） | 重读后重试；同一项目**不要并发跑两个流程** |
| `lock_timeout` | 项目级文件锁超时（另一进程在写） | 等另一个进程结束；确认没有残留的 `.lock` 文件 |
| `json_decode_error` | 产物 JSON 损坏/被截断 | 重跑产生该产物的那一步（例如 `prescreen.json` 坏了就重跑 `feature_prescreen`） |
| `run_id_invalid` | `run_id` 不符合 `^\d{8}T\d{6}$` | 别手写 `run_id`，用 `run_list` 取 |
| `run_not_found` | `runs/<run_id>/` 不存在 | 用 `run_list` 查可用 run，再挑一个传给工具 |

### 2.3 数据与编码

| 错误码 | 触发场景 | 修复指引 |
|---|---|---|
| `encoding_undecodable` | 文件按探测到的编码解不开（二进制/加密/半截文件） | 让用户另存为 UTF-8（或 UTF-16 / GB18030）后重试 |
| `analysis_failed` | 数据可读但计算失败：训练异常、某子集只有一类标签导致指标无定义、run 产物损坏等**非预期失败** | 看 `error.message` 定位；**不要把失败当"空结果"继续**；修数据（补坏样本、删全空列）后重试 |
| `features_empty` | 筛选后无可用特征（初筛无数值特征 / 精筛候选池为空） | 放宽筛选阈值（说明放宽了什么）或 `preprocess_apply` 编码文本列，或 `config.features`/`config.candidates` 显式指定 |

### 2.4 入参与规格

| 错误码 | 触发场景 | 修复指引 |
|---|---|---|
| `config_invalid` | `config` / `thresholds` / `spec` 形态或取值非法：缺 `algorithm`、`params` 键名拼错、`tune_on` 不是 `val`、调参空间含未知参数、`config.features` 里有不存在的列、阈值越界、比例和 ≠ 1、`psi_base_months` 里有不存在的月份、初筛无 train/val 可切 | 按 `hint` 与 [workflow.md](workflow.md) 改入参；**改完把生效值回显给用户** |
| `spec_invalid` | `preprocess_apply` 的 `spec` 非法：未知键 / 未知列 / 未知策略 / 策略与列类型不符（如对文本列 `median`）/ 区间非法 / 空 spec / `encode.drop` 与 `cap`/`binning` 同列冲突 | 消息里带可用策略，按 `docs/preprocess-split.md` §3.1–3.2 修；不要靠猜键名（未知键会被拒绝） |
| `expression_invalid` | 样本划分字段表达式非法：函数调用 / 属性访问 / 赋值 / 分号 / `__` 标识符 / 括号未闭合 / 引用不存在的列 / 三个子集重叠 | 按 [workflow.md](workflow.md) §7.2 示例改写；非法**不是**可降级项 |
| `task_algorithm_mismatch` | 算法不支持该任务类型 | 按 `model_catalog` 换算法（**不降级**，不要偷偷换 `task`） |

### 2.5 环境与闸门

| 错误码 | 触发场景 | 修复指引 |
|---|---|---|
| `dependency_missing` | 运行期依赖缺失（如 `lightgbm` / `xgboost` 未装） | 按 `hint` 里的 pip 命令装，或换不需要该依赖的算法（`gbdt`/`lr`/`dt`） |
| `threshold_not_met` | `model_confirm` 的 KS/PSI 未达阈值 | **给差距值**（`ks_gap`/`psi_gap` + `failed_checks`）；让用户选：①加强筛选/降复杂度重训 ②换算法 ③**显式**调阈值（写清风险）。**不许**自动调小阈值重试 |
| `environment_not_ready` | `server_probe` 检测到依赖缺失或产物目录链路不可用（`doctor` 同语义） | 停流程；报缺哪个依赖 / `artifacts_check.error`；让用户跑 `risklab-mdl-doctor` 拿修复命令 |

## 3. 常见修复动作速查（按症状）

| 症状 | 先查 | 处置 |
|---|---|---|
| 「明明有文件却 `path_not_readable`」 | 路径是目录？含引号？相对路径按谁的 cwd？ | 用绝对路径；Windows 用 `D:/work/x.csv` 或 `D:\\work\\x.csv` |
| 「找不到项目」 | 是不是把 CSV 路径传给了 `project_dir` | `project_dir` 必须是 `…/mdl-<stem>` |
| 「数据读进来是乱码/列名坏了」 | `dataset_load.data.encoding` | 另存 UTF-8 / GB18030 后重跑 |
| 「特征全被筛掉了」 | `prescreen.json.summary.stage_dropped` | 放宽 `iv_min` / `missing_rate_max`；先编码文本列 |
| 「LR 训不了」 | `preprocess_apply.data.model_readiness` | `lr_ready=false` 时先填缺失 + 编码（目标：矩阵全数值无 NaN） |
| 「KS 很高但 PSI 也很高」 | `model_evaluate.data.psi.by_month` | 区分度与稳定性分开看；考虑按月分群或重训 |
| 「确认总是不通过」 | `data.ks_gap` / `data.psi_gap` 符号 | 负值=达标富余，正值=超出量；按差距决定动作 |
| 「报告数字和刚才对不上」 | 是否又跑了新 run | `run_list` 看最新 run；报告只读产物不重算 |

## 4. 环境自检：`doctor`（CLI，不是 MCP 工具）

`doctor` **未注册为 MCP 工具**，只有命令行形式；MCP 侧对应的探针是 `server_probe`。

```powershell
python -m risklab_mdl.doctor            # 中文摘要（退出码 0=可用 / 1=有 FAIL）
python -m risklab_mdl.doctor --json     # 结构化输出（部署脚本/工单用）
risklab-mdl-doctor --json               # 安装后等价的 console script
```

六项探针与失败含义：

| # | 探针 | 检查什么 | FAIL 意味着 |
|---|---|---|---|
| 1 | `python` | Python ≥ 3.11，且 `pyproject.toml` 版本与包内 `__version__` 一致（不一致只 WARN） | 解释器过低 |
| 2 | `dependencies` | 11 个运行时依赖逐个 import 并取版本 | 有依赖没装上（给 pip 命令） |
| 3 | `mcp_sdk` | MCP SDK 服务器类可导入（2.x `MCPServer` / 1.x `FastMCP`） | mcp 版本不匹配 |
| 4 | `matplotlib` | backend 为 `Agg` 且能真的导出 PNG（强制 Agg 后仍未生效 ⇒ FAIL） | 无头环境画不了图 |
| 5 | `chinese_font` | 中文字体 fallback 链 + 中文渲染无缺字形告警 | **只 WARN 不阻断**（图表中文可能变方块） |
| 6 | `artifact_dir` | 临时目录实地跑 `mdl-<stem>/` + manifest 往返 + run 目录命名 | 写盘链路坏了 |

退出码：**任一 FAIL ⇒ 1**（并打印可复制的修复命令）；只有 WARN ⇒ **0**（能跑但有降级）。

`doctor` 失败或 `server_probe` 返回 `environment_not_ready` 时，**不要继续建模流程**，
把探针名 + 明细 + `fix_commands` 原样给用户。

## 5. 重试纪律

1. **同一入参不要盲目重试**：`config_invalid`/`spec_invalid`/`expression_invalid` 重试结果必然相同，
   先改入参。
2. **最多追问 3 轮**：`needs_user_input` 的同一问题追问 3 次还没解决，请用户直接给列名/取值。
3. **写操作失败后先只读复核**：`preprocess_apply` / `dataset_split` 报错后，用 `dataset_columns`
   或重读产物确认状态，再决定重跑哪一步。
4. **降级不静默**：任何为了走通流程而做的取舍（跳过精筛、放宽阈值、抽样分析、内部划分兜底）
   都必须在回复里写出来，并说明后果。
