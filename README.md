# risklab-mdl-v2

风控建模 **计算层** MCP Server（跨平台，Python 3.11+）。三层架构中的 L2：

| 层 | 位置 | 职责 |
|---|---|---|
| L1 编排层 | [`skill/`](skill/SKILL.md) | 流程引导、提问话术、结果解读（给 agent 用） |
| **L2 计算层（本包）** | `src/risklab_mdl/` | 无 UI、弱状态，只做计算与产物落盘 |
| L3 交互层 | `mcp/` | Claude Code / DSH / WorkBuddy 等客户端的 MCP 配置样例 |

服务端不渲染弹窗、不维护流程状态机、不留会话状态；需要用户决策时返回结构化
`needs_user_input` 载荷，由 skill / agent 呈现。**20 个 MCP 工具**全部注册可用。

> **许可**：PolyForm Noncommercial License 1.0.0 —— 允许个人学习、研究、实验、
> 爱好项目等**非商业用途**，**禁止商业使用**。商业授权请开 Issue 联系作者。

## 特性

- **一条链路跑完建模**：读数据 → 画像 → 特征分析 → 初筛/精筛 → 预处理 → 划分 →
  训练 → 调参 → 评价 → 确认 → 报告，20 个 MCP 工具覆盖。
- **产物契约化**：目录由 CSV 路径唯一决定（`mdl-<CSV 文件名>/`），布局与字段冻结，
  可复现、可审计。
- **失败必须可见**：三态返回 `ok / needs_user_input / error`；任何降级进 `warnings`，
  错误必带 `code + message + 可执行 hint`。
- **跨平台**：无 `fcntl` 等 POSIX-only 依赖，原子写 + `filelock`，Windows/macOS/Linux 通用。
- **重依赖惰性加载**：lightgbm / xgboost / matplotlib 只在真正用到时导入。

## 快速开始

```powershell
git clone https://github.com/yuyong513/risklab-mdl-v2.git
cd risklab-mdl-v2

# 1) 安装（可编辑模式，含 pytest 开发依赖）
python -m pip install -e ".[dev]"

# 2) 环境自检（六项探针；退出码 0 = 可用）
python -m risklab_mdl.doctor

# 3) 快速 e2e：合成小样本跑完整链路（约 7 秒）
python -m pytest tests/test_e2e_sample.py -q

# 4) 启动 MCP stdio 服务（供客户端连接）
python -m risklab_mdl            # 等价于 risklab-mdl

# 5) 接客户端：把 mcp/mcp.<client>.json 并入客户端配置，把 skill/ 放进 skills 目录
#    详见 docs/skill使用指南.md
```

安装细节、镜像与排障见 [INSTALL.md](INSTALL.md)；真实 CSV 的完整流程：

```python
project_init(csv_path="D:/work/my_data.csv")      # → D:/work/mdl-my_data/
dataset_columns(project_dir="D:/work/mdl-my_data")
# …按 skill/SKILL.md 的 9 环节继续
```

> 示例宽表**不随仓库分发**。依赖外部样例的用例在样例缺失时会自动 skip，
> 不影响其余测试与工具使用。

## 工具清单（共 20 个）

`build_server().list_tools()` 的真实集合（[tests/test_server_smoke.py](tests/test_server_smoke.py) 严格断言）：

| 家族 | 工具 |
|---|---|
| 环境与项目 | `server_probe`、`project_init` |
| 数据 | `dataset_load`、`dataset_columns`、`dataset_profile` |
| 特征分析 | `feature_analyze`、`feature_charts` |
| 特征筛选 | `feature_prescreen`、`feature_refine` |
| 预处理与划分 | `preprocess_check`、`preprocess_apply`、`dataset_split` |
| 模型 | `model_catalog`、`model_train`、`model_tune`、`model_evaluate` |
| 过程 | `run_list`、`run_compare`、`model_confirm`、`report_build` |

> 第 20 个 MCP 工具是 `server_probe`。**`doctor` 不是 MCP 工具**，它只有 CLI 形式
> （`risklab-mdl-doctor` / `python -m risklab_mdl.doctor`）；两者实现同源，但入口不同。
> 每个工具的「何时调用 / 入参示例 / 结果解读」见 [skill/references/tools.md](skill/references/tools.md)。

## 产物布局

产物目录由 CSV 路径唯一决定，**不静默换位置**：

```
<CSV 同目录>/mdl-<CSV 文件名去扩展名>/
  manifest.json                        项目状态唯一事实来源（schema risklab-mdl.manifest/1）
  data_analysis/                       profile.json、describe.csv、target_distribution.csv、charts/*.png、index.html
  feature_analysis/                    iv_table.csv、woe_detail.csv、binning/*.json、charts/*.png、index.html
  feature_selection/                   prescreen.json、refine.json、selected_features.txt
  preprocess/                          check.json、summary.json、processed.parquet|processed.csv、fill/encode/outlier/manual_bins.json
  split/                               split.json（含 mask 指纹、seed，同 seed 可复现）
  runs/<YYYYmmddTHHMMSS>/              config.json、params.json、metrics.json、model.pkl、importance.csv、evaluation/、tuning/
  final/                               model.pkl（run 模型副本）、confirm.json（阈值 + 指标 + 确认时间）
  reports/                             结果汇总.xlsx（4 个非空 sheet）、过程报告.md、run_compare.csv
```

- 项目目录已存在 ⇒ **复用不覆盖**；历史 run 按时间戳目录区分。
- CSV 所在目录不可写 ⇒ `project_dir_not_writable`，并给修复指引（不静默改到临时目录）。
- 全部产物文件在 [docs/tool-contract.md](docs/tool-contract.md) §4 冻结。

## 验证

```powershell
python -m pytest tests -q                       # 全量：本机 536 passed（15 subtests），约 7–9 分钟
python -m pytest tests/test_e2e_sample.py -q    # 快速 e2e：合成数据全链路，约 7 秒
python scripts\check_layout.py "<project_dir>"  # 产物树契约校验（完整流程后）
```

> 依赖外部示例宽表的用例在样例缺失时会自动 skip，因此公开克隆下的通过数会略少；
> 其余用例与工具行为不受影响。

测试覆盖：core 纯计算、数据/特征/预处理/模型工具契约、三态协议、manifest 并发、
stdio 握手、skill 契约与静态红线（无 `fcntl`、无旧包名）。

## 架构与代码约定

```
src/risklab_mdl/
  server.py           MCP stdio 服务端（20 个工具注册）
  sdk.py              MCP SDK 兼容层
  protocol.py         三态响应构造（ok / needs_user_input / error）
  project.py          项目状态读取（manifest 角色、CSV 定位）
  doctor.py           环境自检（CLI）
  platform/fs.py      跨平台基础设施（原子写 / 文件锁 / 编码探测 / 路径校验）
  artifacts/          产物目录契约（layout / manifest / run 目录）
  core/               纯计算（分箱 / WOE / IV / 指标 / 划分 / 训练 / 调参 / 变换）
  core/utils.py       跨层共享小工具（数值归一 / 比率 / 箱标签）
  data/ features/ preprocess/   读数据、画像、特征分析、预处理
  models/             算法适配器、训练装配、评价
  runs/               run 目录与产物读写
  charts/             PNG + 单文件 HTML 图表
  tools/              MCP 工具薄封装（每模块一个家族）
```

命名与实现约定：

- 公开函数/变量 `snake_case`，类 `PascalCase`，常量 `UPPER_SNAKE`；
  模块私有 helper 一律 `_` 前缀，不跨模块引用私有名。
- 跨层复用的 helper 只放在 `core/utils.py`；三态响应只由 `protocol.py` 构造；
  项目状态读取只由 `project.py` 负责 —— 不在各工具里各写一份。
- 同一概念只允许一个名字：例如「确认阈值」与「初筛阈值」用不同函数名，
  避免同名不同义。
- 重依赖（lightgbm / xgboost / matplotlib）只在函数内惰性导入，统一入口
  `core/deps.py`；`import risklab_mdl.core` 不拉起重依赖。
- 错误信息用中文并带可执行 hint；JSON 字段统一 `snake_case`。
- 跨平台：禁止 `fcntl` 与 POSIX-only 调用，原子写 + `filelock`。

## 已知边界（摘要）

- 坏样本极少时（如 1.75%）KS/AUC/LIFT 波动大；train 与 val 差距大是宽表过拟合，
  不是程序错误 —— 用 `feature_refine`、降复杂度或补坏样本。
- 无时间列 / 只有一个月 ⇒ PSI 跳过并在 `warnings` 明示，不伪造数值。
- 未安装 `pyarrow` 时预处理产物落 `processed.csv`（功能与读数不变）。
- macOS / Linux、MCP SSE/HTTP 传输、Claude Code / WorkBuddy 客户端界面未实测。

完整清单与处置见 [docs/已知边界.md](docs/已知边界.md)。

## 文档索引

| 文档 | 内容 |
|---|---|
| [INSTALL.md](INSTALL.md) | 依赖、安装、doctor、客户端配置、常见报错 |
| [docs/tool-contract.md](docs/tool-contract.md) | 工具契约 + 三态协议 + manifest schema + 产物目录 |
| [docs/data-feature.md](docs/data-feature.md) | 数据与特征分析层：用法、产物字段、宽表性能策略 |
| [docs/preprocess-split.md](docs/preprocess-split.md) | 预处理 / 划分 / 筛选：spec 结构、表达式语法、阈值 |
| [docs/model-layer.md](docs/model-layer.md) | 模型层：能力表、run 结构、调参、评价、确认、报告 |
| [docs/口径说明.md](docs/口径说明.md) | KS/AUC/Gini/LIFT/PSI/IV 口径摘要 + 确认阈值 |
| [docs/口径-metrics.md](docs/口径-metrics.md) | 指标口径的完整定义与边界 |
| [docs/已知边界.md](docs/已知边界.md) | 样本、宽表、PSI、LIFT、客户端实测范围等边界 |
| [docs/skill使用指南.md](docs/skill使用指南.md) | skill 安装、MCP 配置、首次会话、排障 |
| [skill/SKILL.md](skill/SKILL.md) | 编排层：触发条件、9 环节流程、`needs_user_input` 协议 |
| [CHANGELOG.md](CHANGELOG.md) | 2.0.0 版本范围与关键决策 |

## 许可

本项目采用 **[PolyForm Noncommercial License 1.0.0](LICENSE)**：

- ✅ 允许：个人学习、研究、实验、爱好项目；教育、研究、公益等**非商业**场景；
- ✅ 允许：修改、分发（需保留许可与 `Required Notice`，且不得用于商业目的）；
- ❌ 禁止：任何商业用途（含把本软件或其衍生作品用于商业产品/服务）；
- 商业授权请联系作者（开 Issue 或通过 GitHub 主页联系）。

完整条款以 [LICENSE](LICENSE) 为准。
