---
name: risklab-mdl
description: 信用风险建模全流程编排（二分类 / 多分类）。当用户给出一个 CSV 数据文件路径并要求做信用风险建模、违约/好坏标签建模、坏账预测、特征分析（IV/WOE/分箱）、特征筛选、样本划分、模型训练与调参、KS/AUC/LIFT/PSI 评估或模型确认与报告时使用。本 skill 驱动 risklab-mdl MCP 服务端的 20 个计算工具，产物落在 CSV 同目录的 mdl-<csv名>/ 下。
---

# risklab-mdl —— 信用风险建模 skill（编排层）

本 skill 是三层架构里的 **L1 编排层**：它只负责**问什么、调哪个工具、怎么展示、失败怎么办**。
所有计算在 L2 的 MCP 服务端（`risklab-mdl`，20 个工具）里完成，呈现与追问在 L3 的 agent 侧。
服务端**不维护会话状态**：每次调用都必须把路径与本步参数显式传全。

---

## 1. 何时触发

满足任一即触发：

- 用户给出（或粘出）一个 **CSV 文件路径**，且意图是"建模 / 分析 / 评分卡 / 风控"；
- 用户要求做 **信用风险建模**、违约概率、好坏客户识别、坏账预测；
- 用户要求做 **特征分析**（分箱、WOE、IV、缺失分析）、**特征筛选**、**样本划分**、**模型训练/调参**、
  **KS/AUC/Gini/LIFT/PSI 评估**、**模型确认**或**建模报告**；
- 用户接着上一轮的项目要继续（例如"再训一个 xgboost 看看""这个 run 帮我确认"）。

**不触发**：单纯问 pandas/统计概念、只要求清洗数据、要求做回归/深度学习/线上打分服务
（这些超出本服务端范围，见 §7）。用户没给数据文件路径且无项目目录时，先问路径，不要凭空建项目。

---

## 2. 总原则（硬规则，违反即返工）

1. **目标字段前置确认**。`dataset_load` / `dataset_columns` 只给**推断**（`target_candidates` 可能有多个），
   目标列与任务类型必须由用户在正式分析前拍板。manifest 里 `project.target` 为 null 时，
   除 `project_init`/`dataset_load`/`dataset_columns` 外的分析工具会返回 `needs_user_input`。
2. **其余列角色在用到时确认**。id 列 / 时间列 / 划分列不强制前置，但一旦某个环节要用它
   （PSI 要时间列、按字段划分要划分列），当场提出并让用户确认，用户说"没有时间列"就按没有处理
   （PSI 跳过，`warnings` 明示，**不伪造数值**）。
3. **服务端只做计算，交互由 agent 按本 skill 呈现**。工具返回的 `needs_user_input` 只是一个数据载荷
   （`data.question` / `data.context`），没有任何界面；把它翻译成自然语言的提问。
4. **失败必须可见**。`status=error` 时 `error.code`/`message`/`hint` 必须出现在回复里，
   **禁止**把 `error` 当"没有结果"跳过、也禁止用上一次的结果顶替。
   同样地，`warnings` 里的降级（跳过 PSI、抽样、箱数下调、内部划分兜底、中位数填充）必须转述。
5. **不伪造数值**。不估算、不四舍五入到"看起来合理"、不用模型知识编造 KS/AUC/IV。
   所有数字只能来自工具返回的 `data` 或 `artifacts` 指向的文件。
6. **不自行改参数**。阈值、切点、划分比例、调参空间、确认阈值都由用户确认；
   工具报错后**不许**为了让流程走下去偷偷放宽阈值/换算法。改参数必须说明改了什么、为什么、风险是什么。
   唯一例外：用户明确说"你看着办/用默认值"，此时用本文档写明的默认值并回显给用户。

---

## 3. 工具与返回协议

### 3.1 工具集合（20 个，`build_server().list_tools()`）

| 家族 | 工具 |
|---|---|
| 环境与项目 | `server_probe`、`project_init` |
| 数据 | `dataset_load`、`dataset_columns`、`dataset_profile` |
| 特征分析 | `feature_analyze`、`feature_charts` |
| 特征筛选 | `feature_prescreen`、`feature_refine` |
| 预处理与划分 | `preprocess_check`、`preprocess_apply`、`dataset_split` |
| 模型 | `model_catalog`、`model_train`、`model_tune`、`model_evaluate` |
| 过程 | `run_list`、`run_compare`、`model_confirm`、`report_build` |

> **路径约定**：`project_dir` = `<CSV 同目录>/mdl-<CSV 文件名去扩展名>`，由 `project_init` 返回，
> **不是** CSV 路径。除 `project_init` / `dataset_load` 吃 `csv_path` 外，其余工具都吃 `project_dir`。
>
> **环境自检 CLI**：`doctor` 不是 MCP 工具，它只有命令行形式
> （`risklab-mdl-doctor` / `python -m risklab_mdl.doctor`），MCP 侧对应的探针是 `server_probe`。

每个工具的「何时调用 / 入参示例 / 结果解读」见 [references/tools.md](references/tools.md)（唯一真源）。

### 3.2 三态返回

```jsonc
{"ok": true,  "status": "ok",               "data": {}, "artifacts": [], "warnings": [], "error": null}
{"ok": false, "status": "needs_user_input", "data": {"question": {}, "context": {}}, "artifacts": [], "warnings": [], "error": null}
{"ok": false, "status": "error",            "data": {}, "artifacts": [], "warnings": [], "error": {"code": "", "message": "", "hint": ""}}
```

- `artifacts` 是**相对项目目录的 POSIX 路径**，可直接拼成绝对路径给用户打开；
- `warnings` 是非阻断降级提示，**要转述**；
- `status` 三态互斥，`ok == (status == "ok")`。

---

## 4. 交互协议：`needs_user_input` 怎么处理

工具返回 `needs_user_input` 时，`data` 结构固定：

```jsonc
{
  "question": {
    "id": "dataset_confirm_target",
    "text": "…请确认目标字段与任务类型（config.target={column, task, labels?}）",
    "options": [{"value": "binary", "label": "二分类（如 0/1 好坏）"},
                {"value": "multiclass", "label": "多分类（3 类及以上）"}],
    "default": "binary",
    "fields": [{"name": "target_column", "type": "string", "required": true},
               {"name": "task", "type": "string", "required": false},
               {"name": "labels", "type": "object", "required": false}]
  },
  "context": {
    "columns": ["…前 50 列…"], "column_count": 696, "rows": 2000,
    "candidates": ["y", "f_0001", "…"],
    "suggested": "y",
    "label_counts": {"0": 1965, "1": 35},
    "example": {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}}
  }
}
```

处理规则：

1. **呈现**：把 `question.text` 转成一句人话问题，列出 `options` 的 label（标出 `default`），
   把 `context.candidates` + `context.label_counts` 摆出来让用户选；`context.suggested` 作为推荐项
   并说明依据（"取值 2 个、列名像标签"）。
2. **等待**：一次只问一件事，问完停住等回答，**不要**猜一个答案继续跑。
3. **回填**：用户回答后，把答案作为**下一次调用**的显式参数传入，形态是
   `config.target = {"column": "<列名>", "task": "binary|multiclass", "labels": {"1": "bad", "0": "good"}}`。
   没有服务端 session，所以答案只存在于这次调用的入参里（确认后会写进 `manifest.project.target`）。
4. **重试**：回填后重跑同一个工具。若仍返回 `needs_user_input`（例如列名拼错导致找不到目标列），
   把 `question.text` 里新增的理由讲清楚再问一次，最多追问 3 轮，之后请用户直接给列名。

标签映射的判定（**要跟用户确认，不要默认**）：

- 二分类默认 `{"1": "bad", "0": "good"}`，即 **1 = 坏 = 违约**；
- 用户数据里"坏"是 `Y/N`、`bad/good`、`1/2`、`A/B` 之类时，让用户明确说"哪个值代表坏"，
  然后写进 `labels`（`bad/1/坏/违约/正例/positive` 这些 token 会被识别为坏，见服务端 `bad_class` 规则）；
- 多分类（风险等级 `0/1/2/3`）要确认"哪一类算最坏"（默认排序后最后一类），
  以及是否要按类做 OVR。

---

## 5. 9 环节流程

九个环节是**主干顺序**，不是死板流水线：用户跳步（"我已经有特征清单，直接训练"）就跳到对应环节，
但**跳过前必须告知后果**（例：不划分就训练 ⇒ 指标不代表样本外表现；不筛选就训练 ⇒ 宽表过拟合）。

| # | 环节 | 主用工具 | 关键用户输入 |
|---|---|---|---|
| 1 | 数据上传与项目引导 | `server_probe` → `project_init` → `dataset_load` → `dataset_columns` | **目标字段 + 任务类型**（前置确认）、列角色 |
| 2 | 样本数据分析 | `dataset_profile` | 是否抽样、图表数量上限 |
| 3 | 特征分析 | `feature_analyze` → `feature_charts` | 分箱方法 / 箱数 / （自定义切点） |
| 4 | 模型选型 | `model_catalog` | 算法（dt/lr/gbdt/xgb/lgb） |
| 5 | 特征筛选 | `feature_prescreen` | 五道阈值、（可选）`feature_refine` 的迭代参数 |
| 6 | 入模前检查与特征精调 | `preprocess_check` → `preprocess_apply` → `feature_refine` | 缺失策略 / 编码方式 / 盖帽 / 手动分箱切点 |
| 7 | 样本划分 | `dataset_split` | 划分方式 / 比例 / seed / 表达式 |
| 8 | 模型训练（可选调参） | `model_train` → `model_tune` | 算法参数、是否调参、调参空间 |
| 9 | 模型结果确认 | `model_evaluate` → `run_list` → `run_compare` → `model_confirm` → `report_build` | 确认阈值、选哪个 run |

> **顺序纪律（重要）**：`feature_prescreen` 的正确取数是 **train**，`feature_refine` 是 **train + val**。
> 本文主干把"特征筛选(5)"排在"样本划分(7)"之前，此时工具会**使用全量数据**并给出告警
> ——「未找到可用的 split/split.json ⇒ 初筛使用全量数据」。这属于已知口径风险：
> 若用户在意数据穿越，就先做环节 7 的划分再回到环节 5/6（工具会自动只用 train/train+val）。
> 无论哪种顺序，**必须在回复里说明当前用的是哪个口径**。

---

### 环节 1 · 数据上传与项目引导

**问什么**

| 问题 | 选项 | 默认 |
|---|---|---|
| CSV 路径 | 用户给绝对路径（含中文目录也可以；相对路径按 agent 工作目录解析） | — |
| 目标字段是哪个列 | 从 `dataset_columns` 的候选里选（推荐 `context.suggested`） | 无默认，**必须问** |
| 任务类型 | 二分类 / 多分类 | 二分类 |
| 标签映射 | 哪个值代表"坏" | `1=bad, 0=good`（需用户确认） |
| id 列 / 时间列 / 划分列 | 有就指定，没有就明说"没有" | 用推断，不阻断 |

**调哪个工具**（顺序固定）

```jsonc
// 1) 环境探针（会话第一条，失败即停）
{}
// 2) 建/复用项目目录
{"csv_path": "D:/work/sample_2000.csv"}
// 3) 读表 + 角色推断（不阻断）
{"csv_path": "D:/work/sample_2000.csv"}
// 4) 列清单与角色（用于和用户确认目标列）
{"project_dir": "D:/work/mdl-sample_2000"}
```

**如何展示**：项目目录绝对路径 + 「新建 / 复用」；一张小表 `行数 × 列数 / 编码 / 检测到的时间列`；
候选目标列（≤5 个）与各自的取值分布；然后**只问目标字段与任务类型这一件事**。
列很多时不要整表刷屏（`dataset_columns` 返回 691 个特征列是常态），只列角色非 feature 的列 +
缺失率 TOP10 + 高基数列。

**失败与重试**

- `path_not_readable` → 让用户核对路径（Windows 上注意反斜杠与引号），不要自己猜别的路径；
- `project_dir_not_writable` → CSV 所在目录不可写（常见：`C:\Program Files`、只读共享盘）。
  **不静默换位置**，让用户把 CSV 复制到可写目录再给路径；
- `project_dir_conflict` → `mdl-*` 被同名文件占用，让用户删/改名；
- `encoding_undecodable` → 让用户另存为 UTF-8 / UTF-16 / GB18030；
- `server_probe` 返回 `environment_not_ready` → 停流程，报依赖缺失与 `doctor` 修复命令。

---

### 环节 2 · 样本数据分析

**问什么**

| 问题 | 选项 | 默认 |
|---|---|---|
| 是否抽样看画像（大表） | 是（给行数）/ 否 | 不抽样 |
| 图表数量上限 | 30 / 15 / 其它 | 30 |

**调哪个工具**

```jsonc
{"project_dir": "D:/work/mdl-sample_2000",
 "config": {"target": {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}},
            "max_features_for_charts": 30}}
```

**如何展示**（表格字段 + 图表路径 + 指数含义）

- 目标分布表：`label / label_name / count / share`，并给一句 **坏率** `bad_rate`（= 坏样本/有效样本）；
- 缺失概览：缺失率 TOP（列名 / 类型 / 缺失率 / 同值率），>50% 的列单独点出来；
- 特征类型构成：`numeric / categorical / text / datetime / high_cardinality` 各多少列；
- 图表（`artifacts` 里的相对路径，拼成项目绝对路径给用户）：
  `data_analysis/charts/target_distribution.png`、`missing_rate_top.png`、
  `feature_distributions.png`、`monthly_bad_rate.png`（有时间列才有），
  以及单文件汇总 `data_analysis/index.html`；
- **结论话术**：坏率 < 5% 要说"坏样本稀疏，后续 IV/KS 读数会抖"；缺失 > 50% 的列要说
  "这些列后面大概率会被初筛淘汰"；无时间列要说"PSI 将无法计算，跨期稳定性无法评估"。

**失败与重试**：`needs_user_input` ⇒ 目标字段还没确认，走 §4 的流程；
`analysis_failed` ⇒ 看 `error.message`（全空列、标签只有一类等），把原因转述；不要重试同一条命令，
先修数据或换目标列。

---

### 环节 3 · 特征分析

**问什么**

| 问题 | 选项 | 默认 |
|---|---|---|
| 分箱方法 | 等频 `quantile` / 等宽 `equal_width` / 决策树 `dt` / 自定义 `custom` | `quantile` |
| 箱数 | 5 / **8** / 10 / 其它 | 8 |
| 自定义切点（仅 custom） | 逐特征给切点，如 `[-inf, 0, 100, 500, inf]` | — |
| 是否只看部分特征 | 全部 / 指定清单 | 全部（除 id/time/split/target） |

**调哪个工具**

```jsonc
{"project_dir": "D:/work/mdl-sample_2000",
 "config": {"method": "quantile", "bins": 8}}
// 自定义切点
{"project_dir": "D:/work/mdl-sample_2000",
 "config": {"method": "custom", "edges": {"f_0100": ["-inf", 0, 100, 500, "inf"]}}}
// 补图（IV 分布 / TOP 箱图 / 相关性热力图），不重算 IV
{"project_dir": "D:/work/mdl-sample_2000", "config": {"max_features": 15}}
```

**如何展示**

- IV 表（`feature_analysis/iv_table.csv`）字段：`feature / iv / n_bins / n_bins_fitted / missing_rate /
  same_value_rate / method / task / skipped / reason`；回复里给 IV TOP 10 的表；
- IV 分级口径（见 [references/metrics.md](references/metrics.md) §5）：
  `<0.02` 几乎无区分、`0.02–0.1` 弱、`0.1–0.3` 中、`0.3–0.5` 强、`≥0.5` **疑似含未来信息**（要提醒排查）；
- 跳过的特征（`data.skipped`）：逐个给原因（非数值需编码 / 常量 / 全缺失）；
- 图表：`feature_analysis/charts/iv_distribution.png`（含 0.02/0.1/0.3 阈值线）、
  `woe_<feature>.png`（IV TOP10）、`correlation_heatmap.png`、汇总 `feature_analysis/index.html`。

**失败与重试**：`config_invalid`（`method=custom` 缺 `edges`、`features` 里有不存在的列）⇒ 按 `hint` 改入参；
宽表 `warnings` 会说明"只对 IV 最高的 10 个特征出箱图"——这是性能策略，不是失败；
"箱数因样本量不足被下调"必须转述（此时 IV 与历史口径不可比）。

---

### 环节 4 · 模型选型

**问什么**

| 问题 | 选项 | 默认 |
|---|---|---|
| 用哪个算法 | `lgb`（推荐，最快）/ `xgb` / `gbdt` / `lr` / `dt` | `lgb` |
| 是否需要可解释性 | 需要（lr/dt）/ 不强求（GBDT 系） | 不强求 |
| 是否要调参 | 先不调 / 调（后续环节 8） | 先不调 |

**调哪个工具**

```jsonc
{}
```

**如何展示**：5 行能力表——算法 / 可做任务 / 依赖是否装好（`data.available.<algo>.installed`）/
默认关键参数 / 一句话场景（来自 `data.catalog.algorithms.<algo>.notes`）。
`installed=false` 的算法不要推荐，直接把 `install` 里的 pip 命令给用户。
详细场景与默认参数见 [references/models.md](references/models.md)。

**失败与重试**：本工具无入参、只读，正常不会失败；若返回错误，按 `error.code` 报给用户并停下。

---

### 环节 5 · 特征筛选

**问什么**

| 问题 | 选项 | 默认 |
|---|---|---|
| 缺失率上限 | 0.3 / **0.5** / 0.7 | 0.5 |
| 同值率上限 | **0.95** / 0.99 | 0.95 |
| IV 下限 | 0.01 / **0.02** / 0.05 | 0.02 |
| PSI 上限 | **0.10** / 0.25 | 0.10（无时间列时自动跳过） |
| 相关性上限 | **0.8** / 0.9 | 0.8 |

**调哪个工具**

```jsonc
{"project_dir": "D:/work/mdl-sample_2000",
 "config": {"missing_rate_max": 0.5, "same_value_rate_max": 0.95,
            "iv_min": 0.02, "psi_max": 0.1, "corr_max": 0.8}}
```

**如何展示**：`n_input → n_survived` 漏斗（逐阶段淘汰数）、淘汰 TOP 原因表
（`feature / stage / reason`）、存活清单（超过 30 个只给数量 + 前 10 + 指向
`feature_selection/prescreen.json`）、`data.psi`（基期是哪个月）。
必答内容：**当前用的是全量还是 train**（`data.data_scope.source`）。

**失败与重试**：`features_empty` ⇒ 放宽容阈值（说明放宽了什么）或先 `preprocess_apply` 编码文本列；
`config_invalid`（阈值不在 `(0,1)` 内）⇒ 改入参重试；
`warnings` 里的"未找到 split/split.json ⇒ 使用全量"必须转述（存在穿越风险）。

---

### 环节 6 · 入模前检查与特征精调

**问什么**

| 问题 | 选项 | 默认 |
|---|---|---|
| 缺失列怎么处理 | 中位数/均值/众数填充、哨兵值、独立成箱 `as_bin`、删行 `drop` | 数值列中位数，缺失有含义时 `as_bin` |
| 文本/类别列怎么编码 | onehot / ordinal / woe（仅二分类）/ drop | onehot（档数 ≤ 20），高基数用 ordinal/woe |
| 是否盖帽 | 分位数（1%/99%）/ 绝对值区间 / 不盖 | 对偏态列做分位盖帽 |
| 是否手动分箱 | 给切点（如 `[-inf, 0, 100, 500, inf]`）/ 不分箱 | 不分箱 |
| 精筛参数（可选） | 轮数 10 / 保留线 0.05 / 目标特征数区间 | rounds=10, threshold=0.05 |

**调哪个工具**

```jsonc
// 6.1 检查
{"project_dir": "D:/work/mdl-sample_2000"}
// 6.2 落地规格
{"project_dir": "D:/work/mdl-sample_2000",
 "spec": {"missing": {"amount": "median", "channel": "mode"},
          "encode": {"channel": "onehot"},
          "cap": {"score": {"strategy": "quantile", "lower_quantile": 0.01, "upper_quantile": 0.99}},
          "binning": {"amount": {"edges": ["-inf", 0, 100, 500, "inf"], "missing_as_bin": true}}}}
// 6.3 精筛（可选）
{"project_dir": "D:/work/mdl-sample_2000",
 "config": {"scheme": "gain", "rounds": 10, "threshold": 0.05}}
```

**如何展示**

- 6.1：`must_fix_count / suggested_count` 两个数 + 按问题类型聚合（缺失 / 非数值 / 常量 / 极端值 /
  高基数 / 哨兵冲突）各几条 + 每类举 2–3 例，最后给一份**推荐 spec 草案**让用户改；
- 6.2：`processed_path`（parquet 还是 csv 降级）、`rows_in → rows_out`（不同说明删过行）、
  逐列动作摘要（填充值/新增列/截断区间/箱计数），以及 `model_readiness.lr_ready`
  ——`false` 要明说"LR 不可用，GBDT/XGB/LGB 可以"，并列出 `non_numeric_columns`；
- 6.3：`n_candidates → n_selected`、最终清单前 20、`disposition`（非 `converged` 必须解释）、
  `gain_shape.advice`。

**失败与重试**：`spec_invalid` ⇒ 按 `error.message` 里指出的键/列/策略修 `spec`（未知键会被拒绝，
不要靠猜）；`dependency_missing`（精筛缺 lightgbm）⇒ 给 pip 命令，或跳过精筛直接用初筛结果
（并说明"未做精筛"）；`config_invalid`（精筛 val 为空）⇒ 先 `dataset_split` 或改用 `keep/drop` 显式清单。

---

### 环节 7 · 样本划分

**问什么**

| 问题 | 选项 | 默认 |
|---|---|---|
| 划分方式 | 不划分 `none` / 随机 `random` / 分层 `stratified` / 按字段 `field` | 工具侧默认 `none`；**本 skill 建议 7:2:1 分层**（要跟用户明说"全量口径指标不代表样本外表现"） |
| 比例 | 7:2:1 / 6:2:2 / 8:1:1 | 7:2:1 |
| seed | 任意整数（同 seed 可逐行复现） | 42 |
| 分层依据 | 目标列 / 其它类别列 | 目标列 |
| 按字段划分的表达式 | 如 `month in ("2026-01","2026-02")` | — |
| 时间列（可选，给 PSI 用） | 列名 | 用推断/用户指定 |

**调哪个工具**

```jsonc
{"project_dir": "D:/work/mdl-sample_2000",
 "config": {"mode": "stratified", "ratios": [0.7, 0.2, 0.1], "seed": 42,
            "stratify_by": "y", "time_column": "month"}}
```

**如何展示**：`train/val/test` 行数与占比、`label_distribution`（各类占比是否与整体一致）、
`fingerprint`（后续训练复现要核对的掩码指纹）、`mode=none` 时明说"全量口径，指标不代表样本外表现"。

**失败与重试**：`expression_invalid` ⇒ 把 `error.message` 里的位置与原表达式转述，
按 `hint` 的示例改写（禁函数调用/属性访问/赋值/分号）；`config_invalid` ⇒ 比例和必须为 1、分层列必须存在。

---

### 环节 8 · 模型训练（可选调参）

**问什么**

| 问题 | 选项 | 默认 |
|---|---|---|
| 算法与任务 | 见环节 4 | `lgb` + `binary` |
| 用哪些特征 | `selected_features.txt`（精筛/初筛结果）/ 显式清单 / 全部数值列 | 现成清单优先 |
| 是否覆盖默认参数 | 不覆盖 / 给键值（键名必须是该算法默认参数里的） | 不覆盖 |
| 是否调参 | 不调 / preview 看空间 / execute 执行 | 不调 |
| 调参试验数 | **20** / 50 / 100 | 20 |
| 调参空间 | 用默认空间 / 自定义（参数名/类型/范围） | 默认空间 |

**调哪个工具**

```jsonc
// 8.1 训练
{"project_dir": "D:/work/mdl-sample_2000",
 "config": {"algorithm": "lgb", "task": "binary", "seed": 42}}
// 8.2 调参：先预览
{"project_dir": "D:/work/mdl-sample_2000",
 "config": {"algorithm": "lgb", "task": "binary", "trials": 20}, "mode": "preview"}
// 8.3 确认空间后再执行
{"project_dir": "D:/work/mdl-sample_2000",
 "config": {"algorithm": "lgb", "task": "binary", "trials": 20}, "mode": "execute"}
```

**如何展示**：`run_id`、`n_features`、`feature_source`、`data_source`、`split.source`、
train/val/test 三行指标表（KS / AUC / Gini / LIFT / 样本数 / 坏样本数），
并给**一句判读**：train 与 val 的 AUC 差距 > 0.15 就要说"明显过拟合，建议降复杂度或加强筛选"
（宽表 + 少坏样本必然如此，见 §7）；`best_iteration` / `fit_seconds` 顺带给出。
调参 preview：`space_table` 整表 + `expected_trials` + `objective` 公式，等用户点头再 execute。

**失败与重试**：`task_algorithm_mismatch` ⇒ 按 `model_catalog` 换算法，**不要降级绕过**；
`config_invalid`（`params` 键名拼错、`features` 里点名了非数值列）⇒ 按 `hint` 改；
`dependency_missing` ⇒ 给 pip 命令或换算法；`needs_user_input` ⇒ 目标字段未确认。

---

### 环节 9 · 模型结果确认

**问什么**

| 问题 | 选项 | 默认 |
|---|---|---|
| 评估哪个 run | 从 `run_list` 里选（或用最新 run） | 最新且指标达标的 run |
| PSI 基期 | 最早月 / 指定月 / 多月合并池 | 最早月 |
| 确认阈值 | **KS ≥ 0.25、PSI ≤ 0.10** / 用户自定义 | 0.25 / 0.10 |
| 是否出报告 | 是 / 否 | 是 |

**调哪个工具**

```jsonc
// 9.1 指标与跨期稳定性
{"project_dir": "D:/work/mdl-sample_2000", "run_id": "20261116T213542"}
// 9.2 全部 run
{"project_dir": "D:/work/mdl-sample_2000"}
// 9.3 对比（缺省全部）
{"project_dir": "D:/work/mdl-sample_2000"}
// 9.4 确认闸门
{"project_dir": "D:/work/mdl-sample_2000", "run_id": "20261116T213542",
 "thresholds": {"ks_min": 0.25, "psi_max": 0.10}}
// 9.5 报告
{"project_dir": "D:/work/mdl-sample_2000"}
```

**如何展示**

- 指标表：逐子集 `KS / AUC / Gini / LIFT@10% / n / bad / status`；多分类另给 `per_class` 与 `macro`；
  子集 `status="skipped"` 要说明原因（该子集只有一类标签），**不能当成 0 分**；
- PSI：`max_psi` + `max_month`（闸门用最差月）+ `by_month` 明细 + 基期是哪几个月；`psi.enabled=false`
  就说"无时间列/只有一个月，PSI 未计算，不能声称跨期稳定"；
- run 对比表：`run_id / 算法 / val KS / val AUC / PSI / 状态 / best`；
- 确认结论：通过 → 给 `final/model.pkl`、`final/confirm.json`、`manifest.confirmed_run` 与
  `checks[]` 的每条人话；不通过 → 给 `failed_checks` + `ks_gap`/`psi_gap` 的差距值，
  并让用户选：①回去降复杂度重训 ②换算法 ③**显式**调低阈值（写清风险）；
- 报告：`reports/结果汇总.xlsx`（四 sheet）与 `reports/过程报告.md` 的路径。

**失败与重试**：`threshold_not_met` ⇒ 差多少说多少，**不许**自动把阈值改小再调一次；
`run_not_found` ⇒ `run_list` 查真实 `run_id`（别手写 `run_id`）；
`config_invalid`（`psi_base_months` 里有不存在的月份）⇒ 换真实月份，工具**不会**静默回落到最早月。

---

## 6. 展示规范速查

| 内容 | 形态 | 注意 |
|---|---|---|
| 指标 | Markdown 表：逐子集一行，列为 KS/AUC/Gini/LIFT/n/bad | 只写工具返回的数字，保留 4 位小数 |
| 特征清单 | 前 10–20 行 + 数量 + 产物路径 | 691 列时禁止整表刷 |
| 图表 | 给相对路径拼出的绝对路径，说明这张图看什么 | 宽表默认只出 TOP10 箱图 |
| 产物 | 按 `artifacts` 分组列出（data_analysis / feature_analysis / …） | 路径一律 POSIX 形式 |
| 警告 | 单独一段「⚠ 需要注意」，逐条转述 | 不许吞掉 |
| 失败 | 单独一段「✗ 失败」：`code` + 人话原因 + 修复动作 | 不许当成"没结果" |

各目录与每个文件是什么意思，见 [references/artifacts.md](references/artifacts.md)。

---

## 7. 边界与降级（不隐藏）

1. **不做范围外的事**：不写新算法、不做回归/深度学习、不做线上打分服务、不做外部验证（OOT）；
   `tune_on` 结构性只允许 `val`。需要这些就让用户换工具，不要硬凑。
2. **宽表性能**：`sample_2000` 是 2000 行 × 696 列；全链路（画像 → 特征分析 → 初筛 → 精筛 →
   preprocess → 划分 → 训练 → 评价 → 确认 → 报告）本机实测约 **58 s**，其中 `feature_analyze` ≈ 18 s、
   `feature_prescreen` ≈ 26 s 占大头。列数再翻几倍时：先 `sample_rows` 抽样看画像、
   用 `config.features` 收窄分析范围、`max_features_for_charts` 限制出图。
3. **样例数据规模提示**：`sample_2000` 只有 35 个坏样本（坏率 1.75%），属于"勉强能算"的规模。
   在这种数据上：train AUC 可能 0.99+ 而 val AUC 仅 0.6x（过拟合是真实结论，不是 bug）；
   LIFT 可能为 0（尾部 10% 一个坏都没抓到）；IV ≥ 0.5 的特征要怀疑未来信息。
   把这些如实讲给用户，**不要**为了让指标好看而改数据或改口径。
4. **不伪造数值**：无时间列 ⇒ PSI 不下结论；子集只有一类 ⇒ `skipped` 不补 0；
   run 没有 KS ⇒ 对比时 `best=null` 不默认挑第一个。
5. **不自行改参数**：任何阈值/切点/比例/空间的变更都要用户确认后显式传入，并在回复里回显生效值。
6. **重新执行是有代价的**：`feature_analyze` 在划分之后重跑才无数据穿越；
   `preprocess_apply` 用 `missing.drop` 删过行会让已有 `split.json` 的 mask 错位
   （工具会告警并退回内部划分），此时应先划分或重跑划分。

---

## 8. references 索引

| 文件 | 内容 |
|---|---|
| [references/tools.md](references/tools.md) | 20 个工具的「何时调用 / 入参示例 / 结果解读」（唯一真源，覆盖测试来源） |
| [references/workflow.md](references/workflow.md) | 9 环节逐项话术与输入示例（目标字段、标签映射、阈值、切点、表达式、调参空间、确认阈值） |
| [references/models.md](references/models.md) | dt/lr/gbdt/xgb/lgb 场景、默认参数、默认调参空间、二分类/多分类注意点、选型建议 |
| [references/metrics.md](references/metrics.md) | KS/AUC/Gini/LIFT/PSI 口径摘要、多分类 OVR+macro、确认阈值与差距读法 |
| [references/errors.md](references/errors.md) | 错误码表（与工具契约一致）+ 修复指引 + `doctor` CLI 用法 |
| [references/artifacts.md](references/artifacts.md) | `mdl-<csv名>/` 目录结构与每个产物的解读 |
