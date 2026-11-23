# 9 环节交互话术与输入示例

> 配套 [SKILL.md](../SKILL.md) §5。本文给的是**可直接照抄的用户侧输入**与**agent 侧话术**，
> 每一段都对应一个真实工具入参（字段名以服务端为准，见 [tools.md](tools.md)）。
> 所有示例里的 `D:/work/...` 都可以换成用户自己的路径；`mdl-sample_2000` 是
> `project_init` 由 `D:/work/sample_2000.csv` 自动推出的项目目录。

统一的三个占位：

```text
CSV  = D:/work/sample_2000.csv
PROJ = D:/work/mdl-sample_2000            # = <CSV 同目录>/mdl-<CSV 文件名去扩展名>
TGT  = {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}}
```

---

## 环节 1 · 数据上传与项目引导

### 1.1 开场话术（拿到路径后的第一句）

> 我先给这份数据建一个建模项目目录，并读一下表结构。确认三件事就能开始：
> ①目标字段（标签列）是哪个？②二分类还是多分类？③哪个取值代表"坏"？

### 1.2 目标字段（前置确认，必须问）

工具侧需要的信息（`needs_user_input` 的 `data.question` / `data.context`）：

```json
{"question": {"id": "dataset_confirm_target", "default": "binary",
              "options": [{"value": "binary", "label": "二分类（如 0/1 好坏）"},
                          {"value": "multiclass", "label": "多分类（3 类及以上）"}]},
 "context": {"candidates": ["y", "f_0001", "f_0015", "f_0025", "f_0027"],
             "suggested": "y", "label_counts": {"0": 1965, "1": 35}, "rows": 2000,
             "example": {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}}}}
```

agent 话术示例：

> 数据 2000 行 × 696 列。我推断目标字段可能是 `y`（只有 0/1 两个取值：0 有 1965 个、1 有 35 个），
> 其它候选 `f_0001 / f_0015 / f_0025 / f_0027` 取值太少，更像分类特征。
> **确认两件事**：目标字段用 `y` 吗？任务类型是二分类吗？

用户确认后的**回填入参**（这是下一次调用的真实参数）：

```json
{"project_dir": "D:/work/mdl-sample_2000",
 "config": {"target": {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}}}}
```

### 1.3 标签映射（坏样本是哪个值）

| 用户数据形态 | 建议 labels | agent 必须确认的一句话 |
|---|---|---|
| `0/1` | `{"1": "bad", "0": "good"}` | 「1 是坏（违约）对吧？」 |
| `Y/N` | `{"Y": "bad", "N": "good"}` | 「Y = 坏？」 |
| `good/bad` | `{"bad": "bad", "good": "good"}` | 「bad = 坏？」 |
| `1/2`（1 正常、2 逾期） | `{"2": "bad", "1": "good"}` | 「2 才是坏，别按"数值大=好"理解？」 |
| 多分类 `A/B/C/D` 等级 | 不给 labels，只给 `task: "multiclass"` | 「默认排序后最后一类（如 D）算最坏，对吗？」 |

> 为什么不猜：`bad_class` 决定「取哪一列概率 / PSI 用哪个分数 / KS-LIFT 表里谁是坏」。
> 猜错时指标照样算得出来，但方向整体反了——这是最难发现的一类错误。

### 1.4 列角色（用到时才确认）

> 这三列我按推断处理：`sample_id` = id 列、`month` = 时间列、`split` = 划分列。
> 如果它们其实是特征（或 id 不是这个），现在告诉我；不确认我就按推断走（不会阻断流程）。

---

## 环节 2 · 样本数据分析

```json
{"project_dir": "PROJ", "config": {"target": TGT, "max_features_for_charts": 30}}
```

agent 话术示例：

> 坏率 1.75%（35/2000），属于坏样本偏少的场景——后面 KS 会有 ±0.03 左右的抖动，看结论时留意。
> 692 个特征里 58 个缺失率 ≥ 50%（最高的 `f_0009` 缺失 84.85%），这些列在特征筛选时大概率会被淘汰。
> 有时间列 `month`（2026 年 1–5 月），可以做按月坏率与后续的 PSI。
> 图表：`…/mdl-sample_2000/data_analysis/charts/target_distribution.png`、
> `missing_rate_top.png`、`monthly_bad_rate.png`；汇总页 `data_analysis/index.html`。

需要用户决策时只问一件：

> 列太多（692 个），画像图表要不要限制到前 30 个特征？（默认 30，不影响后续 IV 计算）

---

## 环节 3 · 特征分析

### 3.1 分箱方法与箱数

> 分箱默认等频 8 箱、缺失单独成箱。要换成等宽 / 决策树切点 / 你自己给切点吗？
> 箱数 5 / 8 / 10 也可以改（样本量撑不住时会自动下调并向你说明）。

```json
{"project_dir": "PROJ", "config": {"method": "quantile", "bins": 8, "target": TGT}}
```

### 3.2 自定义切点（手动分箱）

关键：切点**首尾可写 `-inf` / `inf`**，内部切点必须**有限、严格递增**，区间语义是
`(-inf, a]、(a, b]、…、(z, inf)`（`right=True, include_lowest=True`）。

```json
{"project_dir": "PROJ",
 "config": {"method": "custom", "edges": {"amount": ["-inf", 0, 100, 500, "inf"]}}}
```

一句话说明用户写法：

| 用户写法 | 落在哪个箱 |
|---|---|
| `amount <= 0` | `(-inf, 0]` |
| `0 < amount <= 100` | `(0, 100]` |
| `100 < amount <= 500` | `(100, 500]` |
| `amount > 500` | `(500, inf)` |
| 缺失 | 独立箱 `__MISSING__` |

### 3.3 结果解读话术

> IV TOP5：`f_0426` 0.675、`f_0491` 0.643、`f_0535` 0.571、`f_0379` 0.547、`f_0531` 0.482。
> **IV ≥ 0.5 通常意味着变量里混入了未来信息**（比如"是否已逾期"这种贷后字段），建议你确认这几列的
> 取数时点。`f_0423` 是非数值列（A/B/C/D 四类），需要先编码才能算 IV。

---

## 环节 4 · 模型选型

```json
{}
```

> 五个算法都能做二分类和多分类，本机都装好了：
> `lgb` 最快、风控主力；`xgb` 正则化完整、宽表稳；`gbdt` 无需额外依赖但单线程慢；
> `lr` 可解释（系数/WOE），宽表上容易不收敛；`dt` 只适合做规则基线。
> 在 691 列 + 35 个坏样本的数据上，我建议 `lgb`。默认参数是 `n_estimators=300, learning_rate=0.05,
> num_leaves=31, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0`。

默认参数与调参空间全表见 [models.md](models.md)。

---

## 环节 5 · 特征筛选

### 5.1 阈值（五道筛子）

> 初筛五道：缺失率 > 50% 淘汰、同值率 > 95% 淘汰、IV < 0.02 淘汰、月度 PSI > 0.10 淘汰、
> 相关性 |r| > 0.8 的保留 IV 更高的那个。要放宽哪一条？

```json
{"project_dir": "PROJ",
 "config": {"missing_rate_max": 0.5, "same_value_rate_max": 0.95,
            "iv_min": 0.02, "psi_max": 0.1, "corr_max": 0.8}}
```

低坏率/宽表常见放宽组合（**要让用户点头**）：

```json
{"project_dir": "PROJ", "config": {"iv_min": 0.01, "psi_max": 0.25, "corr_max": 0.9}}
```

### 5.2 结果话术

> 691 个候选 → 存活 449 个：缺失率淘汰 243 个、相关性淘汰若干、PSI 淘汰 0 个。
> 当前用的是**全量数据**（还没有 `split/split.json`），存在轻微数据穿越风险；
> 想彻底避免，就先做环节 7 的划分，再回来重跑初筛（那时只用 train）。

---

## 环节 6 · 入模前检查与特征精调

### 6.1 检查结果怎么讲

> 必须处理 693 条（不处理则入模矩阵含 NaN）：其中「缺失」类 N 条、「非数值未编码」M 条。
> 建议处理 664 条：主要来自缺失率 ≥ 50% 与极端值。我按下面这份规格处理，你看要不要改：

### 6.2 四段规格示例（缺失 / 编码 / 盖帽 / 手动分箱）

```json
{"project_dir": "PROJ",
 "spec": {
   "missing": {"amount": "median", "channel": "mode", "note": "as_bin"},
   "encode": {"channel": "onehot", "note": "ordinal"},
   "cap": {"score": {"strategy": "quantile", "lower_quantile": 0.01, "upper_quantile": 0.99},
           "balance": {"strategy": "absolute", "min": 0, "max": 100000}},
   "binning": {"amount": {"edges": ["-inf", 0, 100, 500, "inf"], "missing_as_bin": true}}
 }}
```

| 段 | 可选策略 | 关键约束 |
|---|---|---|
| `missing` | `mean` / `median` / `mode` / `sentinel` / `as_bin` / `drop` | `mean/median/sentinel` 仅数值列；`sentinel` 默认 `99999.0`，真实取值里已有该值时**报错**；`as_bin` 要配合 `binning`；`drop` 会删行 |
| `encode` | `onehot` / `ordinal` / `woe` / `drop` | `woe` 仅二分类，有 `split.json` 时只用 train 拟合；档数 > 20 会告警稀疏 |
| `cap` | `absolute`（`min`/`max`）/ `quantile`（`lower_quantile`/`upper_quantile`） | 至少给一个边界；`min < max`；分位 `0 <= lo < hi <= 1` |
| `binning` | `edges` + `missing_as_bin` | 内部切点有限、严格递增、至少一个；缺失落 `-1` |

盖帽的两个等价写法：

```json
{"cap": {"balance": {"strategy": "absolute", "min": 0, "max": 100000}}}
{"cap": {"score": {"strategy": "quantile", "lower_quantile": 0.005, "upper_quantile": 0.995}}}
```

### 6.3 结果话术

> 处理完成：`preprocess/processed.csv`（本机没装 pyarrow，落 CSV；装 pyarrow 会落 parquet），
> 2000 行进、2000 行出（没有删行）。填充 3 列、盖帽 1 列（`f_0003` 截断 7 个点，区间 1795–45000）、
> 手动分箱 1 列（`f_0100` → 3 箱 + 缺失箱）。
> `lr_ready=false`：入模矩阵里还有 1 个非数值列 `f_0423` 和若干缺失——**LR 不可用**，
> GBDT/XGB/LGB 可以继续。

### 6.4 精筛（可选）

> 要不要再按 gain 迭代把弱特征去掉？默认最多 10 轮、每轮保留线是最大 gain 的 5%、
> 目标特征数区间 `[min(5,n), min(60,n)]`，需要 lightgbm。

```json
{"project_dir": "PROJ",
 "config": {"scheme": "gain", "rounds": 10, "threshold": 0.05,
            "target_range": [5, 60], "max_features": 60, "seed": 42,
            "keep": ["f_0426"], "drop": ["f_0012"]}}
```

> 精筛结果：449 个候选 → 60 个（`disposition=truncated_high`，即迭代收敛到的特征数超过上限 60，
> 按累计 gain 截断）。完整清单：`feature_selection/selected_features.txt`。

---

## 环节 7 · 样本划分

### 7.1 三种打法

```json
{"project_dir": "PROJ",
 "config": {"mode": "stratified", "ratios": [0.7, 0.2, 0.1], "seed": 42,
            "stratify_by": "y", "time_column": "month"}}
```

```json
{"project_dir": "PROJ",
 "config": {"mode": "random", "ratios": [0.6, 0.2, 0.2], "seed": 7}}
```

```json
{"project_dir": "PROJ",
 "config": {"mode": "none"}}
```

### 7.2 按字段表达式划分（时间外推）

```json
{"project_dir": "PROJ",
 "config": {"mode": "field",
            "field": {"column": "month",
                      "expr_train": "month in (\"2026-01\",\"2026-02\")",
                      "expr_val":   "month == \"2026-03\"",
                      "expr_test":  "month not in (\"2026-01\",\"2026-02\",\"2026-03\")"}}}
```

> **字符串字面量必须与数据实际取值一致**：本样例数据的 `month` 取值是 `2026年1月` 这种写法
> （不是 `2026-01`），所以真实可跑的表达式要写成
> `month in ("2026年1月","2026年2月")`。写错时工具报 `expression_invalid` 并给位置——
> 先 `dataset_columns` 看一眼真实取值再写。

表达式能力与禁区：

| 支持 | 示例 |
|---|---|
| 比较 | `amount >= 1000`、`score != null` |
| 集合 | `channel in ('app','web')`、`month not in ('2026-01')` |
| 逻辑 / 括号 | `(channel == 'app' \| channel == 'web') & score != null` |
| 字面量 | `'str'`、`123`、`1.5`、`true/false/null`、反引号列名 `` `列 名` `` |

| 禁（⇒ `expression_invalid`） | 为什么 |
|---|---|
| `__import__('os').system(...)` | 函数调用 / 双下划线标识符 |
| `f_signal.amount > 1` | 属性访问 |
| `y = 1` | 单等号赋值（提示改 `==`） |
| `a > 1; b == 0` | 分号 |
| `a > 1 & (b > 0` | 括号未闭合 |

### 7.3 结果话术

> 分层划分（按 `y`，seed=42）：train 1200 / val 400 / test 400，三类坏率都是 1.75%（分层生效）。
> 掩码指纹 `eb8bc466b1398c6e`——后面每次训练都会核对这个指纹，确保用的是同一份划分。
> 产物：`split/split.json`。

---

## 环节 8 · 模型训练与调参

### 8.1 训练

```json
{"project_dir": "PROJ", "config": {"algorithm": "lgb", "task": "binary", "seed": 42}}
```

覆盖默认参数（键名必须是该算法 `default_params` 里的，拼错会 `config_invalid`）：

```json
{"project_dir": "PROJ",
 "config": {"algorithm": "lgb", "task": "binary",
            "params": {"num_leaves": 15, "learning_rate": 0.03, "n_estimators": 500,
                       "min_child_samples": 40, "reg_lambda": 5.0},
            "features": ["f_0559", "f_0011", "f_0258"], "seed": 42}}
```

### 8.2 调参：先预览，再执行

```json
{"project_dir": "PROJ", "config": {"algorithm": "lgb", "task": "binary", "trials": 20},
 "mode": "preview"}
```

预览返回的默认空间（`lgb` + `binary`）：

| 参数 | 类型 | 搜索范围 |
|---|---|---|
| `num_leaves` | int | [15, 127] |
| `learning_rate` | loguniform | [0.01, 0.3]（对数尺度） |
| `max_depth` | int | [3, 10] |
| `min_child_samples` | int | [5, 50] |
| `subsample` | uniform | [0.6, 1.0] |
| `colsample_bytree` | uniform | [0.6, 1.0] |
| `reg_lambda` | loguniform | [0.01, 10.0] |

目标函数：`ks*0.5 + auc*0.4 + lift*0.0 − gap*0.1`，`gap = max(0, train_auc − val_auc)`，
只有 `gap > 0.05` 才全额惩罚；`tune_on` 固定 `val`（**不接受 `oot`/`train`/`cv`**）。

自定义空间（未知参数名 ⇒ `config_invalid`）：

```json
{"project_dir": "PROJ",
 "config": {"algorithm": "lgb", "task": "binary", "trials": 30,
            "space": {"num_leaves": {"type": "int", "low": 8, "high": 64},
                      "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.2},
                      "reg_lambda": {"type": "loguniform", "low": 0.1, "high": 20.0}}},
 "mode": "execute"}
```

### 8.3 结果话术

> run `20261116T213542`：train KS 0.999 / AUC 0.9999，val KS 0.402 / AUC 0.658，test KS 0.406 / AUC 0.702。
> **train 与 val 差距 0.34 是明显过拟合**——691 列只对应 21 个训练期坏样本，这是数据规模的必然结果，
> 不是程序问题。要改善只能：加强特征筛选、降复杂度（`num_leaves` 15 / `max_depth` 4）、或补更多坏样本。

---

## 环节 9 · 模型结果确认

### 9.1 评估

```json
{"project_dir": "PROJ", "run_id": "20261116T213542",
 "config": {"psi_base_months": ["2026-01"]}}
```

```json
{"project_dir": "PROJ", "run_id": "20261116T213542"}
```

> 分数 PSI：基期 2026-01（合并池 590 行），最大 PSI 0.115（2026-04），overall 0.017。
> **闸门看最差月 0.115**，已经超过 0.10 的常规线——跨期稳定性偏弱。

### 9.2 确认（阈值的两种打法）

默认阈值（KS ≥ 0.25、PSI ≤ 0.10）：

```json
{"project_dir": "PROJ", "run_id": "20261116T213542"}
```

用户显式放宽（**必须写清风险**）：

```json
{"project_dir": "PROJ", "run_id": "20261116T213542",
 "thresholds": {"ks_min": 0.25, "psi_max": 0.20}}
```

自定义更严的阈值（用于研究）：

```json
{"project_dir": "PROJ", "run_id": "20261116T213542",
 "thresholds": {"ks_min": 0.30, "psi_max": 0.05}}
```

> 不通过时的话术模板：
> 「确认没通过：KS 0.402 ≥ 0.25 ✅，PSI 0.115 > 0.10 ❌（超出 0.015）。
> 你可以：①先做特征筛选/降复杂度重训 ②换 xgb 试试 ③把 PSI 阈值放宽到 0.15 并接受跨期漂移风险
> ——要哪一个？」（不要替用户选第三个。）

### 9.3 报告

```json
{"project_dir": "PROJ"}
```

> 已生成 `reports/结果汇总.xlsx`（概览 21 行 / 模型指标 4 行 / run 对比 2 行 / 特征重要性 21 行）
> 与 `reports/过程报告.md`；run 对比表 `reports/run_compare.csv`。

---

## 附 · 常见组合速查

| 目标 | 参数组合 |
|---|---|
| 快速跑通（最省时间） | `method=quantile,bins=5` + 跳过 `feature_refine` + `dataset_split(mode=random)` + `lgb` |
| 数据穿越最少的严谨打法 | 先 `dataset_split(stratified)` → 再 `feature_prescreen` → `feature_refine` → 重跑 `feature_analyze` |
| 时间外推验证 | `dataset_split(mode=field)` 按 `month` 表达式 + `dataset_split(config.time_column)` 供 PSI |
| 可解释模型 | `algorithm=lr` + 先 `preprocess_apply` 让 `lr_ready=true` + `feature_refine` 控特征数 |
| 多分类风险等级 | `target.task=multiclass`，指标看 `macro`，确认阈值仍用 KS ≥ 0.25 |
