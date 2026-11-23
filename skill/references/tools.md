# 工具速查（20 个 MCP 工具）

> 唯一真源：`src/risklab_mdl/server.py::register_tools`。本文的工具集合与
> `build_server().list_tools()` 完全一致（**20 个**），`tests/test_skill_contract.py` 逐条校验。
> 每个工具**有且仅有**一段「何时调用 / 入参示例 / 结果解读」，段首用
> `<!-- tool: <工具名> -->` 标记，供覆盖率测试定位。
>
> 通用前提：除 `project_init` / `dataset_load` 吃 `csv_path` 外，其余工具都吃
> `project_dir` = `<CSV 同目录>/mdl-<CSV 文件名去扩展名>`（如 `D:/work/mdl-sample_2000`），
> **不是 CSV 路径**。所有工具返回三态信封 `{ok, status, data, artifacts, warnings, error}`。
>
> 环境探针在 MCP 侧是 `server_probe`；契约文档里的 `doctor` **未注册为 MCP 工具**，
> 它只以 CLI 形式提供（`risklab-mdl-doctor` / `python -m risklab_mdl.doctor`），
> 用法见 [errors.md](errors.md) §4。这是契约文本与注册现状的差异，本文以注册现状为准。

---

## 环境与项目（2）

<!-- tool: server_probe -->
### server_probe —— 环境探针（会话第一条必调）

**何时调用**：每次会话开始时调用一次（无入参）。握手成功不代表能真跑建模，本工具会
在临时目录实地验证「`mdl-<stem>` 目录 + manifest 往返 + run 目录命名」这条链，
并逐个 import 11 个运行时依赖。

**入参示例**：

```json
{}
```

**结果解读**：`data.version`（包版本）、`data.platform`（python/system/cwd/python_executable）、
`data.deps`（逐包 `{import, version, error}`）、`data.artifacts_check`
（`ok` / `manifest_roundtrip` / `run_id` / `cleaned_up`）。顶层同名扁平字段与 `data` 内容一致（兼容早期字段）。
`status=error` 且 `error.code=environment_not_ready` 时**不要继续建模**：把
`deps` 里 `error` 非空的包名和 `error.hint`（建议运行 `risklab-mdl-doctor`）原样转述给用户。

<!-- tool: project_init -->
### project_init —— 建/复用项目目录

**何时调用**：环节 1 第一步。用户给出 CSV 路径后立刻调用；已存在同名 `mdl-*` 目录时
会**复用**（不覆盖历史产物），因此重复调用是安全的。

**入参示例**：

```json
{"csv_path": "D:/work/sample_2000.csv"}
```

**结果解读**：`data.project_dir`（后续所有工具的入参）、`data.csv_path`、`data.created`
（true=本次新建）、`data.manifest_path`、`data.manifest`（`project.target` 为 null 表示目标字段还没确认，
`stages` 六个键全 false，`runs` 空，`confirmed_run` 为 null）。
`warnings` 会明说「复用而不覆盖」或「已创建项目目录」，把这条转述给用户即可。
失败：`path_not_readable`（路径拼错/是目录）、`project_dir_not_writable`（CSV 目录不可写，
**不静默换位置**——让用户把 CSV 复制到可写目录）、`project_dir_conflict`（`mdl-*` 被同名文件占用）。

## 数据（3）

<!-- tool: dataset_load -->
### dataset_load —— 读 CSV + 角色推断（不阻断）

**何时调用**：环节 1 第二步，紧跟 `project_init`。只读表头与全表统计，不写分析产物。

**入参示例**：

```json
{"csv_path": "D:/work/sample_2000.csv"}
```

**结果解读**：`data.encoding`（探测到的编码）、`data.rows`、`data.columns`、
`data.column_count`、`data.dtypes`（语义类型：`bool/integer/float/datetime/categorical/text/empty`）、
`data.roles` = `{id, time, split, target_candidates}`、`data.role_reasons`（逐角色人话理由）、
`data.numeric_feature_count`。**角色是推断，不是结论**：`target_candidates` 可能有多个候选，
必须由用户在前置确认环节拍板（见 SKILL.md「目标字段前置确认」）。
失败：`encoding_undecodable`（文件是二进制/加密/半截 ⇒ 让用户另存 UTF-8 或 GB18030）。

<!-- tool: dataset_columns -->
### dataset_columns —— 列清单与角色（供用户确认/改角色）

**何时调用**：环节 1 第三步。需要把「有哪些列、各自什么角色」摆给用户看并请其确认
目标列 / id 列 / 时间列 / 划分列时调用。

**入参示例**：

```json
{"project_dir": "D:/work/mdl-sample_2000"}
```

**结果解读**：`data.columns[]` 每项 `{name, dtype, pandas_dtype, role, nunique, missing_rate,
same_value_rate, is_high_cardinality}`；`data.roles` / `data.role_reasons` 同 `dataset_load`；
`data.confirmed_target`（null 表示还没确认）。
展示建议：只列「角色非 feature 的列 + 缺失率 TOP 10 + 高基数列」三段，691 个特征列不要整表刷给用户。
失败：`project_dir_not_found`（先 `project_init`）、`manifest_not_found`。

<!-- tool: dataset_profile -->
### dataset_profile —— 样本画像（环节 2）

**何时调用**：目标字段已由用户确认后（`config.target`）。产出 `data_analysis/` 全套画像与图表。

**入参示例**：

```json
{
  "project_dir": "D:/work/mdl-sample_2000",
  "config": {"target": {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}},
             "max_features_for_charts": 30}
}
```

可选键：`target`（缺省取 manifest）、`max_features_for_charts`（默认 30）、`sample_rows`（抽样会显式告警）、
`roles`（只接受 `id`/`time`/`split`）。

**结果解读**：`data.shape`、`data.target`、`data.target_distribution`
（`counts` / `shares` / `bad_rate` / `bad_label` / `missing`）、`data.missing_summary`（缺失率 TOP，
每项含 `column/role/dtype/nunique/missing_rate/same_value_rate`）、`data.feature_types.counts`
（`numeric/categorical/text/datetime/high_cardinality`）、`data.has_time_column`、`data.chart_count`。
`artifacts` 给出 `data_analysis/profile.json`、`describe.csv`、`target_distribution.csv`、
`charts/*.png`、`index.html`（单文件、无外链）。
`status=needs_user_input` ⇒ 还没确认目标字段，把 `data.question` 呈现给用户（见 SKILL.md §4）。
失败：`path_not_readable` / `analysis_failed`。`warnings` 里的「坏率 < 5%」「缺失率 ≥ 50% 的列数」
必须转述——它们决定后面 IV/KS 的读数是否可用。

## 特征分析（2）

<!-- tool: feature_analyze -->
### feature_analyze —— 逐特征分箱 + WOE/IV（环节 3）

**何时调用**：画像确认后。一个特征一行 IV，是特征筛选的输入。

**入参示例**：

```json
{
  "project_dir": "D:/work/mdl-sample_2000",
  "config": {"method": "quantile", "bins": 8,
             "target": {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}}}
}
```

`method` ∈ `quantile`（默认，等频）/ `equal_width`（等宽）/ `dt`（决策树切点）/ `custom`
（需给 `edges`，如 `{"f_0100": ["-inf", 0, 100, 500, "inf"]}`）；`bins` 默认 8（`dt` 时是上限）。

**结果解读**：`data.feature_count`（候选总数）、`data.analyzed_count`（真正算出来的）、
`data.iv_top[]`（`{feature, iv}`，降序，默认 TOP20）、`data.skipped[]`（`{feature, reason}`，如"非数值，需先编码"）、
`data.chart_count`。IV 分级见 [metrics.md](metrics.md) §5（<0.02 无区分 / 0.02–0.1 弱 / 0.1–0.3 中 /
0.3–0.5 强 / ≥0.5 疑似含未来信息）。
`artifacts`：`feature_analysis/iv_table.csv`、`woe_detail.csv`、`binning/<feature>.json`、`charts/*.png`、`index.html`。
必转述的 `warnings`：①「分箱在传入数据全量上拟合，划分完成后需重跑」；②「箱数因样本量不足被下调」
（此时 IV 与历史口径不可比）；③「N 个特征缺失率 ≥ 50%」。宽表（>100 列）只有 IV TOP10 出箱图。

<!-- tool: feature_charts -->
### feature_charts —— 特征分析图表

**何时调用**：`feature_analyze` 之后想补图（相关性热力图、指定特征箱图）时；**不重算 IV**，
只读已有 `iv_table.csv` / `woe_detail.csv`。

**入参示例**：

```json
{"project_dir": "D:/work/mdl-sample_2000", "config": {"max_features": 15}}
```

可选：`features`（只画这些）、`max_features`（默认 30；相关性热力图固定 ≤15 个特征）。

**结果解读**：`artifacts` 为 `feature_analysis/charts/*.png`（IV 分布含 0.02/0.1/0.3 阈值线、
TOP 特征 WOE 箱图、相关性热力图）与 `feature_analysis/index.html`。
无需向用户念数字，给图片路径 + 一句「IV 分布形态 / 最强相关特征对」即可。
失败：缺 `iv_table.csv` 时返回错误（先 `feature_analyze`）；`analysis_failed` 见 [errors.md](errors.md)。

## 预处理与划分（3）

<!-- tool: preprocess_check -->
### preprocess_check —— 入模前检查（环节 6 第一步）

**何时调用**：准备训练前。把「必须处理 / 建议处理」列给用户，作为填写 `preprocess_apply` 的 `spec` 依据。

**入参示例**：

```json
{"project_dir": "D:/work/mdl-sample_2000",
 "config": {"high_missing_rate": 0.5, "high_cardinality_max": 20, "missing_policy": "must_fix"}}
```

可选键：`features`、`target`、`missing_policy`（`must_fix`/`suggested`）、`high_missing_rate`(0.5)、
`high_cardinality_max`(20)、`same_value_rate`(0.999)、`sentinel_value`(99999.0)。

**结果解读**：`data.must_fix[]` / `data.suggested[]`（人可读消息，每条以列名开头，可直接转述）、
`data.must_fix_count` / `data.suggested_count`、`data.by_feature.<列>.issues[]`
（`{code, level, message, suggestion}`，`code` ∈ `missing/not_numeric/high_missing/constant/extreme/
high_cardinality/sentinel_conflict`）。`must_fix` 为空才代表入模矩阵可直接用。
`status=needs_user_input` ⇒ 目标字段未确认。
展示建议：不要逐条列 600+ 条，按 `code` 聚合计数 + 举 3 个例子 + 给出「推荐 spec 草案」请用户确认。

<!-- tool: preprocess_apply -->
### preprocess_apply —— 落地预处理规格（环节 6 第二步）

**何时调用**：用户确认了缺失策略/编码方式/盖帽/分箱切点之后。执行顺序固定
`missing → encode → cap → binning`，产物 `processed.*` 供训练与划分使用。

**入参示例**：

```json
{
  "project_dir": "D:/work/mdl-sample_2000",
  "spec": {
    "missing": {"f_0001": "median", "f_0002": "median", "channel": "mode"},
    "encode": {"channel": "onehot", "note": "ordinal"},
    "cap": {"f_0003": {"strategy": "quantile", "lower_quantile": 0.01, "upper_quantile": 0.99},
            "balance": {"strategy": "absolute", "min": 0, "max": 100000}},
    "binning": {"f_0100": {"edges": ["-inf", 0, 100, 500, "inf"], "missing_as_bin": true}}
  }
}
```

四段都支持「列表形态」`[{column, strategy, ...}]` 与「映射形态」`{列: 策略}` 两种写法；
缺失策略 `mean/median/mode/sentinel/as_bin/drop`；编码 `onehot/ordinal/woe/drop`（woe 仅二分类）；
盖帽 `absolute(min/max)` / `quantile(lower_quantile/upper_quantile)`；分箱 `edges` 首尾可写 `-inf/inf`。

**结果解读**：`data.processed_path`（`preprocess/processed.parquet`，无 pyarrow 时 `.csv`）、
`data.rows_in/rows_out`（不同说明 `drop` 删过行）、`data.fill/encode/cap/binning[]`（逐列实际动作，
含 `n_capped`、`bin_counts`）、`data.model_readiness` = `{lr_ready, non_numeric_columns,
remaining_missing, model_column_count, model_columns}`。
`lr_ready=false` ⇒ LR 不可用（GBDT/XGB/LGB 仍可跑），把 `non_numeric_columns` 与
`remaining_missing` 前几列告诉用户。
失败：`spec_invalid`（未知键/未知列/策略与列类型不符/区间非法/空 spec，消息里带可用策略），
按 [errors.md](errors.md) §3 修 `spec` 后重试。

<!-- tool: dataset_split -->
### dataset_split —— 样本划分（环节 7）

**何时调用**：训练前。默认 `mode=none`（全量口径），风控标准做法是 **7:2:1 分层 + seed=42**
或按时间字段划分。

**入参示例**：

```json
{"project_dir": "D:/work/mdl-sample_2000",
 "config": {"mode": "stratified", "ratios": [0.6, 0.2, 0.2], "seed": 42,
            "stratify_by": "y", "time_column": "month"}}
```

按字段划分：`{"mode": "field", "field": {"column": "month",
"expr_train": "month in (\"2026-01\",\"2026-02\")", "expr_val": "month == \"2026-03\"",
"expr_test": "month not in (\"2026-01\",\"2026-02\",\"2026-03\")"}}`（受限表达式，
支持 `== != > >= < <= in / not in / & / | / ()`，**禁 eval**；字符串字面量必须与数据实际取值一致）。

**结果解读**：`data.mode`、`data.rows`、`data.counts.{train,val,test}`、`data.ratios`、`data.seed`、
`data.fingerprint`（掩码 SHA256 前 16 位，同 seed 同输入必相同，训练时用它核对）、
`data.description`、`data.label_distribution`（核对分层占比偏差）。
`mode=none` 的 `description` 会明说「未划分：全量口径，指标不代表样本外表现」。
失败：`expression_invalid`（非法表达式/引用不存在的列/子集重叠，消息带位置与示例）、
`config_invalid`（比例和不为 1、分层列不存在）。

## 特征筛选（2）

<!-- tool: feature_prescreen -->
### feature_prescreen —— 特征初筛（环节 5）

**何时调用**：`feature_analyze` 之后、训练之前。五道筛子按序生效：
缺失率 → 同值率 → IV → PSI → 相关性。有 `split/split.json` 时**只用 train**。

**入参示例**：

```json
{"project_dir": "D:/work/mdl-sample_2000",
 "config": {"missing_rate_max": 0.5, "same_value_rate_max": 0.95, "iv_min": 0.02,
            "psi_max": 0.1, "corr_max": 0.8}}
```

默认值即上面这五个（全部必须落在 `(0,1)`）。宽表/低坏率场景放宽 `psi_max` 到 0.25 可显著提高存活数。

**结果解读**：`data.n_input` / `data.n_survived`、`data.survived[]`（初筛存活清单）、
`data.dropped[]`（`{feature, stage, reason}`，`stage` 是**第一道**淘汰它的筛子）、
`data.thresholds`（回显生效阈值）、`data.data_scope`（`source` / `rows_used` / `note`）、
`data.psi.{enabled, base_used, note}`、`data.non_numeric[]`、`data.iv_top[]`。
必转述的 `warnings`：「未找到 split/split.json ⇒ 初筛使用全量」（存在穿越风险）、
「N 个特征 IV ≥ 0.5 疑似含未来信息」、「PSI 基期取最早月份」、「非数值列未参与初筛」。
失败：`features_empty`（无数值特征 ⇒ 先 `preprocess_apply` 编码或用 `config.features` 指定）。

<!-- tool: feature_refine -->
### feature_refine —— 特征精筛（gain 迭代）

**何时调用**：初筛之后、训练之前。按 LightGBM gain 迭代去掉弱特征，产出最终入模清单。
需要 `lightgbm`；候选池默认取初筛存活。

**入参示例**：

```json
{"project_dir": "D:/work/mdl-sample_2000",
 "config": {"scheme": "gain", "rounds": 10, "threshold": 0.05, "target_range": [5, 60],
            "keep": ["f_0426"], "drop": ["f_0012"]}}
```

可选键：`scheme`（仅 `gain`，其它值明确报错不降级）、`candidates`（显式候选池，优先级最高）、
`keep`/`drop`（合规刚需的强保留/强剔除）、`rounds`(10)、`threshold`(0.05)、
`target_range`(默认 `[min(5,n), min(60,n)]`)、`max_features`(默认 60)、`seed`(42)、
`params`（覆盖 LightGBM 参数）、`num_boost_round`(1000)、`early_stopping_rounds`(50)。

**结果解读**：`data.n_candidates` / `data.n_selected`、`data.features[]`（最终入模清单）、
`data.disposition` ∈ `converged / truncated_high / relaxed_converged / relaxed_truncated / relaxed_failed`、
`data.message`（拍板值 → 生效值的说明）、`data.attempts[]`（放宽尝试）、
`data.gain_shape`（增益集中度 + `advice`）、`data.split`（用的是哪份划分：`split.json` 还是
`core.split` 阶段兜底，`train_rows`/`val_rows`）。`artifacts` 含
`feature_selection/refine.json` 与 `selected_features.txt`（UTF-8 无 BOM，一行一个）。
`disposition != converged` 必须转述（说明特征数被截断或阈值被放宽）。
失败：`dependency_missing`（缺 lightgbm，按 `hint` 装）、`config_invalid`（val 为空 ⇒ 先 `dataset_split`）。

---

## 模型（4）

<!-- tool: model_catalog -->
### model_catalog —— 算法 × 任务能力表（环节 4）

**何时调用**：环节 4 模型选型。无入参、只读、不训练，用来把「5 算法 × 2 任务 = 10 格」
的能力、默认参数、默认调参空间摆给用户选。

**入参示例**：

```json
{}
```

**结果解读**：`data.catalog` = `{schema, algorithms{dt,lr,gbdt,xgb,lgb}, algorithm_order, tasks,
matrix, matrix_size, label_rules}`；每算法含 `label`、`tasks`、`dependency`/`install`、
`default_params`、`default_space`、`by_task{binary,multiclass}`、`notes`。
`data.available.<algo>` = `{installed, dependency}` —— `installed=false` 的算法**不要推荐**
（如缺 xgboost 时按 `install` 给 pip 命令）。
展示：一张 5 行表（算法 / 可做任务 / 默认关键参数 / 是否装好 / 一句话场景），
详细参数见 [models.md](models.md)。

<!-- tool: model_train -->
### model_train —— 训练一个 run（环节 8）

**何时调用**：特征清单与划分就绪后。一次调用产出一个 `runs/<run_id>/`，并追加 `manifest.runs`。

**入参示例**：

```json
{"project_dir": "D:/work/mdl-sample_2000",
 "config": {"algorithm": "lgb", "task": "binary", "seed": 42}}
```

可选键：`features`（显式特征清单，优先级最高；点名非数值列 ⇒ `config_invalid`）、
`preprocess`（未 `preprocess_apply` 时在内存里执行，不落盘）、`split`（覆盖 `split/split.json`）、
`params`（键名必须是该算法 `default_params` 里的参数，否则 `config_invalid`）、`seed`(42)、
`num_boost_round`、`early_stopping_rounds`。数据优先级：
`preprocess/processed.parquet|csv` → 原始 CSV（+`config.preprocess`）→ 原始 CSV；
特征优先级：`config.features` → `selected_features.txt` → 全部数值列；
划分优先级：`split/split.json` → 内部 80/20 分层（不落盘 + 告警）。

**结果解读**：`data.run_id`（后续 `model_evaluate`/`model_confirm`/`run_compare` 都用它）、
`data.metrics.{train,val,test}` 每项 `{ks, auc, gini, lift, n, bad, good, status, reason}`
（`status="skipped"` + `reason` 表示该子集只有一类标签，**不是 0 分**）、
`data.multi`（多分类的 `per_class`/`macro`，二分类为 null）、`data.classes`/`data.bad_class`、
`data.n_features`/`data.features`、`data.feature_source`、`data.data_source`、
`data.sample_sizes`、`data.split`（`source`/`fingerprint`）、`data.params`（最终超参）、
`data.best_iteration`/`data.used_early_stopping`、`data.fit_seconds`。
`warnings` 必须转述：内部划分兜底、train 中位数填充、LR 未收敛、非数值列未入模、早停未真正生效。
失败：`task_algorithm_mismatch`（该算法不支持该任务，**不降级**，按 `model_catalog` 换算法）、
`dependency_missing`、`config_invalid`、`analysis_failed`；
`status=needs_user_input` ⇒ 目标字段还没确认。

<!-- tool: model_tune -->
### model_tune —— 调参（环节 8 可选）

**何时调用**：`model_train` 出基线后想提升指标时。先 `mode=preview` 把搜索空间给用户看，
确认后再 `mode=execute`（Hyperopt TPE，落一个 `tuned` run）。

**入参示例**：

```json
{"project_dir": "D:/work/mdl-sample_2000",
 "config": {"algorithm": "lgb", "task": "binary", "trials": 20, "seed": 42},
 "mode": "preview"}
```

可选：`space`（自定义参数空间，未知参数名 ⇒ `config_invalid`）、`trials`、`objective`（权重）、
`seed`。`tune_on` 只允许 `val`（传 `oot`/`train`/`cv` ⇒ `config_invalid`，防止在测试集上调参）。

**结果解读**：preview 返回 `data.space`（参数名/类型/范围/默认）、`data.space_table`
（人读表：参数 / 类型 / 搜索范围）、`data.expected_trials`、`data.objective`
（`weights`、`gap_threshold`=0.05、`formula`）、`data.tune_on`、`data.sample_sizes`、`data.notes`，
**不训练**（`warnings` 会说明「未训练任何模型」）；execute 额外返回新 `run_id`、`best_params`、
`trials`（轨迹表在 `runs/<run_id>/tuning/{space.json,trials.csv}`）。
展示：先把 `space_table` 整表给用户 + 「预计 20 次试验、目标 = 0.5×KS + 0.4×AUC − 0.1×衰减」，
用户点头再 execute。调参耗时是训练的 N 倍，宽表先降 `trials`。

<!-- tool: model_evaluate -->
### model_evaluate —— 指标与跨期稳定性（环节 9）

**何时调用**：训练后立即调用（`model_train` 已带 train/val/test 指标，本工具补齐
KS/LIFT 表、曲线与月度 PSI，并刷新 `metrics.json`）。

**入参示例**：

```json
{"project_dir": "D:/work/mdl-sample_2000", "run_id": "20261116T213542",
 "config": {"psi_base_months": ["2026-01"]}}
```

可选：`run_id`（缺省用最新 run）、`config.sample`（`train/val/test`）、`config.by_time`、
`config.time_column`/`config.month_column`、`config.psi_base_months`（合并多个月为基期池）。

**结果解读**：`data.metrics.{train,val,test}`（同上）与 `data.multi`（多分类）、
`data.psi` = `{enabled, reason, base_months, base_label, n_base, n_bins, by_month[{month, psi, n}],
max_psi, max_month, overall_psi, time_column}`、`data.tables`（各表行数）、`data.charts`（PNG 路径）。
**闸门看 `max_psi`（最差月），不看 `overall_psi`**；`enabled=false` 时 `reason` 说明为何没算
（无时间列 / 只有 1 个月 / 基期覆盖全部月份），此时**不得**声称「跨期稳定」。
`artifacts`：`runs/<run_id>/metrics.json`、`evaluation/{psi_by_month,metrics_by_subset,ks_table,lift_table}.csv`、
`evaluation/{ks_curve,lift_chart}.png`。
失败：`run_not_found`（用 `run_list` 查可用 run）、`config_invalid`（`psi_base_months` 里有不存在的月份，
**不静默回落**）。

## 过程（4）

<!-- tool: run_list -->
### run_list —— 列出全部 run

**何时调用**：需要让用户挑 run（确认哪个、对比哪几个）时；也用于在 `run_not_found` 后找回合法 `run_id`。

**入参示例**：

```json
{"project_dir": "D:/work/mdl-sample_2000"}
```

**结果解读**：`data.runs[]` 每项 `{run_id, algorithm, task, status, metrics{ks,auc,gini,lift,subset,psi},
created_at, confirmed, has_model, has_evaluation, primary_ks, metrics_json}`，
按 `run_id` 升序；另有 `data.confirmed_run` / `data.n_runs`。
`status` ∈ `trained / tuned / evaluated / confirmed / failed`。展示成表格：
run_id / 算法 / val KS / val AUC / PSI / 状态 / 是否已确认。

<!-- tool: run_compare -->
### run_compare —— run 对比与最优标记

**何时调用**：有了 ≥2 个 run（不同算法或不同调参）后，需要「哪个模型最好」的结论时。

**入参示例**：

```json
{"project_dir": "D:/work/mdl-sample_2000", "run_ids": ["20261116T213542"]}
```

`run_ids` 缺省 = 全部 run。

**结果解读**：`data.primary_metric`（固定 `ks`）、`data.best`（最优 `run_id`，**没有任何 run 有可用 KS 时为 null**，
不会默认挑第一个）、`data.rows[]`（逐 run × 子集的扁平指标 + `best` 标记）、`data.compare_csv`。
`artifacts` 写 `reports/run_compare.csv`。注意：**主指标 KS 取自 val 优先**，
不要拿 train KS 当比较依据（宽表上 train KS 常接近 1）。

<!-- tool: model_confirm -->
### model_confirm —— 确认闸门（环节 9 收尾）

**何时调用**：用户选定 run 且指标达标后。**只有本工具会写 `final/`**；不达标就必须把差距
如实报给用户，不能"差不多就行"。

**入参示例**：

```json
{"project_dir": "D:/work/mdl-sample_2000", "run_id": "20261116T213542",
 "thresholds": {"ks_min": 0.25, "psi_max": 0.10}}
```

`thresholds` 缺省即 **KS ≥ 0.25 且 PSI ≤ 0.10**；两者可单独覆盖（`ks_min∈[0,1]`，`psi_max≥0`）。

**结果解读**：`data.ks`/`data.ks_source`（`val` 优先，val 被跳过时回落 `train`）、`data.psi`、
`data.ks_gap`/`data.psi_gap`（负值=超标富余，正负号含义见 [metrics.md](metrics.md) §8）、
`data.checks[]`/`data.failed_checks[]`（人话）、`data.confirmed`、`data.confirmed_at`、
`data.final_model`/`data.confirm_path`、`data.metrics`。
通过 ⇒ `final/model.pkl` + `final/confirm.json` + `manifest.confirmed_run` + run 状态 `confirmed`。
失败：`error.code=threshold_not_met`，`data` 里带 `ks/ks_gap/psi/psi_gap/failed_checks`
——把差距值原样转述，并给用户三条路：①回去降复杂度/加筛选重训 ②换算法 ③**显式**调阈值（要写清风险）。
PSI 未计算时只校验 KS，`warnings` 会明说「不代表跨期稳定」。

<!-- tool: report_build -->
### report_build —— 汇总报告（只读产物、不重算）

**何时调用**：流程最后一步（确认之后），或用户中途要一份进度报告时。**只读产物、不重算**。

**入参示例**：

```json
{"project_dir": "D:/work/mdl-sample_2000"}
```

**结果解读**：`data.xlsx`（`reports/结果汇总.xlsx`，四个非空 sheet：`概览` / `模型指标` / `run对比` /
`特征重要性` 及其行数）、`data.markdown`（`reports/过程报告.md`）、`data.n_runs`、`data.best_run`。
报告里的数字一律来自 `runs/*/metrics.json`，因此与确认时的读数一致；若用户看到"报告和刚才的 KS 不一样"，
先查是否又跑了新 run（`run_list`），不要改报告。
失败：无 run 时仍可出概览，`warnings` 会说明缺哪些阶段；`path_not_writable` 见 [errors.md](errors.md)。
