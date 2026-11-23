# 预处理 / 样本划分 / 特征筛选

> 范围：建模流程环节 6.1–6.3 —— 入模前检查 → 预处理落地（缺失/编码/盖帽/手动分箱）
> → 样本划分 → 特征初筛（五道筛子）→ 特征精筛（gain 迭代）。
> 契约来源：[tool-contract.md](./tool-contract.md) §1 / §3.2 / §3.3 / §4；
> 口径来源：[口径-metrics.md](./口径-metrics.md)（`core.split` 的阶段口径）。
> 代码位置：`src/risklab_mdl/{preprocess,tools}/`。上游依赖数据与特征分析层的 `data/`、`features/` 产物。

本层**只做到「可训练的数据 + 划分掩码 + 入模特征清单」为止**：不训练模型、不调参、
不评价、不写 `model_*` 工具（属模型层）。

---

## 1. 五个工具一览

| 工具 | 入参 | 关键产物 | 说明 |
|---|---|---|---|
| `preprocess_check` | `project_dir`, `config?` | `preprocess/check.json` | 逐特征 must_fix / suggested 分类 + 依据 |
| `preprocess_apply` | `project_dir`, `spec` | `preprocess/{fill,encode,outlier,manual_bins,summary}.json`、`preprocess/processed.parquet`（或 `.csv`） | 落地四段规格 + 处理后的数据 |
| `dataset_split` | `project_dir`, `config` | `split/split.json` | none / random / stratified / field（受限表达式，禁 `eval`） |
| `feature_prescreen` | `project_dir`, `config` | `feature_selection/prescreen.json` | 五道筛子：缺失率 → 同值率 → IV → PSI → 相关性 |
| `feature_refine` | `project_dir`, `config` | `feature_selection/refine.json`、`selected_features.txt` | gain 迭代（依赖 lightgbm） |

所有工具返回统一三态 `{ok, status, data, artifacts, warnings, error}`（契约 §1）。
**任何降级都进 `warnings`**（parquet 不可用落 CSV、PSI 跳过、WOE 全量拟合、行被删除、
入模矩阵仍不可用…），不允许「空结果 + ok:true」。

### 1.1 典型调用链

```python
project_init(csv_path=r"D:\work\sample_2000.csv")
dataset_load(csv_path=r"D:\work\sample_2000.csv")
dataset_profile(project_dir=proj, config={"target": {"column": "y", "task": "binary",
                                                     "labels": {"1": "bad", "0": "good"}}})
feature_analyze(project_dir=proj, config={"method": "quantile", "bins": 8})

preprocess_check(project_dir=proj)                       # 先看 must_fix / suggested
preprocess_apply(project_dir=proj, spec={...})            # 落地规格 + processed 数据
dataset_split(project_dir=proj, config={"mode": "stratified", "ratios": [0.7, 0.2, 0.1],
                                        "seed": 42, "stratify_by": "y"})
feature_prescreen(project_dir=proj, config={"iv_min": 0.02, "psi_max": 0.1})
feature_refine(project_dir=proj, config={"scheme": "gain", "rounds": 10, "threshold": 0.05})
```

> **顺序纪律**：`feature_prescreen` 的取数口径是 **train**（有 `split/split.json` 时），
> `feature_refine` 用 **train + val**。先划分再筛选，否则「筛选结果掺入未来信息」——
> 这类错误不报错、结果看着合理，但整条链路的结论全错（详见 `core/split.py` 模块 docstring）。

---

## 2. `preprocess_check` —— 入模前检查

### 2.1 规则（`config` 可逐项覆盖，全部回显在 `check.json.summary.rules`）

| 问题 | 级别 | 判据 | 建议处置 |
|---|---|---|---|
| 非数值列未编码 | **must_fix** | `not is_numeric_series`（dtype ∈ categorical/text） | `spec.encode` 声明 onehot/ordinal/woe |
| 特征列有缺失 | **must_fix** | `missing_rate > 0`（`missing_policy` 可降级） | `spec.missing` 声明填充策略 |
| 高基数类别 | suggested | 非数值且 `nunique > high_cardinality_max`（默认 20）或 dtype=text | 改 ordinal/woe，或先合并低频 |
| 缺失过多 | suggested | `missing_rate >= high_missing_rate`（默认 0.5） | 考虑剔除或 `as_bin` 后看 IV |
| 同值列（常量） | suggested | `nunique <= 1` 或最高频占比 ≥ `same_value_rate`（默认 0.999） | `spec.encode` 用 drop 剔除 |
| 极端值 / 偏态 | suggested | 3σ 偏离 > 3 或 IQR 离群比例 > 1%（`core.transforms` 同口径） | `spec.cap` 盖帽 |
| 哨兵值冲突 | suggested | 数值列真实取值里已出现 `99999.0` | 换填充策略或换哨兵值 |

`config` 键（全部可选）：

```json
{
  "features": ["amount", "channel", "score"],
  "target": {"column": "y", "task": "binary"},
  "missing_policy": "must_fix",
  "high_missing_rate": 0.5,
  "high_cardinality_max": 20,
  "same_value_rate": 0.999,
  "sentinel_value": 99999.0
}
```

### 2.2 产物 `preprocess/check.json`

```json
{
  "schema": "risklab-mdl.preprocess-check/1",
  "version": "2.0.0",
  "generated_at": "2026-…",
  "target": {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}},
  "roles": {"id": "id", "time": "month", "split": "split"},
  "features": ["amount", "channel", "score", "const"],
  "must_fix": ["channel: 非数值列（推断类型 categorical，取值 4 个）未编码，…", "…"],
  "suggested": ["const: 同值列（取值 1 个，最高频占比 100.00%），IV 恒为 0，…", "…"],
  "by_feature": {
    "channel": {
      "dtype": "categorical", "numeric": false, "role": "feature",
      "nunique": 4, "missing_count": 22, "missing_rate": 0.122222,
      "same_value_rate": 0.5, "sigma_deviation": null, "iqr_outlier_ratio": null,
      "issues": [
        {"column": "channel", "code": "not_numeric", "level": "must_fix",
         "message": "channel: 非数值列（…）", "suggestion": "spec.encode …"},
        {"column": "channel", "code": "missing", "level": "must_fix", "...": "..."}
      ],
      "must_fix": ["…"], "suggested": []
    }
  },
  "summary": {
    "rows": 180, "feature_count": 4, "must_fix_count": 3, "suggested_count": 6,
    "issues_by_code": {"missing": 2, "not_numeric": 2, "constant": 1, "extreme": 1},
    "rules": {"missing_policy": "must_fix", "high_missing_rate": 0.5, "...": "..."}
  },
  "warnings": ["…"]
}
```

* `must_fix` / `suggested` 是**人可读的消息列表**（每条以列名开头），供 agent 直接转述；
  结构化明细在 `by_feature.<列>.issues[]`（`code`/`level`/`message`/`suggestion`）。
* 被检查的每一列都在 `by_feature` 里，即使没有任何问题（`issues: []`）——
  「这一列检查过且没问题」与「这一列没检查」必须能区分。
* 未确认目标字段时返回 `needs_user_input`（`question.id=dataset_confirm_target`）。

---

## 3. `preprocess_apply` —— 预处理落地

### 3.1 `spec` 结构（四段都可写成「列表形态」或「`{列: 策略/参数}` 映射形态」）

```json
{
  "missing": [
    {"column": "amount", "strategy": "median"},
    {"column": "channel", "strategy": "mode"},
    {"column": "note", "strategy": "as_bin"}
  ],
  "encode": [
    {"column": "channel", "strategy": "onehot"},
    {"column": "note", "strategy": "woe"}
  ],
  "cap": [
    {"column": "score", "strategy": "quantile", "lower_quantile": 0.01, "upper_quantile": 0.99},
    {"column": "balance", "strategy": "absolute", "min": 0, "max": 100000}
  ],
  "binning": {
    "amount": {"edges": ["-inf", 0, 100, 500, "inf"], "missing_as_bin": true}
  }
}
```

映射形态等价写法：

```json
{
  "missing": {"amount": "median", "channel": {"strategy": "sentinel", "value": 99999.0}},
  "encode": {"channel": "onehot"},
  "cap": {"score": {"strategy": "absolute", "max": 100}},
  "binning": {"amount": [ -inf, 0, 100, 500, inf ]}
}
```

#### 缺失策略（`spec.missing[].strategy`）

| 策略 | 语义 | 约束 |
|---|---|---|
| `mean` / `median` | 数值列均值/中位数填充 | 仅数值列 |
| `mode` | 众数填充（缺失若为唯一众数则仍缺） | 任何列 |
| `sentinel` | 常数填充，默认 `99999.0`（`value` 可覆盖） | 仅数值列；真实取值已含该值时**报错**（缺失与真实值不可区分） |
| `as_bin` | 保留缺失，由分箱落为独立箱 `__MISSING__` | 需与 `spec.binning` 配套，否则 processed 里该列仍有 NaN（`warnings` 说明） |
| `drop` | 删除该列缺失的行 | 会改变行数 ⇒ 已有 `split/split.json` 时 `warnings` 提示 mask 无法对齐 |

#### 编码策略（`spec.encode[].strategy`）

| 策略 | 语义 | 说明 |
|---|---|---|
| `onehot` | 每档一列 0/1，缺失单独成列（`__MISSING__`） | 档数 > 20 时 `warnings` 提示稀疏共线 |
| `ordinal` | 按取值字典序映射为 0,1,2… | 映射表落 `encode.json.mapping`；缺失保留 NaN ⇒ 需先填充 |
| `woe` | 按坏样本率 WOE 编码，未知取值走 `unknown` 档 | **仅二分类**；有 `split/split.json` 时只用 train 行拟合，否则全量拟合并**告警标签泄漏风险** |
| `drop` | 直接剔除该列 | 与 cap/binning 同列冲突 ⇒ `spec_invalid` |

> WOE 表必须回写 `encode.json`（`woe_table.mapping`/`iv`/`fit_scope`），否则上线无法复现同一套编码。

#### 盖帽策略（`spec.cap[].strategy`）

| 策略 | 参数 | 语义 |
|---|---|---|
| `absolute` | `min` / `max`（至少一个） | 按绝对值截断（`min < max`） |
| `quantile` | `lower_quantile` / `upper_quantile`（至少一个，∈[0,1]） | 按分位数截断（**拟合在传入数据上**，与标签无关，报告里写明 `fit_scope`） |

只给单边（如仅 `max`/`upper_quantile`）时只截单边；缺失值不参与分位计算也不被截断。

#### 手动分箱（`spec.binning`）

* `edges`：切点列表，**首尾 `-inf`/`inf` 可写可不写**（写了会被识别为边界，内部统一补 ±∞，
  覆盖全值域）；内部切点必须**有限**、**严格递增**、**至少一个**。
* 箱区间语义与 `core.binning.transform_bins` 一致：`right=True, include_lowest=True`
  ⇒ 箱标签 `(-inf, a]`、`(a, b]`、…、`(z, inf)`。
* 缺失：`missing_as_bin=true`（默认）⇒ 缺失落箱编码 `-1`，箱表里记为 `__MISSING__`；
  设为 `false` 时缺失保留 NaN 并**告警**（入模矩阵不可用）。
* **分箱会替换原列**：该列在 processed 数据里变成箱编码 `0..k-1`（缺失 `-1`），
  原始取值可从 CSV 回溯；完整映射在 `manual_bins.json.bin_counts` / `bin_labels`。

### 3.2 校验规则（违反一律 `spec_invalid`，不静默跳过）

1. `spec` 必须是对象，顶层只允许 `missing` / `encode` / `cap` / `binning`；
2. 四段允许的字段见上（`column` + 各段参数），**未知字段**报错（防拼写错误静默失效）；
3. 列必须存在于数据中；**不得作用于目标列**；不得作用于角色列（id/time/split）；
4. 同一段内同一列不得重复；
5. 策略必须在枚举内，且策略与列类型匹配（mean/median/sentinel 仅数值；cap/binning 仅数值）；
6. 盖帽区间：`min < max`、分位 `0 <= lo < hi <= 1`；
7. 分箱：内部切点有限、严格递增、至少一个；`-inf/inf` 只能出现在首尾；
8. `encode.drop` 的列不得再出现在 `cap`/`binning`（执行顺序会互相矛盾）；
9. 空 spec（四段都为空）报错。

### 3.3 执行顺序（固定，写进 `summary.json.order`）

```
missing（填充/删行） → encode（编码） → cap（盖帽） → binning（手动分箱）
```

理由：① 编码与盖帽需要确定的取值域，所以先填充；② 盖帽只对数值列生效，编码后才有；
③ 分箱作用在最终数值上（先盖帽再分箱，箱内极值才不会被切碎）。

### 3.4 产物

```
preprocess/
  fill.json         # {schema, generated_at, target, order, entries:[{column, strategy, value,
                    #   missing_before, missing_after, rows_removed}]}
  encode.json       # {entries:[{column, strategy, output_columns, mapping|onehot_values|woe_table,
                    #   fit_scope, missing_preserved}], notes}
  outlier.json      # {entries:[{column, strategy, lower, upper, fit_scope, n_capped,
                    #   min_before, max_before, min_after, max_after}]}
  manual_bins.json  # {entries:[{column, edges, interior_edges, n_bins, missing_as_bin,
                    #   missing_bin_label:"__MISSING__", missing_count, bin_counts, bin_labels,
                    #   values_are:"bin_code", row_count}]}
  summary.json      # 处理摘要（缺失前/后、编码列、盖帽列、分箱列、行数/列数、入模就绪）
  processed.parquet # 处理后的数据（供模型层直接读取）；无 pyarrow 时落 processed.csv 并告警
```

`summary.json.model_readiness`：

```json
{
  "lr_ready": false,
  "non_numeric_columns": ["note"],
  "remaining_missing": {"amount": 12},
  "model_column_count": 9,
  "model_columns": ["f_signal", "channel_app", "…"]
}
```

`lr_ready=true` 的含义：除目标列与 id/time/split 角色列外的所有列**都是数值且无缺失**
（LR 的硬门槛；GBDT 可容忍缺失，但这里仍如实报告）。

### 3.5 示例：缺失 + 文本 → LR 就绪

```python
preprocess_apply(project_dir=proj, spec={
  "missing": {"amount": "median", "channel": "mode"},
  "encode": {"channel": "onehot", "note": "ordinal"},
  "cap": {"score": {"strategy": "quantile", "lower_quantile": 0.01, "upper_quantile": 0.99}},
})
# data.model_readiness.lr_ready == True
# processed 数据：channel 被 channel_app/channel_partner/… 替换；note 变成 0..k-1；score 被截断
```

---

## 4. `dataset_split` —— 样本划分

### 4.1 `config`

| 键 | 说明 |
|---|---|
| `mode` | `none` / `random` / `stratified` / `field`（默认 `none`） |
| `ratios` | `[train, val, test]`，合计必须为 1（也接受 `train/val/test` 三个键或 `{ratio}` 形态） |
| `seed` | 随机种子（默认 42）；同 seed + 同输入 ⇒ 逐行相同的掩码 |
| `stratify_by` | 分层列（仅 `stratified`）；缺省用**目标列** |
| `field` | `{column, expr_train, expr_val, expr_test}`（`field` 模式，见 §4.2） |
| `time_column` | 时间列（仅记录进产物，供 PSI 分月使用，不参与划分） |
| `target` | `{column, task, labels?}`；缺省取 `manifest.project.target`（用于标签分布） |

示例：

```json
{"mode": "random", "ratios": [0.7, 0.2, 0.1], "seed": 42, "time_column": "month"}
{"mode": "stratified", "ratios": [0.6, 0.2, 0.2], "seed": 7, "stratify_by": "y"}
{"mode": "field", "field": {
   "column": "month",
   "expr_train": "month in ('2026-01','2026-02') & amount >= 0",
   "expr_val":   "month == '2026-03'",
   "expr_test":  "month not in ('2026-01','2026-02','2026-03')"
 }}
```

### 4.2 字段表达式（受限解析器，**禁 `eval`**）

支持：`==` `!=` `>` `>=` `<` `<=`、`in (...)`、`not in (...)`、`&`、`|`、括号、
数值/字符串字面量、`true`/`false`/`null`、反引号列名 `` `列名` ``。

```text
month in ('2026-01','2026-02')            # 集合包含
amount >= 1000                            # 数值比较（列含非数值会报错）
(channel == 'app' | channel == 'web') & score != null
f_missing == null                         # 等价 isna()
```

**明确拒绝**（⇒ `expression_invalid`，消息带位置、原表达式、支持语法与示例）：

```text
__import__('os').system('echo hi')        # 函数调用 / 双下划线标识符
f_signal.amount > 1                       # 属性访问
y = 1                                     # 单等号赋值（提示改用 ==）
open('C:/tmp/x')                          # 函数调用
f_signal > 1; y == 0                      # 分号
f_signal > 1 & (f_weak > 0                 # 括号未闭合
```

错误示例（真实返回）：

```json
{"code": "expression_invalid",
 "message": "字段表达式非法（位置 16）：出现不允许的字符 '.'（不支持函数调用 / 属性访问 / 赋值 / 通配符）\n★ 原表达式：\"__import__('os').system('x')\"\n★ 只支持 == != > >= < <= in / not in / & / | / 括号 / 数值与字符串字面量。",
 "hint": "示例：month in ('2026-01','2026-02') / amount >= 1000 / …"}
```

其他校验：三个子集**必须互斥**（重叠 ⇒ `expression_invalid`，消息给出重叠行数）；
未被任何条件覆盖的行会被排除并 `warnings` 说明（不静默）。

### 4.3 产物 `split/split.json`

```json
{
  "schema": "risklab-mdl.split/1",
  "version": "2.0.0",
  "generated_at": "2026-…",
  "mode": "stratified",
  "target": {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}},
  "rows": 400,
  "spec": {"mode": "stratified", "ratios": [0.6,0.2,0.2], "train": {"ratio":0.6,"seed":42},
            "val": {"ratio":0.2}, "test": {"ratio":0.2}, "seed": 42,
            "stratify_by": "y", "column": null, "expression": null, "time_column": null},
  "counts": {"train": 240, "val": 80, "test": 80},
  "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
  "seed": 42,
  "fingerprint": "26130619b2da0bf1",
  "description": "分层划分（按 y，seed=42）：train 240 / val 80 / test 80 行",
  "label_distribution": {
    "train": {"count": 240, "counts": {"1": 120, "0": 120}, "shares": {"1": 0.5, "0": 0.5}},
    "val": {"...": "..."}, "test": {"...": "..."}
  },
  "warnings": ["…"]
}
```

* `fingerprint` = 三段掩码按位打包后的 SHA256 前 16 位：**同 seed 同输入必相同**，
  可用来核对「模型层复算的掩码」与「划分落盘结果」是否是同一份。
* `label_distribution` 仅在目标列已知时给出（用于核对分层各类占比偏差）。
* `mode=none` 时 `description` 明确写「未划分：全量口径，指标不代表样本外表现」。

---

## 5. `feature_prescreen` —— 特征初筛

### 5.1 阈值（`config`，默认值见下；全部回显在产物 `thresholds`）

| 键 | 默认 | 语义 |
|---|---|---|
| `missing_rate_max` | 0.5 | 缺失率超过即剔除 |
| `same_value_rate_max` | 0.95 | 最高频取值占比超过即剔除（含缺失作为一档） |
| `iv_min` | 0.02 | IV 低于即剔除 |
| `psi_max` | 0.10 | 逐月 PSI 超过即剔除（无时间列时跳过） |
| `corr_max` | 0.8 | 相关性超过即剔除（保留 IV 更高的那个） |

所有阈值必须在 `(0, 1)` 内（`core.prescreen` 零兜底：阈值由调用方注入，不设隐藏默认；
本工具层的默认值仅作为「显式声明的默认」写进产物，可随时覆盖）。

### 5.2 口径

* **只用 train**：有 `split/split.json` 时按它复算 train 掩码；没有时用**全量**并在
  `data_scope.note` / `warnings` 说明「无切分信息时无从界定训练期」（`core.split` 的既定兜底）。
* **目标缺失行**被剔除（`data_scope.target_missing_rows` 记录行数，分母随之变化）。
* **非数值列不参与初筛**：列进 `non_numeric` 与逐特征表的 `type_check` 阶段（原因：需先编码）。
* **PSI**：时间列取 `config.month_column` → `config.time_column` → manifest 的 `roles.time`；
  无时间列、列不存在、或 train 只有 1 个月 ⇒ **跳过并告警**，不伪造数值。
  基期显式取「最早月份」。
* **多分类**：按类 one-vs-rest 逐类算 IV，**某类 IV 达标即保留**（判定值取各类 IV 最大值），
  逐类明细落 `features[].iv_by_class`；缺失/同值/PSI 三道与标签无关的筛子只跑一遍。

### 5.3 产物 `feature_selection/prescreen.json`

```json
{
  "schema": "risklab-mdl.prescreen/1",
  "target": {"column": "y", "task": "binary"},
  "task": "binary",
  "thresholds": {"missing_rate_max": 0.5, "iv_min": 0.02, "psi_max": 0.1,
                  "corr_max": 0.8, "same_value_rate_max": 0.95},
  "data_scope": {"source": "split/split.json（train 子集）", "mode": "random",
                  "fingerprint": "26130619b2da0bf1", "train_rows": 280, "rows_total": 400,
                  "rows_used": 280, "target_missing_rows": 0},
  "time_column": "month",
  "psi": {"enabled": true, "base_used": "2026-01", "note": "PSI 基期取最早月份；按月对比 3 个月"},
  "stages": ["missing_rate", "same_value_rate", "iv", "psi", "correlation"],
  "summary": {"total_input": 6, "n_survived": 4,
              "stage_dropped": {"missing_rate": 0, "same_value_rate": 1, "iv": 0,
                                 "psi": 0, "correlation": 1},
              "iv_top": [{"feature": "f_signal", "iv": 0.62}, "…"],
              "high_iv_columns": [], "high_psi_columns": []},
  "survived": ["f_signal", "f_weak", "f_noise", "f_missing"],
  "features": [
    {"feature": "f_dup", "survived": false, "dropped_stage": "correlation",
     "reason": "已被保留特征 f_signal 按 |corr|=0.993 超过阈值 0.8",
     "metrics": {"missing_rate": 0.0, "same_value_rate": 0.0, "iv": 0.6, "psi": 0.01}},
    {"feature": "f_signal", "survived": true, "dropped_stage": null, "reason": "",
     "metrics": {"missing_rate": 0.0, "same_value_rate": 0.0, "iv": 0.62, "psi": 0.01}}
  ],
  "non_numeric": ["f_text"],
  "warnings": ["…"]
}
```

* `features[]` 覆盖**所有**候选（含被淘汰的），`dropped_stage` 是**第一道**淘汰它的筛子。
* 多分类记录额外带 `iv_by_class: {"0": 0.12, "1": 0.35, "2": 0.05}`。

---

## 6. `feature_refine` —— 特征精筛（gain 迭代）

### 6.1 `config`

| 键 | 默认 | 说明 |
|---|---|---|
| `scheme` | `gain` | 当前只支持 gain 迭代；`shap`/`boruta` 等**明确报错不降级** |
| `candidates` | 取初筛存活 | 显式指定候选池（优先级最高） |
| `keep` | `[]` | 强制保留（不参与迭代，合规刚需） |
| `drop` | `[]` | 强制剔除（合规刚需） |
| `rounds` | 10 | 最大迭代轮数 |
| `threshold` | 0.05 | 每轮保留线 `gain >= max_gain * threshold` |
| `target_range` | `[min(5,n), min(60,n)]` | 期望最终特征数区间（自适应候选数） |
| `max_features` | `min(60,n)` | 硬上限（超限按累计 gain 截断） |
| `seed` | 42 | LightGBM `random_state` |
| `params` | 见下 | 覆盖 LightGBM 参数 |
| `num_boost_round` / `early_stopping_rounds` | 1000 / 50 | 训练轮数与早停 |

默认参数：二分类 `{objective: binary, metric: auc, num_leaves: 31, learning_rate: 0.05}`；
多分类 `{objective: multiclass, num_class: K, metric: multi_logloss}`（此时轮次表 `valid_auc` 为 null）。

### 6.2 取数口径与候选池

* **候选池优先级**：`config.candidates` > `feature_selection/prescreen.json` 的 `survived` >
  `feature_analysis/iv_table.csv` 的可分析特征 > 数据中全部数值列（**告警建议先初筛**）。
* **绝不默认取所有列**：标签列/月份列混入候选会让 AUC/KS 直接 1.0（标签泄漏）。
  `core.refine` 对 `forbidden`（目标列、id/time/split）与白名单做硬校验。
* **train + val**：有 `split/split.json` 用其 train/val 掩码（val 为空 ⇒ `config_invalid`）；
  没有则按 `core.split` 的阶段兜底（有 split 列按取值取 train/val；只有时间列则最早月为 train）。
* 依赖 `lightgbm`；缺库返回 `error.code=dependency_missing`（附 pip 命令）。

### 6.3 产物

`feature_selection/refine.json`：

```json
{
  "schema": "risklab-mdl.refine/1",
  "method": "gain_iterative",
  "target": {"column": "y", "task": "binary"},
  "candidate_source": "feature_selection/prescreen.json（初筛存活）",
  "candidates": ["f_signal", "f_weak", "f_noise", "f_missing"],
  "n_candidates": 4,
  "params": {"objective": "binary", "metric": "auc", "num_leaves": 31, "…": "…"},
  "refine_config": {"threshold": 0.05, "max_rounds": 10, "target_range": [4, 4],
                     "max_features": 4, "threshold_relax_ladder": [0.03, 0.01],
                     "force_keep": [], "force_drop": [], "seed": 42},
  "split": {"source": "split/split.json", "mode": "random", "fingerprint": "…",
             "counts": {"train": 280, "val": 80, "test": 40},
             "train_rows": 280, "val_rows": 80},
  "summary": {"method": "gain_iterative", "threshold": 0.05, "n_features": 4,
               "target_range": [4, 4], "in_target_range": true,
               "disposition": "converged", "message": "",
               "attempts": [{"threshold": 0.05, "n_features": 4, "in_range": true, "n_rounds": 2}],
               "gain_shape": {"shape": "头部集中", "top10pct_gain_share": 0.82, "…": "…"},
               "rounds": [{"round": 1, "n_input": 4, "n_features": 3, "cut": 1.2,
                            "valid_auc": 0.82, "note": ""}]},
  "features": ["f_signal", "f_weak", "f_missing"],
  "n_selected": 3,
  "gain_top": [{"feature": "f_signal", "total_gain": 120.5, "retained": true}],
  "warnings": []
}
```

`selected_features.txt`：一行一个特征，**UTF-8 无 BOM**，末尾带换行（供模型层直接读）。

`disposition` 取值：`converged` / `truncated_high` / `relaxed_converged` /
`relaxed_truncated` / `relaxed_failed`；非 `converged` 时 `message` 会说明「拍板值 → 生效值」
并进 `warnings`（不静默改参数）。

---

## 7. 错误码

| 错误码 | 触发场景 | hint 要点 |
|---|---|---|
| `spec_invalid` | `spec` 未知键/未知列/未知策略/策略与列类型不符/区间非法/空 spec | 按 §3.1–3.2 修正；消息里给出可用策略与示例 |
| `expression_invalid` | 字段表达式非法（含 `__import__`/属性访问/赋值/函数调用）、引用不存在的列、子集重叠 | 按 §4.2 示例改写（禁 `eval`） |
| `features_empty` | 初筛无数值特征 / 精筛候选池为空 | 先编码或用 `config.features`/`config.candidates` 指定 |
| `config_invalid` | 阈值越界、比例不合法、分层列不存在、初筛无 train/val 可切、精筛 val 为空 | 检查 `config`；先 `dataset_split` |
| `dependency_missing` | 精筛缺 `lightgbm` | `pip install lightgbm` |
| `analysis_failed` | 数据本身问题（全空、好坏样本缺失、无法切分） | 修数据后重试 |
| `project_dir_not_found` / `manifest_not_found` / `encoding_undecodable` … | 沿用契约 §1.2 | 同契约 |

**失败必须可见**：任何跳过/降级（PSI 跳过、parquet 不可用、WOE 全量拟合、行被删除、
入模矩阵仍不可用、初筛回退全量）都写进 `warnings` 与对应产物的 `warnings` 字段。

---

## 8. 给模型层的接口约定

### 8.1 训练数据（`preprocess/processed.*`）

* 路径：`preprocess/processed.parquet`（首选）或 `preprocess/processed.csv`（无 pyarrow 时）；
  以 `preprocess/summary.json.processed_path` / `processed_format` 为准。
* 列：原始列名的保留 + 变换结果：
  * 被 `missing` 填充的列：已填充；
  * 被 `encode` 的列：原列被替换（onehot ⇒ 多列 `<col>_<取值>`；ordinal/woe ⇒ 数值列；drop ⇒ 消失）；
  * 被 `cap` 的列：同名列，数值已截断；
  * 被 `binning` 的列：同名列，值 = 箱编码 `0..k-1`，缺失 `-1`（原值可从 CSV 回溯）；
  * 目标列与 id/time/split 角色列**原样保留**（模型层需要目标与时间列）。
* 行序与原始 CSV 一致（除 `missing.drop` 删行）；若发生删行，`summary.rows_in/rows_out` 不同。
* 读入前先看 `summary.model_readiness`：`lr_ready=false` 时 LR 不可用（GBDT 可继续），
  原因在 `non_numeric_columns` / `remaining_missing`。
* 预处理规格复现：`fill.json`/`encode.json`（含 ordinal 映射、onehot 取值清单、WOE 表）/
  `outlier.json` / `manual_bins.json`（切点 + 箱编码含义）—— 推理期必须用同一套表。

### 8.2 划分（`split/split.json`）

* 掩码**不落逐行文件**，模型层用同一份 spec 复算：

```python
from risklab_mdl.core.split import SplitSpec, split_frame
doc = json.loads((proj / "split" / "split.json").read_text(encoding="utf-8"))
result = split_frame(raw_frame, SplitSpec.from_config(doc["spec"]), target=doc["target"]["column"])
assert result.fingerprint == doc["fingerprint"], "掩码与划分落盘不一致：行序/数据已变，拒绝静默使用"
train, val, test = result.frames(raw_frame)
```

* `raw_frame` 必须是**划分时的那份原始数据**（同一 CSV、同一行序）。
  若先做 `preprocess_apply` 且用了 `missing.drop`（删行），掩码会错位 ——
  此时应「先划分、再对子集分别预处理」，或重跑 `dataset_split`。
* 精筛/调参口径固定为 train + val（`core.split`）；test 只在评价阶段使用，不参与筛选。

### 8.3 初筛 / 精筛产物（`feature_selection/`）

* `prescreen.json.survived`：初筛存活特征清单（模型层训练时的默认候选）。
* `prescreen.json.features[].metrics`：`missing_rate` / `same_value_rate` / `iv` / `psi`
  （多分类另有 `iv_by_class`），可用于报表与复核。
* `refine.json.features` 与 `selected_features.txt`：精筛最终入模特征（两者一致；
  `selected_features.txt` 为 UTF-8 无 BOM、一行一个）。
* `refine.json.summary`：逐轮过程（`rounds`）、放宽尝试（`attempts`）、
  gain 分布形态（`gain_shape`）—— 报表直接引用，不要重算。

---

## 9. 验收实测（本机 Python 3.11.9 / pandas 3.1.1 / lightgbm 4.7.0）

| 验收项 | 命令 | 结果 |
|---|---|---|
| ① 同 seed 划分可复现 | `pytest tests/test_split.py -q` | 通过：两次调用 `fingerprint` 相同；core 纯函数两次掩码逐行 `array_equal`；random 比例误差 < 1%；stratified 各类占比偏差 < 2% |
| ② 危险表达式拒绝 | `pytest tests/test_split_expr.py -q` | 通过：`__import__` / `.` 属性 / `=` 赋值 / 函数调用 / 分号 / 括号未闭合 全部 `expression_invalid`，消息含位置与示例；`in/&/\|/比较/not in/null` 通过 |
| ③ 手动分箱逐箱一致 | `pytest tests/test_preprocess_apply.py -q -k binning` | 通过：箱表计数与独立 `pd.cut` 复算一致；`__MISSING__` 独立箱；缺失编码 `-1` |
| ④ LR 就绪 | `pytest tests/test_preprocess_apply.py -q -k lr_ready` | 通过：fill + onehot/ordinal + cap 后 `lr_ready=true`、入模矩阵全数值无 NaN |
| ⑤ 检查分类 | `pytest tests/test_preprocess_check.py -q` | 通过：文本列/缺失列进 `must_fix`，同值/极端值/高基数进 `suggested` |
| ⑥ 全量测试 + 工具集 | `pytest tests -q` / `pytest tests/test_server_smoke.py -q` | 全绿；`tools/list` = 20 个工具 |
