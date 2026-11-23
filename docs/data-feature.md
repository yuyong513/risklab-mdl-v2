# 数据与特征分析层

> 范围：建模流程环节 1–3 —— 读数据 → 样本画像 → 特征分析（分箱 / WOE / IV）+ 图表。
> 契约来源：[tool-contract.md](./tool-contract.md) §1 / §3.1 / §3.2 / §4；口径见 [口径-metrics.md](./口径-metrics.md)。
> 代码位置：`src/risklab_mdl/{data,features,charts,tools}/`。
> 上游依赖：`core/binning.py` / `core/binning_dt.py`；图表 backend 与中文字体链由 `doctor` 检查。

本层**只做到特征分析为止**：不写预处理、不划分样本、不训练模型（分属预处理 / 划分 / 模型层）。

---

## 1. 工具用法（6 个）

| 工具 | 入参 | 关键产物 | 说明 |
|---|---|---|---|
| `project_init` | `csv_path` | `manifest.json` + 8 个子目录 | 已存在则复用，不覆盖历史产物 |
| `dataset_load` | `csv_path` | 更新 `manifest.project.roles` | 编码探测 + dtype/角色**推断不阻断** |
| `dataset_columns` | `project_dir` | 无（返回 `data.columns`） | 供用户确认/改角色；带 `nunique`/`missing_rate` |
| `dataset_profile` | `project_dir`, `config?` | `data_analysis/*` | 目标分布 / 缺失 / 描述统计 / 类型构成 / 月份交叉 |
| `feature_analyze` | `project_dir`, `config` | `feature_analysis/*` | 逐特征分箱 + WOE/IV（含多分类 OVR） |
| `feature_charts` | `project_dir`, `config?` | `feature_analysis/charts/*` | 只重画图，**不重算 IV** |

所有工具都吃 `project_dir`（除 `project_init` / `dataset_load` 吃 `csv_path`），
返回统一三态：`{ok, status, data, artifacts, warnings, error}`。

### 1.1 典型调用链

```python
project_init(csv_path=r"D:\work\sample_2000.csv")
dataset_load(csv_path=r"D:\work\sample_2000.csv")
dataset_profile(
    project_dir=r"D:\work\mdl-sample_2000",
    config={"target": {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}}},
)
feature_analyze(
    project_dir=r"D:\work\mdl-sample_2000",
    config={"method": "quantile", "bins": 8,
            "target": {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}}},
)
feature_charts(project_dir=r"D:\work\mdl-sample_2000", config={"max_features": 15})
```

`config.target` 确认后会被写进 `manifest.project.target`，后续步骤可以不再重复传。

### 1.2 `dataset_profile` 的 `config`

| 键 | 默认 | 说明 |
|---|---|---|
| `target` | 取 manifest | `{column, task, labels?}`；缺省且 manifest 也没有 ⇒ `needs_user_input` |
| `max_features_for_charts` | 30 | 特征数超过它时只在 `warnings` 说明「图只画前几个」 |
| `sample_rows` | 不抽样 | 传入即抽样（会显式告警口径不同） |
| `roles` | 取 manifest | 只接受 `id` / `time` / `split`，用于标注与月份交叉 |

### 1.3 `feature_analyze` 的 `config`

| 键 | 默认 | 说明 |
|---|---|---|
| `method` | `quantile` | `quantile` / `equal_width` / `dt` / `custom` |
| `bins` | 8 | 请求箱数（`dt` 时是 `max_bins` 上限）；样本量不足会由 core 下调 |
| `features` | 全部（除 role） | 显式列表；不传时按 manifest 角色排除 id/time/split/target |
| `target` | 取 manifest | 同上 |
| `edges` | — | `method=custom` 时 `{feature: [切点…]}`；单特征也可传裸列表 |
| `max_features_for_charts` | 30 | 只影响 `warnings` 措辞与箱图数量（箱图固定画 IV TOP 10） |

---

## 2. 产物说明

```
mdl-<stem>/
  data_analysis/
    profile.json            # 全量画像（shape/target/target_distribution/feature_types/
                            #   missing_summary/columns/monthly/warnings）
    describe.csv            # 数值列描述统计：column,count,mean,std,min,25%,50%,75%,max,missing_rate,role
    target_distribution.csv # 标签分布：label,label_name,count,share
    charts/
      target_distribution.png
      missing_rate_top.png          # 缺失率 TOP 20（全无缺失时不出图）
      feature_distributions.png     # 抽样特征取值分布（最多 6 个）
      monthly_bad_rate.png          # 有时间列时才出
    index.html              # 单文件汇总，图片用 charts/xxx.png 相对路径
  feature_analysis/
    iv_table.csv            # 逐特征：feature,iv,n_bins,n_bins_fitted,missing_rate,
                            #   same_value_rate,method,task,skipped,reason[,iv_<类>]
    woe_detail.csv          # 逐箱：feature,class,bin,bin_index,bin_label,total,bad,good,
                            #   bad_rate,share,woe,iv_contribution,smoothed,note
    binning/<feature>.json  # 可复用 BinSpec（下游直接读）
    charts/
      iv_distribution.png           # IV 降序 + 阈值线 0.02 / 0.1 / 0.3
      woe_<feature>.png             # IV TOP 10 的单特征箱图
      correlation_heatmap.png       # feature_charts 产出（抽样 ≤15 个特征）
    index.html
```

### 2.1 `iv_table.csv` 约定

* 行数 = 待分析特征数（**含被跳过的**），跳过的行 `skipped=True` 且 `reason` 非空 ——
  「一个特征去哪了」永远能在表里查到，不必翻日志。
* `iv`：二分类是该特征的 IV；多分类是各类 OVR IV 的 **macro 平均**（逐类 IV 另开
  `iv_0` / `iv_1` / … 列）。
* `n_bins`：稀疏合并后实际落表的有效箱数；`n_bins_fitted`：`BinSpec` 声明的有效箱数。
  两者不等 ⇒ core 做过稀疏箱合并（`woe_detail.note` 有说明）。
* 排序：按 `iv` 降序，`skipped` 行排末尾。

### 2.2 `binning/<feature>.json` 约定（下游复用）

```json
{
  "schema": "risklab-mdl.binspec/1",
  "generated_at": "2026-…",
  "feature": "f_0001",
  "method": "quantile",
  "task": "binary",
  "target_column": "y",
  "labels": {"1": "bad", "0": "good"},
  "iv": 0.708451,
  "per_class": {
    "binary": {
      "method": "quantile", "n_bins": 3, "n_valid_bins": 3, "requested_bins": 8,
      "bins_downgraded": true, "edges": [...], "missing_as_bin": true,
      "min_bin_count": 50, "min_bin_pct": 0.005,
      "sparse_strategy": "merge_adjacent", "woe_smoothing": 0.5, "fit_on_months": []
    }
  },
  "notes": []
}
```

* 二分类只有 `per_class.binary`；多分类是 `per_class["0"] / ["1"] / …`（one-vs-rest 各一套）。
* `edges` 可直接构造 `BinSpec(method="business_cut", edges=…)` 复现（`missing_as_bin=True`，
  缺失落 `MISSING_BIN=-1`）。
* **被跳过的特征不产出该文件**（没有可复用的分箱规格）。

---

## 3. 口径与已知边界

1. **只在 train 上拟合的语义**：`core.binning.fit_bins` / `core.binning_dt.fit_bins_dt`
   都要求「切点只在训练集上拟合」，但 core 不做切分。`dataset_split` 尚未运行，
   因此 `feature_analyze` 把**传入数据当训练集**，并在 `warnings` 显式写明；
   划分完成后再跑一次即可得到无穿越的分箱。
2. **缺失单独成箱**：特征缺失**不丢行**（只按标签缺失筛行），缺失落 `bin=-1`，
   参与 WOE/IV —— 否则「没有记录」这类信息会被静默抹掉。
3. **多分类不压成二分类**：给逐类 OVR IV + macro，不给「坏率」（那会伪造指标）。
4. **常量列 / 全缺失列 / 非数值列**一律进 `skipped` 并给原因，不产出一张 IV=0 的假读数。
5. **箱数会被下调**：样本量撑不起请求箱数时 `core.binning` 主动下调（`bins_downgraded`），
   `warnings` 会列出被下调的特征 —— 此时 IV 与历史口径不可比。
6. **无时间列就跳过时间维度**：不出按月图、不做月份交叉，`warnings` 明示，**不伪造**。
7. **低坏率/小样本会告警**：坏率 < 5% 或样本 < 500 时提示 IV/KS 读数不稳。

---

## 4. 宽表性能策略（sample_2000：2000 行 × 696 列）

实测（本机 Python 3.11.9 / pandas 3.1.1，见 §6）：全链路 project_init → profile →
feature_analyze → feature_charts **≈ 22–28s**（预算 120s；feature_analyze 占大头）。策略如下：

| 手段 | 做法 |
|---|---|
| 只读需要的列 | `feature_analyze` 先读表头校验列名，再 `usecols=<target + 目标特征>`；`feature_charts` 只为相关性图读抽样列 |
| 图表封顶 | 画像最多 6 个特征的分布图；箱图只画 IV TOP 10；相关性热力图 ≤ 15 个特征；`max_features_for_charts` 默认 30，超限只写 `warnings` |
| 向量化 | 分箱/变换/分箱表全部走 `core.binning`（pandas/numpy 向量化），逐箱统计用 `groupby`，不写 Python 行循环 |
| 图少而准 | 每张图一个 `Figure`，`savefig` 后立即 `plt.close(fig)`（`figure.max_open_warning=0`，避免内存堆积） |
| 抽样可选项 | `config.sample_rows` 可强制抽样，抽样必告警（口径不同） |

---

## 5. 错误码（契约 §1.2 之上补充的两个）

| 错误码 | 触发 | hint 要点 |
|---|---|---|
| `config_invalid` | `config` 结构/取值非法：方法非法、`features` 里有不存在的列、`method=custom` 缺 `edges`、`edges` 形状不对 | 按 §1.2/§1.3 检查字段名与取值 |
| `analysis_failed` | 分析过程的非预期失败（含缺前置产物、core 抛出的 DataError） | 先补齐前置步骤 / 修数据后重试 |

其余沿用契约既有代码：`path_not_readable` / `project_dir_not_found` /
`manifest_not_found` / `encoding_undecodable` / `dependency_missing` / `features_empty`。
**任何降级都进 `warnings`**（跳图、跳月份、抽样、箱数下调、缺字体、只画前 N 个…），
不允许「空结果 + ok:true」。

---

## 6. 参考实测（Windows / Python 3.11，2026-11）

| 项 | 结果 |
|---|---|
| `project_init` → `dataset_profile` → `feature_analyze` → `feature_charts`（sample_2000 复制到临时目录） | 全链路 **27.8s**（分项 3.3s / 19.2s / 4.2s，≤120s 预算） |
| `data_analysis/` 产物 | `profile.json` / `describe.csv` / `target_distribution.csv` / `index.html` + **4 张 PNG** |
| `feature_analysis/iv_table.csv` 行数 | **692**（691 个可分析 + 1 个文本列 `f_0423` 跳过并给原因） |
| `feature_analysis/woe_detail.csv` | 4707 行（逐特征逐箱，含缺失箱 `bin=-1`） |
| `feature_analysis/binning/*.json` | 691 个可复用 BinSpec |
| HTML 离线 | `data_analysis/index.html`、`feature_analysis/index.html` 内 **0 个** `http://`/`https://`，引用的本地 PNG 全部存在 |
| 图片规格 | Agg backend，PNG，**120 dpi**（从 PNG `pHYs` 块读数校验） |

复现命令：

```powershell
cd risklab-mdl-v2
python -m pytest tests -q                                   # 全绿
python -m pytest tests/test_tool_contract.py -q -k sample_2000   # 真实数据全链路（含 120s 预算断言）
```

---

## 7. 给下游的接口约定

### 7.1 `loader`（`risklab_mdl.data.loader`）

* `read_csv_auto(path, *, usecols=None, nrows=None) -> CsvRead(path, encoding, frame, warnings)`：
  唯一读表入口（编码探测 + 重试链），**不要**自己 `pd.read_csv(encoding=...)`。
* `infer_roles(frame) -> (roles, reasons)`：`roles = {id, time, split, target_candidates}`，
  `reasons` 逐角色给人话理由。**推断不阻断**，目标字段由 `config.target` 确认。
* `column_infos(frame, roles=None) -> [{name, dtype, pandas_dtype, role, nunique,
  missing_rate, same_value_rate, is_high_cardinality}]`（`dtype` 是语义类型：
  `bool/integer/float/datetime/categorical/text/empty`）。
* `normalize_target(target) -> {column, task, labels}`；`resolve_task` / `binary_bad_label`
  / `monthly_key`（`YYYY-MM` 归一）可复用。
* `is_numeric_series(series)` / `as_numeric(series) -> (coerced, ratio)`：统一的「能不能当数值」判据。
* `to_jsonable(obj)`：所有落盘 JSON 前过一遍（numpy → Python，NaN/Inf → null）。

### 7.2 `profile`（`risklab_mdl.data.profile`）

* `compute_profile(frame, *, target, roles=None, max_features_for_charts=30, sample_rows=None)
  -> ProfileResult`：纯计算，字段见 `to_payload()` / `data_block()`；
  `describe_frame` / `target_distribution_frame` / `monthly_frame` 是待落盘表格。
* `ProfileResult.feature_types = {"counts": {...}, "columns": {numeric: [...], categorical: [...], text: [...], datetime: [...], high_cardinality: [...]}}`
  —— 预处理的「哪些列需要编码/填充」直接取这里。

### 7.3 feature 分析（`risklab_mdl.features.analysis`）

* `analyze_features(frame, *, target, method, bins, features=None, roles=None,
  custom_edges=None, max_features_for_charts=30) -> FeatureAnalysisResult`
* `FeatureAnalysisResult.iv_table` / `.woe_detail`（列定义见 `IV_TABLE_COLUMNS` /
  `WOE_DETAIL_COLUMNS`）/ `.bin_specs` / `.skipped` / `.iv_top(n)` / `.data_block()`。
* 复用分箱：`bin_specs[feature]["per_class"][class]["edges"]` →
  `BinSpec(method="business_cut", edges=tuple(...), missing_as_bin=True)` → `transform_bins`。
* `FeatureAnalysisResult` 同时供 `feature_prescreen` 挑「IV 低 / 缺失高 / 同值高」的候选，
  但**筛选阈值与判定不在这里做**。
