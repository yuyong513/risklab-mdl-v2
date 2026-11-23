# 产物目录与解读（`mdl-<csv名>/`）

> 布局真源：`src/risklab_mdl/artifacts/layout.py`；目录规范见 `docs/tool-contract.md` §4。
> 本文按**本机实测一次完整流程**（`sample_2000.csv`，2000 行 × 696 列）列清单与解读：
> 全部产物 **744 个文件**，其中 `feature_analysis/binning/*.json` 占 691 个。

## 1. 顶层结构

```text
<CSV 同目录>/mdl-<CSV 文件名去扩展名>/          # 例：D:/work/mdl-sample_2000
  manifest.json                 # 项目状态（唯一真源，schema risklab-mdl.manifest/1）
  data_analysis/                # 环节 2 画像
  feature_analysis/             # 环节 3 分箱 + WOE/IV
  feature_selection/            # 环节 5–6 初筛 / 精筛
  preprocess/                   # 环节 6 预处理落地 + 处理后数据
  split/                        # 环节 7 划分
  runs/<run_id>/                # 环节 8–9 每个训练 run（YYYYmmddTHHMMSS）
  final/                        # 环节 9 确认后的最终模型
  reports/                      # 环节 9 报告
```

约定：

- 目录**不随流程清零**：重新跑某一步只覆盖该步产物，历史 `runs/` 保留；
- 同一个 CSV 再次 `project_init` 会**复用**同一目录（不会新建 `mdl-sample_2000-2`）；
- 产物路径在工具返回的 `artifacts` 里一律是**相对项目目录的 POSIX 路径**，
  给用户时拼成绝对路径（Windows 上正反斜杠都能打开，但展示用正斜杠更稳）。

## 2. `manifest.json`

```jsonc
{
  "schema": "risklab-mdl.manifest/1",
  "project": {
    "csv_path": "D:/work/sample_2000.csv", "stem": "sample_2000",
    "created_at": "…", "updated_at": "…",
    "target": {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}},
    "roles": {"id": "sample_id", "time": "month", "split": "split", "features": ["…数值特征列…"]}
  },
  "stages": {                    # 固定 6 个键
    "data_analysis": {"done": true, "path": "…", "summary": "…"},
    "feature_analysis": {"done": true},
    "feature_selection": {"done": true},
    "preprocess": {"done": true},
    "split": {"done": true},
    "train": {"done": true}
  },
  "runs": [{"run_id": "20261116T213542", "algorithm": "lgb", "task": "binary",
            "status": "evaluated", "metrics": {"ks": 0.402, "auc": 0.658, "subset": "val"},
            "paths": {}, "created_at": "…"}],
  "confirmed_run": "20261116T213542"
}
```

- `stages` 6 个键固定；`runs[].status ∈ {trained, tuned, evaluated, confirmed, failed}`，**只前进**；
- `confirmed_run` 非 null 时必须命中某个 `runs[].run_id`；
- 写 manifest 走「原子替换 + 项目级文件锁」，并发写会 `lock_timeout`/`manifest_conflict`，**不静默覆盖**；
- 想快速判断"做到哪一步了"：读 `stages` + `runs.length` + `confirmed_run`，比翻目录可靠。

## 3. `data_analysis/`（`dataset_profile`）

| 文件 | 内容 | 怎么读 |
|---|---|---|
| `profile.json` | 全量画像：`shape` / `target` / `target_distribution` / `feature_types` / `missing_summary` / `columns` / `monthly` / `warnings` | 结构化取数用这个（agent 展示以 `data` 为准，两者同源） |
| `describe.csv` | 数值列描述统计：`column,count,mean,std,min,25%,50%,75%,max,missing_rate,role` | 判断量纲/异常量级 |
| `target_distribution.csv` | 标签分布：`label,label_name,count,share` | 坏率 = `1` 那一行的 `share` |
| `charts/target_distribution.png` | 标签分布柱图 | 直接给用户看 |
| `charts/missing_rate_top.png` | 缺失率 TOP20（全无缺失时不出图） | 缺失治理优先级 |
| `charts/feature_distributions.png` | 抽样特征取值分布（最多 6 个） | 看是否存在多峰/哨兵值 |
| `charts/monthly_bad_rate.png` | 按月坏率（**有时间列才有**） | 看坏率是否随时间漂移 |
| `index.html` | 单文件汇总（图片用 `charts/xxx.png` 相对路径） | 双击即可看，**无外链**（`http://`/`https://` 数量为 0） |

## 4. `feature_analysis/`（`feature_analyze` + `feature_charts`）

| 文件 | 内容 | 怎么读 |
|---|---|---|
| `iv_table.csv` | 逐特征：`feature,iv,n_bins,n_bins_fitted,missing_rate,same_value_rate,method,task,skipped,reason[,iv_<类>]` | **行数 = 待分析特征数（含被跳过的）**；跳过的行 `skipped=True` 且 `reason` 非空；按 `iv` 降序、跳过行排末尾；多分类 `iv` 是各类 OVR 的 macro 平均，逐类看 `iv_0/iv_1/…` |
| `woe_detail.csv` | 逐箱：`feature,class,bin,bin_index,bin_label,total,bad,good,bad_rate,share,woe,iv_contribution,smoothed,note` | 缺失箱 `bin=-1`；`note` 里说明稀疏合并 |
| `binning/<feature>.json` | 可复用分箱规格 `BinSpec`（schema `risklab-mdl.binspec/1`）：`method/edges/n_bins/requested_bins/missing_as_bin/min_bin_count/…` | 想复现同一套箱：`BinSpec(method="business_cut", edges=…)`；**被跳过的特征没有这个文件** |
| `charts/iv_distribution.png` | IV 降序柱图 + 0.02 / 0.1 / 0.3 阈值线 | 一眼看区分度分布 |
| `charts/woe_<feature>.png` | IV TOP10 的单特征箱图（WOE / 坏率随箱变化） | 检查单调性与缺失箱位置 |
| `charts/correlation_heatmap.png` | 相关性热力图（`feature_charts`，≤15 个特征） | 找强相关对（初筛会剔除低 IV 的那个） |
| `index.html` | 单文件汇总 | 同上 |

注：`n_bins`（落表有效箱数）≠ `n_bins_fitted`（声明的有效箱数）⇒ 做过稀疏箱合并，`woe_detail.note` 有说明；
"箱数因样本量不足被下调"会进 `warnings`，此时 IV 与历史口径不可比。

## 5. `feature_selection/`（`feature_prescreen` + `feature_refine`）

| 文件 | 内容 | 怎么读 |
|---|---|---|
| `prescreen.json` | 五道筛子结果：`thresholds`（生效阈值回显）/ `data_scope`（用了 train 还是全量 + `fingerprint`）/ `psi`（基期）/ `summary.stage_dropped`（逐阶段淘汰数）/ `survived[]` / `features[]`（逐特征 `survived`、`dropped_stage`、`reason`、`metrics`）/ `non_numeric[]` | `features[]` 覆盖**所有**候选，`dropped_stage` 是**第一道**淘汰它的筛子；多分类另有 `iv_by_class` |
| `refine.json` | 精筛：`candidates` / `n_candidates` / `features` / `n_selected` / `disposition` / `attempts` / `gain_shape` / `summary.rounds`（逐轮）/ `split`（用的是 train+val 还是兜底） | `disposition != converged` 要解释（截断 / 放宽阈值） |
| `selected_features.txt` | 一行一个特征，UTF-8 **无 BOM**、末尾换行 | `model_train` 默认直接读它；不要手工加空行/BOM |

## 6. `preprocess/`（`preprocess_check` + `preprocess_apply`）

| 文件 | 内容 | 怎么读 |
|---|---|---|
| `check.json` | `must_fix[]` / `suggested[]`（人话消息）+ `by_feature.<列>.issues[]`（`code/level/message/suggestion`）+ `summary.rules`（生效规则） | `must_fix` 空才代表入模矩阵可直接用；每列都在 `by_feature` 里（`issues: []` = 查过没问题） |
| `fill.json` | 缺失处理：`entries[{column,strategy,value,missing_before,missing_after,rows_removed}]` | `rows_removed > 0` ⇒ 用了 `drop`，行数变了 |
| `encode.json` | 编码：`entries[{column,strategy,output_columns,mapping|onehot_values|woe_table,fit_scope,missing_preserved}]` | **推理期必须复用这里的映射**（ordinal 映射表 / onehot 取值清单 / WOE 表） |
| `outlier.json` | 盖帽：`entries[{column,strategy,lower,upper,fit_scope,n_capped,min_before,max_before,min_after,max_after}]` | `fit_scope` 说明是在哪些数据上拟合的 |
| `manual_bins.json` | 手动分箱：`entries[{column,edges,interior_edges,n_bins,missing_as_bin,missing_count,bin_counts,bin_labels,values_are:"bin_code"}]` | 分箱**替换原列**：processed 里该列变成箱编码 `0..k-1`（缺失 `-1`），原值可从 CSV 回溯 |
| `summary.json` | 执行摘要：`order`（固定 `missing→encode→cap→binning`）、行/列数、`model_readiness`、`processed_path`/`processed_format` | 判断能不能训 LR 看这里（`lr_ready`） |
| `processed.parquet` | 处理后数据（首选） | 训练/划分都优先读它 |
| `processed.csv` | 降级产物（**未装 pyarrow 时**，`warnings` 会说明） | 内容等价，只是慢；`pip install pyarrow` 可换回 parquet |

`model_readiness` 三件套：`lr_ready`（矩阵全数值无缺失）、`non_numeric_columns`、
`remaining_missing`。`lr_ready=false` 时 LR 不可用，GBDT/XGB/LGB 仍可跑。

## 7. `split/split.json`（`dataset_split`）

| 字段 | 含义 |
|---|---|
| `mode` | `none` / `random` / `stratified` / `field` |
| `rows` / `counts` | 总行数与三个子集行数 |
| `ratios` / `seed` | 目标比例与实际种子 |
| `spec` | 复算 mask 所需的完整规格（含 `field.column` / 三段表达式 / `time_column`） |
| `fingerprint` | 三段掩码按位打包的 SHA256 前 16 位：**同 seed 同输入必相同**，训练时用它核对 |
| `label_distribution` | 各子集的标签分布（核对分层偏差） |
| `description` | 人话（`mode=none` 时明说"未划分：全量口径，指标不代表样本外表现"） |

注意：掩码**不落逐行文件**，是按 `spec` 复算的。若先 `preprocess_apply(missing.drop)` 删过行，
行数变化会让 mask 错位 ⇒ 训练层会告警并退回内部划分；此时应"先划分再对子集预处理"或重跑划分。

## 8. `runs/<run_id>/`（`model_train` / `model_tune` / `model_evaluate`）

`run_id` 形如 `20261116T213542`（`^\d{8}T\d{6}$`，同秒冲突按秒递进，`mkdir(exist_ok=False)` 原子占位）。

| 文件 | 内容 |
|---|---|
| `config.json` | 本次训练全部入参：`target/algorithm/task/features/feature_source/data_source/split/seed/classes/bad_class/sample_sizes/input_config`；调参 run 另含 `tuning` 块 |
| `params.json` | 最终超参（默认值 + 覆盖，或调参最优参数） |
| `metrics.json` | `train/val/test` 指标（`ks/auc/gini/lift/n/bad/good/status/reason`）+ 多分类 `per_class`/`macro` + `psi` + `sample_sizes` + `warnings` + `ks_lift_tables`（记录 bad_class / label_scheme） |
| `model.pkl` | 模型包（schema 2.0：`task/num_class/classes/bad_class/algorithm_family/preprocess_spec/training`） |
| `importance.csv` | 特征重要性（降序，含 `gain`/`importance` + `share`） |
| `evaluation/metrics_by_subset.csv` | 逐子集指标长表 |
| `evaluation/ks_table.csv` | KS 表（等频 10 箱；多分类多一列 `bad_class`，`Σbad` = 坏类样本数） |
| `evaluation/lift_table.csv` | LIFT 表（同上口径） |
| `evaluation/psi_by_month.csv` | 逐月 PSI（`month / psi / n`），**要与 `metrics.json.psi.base_months` 一起读** |
| `evaluation/ks_curve.png` / `lift_chart.png` | KS / LIFT 曲线（缺字体或缺 matplotlib 时降级为仅出表并告警） |
| `evaluation/per_class_metrics.csv` | 多分类逐类指标（二分类不产出） |
| `tuning/space.json` / `tuning/trials.csv` | 仅 `model_tune(mode=execute)` 产出：搜索空间与试验轨迹（`衰减` 列是**实际参与扣分**的 `train_cv_gap_penalized`） |

## 9. `final/`（`model_confirm`）

| 文件 | 内容 |
|---|---|
| `final/model.pkl` | run 模型的**副本**（不重训、不改标尺） |
| `final/confirm.json` | 阈值（`ks_min`/`psi_max`）+ 指标（`ks`/`ks_source`/`psi`/`auc`/`gini`/`lift`/`n`/`bad`）+ 来源 `run_id` + `confirmed_at` |

只会在闸门通过时出现。`manifest.confirmed_run` 同步指向该 run，run 状态变 `confirmed`。

## 10. `reports/`（`run_compare` + `report_build`）

| 文件 | 内容 | 怎么读 |
|---|---|---|
| `结果汇总.xlsx` | 四个**非空** sheet：`概览`（21 行）/ `模型指标`（4 行）/ `run对比`（2 行）/ `特征重要性`（21 行） | 只读产物、不重算；数字与 `runs/*/metrics.json` 同源 |
| `过程报告.md` | 阶段完成度、预处理摘要、划分、特征筛选、模型指标表、确认状态、构建提示 | 给业务/评审看的叙述版 |
| `run_compare.csv` | 逐 run 指标对比 + `best` 标记（主指标 KS） | 缺省对比全部 run；`best=null` 表示没有任何 run 有可用 KS |

## 11. 一次完整流程的产物数量预期（本机实测）

| 目录 | 文件数 | 说明 |
|---|---|---|
| `data_analysis/` | 8 | 4 张表/页（`profile.json`、`describe.csv`、`target_distribution.csv`、`index.html`）+ 4 张图 |
| `feature_analysis/` | 705 + 图 | 3 张表/页 + **691** 个 `binning/*.json`；图：`feature_analyze` 出 IV 分布 1 张 + TOP10 箱图 10 张 = 11 张，再跑 `feature_charts` 会多出相关性热力图与按 `max_features` 补齐的箱图（实测 14 张） |
| `feature_selection/` | 3 | `prescreen.json`、`refine.json`、`selected_features.txt` |
| `preprocess/` | 7 | `check/fill/encode/outlier/manual_bins/summary` + `processed.parquet|processed.csv` |
| `split/` | 1 | `split.json` |
| `runs/<run_id>/` | 5 + 6~8 | 5 个主文件 + `evaluation/`（`metrics_by_subset.csv`、`ks_table.csv`、`lift_table.csv`、`psi_by_month.csv`、`ks_curve.png`、`lift_chart.png`，多分类另有 `per_class_metrics.csv`） |
| `final/` | 2 | `model.pkl`、`confirm.json` |
| `reports/` | 3 | `结果汇总.xlsx`、`过程报告.md`、`run_compare.csv` |
| 合计 | **≈ 741–744** | 与是否补跑 `feature_charts`、是否跑 `model_tune` 有关 |

如果用户问"产物怎么这么多"：691 个是每个可分析特征的 `BinSpec`（供复现分箱），
不是日志；`reports/` 与 `final/` 才是对外交付物。
