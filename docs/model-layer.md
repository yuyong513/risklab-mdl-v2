# 模型层：适配器 / 标签编码 / 数据解析 / 调参 / run / 评价 / 确认 / 报告

> 状态：**已实现（2.0.0）**。契约来源：[工具契约](tool-contract.md) §3.4；口径来源：[`core/metrics.py`](../src/risklab_mdl/core/metrics.py)、[口径-metrics.md](口径-metrics.md)。
> 代码位置：`src/risklab_mdl/models/`（能力表 / 适配器 / 标签 / 数据 / 评价 / 训练装配）、
> `src/risklab_mdl/runs/store.py`（run 目录与产物）、`src/risklab_mdl/tools/model.py` 与
> `src/risklab_mdl/tools/artifact.py`（8 个工具）。

## 0. 一句话

**5 算法 × 2 任务 = 10 格**，每格都能 `model_train` → `model_evaluate` → `model_confirm`，
产物落在 `runs/<run_id>/`，模型包装进 schema 2.0 并能独立打分；**1.0 旧包仍可加载打分**。

## 1. 算法 × 任务能力表（唯一真源：`models/catalog.py`）

| 算法 | 库 | 依赖 | binary | multiclass | 早停 | NaN 原生 |
|---|---|---|---|---|---|---|
| `dt` | scikit-learn `DecisionTreeClassifier` | 无 | ✅ | ✅ | ❌ 固定树 | ❌ 需填充 |
| `lr` | scikit-learn `LogisticRegression` | 无 | ✅ | ✅ | ❌ 固定迭代 | ❌ 需填充 |
| `gbdt` | scikit-learn `GradientBoostingClassifier` | 无 | ✅ | ✅ | ❌ 固定迭代 | ❌ 需填充 |
| `xgb` | `xgboost.XGBClassifier` | `pip install xgboost` | ✅ | ✅ | ✅ val 早停 | ✅ |
| `lgb` | `lightgbm.LGBMClassifier` | `pip install lightgbm` | ✅ | ✅ | ✅ val 早停 | ✅ |

* 每格的 `default_params` / `default_space` 只在 `models/catalog.py` 定义一次；
  `model_catalog`（回显）、`model_train`（默认参数）、`model_tune`（搜索空间）都从这里取 ——
  **预览与执行不会漂移**（`tests/test_tune.py::test_execute_writes_run_and_matches_preview_space` 断言）。
* `objective` / `metric` / `num_class` **由 task 决定**：`config.params` 里的同名项会被丢弃并写入
  `warnings`（"objective 由任务类型决定"）。理由：binary 用 `multi:softprob` 会让概率列数不对，
  而这种错误**能跑出数**，一路带到报表。
* `config.params` 的其余键名必须是该格 `default_params` 里的参数：未知键名 → **`config_invalid`**。
  以前不校验，拼错的键（如 `max_depht`）会一路进到 `DecisionTreeClassifier(**params)` 抛
  `TypeError`，被外层映射成 `analysis_failed`（用户看到"程序出错了"，真因只是参数名拼错）。
* 没有 `num_class` 的 sklearn 栈：`dt/lr/gbdt` 的类别数由 `y` 决定，适配器在 fit 后校验
  `classes_ == range(num_class)`，不一致直接报错（防止标签编码与矩阵错位）。

### 1.1 早停口径

| 算法 | 有 val | 无 val |
|---|---|---|
| `xgb` | `early_stopping_rounds` + `eval_set=[(X_val,y_val)]`，`eval_metric=auc/mlogloss` | 不传早停参数，固定 `n_estimators` |
| `lgb` | `callbacks=[early_stopping(esr)]` + `eval_X/eval_y`（4.7）或 `eval_set`（旧版） | 固定轮数 |
| `dt/lr/gbdt` | 固定迭代（不用「内部早停」：那等于偷拿训练集当验证集） | 固定迭代 |

「早停没真正生效」（`best_iteration` 达到上限）会进 `warnings`，不静默。

### 1.2 缺失值与收敛（失败必须可见）

* `xgb` / `lgb`：**不填** NaN（填了会抹掉缺失信息）。
* `dt` / `gbdt` / `lr`：用 **train 的中位数**填充，并把填充值写进模型包 `fill_values`
  （推理期用同一批值，不用推理数据的中位数），同时产出 warnings：
  「已用 train 中位数填充 N 列（共 M 个缺失单元格）」。
* `lr` 未收敛（sklearn `ConvergenceWarning`）→ 捕获并转成 warnings：
  「训练未收敛，系数不可信」。宽表（数百列 + 未缩放）上**很可能出现**，这是真实边界。

## 2. 标签编码与 `bad_class`

### 2.1 编码

* 原始类别 → `0..k-1`；**顺序由 `class_sort_key` 决定**（数字按数值、文本按字典序，
  避免 `"10" < "2"` 这类字典序陷阱）。
* 二分类固定排成 `[good, bad]` ⇒ `predict_proba[:, 1]` 恒为「坏」的概率。
  **但代码不假设这一点**：`predict_score` 与指标计算都按 `bad_index()` 取列 ——
  若某个包的 `classes` 顺序被改成 `[bad, good]`，硬编码第 2 列会让分数方向反转
  （KS/AUC 照样算得出来，只是把最好的客户排到了最差）。
* 编码映射（`classes` / `class_map` / `bad_class`）**随模型包落盘**（schema 2.0），
  推理期不用再猜顺序。
* 标签缺失/未知取值：**剔除该行**并告警（"N 行因标签缺失/未知被剔除"）——
  绝不补 0（补 0 = 把未知判成好客户）。

### 2.2 `bad_class` 判定规则（三条，顺序即优先级）

1. **显式映射**：`target.labels` 中值属于 `bad / 1 / 坏 / 违约 / 正例 / positive`
   的那个原始类别（与 `data.loader.binary_bad_label` 同一套 token）；
2. **二分类缺省**：原始取值 `1`（按字符串比较，兼容 `"1"`）；再没有就取排序后第二类；
3. **多分类缺省**：排序后最后一类（风险等级通常 `0 < 1 < 2` 递增）。

`bad_class` 的用途：① 二分类取正类概率；② 多分类决定「风险分数用哪一列」的概率
（`predict_score` 的 `predict_proba` 列）；③ 月度 PSI 的分数口径。

### 2.3 打分输出列（`predict_score`）

| 任务 | 列 |
|---|---|
| binary | `predict_proba`（坏类概率）、`fico_score` —— **与 schema 1.0 完全一致** |
| multiclass | `predict_proba`（= `bad_class` 那一列）、`fico_score`、`proba_<class>`（逐类一行一列，顺序 = `classes`） |

多分类时 `predict_proba` 与 `proba_<bad_class>` **恒等**（`tests/test_models_matrix.py` 断言）。

## 3. 模型包 schema 2.0（`core/package.py` 的最小扩展）

新增字段（全部有默认值，写入时一并落盘）：

| 字段 | 含义 |
|---|---|
| `task` | `binary` / `multiclass` |
| `num_class` | 类别数（binary 必须为 2） |
| `classes` | 原始类别列表（顺序 = 编码顺序） |
| `bad_class` | 坏类原始取值 |
| `algorithm_family` | `sklearn` / `xgboost` / `lightgbm` |
| `preprocess_spec` | 预处理来源快照（`processed_path` / 数据来源），供报告与排障 |
| `training` | 样本量、训练耗时等诊断信息 |

* `PACKAGE_SCHEMA_VERSION = "2.0"`；`SUPPORTED_SCHEMA_VERSIONS = ("1.0", "2.0")`。
* **1.0 兼容**：`predict_score` 按 `schema_version` 分支，1.0 走旧口径（取 `predict_proba[:,1]`）；
  真·旧 pickle（连 `task` 等属性都不存在）在 `load_package` 里按 dataclass 默认值**补齐字段**，
  否则访问 `pkg.classes` 会 `AttributeError`（症状=「旧模型包打不开」）。
* 版本闸门不变：不支持的版本（如 `0.9`）仍拒绝加载。
* `digest()` 纳入 `task/classes/bad_class`：同一份模型换一套标签编码就是**另一个包**。

## 4. 数据解析（`models/data.py`）

### 4.1 数据来源优先级

| 顺序 | 来源 | 说明 |
|---|---|---|
| ① | `preprocess/processed.parquet` | 预处理产物，口径一致 |
| ② | `preprocess/processed.csv` | 缺 pyarrow / parquet 写失败时的降级产物 |
| ③ | 原始 CSV + `config.preprocess` | 用预处理层的 `validate_spec` + `apply_plan` 在**内存**执行（未落盘，warnings 明示） |
| ④ | 原始 CSV | 文本/缺失原样保留，由模型层告警或报错 |

**为什么不读 `preprocess/*.json` 重放**：`fill.json` / `encode.json` 记录的是**执行结果**
（每列做了什么、产出哪些列），ordinal 的取值映射与 WOE 表都没随产物落盘。拿结果反推规格
会得到「看起来一样、实际不同」的处理链 —— 典型静默降级。只有 `preprocess/*.json` 而无
`processed.*` 时，显式告警并退回原始数据。

### 4.2 特征优先级

`config.features` → `feature_selection/selected_features.txt` → 全部数值列
（自动排除 target / id / time / split 角色列）。

* 自动选择时非数值列**排除 + 告警**（列出前 10 个）："N 个非数值列未入模（需先 preprocess_apply 编码）"；
* 显式 `config.features` 里点名非数值列 → **`config_invalid` 报错**（用户要的列不能用，
  继续跑等于偷偷换特征）。

### 4.3 划分优先级

`split/split.json` 的 `spec` 复算 mask → 无则内部 **80/20 分层**划分（不落盘）。

* 内部划分告警："未使用 split/split.json ⇒ 内部 80/20 分层（seed），该划分不落盘，
  跨 run 复现请显式调用 `dataset_split`"；
* `split.json` 的行数与当前数据行数不一致（预处理 drop 过行）⇒ mask 无法对齐 ⇒
  同样退回内部划分 + 告警；
* val 为空 ⇒ 早停不可用（固定轮数），告警明示。

## 5. run 结构与 manifest

```
runs/<run_id>/
  config.json      本次训练全部入参（target/algorithm/task/features/feature_source/
                   data_source/split/seed/classes/bad_class/sample_sizes/input_config，
                   调参 run 另含 tuning 块）
  params.json      最终超参（默认值 + 覆盖，或调参最优参数）
  metrics.json     train/val/test 指标（+ 多分类 per_class/macro、psi、sample_sizes、warnings）
  model.pkl        模型包（schema 2.0）
  importance.csv   特征重要性（降序；含 gain/importance 与 share 列）
  evaluation/      metrics_by_subset.csv、per_class_metrics.csv（多分类）、
                   ks_table.csv、lift_table.csv、psi_by_month.csv、ks_curve.png、lift_chart.png
  tuning/          space.json、trials.csv（仅 model_tune 的 execute）
```

* `manifest.runs` **只追加**、走 `update_manifest`（锁内读改写）；同 run_id 重复时替换该条
  （不会出现两条同号记录）。
* 状态机：`trained` →（`model_evaluate`）`evaluated` →（`model_confirm`）`confirmed`；
  `model_tune` 的 run 记 `tuned`。状态只前进。
  `update_run` 内置守卫：`confirmed` 之后重跑 `model_evaluate` 只更新指标摘要，
  **不会**把状态打回 `evaluated`（否则 manifest 里"已确认的 run"看起来像没确认过）。
* `manifest.stages.train.done=true` 在首次写入 run 时置位。

## 6. 调参（`model_tune`）

* `mode=preview`：返回 `space` / `space_table`（参数名/类型/范围）/ `expected_trials` /
  `objective` / `tune_on`，**不训练**（测试断言 manifest.runs 不变）；
* `mode=execute`：`core.tuning.tune_model`（hyperopt TPE）→ 落一个 `tuned` run；
  `best_params` 会**真正用于最终模型**（`params.json` 断言）；
* 目标函数：`ks*w_ks + auc*w_auc + lift*w_lift − gap*w_gap`，
  `gap = max(0, train_auc − val_auc)`；只有 `gap > gap_threshold`（默认 **0.05**，
  预览文案的数值与 `core.tuning.TuningConfig.gap_threshold` 同源）才全额惩罚，阈值以内不扣分；
  权重默认 `ks=.5/auc=.4/lift=0/gap=.1`；轨迹表 `衰减` 列是**实际参与扣分的**
  `train_cv_gap_penalized`，`原始衰减` 列是未裁剪的 gap；
* `tune_on` **只允许 `val`**：`oot` / `train` / `cv` 一律 `config_invalid`。
  用 OOT 选超参属于广义数据穿越（在测试集上调参 ⇒ OOT 指标系统性偏高、线上达不到）；
* 多分类的 `cv_ks` / `cv_auc` 取 **macro**（逐类 OVR 平均），`cv_lift` 亦为 macro。

## 7. 评价（`model_evaluate`）

* 二分类单指标全部复用 `core.metrics`（KS/AUC/Gini/LIFT），**不在模型层重算**；
* 多分类：`core.metrics.calc_multiclass_metrics`（OVR 逐类 + `macro` 算术平均 + `weighted` 参考），
  `metrics.json` 每个子集块内同时给 `per_class` / `macro`，另有顶层 `multi.<subset>` 同内容；
  顶层 `ks/auc/gini/lift` 取 **macro**（"一个 run 一个主指标"）；
* 子集标签只有一类时：`status="skipped"` + 原因（**不给 0 冒充满分**）；
* **KS/LIFT 表与图只吃 0/1 标签**：多分类先 `y_bin = (y == bad_index).astype(int)`，
  表里多一列 `bad_class`（写明哪一类算坏），`metrics.json` 里 `ks_lift_tables` 记录
  `{subset, bad_class, bad_index, label_scheme}`。`ks_table/lift_table` 收到非 0/1 标签会
  直接 `DataError` —— 否则 `y==1` 只认第二类，坏类整批被当好人，表/图与
  `per_class/macro` 口径错位且看不出来；
* **PSI（分数按月，基期合并池口径）**：
  - 切点在**基期池**上拟合、各月共用（各月各自切点 ⇒ PSI 恒为 0，看起来"很稳"，实际没测到）；
  - 基期池默认 = 最早一个月，可用 `config.psi_base_months=["2026-01","2026-02"]` 合并多个月；
  - 指定的基期月**不存在** ⇒ `config_invalid`（**不静默回落到最早月**：报表会显示一个你没指定的基期）；
  - 基期覆盖**全部**月份 ⇒ `psi.enabled=false` + `reason="…没有对比月…"`（不是 PSI=0，也不抛异常）；
  - 零占比按 `1e-6` 处理（与 `core.psi` 同口径），不让 ln(0) 变 inf；
  - 无时间列 / 只有 1 个月 ⇒ `psi.enabled=false` + `reason`，**不伪造数值**；
  - 时间列来源：`config.time_column` → `config.month_column` → manifest `roles.time` → 现场推断；
* 评价表与图（Agg + 中文字体 fallback）：缺 matplotlib / 字体时降级为**仅出表**并告警。

## 8. 确认闸门（`model_confirm`）

* 默认阈值：**KS ≥ 0.25**、**PSI ≤ 0.10**（`thresholds={ks_min, psi_max}` 可覆盖）；
* KS 取值：val 优先，val 指标被跳过时回落 train（`ks_source` 显式返回）；
* 不达标 → `error.code=threshold_not_met`，`data` 给
  `ks / ks_gap / psi / psi_gap / failed_checks / checks`（差距值，不给模糊结论）；
* 达标 → `final/model.pkl`（从 run 的 model.pkl 复制）+ `final/confirm.json`
  （阈值、指标、确认时间、来源 run）+ `manifest.confirmed_run` + run 状态 `confirmed`；
* **PSI 未计算时只校验 KS**，并在 warnings 明示「不代表跨期稳定」——不跳过告警；
* val 样本量 < 100 → 附「可靠性有限」告警（阈值判断的统计波动大）；
* `metrics.json` 缺 PSI 结果时会先自动重算评价（warnings 明示），不跳过闸门项目。

## 9. 报告（`report_build` / `run_compare` / `run_list`）

* **只读产物，不重算**：数字一律来自 `runs/*/metrics.json` 与 `importance.csv` ——
  报告自己再算一遍会出现"报告 KS 0.31、确认时 0.24"的对不上，且说不清哪份对；
* `reports/结果汇总.xlsx`（openpyxl，四个 **非空** sheet）：
  - `概览`：项目/数据/目标/角色/阶段完成度/run 数/确认 run/生成时间；
  - `模型指标`：逐 run × 子集 × 指标（多分类另加 macro 行）；
  - `run对比`：逐 run 指标 + `best` 标记（主指标 KS）；
  - `特征重要性`：最优/确认 run 的 TOP20 特征；
* `reports/过程报告.md`：阶段完成度、预处理摘要、划分、特征筛选、模型指标表、确认状态、构建提示；
* `reports/run_compare.csv`：对比表（缺省全部 run）；`data.best` 按主指标 KS 标出；
  没有任何 run 有可用 KS 时 `best=null`（不默认挑第一个）。

## 10. 已知边界（不隐藏）

1. **宽表 + 极少坏样本必然过拟合**：`sample_2000`（2000 行 / 691 列 / 35 坏）上 lgb 典型
   表现是 train AUC≈0.997、val AUC≈0.63（本机实测）。指标本身"有限且算出来了"，
   但它不是可用模型 —— 上线前应做特征筛选（`feature_refine`）与降复杂度。
2. **`gbdt`（sklearn）明显慢**：单线程逐树串行，691 列上约 9s/100 棵；宽表优先 xgb/lgb。
3. **LR 在宽表上常不收敛**（未做缩放/筛选时）：会给 warnings，但不会自动换 solver/缩放 ——
   自动缩放会改变系数口径。
4. **LIFT 可以等于 0**：坏率 1.75% 时尾部 10% 可能一个坏样本都没抓到 ⇒ LIFT=0 是真实值，
   不是 bug；报表侧请看 `lift_seg_size` 与 `is_reliable`。
5. **不做 OOT / 外部验证**：`tune_on` 结构性禁 OOT；`model_evaluate` 只算 train/val/test。
6. **不做回归 / 深度学习 / 服务化**：超出本项目范围（契约冻结 20 工具）。
7. **内存中的 `config.preprocess` 不落盘**：要让后续 run 复用同一口径，请先
   `preprocess_apply` 生成 `processed.*`。
8. **PSI 基期默认单月**：多个月合并池需显式 `config.psi_base_months`；
   指定的月份不存在 ⇒ `config_invalid`（不静默回落），基期覆盖全部月份 ⇒ `psi.enabled=false`；
   跨年/跨口径的月份合并要业务确认（工具不替用户判断）。
9. **`model_confirm` 的 final/model.pkl 是 run 模型的副本**：不做重新训练、不改标尺。

## 11. 测试映射（验收 → 断言）

| 验收条目 | 测试文件 |
|---|---|
| ① 10 格矩阵 / 多分类 per_class+macro / 打包打分 | `tests/test_models_matrix.py`、`tests/test_package_v2.py` |
| ①(补) 多分类 KS/LIFT 表按 bad_class 二值化、bad 计数 == 坏类样本数 | `tests/test_models_matrix.py::test_multiclass_ks_lift_tables_use_bad_class` |
| ② sample_2000 二分类 5 算法（耗时 ≤600s） | `tests/test_models_on_sample.py`（实测 ≈23s 训练+评价） |
| ③ 错误路径（算法/任务/依赖/文本缺失/未知参数） | `tests/test_models_errors.py`、`tests/test_tune.py` |
| ④ run 追踪（run_list / run_compare / 并存 / 报告 / 状态只前进） | `tests/test_runs.py` |
| ⑤ 确认闸门（不达标差距值 / 达标 final+manifest / PSI 闸门） | `tests/test_confirm.py` |
| ⑥ 调参 preview/execute 空间一致、落 run、禁 OOT、gap 阈值 | `tests/test_tune.py` |
| ⑦ PSI 边界（基期覆盖全部月份 / 指定基期不存在） | `tests/test_models_matrix.py::EvaluatePsiEdgeTests` |
| ⑦ 工具总数 20、全量 pytest 全绿 | `tests/test_server_smoke.py`、全量 `python -m pytest tests -q` |

## 12. 给 skill 层的接口约定

* 工具总数 **20**：`server_probe` + 数据与特征 6 个 + 预处理/划分/筛选 5 个 + 模型与过程 8 个
  （`model_catalog / model_train / model_tune / model_evaluate / run_list / run_compare /
  model_confirm / report_build`）。
* skill 需要展示的结果结构：
  - `model_catalog.data.catalog.matrix`（10 行，含 `default_params` / `default_space`）；
  - `model_train.data.metrics.{train,val,test}`（`ks/auc/gini/lift/n/bad/status/reason`）、
    `data.multi.<subset>.{per_class,macro}`（多分类）、`data.run_id`；
  - `model_tune.data.{space,space_table,expected_trials,objective,tune_on}`（preview）；
  - `model_evaluate.data.{metrics,psi.{enabled,max_psi,max_month,base_months},tables,charts}`；
  - `run_list.data.runs[]`（`run_id/algorithm/task/status/metrics/confirmed/has_model`）；
  - `run_compare.data.{best,rows,compare_csv}`；
  - `model_confirm.data.{confirmed,thresholds,ks,psi,ks_gap,psi_gap,checks,failed_checks,final_model}`；
  - 失败按 `error.code` 分支呈现：`task_algorithm_mismatch` / `dependency_missing` /
    `threshold_not_met`（带差距值）/ `run_not_found` / `config_invalid` / `features_empty`。
* 三态：缺目标字段时 `model_train` 返回 `needs_user_input`（`data.question` 需 skill 呈现）。
