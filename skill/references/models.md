# 模型与参数（dt / lr / gbdt / xgb / lgb）

> 唯一真源：`src/risklab_mdl/models/catalog.py`，通过 `model_catalog` 工具回显（`data.catalog`）。
> 本文的默认参数与默认调参空间是**本机实测回显值**，与 `model_train` / `model_tune` 取同一份定义
> ——预览与执行不会漂移。
> 能力表：**5 算法 × 2 任务 = 10 格**，每格都支持 binary 与 multiclass。

## 1. 一览

| 算法 | 库 | 额外依赖 | binary | multiclass | 原生 NaN | 早停 |
|---|---|---|---|---|---|---|
| `dt` | sklearn `DecisionTreeClassifier` | 无 | ✅ | ✅ | ❌ 需填充 | ❌ 固定树 |
| `lr` | sklearn `LogisticRegression` | 无 | ✅ | ✅ | ❌ 需填充 | ❌ 固定迭代 |
| `gbdt` | sklearn `GradientBoostingClassifier` | 无 | ✅ | ✅ | ❌ 需填充 | ❌ 固定迭代 |
| `xgb` | `xgboost.XGBClassifier` | `pip install xgboost` | ✅ | ✅ | ✅ | ✅ val 早停 |
| `lgb` | `lightgbm.LGBMClassifier` | `pip install lightgbm` | ✅ | ✅ | ✅ | ✅ val 早停 |

`model_catalog` 的 `data.available.<algo>.installed` 是**运行期真实探测**结果；
`installed=false` 时不要推荐该算法，直接把 `install` 里的 pip 命令给用户。

## 2. 使用场景（选型建议）

| 场景 | 推荐 | 理由 / 注意 |
|---|---|---|
| 宽表（数百列）+ 低坏率，要效果 | `lgb` → `xgb` | 原生处理缺失、快；宽表 + 极少坏样本**必然过拟合**，务必先筛特征、降复杂度 |
| 需要向业务解释（系数 / WOE 方向） | `lr` | 系数可解释；**硬门槛是不接受缺失**（模型层用 train 中位数填充并告警）；宽表未缩放时大概率不收敛（会告警「训练未收敛，系数不可信」） |
| 规则基线 / 汇报用一棵树 | `dt` | 规则可直接读出；方差大，不建议直接上线 |
| 不想装额外依赖（离线/受限环境） | `gbdt` | 无需 xgboost/lightgbm；但**单线程逐树串行**，691 列上约 9 s/100 棵，宽表明显慢 |
| 多分类风险等级（0/1/2/3） | `lgb` / `xgb` | 指标看 `macro`；`bad_class` 默认排序后最后一类 |
| 样本量极小（< 500 行） | `lr` 或 `dt` | 树模型在几百行上极不稳定；先把特征数压到 10 个以内 |

不建议的用法：

- `gbdt` 跑 691 列宽表只为"省依赖"——慢且无优势；
- `lr` 直接吃未处理的宽表——不收敛 + 系数不稳，先 `preprocess_apply`（`lr_ready=true`）再训；
- 「算法不支持该任务」时想靠改 `task` 绕过——会返回 `task_algorithm_mismatch`，**不降级**。

## 3. 默认参数（`default_params`）

| 算法 | binary / multiclass 共用默认参数 |
|---|---|
| `dt` | `criterion=gini, max_depth=6, min_samples_leaf=20, min_samples_split=10, random_state=42` |
| `lr` | `C=1.0, solver=lbfgs, max_iter=2000, tol=0.0001, fit_intercept=true` |
| `gbdt` | `n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, min_samples_leaf=20, random_state=42` |
| `xgb` | `n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_weight=1.0, reg_lambda=1.0, tree_method=hist, random_state=42` |
| `lgb` | `n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbosity=-1, random_state=42` |

覆盖方式：`model_train` 的 `config.params`（**键名必须是上表里的参数**，拼错 ⇒ `config_invalid`；
`objective` / `metric` / `num_class` 由任务决定，写进 `params` 会被丢弃并告警）。

## 4. 默认调参空间（`default_space`，`model_tune` 使用）

| 算法 | 空间 |
|---|---|
| `dt` | `max_depth: int[3,10]`、`min_samples_leaf: int[5,100]`、`criterion: choice{gini,entropy}` |
| `lr` | `C: loguniform[0.01,10.0]`、`fit_intercept: choice{true,false}` |
| `gbdt` | `n_estimators: int[50,200]`、`learning_rate: loguniform[0.01,0.2]`、`max_depth: int[2,5]`、`subsample: uniform[0.6,1.0]`、`min_samples_leaf: int[5,50]` |
| `xgb` | `n_estimators: int[100,500]`、`learning_rate: loguniform[0.01,0.3]`、`max_depth: int[3,8]`、`subsample: uniform[0.6,1.0]`、`colsample_bytree: uniform[0.6,1.0]`、`reg_lambda: loguniform[0.1,10.0]` |
| `lgb` | `num_leaves: int[15,127]`、`learning_rate: loguniform[0.01,0.3]`、`max_depth: int[3,10]`、`min_child_samples: int[5,50]`、`subsample: uniform[0.6,1.0]`、`colsample_bytree: uniform[0.6,1.0]`、`reg_lambda: loguniform[0.01,10.0]` |

- `type` 语义：`int` 取整数区间、`uniform` 连续均匀、`loguniform` 对数均匀、`choice` 枚举；
- 默认 `trials=20`；`mode=preview` 只回显空间与预计试验数（**不训练**），`mode=execute` 才落 run；
- 自定义 `space` 里出现未知参数名 ⇒ `config_invalid`（防止拼错后静默失效）；
- 目标函数：`ks*w_ks + auc*w_auc + lift*w_lift − gap*w_gap`，
  默认权重 `ks=0.5 / auc=0.4 / lift=0.0 / gap=0.1`，
  `gap = max(0, train_auc − val_auc)`，**只有 `gap > 0.05` 才全额惩罚**；
- `tune_on` **只允许 `val`**：传 `oot`/`train`/`cv` 一律 `config_invalid`（在测试集上调参属于广义数据穿越）。

## 5. 二分类注意点

1. **标签编码**：原始类别 → `0..k-1`，二分类固定排成 `[good, bad]`，`predict_proba[:, 1]` 是坏的概率；
   但指标与打分按 `bad_index()` 取列，不硬编码列号。
2. **坏类判定**（`bad_class`）：① `labels` 里值为 `bad/1/坏/违约/正例/positive` 的键；
   ② 否则原始取值 `1`；③ 再没有取排序后第二类。**这是必须跟用户确认的**（见 SKILL.md §4）。
3. **标签缺失/未知取值**：剔除该行并告警，**绝不补 0**（补 0 = 把未知判成好客户）。
4. **缺失**：`xgb`/`lgb` 不填 NaN（保留缺失信息）；`dt`/`gbdt`/`lr` 用 **train 中位数**填充，
   填充值随模型包落盘（`fill_values`），推理期用同一批值。
5. **早停**：只有提供 val 时 `xgb`/`lgb` 才早停；「早停没真正生效」（`best_iteration` 达到上限）会进 `warnings`。

## 6. 多分类注意点

1. `objective`/`metric`/`num_class` 由 `task=multiclass` 决定（`multi:softprob` / `multi_logloss` 等）；
2. `sklearn` 栈（`dt/lr/gbdt`）无 `num_class`，类别数由 `y` 决定，适配器 fit 后校验
   `classes_ == range(num_class)`，不一致直接报错（防止标签编码与矩阵错位）；
3. 指标按 **OVR + macro**（见 [metrics.md](metrics.md) §7）：顶层 `ks/auc/gini/lift` 取 `macro`；
4. `predict_score` 输出 `predict_proba`（= `bad_class` 那列）、`fico_score`、`proba_<class>`（逐类一列）；
5. `bad_class` 默认排序后最后一类（风险等级通常 `0<1<2` 递增）；**要跟用户确认**；
6. 某类在某子集只有一类标签 ⇒ 该类 `status=skipped` + `reason`，不参与 macro，**不给 0 分**；
7. 多分类的分数 PSI 用 `bad_class` 那一列概率（不是 argmax 分布，也不是逐类各算一个）。

## 7. 训练/调参的耗时经验（本机实测）

| 操作 | 规模 | 耗时 |
|---|---|---|
| `model_train`（`lgb`, 60 特征, 1200 行） | 二分类 | ≈ 0.3 s（`fit_seconds` 0.084） |
| `model_evaluate` | 同上 | ≈ 0.5 s |
| `model_tune` preview | 任意 | ≈ 0.1 s（不训练） |
| `model_tune` execute | `trials=20` | 约为单次训练的 20 倍（宽表先降到 10–20 次） |
| `feature_refine`（`lgb` gain 迭代） | 449 候选 / 3 轮 | ≈ 3.4 s |

## 8. 已知边界（不隐藏，来自实现与实测）

1. 宽表 + 极少坏样本必然过拟合：`sample_2000`（2000 行 / 691 列 / 35 坏）上 `lgb` 典型表现是
   train AUC ≈ 0.998、val AUC ≈ 0.63–0.66；指标"算出来了"≠"模型可用"。
2. `gbdt` 在宽表上慢（单线程），优先 `xgb`/`lgb`。
3. `lr` 宽表未缩放时常不收敛 ⇒ 告警但不自动换 solver/缩放（自动缩放会改变系数口径）。
4. LIFT 可以等于 0：坏率 1.75% 时尾部 10% 可能一个坏样本都没抓到，这是真实值。
5. 不做 OOT / 外部验证、不做回归 / 深度学习 / 服务化。
6. 内存中的 `config.preprocess` **不落盘**：要让后续 run 复用同一口径，先 `preprocess_apply`。
7. `model_confirm` 写的 `final/model.pkl` 是 run 模型的**副本**：不重训、不改标尺。
