# 指标口径说明（v1.0）

> 实现位置：`src/risklab_mdl/core/metrics.py`、`binning.py`、`psi.py`、
> `src/risklab_mdl/models/evaluate.py`（子集/多分类/月度 PSI/KS·LIFT 表）。
> 状态：**v1.0 定稿**（随 2.0.0 发布）。初版内容已全部并入正文；补齐三节
> 「阈值/确认标准」（§8）、「分月评估」（§9）、「多分类 LIFT 业务解读 + PSI 口径决策」（§10），
> 本文件所有口径均已锁定，没有悬置项。
> 本文件写的是**代码实际怎么算**，不是教科书愿望；与代码不一致以代码为准并回头改本文件。

## 0. 记号

| 记号 | 含义 |
|---|---|
| `y` | 标签，二分类取 `1=坏 / 0=好`；多分类取任意离散标签 |
| `s` | 风险分数（概率或评分），**单调**、可含并列 |
| `p` | 正类概率（二分类为 `P(y=1)`） |
| `N_bad / N_good` | 坏/好样本数 |

方向约定：**统一用 `score_direction(y, s)` 判定**（`AUC ≥ 0.5` ⇒ `higher_score_higher_risk`），
不允许各报表自己翻转方向；LIFT 的尾部选择依赖这个判定。

## 1. AUC

定义：`P(s_bad > s_good) + 0.5·P(s_bad = s_good)`（Mann–Whitney U）。

实现：平均秩法（`_average_rank`），并列取平均秩；非有限分数行**剔除**后再算。
与 `sklearn.metrics.roc_auc_score` 数值一致（偏差 < 1e-12，有单测）。

- 取值 `[0,1]`；`AUC < 0.5` 表示分数方向与标签相反（**不自动翻转**）。
- 显著性：`auc_significance(auc, n_bad, n_good)` 用 Hanley–McNeil 标准误给 z 值；
  `|AUC-0.5|` 不显著时不应告警（小样本 30 坏/90 好、AUC=0.4833 的 z≈0.27）。

## 2. Gini

`Gini = 2·AUC − 1`，取值 `[-1, 1]`；`Gini(−s) = −Gini(s)`。
**不做独立实现**，避免与 AUC 口径漂移。

## 3. KS

定义：`KS = max_t |F_bad(t) − F_good(t)|`，`F·` 为按分数阈值累计的分布函数。

实现要点：

1. **按分数值整组推进**：先 `np.unique` 分数、按值降序累计，**同分客户不被拆开**。
   拆开会让同分客户分属累计曲线的两个位置 ⇒ KS 偏大（且样本顺序敏感）。
2. 取**绝对值** ⇒ 方向反转时仍有正确量级；方向信息由 `score_direction` 单独提供。
3. 取值 `[0,1]`。

## 4. LIFT@pct

定义：`LIFT@pct = 段内坏样本率 / 整体坏样本率`。

实现（`calc_lift`）：

- 方向：`direction="auto"` ⇒ 走 `score_direction`；`risk = s` 或 `−s`。
- 取尾部：按风险降序（或 `worst=False` 时升序）累计**权重**，取累计权重首次
  ≥ `pct` 的位置（保证段内占比 ≥ pct），段内至少 1 个样本。
- 支持 `weight`（拒绝负值与缺失）。
- **可靠性**：`is_reliable = seg_size ≥ min_seg_size`（默认 30）；
  不可靠时**不做静默修正**，只置标记由报表提示 —— 「小样本 LIFT」不能当结论。
- `pct` 必须落在 `(0, 1]`。

## 5. IV（分箱）

定义：对每个箱 `i`，

```
bad_pct_i  = (bad_i  + δ) / (N_bad  + δ·K)
good_pct_i = (good_i + δ) / (N_good + δ·K)
WOE_i      = ln(bad_pct_i / good_pct_i)
IV         = Σ_i (bad_pct_i − good_pct_i) · WOE_i
```

- `K` = 箱数（含缺失箱），`δ` = `woe_smoothing`（默认 0.5，归一化平滑；
  `sparse_strategy="keep"` 时 δ=0，用于复现旧口径对比）。
- 稀疏箱**先合并**（`min_bin_count` / `min_bin_pct`，缺失箱不参与合并），
  合并下限 `max(2, total // min_bin_count)` 保证**不塌成单箱**
  （塌成单箱时 IV 恒为 0，会让初筛把所有特征判成无区分度）。
- `WOE是否截断` 标记 `±inf` 箱（δ=0 且某箱全好/全坏时），提醒读数不可信。
- **IV 读数依赖箱数**，因此 `BinSpec.requested_bins` 与实际箱数必须随报表透出。

### 5.1 DT（决策树）分箱的 IV

`fit_bins_dt` 输出 `BinSpec(method="business_cut", edges=…)`，直接喂
`bin_table_processed` 得 IV —— **口径与等频完全一致**（同一套 WOE/IV 公式）。
差异只在切点来源（树切点 vs 分位切点）与后处理（最小箱/单调化合并）。

## 6. PSI（跨期稳定性）

定义：`PSI = Σ_i (test_i − base_i) · ln(test_i / base_i)`，`i` 为**同一套箱**。

实现（`bin_psi` / `screen_psi`）三条硬口径：

1. **缺失箱参与对比**（缺失是独立箱 `-1`，不是哨兵值 `-99`）；
2. **test 中不存在的箱按占比 0 计**（跳过会低估 PSI）；
3. 分母统一加 `eps=1e-6` 防除零。

跨月：`screen_psi` 对每个「基期 vs 对比月」配对算 PSI，按 `base_mode` 汇总：

| `base_mode` | 基期 | 汇总成单个读数 |
|---|---|---|
| `specified` | 指定月份（**多个 ⇒ 合并成一个池**） | 取各对比月的**最大值** |
| `rolling` | 指定月份序列（逐月滚动） | 同上，逐月推进 |
| `multi_base` | ≥2 个基期月（合并池） | 同上 + `multi_base_tolerance` 约束 |

★ **多基期是真·合并池**：切点在池上拟合、对比月与池比较。
detail 的「基期」列写「基期池(A+B)」，**不得**写成 `A→B`（旧实现的假标签问题）。
单基期时标签就是该月，文案逐字不变（有移植测试守护数值与标签）。

`psi_by_month(scores, months, ...)`：对**分数**做月度 PSI，切点在基期上拟合且各月共用
（各月各自切点 ⇒ 每箱占比≈1/n ⇒ PSI 恒 0，看起来稳定但什么都没测到）。

## 7. 多分类口径（OVR + macro）

### 7.1 OVR（one-vs-rest）

对每个类别 `k`：把 `y == k` 视为**坏**、其余视为**好**，用第 `k` 列概率当分数，算：

```
AUC_k = calc_auc(1{y=k}, p_k)
KS_k  = calc_ks(1{y=k}, p_k)
Gini_k= 2·AUC_k − 1
LIFT_k= calc_lift(1{y=k}, p_k, pct=0.10)
```

与 `sklearn.roc_auc_score(multi_class="ovr", average="macro")` 同口径。

### 7.2 macro / weighted

- `macro`：**有效类的算术平均**（`AUC/KS/Gini/LIFT` 各自汇总）—— 进主结论；
- `weighted`：按各类样本数加权平均 —— 仅参考（大类主导，会掩盖小类失效）。

### 7.3 跳过与可靠性（必须可见，不静默）

| 情形 | 处理 |
|---|---|
| 某类在当前样本**无正例/无负例** | 该类 `status="skipped"` + `reason`；**不参与 macro**；`notes` 写明「N 个类别被跳过」 |
| 某类样本数 < `min_class_support`（默认 30） | 仍给指标，但 `is_reliable=False` + `notes` 告警 |
| 所有类都被跳过 | `macro` 全为 `NaN`（JSON 输出为 `null`），`notes` 说明原因 |

### 7.4 输入形态

`calc_multiclass_metrics(y_true, y_score, labels=None, ...)`：

- `y_score` 接受 `(n, k)` 概率矩阵（列顺序 = `labels`）或 `{label: 一维分数}` 映射；
- 一维数组 / 行数不符 / 列数 ≠ 类别数 / 含 NaN ⇒ `DataError`（不猜）；
- `labels` 缺省取 `np.unique(y_true)`（升序）。

### 7.5 KS / LIFT 分箱表的标签口径（多分类必须二值化）

`evaluation/ks_table.csv` / `lift_table.csv` 与 `ks_curve.png` / `lift_chart.png` 的
「坏」= **`bad_class`**：多分类先

```
y_bin = (y == bad_index).astype(int)     # 1 = 坏类（bad_class），0 = 其余类合并
score = proba[:, bad_index]              # 与 per_class[bad_class] 同一列概率
```

* 表里多一列 `bad_class`（写出坏类的原始取值），`metrics.json` 的
  `ks_lift_tables = {subset, bad_class, bad_index, label_scheme}` 记录同一件事；
* 表内 `Σbad` **必须等于坏类样本数**（不是「编码为 1 的那类」）；
* 因此 `max(ks_table.ks)` 与 `per_class[bad_class].ks`（OVR，即同一个二分类问题）
  同源同向，前者因等频 10 箱而 ≤ 后者（少若干个阈值点）；
* `ks_table` / `lift_table` 收到非 0/1 标签直接 `DataError` ——
  把 `0..k-1` 直接喂进去，`y==1` 只认第二类，坏类整批被算成好样本，
  表/图看着有数却与 `per_class/macro` 完全错位。这是**必须报错**的用法，不是可降级项。

## 8. 确认阈值与确认标准（`model_confirm`）

* 默认阈值：**KS ≥ 0.25**、**PSI ≤ 0.10**（`thresholds={ks_min, psi_max}` 可覆盖）；
  `ks_min` 必须落在 `[0,1]`、`psi_max` 不能为负，越界 ⇒ `config_invalid`；
* KS 取值：**val 优先**；val 指标被跳过（该子集只有一类标签）时回落 train，
  `ks_source` 显式返回 `val` / `train`，不混用；
* PSI 取值：`psi.max_psi`（各对比月 PSI 的**最大值**，不是平均、不是 overall）；
  `psi.enabled=false`（无时间列 / 只有 1 个月 / 基期覆盖全部月份）⇒ 该项跳过，
  但 warnings 必须写「PSI 未计算，只校验 KS —— 不代表跨期稳定」；
* 判定：`ks ≥ ks_min` 且（PSI 不可用 或 `psi ≤ psi_max`）才通过；
* 不通过 ⇒ `error.code=threshold_not_met`，`data` 给
  `ks / ks_gap / psi / psi_gap / failed_checks / checks`（**给差距值**，不给"差不多就行"）；
* 通过 ⇒ `final/model.pkl`（run 模型副本）+ `final/confirm.json`
  （阈值 + 指标 + 确认时间 + 来源 run）+ `manifest.confirmed_run`，run 状态 `confirmed`；
* val 样本量 < 100 ⇒ 附「可靠性有限」告警（阈值判断的统计波动大），但仍允许确认；
* **多分类的确认口径**：主指标取 **macro KS**（顶层 `ks`），阈值与二分类同一套
  （不引入第二套阈值）；`bad_class` 的 OVR KS 只作诊断读数 —— 用它做闸门会奖励
  「只学会识别坏类、其余类全错」的模型。多分类的 LIFT 读数见 §10.1。

## 9. 分月评估（区分度与稳定性分开看）

* **KS/AUC 与 PSI 不可互替**：前者是同月内的排序能力，后者是分数分布的跨期漂移；
  一个月内 KS 很高但分布整体右移，照样是上线风险；
* 时间列来源：`config.time_column` → `config.month_column` → manifest `roles.time` → 现场推断；
* 分数 PSI（`models.evaluate.monthly_psi`）：
  - 切点在**基期合并池**上拟合，各月共用同一套切点（各月各自切点 ⇒ 每箱占比≈1/n ⇒ PSI 恒 0）；
  - 基期默认 = 最早一个月；`config.psi_base_months=["2026-01","2026-02"]` 合并成池；
  - 指定的基期月**不存在** ⇒ `config_invalid`（**不静默回落到最早月**）；
  - 基期覆盖**全部**月份 ⇒ `psi.enabled=false` + `reason="…没有对比月…"`（不是 PSI=0）；
  - 只有 1 个月 / 无时间列 ⇒ `enabled=false` + `reason`，不伪造数值；
  - 零占比按 `1e-6` 处理（与 `core.psi` 同口径），不让 ln(0) 变 inf；
* 读数三件套：`by_month[].psi`（逐月）、`max_psi` / `max_month`（最差月，**闸门用它**）、
  `overall_psi`（对比月合并池 vs 基期池，仅参考 —— 一个月左漂、一个月右漂时合并值会互相抵消）；
* 分月评估表：`evaluation/psi_by_month.csv`（月份 / psi / n），配 `metrics.json.psi` 的
  `base_months` / `base_label` 一起读，基期说不清则 PSI 不可比。

## 10. 多分类：LIFT 业务解读与 PSI 口径决策

### 10.1 多分类 LIFT 怎么读（业务解读）

* LIFT@pct 的定义与二分类**完全一致**（§4）：段内坏样本率 / 整体坏样本率；
  多分类下「坏样本」在逐类 OVR 里指**该类样本**（`1{y=k}`），不是「任一坏」；
* 因此每一类的 `LIFT_k` 回答的是：**按该类概率降序取尾部 pct 名单，能抓到该类多少浓度**。
  例如 M2 的 `LIFT=2.4` ⇒ 尾部 10% 名单里 M2 的占比是整体的 2.4 倍；
* `macro.lift` 是各类 `LIFT_k` 的**算术平均**：它没有样本量加权，小类会把它拉偏；
  要"以类样本量加权"看 `weighted.lift`（仅参考）。**不要**把各类 LIFT 相加，
  也不要拿 macro LIFT 当「整体抓坏能力」；
* 顶层 `lift` 与 `macro.lift` 同值（"一个 run 一个主指标"的口径，与 KS/AUC/Gini 一致）；
* 可靠性：`lift_seg_size ≥ 30` 才算可靠（`is_reliable`）；坏率很低时尾部可能一个该类样本都没抓到
  ⇒ `LIFT=0` 是**真实值**（不是 bug），报表请看 `lift_seg_size` / `is_reliable` 再下结论；
* 业务落点：LIFT 是**排序有效性**指标，不是绝对命中率；名单规模由业务容量定，
  LIFT 用来比较不同模型/不同版本的排序质量。

### 10.2 PSI 口径：多分类按 `bad_class` 概率列（已定）

决策：**多分类的分数 PSI 用 `bad_class` 那一列概率**（即 `predict_proba` 列 = `bad_index` 列），
既不是「预测类别（argmax）分布的 PSI」，也不是「逐类概率矩阵每列各算一个 PSI」。

理由：

1. **与闸门/报表同口径**：确认闸门校验的是 `bad_class` 分数，PSI 若换个口径，
   闸门里的两个数就不是同一个分数序列上的事实；
2. **与打分链路同源**：`predict_score` 的 `predict_proba` / `fico_score` 就是这一列，
   PSI 与线上真正使用的分数完全对应；
3. **argmax 分布 PSI 对分界线不敏感**：概率从 0.51 掉到 0.50 就跨类，但分布几乎没变；
   反过来 0.99→0.60 仍不跨类却被判"稳定"，容易得到虚假的"很稳"结论；
4. 逐类各算一个 PSI 会得到 k 个读数、没有唯一结论，而"一个 run 一个主读数"是本项目的口径原则。

需要看逐类分布漂移时，用 `evaluation/per_class_metrics.csv`（逐类 OVR 指标）与分月分数的
逐类分布做补充分析，**不混进** `metrics.json.psi` 字段（避免"PSI 到底指哪一个"的歧义）。

## 11. 附：字段表达式（划分专用）

`docs/tool-contract.md` §3.3 的受限表达式，实现在 `core/expr.py`：

| 写法 | 语义 |
|---|---|
| `==` `!=` `>` `>=` `<` `<=` | 比较（左右都可是列或字面量） |
| `in (...)` / `not in (...)` | 集合包含（列表只能写字面量） |
| `&` / `\|` | 与 / 或（`&` 优先级高于 `\|`） |
| `( )` | 分组 |
| `'str'` / `"str"` / `123` / `1.5` / `true` / `false` / `null` | 字面量 |
| `` `列 名` `` | 反引号包住非常规列名 |

**禁止**：函数调用、属性访问、赋值、分号、`__` 开头标识符、`eval`/`df.query`。
非法输入一律 `ConfigError`，消息带位置、原表达式与可用示例。

缺失语义：与 pandas 一致（`NaN > 1` 为 `False`，`NaN != 1` 为 `True`）；
要显式剔缺失写 `x != null`（等价 `notna()`），`x == null` 等价 `isna()`。
