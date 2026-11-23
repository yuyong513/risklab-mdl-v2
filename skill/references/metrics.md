# 指标口径摘要（KS / AUC / Gini / LIFT / PSI / IV）

> 完整口径见 `docs/口径-metrics.md`（v1.0 定稿）与实现 `src/risklab_mdl/core/metrics.py`。
> 本文是 skill 侧**读数与话术**用的摘要：怎么读、怎么比、什么时候不能说结论。

## 0. 记号与方向

| 记号 | 含义 |
|---|---|
| `y` | 标签；二分类 `1=坏 / 0=好`；多分类任意离散标签 |
| `s` | 风险分数（概率）；**单调**，允许并列 |
| `N_bad / N_good` | 坏 / 好样本数 |
| `score_direction` | 统一方向判定：`AUC ≥ 0.5` ⇒ `higher_score_higher_risk`；各报表**不得自行翻转方向** |

## 1. AUC

`P(s_bad > s_good) + 0.5·P(s_bad = s_good)`（Mann–Whitney U，平均秩实现，并列取平均秩），
非有限分数行先剔除，与 `sklearn.roc_auc_score` 一致（偏差 < 1e-12）。

- 取值 `[0,1]`；`< 0.5` 表示分数方向与标签相反，**系统不自动翻转**。
- 小样本不显著：`|AUC-0.5|` 不显著时**不该**当成"有效果"或"没效果"的结论。
- 读法：`0.5` 无区分、`0.6–0.7` 一般、`0.7–0.8` 好、`>0.85` 先怀疑数据泄漏。

## 2. Gini

`Gini = 2·AUC − 1`，取值 `[-1, 1]`；**不独立实现**（避免与 AUC 口径漂移）。

## 3. KS

`KS = max_t |F_bad(t) − F_good(t)|`，按分数阈值累计的分布函数之差的最大绝对值。

- **同分客户不被拆开**（先 `unique` 再按值降序累计）；拆开会让 KS 偏大且样本顺序敏感。
- 取绝对值 ⇒ 方向反转时仍有正确量级，方向信息看 `score_direction`。
- 取值 `[0,1]`。业绩常用线：`<0.2` 弱、`0.2–0.3` 一般、`0.3–0.4` 好、`>0.5` 先查泄漏。

## 4. LIFT@pct

`LIFT@pct = 段内坏样本率 / 整体坏样本率`。

- 取尾部方式：按风险降序累计权重，取累计权重首次 ≥ `pct` 的位置（段内占比 ≥ pct，至少 1 个样本）；
- 默认 `pct=0.10`（`lift` 字段即 LIFT@10%）；
- **可靠性**：`is_reliable = seg_size ≥ 30`；不可靠时**不修正、只标记**——小样本 LIFT 不能当结论；
- 坏率很低时尾部可能没抓到坏样本 ⇒ **`LIFT=0` 是真实值**，读 `lift_seg_size` / `is_reliable` 再下结论。
- 业务落点：LIFT 是**排序有效性**指标，不是绝对命中率。

## 5. IV（分箱区分度）

对每箱 `i`：

```text
bad_pct_i  = (bad_i  + δ) / (N_bad  + δ·K)
good_pct_i = (good_i + δ) / (N_good + δ·K)
WOE_i      = ln(bad_pct_i / good_pct_i)
IV         = Σ_i (bad_pct_i − good_pct_i) · WOE_i
```

- `K` = 箱数（含缺失箱），`δ` = `woe_smoothing`（默认 0.5；`sparse_strategy="keep"` 时 δ=0）；
- 稀疏箱先合并（缺失箱不参与合并），合并下限 `max(2, total // min_bin_count)` 防塌成单箱；
- **IV 读数依赖箱数**：`requested_bins` 与实际箱数必须一起看（箱数被下调时 IV 不可与历史比）。

IV 分级（skill 话术用）：

| IV | 区分度 | 话术 |
|---|---|---|
| `< 0.02` | 几乎无 | 建议初筛淘汰 |
| `0.02 – 0.1` | 弱 | 可留作辅助 |
| `0.1 – 0.3` | 中 | 主力特征 |
| `0.3 – 0.5` | 强 | 关注稳定性与业务合理性 |
| `≥ 0.5` | **可疑** | 先排查是否含未来信息（贷后字段/标签派生），不要直接入模 |

## 6. PSI（跨期稳定性）

`PSI = Σ_i (test_i − base_i) · ln(test_i / base_i)`，`i` 为**同一套箱**。

三条硬口径：

1. **缺失箱参与对比**（缺失是独立箱 `-1`，不是哨兵值 `-99`）；
2. **test 中不存在的箱按占比 0 计**（跳过会低估 PSI）；
3. 分母统一加 `eps=1e-6` 防除零。

分数 PSI（`model_evaluate`）：

- 切点在**基期合并池**上拟合、各月共用（各月各自切点 ⇒ 占比≈1/n ⇒ PSI 恒 0，看起来"很稳"实际没测到）；
- 基期默认 = 最早一个月；可用 `config.psi_base_months=["2026-01","2026-02"]` 合并成池；
- 指定的基期月不存在 ⇒ `config_invalid`（**不静默回落到最早月**）；
- 基期覆盖全部月份 ⇒ `psi.enabled=false` + `reason`（不是 PSI=0）；
- 无时间列 / 只有 1 个月 ⇒ `enabled=false` + `reason`，**不伪造数值**。

读数三件套与用法：

| 字段 | 含义 | 用法 |
|---|---|---|
| `by_month[].psi` | 逐月 PSI（对比该月 vs 基期池） | 看漂移发生在哪个月 |
| `max_psi` / `max_month` | 最差月的 PSI | **确认闸门用它** |
| `overall_psi` | 对比月合并池 vs 基期池 | **仅参考**（一个月左漂、一个月右漂会互相抵消） |

经验线：`< 0.10` 稳定、`0.10–0.25` 需关注、`> 0.25` 显著漂移（要重训或分群建模）。

## 7. 多分类（OVR + macro）

对每个类别 `k`：把 `y == k` 视为坏、其余为好，用第 `k` 列概率当分数，算
`AUC_k / KS_k / Gini_k / LIFT_k`（与 `sklearn roc_auc_score(multi_class="ovr", average="macro")` 同口径）。

- `macro`：**有效类的算术平均** —— 进主结论；
- `weighted`：按类样本数加权 —— 仅参考（大类主导，会掩盖小类失效）；
- 顶层 `ks/auc/gini/lift` 取 `macro`（"一个 run 一个主指标"）；
- 跳过与可靠性（必须可见）：某类在该样本无正例/无负例 ⇒ `status=skipped` + `reason`，不参与 macro；
  某类样本 < 30 ⇒ 仍给指标但 `is_reliable=false`；所有类都跳过 ⇒ `macro` 为 `null`；
- **KS/LIFT 表与图只吃 0/1 标签**：多分类先 `y_bin = (y == bad_index)` 二值化，
  表里多一列 `bad_class`，且 `Σbad` 必须等于坏类样本数。把 `0..k-1` 直接喂进去会错位且看不出来
  ——实现侧会直接抛 `DataError`。
- **不要**把各类 LIFT 相加，也不要把 `macro.lift` 当"整体抓坏能力"。

## 8. 确认阈值与差距读法（`model_confirm`）

| 项 | 默认 | 说明 |
|---|---|---|
| `ks_min` | **0.25** | 必须落在 `[0,1]`，越界 ⇒ `config_invalid` |
| `psi_max` | **0.10** | 不能为负 |

判定：`ks ≥ ks_min` 且（PSI 不可用 或 `psi ≤ psi_max`）才通过。

- **KS 取值**：val 优先；val 指标被跳过（该子集只有一类标签）时回落 train，`ks_source` 显式返回
  `val`/`train`，不混用；
- **PSI 取值**：`max_psi`（最差月），不是平均、不是 overall；
- 不通过 ⇒ `error.code=threshold_not_met`，`data` 给
  `ks / ks_gap / psi / psi_gap / failed_checks / checks`
  （不通过时 `data` **不含** `confirmed` 字段，别去读它）；
- **差距值怎么读**（实测口径，`tools/model.py` 原样）：
  `ks_gap = ks_min − ks`、`psi_gap = psi − psi_max`。
  **正值 = 不达标方向的差距**，负值 = 达标富余。
  例：`ks=0.402, ks_min=0.25 ⇒ ks_gap=-0.152`（富余 0.152）；
  `psi=0.115, psi_max=0.10 ⇒ psi_gap=+0.015`（超线 0.015）。
  话术里要把符号含义说明白，别把 `-0.15` 说成"差 0.15"。
- PSI 未计算（无时间列/只有一个月/基期覆盖全部月）⇒ **只校验 KS**，`warnings` 必须写明
  「PSI 未计算，只校验 KS —— 不代表跨期稳定」；
- val 样本量 < 100 ⇒ 附「可靠性有限」告警，但仍允许确认；
- **多分类确认口径**：主指标取 **macro KS**，阈值与二分类同一套（不引入第二套）；
  `bad_class` 的 OVR KS 只作诊断读数。

## 9. 展示与判读纪律

1. 指标一律标注**子集**（train / val / test）与**样本量**（`n` / `bad`）；没有 val 的指标不得当"样本外表现"。
2. `status=skipped` 的子集要说明原因，**不能显示成 0**。
3. 报 KS/AUC 时同屏给 PSI（若可用）——**区分度与稳定性不可互替**：KS 高但分布右移照样是上线风险。
4. 只保留 4 位小数；不要二次四舍五入或"取整到 0.3"。
5. 指标之间不要换算造假（如"Gini 0.32 所以 AUC 0.66"可以，因为 `Gini=2·AUC−1`；
   但"KS 0.4 ⇒ 通过率 60%"这类推断不许写）。
