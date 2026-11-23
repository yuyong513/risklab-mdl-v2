# 工具契约（冻结）

> 状态：**已冻结（2.0.0）**。实现必须遵循本文件；工具名/入参名/返回结构不得擅自改名。
> 服务端实现：`src/risklab_mdl/server.py` 的 `register_tools()`；工具实现在 `src/risklab_mdl/tools/`。

## 0. 会话与注册现状

- 服务端注册 **20 个** MCP 工具（§2）：`server_probe` + 19 个业务工具；`tests/test_server_smoke.py` 对工具集合做严格断言。
- 需要用户决策时返回 `needs_user_input`，由 skill / agent 呈现。
- 服务端**不维护会话状态**：工具入参一律显式（路径 + 本步参数），不读服务端缓存；同一份数据在两次调用间被外部改动时，以磁盘为准。

## 1. 通用返回协议（三态）

所有工具返回**一个** JSON 对象（MCP `content` 里的 text 即该对象的 JSON 序列化）：

```json
{
  "ok": true,
  "status": "ok",
  "data": {},
  "artifacts": [],
  "warnings": [],
  "error": null
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `ok` | bool | 便捷布尔：`status == "ok"` 时为 true，其余为 false |
| `status` | `"ok"` \| `"needs_user_input"` \| `"error"` | 三态，互斥 |
| `data` | object | 本步结构化结果（表格、指标、路径、问题载荷） |
| `artifacts` | string[] | 本次**新产生或更新**的产物相对路径（相对项目目录，POSIX 分隔符） |
| `warnings` | string[] | 非阻断提示（缺时间列跳过 PSI、缺字体降级、样本不足以支撑阈值等） |
| `error` | object \| null | 仅 `status == "error"` 时非空，见 §1.2 |

### 1.1 `needs_user_input`（替代早期版本的弹窗）

```json
{
  "ok": false,
  "status": "needs_user_input",
  "data": {
    "question": {
      "id": "dataset_confirm_target",
      "text": "请确认目标字段与任务类型",
      "options": [{"value": "binary", "label": "二分类（如 0/1 好坏）"}],
      "default": "binary",
      "fields": [{"name": "target_column", "type": "string", "required": true}]
    },
    "context": {"columns": ["id", "y", "month"], "candidates": ["y"], "label_counts": {"0": 1965, "1": 35}}
  },
  "artifacts": [],
  "warnings": [],
  "error": null
}
```

- 服务端**只返回数据载荷**，不渲染任何模板/HTML/弹窗；由 skill 或 agent 负责呈现。
- 用户回答后，agent 把答案作为**下一次调用**的显式参数传入（没有服务端 session）。

### 1.2 `error` 载荷与错误码

```json
{"code": "project_dir_not_writable", "message": "CSV 所在目录不可写…", "hint": "把 CSV 复制到可写目录后重试"}
```

失败必须可见（早期版本「失败不可见」教训的硬约束）：**不允许**用空结果 + `ok:true` 掩盖失败。

| 错误码 | 触发场景 | 修复指引要点 |
|---|---|---|
| `path_not_readable` | 文件不存在/不可读/是目录 | 核对路径拼写 |
| `path_not_writable` | 目标文件无法写入 | 查权限/占用 |
| `project_dir_not_writable` | CSV 所在目录不可写，无法建 `mdl-*` | 换可写目录，**不静默换位置** |
| `project_dir_conflict` | `mdl-*` 位置被同名文件占用 | 删除/重命名该文件 |
| `project_dir_not_found` | 传入的 `project_dir` 不存在 | 先 `project_init` |
| `manifest_not_found` | 项目缺 `manifest.json` | 先 `project_init` |
| `manifest_schema_invalid` | manifest 结构非法（含手工改坏） | 按 schema `risklab-mdl.manifest/1` 修 |
| `json_decode_error` | 产物 JSON 损坏/被截断 | 重跑产生该产物的步骤 |
| `encoding_undecodable` | 文件无法按探测到的编码解码（二进制/加密/半截文件） | 另存为 UTF-8（或 UTF-16/GB18030）后重试 |
| `manifest_conflict` | 基于过期 manifest 快照写入（并发 lost update） | 重读后合并；增量变更统一走 `update_manifest(mutator)` |
| `lock_timeout` | 项目级文件锁超时（并发写） | 等另一进程结束，或清理残留 `.lock` |
| `run_id_invalid` | `run_id` 不符合 `^\d{8}T\d{6}$` | 从 `run_list` 取，别手写 |
| `run_not_found` | `runs/<run_id>/` 不存在 | 用 `run_list` 查可用 run |
| `dependency_missing` | 运行期依赖缺失（如 xgboost 未装） | 按 `hint` 里的 pip 命令装 |
| `task_algorithm_mismatch` | 算法不支持该任务类型 | 按 `model_catalog` 换算法 |
| `expression_invalid` | 样本划分字段表达式非法 | 按示例改写（禁 `eval`） |
| `threshold_not_met` | KS/PSI 未达确认阈值 | 给差距值 + 允许调阈值重试 |
| `features_empty` | 筛选后无可用特征 | 放宽带筛选条件 |

## 2. 工具清单（20 个，冻结）

| # | 工具 | 家族 | 用途（一句话） | 核心入参 | 主要产物 |
|---|---|---|---|---|---|
| 1 | `project_init` | 项目 | 由 CSV 路径建/复用 `mdl-<stem>/` 与 manifest | `csv_path` | `manifest.json`、各子目录 |
| 2 | `dataset_load` | 数据 | 读 CSV：编码探测、列清单、角色推断（不阻断） | `csv_path` | `manifest.json`（project.roles 建议） |
| 3 | `dataset_columns` | 数据 | 列出列与角色，供用户确认/改角色 | `project_dir` | 无（返回 `data.columns`） |
| 4 | `dataset_profile` | 数据 | 样本画像：目标分布、缺失/同值/取值数、描述统计 + 图表 | `project_dir`, `config?` | `data_analysis/*` |
| 5 | `feature_analyze` | 特征 | 每特征分箱 + WOE/IV 明细（等频/等宽/DT/自定义） | `project_dir`, `config` | `feature_analysis/*` |
| 6 | `feature_charts` | 特征 | 特征分析图表（IV 分布、TOP 特征箱图、相关性热力图） | `project_dir`, `config?` | `feature_analysis/charts/*.png` |
| 7 | `feature_prescreen` | 特征 | 初步筛选：缺失率/同值率/IV/PSI/相关性五道筛子 | `project_dir`, `config` | `feature_selection/prescreen.json` |
| 8 | `feature_refine` | 特征 | 精细筛选（gain 迭代 / 强保留 / 强剔除 / 轮数） | `project_dir`, `config` | `feature_selection/refine.json`、`selected_features.txt` |
| 9 | `preprocess_check` | 预处理 | 入模前检查：缺失、文本列、类别基数、编码建议 | `project_dir`, `config?` | `preprocess/check.json` |
| 10 | `preprocess_apply` | 预处理 | 落实缺失/编码/盖帽/手动分箱规格 | `project_dir`, `spec` | `preprocess/fill|encode|outlier|manual_bins.json` |
| 11 | `dataset_split` | 划分 | 不划分 / 随机 / 分层 / 字段表达式（受限解析，禁 eval） | `project_dir`, `config` | `split/split.json` |
| 12 | `model_catalog` | 模型 | 算法 × 任务能力表、默认参数、默认调参空间、说明 | 无 | 无（返回 `data.catalog`） |
| 13 | `model_train` | 模型 | 训练一个 run（DT/LR/GBDT/XGB/LGB × binary/multiclass） | `project_dir`, `config` | `runs/<run_id>/*`、manifest.runs 追加 |
| 14 | `model_tune` | 模型 | 调参：`mode=preview` 返回空间说明；`mode=execute` 落 run | `project_dir`, `config`, `mode` | `runs/<run_id>/*` |
| 15 | `model_evaluate` | 模型 | KS/AUC/Gini/LIFT/PSI（多分类按类 OVR + macro） | `project_dir`, `run_id`, `config?` | `runs/<run_id>/metrics.json`、`evaluation/` |
| 16 | `run_list` | 过程 | 列出 run 与指标摘要（按时间升序） | `project_dir` | 无（返回 `data.runs`） |
| 17 | `run_compare` | 过程 | 对比多个 run 的指标，标出最优 | `project_dir`, `run_ids?` | `reports/run_compare.csv` |
| 18 | `model_confirm` | 过程 | 阈值校验通过后锁定最终模型 | `project_dir`, `run_id`, `thresholds?` | `final/model.pkl`、`final/confirm.json`、`manifest.confirmed_run` |
| 19 | `report_build` | 过程 | 汇总报告（读产物，不重算） | `project_dir` | `reports/结果汇总.xlsx`、`reports/过程报告.md` |
| 20 | `server_probe` | 项目 | 环境探针：Python/依赖/产物目录链路（MCP 会话第一条调用） | 无 | 无（返回 `data.version/platform/deps/artifacts_check`） |

> **第 20 个 MCP 工具是 `server_probe`，不是 `doctor`。**
> `doctor`（六项环境自检）**只有 CLI 形式**：`risklab-mdl-doctor` / `python -m risklab_mdl.doctor`，
> 未注册为 MCP 工具；MCP 侧对应的环境探针是 `server_probe`（见 §3.5 与 §5）。
> 二者实现同源（共用依赖清单与产物目录检查约定），但**对外入口不同**，不要按 MCP 工具去调 `doctor`。

> 约定：入参名 `project_dir` 一律指 `<CSV 同目录>/mdl-<stem>` 目录（非 CSV 路径）；除 `project_init` / `dataset_load` 直接吃 `csv_path` 外，其余工具都吃 `project_dir`。

## 3. 分家族详细契约

### 3.1 项目与数据（1–4）

**`project_init`**
- 入参：`csv_path: str`（必填）。
- 行为：解析 `mdl-<stem>`（CSV 同目录）→ 已存在则复用 → 建齐固定子目录 → 无 manifest 时创建（schema `risklab-mdl.manifest/1`）。
- 输出 `data`：`{project_dir, created: bool, manifest}`。
- 失败：`path_not_readable` / `project_dir_not_writable` / `project_dir_conflict`。

**`dataset_load`**
- 入参：`csv_path: str`。
- 输出 `data`：`{project_dir, encoding, rows, columns, roles: {id, time, split, target_candidates}, dtypes: {列: 推断类型}}`。
- 角色推断**不阻断**：目标字段由用户在前置确认环节显式给出（`dataset_confirm_target` 语义并入后续建模入参；本契约以 `config.target` 承载）。

**`dataset_columns`**
- 入参：`project_dir: str`。
- 输出 `data`：`{columns: [{name, dtype, role, nunique, missing_rate}]}`。

**`dataset_profile`**
- 入参：`project_dir: str`、`config?: {target?, max_features_for_charts?, sample_rows?}`。
- 产物：`data_analysis/profile.json`、`describe.csv`、`target_distribution.csv`、`charts/*.png`、`index.html`（单文件、无外链）。
- 输出 `data`：`{target_distribution, missing_summary, chart_count}`；无时间列时跳过时间相关图并在 `warnings` 说明。

### 3.2 特征分析与筛选（5–8）

**`feature_analyze`**
- 入参：`project_dir`、`config: {method: "quantile"|"equal_width"|"dt"|"custom", bins?: int, features?: [str], target?: {...}}`。
- 产物：`feature_analysis/iv_table.csv`、`woe_detail.csv`、`charts/*.png`、`index.html`。
- 输出 `data`：`{feature_count, iv_top: [{feature, iv}], skipped: [{feature, reason}]}`。
- 性能边界：宽表默认只对 `max_features_for_charts`（默认 30）出图，其余只出表并在 `warnings` 说明。

**`feature_charts`**
- 入参：`project_dir`、`config?: {features?: [str], max_features?: int}`。
- 产物：`feature_analysis/charts/*.png`（Agg backend、PNG、120 dpi、中文字体 fallback）。

**`feature_prescreen`**
- 入参：`project_dir`、`config: {missing_rate_max?, iv_min?, psi_max?, corr_max?, same_value_rate_max?}`。
- 产物：`feature_selection/prescreen.json`（逐特征判定 + 淘汰原因）。

**`feature_refine`**
- 入参：`project_dir`、`config: {scheme: "gain", keep?: [str], drop?: [str], rounds?: int, threshold?}`。
- 产物：`feature_selection/refine.json`、`selected_features.txt`（一行一个特征，UTF-8 无 BOM）。

### 3.3 预处理与划分（9–11）

**`preprocess_check`**
- 入参：`project_dir`、`config?: {features?: [str]}`。
- 产物：`preprocess/check.json`：`{must_fix: [...], suggested: [...], by_feature: {...}}`（分类"必须处理/建议处理"）。

**`preprocess_apply`**
- 入参：`project_dir`、`spec: {missing, encode, cap, binning}`（结构见架构文档 §4.2）。
- 校验：区间有序、覆盖全值域、缺失箱声明；非法 → `error.code=spec_invalid`。
- 产物：`preprocess/fill.json`、`encode.json`、`outlier.json`、`manual_bins.json`。

**`dataset_split`**
- 入参：`project_dir`、`config: {mode: "none"|"random"|"stratified"|"field", train?, val?, test?, seed?, stratify_by?, field?: {column, expr_train, expr_val, expr_test}}`。
- 产物：`split/split.json`（含 mask 指纹、各类占比、seed；同 seed 可复现）。
- 受限表达式：支持 `==`、`!=`、`in`、`>=`、`<=`、`&`、`|`；**禁止 `eval`**，非法表达式 `expression_invalid` 并给示例。

### 3.4 模型与过程（12–19）

**`model_catalog`**
- 入参：无。
- 输出 `data.catalog`：`{algorithms: {dt|lr|gbdt|xgb|lgb: {tasks: [binary, multiclass], default_params, default_space, notes}}}`（5 × 2 = 10 格能力表）。

**`model_train`**
- 入参：`project_dir`、`config: {algorithm, task, features?, preprocess?, split?, params?, seed?}`。
- 产物：`runs/<run_id>/config.json`、`params.json`、`metrics.json`、`model.pkl`、`importance.csv`；`manifest.runs` 追加一条。
- 输出 `data`：`{run_id, metrics, algorithm, task}`。
- 失败：算法×任务不匹配 → `task_algorithm_mismatch`（**不降级**）；依赖缺失 → `dependency_missing`。

**`model_tune`**
- 入参：`project_dir`、`config: {algorithm, task, space?, trials?, objective?, seed?}`、`mode: "preview"|"execute"`。
- `preview` 返回 `data.space`（参数名/类型/范围/默认）与预计试验数，**不训练**；`execute` 落一个新 run（`manifest.runs` 追加）。

**`model_evaluate`**
- 入参：`project_dir`、`run_id`、`config?: {sample?: "train"|"val"|"test", by_time?: bool}`。
- 产物：`runs/<run_id>/metrics.json`（`{train, val, test, psi, multi: {per_class, macro}}`）、`evaluation/` 表与图。
- 无时间列：PSI 跳过，`warnings` 明示，不伪造数值。

**`run_list`**
- 入参：`project_dir`。输出 `data.runs`（按 run_id 升序）：`[{run_id, algorithm, task, status, metrics, created_at}]`。

**`run_compare`**
- 入参：`project_dir`、`run_ids?: [str]`（缺省=全部）。
- 产物：`reports/run_compare.csv`；输出 `data.best`（按主指标，默认 KS）。

**`model_confirm`**
- 入参：`project_dir`、`run_id`、`thresholds?: {ks_min?, psi_max?}`（默认 KS≥0.25、PSI≤0.10）。
- 行为：阈值不达标 → `error.code=threshold_not_met` 并给差距值；达标才写 `final/`；样本量不足允许确认但 `warnings` 标注"不可靠"。
- 产物：`final/model.pkl`、`final/confirm.json`（含阈值、指标、确认时间）、`manifest.confirmed_run`。

**`report_build`**
- 入参：`project_dir`。
- 产物：`reports/结果汇总.xlsx`（openpyxl，多 sheet 非空）、`reports/过程报告.md`；只读产物、不重算。

### 3.5 环境自检（第 20 个工具 `server_probe`；`doctor` 是 CLI）

**`server_probe`（MCP 工具，完整定义见 §5）**
- 入参：无。
- 输出：`data.version / data.platform / data.deps / data.artifacts_check`（并保留早期已发布的顶层同名字段）。
- 语义：依赖缺失或产物目录链路不可用 ⇒ `status="error"`、`error.code="environment_not_ready"`，明细仍完整返回。

**`doctor`（CLI，不是 MCP 工具）**
- 调用：`risklab-mdl-doctor` / `python -m risklab_mdl.doctor [--json]`，退出码 0=可用 / 1=有 FAIL。
- 输出：六项（`python` / `dependencies` / `mcp_sdk` / `matplotlib` / `chinese_font` / `artifact_dir`），每项 `{name, status: ok|warn|fail, detail, hint, fix_commands}`。
- 判定口径：
  - `python`：`pyproject.toml` 版本与包内 `__version__` 不一致 ⇒ **WARN**（不影响能否运行，不改变退出码）；Python < 3.11 ⇒ FAIL。
  - `matplotlib`：导入时默认 backend 记入 `detail`；强制 `Agg` 后仍未生效 ⇒ FAIL（防御性硬门禁，不是 WARN）。

## 4. 产物目录与 manifest（工具共用）

```
<CSV 同目录>/mdl-<CSV 文件名去扩展名>/
  manifest.json
  data_analysis/{profile.json, describe.csv, target_distribution.csv, charts/*.png, index.html}
  feature_analysis/{iv_table.csv, woe_detail.csv, charts/*.png, index.html}
  feature_selection/{prescreen.json, refine.json, selected_features.txt}
  preprocess/{check.json, fill.json, encode.json, outlier.json, manual_bins.json}
  split/split.json
  runs/<YYYYmmddTHHMMSS>/{config.json, params.json, metrics.json, model.pkl, importance.csv, evaluation/}
  final/{model.pkl, confirm.json}
  reports/{结果汇总.xlsx, 过程报告.md}
```

`manifest.json`（schema `risklab-mdl.manifest/1`，代码：`src/risklab_mdl/artifacts/manifest.py`）：

```json
{
  "schema": "risklab-mdl.manifest/1",
  "project": {
    "csv_path": "C:/data/x.csv",
    "stem": "x",
    "created_at": "2026-01-01T12:00:00+08:00",
    "updated_at": "2026-01-01T12:00:00+08:00",
    "target": {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}},
    "roles": {"id": null, "time": "month", "split": null, "features": ["x1", "x2"]}
  },
  "stages": {
    "data_analysis": {"done": true, "path": "data_analysis/", "summary": "2000 行 × 696 列"},
    "feature_analysis": {"done": false},
    "feature_selection": {"done": false},
    "preprocess": {"done": false},
    "split": {"done": false},
    "train": {"done": false}
  },
  "runs": [
    {"run_id": "20260101T120000", "algorithm": "lgb", "task": "binary",
     "status": "trained", "metrics": {"ks": 0.35, "auc": 0.72}, "paths": {}, "created_at": "…"}
  ],
  "confirmed_run": null
}
```

- `stages` 键固定为 6 个；`runs[].status ∈ {trained, tuned, evaluated, confirmed, failed}`。
- 写 manifest 一律「原子替换 + 项目级文件锁」（`file_lock`），并刷新 `updated_at`；
  并发写拿不到锁 → `lock_timeout`，**不静默覆盖**。
- **并发语义（代码：`artifacts/manifest.py`）**：
  - 增量变更只走 `update_manifest(project_dir, mutator)`：锁内「读 → 改 → 写」，不会基于过期快照；
  - `write_manifest` 只用于初始化/全量重写；「先读再写回」必须传 `expected_updated_at=<读到的 project.updated_at>`，
    锁内重读比对不一致 → `error.code=manifest_conflict`（过期快照不落盘）；
  - `updated_at` 是严格递增的逻辑时钟：同一秒内连续写按 +1s 递推，保证秒级精度下仍可区分两次写。
- **结构边界（写/读两侧同样校验）**：
  - `confirmed_run` 非 null 时必须命中 `runs[].run_id`；
  - `runs[].run_id` 全局唯一（重复即 `manifest_schema_invalid`）；
  - `runs[].task` 若给出必须 ∈ `{binary, multiclass}`（省略视为未标注，兼容早期产物）；
  - `stages.<name>` 的附加字段类型：`done` 为 bool、`path`/`summary` 为 str。
- `run_id` = `YYYYmmddTHHMMSS`（正则 `^\d{8}T\d{6}$`）；同秒冲突时按秒递推（`…T120001`），
  用 `mkdir(exist_ok=False)` 原子占位，跨平台无竞态。

## 5. 环境探针：`server_probe`

- 入参：无。
- 输出：遵循 §1 通用三态协议（含 `data`），并保留早期已发布的扁平字段
  （两者内容一致：`data.version/platform/deps/artifacts_check` 分别等于顶层同名字段）。

```json
{
  "ok": true, "status": "ok",
  "data": {
    "version": "2.0.0",
    "platform": {"python": "3.11.9", "system": "Windows", "release": "10", "machine": "AMD64", "cwd": "…", "python_executable": "…"},
    "deps": {"mcp": {"import": "mcp", "version": "2.3.0", "error": null}, "…": {}},
    "artifacts_check": {"ok": true, "temp_dir": "…", "project_dir": "…", "manifest_roundtrip": true, "run_id": "20260101T120000", "cleaned_up": true}
  },
  "version": "2.0.0",
  "platform": {"python": "3.11.9", "system": "Windows", "release": "10", "machine": "AMD64", "cwd": "…", "python_executable": "…"},
  "deps": {"mcp": {"import": "mcp", "version": "2.3.0", "error": null}, "…": {}},
  "artifacts_check": {"ok": true, "temp_dir": "…", "project_dir": "…", "manifest_roundtrip": true, "run_id": "20260101T120000", "cleaned_up": true},
  "artifacts": [], "warnings": [], "error": null
}
```

- `ok=false` 时 `error={code:"environment_not_ready", message, hint}`，并保持 `deps`/`artifacts_check` 明细可见（缺什么一眼看到）。
- 与 CLI `doctor` 同源（共用 `dependency_report()` 与 `artifacts_check()` 的实现约定）；`doctor` 不注册为 MCP 工具，MCP 侧的环境探针就是本工具。

## 6. 追加错误码（模型层，**仅追加**，不改既有工具名/入参/返回结构）

> 下列错误码在各工具载荷中实际出现，此处仅补登记；§1.2 表中既有错误码的
> 语义与触发场景**不变**。

| 错误码 | 触发场景 | 修复指引要点 |
|---|---|---|
| `config_invalid` | `config` / `thresholds` / `spec` 形态或取值非法（如 `algorithm` 缺失、`tune_on` 不是 `val`、调参空间含未知参数、`config.features` 里有不存在的列） | 按 `docs/model-layer.md`（模型层）或 `docs/preprocess-split.md`（预处理/划分）修正入参 |
| `analysis_failed` | 数据可读但计算失败：训练异常、某子集标签只有一类导致指标无定义、run 产物损坏等**非预期失败** | 看 `error.message` 定位；**不要**当成「空结果」继续 —— 失败不可用作正常结论 |
