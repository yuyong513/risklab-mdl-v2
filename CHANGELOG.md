# Changelog

## 2.0.0（2026-11）

`risklab-mdl-v2` 是「**skill 编排 + MCP 计算 + agent 交互**」三层架构的
风控建模计算层（MCP Server）。本版本首次公开。

### 交付范围

| 层 | 内容 |
|---|---|
| L2 计算层（本包） | `src/risklab_mdl/`：20 个 MCP 工具、产物契约、6 项 doctor 探针、跨平台基础设施 |
| L1 编排层 | `skill/SKILL.md` + `references/`（tools / workflow / models / metrics / errors / artifacts） |
| L3 接入层 | `mcp/mcp.{claude,dsh,workbuddy}.json` 三客户端配置样例 + `docs/skill使用指南.md` |
| 验证脚本 | `scripts/check_layout.py`（产物树契约校验） |
| 测试 | `tests/`：core 单测、工具契约、三态协议、manifest 并发、stdio 握手、skill 契约、静态红线 |

### 关键决策

1. **服务端不维护会话状态**：工具入参一律显式（路径 + 本步参数）；需要用户决策时返回
   `needs_user_input`，由 skill / agent 呈现。没有服务端 session、弹窗、HTML 模板。
2. **产物位置唯一**：项目目录只由 CSV 路径决定（`<CSV 同目录>/mdl-<CSV stem>/`），
   已存在则复用且不清理历史 run；CSV 目录不可写 ⇒ `project_dir_not_writable`，**不静默换位置**。
3. **run 目录名冻结**：`runs/<YYYYmmddTHHMMSS>/`，同秒冲突按秒递推，`mkdir(exist_ok=False)` 原子占位。
4. **manifest 是唯一状态真源**：schema `risklab-mdl.manifest/1`，原子写 + 项目级 `filelock`；
   增量变更走 `update_manifest`（锁内读改写），并发不静默覆盖；`updated_at` 严格递增。
5. **失败必须可见**：三态 `ok / needs_user_input / error`；任何降级进 `warnings`；
   错误必须带 code + message + 可执行 hint。禁止用空结果 + `ok:true` 掩盖失败。
6. **计算口径先冻结再实现**：KS 同分不拆开、LIFT 尾部按权重、IV 平滑/稀疏箱合并、
   PSI 缺失独立箱、多分类 OVR+macro（见 `docs/口径-metrics.md`）。
7. **确认闸门默认 KS ≥ 0.25、PSI ≤ 0.10**：不达标返回 `threshold_not_met` 并给差距值；
   通过才写 `final/` + `manifest.confirmed_run`；PSI 不可用时只校验 KS 并显式告警。
8. **调参只允许 `tune_on=val`**：OOT/train/cv 一律 `config_invalid`（避免在测试集上调参造成数据穿越）。
9. **报告只读产物不重算**：`report_build` 的数字全部来自 `runs/*/metrics.json` 与
   `importance.csv`，避免「报告 KS 与确认时对不上」。
10. **跨平台**：无 `fcntl`，用 `filelock`；无 POSIX-only 依赖；路径/编码安全写盘。
11. **skill 层不伪造数值**：小样本 LIFT=0、PSI 不可用、指标 skipped 都如实呈现。

### 代码组织（2.0.0 定稿）

- 三态响应只由 `src/risklab_mdl/protocol.py` 构造；
- 项目状态读取（manifest 角色 / CSV 定位）只由 `src/risklab_mdl/project.py` 负责；
- 跨层共享的数值与类别工具只放在 `src/risklab_mdl/core/utils.py`，不在各层各写一份；
- 非核心的私有 helper 统一 `_` 前缀，跨模块不再引用私有名；
- 重依赖（lightgbm / xgboost / matplotlib）保持函数内惰性导入，入口 `core/deps.py`。

### 验证数据（Windows / Python 3.11）

- 全量测试：`python -m pytest tests -q` 全部通过；
- 快速 e2e：`tests/test_e2e_sample.py`（合成数据全链路，约 7 秒）；
- 产物门禁：`scripts/check_layout.py` 退出码 0；
- 依赖：mcp / pandas / numpy / scikit-learn / lightgbm / xgboost / matplotlib /
  hyperopt / openpyxl / PyYAML / filelock。

### 未实测 / 待后续

- Claude Code 与 WorkBuddy 客户端界面未实测（MCP 协议握手已实测）；
- macOS / Linux、MCP SSE 传输未实测；
- 完整调参 `execute`（20 trials）未纳入门禁耗时。

## 许可

PolyForm Noncommercial License 1.0.0：允许非商业用途，禁止商业使用；完整条款见 `LICENSE`。
