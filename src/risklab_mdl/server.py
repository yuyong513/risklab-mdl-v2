"""MCP stdio 服务端：工具注册 + 启动。

★ 工具注册现状
---------------
* 环境门禁：``server_probe`` —— 握手成功即知道「这台机器能不能真跑建模」；
* 数据与特征分析家族 6 个工具（契约 §3.1–3.2）——
  ``project_init`` / ``dataset_load`` / ``dataset_columns`` / ``dataset_profile`` /
  ``feature_analyze`` / ``feature_charts``；
* 预处理 / 划分 / 筛选 5 个工具（契约 §3.2–3.3）——
  ``preprocess_check`` / ``preprocess_apply`` / ``dataset_split`` /
  ``feature_prescreen`` / ``feature_refine``；
* 模型与过程 8 个工具（契约 §3.4）——
  ``model_catalog`` / ``model_train`` / ``model_tune`` / ``model_evaluate`` /
  ``run_list`` / ``run_compare`` / ``model_confirm`` / ``report_build``。

合计 **20 个工具**（``server_probe`` + 6 + 5 + 8），`tests/test_server_smoke.py`
对工具集合做严格断言（少一个/多一个都会红）。

★ 服务端不做的事
----------------
不渲染弹窗、不维护流程状态机、不留会话状态（工具入参一律显式）。
需要用户决策时返回 ``needs_user_input`` 载荷，由 skill/agent 呈现。
"""

from __future__ import annotations

import argparse
import os
import platform as _platform
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__
from .artifacts import (
    RUN_ID_PATTERN,
    allocate_run_dir,
    new_manifest,
    open_project,
    read_manifest,
    write_manifest,
)
from .sdk import McpSdkError, make_server
from .tools.dataset import (
    dataset_columns as _dataset_columns,
    dataset_load as _dataset_load,
    dataset_profile as _dataset_profile,
    project_init as _project_init,
)
from .tools.feature import (
    feature_analyze as _feature_analyze,
    feature_charts as _feature_charts,
)
from .tools.preprocess import (
    preprocess_apply as _preprocess_apply,
    preprocess_check as _preprocess_check,
)
from .tools.selection import (
    feature_prescreen as _feature_prescreen,
    feature_refine as _feature_refine,
)
from .tools.split import dataset_split as _dataset_split
from .tools.model import (
    model_catalog as _model_catalog,
    model_confirm as _model_confirm,
    model_evaluate as _model_evaluate,
    model_train as _model_train,
    model_tune as _model_tune,
)
from .tools.artifact import (
    report_build as _report_build,
    run_compare as _run_compare,
    run_list as _run_list,
)

SERVER_NAME = "risklab-mdl-v2"

#: 运行时依赖的 import 名（包名与 import 名不同的在此映射）。
_DEPS: tuple[tuple[str, str], ...] = (
    ("mcp", "mcp"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("lightgbm", "lightgbm"),
    ("xgboost", "xgboost"),
    ("matplotlib", "matplotlib"),
    ("hyperopt", "hyperopt"),
    ("openpyxl", "openpyxl"),
    ("PyYAML", "yaml"),
    ("filelock", "filelock"),
)


def dependency_report() -> Dict[str, Any]:
    """逐项探测依赖：``{包名: {import, version, error}}``（不抛异常）。"""
    import importlib
    import importlib.metadata as md

    out: Dict[str, Any] = {}
    for pkg, module_name in _DEPS:
        entry: Dict[str, Any] = {"import": module_name, "version": None, "error": None}
        try:
            importlib.import_module(module_name)
            try:
                entry["version"] = md.version(pkg)
            except Exception:  # noqa: BLE001 - 版本号读不到不影响「能 import」
                entry["version"] = "?"
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
        out[pkg] = entry
    return out


def platform_report() -> Dict[str, Any]:
    """平台信息（Python / 操作系统）。"""
    return {
        "python": _platform.python_version(),
        "python_implementation": _platform.python_implementation(),
        "python_executable": sys.executable,
        "system": _platform.system(),
        "release": _platform.release(),
        "machine": _platform.machine(),
        "cwd": os.getcwd(),
        "package_version": __version__,
    }


def artifacts_check() -> Dict[str, Any]:
    """在临时目录实地验证「mdl-<stem>/ + manifest 往返 + run 目录命名」这条链。

    只用系统临时目录，验证完立即删除；失败不抛异常，把错误放进返回值，
    这样客户端拿到的是「探针结论」而不是一次断连。
    """
    result: Dict[str, Any] = {
        "ok": False,
        "temp_dir": None,
        "project_dir": None,
        "manifest_roundtrip": False,
        "run_id": None,
        "run_id_pattern": RUN_ID_PATTERN.pattern,
        "error": None,
    }
    tmp_root: Optional[str] = None
    try:
        tmp_root = tempfile.mkdtemp(prefix="risklab-mdl-probe-")
        result["temp_dir"] = tmp_root
        csv_path = Path(tmp_root) / "probe.csv"
        csv_path.write_text("y,x1\n0,1\n1,2\n", encoding="utf-8")
        layout = open_project(csv_path)
        result["project_dir"] = str(layout.project_dir)
        manifest = new_manifest(csv_path)
        write_manifest(layout.project_dir, manifest)
        reloaded = read_manifest(layout.project_dir)
        result["manifest_roundtrip"] = (
            reloaded.get("schema") == manifest.get("schema")
            and reloaded.get("project", {}).get("stem") == "probe"
        )
        run_id, run_dir = allocate_run_dir(layout.runs_dir)
        result["run_id"] = run_id
        result["run_dir_exists"] = run_dir.is_dir()
        result["ok"] = bool(
            result["manifest_roundtrip"]
            and result["run_dir_exists"]
            and re.match(RUN_ID_PATTERN, run_id) is not None
        )
    except Exception as exc:  # noqa: BLE001 - 探针必须给出结论而不是崩掉
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if tmp_root:
            import shutil

            shutil.rmtree(tmp_root, ignore_errors=True)
            result["cleaned_up"] = True
    return result


def register_tools(mcp: Any) -> Any:
    """把本阶段已冻结的工具注册到服务器实例上。"""

    @mcp.tool(
        name="server_probe",
        description=(
            "环境探针：返回包版本、平台信息、依赖版本矩阵、以及在临时目录里实地验证"
            "『mdl-<CSV名> 目录 + manifest 往返 + run 目录命名』是否可用。"
            "ok=false 时 error.message / deps 里给出缺什么。只读，不改动任何用户数据。"
        ),
    )
    def server_probe() -> Dict[str, Any]:
        deps = dependency_report()
        artifacts = artifacts_check()
        missing: List[str] = [pkg for pkg, info in deps.items() if info["error"]]
        warnings: List[str] = []
        if missing:
            warnings.append(f"缺少依赖：{', '.join(missing)}（按 INSTALL.md 安装后重试）")
        if not artifacts["ok"]:
            warnings.append("产物目录链路不可用，见 artifacts_check.error")
        ok = not missing and bool(artifacts["ok"])
        # 通用协议 §1 要求 data 承载本步结构化结果；扁平字段为早期已发布字段，
        # 保留以兼容既有客户端（两者内容一致，见 docs/tool-contract.md §5）。
        structured: Dict[str, Any] = {
            "version": __version__,
            "platform": platform_report(),
            "deps": deps,
            "artifacts_check": artifacts,
        }
        return {
            "ok": ok,
            "status": "ok" if ok else "error",
            "data": structured,
            "version": structured["version"],
            "platform": structured["platform"],
            "deps": structured["deps"],
            "artifacts_check": structured["artifacts_check"],
            "artifacts": [],
            "warnings": warnings,
            "error": None
            if ok
            else {
                "code": "environment_not_ready",
                "message": "；".join(warnings) or "环境未就绪",
                "hint": "运行 `risklab-mdl-doctor` 查看六项自检与可复制的修复命令",
            },
        }

    @mcp.tool(
        name="project_init",
        description=(
            "由 CSV 路径创建/复用建模项目目录 mdl-<CSV文件名>/ 与 manifest.json（建齐 8 个子目录）。"
            "已存在则复用、不覆盖历史产物。返回 project_dir / created / manifest。"
        ),
    )
    def project_init(csv_path: str) -> Dict[str, Any]:
        return _project_init(csv_path)

    @mcp.tool(
        name="dataset_load",
        description=(
            "读取 CSV：编码探测（UTF-8/GB18030/BOM）、列清单、dtype 推断、"
            "角色推断（id/time/split/target_candidates 各带理由，**推断不阻断**）。"
            "推断结果只作线索，目标字段需用户用 config.target 确认。"
        ),
    )
    def dataset_load(csv_path: str) -> Dict[str, Any]:
        return _dataset_load(csv_path)

    @mcp.tool(
        name="dataset_columns",
        description=(
            "列出项目数据的列与角色（dtype / role / nunique / missing_rate），"
            "供用户确认或修改角色；已确认的目标字段一并返回。"
        ),
    )
    def dataset_columns(project_dir: str) -> Dict[str, Any]:
        return _dataset_columns(project_dir)

    @mcp.tool(
        name="dataset_profile",
        description=(
            "样本画像：目标分布（含坏率/多分类占比）、逐列缺失率与同值率、数值列描述统计、"
            "特征类型构成、月份×标签交叉（有时间列时），并产出 data_analysis/ 下的 "
            "profile.json / describe.csv / target_distribution.csv / charts/*.png / index.html（单文件无外链）。"
            "未确认目标字段时返回 needs_user_input。"
        ),
    )
    def dataset_profile(
        project_dir: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return _dataset_profile(project_dir, config)

    @mcp.tool(
        name="feature_analyze",
        description=(
            "逐特征分箱 + WOE/IV 明细。config.method ∈ {quantile, equal_width, dt, custom}"
            "（custom 需给 config.edges），bins 为请求箱数；多分类按类 one-vs-rest 出逐类 IV + macro。"
            "非数值特征跳过并给原因。产物：feature_analysis/{iv_table.csv, woe_detail.csv, "
            "binning/<feature>.json, charts/*.png, index.html}。"
        ),
    )
    def feature_analyze(
        project_dir: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return _feature_analyze(project_dir, config)

    @mcp.tool(
        name="feature_charts",
        description=(
            "特征分析图表（读已有 iv_table.csv / woe_detail.csv，不重算 IV）："
            "IV 分布（含阈值线）、TOP 特征 WOE 箱图、相关性热力图，"
            "产物 feature_analysis/charts/*.png 与 feature_analysis/index.html。"
        ),
    )
    def feature_charts(
        project_dir: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return _feature_charts(project_dir, config)

    @mcp.tool(
        name="preprocess_check",
        description=(
            "入模前检查：逐特征判定缺失 / 非数值未编码 / 高基数类别 / 同值列 / 极端值 / "
            "哨兵值冲突，分成 must_fix（不处理则入模矩阵不可用）与 suggested（建议处理），"
            "产物 preprocess/check.json（{must_fix, suggested, by_feature}）。"
            "规则可在 config 中覆盖（missing_policy / high_missing_rate / high_cardinality_max）。"
        ),
    )
    def preprocess_check(
        project_dir: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return _preprocess_check(project_dir, config)

    @mcp.tool(
        name="preprocess_apply",
        description=(
            "落实预处理规格 spec={missing, encode, cap, binning}：缺失填充（mean/median/mode/"
            "sentinel/as_bin/drop）、编码（onehot/ordinal/woe/drop）、盖帽（absolute/quantile）、"
            "手动分箱（切点含 -inf/inf，缺失独立箱 __MISSING__）。"
            "产物 preprocess/{fill,encode,outlier,manual_bins}.json + processed.parquet（无 "
            "pyarrow 落 processed.csv）+ summary.json；非法规格返回 spec_invalid。"
        ),
    )
    def preprocess_apply(
        project_dir: str, spec: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return _preprocess_apply(project_dir, spec)

    @mcp.tool(
        name="dataset_split",
        description=(
            "样本划分：mode ∈ {none, random, stratified, field}。field 用受限表达式"
            "（== != > >= < <= in / not in / & / | / 括号，禁 eval，非法返回 expression_invalid），"
            "产物 split/split.json 含 mask 指纹 / 计数 / 实际占比 / seed；同 seed 可逐行复现。"
        ),
    )
    def dataset_split(
        project_dir: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return _dataset_split(project_dir, config)

    @mcp.tool(
        name="feature_prescreen",
        description=(
            "特征初筛（五道筛子：缺失率 → 同值率 → IV → PSI → 相关性）。阈值 config={"
            "missing_rate_max, iv_min, psi_max, corr_max, same_value_rate_max}"
            "（默认 0.5/0.02/0.10/0.8/0.95）。初筛只用 train（有 split/split.json 时），"
            "无时间列则跳过 PSI 并在 warnings 明示；多分类按 OVR 逐类 IV 取最大值。"
            "产物 feature_selection/prescreen.json（逐特征判定 + 淘汰原因 + 指标）。"
        ),
    )
    def feature_prescreen(
        project_dir: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return _feature_prescreen(project_dir, config)

    @mcp.tool(
        name="feature_refine",
        description=(
            "特征精筛：gain 迭代（依赖 lightgbm，缺库返回 dependency_missing）。"
            "候选池默认取初筛存活（无 prescreen 则取特征分析数值列），可用 config.candidates "
            "显式指定；支持 keep/drop/rounds/threshold/target_range。"
            "产物 feature_selection/refine.json + selected_features.txt（UTF-8 无 BOM）。"
        ),
    )
    def feature_refine(
        project_dir: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return _feature_refine(project_dir, config)

    @mcp.tool(
        name="model_catalog",
        description=(
            "算法 × 任务能力表（dt/lr/gbdt/xgb/lgb × binary/multiclass 共 10 格）："
            "每格的默认参数、默认调参空间（参数名/类型/范围/默认）与说明，"
            "并给出 xgboost / lightgbm 的安装状态。只读，不训练。"
        ),
    )
    def model_catalog() -> Dict[str, Any]:
        return _model_catalog()

    @mcp.tool(
        name="model_train",
        description=(
            "训练一个 run：config={algorithm, task, features?, preprocess?, split?, params?, seed?,"
            " num_boost_round?, early_stopping_rounds?}。数据优先取 preprocess/processed.parquet|csv，"
            "特征优先 config.features → selected_features.txt → 全部数值列，划分优先 split/split.json"
            "（无则内部 80/20 分层并 warnings）。产物 runs/<run_id>/{config,params,metrics}.json、"
            "model.pkl、importance.csv，并追加 manifest.runs。"
            "算法×任务不匹配 → task_algorithm_mismatch；缺库 → dependency_missing。"
        ),
    )
    def model_train(
        project_dir: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return _model_train(project_dir, config)

    @mcp.tool(
        name="model_tune",
        description=(
            "调参：mode=preview 返回搜索空间说明与预计试验数（不训练）；mode=execute 用 hyperopt "
            "落一个新 run（runs/<run_id>/tuning/{space.json,trials.csv}）。"
            "目标函数按 KS/AUC/LIFT/gap 加权；tune_on 只允许 val（禁 OOT，传其它值返回 config_invalid）。"
        ),
    )
    def model_tune(
        project_dir: str,
        config: Optional[Dict[str, Any]] = None,
        mode: str = "preview",
    ) -> Dict[str, Any]:
        return _model_tune(project_dir, config, mode)

    @mcp.tool(
        name="model_evaluate",
        description=(
            "在已有 run 上重算 train/val/test 指标（KS/AUC/Gini/LIFT；多分类 OVR 逐类 + macro），"
            "有时间列时按月算分数 PSI（基期合并池口径，无时间列跳过并 warnings），"
            "写 runs/<run_id>/metrics.json 与 evaluation/ 表（含 KS/LIFT 曲线 PNG）。"
        ),
    )
    def model_evaluate(
        project_dir: str,
        run_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return _model_evaluate(project_dir, run_id, config)

    @mcp.tool(
        name="run_list",
        description="列出 run 与指标摘要（按 run_id 升序）：run_id/算法/任务/状态/指标/是否确认。",
    )
    def run_list(project_dir: str) -> Dict[str, Any]:
        return _run_list(project_dir)

    @mcp.tool(
        name="run_compare",
        description=(
            "对比多个 run 的指标（缺省=全部）并按主指标 KS 标出最优；"
            "产物 reports/run_compare.csv，返回 data.best。"
        ),
    )
    def run_compare(
        project_dir: str, run_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        return _run_compare(project_dir, run_ids)

    @mcp.tool(
        name="model_confirm",
        description=(
            "确认闸门：默认 KS≥0.25 且 PSI≤0.10；不达标返回 threshold_not_met 并给差距值。"
            "达标写 final/model.pkl、final/confirm.json 与 manifest.confirmed_run。"
            "PSI 未计算（无时间列）时只校验 KS 并在 warnings 明示。"
        ),
    )
    def model_confirm(
        project_dir: str,
        run_id: Optional[str] = None,
        thresholds: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return _model_confirm(project_dir, run_id, thresholds)

    @mcp.tool(
        name="report_build",
        description=(
            "汇总报告（只读产物、不重算）：reports/结果汇总.xlsx（概览/模型指标/run对比/特征重要性"
            "四个非空 sheet）+ reports/过程报告.md。"
        ),
    )
    def report_build(project_dir: str) -> Dict[str, Any]:
        return _report_build(project_dir)

    return mcp


def build_server() -> Any:
    """构造服务器实例并注册工具（供 main / 测试复用）。"""
    mcp = make_server(SERVER_NAME, __version__)
    return register_tools(mcp)


def main(argv: Optional[List[str]] = None) -> int:
    """``risklab-mdl`` / ``python -m risklab_mdl`` 入口。"""
    parser = argparse.ArgumentParser(description=f"{SERVER_NAME} MCP Server（stdio）")
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse", "streamable-http"],
        help="传输方式；默认 stdio（MCP 客户端标准方式）",
    )
    args = parser.parse_args(argv)
    try:
        mcp = build_server()
    except McpSdkError as exc:
        print(f"[{SERVER_NAME}] MCP SDK 不可用：{exc}", file=sys.stderr)
        print(f"[{SERVER_NAME}] 修复：python -m pip install 'mcp>=2.0.0'", file=sys.stderr)
        return 1
    mcp.run(transport=args.transport)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
