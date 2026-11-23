"""T6 验收 ②：模拟一次 agent 会话——按 skill 的 9 环节顺序把 15 次工具调用真跑一遍。

★ 这条用例在测什么
------------------
不是测单个函数，而是测 **skill 里写的那条流程真的能跑通**：
复制样例 CSV 到临时目录（旧项目只读，绝不原地跑）→ 按 `skill/SKILL.md` §5 的环节顺序
直接调用工具函数 → 每步 `status=ok` → 产物齐全 → `final/confirm.json` 落盘 → 报告生成。

★ 为什么确认阈值用宽松值
------------------------
本机实测（`sample_2000.csv`，lgb，60 特征，7:2:1 分层）：
``val KS = 0.402``、分数 PSI 最差月 ``2026-04 = 0.115``。
默认闸门是 KS ≥ 0.25 且 **PSI ≤ 0.10** —— PSI 0.115 会**故意**卡住（这是工具的正确行为，
`tests/test_confirm.py` 专门测它）。所以本用例用宽松阈值 ``{ks_min: 0.05, psi_max: 0.50}``
保证「通过路径」可断言，同时**先**用严阈值跑一次确认闸门确实会拦下（失败必须可见）。

★ 顺序说明
----------
`feature_prescreen` / `feature_refine` 排在 `dataset_split` 之前，此时初筛用**全量**数据
（工具会在 `warnings` 里明说）。这是 `skill/SKILL.md` §5 主干顺序下的已知口径，
用例把它当**预期**并在断言里记录，而不是当成 bug。
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from risklab_mdl.artifacts.manifest import STAGES
from risklab_mdl.tools.artifact import report_build, run_compare, run_list
from risklab_mdl.tools.dataset import (
    dataset_columns,
    dataset_load,
    dataset_profile,
    project_init,
)
from risklab_mdl.tools.feature import feature_analyze
from risklab_mdl.tools.model import (
    model_catalog,
    model_confirm,
    model_evaluate,
    model_train,
    model_tune,
)
from risklab_mdl.tools.preprocess import preprocess_apply, preprocess_check
from risklab_mdl.tools.selection import feature_prescreen, feature_refine
from risklab_mdl.tools.split import dataset_split

#: 旧项目样例数据（只读引用；不存在时跳过整条用例）。
SAMPLE_CSV = (
    Path(__file__).resolve().parents[2] / "risklab-mcp-main" / "examples" / "sample_2000.csv"
)

TARGET = {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}}

#: 确认闸门：宽松阈值（见模块 docstring）。本机实测 val KS 0.402、最大 PSI 0.115。
LOOSE_THRESHOLDS = {"ks_min": 0.05, "psi_max": 0.50}
#: 严阈值：只用来验证「不达标会被拦下并给出差距值」。
STRICT_THRESHOLDS = {"ks_min": 0.99, "psi_max": 0.01}

#: 9 环节 → 该环节覆盖的工具调用（与 skill/SKILL.md §5 的表一致）。
STEPS: List[tuple] = [
    ("环节1 数据上传与项目引导", "project_init"),
    ("环节1 数据上传与项目引导", "dataset_load"),
    ("环节1 数据上传与项目引导", "dataset_columns"),
    ("环节2 样本数据分析", "dataset_profile"),
    ("环节3 特征分析", "feature_analyze"),
    ("环节5 特征筛选", "feature_prescreen"),
    ("环节6 入模前检查与特征精调", "feature_refine"),
    ("环节6 入模前检查与特征精调", "preprocess_check"),
    ("环节6 入模前检查与特征精调", "preprocess_apply"),
    ("环节7 样本划分", "dataset_split"),
    ("环节4 模型选型", "model_catalog"),
    ("环节8 模型训练", "model_train"),
    ("环节9 模型结果确认", "model_evaluate"),
    ("环节9 模型结果确认", "model_confirm"),
    ("环节9 模型结果确认", "report_build"),
]

#: 每个环节至少要出现的产物目录（相对项目目录）。
REQUIRED_DIRS = [
    "data_analysis",
    "feature_analysis",
    "feature_selection",
    "preprocess",
    "split",
    "runs",
    "final",
    "reports",
]


def _ok(payload: Dict[str, Any], step: str) -> Dict[str, Any]:
    """断言三态自洽 + `status=ok`，并把失败原文抛出去（失败必须可见）。"""
    assert isinstance(payload, dict), f"{step}: 工具必须返回 dict，实际 {type(payload).__name__}"
    assert set(payload) == {"ok", "status", "data", "artifacts", "warnings", "error"}, (
        f"{step}: 信封键不符 {sorted(payload)}"
    )
    assert payload["status"] == "ok" and payload["ok"] is True, (
        f"{step}: 期望 status=ok，实际 status={payload['status']} / error={payload['error']} "
        f"/ warnings={payload['warnings']}"
    )
    assert payload["error"] is None, f"{step}: status=ok 时 error 必须为 None"
    return payload["data"]


def _assert_artifacts_exist(project_dir: Path, payload: Dict[str, Any], step: str) -> None:
    """`artifacts` 里的相对路径必须真的落在磁盘上（POSIX 分隔符）。"""
    for rel in payload["artifacts"]:
        assert "\\" not in rel, f"{step}: artifacts 必须是 POSIX 分隔符：{rel}"
        path = project_dir / rel
        assert path.exists(), f"{step}: 声明的产物不存在：{path}"


@pytest.mark.skipif(not SAMPLE_CSV.is_file(), reason="样例数据 sample_2000.csv 不在本机")
def test_agent_session_runs_through_nine_steps(tmp_path: Path) -> None:
    # ---------- 准备：CSV 复制到临时目录（旧项目只读） ----------
    work = tmp_path / "work"
    work.mkdir()
    csv_path = work / "sample_2000.csv"
    shutil.copy2(SAMPLE_CSV, csv_path)
    assert csv_path.is_file() and SAMPLE_CSV.is_file()
    assert csv_path.parent != SAMPLE_CSV.parent, "绝对不能在旧项目目录里原地跑"

    timings: Dict[str, float] = {}
    started = time.time()
    payloads: Dict[str, Dict[str, Any]] = {}
    covered: List[str] = []
    state: Dict[str, Any] = {"project": None}

    def run(step: str, fn, *args, **kwargs) -> Dict[str, Any]:
        t0 = time.time()
        payload = fn(*args, **kwargs)
        timings[step] = round(time.time() - t0, 2)
        payloads[step] = payload
        data = _ok(payload, step)
        if state["project"] is not None:
            _assert_artifacts_exist(state["project"], payload, step)
        covered.append(step)
        return data

    # ---------- 环节 1：数据上传与项目引导 ----------
    init = run("project_init", project_init, str(csv_path))
    project_dir = Path(init["project_dir"])
    state["project"] = project_dir
    assert project_dir.is_dir(), init
    assert project_dir.name == "mdl-sample_2000", project_dir
    assert init["created"] is True and Path(init["manifest_path"]).is_file()

    load = run("dataset_load", dataset_load, str(csv_path))
    assert load["rows"] == 2000 and load["column_count"] == 696, (load["rows"], load["column_count"])
    assert load["encoding"].lower().replace("_", "-").startswith("utf-8"), load["encoding"]
    assert load["roles"]["id"] == "sample_id" and load["roles"]["time"] == "month"
    assert "y" in load["roles"]["target_candidates"], load["roles"]

    cols = run("dataset_columns", dataset_columns, str(project_dir))
    assert cols["confirmed_target"] is None, "列清单阶段目标字段还没确认"
    names = {item["name"]: item for item in cols["columns"]}
    assert names["y"]["role"] == "target"
    assert names["sample_id"]["role"] == "id" and names["month"]["role"] == "time"

    # ---------- 环节 2：样本数据分析（目标字段在这里拍板） ----------
    profile = run("dataset_profile", dataset_profile, str(project_dir), {"target": TARGET})
    assert profile["shape"] == {"rows": 2000, "columns": 696}
    dist = profile["target_distribution"]
    assert dist["counts"] == {"0": 1965, "1": 35}, dist["counts"]
    assert dist["bad_label"] == "1" and dist["bad_rate"] == pytest.approx(0.0175, abs=1e-6)
    assert profile["has_time_column"] is True and profile["chart_count"] >= 3
    for name in ("profile.json", "describe.csv", "target_distribution.csv", "index.html"):
        assert (project_dir / "data_analysis" / name).is_file(), name
    assert (project_dir / "data_analysis" / "charts" / "target_distribution.png").is_file()

    # ---------- 环节 3：特征分析 ----------
    analysis = run(
        "feature_analyze", feature_analyze, str(project_dir), {"method": "quantile", "bins": 8}
    )
    assert analysis["feature_count"] == 692 and analysis["analyzed_count"] == 691
    assert analysis["skipped"] and analysis["skipped"][0]["feature"] == "f_0423"
    assert analysis["iv_top"] and analysis["iv_top"][0]["iv"] > 0.3
    assert (project_dir / "feature_analysis" / "iv_table.csv").is_file()
    assert (project_dir / "feature_analysis" / "woe_detail.csv").is_file()
    assert (project_dir / "feature_analysis" / "index.html").is_file()
    assert len(list((project_dir / "feature_analysis" / "binning").glob("*.json"))) == 691

    # ---------- 环节 5：特征筛选（此时还没有 split.json ⇒ 全量口径，工具会告警） ----------
    prescreen = run(
        "feature_prescreen",
        feature_prescreen,
        str(project_dir),
        {"iv_min": 0.02, "psi_max": 0.25, "corr_max": 0.8},
    )
    assert prescreen["n_input"] == 691 and 0 < prescreen["n_survived"] < prescreen["n_input"]
    assert prescreen["survived"] and prescreen["dropped"]
    assert prescreen["data_scope"]["source"].startswith("未划分"), prescreen["data_scope"]
    assert any("split/split.json" in w for w in payloads["feature_prescreen"]["warnings"]), (
        "全量口径必须显式告警（数据穿越风险）"
    )
    assert (project_dir / "feature_selection" / "prescreen.json").is_file()

    # ---------- 环节 6：入模前检查与特征精调 ----------
    refine = run(
        "feature_refine",
        feature_refine,
        str(project_dir),
        {"scheme": "gain", "rounds": 3, "threshold": 0.05},
    )
    assert refine["candidate_source"].startswith("feature_selection/prescreen.json")
    assert refine["n_selected"] >= 1 and len(refine["features"]) == refine["n_selected"]
    assert refine["disposition"] in {
        "converged",
        "truncated_high",
        "relaxed_converged",
        "relaxed_truncated",
        "relaxed_failed",
    }, refine["disposition"]
    selected = (project_dir / "feature_selection" / "selected_features.txt").read_bytes()
    assert not selected.startswith(b"\xef\xbb\xbf"), "selected_features.txt 必须无 BOM"
    assert selected.decode("utf-8").split() == refine["features"]

    check = run("preprocess_check", preprocess_check, str(project_dir))
    assert check["feature_count"] >= 600 and check["must_fix_count"] > 0
    assert (project_dir / "preprocess" / "check.json").is_file()

    spec = {
        "missing": {"f_0001": "median", "f_0002": "median", "f_0010": "median"},
        "cap": {"f_0003": {"strategy": "quantile", "lower_quantile": 0.01, "upper_quantile": 0.99}},
        "binning": {"f_0100": {"edges": ["-inf", 5, 15, "inf"], "missing_as_bin": True}},
    }
    applied = run("preprocess_apply", preprocess_apply, str(project_dir), spec)
    assert applied["rows_in"] == applied["rows_out"] == 2000, "本 spec 不删行，掩码不会错位"
    assert applied["columns_in"] == applied["columns_out"] == 696
    readiness = applied["model_readiness"]
    assert readiness["lr_ready"] is False, "还有非数值列与缺失 ⇒ LR 不可用，必须如实报告"
    assert "f_0423" in readiness["non_numeric_columns"]
    processed = project_dir / applied["processed_path"]
    assert processed.is_file(), applied["processed_path"]
    # pyarrow 装了落 parquet，没装落 csv —— 都要能被后续步骤读到
    assert processed.suffix in {".parquet", ".csv"}, processed.suffix
    assert (project_dir / "preprocess" / "summary.json").is_file()
    assert json.loads((project_dir / "preprocess" / "manual_bins.json").read_text("utf-8"))

    # ---------- 环节 7：样本划分 ----------
    split = run(
        "dataset_split",
        dataset_split,
        str(project_dir),
        {
            "mode": "stratified",
            "ratios": [0.6, 0.2, 0.2],
            "seed": 42,
            "stratify_by": "y",
            "time_column": "month",
        },
    )
    assert split["counts"] == {"train": 1200, "val": 400, "test": 400}, split["counts"]
    assert split["seed"] == 42 and len(split["fingerprint"]) == 16
    for subset in ("train", "val", "test"):
        share = split["label_distribution"][subset]["shares"]["1"]
        assert share == pytest.approx(0.0175, abs=1e-3), (subset, share)
    split_doc = json.loads((project_dir / "split" / "split.json").read_text("utf-8"))
    assert split_doc["fingerprint"] == split["fingerprint"]

    # ---------- 环节 4：模型选型 ----------
    catalog = run("model_catalog", model_catalog)
    algorithms = catalog["catalog"]["algorithms"]
    assert sorted(algorithms) == ["dt", "gbdt", "lgb", "lr", "xgb"]
    for name in algorithms:
        assert algorithms[name]["tasks"] == ["binary", "multiclass"]
        assert algorithms[name]["default_params"] and algorithms[name]["default_space"]
    assert catalog["catalog"]["matrix_size"] == 10
    assert catalog["available"]["lgb"]["installed"] is True, "本机应装好 lightgbm"

    # ---------- 环节 8：训练（先预览调参空间，再训练） ----------
    preview = run(
        "model_tune", model_tune, str(project_dir), {"algorithm": "lgb", "task": "binary"}, "preview"
    )
    assert preview["mode"] == "preview" and preview["tune_on"] == "val"
    assert preview["space"] and preview["space_table"] and preview["expected_trials"] == 20
    assert preview["sample_sizes"] == {"train": 1200, "val": 400, "test": 400}

    train = run(
        "model_train", model_train, str(project_dir), {"algorithm": "lgb", "task": "binary", "seed": 42}
    )
    run_id = train["run_id"]
    assert len(run_id) == 15 and run_id[8] == "T", run_id
    assert train["task"] == "binary" and train["bad_class"] == 1
    assert train["feature_source"].startswith("feature_selection/selected_features.txt")
    assert train["data_source"].startswith("preprocess/")
    assert train["split"]["fingerprint"] == split["fingerprint"], "训练必须复算同一份掩码"
    for subset in ("train", "val", "test"):
        block = train["metrics"][subset]
        assert block["status"] == "ok", (subset, block)
        assert 0.0 <= block["ks"] <= 1.0 and 0.0 <= block["auc"] <= 1.0
        assert block["n"] == split["counts"][subset]
    # 宽表 + 35 个坏样本 ⇒ train 明显高于 val，这是数据的真实结论，不是失败
    assert train["metrics"]["train"]["auc"] > train["metrics"]["val"]["auc"]
    for name in ("config.json", "params.json", "metrics.json", "model.pkl", "importance.csv"):
        assert (project_dir / "runs" / run_id / name).is_file(), name

    # ---------- 环节 9：评估 → 闸门 → 报告 ----------
    evaluate = run("model_evaluate", model_evaluate, str(project_dir), run_id)
    psi = evaluate["psi"]
    assert psi["enabled"] is True and psi["max_month"], psi
    assert psi["by_month"] and psi["time_column"] == "month"
    assert evaluate["tables"]["ks_table"] > 0 and evaluate["tables"]["lift_table"] > 0
    assert len(evaluate["charts"]) >= 2
    metrics_doc = json.loads((project_dir / "runs" / run_id / "metrics.json").read_text("utf-8"))
    assert metrics_doc["psi"]["max_psi"] == pytest.approx(psi["max_psi"])

    listed = run("run_list", run_list, str(project_dir))
    assert [item["run_id"] for item in listed["runs"]] == [run_id]
    assert listed["runs"][0]["status"] == "evaluated"
    assert listed["confirmed_run"] is None

    # 9.1 严阈值必被拦下（失败必须可见：给差距值，不写 final/）
    strict = model_confirm(str(project_dir), run_id, STRICT_THRESHOLDS)
    assert strict["status"] == "error", strict
    assert strict["ok"] is False and strict["error"]["code"] == "threshold_not_met", strict["error"]
    assert "confirmed" not in strict["data"], "不通过载荷不得出现 confirmed 字段"
    assert strict["data"]["failed_checks"], "必须给出未通过的检查项"
    # 差距值口径：ks_gap = ks_min - ks（正=还差多少）、psi_gap = psi - psi_max（正=超出多少）
    assert strict["data"]["ks_gap"] == pytest.approx(
        STRICT_THRESHOLDS["ks_min"] - strict["data"]["ks"], abs=1e-6
    )
    assert strict["data"]["psi_gap"] == pytest.approx(
        strict["data"]["psi"] - STRICT_THRESHOLDS["psi_max"], abs=1e-6
    )
    assert strict["data"]["ks_gap"] > 0 and strict["data"]["psi_gap"] > 0
    assert not (project_dir / "final" / "confirm.json").exists(), "不达标不得写 final/"
    assert not (project_dir / "final" / "model.pkl").exists()

    # 9.2 宽松阈值通过（见模块 docstring：实测 val KS 0.402、最大 PSI 0.115）
    confirm = run(
        "model_confirm", model_confirm, str(project_dir), run_id, LOOSE_THRESHOLDS
    )
    assert confirm["confirmed"] is True and confirm["failed_checks"] == []
    assert confirm["thresholds"] == LOOSE_THRESHOLDS, confirm["thresholds"]
    assert confirm["ks_source"] == "val", confirm["ks_source"]
    assert confirm["ks"] >= LOOSE_THRESHOLDS["ks_min"]
    assert confirm["psi"] <= LOOSE_THRESHOLDS["psi_max"]
    confirm_doc = json.loads((project_dir / "final" / "confirm.json").read_text("utf-8"))
    assert confirm_doc["run_id"] == run_id
    assert (project_dir / "final" / "model.pkl").is_file()
    manifest = json.loads((project_dir / "manifest.json").read_text("utf-8"))
    assert manifest["confirmed_run"] == run_id
    assert manifest["project"]["target"]["column"] == "y"
    assert all(manifest["stages"][stage]["done"] for stage in STAGES), manifest["stages"]

    # 9.3 报告
    report = run("report_build", report_build, str(project_dir))
    assert report["n_runs"] == 1 and report["best_run"] == run_id
    assert set(report["sheets"]) == {"概览", "模型指标", "run对比", "特征重要性"}, report["sheets"]
    assert all(count > 0 for count in report["sheets"].values()), report["sheets"]
    assert (project_dir / "reports" / "结果汇总.xlsx").is_file()
    assert (project_dir / "reports" / "过程报告.md").is_file()

    compare = run("run_compare", run_compare, str(project_dir))
    assert compare["best"] == run_id and compare["primary_metric"] == "ks"
    assert (project_dir / "reports" / "run_compare.csv").is_file()

    # ---------- 全局：9 环节全覆盖 + 产物目录齐全 ----------
    assert len(STEPS) == 15, "STEPS 记录的是任务要求的 15 次调用"
    required = {tool for _, tool in STEPS}
    assert len(required) == 15
    assert len({stage for stage, _ in STEPS}) == 9, "STEPS 必须覆盖 9 个环节"
    assert required <= set(covered), f"这些要求的工具没有跑到：{sorted(required - set(covered))}"
    # 另外还跑了 model_tune(preview) / run_list / run_compare（同样属于 skill 流程）
    assert len(set(covered)) >= 15, sorted(set(covered))
    for name in REQUIRED_DIRS:
        directory = project_dir / name
        assert directory.is_dir(), f"缺少产物目录：{name}"
        assert any(directory.rglob("*")), f"产物目录为空：{name}"

    total = round(time.time() - started, 2)
    # 本机实测约 60 s（feature_analyze ≈18 s、feature_prescreen ≈26 s 占大头）
    assert total < 600, f"全链路超出 600 s 预算：{total}s"
    print(
        "\n[e2e] 9 环节 / 15 次工具调用全部 status=ok；总耗时 %.2fs；逐步骤耗时：%s"
        % (total, json.dumps(timings, ensure_ascii=False))
    )
