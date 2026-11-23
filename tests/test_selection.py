"""初筛 / 精筛单测（契约 §3.2；验收 ⑤ 与 T5 接口）。

覆盖：
* ``feature_prescreen`` 五道筛子（缺失率 / 同值率 / IV / PSI / 相关性）与逐特征判定；
* 初筛只用 train（有 ``split/split.json`` 时）、无时间列跳过 PSI 并告警；
* 多分类按类 OVR 逐类 IV（``iv_by_class`` 落表）；
* ``feature_refine`` gain 迭代：候选池取自初筛存活、keep/drop、``selected_features.txt``
  为 UTF-8 无 BOM；缺切分/缺依赖等失败路径必须可见。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures.make_fixtures import write_all  # noqa: E402

from risklab_mdl.tools.dataset import (  # noqa: E402
    dataset_load,
    dataset_profile,
    project_init,
)
from risklab_mdl.tools.selection import feature_prescreen, feature_refine  # noqa: E402
from risklab_mdl.tools.split import dataset_split  # noqa: E402

BINARY_TARGET = {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}}
MULTICLASS_TARGET = {"column": "y", "task": "multiclass"}
ENVELOPE_KEYS = {"ok", "status", "data", "artifacts", "warnings", "error"}


def bootstrap(
    tmp_path: Path,
    name: str = "binary",
    *,
    split: bool = False,
    target: dict = BINARY_TARGET,
    profile: bool = True,
) -> tuple[Path, str]:
    paths = write_all(tmp_path / "data")
    csv = paths[name]
    init = project_init(str(csv))
    assert init["status"] == "ok", init
    assert dataset_load(str(csv))["status"] == "ok"
    project_dir = init["data"]["project_dir"]
    if profile:
        profiled = dataset_profile(project_dir, {"target": target})
        assert profiled["status"] == "ok", profiled
    if split:
        splitted = dataset_split(
            project_dir,
            {"mode": "random", "ratios": [0.7, 0.2, 0.1], "seed": 42, "target": target},
        )
        assert splitted["status"] == "ok", splitted
    return csv, project_dir


def load_json(project_dir: str, rel: str) -> dict:
    return json.loads((Path(project_dir) / rel).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# feature_prescreen
# --------------------------------------------------------------------------- #
def test_prescreen_binary_uses_five_stages_on_train(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path, split=True)
    payload = feature_prescreen(project_dir, {"target": BINARY_TARGET})
    assert set(payload) == ENVELOPE_KEYS
    assert payload["status"] == "ok", payload
    assert payload["artifacts"] == ["feature_selection/prescreen.json"]

    data = payload["data"]
    assert "train" in data["data_scope"]["source"]
    assert data["data_scope"]["rows_used"] < data["data_scope"]["rows_total"], (
        "有 split.json 时初筛只能用 train 子集"
    )
    survived = set(data["survived"])
    assert {"f_signal", "f_weak", "f_missing"} <= survived, data["survived"]
    assert "f_const" not in survived, "常量列应被同值率筛掉"
    assert "f_dup" not in survived, "高相关副本应被相关性筛掉"
    assert "f_text" in data["non_numeric"], "非数值列不参与初筛，但要列出"

    dropped = {item["feature"]: item["stage"] for item in data["dropped"]}
    assert dropped.get("f_dup") == "correlation"
    assert dropped.get("f_const") == "same_value_rate"
    assert dropped.get("f_text") == "type_check"

    doc = load_json(project_dir, "feature_selection/prescreen.json")
    assert doc["schema"] == "risklab-mdl.prescreen/1"
    assert doc["stages"] == [
        "missing_rate",
        "same_value_rate",
        "iv",
        "psi",
        "correlation",
    ]
    records = {record["feature"]: record for record in doc["features"]}
    assert records["f_dup"]["dropped_stage"] == "correlation"
    assert "corr" in records["f_dup"]["reason"].lower() or "0.8" in records["f_dup"]["reason"]
    assert records["f_signal"]["survived"] is True
    assert records["f_signal"]["metrics"]["iv"] is not None
    assert doc["summary"]["stage_dropped"]["correlation"] == 1
    assert doc["summary"]["n_survived"] == data["n_survived"]

    manifest = json.loads(
        (Path(project_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["stages"]["feature_selection"]["done"] is False, "精筛未跑，阶段不能算完成"
    assert "初筛" in manifest["stages"]["feature_selection"]["summary"]


def test_prescreen_without_split_uses_full_data_with_note(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path, split=False)
    payload = feature_prescreen(project_dir, {"target": BINARY_TARGET})
    assert payload["status"] == "ok", payload
    scope = payload["data"]["data_scope"]
    assert scope["rows_used"] == scope["rows_total"]
    assert "全量" in scope["source"]
    assert any("全量" in w for w in payload["warnings"]), payload["warnings"]


def test_prescreen_no_time_column_skips_psi(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path, name="no_time")
    payload = feature_prescreen(project_dir, {"target": BINARY_TARGET})
    assert payload["status"] == "ok", payload
    psi = payload["data"]["psi"]
    assert psi["enabled"] is False
    assert "PSI" in psi["note"]
    assert any("PSI" in w and "跳过" in w for w in payload["warnings"]), payload["warnings"]
    # 无时间列时缺失率筛子照常工作（f_sparse 55% > 50%）
    dropped = {item["feature"]: item["stage"] for item in payload["data"]["dropped"]}
    assert dropped.get("f_sparse") == "missing_rate"


def test_prescreen_multiclass_uses_ovr_iv(tmp_path: Path) -> None:
    _, project_dir = bootstrap(
        tmp_path, name="multiclass", target=MULTICLASS_TARGET
    )
    payload = feature_prescreen(project_dir, {"target": MULTICLASS_TARGET})
    assert payload["status"] == "ok", payload
    assert payload["data"]["task"] == "multiclass"
    assert any("one-vs-rest" in w or "OVR" in w for w in payload["warnings"])
    doc = load_json(project_dir, "feature_selection/prescreen.json")
    records = {record["feature"]: record for record in doc["features"]}
    assert records["f1"]["iv_by_class"], "多分类必须落逐类 IV"
    assert set(records["f1"]["iv_by_class"]) == {"0", "1", "2"}


def test_prescreen_thresholds_are_validated(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    for config in (
        {"target": BINARY_TARGET, "iv_min": 1.5},
        {"target": BINARY_TARGET, "missing_rate_max": 0},
        {"target": BINARY_TARGET, "corr_max": "abc"},
    ):
        payload = feature_prescreen(project_dir, config)
        assert payload["status"] == "error", payload
        assert payload["error"]["code"] == "config_invalid", payload["error"]


def test_prescreen_correlation_threshold_is_configurable(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    default = feature_prescreen(project_dir, {"target": BINARY_TARGET})
    assert default["status"] == "ok", default
    assert "f_dup" not in default["data"]["survived"]
    relaxed = feature_prescreen(
        project_dir, {"target": BINARY_TARGET, "corr_max": 0.999}
    )
    assert relaxed["status"] == "ok", relaxed
    assert "f_dup" in relaxed["data"]["survived"], relaxed["data"]["survived"]


def test_prescreen_requires_target(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path, profile=False)
    payload = feature_prescreen(project_dir)
    assert payload["status"] == "needs_user_input", payload
    assert payload["data"]["question"]["id"] == "dataset_confirm_target"


def test_prescreen_all_text_features_returns_features_empty(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "y": [0, 1] * 50,
            "id": [f"U{i:03d}" for i in range(100)],
            "channel": ["a", "b"] * 50,
            "segment": ["x", "y"] * 50,
        }
    )
    csv = tmp_path / "all_text.csv"
    frame.to_csv(csv, index=False, encoding="utf-8")
    init = project_init(str(csv))
    project_dir = init["data"]["project_dir"]
    assert dataset_load(str(csv))["status"] == "ok"
    assert dataset_profile(project_dir, {"target": BINARY_TARGET})["status"] == "ok"
    payload = feature_prescreen(project_dir, {"target": BINARY_TARGET})
    assert payload["status"] == "error", payload
    assert payload["error"]["code"] == "features_empty"


# --------------------------------------------------------------------------- #
# feature_refine
# --------------------------------------------------------------------------- #
def test_refine_uses_prescreen_survivors(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path, split=True)
    prescreen = feature_prescreen(project_dir, {"target": BINARY_TARGET})
    assert prescreen["status"] == "ok", prescreen
    survivors = prescreen["data"]["survived"]

    payload = feature_refine(project_dir, {"rounds": 5})
    assert set(payload) == ENVELOPE_KEYS
    assert payload["status"] == "ok", payload
    data = payload["data"]
    assert data["n_candidates"] == len(survivors)
    assert data["candidate_source"].startswith("feature_selection/prescreen.json")
    assert set(data["features"]) <= set(survivors), (
        f"精筛特征必须来自初筛存活集：{data['features']} vs {survivors}"
    )
    assert "feature_selection/refine.json" in payload["artifacts"]
    assert "feature_selection/selected_features.txt" in payload["artifacts"]

    doc = load_json(project_dir, "feature_selection/refine.json")
    assert doc["schema"] == "risklab-mdl.refine/1"
    assert doc["n_candidates"] == len(survivors)
    assert doc["features"] == data["features"]
    assert doc["gain_top"], "应落 gain 排名（可复核）"
    assert doc["split"]["train_rows"] > 0 and doc["split"]["val_rows"] > 0

    selected_path = Path(project_dir) / "feature_selection" / "selected_features.txt"
    raw = selected_path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "selected_features.txt 必须是 UTF-8 无 BOM"
    lines = selected_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines == data["features"]

    manifest = json.loads(
        (Path(project_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["stages"]["feature_selection"]["done"] is True
    assert "精筛" in manifest["stages"]["feature_selection"]["summary"]


def test_refine_accepts_explicit_candidates_and_keep_drop(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path, split=True)
    payload = feature_refine(
        project_dir,
        {
            "candidates": ["f_signal", "f_weak", "f_noise", "f_missing"],
            "keep": ["f_noise"],
            "drop": ["f_weak"],
            "rounds": 5,
        },
    )
    assert payload["status"] == "ok", payload
    data = payload["data"]
    assert data["candidate_source"] == "config.candidates"
    assert "f_noise" in data["features"], "白名单必须无条件保留"
    assert "f_weak" not in data["features"], "黑名单必须无条件剔除"


def test_refine_rejects_target_as_candidate(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path, split=True)
    payload = feature_refine(
        project_dir, {"candidates": ["y", "f_signal"], "rounds": 3}
    )
    assert payload["status"] == "error", payload
    assert payload["error"]["code"] == "config_invalid"
    assert "禁止入模" in payload["error"]["message"]


def test_refine_requires_validation_split(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path, split=False)
    dataset_split(project_dir, {"mode": "none", "target": BINARY_TARGET})
    payload = feature_refine(project_dir, {"rounds": 3})
    assert payload["status"] == "error", payload
    assert payload["error"]["code"] == "config_invalid"
    assert "val" in payload["error"]["message"]


def test_refine_without_split_or_time_column_fails_visibly(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path, name="no_time")
    payload = feature_refine(project_dir, {"rounds": 3})
    assert payload["status"] == "error", payload
    assert payload["error"]["code"] == "config_invalid"
    assert "切分" in payload["error"]["message"]


def test_refine_falls_back_when_no_prescreen(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path, split=True)
    payload = feature_refine(project_dir, {"rounds": 5})
    assert payload["status"] == "ok", payload
    assert "全部数值列" in payload["data"]["candidate_source"]
    assert any("建议先跑初筛" in w for w in payload["warnings"]), payload["warnings"]


def test_refine_multiclass_uses_multiclass_objective(tmp_path: Path) -> None:
    """多分类精筛走 multiclass 目标（轮次表 valid_auc 允许为 null，不伪造）。"""
    _, project_dir = bootstrap(
        tmp_path, name="multiclass", target=MULTICLASS_TARGET, split=True
    )
    payload = feature_refine(
        project_dir,
        {"rounds": 5, "candidates": ["f1", "f2", "f3", "f_noise"]},
    )
    assert payload["status"] == "ok", payload
    doc = load_json(project_dir, "feature_selection/refine.json")
    assert doc["task"] == "multiclass"
    assert doc["params"]["objective"] == "multiclass"
    assert doc["params"]["num_class"] == 3
    assert doc["features"], "多分类精筛也要产出特征清单"
    for round_row in doc["summary"]["rounds"]:
        assert round_row["valid_auc"] is None, "多分类用 multi_logloss，不应伪造 AUC"
