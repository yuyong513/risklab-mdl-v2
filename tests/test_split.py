"""``dataset_split`` 单测（契约 §3.3；验收 ①）。

覆盖：同 seed 可复现（指纹 + 逐行掩码）、比例误差、分层各类占比偏差、
mode=none / random / stratified / field、产物结构与 manifest 阶段、错误码。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures.make_fixtures import write_all  # noqa: E402

from risklab_mdl.core.split import SplitSpec, split_frame  # noqa: E402
from risklab_mdl.tools.dataset import dataset_load, project_init  # noqa: E402
from risklab_mdl.tools.split import dataset_split  # noqa: E402

ENVELOPE_KEYS = {"ok", "status", "data", "artifacts", "warnings", "error"}


def bootstrap(tmp_path: Path, name: str = "binary") -> tuple[Path, str]:
    paths = write_all(tmp_path / "data")
    csv = paths[name]
    init = project_init(str(csv))
    assert init["status"] == "ok", init
    assert dataset_load(str(csv))["status"] == "ok"
    return csv, init["data"]["project_dir"]


def load_split(project_dir: str) -> dict:
    return json.loads(
        (Path(project_dir) / "split" / "split.json").read_text(encoding="utf-8")
    )


# --------------------------------------------------------------------------- #
# 验收 ①：同 seed 可复现
# --------------------------------------------------------------------------- #
def test_random_same_seed_same_fingerprint_and_masks(tmp_path: Path) -> None:
    csv, project_dir = bootstrap(tmp_path)
    config = {
        "mode": "random",
        "ratios": [0.7, 0.2, 0.1],
        "seed": 42,
        "target": {"column": "y", "task": "binary"},
    }

    first = dataset_split(project_dir, config)
    second = dataset_split(project_dir, config)
    assert set(first) == ENVELOPE_KEYS
    assert first["status"] == "ok" and second["status"] == "ok", (first, second)
    assert first["data"]["fingerprint"] == second["data"]["fingerprint"]
    assert first["artifacts"] == ["split/split.json"]

    rows = len(pd.read_csv(csv, encoding="utf-8"))
    counts = first["data"]["counts"]
    assert sum(counts.values()) == rows, "三个子集必须覆盖全部行且不重叠"
    assert abs(first["data"]["ratios"]["train"] - 0.7) < 0.01, "random 比例误差须 <1%"

    # 逐行掩码一致（直接走 core 纯函数，两次调用完全相同）
    frame = pd.read_csv(csv, encoding="utf-8")
    spec = SplitSpec.from_config(config)
    a = split_frame(frame, spec)
    b = split_frame(frame, spec)
    for subset in ("train", "val", "test"):
        assert np.array_equal(a.mask(subset).to_numpy(), b.mask(subset).to_numpy()), (
            f"{subset} 掩码两次不一致"
        )
    assert a.fingerprint == first["data"]["fingerprint"]

    doc = load_split(project_dir)
    assert doc["schema"] == "risklab-mdl.split/1"
    assert doc["fingerprint"] == first["data"]["fingerprint"]
    assert doc["seed"] == 42
    assert doc["spec"]["ratios"] == [0.7, 0.2, 0.1]
    assert doc["label_distribution"]["train"]["count"] == counts["train"]
    assert doc["label_distribution"]["train"]["counts"]

    manifest = json.loads(
        (Path(project_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["stages"]["split"]["done"] is True
    assert manifest["stages"]["split"]["path"] == "split/"


def test_random_different_seed_changes_fingerprint(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    a = dataset_split(project_dir, {"mode": "random", "seed": 1})
    b = dataset_split(project_dir, {"mode": "random", "seed": 2})
    assert a["status"] == b["status"] == "ok"
    assert a["data"]["fingerprint"] != b["data"]["fingerprint"]


# --------------------------------------------------------------------------- #
# 验收 ①：分层各类占比偏差 < 2%
# --------------------------------------------------------------------------- #
def test_stratified_class_shares_within_two_percent(tmp_path: Path) -> None:
    csv, project_dir = bootstrap(tmp_path)
    frame = pd.read_csv(csv, encoding="utf-8")
    overall = frame["y"].astype(str).value_counts(normalize=True)
    payload = dataset_split(
        project_dir,
        {
            "mode": "stratified",
            "ratios": [0.6, 0.2, 0.2],
            "seed": 7,
            "stratify_by": "y",
            "target": {"column": "y", "task": "binary"},
        },
    )
    assert payload["status"] == "ok", payload
    distribution = payload["data"]["label_distribution"]
    for subset in ("train", "val", "test"):
        assert distribution[subset]["count"] > 0
        for label, share in distribution[subset]["shares"].items():
            assert abs(share - float(overall[label])) < 0.02, (
                f"{subset} 的类 {label} 占比 {share} 偏离总体 {overall[label]} 超过 2%"
            )
    assert sum(payload["data"]["counts"].values()) == len(frame)


def test_stratified_uses_target_by_default(tmp_path: Path) -> None:
    """未给 stratify_by 时用目标列分层；无目标时给出可读错误。"""
    _, project_dir = bootstrap(tmp_path)
    no_target = dataset_split(project_dir, {"mode": "stratified", "seed": 3})
    assert no_target["status"] == "error"
    assert no_target["error"]["code"] == "config_invalid"
    assert "stratified" in no_target["error"]["message"]

    with_target = dataset_split(
        project_dir,
        {
            "mode": "stratified",
            "seed": 3,
            "stratify_by": "y",
            "target": {"column": "y", "task": "binary"},
        },
    )
    assert with_target["status"] == "ok", with_target


# --------------------------------------------------------------------------- #
# mode=none / field
# --------------------------------------------------------------------------- #
def test_none_mode_marks_full_sample(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = dataset_split(project_dir, {"mode": "none"})
    assert payload["status"] == "ok", payload
    counts = payload["data"]["counts"]
    assert counts["train"] == payload["data"]["rows"]
    assert counts["val"] == counts["test"] == 0
    assert any("全量" in w for w in payload["warnings"])
    doc = load_split(project_dir)
    assert "未划分" in doc["description"]


def test_field_mode_uses_expressions(tmp_path: Path) -> None:
    csv, project_dir = bootstrap(tmp_path)
    frame = pd.read_csv(csv, encoding="utf-8")
    payload = dataset_split(
        project_dir,
        {
            "mode": "field",
            "target": {"column": "y", "task": "binary"},
            "field": {
                "column": "month",
                "expr_train": "month in ('2026-01','2026-02') & f_signal >= -100",
                "expr_val": "month == '2026-03'",
                "expr_test": "month not in ('2026-01','2026-02','2026-03')",
            },
        },
    )
    assert payload["status"] == "ok", payload
    counts = payload["data"]["counts"]
    expected_val = int((frame["month"] == "2026-03").sum())
    expected_train = int(frame["month"].isin(["2026-01", "2026-02"]).sum())
    assert counts["val"] == expected_val
    assert counts["train"] == expected_train
    assert counts["test"] == 0
    doc = load_split(project_dir)
    assert doc["mode"] == "field"
    assert doc["spec"]["expression"]["train"].startswith("month in")
    assert doc["label_distribution"]["train"]["count"] == counts["train"]


def test_field_overlap_is_rejected(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = dataset_split(
        project_dir,
        {
            "mode": "field",
            "field": {
                "expr_train": "month == '2026-01'",
                "expr_val": "month == '2026-01'",
            },
        },
    )
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "expression_invalid"
    assert "重叠" in payload["error"]["message"]


def test_field_requires_expressions(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = dataset_split(project_dir, {"mode": "field"})
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "config_invalid"


# --------------------------------------------------------------------------- #
# 规格校验与前置条件
# --------------------------------------------------------------------------- #
def test_invalid_ratios_and_mode(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    ratios = dataset_split(project_dir, {"mode": "random", "ratios": [0.8, 0.3, 0.1]})
    assert ratios["status"] == "error"
    assert ratios["error"]["code"] == "config_invalid"
    assert "合计" in ratios["error"]["message"]

    mode = dataset_split(project_dir, {"mode": "kfold"})
    assert mode["status"] == "error"
    assert mode["error"]["code"] == "config_invalid"


def test_time_column_is_recorded(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = dataset_split(
        project_dir, {"mode": "random", "seed": 5, "time_column": "month"}
    )
    assert payload["status"] == "ok", payload
    assert load_split(project_dir)["spec"]["time_column"] == "month"


def test_missing_project_dir_returns_error(tmp_path: Path) -> None:
    payload = dataset_split(str(tmp_path / "mdl-nope"), {"mode": "none"})
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "project_dir_not_found"
