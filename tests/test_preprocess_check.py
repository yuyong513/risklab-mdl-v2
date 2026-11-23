"""``preprocess_check`` 单测（契约 §3.3 + ``docs/preprocess-split.md``）。

覆盖：must_fix / suggested 分类（文本未编码、缺失、同值列、极端值、高基数）、
规则可配置（missing_policy / high_missing_rate）、``by_feature`` 明细可追查、
缺目标字段返回 ``needs_user_input``、非法 features 的错误码。
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

from risklab_mdl.preprocess.check import (  # noqa: E402
    DEFAULT_RULES,
    check_frame,
    select_feature_columns,
)
from risklab_mdl.tools.dataset import (  # noqa: E402
    dataset_load,
    dataset_profile,
    project_init,
)
from risklab_mdl.tools.preprocess import preprocess_check  # noqa: E402

TARGET = {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}}
ENVELOPE_KEYS = {"ok", "status", "data", "artifacts", "warnings", "error"}


def bootstrap(tmp_path: Path, name: str = "missing_text_month", *, with_target: bool = True):
    """造数据 → project_init → dataset_load（+ 需要目标时跑 dataset_profile）。"""
    paths = write_all(tmp_path / "data")
    csv = paths[name]
    init = project_init(str(csv))
    assert init["status"] == "ok", init
    assert dataset_load(str(csv))["status"] == "ok"
    project_dir = init["data"]["project_dir"]
    if with_target:
        profiled = dataset_profile(project_dir, {"target": TARGET})
        assert profiled["status"] == "ok", profiled
    return csv, project_dir


# --------------------------------------------------------------------------- #
# 分类结论（验收 ⑤）
# --------------------------------------------------------------------------- #
def test_check_classifies_text_and_missing_as_must_fix(tmp_path: Path) -> None:
    csv, project_dir = bootstrap(tmp_path)
    payload = preprocess_check(project_dir, {"target": TARGET})

    assert set(payload) == ENVELOPE_KEYS
    assert payload["status"] == "ok", payload
    assert payload["artifacts"] == ["preprocess/check.json"]

    data = payload["data"]
    must_fix = data["must_fix"]
    suggested = data["suggested"]
    assert must_fix, "缺失 + 文本数据必须至少有一条 must_fix"
    # ① 文本列未编码 ⇒ must_fix
    assert any(msg.startswith("channel:") for msg in must_fix), must_fix
    assert any(msg.startswith("note:") for msg in must_fix), must_fix
    assert any("非数值" in msg and msg.startswith("channel:") for msg in must_fix)
    # ② 缺失 ⇒ must_fix
    assert any(msg.startswith("amount:") and "缺失" in msg for msg in must_fix), must_fix
    # ③ 同值列 / 极端值 / 高基数 ⇒ suggested
    assert any(msg.startswith("const:") and "同值列" in msg for msg in suggested), suggested
    assert any(msg.startswith("score:") and "极端值" in msg for msg in suggested), suggested
    assert any(
        msg.startswith("note:") and "类别取值数" in msg for msg in suggested
    ), suggested
    # ④ 数值列本身不是 must_fix
    assert not any(msg.startswith("score:") for msg in must_fix), must_fix

    by_feature = data["by_feature"]
    assert set(by_feature) == {"amount", "channel", "score", "const", "note"}
    codes = {issue["code"] for issue in by_feature["channel"]["issues"]}
    assert {"not_numeric", "missing"} <= codes, codes
    assert by_feature["score"]["numeric"] is True
    assert by_feature["score"]["issues"], "score 应有极端值建议"
    assert by_feature["channel"]["must_fix"], "by_feature 要保留逐条 must_fix 明细"

    # ⑤ 落盘产物结构（契约 §3.3 三键）
    check_path = Path(project_dir) / "preprocess" / "check.json"
    doc = json.loads(check_path.read_text(encoding="utf-8"))
    for key in ("must_fix", "suggested", "by_feature"):
        assert key in doc, f"check.json 缺少 {key}"
    assert doc["schema"] == "risklab-mdl.preprocess-check/1"
    assert doc["target"]["column"] == "y"
    assert doc["summary"]["rows"] == 180


def test_missing_policy_can_downgrade_to_suggested(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = preprocess_check(
        project_dir,
        {"target": TARGET, "missing_policy": "suggested", "high_missing_rate": 0.99},
    )
    assert payload["status"] == "ok", payload
    must_fix = payload["data"]["must_fix"]
    suggested = payload["data"]["suggested"]
    assert not any("amount: 缺失" in msg for msg in must_fix), must_fix
    assert any("amount: 缺失" in msg for msg in suggested), suggested
    # 非数值未编码仍然是 must_fix（这个不能降级）
    assert any(msg.startswith("channel:") and "非数值" in msg for msg in must_fix)


def test_by_feature_keeps_clean_numeric_column(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = preprocess_check(project_dir, {"target": TARGET, "features": ["score"]})
    assert payload["status"] == "ok", payload
    by_feature = payload["data"]["by_feature"]
    assert list(by_feature) == ["score"]
    # score 只有极端值建议，不是 must_fix
    assert by_feature["score"]["must_fix"] == []
    assert payload["data"]["must_fix"] == []


# --------------------------------------------------------------------------- #
# 纯函数 check_frame 的规则
# --------------------------------------------------------------------------- #
def test_check_frame_clean_numeric_has_no_issues() -> None:
    frame = pd.DataFrame(
        {
            "y": [0, 1] * 20,
            "x1": np.linspace(0.0, 10.0, 40),
            "x2": np.arange(40, dtype=float),
        }
    )
    result = check_frame(
        frame, target_column="y", features=["x1", "x2"], config={}
    )
    assert result.must_fix == []
    assert result.suggested == []
    assert set(result.by_feature) == {"x1", "x2"}
    assert any("未发现预处理问题" in w for w in result.warnings)


def test_check_frame_constant_and_missing_rules() -> None:
    frame = pd.DataFrame(
        {
            "y": [0, 1] * 5,
            "const": [1.0] * 10,
            "sparse": [1.0, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "text": ["a", "b"] * 5,
        }
    )
    result = check_frame(
        frame, target_column="y", features=["const", "sparse", "text"], config={}
    )
    codes = {issue["code"] for issue in result.by_feature["const"]["issues"]}
    assert "constant" in codes
    sparse_codes = {issue["code"] for issue in result.by_feature["sparse"]["issues"]}
    assert "missing" in sparse_codes and "high_missing" not in sparse_codes
    assert any("text" in msg for msg in result.must_fix)
    # 规则可配置：missing_policy=ignore 后不再出现在任何级别
    ignored = check_frame(
        frame,
        target_column="y",
        features=["const", "sparse", "text"],
        config={"missing_policy": "ignore"},
    )
    assert not any("sparse" in msg for msg in ignored.must_fix + ignored.suggested)


def test_default_rules_are_documented_snapshot() -> None:
    assert DEFAULT_RULES["missing_policy"] == "must_fix"
    assert DEFAULT_RULES["high_missing_rate"] == 0.5
    assert DEFAULT_RULES["high_cardinality_max"] == 20


def test_select_feature_columns_excludes_roles_and_target() -> None:
    frame = pd.DataFrame({"id": [1, 2], "month": ["2026-01", "2026-02"],
                          "split": ["train", "val"], "y": [0, 1], "x": [1.0, 2.0]})
    roles = {"id": "id", "time": "month", "split": "split"}
    assert select_feature_columns(
        frame, roles=roles, target_column="y", features=None
    ) == ["x"]
    assert select_feature_columns(
        frame, roles=roles, target_column="y", features=["x", "month"]
    ) == ["x", "month"]
    with pytest.raises(ValueError):
        select_feature_columns(
            frame, roles=roles, target_column="y", features=["nope"]
        )


# --------------------------------------------------------------------------- #
# 工具层错误路径
# --------------------------------------------------------------------------- #
def test_check_requires_confirmed_target(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path, with_target=False)
    payload = preprocess_check(project_dir)
    assert payload["status"] == "needs_user_input", payload
    assert payload["data"]["question"]["id"] == "dataset_confirm_target"
    assert "y" in payload["data"]["context"]["candidates"]


def test_check_unknown_feature_returns_config_invalid(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = preprocess_check(
        project_dir, {"target": TARGET, "features": ["not_a_column"]}
    )
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "config_invalid"
    assert "not_a_column" in payload["error"]["message"]


def test_check_empty_features_returns_features_empty(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = preprocess_check(project_dir, {"target": TARGET, "features": []})
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "features_empty"


def test_check_missing_project_returns_project_dir_not_found(tmp_path: Path) -> None:
    payload = preprocess_check(str(tmp_path / "mdl-nope"), {"target": TARGET})
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "project_dir_not_found"
