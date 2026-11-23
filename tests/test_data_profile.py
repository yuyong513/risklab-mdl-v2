"""样本画像单测（``risklab_mdl.data.profile``）。

只验数值与结构（画像层不落盘、不出图），因此不需要 matplotlib 参与。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures.make_fixtures import (  # noqa: E402
    binary_frame,
    missing_text_frame,
    multiclass_frame,
    no_time_frame,
)

from risklab_mdl.data.loader import infer_roles  # noqa: E402
from risklab_mdl.data.profile import (  # noqa: E402
    LOW_BAD_RATE,
    compute_profile,
    missing_summary_frame,
)

BINARY_TARGET = {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}}


def _roles(frame: pd.DataFrame) -> dict:
    roles, _ = infer_roles(frame)
    return {key: roles.get(key) for key in ("id", "time", "split")}


def test_profile_binary_core_numbers() -> None:
    frame = binary_frame()
    result = compute_profile(frame, target=BINARY_TARGET, roles=_roles(frame))

    assert result.shape == {"rows": 400, "columns": frame.shape[1]}
    dist = result.target_distribution
    assert dist["task"] == "binary"
    assert dist["counts"] == {"0": 200, "1": 200}
    assert dist["bad_label"] == "1"
    assert dist["bad_rate"] == pytest.approx(0.5)
    assert dist["bad_count"] == 200 and dist["good_count"] == 200
    assert dist["missing"] == 0
    # 落盘表
    assert list(result.target_distribution_frame.columns) == [
        "label",
        "label_name",
        "count",
        "share",
    ]
    assert result.target_distribution_frame["share"].sum() == pytest.approx(1.0)


def test_profile_missing_summary_sorted_and_roles() -> None:
    frame = binary_frame()
    result = compute_profile(frame, target=BINARY_TARGET, roles=_roles(frame))
    missing_rates = [float(row["missing_rate"]) for row in result.missing_summary]
    assert missing_rates == sorted(missing_rates, reverse=True), "缺失汇总要按缺失率降序"
    by_name = {row["column"]: row for row in result.missing_summary}
    assert by_name["id"]["role"] == "id"
    assert by_name["month"]["role"] == "time"
    assert by_name["y"]["role"] == "target"
    assert by_name["f_const"]["same_value_rate"] == pytest.approx(1.0)
    assert by_name["f_missing"]["missing_rate"] == pytest.approx(0.2, abs=0.06)


def test_profile_describe_covers_numeric_only() -> None:
    frame = binary_frame()
    result = compute_profile(frame, target=BINARY_TARGET, roles=_roles(frame))
    describe = result.describe_frame
    for column in ("count", "mean", "std", "min", "25%", "50%", "75%", "max"):
        assert column in describe.columns
    assert "f_signal" in describe.index
    assert "f_text" not in describe.index, "文本列不进描述统计"
    assert describe.loc["f_signal", "role"] == "feature"
    assert describe.loc["y", "role"] == "target"


def test_profile_feature_types_counts() -> None:
    frame = binary_frame()
    result = compute_profile(frame, target=BINARY_TARGET, roles=_roles(frame))
    counts = result.feature_types["counts"]
    assert counts["numeric"] == 7, "f_signal/f_weak/f_noise/f_dup/f_missing/f_const + y"
    assert counts["categorical"] >= 1  # f_text
    assert counts["datetime"] == 0


def test_profile_monthly_cross_binary() -> None:
    frame = binary_frame()
    result = compute_profile(frame, target=BINARY_TARGET, roles=_roles(frame))
    assert result.monthly is not None and len(result.monthly) == 3
    months = [row["month"] for row in result.monthly]
    assert months == sorted(months)
    total = sum(int(row["count"]) for row in result.monthly)
    assert total == 400
    for row in result.monthly:
        assert 0.0 <= float(row["bad_rate"]) <= 1.0


def test_profile_without_time_column_skips_and_warns() -> None:
    frame = no_time_frame()
    roles = {"id": "apply_no", "time": None, "split": None}
    result = compute_profile(frame, target=BINARY_TARGET, roles=roles)
    assert result.monthly is None
    assert result.monthly_frame is None
    assert any("无时间列" in w and "跳过" in w for w in result.warnings)


def test_profile_unparseable_time_column_warns() -> None:
    frame = binary_frame()
    frame["month"] = "不是日期"
    result = compute_profile(frame, target=BINARY_TARGET, roles=_roles(frame))
    assert result.monthly is None
    assert any("无法解析为月份" in w for w in result.warnings)


def test_profile_multiclass_uses_shares_not_bad_rate() -> None:
    frame = multiclass_frame()
    roles = _roles(frame)
    result = compute_profile(
        frame, target={"column": "y", "task": "multiclass", "labels": None}, roles=roles
    )
    dist = result.target_distribution
    assert dist["task"] == "multiclass"
    assert dist["n_classes"] == 3
    assert "bad_rate" not in dist, "多分类不能伪造二分类坏率"
    assert dist["majority_share"] == pytest.approx(1 / 3, abs=0.02)
    assert sum(dist["shares"].values()) == pytest.approx(1.0, abs=1e-5)
    monthly = result.monthly or []
    assert monthly and "label" in monthly[0], "多分类月份交叉按 (月, 类) 展开"


def test_profile_low_bad_rate_and_missing_text_flags() -> None:
    frame = missing_text_frame()
    frame["y"] = 0
    frame.loc[frame.index[:2], "y"] = 1  # 坏率 2/120
    frame["mostly_empty"] = np.where(np.arange(len(frame)) < 20, 1.0, np.nan)
    result = compute_profile(
        frame, target={"column": "y", "task": "binary", "labels": None}, roles={"time": None}
    )
    assert float(result.target_distribution["bad_rate"]) < LOW_BAD_RATE
    assert any("坏样本占比" in w for w in result.warnings)
    assert any("mostly_empty" in w for w in result.warnings), "高缺失列要在 warnings 里点名"


def test_profile_high_missing_column_flagged() -> None:
    frame = binary_frame()
    frame["f_very_missing"] = np.where(np.arange(len(frame)) < 10, 1.0, np.nan)
    result = compute_profile(frame, target=BINARY_TARGET, roles=_roles(frame))
    assert any("f_very_missing" in w for w in result.warnings)


def test_profile_sampling_warns() -> None:
    frame = binary_frame()
    result = compute_profile(
        frame, target=BINARY_TARGET, roles=_roles(frame), sample_rows=100
    )
    assert result.shape["rows"] == 100
    assert any("抽样" in w for w in result.warnings)


def test_profile_target_missing_raises() -> None:
    frame = binary_frame()
    with pytest.raises(ValueError):
        compute_profile(frame, target={"column": "nope"}, roles=_roles(frame))


def test_profile_target_with_missing_labels() -> None:
    frame = binary_frame().copy()
    frame.loc[frame.index[:40], "y"] = np.nan
    result = compute_profile(frame, target=BINARY_TARGET, roles=_roles(frame))
    dist = result.target_distribution
    assert dist["missing"] == 40
    assert dist["valid"] == 360
    assert any("缺失标签" in w for w in result.warnings)


def test_profile_data_block_contract_shape() -> None:
    frame = binary_frame()
    result = compute_profile(frame, target=BINARY_TARGET, roles=_roles(frame))
    block = result.data_block()
    for key in ("target_distribution", "missing_summary", "chart_count"):
        assert key in block, f"data 缺少契约字段 {key}"
    assert block["chart_count"] == 0
    assert len(block["missing_summary"]) <= 20
    assert block["has_time_column"] is True


def test_missing_summary_frame_columns() -> None:
    frame = binary_frame()
    result = compute_profile(frame, target=BINARY_TARGET, roles=_roles(frame))
    table = missing_summary_frame(result.columns)
    assert list(table.columns) == [
        "column",
        "role",
        "dtype",
        "nunique",
        "missing_rate",
        "same_value_rate",
    ]
    assert len(table) == frame.shape[1]
