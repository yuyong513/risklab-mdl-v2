"""特征分析单测（``risklab_mdl.features.analysis``）。

覆盖：4 种分箱方法、二分类 IV 口径、多分类 OVR、跳过的可见理由、
稀疏箱合并提示、WOE 明细结构、失败路径（列不存在 / 方法非法）。
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
)

from risklab_mdl.data.loader import infer_roles  # noqa: E402
from risklab_mdl.features.analysis import (  # noqa: E402
    BINNING_METHODS,
    SKIP_CONSTANT,
    SKIP_NON_NUMERIC,
    WOE_DETAIL_COLUMNS,
    analyze_features,
    describe_bins,
)

BINARY_TARGET = {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}}
NUMERIC_FEATURES = ["f_signal", "f_weak", "f_noise", "f_dup", "f_missing", "f_const"]


def _roles(frame: pd.DataFrame) -> dict:
    roles, _ = infer_roles(frame)
    return {key: roles.get(key) for key in ("id", "time", "split")}


def _run(**kwargs):
    frame = binary_frame()
    params = dict(
        target=BINARY_TARGET,
        method="quantile",
        bins=5,
        features=NUMERIC_FEATURES,
        roles=_roles(frame),
    )
    params.update(kwargs)
    return frame, analyze_features(frame, **params)


# --------------------------------------------------------------------------- #
# 二分类基础口径
# --------------------------------------------------------------------------- #
def test_iv_table_shape_and_ordering() -> None:
    _, result = _run()
    table = result.iv_table
    assert len(table) == len(NUMERIC_FEATURES)
    usable = table[~table["skipped"].astype(bool)]
    assert usable["iv"].tolist() == sorted(usable["iv"].tolist(), reverse=True), (
        "可分析特征必须按 IV 降序排列"
    )
    assert table["skipped"].tolist()[-1] is True or bool(table["skipped"].iloc[-1]), (
        "被跳过的特征排在末尾（na_position=last）"
    )
    ivs = table.set_index("feature")["iv"]
    assert ivs["f_signal"] > ivs["f_noise"], "强信号 IV 必须高于噪声"
    assert ivs["f_dup"] > ivs["f_noise"]
    assert ivs["f_signal"] > 0.3
    assert result.feature_count == len(NUMERIC_FEATURES)
    assert result.skipped == [{"feature": "f_const", "reason": SKIP_CONSTANT}], (
        f"只有常量列该被跳过，实际：{result.skipped}"
    )


def test_constant_feature_is_skipped_with_reason() -> None:
    _, result = _run()
    row = result.iv_table.set_index("feature").loc["f_const"]
    assert bool(row["skipped"]) is True
    assert SKIP_CONSTANT in str(row["reason"])
    assert {"f_const"} == {item["feature"] for item in result.skipped}


def test_text_feature_skipped_with_reason() -> None:
    frame = binary_frame()
    result = analyze_features(
        frame, target=BINARY_TARGET, features=["f_signal", "f_text"], roles=_roles(frame)
    )
    rows = result.iv_table.set_index("feature")
    assert bool(rows.loc["f_text", "skipped"]) is True
    assert SKIP_NON_NUMERIC in rows.loc["f_text", "reason"]
    assert "需先编码" in rows.loc["f_text", "reason"]
    assert result.skipped[0]["feature"] == "f_text"
    # 非数值特征不应出现在 WOE 明细里
    assert "f_text" not in set(result.woe_detail["feature"])


def test_high_missing_feature_analyzed_but_flagged() -> None:
    frame = binary_frame()
    frame["f_sparse"] = np.where(np.arange(len(frame)) % 10 < 7, np.nan, frame["f_signal"])
    result = analyze_features(
        frame, target=BINARY_TARGET, features=["f_sparse"], method="quantile", bins=4
    )
    row = result.iv_table.iloc[0]
    assert bool(row["skipped"]) is False
    assert float(row["missing_rate"]) >= 0.5
    assert any("f_sparse" in w and "缺失率" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# WOE 明细
# --------------------------------------------------------------------------- #
def test_woe_detail_structure_and_consistency() -> None:
    _, result = _run()
    detail = result.woe_detail
    assert list(detail.columns) == list(WOE_DETAIL_COLUMNS)
    assert set(detail["class"]) == {"binary"}
    table = result.iv_table.set_index("feature")
    for feature in ("f_signal", "f_weak"):
        rows = detail[detail["feature"] == feature]
        total = int(rows["total"].sum())
        assert total == 400, "WOE 明细的样本数合计应等于样本量"
        valid = rows[rows["bin"] >= 0]
        assert len(valid) == int(table.loc[feature, "n_bins"])
        # IV = Σ IV贡献
        assert float(rows["iv_contribution"].sum()) == pytest.approx(
            float(table.loc[feature, "iv"]), abs=1e-6
        )
        # 每箱 总用户 = 坏用户 + 好用户
        assert (rows["total"] == rows["bad"] + rows["good"]).all()


def test_woe_sign_follows_bad_rate() -> None:
    """f_signal 越大越坏 ⇒ 最高取值箱的 WOE 必须为正。"""
    _, result = _run()
    rows = result.woe_detail[result.woe_detail["feature"] == "f_signal"]
    valid = rows[rows["bin"] >= 0].sort_values("bin")
    assert len(valid) >= 2
    top = valid.iloc[-1]
    assert float(top["bad_rate"]) > float(valid.iloc[0]["bad_rate"])
    assert float(top["woe"]) > 0


# --------------------------------------------------------------------------- #
# 四种分箱方法
# --------------------------------------------------------------------------- #
def test_equal_width_method() -> None:
    _, result = _run(method="equal_width", features=["f_signal"])
    row = result.iv_table.iloc[0]
    assert bool(row["skipped"]) is False
    assert int(row["n_bins_fitted"]) >= 2
    assert row["method"] == "equal_width"
    assert float(row["iv"]) > 0.1


def test_equal_width_on_near_constant_does_not_crash() -> None:
    """等宽分箱在极端集中取值上可能失败 —— 必须记成 skipped 而不是抛异常。"""
    frame = binary_frame()
    frame["f_almost_const"] = 1.0
    frame.loc[frame.index[0], "f_almost_const"] = 2.0
    result = analyze_features(
        frame, target=BINARY_TARGET, features=["f_almost_const"], method="equal_width"
    )
    assert len(result.iv_table) == 1
    assert result.iv_table.iloc[0]["feature"] == "f_almost_const"


def test_dt_method_respects_max_bins() -> None:
    _, result = _run(method="dt", bins=5, features=["f_signal", "f_weak"])
    table = result.iv_table
    assert (~table["skipped"]).all(), f"DT 分箱不该失败：{table.to_dict('records')}"
    assert table["n_bins_fitted"].max() <= 5
    assert table["method"].unique().tolist() == ["dt"]
    assert float(table["iv"].max()) > 0.2
    # DT 的结果仍可直接复用（binning/<feature>.json 的 spec）
    spec = result.bin_specs["f_signal"]
    assert spec["method"] == "dt"
    assert spec["per_class"]["binary"]["method"] == "business_cut"
    assert len(spec["per_class"]["binary"]["edges"]) == table.set_index("feature").loc[
        "f_signal", "n_bins_fitted"
    ] - 1


def test_custom_edges_method() -> None:
    frame = binary_frame()
    edges = {name: [-0.5, 0.5, 1.5] for name in ("f_signal", "f_weak")}
    result = analyze_features(
        frame, target=BINARY_TARGET, method="custom", custom_edges=edges,
        features=list(edges), roles=_roles(frame),
    )
    table = result.iv_table
    assert (~table["skipped"]).all()
    assert table["n_bins_fitted"].unique().tolist() == [4]  # 3 切点 ⇒ 4 箱
    for name in edges:
        assert result.bin_specs[name]["per_class"]["binary"]["edges"] == edges[name]


def test_custom_method_without_edges_skips_with_reason() -> None:
    frame = binary_frame()
    result = analyze_features(
        frame, target=BINARY_TARGET, method="custom", features=["f_signal"]
    )
    assert bool(result.iv_table.iloc[0]["skipped"]) is True
    assert "custom" in str(result.iv_table.iloc[0]["reason"])


def test_invalid_method_raises() -> None:
    frame = binary_frame()
    with pytest.raises(ValueError):
        analyze_features(frame, target=BINARY_TARGET, method="magic")
    assert "magic" not in BINNING_METHODS


def test_unknown_feature_raises() -> None:
    frame = binary_frame()
    with pytest.raises(ValueError):
        analyze_features(frame, target=BINARY_TARGET, features=["nope"])


def test_unknown_target_raises() -> None:
    frame = binary_frame()
    with pytest.raises(ValueError):
        analyze_features(frame, target={"column": "nope"}, features=["f_signal"])


# --------------------------------------------------------------------------- #
# 多分类 OVR
# --------------------------------------------------------------------------- #
def test_multiclass_ovr_per_class_iv_and_macro() -> None:
    frame = multiclass_frame()
    result = analyze_features(
        frame,
        target={"column": "y", "task": "multiclass", "labels": None},
        features=["f1", "f2", "f3", "f_noise"],
        method="quantile",
        bins=5,
        roles=_roles(frame),
    )
    table = result.iv_table
    assert set(table["task"]) == {"multiclass"}
    class_columns = sorted(c for c in table.columns if str(c).startswith("iv_"))
    assert class_columns == ["iv_0", "iv_1", "iv_2"], f"逐类 IV 列缺失：{list(table.columns)}"
    for row in table.to_dict(orient="records"):
        macro = np.mean([row["iv_0"], row["iv_1"], row["iv_2"]])
        assert float(row["iv"]) == pytest.approx(float(macro), abs=1e-6)
    by_feature = table.set_index("feature")
    # f1 专门区分类 0 ⇒ 类 0 的 OVR IV 最高
    assert by_feature.loc["f1", "iv_0"] > by_feature.loc["f1", "iv_1"]
    assert by_feature.loc["f2", "iv_1"] > by_feature.loc["f2", "iv_0"]
    # 每类各一层 WOE 明细
    detail = result.woe_detail[result.woe_detail["feature"] == "f1"]
    assert set(detail["class"]) == {"0", "1", "2"}
    assert any("one-vs-rest" in w for w in result.warnings)


def test_multiclass_dt_per_class() -> None:
    frame = multiclass_frame()
    result = analyze_features(
        frame,
        target={"column": "y", "task": "multiclass", "labels": None},
        features=["f1"],
        method="dt",
        bins=4,
        roles=_roles(frame),
    )
    row = result.iv_table.iloc[0]
    assert bool(row["skipped"]) is False
    spec = result.bin_specs["f1"]
    assert set(spec["per_class"]) == {"0", "1", "2"}
    for class_spec in spec["per_class"].values():
        assert len(class_spec["edges"]) + 1 <= 4


# --------------------------------------------------------------------------- #
# 其它
# --------------------------------------------------------------------------- #
def test_default_features_exclude_roles() -> None:
    frame = binary_frame()
    result = analyze_features(frame, target=BINARY_TARGET, roles=_roles(frame), bins=4)
    features = set(result.iv_table["feature"])
    assert "id" not in features and "month" not in features and "split" not in features
    assert "y" not in features
    assert "f_text" in features, "未显式指定特征时，非数值列也要出现（便于用户看到跳过原因）"


def test_bin_specs_are_reusable() -> None:
    _, result = _run(method="quantile", bins=5, features=["f_signal"])
    spec = result.bin_specs["f_signal"]
    assert spec["feature"] == "f_signal"
    assert spec["method"] == "quantile"
    parsed = spec["per_class"]["binary"]
    for key in ("edges", "n_valid_bins", "missing_as_bin", "min_bin_count", "sparse_strategy"):
        assert key in parsed
    assert parsed["n_valid_bins"] == len(parsed["edges"]) + 1


def test_describe_bins_roundtrip() -> None:
    _, result = _run(method="quantile", bins=5, features=["f_signal"])
    parsed = result.bin_specs["f_signal"]["per_class"]["binary"]
    assert set(parsed) >= {
        "method",
        "n_bins",
        "n_valid_bins",
        "requested_bins",
        "bins_downgraded",
        "edges",
        "missing_as_bin",
        "min_bin_count",
        "min_bin_pct",
        "sparse_strategy",
        "woe_smoothing",
        "fit_on_months",
    }


def test_missing_values_go_to_missing_bin() -> None:
    frame = missing_text_frame()
    result = analyze_features(
        frame,
        target={"column": "y", "task": "binary", "labels": None},
        features=["amount"],
        method="quantile",
        bins=5,
    )
    detail = result.woe_detail
    missing_rows = detail[detail["bin"] == -1]
    assert len(missing_rows) == 1, "缺失值应落进独立的缺失箱（-1）"
    assert missing_rows.iloc[0]["bin_label"] == "[MISSING]"
    assert int(missing_rows.iloc[0]["total"]) == int(frame["amount"].isna().sum())


def test_data_block_contract_shape() -> None:
    _, result = _run()
    block = result.data_block()
    assert block["feature_count"] == len(NUMERIC_FEATURES)
    assert isinstance(block["iv_top"], list) and block["iv_top"]
    assert set(block["iv_top"][0]) == {"feature", "iv"}
    assert [item["feature"] for item in block["skipped"]] == ["f_const"]
    assert block["target"]["column"] == "y"
