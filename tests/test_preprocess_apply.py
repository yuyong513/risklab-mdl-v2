"""``preprocess_apply`` 单测（契约 §3.3 + ``docs/preprocess-split.md``）。

覆盖（对应验收 ③④）：
* 手动分箱箱表与规格逐箱一致（计数/编码），缺失独立箱 ``__MISSING__``；
* LR 就绪：缺失 + 文本数据经 fill + encode + cap 后入模矩阵无 NaN、全数值；
* onehot / ordinal / woe 三条编码路径；
* 规格校验（未知列/策略/键、区间非法、目标列、空 spec）→ ``spec_invalid``；
* 落盘产物（fill/encode/outlier/manual_bins/summary + processed 数据）与 manifest 阶段。
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

from risklab_mdl.tools.dataset import (  # noqa: E402
    dataset_load,
    dataset_profile,
    project_init,
)
from risklab_mdl.tools.preprocess import preprocess_apply  # noqa: E402
from risklab_mdl.tools.split import dataset_split  # noqa: E402

TARGET = {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}}
ENVELOPE_KEYS = {"ok", "status", "data", "artifacts", "warnings", "error"}


def bootstrap(tmp_path: Path, name: str = "missing_text_month", *, with_target: bool = True):
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


def read_processed(project_dir: str, data: dict) -> pd.DataFrame:
    path = Path(project_dir) / data["processed_path"]
    assert path.is_file(), f"processed 数据不存在：{path}"
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8")


def load_json(project_dir: str, rel: str) -> dict:
    return json.loads((Path(project_dir) / rel).read_text(encoding="utf-8"))


def model_columns(processed: pd.DataFrame) -> list:
    return [c for c in processed.columns if c not in ("id", "y", "month", "split")]


# --------------------------------------------------------------------------- #
# 验收 ③：手动分箱与规格逐箱一致 + 缺失独立箱
# --------------------------------------------------------------------------- #
def test_manual_binning_matches_spec_and_has_missing_bin(tmp_path: Path) -> None:
    csv, project_dir = bootstrap(tmp_path)
    raw = pd.read_csv(csv, encoding="utf-8")
    spec = {
        "binning": {
            "amount": {"edges": ["-inf", 0, 100, 500, "inf"], "missing_as_bin": True}
        }
    }
    payload = preprocess_apply(project_dir, spec)
    assert set(payload) == ENVELOPE_KEYS
    assert payload["status"] == "ok", payload
    assert "preprocess/manual_bins.json" in payload["artifacts"]
    assert any(path.startswith("preprocess/processed.") for path in payload["artifacts"])

    doc = load_json(project_dir, "preprocess/manual_bins.json")
    entry = doc["entries"][0]
    labels = ["(-inf, 0]", "(0, 100]", "(100, 500]", "(500, inf]"]
    assert entry["column"] == "amount"
    assert entry["edges"] == ["-inf", 0.0, 100.0, 500.0, "inf"]
    assert entry["n_bins"] == 4
    assert entry["missing_as_bin"] is True
    assert entry["missing_bin_label"] == "__MISSING__"
    assert list(entry["bin_counts"]) == labels + ["__MISSING__"], entry["bin_counts"]

    # 独立复算：同一套切点的等宽分箱计数必须逐箱一致
    expected_codes = pd.cut(
        raw["amount"], bins=[-np.inf, 0, 100, 500, np.inf],
        right=True, include_lowest=True, labels=False,
    )
    expected_counts = [int((expected_codes == code).sum()) for code in range(4)]
    actual_counts = [entry["bin_counts"][label] for label in labels]
    assert actual_counts == expected_counts, (
        f"分箱计数与规格不一致：{actual_counts} vs {expected_counts}"
    )
    raw_missing = int(raw["amount"].isna().sum())
    assert entry["missing_count"] == raw_missing
    assert entry["bin_counts"]["__MISSING__"] == raw_missing
    assert sum(entry["bin_counts"].values()) == len(raw)

    processed = read_processed(project_dir, payload["data"])
    codes = processed["amount"]
    assert pd.api.types.is_integer_dtype(codes)
    assert int((codes == -1).sum()) == raw_missing
    for code, label in enumerate(labels):
        assert int((codes == code).sum()) == entry["bin_counts"][label], label
    assert set(int(v) for v in codes.unique()) <= {-1, 0, 1, 2, 3}


# --------------------------------------------------------------------------- #
# 验收 ④：LR 就绪（fill + encode + cap）
# --------------------------------------------------------------------------- #
def test_fill_encode_cap_produces_lr_ready_matrix(tmp_path: Path) -> None:
    csv, project_dir = bootstrap(tmp_path)
    raw = pd.read_csv(csv, encoding="utf-8")
    spec = {
        "missing": {"amount": "median", "channel": "mode"},
        "encode": {"channel": "onehot", "note": "ordinal"},
        "cap": {
            "score": {
                "strategy": "quantile",
                "lower_quantile": 0.01,
                "upper_quantile": 0.99,
            }
        },
    }
    payload = preprocess_apply(project_dir, spec)
    assert payload["status"] == "ok", payload
    data = payload["data"]
    assert data["model_readiness"]["lr_ready"] is True, payload["warnings"]
    assert data["model_readiness"]["non_numeric_columns"] == []
    assert data["model_readiness"]["remaining_missing"] == {}

    processed = read_processed(project_dir, data)
    columns = model_columns(processed)
    assert columns, "应至少有一个模型列"
    for column in columns:
        assert pd.api.types.is_numeric_dtype(processed[column]), (
            f"{column} 不是数值列：{processed[column].dtype}"
        )
        assert int(processed[column].isna().sum()) == 0, f"{column} 仍有缺失"
    # 编码列：原始文本列已被独热列替换（空串在 CSV 里即缺失，故 4 档）
    assert "channel" not in processed.columns
    onehot_columns = [c for c in processed.columns if c.startswith("channel_")]
    encode = load_json(project_dir, "preprocess/encode.json")["entries"]
    by_column = {e["column"]: e for e in encode}
    assert by_column["channel"]["strategy"] == "onehot"
    assert len(by_column["channel"]["onehot_values"]) == 4
    assert len(onehot_columns) == 4, onehot_columns
    assert all(int(processed[c].max()) <= 1 and int(processed[c].min()) >= 0
               for c in onehot_columns)
    assert by_column["note"]["strategy"] == "ordinal"
    assert pd.api.types.is_numeric_dtype(processed["note"])

    fill = load_json(project_dir, "preprocess/fill.json")["entries"]
    amount_fill = {e["column"]: e for e in fill}["amount"]
    assert amount_fill["strategy"] == "median"
    assert amount_fill["missing_before"] == int(raw["amount"].isna().sum()) > 0
    assert amount_fill["missing_after"] == 0

    cap = load_json(project_dir, "preprocess/outlier.json")["entries"]
    assert cap[0]["column"] == "score"
    assert cap[0]["n_capped"] > 0
    assert cap[0]["max_after"] <= cap[0]["upper"]
    assert float(processed["score"].max()) <= cap[0]["upper"] + 1e-9

    summary = load_json(project_dir, "preprocess/summary.json")
    assert summary["order"] == ["missing", "encode", "cap", "binning"]
    assert summary["schema"] == "risklab-mdl.preprocess-summary/1"
    assert summary["rows_in"] == summary["rows_out"] == len(raw)
    assert summary["model_readiness"]["lr_ready"] is True


def test_cap_absolute_upper_bound_only(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = preprocess_apply(
        project_dir, {"cap": {"score": {"strategy": "absolute", "max": 3.0}}}
    )
    assert payload["status"] == "ok", payload
    processed = read_processed(project_dir, payload["data"])
    assert float(processed["score"].max()) <= 3.0 + 1e-9
    entry = load_json(project_dir, "preprocess/outlier.json")["entries"][0]
    assert entry["lower"] is None and entry["upper"] == 3.0
    assert entry["n_capped"] >= 1


# --------------------------------------------------------------------------- #
# 编码三条路径
# --------------------------------------------------------------------------- #
def test_ordinal_path_persists_mapping(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = preprocess_apply(
        project_dir,
        {"missing": {"channel": "mode"}, "encode": {"channel": "ordinal"}},
    )
    assert payload["status"] == "ok", payload
    entry = load_json(project_dir, "preprocess/encode.json")["entries"][0]
    assert entry["strategy"] == "ordinal"
    assert set(entry["mapping"]) == {"app", "partner", "store", "web"}
    processed = read_processed(project_dir, payload["data"])
    assert pd.api.types.is_numeric_dtype(processed["channel"])
    assert int(processed["channel"].isna().sum()) == 0
    assert sorted(processed["channel"].unique().tolist()) == [0.0, 1.0, 2.0, 3.0]


def test_woe_path_without_split_warns_leakage(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = preprocess_apply(project_dir, {"encode": {"channel": "woe"}})
    assert payload["status"] == "ok", payload
    assert any("WOE" in w and "标签泄漏" in w for w in payload["warnings"]), payload["warnings"]
    entry = load_json(project_dir, "preprocess/encode.json")["entries"][0]
    assert entry["strategy"] == "woe"
    assert entry["fit_scope"] == "全量（未划分）"
    assert "__MISSING__" in entry["woe_table"]["mapping"]
    processed = read_processed(project_dir, payload["data"])
    assert pd.api.types.is_numeric_dtype(processed["channel"])
    assert int(processed["channel"].isna().sum()) == 0


def test_woe_path_with_split_uses_train_rows(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    split = dataset_split(
        project_dir, {"mode": "random", "ratios": [0.7, 0.2, 0.1], "seed": 42}
    )
    assert split["status"] == "ok", split
    payload = preprocess_apply(project_dir, {"encode": {"channel": "woe"}})
    assert payload["status"] == "ok", payload
    entry = load_json(project_dir, "preprocess/encode.json")["entries"][0]
    assert entry["fit_scope"] == "train（split/split.json）"
    assert any("train 子集" in w for w in payload["warnings"]), payload["warnings"]


# --------------------------------------------------------------------------- #
# 规格校验 → spec_invalid
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "spec, keyword",
    [
        ({"missing": {"nope": "median"}}, "不存在"),
        ({"missing": {"amount": "magic"}}, "未知的缺失策略"),
        ({"encode": {"channel": "target"}}, "未知的编码策略"),
        ({"missing": {"y": "median"}}, "目标列"),
        ({"missing": {"amount": "sentinel"}}, "哨兵值"),
        ({"cap": {"channel": {"strategy": "absolute", "max": 1}}}, "不是数值列"),
        ({"binning": {"amount": {"edges": [0, 100, 50]}}}, "严格递增"),
        ({"binning": {"amount": {"edges": ["-inf", "inf"]}}}, "内部切点"),
        ({"binning": {"amount": {"edges": [0, "inf", 100]}}}, "只能出现在首尾"),
        ({"unknown_section": {"x": "y"}}, "未知键"),
    ],
)
def test_invalid_spec_returns_spec_invalid(tmp_path: Path, spec: dict, keyword: str) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = preprocess_apply(project_dir, spec)
    assert payload["status"] == "error", payload
    assert payload["error"]["code"] == "spec_invalid", payload["error"]
    assert keyword in payload["error"]["message"], payload["error"]["message"]


def test_empty_spec_returns_spec_invalid(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = preprocess_apply(project_dir, {})
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "spec_invalid"
    assert "spec 为空" in payload["error"]["message"]


def test_missing_drop_warns_about_split_alignment(tmp_path: Path) -> None:
    csv, project_dir = bootstrap(tmp_path)
    raw = pd.read_csv(csv, encoding="utf-8")
    split = dataset_split(
        project_dir, {"mode": "random", "ratios": [0.7, 0.2, 0.1], "seed": 42}
    )
    assert split["status"] == "ok", split
    payload = preprocess_apply(project_dir, {"missing": {"amount": "drop"}})
    assert payload["status"] == "ok", payload
    assert payload["data"]["rows_out"] < len(raw)
    assert any("drop" in w for w in payload["warnings"]), payload["warnings"]
    assert any("mask" in w for w in payload["warnings"]), payload["warnings"]
    processed = read_processed(project_dir, payload["data"])
    assert len(processed) == payload["data"]["rows_out"]


def test_as_bin_plus_binning_keeps_missing_bin(tmp_path: Path) -> None:
    """as_bin + binning 配套时缺失落独立箱，且不应再报「缺分箱规格」告警。"""
    csv, project_dir = bootstrap(tmp_path)
    raw = pd.read_csv(csv, encoding="utf-8")
    spec = {
        "missing": {"amount": "as_bin"},
        "encode": {"channel": "onehot", "note": "ordinal"},
        "binning": {"amount": {"edges": ["-inf", 0, 100, 500, "inf"]}},
    }
    payload = preprocess_apply(project_dir, spec)
    assert payload["status"] == "ok", payload
    assert not any("没有对应的 spec.binning" in w for w in payload["warnings"])
    entry = load_json(project_dir, "preprocess/manual_bins.json")["entries"][0]
    assert entry["missing_count"] == int(raw["amount"].isna().sum()) > 0
    processed = read_processed(project_dir, payload["data"])
    assert int((processed["amount"] == -1).sum()) == entry["missing_count"]
    assert payload["data"]["model_readiness"]["lr_ready"] is True, payload["warnings"]


# --------------------------------------------------------------------------- #
# 前置条件与 manifest
# --------------------------------------------------------------------------- #
def test_apply_requires_confirmed_target(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path, with_target=False)
    payload = preprocess_apply(project_dir, {"missing": {"amount": "median"}})
    assert payload["status"] == "needs_user_input", payload
    assert payload["data"]["question"]["id"] == "dataset_confirm_target"
    assert not (Path(project_dir) / "preprocess" / "processed.csv").exists()


def test_manifest_preprocess_stage_marked_done(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = preprocess_apply(project_dir, {"missing": {"amount": "median"}})
    assert payload["status"] == "ok", payload
    manifest = json.loads(
        (Path(project_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    stage = manifest["stages"]["preprocess"]
    assert stage["done"] is True
    assert stage["path"] == "preprocess/"
    assert "填充 1 列" in stage["summary"]


def test_processed_csv_fallback_is_documented(tmp_path: Path) -> None:
    """无 pyarrow 时落 CSV 并在 warnings 说明（有 pyarrow 时必须是 parquet）。"""
    _, project_dir = bootstrap(tmp_path)
    payload = preprocess_apply(project_dir, {"missing": {"amount": "median"}})
    assert payload["status"] == "ok", payload
    fmt = payload["data"]["processed_format"]
    assert fmt in ("parquet", "csv")
    if fmt == "csv":
        assert any("pyarrow" in w and "CSV" in w for w in payload["warnings"])
        assert payload["data"]["processed_path"].endswith(".csv")
    else:  # pragma: no cover - 本机未装 pyarrow
        assert payload["data"]["processed_path"].endswith(".parquet")
