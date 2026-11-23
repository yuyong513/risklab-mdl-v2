"""数据加载层单测（``risklab_mdl.data.loader``）。

覆盖：编码探测（UTF-8 / GB18030 / UTF-8-BOM / UTF-16 / 无法解码）、
语义 dtype 推断、角色推断（含理由与"需确认"警告）、列清单、JSON 友好化。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures.make_fixtures import binary_frame, write_all, write_encoded  # noqa: E402

from risklab_mdl.data.loader import (  # noqa: E402
    ROLE_CONFIRM_WARNING,
    as_numeric,
    column_infos,
    infer_roles,
    infer_semantic_dtype,
    is_numeric_series,
    load_dataset,
    normalize_target,
    read_csv_auto,
    to_jsonable,
)
from risklab_mdl.platform.fs import EncodingUndecodable, PathNotReadable  # noqa: E402


# --------------------------------------------------------------------------- #
# 编码探测
# --------------------------------------------------------------------------- #
def test_read_utf8_plain(tmp_path: Path) -> None:
    paths = write_all(tmp_path)
    read = read_csv_auto(paths["binary"])
    assert read.encoding == "utf-8"
    assert read.rows == 400
    assert "y" in read.columns
    assert isinstance(read.warnings, tuple)


def test_read_gb18030_with_chinese(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"客户号": [f"K{i}" for i in range(6)], "y": [0, 1, 0, 1, 0, 1], "金额": [1.0, 2, 3, 4, 5, 6]}
    )
    path = write_encoded(tmp_path / "gbk.csv", frame, encoding="gb18030")
    read = read_csv_auto(path)
    assert read.encoding == "gb18030", "GB18030 编码的中文不能被误判成 utf-8"
    assert list(read.frame.columns) == ["客户号", "y", "金额"]


def test_read_utf8_bom(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["甲", "乙"]})
    path = write_encoded(tmp_path / "bom.csv", frame, encoding="utf-8-sig")
    read = read_csv_auto(path)
    assert read.encoding == "utf-8-sig"
    assert list(read.frame.columns) == ["a", "b"], "BOM 不能被当成列名的一部分"


def test_read_utf16(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["甲", "乙"]})
    path = write_encoded(tmp_path / "u16.csv", frame, encoding="utf-16")
    read = read_csv_auto(path)
    assert read.encoding == "utf-16"
    assert list(read.frame.columns) == ["a", "b"]


def test_undecodable_raises_typed_error(tmp_path: Path) -> None:
    """二进制/半截文件必须报 ``encoding_undecodable``，而不是返回空表。"""
    path = tmp_path / "binary.csv"
    path.write_bytes(b"\xff\xfd\x81")  # 非 BOM、奇数长度 ⇒ 5 种候选编码全失败
    with pytest.raises(EncodingUndecodable) as err:
        read_csv_auto(path)
    assert err.value.code == "encoding_undecodable"
    assert err.value.to_payload()["hint"]


def test_missing_file_raises_path_not_readable(tmp_path: Path) -> None:
    with pytest.raises(PathNotReadable) as err:
        read_csv_auto(tmp_path / "nope.csv")
    assert err.value.code == "path_not_readable"


def test_read_with_usecols(tmp_path: Path) -> None:
    paths = write_all(tmp_path)
    read = read_csv_auto(paths["binary"], usecols=["y", "f_signal"])
    assert list(read.columns) == ["y", "f_signal"]


# --------------------------------------------------------------------------- #
# 语义 dtype
# --------------------------------------------------------------------------- #
def test_infer_semantic_dtype_basic() -> None:
    assert infer_semantic_dtype(pd.Series([1, 2, 3])) == "integer"
    assert infer_semantic_dtype(pd.Series([1.5, 2.5])) == "float"
    assert infer_semantic_dtype(pd.Series([True, False])) == "bool"
    assert infer_semantic_dtype(pd.Series(["a", "b", "c"])) == "categorical"
    assert infer_semantic_dtype(pd.Series([f"v{i}" for i in range(100)])) == "text"
    assert infer_semantic_dtype(pd.Series([None, None], dtype="object")) == "empty"


def test_infer_semantic_dtype_rescues_numeric_text() -> None:
    """整列数字但混入少量非数字 ⇒ 大部分可解析时仍判数值（CSV 常见病）。"""
    values = pd.Series([f"{i}.5" for i in range(100)] + ["-"])
    assert infer_semantic_dtype(values) == "float"
    digits = pd.Series([str(i) for i in range(100)] + ["N/A"])
    assert infer_semantic_dtype(digits) == "integer"


def test_as_numeric_ratio_and_is_numeric() -> None:
    converted, ratio = as_numeric(pd.Series(["1", "2", "x", None]))
    assert ratio == pytest.approx(2 / 3)
    assert converted.iloc[:2].tolist() == [1.0, 2.0]
    assert np.isnan(converted.iloc[2])
    assert is_numeric_series(pd.Series([1, 2, 3])) is True
    assert is_numeric_series(pd.Series(["甲", "乙", "丙"])) is False


# --------------------------------------------------------------------------- #
# 角色推断
# --------------------------------------------------------------------------- #
def test_infer_roles_on_binary_fixture() -> None:
    frame = binary_frame()
    roles, reasons = infer_roles(frame)
    assert roles["id"] == "id"
    assert roles["time"] == "month"
    assert roles["split"] == "split"
    assert "y" in roles["target_candidates"]
    assert roles["target_candidates"][0] == "y", "标签列应排在候选第一位"
    # 每个角色都要给理由，且注明需确认
    for key in ("id", "time", "split"):
        assert reasons[key], f"{key} 缺少推断理由"
    assert reasons["target_candidates"]["y"]
    assert reasons["_note"] == ROLE_CONFIRM_WARNING


def test_infer_roles_no_target_no_time() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50) * 10})
    roles, reasons = infer_roles(frame)
    assert roles["time"] is None
    assert roles["split"] is None
    assert roles["target_candidates"] == [], "连续数值列不该被当标签候选"
    assert reasons["time"] is None


def test_infer_roles_low_cardinality_numeric_is_candidate() -> None:
    """低基数数值列可以当多分类标签候选（候选不是确认，仍要用户拍板）。"""
    frame = pd.DataFrame(
        {"a": [0.1, 0.2, 0.3, 0.4, 0.5], "grade": [0, 1, 2, 0, 1]}
    )
    roles, reasons = infer_roles(frame)
    assert "grade" in roles["target_candidates"]
    assert reasons["target_candidates"]["grade"]


def test_infer_roles_date_like_and_split_values() -> None:
    frame = pd.DataFrame(
        {
            "周期": ["2026年1月", "2026年2月", "2026年3月", "2026年4月"],
            "part": ["train", "train", "val", "test"],
            "y": [0, 1, 0, 1],
        }
    )
    roles, reasons = infer_roles(frame)
    assert roles["time"] == "周期", "中文「YYYY年M月」要能被识别为时间列"
    assert roles["split"] == "part", "取值落在 train/val/test 的列要识别为划分列"
    assert "y" in roles["target_candidates"]


def test_column_infos_roles_and_rates() -> None:
    frame = binary_frame()
    roles, _ = infer_roles(frame)
    infos = column_infos(frame, roles)
    by_name = {info["name"]: info for info in infos}
    assert by_name["id"]["role"] == "id"
    assert by_name["month"]["role"] == "time"
    assert by_name["split"]["role"] == "split"
    assert by_name["y"]["role"] == "target"
    assert by_name["f_signal"]["role"] == "feature"
    assert by_name["f_missing"]["missing_rate"] == pytest.approx(0.2, abs=0.06)
    assert by_name["f_const"]["same_value_rate"] == pytest.approx(1.0)
    assert by_name["f_text"]["dtype"] in ("categorical", "text")


def test_load_dataset_payload(tmp_path: Path) -> None:
    paths = write_all(tmp_path)
    payload = load_dataset(paths["binary"])
    assert payload["encoding"] == "utf-8"
    assert payload["rows"] == 400
    assert "id" in payload["columns"]
    assert payload["roles"]["id"] == "id"
    assert payload["dtypes"]["f_signal"] == "float"
    assert any("角色为推断" in w for w in payload["warnings"])


def test_to_jsonable_handles_numpy_and_nan() -> None:
    payload = to_jsonable(
        {"a": np.int64(3), "b": np.float64(float("nan")), "c": np.array([1.0, 2.0]),
         "d": (np.bool_(True),), "e": Path("x/y.csv")}
    )
    assert payload["a"] == 3 and isinstance(payload["a"], int)
    assert payload["b"] is None
    assert payload["c"] == [1.0, 2.0]
    assert payload["d"] == [True]
    assert payload["e"] == "x/y.csv"


# --------------------------------------------------------------------------- #
# 目标配置
# --------------------------------------------------------------------------- #
def test_normalize_target_variants() -> None:
    assert normalize_target(None) is None
    assert normalize_target({"column": "y"}) == {"column": "y", "task": None, "labels": None}
    parsed = normalize_target({"column": "y", "task": "binary", "labels": {1: "bad", 0: "good"}})
    assert parsed == {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}}
    with pytest.raises(ValueError):
        normalize_target({"column": ""})
    with pytest.raises(ValueError):
        normalize_target({"column": "y", "task": "regression"})
    with pytest.raises(ValueError):
        normalize_target({"column": "y", "labels": ["bad"]})
