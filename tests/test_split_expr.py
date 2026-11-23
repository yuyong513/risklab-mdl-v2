"""字段表达式单测（验收 ②）：危险写法必须被拒绝且报错可读，合法算子必须通过。

被测链路是**工具层** ``dataset_split(mode="field")`` —— 验收关注的是「用户写了一个
危险表达式时，MCP 客户端收到的是可读的 ``expression_invalid`` + 示例」，
而不是 core 解析器的内部异常。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures.make_fixtures import write_all  # noqa: E402

from risklab_mdl.tools.dataset import dataset_load, project_init  # noqa: E402
from risklab_mdl.tools.split import dataset_split  # noqa: E402


def bootstrap(tmp_path: Path) -> tuple[Path, str]:
    paths = write_all(tmp_path / "data")
    csv = paths["binary"]
    init = project_init(str(csv))
    assert init["status"] == "ok", init
    assert dataset_load(str(csv))["status"] == "ok"
    return csv, init["data"]["project_dir"]


def field_config(train: str, val: str = "month == '2026-03'", test: str = "") -> dict:
    field = {"column": "month", "expr_train": train, "expr_val": val}
    if test:
        field["expr_test"] = test
    return {"mode": "field", "field": field}


# --------------------------------------------------------------------------- #
# 验收 ②：危险/非法表达式一律拒绝
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "f_signal.amount > 1",
        "y = 1",
        "f_signal > len(y)",
        "open('C:/tmp/x')",
        "f_signal > 1; y == 0",
        "y == 1 & (f_signal > 1",
    ],
)
def test_dangerous_expressions_are_rejected(tmp_path: Path, expression: str) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = dataset_split(project_dir, field_config(expression))
    assert payload["status"] == "error", payload
    error = payload["error"]
    assert error["code"] == "expression_invalid", error
    assert error["message"], "错误消息不能为空"
    # 可读性：带原表达式、位置、支持语法与示例
    assert "位置" in error["message"], error["message"]
    assert "只支持" in error["message"], error["message"]
    assert "示例" in error["hint"], error["hint"]
    # 失败可见：不产生 split.json
    assert not (Path(project_dir) / "split" / "split.json").exists()


def test_unknown_column_in_expression_is_reported(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = dataset_split(project_dir, field_config("not_a_column == 1"))
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "expression_invalid"
    assert "not_a_column" in payload["error"]["message"]
    assert "可用列示例" in payload["error"]["hint"]


# --------------------------------------------------------------------------- #
# 验收 ②：合法算子必须通过
# --------------------------------------------------------------------------- #
def test_legal_in_and_comparison_expressions_pass(tmp_path: Path) -> None:
    csv, project_dir = bootstrap(tmp_path)
    frame = pd.read_csv(csv, encoding="utf-8")
    payload = dataset_split(
        project_dir,
        {
            "mode": "field",
            "field": {
                "column": "month",
                "expr_train": "month in ('2026-01', '2026-02') & f_signal >= -100",
                "expr_val": "month == '2026-03' | (month == '2026-01' & f_signal < -100)",
                "expr_test": "month not in ('2026-01', '2026-02', '2026-03')",
            },
        },
    )
    assert payload["status"] == "ok", payload
    counts = payload["data"]["counts"]
    assert counts["train"] == int(frame["month"].isin(["2026-01", "2026-02"]).sum())
    assert counts["val"] == int((frame["month"] == "2026-03").sum())
    assert counts["test"] == 0
    assert counts["train"] + counts["val"] == len(frame)


def test_null_comparison_and_parens_pass(tmp_path: Path) -> None:
    csv, project_dir = bootstrap(tmp_path)
    frame = pd.read_csv(csv, encoding="utf-8")
    payload = dataset_split(
        project_dir,
        {
            "mode": "field",
            "field": {
                "expr_train": "f_missing != null",
                "expr_val": "f_missing == null",
            },
        },
    )
    assert payload["status"] == "ok", payload
    counts = payload["data"]["counts"]
    assert counts["train"] == int(frame["f_missing"].notna().sum())
    assert counts["val"] == int(frame["f_missing"].isna().sum())
    # 覆盖全部行 ⇒ 不应出现「未覆盖」告警
    assert not any("未覆盖" in w for w in payload["warnings"]), payload["warnings"]


def test_expression_with_time_inequality_passes(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = dataset_split(
        project_dir,
        {
            "mode": "field",
            "field": {
                "expr_train": "f_signal >= 0 & f_weak <= 3",
                "expr_val": "f_signal < 0 | f_weak > 3",
            },
        },
    )
    assert payload["status"] == "ok", payload
    assert sum(payload["data"]["counts"].values()) == payload["data"]["rows"]


def test_expression_recorded_in_split_json(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = dataset_split(
        project_dir,
        field_config(
            "month in ('2026-01') & f_signal >= 0", val="month == '2026-02'"
        ),
    )
    assert payload["status"] == "ok", payload
    doc = json.loads(
        (Path(project_dir) / "split" / "split.json").read_text(encoding="utf-8")
    )
    assert doc["mode"] == "field"
    assert doc["spec"]["expression"]["train"] == "month in ('2026-01') & f_signal >= 0"
    assert doc["spec"]["column"] == "month"
    assert doc["fingerprint"] == payload["data"]["fingerprint"]
