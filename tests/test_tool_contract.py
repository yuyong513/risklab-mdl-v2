"""工具契约测试（``risklab_mdl.tools.dataset`` / ``feature``）。

对齐 ``docs/tool-contract.md``：
* §1 三态协议：``{ok, status, data, artifacts, warnings, error}`` 六键齐全、互相自洽；
* §3.1–3.2：各工具 ``data`` 的字段与产物路径；
* §4：``mdl-<stem>/`` 目录、manifest schema 与 stages。

另外包含一条**真实数据全链路**用例（``sample_2000.csv``）：只有样例数据在本机
存在时才跑，用于验收「产物齐全 + 总耗时 ≤ 120s + HTML 离线」。样例始终先复制到
临时目录，**绝不在旧项目里创建 mdl-***。
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures.make_fixtures import write_all  # noqa: E402

from risklab_mdl.artifacts.manifest import MANIFEST_SCHEMA, STAGES  # noqa: E402
from risklab_mdl.charts import find_external_links  # noqa: E402
from risklab_mdl.tools.dataset import (  # noqa: E402
    dataset_columns,
    dataset_load,
    dataset_profile,
    project_init,
)
from risklab_mdl.tools.feature import feature_analyze, feature_charts  # noqa: E402

ENVELOPE_KEYS = {"ok", "status", "data", "artifacts", "warnings", "error"}
BINARY_TARGET = {"column": "y", "task": "binary", "labels": {"1": "bad", "0": "good"}}

#: 项目根 → 旧项目的样例数据（只读引用；不存在时相关用例跳过）。
SAMPLE_CSV = (
    Path(__file__).resolve().parents[2] / "risklab-mcp-main" / "examples" / "sample_2000.csv"
)


def assert_envelope(payload: dict) -> None:
    """三态协议自洽性断言（每个工具返回都要过）。"""
    assert isinstance(payload, dict), f"工具必须返回 dict，实际 {type(payload).__name__}"
    assert set(payload) == ENVELOPE_KEYS, f"载荷键不符：{sorted(payload)}"
    assert payload["status"] in ("ok", "needs_user_input", "error")
    assert isinstance(payload["data"], dict)
    assert isinstance(payload["artifacts"], list)
    assert isinstance(payload["warnings"], list)
    assert all(isinstance(item, str) for item in payload["artifacts"])
    assert all(isinstance(item, str) for item in payload["warnings"])
    assert payload["ok"] is (payload["status"] == "ok")
    if payload["status"] == "error":
        assert isinstance(payload["error"], dict)
        assert payload["error"].get("code") and payload["error"].get("message")
    else:
        assert payload["error"] is None
    # 产物路径一律 POSIX 相对路径（相对项目目录）
    for path in payload["artifacts"]:
        assert "\\" not in path, f"artifacts 必须是 POSIX 分隔符：{path}"


def bootstrap(tmp_path: Path, name: str = "binary") -> tuple[Path, str]:
    """造数据 → project_init → dataset_load，返回 ``(csv_path, project_dir)``。"""
    paths = write_all(tmp_path / "data")
    csv = paths[name]
    init = project_init(str(csv))
    assert_envelope(init)
    assert init["status"] == "ok", init
    load = dataset_load(str(csv))
    assert load["status"] == "ok", load
    return csv, init["data"]["project_dir"]


# --------------------------------------------------------------------------- #
# project_init（§3.1 / §4）
# --------------------------------------------------------------------------- #
def test_project_init_layout_and_manifest(tmp_path: Path) -> None:
    paths = write_all(tmp_path / "data")
    csv = paths["binary"]
    first = project_init(str(csv))
    assert_envelope(first)
    assert first["status"] == "ok"
    assert first["artifacts"] == ["manifest.json"]
    data = first["data"]
    assert data["created"] is True
    project_dir = Path(data["project_dir"])
    assert project_dir.name == "mdl-binary_small", "项目目录 = mdl-<CSV 名去扩展名>"
    assert project_dir.parent == csv.parent, "项目目录必须与 CSV 同目录"

    for sub in (
        "data_analysis",
        "feature_analysis",
        "feature_selection",
        "preprocess",
        "split",
        "runs",
        "final",
        "reports",
    ):
        assert (project_dir / sub).is_dir(), f"缺少固定子目录 {sub}"
    assert (project_dir / "data_analysis" / "charts").is_dir()
    assert (project_dir / "feature_analysis" / "charts").is_dir()

    manifest = json.loads((project_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert set(manifest["stages"]) == set(STAGES)
    assert all(entry["done"] is False for entry in manifest["stages"].values())
    assert manifest["runs"] == [] and manifest["confirmed_run"] is None
    assert manifest["project"]["stem"] == "binary_small"
    assert Path(manifest["project"]["csv_path"]).name == csv.name

    # 复用不覆盖
    second = project_init(str(csv))
    assert second["status"] == "ok"
    assert second["data"]["created"] is False
    assert second["artifacts"] == []
    assert any("复用" in w for w in second["warnings"])


def test_project_init_missing_csv(tmp_path: Path) -> None:
    payload = project_init(str(tmp_path / "nope.csv"))
    assert_envelope(payload)
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "path_not_readable"


# --------------------------------------------------------------------------- #
# dataset_load / dataset_columns
# --------------------------------------------------------------------------- #
def test_dataset_load_payload_and_manifest_roles(tmp_path: Path) -> None:
    paths = write_all(tmp_path / "data")
    csv = paths["binary"]
    payload = dataset_load(str(csv))
    assert_envelope(payload)
    assert payload["status"] == "ok"
    assert payload["artifacts"] == ["manifest.json"]
    data = payload["data"]
    for key in ("project_dir", "encoding", "rows", "columns", "roles", "dtypes"):
        assert key in data, f"data 缺少 {key}"
    assert data["encoding"] == "utf-8"
    assert data["rows"] == 400
    assert data["roles"]["id"] == "id"
    assert data["roles"]["time"] == "month"
    assert data["roles"]["split"] == "split"
    assert "y" in data["roles"]["target_candidates"]
    assert data["role_reasons"]["target_candidates"]["y"]
    assert any("角色为推断" in w for w in payload["warnings"])

    manifest = json.loads(
        (Path(data["project_dir"]) / "manifest.json").read_text(encoding="utf-8")
    )
    roles = manifest["project"]["roles"]
    assert roles["id"] == "id" and roles["time"] == "month"
    assert "f_signal" in roles["features"], "数值特征要写进 manifest.roles.features"


def test_dataset_columns_fields(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = dataset_columns(project_dir)
    assert_envelope(payload)
    assert payload["status"] == "ok"
    columns = payload["data"]["columns"]
    assert len(columns) == 11
    first = columns[0]
    for key in ("name", "dtype", "role", "nunique", "missing_rate"):
        assert key in first, f"columns[] 缺少 {key}"
    roles = {row["name"]: row["role"] for row in columns}
    assert roles["id"] == "id" and roles["y"] == "target"
    assert payload["data"]["confirmed_target"] is None
    assert any("角色为推断" in w for w in payload["warnings"])


def test_dataset_columns_missing_project(tmp_path: Path) -> None:
    payload = dataset_columns(str(tmp_path / "mdl-none"))
    assert_envelope(payload)
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "project_dir_not_found"


# --------------------------------------------------------------------------- #
# dataset_profile（§3.1）
# --------------------------------------------------------------------------- #
def test_dataset_profile_needs_user_input(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = dataset_profile(project_dir)
    assert_envelope(payload)
    assert payload["status"] == "needs_user_input"
    assert payload["ok"] is False and payload["error"] is None
    question = payload["data"]["question"]
    assert question["id"] == "dataset_confirm_target"
    assert question["options"] and question["fields"]
    context = payload["data"]["context"]
    assert "y" in context["candidates"]
    assert context["label_counts"] == {"0": 200, "1": 200}


def test_dataset_profile_contract_artifacts(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = dataset_profile(project_dir, {"target": BINARY_TARGET})
    assert_envelope(payload)
    assert payload["status"] == "ok", payload
    data = payload["data"]
    for key in ("target_distribution", "missing_summary", "chart_count", "shape"):
        assert key in data, f"data 缺少 {key}"
    assert data["target_distribution"]["bad_rate"] == pytest.approx(0.5)
    assert data["chart_count"] >= 3, "画像至少要出 3 张图"
    # 验收 ③：角色是推断出来的，必须显式告警要求用户确认（不阻断）
    assert any("角色为推断，需用户确认" in w for w in payload["warnings"]), payload["warnings"]

    root = Path(project_dir)
    for rel in (
        "data_analysis/profile.json",
        "data_analysis/describe.csv",
        "data_analysis/target_distribution.csv",
        "data_analysis/index.html",
    ):
        assert (root / rel).is_file(), f"缺少产物 {rel}"
    pngs = list((root / "data_analysis" / "charts").glob("*.png"))
    assert len(pngs) >= 3, f"PNG 数量不足：{[p.name for p in pngs]}"
    assert "data_analysis/index.html" in payload["artifacts"]
    assert all(not rel.startswith("/") for rel in payload["artifacts"])

    html = (root / "data_analysis" / "index.html").read_text(encoding="utf-8")
    assert find_external_links(html) == [], "画像 HTML 不允许出现外链"
    for name in pngs:
        assert name.name in html, f"HTML 未引用 {name.name}"

    profile = json.loads((root / "data_analysis" / "profile.json").read_text(encoding="utf-8"))
    assert profile["shape"] == {"rows": 400, "columns": 11}
    assert profile["target"]["column"] == "y" and profile["target"]["task"] == "binary"
    describe = pd.read_csv(root / "data_analysis" / "describe.csv", encoding="utf-8")
    assert "f_signal" in set(describe["column"])
    distribution = pd.read_csv(root / "data_analysis" / "target_distribution.csv", encoding="utf-8")
    assert list(distribution.columns) == ["label", "label_name", "count", "share"]

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["data_analysis"]["done"] is True
    assert manifest["stages"]["data_analysis"]["path"] == "data_analysis/"
    assert manifest["project"]["target"]["column"] == "y"
    assert manifest["project"]["target"]["labels"] == {"1": "bad", "0": "good"}


def test_dataset_profile_without_time_column_skips_charts(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path, name="no_time")
    payload = dataset_profile(project_dir, {"target": BINARY_TARGET})
    assert_envelope(payload)
    assert payload["status"] == "ok", payload
    assert payload["data"]["has_time_column"] is False
    assert any("无时间列" in w and "跳过" in w for w in payload["warnings"])
    charts = [p.name for p in (Path(project_dir) / "data_analysis" / "charts").glob("*.png")]
    assert "monthly_bad_rate.png" not in charts, "没有时间列就不该伪造按月图"


def test_dataset_profile_multiclass(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path, name="multiclass")
    payload = dataset_profile(
        project_dir, {"target": {"column": "y", "task": "multiclass", "labels": None}}
    )
    assert_envelope(payload)
    assert payload["status"] == "ok", payload
    dist = payload["data"]["target_distribution"]
    assert dist["task"] == "multiclass" and dist["n_classes"] == 3
    assert "bad_rate" not in dist


# --------------------------------------------------------------------------- #
# feature_analyze（§3.2）
# --------------------------------------------------------------------------- #
def test_feature_analyze_contract_artifacts(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = feature_analyze(
        project_dir,
        {"method": "quantile", "bins": 5, "target": BINARY_TARGET,
         "features": ["f_signal", "f_weak", "f_text", "f_const"]},
    )
    assert_envelope(payload)
    assert payload["status"] == "ok", payload
    data = payload["data"]
    assert data["feature_count"] == 4
    assert data["analyzed_count"] == 2
    assert set(data["iv_top"][0]) == {"feature", "iv"}
    assert {item["feature"] for item in data["skipped"]} == {"f_text", "f_const"}

    root = Path(project_dir)
    for rel in (
        "feature_analysis/iv_table.csv",
        "feature_analysis/woe_detail.csv",
        "feature_analysis/index.html",
    ):
        assert (root / rel).is_file(), f"缺少产物 {rel}"
    assert (root / "feature_analysis" / "binning" / "f_signal.json").is_file()
    assert not (root / "feature_analysis" / "binning" / "f_text.json").exists(), (
        "被跳过的特征不该产出分箱规格"
    )
    pngs = list((root / "feature_analysis" / "charts").glob("*.png"))
    assert pngs, "至少要有 IV 分布图"
    iv_table = pd.read_csv(root / "feature_analysis" / "iv_table.csv", encoding="utf-8")
    assert list(iv_table.columns)[:2] == ["feature", "iv"]
    assert len(iv_table) == 4
    woe = pd.read_csv(root / "feature_analysis" / "woe_detail.csv", encoding="utf-8")
    assert {"feature", "bin", "woe", "iv_contribution"} <= set(woe.columns)

    html = (root / "feature_analysis" / "index.html").read_text(encoding="utf-8")
    assert find_external_links(html) == []
    for name in pngs:
        assert name.name in html

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["feature_analysis"]["done"] is True
    assert manifest["stages"]["feature_analysis"]["summary"]

    spec = json.loads(
        (root / "feature_analysis" / "binning" / "f_signal.json").read_text(encoding="utf-8")
    )
    assert spec["feature"] == "f_signal" and spec["method"] == "quantile"
    assert spec["per_class"]["binary"]["edges"], "BinSpec 必须可复用（带切点）"


def test_feature_analyze_dt_method(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = feature_analyze(
        project_dir,
        {"method": "dt", "bins": 4, "target": BINARY_TARGET, "features": ["f_signal"]},
    )
    assert_envelope(payload)
    assert payload["status"] == "ok", payload
    iv_table = pd.read_csv(
        Path(project_dir) / "feature_analysis" / "iv_table.csv", encoding="utf-8"
    )
    assert iv_table.iloc[0]["method"] == "dt"
    assert int(iv_table.iloc[0]["n_bins_fitted"]) <= 4


def test_feature_analyze_multiclass_ovr(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path, name="multiclass")
    payload = feature_analyze(
        project_dir,
        {"method": "quantile", "bins": 4,
         "target": {"column": "y", "task": "multiclass", "labels": None},
         "features": ["f1", "f2"]},
    )
    assert_envelope(payload)
    assert payload["status"] == "ok", payload
    iv_table = pd.read_csv(
        Path(project_dir) / "feature_analysis" / "iv_table.csv", encoding="utf-8"
    )
    class_columns = sorted(c for c in iv_table.columns if str(c).startswith("iv_"))
    assert class_columns == ["iv_0", "iv_1", "iv_2"]
    woe = pd.read_csv(
        Path(project_dir) / "feature_analysis" / "woe_detail.csv", encoding="utf-8"
    )
    # CSV 往返后 pandas 会把 "0"/"1"/"2" 再推断成整数，比较时统一成字符串
    assert {str(v) for v in woe["class"]} == {"0", "1", "2"}


def test_feature_analyze_error_paths(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    bad_method = feature_analyze(project_dir, {"method": "magic", "target": BINARY_TARGET})
    assert_envelope(bad_method)
    assert bad_method["status"] == "error"
    assert bad_method["error"]["code"] == "config_invalid"

    bad_feature = feature_analyze(
        project_dir, {"method": "quantile", "target": BINARY_TARGET, "features": ["nope"]}
    )
    assert bad_feature["status"] == "error"
    assert bad_feature["error"]["code"] == "config_invalid"

    no_edges = feature_analyze(
        project_dir, {"method": "custom", "target": BINARY_TARGET, "features": ["f_signal"]}
    )
    assert no_edges["status"] == "error"
    assert no_edges["error"]["code"] == "config_invalid"

    empty = feature_analyze(
        project_dir, {"method": "quantile", "target": BINARY_TARGET, "features": []}
    )
    assert empty["status"] == "error"
    assert empty["error"]["code"] == "features_empty"

    no_target = feature_analyze(str(tmp_path / "mdl-none"), {"method": "quantile"})
    assert no_target["status"] == "error"
    assert no_target["error"]["code"] == "project_dir_not_found"


def test_feature_analyze_needs_target(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = feature_analyze(project_dir, {"method": "quantile"})
    assert_envelope(payload)
    assert payload["status"] == "needs_user_input"
    assert payload["data"]["question"]["id"] == "dataset_confirm_target"


# --------------------------------------------------------------------------- #
# feature_charts（§3.2）
# --------------------------------------------------------------------------- #
def test_feature_charts_requires_analysis(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    payload = feature_charts(project_dir, {})
    assert_envelope(payload)
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "analysis_failed"
    assert "feature_analyze" in payload["error"]["hint"]


def test_feature_charts_generates_iv_corr_woe(tmp_path: Path) -> None:
    _, project_dir = bootstrap(tmp_path)
    feature_analyze(
        project_dir,
        {"method": "quantile", "bins": 5, "target": BINARY_TARGET,
         "features": ["f_signal", "f_weak", "f_noise", "f_dup"]},
    )
    payload = feature_charts(project_dir, {"max_features": 4})
    assert_envelope(payload)
    assert payload["status"] == "ok", payload
    charts = payload["data"]["charts"]
    assert "iv_distribution.png" in charts
    assert "correlation_heatmap.png" in charts
    assert any(name.startswith("woe_") for name in charts)
    root = Path(project_dir)
    for name in charts:
        assert (root / "feature_analysis" / "charts" / name).is_file()
    html = (root / "feature_analysis" / "index.html").read_text(encoding="utf-8")
    assert find_external_links(html) == []


# --------------------------------------------------------------------------- #
# 真实样例全链路（验收 ①②）
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not SAMPLE_CSV.is_file(), reason="样例数据 sample_2000.csv 不在本机")
def test_full_chain_on_sample_2000(tmp_path: Path) -> None:
    """project_init → dataset_profile → feature_analyze → feature_charts。

    ★ 样例先复制到临时目录：旧项目 ``risklab-mcp-main/`` 严格只读，
      绝不在里面创建 ``mdl-*``。
    """
    csv = tmp_path / "sample_2000.csv"
    shutil.copyfile(SAMPLE_CSV, csv)
    started = time.time()

    init = project_init(str(csv))
    assert init["status"] == "ok", init
    project_dir = init["data"]["project_dir"]
    assert Path(project_dir).parent == tmp_path, "项目目录不能落在旧项目里"

    profile = dataset_profile(project_dir, {"target": BINARY_TARGET})
    assert profile["status"] == "ok", profile
    assert profile["data"]["target_distribution"]["bad_rate"] == pytest.approx(35 / 2000)
    assert profile["data"]["chart_count"] >= 3

    analysis = feature_analyze(
        project_dir,
        {"method": "quantile", "bins": 8, "target": BINARY_TARGET,
         "max_features_for_charts": 30},
    )
    assert analysis["status"] == "ok", analysis
    root = Path(project_dir)
    iv_table = pd.read_csv(root / "feature_analysis" / "iv_table.csv", encoding="utf-8")
    assert len(iv_table) == 692, f"692 个 f_* 特征应逐行在表里，实际 {len(iv_table)}"
    assert iv_table["skipped"].sum() == 1, "f_0423 是文本列，应被跳过并给原因"
    assert analysis["data"]["feature_count"] == 692

    charts = feature_charts(project_dir, {})
    assert charts["status"] == "ok", charts

    elapsed = time.time() - started
    assert elapsed <= 120, f"全链路耗时 {elapsed:.1f}s 超过 120s 预算"

    # 产物齐全（验收 ①②）
    for rel in (
        "data_analysis/profile.json",
        "data_analysis/describe.csv",
        "data_analysis/target_distribution.csv",
        "data_analysis/index.html",
        "feature_analysis/iv_table.csv",
        "feature_analysis/woe_detail.csv",
        "feature_analysis/index.html",
    ):
        assert (root / rel).is_file(), f"缺少产物 {rel}"
    profile_pngs = list((root / "data_analysis" / "charts").glob("*.png"))
    assert len(profile_pngs) >= 3
    assert list((root / "feature_analysis" / "charts").glob("*.png"))
    for html_path in (
        root / "data_analysis" / "index.html",
        root / "feature_analysis" / "index.html",
    ):
        text = html_path.read_text(encoding="utf-8")
        assert find_external_links(text) == [], f"{html_path.name} 有外链"
        for ref in [seg.split('"')[0] for seg in text.split('src="')[1:]]:
            assert (html_path.parent / ref).is_file(), f"{html_path.name} 引用了不存在的图 {ref}"

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["data_analysis"]["done"] is True
    assert manifest["stages"]["feature_analysis"]["done"] is True
