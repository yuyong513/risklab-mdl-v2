"""图表层单测（``risklab_mdl.charts``）。

覆盖：Agg backend、PNG 120 dpi、中文字体 fallback 不阻断、7 个图表函数、
空数据返回 ``None``（不伪造图）、单文件 HTML 的离线断言。
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures.make_fixtures import binary_frame  # noqa: E402

from risklab_mdl.charts import (  # noqa: E402
    PNG_DPI,
    ChartRenderer,
    build_report_html,
    find_external_links,
    sanitize_filename,
    setup_matplotlib,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def png_dpi(path: Path) -> int | None:
    """从 PNG 的 ``pHYs`` 块读出 dpi（无该块返回 None）。"""
    data = path.read_bytes()
    assert data.startswith(PNG_MAGIC), f"{path} 不是 PNG"
    index = data.find(b"pHYs")
    if index < 0:
        return None
    x_pixels_per_unit, _y, unit = struct.unpack(">IIB", data[index + 4 : index + 13])
    if unit != 1:  # 1 = 米
        return None
    return round(x_pixels_per_unit * 0.0254)


def assert_png(path: Path, *, dpi: int = PNG_DPI) -> None:
    assert path.is_file(), f"图表未生成：{path}"
    assert path.stat().st_size > 1000, f"PNG 太小，可能是空图：{path}"
    measured = png_dpi(path)
    assert measured == dpi, f"{path.name} 的 dpi={measured}，期望 {dpi}"


# --------------------------------------------------------------------------- #
# 环境
# --------------------------------------------------------------------------- #
def test_setup_uses_agg_backend_and_reports_font() -> None:
    plt, font, warnings = setup_matplotlib()
    assert matplotlib.get_backend().lower() == "agg"
    assert isinstance(warnings, list)
    # 本机装了 Microsoft YaHei 等中文字体；缺字体也允许（只要求不报错 + 有告警）
    if font is None:
        assert warnings, "没有中文字体时必须给出告警（不静默）"
    plt.close("all")


def test_renderer_saves_120dpi(tmp_path: Path) -> None:
    renderer = ChartRenderer()
    payload = {
        "column": "y",
        "task": "binary",
        "labels": {"1": "bad", "0": "good"},
        "counts": {"0": 200, "1": 200},
        "shares": {"0": 0.5, "1": 0.5},
        "bad_rate": 0.5,
        "bad_label": "1",
    }
    path = renderer.target_distribution(payload, tmp_path / "charts" / "target.png")
    assert path is not None
    assert_png(path)
    assert renderer.rendered == [path]


# --------------------------------------------------------------------------- #
# 逐个图表函数
# --------------------------------------------------------------------------- #
def test_missing_rate_chart(tmp_path: Path) -> None:
    renderer = ChartRenderer()
    rows = [
        {"column": "a", "missing_rate": 0.5},
        {"column": "b", "missing_rate": 0.2},
        {"column": "c", "missing_rate": 0.0},
    ]
    path = renderer.missing_rate(rows, tmp_path / "missing.png")
    assert path is not None
    assert_png(path)
    assert renderer.missing_rate([{"column": "x", "missing_rate": 0.0}], tmp_path / "none.png") is None


def test_feature_distributions_chart(tmp_path: Path) -> None:
    frame = binary_frame()
    renderer = ChartRenderer()
    path = renderer.feature_distributions(
        frame, ["f_signal", "f_text", "f_const"], tmp_path / "dist.png", max_features=3
    )
    assert path is not None
    assert_png(path)


def test_iv_distribution_chart_with_thresholds(tmp_path: Path) -> None:
    renderer = ChartRenderer()
    rows = [
        {"feature": "a", "iv": 0.5, "skipped": False},
        {"feature": "b", "iv": 0.15, "skipped": False},
        {"feature": "c", "iv": 0.01, "skipped": False},
        {"feature": "d", "iv": float("nan"), "skipped": True},
    ]
    path = renderer.iv_distribution(rows, tmp_path / "iv.png")
    assert path is not None
    assert_png(path)
    assert renderer.iv_distribution([], tmp_path / "empty.png") is None


def test_correlation_heatmap_chart(tmp_path: Path) -> None:
    frame = binary_frame()
    renderer = ChartRenderer()
    path = renderer.correlation_heatmap(
        frame, ["f_signal", "f_dup", "f_weak", "f_noise"], tmp_path / "corr.png"
    )
    assert path is not None
    assert_png(path)
    # 只有一个数值列 ⇒ 不出图（不伪造）
    assert renderer.correlation_heatmap(frame, ["f_signal"], tmp_path / "one.png") is None


def test_monthly_bad_rate_chart(tmp_path: Path) -> None:
    rows = [
        {"month": "2026-01", "count": 100, "bad_count": 10, "bad_rate": 0.1},
        {"month": "2026-02", "count": 120, "bad_count": 18, "bad_rate": 0.15},
    ]
    renderer = ChartRenderer()
    path = renderer.monthly_bad_rate(rows, tmp_path / "monthly.png")
    assert path is not None
    assert_png(path)
    assert renderer.monthly_bad_rate([], tmp_path / "none.png") is None


def test_woe_bins_chart(tmp_path: Path) -> None:
    detail = pd.DataFrame(
        {
            "feature": ["f"] * 3,
            "class": ["binary"] * 3,
            "bin": [0, 1, 2],
            "bin_index": [0, 1, 2],
            "bin_label": ["(-inf, 0]", "(0, 1]", "(1, inf]"],
            "total": [100, 150, 150],
            "bad": [10, 45, 90],
            "good": [90, 105, 60],
            "bad_rate": [0.1, 0.3, 0.6],
            "share": [0.25, 0.375, 0.375],
            "woe": [-1.0, 0.0, 1.0],
            "iv_contribution": [0.1, 0.0, 0.25],
            "smoothed": [False] * 3,
            "note": [""] * 3,
        }
    )
    renderer = ChartRenderer()
    path = renderer.woe_bins(detail, "f", tmp_path / "woe.png")
    assert path is not None
    assert_png(path)
    assert renderer.woe_bins(detail, "missing_feature", tmp_path / "none.png") is None


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def test_html_report_is_offline_and_refs_exist(tmp_path: Path) -> None:
    charts = tmp_path / "charts"
    charts.mkdir()
    (charts / "a.png").write_bytes(PNG_MAGIC + b"0" * 100)
    report = build_report_html(
        title="测试报告",
        out_path=tmp_path / "index.html",
        sections=[
            {"title": "概览", "metrics": [{"label": "行数", "value": "400"}]},
            {"title": "图表", "charts": [{"file": "a.png", "caption": "图 A"}]},
        ],
    )
    html = Path(report["path"]).read_text(encoding="utf-8")
    assert find_external_links(html) == []
    assert "http://" not in html and "https://" not in html
    assert 'src="charts/a.png"' in html
    for ref in report["chart_refs"]:
        assert (tmp_path / ref).is_file(), f"HTML 引用的本地图片不存在：{ref}"
    assert report["warnings"] == []


def test_html_defangs_external_links(tmp_path: Path) -> None:
    """特征名里带 URL 也不能让报告出现外链。"""
    report = build_report_html(
        title="含链接的报告",
        out_path=tmp_path / "defang.html",
        sections=[{"title": "说明", "text": ["来源 https://example.com/a 与 http://x/y"]}],
    )
    html = Path(report["path"]).read_text(encoding="utf-8")
    assert "https://" not in html and "http://" not in html
    assert "hxxp://" in html
    assert report["warnings"], "去链化必须留下告警"


def test_sanitize_filename() -> None:
    assert sanitize_filename("f_0001") == "f_0001"
    assert sanitize_filename("客户 号/金额") == "chart", "中文名会退化成 fallback"
    assert "/" not in sanitize_filename("a/b")
    assert sanitize_filename("") == "chart"
    assert sanitize_filename("!!!") == "chart"
