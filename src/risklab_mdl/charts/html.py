"""单文件 HTML 汇总报告（**离线**：不引用任何外部资源）。

为什么"单文件 + 无外链"
-----------------------
产物要能在**离线/内网**环境直接双击打开。因此：

* 样式全部内联 ``<style>``，不引 CDN；
* 图片用**相对路径** ``charts/xxx.png``（与本 HTML 同目录结构），不写绝对路径；
* 生成后做一次硬检查：正文若出现 ``http://`` / ``https://`` 一律**去链化**
  （替换成 ``hxxp://``），并在返回的 ``warnings`` 里说明 —— 这样即使特征名
  里带了 URL 也不会让报告变成"有外链"。
"""

from __future__ import annotations

import html as _html
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..platform.fs import atomic_write_text

__all__ = [
    "build_report_html",
    "find_external_links",
]

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { font-family: "Microsoft YaHei", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
       margin: 0; padding: 24px 32px 64px; color: #1f2933; background: #f7f9fb; }
h1 { font-size: 22px; margin: 0 0 6px; }
h2 { font-size: 17px; margin: 28px 0 10px; padding-bottom: 6px; border-bottom: 1px solid #d9e2ec; }
h3 { font-size: 14px; margin: 18px 0 8px; color: #334e68; }
.subtitle { color: #627d98; font-size: 13px; margin-bottom: 18px; }
.card { background: #fff; border: 1px solid #e4ebf2; border-radius: 8px; padding: 16px 18px; margin: 14px 0; }
.metrics { display: flex; flex-wrap: wrap; gap: 12px; margin: 10px 0 4px; }
.metric { background: #eef4fb; border-radius: 6px; padding: 8px 14px; min-width: 130px; }
.metric .label { display: block; font-size: 11px; color: #627d98; }
.metric .value { font-size: 16px; font-weight: 600; }
ul { margin: 6px 0 6px 18px; padding: 0; }
li { margin: 3px 0; font-size: 13px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; margin: 8px 0 4px; }
th, td { border: 1px solid #e1e8ef; padding: 5px 8px; text-align: left; }
th { background: #eef4fb; position: sticky; top: 0; }
tbody tr:nth-child(even) { background: #fafcfe; }
figure { margin: 14px 0; }
figure img { max-width: 100%; border: 1px solid #e4ebf2; border-radius: 6px; background: #fff; }
figcaption { font-size: 12px; color: #627d98; margin-top: 6px; }
.note { font-size: 12px; color: #8a6d3b; background: #fff8e6; border: 1px solid #f5e3b3;
        border-radius: 6px; padding: 8px 12px; margin: 10px 0; }
footer { margin-top: 36px; font-size: 11px; color: #9fb3c8; }
"""


def find_external_links(text: str) -> List[str]:
    """返回正文里出现的外链前缀（``http://`` / ``https://``），用于断言离线。"""
    hits: List[str] = []
    lowered = text.lower()
    for token in ("http://", "https://"):
        if token in lowered:
            hits.append(token)
    return hits


def _defang(text: str) -> str:
    """去链化：把 http(s):// 换成 hxxp://，保证报告里没有可用外链。"""
    return text.replace("http://", "hxxp://").replace("https://", "hxxp://")


def _render_table(table: Dict[str, Any]) -> str:
    columns = [str(c) for c in (table.get("columns") or [])]
    rows: Iterable[Sequence[Any]] = table.get("rows") or []
    if not columns:
        return ""
    head = "".join(f"<th>{_html.escape(c)}</th>" for c in columns)
    body_parts: List[str] = []
    for row in rows:
        cells = "".join(f"<td>{_html.escape('' if v is None else str(v))}</td>" for v in row)
        body_parts.append(f"<tr>{cells}</tr>")
    if not body_parts:
        body_parts.append(
            f"<tr><td colspan=\"{len(columns)}\">（无数据）</td></tr>"
        )
    return (
        "<table><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + "".join(body_parts)
        + "</tbody></table>"
    )


def build_report_html(
    *,
    title: str,
    out_path: Path,
    sections: Sequence[Dict[str, Any]],
    subtitle: Optional[str] = None,
    charts_rel: str = "charts",
    generated_at: Optional[str] = None,
    footer: str = "risklab-mdl-v2 自动生成",
) -> Dict[str, Any]:
    """生成单文件 HTML 报告，返回 ``{path, chart_refs, warnings}``。

    ``sections`` 每项支持：``title`` / ``text``（字符串列表）/ ``metrics``
    （``[{label, value}]``）/ ``table``（``{columns, rows}``）/ ``charts``
    （``[{file, caption}]``，``file`` 是 ``charts_rel`` 目录下的 PNG 文件名）。
    """
    chart_refs: List[str] = []
    parts: List[str] = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_html.escape(title)}</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        f"<h1>{_html.escape(title)}</h1>",
    ]
    stamp = generated_at or datetime.now().astimezone().isoformat(timespec="seconds")
    meta = f"生成时间 {stamp}"
    if subtitle:
        meta = f"{subtitle} ｜ {meta}"
    parts.append(f'<div class="subtitle">{_html.escape(meta)}</div>')

    for section in sections:
        parts.append(f"<h2>{_html.escape(str(section.get('title', '')))}</h2>")
        text_lines = section.get("text") or []
        if text_lines:
            parts.append('<div class="card"><ul>')
            for line in text_lines:
                parts.append(f"<li>{_html.escape(str(line))}</li>")
            parts.append("</ul></div>")
        metrics = section.get("metrics") or []
        if metrics:
            parts.append('<div class="metrics">')
            for metric in metrics:
                parts.append(
                    '<div class="metric"><span class="label">'
                    + _html.escape(str(metric.get("label", "")))
                    + '</span><span class="value">'
                    + _html.escape(str(metric.get("value", "")))
                    + "</span></div>"
                )
            parts.append("</div>")
        table = section.get("table")
        if table:
            parts.append('<div class="card">' + _render_table(table) + "</div>")
        for chart in section.get("charts") or []:
            file_name = str(chart.get("file", ""))
            if not file_name:
                continue
            rel = f"{charts_rel.rstrip('/')}/{file_name}"
            chart_refs.append(rel)
            parts.append("<figure>")
            parts.append(
                f'<img src="{_html.escape(rel)}" alt="{_html.escape(str(chart.get("caption", file_name)))}">'
            )
            caption = chart.get("caption") or file_name
            parts.append(f"<figcaption>{_html.escape(str(caption))}</figcaption>")
            parts.append("</figure>")
        note = section.get("note")
        if note:
            parts.append(f'<div class="note">{_html.escape(str(note))}</div>')

    parts.append(f"<footer>{_html.escape(footer)}</footer>")
    parts.append("</body>")
    parts.append("</html>")

    document = "\n".join(parts) + "\n"
    warnings: List[str] = []
    if find_external_links(document):
        document = _defang(document)
        warnings.append(
            "报告中原本出现 http:// / https:// 文本，已去链化（替换为 hxxp://）以满足离线要求"
        )

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 走平台层原子写：临时文件 + os.replace，避免中断留下半截 HTML
    atomic_write_text(path, document)
    return {"path": path, "chart_refs": chart_refs, "warnings": warnings}
