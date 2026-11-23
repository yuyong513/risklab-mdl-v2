"""``risklab_mdl.charts`` —— 图表层。

matplotlib **Agg** backend 生成 PNG（120 dpi）+ 单文件 HTML（无外链），
中文字体走与 ``doctor`` 相同的 fallback 链（缺字体只告警、不报错）。

模块划分：

* :mod:`risklab_mdl.charts.figures` —— 各图表函数 + :class:`ChartRenderer`；
* :mod:`risklab_mdl.charts.html` —— 单文件 HTML 汇总与离线断言。

本包**不做业务取数**：只接受已经算好的结构化结果（画像/IV 表/WOE 明细），
因此可以被任意分析步骤复用，也便于单测直接喂构造数据。
"""

from .figures import (
    PNG_DPI,
    ChartRenderer,
    figure_paths,
    sanitize_filename,
    setup_matplotlib,
)
from .html import build_report_html, find_external_links

__all__ = [
    "PNG_DPI",
    "ChartRenderer",
    "build_report_html",
    "figure_paths",
    "find_external_links",
    "sanitize_filename",
    "setup_matplotlib",
]
