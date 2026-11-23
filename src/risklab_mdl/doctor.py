"""环境自检（doctor）：一条命令回答「这台机器能不能真跑建模」。

用法::

    python -m risklab_mdl.doctor        # 中文摘要；退出码 0=可用 / 1=有 FAIL
    risklab-mdl-doctor                  # 等价（控制台入口）
    risklab-mdl-doctor --json           # 结构化输出，供部署脚本消费

六项探针（冻结契约）
-------------------
1. ``python``          Python 版本 >= 3.11
2. ``dependencies``    依赖版本矩阵（11 个运行时依赖逐个 import + 取版本）
3. ``mcp_sdk``         MCP SDK 的服务器类可导入（``mcp.server.mcpserver.MCPServer`` 或 1.x ``FastMCP``）
4. ``matplotlib``      backend 为 Agg 且能真的画出一张 PNG
5. ``chinese_font``    中文字体 fallback 检测（缺字体只 WARN，不阻断 —— 见 R5/Q10）
6. ``artifact_dir``    产物目录可写性（临时目录里实地跑一遍 mdl-<stem> + manifest + run 目录）

★ 设计原则（沿用早期版本 doctor 的正确做法）
- 每项独立：任何一项抛异常只记成该项 FAIL，绝不让自检自己崩掉；
- FAIL / WARN 分开：FAIL=现在跑不了（退出码 1），WARN=能跑但有降级（退出码 0）；
- 只给**可复制执行**的修复命令，不只给一句错误文本；
- 只读为主：唯一的写动作是临时目录里的探针文件，验证完立即删除。
"""

from __future__ import annotations

import argparse
import json
import platform as _platform
import shutil
import sys
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .artifacts import (
    RUN_ID_PATTERN,
    allocate_run_dir,
    new_manifest,
    open_project,
    read_manifest,
    write_manifest,
)
from .sdk import describe_api

#: 项目根（src/risklab_mdl/doctor.py → src → 项目根）。源码态用于读 pyproject。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 探针顺序（输出与 JSON 均按此顺序）。
CHECK_NAMES: tuple[str, ...] = (
    "python",
    "dependencies",
    "mcp_sdk",
    "matplotlib",
    "chinese_font",
    "artifact_dir",
)

_STATUS_LABEL = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}

#: 中文字体 fallback 链（Windows / macOS / Linux 常见）。
CHINESE_FONT_CANDIDATES: tuple[str, ...] = (
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "DengXian",
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
)


@dataclass
class Check:
    """单项自检结果。"""

    name: str
    title: str
    status: str                 # ok | warn | fail
    detail: str
    hint: str = ""
    fix_commands: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)


def _pyproject_version() -> Optional[str]:
    """源码态读 pyproject 版本（wheel 安装时返回 None，不视为故障）。"""
    path = PROJECT_ROOT / "pyproject.toml"
    if not path.is_file():
        return None
    try:
        import tomllib

        with path.open("rb") as fh:
            return tomllib.load(fh)["project"]["version"]
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# 1. Python 版本
# --------------------------------------------------------------------------- #


def check_python() -> Check:
    version = _platform.python_version()
    info = sys.version_info
    pkg_version = _pyproject_version()
    detail = f"{version}（{_platform.python_implementation()}），可执行文件 {sys.executable}"
    if info < (3, 11):
        return Check(
            name="python",
            title="Python 版本",
            status="fail",
            detail=f"{detail}，低于要求的 3.11",
            hint="使用 Python 3.11+ 重建虚拟环境后再安装本项目",
            fix_commands=[
                "python -m venv .venv311",
                r".\.venv311\Scripts\activate    # Windows",
                "source .venv311/bin/activate  # macOS / Linux",
            ],
        )
    if pkg_version and pkg_version != __version__:
        # 版本号不一致是**发行/打包**问题，不影响本机能否跑建模：降级 WARN，
        # 不让它把 doctor 退出码变成 1（FAIL 只留「现在跑不了」的项）。
        return Check(
            name="python",
            title="Python 版本",
            status="warn",
            detail=(
                f"{detail}；pyproject.toml 写的是 {pkg_version}，"
                f"与包内 __version__={__version__} 不一致"
            ),
            hint=(
                "WARN 不计入 FAIL/退出码：运行不受影响，但安装出的分发版本号与运行逻辑不符，"
                "建议把两处统一（以 pyproject.toml 为准改 __init__.py，或相反）"
            ),
            fix_commands=[f'# 统一版本号：修改 src/risklab_mdl/__init__.py 的 __version__ 为 "{pkg_version}"'],
        )
    return Check(name="python", title="Python 版本", status="ok", detail=detail)


# --------------------------------------------------------------------------- #
# 2. 依赖版本矩阵
# --------------------------------------------------------------------------- #


def check_dependencies() -> Check:
    from .server import dependency_report  # 依赖清单唯一真源在 server._DEPS

    report = dependency_report()
    missing: List[str] = []
    parts: List[str] = []
    for pkg, info in report.items():
        if info["error"]:
            missing.append(pkg)
            parts.append(f"{pkg}=缺失")
        else:
            parts.append(f"{pkg}={info['version']}")
    total = len(report)
    detail = f"{total - len(missing)}/{total} 可用：" + ", ".join(parts)
    if missing:
        return Check(
            name="dependencies",
            title="依赖版本矩阵",
            status="fail",
            detail=detail + f"；缺失 {', '.join(missing)}",
            hint="在项目根执行可编辑安装（同时装上所有声明依赖）",
            fix_commands=[
                f'python -m pip install -e "{PROJECT_ROOT}"',
                f'python -m pip install {" ".join(missing)}',
            ],
            data={"report": report},
        )
    return Check(
        name="dependencies",
        title="依赖版本矩阵",
        status="ok",
        detail=detail,
        data={"report": report},
    )


# --------------------------------------------------------------------------- #
# 3. MCP SDK API
# --------------------------------------------------------------------------- #


def check_mcp_sdk() -> Check:
    api = describe_api()
    resolved = api.get("resolved_api")
    attempts = "；".join(
        f"{c['module']}.{c['class']}={'可导入' if c['importable'] else '不可用'}"
        + (f"（{c.get('error')}）" if c.get("error") else "")
        for c in api["candidates"]
    )
    if not resolved:
        return Check(
            name="mcp_sdk",
            title="MCP SDK API",
            status="fail",
            detail=f"mcp={api.get('sdk_version')}，没有可用的服务器类；尝试：{attempts}",
            hint="安装 mcp 2.x（提供 MCPServer）或 1.x（提供 FastMCP）",
            fix_commands=['python -m pip install "mcp>=2.0.0"'],
            data=api,
        )
    return Check(
        name="mcp_sdk",
        title="MCP SDK API",
        status="ok",
        detail=f"mcp={api.get('sdk_version')} → {resolved}（{attempts}）",
        data=api,
    )


# --------------------------------------------------------------------------- #
# 4. matplotlib backend
# --------------------------------------------------------------------------- #


def check_matplotlib() -> Check:
    try:
        import matplotlib

        # ★ 先记录导入时的默认 backend，再强制 Agg：早期评审指出「force 之后
        #   再判断 != agg」恒不可达（force=True 保证切换），判定已改为
        #   「force 后仍非 Agg ⇒ FAIL（防御性硬门禁，见下）」，默认 backend 只
        #   作为环境事实写进 detail。
        default_backend = matplotlib.get_backend()
        matplotlib.use("Agg", force=True)
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except Exception as exc:  # noqa: BLE001
        return Check(
            name="matplotlib",
            title="matplotlib backend",
            status="fail",
            detail=f"导入 matplotlib/Agg 失败：{type(exc).__name__}: {exc}",
            hint="重新安装 matplotlib",
            fix_commands=['python -m pip install --force-reinstall "matplotlib>=3.7"'],
        )
    try:
        figure = Figure(figsize=(2, 2), dpi=72)
        FigureCanvasAgg(figure)
        axes = figure.add_subplot(1, 1, 1)
        axes.plot([0, 1], [1, 0])
        buffer = BytesIO()
        figure.savefig(buffer, format="png")
        size = len(buffer.getvalue())
    except Exception as exc:  # noqa: BLE001
        return Check(
            name="matplotlib",
            title="matplotlib backend",
            status="fail",
            detail=f"backend={matplotlib.get_backend()}，但建图/导出 PNG 失败：{type(exc).__name__}: {exc}",
            hint="无头环境需要 Agg backend；检查 matplotlib 安装是否完整",
            fix_commands=['python -m pip install --force-reinstall "matplotlib>=3.7"'],
        )
    backend = matplotlib.get_backend()
    detail = (
        f"默认 backend={default_backend}，已强制切换为 {backend}；"
        f"已建图并导出 PNG（{size} 字节，未落盘）"
    )
    if backend.lower() != "agg":
        # 正常情况下 force=True 后必然是 Agg；走到这里说明有别的代码/插件在
        # 导入后重置了 backend，无头画图不可靠 —— 按 FAIL 处理（不是 WARN）。
        return Check(
            name="matplotlib",
            title="matplotlib backend",
            status="fail",
            detail=detail + "；force=True 后仍不是 Agg，无头画图链路不可靠",
            hint="检查是否有其他代码/插件在导入后重置 backend；本项目要求显式 matplotlib.use('Agg', force=True)",
        )
    return Check(
        name="matplotlib",
        title="matplotlib backend",
        status="ok",
        detail=detail,
        data={"backend": backend, "default_backend": default_backend, "png_bytes": size},
    )


# --------------------------------------------------------------------------- #
# 5. 中文字体
# --------------------------------------------------------------------------- #


def _installed_font_families() -> set:
    from matplotlib import font_manager

    return {f.name for f in font_manager.fontManager.ttflist}


def check_chinese_font() -> Check:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        families = _installed_font_families()
    except Exception as exc:  # noqa: BLE001
        return Check(
            name="chinese_font",
            title="中文字体 fallback",
            status="warn",
            detail=f"无法枚举系统字体：{type(exc).__name__}: {exc}",
            hint="图表中的中文可能显示为方块；安装中文字体后重试",
        )
    found = [name for name in CHINESE_FONT_CANDIDATES if name in families]
    if not found:
        return Check(
            name="chinese_font",
            title="中文字体 fallback",
            status="warn",
            detail=f"未在 {len(families)} 个已安装字体中找到候选中文（{', '.join(CHINESE_FONT_CANDIDATES[:4])} …）",
            hint="缺字体时图表中文会显示为方块；本项目按 fallback 链降级并在 warnings 中提示，不阻断",
            fix_commands=[
                "# Windows 自带微软雅黑/黑体，通常无需安装；Linux 可装：",
                "sudo apt-get install -y fonts-noto-cjk      # Debian/Ubuntu",
                "brew install --cask font-noto-sans-cjk-sc  # macOS",
            ],
            data={"available_families": len(families)},
        )
    # 真渲染一次中文：把命中的字体放进 font.sans-serif fallback 链，捕获「缺字形」告警。
    # ★ 只查字体列表不够：字体装了但没被绘图链路用上，图上照样是方块。
    missing_glyph = False
    try:
        with matplotlib.rc_context(
            {
                "font.family": "sans-serif",
                "font.sans-serif": list(found) + ["DejaVu Sans"],
                "axes.unicode_minus": False,
            }
        ):
            figure = Figure(figsize=(2, 1), dpi=72)
            FigureCanvasAgg(figure)
            axes = figure.add_subplot(1, 1, 1)
            axes.set_title("中文标题测试")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                buffer = BytesIO()
                figure.savefig(buffer, format="png")
        missing_glyph = any("missing" in str(w.message).lower() for w in caught)
    except Exception as exc:  # noqa: BLE001
        return Check(
            name="chinese_font",
            title="中文字体 fallback",
            status="warn",
            detail=f"找到中文字体 {found[0]}，但中文渲染测试失败：{type(exc).__name__}: {exc}",
            hint="图表中中文可能异常；其余功能不受影响",
            data={"found": found},
        )
    if missing_glyph:
        return Check(
            name="chinese_font",
            title="中文字体 fallback",
            status="warn",
            detail=f"字体列表含 {found[0]}，但按 fallback 链渲染中文仍出现缺字形告警（字体文件可能不完整）",
            hint="换一份完整 CJK 字体（Noto Sans CJK / 微软雅黑）；图表层会按 fallback 链降级并告警，不阻断建模",
            fix_commands=[
                "sudo apt-get install -y fonts-noto-cjk      # Debian/Ubuntu",
                "brew install --cask font-noto-sans-cjk-sc  # macOS",
            ],
            data={"found": found},
        )
    return Check(
        name="chinese_font",
        title="中文字体 fallback",
        status="ok",
        detail=f"命中 {found[0]}（候选命中 {len(found)} 个），按 fallback 链渲染中文无缺字形告警",
        data={"found": found, "fallback_chain": list(found)},
    )


# --------------------------------------------------------------------------- #
# 6. 产物目录可写性
# --------------------------------------------------------------------------- #


def check_artifact_dir() -> Check:
    tmp_root: Optional[str] = None
    try:
        tmp_root = tempfile.mkdtemp(prefix="risklab-mdl-doctor-")
        csv_path = Path(tmp_root) / "doctor_probe.csv"
        csv_path.write_text("y,x1\n0,1\n1,2\n", encoding="utf-8")
        layout = open_project(csv_path)
        manifest = new_manifest(csv_path)
        write_manifest(layout.project_dir, manifest)
        reloaded = read_manifest(layout.project_dir)
        run_id, run_dir = allocate_run_dir(layout.runs_dir)
        problems: List[str] = []
        if reloaded.get("schema") != new_manifest(csv_path).get("schema"):
            problems.append("manifest schema 往返不一致")
        if not run_dir.is_dir():
            problems.append("run 目录未创建")
        if not RUN_ID_PATTERN.match(run_id):
            problems.append(f"run_id 不符合 {RUN_ID_PATTERN.pattern}")
        if problems:
            return Check(
                name="artifact_dir",
                title="产物目录可写性",
                status="fail",
                detail=f"临时目录 {tmp_root} 上链路异常：" + "；".join(problems),
                hint="这是平台层缺陷，不是环境问题；请连同上面细节反馈",
            )
        return Check(
            name="artifact_dir",
            title="产物目录可写性",
            status="ok",
            detail=(
                f"临时目录可建 mdl-doctor_probe/，manifest 写—读往返一致，"
                f"run 目录名 {run_id} 匹配 {RUN_ID_PATTERN.pattern}"
            ),
            data={"temp_dir": tmp_root, "run_id": run_id},
        )
    except Exception as exc:  # noqa: BLE001
        return Check(
            name="artifact_dir",
            title="产物目录可写性",
            status="fail",
            detail=f"临时目录 {tmp_root} 写入/建目录失败：{type(exc).__name__}: {exc}",
            hint="检查系统临时目录权限（TEMP/TMP 环境变量指向的目录必须可写）",
            fix_commands=[
                "python -c \"import tempfile,os;print(tempfile.gettempdir())\"",
                "# 确认该目录存在且当前用户可写；或设置 TMPDIR 指向可写目录后重试",
            ],
        )
    finally:
        if tmp_root:
            shutil.rmtree(tmp_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #

_CHECKS = (
    check_python,
    check_dependencies,
    check_mcp_sdk,
    check_matplotlib,
    check_chinese_font,
    check_artifact_dir,
)


def run_all_checks() -> List[Check]:
    """跑全部六项；任何一项抛异常都降级成该项 FAIL（自检自身不能崩）。"""
    results: List[Check] = []
    for name, func in zip(CHECK_NAMES, _CHECKS):
        try:
            result = func()
        except Exception as exc:  # noqa: BLE001
            result = Check(
                name=name,
                title=name,
                status="fail",
                detail=f"自检项自身异常：{type(exc).__name__}: {exc}",
                hint="这是 doctor 的缺陷，请反馈这段输出",
            )
        # 保证 name 与契约一致（防止函数里写错）
        if result.name != name:
            result = Check(
                name=name,
                title=result.title,
                status=result.status,
                detail=result.detail,
                hint=result.hint,
                fix_commands=result.fix_commands,
                data=result.data,
            )
        results.append(result)
    return results


def summarize(results: Sequence[Check]) -> int:
    """返回退出码：有 FAIL ⇒ 1，否则 0。"""
    return 1 if any(r.status == "fail" for r in results) else 0


def _print_human(results: Sequence[Check]) -> None:
    print(f"risklab-mdl-v2 doctor  v{__version__}")
    print(
        f"{_platform.system()} {_platform.release()} | "
        f"Python {_platform.python_version()} | {sys.executable}"
    )
    print("-" * 78)
    for index, result in enumerate(results, start=1):
        label = _STATUS_LABEL.get(result.status, result.status.upper())
        print(f"[{label}] {index}. {result.title}")
        print(f"        {result.detail}")
        if result.hint:
            print(f"        说明：{result.hint}")
    print("-" * 78)
    fails = [r for r in results if r.status == "fail"]
    warns = [r for r in results if r.status == "warn"]
    if fails:
        print(f"结论：{len(fails)} 项失败 / {len(results)} 项，退出码 1")
    else:
        print(f"结论：全部通过（{len(results)}/{len(results)}，其中 {len(warns)} 项告警），退出码 0")
    commands: List[str] = []
    for result in results:
        for command in result.fix_commands:
            if command not in commands:
                commands.append(command)
    if commands:
        print("\n可复制的修复命令：")
        for command in commands:
            print(f"  {command}")


def _use_utf8_stdout() -> None:
    """把 stdout/stderr 固定为 UTF-8。

    ★ 为什么需要
    Windows 中文环境下「重定向到文件」或「被 agent 当管道读取」时，Python 默认按
    本地编码（cp936）输出，读取方若按 UTF-8 解码就是满屏乱码 —— 调用本 doctor 的
    往往正是 agent。控制台直连时 Python 走 WriteConsoleW，编码设置不影响显示，
    因此固定 UTF-8 对两边都安全。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # 已关闭或不支持重配置的流，忽略
            pass


def main(argv: Optional[List[str]] = None) -> int:
    _use_utf8_stdout()
    parser = argparse.ArgumentParser(description="risklab-mdl-v2 环境自检（六项探针）")
    parser.add_argument("--json", action="store_true", help="输出结构化 JSON（供部署脚本消费）")
    args = parser.parse_args(argv)

    results = run_all_checks()
    code = summarize(results)
    if args.json:
        payload = {
            "ok": code == 0,
            "version": __version__,
            "exit_code": code,
            "platform": {
                "system": _platform.system(),
                "release": _platform.release(),
                "python": _platform.python_version(),
                "executable": sys.executable,
            },
            "checks": [asdict(r) for r in results],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_human(results)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
