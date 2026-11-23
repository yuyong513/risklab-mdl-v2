"""校验 ``mdl-*`` 产物树是否满足冻结契约（发布门禁用）。

用法::

    python scripts/check_layout.py <项目目录>
    python scripts/check_layout.py ..\\mdl-sample_2000 --json

校验项
------
1. 目录名形如 ``mdl-<stem>``，``manifest.json`` 存在；
2. manifest schema = ``risklab-mdl.manifest/1``，必需字段齐全（再做一次
   ``validate_manifest`` 全量校验，与运行期完全同一套规则）；
3. 6 个阶段的 ``done`` 全为 true；
4. ``runs/`` 下目录名匹配 ``^\\d{8}T\\d{6}$``，且每个 run 有
   ``config.json / params.json / metrics.json / model.pkl / importance.csv``
   与非空的 ``evaluation/``；
5. 8 个固定子目录与各阶段必需文件齐全（data_analysis / feature_analysis /
   feature_selection / preprocess / split / runs / final / reports）；
6. ``reports/结果汇总.xlsx`` 能被 openpyxl 打开，且没有空 sheet；
7. ``final/confirm.json`` 存在、可解析，且 ``run_id`` 与 manifest 的
   ``confirmed_run`` 一致。

**一次列全所有问题**（不第一处就退出），失败 ``exit 1``；全部通过 ``exit 0``。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

#: 项目根（scripts/check_layout.py → scripts → 项目根）。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 源码树（未安装时也能直接运行本脚本；安装后这条不起作用）。
_SRC = PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from risklab_mdl.artifacts.layout import (  # noqa: E402
    CHART_SUBDIRS,
    MODEL_DIR_PREFIX,
    PROJECT_SUBDIRS,
)
from risklab_mdl.artifacts.manifest import (  # noqa: E402
    MANIFEST_SCHEMA,
    STAGES,
    ManifestError,
    validate_manifest,
)
from risklab_mdl.artifacts.store import RUN_ID_PATTERN  # noqa: E402

#: 各阶段必须存在的产物（相对项目目录，POSIX 路径）。
REQUIRED_FILES: Dict[str, tuple[str, ...]] = {
    "data_analysis": (
        "data_analysis/profile.json",
        "data_analysis/describe.csv",
        "data_analysis/target_distribution.csv",
        "data_analysis/index.html",
    ),
    "feature_analysis": (
        "feature_analysis/iv_table.csv",
        "feature_analysis/woe_detail.csv",
        "feature_analysis/index.html",
    ),
    "feature_selection": (
        "feature_selection/prescreen.json",
        "feature_selection/refine.json",
        "feature_selection/selected_features.txt",
    ),
    "preprocess": (
        "preprocess/check.json",
        "preprocess/summary.json",
        # 预处理数据二选一（pyarrow 缺失时落 CSV）
        "preprocess/processed.parquet|preprocess/processed.csv",
    ),
    "split": ("split/split.json",),
    "final": ("final/confirm.json", "final/model.pkl"),
    "reports": ("reports/结果汇总.xlsx", "reports/过程报告.md"),
}

#: 每个 run 目录必须存在的文件。
RUN_REQUIRED_FILES: tuple[str, ...] = (
    "config.json",
    "params.json",
    "metrics.json",
    "model.pkl",
    "importance.csv",
)

#: run 目录里必须存在的子目录。
RUN_REQUIRED_DIRS: tuple[str, ...] = ("evaluation",)

#: 固定子目录里的图表目录（存在即可，不强制有图；缺图由工具 warnings 说明）。
OPTIONAL_CHART_DIRS: tuple[str, ...] = tuple(f"{name}/charts" for name in CHART_SUBDIRS)

#: Excel 报告文件名。
XLSX_NAME = "结果汇总.xlsx"


class LayoutChecker:
    """收集全部问题，最后统一输出（不在第一处就退出）。"""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self.problems: List[str] = []
        self.notes: List[str] = []
        self.manifest: Optional[Dict[str, Any]] = None
        self.run_ids: List[str] = []
        self.confirmed_run: Optional[str] = None

    # ---------- 基础工具 ----------
    def problem(self, message: str) -> None:
        self.problems.append(message)

    def expect_dir(self, rel: str, *, label: str = "") -> bool:
        path = self.project_dir / rel
        if path.is_dir():
            return True
        self.problem(f"缺少目录：{rel}{('（' + label + '）') if label else ''}")
        return False

    def expect_file(self, rel: str) -> bool:
        path = self.project_dir / rel
        if path.is_file():
            return True
        self.problem(f"缺少文件：{rel}")
        return False

    # ---------- ① 目录与 manifest ----------
    def check_root(self) -> None:
        if not self.project_dir.is_dir():
            self.problem(f"项目目录不存在：{self.project_dir}")
            return
        if not self.project_dir.name.startswith(MODEL_DIR_PREFIX):
            self.problem(
                f"目录名必须是 {MODEL_DIR_PREFIX}<CSV 文件名>，实际是：{self.project_dir.name}"
            )
        manifest_path = self.project_dir / "manifest.json"
        if not manifest_path.is_file():
            self.problem("缺少文件：manifest.json")
            return
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.problem(f"manifest.json 无法解析：{type(exc).__name__}: {exc}")
            return

        if not isinstance(raw, dict):
            self.problem("manifest.json 顶层必须是 JSON 对象")
            return
        self.manifest = raw

        # ① schema / 必需字段（一次列全）
        if raw.get("schema") != MANIFEST_SCHEMA:
            self.problem(
                f"manifest.schema 必须是 {MANIFEST_SCHEMA!r}，实际是 {raw.get('schema')!r}"
            )
        project = raw.get("project")
        if not isinstance(project, dict):
            self.problem("manifest 缺少 project 对象")
        else:
            for key in ("csv_path", "stem", "created_at", "updated_at"):
                value = project.get(key)
                if not isinstance(value, str) or not value.strip():
                    self.problem(f"manifest.project.{key} 必须是非空字符串")
            target = project.get("target")
            if target is not None:
                if not isinstance(target, dict):
                    self.problem("manifest.project.target 必须是对象或 null")
                else:
                    if not isinstance(target.get("column"), str) or not target["column"].strip():
                        self.problem("manifest.project.target.column 必须是非空字符串")
                    if target.get("task") not in ("binary", "multiclass"):
                        self.problem("manifest.project.target.task 必须是 binary 或 multiclass")

        # 再走运行期同一套校验（规则不在这里重写一遍）
        try:
            validate_manifest(raw)
        except ManifestError as exc:
            self.problem(f"manifest 全量校验失败：{exc.message}")

        # ② 阶段 done
        stages = raw.get("stages")
        if not isinstance(stages, dict):
            self.problem("manifest 缺少 stages 对象")
        else:
            unknown = sorted(set(stages) - set(STAGES))
            if unknown:
                self.problem(f"manifest.stages 含未知阶段：{unknown}")
            for stage in STAGES:
                entry = stages.get(stage)
                if not isinstance(entry, dict):
                    self.problem(f"manifest.stages.{stage} 缺失或不是对象")
                elif entry.get("done") is not True:
                    self.problem(f"阶段未完成：{stage}.done={entry.get('done')!r}（期望 true）")

    # ---------- ② 产物目录 ----------
    def check_directories(self) -> None:
        if not self.project_dir.is_dir():
            return
        for name in PROJECT_SUBDIRS:
            if not self.expect_dir(name):
                continue
            directory = self.project_dir / name
            if name != "runs" and not any(directory.rglob("*")):
                self.problem(f"产物目录为空：{name}/")
        for rel in OPTIONAL_CHART_DIRS:
            if not (self.project_dir / rel).is_dir():
                self.notes.append(f"未找到图表目录：{rel}（工具可能因降级跳过了图表）")

    # ---------- ③ 必需文件 ----------
    def check_required_files(self) -> None:
        if not self.project_dir.is_dir():
            return
        for files in REQUIRED_FILES.values():
            for rel in files:
                if "|" in rel:
                    if not any((self.project_dir / candidate).is_file() for candidate in rel.split("|")):
                        self.problem(f"缺少文件：{rel.replace('|', ' 或 ')}")
                    continue
                self.expect_file(rel)

    # ---------- ④ runs ----------
    def check_runs(self) -> None:
        runs_dir = self.project_dir / "runs"
        if not runs_dir.is_dir():
            return
        directories = sorted(p for p in runs_dir.iterdir() if p.is_dir())
        if not directories:
            self.problem("runs/ 下没有任何 run 目录（模型尚未训练）")
        for directory in directories:
            name = directory.name
            if not RUN_ID_PATTERN.match(name):
                self.problem(
                    f"run 目录名不合法：runs/{name}（必须匹配 {RUN_ID_PATTERN.pattern}）"
                )
                continue
            self.run_ids.append(name)
            for filename in RUN_REQUIRED_FILES:
                if not (directory / filename).is_file():
                    self.problem(f"run {name} 缺少文件：runs/{name}/{filename}")
            for sub in RUN_REQUIRED_DIRS:
                subdir = directory / sub
                if not subdir.is_dir():
                    self.problem(f"run {name} 缺少目录：runs/{name}/{sub}/")
                elif not any(subdir.iterdir()):
                    self.problem(f"run {name} 目录为空：runs/{name}/{sub}/")

        # manifest.runs 与磁盘 run 目录必须一致（双向）
        manifest = self.manifest or {}
        declared = manifest.get("runs")
        if isinstance(declared, list):
            declared_ids = [str(entry.get("run_id")) for entry in declared if isinstance(entry, dict)]
            for run_id in sorted(set(declared_ids) - set(self.run_ids)):
                self.problem(f"manifest.runs 声明了 run {run_id}，但磁盘上缺少 runs/{run_id}/")
            for run_id in sorted(set(self.run_ids) - set(declared_ids)):
                self.problem(f"磁盘存在 runs/{run_id}/，但 manifest.runs 未登记")
        self.confirmed_run = manifest.get("confirmed_run") if isinstance(manifest, dict) else None

    # ---------- ⑤ 结果汇总.xlsx ----------
    def check_workbook(self) -> None:
        path = self.project_dir / "reports" / XLSX_NAME
        if not path.is_file():
            return
        try:
            from openpyxl import load_workbook
        except ImportError:
            self.problem("未安装 openpyxl，无法校验 结果汇总.xlsx（pip install openpyxl）")
            return
        try:
            workbook = load_workbook(path, read_only=True, data_only=True)
        except Exception as exc:  # noqa: BLE001 - 打不开就是门禁失败
            self.problem(
                f"结果汇总.xlsx 无法被 openpyxl 打开：{type(exc).__name__}: {exc}"
            )
            return
        try:
            names = list(workbook.sheetnames)
            if not names:
                self.problem("结果汇总.xlsx 没有任何 sheet")
            for name in names:
                sheet = workbook[name]
                filled = 0
                for row in sheet.iter_rows(values_only=True):
                    if any(value is not None and str(value).strip() != "" for value in row):
                        filled += 1
                        break
                if not filled:
                    self.problem(f"结果汇总.xlsx 的 sheet「{name}」为空")
        finally:
            workbook.close()

    # ---------- ⑥ final/confirm.json ----------
    def check_confirm(self) -> None:
        path = self.project_dir / "final" / "confirm.json"
        if not path.is_file():
            return
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.problem(f"final/confirm.json 无法解析：{type(exc).__name__}: {exc}")
            return
        if not isinstance(document, dict):
            self.problem("final/confirm.json 顶层必须是 JSON 对象")
            return
        run_id = document.get("run_id")
        if not isinstance(run_id, str) or not RUN_ID_PATTERN.match(run_id):
            self.problem(
                f"final/confirm.json 的 run_id 不合法：{run_id!r}"
                f"（必须匹配 {RUN_ID_PATTERN.pattern}）"
            )
        elif run_id not in self.run_ids:
            self.problem(f"final/confirm.json 的 run_id={run_id} 在 runs/ 下不存在")
        if self.confirmed_run and run_id != self.confirmed_run:
            self.problem(
                f"final/confirm.json 的 run_id={run_id} 与 manifest.confirmed_run="
                f"{self.confirmed_run} 不一致"
            )
        if not self.confirmed_run:
            self.problem(
                "final/confirm.json 存在，但 manifest.confirmed_run 为空（未登记最终模型）"
            )
        if self.confirmed_run and self.confirmed_run not in self.run_ids:
            self.problem(
                f"manifest.confirmed_run={self.confirmed_run} 在 runs/ 下不存在"
            )
        if not isinstance(document.get("metrics"), dict):
            self.problem("final/confirm.json 缺少 metrics 对象")
        if not isinstance(document.get("thresholds"), dict):
            self.problem("final/confirm.json 缺少 thresholds 对象")

    # ---------- 汇总 ----------
    def run(self) -> int:
        self.check_root()
        self.check_directories()
        self.check_required_files()
        self.check_runs()
        self.check_workbook()
        self.check_confirm()
        return 1 if self.problems else 0

    def report(self) -> None:
        print(f"[check_layout] 项目目录：{self.project_dir}")
        if self.problems:
            print(f"[check_layout] 契约校验失败：{len(self.problems)} 项")
            for problem in self.problems:
                print(f"  ✗ {problem}")
            if self.notes:
                print(f"[check_layout] 说明（{len(self.notes)} 条，不计入失败）：")
                for note in self.notes:
                    print(f"  - {note}")
            print("[check_layout] 退出码：1")
            return
        print("[check_layout] 契约校验通过：")
        print(f"  ✓ manifest schema={MANIFEST_SCHEMA}，{len(STAGES)} 个阶段全部 done")
        print(f"  ✓ {len(PROJECT_SUBDIRS)} 个固定子目录齐全，必需文件齐全")
        print(f"  ✓ runs/ 下 {len(self.run_ids)} 个 run 目录命名与产物齐全：{', '.join(self.run_ids)}")
        print(f"  ✓ reports/{XLSX_NAME} 可被 openpyxl 打开且每个 sheet 非空")
        print(f"  ✓ final/confirm.json 存在（confirmed_run={self.confirmed_run}）")
        if self.notes:
            for note in self.notes:
                print(f"  - {note}")
        print("[check_layout] 退出码：0")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="校验 mdl-* 产物树契约（失效项一次列全，exit 1）"
    )
    parser.add_argument("project_dir", help="待校验项目目录（形如 .../mdl-<CSV 文件名>）")
    parser.add_argument(
        "--json", action="store_true", help="以 JSON 输出校验结果（供 CI 消费）"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    project_dir = Path(args.project_dir).expanduser().resolve()
    checker = LayoutChecker(project_dir)
    code = checker.run()
    if args.json:
        payload = {
            "ok": code == 0,
            "project_dir": project_dir.as_posix(),
            "problems": checker.problems,
            "notes": checker.notes,
            "run_ids": checker.run_ids,
            "confirmed_run": checker.confirmed_run,
            "exit_code": code,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        checker.report()
    return code


if __name__ == "__main__":  # pragma: no cover - 手工执行入口
    raise SystemExit(main())
