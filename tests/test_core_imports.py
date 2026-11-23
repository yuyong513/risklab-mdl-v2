"""core 层导入图守护：导入不拉起重依赖 / 无 POSIX-only 依赖 / 无旧包名。

★ 为什么要单独一个测试文件
--------------------------
「``import risklab_mdl.core.binning`` 顺手把 lightgbm 拉起来」这类问题
**不会在功能测试里暴露** —— 装了库的机器上一切正常，只在**没装库**的环境
（或 MCP 服务端启动路径）上炸，且错误埋在 import traceback 里，
工具层拿不到 error 载荷（违反「失败必须可见」）。

⇒ 三道守护：

1. **静态**：AST 扫描 —— core 任何模块的**模块级** import 不得出现
   lightgbm / xgboost / matplotlib（函数内惰性 import 才是允许的写法）；
2. **动态**：干净子进程里 ``import risklab_mdl.core.*`` 后检查 ``sys.modules``
   —— 上一条被绕过时（如 ``from x import y`` 写进函数里被误移到模块级）仍能抓住；
3. **平台**：src 里不得有 ``import fcntl``（Windows 上导入即崩），
   也不得出现旧包名 ``risklab`` 的独立 token（与 T1 的
   ``tests/test_platform.py::test_src_has_no_old_package_reference`` 互为独立守护）。

★ 为什么用子进程做动态检查
  本进程里别的测试（如 ``test_core_refine``）会合法地 import lightgbm，
  ``sys.modules`` 已被污染 ⇒ 在本进程断言「未导入」既会假红也会假绿。
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
CORE_ROOT = SRC_ROOT / "risklab_mdl" / "core"

HEAVY_DEPS = ("lightgbm", "xgboost", "matplotlib")

#: core 层应有（且必须可导入）的模块
CORE_MODULES = (
    "errors",
    "deps",
    "binning",
    "binning_dt",
    "psi",
    "correlation",
    "encode",
    "metrics",
    "expr",
    "split",
    "prescreen",
    "refine",
    "tuning",
    "train",
    "package",
    "model_doc",
    "transforms",
    "utils",
)

OLD_PACKAGE_RE = re.compile(r"(?<![A-Za-z0-9_])risklab(?![A-Za-z0-9_-])")


def _core_python_files() -> list[Path]:
    return sorted(CORE_ROOT.glob("*.py"))


class StaticImportGraphTests(unittest.TestCase):
    """静态：core 模块级 import 不得含重依赖。"""

    def test_module_level_imports_exclude_heavy_deps(self) -> None:
        offenders: list[str] = []
        for path in _core_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:  # 只扫模块级（函数体内的惰性 import 合法）
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        names = [node.module.split(".")[0]]
                for name in names:
                    if name in HEAVY_DEPS:
                        offenders.append(
                            f"{path.name}:{node.lineno}: import {name}"
                        )
        self.assertEqual(
            offenders,
            [],
            f"core 模块级 import 拉起重依赖（应改为函数内惰性 import）：{offenders}",
        )

    def test_no_lazy_import_violates_stdlib_only(self) -> None:
        """``deps.py`` 顶层不得 import 任何第三方包（它是惰性入口本身）。"""
        tree = ast.parse((CORE_ROOT / "deps.py").read_text(encoding="utf-8"))
        third_party: list[str] = []
        stdlib = set(sys.stdlib_module_names)
        for node in tree.body:
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name not in stdlib and name != "risklab_mdl":
                    third_party.append(f"{name}@{node.lineno}")
        self.assertEqual(
            third_party, [], f"deps.py 顶层出现第三方依赖：{third_party}"
        )


class DynamicImportGraphTests(unittest.TestCase):
    """动态：干净子进程里导入 core 不拉起重依赖。"""

    def _run_import_probe(self, imports: str) -> tuple[int, str, str]:
        script = (
            "import sys\n"
            f"{imports}\n"
            "heavy = [m for m in ('lightgbm','xgboost','matplotlib') if m in sys.modules]\n"
            "print('|'.join(heavy))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

    def test_import_all_core_modules_does_not_load_heavy_deps(self) -> None:
        imports = "\n".join(
            f"import risklab_mdl.core.{name}" for name in CORE_MODULES
        )
        code, out, err = self._run_import_probe(imports)
        self.assertEqual(code, 0, f"core 模块导入失败：{err}")
        self.assertEqual(
            out,
            "",
            f"导入 core 后 sys.modules 出现重依赖：{out.split('|')}",
        )

    def test_import_core_package_only(self) -> None:
        code, out, err = self._run_import_probe("import risklab_mdl.core")
        self.assertEqual(code, 0, err)
        self.assertEqual(out, "", f"import risklab_mdl.core 拉起了：{out}")

    def test_lazy_require_works_and_reports_model_error(self) -> None:
        """重依赖**实际存在**时，惰性入口能拿到模块（不在这里断言缺失分支）。"""
        code, out, err = self._run_import_probe(
            "import risklab_mdl.core.deps as d\n"
            "mod = d.require_lightgbm()\n"
            "print('loaded', mod.__name__)"
        )
        self.assertEqual(code, 0, err)
        self.assertIn("loaded lightgbm", out)


class PlatformGuardTests(unittest.TestCase):
    """src 全局：无 fcntl、无旧包名 token。"""

    def _src_files(self) -> list[Path]:
        return sorted(SRC_ROOT.rglob("*.py"))

    def test_no_fcntl_import(self) -> None:
        offenders: list[str] = []
        for path in self._src_files():
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                stripped = line.strip()
                if stripped.startswith(("import fcntl", "from fcntl")):
                    offenders.append(f"{path.relative_to(SRC_ROOT)}:{lineno}")
        self.assertEqual(offenders, [], f"发现 POSIX-only 的 fcntl 导入：{offenders}")

    def test_no_old_package_token_in_src(self) -> None:
        """旧包名 ``risklab`` 的独立 token 不得出现（``risklab_mdl`` / ``risklab-mdl`` 除外）。"""
        offenders: list[str] = []
        for path in self._src_files():
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if OLD_PACKAGE_RE.search(line):
                    offenders.append(
                        f"{path.relative_to(SRC_ROOT)}:{lineno}: {line.strip()[:100]}"
                    )
        self.assertEqual(offenders, [], f"src 中出现旧包名 risklab：{offenders}")

    def test_core_files_exist(self) -> None:
        for name in CORE_MODULES:
            self.assertTrue(
                (CORE_ROOT / f"{name}.py").is_file(), f"缺少 core 模块 {name}.py"
            )


class CoreExportTests(unittest.TestCase):
    """``core/__init__.py`` 的导出必须覆盖全部模块，且名字可导入。"""

    def test_package_exports_all_modules(self) -> None:
        import importlib

        import risklab_mdl.core as core

        missing = [name for name in CORE_MODULES if not hasattr(core, name)]
        self.assertEqual(missing, [], f"core.__init__ 未导出：{missing}")
        for name in CORE_MODULES:
            mod = getattr(core, name)
            self.assertEqual(
                mod.__name__, f"risklab_mdl.core.{name}", f"{name} 导出的是别的模块"
            )
        self.assertTrue(importlib.import_module("risklab_mdl.core").__all__)

    def test_public_api_present(self) -> None:
        """关键 API 形态冻结（后续 T3/T4/T5 依赖这些名字）。"""
        from risklab_mdl.core import binning, binning_dt, metrics, split, transforms

        for name in ("BinSpec", "fit_bins", "transform_bins", "bin_table_processed"):
            self.assertTrue(hasattr(binning, name), f"binning 缺少 {name}")
        for name in ("DTBinningSpec", "DTBinResult", "fit_bins_dt"):
            self.assertTrue(hasattr(binning_dt, name), f"binning_dt 缺少 {name}")
        for name in ("SplitSpec", "SplitResult", "split_frame"):
            self.assertTrue(hasattr(split, name), f"split 缺少 {name}")
        for name in ("calc_multiclass_metrics", "MultiClassMetrics"):
            self.assertTrue(hasattr(metrics, name), f"metrics 缺少 {name}")
        for name in ("apply_encode", "apply_fill", "apply_outlier", "profile_features"):
            self.assertTrue(hasattr(transforms, name), f"transforms 缺少 {name}")


if __name__ == "__main__":
    unittest.main()
