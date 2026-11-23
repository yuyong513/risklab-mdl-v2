"""doctor 单测：六项探针结构、退出码语义、缺依赖场景必须非零。

★ 为什么专门测「缺依赖 ⇒ 非零」
-----------------------------
「失败不可见」是旧项目的核心教训之一：环境缺依赖时如果自检仍然退出 0，
部署脚本会以为一切正常，直到真正建模时才炸。这里用 mock 伪造缺失依赖，
断言 doctor 报 FAIL、退出码 1、并给出可复制的修复命令。
"""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest import mock

from risklab_mdl import doctor as doctor_mod


class DoctorStructureTests(unittest.TestCase):
    def test_six_checks_in_frozen_order(self) -> None:
        results = doctor_mod.run_all_checks()
        self.assertEqual(len(results), 6)
        self.assertEqual([r.name for r in results], list(doctor_mod.CHECK_NAMES))
        self.assertEqual(
            list(doctor_mod.CHECK_NAMES),
            ["python", "dependencies", "mcp_sdk", "matplotlib", "chinese_font", "artifact_dir"],
        )
        for result in results:
            self.assertIn(result.status, {"ok", "warn", "fail"})
            self.assertTrue(result.title, f"{result.name} 缺标题")
            self.assertTrue(result.detail, f"{result.name} 缺详情")

    def test_self_exception_is_downgraded_to_fail_instead_of_crashing(self) -> None:
        def boom():
            raise RuntimeError("自检项炸了")

        checks = list(doctor_mod._CHECKS)
        checks[0] = boom
        with mock.patch.object(doctor_mod, "_CHECKS", tuple(checks)):
            results = doctor_mod.run_all_checks()
        self.assertEqual(results[0].status, "fail")
        self.assertIn("自检项炸了", results[0].detail)
        self.assertEqual(len(results), 6, "单项崩溃不得影响其余检查项")

    def test_summarize_exit_codes(self) -> None:
        ok = doctor_mod.Check("python", "Python 版本", "ok", "fine")
        warn = doctor_mod.Check("chinese_font", "中文字体", "warn", "缺字体")
        fail = doctor_mod.Check("dependencies", "依赖", "fail", "缺 pandas")
        self.assertEqual(doctor_mod.summarize([ok, warn]), 0, "WARN 不应让退出码非零")
        self.assertEqual(doctor_mod.summarize([ok, warn, fail]), 1)


class MissingDependencyTests(unittest.TestCase):
    def _missing_report(self) -> dict:
        return {
            "mcp": {"import": "mcp", "version": "2.3.0", "error": None},
            "pandas": {"import": "pandas", "version": None, "error": "ModuleNotFoundError: No module named 'pandas'"},
            "matplotlib": {"import": "matplotlib", "version": None, "error": "ModuleNotFoundError"},
        }

    def test_check_dependencies_reports_fail_and_fix_commands(self) -> None:
        with mock.patch("risklab_mdl.server.dependency_report", return_value=self._missing_report()):
            check = doctor_mod.check_dependencies()
        self.assertEqual(check.status, "fail")
        self.assertIn("pandas", check.detail)
        self.assertIn("matplotlib", check.detail)
        joined = " ".join(check.fix_commands)
        self.assertIn("pip install", joined)
        self.assertIn("pandas", joined)

    def test_doctor_exit_code_is_nonzero_when_dependency_missing(self) -> None:
        fake = doctor_mod.Check(
            "dependencies",
            "依赖版本矩阵",
            "fail",
            "缺 mcp/pandas",
            hint="装依赖",
            fix_commands=['python -m pip install -e "C:/proj"'],
        )
        checks = list(doctor_mod._CHECKS)
        checks[doctor_mod.CHECK_NAMES.index("dependencies")] = lambda: fake
        buffer = io.StringIO()
        with mock.patch.object(doctor_mod, "_CHECKS", tuple(checks)), redirect_stdout(buffer):
            code = doctor_mod.main([])
        self.assertEqual(code, 1, "缺依赖时必须退出码 1")
        output = buffer.getvalue()
        self.assertIn("[FAIL]", output)
        self.assertIn("pip install", output, "必须打印可复制的修复命令")
        self.assertIn("退出码 1", output)


class DoctorDecisionTests(unittest.TestCase):
    """T9 修正后的判定口径：版本不一致 = WARN；force 后仍非 Agg = FAIL。"""

    def test_pyproject_version_mismatch_is_warn_and_keeps_exit_code_zero(self) -> None:
        with mock.patch.object(doctor_mod, "_pyproject_version", return_value="9.9.9"):
            check = doctor_mod.check_python()
        self.assertEqual(check.status, "warn", check.detail)
        self.assertIn("9.9.9", check.detail)
        self.assertIn(doctor_mod.__version__, check.detail)
        self.assertIn("WARN", check.hint, "输出必须说明为什么是 WARN 而不是 FAIL")
        self.assertEqual(doctor_mod.summarize([check]), 0, "WARN 不得让退出码非零")

    def test_python_too_old_still_fails(self) -> None:
        fake_platform = mock.Mock()
        fake_platform.python_version.return_value = "3.10.0"
        fake_platform.python_implementation.return_value = "CPython"
        fake_sys = mock.Mock(version_info=(3, 10, 0), executable="python")
        with mock.patch.object(doctor_mod, "_pyproject_version", return_value=None), mock.patch.object(
            doctor_mod, "_platform", fake_platform
        ), mock.patch.object(doctor_mod, "sys", fake_sys):
            check = doctor_mod.check_python()
        self.assertEqual(check.status, "fail")
        self.assertIn("3.11", check.detail)

    def test_matplotlib_backend_guard_fails_when_force_does_not_take_effect(self) -> None:
        # 模拟「force=True 后仍拿不到 Agg」的异常环境，覆盖原恒不可达分支
        with mock.patch("matplotlib.get_backend", return_value="TkAgg"):
            check = doctor_mod.check_matplotlib()
        self.assertEqual(check.status, "fail", check.detail)
        self.assertIn("TkAgg", check.detail)
        self.assertEqual(doctor_mod.summarize([check]), 1)


class RealEnvironmentTests(unittest.TestCase):
    """真实环境断言（本机已装齐依赖；缺依赖的机器上这些会明确失败而不是静默跳过）。"""

    def test_current_env_passes_doctor(self) -> None:
        results = doctor_mod.run_all_checks()
        fails = [f"{r.name}: {r.detail}" for r in results if r.status == "fail"]
        self.assertEqual(fails, [], f"doctor 存在失败项：{fails}")
        self.assertEqual(doctor_mod.summarize(results), 0)

    def test_matplotlib_check_ok(self) -> None:
        check = doctor_mod.check_matplotlib()
        self.assertEqual(check.status, "ok", check.detail)
        self.assertEqual(check.data.get("backend", "").lower(), "agg")

    def test_chinese_font_never_fails(self) -> None:
        # 缺字体只 WARN（R5/Q10：不阻断建模）
        check = doctor_mod.check_chinese_font()
        self.assertIn(check.status, {"ok", "warn"}, check.detail)

    def test_artifact_dir_check_ok(self) -> None:
        check = doctor_mod.check_artifact_dir()
        self.assertEqual(check.status, "ok", check.detail)

    def test_mcp_sdk_check_ok(self) -> None:
        check = doctor_mod.check_mcp_sdk()
        self.assertEqual(check.status, "ok", check.detail)
        self.assertIsNotNone(check.data.get("resolved_api"))

    def test_json_mode_is_machine_readable(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = doctor_mod.main(["--json"])
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["exit_code"], code)
        self.assertEqual(len(payload["checks"]), 6)
        self.assertEqual([c["name"] for c in payload["checks"]], list(doctor_mod.CHECK_NAMES))


if __name__ == "__main__":
    unittest.main()
