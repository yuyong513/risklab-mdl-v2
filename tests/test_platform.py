"""platform 层单测：原子写、编码安全 JSON、路径校验、跨平台文件锁、静态红线。

运行：``python -m pytest tests/test_platform.py -q``（unittest 风格，亦可用 ``python -m unittest``）
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from pathlib import Path
from unittest import mock

from risklab_mdl.platform import fs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


class AtomicWriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="risklab-fs-")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_json_roundtrip_keeps_chinese_and_is_utf8_without_bom(self) -> None:
        target = self.tmp / "manifest.json"
        payload = {"名称": "目标变量", "nested": {"值": [1, 2, 3]}, "ok": True}
        fs.atomic_write_json(target, payload)

        self.assertEqual(fs.read_json(target), payload)
        raw = target.read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"), "不应写 BOM")
        self.assertIn("目标变量".encode("utf-8"), raw, "中文应原样写入（ensure_ascii=False）")
        self.assertTrue(raw.endswith(b"\n"), "末尾应有换行，便于 diff")

    def test_write_leaves_no_temp_files(self) -> None:
        target = self.tmp / "a.json"
        fs.atomic_write_json(target, {"a": 1})
        fs.atomic_write_json(target, {"a": 2})
        names = sorted(p.name for p in self.tmp.iterdir())
        self.assertEqual(names, ["a.json"], f"目录里应只剩目标文件，实际：{names}")
        self.assertEqual(fs.read_json(target), {"a": 2})

    def test_creates_missing_parent_dirs(self) -> None:
        target = self.tmp / "runs" / "deep" / "config.json"
        fs.atomic_write_json(target, {"run_id": "20260101T120000"})
        self.assertTrue(target.is_file())
        self.assertEqual(fs.read_json(target)["run_id"], "20260101T120000")

    def test_unwritable_target_raises_with_code(self) -> None:
        blocker = self.tmp / "blocker"
        blocker.write_text("i am a file", encoding="utf-8")
        with self.assertRaises(fs.PlatformError) as ctx:
            fs.atomic_write_json(blocker / "x.json", {"a": 1})
        self.assertEqual(ctx.exception.code, "path_not_writable")

    def test_json_serialize_error_is_explicit(self) -> None:
        with self.assertRaises(fs.PlatformError) as ctx:
            fs.atomic_write_json(self.tmp / "x.json", {"bad": object()})
        self.assertEqual(ctx.exception.code, "json_encode_error")


class ReadJsonTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="risklab-json-")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_missing_file_raises_path_not_readable(self) -> None:
        with self.assertRaises(fs.PathNotReadable) as ctx:
            fs.read_json(self.tmp / "nope.json")
        self.assertEqual(ctx.exception.code, "path_not_readable")

    def test_missing_file_with_default(self) -> None:
        value = fs.read_json(self.tmp / "nope.json", default={"x": 1}, has_default=True)
        self.assertEqual(value, {"x": 1})

    def test_corrupt_json_raises_json_decode_error(self) -> None:
        bad = self.tmp / "bad.json"
        bad.write_text('{"a": 1,', encoding="utf-8")
        with self.assertRaises(fs.PlatformError) as ctx:
            fs.read_json(bad)
        self.assertEqual(ctx.exception.code, "json_decode_error")
        self.assertEqual(ctx.exception.to_payload()["code"], "json_decode_error")


class PathTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="risklab-path-")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_normalize_path_makes_absolute_and_collapses_dotdot(self) -> None:
        result = fs.normalize_path(self.tmp / "a" / ".." / "b")
        self.assertTrue(result.is_absolute())
        self.assertEqual(result, (self.tmp / "b").resolve())

    def test_normalize_path_rejects_empty(self) -> None:
        with self.assertRaises(fs.PlatformError) as ctx:
            fs.normalize_path("")
        self.assertEqual(ctx.exception.code, "path_invalid")

    def test_is_writable_and_check_writable(self) -> None:
        self.assertTrue(fs.is_writable(self.tmp))
        self.assertEqual(fs.check_writable(self.tmp), self.tmp.resolve())

    def test_check_writable_maps_os_error_to_typed_error(self) -> None:
        """真实映射：写探针失败 ⇒ is_writable False ⇒ check_writable 抛 typed error。"""
        with mock.patch(
            "risklab_mdl.platform.fs.tempfile.mkstemp",
            side_effect=OSError("disk full"),
        ):
            self.assertFalse(fs.is_writable(self.tmp), "探针失败时 is_writable 必须返回 False")
            with self.assertRaises(fs.PathNotWritable) as ctx:
                fs.check_writable(self.tmp, label="目录")
        self.assertEqual(ctx.exception.code, "path_not_writable")
        self.assertIn("目录不可写", str(ctx.exception))
        self.assertTrue(ctx.exception.to_payload()["hint"], "typed error 必须带修复指引")

    def test_require_readable(self) -> None:
        existing = self.tmp / "data.csv"
        existing.write_text("y\n1\n", encoding="utf-8")
        self.assertEqual(fs.require_readable(existing), existing.resolve())
        with self.assertRaises(fs.PathNotReadable):
            fs.require_readable(self.tmp / "missing.csv")
        with self.assertRaises(fs.PathNotReadable):
            fs.require_readable(self.tmp)  # 目录不是文件


class EncodingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="risklab-enc-")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_detect_utf8(self) -> None:
        path = self.tmp / "u8.csv"
        path.write_bytes("目标,特征\n1,2\n".encode("utf-8"))
        self.assertEqual(fs.detect_encoding(path), "utf-8")
        self.assertIn("目标", fs.read_text_auto(path))

    def test_detect_utf8_bom(self) -> None:
        path = self.tmp / "bom.csv"
        path.write_bytes("目标,特征\n1,2\n".encode("utf-8-sig"))
        self.assertEqual(fs.detect_encoding(path), "utf-8-sig")

    def test_detect_gb18030(self) -> None:
        path = self.tmp / "gbk.csv"
        path.write_bytes("目标,特征\n1,2\n".encode("gb18030"))
        self.assertEqual(fs.detect_encoding(path), "gb18030")
        self.assertIn("目标", fs.read_text_auto(path))

    # ---------- I3：UTF-16 BOM 与解码失败的 typed error ----------

    def test_detect_utf16_le_bom_and_read(self) -> None:
        path = self.tmp / "u16le.csv"
        path.write_bytes("目标,特征\n1,2\n".encode("utf-16"))  # Windows/Excel 常见：FF FE BOM
        self.assertTrue(path.read_bytes().startswith(b"\xff\xfe"), "测试数据应带 UTF-16 LE BOM")
        self.assertEqual(fs.detect_encoding(path), "utf-16")
        self.assertIn("目标", fs.read_text_auto(path))

    def test_detect_utf16_be_bom_and_read(self) -> None:
        path = self.tmp / "u16be.csv"
        path.write_bytes("\ufeff目标,特征\n1,2\n".encode("utf-16-be"))  # FE FF BOM
        self.assertTrue(path.read_bytes().startswith(b"\xfe\xff"), "测试数据应带 UTF-16 BE BOM")
        self.assertEqual(fs.detect_encoding(path), "utf-16")
        self.assertIn("目标", fs.read_text_auto(path))

    def test_undecodable_file_raises_typed_error_not_raw_unicode_error(self) -> None:
        path = self.tmp / "binary.csv"
        path.write_bytes(b"\xff\n")  # utf-8 与 gb18030 都无法解码，且不是任何 BOM 前缀
        with self.assertRaises(fs.EncodingUndecodable) as ctx:
            fs.read_text_auto(path)
        self.assertEqual(ctx.exception.code, "encoding_undecodable")
        self.assertEqual(ctx.exception.to_payload()["code"], "encoding_undecodable")
        self.assertIn("UTF-8", ctx.exception.hint)

    def test_undecodable_json_maps_to_typed_error(self) -> None:
        bad = self.tmp / "bad.json"
        bad.write_bytes(b"\xff\n")
        with self.assertRaises(fs.PlatformError) as ctx:
            fs.read_json(bad)
        self.assertEqual(ctx.exception.code, "encoding_undecodable")


class FileLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="risklab-lock-")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_lock_creates_file_and_is_reusable_after_release(self) -> None:
        lock_path = self.tmp / ".manifest.lock"
        with fs.file_lock(lock_path) as acquired:
            self.assertEqual(acquired, lock_path.resolve())
            self.assertTrue(lock_path.exists())
        with fs.file_lock(lock_path, timeout=5):
            pass  # 释放后可再次获取，说明没留死锁

    # ---------- I4：同线程同路径可重入 ----------

    def test_nested_lock_does_not_timeout(self) -> None:
        lock_path = self.tmp / ".manifest.lock"
        with fs.file_lock(lock_path, timeout=1):
            with fs.file_lock(lock_path, timeout=0.5):
                with fs.file_lock(lock_path, timeout=0.5):
                    self.assertTrue(lock_path.exists())
        with fs.file_lock(lock_path, timeout=1):
            pass  # 嵌套全部退出后计数清零，可重新获取

    def test_nested_lock_keeps_outer_lock_until_outer_exit(self) -> None:
        lock_path = self.tmp / ".manifest.lock"
        outcomes = []

        def probe() -> None:
            try:
                with fs.file_lock(lock_path, timeout=0.4):
                    outcomes.append("acquired")
            except fs.LockTimeout:
                outcomes.append("timeout")

        with fs.file_lock(lock_path, timeout=1):
            with fs.file_lock(lock_path, timeout=0.4):
                pass  # 内层退出
            worker = threading.Thread(target=probe)
            worker.start()
            worker.join(timeout=15)
            self.assertEqual(outcomes, ["timeout"], "内层退出后外层仍持锁（跨线程应互斥）")
        # 外层退出后，其他线程可以拿到锁
        worker = threading.Thread(target=probe)
        worker.start()
        worker.join(timeout=15)
        self.assertEqual(outcomes, ["timeout", "acquired"])

    def test_nested_lock_releases_on_exception(self) -> None:
        lock_path = self.tmp / ".manifest.lock"
        with self.assertRaises(RuntimeError):
            with fs.file_lock(lock_path, timeout=1):
                with fs.file_lock(lock_path, timeout=0.5):
                    raise RuntimeError("boom")
        with fs.file_lock(lock_path, timeout=1):
            pass  # 异常路径也必须把计数清干净

    def test_lock_timeout_when_another_process_holds_it(self) -> None:
        lock_path = self.tmp / ".manifest.lock"
        script = textwrap.dedent(
            f"""
            import time
            from risklab_mdl.platform.fs import file_lock

            with file_lock(r"{lock_path}", timeout=10):
                print("locked", flush=True)
                time.sleep(5)
            """
        )
        holder = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert holder.stdout is not None
            first = holder.stdout.readline().strip()
            self.assertEqual(first, "locked", "子进程未能先拿到锁")
            with self.assertRaises(fs.LockTimeout) as ctx:
                with fs.file_lock(lock_path, timeout=0.5):
                    self.fail("不应拿到被其他进程持有的锁")
            self.assertEqual(ctx.exception.code, "lock_timeout")
        finally:
            holder.kill()
            holder.wait(timeout=30)


class StaticGuardTests(unittest.TestCase):
    """静态红线：无 POSIX-only 依赖、无旧包名引用（验收 ⑥）。"""

    def _python_files(self) -> list:
        return sorted(SRC_ROOT.rglob("*.py"))

    def test_src_has_no_fcntl_import(self) -> None:
        pattern = re.compile(r"(?m)^\s*(?:import\s+fcntl|from\s+fcntl\s+import)")
        offenders = [
            str(path.relative_to(SRC_ROOT))
            for path in self._python_files()
            if pattern.search(path.read_text(encoding="utf-8"))
        ]
        self.assertEqual(offenders, [], f"发现 fcntl 导入（Windows 上会导入即崩）：{offenders}")

    def test_src_has_no_old_package_reference(self) -> None:
        # 旧包名是 risklab（本项目是 risklab_mdl，脚本名是 risklab-mdl）
        pattern = re.compile(r"(?<![A-Za-z0-9_])risklab(?![A-Za-z0-9_-])")
        offenders = []
        for path in self._python_files():
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(SRC_ROOT)}:{lineno}: {line.strip()}")
        self.assertEqual(offenders, [], f"发现旧包名 risklab 引用：{offenders}")

    def test_source_tree_has_required_modules(self) -> None:
        required = [
            "risklab_mdl/__init__.py",
            "risklab_mdl/__main__.py",
            "risklab_mdl/server.py",
            "risklab_mdl/doctor.py",
            "risklab_mdl/sdk.py",
            "risklab_mdl/platform/fs.py",
            "risklab_mdl/artifacts/layout.py",
            "risklab_mdl/artifacts/manifest.py",
            "risklab_mdl/artifacts/store.py",
        ]
        for rel in required:
            self.assertTrue((SRC_ROOT / rel).is_file(), f"缺少模块：src/{rel}")


if __name__ == "__main__":
    unittest.main()
