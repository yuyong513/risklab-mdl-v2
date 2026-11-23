"""artifacts 层单测：``mdl-<stem>/`` 布局、manifest 读写与校验、run 目录命名。"""

from __future__ import annotations

import json
import re
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from risklab_mdl.artifacts import layout as layout_mod
from risklab_mdl.artifacts import manifest as manifest_mod
from risklab_mdl.artifacts import store as store_mod
from risklab_mdl.platform import fs


def _write_csv(path: Path, rows: int = 3) -> Path:
    lines = ["y,x1,x2"]
    for i in range(rows):
        lines.append(f"{i % 2},{i},{i * 2}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class ProjectDirTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="risklab-layout-")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_project_dir_is_sibling_of_csv(self) -> None:
        csv_path = self.tmp / "sample_2000.csv"
        self.assertEqual(layout_mod.project_dir_for(csv_path), (self.tmp / "mdl-sample_2000").resolve())

    def test_project_dir_keeps_inner_dots_of_stem(self) -> None:
        csv_path = self.tmp / "my.data.v2.csv"
        self.assertEqual(layout_mod.project_dir_for(csv_path), (self.tmp / "mdl-my.data.v2").resolve())

    def test_open_project_creates_all_contracted_dirs(self) -> None:
        csv_path = _write_csv(self.tmp / "x.csv")
        layout = layout_mod.open_project(csv_path)
        self.assertEqual(layout.project_dir, (self.tmp / "mdl-x").resolve())
        for name in layout_mod.PROJECT_SUBDIRS:
            self.assertTrue((layout.project_dir / name).is_dir(), f"缺少子目录 {name}/")
        for name in layout_mod.CHART_SUBDIRS:
            self.assertTrue((layout.project_dir / name / "charts").is_dir(), f"缺少 {name}/charts/")
        self.assertEqual(layout.csv_path, csv_path.resolve())

    def test_open_project_reuses_existing_dir(self) -> None:
        csv_path = _write_csv(self.tmp / "x.csv")
        first = layout_mod.open_project(csv_path)
        marker = first.runs_dir / "20260101T000000"
        marker.mkdir()
        second = layout_mod.open_project(csv_path)
        self.assertEqual(first.project_dir, second.project_dir)
        self.assertTrue(marker.is_dir(), "复用不得清空历史 run")

    def test_missing_csv_raises_path_not_readable(self) -> None:
        with self.assertRaises(fs.PathNotReadable) as ctx:
            layout_mod.open_project(self.tmp / "nope.csv")
        self.assertEqual(ctx.exception.code, "path_not_readable")

    def test_existing_file_at_project_dir_is_conflict(self) -> None:
        csv_path = _write_csv(self.tmp / "y.csv")
        (self.tmp / "mdl-y").write_text("occupied", encoding="utf-8")
        with self.assertRaises(layout_mod.LayoutError) as ctx:
            layout_mod.open_project(csv_path)
        self.assertEqual(ctx.exception.code, "project_dir_conflict")

    def test_unwritable_csv_dir_maps_to_project_dir_not_writable(self) -> None:
        csv_path = _write_csv(self.tmp / "z.csv")
        with mock.patch.object(
            layout_mod,
            "check_writable",
            side_effect=fs.PathNotWritable("目录不可写"),
        ):
            with self.assertRaises(fs.PathNotWritable) as ctx:
                layout_mod.open_project(csv_path)
        self.assertEqual(ctx.exception.code, "project_dir_not_writable")

    def test_unknown_subdir_is_rejected(self) -> None:
        csv_path = _write_csv(self.tmp / "x.csv")
        layout = layout_mod.open_project(csv_path)
        with self.assertRaises(layout_mod.LayoutError) as ctx:
            layout.subdir("..")
        self.assertEqual(ctx.exception.code, "subdir_unknown")

    def test_resolve_project_reads_csv_path_from_manifest(self) -> None:
        csv_path = _write_csv(self.tmp / "x.csv")
        layout = layout_mod.open_project(csv_path)
        manifest_mod.ensure_manifest(layout.project_dir, csv_path)
        resolved = layout_mod.resolve_project(layout.project_dir)
        self.assertEqual(resolved.csv_path, csv_path.resolve())
        self.assertEqual(resolved.project_dir, layout.project_dir)

    def test_resolve_missing_project_dir_reports_code(self) -> None:
        with self.assertRaises(layout_mod.LayoutError) as ctx:
            layout_mod.resolve_project(self.tmp / "mdl-none")
        self.assertEqual(ctx.exception.code, "project_dir_not_found")


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="risklab-manifest-")
        self.tmp = Path(self._tmp.name)
        self.csv_path = _write_csv(self.tmp / "proj.csv")
        self.layout = layout_mod.open_project(self.csv_path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_new_manifest_matches_schema(self) -> None:
        manifest = manifest_mod.new_manifest(self.csv_path)
        self.assertEqual(manifest["schema"], manifest_mod.MANIFEST_SCHEMA)
        self.assertEqual(manifest["schema"], "risklab-mdl.manifest/1")
        self.assertEqual(sorted(manifest["stages"]), sorted(manifest_mod.STAGES))
        self.assertTrue(all(entry["done"] is False for entry in manifest["stages"].values()))
        self.assertEqual(manifest["runs"], [])
        self.assertIsNone(manifest["confirmed_run"])
        self.assertEqual(manifest["project"]["stem"], "proj")
        manifest_mod.validate_manifest(manifest)  # 不应抛

    def test_write_read_roundtrip(self) -> None:
        manifest = manifest_mod.new_manifest(self.csv_path)
        manifest_mod.write_manifest(self.layout.project_dir, manifest)
        reloaded = manifest_mod.read_manifest(self.layout.project_dir)
        self.assertEqual(reloaded["schema"], manifest_mod.MANIFEST_SCHEMA)
        self.assertEqual(reloaded["project"]["csv_path"], self.csv_path.resolve().as_posix())
        self.assertEqual(reloaded["project"]["stem"], "proj")
        # 落盘后是 UTF-8 无 BOM 的合法 JSON
        raw = (self.layout.project_dir / "manifest.json").read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        json.loads(raw.decode("utf-8"))

    def test_write_refreshes_updated_at(self) -> None:
        manifest = manifest_mod.new_manifest(self.csv_path)
        created_at = manifest["project"]["created_at"]
        # 注入 now：避免「同一秒内写入 ⇒ updated_at 恰好相同」这种恒等断言
        with mock.patch.object(manifest_mod, "now_iso", return_value="2026-05-05T10:00:00+08:00"):
            manifest_mod.write_manifest(self.layout.project_dir, manifest)
        reloaded = manifest_mod.read_manifest(self.layout.project_dir)
        self.assertEqual(reloaded["project"]["updated_at"], "2026-05-05T10:00:00+08:00")
        self.assertEqual(reloaded["project"]["created_at"], created_at, "created_at 不得被刷新")

    def test_same_second_writes_get_strictly_increasing_updated_at(self) -> None:
        """同一秒内连写，updated_at 必须严格递增（乐观校验才可能发现丢更新）。"""
        fixed = "2026-05-05T10:00:00+08:00"
        with mock.patch.object(manifest_mod, "now_iso", return_value=fixed):
            manifest_mod.ensure_manifest(self.layout.project_dir, self.csv_path)
            first = manifest_mod.update_manifest(self.layout.project_dir, lambda m: None)
            second = manifest_mod.update_manifest(self.layout.project_dir, lambda m: None)
        self.assertEqual(first["project"]["updated_at"], "2026-05-05T10:00:01+08:00")
        self.assertEqual(second["project"]["updated_at"], "2026-05-05T10:00:02+08:00")
        self.assertGreater(second["project"]["updated_at"], first["project"]["updated_at"])

    def test_missing_manifest_reports_code(self) -> None:
        with self.assertRaises(manifest_mod.ManifestError) as ctx:
            manifest_mod.read_manifest(self.layout.project_dir)
        self.assertEqual(ctx.exception.code, "manifest_not_found")

    def test_write_to_missing_project_dir_reports_code(self) -> None:
        manifest = manifest_mod.new_manifest(self.csv_path)
        with self.assertRaises(manifest_mod.ManifestError) as ctx:
            manifest_mod.write_manifest(self.tmp / "mdl-not-created", manifest)
        self.assertEqual(ctx.exception.code, "project_dir_not_found")
        self.assertFalse((self.tmp / "mdl-not-created").exists(), "不得偷偷建一个空项目目录")

    def test_update_missing_project_dir_reports_code(self) -> None:
        with self.assertRaises(manifest_mod.ManifestError) as ctx:
            manifest_mod.update_manifest(self.tmp / "mdl-nope", lambda m: None)
        self.assertEqual(ctx.exception.code, "project_dir_not_found")

    def test_invalid_schema_is_rejected_on_write(self) -> None:
        manifest = manifest_mod.new_manifest(self.csv_path)
        manifest["schema"] = "risklab-mdl.manifest/999"
        with self.assertRaises(manifest_mod.ManifestError) as ctx:
            manifest_mod.write_manifest(self.layout.project_dir, manifest)
        self.assertEqual(ctx.exception.code, "manifest_schema_invalid")

    def test_invalid_run_status_is_rejected(self) -> None:
        manifest = manifest_mod.new_manifest(self.csv_path)
        manifest["runs"].append({"run_id": "20260101T120000", "status": "unknown"})
        with self.assertRaises(manifest_mod.ManifestError) as ctx:
            manifest_mod.write_manifest(self.layout.project_dir, manifest)
        self.assertEqual(ctx.exception.code, "manifest_schema_invalid")
        self.assertIn("status", str(ctx.exception))

    def test_confirmed_run_must_exist_in_runs(self) -> None:
        manifest = manifest_mod.new_manifest(self.csv_path)
        manifest["runs"].append({"run_id": "20260101T120000", "status": "evaluated"})
        manifest["confirmed_run"] = "20260101T120000"
        manifest_mod.write_manifest(self.layout.project_dir, manifest)  # 合法
        manifest["confirmed_run"] = "20260101T130000"
        with self.assertRaises(manifest_mod.ManifestError):
            manifest_mod.write_manifest(self.layout.project_dir, manifest)

    # ---------- I2：manifest schema 边界校验 ----------

    def test_confirmed_run_requires_existing_run_even_when_runs_empty(self) -> None:
        manifest = manifest_mod.new_manifest(self.csv_path)
        manifest["confirmed_run"] = "20260101T120000"  # runs 为空 ⇒ 非法
        with self.assertRaises(manifest_mod.ManifestError) as ctx:
            manifest_mod.write_manifest(self.layout.project_dir, manifest)
        self.assertEqual(ctx.exception.code, "manifest_schema_invalid")
        self.assertIn("confirmed_run", str(ctx.exception))

    def test_duplicate_run_id_is_rejected(self) -> None:
        manifest = manifest_mod.new_manifest(self.csv_path)
        duplicate = {"run_id": "20260101T120000", "status": "trained", "task": "binary"}
        manifest["runs"] = [dict(duplicate), dict(duplicate)]
        with self.assertRaises(manifest_mod.ManifestError) as ctx:
            manifest_mod.write_manifest(self.layout.project_dir, manifest)
        self.assertEqual(ctx.exception.code, "manifest_schema_invalid")
        self.assertIn("重复", str(ctx.exception))
        self.assertIn("全局唯一", str(ctx.exception))

    def test_run_task_must_be_known_task_type(self) -> None:
        manifest = manifest_mod.new_manifest(self.csv_path)
        manifest["runs"] = [
            {"run_id": "20260101T120000", "status": "trained", "task": "regression"}
        ]
        with self.assertRaises(manifest_mod.ManifestError) as ctx:
            manifest_mod.write_manifest(self.layout.project_dir, manifest)
        self.assertEqual(ctx.exception.code, "manifest_schema_invalid")
        self.assertIn("task", str(ctx.exception))

        for task in ("binary", "multiclass"):
            manifest["runs"][0]["task"] = task
            manifest_mod.write_manifest(self.layout.project_dir, manifest)  # 合法任务类型通过

    def test_stage_extra_fields_are_type_checked(self) -> None:
        manifest = manifest_mod.new_manifest(self.csv_path)
        manifest["stages"]["data_analysis"] = {"done": True, "path": "data_analysis/", "summary": "ok"}
        manifest_mod.write_manifest(self.layout.project_dir, manifest)  # 合法

        manifest["stages"]["data_analysis"] = {"done": True, "path": 123, "summary": ["x"]}
        with self.assertRaises(manifest_mod.ManifestError) as ctx:
            manifest_mod.write_manifest(self.layout.project_dir, manifest)
        self.assertEqual(ctx.exception.code, "manifest_schema_invalid")
        message = str(ctx.exception)
        self.assertIn("stages.data_analysis.path", message)
        self.assertIn("stages.data_analysis.summary", message)

    # ---------- I1：并发/过期快照不得静默覆盖 ----------

    def test_stale_write_with_expected_updated_at_is_rejected(self) -> None:
        manifest_mod.ensure_manifest(self.layout.project_dir, self.csv_path)
        base = manifest_mod.read_manifest(self.layout.project_dir)
        stale = json.loads(json.dumps(base))  # 模拟「先读过」的快照
        # 期间别人完成了一次更新
        manifest_mod.update_manifest(
            self.layout.project_dir,
            lambda m: manifest_mod.stage_done(m, "data_analysis", path="data_analysis/"),
        )
        stale["stages"]["split"]["done"] = True
        with self.assertRaises(manifest_mod.ManifestError) as ctx:
            manifest_mod.write_manifest(
                self.layout.project_dir,
                stale,
                expected_updated_at=base["project"]["updated_at"],
            )
        self.assertEqual(ctx.exception.code, "manifest_conflict")
        self.assertIn("update_manifest", ctx.exception.hint)
        reloaded = manifest_mod.read_manifest(self.layout.project_dir)
        self.assertFalse(reloaded["stages"]["split"]["done"], "过期快照不得覆盖磁盘状态")
        self.assertTrue(reloaded["stages"]["data_analysis"]["done"], "别人的更新不得丢失")

    def test_concurrent_writers_cannot_silently_overwrite(self) -> None:
        """两个线程各读同一快照后并发写回：必须恰好一个成功、一个 manifest_conflict。

        旧实现（``write_manifest`` 不校验 ``expected_updated_at``）下两个线程都会
        「成功」，后写者静默覆盖前者的变更 —— 本用例正是那条 lost update 的复现；
        加 ``threading.Barrier`` 保证两个快照都基于同一版本。
        """
        manifest_mod.ensure_manifest(self.layout.project_dir, self.csv_path)
        base = manifest_mod.read_manifest(self.layout.project_dir)
        expected = base["project"]["updated_at"]
        barrier = threading.Barrier(2)
        outcomes: list = []

        def worker(stage: str) -> None:
            snapshot = json.loads(json.dumps(base))
            snapshot["stages"][stage]["done"] = True
            barrier.wait(timeout=30)
            try:
                manifest_mod.write_manifest(
                    self.layout.project_dir, snapshot, expected_updated_at=expected
                )
                outcomes.append(("ok", stage))
            except manifest_mod.ManifestError as exc:
                outcomes.append((exc.code, stage))

        workers = [
            threading.Thread(target=worker, args=(stage,))
            for stage in ("data_analysis", "split")
        ]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join(timeout=60)

        self.assertEqual(sorted(status for status, _ in outcomes), ["manifest_conflict", "ok"], outcomes)
        reloaded = manifest_mod.read_manifest(self.layout.project_dir)
        winner = [stage for status, stage in outcomes if status == "ok"][0]
        loser = "split" if winner == "data_analysis" else "data_analysis"
        self.assertTrue(reloaded["stages"][winner]["done"], "成功者的变更必须落盘")
        self.assertFalse(reloaded["stages"][loser]["done"], "失败者不得覆盖成功者的结果")

    def test_update_manifest_is_locked_read_modify_write(self) -> None:
        manifest_mod.ensure_manifest(self.layout.project_dir, self.csv_path)
        updated = manifest_mod.update_manifest(
            self.layout.project_dir,
            lambda m: manifest_mod.stage_done(m, "data_analysis", path="data_analysis/"),
        )
        self.assertTrue(updated["stages"]["data_analysis"]["done"])
        self.assertEqual(updated["stages"]["data_analysis"]["path"], "data_analysis/")
        reloaded = manifest_mod.read_manifest(self.layout.project_dir)
        self.assertTrue(reloaded["stages"]["data_analysis"]["done"])
        self.assertFalse(reloaded["stages"]["train"]["done"])

    def test_ensure_manifest_is_idempotent(self) -> None:
        first = manifest_mod.ensure_manifest(self.layout.project_dir, self.csv_path)
        second = manifest_mod.ensure_manifest(self.layout.project_dir, self.csv_path)
        self.assertEqual(first["project"]["created_at"], second["project"]["created_at"])

    def test_unknown_stage_is_rejected(self) -> None:
        manifest = manifest_mod.new_manifest(self.csv_path)
        with self.assertRaises(manifest_mod.ManifestError) as ctx:
            manifest_mod.stage_done(manifest, "not_a_stage")
        self.assertEqual(ctx.exception.code, "stage_unknown")


class RunStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="risklab-runs-")
        self.tmp = Path(self._tmp.name)
        self.runs_dir = self.tmp / "runs"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_run_id_matches_frozen_pattern(self) -> None:
        run_id, run_dir = store_mod.allocate_run_dir(self.runs_dir)
        self.assertRegex(run_id, r"^\d{8}T\d{6}$")
        self.assertIsNotNone(store_mod.RUN_ID_PATTERN.match(run_id))
        self.assertTrue(run_dir.is_dir())
        self.assertEqual(run_dir.name, run_id)

    def test_same_second_conflict_rolls_to_next_second(self) -> None:
        fixed = datetime(2026, 1, 1, 12, 0, 0)
        first_id, first_dir = store_mod.allocate_run_dir(self.runs_dir, now=fixed)
        second_id, second_dir = store_mod.allocate_run_dir(self.runs_dir, now=fixed)
        self.assertEqual(first_id, "20260101T120000")
        self.assertEqual(second_id, "20260101T120001")
        self.assertNotEqual(first_dir, second_dir)

    def test_allocate_run_id_is_consistent_with_dir(self) -> None:
        run_id = store_mod.allocate_run_id(self.runs_dir, now=datetime(2026, 2, 3, 4, 5, 6))
        self.assertEqual(run_id, "20260203T040506")
        self.assertTrue((self.runs_dir / run_id).is_dir())

    def test_list_and_latest_are_sorted(self) -> None:
        for second in (5, 1, 3):
            store_mod.allocate_run_dir(self.runs_dir, now=datetime(2026, 1, 1, 0, 0, second))
        # 干扰项：非时间戳目录不应出现在列表里
        (self.runs_dir / "debug").mkdir()
        ids = store_mod.list_run_ids(self.runs_dir)
        self.assertEqual(ids, ["20260101T000001", "20260101T000003", "20260101T000005"])
        self.assertEqual(store_mod.latest_run_id(self.runs_dir), "20260101T000005")

    def test_validate_run_id_rejects_bad_values(self) -> None:
        for bad in ("20260101-120000", "2026-01-01T12:00:00", "../escape", "", "20260101T1200000"):
            with self.assertRaises(store_mod.StoreError, msg=bad):
                store_mod.validate_run_id(bad)
        self.assertEqual(store_mod.validate_run_id("20260101T120000"), "20260101T120000")

    def test_resolve_run_dir_reports_missing(self) -> None:
        store_mod.allocate_run_dir(self.runs_dir, now=datetime(2026, 1, 1, 1, 1, 1))
        found = store_mod.resolve_run_dir(self.runs_dir, "20260101T010101")
        self.assertTrue(found.is_dir())
        with self.assertRaises(store_mod.StoreError) as ctx:
            store_mod.resolve_run_dir(self.runs_dir, "20260101T010102")
        self.assertEqual(ctx.exception.code, "run_not_found")

    def test_layout_run_dir_matches_store_allocation(self) -> None:
        csv_path = _write_csv(self.tmp / "p.csv")
        layout = layout_mod.open_project(csv_path)
        run_id, run_dir = store_mod.allocate_run_dir(layout.runs_dir)
        self.assertEqual(layout.run_dir(run_id), run_dir)
        self.assertEqual(layout.rel(run_dir), f"runs/{run_id}")


class AcceptanceChainTests(unittest.TestCase):
    """验收 ⑤ 的最小闭环：临时目录建 mdl-<stem>/ + manifest 往返 + run 目录命名。"""

    def test_end_to_end_on_temp_dir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="risklab-accept-") as tmp:
            csv_path = _write_csv(Path(tmp) / "accept.csv")
            layout = layout_mod.open_project(csv_path)
            self.assertEqual(layout.project_dir.name, "mdl-accept")

            manifest = manifest_mod.new_manifest(csv_path)
            manifest_mod.stage_done(manifest, "data_analysis", path="data_analysis/")
            manifest_mod.write_manifest(layout.project_dir, manifest)
            reloaded = manifest_mod.read_manifest(layout.project_dir)
            self.assertTrue(reloaded["stages"]["data_analysis"]["done"])

            run_id, run_dir = store_mod.allocate_run_dir(layout.runs_dir)
            self.assertRegex(run_dir.name, r"^\d{8}T\d{6}$")
            self.assertRegex(run_id, r"^\d{8}T\d{6}$")


if __name__ == "__main__":
    unittest.main()
