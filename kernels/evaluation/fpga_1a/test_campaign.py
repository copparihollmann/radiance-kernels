"""End-to-end checks for the data-driven FPGA-1A campaign adapter."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from campaign import (SafetyAbort, archive_incomplete_safety_attempt,
                      descendant_pids, exclusive_host_lock, execute,
                      external_firesim_activity, inventory,
                      is_firesim_hardware_process, load_assets, make_plan,
                      process_table, sha256, summarize)


class CampaignTest(unittest.TestCase):
    def test_archives_incomplete_safety_attempt_before_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory) / "job"
            result_dir.mkdir()
            for name, contents in (
                    ("safety-abort.json", "{}\n"),
                    ("request.json", "{}\n"),
                    ("stdout.log", "partial stdout\n"),
                    ("stderr.log", "partial stderr\n")):
                (result_dir / name).write_text(contents)
            first = archive_incomplete_safety_attempt(result_dir)
            self.assertEqual(first.name, "safety-abort-001")
            self.assertFalse((result_dir / "safety-abort.json").exists())
            self.assertEqual((first / "stdout.log").read_text(), "partial stdout\n")
            (result_dir / "safety-abort.json").write_text("{}\n")
            second = archive_incomplete_safety_attempt(result_dir)
            self.assertEqual(second.name, "safety-abort-002")

    def test_process_table_tolerates_disappearing_proc_entry(self):
        with patch.object(Path, "read_bytes", side_effect=ProcessLookupError):
            self.assertEqual(process_table(), [])

    def test_cooperative_host_lock_is_exclusive_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "firesim-host.lock"
            lock.write_bytes(b"unchanged")
            before = lock.stat()
            with exclusive_host_lock(lock):
                with self.assertRaises(SafetyAbort):
                    with exclusive_host_lock(lock):
                        pass
            after = lock.stat()
            self.assertEqual(lock.read_bytes(), b"unchanged")
            self.assertEqual(after.st_size, before.st_size)
            self.assertEqual(after.st_mtime_ns, before.st_mtime_ns)

    def test_cooperative_host_lock_must_already_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SafetyAbort):
                with exclusive_host_lock(Path(directory) / "missing.lock"):
                    pass

    def test_external_firesim_activity_excludes_owned_process_tree(self):
        rows = [
            {"pid": 10, "ppid": 1, "uid": 1000,
             "argv": ["python3", "campaign.py", "run"]},
            {"pid": 11, "ppid": 10, "uid": 1000,
             "argv": ["bash", "/copy/runtime/run.sh", "-b", "1SM_DRAM",
                      "--bdf", "1b:00.0"]},
            {"pid": 12, "ppid": 11, "uid": 0,
             "argv": ["python3", "/usr/local/bin/firesim-fpga-util.py"]},
            {"pid": 20, "ppid": 1, "uid": 2000,
             "argv": ["bash", "/other/firesim-rundir/run.sh", "-b",
                      "2SM_HBM", "--bdf", "61:00.0"]},
            {"pid": 21, "ppid": 1, "uid": 2000,
             "argv": ["rg", "FireSim-corigine|program_fpga"]},
        ]
        self.assertEqual(descendant_pids(rows, {10}), {10, 11, 12})
        activity = external_firesim_activity(rows, {10})
        self.assertEqual([row["pid"] for row in activity], [20])

    def test_firesim_hardware_process_signatures(self):
        self.assertTrue(is_firesim_hardware_process(
            ["./FireSim-corigine_xb10", "+bus=0x1b"]))
        self.assertTrue(is_firesim_hardware_process(
            ["vivado", "-source", "/usr/local/bin/program_fpga.tcl"]))
        self.assertTrue(is_firesim_hardware_process(
            ["bash", "/other/firesim-rundir/run.sh", "-b", "1SM_DRAM",
             "/tmp/workload.elf"]))
        self.assertTrue(is_firesim_hardware_process(
            ["bash", "/other/firesim-rundir/run.sh", "--fpga", "serial",
             "/tmp/workload.elf"]))
        self.assertTrue(is_firesim_hardware_process(
            ["python3", "/deploy/firesim", "runworkload"]))
        self.assertFalse(is_firesim_hardware_process(
            ["rg", "FireSim-corigine|program_fpga.tcl"]))

    def test_plan_run_resume_and_summarize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            assets.mkdir()
            (assets / "one.bit").write_bytes(b"prebuilt-bitstream")
            (assets / "runtime.yaml").write_text("runtime: test\n")
            (assets / "runner.py").write_text(
                "print('cluster 0 core 0: start=1 end=1201 cycles=1200')\n"
                "print('cluster 0 core 1: start=2 end=1235 cycles=1233')\n"
                "print('kernel_cycles=1234')\nprint('PASS')\n"
            )
            elf = root / "kernel.soc.elf"
            elf.write_bytes(b"elf")
            build = {
                "hardware_rebuilt": False, "no_vpu": True,
                "workloads": [{
                    "name": "stream--copy--1sm", "family": "stream",
                    "operation": "copy", "sm_count": 1, "loader": "logical",
                    "elf": str(elf), "elf_sha256": sha256(elf), "status": "built",
                }],
            }
            build_path = root / "build.json"
            build_path.write_text(json.dumps(build))
            asset_spec = {
                "schema": "radiance-fpga-1a-assets-v1",
                "targets": [{
                    "id": "one", "kind": "firesim", "sm_count": 1,
                    "memory": "hbm", "loader": "logical",
                    "bitstream": "one.bit", "runtime_config": "runtime.yaml",
                    "runner": "runner.py",
                    "require_per_sm_timing": True,
                    "run_argv": ["python3", "{runner}", "{elf}", "{bitstream}"],
                    "reuse_argv": ["python3", "missing-reuse-runner.py", "{elf}"],
                }],
            }
            asset_path = root / "assets.json"
            asset_path.write_text(json.dumps(asset_spec))
            self.assertEqual(inventory(assets)["file_count"], 3)
            spec, receipts = load_assets(asset_path, assets)
            self.assertEqual(spec["targets"][0]["id"], "one")
            self.assertIn("one.bit", receipts["one.bitstream"]["path"])
            run_root = root / "runs"
            plan = make_plan(build_path, asset_path, assets, run_root,
                             warmups=1, repetitions=3, allow_incomplete=True)
            self.assertEqual(len(plan["jobs"]), 4)
            plan_path = run_root / "plan.json"
            host_lock = root / "host.lock"
            host_lock.write_bytes(b"")
            with self.assertRaisesRegex(SafetyAbort, "require --host-lock"):
                execute(plan_path, rerun_failed=False)
            with self.assertRaisesRegex(ValueError, "selection produced no jobs"):
                execute(plan_path, rerun_failed=False, host_lock=host_lock,
                        exclude_families={"stream"})
            with self.assertRaisesRegex(ValueError, "unknown campaign job IDs"):
                execute(plan_path, rerun_failed=False, host_lock=host_lock,
                        include_jobs={"not-a-real-job"})
            selected_id = plan["jobs"][0]["id"]
            execute(plan_path, rerun_failed=False, host_lock=host_lock,
                    include_jobs={selected_id})
            self.assertTrue(
                Path(plan["jobs"][0]["result_dir"], "result.json").is_file()
            )
            self.assertFalse(
                Path(plan["jobs"][1]["result_dir"], "result.json").exists()
            )
            execute(plan_path, rerun_failed=False, host_lock=host_lock)
            execute(plan_path, rerun_failed=False, host_lock=host_lock)
            report = summarize(plan_path, root / "summary.json")
            self.assertFalse(report["missing_jobs"])
            self.assertEqual(report["rows"][0]["measure_passes"], 3)
            self.assertEqual(report["rows"][0]["median_kernel_cycles"], 1234)
            result = json.loads(Path(plan["jobs"][0]["result_dir"], "result.json").read_text())
            self.assertTrue(result["per_sm_timing_complete"])
            self.assertEqual(len(result["core_kernel_cycles"]), 2)
            for job in plan["jobs"]:
                request = json.loads(Path(job["result_dir"], "request.json").read_text())
                self.assertFalse(request["reused_loaded_bitstream"])
                self.assertNotIn("missing-reuse-runner.py", request["argv"])

    def test_failed_job_stops_campaign_and_requires_explicit_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            assets.mkdir()
            (assets / "one.bit").write_bytes(b"prebuilt-bitstream")
            (assets / "runner.py").write_text("print('FAIL')\n")
            workloads = []
            for operation in ("copy", "scale"):
                elf = root / f"{operation}.elf"
                elf.write_bytes(operation.encode())
                workloads.append({
                    "name": f"stream--{operation}--1sm", "family": "stream",
                    "operation": operation, "sm_count": 1, "loader": "logical",
                    "elf": str(elf), "elf_sha256": sha256(elf), "status": "built",
                })
            build_path = root / "build.json"
            build_path.write_text(json.dumps({
                "hardware_rebuilt": False, "no_vpu": True,
                "workloads": workloads,
            }))
            asset_path = root / "assets.json"
            asset_path.write_text(json.dumps({
                "schema": "radiance-fpga-1a-assets-v1",
                "targets": [{
                    "id": "one", "kind": "firesim", "sm_count": 1,
                    "memory": "dram", "loader": "logical",
                    "bitstream": "one.bit", "runner": "runner.py",
                    "run_argv": ["python3", "{runner}", "{elf}", "{bitstream}"],
                }],
            }))
            run_root = root / "runs"
            plan = make_plan(build_path, asset_path, assets, run_root,
                             warmups=0, repetitions=1, allow_incomplete=True)
            host_lock = root / "host.lock"
            host_lock.write_bytes(b"")
            with self.assertRaisesRegex(SafetyAbort, "failed; stopping"):
                execute(run_root / "plan.json", rerun_failed=False,
                        host_lock=host_lock)
            first = Path(plan["jobs"][0]["result_dir"])
            second = Path(plan["jobs"][1]["result_dir"])
            self.assertEqual(json.loads((first / "result.json").read_text())["status"],
                             "failed")
            self.assertFalse((second / "request.json").exists())
            with self.assertRaisesRegex(SafetyAbort, "prior failed result"):
                execute(run_root / "plan.json", rerun_failed=False,
                        host_lock=host_lock)

    def test_rejects_omitted_bitstream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.bit").write_bytes(b"one")
            (root / "two.bit").write_bytes(b"two")
            (root / "runner.py").write_text("print('PASS')\n")
            spec = {
                "schema": "radiance-fpga-1a-assets-v1",
                "targets": [{
                    "id": "one", "kind": "firesim", "sm_count": 1,
                    "memory": "dram", "loader": "logical",
                    "bitstream": "one.bit", "runner": "runner.py",
                    "run_argv": ["python3", "{runner}", "{elf}", "{bitstream}"],
                }],
            }
            path = root / "assets.json"
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "two.bit"):
                load_assets(path, root)

    def test_registry_selected_bitstream_and_single_bdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "firesim_radiance1SM_DRAM.tar.gz"
            driver = root / "FireSim-corigine_mimicturbo_gt-radiance1SM_DRAM"
            runner = root / "run.sh"
            archive.write_bytes(b"archive")
            driver.write_bytes(b"driver")
            runner.write_text("#!/bin/sh\n")
            spec = {
                "schema": "radiance-fpga-1a-assets-v1",
                "single_fpga_bdf": "1b:00.0",
                "targets": [{
                    "id": "one", "kind": "firesim", "sm_count": 1,
                    "memory": "dram", "loader": "logical", "bdf": "1b:00.0",
                    "bitstream": archive.name, "driver": driver.name,
                    "runner": runner.name, "bitstream_config": "1SM_DRAM",
                    "run_argv": [
                        "bash", "{runner}", "-b", "{bitstream_config}",
                        "--bdf", "{bdf}", "{elf}",
                    ],
                }],
            }
            path = root / "assets.json"
            path.write_text(json.dumps(spec))
            loaded, _ = load_assets(path, root)
            self.assertEqual(loaded["targets"][0]["bdf"], "1b:00.0")
            spec["targets"][0]["driver"] = "wrong-driver"
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "does not match driver"):
                load_assets(path, root)


if __name__ == "__main__":
    unittest.main()
