"""End-to-end checks for the data-driven FPGA-1A campaign adapter."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from campaign import (FPGA_BDF_ALLOWLIST, SafetyAbort, archive_prior_attempt,
                      descendant_pids, exclusive_host_lock, execute,
                      external_firesim_activity, inventory,
                      is_firesim_hardware_process, load_assets, make_plan,
                      parallel_execution_groups, process_table, sha256,
                      summarize, target_execution_order, update_owned_pids)


def firesim_job(job_id: str, target_id: str, bdf: str | None) -> dict:
    target = {"id": target_id, "kind": "firesim", "sm_count": 2,
              "memory": "dram", "loader": "logical"}
    if bdf is not None:
        target["bdf"] = bdf
    return {"id": job_id, "target_id": target_id, "target": target}


def coordination_ack(root: Path, host_lock: Path) -> Path:
    path = root / "host-coordination.json"
    path.write_text(json.dumps({
        "schema": "firesim-host-wide-lock-coordination-v1",
        "host_lock": str(host_lock.resolve()),
        "programming_rescans_host_pcie": True,
        "complete_driver_lifetime_locked": True,
        "acknowledged_users": ["test-user-a", "test-user-b"],
    }))
    return path


class CampaignTest(unittest.TestCase):
    def test_primary_dram_scaling_targets_run_before_secondary_models(self):
        targets = [
            {"id": "baseline", "sm_count": 2,
             "bitstream_config": "2SM_DRAM_baseline"},
            {"id": "hbm", "sm_count": 2,
             "bitstream_config": "2SM_HBM"},
            {"id": "four", "sm_count": 4,
             "bitstream_config": "4SM_DRAM"},
            {"id": "one", "sm_count": 1,
             "bitstream_config": "1SM_DRAM"},
            {"id": "two", "sm_count": 2,
             "bitstream_config": "2SM_DRAM"},
        ]
        ordered = sorted(targets, key=target_execution_order)
        self.assertEqual([target["id"] for target in ordered],
                         ["one", "two", "four", "baseline", "hbm"])

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
            profile = result_dir / "memory-profile"
            profile.mkdir()
            (profile / ".profile-start.123.marker").write_text("")
            (profile / "partial.csv").write_text("cycle,value\n1,2\n")
            first = archive_prior_attempt(result_dir)
            self.assertEqual(first.name, "attempt-001")
            self.assertFalse((result_dir / "safety-abort.json").exists())
            self.assertFalse((result_dir / "memory-profile").exists())
            self.assertEqual((first / "stdout.log").read_text(), "partial stdout\n")
            self.assertEqual(
                (first / "memory-profile/partial.csv").read_text(),
                "cycle,value\n1,2\n",
            )
            (result_dir / "safety-abort.json").write_text("{}\n")
            second = archive_prior_attempt(result_dir)
            self.assertEqual(second.name, "attempt-002")

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

    def test_external_activity_is_scoped_to_target_bdf_and_fails_closed(self):
        rows = [
            {"pid": 20, "ppid": 1, "uid": 2000,
             "argv": ["bash", "/other/run.sh", "-b", "2SM_HBM",
                      "--bdf", "1b:00.0"]},
            {"pid": 21, "ppid": 20, "uid": 0,
             "argv": ["vivado", "-source", "program_fpga.tcl"]},
            {"pid": 30, "ppid": 1, "uid": 3000,
             "argv": ["bash", "/target/run.sh", "-b", "4SM_DRAM",
                      "--bdf=0000:1a:00.0"]},
            {"pid": 31, "ppid": 30, "uid": 0,
             "argv": ["vivado", "-source", "program_fpga.tcl"]},
            {"pid": 40, "ppid": 1, "uid": 4000,
             "argv": ["python3", "/deploy/firesim", "runworkload"]},
            {"pid": 50, "ppid": 1, "uid": 5000,
             "argv": ["/generated/VFireSim", "+bus=0x60", "+device=0"]},
        ]
        self.assertEqual(
            [row["pid"] for row in
             external_firesim_activity(rows, {10}, "1a:00.0")],
            [30, 31, 40],
        )
        self.assertEqual(
            [row["pid"] for row in
             external_firesim_activity(rows, {10}, "1b:00.0")],
            [20, 21, 40],
        )
        with self.assertRaises(ValueError):
            external_firesim_activity(rows, {10}, "not-a-bdf")

    def test_other_card_activity_does_not_block_but_unknown_card_does(self):
        """Nicolas on 60/61 must not block us; anything unattributable must."""
        other_card = {"pid": 60, "ppid": 1, "uid": 2000,
                      "argv": ["bash", "/other/run.sh", "-b", "quad_saturn",
                               "--bdf", "60:00.0"]}
        other_card_child = {"pid": 61, "ppid": 60, "uid": 2000,
                            "argv": ["./FireSim-corigine_quad_saturn",
                                     "+bus=0x60", "+device=0x00"]}
        same_card = {"pid": 70, "ppid": 1, "uid": 2000,
                     "argv": ["bash", "/other/run.sh", "-b", "2SM_HBM",
                              "--bdf", "1a:00.0"]}
        no_card = {"pid": 80, "ppid": 1, "uid": 2000,
                   "argv": ["python3", "/deploy/firesim", "runworkload"]}
        # A driver whose own plusargs and whose launcher disagree about the
        # card resolves to two BDFs, so it must block every target.
        ambiguous_parent = {"pid": 90, "ppid": 1, "uid": 2000,
                            "argv": ["bash", "/other/run.sh", "-b", "x",
                                     "--bdf", "1b:00.0"]}
        ambiguous = {"pid": 91, "ppid": 90, "uid": 2000,
                     "argv": ["./FireSim-corigine_other", "+bus=0x61",
                              "+device=0x00"]}

        ours = {"pid": 999, "ppid": 1, "uid": 1000,
                "argv": ["python3", "campaign.py", "run"]}
        rows = [ours, other_card, other_card_child]
        self.assertEqual(
            external_firesim_activity(rows, {999}, "1a:00.0"), [])
        # Without a target BDF the guard stays host-wide, which is exactly the
        # behaviour that made the controller unrunnable while 60/61 are busy.
        self.assertEqual(
            [row["pid"] for row in external_firesim_activity(rows, {999})],
            [60, 61])

        self.assertEqual(
            [row["pid"] for row in
             external_firesim_activity([ours, same_card], {999}, "1a:00.0")],
            [70])
        self.assertEqual(
            [row["pid"] for row in
             external_firesim_activity([ours, no_card], {999}, "1a:00.0")],
            [80])
        self.assertEqual(
            [row["pid"] for row in external_firesim_activity(
                [ours, ambiguous_parent, ambiguous], {999}, "1a:00.0")],
            [91])
        with self.assertRaises(ValueError):
            external_firesim_activity(rows, {999}, "1a:00")

    def test_parallel_groups_key_firesim_jobs_by_card_not_target_id(self):
        """Five target configurations over two cards are two parallel groups."""
        layout = [
            ("1sm-dram", "1a:00.0", 9),
            ("2sm-dram", "1a:00.0", 9),
            ("2sm-dram-baseline", "1b:00.0", 9),
            ("2sm-hbm", "1b:00.0", 9),
            ("4sm-dram", "1a:00.0", 9),
        ]
        jobs = [firesim_job(f"{target_id}--w{index}", target_id, bdf)
                for target_id, bdf, count in layout
                for index in range(count)]
        groups = parallel_execution_groups(jobs)
        self.assertEqual(sorted(groups), ["fpga:1a:00.0", "fpga:1b:00.0"])
        self.assertEqual(len(groups["fpga:1a:00.0"]), 27)
        self.assertEqual(len(groups["fpga:1b:00.0"]), 18)
        # Each card keeps its jobs in plan order so it reprograms between
        # configurations instead of interleaving them.
        self.assertEqual(
            [job["id"] for job in groups["fpga:1a:00.0"][:2]],
            ["1sm-dram--w0", "1sm-dram--w1"])
        self.assertEqual(groups["fpga:1b:00.0"][0]["id"],
                         "2sm-dram-baseline--w0")
        self.assertEqual(
            sum(len(jobs) for jobs in groups.values()), len(jobs))

    def test_parallel_groups_reject_missing_foreign_and_excess_cards(self):
        self.assertEqual(FPGA_BDF_ALLOWLIST, frozenset({"1a:00.0", "1b:00.0"}))
        with self.assertRaisesRegex(ValueError, "declares no bdf"):
            parallel_execution_groups([firesim_job("a", "a", "1a:00.0"),
                                       firesim_job("b", "b", None)])
        with self.assertRaisesRegex(ValueError, "declares no bdf"):
            parallel_execution_groups([firesim_job("a", "a", "")])
        with self.assertRaisesRegex(ValueError, "unparseable FPGA BDF"):
            parallel_execution_groups([firesim_job("a", "a", "1a:00")])
        with self.assertRaisesRegex(ValueError, "outside the campaign allowlist"):
            parallel_execution_groups([firesim_job("a", "a", "1a:00.0"),
                                       firesim_job("b", "b", "61:00.0")])
        with self.assertRaisesRegex(ValueError, "outside the campaign allowlist"):
            parallel_execution_groups([firesim_job("a", "a", "1a:00.0"),
                                       firesim_job("b", "b", "1b:00.0"),
                                       firesim_job("c", "c", "b1:00.0")])
        # The allowlist already makes a third card unreachable; the group cap
        # is the independent backstop if the allowlist is ever widened.
        with patch("campaign.FPGA_BDF_ALLOWLIST",
                   frozenset({"1a:00.0", "1b:00.0", "b1:00.0"})):
            with self.assertRaisesRegex(ValueError,
                                        "limited to 2 concurrent FPGAs"):
                parallel_execution_groups([firesim_job("a", "a", "1a:00.0"),
                                           firesim_job("b", "b", "1b:00.0"),
                                           firesim_job("c", "c", "b1:00.0")])

    def test_parallel_groups_keep_model_targets_per_target_id(self):
        model = {"id": "m0", "target_id": "model-a",
                 "target": {"id": "model-a", "kind": "model"}}
        other = {"id": "m1", "target_id": "model-b",
                 "target": {"id": "model-b", "kind": "model"}}
        groups = parallel_execution_groups([model, other, dict(model, id="m2")])
        self.assertEqual(sorted(groups), ["target:model-a", "target:model-b"])
        self.assertEqual(len(groups["target:model-a"]), 2)

    def test_owned_pid_history_survives_reparenting_but_not_exit(self):
        initial = [
            {"pid": 10, "ppid": 1, "uid": 1000,
             "argv": ["python3", "campaign.py"]},
            {"pid": 11, "ppid": 10, "uid": 1000,
             "argv": ["timeout", "run.sh"]},
            {"pid": 12, "ppid": 11, "uid": 1000,
             "argv": ["./FireSim-corigine_xb10"]},
        ]
        owned = update_owned_pids(initial, {10, 11})
        self.assertEqual(owned, {10, 11, 12})
        reparented = [
            {"pid": 10, "ppid": 1, "uid": 1000,
             "argv": ["python3", "campaign.py"]},
            {"pid": 12, "ppid": 1, "uid": 1000,
             "argv": ["./FireSim-corigine_xb10"]},
            {"pid": 20, "ppid": 1, "uid": 2000,
             "argv": ["./FireSim-corigine_other"]},
        ]
        owned = update_owned_pids(reparented, owned)
        self.assertEqual(owned, {10, 12})
        self.assertEqual(
            [row["pid"] for row in external_firesim_activity(reparented, owned)],
            [20],
        )
        exited = [reparented[0], reparented[2]]
        self.assertEqual(update_owned_pids(exited, owned), {10})

    def test_firesim_hardware_process_signatures(self):
        self.assertTrue(is_firesim_hardware_process(
            ["./FireSim-corigine_xb10", "+bus=0x1b"]))
        self.assertTrue(is_firesim_hardware_process(
            ["/generated-src/design/VFireSim", "+bus=0x1b",
             "+device=0x00"]))
        self.assertFalse(is_firesim_hardware_process(
            ["/generated-src/design/VFireSim", "+mm_backendLatency_0=2"]))
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
            ack = coordination_ack(root, host_lock)
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
                    host_coordination_ack=ack,
                    include_jobs={selected_id})
            self.assertTrue(
                Path(plan["jobs"][0]["result_dir"], "result.json").is_file()
            )
            self.assertFalse(
                Path(plan["jobs"][1]["result_dir"], "result.json").exists()
            )
            execute(plan_path, rerun_failed=False, host_lock=host_lock,
                    host_coordination_ack=ack)
            execute(plan_path, rerun_failed=False, host_lock=host_lock,
                    host_coordination_ack=ack)
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
            ack = coordination_ack(root, host_lock)
            with self.assertRaisesRegex(SafetyAbort, "failed; stopping"):
                execute(run_root / "plan.json", rerun_failed=False,
                        host_lock=host_lock, host_coordination_ack=ack)
            first = Path(plan["jobs"][0]["result_dir"])
            second = Path(plan["jobs"][1]["result_dir"])
            self.assertEqual(json.loads((first / "result.json").read_text())["status"],
                             "failed")
            self.assertFalse((second / "request.json").exists())
            with self.assertRaisesRegex(SafetyAbort, "prior failed result"):
                execute(run_root / "plan.json", rerun_failed=False,
                        host_lock=host_lock, host_coordination_ack=ack)

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


class DualCardExecutionTest(unittest.TestCase):
    """The controller must drive 1a:00.0 and 1b:00.0 at the same time."""

    CONFIGS = (("dram-v2", "2SM_DRAM_v2", "dram", "1a:00.0"),
               ("hbm-v2", "2SM_HBM_v2", "hbm", "1b:00.0"))

    def build_plan(self, root: Path) -> tuple[dict, Path, Path, Path]:
        assets = root / "assets"
        assets.mkdir()
        (assets / "runner.py").write_text("print('PASS')\n")
        targets = []
        for target_id, config, memory, bdf in self.CONFIGS:
            (assets / f"firesim_radiance{config}.tar.gz").write_bytes(
                config.encode())
            (assets / f"FireSim-corigine_mimicturbo_gt-radiance{config}"
             ).write_bytes(config.encode())
            targets.append({
                "id": target_id, "kind": "firesim", "sm_count": 2,
                "memory": memory, "loader": "logical", "bdf": bdf,
                "bitstream": f"firesim_radiance{config}.tar.gz",
                "driver": f"FireSim-corigine_mimicturbo_gt-radiance{config}",
                "runner": "runner.py", "bitstream_config": config,
                "run_argv": ["python3", "{runner}", "-b", "{bitstream_config}",
                             "--bdf", "{bdf}", "{elf}"],
            })
        workloads = []
        for name in ("stream--copy--2sm", "stream--add--2sm"):
            elf = root / f"{name}.elf"
            elf.write_bytes(name.encode())
            workloads.append({
                "name": name, "family": "stream", "operation": name.split("--")[1],
                "sm_count": 2, "loader": "logical", "elf": str(elf),
                "elf_sha256": sha256(elf), "status": "built",
            })
        build_path = root / "build.json"
        build_path.write_text(json.dumps({
            "hardware_rebuilt": False, "no_vpu": True, "workloads": workloads}))
        asset_path = root / "assets.json"
        asset_path.write_text(json.dumps({
            "schema": "radiance-fpga-1a-assets-v1", "targets": targets}))
        run_root = root / "runs"
        plan = make_plan(build_path, asset_path, assets, run_root,
                         warmups=0, repetitions=1, allow_incomplete=True)
        host_lock = root / "host.lock"
        host_lock.write_bytes(b"")
        return (plan, run_root / "plan.json", host_lock,
                coordination_ack(root, host_lock))

    def test_dual_card_parallel_run_scopes_the_guard_to_each_card(self):
        observed_bdfs = []

        def recorder(rows=None, owned_roots=None, target_bdf=None):
            observed_bdfs.append(target_bdf)
            return []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, plan_path, host_lock, ack = self.build_plan(root)
            self.assertEqual(len(plan["jobs"]), 4)
            with patch("campaign.external_firesim_activity", recorder):
                execute(plan_path, rerun_failed=False, parallel_targets=True,
                        host_lock=host_lock, host_coordination_ack=ack)
            self.assertEqual(set(observed_bdfs), {"1a:00.0", "1b:00.0"})
            self.assertNotIn(None, observed_bdfs)
            for job in plan["jobs"]:
                result = json.loads(
                    Path(job["result_dir"], "result.json").read_text())
                self.assertEqual(result["status"], "passed")
                self.assertFalse(result["reused_loaded_bitstream"])
                request = json.loads(
                    Path(job["result_dir"], "request.json").read_text())
                self.assertNotIn("--no-program", request["argv"])

    def test_unparseable_bdf_stops_the_job_instead_of_widening_the_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, plan_path, host_lock, ack = self.build_plan(root)
            mutated = json.loads(plan_path.read_text())
            for job in mutated["jobs"]:
                job["target"]["bdf"] = "1g:00.0"
            plan_path.write_text(json.dumps(mutated))
            with patch("campaign.external_firesim_activity",
                       lambda *a, **k: []):
                with self.assertRaisesRegex(SafetyAbort, "unparseable FPGA BDF"):
                    execute(plan_path, rerun_failed=False, host_lock=host_lock,
                            host_coordination_ack=ack)

    def test_firesim_no_program_reuse_is_still_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, plan_path, host_lock, ack = self.build_plan(root)
            mutated = json.loads(plan_path.read_text())
            for job in mutated["jobs"]:
                job["target"]["run_argv"] = (
                    job["target"]["run_argv"][:-1] + ["--no-program"] +
                    job["target"]["run_argv"][-1:])
            plan_path.write_text(json.dumps(mutated))
            with patch("campaign.external_firesim_activity",
                       lambda *a, **k: []):
                with self.assertRaisesRegex(SafetyAbort, "--no-program"):
                    execute(plan_path, rerun_failed=False, host_lock=host_lock,
                            host_coordination_ack=ack)


if __name__ == "__main__":
    unittest.main()
