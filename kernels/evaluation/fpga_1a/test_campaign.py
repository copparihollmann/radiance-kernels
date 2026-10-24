"""End-to-end checks for the data-driven FPGA-1A campaign adapter."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from campaign import execute, inventory, load_assets, make_plan, sha256, summarize


class CampaignTest(unittest.TestCase):
    def test_plan_run_resume_and_summarize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            assets.mkdir()
            (assets / "one.bit").write_bytes(b"prebuilt-bitstream")
            (assets / "runtime.yaml").write_text("runtime: test\n")
            (assets / "runner.py").write_text(
                "print('PASS')\nprint('kernel_cycles=1234')\n"
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
                    "run_argv": ["python3", "{runner}", "{elf}", "{bitstream}"],
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
            execute(plan_path, rerun_failed=False)
            execute(plan_path, rerun_failed=False)
            report = summarize(plan_path, root / "summary.json")
            self.assertFalse(report["missing_jobs"])
            self.assertEqual(report["rows"][0]["measure_passes"], 3)
            self.assertEqual(report["rows"][0]["median_kernel_cycles"], 1234)

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


if __name__ == "__main__":
    unittest.main()
