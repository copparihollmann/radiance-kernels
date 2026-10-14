"""Reject unverified simulator completion in SpectFormer handoffs."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kernels.spectformer.package_handoff import checked_receipt_sources, package, sha256
from kernels.spectformer.verify_handoff import verify


HERE = Path(__file__).resolve().parent


class HandoffReceiptTest(unittest.TestCase):
    def setUp(self):
        temporary_root = HERE.parents[2] / ".tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        (self.run_dir / "logits.bin").write_bytes(bytes.fromhex("0000803f00000040"))
        (self.run_dir / "verified-reference.bin").write_bytes(
            (self.run_dir / "logits.bin").read_bytes())
        self.build = {"radiance_elf_sha256": "elf-digest"}
        self.export = {"weights_sha256": "weights-digest",
                       "input_sha256": "input-digest"}
        self.raw_path = self.run_dir / "result.json"
        self.raw_path.write_text(json.dumps({
            "status": "passed", "reference_checked": False,
            "output_kind": "logits", "elf_sha256": "elf-digest",
            "weights_sha256": "weights-digest", "input_sha256": "input-digest",
        }))
        self.verified_path = self.run_dir / "verification.json"
        self.verified_path.write_text(json.dumps({
            "status": "passed", "reference_checked": True,
            "all_finite": True, "out_of_tolerance": 0,
            "scope": "logits", "output_count": 2,
            "elf_sha256": "elf-digest", "weights_sha256": "weights-digest",
            "input_sha256": "input-digest",
            "run_result_sha256": sha256(self.raw_path),
            "output_sha256": sha256(self.run_dir / "logits.bin"),
            "reference_sha256": sha256(self.run_dir / "verified-reference.bin"),
        }))

    def tearDown(self):
        self.temporary.cleanup()

    def test_no_reference_run_cannot_be_packaged_as_pass(self):
        with self.assertRaisesRegex(ValueError, "lacks a checked numerical pass"):
            checked_receipt_sources("case", self.raw_path, self.build,
                                    self.export, "logits", 2)

    def test_checked_receipt_is_bound_to_raw_run_and_reference(self):
        sources = checked_receipt_sources("case", self.verified_path,
                                          self.build, self.export, "logits", 2)
        self.assertEqual(set(sources), {
            "simulator-result.json", "simulator-verification.json", "reference.bin"})
        raw = json.loads(self.raw_path.read_text())
        raw["input_sha256"] = "other-input"
        self.raw_path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "raw simulator run differs"):
            checked_receipt_sources("case", self.verified_path, self.build,
                                    self.export, "logits", 2)

    def test_handoff_verifier_requires_a_numerical_pass(self):
        case_dir = self.root / "case"
        case_dir.mkdir()
        for source, destination in (
                (self.raw_path, "simulator-result.json"),
                (self.verified_path, "simulator-verification.json"),
                (self.run_dir / "verified-reference.bin", "reference.bin")):
            (case_dir / destination).write_bytes(source.read_bytes())
        (case_dir / "weights.bin").write_bytes(b"weights")
        (case_dir / "input.bin").write_bytes(b"input")
        files = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in case_dir.iterdir()
        }
        case = {
            "name": "case", "output_kind": "logits", "simulator_status": "passed",
            "output": {"elements": 2, "bytes": 8},
            "gmem_loads": [
                {"file": "weights.bin", "gpu_address": "0x30000000", "bytes": 7},
                {"file": "input.bin", "gpu_address": "0x30000040", "bytes": 5},
            ],
            "files": files,
        }
        self.write_bundle(case_dir, case)
        self.assertEqual(verify(self.root), len(files))
        del case["files"]["simulator-verification.json"]
        self.write_bundle(case_dir, case)
        with self.assertRaisesRegex(ValueError, "lacks a reference check"):
            verify(self.root)

    def test_package_uses_verified_reference(self):
        build_dir, export_dir = self.root / "build", self.root / "export"
        build_dir.mkdir()
        export_dir.mkdir()
        files = {
            "kernel.radiance.elf": b"gpu-elf",
            "kernel.soc.elf": b"soc-elf",
            "weights.bin": b"weights",
            "input.bin": b"input",
            "golden.bin": b"different-golden-output",
        }
        for name, data in files.items():
            directory = build_dir if name.endswith(".elf") else export_dir
            (directory / name).write_bytes(data)
        elf_hash = sha256(build_dir / "kernel.radiance.elf")
        weight_hash = sha256(export_dir / "weights.bin")
        input_hash = sha256(export_dir / "input.bin")
        build = {
            "model": "spectformer_vanilla", "device_elf_built": True,
            "runtime_input": True,
            "source_image_manifest": str(export_dir / "manifest.json"),
            "weights_sha256": weight_hash, "image_sha256": "image-digest",
            "input_address": 0x30000040, "weights_base_address": 0x30000000,
            "radiance_elf_sha256": elf_hash,
            "soc_elf_sha256": sha256(build_dir / "kernel.soc.elf"),
            "output_kind": "logits", "output_count": 2,
            "logits_offset": 0, "stages": 1,
            "logit_rtol": 0.005, "logit_atol": 0.001,
        }
        export = {
            "model": "spectformer_vanilla", "variant": "vanilla",
            "image_size": 224, "weight_provenance": "test",
            "weights_sha256": weight_hash, "image_sha256": "image-digest",
            "input_sha256": input_hash,
            "golden_sha256": sha256(export_dir / "golden.bin"),
            "base_address": 0x30000000,
            "entries": {"input.image": {"address": 0x30000040, "size_bytes": 5}},
            "num_classes": 2,
        }
        (build_dir / "manifest.json").write_text(json.dumps(build))
        (export_dir / "manifest.json").write_text(json.dumps(export))
        raw = json.loads(self.raw_path.read_text())
        raw.update(elf_sha256=elf_hash, weights_sha256=weight_hash,
                   input_sha256=input_hash)
        self.raw_path.write_text(json.dumps(raw))
        checked = json.loads(self.verified_path.read_text())
        checked.update(elf_sha256=elf_hash, weights_sha256=weight_hash,
                       input_sha256=input_hash,
                       run_result_sha256=sha256(self.raw_path))
        self.verified_path.write_text(json.dumps(checked))
        out = self.root / "handoff"
        with patch("kernels.spectformer.package_handoff.subprocess.check_output",
                   return_value="10001080 B v_arena\n"):
            record = package("case", build_dir, export_dir,
                             self.verified_path, out, Path("nm"))
        self.assertEqual(record["simulator_status"], "passed")
        self.assertEqual(record["files"]["reference.bin"]["sha256"],
                         sha256(self.run_dir / "verified-reference.bin"))
        (out / "manifest.json").write_text(json.dumps({
            "schema": "spectformer.fpga.handoff.v1", "cases": [record]}))
        self.assertEqual(verify(out), len(record["files"]))

    def write_bundle(self, case_dir, case):
        (case_dir / "load-map.json").write_text(json.dumps(case))
        (self.root / "manifest.json").write_text(json.dumps({
            "schema": "spectformer.fpga.handoff.v1", "cases": [case]}))


if __name__ == "__main__":
    unittest.main()
