"""Reject incomplete or contradictory model-chain FireSim reports."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from collect_firesim_result import collect, parse_uart


VALID_UART = """model_chain_hbm: launch cluster 0
model_status=600d0000
gpu_start_cycle64=4294967290
gpu_end_cycle64=4294967310
kernel_cycles64=20
PASS
"""


class CollectFiresimResultTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.elf = self.root / "model.soc.elf"
        self.elf.write_bytes(b"test ELF identity")
        self.receipt = self.root / "model.soc.elf.json"
        self.source = self.root / "source-manifest.json"
        source_hash = hashlib.sha256(b"source ELF").hexdigest()
        self.receipt.write_text(json.dumps({
            "firesim_elf_sha256": hashlib.sha256(self.elf.read_bytes()).hexdigest(),
            "source_elf_sha256": source_hash}))
        self.source.write_text(json.dumps({
            "model": "tinyllama", "stage_limit": None, "stages": 1133,
            "scope": "full_dimension_checkpoint_decoder",
            "device_elf_built": True,
            "target_protocol": "nightly_hbm_printbuf_cluster0",
            "soc_elf_sha256": source_hash}))
        self.hwdb = self.root / "hwdb.yaml"
        self.hwdb.write_text("test HWDB identity\n")
        self.workload = self.root / "workload.json"
        self.workload.write_text("{}\n")
        self.uart = self.root / "uart.log"
        self.uart.write_text(VALID_UART)

    def collect(self, exit_code=0):
        return collect(model="tinyllama", sm_count=1, job_id="test-job",
                       bitstream_id="test-bitstream", elf=self.elf,
                       receipt=self.receipt, source_manifest=self.source,
                       hwdb_config=self.hwdb,
                       workload_config=self.workload, uart_log=self.uart,
                       firesim_exit_code=exit_code, counter_files=[])

    def test_valid_64_bit_wrap_and_source_hashes(self):
        result = self.collect()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["kernel_cycles64"], 20)
        self.assertEqual(result["gpu_start_cycle64"], 4294967290)
        self.assertEqual(result["gpu_end_cycle64"], 4294967310)

    def test_duplicate_and_partial_reports_are_rejected(self):
        for log in (VALID_UART + "PASS\n",
                    VALID_UART.replace("kernel_cycles64=20\n", ""),
                    VALID_UART + "model_status=600d0000\n",
                    VALID_UART.replace("kernel_cycles64=20", "kernel_cycles64=21"),
                    VALID_UART.replace("model_status=600d0000\n", "").replace(
                        "PASS\n", "PASS\nmodel_status=600d0000\n"),
                    VALID_UART.replace("model_status=600d0000",
                                       "model_status=00000001"),
                    VALID_UART.replace("PASS", "FAIL")):
            with self.subTest(log=log):
                self.assertEqual(parse_uart(log)["status"], "failed")

    def test_elf_mismatch_and_nonzero_firesim_exit_are_rejected(self):
        self.elf.write_bytes(b"different staged ELF")
        self.assertIn("staged ELF differs from its receipt",
                      self.collect()["problems"])
        self.elf.write_bytes(b"test ELF identity")
        self.assertIn("FireSim exited 124", self.collect(124)["problems"])

    def test_wrong_model_source_is_rejected(self):
        source = json.loads(self.source.read_text())
        source["stage_limit"] = 1
        self.source.write_text(json.dumps(source))
        self.assertIn("source manifest is not the requested full model HBM build",
                      self.collect()["problems"])


if __name__ == "__main__":
    unittest.main()
