"""Reject incomplete or contradictory model-chain FireSim reports."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from collect_firesim_result import check_source, collect, parse_uart, parse_uart_batch2


VALID_UART = """model_chain_hbm: launch cluster 0
model_status=600d0000
gpu_start_cycle64=4294967290
gpu_end_cycle64=4294967310
kernel_cycles64=20
PASS
"""

VALID_BATCH_UART = """model_chain_hbm: launch clusters=2
cluster_status=0:0x600d0000
cluster=0
gpu_start_cycle64=100
gpu_end_cycle64=140
kernel_cycles64=40
cluster_status=1:0x600d0000
cluster=1
gpu_start_cycle64=110
gpu_end_cycle64=165
kernel_cycles64=55
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
        source = json.loads((Path(__file__).resolve().parent /
                             "evaluation/tinyllama-2g-hbm-build.json").read_text())
        source["soc_elf_sha256"] = source_hash
        self.source.write_text(json.dumps(source))
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
        self.assertEqual(parse_uart(VALID_UART.replace(
            "model_status=600d0000", "model_status=0x600d0000"))["status"], "passed")

    def test_unsupported_four_sm_report_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "only one-SM and two-SM"):
            collect(model="tinyllama", sm_count=4, job_id="test-four",
                    bitstream_id="test-four-sm", elf=self.elf,
                    receipt=self.receipt, source_manifest=self.source,
                    hwdb_config=self.hwdb, workload_config=self.workload,
                    uart_log=self.uart, firesim_exit_code=0,
                    counter_files=[])

    def test_two_cluster_report_binds_both_statuses_and_native_check(self):
        source = json.loads(self.source.read_text())
        source.update({"batch_clusters": 2, "hbm_launch_clusters": 2,
                       "target_protocol": "nightly_hbm_printbuf_clusters",
                       "device_check_all_stages": False,
                       "verified_tensors": source["verified_tensors"][:135],
                       "hbm_verifier_files": 0})
        self.source.write_text(json.dumps(source))
        native = self.root / "native-result.json"
        native.write_text(json.dumps({
            "status": "passed", "replicas": 2,
            "stages_per_replica": source["stages"],
            "device_elf_sha256": source["radiance_elf_sha256"],
            "weight_image_sha256": source["weight_image_sha256"],
            "checkpoint_sha256": source["checkpoint_sha256"],
        }))
        self.uart.write_text(VALID_BATCH_UART)

        def collect_two(native_path=native):
            return collect(model="tinyllama", sm_count=2, job_id="test-two",
                           bitstream_id="test-two-sm", elf=self.elf,
                           receipt=self.receipt, source_manifest=self.source,
                           hwdb_config=self.hwdb, workload_config=self.workload,
                           uart_log=self.uart, firesim_exit_code=0,
                           counter_files=[], native_result=native_path)

        result = collect_two()
        self.assertEqual(result["status"], "passed")
        self.assertIsNone(result["kernel_cycles64"])
        self.assertEqual(result["cluster_timings"][1]["kernel_cycles64"], 55)
        self.assertIn("two-cluster native check receipt is required",
                      collect_two(None)["problems"])
        for uart in (VALID_BATCH_UART.replace("cluster_status=1:0x600d0000",
                                              "cluster_status=1:0x00000001"),
                     VALID_BATCH_UART.replace("cluster=1\n", ""),
                     VALID_BATCH_UART.replace("kernel_cycles64=55", "kernel_cycles64=56"),
                     VALID_BATCH_UART + "cluster_status=1:0x600d0000\n"):
            with self.subTest(uart=uart):
                self.assertEqual(parse_uart_batch2(uart)["status"], "failed")

    def test_deepseek_two_cluster_source_scope(self):
        path = (Path(__file__).resolve().parent /
                "evaluation/deepseek-2g-hbm-build.json")
        source = json.loads(path.read_text())
        source.update({"batch_clusters": 2, "hbm_launch_clusters": 2,
                       "target_protocol": "nightly_hbm_printbuf_clusters",
                       "device_check_all_stages": False,
                       "verified_tensors": source["verified_tensors"][:171],
                       "hbm_verifier_files": 0})
        self.assertEqual(check_source("deepseek_r1_distill_qwen_1_5b", source, 2), [])
        source["reference_output_sha256"] = "0" * 64
        self.assertIn("two-cluster decoder build lacks its selected checks",
                      check_source("deepseek_r1_distill_qwen_1_5b", source, 2))

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
        self.assertIn("source manifest is not the pinned full model HBM build",
                      self.collect()["problems"])

    def test_partial_decoder_check_is_rejected(self):
        source = json.loads(self.source.read_text())
        source["device_check_all_stages"] = False
        self.source.write_text(json.dumps(source))
        self.assertIn("decoder build does not check every scheduled stage",
                      self.collect()["problems"])

    def test_smolvla_requires_final_action_reference(self):
        path = (Path(__file__).resolve().parent /
                "evaluation/smolvla-2g-hbm-build.json")
        source = json.loads(path.read_text())
        self.assertEqual(check_source("smolvla_base", source), [])
        source["output_validation"] = "finite_range_only"
        self.assertIn("SmolVLA build lacks its final-action reference check",
                      check_source("smolvla_base", source))


if __name__ == "__main__":
    unittest.main()
