#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE = Path(__file__).with_name("model_campaign.py")
SPEC = importlib.util.spec_from_file_location("model_campaign", MODULE)
MC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MC)


class ModelCampaignTests(unittest.TestCase):
    def test_cluster_cycle_parser(self):
        line = ("Cyclotron: nightly_cluster_cycles cluster=3 cycles=17 "
                "start=41 end=58")
        self.assertEqual(MC.CLUSTER_CYCLES.findall(line),
                         [("3", "17", "41", "58")])

    def test_output_address_rejects_ordinary_elf(self):
        with self.assertRaises(ValueError):
            MC.output_address(Path(__import__("sys").executable))

    def test_resume_identity_fields_are_recorded(self):
        # run() uses these fields to reject a stale passing result after an
        # artifact rebuild or functional/timing-mode change.
        source = MODULE.read_text()
        self.assertIn('old.get("elf_sha256") == row["elf_sha256"]', source)
        self.assertIn('old.get("timing") == args.timing', source)

    def test_stream_config_is_correctness_gated(self):
        row = {"name": "stream--copy--2sm", "family": "stream", "sm_count": 2}
        result = {"stream_elements": 4, "output_element_count": 4,
                  "expected_digest": "1234"}
        original = MC.output_address
        MC.output_address = lambda _elf: 0x10004140
        try:
            with tempfile.TemporaryDirectory() as directory:
                config = MC.make_config(
                    row, result, Path("kernel.elf"), Path(directory), 99, False,
                )
        finally:
            MC.output_address = original
        self.assertIn("num_clusters = 2", config)
        self.assertIn("words = 4", config)
        self.assertIn('expected_fnv64 = "1234"', config)
        self.assertIn("0xc1d00001", config)
        self.assertIn("report_nightly_cycles = true", config)

    def test_stream_dot_checks_exact_sm_partials(self):
        row = {"name": "stream-dot--4sm", "family": "stream_dot",
               "sm_count": 4}
        result = {"stream_elements": 1048576, "output_element_count": 4,
                  "expected_digest": "abcd"}
        original = MC.output_address
        MC.output_address = lambda _elf: 0x10804140
        try:
            gmem, smem = MC.workload_checks(row, result, Path("kernel.elf"))
        finally:
            MC.output_address = original
        self.assertEqual(smem, [])
        self.assertIn("address = 0x10804140", gmem[0])
        self.assertIn("words = 4", gmem[0])
        self.assertIn('expected_fnv64 = "abcd"', gmem[0])

    def test_nearn_checks_complete_float_output(self):
        row = {"name": "nearn--records-1048576--4sm", "family": "nearn",
               "sm_count": 4}
        result = {"output_element_count": 1048576,
                  "expected_digest": "abcd"}
        original = MC.output_address
        MC.output_address = lambda _elf: 0x11804140
        try:
            gmem, smem = MC.workload_checks(row, result, Path("kernel.elf"))
        finally:
            MC.output_address = original
        self.assertEqual(smem, [])
        self.assertIn("address = 0x11804140", gmem[0])
        self.assertIn("words = 1048576", gmem[0])
        self.assertIn('expected_fnv64 = "abcd"', gmem[0])

    def test_parallel_scatter_uses_exact_destination_coverage(self):
        row = {"name": "xrage--scatter-parallel--4sm", "family": "xrage", "sm_count": 4}
        result = {"output_elements": 10, "expected_digest": None,
                  "exploratory_probe_index": 3, "expected_nonzero_outputs": 7}
        original = MC.output_address
        MC.output_address = lambda _elf: 0x10000040
        try:
            gmem, smem = MC.workload_checks(row, result, Path("kernel.elf"))
        finally:
            MC.output_address = original
        self.assertEqual(smem, [])
        self.assertIn("address = 0x10000040", gmem[0])
        self.assertIn("words = 20", gmem[0])
        self.assertIn("expected_nonzero_u64 = 7", gmem[0])

    def test_bf16_gemv_checks_every_packed_output_word(self):
        row = {"name": "gemv-bf16--4sm", "family": "gemv_bf16",
               "sm_count": 4}
        result = {"output_words": 8, "expected_digest": "feedface"}
        original = MC.output_address
        MC.output_address = lambda _elf: 0x10004140
        try:
            gmem, smem = MC.workload_checks(row, result, Path("kernel.elf"))
        finally:
            MC.output_address = original
        self.assertEqual(smem, [])
        self.assertIn("address = 0x10004140", gmem[0])
        self.assertIn("words = 8", gmem[0])
        self.assertIn('expected_fnv64 = "feedface"', gmem[0])

    def test_spmv_jds_checks_every_output_row(self):
        row = {"name": "spmv-jds--dubcova3--4sm", "family": "spmv_jds",
               "sm_count": 4}
        result = {"output_element_count": 146689,
                  "expected_digest": "0123456789abcdef"}
        original = MC.output_address
        MC.output_address = lambda _elf: 0x10404140
        try:
            gmem, smem = MC.workload_checks(row, result, Path("kernel.elf"))
        finally:
            MC.output_address = original
        self.assertEqual(smem, [])
        self.assertIn("address = 0x10404140", gmem[0])
        self.assertIn("words = 146689", gmem[0])
        self.assertIn('expected_fnv64 = "0123456789abcdef"', gmem[0])

    def test_spatter_output_padding_adjusts_check_address(self):
        row = {"name": "spatter--case-4-pad--4sm", "family": "spatter",
               "sm_count": 4}
        result = {"output_elements": 4, "expected_digest": "abcd",
                  "output_pad_cache_lines": 12}
        original = MC.output_address
        MC.output_address = lambda _elf: 0x10204140
        try:
            gmem, smem = MC.workload_checks(row, result, Path("kernel.elf"))
        finally:
            MC.output_address = original
        self.assertEqual(smem, [])
        self.assertIn("address = 0x10204440", gmem[0])

    def test_summary_groups_sm_tokens_at_end_or_in_middle(self):
        records = (
            ("stream-copy-1sm-robust", 1, 120),
            ("stream-copy-2sm-robust", 2, 70),
            ("stream-copy-4sm-robust", 4, 40),
            ("xrage--gather--1sm", 1, 200),
            ("xrage--gather--2sm", 2, 110),
            ("xrage--gather--4sm", 4, 60),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (name, sm_count, cycles) in enumerate(records):
                path = root / "jobs" / str(index) / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({
                    "status": "passed", "name": name,
                    "sm_count": sm_count, "kernel_cycles": cycles,
                }))
            MC.summarize(root)
            rows = {
                row["workload"]: row
                for row in json.loads((root / "summary.json").read_text())["workloads"]
            }
        self.assertEqual(set(rows), {"stream-copy-robust", "xrage--gather"})
        self.assertEqual(rows["stream-copy-robust"]["speedup_2_to_4"], 1.75)
        self.assertAlmostEqual(rows["xrage--gather"]["speedup_2_to_4"], 110 / 60)


if __name__ == "__main__":
    unittest.main()
