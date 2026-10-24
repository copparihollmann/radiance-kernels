#!/usr/bin/env python3
import importlib.util
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


if __name__ == "__main__":
    unittest.main()
