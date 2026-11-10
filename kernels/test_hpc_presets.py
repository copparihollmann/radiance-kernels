"""hpc_presets.py and hpc.mk must describe the same hardware."""

import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import hpc_presets

HERE = Path(__file__).resolve().parent
MK = HERE / "hpc.mk"


def mk_value(preset: str, key: str) -> str:
    """Ask make itself, so the test reads the file the build reads."""
    with tempfile.TemporaryDirectory() as d:
        probe = Path(d) / "probe.mk"
        probe.write_text(
            f"include {MK}\nprobe:\n\t@echo $({key})\n"
        )
        out = subprocess.check_output(
            ["make", "-f", str(probe), "probe", f"PRESET={preset}"],
            text=True, cwd=d,
        )
    return out.strip()


class PresetTableTest(unittest.TestCase):
    def test_every_preset_is_declared_in_both_places(self):
        declared = set(re.findall(r"^HPC_CLUSTERS_(\w+)\s*:=", MK.read_text(),
                                  re.M))
        self.assertEqual(declared, set(hpc_presets.PRESETS))

    def test_cluster_and_bank_counts_agree(self):
        for name, preset in hpc_presets.PRESETS.items():
            with self.subTest(preset=name):
                self.assertEqual(mk_value(name, "HPC_CLUSTERS"),
                                 str(preset.clusters))
                self.assertEqual(mk_value(name, "HPC_L1_BANKS"),
                                 str(preset.l1_banks))

    def test_drain_protocol_agrees(self):
        for key, value in hpc_presets.DRAIN.items():
            with self.subTest(knob=key):
                mk_key = key.replace("NIGHTLY_ROBUST_", "HPC_ROBUST_")
                self.assertEqual(
                    mk_value(hpc_presets.DEFAULT_PRESET, mk_key), str(value))

    def test_only_the_pipe_presets_vary_the_cluster_count(self):
        varying = {p.clusters for n, p in hpc_presets.PRESETS.items()
                   if n not in hpc_presets.SM_SCALING_PRESETS}
        self.assertEqual(varying, {2})

    def test_an_unknown_preset_is_rejected_by_make(self):
        with self.assertRaises(subprocess.CalledProcessError):
            mk_value("9sm_imaginary", "HPC_CLUSTERS")


if __name__ == "__main__":
    unittest.main()
