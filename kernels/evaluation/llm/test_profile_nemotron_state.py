import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_nemotron_state import build, pinned_config, profile


class NemotronStateProfileTest(unittest.TestCase):
    def test_pinned_hybrid_state_and_context_growth(self):
        config, _ = pinned_config()
        short = profile(config, batch=1, context=128, ssm_bytes=2)
        long = profile(config, batch=1, context=8192, ssm_bytes=2)
        self.assertEqual((short["mamba_layers"], short["attention_layers"],
                          short["mlp_only_layers"]), (21, 4, 17))
        self.assertEqual(short["mamba_inner_width"], 96 * 80)
        self.assertEqual(short["mamba_conv_channels"], 96 * 80 + 2 * 8 * 128)
        self.assertEqual(short["initial_cache_inner_width"], 2 * 3136)
        self.assertNotEqual(short["initial_cache_inner_width"],
                            short["mamba_inner_width"])
        self.assertEqual(short["carried_conv_bf16_bytes"],
                         long["carried_conv_bf16_bytes"])
        self.assertEqual(short["carried_ssm_bytes"], long["carried_ssm_bytes"])
        self.assertEqual(long["attention_kv_bf16_bytes"] //
                         short["attention_kv_bf16_bytes"], 64)

    def test_state_dtype_changes_only_ssm(self):
        config, _ = pinned_config()
        narrow = profile(config, 1, 128, 2)
        wide = profile(config, 1, 128, 4)
        self.assertEqual(wide["carried_ssm_bytes"], 2 * narrow["carried_ssm_bytes"])
        self.assertEqual(wide["attention_kv_bf16_bytes"],
                         narrow["attention_kv_bf16_bytes"])
        self.assertEqual(wide["carried_conv_bf16_bytes"],
                         narrow["carried_conv_bf16_bytes"])
        report = build()
        self.assertEqual(report["status"], "analytical_capacity_only")
        self.assertEqual(report["pinned_attention_cache_handoff"],
                         "missing_in_block_call")


if __name__ == "__main__":
    unittest.main()
