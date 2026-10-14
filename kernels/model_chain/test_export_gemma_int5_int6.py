"""Capacity and packed-weight arithmetic controls for Gemma."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from checkpoint import pack_signed_bits, unpack_signed_bits
from export_gemma_int5_int6 import (HIGH_LIMIT, LOW_LIMIT, plan)


HERE = Path(__file__).resolve().parent


class PackedGemmaTest(unittest.TestCase):
    def test_full_image_uses_two_disjoint_regions(self):
        image = plan()
        low, high = image["segments"]
        self.assertEqual(image["layers"], 26)
        self.assertLess(low["gpu_base_address"] + low["image_size_bytes"], LOW_LIMIT)
        self.assertLess(high["gpu_base_address"] + high["image_size_bytes"], HIGH_LIMIT)
        items = {item["logical_name"]: item for item in image["parameters"]}
        self.assertEqual(items["embed_tokens"]["gpu_address"],
                         items["lm_head"]["gpu_address"])
        self.assertEqual(items["embed_tokens"]["scale_gpu_address"],
                         items["lm_head"]["scale_gpu_address"])
        self.assertEqual(sum(item["segment_index"] == 0
                             for item in image["parameters"]), 15)

    def test_bit_stream_roundtrip_including_incomplete_tail(self):
        for bits in (5, 6):
            for count in (1, 3, 4, 8, 13, 127):
                with self.subTest(bits=bits, count=count):
                    limit = (1 << (bits - 1)) - 1
                    values = np.random.default_rng(bits + count).integers(
                        -limit, limit + 1, count, dtype=np.int8)
                    packed = pack_signed_bits(values, bits)
                    np.testing.assert_array_equal(
                        unpack_signed_bits(packed, count, bits), values)

    def test_native_packed_kernels_match_known_values(self):
        five = pack_signed_bits(np.asarray([-15, 0, 15, 3, -4, 7, 1, -1],
                                           dtype=np.int8), 5)
        six = pack_signed_bits(np.asarray([-31, 0, 31, 7], dtype=np.int8), 6)
        table = pack_signed_bits(np.asarray([1] * 64 + [-1] * 64,
                                            dtype=np.int8), 5)
        def initializer(values):
            return ", ".join(str(int(value)) for value in values)
        source = f"""
#include <stdint.h>
#include "pipeline_packed.hpp"
int main() {{
  const uint8_t five[] = {{{initializer(five)}}};
  const int32_t expected_five[] = {{-15, 0, 15, 3, -4, 7, 1, -1}};
  for (uint32_t i = 0; i < 8; ++i)
    if (model_chain::signed_packed_weight(five, i, 5) != expected_five[i]) return 1;
  const uint8_t six[] = {{{initializer(six)}}};
  const int32_t expected_six[] = {{-31, 0, 31, 7}};
  for (uint32_t i = 0; i < 4; ++i)
    if (model_chain::signed_packed_weight(six, i, 6) != expected_six[i]) return 2;
  const float x[] = {{2}};
  const float scales[] = {{1, .5f, 2, 3}};
  float y[4] = {{0}};
  model_chain::linear_i6(x, six, scales, y, 1, 1, 4, 0, 1);
  if (y[0] != -62 || y[1] != 0 || y[2] != 124 || y[3] != 42) return 3;
  const uint8_t table[] = {{{initializer(table)}}};
  const float table_scales[] = {{2, 3}};
  const int32_t ids[] = {{1}};
  float embedded[64] = {{0}};
  model_chain::embedding_i5_group64(ids, table, table_scales,
                                   embedded, 1, 64, 1, 0, 1);
  for (float value : embedded) if (value != -3) return 4;
  float ones[64];
  for (float &value : ones) value = 1;
  float logits[2] = {{0}};
  model_chain::linear_i5_tied(ones, table, table_scales,
                              logits, 1, 64, 2, 0, 1);
  if (logits[0] != 128 || logits[1] != -192) return 5;
  return 0;
}}
"""
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as dirname:
            target = Path(dirname)
            (target / "check.cpp").write_text(source)
            subprocess.run(["g++", "-O2", "-std=c++20", "-I", str(HERE),
                            str(target / "check.cpp"), "-o", str(target / "check")],
                           check=True)
            subprocess.run([str(target / "check")], check=True)


if __name__ == "__main__":
    unittest.main()
