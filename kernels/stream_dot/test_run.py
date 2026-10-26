"""Focused tests for the faithful BabelStream-derived Dot generator."""

import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

RUN_PATH = Path(__file__).resolve().parent / "run.py"
RUN_SPEC = importlib.util.spec_from_file_location("stream_dot_run", RUN_PATH)
assert RUN_SPEC is not None and RUN_SPEC.loader is not None
run = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(run)


class StreamDotRunTest(unittest.TestCase):
    def test_partition_covers_each_element_once(self):
        for sms in (1, 2, 4):
            ranges = [run.partition(1048592, sms, sm) for sm in range(sms)]
            self.assertEqual(ranges[0][0], 0)
            self.assertEqual(ranges[-1][1], 1048592)
            self.assertTrue(all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])))
            self.assertTrue(all(begin % 16 == 0 for begin, _ in ranges))

    def test_records_upstream_and_exact_partial_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run.prepare(256, Path(directory), sms=4,
                                 loader="logical", warps_per_core=2)
            self.assertEqual(result["reference_repository"], "UoB-HPC/BabelStream")
            self.assertEqual(result["reference_revision"], run.UPSTREAM_REVISION)
            self.assertIsNotNone(result["kernel_revision"])
            self.assertEqual(result["output_element_count"], 4)
            self.assertEqual(len(result["expected_partial_words"]), 4)
            self.assertEqual(result["correctness"],
                             "exact-device-order-partials-host-final-scalar-and-guards")
            config = (run.GENERATED / "config.h").read_text()
            self.assertIn("#define STREAM_DOT_EXPECTED_FINAL_WORD ", config)
            self.assertEqual(
                result["expected_final_word"],
                f"{run.float_word(run.expected_final_sum(run.expected_partials(256, 4, 2))):08x}",
            )
            self.assertLess(result["relative_error_vs_scalar_golden"], 1e-4)
            self.assertEqual(
                result["host_final_reduction"],
                "integer-ieee754-binary32-rne-positive-finite",
            )

    def test_host_final_reduction_is_integer_only(self):
        source = (RUN_PATH.parent / "host.cpp").read_text()
        self.assertIn("add_positive_finite_f32_rne", source)
        self.assertNotRegex(source, re.compile(r"\bfloat\b"))
        self.assertNotIn("FloatWord", source)

    def test_integer_binary32_adder_matches_native_rne(self):
        compiler = shutil.which("g++") or shutil.which("clang++")
        self.assertIsNotNone(compiler)
        program = r'''
#include <stdint.h>
#include <stdio.h>
#include "softfloat_positive_f32.h"

union WordFloat { uint32_t word; float value; };

static uint32_t native_add(uint32_t a, uint32_t b) {
  WordFloat left = {.word = a};
  WordFloat right = {.word = b};
  volatile float sum = left.value + right.value;
  WordFloat result = {.value = sum};
  return result.word;
}

int main() {
  uint32_t state = 0x6d2b79f5u;
  for (uint32_t i = 0; i < 200000u; ++i) {
    state = state * 1664525u + 1013904223u;
    const uint32_t a = (((state >> 24u) % 250u) + 1u) << 23u |
        (state & 0x007fffffu);
    state = state * 1664525u + 1013904223u;
    const uint32_t b = (((state >> 24u) % 250u) + 1u) << 23u |
        (state & 0x007fffffu);
    bool valid = true;
    const uint32_t actual =
        stream_dot_add_positive_finite_f32_rne(a, b, &valid);
    const uint32_t expected = native_add(a, b);
    if (!valid || actual != expected) {
      fprintf(stderr, "%08x + %08x: actual=%08x expected=%08x valid=%d\n",
              a, b, actual, expected, valid);
      return 1;
    }
  }
  return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "test.cpp"
            binary = root / "test"
            source.write_text(program)
            subprocess.run(
                [compiler, "-std=c++20", "-O2", "-fno-fast-math",
                 "-I", str(RUN_PATH.parent), str(source), "-o", str(binary)],
                check=True,
            )
            subprocess.run([str(binary)], check=True)

    def test_sm_count_changes_only_reduction_partition(self):
        one = run.expected_partials(1024, 1, 8)
        two = run.expected_partials(1024, 2, 8)
        four = run.expected_partials(1024, 4, 8)
        self.assertEqual((len(one), len(two), len(four)), (1, 2, 4))
        self.assertEqual(two[0], two[1])
        self.assertTrue(all(value == four[0] for value in four))

    def test_block_cyclic_chunks_cover_each_element_once(self):
        elements = 1048592
        for sms in (1, 2, 4):
            owners = [0] * elements
            chunk_elements = 2048
            chunks = (elements + chunk_elements - 1) // chunk_elements
            for sm in range(sms):
                for chunk in range(sm, chunks, sms):
                    begin = chunk * chunk_elements
                    end = min(elements, begin + chunk_elements)
                    for index in range(begin, end):
                        owners[index] += 1
            self.assertTrue(all(owner == 1 for owner in owners))

    def test_records_bank_staggered_block_cyclic_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run.prepare(
                8192, Path(directory), sms=4, warps_per_core=8,
                block_cyclic_chunk_elements=2048)
            self.assertEqual(result["dot_block_cyclic_chunk_elements"], 2048)
            self.assertEqual(result["dot_block_cyclic_chunk_bytes_per_input"], 8192)
            self.assertEqual(
                result["partition"],
                "disjoint-block-cyclic-cache-line-complete-input-chunks")
            self.assertIn("-blockcyclic2048--", result["name"])
            self.assertEqual(len(result["expected_partial_words"]), 4)
            self.assertLess(result["relative_error_vs_scalar_golden"], 1e-4)

    def test_rejects_non_cache_line_block_cyclic_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "multiple of 16"):
                run.prepare(256, Path(directory),
                            block_cyclic_chunk_elements=17)

    def test_requires_robust_prerequisites(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "requires robust_end_drain"):
                run.prepare(256, Path(directory), robust_spin_until_reset=True)
            with self.assertRaisesRegex(ValueError, "requires robust_end_drain"):
                run.prepare(256, Path(directory), robust_serial_fence=True)


if __name__ == "__main__":
    unittest.main()
