import csv
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mx_vpu.axi_trace import Event, GPU_BASE, GPU_SIZE, HASH, bursts, convert, physical, route


def event(**updates):
    values = dict(cycle=7, sequence=3, source="simt.l2_miss", operation="read",
                  address=0, size_bytes=32, address_space="gpu_local", boundary="dram")
    values.update(updates)
    return Event.from_json(values)


class TraceChecks(unittest.TestCase):
    def test_all_gpu_paths_and_hash_roundtrip(self):
        paths = set()
        for local in range(0, 4096, 32):
            address = physical(local, "gpu_local")
            self.assertEqual(HASH.inverse(address), GPU_BASE+local)
            paths.add(route(address))
        self.assertEqual(paths, {1, 2, 3, 4})

    def test_does_not_hash_physical_addresses_twice(self):
        address = HASH.forward(GPU_BASE+128)
        self.assertEqual(physical(address, "system_physical"), address)

    def test_hash_unit_splitting_preserves_every_byte(self):
        result = list(bursts(event(size_bytes=256), 12, 5))
        self.assertEqual(len(result), 8)
        self.assertEqual(sum(r["beats"]*8 for r in result), 256)
        self.assertTrue(all(r["cycle"] == 17 for r in result))
        for r in result:
            self.assertEqual(r["axi_path"], route(int(r["address"], 16)))
            self.assertEqual(r["fragment"], result.index(r))

    def test_physical_burst_does_not_cross_4k(self):
        result = list(bursts(event(address=0x80000ff8, size_bytes=32,
                                  address_space="system_physical"), 1, 1))
        self.assertEqual([r["beats"] for r in result], [1, 3])
        self.assertTrue(all(r["axi_path"] == 0 for r in result))

    def test_physical_burst_does_not_cross_region(self):
        result = list(bursts(event(address=GPU_BASE+GPU_SIZE//4-8, size_bytes=32,
                                  address_space="system_physical"), 1, 1))
        self.assertEqual([r["axi_path"] for r in result], [1, 2])

    def test_256_beat_limit(self):
        result = list(bursts(event(address=0x80000000, size_bytes=4096,
                                  address_space="system_physical"), 1, 1))
        self.assertEqual([r["beats"] for r in result], [256, 256])

    def test_address_range_and_recorded_route_are_checked(self):
        for address in (-8, GPU_SIZE):
            with self.assertRaises(ValueError):
                physical(address, "gpu_local")
        with self.assertRaises(ValueError):
            list(bursts(event(axi_path=0), 1, 1))
        with self.assertRaises(ValueError):
            list(bursts(event(address=GPU_SIZE-8, size_bytes=32), 1, 1))

    def test_rejects_fake_dram_events_and_invalid_fields(self):
        for update in (dict(boundary="dma"), dict(address_space="unknown"),
                       dict(size_bytes=0), dict(address=1), dict(operation="evict"),
                       dict(cycle=True), dict(sequence=-1), dict(size_bytes=7),
                       dict(sequence=1 << 64), dict(cycle=1 << 64),
                       dict(source="bad,source"), dict(source="x"*65)):
            with self.subTest(update=update), self.assertRaises(ValueError):
                event(**update)

    def test_clock_overflow_and_zero_ratio(self):
        for numerator, denominator in ((0, 1), (1, 0), (-1, 1)):
            with self.assertRaises(ValueError):
                list(bursts(event(), numerator, denominator))
        with self.assertRaises(ValueError):
            list(bursts(event(cycle=1 << 63), 4, 1))

    def test_streamed_conversion_and_failure_has_no_pass_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"events.jsonl"
            first = event().__dict__
            second = event(cycle=8, sequence=4, operation="write", address=32).__dict__
            source.write_text(json.dumps(first)+"\n"+json.dumps(second)+"\n")
            result = convert(source, root/"out", 1, 1)
            self.assertEqual(result["input_records"], 2)
            self.assertEqual(result["transferred_bytes"], 64)
            self.assertFalse(result["whole_program_latency_valid"])
            rows = []
            for path in (root/"out").glob("*.csv.gz"):
                with gzip.open(path, "rt") as stream:
                    rows += list(csv.DictReader(stream))
            self.assertEqual(sorted(int(r["sequence"]) for r in rows), [3, 4])
            self.assertEqual(sorted(int(r["write"]) for r in rows), [0, 1])
            source.write_text(json.dumps(second)+"\n"+json.dumps(first)+"\n")
            with self.assertRaisesRegex(ValueError, "input line 2"):
                convert(source, root/"failed", 1, 1)
            self.assertFalse((root/"failed/manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
