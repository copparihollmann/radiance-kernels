#!/usr/bin/env python3
"""Decode where each linked STREAM array lands in DDR3 bank/rank space.

This is the measurement that decides whether STREAM's row-buffer thrashing is a
layout artifact.  It is pure arithmetic over symbol addresses read out of a
linked device ELF -- it touches no hardware and no simulator.

The two maps it composes are both read out of the RTL, not assumed:

  1. GPU address hash (radiance AddressHashParams, applied by
     WithGPUAddressHash; the loader-side mirror is soc/scramble_gpu_elf.py, and
     kernels/common.mk:199-203 pins the parameters actually used:
     base 0x0, size 0x8000_0000, unit 32 B, slices 4).
     Inside the range an offset splits into a 32 B unit index v and a byte;
     with s = log2(slices), o = v >> s is the unit's index WITHIN its slice and
     c = (v & (slices-1)) ^ fold(o) selects the slice.  Consecutive 32 B units
     round-robin over the four slices, and each slice's units stay dense.
     So the slice-local byte offset is (o << 5) | byte -- i.e. a client-address
     distance of N bytes becomes a slice-local distance of N/4 bytes.

  2. DDR3 address map (midas DramCommon.scala:118-137, the default open-page
     8 KB assignment that FRFCFS8GBQuadRank runs with): bank = addr[15:13],
     rank = addr[17:16], row = addr[18+].  Each GPU L2 slice owns one
     contiguous 512 MiB DRAM channel region, and 512 MiB is aligned far above
     bit 17, so bits [17:13] of the slice-local offset ARE the bank and rank
     bits for that channel.

Two arrays streamed in lockstep conflict when, at the same element index, they
decode to the same rank AND the same bank -- then every read<->writeback
alternation is a forced PRE+ACT (tRP 14 + tRCD 14).  Different bank, or
different rank, and both rows stay open.

usage: stream_layout_bank_map.py --nm LLVM_NM --elf kernel.radiance.elf [--json OUT]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

# kernels/common.mk:200-203 (MU_ADDR_HASH_SIZE / _UNIT / _SLICES)
HASH_SIZE = 0x80000000
HASH_UNIT = 32
HASH_SLICES = 4
UNIT_BITS = HASH_UNIT.bit_length() - 1
SLICE_BITS = HASH_SLICES.bit_length() - 1

# midas DramCommon.scala:118-133, quad-rank DDR3
BANK_OFFSET, BANK_MASK = 13, 7
RANK_OFFSET, RANK_MASK = 16, 3

BANK_STRIDE_SLICE_LOCAL = 1 << BANK_OFFSET          # 8 KiB
RANK_STRIDE_SLICE_LOCAL = 1 << RANK_OFFSET          # 64 KiB
BANK_RANK_PERIOD_SLICE_LOCAL = 1 << (RANK_OFFSET + 2)  # 256 KiB
CLIENT_PER_SLICE_LOCAL = HASH_SLICES                # 4 client bytes per slice byte


def fold(o: int) -> int:
    acc = 0
    while o:
        acc ^= o & (HASH_SLICES - 1)
        o >>= SLICE_BITS
    return acc


def slice_index(client: int) -> int:
    v = client >> UNIT_BITS
    return (v & (HASH_SLICES - 1)) ^ fold(v >> SLICE_BITS)


def slice_local(client: int) -> int:
    """Byte offset inside the owning slice's contiguous DRAM channel region."""
    o = client >> (UNIT_BITS + SLICE_BITS)
    return (o << UNIT_BITS) | (client & (HASH_UNIT - 1))


def bank(client: int) -> int:
    return (slice_local(client) >> BANK_OFFSET) & BANK_MASK


def rank(client: int) -> int:
    return (slice_local(client) >> RANK_OFFSET) & RANK_MASK


def symbols(nm: Path, elf: Path) -> dict[str, int]:
    out = subprocess.check_output([str(nm), "-n", str(elf)], text=True)
    found = {}
    for line in out.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[2].startswith("stream_"):
            found[fields[2]] = int(fields[0], 16)
    return found


def collision_fraction(base_a: int, base_b: int, elements: int,
                       stride_elements: int = 16) -> float:
    """Fraction of sampled element indices at which the two arrays decode to the
    same rank AND the same bank.  Sampled once per 64 B line of float32, which is
    the granularity at which the two streams actually move together."""
    same = total = 0
    for e in range(0, elements, stride_elements):
        a, b = base_a + 4 * e, base_b + 4 * e
        total += 1
        if rank(a) == rank(b) and bank(a) == bank(b):
            same += 1
    return same / total if total else 0.0


def analyse(nm: Path, elf: Path) -> dict:
    found = symbols(nm, elf)
    arrays = {k: v for k, v in found.items()
              if k in ("stream_a", "stream_b", "stream_c")}
    if len(arrays) != 3:
        raise SystemExit(f"expected stream_a/b/c in {elf}, found {sorted(found)}")
    guard_before = found["stream_guard_before"]
    guard_after = found["stream_guard_after"]
    # The output array is the one in .bss, i.e. the one between the two guards.
    output = next(k for k, v in arrays.items() if guard_before < v < guard_after)
    elements = (guard_after - found[output]) // 4

    report = {
        "elf": str(elf),
        "elements": elements,
        "output_array": output,
        "arrays": {
            name: {
                "client_address": f"0x{addr:08x}",
                "slice_local_offset": f"0x{slice_local(addr):08x}",
                "bank": bank(addr),
                "rank": rank(addr),
                "slice_of_first_unit": slice_index(addr),
                "role": "output" if name == output else "input",
            }
            for name, addr in sorted(arrays.items(), key=lambda kv: kv[1])
        },
        "pairs": [],
        "worst_same_bank_and_rank_fraction": 0.0,
    }
    names = sorted(arrays, key=lambda n: arrays[n])
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            lo, hi = arrays[names[i]], arrays[names[j]]
            frac = collision_fraction(lo, hi, elements)
            report["pairs"].append({
                "pair": f"{names[i]}->{names[j]}",
                "client_delta": f"0x{hi - lo:x}",
                "client_delta_kib": (hi - lo) / 1024.0,
                "slice_local_delta": f"0x{slice_local(hi) - slice_local(lo):x}",
                "slice_local_delta_kib": (slice_local(hi) - slice_local(lo)) / 1024.0,
                "slice_local_delta_mod_bank_rank_period":
                    f"0x{(slice_local(hi) - slice_local(lo)) % BANK_RANK_PERIOD_SLICE_LOCAL:x}",
                "bank_delta": (bank(hi) - bank(lo)) % (BANK_MASK + 1),
                "rank_delta": (rank(hi) - rank(lo)) % (RANK_MASK + 1),
                "same_bank_and_rank_fraction": frac,
            })
            report["worst_same_bank_and_rank_fraction"] = max(
                report["worst_same_bank_and_rank_fraction"], frac)
    # Only the arrays the kernel actually streams matter; for Copy and Scale one
    # of the three is allocated but never touched.  The caller decides which
    # pairs to gate on, so both the per-pair and the worst-case numbers are here.
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nm", type=Path, required=True)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = analyse(args.nm.resolve(), args.elf.resolve())
    text = json.dumps(report, indent=2) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
