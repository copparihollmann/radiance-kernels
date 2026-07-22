#!/usr/bin/env python3
"""Reconstruct the kernel's attention output O from a CYCLOTRON trace-db (`--gen-trace true`)
and compare against a golden .npy (bf16 codes as uint16).

Cyclotron records every GMEM access in the `dmem` table (columns: store, address, size,
data) with the DEVICE address (0x40.......), which is exactly where the kernel's finalize_O
SIMT stores land -- so reconstruction just needs the store rows in [base, base+nbytes).
(The older inst-based fa_verify_sqlite.py parsed rs1/rs2 register values, which do NOT hold
the full effective address for SIMT global stores; use this dmem-based path for cyclotron.)

Usage:
  fa_verify_dmem.py <trace.sqlite> --base 0x40040000 --rows 512 --cols 64 \
                    --golden golden_O_gqa_mx_u16.npy
"""
import argparse, sqlite3, struct
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--base", default="0x40040000")
ap.add_argument("--rows", type=int, required=True)
ap.add_argument("--cols", type=int, required=True)
ap.add_argument("--golden", required=True)
a = ap.parse_args()

base = int(a.base, 16)
nb = a.rows * a.cols * 2
buf = bytearray(nb)
seen = bytearray(nb)

cur = sqlite3.connect(f"file:{a.trace}?mode=ro", uri=True).cursor()
nst = 0
for addr, size, data in cur.execute(
        "select address, size, data from dmem where store=1 and address>=? and address<?",
        (base, base + nb)):
    off = addr - base
    sz = min(int(size), nb - off)
    if sz <= 0:
        continue
    packed = struct.pack("<Q", int(data) & 0xFFFFFFFFFFFFFFFF)[:sz]
    buf[off:off + sz] = packed
    for k in range(sz):
        seen[off + k] = 1
    nst += 1

O = np.frombuffer(bytes(buf), dtype=np.uint16).reshape(a.rows, a.cols)
gold = np.load(a.golden).astype(np.uint16).reshape(a.rows, a.cols)
covered = sum(seen) // 2

def bf16_to_f32(u):
    return np.frombuffer(np.left_shift(u.astype(np.uint32), 16).tobytes(),
                         dtype=np.float32).reshape(u.shape)

Of, Gf = bf16_to_f32(O), bf16_to_f32(gold)
num = np.linalg.norm(Of - Gf); den = np.linalg.norm(Gf)
rel = 100 * num / den if den else float("inf")
exact = 100 * np.mean(O == gold)
print(f"dmem store rows in O range: {nst}; cells covered: {covered}/{a.rows*a.cols}")
print(f"exact bf16-code match: {exact:.1f}%; max abs diff {np.max(np.abs(Of-Gf)):.4f}")
print(f"float Frobenius rel err vs golden: {rel:.4f}%")
