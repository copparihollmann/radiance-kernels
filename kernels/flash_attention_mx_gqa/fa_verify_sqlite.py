#!/usr/bin/env python3
"""Verify FA output O by reconstructing it from the cyclotron/RTL trace-db (.sqlite `inst` table),
instead of the text `.out` [ISSUE] format fa_verify_out.py expects (our cyclotron/Verilator emit a
sqlite trace-db). SIMT global store: per-lane effective addr = rs1_data[lane], word = rs2_data[lane]
(a 32-bit word = 2 packed bf16). Collect words landing in [base, base+nbytes) -> reconstruct O ->
compare to a golden .npy (bf16 codes as uint16). Usage:
  fa_verify_sqlite.py <trace.sqlite> --base 0x40040000 --rows 64 --cols 128 --golden golden_O_flash_u16.npy
"""
import argparse, re, sqlite3, struct, subprocess, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("trace"); ap.add_argument("--base", default="0x40040000")
ap.add_argument("--rows", type=int, required=True); ap.add_argument("--cols", type=int, required=True)
ap.add_argument("--golden", required=True)
ap.add_argument("--elf", help="elf to identify store PCs (else any instr with in-range rs1 -- unsafe)")
a = ap.parse_args()
base = int(a.base, 16); nbytes = a.rows * a.cols * 2
obuf = bytearray(nbytes); seen = bytearray(nbytes)  # seen: which bytes were written

# Restrict to STORE instruction PCs (an add/other instr can hold an in-range value in rs1 and
# corrupt the reconstruction). Identify sw/sh PCs from the elf disassembly.
store_pcs = None
if a.elf:
    OBJ = "/scratch2/agustin/radiance-kernels/llvm/llvm-muon/bin/llvm-objdump"
    dis = subprocess.run([OBJ, "-d", a.elf], capture_output=True, text=True).stdout
    store_pcs = {int(m.group(1), 16) for ln in dis.splitlines()
                 if (m := re.match(r"^\s*([0-9a-f]+):\s+[0-9a-f ]+\s+s[whb](\.global)?\b", ln))}

cur = sqlite3.connect(f"file:{a.trace}?mode=ro", uri=True).cursor()
nstore = 0
q = "select pc, rs1_data, rs2_data from inst where has_rs1=1 and has_rs2=1"
for pc, r1, r2 in cur.execute(q):
    if store_pcs is not None and pc not in store_pcs:
        continue
    try:
        addrs = [int(x) for x in r1.split(",")]
        words = [int(x) & 0xFFFFFFFF for x in r2.split(",")]
    except (ValueError, AttributeError):
        continue
    for addr, word in zip(addrs, words):
        if base <= addr < base + nbytes - 1 and (addr - base) % 2 == 0:
            off = addr - base
            obuf[off:off+4 if off+4 <= nbytes else nbytes] = struct.pack("<I", word)[:min(4, nbytes-off)]
            for k in range(min(4, nbytes-off)):
                seen[off+k] = 1
            nstore += 1

O = np.frombuffer(bytes(obuf), dtype=np.uint16).reshape(a.rows, a.cols)
covered = sum(seen) // 2
gold = np.load(a.golden).astype(np.uint16).reshape(a.rows, a.cols)

def bf16_to_f32(u):
    return np.frombuffer(np.left_shift(u.astype(np.uint32), 16).tobytes(), dtype=np.float32).reshape(u.shape)

Of, Gf = bf16_to_f32(O), bf16_to_f32(gold)
num = np.linalg.norm(Of - Gf); den = np.linalg.norm(Gf)
rel = 100 * num / den if den else float("inf")
exact = 100 * np.mean(O == gold)
print(f"store words parsed into O range: {nstore}; cells covered: {covered}/{a.rows*a.cols}")
print(f"exact bf16-code match: {exact:.1f}%; max abs diff {np.max(np.abs(Of-Gf)):.4f}")
print(f"float Frobenius rel err vs golden: {rel:.4f}%")
