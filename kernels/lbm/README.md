# Radiance LBM

This directory is a bounded scalar port of the official default Vortex/
Parboil LBM workload. The read-only reference is Vortex revision
`b447578d9f3bb116d658c614cf3407153f0536da`, specifically
`tests/opencl/lbm`. The generator verifies the Git blob ID, SHA-256, and byte
count of the upstream Makefile, kernel, layout, macros, host initialization,
`32_32_8_ldc.of`, and `32_32_8_ldc.gold` before producing anything. It also
fails if the default ceases to be exactly one timestep, GATHER, or 32x32x8
with the pinned 40x32x12 padded allocation.

The port preserves the D3Q19 gather, obstacle bounce-back, ACCEL override,
collision equations, and source operation order. FP contraction and compiler
loop vectorization are disabled. The device code uses scalar FP32 only: no
VPU, vector ISA, Gemmini, or MX operations.

## Physical-SM ownership

The 8,192 active cells form 2,048 consecutive groups of four complete AoS
cells. Each group is 80 floats (320 bytes), exactly five 64-byte cache lines.
Because `SIZE_X` is divisible by four and every padded row is cache-line
aligned, groups never cross a row and no cache line is shared by two physical
`threadblock_id` owners. The 1/2/4-SM boundaries are respectively
`[0,2048]`, `[0,1024,2048]`, and `[0,512,1024,1536,2048]`.
`partition_audit.py` enumerates every group and every owned cache line, failing
on gaps, overlap, misalignment, or a changed five-line unit. Each preparation
writes the complete audit to `partition-audit.json`.

## Embedded data and correctness

`lbm.py` reconstructs the exact upstream initialization, including GATHER
placement, obstacle input, two-cell z margin, wall flags, and lid-driven
cavity ACCEL flags. It executes one ordered FP32 reference timestep and fails
unless its independently exported 3x8192 velocity words match the pinned
gold file bit-for-bit. The ELF embeds the full padded source grid, initialized
destination/halo, all 19 expected populations for every active cell, and the
official velocity gold.

The embedded destination already NaN-poisons all 155,648 active populations
while preserving initialized halo values, so model checks detect every missing
store. Before launch, the RV64 checker repeats that poison and installs two
64-byte guards.
After launch it checks every population exactly and verifies its complete
FNV-1a digest. It then independently derives all 24,576 velocity components
from the result using the upstream GATHER exporter order and applies the
upstream tolerance (`1e-4 * max(abs(gold))` absolute or strict 0.2% relative).
It also verifies the embedded oracle/gold digests, finiteness, guards, and
reports the actual population and velocity digests.

Cyclotron does not execute the fused RV64 checker, so `functional_smoke.py`
independently checks the FNV-1a digest of all 307,200 words in the complete
padded destination grid. This includes the active results, flags, and every
initialized halo/padding word; the expected digest comes from the same pinned
ordered model and is recorded in the artifact manifest.

## Software-only use

```sh
python3 lbm.py --sms 4 --out runs/4sm --prepare-only
python3 partition_audit.py --sms 4
python3 -m unittest -v test_lbm.py
python3 lbm.py --sms 1 --out runs/1sm
python3 functional_smoke.py runs/1sm
```

The builder uses `RISCV64_TOOLCHAIN_PATH` when set and otherwise discovers the
workspace-local toolchain. These commands only prepare/build software or run
the functional simulator. Nothing in this directory invokes FireSim, programs
an FPGA, or accesses another user's run directory.
