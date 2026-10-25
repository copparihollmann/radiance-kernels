# Scalar JDS SpMV for Radiance

This directory is a correctness-first, scalar, no-VPU port of Parboil's
`spmv_jds_naive` as pinned in Vortex revision
`b447578d9f3bb116d658c614cf3407153f0536da`.  The reference kernel is
`tests/opencl/spmv/kernel.cl`, SHA-256
`8902bc12280ff7474fa6ea2c5dcb421006e80a7641badeefb2095942d3430674`.

The device retains the upstream JDS equation and strictly increasing `k`
accumulation order:

```text
j = jds_ptr[k] + jds_row
sum = sum + data[j] * x[index[j]]
output[perm[jds_row]] = sum
```

Only the work ownership changes.  The generated inverse permutation maps
original output rows back to their JDS rows, letting each SM own contiguous,
complete groups of 16 float outputs (one 64-byte cache line).  Boundaries are
selected by cumulative real nonzeros, so 1/2/4-SM artifacts remain line-safe
without the severe load imbalance of an equal-row split.

The upstream converter gives zero padding an index of `-1`, causing an
undefined `x[-1]` load.  This port records the same zero padding term but uses
index zero.  All real matrix entries, group bounds, pointer arithmetic, and
accumulation positions are unchanged.

## Deterministic admission fixture

```sh
python3 -m unittest -v test_run.py
python3 run.py --fixture-rows 257 --fixture-seed 7 --sms 4 \
  --out runs/fixture-4sm --prepare-only
RISCV64_TOOLCHAIN_PATH=/path/to/riscv-tools make kernel.soc.elf
```

The host prefills every output with NaN, launches the kernel, checks both
64-byte guards, compares every output against the ordered float32 oracle using
the upstream absolute/relative tolerance, and requires the exact full-output
FNV-1a digest.

## Pinned MatrixMarket datasets

Large matrices are converted to binary build inputs; no matrix text is emitted
into C++ source.  If the pinned Vortex blobs are materialized as files, the
bundled Dubcova3 case can be prepared with:

```sh
python3 run.py \
  --matrix /path/to/Dubcova3.mtx \
  --vector /path/to/Dubcova3.vec \
  --gold /path/to/Dubcova3.gold \
  --dataset-name Dubcova3 --sms 4 \
  --out runs/dubcova3-4sm --prepare-only
```

The importer accepts the pinned real symmetric coordinate format, mirrors
off-diagonal entries, preserves source order within every row, converts values
to float32, and refuses the artifact unless the complete generated oracle also
passes the upstream gold tolerance.  `result.json` records reference hashes,
dataset hashes, generated-binary hashes, JDS dimensions, partition boundaries,
per-SM nonzero counts, arithmetic policy, and robust-completion settings.

No FPGA or FireSim action is performed by this directory.
