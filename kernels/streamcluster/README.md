# Scalar StreamCluster `pgain` for Radiance

This directory is a bounded, correctness-first port of the Vortex Rodinia
StreamCluster test pinned at revision
`b447578d9f3bb116d658c614cf3407153f0536da`. The immutable authorities are:

- `tests/opencl/streamcluster/kernel.cl`, Git blob
  `be10036d449152cd6523b85bf7c44583abd7d9bb`, SHA-256
  `8b29f8baefd367e250ed80dab6441c365c7924a9aa1db1dd8ea2948a1740b097`.
- `tests/opencl/streamcluster/main.cc`, Git blob
  `97039e1c5bb3c52136cb342ddfed7d2176e6c6ca`, SHA-256
  `ea6c8d8acc2cb1017f73dcf74eb95dbc261c5dd1672b916e92c82809ee57fedc`.
- `tests/opencl/streamcluster/Makefile`, Git blob
  `1471bde239819f5fe53b89ed62f0d60cb829beed`, SHA-256
  `accfb9fcf8d13e8b4924235c7929144846a372d3c044c483aa85dc61b9297e2a`.

Preparation reads those objects from the local read-only reference repository
and fails unless its HEAD, blob IDs, and content hashes all match. The official
generated inputs and outputs also have pinned hashes, so a changed C-random
sequence, point ABI, equation, or accumulation order fails closed.

## Exact official contract

The default is `num=128`, `dim=32`, local size 16, `srand(9)` using the glibc C
`rand()` sequence, centers at every eighth point (`K=16`), and candidate
`x=64`. Coordinates are dimension-major. `Point` has the exact 12-byte
`float weight; int32_t assign; float cost` ABI. Each point is initialized to
its nearest open center using strict-less tie handling and ordered FP32
squared-distance accumulation.

The device preserves the two upstream pgain branches:

```text
if x_cost < current_cost:
    switch[point] = '1'
    work[point, K] = x_cost - current_cost
else:
    work[point, center_table[assign]] += current_cost - x_cost
```

The upstream zeroing kernels are fused into each owned point row. This retains
the complete zeroing-plus-pgain final semantics and accounts for both phases'
work while allowing NaN/byte poison to detect skipped points. It is therefore
labeled as fused full-sequence timing, not as the upstream host's standalone
`pgain_kernel` timing interval. The official candidate is itself an open center, so the
faithful official result has zero switch bytes; this is recorded rather than
artificially changing the workload.

## Physical-SM ownership

Physical `threadblock_id` selects one contiguous point interval. A point owns
its complete `(K+1)`-float work row. Internal boundaries are multiples of the
minimum point granule whose work rows occupy complete 64-byte cache lines. For
the official 4-SM case the intervals are `[0,32)`, `[32,64)`, `[64,96)`, and
`[96,128)`.

The compact 4-SM switch intervals are only 32 bytes and would false-share.
Instead, each SM receives a separate 64-byte-aligned switch segment. The host
maps those segments back to all 128 logical switch bytes, checks each byte,
and verifies every padding byte remains poisoned. Thus no output cache line
has multiple SM writers.

The kernel is scalar, disables FP contraction and compiler vectorization, and
uses no VPU, Gemmini, or MX path.

## Correctness and use

```sh
make test
python3 streamcluster.py --sms 4 --out runs/official-4sm --prepare-only
python3 streamcluster.py --sms 4 --out runs/official-4sm
python3 functional_smoke.py runs/official-4sm
```

The RV64 checker poisons every work float with NaN and all switch storage,
checks four 64-byte guards, compares every `128*(K+1)` work float using the
pinned upstream relative tolerance, requires the exact full-work digest,
compares all 128 logical switch bytes, requires the exact logical-switch
digest, and checks untouched switch padding.

Cyclotron starts the device ELF directly rather than running the fused RV64
checker, so its untouched switch padding retains BSS zero instead of the
checker's `0xA5` poison. Separate pinned physical digests cover those two
initialization contexts; all logical switch bytes are identical.

Larger `--num`/`--dim` settings are explicitly labeled derived. They retain
seed 9, centers every 8, `x=num/2`, dimension-major generation, nearest-center
initialization, and the exact pgain equation. Sizes that cannot be partitioned
without splitting an output cache line are rejected.

`functional_smoke.py` runs only the local Cyclotron functional model. This
directory does not invoke FireSim, program an FPGA, or access another user's
run directory.
