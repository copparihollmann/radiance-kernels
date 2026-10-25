# Faithful scalar Pathfinder port

This directory ports Pathfinder from the pinned Vortex/Rodinia OpenCL fixture
at revision `b447578d9f3bb116d658c614cf3407153f0536da`. It preserves the integer
dynamic program exactly: row 0 seeds the path costs; for each later row and
column, the new cost is that wall value plus the minimum of the previous
row's left, same-column, and right costs. At either edge, the missing neighbor
is the same-column cost. Rows are advanced strictly in order, and correctness
means exact equality of every value in the complete final row.

## Pinned provenance

| Source | Git blob | SHA-256 |
|---|---|---|
| `tests/opencl/pathfinder/main.cc` | `a2b9d676969f75f10bc196544dd5c0a2d5aeba10` | `7fe6fb7a663b44f6673b98b05462c8afd061a2ee47739aaa61a5206cc65ff001` |
| `tests/opencl/pathfinder/kernel.cl` | `d78878978ddd2a6ffb4a9ce0e9275f47b2e086af` | `c9084e4db3236f60bb90de31fdbdacd21f5bb411442f0bf1e65eda09ace67149` |
| `tests/opencl/pathfinder/Makefile` | `0e11c09eb6a89be32c7d40a5cbbe0fe8d6b611ce` | `4189687963afc7ebaa645240486cc7bbd694406e8d8685642e460404e966b437` |

The upstream-sized canary is exactly `16` rows by `64` columns and reproduces
the pinned host's `srand(9)` followed by `rand()%10`. The upstream pyramid
height (`2`) and block size (`16`) describe its OpenCL schedule, not different
math.

## Multi-SM schedule and correctness

The complete final row is partitioned into contiguous groups of sixteen
`int32_t` values, exactly one 64-byte cache line. Every line belongs to one and
only one physical SM in 1-, 2-, and 4-SM builds. Since Radiance has no safe
cross-SM barrier inside one launch, each SM uses private ping-pong rows to
advance only the dependency cone required by its final-row interval. That
cone contracts by one column on each side after every phase. All scheduled
threads stride through every local seed, DP, and output interval. After the
seed and after every DP row, every warp uniformly fences its global scratch
stores and reaches local barrier ID `12`; the same synchronization publishes
the final outputs. ID `12` is reserved by this body and does not collide with
the scheduler (`0`), robust completion (`13` and `14`), or END (`15`).

Neighboring SMs deliberately recompute halo columns because there is no safe
cross-SM row barrier. This redundancy changes scheduling, not semantics: each
SM uses private intermediate scratch and final cache lines remain exclusive.
Every result records exact per-row intervals, total work, unique work,
redundant halo work, peak overlap, per-SM totals, and strided thread counts.
For the meaningful workload, every one of the 256 scheduled threads per SM
has useful work in every seed, DP, and output phase.

Exact meaningful-workload accounting is:

| SMs | Redundant seed loads | Redundant DP updates | Final-write overlap | Total useful iterations |
|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 1,052,672 |
| 2 | 510 | 64,770 | 0 | 1,117,952 |
| 4 | 1,530 | 194,310 | 0 | 1,248,512 |

`run.py` computes expected values with a separate full-row CPU DP. It also
models the contracting-cone implementation and refuses generation if the two
differ. The RV64 checker compares every final-row value exactly, verifies the
owner of every value, checks independent FNV-64 digests of both complete
arrays, and checks three surrounding guard lines. The Cyclotron functional
driver independently checks those same two full arrays and three guards.

The Muon kernel is scalar and is built with LLVM loop and SLP vectorization
disabled. It contains no VPU, vector ISA, Gemmini, or MX path.

## Workloads

The upstream-sized correctness canary is intentionally small:

```sh
for sm in 1 2 4; do
  python3 run.py --preset upstream-canary --sms "$sm" \
    --loader logical --out "runs/logical-canary/${sm}sm"
done
```

The documented meaningful workload is `256 × 4096` = `1,048,576` wall cells
(4 MiB of input), producing a complete 16 KiB final row. It retains the same
seed, value distribution, integer recurrence, boundary rule, phase order, and
exact checker:

```sh
for sm in 1 2 4; do
  python3 run.py --preset meaningful --sms "$sm" --loader logical \
    --out "runs/meaningful-parallel/${sm}sm"
done
```

Run focused tests and the functional/no-timing model on the canaries with:

```sh
python3 -m unittest -v test_pathfinder.py
python3 model_check.py \
  --build-root runs/logical-canary \
  --cyclotron /scratch/agustin/radiance-fpga-1a/src/radiance/cyclotron/target-native/release/cyclotron \
  --out runs/model-functional-canary
```

Functional-model cycles establish execution and complete correctness checks;
they are not DRAM/HBM or FPGA scaling measurements.

Nothing in this directory programs an FPGA, invokes FireSim, rebuilds a
bitstream, or establishes FPGA performance/scaling evidence.
