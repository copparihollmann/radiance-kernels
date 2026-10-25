# Faithful scalar B+Tree port

This directory ports both B+Tree operations from the pinned Vortex OpenCL
fixture at revision `b447578d9f3bb116d658c614cf3407153f0536da`:

- `findK`: exact point lookup, returning the referenced record payload or `-1`.
- `findRangeK`: exact inclusive range lookup, returning `(start, length)`.

The default `256` keys, `32` point queries, `16` range queries, order `16`,
range span `8`, payload offset `100000`, and one continuous `srand(23)` libc
query stream reproduce the upstream default fixture. This fixture is a
correctness canary only; its size is not performance or scaling evidence.

## Pinned provenance

| Source | Git blob | SHA-256 |
|---|---|---|
| `tests/opencl/btree/main.cc` | `a4a2d1e940691119720edb7115dd81cc17203230` | `b0de0a5a8ee00fe7d8612cfc38652314b8a19aa376f3a34e34d37d5cd538df52` |
| `tests/opencl/btree/kernel.cl` | `25b613fd2a0f55b59da6ccf06662a09752e12adf` | `24adee3ea370fe666a42a04db1bd6becab8b59102324d7ac6527a6dd6738b3a0` |
| `tests/opencl/btree/Makefile` | `b743ffc8207e27114d80ba28e3fbf0de6eb00113` | `960ffd255f1e8a757214be9aeaf603e5e42ce0d3d8ded214a9d9514877a8e98b` |

The pinned checkout is sparse, so tests read these paths directly from its Git
object database rather than depending on materialized files.

## Explicit ABI mapping

The OpenCL fixture uses ABI-sensitive C types. The Radiance wire format maps:

- OpenCL/C++ `int` to `int32_t`.
- OpenCL `long` / host `cl_long` traversal state to `uint32_t`, after generator
  bounds checks prove all node indices, counts, and heights fit.
- `knode::bool` plus its three trailing ABI padding bytes to one explicit
  `uint32_t` containing `0` or `1`.

The resulting node remains exactly 148 bytes with `indices` at byte 4, `keys`
at byte 72, the leaf flag at byte 140, and `num_keys` at byte 144. Compile-time
assertions enforce the layout in both host and device builds.

## Correctness and ownership

`run.py` independently computes every expected result with Python
`bisect_left`; the oracle never walks the generated tree. It also interprets
the flattened tree with the exact device traversal and rejects generation if
the two methods disagree.

Every device output is a 16-byte record containing semantic result fields plus
query/SM ownership tags. Four records occupy one 64-byte cache line. The 1, 2,
and 4 physical-SM builds use disjoint contiguous line-granule partitions, so
every point and range query is owned exactly once and no output line is split
between SMs. The RV64 checker validates every semantic and metadata field,
three surrounding guards, and independent FNV-64 digests of both complete
output arrays. The focused Cyclotron driver independently checks both complete
output digests and the exact contents of all three guard regions.

The Muon kernel is scalar and contains no VPU, vector ISA, Gemmini, or MX path.

## Build and functional check

Build the upstream-size logical canaries:

```sh
for sm in 1 2 4; do
  python3 run.py --sms "$sm" --loader logical --out "runs/logical/${sm}sm"
done
```

Run focused tests:

```sh
python3 -m unittest -v test_run.py
```

Run the three logical artifacts in the functional/no-timing model:

```sh
python3 model_check.py \
  --build-root runs/logical \
  --cyclotron /scratch/agustin/radiance-fpga-1a/src/radiance/cyclotron/target-native/release/cyclotron \
  --out runs/model-functional
```

Nothing in this directory programs an FPGA, invokes FireSim, rebuilds a
bitstream, or establishes performance/scaling evidence.
