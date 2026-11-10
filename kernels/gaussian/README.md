# Gaussian elimination

This is a scalar, no-pivot port of the complete Vortex/Rodinia Gaussian
elimination workload pinned at Vortex revision
`b447578d9f3bb116d658c614cf3407153f0536da`.  The upstream directory is
spelled `tests/opencl/guassian`; `run.py` verifies the Git blob and SHA-256 of
`main.cc`, `kernel.cl`, `Makefile`, and `matrix4.txt` before generating data.

The old port tested only isolated Fan1/Fan2 traces for pivots 0--3.  This port
runs all `N-1` pivots in the required order:

1. all SMs execute `Fan1(t)`;
2. the host waits for every core in every SM;
3. all SMs execute `Fan2(t)`;
4. the host waits again before advancing to `t+1`.

These host-mediated relaunches are intentional device-wide barriers.  A local
Muon barrier cannot synchronize separate SMs.  GPU DRAM retains `A`, `b`, and
`m` across the inter-stage soft reset.

## Workloads and partition

- `canary`: the exact upstream 4x4 `matrix4.txt` fixture.
- `meaningful`: the pinned host's `srand(7)` generated input at 64x64, including
  its `+N` diagonal-dominance rule.  This is eight times the cubic elimination
  work of the upstream generated default (`-s 32`).

Rows are assigned in absolute 16-row (64-byte `float`) groups block-cyclically
across 1, 2, or 4 SMs.  Each active row is visited exactly once.  A complete
cache line of `b` and the ownership array belongs to one SM; because the
meaningful row stride is 64 floats, every `A` and `m` line is exclusive too.
The 4x4 canary necessarily uses only the owner of its one partial cache line.

Fan2 retains the upstream arithmetic and 2-D independence: work is flattened
over `(owned row, column)`, and only the pivot-column work-item updates `b`.
Floating-point contraction is disabled and the independent Python oracle uses
explicit binary32 multiply then subtract in the same mathematical phase.

## Correctness

`run.py` independently computes every Fan1/Fan2 transition, final multiplier
matrix, transformed `A`, transformed `b`, and binary32 back substitution.  The
host still reports the exhaustive per-word/ULP comparison against that oracle,
but that trajectory comparison is diagnostic: independent IEEE-754 execution
can accumulate beyond one ULP across 126 dependent stages even when the final
system is correct.

Acceptance does not use the generated oracle residual.  `actual_checker.h`
performs integer-only IEEE-754 binary32 back substitution directly on the
FPGA-produced transformed `A` and `b`.  It then evaluates that solution against
the untouched original `A` and `b`.  Every `A*x` product is exact; a signed
fixed-point accumulator tracks discarded tails with outward error, producing a
rigorous upper bound on `||A*x-b||inf`.  The checker proves that upper bound is
at most `tolerance*||b||inf`.  Since

`||b||inf <= ||A||inf*||x||inf + ||b||inf`,

this is stricter than the documented normwise backward-error requirement

`||A*x-b||inf / (||A||inf*||x||inf + ||b||inf)`.

The tolerance is `64*N*u`, where `u=2^-24` is binary32 unit roundoff.  This is
a conservative first-order `O(N*u)` allowance for no-pivot elimination and
back substitution; the generated diagonally-dominant workload is required to
fall below it.

The checker fails closed on nonfinite data, zero pivots, soft-float overflow,
fixed-point range failure, changed original-input digests, an uncertified
residual, incomplete launches, incorrect strict stage ownership, or any guard
change.  Complete transformed `A`, `b`, and `m` finiteness remains mandatory.
The host uses no floating-point instructions.  Device arithmetic, all 126
Fan1/Fan2 stages, useful work, and partitioning are unchanged.

Functional-model testing independently reloads each pre-stage oracle snapshot,
runs every Fan1/Fan2 stage for each 1/2/4-SM artifact, and checks exact FNV-64
digests of complete `A`, `b`, `m`, ownership, and guards.  Functional-model
cycles are useful only for regression comparison; they are not FPGA timing.

```sh
make test
python3 run.py --profile meaningful --sms 4 --out runs/build/meaningful/4sm
python3 model_check.py --build-root runs/build \
  --cyclotron /path/to/cyclotron --out runs/model-functional
```

The device ISA is scalar `rv32im_zfinx_zhinx`; there is no VPU, vector,
Gemmini, or MX path.
