# Faithful scalar Radiance Hotspot3D

This directory ports the Vortex/Rodinia `hotspotOpt1` workload from pinned
Vortex revision `b447578d9f3bb116d658c614cf3407153f0536da`.  It performs the
same six-neighbor, boundary-clamped transient thermal update, including the
fixed ambient contribution, coefficient derivation, deterministic `srand(7)`
temperature/power fields, z-column sweep, and one device launch per timestep.

The Muon kernel is scalar and explicitly disables LLVM loop and SLP
vectorization.  It contains no VPU, vector ISA, Gemmini, or MX path.

## Pinned provenance

| Source | Git blob | SHA-256 |
|---|---|---|
| `tests/opencl/hotspot3D/main.cc` | `af729c56b76263cb3869b80787173363f4ff69ae` | `3954ddc7b8a841605c09d21fa8ef1f4cc07b1216fc9c3748a425d7b88907feec` |
| `tests/opencl/hotspot3D/kernel.cl` | `ec62ac18e872fdb8cd514f146626b6e4ed590ea4` | `f3823a9850bab4460cbaa3763f55c8d2ad2f42d44e13bc11399a43a0e2ef0e47` |
| `tests/opencl/hotspot3D/Makefile` | `8cabdd94d49fa92918e25b8ae90d9d952096be80` | `38c9f1d989f66e6e23c55aaf5760f4a8d9270bdf310874de6da9f1690c2acf64` |

`run.py` verifies all three objects directly from the pinned sparse checkout
before preparing a case.

## Workloads

- `canary`: 16 x 16 x 4, two timesteps.  These are exactly the pinned
  upstream defaults and are retained as a provenance/correctness canary.
- `meaningful`: 64 x 64 x 8, four timesteps.  This is 32,768 cells and 131,072
  full stencil updates, large enough to distribute thousands of complete
  cache lines across four SMs while remaining practical in the functional
  model.  It changes only upstream command-line dimensions and iteration
  count, not the equation, inputs, coefficients, or launch semantics.

## Oracle and floating-point contract

The generator computes every output cell for every timestep with an
independent full-field CPU oracle.  Unlike the device's optimized z-column
sweep, the oracle directly indexes all six clamped neighbors.  It explicitly
rounds every input conversion, multiply, and add to IEEE-754 binary32.  The
device disables contraction so its expression has the same visible rounding
points.

Cyclotron executes the Muon image but not the fused RV64 host, so it cannot
perform host-driven relaunches.  For functional testing only, the ELF embeds
the oracle input state for every timestep.  A high-bit model argument selects
one transition, and the harness runs and checks every transition separately
for each 1/2/4-SM artifact.  Ordinary arguments never use these snapshots and
retain the upstream host-driven ping-pong behavior; this mechanism does not
remove, repeat, or substitute useful work in a tested transition.

The complete RV64 checker accepts at most `max(32, 16 * timesteps)` ULP per
cell (32 ULP for the canary, 64 ULP for the meaningful case), rejects every
non-finite value, and checks every output.  Each update has seventeen rounded
products/additions; because all thermal terms are positive and do not cancel,
the first-order bound is under nine ULP per timestep.  The 16-ULP-per-timestep
allowance is therefore conservative and is substantially tighter than the pinned host's
`1e-3 * abs(reference) + 1e-3` tolerance.  Cyclotron currently also checks the
exact full-output FNV-64 digest; that is a stronger regression check, not a
claim that the portable semantic contract requires bit identity.

## Multi-SM ownership

One upstream work-item owns one `(x,y)` column and sweeps all z layers.  The
Radiance mapping partitions columns into disjoint contiguous groups of 16
float32 values (one 64-byte line).  Every supported plane has a multiple of 16
columns, so a line has one SM owner in every layer and every timestep.  The
generator audits 1/2/4-SM coverage, and the final timestep writes one owner tag
per column; the host and functional model validate the complete owner field
and four surrounding guards.

## Build and functional-model validation

```sh
for profile in canary meaningful; do
  for sm in 1 2 4; do
    python3 run.py --profile "$profile" --sms "$sm" \
      --loader logical --out "runs/logical/${profile}/${sm}sm"
  done
done

python3 -m unittest -v test_run.py

python3 model_check.py \
  --build-root runs/logical \
  --cyclotron /scratch/agustin/radiance-fpga-1a/src/radiance/cyclotron/target-native/release/cyclotron \
  --out runs/model-functional
```

Nothing here programs an FPGA, invokes FireSim, rebuilds a bitstream, edits a
service, or establishes FPGA performance/scaling evidence.

When the shell exports neither `RISCV64_TOOLCHAIN_PATH` nor `RISCV`, `run.py`
uses Agustin's existing private toolchain at
`/scratch/agustin/radiance-fpga-1a/toolchains/riscv-tools-prefix/riscv-tools`.
