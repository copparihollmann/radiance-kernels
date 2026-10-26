# Exact-fixture upstream Hotspot3D differential

This directory compares the exact measured Radiance `meaningful` Hotspot3D
fixture (64 x 64 x 8, four timesteps) with the original Vortex OpenCL source.
It is a CPU-only source-fidelity test. It does not invoke Cyclotron, FireSim,
an FPGA, a bitstream, or any service.

## Pinned inputs

- Radiance fixture/oracle base:
  `8bf256a5406e749a6342522e398b468d56c1b5db`
- Vortex revision:
  `b447578d9f3bb116d658c614cf3407153f0536da`
- Vortex source: `tests/opencl/hotspot3D/kernel.cl`
- Git blob: `ec62ac18e872fdb8cd514f146626b6e4ed590ea4`
- SHA-256: `f3823a9850bab4460cbaa3763f55c8d2ad2f42d44e13bc11399a43a0e2ef0e47`

The tracked `.cl` file is verified by byte count, Git-blob identity, SHA-256,
and (when the local pinned checkout exists) direct byte comparison. The
Vortex `main.cc` and `Makefile` objects are also verified at the same revision.

## Execution and numerical contract

The harness first tries to compile and run the byte-exact source through a
native OpenCL **CPU** device. If no CPU ICD/platform exists, that attempt is
recorded as unavailable. It then always creates a mechanical C wrapper whose
prefix only defines `__kernel`, `__global`, and two-dimensional
`get_global_id`; the original kernel body appears byte-for-byte exactly once.
The suffix drives the original one-work-item-per-column body and preserves its
z sweep, clamped boundaries, equation, term order, and one launch per step.

Two wrapper builds are required:

- strict: contraction disabled, and all 32,768 words must be bit-exact at
  every timestep;
- contracted: contraction enabled to exercise the material OpenCL evaluation
  choice, with no nonfinite values and at most `max(32, 16 * step)` ULP.

Thus each execution checks 131,072 outputs, not just the final field. The
contracted allowance is the existing Radiance conservative 16-ULP-per-step
bound with its 32-ULP minimum. The strict result is the direct operation-order
proof; the contracted result bounds compiler-permitted multiply-add fusion.

Every Python process, compiler, OpenCL attempt, and native runner inherits a
hard `RLIMIT_AS` ceiling of 4 GiB. Both native runners also establish the same
limit themselves.

## Reproduce

From this directory:

```sh
python3 upstream_diff.py \
  --build-dir .build/full \
  --output evidence/hotspot3d-upstream-differential.json
python3 -m unittest -v test_upstream_diff.py
sha256sum evidence/hotspot3d-upstream-differential.json
```

The generated JSON records every source and fixture identity, every per-step
comparison, compiler identities/flags, the OpenCL attempt, SPIR64 compilation,
the memory ceiling, and limitations. On a host without an OpenCL CPU platform,
the mechanical wrappers prove source-level equivalence but do not prove the
behavior of a particular OpenCL device compiler.
