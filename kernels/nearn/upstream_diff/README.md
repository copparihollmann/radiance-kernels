# Exact 1M Vortex Nearn differential

This focused harness checks the exact 1,048,576-record fixture used by the
Radiance Nearn measurements against Vortex commit
`b447578d9f3bb116d658c614cf3407153f0536da`, path
`tests/opencl/nearn/kernel.cl`.

The pinned kernel is preserved byte-for-byte as base64 because the upstream
662-byte blob uses CRLF line endings. The generator verifies its SHA-256,
optionally verifies it directly against a local Vortex Git object, compiles it
to SPIR64 LLVM bitcode, and first attempts to execute it with an OpenCL 1.2
runtime. If no OpenCL platform exists, it records that fact and uses a native
CPU wrapper which embeds the unchanged kernel body. The wrapper only defines
the OpenCL qualifiers, `get_global_id`, and binary32 `sqrtf` mapping needed to
call that body.

The fallback runs both contraction-disabled and contraction-enabled builds;
the latter explicitly enables x86 FMA so it is not silently identical to the
generic strict build.
Every one of the 1,048,576 output words must be within one ULP of the exact
Radiance fixture oracle, and the stable top-five order by `(distance, original
index)` must match exactly. The evidence explicitly states that this fallback
does not prove behavior of a particular OpenCL device compiler or FPGA.

Run the full differential from the repository root:

```sh
python3 kernels/nearn/upstream_diff/upstream_diff.py \
  --output kernels/nearn/upstream_diff/evidence/nearn-upstream-1m-differential.json
python3 -m unittest kernels/nearn/upstream_diff/test_upstream_diff.py
```

Generated binaries and 12 MiB of fixture/output data stay under the ignored
`kernels/nearn/upstream_diff/.build/` directory.
