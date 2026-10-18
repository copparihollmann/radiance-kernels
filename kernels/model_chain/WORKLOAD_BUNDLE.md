# Portable workload handoffs

`workload_bundle.py` exports qualified device binaries, launch carriers, load
images, references and evidence into a relocatable, content-addressed bundle.
It does not infer FPGA qualification from simulator results.

Build from a local recipe (all source paths may be absolute):

```sh
python3 workload_bundle.py build --recipe recipe.json --output handoff
```

The recipe supplies `target`, `provenance`, `readme`, `evidence` (a label/path
map), and `workloads`. Each workload supplies `name`, `family`, `config` (the
actual qualified Cyclotron TOML), `device_elf`, `soc_elf`,
`software_qualification`, optional `sources` (label/path map), and `checks`.
Checks specify a relative readback `file`, logical GPU `address`, byte count,
and either `mode=sha256` with `expected_sha256`, or `reference`, `dtype` and
`mode=exact/close/gemm`. Close checks retain their explicit `atol`/`rtol`.
GEMM uses the original complete-output numerical limits and optional per-SM
`smem_reports`. The byte count of every load must match its qualified config.

The builder materializes `gmem_patches` into private copies of their owning
load images. It checks the old and replacement hashes before writing each
patch and records their provenance. Source checkpoint files are never changed.
This is necessary for prepared FP8 weights; copying only `gmem_load` would
silently omit the initialization overlays used by the qualified run.

On the receiving machine (Python 3.11+, NumPy for `check`):

```sh
python3 workload_bundle.py verify
python3 workload_bundle.py list
python3 workload_bundle.py prepare NAME --loader logical --output stage
python3 workload_bundle.py check NAME --results results
```

`prepare` produces a single RV64 boot ELF containing all GPU images, verified
ELF/image receipts, a complete readback plan and a FireSim bare-metal workload
JSON. `--loader direct-hashed` instead pre-scrambles GPU loads for direct
LoadMem; its hash geometry must match the actual bitstream. A pre-hashed image
must never pass through a bus loader that applies the hash again. The packer
preserves non-cache-line-sized input blobs without padding their logical size.

The FPGA operator must supply the matching bitstream/HWDB and a suitable
completion/readback adapter. The current frozen kernels park secondary cores,
so waiting only for all cores to finish is insufficient. Launch-carrier
structure checks do not verify hardware completion, cache visibility or VPU
RTL compatibility. Preserve these limitations in the handoff's README.

Test with `TMPDIR` set inside the managed task and `PYTHONPATH` set to the
repository root:

```sh
python3 -m unittest test_workload_bundle test_package_firesim
```

The controls cover relocation, corruption, checked FP8 overlays, missing
replicas, nonfinite outputs, complete GEMM data/mailboxes, and both logical
and hashed ELF image packaging.
