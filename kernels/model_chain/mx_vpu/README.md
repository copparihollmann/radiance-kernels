# Rakanic model backend and physical AXI trace preparation

The selected software target is **Rakanic/cyclotron base `26ab403` with its
unmodified libgemmini `d5e5e8e`**, `mode="both"`, `preset="radiance"`, and no
Gemmini timing overrides. Its VPU scheduler falls back to one unit when the
preset's unit count is zero. Optional two-unit timing profiles in the hardware
checkpoint are separate experiments and are not the selected target.

`compile_decoder.py --execution-backend rakanic_mx_vpu` connects E4M3 matrix
operations and BF16 residual/bias additions to TinyLlama, DeepSeek and Gemma's
actual graph dependencies and greedy loops. `--mx-linear-policy all` accelerates
every linear; `q_proj` keeps other projections on Muon as an explicit FP32
fallback policy. Norm, attention and remaining operations retain their existing
implementations. The manifest lists every stage's backend and the matrix MACs
covered. Compact INT4/5/6/8 and FP16 storage, including tied embeddings, is read
and converted a tile at a time, without creating a full FP8 checkpoint image.

`compile_smolvla.py` exposes the same backend for a selected `--projection-module`
(`action_in_proj` or `state_proj`) and its VPU bias addition. The action projection
remains inside all ten denoising iterations; the state projection remains in the
prefix. All other graph stages and dependencies are retained. This is a limited
projection subset, not acceleration of the entire SmolVLA matrix workload.

The shared scratchpad workspace uses row ranges starting at 0, 2048 and 4096,
and exponent words at bytes `0x18000` and `0x18100`. Only thread zero issues
commands. Exponents use shared memory to avoid a demonstrated private-array
indexing problem in the available Muon LLVM toolchain. MX output tiles are
16×16 with K blocks of 32. Operands use per-row/per-column power-of-two block
scales; MX computes with unit E8M0 scales, then Muon restores the block scales
and sums block results in FP32. VPU additions round inputs and outputs to BF16.

Device MX packing now uses the first warp's lanes, with thread zero retaining
MMIO ownership and the command/fence sequence. A shared-memory fence publishes
the packed tile before its command. The host runner retains serial packing.
E4M3 conversion rounds integer significands directly; normalization multiplies
by the inverse power of two. This removes floating-point divisions without
changing finite-input rounding. `test_quantization.py` checks an independent
nearest-value oracle on every finite BF16 value, every E4M3 midpoint and its
adjacent FP32 values, and one million random bit patterns. It also compares
normalization results bit for bit.

The retained reduced TinyLlama control (one layer, prefill one, no decode) passed
all embedded output checks on one and two actual clusters with the cooperative
packer. Its first MX projection fell from 1,417,606 to 112,485 source cycles
(12.6x); the complete control fell from 9,266,403 to 800,939 cycles (11.6x).
These are kernel improvements in the existing instruction/cycle model, not a
change to its clock calibration. They do not establish full-model completion.
Task artifacts `artifacts/e2e-hbm-audit/packing-v1/` and `packing-warp/` retain
the separate conversion-only and cooperative-packing experiments, ELFs,
commands, logs, progress journals, and drained HBM reports.

The compact helper layout subsequently keeps the FP8 converter, initializer,
and seven-argument VPU helper in shared device functions. This restores the
full Gemma ELF's fit below its unchanged weight image. Its one/two-SM reduced
controls pass, with a 115,846-cycle projection (12.24x) and 830,700-cycle full
control (11.15x). See `packing-compact/`, `packing-compact-performance.json`,
and `packing-full-v5/gemma_2_2b_it/placement-check.json`. Rejected overlapping
ELFs and the failed attempt to outline the larger-argument tiler remain in
separate artifacts; they are not qualification results.

Increasing decoder occupancy also required a runtime fix: `mu_schedule` now
joins its worker warps and fences memory before returning. The delayed-worker
device regression (`test_schedule_join.cpp`) originally observed 224 missing
writes; it passes on one and two SMs with the corrected runtime. An eight-warp
reduced graph passes both configurations as well. This enables testing higher
occupancy without letting later stages overtake workers from an earlier one.

`rebuild_e2e_backend.py` rebuilds affected full-model stage objects into a new
directory using retained link receipts. It checks generated sources against
their pinned archives and preserves original checkpoints, graphs and goldens.
Rebuilt native executables and device ELFs still require separate execution.
The helper accepts `--warps 2|4|8` with an explicit qualified `--runtime`
archive; it changes the launch occupancy while retaining graph bodies and
references, and records the resulting schedule and runtime hash.

`reference.py` generates format references independently in NumPy, including
the pinned model's product floor, truncation, accumulator precision ramp and
BF16 additions. It calls no C++ arithmetic reference. The graph generators
record drift separately from format-reference agreement. **A numerical match
to the format reference does not turn a failed original-output comparison into
a passing variant.** Reduced Gemma's all-linear trial changes a greedy token.
Both full SmolVLA reference trials exceed the original action tolerance: the
action-input projection fails 1089/1600 values (max error 0.0444224), and the
state projection fails 247/1600 (max error 0.0232617), at unchanged
`rtol=atol=0.001`. Those comparisons remain failed.

Native tests link the real C API model. The small operand controls match the
independent format reference exactly for partial tiles and all nine supported
weight layouts. A reduced decoder ELF passed one-cluster Cyclotron execution
with 52 MX loops, four VPU commands and `tohost=0`. A concurrent two-cluster
control passed with 26 MX loops/two VPU commands and `tohost=0` on each cluster.
Two-cluster execution uses the documented local launch/scheduling patch on top
of Rakanic's pin. These are reduced controls, not full-checkpoint device passes
or hardware timing measurements.

For native checks, set `RADIANCE_LIBGEMMINI_ROOT` to the pinned library checkout
and `RADIANCE_LIBGEMMINI_ARCHIVE` to the archive produced by Cyclotron's
`cargo build --features gemmini`. Keep `TMPDIR` inside the managed task:

```bash
python3 -m unittest discover -s kernels/model_chain/mx_vpu -p test_backend.py -v
python3 -m unittest discover -s kernels/model_chain -p test_mx_vpu_decoder.py -v
python3 kernels/model_chain/compile_decoder.py --model tinyllama --layers 1 \
  --prefill 2 --decode-steps 1 --generation greedy --device-check-all-stages \
  --execution-backend rakanic_mx_vpu --verify-native --out-root "$TASK_DIR/.tmp/new-run"
```

The native transport has a fixed callback mapping for its read-only unit
scales. Native command cycles omit Muon host arithmetic and are not whole-model
latency. Full checkpoint native runners preserve their existing image mappings
and link the model when the backend is selected. Full-checkpoint device
execution, original-output acceptance and HBM evaluation need their own receipts.

Full-depth generated C++ checks now pass the independent format reference for
TinyLlama (1133 stages), DeepSeek (1691), Gemma (1496), and SmolVLA (3673 executed
stages, 1600 final action values checked). The three decoder fixtures use
`q_proj` and two greedy decode steps; all six tokens match their original
checkpoint references. SmolVLA's action-input projection variant still fails
the original action comparison above. Its format-reference pass is separate.
These are host executions of the generated schedule linked to the real C API,
not full checkpoint Cyclotron ISA or RTL passes.

`fp32_reference.py` defines the fallback reduction and polynomial operation
order independently in NumPy. FP32 BLAS/libm differences can cross FP8/BF16
rounding boundaries, so this variant disables FP contraction in device/native
builds. An earlier full TinyLlama comparison failed before that reference-order
correction; no tolerance was changed. `run_full_models.py` serializes reference
generation, native checks and optional device builds and saves each command,
log hash and failure. Device builds alone do not establish execution.

The pinned hardware remains blocked: Radiance forces external shared scratchpad
memory, while Gemmini's VPU requires internal fixed-latency SRAM. See Chipyard's
`docs/radiance-mx-vpu-hbm/README.md` for the failing elaboration. This does not
prevent software checks through the selected Rakanic integration.

## Physical AXI trace preparation

`axi_trace.py` consumes streaming JSONL (optionally gzip) and creates a new output
directory containing per-path gzipped CSV files and a hashed manifest. Example:

```json
{"cycle":7,"sequence":3,"source":"simt.l2_miss","operation":"read","address":0,"size_bytes":32,"address_space":"gpu_local","boundary":"dram"}
```

This is a schema example, not a captured SIMT request. Addresses must explicitly
be `gpu_local`, `system_unhashed`, or `system_physical`. Only the first two receive
the pinned Radiance GPU address hash. `axi_path` is optional; if supplied it must
match physical routing. The operation must be `read` or `write`. Events must be
observed at the DRAM boundary, with aligned whole 8-byte beats; pre-L2 DMA cannot
substitute for L2 miss traffic. Cycles must not decrease and global sequence must
strictly increase. The sequence and cycle must fit unsigned 64-bit values.

```bash
python3 kernels/model_chain/mx_vpu/axi_trace.py events.jsonl.gz new-output \
  --ratio-num 1 --ratio-den 1
TMPDIR="$TASK_DIR/.tmp" PYTHONDONTWRITEBYTECODE=1 \
  python3 -m unittest discover -s kernels/model_chain/mx_vpu -v
```

The ratio is **replay clock frequency / source clock frequency**, and must be
provided explicitly from the actual system. The example 1/1 is only a control
value. Conversion rounds issue cycles upward and splits at address-hash units,
memory regions, 4 KiB boundaries, and 256 beats. It preserves source order and
records fragment indices. Ports are host 0 and GPU slices 1–4, covering the
physical GPU range `[0x100000000, 0x180000000)`.

Memory use is bounded by one input line (16 KiB maximum), one event and five
compressed output streams. Failed conversion leaves no success manifest;
partial files may remain for diagnosis. Reusing an output directory is rejected.
Sources may use only letters, digits, periods, underscores and hyphens, up to
64 characters. Data payload is not captured.

The paired FireSim `AXI4TraceReplay` target streams one physical path into one
HBM channel. Its C++ reader enforces the same routing, order and burst bounds.
The captured MX L2-fill control has passed replay on all four GPU paths: 274
requests, 8768 bytes, and all request addresses, directions, lengths and order
match at the HBM scheduler. All 314 issued HBM commands pass pinned Ramulator
2.1 (`b30320bc938`) with zero legality violations. A separate synthetic control
exercises all five paths, reads/writes, backpressure, long AXI bursts, and
refresh; empty and deliberately illegal command traces are rejected. The
synthetic long-burst check validates transport and issued-command legality;
the pinned HBM scheduler issues one CAS per AXI transaction and has not been
validated for timing of transactions spanning multiple physical DRAM bursts.
See FireSim's `scripts/hbm-validation/run_axi_replay.py` and the hardware
checkpoint evidence for commands, complete runtime plusargs and failed runs.
Offline replay
does not feed memory timing back into the running program and must not be
reported as whole-program latency. SIMT physical writeback addresses, combined
Muon/MX cache traffic and multi-SM capture remain unimplemented/unverified.

## Full graph verification queue

`audit_stitched_models.py --models <saved-models> --out <new-report.json>` checks
complete graph lineage, scheduled stage order, generated sources, ELF binding,
and every loaded compact weight/input image hash. It covers decoder hidden-state
and KV-cache handoffs and SmolVLA's three camera prefixes and ten denoising steps.
This structural check does not establish device execution.

`collect_smolvla_stage_reference.py` captures the pinned upstream policy's leaf
linear and norm outputs with an independent MX/BF16 action projection. The saved
full fixture maps 1,930 module boundaries; this is not every graph operation.
`prepare_e2e_native.py` relinks the retained native stage objects against the
specified current C API archive and checks those boundaries at unchanged
atol=rtol=0.001. Native runs execute full generated C++ on the host. They do not
measure device ISA/HBM timing, and a final-format pass does not erase original
checkpoint drift or intermediate failures.

`run_e2e_queue.py --jobs <jobs.json> --out <new-results-dir> --tmpdir "$TMPDIR"`
executes retained commands serially, records hashes/PIDs/status/logs, and checks
complete outputs rather than accepting a zero exit alone. Jobs are explicit
native, ISA, HPC or nightly GEMM cases. Full ISA jobs have no wall-time cutoff;
cycle bounds stay in their Cyclotron configs. HPC checks compare entire output
hashes with saved independent oracle hashes. GEMM checks every BF16 output and
the GPU's mailbox verification reports using the original nightly criterion.
A dependency receipt reserves the simulator slot; build receipts gate relinked
ELFs. Progress journals record retired function entries, not stage completion.

`summarize_e2e.py --task "$TASK_DIR" [--watch]` writes the current verification
snapshot. Per-job receipts remain authoritative. A failed original-checkpoint
comparison remains failed even if the format reference passes. Full two-SM
model execution is separate from one-SM logical partition checks and from the
reduced synthetic two-SM controls.

The full-output checks exposed two multi-cluster issues: Cyclotron omitted the
cluster field of `mhartid`, and `mu_schedule` shared one global launch context.
The CSR now matches the Radiance RTL layout. The runtime now reserves one aligned
context per global core, so asynchronous clusters can pass different callbacks
and replica arguments. `relink_e2e_runtime.py` preserves the original ELFs and
relinks unchanged generated stage objects/reference data with the corrected
runtime, recording both ELF hashes and checking image placement again. It
recompiles only auto-removed intermediate objects from their retained sources.
