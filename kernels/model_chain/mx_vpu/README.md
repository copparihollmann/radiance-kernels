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
That infrastructure has passed compilation and elaboration checks only. No
complete HBM replay or Ramulator legality result is available. Offline replay
does not feed memory timing back into the running program and must not be
reported as whole-program latency. SIMT physical writeback addresses, combined
Muon/MX cache traffic and multi-SM capture remain unimplemented/unverified.
