# FPGA-1A HPC campaign

This adapter runs the 1/2/4-SM HPC ELF matrix with copied, prebuilt FireSim
bitstreams and DRAM/HBM models. It never invokes a hardware build. Keep the
handoff under the current user's scratch directory; do not point it at another
user's private tree.

The asset manifest is explicit because loader semantics are correctness
critical: `direct-hashed` ELFs are only paired with direct physical LoadMem
paths, while `logical` ELFs are paired with loaders that apply the RTL address
mapping. Each FireSim target names one bitstream, so every copied configuration
is represented and audited independently. Start from `assets.example.json`, but
replace its example commands with Nicolas's copied scripts and exact arguments.

```sh
ROOT=/scratch/agustin/radiance-fpga-1a
python3 campaign.py inventory --assets "$ROOT/handoff" \
  --out "$ROOT/reports/asset-inventory.json"
python3 make_single_fpga_assets.py \
  --input "$ROOT/artifacts/nicolas-fpga-assets-v1/assets.json" \
  --output "$ROOT/runs/campaign-single-1a/assets.json" --bdf 1a:00.0
python3 campaign.py plan \
  --build-manifest "$ROOT/runs/source-matrix-v4/manifest.json" \
  --asset-manifest "$ROOT/runs/campaign-single-1a/assets.json" \
  --assets "$ROOT/artifacts/nicolas-fpga-assets-v1" \
  --out "$ROOT/runs/campaign-single-1a"
python3 campaign.py run --plan "$ROOT/runs/campaign-single-1a/plan.json" \
  --host-lock /scratch/agustin/share/firesim-host.lock
python3 campaign.py summarize --plan "$ROOT/runs/campaign-single-1a/plan.json" \
  --out "$ROOT/reports/campaign-single-1a.json"
```

Planning requires all four STREAM operations, five GPU-STREAM Spatter cases,
xRAGE Gather, ordered and parallel xRAGE Scatter, and no-VPU FP8 MX-GEMM for
both loader modes at every SM count. The default experiment is one warmup plus
three measured repetitions. Every input ELF, bitstream, model configuration,
runtime configuration, script, and parser is hashed before execution; changed
assets invalidate the plan. Copied simulator drivers are first-class hashed assets,
not untracked command-line inputs. Completed repetitions are resumable.

The manifest/model adapters also accept correctness-gated optional families:
BabelStream-derived Dot, BF16 GEMV, Vortex-derived Nearn and Stencil3D, and
Vortex/Parboil JDS SpMV. These additions do not weaken the required primary
matrix. Each optional family must provide its own complete output oracle,
logical/direct 1/2/4-SM artifacts, and fail-closed admission audit before it is
included in a hardware plan.

FireSim targets provide one fixed `bdf`. FPGA target groups must be sequential:
Corigine programming disconnects the selected card and performs a host-wide PCIe
rescan, so `--parallel-targets` is rejected for FireSim plans. The campaign
manifest maps every configuration to the one reserved physical card
(`1a:00.0`). Every job uses `run_argv` and fully reprograms that card. Plans and
commands containing `--no-program` are rejected: repeated image reuse preceded
the observed XDMA/PCIe loss, while fresh programming restores the complete
driver lifecycle before each workload.

The run command also enforces host-wide quietness. Before each fixed-BDF job it
scans the live `/proc` process tree for any FireSim runner, FPGA programmer,
driver, or manager lifecycle command outside its own descendants. It refuses to
start if one exists. While the job is live it polls again four times per second;
if another user's FireSim activity appears, it terminates its own process group,
writes `safety-abort.json`, and stops the campaign as soon as it detects the
conflict. A safety abort is deliberately not recorded as a kernel failure, so
the incomplete job is reprogrammed and retried on the next quiet-host run. This
guard is required even when different BDFs are selected: the rescan is
host-wide, not card-local. It is a fail-closed backup, not a substitute for an
exclusive host reservation: a directly invoked programmer can begin its rescan
before another process has time to observe it.

For a cooperative reservation, pass `--host-lock PATH` to `campaign.py run`.
This argument is mandatory for every plan containing FireSim hardware targets.
`PATH` must be an existing neutral lock file that every FireSim launcher and
manager on the host has agreed to honor for its complete driver lifetime. The
campaign opens it read-only, takes a nonblocking exclusive `flock`, and holds
that lock until every target group exits. It fails closed if the file is absent
or busy and never changes the file contents or timestamps. Do not point this at
another user's private lock without their agreement. This mechanism prevents
the programming race only when every hardware entry point uses the same inode;
the process monitor remains a fallback for non-cooperating commands.

Use repeatable `--include-job JOB_ID` arguments to execute an exact subset of
a hash-verified plan, such as one reduced loader canary before expanding to a
whole configuration matrix. Unknown IDs fail closed. Every selected job uses
the complete programming form, including when resuming a partial campaign.

For Nicolas's launcher, `-b NAME` selects the paired archive and driver from
the copied registry; `--bdf 1a:00.0` selects the reserved card; and
`--fastloadmem ELF` selects the workload. For example:
`run.sh -b 2SM_HBM --bdf 1a:00.0 --fastloadmem ELF`. Registry-driven plans retain and hash
the explicit archive/driver paths, and planning rejects a `-b` name whose
expected filenames do not match those receipts.

The generic parser recognizes `PASS`, FireSim's `*** PASSED ***`, and
`kernel_cycles=N`. Set `require_per_sm_timing` to require exactly two valid
`cluster C core K: start=S end=E cycles=D` records per physical SM, with the full
expected cluster/core set and `D == E - S`. A target may additionally define
`parser` and `parser_argv`;
the parser should emit one JSON object on stdout. All job requests, logs,
per-run results, and the median/min/max summary stay under the chosen campaign
directory.

`model_campaign.py` drives only Cyclotron's public generic DRAM timing model.
It selects logical-loader ELFs, initializes the nightly launch mailboxes for
every physical SM, checks output digests (or exact distinct-destination
coverage for parallel overlapping scatter), checks GEMM's device verifier, and reports the
nightly START-to-END kernel span. It deliberately records `is_hbm: false`;
HBM results must come from Nicolas's copied model/configuration assets through
`campaign.py`.

## Destination cache-line ownership

Radiance SMs have private L1s with line-granularity writeback, so a 64-byte
line written by two SMs can lose or tear words even when every byte written is
individually correct. A digest over the result cannot see that: the line is
shared on every run and happens to hold the right bytes on the run you
measured. That is exactly how the xRAGE Scatter defect survived
(`reports/DEFECT-xrage-4sm-cache-line-2026-11-02.md`), and why manifest fields
such as `"partition": "disjoint-contiguous-64-byte-output-lines"` are an
assertion, not evidence.

Four scripts compute the invariant instead. Each builds an owner map over
*every* work item, maps each item to its real address, and then independently
collects the set of SMs that write each 64-byte line. `cache_line_exclusive`
and `coverage_exactly_once` are computed values in their output, never
literals. Each also has an unsafe mode that reproduces the schedule the repair
replaced and must still fail: a detector that cannot fail proves nothing.

| script | covers | unsafe mode |
| --- | --- | --- |
| `audit_dense_contiguous_line_ownership.py` | any dense, directly indexed output: STREAM, stencil3d, nearn, hotspot3d, gemv, pathfinder, btree, spmv-jds, dot inputs | `--schedule round-robin` |
| `audit_spatter_gather_output_line_ownership.py` | Spatter/xRAGE Gather and MultiGather dense output, all three compiled schedules, plus transfer coverage | `--schedule round-robin` |
| `audit_spatter_scatter_task_line_ownership.py` | the parallel-collision Scatter/GS/MultiScatter task partition, resolving each task through the pattern indirection | `--schedule round-robin` |
| `audit_xrage_destination_line_ownership.py` | the ordered destination-owner Scatter group list | `--boundary-mode rank` |

`audit_hpc_v2_line_ownership.py` runs all of them over the 21 HPC workloads in
`runs/hpc-v2-memory-models-dual-1a1b-v1/plan.json` at 1, 2 and 4 SMs, pairs each
with its unsafe control, and emits one matrix. It exits non-zero if any kernel
shares a line.

```sh
ROOT=/scratch/agustin/radiance-fpga-1a
python3 audit_hpc_v2_line_ownership.py \
  --xrage-suite "$ROOT/artifacts/xrage-inputs-v1/xrage-suite.json" \
  --xrage-destinations "$ROOT/artifacts/xrage-inputs-v1/xrage-scatter.u32le" \
  --out "$ROOT/runs/hpc-v2-line-ownership-audit-v1/matrix.json"
```

Two rows do not pass:

- `dot-1m-partials` is a **defect**. `kernels/stream_dot/kernel.cpp:63` writes
  `stream_dot_partial[block_id]` into the 16-byte array that
  `kernels/stream_dot/run.py` emits as `.zero 16`, so at 2 SMs SM0 writes bytes
  0-3 and SM1 bytes 4-7 of one 64-byte line, and at 4 SMs all four SMs write
  that same line. Unlike xRAGE this is one naturally aligned 32-bit store per
  SM, so there is no multi-store payload to tear; the exposure is line-granular
  writeback. The 1/2/4-SM FPGA runs all pass with every partial word checked
  exactly, which is evidence that this memory system tolerates the sharing --
  not that the invariant holds.
- `gemm-bf16-512` is **undetermined**, not safe. See the row's `missing` field.
