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
  --output "$ROOT/runs/campaign-single-1b/assets.json" --bdf 1b:00.0
python3 campaign.py plan \
  --build-manifest "$ROOT/runs/source-matrix-v4/manifest.json" \
  --asset-manifest "$ROOT/runs/campaign-single-1b/assets.json" \
  --assets "$ROOT/artifacts/nicolas-fpga-assets-v1" \
  --out "$ROOT/runs/campaign-single-1b"
python3 campaign.py run --plan "$ROOT/runs/campaign-single-1b/plan.json"
python3 campaign.py summarize --plan "$ROOT/runs/campaign-single-1b/plan.json" \
  --out "$ROOT/reports/campaign-single-1b.json"
```

Planning requires all four STREAM operations, five GPU-STREAM Spatter cases,
xRAGE Gather, ordered and parallel xRAGE Scatter, and no-VPU FP8 MX-GEMM for
both loader modes at every SM count. The default experiment is one warmup plus
three measured repetitions. Every input ELF, bitstream, model configuration,
runtime configuration, script, and parser is hashed before execution; changed
assets invalidate the plan. Copied simulator drivers are first-class hashed assets,
not untracked command-line inputs. Completed repetitions are resumable.

FireSim targets provide one fixed `bdf` and may provide a `reuse_argv`. FPGA
target groups must be sequential: Corigine programming disconnects the selected
card and performs a host-wide PCIe rescan, so `--parallel-targets` is rejected
for FireSim plans. The recommended manifest maps every configuration to one
physical card (`1b:00.0`). The first pending job for each configuration uses
`run_argv` and therefore reprograms that card; only later jobs in that same
process use `reuse_argv` with `--no-program`.

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

For Nicolas's launcher, `-b NAME` selects the paired archive and driver from
the copied registry; the final positional argument is the workload ELF. For
example, the programming form is `run.sh -b 2SM_HBM --bdf 1b:00.0 ELF`, while
the same-image reuse form adds `--no-program`. Registry-driven plans retain and
hash the explicit archive/driver paths, and planning rejects a `-b` name whose
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
