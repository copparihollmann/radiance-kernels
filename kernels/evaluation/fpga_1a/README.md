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
python3 campaign.py plan \
  --build-manifest "$ROOT/runs/source-matrix-v4/manifest.json" \
  --asset-manifest "$ROOT/handoff/assets.json" \
  --assets "$ROOT/handoff" --out "$ROOT/runs/campaign-v1"
python3 campaign.py run --plan "$ROOT/runs/campaign-v1/plan.json" --parallel-targets
python3 campaign.py summarize --plan "$ROOT/runs/campaign-v1/plan.json" \
  --out "$ROOT/reports/campaign-v1.json"
```

Planning requires all four STREAM operations, five GPU-STREAM Spatter cases,
xRAGE Gather, ordered and parallel xRAGE Scatter, and no-VPU FP8 MX-GEMM for
both loader modes at every SM count. The default experiment is one warmup plus
three measured repetitions. Every input ELF, bitstream, model configuration,
runtime configuration, script, and parser is hashed before execution; changed
assets invalidate the plan. Copied simulator drivers are first-class hashed assets,
not untracked command-line inputs. Completed repetitions are resumable.

FireSim targets may provide a fixed `bdf` and a `reuse_argv`. With
`--parallel-targets`, each target runs serially on its own distinct FPGA while
different targets run concurrently. The first pending job for each target always
uses `run_argv` to program the copied image; only later jobs in that same process
use `reuse_argv` (normally Nicolas's launcher with `--no-program`). This avoids
flashing on every workload without trusting stale state from an earlier run.

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
