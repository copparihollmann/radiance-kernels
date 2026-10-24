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
the copied registry; `--bdf 1a:00.0` selects the reserved card; and the final
positional path is the workload ELF. For example:
`run.sh -b 2SM_HBM --bdf 1a:00.0 ELF`. Registry-driven plans retain and hash
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
