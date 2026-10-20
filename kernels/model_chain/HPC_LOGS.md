# Collecting HPC run logs

Your FireSim setup and run commands stay with you. `hpc.py` only selects workloads
and parses saved logs/readbacks. It never launches or configures FireSim.

The primary HPC suite has 12 cases on each of 1, 2 and 4 SMs. Eleven additional
two-partition controls use one physical SM; select them with `--include-controls`.

```sh
python3 hpc.py list --sms 2
python3 hpc.py collect --sms 2 --logs saved-runs --output reports/2sm
```

Save files under each exact workload ID:

```text
saved-runs/
  stream--copy--2sm/
    console.log
    run-context.json
    readbacks/
      output.bin
```

Keep the full console log. `run-context.json` records the executed ELF hash,
workload ID, topology and actual clock/counter semantics as described in
`PROFILE.md`. `readbacks/` is optional; when present, all regions named in the
prepared dump plan are checked. GEMM also needs its per-SM mailbox files.

The script writes one row per workload in `results.csv` and `results.json`,
plus individual parsed profiles and correctness results. Report columns are
work cycles excluding setup/checking, IPC, SIMT issue utilization, MX utilization,
VPU utilization, and lane efficiency. Unsupported data is null, not zero. Existing
software model cycles appear in a separate column and are never substituted for
measurements from your run.

The current release binaries do not emit the new profiling records. Including
this parser does not enable counters or create measurements. Without records,
collection still reports available correctness results and marks profiling
unavailable. The optional counter helpers and their output format are in
`PROFILE.md`; instrumented workload integration remains pending. MX/VPU busy
counters and lane efficiency require an appropriate trace/counter source.
