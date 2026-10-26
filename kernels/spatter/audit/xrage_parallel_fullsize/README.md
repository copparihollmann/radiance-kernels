# Full-size xRAGE parallel-Scatter audit

This directory reproduces and audits the full pinned xRAGE9 parallel-Scatter
artifacts derived from source commit
`8bf256a5406e749a6342522e398b468d56c1b5db`.

`rebuild_and_check.sh` performs only local compilation and Cyclotron execution.
It does **not** execute `run.sh`, FireSim, FPGA programming, or a campaign run.
It builds:

- direct-hashed 1/2/4-SM FPGA artifacts with the qualified post-20k completion
  policy; and
- logical 1/2/4-SM shadow artifacts for Cyclotron, which cannot model the
  direct-loader address hash.  The shadows retain the exact transfer list,
  task partition, device kernel, end drain, serial fence, three drain passes,
  and 20,000-cycle post-spin.  Only the direct-address transform and the
  FPGA-only spin-until-reset are disabled.

The script runs the Python and native focused tests, checks all 6,664,304
transfers against the pinned input, and runs every logical shadow under stock
Cyclotron.  A tiny `LD_PRELOAD` gate pauses Cyclotron immediately after its
full GMEM PASS line while its 4-GiB `FlatMemory` is live.  The runner snapshots
the complete output through `/proc/PID/mem`, recomputes Cyclotron's FNV and
coverage, then applies the same `parallel_scatter_winner.hpp` classifier to
every output.  It finally prepares a five-image FPGA-1A plan from the copied
Nicolas assets and invokes `audit.py` to validate the entire result fail-closed.

The FPGA plan is inert data.  Each planned command must be exactly of the form
`run.sh -b CONFIG --bdf 1a:00.0 --fastloadmem ELF`, wrapped only by `timeout`.
No reuse or `--no-program` command is accepted.

Typical invocation from the repository root:

```sh
kernels/spatter/audit/xrage_parallel_fullsize/rebuild_and_check.sh
```

The default output is the worktree-local ignored directory
`experiments/xrage-parallel-winner-fullsize-8bf256a-v1`.  Paths may be
overridden with the environment variables declared at the top of the script.
