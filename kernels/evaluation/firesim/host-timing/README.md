# Host release-to-completion measurements

This is the first U250 measurement series that removes boot, ELF loading,
host input setup, and output readback from the reported interval. The RV64
host reads `rdcycle` immediately before releasing Muon reset and immediately
after the all-finished register asserts. The interval still includes the
reset MMIO, launch, and completion polling. Its units are RV64 host cycles;
the host/GPU clock relation has not been calibrated. The FireSim driver's
whole-program target cycles remain a separate field.

The [plan](plan.csv) pins seven queue jobs, their timed and untimed ELF hashes,
and expected full-output checks. Jobs 1959–1960 are small instrumentation
checks. Jobs 1961–1964 and 1966 use 1,048,576-element STREAM Copy/Triad and the
original 256-pattern × 1024-repetition GPU STREAM Spatter Gather/Scatter
cases. Job 1964 uses an ordered Scatter schedule; job 1966 uses the standard
parallel schedule, which is deterministic on this collision-free input.
These are the first workload-sized timing candidates. Every
timed ELF was built with `--full-host-check --host-timing`; its untimed control
was built from the same input and full-readback setting. `device_image()`
reports byte-identical RV32 LOAD images for all seven pairs. This establishes
that the added `rdcycle` reads and UART line change the RV64 host only.
The planned GPU STREAM Gather and Scatter expected digests are
`b0f2661f6d70c8cf` and `6395beb9b9c4eb25`, respectively; both equal the
[pinned upstream serial Spatter goldens](../../../spatter/evaluation/upstream-golden-results.csv)
for those exact suite cases. The ordered and parallel Scatter plans have the
same expected final digest because this input has no destination collisions;
their schedules and timing may still differ.

All seven jobs use the existing `radiance_u250` image and HWDB SHA-256
`d4015580a0f7d58c0cd2606832d9579981feef98f4c5c1a29d25f7ba4f3c6bc4`.
The bitstream is identified in the [main FPGA report](../README.md); its
source metadata is dirty, so it is a diagnostic baseline rather than a
fully reproducible hardware revision. The older installed Muon compiler was
used with `MU_STACK_WORD_STRIDE=1` for both the runtime and these ELFs.
The local build archive is
`/scratch/agustin/projects/chipyard/radiance-host-timing-builds-20261006.tar.gz`
(SHA-256 `596b1b836f1000f08781fa8ad288a376336ca9b6a907b20dee09066cdfe64b51`).
It retains all seven timed/control ELF pairs, build logs, and manifests.

Run `../host_interval.py` on each completed job with the two ELF paths from
the plan and an output directory. It refuses a failed or incomplete UART,
missing timing line, changed job-local ELF, changed HWDB, or a changed RV32
device image. It archives the raw UART, queue request, build manifest, logs,
and binary hashes. Only a passing guest result with a complete output digest
can be called a measurement here.
`capture_series.py --out /scratch/agustin/projects/chipyard/radiance-host-interval-artifacts`
checks all planned jobs once. With `--watch`, it polls the queue and records
each completed job under that output directory; `results.csv` distinguishes
captured, pending, queue-failed, and capture-failed cases. It never changes
the source ELFs or the hardware image.

As of 2026-10-06 03:10 PDT, the seven jobs were queued behind other users of
the U250; no runtime number from this series has passed the gate yet. The
previous [comparison table](../comparison.csv) remains the measured RTL/model
result set. No LLM execution, cache counters, interconnect counters, GPU
issue utilization, or calibrated HBM bandwidth is implied by this plan.
