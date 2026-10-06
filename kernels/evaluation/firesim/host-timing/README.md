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

All six jobs use the existing `radiance_u250` image and HWDB SHA-256
`d4015580a0f7d58c0cd2606832d9579981feef98f4c5c1a29d25f7ba4f3c6bc4`.
The bitstream is identified in the [main FPGA report](../README.md); its
source metadata is dirty, so it is a diagnostic baseline rather than a
fully reproducible hardware revision. The older installed Muon compiler was
used with `MU_STACK_WORD_STRIDE=1` for both the runtime and these ELFs.

Run `../host_interval.py` on each completed job with the two ELF paths from
the plan and an output directory. It refuses a failed or incomplete UART,
missing timing line, changed job-local ELF, changed HWDB, or a changed RV32
device image. It archives the raw UART, queue request, build manifest, logs,
and binary hashes. Only a passing guest result with a complete output digest
can be called a measurement here.

As of 2026-10-06 03:10 PDT, the seven jobs were queued behind other users of
the U250; no runtime number from this series has passed the gate yet. The
previous [comparison table](../comparison.csv) remains the measured RTL/model
result set. No LLM execution, cache counters, interconnect counters, GPU
issue utilization, or calibrated HBM bandwidth is implied by this plan.
