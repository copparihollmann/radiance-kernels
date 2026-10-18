# Worker stack stability across launches

The old C++ `mu_schedule_workers` allocated a 256-byte lane-interleaved frame,
called `mu_schedule_standalone`, then executed `tmc(0)`. Retirement prevented its
compiler-generated epilogue from restoring SP. A later wspawn restarted the worker
with that lowered SP, leaking another frame on each launch.

With 128 KiB reserved per warp, repeated launches eventually reached neighbouring
warps' saved state. The four-worker SmolVLA run stalled around launch 2560: core 0
waited at its stage-end barrier while core 1's manager had reached worker retirement.
A small unassisted repeated-launch control failed at launch 2561 without model
weights or accelerator arithmetic. This issue exists independently of assistance.

The worker entry is now naked assembly. It enables lanes and calls the separately
framed standalone routine, which restores its own SP before returning to the
retirement instruction. The entry itself owns no frame. Manager joining and
cross-core barriers remain in place.

`test_schedule_worker_stack.cpp` checks every participating lane's SP on every
launch. The old runtime detects a 256-byte difference in 96 worker lanes on the
second launch. The fixed runtime must retain every initial SP and produce exactly
4096 calls per lane on both one and two SMs. A separate 4096-launch copy-helper
control exercises assistance enabled and disabled with the same kernel objects.
Full graphs are separate acceptance checks.

Qualification uses a new managed artifact archive rather than overwriting
`lib/libmuonrt.a` or old archives. Runtime-only relink receipts bind the new archive
and retain identical kernel/stage objects and frozen references. Preserve failed
runs and old ELF hashes; do not resume them as corrected executions.

`build_device` accepts `RADIANCE_MUONRT_ARCHIVE=/path/to/qualified.a` and
records that archive's SHA256 in the build manifest. Direct make builds can use
`MUONRT_ARCHIVE=/path/to/qualified.a`. The default normal/tapeout archive selection
is unchanged; use an explicit qualified archive for these model evaluations.
