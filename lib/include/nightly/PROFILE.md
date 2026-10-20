# Kernel profiling

`profile.h` provides optional 64-bit Muon cycle and instruction snapshots.
`profile_host.h` collects committed snapshots and prints JSON on Rocket after GPU work.
These helpers are not yet enabled in the existing workload archives.

## Cost and measurement boundaries

`NIGHTLY_PROFILE=0` is the default. Its BEGIN, END and PUBLISH macros expand to nothing
and do not evaluate their arguments. Verify the executable load segments against a build
without those calls before claiming identical performance binaries. Identical source or
successful compilation alone does not establish that property.

`NIGHTLY_PROFILE=1` inserts counter reads. Each 64-bit read uses high/low/high and retries
when the low half wraps. Sampling costs instructions and can change scheduling, register
pressure, cache behavior and accelerator overlap. Buffering or moving prints outside the
interval removes print work from the interval; it does not make the counter reads free.
Do not subtract an empty-probe calibration and call the result exact.

For zero additional target instructions, observe PCs and counters in Cyclotron or an RTL
monitor while running the unchanged ELF. This can increase simulator wall time. The current
Cyclotron stage-entry journal is such an observer, but entry-to-next-entry intervals include
runtime between stages, and its final interval includes checking. It is not a clean compute
window. FPGA observation also needs signals/counters exposed by the actual image.

Start after launch, input loading and runtime initialization. End after the workload's
required DMA, stores, synchronization and completion fences, before output verification.
Do not add a fence solely to make a profiling boundary convenient: it can serialize the
algorithm. Sample every participating core; cluster 0 finishing does not establish that
the other SMs finished. Required output movement belongs inside the work interval.

## Device use

Include `profile.h` in the device entry translation unit. A boot warp with one active lane
on each core can keep one local span:

```cpp
NightlyProfileSpan span;
// Existing launch gate and setup.
NIGHTLY_PROFILE_BEGIN(span);
// Existing schedule calls and required algorithm completion/fences.
NIGHTLY_PROFILE_END(span);
NIGHTLY_PROFILE_PUBLISH(vx_core_id(), span);
// Existing verification/completion path.
```

Publish writes uncached printBuf slots 32..47 and ready slots 56..57. Reserve these slots in
the workload; do not combine with another mailbox protocol using the same slots. Publication
happens after the end sample. It is not an MX/VPU completion fence. The two cores write
disjoint slots; physical SM identity comes from each SM's own printBuf mapping.

For finer phases, keep begin/end samples in a caller-owned, bounded buffer and export it
after work. Each additional sample still has a cost. Never print from a hot loop. Stage
buffers and their export protocol are not provided by this first helper.

`NIGHTLY_PROFILE_HAS_INSTRET=1` is an explicit capability assertion. Leave it zero for the
current Cyclotron: its CSR file exposes minstret storage without updating it. The parser
must receive null instruction counts for that backend. In current Muon RTL, the counter
named minstret increments on execute request acceptance; describe it as issued warp
instructions when that behavior is confirmed for the deployed source. It is not a lane
instruction count. Unsupported HPM, MX, VPU or lane counters must remain unavailable.

## Host use and output

Include `profile_host.h` after `host.h`. Call `nightly_profile_host_clear(sm_count)` before
launch. After every core completes, `nightly_profile_host_report(sm_count)` returns 1 and
prints one committed record per core, or returns 0 without printing if data is missing.
The host must use an appropriate completion/timeout protocol; do not poll the counter data
throughout the measured interval. Post-run reporting still belongs before HTIF exit.

The prefix is `RADIANCE_PROFILE ` followed by one JSON object. Raw fields are unsigned
64-bit decimal counters, except unavailable instruction fields, which are JSON null:

```text
RADIANCE_PROFILE {"schema":1,"sm":0,"core":0,"cycle_begin":8589934592,"cycle_end":8589934692,"instructions_begin":null,"instructions_end":null}
```

Use the supplied parser with a context file recording workload, exact device ELF hash,
physical topology, backend, memory model and clock domain:

```json
{
  "schema": 1,
  "workload": "example-1sm",
  "device_elf_sha256": "<64 lowercase hex digits from the executed ELF>",
  "sm_count": 1,
  "cores_per_sm": 2,
  "backend": "cyclotron-coupled",
  "memory_model": "hbm2-4ch",
  "clock_domain": "muon",
  "common_cycle_epoch": true,
  "instruction_counter_semantics": "unavailable",
  "peak_issue_per_core_per_cycle": null
}
```

```sh
python3 profile_output.py --log console.log --context run-context.json \
  --out profile.json --csv profile.csv
```

Use a complete, unmodified console log. Mixed runs, duplicate cores, malformed records,
missing cores, topology mismatches and counter resets are errors. The run context is an
external provenance record; the console does not authenticate the ELF or hardware image.

With a confirmed common counter epoch, whole-program work cycles are latest end minus
earliest begin. Sum instructions over participating cores and divide by those cycles for
whole-chip IPC. With unsynchronized epochs, only per-core durations and IPC are available;
the parser does not label the longest local duration as whole-program latency. Cycles are
not summed across parallel SMs to obtain latency.

Issue utilization needs a confirmed issue counter and the actual per-core peak from that
configuration. MX/VPU utilization and lane efficiency need their own qualified counters.
No clock frequency or ratio, matrix throughput, VPU peak, or tapeout utilization constant
is assumed. Keep Rocket setup/wait/check counters separate from Muon counters. Compare
HBM and DRAM only with matching workloads and explicit memory/clock configurations.
