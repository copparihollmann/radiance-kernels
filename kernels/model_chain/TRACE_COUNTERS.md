# Using profiling logs, traces and external counters

This guide accompanies `radiance-profiling-ready-1-2-4sm-v1.tar.zst`.
Run the commands below from the extracted bundle directory. The coworker manages
FireSim and the hardware image; these Python commands only parse saved output.

## Start with the counters already in the binaries

No profiling enable flag is needed for these ELFs. Run the prepared `boot.elf`
on the matching SM configuration and save the complete console log, including
`RADIANCE_CONTEXT`, every `RADIANCE_PROFILE`, `RADIANCE_HOST` and `RADIANCE_DONE`.

```text
saved-runs/
  tinyllama-2sm/
    console.log
  stream--copy--2sm/
    console.log
```

Use exact workload IDs from `workload-list.json`. For a 2-SM run there must be
four PROFILE records: two cores on each of two SMs.

For an image with equal MX and Muon frequencies:

```sh
python3 results.py --sms 2 --logs saved-runs --output reports --mx-clock-ratio 1
python3 hpc.py collect --sms 2 --logs saved-runs --output reports-hpc --mx-clock-ratio 1
```

These reports already provide GPU work cycles, instructions, IPC, SIMT issue
utilization and MX utilization. Boot/loading, explicit input initialization,
post-run verification and printing are outside the GPU work interval. Required
scheduling, accelerator setup, DMA and completion fences are inside it. Rocket
setup/wait counters are separate. A complete performance log does not by itself
establish numerical correctness; optional readbacks use the bundle's dump plan.

`--mx-clock-ratio` means **MX frequency / Muon frequency**. It is not a trace
switch. Use the actual ratio of the image. Add `--gpu-clock-hz` with the actual
Muon target frequency, not FPGA host frequency or simulation speed. For example,
use `--gpu-clock-hz 500000000` only if Muon really runs at 500 MHz. The HPC collector
then reports useful payload bandwidth for cases with known logical byte counts.
That is distinct from physical HBM/DRAM traffic.

## Lane efficiency: add a Muon issue trace if the image exposes one

**The tar does not include an FPGA Muon trace bridge or a switch that enables
one.** The pinned graphics targets use `WithNoTraceIO`, so enabling ordinary
Rocket TracerV is not a solution for Muon lane masks. Radiance's existing
`trace=true` path instantiates a simulation tracer backed by DPI/Cyclotron;
simply enabling that path is not a supported FPGA trace recipe.

If the image has no suitable trace bridge, run the normal counters above and
leave lane efficiency unavailable. To obtain it on FPGA, the image owner needs
a bridge or equivalent observer that exports these fields for **each accepted
Muon execute request**:

| CSV field | Meaning |
|---|---|
| `cycle` | Muon cycle, aligned to the `mcycle` epoch in PROFILE records |
| `sm` | Physical SM index, starting at 0; use the instance identity, not the aliased 1-bit cluster CSR |
| `core` | Local core index, 0 or 1 within that SM |
| `tmask` | 16-bit active-lane mask for this accepted warp instruction |

The RTL observation point is `Backend.scala`: `execute.io.req.fire` and
`executeIn.uop.tmask`. Count one event per accepted warp instruction, including
stores. A register-writeback-only or per-lane trace has different semantics.
If the bridge uses a different cycle epoch, translate it to Muon `mcycle` before
collection; raw FPGA wall-clock timestamps cannot be used directly.

Normalize the actual exported data to CSV, then save it here:

```text
saved-runs/tinyllama-2sm/issue-trace.csv.gz
```

Required format (the following rows illustrate syntax only):

```csv
cycle,sm,core,tmask
100000,0,0,0xffff
100001,0,1,0x00ff
100002,1,0,0xffff
100003,1,1,0x0001
```

Decimal and `0x` integers are accepted. Export all accepted instructions covering
all participating cores' measured windows, without sampling, truncation or dropped
events. Stream or gzip large traces. Do not use the illustrative rows as input
for a real run. The parser checks that participating cores have events; that is
not proof that the trace is complete. The operator must check capture/drop status.

`results.py` and `hpc.py` automatically discover `issue-trace.csv.gz` or
`issue-trace.csv` alongside `console.log`; rerun the same collection command.
For one workload, explicitly select the trace:

```sh
python3 profile_output.py \
  --log saved-runs/tinyllama-2sm/console.log \
  --issue-trace saved-runs/tinyllama-2sm/issue-trace.csv.gz \
  --mx-clock-ratio 1 --out profile.json --csv cores.csv
```

The parser keeps events in each core's `[cycle_begin, cycle_end)` window and
computes `lane_efficiency = total active lanes / (16 * accepted instructions)`.
Whole-workload metrics are in `profile.json`; `cores.csv` contains per-core rows.
The collectors' `results.csv` contains one summary row per workload.

## VPU utilization and physical memory traffic: supply measured counters

The current binary CSR interface does not expose these counters. They need an
existing image/bridge counter or an external trace-derived measurement. A parser
flag cannot create them. If unavailable, leave the fields null.

If that source exists, snapshot or derive it over the GPU work interval, aggregate
all participating SMs/units or memory channels, and append one JSON record of each
supported type to the **saved console log after the run**. Keep the original log.
Use actual values and identify the measurement source. These are syntax examples,
not measurements:

```text
RADIANCE_ENGINE {"engine":"vpu","scope":"work","source":"REPLACE_WITH_ACTUAL_COUNTER_SOURCE","busy_cycles":20,"capacity_cycles":100}
RADIANCE_MEMORY {"scope":"work","source":"REPLACE_WITH_ACTUAL_AXI_BOUNDARY","read_bytes":100,"write_bytes":10}
```

For VPU, both values must use the same clock and unit accounting. If busy_cycles
sums busy cycles of N units, capacity_cycles must sum their available cycles over
the same interval (N * interval for identical continuously available units).
Do not divide summed unit activity by just one unit's capacity. Name the VPU
units and the counter semantics in the source or a companion measurement note.

For memory, use actual transferred bytes at one named HBM/DRAM boundary, covering
all selected GPU channels over the same work window. Exclude Rocket boot/loading
traffic when reporting GPU bandwidth; do not add L2 and memory-controller traffic
together. Whole-run counters that include loading cannot be labeled `scope:work`.
If the trace clock differs, convert the window boundaries first. Record which
channels, address range and boundary were counted in a companion note.

Rerun the collector with the actual Muon frequency, for example **only for a
500-MHz Muon image with equal MX/Muon clocks**:

```sh
python3 results.py --sms 2 --logs saved-runs --output reports \
  --mx-clock-ratio 1 --gpu-clock-hz 500000000
```

VPU utilization is busy_cycles / capacity_cycles. Physical memory bandwidth is
(read_bytes + write_bytes) * Muon_Hz / GPU_work_cycles. The report preserves byte
counts and keeps useful payload bandwidth separate. Utilizations are fractions:
0.25 means 25%. Different HBM/DRAM runs should use separate log/report directories
and record their memory configuration, target clocks and image/HWDB identity.

## What to send back

Send `results.csv`, `results.json` and the per-workload profile JSON files, along
with the complete console logs, actual clock ratio/frequency and image identity.
Include trace/counter source notes when optional metrics are present. Keep large
raw traces compressed; numerical readbacks are independent of these profiles.
No kernel recompilation is needed to parse additional data from the same run.
Adding an FPGA bridge, if needed, is an image build change and requires another run.
