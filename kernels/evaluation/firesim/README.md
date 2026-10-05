# Radiance kernels on the U250 FireSim image

This directory records a one-cluster FPGA check of the STREAM and Spatter
kernels. The [comparison table](comparison.csv) joins these runs to the
existing full-SoC Verilator and Cyclotron results. The [plan](plan.csv) gives
every job, ELF hash, host-side check, and expected outcome;
[results.csv](results.csv) gives the observed UART outcome and raw target-cycle
count. The `raw/<job-id>/` directories preserve the queue request, console,
FireSim logs, and available memory statistics for each completed job. The
[artifact manifest](artifacts.csv) hashes every retained raw file.

## Hardware and software provenance

- Target: Xilinx Alveo U250 with the
  `FireSimE4M3MxGemminiRadianceConfig-BaseXilinxAlveoU250Config` image. The
  generated [device tree](device-tree.dts) has two Muon cores and MX-Gemmini. These kernels use
  Muon; they issue no MX-Gemmini instructions.
- Bitstream archive: `/scratch/nicorakela/firesim_radiance-slim.tar.gz`,
  SHA-256 `cfa8bce8c76d216c12a5771337566a0c543cde74330fcf5323094f215ce4c3cc`.
  The contained `firesim.bit` hashes to
  `c98c7c5761bcd496c5ca12f48a1500be2c35edcf11c16066124fe89e7c1b2ade`.
- The [bitstream metadata](bitstream-metadata.txt) names FireSim revision
  `b084672c2f8cf32e55d78f73a001074c23f8a2b8-dirty`. The `-dirty`
  suffix means the exact source diff of the built hardware has not been
  recovered. This image is a different hardware build from the pinned
  Verilator measurements in the [main kernel report](../../README.md).
- The immutable [FireSim HWDB entry](hwdb-entry.yaml) hashes to
  `d4015580a0f7d58c0cd2606832d9579981feef98f4c5c1a29d25f7ba4f3c6bc4`.
  Each queue job pinned that entry and flashed its bitstream.
- The matching FireSim U250 software driver hashes to
  `37f9039c56d21d5a7eff21a4ffe3c0df3a9af991ae346781c8d76c3e7e8ba318`.
- The submitted binaries are the previously evaluated fused RV64 host/RV32
  Muon ELFs. The input to every queue job was checked against the
  [ELF inventory](../elf-snapshot.csv) before submission. No RTL source was
  changed for these runs.

The 16 deterministic main rows cover all four STREAM operations, all five
Spatter transfer families on the GPU STREAM deck, xRAGE asteroid Gather and
ordered Scatter, both LULESH trace operations, a scaled AMG Gather, and
materialized and fused Gather→Scatter chains. A seventeenth row runs parallel
xRAGE9 Scatter as exploratory: overlapping destinations have no deterministic
final output in that mapping, and its host check covers only guards and a
nonzero probe. The small Gather smoke test, two negative controls, and three
diagnostic runs are listed separately in `plan.csv`. The ordered Scatter
mappings execute overlapping destinations deterministically; they are not
CUDA atomic throughput measurements. AMG uses 1,024 repetitions in place of
the much larger original count.

Fourteen deterministic cases passed their FPGA host check. Ordered xRAGE9
and LULESH Scatter failed. The exploratory parallel xRAGE9 mapping passed
its guard/nonzero-probe check, which does not establish a unique final array.

The original-size ordered xRAGE9 Scatter ELF (job 1658) reached the guest
check but reported `*** FAILED *** (code = 1)` after 136,324,022 whole-program
target cycles. The same ELF passed Cyclotron's complete-output digest check;
its FPGA host program checks 64 output samples and guards. The UART does not
say which of those checks failed; the guard-only diagnostic below isolates
the sample comparison. This remains an unresolved FPGA output discrepancy,
not a passed FPGA result. Its raw log is retained under `raw/1658/` and the
case remains a failed row in `comparison.csv`.

Diagnostic job 1668 uses a copy of the same ELF with only the host's final
sample-digest branch at virtual address `0x80000774` (ELF offset `0x1774`)
changed to proceed to success. The Muon code, input data, and guard checks
are unchanged. Its SHA-256 is
`55d25dd54af7463ea2db23721b2115139956766b4b0e2425c31c94d281fea1fe`.
`capture.py` compares both local ELFs byte for byte outside that four-byte
instruction.
It passed on the U250 after the same 136,324,022 target cycles as job 1658.
The original guest failure therefore comes from the 64-output sample
comparison, not the guard check. This diagnostic is never counted as a
correctness pass for ordered xRAGE9.

Job 1669 tests a separately archived newer rebuild of the same ordered
xRAGE9 mapping (ELF SHA-256
`a2a7b29f021e1794dab9a16aa9a0cd9f7c669bc4cfdc359654754960af10f90b`).
Its build record has the same expected complete and sampled digests as job
1658. A fresh Cyclotron functional run of this exact rebuild checked all
2,051,101 output elements: digest `215de6e81154b9c5`, intact guards, and
2,383,756 nonzero words. The digest matches the independent upstream serial
golden. Its [log and input configuration](raw/model-diagnostic) are archived;
the checker binary is identified by SHA-256
`53236c3bfd41ed0163e6f7629cefe077ded2df1cfa4589228c176003bd6b6e5c`
in the earlier dependency snapshot. Functional model steps are not GPU
performance cycles. Its U250 guest check also failed after 136,324,022
whole-program target cycles. The newer software build did not remove the FPGA
host-check failure; the UART does not distinguish its samples from its guards.
This diagnostic is not a workload pass.

Job 1671 is a two-entry ordered-overlap Scatter smoke ELF that passed the
earlier Verilator and model checks. It also passed the complete FPGA host
digest check. This shows the ordered path can work on a tiny input; it does
not explain the two original-size ordered failures.

## Correctness checks

Each ELF starts the Muon cores from the RV64 host, waits for completion, reads
the output, checks guard regions, and reports success or failure through
`tohost`. The `host_check` column distinguishes a complete output digest from
a sample-and-guards check. The archived per-ELF build records in
[`raw/build-metadata`](raw/build-metadata) fix the readback mode and expected
digest for each submitted workload; `verify.py` checks them against `plan.csv`.
The Cyclotron rows in `comparison.csv` check
complete deterministic output digests. The deterministic Spatter model rows
were separately compared with the unmodified upstream serial Spatter kernels;
the [golden tables](../../spatter/evaluation/README.md) record input and output
hashes. STREAM uses its generated arithmetic reference.

Job 1650 is a negative control. It changes the first 32-bit Gather pattern
entry from 0 to 1 at file offset `0x6000` in a copy of the small smoke ELF,
without changing the expected digest. The unmodified ELF passes; the mutated
ELF reports `*** FAILED ***` with driver exit code 1. The queue lifecycle can
still say `DONE` for a failed guest program, so `capture.py` checks the UART
result and driver exit code rather than treating queue completion as success.
An earlier control attempt, job 1647, failed during FireSim `infrasetup` because
the host SSH connection closed; it produced no guest result and was retried as
1650. The earlier setup attempts 1638–1642 also produced no guest result.
Their queue requests and setup logs, along with job 1647, are retained under
[`raw/setup-failures`](raw/setup-failures) and excluded from the workload
table. Job 1666 changes the first STREAM Copy source element from `1.0f` to
`0.5f` at ELF file offset `0x6000` while keeping the sampled output digest
unchanged. Its FPGA UART reports `*** FAILED ***` with driver exit 1, while
the unmodified Copy ELF passes. This confirms that the sample-and-guards path
catches that changed sampled value.

## Reading the numbers

`rtl_gpu_cycles` and `rtl_issue_slot_use` come from the earlier full-SoC
Verilator Muon performance reports. `model_gpu_cycles` and
`model_gmem_bytes_issued` come from Cyclotron's generic DRAM timing model.
`firesim_target_cycles_whole_program` is the FireSim driver's total emulated
target cycles, including boot, ELF loading, host code, Muon work, and output
verification. It is **not** a GPU kernel latency and must not be compared
directly with the RTL or model GPU-cycle columns. For a failed row it is the
cycle count to the failed guest check, not a successful workload timing.
Several short programs have
the same 50,119,127-cycle total. No FireSim GPU issue utilization, GPU launch
latency, HBM bandwidth, or two-cluster scaling is measured here. The exported
`memory_stats0.csv` files contain only a header, so they cannot supply a
bandwidth estimate. The FireSim `Host Frequency` and wallclock rate are FPGA
emulation rates, not a calibrated Radiance GPU clock.

To get FPGA launch latency without changing RTL, instrument the RV64 kernel
host to read its cycle counter immediately before releasing Muon reset and
after the all-finished register asserts, and emit that interval through the
host console. Then rebuild and rerun the ELFs with the same output checks and
an explicit hardware/host clock-domain interpretation. Per-core utilization
and calibrated memory bandwidth require counters or traces that this bitstream
does not currently export. The present comparison table uses the existing RTL
and model measurements for those quantities.

## Per-case results

The RTL and model columns are kernel-cycle measurements from different
simulators and hardware revisions. The U250 column records the guest output
check. Its cycle column is the whole-program FireSim count described above.
A dash means that fidelity was not run for that case.

| Case | RTL GPU cycles | Model GPU cycles | U250 guest | U250 whole-program cycles |
| --- | ---: | ---: | --- | ---: |
| STREAM Copy | 3,970,659 | 3,987,096 | pass | 48,114,362 |
| STREAM Scale | 3,971,774 | 4,024,544 | pass | 48,114,362 |
| STREAM Add | 7,485,262 | 6,346,569 | pass | 48,114,362 |
| STREAM Triad | 7,750,785 | 6,346,349 | pass | 48,114,362 |
| GPU STREAM Gather | 842,623 | 1,998,857 | pass | 50,119,127 |
| GPU STREAM Scatter | 1,247,066 | 1,040,016 | pass | 50,119,127 |
| GPU STREAM GatherScatter | 2,158,179 | 2,038,053 | pass | 50,119,127 |
| GPU STREAM MultiScatter | 1,360,268 | 1,046,089 | pass | 50,119,127 |
| GPU STREAM MultiGather | 945,962 | 1,999,593 | pass | 50,119,127 |
| xRAGE5 Gather | — | 100,529,234 | pass | 204,486,032 |
| xRAGE9 parallel Scatter | — | 98,895,114 | probe pass | 182,433,617 |
| xRAGE9 ordered Scatter | — | 96,235,005 | fail | 136,324,022 |
| LULESH Gather | — | 74,419,767 | pass | 50,119,127 |
| LULESH ordered Scatter | — | 74,611,507 | fail | 70,166,777 |
| AMG Gather, 1,024 repetitions | — | 1,156,957 | pass | 50,119,127 |
| Materialized Gather→Scatter | 2,294,139 | 3,100,336 | pass | 48,114,362 |
| Fused GatherScatter | 1,457,400 | 1,060,936 | pass | 50,119,127 |

The exact case sizes, host check mode, model memory traffic, RTL issue-slot
use, ELF hash, and queue job ID are in [comparison.csv](comparison.csv).
The parallel xRAGE9 row checks guards and one nonzero output probe; its
overlapping writes have no deterministic final-output golden.

## Capture and checks

With access to the original FireSim queue and local ELF archive, refresh the
portable log snapshot and comparison table from the repository root:

```sh
python3 kernels/evaluation/firesim/capture.py --require-complete
python3 kernels/evaluation/firesim/make_comparison.py --check
python3 kernels/evaluation/firesim/verify.py
```

The first command compares the queue manifest and local ELF content to
`plan.csv`, checks the pinned HWDB hash, copies the small raw records, parses
the UART outcome, and labels a failed workload as such. It rejects missing
guest results and malformed outcomes. The second
command joins those outcomes to the already committed RTL and model result
CSVs, checking the model-to-FireSim ELF hashes. Use `make_comparison.py`
without `--check` only when regenerating the table. `verify.py` checks the
copied logs, queue requests, UART and artifact hashes, and rebuilds the
comparison rows from the RTL/model source CSVs without queue access. While
jobs are still pending, run it with `--allow-pending`. The bitstream and large
ELF files stay in the local artifact archive; their hashes and the raw job
records are kept here for later reporting.
