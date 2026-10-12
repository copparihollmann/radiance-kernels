# Native Spatter application traces

This run series uses the original repetition counts in Spatter's CPU
application-trace decks. The JSON inputs are copied byte for byte into
[`kernels/spatter/inputs/standard-suite`](../../../spatter/inputs/standard-suite)
from `hpcgarage/spatter` revision
`ec8923711f8dc21eedff7189f12b02eb06845d2f`; its `LICENSE` and `COPYING`
files are alongside the inputs. The original xRAGE asteroid deck is separate
and is covered by the [earlier U250 campaign](../README.md) and the
[output-initialization follow-up](../output-initialization/README.md).

There are 17 original-count cases in the AMG, LULESH, and Nekbone CPU trace
decks: 2, 12, and 3 respectively. LULESH case 1 already passed a complete
U250 output digest in job 1659. LULESH case 3 passed complete readback after
the host output clear in job 1679. All 15 remaining cases passed complete
U250 output and guard checks in
[plan.csv](plan.csv), with the observed UART results in [results.csv](results.csv).
Every planned ELF has a complete host output digest and guards; Scatter cases
with repeated destinations use the destination-owner ordered policy.
Job 1689, LULESH case 8, failed during `infrasetup` when the local SSH
connection reset. It produced no guest UART and is retained as an
infrastructure failure in the table and raw logs. Job 1705 retried the same
ELF and SHA-256 and passed. The trace series therefore has 16 queue attempts
for 15 new cases, with 15 guest passes and one setup failure.

| Deck | Case | Transfer | Repetitions | Passing U250 job | Whole-program target cycles |
| --- | ---: | --- | ---: | ---: | ---: |
| AMG CPU | 0 | Gather | 1,454,647 | 1681 | 138,328,787 |
| AMG CPU | 1 | Gather | 1,454,647 | 1682 | 204,486,032 |
| LULESH CPU | 0 | ordered Scatter | 577,806 | 1683 | 158,376,437 |
| LULESH CPU | 1 | Gather | 231,198 | 1659 | 50,119,127 |
| LULESH CPU | 2 | ordered Scatter | 167,805 | 1684 | 92,219,192 |
| LULESH CPU | 3 | ordered Scatter | 128,002 | 1679 | 224,533,682 |
| LULESH CPU | 4 | Gather | 96,360 | 1685 | 50,119,127 |
| LULESH CPU | 5 | Gather | 96,360 | 1686 | 50,119,127 |
| LULESH CPU | 6 | Gather | 96,186 | 1687 | 50,119,127 |
| LULESH CPU | 7 | ordered Scatter | 88,011 | 1688 | 70,166,777 |
| LULESH CPU | 8 | Gather | 76,794 | 1705 | 50,119,127 |
| LULESH CPU | 9 | Gather | 76,794 | 1690 | 50,119,127 |
| LULESH CPU | 10 | Gather | 76,794 | 1691 | 50,119,127 |
| LULESH CPU | 11 | Gather | 72,270 | 1692 | 50,119,127 |
| Nekbone CPU | 0 | Gather | 982,980 | 1693 | 138,328,787 |
| Nekbone CPU | 1 | Gather | 982,980 | 1694 | 182,433,617 |
| Nekbone CPU | 2 | Gather | 491,490 | 1695 | 116,276,372 |

These are transfer microbenchmarks extracted from the named applications,
not complete application runs. The original LULESH case 1 and corrected case
3 records are in the linked earlier campaigns; the 15 new cases are in this
directory.

The [upstream comparison](../../../spatter/evaluation/native-upstream-golden.csv)
is independent of the FPGA run. It executes the pinned upstream Spatter serial
kernel on the exact generated source and pattern files for all 15 new cases.
All 15 complete output digests agree with the Radiance build expectations.
The table includes upstream source, input, output, suite, and ELF SHA-256
values; the [artifact manifest](../../../spatter/evaluation/native-upstream-golden-artifacts.csv)
identifies the retained local golden binaries. A full U250 digest pass then
checks the FPGA result against the same value. The queue uses a reusable
FireSim `agustin-radiance-spatter-smoke` workload descriptor, but each job
stages the distinct ELF named and hashed in `plan.csv`.
The input copies, footprint inventory, golden arrays, and source snapshots
can be checked together with:

```sh
python3 kernels/spatter/tools/verify-native-evidence.py \
  --upstream /scratch/agustin/projects/spatter-data/spatter-upstream \
  --inputs-root /scratch/agustin/projects/chipyard/firesim-hpc-inputs
```

The 114-row [size inventory](../../../spatter/evaluation/native-workload-footprint.csv)
covers all 13 JSON files in the upstream standard suite. Thirty-eight cases
fit the current single-launch RV32 task and GPU address checks at their native
counts; 76 do not. The 38 comprise these 17 application traces, 7 large
Pennant traces, 5 GPU STREAM cases, 5 CPU STREAM cases, and 4 pattern-size
tests. The prior U250 campaign covered all five GPU STREAM cases. The size
inventory is a build-feasibility check, not evidence that all 38 ran. In
particular, the Pennant cases have up to two billion transfers, and the cases
that exceed the address or task limits need a chunked mapping before any
native-size FPGA result is possible. The GPU AMG deck remains distinct from
the CPU AMG deck measured here: the earlier GPU AMG result used 1,024
repetitions rather than its original 14,705,882. A narrow
[two-chunk Gather plan](../../../spatter/evaluation/amg-gpu-chunk-plan.json)
now covers the original GPU AMG case 0 count with source indices rebased at
the chunk boundary. The pinned upstream serial implementation also ran the
complete original-count case and matched the final chunk digest. The
[GPU AMG chunk report](../amg-gpu-chunks/README.md) records two complete-output
FPGA passes and separates their whole-program cycles from GPU latency.

| Standard-suite group | Cases | Single-launch fit | Current FPGA work |
| --- | ---: | ---: | --- |
| CPU AMG, LULESH, Nekbone traces | 17 | 17 | All 17 passed complete-output FPGA checks |
| CPU Pennant traces | 17 | 7 | None run at original count |
| GPU application traces | 34 | 0 | GPU AMG case 0 passed in two Gather jobs; other native cases unrun |
| Basic tests | 46 | 14 | All 5 GPU STREAM cases passed complete-output FPGA checks; CPU STREAM and pattern-size cases unrun |

The per-job target-cycle numbers include boot, ELF loading, host zeroing,
Muon execution, and complete output readback. They are not GPU kernel
latencies or utilization numbers. These trace cases are independent Spatter
microbenchmarks. They do not form a full AMG, LULESH, or Nekbone application
dependency graph; the existing Gather→Scatter composition tests exercise
one explicit two-stage mapping only.

To refresh the queue evidence:

```sh
python3 kernels/evaluation/firesim/capture_cases.py \
  --plan kernels/evaluation/firesim/native-traces/plan.csv \
  --out kernels/evaluation/firesim/native-traces --require-complete
python3 kernels/evaluation/firesim/verify_cases.py \
  kernels/evaluation/firesim/native-traces \
  --golden-table kernels/spatter/evaluation/native-upstream-golden.csv
```

The `raw/<job-id>/` folders preserve the request, UART, FireSim driver log,
build record, and available counters. `artifacts.csv` hashes every retained
raw file. Large ELFs and upstream output binaries remain in the local
artifact archive and are identified by SHA-256 in the plan and golden tables.
