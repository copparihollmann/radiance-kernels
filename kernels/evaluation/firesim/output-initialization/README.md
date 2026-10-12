# Output initialization on the U250

The first U250 run of native-size ordered xRAGE9 Scatter (job 1658) and
LULESH Scatter case 3 (job 1660) failed their 64-position output checks.
The [original campaign](../README.md) retains those failures. This follow-up
isolates the cause and records the corrected runs in [plan.csv](plan.csv),
[results.csv](results.csv), and the hashed [raw log manifest](artifacts.csv).

The fused ELF contains the Muon RV32 program and its initialized data in
separate `LOAD` segments. The output array is in the RV32 BSS. In the archived
LULESH ELF, the final RV32 `LOAD` segment has `FileSiz=0` and `MemSiz=0`, so
loading the ELF does not initialize that output storage. The old RV64 host
initialized two 16-word guards and started Muon without clearing the output.
The Cyclotron functional model happened to provide zeroed memory. The U250
did not guarantee it. Unwritten Scatter destinations then carried stale data
into the output digest.

Job 1676 rebuilt the RV64 host to print the 64 samples after a failed check.
Its six RV32 segments were replaced with exact bytes from job 1660's ELF;
[`splice_device.py`](../splice_device.py) verifies their addresses, hashes,
and `LOAD` sizes. The diagnostic failed with intact guards. Comparing its UART
samples with the complete upstream serial Spatter output gives 10/64 matches.
Every one of the 54 mismatches has an upstream value of zero; every sampled
nonzero value matches. The comparison is in
[`raw/1676/sample-comparison.json`](raw/1676/sample-comparison.json), and the
checker is [`compare_samples.py`](../compare_samples.py).

The host now writes zero to every output word before releasing Muon reset.
Jobs 1677 and 1678 keep the respective failed ELFs' RV32 `LOAD` segments
byte for byte and pass the original 64-sample check. Jobs 1679 and 1680 use
the same RV32 images with a complete output digest in the RV64 host. Both pass.
The complete digests are `b27b51a43dc716b5` for LULESH case 3
(1,024,369 output elements) and `215de6e81154b9c5` for xRAGE9
(2,051,101 elements). Those expected digests independently match the pinned
upstream Spatter serial kernels; the [golden record](../../../spatter/evaluation/upstream-golden-results.csv)
contains the source and output hashes. The changed behavior before Muon launch
is the explicit output clear. No RTL or bitstream changed.

| Job | Case | Host check | Guest result | Whole-program target cycles |
| ---: | --- | --- | --- | ---: |
| 1676 | LULESH case 3, diagnostic without clear | 64 samples and guards | fail | 168,400,262 |
| 1677 | LULESH case 3, output cleared | 64 samples and guards | pass | 92,219,192 |
| 1678 | xRAGE9, output cleared | 64 samples and guards | pass | 202,481,267 |
| 1679 | LULESH case 3, output cleared | all output words and guards | pass | 224,533,682 |
| 1680 | xRAGE9, output cleared | all output words and guards | pass | 467,110,247 |

The cycle figures include boot, ELF loading, host initialization, Muon work,
and readback. In particular, full readback adds substantial host time. They
are not GPU kernel latencies. The U250 bitstream provenance and its unrecovered
dirty source diff are recorded in the [original campaign](../README.md).

Original-count AMG and the further xRAGE5 Gather readback have their own
[native-trace](../native-traces/README.md) and
[xRAGE](../xrage-full/README.md) records.

To refresh the queue snapshot on this machine:

```sh
python3 kernels/evaluation/firesim/capture_cases.py \
  --plan kernels/evaluation/firesim/output-initialization/plan.csv \
  --out kernels/evaluation/firesim/output-initialization --require-complete
python3 kernels/evaluation/firesim/verify_cases.py \
  kernels/evaluation/firesim/output-initialization
```

The exact fused ELFs and upstream output binaries remain in the local
artifact archive because they are too large for the repository. Their SHA-256
values are in `plan.csv` and the golden table. The raw UART, queue request,
build manifest, driver log, and memory statistics are retained here.
