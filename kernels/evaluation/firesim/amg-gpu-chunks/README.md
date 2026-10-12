# Original-count GPU AMG Gather, split into two jobs

Spatter's AMG GPU case 0 has 256 pattern entries and 14,705,882 repetitions:
3,764,705,792 Gather transfers. Its source array is 1,882,363,840 bytes,
which exceeds the current `0x70000000` Muon input window by 3,315,648 bytes.
The [count plan](../../../spatter/evaluation/amg-gpu-chunk-plan.json) divides
the repetitions into two contiguous ranges of 7,352,941. Each source array
is 941,187,392 bytes and uses values at the original global source indices.
Together the jobs perform the original transfer count and 60,235,292,672
logical payload bytes, counting one eight-byte read and one eight-byte write
per transfer.

The pinned upstream Spatter serial backend ran each chunk's exact input and
pattern arrays; both complete output digests match the corresponding Radiance
expectations in the [chunk golden table](../../../spatter/evaluation/amg-gpu-chunks-upstream.csv).
The source windows agree byte for byte where they overlap. They were combined
into the original 1,882,363,840-byte source array, and the **unmodified
upstream serial Gather** then ran once at the complete original count. Its
output SHA-256 is
`70273629aeb58c1d14caff69868e10b61bc7b563524500a8396c8d72edd15a44`
and its complete digest is `0e8f823875db84ff`, exactly the final chunk's
expected digest. The [full-run record](../../../spatter/evaluation/amg-gpu-full-upstream.json)
fixes the source, pattern, driver, plan, and output hashes; the
[artifact manifest](../../../spatter/evaluation/amg-gpu-full-artifacts.csv)
also hashes the retained source and output files.
`verify-native-evidence.py`, called by the follow-up archive exporter, checks
the original-count output file and digest again from the retained binaries.

This equality follows from this case's `wrap=1`: each dense output slot is
overwritten in every repetition, so its final value comes from the last
repetition. The first chunk still has its own compiled transfer schedule to
preserve the original work count. U250 jobs 1699 and 1700 both passed their
complete host output digest and guard checks. Their [plan](plan.csv),
[results](results.csv), and [artifact hashes](artifacts.csv) record the exact
ELFs and guest outcomes. The final chunk's digest equals the unmodified
upstream full-count run's digest.

| Job | Repetition range | Complete output digest | Whole-program target cycles |
| ---: | --- | --- | ---: |
| 1699 | 0–7,352,940 | `bcdd15879381d51d` | 10,921,959,722 |
| 1700 | 7,352,941–14,705,881 | `0e8f823875db84ff` | 10,921,959,722 |

The two jobs reboot the target independently. Their whole-program target
cycles include ELF loading and do not combine into a single-launch GPU
latency or upstream throughput number. The current software path has no
on-device streaming between launches, so this validates the count
decomposition and output, not a performance-equivalent implementation of
upstream's one-launch CUDA benchmark.

```sh
python3 kernels/evaluation/firesim/capture_cases.py \
  --plan kernels/evaluation/firesim/amg-gpu-chunks/plan.csv \
  --out kernels/evaluation/firesim/amg-gpu-chunks --require-complete
python3 kernels/evaluation/firesim/verify_cases.py \
  kernels/evaluation/firesim/amg-gpu-chunks \
  --golden-table kernels/spatter/evaluation/amg-gpu-chunks-upstream.csv
```
