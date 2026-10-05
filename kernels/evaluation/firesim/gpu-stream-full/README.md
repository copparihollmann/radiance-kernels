# Complete U250 readback for GPU STREAM Scatter families

The first U250 campaign checked 64 output positions and guards for standard
GPU STREAM Scatter, GatherScatter, and MultiScatter. Jobs 1702–1704 repeat
those three cases with a complete output digest in the RV64 host and explicit
output initialization. The rebuilt ELFs contain exact copies of the original
jobs' six RV32 `LOAD` segments, including Muon instructions and input data;
[`splice_device.py`](../splice_device.py) checks hashes, virtual addresses,
and segment sizes. The [plan](plan.csv) records both original and new ELF
paths and the expected complete digest. The corresponding digests match the
independent [upstream serial golden table](../../../spatter/evaluation/upstream-golden-results.csv).

All three new jobs passed their complete output and guard checks. The earlier
U250 Gather and MultiGather jobs already checked complete output digests, so
all five GPU STREAM transfer families now have complete U250 checks at 256
pattern entries and 1,024 repetitions. The [results](results.csv) and
[raw artifact hashes](artifacts.csv) record the guest outcomes.

| Job | Family | Whole-program target cycles |
| ---: | --- | ---: |
| 1702 | Scatter | 72,171,542 |
| 1703 | GatherScatter | 94,223,957 |
| 1704 | MultiScatter | 94,223,957 |

These totals include boot, host zeroing, and readback, so they are not GPU
kernel latencies or directly comparable to the earlier sample-check totals.

```sh
python3 kernels/evaluation/firesim/capture_cases.py \
  --plan kernels/evaluation/firesim/gpu-stream-full/plan.csv \
  --out kernels/evaluation/firesim/gpu-stream-full --require-complete
python3 kernels/evaluation/firesim/verify_cases.py \
  kernels/evaluation/firesim/gpu-stream-full
```
