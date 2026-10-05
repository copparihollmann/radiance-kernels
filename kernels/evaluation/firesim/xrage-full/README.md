# Complete U250 readback for xRAGE5 Gather

The original-size xRAGE asteroid pattern 5 has 8,368,968 Gather transfers
and a dense output of the same length. U250 job 1645 passed a 64-position
sample check. Job 1696 reuses its exact six RV32 `LOAD` segments and rebuilds
only the RV64 host to initialize the output and check every output word. The
[plan](plan.csv) records both ELF paths and hashes; the device-image checker
in [`capture_cases.py`](../capture_cases.py) verifies the RV32 segments are
identical. The expected complete digest `2260887d7f6bc955` independently
matches the [pinned upstream Spatter serial run](../../../spatter/evaluation/upstream-golden-results.csv).

Job 1696 passed the complete output and guard check after 1,571,735,762
whole-program target cycles. The observed UART outcome is in
[results.csv](results.csv), with raw queue records hashed in
[artifacts.csv](artifacts.csv). The cycle count includes boot, output
zeroing, and readback; it is not Gather kernel latency.

```sh
python3 kernels/evaluation/firesim/capture_cases.py \
  --plan kernels/evaluation/firesim/xrage-full/plan.csv \
  --out kernels/evaluation/firesim/xrage-full --require-complete
python3 kernels/evaluation/firesim/verify_cases.py \
  kernels/evaluation/firesim/xrage-full
```
