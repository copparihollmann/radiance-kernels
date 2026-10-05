# Gather count-chunk smoke test

This is a small control for the software decomposition used when a Gather's
original source array is too large for one Muon ELF. The
[input](../../../spatter/chunk-smoke.json) has five repetitions, three pattern
positions, `delta=3`, and `wrap=1`. One ELF executes all five repetitions.
Another executes only repetitions 3 and 4, with the source array generated
from the matching global indices. Since `wrap=1`, that final chunk overwrites
every dense output slot, so its final array must equal the unsplit run's.

The pinned upstream Spatter serial backend executed both ELFs' exact source
and pattern arrays, plus a first-chunk control. All three upstream output
digests match the generated Radiance expectations in the
[golden table](../../../spatter/evaluation/chunk-smoke-upstream.csv). The
unsplit and final-chunk digests are both `ccf6545c128b8c14`; the first
chunk's intermediate digest is `d57e7ef7d8db33fc`.

FireSim jobs 1697 and 1698 both passed complete output and guard checks on
the U250 for the unsplit and final-chunk ELFs. Their [plan](plan.csv),
[results](results.csv), and [raw artifact hashes](artifacts.csv) retain the
guest outcomes. This control validates the input rebasing on a small case.
It does not establish that the full 14,705,882-repetition AMG GPU trace ran,
nor can separate booted jobs provide its single-launch latency.

```sh
python3 kernels/evaluation/firesim/capture_cases.py \
  --plan kernels/evaluation/firesim/gather-chunking/plan.csv \
  --out kernels/evaluation/firesim/gather-chunking --require-complete
python3 kernels/evaluation/firesim/verify_cases.py \
  kernels/evaluation/firesim/gather-chunking \
  --golden-table kernels/spatter/evaluation/chunk-smoke-upstream.csv
```
