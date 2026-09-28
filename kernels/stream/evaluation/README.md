# STREAM evaluation

Each operation ran independently on 1,048,576 float32 elements using the
`RadianceSingleClusterConfig` Cyclotron timing model. Every run passed a
complete output digest and guard check. The simulator used the same Radiance
checkout and generic DRAM timing configuration described in
`../../spatter/evaluation/README.md`. The model cycles are relative timing
results; no GPU clock or calibrated HBM bandwidth is available.

| Operation | Model GPU cycles | Logical payload bytes | Model global-memory bytes issued | Logical bytes/cycle |
| --- | ---: | ---: | ---: | ---: |
| Copy | 3,987,096 | 8,388,608 | 8,412,416 | 2.104 |
| Scale | 4,024,544 | 8,388,608 | 8,412,416 | 2.084 |
| Add | 6,346,569 | 12,582,912 | 12,606,720 | 1.983 |
| Triad | 6,346,349 | 12,582,912 | 12,606,720 | 1.983 |

[Model results](current-build-model-results.csv) and
[memory counters](current-build-memory.csv) retain the exact values. Logical
payload counts source reads and destination writes. Model memory bytes include
cache-line and transaction effects. These results are for four independent
ELFs; they are not a measurement of a sequential Copy→Scale→Add→Triad run.

The small 256-element cases validate the same four kernel paths on the full
SoC RTL simulator and on Cyclotron. Their paired cycle and correctness results
will be recorded in `current-build-smoke-results.csv` after all four finish.
