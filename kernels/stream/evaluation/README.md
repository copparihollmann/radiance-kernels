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
cache-line and transaction effects. The memory table also records GPU LSU
global load and store queue issue counts for each recorded run; those counts
are not separate DRAM read and write byte measurements. These results are for
four independent ELFs; they are not a measurement of a sequential
Copy→Scale→Add→Triad run.

The 256-element Copy, Scale, Add, and Triad cases all passed full-output checks
on both the full SoC RTL simulator and Cyclotron using identical ELFs. Their
RTL GPU cycles were 9,085, 8,434, 8,793, and 9,045 respectively. The paired
ELF hashes, checks, and exact cycle values are in
[smoke results](current-build-smoke-results.csv).

The four original-size full SoC Verilator cases use separate directories under
`../runs/rtl-full/`. Scale passed its RV64 host sample and guard check in
3,971,774 GPU cycles on the 16-thread simulator. Its identical-ELF Cyclotron
run passed the complete output digest in 4,024,544 model cycles. Copy is
active on the 8-thread simulator and Add on the 16-thread simulator; Triad is
queued. The earlier partial attempts are archived under
`../runs/rtl-full-lost-session-20260929/` with interrupted status. A later
Triad attempt was stopped during host CPU saturation and is retained
under `../runs/rtl-full-interrupted-load-20260929/` without a cycle result.
An 8-thread Copy attempt was stopped to switch to the verified 16-thread
simulator; its partial files remain under
`../runs/rtl-full-interrupted-switch-20260929/` without a cycle result.
Later Copy and Add attempts were stopped when host CPU pressure rose;
their partial files remain under
`../runs/rtl-full-interrupted-overload-20260929/` without a cycle result.
The [paired table](current-build-results.csv) records completed cases. As each
run completes, validate it against its build ELF and the complete Cyclotron
output readback, then update the paired table from `kernels/stream`. Pass only
completed case names in Copy, Scale, Add, Triad order. The current table is
reproduced by:

```sh
python3 ../spatter/tools/spatter-validate.py --build-root runs \
  --rtl-root runs/rtl-full --model-root runs/model \
  --output evaluation/current-build-results.csv scale-1048576
```

After all four finish, pass `copy-1048576 scale-1048576 add-1048576
triad-1048576` instead.

The same validator was rerun on all four existing 256-element paired checks
and reproduced the committed smoke CSV byte for byte.
