# Workload artifact inventory

This directory preserves the provenance behind the [reported workload
results](../WORKLOAD_RESULTS.md). The [STREAM and Spatter summary](STREAM_SPATTER_SUMMARY.md)
gives the PR-style methodology, utilization and latency metrics, workload
equivalence, results, and limits. `runs.csv` indexes every discovered run and
records its status, input and source hashes, simulator binary hash, output
digest, and cycle count. `artifacts.csv` indexes every file in the run roots
with its size and SHA-256. `raw-metadata.tar.gz` contains the indexed JSON,
logs, and model counter summaries, along with both CSV inventories.
`elf-snapshot.csv` indexes a second local path for every ELF.
`dependency-snapshot.csv` indexes local copies of the input JSON decks and
simulator binaries named by the run records.

The current snapshot indexes 194 run records and 818 files across seven
roots. It preserves 624 non-ELF files in the metadata archive. The 194 ELFs
remain in the run roots and have local
hardlinks under the ignored `elf-snapshot/` directory. The hardlinks preserve
the binaries if a run directory is removed without copying the data blocks.
Every ELF has a SHA-256 in `artifacts.csv` and `elf-snapshot.csv`. All declared
ELF and available input-suite hashes
matched when this snapshot was generated. The archive's contents were checked
against all 624 recorded file hashes.
`verify_artifact_snapshot.py` repeats that check and verifies the ELF and
dependency snapshots, the run JSON records, and input/simulator references.
The generator also checked 134 rows in the current-build cycle and memory
tables against the indexed result JSON status and cycle fields.
`verify_report.py` checks the 15 workload rows and two full-size composition
rows in `WORKLOAD_RESULTS.md`
against the model cycle and memory tables, then regenerates every table row
from its raw Cyclotron log and memory summary. It also checks build/model ELF
hashes, complete output digests and guards, and the exploratory status of
parallel xRAGE9 Scatter. The recorded timing-config and checker-source hashes
must match the retained files in `spatter/tools`. It also verifies the committed
[`workload-results.csv`](workload-results.csv), a consolidated table with cycle,
traffic, directional GPU LSU issue counts, correctness, and
input/source/ELF/timing/checker provenance for all
15 reported workloads.
For each completed full-size RTL case, it regenerates the paired RTL/model CSV
from the retained runs, checks the RTL log's finish signal and cycle count, and
rejects a missing paired row. It checks all nine workload and two composition
full-size RTL/model cycle summaries in `WORKLOAD_RESULTS.md` against those
paired rows, including the stated RTL host checks and workload cycle ratios.
It also regenerates all 22 small paired RTL/model cases across the STREAM and
Spatter smoke, LULESH app-trace, ordered-composition, current single-launch
chain regressions, and threaded Verilator
probe tables. These checks
verify the complete model output digest, build/RTL/model ELF hashes, and the
RTL finish signal and cycle count against the archived logs.
After intentional model-result changes, regenerate that table with
`python3 kernels/evaluation/verify_report.py --write-csv`.
The dependency snapshot covers 24 input/simulator path records (17 distinct
contents). It includes the 517,377,170-byte xRAGE input deck and the four
referenced simulator binaries. Hardlinks avoid copying data where permitted;
the sandbox required a verified local copy of the xRAGE deck. These snapshots
are ignored by Git. The three small upstream standard-suite decks used for
GPU STREAM, LULESH, and AMG are also tracked directly in
[`spatter/inputs`](../spatter/inputs/README.md).

| Root ID | Location |
| --- | --- |
| `spatter-current` | `kernels/spatter/runs` in this checkout |
| `stream-current` | `kernels/stream/runs` in this checkout |
| `prior-vcs` | `spatter-vcs-runs` in the Chipyard workspace |
| `prior-verilator` | `spatter-verilator-runs` in the Chipyard workspace |
| `prior-sampled-rtl` | `spatter-sampled-rtl-runs` in the Chipyard workspace |
| `prior-cyclotron` | `spatter-cyclotron-runs` in the Chipyard workspace |
| `prior-sampled-builds` | `spatter-sampled-builds` in the Chipyard workspace |

`passed` and `exploratory` rows in the result tables are completed runs.
`built`, `failed`, and `interrupted` rows in this inventory preserve the
history and should not be reported as completed performance results. No run
is marked `running` in this snapshot. Refresh the inventory after any new
simulation before reporting its results:

```sh
python3 kernels/evaluation/make_inventory.py --workspace /path/to/chipyard
python3 kernels/evaluation/snapshot_elfs.py --workspace /path/to/chipyard
python3 kernels/evaluation/snapshot_dependencies.py
python3 kernels/evaluation/verify_report.py
python3 kernels/evaluation/verify_artifact_snapshot.py
```

On 2026-09-29, five live simulator session handles disappeared after an
environment change. Their logs stopped advancing before the Verilog finish
marker. The partial runs, including the original `result-at-interruption.json`
files, are retained under `spatter/runs/rtl-lost-session-20260929/` and
`stream/runs/rtl-full-lost-session-20260929/`. Their indexed status is
`interrupted`; they contribute no reported GPU cycles. The new full-size RTL
runs use the already checked 8- and 16-thread Verilator binaries.
An additional full-size Triad attempt was stopped before completion when host
CPU pressure rose; its partial files remain under
`stream/runs/rtl-full-interrupted-load-20260929/` with interrupted status.
An 8-thread full-size Copy attempt was later stopped to use the verified
16-thread simulator; its partial files remain under
`stream/runs/rtl-full-interrupted-switch-20260929/` with interrupted status.
A concurrent 16-thread Copy restart and a later 8-thread Add attempt were
stopped when host CPU pressure rose; their partial files remain under
`stream/runs/rtl-full-interrupted-overload-20260929/` with interrupted status.

The archive is a compact raw-record snapshot, not a replacement for the
local ELF binaries or the input and simulator binaries.
These large snapshots are local and are not committed to Git; their CSV indexes are
committed. **Keep `elf-snapshot/` and `dependency-snapshot/` with this checkout
when preserving results for a paper. Git alone cannot restore the large ELF,
xRAGE input, and simulator binaries.** The hardlinks under `elf-snapshot/` also
share storage with the run roots, so deleting both removes the binaries.
The archive and manifests are a point-in-time capture. Refresh them after any
new run, then rerun both verifiers before extracting final numbers. A
`running`, `built`, `failed`, or `interrupted` record is
never a completed performance measurement.

The run index records
each ELF hash, input path and hash, source hash, simulator path and hash, and
the checker and timing-config hashes where available. These let a later
report distinguish current-branch measurements, older builds, complete
checks, and exploratory overlapping Scatter results.

The independent upstream Spatter golden records are kept separately in
[`spatter/evaluation`](../spatter/evaluation/README.md). The committed
comparison tables cover 16 deterministic cases, including one scaled AMG
mapping and one full-size composition. The committed artifact manifest hashes
all 135 locally retained golden files. Run
`python3 kernels/spatter/tools/verify-upstream-golden.py --upstream
/path/to/spatter` to verify them against the pinned upstream source. Keep
`kernels/spatter/golden-runs/` with the ELF and dependency snapshots when
retaining raw inputs and outputs for publication.

For a portable report archive, use
`export_report_bundle.py`. It verifies the report tables and both snapshot
manifests, then packages the raw metadata, ELF, input, simulator, and golden
files with Git bundles for this kernel branch and the pinned original Spatter
source. It requires all five full-size Spatter GPU STREAM, all four full-size
STREAM, and both full-size fused/materialized composition RTL cases to pass
before labeling an archive complete. Any
running or missing case requires `--draft`, which marks the archive
provisional. The resulting
archive stays local and is not pushed:

```sh
python3 kernels/evaluation/export_report_bundle.py \
  --upstream /path/to/spatter --output /path/to/report-bundle.tar.gz
```
