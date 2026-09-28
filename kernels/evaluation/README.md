# Workload artifact inventory

This directory preserves the provenance behind the [reported workload
results](../WORKLOAD_RESULTS.md). `runs.csv` indexes every discovered run and
records its status, input and source hashes, simulator binary hash, output
digest, and cycle count. `artifacts.csv` indexes every file in the run roots
with its size and SHA-256. `raw-metadata.tar.gz` contains the indexed JSON,
logs, and model counter summaries, along with both CSV inventories.
`elf-snapshot.csv` indexes a second local path for every ELF.
`dependency-snapshot.csv` indexes local copies of the input JSON decks and
simulator binaries named by the run records.

The current snapshot indexes 133 run records and 560 files across seven
roots. It preserves 427 non-ELF files in the metadata archive. The 133 ELFs
total 1,513,616,136 bytes; they remain in the run roots and have local
hardlinks under the ignored `elf-snapshot/` directory. The hardlinks preserve
the binaries if a run directory is removed without copying the data blocks.
Every ELF has a SHA-256 in `artifacts.csv` and `elf-snapshot.csv`. All declared
ELF and available input-suite hashes
matched when this snapshot was generated. The archive's contents were checked
against all 427 recorded file hashes.
The generator also checked 74 rows in the current-build cycle and memory
tables against the indexed result JSON status and cycle fields.
`verify_report.py` checks the 14 workload rows in `WORKLOAD_RESULTS.md`
against the model cycle and memory tables, including the exploratory status
of parallel xRAGE9 Scatter.
The dependency snapshot covers 19 input/simulator path records (16 distinct
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
`built`, `running`, and `interrupted` rows in this inventory preserve the
history and should not be reported as completed performance results. Some RTL
jobs are still running; their log hashes describe this snapshot and will
change when the jobs finish. Refresh the inventory and archive after they
complete:

```sh
python3 kernels/evaluation/make_inventory.py --workspace /path/to/chipyard
python3 kernels/evaluation/snapshot_elfs.py --workspace /path/to/chipyard
python3 kernels/evaluation/snapshot_dependencies.py
python3 kernels/evaluation/verify_report.py
```

The archive is a compact raw-record snapshot, not a replacement for the
1.5 GB of local ELF binaries, input decks, or simulator binaries. The binary
snapshots are local and are not committed to Git; their CSV indexes are
committed. The run index records
each ELF hash, input path and hash, source hash, simulator path and hash, and
the checker and timing-config hashes where available. These let a later
report distinguish current-branch measurements, older builds, complete
checks, and exploratory overlapping Scatter results.
