# Workload artifact inventory

This directory preserves the provenance behind the [reported workload
results](../WORKLOAD_RESULTS.md). `runs.csv` indexes every discovered run and
records its status, input and source hashes, simulator binary hash, output
digest, and cycle count. `artifacts.csv` indexes every file in the run roots
with its size and SHA-256. `raw-metadata.tar.gz` contains the indexed JSON,
logs, and model counter summaries, along with both CSV inventories.
`elf-snapshot.csv` indexes a second local path for every ELF.

The current snapshot indexes 127 run records and 535 files across seven
roots. It preserves 408 non-ELF files in the metadata archive. The 127 ELFs
total 1,513,397,448 bytes; they remain in the run roots and have local
hardlinks under the ignored `elf-snapshot/` directory. The hardlinks preserve
the binaries if a run directory is removed without copying the data blocks.
Every ELF has a SHA-256 in `artifacts.csv` and `elf-snapshot.csv`. All declared
ELF and available input-suite hashes
matched when this snapshot was generated. The archive's contents were checked
against all 408 recorded file hashes.
The generator also checked 66 rows in the current-build cycle and memory
tables against the indexed result JSON status and cycle fields.

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
```

The archive is a compact raw-record snapshot, not a replacement for the
1.5 GB of local ELF binaries or upstream input decks. The hardlink snapshot
is local and is not committed to Git; its CSV index is committed. The run index records
each ELF hash, input path and hash, source hash, simulator path and hash, and
the checker and timing-config hashes where available. These let a later
report distinguish current-branch measurements, older builds, complete
checks, and exploratory overlapping Scatter results.
