# Selecting portable workloads

`manifest.json` is the master record in an extracted workload package. Each
entry names its binaries, inputs, complete output checks, source archives and
qualification receipts. Keep the `blobs/` directory beside it. Blob names are
SHA-256 hashes; shared weights are stored once.

Use the included selector from the package directory:

```sh
python3 workloads.py list
python3 workloads.py list --kind full_model --sms 4
python3 workloads.py list --kind hpc --benchmark stream --sms 2 --format json
python3 workloads.py show tinyllama-2sm
python3 workloads.py plan --kind full_model --sms 2 --loader logical > selected.json
```

`list` shows every matching entry and its software status. `plan` selects passed
entries by default and emits JSON without starting a run or copying weights.
Use `--status any` to inspect plans for unqualified entries. Each job contains
`prepare_argv` and `check_argv` arrays, suitable for `subprocess.run(...,
check=True)`. Preparation writes the boot ELF, FireSim workload JSON and dump
plan. Board execution uses the operator's HWDB, runtime configuration and
completion/readback adapter; `board_run_argv` is null until that is supplied.

Choose `--loader direct-hashed` only for direct physical loading with the
recorded 2 GiB, four-slice, 32-byte HBM hash. Choose `logical` when the loader
applies the hash or the image is unhashed. Match this choice to the bitstream.

The metadata separates graph scope from hardware topology:

| Field | Meaning |
|---|---|
| `kind` | `full_model`, `layer`, `hpc`, or `control` |
| `model` | Checkpoint/model identifier for a full model or layer |
| `benchmark` | HPC family, such as `stream`, `spatter`, `xrage`, or `fp8_gemm` |
| `sm_count`, `num_clusters` | Physical SM count; these values must agree |
| `logical_partitions` | Software work partitions; does not select hardware |
| `software_qualification.status` | Completion/output result for this exact binary |
| `timing_qualification.status` | Separate native timing qualification |
| `fpga_execution_verified` | Whether FPGA execution has been verified |

The current all-SM suite has 12 full-model cases and 47 complete HPC cases.
It contains no layer-only workloads. A model's individual stage map is a debug
artifact, not a standalone layer executable. Names ending in `--1` or `--2`
are older one-SM HPC controls with one or two logical partitions; `--2sm`
and `--4sm` use physical two- and four-SM execution. Always select hardware
from `sm_count`, not from a filename suffix.

Models run independent complete replicas, one per SM, using core 0 for
computation. HPC cases partition one problem across the selected SMs. These
model binaries do not shard one inference across SMs. Completion/readback
requirements are in each entry's `completion_contract`.

Four-SM packages for the narrow-ID RTL include `smem_init_per_cluster`.
Custom launchers must apply common words and each physical SM's own words
before releasing GPU reset. Those per-SM tags also select stack/TLS placement
during startup. The included four-SM carriers perform these writes.
