# Spatter on nightly HBM Radiance

This directory builds the five transfer families from the Spatter GPU STREAM
deck as separate 1-SM and 2-SM SoC ELFs: Gather, Scatter, GS, MultiScatter,
and MultiGather. It reuses the pattern parser, address plan, 64-bit payload,
and serial reference from [`kernels/spatter`](../../spatter/README.md). The
default build is smoke Gather; select a deck and case explicitly for a
different workload.

| GPU STREAM case | Transfer | Pattern length × count | Destination elements |
| --- | --- | ---: | ---: |
| 0 | Gather | 256 × 1,024 | 256 |
| 1 | Scatter | 256 × 1,024 | 262,144 |
| 2 | GS | 256 × 1,024 | 262,144 |
| 3 | MultiScatter | 256 × 1,024 | 262,144 |
| 4 | MultiGather | 256 × 1,024 | 256 |

The generator materializes the complete source and serial output in separate
GPU memory regions, with a zeroed destination. `spatter_body.hpp` is compiled
into the nightly kernel and the Cyclotron control. A cluster owns a disjoint
subset of output positions for Gather and MultiGather, and a disjoint subset
of transfers for nonoverlapping Scatter, GS, and MultiScatter. For repeated
scatter destinations, `--collision-policy ordered` groups transfers by
destination and retains serial last-writer order within each group. The
default parallel policy rejects repeated destinations before building.

The kernel stamps its end before checking the output. After all clusters
finish, every cluster compares every 64-bit output against the golden array;
the host checks 64 positions. The UART report includes each core's start and
end stamps, the whole-kernel GPU cycle interval, logical payload bytes, and
logical bytes per GPU cycle. The logical byte count is 8 bytes read plus 8
bytes written per transfer. It does not measure cache-line traffic. The
full-output checks and host reads are outside the timed kernel interval.

## Build and verify

Build one case at a time because the generated header and blob files are
shared between cases. Keep separate copies of each case's ELFs and
`data/spatter.json` before selecting the next case:

```sh
make -j1 SMS="1 2" GEN_ARGS="--suite ../../spatter/inputs/standard-suite/basic-tests/gpu-stream.json --case 0"
```

Run `--case 0` through `--case 4` to cover the original GPU STREAM deck.
`build/1sm/spatter.soc.elf` launches cluster 0; the 2-SM ELF launches both
clusters. The standard nightly build expects direct LoadMem into a 2 GiB GPU
address range hashed across four slices in 32-byte units and two cores per
cluster. Match the actual bitstream's topology, memory map, and loader before
running either ELF. `gen_data.py` caps generated blobs at 128 MiB by default;
raise `--max-layout-mib` deliberately for a larger trace.

[`evaluation/device-controls.json`](evaluation/device-controls.json) records
12 smoke runs, ten original-size GPU STREAM runs, and two fused GS composition
runs. Each used Cyclotron to
execute the shared device transfer body and compared the **entire** output
byte for byte with the serial reference. The two-partition control applies
the partitions sequentially on one simulated cluster; its cycle count is
not a two-SM performance measurement. For every original-size case, source
and golden SHA-256 values also equal the existing [pinned upstream Spatter
serial oracle](../../spatter/evaluation/upstream-golden-results.csv). The
oracle uses the unmodified Spatter serial transfer loops at revision
`ec8923711f8dc21eedff7189f12b02eb06845d2f`. The fused GS output also
matches the [independent upstream Gather→Scatter composition
oracle](../../spatter/evaluation/upstream-golden-composition-results.csv),
including its complete output SHA-256. This covers the `gather-final-wrap`
case where the Gather result is reused across Scatter repetitions.

[`evaluation/standard-gpu-stream-builds.json`](evaluation/standard-gpu-stream-builds.json)
records the ten SoC ELF hashes, exact input and golden hashes, and checks
that their GPU load segments fit the declared memory map without overlap.
These images compiled, but they have **not** been run on the October 11 XB10
1-SM or 2-SM bitstreams. Cyclotron functional cycles in the control receipt
do not establish FPGA latency, bandwidth, utilization, or cache behavior.

To reproduce the device checks, run `test_device.py` with an output directory
under the current task, a Cyclotron build, a RISC-V toolchain, and Muon LLVM.
The default uses all six smoke cases; pass `--suite` with the GPU STREAM JSON
deck for the five original-size cases, or with `inputs/composed-gpu-stream.json`
for the fused chain. `record_controls.py` checks and combines all three sets
of retained `results.json` files. `record_builds.py`
checks the staged SoC ELFs against the upstream oracle before writing its
receipt.

After a real FireSim run, `collect_result.py` takes the exact staged ELF,
its matching `spatter.json`, the build receipt, UART log, FireSim exit code,
HWDB file, and workload file. It requires one PASS, exact full-output checks
from every cluster, consistent per-core cycle stamps, and all expected
hashes. A synthetic UART test covers failure cases; it is not a hardware
result. For example:

```sh
python3 kernels/nightly/spatter/collect_result.py \
  --case 0 --sm-count 1 --cores-per-cluster 2 \
  --bitstream-id "$HWDB_ENTRY" --elf "$STAGED_ELF" \
  --data-receipt "$CASE_DIR/spatter.json" \
  --build-receipt kernels/nightly/spatter/evaluation/standard-gpu-stream-builds.json \
  --hwdb-config "$HWDB_FILE" --workload-config "$WORKLOAD_FILE" \
  --uart-log "$UART_LOG" --firesim-exit-code "$FIRESIM_EXIT_CODE" \
  --out "$RESULT_JSON"
```

The original GPU STREAM deck covers five Spatter transfer families. It is
not the full xRAGE or other application trace collection. Generated patterns,
destination collisions, and reference coverage for those traces are
documented in [`kernels/spatter`](../../spatter/README.md). This nightly
mapping does not implement atomics; ordered collisions mean serial
last-writer semantics, which can differ from concurrent CUDA Scatter.

## LANL xRAGE asteroid cases

The [xRAGE build receipt](evaluation/lanl-xrage-builds.json) adds the two
original-size cases selected from LANL `datafiles/xrage/asteroid/spatter.json`:

| Source case | Transfer | Pattern length × count | Collision rule |
| --- | --- | ---: | --- |
| 5 (array index 4) | Gather | 8,368,968 × 1 | Parallel, with disjoint output ownership |
| 9 (array index 8) | Scatter | 6,664,304 × 1 | Ordered last writer per destination |

`extract_cases.py` streams those two objects from the pinned 517,377,170-byte
LANL input into one smaller suite. It verifies the original file's SHA-256
before writing and uses less than 20 MiB RAM. The suite and all generated
blobs belong in the current task's `$TMPDIR`; they are not checked into Git.
From the repository root:

```sh
python3 kernels/nightly/spatter/extract_cases.py \
  --source /path/to/lanl/spatter.json \
  --out "$TMPDIR/xrage5-and-9.json" --case 4 --case 8
cd kernels/nightly/spatter
make -j1 SMS="1 2" GEN_ARGS="--suite $TMPDIR/xrage5-and-9.json --case 0 --max-layout-mib 256"
```

Save both ELFs and `data/spatter.json` before changing `GEN_ARGS` to
`--case 1 --collision-policy ordered --max-layout-mib 256` for xRAGE9.
The local Muon installation used for the recorded ELFs required a one-word
stack stride, its bundled libc++ headers, and the available RV64 bare-metal
sysroot for headers. Set `LLVM_MUON` and `RISCV_TOOLCHAIN_PATH` to the actual
installations, then use these additional make variables for that toolchain:

```sh
make -j1 SMS="1 2" \
  LLVM_MUON="$LLVM_MUON" \
  RISCV_TOOLCHAIN_PATH="$RISCV_TOOLCHAIN_PATH" \
  RISCV_SYSROOT="$RISCV_TOOLCHAIN_PATH/riscv64-unknown-elf" \
  RISCV64_TOOLCHAIN_PATH="$RISCV_TOOLCHAIN_PATH" RISCV="$RISCV_TOOLCHAIN_PATH" \
  MU_LIBC_INCLUDE="$LLVM_MUON/include/c++/v1" MU_STACK_WORD_STRIDE=1 \
  KERNEL_MU_CFLAGS="-nostdinc++ -isystem $LLVM_MUON/include/c++/v1" \
  GEN_ARGS="--suite $TMPDIR/xrage5-and-9.json --case 0 --max-layout-mib 256"
```

The build receipt identifies each ELF by SHA-256; a fresh toolchain may emit
different bytes and needs its own validation receipt.
`record_lanl_builds.py` compares both cases' source and complete golden
hashes with the pinned upstream Spatter serial oracle, checks the one- and
two-partition Cyclotron device-body results, and verifies all four SoC ELF
load segments against the 2 GiB, four-slice, 32-byte hash map. The exact
one- and two-partition outputs passed byte-for-byte. These checks do not
measure FPGA latency or bandwidth, and no run on the October XB10 images is
claimed. xRAGE9's ordered serial result should not be compared directly with
Spatter's CUDA atomic-scatter throughput.
