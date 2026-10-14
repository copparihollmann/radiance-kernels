# STREAM on nightly HBM Radiance

This directory runs the four independent FP32 operations already defined in
[`kernels/stream`](../../stream/README.md): Copy `C=A`, Scale `B=2*C`, Add
`C=A+B`, and Triad `A=B+2*C`. Each ELF runs one operation. The generator
imports the input and reference formulas from `kernels/stream/run.py`; it
initializes the destination with a NaN pattern and stores the complete
expected output in a separate GPU memory region. Unused inputs occupy one
four-byte sentinel rather than a full array.

The same device operation body is compiled for one or two clusters.
`nightly.mk` writes separate `build/1sm/stream.soc.elf` and
`build/2sm/stream.soc.elf` files. The two-cluster version splits element
indices between the clusters. The host waits for every cluster, reports GPU
cycles and logical bytes per cycle, then asks each cluster to compare every
FP32 output bit against the reference. Verification runs after the kernel's
end stamp and is excluded from the reported kernel interval. A 64-position
host readback checks the same output independently. Copy and Scale count 8
logical bytes per element; Add and Triad count 12. These counts omit cache-line
and hash amplification.

Build one case at a time because the generated header and data blobs are
shared between cases. For example:

```sh
make -j1 SMS="1 2" GEN_ARGS="--kind triad --n 1000000"
```

The generated `data/stream.json` records the operation, size, reference
source hash, input hashes, and full golden hash. The standard nightly build
uses a 2 GiB GPU hash range with four slices and 32-byte units for direct
LoadMem. Match the actual bitstream's memory map, loader, and cores per
cluster before using either ELF in FireSim. The 1-SM image launches cluster 0;
the 2-SM image launches both clusters. Neither has run on the October 11 XB10
bitstreams yet.

The [local control receipt](evaluation/device-control.json) covers all four
operations with 4,102 elements, including the tail after several grid strides.
Cyclotron executed the shared device body once
with one logical partition and once with two partitions applied sequentially;
all eight outputs matched the upstream FP32 reference exactly. This proves
the indexing and arithmetic for the small control. It does not model
concurrent two-SM traffic, cache behavior, utilization, or FPGA latency.
`test_device.py` reproduces the control when given an output directory,
Cyclotron checkout, RISC-V toolchain, and Muon LLVM toolchain. The four
one-million-element 1-SM and 2-SM candidate ELFs were also compiled with the
nightly build, and the [build receipt](evaluation/million-element-builds.json)
records their hashes. Every one-million-element golden SHA-256 also matched an
independently computed FP32 vector of the corresponding STREAM formula. The
ELFs still need RTL or FPGA correctness runs before
their host cycle and bytes-per-cycle reports can be used as performance data.

After a FireSim run, `collect_result.py` takes the staged ELF, its matching
`data/stream.json`, the build receipt, raw UART log, FireSim exit code, HWDB
file, and workload file. It requires one PASS, exact checks from every
cluster, consistent per-core cycle stamps, and the expected ELF/data hashes.
It rejects a wrapped 32-bit cycle interval. The HWDB file's hash records what
was supplied; the FireSim job record must still bind that file to the actual
bitstream. For example:

```sh
python3 kernels/nightly/stream/collect_result.py \
  --operation triad --sm-count 2 --cores-per-cluster 2 \
  --bitstream-id "$HWDB_ENTRY" --elf "$STAGED_ELF" \
  --data-receipt "$CASE_DIR/stream.json" \
  --build-receipt kernels/nightly/stream/evaluation/million-element-builds.json \
  --hwdb-config "$HWDB_FILE" --workload-config "$WORKLOAD_FILE" \
  --uart-log "$UART_LOG" --firesim-exit-code "$FIRESIM_EXIT_CODE" \
  --out "$RESULT_JSON"
```

`test_collect_result.py` exercises the parser with synthetic UART text,
including failed and incomplete reports. A local synthetic end-to-end
collector call was also checked against an actual compiled ELF and its data
receipt. Neither synthetic check is a hardware result.
