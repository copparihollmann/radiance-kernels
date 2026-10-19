# Radiance workloads

This package contains the binaries, input images and output checks for 43
workload configurations tested in Cyclotron. Use `manifest.json` to locate the
files and addresses for each case. Files under `blobs/` are named by SHA-256 and
shared between cases that use the same data.

## Included configurations

| Workload | Cases | SM count | Execution |
|---|---:|---|---|
| SmolVLA, TinyLlama, DeepSeek 1.5B, Gemma 2 2B | 8 | 1, 2 | One model replica per SM |
| STREAM: copy, scale, add, triad | 12 | 1, 2 | One-SM partition controls and concurrent two-SM runs |
| Spatter: gather, scatter, gather/scatter, multiscatter, multigather | 15 | 1, 2 | One-SM partition controls and concurrent two-SM runs |
| xRAGE gather and scatter | 6 | 1, 2 | Original-size inputs; partition controls and concurrent two-SM runs |
| FP8 GEMM, 512 × 512 × 512 | 2 | 1, 2 | Separate one- and two-SM cases |

The model cases include every layer and the frozen test inputs. The decoders
use their recorded short prefill/decode sequences; SmolVLA includes the camera
inputs and full denoising loop. The two-SM model cases run two independent
replicas. Four-SM workloads and model sharding are not included.
Model computation stays on core 0; core 1 waits. The v3 decoder binaries
schedule one warp per stage, while SmolVLA schedules four. The newer decoder
candidates also schedule four warps. The current buffer handoff does not
provide coherence between the private core caches. Replica runs exercise
throughput, not single-inference scaling.
Four-SM generation and qualification are being added in a separate revision;
the v3 archives keep their existing one-/two-SM contents.

For STREAM, Spatter and xRAGE, names ending in `--1` or `--2` are the original
one-SM controls with one or two logical partitions. Names ending in `--2sm`
run concurrently on two real SMs, with one partition per SM. Ordered scatter
keeps every write to a destination on its owning SM and preserves last-writer
order within that destination.

## FireSim targets

The Chipyard `graphics` branch was checked at
`30d64bf63d637c049b6661cb64efd3cdb939d093`. It defines these targets in
[`TargetConfigs.scala`](https://github.com/ucb-bar/chipyard/blob/30d64bf63d637c049b6661cb64efd3cdb939d093/generators/firechip/chip/src/main/scala/TargetConfigs.scala).
The package prefix is `firechip.chip`.

| SMs | Standard target | HBM target |
|---:|---|---|
| 1 | `FireSimE4M3MxGemminiRadianceConfig` | `FireSimE4M3MxGemminiHBMRadianceConfig` |
| 2 | `FireSimE4M3MxGemminiTwoClusterRadianceConfig` | `FireSimE4M3MxGemminiTwoClusterHBMRadianceConfig` |
| 4 | `FireSimE4M3MxGemminiFourClusterRadianceConfig` | `FireSimE4M3MxGemminiFourClusterHBMRadianceConfig` |

These Radiance targets select `e4m3SingleNoLutMxFPConfig`, which has no VPU.
All four model binaries in this package issue VPU commands. Running them
therefore requires a Radiance image with a compatible VPU implementation.
The separate `FireSimMxE4M3VpuGemminiRocketConfig` uses Rocket/RoCC rather than
the Radiance Muon execution path.

The branch pins Radiance at `dd1a74f`, Gemmini at `7c02748`, and FireSim at
`b084672`. Enabling the VPU in that Radiance shared-memory configuration fails
RTL elaboration with `VPU needs a dual-ported SyncReadMem bank`. The source
targets establish the available SM counts; they do not resolve this interface
issue. `firesim/graphics-targets.json` records the pins and compatibility map.

The HBM variants use a 2 GiB GPU region split into four 512 MiB slices, with
32-byte address hashing. The host has a separate memory region/channel.
Match the image-loading method below to the target and loader on the FPGA
machine. The operator also needs the built bitstream, HWDB entry, runtime
configuration and board driver.

## Unpack and verify

Use Python 3.11 or newer. Output checking also needs NumPy.

```sh
sha256sum -c radiance-workloads-v3.tar.zst.sha256
tar --zstd -xf radiance-workloads-v3.tar.zst
cd fpga-handoff-v3
python3 workload_bundle.py verify
python3 workload_bundle.py list
```

Allow 12 GiB for the extracted package and another 2.5 GiB for each staged
large model. No compiler or model download is needed to use the supplied
binaries. Build information and source snapshots are referenced in the
manifest. Paths inside historical logs refer to the original build machine;
the package commands use paths relative to this directory.

## Stage a workload

For a bus loader that applies the target's address hash, or for an unhashed
target, use logical addresses:

```sh
python3 workload_bundle.py prepare tinyllama-1sm \
  --loader logical --output stage-tiny
```

For direct FireSim LoadMem on the HBM targets above, prepare the physical hash
layout before loading:

```sh
python3 workload_bundle.py prepare tinyllama-1sm \
  --loader direct-hashed --hash-size 0x80000000 \
  --hash-unit 32 --hash-slices 4 --output stage-tiny
```

The output directory contains `boot.elf`, its verification receipt,
`dump-plan.json` and `firesim-workload.json`. Install the ELF and workload JSON
together in the site's FireSim workload directory, then select the matching
HWDB entry in its runtime configuration. The boot ELF embeds every required
input image, including the checked FP8 initialization patches.

Do not pass a pre-hashed ELF through a loader that applies the hash again.
For other memory layouts, use the actual bitstream's mapping.

A direct board loader can instead use `device_elf`, `loads`, `smem_init` and
`nightly_host_handshake` from the manifest. Load the device ELF and all images
before releasing reset. Respect each ELF segment's file and memory sizes,
including zero-initialized BSS. GPU addresses are logical 32-bit addresses;
their CPU alias is `0x100000000 + address`.

## Completion and readback

The RV64 host carriers release the GPU and handle the mailbox GO protocol.
They need a completion/readback adapter in the board runner. These kernels
park secondary cores after the final barrier; core 0 executes the test
`tohost` instruction. Waiting only for `ALL_FINISHED` can therefore hang.
Observe core-0 completion in every configured cluster before reading the output.

For nightly GEMM, the carrier waits for both per-core END stamps in every
cluster before writing GO to slot 24. The checker requires the full 64-byte
verification mailbox from each cluster.

After completion and the target's cache visibility/drain operation, read every
region in `dump-plan.json` into a results directory using the listed filenames.
Create the `tensors/` subdirectory for decoder outputs. GPU readback addresses
use the CPU alias above; GEMM mailbox addresses are already host MMIO addresses.

```sh
python3 workload_bundle.py check tinyllama-1sm \
  --results results > check-result.json
```

The command returns nonzero for missing or truncated files and failed
comparisons. It checks every decoder stage and replica, exact integer tokens,
complete HPC output hashes, and GEMM output plus mailboxes. SmolVLA checks all
1,600 actions against both format and original-checkpoint references at
`atol=rtol=0.001`, along with the requested arena guards. Decoder float checks
use the tolerances recorded in each manifest entry. GEMM retains its original
1% Frobenius-error limit, at most 1% bad values, and no poison or nonfinite data.

Keep the driver/console log, termination result, readbacks, `check-result.json`,
boot ELF hash, bitstream/HWDB revision and clock settings with each run.
Measure GPU-active time separately from loading and host readback.

## Debug symbols and software cycle counts

The separate `radiance-workloads-debug-v3.tar.zst` companion covers all 43
configurations. Unpack it next to the base bundle. From the directory containing
the archives:

```sh
sha256sum -c radiance-workloads-debug-v3.tar.zst.sha256
tar --zstd -xf radiance-workloads-debug-v3.tar.zst
cd fpga-debug-v3
sha256sum -c SHA256SUMS
python3 debug_symbols.py bundle --base-bundle ../fpga-handoff-v3 --debug-bundle .
```

Every debug ELF preserves the qualified entry point and loaded bytes, addresses,
segment flags and BSS extents. Use it as the GPU symbol file while loading the
base package's boot ELF and inputs. Source-line tables and function/data symbols
are included; local-variable debug information is not. Runtime-library code may
have symbols without source lines. The companion includes generated sources,
model stage-address maps and build receipts.

`software-cycle-baseline/cycles.csv` contains the 35 completed HPC timing cases.
These are full simulator-run totals at the documented 1:1 software clock ratio,
including runtime overhead. The eight AI timing cells remain empty until their
unassisted full-model timing runs complete. Helper-assisted ticks do not enter
the table. These results do not establish FPGA latency.

## Test results

All four complete model graphs passed on one and two SMs in functional
Cyclotron. The simulator accelerated selected fallback math through checked
helpers; the binaries retain the device implementations for hardware execution.
Scheduling, barriers and MX/VPU commands ran through the ISA path.

The original 26 HBM/HPC checks and 11 additional concurrent two-SM HPC cases
passed without helper acceleration. The new cases require completion from both
clusters, HBM traffic from all four cores, and an exact complete-output match
against the unchanged upstream reference. One-SM negative controls, which leave
the second partition unwritten, are rejected by the same output checks. The v3
base archive retains failed receipts from a separate 4,096-launch stress test:
its journal recorded one entry per cluster. A subsequent logger fix restores
all 4,096 entries per cluster on one/two SMs; every output byte, modeled cycle
count and complete HBM report matches the earlier runs. The original failed
receipts remain unchanged.

The package has not been run on FPGA or in a complete FireSim SoC. Its results
do not establish hardware latency. Per-model MX coverage, FP32 fallbacks,
reference comparisons and source hashes are in the manifest's evidence records.
