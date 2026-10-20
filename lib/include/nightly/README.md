# Nightly RTL kernel headers

These headers support Muon + MxGemmini kernels on the "nightly" Radiance RTL. The kernels are in
`kernels/nightly`. Do not use them on the taped-out part.

## RTL

The nightly RTL is:

* Radiance: branch `main` (the split host/GPU L2, one DRAM channel per L2 slice, and the config
  `RadianceHBMConfig` (2 SMs)). `RadianceHBMConfig` hashes GPU memory over
  the 4 L2 slices at 32 B granularity, so a kernel ELF must be scrambled before `+loadmem` (see
  "Build and run"). Its Gemmini is the E4M3 MxGemmini: E4M3 inputs only, with SPAD_REQUANT
  (`fa_mxfp8_sr`) and the loop retire counter. MXFP4 kernels (`gemm_mx FMT=fp4`) run on
  `RadianceFP4HBMConfig`: the same memory system with an MXFP4-only Gemmini.
  The kernels need these fixes, which are on `main`:
  * Muon FPPipe: the shared CVFPU returns packets of different operation groups out of order.
    The fix tags the fp32-to-bf16 convert and allows one operation group in flight per pipe.
  * Muon SFUPipe: `fence.s` waits for the shared-memory queue of its own warp only.
  * CollectorNode: one request fires when all 16 lanes are valid (no per-source state).
  * GemminiTile requantizer input: a beat is valid only when its request fires.
* Radiance branch `firesim-debug` adds the Muon store fixes that `llama_e2e` needs on FireSim:
  * Coalescer: merge sub-word stores from several lanes into one word byte by byte (68c74ef).
  * L0d: per-byte dirty mask, write-backs as PutPartialData (579df5a).
  * Shared memory: forward a same-cycle write to a read of the same address (ac454ac).
* Gemmini: branch `firesim-hbm` (`gemmini-mx-cleanup` with these fixes):
  * Scratchpad: back-pressure shared-memory reads on room in the DMA queue.
  * ExecuteController: pop each operand read response when the mesh accepts it.
  * LoopMatmulStCSpad: the store's row step comes from its own loop, not the global bounds.
  * StoreController: a DRAM store may carry its own J bound in rs1[63:56] (MXFP4 C stores).
  * LoopMatmul: do not force the A/B load loops when they are idle.

## Headers

| Header | Contents |
|---|---|
| `pb.h` | printBuf mailbox, shared by the host and the GPU |
| `device.h` | GPU launch, per-core start and end timestamps, barriers |
| `cluster_id.h` | Four-SM physical cluster identity from the host mailbox |
| `host.h` | Host launch, result wait, integer helpers (the host Rocket has no FPU) |
| `mx.h` | MxGemmini commands: loads, loops, scale memory, requantizer, stores |
| `verify.h` | GPU-side comparison of a result tensor with a golden |

## Build and run

1. Use the muon LLVM toolchain with the stack word stride (`llvm/llvm-muon`).
2. In a kernel directory, run `make`. The build writes `build/1sm/<kernel>.soc.elf` and
   `build/2sm/<kernel>.soc.elf`.
3. Run both ELFs on `RadianceHBMConfig` with `+loadmem`. The 1sm ELF launches cluster 0 only.
   The host prints the mesh utilization and the result check.

The build scrambles the GPU load segments into the hashed DRAM layout (`MU_ADDR_HASH=1`, the
default; `soc/scramble_gpu_elf.py`). Use `MU_ADDR_HASH=0` only to load through TSI
(`RadianceHBMTSIConfig`) or to run on a config without the hash. A scrambled ELF fails on a config
without the hash.

## Four SMs

Use `make SMS=4` with the matching four-SM image. In the current RTL, both the
cluster-ID CSR and the cluster field of `mhartid` are one bit wide. SMs 2 and 3
therefore report the same IDs as SMs 0 and 1.

The four-SM build uses a separate runtime archive. Before releasing GPU reset,
the host writes `0xC1D00000 | cluster_index` to slot 7 of each cluster's printBuf
and sets LAUNCH. Startup uses that tag to place stacks and TLS; the scheduler
and kernel use it for work ownership. Both scheduler context selection and
worker entry must use `mu_cluster_id()`.

Custom loaders must write the tags before releasing reset at `0x41000000`.
Each slot is 64 bits wide; write the tag to both halves at host address
`0x40080038 + cluster_index * 0x100000`. The CSR fallback supports older hosts
on configurations with full IDs; it does not repair a missing tag on the
narrow-ID four-SM image. Include `cluster_id.h` in the device entry translation
unit only, and compile the startup runtime with
`-DMU_CLUSTER_ID_FROM_PRINTBUF=1`. One- and two-SM builds retain their existing
runtime archive.
