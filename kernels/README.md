# HPC kernels on Radiance

Scalar SIMT ports of a set of HPC benchmarks, written against the nightly
Radiance RTL. They build the same way as every other kernel here -- `make` in
the kernel's directory, output in `*.soc.elf` and `*.radiance.elf` -- with two
extra axes described below.

Each kernel directory has its own README covering what was ported, which
upstream revision it was pinned to, and how its work is divided across SMs.

| Kernel | Upstream | What it is |
|---|---|---|
| [btree](btree) | Vortex | B+Tree point and range lookup |
| [gaussian](gaussian) | Rodinia | Gaussian elimination, Fan1 + Fan2 |
| [gemv](gemv) | -- | BF16 matrix-vector product, warp per row |
| [hotspot3d](hotspot3d) | Rodinia | 3D thermal simulation, column sweep |
| [lbm](lbm) | Parboil | D3Q19 lattice Boltzmann, one timestep |
| [nearn](nearn) | Rodinia | Nearest neighbour, one distance per record |
| [pathfinder](pathfinder) | Rodinia | Dynamic-programming wall traversal |
| [sfilter](sfilter) | Vortex | 3x3 spatial filter |
| [spatter](spatter) | Spatter | Gather, Scatter, GS, MultiGather, MultiScatter |
| [spmv_jds](spmv_jds) | Parboil | Sparse matrix-vector product, JDS format |
| [stencil3d](stencil3d) | Vortex | 27-point box stencil |
| [stream](stream) | STREAM | Copy, Scale, Add, Triad |
| [stream_dot](stream_dot) | BabelStream | Dot product with a shared-memory tree |
| [streamcluster](streamcluster) | Rodinia | pgain, one gain evaluation per point |

## Presets: what hardware an ELF is for

A kernel does not name a cluster count, a drain protocol or an L1 mode. Those
are properties of the bitstream, so they travel together under one name in
[`hpc.mk`](hpc.mk):

```sh
make PRESET=2sm_ddr3_v2        # the default
make PRESET=4sm_pipe OCC=2     # 4 clusters, two warps per core
make print-preset              # what the current settings expand to
```

| Preset | Clusters | Memory |
|---|---:|---|
| `1sm_pipe`, `2sm_pipe`, `4sm_pipe` | 1, 2, 4 | 30-cycle latency pipe, **no DRAM model** |
| `2sm_ddr3`, `2sm_hbm` | 2 | first-generation real timing |
| `2sm_ddr3_v2`, `2sm_hbm_v2` | 2 | DDR3 QuadRank, HBM2 QuadChannel |
| `2sm_ddr3_v3`, `2sm_hbm_v3` | 2 | 64-byte transactions, incoherent 4-bank L1 |

One consequence is worth stating plainly, because it is easy to read the wrong
way: **the cluster count only varies on the latency-pipe presets, which have no
DRAM model at all.** An SM-scaling sweep and a real-memory timing arm cannot be
crossed. Any number that appears to be both is one or the other.

Generated data is partitioned for a cluster count, so changing the preset
regenerates it. If the two ever disagree, the kernel's `HPC_SM_COUNT`
static_assert fails the build rather than letting the ELF read another
cluster's share. [`hpc_presets.py`](hpc_presets.py) is the same table for the
data generators, which record the preset in their manifests;
`test_hpc_presets.py` asks `make` itself for each value so the two cannot
drift.

## ILP: how hard a kernel unrolls

Every kernel builds three binaries per preset:

```
kernel.radiance.elf        ILP 1, the baseline
kernel_ilp2.radiance.elf   ILP 2
kernel_ilp4.radiance.elf   ILP 4
```

Each is `kernel_impl.hpp` compiled at a different unroll factor, the same
arrangement [`vecadd`](vecadd) uses.

This is **not** vectorization, and calling it that will mislead you. The device
is `rv32im_zfinx_zhinx`: there is no vector ISA and no vector unit, and a
warp's 16 lanes already cover 16 consecutive elements, so unrolling buys no
wider access. What it buys is independent chains. These kernels sit at a few
percent of the arithmetic ceiling while nowhere near the bandwidth roof,
because a cluster runs out of outstanding line fills long before it runs out of
ALU. An unrolled body whose slots do not depend on each other is what moves
them; one that reuses the same registers across slots just trades a memory
stall for a WAR/WAW stall.

`mu_grid_stride<ILP>` in [`../lib/include/mu_partition.h`](../lib/include/mu_partition.h)
is the shared walk -- unrolled body, scalar tail, so no kernel has to make its
extent a multiple of anything.

## Dividing work across SMs

Every policy in `mu_partition.h` hands each cluster whole 64-byte lines, so a
line is never shared between two cluster-private L1s. `MuOwnedSpace` maps the
elements one cluster owns onto a dense `[0, count)` space, and the kernel walks
that with an ordinary grid-stride loop:

- `kContiguous` -- one run per cluster. Best locality, but a cluster only ever
  touches its own slice of the address space, so it cannot spread DRAM bank and
  rank pressure.
- `kCyclicLines` -- single lines round-robin. Every cluster walks the whole
  range.
- `kCyclicChunks` -- runs of N lines round-robin. The middle ground.

Three kernels keep their ownership explicit instead, and say so in their
source: `spmv_jds`, `sfilter` and `streamcluster` take boundaries from their
generator, which balances nonzeros or assigns whole rows rather than splitting
a flat index space.

## Measuring lane occupancy

Every "% of peak ops/cycle" figure divides by a ceiling that assumes all 16
lanes of a warp are live, and nothing in the ordinary counters measures that:
`minstret` retires once per warp however many lanes were active. Build with
`NIGHTLY_LANE_PROFILE=1` to sample the active thread mask instead. That build
measures occupancy and **not** performance -- it costs a CSR read, a popcount
and two adds per sample -- so take timing from the uninstrumented build.
`stream`'s Copy is the control: it has no divergence and must read 16.0.
