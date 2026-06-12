# autocomp-generated Muon kernels

Kernel sources produced by the autocomp optimization loop for the Muon SIMT GPU.
These are the best kernels per problem; the full results, metrics, traces, and the
cyclotron analytical-model change are in the radiance repository under `autocomp/`
(branch `feat/radiance-autocomp`).

Headline kernels:
- `sol0_smem_manual.cpp` / `matmul_smem_DISCOVERED_bw6_2.42x.cpp` — shared-memory matmul
  (2.43x kernel-only over the naive baseline; RTL-confirmed).
- `muon_1_beam_iters8.cpp` — conv patch-embed (1.48x kernel-only).

Each `muon_<prob>_*.cpp` is the best discovered kernel for that problem. To build/run, substitute
into the matching `harnesses/muon/test<prob>` harness in the autocomp repo and run on cyclotron.
