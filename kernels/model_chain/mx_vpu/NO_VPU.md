# Running on an E4M3 image without a VPU

STREAM, Spatter, xRAGE and FP8 GEMM already use no Gemmini VPU. The four generated
model variants need a replacement for their BF16 residual and bias additions.
Their norms, softmax and other existing Muon fallbacks stay on Muon; selected FP8
matrix operations stay on MX-Gemmini.

`muon_bf16_add.hpp` implements the replacement on Muon. It rounds both operands
to BF16, adds them, then rounds the result to BF16, preserving the format reference.
It keeps the original helper's lane-0 writer convention, including partitioned
call sites. It emits no shared-memory/MMIO commands.

For a fresh compile, define `MX_MODEL_NO_VPU=1` in every generated device
translation unit that includes `mx_vpu/backend.hpp`. For existing generated
objects, compile `no_vpu_runtime.cpp` with the same device ABI and link it into
the model. Its strong `model_chain::mx_vpu_add` definition replaces the generated
weak out-of-line helper. Reusing the other objects avoids recompiling full model
graphs. Verify the final symbol is strong (`T` in llvm-nm) and inspect its code;
changing only a manifest flag does not remove VPU instructions.

The delivery replaces all 12 model variants (four models, on 1, 2 and 4 SMs). It retains the
profiling hosts, debug ELFs, input/checkpoint blobs, references and final four-SM
runtime. Output addresses and debug stage maps are regenerated after linking.
The active master manifest records `requires_vpu:false` and the Muon BF16 fallback.
It does not claim a fresh exhaustive full-model or FPGA correctness pass.

`test_muon_bf16_add.cpp` compares the new arithmetic against the original
libgemmini VPU backend, including tails, broadcasts and in-place output. Compile
it together with `runtime.cpp` and the pinned libgemmini archive using
`-ffp-contract=off`. The delivery also runs short RV32 controls on 1, 2 and 4 SMs with
Gemmini disabled, comparing all outputs to the separate numerical format reference.
