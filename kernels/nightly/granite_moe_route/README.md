# Granite 3.1 MoE routing device milestone

This is a bounded Radiance component for the official
`ibm-granite/granite-3.1-1b-a400m-instruct` architecture (1,334,628,352
checkpoint parameters). It is not a full Granite layer or model.

The representative invocation uses 128 tokens, 32 experts, and top-8 routing.
It consumes synthetic deterministic FP32 router logits and produces:

- 1,024 selected expert IDs;
- 32 expert counts and 33 prefix offsets;
- 1,024 assignment IDs stably grouped by expert; and
- the 1,024-entry inverse assignment map.

All 3,137 uint32 words are compared exactly with the shared CPU oracle after
the timed kernel. Exact ties prefer the lower expert ID and expert groups retain
canonical token/rank order. Those are deterministic test conventions; upstream
PyTorch does not guarantee either convention for ties.

Selected-softmax weights, token payload movement, router and expert GEMMs,
weighted combination, checkpoint tensors, and end-to-end model execution are
out of scope. The component therefore establishes routing/scatter structure,
not BF16/MXFP8 checkpoint equivalence.

Build the one-cluster image with:

```sh
make SMS=1 RISCV64_TOOLCHAIN_PATH=/path/to/riscv-tools
```

The CPU oracle tests are in `kernels/granite_moe_reference`. Cyclotron must use
the unscrumbled `build/1sm/granite_moe_route.radiance.elf`; the fused
`*.soc.elf` is address-hashed for the matching FPGA configuration.
