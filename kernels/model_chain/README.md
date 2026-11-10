# Connected model kernels

A small three-stage device check that exercises tensor handoffs inside **one**
Radiance executable: `RMSNorm -> FP32 linear -> residual`. Each stage leaves
its output in a named GMEM buffer and the next stage reads that buffer, so the
kernel tests the handoff itself rather than any one operator. A failed
comparison sends a nonzero `tohost` result.

`gen_data.py` generates the weights and the reference outputs with numpy.
`check_layout.py` runs against the linked ELF and verifies that the buffers
landed where the host expects.

```sh
make                    # builds model_chain.soc.elf
make check-layout       # verifies the GMEM buffer layout of the built ELF
```

## What this is not

The build uses reduced control dimensions -- hidden width 32, FFN width 64,
head width 8, vocabulary 64, one layer -- with deterministic generated FP32
weights. It tests the software mapping and the code generation. It does **not**
execute a real checkpoint, the original model dimensions, MX quantization, or a
complete model workload. The reduction preserves stage order and the
grouped-query head ratio; it does not preserve every width ratio of the
original network.

The tooling that packs real checkpoints into this layout, and the end-to-end
decoder compiler that drives it, are not in this branch.

Generated ELFs and intermediate files stay local under `generated/`.
