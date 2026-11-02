# Qwen3.5 Gated DeltaNet CPU reference

This directory is an independent, deterministic CPU oracle for the recurrent
state update in Qwen3.5's Gated DeltaNet. It has no PyTorch, checkpoint,
network, or device dependency.

## Pinned architecture sources

The representative model is `Qwen/Qwen3.5-4B` at official checkpoint revision
[`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a).
Its pinned
[`config.json`](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json)
is the source for dimensions. The Hub's pinned
[`safetensors` metadata](https://huggingface.co/api/models/Qwen/Qwen3.5-4B/revision/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)
reports exactly **4,659,865,088 parameters** (4,659,861,248 BF16 and 3,840
FP32) for the complete multimodal checkpoint. The model is marketed as 4B;
the exact Hub count includes the vision stack.

The recurrence and gate equations are pinned to the Qwen/Hugging Face source
at Transformers revision
[`5f6629d89508568afaadf5d60cace11ec0edebbc`](https://github.com/huggingface/transformers/blob/5f6629d89508568afaadf5d60cace11ec0edebbc/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L278-L283):
the sequential
[`torch_recurrent_gated_delta_rule`](https://github.com/huggingface/transformers/blob/5f6629d89508568afaadf5d60cace11ec0edebbc/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L423-L483)
and the
[`Qwen3_5GatedDeltaNet` wiring](https://github.com/huggingface/transformers/blob/5f6629d89508568afaadf5d60cace11ec0edebbc/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L490-L650).

Representative Qwen3.5-4B text dimensions are:

| Quantity | Value |
| --- | ---: |
| Hidden width | 2,560 |
| Decoder layers | 32 |
| Layer pattern | 3 DeltaNet + 1 full attention, repeated 8 times |
| DeltaNet Q/K heads | 16 |
| DeltaNet value heads | 32 |
| Q/K head dimension | 128 |
| Value head dimension | 128 |
| Expanded recurrent state per DeltaNet layer | `[32, 128, 128]` FP32 = 524,288 floats (2 MiB) |
| Causal-convolution width | 4 |
| MLP intermediate width | 9,216 |

## Implemented equation and layout

Q/K heads are repeated from 16 to 32 before this interface, exactly as in the
pinned source. For each expanded value head and token, the oracle computes:

```text
q = q / sqrt(sum(q*q) + 1e-6) / sqrt(128)
k = k / sqrt(sum(k*k) + 1e-6)
beta = sigmoid(b)
g = -exp(A_log) * softplus(a + dt_bias)
S = exp(g) * S
delta = (v - k^T S) * beta
S = S + k outer delta
o = q^T S
```

The canonical state layout is row-major `[head][key][value]`, and state is
FP32. The streaming implementation is checked at every token against a
separately indexed full-sequence implementation, including all gate values,
outputs, and state elements. Tests cover a small diagnostic shape, the full
`[32,128,128]` recurrent shape, and a prefill plus one cached decode step.

Run:

```sh
make test
```

Native comparison uses
`abs(a-b) <= 2e-5 + 2e-5*max(abs(a),abs(b))` and rejects non-finite values.

## Deliberate exclusions

This is not full Qwen or a complete DeltaNet layer. It excludes input/QKV/a/b/z
and output projections, the width-4 causal convolution and SiLU, gated
RMSNorm, full-attention layers, MLPs, embeddings, vision, batching and masks,
checkpoint loading/equivalence, training, quantization, and performance
claims. The synthetic vectors establish only the stated primitive and
numerical contract.
