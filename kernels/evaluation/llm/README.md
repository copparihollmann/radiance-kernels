# LLM workload inputs and model schedules

## PR #1 model stitching

[`pr1-models.json`](pr1-models.json) pins the four models named in
[radiance-kernels PR #1](https://github.com/ucb-bar/radiance-kernels/pull/1):
TinyLlama-1.1B, DeepSeek-R1-Distill-Qwen-1.5B, Gemma-2-2B, and SmolVLA-base.
The decoder dimensions and SmolVLA policy settings come from the linked,
revision-pinned model configs. Gemma's official config is access gated here;
its fields are cross-checked against PR #1 and the existing Gemma kernels.

[`stitch.py`](stitch.py) emits a tensor dependency graph at model dimensions.
For TinyLlama, DeepSeek, and Gemma it orders embedding, every decoder layer,
Q/K/V projections, optional QKV bias, RoPE, per-layer KV append, causal GQA,
output projection, residuals, gated FFN, final norm, and LM head. It includes
Gemma's four-norm layer wiring, alternating local/global attention, attention
score cap, and final-logit cap. Prefill and each decode step consume the KV
buffers produced by the previous step. DeepSeek's pinned config has
`use_sliding_window=false`; its declared window is therefore not applied.
The default `teacher_forced` mode accepts fixed decode token IDs for checking;
`--generation greedy` adds an argmax stage from each pass's last logits to the
next pass's token input. The standalone PR #1 kernels have no argmax driver;
the reduced connected build below includes one.
The graph checks tensor definitions, positive shapes, and named kernel
directories before export.
The full-dimension tests keep query head, KV head, head width, sequence, and
cache axes distinct. Gemma-2 is a useful check: its attention output width is
`8 × 256 = 2048`, while the residual stream is 2304 wide; the output
projection bridges those widths. A flat element count alone would miss this
contract. The graph still needs checkpoint execution to establish numerical
equivalence, especially for Gemma.

SmolVLA's graph starts after the policy's image resize/pad to `512×512`,
state pad to 32 features, and language tokenization up to 48 tokens. It covers
three image inputs, patch embedding, vision encoding,
the connector, language and state inputs, 16 VLM layers with per-layer prefix
K/V caches, and 10 action denoising iterations with 16 expert layers each.
It keeps the prefix padding mask, attention mask, and position IDs as distinct
logical tensors; each denoising step has a `[batch, action tokens, prefix + action
tokens]` attention mask. Self-attention concatenates action K/V for its local
attention call while later steps read the same prefix cache. Cross-attention
reads that prefix cache directly. The Euler steps run from time 1 to 0.
It marks the vision encoder, multimodal connector, cross-attention, and action operations
as missing device stages. Its fixed 512-pixel and maximum-token shapes are
planning assumptions from the pinned configs; real image preprocessing,
padding, mask values, and action-expert behavior still need a checkpoint
reference. Image and language embedding scale stages are explicit. The mask
tensors in the graph encode required rank and dependencies;
they do not claim a mask implementation or numerical equivalence to LeRobot.
The branch and cache-lifetime interpretation is checked against the pinned
policy and backbone configs plus [LeRobot v0.5.1 sources](smolvla-implementation.json).
That implementation concatenates suffix K/V for a self-attention call without
replacing the stored prefix cache. The graph therefore keeps the prefix cache
as the input to every denoising step. This source-level check does not replace
a checkpoint run with real images, masks, and action outputs.

Run, for example:

```sh
python3 kernels/evaluation/llm/stitch.py --model tinyllama \
  --batch 1 --prefill 128 --decode-steps 4 --out /tmp/tinyllama-schedule.json
python3 kernels/evaluation/llm/stitch.py --model tinyllama \
  --batch 1 --prefill 128 --decode-steps 4 --generation greedy \
  --out /tmp/tinyllama-greedy-schedule.json
python3 kernels/evaluation/llm/stitch.py --model smolvla_base \
  --batch 1 --out /tmp/smolvla-schedule.json
python3 kernels/evaluation/llm/reference.py --model tinyllama
python3 kernels/evaluation/llm/verify_sources.py
python3 -m unittest discover -s kernels/evaluation/llm -p 'test_*.py'
```

[`reference.py`](reference.py) executes the three decoder graphs with small
dimensions and deterministic generated weights. It checks that a three-token
prefill followed by two cached decode steps gives the same final logits as a
single five-token causal pass. The test also changes a decode token to ensure
the output responds. This checks the software handoff and masks; it does not
check checkpoint fidelity, MX quantization, or device execution. SmolVLA has
only a topology graph because the PR #1 kernels do not implement its complete
vision and action path.

[`checkpoint.py`](checkpoint.py) binds graph parameters to their checkpoint
tensor names. With optional PyTorch, Transformers, and safetensors packages,
[`verify_checkpoint.py`](verify_checkpoint.py) runs real checkpoint weights
through the stitched NumPy graph and compares its last logits to an independent
Transformers full causal pass. For example:

```sh
python3 kernels/evaluation/llm/verify_checkpoint.py --model tinyllama \
  --checkpoint-dir /path/to/pinned/TinyLlama/snapshot --full
```

The recorded [checkpoint checks](checkpoint-results.json) use the pinned
TinyLlama and DeepSeek checkpoint revisions, three prefill tokens, and two
cached decode tokens. All 22 TinyLlama layers passed with maximum absolute
logit error `1.86e-5`; all 28 DeepSeek layers passed with error `3.67e-5`
against Transformers 5.9.0 in FP32. The file records the exact config and
weight hashes, tensor binding counts, tolerances, and software versions.
These checks establish the decoder dataflow for those two checkpoints; they
do not validate MX quantization or produce device timing. Gemma's checkpoint
is access gated here, and SmolVLA still needs a complete numerical backend.

| Model | Graph scope | Existing PR #1 primitives | Main device gaps |
| --- | --- | --- | --- |
| TinyLlama | 22 decoder layers, prefill and decode | MX GEMM/GEMV, RMSNorm-QKV, RoPE-QKV, d=64 causal GQA, SwiGLU | Shared activation/KV buffers, full-head tiling, embedding/final head, checked multi-layer launch. |
| DeepSeek-R1-Distill-Qwen-1.5B | 28 decoder layers, prefill and decode | MX GEMM/GEMV, QKV bias, RoPE, SwiGLU | d=128 attention, shared buffers/KV, full-head tiling and launch. |
| Gemma-2-2B | 26 decoder layers, alternating local/global attention | Gemma norms, GeGLU, score and logit caps, d=256 attention | Shared buffers/KV; Gemma attention output lacks an on-device numerical readback in PR #1. |
| SmolVLA-base | 3 cameras, 16 VLM layers, 16 expert layers × 10 denoising steps | Patch embedding, LayerNorm, GeGLU, bias and MX projections | Vision encoder, connector, multimodal masks, expert cross-attention, action flow, shared buffers. |

The `kernel` field in a schedule means a standalone primitive exists. It does
**not** mean that primitive accepts the preceding stage's output: PR #1
generates fixed inputs and a separate ELF for each kernel. The first device
milestone now has a [connected reduced decoder build](../../model_chain/README.md)
for TinyLlama, DeepSeek, and Gemma. It emits a single Radiance ELF per family,
with shared activation/KV buffers, one synthetic decoder layer, prefill, and
cached decode. Native checks compare every floating-point stage with this
directory's NumPy executor; all three reduced builds also pass Cyclotron's
functional device execution. One-token TinyLlama prefill followed by one cached
decode step has passed VCS RTL; multi-token prefill plus decode still needs an
RTL result, full-dimension tiling,
real checkpoint weights, and MX-Gemmini integration. SmolVLA lacks complete
vision and action numerical/device paths. Only completed device runs can
produce end-to-end latency, utilization, cache, or memory measurements.

This directory records candidate workload inputs for an initial Radiance
performance evaluation. It contains no measured LLM cycles. The
[model source table](inputs/sources.csv) pins five official
checkpoint revisions and SHA-256 hashes for their small `config.json` files;
the files are retained under `inputs/`. The weights, tokenizer, example
prompts, and quantization format have not yet been pinned or run.

Run `python3 estimate_kv.py` to regenerate [kv-working-set.csv](kv-working-set.csv).
The script verifies each local config against its recorded hash before
calculating BF16 resident KV bytes for a 256-token prompt at batch 1 and 8
and an 8192-token prompt at batch 1. For each attention layer it uses
`batch × min(context, sliding_window) × 2 (K and V) × KV heads × head_dim ×
2 bytes`. Full-attention layers use `context` instead of a window. It counts
only Nemotron's four attention layers; its Mamba state is omitted. The
`new_token_kv_write_bytes` column counts one new K/V pair per attention layer
and batch item. OLMo's config has `use_cache=false`, so its rows describe an
explicit proposed cached-decode mapping that still needs a reference check.

These are logical working-set sizes. They do not include weights,
activations, padding, allocator overhead, memory-transaction amplification,
cache reuse, controller behavior, or measured cycles. A larger L2 than the
checked U250's logical 512 KiB controller may improve locality within tiles
or across operations even though none of the full KV sets fits in that cache.
The full checkpoint fit and weight-streaming policy must be established
before using any row as an executable benchmark input.

## From existing kernels to one decoder layer

The first Qwen3 layer should execute this dependency chain on shared device
buffers, checking each boundary against a reference run from the pinned
checkpoint:

`RMSNorm → Q/K/V projection → RoPE + KV append → causal attention → output
projection + residual → RMSNorm → gate/up projection → SiLU(gate) × up → down
projection + residual`.

| Stage | Existing Radiance starting point | Work required for this workload |
| --- | --- | --- |
| Projections | [MX GEMM](../../gemm_mxgemmini/README.md) and [batched GEMV](../../gemv_batched_fp8_m128/README.md) | Tile Qwen3's 1024-wide projections; choose and record one checkpoint-to-MX quantization rule, including batch-1 decode. |
| Norm and rotary position | [Fused RMSNorm/QKV](../../rmsnorm_qkv_fused/README.md) and [fused RoPE/QKV](../../rope_qkv_fused/README.md) | Adapt the fixed TinyLlama/Llama shapes to Qwen3's 128-wide heads; write KV in the layout consumed by attention. |
| Attention | [MX GQA attention](../../flash_attention_mx_gqa/README.md) | Support Qwen3's 16 query heads, 8 KV heads and 128-wide heads, plus both prefill and one-token decode. The present kernel uses a fixed TinyLlama shape and its output is checked offline, so it is not an end-to-end Qwen3 pass. |
| MLP and residual | GEMM tiles and existing SIMT elementwise examples | Add Qwen3 `SiLU(gate) × up`, residuals, and exact intermediate checks. |
| Layer and sequence schedule | Standalone ELFs and the STREAM/Spatter host launch path | Reuse intermediate buffers, order all stages, avoid unnecessary host copies, append KV over four decode steps, and emit stage/total timing and traffic traces. |

The same chain can then be adapted to OLMo's local/full attention. Qwen3 MoE
needs router, token grouping, expert weight placement, expert GEMMs, and
weighted regrouping. Nemotron adds Mamba state updates and their boundary
with attention. The existing STREAM/Spatter kernels provide controlled memory
tests for the same hardware configurations; they do not substitute for these
model-specific stages. Full checkpoint and tokenizer outputs, numerical
tolerances after quantization, and end-to-end execution remain open.
