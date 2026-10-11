# LLM workload inputs and model schedules

## PR #1 model stitching

[`pr1-models.json`](pr1-models.json) pins the four models named in
[radiance-kernels PR #1](https://github.com/ucb-bar/radiance-kernels/pull/1):
TinyLlama-1.1B, DeepSeek-R1-Distill-Qwen-1.5B, Gemma-2-2B, and SmolVLA-base.
The decoder dimensions and SmolVLA policy settings come from the linked,
revision-pinned model configs. Gemma's gated checkpoint is available locally;
its config, tensor shapes, and shard hashes are checked against the pinned
revision in the [checkpoint binding record](../../model_chain/evaluation/gemma-checkpoint-binding.json).

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
contract. The [checkpoint execution controls](../../model_chain/README.md)
subsequently checked all three decoder graphs against upstream models.

SmolVLA's graph starts after the policy's image resize/pad to `512×512`,
state pad to 32 features, and language tokenization up to 48 tokens. It covers
three image inputs, patch embedding, vision encoding,
the connector, language and state inputs, 16 VLM layers with per-layer prefix
K/V caches, and 10 action denoising iterations with 16 expert layers each.
The VLM and expert layers expose their RMSNorm, Q/K/V projection, RoPE,
masked attention, output projection, residual, and gated FFN stages. Prefix
K/V is captured after projection and K rotation. Cross-attention projects the
stored prefix K/V into the expert space; self-attention appends temporary
action K/V. The logical graph uses the checkpoint's 960-wide VLM stream,
720-wide expert stream, 15 query heads, 5 KV heads, and 64-element head width.
The JSON `execution_schedule` records the loop boundaries as well as the
flattened dependencies. One `sample_actions` call builds the camera embeddings
and prefix cache once. It then runs 10 sequential denoising iterations. Each
iteration evaluates 16 ordered expert layers, alternating self- and
cross-attention, and carries the updated 50-token action tensor into the next
iteration. The timestep is a generated constant `1 - step/10`, and the Euler
update uses `-1/10`. The prefix K/V tensors remain read-only throughout those
iterations. Each camera's 12 SigLIP vision layers are also decomposed into
LayerNorm, bidirectional attention, MLP, and residual stages. The connector
has distinct pixel shuffle and projection stages. The subsequent
[connected checkpoint build](../../model_chain/README.md) and full FP32
host comparison cover this graph.
At the policy boundary, `select_action` refills its queue with one 50-action
chunk when empty, then returns one action per call, unpadded from 32 to 6
features. These queue operations are schedule metadata, outside the graph's
`sample_actions` scope. The pinned configuration has no real-time chunking
configuration; that alternate path is not represented.
It keeps the prefix padding mask, attention mask, and position IDs as distinct
logical tensors; each denoising step has a `[batch, action tokens, prefix + action
tokens]` attention mask. Self-attention concatenates action K/V for its local
attention call while later steps read the same prefix cache. Cross-attention
reads that prefix cache directly. The last Euler update reaches time 0.
It marks the bidirectional vision attention, pixel shuffle, multimodal masks,
expert attention, and action operations as missing device stages. Its fixed
512-pixel and maximum-token shapes are
planning assumptions from the pinned configs; the exact-input FP32 checkpoint
comparison and its BF16 precision difference are recorded in the
[connected build](../../model_chain/README.md). Image and language embedding
scale stages are explicit. The mask
tensors in this initial graph inventory encode required rank and dependencies;
the later checkpoint run tests their implementation and numerical behavior.
The branch and cache-lifetime interpretation is checked against the pinned
policy and backbone configs plus [LeRobot v0.5.1 sources](smolvla-implementation.json).
That implementation concatenates suffix K/V for a self-attention call without
replacing the stored prefix cache. The graph therefore keeps the prefix cache
as the input to every denoising step. The later full FP32 host comparison
provides the graph-to-checkpoint numerical check.

The [SmolVLA checkpoint binding check](smolvla-checkpoint-bindings.json) reads
the real pinned `model.safetensors` header. Of 500 tensors, 499 bind to
explicit graph stages; the language-model head is unused for action inference.
Those 499 parameters have 2,272 stage uses across the camera and denoising
loops.
Names and shapes match the checkpoint; this is a static binding check, not
numerical execution.

The graph has 3,673 stages. In the initial PR #1 primitive inventory, existing
standalone kernels covered 2,719 stages and 954 had no matching primitive.
The JSON preserves that inventory by operator. The subsequent connected ELF
implements all 3,673 stages with scalar SIMT device code; its full Cyclotron
run remains incomplete after a 24-hour wall limit.

Regenerate the binding and gap record with:

```sh
python3 kernels/evaluation/llm/verify_smolvla_checkpoint.py \
  --checkpoint-dir /path/to/lerobot/smolvla_base/snapshot \
  --out kernels/evaluation/llm/smolvla-checkpoint-bindings.json
```

The [action embedding control](smolvla-action-pytorch-results.json) executes
the six real action/time embedding tensors for one 50-token chunk at time 1.
The separate NumPy and PyTorch calculations agree at every stage; the largest
absolute error is `1.08e-6`. The [NumPy result](smolvla-action-results.json)
records stage hashes and confirms that changing either time or action input
changes the output. This follows the pinned `embed_suffix` formula without
instantiating the full LeRobot policy. It does not validate vision, expert
attention, ten-step denoising, or a Radiance executable. A separate
[upstream policy run](smolvla-policy-results.json) strictly loads the pinned
checkpoint in LeRobot 0.5.1 and produces one 50-action chunk with three
deterministic, preprocessed camera images, 48 language token IDs, padded state,
and fixed noise. It calls the vision model and connector three times each,
then the action projection ten times. It produces a `[1,50,32]` action tensor.
The independent NumPy action embedding agrees with the actual upstream
`embed_suffix` method to `1.08e-6` maximum absolute error. This run supplies a
reproducible software golden output. It does not establish that the stitched
graph or Radiance computes the same values, and the inputs are synthetic rather
than a robot observation. Runtime hooks also match the graph's order and input
shapes for 36 vision-layer calls, 16 VLM-layer calls, and 160 expert-layer
calls. This checks loop structure, not intermediate numerical values.
Reproduce it with the pinned checkpoint and a Python
environment containing `lerobot==0.5.1` and `transformers==5.3.0`:

```sh
python3 kernels/evaluation/llm/verify_smolvla_policy.py \
  --checkpoint-dir /path/to/lerobot/smolvla_base/snapshot \
  --out kernels/evaluation/llm/smolvla-policy-results.json
```

The checkpoint's language embedding is BF16. A second
[policy run](smolvla-policy-fp32-results.json) casts the same pinned policy to
FP32 before sampling, matching the connected Radiance kernel's weight and
arithmetic path. Both runs use byte-identical inputs and the same checkpoint.
The [precision comparison](../../model_chain/evaluation/smolvla-precision-comparison.json)
finds a `0.0263` maximum action difference; 405 of 1,600 actions differ by
more than `atol=1e-3, rtol=1e-3`. This quantifies the precision change rather
than treating the two policy outputs as interchangeable.

The action-only PyTorch check uses an optional local environment with `torch`,
`safetensors`, and `numpy`:

```sh
python3 kernels/evaluation/llm/smolvla_action_reference.py \
  --checkpoint-dir /path/to/lerobot/smolvla_base/snapshot
python3 kernels/evaluation/llm/verify_smolvla_action.py \
  --checkpoint-dir /path/to/lerobot/smolvla_base/snapshot
```

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
check checkpoint fidelity, MX quantization, or device execution. SmolVLA is
outside this small decoder reference; its full vision and action path is
compiled under [`kernels/model_chain`](../../model_chain/README.md).

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
do not validate MX quantization or produce device timing. The pinned Gemma
checkpoint is now available locally: its [26-layer reference check](../../model_chain/evaluation/gemma-checkpoint-full-reference.json)
matches a Transformers eager-attention pass to `3.67e-5` maximum logit error.
SmolVLA has full generated C++ stages and a Radiance ELF; complete device
output validation is still pending.

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
with shared activation/KV buffers, prefill, and cached decode. The
[full-depth reduced builds](../../model_chain/evaluation/full-depth-functional-results.json)
execute 22, 28, and 26 synthetic decoder layers, respectively. Native checks
compare every floating-point stage with this directory's NumPy executor;
all three full-depth reduced builds pass Cyclotron's functional device execution.
One-token TinyLlama prefill followed by one cached decode step has passed VCS
RTL. Full-dimension checkpoint ELFs have since compiled for TinyLlama,
DeepSeek, and quantized Gemma; their full-depth one-token prefill/decode
Cyclotron runs [passed every stage](../../model_chain/README.md) against the
mapped-precision references. The full 3,673-stage SmolVLA checkpoint ELF also
compiled; its device action comparison remains incomplete after a 24-hour
wall limit. MX-Gemmini integration
and timed full-model runs remain open. Only timed device runs can produce
end-to-end latency, utilization, cache, or memory measurements.

The full-depth [host checks](../../model_chain/README.md) now execute the
generated Radiance stage C++ with pinned checkpoint images for all three
decoders, checking every stage against the mapped-precision NumPy reference.
SmolVLA's full host run checks all 1,600 actions against the FP32 upstream
policy. Its [final Euler stage](../../model_chain/evaluation/smolvla-final-euler-functional-results.json)
also passed Cyclotron with real intermediate tensors from that host run.
The full-depth decoder Cyclotron runs have completed. SmolVLA has passing
targeted Cyclotron controls, while its complete action-chunk device check is
still open.

This directory records candidate workload inputs for an initial Radiance
performance evaluation. It contains no timed LLM performance measurements. The
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
