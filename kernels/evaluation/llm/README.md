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
The [greedy inference controls](../../model_chain/README.md#checkpoint-greedy-inference)
also connect full-depth prefill, argmax, KV carry, and two cached decode
passes for TinyLlama, DeepSeek, and Gemma. Their host-selected token IDs
match independent full-depth Transformers runs on the recorded inputs. A
full-width one-layer TinyLlama greedy run passes Cyclotron; full-depth greedy
device execution and timed inference remain open.
SmolVLA's full host run checks all 1,600 actions against the FP32 upstream
policy. Its [final Euler stage](../../model_chain/evaluation/smolvla-final-euler-functional-results.json)
also passed Cyclotron with real intermediate tensors from that host run.
The full-depth decoder Cyclotron runs have completed. SmolVLA has passing
targeted Cyclotron controls, while its complete action-chunk device check is
still open. The [bounded runtime diagnostic](../../model_chain/evaluation/smolvla-simulator-diagnostic.json)
records why the current one-warp scalar mapping is expensive to interpret and
what its short Cyclotron run establishes.

## Shape traces for the pinned models

Run `python3 profile_workloads.py --out-dir profiles` to regenerate the
[17-case trace index](profiles/index.json). The cases include TinyLlama,
DeepSeek, Gemma, and Qwen3 at each fixed decoder handoff shape, 128-token
prefill at batch 1 and 8, 1,024-token
prefill at batch 1, and one SmolVLA action-queue refill. Every trace records
stage order, tensor dependencies, shapes, and per-phase operand sizes. Decoder
prefill and cached decode remain separate phases. SmolVLA's trace retains
the one-time vision/VLM prefix, all ten denoising iterations, and the
50-action queue semantics; the listed stages represent one refill.

These are shape-derived traces. The operand byte totals sum tensor sizes at
stage boundaries using model logical dtypes and the current FP32/32-bit
software buffer mapping. They exclude scratch space and do not predict cache
misses or DRAM transactions. Each trace also binds to a packed-weight image
manifest, records its storage footprint, and counts tied address ranges once;
regeneration checks the pinned config hash but does not rehash the raw weight
bytes. The trace also counts named tensors alive at each stage. It starts with
graph inputs resident, retains all graph outputs, and allocates each stage
result before releasing its last input. It assumes distinct storage for every
live tensor. The peak is a software graph
footprint under those assumptions, not a measured L2 working set or an exact
device capacity requirement: in-place reuse can lower it, while scratch and
other buffers can raise it. The MAC totals cover linear stages only. The 128-
and 1,024-token cases are **not compiled device programs**. For example,
Gemma's largest mapped tensor in the long-context case is a
1,048,576,000-byte prefill logit
tensor. Together with its 2,019,513,344-byte packed image, even this lower
bound exceeds the 2 GiB target; even its 128-token, batch-1 prefill crosses
that bound. These shapes would need output tiling or another placement policy.
These traces identify software mapping work; they are not utilization,
latency, or performance results.

The [Qwen3 fixed-shape trace](profiles/qwen3_0_6b-fixed_handoff_shape.json)
matches the verified three-token prefill and two-step greedy schedule: 1,607
stages, 2.98 billion linear MACs, and a 1,192,230,912-byte packed FP16 image.
For batch 8 × 128 tokens and batch 1 × 1,024 tokens, the current graph
materializes all prefill logits. Its estimated peak named-tensor footprint
is about 862 MB in either case. The image plus that footprint is about
2.05 GB, leaving under 94 MB of a 2 GiB region before program, scratch, and
placement costs. These two shapes need a capacity check before device runs;
projecting only the last prefill token's logits is one candidate software
change, subject to an output-equivalence check.

The [last-logit graph profile](profiles/qwen3-prefill-last-logit.json) adds an
explicit `prefill.last_hidden` selection before Qwen3's tied LM head for
greedy generation. Decoder and KV dependencies stay the same. For a batch-1,
1,024-token prefill, it reduces that FP32 logit tensor from 622,329,856 to
607,744 bytes and the prefill LM-head work from 159.3 billion to 155.6
million MACs. A reduced NumPy control projects only the selected hidden row
and matches the last row of the full logits exactly, including the selected
token. A separate [full-width checkpoint check](../../model_chain/evaluation/qwen3-prefill-last-logit-check.json)
streams the packed FP16 tied table and reproduces the saved final prefill
logit row within `1.43e-5`, with the same top token. These are graph and
arithmetic checks. A
[connected reduced device comparison](../../model_chain/evaluation/qwen3-prefill-last-device-comparison.json)
now passes Cyclotron and finds exact equality for the selected prefill row
and two decode logit rows. The full-dimension checkpoint version still needs
compilation and device execution before the long-context savings can be
measured on Radiance.
The [final-token selection kernel](../../select_last_token/README.md) has
separately passed Cyclotron on the saved full-width Qwen3 hidden state, with
exact equality for all 1,024 selected values. Its full-model binding remains
to be compiled and checked.

This directory records candidate workload inputs for an initial Radiance
performance evaluation. It contains no timed LLM performance measurements. The
[model source table](inputs/sources.csv) pins five official
checkpoint revisions and SHA-256 hashes for their small `config.json` files;
the files are retained under `inputs/`. Qwen3's checkpoint hash, FP16 device
image, and one- and multi-token reference checks are now recorded above. The
other architecture candidates still need weight, tokenizer, prompt, and
precision controls before they can be treated as executable workloads.

Run `python3 inventory.py` to regenerate
[architecture-inventory.json](architecture-inventory.json). It verifies the
five config hashes and retains **each layer in execution order**, so an
alternating attention pattern is not flattened into a layer count. The counts
below are derived from those configs:

| Candidate | Ordered layer structure | Mapping consequence |
| --- | --- | --- |
| Qwen3-0.6B | 28 full-attention, dense-FFN layers | Dense prefill and cached decode baseline. |
| OLMo-3-7B | Repeat 3 sliding-attention layers then 1 full-attention layer, 8 times | Keep the 4,096-token local window distinct from the full-context KV layers. Its config disables caching by default; the reduced upstream control below checks an explicit cached-decode override. |
| Qwen3-30B-A3B | 48 full-attention layers with MoE FFNs | Route each token to 8 of 128 experts, then combine their outputs in layer order. |
| Nemotron-3-Nano-4B | 21 Mamba2 blocks, 4 full-attention blocks, 17 MLP-only blocks in the recorded sequence | Carry Mamba state and attention KV with their separate lifetimes. |
| Mistral-7B-v0.3 | 32 full-attention, dense-FFN layers | Held-out dense model; the graph and reduced upstream check below are available. |

The inventory names block types from config fields; it does not claim that
these models are compiled or timed on Radiance. The
Nemotron `M`, `*`, and `-` symbols respectively denote Mamba2, attention,
and MLP-only blocks in its [official pinned configuration code](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/blob/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f/configuration_nemotron_h.py).
Qwen3-MoE uses the config's expert count, top-k, sparse step, and explicit
dense-FFN exceptions according to its [decoder implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py);
the pinned model has an MoE FFN in every layer. OLMo, Qwen3-MoE, and Mistral
now have ordered software schedules and reduced reference controls below.
Nemotron has an ordered hybrid software schedule, a reduced upstream Mamba
mixer check, and a connected reduced-width 42-block Radiance execution below.

[`profile_architecture_shapes.py`](profile_architecture_shapes.py) builds each
pinned graph at batch 1 × 128, batch 8 × 128, and batch 1 × 8,192 tokens,
followed by two dependent greedy decode steps. The compact
[architecture shape matrix](architecture-shapes.json) records phase order,
stage mix, idealized arithmetic counts, named tensor sizes, and K/V storage.
It loads no model weights. The long-context prefill rows are:

| Candidate | Dense linear MACs | Selected expert MACs | Visible attention dot MACs | Graph K/V after decode | Window-trimmed K/V estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 4.88 T | 0 | 3.85 T | 0.88 GiB | 0.88 GiB |
| OLMo-3-7B | 56.42 T | 0 | 7.15 T | 4.00 GiB | 2.50 GiB |
| Qwen3-30B-A3B | 10.07 T | 14.84 T | 13.20 T | 0.75 GiB | 0.75 GiB |
| Nemotron-3-Nano-4B | 25.81 T | 0 | 1.37 T | 0.13 GiB | 0.13 GiB |
| Mistral-7B-v0.3 | 58.27 T | 0 | 8.80 T | 1.00 GiB | 1.00 GiB |

OLMo's graph currently stores full K/V history even for its sliding layers;
the smaller number requires a window-evicting cache policy. Qwen3-MoE's
selected expert work is counted separately from dense linear work. Nemotron's
K/V number excludes Mamba state, which has its own
[state profile](nemotron-state-profiles.json). These rows distinguish useful
cache and routing experiments; they do not establish that any model is
compute-bound or memory-bound. Attention counts include only visible QK and
weighted-V dot products, and operand bytes are tensor sizes, not measured
traffic. The graphs also materialize every prefill logit, so several 8,192-
token named-tensor footprints exceed the current 2 GiB device map before
weights. The matrix is a workload plan, not an executable long-context or
timed hardware result. Regenerate with
`python3 kernels/evaluation/llm/profile_architecture_shapes.py --out
kernels/evaluation/llm/architecture-shapes.json`.
Its full checkpoint execution remains open.
All candidates need full checkpoint, input, and device checks before
model-level performance comparisons.

### Nemotron hybrid state capacity

[`profile_nemotron_state.py`](profile_nemotron_state.py) reads the pinned
Nemotron config and writes [three batch/context cases](nemotron-state-profiles.json).
It follows the recorded 42-block order: 21 Mamba2, 4 attention, and 17
MLP-only blocks. The revision-pinned [modeling code](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/blob/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f/modeling_nemotron_h.py)
allocates an initial Mamba cache at `expand × hidden_size = 6,272` channels.
The mixer itself has `96 × 80 = 7,680` inner channels and `9,728`
convolution channels after adding two sets of eight 128-wide B/C groups.
Prefill replaces the initial cache tensors with the mixer's convolution and
SSM states; decode must carry these resulting shapes, not the constructor's
initial allocation. The SSM state is constant per layer at
`[batch, 96, 80, 128]`, while the four attention layers' K/V history grows
with context. The profile includes separate BF16 and FP32 SSM storage cases
because the retained state dtype depends on the executed path.

| Batch × context | Initial Mamba cache, BF16 | Carried state, BF16 SSM | Carried state, FP32 SSM |
| --- | ---: | ---: | ---: |
| 1 × 128 | 33.16 MiB | 42.93 MiB | 82.31 MiB |
| 8 × 128 | 265.29 MiB | 343.47 MiB | 658.47 MiB |
| 1 × 8,192 | 33.16 MiB | 168.93 MiB | 208.31 MiB |

The carried totals include BF16 convolution and attention KV storage.
These are analytical resident bytes, not observed cache occupancy or HBM
traffic. The source's cache update can replace its initial allocation, so
the initial and carried columns are different moments rather than additive
buffers. The KV term represents an intended cached-attention mapping. In the
pinned `NemotronHBlock.forward`, the attention mixer is called without the
hybrid cache argument, despite the mixer's own cache-update path. A reference
run must settle this handoff before we claim that the pinned code performs
cached attention across decode steps. The connected reduced-width decoder
below tests the Mamba and attention state handoffs in Radiance. The pinned
attention-cache behavior still needs an upstream network-level check before
a performance claim.

Regenerate with `python3 kernels/evaluation/llm/profile_nemotron_state.py
--out kernels/evaluation/llm/nemotron-state-profiles.json`.

The [full-dimension hybrid schedule](nemotron3-nano-4b-graph.json) uses a
128-token prefill followed by two greedy decode steps. Its 1,322 stages keep
the config's 42 blocks in order. Mamba blocks expose input projection,
causal depthwise convolution, separate convolution-state carry, B/C and
time-step slices, selective scan with bias and softplus applied inside the
kernel, separate SSM-state output, gated RMSNorm, and output projection.
Attention blocks keep Q/K/V, causal GQA,
K/V append, and output projection without RoPE. MLP-only blocks use
up projection, squared ReLU, and down projection. Every block has its own
pre-norm and residual addition. The prefill LM head reads only the last
hidden token, matching `num_logits_to_keep=1`. The graph assumes FP32 SSM
storage and labels this choice in each state stage. It names existing
standalone primitives for 660 stages; 662 still lack a named device stage.
The [depthwise convolution primitive](../../mamba_depthwise_conv/README.md)
handles the causal four-tap filter and carries its raw-sample cache through
prefill and decode on a reduced fixture. The graph now binds its 63 convolution
stages to that primitive; the connected decoder below binds their paired
state outputs to the following decode calls.
The [selective scan primitive](../../mamba_selective_scan/README.md) covers
time-step preparation, the post-convolution recurrence, and final-state
carry on a small FP32 fixture.
Its [Cyclotron check](../../mamba_selective_scan/evaluation/cyclotron.json)
compares all 224 output and 128 final-state values with an independent NumPy
golden. The graph gives the 63 scan stages that primitive's name. The
connected reduced decoder below implements projection slicing, gate and norm,
and separate convolution and scan state outputs; pinned-weight validation
remains open.
The [reduced upstream mixer check](../../mamba_selective_scan/evaluation/transformers-reduced-cyclotron.json)
also extracts seven tokens of intermediate scan inputs from Transformers 5.3,
checks NumPy against its cached CPU fallback, and passes the resulting device
fixture on Cyclotron. This is random-weight, reduced-width evidence rather
than a pinned-checkpoint or full-model validation.
The [two-kernel handoff check](../../mamba_selective_scan/evaluation/conv-scan-handoff-cyclotron.json)
feeds the verified convolution device output into a separately built scan ELF
through a host file. It checks the `x/B/C` split and scan result on one reduced
mixer, without claiming connected execution or full-model performance.
The [connected reduced pair](../../mamba_conv_scan/README.md) then runs those
two callbacks in one ELF, passing the 48-channel convolution output directly
to the scan on device. Its [Cyclotron receipt](../../mamba_conv_scan/evaluation/transformers-reduced-cyclotron.json)
checks both outputs and both carried states across prefill and decode on two
Muon cores; all four hashes match the separate checks. It remains a reduced,
random-weight pair. The connected 42-block execution below adds the remaining
network stages. The pinned source's attention cache handoff remains a
reference discrepancy; no pinned-checkpoint numerical comparison is available.
The [reduced full mixer](../../mamba_mixer/README.md) adds input projection,
gated RMSNorm, and output projection around that connected pair. Its final
output agrees with the seeded Transformers CPU fallback within `4.11e-6`
maximum absolute error over seven tokens, including two cached decode calls.
This is a mixer-level check at reduced width; it does not resolve the pinned
network or attention-cache discrepancy.

[`nemotron_graph_reference.py`](nemotron_graph_reference.py) now executes the
named stages of the ordered hybrid graph in NumPy, including separate
convolution and SSM state outputs. The [42-block reduced control](nemotron-reduced-hybrid-chain.json)
uses synthetic FP32 weights and a three-token prefill followed by two decode
steps. It checks 1,320 teacher-forced stages and 1,322 greedy stages. The
greedy graph matches all 1,322 ordered stage IDs, operations, tensor edges,
and outputs of the pinned graph; its generated tokens are `[52, 52]`. Both
teacher-forced and greedy cached outputs match a five-token causal pass
exactly. Removing the first Mamba convolution state changes its next output
by `0.104`; bypassing the first attention layer's K/V cache changes its
attention output by `2.03`. The control peaked at about 43 MiB RSS.

An [independent mixer comparison](nemotron-graph-mixer-reference.json) binds
the new graph executor's first Mamba layer to the same random weights as the
installed Transformers 5.3.0 CPU fallback. Its largest mixer-output error is
`3.73e-7`; its final convolution and SSM states differ by at most `2.38e-7`
and `2.98e-8`. This check peaked at about 357 MiB RSS. The full graph uses
reduced widths and an eight-token scan chunk, so its exact five-token
self-consistency does not settle the known chunk-boundary difference in the
upstream fallback. The 42-block network has not been compared with the pinned
checkpoint. The pinned source's attention-cache handoff and config-parser
issues still need a reviewed reference adapter before full-model equivalence
can be claimed.

The [connected reduced-width build](../../model_chain/evaluation/nemotron-full-depth-reduced-build.json)
runs all 42 blocks in a single Radiance ELF. It passed 1,322 native stage
checks and [all 1,322 Cyclotron checks](../../model_chain/evaluation/nemotron-full-depth-reduced-functional.json)
with `tohost=0`. The one-layer Mamba check and full-depth verifier are in the
[connected-kernel record](../../model_chain/README.md#nemotron-hybrid-decoder-control).
These use deterministic synthetic FP32 weights, a three-token prompt, and
reduced widths; the pinned graph uses a 128-token prefill and original model
dimensions. The 3,629,544 Cyclotron cycles are functional ISA cycles, not
timed hardware performance. The kernel path does not validate the pinned
checkpoint or settle the upstream attention-cache behavior.

At the pinned graph's 128-token prompt length, a
[reduced one-layer mixer comparison](nemotron-context128-mixer-reference.json)
passes against the Transformers 5.3.0 CPU fallback over prefill and two cached
decode calls. It uses an eight-token scan chunk and a separate seeded weight
fixture; maximum mixer-output difference is `2.76e-6`. The connected
[42-block Radiance control](../../model_chain/evaluation/nemotron-context128-full-functional.json)
also passes all 1,322 Cyclotron stage checks for 128 prefill tokens and two
greedy decode steps. The [native diagnostic](../../model_chain/evaluation/nemotron-context128-native-diagnostic.json)
records 10 sparse later-residual mismatches at the strict default tolerance
and the explicitly calibrated `2e-5` absolute tolerance used for the passing
build. Widths, weights, and scan chunk size remain reduced; this is still not
the pinned checkpoint or hardware performance.

The [full hybrid reference check](verify_nemotron_network_reduced.py) also
compares the reduced 42-block graph with a same-weight Transformers 5.3.0
model forward. It instantiates Mamba, dense MLP, and attention modules in the
pinned order, retains their states across the two fixed decode calls, and
checks each block's pre-norm, mixer output, and residual. It also checks the
final norm and logits from both the component path and the library's model
forward. The installed library maps this model's `-` block to a MoE slot; the
check temporarily maps that slot to its dense `NemotronHMLP` class, then
restores the registry. This is an explicit compatibility adapter, not an
execution of the pinned remote model code or its checkpoint.

| Same-weight CPU case | Compared values | Maximum absolute error | Values outside `1e-4` relative + `2e-5` absolute | Values outside `1e-3` relative + `2e-4` absolute |
| --- | ---: | ---: | ---: | ---: |
| [3-token prefill](nemotron-network-short-reference.json) | 20,864 | `1.18e-5` | 0 | 0 |
| [128-token prefill](nemotron-network-context128-reference.json) | 532,864 | `1.83e-4` | 368 | 0 |
| [Causal mask removed](nemotron-network-noncausal-control.json) | 532,864 | `2.56` | 372,126 | 355,957 |
| [Mamba state dropped](nemotron-network-no-mamba-control.json) | 532,864 | `13.14` | 8,556 | 8,185 |

The 128-token final logits differ by at most `6.00e-6`; the larger raw
differences appear in internal late-prefill values. The deliberate faults
show that this fixture detects future-token attention and a missing Mamba
decode state. The comparison uses seeded FP32 weights and reduced widths;
its 390 checked tensors are block boundary and final tensors, while the
connected Cyclotron result above checks all 1,322 graph stages against the
separate NumPy oracle. Neither check establishes full pinned-checkpoint
equivalence or hardware performance.

Reproduce the four CPU cases in an environment with Transformers 5.3.0 and
PyTorch 2.9.1, using a task-local `TMPDIR`:

```sh
python3 kernels/evaluation/llm/verify_nemotron_network_reduced.py \
  --layers 42 --prefill 3 --out "$TMPDIR/nemotron-network-short.json"
python3 kernels/evaluation/llm/verify_nemotron_network_reduced.py \
  --layers 42 --prefill 128 --out "$TMPDIR/nemotron-network-context128.json"
python3 kernels/evaluation/llm/verify_nemotron_network_reduced.py \
  --layers 42 --prefill 128 --fault noncausal_attention \
  --out "$TMPDIR/nemotron-network-noncausal.json"  # expected to fail
python3 kernels/evaluation/llm/verify_nemotron_network_reduced.py \
  --layers 42 --prefill 128 --fault drop_mamba_state \
  --out "$TMPDIR/nemotron-network-no-mamba.json"  # expected to fail
```

Reproduce these controls with task-local `TMPDIR` and a Python environment
containing Transformers 5.3.0 and PyTorch 2.9.1:

```sh
python3 kernels/evaluation/llm/check_nemotron_graph_chain.py \
  --out "$TMPDIR/nemotron-reduced-hybrid-chain.json"
"$TRANSFORMERS_PYTHON" kernels/evaluation/llm/verify_nemotron_graph_reduced.py \
  --out "$TMPDIR/nemotron-graph-mixer-reference.json"
```
Regenerate with:

```sh
python3 kernels/evaluation/llm/stitch.py --model nemotron3_nano_4b \
  --batch 1 --prefill 128 --decode-steps 2 --generation greedy --compact \
  --out kernels/evaluation/llm/nemotron3-nano-4b-graph.json
```

[`audit_nemotron_source.py`](audit_nemotron_source.py) checks the exact
revision-pinned modeling file by SHA-256 and Python syntax tree. The
[saved audit](nemotron-source-audit.json) records that the attention block's
mixer call passes only `cache_position`, although its attention mixer accepts
`past_key_value` and `attention_mask`. It also records that the hybrid cache's
default sequence-length selector includes MLP-only layers: the first such
layer is 1, before the first attention layer at 12. This is a source contract
finding, not a run of the pretrained checkpoint. It explains why the graph's
intended cached-attention behavior cannot yet be claimed equivalent to the
pinned implementation. Reproduce with
`python3 kernels/evaluation/llm/audit_nemotron_source.py --out
kernels/evaluation/llm/nemotron-source-audit.json`; the script fetches only
the pinned 79 KB modeling file into memory.

The pinned CPU fallback also differs from the graph's Mamba head-group
mapping during prefill. In the pinned source, `B.repeat` and `C.repeat`
assign two groups to four heads as `0,1,0,1`; its cached decode branch uses
`expand` and assigns `0,0,1,1`. The graph uses the contiguous mapping for
both phases, as does the installed Transformers 5.3 reference above.
[`verify_nemotron_pinned_prefill.py`](verify_nemotron_pinned_prefill.py)
executes one reduced block with the exact revision-pinned modeling and config
files and the same synthetic FP32 weights. A small in-memory CPU normalization
shim substitutes for its missing Mamba Triton dependency; the pinned source
bytes are checked by SHA-256 and left unedited. Its projection, convolution,
and gate agree with the graph before the scan diverges:

| Prefill | Standard graph vs pinned scan, max abs | Standard graph vs pinned mixer output, max abs | Alternating-group control vs pinned scan/output, max abs |
| --- | ---: | ---: | ---: |
| [3 tokens](nemotron-pinned-prefill3.json) | `0.134` | `0.0824` | `1.19e-7` / `2.98e-7` |
| [128 tokens](nemotron-pinned-prefill128.json) | `0.484` | `0.292` | `1.79e-7` / `4.77e-7` |

Only changing the reference scan's head-group index to the pinned prefill
order removes the difference. This is evidence of a source-level prefill
inconsistency, not an executed checkpoint or a reason to relabel the graph
as pinned-source equivalent. Before using this checkpoint for model-level
claims, the prefill group layout and attention cache handoff need a reviewed
target implementation and same-weight end-to-end checks.

To reproduce, download the pinned
[modeling file](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/blob/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f/modeling_nemotron_h.py)
and [configuration file](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/blob/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f/configuration_nemotron_h.py)
into the task's `$TMPDIR`, then run:

```sh
curl -fLsS -o "$TMPDIR/modeling_nemotron_h.py" \
  'https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/resolve/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f/modeling_nemotron_h.py'
curl -fLsS -o "$TMPDIR/configuration_nemotron_h.py" \
  'https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/resolve/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f/configuration_nemotron_h.py'
ulimit -v 6291456
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 kernels/evaluation/llm/verify_nemotron_pinned_prefill.py \
    --modeling "$TMPDIR/modeling_nemotron_h.py" \
    --configuration "$TMPDIR/configuration_nemotron_h.py" --prefill 128
```

The 128-token CPU check peaked at about 1.3 GiB RAM and wrote no build tree.

The installed Transformers 5.3.0 is also not a drop-in numerical reference
for this checkpoint. [`check_nemotron_config.py`](check_nemotron_config.py)
passes the pinned config to its `NemotronHConfig.from_dict`; the
[recorded result](nemotron-config-compatibility.json) is `KeyError: '-'`
because this version's legacy-pattern parser does not accept MLP-only
blocks. That check uses about 247 MB peak RAM and loads no weights. A valid
numerical check needs a compatible implementation or a reviewed adapter
that preserves the 17 MLP-only blocks and fixes the cache handoff, followed
by prefill/decode comparisons against real checkpoint outputs.

There is also a small temporal-state control using the installed library's
single Mamba2 mixer, separate from the pinned network. The
[recorded CPU fallback run](nemotron-mamba-cache-control.json) compares a
full causal pass with a three-token prefill plus two cached steps at reduced
width. When all five tokens stay inside one eight-token scan chunk, maximum
output error is `2.38e-7`. With the checkpoint's 256-token chunk size, a
255-token prefill plus two steps first differs above `1e-5` at token 256;
maximum error is `0.001224`. This is a concrete boundary case to resolve
before treating cached and full-sequence checks as interchangeable. It is
specific to Transformers 5.3.0's CPU fallback, random reduced weights, and
PyTorch 2.9.1; it does not validate or diagnose the pinned checkpoint's
runtime. Reproduce with
`python3 kernels/evaluation/llm/verify_nemotron_mamba_cache.py --out
kernels/evaluation/llm/nemotron-mamba-cache-control.json` in that environment.
The hybrid graph therefore records a reference comparison rule: use the
same prefill length and one-token cached decode partition for the primary
phase-by-phase check, then report the full-sequence comparison separately.

### OLMo 3 local/global attention control

[`verify_olmo3_reduced.py`](verify_olmo3_reduced.py) executes the emitted
four-layer, random-weight graph and checks it against a direct NumPy mapping
and independent Transformers 5.3.0 execution. It
preserves the pinned order of three sliding-attention layers followed by one
full-attention layer, but sets the local window to three tokens so six input
tokens exercise the boundary. The NumPy path implements the model's
projection-wide Q/K RMSNorm before head splitting, YaRN rotary frequencies,
causal local/full masks, per-layer K/V carry, post-attention and
post-feedforward RMSNorm, and the untied output head. The source config sets
`use_cache=false`; both paths explicitly enable a cache for this test.
The [saved result](olmo3-reduced-reference.json) checks all 273 stages across
a four-token prefill and two cached decode tokens. The staged and direct
NumPy outputs agree exactly, and the largest upstream logit difference is `1.79e-7`;
upstream cached and full causal outputs agree within `2.24e-7`.
The connected compiler's separate OLMo reference, bound to this same random
weight fixture, differs from the staged graph by at most `2.44e-6` across
all 273 stages. A [four-layer Radiance control](../../model_chain/README.md)
also passes in Cyclotron using its own deterministic synthetic weights and a
three-token local window.
The [32-layer reduced Radiance control](../../model_chain/evaluation/olmo3-full-depth-reduced-functional.json)
passes all 2,123 stage checks in Cyclotron with a three-token prefill and two
greedy decode steps. Its [build](../../model_chain/evaluation/olmo3-full-depth-reduced-build.json)
records the calibrated native FP32 tolerance and its
[topology check](../../model_chain/evaluation/olmo3-full-depth-topology-check.json)
matches the pinned stage dependencies. The short executed prompt does not
cross the pinned 4,096-token local window, and the weights and widths are
synthetic reduced controls.
Forcing all four mapped layers to full attention changes the final logits by
`0.227` in both NumPy paths, so the test detects a lost local mask. The separately calculated
YaRN inverse frequencies are also checked against upstream. Reproduce with
`python3 kernels/evaluation/llm/verify_olmo3_reduced.py` in the pinned
Transformers environment. This is a reduced graph and operation-order check; a pinned
checkpoint, full-size Radiance executable, and performance run remain open.

`stitch.py --model olmo3_7b` also emits a pinned, full-dimension tensor
dependency graph. The [saved 4,097-token prefill plus two-step greedy
schedule](olmo3-7b-graph.json) has all 32 layers in their 3-local/1-full
order and 2,123 stages. The input length crosses the 4,096-token local
window. Each phase records Q/K normalization across the complete projection
**before** head reshaping, YaRN rotation, layer-specific causal masks, and
the K/V dependency carried into the next phase. It retains the full logical
KV history and applies the local window at attention; this does not prescribe
a physical cache allocation. The output is a shape and dependency plan:
965 stages have no named device primitive, and even the named primitives
are starting points rather than connected, shape-compatible implementations.
It has no checkpoint tensors or device latency. Regenerate it with:

```sh
python3 kernels/evaluation/llm/stitch.py --model olmo3_7b \
  --batch 1 --prefill 4097 --decode-steps 2 --generation greedy --compact \
  --out kernels/evaluation/llm/olmo3-7b-graph.json
```

[`profile_olmo3_shapes.py`](profile_olmo3_shapes.py) derives
[four scenario summaries](olmo3-shape-profiles.json) from that pinned graph
without downloading weights. It counts each distinct parameter once and
checks the total against the config's independent block formula: 7.298
billion elements, or 13.59 GiB at BF16. That weight footprint alone exceeds
both 2 and 4 GiB device-memory budgets. Running this full model on such a
target needs a weight streaming, quantization, or partitioning policy before
timing can be interpreted.

| Batch × prefill | BF16 KV with local window | BF16 full prefill logits | Peak named tensors with FP32 graph buffers |
| --- | ---: | ---: | ---: |
| 1 × 128 | 0.06 GiB | 0.02 GiB | 0.18 GiB |
| 8 × 128 | 0.50 GiB | 0.19 GiB | 1.41 GiB |
| 1 × 4,097 | 2.00 GiB | 0.77 GiB | 5.60 GiB |
| 1 × 8,192 | 2.50 GiB | 1.53 GiB | 11.19 GiB |

The KV column assumes local layers keep at most 4,096 tokens while global
layers keep the full context. The current graph instead retains full logical
history and masks local attention at use; its FP32 live-tensor column counts
that graph's distinct named buffers and retained logits, with no reuse or
scratch allocation. These columns answer different capacity questions and
cannot be added as independent allocations. Projecting only the last prefill
token's logits and bounding physical local-layer KV are software changes to
check against the upstream output before a long-context device run. The
record also counts linear and attention dot-product MACs by phase; none of its
byte or MAC counts are measured cache traffic, utilization, or latency.

### Qwen3 sparse expert routing control

[`verify_qwen3_moe_router.py`](verify_qwen3_moe_router.py) isolates the
sparse-FFN operation in the pinned Qwen3-30B-A3B family. A reduced block has
16 experts with the pinned top-8 selection and selected-score renormalization.
It routes seven input tokens, applies each selected expert's gate/up and down
projections, weights their outputs, and accumulates them in token order. The
[saved comparison](qwen3-moe-router-reference.json) matches the selected
expert IDs from Transformers 5.3.0 exactly; the maximum selected-score error
is `1.19e-7` and the maximum output error is `7.45e-8`. Omitting the
selected-score renormalization changes the result by `0.0643`, so the control
detects that mapping mistake. This is one random-weight FFN block, not the
48-layer decoder or a checkpoint-backed run. A
[connected reduced Radiance FFN](../../qwen3_moe_ffn/README.md) now runs
the same router, top-8 selection, selected-score normalization, expert
projections, SiLU, and weighted reduction in one ELF. Its Cyclotron check
passes all seven tokens; the full model and MX mapping remain open. Run the
independent CPU check with
`python3 kernels/evaluation/llm/verify_qwen3_moe_router.py` in the
Transformers environment. The [stage-level NumPy control](moe_ops.py)
evaluates the eight sparse-FFN operations using the graph's actual tensor
shapes and dependencies. Its [small test](test_moe_ops.py) compares those
stages with the fused routing and expert equation; this is a CPU control,
not a connected decoder run.

`stitch.py --model qwen3_30b_a3b` now emits the
[48-layer tensor dependency graph](qwen3-30b-a3b-graph.json). Its saved
batch-1, 128-token prefill and two greedy decode steps contain 3,467 stages.
Each layer keeps per-head Q/K normalization before RoPE, its own K/V cache,
and the attention output projection from 32 × 128 to the 2,048-wide residual
stream. The FFN stages expose router probabilities, top-8 indices and
renormalized scores, token grouping, packed gate/up expert weights, SwiGLU,
expert down projections, and weighted recombination. The graph records the
ordered cache and token dependencies across decode steps. Of its stages,
1,733 lack a named device primitive; the rest are starting points, not
connected implementations. The reduced router control above validates one
FFN operation. A separate [reduced stitched-decoder check](qwen3-moe-reduced-reference.json)
executes every stage of a two-layer graph with the same random weights as
Transformers 5.3.0: 153 stages for a three-token prefill and two cached decode
tokens, with all phase logits within `1.49e-7` and identical per-layer expert
choices. A 155-stage greedy graph selects the same two next tokens `[63, 9]`
and keeps logits within `1.49e-7`. Omitting top-k score renormalization moves
the final logits by `0.0130`. Run
`python3 kernels/evaluation/llm/verify_qwen3_moe_reduced.py` to reproduce it.
The [low-memory graph control](moe_graph_reference.py) composes the existing
dense NumPy evaluator with the sparse stages without loading Torch. Its
[tests](test_moe_graph_reference.py) check a two-layer, three-token prefill and
two dependent decode steps against a five-token causal pass, check greedy
token feedback, and reject an oversized expert weight before allocation. The
four MoE tests peak near 36 MiB RSS. This is a synthetic regression for graph
ordering; the Transformers comparison above is the upstream numerical check.
The full-dimension, 48-layer graph has not been numerically executed or
compiled for Radiance, and it does not include checkpoint weights. A
[48-layer reduced-width Radiance control](../../model_chain/README.md) now
executes the same ordered stage dependencies with synthetic FP32 weights;
the original dimensions and checkpoint remain open.
Regenerate the graph with:

```sh
python3 kernels/evaluation/llm/stitch.py --model qwen3_30b_a3b \
  --batch 1 --prefill 128 --decode-steps 2 --generation greedy --compact \
  --out kernels/evaluation/llm/qwen3-30b-a3b-graph.json
```

### Held-out Mistral dense control

`stitch.py --model mistral_7b_v03` emits a config-pinned
[32-layer graph](mistral-7b-v03-graph.json) with full causal attention,
32 query heads, 8 KV heads, 128-wide heads, RoPE, untied output projection,
and ordered cached decode. The saved batch-1, 128-token prefill plus two
greedy steps has 1,643 stages, including 293 without a named device
primitive. The pinned config has `sliding_window: null`; the graph therefore
does not impose a local-attention mask.

The [reduced upstream check](mistral-reduced-reference.json) binds a
two-layer stitched graph to the same random weights as Transformers 5.3.0.
Three-token prefill and two cached decode phases agree to at most `8.94e-8`
in logits, and cached versus full causal outputs agree to `4.47e-8`.
The greedy graph selects the same next tokens `[0, 43]` and agrees to at
most `8.94e-8`; changing the second decode token changes the final logits
by `0.388`. Run `verify_mistral_reduced.py` to reproduce it. This model is
reserved as a held-out architecture for design-rule checks. A separate
[32-layer reduced-width Radiance control](../../model_chain/README.md)
now executes the same ordered dependencies with synthetic weights and checks
all stages in Cyclotron. The full checkpoint and full dimensions have not been
executed on Radiance.

```sh
python3 kernels/evaluation/llm/stitch.py --model mistral_7b_v03 \
  --batch 1 --prefill 128 --decode-steps 2 --generation greedy --compact \
  --out kernels/evaluation/llm/mistral-7b-v03-graph.json
```

### Qwen3-0.6B graph control

`stitch.py --model qwen3_0_6b` reads the hash-checked candidate config and
emits prefill followed by an ordered cached-decode loop. Each of its 28
decoder layers keeps separate query and KV head axes. In the pinned
[Qwen3 decoder implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py),
query and key projections pass through per-head RMSNorm **before** RoPE;
values do not. The graph preserves that order, uses the checkpoint's
1,000,000 RoPE base, appends K/V by layer, and ties the output head to the
input embedding weights. The `execution_schedule` records which logits select
each greedy token and which KV buffers carry into the next step.

For batch 1, a 256-token prefill and four greedy decode steps produce 2,679
logical stages in the [saved graph](qwen3-0.6b-graph.json). Of those, 709 have no named device
primitive in this inventory; the other 1,970 name starting-point primitives,
not a compiled shared-buffer program or a claim of shape compatibility.
Regenerate this case with:

```sh
python3 kernels/evaluation/llm/stitch.py --model qwen3_0_6b \
  --batch 1 --prefill 256 --decode-steps 4 --generation greedy --compact \
  --out kernels/evaluation/llm/qwen3-0.6b-graph.json
```

The [reduced upstream check](qwen3-reduced-reference.json) runs a two-layer,
random-weight Qwen3 through independent Transformers 5.3.0 and the stitched
NumPy executor with the **same weights**. Prefill and two cached decode
outputs agree to at most `1.19e-7` absolute error; an upstream cached pass
also matches an upstream full causal pass to `8.94e-8`. Zeroing the first
query norm in the stitched path changes final logits by `0.0332`. Run
`verify_qwen3_reduced.py` to reproduce the check. These controls establish
the reduced decoder operation order and cache handoff. The original Qwen3
checkpoint's saved embedding and output head were then compared element by
element. The [full-checkpoint upstream check](qwen3-checkpoint-reference.json)
compares one-token prefill and greedy cached decode against Transformers
5.3.0 in FP32: the largest logit error is `2.96e-5`, and the selected token
agrees. A [three-token prefill, two-step greedy check](qwen3-checkpoint-multitoken-reference.json)
also matches upstream with at most `2.10e-5` logit error and the same two
selected tokens. An original-width, 28-layer Radiance ELF builds with a converted FP16
weight image for each of the one- and multi-token reference shapes, and a
seven-stage real-weight prefix passes Cyclotron; see the
[checkpoint-weight device record](../../model_chain/README.md#qwen3-checkpoint-weight-path).
The complete full-weight device run remains pending.

Run `python3 estimate_kv.py` to regenerate [kv-working-set.csv](kv-working-set.csv).
The script verifies each local config against its recorded hash before
calculating BF16 resident KV bytes for a 256-token prompt at batch 1 and 8
and an 8192-token prompt at batch 1. For each attention layer it uses
`batch × min(context, sliding_window) × 2 (K and V) × KV heads × head_dim ×
2 bytes`. Full-attention layers use `context` instead of a window. It counts
only Nemotron's four attention layers; its Mamba state is omitted. The
`new_token_kv_write_bytes` column counts one new K/V pair per attention layer
and batch item. OLMo's config has `use_cache=false`, so its rows describe an
explicit proposed cached-decode mapping. The reduced upstream check above
validates the operation and cache order; the full checkpoint path remains open.

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
