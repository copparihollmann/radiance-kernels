# Connected model kernels

The kernels in this directory exercise tensor handoffs inside **one Radiance
executable**. `kernel.cpp` is a small three-stage device check:
`RMSNorm → FP32 linear → residual`. `compile_decoder.py` reads the decoder
dependency graphs in `kernels/evaluation/llm/stitch.py` and emits one SoC ELF
per selected model. The emitted program executes embedding, every operation
of the requested decoder layer, KV append, final norm, LM head, and any decode
steps in order. Its stage outputs remain in named GMEM buffers and the next
stage reads those buffers. A failed final or attention/activation comparison
sends a nonzero `tohost` result.

The default decoder build uses the reduced control dimensions from
`reference.py`: hidden width 32, FFN width 64, head width 8, and vocabulary
64. The default is one layer, three prefill tokens, and one cached decode
token, with deterministic generated FP32 weights. It tests the software
mapping and code generation. It does **not** execute the full checkpoint,
the original model dimensions, MX quantization, or a complete model workload.
The reduction preserves stage order and the grouped-query head ratio, but it
does not preserve every width ratio of the original network.
Generated ELF files and logs remain local under `generated/`.

## Checkpoint-weight device path

`export_decoder_weights.py` packs pinned safetensors into the row-major FP32
layout consumed by the generated Radiance kernels. The raw binary remains
ignored under `generated/`; the tracked [TinyLlama image manifest](evaluation/tinyllama-one-layer-weight-image.json)
records every logical parameter, checkpoint tensor, shape, GPU address, and
hash. `compile_decoder.py --checkpoint-dir ... --weight-image ...` builds an
ELF at the pinned model dimensions and places parameter pointers at those GPU
addresses. `run_functional.py` verifies the binary hash, asks Cyclotron to
preload it through `CYCLOTRON_WEIGHTS`, and checks device outputs against the
NumPy reference. This path changes no RTL.

The first [checkpoint device probe](evaluation/tinyllama-checkpoint-three-stage-functional-results.json)
passed the embedding, attention RMSNorm, and Q projection stages of a
one-layer TinyLlama prefill at the original 2048-wide hidden dimension.
Each stage was checked on the device against the pinned checkpoint NumPy
result at `rtol=5e-3, atol=5e-4`. Cyclotron reported 2,615,647 functional
ISA cycles. This is a three-stage prefix, not a complete layer or model, and
the cycle count is not a performance measurement.

The [complete one-layer prefill result](evaluation/tinyllama-checkpoint-one-layer-functional-results.json)
passes all 20 stages, including causal attention, the MLP, final norm, and
the 32,000-logit output projection, using the same checkpoint image and
original tensor widths. Cyclotron reported `tohost=0` after 64,348,772
functional ISA cycles. This is one layer with one input token; TinyLlama has
22 decoder layers, and a full-checkpoint Radiance executable has not passed.
The [one-layer cached-decode run](evaluation/tinyllama-checkpoint-one-layer-decode-functional-results.json)
passed all 40 prefill and decode stages, including the KV-cache handoff,
at `tohost=0` and 128,796,327 functional ISA cycles. It used token ID 1 for
both steps. These cycle counts are functional simulator instruction counts,
not device latency or throughput.

A [DeepSeek checkpoint prefix](evaluation/deepseek-checkpoint-three-stage-functional-results.json)
also passed its first three device stages: embedding, RMSNorm, and Q
projection at the pinned model dimensions. Its 2,054,187,008-byte
[one-layer image](evaluation/deepseek-one-layer-weight-image.json) contains
15 distinct parameters. This is a real-weight arithmetic check, not a
complete DeepSeek layer or model.

The DeepSeek placement check found a software address collision. At image
base `0x40000000`, the up-projection weights occupy
`[0x7c3c5000, 0x7f845000)`, overlapping Muon's reserved warp stacks
`[0x7ee00000, 0x7f000000)` from `lib/src/mu_start.S`. The
[18-stage run](evaluation/deepseek-checkpoint-eighteen-stage-overlap-failure.json)
failed at up-projection element 220; a GMEM dump showed 488 changed weight
words. The [complete one-layer attempt](evaluation/deepseek-checkpoint-one-layer-overlap-failure.json)
failed at the same stage and element. Reusing the identical binary image at
base `0x30000000` moved that
projection below the stack, and the
[18-stage control](evaluation/deepseek-checkpoint-eighteen-stage-shifted-functional-results.json)
passed. Its [shifted manifest](evaluation/deepseek-one-layer-shifted-weight-image.json)
still places `lm_head` across the stack. `split_decoder_weights.py` preserves
the parameter bytes while placing the first 1,120,692,224 bytes at
`0x30000000` and the 933,494,784-byte LM head at `0x80000000`; its
[manifest](evaluation/deepseek-one-layer-segmented-weight-image.json) records
both segment hashes and addresses. The
[complete one-layer prefill result](evaluation/deepseek-checkpoint-one-layer-segmented-functional-results.json)
passed all 23 stages against the pinned checkpoint NumPy reference in
Cyclotron (`tohost=0`, 164,746,829 functional ISA cycles). The combined image
SHA-256 is identical to the original contiguous image. The
[cached-decode run](evaluation/deepseek-checkpoint-one-layer-decode-segmented-functional-results.json)
passed all 46 prefill and decode stages with the same two segments
(`tohost=0`, 329,641,171 functional ISA cycles). It used token ID 1 for both
steps, as did the TinyLlama one-layer decode control. These are one-layer
results, not a 28-layer model run.
The compiler now rejects any stage whose weight interval intersects the
reserved stack range before building an ELF. This is a software placement
issue; no RTL was changed.

### Full-depth checkpoint builds

`export_decoder_fp16.py` stores linear and embedding weights as IEEE FP16,
leaving norms and biases FP32. It uses two GMEM segments around the warp-stack
reservation. The [TinyLlama image](evaluation/tinyllama-full-depth-fp16-image.json)
contains all 201 logical parameters for 22 layers (2,200,281,088 bytes), and
the [DeepSeek image](evaluation/deepseek-full-depth-fp16-image.json) contains
all 339 parameters for 28 layers (3,554,465,792 bytes). The exporter checks
the pinned checkpoint hashes and records every parameter address and packed
hash. Both full-depth Radiance ELFs built and passed the cache-line layout
check: [TinyLlama](evaluation/tinyllama-full-depth-fp16-build.json) has 754
checked stages; [DeepSeek](evaluation/deepseek-full-depth-fp16-build.json) has
1,126. Each ELF schedules one-token prefill and one cached decode token. Their
full-depth Cyclotron runs have been started; these build records do **not**
claim a completed device execution. Compared with the original FP32 NumPy
checkpoint, FP16 changes the final logits by at most `2.48e-5` for TinyLlama
and `9.18e-6` for DeepSeek on the recorded token input; the top logit index
is unchanged. These numbers are software reference comparisons, not RTL
performance data.

The gated Gemma checkpoint is now locally available at the pinned revision.
[Shard and shape validation](evaluation/gemma-checkpoint-binding.json) confirms
all 288 checkpoint tensors bind to the 996-stage, 26-layer graph, including
the tied input/output embedding. A one-layer NumPy graph matches the pinned
Transformers model using eager attention: [the reference result](evaluation/gemma-checkpoint-one-layer-reference.json)
reports maximum final-logit error `1.72e-5`. The
[full 26-layer comparison](evaluation/gemma-checkpoint-full-reference.json)
also passed, with maximum final-logit error `3.67e-5` for a three-token prefill
and two cached decode tokens compared against a full causal Transformers pass.
The default Transformers attention backend produced a larger difference, so
both reference records name the eager backend explicitly.

Gemma's full FP16 image does not fit the safe 32-bit GMEM regions. The
[full-depth INT8 image](evaluation/gemma-full-depth-int8-image.json) uses
per-output-channel scales for linear weights and per-row scales for embeddings;
norms remain FP32. It contains 289 logical parameters in 3,209,761,792 bytes
across two safe segments. The [996-stage ELF](evaluation/gemma-full-depth-int8-build.json)
built and passed the layout check, but its full-depth functional run has not
completed. The [three-stage checkpoint probe](evaluation/gemma-checkpoint-three-stage-int8-functional-results.json)
passed in Cyclotron at `tohost=0`. INT8 is a lossy model variant: its full-depth
NumPy logits differ from the original checkpoint by up to `2.67` after the
final softcap (RMS `0.367`) for the recorded one-token input, although the top
logit index is unchanged. It must not be reported as numerically equivalent
to the original Gemma checkpoint.

Gemma's checkpoint ties the input embedding and output head. A second
[image layout](evaluation/gemma-full-depth-int8-fp16-tied-image.json) keeps
that one table in FP16 and reads it transposed for the LM head, with the other
matrix weights in per-channel INT8. It fits in 3,207,713,792 bytes. The
[one-layer ELF](evaluation/gemma-one-layer-int8-fp16-tied-build.json) built
and passed layout validation. Its one-layer NumPy final-logit error falls from
`0.996` for the all-INT8 image to `0.421` maximum absolute error, and from
`0.120` to `0.0285` RMS. The [full-depth tied ELF](evaluation/gemma-full-depth-int8-fp16-tied-build.json)
also built with all 996 stages. Its full-depth NumPy final-logit error is
`0.874` maximum and `0.184` RMS, compared with `2.67` and `0.367` for the
all-INT8 image. The top logit index is unchanged on this input. This is still
a lossy Gemma variant, and its device run has not completed.

The [SmolVLA buffer plan](evaluation/smolvla-buffer-plan.json) computes
lifetimes for all 3,683 graph tensors while retaining the three vision
branches, one cached prefix, ten denoising iterations, and action carry between
iterations. Reusing buffers reduces the calculated FP32 activation arena from
3,937,703,552 to 47,193,024 bytes; the peak simultaneously live data is
34,610,112 bytes. The
[checkpoint image](evaluation/smolvla-full-checkpoint-image.json) now packs
all 499 used weights in their device layouts (1,610,949,504 FP32 bytes) and
checks their pinned checkpoint and per-parameter hashes. The
[3,673-stage action-chunk ELF](evaluation/smolvla-full-action-chunk-build.json)
also compiles and passes the Radiance buffer-layout check. It preserves the
three vision branches, VLM prefix cache, ten expert denoising iterations,
and action carry in one executable. That first ELF only checks for finite
outputs. A second [full ELF](evaluation/smolvla-full-action-chunk-exact-golden-build.json)
uses the exact [upstream input image](evaluation/smolvla-exact-input-image.json)
and checks all 1,600 actions against the
[upstream CPU policy explicitly cast to FP32](../evaluation/llm/smolvla-policy-fp32-results.json)
at `rtol=1e-3, atol=1e-3`; it has compiled and passed the Radiance buffer
layout check. Its complete Cyclotron run is in progress, so the output check
has not yet passed on the device. The checkpoint's original language embedding
weights are BF16. Casting the policy to FP32 with the same pinned weights and
inputs changes its final actions by up to `0.0263`; 405 of 1,600 values exceed
the device check tolerance. The [precision comparison](evaluation/smolvla-precision-comparison.json)
links both [original checkpoint](../evaluation/llm/smolvla-policy-results.json)
and FP32 policy output hashes. The graph schedule is unchanged by this cast;
the numerical distinction must accompany any SmolVLA result. An earlier
[native comparison against the original BF16 output](evaluation/smolvla-bf16-golden-native-mismatch.json)
failed at 59 action elements under the looser `1e-2` tolerance, which led to
the explicit FP32 policy control.
The [first-stage Cyclotron probe](evaluation/smolvla-stage1-functional-results.json)
passed the vision patch embedding with real weights after 142,591,422
functional ISA cycles, checking only that its output is finite. It does not
validate the full action chunk or its numerical output. An initial exact-input
attempt [failed before the first stage](evaluation/smolvla-exact-input-overlap-failure.json)
because the input blob at `0x10000000` overwrote the ELF entry point. The
exporter now places it at `0x20000000`; the runner checks all preloads against
ELF segments, reserved stacks, and each other before starting Cyclotron.
The [native first-stage comparison](evaluation/smolvla-stage1-upstream-reference.json)
uses the generated C++ stage with real checkpoint weights and byte-exact
camera input. Its 1024 by 768 patch tokens match a PyTorch convolution plus
position embedding using the pinned tensors to `1.34e-5` maximum absolute
error. This checks the first vision operation more strongly than the finite
Cyclotron probe. `run_smolvla_native.py` can execute the same generated
stage functions through the entire action schedule on a CPU and compare the
resulting 1,600 actions with the FP32 upstream policy golden values. The
[full native run](evaluation/smolvla-full-fp32-native-results.json) passed all
3,673 stages and all action elements, with `4.68e-6` maximum absolute error.
It remains a software arithmetic check; Cyclotron supplies the device
instruction check.

To reproduce the exact-input SmolVLA build, set
`SMOLVLA_CHECKPOINT_DIR` to the pinned checkpoint directory containing
`config.json` and `model.safetensors`, then run from the repository root:

```sh
python3 kernels/evaluation/llm/verify_smolvla_policy.py \
  --checkpoint-dir "$SMOLVLA_CHECKPOINT_DIR" \
  --out kernels/evaluation/llm/smolvla-policy-results.json
python3 kernels/evaluation/llm/verify_smolvla_policy.py \
  --checkpoint-dir "$SMOLVLA_CHECKPOINT_DIR" --promote-fp32 \
  --out kernels/evaluation/llm/smolvla-policy-fp32-results.json
python3 kernels/evaluation/llm/compare_smolvla_precision.py \
  --checkpoint-reference kernels/evaluation/llm/smolvla-policy-results.json \
  --fp32-reference kernels/evaluation/llm/smolvla-policy-fp32-results.json \
  --out kernels/model_chain/evaluation/smolvla-precision-comparison.json
python3 kernels/model_chain/export_smolvla_weights.py \
  --checkpoint-dir "$SMOLVLA_CHECKPOINT_DIR"
python3 kernels/model_chain/export_smolvla_inputs.py \
  --golden kernels/evaluation/llm/smolvla-policy-results.json \
  --out-dir kernels/model_chain/generated/smolvla-exact-inputs
python3 kernels/model_chain/compile_smolvla.py \
  --weight-image kernels/model_chain/generated/checkpoint-smolvla/smolvla_base/weights-image.json \
  --input-image kernels/model_chain/generated/smolvla-exact-inputs/inputs-image.json \
  --golden-output kernels/evaluation/llm/smolvla-policy-fp32-results.json \
  --out-root kernels/model_chain/generated/checkpoint-smolvla-exact-golden-elf \
  --stages-per-object 40 --build
python3 kernels/model_chain/run_smolvla_functional.py \
  --generated-root kernels/model_chain/generated/checkpoint-smolvla-exact-golden-elf \
  --sim-cycles 100000000000 --timeout 86400
```

For the faster native math check after that build:

```sh
python3 kernels/model_chain/run_smolvla_native.py \
  --generated-root kernels/model_chain/generated/checkpoint-smolvla-exact-golden-elf \
  --stage-limit 1 \
  --dump kernels/model_chain/generated/checkpoint-smolvla-exact-golden-elf/smolvla_base/native/stage1.bin
python3 kernels/model_chain/verify_smolvla_stage1.py \
  --checkpoint-dir "$SMOLVLA_CHECKPOINT_DIR" \
  --input-image kernels/model_chain/generated/smolvla-exact-inputs/inputs-image.json \
  --stage-dump kernels/model_chain/generated/checkpoint-smolvla-exact-golden-elf/smolvla_base/native/stage1.bin
python3 kernels/model_chain/run_smolvla_native.py \
  --generated-root kernels/model_chain/generated/checkpoint-smolvla-exact-golden-elf \
  --threads 8
```

The full simulator run is compute intensive. The binary images, ELF, build
log, and simulator log remain under ignored `generated/`; the linked JSON
manifests record their hashes and stage counts.

The [capacity plan](evaluation/checkpoint-image-capacity.json) calculates the
whole-model FP32 image sizes from the graph: 4,400,193,536 bytes for
TinyLlama, 7,108,352,000 for DeepSeek, and 12,816,663,552 for Gemma. At the
current `0x40000000` base, none fits in 32-bit device addresses. The
one-layer TinyLlama image does fit (700,473,344 bytes). Full checkpoint
execution therefore requires a staged placement policy or a supported
lower-precision device path. A one-layer DeepSeek image fits at the same base
(2,054,187,008 bytes); a one-layer Gemma FP32 image is 5,030,065,152 bytes
and exceeds the address space even before placing other data. For DeepSeek,
32-bit capacity alone is insufficient: a contiguous one-layer image intersects
the warp stacks at either tested base, so the passing run uses two segments.

To reproduce the one-layer checkpoint build with the pinned TinyLlama
`config.json` and `model.safetensors` in `CHECKPOINT_DIR`:

```sh
python3 kernels/model_chain/export_decoder_weights.py \
  --model tinyllama --layers 1 --checkpoint-dir "$CHECKPOINT_DIR"
python3 kernels/model_chain/compile_decoder.py \
  --model tinyllama --layers 1 --prefill 1 --decode-steps 0 \
  --device-check-all-stages --stages-per-object 10 \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --weight-image kernels/model_chain/generated/checkpoint-weights/tinyllama/weights-image.json \
  --out-root kernels/model_chain/generated/checkpoint-one-layer --build
python3 kernels/model_chain/run_functional.py --models tinyllama \
  --generated-root kernels/model_chain/generated/checkpoint-one-layer \
  --sim-cycles 250000000 --timeout 240 \
  --out kernels/model_chain/generated/checkpoint-one-layer/result.json
```

The compiler needs NumPy, PyTorch, and safetensors for checkpoint builds.
The results above used the pinned checkpoint SHA-256 recorded in the image
manifest; the exporter and compiler reject a different checkpoint.
For DeepSeek, export the pinned one-layer image and then split it with:

```sh
python3 kernels/model_chain/export_decoder_weights.py \
  --model deepseek_r1_distill_qwen_1_5b --layers 1 \
  --checkpoint-dir "$DEEPSEEK_CHECKPOINT_DIR"
python3 kernels/model_chain/split_decoder_weights.py \
  --source-image kernels/model_chain/generated/checkpoint-weights/deepseek_r1_distill_qwen_1_5b/weights-image.json \
  --out-dir kernels/model_chain/generated/checkpoint-weights/deepseek-segmented
python3 kernels/model_chain/compile_decoder.py \
  --model deepseek_r1_distill_qwen_1_5b --layers 1 --prefill 1 --decode-steps 0 \
  --device-check-all-stages --stages-per-object 10 \
  --checkpoint-dir "$DEEPSEEK_CHECKPOINT_DIR" \
  --weight-image kernels/model_chain/generated/checkpoint-weights/deepseek-segmented/weights-image.json \
  --out-root kernels/model_chain/generated/checkpoint-deepseek-one-layer-segmented --build
python3 kernels/model_chain/run_functional.py \
  --models deepseek_r1_distill_qwen_1_5b \
  --generated-root kernels/model_chain/generated/checkpoint-deepseek-one-layer-segmented \
  --sim-cycles 500000000 --timeout 600 \
  --out kernels/model_chain/generated/checkpoint-deepseek-one-layer-segmented/result.json
```

For the cached-decode control, use `--decode-steps 1`, a separate
`--out-root`, and `--timeout 1200` in the corresponding build and run commands.

| Decoder family | Stages in one ELF | Native max error | Cyclotron functional cycles |
| --- | ---: | ---: | ---: |
| TinyLlama | 40 | `9.54e-7` | 128,536 |
| DeepSeek-R1-Distill-Qwen-1.5B | 46 | `1.07e-6` | 130,278 |
| Gemma-2-2B | 46 | `1.07e-6` | 123,153 |

The [full-depth native checks](evaluation/full-depth-native-results.json) and
[Cyclotron runs](evaluation/full-depth-functional-results.json) use
the original layer counts while keeping the small synthetic tensor dimensions.
They execute one prefill token and one cached decode token through every stage:

| Decoder family | Layers | Checked device stages | Native maximum error | Cyclotron functional cycles |
| --- | ---: | ---: | ---: | ---: |
| TinyLlama | 22 | 754 | `5.72e-6` | 1,197,586 |
| DeepSeek-R1-Distill-Qwen-1.5B | 28 | 1,126 | `7.15e-6` | 1,539,629 |
| Gemma-2-2B | 26 | 996 | `1.81e-5` | 1,272,040 |

The separate [full-depth reference controls](evaluation/full-depth-tinyllama-reference.json)
([DeepSeek](evaluation/full-depth-deepseek-reference.json),
[Gemma](evaluation/full-depth-gemma-reference.json)) compare cached decode
with one full causal pass, then change a decode token and confirm that the
logits respond. These are CPU checks of graph wiring. The native results do
not by themselves establish device execution; the linked Cyclotron result
checks every stage and reports `tohost=0` for all three.
Reproduce the controls with `reference.py --model MODEL --layers N` and
`compile_decoder.py --model MODEL --layers N --prefill 1 --decode-steps 1
--device-check-all-stages --verify-native --out-root
kernels/model_chain/generated/full-depth`, using `N=22,28,26` for the models
above. `record_full_depth.py` checks the generated manifests and logs before
writing the native result record. The
[sharded build control](evaluation/sharded-control-functional-results.json)
passes all 40 stages of a one-layer TinyLlama program in Cyclotron at `-O3`.
`--stages-per-object 20` puts stage functions, initialized data, and device
checks into smaller object files while preserving one linked ELF and the same
schedule. It is the build route for the full-depth device check. Reproduce the
device runs using the same `compile_decoder.py` arguments with `--build
--stages-per-object 20`, then run `run_functional.py --generated-root
kernels/model_chain/generated/full-depth --sim-cycles 10000000 --timeout 600`.
The larger simulator limit is required because the default 1,000,000 cycles
ends before these programs finish. The functional cycles are not performance
latency measurements.

| Model | Full-dimension schedule | Reduced Radiance ELF | Checkpoint device result | Full-checkpoint numerical check | VCS RTL |
| --- | --- | --- | --- | --- | --- |
| TinyLlama | 22 layers | 22 reduced layers, prefill and decode | One checkpoint layer passed; 754-stage full-checkpoint FP16 ELF built, functional run pending | 22 layers against Transformers; FP16 error measured | Reduced one-token prefill and cached decode passed |
| DeepSeek-R1-Distill-Qwen-1.5B | 28 layers | 28 reduced layers, prefill and decode | One checkpoint layer passed; 1,126-stage full-checkpoint FP16 ELF built, functional run pending | 28 layers against Transformers; FP16 error measured | Pending |
| Gemma-2-2B | 26 layers | 26 reduced layers, prefill and decode | 46-stage one-layer INT8 run passed; 996-stage all-INT8 and tied-FP16 ELFs built, functional runs pending | All 26 layers match Transformers eager attention; quantized variants have measured error | Pending |
| SmolVLA-base | 36 vision layers, 16 VLM layers, and 160 expert layer calls decomposed | Full 3,673-stage checkpoint ELF built | Exact-input device output comparison pending; 47 MB activation arena | 499 used tensors bound; full generated C++ schedule passes against FP32 upstream action chunk | Pending |

The full-checkpoint Python controls are described in
[`kernels/evaluation/llm/README.md`](../evaluation/llm/README.md). The reduced
ELFs use small synthetic weights; the full-checkpoint ELFs in this table use
the pinned checkpoints and original dimensions. A successful build and layout
check establish compilation, while a pending functional run does not yet
establish device correctness.
The [Gemma one-layer INT8 result](evaluation/gemma-checkpoint-one-layer-int8-functional-results.json)
checked all 46 prefill and decode stages against the quantized NumPy reference
and passed at 960,681,976 functional cycles. Its final logits differ from the
original unquantized checkpoint by up to `0.996` for this input; the result
therefore establishes execution of the documented INT8 mapping, not numerical
equivalence to the original checkpoint.

Full-depth generated-stage C++ also runs on the host with the same checkpoint
images and source files used to build the ELFs. `run_checkpoint_native.py`
checks every stage tensor against the mapped-precision NumPy reference at
`rtol=5e-3, atol=5e-4`:

| Model and image | Layers | Stages checked | Host result |
| --- | ---: | ---: | --- |
| [TinyLlama FP16](evaluation/tinyllama-full-depth-fp16-native-results.json) | 22 | 754 | Passed |
| [DeepSeek FP16](evaluation/deepseek-full-depth-fp16-native-results.json) | 28 | 1,126 | Passed |
| [Gemma INT8 body, tied FP16 embedding](evaluation/gemma-full-depth-int8-fp16-tied-native-results.json) | 26 | 996 | Passed |

The [SmolVLA full host result](evaluation/smolvla-full-fp32-native-results.json)
checks its 3,673-stage action chunk against the pinned FP32 upstream policy.
These host results validate full graph execution and mapped arithmetic, while
the separate Cyclotron runs are needed to establish full-depth device
instruction execution. The generated C++ uses scalar SIMT operations; these
runs do not measure performance.
`audit_full_models.py` recomputes each pinned graph's stage count, verifies the
ELF and checkpoint-image hashes and placement, checks the upstream and host
records, and reports whether a matching full-depth Cyclotron result exists.
Run `python3 kernels/model_chain/audit_full_models.py`; add `--out
kernels/model_chain/generated/four-model-readiness.json` for a local snapshot.
The audit reports device validation as pending while those runs are active.

Reproduce the decoder host checks with the compiled directories:

```sh
python3 kernels/model_chain/run_checkpoint_native.py --model tinyllama --generated-root kernels/model_chain/generated/checkpoint-fp16-full-depth-elf
python3 kernels/model_chain/run_checkpoint_native.py --model deepseek_r1_distill_qwen_1_5b --generated-root kernels/model_chain/generated/checkpoint-fp16-full-depth-elf
python3 kernels/model_chain/run_checkpoint_native.py --model gemma_2_2b_it --generated-root kernels/model_chain/generated/checkpoint-int8-fp16-tied-full-depth-elf
```
The generated C++ stores tensors as contiguous arrays, but the schedule keeps
query heads, KV heads, head width, token position, and cache lifetime as
separate logical dimensions. The full-dimension graph tests check those shapes
against the pinned configs. The reduced device controls change dimensions and
weights, so their passing results do not establish valid execution of an
upstream checkpoint. A flattened captured operation list alone is not treated
as a model execution.

The native checks compare every floating-point stage with the independent
NumPy graph at `rtol=1e-4, atol=1e-5`. The device check compares the same
stage buffers at `rtol=5e-3, atol=5e-4` and finishes with `tohost=0` for all
three families. The exact ELF, simulator, config, and log hashes are in
[`evaluation/functional-results.json`](evaluation/functional-results.json).
The test suite also changes one intermediate golden value and confirms that
the native stage check fails, so the comparison path is exercised in both
directions.
The listed cycles are counts in a functional ISA model, **not** RTL timing or
performance measurements. A two-step greedy TinyLlama build also passed all
device stages at 167,027 functional cycles, including an explicit integer
argmax check, next-token embedding, and KV handoffs; see
[`evaluation/greedy-functional-results.json`](evaluation/greedy-functional-results.json).
A two-layer TinyLlama build passed at 236,765 functional cycles, checking the
handoff between decoder layers as well as cached decode; see
[`evaluation/two-layer-functional-results.json`](evaluation/two-layer-functional-results.json).

To rebuild all three controls with the local Muon and RISC-V toolchains:

```sh
python3 kernels/model_chain/compile_decoder.py --model tinyllama --device-check-all-stages --build
python3 kernels/model_chain/compile_decoder.py --model deepseek_r1_distill_qwen_1_5b --device-check-all-stages --build
python3 kernels/model_chain/compile_decoder.py --model gemma_2_2b_it --device-check-all-stages --build
python3 kernels/model_chain/run_functional.py
python3 -m unittest discover -s kernels/model_chain -p 'test_*.py'
```

The greedy and two-layer variants use `--generation greedy --decode-steps 2`
and `--layers 2`, respectively. Their local ELFs are under
`generated/variants/` and are ignored by git. The result files retain the
exact binary and simulator hashes.
`run_vcs.py --model tinyllama` runs the built one-layer TinyLlama SoC ELF on
the local Radiance VCS RTL simulator and records its status and exact hashes.
The decoder RTL run is separate from the short three-stage handoff result.
The smaller one-token prefill build completed all 20 stage checks in VCS with
no device failure; its 31,864 functional cycles and RTL result are recorded in
[`evaluation/prefill-functional-results.json`](evaluation/prefill-functional-results.json)
and [`evaluation/rtl-tinyllama-prefill-result.json`](evaluation/rtl-tinyllama-prefill-result.json).
The one-token prefill plus one cached decode build completed all 40 stage checks
in VCS. Its functional and RTL records are
[`evaluation/short-decode-functional-results.json`](evaluation/short-decode-functional-results.json)
and [`evaluation/rtl-tinyllama-short-decode-result.json`](evaluation/rtl-tinyllama-short-decode-result.json).
An initial one-layer prefill-plus-decode VCS attempt reached its 600-second
wall-clock limit with a three-token prefill and without a device result; its
exact binary and log hashes are
in [`evaluation/rtl-tinyllama-result.json`](evaluation/rtl-tinyllama-result.json).
This is an incomplete run, not a passing decoder RTL result.
The completed short prefill can be rebuilt and rerun with:

```sh
python3 kernels/model_chain/compile_decoder.py --model tinyllama --prefill 1 --decode-steps 0 --out-root kernels/model_chain/generated/variants/prefill-only --device-check-all-stages --build
python3 kernels/model_chain/run_vcs.py --model tinyllama --generated-root kernels/model_chain/generated/variants/prefill-only --out kernels/model_chain/evaluation/rtl-tinyllama-prefill-result.json --timeout 600
```
For the cached-decode control, use `--prefill 1 --decode-steps 1` and
`--out-root kernels/model_chain/generated/variants/short-decode` in the build,
then set the same directory with `run_vcs.py --generated-root`; allow an
1800-second wall-clock limit. These VCS runs check correctness. Their
host-inclusive simulation times are not kernel latencies.

`RISCV_TOOLCHAIN_PATH`, `RISCV64_TOOLCHAIN_PATH`, and `LLVM_MUON` can override
the local defaults. `--verify-native` regenerates and checks the C++ schedule
without a device toolchain; `--generation greedy --decode-steps 2` checks the
greedy token dependency. Each generated directory contains `manifest.json`,
`native.log`, `build.log`, `kernel.radiance.elf`, and `kernel.soc.elf`.

The Muon L0 data cache is private per core. A first VCS run of the three-stage
check failed on exactly one 16-element output row after a cross-core handoff
(`tohost=33`). The callbacks now assign every stage to one warp on core 0,
with a schedule barrier and manager fence between stages. A later VCS check
found four corrupt tail elements when the output occupied only part of its
last cache line. The intermediate and output buffers now start on
64-byte boundaries and occupy whole cache lines; `check_layout.py` enforces
that ELF layout. The final three-stage VCS run passed all three device checks.
Its ELF and simulator hashes and the unmeasured host-inclusive simulation time
are recorded in
[`evaluation/rtl-handoff-result.json`](evaluation/rtl-handoff-result.json).
The raw log remains local under `generated/rtl/handoff-vcs.log`.
With the local Muon and RISC-V toolchains configured, rebuild this smaller
check using `make -C kernels/model_chain data kernel.soc.elf`, then run
`python3 kernels/model_chain/run_vcs.py --model handoff`.

The one-warp mapping establishes correctness for a small program. A scalable
two-core mapping needs an explicit handoff protocol or ownership-preserving
tiling, plus RTL validation and performance measurement.

The four full-depth checkpoint programs now compile to Radiance ELFs. These
builds use FP16 checkpoint images with FP32 arithmetic for TinyLlama and
DeepSeek, INT8 weight formats for Gemma, and FP32 weights and arithmetic for
SmolVLA. The numerical effects of those choices are recorded with the
reference checks above. The next correctness gate is a completed functional
ISA run with output validation for each **full-checkpoint** ELF; those runs
are distinct from the completed reduced-dimension and short checkpoint
controls. A successful native execution of generated C++ checks the schedule
and math on the host, but does not substitute for the ISA run.

The remaining evaluation work is to run representative multi-token prefill
and cached decode cases, connect the existing MX-Gemmini kernels where the
data types and layouts agree, and validate those longer programs on RTL.
SmolVLA also needs a device check of its exact-input 10-step action chunk;
the current FP32 mapping must be reported separately from the original BF16
checkpoint policy. Performance comparisons need a fixed hardware baseline,
timing-capable RTL or FPGA runs, and measured memory and compute counters.
Cyclotron functional cycle counts and host run times cannot serve as those
performance numbers. `kernels/evaluation/llm/README.md` describes the pinned
upstream model and checkpoint controls.
