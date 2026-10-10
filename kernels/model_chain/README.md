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
| TinyLlama | 22 layers | 22 reduced layers, prefill and decode | One layer, one-token prefill and cached decode, 40 stages passed in Cyclotron | 22 layers against Transformers | Reduced one-token prefill and cached decode passed |
| DeepSeek-R1-Distill-Qwen-1.5B | 28 layers | 28 reduced layers, prefill and decode | One layer, prefill and cached decode, 46 stages passed with segmented weights in Cyclotron | 28 layers against Transformers | Pending |
| Gemma-2-2B | 26 layers | 26 reduced layers, prefill and decode | Pending | Checkpoint access pending | Pending |
| SmolVLA-base | 36 vision layers, 16 VLM layers, and 160 expert layer calls decomposed | Pending | Pending | 499 used tensors bound; upstream action chunk and runtime layer order/shape checked; graph numerical comparison pending | Pending |

The full-dimension schedules are dependency graphs, not compiled full-model
runs. The full-checkpoint Python controls are described in
[`kernels/evaluation/llm/README.md`](../evaluation/llm/README.md). The three
full-depth device ELFs in the table use small synthetic weights. The separate
one-layer TinyLlama checkpoint ELF above uses real weights and full dimensions.
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

The remaining work to compile the four **full** models is substantial:

1. Extend the checkpoint weight-image path to all layers. The current
   one-layer TinyLlama image and all 40 prefill and decode device stages pass. Full FP32
   images exceed the 32-bit address space, so the complete decoder needs
   staged loading or a validated lower-precision format. DeepSeek's one-layer
   prefill and cached decode pass using two disjoint FP32 weight regions; its
   28-layer image still exceeds device capacity. Keep pinned checkpoint checks at
   intermediate boundaries.
2. Tile the full hidden, FFN, head, sequence, and vocabulary dimensions;
   connect the existing MX-Gemmini kernels through the shared-buffer path and
   specify FP8/BF16 conversions. The current executable uses scalar FP32 SIMT.
3. Validate multi-token prefill and cached decode together on RTL, then run
   complete workloads with input and output checks. The three full-depth
   reduced decoder ELFs have passed Cyclotron functional execution; one-token TinyLlama prefill and
   cached decode passed RTL. The three-token prefill attempt reached its first
   wall-clock limit without a result.
4. Implement SmolVLA's vision encoder, connector, masks, expert attention,
   and action denoising on the device with a checkpoint numerical control.
   All vision, VLM, and expert layers are decomposed in `stitch.py`; 499 used
   checkpoint parameters bind by name and shape. The pinned upstream policy
   produces a complete 10-step action chunk, but the graph is not yet a
   numerical backend or executable.

Until those steps pass, the full four-model compilation and end-to-end
performance evaluation remain open. `kernels/evaluation/llm/README.md`
describes the model graphs and existing TinyLlama/DeepSeek checkpoint checks.
