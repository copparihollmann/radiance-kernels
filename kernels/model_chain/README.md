# Connected model kernels

## Two-cluster decoder batch

`compile_decoder.py --batch-clusters 2` gives each cluster its own padded
activation and KV-cache buffers while both read the same checkpoint weights.
The current batch uses the same fixed prompt on each cluster. Each cluster
runs the complete graph on its core 0 and checks the selected final and
attention/activation tensors before reporting its own printBuf status.
`prepare_hbm.py` launches and waits for both clusters. This mapping tests
two independent inferences; it does not split a single inference across SMs.

The first [two-cluster TinyLlama candidate](evaluation/tinyllama-int8-batch2-hbm-candidate.json)
contains all 1,133 stages per request and the same pinned INT8 weight image
as the one-cluster build. Both generated C++ schedules pass 135 selected
tensor checks per request against the packed-weight reference and produce
greedy token IDs `[29892, 2787]`. The largest absolute difference is
`0.00052643`, within the device check's `rtol=0.005, atol=0.0005` tolerance.
Cyclotron ran the checkpoint-backed first two stages once with each replica's
address range selected; both passed. Changing only the second replica's input
made its device check fail. Cyclotron has one cluster in this setup, so those
runs check address selection and arithmetic, not concurrent two-SM execution.
The full two-cluster HBM SoC ELF and its 2 GiB, 32-byte, four-slice hashed
FireSim package passed layout and byte-reconstruction checks. The package
still needs the exact target HWDB match and an FPGA run before it can supply
two-SM latency or throughput. The per-channel INT8 weights are a lossy
variant of the original checkpoint.

The [DeepSeek-R1-Distill-Qwen-1.5B candidate](evaluation/deepseek-int8-batch2-hbm-candidate.json)
uses the same mapping for 1,691 stages per request. Both generated C++
schedules passed 171 selected tensor checks per replica, with maximum
absolute error `0.00231934` within the same device tolerance, and both
generated `[22573, 4396]`. The checkpoint-backed first two stages passed
in Cyclotron when each replica address range was selected separately;
changing the second replica's input failed its device check. Its compact
2 GiB hashed ELF passed source and weight byte reconstruction. A concurrent
two-SM FPGA run remains to be measured.

The [Gemma 2 2B it two-cluster build](evaluation/gemma-int5-int6-batch2-hbm-candidate.json)
runs the pinned 26-layer graph for input `[2, 100]` and two greedy steps.
Its 1,496-stage schedule passed 159 selected tensor checks on each CPU
replica and generated `[109, 235287]` for both, with maximum absolute error
`0.000128746` at the same device tolerance. The original packed image has two
segments. Duplicating the activations made the device ELF overlap the second
segment, so `relocate_decoder_segment.py` moves that segment by 8 MiB in the
GPU address map without changing its bytes. The rebuilt ELF passes placement
and compact 2 GiB hash reconstruction. Separate two-stage Cyclotron probes
passed with cluster 0 and cluster 1 addresses; changing only cluster 1's
input failed its device check. Full two-cluster device execution and FPGA
performance remain pending.

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

## Qwen3 reduced device control

The candidate [Qwen3-0.6B graph](../evaluation/llm/qwen3-0.6b-graph.json)
keeps query/key head normalization before RoPE, the pinned RoPE base,
per-layer KV carry, and the tied embedding/output head. A separate
[Transformers comparison](../evaluation/llm/qwen3-reduced-reference.json)
checks prefill and two cached decode outputs with the same random weights at
reduced dimensions. `compile_decoder.py` now emits the head-norm and tied-head
stages in the connected Radiance program. Its [one-layer device control](evaluation/qwen3-reduced-functional.json)
ran one prefill token and one cached decode token: native C++ matched all 44
stage outputs (maximum absolute error `1.31e-6`), and Cyclotron completed all
44 device stage checks with `tohost=0` after 67,496 functional ISA cycles.
The [two-layer greedy control](evaluation/qwen3-reduced-greedy-functional.json)
starts with three prefill tokens, selects two more from its own logits, and
carries each layer's KV cache between calls. Its 123 floating-point and two
integer stage checks passed in Cyclotron with `tohost=0` after 304,914
functional ISA cycles; the native C++ check's maximum absolute error was
`1.55e-6`. The device controls use `compile_decoder.py`'s deterministic
weight fixture; the independent Transformers comparison uses a different
random fixture and checks the same graph structure.
This is a correctness control, not a latency measurement. It uses generated
FP32 weights and reduced tensor widths, not the Qwen3 checkpoint. The
checkpoint-weight route and its separate device status are below.

Reproduce the reduced control with the Muon toolchain and Cyclotron paths set:

```sh
python3 kernels/model_chain/compile_decoder.py --model qwen3_0_6b \
  --layers 1 --prefill 1 --decode-steps 1 \
  --out-root path/to/generated --device-check-all-stages \
  --stages-per-object 16 --device-opt O1 --verify-native --build
python3 kernels/model_chain/run_functional.py --models qwen3_0_6b \
  --generated-root path/to/generated \
  --cyclotron-root path/to/chipyard/generators/radiance/cyclotron \
  --sim-cycles 100000000 --timeout 600 --out path/to/result.json
```

## Qwen3 checkpoint-weight path

The pinned Qwen3-0.6B checkpoint has a saved `lm_head.weight` as well as an
embedding table. The [checkpoint check](evaluation/qwen3-checkpoint-check.json)
hashes the original weights and compares all 155,582,464 elements of those
two tensors. They are equal, so the device image stores one FP16 table and
the output projection reads its transpose. Query and key norm weights are
bound separately. The [full image manifest](evaluation/qwen3-full-depth-fp16-image.json)
records 310 distinct parameters, their checkpoint keys and device addresses,
and the 1,192,230,912-byte image hash. The table is exported in row chunks;
the exporter does not hold the full vocabulary as an FP32 array. The
[placement plan](evaluation/qwen3-fp16-plan.json) fits that image below the
2 GiB GPU memory boundary and outside the reserved warp stacks.

The [one-layer prefix build](evaluation/qwen3-one-layer-prefix-build.json)
uses the [one-layer image](evaluation/qwen3-one-layer-fp16-image.json) and
checks the first seven original-width stages, through query/key head norms.
All seven [Cyclotron checks](evaluation/qwen3-one-layer-prefix-functional-results.json)
passed with `tohost=0` in 10,401,738 functional ISA cycles. This prefix does
not exercise the tied output projection. A [28-layer ELF](evaluation/qwen3-full-depth-fp16-build.json)
also builds for one-token prefill and one greedy decode step, with 1,071
ordered stages and the tied FP16 projection. Its checkpoint-backed NumPy
reference selects the same top token before and after FP16 storage conversion
on both steps; the largest final-logit difference is `1.01e-5`. The
[independent checkpoint comparison](../evaluation/llm/qwen3-checkpoint-reference.json)
checks all 28 layers against Transformers 5.3.0 with FP32 arithmetic:
prefill and greedy decode logits differ by at most `2.96e-5`, and both select
the same token. The upstream cached decode also agrees with an upstream full
causal pass to `2.49e-5`. Transformers loads its two saved matrices as
separate storage, while their values are exactly equal as checked above.
The [three-token prefill and two-step greedy check](../evaluation/llm/qwen3-checkpoint-multitoken-reference.json)
runs a 1,607-stage connected graph and compares its output logits with upstream
execution. Both paths select
tokens `[16, 8]`; the largest logit error across prefill and decode is
`2.10e-5`, and upstream cached versus full causal output differs by at most
`1.43e-5`. A [matching 1,607-stage Radiance ELF](evaluation/qwen3-full-depth-multitoken-fp16-build.json)
now builds with the same three input IDs and two dependent greedy steps. Its
checkpoint-backed reference also selects `[16, 8]`; FP16 storage conversion
keeps the top token at every output, with at most `1.36e-5` logit difference.
The [one-token full-depth device run](evaluation/qwen3-full-depth-fp16-functional.json)
now passes all 1,071 ordered stages in Cyclotron with `tohost=0` after
2,950,225,579 functional cycles. The [saved-artifact verification](evaluation/qwen3-full-depth-fp16-functional-verification.json)
rechecks the ELF, all generated device sources, reference, simulator,
configuration, completion log, and 1,192,230,912-byte checkpoint image by
hash. This is a full-dimension FP16 checkpoint mapping with one prefill token
and one dependent greedy decode step. Its FP16 arithmetic is not bit-for-bit
upstream FP32 execution. The three-token/two-step ELF is running separately;
it has no completed device result yet. Cyclotron functional cycles are not
hardware latency or utilization.

The [last-prefill-logit checkpoint check](evaluation/qwen3-prefill-last-logit-check.json)
uses the saved three-token full-width reference and streams the packed FP16
tied embedding table in 1,024-row chunks. Projecting only the final hidden
row reproduces the saved final prefill-logit row within `1.43e-5` maximum
absolute error and selects token `16` in both paths. Projecting the first
hidden row instead differs by `12.14` at maximum; a deliberately corrupted
packed-table hash is rejected. This checks the numerical basis for the
[last-logit graph mode](../evaluation/llm/profiles/qwen3-prefill-last-logit.json)
without executing that optimized graph on device. Reproduce with:

```sh
OPENBLAS_NUM_THREADS=1 python3 kernels/model_chain/verify_qwen3_last_logit.py \
  --build-dir "$TMPDIR/qwen3-multitoken-build/qwen3_0_6b" \
  --weight-image "$TMPDIR/qwen3-full-image/qwen3_0_6b/weights-image.json" \
  --out "$TMPDIR/qwen3-last-logit-check.json"
```

The [standalone device selection check](../select_last_token/README.md) also
copies the saved full-width final hidden row on Radiance and matches all
1,024 values exactly in Cyclotron. The optimized full-model ELF remains to
be built and checked.

The checkpoint file and binary image stay in the local task directory. To
reproduce the build, use the pinned checkpoint from the linked check and
task-local paths for the output:

```sh
python3 kernels/model_chain/export_decoder_fp16.py \
  --model qwen3_0_6b --layers 28 --checkpoint-dir "$CHECKPOINT" \
  --out-dir "$WEIGHTS"
python3 kernels/model_chain/compile_decoder.py \
  --model qwen3_0_6b --layers 28 --prefill 1 --decode-steps 1 \
  --generation greedy --token-ids 5 --stages-per-object 20 \
  --device-check-all-stages --checkpoint-dir "$CHECKPOINT" \
  --weight-image "$WEIGHTS/qwen3_0_6b/weights-image.json" \
  --out-root "$BUILD" --build
python3 kernels/model_chain/compile_decoder.py \
  --model qwen3_0_6b --layers 28 --prefill 3 --decode-steps 2 \
  --generation greedy --token-ids 5 6 7 --stages-per-object 20 \
  --device-check-all-stages --checkpoint-dir "$CHECKPOINT" \
  --weight-image "$WEIGHTS/qwen3_0_6b/weights-image.json" \
  --out-root "$BUILD_MULTI" --build
python3 kernels/evaluation/llm/verify_qwen3_checkpoint.py \
  --checkpoint-dir "$CHECKPOINT" --input-token 5 \
  --out "$BUILD/qwen3-checkpoint-reference.json"
python3 kernels/evaluation/llm/verify_qwen3_checkpoint.py \
  --checkpoint-dir "$CHECKPOINT" --prefill-ids 5 6 7 --decode-steps 2 \
  --out "$BUILD/qwen3-checkpoint-multitoken-reference.json"
python3 kernels/model_chain/run_functional.py --models qwen3_0_6b \
  --generated-root "$BUILD" --cyclotron-root "$CYCLOTRON" \
  --sim-cycles 12000000000 --timeout 21600 \
  --out "$BUILD/functional-results.json"
python3 kernels/model_chain/verify_functional.py \
  --result "$BUILD/qwen3_0_6b/functional-result.json" \
  --simulator "$CYCLOTRON/target/release/cyclotron" \
  --config "$BUILD/cyclotron-config.toml" \
  --source-config kernels/evaluation/llm/inputs/qwen3-0.6b.json \
  --out "$BUILD/qwen3_0_6b/verification.json"
```

The upstream FP32 comparison reused the same checkpoint file and reached
about 4.55 GiB peak RSS on the recorded single-threaded run. The FP16 image
export reached about 0.79 GiB peak RSS.

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

### Checkpoint greedy inference

`infer_decoder.py` accepts a text prompt when a local tokenizer is present, or
explicit token IDs, and compiles the **full decoder depth** into one Radiance
ELF. The graph selects the last-position argmax after each pass, feeds that ID
into the next pass, and carries every layer's KV cache. The generated C++ host
runner executes the same stage sources as the ELF, checks every stage against
the checkpoint-bound NumPy reference, and reports the selected IDs. For
example, with a locally cached TinyLlama checkpoint and exported FP16 image:

```sh
python3 kernels/model_chain/infer_decoder.py --model tinyllama \
  --checkpoint-dir /path/to/TinyLlama/snapshot \
  --weight-image /path/to/weights-image.json \
  --out-root /path/to/local/output --prompt 'Hello' --new-tokens 2
```

The output directory contains `kernel.radiance.elf`, `manifest.json`,
`native-check/native-result.json`, and `inference-result.json`. The prompt
tokens and expected stage values are built into this ELF; a different prompt
requires another build. The reported text comes from the tokenizer applied to
the IDs selected by the generated C++ stages on the host. This command does
not run the ELF on a device. `run_functional.py` can separately check its
instruction execution in Cyclotron, but its `tohost` result is a stage
pass/fail signal rather than a token stream. A local tokenizer is absent from
the pinned Gemma snapshot, so Gemma currently accepts token IDs and reports
token IDs unless `--tokenizer-dir` is supplied. These are functional checks,
not latency or throughput measurements.

For an independent full-depth control, `verify_checkpoint.py --full
--generation greedy --token-ids ...` compares selected IDs and final logits
from the stitched NumPy graph with a Transformers causal forward pass using
the same pinned checkpoint. This checks the original FP32 model graph;
`infer_decoder.py` checks the precision mapped into the exported image.

The following fixed prompts now have full-depth Radiance ELFs and passing
generated C++ host runs. Each of the two selected IDs also matches a separate
full-depth Transformers greedy run on the pinned original checkpoint:

| Checkpoint | Input | Host stage checks | Selected IDs | Output |
| --- | --- | ---: | --- | --- |
| [TinyLlama](evaluation/tinyllama-hello-greedy-inference.json) | `Hello` → `[1, 15043]` | [1,133/1,133](evaluation/tinyllama-hello-greedy-native.json) | `[29892, 2787]` | `, World` |
| [DeepSeek-R1-Distill-Qwen-1.5B](evaluation/deepseek-hello-greedy-inference.json) | `Hello` → `[9707]` | [1,691/1,691](evaluation/deepseek-hello-greedy-native.json) | `[22573, 4396]` | ` ensuring correct` |
| [Gemma-2-2B-it](evaluation/gemma-token-greedy-inference.json) | `[2, 100]` | [1,496/1,496](evaluation/gemma-token-greedy-native.json) | `[109, 235287]` | tokenizer unavailable locally |

The independent comparisons are recorded for
[TinyLlama](evaluation/tinyllama-hello-greedy-upstream.json),
[DeepSeek](evaluation/deepseek-hello-greedy-upstream.json), and
[Gemma](evaluation/gemma-token-greedy-upstream.json). The first two mapped
images store matrix weights in FP16; Gemma stores most matrices in INT8 and
the tied embedding in FP16. On this Gemma input, its mapped logits differ
from the original FP32 checkpoint by up to `0.507` after softcap, although
the two selected IDs agree. The Gemma mapping should not be described as
numerically equivalent to the original checkpoint.

A separate [one-layer TinyLlama Cyclotron run](evaluation/tinyllama-greedy-one-layer-functional.json)
passed all 62 stages, including both integer argmax outputs, at `tohost=0`
after 324,040,787 functional ISA cycles. Its [native control](evaluation/tinyllama-greedy-one-layer-native.json)
also passed. That is a full-width one-layer FP32 checkpoint test with a
three-token prefill and two greedy decode steps. It confirms device execution
of the new token handoff, but it is not a full-depth greedy device result.
The three full-depth greedy runs above are host executions of the generated
stage sources; none provides Radiance latency or throughput.

### FireSim handoff for checkpoint ELFs

The SoC ELF produced by `infer_decoder.py` contains the host program and the
Radiance device program, but its checkpoint weights remain in separate GPU
memory images. Loading that ELF alone would leave the model parameters
uninitialized. `package_firesim.py` adds each checked image as a `PT_LOAD`
segment at its CPU-visible GPU DRAM alias, so a single ELF can be staged by a
FireSim workload. SmolVLA also needs its exact input image. For example, on
an unhashed memory system:

```sh
python3 kernels/model_chain/package_firesim.py \
  --elf /path/to/tinyllama/kernel.soc.elf \
  --image /path/to/tinyllama/weights-image.json \
  --output /path/to/handoff/tinyllama.soc.elf
python3 kernels/model_chain/package_firesim.py \
  --verify /path/to/handoff/tinyllama.soc.elf.json
```

The `.elf.json` receipt records the source ELF hash, output ELF hash, GPU
addresses, and SHA-256 of every embedded image. Verification rehashes the ELF
and each embedded byte range. The model prompt, generated token checks, and
SmolVLA action checks are fixed at compile time; these files are reproducible
workload fixtures, not interactive inference servers. The packer does not
change the instruction stream.

For the nightly HBM launch and completion protocol, adapt an existing generated
source bundle with `prepare_hbm.py`. It copies all scheduled stage sources and
changes only the launch, completion, and verification reporting paths. The
resulting SoC ELF still needs the checkpoint images:

The current adapter launches cluster 0 only and schedules the dependent stages
on core 0 of that cluster. The other core and the second SM on a two-SM image
stay idle. Its cycles describe a single-core software mapping within the
one-SM baseline. A two-SM model mapping needs separate per-cluster
activations and outputs, shared weights, and synchronization at dependent
stage boundaries before a speedup comparison is meaningful.

```sh
python3 kernels/model_chain/prepare_hbm.py \
  --source /path/to/generated/tinyllama --out /path/to/hbm/tinyllama --build
```

For a target that directly loads a hashed GPU DRAM region, pass the **exact**
RTL hash size, unit, and slice count. The hash-aware mode scatters both device
program bytes and checkpoint bytes into the physical hash layout. It emits
load segments only for occupied physical pages, so FireSim need not write the
empty gaps in the hash range. Verification
reconstructs each logical source region from the output ELF and compares its
SHA-256 with the original. If the bitstream confirms a 4 GiB GPU hash range
with 32-byte units and four slices, the packaging command is:

```sh
python3 kernels/model_chain/package_firesim.py \
  --elf /path/to/hbm/tinyllama/kernel.soc.elf \
  --image /path/to/tinyllama/weights-image.json \
  --hash-size 0x100000000 --hash-unit 32 --hash-slices 4 \
  --output /path/to/handoff/tinyllama-hbm.soc.elf
python3 kernels/model_chain/package_firesim.py \
  --verify /path/to/handoff/tinyllama-hbm.soc.elf.json
```

When transferring an earlier logical handoff ELF, the HBM packer can read its
embedded checkpoint bytes using `--receipt /path/to/tinyllama.soc.elf.json`
instead of `--image`. Keep that receipt beside its `.elf` file. The packer
checks the donor ELF, each recorded load segment, and each embedded image hash
before creating the HBM ELF; it needs no second copy of the weight files.
Hash mode also accepts a device-only SoC ELF without `--image` or `--receipt`,
which lets the nightly smoke check use the same target memory map.
`build_firesim_targets.py` applies the packer to selected models in a portable
handoff directory, checks that the declared hash range covers their highest
checkpoint addresses, verifies each output, and writes `target-bundle.json`.
Its hash arguments must come from the target bitstream configuration.

This 4 GiB layout is an example, not a verified configuration for the XB10
images. The repository's documented standard `RadianceHBMConfig` hashes only
2 GiB; the four complete checkpoint images end at GPU addresses
`0xa4d16000` (TinyLlama), `0xf5170000` (DeepSeek/Qwen), `0xe0c9ec00`
(Gemma), and `0x914c6680` (SmolVLA). The packer refuses an image beyond the
declared hash range. Confirm the target bitstream's memory size, hash
parameters, LoadMem path, and number of cores per cluster before packaging.
Packaging and host checks alone do not establish FPGA correctness or timing.
The model HBM adapter reads both halves of Radiance's `mcycle` CSR and posts
64-bit start/end stamps through printBuf. Its host reports `kernel_cycles64`;
the standard nightly `kernel_cycles` field is 32-bit and should not be used
for these long model chains. This depends on the target image implementing
the `mcycleh` CSR present in the current Radiance source.

After a FireSim run, `collect_firesim_result.py` can hash the staged ELF,
its packer receipt, the matching full-model HBM build manifest, HWDB and
workload files, UART log, and any counter files. It accepts a result only if
the manifest identifies the complete model, the receipt binds it to that
source ELF, FireSim exited zero, and exactly one UART report contains the
device PASS status with consistent 64-bit cycle stamps. It also requires the
pinned checkpoint and every-stage checks for the three decoders, or the final
action-reference check for SmolVLA. The compact source
manifests are [TinyLlama](evaluation/tinyllama-2g-hbm-build.json),
[DeepSeek/Qwen](evaluation/deepseek-2g-hbm-build.json),
[Gemma](evaluation/gemma-2g-hbm-build.json), and
[SmolVLA](evaluation/smolvla-2g-hbm-build.json). The collector preserves raw
cycle counts and artifact hashes; it does not infer seconds, utilization,
power, or agreement between a supplied HWDB file and the physical bitstream.
Its [controls](test_collect_firesim_result.py) use synthetic UART text; no
full-model FPGA result is recorded here yet.

For the 2 GiB GPU map in the public HBM configuration, SmolVLA's FP32 image
fits if its two weight segments and exact input image are moved below the warp
stacks. `relocate_images.py` verifies the original bytes and rejects overlaps
with the device ELF, stacks, other images, or the declared GPU limit. The
output manifests use symlinks to the original bytes; the final packaged ELF
is self-contained. Rebuild the SmolVLA device program with the relocated
manifests because its stage code contains absolute parameter addresses:

```sh
python3 kernels/model_chain/relocate_images.py \
  --image /path/to/smolvla/weights-image.json \
  --image /path/to/smolvla-inputs/inputs-image.json \
  --base 0x14000000 --base 0x62b8c000 --base 0x01000000 \
  --device-elf /path/to/smolvla/kernel.radiance.elf \
  --gpu-size 0x80000000 --out-dir /path/to/compact-images
python3 kernels/model_chain/compile_smolvla.py \
  --weight-image /path/to/compact-images/image-0.json \
  --input-image /path/to/compact-images/image-1.json \
  --golden-output kernels/evaluation/llm/smolvla-policy-fp32-results.json \
  --out-root /path/to/compact-build --build
python3 kernels/model_chain/prepare_hbm.py \
  --source /path/to/compact-build/smolvla_base \
  --out /path/to/compact-hbm --build
python3 kernels/model_chain/package_firesim.py \
  --elf /path/to/compact-hbm/kernel.soc.elf \
  --image /path/to/compact-images/image-0.json \
  --image /path/to/compact-images/image-1.json \
  --hash-size 0x80000000 --hash-unit 32 --hash-slices 4 \
  --compact-hash \
  --output /path/to/smolvla-base-2g-hbm.soc.elf
```

This placement ends at GPU offset `0x74052680`. Check the generated device
ELF's load segments against the relocated manifests again after rebuilding,
and verify the packaged ELF with `package_firesim.py --verify`. The optional
`--compact-hash` lays the same physical GPU load regions into consecutive
ELF file offsets. It avoids a sparse 2 GiB file and leaves each segment's
physical load address unchanged. The October 11
XB10 bitstream's exact HWDB and FPGA execution remain to be checked.

`export_decoder_int8_untied.py` provides a full-depth alternative for
TinyLlama and DeepSeek when the GPU region is 2 GiB. It keeps norms and biases
FP32 and packs the untied embedding and all linear weights with the existing
per-channel INT8 kernels. At GPU base `0x14000000`, the TinyLlama image is
1,102,157,824 bytes and ends at `0x55b19800`; the DeepSeek image is
1,781,318,656 bytes and ends at `0x7e2cc400`. Both leave the device program
and reserved warp stacks clear. The exporter verifies pinned checkpoint hashes
and writes a manifest for `compile_decoder.py`.

These are lossy checkpoint conversions. On the fixed greedy fixtures, the
generated host C++ passed 1,133/1,133 TinyLlama stages and 1,691/1,691
DeepSeek stages against their respective INT8 references. Both produced the
same two tokens as the upstream checkpoint fixtures; their logits differ. The
Radiance and nightly HBM source ELFs compile and the 2 GiB compact-hash
FireSim packages pass local reconstruction checks. Hardware correctness and
performance remain unmeasured for these builds.

The first two stages were also run in Cyclotron with INT8 bytes recovered
directly from each compact FireSim ZIP. In each case, the recovered embedding
and first attention-norm parameter hashes matched the exporter. The generated
device program checked both stage outputs against its checkpoint-backed INT8
reference and returned `tohost=0`:

| Model | Recovered image | Two-stage build | Cyclotron result | Functional cycles |
| --- | --- | --- | --- | ---: |
| TinyLlama | [65,672,192 bytes](evaluation/tinyllama-2g-embedding-prefix-image.json) | [build](evaluation/tinyllama-2g-embedding-prefix-build.json) | [2/2 passed](evaluation/tinyllama-2g-embedding-prefix-functional-results.json) | 312,457 |
| DeepSeek-R1-Distill-Qwen-1.5B | [233,987,584 bytes](evaluation/deepseek-2g-embedding-prefix-image.json) | [build](evaluation/deepseek-2g-embedding-prefix-build.json) | [2/2 passed](evaluation/deepseek-2g-embedding-prefix-functional-results.json) | 140,383 |

`extract_hashed_parameter.py` makes these subset images from each ZIP's stored
model ELF using `--parameter embed_tokens --parameter layers.0.attn_norm`.
Build with the model's pinned checkpoint, its fixed prefill IDs, and
`compile_decoder.py --stage-limit 2 --device-check-all-stages --build`; then
run `run_functional.py`. These are original-dimension prefix checks, not
execution or latency measurements of the complete decoder programs.

Gemma needs a smaller image for this map. `export_gemma_int4_tied.py` keeps
the shared input/output embedding and attention matrices in INT8, packs the
feed-forward matrices in signed INT4, and leaves norms in FP32. The full
26-layer image is 1,790,800,896 bytes and ends at `0x7edd7400`, below the
reserved stacks. A full 1,496-stage greedy Radiance ELF compiled, and its
image does not overlap the program. This is a **feasibility build**, not a
validated Gemma workload: its two tokens are `[5231, 651]`, while the pinned
checkpoint produces `[109, 235287]`. A group-32 INT4 numerical reference
also differs, producing `[235248, 109]`. The measured comparisons and image
hashes are in [the weight-format trials](evaluation/gemma-2g-weight-format-trials.json).
`evaluate_gemma_group32.py --checkpoint-dir /path/to/pinned/Gemma/snapshot`
repeats the group-32 reference check without exporting another image. It used
about 7.5 GiB of peak RAM in the recorded run.
Do not use these INT4 conversions to report Gemma inference performance.

`export_gemma_int5_int6.py` provides a stronger compact format: the shared
embedding/output table uses signed INT5 with one scale per 64 hidden values,
feed-forward matrices use INT6, attention matrices use INT8, and norms remain
FP32. The [image](evaluation/gemma-2g-int5-int6-image.json) holds all 26
layers in 2,019,513,344 bytes across `0x01000000..0x0f435400` and
`0x14200000..0x7e3c0000`. The [1,496-stage build](evaluation/gemma-2g-int5-int6-build.json)
compiled and passed device/image/stack placement. Its quantized reference
generated `[109, 235287]`, the two upstream checkpoint tokens. The
[generated host C++ run](evaluation/gemma-2g-int5-int6-native.json) passed all
1,496 stage checks against that reference. The [2 GiB hashed FireSim
package](evaluation/gemma-2g-int5-int6-package.json) reconstructed and
SHA-checked both image segments and the device program; the local FESVR
loader accepted the ELF. These are fixed-fixture correctness checks. Logits
differ from the original checkpoint, and neither FPGA correctness nor timing
has been measured. The original full checkpoint build remains available for
a larger confirmed GPU memory map.

The [small Cyclotron packed-operator probe](evaluation/gemma-2g-packed-device-probe.json)
checks INT5 unpacking, group-64 INT5 embedding, INT6 linear, and the tied
INT5 output head in two dependent Radiance stages. It passed with `tohost=0`
in 3,982 functional simulator cycles. It uses small fixed arrays and does not
establish execution of the full 1,496-stage Gemma ELF. Rebuild it with
`run_packed_device_probe.py --out-dir "$TMPDIR/packed-device-probe"
--cyclotron-root "$CYCLOTRON_ROOT" --llvm-muon "$LLVM_MUON"
--riscv-toolchain "$RISCV_TOOLCHAIN_PATH"`.

An additional [checkpoint-backed Cyclotron probe](evaluation/gemma-2g-embedding-prefix-functional-results.json)
uses the actual packed Gemma bytes recovered from the compact FireSim ELF.
The [subset image](evaluation/gemma-2g-embedding-prefix-image.json) contains
the embedding table and first attention norm only (405,513,216 bytes); both
parameter hashes match the exporter. The [two-stage Radiance build](evaluation/gemma-2g-embedding-prefix-build.json)
checked the embedding and norm outputs against the mapped-precision checkpoint
reference. Cyclotron passed both stages with `tohost=0` in 362,401 functional
cycles. This checks real checkpoint data on the device ISA model. It does not
exercise the remaining 1,494 stages or measure FPGA latency.

The extractor reads the stored ELF member in the existing handoff ZIP and
writes only the requested parameter range; it does not unpack a second model
ELF. The checkpoint build needs Python with `safetensors`, PyTorch, and NumPy.
Reproduce the prefix with task-local output directories:

```sh
python3 kernels/model_chain/extract_hashed_parameter.py \
  --zip "$GEMMA_HANDOFF_ZIP" \
  --elf-member gemma-int5-int6-2g/gemma-int5-int6-2g.soc.elf \
  --parameter embed_tokens --parameter layers.0.attn_norm \
  --out-dir "$WEIGHTS"
python3 kernels/model_chain/compile_decoder.py \
  --model gemma_2_2b_it --layers 26 --prefill 2 --decode-steps 2 \
  --generation greedy --token-ids 2 100 --stages-per-object 20 \
  --device-check-all-stages --stage-limit 2 \
  --checkpoint-dir "$GEMMA_CHECKPOINT" \
  --weight-image "$WEIGHTS/weights-image.json" --out-root "$BUILD" --build
python3 kernels/model_chain/run_functional.py \
  --models gemma_2_2b_it --generated-root "$BUILD" \
  --cyclotron-root "$CYCLOTRON_ROOT" \
  --sim-cycles 100000000 --timeout 600 --out "$BUILD/functional-results.json"
```

For the fixed Gemma rebuild, set `GEMMA_CHECKPOINT` to the pinned snapshot and
choose task-local `WEIGHTS`, `BUILD`, and `HBM` directories. The Muon toolchain
must be available through `LLVM_MUON`:

```sh
python3 kernels/model_chain/export_gemma_int5_int6.py \
  --checkpoint-dir "$GEMMA_CHECKPOINT" --out-dir "$WEIGHTS"
python3 kernels/model_chain/compile_decoder.py \
  --model gemma_2_2b_it --layers 26 --prefill 2 --decode-steps 2 \
  --generation greedy --token-ids 2 100 --stages-per-object 20 \
  --device-check-all-stages --checkpoint-dir "$GEMMA_CHECKPOINT" \
  --weight-image "$WEIGHTS/gemma_2_2b_it/weights-image.json" \
  --out-root "$BUILD" --build
python3 kernels/model_chain/run_checkpoint_native.py \
  --model gemma_2_2b_it --generated-root "$BUILD" --threads 4 --jobs 2
python3 kernels/model_chain/prepare_hbm.py \
  --source "$BUILD/gemma_2_2b_it" --out "$HBM" --build
python3 kernels/model_chain/package_firesim.py \
  --elf "$HBM/kernel.soc.elf" \
  --image "$WEIGHTS/gemma_2_2b_it/weights-image.json" \
  --hash-size 0x80000000 --hash-unit 32 --hash-slices 4 --compact-hash \
  --output "$HBM/gemma-int5-int6-2g.soc.elf"
```

The target bitstream must use this exact memory map before staging the ELF.

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
full-depth [Cyclotron results](evaluation/tinyllama-full-depth-fp16-functional-results.json)
([DeepSeek](evaluation/deepseek-full-depth-fp16-functional-results.json))
passed every stage check at `tohost=0`. Compared with the original FP32 NumPy
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
built and passed the layout check. The [three-stage checkpoint probe](evaluation/gemma-checkpoint-three-stage-int8-functional-results.json)
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
a lossy Gemma variant. Its [full-depth Cyclotron run](evaluation/gemma-full-depth-int8-fp16-tied-functional-results.json)
passed all 996 mapped-precision stage checks at `tohost=0`; this does not make
the mapped logits equivalent to the original unquantized checkpoint.

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
layout check. Its [24-hour Cyclotron attempt](evaluation/smolvla-full-action-chunk-fp32-functional-results.json)
reached its wall-clock limit without a terminal device result. The output check
has not yet passed on the device; the timeout is not a demonstrated output
mismatch. A [bounded simulator diagnostic](evaluation/smolvla-simulator-diagnostic.json)
checks the generated mapping and exercises the same ELF with a four-million-
cycle limit. All 3,673 stages are scheduled with one warp; 1,492 scalar
`linear` calls represent 354.13 billion multiply-accumulates, including
260.92 billion in the vision layers. The bounded run loaded the pinned weight
and input images and reached its cycle limit in 11.0 seconds on this host.
That check shows execution advances past startup and makes a long full run
plausible. It covers only the start of the program, so it cannot exclude a
later stall or validate the action output. Its wall time measures the host
simulator, not Radiance latency. The 24-hour log has no per-stage progress
markers; we still need those or bounded graph segments to locate its stopping
point. The checkpoint's original language embedding
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

The compact 2 GiB handoff has a stronger [first-stage device check](evaluation/smolvla-2g-stage1-exact-functional-results.json).
The probe recovered just the three patch-embedding parameters (5,508,096
bytes) and the full fixed input (9,444,288 bytes) from the packaged FireSim
ELF. Their [weight](evaluation/smolvla-2g-stage1-exact-weights-image.json)
and [input](evaluation/smolvla-2g-stage1-exact-input-image.json) manifests
retain the verified source hashes. The corresponding compact
[weight parameter map](evaluation/smolvla-2g-source-weights-image.json) and
[input parameter map](evaluation/smolvla-2g-source-input-image.json) are
recorded so `extract_hashed_parameter.py --image-manifest` can recover these
subsets from the handoff ZIP without materializing the full ELF.
`generate_smolvla_stage1_golden.py`
recomputed the first vision output directly with the pinned PyTorch layer;
its [3,145,728-byte golden image](evaluation/smolvla-2g-stage1-pytorch-golden-image.json)
matches the previously recorded upstream reference hash. The
[Radiance build](evaluation/smolvla-2g-stage1-exact-build.json) compared all
786,432 patch outputs with that golden at `rtol=atol=1e-3`. Cyclotron passed
with `tohost=0` after 167,726,096 functional cycles. A separate
[generated-host-golden control](evaluation/smolvla-2g-stage1-native-golden-functional-results.json)
also passed. These are one-stage correctness checks; the other 3,672 stages
and FPGA latency remain unmeasured for the compact package.

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
The [last-stage Cyclotron probe](evaluation/smolvla-final-euler-functional-results.json)
uses the prior action and final velocity produced by that full native schedule
with the exact checkpoint inputs. It links the unchanged generated
`denoise9.euler` stage from the full ELF source and checks all 1,600 actions
against the FP32 upstream policy at `rtol=1e-3, atol=1e-3`. It passed at
63,275 functional ISA cycles. The result records the source, fixture, ELF,
and simulator hashes. This checks the final action carry and Euler update on
Radiance; a zero device output would fail 1,482 of the 1,600 comparisons. It
does not establish that all 3,673 stages execute on the device.

An [RTL VCS attempt](evaluation/smolvla-final-euler-vcs-attempt.json) used the
SoC ELF for this same final-stage probe and the locally built
`RadianceSingleClusterFastConfig` simulator. The process reached its 600-second
wall limit after UART startup without an observed device check or failure
marker. Its status is **incomplete**; the attempt provides no RTL correctness
result for SmolVLA. The record pins the ELF, simulator, and log hashes. The
local VCS binary predates this probe ELF, so a fresh simulator build and a
longer run would be needed before drawing an RTL conclusion.

The [ten-step Euler chain probe](evaluation/smolvla-euler-chain-functional-results.json)
passed in Cyclotron at `tohost=0` after 620,941 functional ISA cycles. It
executes the unchanged `denoise0.euler` through `denoise9.euler` functions
in order, using the full native run's ten `action_out` tensors as velocities.
It checks all 1,600 action values after every update and the alternating
action buffers. Its fixture control recovers the FP32 upstream final action
within `4.68e-6` maximum absolute error. This validates the action carry
across ten iterations. The expert networks that produce the velocities are
checked by the full generated-C++ host run; the full ELF device run remains
pending.
The [negative control](evaluation/smolvla-euler-chain-negative-control.json)
changes the first expected action to `1000000.0f` in a separate probe ELF.
Cyclotron rejects it at iteration 1, element 0 (`tohost=65537`), which checks
that the intermediate device comparisons are active.

The [expert attention probe](evaluation/smolvla-expert-attention-functional-results.json)
checks the first `masked_gqa` stage of the action expert with its real query,
291-token KV cache, and suffix attention mask. A separate NumPy grouped-query
attention calculation matches the generated native stage to `3.58e-7`
maximum absolute error. The copied generated stage then passed on Cyclotron
against that independent output (`tohost=0`, 18,169,579 functional ISA cycles)
for all 48,000 output values. The
[negative control](evaluation/smolvla-expert-attention-negative-control.json)
changes one expected value and is rejected at element 0 (`tohost=1`). This
checks a representative expert attention operation; it does not establish
device execution of every expert layer or the full action graph.

The [expert projection probe](evaluation/smolvla-expert-linear-functional-results.json)
checks `denoise0.expert00.q_proj` with its input traced from the full native
schedule and the same packed FP32 weight image used by the full Radiance ELF.
An independent NumPy matrix product agrees with the native stage within
`5.48e-6` maximum absolute error. Cyclotron preloaded both weight-image
segments and accepted all 48,000 outputs (`tohost=0`, 21,372,548 functional
ISA cycles). The [negative control](evaluation/smolvla-expert-linear-negative-control.json)
replaces one expected value and fails at element 0 (`tohost=1`). This is a
targeted device check of one weighted expert stage, not a full action run or a
performance latency measurement.

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

To reproduce the bounded simulator diagnostic without replacing the full-run
log or result, use a separate work directory:

```sh
python3 kernels/model_chain/profile_smolvla_simulator.py \
  --generated-root kernels/model_chain/generated/checkpoint-smolvla-exact-golden-elf \
  --cyclotron-root /path/to/cyclotron \
  --work-dir "$TMPDIR/smolvla-runtime" --cycle-cap 4000000 --timeout 180 \
  --out kernels/model_chain/evaluation/smolvla-simulator-diagnostic.json
```

The diagnostic counts scalar linear work from the generated stage sources and
records ELF, image, simulator, config, and log hashes. The result represents a
cycle-capped startup check, not a full-model correctness or performance run.

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

To reproduce the last-stage device probe, first run the generated native
binary twice to capture its input tensors. The first run stops after the prior
Euler action; the second traces the final velocity and confirms the complete
action result. Both fixture files remain under ignored `generated/`:

```sh
mkdir -p kernels/model_chain/generated/smolvla-late-stage-probe
OMP_NUM_THREADS=8 kernels/model_chain/generated/checkpoint-smolvla-exact-golden-elf/smolvla_base/native/smolvla_native 3397 kernels/model_chain/generated/smolvla-late-stage-probe/denoise8-euler.bin > kernels/model_chain/generated/smolvla-late-stage-probe/denoise8.log 2>&1
OMP_NUM_THREADS=8 SMOLVLA_NATIVE_TRACE_DIR="$PWD/kernels/model_chain/generated/smolvla-late-stage-probe" kernels/model_chain/generated/checkpoint-smolvla-exact-golden-elf/smolvla_base/native/smolvla_native > kernels/model_chain/generated/smolvla-late-stage-probe/full-trace.log 2>&1
python3 kernels/model_chain/probe_smolvla_euler.py \
  --full-generated-root kernels/model_chain/generated/checkpoint-smolvla-exact-golden-elf \
  --previous-action kernels/model_chain/generated/smolvla-late-stage-probe/denoise8-euler.bin \
  --velocity kernels/model_chain/generated/smolvla-late-stage-probe/3672.bin \
  --previous-log kernels/model_chain/generated/smolvla-late-stage-probe/denoise8.log \
  --trace-log kernels/model_chain/generated/smolvla-late-stage-probe/full-trace.log \
  --golden-output kernels/evaluation/llm/smolvla-policy-fp32-results.json \
  --out-root kernels/model_chain/generated/smolvla-final-euler-probe \
  --out kernels/model_chain/evaluation/smolvla-final-euler-functional-results.json
python3 kernels/model_chain/probe_smolvla_euler_chain.py \
  --full-generated-root kernels/model_chain/generated/checkpoint-smolvla-exact-golden-elf \
  --trace-dir kernels/model_chain/generated/smolvla-late-stage-probe \
  --input-image kernels/model_chain/generated/smolvla-exact-inputs/inputs-image.json \
  --golden-output kernels/evaluation/llm/smolvla-policy-fp32-results.json \
  --out-root kernels/model_chain/generated/smolvla-euler-chain-probe \
  --out kernels/model_chain/evaluation/smolvla-euler-chain-functional-results.json \
  --negative-out kernels/model_chain/evaluation/smolvla-euler-chain-negative-control.json
```

To reproduce the expert attention probe, generate five trace tensors from a
copy of the full generated sources. The copy keeps the audited full-native
binary and its result untouched:

```sh
mkdir -p kernels/model_chain/generated/smolvla-attention-fixture
cp -a kernels/model_chain/generated/checkpoint-smolvla-exact-golden-elf/smolvla_base kernels/model_chain/generated/smolvla-attention-fixture/
python3 kernels/model_chain/run_smolvla_native.py \
  --generated-root kernels/model_chain/generated/smolvla-attention-fixture \
  --stage-limit 938 --threads 8 \
  --trace-dir kernels/model_chain/generated/smolvla-attention-fixture/traces \
  --trace-stage denoise0.expert00.q_rope \
  --trace-stage denoise0.expert00.k_append \
  --trace-stage denoise0.expert00.v_append \
  --trace-stage denoise0.attention_mask \
  --trace-stage denoise0.expert00.attention \
  --out kernels/model_chain/generated/smolvla-attention-fixture/fixture-result.json
python3 kernels/model_chain/probe_smolvla_attention.py \
  --full-generated-root kernels/model_chain/generated/checkpoint-smolvla-exact-golden-elf \
  --fixture-root kernels/model_chain/generated/smolvla-attention-fixture \
  --out-root kernels/model_chain/generated/smolvla-attention-probe \
  --out kernels/model_chain/evaluation/smolvla-expert-attention-functional-results.json \
  --negative-out kernels/model_chain/evaluation/smolvla-expert-attention-negative-control.json
```

To reproduce the weighted expert projection, trace the input and output of
stage 931 from the same full generated sources:

```sh
mkdir -p kernels/model_chain/generated/smolvla-linear-fixture
cp -a kernels/model_chain/generated/checkpoint-smolvla-exact-golden-elf/smolvla_base kernels/model_chain/generated/smolvla-linear-fixture/
python3 kernels/model_chain/run_smolvla_native.py \
  --generated-root kernels/model_chain/generated/smolvla-linear-fixture \
  --stage-limit 931 --threads 8 \
  --trace-dir kernels/model_chain/generated/smolvla-linear-fixture/traces \
  --trace-stage denoise0.expert00.input_norm \
  --trace-stage denoise0.expert00.q_proj \
  --out kernels/model_chain/generated/smolvla-linear-fixture/fixture-result.json
python3 kernels/model_chain/probe_smolvla_linear.py \
  --full-generated-root kernels/model_chain/generated/checkpoint-smolvla-exact-golden-elf \
  --fixture-root kernels/model_chain/generated/smolvla-linear-fixture \
  --out-root kernels/model_chain/generated/smolvla-linear-probe \
  --out kernels/model_chain/evaluation/smolvla-expert-linear-functional-results.json \
  --negative-out kernels/model_chain/evaluation/smolvla-expert-linear-negative-control.json
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
| TinyLlama | 22 layers | 22 reduced layers, prefill and decode | [754/754 full-checkpoint FP16 stages passed](evaluation/tinyllama-full-depth-fp16-functional-results.json), `tohost=0` | 22 layers against Transformers; FP16 error measured | Reduced one-token prefill and cached decode passed |
| DeepSeek-R1-Distill-Qwen-1.5B | 28 layers | 28 reduced layers, prefill and decode | [1,126/1,126 full-checkpoint FP16 stages passed](evaluation/deepseek-full-depth-fp16-functional-results.json), `tohost=0` | 28 layers against Transformers; FP16 error measured | Pending |
| Gemma-2-2B | 26 layers | 26 reduced layers, prefill and decode | [996/996 tied-FP16/INT8 stages passed](evaluation/gemma-full-depth-int8-fp16-tied-functional-results.json), `tohost=0` | All 26 layers match Transformers eager attention; quantized variant has measured error | Pending |
| SmolVLA-base | 36 vision layers, 16 VLM layers, and 160 expert layer calls decomposed | Full 3,673-stage checkpoint ELF built | [Full run timed out at 24 hours](evaluation/smolvla-full-action-chunk-fp32-functional-results.json); targeted stages passed | 499 used tensors bound; full generated C++ schedule passes against FP32 upstream action chunk | Pending |

The full-checkpoint Python controls are described in
[`kernels/evaluation/llm/README.md`](../evaluation/llm/README.md). The reduced
ELFs use small synthetic weights; the full-checkpoint ELFs in this table use
the pinned checkpoints and original dimensions. The three passing decoder
runs each checked one-token prefill and one cached decode token. The SmolVLA
timeout does not establish full action-chunk device correctness.
Cyclotron reported 5,053,982,746 functional cycles for TinyLlama,
7,500,712,529 for DeepSeek, and 5,808,146,549 for tied-FP16/INT8 Gemma.
These counts are functional ISA simulation steps, not device latency,
throughput, or hardware utilization.
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
It requires a matching tracked evaluation record before reporting a full-depth
device pass; an ignored local simulator result alone is insufficient.
For SmolVLA it also compares the compiled schedule hash, verifies that each
Euler output feeds the next denoising step, that all 16 expert layers connect
in order within each step, and that the prefix KV cache is read across the
ten iterations. The lineage regression tests reject a reset action or skipped
expert layer.
Run `python3 kernels/model_chain/audit_full_models.py`; add `--out
kernels/model_chain/generated/four-model-readiness.json` for a local snapshot.
The audit reports three passing full-depth decoder device runs and one
incomplete SmolVLA action-chunk run.

### Full-depth upstream token sequence

The pinned decoder controls in `verify_checkpoint.py` use token IDs
`[1, 2, 3]` for prefill and `[4], [5]` for two cached decode steps. The
compiler now accepts `--token-ids 1 2 3 4 5`, and the build manifest records
the IDs bound to each graph input. This is an explicit choice: older default
builds reused ID 1 for each decode input. The following separate ELFs use
the pinned full layer counts, original tensor widths, the same checkpoint
images as the one-token builds, and all-stage device checks:

| Model | Layers | Stages | Radiance build | Full generated-C++ host check | Mapped vs original final-logit max error |
| --- | ---: | ---: | --- | --- | ---: |
| TinyLlama | 22 | 1,131 | [build](evaluation/tinyllama-full-depth-fp16-multitoken-build.json) | [passed](evaluation/tinyllama-full-depth-fp16-multitoken-native-results.json) | `1.41e-5` |
| DeepSeek-R1-Distill-Qwen-1.5B | 28 | 1,689 | [build](evaluation/deepseek-full-depth-fp16-multitoken-build.json) | [passed](evaluation/deepseek-full-depth-fp16-multitoken-native-results.json) | `1.12e-5` |
| Gemma-2-2B-it | 26 | 1,494 | [build](evaluation/gemma-full-depth-int8-fp16-tied-multitoken-build.json) | [passed](evaluation/gemma-full-depth-int8-fp16-tied-multitoken-native-results.json) | `0.674` |

The original-precision NumPy graph was separately checked against the pinned
Transformers checkpoints for this same three-plus-two token shape:
[TinyLlama and DeepSeek](../evaluation/llm/checkpoint-results.json) and
[Gemma](evaluation/gemma-checkpoint-full-reference.json). The host checks
above compare every generated C++ stage against the mapped-precision NumPy
graph. The audit also checks each layer's hidden-state handoff, verifies that
each cached decode appends to the preceding phase's K/V cache, and checks
causal positions and Gemma's alternating attention windows. Regression tests
reject a skipped cache phase, reset hidden state, or reset decode position.
These controls establish the graph schedule and stage arithmetic on
the host; the longer ELFs have **not** completed a full Cyclotron run. FP16
has a small measured effect on the TinyLlama and DeepSeek final logits for
this input. Gemma's INT8 body changes final logits substantially even though
the top logit index is unchanged in all three recorded outputs. It is a
quantized variant, not numerically equivalent to the unquantized checkpoint.
These results contain no device latency or throughput measurement.

For a shorter device control at the **original tensor dimensions**, one-layer
ELFs use the same `[1, 2, 3] → [4] → [5]` token schedule. TinyLlama and
DeepSeek use their FP32 checkpoint images; Gemma uses the INT8 body with
tied FP16 embedding. The independent one-layer Transformers controls pass
for this input:

| Model | Checked stages in one ELF | Radiance build | One-layer upstream reference | Generated C++ | Cyclotron |
| --- | ---: | --- | --- | --- | --- |
| TinyLlama | 60 | [build](evaluation/tinyllama-one-layer-multitoken-build.json) | [passed, max logit error `6.68e-6`](evaluation/tinyllama-one-layer-multitoken-reference.json) | [60 stages passed](evaluation/tinyllama-one-layer-multitoken-native-results.json) | [passed, `tohost=0`, 322,406,150 functional cycles](evaluation/tinyllama-one-layer-multitoken-functional-results.json) |
| DeepSeek-R1-Distill-Qwen-1.5B | 69 | [build](evaluation/deepseek-one-layer-multitoken-build.json) | [passed, max logit error `2.86e-5`](evaluation/deepseek-one-layer-multitoken-reference.json) | [69 stages passed](evaluation/deepseek-one-layer-multitoken-native-results.json) | [passed, `tohost=0`, 825,405,246 functional cycles](evaluation/deepseek-one-layer-multitoken-functional-results.json) |
| Gemma-2-2B-it | 69 | [build](evaluation/gemma-one-layer-int8-fp16-tied-multitoken-build.json) | [passed, max logit error `1.72e-5`](evaluation/gemma-checkpoint-one-layer-reference.json) | [69 stages passed](evaluation/gemma-one-layer-int8-fp16-tied-multitoken-native-results.json) | [passed, `tohost=0`, 7,440,428,542 functional cycles](evaluation/gemma-one-layer-multitoken-functional-results.json) |

These device controls exercise nontrivial causal masks and KV handoffs with
real checkpoint data. The one-layer upstream errors refer to the original
unquantized checkpoint. Gemma's one-layer mapped INT8/FP16 final logits also
differ from that checkpoint by up to `0.206` for this input. A passing
one-layer device result alone does not prove multi-token full-depth device
execution. The separate full-depth Gemma one-token prefill/decode result above
passes all 996 stages.
`run_checkpoint_native.py --allow-partial` reproduces the generated-C++
one-layer checks; without that flag it requires the full pinned layer count.

To rebuild a row, set `MODEL`, `LAYERS`, `CHECKPOINT_DIR`, and `WEIGHT_IMAGE`
to its pinned model name, full layer count, local checkpoint snapshot, and
corresponding full-depth image manifest. The compiler verifies the checkpoint
hashes and places all stages in one ELF. Use a Python environment with NumPy,
PyTorch, and safetensors installed:

```sh
python3 kernels/model_chain/compile_decoder.py \
  --model "$MODEL" --layers "$LAYERS" --prefill 3 --decode-steps 2 \
  --token-ids 1 2 3 4 5 --device-check-all-stages \
  --stages-per-object 20 --checkpoint-dir "$CHECKPOINT_DIR" \
  --weight-image "$WEIGHT_IMAGE" \
  --out-root kernels/model_chain/generated/checkpoint-multitoken-upstream-elf \
  --build
python3 kernels/model_chain/run_checkpoint_native.py \
  --model "$MODEL" \
  --generated-root kernels/model_chain/generated/checkpoint-multitoken-upstream-elf \
  --threads 8 --jobs 2
```

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

The four full-depth checkpoint programs compile to Radiance ELFs. These
builds use FP16 checkpoint images with FP32 arithmetic for TinyLlama and
DeepSeek, INT8 weight formats for Gemma, and FP32 weights and arithmetic for
SmolVLA. The numerical effects of those choices are recorded with the
reference checks above. TinyLlama, DeepSeek, and Gemma passed full-depth
Cyclotron checks for one-token prefill and cached decode. The full SmolVLA
action-chunk attempt reached a 24-hour wall limit without a result; its
targeted Cyclotron controls and full generated-C++ host check pass. The
five-token full-depth decoder ELFs pass host checks, but only their one-layer
five-token controls have completed device checks. Native execution checks the
schedule and math on the host; it does not substitute for a full device run.

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
