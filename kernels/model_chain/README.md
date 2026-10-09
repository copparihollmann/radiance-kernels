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

The current decoder build uses the reduced control dimensions from
`reference.py`: hidden width 32, FFN width 64, head width 8, and vocabulary
64. The default is one layer, three prefill tokens, and one cached decode
token, with deterministic generated FP32 weights. It tests the software
mapping and code generation. It does **not** execute the full checkpoint,
the original model dimensions, MX quantization, or a complete model workload.
The reduction preserves stage order and the grouped-query head ratio, but it
does not preserve every width ratio of the original network.
Generated ELF files and logs remain local under `generated/`.

| Decoder family | Stages in one ELF | Native max error | Cyclotron functional cycles |
| --- | ---: | ---: | ---: |
| TinyLlama | 40 | `9.54e-7` | 128,536 |
| DeepSeek-R1-Distill-Qwen-1.5B | 46 | `1.07e-6` | 130,278 |
| Gemma-2-2B | 46 | `1.07e-6` | 123,153 |

| Model | Full-dimension schedule | Reduced Radiance ELF | Full-checkpoint numerical check | VCS RTL |
| --- | --- | --- | --- | --- |
| TinyLlama | 22 layers | 1 layer, prefill and decode | 22 layers against Transformers | 1-token prefill and 1 cached decode passed |
| DeepSeek-R1-Distill-Qwen-1.5B | 28 layers | 1 layer, prefill and decode | 28 layers against Transformers | Pending |
| Gemma-2-2B | 26 layers | 1 layer, prefill and decode | Checkpoint access pending | Pending |
| SmolVLA-base | Vision and action topology; 16 VLM layers and 160 expert layer calls decomposed | Pending | Static binding of all 500 checkpoint tensors; numerical reference pending | Pending |

The full-dimension schedules are dependency graphs, not compiled model runs.
The checkpoint checks are Python reference checks described in
[`kernels/evaluation/llm/README.md`](../evaluation/llm/README.md); the device
ELFs in this directory use small synthetic weights.
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

1. Bind real checkpoint tensors to device-accessible memory without embedding
   billions of weights in C++ source. Establish a placement and loading policy
   for each model and use the pinned checkpoint reference at intermediate
   boundaries.
2. Tile the full hidden, FFN, head, sequence, and vocabulary dimensions;
   connect the existing MX-Gemmini kernels through the shared-buffer path and
   specify FP8/BF16 conversions. The current executable uses scalar FP32 SIMT.
3. Validate multi-token prefill and cached decode together on RTL, then run
   complete workloads with input and output checks. The three decoder ELFs
   have passed Cyclotron functional execution; one-token TinyLlama prefill and
   cached decode passed RTL. The three-token prefill attempt reached its first
   wall-clock limit without a result.
4. Implement SmolVLA's vision encoder, connector, masks, expert attention,
   and action denoising on the device with a checkpoint numerical control.
   The VLM and expert layers are decomposed in `stitch.py` and their parameter
   shapes match the pinned checkpoint, but the graph is not a numerical backend.

Until those steps pass, the full four-model compilation and end-to-end
performance evaluation remain open. `kernels/evaluation/llm/README.md`
describes the model graphs and existing TinyLlama/DeepSeek checkpoint checks.
