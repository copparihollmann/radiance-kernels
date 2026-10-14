# SpectFormer inference

This directory maps the pinned [SpectFormers source](https://github.com/badripatro/SpectFormers/tree/b6cf487a0ea71eeb252d5ba0fc7410b3f5dbd256) into connected Radiance inference programs. It covers vanilla XS/S and hierarchical S/B/L at 224×224 resolution with 1,000 output logits. Each variant builds a Radiance ELF and a SoC ELF through `kernels/common.mk`. A reusable ELF reads a new normalized FP32 RGB image from GMEM without recompilation and writes its logits to a fixed output buffer.

The mapping follows the upstream `eval()` methods. It preserves the graph's branches and residuals rather than treating a flattened operator list as the model. All arithmetic currently runs on Muon FP32. Complex spectral weights are pairs of FP32 values; no complex ISA instruction or RTL change is needed.

| Family | Variants | Connected graph | Stages |
| --- | --- | --- | ---: |
| Vanilla | XS, S | Patch projection and position embedding → four spectral blocks → eight or fifteen attention blocks → token mean → classifier | 128, 198 |
| Hierarchical | S, B, L | Convolutional stem → four stages with downsampling; spectral mixing in stages 1–2 and attention in stages 3–4 → class attention → classifier and dense auxiliary head | 233, 305, 403 |

The hierarchical inference path keeps the upstream class token construction and dense auxiliary head: class logits are combined with half the per-class maximum over auxiliary token logits. Training-only token mixing is absent from upstream `eval()`. The spectral transform is separable. The host uses a 7×M factorization for 14, 28 and 56 point axes; the Radiance device uses the direct transform because the factored device path fails a numerical check. This is a correctness-first scalar mapping, not a performance-tuned FFT. The classifier, convolutions, attention, normalization, GELU, and residual operations are also scalar FP32 kernels. `compile.py` and `hierarchical_compile.py` state the stage order and intermediate buffers explicitly.

## Files and data flow

| File | Role |
| --- | --- |
| `prepare.py`, `hierarchical_prepare.py` | Import pinned upstream source, optionally load a strict checkpoint, export aligned weights, one input, and upstream logits. Record shapes, addresses, and hashes. |
| `compile.py`, `hierarchical_compile.py` | Generate the connected C++ schedule and build Radiance and SoC ELFs. |
| `spectral.hpp`, `hierarchical_ops.hpp` | FP32 operations called by the generated schedules. |
| `reference.py` | Run upstream PyTorch on a new input to make an independent logit reference. |
| `run_reusable.py` | Load separate weights and input into Cyclotron, run an ELF, dump logits, and compare with upstream. `--no-reference` dumps inference logits without importing PyTorch. |
| `run_native.py`, `trace_blocks.py` | Execute generated stage C++ on the host and compare final logits and optional block outputs with hooks on the upstream graph. |
| `train_checkpoint.py` | Make a small, reproducible trained-weight functional fixture for vanilla XS or hierarchical S. |
| `probe_mx.py`, `probe_mx_spike.py` | Extract a source-derived attention projection tile, run the MX arithmetic model, and compare its BF16 output bit for bit with an MX-enabled Spike execution. |

The exporter writes `image.bin` for the original self-checking mode, plus `weights.bin`, `input.bin`, and `golden.bin` for runtime input mode. `manifest.json` records addresses and SHA-256 values. Runtime device inference does not embed golden logits into the ELF; `run_reusable.py` checks the dumped output afterward. The host check uses the same generated stage code. It checks graph scheduling and arithmetic but does not establish that a full-size device run has completed.

## Reproduce

Use Python with PyTorch, torchvision, timm, NumPy, and Pillow. This work was checked with PyTorch 2.10 and timm 1.0.27. Keep downloaded source, checkpoints, and generated output outside the repository, for example under the task's `$TMPDIR`.

```sh
mkdir -p "$TMPDIR/spectformer"
curl -L --fail -o "$TMPDIR/spectformer/vanilla.py" \
  https://raw.githubusercontent.com/badripatro/SpectFormers/b6cf487a0ea71eeb252d5ba0fc7410b3f5dbd256/vanilla_architecture/spectformer.py
curl -L --fail -o "$TMPDIR/spectformer/hierarchical.py" \
  https://raw.githubusercontent.com/badripatro/SpectFormers/b6cf487a0ea71eeb252d5ba0fc7410b3f5dbd256/hierarchical_architecture/spectformer.py
```

The exporters reject source files whose hashes differ from `611aee409308e898d23fcbb1c75de3a0c5fafc205df74c6250e49bfe2ac53250` (vanilla) or `38f0097924ecb270d6b7b1e90daf2423c1ffc6c6cea70c570e11f89bb73e68c6` (hierarchical). Set `LLVM_MUON` to your Muon toolchain before building. From the repository root, a full-size vanilla XS build and host check are:

```sh
python kernels/spectformer/prepare.py \
  --upstream-source "$TMPDIR/spectformer/vanilla.py" \
  --out "$TMPDIR/spectformer/xs"
python kernels/spectformer/compile.py \
  --image-manifest "$TMPDIR/spectformer/xs/manifest.json" \
  --out "$TMPDIR/spectformer/xs-build" --runtime-input --build
python kernels/spectformer/run_native.py \
  --target "$TMPDIR/spectformer/xs-build" --trace-blocks
```

For vanilla S, add `--depth 19` to `prepare.py`. The hierarchical variants use:

```sh
python kernels/spectformer/hierarchical_prepare.py \
  --upstream-source "$TMPDIR/spectformer/hierarchical.py" \
  --variant h-s --out "$TMPDIR/spectformer/h-s"
python kernels/spectformer/hierarchical_compile.py \
  --image-manifest "$TMPDIR/spectformer/h-s/manifest.json" \
  --out "$TMPDIR/spectformer/h-s-build" --build
python kernels/spectformer/run_native.py \
  --target "$TMPDIR/spectformer/h-s-build" --trace-blocks
```

Replace `h-s` with `h-b` or `h-l` for the other variants. `--checkpoint` on either exporter loads an exact state dictionary, and `--input-npy` on vanilla or `--input` on hierarchical supplies the exported image. A later input can be supplied to `run_native.py --input image.npy` without regenerating the program. The NPY interface is finite FP32 NCHW `[1,3,224,224]`; `reference.py` also accepts RGB image files and applies bicubic resize to 256, center crop to 224, and ImageNet normalization.

For device inference, point `--cyclotron-root` at a built Cyclotron checkout:

```sh
python kernels/spectformer/run_reusable.py \
  --target "$TMPDIR/spectformer/xs-build" \
  --input /path/to/normalized-image.npy \
  --out "$TMPDIR/spectformer/xs-device-run" \
  --cyclotron-root /path/to/cyclotron --timeout 1800
```

The runner writes `logits.bin`, the reference vector, Cyclotron log, and a JSON receipt with input, weight, ELF, simulator, and output hashes. `--no-reference` runs inference and dumps logits without PyTorch; its receipt says `reference_checked: false`. To check that saved run later, without repeating the device simulation, use `verify_reusable.py` with the same target, input, output directory, and Cyclotron root. It verifies the run hashes and completion log, computes the pinned upstream output, and writes `verification.json` plus `verified-reference.bin`. A no-reference run is only a completion check until this verification passes.

The runner streams Cyclotron output directly to `cyclotron.log` and scans
completion and preload markers one line at a time with bounded memory. A new
reduced 32×32, ten-logit control reproduced the prior
output SHA-256 and 430,609 functional cycles with this path; its
[upstream verification](evaluation/streamed-log-control.json) passed all ten
logits with maximum absolute error `5.24e-4`. A deliberate zero-second
timeout produced a [failed receipt](evaluation/streamed-log-timeout-control.json)
with no output hash or cycle count, and the upstream verifier rejected it.

To isolate a vanilla block, compile a window with `--start-at-block N` and
`--stop-after-block N`. The runner computes the pinned upstream output before
block N, preloads that tensor into the device token buffer, and compares the
device block output with the next upstream hook. This checks one device block
without replaying earlier device blocks; it cannot establish full-network
correctness or accumulated numerical error. Block-seeded runs require the
reference check and reject `--no-reference`. For example:

```sh
python kernels/spectformer/compile.py --image-manifest "$EXPORT/manifest.json" \
  --out "$TMPDIR/spectformer/block2" --runtime-input \
  --start-at-block 2 --stop-after-block 2 --build
python kernels/spectformer/run_reusable.py --target "$TMPDIR/spectformer/block2" \
  --input "$HELDOUT_NPY" --out "$TMPDIR/spectformer/block2-run" \
  --cyclotron-root "$CYCLOTRON_ROOT" --timeout 1800 --sim-cycles 1000000000
```

The trained fixtures below came from the [scikit-learn 1.6.1 handwritten digits data](https://github.com/scikit-learn/scikit-learn/blob/1.6.1/sklearn/datasets/data/digits.csv.gz), SHA-256 `09f66e6debdee2cd2b5ae59e0d6abbb73fc2b0e0185d2e1957e9ebb51e23aa22`. `train_checkpoint.py` takes four examples from each of ten digit classes, resizes them to 224×224, and makes 20 AdamW steps on the full upstream model. Digits 0–9 occupy output indices 0–9 in the original 1,000-logit head. Two separate images are held out for inference checks. This is a trained-weight test of checkpoint loading and inference, with no ImageNet accuracy claim. Each training receipt records the source and dataset hashes, selected indices, seed, optimizer settings, and checkpoint hash. Checkpoint hashes: `2ca40d05343b340fbb3c4fc8ec83ea1d0c5bd3a20c4066f4dbae767e50db9ee5` (XS) and `3a8d80d44d4a280cf20e8db2048b5a526371186fc0c5ded7729b4d57604dd28e` (hierarchical S).

```sh
curl -L --fail -o "$TMPDIR/spectformer/digits.csv.gz" \
  https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.6.1/sklearn/datasets/data/digits.csv.gz
python kernels/spectformer/train_checkpoint.py --variant xs \
  --upstream-source "$TMPDIR/spectformer/vanilla.py" \
  --data-root "$TMPDIR/spectformer/digits.csv.gz" \
  --out "$TMPDIR/spectformer/xs-trained"
python kernels/spectformer/train_checkpoint.py --variant h-s \
  --upstream-source "$TMPDIR/spectformer/hierarchical.py" \
  --data-root "$TMPDIR/spectformer/digits.csv.gz" \
  --out "$TMPDIR/spectformer/h-s-trained"
```

Pass the resulting `checkpoint.pth` to the matching exporter. The exporter checks every checkpoint tensor against the source model; `run_native.py --input` or `run_reusable.py --input` then takes either held-out NPY file.

## Checks completed

The [connected-model evaluation receipt](evaluation/connected-models.json) records build hashes, maximum errors, output hashes where captured, and source receipt hashes. The tolerance is `abs(actual - reference) <= 0.005 * abs(reference) + 0.001` for every logit or checked block element. A passing comparison also requires finite outputs. Reported error magnitudes are measured errors, not the tolerance.

| Check | Result |
| --- | --- |
| Current full-size ELF builds | The [corrected build record](evaluation/current-variant-builds.json) hashes Radiance and SoC ELFs for vanilla XS/S and hierarchical S/B/L: 128/198/233/305/403 stages. Vanilla S uses an embedded random-weight fixture; the other four ELFs read runtime input. |
| Current generated C++ vs upstream PyTorch | All five host runs pass all 1,000 logits and their 12/19/16/22/30 intermediate block checks. Maximum absolute logit error is `6.01e-5` across these runs. XS and hierarchical S use trained checkpoints; S, B, and L use pinned random-weight exports. |
| Trained-weight held-out inputs | Vanilla XS: `1.67e-6`, `2.62e-6` maximum logit error. Hierarchical S: `5.01e-6`, `3.34e-6`. Each pair used one unchanged ELF and checkpoint with two distinct inputs. |
| Corrected full-size trained host execution | The current 128-stage XS and 233-stage hierarchical S generated programs pass against the pinned upstream checkpoints on held-out input 0. All 1,000 logits are finite and within tolerance; maximum absolute errors are `2.38e-6` and `4.29e-6`. The [XS trace](evaluation/xs-trained-fixed-full-trace-native.json) checks all 12 blocks, and the [hierarchical S trace](evaluation/hs-trained-fixed-full-trace-native.json) checks all 16 blocks. Every block passes its upstream forward-hook comparison. These are host runs of generated stage code, with the device ELF hashes recorded in the receipts. |
| Intermediate graph checks | Every upstream block output matched: 12/12 XS, 16/16 hierarchical S, 22/22 B, and 30/30 L. A deliberately altered hierarchical trace was rejected. |
| Spectral transform vs NumPy | 14, 28, and 56 point transforms pass `rtol=atol=3e-5` in `test_spectral.py`. |
| Reusable device control | A reduced 32×32, width-24, five-block vanilla ELF ran two images with one unchanged ELF and weight image in Cyclotron. Both passed all-logit upstream checks; maximum errors were `3.12e-4` and `5.24e-4`. |
| Full-size device run | The [prior trained vanilla XS run](evaluation/xs-trained-old-full-cyclotron.json) completed 3,880,531,488 functional cycles, but 984/1,000 logits failed the upstream comparison (maximum error `0.476`). The [corrected 128-stage run](evaluation/xs-trained-fixed-full-cyclotron.json) completed 3,435,965,826 functional cycles with finite logits, but 241/1,000 remain outside tolerance (maximum error `0.00818`, RMS `0.00249`). Both runs agreed on the top-1 class; neither passed the full-vector check. |
| Trained XS device prefixes | Corrected [block 0](evaluation/xs-trained-block0-cyclotron.json) and [block 1](evaluation/xs-trained-block1-cyclotron.json) pass all 75,264 upstream tensor values each. Block 1 completes 23 stages in 431,266,859 functional cycles with maximum absolute error `6.32e-6`. The full-logit mismatch therefore begins later in the graph. |
| Hierarchical S device run | Before the FFT/GELU fix, [block 1.0](evaluation/hs-trained-old-block1-0-cyclotron.json) completed 21 stages but 171,209/200,704 elements failed tolerance (maximum error `0.239`). The corrected [21-stage block 1.0 run](evaluation/hs-trained-block1-0-cyclotron.json) now passes all 200,704 values against the upstream checkpoint, with maximum absolute error `1.73e-4` and 639,819,828 functional Cyclotron cycles. The prior [full 233-stage run](evaluation/hs-trained-old-full-cyclotron.json) completed with 1,000/1,000 logits as NaN; corrected full-model execution remains pending. |

The [full-size prefix receipt](evaluation/full-size-prefixes.json) records the earlier failing device build. Its one-stage trained XS patch embedding matched all 75,264 upstream tensor elements (maximum absolute error `5.60e-6`, 42,985,699 functional cycles), and the hierarchical S stem matched all 200,704 elements (maximum error `1.73e-4`, 408,283,353 cycles). The old 12-stage XS block 0 failed with a maximum error of `0.783`. Stage probes traced this to the factored FFT and the device `erff`/`expf` path. The device FFT now uses the direct transform; GELU uses a degree-16 Chebyshev approximation to the upstream exact GELU, with a maximum observed error of about `1.2e-6` on the 301,056 checkpoint-backed FC1 values. `test_device_fft.py` and `test_device_gelu.py` each failed with the prior implementation and pass with these changes. After the fix, `python3 -m unittest kernels.spectformer.test_device_fft kernels.spectformer.test_device_gelu kernels.spectformer.test_spectral -v` passed: the device FFT test covers 14-, 28-, and 56-point rows, the device GELU test checks the in-place operation, and the host spectral test checks the three spatial grids. The corrected [trained block-0 receipt](evaluation/xs-trained-block0-cyclotron.json) covers all 12 stages: Cyclotron completed in 237,126,163 functional cycles, and all 75,264 outputs matched a freshly computed upstream forward-hook tensor with maximum absolute error `5.84e-6`. That fresh reference used PyTorch 2.9.1+cpu, timm 0.9.16, and NumPy 2.2.6; it differs from the previously saved reference by at most `3.58e-7`. Build prefixes with `compile.py --runtime-input --stop-after-patch` or `--stop-after-block 0`, and `hierarchical_compile.py --stop-after-stem` or `--stop-after-block 1.0`. A prefix pass does not establish final logits or full-model device execution.

Earlier reduced [depth-12](evaluation/depth12-cyclotron.json) and [depth-19](evaluation/depth19-cyclotron.json) Cyclotron controls pass with the original small width and image size. The new two-input control verifies runtime input loading and output dumping. Cyclotron's functional cycle count is an ISA simulation count, not measured latency, throughput, utilization, or PPA. The timeout is a wall-clock limit, not evidence of device correctness failure. The complete programs do not use MX-Gemmini yet; the [MX arithmetic probe](evaluation/xs-mx-probe.json) is a partial tile experiment, not MX Spike execution.

An additional [trained XS tile check](evaluation/xs-trained-mx-spike.json) executed a source-derived 16×16×32 attention projection tile in **MX-enabled Spike**. All 256 BF16 results matched `mx_golden` bit for bit. A [one-bit negative control](evaluation/xs-trained-mx-spike-negative.json) produced one mismatch and a nonzero exit. The [FP8 tile receipt](evaluation/xs-trained-mx-probe.json) reports maximum absolute error `0.00504` against its FP32 *partial projection*; exact agreement between two MX models does not establish FP32 model equivalence. `probe_mx.py` reads the input tensor from the export, so a trained checkpoint and held-out input can be used together. Reproduce the Spike check after that probe with:

```sh
python kernels/spectformer/probe_mx.py \
  --image-manifest "$TMPDIR/spectformer/xs/manifest.json" \
  --out "$TMPDIR/spectformer/xs-mx-tile"
python kernels/spectformer/probe_mx_spike.py \
  --tile "$TMPDIR/spectformer/xs-mx-tile" \
  --out "$TMPDIR/spectformer/xs-mx-spike" \
  --gemmini-source /path/to/mx-enabled/gemmini-rocc-tests \
  --spike /path/to/mx-enabled/spike \
  --gemmini-library-dir /path/to/mx-enabled/libgemmini \
  --gcc /path/to/riscv64-unknown-linux-gnu-gcc \
  --pk /path/to/pk
```

The next step is to locate the first block where the corrected full-size device
output diverges from the pinned upstream checkpoint, fix that numerical path,
then repeat the complete logit comparison across variants. The failed full
run's cycle count is diagnostic only. Scalar convolutions and dense
projections, especially at 56×56, are likely cost centers; stage-level
simulator progress and tile mappings should be measured before claiming
performance. MX-Gemmini projections and a Spike check are separate
integration steps. No RTL, FPGA bitstream, or hardware performance result is
part of this directory.

## FPGA handoff artifacts

`package_handoff.py` copies a built runtime-input ELF, its separate weight and input images, an upstream output vector, and a numerically checked Cyclotron receipt when available. Pass `verification.json` after a `--no-reference` run, or a `result.json` produced with the reference check enabled. A completion-only receipt is rejected as a simulator pass. The packager verifies hashes before copying the large weight image and writes a per-case `load-map.json` with the exact GPU addresses and output readback location. For example:

```sh
python kernels/spectformer/package_handoff.py --out "$TMPDIR/spectformer/fpga-handoff" \
  --case vanilla-xs-patch "$TMPDIR/spectformer/xs-trained-patch-probe" \
  "$TMPDIR/spectformer/xs-trained-export" \
  "$TMPDIR/spectformer/xs-trained-patch-device/result.json"
```

These are **handoff inputs**, not ready-to-submit FireSim queue jobs. The current fused SoC ELF only waits for the GPU; it neither loads the separate weight/input images nor checks the output. The existing U250 queue flow used self-contained ELFs. An FPGA run therefore needs a verified preload path or a new self-contained host/ELF, plus full output readback. The package records this explicitly and marks any missing full-model simulator result as unverified.
