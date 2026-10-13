# SpectFormer inference

This directory maps the pinned [SpectFormers source](https://github.com/badripatro/SpectFormers/tree/b6cf487a0ea71eeb252d5ba0fc7410b3f5dbd256) into connected Radiance inference programs. It covers vanilla XS/S and hierarchical S/B/L at 224×224 resolution with 1,000 output logits. Each variant builds a Radiance ELF and a SoC ELF through `kernels/common.mk`. A reusable ELF reads a new normalized FP32 RGB image from GMEM without recompilation and writes its logits to a fixed output buffer.

The mapping follows the upstream `eval()` methods. It preserves the graph's branches and residuals rather than treating a flattened operator list as the model. All arithmetic currently runs on Muon FP32. Complex spectral weights are pairs of FP32 values; no complex ISA instruction or RTL change is needed.

| Family | Variants | Connected graph | Stages |
| --- | --- | --- | ---: |
| Vanilla | XS, S | Patch projection and position embedding → four spectral blocks → eight or fifteen attention blocks → token mean → classifier | 128, 198 |
| Hierarchical | S, B, L | Convolutional stem → four stages with downsampling; spectral mixing in stages 1–2 and attention in stages 3–4 → class attention → classifier and dense auxiliary head | 233, 305, 403 |

The hierarchical inference path keeps the upstream class token construction and dense auxiliary head: class logits are combined with half the per-class maximum over auxiliary token logits. Training-only token mixing is absent from upstream `eval()`. The spectral transform uses a separable 2D transform with a 7×M factorization for the 14, 28 and 56 point axes. This improves the scalar mapping but is still a correctness-first implementation, not a performance-tuned FFT. The classifier, convolutions, attention, normalization, GELU, and residual operations are also scalar FP32 kernels. `compile.py` and `hierarchical_compile.py` state the stage order and intermediate buffers explicitly.

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

The runner writes `logits.bin`, the reference vector, Cyclotron log, and a JSON receipt with input, weight, ELF, simulator, and output hashes. `--no-reference` runs inference and dumps logits without PyTorch; its receipt says `reference_checked: false`.

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
| Full-size ELF builds | Vanilla XS/S and hierarchical S/B/L built: 128/198/233/305/403 stages. |
| Full-size generated C++ vs upstream PyTorch | All five variants pass on the host; maximum absolute logit error is at most `6.11e-5` on random-weight fixtures. |
| Trained-weight held-out inputs | Vanilla XS: `1.67e-6`, `2.62e-6` maximum logit error. Hierarchical S: `5.01e-6`, `3.34e-6`. Each pair used one unchanged ELF and checkpoint with two distinct inputs. |
| Intermediate graph checks | Every upstream block output matched: 12/12 XS, 16/16 hierarchical S, 22/22 B, and 30/30 L. A deliberately altered hierarchical trace was rejected. |
| Spectral transform vs NumPy | 14, 28, and 56 point transforms pass `rtol=atol=3e-5` in `test_spectral.py`. |
| Reusable device control | A reduced 32×32, width-24, five-block vanilla ELF ran two images with one unchanged ELF and weight image in Cyclotron. Both passed all-logit upstream checks; maximum errors were `3.12e-4` and `5.24e-4`. |
| Full-size device run | The optimized 224×224 vanilla XS ELF timed out after 600 seconds before producing `tohost` or logits. No full-size variant has a completed device inference result yet. |

Earlier reduced [depth-12](evaluation/depth12-cyclotron.json) and [depth-19](evaluation/depth19-cyclotron.json) Cyclotron controls pass with the original small width and image size. The new two-input control verifies runtime input loading and output dumping. Cyclotron's functional cycle count is an ISA simulation count, not measured latency, throughput, utilization, or PPA. The timeout is a wall-clock limit, not evidence of device correctness failure. The complete programs do not use MX-Gemmini yet; the [MX arithmetic probe](evaluation/xs-mx-probe.json) is a partial tile experiment, not MX Spike execution.

The next execution step is to make one full-size 224×224 program complete in a practical device simulator and compare dumped logits with the pinned upstream reference, then repeat across variants and trained checkpoints. Scalar convolutions and dense projections, especially at 56×56, are likely cost centers; stage-level simulator progress and tile mappings should be measured before changing arithmetic or claiming performance. MX-Gemmini projections and a Spike check are separate integration steps. No RTL, FPGA bitstream, or hardware performance result is part of this directory.
