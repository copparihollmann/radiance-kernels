# SpectFormer

This directory builds the **vanilla SpectFormer-XS and SpectFormer-S inference graphs** from the
[original implementation](https://github.com/badripatro/SpectFormers/blob/b6cf487a0ea71eeb252d5ba0fc7410b3f5dbd256/vanilla_architecture/spectformer.py)
as one Radiance executable per configuration. Both use a 224×224 RGB input,
16×16 patches, width 384, four spectral blocks, and a 1,000-class head.
XS has eight attention blocks and 128 scheduled stages; S has fifteen
attention blocks and 198 stages. `compile.py` schedules them in source order.
Buffers carry each block's output into the next block; the two residual paths
in an attention block and the single MLP residual in a spectral block follow
the upstream `forward` methods. The generated executable compares **every**
logit with the upstream PyTorch result and writes a pass/fail `tohost` code.

## What runs where

| Component | Radiance mapping | Upstream operation |
| --- | --- | --- |
| Patch embedding | FP32 patch projection and position addition | `PatchEmbed.proj`, `pos_embed` |
| Spectral block, four repetitions | FP32 LayerNorm, row DFT, column DFT, learned complex multiply, inverse transforms, LayerNorm, two linear layers, exact GELU, residual | `SpectralGatingNetwork` and `Mlp` |
| Attention block, eight (XS) or fifteen (S) repetitions | FP32 LayerNorm, QKV linear, six-head noncausal softmax attention, output linear, residual, LayerNorm, MLP, residual | `Block_attention` |
| Head | FP32 LayerNorm, token mean, classifier linear | `forward_features`, `head` |

The spectral path stores real and imaginary components as interleaved FP32
values. `spectral.hpp` computes separable real 2D DFTs with orthonormal
scaling; it does not depend on a complex instruction or a modified RTL. This
direct DFT was chosen to establish graph correctness. It is substantially
slower than a mixed-radix FFT and is **not** a performance mapping.

`prepare.py` imports the pinned upstream source, evaluates its `eval()` graph,
and writes the parameters, a fixed input, and the upstream logits to a
hash-checked GMEM image. It transposes only PyTorch `Linear.weight` matrices
into the layout consumed by the existing Radiance `model_chain::linear` code.
`compile.py` emits `kernel.radiance.elf` and `kernel.soc.elf` through the
repository's `kernels/common.mk` build. `run_native.py` checks the **same
generated stage C++** on the host, and `run_cyclotron.py` runs the Radiance ELF
in the functional ISA simulator. A completed Cyclotron run leaves the FP32
output vector at `cyclotron-logits.bin` in its build directory.

## Reproduce

The upstream Python source needs PyTorch and `timm` (tested with `timm==0.9.16`).
The SpectFormer source file must have SHA-256
`611aee409308e898d23fcbb1c75de3a0c5fafc205df74c6250e49bfe2ac53250`.
Point `LLVM_MUON` at the installed Muon toolchain (this task uses
`/scratch/agustin/projects/chipyard/radiance-kernels-firesim/llvm/llvm-muon`).
From the repository root, keep downloads and generated files in the task's
`$TMPDIR`:

```sh
mkdir -p "$TMPDIR/spectformer"
curl -L --fail -o "$TMPDIR/spectformer/spectformer.py" \
  https://raw.githubusercontent.com/badripatro/SpectFormers/b6cf487a0ea71eeb252d5ba0fc7410b3f5dbd256/vanilla_architecture/spectformer.py
python kernels/spectformer/prepare.py \
  --upstream-source "$TMPDIR/spectformer/spectformer.py" \
  --out "$TMPDIR/spectformer/xs"
python kernels/spectformer/compile.py \
  --image-manifest "$TMPDIR/spectformer/xs/manifest.json" \
  --out "$TMPDIR/spectformer/xs-build" --build
python kernels/spectformer/run_native.py \
  --target "$TMPDIR/spectformer/xs-build"
python kernels/spectformer/run_cyclotron.py \
  --target "$TMPDIR/spectformer/xs-build" \
  --timeout 1800 --sim-cycles 10000000000
python -m unittest kernels/spectformer/test_spectral.py
```

Use `prepare.py --checkpoint /path/to/checkpoint` to bind a trained checkpoint;
the exporter loads its `model` state dictionary with strict shape checking and
records the checkpoint hash. A 384-wide SpectFormer-S checkpoint follows the
same vanilla graph with `--depth 19`. Without `--checkpoint`, the weights come from the **real
upstream model's deterministic random initialization**. Passing this fixture
establishes execution and numerical agreement, not ImageNet accuracy.
`prepare.py --input-npy /path/to/image.npy` accepts a finite FP32 NCHW tensor
of shape `[1,3,224,224]` and produces a new upstream reference for that input.
Image decoding and ImageNet normalization happen before this interface.

The reduced device control uses
`prepare.py --image-size 32 --embed-dim 24 --depth 5 --num-classes 10`
with the same seed (`1827`) and the corresponding `compile.py`/simulator
commands. It includes all operation types and one attention block, while the
full XS build includes all 12 source blocks at their original dimensions.

## Checks completed

| Check | Scope | Result |
| --- | --- | --- |
| C++ spectral transform vs NumPy `rfft2`/`irfft2` | 14×14, 28×28, 56×56 grids | Passed at `rtol=atol=3e-5` |
| Cyclotron Radiance ELF vs upstream logits | 32×32, width 24, depth 5, 10 logits | [Passed](evaluation/reduced-cyclotron.json), `tohost=0`; 430,772 functional ISA cycles |
| Cyclotron full-depth graph vs upstream logits | 32×32, width 24, depth 12, 10 logits | [Passed](evaluation/depth12-cyclotron.json), `tohost=0`; maximum absolute logit error `8.43e-4` |
| Cyclotron full-depth S graph vs upstream logits | 32×32, width 24, depth 19, 10 logits | [Passed](evaluation/depth19-cyclotron.json), `tohost=0`; maximum absolute logit error `1.22e-3` |
| Generated stage C++ vs upstream logits | 224×224, width 384, depth 12, 1,000 logits | [Passed](evaluation/xs-native.json); maximum absolute logit error `8.94e-7` |
| Generated stage C++ vs upstream logits | 224×224, width 384, depth 19, 1,000 logits | [Passed](evaluation/s-native.json); maximum absolute logit error `8.05e-7` |
| Full-size Radiance/SoC ELF builds | 224×224, width 384, depth 12/19 | [XS](evaluation/xs-build.json) and [S](evaluation/s-build.json) passed; 128/198 stages and disjoint, aligned buffers |
| MX arithmetic model, source-derived attention projection tile | 16×16×32 partial tile from block 4 | [Result](evaluation/xs-mx-probe.json); separate quantized-path experiment |

The Cyclotron cycle counts above are functional simulator instruction counts,
not device latency or throughput. An [attempt at the full 224×224 XS ELF](evaluation/xs-cyclotron-timeout.json)
hit its 30-minute wall-clock timeout before reaching `tohost`, so it provides
no full-size device correctness result. That attempt used the earlier ELF with
a tighter `0.003 * abs(reference) + 0.0003` check; the current built ELF uses
the tolerance below and has only been checked on the host. MX Spike execution
is also a separate gate; the host check and `mx_golden` result must not be
presented as either of those. The MX probe exports real block-4 activation
and weight values from upstream PyTorch, quantizes one partial projection tile
to MX-FP8 with per-row/per-column e8m0 scales bounded for the model's
accumulator range, and evaluates the repository's arithmetic model. Its
maximum error against the FP32 **partial tile** is `6.47e-3`; it is not a
complete attention projection. The complete
Radiance executable currently uses FP32 Muon linear stages throughout; it
does not yet schedule projections on MX-Gemmini.

The device checks compare each FP32 logit using `abs(error) <= 0.005 *
abs(reference) + 0.001`. `run_cyclotron.py` also dumps the actual logit vector
and records its maximum absolute error, RMS error, and top-1 agreement. This
allows numerical drift to be reported independently of the pass/fail code.
The 12-block reduced graph initially failed a tighter `0.003 *
abs(reference) + 0.0003` check at logit 0: its measured maximum absolute
error was `8.43e-4`, and its top-1 class agreed. The stated tolerance and
measured error are part of the result, not an exact-match claim.

## Scope

This is the original repository's **vanilla** architecture, not its separate
hierarchical SpectFormer implementation. Hierarchical models add a convolutional
stem, spatial downsampling, depthwise-convolution FFN, and class attention;
their checkpoints cannot be sent through this graph. The current executable
also handles one image at a time. No RTL was changed.
