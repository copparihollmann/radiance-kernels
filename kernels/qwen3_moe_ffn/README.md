# Reduced Qwen3 sparse expert FFN

This scalar Radiance program runs one sparse expert FFN from router input to
weighted output. For each token it computes 16 router logits and their
softmax, selects the top eight experts, renormalizes those eight scores,
applies each selected expert's gate/up projection and SiLU, applies its down
projection, and sums the scored outputs. The seven-token fixture uses the
same seeded, random-weight Transformers 5.3 block as the
[independent CPU check](../evaluation/llm/qwen3-moe-router-reference.json).
The generator checks its input, selected IDs, scores, and output against
that record and the upstream block before emitting device data.

Run from the repository root. Set `TRANSFORMERS_PYTHON` to a Python with
Torch and Transformers 5.3; build files go under the task temporary directory:

```sh
python3 kernels/qwen3_moe_ffn/test_device.py \
  --reference-python "$TRANSFORMERS_PYTHON" \
  --out "$TMPDIR/qwen3-moe-ffn"
```

The [Cyclotron result](evaluation/transformers-reduced-cyclotron.json)
passed on two Muon cores in 78,125 functional cycles. It checked 56 expert
indices exactly, 56 selected scores, and 224 FFN output values. Maximum
absolute errors against the independent NumPy calculation were `1.76e-6`
for scores and `1.79e-6` for output. The
[normalization control](evaluation/no-score-renorm-control.json) removed the
selected-score renormalization and failed with `tohost=561`. The positive
build directory used about 1.5 MiB.

This validates one reduced block, not the 48-layer Qwen3-30B-A3B model or
its checkpoint. The scalar kernel does not exercise MX-Gemmini. Cyclotron
functional cycles are not RTL latency or a hardware performance result.
