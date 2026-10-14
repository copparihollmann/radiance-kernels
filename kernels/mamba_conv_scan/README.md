# Connected Mamba2 convolution and selective scan

This reduced Radiance program runs a causal depthwise convolution, then reads
its output in a selective scan **within the same ELF**. It carries both the
four-tap convolution state and the SSM state across a five-token prefill and
two one-token decode calls. The convolution's 48 output channels split into
32 scan `x` channels, eight `B` values, and eight `C` values. The scan also
applies the projected `dt` bias and softplus before the recurrence.

The fixture uses one seeded, random-weight Transformers 5.3 Mamba2 mixer at
reduced width. Its inputs, weights, and prefill/decode schedule are checked
against the independently generated convolution and scan fixtures. The scan
golden uses the previously verified convolution device output. The joined
device result is compared against both independent device results by exact
output hash, and against the NumPy goldens by FP32 tolerance. The independent
goldens were also compared with the Transformers CPU fallback.

Run from the repository root after the standalone upstream convolution and
its scan handoff have passed. Set `TRANSFORMERS_PYTHON` to a Python with Torch
and Transformers 5.3. All build files stay under the task temporary directory.

```sh
python3 kernels/mamba_depthwise_conv/test_device.py --source upstream \
  --reference-python "$TRANSFORMERS_PYTHON" \
  --out "$TMPDIR/mamba-conv-upstream"
python3 kernels/mamba_selective_scan/test_device.py --source upstream \
  --reference-python "$TRANSFORMERS_PYTHON" \
  --conv-output "$TMPDIR/mamba-conv-upstream/output.bin" \
  --conv-result "$TMPDIR/mamba-conv-upstream/result.json" \
  --out "$TMPDIR/mamba-conv-scan-handoff"
python3 kernels/mamba_conv_scan/test_device.py \
  --conv-dir "$TMPDIR/mamba-conv-upstream" \
  --scan-dir "$TMPDIR/mamba-conv-scan-handoff" \
  --out "$TMPDIR/mamba-connected"
```

The [recorded check](evaluation/transformers-reduced-cyclotron.json) passed
Cyclotron with two Muon cores in 26,032 functional cycles. It checked all 336
convolution outputs, 192 final convolution state values, 224 scan outputs,
and 128 final SSM state values. Maximum errors against NumPy were
`1.56e-6`, zero, `1.49e-7`, and `4.47e-8`, respectively. The four device
output hashes exactly matched the separate convolution and handoff runs.

This is a connected pair of mixer primitives, not a full Mamba layer or a
full Nemotron model. Input projection, gate, normalization, output projection,
the remaining blocks, and pinned checkpoint weights are absent. Cyclotron's
functional cycle count is not RTL latency or a hardware performance result.
