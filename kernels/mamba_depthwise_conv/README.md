# Mamba2 causal depthwise convolution

This Radiance kernel takes the projected convolution channels and applies the
causal depthwise filter, bias, and SiLU activation. Each channel carries the
last four raw projected samples from a five-token prefill into two one-token
decode calls. The final convolution state and every activated output are
checked separately. The four-tap width matches the pinned hybrid model; the
test uses 48 channels at reduced width.

The standalone check builds outside the checkout:

```sh
python3 kernels/mamba_depthwise_conv/test_device.py \
  --out "$TMPDIR/mamba-conv-synthetic"
python3 kernels/mamba_depthwise_conv/test_device.py --source upstream \
  --reference-python "$TRANSFORMERS_PYTHON" \
  --out "$TMPDIR/mamba-conv-upstream"
```

The synthetic input starts with nonzero state. The upstream input uses one
random-weight Transformers 5.3 Mamba2 mixer, starts with zero state, and
captures its convolution activation in prefill and cached decode. NumPy's
serial reference checks the captured output and final cache state before the
device build. The [upstream fixture receipt](evaluation/transformers-reduced-cyclotron.json)
then records a Cyclotron pass for all 336 outputs and 192 state elements. The
[synthetic receipt](evaluation/cyclotron.json) exercises a separate input and
reset-at-decode control. Neither run uses the pinned checkpoint.

`test_device.py` saves the verified device convolution output as `output.bin`
in its task temporary directory. The scan test can use that file as its input
with `--conv-output` and `--conv-result`. This handoff runs two separate ELFs
with a host file between them. It checks the numerical boundary; it is not a
connected on-device Mamba block or hardware performance measurement.
