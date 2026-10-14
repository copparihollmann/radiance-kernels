# Reduced Mamba2 mixer

This scalar Radiance program connects the five operations of one Mamba2
mixer: input projection, causal depthwise convolution, selective scan, gated
group RMSNorm, and output projection. Its 48 convolution channels split into
32 scan values and two eight-value `B/C` groups. It carries convolution and
SSM states from a five-token prefill through two one-token decode calls.

The fixture instantiates the same seeded, random-weight Transformers 5.3
CPU-fallback mixer used by the [separate convolution](../mamba_depthwise_conv/README.md)
and [scan](../mamba_selective_scan/README.md) checks. It checks that the
upstream input projection reproduces their saved convolution and `dt` inputs,
then records the upstream mixer output. The device starts with zeroed
projection buffers, so the input projection must actually produce those
values before the dependent callbacks run. The final output is compared
directly with the upstream mixer, and intermediate outputs and both final
states are compared with the independent NumPy checks.

Run from the repository root after generating the two checked fixture
directories described in the [connected pair](../mamba_conv_scan/README.md):

```sh
python3 kernels/mamba_mixer/test_device.py \
  --reference-python "$TRANSFORMERS_PYTHON" \
  --conv-dir "$TMPDIR/mamba-conv-upstream" \
  --scan-dir "$TMPDIR/mamba-conv-scan-handoff" \
  --out "$TMPDIR/mamba-mixer"
```

The [recorded Cyclotron run](evaluation/transformers-reduced-cyclotron.json)
passed on two Muon cores in 54,802 functional cycles. It checked 336
projected convolution inputs, 28 projected time steps, 224 projected gates,
336 convolution outputs, 192 convolution-state values, 224 scan outputs,
128 SSM-state values, and 224 final mixer outputs. The projected arrays
matched their source fixture exactly. Maximum absolute errors for the
convolution output, scan output, and final mixer output were `1.56e-6`,
`1.49e-7`, and `4.11e-6`. The fixture generator peaked at about 356 MiB
RSS; its build directory used about 1.6 MiB.
As a [negative control](evaluation/zero-gated-input-control.json), setting
the gated RMSNorm input to zero made Cyclotron fail the final output check
with `tohost=449` while the preceding callbacks remained unchanged.

This validates one reduced mixer with random weights, not the pinned
Nemotron checkpoint or its 42-block hybrid schedule. The surrounding block
norm and residual, other Mamba blocks, attention/MLP blocks, and LM head are
not in this ELF. Cyclotron functional cycles are not RTL latency or a
hardware performance result.
