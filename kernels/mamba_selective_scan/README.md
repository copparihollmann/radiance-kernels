# Mamba2 selective scan

This scalar Radiance kernel computes the recurrent part of one Mamba2 mixer. For
each token, head, channel and state element, it applies

`s[t] = exp(dt[t] * A) * s[t-1] + dt[t] * B[t] * x[t]`

and emits `D * x[t] + sum(s[t] * C[t])`. The scan carries its final state from
a prefill launch into a decode launch. The inputs are already split from the
projection and depthwise convolution: `x`, `B`, and `C` are post-convolution;
`dt` is positive after bias and softplus (plus any configured limit); `A` is
already `-exp(A_log)`.

This is one primitive, not a full Mamba block or a runnable Nemotron network.
The projection, convolution, gate, norm, graph binding, real weights and
checkpoint comparison still need integration. The kernel uses FP32 state and
does not reproduce the chunk-boundary behavior of the Transformers CPU
fallback. Match prefill/decode partitions when comparing references.

Run the small standalone check in a task temporary directory:

```sh
python3 kernels/mamba_selective_scan/test_device.py \
  --out "$TMPDIR/mamba-selective-scan"
```

`test_device.py` writes the fixture, build log, ELF, simulator log and hashed
result receipt only to that directory. Its independent NumPy reference uses
nonzero initial state and checks the output and final state. A control resets
the state at the prefill/decode boundary and must change decode output. The
Cyclotron result checks device output with FP32 tolerance and independently
compares the dumped output and final state with the NumPy golden. Its cycle
count is a functional simulator diagnostic, not a hardware performance result.

For a second input source, `gen_upstream_data.py` extracts `x`, `B`, `C`, `dt`,
`A`, and `D` from a seven-token, random-weight Transformers Mamba2 mixer. It
runs a five-token prefill followed by two cached decode calls. A hook captures
the upstream scan output; the NumPy recurrence checks that output, the final
SSM state, and the mixer output after its existing gate/norm and projection.
Run the same Radiance check using a Python environment with Torch and
Transformers 5.3:

```sh
python3 kernels/mamba_selective_scan/test_device.py --source upstream \
  --reference-python "$TRANSFORMERS_PYTHON" \
  --out "$TMPDIR/mamba-selective-scan-upstream"
```

The [recorded reduced upstream check](evaluation/transformers-reduced-cyclotron.json)
has `8.94e-8` maximum scan-output error and `2.24e-8` final-state error
between NumPy and the Transformers CPU fallback. Cyclotron then passes on
those exact extracted inputs. This checks the scan contract for one reduced
mixer with random weights. It does not use the pinned Nemotron checkpoint or
validate a connected full model.
