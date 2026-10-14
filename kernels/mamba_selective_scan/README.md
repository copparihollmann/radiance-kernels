# Mamba2 selective scan

This scalar Radiance kernel computes the recurrent part of one Mamba2 mixer. For
each token, head, channel and state element, it applies

`s[t] = exp(dt[t] * A) * s[t-1] + dt[t] * B[t] * x[t]`

and emits `D * x[t] + sum(s[t] * C[t])`. The scan carries its final state from
a prefill launch into a decode launch. The inputs are already split from the
projection and depthwise convolution: `x`, `B`, and `C` are post-convolution;
`dt` is the raw projected time step. The kernel adds `dt_bias`, applies
softplus and the configured limits, and then performs the recurrence. `A` is
already `-exp(A_log)`; that transform can be done once when binding weights.

This is one primitive, not a full Mamba block or a runnable Nemotron network.
The projection, gate, norm, graph binding, real weights and
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

For a second input source, `gen_upstream_data.py` extracts `x`, `B`, `C`, raw
`dt`, `dt_bias`, `A`, and `D` from a seven-token, random-weight Transformers
Mamba2 mixer with the pinned four-tap convolution width. It runs a five-token
prefill followed by two cached decode calls. A hook captures
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
has `5.96e-8` maximum scan-output error and `1.49e-8` final-state error
between NumPy and the Transformers CPU fallback. On those extracted inputs,
Cyclotron differs from the NumPy golden by at most `8.94e-8` in scan output
and `4.47e-8` in final state. This checks the scan contract for one reduced
mixer with random weights. It does not use the pinned Nemotron checkpoint or
validate a connected full model.

To feed the verified depthwise-convolution device output into a second scan
ELF, run:

```sh
python3 kernels/mamba_selective_scan/test_device.py --source upstream \
  --reference-python "$TRANSFORMERS_PYTHON" \
  --conv-output "$TMPDIR/mamba-conv-upstream/output.bin" \
  --conv-result "$TMPDIR/mamba-conv-upstream/result.json" \
  --out "$TMPDIR/mamba-conv-scan-handoff"
```

The [handoff receipt](evaluation/conv-scan-handoff-cyclotron.json)
binds both runs to the same reduced mixer input and source hash, checks the
48-channel split into `x/B/C`, and records the second Cyclotron pass. The
host file connects two device executions; this is a numerical composition
check, not an on-device shared-buffer execution.
