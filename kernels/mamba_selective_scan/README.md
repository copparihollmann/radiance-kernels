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
