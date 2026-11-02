# Mamba selective-state CPU reference

This directory is an isolated, CPU-only correctness foundation for the
state-spaces Mamba-790M mixer. It models two stateful operations:

1. Width-4 depthwise causal convolution with a persistent ring state.
2. One-token selective state-space update with persistent FP32 state.

It deliberately does not access FPGA hardware, download checkpoints, or model
the input/output projection GEMMs. All test data is deterministic and
synthetic.

## Canonical Mamba-790M dimensions

| Name | Value |
|---|---:|
| `d_model` | 1536 |
| `d_inner` | 3072 |
| `d_state` | 16 |
| `conv_width` | 4 |
| `dt_rank` | 96 |

`Dimensions` is configurable so unit tests can use smaller shapes. The test
suite exercises both a small configuration and the canonical dimensions.

## Operation and state conventions

For channel `i`, the convolution state is stored and exposed as
`conv_state[i][age]`, ordered from the oldest sample to the newest sample.
The current input replaces the oldest ring entry before convolution. The
depthwise kernel's final tap therefore multiplies the current input. A SiLU is
applied to the convolution result, matching the Mamba mixer dataflow.

The sequence-dependent `dt_features`, `B`, and `C` inputs represent the output
of Mamba's `x_proj`. For each channel and state element, the one-token update
is:

```text
dt_i       = softplus(dt_bias_i + dot(dt_weight_i, dt_features))
A_ij       = -exp(A_log_ij)
state_ij   = exp(dt_i * A_ij) * state_ij
             + dt_i * B_j * conv_activated_i
y_i        = (sum_j state_ij * C_j + D_i * conv_activated_i) * SiLU(gate_i)
```

The convolution ring and selective recurrent state are both `float` (FP32).
This is especially important for the recurrent state; it is not narrowed to
BF16.

## Correctness test

Run:

```sh
make test
```

The test executable is written to `/tmp`, keeping the source directory clean.
It checks:

- Exact ring ordering and wraparound with known values.
- Streaming prefill plus one cached decode token against an independently
  indexed full-sequence implementation.
- Every convolution output, activated value, projected timestep, final output,
  convolution-state element, and recurrent-state element at every token.
- State equality at the prefill/decode boundary.
- Both configurable small dimensions and canonical Mamba-790M dimensions.
- Fail-closed tensor-shape validation.

Comparisons use:

```text
abs(actual - expected) <= 2e-5 + 2e-5 * max(abs(actual), abs(expected))
```

The tolerance allows small FP32 evaluation differences while remaining tight
enough to catch state-ordering, discretization, and indexing errors. This is a
primitive-level numerical contract, not a claim of checkpoint-level model
equivalence.
