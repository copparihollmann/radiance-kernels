# Bounded Mamba selective-state device milestone

This directory is a first Radiance device bring-up for the selective-state
recurrence used by Mamba. It is intentionally not a full Mamba kernel.
The model context is Mamba-790M (roughly 790 million parameters, hidden width
1536); the bounded component below uses a reduced 64-channel representative
shape so every value can be checked during bring-up.

## Included scope

The fixed representative workload is FP32 with `T=17`, `channels=64`, and
`state=16`. For every sequence position and channel it computes:

```text
A_ij       = -exp(A_log_ij)
state_ij   = exp(delta_i * A_ij) * state_ij
             + delta_i * B_j * x_activated_i
y_i        = (sum_j state_ij * C_j + D_i * x_activated_i) * SiLU(gate_i)
```

Channels are device-parallel (one 16-lane warp on each of the cluster's two
cores, with two channels per thread); time is scanned in order within each
channel.
The persistent scan state is FP32. This first correctness milestone places
each channel's state in an explicitly disjoint global-memory slice because
Muon lane-local arrays alias under Cyclotron for this kernel; on-chip state
residency remains a later performance milestone. The checked results are all
`T*channels` outputs and the complete final `channels*state` state.

The following are out of scope: causal convolution, timestep projection and
softplus, input/output projection GEMMs, checkpoint weights, batching, dynamic
shapes, persistent decode state across launches, performance tuning, and full
Mamba/model equivalence. `delta` and `x_activated` are boundary tensors taken
from the existing `kernels/mamba_state_reference` CPU oracle.

## Correctness contract

`gen_data.cpp` includes the existing CPU oracle and uses its independently
indexed full-sequence path to generate the device inputs and goldens. The
generated inputs bound all exponential arguments to approximately
`[-1.5, 0.8]`; the device uses a degree-10 FP32 polynomial on that interval.
Comparison is explicitly tolerance checked rather than claimed bit-exact:

```text
abs(actual - expected) <= 2e-5 + 2e-5 * max(abs(actual), abs(expected))
```

Non-finite values fail. The kernel writes one BF16 status code per checked
float (zero on success and `1.0` on failure), and the established nightly BF16
verifier checks every code with zero error allowed, including poison
detection. This deliberately avoids depending on any uncommitted shared-header
work. Status construction is in the timed region, so reported cycles are
correctness-bring-up numbers, not pure scan performance.

## Software-only checks

No FPGA is needed for:

```sh
make test
make software-data
make build/1sm/mamba_selective_state.mu.o \
  RISCV64_TOOLCHAIN_PATH=/path/to/riscv-tools
```

`make test` compares the portable device math against the CPU oracle at both a
small shape and the exact representative device shape, checking every output
and recurrent-state element after every step. The object build cross-compiles
the device source but does not launch, program, or otherwise access an FPGA.
The vectors are deterministic and synthetic; this first milestone does not
claim coverage of extreme-value or long-sequence numerical behavior.

The usual nightly `make` can build a fused image for a later simulator or FPGA
run, but no hardware execution is part of this milestone.
