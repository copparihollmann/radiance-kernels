# Bounded Qwen3.5 Gated DeltaNet state kernel

This is a one-SM Radiance correctness milestone for the recurrent core of one
official-size Qwen3.5 DeltaNet value head. It is intentionally not a complete
DeltaNet layer or Qwen model. Architecture citations and full representative
model dimensions are in
[`../../qwen35_deltanet_reference/README.md`](../../qwen35_deltanet_reference/README.md).

## Exact boundary

The fixed synthetic workload is FP32 with `T=9`, one expanded value head,
`K=128`, and `V=128`. Its inputs match the pinned Transformers recurrent
function boundary: raw post-convolution `q`, `k`, `v`, `beta=sigmoid(b)`, and
the log-decay gate `g=-exp(A_log)*softplus(a+dt_bias)`. For every token:

```text
q = q / sqrt(sum(q*q) + 1e-6) / sqrt(K)
k = k / sqrt(sum(k*k) + 1e-6)
S = exp(g) * S
delta = (v - k^T S) * beta
S = S + k outer delta
o = q^T S
```

One cluster's threads own disjoint value columns. Time is scanned in order and
the persistent FP32 state is stored in device-friendly `[V][K]` global-memory
layout. This is a correctness choice, not a claim that the final design should
leave state off chip.

The host checks every one of the `9*128 = 1,152` outputs and every one of the
`128*128 = 16,384` final-state values against goldens from the independent CPU
oracle. Outputs and state begin with `0xEEEEEEEE` poison. A pass therefore
means all 17,536 checked FP32 values were written, finite, and in tolerance.

## Numerical contract

- Inputs, arithmetic, outputs, and persistent state are FP32.
- Floating-point contraction is disabled for the kernel and native test.
- Q/K sums are accumulated in increasing key-index order.
- The device computes `exp(g)` with a degree-10 FP32 Taylor polynomial.
- Generated `g` is asserted to remain in `[-0.8, 0]`.
- Goldens use the CPU oracle's `std::exp` implementation.
- Each value must satisfy
  `abs(actual-expected) <= 4e-5 + 4e-5*max(abs(actual),abs(expected))`.
- NaN and infinity fail closed.

The native test checks every state element after every token, not only the
final device-visible state. It also sweeps the exponential approximation over
the full supported interval and exercises tolerance failure controls.

```sh
make test
make software-data
make build/1sm/qwen35_deltanet_state.mu.o \
  RISCV64_TOOLCHAIN_PATH=/path/to/riscv-tools
```

The standard `make` builds a fused one-SM SoC ELF for a later host-capable RTL
run. Cyclotron executes a device-only ELF and cannot post the ordinary nightly
launch handshake, so its build uses the same entry and math with a conditional
standalone epilogue. After a barrier, lane zero scans all 17,536 values against
the embedded goldens and prints an explicit pass/fail:

```sh
make build/1sm/qwen35_deltanet_state.radiance.elf \
  DEFS=-DQ35_CYCLOTRON=1 MU_ADDR_HASH=0 \
  RISCV64_TOOLCHAIN_PATH=/path/to/riscv-tools
cyclotron cyclotron.toml \
  --binary-path build/1sm/qwen35_deltanet_state.radiance.elf
```

For a checker negative control, add `-DQ35_INJECT_ERROR=1` alongside
`Q35_CYCLOTRON`.  That build corrupts exactly one checked output after the
recurrence and must print `qwen35_deltanet_state Cyclotron FAIL`.  The macro
is never enabled in production or FPGA images.

Rebuild without `Q35_CYCLOTRON` for the normal RV64-hosted image. This
bring-up must not launch an FPGA.

## Current evidence (2026-11-02)

```text
CPU oracle:       PASS, 2,115,378 values checked
Portable core:    PASS, 149,871 values checked
Core max abs err: 5.96046e-08
Core max fraction of allowed tolerance: 0.00102782
Generated gates:  g in [-0.223648, -0.142339]
Muon object:      PASS, one-SM cross-compile
Hosted SoC ELF:   PASS, complete build
Disassembly:      0 <unknown> instructions
Cyclotron 0.1.0: PASS, all 17,536 output/final-state comparisons,
                  998,910 functional ISA cycles
```

The Cyclotron executable was built from checkout commit
`f9ed28e25d63257811154cad6d46292d81273b75`; that checkout had a pre-existing
modification to `src/sim/top.rs`, so the cycle count is evidence from that exact
local binary rather than a clean-upstream performance result. The run was
functional (not the timing model), and no FPGA was launched.

## Explicit exclusions

Excluded and unsupported here: full Qwen; multiple heads and Q/K-head repeat
inside the device kernel; QKV, a/b/z, and output projections; causal
convolution and SiLU; formation of beta/g from logits on device; gated
RMSNorm; full-attention layers; MLPs; embeddings; vision; masks and batching;
prefill chunk kernels; quantization; checkpoint loading or equivalence;
training; performance characterization; and persistent decode state across
kernel launches. Passing this primitive does not imply any omitted stage or
checkpoint-level output is correct.
