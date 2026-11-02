# Bounded MiniCPM3 MLA KV-path device milestone

This directory is a first one-SM Radiance bring-up for MiniCPM3's KV latent
compression and reconstruction. It is intentionally a component kernel, not a
full attention layer or model.

The official source/config pin, exact config-derived parameter count, and
host-only oracle are documented in
[`../../minicpm3_mla_reference/README.md`](../../minicpm3_mla_reference/README.md).
The official 4B-class dimensions are hidden 2,560, 40 heads, KV latent rank
256, shared RoPE-key width 32, non-RoPE key width 64 per head, and value width
64 per head. The fixed bounded workload preserves the same topology at:

```text
tokens=7, hidden=64, heads=4, KV latent rank=16,
shared RoPE key=8, non-RoPE key/head=16, value/head=16
```

For each token, the component computes:

```text
[latent | rope_key] = hidden @ transpose(kv_a_weight)
normalized_latent   = RMSNorm(latent, epsilon=1e-5) * norm_weight
[head_0 key | value, ...] = normalized_latent @ transpose(kv_b_weight)
```

All 1,176 public FP32 stage values are retained and checked: 112 compressed
latent values, 56 shared RoPE-key values, 112 normalized-latent values, and
896 reconstructed non-RoPE-key/value values. The kernel creates one status
code per value and the standard nightly verifier requires every status to be
exactly zero; untouched status poison and non-finite results fail. Packed
32-bit status stores avoid relying on concurrent sub-word stores.

## Numerical contract

Inputs, synthetic weights, stage storage, and component arithmetic are IEEE
binary32. Dot products and RMS reduction are accumulated left-to-right with
FP contraction disabled. RMSNorm uses the device's binary32 square root. The
independent Python oracle instead uses `math.fsum` and binary64 square root,
then rounds each public stage to binary32 before the next stage.

Every output must be finite and satisfy:

```text
abs(actual - oracle) <= 3e-5 + 3e-5 * max(abs(actual), abs(oracle))
```

The vectors are deterministic and bounded. This contract covers this fixed
component workload only; it is not a general long-context or checkpoint
accuracy guarantee. Status generation is in the timed region, so cycle counts
include full-output checking setup rather than measuring projection alone.

## Explicit exclusions

This milestone does **not** implement query compression/reconstruction, RoPE,
attention scores, masking, softmax, cache management, output projection,
residuals, MLPs, a decoder layer, tokenization, or full MiniCPM3. It uses no
downloaded weights and makes no claim of checkpoint, logits, generation, or
end-to-end MiniCPM3 equivalence. It also does not claim that the bounded shape
is a performance proxy for the official dimensions.

## Software-only validation

No FPGA is needed for:

```sh
make test
make software-data
make build/1sm/minicpm3_mla.mu.o RISCV64_TOOLCHAIN_PATH=/path/to/riscv-tools
```

`make test` runs the dependency-free oracle tests, generates the deterministic
blobs, and compares all 1,176 outputs from the portable kernel math against the
independent oracle. The object target cross-compiles the exact device source
without launching hardware. A full `make` additionally builds the one-SM fused
SoC ELF when the local RISC-V toolchain is available.

The ordinary nightly kernel waits for its RV64 host's launch/verification
protocol. For Cyclotron's device-only execution, compile the same entry and
math with the small conditional standalone epilogue; it scans every packed
status word and prints `minicpm3_mla Cyclotron PASS` only when all 1,176
comparisons pass:

```sh
make build/1sm/minicpm3_mla.radiance.elf \
  DEFS=-DMCM3_CYCLOTRON=1 MU_ADDR_HASH=0 \
  RISCV64_TOOLCHAIN_PATH=/path/to/riscv-tools
cyclotron cyclotron.toml \
  --binary-path build/1sm/minicpm3_mla.radiance.elf
```

The full-output checker also has a compile-time negative control.  Building
the Cyclotron target with both `MCM3_CYCLOTRON=1` and
`MCM3_INJECT_ERROR=1` corrupts exactly one reconstructed value after the
component finishes; that binary must print `minicpm3_mla Cyclotron FAIL`.
This mutation is never enabled in a production or FPGA ELF.

Rebuild without `MCM3_CYCLOTRON` for the normal RV64-hosted nightly ELF.

## Accepted validation state

On 2026-11-02 the main integration pass independently reran the four oracle
tests and the portable component comparison.  All 1,176 values passed; the
maximum absolute error was `3.57628e-07`, or `0.00650702` of the declared
tolerance budget.  A forced Muon/RV64 rebuild succeeded.  The standalone
Cyclotron build printed `minicpm3_mla Cyclotron PASS` after 8,014 functional
cycles, while the one-value mutation build printed
`minicpm3_mla Cyclotron FAIL`.  The final production ELF was then rebuilt
without either test macro.  No FireSim or FPGA result exists yet, so this
component is simulator-validated rather than FireSim-measured.
