# Inputs for the LLM evaluation proposal

This directory records candidate workload inputs for an initial Radiance
performance evaluation. It contains no measured LLM cycles. The
[model source table](inputs/sources.csv) pins five official
checkpoint revisions and SHA-256 hashes for their small `config.json` files;
the files are retained under `inputs/`. The weights, tokenizer, example
prompts, and quantization format have not yet been pinned or run.

Run `python3 estimate_kv.py` to regenerate [kv-working-set.csv](kv-working-set.csv).
The script verifies each local config against its recorded hash before
calculating BF16 resident KV bytes for a 256-token prompt at batch 1 and 8
and an 8192-token prompt at batch 1. For each attention layer it uses
`batch × min(context, sliding_window) × 2 (K and V) × KV heads × head_dim ×
2 bytes`. Full-attention layers use `context` instead of a window. It counts
only Nemotron's four attention layers; its Mamba state is omitted. The
`new_token_kv_write_bytes` column counts one new K/V pair per attention layer
and batch item. OLMo's config has `use_cache=false`, so its rows describe an
explicit proposed cached-decode mapping that still needs a reference check.

These are logical working-set sizes. They do not include weights,
activations, padding, allocator overhead, memory-transaction amplification,
cache reuse, controller behavior, or measured cycles. A larger L2 than the
checked U250's logical 512 KiB controller may improve locality within tiles
or across operations even though none of the full KV sets fits in that cache.
The full checkpoint fit and weight-streaming policy must be established
before using any row as an executable benchmark input.

## From existing kernels to one decoder layer

The first Qwen3 layer should execute this dependency chain on shared device
buffers, checking each boundary against a reference run from the pinned
checkpoint:

`RMSNorm → Q/K/V projection → RoPE + KV append → causal attention → output
projection + residual → RMSNorm → gate/up projection → SiLU(gate) × up → down
projection + residual`.

| Stage | Existing Radiance starting point | Work required for this workload |
| --- | --- | --- |
| Projections | [MX GEMM](../../gemm_mxgemmini/README.md) and [batched GEMV](../../gemv_batched_fp8_m128/README.md) | Tile Qwen3's 1024-wide projections; choose and record one checkpoint-to-MX quantization rule, including batch-1 decode. |
| Norm and rotary position | [Fused RMSNorm/QKV](../../rmsnorm_qkv_fused/README.md) and [fused RoPE/QKV](../../rope_qkv_fused/README.md) | Adapt the fixed TinyLlama/Llama shapes to Qwen3's 128-wide heads; write KV in the layout consumed by attention. |
| Attention | [MX GQA attention](../../flash_attention_mx_gqa/README.md) | Support Qwen3's 16 query heads, 8 KV heads and 128-wide heads, plus both prefill and one-token decode. The present kernel uses a fixed TinyLlama shape and its output is checked offline, so it is not an end-to-end Qwen3 pass. |
| MLP and residual | GEMM tiles and existing SIMT elementwise examples | Add Qwen3 `SiLU(gate) × up`, residuals, and exact intermediate checks. |
| Layer and sequence schedule | Standalone ELFs and the STREAM/Spatter host launch path | Reuse intermediate buffers, order all stages, avoid unnecessary host copies, append KV over four decode steps, and emit stage/total timing and traffic traces. |

The same chain can then be adapted to OLMo's local/full attention. Qwen3 MoE
needs router, token grouping, expert weight placement, expert GEMMs, and
weighted regrouping. Nemotron adds Mamba state updates and their boundary
with attention. The existing STREAM/Spatter kernels provide controlled memory
tests for the same hardware configurations; they do not substitute for these
model-specific stages. Full checkpoint and tokenizer outputs, numerical
tolerances after quantization, and end-to-end execution remain open.
