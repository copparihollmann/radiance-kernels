# Bounded BigBird block-sparse attention on one Radiance SM

This directory is a correctness-first 1-SM component bring-up. It computes
forward scaled-dot-product attention for one deterministic synthetic head over
enough blocks to exercise BigBird global, local, and random/noncontiguous
connectivity. It does not implement a full model.

## Official source and representative context

The contract is pinned to Google Research's official BigBird repository at
commit
[`5e7d375950f9b001e13f1bbafb814c2956ed80af`](https://github.com/google-research/bigbird/tree/5e7d375950f9b001e13f1bbafb814c2956ed80af).
The relevant upstream function is
[`bigbird_block_sparse_attention`](https://github.com/google-research/bigbird/blob/5e7d375950f9b001e13f1bbafb814c2956ed80af/bigbird/core/attention.py#L404-L635):
it applies `1/sqrt(head_dim)`, softmax, two global end blocks, a three-block
sliding window, and random blocks. The pinned
[`flags.py`](https://github.com/google-research/bigbird/blob/5e7d375950f9b001e13f1bbafb814c2956ed80af/bigbird/core/flags.py#L61-L121)
gives the BERT-like BigBird-Base context of hidden size 768, 12 heads (64
features per head), and three random blocks. The algorithm and motivation are
from Zaheer et al.,
[“Big Bird: Transformers for Longer Sequences”](https://arxiv.org/abs/2007.14062)
(NeurIPS 2020).

Using the pinned repository's 50,358-piece `gpt2.model` vocabulary and its
official Base configuration, the BERT-style encoder backbone contains exactly
127,468,800 parameters. This count includes embeddings, 12 encoder layers,
the postnorm encoder stack's input normalization, and pooler; it excludes
pretraining and task heads. The compile-time reference function recomputes the
count from those dimensions and the native test asserts the exact value. No
checkpoint weights are used by this bounded component.

This bounded workload keeps a representative 1024-token sequence, 64-token
blocks, 16 total blocks, two global end blocks, a three-block local window, and
three random blocks for each of 14 non-global query blocks. It reduces batch,
heads, and head width to `1 x 1 x 8`; Q/K/V are deterministic synthetic FP32
tensors, all tokens are valid, and dropout is disabled. The 42 random edges are
deterministic, distinct, non-global, outside the local window, and therefore
noncontiguous. The pinned generic flags default to 16-token blocks; this
bounded component deliberately uses 64-token blocks to keep the 1024-token
test at 16 blocks. It preserves the global/local/random sparse-attention rule,
but it is not a claim of identical default shape.

## Independent oracle and complete checks

`../../bigbird_blocksparse_reference` owns the CPU oracle. It builds a dense
16x16 block mask and scans all 1024 key positions with double-precision dot
products and stable softmax. The device core does not share that traversal: it
enumerates only the sparse key blocks, uses FP32 accumulation, and evaluates a
degree-12 exponential polynomial over generated max-shifted logits. The actual
generated span is `[-0.668864, 0]`, within the documented `[-1.2, 0]` bound.

The fixed kernel divides 1024 independent query positions over one warp on
each of the SM's two Muon cores. For every one of 8,192 FP32 outputs it writes a
status bit for

```text
abs(actual - oracle) <= 3e-5 + 3e-5 * max(abs(actual), abs(oracle))
```

with non-finite values rejected. It also constructs and checks the exact
16-bit attended-block mask for every query block. Thus all 8,192 numerical
elements and all 16 structural rows are checked; status buffers begin as
`0xEEEEEEEE`, and the post-timing exact-u32 verifier rejects untouched poison.
`BBSA_INJECT_ERROR=1` is available as a device failure control. Status
construction is included in the timed region, so any reported cycle count is a
bring-up number, not a pure attention benchmark.

## Reproduction and current evidence

Native checks and deterministic data generation need no FPGA:

```sh
make -C ../../bigbird_blocksparse_reference test
make test
make software-data
```

The current native evidence is:

```text
CPU dense oracle: 8192 outputs, 42 noncontiguous random edges,
                  global/local/random mutation controls PASS
Portable device core: 8704 outputs across small + device shapes,
                      32 structural rows PASS
maximum absolute device-core error: 4.84288e-08
maximum fraction of allowed tolerance: 0.00156438
```

The 1-SM object and complete SoC image build with:

```sh
make SMS=1 \
  RISCV_TOOLCHAIN_PATH=/path/to/riscv-tools \
  RISCV64_TOOLCHAIN_PATH=/path/to/riscv-tools
```

Cyclotron uses the same entry and arithmetic with a standalone completion
epilogue that scans all 8,208 numerical/structural status words:

```sh
make build/1sm/bigbird_blocksparse_attention.radiance.elf \
  DEFS=-DBBSA_CYCLOTRON=1 MU_ADDR_HASH=0 \
  RISCV64_TOOLCHAIN_PATH=/path/to/riscv-tools
cyclotron cyclotron.toml \
  --binary-path build/1sm/bigbird_blocksparse_attention.radiance.elf
```

Adding `-DBBSA_INJECT_ERROR=1` is the checker negative control and must print
`bigbird_blocksparse_attention Cyclotron FAIL`. Neither test macro is enabled
in production or FPGA images.

On 2026-11-02 the standalone build passed under Cyclotron 0.1.0 at pinned
revision `5c01f13103d0a433600699283288a03a2a021a4d`:

```text
production checker: bigbird_blocksparse_attention Cyclotron PASS
functional simulation cycles: 21,296,743
injected-error control: bigbird_blocksparse_attention Cyclotron FAIL
negative-control simulation cycles: 21,298,031
```

These are functional-simulator cycles, not FPGA performance measurements. The
ordinary production image was rebuilt afterwards without either test macro;
its Radiance ELF SHA-256 is
`2841e1fd5095ca86e76aac1d95716e1b280b6b03f129f99d5cc89ef6405441b4`,
and its disassembly contains no `<unknown>` instructions. No FPGA was launched,
and no FireSim performance result exists yet.

## Explicit exclusions

This milestone excludes Q/K/V projection GEMMs, output projection, batching,
multiple heads, padding/attention masks, dropout, backward/training, dynamic
shapes, checkpoint weights, tokenizer/data preprocessing, encoder layers,
task heads, and performance tuning. It makes no claim of full BigBird behavior,
published model quality, checkpoint equivalence, or full BigBird/checkpoint
equivalence.
