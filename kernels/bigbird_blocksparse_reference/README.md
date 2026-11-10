# BigBird block-sparse CPU reference

This directory is an independent, deterministic CPU oracle for a bounded
BigBird attention component. It contains no Radiance headers and does not
download a checkpoint.

## Upstream contract and pin

The source contract is Google Research's official BigBird repository at commit
[`5e7d375950f9b001e13f1bbafb814c2956ed80af`](https://github.com/google-research/bigbird/tree/5e7d375950f9b001e13f1bbafb814c2956ed80af).
In particular, the pinned
[`bigbird_block_sparse_attention`](https://github.com/google-research/bigbird/blob/5e7d375950f9b001e13f1bbafb814c2956ed80af/bigbird/core/attention.py#L404-L635)
uses scaled dot-product softmax over two global end blocks, a three-block local
window, and random blocks. The repository README states the two-global/
three-local structure and recommends full attention below sequence length 1024.
The algorithm is described in Zaheer et al.,
[“Big Bird: Transformers for Longer Sequences”](https://arxiv.org/abs/2007.14062)
(NeurIPS 2020).

The representative model context is BigBird-Base/BERT-like attention:
`hidden_size=768`, 12 heads, and therefore 64 features per head; the pinned
flags default to 12 heads and three random blocks. This bounded test keeps the
official 1024-token lower sparse-attention regime, 64-token blocks, two global
blocks, three local blocks, and three deterministic random blocks. It reduces
the tensor to one batch, one synthetic head, and `head_dim=8` so that a first
1-SM scalar SIMT implementation and simulator run remain tractable. The
pinned generic flags use a 16-token block by default; choosing 64 here reduces
the graph to 16 blocks while retaining the same sparse connectivity rule.

The pinned default `gpt2.model` contains 50,358 pieces.  Counting the pinned
BigBird `BertModel` shapes (word/position/type embeddings, 12 encoder layers,
the postnorm stack's input normalization, and pooler) gives **127,468,800
trainable scalars** for the representative BigBird-Base encoder backbone.
This transparent config-derived count excludes MLM and downstream task heads;
it is not a checkpoint-file inventory.

## Oracle and tests

The graph generator uses a fully specified xorshift32 stream and rejects global
and local candidates. The numerical oracle independently materializes a dense
16x16 block mask, scans all 1024 key positions, accumulates dot products and
stable softmax in `double`, then rounds outputs to FP32. It does not call or
share the sparse device traversal.

Run:

```sh
make test
```

The test checks graph determinism and legality, both dense global rows, every
one of the 42 noncontiguous random edges, all 8,192 numerical outputs against a
second independently indexed sparse traversal, and global/local/random edge
mutation controls. Shape errors fail closed.

This is primitive-level forward attention with no padding mask and dropout
disabled. It is not a full BigBird encoder, training implementation, tokenizer,
projection/GEMM path, multi-head integration, checkpoint loader, or claim of
BigBird/checkpoint equivalence.
