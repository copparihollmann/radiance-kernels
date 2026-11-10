# MiniCPM3 MLA KV-path reference

This directory is a bounded, host-only oracle for the KV compression and
reconstruction path in MiniCPM3 Multi-head Latent Attention (MLA). It is not a
MiniCPM3 implementation and does not load checkpoint weights.

## Pinned official source

The architectural source of truth is OpenBMB's official
`openbmb/MiniCPM3-4B` repository at immutable revision
`0a2cd817886710fff15d181edeae4c249443ed67`:

- [config.json](https://huggingface.co/openbmb/MiniCPM3-4B/blob/0a2cd817886710fff15d181edeae4c249443ed67/config.json)
  (`sha256 cf1d08cb7c1815c676e685bd6ce94eb8b85a57d53871e6e159ee8c650717d98a`)
- [configuration_minicpm.py](https://huggingface.co/openbmb/MiniCPM3-4B/blob/0a2cd817886710fff15d181edeae4c249443ed67/configuration_minicpm.py)
  (`sha256 60c9d2cba0d4aebcd64e573cb072e80caf6b067f9bdfcb06079990c0d61a3bba`)
- [modeling_minicpm.py](https://huggingface.co/openbmb/MiniCPM3-4B/blob/0a2cd817886710fff15d181edeae4c249443ed67/modeling_minicpm.py)
  (`sha256 b68ac53fd9498b880fb7d9a8c7e409f29153a69b04ed2730d29a4791d1781bb0`)

The pinned config declares a 4B-class dense model with 62 decoder layers,
hidden size 2,560, intermediate size 6,400, 40 attention heads, query LoRA
rank 768, KV LoRA rank 256, 64 non-RoPE query/key dimensions per head, 32
RoPE dimensions per head, and 64 value dimensions per head. Its maximum
position count is 32,768 and its vocabulary has 73,448 entries.

Counting unique learned scalars from those dimensions gives **4,073,875,968
parameters**: 188,026,880 tied embedding/LM-head parameters, 62 × 62,674,944
decoder-layer parameters, and 2,560 final RMSNorm parameters. This count
assumes the pinned configuration class's default `tie_word_embeddings=True`,
no projection biases, and excludes non-parameter RoPE buffers. “4B” is the
model's rounded public name; the exact number here is a transparent
config-derived count, not a checkpoint-file inventory.

The relevant official forward path is:

```text
[hidden=2560]
  -- kv_a_proj_with_mqa --> [latent=256 | shared RoPE key=32]
  -- RMSNorm(latent) --> [latent=256]
  -- kv_b_proj --> [40 heads][non-RoPE key=64 | value=64]
```

This oracle follows exactly that boundary. It does not apply RoPE, form
queries, calculate attention scores/softmax, apply the output projection, or
run any other decoder operation.

## Independence and determinism

`minicpm3_mla_reference.py` uses only the Python standard library. Synthetic
binary32 inputs come from a specified 32-bit LCG. Oracle reductions use
`math.fsum` and binary64 square root, with each public stage rounded to
binary32 before it feeds the next stage. The Radiance component instead uses
sequential binary32 reductions, so its implementation and reduction behavior
are not copied into this oracle.

Run the native reference tests with:

```sh
python3 -m unittest -v test_minicpm3_mla_reference.py
```

The fixed Radiance workload is documented separately in
`kernels/nightly/minicpm3_mla`.
