# flash_attention_mx_gqa

MXFP8 **flash attention** with grouped-query attention and a causal mask —
TinyLlama prefill shape (`Sq = 64`, `Sk = 256`, `d = 64`, 8 query heads : 2 KV
heads).

Per query head it runs a full streaming online-softmax pass over its KV head's key
blocks, reusing the same K/V across the query heads that share it (GQA). Each
key block does two MX-Gemmini GEMMs — `S = Q@K^T` then `O = P@V` — with the
softmax and the P-tile requantization to fp8 in between; causal blocks fully above
the diagonal are skipped. Both GEMMs use **non-square** tiles, so the mesh core's
square-tile assert is relaxed. Verified against the embedded golden (`tohost 0`).
