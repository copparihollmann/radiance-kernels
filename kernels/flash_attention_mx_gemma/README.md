# flash_attention_mx_gemma

The Gemma-2 variant of the MXFP8 **flash attention** kernel (`Sq = 64`,
`Sk = 256`, `d = 256`, 8 query heads : 4 KV heads).

Same streaming online-softmax structure as `flash_attention_mx_gqa` (per-key-block
`S = Q@K^T` → softmax → requant → `O = P@V`), plus the two Gemma-2 attention
specifics folded into the online softmax:

- **attention soft-cap** (`cap = 50`): each scaled score is passed through
  `cap * tanh(s / cap)` before the running max/exp.
- **sliding-window** masking (`window = 128`): keys older than
  `query_pos - window` are masked (Gemma's even layers).

Verified against the embedded golden (`tohost 0`).
