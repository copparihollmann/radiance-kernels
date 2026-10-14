# Final-token selection

Greedy generation consumes the final prefill hidden token at the LM head.
This scalar Radiance kernel copies the last row of a `[batch, tokens, width]`
tensor to `[batch, 1, width]`. The graph makes this operation explicit so
long prompts need only one row of output logits. It does not change the
attention or KV-cache computation over the full prompt.

Run either check from the repository root, with build files under the task
temporary directory:

```sh
python3 kernels/select_last_token/test_device.py \
  --out "$TMPDIR/select-last-synthetic"
python3 kernels/select_last_token/test_device.py --source qwen3 \
  --build-dir "$TMPDIR/qwen3-multitoken-build/qwen3_0_6b" \
  --out "$TMPDIR/select-last-qwen3"
```

The [synthetic check](evaluation/cyclotron.json) uses two batches, five
tokens, and an odd width of 33. The
[checkpoint-backed check](evaluation/qwen3-checkpoint-cyclotron.json) uses
the saved full-width Qwen3 final-norm prefill tensor: one batch, three
tokens, width 1,024. Cyclotron passed all 1,024 selected values with exact
FP32 equality in 8,596 functional cycles. A
[first-token negative control](evaluation/first-token-control.json) failed
with `tohost=2049`, showing that the check distinguishes the selected
position. The saved Qwen3 hidden state is bound to the pinned checkpoint,
config, input IDs, and upstream comparison by source hashes.

This is a standalone kernel and its own device check. The last-token mode
has not yet been connected to the full Qwen3 Radiance executable. Its
functional cycle count is not RTL latency or a hardware performance result.
