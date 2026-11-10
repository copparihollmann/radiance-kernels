# BabelStream-derived Dot for Radiance

This port preserves the float32 Dot equation and two-stage CUDA reduction
structure from `UoB-HPC/BabelStream` revision
`5cb9445483786f3502b12b8faab8414f8c5873b1`. Each physical Radiance SM owns
a disjoint contiguous cache-line-aligned input range. Threads accumulate
strided products, a shared-memory tree creates one partial per SM, and the RV64
host combines those partials in SM order and bit-exactly validates the final
scalar. This mirrors upstream CUDA's device block partials plus host final sum.

The optional `--block-cyclic-chunk-elements N` schedule assigns complete,
cache-line-aligned chunks round-robin across physical SMs. With `N=2048`, each
float32 input chunk is 8 KiB, staggering simultaneous SM streams across the
documented DRAM bank bits while preserving every input product exactly once.
Only ownership and each thread's legal reduction order change; the Dot
equation, full input size, shared-memory tree, exact generated partial oracle,
integer-only host final sum, and guard checks remain intact.

The per-SM reduction partials are padded to one 64-byte cache line each
(`.zero 64 * sms`, indexed `stream_dot_partial[block_id * 16]`) rather than
packed one float32 per SM, so no two SMs write the same destination line. Only
addresses change: every partial value, the digest and the final scalar are
identical to the packed layout.

The input length is not reduced for performance runs. Every partial and the
ordered host final sum are checked bit-for-bit against an independent
emulation of the device's float32 operation order, and both surrounding guards
are checked. Because this is a port and not an unmodified upstream run, results must be described as
"BabelStream-derived Dot" rather than official BabelStream benchmark results.
