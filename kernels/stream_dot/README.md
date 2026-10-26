# BabelStream-derived Dot for Radiance

This port preserves the float32 Dot equation and two-stage CUDA reduction
structure from `UoB-HPC/BabelStream` revision
`5cb9445483786f3502b12b8faab8414f8c5873b1`. Each physical Radiance SM owns
a disjoint contiguous cache-line-aligned input range. Threads accumulate
strided products, a shared-memory tree creates one partial per SM, and the RV64
host combines those partials in SM order and bit-exactly validates the final
scalar. This mirrors upstream CUDA's device block partials plus host final sum.

The input length is not reduced for performance runs. Every partial and the
ordered host final sum are checked bit-for-bit against an independent
emulation of the device's float32 operation order, and both surrounding guards
are checked. Because this is a port and not an unmodified upstream run, results must be described as
"BabelStream-derived Dot" rather than official BabelStream benchmark results.
