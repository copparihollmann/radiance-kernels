# Nearn on Radiance

This is a scalar, no-VPU derivation of Vortex's Rodinia Nearn OpenCL kernel at
revision `b447578d9f3bb116d658c614cf3407153f0536da`. It preserves the original
Euclidean-distance equation for every latitude/longitude record. The scaling
dataset is deterministic and synthetic, so results are labeled derived rather
than official Rodinia scores.

The exact 662-byte upstream `tests/opencl/nearn/kernel.cl` blob has SHA-256
`43d6322e0be3adddb4c060b05dcb77a2ffd489ccdd07a5c4294e2f5f7cc17b34`;
the digest is carried into every generated build manifest.

Physical SMs own disjoint contiguous groups of 16 float outputs. The RV64 host
checks every distance against an independent CPU `fmaf`/`sqrtf` oracle within
the configured ULP bound, verifies the ordered top-five indices, and checks
both output guards. Robust FPGA variants keep all completion work inside the
reported kernel interval.
