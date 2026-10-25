# Radiance 3D stencil

This is a faithful bare-metal port of Vortex regression `stencil3d` at pinned
revision `b447578d9f3bb116d658c614cf3407153f0536da`. It computes one 27-point
average for every cell, with the original boundary-replication rule and the
source-valid `64 x 64 x 64` scale profile. The pinned Makefile invokes size
32, so size 64 is not described as the repository default. The input follows
the reference host's `srand(50); float(rand()) / RAND_MAX` sequence.

The only multi-SM transformation is ownership: physical SMs receive disjoint,
contiguous blocks aligned to 16 float32 elements (one 64-byte line). Every SM
therefore executes part of the original output domain exactly once; no work is
removed or repeated. The RV64 host compares every output against an independent
CPU oracle using the reference's six-ULP tolerance and checks both guards.
Before launch it NaN-poisons every output, and at runtime it verifies that the
embedded oracle still matches its generated FNV-1a digest. The output digest
is diagnostic because the six-ULP upstream tolerance, not bit identity, is the
acceptance rule.

Example logical artifact:

```sh
python3 run.py --size 64 --sms 4 --loader logical \
  --out /scratch/agustin/radiance-fpga-1a/runs/stencil3d/4sm
```

FPGA candidates additionally use `--robust-end-drain
--robust-spin-until-reset --robust-serial-fence --robust-post-spin 20000` and
must pass the logical model before being scheduled on FPGA 1A.
