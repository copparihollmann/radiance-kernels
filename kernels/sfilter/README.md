# Radiance spatial filter

This directory implements the pinned Vortex `tests/opencl/sfilter` 3x3
spatial-filter equation on scalar Radiance Muon cores. The source reference is
Vortex revision `b447578d9f3bb116d658c614cf3407153f0536da`; the kernel blob is
`a0be9d109f3d544114ed2e14802d55f566bef88f`. The nine FP32 multiplies and
left-to-right additions are preserved, with FP contraction disabled. The port
does not use a VPU, vector ISA, Gemmini, or MX operations.

The default dataset remains the pre-repair Radiance contract: 48x48 FP32,
Python `random.Random(0)` inputs in `[0,100)`, nine coefficients equal to one,
and an all-zero one-pixel output boundary. Generation fails if those default
input and expected words differ from the tracked legacy `data` and `expected`
arrays.

## Physical-SM ownership

`threadblock_id` is the physical SM ID supplied by `nightly/device.h`. Each SM
owns complete contiguous interior rows. Since supported row widths are
multiples of 16 FP32 elements, every ownership boundary is also a 64-byte
cache-line boundary. For the default 48x48 case, 4 SMs use the half-open row
intervals `[1,13)`, `[13,25)`, `[25,36)`, and `[36,47)`.
The generated `result.json` records each physical ID, its row interval, pixel
count, byte interval, cache-line offsets, and a digest of the ownership table.
Generation enumerates every interior pixel and fails on a gap or duplicate.

## Correctness

The RV64 host initializes both 64-byte guards, leaves the required boundary at
positive zero, and poisons every interior output with a NaN before launch. It
then checks all `size*size` output words against the generated oracle using the
upstream four-ULP rule, checks finite values and zero boundaries, verifies both
guards, verifies the oracle digest, and reports the complete actual-output
digest. The actual digest is diagnostic rather than an exactness gate because
the upstream contract explicitly permits four ULP.

Prepare or build a software artifact with, for example:

```sh
python3 sfilter.py --size 48 --sms 4 --out runs/4sm --prepare-only
python3 sfilter.py --size 16 --sms 4 --out runs/smoke
python3 functional_smoke.py runs/smoke
python3 -m unittest -v test_sfilter.py
```

The builder uses `RISCV64_TOOLCHAIN_PATH` when set and otherwise discovers the
workspace-local `toolchains/riscv-tools-prefix/riscv-tools` installation.

These commands only generate/build software. They do not run FireSim or touch
an FPGA.
