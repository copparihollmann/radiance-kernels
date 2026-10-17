# Physical AXI trace preparation

This directory currently implements trace conversion, not model MX+VPU kernels.
The pinned hardware is blocked: Radiance forces external shared scratchpad
memory, while Gemmini's VPU requires internal fixed-latency SRAM. See Chipyard's
`docs/radiance-mx-vpu-hbm/README.md` for the failing elaboration and check results.
The existing four model graphs have not been converted to MX+VPU variants.

`axi_trace.py` consumes streaming JSONL (optionally gzip) and creates a new output
directory containing per-path gzipped CSV files and a hashed manifest. Example:

```json
{"cycle":7,"sequence":3,"source":"simt.l2_miss","operation":"read","address":0,"size_bytes":32,"address_space":"gpu_local","boundary":"dram"}
```

This is a schema example, not a captured SIMT request. Addresses must explicitly
be `gpu_local`, `system_unhashed`, or `system_physical`. Only the first two receive
the pinned Radiance GPU address hash. `axi_path` is optional; if supplied it must
match physical routing. The operation must be `read` or `write`. Events must be
observed at the DRAM boundary, with aligned whole 8-byte beats; pre-L2 DMA cannot
substitute for L2 miss traffic. Cycles must not decrease and global sequence must
strictly increase. The sequence and cycle must fit unsigned 64-bit values.

```bash
python3 kernels/model_chain/mx_vpu/axi_trace.py events.jsonl.gz new-output \
  --ratio-num 1 --ratio-den 1
TMPDIR="$TASK_DIR/.tmp" PYTHONDONTWRITEBYTECODE=1 \
  python3 -m unittest discover -s kernels/model_chain/mx_vpu -v
```

The ratio is **replay clock frequency / source clock frequency**, and must be
provided explicitly from the actual system. The example 1/1 is only a control
value. Conversion rounds issue cycles upward and splits at address-hash units,
memory regions, 4 KiB boundaries, and 256 beats. It preserves source order and
records fragment indices. Ports are host 0 and GPU slices 1–4, covering the
physical GPU range `[0x100000000, 0x180000000)`.

Memory use is bounded by one input line (16 KiB maximum), one event and five
compressed output streams. Failed conversion leaves no success manifest;
partial files may remain for diagnosis. Reusing an output directory is rejected.
Sources may use only letters, digits, periods, underscores and hyphens, up to
64 characters. Data payload is not captured.

The paired FireSim `AXI4TraceReplay` target streams one physical path into one
HBM channel. Its C++ reader enforces the same routing, order and burst bounds.
That infrastructure has passed compilation and elaboration checks only. No
complete HBM replay or Ramulator legality result is available. Offline replay
does not feed memory timing back into the running program and must not be
reported as whole-program latency. SIMT physical writeback addresses, combined
Muon/MX cache traffic and multi-SM capture remain unimplemented/unverified.
