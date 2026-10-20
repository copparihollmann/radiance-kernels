# Radiance boot diagnostics

Small tests for one Radiance SoC containing **1, 2, or 4 SMs**, with two Muon cores per SM,
8 warps per core and 16 lanes per warp. No weights, Linux image, MX, or VPU instructions.
They isolate startup; passing them does not qualify TinyLlama or accelerator execution.

The package contains ready-to-load RV64 `*-logical.elf` and `*-direct-hashed.elf` boot files,
the RV32 device ELFs, symbols, linker maps, source, build commands, and `workloads.json`.
The boot ELFs already contain the complete GPU payload and its zero-initialized data.
Use the RV64 boot ELF in FireSim. The RV32 `.device.elf` is for disassembly and Cyclotron.

Run these in order on the matching SM image:

| Stage | What a pass establishes | Where a failure points |
|---|---|---|
| `host-htif` | Rocket entry, stack, console, and successful HTIF exit | Host loading, entry point, HTIF, or host ISA |
| `host-mailboxes` | Host write/read access to every physical SM's print buffer | MMIO address map or SM topology |
| `gpu-entry` | Every core reaches GPU `_start` and writes its local mailbox | Reset release, instruction loading/hash, or GPU boot |
| `gpu-scheduler` | All lanes execute, stacks and scheduler identities are distinct, and the host checks every memory result | Runtime, stack mapping, scheduling, or memory visibility |

`gpu-entry` deliberately uses no GPU stack, C++ runtime, or global-memory stores.
`gpu-scheduler` checks 256 lane results per SM, including the actual stack-region index
of every warp. Four-SM startup receives physical SM tags before reset release to avoid
the pinned RTL's one-bit cluster ID aliasing SMs 2/3 onto SMs 0/1.

## Run a test

Extract the archive and enter its directory, then:

```sh
python3 debug.py verify
python3 debug.py list --sms 2
python3 debug.py install --sms 2 --stage host-htif --loader logical \
  --deploy-dir "$FIRESIM/deploy"
```

Set `workload.workload_name` in your existing FireSim runtime YAML to
`host-htif-2sm-logical.json`. Keep the matching hardware configuration and **one simulated
SoC**. From your configured FireSim deploy environment, run `firesim infrasetup` and
`firesim runworkload`, using your site's usual runtime/HWDB arguments.

Repeat with `host-mailboxes`, then `gpu-entry`, then `gpu-scheduler`. Each prints
`[boot-debug]` breadcrumbs and exits through HTIF with success or an error code.
The GPU tests print one line per core as its phase changes. GPU timeout is 200 million
host cycles after reset release; an MMIO transaction that never returns cannot reach
that timeout, so the last `*_BEGIN` message also matters.

The host-only stages have only `logical` files because they contain no GPU load segments.
For the GPU stages choose the loader that matches your actual memory loader:

- `logical`: loads travel through the SoC's address mapping, or the image has no GPU hash.
- `direct-hashed`: direct physical LoadMem bypasses the GPU hash. These files preapply
  the recorded 2 GiB / 32-byte / four-slice hash. Do not apply it again in the loader.

```sh
python3 debug.py install --sms 2 --stage gpu-entry --loader direct-hashed \
  --deploy-dir "$FIRESIM/deploy"
# workload.workload_name: gpu-entry-2sm-direct-hashed.json
```

The exact Chipyard `graphics` source pin and 1/2/4-SM target names are in `provenance.json`.
These tests do not require VPU hardware. A full model still needs a compatible VPU image.
No FPGA pass is claimed by the supplied software validation report.

After a run, save the entire console and the actual process exit code:

```sh
python3 debug.py check --sms 2 --stage gpu-entry --loader direct-hashed \
  --log /path/to/uartlog --firesim-exit-code 0
```

Send the selected workload name, `uartlog`, FireSim exit code, and `debug.py verify` output
with a failure report. If `HOST_BOOT` is absent, start with the host ELF/HTIF path.
If mailboxes pass but no GPU phase appears, examine reset and GPU instruction loading.
If entry passes and scheduler fails, use its per-core phase and memory mismatches.
`HOST_TRAP` includes `mcause`, `mepc`, and `mtval`; symbolize its PC against the selected
boot ELF. All boot files include DWARF debug information.

## Host layout

Host entry is `0x80000000`; the linker keeps host sections below `0x80070000`, reserves
`0x80070000..0x80080000` for its downward-growing stack, and puts `tohost` and `fromhost`
on separate 64-byte boundaries in an explicit `.htif` load section. Startup initializes
`gp`, clears BSS, and installs a machine trap handler before calling C. Host instructions
target RV64IMAC with CSR support and use no floating-point ABI or libc startup.
HTIF uses syscall-proxy writes and the standard `(status << 1) | 1` exit encoding.

GPU entry is `0x10000000`, loaded through its host alias at `0x110000000`. Per-SM print
buffers are at host `0x40080000 + sm * 0x100000`, and local GPU `0x80000`. The host writes
both halves of each mailbox slot, fences, and releases GPU reset at `0x41000000`.
GPU boot warps park after publishing completion so a simulation-only GPU-idle stop cannot
preempt host checking or HTIF exit. Host/GPU cycle counts are separate and are diagnostic
measurements; they are not full-model latency estimates.
