#!/usr/bin/env python3
"""Build one float32 STREAM Copy, Scale, Add, or Triad Radiance kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess


ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated"
KINDS = {"copy": 0, "scale": 1, "add": 2, "triad": 3}
OUTPUT = {"copy": "c", "scale": "b", "add": "c", "triad": "a"}
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1


def input_value(name: str, i: int) -> float:
    if name == "a":
        return float(i % 1024 + 1)
    if name == "b":
        return float((3 * i) % 1024 + 2)
    return float((5 * i) % 1024 + 3)


def result_value(kind: str, i: int, scalar: float) -> float:
    a, b, c = (input_value(name, i) for name in "abc")
    return {"copy": a, "scale": scalar * c, "add": a + b,
            "triad": b + scalar * c}[kind]


def float_word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def digest(words) -> int:
    value = FNV_OFFSET
    for word in words:
        value = ((value ^ word) * FNV_PRIME) & MASK64
    return value


def write_input(path: Path, name: str, elements: int) -> None:
    with path.open("wb") as stream:
        for start in range(0, elements, 32768):
            words = [input_value(name, i)
                     for i in range(start, min(start + 32768, elements))]
            stream.write(struct.pack(f"<{len(words)}f", *words))


def prepare(kind: str, elements: int, out: Path,
            full_host_check: bool = False, sms: int = 1,
            loader: str = "logical", warps_per_core: int = 4,
            scalar: float = 3.0, robust_end_drain: bool = False,
            diagnostic_readback: bool = False,
            robust_spin_until_reset: bool = False,
            robust_serial_fence: bool = False,
            robust_post_spin: int = 500,
            block_cyclic_lines: bool = False,
            block_cyclic_chunk_lines: int = 0,
            robust_cluster_l1_manager: bool = False,
            robust_split_l1_drain: bool = False) -> dict:
    if elements < 2 or elements & 1:
        raise ValueError("elements must be an even integer of at least 2")
    if elements * 12 + 0x100000 > 0x70000000:
        raise ValueError("STREAM arrays exceed the GPU DRAM address window")
    if warps_per_core not in (1, 2, 4, 8):
        raise ValueError("warps_per_core must be 1, 2, 4, or 8")
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError("scalar must be finite and greater than zero")
    if robust_spin_until_reset and not robust_end_drain:
        raise ValueError("robust_spin_until_reset requires robust_end_drain")
    if robust_serial_fence and not robust_end_drain:
        raise ValueError("robust_serial_fence requires robust_end_drain")
    if robust_cluster_l1_manager and not robust_end_drain:
        raise ValueError("robust_cluster_l1_manager requires robust_end_drain")
    if robust_split_l1_drain and not robust_end_drain:
        raise ValueError("robust_split_l1_drain requires robust_end_drain")
    if robust_split_l1_drain and robust_cluster_l1_manager:
        raise ValueError("select only one shared-L1 drain candidate")
    if robust_post_spin < 0:
        raise ValueError("robust_post_spin must be nonnegative")
    if block_cyclic_lines and block_cyclic_chunk_lines:
        raise ValueError("select only one STREAM block-cyclic schedule")
    if block_cyclic_chunk_lines < 0:
        raise ValueError("block-cyclic chunk lines must be nonnegative")
    GENERATED.mkdir(exist_ok=True)
    output = OUTPUT[kind]
    for name in "abc":
        if name != output:
            write_input(GENERATED / f"{name}.bin", name, elements)
        else:
            (GENERATED / f"{name}.bin").write_bytes(b"")
    values = [float_word(result_value(kind, i, scalar)) for i in range(elements)]
    expected = digest(values)
    samples = min(64, elements) if elements > 1024 and not full_host_check else 0
    positions = [i * (elements - 1) // (samples - 1) for i in range(samples)] if samples else []
    expected_sample = digest(values[i] for i in positions) if samples else None
    config = [
        "#pragma once",
        f"#define HPC_SM_COUNT {sms}u",
        f"#define NIGHTLY_CLUSTERS {sms}",
        f"#define NIGHTLY_OCC {warps_per_core}",
        f"#define NIGHTLY_ROBUST_GMEM_END {int(robust_end_drain)}",
        f"#define NIGHTLY_ROBUST_SPIN_UNTIL_RESET {int(robust_spin_until_reset)}",
        f"#define NIGHTLY_ROBUST_SERIAL_FENCE {int(robust_serial_fence)}",
        f"#define NIGHTLY_ROBUST_CLUSTER_L1_MANAGER {int(robust_cluster_l1_manager)}",
        f"#define NIGHTLY_ROBUST_SPLIT_L1_DRAIN {int(robust_split_l1_drain)}",
        f"#define NIGHTLY_ROBUST_POST_SPIN {robust_post_spin}u",
        f"#define STREAM_KIND {KINDS[kind]}",
        f"#define STREAM_ELEMENTS {elements}u",
        f"#define STREAM_SCALAR {scalar.hex()}f",
        f"#define STREAM_READBACK_SAMPLES {samples}u",
        f"#define STREAM_EXPECTED_READBACK_DIGEST 0x{(expected_sample or expected):016x}ULL",
        f"#define STREAM_BLOCK_CYCLIC_CHUNK_LINES {block_cyclic_chunk_lines}u",
        f"#define STREAM_BLOCK_CYCLIC_LINES {int(block_cyclic_lines)}",
    ]
    (GENERATED / "config.h").write_text("\n".join(config) + "\n")
    asm = ['.section .data,"aw",@progbits']
    for name in "abc":
        if name == output:
            continue
        asm += [".balign 64", f".globl stream_{name}", f"stream_{name}:",
                f'.incbin "generated/{name}.bin"']
    asm += [
        '.section .bss,"aw",@nobits', '.balign 64',
        '.globl stream_guard_before', 'stream_guard_before:', '.zero 64',
        f'.globl stream_{output}', f'stream_{output}:', f'.zero {elements * 4}',
        '.globl stream_guard_after', 'stream_guard_after:', '.zero 64',
    ]
    (ROOT / "data.S").write_text("\n".join(asm) + "\n")
    source = hashlib.sha256()
    build_rules = ROOT.parent / "addr_hash_stamp.mk"
    source.update(build_rules.name.encode())
    source.update(build_rules.read_bytes())
    for path in (ROOT.parents[1] / "lib/include/mu_partition.h",
                 ROOT.parents[1] / "lib/include/nightly/device.h",
                 ROOT.parents[1] / "lib/include/nightly/host.h"):
        source.update(path.relative_to(ROOT.parents[1]).as_posix().encode())
        source.update(path.read_bytes())
    for name in ("run.py", "Makefile", "kernel.cpp", "host.cpp", "emit_symbols.py"):
        source.update(name.encode())
        source.update((ROOT / name).read_bytes())
    case = {"kind": f"stream-{kind}", "stream_operation": kind,
            "sm_count": sms,
            "loader": loader, "gpu_address_hash": loader == "direct-hashed",
            "pattern_length": elements, "stream_elements": elements,
            "stream_scalar": scalar,
            "benchmark_contract": "stream-v5.10-elemental-equation"
                                  if scalar == 3.0 else "stream-like",
            "count": 1, "output_elements": elements,
            "logical_payload_bytes": elements * (12 if kind in ("add", "triad") else 8),
            "expected_digest": f"{expected:016x}",
            "expected_sample_digest": f"{expected_sample:016x}" if samples else None,
            "readback_samples": samples, "destination_overlap": False,
            "collision_policy": "parallel",
            "address_plan": {"read": "dense", "write": "dense", "operation": kind},
            "suite_sha256": hashlib.sha256(
                f"stream:{kind}:{elements}:scalar{scalar.hex()}:occ{warps_per_core}:"
                f"robust{int(robust_end_drain)}:spin{int(robust_spin_until_reset)}:"
                f"serial{int(robust_serial_fence)}:"
                f"l1manager{int(robust_cluster_l1_manager)}:"
                f"splitl1{int(robust_split_l1_drain)}:"
                f"post{robust_post_spin}:blockcyclic{int(block_cyclic_lines)}:"
                f"chunklines{block_cyclic_chunk_lines}".encode()
            ).hexdigest(),
            "case": KINDS[kind],
            "kernel_revision": subprocess.check_output(
                ["git", "-C", str(ROOT.parents[1]), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            "kernel_source_sha256": source.hexdigest(),
            "correctness": "sample-checked" if samples else "digest-checked",
            "host_full_readback": full_host_check or not samples,
            "status": "prepared", "simulator": None,
            "element_format": "float32", "output_element_count": elements,
            "scalar_only": True, "no_vpu": True, "vpu": False,
            "vector_isa": False, "gemmini": False, "mx": False,
            "muon_warps_per_core": warps_per_core,
            "robust_end_drain": robust_end_drain,
            "robust_spin_until_reset": robust_spin_until_reset,
            "robust_serial_fence": robust_serial_fence,
            "robust_cluster_l1_manager": robust_cluster_l1_manager,
            "robust_split_l1_drain": robust_split_l1_drain,
            "robust_completion_mode": (
                "cluster-l1-manager-candidate" if robust_cluster_l1_manager
                else "split-per-core-l1-candidate" if robust_split_l1_drain
                else "per-core-l1" if robust_end_drain else "disabled"
            ),
            "robust_post_spin": robust_post_spin,
            "stream_block_cyclic_lines": block_cyclic_lines,
            "stream_block_cyclic_chunk_lines": block_cyclic_chunk_lines,
            "diagnostic_readback": diagnostic_readback,
            "checker_word_pairs": elements // 2}
    (GENERATED / "manifest.json").write_text(json.dumps(case, indent=2) + "\n")
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(case, indent=2) + "\n")
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--elements", type=int, default=65536)
    parser.add_argument("--sms", type=int, choices=(1, 2, 4), default=1,
                        help="physical SMs used by this ELF")
    parser.add_argument("--loader", choices=("logical", "direct-hashed"),
                        default="logical",
                        help="logical GPU loading, or direct loading that bypasses the RTL hash")
    parser.add_argument("--warps-per-core", type=int, choices=(1, 2, 4, 8), default=4,
                        help="Muon occupancy used to trade latency hiding for queue pressure")
    parser.add_argument("--scalar", type=float, default=3.0,
                        help="Scale/Triad scalar (STREAM v5.10 uses 3.0)")
    parser.add_argument("--robust-end-drain", action="store_true",
                        help="settle store queues and capacity-drain L0d/L1 before END")
    parser.add_argument("--robust-spin-until-reset", action="store_true",
                        help="FPGA-only: keep workers live after END until the host soft-reset")
    parser.add_argument("--robust-serial-fence", action="store_true",
                        help="fence each worker warp serially before robust cache drains")
    parser.add_argument("--robust-cluster-l1-manager", action="store_true",
                        help="candidate: one core per cluster drains shared L1 after synchronized private L0 drains")
    parser.add_argument("--robust-split-l1-drain", action="store_true",
                        help="candidate: split each shared-L1 capacity walk across cluster cores")
    parser.add_argument("--robust-post-spin", type=int, default=500,
                        help="target cycles after the final robust cache drain (measured)")
    parser.add_argument("--diagnostic-readback", action="store_true",
                        help="print actual digest and guards when the RV64 check fails")
    parser.add_argument("--block-cyclic-lines", action="store_true",
                        help="round-robin complete 64-byte lines across physical SMs")
    parser.add_argument("--block-cyclic-chunk-lines", type=int, default=0,
                        help="round-robin chunks containing this many complete cache lines")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--full-host-check", action="store_true",
                        help="digest every FP32 output word on the RV64 host")
    parser.add_argument("--host-timing", action="store_true",
                        help="print RV64 release-to-all-finished cycle interval")
    args = parser.parse_args()
    if args.robust_spin_until_reset and not args.robust_end_drain:
        parser.error("--robust-spin-until-reset requires --robust-end-drain")
    if args.robust_serial_fence and not args.robust_end_drain:
        parser.error("--robust-serial-fence requires --robust-end-drain")
    if args.robust_cluster_l1_manager and not args.robust_end_drain:
        parser.error("--robust-cluster-l1-manager requires --robust-end-drain")
    if args.robust_split_l1_drain and not args.robust_end_drain:
        parser.error("--robust-split-l1-drain requires --robust-end-drain")
    if args.robust_split_l1_drain and args.robust_cluster_l1_manager:
        parser.error("select only one shared-L1 drain candidate")
    out = args.out.resolve()
    case = prepare(args.kind, args.elements, out, args.full_host_check,
                   args.sms, args.loader, args.warps_per_core, args.scalar,
                   args.robust_end_drain, args.diagnostic_readback,
                   args.robust_spin_until_reset, args.robust_serial_fence,
                   args.robust_post_spin, args.block_cyclic_lines,
                   args.block_cyclic_chunk_lines,
                   args.robust_cluster_l1_manager,
                   args.robust_split_l1_drain)
    case["host_timing_requested"] = args.host_timing
    (out / "result.json").write_text(json.dumps(case, indent=2) + "\n")
    if args.prepare_only:
        return
    build_env = os.environ.copy()
    build_env["MU_ADDR_HASH"] = "1" if args.loader == "direct-hashed" else "0"
    if args.host_timing:
        build_env["EXTRA_HOST_CXXFLAGS"] = (
            build_env.get("EXTRA_HOST_CXXFLAGS", "") +
            " -DRAD_HOST_TIMING=1").strip()
    if args.diagnostic_readback:
        build_env["EXTRA_HOST_CXXFLAGS"] = (
            build_env.get("EXTRA_HOST_CXXFLAGS", "") +
            " -DSTREAM_DIAGNOSTIC_READBACK=1").strip()
    with (out / "build.log").open("w") as log:
        subprocess.run(["make", "kernel.soc.elf"], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, env=build_env)
    shutil.copy2(ROOT / "kernel.soc.elf", out / "kernel.soc.elf")
    device_elf = out / "kernel.radiance.elf"
    shutil.copy2(ROOT / "kernel.radiance.elf", device_elf)
    case["device_elf_sha256"] = hashlib.sha256(device_elf.read_bytes()).hexdigest()
    case["status"] = "built"
    (out / "result.json").write_text(json.dumps(case, indent=2) + "\n")
    print(json.dumps(case, indent=2))


if __name__ == "__main__":
    main()
