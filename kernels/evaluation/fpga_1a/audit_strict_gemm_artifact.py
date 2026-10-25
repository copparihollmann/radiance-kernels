#!/usr/bin/env python3
"""Audit the exact no-VPU, balanced-B MXFP8 GEMM artifacts queued for FPGA.

This intentionally checks the preserved build products, not just descriptive
manifest fields.  In particular, the device disassembly must show the V3
entry point and parity selection of both duplicated B payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


EXPECTED_GENERATOR_ARGS = (
    "--fmt fp8 --M 512 --N 512 --K 512 --TM 128 --TN 64 --TK 256 "
    "--pretile --bdup"
)
EXPECTED_MACROS = {
    "GEMM_M": 512,
    "GEMM_N": 512,
    "GEMM_K": 512,
    "GEMM_TM": 128,
    "GEMM_TN": 64,
    "GEMM_TK": 256,
    "GEMM_FP4": 0,
    "GEMM_BDUP": 1,
    "GEMM_PRETILE": 1,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_macros(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for name, raw in re.findall(
            r"^#define\s+([A-Z][A-Z0-9_]*)\s+(0x[0-9a-fA-F]+|[0-9]+)u?\b",
            text, re.MULTILINE):
        values[name] = int(raw, 0)
    return values


def function_block(disassembly: str, symbol: str) -> str:
    match = re.search(
        rf"^[0-9a-f]+ <{re.escape(symbol)}>:\n(.*?)(?=^[0-9a-f]+ <[^>]+>:\n|\Z)",
        disassembly,
        re.MULTILINE | re.DOTALL,
    )
    require(match is not None, f"device disassembly lacks {symbol}")
    return match.group(1)


def isa_has_vector(arch: str) -> bool:
    # RISC-V's base vector extension is the standalone `v` token.  Zv* and
    # experimental vector subsets are also rejected for this no-VPU build.
    tokens = arch.lower().split("_")
    base = tokens[0] if tokens else ""
    base_letters = re.sub(r"^rv(?:32|64)i[0-9p]*", "", base)
    return "v" in base_letters or any(t == "v" or t.startswith("zv") for t in tokens)


def read_arch(readelf: Path, elf: Path) -> str:
    output = subprocess.check_output([str(readelf), "-A", str(elf)], text=True)
    match = re.search(r"(?:Value:|Tag_RISCV_arch:)\s*\"?(rv(?:32|64)[^\"\s]+)", output)
    require(match is not None, f"cannot read RISC-V ISA from {elf}")
    return match.group(1)


def audit(repo: Path, manifest_path: Path, device_readelf: Path,
          host_readelf: Path) -> dict:
    repo = repo.resolve()
    manifest_path = manifest_path.resolve()
    gemm = repo / "kernels/nightly/gemm_mx"
    manifest = json.loads(manifest_path.read_text())
    rows = manifest.get("workloads", [])
    require(len(rows) == 3, f"expected three strict GEMM rows, found {len(rows)}")
    require(manifest.get("no_vpu") is True, "manifest does not assert no_vpu=true")

    generator_stamp = gemm / "build/gen_args.stamp"
    defs_stamp = gemm / "build/defs.stamp"
    header = gemm / "gemm_data.h"
    schedule_source = repo / "lib/src/mu_schedule.cpp"
    require(generator_stamp.read_text().strip() == EXPECTED_GENERATOR_ARGS,
            "preserved generator arguments do not describe the strict 512^3 bdup fixture")
    require("MU_ADDR_HASH=1" in defs_stamp.read_text().split(),
            "preserved build did not enable direct GPU address hashing")
    macros = parse_macros(header.read_text())
    for name, value in EXPECTED_MACROS.items():
        require(macros.get(name) == value, f"{name} is {macros.get(name)!r}, expected {value}")
    for name in ("GEMM_B_ADDR", "GEMM_B2_ADDR", "GEMM_BSC_ADDR", "GEMM_BSC2_ADDR"):
        require(name in macros, f"generated header lacks {name}")
    require(macros["GEMM_B_ADDR"] != macros["GEMM_B2_ADDR"], "B copies alias")
    require(macros["GEMM_BSC_ADDR"] != macros["GEMM_BSC2_ADDR"], "B-scale copies alias")
    schedule_text = schedule_source.read_text()
    for contract in (
            "tid_global = threads_per_cluster * cluster_id + tid_in_cluster",
            "threads_per_threadblock = threads_per_cluster",
            "threadblock_id = tid_global / threads_per_threadblock",
            "callback(arg, tid_in_threadblock, threads_per_threadblock, threadblock_id)"):
        require(contract in schedule_text,
                f"mu_schedule no longer proves one threadblock per cluster: {contract}")

    audited = []
    for row in sorted(rows, key=lambda item: item["sm_count"]):
        sm = int(row["sm_count"])
        require(sm in (1, 2, 4), f"unexpected SM count {sm}")
        require(row.get("vpu") is False, f"{sm}SM row is not marked VPU-free")
        require(row.get("format") == "mxfp8-e4m3", f"{sm}SM row has wrong format")
        require(row.get("shape") == {"m": 512, "n": 512, "k": 512},
                f"{sm}SM row has wrong shape")
        require(row.get("b_copy_policy") == "cluster-parity",
                f"{sm}SM row lacks cluster-parity policy")
        require(row.get("correctness") ==
                "nightly-device-full-bit-exact-and-host-spot-check",
                f"{sm}SM row lacks strict full-output correctness")
        require(row.get("loader") == "direct-hashed" and
                row.get("gpu_address_hash") is True,
                f"{sm}SM row is not direct-hashed")

        queued = Path(row["elf"]).resolve()
        built = gemm / f"build/{sm}sm/gemm_mx.soc.elf"
        device = gemm / f"build/{sm}sm/gemm_mx.radiance.elf"
        dump_path = gemm / f"build/{sm}sm/gemm_mx.radiance.dump"
        for path in (queued, built, device, dump_path):
            require(path.is_file(), f"missing preserved artifact: {path}")
        queued_hash = sha256(queued)
        built_hash = sha256(built)
        require(queued_hash == row.get("elf_sha256"), f"{sm}SM manifest ELF hash mismatch")
        require(queued_hash == built_hash, f"{sm}SM queued ELF differs from build output")

        disassembly = dump_path.read_text()
        cluster_id = function_block(disassembly, "mu_cluster_id")
        require("lw.shared" in cluster_id and "csrr" in cluster_id and
                re.search(r"\bandi\s+\w+,\s*\w+,\s*0xffff\b", cluster_id),
                f"{sm}SM binary lacks printBuf cluster index with CSR fallback")
        function_block(disassembly, "_ZL8entry_v3Pvjjj")
        require("<_ZL8entry_v1Pvjjj>:" not in disassembly and
                "<_ZL8entry_v2Pvjjj>:" not in disassembly,
                f"{sm}SM binary retained a non-V3 entry point")
        load_b3 = function_block(disassembly, "_ZL7load_b3jjjj")
        # The fourth RV32 ABI argument is a3, i.e. load_b3's threadblock/cluster ID.
        require(re.search(r"\bandi\s+\w+,\s*a3,\s*0x1\b", load_b3) is not None,
                f"{sm}SM load_b3 does not select copies with tb & 1")
        for name in ("GEMM_B_ADDR", "GEMM_B2_ADDR", "GEMM_BSC_ADDR", "GEMM_BSC2_ADDR"):
            address = f"0x{macros[name]:x}"
            require(address in load_b3,
                    f"{sm}SM load_b3 does not contain {name}={address}")

        device_arch = read_arch(device_readelf, device)
        host_arch = read_arch(host_readelf, queued)
        require(not isa_has_vector(device_arch), f"{sm}SM device ISA has vector support: {device_arch}")
        require(not isa_has_vector(host_arch), f"{sm}SM host ISA has vector support: {host_arch}")
        audited.append({
            "sm_count": sm,
            "queued_elf": str(queued),
            "queued_elf_sha256": queued_hash,
            "preserved_build_elf": str(built),
            "preserved_build_elf_sha256": built_hash,
            "byte_identical_to_preserved_build": True,
            "device_entry": "entry_v3",
            "b_and_bscale_selection": "threadblock_id & 1",
            "device_isa": device_arch,
            "host_isa": host_arch,
            "vpu": False,
        })

    return {
        "schema": "radiance-strict-gemm-artifact-audit-v1",
        "status": "pass",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "source_revision": manifest.get("source_revision"),
        "source_worktree_sha256": manifest.get("source_state", {}).get("worktree_sha256"),
        "generator_args": EXPECTED_GENERATOR_ARGS,
        "generator_stamp_sha256": sha256(generator_stamp),
        "defs": defs_stamp.read_text().strip(),
        "defs_stamp_sha256": sha256(defs_stamp),
        "generated_header_sha256": sha256(header),
        "mu_schedule_source_sha256": sha256(schedule_source),
        "generated_macros": {name: macros[name] for name in sorted(EXPECTED_MACROS)},
        "b_copy_addresses": {
            name: f"0x{macros[name]:08x}"
            for name in ("GEMM_B_ADDR", "GEMM_B2_ADDR", "GEMM_BSC_ADDR", "GEMM_BSC2_ADDR")
        },
        "threadblock_identity": "mu_schedule maps one threadblock to one cluster",
        "artifacts": audited,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device-readelf", type=Path)
    parser.add_argument("--host-readelf", type=Path, default=Path("readelf"))
    args = parser.parse_args()
    device_readelf = args.device_readelf or (
        args.repo / "llvm/llvm-muon/bin/llvm-readelf")
    evidence = audit(args.repo, args.manifest, device_readelf, args.host_readelf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"PASS: audited {len(evidence['artifacts'])} strict GEMM artifacts -> {args.out}")


if __name__ == "__main__":
    main()
