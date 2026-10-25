#!/usr/bin/env python3
"""Audit exact BF16 GEMV artifacts before they can enter the FPGA queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_arch(readelf: Path, elf: Path) -> str:
    output = subprocess.check_output([str(readelf), "-A", str(elf)], text=True)
    match = re.search(r'(?:Value:|Tag_RISCV_arch:)\s*"?(rv(?:32|64)[^"\s]+)', output)
    require(match is not None, f"cannot read RISC-V ISA from {elf}")
    return match.group(1)


def has_vector_isa(arch: str) -> bool:
    tokens = arch.lower().split("_")
    base = tokens[0] if tokens else ""
    base_letters = re.sub(r"^rv(?:32|64)i[0-9p]*", "", base)
    return "v" in base_letters or any(
        token == "v" or token.startswith("zv") for token in tokens
    )


def audit(repo: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    rows = manifest.get("workloads", [])
    require(manifest.get("no_vpu") is True, "manifest lacks no_vpu=true")
    require(len(rows) == 6, f"expected six logical/direct GEMV rows, found {len(rows)}")
    device_readelf = repo / "llvm/llvm-muon/bin/llvm-readelf"
    device_objdump = repo / "llvm/llvm-muon/bin/llvm-objdump"
    audited = []
    paired: dict[int, list[str]] = {}
    for row in rows:
        sm = int(row.get("sm_count", 0))
        loader = row.get("loader")
        require(sm in (1, 2, 4), f"unexpected SM count: {sm}")
        require(loader in ("logical", "direct-hashed"), f"bad loader: {loader}")
        require(row.get("family") == "gemv_bf16", "wrong family")
        require(row.get("no_vpu") is True, f"{sm}SM/{loader} lacks no_vpu=true")
        require(row.get("rows") == 256 and row.get("cols") == 4096,
                f"{sm}SM/{loader} has wrong GEMV dimensions")
        require(row.get("element_format") == "bf16", f"{sm}SM/{loader} is not BF16")
        require(row.get("correctness") == "full-output-exact-device-order-digest",
                f"{sm}SM/{loader} lacks exact full-output checking")
        require(row.get("partition") == "contiguous-row-ranges-per-sm",
                f"{sm}SM/{loader} has wrong partition")
        require(row.get("robust_end_drain") is True,
                f"{sm}SM/{loader} lacks robust end drain")
        require(bool(row.get("robust_spin_until_reset")) == (loader == "direct-hashed"),
                f"{sm}SM/{loader} has wrong spin/reset mode")

        fused = Path(row["elf"]).resolve()
        device = fused.parent / "kernel.radiance.elf"
        result = json.loads((fused.parent / "result.json").read_text())
        require(fused.is_file() and device.is_file(), f"missing artifact for {sm}SM/{loader}")
        require(sha256(fused) == row.get("elf_sha256"),
                f"fused ELF hash mismatch for {sm}SM/{loader}")
        device_hash = sha256(device)
        require(device_hash == row.get("device_elf_sha256") == result.get("device_elf_sha256"),
                f"device ELF hash mismatch for {sm}SM/{loader}")
        device_arch = read_arch(device_readelf, device)
        host_arch = read_arch(Path("readelf"), fused)
        require(not has_vector_isa(device_arch), f"device ISA enables vector: {device_arch}")
        require(not has_vector_isa(host_arch), f"host ISA enables vector: {host_arch}")
        disassembly = subprocess.check_output(
            [str(device_objdump), "-d", str(device)], text=True
        )
        require("fmadd.h" in disassembly and "fadd.h" in disassembly,
                f"{sm}SM/{loader} lacks scalar BF16 GEMV arithmetic")
        require(not re.search(r"\b(?:vsetvli|vle\d+|vse\d+|vl\d+r|vs\d+r)\b", disassembly),
                f"{sm}SM/{loader} contains vector instructions")
        require("gemmini" not in disassembly.lower() and "vpu" not in disassembly.lower(),
                f"{sm}SM/{loader} contains accelerator/VPU symbols")
        paired.setdefault(sm, []).append(result["kernel_source_sha256"])
        audited.append({
            "sm_count": sm, "loader": loader,
            "fused_elf": str(fused), "fused_elf_sha256": sha256(fused),
            "device_elf": str(device), "device_elf_sha256": device_hash,
            "device_isa": device_arch, "host_isa": host_arch,
            "scalar_bf16_fma": True, "vpu": False,
        })
    require(set(paired) == {1, 2, 4} and all(len(values) == 2 for values in paired.values()),
            "logical/direct SM coverage is incomplete")
    require(all(len(set(values)) == 1 for values in paired.values()),
            "logical/direct source hashes disagree")
    return {
        "schema": "radiance-gemv-bf16-artifact-audit-v1",
        "status": "pass", "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "dimensions": {"rows": 256, "cols": 4096},
        "reference_contract": "exact-device-order-and-independent-fp32-dot",
        "partition": "contiguous-row-ranges-per-sm",
        "artifacts": audited,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evidence = audit(args.repo.resolve(), args.manifest.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"PASS: audited {len(evidence['artifacts'])} GEMV artifacts -> {args.out}")


if __name__ == "__main__":
    main()
