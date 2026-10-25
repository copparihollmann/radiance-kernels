#!/usr/bin/env python3
"""Audit faithful no-VPU Vortex stencil3d artifacts before FPGA use."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


REFERENCE_REVISION = "b447578d9f3bb116d658c614cf3407153f0536da"


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


def audit(repo: Path, logical_path: Path, direct_path: Path) -> dict:
    manifests = [json.loads(logical_path.read_text()),
                 json.loads(direct_path.read_text())]
    rows = [row for manifest in manifests for row in manifest.get("workloads", [])]
    require(all(manifest.get("no_vpu") is True for manifest in manifests),
            "manifest lacks no_vpu=true")
    require(all(manifest.get("hardware_rebuilt") is False for manifest in manifests),
            "manifest does not assert hardware_rebuilt=false")
    require(len(rows) == 6, f"expected six logical/direct stencil rows, found {len(rows)}")
    device_readelf = repo / "llvm/llvm-muon/bin/llvm-readelf"
    device_objdump = repo / "llvm/llvm-muon/bin/llvm-objdump"
    paired: dict[int, list[str]] = {}
    audited = []
    for row in rows:
        sm = int(row.get("sm_count", 0))
        loader = row.get("loader")
        require(sm in (1, 2, 4), f"unexpected SM count: {sm}")
        require(loader in ("logical", "direct-hashed"), f"bad loader: {loader}")
        require(row.get("family") == "stencil3d", "wrong family")
        require(row.get("size") == 64 and row.get("output_element_count") == 64 ** 3,
                f"{sm}SM/{loader} has wrong volume")
        require(row.get("reference_revision") == REFERENCE_REVISION,
                f"{sm}SM/{loader} has wrong Vortex reference")
        require(row.get("benchmark_contract") ==
                "vortex-stencil3d-27-point-boundary-replication",
                f"{sm}SM/{loader} has wrong equation contract")
        require(row.get("partition") == "disjoint-contiguous-64-byte-output-lines",
                f"{sm}SM/{loader} has wrong multi-SM ownership")
        require(row.get("max_ulp") == 6 and
                row.get("correctness") == "complete-output-reference-check",
                f"{sm}SM/{loader} lacks the original complete six-ULP check")
        require(row.get("robust_end_drain") is True and
                row.get("robust_serial_fence") is True,
                f"{sm}SM/{loader} lacks robust completion")
        require(bool(row.get("robust_spin_until_reset")) == (loader == "direct-hashed"),
                f"{sm}SM/{loader} has wrong FPGA spin/reset mode")

        fused = Path(row["elf"]).resolve()
        device = fused.parent / "kernel.radiance.elf"
        result = json.loads((fused.parent / "result.json").read_text())
        require(fused.is_file() and device.is_file(), f"missing artifact for {sm}SM/{loader}")
        require(sha256(fused) == row.get("elf_sha256") == result.get("elf_sha256"),
                f"fused ELF hash mismatch for {sm}SM/{loader}")
        device_hash = sha256(device)
        require(device_hash == row.get("device_elf_sha256") ==
                result.get("device_elf_sha256"),
                f"device ELF hash mismatch for {sm}SM/{loader}")
        device_arch = read_arch(device_readelf, device)
        host_arch = read_arch(Path("readelf"), fused)
        require(not has_vector_isa(device_arch), f"device ISA enables vector: {device_arch}")
        require(not has_vector_isa(host_arch), f"host ISA enables vector: {host_arch}")
        disassembly = subprocess.check_output(
            [str(device_objdump), "-d", str(device)], text=True,
        )
        require(len(re.findall(r"\bfadd\.s\b", disassembly)) == 27,
                f"{sm}SM/{loader} does not preserve 27 stencil additions")
        require(len(re.findall(r"\bfdiv\.s\b", disassembly)) == 1,
                f"{sm}SM/{loader} does not preserve one average division")
        require(not re.search(r"\b(?:vsetvli|vle\d+|vse\d+|vl\d+r|vs\d+r)\b",
                              disassembly),
                f"{sm}SM/{loader} contains vector instructions")
        require("gemmini" not in disassembly.lower() and "vpu" not in disassembly.lower(),
                f"{sm}SM/{loader} contains accelerator/VPU symbols")
        paired.setdefault(sm, []).append(result["kernel_source_sha256"])
        audited.append({
            "sm_count": sm, "loader": loader,
            "fused_elf": str(fused), "fused_elf_sha256": sha256(fused),
            "device_elf": str(device), "device_elf_sha256": device_hash,
            "device_isa": device_arch, "host_isa": host_arch,
            "float_add_instructions": 27, "float_divide_instructions": 1,
            "vpu": False,
        })
    require(set(paired) == {1, 2, 4} and all(len(values) == 2 for values in paired.values()),
            "logical/direct SM coverage is incomplete")
    require(all(len(set(values)) == 1 for values in paired.values()),
            "logical/direct source hashes disagree")
    return {
        "schema": "radiance-stencil3d-artifact-audit-v1",
        "status": "pass", "reference_revision": REFERENCE_REVISION,
        "logical_manifest": str(logical_path.resolve()),
        "logical_manifest_sha256": sha256(logical_path),
        "direct_manifest": str(direct_path.resolve()),
        "direct_manifest_sha256": sha256(direct_path),
        "volume": [64, 64, 64], "max_ulp": 6,
        "partition": "disjoint-contiguous-64-byte-output-lines",
        "artifacts": sorted(audited, key=lambda row: (row["sm_count"], row["loader"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--logical-manifest", type=Path, required=True)
    parser.add_argument("--direct-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evidence = audit(args.repo.resolve(), args.logical_manifest.resolve(),
                     args.direct_manifest.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"PASS: audited {len(evidence['artifacts'])} stencil3d artifacts -> {args.out}")


if __name__ == "__main__":
    main()
