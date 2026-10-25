#!/usr/bin/env python3
"""Audit scalar/no-VPU logical and direct ELFs for one HPC family."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


FAMILIES = ("hotspot3d", "pathfinder", "gaussian")
VECTOR_INSTRUCTION = re.compile(
    r"\b(?:vsetvli|vsetivli|vsetvl|vle\d+|vse\d+|vl\d+r|vs\d+r)\b"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_architecture(output: str) -> str:
    match = re.search(r'Tag_RISCV_arch:\s*"([^"]+)"', output)
    if match is None:
        match = re.search(
            r"TagName:\s*arch\s*(?:\r?\n)\s*Value:\s*([^\s]+)", output)
    require(match is not None, "missing RISC-V architecture tag")
    return match.group(1)


def architecture(readelf: Path, elf: Path) -> str:
    output = subprocess.check_output([str(readelf), "-A", str(elf)], text=True)
    try:
        return parse_architecture(output)
    except SystemExit as error:
        raise SystemExit(f"{error}: {elf}") from None


def has_vector_isa(arch: str) -> bool:
    base, *extensions = arch.lower().split("_")
    base_letters = re.sub(r"^rv(?:32|64)", "", base)
    return "v" in base_letters or any(
        re.match(r"^v(?:\d|$)", extension) is not None or
        extension.startswith("zv") for extension in extensions)


def expected_identities(family: str) -> set[tuple[int, str]]:
    require(family in FAMILIES, f"unsupported family: {family}")
    return {(sm, loader) for sm in (1, 2, 4)
            for loader in ("logical", "direct-hashed")}


def audit(repo: Path, family: str, logical_path: Path,
          direct_path: Path) -> dict:
    manifests = [json.loads(logical_path.read_text()),
                 json.loads(direct_path.read_text())]
    require(all(item.get("hardware_rebuilt") is False for item in manifests),
            "manifest does not assert hardware_rebuilt=false")
    require(all(item.get("no_vpu") is True for item in manifests),
            "manifest does not assert no_vpu=true")
    rows = [row for manifest in manifests
            for row in manifest.get("workloads", [])]
    require(len(rows) == 6, f"expected six {family} artifacts, found {len(rows)}")

    readelf = repo / "llvm/llvm-muon/bin/llvm-readelf"
    objdump = repo / "llvm/llvm-muon/bin/llvm-objdump"
    seen = set()
    source_hashes: dict[int, set[str]] = {}
    audited = []
    for row in rows:
        sm = int(row.get("sm_count", 0))
        loader = row.get("loader")
        seen.add((sm, loader))
        label = f"{family}/{sm}SM/{loader}"
        require(row.get("family") == family, f"{label}: wrong family")
        require(row.get("scalar_only") is True and row.get("no_vpu") is True and
                row.get("vpu") is False and row.get("vector_isa") is False and
                row.get("gemmini") is False and row.get("mx") is False,
                f"{label}: scalar/no-VPU metadata changed")
        fused = Path(row["elf"]).resolve()
        device = Path(row["device_elf"]).resolve()
        result = json.loads((fused.parent / "result.json").read_text())
        require(fused.is_file() and device.is_file(),
                f"{label}: missing fused or device ELF")
        require(sha256(fused) == row.get("elf_sha256") ==
                result.get("elf_sha256"), f"{label}: fused ELF hash mismatch")
        device_hash = sha256(device)
        require(device_hash == row.get("device_elf_sha256") ==
                result.get("device_elf_sha256"),
                f"{label}: device ELF hash mismatch")
        arch = architecture(readelf, device)
        require(not has_vector_isa(arch),
                f"{label}: vector ISA enabled: {arch}")
        disassembly = subprocess.check_output(
            [str(objdump), "-d", str(device)], text=True)
        require(VECTOR_INSTRUCTION.search(disassembly.lower()) is None,
                f"{label}: vector instruction found")
        lowered = disassembly.lower()
        require(all(token not in lowered for token in
                    ("gemmini", "mxgemm", "mvin", "mvout", "vpu")),
                f"{label}: accelerator/VPU symbol found")
        source_hashes.setdefault(sm, set()).add(result["kernel_source_sha256"])
        audited.append({
            "family": family, "sm_count": sm, "loader": loader,
            "elf_sha256": row["elf_sha256"],
            "device_elf_sha256": device_hash, "device_isa": arch,
            "scalar_only": True, "vector_isa": False,
            "gemmini": False, "mx": False, "vpu": False,
        })

    require(seen == expected_identities(family),
            f"{family}: logical/direct 1/2/4-SM coverage is incomplete")
    require(all(len(hashes) == 1 for hashes in source_hashes.values()),
            f"{family}: logical/direct artifacts do not share source hashes")
    return {
        "schema": "radiance-scalar-hpc-artifact-audit-v1",
        "status": "pass", "family": family,
        "artifact_count": len(audited),
        "source_hashes": {str(sm): next(iter(hashes))
                          for sm, hashes in sorted(source_hashes.items())},
        "artifacts": sorted(audited,
                            key=lambda row: (row["sm_count"], row["loader"])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--logical-manifest", type=Path, required=True)
    parser.add_argument("--direct-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.repo.resolve(), args.family,
                   args.logical_manifest.resolve(),
                   args.direct_manifest.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
