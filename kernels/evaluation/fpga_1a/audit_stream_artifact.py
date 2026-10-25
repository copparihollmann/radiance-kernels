#!/usr/bin/env python3
"""Audit scalar/no-VPU STREAM device ELFs and logical/direct artifact parity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_architecture(output: str) -> str:
    """Extract the ISA string from GNU or LLVM readelf attribute output."""
    match = re.search(r'Tag_RISCV_arch:\s*"([^"]+)"', output)
    if match is None:
        match = re.search(
            r"TagName:\s*arch\s*(?:\r?\n)\s*Value:\s*([^\s]+)", output
        )
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
        extension.startswith("zv")
        for extension in extensions
    )


def audit(repo: Path, logical_path: Path, direct_path: Path) -> dict:
    manifests = [json.loads(logical_path.read_text()),
                 json.loads(direct_path.read_text())]
    rows = [row for manifest in manifests for row in manifest.get("workloads", [])]
    require(len(rows) == 12, f"expected twelve STREAM artifacts, found {len(rows)}")
    require(all(manifest.get("hardware_rebuilt") is False for manifest in manifests),
            "manifest does not assert hardware_rebuilt=false")
    require(all(manifest.get("no_vpu") is True for manifest in manifests),
            "manifest does not assert no_vpu=true")

    readelf = repo / "llvm/llvm-muon/bin/llvm-readelf"
    objdump = repo / "llvm/llvm-muon/bin/llvm-objdump"
    expected = {(operation, sm, loader)
                for operation in ("stream-copy", "stream-scale")
                for sm in (1, 2, 4)
                for loader in ("logical", "direct-hashed")}
    seen = set()
    source_hashes: dict[tuple[str, int], set[str]] = {}
    audited = []
    for row in rows:
        operation = row.get("operation")
        sm = int(row.get("sm_count", 0))
        loader = row.get("loader")
        seen.add((operation, sm, loader))
        require(row.get("family") == "stream", "wrong artifact family")
        require(row.get("scalar_only") is True and row.get("no_vpu") is True and
                row.get("vpu") is False and row.get("vector_isa") is False and
                row.get("gemmini") is False and row.get("mx") is False,
                f"{operation}/{sm}SM/{loader}: scalar/no-VPU metadata changed")
        fused = Path(row["elf"]).resolve()
        device = fused.parent / "kernel.radiance.elf"
        result = json.loads((fused.parent / "result.json").read_text())
        require(fused.is_file() and device.is_file(),
                f"{operation}/{sm}SM/{loader}: missing fused or device ELF")
        require(sha256(fused) == row.get("elf_sha256"),
                f"{operation}/{sm}SM/{loader}: fused ELF hash mismatch")
        device_hash = sha256(device)
        require(device_hash == row.get("device_elf_sha256") ==
                result.get("device_elf_sha256"),
                f"{operation}/{sm}SM/{loader}: device ELF hash mismatch")
        arch = architecture(readelf, device)
        require(not has_vector_isa(arch),
                f"{operation}/{sm}SM/{loader}: vector ISA enabled: {arch}")
        disassembly = subprocess.check_output([str(objdump), "-d", str(device)], text=True)
        require(not re.search(
            r"\b(?:vsetvli|vsetivli|vsetvl|vle\d+|vse\d+|vl\d+r|vs\d+r)\b",
            disassembly), f"{operation}/{sm}SM/{loader}: vector instruction found")
        lowered = disassembly.lower()
        require("gemmini" not in lowered and "vpu" not in lowered and
                "mvin" not in lowered and "mvout" not in lowered,
                f"{operation}/{sm}SM/{loader}: accelerator/VPU symbol found")
        source_hashes.setdefault((operation, sm), set()).add(
            result["kernel_source_sha256"])
        audited.append({
            "operation": operation,
            "sm_count": sm,
            "loader": loader,
            "device_elf_sha256": device_hash,
            "device_isa": arch,
            "vector_isa": False,
            "gemmini": False,
            "mx": False,
            "vpu": False,
        })
    require(seen == expected, "logical/direct Copy/Scale 1/2/4-SM coverage is incomplete")
    require(all(len(hashes) == 1 for hashes in source_hashes.values()),
            "logical/direct artifacts do not share source hashes")
    return {
        "schema": "radiance-stream-scalar-artifact-audit-v1",
        "status": "pass",
        "artifact_count": len(audited),
        "artifacts": audited,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--logical-manifest", type=Path, required=True)
    parser.add_argument("--direct-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.repo.resolve(), args.logical_manifest, args.direct_manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
