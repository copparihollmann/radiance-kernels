#!/usr/bin/env python3
"""Fail-closed audit for faithful scalar Vortex/Parboil LBM artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess


REFERENCE_REVISION = "b447578d9f3bb116d658c614cf3407153f0536da"
REFERENCE_KERNEL_SHA256 = (
    "174ff2dfbe7b629c4277b22adfc186c551eb20c16cc3b6ce04937471fb0a083b"
)
REFERENCE_OBSTACLE_SHA256 = (
    "3d9454aaf7abb4d4b4d9866a1b11d0024e770ad25887b43d0bbe8d951517f2c6"
)
REFERENCE_GOLD_SHA256 = (
    "727feb99cd3db0f6951ab7a2bbd8fd850edb007837e774590448dff652923f52"
)
EXPECTED_BOUNDARIES = {
    1: [0, 2048],
    2: [0, 1024, 2048],
    4: [0, 512, 1024, 1536, 2048],
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


def read_arch(readelf: Path, elf: Path) -> str:
    output = subprocess.check_output([str(readelf), "-A", str(elf)], text=True)
    match = re.search(r'(?:Value:|Tag_RISCV_arch:)\s*"?(rv(?:32|64)[^"\s]+)',
                      output)
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
    module_path = repo / "kernels/lbm/lbm.py"
    spec = importlib.util.spec_from_file_location("radiance_lbm_audit",
                                                  module_path)
    require(spec is not None and spec.loader is not None,
            f"cannot import current LBM source: {module_path}")
    lbm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lbm)
    current_source_hash = lbm.source_digest()
    current_revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
    ).strip()
    manifests = [json.loads(logical_path.read_text()),
                 json.loads(direct_path.read_text())]
    rows = [row for manifest in manifests
            for row in manifest.get("workloads", [])]
    require(all(manifest.get("no_vpu") is True for manifest in manifests),
            "manifest lacks no_vpu=true")
    require(all(manifest.get("hardware_rebuilt") is False
                for manifest in manifests),
            "manifest does not assert hardware_rebuilt=false")
    require(len(rows) == 6,
            f"expected six logical/direct LBM rows, found {len(rows)}")

    device_readelf = repo / "llvm/llvm-muon/bin/llvm-readelf"
    device_objdump = repo / "llvm/llvm-muon/bin/llvm-objdump"
    paired: dict[int, list[str]] = {}
    identities = set()
    artifact_revisions = set()
    output_digests = set()
    audited = []
    for row in rows:
        sm = int(row.get("sm_count", 0))
        loader = row.get("loader")
        label = f"{sm}SM/{loader}"
        require(sm in EXPECTED_BOUNDARIES, f"unexpected SM count: {sm}")
        require(loader in ("logical", "direct-hashed"), f"bad loader: {loader}")
        require((sm, loader) not in identities,
                f"duplicate LBM artifact identity: {label}")
        identities.add((sm, loader))
        require(row.get("family") == "lbm", f"{label} has wrong family")
        require(row.get("operation") == "d3q19-gather-stream-collide",
                f"{label} has wrong operation")
        require(row.get("benchmark_contract") ==
                "vortex-parboil-lbm-official-default-one-timestep",
                f"{label} has wrong benchmark contract")
        require(row.get("dataset_contract") ==
                "32_32_8_ldc.of-and-32_32_8_ldc.gold",
                f"{label} has wrong dataset contract")
        require(row.get("dimensions") == [32, 32, 8] and
                row.get("padded_dimensions") == [40, 32, 12] and
                row.get("timesteps") == 1,
                f"{label} changes the official dimensions or timestep count")
        require(row.get("active_cells") == 8192 and
                row.get("destination_populations_checked") == 155648 and
                row.get("velocity_words_checked") == 24576 and
                row.get("expected_output_raw_words") == 307200,
                f"{label} does not cover the full output contract")
        require(row.get("expected_population_digest") == "5f16d36bf8d711bd" and
                row.get("expected_output_raw_digest") == "e1fa359562105965" and
                row.get("gold_velocity_digest") == "dac47aa72f907f9d",
                f"{label} has wrong oracle digests")
        require(row.get("reference_revision") == REFERENCE_REVISION and
                row.get("reference_verified") is True and
                row.get("official_dimensions_verified") is True and
                row.get("official_default_one_timestep_verified") is True,
                f"{label} lacks pinned official-reference verification")
        reference_files = row.get("reference_files") or {}
        require(reference_files.get("tests/opencl/lbm/kernel.cl", {}).get(
                    "sha256") == REFERENCE_KERNEL_SHA256,
                f"{label} has wrong pinned kernel")
        require(reference_files.get("tests/opencl/lbm/32_32_8_ldc.of", {}).get(
                    "sha256") == REFERENCE_OBSTACLE_SHA256,
                f"{label} has wrong pinned obstacle field")
        require(reference_files.get("tests/opencl/lbm/32_32_8_ldc.gold", {}).get(
                    "sha256") == REFERENCE_GOLD_SHA256,
                f"{label} has wrong pinned velocity gold")

        partition = row.get("partition") or {}
        require(partition.get("scheme") ==
                "physical-threadblock-contiguous-groups-of-four-complete-cells",
                f"{label} has wrong physical-SM ownership")
        require(partition.get("group_boundaries") == EXPECTED_BOUNDARIES[sm] and
                partition.get("group_cells") == 4 and
                partition.get("group_bytes") == 320 and
                partition.get("cache_lines_per_group") == 5,
                f"{label} has wrong cache-line work groups")
        require(partition.get("complete") is True and
                partition.get("overlap") is False and
                partition.get("cache_line_safe") is True,
                f"{label} lacks complete disjoint cache-line ownership")
        require(row.get("operation_order") ==
                "pinned-kernel.cl-scalar-fp32-no-contraction",
                f"{label} changes floating-point operation order")
        require(row.get("no_vpu") is True and row.get("scalar_only") is True and
                row.get("vpu") is False and row.get("vector_isa") is False and
                row.get("gemmini") is False and row.get("mx") is False,
                f"{label} is not scalar no-VPU/no-accelerator")
        require(row.get("robust_end_drain") is True and
                row.get("robust_serial_fence") is True and
                row.get("robust_post_spin") == 20000,
                f"{label} lacks robust completion")
        require(bool(row.get("robust_spin_until_reset")) ==
                (loader == "direct-hashed"),
                f"{label} has wrong FPGA spin/reset mode")

        fused = Path(row["elf"]).resolve()
        result_path = fused.parent / "result.json"
        result = json.loads(result_path.read_text())
        device = Path(row["device_elf"]).resolve()
        require(fused.is_file() and device.is_file(),
                f"missing artifact for {label}")
        require(Path(result["device_elf"]).resolve() == device,
                f"device ELF path drift for {label}")
        for field in ("expected_population_digest", "expected_output_raw_digest",
                      "expected_output_raw_words", "gold_velocity_digest"):
            require(result.get(field) == row.get(field),
                    f"manifest/result {field} drift for {label}")
        fused_hash = sha256(fused)
        device_hash = sha256(device)
        require(fused_hash == row.get("elf_sha256") == result.get("elf_sha256"),
                f"fused ELF hash mismatch for {label}")
        require(device_hash == row.get("device_elf_sha256") ==
                result.get("device_elf_sha256"),
                f"device ELF hash mismatch for {label}")
        device_arch = read_arch(device_readelf, device)
        host_arch = read_arch(Path("readelf"), fused)
        require(not has_vector_isa(device_arch),
                f"device ISA enables vector for {label}: {device_arch}")
        require(not has_vector_isa(host_arch),
                f"host ISA enables vector for {label}: {host_arch}")
        disassembly = subprocess.check_output(
            [str(device_objdump), "-d", str(device)], text=True,
        )
        for instruction in ("fadd.s", "fsub.s", "fmul.s", "fdiv.s"):
            require(re.search(rf"\b{re.escape(instruction)}\b", disassembly)
                    is not None,
                    f"{label} lacks scalar {instruction} arithmetic")
        require(re.search(r"\bf(?:madd|msub|nmadd|nmsub)\.s\b", disassembly)
                is None, f"{label} contracts the pinned FP32 equation")
        require(not re.search(
            r"\b(?:vsetvli|vsetivli|vsetvl|vle\d+|vse\d+|vl\d+r|vs\d+r)\b",
            disassembly), f"{label} contains vector instructions")
        require("gemmini" not in disassembly.lower() and
                "vpu" not in disassembly.lower(),
                f"{label} contains accelerator/VPU symbols")
        source_hash = result.get("kernel_source_sha256")
        require(source_hash == current_source_hash,
                f"{label} was not built from the current LBM source")
        artifact_revision = result.get("kernel_revision")
        require(bool(artifact_revision), f"{label} lacks a source revision")
        ancestor = subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor",
             artifact_revision, current_revision],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        require(ancestor.returncode == 0,
                f"{label} source revision is not an ancestor of current HEAD")
        artifact_revisions.add(artifact_revision)
        paired.setdefault(sm, []).append(source_hash)
        output_digests.add(result.get("expected_output_raw_digest"))
        audited.append({
            "sm_count": sm, "loader": loader,
            "group_boundaries": partition["group_boundaries"],
            "fused_elf": str(fused), "fused_elf_sha256": fused_hash,
            "device_elf": str(device), "device_elf_sha256": device_hash,
            "device_isa": device_arch, "host_isa": host_arch,
            "complete_output_words": row["expected_output_raw_words"],
            "scalar_fp32_no_contraction": True, "vpu": False,
        })

    require(set(paired) == {1, 2, 4} and
            all(len(values) == 2 for values in paired.values()),
            "logical/direct SM coverage is incomplete")
    require(identities == {(sm, loader) for sm in (1, 2, 4)
                           for loader in ("logical", "direct-hashed")},
            "logical/direct artifact identities are incomplete")
    require(len({value for values in paired.values() for value in values}) == 1,
            "logical/direct artifacts do not share one source identity")
    require(len(artifact_revisions) == 1,
            "logical/direct artifacts do not share one source revision")
    require(output_digests == {"e1fa359562105965"},
            "logical/direct artifacts do not share one output oracle")
    return {
        "schema": "radiance-lbm-artifact-audit-v1",
        "status": "pass", "reference_revision": REFERENCE_REVISION,
        "current_source_revision": current_revision,
        "current_source_sha256": current_source_hash,
        "artifact_source_revision": next(iter(artifact_revisions)),
        "logical_manifest": str(logical_path.resolve()),
        "logical_manifest_sha256": sha256(logical_path),
        "direct_manifest": str(direct_path.resolve()),
        "direct_manifest_sha256": sha256(direct_path),
        "dimensions": [32, 32, 8], "timesteps": 1,
        "artifacts": sorted(audited,
                            key=lambda row: (row["sm_count"], row["loader"])),
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
    print(f"PASS: audited {len(evidence['artifacts'])} LBM artifacts -> {args.out}")


if __name__ == "__main__":
    main()
