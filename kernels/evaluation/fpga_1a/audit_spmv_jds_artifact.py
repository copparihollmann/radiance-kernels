#!/usr/bin/env python3
"""Fail-closed audit for faithful scalar Vortex/Parboil JDS SpMV artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


REFERENCE_REVISION = "b447578d9f3bb116d658c614cf3407153f0536da"
REFERENCE_KERNEL_BLOB = "3f44205e892b6523a010a04cbb1fd5cad995a02f"
REFERENCE_KERNEL_SHA256 = "8902bc12280ff7474fa6ea2c5dcb421006e80a7641badeefb2095942d3430674"
REFERENCE_MAIN_SHA256 = "f3ed7dc317c2bca7fe238ff2c665c1263fa13ade8598675b911df6ca40037f49"
REFERENCE_CONVERTER_SHA256 = "99b58152f88e4ace7a84fbfcb5882384bd58d7fce235ec36c4495c6f892b253b"
DATASETS = {
    "Dubcova3": {
        "dimension": 146689,
        "stored_nnz": 1891669,
        "expanded_nnz": 3636649,
        "jds_padded_entries": 3637664,
        "hashes": {
            "matrix_sha256": "c8c977dac955ee488951a08a41a1196fbaccc78ace1be9b79e526efc7b3e3997",
            "vector_sha256": "cb648152051ab17f80c69c7c9caccdc12611b42a2765e9f56809d1ef8823f1a9",
            "gold_sha256": "a0d78da12e51f1270bc4a21d4bc4388fd793b6ac3833e2c333ae7c47a1f5a492",
        },
    },
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


def audit(repo: Path, logical_path: Path, direct_path: Path,
          dataset_name: str) -> dict:
    expected = DATASETS[dataset_name]
    manifests = [json.loads(logical_path.read_text()),
                 json.loads(direct_path.read_text())]
    rows = [row for manifest in manifests for row in manifest.get("workloads", [])]
    require(all(manifest.get("no_vpu") is True for manifest in manifests),
            "manifest lacks no_vpu=true")
    require(all(manifest.get("hardware_rebuilt") is False for manifest in manifests),
            "manifest does not assert hardware_rebuilt=false")
    require(len(rows) == 6, f"expected six logical/direct SpMV rows, found {len(rows)}")

    device_readelf = repo / "llvm/llvm-muon/bin/llvm-readelf"
    device_objdump = repo / "llvm/llvm-muon/bin/llvm-objdump"
    paired: dict[int, list[str]] = {}
    generated_identities = set()
    audited = []
    for row in rows:
        sm = int(row.get("sm_count", 0))
        loader = row.get("loader")
        label = f"{sm}SM/{loader}"
        require(sm in (1, 2, 4), f"unexpected SM count: {sm}")
        require(loader in ("logical", "direct-hashed"), f"bad loader: {loader}")
        require(row.get("family") == "spmv_jds", f"{label} has wrong family")
        require(row.get("operation") == "sparse-matrix-vector-product",
                f"{label} has wrong operation")
        require(row.get("dataset") == dataset_name, f"{label} has wrong dataset")
        for field in ("dimension", "stored_nnz", "expanded_nnz",
                      "jds_padded_entries"):
            require(row.get(field) == expected[field],
                    f"{label} has wrong {field}")
        require(row.get("output_element_count") == expected["dimension"],
                f"{label} does not check every output row")
        require(row.get("reference_revision") == REFERENCE_REVISION,
                f"{label} has wrong Vortex revision")
        require(row.get("reference_kernel_git_blob") == REFERENCE_KERNEL_BLOB and
                row.get("reference_kernel_sha256") == REFERENCE_KERNEL_SHA256,
                f"{label} has wrong pinned kernel identity")
        require(row.get("reference_main") == "tests/opencl/spmv/main.cc" and
                row.get("reference_main_sha256") == REFERENCE_MAIN_SHA256,
                f"{label} has wrong pinned host identity")
        require(row.get("reference_converter") ==
                "tests/opencl/spmv/convert_dataset.c" and
                row.get("reference_converter_sha256") ==
                REFERENCE_CONVERTER_SHA256,
                f"{label} has wrong pinned converter identity")
        require(row.get("benchmark_contract") ==
                "pinned-vortex-parboil-spmv-jds-naive",
                f"{label} has wrong benchmark contract")
        require(row.get("reference_contract") ==
                "vortex-parboil-jds-equation-and-upstream-gold-tolerance",
                f"{label} has wrong reference contract")
        require(row.get("accumulation_order") ==
                "strict-increasing-k-no-contraction",
                f"{label} changes accumulation order")
        require(row.get("partition") ==
                "nonzero-balanced-contiguous-16-float-output-line-aligned",
                f"{label} has wrong multi-SM partition")
        boundaries = row.get("partition_boundaries", [])
        require(len(boundaries) == sm + 1 and boundaries[0] == 0 and
                boundaries[-1] == expected["dimension"],
                f"{label} partition does not exactly cover all rows")
        require(all(a < b for a, b in zip(boundaries, boundaries[1:])),
                f"{label} partition overlaps or has empty ownership")
        require(all(value % 16 == 0 for value in boundaries[1:-1]),
                f"{label} splits a 64-byte output cache line")
        work = row.get("partition_nnz", [])
        require(len(work) == sm and sum(work) == expected["expanded_nnz"],
                f"{label} partition does not cover every real nonzero")
        require(row.get("partition_complete") is True and
                row.get("partition_overlap") is False,
                f"{label} lacks exact-once partition proof")
        require(row.get("correctness") ==
                "complete-output-upstream-tolerance-exact-digest-and-guards" and
                bool(row.get("expected_digest")),
                f"{label} lacks full-output correctness")
        validation = row.get("upstream_gold_validation") or {}
        require(validation.get("checked") == expected["dimension"] and
                validation.get("bad") == 0,
                f"{label} does not pass the complete upstream gold oracle")
        files = row.get("dataset_files") or {}
        for field, value in expected["hashes"].items():
            require(files.get(field) == value, f"{label} has wrong {field}")
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
        device = fused.parent / "kernel.radiance.elf"
        result_path = fused.parent / "result.json"
        require(fused.is_file() and device.is_file() and result_path.is_file(),
                f"missing artifact for {label}")
        result = json.loads(result_path.read_text())
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
        require(re.search(r"\bfmul\.s\b", disassembly) is not None and
                re.search(r"\bfadd\.s\b", disassembly) is not None,
                f"{label} lacks scalar multiply-then-add arithmetic")
        require(re.search(r"\bf(?:madd|msub|nmadd|nmsub)\.s\b", disassembly) is None,
                f"{label} contracts the pinned multiply-then-add equation")
        require(not re.search(r"\b(?:vsetvli|vle\d+|vse\d+|vl\d+r|vs\d+r)\b",
                              disassembly),
                f"{label} contains vector instructions")
        require("gemmini" not in disassembly.lower() and
                "vpu" not in disassembly.lower(),
                f"{label} contains accelerator/VPU symbols")
        source_hash = result.get("kernel_source_sha256")
        require(bool(source_hash), f"{label} lacks source hash")
        generated = row.get("generated_sha256") or {}
        require(set(generated) == {"data.bin", "expected.bin", "group_bounds.bin",
                                   "indices.bin", "jds_ptr.bin",
                                   "row_to_jds.bin", "x.bin"},
                f"{label} lacks complete generated-array hashes")
        generated_identities.add(json.dumps(generated, sort_keys=True))
        paired.setdefault(sm, []).append(source_hash)
        audited.append({
            "sm_count": sm, "loader": loader,
            "partition_boundaries": boundaries, "partition_nnz": work,
            "fused_elf": str(fused), "fused_elf_sha256": fused_hash,
            "device_elf": str(device), "device_elf_sha256": device_hash,
            "device_isa": device_arch, "host_isa": host_arch,
            "scalar_multiply_then_add": True, "vpu": False,
        })

    require(set(paired) == {1, 2, 4} and
            all(len(values) == 2 for values in paired.values()),
            "logical/direct SM coverage is incomplete")
    require(len({value for values in paired.values() for value in values}) == 1,
            "logical/direct artifacts do not share one source identity")
    require(len(generated_identities) == 1,
            "logical/direct artifacts do not share identical generated JDS arrays")
    return {
        "schema": "radiance-spmv-jds-artifact-audit-v1",
        "status": "pass", "dataset": dataset_name,
        "reference_revision": REFERENCE_REVISION,
        "reference_kernel_git_blob": REFERENCE_KERNEL_BLOB,
        "reference_kernel_sha256": REFERENCE_KERNEL_SHA256,
        "logical_manifest": str(logical_path.resolve()),
        "logical_manifest_sha256": sha256(logical_path),
        "direct_manifest": str(direct_path.resolve()),
        "direct_manifest_sha256": sha256(direct_path),
        "artifacts": sorted(audited,
                            key=lambda row: (row["sm_count"], row["loader"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--logical-manifest", type=Path, required=True)
    parser.add_argument("--direct-manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="Dubcova3")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evidence = audit(args.repo.resolve(), args.logical_manifest.resolve(),
                     args.direct_manifest.resolve(), args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"PASS: audited {len(evidence['artifacts'])} JDS SpMV artifacts -> {args.out}")


if __name__ == "__main__":
    main()
