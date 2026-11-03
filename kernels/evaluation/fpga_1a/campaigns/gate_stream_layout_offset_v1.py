#!/usr/bin/env python3
"""Software gate for the STREAM array-layout offset experiment.

Refuses to declare the artifacts shippable unless, for every one of the twelve
builds (3 layouts x {copy, triad} x {logical, direct-hashed} at 2 SM):

  * the recorded build configuration is the ACCEPTED 2SM_DRAM_v2 configuration
    in every field except the new array offset;
  * the expected output digest is BIT-IDENTICAL to the accepted one.  Moving a
    base address cannot change arithmetic, so a changed digest means the build
    is wrong and nothing may ship;
  * the device ELF is still scalar: no vector ISA, no vector instruction, no
    Gemmini/VPU symbol;
  * the measured bank/rank decode of the linked arrays does what the variant
    claims: the baseline still collides on bank AND rank at every element index
    for at least one pair, and both offset variants collide at NO element index
    for ANY pair of arrays the kernel actually streams;
  * the Cyclotron functional model ran the kernel's own full host check on every
    logical-loader build and all of them passed.

Pure analysis over build outputs.  No FPGA, no FireSim, no simulator launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

# runs/hpc-profile-guard-builds-v1/stream-<op>/2sm/result.json -- the digests the
# accepted, currently-deployed 2SM_DRAM_v2 artifacts assert.
ACCEPTED_DIGEST = {
    "copy": "49fabae987622325",
    "triad": "d463e7f745222325",
}
# The arrays each operation actually streams.  Copy allocates stream_b and never
# touches it, so a bank collision involving stream_b is not a real conflict.
LIVE_ARRAYS = {
    "copy": {"stream_a", "stream_c"},
    "triad": {"stream_a", "stream_b", "stream_c"},
}
VARIANT_OFFSET = {"baseline": 0, "bank32k": 32768, "bankrank288k": 294912}
VECTOR_INSTRUCTION = re.compile(
    r"\b(?:vsetvli|vsetivli|vsetvl|vle\d+|vse\d+|vl\d+r|vs\d+r)\b")


def fail(message: str) -> None:
    raise SystemExit(f"GATE FAIL: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def live_pairs(bank_map: dict, operation: str) -> list[dict]:
    live = LIVE_ARRAYS[operation]
    return [p for p in bank_map["pairs"]
            if set(p["pair"].split("->")) <= live]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                   text=True).strip()
    if head != args.revision:
        fail(f"worktree moved: {head} != {args.revision}")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain=v1"],
                               text=True):
        fail("worktree became dirty during the build")

    readelf = repo / "llvm/llvm-muon/bin/llvm-readelf"
    objdump = repo / "llvm/llvm-muon/bin/llvm-objdump"

    audited = []
    seen = set()
    source_hashes: dict[tuple[str, str], set[str]] = {}
    for loader, spin in (("logical", False), ("direct-hashed", True)):
        manifest = json.loads((args.build / f"{loader}-manifest.json").read_text())
        if manifest.get("hardware_rebuilt") is not False:
            fail(f"{loader}: manifest does not assert hardware_rebuilt=false")
        if manifest.get("no_vpu") is not True:
            fail(f"{loader}: manifest does not assert no_vpu=true")
        rows = manifest.get("workloads", [])
        if len(rows) != 6:
            fail(f"{loader}: expected 6 artifacts, found {len(rows)}")
        for row in rows:
            name = row["name"]
            match = re.fullmatch(r"stream-(copy|triad)-layout-([a-z0-9]+)-2sm", name)
            if match is None:
                fail(f"unexpected workload name: {name}")
            operation, variant = match.group(1), match.group(2)
            if variant not in VARIANT_OFFSET:
                fail(f"unexpected layout variant: {name}")
            key = (operation, variant, loader)
            if key in seen:
                fail(f"duplicate artifact identity: {key}")
            seen.add(key)

            elf = Path(row["elf"]).resolve()
            device = elf.parent / "kernel.radiance.elf"
            result = json.loads((elf.parent / "result.json").read_text())

            # --- configuration is the accepted one, except for the offset -----
            expectations = {
                "family": "stream",
                "operation": f"stream-{operation}",
                "sm_count": 2,
                "loader": loader,
                "stream_elements": 1048576,
                "muon_warps_per_core": 4,
                "robust_end_drain": True,
                "robust_serial_fence": False,
                "robust_spin_until_reset": spin,
                "robust_post_spin": 20000,
                "diagnostic_readback": True,
                "host_full_readback": True,
                "correctness": "digest-checked",
                "benchmark_contract": "stream-v5.10-elemental-equation",
                "stream_block_cyclic_lines": False,
                "stream_block_cyclic_chunk_lines": 0,
                "stream_array_offset_bytes": VARIANT_OFFSET[variant],
                "scalar_only": True,
                "no_vpu": True,
                "vpu": False,
                "vector_isa": False,
                "gemmini": False,
                "mx": False,
                "expected_digest": ACCEPTED_DIGEST[operation],
                "status": "built",
            }
            for field, want in expectations.items():
                got = row.get(field)
                if got != want:
                    fail(f"{name}/{loader}: {field} is {got!r}, accepted value is {want!r}")
            if result.get("stream_scalar") != 3.0:
                fail(f"{name}/{loader}: scalar is not 3.0")
            if result.get("readback_samples") != 0:
                fail(f"{name}/{loader}: host check is sampled, not full")

            # --- artifacts are the ones the manifest pins ---------------------
            if not elf.is_file() or sha256(elf) != row.get("elf_sha256"):
                fail(f"{name}/{loader}: fused ELF missing or changed")
            if not device.is_file():
                fail(f"{name}/{loader}: device ELF missing")
            device_hash = sha256(device)
            if (device_hash != row.get("device_elf_sha256") or
                    device_hash != result.get("device_elf_sha256")):
                fail(f"{name}/{loader}: device ELF hash mismatch")

            # --- still scalar, still no accelerator ---------------------------
            attributes = subprocess.check_output([str(readelf), "-A", str(device)],
                                                 text=True)
            arch_match = (re.search(r'Tag_RISCV_arch:\s*"([^"]+)"', attributes) or
                          re.search(r"TagName:\s*arch\s*(?:\r?\n)\s*Value:\s*([^\s]+)",
                                    attributes))
            if arch_match is None:
                fail(f"{name}/{loader}: missing RISC-V architecture tag")
            arch = arch_match.group(1)
            base = re.sub(r"^rv(?:32|64)", "", arch.lower().split("_")[0])
            if "v" in base:
                fail(f"{name}/{loader}: vector ISA enabled: {arch}")
            disassembly = subprocess.check_output([str(objdump), "-d", str(device)],
                                                  text=True)
            if VECTOR_INSTRUCTION.search(disassembly):
                fail(f"{name}/{loader}: vector instruction found")
            lowered = disassembly.lower()
            if any(token in lowered for token in ("gemmini", "vpu", "mvin", "mvout")):
                fail(f"{name}/{loader}: accelerator/VPU symbol found")

            # --- the layout actually moved the way the variant claims ---------
            bank_map = json.loads(
                (args.evidence / f"{loader}--stream-{operation}-{variant}.bank-map.json")
                .read_text())
            if bank_map["elements"] != 1048576:
                fail(f"{name}/{loader}: bank map decoded {bank_map['elements']} elements")
            pairs = live_pairs(bank_map, operation)
            if not pairs:
                fail(f"{name}/{loader}: no live array pairs decoded")
            worst = max(p["same_bank_and_rank_fraction"] for p in pairs)
            if variant == "baseline":
                # The control must still exhibit the defect, or there is nothing
                # to fix and the experiment is void.
                if worst < 0.8:
                    fail(f"{name}/{loader}: baseline no longer collides "
                         f"(worst live pair {worst}); the premise is gone")
            elif worst != 0.0:
                fail(f"{name}/{loader}: offset variant still collides on bank+rank "
                     f"at {worst:.4f} of element indices")

            source_hashes.setdefault((operation, variant), set()).add(
                result["kernel_source_sha256"])
            audited.append({
                "name": name, "operation": operation, "variant": variant,
                "loader": loader, "sm_count": 2,
                "array_offset_bytes": VARIANT_OFFSET[variant],
                "array_offset_slice_local_bytes": VARIANT_OFFSET[variant] // 4,
                "expected_digest": row["expected_digest"],
                "expected_digest_matches_accepted": True,
                "device_elf_sha256": device_hash,
                "device_isa": arch,
                "worst_live_pair_same_bank_and_rank_fraction": worst,
                "arrays": bank_map["arrays"],
                "live_pairs": pairs,
            })

    want = {(op, variant, loader)
            for op in ACCEPTED_DIGEST for variant in VARIANT_OFFSET
            for loader in ("logical", "direct-hashed")}
    if seen != want:
        fail(f"incomplete matrix; missing {sorted(want - seen)}")
    if any(len(h) != 1 for h in source_hashes.values()):
        fail("logical and direct-hashed artifacts do not share source hashes")

    # --- the kernel's own full host check actually ran and passed -------------
    summary = json.loads((args.model / "summary.json").read_text())
    if summary.get("memory_model") != "functional-no-timing":
        fail("functional model summary is not functional-no-timing")
    if (summary.get("result_count"), summary.get("passed"), summary.get("failed")) != (6, 6, 0):
        fail(f"functional matrix is not 6/6 passing: {summary.get('result_count')}/"
             f"{summary.get('passed')}/{summary.get('failed')}")
    names = {row.get("workload") for row in summary.get("workloads", [])}
    expected_names = {f"stream-{op}-layout-{variant}"
                      for op in ACCEPTED_DIGEST for variant in VARIANT_OFFSET}
    if names != expected_names:
        fail(f"functional summary workloads are {sorted(names)}, expected "
             f"{sorted(expected_names)}")

    report = {
        "schema": "radiance-stream-layout-offset-gate-v1",
        "status": "pass",
        "revision": args.revision,
        "artifact_count": len(audited),
        "accepted_digests": ACCEPTED_DIGEST,
        "digests_unchanged": True,
        "functional_model": {
            "summary": str((args.model / "summary.json").resolve()),
            "result_count": summary["result_count"],
            "passed": summary["passed"],
            "failed": summary["failed"],
            "workloads": summary.get("workloads", []),
        },
        "artifacts": audited,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"GATE PASS: {len(audited)} artifacts, digests unchanged, "
          f"functional {summary['passed']}/{summary['result_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
