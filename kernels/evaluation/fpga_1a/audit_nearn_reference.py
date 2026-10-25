#!/usr/bin/env python3
"""Audit Radiance Nearn against its pinned Vortex/OpenCL source and fixture."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile


REFERENCE_REVISION = "b447578d9f3bb116d658c614cf3407153f0536da"
REFERENCE_KERNEL = "tests/opencl/nearn/kernel.cl"
REFERENCE_KERNEL_SHA256 = "43d6322e0be3adddb4c060b05dcb77a2ffd489ccdd07a5c4294e2f5f7cc17b34"
REFERENCE_KERNEL_BYTES = 662
FIXTURE_RECORDS = 4096
EXPECTED_DIGEST = 0xEAE785DADD0EC6A7
EXPECTED_TOP_FIVE = [2414, 2114, 2609, 1950, 828]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_generator(path: Path):
    spec = importlib.util.spec_from_file_location("nearn_run_audit", path)
    require(spec is not None and spec.loader is not None,
            f"cannot import generator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_expected_words(path: Path) -> list[int]:
    text = path.read_text()
    match = re.search(
        r"const uint32_t expected_nearn_raw\[\] = \{(.*?)\};",
        text, re.DOTALL,
    )
    require(match is not None, f"missing expected_nearn_raw in {path}")
    return [int(token, 16) for token in re.findall(r"0x[0-9a-fA-F]+", match.group(1))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    reference = args.reference.resolve()
    nearn = repo / "kernels/nearn"

    resolved = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", REFERENCE_REVISION],
        text=True,
    ).strip()
    require(resolved == REFERENCE_REVISION, "Vortex revision did not resolve exactly")
    upstream = subprocess.check_output([
        "git", "-C", str(reference), "cat-file", "blob",
        f"{REFERENCE_REVISION}:{REFERENCE_KERNEL}",
    ])
    require(len(upstream) == REFERENCE_KERNEL_BYTES,
            f"unexpected upstream kernel size: {len(upstream)}")
    require(sha256_bytes(upstream) == REFERENCE_KERNEL_SHA256,
            "upstream kernel digest mismatch")

    upstream_compact = "".join(upstream.decode().split())
    require(
        "*dist=(float)sqrt((lat-latLong->lat)*(lat-latLong->lat)+"
        "(lng-latLong->lng)*(lng-latLong->lng));" in upstream_compact,
        "pinned upstream Euclidean-distance equation changed",
    )
    current_kernel = (nearn / "kernel.cpp").read_text()
    current_compact = "".join(current_kernel.split())
    for term in (
        "constfloatdx=NEARN_QUERY_LAT-nearn_locations[i].lat;",
        "constfloatdy=NEARN_QUERY_LNG-nearn_locations[i].lng;",
        "nearn_distances[i]=sqrtf(dx*dx+dy*dy);",
    ):
        require(term in current_compact, f"Radiance equation term missing: {term}")

    old_data = nearn / "data"
    old_expected = nearn / "expected"
    with tempfile.TemporaryDirectory() as directory:
        generated = Path(directory)
        env = {**os.environ, "NEARN_NUM_RECORDS": str(FIXTURE_RECORDS)}
        subprocess.run(
            [sys.executable, str(nearn / "nearn.py")], cwd=generated,
            env=env, check=True,
        )
        require((generated / "data").read_bytes() == old_data.read_bytes(),
                "tracked Nearn input fixture is not reproducible")
        require((generated / "expected").read_bytes() == old_expected.read_bytes(),
                "tracked Nearn expected fixture is not reproducible")

    generator = load_generator(nearn / "run.py")
    require(generator.REFERENCE_REVISION == REFERENCE_REVISION,
            "generator references a different Vortex revision")
    require(generator.REFERENCE_KERNEL == REFERENCE_KERNEL,
            "generator references a different Vortex kernel")
    require(generator.REFERENCE_KERNEL_SHA256 == REFERENCE_KERNEL_SHA256,
            "generator references a different Vortex kernel digest")
    query_lat, query_lng, locations = generator.reference_dataset(FIXTURE_RECORDS)
    output = generator.reference_output(query_lat, query_lng, locations)
    words = [struct.unpack("<I", struct.pack("<f", value))[0] for value in output]
    fixture_words = parse_expected_words(old_expected)
    require(words == fixture_words,
            "new reference generator differs from the tracked 4096-record fixture")
    digest = generator.digest(words)
    top = sorted(range(len(output)), key=lambda index: (output[index], index))[:5]
    require(digest == EXPECTED_DIGEST, f"unexpected fixture digest: {digest:016x}")
    require(top == EXPECTED_TOP_FIVE, f"unexpected top-five indices: {top}")

    evidence = {
        "schema": "radiance-nearn-upstream-equivalence-v1",
        "status": "passed",
        "reference_repository": "vortexgpgpu/vortex",
        "reference_revision": resolved,
        "reference_kernel": REFERENCE_KERNEL,
        "reference_kernel_bytes": len(upstream),
        "reference_kernel_sha256": sha256_bytes(upstream),
        "radiance_kernel": str((nearn / "kernel.cpp").resolve()),
        "radiance_kernel_sha256": sha256(nearn / "kernel.cpp"),
        "equation": "sqrt((query_lat-lat)^2 + (query_lng-lng)^2)",
        "fixture_records": FIXTURE_RECORDS,
        "fixture_input_sha256": sha256(old_data),
        "fixture_expected_sha256": sha256(old_expected),
        "fixture_bit_exact": True,
        "fixture_expected_digest": f"{digest:016x}",
        "fixture_top_five_indices": top,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
