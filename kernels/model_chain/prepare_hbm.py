#!/usr/bin/env python3
"""Build a fixed model-chain source bundle for the nightly HBM launch protocol.

The checkpoint graph and stage code are copied unchanged.  Only launch,
completion, verification reporting, and the host program change.  The output
is a new build directory; the source checkpoint build is never edited.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from compile_decoder import build_device


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform_kernel(source):
    if source.count('#include "kernel_verify.h"') != 1:
        raise ValueError("expected one verification header")
    source = source.replace('#include "kernel_verify.h"',
                            '#include "kernel_verify.h"\n'
                            '#include "model_hbm_bridge.h"')
    if source.count("\nint main() {\n") != 1:
        raise ValueError("expected one device main")
    source = source.replace("\nint main() {\n",
                            "\nint main() {\n  model_hbm_wait_launch();\n")
    park = ("  if (vx_core_id() != 0) { for (;;) {} }"
            if "  if (vx_core_id() != 0) { for (;;) {} }" in source else
            "  if (mu_hart_id() != 0) { for (;;) {} }")
    if source.count(park) != 1:
        raise ValueError("expected one nonzero-hart park")
    source = source.replace(park, "  if (mu_hart_id() != 0) { "
                            "model_hbm_end(); return 0; }")
    source, chunk_returns = re.subn(
        r"if \(!verify_chunk_(\d+)\(\)\) return 0;",
        r"if (!verify_chunk_\1()) { model_hbm_end(); return 0; }",
        source)
    source = source.replace("mu_tohost(", "model_hbm_status(")
    source, status_returns = re.subn(
        r"model_hbm_status\(([^;]+)\);\s*return 0;",
        r"model_hbm_status(\1); model_hbm_end(); return 0;",
        source)
    if not status_returns or "mu_tohost(" in source:
        raise ValueError("device completion status was not transformed")
    # A long model run must expose progress in the UART while the host waits.
    schedule_count = 0

    def schedule_with_heartbeat(match):
        nonlocal schedule_count
        schedule_count += 1
        if schedule_count == 1 or schedule_count % 64:
            return match.group(0)
        return ("  if (mu_hart_id() == 0) model_hbm_put(" 
                "NIGHTLY_PB_HEARTBEAT, " + str(schedule_count) + "u);\n"
                + match.group(0))

    source = re.sub(r"^  mu_schedule\([^\n]+\);", schedule_with_heartbeat,
                    source, flags=re.MULTILINE)
    if not schedule_count:
        raise ValueError("no scheduled stages found")
    return source, {"scheduled_calls": schedule_count,
                    "chunk_return_sites": chunk_returns,
                    "status_return_sites": status_returns}


def prepare(source_dir, output_dir):
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if source_dir == output_dir or source_dir in output_dir.parents:
        raise ValueError("output must be outside the source build")
    source_manifest_path = source_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    clusters = source_manifest.get("batch_clusters", 1)
    if clusters not in (1, 2):
        raise ValueError("only one or two cluster model batches are supported")
    names = source_manifest.get("device_source_files", [])
    if "kernel.cpp" not in names or not names:
        raise ValueError("source build has no device source list")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        if Path(name).name != name or not name.endswith(".cpp"):
            raise ValueError(f"invalid generated source name: {name}")
        shutil.copy2(source_dir / name, output_dir / name)
    host_source = (HERE / "host_hbm.cpp").read_text()
    (output_dir / "host.cpp").write_text(
        f"#define MODEL_HBM_CLUSTERS {clusters}u\n" + host_source)
    shutil.copy2(HERE / "model_hbm_bridge.h", output_dir / "model_hbm_bridge.h")
    shutil.copy2(HERE / "model_hbm_pb.h", output_dir / "model_hbm_pb.h")
    transformed, changes = transform_kernel((output_dir / "kernel.cpp").read_text())
    (output_dir / "kernel.cpp").write_text(transformed)
    verifier_count = 0
    for verifier in output_dir.glob("verify_chunk_*.cpp"):
        text = verifier.read_text()
        if '#include "kernel_verify.h"' not in text or "mu_tohost(" not in text:
            raise ValueError(f"unexpected verifier source: {verifier}")
        text = text.replace('#include "kernel_verify.h"',
                            '#include "kernel_verify.h"\n'
                            '#include "model_hbm_bridge.h"')
        verifier.write_text(text.replace("mu_tohost(", "model_hbm_status("))
        verifier_count += 1
    dependencies = [name for name in names if name != "kernel.cpp"]
    makefile = ("PROJECT = model_chain\nMU_SRCS = kernel.cpp\n"
                "HOST_SRCS = host.cpp\n"
                "MU_SRC_DEPS = " + " ".join(dependencies) + "\n"
                f"RADIANCE_LIB_PATH := {ROOT / 'lib'}\n"
                "RADIANCE_INCLUDE_PATH := $(RADIANCE_LIB_PATH)/include\n"
                "GEMMINI_SW_PATH := $(RADIANCE_LIB_PATH)/mxgemmini\n"
                f"SOC_DIR := {ROOT / 'soc'}\n"
                f"LLVM_MUON ?= {ROOT / 'llvm/llvm-muon'}\n"
                f"MU_LINKER_SCRIPT := {ROOT / 'lib/linker/mu_link_nightly.ld'}\n"
                "MU_ADDR_HASH := 0\n"
                f"EXTRA_MU_CFLAGS += -I{HERE}\n"
                f"include {ROOT / 'kernels/common.mk'}\n")
    (output_dir / "Makefile").write_text(makefile)
    manifest = dict(source_manifest)
    manifest.update({"device_elf_built": False,
                     "device_execution": False,
                     "target_protocol": ("nightly_hbm_printbuf_cluster0" if clusters == 1
                                         else "nightly_hbm_printbuf_clusters"),
                     "hbm_launch_clusters": clusters,
                     "source_manifest_sha256": hash_file(source_manifest_path),
                     "source_device_elf_sha256": source_manifest.get(
                         "radiance_elf_sha256"),
                     "hbm_transform": changes,
                     "hbm_verifier_files": verifier_count})
    manifest.pop("radiance_elf_sha256", None)
    manifest.pop("soc_elf_sha256", None)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    target = prepare(args.source, args.out)
    if args.build:
        build_device(target)
    print(target)


if __name__ == "__main__":
    main()
