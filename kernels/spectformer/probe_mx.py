#!/usr/bin/env python3
"""Probe one source-derived SpectFormer projection tile in the MX math model.

This characterizes a possible FP8 systolic mapping. The full Radiance ELF
currently uses FP32 Muon linear stages, so this probe is not an end-to-end
MX execution claim and does not replace the upstream FP32 golden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from prepare import load_upstream, sha256, UPSTREAM_SOURCE_SHA256


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "lib/golden"))
from golden import fp8_code, bf16_to_f32, block_exp  # noqa: E402


def digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def run(manifest_path: Path, out: Path) -> dict:
    import torch

    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest["upstream_source"])
    if sha256(source) != UPSTREAM_SOURCE_SHA256:
        raise ValueError("upstream source differs from pinned revision")
    if manifest["depth"] < 5:
        raise ValueError("no attention projection at block 4")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(manifest["seed"])
    upstream = load_upstream(source)
    model = upstream.SpectFormer(img_size=manifest["image_size"], patch_size=16,
                                 embed_dim=manifest["embed_dim"], depth=manifest["depth"],
                                 num_classes=manifest["num_classes"], drop_rate=0.0,
                                 drop_path_rate=0.0)
    if manifest["checkpoint"]:
        checkpoint = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.eval()
    activation = []
    hook = model.blocks[4].attn.proj.register_forward_pre_hook(
        lambda _module, inputs: activation.append(inputs[0].detach().cpu().numpy()))
    size = manifest["image_size"]
    input_entry = manifest["entries"]["input.image"]
    image_path = manifest_path.parent / "image.bin"
    with image_path.open("rb") as stream:
        stream.seek(input_entry["offset_bytes"])
        input_bytes = stream.read(input_entry["size_bytes"])
    if hashlib.sha256(input_bytes).hexdigest() != input_entry["sha256"]:
        raise ValueError("exported input differs from the pinned image manifest")
    image = torch.from_numpy(np.frombuffer(input_bytes, dtype="<f4").copy().reshape(
        1, 3, size, size))
    with torch.no_grad():
        logits = model(image).detach().cpu().numpy().astype("<f4")
    hook.remove()
    if digest(logits) != manifest["entries"]["output.logits"]["sha256"]:
        raise ValueError("reconstructed source model does not match exported logits")
    if len(activation) != 1:
        raise ValueError("projection hook did not fire exactly once")
    x = np.ascontiguousarray(activation[0][0, :16, :32], dtype=np.float32)
    weight = np.ascontiguousarray(model.blocks[4].attn.proj.weight.detach().cpu().numpy()[:16, :32].T,
                                  dtype=np.float32)
    # One e8m0 scale per 32-element K group and A row/B column. Keep the
    # encoded FP8 magnitude below roughly 2: the MX accumulator model has a
    # deliberately narrow internal exponent range, so a scale chosen solely
    # to use FP8's full 448 range can overflow before the e8m0 rescale.
    def scales(maximum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        exponents = np.asarray([
            block_exp(int(np.float32(value).view(np.uint32))) + 8 if value else 0
            for value in maximum], dtype=np.int32)
        if np.any((exponents + 127 < 0) | (exponents + 127 > 254)):
            raise ValueError("source tile needs an e8m0 scale outside the finite range")
        codes = (exponents + 127).astype(np.uint8)[None, :]
        values = np.ldexp(np.ones(len(exponents), dtype=np.float32), exponents)
        return codes, values

    scale_a, factor_a = scales(np.max(np.abs(x), axis=1))
    scale_b, factor_b = scales(np.max(np.abs(weight), axis=0))
    encode = np.vectorize(fp8_code, otypes=[np.uint8])
    a = encode(x / factor_a[:, None])
    b = encode(weight / factor_b[None, :])
    out.mkdir(parents=True, exist_ok=True)
    for name, values in (("A.bin", a), ("B.bin", b), ("SA.bin", scale_a), ("SB.bin", scale_b)):
        (out / name).write_bytes(values.tobytes())
    executable = out / "mx_golden"
    subprocess.run(["g++", "-O2", "-std=c++17", str(ROOT / "lib/golden/mx_golden.cpp"),
                    "-o", str(executable)], check=True)
    result_path = out / "C.bin"
    subprocess.run([str(executable), "16", "16", "32", str(out / "A.bin"),
                    str(out / "B.bin"), str(out / "SA.bin"), str(out / "SB.bin"),
                    str(result_path), "0"], check=True)
    hardware = bf16_to_f32(np.frombuffer(result_path.read_bytes(), dtype="<u2").reshape(16, 16))
    # Decode the exact bits accepted by the hardware model. This comparison
    # separates FP8 quantization from the MX accumulator's arithmetic.
    qa = torch.from_numpy(a.copy()).view(torch.float8_e4m3fn).float().numpy()
    qb = torch.from_numpy(b.copy()).view(torch.float8_e4m3fn).float().numpy()
    ideal_quantized = (qa * factor_a[:, None]) @ (qb * factor_b[None, :])
    reference = x @ weight
    if not (np.isfinite(hardware).all() and np.isfinite(ideal_quantized).all()):
        raise ValueError("MX probe produced nonfinite values; operand scales are unsuitable")
    record = {
        "scope": "spectformer_block4_attention_projection_16x16x32_partial_tile",
        "model": "spectformer_vanilla", "upstream_source_sha256": sha256(source),
        "image_sha256": manifest["image_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "weight_provenance": manifest["weight_provenance"],
        "format": "mx_fp8_e4m3_e8m0_bounded_per_row_column_scales_bf16_output",
        "operand_sha256": {name: sha256(out / name) for name in ("A.bin", "B.bin", "SA.bin", "SB.bin")},
        "model_output_sha256": sha256(result_path),
        "mx_model_sha256": sha256(executable),
        "max_abs_error_vs_fp32_partial_projection": float(np.max(np.abs(hardware - reference))),
        "rms_error_vs_fp32_partial_projection": float(np.sqrt(np.mean((hardware - reference) ** 2))),
        "max_abs_error_vs_ideal_fp8_matmul": float(np.max(np.abs(hardware - ideal_quantized))),
        "spike_execution": False, "radiance_device_execution": False,
        "performance_measurement": False,
    }
    (out / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.image_manifest, args.out), indent=2))


if __name__ == "__main__":
    main()
