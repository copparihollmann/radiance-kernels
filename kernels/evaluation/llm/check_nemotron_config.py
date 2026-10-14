#!/usr/bin/env python3
"""Check whether installed Transformers accepts the pinned Nemotron config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from profile_nemotron_state import pinned_config


def check() -> dict:
    import transformers
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

    config, source = pinned_config()
    pattern = config["hybrid_override_pattern"]
    try:
        parsed = NemotronHConfig.from_dict(config)
    except (KeyError, ValueError) as error:
        return {
            "schema": "radiance.nemotron-config-compatibility.v1",
            "model": source["model"], "revision": source["revision"],
            "config_sha256": source["config_sha256"],
            "transformers_version": transformers.__version__,
            "pattern_symbols": sorted(set(pattern)),
            "status": "incompatible",
            "exception": type(error).__name__,
            "detail": str(error),
            "scope": "config parser only; no weights or model execution",
        }
    return {
        "schema": "radiance.nemotron-config-compatibility.v1",
        "model": source["model"], "revision": source["revision"],
        "config_sha256": source["config_sha256"],
        "transformers_version": transformers.__version__,
        "pattern_symbols": sorted(set(pattern)),
        "status": "compatible",
        "parsed_layer_types": parsed.layers_block_type,
        "scope": "config parser only; no weights or model execution",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = json.dumps(check(), indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
