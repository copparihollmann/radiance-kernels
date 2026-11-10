"""Where the pinned upstream Vortex checkout lives.

Four kernels (lbm, streamcluster, hotspot3d, gaussian) verify the blob IDs and
hashes of their upstream sources before generating anything, so they need a
Vortex checkout at the pinned revision.  Which path that is depends on the
machine, so it is not baked into the generators:

    export RADIANCE_VORTEX_REFERENCE=/path/to/vortex
    git -C /path/to/vortex rev-parse HEAD   # must be REVISION below

If the variable is unset, the historical location on the development machine
is used, so existing scripts keep working.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The revision every pinned blob ID in those generators was taken from.
REVISION = "b447578d9f3bb116d658c614cf3407153f0536da"
REPOSITORY = "vortexgpgpu/vortex"

#: Where it sits on the machine this was developed on.
FALLBACK = Path("/scratch/agustin/radiance-fpga-1a/reference/vortex")

ENV_VAR = "RADIANCE_VORTEX_REFERENCE"


def default_path() -> Path:
    """The checkout to use, without checking that it exists."""
    value = os.environ.get(ENV_VAR)
    return Path(value) if value else FALLBACK


def require(path: Path | None = None) -> Path:
    """The checkout, or a ValueError naming the variable to set."""
    path = Path(path) if path is not None else default_path()
    if not path.is_dir():
        raise ValueError(
            f"reference repository does not exist: {path}\n"
            f"Clone {REPOSITORY} at {REVISION} and point {ENV_VAR} at it."
        )
    return path
