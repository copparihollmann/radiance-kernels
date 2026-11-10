"""Named hardware presets for the HPC kernels.

The same table as ``kernels/hpc.mk``.  A preset fixes the cluster count, the
end-of-kernel drain protocol and the L1 coherence mode together, because all
three are properties of the bitstream rather than choices a kernel makes.

A kernel's ``run.py`` resolves ``--preset`` here, records the name in its
manifest, and passes ``PRESET=`` and ``OCC=`` to ``make``; the defines
themselves are applied by ``hpc.mk``, so the generated ``config.h`` carries
only the shape of the data.

Keep this in step with ``hpc.mk``: ``test_hpc_presets.py`` fails if they drift.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    name: str
    clusters: int
    #: Cluster L1 banks when the L1 is incoherent; 0 means coherent, no flush.
    l1_banks: int
    #: What the memory system is, for the manifest.  Not a device define.
    memory_model: str
    note: str


#: The drain protocol every HPC measurement was taken with.  The device
#: publishes END through global memory, fences one core at a time, then spins
#: until the host resets it.  Changing these invalidates recorded cycle counts.
DRAIN = {
    "NIGHTLY_ROBUST_GMEM_END": 1,
    "NIGHTLY_ROBUST_SPIN_UNTIL_RESET": 1,
    "NIGHTLY_ROBUST_SERIAL_FENCE": 1,
    "NIGHTLY_ROBUST_POST_SPIN": 20000,
}

PRESETS = {
    p.name: p
    for p in (
        Preset("1sm_pipe", 1, 0, "latency-pipe",
               "30-cycle fixed return, no DRAM model"),
        Preset("2sm_pipe", 2, 0, "latency-pipe",
               "30-cycle fixed return, no DRAM model"),
        Preset("4sm_pipe", 4, 0, "latency-pipe",
               "30-cycle fixed return, no DRAM model"),
        Preset("2sm_ddr3", 2, 0, "DDR3", "first-generation DDR3 timing"),
        Preset("2sm_hbm", 2, 0, "HBM2", "first-generation HBM2 timing"),
        Preset("2sm_ddr3_v2", 2, 0, "DDR3_v2", "DDR3 QuadRank"),
        Preset("2sm_hbm_v2", 2, 0, "HBM2_v2", "HBM2 QuadChannel"),
        Preset("2sm_ddr3_v3", 2, 4, "DDR3_v3",
               "64-byte transactions, incoherent 4-bank cluster L1"),
        Preset("2sm_hbm_v3", 2, 4, "HBM2_v3",
               "64-byte transactions, incoherent 4-bank cluster L1"),
    )
}

DEFAULT_PRESET = "2sm_ddr3_v2"

#: Cluster counts only ever vary on the latency-pipe bitstreams, so an SM
#: scaling sweep and a real-DRAM timing arm cannot be crossed.  Anything that
#: reads an SM axis as a real-memory result is reading it wrong.
SM_SCALING_PRESETS = ("1sm_pipe", "2sm_pipe", "4sm_pipe")


def resolve(name: str) -> Preset:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(
            f"unknown preset {name!r}; choose one of {', '.join(PRESETS)}"
        ) from None


def make_args(name: str, warps_per_core: int) -> list[str]:
    """The ``make`` arguments that build for this preset."""
    resolve(name)
    return [f"PRESET={name}", f"OCC={warps_per_core}"]


def manifest(name: str, warps_per_core: int) -> dict:
    """The fields a kernel manifest records about the hardware it targets."""
    preset = resolve(name)
    return {
        "preset": preset.name,
        "sm_count": preset.clusters,
        "memory_model": preset.memory_model,
        "l1_banks": preset.l1_banks,
        "l1_coherent": preset.l1_banks == 0,
        "muon_warps_per_core": warps_per_core,
        "drain_protocol": dict(DRAIN),
    }
