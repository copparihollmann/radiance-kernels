# hpc.mk -- the platform axis for the HPC kernels (btree, lbm, stream, ...).
#
# A kernel Makefile sets PROJECT / MU_SRCS / HOST_SRCS, includes ../common.mk,
# then includes this file.  Two things vary independently and are easy to
# confuse, so they are named separately:
#
#   PRESET  what hardware the ELF is built for.  Fixes the cluster count, the
#           end-of-kernel drain protocol and the L1 coherence mode together,
#           because all three are properties of the bitstream rather than
#           choices a kernel gets to make.
#   ILP     how hard the kernel unrolls.  One .cpp per factor, all including
#           the kernel's kernel_impl.hpp -- the same arrangement as
#           kernels/vecadd.  That axis lives in the kernel Makefile, not here.
#
#   make PRESET=4sm_pipe                  # 4-cluster latency-pipe bitstream
#   make PRESET=2sm_ddr3_v3 OCC=2         # v3 bitstream, two warps per core
#
# Presets, and what distinguishes them:
#
#   1sm_pipe 2sm_pipe 4sm_pipe   30-cycle latency pipe, no DRAM model.  The
#                                only presets that vary the cluster count, so
#                                an SM-scaling sweep and a real-DRAM timing
#                                arm cannot be crossed.
#   2sm_ddr3 2sm_hbm             first-generation real memory timing.
#   2sm_ddr3_v2 2sm_hbm_v2       re-characterized timing (DDR3 QuadRank,
#                                HBM2 QuadChannel).
#   2sm_ddr3_v3 2sm_hbm_v3       64-byte transactions and a 4-bank cluster L1
#                                that is incoherent, so kernels must flush it
#                                explicitly before the host reads results.

PRESET ?= 2sm_ddr3_v2

HPC_PRESETS := 1sm_pipe 2sm_pipe 4sm_pipe 2sm_ddr3 2sm_hbm \
               2sm_ddr3_v2 2sm_hbm_v2 2sm_ddr3_v3 2sm_hbm_v3

ifeq ($(filter $(PRESET),$(HPC_PRESETS)),)
$(error PRESET=$(PRESET) is not one of: $(HPC_PRESETS))
endif

HPC_CLUSTERS_1sm_pipe    := 1
HPC_CLUSTERS_2sm_pipe    := 2
HPC_CLUSTERS_4sm_pipe    := 4
HPC_CLUSTERS_2sm_ddr3    := 2
HPC_CLUSTERS_2sm_hbm     := 2
HPC_CLUSTERS_2sm_ddr3_v2 := 2
HPC_CLUSTERS_2sm_hbm_v2  := 2
HPC_CLUSTERS_2sm_ddr3_v3 := 2
HPC_CLUSTERS_2sm_hbm_v3  := 2

# The v3 cluster L1 is incoherent and 4-banked (radiance L1IncoherentKey,
# L1BanksKey); a kernel there publishes results with mu_l1_flush.  0 means a
# coherent L1 and no flush.
HPC_L1_BANKS_2sm_ddr3_v3 := 4
HPC_L1_BANKS_2sm_hbm_v3  := 4

HPC_CLUSTERS := $(HPC_CLUSTERS_$(PRESET))
HPC_L1_BANKS := $(or $(HPC_L1_BANKS_$(PRESET)),0)

# End-of-kernel drain protocol.  This combination is what the HPC measurements
# were taken with: the device publishes END through global memory, fences one
# core at a time, then spins until the host resets it.  Override a single knob
# on the command line if an experiment needs to; do not quietly change the
# defaults, because every recorded cycle count assumes them.
HPC_ROBUST_GMEM_END         ?= 1
HPC_ROBUST_SPIN_UNTIL_RESET ?= 1
HPC_ROBUST_SERIAL_FENCE     ?= 1
HPC_ROBUST_POST_SPIN        ?= 20000

# Register budget.  Muon's renamer allocates a physical register on a warp's
# first write of an architectural register and never frees it, and a core has
# 256, so warps_per_core * registers_per_warp must stay <= 256.  Same rule as
# kernels/nightly/nightly.mk.
OCC ?= 4
HPC_GPRS := $(if $(filter 1 2,$(OCC)),128,$(if $(filter 3 4,$(OCC)),64,32))

HPC_DEFS := \
  -DNIGHTLY_CLUSTERS=$(HPC_CLUSTERS) \
  -DNIGHTLY_OCC=$(OCC) \
  -DMU_L1_BANKS=$(HPC_L1_BANKS) \
  -DNIGHTLY_ROBUST_GMEM_END=$(HPC_ROBUST_GMEM_END) \
  -DNIGHTLY_ROBUST_SPIN_UNTIL_RESET=$(HPC_ROBUST_SPIN_UNTIL_RESET) \
  -DNIGHTLY_ROBUST_SERIAL_FENCE=$(HPC_ROBUST_SERIAL_FENCE) \
  -DNIGHTLY_ROBUST_POST_SPIN=$(HPC_ROBUST_POST_SPIN)u \
  -DHPC_PRESET_NAME=\"$(PRESET)\"

MU_CFLAGS    += $(HPC_DEFS) -mllvm -riscv-max-allocatable-gprs=$(HPC_GPRS)
HOST_CXXFLAGS += $(HPC_DEFS)

# Changing a preset changes no file, so stamp it and let the objects depend on
# the stamp.  Without this, `make PRESET=4sm_pipe` after `make` silently keeps
# the 2-cluster objects.
#
# The stamp alone is not enough for the generated DATA, because every `make`
# invocation rewrites it during parsing -- including `make clean`, which would
# consume the change and leave data generated for the previous preset in place.
# So check the data itself: if generated/config.h was written for a different
# cluster count, remove it and let the kernel's data rule regenerate it.
$(shell [ -f generated/config.h ] && \
  ! grep -qx "\#define HPC_SM_COUNT $(HPC_CLUSTERS)u" generated/config.h && \
  rm -f generated/config.h)

HPC_STAMP := .hpc_preset.stamp
HPC_STAMP_TEXT := $(PRESET) OCC=$(OCC) $(HPC_DEFS)
$(shell printf '%s\n' "$(HPC_STAMP_TEXT)" | cmp -s - $(HPC_STAMP) || printf '%s\n' "$(HPC_STAMP_TEXT)" > $(HPC_STAMP))

$(addsuffix .mu.o,$(basename $(MU_SRCS))) $(addsuffix .host.o,$(basename $(HOST_SRCS))): $(HPC_STAMP)

.PHONY: print-preset
print-preset:
	@echo "preset=$(PRESET) clusters=$(HPC_CLUSTERS) occ=$(OCC) gprs=$(HPC_GPRS) l1_banks=$(HPC_L1_BANKS)"
