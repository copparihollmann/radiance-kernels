# memlat: dependent-load latency microbenchmark (see memlat.h).
#
# OCC=4 is deliberate and load bearing: memlat gives every background warp its own disjoint slice
# of the source region (MEMLAT_BG_SLICES), so MU_NUM_CORES * OCC must not exceed that slice count.
# kernel.cpp static_asserts it.
PROJECT  = memlat
MU_SRCS  = kernel.cpp
HOST_SRCS = host.cpp
DEPS     = memlat.h
OCC     ?= 4
include ../nightly.mk
