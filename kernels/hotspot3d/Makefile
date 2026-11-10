PROJECT = kernel
MU_SRCS = \
	kernel.cpp \
	kernel_ilp2.cpp \
	kernel_ilp4.cpp
HOST_SRCS = host.cpp
MU_BIN_OBJS = data.mu.o

include ../common.mk
include ../hpc.mk
include ../addr_hash_stamp.mk

# This port is intentionally scalar.  In particular, do not enable the VPU,
# vector ISA, Gemmini, or MX paths.
MU_CFLAGS += -isystem $(LLVM_MUON)/include/c++/v1
MU_CFLAGS += -fno-vectorize -fno-slp-vectorize
HOST_CXXFLAGS += -O3
MU_NM = $(LLVM_MUON)/bin/llvm-nm

HOTSPOT3D_GENERATED = generated/config.h generated/power.bin \
	generated/initial.bin generated/expected.bin generated/model_inputs.bin

# The data is partitioned for a cluster count, so it is regenerated whenever
# the preset changes -- otherwise the HPC_SM_COUNT static_assert fires.
data.S $(HOTSPOT3D_GENERATED) &: run.py $(HPC_STAMP)
	python3 run.py --profile canary --preset $(PRESET) --warps-per-core $(OCC) --out runs/default --prepare-only

data.mu.o: data.S $(HOTSPOT3D_GENERATED)
	$(MU_CXX) $(MU_CFLAGS) -x assembler-with-cpp -c $< -o $@

$(addsuffix .mu.o,$(basename $(MU_SRCS))): generated/config.h kernel_impl.hpp

generated/symbols.h: kernel.radiance.elf generated/config.h emit_symbols.py
	python3 emit_symbols.py $(MU_NM) $< $@

host.host.o: generated/config.h generated/symbols.h

.PHONY: test clean-generated
test:
	python3 -m unittest -v test_run.py

clean-generated:
	rm -rf generated data.S runs
