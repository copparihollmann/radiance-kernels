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

MU_CFLAGS += -isystem $(LLVM_MUON)/include/c++/v1
MU_CFLAGS += -ffp-contract=off -fno-vectorize -fno-slp-vectorize
HOST_CXXFLAGS += -O3
MU_NM = $(LLVM_MUON)/bin/llvm-nm

STREAMCLUSTER_GENERATED = generated/config.h generated/points.bin \
	generated/coordinates.bin generated/center_table.bin \
	generated/work_expected.bin generated/switch_expected.bin

# The data is partitioned for a cluster count, so it is regenerated whenever
# the preset changes -- otherwise the HPC_SM_COUNT static_assert fires.
data.S $(STREAMCLUSTER_GENERATED) &: streamcluster.py oracle.py $(HPC_STAMP)
	python3 streamcluster.py --preset $(PRESET) --warps-per-core $(OCC) --out runs/default --prepare-only

data.mu.o: data.S $(STREAMCLUSTER_GENERATED)
	$(MU_CXX) $(MU_CFLAGS) -x assembler-with-cpp -c $< -o $@

$(addsuffix .mu.o,$(basename $(MU_SRCS))): generated/config.h kernel_impl.hpp

generated/symbols.h: kernel.radiance.elf generated/config.h emit_symbols.py
	python3 emit_symbols.py $(MU_NM) $< $@

host.host.o: generated/config.h generated/symbols.h

.PHONY: test clean-generated
test:
	python3 -m unittest -v test_streamcluster.py

clean-generated:
	rm -rf generated data.S
