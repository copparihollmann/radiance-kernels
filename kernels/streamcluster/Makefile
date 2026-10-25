PROJECT = kernel
MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp
MU_BIN_OBJS = data.mu.o

include ../common.mk
include ../addr_hash_stamp.mk

MU_CFLAGS += -isystem $(LLVM_MUON)/include/c++/v1
MU_CFLAGS += -ffp-contract=off -fno-vectorize -fno-slp-vectorize
HOST_CXXFLAGS += -O3
MU_NM = $(LLVM_MUON)/bin/llvm-nm

STREAMCLUSTER_GENERATED = generated/config.h generated/points.bin \
	generated/coordinates.bin generated/center_table.bin \
	generated/work_expected.bin generated/switch_expected.bin

data.S $(STREAMCLUSTER_GENERATED) &: streamcluster.py oracle.py
	python3 streamcluster.py --out runs/default --prepare-only

data.mu.o: data.S $(STREAMCLUSTER_GENERATED)
	$(MU_CXX) $(MU_CFLAGS) -x assembler-with-cpp -c $< -o $@

kernel.mu.o: generated/config.h

generated/symbols.h: kernel.radiance.elf generated/config.h emit_symbols.py
	python3 emit_symbols.py $(MU_NM) $< $@

host.host.o: generated/config.h generated/symbols.h

.PHONY: test clean-generated
test:
	python3 -m unittest -v test_streamcluster.py

clean-generated:
	rm -rf generated data.S
