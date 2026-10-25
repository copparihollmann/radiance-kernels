PROJECT = kernel
MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp
MU_BIN_OBJS = data.mu.o

include ../common.mk
include ../addr_hash_stamp.mk

# Deliberately scalar: no vector ISA, VPU, Gemmini, or MX headers/instructions.
MU_CFLAGS += -isystem $(LLVM_MUON)/include/c++/v1
MU_CFLAGS += -fno-vectorize -fno-slp-vectorize
HOST_CXXFLAGS += -O3
MU_NM = $(LLVM_MUON)/bin/llvm-nm

BTREE_GENERATED = generated/config.h generated/nodes.bin \
	generated/records.bin generated/point_queries.bin \
	generated/range_queries.bin generated/point_expected.bin \
	generated/range_expected.bin

data.S $(BTREE_GENERATED) &: run.py
	python3 run.py --out runs/default --prepare-only

data.mu.o: data.S $(BTREE_GENERATED)
	$(MU_CXX) $(MU_CFLAGS) -x assembler-with-cpp -c $< -o $@

kernel.mu.o: generated/config.h wire_format.hpp

generated/symbols.h: kernel.radiance.elf generated/config.h emit_symbols.py
	python3 emit_symbols.py $(MU_NM) $< $@

host.host.o: generated/config.h generated/symbols.h wire_format.hpp

.PHONY: test clean-generated
test:
	python3 -m unittest -v test_run.py

clean-generated:
	rm -rf generated data.S
