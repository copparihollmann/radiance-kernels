PROJECT = kernel
MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp
MU_BIN_OBJS = data.mu.o

include ../common.mk
include ../addr_hash_stamp.mk

# Preserve the pinned scalar FP32 expression tree.  This port deliberately
# emits neither vector ISA nor VPU/Gemmini/MX operations.
MU_CFLAGS += -isystem $(LLVM_MUON)/include/c++/v1
MU_CFLAGS += -fno-fast-math -ffp-contract=off -fno-vectorize -fno-slp-vectorize
HOST_CXXFLAGS += -O3 -fno-fast-math -ffp-contract=off -fno-tree-vectorize
MU_NM = $(LLVM_MUON)/bin/llvm-nm

LBM_GENERATED = generated/config.h generated/input.bin \
	generated/output_initial.bin generated/expected_populations.bin \
	generated/gold_velocity.bin

data.S $(LBM_GENERATED) &: lbm.py
	python3 lbm.py --sms 1 --out runs/default --prepare-only

data.mu.o: data.S $(LBM_GENERATED)
	$(MU_CXX) $(MU_CFLAGS) -x assembler-with-cpp -c $< -o $@

kernel.mu.o: generated/config.h

generated/symbols.h: kernel.radiance.elf generated/config.h emit_symbols.py
	python3 emit_symbols.py $(MU_NM) $< $@

host.host.o: generated/config.h generated/symbols.h

.PHONY: test audit clean-generated
test:
	python3 -m unittest -v test_lbm.py

audit:
	python3 partition_audit.py --sms 1
	python3 partition_audit.py --sms 2
	python3 partition_audit.py --sms 4

clean-generated:
	rm -rf generated data.S runs
