PROJECT = kernel
MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp
MU_BIN_OBJS = data.mu.o

include ../common.mk
include ../addr_hash_stamp.mk

# This port is intentionally scalar.  Disabling contraction also makes the
# generated ordered-float32 oracle bit reproducible.
MU_CFLAGS += -isystem $(LLVM_MUON)/include/c++/v1
MU_CFLAGS += -ffp-contract=off -fno-vectorize -fno-slp-vectorize
HOST_CXXFLAGS += -O3
MU_NM = $(LLVM_MUON)/bin/llvm-nm

SPMV_GENERATED = generated/config.h generated/data.bin \
	generated/indices.bin generated/row_to_jds.bin generated/x.bin \
	generated/jds_ptr.bin generated/group_bounds.bin generated/expected.bin

data.S $(SPMV_GENERATED) &: run.py
	python3 run.py --fixture-rows 257 --fixture-seed 7 \
		--out runs/default --prepare-only

data.mu.o: data.S $(SPMV_GENERATED)
	$(MU_CXX) $(MU_CFLAGS) -x assembler-with-cpp -c $< -o $@

kernel.mu.o: generated/config.h

generated/symbols.h: kernel.radiance.elf generated/config.h emit_symbols.py
	python3 emit_symbols.py $(MU_NM) $< $@

host.host.o: generated/config.h generated/symbols.h

.PHONY: test clean-generated
test:
	python3 -m unittest -v test_run.py

clean-generated:
	rm -rf generated data.S
