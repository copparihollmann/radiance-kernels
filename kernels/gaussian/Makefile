PROJECT = kernel
MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp
MU_BIN_OBJS = data.mu.o

include ../common.mk
include ../addr_hash_stamp.mk

# Deliberately scalar: do not add a VPU, vector, Gemmini, or MX path.
MU_CFLAGS += -isystem $(LLVM_MUON)/include/c++/v1
MU_CFLAGS += -fno-vectorize -fno-slp-vectorize
HOST_CXXFLAGS += -O3
MU_NM = $(LLVM_MUON)/bin/llvm-nm

GAUSSIAN_GENERATED = generated/config.h generated/original_a.bin \
	generated/original_b.bin generated/expected_a.bin \
	generated/expected_b.bin generated/expected_m.bin \
	generated/expected_solution.bin \
	generated/model_a.bin generated/model_b.bin generated/model_m.bin

data.S $(GAUSSIAN_GENERATED) &: run.py
	python3 run.py --profile canary --sms 1 --out runs/default --prepare-only

data.mu.o: data.S $(GAUSSIAN_GENERATED)
	$(MU_CXX) $(MU_CFLAGS) -x assembler-with-cpp -c $< -o $@

kernel.mu.o: generated/config.h

generated/symbols.h: kernel.radiance.elf generated/config.h emit_symbols.py
	python3 emit_symbols.py $(MU_NM) $< $@

host.host.o: generated/config.h generated/symbols.h actual_checker.h \
	softfloat_f32.h

.PHONY: test clean-generated
test:
	python3 -m unittest -v test_run.py

clean-generated:
	rm -rf generated data.S runs
