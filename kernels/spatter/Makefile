PROJECT = kernel
MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp
MU_BIN_OBJS = data.mu.o

include ../common.mk

# The installed Muon clang does not add its bundled libc++ headers by default.
MU_CFLAGS += -isystem $(LLVM_MUON)/include/c++/v1

# Readback hashes every output word; keep that verification path efficient.
HOST_CXXFLAGS += -O3
HOST_CXXFLAGS += $(EXTRA_HOST_CXXFLAGS)

MU_NM = $(LLVM_MUON)/bin/llvm-nm

# Match other radiance-kernels directories: plain `make` has a small default
# input. run.py prepares a selected benchmark before invoking this same Makefile.
data.S generated/config.h generated/pattern.bin generated/pattern_gather.bin generated/pattern_scatter.bin generated/source.bin generated/group_offsets.bin generated/group_tasks.bin &: run.py plan.py smoke.json
	python3 run.py --suite smoke.json --case 0 --out runs/default --prepare-only

data.mu.o: data.S generated/pattern.bin generated/pattern_gather.bin generated/pattern_scatter.bin generated/source.bin generated/group_offsets.bin generated/group_tasks.bin
	$(MU_CXX) $(MU_CFLAGS) -x assembler-with-cpp -c $< -o $@

kernel.mu.o: generated/config.h spatter_ops.hpp

generated/symbols.h: kernel.radiance.elf generated/config.h emit_symbols.py
	python3 emit_symbols.py $(MU_NM) $< $@

host.host.o: generated/config.h generated/symbols.h

.PHONY: clean-generated
clean-generated:
	rm -rf generated data.S
