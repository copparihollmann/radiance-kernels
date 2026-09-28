PROJECT = kernel
MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp
MU_BIN_OBJS = data.mu.o

include ../common.mk

MU_CFLAGS += -isystem $(LLVM_MUON)/include/c++/v1
HOST_CXXFLAGS += -O3
MU_NM = $(LLVM_MUON)/bin/llvm-nm

data.S generated/config.h generated/a.bin generated/b.bin generated/c.bin &: run.py
	python3 run.py --kind copy --elements 256 --out runs/default --prepare-only

data.mu.o: data.S generated/a.bin generated/b.bin generated/c.bin
	$(MU_CXX) $(MU_CFLAGS) -x assembler-with-cpp -c $< -o $@

kernel.mu.o: generated/config.h

generated/symbols.h: kernel.radiance.elf generated/config.h emit_symbols.py
	python3 emit_symbols.py $(MU_NM) $< $@

host.host.o: generated/config.h generated/symbols.h

.PHONY: clean-generated
clean-generated:
	rm -rf generated data.S
