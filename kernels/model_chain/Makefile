PROJECT = model_chain
MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

include ../common.mk

data: gen_data.py
	python3 gen_data.py

kernel.mu.o: data pipeline_math.hpp

.PHONY: check-layout
check-layout: kernel.radiance.elf
	python3 check_layout.py $< --nm $(LLVM_MUON)/bin/llvm-nm

kernel.soc.elf: check-layout
