PROJECT = kernel
MU_SRCS = kernel.cpp

include ../common.mk

data: gen_upstream_data.py
	$(TRANSFORMERS_PYTHON) gen_upstream_data.py --out data --receipt fixture.json

kernel.mu.o: data
