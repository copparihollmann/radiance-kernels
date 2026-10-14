PROJECT = kernel
MU_SRCS = kernel.cpp

include ../common.mk

data: gen_data.py
	python3 gen_data.py --out data --receipt fixture.json

kernel.mu.o: data
