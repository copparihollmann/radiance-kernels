PROJECT   = spatter
MU_SRCS   = kernel.cpp
HOST_SRCS = host.cpp
DEPS      = spatter_data.h spatter_body.hpp
BLOBS     = data/spatter.region1.bin data/spatter.region2.bin data/spatter.region3.bin
OCC       = 4
override KERNEL_MU_CFLAGS += -I$(RK_ROOT)/kernels/spatter
GEN_ARGS ?= --suite $(RK_ROOT)/kernels/spatter/smoke.json --case 0
include ../nightly.mk

spatter_data.h data/spatter.region1.bin data/spatter.region2.bin data/spatter.region3.bin &: gen_data.py $(GEN_STAMP) $(RK_ROOT)/kernels/spatter/run.py $(RK_ROOT)/kernels/spatter/plan.py
	python3 gen_data.py $(GEN_ARGS)
