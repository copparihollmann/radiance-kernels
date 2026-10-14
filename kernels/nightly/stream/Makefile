PROJECT   = stream
MU_SRCS   = kernel.cpp
HOST_SRCS = host.cpp
DEPS      = stream_data.h stream_ops.hpp
BLOBS     = data/stream.region1.bin data/stream.region2.bin data/stream.region3.bin
OCC       = 4
include ../nightly.mk

stream_data.h data/stream.region1.bin data/stream.region2.bin data/stream.region3.bin &: gen_data.py $(GEN_STAMP) $(RK_ROOT)/kernels/stream/run.py
	python3 gen_data.py $(GEN_ARGS)
