PROJECT   = granite_moe_route
MU_SRCS   = kernel.cpp
HOST_SRCS = host.cpp
DEPS      = granite_moe_route_data.h
BLOBS     = data/granite_moe_route.region1.bin data/granite_moe_route.region2.bin data/granite_moe_route.region3.bin
SMS      ?= 1
include ../nightly.mk

granite_moe_route_data.h data/granite_moe_route.region1.bin data/granite_moe_route.region2.bin data/granite_moe_route.region3.bin &: gen_data.py $(GEN_STAMP)
	python3 gen_data.py $(GEN_ARGS)
