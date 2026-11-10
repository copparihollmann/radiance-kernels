PROJECT   = mamba_selective_state
MU_SRCS   = kernel.cpp
HOST_SRCS = host.cpp
DEPS      = mamba_selective_state_data.h selective_state_core.hpp
BLOBS     = data/mamba_selective_state.region1.bin data/mamba_selective_state.region2.bin data/mamba_selective_state.region3.bin
SMS      ?= 1
# Two cores x one 16-lane warp provide 32 threads; each thread scans two of the
# 64 independent channels.  OCC=1 maximizes the recurrence's register budget.
OCC      = 1
KERNEL_MU_CFLAGS = -ffp-contract=off

include ../nightly.mk

CXX ?= g++
SOFTWARE_CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off
ORACLE = ../../mamba_state_reference/mamba_state_reference.hpp
TEST_BIN ?= /tmp/radiance-mamba-selective-state-test
GEN_BIN ?= /tmp/radiance-mamba-selective-state-gen

mamba_selective_state_data.h data/mamba_selective_state.region1.bin data/mamba_selective_state.region2.bin data/mamba_selective_state.region3.bin &: $(GEN_BIN) $(GEN_STAMP)
	$(GEN_BIN)

$(GEN_BIN): gen_data.cpp $(ORACLE)
	$(CXX) $(SOFTWARE_CXXFLAGS) gen_data.cpp -o $@

$(TEST_BIN): test_selective_state.cpp selective_state_core.hpp $(ORACLE)
	$(CXX) $(SOFTWARE_CXXFLAGS) test_selective_state.cpp -o $@

.PHONY: test software-data
test: $(TEST_BIN)
	$(TEST_BIN)

software-data: mamba_selective_state_data.h
