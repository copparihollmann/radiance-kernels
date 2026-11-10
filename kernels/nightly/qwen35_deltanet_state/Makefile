PROJECT   = qwen35_deltanet_state
MU_SRCS   = kernel.cpp
HOST_SRCS = host.cpp
DEPS      = qwen35_deltanet_state_data.h deltanet_state_core.hpp
BLOBS     = data/qwen35_deltanet_state.region1.bin data/qwen35_deltanet_state.region2.bin data/qwen35_deltanet_state.region3.bin
SMS      ?= 1
OCC       = 1
KERNEL_MU_CFLAGS = -ffp-contract=off

include ../nightly.mk

CXX ?= g++
SOFTWARE_CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off
ORACLE = ../../qwen35_deltanet_reference/qwen35_deltanet_reference.hpp
TEST_BIN = build/test_deltanet_state
GEN_BIN = build/gen_data

qwen35_deltanet_state_data.h data/qwen35_deltanet_state.region1.bin data/qwen35_deltanet_state.region2.bin data/qwen35_deltanet_state.region3.bin &: $(GEN_BIN) $(GEN_STAMP)
	$(GEN_BIN)

$(GEN_BIN): gen_data.cpp $(ORACLE)
	mkdir -p build
	$(CXX) $(SOFTWARE_CXXFLAGS) gen_data.cpp -o $@

$(TEST_BIN): test_deltanet_state.cpp deltanet_state_core.hpp $(ORACLE)
	mkdir -p build
	$(CXX) $(SOFTWARE_CXXFLAGS) test_deltanet_state.cpp -o $@

.PHONY: test software-data
test: $(TEST_BIN)
	$(TEST_BIN)

software-data: qwen35_deltanet_state_data.h
