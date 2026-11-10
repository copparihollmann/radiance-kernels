PROJECT   = bigbird_blocksparse_attention
MU_SRCS   = kernel.cpp
HOST_SRCS = host.cpp
DEPS      = bigbird_blocksparse_data.h blocksparse_core.hpp
BLOBS     = data/bigbird_blocksparse.region1.bin data/bigbird_blocksparse.region2.bin data/bigbird_blocksparse.region3.bin
SMS      ?= 1
# One warp per core; 32 threads split the 1024 independent query positions.
OCC       = 1
KERNEL_MU_CFLAGS = -ffp-contract=off

include ../nightly.mk

CXX ?= g++
SOFTWARE_CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off
ORACLE = ../../bigbird_blocksparse_reference/bigbird_blocksparse_reference.hpp
TEST_BIN ?= build/software/bigbird_blocksparse_attention_test
GEN_BIN ?= build/software/bigbird_blocksparse_attention_gen

bigbird_blocksparse_data.h data/bigbird_blocksparse.region1.bin data/bigbird_blocksparse.region2.bin data/bigbird_blocksparse.region3.bin &: $(GEN_BIN) $(GEN_STAMP)
	$(GEN_BIN)

$(GEN_BIN): gen_data.cpp $(ORACLE) | build/software
	$(CXX) $(SOFTWARE_CXXFLAGS) gen_data.cpp -o $@

$(TEST_BIN): test_blocksparse.cpp blocksparse_core.hpp $(ORACLE) | build/software
	$(CXX) $(SOFTWARE_CXXFLAGS) test_blocksparse.cpp -o $@

build/software:
	mkdir -p $@

.PHONY: test software-data
test: $(TEST_BIN)
	$(TEST_BIN)

software-data: bigbird_blocksparse_data.h
