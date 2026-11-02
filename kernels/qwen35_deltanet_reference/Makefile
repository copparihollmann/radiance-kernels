CXX ?= g++
CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off
TEST_BIN := build/test_qwen35_deltanet_reference

.PHONY: all test clean

all: test

$(TEST_BIN): qwen35_deltanet_reference.hpp test_qwen35_deltanet_reference.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) test_qwen35_deltanet_reference.cpp -o $@

test: $(TEST_BIN)
	$(TEST_BIN)

clean:
	rm -rf build
