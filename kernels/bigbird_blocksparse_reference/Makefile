CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off
TEST_BIN ?= build/software/bigbird_blocksparse_reference_test

.PHONY: all test clean

all: test

$(TEST_BIN): bigbird_blocksparse_reference.hpp test_bigbird_blocksparse_reference.cpp | build/software
	$(CXX) $(CXXFLAGS) test_bigbird_blocksparse_reference.cpp -o $@

build/software:
	mkdir -p $@

test: $(TEST_BIN)
	$(TEST_BIN)

clean:
	rm -rf build
