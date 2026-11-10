CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic -Werror
TEST_BIN ?= /tmp/radiance-mamba-state-reference-test

.PHONY: all test clean

all: test

$(TEST_BIN): mamba_state_reference.hpp test_mamba_state_reference.cpp
	$(CXX) $(CXXFLAGS) test_mamba_state_reference.cpp -o $@

test: $(TEST_BIN)
	$(TEST_BIN)

clean:
	rm -f $(TEST_BIN)
