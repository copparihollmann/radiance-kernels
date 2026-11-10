PROJECT   = minicpm3_mla
MU_SRCS   = kernel.cpp
HOST_SRCS = host.cpp
DEPS      = minicpm3_mla_data.h mla_component.hpp
BLOBS     = data/minicpm3_mla.region1.bin data/minicpm3_mla.region2.bin data/minicpm3_mla.region3.bin
SMS       = 1
OCC       = 2
KERNEL_MU_CFLAGS = -ffp-contract=off

include ../nightly.mk

CXX ?= g++
SOFTWARE_CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off
REFERENCE_DIR = ../../minicpm3_mla_reference
REFERENCE = $(REFERENCE_DIR)/minicpm3_mla_reference.py
REFERENCE_TEST = $(REFERENCE_DIR)/test_minicpm3_mla_reference.py
TEST_BIN = build/test_mla_component

minicpm3_mla_data.h data/minicpm3_mla.region1.bin data/minicpm3_mla.region2.bin data/minicpm3_mla.region3.bin &: gen_data.py $(REFERENCE) $(GEN_STAMP)
	python3 gen_data.py

$(TEST_BIN): test_mla_component.cpp mla_component.hpp minicpm3_mla_data.h $(BLOBS)
	$(CXX) $(SOFTWARE_CXXFLAGS) test_mla_component.cpp -o $@

.PHONY: test reference-test software-data
reference-test:
	python3 $(REFERENCE_TEST)

test: reference-test $(TEST_BIN)
	$(TEST_BIN)

software-data: minicpm3_mla_data.h
