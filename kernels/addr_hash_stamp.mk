# The HPC kernels reuse one output path for hashed and unhashed SoC ELFs.
# Record the effective build settings so changing them rebuilds the fused ELF.
# Include after ../common.mk so every variant target is covered, not just the
# baseline one.
ADDR_HASH_STAMP := generated/address-hash.stamp
ADDR_HASH_SETTINGS := MU_ADDR_HASH=$(MU_ADDR_HASH) MU_ADDR_HASH_SIZE=$(MU_ADDR_HASH_SIZE) MU_ADDR_HASH_UNIT=$(MU_ADDR_HASH_UNIT) MU_ADDR_HASH_SLICES=$(MU_ADDR_HASH_SLICES)
$(shell mkdir -p generated; printf '%s\n' '$(ADDR_HASH_SETTINGS)' | cmp -s - $(ADDR_HASH_STAMP) || printf '%s\n' '$(ADDR_HASH_SETTINGS)' > $(ADDR_HASH_STAMP))

$(patsubst %.elf,%.hashed.elf,$(RADIANCE_TARGETS)) $(SOC_TARGETS): $(ADDR_HASH_STAMP)
