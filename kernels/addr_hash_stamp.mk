# STREAM and Spatter reuse one output path for hashed and unhashed SoC ELFs.
# Record the effective build settings so changing them rebuilds the fused ELF.
ADDR_HASH_STAMP := generated/address-hash.stamp
ADDR_HASH_SETTINGS := MU_ADDR_HASH=$(MU_ADDR_HASH) MU_ADDR_HASH_SIZE=$(MU_ADDR_HASH_SIZE) MU_ADDR_HASH_UNIT=$(MU_ADDR_HASH_UNIT) MU_ADDR_HASH_SLICES=$(MU_ADDR_HASH_SLICES)
$(shell mkdir -p generated; printf '%s\n' '$(ADDR_HASH_SETTINGS)' | cmp -s - $(ADDR_HASH_STAMP) || printf '%s\n' '$(ADDR_HASH_SETTINGS)' > $(ADDR_HASH_STAMP))

$(PROJECT).radiance.hashed.elf $(PROJECT).soc.elf: $(ADDR_HASH_STAMP)
