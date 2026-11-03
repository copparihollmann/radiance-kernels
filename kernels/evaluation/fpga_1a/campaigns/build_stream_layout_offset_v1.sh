#!/usr/bin/env bash
# Build the STREAM array-layout offset experiment: three layouts x two
# operations x 2 SM, in both loaders.  BUILDS AND SOFTWARE GATES ONLY.
# No FPGA is touched, no FireSim runner is invoked, no hardware job is launched.
#
# WHAT IS BEING TESTED
# --------------------
# On 2SM_DRAM_v2 the FASED rankPower counters put STREAM's row-conflict rate
# (numACT / (numCASR+numCASW)) at 0.52-0.65, against ~0.045 for the spatter
# kernels and an ideal of ~0.004 for a dense stream through an 8 KB page.  The
# diagnosis on record is a LAYOUT artifact, and it is confirmed by reading the
# linked symbols out of the deployed device ELF
# (runs/hpc-profile-guard-builds-v1/stream-*/2sm/kernel.radiance.elf):
#
#     stream array 0  client 0x10004000   slice-local 0x04001000   bank 0 rank 0
#     stream array 1  client 0x10404000   slice-local 0x04101000   bank 0 rank 0
#     stream output   client 0x10805140   slice-local 0x04201440   bank 0 rank 0
#
# The two input arrays are exactly 4 MiB apart in client space, which the
# 4-slice GPU address hash turns into exactly 1 MiB apart in slice-local space,
# and 1 MiB is an exact multiple of the 256 KiB bank+rank period of the DDR3
# map.  They are in the same bank and the same rank at EVERY element index.
#
# THE FIX UNDER TEST, AND WHY THE OBVIOUS OFFSET IS THE WRONG SIZE
# ----------------------------------------------------------------
# The GPU address hash sends consecutive 32 B units round-robin across 4 L2
# slices and keeps each slice's units dense, so a client-space shift of N bytes
# is only an N/4 shift in the slice-local space the DDR3 map decodes.  The
# "shift one array base by 8 KB" suggestion is therefore a 2 KiB slice-local
# shift -- a quarter of a bank -- and it does NOT separate the banks.  The
# offsets here are chosen in slice-local terms and then multiplied by 4:
#
#   baseline        offset      0 B  ->      0 B slice-local  (control)
#   bank32k         offset  32 KiB  ->   8 KiB slice-local  = +1 bank per array
#   bankrank288k    offset 288 KiB  ->  72 KiB slice-local  = +1 bank AND +1 rank
#
# Array k of the three is pushed k*offset, so every pair is separated.
#
# WHAT MUST NOT CHANGE
# --------------------
# Everything else is held at the accepted 2SM_DRAM_v2 configuration, taken from
# runs/hpc-v2-memory-models-1a-v1/build-manifest.json (NOT from a description of
# it): 1048576 float32 elements, --loader direct-hashed for the FPGA artifacts,
# --warps-per-core 4 (the accepted value; it is 4, not 8), --robust-end-drain,
# --robust-spin-until-reset, --robust-post-spin 20000, --diagnostic-readback,
# --full-host-check, scalar 3.0, robust_serial_fence FALSE for both Copy and
# Triad.  Moving a base address cannot change arithmetic, so the gate below
# REFUSES to ship any build whose expected digest differs from the accepted one.
set -euo pipefail

root=/scratch/agustin/radiance-fpga-1a
repo=$root/worktrees/stream-layout-offset-v1
out=$root/runs/stream-layout-offset-v1/build
model=$root/runs/stream-layout-offset-v1/model-functional
evidence=$root/runs/stream-layout-offset-v1/evidence
toolchain=$root/toolchains/riscv-tools-prefix/riscv-tools
cyclotron=$root/src/radiance/cyclotron/target-native/release/cyclotron
timing=$root/src/radiance/cyclotron/config/timing
llvm_muon=$root/src/radiance-kernels/llvm/llvm-muon
muon_runtime=$root/src/radiance-kernels/lib/libmuonrt.a
build_lock=$root/locks/stream-layout-offset-v1-build.lock

# The accepted artifacts were profiled with a host built with this knob, so the
# new arms are built with it too; otherwise the FASED ROI window would not be
# bracketed by quiet guard cycles the way the hpc-v2 record's was.
profile_guard=1000000ULL

test -x "$llvm_muon/bin/clang++"
test -f "$muon_runtime"
test -d "$repo"

if [[ ! -e $repo/llvm/llvm-muon ]]; then ln -s "$llvm_muon" "$repo/llvm/llvm-muon"; fi
if [[ ! -e $repo/lib/libmuonrt.a ]]; then ln -s "$muon_runtime" "$repo/lib/libmuonrt.a"; fi
test "$(readlink -f "$repo/llvm/llvm-muon")" = "$(readlink -f "$llvm_muon")"
test "$(readlink -f "$repo/lib/libmuonrt.a")" = "$(readlink -f "$muon_runtime")"
export LLVM_MUON=$repo/llvm/llvm-muon

if [[ -n $(git -C "$repo" status --porcelain=v1) ]]; then
  echo "worktree is dirty before build" >&2; exit 1
fi
revision=$(git -C "$repo" rev-parse HEAD)
echo "build revision: $revision"

mkdir -p "$(dirname "$build_lock")" "$out" "$evidence"
exec 9>"$build_lock"
flock -x 9

cd "$repo"
python3 -I -c "
import sys, unittest
sys.path.insert(0, '$repo')
r = unittest.main(module=None, argv=['x', 'kernels.stream.test_run'], exit=False).result
sys.exit(0 if r.wasSuccessful() else 1)
"

# variant name -> client-space offset in bytes
declare -A OFFSET=(
  [baseline]=0
  [bank32k]=32768
  [bankrank288k]=294912
)

for loader in logical direct-hashed; do
  completion_args=(--robust-end-drain --robust-post-spin 20000 --diagnostic-readback)
  if [[ $loader == direct-hashed ]]; then
    completion_args+=(--robust-spin-until-reset)
  fi
  for operation in copy triad; do
    for variant in baseline bank32k bankrank288k; do
      dest=$out/$loader/stream-$operation-layout-$variant-2sm
      if [[ -e $dest/kernel.soc.elf ]]; then
        echo "refusing to overwrite existing ELF: $dest" >&2; exit 1
      fi
      mkdir -p "$dest"
      echo "=== $loader / stream-$operation / $variant (offset ${OFFSET[$variant]} B) ==="
      EXTRA_HOST_CXXFLAGS="-DNIGHTLY_PROFILE_GUARD_CYCLES=$profile_guard" \
      RISCV64_TOOLCHAIN_PATH="$toolchain" nice -n 15 \
        python3 -I "$repo/kernels/stream/run.py" \
          --kind "$operation" --elements 1048576 --sms 2 \
          --loader "$loader" --warps-per-core 4 \
          --array-offset-bytes "${OFFSET[$variant]}" \
          "${completion_args[@]}" --full-host-check \
          --out "$dest" >"$dest/run.stdout" 2>"$dest/run.stderr" || {
            echo "build failed: $dest; see $dest/run.stderr" >&2; exit 1; }
      cp -p "$repo/kernels/stream/data.S" "$dest/data.S"
      cp -p "$repo/kernels/stream/generated/config.h" "$dest/config.h"
      cp -p "$repo/kernels/stream/generated/symbols.h" "$dest/symbols.h"
      python3 -I "$repo/kernels/evaluation/fpga_1a/campaigns/stream_layout_bank_map.py" \
        --nm "$llvm_muon/bin/llvm-nm" --elf "$dest/kernel.radiance.elf" \
        --json "$evidence/$loader--stream-$operation-$variant.bank-map.json" >/dev/null
    done
  done
  python3 -I "$repo/kernels/evaluation/fpga_1a/assemble_manifest.py" \
    --root "$out/$loader" --family stream \
    --out "$out/$loader-manifest.json"
done

# Functional execution of the kernel's own full host check, in the Cyclotron
# functional model.  Software only; the logical-loader ELFs are the ones the
# model loads, exactly as scripts/build-and-gate-stream-array-padding-v1.sh did.
nice -n 15 python3 -I "$repo/kernels/evaluation/fpga_1a/model_campaign.py" \
  --build-manifest "$out/logical-manifest.json" \
  --cyclotron "$cyclotron" --timing-dir "$timing" \
  --out "$model" --sms 2 --families stream \
  --repetitions 1 --timeout 2000000000 --functional --keep-going

python3 -I "$repo/kernels/evaluation/fpga_1a/campaigns/gate_stream_layout_offset_v1.py" \
  --repo "$repo" --revision "$revision" --build "$out" --model "$model" \
  --evidence "$evidence" --out "$evidence/gate.json"

test -z "$(git -C "$repo" status --porcelain=v1)"
echo "PASS: STREAM layout-offset artifacts are built and gated"
