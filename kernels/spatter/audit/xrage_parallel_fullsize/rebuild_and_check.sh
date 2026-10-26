#!/usr/bin/env bash
set -euo pipefail

base_commit=8bf256a5406e749a6342522e398b468d56c1b5db
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$script_dir" rev-parse --show-toplevel)
root=${XRAGE_AUDIT_ROOT:-$repo/experiments/xrage-parallel-winner-fullsize-8bf256a-v1}
suite=${XRAGE_SUITE:-/scratch/agustin/radiance-fpga-1a/artifacts/xrage-inputs-v1/xrage-suite.json}
toolchain=${RISCV64_TOOLCHAIN_PATH:-/scratch/agustin/radiance-fpga-1a/toolchains/riscv-tools-prefix/riscv-tools}
muon=${LLVM_MUON:-/scratch/agustin/radiance-fpga-1a/src/radiance-kernels/llvm/llvm-muon}
runtime_archive=${MUON_RUNTIME_ARCHIVE:-/scratch/agustin/radiance-fpga-1a/src/radiance-kernels/lib/libmuonrt.a}
cyclotron=${CYCLOTRON:-/scratch/agustin/radiance-fpga-1a/src/radiance/cyclotron/target-native/release/cyclotron}
timing=${CYCLOTRON_TIMING:-/scratch/agustin/radiance-fpga-1a/src/radiance/cyclotron/config/timing}
asset_manifest=${FPGA_ASSET_MANIFEST:-/scratch/agustin/radiance-fpga-1a/runs/fpga-campaign-single-1a-v2/assets.json}
asset_root=${FPGA_ASSET_ROOT:-/scratch/agustin/radiance-fpga-1a/artifacts/nicolas-fpga-assets-v1}
readelf=${RISCV_READELF:-$toolchain/bin/riscv64-unknown-elf-readelf}

git -C "$repo" merge-base --is-ancestor "$base_commit" HEAD || {
  echo "audit branch must descend from exact source commit $base_commit" >&2
  exit 1
}
for protected in kernel.cpp spatter_ops.hpp plan.py host.cpp run.py \
                 parallel_scatter_winner.hpp; do
  git -C "$repo" diff --quiet "$base_commit" -- "kernels/spatter/$protected" || {
    echo "protected source differs from $base_commit: $protected" >&2
    exit 1
  }
done
[[ ! -e "$root" ]] || {
  echo "refusing existing audit output: $root" >&2
  exit 1
}
for path in "$suite" "$cyclotron" "$asset_manifest" "$readelf" \
            "$muon/bin/clang++" "$runtime_archive"; do
  [[ -f "$path" ]] || { echo "missing required input: $path" >&2; exit 1; }
done

mkdir -p "$root/tests" "$root/build/logical" "$root/build/direct-hashed" \
  "$root/cyclotron-functional" "$root/fpga-plan"
cp --preserve=mode,timestamps "$runtime_archive" "$repo/lib/libmuonrt.a"
python3 - "$root/toolchain.json" "$muon/bin/clang++" "$runtime_archive" \
  "$repo/lib/libmuonrt.a" <<'PY'
import hashlib, json, pathlib, sys
out, compiler, source, copied = map(pathlib.Path, sys.argv[1:])
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(source) == digest(copied)
out.write_text(json.dumps({
  "muon_compiler": str(compiler.resolve()),
  "muon_compiler_sha256": digest(compiler.resolve()),
  "runtime_archive_source": str(source.resolve()),
  "runtime_archive_copy": str(copied.resolve()),
  "runtime_archive_sha256": digest(copied),
}, indent=2) + "\n")
PY

python_log="$root/tests/python-tests.log"
python3 -m unittest -v \
  kernels.spatter.test_run \
  kernels.spatter.audit.xrage_parallel_fullsize.test_audit \
  >"$python_log" 2>&1
python_tests=$(grep -E '^Ran [0-9]+ tests in ' "$python_log" | awk '{print $2}')
[[ "$python_tests" == 26 ]] || { cat "$python_log" >&2; exit 1; }
python3 - "$root/tests/python-tests.json" "$python_log" <<'PY'
import hashlib, json, pathlib, sys
out, log = map(pathlib.Path, sys.argv[1:])
out.write_text(json.dumps({"status":"passed", "tests":26,
                           "log":str(log.resolve()),
                           "log_sha256":hashlib.sha256(log.read_bytes()).hexdigest()},
                          indent=2) + "\n")
PY

native="$root/tests/test_parallel_scatter_winner"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I kernels/spatter \
  kernels/spatter/test_parallel_scatter_winner.cpp -o "$native"
native_log="$root/tests/native-test.log"
"$native" "$(dirname "$suite")/xrage-scatter.u32le" >"$native_log" 2>&1
grep -F 'focused admissible-winner tests: PASS' "$native_log" >/dev/null
grep -F 'full pinned xRAGE9: transfers=6664304 written_destinations=1191878 repeated_transfers=5472426 PASS' "$native_log" >/dev/null
python3 - "$root/tests/native-test.json" "$native_log" <<'PY'
import hashlib, json, pathlib, sys
out, log = map(pathlib.Path, sys.argv[1:])
out.write_text(json.dumps({"status":"passed", "transfers":6664304,
                           "written_destinations":1191878,
                           "repeated_transfers":5472426,
                           "log":str(log.resolve()),
                           "log_sha256":hashlib.sha256(log.read_bytes()).hexdigest()},
                          indent=2) + "\n")
PY

for loader in logical direct-hashed; do
  for sms in 1 2 4; do
    name="xrage-parallel-winner-post20k--scatter-parallel--${sms}sm"
    out="$root/build/$loader/$name"
    completion=(--robust-end-drain --robust-serial-fence --robust-drain-passes 3
                --robust-post-spin 20000 --diagnostic-readback)
    if [[ "$loader" == direct-hashed ]]; then
      completion+=(--robust-spin-until-reset)
    fi
    RISCV64_TOOLCHAIN_PATH="$toolchain" LLVM_MUON="$muon" \
      python3 kernels/spatter/run.py \
      --suite "$suite" --case 1 --sms "$sms" --loader "$loader" \
      --collision-policy parallel --warps-per-core 8 "${completion[@]}" \
      --build-only --out "$out"
    cp --preserve=mode,timestamps kernels/spatter/kernel.radiance.elf \
      "$out/device.radiance.elf"
    python3 "$script_dir/audit.py" attach-device \
      --result "$out/result.json" --device "$out/device.radiance.elf"
  done
  python3 kernels/evaluation/fpga_1a/assemble_manifest.py \
    --root "$root/build/$loader" --family xrage \
    --out "$root/build/$loader-manifest.json"
done

available_before=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
/usr/bin/time -v -o "$root/cyclotron-functional/resource-time.txt" \
  python3 kernels/evaluation/fpga_1a/model_campaign.py \
    --build-manifest "$root/build/logical-manifest.json" \
    --cyclotron "$cyclotron" --timing-dir "$timing" \
    --out "$root/cyclotron-functional" --sms 1 2 4 --families xrage \
    --functional --repetitions 1 --timeout 2000000000 --keep-going
available_after=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
max_rss=$(awk -F: '/Maximum resident set size/ {gsub(/^[[:space:]]+/, "", $2); print $2}' \
  "$root/cyclotron-functional/resource-time.txt")

mkdir -p "$root/cyclotron-exact"
exact_checker="$root/tests/check_parallel_scatter_dump"
stop_library="$root/tests/libstop_after_gmem.so"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I kernels/spatter \
  "$script_dir/check_parallel_scatter_dump.cpp" -o "$exact_checker"
gcc -std=c11 -O2 -Wall -Wextra -Werror -shared -fPIC \
  "$script_dir/stop_after_gmem.c" -ldl -o "$stop_library"
/usr/bin/time -v -o "$root/cyclotron-exact/resource-time.txt" \
  python3 "$script_dir/run_exact_cyclotron.py" \
    --coverage-root "$root/cyclotron-functional" \
    --out "$root/cyclotron-exact" --cyclotron "$cyclotron" \
    --stop-library "$stop_library" --checker "$exact_checker" \
    --pattern "$(dirname "$suite")/xrage-scatter.u32le"
exact_max_rss=$(awk -F: '/Maximum resident set size/ {gsub(/^[[:space:]]+/, "", $2); print $2}' \
  "$root/cyclotron-exact/resource-time.txt")
available_exact_after=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
python3 - "$root/resource.json" "$available_before" "$available_after" \
  "$max_rss" "$available_exact_after" "$exact_max_rss" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
  "mem_available_kib_before": int(sys.argv[2]),
  "mem_available_kib_after_coverage": int(sys.argv[3]),
  "coverage_maximum_resident_kib": int(sys.argv[4]),
  "mem_available_kib_after_exact": int(sys.argv[5]),
  "exact_maximum_resident_kib": int(sys.argv[6]),
  "resource_issue": None,
  "execution_policy": "sequential Cyclotron rows; no FPGA process launched"
}, indent=2) + "\n")
PY

python3 kernels/evaluation/fpga_1a/campaign.py plan \
  --build-manifest "$root/build/direct-hashed-manifest.json" \
  --asset-manifest "$asset_manifest" --assets "$asset_root" \
  --out "$root/fpga-plan" --warmups 0 --repetitions 1 \
  --allow-incomplete-build

python3 "$script_dir/audit.py" collect \
  --repo "$repo" --root "$root" --suite "$suite" --asset-root "$asset_root" \
  --readelf "$readelf" --out "$script_dir/evidence.json"

echo "audit complete; no FPGA command was launched"
