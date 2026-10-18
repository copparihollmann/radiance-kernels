#!/usr/bin/env python3
"""Execute generated SmolVLA stage C++ on the CPU for a fast math check.

This uses the same generated stage functions and checkpoint image as the
Radiance ELF. It is a software arithmetic check, not device or RTL execution.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from compile_decoder import symbol, verify_backend_sources, write_if_changed
from split_decoder_weights import image_segments, verify_image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "kernels/evaluation/llm"))
from stitch import build  # noqa: E402
from plan_buffers import plan  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(target: Path, extra_trace_stages: tuple[str, ...] = (),
             stage_references: Path | None = None,
             original_golden: Path | None = None) -> Path:
    manifest = json.loads((target / "manifest.json").read_text())
    verify_backend_sources(manifest)
    for name,digest in manifest.get('fallback_source_sha256',{}).items():
        if name not in ('pipeline_smolvla.hpp','pipeline_math.hpp') or sha256(HERE/name) != digest:
            raise ValueError('FP32 fallback source differs from the generated numerical policy')
    for name,digest in (manifest.get('fallback_call_source_sha256') or {}).items():
        if name not in ('fallback_abi.hpp','fallback_runtime.cpp','fallback_codegen.py') or sha256(HERE/name) != digest:
            raise ValueError('pure helper call source differs from the generated graph')
    graph = build("smolvla_base")
    storage = plan(graph)
    expected_validation = ("all_elements_vs_mx_vpu_format_reference"
                           if manifest.get("execution_backend") == "rakanic_mx_vpu"
                           else "all_elements_vs_upstream_policy")
    if (manifest["stages"] != len(graph.stages) or
            manifest["execution_schedule_sha256"] !=
            storage["execution_schedule_sha256"] or
            manifest["output_validation"] != expected_validation):
        raise ValueError("native run requires the full exact-golden ELF sources")
    images = []
    for key, digest in (("weight_image_manifest", "weight_image_sha256"),
                        ("input_image_manifest", "input_image_sha256")):
        image, paths = verify_image(Path(manifest[key]))
        if image["image_sha256"] != manifest[digest]:
            raise ValueError(f"{key} differs from ELF build")
        images.append((image, paths))
    native = target / "native"
    native.mkdir(exist_ok=True)
    stub = "#pragma once\n#define __global\n#define MU_NUM_THREADS 32\n#define MU_NUM_CORES 1\n"
    write_if_changed(native / "mu_intrinsics.h", stub)
    write_if_changed(native / "mu_schedule.h", "#pragma once\n")
    reference_header = ''
    if stage_references is not None:
        reference_manifest = json.loads((stage_references/'manifest.json').read_text())
        if reference_manifest['missing_modules'] or reference_manifest['stages'] != len(graph.stages):
            raise ValueError('stage references do not cover the complete pinned module schedule')
        final = reference_manifest['final']
        if (final['checkpoint_weight_sha256'] != manifest['checkpoint_sha256'] or
                final['output_sha256'] != manifest['golden_output_sha256'] or
                reference_manifest['projection_module'] != manifest.get('projection_module')):
            raise ValueError('stage references differ from the executable precision policy')
        entries = {entry['index']:entry for entry in reference_manifest['tensors']}
        if len(entries) != reference_manifest['checked_stages']:
            raise ValueError('duplicate or missing stage reference records')
        followed_by_bias = {s['reads'][0] for s in graph.stages if s['op'] == 'bias_add'}
        expected_boundaries = set()
        for index,stage in enumerate(graph.stages):
            attrs = stage['attrs']
            parameter = None
            if stage['op'] == 'bias_add' or (stage['op'] == 'linear' and stage['id'] not in followed_by_bias):
                parameter = attrs.get('parameter')
            elif stage['op'] in ('rmsnorm','layernorm'):
                parameter = attrs.get('parameter') or next((p for p in attrs.get('parameters',{}) if p.endswith('.weight')),None)
            if parameter and parameter.endswith(('.weight','.bias')):
                expected_boundaries.add(index)
        if set(entries) != expected_boundaries:
            raise ValueError('stage references do not cover all pinned module boundaries')
        for index, entry in entries.items():
            stage = graph.stages[index]
            if (entry['id'] != stage['id'] or entry['bytes'] != 4*math.prod(stage['shape']) or
                    sha256(stage_references/entry['path']) != entry['sha256'] or
                    entry['path'] != f'{index:04d}.bin'):
                raise ValueError(f'stage reference differs at {index}')
        flags = ','.join('true' if i in entries else 'false' for i in range(len(graph.stages)))
        reference_header += '''
static unsigned boundary_bad=0,boundary_checked=0;
static const bool boundary_flags[] = {FLAGS};
static void check_boundary(unsigned index,const unsigned char* data,unsigned bytes) {
  if (!boundary_flags[index]) return;
  char name[4096]; snprintf(name,sizeof name,"%s/%04u.bin",DIRECTORY,index);
  FILE* f=fopen(name,"rb"); if(!f){perror(name);exit(4);}
  unsigned bad=0; float maximum=0; float golden[4096];
  const float* actual=(const float*)data;
  for(unsigned offset=0;offset<bytes/4;) {
    unsigned n=bytes/4-offset; if(n>4096)n=4096;
    if(fread(golden,4,n,f)!=n){fprintf(stderr,"short stage reference %u\\n",index);exit(4);}
    for(unsigned j=0;j<n;++j) {
      const float v=actual[offset+j],g=golden[j],e=fabsf(v-g);
      if(!std::isfinite(v)||e>0.001f+0.001f*fabsf(g))++bad;
      if(e>maximum)maximum=e;
    }
    offset+=n;
  }
  if(fgetc(f)!=EOF){fprintf(stderr,"long stage reference %u\\n",index);exit(4);} fclose(f);
  ++boundary_checked; if(bad)++boundary_bad;
  printf("BOUNDARY stage=%u elements=%u bad=%u max_abs=%.9g\\n",index,bytes/4,bad,maximum);
  fflush(stdout);
}
'''.replace('FLAGS',flags).replace('DIRECTORY',json.dumps(str(stage_references.resolve())))
    if original_golden is not None:
        import numpy as np
        original = json.loads(original_golden.read_text())
        if manifest.get('original_golden_sha256') and sha256(original_golden) != manifest['original_golden_sha256']:
            raise ValueError('original golden differs from the device output check')
        if (original['checkpoint_weight_sha256'] != manifest['checkpoint_sha256'] or
                original['input_sha256'] != images[1][0]['source_input_sha256']):
            raise ValueError('original checkpoint reference uses different weights or inputs')
        values = np.asarray(original['output_values'],np.float32)
        if (values.shape != tuple(graph.tensors[graph.outputs[-1]]['shape']) or
                hashlib.sha256(values.tobytes()).hexdigest() != original['output_sha256']):
            raise ValueError('original checkpoint reference has invalid output shape or hash')
        reference_header += ('static const float original_actions[] = {'+
                             ','.join(f'{float(value):.9e}f' for value in values.ravel())+'};\n')
    preload = []
    for image, paths in images:
        for segment, path in zip(image_segments(image), paths):
            preload.append(
                f"  if (!map_image({segment['gpu_base_address']}ull, "
                f"{segment['image_size_bytes']}ull, "
                f"{json.dumps(str(path.resolve()))})) return 2;")
    copies = []
    for item in images[1][0]["parameters"]:
        offset = storage["allocation"][item["logical_name"]]["offset_bytes"]
        copies.append(
            f"  memcpy(v_arena + {offset}u, "
            f"(const void*){item['gpu_address']}ull, {item['size_bytes']}u);")
    names = ["stage_" + symbol(stage["id"]) for stage in graph.stages]
    unknown = set(extra_trace_stages) - {stage["id"] for stage in graph.stages}
    if unknown:
        raise ValueError(f"unknown trace stages: {sorted(unknown)}")
    prototypes = "\n".join(
        f"void {name}(void*, uint32_t, uint32_t, uint32_t);" for name in names)

    def trace_stage(stage: dict) -> bool:
        name = stage["id"]
        return (name in extra_trace_stages or
                name.endswith(("vision.final_norm", "connector", "state_proj",
                               "vlm.final_norm", "action_out")) or
                name in ("prefix.merge", "prefix.attention_mask",
                         "prefix.position_ids") or
                (name.startswith("vlm.layer") and name.endswith(
                    ("input_norm", "o_proj", "ffn_residual"))))

    entries = ",\n".join(
        f"  {{{json.dumps(stage['id'])}, {name}, "
        f"{storage['allocation'][stage['writes']]['offset_bytes']}u, "
        f"{4 * math.prod(stage['shape'])}u, "
        f"{'true' if trace_stage(stage) else 'false'}}}"
        for name, stage in zip(names, graph.stages))
    source = """#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
extern unsigned char v_arena[];
extern float v_golden_output[];
""" + prototypes + """
struct Stage { const char* name; void (*run)(void*, uint32_t, uint32_t, uint32_t);
               uint32_t offset; uint32_t bytes; bool trace; };
static const Stage stages[] = {
""" + entries + "\n};\n" + """
static bool map_image(uint64_t base, uint64_t size, const char* path) {
  int file = open(path, O_RDONLY);
  if (file < 0) { perror(path); return false; }
  void* memory = mmap((void*)base, size, PROT_READ,
                      MAP_PRIVATE | MAP_FIXED_NOREPLACE, file, 0);
  close(file);
  if (memory == MAP_FAILED || (uint64_t)memory != base) {
    perror("mmap image"); return false;
  }
  return true;
}
int main(int argc, char** argv) {
  const uint32_t total = sizeof(stages) / sizeof(stages[0]);
  const char* trace_dir = getenv("SMOLVLA_NATIVE_TRACE_DIR");
  uint32_t limit = argc > 1 ? (uint32_t)strtoul(argv[1], nullptr, 10) : total;
  if (limit == 0 || limit > total) return 2;
""" + "\n".join(preload) + "\n" + "\n".join(copies) + """
  for (uint32_t i = 0; i < limit; ++i) {
    #pragma omp parallel for schedule(static)
    for (uint32_t tid = 0; tid < 32; ++tid)
      stages[i].run(nullptr, tid, 32, 0);
    if (trace_dir && stages[i].trace) {
      char filename[4096];
      snprintf(filename, sizeof(filename), "%s/%04u.bin", trace_dir, i + 1);
      FILE* trace = fopen(filename, "wb");
      if (!trace || fwrite(v_arena + stages[i].offset, 1, stages[i].bytes,
                           trace) != stages[i].bytes) return 3;
      fclose(trace);
    }
    if ((i + 1) % 100 == 0 || i + 1 == limit) {
      fprintf(stderr, "completed %u/%u: %s\\n", i + 1, total, stages[i].name);
      fflush(stderr);
    }
  }
  const Stage& final = stages[limit - 1];
  if (argc > 2) {
    FILE* dump = fopen(argv[2], "wb");
    if (!dump || fwrite(v_arena + final.offset, 1, final.bytes, dump) != final.bytes)
      return 3;
    fclose(dump);
  }
  if (limit != total) return 0;
  const float* output = (const float*)(v_arena + final.offset);
  uint32_t failures = 0, first_failure = 0;
  float max_error = 0.0f;
  for (uint32_t i = 0; i < final.bytes / sizeof(float); ++i) {
    float error = fabsf(output[i] - v_golden_output[i]);
    if (!std::isfinite(output[i]) || error > 0.001f + 0.001f * fabsf(v_golden_output[i])) {
      if (!failures) first_failure = i;
      ++failures;
    }
    if (error > max_error) max_error = error;
  }
  printf("actions=%u failures=%u first_failure=%u max_abs_error=%.9g\\n",
         final.bytes / 4, failures, first_failure, max_error);
  return failures ? 1 : 0;
}
"""
    if reference_header:
        write_if_changed(native/'reference_checks.hpp',reference_header)
        source = source.replace('int main(int argc, char** argv)',
                                '#include "reference_checks.hpp"\nint main(int argc, char** argv)')
    extra_failures = []
    if stage_references is not None:
        source = source.replace('    if (trace_dir && stages[i].trace)',
                                '    check_boundary(i,v_arena+stages[i].offset,stages[i].bytes);\n'
                                '    if (trace_dir && stages[i].trace)')
        source = source.replace('  return failures ? 1 : 0;',
                                '  printf("BOUNDARY_TOTAL checked=%u failed_stages=%u\\n",boundary_checked,boundary_bad);\n'
                                '  return failures ? 1 : 0;')
        extra_failures.append('boundary_bad')
    if original_golden is not None:
        source = source.replace('  return failures ? 1 : 0;', '''
  unsigned original_bad=0; float original_max=0;
  for(unsigned i=0;i<final.bytes/4;++i) {
    const float e=fabsf(output[i]-original_actions[i]);
    if(!std::isfinite(output[i])||e>0.001f+0.001f*fabsf(original_actions[i]))++original_bad;
    if(e>original_max)original_max=e;
  }
  printf("ORIGINAL actions=%u failures=%u max_abs_error=%.9g\\n",final.bytes/4,original_bad,original_max);
  return failures ? 1 : 0;''')
        extra_failures.append('original_bad')
    if extra_failures:
        source = source.replace('  return failures ? 1 : 0;',
                                '  return failures || '+' || '.join(extra_failures)+' ? 1 : 0;')
    if manifest.get('replica_buffers'):
        from native_replicas import wrap_driver
        source = wrap_driver(source, preload, manifest['replica_buffers'],
                             boundaries=stage_references is not None)
    write_if_changed(native / "main.cpp", source)
    return native


def compile_native(target: Path, jobs: int = 2,
                   extra_trace_stages: tuple[str, ...] = (),
                   stage_references: Path | None = None,
                   original_golden: Path | None = None) -> Path:
    native = generate(target, extra_trace_stages, stage_references, original_golden)
    compiler = ["g++", "-std=c++17", "-O2", "-fopenmp", "-I", str(native),
                "-I", str(HERE)]
    sources = [target / "model_data.cpp", *sorted(target.glob("stage_chunk_*.cpp")),
               native / "main.cpp"]
    use_mx = json.loads((target/"manifest.json").read_text()).get("execution_backend") == "rakanic_mx_vpu"
    if json.loads((target/'manifest.json').read_text()).get('fallback_call_abi'):
        sources.append(target/'fallback_runtime.cpp')
    if use_mx:
        sources.append(target/"mx_vpu_runtime.cpp")
        compiler += ["-ffp-contract=off", "-I", os.environ["RADIANCE_LIBGEMMINI_ROOT"]]

    def compile_one(source: Path) -> Path:
        obj = native / (source.stem + ".o")
        dependencies = [source, native / "mu_intrinsics.h",
                        native / "mu_schedule.h", HERE / "pipeline_smolvla.hpp",
                        HERE / "pipeline_math.hpp"]
        if use_mx:
            dependencies.append(HERE/"mx_vpu/backend.hpp")
        if (target/'fallback_runtime.cpp').exists():
            dependencies.append(HERE/'fallback_abi.hpp')
        if source.name == 'main.cpp' and (stage_references is not None or original_golden is not None):
            dependencies.append(native/'reference_checks.hpp')
        if (not obj.exists() or obj.stat().st_mtime <
                max(path.stat().st_mtime for path in dependencies)):
            result = subprocess.run(compiler + ["-c", str(source), "-o", str(obj)],
                                    capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"{source}: {result.stdout}{result.stderr}")
        return obj

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        objects = list(executor.map(compile_one, sources))
    binary = native / "smolvla_native"
    subprocess.run(["g++", "-fopenmp", *map(str, objects),
                    *([os.environ["RADIANCE_LIBGEMMINI_ARCHIVE"]] if use_mx else []), "-o", str(binary)],
                   check=True)
    return binary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--stage-limit", type=int)
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stage-references",type=Path)
    parser.add_argument("--original-golden",type=Path)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--trace-stage", action="append", default=[],
                        help="also dump this stage output when --trace-dir is set")
    args = parser.parse_args()
    if args.stage_limit is not None and not 1 <= args.stage_limit <= 3673:
        parser.error("--stage-limit must be between 1 and 3673")
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    target = (args.generated_root / "smolvla_base").resolve()
    if args.trace_stage and not args.trace_dir:
        parser.error("--trace-stage requires --trace-dir")
    binary = compile_native(target, args.jobs, tuple(args.trace_stage),args.stage_references,args.original_golden)
    command = [str(binary)]
    if args.stage_limit is not None:
        command.append(str(args.stage_limit))
    if args.dump:
        if args.stage_limit is None:
            command.append("3673")
        command.append(str(args.dump.resolve()))
    if args.threads <= 0 or args.threads > 32:
        parser.error("--threads must be between 1 and 32")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        env["SMOLVLA_NATIVE_TRACE_DIR"] = str(args.trace_dir.resolve())
    log_path = target / ("native-run-full.log" if args.stage_limit is None
                         else f"native-run-stage{args.stage_limit}.log")
    with log_path.open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                env=env)
    build_manifest = json.loads((target / "manifest.json").read_text())
    log_text = log_path.read_text()
    model_counts = re.search(r"MX_VPU_MODEL commands=(\d+) mx=(\d+) vpu=(\d+) cycles=(\d+)", log_text)
    boundaries = re.search(r'BOUNDARY_TOTAL checked=(\d+) failed_stages=(\d+)',log_text)
    original = re.search(r'ORIGINAL actions=(\d+) failures=(\d+) max_abs_error=([^\s]+)',log_text)
    if args.stage_limit is None and ((args.stage_references and not boundaries) or (args.original_golden and not original)):
        raise ValueError('full native run did not report the required independent comparisons')
    match = re.search(
        r"actions=(\d+) failures=(\d+) first_failure=(\d+) max_abs_error=([\deE+.-]+)",
        log_text)
    if args.stage_limit is None and not match:
        raise ValueError("native full-action run did not report its output comparison")
    if args.stage_limit is None and build_manifest.get("execution_backend") == "rakanic_mx_vpu" and (
            model_counts is None or int(model_counts[2]) == 0 or int(model_counts[3]) == 0):
        raise ValueError("full MX+VPU graph did not execute both command families")
    record = {"model": "smolvla_base", "scope": "generated_stage_cpp_on_cpu",
              "stage_count": args.stage_limit or 3673,
              "host_threads": args.threads,
              "status": "passed" if result.returncode == 0 else "failed",
              "process_exit_code": result.returncode,
              "execution_backend": build_manifest.get("execution_backend", "muon"),
              "model_commands": ({key: int(value) for key, value in zip(
                  ("total", "mx", "vpu", "cycles_excluding_muon_work"), model_counts.groups())} if model_counts else None),
              "original_action_comparison": build_manifest.get("original_action_comparison"),
              "upstream_golden_compared": args.stage_limit is None and build_manifest.get("execution_backend", "muon") == "muon",
              "format_reference_compared": args.stage_limit is None and build_manifest.get("execution_backend") == "rakanic_mx_vpu",
              "golden_output_sha256": build_manifest["golden_output_sha256"],
              "golden_policy_precision": build_manifest["golden_policy_precision"],
              "comparison_tolerance": (build_manifest["output_check_tolerance"]
                                       if args.stage_limit is None else None),
              "output_elements": int(match.group(1)) if match else None,
              "failed_output_elements": int(match.group(2)) if match else None,
              "first_failed_element": int(match.group(3)) if match else None,
              "max_abs_error": float(match.group(4)) if match else None,
              "native_binary_sha256": sha256(binary),
              "generated_source_sha256": hashlib.sha256(
                  "".join(sha256(target / name) for name in build_manifest[
                      "device_source_files"]).encode()).hexdigest(),
              "device_elf_sha256": build_manifest.get("radiance_elf_sha256"),
              "device_elf_built": build_manifest.get("device_elf_built", False),
              "log_sha256": sha256(log_path),
              "log_path": str(log_path),
              "log_tail": log_text[-1200:],
              "rtl_execution": False, "device_execution": False,
              "performance_measurement": False}
    record['fallback_source_sha256'] = build_manifest.get('fallback_source_sha256')
    record['native_driver_sha256'] = sha256(target/'native/main.cpp')
    if build_manifest.get('execution_backend') == 'rakanic_mx_vpu':
        record['libgemmini_archive_sha256'] = sha256(Path(os.environ['RADIANCE_LIBGEMMINI_ARCHIVE']))
    if args.stage_references:
        record['module_boundaries'] = dict(checked=int(boundaries[1]),failed_stages=int(boundaries[2])) if boundaries else None
        record['stage_reference_manifest_sha256'] = sha256(args.stage_references/'manifest.json')
    if args.original_golden:
        record['actual_original_action_comparison'] = (dict(elements=int(original[1]),failed_elements=int(original[2]),
            passed=int(original[2]) == 0,max_abs_error=float(original[3]),rtol=1e-3,atol=1e-3) if original else None)
        record['original_golden_sha256'] = sha256(args.original_golden)
    if args.dump and args.dump.is_file():
        record["dump_sha256"] = sha256(args.dump)
    replica_failure = False
    if build_manifest.get('replica_buffers'):
        from native_replicas import validate_log
        count = build_manifest['num_clusters']
        qualified = validate_log(log_text, count, args.stage_limit or 3673,
                                 args.stage_references is not None,
                                 args.original_golden is not None,
                                 full=args.stage_limit is None)
        record['num_replicas'] = count
        record['total_stage_invocations'] = (args.stage_limit or 3673) * count
        record['replica_validation'] = qualified
        replica_failure = not qualified['passed']
        if replica_failure:
            record['status'] = 'failed'
        if args.stage_limit is None:
            outputs = [x['format_actions'][0] for x in qualified['replicas'] if len(x['format_actions']) == 1]
            record['output_elements'] = sum(int(x[0]) for x in outputs)
            record['failed_output_elements'] = sum(int(x[1]) for x in outputs)
            record['max_abs_error'] = max((float(x[2]) for x in outputs), default=None)
            if args.stage_references:
                records = [x['boundaries'][0] for x in qualified['replicas'] if len(x['boundaries']) == 1]
                record['module_boundaries'] = dict(checked=sum(int(x[0]) for x in records),
                                                 failed_stages=sum(int(x[1]) for x in records))
            if args.original_golden:
                records = [x['original_actions'][0] for x in qualified['replicas'] if len(x['original_actions']) == 1]
                record['actual_original_action_comparison'] = dict(
                    elements=sum(int(x[0]) for x in records),failed_elements=sum(int(x[1]) for x in records),
                    passed=bool(qualified['passed']),max_abs_error=max((float(x[2]) for x in records), default=None),
                    rtol=1e-3,atol=1e-3)
    destination = args.out or target / "native-result.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(f"SmolVLA native: {record['status']} ({record['stage_count']} stages)")
    if result.returncode or replica_failure:
        raise SystemExit(result.returncode or 1)


if __name__ == "__main__":
    main()
