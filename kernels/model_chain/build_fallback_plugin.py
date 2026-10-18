#!/usr/bin/env python3
"""Build a pinned pure-FP32 plugin; no model graph or reference data is linked."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
SOURCES = ('fallback_abi.hpp','fallback_runtime.cpp','pipeline_math.hpp',
           'pipeline_smolvla.hpp','pipeline_int8.hpp','pipeline_int4.hpp')


def build(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'fallback.so').exists():
        raise FileExistsError('use a new immutable plugin directory')
    hashes = {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in SOURCES}
    digest = hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    source = out/'plugin.cpp'
    source.write_text('''#include "fallback_abi.hpp"
extern "C" const char* rk_fallback_sources_v1() {return "'''+digest+'''";}
extern "C" int rk_fallback_apply_v1(unsigned char* ram,const uint32_t* words,
                                    uint32_t context,uint32_t tid,uint32_t threads) {
  model_chain::FallbackDescriptor descriptor{};
  for(unsigned i=0;i<32;++i)descriptor.word[i]=words[i];
  return model_chain::fallback_dispatch(&descriptor,context,tid,threads,(uintptr_t)ram)?0:1;
}
extern "C" int rk_fallback_batch_v1(unsigned char* ram,const uint32_t* words,
    uint32_t context,const uint32_t* tids,uint32_t lanes,uint32_t threads,uint32_t parallel) {
  int failures=0;
  #pragma omp parallel for schedule(static) num_threads(8) if(parallel && lanes>1) reduction(|:failures)
  for(uint32_t i=0;i<lanes;++i)
    failures |= rk_fallback_apply_v1(ram,words,context,tids[i],threads);
  return failures;
}
''')
    command=['g++','-O3','-std=c++17','-ffp-contract=off','-fno-fast-math','-fopenmp','-shared','-fPIC',
             '-I'+str(HERE),str(source),'-o',str(out/'fallback.so')]
    with (out/'build.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=dict(command=command,source_sha256=hashes,combined_source_sha256=digest,
                plugin_sha256=hashlib.sha256((out/'fallback.so').read_bytes()).hexdigest(),
                scope='only existing pure FP32 fallback headers, no reference tensors or graph execution')
    (out/'build.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.out.resolve()),indent=2))
