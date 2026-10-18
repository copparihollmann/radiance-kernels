#!/usr/bin/env python3
"""Relink retained stage objects to the current C API; add upstream SmolVLA checks.

This executes generated stage C++ on the host. ISA/HBM validation is a separate job.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--task',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
 a=p.parse_args();root=a.task.resolve();out=root/'artifacts/e2e-hbm-audit/native';out.mkdir(exist_ok=True)
 current=root/'artifacts/e2e-hbm-audit/libgemmini-radiance-current.a'
 if not current.exists():current.write_bytes(a.archive.read_bytes())
 assert sha(current)==sha(a.archive), 'retained C API archive differs from supplied archive'
 jobs=[]
 for name in ['tinyllama','deepseek_r1_distill_qwen_1_5b','gemma_2_2b_it','smolvla_base']:
  target=(root/'.tmp/smolvla-mx-2g'/name if name=='smolvla_base' else root/'.tmp/mx-full-canonical'/name)
  native=target/('native' if name=='smolvla_base' else 'native-check');dest=out/name;dest.mkdir(exist_ok=True)
  m=json.loads((root/'artifacts/models'/name/'manifest.json').read_text())
  with tarfile.open(root/'artifacts/models'/name/'generated-sources.tar.gz','r:gz') as ar:
   files={Path(x.name).name:x for x in ar.getmembers() if x.isfile()}
   for source in m['device_source_files']:assert sha(target/source)==hashlib.sha256(ar.extractfile(files[source]).read()).hexdigest(),source
  objects=[native/'model_data.o',*sorted(native.glob('stage_chunk_*.o')),*sorted(native.glob('verify_chunk_*.o')),native/'mx_vpu_runtime.o']
  if name=='smolvla_base':
   ref=root/'artifacts/e2e-hbm-audit/smolvla-reference-v2';r=json.loads((ref/'manifest.json').read_text());assert not r['missing_modules']
   checks={x['index'] for x in r['tensors']};flags=','.join('true' if i in checks else 'false' for i in range(m['stages']))
   helper='''
static unsigned boundary_bad=0,boundary_checked=0;
static const bool boundary_flags[] = {FLAGS};
static void check_boundary(unsigned index,const unsigned char* data,unsigned bytes) {
  if (!boundary_flags[index]) return;
  char name[4096]; snprintf(name,sizeof name,"REFERENCE/%04u.bin",index);
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
'''.replace('FLAGS',flags).replace('REFERENCE',str(ref))
   source=(native/'main.cpp').read_text().replace('int main(int argc, char** argv)',helper+'\nint main(int argc, char** argv)')
   source=source.replace('    if (trace_dir && stages[i].trace)', '    check_boundary(i,v_arena+stages[i].offset,stages[i].bytes);\n    if (trace_dir && stages[i].trace)')
   source=source.replace('  return failures ? 1 : 0;', '  printf("BOUNDARY_TOTAL checked=%u failed_stages=%u\\n",boundary_checked,boundary_bad);\n  return failures || boundary_bad ? 1 : 0;')
   (dest/'main.cpp').write_text(source)
   cmd=['g++','-std=c++17','-O2','-fopenmp','-ffp-contract=off','-c',str(dest/'main.cpp'),'-o',str(dest/'main.o')]
   subprocess.run(cmd,check=True);objects.append(dest/'main.o')
  else:objects.append(native/'main.o')
  binary=dest/'model-native';cmd=['g++','-fopenmp',*map(str,objects),str(current),'-o',str(binary)]
  subprocess.run(cmd,check=True)
  record=dict(model=name,command=[str(binary)],cwd=str(dest),scope='full generated stage C++ on CPU; not ISA or HBM timing',
      expected_stages=m['stages'],reference_generated_token_ids=m.get('reference_generated_token_ids'),format_comparison=m.get('format_comparison'),
      original_action_comparison=m.get('original_action_comparison'),binary_sha256=sha(binary),link=cmd,archive_sha256=sha(current),
      objects_sha256={str(o):sha(o) for o in objects})
  (dest/'build.json').write_text(json.dumps(record,indent=2)+'\n');jobs.append(record)
 (out/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
 print('Prepared four current-library full graph native checks.',flush=True)


if __name__=='__main__':main()
