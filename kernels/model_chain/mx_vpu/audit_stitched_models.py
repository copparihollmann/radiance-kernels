#!/usr/bin/env python3
"""Check complete graph lineage and retained ELF/source bindings before execution."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile
import tomllib

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE.parents[1]/'evaluation/llm')]
from audit_full_models import check_decoder_lineage,check_smolvla_lineage
from compile_decoder import verify_backend_sources
from stitch import build,model_specs


def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for data in iter(lambda:f.read(1<<20),b''):h.update(data)
 return h.hexdigest()


def audit(directory):
 m=json.loads((directory/'manifest.json').read_text());name=m['model'];verify_backend_sources(m)
 assert m['stage_limit'] is None and m['execution_backend']=='rakanic_mx_vpu'
 assert digest(directory/'kernel.radiance.elf')==m['radiance_elf_sha256']
 if name=='smolvla_base':
  graph=build(name);lineage=check_smolvla_lineage(graph,m['execution_schedule_sha256'])
 else:
  assert m['layers']==model_specs()[name]['num_hidden_layers']
  graph=build(name,prefill=m['prefill_tokens'],decode_steps=m['decode_steps'],generation=m['generation'])
  lineage=check_decoder_lineage(graph,m['prefill_tokens'],m['decode_steps'])
 assert len(graph.stages)==m['stages']
 # Bind the actual launch images to every retained input/weight segment, not just
 # the ELF and source archive. Hash compact images in blocks without copying them.
 loads=tomllib.loads((directory/'cyclotron-hbm-config.toml').read_text())['sim']['gmem_load']
 segments=[]
 for kind in ('weight','input'):
  if m.get(kind+'_image_manifest'):
   metadata=json.loads(Path(m[kind+'_image_manifest']).read_text())
   assert metadata['image_sha256']==m[kind+'_image_sha256']
   segments.extend(metadata.get('segments',[metadata]))
 assert len(segments)==len(loads)
 images=[]
 for image in loads:
  segment=next(x for x in segments if x['gpu_base_address']==image['address'])
  path=Path(image['path'])
  assert path.stat().st_size==image['bytes']==segment['image_size_bytes']
  actual=digest(path);assert actual==segment['image_sha256'],str(path)
  images.append(dict(path=str(path),address=image['address'],bytes=image['bytes'],sha256=actual))
 with tarfile.open(directory/'generated-sources.tar.gz','r:gz') as archive:
  members={Path(x.name).name:x for x in archive.getmembers() if x.isfile()}
  sources={n:archive.extractfile(members[n]).read() for n in m['device_source_files']}
 prior=json.loads((directory/'native-result.json').read_text())
 if name=='smolvla_base':
  combined=hashlib.sha256(''.join(hashlib.sha256(sources[n]).hexdigest() for n in m['device_source_files']).encode()).hexdigest()
  assert combined==prior['generated_source_sha256']
 else:
  assert all(hashlib.sha256(sources[n]).hexdigest()==h for n,h in prior['device_source_files_sha256'].items())
 kernel=sources['kernel.cpp'].decode()
 scheduled=re.findall(r'\bmu_schedule\((stage_[A-Za-z0-9_]+)',kernel)
 if name=='smolvla_base':
  from compile_decoder import symbol
  expected=['stage_'+symbol(s['id']) for s in graph.stages]
 else:expected=['stage_'+str(i) for i in range(len(graph.stages))]
 assert scheduled==expected,(name,'stage order differs')
 # Every read must be an input or the output of an earlier stage.
 produced={s['writes']:i for i,s in enumerate(graph.stages)}
 for i,s in enumerate(graph.stages):
  for name_ in s['reads']:
   if name_ in produced:assert produced[name_]<i,(s['id'],name_)
 return dict(model=name,status='passed',stages=len(graph.stages),lineage=lineage,
  image_checks=images,scheduled_functions=len(scheduled),elf_sha256=m['radiance_elf_sha256'],source_archive_sha256=digest(directory/'generated-sources.tar.gz'),
  generated_source_sha256={n:hashlib.sha256(b).hexdigest() for n,b in sources.items()},
  format_comparison=m.get('format_comparison'),original_action_comparison=m.get('original_action_comparison'),
  device_execution=False,scope='structural graph and build binding audit')


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 results=[]
 for directory in sorted(a.models.iterdir()):
  if (directory/'manifest.json').exists():results.append(audit(directory))
 assert len(results)==4
 a.out.write_text(json.dumps(dict(status='passed',models=results),indent=2)+'\n')
 print('Graph lineage and exact source/ELF binding passed for all four full graphs.')


if __name__=='__main__':main()
