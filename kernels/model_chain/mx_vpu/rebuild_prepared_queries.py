#!/usr/bin/env python3
"""Derive full decoder ELFs with immutable query-weight conversion prepared once.

Retain compact checkpoint storage, original graph sources, reference tensors and
the qualified runtime. Additional read-only lookup images fit existing holes in
the 2 GiB GPU region. This is a build, never an execution/qualification result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from mx_vpu.prepare_weights import direct, int8_lookup
from split_decoder_weights import STACK_BOTTOM, STACK_TOP, image_segments, verify_image, verify_image_placement


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def allocate(occupied, size):
    start = 1<<20
    size = (size+4095)//4096*4096
    for low,high in sorted(occupied)+[(1<<31,1<<31)]:
        if start+size <= low:
            occupied.append((start,start+size))
            return start
        start = max(start,((high+4095)//4096)*4096)
    raise ValueError('prepared query metadata does not fit the 2 GiB GPU region')


def prepare_images(manifest_path, elf, out, packing):
    image,paths = verify_image(manifest_path)
    segments = image_segments(image)
    # Keep the whole existing program arena available for new code, not merely
    # the previous ELF's occupied bytes. Inputs/checkpoints/stacks stay intact.
    following = min(s['gpu_base_address'] for s in segments if s['gpu_base_address'] > 0x10000000)
    occupied = [(0,1<<20),(0x10000000,following),(STACK_BOTTOM,STACK_TOP)]
    occupied += [(s['gpu_base_address'],s['gpu_base_address']+s['image_size_bytes']) for s in segments]
    records,bindings,patches = [],{},[]
    for param in image['parameters']:
        if not param['logical_name'].endswith('.q_proj'):continue
        if param['storage_dtype'] != 'int8_scaled':
            raise ValueError('prepared query build requires pinned INT8 column scales')
        k,n = param['shape']
        path = paths[param.get('segment_index',0)]
        with path.open('rb') as stream:
            stream.seek(param['offset_bytes']);raw=stream.read(param['size_bytes'])
        if hashlib.sha256(raw).hexdigest() != param['packed_sha256']:
            raise ValueError('selected query weight hash differs from its checkpoint image')
        codes = np.frombuffer(raw[:param['weight_size_bytes']],np.int8).reshape(k,n)
        scales = np.frombuffer(raw,np.dtype('<f4'),
                               count=param['scale_count'],offset=param['scale_offset_bytes']-param['offset_bytes'])
        if packing == 'overlay':
            converted,aux = direct(codes.astype(np.float32)*scales)
            patch_path=out/(param['logical_name']+'.mx-fp8.bin');patch_path.write_bytes(converted)
            original_codes=raw[:param['weight_size_bytes']]
            patches.append(dict(address=param['gpu_address'],bytes=len(converted),path=str(patch_path),
                expected_old_sha256=hashlib.sha256(original_codes).hexdigest(),
                expected_new_sha256=hashlib.sha256(converted).hexdigest(),
                original_fnv64=fnv64(original_codes),replacement_fnv64=fnv64(converted)))
        else:aux = int8_lookup(codes,scales)
        base = allocate(occupied,len(aux))
        filename = param['logical_name']+('.mx-exponents.bin' if packing == 'overlay' else '.mx-lookup.bin')
        (out/filename).write_bytes(aux)
        index = len(records)
        records.append(dict(index=index,image_file=filename,gpu_base_address=base,
                            image_size_bytes=len(aux),image_sha256=hashlib.sha256(aux).hexdigest()))
        bindings['v_parameter_'+param['logical_name'].replace('.','_')] = dict(
            metadata_address=base,shape=[k,n],logical_name=param['logical_name'],
            packing='DirectE4M3' if packing == 'overlay' else 'Int8Lookup',
            original_packed_sha256=param['packed_sha256'])
    if not bindings:raise ValueError('no query matrices found')
    all_hash = hashlib.sha256()
    for segment in records:all_hash.update((out/segment['image_file']).read_bytes())
    metadata = dict(model=image['model'],dtype='mx_direct_overlay_v1' if packing == 'overlay' else 'mx_int8_lookup_v1',
        checkpoint_weight_sha256=image['checkpoint_weight_sha256'],
        source_manifest_sha256=sha(manifest_path),source_manifest=str(manifest_path),
        preparation_source_sha256=sha(HERE/'prepare_weights.py'),
        image_size_bytes=sum(s['image_size_bytes'] for s in records),segments=records,
        image_sha256=all_hash.hexdigest(),initialization_patches=patches,
        parameters=[dict(logical_name=b['logical_name']+'.mx_lookup',
            segment_index=i,offset_bytes=0,gpu_address=records[i]['gpu_base_address'],
            size_bytes=records[i]['image_size_bytes'],packed_sha256=records[i]['image_sha256'])
            for i,b in enumerate(bindings.values())])
    verify_image_placement(elf,[image,metadata])
    (out/'prepared-image.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return image,metadata,bindings


def fnv64(raw):
    # Native startup also checks the original mapped bytes before modifying a
    # private mapping. SHA-256 binds both files in the simulator/launch receipts.
    value=14695981039346656037
    for byte in raw:value=((value^byte)*1099511628211)&0xffffffffffffffff
    return value


def rewrite(source,bindings):
    seen=[]
    def replace(match):
        args=[s.strip() for s in match[1].split(',')]
        if len(args) != 10 or args[1] not in bindings or args[4] != 'model_chain::mx_model::I8':
            raise ValueError('unexpected selected MX call in the pinned decoder')
        binding=bindings[args[1]]
        if [int(args[6]),int(args[7])] != binding['shape']:
            raise ValueError('query matrix shape differs from prepared metadata')
        args[2]=f"(const __global float*)0x{binding['metadata_address']:08x}u"
        args[4]='model_chain::mx_model::'+binding['packing']
        seen.append(binding['logical_name'])
        return 'model_chain::mx_vpu_linear_prepacked('+', '.join(args)+');'
    text=re.sub(r'model_chain::mx_vpu_linear\(([^;\n]+)\);',replace,source)
    return text,seen


def build_model(root,dest,model,packing):
    audit=root/'artifacts/e2e-hbm-audit'
    base=audit/'activation-reuse-full-v1'/model
    previous=json.loads((base/'build.json').read_text())
    if sha(base/'kernel.radiance.elf') != previous['elf_sha256']:
        raise ValueError('baseline ELF differs from its build receipt')
    for path,digest in previous['device_objects_sha256'].items():
        if sha(Path(path)) != digest:raise ValueError('baseline device object changed')
    header_hash,preparation_hash=sha(HERE/'backend.hpp'),sha(HERE/'prepare_weights.py')
    generated=root/'.tmp/mx-full-canonical'/model
    graph_manifest=json.loads((generated/'manifest.json').read_text())
    image,metadata,bindings=prepare_images(Path(graph_manifest['weight_image_manifest']),
                                           base/'kernel.radiance.elf',dest,packing)
    commands=previous['commands']
    link=next(c.copy() for c in commands if '-o' in c and c[c.index('-o')+1]==str(base/'kernel.radiance.elf'))
    native_link=commands[-1].copy()
    index=next(i for i,v in enumerate(link)if v.endswith('.mu.o'))
    flags=link[:index]
    with tarfile.open(root/'artifacts/models'/model/'generated-sources.tar.gz') as archive:
        pinned={Path(x.name).name:archive.extractfile(x).read() for x in archive.getmembers() if x.isfile()}
    emitted,used,receipts=[],set(),[]
    for source in sorted(generated.glob('stage_chunk_*.cpp')):
        content=source.read_bytes()
        if content != pinned[source.name]:raise ValueError('original generated source changed')
        text,seen=rewrite(content.decode(),bindings)
        if not seen:continue
        used.update(seen);target=dest/source.name;target.write_text(text);emitted.append(source.name)
        obj=dest/(source.stem+'.mu.o');command=flags+['-c',str(target),'-o',str(obj)]
        with(dest/(source.stem+'-device.log')).open('w')as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        receipts.append(command)
        old=next(v for v in link if Path(v).name==obj.name);link[link.index(old)]=str(obj)
        native_obj=dest/(source.stem+'.o')
        command=['g++','-O2','-std=c++17','-fopenmp','-ffp-contract=off',
                 '-I'+str(generated/'native-check'),'-I'+str(HERE.parent),'-c',str(target),'-o',str(native_obj)]
        with(dest/(source.stem+'-native.log')).open('w')as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        receipts.append(command)
        old=next(v for v in native_link if Path(v).name==native_obj.name);native_link[native_link.index(old)]=str(native_obj)
    if used != {b['logical_name']for b in bindings.values()}:raise ValueError('not every prepared query is connected')
    main=(generated/'native-check/main.cpp').read_text()
    loads='\n'.join(f'  if (!map_image({s["gpu_base_address"]}ull,{s["image_size_bytes"]}ull,{json.dumps(str(dest/s["image_file"]))})) return 2;'for s in metadata['segments'])
    patches=metadata['initialization_patches']
    if patches:
        helper='''
static bool patch_private_image(uint64_t base,uint64_t bytes,const char* path,uint64_t expected,uint64_t replacement){
  uint8_t* target=(uint8_t*)base;
  uint64_t hash=14695981039346656037ull;
  for(uint64_t i=0;i<bytes;++i)hash=(hash^target[i])*1099511628211ull;
  if(hash!=expected){fprintf(stderr,"prepared weight precondition failed\\n");return false;}
  const uint64_t page=base&~4095ull,end=(base+bytes+4095)&~4095ull;
  if(mprotect((void*)page,end-page,PROT_READ|PROT_WRITE)){perror("mprotect weight");return false;}
  FILE* input=fopen(path,"rb");if(!input){perror(path);return false;}
  const bool complete=fread(target,1,bytes,input)==bytes && fgetc(input)==EOF;
  fclose(input);
  hash=14695981039346656037ull;
  for(uint64_t i=0;i<bytes;++i)hash=(hash^target[i])*1099511628211ull;
  if(mprotect((void*)page,end-page,PROT_READ)){perror("mprotect weight readonly");return false;}
  return complete && hash==replacement;
}
'''
        main=main.replace('int main() {',helper+'\nint main() {',1)
        # Original map_image preloads occur before the first stage. Patch only
        # those MAP_PRIVATE mappings, then restore read-only page protections.
        patch_calls='\n'.join(f'  if(!patch_private_image({p["address"]}ull,{p["bytes"]}ull,{json.dumps(p["path"])},{p["original_fnv64"]}ull,{p["replacement_fnv64"]}ull)) return 2;'for p in patches)
        marker='  constexpr uint32_t stage_count'
        if marker not in main:raise ValueError('native driver has no stage loop boundary')
        main=main.replace(marker,patch_calls+'\n'+marker,1)
    main=main.replace('int main() {','int main() {\n'+loads,1);(dest/'native-main.cpp').write_text(main)
    command=['g++','-O2','-std=c++17','-fopenmp','-ffp-contract=off','-c',str(dest/'native-main.cpp'),'-o',str(dest/'main.o')]
    subprocess.run(command,check=True);receipts.append(command)
    old=next(v for v in native_link if Path(v).name=='main.o');native_link[native_link.index(old)]=str(dest/'main.o')
    elf=dest/'kernel.radiance.elf';link[link.index('-o')+1]=str(elf)
    with(dest/'link.log').open('w')as log:subprocess.run(link,stdout=log,stderr=subprocess.STDOUT,check=True)
    receipts.append(link);verify_image_placement(elf,[image,metadata])
    native_link[native_link.index('-o')+1]=str(dest/'model-native')
    with(dest/'native-link.log').open('w')as log:subprocess.run(native_link,stdout=log,stderr=subprocess.STDOUT,check=True)
    receipts.append(native_link)
    if (sha(HERE/'backend.hpp'),sha(HERE/'prepare_weights.py')) != (header_hash,preparation_hash):
        raise ValueError('prepared backend changed during the build')
    result=dict(model=model,scope='derived full graph; prepared query weights; build only; execution pending',
        commands=receipts,prepared_image=str(dest/'prepared-image.json'),prepared_bytes=metadata['image_size_bytes'],
        packing=packing,initialization_patches=patches,
        source_build_sha256=sha(base/'build.json'),selected_bindings=bindings,derived_sources=emitted,
        original_checkpoint_manifest_sha256=sha(Path(graph_manifest['weight_image_manifest'])),
        backend_sha256=sha(HERE/'backend.hpp'),preparation_sha256=sha(HERE/'prepare_weights.py'),
        elf=str(elf),elf_sha256=sha(elf),native_sha256=sha(dest/'model-native'),
        image_placement_verified=True,scheduled_warps=previous['scheduled_warps'],
        source_sha256={s:sha(dest/s) for s in emitted},
        objects_sha256={v:sha(Path(v))for c in (link,native_link)for v in c if v.endswith(('.mu.o','.o'))})
    (dest/'build.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Built {model}: {metadata["image_size_bytes"]} additional bytes, {len(emitted)} derived chunks',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--model',action='append',choices=['tinyllama','deepseek_r1_distill_qwen_1_5b','gemma_2_2b_it'])
    parser.add_argument('--packing',choices=['lookup','overlay'],default='overlay')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    for model in args.model or ['tinyllama','deepseek_r1_distill_qwen_1_5b','gemma_2_2b_it']:
        dest=args.out/model;dest.mkdir();build_model(args.task.resolve(),dest,model,args.packing)


if __name__=='__main__':main()
