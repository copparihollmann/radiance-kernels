#!/usr/bin/env python3
"""Validate boot layout, HTIF symbols, debug information and exact GPU payloads."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def read(path):
    data=path.read_bytes()
    assert data[:6]==b'\x7fELF\x02\x01', str(path)
    entry,phoff,shoff=struct.unpack_from('<QQQ',data,24)
    flags=struct.unpack_from('<I',data,48)[0]
    phsize,phnum,shsize,shnum,shstr=struct.unpack_from('<5H',data,54)
    assert phsize==56 and shsize==64
    segments=[struct.unpack_from('<II6Q',data,phoff+i*phsize) for i in range(phnum)]
    sections=[struct.unpack_from('<II4QII2Q',data,shoff+i*shsize) for i in range(shnum)]
    strings=data[sections[shstr][4]:sections[shstr][4]+sections[shstr][5]]
    names=[strings[s[0]:].split(b'\0',1)[0].decode() for s in sections]
    symbols={}
    for section in sections:
        if section[1]!=2: continue
        strings_section=sections[section[6]]
        strings=data[strings_section[4]:strings_section[4]+strings_section[5]]
        for offset in range(section[4],section[4]+section[5],24):
            name,info,other,index,value,size=struct.unpack_from('<IBBHQQ',data,offset)
            symbols[strings[name:].split(b'\0',1)[0].decode()]=(value,size,index)
    return data,entry,flags,segments,sections,names,symbols


def audit(root):
    rows=json.loads((root/'workloads.json').read_text())['workloads']
    verified=[]
    for row in rows:
        path=root/row['boot_elf']
        data,entry,flags,segments,sections,names,symbols=read(path)
        assert hashlib.sha256(data).hexdigest()==row['sha256']
        assert entry==symbols['_start'][0]==0x80000000
        assert flags&6==0, 'host must use the integer ABI'
        assert '.debug_info' in names
        loads=sorted((s for s in segments if s[0]==1 and s[6]),key=lambda s:s[4])
        for s in loads:
            assert s[3]==s[4] and s[5]<=s[6] and s[2]+s[5]<=len(data)
            assert s[2]%s[7]==s[3]%s[7]
        for left,right in zip(loads,loads[1:]):
            assert left[4]+left[6]<=right[4], 'overlapping PT_LOADs'
        assert any(s[1]&1 and s[4]<=entry<s[4]+s[5] for s in loads)
        for name in ['tohost','fromhost']:
            address,size,index=symbols[name]
            assert address%64==0 and size==8 and names[index]=='.htif'
            seg=next(s for s in loads if s[4]<=address<address+8<=s[4]+s[5])
            assert data[seg[2]+address-seg[4]:seg[2]+address-seg[4]+8]==bytes(8)
        assert abs(symbols['tohost'][0]-symbols['fromhost'][0])>=64
        assert symbols['_end'][0] < symbols['__stack_bottom'][0] < symbols['__stack_top'][0]
        assert all(s[4]+s[6]<0x80070000 for s in loads if s[4]<0x100000000)
        if row.get('device_elf') and row['loader']=='logical':
            gpu=(root/row['device_elf']).read_bytes()
            phoff=struct.unpack_from('<I',gpu,28)[0]
            count=struct.unpack_from('<H',gpu,44)[0]
            for i in range(count):
                kind,off,va,pa,fs,ms,gflags,alignment=struct.unpack_from('<8I',gpu,phoff+32*i)
                if kind!=1 or not ms: continue
                gpu_data=gpu[off:off+fs]+bytes(ms-fs)
                seg=next(s for s in loads if s[4]==0x100000000+va)
                assert seg[5]==seg[6]==len(gpu_data)
                assert data[seg[2]:seg[2]+seg[5]]==gpu_data
        verified.append(dict(name=row['name'],sha256=row['sha256'],entry=entry,
            tohost=symbols['tohost'][0],fromhost=symbols['fromhost'][0],
            host_stack_bottom=symbols['__stack_bottom'][0],host_stack_top=symbols['__stack_top'][0],
            debug_info=True,integer_host_abi=True,nonoverlapping_loads=True))
    return dict(passed=True,cases=len(verified),checks=verified,hardware_execution_verified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle',type=Path)
    args=p.parse_args()
    print(json.dumps(audit(args.bundle),indent=2))
