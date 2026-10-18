#!/usr/bin/env python3
"""Independent upstream module-boundary tensors for the full MX/VPU policy graph."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE.parents[1]/"evaluation/llm")]
from stitch import build
from verify_smolvla_policy import run
from mx_vpu import reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    graph=build("smolvla_base");queues=defaultdict(list);mapped=[]
    next_bias={s["reads"][0] for s in graph.stages if s["op"]=="bias_add"}
    for index,stage in enumerate(graph.stages):
        attrs=stage["attrs"];parameter=None
        if stage["op"]=="bias_add":parameter=attrs.get("parameter")
        elif stage["op"]=="linear" and stage["id"] not in next_bias:parameter=attrs.get("parameter")
        elif stage["op"] in ("rmsnorm","layernorm"):
            parameter=attrs.get("parameter") or next((k for k in attrs.get("parameters",{}) if k.endswith('.weight')),None)
        if parameter and parameter.endswith((".weight",".bias")):
            module=parameter.rsplit('.',1)[0]
            entry=dict(index=index,id=stage['id'],module=module,shape=stage['shape'],bytes=4*int(np.prod(stage['shape'])))
            queues[module].append(entry);mapped.append(entry)
    observed=defaultdict(int);records=[]
    def sink(module,tensor):
        if module not in queues:return
        ordinal=observed[module];observed[module]+=1
        if ordinal>=len(queues[module]):raise ValueError(f'extra upstream module call {module}:{ordinal}')
        entry=queues[module][ordinal];a=tensor.detach().to(dtype=__import__('torch').float32).contiguous().numpy()
        if a.nbytes!=entry['bytes']:raise ValueError(f'module shape differs at {entry["id"]}: {a.shape} vs {entry["shape"]}')
        target=args.out/f'{entry["index"]:04d}.bin';a.astype('<f4',copy=False).tofile(target)
        records.append(dict(**entry,path=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    def projection(value,weight,bias):
        import torch
        result=reference.linear(value.detach().numpy(),weight.detach().numpy().T)
        if bias is not None:result=reference.add(result,bias.detach().numpy())
        return torch.from_numpy(result)
    result=run(args.checkpoint_dir,True,action_projection=projection,module_tensor_sink=sink)
    missing={name:len(entries)-observed[name] for name,entries in queues.items() if len(entries)!=observed[name]}
    record=dict(scope='independent upstream module outputs, not all graph operations',stages=len(graph.stages),
        checked_stages=len(records),missing_modules=missing,tensors=records,final=result,
        projection_module='action_in_proj',format_reference_sha256=hashlib.sha256((HERE/'reference.py').read_bytes()).hexdigest(),
        rtol=0.001,atol=0.001)
    (args.out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(captured=len(records),mapped=len(mapped),missing=missing)),flush=True)
    if missing:raise SystemExit(1)


if __name__=='__main__':main()
