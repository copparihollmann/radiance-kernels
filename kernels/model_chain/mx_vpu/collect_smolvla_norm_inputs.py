#!/usr/bin/env python3
"""Capture upstream norm inputs to distinguish reduction error from input drift.

Runs the complete pinned policy, with the existing independent MX projection
reference. It neither regenerates nor overwrites any acceptance golden.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / 'evaluation/llm')]
from stitch import build
from verify_smolvla_policy import run
from mx_vpu import reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--indices', type=int, nargs='+', default=[395, 413, 431, 634])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    graph = build('smolvla_base')
    wanted = set(args.indices)
    assert all(0 <= i < len(graph.stages) and graph.stages[i]['op'] == 'layernorm' for i in wanted)
    queues = defaultdict(list)
    for index, stage in enumerate(graph.stages):
        if stage['op'] != 'layernorm':
            continue
        key = next(k for k in stage['attrs']['parameters'] if k.endswith('.weight'))
        queues[key.rsplit('.', 1)[0]].append((index, stage))
    calls = defaultdict(int)
    records = []

    def sink(module, tensor):
        if module not in queues:
            return
        ordinal = calls[module]
        calls[module] += 1
        assert ordinal < len(queues[module]), 'unexpected upstream module call'
        index, stage = queues[module][ordinal]
        if index not in wanted:
            return
        array = tensor.detach().float().contiguous().numpy()
        assert tuple(array.shape) == tuple(stage['shape'])
        path = args.out / f'{index:04d}-input.bin'
        array.astype('<f4', copy=False).tofile(path)
        records.append(dict(index=index, id=stage['id'], module=module, ordinal=ordinal,
                            shape=list(array.shape), path=path.name,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def projection(value, weight, bias):
        import torch
        output = reference.linear(value.detach().numpy(), weight.detach().numpy().T)
        if bias is not None:
            output = reference.add(output, bias.detach().numpy())
        return torch.from_numpy(output)

    result = run(args.checkpoint_dir, True, action_projection=projection, module_input_sink=sink)
    missing = sorted(wanted - {r['index'] for r in records})
    manifest = dict(scope='diagnostic upstream inputs; acceptance goldens unchanged',
                    tensors=records, missing=missing, full_policy_result=result,
                    reference_sha256=hashlib.sha256((HERE / 'reference.py').read_bytes()).hexdigest())
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(captured=len(records), missing=missing)), flush=True)
    if missing:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
