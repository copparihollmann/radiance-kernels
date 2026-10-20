#!/usr/bin/env python3
"""Collect whole-workload logs for models and HPC; no FireSim orchestration."""
import argparse
import json
from pathlib import Path
from workloads import select
from hpc import collect
from workload_bundle import load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sms', type=int, nargs='+', choices=[1, 2, 4])
    parser.add_argument('--kind', action='append', choices=['full_model', 'hpc'])
    parser.add_argument('--include-controls', action='store_true')
    parser.add_argument('--mx-clock-ratio', type=float)
    parser.add_argument('--gpu-clock-hz', type=float)
    args = parser.parse_args()
    selected = select(load_manifest(args.root), kinds=args.kind or ['full_model', 'hpc'], sms=args.sms)
    if not args.include_controls:
        selected = [(r, m) for r, m in selected if not (m['kind']=='hpc' and m['sm_count']==1 and m['logical_partitions']==2)]
    if not selected:
        parser.error('no matching workloads')
    rows = collect(args.root, selected, args.logs, args.output,
                   mx_clock_ratio=args.mx_clock_ratio, gpu_clock_hz=args.gpu_clock_hz)
    print(json.dumps(dict(workloads=len(rows), parsed=sum(r['profile_status']=='parsed' for r in rows), output=str(args.output))))
    if any(r['profile_status']=='invalid' or r['correctness']=='failed' for r in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
