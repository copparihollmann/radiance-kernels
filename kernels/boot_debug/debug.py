#!/usr/bin/env python3
"""Select, verify, install, and check a Radiance boot diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    commands=parser.add_subparsers(dest='command', required=True)
    commands.add_parser('verify')
    for command in ['list', 'install', 'check']:
        sub=commands.add_parser(command)
        sub.add_argument('--sms', type=int, choices=[1,2,4])
        sub.add_argument('--stage', choices=['host-htif','host-mailboxes','gpu-entry','gpu-scheduler'])
        sub.add_argument('--loader', choices=['logical','direct-hashed'])
        if command=='install':
            sub.add_argument('--deploy-dir', type=Path, required=True)
        if command=='check':
            sub.add_argument('--log', type=Path, required=True)
            sub.add_argument('--firesim-exit-code', type=int, required=True)
    args=parser.parse_args()
    root=args.bundle.resolve()
    data=json.loads((root/'workloads.json').read_text())
    if args.command=='verify':
        for line in (root/'SHA256SUMS').read_text().splitlines():
            expected,name=line.split('  ',1)
            path=(root/name).resolve()
            if not path.is_relative_to(root) or sha(path)!=expected:
                raise ValueError(f'hash mismatch or unsafe path: {name}')
        print('PASS bundle hashes')
        return
    rows=[r for r in data['workloads'] if (args.sms is None or r['sm_count']==args.sms)
        and (args.stage is None or r['stage']==args.stage)
        and (args.loader is None or r['loader']==args.loader)]
    if args.command=='list':
        print(json.dumps(rows,indent=2))
        return
    if len(rows)!=1:
        parser.error('select exactly one case with --sms, --stage, and --loader')
    row=rows[0]
    if args.command=='check':
        text=args.log.read_text(errors='replace')
        stage=['host-htif','host-mailboxes','gpu-entry','gpu-scheduler'].index(row['stage'])
        banner=f'[boot-debug] HOST_BOOT stage=0x{stage:016x} sms=0x{row["sm_count"]:016x}'
        passed=(args.firesim_exit_code==0 and '[boot-debug] '+row['expected_pass'] in text
            and '[boot-debug] FAIL' not in text and '[boot-debug] HOST_TRAP' not in text
            and banner in text)
        print(json.dumps(dict(passed=passed, workload=row['name'], log_sha256=sha(args.log),
            firesim_exit_code=args.firesim_exit_code)))
        sys.exit(0 if passed else 1)
    elf=root/row['boot_elf']
    if sha(elf)!=row['sha256']:
        raise ValueError('boot ELF hash mismatch')
    directory=args.deploy_dir.resolve()/'workloads'
    target=directory/row['name']
    target.mkdir(parents=True,exist_ok=True)
    destination=target/'boot.elf'
    if destination.exists() and sha(destination)!=row['sha256']:
        raise FileExistsError(f'refusing to overwrite a different ELF: {destination}')
    shutil.copyfile(elf,destination)
    source=root/'firesim-workloads'/(row['name']+'.json')
    config=directory/source.name
    if config.exists() and config.read_bytes()!=source.read_bytes():
        raise FileExistsError(f'refusing to overwrite a different workload: {config}')
    shutil.copyfile(source,config)
    print(f'Installed {row["name"]}; set workload.workload_name: {source.name}')


if __name__=='__main__':
    main()
