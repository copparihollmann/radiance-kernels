#!/usr/bin/env python3
"""Relink retained device objects after a scheduler fix, preserving original ELFs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from split_decoder_weights import verify_image_placement


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, type=Path)
    parser.add_argument('--out', type=Path, help='new directory for relinked ELFs')
    parser.add_argument('--llvm', required=True, type=Path)
    parser.add_argument('--riscv', required=True, type=Path)
    args = parser.parse_args()
    root = args.task.resolve()
    out = args.out or root / 'artifacts/e2e-hbm-audit/runtime-relinked-v2'
    out.mkdir(exist_ok=True)
    env = {**os.environ, 'TMPDIR': str(root / '.tmp'), 'LLVM_MUON': str(args.llvm),
           'RISCV_TOOLCHAIN_PATH': str(args.riscv), 'RISCV64_TOOLCHAIN_PATH': str(args.riscv),
           'RISCV_SYSROOT': str(args.riscv / 'riscv64-unknown-elf'),
           'MU_LIBC_INCLUDE': str(args.riscv / 'riscv64-unknown-elf/include'),
           'MU_USE_LIBC': '1', 'EXTRA_MU_CFLAGS': '-nostdinc++ -isystem ' + str(args.llvm / 'include/c++/v1')}
    sources = {'tiny-two-sm': root / '.tmp/mx-two-sm-control/tinyllama'}
    for name in ('tinyllama', 'deepseek_r1_distill_qwen_1_5b', 'gemma_2_2b_it'):
        sources[name] = root / '.tmp/mx-full-canonical' / name
    sources['smolvla_base'] = root / '.tmp/smolvla-mx-2g/smolvla_base'
    records = []
    for name, source in sources.items():
        dest = out / name
        dest.mkdir(exist_ok=False)
        dry = subprocess.check_output(['make', '-n', '-W', 'kernel.mu.o', 'kernel.radiance.elf'], cwd=source, env=env, text=True)
        (dest / 'make-dry-run.log').write_text(dry)
        links = [shlex.split(line) for line in dry.splitlines() if 'clang++ ' in line and ' -o kernel.radiance.elf' in line]
        assert len(links) == 1, name
        link = links[0]
        first_object = next(i for i, arg in enumerate(link) if arg.endswith('.mu.o'))
        commands = []
        objects = []
        for i, arg in enumerate(link):
            if not arg.endswith('.mu.o'):
                continue
            original = source / arg
            if original.exists():
                link[i] = str(original)
                objects.append(original)
            else:
                # Make auto-removes some intermediate objects. Recompile only that
                # exact retained source; all stage bodies and golden data stay pinned.
                cpp = original.with_name(original.name.replace('.mu.o', '.cpp'))
                target = dest / original.name
                command = link[:first_object] + ['-c', str(cpp), '-o', str(target)]
                commands.append(command)
                with (dest / ('compile-' + cpp.stem + '.log')).open('w') as output:
                    subprocess.run(command, cwd=source, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
                link[i] = str(target)
                objects.append(target)
        elf = dest / 'kernel.radiance.elf'
        link[link.index('-o') + 1] = str(elf)
        commands.append(link)
        with (dest / 'link.log').open('w') as output:
            subprocess.run(link, cwd=source, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
        manifest = json.loads((source / 'manifest.json').read_text())
        images = [json.loads(Path(manifest[key]).read_text())
                  for key in ('weight_image_manifest', 'input_image_manifest', 'stage_golden_image_manifest') if manifest.get(key)]
        if images:
            verify_image_placement(elf, images)
        record = dict(model=name, elf=str(elf), elf_sha256=sha(elf),
                      original_elf=str(source / 'kernel.radiance.elf'), original_elf_sha256=sha(source / 'kernel.radiance.elf'),
                      runtime_sha256=sha(root / 'radiance-kernels/lib/libmuonrt.a'),
                      scheduler_source_sha256=sha(root / 'radiance-kernels/lib/src/mu_schedule.cpp'),
                      commands=commands, objects_sha256={str(p): sha(p) for p in objects}, image_placement_verified=True,
                      scope='same generated stage objects and references; per-core scheduler contexts; execution pending')
        (dest / 'build.json').write_text(json.dumps(record, indent=2) + '\n')
        records.append(record)
        print('Relinked', name, flush=True)
    (out / 'builds.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
