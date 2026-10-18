#!/usr/bin/env python3
"""Rebuild MX stage objects against the current backend, retaining all fixtures.

Uses recorded full-model link commands and pinned generated sources. The new
ELFs and host executables are qualification candidates, never execution passes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from split_decoder_weights import verify_image_placement


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--warps', type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument('--runtime', type=Path, help='qualified alternate runtime archive')
    parser.add_argument('--model', action='append', choices=[
        'tinyllama', 'deepseek_r1_distill_qwen_1_5b', 'gemma_2_2b_it', 'smolvla_base'])
    args = parser.parse_args()
    if args.warps > 1 and args.runtime is None:
        parser.error('multi-warp builds require an explicit qualified --runtime with worker joining')
    root = args.task.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    audit = root / 'artifacts/e2e-hbm-audit'
    header = root / 'radiance-kernels/kernels/model_chain/mx_vpu/backend.hpp'
    header_hash = sha(header)
    env = {**os.environ, 'TMPDIR': str(root / '.tmp')}
    for model in args.model or ['tinyllama', 'deepseek_r1_distill_qwen_1_5b',
                                'gemma_2_2b_it', 'smolvla_base']:
        dest = args.out / model
        dest.mkdir()
        generated = root / ('.tmp/smolvla-mx-2g' if model == 'smolvla_base'
                            else '.tmp/mx-full-canonical') / model
        original = json.loads((audit / 'runtime-relinked-v2' / model / 'build.json').read_text())
        link = original['commands'][-1].copy()
        first_object = next(i for i, x in enumerate(link) if x.endswith('.mu.o'))
        flags = link[:first_object]
        commands = []
        replacements = {}
        with tarfile.open(root / 'artifacts/models' / model / 'generated-sources.tar.gz') as archive:
            pinned = {Path(x.name).name: archive.extractfile(x).read()
                      for x in archive.getmembers() if x.isfile() and x.name.endswith('.cpp')}
        for i, obj in enumerate(link):
            if not obj.endswith('.mu.o'):
                continue
            path = Path(obj)
            assert sha(path) == original['objects_sha256'][obj], 'retained object changed: ' + obj
            source = generated / path.name.replace('.mu.o', '.cpp')
            contents = source.read_bytes()
            assert contents == pinned[source.name], 'generated source changed: ' + str(source)
            derived_main = source.name == 'kernel.cpp' and args.warps != 1
            if not derived_main and not any(name in contents for name in
                                           (b'model_chain::mx_vpu_linear(', b'model_chain::mx_vpu_add(')):
                continue
            compile_source = source
            if derived_main:
                text, declarations = re.subn(r'(__mu_num_warps\s*=\s*)1;',
                                             rf'\g<1>{args.warps};', contents.decode())
                text, calls = re.subn(r'(mu_schedule\([^;\n]+, )1(\);)',
                                     rf'\g<1>{args.warps}\2', text)
                assert declarations == 1 and calls > 0, 'unrecognized one-warp driver'
                compile_source = dest / 'kernel.cpp'
                compile_source.write_text(text)
            target = dest / path.name
            command = flags + ['-I' + str(generated), '-c', str(compile_source), '-o', str(target)]
            commands.append(command)
            with (dest / (source.stem + '.log')).open('w') as log:
                subprocess.run(command, cwd=generated, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            link[i] = str(target)
            replacements[source.stem] = dict(source_sha256=sha(source),
                                             compiled_source_sha256=sha(compile_source),
                                             device_sha256=sha(target))
        runtime = (args.runtime.resolve() if args.runtime else root / 'radiance-kernels/lib/libmuonrt.a')
        if args.runtime:
            original_runtime = str(root / 'radiance-kernels/lib/libmuonrt.a')
            assert link.count(original_runtime) == 1
            link[link.index(original_runtime)] = str(runtime)
        link[link.index('-o') + 1] = str(dest / 'kernel.radiance.elf')
        commands.append(link)
        with (dest / 'link.log').open('w') as log:
            subprocess.run(link, cwd=generated, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest = json.loads((generated / 'manifest.json').read_text())
        images = [json.loads(Path(manifest[key]).read_text()) for key in
                  ('weight_image_manifest', 'input_image_manifest', 'stage_golden_image_manifest')
                  if manifest.get(key)]
        if images:
            placement = dict(elf_sha256=sha(dest / 'kernel.radiance.elf'))
            try:
                verify_image_placement(dest / 'kernel.radiance.elf', images)
            except ValueError as error:
                placement.update(passed=False, error=str(error))
                (dest / 'placement-check.json').write_text(json.dumps(placement, indent=2) + '\n')
                raise
            placement['passed'] = True
            (dest / 'placement-check.json').write_text(json.dumps(placement, indent=2) + '\n')
        # Rebuild the same native stage objects as an additional full graph
        # format check; retain its original independent goldens and driver.
        native_build = json.loads((audit / 'native' / model / 'build.json').read_text())
        native_link = native_build['link'].copy()
        native_dir = generated / ('native' if model == 'smolvla_base' else 'native-check')
        for i, obj in enumerate(native_link):
            path = Path(obj)
            if obj in native_build['objects_sha256']:
                assert sha(path) == native_build['objects_sha256'][obj], 'native object changed: ' + obj
            if path.stem not in replacements:
                continue
            source = generated / (path.stem + '.cpp')
            target = dest / path.name
            command = ['g++', '-O2', '-std=c++17', '-fopenmp', '-ffp-contract=off',
                       '-I' + str(native_dir), '-I' + str(header.parent.parent),
                       '-c', str(source), '-o', str(target)]
            commands.append(command)
            with (dest / (path.stem + '-native.log')).open('w') as log:
                subprocess.run(command, cwd=generated, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            native_link[i] = str(target)
            replacements[path.stem]['native_sha256'] = sha(target)
        native_link[native_link.index('-o') + 1] = str(dest / 'model-native')
        commands.append(native_link)
        with (dest / 'native-link.log').open('w') as log:
            subprocess.run(native_link, cwd=generated, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        assert sha(header) == header_hash, 'backend changed during build'
        result = dict(model=model, backend_sha256=header_hash, replacements=replacements,
                      commands=commands, elf=str(dest / 'kernel.radiance.elf'),
                      elf_sha256=sha(dest / 'kernel.radiance.elf'),
                      native_sha256=sha(dest / 'model-native'), image_placement_verified=True,
                      scheduled_warps=args.warps, runtime_archive=str(runtime), runtime_sha256=sha(runtime),
                      device_objects_sha256={x: sha(Path(x)) for x in link if x.endswith('.mu.o')},
                      scope='build only; execution pending')
        (dest / 'build.json').write_text(json.dumps(result, indent=2) + '\n')
        print('Rebuilt', model, len(replacements), 'MX stage chunks', flush=True)


if __name__ == '__main__':
    main()
