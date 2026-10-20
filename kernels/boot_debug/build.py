#!/usr/bin/env python3
"""Build small, self-terminating FireSim boot diagnostics. No model weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[1]
sys.path.insert(0, str(REPO / 'kernels/model_chain'))
from package_firesim import package_hashed, verify_package


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loads32(path):
    data = path.read_bytes()
    assert data[:6] == b'\x7fELF\x01\x01'
    offset = struct.unpack_from('<I', data, 28)[0]
    size, count = struct.unpack_from('<HH', data, 42)
    assert size == 32
    result = []
    for i in range(count):
        kind, off, va, pa, fs, ms, flags, alignment = struct.unpack_from('<8I', data, offset+i*32)
        if kind == 1 and ms:
            assert va == pa and fs <= ms
            assert 0x10000000 <= va < va+ms <= 0x80000000
            result.append((va, flags, data[off:off+fs] + bytes(ms-fs)))
    assert result
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gnu', type=Path, required=True, help='GNU RISC-V toolchain root')
    p.add_argument('--llvm', type=Path, required=True, help='Muon LLVM root')
    p.add_argument('--runtime-one-two', type=Path, required=True)
    p.add_argument('--runtime-four', type=Path, required=True)
    args = p.parse_args()
    out = args.out.resolve()
    if (out/'workloads.json').exists():
        raise FileExistsError('refusing to replace a completed bundle')
    out.mkdir(exist_ok=True, parents=True)
    work = out / 'build'
    work.mkdir(exist_ok=True)
    commands = []
    def run(command, cwd=work):
        command = list(map(str, command))
        commands.append(command)
        with (out/'build.log').open('a') as log:
            result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'command failed; see {out / "build.log"}: {command}')
    gcc = args.gnu/'bin/riscv64-unknown-elf-gcc'
    objcopy = args.gnu/'bin/riscv64-unknown-elf-objcopy'
    ld = args.gnu/'bin/riscv64-unknown-elf-ld'
    llvm = args.llvm/'bin/clang++'
    cflags = [str(llvm), '--sysroot='+str(args.gnu/'riscv64-unknown-elf'),
        '--gcc-toolchain='+str(args.gnu), '-nodefaultlibs', '-Xclang', '-target-feature', '-Xclang', '+vortex',
        '-march=rv32im_zfinx_zhinx', '-mabi=ilp32', '-O3', '-g', '-std=c++20', '-mcmodel=medany',
        '-fno-exceptions', '-fno-rtti', '-fdata-sections', '-ffunction-sections', '-DRADIANCE',
        '-DRADIANCE_DEVICE', '-DNDEBUG', '-DLLVM_VORTEX', '-I'+str(REPO/'lib/include'),
        '-nostdinc++', '-isystem', str(args.llvm/'include/c++/v1'),
        '-idirafter', str(args.gnu/'riscv64-unknown-elf/include'),
        '-mllvm', '-riscv-stack-word-stride=16', '-mllvm', '-riscv-max-allocatable-gprs=32']
    hostflags = ['-march=rv64imac_zicsr_zifencei', '-mabi=lp64', '-mcmodel=medany', '-msmall-data-limit=0',
        '-mno-relax', '-ffreestanding', '-fno-builtin', '-fno-common', '-fno-stack-protector',
        '-nostdlib', '-static', '-O2', '-g', '-Wl,--no-relax', '-Wl,--build-id=none']
    rows = []
    for sm in [1, 2, 4]:
        for stage, kind in enumerate(['host-htif', 'host-mailboxes', 'gpu-entry', 'gpu-scheduler']):
            folder = work/f'{kind}-{sm}sm'
            folder.mkdir(exist_ok=True)
            device = None
            result_address = 0
            if stage >= 2:
                device = folder/'device.elf'
                if stage == 2:
                    run(cflags+['-nostartfiles', '-fuse-ld=lld',
                        '-Wl,-T,'+str(REPO/'lib/linker/mu_link.ld'), str(SOURCE/'gpu_entry.S'), '-o', str(device)])
                else:
                    flags = cflags+['-DDEBUG_SMS='+str(sm)]+(['-DMU_CLUSTER_ID_FROM_PRINTBUF=1'] if sm==4 else [])
                    obj = folder/'device.o'
                    run(flags+['-c', str(SOURCE/'gpu_schedule.cpp'), '-o', str(obj)])
                    runtime = args.runtime_four if sm==4 else args.runtime_one_two
                    run(cflags+[str(obj), '-nostartfiles', '-fuse-ld=lld',
                        '-Wl,-Bstatic,-T,'+str(REPO/'lib/linker/mu_link.ld')+',-z,norelro', str(runtime.resolve()),
                        str(REPO/'lib/tohost.S'), '-L'+str(args.llvm/'lib/riscv32-unknown-elf'), '-lc', '-lm',
                        str(args.llvm/'lib/clang/18/lib/riscv32-unknown-elf/libclang_rt.builtins.a'), '-o', str(device)])
                    symbols = subprocess.check_output([str(args.llvm/'bin/llvm-nm'), str(device)], text=True)
                    result_address = next(int(x.split()[0],16) for x in symbols.splitlines() if x.split()[-1:] == ['debug_output'])
            phdrs, sections, objects = [], [], []
            if device:
                for i, (va, flags, data) in enumerate(loads32(device)):
                    raw, obj = folder/f'gpu{i}.bin', folder/f'gpu{i}.o'
                    raw.write_bytes(data)
                    run([ld, '-r', '-b', 'binary', str(raw), '-o', str(obj)])
                    run([objcopy, '--rename-section', f'.data=.gpu{i}', str(obj)])
                    objects.append(str(obj))
                    phdrs.append(f'gpu{i} PT_LOAD FLAGS({flags});')
                    sections.append(f'.gpu{i} {0x100000000+va:#x} : {{ KEEP(*(.gpu{i})) }} :gpu{i}')
            linker = folder/'host.ld'
            linker.write_text((SOURCE/'host.ld').read_text().replace('/* GPU_PHDRS */', '\n'.join(phdrs))
                .replace('/* GPU_SECTIONS */', '\n'.join(sections)))
            name = f'{kind}-{sm}sm'
            logical = out/(name+'-logical.elf')
            run([gcc]+hostflags+['-DDEBUG_SMS='+str(sm), '-DDEBUG_STAGE='+str(stage),
                '-DDEBUG_RESULT_ADDRESS='+hex(result_address), '-T', str(linker),
                '-Wl,-Map,'+str(out/(name+'.map')), str(SOURCE/'host_start.S'), str(SOURCE/'host.c')]+objects+['-o', str(logical)])
            variants = [('logical', logical)]
            if device:
                hashed = out/(name+'-direct-hashed.elf')
                package_hashed(logical, [], hashed, 0x80000000, 32, 4, compact=True)
                verify_package(hashed.with_suffix('.elf.json'))
                variants.append(('direct-hashed', hashed))
                shutil.copyfile(device, out/(name+'.device.elf'))
            for loader, elf in variants:
                row = dict(name=name+'-'+loader, kind='control', stage=kind, sm_count=sm, loader=loader,
                    boot_elf=elf.name, sha256=digest(elf), expected_pass='PASS host-htif' if stage==0 else
                    'PASS host-mailboxes' if stage==1 else 'PASS all-sm',
                    htif_exit=True, requires_vpu=False, requires_mx=False, fpga_verified=False,
                    result_address=result_address, timeout_host_cycles=200000000 if stage>=2 else None)
                if device:
                    row['device_elf'] = name+'.device.elf'
                    row['device_sha256'] = digest(device)
                rows.append(row)
                install = out/'firesim-workloads'/row['name']
                install.mkdir(parents=True, exist_ok=True)
                # Relative link keeps the extracted diagnostic bundle small.
                if not (install/'boot.elf').is_symlink():
                    (install/'boot.elf').symlink_to('../../'+elf.name)
                (install.parent/(row['name']+'.json')).write_text(json.dumps(dict(
                    benchmark_name=row['name'], common_bootbinary='boot.elf', common_rootfs=None,
                    common_outputs=[], common_simulation_outputs=['uartlog']), indent=2)+'\n')
    (out/'workloads.json').write_text(json.dumps(dict(schema_version=1, purpose='Radiance boot diagnosis',
        hardware_execution_verified=False, workloads=rows,
        runtime_sha256=dict(one_two=digest(args.runtime_one_two), four=digest(args.runtime_four)),
        source_sha256={p.name:digest(p) for p in SOURCE.iterdir() if p.is_file()}, commands=commands), indent=2)+'\n')
    shutil.copytree(SOURCE, out/'sources', ignore=shutil.ignore_patterns('__pycache__'))
    for filename in ['README.md', 'debug.py', 'verify_elf.py']:
        shutil.copyfile(SOURCE/filename, out/filename)
    print(json.dumps(dict(output=str(out), cases=len(rows))))


if __name__ == '__main__':
    main()
