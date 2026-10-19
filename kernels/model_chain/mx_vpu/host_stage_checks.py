"""Keep evaluation oracles outside GPU RAM and check actual GPU tensor dumps.

The model retains all stage outputs and its original graph. Only test-oracle
storage and the verifier tail change; no reference data enters simulator RAM.
"""
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gold_lines(path):
    result = {}
    with path.open() as stream:
        for line in stream:
            match = re.match(r'alignas\(64\) __global (float|int32_t) (v_gold_\w+)\[(\d+)\] = \{(.*)\};', line)
            if match:
                if match[2] in result:
                    raise ValueError('duplicate frozen reference: '+match[2])
                result[match[2]] = hashlib.sha256(line.strip().encode()).hexdigest()
    return result


def prepare(target: Path, frozen_data: Path | None = None):
    manifest = json.loads((target/'manifest.json').read_text())
    if manifest['batch_clusters'] not in (2,4) or not manifest['device_check_all_stages'] or manifest['generation'] != 'greedy':
        raise ValueError('host stage checker requires the complete two- or four-replica greedy source bundle')
    if frozen_data and gold_lines(target/'model_data.cpp') != gold_lines(frozen_data):
        raise ValueError('new format references differ from the frozen full-model reference')
    chunks = sorted(target.glob('verify_chunk_*.cpp'))
    records = []
    pattern = (r'i < (\d+);[^\n]*\n?\s*if \('
               r'(?:!mu_close\()?([\w]+)\[i \+ replica \* (\d+)u\]'
               r'(?:, | != )([\w]+)\[i\]')
    for path in chunks:
        source = path.read_text()
        for count,out,stride,gold in re.findall(pattern,source):
            dtype = 'int32' if f'extern __global int32_t {out}[];' in source else 'float32'
            records.append(dict(symbol=out,reference_symbol=gold,elements=int(count),
                                stride_elements=int(stride),dtype=dtype))
    if len(records) != manifest['stages'] or len({x['symbol']for x in records}) != len(records):
        raise ValueError('verifier bundle does not cover every distinct graph stage')
    by_gold = {row['reference_symbol']:row for row in records}
    refs = target/'references'
    refs.mkdir(exist_ok=False)
    found = set()
    with (target/'model_data.cpp').open() as source, (target/'model_data.device.cpp').open('w') as device:
        for line in source:
            match = re.match(r'alignas\(64\) __global (float|int32_t) (v_gold_\w+)\[(\d+)\] = \{(.*)\};',line)
            if not match:
                device.write(line)
                continue
            if match[2] in by_gold:
                row = by_gold[match[2]]
                dtype = '<i4' if row['dtype'] == 'int32' else '<f4'
                values = np.fromstring(match[4].replace('f',''),sep=',',dtype=dtype)
                if values.size != row['elements'] or row['stride_elements'] < row['elements']:
                    raise ValueError('reference shape differs from verifier: '+match[2])
                path = refs/(match[2]+'.bin')
                path.write_bytes(values.tobytes())
                row.update(path=str(path.resolve()),sha256=sha(path))
                found.add(match[2])
    if found != set(by_gold):
        raise ValueError('missing stage oracles')
    text = (target/'kernel.cpp').read_text()
    text,prototypes = re.subn(r'^bool verify_chunk_\d+\(uint32_t\);\n','',text,flags=re.M)
    text,calls = re.subn(r'^  if \(!verify_chunk_\d+\(replica\)\) return 0;\n','',text,flags=re.M)
    if prototypes != len(chunks) or calls != len(chunks):
        raise ValueError('unexpected generated verifier tail')
    tokens = manifest['reference_generated_token_ids']
    declarations = ''.join(f'extern __global int32_t v_decode{i}_token_ids[];\n'for i in range(len(tokens)))
    if text.count('int main() {') != 1:
        raise ValueError('unexpected kernel entry')
    text = text.replace('int main() {',declarations+'\nint main() {')
    marker = '  mu_tohost(errors ? ((errors << 1) | 1u) : 0u);'
    if text.count(marker) != 1:
        raise ValueError('unexpected tohost tail')
    checks = ''.join(f'  if (v_decode{i}_token_ids[replica*16u] != {int(value)}) {{ mu_tohost({2*i+1}u); return 0; }}\n'for i,value in enumerate(tokens))
    text = text.replace(marker,checks+marker)
    (target/'kernel.device.cpp').write_text(text)
    original_sources = manifest['device_source_files']
    manifest.update(device_check_all_stages=False,host_stage_checks=True,
                    device_verification_mode='embedded_greedy_tokens_and_host_comparison_of_all_actual_GPU_buffers',
                    native_reference_sources=['model_data.cpp',*[p.name for p in chunks]],
                    device_source_files=['kernel.device.cpp','model_data.device.cpp',*[name for name in original_sources if name.startswith('stage_chunk_') or name in('fallback_runtime.cpp','mx_vpu_runtime.cpp')]],
                    host_stage_reference_manifest=str((target/'host-stage-references.json').resolve()))
    # Keep the ordinary device build aligned with the manifest. The public ELF
    # name stays stable while the generated source keeps its oracle-free name.
    makefile = (target/'Makefile').read_text()
    makefile, entries = re.subn(r'^MU_SRCS = kernel.cpp$',
                              'MU_SRCS = kernel.device.cpp',makefile,flags=re.M)
    dependencies = ' '.join(name for name in manifest['device_source_files']
                            if name != 'kernel.device.cpp')
    makefile, dependency_lines = re.subn(r'^MU_SRC_DEPS = .*$',
        'MU_SRC_DEPS = '+dependencies,makefile,flags=re.M)
    if entries != 1 or dependency_lines != 1:
        raise ValueError('unexpected generated device build source list')
    makefile = makefile.replace('kernel.mu.o:', 'kernel.device.mu.o:')
    makefile = makefile.replace('kernel.mu.o ', 'kernel.device.mu.o ')
    makefile += ('\nkernel.radiance.elf: kernel.device.radiance.elf\n'
                 '\tcp $< $@\n')
    (target/'Makefile').write_text(makefile)
    reference_manifest = dict(schema='radiance-host-stage-references-v1',stages=manifest['stages'],
        num_replicas=manifest['batch_clusters'],records=records,rtol=.005,atol=.0005,
        frozen_data_sha256=sha(frozen_data)if frozen_data else None,
        scope='independentfrozenformatreferences; compare alltensorvalues read fromcompletedactualGPUruns, never inject into GPU RAM')
    (target/'host-stage-references.json').write_text(json.dumps(reference_manifest,indent=2)+'\n')
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return reference_manifest


def check(target: Path, result_dir: Path, clusters: int):
    refs = json.loads((target/'host-stage-references.json').read_text())
    if clusters not in (1,2,4):
        raise ValueError('unsupported replica count')
    rows = []
    for cluster in range(clusters):
        bad_stages = 0
        checked = 0
        failures = []
        max_error = 0.
        for item in refs['records']:
            path = Path(item['path'])
            if sha(path) != item['sha256']:
                raise ValueError('frozen stage reference changed: '+str(path))
            dtype = '<i4' if item['dtype'] == 'int32' else '<f4'
            actual_path = result_dir/'tensors'/f'{cluster}-{item["symbol"]}.bin'
            expected = np.fromfile(path,dtype=dtype)
            actual = np.fromfile(actual_path,dtype=dtype)if actual_path.exists()else np.array([],dtype=dtype)
            shape = actual.shape == expected.shape == (item['elements'],)
            if shape:
                if item['dtype'] == 'int32':ok = actual == expected;error = 0.
                else:
                    delta = np.abs(actual-expected)
                    ok = np.isfinite(actual)&(delta<=np.float32(refs['rtol'])*np.abs(expected)+np.float32(refs['atol']))
                    error = float(np.max(delta))
                    max_error = max(max_error,error)
                count = int(np.count_nonzero(~ok))
            else:count = item['elements'];error = None
            checked += 1
            if count:
                bad_stages += 1
                failures.append(dict(symbol=item['symbol'],failed_elements=count,shape_passed=shape,max_abs_error=error))
        rows.append(dict(cluster=cluster,checked_stages=checked,failed_stages=bad_stages,
                         max_abs_error=max_error,failures=failures,passed=checked==refs['stages']and bad_stages==0))
    return dict(passed=all(row['passed']for row in rows),clusters=rows,
                scope='full actualGPU stagebuffers compared onhost; not nativeCPU execution',
                reference_manifest_sha256=sha(target/'host-stage-references.json'))
