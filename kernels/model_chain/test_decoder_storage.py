"""Execute every reduced decoder stage using reused four-replica storage."""
import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from compile_decoder import generate
from decoder_storage import pooled_source, replicated_plan
import reference
import stitch


class DecoderStorage(unittest.TestCase):
    def test_all_three_decoder_graphs_preserve_every_stage_and_kv_dependency(self):
        for model in ('tinyllama', 'deepseek_r1_distill_qwen_1_5b', 'gemma_2_2b_it'):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                target = generate(model, 1, 2, 2, 'greedy', root,
                                  batch_clusters=4, device_check_all_stages=True,
                                  stages_per_object=20)
                spec = reference.reduced_spec(stitch.model_specs()[model])
                spec['num_hidden_layers'] = 1
                graph = stitch.build(model, prefill=2, decode_steps=2,
                                     generation='greedy', specs={model: spec})
                preserve = [name for name in graph.tensors if name.startswith('decode') and name.endswith('.token_ids')]
                layout = replicated_plan(graph, 4, preserve)
                self.assertLess(layout['arena_bytes'], layout['unreused_bytes'])
                data = target / 'model_data.cpp'
                original = data.read_text()
                data.write_text(pooled_source(original, layout))
                (root / 'mu_intrinsics.h').write_text('#pragma once\n#define __global\n#define MU_NUM_CORES 1\ninline unsigned vx_core_id(){return 0;}\n')
                (root/'mu_schedule.h').write_text('#pragma once\n')
                (root/'kernel_verify.h').write_text('#pragma once\n')
                bridge = '#include <stdint.h>\n'
                for i in range(len(graph.stages)):
                    bridge += f'void stage_{i}(void*,uint32_t,uint32_t,uint32_t);\n'
                bridge += 'extern "C" void run(unsigned stage,unsigned replica){switch(stage){\n'
                bridge += ''.join(f'case {i}:stage_{i}((void*)(uintptr_t)replica,0,1,0);break;\n' for i in range(len(graph.stages)))
                bridge += '}}\n'
                (root/'bridge.cpp').write_text(bridge)
                library = root/'stages.so'
                compiled = subprocess.run(['g++','-O2','-ffp-contract=off','-shared','-fPIC',
                                '-I'+str(root),'-I'+str(target),'-I'+str(Path(__file__).parent),str(data),
                                *map(str,sorted(target.glob('stage_chunk_*.cpp'))),
                                str(root/'bridge.cpp'),'-o',str(library)],check=False,
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                self.assertEqual(compiled.returncode, 0, compiled.stderr.decode()[:2500])
                lib=ctypes.CDLL(str(library));lib.run.argtypes=[ctypes.c_uint,ctypes.c_uint]
                for index, stage in enumerate(graph.stages):
                    name=stage['writes'];symbol='v_'+name.replace('.','_')
                    count=int(np.prod(graph.tensors[name]['shape']))
                    scalar=ctypes.c_int32 if graph.tensors[name]['dtype']=='int32' else ctypes.c_float
                    expected=np.ctypeslib.as_array((scalar*count).in_dll(lib,'v_gold_'+name.replace('.','_')))
                    stride=layout['allocation'][name]['replica_stride_elements']
                    actual=np.ctypeslib.as_array((scalar*(4*stride)).in_dll(lib,symbol))
                    for replica in range(4):
                        lib.run(index,replica)
                        np.testing.assert_allclose(actual[replica*stride:replica*stride+count],expected,
                                                   atol=.0005,rtol=.005,err_msg=f'{model} stage{index} replica{replica}')
                for name in preserve:
                    item=layout['allocation'][name]
                    self.assertEqual(item['last_use'],len(graph.stages))
                with self.assertRaisesRegex(ValueError,'missing stage activation'):
                    pooled_source(original.replace('v_prefill_embedding[', 'v_prefill_embedding_changed['),layout)
