"""Qualification must reject changed fixtures, missing/reordered work and bad units."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from qualify_decoder import check_hashes, expected_accelerator_counts, stage_order


class FullDecoderAudit(unittest.TestCase):
    def test_exact_stage_order_is_required_independently_for_each_sm(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'progress.jsonl'
            rows = [dict(event='stage_entry',cluster=c,symbols=[f'_Z7stage_{i}Pvjjj'])
                    for i in range(3)for c in range(2)]
            def write(items):
                path.write_text(''.join(json.dumps(x)+'\n'for x in items))
            write(rows)
            self.assertEqual(stage_order(path,2,3),{'0':3,'1':3})
            for broken in (rows[:-1],rows+[rows[-1]],rows[:-2]+rows[-2:][::-1][:1],
                           rows[:1]+rows[3:]+rows[1:3]):
                with self.subTest(broken=broken):
                    write(broken)
                    with self.assertRaisesRegex(ValueError,'omits, repeats, or reorders'):
                        stage_order(path,2,3)
            broken = copy.deepcopy(rows)
            broken[-1]['cluster'] = 2
            write(broken)
            with self.assertRaisesRegex(ValueError,'unexpected cluster'):
                stage_order(path,2,3)
            broken = copy.deepcopy(rows)
            broken[-1]['symbols'].append('_Z8stage_91Pvjjj')
            write(broken)
            with self.assertRaisesRegex(ValueError,'ambiguous stage'):
                stage_order(path,2,3)

    def test_graph_shapes_and_native_counters_use_distinct_loop_tile_units(self):
        spec = dict(hidden_size=32,num_attention_heads=4,head_dim=8)
        refs = dict(records=[dict(symbol='v_prefill_layer00_q_proj',elements=33*32),
                             dict(symbol='v_decode0_layer00_q_proj',elements=32),
                             dict(symbol='v_prefill_layer00_attn_residual',elements=513),
                             dict(symbol='v_prefill_embedding',elements=512)])
        m = dict(stages=4,mx_linear_policy='q_proj',batch_clusters=2,mx_linear_macs=34*32*32,
                 stage_backends=[dict(stage='prefill.layer00.q_proj',backend='fp8_mx'),
                                 dict(stage='decode0.layer00.q_proj',backend='fp8_mx'),
                                 dict(stage='prefill.layer00.attn_residual',backend='bf16_vpu'),
                                 dict(stage='prefill.embedding',backend='muon')])
        counts = [['106','16','6','0']]
        result = expected_accelerator_counts(m,refs,spec,counts)
        self.assertEqual(result,dict(total_commands=53,mx_loops=8,mesh_tiles=16,vpu_commands=3))
        for bad in ([['106','32','6','0']],[['106','16','5','0']],[] ):
            with self.assertRaisesRegex(ValueError,'native'):
                expected_accelerator_counts(m,refs,spec,bad)
        broken = copy.deepcopy(refs)
        broken['records'][0]['elements'] += 1
        with self.assertRaisesRegex(ValueError,'query output shape'):
            expected_accelerator_counts(m,broken,spec,counts)
        broken = dict(m,mx_linear_macs=m['mx_linear_macs']-1)
        with self.assertRaisesRegex(ValueError,'MAC count'):
            expected_accelerator_counts(broken,refs,spec,counts)

    def test_rehash_inputs_instead_of_trusting_previous_green_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'fixture'
            path.write_bytes(b'frozen input')
            hashes = {str(path):hashlib.sha256(path.read_bytes()).hexdigest()}
            check_hashes(hashes)
            path.write_bytes(b'changed input')
            with self.assertRaisesRegex(ValueError,'input changed'):
                check_hashes(hashes)
            with self.assertRaisesRegex(ValueError,'missing pinned'):
                check_hashes({})

    def test_broadcast_bias_work_is_included_for_biased_decoders(self):
        spec = dict(hidden_size=32,num_attention_heads=4,head_dim=8)
        m = dict(stages=4,mx_linear_policy='q_proj',batch_clusters=2,mx_linear_macs=32*32,
                 stage_backends=[dict(stage='decode0.layer00.q_proj',backend='fp8_mx'),
                                 *[dict(stage='decode0.layer00.'+name,backend='bf16_vpu')
                                   for name in ('q_bias','k_bias','v_bias')]])
        refs = dict(records=[dict(symbol='v_decode0_layer00_'+name,elements=32)
                             for name in ('q_proj','q_bias','k_bias','v_bias')])
        result = expected_accelerator_counts(m,refs,spec,[['36','4','6','0']])
        self.assertEqual(result,dict(total_commands=18,mx_loops=2,mesh_tiles=4,vpu_commands=3))


if __name__ == '__main__':
    unittest.main()
