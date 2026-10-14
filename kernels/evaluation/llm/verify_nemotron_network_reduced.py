"""Compare the reduced hybrid graph with a same-weight Transformers model.

Transformers 5.3 represents the pinned dense-MLP-only block as a MoE slot.
The adapter below changes that slot to NemotronHMLP during model construction,
then restores the library registry. No installed source file is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource

import numpy as np
import torch
import transformers

import nemotron_graph_reference as graph_reference
import reference
import stitch
from transformers.models.nemotron_h import configuration_nemotron_h as configuration
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h import modeling_nemotron_h as modeling
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHAttention, NemotronHHybridDynamicCache, NemotronHModel,
    NemotronHMamba2Mixer, NemotronHMLP, NemotronHRMSNorm,
)


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def run(layers: int = 42, prefill: int = 128,
        fault: str = 'none') -> dict:
    if prefill <= 0:
        raise ValueError('prefill must be positive')
    if fault not in ('none', 'noncausal_attention', 'drop_mamba_state'):
        raise ValueError(f'unsupported control fault: {fault}')
    if transformers.__version__ != '5.3.0':
        raise ValueError('the reduced network comparison requires Transformers 5.3.0')
    torch.set_num_threads(1)
    torch.manual_seed(17)
    pinned = stitch.model_specs()['nemotron3_nano_4b']
    spec = graph_reference.control_spec(pinned, layers)
    kinds = spec['layer_kinds']
    if kinds != pinned['layer_kinds'][:layers]:
        raise ValueError('reduced block order differs from the pinned model')
    blocks = ['mamba' if x == 'mamba2' else
              'attention' if x == 'full_attention' else 'moe'
              for x in kinds]
    config = NemotronHConfig(
        layers_block_type=blocks, hidden_size=spec['hidden_size'],
        intermediate_size=spec['intermediate_size'],
        vocab_size=spec['vocab_size'],
        num_attention_heads=spec['num_attention_heads'],
        num_key_value_heads=spec['num_key_value_heads'],
        head_dim=spec['head_dim'],
        mamba_num_heads=spec['mamba_num_heads'],
        mamba_head_dim=spec['mamba_head_dim'],
        mamba_n_groups=spec['n_groups'],
        ssm_state_size=spec['ssm_state_size'],
        mamba_d_conv=spec['conv_kernel'],
        mamba_chunk_size=spec['chunk_size'],
        mlp_hidden_act='relu2', layer_norm_epsilon=1e-5,
        use_mamba_kernels=False)
    config._attn_implementation = 'eager'
    modules = []
    for index, kind in enumerate(kinds):
        if kind == 'mamba2':
            modules.append(NemotronHMamba2Mixer(config, layer_idx=index).eval())
        elif kind == 'full_attention':
            modules.append(NemotronHAttention(config, layer_idx=index).eval())
        else:
            modules.append(NemotronHMLP(config).eval())
    norms = [NemotronHRMSNorm(spec['hidden_size'], eps=1e-5).eval()
             for _ in kinds]
    final_norm = NemotronHRMSNorm(spec['hidden_size'], eps=1e-5).eval()
    embedding = (np.random.default_rng(19).standard_normal(
        (spec['vocab_size'], spec['hidden_size'])).astype(np.float32) * 0.1)

    class Weights:
        def embedding_row(self, name, row, width):
            if name != 'backbone.embeddings.weight' or width != spec['hidden_size']:
                raise ValueError(f'unexpected embedding {name}')
            return embedding[row]

        def __call__(self, name, shape):
            parts = name.split('.')
            if (len(parts) == 5 and parts[:2] == ['backbone', 'layers']
                    and parts[3:] == ['norm', 'weight']):
                value = np.ones(shape, dtype=np.float32)
            elif name == 'backbone.norm_f.weight':
                value = np.ones(shape, dtype=np.float32)
            elif name == 'lm_head.weight':
                value = reference.parameter(name, shape)
            else:
                prefix = 'backbone.layers.'
                if not name.startswith(prefix):
                    raise ValueError(f'unexpected parameter {name}')
                index, component = name[len(prefix):].split('.mixer.', 1)
                target = modules[int(index)]
                for part in component.split('.'):
                    target = getattr(target, part)
                value = target.detach().numpy().astype(np.float32)
                if component.endswith('.weight') and value.ndim == 2:
                    value = value.T
            if value.shape != shape:
                raise ValueError(f'{name}: {value.shape} != {shape}')
            return value

    graph = stitch.build('nemotron3_nano_4b', prefill=prefill, decode_steps=2,
                         generation='teacher_forced',
                         specs={'nemotron3_nano_4b': spec})
    ids = np.array([[1 + i % 63 for i in range(prefill + 2)]], dtype=np.int32)
    values = graph_reference.execute(graph, {
        'prefill.token_ids': ids[:, :prefill],
        'decode0.token_ids': ids[:, prefill:prefill + 1],
        'decode1.token_ids': ids[:, prefill + 1:prefill + 2],
    }, weights=Weights())
    cache = NemotronHHybridDynamicCache(config, batch_size=1,
                                        dtype=torch.float32, device='cpu')
    discrepancies = []
    strict_failures = []
    accepted_failures = []
    total_elements = 0

    def record(name, got, expected):
        nonlocal total_elements
        actual = got.detach().numpy()
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError(f'{name}: invalid output')
        delta = np.abs(actual - expected)
        strict_limit = 2e-5 + 1e-4 * np.abs(expected)
        accepted_limit = 2e-4 + 1e-3 * np.abs(expected)
        total_elements += delta.size
        discrepancies.append((name, float(np.max(delta))))
        failed = np.argwhere(delta > strict_limit)
        if failed.size:
            index = tuple(int(x) for x in failed[0])
            strict_failures.append({
                'stage': name, 'count': len(failed), 'first_index': index,
                'first_actual': float(actual[index]),
                'first_reference': float(expected[index]),
                'first_abs_error': float(delta[index]),
            })
        accepted_count = int(np.count_nonzero(delta > accepted_limit))
        if accepted_count:
            accepted_failures.append({'stage': name, 'count': accepted_count})

    position = 0
    lm_weight = torch.from_numpy(reference.parameter(
        'lm_head.weight', (spec['hidden_size'], spec['vocab_size'])))
    with torch.no_grad():
        for phase, tokens in [('prefill', ids[:, :prefill]),
                              ('decode0', ids[:, prefill:prefill + 1]),
                              ('decode1', ids[:, prefill + 1:prefill + 2])]:
            hidden = torch.from_numpy(embedding[tokens])
            for index, kind in enumerate(kinds):
                prefix = f'{phase}.layer{index:02d}'
                normalized = norms[index](hidden)
                record(f'{prefix}.pre_norm', normalized, values[f'{prefix}.pre_norm'])
                if kind == 'mamba2':
                    mixed = modules[index](normalized, cache_params=cache)
                    stage = 'out_proj'
                elif kind == 'full_attention':
                    q = normalized.shape[1]
                    key_pos = torch.arange(position + q)
                    query_pos = torch.arange(position, position + q)
                    mask = torch.where(key_pos[None, :] <= query_pos[:, None],
                                       0.0, -1e9)[None, None]
                    if fault == 'noncausal_attention':
                        mask.zero_()
                    mixed, _ = modules[index](normalized,
                        past_key_values=cache,
                        cache_position=torch.arange(position, position + q),
                        attention_mask=mask)
                    stage = 'o_proj'
                else:
                    mixed = modules[index](normalized)
                    stage = 'down_proj'
                record(f'{prefix}.{stage}', mixed, values[f'{prefix}.{stage}'])
                hidden = hidden + mixed
                record(f'{prefix}.residual', hidden, values[f'{prefix}.residual'])
            final = final_norm(hidden)
            record(f'{phase}.final_norm', final, values[f'{phase}.final_norm'])
            logits = final[:, -1:, :] @ lm_weight
            record(f'{phase}.lm_head', logits, values[f'{phase}.lm_head'])
            position += tokens.shape[1]
            cache.has_previous_state = fault != 'drop_mamba_state'
    class DenseMLPAdapter(NemotronHMLP):
        def __init__(self, config, layer_idx):
            super().__init__(config)

    original_moe = modeling.MIXER_TYPES['moe']
    try:
        modeling.MIXER_TYPES['moe'] = DenseMLPAdapter
        model = NemotronHModel(config).eval()
    finally:
        modeling.MIXER_TYPES['moe'] = original_moe
    with torch.no_grad():
        model.embeddings.weight.copy_(torch.from_numpy(embedding))
        for index in range(layers):
            model.layers[index].norm = norms[index]
            model.layers[index].mixer = modules[index]
        model.norm_f = final_norm
        cache2 = NemotronHHybridDynamicCache(config, batch_size=1,
                                             dtype=torch.float32, device='cpu')
        position = 0
        for phase, tokens in [('prefill', ids[:, :prefill]),
                              ('decode0', ids[:, prefill:prefill + 1]),
                              ('decode1', ids[:, prefill + 1:prefill + 2])]:
            query_len = tokens.shape[1]
            response = model(input_ids=torch.from_numpy(tokens).long(),
                             past_key_values=cache2, use_cache=True,
                             cache_position=torch.arange(position, position + query_len),
                             attention_mask=torch.ones((1, position + query_len),
                                                       dtype=torch.long))
            expected = values[f'{phase}.final_norm']
            record(f'{phase}.model_forward_final_norm', response.last_hidden_state,
                   expected)
            model_logits = response.last_hidden_state[:, -1:, :] @ lm_weight
            record(f'{phase}.model_forward_lm_head', model_logits,
                   values[f'{phase}.lm_head'])
            position += query_len

    fixture = hashlib.sha256()
    fixture.update(embedding.tobytes())
    fixture.update(lm_weight.numpy().tobytes())
    for index, module in enumerate(modules):
        for name, value in sorted(module.state_dict().items()):
            fixture.update(f'{index}.{name}\0'.encode())
            fixture.update(value.detach().numpy().tobytes())
        fixture.update(f'{index}.pre_norm\0'.encode())
        fixture.update(norms[index].weight.detach().numpy().tobytes())
    fixture.update(b'final_norm\0')
    fixture.update(final_norm.weight.detach().numpy().tobytes())
    top = sorted(discrepancies, key=lambda item: item[1], reverse=True)[:20]
    result = {
        'schema': 'radiance.nemotron-reduced-network-reference.v1',
        'status': 'passed' if not accepted_failures else 'failed',
        'scope': 'same-weight reduced hybrid network with cached decode',
        'control_fault': fault,
        'pinned_model': pinned['name'],
        'pinned_config_source': pinned['source'],
        'pinned_config_sha256': pinned['source_sha256'],
        'layers': layers, 'prefill_tokens': prefill, 'decode_steps': 2,
        'layer_kinds': kinds,
        'graph_stages': len(graph.stages),
        'input_token_ids': ids.ravel().tolist(),
        'parameter_fixture_sha256': fixture.hexdigest(),
        'checked_tensor_outputs': len(discrepancies),
        'checked_tensor_values': total_elements,
        'max_abs_error': top[0][1],
        'largest_differences': [{'stage': name, 'max_abs_error': error}
                                for name, error in top],
        'final_logit_max_abs_error': {
            name: error for name, error in discrepancies
            if name.endswith('.lm_head')
        },
        'model_forward_final_hidden_max_abs_error': {
            name: error for name, error in discrepancies
            if name.endswith('.model_forward_final_norm')
        },
        'model_forward_final_logit_max_abs_error': {
            name: error for name, error in discrepancies
            if name.endswith('.model_forward_lm_head')
        },
        'strict_tolerance': {'rtol': 1e-4, 'atol': 2e-5,
                             'failing_tensors': len(strict_failures),
                             'failing_elements': sum(item['count'] for item in strict_failures),
                             'first_failure_by_tensor': strict_failures},
        'accepted_tolerance': {'rtol': 1e-3, 'atol': 2e-4,
                               'failing_tensors': len(accepted_failures),
                               'failing_elements': sum(item['count'] for item in accepted_failures),
                               'failures': accepted_failures},
        'reference_adapter': (
            'Transformers 5.3.0 NemotronHModel forward with its MoE slot '
            'mapped temporarily to NemotronHMLP for the pinned dense MLP blocks; '
            'the global registry is restored after construction.'
        ),
        'transformers_version': transformers.__version__,
        'torch_version': torch.__version__,
        'transformers_modeling_source_sha256': sha256(Path(modeling.__file__)),
        'transformers_config_source_sha256': sha256(Path(configuration.__file__)),
        'graph_reference_source_sha256': sha256(Path(graph_reference.__file__)),
        'graph_builder_source_sha256': sha256(Path(stitch.__file__)),
        'graph_math_source_sha256': sha256(Path(reference.__file__)),
        'model_specs_sha256': sha256(HERE / 'pr1-models.json'),
        'comparison_source_sha256': sha256(Path(__file__)),
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'checkpoint_weights': False,
        'pinned_upstream_network_equivalence': False,
        'device_execution': False,
        'performance_measurement': False,
    }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layers', type=int, default=42)
    parser.add_argument('--prefill', type=int, default=128)
    parser.add_argument('--fault', choices=('none', 'noncausal_attention',
                                            'drop_mamba_state'), default='none',
                        help='deliberate control fault for sensitivity checks')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.layers, args.prefill, args.fault)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], result['layers'], result['prefill_tokens'],
          result['checked_tensor_outputs'], result['max_abs_error'])
    if result['status'] != 'passed':
        raise SystemExit(1)
