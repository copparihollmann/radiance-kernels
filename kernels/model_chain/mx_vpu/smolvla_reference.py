"""Independent arithmetic policy for the pinned SmolVLA vision partition.

These NumPy operations define FP32 reduction/activation order before FP8 rounding.
The original checkpoint comparison always uses the separately frozen Torch run.
No C++ kernels or deployment outputs are used to produce this reference.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from mx_vpu import fp32_reference as fp32, reference

VISION_POLICY = 'sequential_fp32_kahan_layernorm_polynomial_vision_v1'


def qualify_result(result, original_path, module, start, count, fp32_vision):
    original = json.loads(original_path.read_text())
    if (original['checkpoint_weight_sha256'] != result['checkpoint_weight_sha256'] or
            original['input_sha256'] != result['input_sha256']):
        raise ValueError('original golden uses different checkpoint or inputs')
    actual = np.asarray(result['output_values'], np.float32)
    baseline = np.asarray(original['output_values'], np.float32)
    if actual.shape != baseline.shape:
        raise ValueError('original golden has a different action shape')
    delta = np.abs(actual-baseline)
    failures = int(np.count_nonzero(delta > 1e-3+1e-3*np.abs(baseline)))
    here = Path(__file__).resolve().parent
    result.update(execution_backend='rakanic_mx_vpu', projection_module=module,
                  projection_columns=dict(start=start,count=count),
                  vision_arithmetic_policy=VISION_POLICY if fp32_vision else 'upstream_torch',
                  arithmetic_reference_sha256={name:hashlib.sha256((here/name).read_bytes()).hexdigest()
                                               for name in ('smolvla_reference.py','fp32_reference.py')},
                  format_reference_sha256=hashlib.sha256((here/'reference.py').read_bytes()).hexdigest(),
                  format_reference_produced=True,
                  original_action_comparison=dict(passed=failures == 0,failed_elements=failures,
                      rtol=1e-3,atol=1e-3,max_abs_error=float(delta.max()),
                      original_golden_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest()))
    return result


def projection_function(start=0, count=None, fp32_vision=False):
    import torch

    def project(value, weight, bias):
        if value.device.type != 'cpu' or value.dtype != torch.float32:
            raise ValueError('format reference requires FP32 CPU tensors')
        end = weight.shape[0] if count is None else start+count
        if start < 0 or not start < end <= weight.shape[0]:
            raise ValueError('projection partition is outside the weight matrix')
        x, w = value.detach().numpy(), weight.detach().numpy().T
        selected = reference.linear(x, w[:, start:end])
        if bias is not None:
            selected = reference.add(selected, bias[start:end].detach().numpy())
        if start == 0 and end == weight.shape[0]:
            return torch.from_numpy(selected)
        if fp32_vision:
            # Only the fallback columns use FP32. Selected columns are not
            # redundantly evaluated with a different arithmetic policy.
            result = np.empty((*x.shape[:-1], w.shape[1]), np.float32)
            for band in (slice(0, start), slice(end, w.shape[1])):
                if band.start == band.stop:
                    continue
                result[..., band] = fp32.linear(x, w[:, band])
                if bias is not None:
                    result[..., band] += bias[band].detach().numpy()
            result[..., start:end] = selected
            return torch.from_numpy(result)
        output = torch.nn.functional.linear(value, weight, bias)
        output[..., start:end] = torch.from_numpy(selected)
        return output

    return project


def install_vision_arithmetic(vision):
    """Keep upstream graph/hooks; replace only declared vision primitives."""
    import torch

    def array(tensor):
        if tensor.device.type != 'cpu' or tensor.dtype != torch.float32:
            raise ValueError('vision arithmetic requires FP32 CPU tensors')
        return tensor.detach().numpy()

    for module in vision.modules():
        if isinstance(module, torch.nn.Linear):
            def forward(value, _module=module):
                result = fp32.linear(array(value), array(_module.weight).T)
                if _module.bias is not None:
                    result += array(_module.bias)
                return torch.from_numpy(result)
            module.forward = forward
        elif isinstance(module, torch.nn.LayerNorm):
            def forward(value, _module=module):
                return torch.from_numpy(fp32.layernorm(array(value), array(_module.weight),
                                                       array(_module.bias), _module.eps))
            module.forward = forward
    patch = vision.embeddings.patch_embedding
    if (patch.stride != patch.kernel_size or patch.padding not in ((0, 0), 'valid') or
            patch.dilation != (1, 1) or patch.groups != 1):
        raise ValueError('unsupported vision patch convolution')
    patch.forward = lambda value: torch.from_numpy(fp32.patch_convolution(
        array(value), array(patch.weight), array(patch.bias)))
    for layer in vision.encoder.layers:
        if layer.mlp.config.hidden_act != 'gelu_pytorch_tanh':
            raise ValueError('unsupported vision activation')
        layer.mlp.activation_fn.forward = lambda value: torch.from_numpy(fp32.gelu_tanh(array(value)))
        module = layer.self_attn

        def attention(hidden_states, attention_mask=None, _module=module, **kwargs):
            if attention_mask is not None:
                allowed = (attention_mask.all() if attention_mask.dtype == torch.bool
                           else (attention_mask == 0).all())
                if not allowed:
                    raise ValueError('vision reference requires the pinned unmasked images')
            if _module.training and _module.dropout:
                raise ValueError('vision reference requires inference without dropout')
            shape = (*hidden_states.shape[:-1], _module.num_heads, _module.head_dim)
            q, k, v = [array(projection(hidden_states)).reshape(shape)
                       for projection in (_module.q_proj, _module.k_proj, _module.v_proj)]
            attended = torch.from_numpy(fp32.bidirectional_mha(q, k, v))
            return _module.out_proj(attended), None

        module.forward = attention
