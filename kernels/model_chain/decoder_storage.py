"""Reuse decoder activation storage after its last graph consumer.

Stage functions keep their existing tensor pointers and replica strides. Assembly
aliases bind them to a shared arena; checkpoint weights and arithmetic are unchanged.
Only graph outputs and explicitly retained tensors remain available after completion.
"""
import copy
import math
import re

from plan_buffers import plan


def replicated_plan(graph, replicas, preserve=()):
    if replicas not in (1, 2, 4):
        raise ValueError('expected one, two, or four replicas')
    graph = copy.deepcopy(graph)
    if set(preserve) - set(graph.tensors):
        raise ValueError('retained tensor is absent from graph')
    graph.outputs = list(dict.fromkeys([*graph.outputs, *preserve]))
    strides = {}
    for name, tensor in graph.tensors.items():
        elements = math.prod(tensor['shape'])
        strides[name] = (elements + 15) // 16 * 16
        tensor['shape'] = [replicas, strides[name]]
    result = plan(graph)
    result.update(replicas=replicas, retained_tensors=graph.outputs,
                  scope='Activation placement only; complete model execution and numerical checks are separate.')
    for name, allocation in result['allocation'].items():
        allocation['replica_stride_elements'] = strides[name]
    return result


def pooled_source(source, layout):
    if 'rk_activation_arena' in source:
        raise ValueError('source already has an activation arena')
    allocation = {'v_' + name.replace('.', '_'): item
                  for name, item in layout['allocation'].items() if item['birth'] >= 0}
    found = set()
    def replace(match):
        qualifier, dtype, name, elements, values = match.groups()
        qualifier = qualifier or ''
        if name not in allocation:
            return match[0]
        item = allocation[name]
        if (int(elements) * 4 != item['size_bytes'] or
                values.strip() not in ('0', '0.0f')):
            raise ValueError('activation shape or initialization changed: ' + name)
        found.add(name)
        declaration = f'extern {qualifier}{dtype} {name}[];\n'
        # Same object contains the arena, so aliases are defined even across
        # separate stage objects; .size preserves normal ELF buffer inspection.
        assembly = (f'.global {name}\\n.type {name}, @object\\n'
                    f'.set {name}, rk_activation_arena+{item["offset_bytes"]}\\n'
                    f'.size {name}, {item["size_bytes"]}\\n')
        return declaration + '__asm__("' + assembly + '");'
    result = re.sub(r'alignas\(64\) (__global )?(float|int32_t) (\w+)\[(\d+)\] = \{([^}]*)\};',
                    replace, source)
    if found != set(allocation):
        raise ValueError('missing stage activation declarations: ' + str(sorted(set(allocation) - found)))
    qualifier = '__global ' if '__global ' in source else ''
    # Prefix after includes so uint8 storage needs no additional dependency.
    return (('#include <mu_intrinsics.h>\n' if qualifier else '')
            + f'alignas(64) {qualifier}unsigned char rk_activation_arena[{layout["arena_bytes"]}] = {{0}};\n'
            + result)
