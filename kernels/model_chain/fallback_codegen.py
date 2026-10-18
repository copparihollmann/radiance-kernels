"""Emit a stable call ABI around existing pure fallback operations."""
import re
import struct

# opcode, number of non-thread arguments, pointer argument indices, float indices
OPERATIONS = {
    'linear': (1,7,(0,1,2,3),()),
    'linear_i8': (2,7,(0,1,2,3),()),
    'linear_i4': (3,7,(0,1,2,3),()),
    'linear_f16': (4,6,(0,1,2),()),
    'layernorm': (5,7,(0,1,2,3),(6,)),
    'rmsnorm': (6,6,(0,1,2),(5,)),
    'patch_embed': (7,9,(0,1,2,3,4),()),
    'bidirectional_mha': (8,8,(0,1,2,3,4),()),
    'masked_gqa': (9,11,(0,1,2,3,4,5),()),
    'gelu_tanh_values': (10,3,(0,1),()),
    'silu_values': (11,3,(0,1),()),
    'residual': (12,4,(0,1,2),()),
    'bias_add': (13,5,(0,1,2),()),
    'copy_values': (14,3,(0,1),()),
    'scale_values': (15,4,(0,1),(3,)),
    'gated_activation': (16,5,(0,1,2),()),
    'euler_step': (17,5,(0,1,2),(4,)),
    'linear_except_columns': (18,8,(0,1,2),()),
}


def arguments(text):
    result, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif char == ',' and depth == 0:
            result.append(text[start:index].strip())
            start = index+1
        if depth < 0:
            raise ValueError('unbalanced pure helper arguments')
    if depth:
        raise ValueError('unbalanced pure helper arguments')
    return result+[text[start:].strip()]


def emit_call(body, name, context='(void*)v_arena', replica_strides=None, pointer_aliases=None):
    """Return descriptor declaration and call, or None for a non-pure operation.

    The original helper remains compiled in fallback_runtime.cpp. No arithmetic,
    reference values, operation order, or model dependency is replaced here.
    """
    match = re.fullmatch(r'(?:model_chain::)?(\w+)\((.*)\);?',body.strip(),re.S)
    if not match:
        return None
    if match[1] == 'mx_vpu_linear_partitioned':
        args = arguments(match[2])
        # Only the exact FP32 remainder is pure. Keep selected columns on the
        # original MX helper; all other storage formats retain their old body.
        if len(args) != 12 or args[4] != 'model_chain::mx_model::F32' or args[2] != 'nullptr':
            return None
        pure = 'linear_except_columns('+', '.join([args[0],args[1],args[3],*args[5:]])+');'
        declaration, call = emit_call(pure,name,context,replica_strides,pointer_aliases)
        return declaration, call+'\n  model_chain::mx_vpu_linear_columns('+', '.join(args)+');'
    if match[1] not in OPERATIONS:
        return None
    opcode, count, pointers, floats = OPERATIONS[match[1]]
    args = arguments(match[2])
    if len(args) != count+2 or args[-2:] != ['tid','tpb']:
        raise ValueError('pure fallback call does not have the expected thread ABI')
    words = ['0']*32
    words[0:2] = ['0x524b4631u',str(opcode)+'u']
    mask = 0
    for index,arg in enumerate(args[:-2]):
        slot = index+3
        if index in pointers:
            if arg in ('nullptr','0'):
                words[slot]='0'
                continue
            relative = re.search(r'v_arena\s*\+\s*(\d+)u?',arg)
            if relative:
                words[slot] = relative[1]+'u'
                mask |= 1<<slot
            elif pointer_aliases and arg in pointer_aliases:
                words[slot] = str(pointer_aliases[arg])+'u'
                mask |= 1<<slot
            elif replica_strides and arg in replica_strides:
                words[slot] = '(uintptr_t)'+arg
                mask |= (1<<slot) | 0x80000000
                words[slot+16] = str(replica_strides[arg])+'u'
            else:
                words[slot] = '(uintptr_t)('+arg+')'
        elif index in floats:
            bits = struct.unpack('<I',struct.pack('<f',float(arg.removesuffix('f'))))[0]
            words[slot]=str(bits)+'u'
        elif arg in ('true','false'):
            words[slot]='1u' if arg=='true' else '0u'
        else:
            # Dynamic tensor dimensions would require a different ABI. Keep
            # those stages on their original implementation rather than guess.
            words[slot]=str(int(arg.removesuffix('u'),0))+'u'
    words[2] = hex(mask)+'u'
    declaration = ('alignas(64) const __global model_chain::FallbackDescriptor '+name+' = {{'
                   + ','.join(words)+'}};\n')
    return declaration, f'model_chain::rk_fallback_call(&{name}, {context}, tid, tpb);'
