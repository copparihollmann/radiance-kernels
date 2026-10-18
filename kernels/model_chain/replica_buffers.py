"""Plan disjoint guarded activation/scratch regions below the program image."""

from split_decoder_weights import image_segments


def plan_replicas(arena_bytes, scratch_bytes, replicas=2, base=0x02000000):
    if replicas not in (1, 2) or arena_bytes <= 0 or scratch_bytes <= 0:
        raise ValueError('replicas require positive activation and scratch sizes')
    if base < 0x100000 or base % 4096:
        raise ValueError('replica pool must be page aligned above reserved low memory')
    arena_bytes = (arena_bytes + 63) // 64 * 64
    body = (arena_bytes + scratch_bytes + 4095) // 4096 * 4096
    stride = body + 8192
    size = stride * replicas
    if base + size > 0x10000000:
        raise ValueError('replica buffers must fit below the program region')
    return dict(base=base, bytes=size, replicas=replicas, stride=stride,
                guard_bytes=4096, arena_bytes=arena_bytes,
                scratch_offset=arena_bytes, scratch_bytes=scratch_bytes,
                body_bytes=body)


def check_images(plan, images):
    start, end = plan['base'], plan['base'] + plan['bytes']
    for image in images:
        if image is None:
            continue
        for segment in image_segments(image):
            low = segment['gpu_base_address']
            high = low + segment['image_size_bytes']
            if start < high and end > low:
                raise ValueError('replica pool overlaps a loaded image')


def reserved_image(plan):
    # verify_image_placement consumes ranges; this region is zeroed RAM, not
    # another checkpoint file to load. Include its guards and scratch space.
    return dict(segments=[dict(gpu_base_address=plan['base'],
                               image_size_bytes=plan['bytes'])])


def stage_buffers(plan):
    return ('  __global unsigned char* v_arena = (__global unsigned char*)context;\n'
            '  __global float* v_attention_scratch = (__global float*)'
            f'(v_arena + {plan["scratch_offset"]}u);\n')
