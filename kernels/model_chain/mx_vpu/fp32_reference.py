"""Independent FP32 fallback math with the device's defined operation order.

BLAS reductions/libm may differ by a few FP32 ulps. Such differences can cross
an FP8/BF16 rounding boundary. This reference uses sequential FP32 dot products
and the existing device polynomial, without calling the C++ implementation.
Compile this numerical variant with FP contraction disabled on host and device.
"""
import numpy as np

F = np.float32


def linear(x, weight):
    result = np.zeros((*x.shape[:-1], weight.shape[1]), np.float32)
    for k in range(weight.shape[0]):
        result += x[..., k, None] * weight[k]
    return result


def rmsnorm(x, gamma, epsilon):
    total = np.zeros(x.shape[:-1], np.float32)
    for k in range(x.shape[-1]):
        total += x[..., k] * x[..., k]
    scale = F(1) / np.sqrt(total/F(x.shape[-1])+F(epsilon))
    return (x*scale[..., None])*gamma


def exponential(value):
    value = np.asarray(value, np.float32)
    clipped = np.minimum(np.maximum(value, F(-80)), F(80))
    z = clipped * F(1.4426950408889634)
    n = np.trunc(z + np.where(z >= 0, F(0.5), F(-0.5))).astype(np.int32)
    r = clipped - n.astype(np.float32)*F(0.6931471805599453)
    p = F(1)/F(5040)
    for coefficient in (F(1)/F(720), F(1)/F(120), F(1)/F(24), F(1)/F(6), F(0.5), F(1), F(1)):
        p = coefficient + r*p
    bits = ((n+127).astype(np.uint32) << 23)
    return np.where(value < F(-80), F(0), p*bits.view(np.float32))


def tanh(value):
    e = exponential(F(-2)*np.abs(value))
    magnitude = (F(1)-e)/(F(1)+e)
    return np.where(value >= 0, magnitude, -magnitude)


def gated(gate, up, function):
    if function == "gelu_tanh":
        argument = F(0.7978845608028654)*(gate + ((F(0.044715)*gate)*gate)*gate)
        activation = (F(0.5)*gate)*(F(1)+tanh(argument))
    else:
        activation = gate/(F(1)+exponential(-gate))
    return activation*up


def softcap(value, cap):
    return F(cap)*tanh(value/F(cap))


def attention(q, k, v, attrs):
    batch, queries, heads, width = q.shape
    keys, kvheads = k.shape[1:3]
    output = np.empty_like(q)
    allowed = np.arange(keys)[None, :] <= attrs["query_start"]+np.arange(queries)[:, None]
    if attrs["window"]:
        allowed &= np.arange(keys)[None, :] > attrs["query_start"]+np.arange(queries)[:, None]-attrs["window"]
    for head in range(heads):
        kv = head//(heads//kvheads)
        score = np.zeros((batch, queries, keys), np.float32)
        for d in range(width):
            score += q[:, :, head, d, None]*k[:, None, :, kv, d]
        score /= np.sqrt(F(width))
        if attrs["softcap"]:
            score = softcap(score, attrs["softcap"])
        maximum = np.full((batch, queries), F(-3.402823466e38), np.float32)
        for key in range(keys):
            maximum = np.where(allowed[:, key] & (score[:, :, key] > maximum), score[:, :, key], maximum)
        exp = exponential(score-maximum[..., None])
        denominator = np.zeros_like(maximum)
        for key in range(keys):
            denominator += np.where(allowed[:, key], exp[:, :, key], F(0))
        result = np.zeros((batch, queries, width), np.float32)
        for key in range(keys):
            weight = np.where(allowed[:, key], exp[:, :, key]/denominator, F(0))
            result += weight[..., None]*v[:, key, kv, None, :]
        output[:, :, head] = result
    return output.reshape(batch, queries, heads*width)


FALLBACKS = dict(linear=linear, rmsnorm=rmsnorm, attention=attention,
                 gated_activation=gated, softcap=softcap)
