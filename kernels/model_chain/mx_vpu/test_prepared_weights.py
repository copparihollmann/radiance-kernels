"""Prepared byte codes must preserve independent FP8 normalization exactly."""
import struct
import unittest

import numpy as np

from prepare_weights import direct, encode_fp8, int8_lookup
from reference import _fp8


def decode(codes):
    raw = codes & 127
    exponent, fraction = raw >> 3, raw & 7
    value = np.where(exponent == 0, fraction.astype(np.float32)*np.float32(2**-9),
                     (np.float32(1)+fraction.astype(np.float32)/np.float32(8))*
                     np.exp2(exponent.astype(np.int32)-7).astype(np.float32))
    return np.copysign(value,np.where(codes & 128,np.float32(-1),np.float32(1)))


class PreparedWeightChecks(unittest.TestCase):
    def test_all_finite_bf16_rounding(self):
        values = (np.arange(65536,dtype=np.uint32)<<16).view(np.float32)
        values = values[np.isfinite(values)]
        np.testing.assert_array_equal(decode(encode_fp8(values)).view('u4'),_fp8(values).view('u4'))

    def test_int8_lookup_matches_direct_codes_across_exponents(self):
        rng = np.random.default_rng(9326)
        m,n = 289,300
        codes = np.empty((m,n),np.int8)
        for block,limit in enumerate((0,1,2,4,8,16,32,64,128,128)):
            rows = min(32,m-32*block)
            if rows <= 0:break
            codes[32*block:32*block+rows] = (0 if limit == 0 else
                rng.integers(-limit,limit,(rows,n),dtype=np.int8))
        scales = (rng.uniform(1,2,n)*np.exp2(rng.integers(-149,110,n))).astype(np.float32)
        scales[::3] *= -1
        scales[:2] = [np.float32(0),np.float32(-0.)]
        aux = int8_lookup(codes,scales)
        lo,eo = struct.unpack_from('<II',aux)
        table = np.frombuffer(aux[lo:eo],np.uint8).reshape(n,260)
        exponents = np.frombuffer(aux[eo:],np.int8).reshape(-1,n)
        actual = np.empty_like(codes,dtype=np.uint8)
        for block,e in enumerate(exponents):
            q = codes[block*32:(block+1)*32].astype(np.int32)
            band = table[:,0].view(np.int8).astype(np.int32)-e.astype(np.int32)
            base = table[np.arange(n)[None,:],4+(q&255)]
            one = (band[None,:] > 0)&((q == 1)|(q == -1))
            special = table[np.arange(n)[None,:],np.where(q == 1,2,3)]
            magnitudes = np.where(one,(special&127)+8*(band[None,:]-1),(base&127)+8*band[None,:])
            signs = np.where(one,special,base)&128
            actual[block*32:(block+1)*32] = np.where(q == 0,table[:,1],signs|magnitudes).astype(np.uint8)
        expected,_ = direct(codes.astype(np.float32)*scales)
        np.testing.assert_array_equal(actual,np.frombuffer(expected,np.uint8).reshape(m,n))

    def test_reject_nonfinite_weights(self):
        with self.assertRaises(ValueError):direct(np.array([[np.nan]],np.float32))


if __name__ == '__main__':unittest.main()
