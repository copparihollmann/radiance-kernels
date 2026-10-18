import ctypes as C
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mx_vpu import fp32_reference as ref


class FallbackReferenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        directory = Path(cls.temp.name)
        source = directory/"fallback.cpp"
        source.write_text('''#include "pipeline_smolvla.hpp"
extern "C" {
void linear_(float*x,float*w,float*y,unsigned m,unsigned k,unsigned n) {
 model_chain::linear(x,w,nullptr,y,m,k,n,0,1); }
void norm_(float*x,float*g,float*y,unsigned m,unsigned k,float e) {
 model_chain::rmsnorm(x,g,y,m,k,e,0,1); }
void gate_(float*x,float*u,float*y,unsigned n,unsigned gelu) {
 model_chain::gated_activation(x,u,y,n,gelu,0,1); }
void cap_(float*x,float*y,unsigned n,float cap) {
 model_chain::softcap(x,y,n,cap,0,1); }
void layer_(float*x,float*g,float*b,float*y,unsigned m,unsigned k,float e) {
 model_chain::layernorm(x,g,b,y,m,k,e,0,1); }
void patch_(float*x,float*w,float*b,float*p,float*y) {
 model_chain::patch_embed(x,w,b,p,y,8,2,3,7,0,1); }
void vision_(float*q,float*k,float*v,float*y,float*s) {
 model_chain::bidirectional_mha(q,k,v,y,s,5,2,8,0,1); }
void gelu_(float*x,float*y,unsigned n) {
 model_chain::gelu_tanh_values(x,y,n,0,1); }
void attention_(float*q,float*k,float*v,float*y,unsigned b,unsigned qt,unsigned kt,
 unsigned qh,unsigned kh,unsigned d,unsigned start,unsigned window,float cap) {
 model_chain::causal_gqa(q,k,v,y,b,qt,kt,qh,kh,d,start,window,cap,0,1); }
}
''')
        subprocess.run(["g++", "-O2", "-std=c++17", "-ffp-contract=off", "-shared", "-fPIC",
                        "-I"+str(HERE.parent), str(source), "-o", str(directory/"fallback.so")], check=True)
        cls.lib = C.CDLL(str(directory/"fallback.so"))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def p(value):
        return value.ctypes.data_as(C.POINTER(C.c_float))

    def test_sequential_dot_and_norm(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(2,127)).astype(np.float32)
        w = rng.normal(size=(127,19)).astype(np.float32)
        y = np.zeros((2,19),np.float32)
        self.lib.linear_(self.p(x),self.p(w),self.p(y),2,127,19)
        np.testing.assert_array_equal(y,ref.linear(x,w))
        gamma = rng.normal(size=127).astype(np.float32)
        y = np.zeros_like(x)
        self.lib.norm_(self.p(x),self.p(gamma),self.p(y),2,127,C.c_float(1e-6))
        np.testing.assert_array_equal(y,ref.rmsnorm(x,gamma,1e-6))

    def test_polynomial_activations(self):
        x = np.linspace(-100,100,1001,dtype=np.float32)
        up = np.full_like(x,0.37)
        for gelu in (0,1):
            y = np.zeros_like(x)
            self.lib.gate_(self.p(x),self.p(up),self.p(y),len(x),gelu)
            np.testing.assert_array_equal(y,ref.gated(x,up,"gelu_tanh" if gelu else "silu"))
        self.lib.cap_(self.p(x),self.p(y),len(x),C.c_float(30))
        np.testing.assert_array_equal(y,ref.softcap(x,30))

    def test_causal_grouped_attention_window_and_softcap(self):
        rng = np.random.default_rng(19)
        q = rng.normal(size=(2,3,4,8)).astype(np.float32)
        k = rng.normal(size=(2,5,2,8)).astype(np.float32)
        v = rng.normal(size=k.shape).astype(np.float32)
        for window,cap in ((0,0),(3,5)):
            out = np.zeros_like(q)
            self.lib.attention_(self.p(q),self.p(k),self.p(v),self.p(out),2,3,5,4,2,8,2,window,C.c_float(cap))
            attrs=dict(query_start=2,window=window or None,softcap=cap or None)
            np.testing.assert_array_equal(out.reshape(2,3,32),ref.attention(q,k,v,attrs))

    def test_vision_arithmetic_matches_declared_fp32_order(self):
        rng = np.random.default_rng(290)
        x = rng.normal(size=(3,17)).astype(np.float32)
        gamma, bias = rng.normal(size=(2,17)).astype(np.float32)
        y = np.empty_like(x)
        self.lib.layer_(self.p(x),self.p(gamma),self.p(bias),self.p(y),3,17,C.c_float(1e-6))
        np.testing.assert_array_equal(y,ref.layernorm(x,gamma,bias,1e-6))
        self.lib.gelu_(self.p(x),self.p(y),x.size)
        np.testing.assert_array_equal(y,ref.gelu_tanh(x))
        image = rng.normal(size=(1,3,8,8)).astype(np.float32)
        weight = rng.normal(size=(7,3,2,2)).astype(np.float32)
        bias = rng.normal(size=7).astype(np.float32)
        position = rng.normal(size=(16,7)).astype(np.float32)
        y = np.empty_like(position)
        self.lib.patch_(*(self.p(a) for a in (image,weight,bias,position,y)))
        expected = ref.patch_convolution(image,weight,bias).reshape(1,7,16).transpose(0,2,1)
        np.testing.assert_array_equal(y,expected[0]+position)
        q,k,v = rng.normal(size=(3,1,5,2,8)).astype(np.float32)
        y = np.empty((1,5,16),np.float32)
        scratch = np.empty(5,np.float32)
        self.lib.vision_(*(self.p(a) for a in (q,k,v,y,scratch)))
        np.testing.assert_array_equal(y,ref.bidirectional_mha(q,k,v))


if __name__ == "__main__":
    unittest.main()
