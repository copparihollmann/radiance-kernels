#!/usr/bin/env python3
# Validate a SCALAR (C-portable) bf16->fp6-code + fixed-point finder against the
# reference tensor implementations, over ALL 65536 bf16 bit patterns.
import sys
sys.path.insert(0, "/scratch/agustin/projects/radiance-kernels/lib/mxgemmini")
import torch, struct

# ---- reference (from lut_mapping_demo, imported without running its script body) ----
import importlib.util
spec = importlib.util.spec_from_file_location("lmd", "/scratch/agustin/projects/radiance-kernels/lib/mxgemmini/lut_mapping_demo.py")
# We can't exec the module (it runs a big script). Re-implement the reference calls by
# copying the two functions' logic via source is messy; instead use the tensor fns by
# pasting minimal copies:
def _bf16_to_e4m2_rne(x):
    bf16 = x.to(torch.bfloat16); bits = bf16.view(torch.int16).to(torch.int32) & 0xFFFF
    sign_bit=(bits>>15)&1; E=(bits>>7)&0xFF; M=bits&0x7F
    sign_f=torch.where(sign_bit==0,torch.ones_like(x),-torch.ones_like(x)); e=E-127
    q=(M>>5)&3; r=((M>>4)&1).bool(); sticky=(M&0xF).ne(0); lsb=((M>>5)&1).bool()
    round_up=r&(sticky|lsb); sig_rounded=q+round_up.to(torch.int32)
    carry=sig_rounded>=4; mant_out=torch.where(carry,torch.zeros_like(sig_rounded),sig_rounded)
    exp_out=torch.where(carry,e+1,e); is_overflow=exp_out>7; safe_exp=exp_out.float().clamp(-10,10)
    val_normal=sign_f*(1.0+mant_out.float()*0.25)*torch.pow(torch.full_like(x,2.0),safe_exp)
    val_normal=torch.where(is_overflow,sign_f*float('inf'),val_normal)
    quantum=2.0**-8
    is_e7=(e==-7)&(E>=1); k7=torch.where(M<=32,torch.full_like(M,2),torch.where(M<=95,torch.full_like(M,3),torch.full_like(M,4))); val_e7=sign_f*k7.float()*quantum
    is_e8=(e==-8)&(E>=1); k8=torch.where(M<64,torch.ones_like(M),torch.full_like(M,2)); val_e8=sign_f*k8.float()*quantum
    is_e9=(e==-9)&(E>=1); k9=torch.where(M==0,torch.zeros_like(M),torch.ones_like(M)); val_e9=sign_f*k9.float()*quantum
    result=val_normal; result=torch.where(is_e7,val_e7,result); result=torch.where(is_e8,val_e8,result); result=torch.where(is_e9,val_e9,result)
    result=torch.where(((e<=-10)&(E>=1))|(E==0),torch.zeros_like(x),result)
    is_nan=(E==255)&(M!=0); is_inf=(E==255)&(M==0)
    result=torch.where(is_inf,sign_f*float('inf'),result); result=torch.where(is_nan,torch.full_like(x,float('nan')),result)
    return result
def _e4m2_to_fp6(x):
    sign=x.sign(); ax=x.abs()
    mapToZero=(ax<=0.0546875); mapToMax=(ax>=32.0)|~torch.isfinite(ax)
    mapToSubnorm=(ax>=0.0625)&(ax<=0.21875)
    sub_val=torch.where(ax<=0.078125,torch.full_like(ax,0.0625),torch.where(ax<=0.15625,torch.full_like(ax,0.125),torch.full_like(ax,0.1875)))
    out=x.clone(); out=torch.where(mapToZero,torch.zeros_like(x),out); out=torch.where(mapToMax,sign*28.0,out); out=torch.where(mapToSubnorm,sign*sub_val,out)
    return out
def hw_bf16_to_fp6(x): return _e4m2_to_fp6(_bf16_to_e4m2_rne(x))
import math as _math
def _fp6_value_to_code(v):
    if v==0.0: return 0
    s=1 if v<0 else 0; av=abs(v); e_bits,m_bits,bias=3,2,3; emin=1-bias
    if av<2.0**emin:
        quantum=2.0**(emin-m_bits); mant=int(round(av/quantum)); return (s<<(e_bits+m_bits))|max(0,min(mant,(1<<m_bits)-1))
    E=int(_math.floor(_math.log2(av))); base=2.0**E; mant=int(round((av-base)/(base/4)))
    if mant>=4: mant=0; E+=1
    biased=min(E+bias,(1<<e_bits)-1); mant=min(mant,(1<<m_bits)-1); return (s<<(e_bits+m_bits))|(biased<<m_bits)|mant

# reference: bf16 bits -> fp6 code
def ref_bf16bits_to_fp6code(bits):
    x = torch.tensor([struct.unpack('<f', struct.pack('<I', bits<<16))[0]], dtype=torch.float32)
    v = float(hw_bf16_to_fp6(x)[0])
    if v!=v: return 0  # NaN -> code 0 (we won't feed NaN)
    return _fp6_value_to_code(v)

# ---------------- SCALAR C-portable candidate ----------------
def c_bf16_to_fp6_code(bits):
    sign = (bits>>15)&1
    E = (bits>>7)&0xFF
    Mm = bits & 0x7F
    if E==0: return 0                 # zero/subnormal bf16 -> 0
    if E==255: return (sign<<5)|(0b111<<2)|0b11  # inf/nan -> max mag (28.0) code; we avoid NaN inputs
    e = E-127
    # --- bf16 -> E4M2 (RNE), produce (is_zero, |val| category) as fp6 directly ---
    # We compute the fp6 code by first getting E4M2 magnitude value, then E4M2->fp6.
    # Represent E4M2 result as (kind, absval) using integer logic:
    q = (Mm>>5)&3; r=(Mm>>4)&1; sticky = 1 if (Mm&0xF)!=0 else 0; lsb=(Mm>>5)&1
    round_up = r & (sticky|lsb); sig = q+round_up      # 0..4
    carry = 1 if sig>=4 else 0
    mant_e = 0 if carry else sig                        # E4M2 mantissa (0..3)
    exp_e = (e+1) if carry else e                       # E4M2 unbiased exp
    quantum = 2.0**-8
    if e>=-6:
        if exp_e>7:
            av = float('inf')
        else:
            av = (1.0 + mant_e*0.25)*(2.0**exp_e)
    elif e==-7:
        k = 2 if Mm<=32 else (3 if Mm<=95 else 4); av = k*quantum
    elif e==-8:
        k = 1 if Mm<64 else 2; av = k*quantum
    elif e==-9:
        k = 0 if Mm==0 else 1; av = k*quantum
    else:
        av = 0.0
    # E4M2 abs -> fp6 abs
    if av==0.0 or av<=0.0546875:
        return 0
    if (av>=32.0) or (av==float('inf')):
        fp6_abs = 28.0
    elif 0.0625<=av<=0.21875:
        fp6_abs = 0.0625 if av<=0.078125 else (0.125 if av<=0.15625 else 0.1875)
    else:
        fp6_abs = av
    # fp6 abs value -> code (subnormal-aware), matching _fp6_value_to_code
    if fp6_abs==0.0: return 0
    # subnormal region: av < 2^emin = 0.25
    if fp6_abs < 0.25:
        # quantum = 2^(emin-mbits)=2^-4=0.0625
        mant = int(round(fp6_abs/0.0625)); mant = max(0,min(mant,3))
        code = mant
    else:
        Efp = int(_math.floor(_math.log2(fp6_abs))); base=2.0**Efp
        mant = int(round((fp6_abs-base)/(base/4)))
        if mant>=4: mant=0; Efp+=1
        biased = min(Efp+3,7); mant=min(mant,3)
        code = (biased<<2)|mant
    return (sign<<5)|code

# ---- compare over all bf16 patterns except inf/nan (E==255) ----
mism=0; examples=[]
for bits in range(65536):
    E=(bits>>7)&0xFF
    if E==255: continue
    ref = ref_bf16bits_to_fp6code(bits)
    got = c_bf16_to_fp6_code(bits)
    if ref!=got:
        mism+=1
        if len(examples)<12: examples.append((bits,ref,got))
print("mismatches (excl inf/nan):", mism, "/ 65280")
for b,rf,g in examples:
    x=struct.unpack('<f',struct.pack('<I',b<<16))[0]
    print(f"  bits=0x{b:04x} x={x:g} ref=0x{rf:02x} got=0x{g:02x}")

# ---------------- INTEGER-ONLY (libm-free) candidate for device ----------------
def int_bf16_to_fp6_code(bits):
    sign=(bits>>15)&1; E=(bits>>7)&0xFF; Mm=bits&0x7F
    if E==0: return 0
    if E==255: return (sign<<5)|0x1F
    e=E-127
    if e<=-7: return 0
    q=(Mm>>5)&3; r=(Mm>>4)&1; sticky=1 if (Mm&0xF) else 0; lsb=(Mm>>5)&1
    sig=q+(r&(sticky|lsb)); carry=1 if sig>=4 else 0
    mant_e=0 if carry else sig; exp_e=(e+1) if carry else e
    if exp_e<=-5: return 0
    if exp_e>=5:  return (sign<<5)|0x1F
    if exp_e==-4: return (sign<<5)|(1 if mant_e<=1 else 2)
    if exp_e==-3: return (sign<<5)|(2 if mant_e<=1 else 3)
    return (sign<<5)|(((exp_e+3)&7)<<2)|mant_e

mism2=0; ex2=[]
for bits in range(65536):
    if (bits>>7)&0xFF==255: continue
    rf=ref_bf16bits_to_fp6code(bits); g=int_bf16_to_fp6_code(bits)
    if rf!=g:
        mism2+=1
        if len(ex2)<12: ex2.append((bits,rf,g))
print("INTEGER version mismatches:", mism2,"/ 65280")
for b,rf,g in ex2:
    import struct as _s; x=_s.unpack('<f',_s.pack('<I',b<<16))[0]
    print(f"  bits=0x{b:04x} x={x:g} ref=0x{rf:02x} got=0x{g:02x}")
