#include "mx_vpu/backend.hpp"
#include "mx_vpu/muon_bf16_add.hpp"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
static unsigned bits(float v){unsigned x;memcpy(&x,&v,4);return x;}
int main(){
 unsigned seed=123,checks=0;
 for(unsigned n:{0u,1u,7u,8u,9u,255u,256u,257u,1025u})for(unsigned width:{1u,7u,n?n:1}){
  std::vector<float>a(n),b(width),v(n,-333),s(n,-333);
  auto random=[&](){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return (float(int(seed%200001)-100000))/257.f;};
  for(auto& x:a)x=random();for(auto& x:b)x=random();
  for(unsigned i=0;i<n&&i<32;i++){unsigned x=0x3f800000u+i*0x8000u;memcpy(&a[i],&x,4);}
  model_chain::mx_vpu_add(a.data(),b.data(),v.data(),n,width,0,64);
  model_chain::mx_muon_bf16_add(a.data(),b.data(),s.data(),n,width,0,64);
  for(unsigned i=0;i<n;i++){if(bits(v[i])!=bits(s[i])){fprintf(stderr,"mismatch n=%u width=%u i=%u %x %x\n",n,width,i,bits(v[i]),bits(s[i]));return 1;}checks++;}
  auto inplace=a;model_chain::mx_muon_bf16_add(inplace.data(),b.data(),inplace.data(),n,width,0,64);
  for(unsigned i=0;i<n;i++)assert(bits(inplace[i])==bits(v[i]));
  model_chain::mx_muon_bf16_add(a.data(),b.data(),s.data(),n,width,1,64);
  for(unsigned i=0;i<n;i++)assert(bits(s[i])==bits(v[i]));
 }
 printf("BF16_ADD_EQUIVALENCE checked=%u cases=27 tails_broadcast_inplace_nonowner=pass\n",checks);
}
