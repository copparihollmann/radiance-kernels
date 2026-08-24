#include <stdint.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

#include "mxgemm.data.r5p.h"
static const uint8_t A_lut[64][16] = {0};
static const uint8_t B_lut[64][16] = {0};
static const uint8_t C_lut[64][16] = {0};
#include "mxgemm_lib.hpp"

extern "C" void vx_putchar(int c);
// Fully arithmetic — no string/table memory reads (which -O3 can vectorize into unaligned
// halfword loads that cyclotron forbids). Print the markers as explicit char constants.
static inline void _nib(unsigned d) { vx_putchar(d < 10 ? ('0' + d) : ('a' + d - 10)); }
static void _hex4(unsigned v) {
    for (int s = 12; s >= 0; s -= 4) _nib((v >> s) & 0xf);
}

constexpr GemmConfig CFG{
    .TILE_M = 128, .TILE_N = 128, .TILE_K = 128,
    .DATATYPE = GemmDatatype::FP8, .QUANT_OUTPUT = false,
};

void mxgemm_entry(void *a, uint32_t tid, uint32_t th, uint32_t tb) {
    auto Cg = reinterpret_cast<uint8_t *>(0x40000000);
    mxgemm<CFG>(CFG.TILE_M, CFG.TILE_N, CFG.TILE_K, Cg, tid, th, tb);
    if (tid == 0 && tb == 0) {
        gemmini_fence();
        volatile uint32_t *C32 = reinterpret_cast<volatile uint32_t *>(0x40000000);
        vx_putchar('M'); vx_putchar('X'); vx_putchar('O'); vx_putchar('\n');
        // R5's 16x16 result is the top-left corner of the 128-wide output; read 4-byte words
        // (two bf16 per word) so every load is 4-byte aligned (cyclotron forbids a load crossing a word).
        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 16; j++) {
                unsigned k = (unsigned)(i * 128 + j);
                unsigned w = C32[k >> 1];
                unsigned bf = (k & 1) ? (w >> 16) : (w & 0xffff);
                _hex4(bf & 0xffff); vx_putchar('\n');
            }
        vx_putchar('M'); vx_putchar('X'); vx_putchar('E'); vx_putchar('\n');
    }
}

int main() { mu_schedule(mxgemm_entry, nullptr, 2); return 0; }
