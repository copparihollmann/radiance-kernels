#include <stdint.h>
#include <float.h>

static inline void kernel_body(
    void* raw_arg,
    uint32_t tid_in_threadblock,
    uint32_t threads_per_threadblock,
    uint32_t threadblock_id
) {
    (void)threadblock_id;
    (void)threads_per_threadblock;
    auto* a = reinterpret_cast<KernelArgs*>(raw_arg);

    const uint32_t lane_id = tid_in_threadblock % 16u;
    const uint32_t warp_id = tid_in_threadblock / 16u;

    const uint32_t cols = a->cols;

    // SMEM layout:
    // Each warp gets 16 floats (64 bytes) for reduction scratch.
    // 8 warps * 64 bytes = 512 bytes total — well within 128 KiB.
    const uint32_t SMEM_FLOATS_PER_WARP = 16u;
    const uint32_t smem_warp_base = warp_id * SMEM_FLOATS_PER_WARP * 4u;
    // Lane's slot in SMEM for this warp
    const uint32_t smem_lane_addr = smem_warp_base + lane_id * 4u;

    // Each warp processes one row at a time, striding by 8 warps
    for (uint32_t row = warp_id; row < a->rows; row += 8u) {
        const uint32_t global_base = row * cols;

        // ----------------------------------------------------------------
        // Phase 1: Find row max using a loop over column tiles
        // Each lane handles columns: lane_id, lane_id+16, lane_id+32, ...
        // ----------------------------------------------------------------
        float lane_max = -3.402823466e+38f; // -FLT_MAX

        // Unroll by 4 tiles (64 columns) per iteration for ILP
        uint32_t col_tile = 0u;
        // Process 4 tiles at a time
        for (; col_tile + 64u <= cols; col_tile += 64u) {
            uint32_t base0 = global_base + col_tile + lane_id;
            uint32_t base1 = global_base + col_tile + 16u + lane_id;
            uint32_t base2 = global_base + col_tile + 32u + lane_id;
            uint32_t base3 = global_base + col_tile + 48u + lane_id;

            uint32_t bits0, bits1, bits2, bits3;
            asm volatile("lw.global %0, 0(%1)" : "=r"(bits0) : "r"(reinterpret_cast<uint32_t>(&a->in[base0])) : "memory");
            asm volatile("lw.global %0, 0(%1)" : "=r"(bits1) : "r"(reinterpret_cast<uint32_t>(&a->in[base1])) : "memory");
            asm volatile("lw.global %0, 0(%1)" : "=r"(bits2) : "r"(reinterpret_cast<uint32_t>(&a->in[base2])) : "memory");
            asm volatile("lw.global %0, 0(%1)" : "=r"(bits3) : "r"(reinterpret_cast<uint32_t>(&a->in[base3])) : "memory");

            float f0 = __builtin_bit_cast(float, bits0);
            float f1 = __builtin_bit_cast(float, bits1);
            float f2 = __builtin_bit_cast(float, bits2);
            float f3 = __builtin_bit_cast(float, bits3);

            if (f0 > lane_max) lane_max = f0;
            if (f1 > lane_max) lane_max = f1;
            if (f2 > lane_max) lane_max = f2;
            if (f3 > lane_max) lane_max = f3;
        }
        // Process 2 remaining tiles
        for (; col_tile + 32u <= cols; col_tile += 32u) {
            uint32_t base0 = global_base + col_tile + lane_id;
            uint32_t base1 = global_base + col_tile + 16u + lane_id;

            uint32_t bits0, bits1;
            asm volatile("lw.global %0, 0(%1)" : "=r"(bits0) : "r"(reinterpret_cast<uint32_t>(&a->in[base0])) : "memory");
            asm volatile("lw.global %0, 0(%1)" : "=r"(bits1) : "r"(reinterpret_cast<uint32_t>(&a->in[base1])) : "memory");

            float f0 = __builtin_bit_cast(float, bits0);
            float f1 = __builtin_bit_cast(float, bits1);

            if (f0 > lane_max) lane_max = f0;
            if (f1 > lane_max) lane_max = f1;
        }
        // Process 1 remaining tile
        for (; col_tile + 16u <= cols; col_tile += 16u) {
            uint32_t base0 = global_base + col_tile + lane_id;
            uint32_t bits0;
            asm volatile("lw.global %0, 0(%1)" : "=r"(bits0) : "r"(reinterpret_cast<uint32_t>(&a->in[base0])) : "memory");
            float f0 = __builtin_bit_cast(float, bits0);
            if (f0 > lane_max) lane_max = f0;
        }
        // Handle partial tile (fewer than 16 remaining columns)
        {
            uint32_t col = col_tile + lane_id;
            if (col < cols) {
                uint32_t bits;
                asm volatile("lw.global %0, 0(%1)" : "=r"(bits) : "r"(reinterpret_cast<uint32_t>(&a->in[global_base + col])) : "memory");
                float f = __builtin_bit_cast(float, bits);
                if (f > lane_max) lane_max = f;
            }
        }

        // ----------------------------------------------------------------
        // Phase 2: Warp-level reduction to find global max
        // ----------------------------------------------------------------
        // Write per-lane max to SMEM
        store_shared(smem_lane_addr, 0, __builtin_bit_cast(uint32_t, lane_max));

        mu_fence_smem();

        // Lane 0 reduces all 16 lane maxima and broadcasts
        if (lane_id == 0u) {
            float rmax = __builtin_bit_cast(float, load32_shared(smem_warp_base + 0u));
            float v1  = __builtin_bit_cast(float, load32_shared(smem_warp_base + 4u));
            float v2  = __builtin_bit_cast(float, load32_shared(smem_warp_base + 8u));
            float v3  = __builtin_bit_cast(float, load32_shared(smem_warp_base + 12u));
            float v4  = __builtin_bit_cast(float, load32_shared(smem_warp_base + 16u));
            float v5  = __builtin_bit_cast(float, load32_shared(smem_warp_base + 20u));
            float v6  = __builtin_bit_cast(float, load32_shared(smem_warp_base + 24u));
            float v7  = __builtin_bit_cast(float, load32_shared(smem_warp_base + 28u));
            float v8  = __builtin_bit_cast(float, load32_shared(smem_warp_base + 32u));
            float v9  = __builtin_bit_cast(float, load32_shared(smem_warp_base + 36u));
            float v10 = __builtin_bit_cast(float, load32_shared(smem_warp_base + 40u));
            float v11 = __builtin_bit_cast(float, load32_shared(smem_warp_base + 44u));
            float v12 = __builtin_bit_cast(float, load32_shared(smem_warp_base + 48u));
            float v13 = __builtin_bit_cast(float, load32_shared(smem_warp_base + 52u));
            float v14 = __builtin_bit_cast(float, load32_shared(smem_warp_base + 56u));
            float v15 = __builtin_bit_cast(float, load32_shared(smem_warp_base + 60u));

            if (v1  > rmax) rmax = v1;
            if (v2  > rmax) rmax = v2;
            if (v3  > rmax) rmax = v3;
            if (v4  > rmax) rmax = v4;
            if (v5  > rmax) rmax = v5;
            if (v6  > rmax) rmax = v6;
            if (v7  > rmax) rmax = v7;
            if (v8  > rmax) rmax = v8;
            if (v9  > rmax) rmax = v9;
            if (v10 > rmax) rmax = v10;
            if (v11 > rmax) rmax = v11;
            if (v12 > rmax) rmax = v12;
            if (v13 > rmax) rmax = v13;
            if (v14 > rmax) rmax = v14;
            if (v15 > rmax) rmax = v15;

            // Broadcast: write back to lane 0 slot (all lanes will read from slot 0)
            store_shared(smem_warp_base, 0, __builtin_bit_cast(uint32_t, rmax));
        } else {
            asm volatile("nop");
        }

        mu_fence_smem();

        // All lanes read the broadcast max from lane-0 slot
        const float global_max = __builtin_bit_cast(float, load32_shared(smem_warp_base));

        // ----------------------------------------------------------------
        // Phase 3: Compute exp(v - max) and accumulate partial sum
        // Re-read values from global memory (no SMEM staging of values)
        // ----------------------------------------------------------------
        float lane_sum = 0.0f;

        col_tile = 0u;
        // Process 4 tiles at a time for ILP
        for (; col_tile + 64u <= cols; col_tile += 64u) {
            uint32_t base0 = global_base + col_tile + lane_id;
            uint32_t base1 = global_base + col_tile + 16u + lane_id;
            uint32_t base2 = global_base + col_tile + 32u + lane_id;
            uint32_t base3 = global_base + col_tile + 48u + lane_id;

            uint32_t bits0, bits1, bits2, bits3;
            asm volatile("lw.global %0, 0(%1)" : "=r"(bits0) : "r"(reinterpret_cast<uint32_t>(&a->in[base0])) : "memory");
            asm volatile("lw.global %0, 0(%1)" : "=r"(bits1) : "r"(reinterpret_cast<uint32_t>(&a->in[base1])) : "memory");
            asm volatile("lw.global %0, 0(%1)" : "=r"(bits2) : "r"(reinterpret_cast<uint32_t>(&a->in[base2])) : "memory");
            asm volatile("lw.global %0, 0(%1)" : "=r"(bits3) : "r"(reinterpret_cast<uint32_t>(&a->in[base3])) : "memory");

            float e0 = mu_exp(__builtin_bit_cast(float, bits0) - global_max);
            float e1 = mu_exp(__builtin_bit_cast(float, bits1) - global_max);
            float e2 = mu_exp(__builtin_bit_cast(float, bits2) - global_max);
            float e3 = mu_exp(__builtin_bit_cast(float, bits3) - global_max);

            // Store exp values back to SMEM for reuse in normalize phase
            // Layout: col_tile/16 * 16 floats per tile, but we need per-lane storage
            // Since cols can be large, we re-compute exp in Phase 4 to avoid SMEM overflow
            lane_sum += e0;
            lane_sum += e1;
            lane_sum += e2;
            lane_sum += e3;
        }
        for (; col_tile + 32u <= cols; col_tile += 32u) {
            uint32_t base0 = global_base + col_tile + lane_id;
            uint32_t base1 = global_base + col_tile + 16u + lane_id;

            uint32_t bits0, bits1;
            asm volatile("lw.global %0, 0(%1)" : "=r"(bits0) : "r"(reinterpret_cast<uint32_t>(&a->in[base0])) : "memory");
            asm volatile("lw.global %0, 0(%1)" : "=r"(bits1) : "r"(reinterpret_cast<uint32_t>(&a->in[base1])) : "memory");

            float e0 = mu_exp(__builtin_bit_cast(float, bits0) - global_max);
            float e1 = mu_exp(__builtin_bit_cast(float, bits1) - global_max);

            lane_sum += e0;
            lane_sum += e1;
        }
        for (; col_tile + 16u <= cols; col_tile += 16u) {
            uint32_t base0 = global_base + col_tile + lane_id;
            uint32_t bits0;
            asm volatile("lw.global %0, 0(%1)" : "=r"(bits0) : "r"(reinterpret_cast<uint32_t>(&a->in[base0])) : "memory");
            float e0 = mu_exp(__builtin_bit_cast(float, bits0) - global_max);
            lane_sum += e0;
        }
        {
            uint32_t col = col_tile + lane_id;
            if (col < cols) {
                uint32_t bits;
                asm volatile("lw.global %0, 0(%1)" : "=r"(bits) : "r"(reinterpret_cast<uint32_t>(&a->in[global_base + col])) : "memory");
                float e = mu_exp(__builtin_bit_cast(float, bits) - global_max);
                lane_sum += e;
            }
        }

        // ----------------------------------------------------------------
        // Phase 4: Warp-level reduction to find global sum
        // ----------------------------------------------------------------
        store_shared(smem_lane_addr, 0, __builtin_bit_cast(uint32_t, lane_sum));

        mu_fence_smem();

        if (lane_id == 0u) {
            float rsum = __builtin_bit_cast(float, load32_shared(smem_warp_base + 0u));
            float s1  = __builtin_bit_cast(float, load32_shared(smem_warp_base + 4u));
            float s2  = __builtin_bit_cast(float, load32_shared(smem_warp_base + 8u));
            float s3  = __builtin_bit_cast(float, load32_shared(smem_warp_base + 12u));
            float s4  = __builtin_bit_cast(float, load32_shared(smem_warp_base + 16u));
            float s5  = __builtin_bit_cast(float, load32_shared(smem_warp_base + 20u));
            float s6  = __builtin_bit_cast(float, load32_shared(smem_warp_base + 24u));
            float s7  = __builtin_bit_cast(float, load32_shared(smem_warp_base + 28u));
            float s8  = __builtin_bit_cast(float, load32_shared(smem_warp_base + 32u));
            float s9  = __builtin_bit_cast(float, load32_shared(smem_warp_base + 36u));
            float s10 = __builtin_bit_cast(float, load32_shared(smem_warp_base + 40u));
            float s11 = __builtin_bit_cast(float, load32_shared(smem_warp_base + 44u));
            float s12 = __builtin_bit_cast(float, load32_shared(smem_warp_base + 48u));
            float s13 = __builtin_bit_cast(float, load32_shared(smem_warp_base + 52u));
            float s14 = __builtin_bit_cast(float, load32_shared(smem_warp_base + 56u));
            float s15 = __builtin_bit_cast(float, load32_shared(smem_warp_base + 60u));

            rsum += s1;
            rsum += s2;
            rsum += s3;
            rsum += s4;
            rsum += s5;
            rsum += s6;
            rsum += s7;
            rsum += s8;
            rsum += s9;
            rsum += s10;
            rsum += s11;
            rsum += s12;
            rsum += s13;
            rsum += s14;
            rsum += s15;

            store_shared(smem_warp_base, 0, __builtin_bit_cast(uint32_t, rsum));
        } else {
            asm volatile("nop");
        }

        mu_fence_smem();

        const float global_sum = __builtin_bit_cast(float, load32_shared(smem_warp_base));
        const float inv_sum = 1.0f / global_sum;

        // ----------------------------------------------------------------
        // Phase 5: Normalize — re-read from global, recompute exp, write output
        // ----------------------------------------------------------------
        col_tile = 0u;
        for (; col_tile + 64u <= cols; col_tile += 64u) {
            uint32_t base0 = global_base + col_tile + lane_id;
            uint32_t base1 = global_base + col_tile + 16u + lane_id;
            uint32_t base2 = global_base + col_tile + 32u + lane_id;
            uint32_t base3 = global_base + col_tile + 48u + lane_id;

            uint32_t bits0, bits1, bits2, bits3;
            asm volatile("lw.global %0, 0(%1)" : "=r"(bits0) : "r"(reinterpret_cast<uint32_t>(&a->in[base0])) : "memory");
            asm volatile("lw.global %0, 0(%1)" : "=r"(bits1) : "r"(reinterpret_cast<uint32_t>(&a->in[base1])) : "memory");
            asm volatile("lw.global %0, 0(%1)" : "=r"(bits2) : "r"(reinterpret_cast<uint32_t>(&a->in[base2])) : "memory");
            asm volatile("lw.global %0, 0(%1)" : "=r"(bits3) : "r"(reinterpret_cast<uint32_t>(&a->in[base3])) : "memory");

            float out0 = mu_exp(__builtin_bit_cast(float, bits0) - global_max) * inv_sum;
            float out1 = mu_exp(__builtin_bit_cast(float, bits1) - global_max) * inv_sum;
            float out2 = mu_exp(__builtin_bit_cast(float, bits2) - global_max) * inv_sum;
            float out3 = mu_exp(__builtin_bit_cast(float, bits3) - global_max) * inv_sum;

            a->out[base0] = out0;
            a->out[base1] = out1;
            a->out[base2] = out2;
            a->out[base3] = out3;
        }
        for (; col_tile + 32u <= cols; col_tile += 32u) {
            uint32_t base0 = global_base + col_tile + lane_id;
            uint32_t base1 = global_base + col_tile + 16u + lane_id;

            uint32_t bits0, bits1;
            asm volatile("lw.global %0, 0(%1)" : "=r"(bits0) : "r"(reinterpret_cast<uint32_t>(&a->in[base0])) : "memory");
            asm volatile("lw.global %0, 0(%1)" : "=r"(bits1) : "r"(reinterpret_cast<uint32_t>(&a->in[base1])) : "memory");

            float out0 = mu_exp(__builtin_bit_cast(float, bits0) - global_max) * inv_sum;
            float out1 = mu_exp(__builtin_bit_cast(float, bits1) - global_max) * inv_sum;

            a->out[base0] = out0;
            a->out[base1] = out1;
        }
        for (; col_tile + 16u <= cols; col_tile += 16u) {
            uint32_t base0 = global_base + col_tile + lane_id;
            uint32_t bits0;
            asm volatile("lw.global %0, 0(%1)" : "=r"(bits0) : "r"(reinterpret_cast<uint32_t>(&a->in[base0])) : "memory");
            float out0 = mu_exp(__builtin_bit_cast(float, bits0) - global_max) * inv_sum;
            a->out[base0] = out0;
        }
        {
            uint32_t col = col_tile + lane_id;
            if (col < cols) {
                uint32_t bits;
                asm volatile("lw.global %0, 0(%1)" : "=r"(bits) : "r"(reinterpret_cast<uint32_t>(&a->in[global_base + col])) : "memory");
                float out = mu_exp(__builtin_bit_cast(float, bits) - global_max) * inv_sum;
                a->out[global_base + col] = out;
            }
        }

        // Fence before next row to ensure SMEM reduction scratch is safe to reuse
        mu_fence_smem();
    }
}
