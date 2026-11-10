// Launch and fully verify the bounded Qwen3.5 DeltaNet recurrent component.
#include <nightly/host.h>

#include "deltanet_state_core.hpp"
#include "qwen35_deltanet_state_data.h"

static uint32_t float_bits(float value) {
    union {
        float f;
        uint32_t u;
    } bits = {value};
    return bits.u;
}

static uint32_t check_tensor(const char *name, uint32_t actual_address,
                             uint32_t expected_address, uint32_t count,
                             uint32_t *poison, uint32_t *first_bad,
                             float *max_absolute_error) {
    volatile float *actual = reinterpret_cast<volatile float *>(
        nightly_gpu_ptr(actual_address));
    volatile float *expected = reinterpret_cast<volatile float *>(
        nightly_gpu_ptr(expected_address));
    uint32_t bad = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const float observed = actual[i];
        const float golden = expected[i];
        const float error = qwen35_deltanet_state::absf(observed - golden);
        if (error > *max_absolute_error)
            *max_absolute_error = error;
        if (!qwen35_deltanet_state::within_tolerance(observed, golden)) {
            if (!bad)
                *first_bad = i;
            ++bad;
            if (float_bits(observed) == 0xEEEEEEEEU)
                ++*poison;
        }
    }
    nh_puts(name);
    nh_puts(": checked=");
    nh_putu(count);
    nh_puts(" bad=");
    nh_putu(bad);
    nh_puts(" first_bad=");
    nh_putu(*first_bad);
    nh_puts("\n");
    return bad;
}

int main() {
    nh_puts("qwen35_deltanet_state: recurrent boundary only T=");
    nh_putu(Q35_SEQUENCE);
    nh_puts(" K=");
    nh_putu(Q35_KEY_DIM);
    nh_puts(" V=");
    nh_putu(Q35_VALUE_DIM);
    nh_puts(" clusters=");
    nh_putu(NIGHTLY_CLUSTERS);
    nh_puts(" atol=4e-5 rtol=4e-5\n");
    nh_flush();

    nightly_host_launch(NIGHTLY_CLUSTERS);
    const int finished = nightly_host_wait(NIGHTLY_CLUSTERS);
    const uint32_t cycles = nightly_host_report(NIGHTLY_CLUSTERS);

    uint32_t poison = 0;
    uint32_t first_output = 0;
    uint32_t first_state = 0;
    float max_absolute_error = 0.0F;
    const uint32_t output_bad =
        check_tensor("output", Q35_OUTPUT_ADDR, Q35_GOLD_OUTPUT_ADDR,
                     Q35_OUTPUT_ELEMENTS, &poison, &first_output,
                     &max_absolute_error);
    const uint32_t state_bad =
        check_tensor("final_state", Q35_FINAL_STATE_ADDR,
                     Q35_GOLD_STATE_ADDR, Q35_STATE_ELEMENTS, &poison,
                     &first_state, &max_absolute_error);
    nh_puts("poison=");
    nh_putu(poison);
    nh_puts(" max_abs_error_x1e9=");
    nh_putu(static_cast<uint32_t>(max_absolute_error * 1.0e9F));
    nh_puts(" checked_values_per_1000_cycles=");
    nh_putu(cycles ? (1000ULL * Q35_CHECKS) / cycles : 0);
    nh_puts("\n");

    const int pass = finished && output_bad == 0 && state_bad == 0;
    nh_puts(pass ? "\nPASS\n" : "\nFAIL\n");
    nh_flush();
    exit(pass ? 0 : 1);
}
