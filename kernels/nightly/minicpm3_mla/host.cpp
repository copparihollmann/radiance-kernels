// Launch and report the bounded one-SM MiniCPM3 MLA KV-path component.
#include <nightly/host.h>

#include "minicpm3_mla_data.h"

int main() {
    nh_puts("minicpm3_mla: KV path only tokens=");
    nh_putu(MCM3_TOKENS);
    nh_puts(" hidden=");
    nh_putu(MCM3_HIDDEN);
    nh_puts(" rank=");
    nh_putu(MCM3_KV_RANK);
    nh_puts(" heads=");
    nh_putu(MCM3_HEADS);
    nh_puts(" atol=3e-5 rtol=3e-5\n");
    nh_flush();

    nightly_host_clear_verify(NIGHTLY_CLUSTERS);
    nightly_host_launch(NIGHTLY_CLUSTERS);
    const int finished = nightly_host_wait(NIGHTLY_CLUSTERS);
    const uint32_t cycles = nightly_host_report(NIGHTLY_CLUSTERS);
    const int correct = nightly_host_check(
        "all_stage_status", NIGHTLY_CLUSTERS, MCM3_STATUS_ADDR,
        MCM3_GOLD_STATUS_ADDR, MCM3_OUTPUT_ELEMENTS,
        /*max_err_bp=*/0, /*max_bad=*/0);
    nh_puts("checked_outputs_per_1000_cycles=");
    nh_putu(cycles ? (1000ULL * MCM3_OUTPUT_ELEMENTS) / cycles : 0);
    nh_puts("\n");
    const int pass = finished && correct;
    nh_puts(pass ? "\nPASS\n" : "\nFAIL\n");
    nh_flush();
    exit(pass ? 0 : 1);
}
