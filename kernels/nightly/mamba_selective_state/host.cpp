// Launch and report the bounded selective-state correctness milestone.
#include <nightly/host.h>

#include "mamba_selective_state_data.h"

int main() {
    nh_puts("mamba_selective_state: selective scan only T=");
    nh_putu(MSS_SEQUENCE);
    nh_puts(" channels=");
    nh_putu(MSS_CHANNELS);
    nh_puts(" state=");
    nh_putu(MSS_STATE);
    nh_puts(" atol=2e-5 rtol=2e-5\n");
    nh_flush();
    nightly_host_clear_verify(NIGHTLY_CLUSTERS);
    nightly_host_launch(NIGHTLY_CLUSTERS);
    const int finished = nightly_host_wait(NIGHTLY_CLUSTERS);
    const uint32_t cycles = nightly_host_report(NIGHTLY_CLUSTERS);
    const int correct = nightly_host_check(
        "oracle_tolerance_status", NIGHTLY_CLUSTERS, MSS_STATUS_ADDR,
        MSS_GOLD_STATUS_ADDR, MSS_CHECKS, /*max_err_bp=*/0, /*max_bad=*/0);
    nh_puts("checked_floats_per_1000_cycles=");
    nh_putu(cycles ? (1000ull * MSS_CHECKS) / cycles : 0);
    nh_puts("\n");
    const int pass = finished && correct;
    nh_puts(pass ? "\nPASS\n" : "\nFAIL\n");
    nh_flush();
    exit(pass ? 0 : 1);
}
