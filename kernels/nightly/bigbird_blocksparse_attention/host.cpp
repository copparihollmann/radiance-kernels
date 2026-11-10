#include <nightly/host.h>

#include "bigbird_blocksparse_data.h"

int main() {
    nh_puts("bigbird_blocksparse_attention: seq=");
    nh_putu(BBSA_SEQUENCE);
    nh_puts(" block=");
    nh_putu(BBSA_BLOCK_SIZE);
    nh_puts(" blocks=");
    nh_putu(BBSA_BLOCKS);
    nh_puts(" head_dim=");
    nh_putu(BBSA_HEAD_DIM);
    nh_puts(" random=");
    nh_putu(BBSA_RANDOM_BLOCKS);
    nh_puts(" one_head one_sm\n");
    nh_flush();

    nightly_host_clear_verify(NIGHTLY_CLUSTERS);
    nightly_host_launch(NIGHTLY_CLUSTERS);
    const int finished = nightly_host_wait(NIGHTLY_CLUSTERS);
    const uint32_t cycles = nightly_host_report(NIGHTLY_CLUSTERS);
    const int correct = nightly_host_check_u32(
        "all_numerical_and_structural_status", NIGHTLY_CLUSTERS,
        BBSA_STATUS_ADDR, BBSA_GOLD_STATUS_ADDR, BBSA_CHECKS);
    nh_puts("outputs_per_1000_cycles=");
    nh_putu(cycles ? (1000ull * BBSA_OUTPUT_ELEMENTS) / cycles : 0);
    nh_puts("\n");
    const int pass = finished && correct;
    nh_puts(pass ? "\nPASS\n" : "\nFAIL\n");
    nh_flush();
    exit(pass ? 0 : 1);
}
