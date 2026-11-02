// Host launch and exact structural check for the Granite MoE routing milestone.
#include <nightly/host.h>
#include "granite_moe_route_data.h"

int main() {
  nh_puts("granite_moe_route: tokens="); nh_putu(GMR_TOKENS);
  nh_puts(" experts="); nh_putu(GMR_EXPERTS);
  nh_puts(" top_k="); nh_putu(GMR_TOP_K);
  nh_puts(" assignments="); nh_putu(GMR_ASSIGNMENTS); nh_puts("\n"); nh_flush();
  nightly_host_clear_verify(NIGHTLY_CLUSTERS);
  nightly_host_launch(NIGHTLY_CLUSTERS);
  const int finished = nightly_host_wait(NIGHTLY_CLUSTERS);
  const uint32_t cycles = nightly_host_report(NIGHTLY_CLUSTERS);
  const int exact = nightly_host_check_u32(
      "routing_structure", NIGHTLY_CLUSTERS, GMR_OUTPUT_ADDR, GMR_GOLD_ADDR,
      GMR_OUTPUT_WORDS);
  nh_puts("assignments_per_1000_cycles=");
  nh_putu(cycles ? (1000ull * GMR_ASSIGNMENTS) / cycles : 0); nh_puts("\n");
  const int pass = finished && exact;
  nh_puts(pass ? "\nPASS\n" : "\nFAIL\n"); nh_flush();
  exit(pass ? 0 : 1);
}
