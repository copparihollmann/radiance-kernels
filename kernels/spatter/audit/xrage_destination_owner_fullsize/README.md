# Full-size xRAGE destination-owner exact audit

This audit is software-only. It snapshots the complete 2,051,101-element
Cyclotron output for the 1-, 2-, and 4-SM logical shadows and compares every
64-bit word against an independent serial last-writer oracle reconstructed
from all 6,664,304 pinned xRAGE transfers. It does not launch FireSim, program
an FPGA, or accept an arbitrary colliding-write winner.

The expected complete-output FNV is `215de6e81154b9c5`. Passing requires zero
word mismatches at every SM count, in addition to Cyclotron's own complete
GMEM digest and coverage checks.
