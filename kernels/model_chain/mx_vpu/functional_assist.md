# Functional helper assistance

This experimental path accelerates pure fallback arithmetic inside Cyclotron's
functional core. It does not measure Muon execution time or memory traffic. Core
scheduling, stage wrappers, barriers, MX/VPU instructions and MMIO still execute
from the ELF. A complete fallback implementation remains in that ELF for
unassisted differential execution.

Generate SmolVLA or a decoder with `--fallback-call-abi` to emit a readonly
32-word descriptor and a four-register-argument `rk_fallback_call` around supported
pure operations. The default generator output is unchanged. Both cluster contexts
resolve activation and attention-scratch pointers relative to their own arena.
Native builds use full-width host pointers; the RV32 bridge widens GPU words.

`build_fallback_plugin.py --out <new-directory>` compiles only the existing math
headers with FP contraction and fast-math disabled. It embeds their combined
SHA256 and records compiler flags and plugin SHA256 in `build.json`. Independent
output lanes may use eight host threads; each output's reduction order stays
unchanged. No model graph or reference tensors are linked into the plugin.

An explicit `[sim] functional_assist = "plan.json"` enables the bridge. Its plan
uses schema `radiance-pure-fallback-v1` and binds `elf_sha256`, `helper_sha256`,
`plugin`, `plugin_sha256`, `combined_source_sha256`, explicit `writable` byte
ranges, and a fresh `report` path. Relative paths resolve beside the plan.
Cyclotron rejects assistance with core timing, HBM or ISA tracing, or with a
Gemmini configuration other than `both/radiance` without overrides.

Before native writes, the bridge checks helper code in live RAM, descriptors,
alignment, overflow, operation-specific footprints, aliases, writable ranges,
lane IDs and uniform arguments/return addresses. It preserves guest registers,
SP, active masks and reconvergence state. Reports label assistance as functional
only and count operations, work and host parallelism. Attention work counters
use dense shapes even where the original helper skips masked products.

The queue must explicitly use `execution_mode="assisted_functional_core"` and
supply `assistance_report`; a matching completed report and positive work counts
are required alongside all ordinary stage and cluster completion checks. Assisted
runs cannot pass as unassisted functional execution or timing/HBM execution.

Qualification receipts live in the managed task's `artifacts/e2e-hbm-audit`:

- `fallback-operations-v7`: all original 17 operations, tails, sparse masks, in-place work,
  distinct replicas, SP/s2 canaries and actual MX-to-helper-to-VPU dependencies;
  complete RAM dumps match unassisted execution on one and two SMs.
- `fallback-full-projection-v2`: full-size INT8/FP32 projection, independent
  sequential FP32 golden, poisoned outputs, three alternating one-SM runs and
  a two-SM differential pair. Its speedup is a host-process measurement.
- `smolvla-stack-fixed-qualification-v1/qualified-result-v2.json`: complete
  one-/two-SM assisted SmolVLA, including all format and original action checks.
- `fallback-partition-control-v3`: FP32 remainder of a mixed projection, retaining
  actual MX columns, with K/N tails and empty/full partitions on one and two SMs.
- `compact-helper-control-v3`: seven additional decoder operations, independent
  FP32 references, prefill/decode/window/cap cases and complete ISA differentials.
- `host-stage-controls-v1`: all three reduced decoder graphs on one/two SMs with
  assistance enabled/disabled. Every actual stage buffer matches the frozen
  reference and is bit-identical between the two execution paths.
- `functional-assist-default-hbm-v3`: default unassisted one-/two-SM controls
  retain identical complete HBM reports while exercising sparse PC snapshots.

The ABI covers 25 pure operations. Decoder additions cover I6 linear, grouped I5
and I8 tied linear, FP16/FP32 tied linear, causal grouped-query attention and logit
softcap. Packed weights stay compact; each operation uses its existing arithmetic.
SmolVLA's mixed projection assists only its FP32 remainder while the selected
columns execute on actual MX. Reduced controls do not establish full-model passes.

On the GPU, calls from the descriptor dispatcher are forced inline. Expanding
the dispatcher had caused the compiler to outline helpers with more than eight
arguments; their stack-passed arguments do not work with this toolchain's
lane-interleaved private stacks. Causal attention and remaining long graph calls
(RoPE, KV append, packed embedding) also require inlining. Failed outlined-call
controls are retained separately; no arithmetic or numerical tolerance changed.

For full decoder replicas, `host_stage_checks.prepare(target, frozen_data)` can
move only test-oracle storage off the GPU. It requires a sharded two-replica
greedy source bundle and identical frozen golden tensors. All model tensors,
stage calls, weights and embedded greedy-token checks remain on the GPU. Both
the manifest and ordinary Makefile select the oracle-free device sources;
the original CPU reference/verifier sources remain available for native checks.
The run must dump every typed stage buffer for every active replica. Queue jobs
must provide `model_manifest` and `host_stage_result_dir`: successful tohost and
stage entries alone cannot pass. The host compares actual GPU RAM against the
frozen references using the original FP32 tolerances and exact integer checks.
This is device output checking, separate from executing the native CPU graph.

The repeated-launch worker stack bug is fixed separately; see `worker_stack.md`.
The corrected complete one-SM SmolVLA assisted graph now passes all3673stages,
all1600format andoriginalactions, guards and actual4608MXloops/3072VPUcommands.
All354093515776fallbacklinearMACs are counted, alongside37748736actualMXMACs.
Complete one- and two-SM assisted SmolVLA execution now passes; all three final action buffers are bit-identical. Other full models remain incomplete. Native
two-replica SmolVLA has7346stages/3860independentboundaries passing.
