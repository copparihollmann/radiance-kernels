# Full-model timing

The full-model functional results use pure fallback helpers. Their ticks do not
measure Muon computation or HBM latency. This runner uses the same qualified
ELFs and input images with helpers disabled, core timing enabled, and coupled
Ramulator HBM. It covers all four models on one and two SMs.

From the managed task directory:

```sh
python3 radiance-kernels/kernels/model_chain/mx_vpu/full_model_timing.py prepare \
  --bundle artifacts/fpga-handoff-v3 \
  --cyclotron chipyard/generators/radiance/cyclotron \
  --simulator artifacts/e2e-hbm-audit/cyclotron-progress-v8 \
  --runtime-library artifacts/e2e-hbm-audit/runtime-dependency-backup/libramulator.so \
  --output artifacts/e2e-hbm-audit/full-model-timing-v1

python3 radiance-kernels/kernels/model_chain/mx_vpu/full_model_timing.py run \
  --root artifacts/e2e-hbm-audit/full-model-timing-v1 --tmpdir "$TMPDIR"

python3 radiance-kernels/kernels/model_chain/mx_vpu/full_model_timing.py report \
  --root artifacts/e2e-hbm-audit/full-model-timing-v1

python3 radiance-kernels/kernels/model_chain/mx_vpu/full_model_timing.py status \
  --root artifacts/e2e-hbm-audit/full-model-timing-v1
```

Preparation creates a new directory and checks the supplied binary, checkpoint,
configuration and reference hashes. It reuses existing compact images. The
default cycle limit is 10 trillion; reaching it is a failed attempt, never a
completed result. Clock periods default to the documented 1:1 software control.
Set `--source-ticks` and `--memory-ticks` when preparing a different clock ratio.
These settings alone do not calibrate the simulator against a board.

`--runtime-library` pins the supplied `libramulator.so`, adds its hash to every
job, and sets the loader directory explicitly. Keep this copy outside temporary
build directories for long runs. Preparation, resumed runs and new case launches
reject a changed library. Omitting the option retains the simulator's normal
loader search path.

Run one heavy simulator at a time, including other control queues. The runner
locks its own plan directory. It resumes completed cases by checking their
recorded inputs and acceptance checks again. A failed or interrupted case gets
a new attempt directory when rerun; the previous attempt remains intact. It
does not resume a simulated machine from an arbitrary cycle.

`status` checks the saved worker PID against its owner and command and checks
each simulator against its recorded command. It shows stopped processes and
the latest progress, including a partially written journal. A saved `running`
receipt can survive an interrupted worker, such as a full scratch filesystem;
it does not prove that the worker is still alive. An unconfirmed process needs
inspection before restarting. Status collection itself never starts or resumes
a job. Pass `--control-queue path/to/results.json` to also show live control
simulators while the model simulator is paused for a serial control run.

A case passes only with successful completion in every configured cluster,
every stage reached in order, the expected MX/VPU work, complete output checks,
and matching HBM configuration, clocks and drained outstanding requests. Decoder
checks include every retained stage tensor and exact original greedy tokens.
SmolVLA checks the original and format actions plus arena guards at the existing
tolerance. Instruction and memory traces stay disabled. Sparse progress journals
and final performance summaries retain stage cycles and memory/stall statistics.

Each completed case has a `result.json` with its stage intervals and major
sections. Intervals run from one core-0/warp-0 stage entry to the next and include
launch and barrier gaps. The last interval includes final checking and the
global completion tail. Two-SM timelines overlap: do not add them together or
interpret them as isolated kernel-body measurements. Loading precedes cycle zero;
host process wall time includes loading, simulation and checking.

Performance-summary latency sums and queue counters overlap; they are not
exclusive stage costs. The Radiance preset models ALU/FPU result latency in
its hazard model. Zero execute-queue busy time does not mean zero computation.

`timing-summary.json` and `cycles.csv` contain completed results and explicit
pending cases. Pending cycles are empty. Existing helper-assisted functional
passes, prefixes, projected runtimes and queued jobs do not complete this suite.
The console's `simulated_cycles` is a count; `global_finish_cycle` is the
zero-based finish timestamp and must be one less. The HBM source clock after
accelerator finalization and draining is recorded separately.
