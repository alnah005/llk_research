# Chapter 8 -- Design Critique, Failure Modes, and Gotchas

This final chapter critically evaluates the Pi0.5 simulation framework's design decisions, catalogs the failure modes that can crash or corrupt a run, documents the behavioral differences between the passthrough kernel and a production kernel, and collects the operational gotchas that every operator must know before running on real hardware.

## Sections

1. [Design Critique](01_design_critique.md) -- Pipeline flattening information loss, corrected TTFT assumptions, custom JSON parser limitations, single-channel bulk serialization, fork/exec coupling, and D2H slot mismatch severity
2. [Failure Modes](02_failure_modes.md) -- Zombie processes, `/dev/shm` leaks, IOMMU mapping race, PCIe alignment violations, `TT_VISIBLE_DEVICES` crashes, sentinel duplication, FIFO underrun, launcher premature exit, and D2H spin-wait behavior
3. [Production Kernel Differences](03_production_kernel_differences.md) -- H2D pop timing, signal-wait for denoise, synthetic vs. real action output, and core assignment
4. [Operational Gotchas](04_operational_gotchas.md) -- `/dev/shm` cleanup, environment variables, PCIe alignment, process cleanup, `output_tokens=1`, FIFO sizing, stagger auto-calculation, and action history corruption propagation

---

**Previous:** [Chapter 7 -- Metrics, Reporting, and Benchmark Orchestration](../ch7_metrics_reporting_and_benchmark_orchestration/index.md)
