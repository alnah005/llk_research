# Chapter 7 -- Metrics, Reporting, and Benchmark Orchestration

This chapter documents every metric computed by the Pi0.5 benchmark framework, the exact formulas and code paths used to measure them, the output formats for each execution mode, and the `run_hybrid.sh` script that orchestrates an end-to-end benchmark run.

## Sections

1. [Timing Metrics](01_timing_metrics.md) -- TTFT, ITL, TPOT, throughput, peak TPS, corrected TTFT, and the `compute_stats()` function with its P99 gotcha
2. [Bulk Transfer Metrics](02_bulk_transfer_metrics.md) -- `BulkTransferMetrics` struct, H2D/D2H bandwidth computation, and the `pcie_bandwidth_gbps` informational-only caveat
3. [Output Format](03_output_format.md) -- Simulation vs. socket/hybrid output tables, per-user summary, and the `redraw_status()` live status line
4. [run_hybrid.sh Orchestration](04_run_hybrid_orchestration.md) -- Line-by-line walkthrough of the orchestration script, env vars, IOMMU sleep, end-to-end workflow, and `_exit()` rationale

---

**Previous:** [Chapter 6 -- Decode Scheduler Integration](../ch6_decode_scheduler_integration/index.md)

**Next:** [Chapter 8 -- Design Critique, Failure Modes, and Gotchas](../ch8_design_critique_failure_modes_and_gotchas/index.md)
