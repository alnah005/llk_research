# Agent B Review: Chapter 7 — Pass 1

**No feedback — chapter approved.**

All four sections (01-04) were verified against the source files `pi05_pipeline_runner.cpp` and `run_hybrid.sh`. Specific items checked:

- `compute_stats()` implementation, Stats struct, P99 index table (all arithmetic verified correct)
- `BulkTransferMetrics` struct fields and line-level references (lines 241-246 match)
- TTFT, ITL, TPOT measurement code paths in both simulation (lines 1302-1334) and socket (lines 1039-1084) modes
- Corrected TTFT formula and its simulation-only scope (lines 1432-1434, absent from socket output)
- Peak TPS sliding-window algorithm and its simulation-only scope (lines 1420-1429, absent from socket path)
- Bandwidth formulas (bytes/nanosecond = GB/s derivation is correct)
- H2D/D2H payload size calculations with default PixelPayloadConfig
- `pcie_bandwidth_gbps` being informational-only (parsed at line 435, never referenced in computation)
- Output format differences between simulation (lines 1437-1455) and socket/hybrid (lines 1178-1192)
- `redraw_status()` implementation and redraw interval logic
- `status_tag()` defined but never invoked (confirmed via grep)
- `run_hybrid.sh` content matches source verbatim (14 lines)
- DeviceLauncher fork/exec pattern, prctl, descriptor polling, timeout logic (lines 632-725)
- Sentinel shutdown protocol sequence and `_exit()` rationale (lines 1131-1196)
- IOMMU 2-second sleep distinction (script line 8 vs. C++ line 920)

No factual errors, no critical coherence issues, no structural gaps found.
