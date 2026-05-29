# 7.3 -- Output Format

The benchmark produces two distinct output formats depending on the execution mode: a full table for simulation mode with additional derived metrics, and a condensed table for socket/hybrid mode that adds bulk transfer data. Both modes share the same live status line during execution.

---

## 7.3.1 Simulation Mode Output

In simulation mode (no `PI05_HAS_SOCKETS`, no `socket` config block), the benchmark prints a comprehensive results table:

```
============ Pi0.5 Benchmark Results ============
Requests:         12 (2.87s)
Tokens:           768 generated
Throughput:       4.18 req/s, 267.60 tok/s (peak 312)
Peak concurrent:  4
--- TTFT (ms) ---
  Mean=12.45 Median=11.20 P99=18.73
  Corrected (pipeline-only): single=10.0 mean=11.4
--- ITL (ms) ---
  Mean=3.74 Median=3.50 P99=8.12
--- TPOT (ms) ---
  Mean=3.81 Median=3.72 P99=5.44
=================================================
```

This output is generated at lines 1437--1455 of `pi05_pipeline_runner.cpp`:

```cpp
// pi05_pipeline_runner.cpp:1437-1455
std::cout << "\n============ Pi0.5 Benchmark Results ============" << std::endl;
std::cout << "Requests:         " << grand_total_turns << " (" << bench_duration_s << "s)" << std::endl;
std::cout << "Tokens:           " << grand_total_tokens << " generated" << std::endl;
std::cout << "Throughput:       " << req_throughput << " req/s, "
          << output_tok_throughput << " tok/s (peak " << peak_output_tps << ")" << std::endl;
std::cout << "Peak concurrent:  " << peak_concurrent << std::endl;
std::cout << "--- TTFT (ms) ---" << std::endl;
std::cout << "  Mean=" << ttft_stats.mean << " Median=" << ttft_stats.median
          << " P99=" << ttft_stats.p99 << std::endl;
std::cout << "  Corrected (pipeline-only): single=" << std::setprecision(1)
          << corrected_ttft_single_ms << " mean=" << corrected_ttft_mean_ms << std::endl;
std::cout << std::setprecision(2);
std::cout << "--- ITL (ms) ---" << std::endl;
std::cout << "  Mean=" << itl_stats.mean << " Median=" << itl_stats.median
          << " P99=" << itl_stats.p99 << std::endl;
std::cout << "--- TPOT (ms) ---" << std::endl;
std::cout << "  Mean=" << tpot_stats.mean << " Median=" << tpot_stats.median
          << " P99=" << tpot_stats.p99 << std::endl;
std::cout << "=================================================" << std::endl;
```

**Fields unique to simulation mode:**

| Field | Description |
|-------|-------------|
| `Tokens: N generated` | Separate line showing total token count |
| `req/s` | Request-level throughput ($\text{grand\_total\_turns} / \text{bench\_duration\_s}$) |
| `(peak N)` | Peak 1-second sliding window TPS |
| `Corrected (pipeline-only)` | Theoretical TTFT: `single` and `mean` variants |

**Precision:** The output uses `std::fixed` with `std::setprecision(2)` for most values, switching to `setprecision(1)` for corrected TTFT (line 1446).

---

## 7.3.2 Socket / Hybrid Mode Output

In socket or hybrid mode (`PI05_HAS_SOCKETS` compiled, `socket` config block present), the benchmark prints a condensed table with bulk transfer data appended:

```
============ Pi0.5 Benchmark (Hybrid) ============
Requests:         12 (3.41s)
Throughput:       225.22 tok/s
Peak concurrent:  4
--- TTFT (ms): Mean=14.32 Median=13.10 P99=21.55
--- ITL  (ms): Mean=4.22 Median=3.98 P99=9.01
--- TPOT (ms): Mean=4.35 Median=4.10 P99=6.78
--- Bulk: H2D=482.3us (0.94 GB/s)  D2H=12.5us (0.26 GB/s)
=================================================
```

Generated at lines 1178--1192:

```cpp
// pi05_pipeline_runner.cpp:1178-1192
std::cout << "\n============ Pi0.5 Benchmark (" << mode_label << ") ============" << std::endl;
std::cout << "Requests:         " << grand_total_turns << " (" << bench_duration_s << "s)" << std::endl;
std::cout << "Throughput:       " << output_tok_throughput << " tok/s" << std::endl;
std::cout << "Peak concurrent:  " << peak_concurrent << std::endl;
std::cout << "--- TTFT (ms): Mean=" << ttft_stats.mean << " Median=" << ttft_stats.median
          << " P99=" << ttft_stats.p99 << std::endl;
std::cout << "--- ITL  (ms): Mean=" << itl_stats.mean << " Median=" << itl_stats.median
          << " P99=" << itl_stats.p99 << std::endl;
std::cout << "--- TPOT (ms): Mean=" << tpot_stats.mean << " Median=" << tpot_stats.median
          << " P99=" << tpot_stats.p99 << std::endl;
std::cout << "--- Bulk: H2D=" << std::setprecision(1) << h2d_stats.mean << "us ("
          << std::setprecision(2) << h2d_bw_gbps << " GB/s)"
          << "  D2H=" << std::setprecision(1) << d2h_stats.mean << "us ("
          << std::setprecision(2) << d2h_bw_gbps << " GB/s)" << std::endl;
std::cout << "=================================================" << std::endl;
```

**Differences from simulation mode:**

| Aspect | Simulation | Socket/Hybrid |
|--------|-----------|---------------|
| Header | `"Pi0.5 Benchmark Results"` | `"Pi0.5 Benchmark (Socket)"` or `"(Hybrid)"` |
| `Tokens` line | Present | Absent |
| `req/s` | Shown | Not shown |
| Peak TPS | Shown `(peak N)` | Not computed |
| Corrected TTFT | Shown | Not computed |
| TTFT/ITL/TPOT format | Multi-line with indented stats | Single-line per metric |
| Bulk metrics line | Absent | `H2D=Xus (Y GB/s) D2H=Xus (Y GB/s)` |
| Bulk precision | N/A | 1 decimal for time (us), 2 for bandwidth (GB/s) |

**Mode label logic:** The header prints `"Hybrid"` when `socket.mode == "bulk_only"`, and `"Socket"` when `socket.mode == "full"`:

```cpp
// pi05_pipeline_runner.cpp:1176
const char* mode_label = bulk_only ? "Hybrid" : "Socket";
```

---

## 7.3.3 Per-User Summary

Before the metrics table, both modes print an aggregate per-user summary:

```cpp
// pi05_pipeline_runner.cpp:1152-1160 (socket/hybrid)
// pi05_pipeline_runner.cpp:1401-1409 (simulation)
uint32_t grand_total_tokens = 0;
uint32_t grand_total_turns = 0;
for (uint32_t u = 0; u < num_users; u++) {
    grand_total_tokens += users[u].total_tokens;
    grand_total_turns += users[u].current_turn;
}

std::cout << "\nDone: " << num_users << " users, " << grand_total_turns
          << " turns, " << grand_total_tokens << " tokens." << std::endl;
```

Example output: `Done: 4 users, 12 turns, 768 tokens.`

Each `UserSession` also carries a `status_tag()` method for diagnostic display:

```cpp
// pi05_pipeline_runner.cpp:148-153
std::string status_tag() const {
    if (done && ctx_exhausted) return " [ctx_exhausted]";
    if (done) return " [done]";
    return "";
}
```

The `[ctx_exhausted]` tag indicates a user whose context window was exhausted before all turns completed -- the scheduler set `out.ctx_exhausted = true` on the final token, and the user was marked done early without advancing to the next turn. Although `status_tag()` is defined, it is not printed in the current output format -- it exists for debugging and potential future per-user detail tables.

Each `UserSession` tracks:

| Field | Type | Purpose |
|-------|------|---------|
| `slot_id` | `uint32_t` | Scheduler slot assignment |
| `current_turn` | `uint32_t` | Number of completed turns |
| `total_tokens` | `uint32_t` | Total tokens received across all turns |
| `ctx_exhausted` | `bool` | Whether the context window was exhausted |
| `done` | `bool` | Whether all turns are complete |

---

## 7.3.4 Live Status Line: `redraw_status()`

During the benchmark's main event loop, a live status line is redrawn periodically to show progress:

```cpp
// pi05_pipeline_runner.cpp:155-164
void redraw_status(const std::vector<UserSession>& users, uint32_t num_users,
                   uint32_t total_tokens) {
    uint32_t active = 0, done = 0;
    for (uint32_t i = 0; i < num_users; i++) {
        if (users[i].done) done++;
        else active++;
    }
    std::cout << "\r  Active: " << active << " | Done: " << done
              << " | Total tokens: " << total_tokens
              << "        " << std::flush;
}
```

**Redraw trigger:** The status line is refreshed at most every 100ms, controlled by a dirty flag and timer check:

```cpp
// pi05_pipeline_runner.cpp:1283 (simulation)
constexpr auto REDRAW_INTERVAL = std::chrono::milliseconds(100);
```

```cpp
// pi05_pipeline_runner.cpp:1019 (socket/hybrid)
constexpr auto REDRAW_INTERVAL_SOCKET = std::chrono::milliseconds(100);
```

The redraw logic inside the main loop:

```cpp
// pi05_pipeline_runner.cpp:1288-1293
if (!mgr.try_pop_output(out)) {
    if (display_dirty && Clock::now() - last_redraw >= REDRAW_INTERVAL) {
        redraw_status(users, num_users, grand_running_total);
        last_redraw = Clock::now();
        display_dirty = false;
    }
    std::this_thread::yield();
    continue;
}
```

Key behaviors:

- **`\r` carriage return**: the line overwrites itself in-place on the terminal, avoiding scroll.
- **`std::flush`**: forces the output buffer to flush without a newline, ensuring the status appears immediately.
- **Trailing spaces**: the `"        "` padding erases any leftover characters from a previous longer line.
- **Dirty flag**: `display_dirty` is set to `true` each time a token arrives. The status is only redrawn if at least one token has been received since the last redraw, avoiding unnecessary I/O during idle spins.
- **Final newline**: after the main loop exits, a final `redraw_status()` call is followed by `std::cout << std::endl;` to move past the status line before printing results.

Example live output (refreshing in place):

```
  Active: 3 | Done: 1 | Total tokens: 245
```

---

## 7.3.5 Startup Diagnostic Output

Before the benchmark loop begins, both modes print configuration information:

```
Config: examples/pi05_config.json
  num_users=4 input_tokens=128 output_tokens=64 max_turns=3
  Pipeline: 10 stages, 10000us total, 1000us/stage avg, bottleneck=1200us
  accept_rate=1.00 seed=42 stagger=2500us
```

Socket mode additionally prints connection diagnostics:

```
Socket mode=bulk_only H2D=455688B (458752B padded) FIFO=524288
Starting device launcher: /path/to/pi05_device_launcher
Device launcher running, descriptors ready.
Bulk channels connected (H2D + D2H).
Allocated 4 slots (ids 0..3)
Submitted initial requests (with bulk H2D) for 4 users
```

---

**Next:** [run_hybrid.sh Orchestration](04_run_hybrid_orchestration.md)
