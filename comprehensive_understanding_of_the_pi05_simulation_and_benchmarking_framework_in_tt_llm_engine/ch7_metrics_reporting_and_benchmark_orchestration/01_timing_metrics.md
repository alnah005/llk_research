# 7.1 -- Timing Metrics

All timing metrics are collected inside the main event loop of `pi05_pipeline_runner.cpp`. The framework uses `std::chrono::steady_clock` throughout, aliased as `Clock` at line 54:

```cpp
// pi05_pipeline_runner.cpp:54-55
using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
```

Every metric is accumulated into `std::vector<double>` sample vectors measured in milliseconds, then summarized via `compute_stats()` after the benchmark completes.

---

## 7.1.1 Time To First Token (TTFT)

TTFT measures the wall-clock time in milliseconds from when a user's turn is submitted (`SUBMIT` request pushed) to the arrival of the first non-`EMPTY_TOKEN` output for that user on that turn.

**Measurement point:** Each `UserSession` records `turn_start` at the moment `push_request()` is called for the `SUBMIT`. When the first token arrives with `out.token_id != pm::EMPTY_TOKEN`, the difference is computed:

```cpp
// pi05_pipeline_runner.cpp:1302-1311 (simulation mode, identical logic at lines 1039-1048 for socket mode)
if (out.token_id != pm::EMPTY_TOKEN) {
    u.total_tokens++;
    grand_running_total++;
    auto now = Clock::now();
    if (!u.first_token_received) {
        double ttft_ms = std::chrono::duration<double, std::milli>(now - u.turn_start).count();
        ttft_samples.push_back(ttft_ms);
        u.first_token_time = now;
        u.first_token_received = true;
    }
```

**Per-turn granularity:** TTFT is measured once per turn, not once per user. A user running 3 turns contributes 3 independent TTFT samples. The `turn_start` and `first_token_received` fields are reset at the start of each subsequent turn:

```cpp
// pi05_pipeline_runner.cpp:1349-1352
u.turn_start = Clock::now();
u.first_token_received = false;
u.turn_token_count = 0;
```

**What TTFT includes:**

- Queue wait time inside `DecodeScheduler`
- Full pipeline latency (all stages) for the prefill pass
- Contention from other active users sharing pipeline stages

**EMPTY_TOKEN guard:** The token `pm::EMPTY_TOKEN` is used as a completion marker and is explicitly excluded. Only real tokens trigger TTFT recording.

**`turn_start` baseline:** Set at `Clock::now()` immediately after the `SUBMIT` request is pushed, meaning TTFT includes any queue wait time inside the `DecodeScheduler`.

**Aggregation:** Mean, median, P99 via `compute_stats(ttft_samples)`.

---

## 7.1.2 Inter-Token Latency (ITL)

ITL measures the wall-clock time in milliseconds between consecutive non-empty token arrivals **globally across all users** -- not per-user.

**Measurement point:** A single `last_global_token_time` timestamp is maintained. Every time any user receives a non-`EMPTY_TOKEN` output, the delta from the last global token is recorded:

```cpp
// pi05_pipeline_runner.cpp:1312-1316
if (global_token_seen) {
    double itl_ms = std::chrono::duration<double, std::milli>(now - last_global_token_time).count();
    itl_samples.push_back(itl_ms);
}
token_timestamps.push_back(now);
last_global_token_time = now;
global_token_seen = true;
```

**Key design choice:** ITL is cross-user, not per-user. This means it measures the system's aggregate token emission rate. When multiple users are generating tokens concurrently, ITL will be much lower than per-user TPOT because tokens from different users interleave.

**First token excluded:** The `global_token_seen` flag ensures no ITL sample is recorded for the very first token in the entire benchmark run.

**Aggregation:** Mean, median, P99 via `compute_stats(itl_samples)`.

---

## 7.1.3 Time Per Output Token (TPOT)

TPOT measures the average time in milliseconds between consecutive tokens **for a single user within a single turn**, computed only when the turn produced at least two tokens.

```cpp
// pi05_pipeline_runner.cpp:1330-1334
if (u.turn_token_count > 1) {
    double tpot_ms = std::chrono::duration<double, std::milli>(
        u.last_token_time - u.first_token_time).count() / (u.turn_token_count - 1);
    tpot_samples.push_back(tpot_ms);
}
```

The formula:

$$\text{TPOT} = \frac{t_{\text{last}} - t_{\text{first}}}{\text{token\_count} - 1}$$

Key details:

- **Guard: `turn_token_count > 1`**: if a turn produces only one token, division by zero would occur; the metric is silently skipped. This can happen when `max_decode = 1` or when context is exhausted after a single token.
- **Excludes TTFT**: by using `first_token_time` rather than `turn_start`, TPOT isolates decode-phase latency from prefill latency.
- **Per-turn granularity**: each completed turn contributes one TPOT sample, regardless of how many tokens it produced.

**Aggregation:** Mean, median, P99 via `compute_stats(tpot_samples)`.

---

## 7.1.4 Output Throughput

Output throughput measures the aggregate token generation rate across all users for the entire benchmark run.

```cpp
// pi05_pipeline_runner.cpp:1415-1416
double bench_duration_s = std::chrono::duration<double>(bench_end - bench_start).count();
double output_tok_throughput = grand_total_tokens / bench_duration_s;
```

$$\text{output\_throughput} = \frac{\text{grand\_total\_tokens}}{\text{bench\_duration\_s}} \quad \text{(tok/s)}$$

**Scope of `bench_duration_s`:** Measured from `bench_start` (just before the first user submission) to `bench_end` (after the last user completes all turns). This includes stagger delays, all turns across all users, and turn-resubmission overhead. It does **not** include the ALLOCATE phase or the EVICT teardown.

**Request throughput** (simulation only): Additionally computed as `grand_total_turns / bench_duration_s`, measuring requests (turns) per second. This `req_throughput` (req/s) is not computed in socket/hybrid mode.

---

## 7.1.5 Peak Output TPS (Simulation Only)

Peak output TPS finds the maximum number of tokens observed within any 1-second sliding window during the benchmark. This metric is only computed in simulation mode.

```cpp
// pi05_pipeline_runner.cpp:1420-1429
uint32_t peak_output_tps = 0;
if (!token_timestamps.empty()) {
    size_t left = 0;
    for (size_t right = 0; right < token_timestamps.size(); right++) {
        while (std::chrono::duration<double>(
                   token_timestamps[right] - token_timestamps[left]).count() > 1.0) {
            left++;
        }
        peak_output_tps = std::max(peak_output_tps,
            static_cast<uint32_t>(right - left + 1));
    }
}
```

The algorithm uses a two-pointer sliding window over the sorted `token_timestamps` vector. Timestamps are naturally sorted because tokens are consumed in arrival order from a single-threaded event loop.

**Why simulation only:** In socket/hybrid mode, the output table omits peak TPS -- the metric is less meaningful when real hardware transfer times dominate and token timestamps reflect I/O wait rather than pipeline throughput. Peak TPS captures burst capacity: for a balanced pipeline with $N$ concurrent users and stage duration $d$ microseconds, the theoretical peak is $\lfloor 10^6 / d \rfloor$ tokens/second.

---

## 7.1.6 Corrected TTFT (Simulation Only)

Corrected TTFT provides a theoretical pipeline-only estimate that excludes prefill contention effects. Two variants are computed:

```cpp
// pi05_pipeline_runner.cpp:1432-1434
double corrected_ttft_single_ms = total_latency_us / 1000.0;
double corrected_ttft_mean_ms = (total_latency_us
    + (num_users - 1) * max_non_vision_stage_us / 2.0) / 1000.0;
```

**Single-user corrected TTFT:**

$$\text{corrected\_ttft\_single} = \frac{\text{total\_latency\_us}}{1000} \quad \text{(ms)}$$

This is the theoretical minimum TTFT when only one user is active -- equal to the full pipeline traversal latency (sum of all stages across all phases, accounting for `loop_count`).

**Mean corrected TTFT with stagger:**

$$\text{corrected\_ttft\_mean} = \frac{\text{total\_latency\_us} + \frac{(N_{\text{users}} - 1) \cdot d_{\text{bottleneck}}}{2}}{1000} \quad \text{(ms)}$$

where $d_{\text{bottleneck}}$ is `max_non_vision_stage_us` -- the slowest stage duration excluding the vision phase. The $(N-1) \cdot d/2$ term models the average additional queuing delay when $N$ users are staggered across pipeline stages.

**Why exclude vision:** The vision phase typically has a much longer stage duration but runs once per turn (not per decode step). Including it in the stagger contention model would overestimate queuing delays, since vision stages do not contribute to steady-state decode contention.

**Why simulation only:** Corrected TTFT is a pipeline model prediction. In socket/hybrid mode, real hardware introduces additional latency sources (PCIe transfers, IOMMU overhead) that make the corrected estimate misleading.

---

## 7.1.7 `compute_stats()` -- Statistical Summary

All timing metrics are aggregated through a single utility function:

```cpp
// pi05_pipeline_runner.cpp:166-182
struct Stats {
    double mean = 0;
    double median = 0;
    double p99 = 0;
};

Stats compute_stats(std::vector<double>& samples) {
    if (samples.empty()) return {};
    std::sort(samples.begin(), samples.end());
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    size_t n = samples.size();
    return {
        .mean = sum / static_cast<double>(n),
        .median = samples[n / 2],
        .p99 = samples[std::min(n - 1, static_cast<size_t>(std::ceil(0.99 * n) - 1))],
    };
}
```

**Implementation details:**

| Statistic | Formula | Notes |
|-----------|---------|-------|
| **Mean** | $\frac{\sum x_i}{n}$ | Standard arithmetic mean |
| **Median** | `samples[n / 2]` | Integer division -- for even $n$, takes the upper-middle element (not the average of the two middle values) |
| **P99** | `samples[ceil(0.99 * n) - 1]` | Ceiling-based index; clamped to `n - 1` via `std::min` to prevent out-of-bounds |

**GOTCHA -- P99 with small sample sizes.** The P99 calculation is only statistically meaningful when $n \geq 100$. With fewer samples, `ceil(0.99 * n) - 1` collapses toward the maximum:

| $n$ | P99 index | What it actually is |
|-----|-----------|---------------------|
| 1   | 0         | The single sample (= mean = median) |
| 4   | 3         | The maximum value |
| 10  | 9         | The maximum value |
| 50  | 49        | The maximum value |
| 100 | 98        | True 99th percentile |
| 1000| 989       | True 99th percentile |

In a typical benchmark run with 4 users and 3 turns each, there are only 12 TTFT samples and 12 TPOT samples -- making the reported P99 essentially the maximum observed value. The ITL sample count is larger (total tokens minus one), so P99 is somewhat more reliable there.

**Mutating sort:** The function takes `std::vector<double>&` (non-const reference) and sorts in place. After `compute_stats()` returns, the input vector is sorted. This is intentional (avoids a copy) and harmless since the function is only called once per metric at the end of the benchmark -- the vector cannot be iterated in insertion order afterward.

---

**Next:** [Bulk Transfer Metrics](02_bulk_transfer_metrics.md)
