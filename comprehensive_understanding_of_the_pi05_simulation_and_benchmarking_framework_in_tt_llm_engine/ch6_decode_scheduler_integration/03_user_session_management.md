# 6.3 User Session Management

## Overview

The `pi05_pipeline_runner` manages multiple concurrent user sessions, each representing an independent robotics agent performing multi-turn inference. This section covers the `UserSession` struct, the `slot_to_user` routing map, staggered initial submission, the multi-turn re-SUBMIT loop with action-history reuse, and the differences between simulation and socket mode sessions.

## 6.3.1 The UserSession Struct

Each concurrent user is tracked by a `UserSession` instance:

```cpp
// examples/pi05_pipeline_runner.cpp, lines 135-153
struct UserSession {
    uint32_t slot_id = pm::INVALID_SLOT;
    uint32_t current_turn = 0;
    uint32_t total_tokens = 0;
    bool ctx_exhausted = false;
    bool done = false;

    TimePoint turn_start{};
    TimePoint first_token_time{};
    TimePoint last_token_time{};
    uint32_t turn_token_count = 0;
    bool first_token_received = false;

    std::string status_tag() const {
        if (done && ctx_exhausted) return " [ctx_exhausted]";
        if (done) return " [done]";
        return "";
    }
};
```

| Field | Purpose |
|-------|---------|
| `slot_id` | DecodeScheduler slot assigned during ALLOCATE; maps 1:1 to a KV cache slot |
| `current_turn` | Which turn this user is on (1-indexed after first SUBMIT) |
| `total_tokens` | Cumulative tokens across all turns (for final reporting) |
| `ctx_exhausted` | Set if any turn hit `max_seq_len` (propagated from `OutputMessage::ctx_exhausted`) |
| `done` | Terminal flag: `true` when `current_turn >= max_turns` or `ctx_exhausted` |
| `turn_start` | Timestamp when the current turn's SUBMIT was pushed |
| `first_token_time` | Timestamp of the first decoded token in the current turn (for TTFT) |
| `last_token_time` | Timestamp of the most recent token (for TPOT calculation) |
| `turn_token_count` | Tokens decoded in the current turn only (reset on re-SUBMIT) |
| `first_token_received` | Guards single-shot TTFT recording per turn |

The `UserSession` is purely host-side bookkeeping. It has no representation inside `DecodeScheduler` -- the scheduler tracks its own internal per-slot state via `UserTable` (state, position, generation counts). The runner bridges the two via `slot_id`.

Users are stored in a flat vector sized to `num_users`:

```cpp
// examples/pi05_pipeline_runner.cpp, line 1223
std::vector<UserSession> users(num_users);
```

## 6.3.2 The slot_to_user Routing Map

`OutputMessage` from the scheduler identifies the source by `slot_id`, but the runner tracks users by index. The `slot_to_user` map bridges this:

```cpp
// examples/pi05_pipeline_runner.cpp, lines 1243-1246
std::unordered_map<uint32_t, uint32_t> slot_to_user;
for (uint32_t u = 0; u < num_users; u++) {
    slot_to_user[users[u].slot_id] = u;
}
```

This map is built once after ALLOCATE and remains immutable for the benchmark's lifetime. In the main loop, every `OutputMessage` is routed through this map:

```cpp
// examples/pi05_pipeline_runner.cpp, lines 1299-1300
auto it = slot_to_user.find(out.slot_id);
if (it == slot_to_user.end()) continue;
UserSession& u = users[it->second];
```

Messages with unknown `slot_id` values are silently dropped. This is a safety measure -- in theory, every slot_id in an `OutputMessage` should match one of the allocated slots. But the `generation` counter mechanism in the scheduler can produce stale messages from a previous generation after EVICT/STOP, which would map to a recycled slot. The lookup guard prevents misrouting.

## 6.3.3 Staggered Initial Submission

To avoid a thundering-herd effect where all users simultaneously enter the prefill queue, Pi0.5 staggers the initial SUBMIT submissions:

```cpp
// examples/pi05_pipeline_runner.cpp, lines 1258-1261
for (uint32_t u = 0; u < num_users; u++) {
    if (u > 0 && stagger_us > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(stagger_us));
    }
    // ... SUBMIT ...
}
```

The stagger interval defaults to:

```cpp
// examples/pi05_pipeline_runner.cpp, lines 831-832
const uint32_t stagger_us = stagger_explicit ? stagger_us_cli
    : static_cast<uint32_t>(total_latency_us / num_users);
```

$$\text{stagger\_us} = \frac{T_{\text{pipeline\_total}}}{\text{num\_users}}$$

This spaces submissions evenly across one full pipeline latency. The rationale: if all $N$ users submit simultaneously, their prefill tokens compete for pipeline bandwidth. Spacing them by $T/N$ microseconds ensures that user $k$'s prefill enters the pipeline just as user $k-1$'s prefill has advanced one pipeline-stage's worth, achieving near-optimal pipeline utilization.

For example, with 4 users and a pipeline latency of $T = 1{,}320\mu s$ (220 stages $\times$ 6us/stage):

| User | Submission delay | Pipeline entry |
|------|-----------------|----------------|
| 0 | 0 us | $t = 0$ |
| 1 | 330 us | $t = 330\mu s$ |
| 2 | 660 us | $t = 660\mu s$ |
| 3 | 990 us | $t = 990\mu s$ |

By the time user 3 enters the pipeline, user 0's prefill is 75% through the stages. The `--stagger-us` CLI flag allows overriding this default, or set to 0 to disable staggering entirely.

### Why Stagger Matters

Without staggering, all $N$ users submit simultaneously. The prefill queue processes them in order, and the pipeline must sequentially prefill each user. The first user's TTFT is approximately $S \times d$ (one pipeline traversal), but the last user's TTFT is approximately $N \times S \times d$ (waiting for all prior prefills). This creates:

- High variance in TTFT measurements
- Unrealistically pessimistic P99 TTFT
- Poor pipeline utilization during the initial ramp-up

With staggering, each user enters the pipeline after the previous user's prefill is partially through, overlapping prefill and decode across users.

## 6.3.4 The Multi-Turn Re-SUBMIT Loop

The core of Pi0.5's session management is the completion handler that decides whether to re-SUBMIT or mark the user done.

### Token Processing

Each non-empty token updates timing state:

```cpp
// examples/pi05_pipeline_runner.cpp, lines 1302-1320
if (out.token_id != pm::EMPTY_TOKEN) {
    u.total_tokens++;
    grand_running_total++;
    auto now = Clock::now();
    if (!u.first_token_received) {
        double ttft_ms = std::chrono::duration<double, std::milli>(
            now - u.turn_start).count();
        ttft_samples.push_back(ttft_ms);
        u.first_token_time = now;
        u.first_token_received = true;
    }
    if (global_token_seen) {
        double itl_ms = std::chrono::duration<double, std::milli>(
            now - last_global_token_time).count();
        itl_samples.push_back(itl_ms);
    }
    // ...
}
```

TTFT (Time To First Token) is measured per-turn, from the SUBMIT timestamp to the first token arrival. ITL (Inter-Token Latency) is measured globally across all users.

### Completion and Re-SUBMIT

On completion, the runner either marks the user done or begins the next turn:

```cpp
// examples/pi05_pipeline_runner.cpp, lines 1326-1358
if (!out.is_complete) continue;

u.ctx_exhausted = out.ctx_exhausted;

if (u.turn_token_count > 1) {
    double tpot_ms = std::chrono::duration<double, std::milli>(
        u.last_token_time - u.first_token_time).count() / (u.turn_token_count - 1);
    tpot_samples.push_back(tpot_ms);
}

active_users--;

if (u.current_turn >= max_turns || u.ctx_exhausted) {
    u.done = true;
    users_done++;
    continue;
}

// Re-SUBMIT for next turn
auto prompt = make_prompt(prompt_len, it->second * max_turns + u.current_turn);
pm::ISRequest req{};
req.type = pm::RequestType::SUBMIT;
req.request_id = req_id++;
req.slot_id = u.slot_id;
req.tokens = prompt;
req.gen.max_new_tokens = max_decode;
push_request(mgr, req);
u.current_turn++;
u.turn_start = Clock::now();
u.first_token_received = false;
u.turn_token_count = 0;
active_users++;
peak_concurrent = std::max(peak_concurrent, active_users);
```

The control flow is:

1. On `is_complete`, compute TPOT for the completed turn (if more than 1 token was produced)
2. If `current_turn >= max_turns` or `ctx_exhausted`: mark done, increment `users_done`
3. Otherwise: generate a new prompt, send a fresh `SUBMIT`, reset per-turn timing state
4. Track `active_users` for peak concurrency reporting

Note the timing reset: `turn_start = Clock::now()`, `first_token_received = false`, `turn_token_count = 0`. Each turn produces independent TTFT and TPOT samples.

### TPOT Formula

Per-user time-per-output-token is computed on each turn completion:

$$
\text{TPOT} = \frac{t_{\text{last}} - t_{\text{first}}}{n_{\text{tokens}} - 1}
$$

The `turn_token_count > 1` guard prevents division by zero when a turn produces exactly one token. This measures the average inter-token interval during decode, excluding the first token (which includes prefill latency and is tracked separately as TTFT).

## 6.3.5 Action History Reuse: D2H Output to H2D Input

In socket mode (real hardware), the runner implements the robotics feedback loop where the model's action output becomes the next turn's action history. This runs exclusively through the bulk socket path.

### Buffers

```cpp
// examples/pi05_pipeline_runner.cpp, lines 976-978
std::vector<uint8_t> action_output(ACTION_OUTPUT_BYTES);
std::vector<uint8_t> action_history(pipeline_config.pixel.action_history_bytes);
```

### Turn 1: No action history

```cpp
// examples/pi05_pipeline_runner.cpp, lines 991-993
double h2d_us = bulk_h2d->send(users[u].slot_id,
    pixel_data.data(), static_cast<uint32_t>(pixel_data.size()),
    nullptr, 0);  // no action history on first turn
```

The first turn sends pixel data only. The `action_data` pointer is `nullptr` and `action_bytes` is 0.

### Turn N+1: Feed action history from turn N

After the D2H read completes for a turn:

```cpp
// examples/pi05_pipeline_runner.cpp, lines 1095-1107
auto pixel_data = generate_synthetic_pixels(pipeline_config.pixel,
    it->second * max_turns + u.current_turn);
std::memcpy(action_history.data(), action_output.data(),
    std::min(static_cast<uint32_t>(action_output.size()),
             pipeline_config.pixel.action_history_bytes));

double h2d_us = bulk_h2d->send(u.slot_id,
    pixel_data.data(), static_cast<uint32_t>(pixel_data.size()),
    action_history.data(), pipeline_config.pixel.action_history_bytes);
```

### Payload Sizes

The `PixelPayloadConfig` struct defines the dimensions:

```cpp
// examples/pi05_pipeline_runner.cpp, lines 231-239
struct PixelPayloadConfig {
    uint32_t width = 224;
    uint32_t height = 224;
    uint32_t channels = 3;
    uint32_t frames = 3;
    uint32_t action_history_bytes = 4096;

    uint32_t pixel_bytes() const { return width * height * channels * frames; }
    uint32_t total_bytes() const { return pixel_bytes() + action_history_bytes; }
};
```

| Component | Size |
|-----------|------|
| Pixel data | $224 \times 224 \times 3 \times 3 = 451{,}584$ bytes |
| Action history (H2D) | 4,096 bytes (action output is 3,200 bytes, padded to 4,096 for tensor alignment) |
| Total H2D payload | $451{,}584 + 4{,}096 + 8_{\text{header}} = 455{,}688$ bytes |
| Action output (D2H) | $50 \times 32 \times 2 = 3{,}200$ bytes + 8 byte header |

### Data Flow Diagram

```
Turn 1                          Turn 2                          Turn 3
+----------+                    +----------+                    +----------+
| Pixels_1 |--H2D--+           | Pixels_2 |--H2D--+           | Pixels_3 |--H2D--+
| (no hist)|       |           |+Action_1 |       |           |+Action_2 |       |
+----------+       v           +----------+       v           +----------+       v
              +---------+                    +---------+                    +---------+
              | Device  |                    | Device  |                    | Device  |
              |Inference|                    |Inference|                    |Inference|
              +----+----+                    +----+----+                    +----+----+
                   |                              |                              |
                   v                              v                              v
              +---------+                    +---------+                    +---------+
              |Action_1 |--D2H-->copy-->     |Action_2 |--D2H-->copy-->     |Action_3 |--D2H-->
              |(3200B)  |                    |(3200B)  |                    |(3200B)  |  done
              +---------+                    +---------+                    +---------+
```

## 6.3.6 Prompt Seed Convention

Each turn uses a distinct prompt seed to ensure different synthetic prompt tokens:

```cpp
// examples/pi05_pipeline_runner.cpp, lines 1262 (initial), 1344 (re-submit)
auto prompt = make_prompt(prompt_len, it->second * max_turns + u.current_turn);
```

The seed formula `user_index * max_turns + current_turn` guarantees that every (user, turn) pair produces a unique prompt, preventing the simulator from accidentally collapsing distinct requests into identical pipeline behavior. The `make_prompt` function generates a deterministic sequence:

```cpp
// examples/pi05_pipeline_runner.cpp, lines 127-133
std::vector<uint32_t> make_prompt(uint32_t len, uint32_t seed) {
    std::vector<uint32_t> prompt(len);
    for (uint32_t i = 0; i < len; i++) {
        prompt[i] = seed + i + 2;
    }
    return prompt;
}
```

The `+2` offset avoids token IDs 0 and 1 (the default EOS token), preventing premature generation termination.

## 6.3.7 Active User Tracking and Peak Concurrency

The runner maintains `active_users` and `peak_concurrent` counters. `active_users` increments on SUBMIT and decrements on `is_complete`. On re-SUBMIT, the sequence is `active_users--` (completion) followed by `active_users++` (new SUBMIT). Since re-SUBMIT happens synchronously in the same loop iteration, `active_users` transiently drops by 1 and then returns. `peak_concurrent` captures the high-water mark via `std::max`.

In steady state with $N$ users and $M$ turns, the peak concurrent is $N$ (all users are active simultaneously after the staggered initial submission).

## 6.3.8 Teardown: EVICT and Cleanup

After all turns complete, simulation mode explicitly EVICTs each slot:

```cpp
// examples/pi05_pipeline_runner.cpp, lines 1362-1396
for (auto& u : users) {
    pm::ISRequest req{};
    req.type = pm::RequestType::EVICT;
    req.request_id = req_id++;
    req.slot_id = u.slot_id;
    push_request(mgr, req);
}
auto teardown_deadline = Clock::now() + std::chrono::seconds(timeout_s);
bool teardown_clean = false;
while (Clock::now() < teardown_deadline) {
    pm::OutputMessage out;
    while (mgr.try_pop_output(out)) {}
    bool all_inactive = true;
    for (auto& u : users) {
        if (mgr.get_user_state(u.slot_id) != pm::UserState::INACTIVE) {
            all_inactive = false;
            break;
        }
    }
    if (all_inactive) {
        teardown_clean = true;
        break;
    }
    std::this_thread::yield();
}
```

The teardown waits for all slots to reach `UserState::INACTIVE`, polling `get_user_state()` until timeout. During the wait, it drains any remaining output messages to prevent the output queue from blocking the scheduler's reader thread.

In socket mode, teardown is different -- the runner sends a bulk H2D sentinel, waits for the device kernel's sentinel echo, then stops the scheduler. Slot EVICTs are skipped because the process terminates via `_exit()`.

## 6.3.9 Simulation vs. Socket Mode Session Differences

| Aspect | Simulation mode | Socket mode |
|--------|----------------|-------------|
| Bulk data | None -- tokens only | H2D pixels + action_history, D2H action output |
| Action history | Not applicable | Copied from D2H buffer to H2D buffer between turns |
| EVICT | Explicit, with drain loop | Skipped; `mgr->stop()` handles cleanup |
| Shutdown | `mgr.stop()` after EVICT drain | Sentinel + echo + `mgr->stop()` + `_exit()` |
| Exit method | `return EXIT_SUCCESS` | `_exit(EXIT_SUCCESS)` to avoid atexit race |

The `_exit()` call in socket mode (`pi05_pipeline_runner.cpp`, line 1196) is a workaround for a tt-metal `ShmResourceTracker` cleanup that races with socket teardown, causing double-free crashes. Simulation mode uses a normal `return` since there are no shared-memory resources to race against.

## 6.3.10 Complete Lifecycle Diagram

```
 Timeline for a single user (3 turns):

 ----[ALLOCATE]---[SUBMIT t=1]---------[complete]---[SUBMIT t=2]--
                  |  prefill  | decode  |           |  prefill  |
                  |<- TTFT -->|         |           |<- TTFT -->|
                              |<-TPOT ->|           |
                                        |
                                        v
                               D2H: action_output_1
                               action_history_2 := action_output_1
                               H2D: pixel_data_2 + action_history_2

 -------[complete]---[SUBMIT t=3]--------[complete]---[EVICT]----
        |           |  prefill  | decode  |           |
        v           |<- TTFT -->|         |
  D2H: action_output_2                   v
                                    D2H: action_output_3
                                    done = true
```

---

**Next:** [Chapter 7 -- Metrics, Reporting, and Benchmark Orchestration](../ch7_metrics_reporting_and_benchmark_orchestration/index.md)
