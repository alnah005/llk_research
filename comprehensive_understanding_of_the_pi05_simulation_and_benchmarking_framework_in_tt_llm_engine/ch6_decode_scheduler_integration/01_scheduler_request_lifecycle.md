# 6.1 Scheduler Request Lifecycle

## Overview

`pi05_pipeline_runner` creates a `DecodeScheduler` backed by either `PipelineSimulatorConfig` (simulation mode) or `SocketConfig` (hardware mode), configures it with Pi0.5-specific `SchedulerParams`, and then drives it through a rigid ALLOCATE-all / SUBMIT-per-turn / EVICT-all lifecycle. This section traces that lifecycle in detail, explaining each request type and why Pi0.5 deliberately avoids `CONTINUE`.

## 6.1.1 Creating the DecodeScheduler

The `DecodeScheduler` constructor accepts a `PipelineConfig` variant and a `SchedulerParams` struct:

```cpp
// include/tt_llm_engine/scheduler/decode/decode_scheduler.hpp, line 22
DecodeScheduler(pipeline::PipelineConfig config, SchedulerParams params = {});
```

`PipelineConfig` is a `std::variant<MockConfig, SocketConfig, PipelineSimulatorConfig>` (defined at `include/tt_llm_engine/pipeline/pipeline_types.hpp`, line 98). The variant determines which `PipelineInterface` implementation backs the scheduler. Pi0.5 uses two of the three options:

**Simulation mode** (default, no sockets):

```cpp
// examples/pi05_pipeline_runner.cpp, lines 1208-1218
pl::PipelineSimulatorConfig sim_config{
    .num_stages = total_stages,
    .stage_duration_us = effective_stage_us,
    .accept_rate = accept_rate,
    .seed = seed,
    .batch_prefill = true,
};
pm::DecodeScheduler mgr(sim_config, pm::SchedulerParams{
    .max_users = num_users,
    .skip_eos_writeback = true,
});
```

**Socket mode** (bulk_only hybrid -- uses `PipelineSimulator` for token pipeline, real sockets for bulk data):

```cpp
// examples/pi05_pipeline_runner.cpp, lines 881-892
pl::PipelineSimulatorConfig sim_config{
    .num_stages = total_stages,
    .stage_duration_us = effective_stage_us,
    .accept_rate = accept_rate,
    .seed = seed,
    .batch_prefill = true,
};
mgr = std::make_unique<pm::DecodeScheduler>(sim_config, pm::SchedulerParams{
    .max_users = num_users,
    .skip_eos_writeback = true,
});
```

**Socket mode** (full `SocketPipeline`):

```cpp
// examples/pi05_pipeline_runner.cpp, lines 893-901
pl::SocketConfig socket_cfg{
    pipeline_config.socket.h2d_token_socket_id,
    pipeline_config.socket.d2h_token_socket_id,
    pipeline_config.socket.connect_timeout_ms,
};
mgr = std::make_unique<pm::DecodeScheduler>(socket_cfg, pm::SchedulerParams{
    .max_users = num_users,
    .skip_eos_writeback = true,
});
```

In every case, Pi0.5 sets exactly two `SchedulerParams` fields: `max_users` and `skip_eos_writeback = true`. The `batch_prefill` flag lives on `PipelineSimulatorConfig`, not `SchedulerParams`. Their mechanics are detailed in [Section 6.2](02_batch_prefill_and_skip_eos.md).

## 6.1.2 Request Types

All interactions with `DecodeScheduler` flow through a single push/pop interface using `ISRequest` messages. Five request types are defined:

```cpp
// include/tt_llm_engine/scheduler/decode/decode_types.hpp, lines 61-67
enum class RequestType : uint8_t {
    ALLOCATE = 1,
    SUBMIT = 2,
    CONTINUE = 3,
    EVICT = 4,
    STOP = 5,
};
```

| Type | Purpose | Pi0.5 usage |
|------|---------|-------------|
| `ALLOCATE` | Reserve a slot; scheduler returns a `slot_id` | Yes -- all slots allocated upfront |
| `SUBMIT` | Start prefill+decode for a slot with fresh context (position 0) | Yes -- once per turn |
| `CONTINUE` | Append tokens to existing KV context and resume decode | **No** -- never used |
| `EVICT` | Release a slot, reset KV, return to free pool | Yes -- at shutdown (simulation mode) |
| `STOP` | Halt generation without releasing the slot | Not used by Pi0.5 |

### The ISRequest Struct

```cpp
// include/tt_llm_engine/scheduler/decode/decode_types.hpp, lines 81-87
struct ISRequest {
    RequestType type = RequestType::ALLOCATE;
    uint32_t request_id = 0;
    uint32_t slot_id = INVALID_SLOT;
    std::vector<uint32_t> tokens;
    GenerationParams gen;
};
```

The `request_id` is a monotonically increasing identifier assigned by the caller. It appears in `SchedulerResponse` so the caller can correlate responses to requests. The `slot_id` field is unused for ALLOCATE (the scheduler assigns it) but required for all other request types. For `SUBMIT`, the caller also sets `tokens` (the prompt) and `gen.max_new_tokens`.

### Pushing Requests

Requests are pushed via a bounded queue. The runner uses a retry-loop wrapper:

```cpp
// examples/pi05_pipeline_runner.cpp, lines 111-115
void push_request(pm::DecodeScheduler& mgr, const pm::ISRequest& req) {
    while (!mgr.push_request(req)) {
        std::this_thread::yield();
    }
}
```

## 6.1.3 The ALLOCATE Phase

Pi0.5 allocates **all** user slots upfront before any work begins:

```cpp
// examples/pi05_pipeline_runner.cpp, lines 1225-1240
for (uint32_t u = 0; u < num_users; u++) {
    pm::ISRequest req{};
    req.type = pm::RequestType::ALLOCATE;
    req.request_id = req_id++;
    push_request(mgr, req);
}
for (uint32_t u = 0; u < num_users; u++) {
    pm::SchedulerResponse resp{};
    if (!poll_response_timed(mgr, resp, std::chrono::seconds(timeout_s))
        || resp.error_code != 0) {
        throw std::runtime_error(
            "ALLOCATE failed for user " + std::to_string(u) + " ...");
    }
    users[u].slot_id = resp.slot_id;
}
```

This is a two-pass pattern: push all N requests, then poll all N responses. Each `ALLOCATE` goes through the scheduler's `api_loop()`, which performs:

```cpp
// src/scheduler/decode/decode_scheduler.cpp, lines 661-678
case RequestType::ALLOCATE: {
    uint32_t uid = free_ids.allocate();
    if (uid != INVALID_SLOT) {
        user_table.reset(uid);
        evict_pending.clear(uid);
        stop_pending.clear(uid);
        spec_state.reset(uid);
    }
    SchedulerResponse resp{
        .request_id = req.request_id,
        .slot_id = uid,
        .error_code = (uid == INVALID_SLOT) ? 1 : 0,
        .request_type = RequestType::ALLOCATE,
    };
    while (!response_queue.try_push(resp)) {
        _mm_pause();
    }
    break;
}
```

If no slot is available, `error_code = 1` and `slot_id = INVALID_SLOT`.

### SchedulerResponse

```cpp
// include/tt_llm_engine/scheduler/decode/decode_types.hpp, lines 89-94
struct SchedulerResponse {
    uint32_t request_id = 0;
    uint32_t slot_id = INVALID_SLOT;
    int32_t error_code = 0;
    RequestType request_type = RequestType::ALLOCATE;
};
```

The `error_code` is 0 on success. For `ALLOCATE`, `slot_id` is the assigned slot. The `poll_response_timed` helper spins with a deadline:

```cpp
// examples/pi05_pipeline_runner.cpp, lines 117-125
bool poll_response_timed(pm::DecodeScheduler& mgr, pm::SchedulerResponse& resp,
                         std::chrono::seconds timeout) {
    auto deadline = Clock::now() + timeout;
    while (Clock::now() < deadline) {
        if (mgr.try_pop_response(resp)) return true;
        std::this_thread::yield();
    }
    return false;
}
```

## 6.1.4 The SUBMIT Phase

After allocation, Pi0.5 submits work for each user. Each `SUBMIT` resets the slot to position 0 and begins prefill from scratch:

```cpp
// examples/pi05_pipeline_runner.cpp, lines 1258-1277
for (uint32_t u = 0; u < num_users; u++) {
    if (u > 0 && stagger_us > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(stagger_us));
    }
    auto prompt = make_prompt(prompt_len, u * max_turns);
    pm::ISRequest req{};
    req.type = pm::RequestType::SUBMIT;
    req.request_id = req_id++;
    req.slot_id = users[u].slot_id;
    req.tokens = prompt;
    req.gen.max_new_tokens = max_decode;
    push_request(mgr, req);
    users[u].current_turn = 1;
    users[u].turn_start = Clock::now();
    ...
}
```

Inside `DecodeScheduler::Impl::handle_api_requests`, the `SUBMIT` handler resets all per-slot state:

```cpp
// src/scheduler/decode/decode_scheduler.cpp, lines 680-701
case RequestType::SUBMIT: {
    uint32_t uid = req.slot_id;
    prompt_table.store(uid, req.tokens.data(), static_cast<uint32_t>(req.tokens.size()));
    user_table.state[uid].store(UserState::PREFILL, std::memory_order_release);
    user_table.current_position[uid].store(0, std::memory_order_relaxed);
    user_table.prefill_pos[uid] = 0;
    user_table.prefill_start_pos[uid] = 0;
    user_table.max_new_tokens[uid] = req.gen.max_new_tokens;
    user_table.tokens_generated[uid] = 0;
    user_table.in_flight_count[uid].store(0, std::memory_order_relaxed);
    user_table.prefill_chunk_remaining[uid] = params.chunk_size;
    // ... generation params ...
    spec_state.reset(uid);
    prefill_queue.push(uid);
    break;
}
```

The critical detail: **`current_position = 0` and `prefill_start_pos = 0`**. This means the entire KV cache for this slot is logically overwritten from position 0. There is no KV continuity from the previous turn.

## 6.1.5 Why SUBMIT, Not CONTINUE

In a standard LLM chat scenario, multi-turn conversation uses `CONTINUE`: turn N+1 appends new tokens to the existing KV cache, preserving the conversation's context window. The `handle_local_continue` handler reads `current_position` and begins prefill from that offset:

```cpp
// src/scheduler/decode/decode_scheduler.cpp, lines 931-952
void handle_local_continue(const ISRequest& req) {
    uint32_t uid = req.slot_id;
    uint32_t start_pos = user_table.current_position[uid].load(std::memory_order_relaxed);
    prompt_table.store(uid, req.tokens.data(), static_cast<uint32_t>(req.tokens.size()));
    user_table.state[uid].store(UserState::PREFILL, std::memory_order_release);
    user_table.prefill_pos[uid] = start_pos;
    user_table.prefill_start_pos[uid] = start_pos;
    // ...
}
```

Pi0.5 is a **vision-language-action (VLA)** model for robotics. Each inference turn processes a new camera frame independently:

| Aspect | CONTINUE | SUBMIT |
|--------|----------|--------|
| KV position | Resumes from `current_position` | Resets to 0 |
| Prior context | Preserved in KV cache | Implicitly overwritten |
| Use case | Multi-turn chat (context accumulates) | Independent frames |

- **No KV continuity**: the "prompt" for turn N+1 is an entirely new set of image tokens, not a continuation of turn N's context. There is no conversational history to preserve in KV cache.
- **Action history travels out-of-band**: the model's output from turn N (a 50x32 bfloat16 action tensor, 3,200 bytes) is fed back as `action_history` through the **bulk H2D socket channel**, not through the token path. This data is injected into the model's input tensor, not appended to KV.
- **Position reset is intentional**: each frame starts at position 0 to get a clean KV slate.

The runner's re-SUBMIT on completion confirms this pattern:

```cpp
// examples/pi05_pipeline_runner.cpp, lines 1344-1358
auto prompt = make_prompt(prompt_len, it->second * max_turns + u.current_turn);
pm::ISRequest req{};
req.type = pm::RequestType::SUBMIT;  // NOT CONTINUE
req.request_id = req_id++;
req.slot_id = u.slot_id;
req.tokens = prompt;
req.gen.max_new_tokens = max_decode;
push_request(mgr, req);
u.current_turn++;
```

Using SUBMIT resets `prefill_start_pos = 0`, which means every position in the KV cache will be overwritten during the next prefill. Using CONTINUE would append to existing KV, eventually exhausting the context window and accumulating stale attention from irrelevant prior frames. This fresh-context pattern is the reason `skip_eos_writeback` exists (see [Section 6.2](02_batch_prefill_and_skip_eos.md)).

## 6.1.6 OutputMessage and the Decode Loop

Decoded tokens arrive via `try_pop_output()`. The `OutputMessage` struct:

```cpp
// include/tt_llm_engine/scheduler/decode/decode_types.hpp, lines 96-103
struct OutputMessage {
    uint32_t slot_id = INVALID_SLOT;
    uint32_t token_id = EMPTY_TOKEN;
    bool is_complete = false;
    bool ctx_exhausted = false;
    uint32_t tokens_generated = 0;
    uint32_t generation = 0;
};
```

| Field | Meaning |
|-------|---------|
| `slot_id` | Which user slot produced this token |
| `token_id` | The decoded token (or `EMPTY_TOKEN` on a completion-only message) |
| `is_complete` | True on the final message for this generation |
| `ctx_exhausted` | True if completion was due to hitting `max_seq_len` |
| `tokens_generated` | Running count within this generation |
| `generation` | Monotonic counter, incremented on EVICT/STOP to detect stale messages |

The runner's main loop polls `try_pop_output()` in a non-blocking spin:

```cpp
// examples/pi05_pipeline_runner.cpp, lines 1286-1296
while (users_done < num_users) {
    pm::OutputMessage out;
    if (!mgr.try_pop_output(out)) {
        // throttled display redraw
        std::this_thread::yield();
        continue;
    }
    // process out.slot_id, out.token_id, out.is_complete ...
}
```

When `is_complete` is true, the runner either marks the user done (if `current_turn >= max_turns || ctx_exhausted`) or re-SUBMITs for the next turn.

## 6.1.7 The EVICT Phase (Simulation Mode)

After all turns complete, the simulation-mode path sends `EVICT` for every slot, waits for all to reach `UserState::INACTIVE`, and drains remaining output messages. The socket-mode path skips explicit EVICT, instead calling `mgr->stop()` directly after the sentinel shutdown protocol (see [Chapter 4](../ch4_bulk_h2d_d2h_socket_protocol/index.md)). For the full teardown code and analysis, see [Section 6.3.8](./03_user_session_management.md).

## 6.1.8 SchedulerParams for Pi0.5

The full `SchedulerParams` struct:

```cpp
// include/tt_llm_engine/scheduler/decode/decode_types.hpp, lines 29-52
struct SchedulerParams {
    uint32_t max_users = DEFAULT_MAX_USERS;       // Pi0.5: set to num_users
    uint32_t chunk_size = DEFAULT_CHUNK_SIZE;      // default 24
    uint32_t max_seq_len = DEFAULT_MAX_SEQ_LEN;   // default 131072
    uint32_t eos_token = DEFAULT_EOS_TOKEN;        // default 1
    uint32_t think_open_token_id = EMPTY_TOKEN;    // unused
    uint32_t think_close_token_id = EMPTY_TOKEN;   // unused
    int writer_cpu = AUTO_CPU;                     // auto
    int reader_cpu = AUTO_CPU;                     // auto
    int api_cpu = AUTO_CPU;                        // auto
    bool skip_eos_writeback = false;               // Pi0.5: true
};
```

Pi0.5 overrides two fields:

| Parameter | Pi0.5 Value | Default | Reason |
|-----------|-------------|---------|--------|
| `max_users` | `num_users` (CLI) | 64 | Sizes internal tables (user_table, prompt_table, free_ids, decode_staging) to exactly the concurrent user count, avoiding wasted memory |
| `skip_eos_writeback` | `true` | `false` | No CONTINUE, so EOS writeback is wasted work |

All other parameters (chunk_size, max_seq_len, eos_token, thinking tokens) use their defaults. Pi0.5 does not use speculative decoding or thinking-phase sampling, so the thinking token gates remain disabled (`EMPTY_TOKEN`).

## 6.1.9 Scheduler Internal Threading Model

The `DecodeScheduler` runs three dedicated threads (created on `start()`):

1. **`writer_thread`**: Injects tokens into the pipeline. Priority: decode staging entries first, then prefill queue chunks.
2. **`reader_thread`**: Reads `ResultDescriptor`s from the pipeline, emits `OutputMessage`s, manages spec-decode verification.
3. **`api_thread`**: Processes `ISRequest`s from the request queue (ALLOCATE, SUBMIT, CONTINUE, EVICT, STOP handlers).

```cpp
// src/scheduler/decode/decode_scheduler.cpp, lines 193-195
writer_thread = std::thread([this] { writer_loop(); });
reader_thread = std::thread([this] { reader_loop(); });
api_thread = std::thread([this] { api_loop(); });
```

Each thread can be pinned to a specific CPU core via `SchedulerParams::writer_cpu`, `reader_cpu`, `api_cpu`. By default (`AUTO_CPU`), the scheduler auto-assigns cores spread across the available CPU set.

## 6.1.10 Complete Lifecycle Diagram

```
Phase 1: Startup
  ALLOCATE x num_users  ────>  SchedulerResponse { slot_id }

Phase 2: Turn 1
  SUBMIT x num_users (staggered)  ────>  OutputMessage stream { token_id }
                                         OutputMessage { is_complete=true }

Phase 3: Turns 2..N (if current_turn < max_turns && !ctx_exhausted)
  re-SUBMIT (NOT CONTINUE)  ────>  OutputMessage stream ...

Phase 4: Shutdown
  EVICT x num_users  ────>  wait INACTIVE  ────>  stop()
```

---

**Next:** [Batch Prefill and skip_eos_writeback](02_batch_prefill_and_skip_eos.md)
