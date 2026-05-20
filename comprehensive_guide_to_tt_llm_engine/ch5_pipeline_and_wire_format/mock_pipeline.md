# MockPipeline: In-Process Queue Backend for Testing

*Source: `mock_pipeline.hpp`*

## Purpose

`MockPipeline` is the simplest concrete implementation of `PipelineInterface`. It exists for one reason: to let the entire scheduling algorithm, speculative decode protocol, and threading model run under unit tests without any hardware, timing constraints, or external dependencies. Despite its simplicity, it faithfully implements the full contract -- FIFO ordering, sentinel termination, and thread-safe concurrent access from Writer and Reader threads.

## Construction

```cpp
MockPipeline(int latency_min_us = 0, int latency_max_us = 0,
             uint32_t seed = 42, float accept_rate = 1.0f);
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `latency_min_us` | 0 | Minimum simulated read latency in microseconds |
| `latency_max_us` | 0 | Maximum simulated read latency in microseconds |
| `seed` | 42 | RNG seed for deterministic test replay |
| `accept_rate` | 1.0 | Probability that a BASE token is "accepted" by the simulated model |

With default parameters, MockPipeline is **instant and deterministic**: `inject()` immediately enqueues a result, and `read_result()` immediately dequeues it.

## Architecture

```
Writer Thread                          Reader Thread
     |                                      |
     v                                      v
  inject(desc)                         read_result()
     |                                      |
     +---> [result_queue_] ----cv_----> pop from queue
     |     (std::queue)                     |
     |                                      +---> optional sleep(latency)
     +---> [inject_log_]                    |
           (std::vector)                    v
                                       return ResultDescriptor
```

`inject()` computes a `ResultDescriptor` synchronously, pushes it into a `std::queue`, and notifies the condition variable. `read_result()` waits on the condition variable, pops the front element, optionally sleeps to simulate latency, and returns it.

## Token Model

The mock's token model is entirely deterministic (given a fixed seed and accept_rate):

### Decode Token Generation

For an injection with `token_id = T` at `position = P`:

```
Given: inject(slot_id=S, token_id=T, position=P, token_type=TYPE)

Result:
  slot_id            = S
  actual_token       = T + 1        (if accepted)
                     = T + 3        (if rejected, BASE tokens only)
  actual_token_type  = TYPE
  actual_token_pos   = P + 1
  predicted_token    = actual_token + 1
  predicted_token_type = SPEC
  predicted_token_pos  = P + 2
```

### Why +1 and +3?

The offsets are arbitrary but chosen to be: (1) **Distinct** -- `+1` (accept) and `+3` (reject) never collide for the same `token_id`. (2) **Predictable** -- test code can compute expected values without querying the mock. (3) **Non-zero offset** -- ensures `actual_token != token_id`, which would incorrectly signal "no change" in some test harnesses.

### Accept/Reject Logic

The accept/reject decision applies **only to `TokenType::BASE` tokens**. Speculative tokens (`TokenType::SPEC`) are always accepted:

```cpp
bool accept = true;
if (desc.token_type == TokenType::BASE && accept_rate_ < 1.0f) {
    std::lock_guard<std::mutex> rng_lock(rng_mtx_);
    accept = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng_) < accept_rate_;
}
result.actual_token = accept ? desc.token_id + 1 : desc.token_id + 3;
```

This models the real hardware behavior where speculative tokens are always processed without a secondary acceptance check -- the acceptance logic lives in the scheduler's result classification (Ch4), not in the device. When `accept_rate_ == 1.0f` (the default), the RNG is never consulted, giving fully deterministic behavior.

### Non-Last Prefill

When `prefill_token_id != EMPTY_TOKEN`, the mock returns a stub result with only `slot_id` set. All other fields remain at defaults (`actual_token == EMPTY_TOKEN`). This matches the contract for chunked prefill (Ch3): non-last prefill chunks produce no decode output.

## Latency Injection

When `latency_max_us > 0`, `read_result()` injects a random delay **after** popping the result from the queue but **before** returning it:

```cpp
if (latency_max_us_ > 0) {
    std::lock_guard<std::mutex> rng_lock(rng_mtx_);
    int delay = std::uniform_int_distribution<int>(latency_min_us_, latency_max_us_)(rng_);
    std::this_thread::sleep_for(std::chrono::microseconds(delay));
}
```

Key design choice: the delay is applied on the **Reader** side, not the Writer side. This means `inject()` is always non-blocking, while `read_result()` simulates the time a real pipeline would take to produce a result. This matches SocketPipeline's asymmetry: `inject()` is a fast H2D socket write, while `read_result()` blocks waiting for the device.

## Inject Log

```cpp
std::vector<InjectDescriptor> get_inject_log() const {
    std::lock_guard<std::mutex> lock(log_mtx_);
    return inject_log_;
}
```

The inject log is a complete, ordered record of every `InjectDescriptor` passed to `inject()`. The log returns a copy (not a reference), so it is safe to inspect while the pipeline is still running. The log has its own mutex (`log_mtx_`) independent of the queue mutex, so logging never contends with the read path.

## Shutdown Behaviour

```cpp
void request_stop() override {
    std::lock_guard lock(mtx_);
    shutdown_ = true;
    cv_.notify_all();
}

void shutdown() override { /* no-op */ }
void reset_kv(uint32_t) override { /* no-op */ }
```

`request_stop()` sets the `shutdown_` flag and wakes any thread blocked in `read_result()`. If results remain queued, they drain first before the sentinel is returned. `shutdown()` is a no-op -- no external device to synchronize with.

## Thread Safety Summary

| Mutex | Protects | Held By |
|-------|----------|---------|
| `mtx_` | `result_queue_`, `shutdown_` flag, `cv_` | `inject()`, `read_result()`, `request_stop()` |
| `rng_mtx_` | `rng_` (Mersenne Twister state) | `inject()` (accept decision), `read_result()` (latency sampling) |
| `log_mtx_` | `inject_log_` | `inject()` (append), `get_inject_log()` (copy) |

Three independent mutexes, three independent concerns. No lock is ever held while another is acquired, so deadlock is impossible.

## Worked Example: Speculative Decode Acceptance

**Setup:**
```cpp
auto pipeline = std::make_unique<MockPipeline>(0, 0, 42, /*accept_rate=*/1.0f);
```

**Step 1 -- Inject a BASE decode token:**
```cpp
pipeline->inject({.slot_id = 3, .token_id = 100, .position = 5,
                  .token_type = TokenType::BASE});
```

**Step 2 -- Read and verify:**
```cpp
auto result = pipeline->read_result();
// result.slot_id            == 3       (same slot)
// result.actual_token       == 101     (100 + 1, accepted)
// result.actual_token_type  == BASE
// result.actual_token_pos   == 6       (5 + 1)
// result.predicted_token    == 102     (101 + 1)
// result.predicted_token_type == SPEC
// result.predicted_token_pos == 7      (5 + 2)
```

## Worked Example: Speculative Decode Rejection

**Setup:**
```cpp
auto pipeline = std::make_unique<MockPipeline>(0, 0, 42, /*accept_rate=*/0.0f);
```

**Step 1 -- Inject a SPEC token (always accepted regardless of accept_rate):**
```cpp
pipeline->inject({.slot_id = 3, .token_id = 102, .position = 7,
                  .token_type = TokenType::SPEC});
auto spec_result = pipeline->read_result();
// spec_result.actual_token == 103  (102 + 1, SPEC always accepted)
```

**Step 2 -- Inject the BASE verification token (rejected at accept_rate=0.0):**
```cpp
pipeline->inject({.slot_id = 3, .token_id = 102, .position = 7,
                  .token_type = TokenType::BASE});
auto base_result = pipeline->read_result();
// base_result.actual_token == 105  (102 + 3, rejected!)
```

**Step 3 -- Scheduler detects mismatch:**
```
spec_result.actual_token (103) != base_result.actual_token (105)
```

The scheduler discards the speculative result, uses 105 as the actual next token, and restarts speculation.

## Worked Example: Prefill Sequence

```cpp
auto pipeline = std::make_unique<MockPipeline>();

// Non-last prefill chunk 1
pipeline->inject({.slot_id = 0, .token_id = 50, .prefill_token_id = 10, .position = 0});
auto r1 = pipeline->read_result();
assert(r1.slot_id == 0);
assert(r1.actual_token == EMPTY_TOKEN);  // Non-last prefill: no decode token

// Non-last prefill chunk 2
pipeline->inject({.slot_id = 0, .token_id = 50, .prefill_token_id = 11, .position = 1});
auto r2 = pipeline->read_result();
assert(r2.actual_token == EMPTY_TOKEN);

// Last prefill: prefill_token_id == EMPTY_TOKEN triggers decode path
pipeline->inject({.slot_id = 0, .token_id = 50, .position = 2});
auto r3 = pipeline->read_result();
assert(r3.actual_token == 51);      // 50 + 1 (decode token model applies)
assert(r3.predicted_token == 52);   // 51 + 1

// Verify the full sequence via inject log
auto log = pipeline->get_inject_log();
assert(log.size() == 3);
assert(log[0].prefill_token_id == 10);
assert(log[1].prefill_token_id == 11);
assert(log[2].prefill_token_id == EMPTY_TOKEN);  // Last chunk / decode
```

---

| | Navigation | |
|:---|:---:|---:|
| [PipelineInterface](./pipeline_interface.md) | [Table of Contents](../index.md) | [PipelineSimulator](./pipeline_simulator.md) |
