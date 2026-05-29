# 3.1 Simulator Design

## Overview

`PipelineSimulator` is a purely host-side timing model that faithfully
reproduces three systolic-pipeline invariants -- latency, throughput cap, and
backpressure -- without any background tick thread.  Rather than advancing a
clock at fixed intervals and stamping tokens on tick boundaries, the simulator
computes an exact `exit_time` for every injected token at the moment of
injection and then busy-waits until that time in `read_result()`.

The class lives in a single header:

```
include/tt_llm_engine/pipeline/pipeline_simulator.hpp
```

It implements the `PipelineInterface` abstract base
(`pipeline_interface.hpp`, line 13) and is selected at runtime when the
`DecodeScheduler` receives a `PipelineSimulatorConfig` variant.

---

## 1. Class Anatomy

```cpp
// include/tt_llm_engine/pipeline/pipeline_simulator.hpp:48-50
class PipelineSimulator : public PipelineInterface {
public:
    using clock = std::chrono::steady_clock;
```

### 1.1 Constructor Parameters

```cpp
// include/tt_llm_engine/pipeline/pipeline_simulator.hpp:52-76
PipelineSimulator(uint32_t numStages = 64,
                  uint32_t stageDuration = 44,
                  uint32_t tokenId = EMPTY_TOKEN,
                  float acceptRate = 1.0f,
                  uint32_t seed = 42,
                  uint32_t safeVocabBase = 0,
                  uint32_t safeVocabModulus = 0,
                  bool batchPrefill = false)
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `numStages` | 64 | Number of pipeline stages (e.g. 16 devices x 4 stages) |
| `stageDuration` | 44 | Microseconds per stage tick |
| `tokenId` | `EMPTY_TOKEN` | Fixed decode output token; `EMPTY_TOKEN` uses the mock model |
| `acceptRate` | 1.0 | Speculative-decode accept probability |
| `seed` | 42 | PRNG seed for accept/reject rolls |
| `safeVocabBase` | 0 | Base offset for modular token wrapping |
| `safeVocabModulus` | 0 | Ring size; 0 = disabled, >= 5 required when enabled |
| `batchPrefill` | false | Non-last prefill tokens complete immediately |

### 1.2 Derived Constants

Two timing constants are computed once in the initializer list and never change:

```cpp
// include/tt_llm_engine/pipeline/pipeline_simulator.hpp:61-62
stagePeriod(std::chrono::microseconds(stageDuration)),
totalLatency(stagePeriod * numStages),
```

$$
\text{stagePeriod} = \text{stageDuration} \;\mu\text{s}
$$

$$
\text{totalLatency} = \text{numStages} \times \text{stagePeriod}
$$

For the default values (64 stages, 44 us each):

$$
\text{totalLatency} = 64 \times 44 = 2816 \;\mu\text{s} \approx 2.8 \;\text{ms}
$$

### 1.3 Private State

```cpp
// include/tt_llm_engine/pipeline/pipeline_simulator.hpp:193-214
struct InflightToken {
    clock::time_point exitTime;
    ResultDescriptor result;
};

uint32_t numStages;
clock::duration stagePeriod;
clock::duration totalLatency;
uint32_t tokenId;
float acceptRate;
std::mt19937 rng;
uint32_t safeVocabBase;
uint32_t safeVocabModulus;
bool batchPrefill;

std::mutex mu;
std::condition_variable emitCv;     // notified when a token is queued
std::condition_variable injectCv;   // notified when a token is consumed
std::deque<InflightToken> inflight; // FIFO, ordered by exitTime (monotonic)
clock::time_point lastEnter{};      // for ingest rate cap

std::atomic<bool> stop{false};
```

Key observations:

- **Two condition variables** partition the wait space: `emitCv` wakes the
  reader when new tokens arrive, `injectCv` wakes the writer when the in-flight
  count drops below `numStages`.
- **`inflight`** is a `std::deque` (not `std::queue`) so the reader can peek
  `.front()` without popping.
- **`lastEnter`** records the most recent token's enter timestamp, enabling the
  throughput-cap invariant.  It is default-initialized to the epoch (time_point
  zero), so the first injection always enters immediately -- `clock::now()` is
  always greater than the epoch.
- **`stop`** uses acquire/release ordering for cross-thread visibility without
  the overhead of sequential consistency.

---

## 2. Three Invariants

The simulator enforces three invariants that mirror the behavior of a real
systolic pipeline on Tenstorrent hardware.  These are documented directly in
the class comment block:

```cpp
// include/tt_llm_engine/pipeline/pipeline_simulator.hpp:22-32
// Models a systolic pipeline of `numStages` stages, each of `stageDuration`
// microseconds, with three invariants that mirror real-device behavior:
//
//   1. Latency        -- each accepted token exits exactly
//                       numStages * stageDuration after it enters.
//   2. Throughput cap -- at most one token enters the pipeline per stage
//                       period (peak 1 / stageDuration tokens / second).
//   3. Backpressure   -- at most numStages tokens are in flight at once;
//                       inject() blocks beyond that.
```

### Invariant 1: Latency

Every token that enters the pipeline exits exactly `totalLatency` later:

$$
t_{\text{exit}} = t_{\text{enter}} + \text{numStages} \times \text{stagePeriod}
$$

This is computed in `inject()` at line 104:

```cpp
// include/tt_llm_engine/pipeline/pipeline_simulator.hpp:104
inflight.push_back({enter + totalLatency, makeResult(desc)});
```

### Invariant 2: Throughput Cap

At most one token can enter the pipeline per stage period, yielding a peak
throughput of:

$$
\text{throughput}_{\max} = \frac{1}{\text{stagePeriod}}
$$

For the default 44 us stage period:

$$
\text{throughput}_{\max} = \frac{1}{44 \;\mu\text{s}} \approx 22{,}727 \;\text{tokens/s}
$$

Enforced at line 102 via the `lastEnter` rate limiter:

```cpp
// include/tt_llm_engine/pipeline/pipeline_simulator.hpp:101-103
auto now = clock::now();
auto enter = std::max(now, lastEnter + stagePeriod);
lastEnter = enter;
```

The `max(now, lastEnter + stagePeriod)` expression provides **natural burst
absorption**: if the pipeline has been idle (no injects for a while), `now`
exceeds `lastEnter + stagePeriod`, so the first inject enters immediately.
Subsequent rapid injects are then rate-limited to one per `stagePeriod`.

### Invariant 3: Backpressure

At most `numStages` tokens can be in flight simultaneously.  When the limit is
reached, `inject()` blocks on `injectCv` until a token is consumed (see the
`injectCv.wait()` call at line 91 of the full `inject()` listing in section 3
below).

This mirrors the physical constraint that a systolic pipeline of $N$ stages can
hold at most $N$ tokens at once (one per stage).  When the FIFO is full,
backpressure stalls the writer thread until the reader consumes a token and
calls `injectCv.notify_one()`.

---

## 3. inject() Path

The `inject()` method is called by the `DecodeScheduler`'s writer thread for
every token (prefill or decode) that must pass through the pipeline:

```cpp
// include/tt_llm_engine/pipeline/pipeline_simulator.hpp:80-106
void inject(const InjectDescriptor& desc) override {
    // Batch prefill: non-last prefill tokens complete immediately.
    if (batchPrefill && desc.prefill_token_id != EMPTY_TOKEN) {
        std::lock_guard<std::mutex> lock(mu);
        inflight.push_back({clock::now(), makeResult(desc)});
        emitCv.notify_one();
        return;
    }

    std::unique_lock<std::mutex> lock(mu);
    // Backpressure: real device can't hold more than numStages in-flight.
    injectCv.wait(lock, [this] {
        return inflight.size() < numStages || stop.load(std::memory_order_acquire);
    });
    if (stop.load(std::memory_order_acquire)) return;

    auto now = clock::now();
    auto enter = std::max(now, lastEnter + stagePeriod);
    lastEnter = enter;
    inflight.push_back({enter + totalLatency, makeResult(desc)});
    emitCv.notify_one();
}
```

### Step-by-step execution for a normal (non-batch-prefill) token

1. **Acquire lock** -- `std::unique_lock<std::mutex>`.
2. **Backpressure check** -- if `inflight.size() >= numStages`, block on
   `injectCv` until space opens up.
3. **Stop check** -- if `request_stop()` was called, return immediately.
4. **Compute enter time** -- `enter = max(now, lastEnter + stagePeriod)`.
   This is the rate-limiting step.
5. **Update `lastEnter`** -- record this enter time for the next injection.
6. **Compute exit time and enqueue** -- push `{enter + totalLatency, result}`
   onto the FIFO.
7. **Notify reader** -- `emitCv.notify_one()` wakes the reader if it was
   waiting for a token.

### FIFO ordering guarantee

Because each `enter` time is monotonically non-decreasing (each new enter is at
least `lastEnter + stagePeriod`), and `totalLatency` is a constant, all exit
times in the FIFO are also monotonically non-decreasing.  This means the FIFO
is always in correct emit order -- the head element always has the earliest exit
time.

---

## 4. read_result() Path

The `read_result()` method is called by the `DecodeScheduler`'s reader thread.
It blocks until the next token's exit time has arrived:

```cpp
// include/tt_llm_engine/pipeline/pipeline_simulator.hpp:109-138
ResultDescriptor read_result() override {
    std::unique_lock<std::mutex> lock(mu);
    emitCv.wait(lock, [this] {
        return !inflight.empty() || stop.load(std::memory_order_acquire);
    });
    if (stop.load(std::memory_order_acquire) && inflight.empty()) {
        return ResultDescriptor{.slot_id = INVALID_SLOT};
    }

    auto exitTime = inflight.front().exitTime;
    lock.unlock();

    // Busy-wait until the head token's exit_time.
    while (clock::now() < exitTime) {
        if (stop.load(std::memory_order_acquire)) {
            return ResultDescriptor{.slot_id = INVALID_SLOT};
        }
    }

    lock.lock();
    auto result = std::move(inflight.front().result);
    inflight.pop_front();
    injectCv.notify_one();
    return result;
}
```

### Step-by-step execution

1. **Wait for non-empty FIFO** -- blocks on `emitCv` until at least one token
   is in flight.
2. **Peek exit time** -- reads `exitTime` from the front element.
3. **Unlock** -- drops the mutex so the writer thread can continue injecting
   while the reader waits.
4. **Busy-wait** -- spins on `clock::now() < exitTime` until the token's exit
   time arrives.
5. **Stop check** -- inside the spin loop, checks `stop` to allow prompt
   shutdown.
6. **Re-lock and pop** -- acquires the mutex, moves the result, pops the front
   element, and notifies the writer via `injectCv.notify_one()`.

### Single-reader contract

The comment at line 108 states: "Single-reader contract (DecodeScheduler has one
reader_loop)."  This contract is essential for correctness: once the reader
peeks the front element and unlocks, it relies on no other thread popping that
element.  The `DecodeScheduler` enforces this by having exactly one
`reader_thread` (line 194 of `decode_scheduler.cpp`):

```cpp
// src/scheduler/decode/decode_scheduler.cpp:194
reader_thread = std::thread([this] { reader_loop(); });
```

---

## 5. Why No Tick Thread

The class header comment (lines 40-47) explains the rationale:

```
// Why no tick thread: the previous design ran a busy-spinning thread at
// stageDuration cadence with a token-bucket cap. That added two unnecessary
// costs -- wall-clock tick quantization (every emit landed on a tick
// boundary, padding TPOT by up to one stageDuration on cold sequences)
// and ~1-2us per tick of spin-loop overhead from clock::now() and mutex
// traffic. Per-token timestamps eliminate both: timing precision is now
// bounded only by sleep_until resolution (typically ~1us on Linux), and
// the simulator burns no CPU while idle.
```

**Note:** The comment references `sleep_until` as the timing mechanism, but the implementation actually uses a **busy-wait spin loop** (lines 127-131) for the reasons described below. The comment predates the busy-wait switch and was not updated.

### Problem 1: Tick Quantization Error

A tick-based design advances a clock every `stagePeriod` and only emits tokens
on tick boundaries.  If a token's true exit time falls between ticks, it must
wait until the next tick, adding up to one full `stagePeriod` of padding:

$$
\text{TPOT}_{\text{tick}} = \text{TPOT}_{\text{true}} + \epsilon, \quad 0 \le \epsilon < \text{stagePeriod}
$$

For a 44 us stage period, cold-start sequences could see up to 44 us of
artificial padding per token.  The timestamp model eliminates this: exit times
are computed to the nanosecond and honored at `steady_clock` resolution
(typically < 100 ns on modern Linux).

### Problem 2: Spin-loop Overhead

A tick thread must call `clock::now()` and acquire/release a mutex on every
tick, even when no tokens are in flight.  At 44 us tick intervals, this amounts
to ~22,727 lock acquisitions per second.  The measured overhead was 1-2 us per
tick of pure bookkeeping, consuming ~2-5% of a CPU core at idle.

The timestamp design burns zero CPU when no tokens are in flight (the reader
blocks on `emitCv`), and the only busy-wait happens when a token's exit is
imminent.

### Trade-off: Busy-wait vs sleep_until

The comment at line 124-126 explains the deliberate choice:

```
// Trades CPU burn for sub-microsecond emit timing -- sleep_until on a
// non-RT kernel adds 5-50us of wake-up jitter per token, which lands
// directly in TPOT at high throughput.
```

Using `std::this_thread::sleep_until(exitTime)` would avoid the spin-loop CPU
burn, but non-real-time Linux kernels have a timer slack of 5-50 us for
`nanosleep`-family calls.  At the Pi0.5 default of 44 us per stage, 50 us of
jitter would more than double the apparent TPOT.  The busy-wait guarantees
sub-microsecond timing accuracy at the cost of pegging one core at 100% during
decode.

### Comparative Analysis

| Approach | Timing precision | CPU cost when idle | CPU cost when active |
|---|---|---|---|
| Tick thread | $0$ to $+\text{stagePeriod}$ | High (spin loop) | High (spin loop) |
| `sleep_until` | $\pm$ 5--50 us | Zero | Low |
| Busy-wait (current) | Sub-microsecond | Zero | High (one core) |

The busy-wait approach is optimal for benchmarking because:

- It burns zero CPU when no tokens are in flight (the reader blocks on `emitCv`).
- During active generation, it delivers sub-microsecond timing accuracy, so TPOT
  measurements are not polluted by scheduler jitter.
- The 5--50 us wake-up jitter from `sleep_until` on a non-real-time Linux kernel
  would translate directly into TPOT measurement noise at high throughput.

---

## 6. request_stop() and shutdown()

```cpp
// include/tt_llm_engine/pipeline/pipeline_simulator.hpp:78,142-149
~PipelineSimulator() override { request_stop(); }

void request_stop() override {
    stop.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> lock(mu);
    emitCv.notify_all();
    injectCv.notify_all();
}

void shutdown() override {}
```

`request_stop()` sets the atomic `stop` flag with release semantics and then
acquires the mutex before notifying both condition variables.  The mutex
acquisition ensures that the notify is visible after the store -- any thread
that is currently in a wait predicate will re-check `stop` after waking.

Both `inject()` and `read_result()` check `stop` in their wait predicates and
spin loops, so they return promptly.  The destructor calls `request_stop()` as
a safety net (line 78) to guarantee cleanup even if the caller forgets.

`shutdown()` is a no-op because there is no device-side state to clean up in
simulation mode.

---

## 7. PipelineSimulatorConfig Mapping

The `PipelineSimulatorConfig` struct (defined in `pipeline_types.hpp` lines
82-96) provides the JSON-friendly configuration surface:

```cpp
// include/tt_llm_engine/pipeline/pipeline_types.hpp:82-96
struct PipelineSimulatorConfig {
    uint32_t num_stages = 64;
    uint32_t stage_duration_us = 44;
    uint32_t decode_token_id = EMPTY_TOKEN;
    float accept_rate = 1.0f;
    uint32_t seed = 42;
    uint32_t safe_vocab_base = 0;
    uint32_t safe_vocab_modulus = 0;
    bool batch_prefill = false;
};
```

The `DecodeScheduler` constructor maps these fields to the `PipelineSimulator`
constructor via `std::visit`:

```cpp
// src/scheduler/decode/decode_scheduler.cpp:99-103
} else if constexpr (std::is_same_v<T, PipelineSimulatorConfig>) {
    return std::make_unique<PipelineSimulator>(
        cfg.num_stages, cfg.stage_duration_us, cfg.decode_token_id,
        cfg.accept_rate, cfg.seed, cfg.safe_vocab_base, cfg.safe_vocab_modulus,
        cfg.batch_prefill);
```

| Config field | Constructor param | Notes |
|-------------|-------------------|-------|
| `num_stages` | `numStages` | Direct pass-through |
| `stage_duration_us` | `stageDuration` | Microseconds; constructor converts to `std::chrono::microseconds` |
| `decode_token_id` | `tokenId` | `EMPTY_TOKEN` activates the mock token model |
| `accept_rate` | `acceptRate` | Only used when `tokenId == EMPTY_TOKEN` and `token_type == BASE` |
| `seed` | `seed` | Seeds `std::mt19937` for accept/reject rolls |
| `safe_vocab_base` | `safeVocabBase` | Modular arithmetic base |
| `safe_vocab_modulus` | `safeVocabModulus` | Ring size; enforces >= 5 at construction |
| `batch_prefill` | `batchPrefill` | Immediate-complete mode for non-last prefill tokens |

### Pi0.5 Pipeline Runner Usage

In `pi05_pipeline_runner.cpp`, the simulator is configured for Pi0.5 benchmarks
with `batch_prefill = true`:

```cpp
// examples/pi05_pipeline_runner.cpp:881-887, 1208-1214
pl::PipelineSimulatorConfig sim_config{
    .num_stages = total_stages,
    .stage_duration_us = effective_stage_us,
    .accept_rate = accept_rate,
    .seed = seed,
    .batch_prefill = true,
};
```

The `total_stages` and `effective_stage_us` values are derived from the JSON
config's `pipeline.num_devices`, `pipeline.stages_per_device`, and
`pipeline.stage_duration_us` fields (covered in Chapter 2).

---

**Next:** [Batch Prefill Behavior](02_batch_prefill_behavior.md)
