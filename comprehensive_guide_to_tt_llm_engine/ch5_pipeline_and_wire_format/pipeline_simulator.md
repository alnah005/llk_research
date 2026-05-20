# PipelineSimulator: Timing-Accurate Systolic Pipeline Model

*Source: `pipeline_simulator.hpp`*

## Purpose

`PipelineSimulator` bridges the gap between `MockPipeline` (zero-latency, logic-only) and `SocketPipeline` (real hardware). It models a systolic pipeline with configurable depth and stage duration, producing timing-accurate token throughput without requiring any hardware. This makes it the backend of choice for performance benchmarks, throughput regression tests, and integration tests that need realistic timing behavior.

## Construction

```cpp
PipelineSimulator(uint32_t numStages = 64, uint32_t stageDuration = 44,
                  uint32_t tokenId = EMPTY_TOKEN, float acceptRate = 1.0f,
                  uint32_t seed = 42, uint32_t safeVocabBase = 0,
                  uint32_t safeVocabModulus = 0);
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `numStages` | 64 | Pipeline depth (number of systolic stages) |
| `stageDuration` | 44 | Duration of one stage in microseconds |
| `tokenId` | `EMPTY_TOKEN` | If set, override all actual tokens with this fixed value |
| `acceptRate` | 1.0 | BASE token acceptance probability (same semantics as MockPipeline) |
| `seed` | 42 | RNG seed for deterministic accept/reject |
| `safeVocabBase` | 0 | Base offset for modular vocabulary wrapping |
| `safeVocabModulus` | 0 | Modulus for safe-vocab wrapping (0 = disabled, must be $\geq 5$ if enabled) |

With the default configuration ($N = 64$ stages, $\Delta t = 44\,\mu s$):

$$\text{Latency} = 64 \times 44\,\mu s = 2816\,\mu s \approx 2.8\,\text{ms}$$

$$\text{Max throughput} = \frac{1}{44\,\mu s} \approx 22{,}727\,\text{tokens/s}$$

## The Three Invariants

### Invariant 1: Fixed Transit Latency

Every token experiences exactly $N \cdot \Delta t$ of pipeline latency:

$$t_{\text{exit}} = t_{\text{enter}} + N \times \Delta t$$

There is no jitter, no variable latency, no priority reordering. The pipeline is a strict FIFO with deterministic latency.

### Invariant 2: Throughput Cap

Consecutive injections must be separated by at least one stage period:

$$t_{\text{enter}}[n] = \max\!\bigl(t_{\text{now}},\; t_{\text{enter}}[n-1] + \Delta t\bigr)$$

If the Writer calls `inject()` faster than once per $\Delta t$, the entry time is pushed forward. This caps sustained throughput at $\frac{1}{\Delta t}$.

### Invariant 3: Backpressure

The pipeline can hold at most $N$ tokens simultaneously. When the inflight deque reaches capacity, `inject()` blocks:

```cpp
injectCv.wait(lock, [this] {
    return inflight.size() < numStages || stop.load(std::memory_order_acquire);
});
```

The Writer thread stalls until `read_result()` pops the front token. This models the physical constraint that a systolic pipeline of $N$ stages can hold at most $N$ tokens.

## No Tick Thread

A common approach to pipeline simulation is a dedicated "tick thread" that advances state at each stage period. PipelineSimulator deliberately avoids this:

```
inject():
    enter_time = max(steady_clock::now(), last_enter + stage_period)
    exit_time  = enter_time + total_latency
    push {result, exit_time} into inflight deque
    last_enter = enter_time

read_result():
    wait on emitCv until inflight is non-empty (or stop)
    head = inflight.front()
    unlock, busy-wait until steady_clock::now() >= head.exit_time
    re-lock, pop and return head.result, notify injectCv
```

The simulator busy-waits (spins on `clock::now()`) rather than using `sleep_until()`, which would be subject to OS scheduling jitter (typically 50--100 microseconds on Linux). For a pipeline with $\Delta t = 44\,\mu s$, even a $50\,\mu s$ overshoot would distort the timing model. The busy-wait is acceptable because the Reader thread has nothing else to do while waiting.

### Timing Example

With $N = 4$ stages and $\Delta t = 100\,\mu s$:

| Token | $t_{\text{inject}}$ | $t_{\text{enter}}$ | $t_{\text{exit}}$ |
|-------|---------------------|---------------------|-------------------|
| A | 0 | 0 | 400 |
| B | 50 | 100 | 500 |
| C | 80 | 200 | 600 |
| D | 500 | 500 | 900 |

Tokens B and C are injected before a full stage period has elapsed, so their enter times are pushed forward. Token D is injected well after the minimum interval, so it enters immediately.

## Token Model

The simulator uses the [same token model as MockPipeline](./mock_pipeline.md#token-model): `actual_token = token_id + 1` (accepted) or `token_id + 3` (rejected), with acceptance applying only to BASE tokens. Any test that passes against MockPipeline produces the same token sequences against PipelineSimulator -- only the timing differs.

The simulator adds two additional token generation strategies beyond the simple offset:

**Fixed Token ID**: When `tokenId != EMPTY_TOKEN`, every decode produces that fixed value as `actual_token` regardless of input. Useful for tests that verify scheduling logic without caring about token values.

**Modular Safe-Vocab Wrapping**: When `safeVocabModulus > 0`, tokens are constrained to a range:

$$\text{actual\_token} = \text{safeVocabBase} + \bigl((\text{input\_offset} + \delta) \bmod M\bigr)$$

where $\delta = 1$ for accepted, $\delta = 3$ for rejected, and $M = \text{safeVocabModulus}$. This keeps all generated tokens within $[\text{safeVocabBase},\, \text{safeVocabBase} + M)$, important for integration tests that feed tokens into a real tokenizer.

## Safe-Vocab Constraint

```cpp
if (safeVocabModulus != 0 && safeVocabModulus < 5)
    throw std::invalid_argument("safe_vocab_modulus must be 0 or >= 5");
```

The minimum of 5 ensures that the `+1` and `+3` offsets in the token model, plus the `+1` for the predicted token, never wrap around to collide. A modulus of 1--4 would make accepted and rejected tokens indistinguishable.

## Backpressure in Practice

Consider a 4-stage pipeline with $100\,\mu s$ stage duration:

```
Time(us)   inject()                    read_result()
  0        Token A enters (stage 0)
 100       Token B enters (stage 0)
 200       Token C enters (stage 0)
 300       Token D enters (stage 0)    -- pipeline now full (4 in-flight)
 300+      Token E: inject() BLOCKS
 400       Token A exits               Token E enters (unblocked)
 500       Token B exits
```

The scheduler sees the first 4 injections return immediately, then the 5th blocks for ~$100\,\mu s$. This matches the real hardware's backpressure behavior.

## Lifecycle Methods

| Method | Implementation |
|--------|---------------|
| `reset_kv(slot_id)` | No-op. |
| `request_stop()` | Sets `stop` atomic flag, notifies both `injectCv` and `emitCv`. Breaks the busy-wait loop. |
| `shutdown()` | No-op. No device to handshake with. |

## When to Use Which

- **"Does the scheduler inject the right tokens in the right order?"** -- Use MockPipeline. Check the inject log.
- **"Does the scheduler maintain throughput under backpressure?"** -- Use PipelineSimulator. Measure wall-clock time.
- **"Does the scheduler handle speculative rejection correctly?"** -- Use MockPipeline with `accept_rate < 1.0`.
- **"Does the scheduler stall correctly when the pipeline is full?"** -- Use PipelineSimulator. Verify that inject blocks.

---

| | Navigation | |
|:---|:---:|---:|
| [MockPipeline](./mock_pipeline.md) | [Table of Contents](../index.md) | [SocketPipeline](./socket_pipeline.md) |
