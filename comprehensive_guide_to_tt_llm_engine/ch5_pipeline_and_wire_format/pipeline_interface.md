# PipelineInterface: The Abstract Contract

*Source: `pipeline_interface.hpp`, `pipeline_types.hpp`*

## The Five Virtual Methods

```cpp
class PipelineInterface {
public:
    virtual ~PipelineInterface() = default;
    virtual void inject(const InjectDescriptor& desc) = 0;
    virtual ResultDescriptor read_result() = 0;
    virtual void reset_kv(uint32_t slot_id) = 0;
    virtual void request_stop() = 0;
    virtual void shutdown() = 0;
};
```

There are no optional methods, no CRTP, no template parameters. Every implementation must provide all five.

### Hot-Path Methods

| Method | Caller Thread | Blocking? | Semantics |
|--------|--------------|-----------|-----------|
| `inject(desc)` | Writer (Ch2) | Impl-defined | Sends one token descriptor to the pipeline. May block if backpressure applies. |
| `read_result()` | Reader (Ch2) | Yes | Blocks until one result descriptor is available. Returns a sentinel (`slot_id == INVALID_SLOT`) when the pipeline has been stopped. |

The Writer thread (Ch2) calls `inject()` for every token -- both regular decode tokens and speculative BASE+SPEC pairs (Ch4). The Reader thread calls `read_result()` in a tight loop, consuming results and dispatching them to the result classification logic. The scheduler guarantees single-writer and single-reader access to these methods respectively.

### Lifecycle Methods

| Method | Purpose | When Called |
|--------|---------|------------|
| `reset_kv(slot_id)` | Hint that a slot's KV-cache can be invalidated. | On request completion. Currently a no-op in all three backends. |
| `request_stop()` | Signal the pipeline to stop producing results. Non-blocking. Sets internal flags only -- must not perform socket I/O. | Shutdown phase 1, before thread join. |
| `shutdown()` | Perform any finalization (e.g., sentinel exchange with device). May block. | Shutdown phase 2, after all threads joined. |

## Two-Phase Shutdown Protocol

Shutdown is split into two phases to avoid deadlocks:

```
Main Thread                Read Thread
    |                          |
    +-- request_stop() ------->|  (sets atomic flag)
    |                          +-- read_result() returns sentinel
    |                          +-- exits loop
    +-- join(read_thread) ---->|
    +-- shutdown() ----------->|  (sends sentinel, drains, barrier)
```

**Phase 1 -- `request_stop()`**: The scheduler calls this to signal "no more tokens will be injected." Implementations set an atomic flag. This unblocks any `read_result()` call that is waiting, causing it to return `ResultDescriptor{.slot_id = INVALID_SLOT}`.

**Phase 2 -- `shutdown()`**: Called after the Reader has observed the sentinel and exited its loop. For `SocketPipeline`, this is where the sentinel exchange with the device happens (see [SocketPipeline](./socket_pipeline.md)). For MockPipeline and PipelineSimulator, `shutdown()` is a no-op.

Why two phases? If `shutdown()` were called while `read_result()` was still blocking on the D2H socket, the sentinel exchange would deadlock -- the Reader would be waiting on D2H while shutdown tries to read D2H for the sentinel response.

## InjectDescriptor

```cpp
struct InjectDescriptor {
    uint32_t slot_id = INVALID_SLOT;
    uint32_t token_id = EMPTY_TOKEN;
    uint32_t prefill_token_id = EMPTY_TOKEN;
    uint32_t position = 0;
    TokenType token_type = TokenType::BASE;
    bool spec_flag = false;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int32_t top_k = -1;
};
```

| Field | Type | Meaning |
|-------|------|---------|
| `slot_id` | `uint32_t` | KV-cache slot. `INVALID_SLOT` is the shutdown sentinel. |
| `token_id` | `uint32_t` | The token being decoded. |
| `prefill_token_id` | `uint32_t` | If not `EMPTY_TOKEN`, this is a **non-last prefill** injection (Ch3). Only `slot_id` is echoed back. |
| `position` | `uint32_t` | Sequence position within the slot's context window. |
| `token_type` | `TokenType` | `BASE` or `SPEC` -- distinguishes speculative pairs (Ch4). |
| `spec_flag` | `bool` | Additional speculative flag for the default wire format. |
| `temperature` | `float` | Sampling temperature. Default $1.0$. |
| `top_p` | `float` | Nucleus sampling threshold. Default $1.0$ (disabled). |
| `top_k` | `int32_t` | Top-k sampling. Default $-1$ (disabled). |

## ResultDescriptor

```cpp
struct ResultDescriptor {
    uint32_t slot_id = INVALID_SLOT;
    uint32_t actual_token = EMPTY_TOKEN;
    TokenType actual_token_type = TokenType::BASE;
    uint32_t actual_token_pos = 0;
    uint32_t predicted_token = EMPTY_TOKEN;
    TokenType predicted_token_type = TokenType::BASE;
    uint32_t predicted_token_pos = 0;
    std::array<uint32_t, 32> p_indices{};
    std::array<uint16_t, 32> p_scores{};
};
```

| Field | Type | Meaning |
|-------|------|---------|
| `slot_id` | `uint32_t` | Which KV-cache slot produced this result. `INVALID_SLOT` = shutdown sentinel. |
| `actual_token` | `uint32_t` | The token the model actually produced (after sampling). |
| `actual_token_type` | `TokenType` | Whether this actual token was generated from a BASE or SPEC injection. |
| `actual_token_pos` | `uint32_t` | Sequence position of the actual token. |
| `predicted_token` | `uint32_t` | Speculative prediction for the *next* token after `actual_token`. |
| `predicted_token_type` | `TokenType` | Set to `SPEC` by mock/simulator backends and DeepSeek deserialization. Remains at default (`BASE`) in default wire format results, which do not transmit this field. |
| `predicted_token_pos` | `uint32_t` | Sequence position of the predicted token. |
| `p_indices` | `array<uint32_t, 32>` | Top-32 token indices (DeepSeek only). Uses `uint32_t` -- not `uint16_t` -- because vocabulary sizes exceed 65535. |
| `p_scores` | `array<uint16_t, 32>` | Corresponding bf16-packed scores (DeepSeek only). |

The `p_indices` and `p_scores` arrays are only populated in DeepSeek mode. In default mode, they remain zero-initialized. These arrays support the extended speculative verification protocol (Ch4).

## Sentinel Values

```cpp
static constexpr uint32_t INVALID_SLOT = std::numeric_limits<uint32_t>::max();  // 0xFFFFFFFF
static constexpr uint32_t EMPTY_TOKEN  = std::numeric_limits<uint32_t>::max();  // 0xFFFFFFFF
```

Both are `0xFFFFFFFF`. Their meaning depends on context:

| Constant | Where | Meaning |
|----------|-------|---------|
| `INVALID_SLOT` | `InjectDescriptor.slot_id` | Shutdown sentinel (inject direction) |
| `INVALID_SLOT` | `ResultDescriptor.slot_id` | Shutdown sentinel (result direction) |
| `EMPTY_TOKEN` | `InjectDescriptor.prefill_token_id` | This is a decode step, not a prefill |
| `EMPTY_TOKEN` | `ResultDescriptor.actual_token` | Non-last prefill stub or sentinel |

Using `uint32_t::max` as a sentinel guarantees no collision with any valid slot index or token ID.

## TokenType Enum

```cpp
enum class TokenType : uint8_t { BASE = 0, SPEC = 1 };
```

Distinguishes base decode tokens from speculative tokens. Serialized to a `uint32_t` word in the DeepSeek wire format (zero-extended) and reconstructed via `static_cast` on deserialization. The default wire format does not transmit `TokenType` directly -- it uses the `spec_flag` boolean instead.

## Configuration Structs

Each backend has a corresponding config struct (`MockConfig`, `SocketConfig`, `PipelineSimulatorConfig`), unified into a single variant type. See the full struct definitions and `std::visit` dispatch code in the [chapter overview](./index.md#selecting-a-backend-the-pipelineconfig-variant).

## Contract Summary

The `PipelineInterface` contract can be summarized as four invariants:

1. **Thread safety**: `inject()` and `read_result()` must be safe to call concurrently from different threads.
2. **FIFO ordering**: Results appear in `read_result()` in the same order as their corresponding `inject()` calls.
3. **Sentinel termination**: After `request_stop()`, `read_result()` must eventually return a sentinel (queued results may drain first).
4. **Shutdown idempotency**: Calling `shutdown()` on an already-stopped pipeline must be safe.

---

| | Navigation | |
|:---|:---:|---:|
| [Chapter 5 Overview](./index.md) | [Table of Contents](../index.md) | [MockPipeline](./mock_pipeline.md) |
