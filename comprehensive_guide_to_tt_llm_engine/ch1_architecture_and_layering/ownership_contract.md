# Ownership Contract

tt-llm-engine enforces a strict separation between the Inference Server (IS) and the Pipeline Manager (PM, also called the Decode Scheduler or DS). IS operates at millisecond timescales; PM operates at microsecond timescales. All communication flows through three bounded queues -- neither side calls the other's functions or shares mutexes. Any leakage across this boundary creates latency hazards that affect every concurrent user.

## The Timescale Argument

The motivation for strict ownership separation is quantified in `docs/scheduler/decode.md`:

| Aspect | Inference Server | Decode Scheduler |
|--------|-----------------|-----------------|
| **Timescale** | Milliseconds (HTTP, TLS, JSON, tokenization) | Microseconds (one inject per pipeline tick) |
| **Workload** | I/O-bound: network, string processing, session management | Compute-bound: tight scheduling loop, pipeline driving |
| **Stall cost** | Acceptable -- users tolerate ms-level API latency | Catastrophic -- every stalled tick = wasted pipeline slot for ALL users |
| **State** | Chat history, auth, WebSocket connections | User table, prompt table, decode staging, KV cache metadata |

If both sides run in the same thread, any HTTP handler latency spike starves the pipeline:

```
Combined thread (bad):
  handle HTTP request (2ms)
  +-- parse JSON body
  +-- tokenize prompt (0.5ms)
  +-- ...pipeline starves for 2ms -> ~45 wasted ticks at ~44us/tick...
  +-- finally inject next token

Separated (good):
  IS thread:                     PM writer thread:
    handle HTTP (2ms)              inject token (every tick)
    +-- parse JSON                 inject token
    +-- tokenize                   inject token
    +-- push to queue ------->     inject token
    +-- respond to client          inject token
```

## Terminology

Throughout the codebase and this guide, "PM" and "DS" are used interchangeably. "PM" (Pipeline Manager) is the original name from internal docs and variable naming conventions. "DS" (Decode Scheduler) is the component-specific class name (`DecodeScheduler`). This guide uses "PM" when discussing the architectural role and "Decode Scheduler" when referring to the specific C++ class.

## IS vs. PM Responsibility Matrix

The following table is derived from the responsibilities section of `docs/overview.md` and `docs/scheduler/decode.md`, cross-referenced against the actual implementation in `decode_scheduler.cpp`.

| Concern | IS Owns | PM Owns |
|---------|---------|---------|
| **Client endpoints** | HTTP/WebSocket/gRPC endpoint management, TLS termination, request routing | -- |
| **Tokenization** | Encode prompts to token IDs, decode output token IDs to text. All string processing happens on the IS side. | -- |
| **Chat history** | Multi-turn conversation context, chat template formatting | -- |
| **Session identity** | `session_id` (external, client-facing) | `user_id` / `slot_id` (internal, KV cache index) |
| **Session-to-slot mapping** | Maintains `session_id -> user_id` and `user_id -> session_id` hash maps | Returns allocated `user_id` via `SchedulerResponse` |
| **Authentication & rate limiting** | All auth, API keys, per-tenant rate limits, request validation | -- |
| **User slot allocation** | Sends `ALLOCATE` request | Executes allocation via `FreeIdPool.allocate()`, returns slot or error |
| **Prompt delivery** | Tokenizes and sends tokens via `ISRequest` (SUBMIT/CONTINUE) | Stores in `PromptTable`, drives prefill injection |
| **Prefill/decode scheduling** | -- | Chunked prefill interleaving, decode priority, pipeline injection |
| **Speculative decode** | Configures `spec_decode` per-user via `GenerationParams` | Drives pair injection, verifies acceptance, manages `SpecDecodeState` |
| **Pipeline injection** | -- | One token per tick via `PipelineInterface::inject()` |
| **In-flight tracking** | -- | `in_flight_count`, `pending_count`, `prefill_in_flight`, `post_complete_in_flight` |
| **Completion detection** | Receives `OutputMessage.is_complete` | Detects EOS, `max_new_tokens`, context exhaustion |
| **KV cache lifecycle** | Decides "why/when" (tier policy, idle-time demotion thresholds, SLA classes, eviction triggers) | Executes "can/how" (retain on COMPLETE, free on CANCEL, `reset_kv()` on cleanup, tracks residency and capacity) |
| **Context overflow** | Pre-validates `get_current_position() + prompt_len <= max_seq_len` before SUBMIT | Defensive guard in writer; marks COMPLETE with `ctx_exhausted` on decode-side overflow |
| **Wait/retry/timeout** | Decides user-facing behavior when PM returns `NO_SPACE` or admission failure | Returns admission failure promptly; does not queue waiting users |
| **Output streaming** | Drains `output_queue`, routes tokens to client WebSocket/SSE/HTTP | Pushes `OutputMessage` with `token_id`, `is_complete`, `ctx_exhausted`, `generation` |
| **Cancellation** | Sends `CANCEL` request | Executes deferred cancellation (mark, drain in-flight, finalize cleanup) |

## What IS Must Never Do

These are not suggestions -- violating any of them creates correctness or performance bugs:

1. **Never call PM functions directly.** All interaction flows through bounded queues (`request_queue`, `response_queue`, `output_queue`). Direct function calls would introduce coupling between IS thread timing and the PM hot path, breaking the timescale separation. No shared memory, no direct function calls, no mutexes shared between IS and PM.

2. **Never perform DMA or device I/O.** IS never touches H2D/D2H sockets, device memory, or the `PipelineInterface`. The wire format is entirely PM-internal.

3. **Never manage user_id allocation internally.** IS must not guess or fabricate user IDs. The only way to obtain a valid `user_id` is through the `ALLOCATE` response from PM. The `FreeIdPool` is PM-owned.

4. **Never assume a slot is free after CANCEL.** A CANCEL request initiates deferred cleanup. The slot is not actually free until PM has drained all in-flight tokens and pushed a cancel acknowledgment `SchedulerResponse` back. IS should wait for this response before considering the slot recyclable.

5. **Never maintain per-user state that duplicates PM state.** IS tracks session identity and chat history. PM tracks scheduling state (user table, prefill position, in-flight counts). Duplicating PM state in IS leads to inconsistency.

## What PM Must Never Do

1. **Never own HTTP sessions or client-facing connections.** PM has no concept of a WebSocket, an HTTP request ID, or a client IP address. It sees `user_id` values and queue messages. When it cannot satisfy an allocation request, it returns `error_code=1` immediately; the IS decides what to tell the client.

2. **Never maintain client-facing wait queues.** When capacity is exhausted (no free slots, KV tier full), PM returns failure promptly via `SchedulerResponse` with `error_code != 0`. IS decides whether to queue the client, retry, or return an error.

3. **Never own business-level fairness policy.** PM schedules FIFO by default (with a pluggable `SchedulerPolicy` interface for priority scheduling). Deciding which tenant, SLA class, or request priority gets preferential treatment is IS business logic.

4. **Never block on IS.** PM threads must never wait for IS to drain a queue, acknowledge a response, or complete an HTTP request. If the `output_queue` is full, PM spins briefly with `_mm_pause()` -- a brief backpressure signal, not a blocking wait. PM never calls any IS function. See [Key Invariant 1](./key_invariants.md).

5. **Never store or process natural language.** PM deals exclusively in token IDs, slot IDs, and positions. No strings, no prompts as text, no JSON, no UTF-8. This is essential for the hardware deployment path.

## The session_id to user_id Mapping Protocol

The mapping between external session identity and internal KV cache index is split cleanly:

```
Inference Server                     Pipeline Manager (Decode Scheduler)
      |                                        |
      |  1. IS receives/creates session_id     |
      |                                        |
      |-- ISRequest(ALLOCATE, request_id=1) -->|
      |                                        |
      |                          2. PM: FreeIdPool.allocate()
      |                             -> uid = 3 (or INVALID_SLOT)
      |                                        |
      |<-- SchedulerResponse(request_id=1,  ---|
      |        slot_id=3, error_code=0)        |
      |                                        |
      |  3. IS stores mapping:                 |
      |     session_to_user[sid] = 3           |
      |     user_to_session[3] = sid           |
      |                                        |
      |  4. All future traffic keyed by uid:   |
      |-- ISRequest(SUBMIT, slot_id=3, ...) -->|
      |                                        |
      |<-- OutputMessage(slot_id=3, tok=...) --|
      |                                        |
      |  5. IS routes output by reverse map:   |
      |     sid = user_to_session[3]           |
      |     session[sid].stream_token(tok)     |
```

Key properties of this protocol:

- **IS is the source of truth for session identity.** PM never sees `session_id` -- it only sees `slot_id` (equivalently `user_id` or `uid`; the three names are interchangeable in the codebase).

- **PM is the source of truth for slot allocation.** IS does not decide which slot a session gets -- it requests allocation and PM returns the result.

- **The mapping is 1:1.** One session maps to exactly one slot. A slot cannot serve two sessions simultaneously. When a session ends, IS sends CANCEL, waits for the cancel ack, and removes the mapping.

- **`slot_id` is an integer in `[0, max_users)`.** It directly indexes all per-user arrays in the `UserTable`, `PromptTable`, `DecodeStaging`, and `SpecDecodeState`. This is critical for the hardware deployment path: no hash tables, no dynamic lookups -- just array indexing.

- **`request_id` is transport-only.** PM echoes `request_id` in `SchedulerResponse` so IS can correlate async responses. PM does not track request timelines; its runtime state is keyed entirely by `slot_id`.

## The Three-Queue Boundary

The ownership contract is enforced at runtime through three bounded queues. No other communication channel exists between IS and PM:

| Queue | Direction | Type | Capacity | Producer | Consumer |
|-------|-----------|------|----------|----------|----------|
| `request_queue` | IS -> PM | `BoundedQueue<ISRequest>` | `2 * max_users` | IS HTTP handlers (MPSC) | PM API handler thread |
| `response_queue` | PM -> IS | `BoundedQueue<SchedulerResponse>` | `2 * max_users` | PM threads (MPSC) | IS allocation waiters |
| `output_queue` | PM -> IS | `BoundedQueue<OutputMessage>` | `256 * max_users` | PM Reader + Writer threads (MPSC) | IS output loop |

All three queues carry fixed-size messages. `ISRequest` is the only message with a variable-length field (`std::vector<uint32_t> tokens`), and even that is bounded by `max_seq_len` (default 131072). The queue implementations (`BoundedQueue<T>`) use a mutex-based ring buffer sized at construction -- no dynamic allocation after initialization.

**Backpressure semantics:**

| Queue | If full... |
|-------|-----------|
| `request_queue` | IS returns HTTP 503 to the client. PM never blocks waiting for requests. |
| `response_queue` | PM spins briefly. Allocation is rare; the queue drains quickly. |
| `output_queue` | PM spins with `_mm_pause()` until space is available. This is a deliberate design choice: dropping output tokens would require complex IS-side recovery, while brief backpressure on the hot path is bounded by the queue draining on the IS side. |

These queues enable independent deployment (same process, different process, or different machine), independent scaling (PM on accelerator co-processor, IS on host CPU), and independent testing (mock queues for unit tests).

## Message Types Crossing the Boundary

All three message types are defined in `include/tt_llm_engine/scheduler/decode/decode_types.hpp`:

| Struct | Direction | Key Fields | Purpose |
|--------|-----------|------------|---------|
| `ISRequest` | IS -> PM | `type` (ALLOCATE/SUBMIT/CONTINUE/CANCEL), `request_id`, `slot_id`, `tokens` (vec), `gen` (GenerationParams) | Carries all IS commands; `tokens` is bounded by `max_seq_len` |
| `SchedulerResponse` | PM -> IS | `request_id`, `slot_id`, `error_code` | Echoes `request_id` for correlation; `error_code=1` means no free slots |
| `OutputMessage` | PM -> IS | `slot_id`, `token_id`, `is_complete`, `ctx_exhausted`, `tokens_generated`, `generation` | One per generated token; `is_complete` signals EOS, max_new_tokens, or context exhaustion |

The `generation` field on `OutputMessage` enables the IS to detect and discard stale output tokens that arrive after a cancel-and-reallocate cycle. The PM advances the generation counter on each CANCEL via `decode_staging.advance_generation(uid)` and stamps every output with the current generation. The IS compares the output's generation with the expected generation for the session to filter late arrivals.

The full type reference, including `GenerationParams` fields and serialization details, is covered in [Chapter 5](../ch5_pipeline_and_wire_format/index.md) and [Chapter 6](../ch6_lifecycle_is_integration_kv_tiering/index.md).

## Ownership Split in Practice: Cancellation Example

The cancellation flow illustrates the ownership split concretely:

1. **IS decides to cancel** (client disconnects, timeout fires, explicit request). IS sends `ISRequest(CANCEL, slot_id=3, request_id=42)`.

2. **PM executes deferred cancellation.** The API handler thread marks `cancel_pending`, stashes the `request_id`, advances the generation counter (invalidating stale staging entries), and removes the user from the `PrefillQueue`. But it does not immediately free the slot -- there may be in-flight tokens in the pipeline.

3. **PM drains in-flight tokens.** As results arrive for slot 3, the Writer and Reader threads decrement `in_flight_count` and `pending_count`. When both reach zero, `maybe_finalize_cleanup` runs: CAS state to INACTIVE, call `pipeline_->reset_kv(3)`, reset spec state, free the ID back to `FreeIdPool`, and push a cancel ack response. (Note: `user_table.reset()` and `cancel_pending.clear()` are deferred to the next ALLOCATE for this slot.)

4. **IS receives the cancel ack.** `SchedulerResponse(request_id=42, slot_id=3, error_code=0)` arrives on the `response_queue`. IS removes the session-to-user mapping. The slot is now truly free.

IS never knows about `in_flight_count`, `pending_count`, or `maybe_finalize_cleanup`. PM never knows about the client's WebSocket or the HTTP request that triggered the cancel. Each side handles its own concern with no cross-layer leakage.

---

**Next:** [`component_map.md`](./component_map.md)
