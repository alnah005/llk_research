# IS Integration Patterns

[< Context Exhaustion](context_exhaustion.md) | [KV Cache Tiering >](kv_cache_tiering.md)

## Overview

The Inference Server (IS) is the external system that manages client connections, routing, batching policies, and the overall serving protocol. The decode scheduler exposes a narrow, non-blocking interface that the IS integrates against. This section describes the integration contract: the three queues, backpressure behavior, session-to-slot mapping, and the IS-side logic needed for correct operation.

The IS and the decode scheduler can run in-process (linked into the same binary) or as independent processes communicating over shared memory or sockets. The queue interface is designed to support both deployment modes (see Ch5 for `SocketPipeline` and transport details, Ch1 for the IS/PM ownership contract).

## Public API Surface

The IS-facing API consists of one push function and two non-blocking poll functions, plus read-only accessors for diagnostics and state inspection.

### Mutating Operations

```cpp
// Submit any lifecycle request (ALLOCATE, SUBMIT, CONTINUE, CANCEL).
// Returns false if request_queue is full (capacity: 2*max_users).
bool push_request(const ISRequest& request);

// Non-blocking poll for scheduler responses (ALLOCATE acks, cancel acks).
// Returns true and populates `response` if a response is available.
bool try_pop_response(SchedulerResponse& response);

// Non-blocking poll for output tokens.
// Returns true and populates `output` if a token is available.
bool try_pop_output(OutputMessage& output);
```

### Read-Only Accessors

```cpp
UserState get_user_state(uint32_t slot_id) const;
uint32_t  get_current_position(uint32_t slot_id) const;
```

These accessors read atomics from the UserTable and are safe to call from the IS thread at any time. They are informational -- the IS should not use them to make ordering decisions that depend on the PM's internal state. The canonical way to observe completion is to poll `try_pop_output` and check `is_complete`.

## The Three-Queue Interface

All communication between the IS and the decode scheduler flows through three bounded mutex-based queues (`BoundedQueue`, which uses `std::mutex` -- "Not lock-free -- acceptable for the API path"):

```
                    +------------------------------------------+
                    |           Decode Scheduler                |
                    |                                          |
  ISRequest ------->|  request_queue   (max_users * 2)         |
                    |                                          |
  SchedulerResponse<|  response_queue  (max_users * 2)         |
                    |                                          |
  OutputMessage <---|  output_queue    (max_users * 256)       |
                    |                                          |
                    +------------------------------------------+
```

### Queue Properties

| Queue | Direction | Capacity | Full Behavior |
|-------|-----------|----------|---------------|
| `request_queue` | IS -> PM | $2 \times \text{max\_users}$ | `push_request` returns `false` (caller must retry) |
| `response_queue` | PM -> IS | $2 \times \text{max\_users}$ | Scheduler spin-waits on `try_push` |
| `output_queue` | PM -> IS | $256 \times \text{max\_users}$ | Reader spin-waits on `try_push` |

### Queue Sizing Rationale

```cpp
request_queue(p.max_users * 2),
response_queue(p.max_users * 2),
output_queue(p.max_users * 256)
```

| Queue | Capacity | Rationale |
|-------|----------|-----------|
| `request_queue` | $2 \times \text{max\_users}$ | Each user may have one active request plus one queued (e.g., `SUBMIT` followed immediately by `CANCEL`). |
| `response_queue` | $2 \times \text{max\_users}$ | Mirrors request queue. Only `ALLOCATE` and `CANCEL` produce responses. |
| `output_queue` | $256 \times \text{max\_users}$ | Output is per-token, so the queue must absorb bursts. With speculative decode (see Ch4), multiple tokens per step are possible. |

## Backpressure

Backpressure propagates through the system in a specific chain:

```
IS slow to drain output_queue
    -> Reader spins on output_queue.try_push()
        -> Reader throughput drops
            -> Device output accumulates
                -> Writer cannot submit new batches (pipeline full)
                    -> Scheduler request_queue backs up
                        -> IS push_request spins
```

**Output queue backpressure (most common).** If the IS cannot drain `OutputMessage`s fast enough, the Reader's `emit_token` call spins on `try_push`. Since the Reader is synchronous with the device pipeline, this directly throttles throughput. With $256 \times \text{max\_users}$ capacity, the IS has a generous buffer, but sustained slow draining will cause pipeline stalls.

**Response queue backpressure (rare).** The scheduler pushes `SchedulerResponse` synchronously after processing `ALLOCATE` and `CANCEL`. Since responses are infrequent (only on allocation and cancellation, not per-token), this is rarely a bottleneck.

**Request queue backpressure (IS-side).** If the IS pushes requests faster than the scheduler processes them, `push_request` returns `false` and the IS must retry. The scheduler drains all available requests each iteration of its main loop (via a `while (request_queue.try_pop(req))` loop), interleaved with pipeline management.

## Recommended IS Polling Pattern

```
// IS event loop (pseudocode)
while (running) {
    // 1. Drain outputs first (highest volume)
    OutputMessage msg;
    while (scheduler.try_pop_output(msg)) {
        handle_output(msg);
    }

    // 2. Drain responses (allocation confirmations, cancel acks)
    SchedulerResponse resp;
    while (scheduler.try_pop_response(resp)) {
        handle_response(resp);
    }

    // 3. Submit new requests (from client connections)
    for (auto& req : pending_requests) {
        if (!scheduler.push_request(req)) {
            // Queue full -- retry next iteration
            break;
        }
    }

    // 4. Optional: yield or sleep briefly if all queues empty
}
```

Draining outputs first is important because the output queue is the highest-volume and the most likely to cause backpressure if neglected.

## Session-to-Slot Mapping

The IS must maintain a bidirectional mapping between its own session identifiers (client connections, request IDs, conversation threads) and the scheduler's slot IDs. The scheduler is slot-aware but session-unaware -- it operates purely on `uint32_t slot_id` values.

### IS-Side Session Table (Pseudocode)

```
struct ISSession {
    uint64_t   client_id;           // IS-side client identifier
    uint32_t   slot_id;             // Scheduler slot (INVALID until ALLOCATE succeeds)
    uint32_t   expected_generation; // For stale message filtering
    SessionState state;             // IS-level state machine
};

// Mapping in both directions
HashMap<uint64_t, ISSession> sessions;      // client_id -> session
HashMap<uint32_t, uint64_t>  slot_to_client; // slot_id -> client_id
```

### Lifecycle from the IS Perspective

```
Client connects
    -> IS creates ISSession { client_id, slot_id=INVALID, gen=0 }
    -> IS sends ALLOCATE request (with IS-assigned request_id)

IS receives SchedulerResponse for ALLOCATE
    -> If error_code == 0: store slot_id, update slot_to_client map
    -> If error_code == 1: queue client or reject (no capacity)

Client sends prompt
    -> IS sends SUBMIT { slot_id, tokens, gen_params }

IS receives OutputMessage stream
    -> Filter by generation counter (discard if msg.generation < expected)
    -> Forward token_id to client
    -> On is_complete: mark session as complete
    -> On ctx_exhausted: handle context management

Client sends follow-up
    -> IS sends CONTINUE { slot_id, new_tokens, new_gen_params }
    -> IS increments expected_generation

Client disconnects
    -> IS sends CANCEL { slot_id }
    -> IS increments expected_generation (to filter stale outputs)

IS receives SchedulerResponse (cancel ack)
    -> Remove slot_to_client mapping
    -> Destroy ISSession
```

## Generation Counter Protocol (IS-Side)

The generation counter is the IS's primary mechanism for handling message ordering and staleness:

1. **On ALLOCATE success:** Set `expected_generation = 0`.
2. **On receiving OutputMessage:** If `msg.generation < expected_generation`, discard the message. Otherwise, process it.
3. **On sending CANCEL:** Increment `expected_generation`. This ensures all subsequent outputs from the old generation are discarded, even if they arrive after the cancel is sent but before the cancel ack.
4. **On sending CONTINUE:** Increment `expected_generation`. Same rationale -- tokens from the old generation must be discarded.
5. **On receiving cancel ack:** The slot is now clean. No more messages will arrive for the old generation.

### Why Increment on Send, Not on Ack?

The IS increments `expected_generation` when it **sends** the CANCEL/CONTINUE, not when it receives the ack. This is because tokens from the old generation may be in the output queue between the send and the ack. If the IS waited for the ack before incrementing, it would process those stale tokens as if they were valid.

```
Time ----------------------------------------------------------->

IS sends CANCEL (gen 0->1)
    |
    +-- Stale OutputMessage { gen=0, token=42 }  -> DISCARDED (gen < expected)
    +-- Stale OutputMessage { gen=0, token=17 }  -> DISCARDED
    |
IS receives cancel ack
    +-- Slot is clean
```

## IS-Side Pseudocode

A complete IS integration sketch, showing all the key decision points:

```
function handle_output(msg: OutputMessage):
    session = slot_to_client[msg.slot_id]
    if session is None:
        return  // orphan message, slot was freed

    if msg.generation < session.expected_generation:
        return  // stale message from prior generation

    forward_token_to_client(session.client_id, msg.token_id)

    if msg.is_complete:
        if msg.ctx_exhausted:
            notify_client_context_full(session.client_id)
        else:
            notify_client_generation_complete(session.client_id)
        session.state = COMPLETE

function handle_response(resp: SchedulerResponse):
    pending = pending_requests[resp.request_id]

    if pending.type == ALLOCATE:
        if resp.error_code == 0:
            session = sessions[pending.client_id]
            session.slot_id = resp.slot_id
            slot_to_client[resp.slot_id] = pending.client_id
            session.state = ALLOCATED
        else:
            reject_or_queue_client(pending.client_id)

    else if pending.type == CANCEL:
        session = sessions[pending.client_id]
        slot_to_client.remove(session.slot_id)
        session.slot_id = INVALID
        session.state = TERMINATED

function handle_client_message(client_id, prompt_tokens, gen_params):
    session = sessions[client_id]

    if session.state == ALLOCATED:
        scheduler.push_request(ISRequest {
            type: SUBMIT,
            slot_id: session.slot_id,
            tokens: prompt_tokens,
            gen: gen_params
        })
        session.state = GENERATING

    else if session.state == COMPLETE:
        scheduler.push_request(ISRequest {
            type: CONTINUE,
            slot_id: session.slot_id,
            tokens: prompt_tokens,
            gen: gen_params
        })
        session.expected_generation += 1
        session.state = GENERATING

function handle_client_disconnect(client_id):
    session = sessions[client_id]
    if session.slot_id != INVALID:
        scheduler.push_request(ISRequest {
            type: CANCEL,
            slot_id: session.slot_id
        })
        session.expected_generation += 1
```

## Deployment Topologies

The three-queue interface supports multiple deployment modes:

### In-Process (Linked)

The IS and decode scheduler are compiled into the same binary. Queue operations are direct function calls on shared memory. Lowest latency.

```
+---------------------------------------------+
|                Single Process                |
|  +---------+         +------------------+    |
|  |   IS    | <-----> | DecodeScheduler  |    |
|  |         |  queues  |   + Pipeline     |    |
|  +---------+         +------------------+    |
+---------------------------------------------+
```

### Out-of-Process (Socket)

The IS and decode scheduler run as separate processes. The queue interface is wrapped by a transport layer that serializes structures over sockets. See Ch5 for `SocketPipeline` and wire format details.

```
+--------------+           +------------------+
|     IS       |  socket   | DecodeScheduler  |
|              | <-------> |   + Pipeline     |
| (process A)  |           |  (process B)     |
+--------------+           +------------------+
```

### Disaggregated (Prefill + Decode Separate)

In a fully disaggregated deployment, the IS coordinates between a prefill scheduler and a decode scheduler on separate hardware. The `CONTINUE` (disaggregated) path handles the handoff. See [Multi-Turn Continue](multi_turn_continue.md).

```
+----------+       +-----------------+       +------------------+
|    IS    | ----> | Prefill Sched.  | ----> | Decode Scheduler |
|          | <---- |  (prefill node)  |       |  (decode node)   |
|          | <---------------------------------|                  |
+----------+       +-----------------+       +------------------+
```

## Error Handling Patterns

| Scenario | IS Action |
|----------|-----------|
| `ALLOCATE` returns `error_code = 1` | Queue the client request; retry when a slot frees (cancel ack received) |
| Output queue drains empty for extended period | Normal -- user may be in prefill. No action needed. |
| `ctx_exhausted = true` on output | Do not send CONTINUE without context management. See [Context Exhaustion](context_exhaustion.md). |
| IS crashes and restarts | Must `CANCEL` all previously allocated slots (if slot IDs were persisted) or wait for scheduler timeout |
| Scheduler crashes | IS detects via heartbeat or socket close. All sessions are lost; re-allocate on restart. |

## IS Contract Summary

The IS must uphold these rules:

1. **ALLOCATE before SUBMIT.** Never send SUBMIT/CONTINUE/CANCEL for an unallocated slot.
2. **One active request per slot.** Do not send SUBMIT while a previous SUBMIT is still in progress.
3. **CONTINUE only after completion.** Only send CONTINUE when the previous turn's `is_complete = true` (and `ctx_exhausted = false`).
4. **CANCEL to free.** Always CANCEL slots when sessions end. Failure to do so leaks slots.
5. **Filter on generation.** Always compare `OutputMessage.generation` against expected to avoid delivering stale tokens.
6. **Drain output promptly.** The output queue is shared across all slots. Falling behind stalls the entire decode pipeline.
7. **Handle error_code = 1.** ALLOCATE can fail. The IS must gracefully handle slot exhaustion.

---

*[< Context Exhaustion](context_exhaustion.md) | [KV Cache Tiering >](kv_cache_tiering.md)*
