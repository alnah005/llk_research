# Chapter 6: Lifecycle, IS Integration, and KV Tiering

## Overview

This chapter is written from the **Inference Server's perspective**. The IS is the primary consumer of every lifecycle operation exposed by the Pipeline Manager: it allocates slots, submits prompts, continues conversations, cancels requests, and drains output tokens. Understanding the three-queue contract and the state machine that governs each slot is essential before any of those operations make sense.

We begin with the state machine as the central abstraction, then show how every IS-facing operation maps to transitions in that machine. Later sections cover multi-turn continuation, deferred cancellation, context exhaustion, the queue integration patterns, and the planned KV-cache tiering design.

## Chapter Map

| Section | File | Focus |
|---------|------|-------|
| 6.1 | [Session State Machine](session_state_machine.md) | `UserState` transitions, request types, the three interface structs, generation counter |
| 6.2 | [Multi-Turn Continue](multi_turn_continue.md) | Local vs. disaggregated continue, KV reuse, position accounting |
| 6.3 | [Deferred Cancellation](deferred_cancellation.md) | Four-phase cancel protocol, `maybe_finalize_cleanup`, correctness properties |
| 6.4 | [Context Exhaustion](context_exhaustion.md) | Decode-side and prefill-side detection, `ctx_exhausted` signaling |
| 6.5 | [IS Integration Patterns](is_integration_patterns.md) | Three-queue interface, backpressure, session-to-slot mapping, deployment modes |
| 6.6 | [KV Cache Tiering](kv_cache_tiering.md) | **PLANNED / NOT YET IMPLEMENTED.** T0/T1/T2 design, `TierCommand` API |

## The Three-Queue Contract at a Glance

The IS and PM communicate through exactly three bounded mutex-based queues ("Not lock-free -- acceptable for the API path" -- `bounded_queue.hpp`). No shared mutable state is accessed outside of these queues and the atomic fields documented in [Chapter 2](../ch2_threading_and_data_structures/index.md).

```
  Inference Server                        Pipeline Manager
  ================                        ================

  push_request(ISRequest) ------>  [ request_queue ]  ------> scheduler loop
                                    capacity: 2*max_users

  try_pop_response(resp)  <------  [ response_queue ] <------ ALLOCATE acks, cancel acks
                                    capacity: 2*max_users

  try_pop_output(msg)     <------  [ output_queue ]   <------ per-token output
                                    capacity: 256*max_users
```

The asymmetry in capacity is deliberate: a single SUBMIT can produce hundreds of output tokens, so the output queue is sized at $256 \times \text{max\_users}$ to avoid backpressure stalls during bursty decode phases.

## The Central State Machine

The following diagram shows the complete user session lifecycle. Every section in this chapter elaborates on one or more edges in this graph.

```
                        +----------------------------------------------------+
                        |              CANCEL (any state)                     |
                        |   (deferred -- see deferred_cancellation.md)       |
                        +----------------------------------------------------+

  +--------------+  ALLOCATE   +--------------------+  SUBMIT   +----------+
  |  INACTIVE    | ----------> |  INACTIVE(allocated) | -------> |  PREFILL |
  | (no slot)    |             |  (slot assigned)     |          |          |
  +--------------+             +--------------------+          +-----+----+
        ^                                                           |
        |                                                     Writer completes
        |                                                     all chunks
        |                                                           |
        |                                                           v
        |                    +----------+   token-by-token   +----------+
        |    free slot       | COMPLETE  | <---------------- |  DECODE  |
        |<----------------- |          |   (EOS / max_new   |          |
        |   (no CONTINUE)    +----+-----+    / ctx_exhaust)  +----------+
        |                         |
        |                         | CONTINUE
        |                         v
        |                    +----------+
        |                    |  PREFILL  |  (local continue: KV reuse)
        |                    |  or       |
        |                    |  DECODE   |  (disaggregated continue)
        |                    +----------+
        |                         |
        |                         +---- ... (back into decode loop)
        |
        +-------- CANCEL (deferred, four-phase cleanup)
```

## Relationship to Prior Chapters

This chapter builds directly on the internal machinery described in Chapters 2--5 but views it from the outside -- from the perspective of the IS or any external client:

- **Chapter 1** defines the IS/PM ownership contract and the component map. This chapter shows the concrete protocol that implements that contract.
- **Chapter 2** describes the `UserTable`, `FreeIdPool`, `CancelBitmap`, and `DecodeStaging` internals. This chapter shows how those structures are manipulated through the request/response protocol.
- **Chapter 3** covers the Writer's priority system, chunked prefill, and decode loopback. This chapter shows how `SUBMIT` and `CONTINUE` feed work into those pipelines.
- **Chapter 4** details speculative decode. This chapter notes where `spec_state` is reset and when `spec_decode_enabled` is set, but defers to Ch4 for the protocol itself.
- **Chapter 5** covers `PipelineInterface` and its implementations. This chapter references `pipeline_->reset_kv()` and `pipeline_->prefill()` / `pipeline_->decode()` but defers to Ch5 for wire format and transport.

## Reading Order

For a first read, follow the sections in order: the state machine formalizes the rules, multi-turn and cancellation cover the critical edge cases, and IS integration shows how to consume the interface correctly. If debugging a specific issue, jump directly to the relevant section -- each is self-contained with back-references where needed.

---

*Next: [Session State Machine](session_state_machine.md)*
