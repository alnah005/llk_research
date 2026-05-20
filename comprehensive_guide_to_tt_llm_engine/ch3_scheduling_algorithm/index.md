# Chapter 3: Scheduling Algorithm

This chapter specifies the complete scheduling flow for regular (non-speculative) decode in the tt-llm-engine decode scheduler. Every algorithmic component is presented invariant-first: the invariant is stated, then the code mechanism that enforces it is shown. Speculative decode extends this flow with paired BASE/SPEC injection and verification logic, covered in [Chapter 4](../ch4_speculative_decode/index.md).

## The Fundamental Scheduling Constraint

**Invariant: exactly one token is injected into the pipeline per Writer tick, and decode always has priority over prefill.**

The pipeline is a fixed-latency FIFO of $D$ stages (e.g., $D = 64$ for a 16-device, 4-stage-per-device configuration). Every pipeline slot that goes unused is throughput permanently lost -- it cannot be reclaimed. The Writer thread's job is to keep the pipeline fully utilized by injecting one token per tick, drawn from whichever source produces the highest-priority work.

The priority ordering is:

1. **Decode tokens** from the `DecodeStaging` FIFO (populated by the Reader thread from pipeline results).
2. **Prefill tokens** from the `PrefillQueue` (populated by the API handler on SUBMIT/CONTINUE).

This priority is absolute. A decode token waiting in staging always preempts a prefill token. The rationale is direct: decode users are actively generating output visible to end-users. Every tick of latency on the decode path adds directly to time-per-output-token (TPOT). Prefill, by contrast, is background KV-cache population that tolerates delay -- the user sees no output during prefill anyway. As a concrete example: with $D = 64$, $N = 32$ active decode users, and a 10,000-token prefill prompt, removing decode priority would stall all 32 users by $10{,}000 \times \tau \approx 440\text{ms}$ (at $\tau = 44\mu s$). Absolute priority eliminates this stall entirely.

## Design Goals

The scheduling algorithm is constrained by three goals, listed in priority order:

1. **Minimize TPOT.** Decode users must receive their loopback re-injection as soon as the pipeline delivers a result. No tick should be wasted on lower-priority work when a decode token is waiting.

2. **Prevent starvation.** A single user with a 128K-token prompt must not monopolize the Writer for thousands of ticks, starving all decode users of re-injection. Chunked prefill with round-robin rotation bounds the maximum delay any decode user experiences due to prefill activity.

3. **Hardware-deployable.** Every decision in the scheduling loop uses fixed-size structures, bounded iteration, and no dynamic allocation. The algorithm is designed to run identically on a hardware co-processor (see [Chapter 1 -- Key Invariants](../ch1_architecture_and_layering/key_invariants.md#hardware-deployment-path)).

## One Token Per Pipeline Tick

The pipeline processes one token per stage per tick. The Writer injects at the pipeline's input; the Reader reads from the pipeline's output. With $D$ stages and a tick period of $\tau$ microseconds, a token injected at tick $t$ produces a result at tick $t + D$. In steady state with $N$ active decode users ($N \leq D$), each user receives one output token every $D$ ticks, giving a per-user TPOT of:

$$\text{TPOT} = D \cdot \tau$$

The scheduling algorithm's job is to fill every tick with useful work, prioritizing decode to keep this formula tight.

## Chapter Roadmap

| File | Content |
|------|---------|
| [`writer_priority_and_decode.md`](./writer_priority_and_decode.md) | Writer loop two-tier priority, decode token injection, `device_sampling_params`, no-work path |
| [`chunked_prefill.md`](./chunked_prefill.md) | Chunk size, round-robin rotation, starvation prevention, position management, last-token transition, context exhaustion |
| [`decode_loopback.md`](./decode_loopback.md) | Core decode cycle, pipeline latency, steady-state throughput, `current_position` tracking, stale entry filtering |
| [`completion_and_eos_writeback.md`](./completion_and_eos_writeback.md) | Three completion triggers, `emit_token` internals, EOS write-back, state transition, generation counter |
| [`regular_decode_flow.md`](./regular_decode_flow.md) | End-to-end walkthrough, multi-user interleaving, multi-turn CONTINUE, timing analysis |

## Relationship to Prior Chapters

This chapter assumes familiarity with:

- The three-thread architecture and thread interaction matrix from [Chapter 2 -- Threading Model](../ch2_threading_and_data_structures/threading_model.md).
- The `DecodeStaging` FIFO, `pending_count` reference counting, and `ReaderClaim` RAII guard from [Chapter 2 -- Decode Staging](../ch2_threading_and_data_structures/decode_staging.md).
- The `UserTable` field catalog, `UserState` FSM, and phase-based ownership from [Chapter 2 -- UserTable](../ch2_threading_and_data_structures/user_table.md).
- The lock-free busy-count protocol from [Chapter 2 -- Lock-Free Design](../ch2_threading_and_data_structures/lock_free_design.md).
- The six system invariants from [Chapter 1 -- Key Invariants](../ch1_architecture_and_layering/key_invariants.md).

Data structure layouts and memory ordering details are not repeated here. This chapter focuses exclusively on the algorithmic logic that orchestrates those structures.

All source references in this chapter point to `decode_scheduler.cpp` unless otherwise noted.

---

**Next:** [`writer_priority_and_decode.md`](./writer_priority_and_decode.md)
