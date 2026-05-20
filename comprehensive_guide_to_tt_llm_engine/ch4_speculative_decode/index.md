# Chapter 4: Speculative Decode

Speculative decode is a KV-cache management protocol. Every design choice in the protocol -- pair injection, result classification, deferred completion, the state machine -- exists to answer one question: which KV positions get written, in what order, so that the model always sees correct context without ever rolling back a cache entry.

The engine implements MTP (Multi-Token Prediction) speculative decode. Each pipeline result carries not only a sampled token (`actual_token`) but also a predicted next token (`predicted_token`). The scheduler uses this prediction speculatively: it injects the verified token at position $P$ and the prediction at position $P+1$ as a back-to-back pair. If the next BASE result confirms the prediction, the user gains a bonus output token (the SPEC result at $P+1$) for free. If the prediction is wrong, the BASE result's corrected token overwrites the stale KV at $P+1$, and a new prediction extends to $P+2$. No explicit rollback ever occurs.

This "overwrite-not-rollback" property is the architectural foundation of the entire protocol. It holds because of three guarantees:

1. **Single speculative position**: At most one KV position is speculative per user at any time.
2. **Pipeline FIFO ordering**: The Writer injects BASE before SPEC; the pipeline delivers results in the same order.
3. **In-place overwrite**: On mismatch, the correction BASE writes to the exact position that the stale SPEC occupied.

## Pipeline Bandwidth Tradeoff

Speculative decode doubles the per-user pipeline bandwidth consumption:

| Mode | Pipeline slots per user per round | Output tokens per round (best case) | Output tokens per round (worst case) |
|------|----------------------------------|--------------------------------------|---------------------------------------|
| Regular | 1 | 1 | 1 |
| Speculative | 2 | 2 (ACCEPT + CONTINUE) | 1 (REJECT, wasted SPEC slot) |

With a pipeline of depth $D$ slots (e.g., $D = 64$ for a 16-device, 4-stage configuration), the total capacity is $D$ injections per round. If $M$ users run in speculative mode and $R$ users run in regular mode, the slot equation is:

$$2M + R \leq D$$

### Slot Arithmetic Example

With $D = 64$ pipeline slots:

| Configuration | Regular Users | Spec Users | Slots Used | Prefill Bandwidth |
|---|---|---|---|---|
| All regular | 64 | 0 | 64 | 0 ticks/period |
| Mixed | 24 | 20 | $24 + 40 = 64$ | 0 ticks/period |
| Half spec | 0 | 32 | 64 | 0 ticks/period |
| Light load | 10 | 10 | 30 | 34 ticks/period for prefill |

At 100% accept rate, the mixed configuration produces $20 \times 2 + 24 = 64$ tokens per round -- identical to 64 regular users. But the 20 speculative users individually see tokens arrive at $2\times$ the per-user rate. The tradeoff is individual latency versus aggregate throughput, analyzed in detail in [`throughput_analysis.md`](./throughput_analysis.md).

## Enabling Speculative Decode

Speculative decode is enabled per-user via `GenerationParams::spec_decode = true`, which sets `UserTable::spec_decode_enabled[uid]` on SUBMIT or CONTINUE. The flag is checked by the Writer when popping from `DecodeStaging` (line 239 of `decode_scheduler.cpp`):

```cpp
bool do_spec = user_table.spec_decode_enabled[uid] && !entry.skip_spec;
```

The `skip_spec` override is used for EOS write-back entries (see [`pair_injection.md`](./pair_injection.md)), which must inject only the BASE token even when speculation is enabled.

## How Two Injections Map to One Staging Entry

A speculative decode user's loopback produces a single `DecodeStagingEntry` carrying both tokens (lines 510-514 of `decode_scheduler.cpp`):

```cpp
auto stage_pair = [&]() {
    decode_staging.stage(uid,
        result.actual_token, result.actual_token_pos,
        result.predicted_token, result.predicted_token_pos);
};
```

The Writer unpacks this into two `pipeline_->inject()` calls (lines 247-268): one BASE at `position` with `TokenType::BASE`, and one SPEC at `spec_position` with `TokenType::SPEC`. Before injecting, the Writer bumps `in_flight_count` by 2 atomically (line 244), ensuring the Reader correctly accounts for both results when they arrive $D$ ticks later. The single staging entry / dual injection pattern is central to the protocol.

## Chapter Roadmap

| File | Content |
|------|---------|
| [`result_classification.md`](./result_classification.md) | Worked 10-token generation example traced tick by tick, then the four-case classification with formal state-space analysis |
| [`pair_injection.md`](./pair_injection.md) | Writer spec mechanics, staging entry structure, prediction handoff chain, KV position diagrams, REJECT correction, `skip_spec` |
| [`state_machine_detail.md`](./state_machine_detail.md) | `SpecDecodeState` fields and formal transition table (T1-T11), cycles, `defer_complete` mechanism, correctness properties |
| [`throughput_analysis.md`](./throughput_analysis.md) | Regular vs spec comparison, system-level analysis, accept rate threshold, diagnostic counters, worked deployment examples |
| [`thinking_phase_and_relaxed_acceptance.md`](./thinking_phase_and_relaxed_acceptance.md) | Think tokens, phase tracking, argmax vs relaxed sampling, `check_acceptance` with worked bf16 p_scores examples |

## Relationship to Prior Chapters

This chapter builds directly on:

- **Chapter 2 -- SpecDecodeState**: The data structure layout and field semantics are introduced in [Chapter 2 -- SpecDecodeState](../ch2_threading_and_data_structures/spec_decode_state.md). This chapter focuses on the protocol that drives those fields through their state transitions.
- **Chapter 2 -- DecodeStaging**: The `DecodeStagingEntry` structure, `pending_count` reference counting, and `ReaderClaim` RAII guard are covered in [Chapter 2 -- Decode Staging](../ch2_threading_and_data_structures/decode_staging.md). This chapter shows how spec decode populates those entries differently from regular decode.
- **Chapter 3 -- Regular Decode Flow**: The non-speculative decode loopback, completion, and EOS write-back mechanics from [Chapter 3](../ch3_scheduling_algorithm/index.md) are prerequisites. Speculative decode extends but does not replace the regular flow -- `emit_token`, `stage_eos_writeback`, and `set_position_on_complete` are reused unchanged.
- **Chapter 3 -- Writer Priority**: The Writer priority ordering and `device_sampling_params()` from [Chapter 3 -- Writer Priority and Decode](../ch3_scheduling_algorithm/writer_priority_and_decode.md).

Data structure layouts and memory ordering details are not repeated here. This chapter focuses on the speculative protocol logic, the KV-cache management, and the state machine that orchestrates it.

All source references point to `decode_scheduler.cpp` unless otherwise noted.

---

**Next:** [`result_classification.md`](./result_classification.md)
