# Pair Injection

The Writer injects speculative decode tokens as back-to-back (BASE, SPEC) pairs. This section covers the Writer-side mechanics, the `DecodeStagingEntry` fields that carry pair data, the three-stage prediction handoff from Reader to Writer, the KV correction strategy on REJECT with position diagrams, and the `skip_spec` mechanism for EOS write-back.

## DecodeStagingEntry: The Pair Carrier

Each entry in the decode staging FIFO carries a complete injection pair (defined in `decode_staging.hpp`, lines 17-28):

```cpp
struct DecodeStagingEntry {
    uint32_t slot_id      = INVALID_SLOT;
    uint32_t token_id     = EMPTY_TOKEN;   // BASE token
    uint32_t position     = 0;             // BASE KV position
    uint32_t spec_token_id = EMPTY_TOKEN;  // SPEC token (predicted)
    uint32_t spec_position = 0;            // SPEC KV position
    uint32_t generation   = 0;
    bool     skip_spec    = false;
};
```

| Field | Populated by | Used by |
|---|---|---|
| `slot_id` | Reader (from `ResultDescriptor.slot_id`) | Writer (identifies the user) |
| `token_id` | Reader (from `result.actual_token`) | Writer (BASE inject `token_id`) |
| `position` | Reader (from `result.actual_token_pos`) | Writer (BASE inject `position`) |
| `spec_token_id` | Reader (from `result.predicted_token`) | Writer (SPEC inject `token_id`) |
| `spec_position` | Reader (from `result.predicted_token_pos`) | Writer (SPEC inject `position`) |
| `generation` | `DecodeStaging::stage()` (from current generation counter) | Writer (stale-entry detection via cancel) |
| `skip_spec` | Reader (true only for EOS write-back) | Writer (suppresses SPEC injection) |

The Reader populates the entry via two paths:

1. **`stage_pair()`** (line 510-514): Used for normal speculation. Copies `result.actual_token`, `result.actual_token_pos`, `result.predicted_token`, and `result.predicted_token_pos` directly from the `ResultDescriptor`.

2. **`stage_eos_writeback()`** (line 423-427): Used on completion. Sets `skip_spec=true`, `spec_token_id=EMPTY_TOKEN`, `spec_position=0`. The Writer sees `skip_spec` and injects only the BASE token.

For regular (non-speculative) decode, `spec_token_id` is `EMPTY_TOKEN` and `spec_position` is 0. The Writer's `do_spec` check evaluates to false, and only the BASE is injected. The staging entry format is unified across both modes -- see [Chapter 2 -- Decode Staging](../ch2_threading_and_data_structures/decode_staging.md) for the data structure details.

## Writer Injection Sequence

When the Writer pops a `DecodeStagingEntry` from the FIFO (lines 225-269 of `decode_scheduler.cpp`), it follows these steps:

### Step 1: Cancel Check (line 233)

If `cancel_pending.is_set(uid)`, release the staging claim and finalize. Detailed in [Chapter 3 -- Writer Priority](../ch3_scheduling_algorithm/writer_priority_and_decode.md).

### Step 2: Determine Speculative Mode (line 239)

```cpp
bool do_spec = user_table.spec_decode_enabled[uid] && !entry.skip_spec;
```

Both conditions must be true for a SPEC injection: the user has spec decode enabled, and the entry is not marked as a BASE-only write-back.

### Step 3: Bump in_flight_count (line 244)

```cpp
user_table.in_flight_count[uid].fetch_add(do_spec ? 2 : 1, std::memory_order_release);
```

For speculative pairs, `in_flight_count` is incremented by 2, accounting for both the BASE and SPEC results that will return from the pipeline. This increment happens BEFORE `release()` to maintain the busy-count invariant -- the combined busy count (`in_flight_count + pending_count`) never drops to zero during the handoff. See [Chapter 2 -- Decode Staging](../ch2_threading_and_data_structures/decode_staging.md#writer-thread-processing-flow).

### Step 4: Release Staging Claim (line 245)

```cpp
decode_staging.release(uid);
```

After this point, the slot's liveness is held by `in_flight_count` alone.

### Step 5: Compute Sampling Parameters (line 246)

```cpp
auto sp = device_sampling_params(uid);
```

Outside thinking phase: $k = 1$, $\text{top\_p} = 1.0$ (argmax). Inside thinking phase: user's `top_p`/`top_k` passed through for relaxed acceptance support. See [`thinking_phase_and_relaxed_acceptance.md`](./thinking_phase_and_relaxed_acceptance.md).

### Step 6: Inject BASE (lines 247-256)

```cpp
pipeline_->inject(InjectDescriptor{
    .slot_id = uid,
    .token_id = entry.token_id,
    .position = entry.position,
    .token_type = TokenType::BASE,
    .spec_flag = false,
    .temperature = sp.temperature,
    .top_p = sp.top_p,
    .top_k = sp.top_k,
});
```

### Step 7: Inject SPEC (lines 257-268)

```cpp
if (do_spec) {
    pipeline_->inject(InjectDescriptor{
        .slot_id = uid,
        .token_id = entry.spec_token_id,
        .position = entry.spec_position,
        .token_type = TokenType::SPEC,
        .spec_flag = true,
        .temperature = sp.temperature,
        .top_p = sp.top_p,
        .top_k = sp.top_k,
    });
}
```

Both injections use the same sampling parameters and happen back-to-back on the same Writer tick. The pipeline preserves per-slot FIFO ordering, so the BASE result will be returned to the Reader before the SPEC result. This ordering is critical for the REJECT correction mechanism.

## Prediction Handoff: Reader to Writer via SpecDecodeState

The prediction flows from the pipeline result to the next injection through three handoff points:

### Handoff 1: Pipeline to SpecDecodeState

When the Reader processes a result, the `ResultDescriptor.predicted_token` carries the MTP head's prediction. The Reader stores it:

- **INITIAL** (line 533): `ss.unverified_token[uid] = result.predicted_token`
- **REJECT** (line 561): `ss.unverified_token[uid] = result.predicted_token` (new prediction replaces old)
- **CONTINUE** (line 578): `ss.unverified_token[uid] = result.predicted_token` (next round's prediction)

### Handoff 2: SpecDecodeState to DecodeStagingEntry

When the Reader calls `stage_pair()` (line 510-514), the predicted token flows from `result.predicted_token` into `DecodeStagingEntry.spec_token_id`. The `unverified_token` stored in SpecDecodeState is NOT the source for the staging entry -- both come from the same `ResultDescriptor`. The `unverified_token` copy is kept for verification by the *next* BASE result, not for injection.

### Handoff 3: DecodeStagingEntry to Pipeline

The Writer reads `entry.spec_token_id` and `entry.spec_position` from the staged entry and passes them to `pipeline_->inject()`. This completes the prediction's journey from device model output back to device model input.

```
Pipeline result (D ticks ago)
    |
    | ResultDescriptor.predicted_token
    v
+---+---+
| Reader |----> ss.unverified_token[uid]  (for verification next round)
|        |----> stage_pair()
|        |        |
+--------+        | DecodeStagingEntry.spec_token_id
                  v
            DecodeStaging FIFO
                  |
                  | try_pop()
                  v
            +-----+----+
            |  Writer   |----> pipeline_->inject(SPEC)
            +-----------+
                  |
                  v  (D ticks later)
            Pipeline result
                  |
                  v
            Reader: check_acceptance(uid, unverified_token, result)
```

## KV Cache Diagrams

### INITIAL + First Pair

After prefill for a user with prompt tokens $[t_0, ..., t_{N-1}]$, the first decode result arrives at position $N$. The Reader calls `stage_pair()`:

```
Before Writer processes the pair:
  KV:  [t0] [t1] ... [t_{N-1}] [first_decode_token]
  Pos:  0    1         N-1       N

Writer injects BASE at pos N, SPEC at pos N+1:
  KV:  [t0] [t1] ... [t_{N-1}] [actual_N]  [predicted_{N+1}]
  Pos:  0    1         N-1       N           N+1
                                  ^ BASE      ^ SPEC (unverified)
```

### ACCEPT Path (Prediction Correct)

The BASE result at position $N+1$ confirms `predicted_{N+1}` matches. No KV changes needed -- the speculative data was correct all along.

```
KV:  ... [actual_N]  [predicted_{N+1}]
Pos:      N           N+1
          verified     now confirmed valid
```

When the SPEC result (CONTINUE) arrives, it carries `actual_token` (bonus) and `predicted_token`. The Reader stages a new pair:

```
Writer injects BASE at pos N+1, SPEC at pos N+2:
  KV:  ... [actual_N]  [bonus_{N+1}]  [predicted_{N+2}]
  Pos:      N           N+1            N+2
                         ^ BASE writes   ^ SPEC (unverified)
```

### REJECT Path (Prediction Wrong)

The BASE result at position $N+1$ reveals `actual_token != predicted_{N+1}`. The KV at $N+1$ is stale:

```
Before correction:
  KV:  ... [actual_N]  [WRONG prediction]
  Pos:      N           N+1
                         ^ stale, must be overwritten
```

The Reader calls `stage_pair()` with the correction. Critically, `result.actual_token_pos` in this case is position $N+1$ -- the same position that holds stale speculative data:

```
Writer injects BASE at pos N+1 (overwrite), SPEC at pos N+2:
  KV:  ... [actual_N]  [CORRECTED actual_{N+1}]  [predicted_{N+2}]
  Pos:      N           N+1                        N+2
                         ^ BASE overwrites stale    ^ SPEC (unverified)
```

The pipeline processes BASE before SPEC (per-slot FIFO). So when the SPEC at $N+2$ computes attention, it sees the corrected KV at $N+1$. **No rollback is needed.**

### Why No Explicit Rollback

The "overwrite-not-rollback" property holds because of the three guarantees stated in the [index](./index.md):

1. **Single speculative position**: At most one KV position ($P_{\text{spec}}$) is speculative at any time.
2. **Pipeline FIFO ordering**: The Writer injects BASE before SPEC. The SPEC injection sees the BASE's KV write.
3. **In-place overwrite**: The correction BASE writes to the exact same position as the stale SPEC.

An explicit KV rollback API is never needed. This simplifies the pipeline interface and eliminates a class of race conditions that would arise from asynchronous rollback commands.

## REJECT Correction: Detailed Position Trace

To make the correction mechanics concrete, here is a position-by-position trace across two rounds where the first prediction is wrong:

```
Round 1 (INITIAL):
  Result: actual_token=A at pos=5, predicted_token=X at pos=6
  Stage: BASE=A@5, SPEC=X@6
  Writer: inject A@5 (BASE), inject X@6 (SPEC)
  KV after: [..., A@5, X@6]
  unverified_token = X

Round 2 (BASE arrives -- REJECT):
  Result: actual_token=B at pos=6, predicted_token=Y at pos=7
  B != X, so REJECT
  Stage: BASE=B@6, SPEC=Y@7
  Writer: inject B@6 (BASE, overwrites X), inject Y@7 (SPEC)
  KV after: [..., A@5, B@6, Y@7]
  unverified_token = Y

Round 2 (SPEC arrives -- STALE):
  Result from SPEC@6 (computed with stale X in cache): DISCARD
  No output, no staging.

Round 3 (BASE arrives):
  Result: actual_token=C at pos=7, predicted_token=Z at pos=8
  check_acceptance(Y, result):
    If C == Y -> ACCEPT
    If C != Y -> REJECT (repeat correction)
```

The key observation: in Round 2, the BASE injection at position 6 overwrites the stale X with the correct B. The SPEC at position 7 is processed by the pipeline after the BASE at position 6 (FIFO ordering), so Y's KV computation sees `[..., A@5, B@6]` -- the corrected context.

## `skip_spec`: Suppressing Phantom SPEC on Completion

When the Reader detects completion inside `emit_token`, it calls `stage_eos_writeback()` (lines 423-428):

```cpp
auto stage_eos_writeback = [&]() {
    user_table.post_complete_in_flight[uid].fetch_add(1, std::memory_order_release);
    decode_staging.stage(uid,
        result.actual_token, result.actual_token_pos,
        EMPTY_TOKEN, 0, /*skip_spec=*/true);
};
```

The `skip_spec=true` flag ensures the Writer does NOT inject a SPEC token alongside the EOS write-back. Without this flag, the Writer would see `spec_decode_enabled[uid] == true` and inject a phantom SPEC with `token_id=EMPTY_TOKEN` at position 0 -- producing a spurious pipeline result that no part of the state machine expects. The `skip_spec` flag prevents this by overriding `spec_decode_enabled`.

This flag also causes `in_flight_count` to be incremented by 1 instead of 2 (line 244), correctly reflecting that only one pipeline result will arrive from this injection.

## `in_flight_count` Accounting

For spec decode, `in_flight_count` is bumped by 2 on each pair injection and decremented by 1 on each pipeline result (BASE and SPEC each produce one result). The count reaches zero only when both results from the final pair have been processed.

| Event | `in_flight_count` delta |
|---|---|
| Writer injects pair (BASE + SPEC) | +2 |
| Reader receives BASE result | -1 |
| Reader receives SPEC result | -1 |
| EOS write-back (skip_spec, BASE only) | +1 |
| Reader receives EOS write-back result | -1 |

This accounting is critical for the deferred cancellation protocol: `maybe_finalize_cleanup` requires `in_flight_count == 0` before freeing the slot. With speculative decode, the count takes longer to drain because each round produces two results.

## Injection Sequence Summary

| Case | Injects | BASE position | SPEC position | Notes |
|------|---------|---------------|---------------|-------|
| INITIAL | BASE + SPEC | `actual_token_pos` | `predicted_token_pos` | First pair after prefill |
| CONTINUE | BASE + SPEC | `actual_token_pos` | `predicted_token_pos` | Normal steady-state |
| REJECT | BASE + SPEC | `actual_token_pos` (= prev SPEC pos) | `predicted_token_pos` | BASE overwrites stale KV |
| Completion (EOS) | BASE only | `actual_token_pos` | -- | `skip_spec=true` |
| ACCEPT | (none) | -- | -- | Waits for SPEC result |
| STALE | (none) | -- | -- | Discarded; REJECT already staged pair |

---

**Next:** [`state_machine_detail.md`](./state_machine_detail.md)
