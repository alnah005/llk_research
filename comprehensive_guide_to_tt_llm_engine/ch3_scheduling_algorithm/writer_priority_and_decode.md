# Writer Priority and Decode Path

The Writer thread is the sole entry point to the pipeline. Its loop body implements a strict two-tier priority scheme that determines which token to inject on each tick. This section covers the priority logic, the decode injection path, sampling parameter computation, and the no-work fallthrough.

## Governing Invariant

**Invariant: decode users are never starved by prefill.**

Every decode token waiting in the `DecodeStaging` FIFO is injected before any prefill token is considered. The Writer checks `decode_staging.try_pop()` first; only if the FIFO is empty does it fall through to `prefill_queue.try_front()`. There is no fairness weighting, no round-robin between the two tiers, and no aging mechanism. Decode is unconditionally first.

This invariant is enforced structurally by the ordering of the two `if` blocks in `writer_loop()` (lines 225-342 of `decode_scheduler.cpp`). The `continue` statement after a successful decode injection prevents the prefill path from executing on the same tick.

## Writer Loop Structure

Each iteration follows this decision sequence:

```
Writer tick N:
  |
  +--[1] decode_staging.try_pop(entry)
  |       |
  |       +-- success --> [2] cancel check
  |       |                    |
  |       |                    +-- cancelled --> release + finalize, goto tick N+1
  |       |                    |
  |       |                    +-- active --> [3] compute do_spec
  |       |                                       |
  |       |                                       +--[4] bump in_flight_count
  |       |                                       |
  |       |                                       +--[5] release staging claim
  |       |                                       |
  |       |                                       +--[6] compute device_sampling_params
  |       |                                       |
  |       |                                       +--[7] inject BASE token
  |       |                                       |
  |       |                                       +--[8] if do_spec: inject SPEC token
  |       |                                       |
  |       |                                       +-- goto tick N+1
  |       |
  |       +-- failure (empty) --> [9] try prefill path
  |
  (tick N+1)
```

The loop has no explicit yield or sleep on the empty path. When neither source has work, the `running` atomic load at the top of the next iteration acts as the spin check. This spin-wait design keeps the Writer's wakeup latency at the sub-microsecond level -- critical for re-injecting decode tokens the instant they arrive in staging.

## Decode Path: Step by Step

When `decode_staging.try_pop(entry)` succeeds, the Writer holds a `DecodeStagingEntry` containing:

| Field | Source | Description |
|-------|--------|-------------|
| `entry.slot_id` | Reader (via `stage()`) | The user slot this token belongs to |
| `entry.token_id` | `ResultDescriptor.actual_token` | The sampled BASE token to re-inject |
| `entry.position` | `ResultDescriptor.actual_token_pos` | Device KV position for the BASE token |
| `entry.spec_token_id` | `ResultDescriptor.predicted_token` | Speculative token (if spec decode) |
| `entry.spec_position` | `ResultDescriptor.predicted_token_pos` | Device KV position for the SPEC token |
| `entry.generation` | `decode_staging.generation[uid]` at stage time | Stale-entry detection tag |
| `entry.skip_spec` | Set by `stage_eos_writeback` | If true, suppress SPEC injection |

The Writer processes this entry through six steps:

### Step 1: Cancel Check

```cpp
if (cancel_pending.is_set(uid)) {
    decode_staging.release(uid);
    maybe_finalize_cleanup(uid);
    continue;
}
```

**Invariant: cancelled users never receive pipeline injections after cancel is observed.**

The cancel check is the first operation after the pop. If the cancel bit is set, the entry is dropped, the `pending_count` claim is released, and `maybe_finalize_cleanup()` is called to attempt slot reclamation. The `continue` skips all further processing.

**Reads:** `cancel_pending` bitmap (acquire load on the word containing bit `uid`).

**Writes:** `decode_staging.pending_count[uid]` (fetch_sub via `release()`).

The `pending_count` mechanism guarantees that any stale entry still in the FIFO belongs to a slot whose `cancel_pending` bit is set. This is because `cancel_pending.clear(uid)` runs only during ALLOCATE, which runs only after `maybe_finalize_cleanup()` completes, which requires `pending_count == 0`. Therefore the Writer can safely use the cancel bit as a generation filter -- no generation comparison is needed. The full cancellation protocol is described in [Chapter 2 -- Lock-Free Design](../ch2_threading_and_data_structures/lock_free_design.md).

### Step 2: Determine Speculative Mode

```cpp
bool do_spec = user_table.spec_decode_enabled[uid] && !entry.skip_spec;
```

**Reads:** `user_table.spec_decode_enabled[uid]` (plain bool), `entry.skip_spec`.

The `do_spec` flag determines whether a SPEC token is paired with the BASE injection. For regular (non-speculative) decode, `spec_decode_enabled` is false and `do_spec` is always false. The `skip_spec` flag is set for EOS write-back entries (see [`completion_and_eos_writeback.md`](./completion_and_eos_writeback.md)), which must inject only the BASE token even when spec decode is enabled.

### Step 3: Pre-Increment `in_flight_count`

```cpp
user_table.in_flight_count[uid].fetch_add(
    do_spec ? 2 : 1, std::memory_order_release);
```

**Invariant: the combined busy count (`in_flight_count + pending_count`) for a live slot never transiently reaches zero.**

**Writes:** `user_table.in_flight_count[uid]` (fetch_add with release).

The `in_flight_count` is incremented BEFORE the staging claim is released. This prevents a dangerous zero-crossing:

```
Without pre-increment (BROKEN):
  Writer: release(uid)           // pending_count drops
  --- gap ---                    // in_flight=0, pending=0 momentarily
  Cancel: maybe_finalize_cleanup sees both at zero -> frees slot
  Writer: inject token for freed slot -> corruption

With pre-increment (CORRECT):
  Writer: in_flight_count++      // in_flight >= 1
  Writer: release(uid)           // pending_count drops, but in_flight holds
```

The release ordering on the `fetch_add` ensures that any subsequent `maybe_finalize_cleanup()` on another thread sees the updated count before the staging claim is released.

For non-speculative decode, `in_flight_count` is incremented by 1. For speculative decode, it is incremented by 2 (one for BASE, one for SPEC), because both tokens will produce pipeline results that must be accounted for.

### Step 4: Release Staging Claim

```cpp
decode_staging.release(uid);
```

**Writes:** `decode_staging.pending_count[uid]` (fetch_sub via `release()`).

Decrements `pending_count` for the slot. After this point, the staging entry has been consumed and the Writer owns the token data by value (it was copied during `try_pop`). The slot's liveness is now protected by `in_flight_count` alone.

### Step 5: Compute Sampling Parameters

```cpp
auto sp = device_sampling_params(uid);
```

**Invariant: outside thinking phase, the device always performs argmax sampling ($k = 1$, $\text{top\_p} = 1.0$).**

**Reads:** `user_table.temperature[uid]` (plain float), `user_table.in_thinking_phase[uid]` (acquire load), and conditionally `user_table.top_p[uid]` (plain float) and `user_table.top_k[uid]` (plain int32_t) if in thinking phase.

The `device_sampling_params()` function (lines 770-778) computes the sampling parameters to send to the device along with the token:

```cpp
DeviceSamplingParams device_sampling_params(uint32_t uid) const {
    const float t = user_table.temperature[uid];
    const bool in_thinking = user_table.in_thinking_phase[uid].load(
        std::memory_order_acquire);
    if (!in_thinking) {
        return {t, 1.0f, 1};
    }
    int32_t k = user_table.top_k[uid];
    return {t, user_table.top_p[uid], k};
}
```

**Outside thinking phase** (the common case for regular decode): Forces $k = 1$ and $\text{top\_p} = 1.0$. With $k = 1$, the device returns only the single highest-probability token in `p_indices[0]`, saving device cycles and avoiding unnecessary `p_indices`/`p_scores` traffic. Temperature is passed through but does not change the argmax selection -- the highest-logit token is invariant under monotonic temperature scaling.

**Inside thinking phase:** The user's `top_p` and `top_k` are passed through. These are used by the device to populate the `p_indices` and `p_scores` arrays in the `ResultDescriptor`, which the Reader's `check_acceptance()` function uses for relaxed speculative acceptance (see [Chapter 4 -- Thinking Phase and Relaxed Acceptance](../ch4_speculative_decode/thinking_phase_and_relaxed_acceptance.md)). The `p_indices` array in `ResultDescriptor` has a fixed bound of 32 entries, so values above 32 would read past the array -- but the device kernel caps the effective $k$ accordingly.

The `in_thinking_phase` load uses acquire ordering, paired with the Reader's release store when toggling the flag in `emit_token`. This ensures the Writer sees the most recent phase transition before constructing the next inject.

### Step 6: Pipeline Injection

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

**Writes:** Into the pipeline (via `inject()`).

The BASE token is always injected. For non-speculative decode, this is the only injection. The `InjectDescriptor` carries the slot ID, token ID, KV position, sampling parameters, and type tags. The `prefill_token_id` field defaults to `EMPTY_TOKEN` (not set for decode tokens -- it is only meaningful during prefill).

If `do_spec` is true (speculative decode), a second injection follows immediately:

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

The SPEC injection uses the speculated token from `entry.spec_token_id` at position `entry.spec_position` (one past the BASE position). Both share the same sampling parameters. The two injections happen back-to-back on the same Writer tick. For the full speculative decode protocol, see [Chapter 4](../ch4_speculative_decode/index.md).

For regular (non-speculative) decode, `do_spec` is false and only the BASE token is injected. The `DecodeStagingEntry` still carries `spec_token_id` and `spec_position` fields, but they are set to `EMPTY_TOKEN` and 0 by the Reader's non-spec staging call.

## No-Work Path

When both `decode_staging.try_pop()` and `prefill_queue.try_front()` return false, the Writer loop falls through to the top of the `while` loop. The only operation is the `running.load(std::memory_order_acquire)` check. There is no `_mm_pause()`, no `std::this_thread::yield()`, and no sleep.

**Rationale:** The Writer is pinned to a dedicated CPU core (see [Chapter 2 -- CPU Affinity](../ch2_threading_and_data_structures/cpu_affinity.md)). Yielding or pausing would increase wakeup latency when a decode token arrives in staging. The spin cost is a dedicated core, which is acceptable for a production inference server that already dedicates an entire cluster to inference.

## Timing Cost per Step

The following table shows the time cost of each step in the decode path, relative to pipeline tick duration:

| Step | Operation | Approximate Cost | Blocks? |
|------|-----------|-----------------|---------|
| 1 | `try_pop` from BoundedQueue | Sub-microsecond (uncontended mutex) | No |
| 2 | `cancel_pending.is_set` | Single atomic load | No |
| 3 | Read `spec_decode_enabled` + `skip_spec` | Two plain loads | No |
| 4 | `in_flight_count.fetch_add` | One atomic RMW | No |
| 5 | `pending_count.fetch_sub` | One atomic RMW | No |
| 6 | `device_sampling_params` | One atomic load + scalar reads | No |
| 7 | `pipeline_->inject` (BASE) | Pipeline-dependent | Yes (if pipeline full) |
| 8 | `pipeline_->inject` (SPEC, if applicable) | Pipeline-dependent | Yes (if pipeline full) |

Steps 1-6 combined cost well under 1 microsecond on modern hardware. Step 7 is the dominant cost and is bounded by pipeline throughput, not scheduler overhead.

---

**Next:** [`chunked_prefill.md`](./chunked_prefill.md)
