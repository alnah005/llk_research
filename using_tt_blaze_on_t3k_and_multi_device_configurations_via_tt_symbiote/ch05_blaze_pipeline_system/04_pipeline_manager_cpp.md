# 5.4 Pipeline Manager (C++)

This section covers the C++20 `PipelineManager` -- the continuous-batching engine that sits on the host side and drives the device pipeline. It manages user sessions, token injection, result processing, prefill scheduling, and speculative decoding via three pinned threads and lock-free data structures.

**Prerequisites:** Section 5.3 (PipelineBlock, H2D/D2H sockets), familiarity with C++ atomics and threading.

---

## Architecture Overview

```
  Inference Server (HTTP)
       |
       v
  +--------------------+     request_queue      +------------------+
  |   push_request()   | ----(ISRequest)------> |   API Thread     |
  |   try_pop_response | <---(PMResponse)------ |   (api_loop)     |
  |   try_pop_output   | <---(OutputMessage)--- |                  |
  +--------------------+     response_queue     +------------------+
                              output_queue            |       |
                                              prefill_queue  decode_staging
                                                      |       |
                                                      v       v
  +------------------+                         +------------------+
  |  Writer Thread   | <--- decode_staging --- |  Reader Thread   |
  |  (writer_loop)   |                         |  (reader_loop)   |
  |                  |                         |                  |
  |  inject() -----> |  H2D Socket ------>     |  <--- D2H Socket |
  |                  |  (64-byte pages)   HW   |  read_result()   |
  +------------------+                  Pipeline+------------------+
```

The three threads each have a dedicated role:
- **Writer thread**: injects tokens into the device pipeline (hot path).
- **Reader thread**: consumes results from the device pipeline (hot path).
- **API thread**: processes incoming requests (ALLOCATE, SUBMIT, CONTINUE, CANCEL).

---

## PipelineManager::Impl (pipeline_manager.cpp, line 33)

The `Impl` struct holds all state:

```cpp
struct PipelineManager::Impl {
    std::unique_ptr<PipelineInterface> pipeline_;  // MockPipeline or SocketPipeline
    ManagerParams params;                          // max_users, chunk_size, etc.

    FreeIdPool free_ids;           // bitmap-based slot allocator
    UserTable user_table;          // per-slot parallel vectors
    PromptTable prompt_table;      // per-turn prompt token storage
    CancelBitmap cancel_pending;   // multi-word atomic bitmap
    DecodeStaging decode_staging;  // reader-to-writer FIFO
    PrefillQueue prefill_queue;    // round-robin prefill scheduler
    SpecDecodeState spec_state;    // per-user speculation tracking

    BoundedQueue<ISRequest> request_queue;
    BoundedQueue<PMResponse> response_queue;
    BoundedQueue<OutputMessage> output_queue;

    std::thread writer_thread, reader_thread, api_thread;
    std::atomic<bool> running{false};
};
```

### CPU Affinity (lines 94-155)

Each thread is pinned to a distinct CPU core. `resolve_cpu_assignments()` auto-selects cores from the process's available set, spacing them by `stride = N/3` to avoid sharing physical cores:

```cpp
struct ManagerParams {
    uint32_t max_users = 64;
    uint32_t chunk_size = 24;      // prefill tokens per round-robin slice
    uint32_t max_seq_len = 131072; // 128K context
    uint32_t eos_token = 1;
    int writer_cpu = AUTO_CPU;     // CPU core affinity (AUTO, NOPIN, or core id)
    int reader_cpu = AUTO_CPU;
    int api_cpu = AUTO_CPU;
};
```

Override with explicit core IDs, or disable pinning with `NOPIN_CPU`.

> **Warning:** If the process CPU affinity mask has fewer than 3 available cores, multiple pipeline manager threads share a core. On high-throughput workloads this causes writer/reader contention and up to 2x latency regression. Verify with `taskset -p <pid>`.

---

## Writer Thread (line 203)

The writer loop has two priority levels:

**Priority 1 -- Decode tokens:** Pop from `decode_staging` FIFO. Each entry carries a token that completed one pipeline pass and needs re-injection for the next decode step.

```
writer_loop:
  while running:
    if decode_staging.try_pop(entry):
      if cancel_pending -> release, finalize, continue
      bump in_flight_count (+1 or +2 if spec)
      pipeline_->inject(InjectDescriptor{slot, token, position, BASE})
      if spec_decode: pipeline_->inject({slot, spec_token, spec_pos, SPEC})
      continue

    if prefill_queue.try_front(uid):
      ... inject one prefill token per iteration (chunked) ...
      continue
```

**Priority 2 -- Prefill tokens:** If no decode work, process the prefill queue in chunked round-robin. Each user gets `chunk_size` (default 24) tokens before the queue rotates to the next user. This prevents one long prompt from starving decode tokens.

> **Warning:** The `in_flight_count` is bumped BEFORE releasing the staging claim. This prevents a momentary zero-count during the handoff that could trigger premature cleanup by a concurrent CANCEL.

> **Warning (failure mode):** If `decode_staging` FIFO is full (capacity = `max_users * 4`), `stage()` drops the entry. This is a non-recoverable error -- the loopback token is lost and the user's generation stalls permanently.

### Wire Format (wire_format.hpp)

Each injection is serialized into a 64-byte page:

```
Default layout (16 x uint32):
  [0] slot_id    [1] token_id    [2] position    [3] prefill_token_id
  [4] spec_flag  [5] temperature [6] top_p       [7] top_k
  [8-15] reserved
```

A DeepSeek MD alternative layout is available (`use_deepseek_md_format=true`) with different field positions for compatibility with the on-device kernel's expected format.

> **Warning:** The DeepSeek MD wire format places `slot_id` at word[6] instead of word[0]. Mixing formats between the pipeline manager and device kernel causes silent data corruption -- the slot_id field reads as zero, misdirecting all tokens to slot 0.

---

## Reader Thread (line 350)

The reader loop processes pipeline results:

```
reader_loop:
  while running:
    result = pipeline_->read_result()  // blocking
    if result.slot_id == INVALID_SLOT: break  // sentinel

    ReaderClaim scratch(uid)  // RAII: bumps pending_count
    in_flight_count--

    if cancel_pending -> finalize, continue
    if post_complete_in_flight > 0 -> skip (EOS write-back result)
    if prefill_in_flight > 0 -> skip (non-last prefill result)

    // spec decode state machine or simple emit
    emit_token(result.actual_token, ...)
    stage_pair(uid, result.actual_token, result.predicted_token)
```

The `ReaderClaim` RAII guard (line 337) bumps `pending_count` on construction and releases on destruction. This prevents the cancel cleanup from freeing a slot while the reader is mid-iteration.

### Non-Spec Decode Path

Simple: emit the actual token. If not complete, stage a write-back entry for the writer to re-inject.

### Spec Decode State Machine

The reader maintains per-slot state in `SpecDecodeState`:

```
                     INITIAL
                    (no speculation)
                         |
                    first BASE result
                         |
                    +----v--------+
                    | UNVERIFIED  |  <-- predicted_token stored
                    +----+--------+
                         |
                    next BASE result
                    /           \
            match?               no match?
              |                      |
         +----v----+           +-----v-----+
         | VERIFIED |           | REJECT     |
         | (ACCEPT) |           | new unverif|
         +----+----+           +-----+-----+
              |                      |
         SPEC result            SPEC result
              |                      |
         bonus token            STALE (discard)
```

After an ACCEPT, the reader waits for the paired SPEC result before publishing completion. After a REJECT, the stale SPEC result is discarded.

> **Warning:** The defer-complete mechanism (`defer_complete=true` in `emit_token`) is critical. Without it, the client could send a CONTINUE request (starting the next turn's prefill) before the stale SPEC result is consumed, causing a `prefill_in_flight` count mismatch. If the flush path is never reached (e.g., device hang), the completion message is lost and the user appears stuck in DECODE state forever. Use `dump_diagnostics()` to detect this.

---

## API Thread (line 582)

Processes `ISRequest` messages from the request queue:

| RequestType | Action |
|-------------|--------|
| `ALLOCATE` | `free_ids.allocate()` -> return slot_id in PMResponse |
| `SUBMIT` | Store prompt, set state=PREFILL, push to prefill_queue |
| `CONTINUE` | Resume multi-turn: local or disaggregated decode |
| `CANCEL` | Mark cancel_pending, advance generation, maybe finalize |

### User Lifecycle State Machine

```
  ALLOCATE         SUBMIT          last prefill token    max_new_tokens
  ---------> INACTIVE ---------> PREFILL ----------------> DECODE ---------> COMPLETE
                ^                                                             |
                |                          CANCEL (any state)                 |
                +------------ finalize_cleanup() <----------------------------+
                              (reset_kv, free slot)

  CONTINUE (from COMPLETE) ---------> PREFILL (multi-turn, KV preserved)
```

---

## Cancellation Protocol

Cancellation is non-trivial because tokens may be in-flight in the device pipeline:

1. **API thread**: `cancel_pending.mark(uid)` (release), stashes `cancel_request_id`, `decode_staging.advance_generation(uid)`.
2. **Writer thread**: on next pop, sees `cancel_pending` -> releases entry, calls `maybe_finalize_cleanup()`.
3. **Reader thread**: on next result, sees `cancel_pending` -> skips processing, calls `maybe_finalize_cleanup()`.
4. **`maybe_finalize_cleanup()`** (line 680): only runs when `in_flight_count == 0` AND `pending_count == 0`. Uses CAS on `state` to ensure exactly one thread wins cleanup.

```cpp
struct CancelBitmap {
    std::vector<std::atomic<uint64_t>> bits;
    void mark(uint32_t uid)  { bits[uid/64].fetch_or(1ULL << (uid%64), release); }
    bool is_set(uint32_t uid) { return (bits[uid/64].load(acquire) >> (uid%64)) & 1; }
};
```

> **Warning:** The `cancel_request_id` is stashed BEFORE `cancel_pending.mark()` (line 651). The release-acquire ordering on the CancelBitmap guarantees visibility. If this ordering were reversed, the cleanup winner could read a stale request_id and emit a PMResponse with the wrong request_id.

---

## Key Data Structures

### UserTable (user_table.hpp, line 17)

Parallel vectors indexed by slot_id, sized at `max_users` (default 64):

```cpp
struct UserTable {
    vector<atomic<UserState>> state;          // INACTIVE/PREFILL/DECODE/COMPLETE
    vector<atomic<uint32_t>> current_position;
    vector<atomic<uint32_t>> in_flight_count; // tokens in device pipeline
    vector<atomic<uint32_t>> prefill_in_flight;
    vector<uint32_t> max_new_tokens, tokens_generated;
    vector<bool> spec_decode_enabled, ignore_eos;
    vector<float> temperature, top_p;
    vector<int32_t> top_k;
};
```

### DecodeStaging (decode_staging.hpp)

Reader-to-writer handoff FIFO. Each entry carries `{slot_id, token_id, position, spec_token_id, spec_position, generation, skip_spec}`. The `pending_count` per slot is the reference count covering staged FIFO entries and in-progress reader iterations. Combined with `in_flight_count`, it forms a busy count: the slot is safe to free only when both reach zero.

### FreeIdPool (free_id_pool.hpp)

Bitmap-based allocator using `__builtin_ctzll` + CAS for O(1) allocation. Supports up to `max_users` slots across multiple atomic 64-bit words.

### PrefillQueue (prefill_queue.hpp)

Mutex-guarded deque with `push/pop_front/rotate/remove`. The `rotate()` method moves the front user to the back for chunked round-robin scheduling.

---

## MockConfig vs SocketConfig

The `PipelineConfig` variant selects the backend: `std::variant<MockConfig, SocketConfig>`.

**MockConfig**: In-memory testing backend. Deterministic token model: `actual = token_id + 1` (accept) or `+3` (reject). Configurable latency and accept rate.

**SocketConfig**: Real hardware via H2D/D2H sockets. Connects with a configurable timeout (default 30s). Supports both default and DeepSeek MD wire formats (`use_deepseek_md_format`).

> **Warning:** `SocketConfig` requires the `PM_HAS_METALIUM` compile flag. Building without it triggers a compile-time assertion if a `SocketConfig` is passed.

> **Warning (failure mode):** `SocketPipeline::shutdown()` sends a sentinel via H2D and drains D2H until it receives the sentinel back. If the device-side kernel crashed and never echoes the sentinel, `shutdown()` hangs forever.

---

## Diagnostic Dump

`dump_diagnostics()` (lines 820-852) prints per-slot state for debugging hangs:

```
=== PipelineManager diagnostics ===
  running: yes
  prefill_queue_size: 2
  decode_staging_size: 5
  slot 3: state=DECODE in_flight=1 pos=142 generated=18/256 cancel=0
  slot 7: state=PREFILL in_flight=0 pos=89 generated=0/512 cancel=0
===================================
```

Call this from a signal handler (SIGUSR1) or watchdog timer to diagnose pipeline stalls.

---

| Previous | Up | Next |
|----------|-----|------|
| [5.3 Stage Kinds and Execution](03_stage_kinds_and_execution.md) | [Table of Contents](../README.md) | [5.5 Hardware Configurations](05_hardware_configurations.md) |
