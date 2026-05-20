# Design Decisions

[< Previous: Integration Methods](integration_methods.md)

---

This file catalogs the sixteen key design decisions that shaped tt-llm-engine's architecture. Each entry follows a consistent structure: the decision as a concise statement, the code that implements it, the alternative that was considered, why the alternative was rejected, and a cross-reference to the chapter where the full implementation details live.

These decisions are not incidental -- they define the system's character. Taken together, they produce a scheduler that operates at microsecond granularity on the hot path, never dynamically allocates during steady-state execution, and is structured for eventual migration to a hardware co-processor.

## Summary Table

| # | Decision | Rationale | Key Source | Chapter Reference |
|---|----------|-----------|------------|-------------------|
| 1 | Configurable capacity (max_users) | Support different pipeline widths without recompilation | `decode_types.hpp`, `SchedulerParams::max_users` | [Ch2](../ch2_threading_and_data_structures/index.md) |
| 2 | Decode priority over prefill | Minimize TPOT; decode users actively waiting for output | `decode_scheduler.cpp` (writer loop) | [Ch3](../ch3_scheduling_algorithm/writer_priority_and_decode.md) |
| 3 | Chunked prefill (default 24) | Prevent long prompts from starving decode for thousands of ticks | `decode_types.hpp` DEFAULT_CHUNK_SIZE=24, `prefill_queue.hpp` | [Ch3](../ch3_scheduling_algorithm/chunked_prefill.md) |
| 4 | KV cache retention on COMPLETE | Enable multi-turn without re-prefilling; KV reused on CONTINUE | `decode_scheduler.cpp` (COMPLETE state handling) | [Ch6](../ch6_lifecycle_is_integration_kv_tiering/session_state_machine.md) |
| 5 | Spec decode bandwidth cost (2 slots/user) | Up to 2x individual throughput at cost of halved aggregate concurrency | `decode_staging.hpp` (DecodeStagingEntry), `decode_scheduler.cpp` | [Ch4](../ch4_speculative_decode/throughput_analysis.md) |
| 6 | FIFO default scheduling | Predictable, fair behavior; pluggable policy interface for SLA | `prefill_queue.hpp`, `decode_staging.hpp` FIFO | [Ch3](../ch3_scheduling_algorithm/writer_priority_and_decode.md) |
| 7 | Lock-free hot path (Writer + Reader) | Microsecond-scale scheduling; no mutex on per-tick injection/result path | `decode_staging.hpp`, `free_id_pool.hpp`, `user_table.hpp` | [Ch2](../ch2_threading_and_data_structures/lock_free_design.md) |
| 8 | No KV rollback on spec mismatch | Re-inject at same position overwrites stale entry; only one speculative depth | `decode_scheduler.cpp` (REJECT path) | [Ch4](../ch4_speculative_decode/pair_injection.md) |
| 9 | No PM waiting queue | Return admission failure promptly; IS owns wait/retry/timeout policy | `decode_scheduler.cpp` (ALLOCATE), `free_id_pool.hpp` | [Ch1](../ch1_architecture_and_layering/ownership_contract.md) |
| 10 | Fixed-size pre-allocated data structures | No dynamic allocation on hot path; maps to hardware registers/SRAM | All files in `src/common/`, `src/scheduler/decode/` | [Ch2](../ch2_threading_and_data_structures/index.md) |
| 11 | Generation counters for stale detection | Writer drops stale entries by generation comparison; no timing dependencies | `decode_staging.hpp` (generation field) | [Ch2](../ch2_threading_and_data_structures/decode_staging.md) |
| 12 | ReaderClaim RAII pattern | Prevents premature finalization during reader iteration window | `decode_staging.hpp` (pending_count), ReaderClaim | [Ch2](../ch2_threading_and_data_structures/lock_free_design.md) |
| 13 | defer_complete for spec decode | Prevents client from seeing completion before stale SPEC result consumed | `spec_decode_state.hpp` (pending_complete, signal_to_exit) | [Ch4](../ch4_speculative_decode/state_machine_detail.md) |
| 14 | pending_count dual reference counting | Covers FIFO entries + reader iterations; closes cancel/recycle race | `decode_staging.hpp` (pending_count) | [Ch2](../ch2_threading_and_data_structures/decode_staging.md) |
| 15 | EOS write-back pattern | Writes completing token to KV so next turn sees terminator in context | `decode_staging.hpp` (skip_spec), `user_table.hpp` (post_complete_in_flight) | [Ch3](../ch3_scheduling_algorithm/completion_and_eos_writeback.md) |
| 16 | Keep KV alive until IS-driven eviction | Completed sessions stay resident until CANCEL; PM does not own eviction | `decode_types.hpp` (UserState::COMPLETE), IS-driven CANCEL | [Ch6](../ch6_lifecycle_is_integration_kv_tiering/kv_cache_tiering.md) |

---

## 1. Configurable Capacity (max_users)

**Decision:** The maximum number of concurrent users is configurable at construction time via `SchedulerParams::max_users`, not hardcoded.

**Code:**

```cpp
// include/tt_llm_engine/scheduler/decode/decode_types.hpp
static constexpr uint32_t DEFAULT_MAX_USERS = 64;

struct SchedulerParams {
    uint32_t max_users = DEFAULT_MAX_USERS;
    ...
};
```

Source: `include/tt_llm_engine/scheduler/decode/decode_types.hpp`, lines 20-30.

Every data structure sizes itself from this parameter:

```cpp
// src/common/free_id_pool.hpp
explicit FreeIdPool(uint32_t max_users = DEFAULT_MAX_USERS)
    : max_users_(max_users),
      num_words_((max_users + BITS_PER_WORD - 1) / BITS_PER_WORD),
      bitmap(num_words_) { ... }

// src/common/user_table.hpp
explicit UserTable(uint32_t max_users = DEFAULT_MAX_USERS)
    : max_users_(max_users),
      state(max_users),
      current_position(max_users),
      ... { }
```

**Alternative considered:** Hardcode `MAX_USERS = 64` as a compile-time constant throughout.

**Why rejected:** Different pipeline configurations have different slot budgets. A 16-device pipeline with speculative decode might support 32 users (each consuming 2 slots from 64 total), while a wider pipeline or a time-sliced admission scheme might support 128+. Hardcoding 64 would force recompilation for different deployment targets. The backward-compatible constant `MAX_USERS = DEFAULT_MAX_USERS` is retained for existing code:

```cpp
static constexpr uint32_t MAX_USERS = DEFAULT_MAX_USERS;
```

Source: `include/tt_llm_engine/scheduler/decode/decode_types.hpp`, line 27.

**Tradeoff:** Runtime-sized data structures use `std::vector` rather than `std::array`, introducing one level of indirection. However, all vectors are allocated once at construction and never resized -- no dynamic allocation occurs on the hot path. The vectors' backing memory is contiguous and cache-friendly.

**Cross-reference:** [Chapter 2: Threading and Data Structures](../ch2_threading_and_data_structures/index.md) -- all data structures use this parameter.

---

## 2. Decode Priority Over Prefill

**Decision:** The Writer thread always drains decode tokens from the `DecodeStaging` FIFO before processing any prefill tokens from the `PrefillQueue`. Decode has unconditional priority.

**Code:** The Writer loop is structured as a two-tier priority check:

```
// Writer loop (pseudocode from decode_scheduler.cpp):
// Tier 1: decode tokens from staging FIFO
if (staging.try_pop(entry)) {
    // ... process decode token, inject into pipeline
}
// Tier 2: prefill tokens (only reached when staging is empty)
else if (prefill_queue.try_front(uid)) {
    // ... process prefill token
}
```

**Alternative considered:** Fair-share scheduling between decode and prefill, giving each a proportional slice of pipeline bandwidth.

**Why rejected:** Decode users are actively generating output -- every tick of latency is directly visible as increased TPOT (Time Per Output Token). Prefill is background KV cache population that does not produce user-visible output until complete. A single pipeline tick spent on prefill when a decode token is waiting adds one tick of latency to every decode user's output stream. The timescale separation is fundamental: IS operates at millisecond granularity, PM at microsecond granularity.

**Tradeoff:** A long burst of decode tokens can delay prefill starts, increasing TTFT (Time To First Token) for newly submitted users. The chunked prefill mechanism (Decision 3) mitigates the reverse problem (long prefills starving decode), but there is no reciprocal mechanism to bound decode's preemption of prefill.

**Cross-reference:** [Chapter 3: Scheduling Algorithm](../ch3_scheduling_algorithm/writer_priority_and_decode.md) -- full Writer loop analysis.

---

## 3. Chunked Prefill (Default 24 Tokens)

**Decision:** Long prompts are injected in fixed-size chunks (default 24 tokens) with round-robin rotation across queued prefill users, rather than injecting an entire prompt contiguously.

**Code:**

```cpp
// include/tt_llm_engine/scheduler/decode/decode_types.hpp
static constexpr uint32_t DEFAULT_CHUNK_SIZE = 24;

struct SchedulerParams {
    uint32_t chunk_size = DEFAULT_CHUNK_SIZE;
    ...
};
```

Source: `include/tt_llm_engine/scheduler/decode/decode_types.hpp`, line 22.

The PrefillQueue supports rotation for round-robin interleaving:

```cpp
// src/scheduler/decode/prefill_queue.hpp
void rotate() {
    std::lock_guard<std::mutex> lock(mtx);
    if (queue.size() > 1) {
        uint32_t front = queue.front();
        queue.pop_front();
        queue.push_back(front);
    }
}
```

After exhausting a chunk, the Writer resets `prefill_chunk_remaining` to `chunk_size` and calls `rotate()` to move the current user to the back of the queue.

**Alternative considered:** Inject entire prompts contiguously (no chunking). Simpler code, no rotation logic.

**Why rejected:** A single 128K-token prompt would monopolize the pipeline for 128K ticks, starving all other prefill users for the duration. With chunked prefill at 24 tokens per chunk, a second user's prefill begins after at most 24 ticks. The chunk size of 24 is a tunable that balances prefill fairness against the overhead of rotation and per-chunk bookkeeping.

**Tradeoff:** Smaller chunks increase the number of PrefillQueue rotations and reduce prefill throughput due to more frequent context switches. The default of 24 was chosen as a balance: large enough to amortize per-chunk overhead, small enough to limit worst-case decode starvation to 24 ticks between yielding to decode.

**Cross-reference:** [Chapter 3: Scheduling Algorithm](../ch3_scheduling_algorithm/chunked_prefill.md) -- chunked round-robin mechanics.

---

## 4. KV Cache Retention on COMPLETE

**Decision:** When a user's generation completes (EOS, max_new_tokens, or context exhaustion), the user transitions to COMPLETE state but the KV cache is retained and `current_position` is preserved. The slot is not freed until an explicit CANCEL.

**Code:**

```cpp
// include/tt_llm_engine/scheduler/decode/decode_types.hpp
enum class UserState : uint8_t {
    INACTIVE = 0,
    PREFILL  = 1,
    DECODE   = 2,
    COMPLETE = 3,  // KV retained, slot still allocated
};
```

The state machine allows CONTINUE from COMPLETE, re-entering PREFILL from the existing position:

```
INACTIVE -> PREFILL -> DECODE -> COMPLETE -> (CONTINUE) -> PREFILL -> ...
                                          -> (CANCEL)   -> INACTIVE
```

**Alternative considered:** Auto-free the slot on completion, requiring re-allocation and full re-prefill for multi-turn conversations.

**Why rejected:** Multi-turn conversations are the primary use case for production LLM serving. Re-prefilling the entire conversation history on every turn would waste pipeline bandwidth proportional to the accumulated context length. By retaining KV, a CONTINUE operation only needs to prefill the new turn's prompt tokens, starting from the existing `current_position`.

**Tradeoff:** Completed sessions occupy a slot and device DRAM until explicitly cancelled. In a system with 64 slots, idle completed sessions reduce the number of available slots for new users. The IS must implement an eviction policy (e.g., LRU idle-time-based CANCEL) to reclaim slots under pressure. The cost is that the IS owns eviction policy, but this is consistent with the IS/PM ownership boundary (see Decision 16).

**Cross-reference:** [Chapter 6: Lifecycle, Integration, and KV Tiering](../ch6_lifecycle_is_integration_kv_tiering/session_state_machine.md) -- state machine; [multi_turn_continue.md](../ch6_lifecycle_is_integration_kv_tiering/multi_turn_continue.md) -- CONTINUE mechanics.

---

## 5. Speculative Decode Bandwidth Cost

**Decision:** Each speculative decode user consumes 2 pipeline slots per round (a BASE/SPEC pair), trading aggregate pipeline concurrency for individual throughput.

**Code:**

The `DecodeStagingEntry` carries both base and speculative token information in a single entry:

```cpp
// src/scheduler/decode/decode_staging.hpp
struct DecodeStagingEntry {
    uint32_t slot_id = INVALID_SLOT;
    uint32_t token_id = EMPTY_TOKEN;      // BASE token
    uint32_t position = 0;                // BASE position
    uint32_t spec_token_id = EMPTY_TOKEN; // SPEC token
    uint32_t spec_position = 0;           // SPEC position
    uint32_t generation = 0;
    bool skip_spec = false;
};
```

The Writer bumps `in_flight_count` by 2 before injecting the pair:

```
// Writer (pseudocode):
in_flight_count[uid] += (do_spec ? 2 : 1);  // bump BEFORE releasing staging claim
pipeline_->inject(base_descriptor);
pipeline_->inject(spec_descriptor);           // back-to-back
```

**Alternative considered:** Multi-token speculation (predict N tokens ahead for N+1 throughput per round).

**Why rejected:** The current MTP (Multi-Token Prediction) approach predicts exactly one token ahead. Each spec user produces at most 2 output tokens per round (on match) at a cost of 2 pipeline slots. The design keeps speculation depth at 1, which eliminates the need for KV rollback (see Decision 8). With N pipeline slots total, M spec users consume 2M slots, leaving N-2M for regular users. Example: with D=64 pipeline slots, 20 spec users + 24 regular users = 64 slots consumed.

The `spec_accepts` and `spec_rejects` counters in `UserTable` enable runtime monitoring of accept rates:

```cpp
std::vector<uint32_t> spec_accepts;
std::vector<uint32_t> spec_rejects;
```

Source: `src/common/user_table.hpp`, lines 43-44.

When to enable spec decode: generally worthwhile when accept rate exceeds ~80%.

**Cross-reference:** [Chapter 4: Speculative Decode](../ch4_speculative_decode/throughput_analysis.md) -- throughput analysis; [pair_injection.md](../ch4_speculative_decode/pair_injection.md) -- Writer mechanics.

---

## 6. FIFO Default Scheduling

**Decision:** The default scheduling policy is FIFO -- users are served in the order they enter the prefill queue, and decode tokens are processed in FIFO order from the staging FIFO. A pluggable `SchedulerPolicy` interface supports alternative policies (e.g., `PriorityPolicy` for SLA-aware scheduling).

**Code:**

The PrefillQueue is a straightforward FIFO deque:

```cpp
// src/scheduler/decode/prefill_queue.hpp
struct PrefillQueue {
    std::deque<uint32_t> queue;
    std::mutex mtx;

    void push(uint32_t uid) {
        std::lock_guard<std::mutex> lock(mtx);
        queue.push_back(uid);
    }

    bool try_front(uint32_t& uid) {
        std::lock_guard<std::mutex> lock(mtx);
        if (queue.empty()) return false;
        uid = queue.front();
        return true;
    }
    ...
};
```

The decode staging FIFO (`DecodeStaging::fifo`) is also a BoundedQueue that pops in FIFO order.

**Alternative considered:** Priority scheduling based on SLA class, user wait time, or request urgency.

**Why rejected for default:** FIFO provides predictable, deterministic behavior that is easiest to reason about during development and debugging. Priority scheduling introduces the risk of starvation (low-priority users never served) and requires policy configuration that varies across deployments. The architecture supports priority scheduling as a policy plug-in without changing the core data structures -- the PrefillQueue could be replaced with a priority queue, and the DecodeStaging FIFO could be augmented with priority lanes. The IS layer is the appropriate owner of scheduling policy, not the PM.

**Cross-reference:** [Chapter 1: Architecture and Layering](../ch1_architecture_and_layering/ownership_contract.md) -- IS owns business-level fairness policy; [Chapter 3](../ch3_scheduling_algorithm/writer_priority_and_decode.md) -- Writer loop structure.

---

## 7. Lock-Free Hot Path

**Decision:** The Writer and Reader thread loops -- the "hot path" that runs at pipeline tick frequency -- use only atomic operations and SPSC staging. No mutexes are acquired on the hot path.

**Code:**

The key hot-path data is communicated via atomics:

```cpp
// src/common/user_table.hpp (selected atomic fields)
std::vector<std::atomic<UserState>> state;           // Writer reads, Reader writes
std::vector<std::atomic<uint32_t>> in_flight_count;  // Writer increments, Reader decrements
std::vector<std::atomic<uint32_t>> current_position;
std::vector<std::atomic<bool>> in_thinking_phase;    // Reader writes (release), Writer reads (acquire)
```

The CancelBitmap uses atomic fetch_or/load with release/acquire semantics:

```cpp
// src/common/user_table.hpp
void mark(uint32_t uid) {
    bits[w].fetch_or(uint64_t(1) << bit, std::memory_order_release);
}
bool is_set(uint32_t uid) const {
    return (bits[w].load(std::memory_order_acquire) >> bit) & 1;
}
```

Key atomic patterns used on the hot path:

| Pattern | Where | Ordering |
|---------|-------|----------|
| CAS on bitmap word | `FreeIdPool::allocate()` | acq_rel / relaxed |
| fetch_or (set bit) | `FreeIdPool::free()` | release |
| fetch_add / fetch_sub | `DecodeStaging::pending_count` | acq_rel |
| load / store | `UserTable::state`, `in_thinking_phase` | acquire / release |
| compare_exchange_strong | `maybe_finalize_cleanup` (state -> INACTIVE) | acq_rel |

The PrefillQueue mutex is acceptable because it is only contended on SUBMIT (a rare IS-initiated operation), not on per-tick injection.

**Alternative considered:** Use a single scheduler lock protecting all shared state, simplifying the concurrency model.

**Why rejected:** The hot path runs at pipeline tick frequency -- potentially hundreds of thousands of iterations per second. A mutex acquisition on every tick would add microseconds of latency per iteration and create contention between the Writer and Reader threads. The atomic-based design enables both threads to operate independently with sub-microsecond coordination overhead. The complexity cost is paid in the form of careful memory ordering annotations and the `pending_count` reference counting mechanism. The codebase uses TSAN (`DS_ENABLE_TSAN`) to validate ordering correctness.

**Cross-reference:** [Chapter 2: Threading and Data Structures](../ch2_threading_and_data_structures/lock_free_design.md) -- memory ordering discipline; [threading_model.md](../ch2_threading_and_data_structures/threading_model.md) -- thread interaction matrix.

---

## 8. No KV Rollback for Speculative Mismatch

**Decision:** When speculative decode produces a mismatch (the predicted token does not match the actual sampled token), the stale KV entry is overwritten by re-injecting the correct token at the same position. No explicit KV rollback or invalidation is performed.

**Code:**

On REJECT, the Reader stages a pair where the BASE token goes to the previous speculative position (overwriting the stale KV entry):

```
// Reader REJECT path (pseudocode):
// BASE at spec_position (overwrites stale KV)
// SPEC at spec_position + 1 (new speculation)
staging.stage(uid, correct_token, spec_position,
              new_prediction, spec_position + 1);
```

**Alternative considered:** Explicit KV rollback: on mismatch, issue a special command to the device to invalidate the stale KV entry before writing the correction.

**Why rejected:** The design limits speculation depth to exactly 1 token. At any given time, there is at most one unverified speculative KV entry per user. Because the pipeline processes tokens in order, injecting the correction at the same position overwrites the stale entry before any downstream token could read it. This eliminates the need for a rollback command entirely, simplifying both the host scheduler and the device kernel. A deeper speculation scheme (predicting N tokens ahead) would require rollback of up to N-1 entries, adding a KV invalidation command to the wire protocol and a rollback path in the device kernel.

**Tradeoff:** This limits speculation depth to exactly one token ahead. Multi-token speculation could theoretically yield higher speedups at higher risk, but the single-token design keeps the state machine manageable (four cases: ACCEPT, CONTINUE, REJECT, STALE) and avoids a downstream invalidation chain.

**Cross-reference:** [Chapter 4: Speculative Decode](../ch4_speculative_decode/pair_injection.md) -- REJECT injection mechanics; [result_classification.md](../ch4_speculative_decode/result_classification.md) -- the four result cases.

---

## 9. No PM Waiting Queue

**Decision:** When the PM cannot admit a new user (no free slots), it returns an immediate failure (`error_code=1` in the `SchedulerResponse`). It does not maintain an internal waiting queue.

**Code:**

The FreeIdPool returns `INVALID_SLOT` when no bits are set:

```cpp
// src/common/free_id_pool.hpp
uint32_t allocate() {
    for (uint32_t w = 0; w < num_words_; w++) {
        uint64_t current = bitmap[w].load(std::memory_order_relaxed);
        while (current != 0) {
            uint32_t bit = static_cast<uint32_t>(__builtin_ctzll(current));
            uint64_t mask = uint64_t(1) << bit;
            if (bitmap[w].compare_exchange_weak(
                    current, current & ~mask,
                    std::memory_order_acq_rel, std::memory_order_relaxed)) {
                return w * BITS_PER_WORD + bit;
            }
        }
    }
    return INVALID_SLOT;  // no free slots
}
```

The ALLOCATE handler checks for `INVALID_SLOT` and returns an error immediately:

```
// ALLOCATE handler (pseudocode):
uint32_t uid = free_ids.allocate();
if (uid == INVALID_SLOT) {
    push_response({.request_id = req.request_id, .error_code = 1});
    return;
}
```

**Alternative considered:** Queue incoming ALLOCATE requests internally and admit users as slots become available.

**Why rejected:** A PM-owned waiting queue would mean the PM owns user-facing wait/retry/timeout policy, violating the IS/PM ownership boundary (see [Chapter 1](../ch1_architecture_and_layering/ownership_contract.md)). Different inference server deployments have different SLA requirements -- some should return HTTP 503 immediately, others should queue with a deadline, others should evict idle sessions. All of these are IS-level policy decisions. By returning failure promptly, the PM keeps its responsibility clean: it manages hardware resources, not business logic. The IS receives the failure and applies its own wait/retry/timeout/eviction strategy.

**Cross-reference:** [Chapter 1: Architecture and Layering](../ch1_architecture_and_layering/ownership_contract.md) -- IS/PM responsibility matrix; [Chapter 6](../ch6_lifecycle_is_integration_kv_tiering/is_integration_patterns.md) -- backpressure semantics.

---

## 10. Fixed-Size Pre-Allocated Data Structures

**Decision:** All data structures are pre-allocated at construction time with fixed sizes derived from `max_users` and `max_seq_len`. No dynamic memory allocation occurs on the hot path.

**Code:**

| Data Structure | Backing | Size |
|---------------|---------|------|
| `FreeIdPool` | `vector<atomic<uint64_t>>` | `ceil(max_users / 64)` words |
| `UserTable` | Parallel `vector`s indexed by slot_id | `max_users` entries per vector |
| `DecodeStaging` | `BoundedQueue` + per-slot vectors | FIFO capacity = `max_users * 4` |
| `PrefillQueue` | `std::deque<uint32_t>` | Up to `max_users` entries |
| `SpecDecodeState` | Parallel `vector`s | `max_users` entries per vector |
| `BoundedQueue` (IS-PM) | `vector<T>` ring buffer | Request: `2*max_users`, Output: `256*max_users` |
| `PromptTable` | Flat 2D array (max_users x max_seq_len) | Fixed at construction |

The UserTable is a set of parallel vectors sized at construction:

```cpp
// src/common/user_table.hpp
explicit UserTable(uint32_t max_users = DEFAULT_MAX_USERS)
    : max_users_(max_users),
      state(max_users),
      current_position(max_users),
      max_new_tokens(max_users, 0),
      tokens_generated(max_users, 0),
      in_flight_count(max_users),
      ... { }
```

The PromptTable pre-allocates a flat 2D array:

```cpp
// src/common/user_table.hpp
explicit PromptTable(uint32_t max_users = DEFAULT_MAX_USERS,
                     uint32_t max_seq_len = DEFAULT_MAX_SEQ_LEN)
    : max_users_(max_users),
      max_seq_len_(max_seq_len),
      tokens_(static_cast<size_t>(max_users) * max_seq_len, 0),
      lengths(max_users, 0) {}
```

**Alternative considered:** Use dynamic containers (e.g., `std::unordered_map`, `std::queue` with heap allocation) for simpler code.

**Why rejected:** The engine is designed for eventual migration to a hardware co-processor where dynamic allocation is impossible. Pre-allocated parallel arrays map directly to register files or SRAM. Bitmaps map to single machine words (for max_users <= 64). Bounded queues map to DMA FIFOs. Integer mode constants (no string comparisons in scheduling loops) map to register compares.

| Data Structure | Hardware Analog |
|---------------|----------------|
| `FreeIdPool` bitmap | Single 64-bit register (for max_users <= 64) |
| `UserTable` parallel arrays | Register file / SRAM |
| `CancelBitmap` | Single 64-bit register (for max_users <= 64) |
| `DecodeStaging` FIFO | DMA FIFO |
| `BoundedQueue` ring buffer | DMA ring buffer |

Even on the host, avoiding `malloc` on the hot path eliminates a source of unpredictable latency.

**Tradeoff:** Memory footprint is proportional to `max_users * max_seq_len` regardless of actual occupancy. For the default configuration (64 users, 128K tokens), the `PromptTable` alone allocates `64 * 131072 * 4 = 32 MB`. This is acceptable for a server-class deployment but may be excessive for edge devices. Reducing `max_seq_len` proportionally reduces the footprint.

**Cross-reference:** [Chapter 2: Threading and Data Structures](../ch2_threading_and_data_structures/index.md) -- all data structure files.

---

## 11. Generation Counters for Stale Entry Detection

**Decision:** Each slot has a monotonically increasing generation counter in `DecodeStaging`. When a user is cancelled, the generation is advanced. The Writer drops `DecodeStagingEntry` entries whose generation does not match the current generation, eliminating stale entries without timing dependencies.

**Code:**

```cpp
// src/scheduler/decode/decode_staging.hpp
std::vector<std::atomic<uint32_t>> generation;

void advance_generation(uint32_t uid) {
    generation[uid].fetch_add(1, std::memory_order_release);
}

void stage(uint32_t uid, uint32_t tok, uint32_t pos,
           uint32_t spec_tok, uint32_t spec_pos, bool skip_spec = false) {
    pending_count[uid].fetch_add(1, std::memory_order_acq_rel);
    uint32_t gen = generation[uid].load(std::memory_order_relaxed);
    DecodeStagingEntry entry{
        .slot_id = uid,
        .token_id = tok,
        .position = pos,
        .spec_token_id = spec_tok,
        .spec_position = spec_pos,
        .generation = gen,
        .skip_spec = skip_spec,
    };
    if (!fifo.try_push(entry)) {
        pending_count[uid].fetch_sub(1, std::memory_order_acq_rel);
        std::cerr << "FATAL: decode_staging FIFO full" << std::endl;
    }
}
```

Source: `src/scheduler/decode/decode_staging.hpp`, lines 55-79.

The Writer checks the generation on pop:

```
// Writer (pseudocode):
DecodeStagingEntry entry;
if (staging.try_pop(entry)) {
    if (entry.generation != staging.generation[entry.slot_id].load(...)) {
        staging.release(entry.slot_id);  // stale -- drop and release pending_count
        continue;
    }
    // ... process entry
}
```

**Alternative considered:** Drain the staging FIFO synchronously on cancel, removing all entries for the cancelled slot.

**Why rejected:** Synchronous draining would require the API thread to lock the FIFO and scan it, blocking the Writer thread. Generation counters are a lock-free mechanism: the API thread atomically increments the generation (a single `fetch_add`), and the Writer lazily drops stale entries during its normal loop. There are no timing dependencies between the cancel and the Writer's drain.

**Tradeoff:** Adds one `atomic<uint32_t>` per slot in `DecodeStaging` and one `uint32_t` field per staging entry. The cost is negligible.

**Cross-reference:** [Chapter 2: Threading and Data Structures](../ch2_threading_and_data_structures/decode_staging.md) -- staging mechanics; [Chapter 6](../ch6_lifecycle_is_integration_kv_tiering/deferred_cancellation.md) -- cancellation protocol.

---

## 12. ReaderClaim RAII Pattern

**Decision:** The Reader thread bumps `pending_count` for a slot at the start of each iteration via an RAII guard (`ReaderClaim`). The destructor releases the count and calls `maybe_finalize_cleanup` if a cancel is pending. This prevents premature finalization during the window between `in_flight_count` decrement and a follow-on `stage()`.

**Code:**

The pattern works because `pending_count` covers the full Reader iteration lifetime:

```cpp
// src/scheduler/decode/decode_staging.hpp
void release(uint32_t uid) {
    pending_count[uid].fetch_sub(1, std::memory_order_acq_rel);
}
```

The claim is taken before the Reader processes a result and released (via RAII destructor) after the Reader has either staged a new entry or decided not to. This closes a race where:

1. Reader decrements `in_flight_count` (both counts momentarily zero).
2. API thread observes zero in_flight and zero pending, finalizes cleanup.
3. Reader stages a new entry for the now-freed slot.

With ReaderClaim, `pending_count` is nonzero throughout steps 1-3, so the API thread's `maybe_finalize_cleanup` check fails at step 2.

**Alternative considered:** Check `cancel_pending` before every staging operation and skip the stage if cancelled.

**Why rejected:** A point check creates a TOCTOU race -- the cancel could arrive between the check and the stage. The RAII pattern provides a continuous hold: as long as the Reader is processing a result for a slot, the slot cannot be finalized. The `pending_count` mechanism is already needed for staging FIFO entries (Decision 14), so extending it to cover Reader iterations adds minimal complexity.

**Cross-reference:** [Chapter 2: Threading and Data Structures](../ch2_threading_and_data_structures/lock_free_design.md) -- ReaderClaim RAII; [Chapter 6](../ch6_lifecycle_is_integration_kv_tiering/deferred_cancellation.md) -- `maybe_finalize_cleanup` protocol.

---

## 13. defer_complete for Speculative Decode

**Decision:** When speculative decode's ACCEPT case detects completion (EOS or max_new_tokens), it does not publish the completion immediately. Instead, it stashes the `OutputMessage` in `SpecDecodeState::pending_complete` and sets `signal_to_exit`. The completion is flushed only after the paired SPEC result has been consumed (CONTINUE or STALE path).

**Code:**

```cpp
// src/scheduler/decode/spec_decode_state.hpp
struct SpecDecodeState {
    std::vector<OutputMessage> pending_complete;
    std::vector<bool> has_pending_complete;
    std::vector<bool> signal_to_exit;
    ...
};
```

The ACCEPT path stashes instead of publishing:

```
// Reader ACCEPT path (pseudocode):
if (completion_detected) {
    signal_to_exit[uid] = true;
    pending_complete[uid] = output_msg;
    has_pending_complete[uid] = true;
    // do NOT publish output_msg to output_queue yet
}
```

The CONTINUE or STALE path flushes:

```
// Reader CONTINUE/STALE path (pseudocode):
if (signal_to_exit[uid] && has_pending_complete[uid]) {
    output_queue.push(pending_complete[uid]);
    has_pending_complete[uid] = false;
    // ... transition to COMPLETE
}
```

**Alternative considered:** Publish the completion immediately on ACCEPT and let the IS ignore the subsequent stale SPEC/STALE result.

**Why rejected:** If the IS sees the completion and immediately issues a CONTINUE for the next turn, the new turn's prefill could begin while the stale SPEC result is still in the pipeline. The stale result would then be misinterpreted as belonging to the new turn. Deferring the completion until the stale result is consumed guarantees that the IS cannot race ahead. The cost is one additional field per slot in SpecDecodeState (a single `OutputMessage` and two booleans).

**Tradeoff:** The completion is delayed by exactly one pipeline round (the time for the stale SPEC/STALE result to arrive). This is typically tens of microseconds -- imperceptible to the end user.

**Cross-reference:** [Chapter 4: Speculative Decode](../ch4_speculative_decode/state_machine_detail.md) -- signal_to_exit and pending_complete; [result_classification.md](../ch4_speculative_decode/result_classification.md) -- CONTINUE-suppress case.

---

## 14. pending_count Dual Reference Counting

**Decision:** The `DecodeStaging::pending_count` per-slot counter serves a dual purpose: it counts both entries in the staging FIFO for that slot AND in-progress Reader iterations for that slot. Combined with `in_flight_count`, it forms the complete "busy count" that `maybe_finalize_cleanup` gates on.

**Code:**

The count is incremented in two places:

1. **Before FIFO push** (in `stage()`):
```cpp
// src/scheduler/decode/decode_staging.hpp
void stage(uint32_t uid, ...) {
    // Claim the slot BEFORE publishing the entry.
    pending_count[uid].fetch_add(1, std::memory_order_acq_rel);
    ...
    if (!fifo.try_push(entry)) {
        pending_count[uid].fetch_sub(1, std::memory_order_acq_rel);
        ...
    }
}
```

Source: `src/scheduler/decode/decode_staging.hpp`, lines 60-79.

2. **At Reader iteration start** (via ReaderClaim RAII):
```
// Reader (pseudocode):
pending_count[uid].fetch_add(1, ...);  // ReaderClaim constructor
// ... process result, possibly stage new entry ...
pending_count[uid].fetch_sub(1, ...);  // ReaderClaim destructor
```

The count is decremented when the Writer pops and processes an entry (`staging.release(uid)`) and when the Reader's RAII claim is destroyed.

`maybe_finalize_cleanup` gates on both counts:

```
// maybe_finalize_cleanup (pseudocode):
if (cancel_pending[uid]
    && in_flight_count[uid].load() == 0
    && pending_count[uid].load() == 0) {
    // CAS state -> INACTIVE; exactly one thread wins
    ...
}
```

**Alternative considered:** Track FIFO entries and Reader iterations with separate counters.

**Why rejected:** Both counters serve the same purpose: preventing premature cleanup of a slot that still has outstanding references. Splitting them would double the atomic operations and make the cleanup gate more complex (checking three conditions instead of two) for no functional benefit. The single counter is simpler and correct: any nonzero value means the slot is still referenced.

The critical race this closes: the Reader decrements `in_flight_count` to zero, then stages a new entry. Without `pending_count`, there is a window after the `in_flight_count` decrement but before the `stage()` call where both counts are zero and cleanup could fire. The `stage()` function increments `pending_count` BEFORE pushing to the FIFO, ensuring the gap never exists.

**Cross-reference:** [Chapter 2: Threading and Data Structures](../ch2_threading_and_data_structures/decode_staging.md) -- pending_count mechanics; [Chapter 2](../ch2_threading_and_data_structures/lock_free_design.md) -- cancel/recycle race prevention.

---

## 15. EOS Write-Back Pattern

**Decision:** When a generation completes, the completing token is staged back to the pipeline with `skip_spec=true`, writing it into the KV cache at the final position. This ensures the next turn's context includes the EOS token. A `post_complete_in_flight` counter tracks the write-back so the Reader discards its loopback result.

**Code:**

The `DecodeStagingEntry` has a `skip_spec` flag for this purpose:

```cpp
// src/scheduler/decode/decode_staging.hpp
struct DecodeStagingEntry {
    ...
    bool skip_spec = false;  // BASE-only write-back
};
```

The UserTable tracks post-completion in-flight tokens:

```cpp
// src/common/user_table.hpp
// Counts BASE-type results from post-completion EOS write-back injects that
// must be discarded before they reach the decode path.
std::vector<std::atomic<uint32_t>> post_complete_in_flight;
```

Source: `src/common/user_table.hpp`, line 29.

The staging call uses `skip_spec=true`:

```
// stage_eos_writeback (pseudocode):
staging.stage(uid, completing_token, actual_token_pos,
              EMPTY_TOKEN, 0, /*skip_spec=*/true);
post_complete_in_flight[uid].fetch_add(1, ...);
```

The Reader discards the write-back result:

```
// Reader (pseudocode):
if (post_complete_in_flight[uid].load() > 0 && result.actual_token_type == BASE) {
    post_complete_in_flight[uid].fetch_sub(1, ...);
    in_flight_count[uid].fetch_sub(1, ...);
    return;  // discard
}
```

**Alternative considered:** Have the IS prepend the EOS token to the next turn's prompt, letting the normal prefill path write it into KV.

**Why rejected:** The PM cannot assume the IS will prepend the EOS. Some IS implementations may use different chat templates that handle turn boundaries differently. By writing the EOS into KV at completion time, the PM guarantees that the KV cache is always in a consistent state for the next turn, regardless of IS behavior. The `current_position` after write-back points past the EOS, so the next CONTINUE starts prefill at exactly the right offset.

**Tradeoff:** The write-back consumes one pipeline injection and produces one result that must be discarded (tracked by `post_complete_in_flight`). This adds one pipeline latency period between the completion of one turn and the readiness for the next.

**Cross-reference:** [Chapter 3: Scheduling Algorithm](../ch3_scheduling_algorithm/completion_and_eos_writeback.md) -- EOS write-back mechanics; [Chapter 6](../ch6_lifecycle_is_integration_kv_tiering/multi_turn_continue.md) -- multi-turn KV reuse.

---

## 16. Keep KV Alive Until IS-Driven Eviction

**Decision:** Completed sessions remain resident in device DRAM until the IS explicitly issues a CANCEL. The PM does not auto-evict based on idle time or memory pressure. The IS controls all eviction policy.

**Code:**

This decision is visible in the state machine: COMPLETE retains the slot and KV. There is no auto-eviction in the PM. The PM provides building blocks for the IS to make eviction decisions:

```cpp
// include/tt_llm_engine/scheduler/decode/decode_types.hpp
enum class UserState : uint8_t {
    ...
    COMPLETE = 3,  // KV retained, slot allocated, no active generation
};
```

The PM exposes diagnostics:

```cpp
// include/tt_llm_engine/scheduler/decode/decode_scheduler.hpp
UserState get_user_state(uint32_t slot_id) const;
uint32_t get_current_position(uint32_t slot_id) const;
```

> **Status: Planned** -- The formal KV tier management API (TierCommand, TierCommandResult, tier_controller_tick) is specified in internal documentation but not yet implemented in source code. The current implementation relies on explicit CANCEL from the IS to free slots.

**Alternative considered:** Auto-evict completed sessions after a configurable idle timeout within the PM.

**Why rejected:** Idle timeout is a business-level policy that depends on factors the PM does not know: the user's session state, their SLA tier, whether they are likely to send another message, and the overall system load. The IS has this context; the PM does not. By keeping KV alive until explicitly told to evict, the PM avoids making policy decisions that belong to the IS layer. The cost is that the IS must actively manage session lifecycle, but this is consistent with the IS/PM ownership boundary established in [Chapter 1](../ch1_architecture_and_layering/ownership_contract.md).

**Cross-reference:** [Chapter 6: Lifecycle, Integration, and KV Tiering](../ch6_lifecycle_is_integration_kv_tiering/kv_cache_tiering.md) -- planned tier management; [Chapter 1](../ch1_architecture_and_layering/ownership_contract.md) -- IS/PM ownership contract.

---

## Cross-Cutting Themes

Several themes recur across these decisions:

### Ownership Clarity

Decisions 2, 4, 9, and 16 all reflect the IS/PM ownership split. The PM handles the "can/how" (scheduling mechanics, KV lifecycle execution) while the IS handles the "why/when" (eviction policy, admission policy, SLA differentiation). This split is the single most important architectural decision in the system.

### Hardware Deployment Path

Decisions 1, 7, 10, and 11 are all shaped by the eventual hardware deployment target. The data structures are designed to map directly to fixed hardware resources (registers, SRAM, DMA FIFOs). The lock-free hot path avoids OS-level synchronization that would not exist in a hardware implementation. The configurable max_users allows the same code to target different pipeline widths.

### Correctness Over Performance

Decisions 12 (ReaderClaim), 13 (defer_complete), 14 (pending_count), and 15 (EOS write-back) prioritize correctness guarantees over performance optimization. Each adds a small amount of state or latency to close a race condition or maintain KV consistency. The system favors provably correct behavior under all thread interleavings over marginal performance gains.

### Simplicity Over Generality

Decisions 8 (no KV rollback) and the mutex-based `BoundedQueue` (used for IS-PM communication rather than a lock-free alternative) choose simpler implementations that are sufficient for the current requirements over more general solutions that would add complexity. The system can be extended later (multi-token speculation, lock-free MPSC) if the need arises.

---

[< Previous: Integration Methods](integration_methods.md)
