# PromptTable and CancelBitmap

These two structures serve distinct purposes but share the same design philosophy: fixed-size, pre-allocated, indexed by slot ID, with carefully bounded access patterns that avoid cross-thread contention. Both are defined in `src/common/user_table.hpp`.

## PromptTable

### Structure

```cpp
struct PromptTable {
    uint32_t max_users_;
    uint32_t max_seq_len_;
    std::vector<uint32_t> tokens_;   // flattened 2D: max_users * max_seq_len
    std::vector<uint32_t> lengths;   // per-slot prompt length
};
```

The token storage is a single contiguous vector of size `max_users * max_seq_len`. Row `uid` starts at offset `uid * max_seq_len`. With the defaults (`max_users=64`, `max_seq_len=131072`), this pre-allocates `64 * 131072 * 4 = 32 MB` -- a fixed cost paid once at construction.

### store()

```cpp
void store(uint32_t uid, const uint32_t* prompt, uint32_t len) {
    uint32_t clamped = std::min(len, max_seq_len_);
    size_t base = static_cast<size_t>(uid) * max_seq_len_;
    for (uint32_t i = 0; i < clamped; i++) {
        tokens_[base + i] = prompt[i];
    }
    lengths[uid] = clamped;
}
```

Called by the API thread during SUBMIT and CONTINUE processing. The prompt is copied in full; lengths are clamped to `max_seq_len`. The element-by-element copy avoids any memcpy-related UB concerns and produces straightforward auto-vectorization.

### get_token() and get_length()

```cpp
uint32_t get_token(uint32_t uid, uint32_t idx) const {
    return tokens_[static_cast<size_t>(uid) * max_seq_len_ + idx];
}

uint32_t get_length(uint32_t uid) const { return lengths[uid]; }
```

Called by the Writer thread during prefill. No bounds checking -- the caller is responsible for ensuring `idx < get_length(uid)`.

### clear()

```cpp
void clear(uint32_t uid) { lengths[uid] = 0; }
```

Called by the Writer thread after all prefill tokens have been injected, and in the context-exhaustion path. Does not zero the token data; simply sets the length to 0, which makes the row logically empty. The actual token data in the buffer becomes stale and will be overwritten on the next `store()`.

### prefill_start_pos Decoupling

Prompts are always stored starting at index 0 in the PromptTable, regardless of the user's current device position. The mapping from prompt index to device position is handled by `UserTable::prefill_start_pos`, which bridges the gap for multi-turn continuation. See [UserTable -- prefill_start_pos](./user_table.md#prefill_start_pos-decoupling-prompt-index-from-device-position) for the full explanation and worked examples.

### Thread Safety

PromptTable has no atomics because no two threads ever access the same row concurrently:

1. The API thread writes a row during SUBMIT or CONTINUE processing, before the user enters the PrefillQueue.
2. The Writer thread reads the row after popping the user from PrefillQueue. The PrefillQueue push/pop (which goes through a mutex) establishes the happens-before relationship.
3. The Writer calls `clear()` when prefill completes.
4. No new write to the row can occur until the next SUBMIT or CONTINUE, which can only happen after the current generation completes (COMPLETE state) or the slot is cancelled and reallocated.

The API thread never writes a row while the Writer is reading it because the slot transitions (INACTIVE -> PREFILL via the API, and PREFILL -> DECODE via the Writer) enforce a happens-before ordering mediated by the `state` atomic and the PrefillQueue push/pop.

## CancelBitmap

`CancelBitmap` tracks which slots have pending cancellation. It is a multi-word atomic bitmap, structurally identical to `FreeIdPool`'s bitmap but with inverted semantics (set bit = cancel pending, cleared bit = no cancel).

### Structure

```cpp
struct CancelBitmap {
    static constexpr uint32_t BITS_PER_WORD = 64;

    uint32_t num_words_;
    std::vector<std::atomic<uint64_t>> bits;
};
```

Initialized with all bits cleared (no cancellations pending):

```cpp
explicit CancelBitmap(uint32_t max_users = DEFAULT_MAX_USERS)
    : num_words_((max_users + BITS_PER_WORD - 1) / BITS_PER_WORD),
      bits(num_words_) {
    for (uint32_t w = 0; w < num_words_; w++) {
        bits[w].store(0, std::memory_order_relaxed);
    }
}
```

### Why a Bitmap Instead of Per-Slot Atomics

A bitmap consolidates 64 cancel flags into a single cache line. The Writer thread checks cancel status on every decode staging entry and every prefill peek -- this is the hottest read path. Having all flags in one (or a few) cache lines means the check is almost always an L1 hit, compared to per-slot `atomic<bool>` values which would scatter across multiple cache lines.

### mark()

```cpp
void mark(uint32_t uid) {
    uint32_t w = uid / BITS_PER_WORD;
    uint32_t bit = uid % BITS_PER_WORD;
    bits[w].fetch_or(uint64_t(1) << bit, std::memory_order_release);
}
```

Called by the API thread when processing a CANCEL request. The `release` ordering is critical -- it ensures that the `cancel_request_id[uid]` store (which the API thread performs immediately before calling `mark()`) is visible to any thread that subsequently observes the bit as set via an acquire-load in `is_set()`.

### clear()

```cpp
void clear(uint32_t uid) {
    uint32_t w = uid / BITS_PER_WORD;
    uint32_t bit = uid % BITS_PER_WORD;
    bits[w].fetch_and(~(uint64_t(1) << bit), std::memory_order_release);
}
```

Called in two places:

1. **API thread during ALLOCATE**: `cancel_pending.clear(uid)` is called after `user_table.reset(uid)` and before the slot becomes visible to any other thread. This clears any stale cancel bit from a previous lifecycle.
2. **API thread during CANCEL of INACTIVE slot**: Direct synchronous cleanup path.

Important: `clear()` is NOT called during `maybe_finalize_cleanup()`. In the asynchronous cleanup path, the cancel bit remains set after the slot transitions to INACTIVE and is freed back to `FreeIdPool`. The bit is only cleared on the next ALLOCATE for the slot. This is safe because the slot is in the free pool and no thread accesses it between `free_ids.free(uid)` and the next `free_ids.allocate()`.

The `release` ordering ensures that subsequent operations on the slot (after re-allocation) do not observe stale cancel state.

### is_set()

```cpp
bool is_set(uint32_t uid) const {
    uint32_t w = uid / BITS_PER_WORD;
    uint32_t bit = uid % BITS_PER_WORD;
    return (bits[w].load(std::memory_order_acquire) >> bit) & 1;
}
```

Called by all three threads:

- **Writer thread**: checks before injecting decode or prefill tokens. If set, the entry is dropped and `maybe_finalize_cleanup()` is called.
- **Reader thread**: checks on every received result. If set, the result is discarded and cleanup is attempted.
- **API thread**: checked in `maybe_finalize_cleanup()` to determine if cleanup is needed.

The `acquire` ordering pairs with the `release` on `mark()`, establishing the happens-before relationship that makes `cancel_request_id[uid]` visible to the thread reading the cancel bit.

### Memory Ordering Discipline

The CancelBitmap's release-acquire pairs serve as the synchronization backbone for the entire cancellation protocol. The ordering chain is:

```
API thread:
  cancel_request_id[uid] = req.request_id;   // plain store
  cancel_pending.mark(uid);                    // release store on bitmap

Any thread (via is_set):
  if (cancel_pending.is_set(uid)) {            // acquire load on bitmap
      // cancel_request_id[uid] is now visible   // plain load is safe
  }
```

This avoids making `cancel_request_id` atomic while still guaranteeing visibility. The bitmap acts as a publication flag with release-acquire semantics. The protocol is documented explicitly in the code:

```cpp
// Per-slot CANCEL request_id ... The release-acquire pair on
// CancelBitmap is what makes this plain-store / plain-load safe;
// no other reader is allowed to touch this slot.
std::vector<uint32_t> cancel_request_id;
```

### Thread Access Summary

| Operation | Thread | Ordering | Purpose |
|-----------|--------|----------|---------|
| `mark(uid)` | API handler | release | Signal cancellation |
| `clear(uid)` | API handler (ALLOCATE, synchronous CANCEL cleanup) | release | Clear on reallocation or synchronous cleanup |
| `is_set(uid)` | Writer | acquire | Skip cancelled users before inject |
| `is_set(uid)` | Reader | acquire | Skip cancelled users on result, trigger cleanup |
| `is_set(uid)` | Any (via `maybe_finalize_cleanup`) | acquire | Gate cleanup on cancellation status |

### Relationship to FreeIdPool

Both structures use the same multi-word atomic bitmap pattern, but their semantics are inverted:

| Aspect | FreeIdPool | CancelBitmap |
|--------|-----------|-------------|
| Bit meaning | 1 = free, 0 = allocated | 1 = cancel pending, 0 = normal |
| Set operation | `free()` (return to pool) | `mark()` (request cancellation) |
| Clear operation | `allocate()` (claim slot) | `clear()` (ack/reset cancellation) |
| Allocation pattern | CAS (claim exactly one) | fetch_or (fire-and-forget) |

---

**Previous:** [UserTable](./user_table.md) | **Next:** [Decode Staging](./decode_staging.md)
