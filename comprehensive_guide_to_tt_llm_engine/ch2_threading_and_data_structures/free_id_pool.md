# FreeIdPool

`FreeIdPool` is the bitmap-based user ID allocator. It maps the abstract operation "give me a free slot" to a lock-free bit scan and compare-and-swap, and the reverse operation "return this slot" to an atomic OR. The structure lives in `src/common/free_id_pool.hpp`.

## Structure

```cpp
struct FreeIdPool {
    static constexpr uint32_t BITS_PER_WORD = 64;

    uint32_t max_users_;
    uint32_t num_words_;
    std::vector<std::atomic<uint64_t>> bitmap;
};
```

The bitmap is a vector of `atomic<uint64_t>` words, with `num_words_ = ceil(max_users / 64)`. A set bit means the corresponding slot is **free**; a cleared bit means it is **allocated**.

### Initialization

```cpp
explicit FreeIdPool(uint32_t max_users = DEFAULT_MAX_USERS)
    : max_users_(max_users),
      num_words_((max_users + BITS_PER_WORD - 1) / BITS_PER_WORD),
      bitmap(num_words_) {
    for (uint32_t w = 0; w < num_words_; w++) {
        uint32_t base = w * BITS_PER_WORD;
        uint32_t bits_in_word = (base + BITS_PER_WORD <= max_users_)
                                    ? BITS_PER_WORD
                                    : (max_users_ - base);
        uint64_t mask = (bits_in_word == BITS_PER_WORD) ? ~uint64_t(0)
                                                        : (uint64_t(1) << bits_in_word) - 1;
        bitmap[w].store(mask, std::memory_order_relaxed);
    }
}
```

For `max_users = 64` (the default), this is exactly one word with all 64 bits set (`0xFFFFFFFFFFFFFFFF`). For `max_users = 100`, two words: word 0 has all 64 bits set, word 1 has bits 0..35 set and bits 36..63 clear.

## allocate(): CAS-based Slot Acquisition

```cpp
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
            // CAS failed: 'current' is reloaded by compare_exchange_weak
        }
    }
    return INVALID_SLOT;
}
```

### Step-by-step

1. **Word scan**: Iterate through words looking for one with at least one set bit.
2. **Find lowest free bit**: `__builtin_ctzll(current)` returns the index of the lowest set bit (count trailing zeros). This is a single hardware instruction on x86 (`TZCNT` or `BSF`) and ARM (`RBIT` + `CLZ`).
3. **CAS claim**: Attempt to atomically clear the bit. `compare_exchange_weak` compares the current word value against `current`; if unchanged, it writes `current & ~mask` (clears the bit). On success, the slot is claimed. On failure, `current` is reloaded with the new value and the inner loop retries.
4. **Return slot ID**: `w * 64 + bit` maps the word index and bit position to a slot ID.

### Memory Ordering

- **Success**: `acq_rel`. The acquire side ensures that any state previously released by a `free()` on this slot (e.g., `UserTable::reset()` clearing the slot's fields) is visible to the allocating thread. The release side ensures the allocation is visible to other threads that might observe the slot's state.
- **Failure**: `relaxed`. On CAS failure, we only need the reloaded value; no ordering constraints are required.

### Complexity

Allocation is O(max_users/64) in the worst case (scanning all words). For the default 64 users (1 word), it is O(1). The CTZ instruction is constant-time. In practice, the first word almost always has a free bit, so allocation is effectively O(1).

### Thread Safety

Only the API thread calls `allocate()`. However, `free()` can be called concurrently by any of the three threads -- the Writer thread (cleanup winner, rare), the Reader thread (cleanup winner), or the API thread itself (synchronous cleanup of INACTIVE slot on cancel, or cleanup winner). The CAS ensures correctness even under concurrent modifications to the same word.

## free(): Atomic OR

```cpp
void free(uint32_t uid) {
    uint32_t w = uid / BITS_PER_WORD;
    uint32_t bit = uid % BITS_PER_WORD;
    bitmap[w].fetch_or(uint64_t(1) << bit, std::memory_order_release);
}
```

Setting a bit is unconditional -- `fetch_or` atomically sets the bit regardless of its current value. No CAS loop is needed because setting a bit from 0 to 1 is idempotent. The `release` ordering ensures that all cleanup work performed before `free()` (resetting `UserTable`, clearing `CancelBitmap`, resetting `SpecDecodeState`, calling `pipeline_->reset_kv()`) is visible to any thread that subsequently observes the bit as set and calls `allocate()`.

### Who Calls free()?

`free()` is called from `maybe_finalize_cleanup()`, which can run on any of the three threads:

- **Writer thread**: When it pops a staging entry for a cancelled slot and finds `pending_count` drops to zero.
- **Reader thread**: When it processes a result for a cancelled slot and finds both `in_flight_count` and `pending_count` at zero.
- **API thread**: When processing a CANCEL for a slot that is INACTIVE with zero in-flight and zero pending.

Exactly one thread wins the cleanup CAS on `user_table.state` (see [Lock-Free Design](./lock_free_design.md)), so `free()` is called exactly once per cancel cycle.

## count_free()

```cpp
uint32_t count_free() const {
    uint32_t total = 0;
    for (uint32_t w = 0; w < num_words_; w++) {
        total += static_cast<uint32_t>(
            __builtin_popcountll(bitmap[w].load(std::memory_order_relaxed)));
    }
    return total;
}
```

Population count across all words using `POPCNT`. Uses `relaxed` ordering because this is a diagnostic query -- the exact count may be slightly stale. Used by tests and monitoring, not by scheduling logic.

## Lifetime Interaction with Cancel

A slot is only freed after `maybe_finalize_cleanup` verifies all reference counts are zero, wins the cleanup CAS, and completes device-side reset. See [Lock-Free Design -- CAS in maybe_finalize_cleanup](./lock_free_design.md#cas-in-maybe_finalize_cleanup) for the full gate sequence. This ordering prevents a new ALLOCATE from reusing a slot while stale references still exist in the pipeline or staging structures.

## Hardware Deployment Path

With `max_users = 64`, the entire FreeIdPool is a single 64-bit register. `allocate()` maps to a CTZ + atomic bit clear, `free()` to an atomic bit set, and `count_free()` to POPCNT. For larger deployments (`max_users > 64`), the multi-word scan adds a linear scan over `ceil(max_users/64)` registers -- trivially small even at `max_users = 256`. The sequential word scan in `allocate()` maps naturally to a priority encoder in hardware. See the [Hardware Deployment Path Mapping](./index.md#hardware-deployment-path-mapping) table in the chapter overview for the full software-to-hardware correspondence.

---

**Previous:** [CPU Affinity](./cpu_affinity.md) | **Next:** [UserTable](./user_table.md)
