# PrefillQueue and BoundedQueue

The scheduler uses two distinct queue types: `PrefillQueue` (a mutex-protected deque for chunked round-robin prefill scheduling) and `BoundedQueue<T>` (a mutex-protected ring buffer for inter-component communication). Neither is lock-free, which is a deliberate choice explained at the end of this section.

## PrefillQueue

`PrefillQueue` is a bounded session queue for users in PREFILL state. It orders the prefill work that the Writer thread processes when no decode tokens are waiting. Defined in `src/scheduler/decode/prefill_queue.hpp`.

### Structure

```cpp
struct PrefillQueue {
    std::deque<uint32_t> queue;
    std::mutex mtx;
};
```

A `deque` of slot IDs protected by a mutex. The deque (rather than a ring buffer) is used because the `remove()` operation requires arbitrary-position erasure, which is not efficiently supported by a fixed ring buffer.

### Operations

**push(uid)**: Append a slot ID to the back of the queue.

```cpp
void push(uint32_t uid) {
    std::lock_guard<std::mutex> lock(mtx);
    queue.push_back(uid);
}
```

Called by the API thread during SUBMIT and CONTINUE processing. This is an infrequent operation (once per user per turn), so mutex contention is negligible.

**try_front(uid)**: Peek at the front element without removing it.

```cpp
bool try_front(uint32_t& uid) {
    std::lock_guard<std::mutex> lock(mtx);
    if (queue.empty()) return false;
    uid = queue.front();
    return true;
}
```

Called by the Writer thread. If a prefill candidate exists, the Writer reads the slot's state to decide whether to inject, rotate, or pop. Does NOT remove the element -- the Writer may need to rotate or continue injecting tokens for the same user across multiple loop iterations.

**pop_front()**: Remove the front element.

```cpp
void pop_front() {
    std::lock_guard<std::mutex> lock(mtx);
    if (!queue.empty()) queue.pop_front();
}
```

Called by the Writer thread after the last prefill token is injected, or when the front slot is cancelled.

**rotate()**: Move the front element to the back for chunked round-robin scheduling.

```cpp
void rotate() {
    std::lock_guard<std::mutex> lock(mtx);
    if (queue.size() > 1) {
        uint32_t front = queue.front();
        queue.pop_front();
        queue.push_back(front);
    }
}
```

Called by the Writer thread when a user's chunk is exhausted (`prefill_chunk_remaining == 0`). This is the core of the interleaved prefill scheduling described in Chapter 1: inject `chunk_size` tokens for one user, rotate, inject for the next, etc. This prevents a single long prompt from starving other users.

**remove(uid)**: Remove a specific slot from anywhere in the queue.

```cpp
void remove(uint32_t uid) {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = std::remove(queue.begin(), queue.end(), uid);
    queue.erase(it, queue.end());
}
```

Called by the API thread during CANCEL processing. Uses `std::remove` + `erase` to handle the case where the slot appears anywhere in the queue (or not at all). This is O(n) in the queue size, but the queue is bounded by `max_users` and cancellation is rare relative to the per-tick scheduling frequency.

### Thread Access Pattern

| Thread | Operations |
|--------|-----------|
| API handler | `push()`, `remove()` |
| Writer | `try_front()`, `pop_front()`, `rotate()` |
| Reader | None |

### Contention Analysis

The mutex is contended only when the API thread and Writer thread both attempt to access the queue simultaneously. The API thread touches the queue on SUBMIT, CONTINUE, and CANCEL -- infrequent, per-session events at millisecond timescales. The Writer thread touches it on every prefill iteration, but only when there are active prefill users and no decode tokens waiting (decode has priority).

In steady-state decode (all users in DECODE state, decode staging non-empty), the Writer never touches PrefillQueue. The mutex is effectively uncontended on the hot path.

## BoundedQueue<T>

`BoundedQueue<T>` is a generic mutex-protected ring buffer used for all inter-component communication queues. Defined in `src/common/bounded_queue.hpp`.

### Structure

```cpp
template <typename T>
struct BoundedQueue {
    std::vector<T> buffer;
    int capacity_ = 0;
    int head = 0;       // next write position
    int tail = 0;       // next read position
    int count = 0;      // current occupancy
    mutable std::mutex mtx;
};
```

The buffer is a pre-allocated vector of `T` elements. `head` is the write position, `tail` is the read position, and `count` tracks occupancy. All operations are protected by the same mutex.

### Operations

**try_push(item)**: Non-blocking enqueue. Returns false if the queue is full.

```cpp
bool try_push(const T& item) {
    std::lock_guard<std::mutex> lock(mtx);
    if (count >= capacity_) return false;
    buffer[head] = item;
    head = (head + 1) % capacity_;
    ++count;
    return true;
}
```

**try_pop(item)**: Non-blocking dequeue. Returns false if the queue is empty.

```cpp
bool try_pop(T& item) {
    std::lock_guard<std::mutex> lock(mtx);
    if (count == 0) return false;
    item = buffer[tail];
    tail = (tail + 1) % capacity_;
    --count;
    return true;
}
```

Both operations copy the element (not move), which is appropriate because all message types (`ISRequest`, `SchedulerResponse`, `OutputMessage`, `DecodeStagingEntry`) are small, fixed-size structs.

### Queue Instances and Capacities

The `DecodeScheduler::Impl` constructor creates four `BoundedQueue` instances with specific capacities:

| Queue | Type Parameter | Capacity | Producer | Consumer |
|---|---|---|---|---|
| `request_queue` | `ISRequest` | `2 * max_users` | IS threads (external) | API thread |
| `response_queue` | `SchedulerResponse` | `2 * max_users` | API thread, cleanup winner | IS threads (external) |
| `output_queue` | `OutputMessage` | `256 * max_users` | Reader thread, Writer (rare) | IS output loop |
| `decode_staging.fifo` | `DecodeStagingEntry` | `4 * max_users` | Reader thread (stage), API thread (disaggregated continue) | Writer thread (try_pop) |

With `DEFAULT_MAX_USERS = 64`:

| Queue | Capacity |
|---|---|
| `request_queue` | 128 |
| `response_queue` | 128 |
| `output_queue` | 16,384 |
| `decode_staging.fifo` | 256 |

### Capacity Rationale

**request_queue (2 * max_users):** At most one request per slot can be in-flight at a time (the IS awaits a response before sending the next request for the same slot). The 2x multiplier provides headroom for concurrent requests across different slots.

**response_queue (2 * max_users):** One response per request. The 2x multiplier matches the request queue.

**output_queue (256 * max_users):** This is the largest queue because output tokens arrive at pipeline speed (microseconds per token) while the IS drains them at application speed (potentially slower). Each user can generate one token per pipeline round-trip (typically every ~2.8ms for a 64-stage pipeline at 44us per stage). The 256x factor provides buffering for roughly 256 tokens per user, tolerating an IS output drain stall of up to ~700ms without dropping tokens. This absorbs IS-side latency spikes from HTTP processing, JSON serialization, and WebSocket transmission.

**decode_staging.fifo (4 * max_users):** Multiple entries per slot can be in-flight simultaneously: a decode loopback entry, an EOS write-back entry, and a Reader scratch claim entry. The 4x multiplier ensures the FIFO never overflows in normal operation.

### Backpressure Pattern

When a `try_push()` fails (queue full), the caller spins with `_mm_pause()`:

```cpp
while (!output_queue.try_push(msg)) {
    _mm_pause();
}
```

This pattern appears in the Reader thread for output messages and in the API thread for response messages. The `_mm_pause()` intrinsic (Intel PAUSE instruction) yields the CPU pipeline for a few cycles per iteration without an OS context switch, keeping latency low during transient fullness. The spin is bounded because:

- `response_queue` is drained by the IS, which is always running.
- `output_queue` is drained by the IS output loop.
- The decode_staging FIFO is drained by the Writer, which runs continuously.

## Why Lock-Free SPSC Was Not Used

Given that several of these queues are single-producer/single-consumer (SPSC), a natural question is why they use a mutex-based implementation rather than a lock-free SPSC ring buffer. Five factors justify the current design:

1. **MPSC access pattern.** The `request_queue` is MPSC (multiple IS HTTP handler threads can push concurrently). The `output_queue` has multiple producers (both the Reader thread and, for prefill-completion messages, the Writer thread). A lock-free SPSC queue would not be correct for these patterns without external serialization.

2. **The PrefillQueue needs mid-queue removal.** `std::deque` with a mutex naturally supports `rotate()` and `remove()`. A lock-free queue would need a fundamentally different data structure to support these operations.

3. **Minimal contention on hot path.** The performance-critical Writer and Reader threads access `decode_staging.fifo` and `output_queue`. The mutex is uncontended in the common case (one thread pushes, the other pops, they rarely collide on the same microsecond). An uncontended `std::mutex` on Linux (using futex) is essentially a single atomic CAS -- comparable to lock-free. Each critical section is a single cache-line copy + counter update -- under 100 nanoseconds.

4. **Hardware portability.** The `BoundedQueue`'s ring buffer with head/tail/count maps trivially to a hardware FIFO or DMA mailbox. Lock-free SPSC algorithms rely on specific memory model guarantees that may not translate cleanly to a hardware co-processor's memory model. A simple ring buffer with explicit synchronization primitives (whatever the hardware provides) is more portable.

5. **Simplicity and correctness.** Lock-free data structures are notoriously difficult to verify. The `BoundedQueue` is straightforward, obviously correct, and easy to reason about during code review. For a production scheduler where correctness is paramount, this tradeoff favors simplicity.

The hot-path mutex avoidance strategy focuses on the data structures where contention would matter: `UserTable` fields (atomics), `FreeIdPool` (CAS), `CancelBitmap` (atomic bitmap), and `pending_count` (atomic counter). The BoundedQueue mutex is reserved for the non-critical communication paths where a few hundred nanoseconds of overhead is negligible compared to the pipeline tick period.

---

**Previous:** [Decode Staging](./decode_staging.md) | **Next:** [SpecDecodeState](./spec_decode_state.md)
