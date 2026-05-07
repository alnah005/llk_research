# 08 -- Multi-Host Pipeline Infrastructure: Pipeline Manager Architecture

[<< Previous: Pipeline Builder](01_pipeline_builder.md) | [Next: Scheduling Algorithm >>](03_scheduling_algorithm.md)

---

## 8.12 Overview

The `pipeline_manager/` directory contains a C++20 library that separates workload scheduling from inference server request management. The library sits between an inference server (IS) and the hardware pipeline, providing low-latency interleaved prefill and decode with speculative decode support. All data structures are fixed-size and pre-allocated, with no dynamic allocation on the hot path -- the design is intended for eventual migration to a hardware co-processor.

The library is structured as:

```
pipeline_manager/
  include/pipeline_manager/
    pipeline_manager.hpp          -- PipelineManager public API
    pipeline_interface.hpp        -- Abstract PipelineInterface
    pipeline_manager_types.hpp    -- All types, enums, constants
    wire_format.hpp               -- 64-byte page serialization
    mock_pipeline.hpp             -- MockPipeline for testing
    socket_pipeline.hpp           -- SocketPipeline for hardware
  src/
    pipeline_manager.cpp          -- Implementation (Impl struct)
    socket_pipeline.cpp           -- SocketPipeline implementation
    user_table.hpp                -- UserTable, PromptTable, CancelBitmap
    prefill_queue.hpp             -- PrefillQueue
    decode_staging.hpp            -- DecodeStaging, DecodeStagingEntry
    free_id_pool.hpp              -- FreeIdPool (bitmap allocator)
    bounded_queue.hpp             -- BoundedQueue<T>
    spec_decode_state.hpp         -- SpecDecodeState
```

Everything lives in `namespace tt_blaze::pipeline_manager`.

---

## 8.13 PipelineManager Public API

`PipelineManager` is the only class the inference server interacts with. Its constructor takes a `PipelineConfig` variant (to select the backend) and optional `ManagerParams`. The public interface is narrow:

```cpp
class PipelineManager {
public:
    PipelineManager(PipelineConfig config, ManagerParams params = {});
    ~PipelineManager();

    void start();   // Launch writer, reader, and api threads
    void stop();    // Signal threads and join

    // Queue interface -- all thread-safe
    bool push_request(const ISRequest& request);
    bool try_pop_response(PMResponse& response);
    bool try_pop_output(OutputMessage& output);

    // Diagnostic queries -- safe from any thread
    UserState get_user_state(uint32_t slot_id) const;
    uint32_t get_in_flight_count(uint32_t slot_id) const;
    uint32_t get_tokens_generated(uint32_t slot_id) const;
    uint32_t get_current_position(uint32_t slot_id) const;
    uint32_t get_generation(uint32_t slot_id) const;
    // ...

    void dump_diagnostics(std::ostream& os) const;

    // Access underlying pipeline (for tests)
    template <typename T>
    T& get_pipeline() const;
};
```

All state is hidden behind a `std::unique_ptr<Impl>` pimpl. The IS never touches internal data structures directly; all interaction flows through the three bounded queues.

---

## 8.14 PipelineConfig and ManagerParams

### PipelineConfig

The backend is selected via a `std::variant`:

```cpp
struct MockConfig {
    int latency_min_us = 0;
    int latency_max_us = 0;
    uint32_t seed = 42;
    float accept_rate = 1.0f;   // 1.0 = always accept speculative tokens
};

struct SocketConfig {
    std::string h2d_socket_id;
    std::string d2h_socket_id;
    uint32_t connect_timeout_ms = 30000;
    bool use_deepseek_md_format = false;
};

using PipelineConfig = std::variant<MockConfig, SocketConfig>;
```

The constructor uses `std::visit` to instantiate the appropriate `PipelineInterface` implementation. `SocketConfig` requires the `PM_HAS_METALIUM` compile flag; builds without Metal link against `pipeline_manager_core` which omits `SocketPipeline`.

### ManagerParams

Runtime tuning of the manager:

```cpp
struct ManagerParams {
    uint32_t max_users = 64;         // Size of all per-user arrays
    uint32_t chunk_size = 24;        // Prefill tokens per chunk before rotation
    uint32_t max_seq_len = 131072;   // 128K context window
    uint32_t eos_token = 1;          // Token ID that signals end of sequence

    // CPU affinity for the three threads
    static constexpr int AUTO_CPU = -1;   // auto-select from available set
    static constexpr int NOPIN_CPU = -2;  // disable pinning
    int writer_cpu = AUTO_CPU;
    int reader_cpu = AUTO_CPU;
    int api_cpu = AUTO_CPU;
};
```

`max_users` is not hardcoded to 64 -- it is a deployment parameter. All data structures scale with it. `chunk_size = 24` means the writer injects at most 24 prefill tokens for one user before rotating to the next, preventing a single long prompt from starving decode users.

Thread CPU pinning uses `pthread_setaffinity_np`. When set to `AUTO_CPU`, the manager queries `sched_getaffinity` for available cores and spaces the three threads across them with stride `N/3` to minimize cache-line contention.

---

## 8.15 PipelineInterface

`PipelineInterface` is the abstract boundary between the manager and the actual pipeline hardware. It has five methods:

```cpp
class PipelineInterface {
public:
    virtual void inject(const InjectDescriptor& desc) = 0;
    virtual ResultDescriptor read_result() = 0;
    virtual void reset_kv(uint32_t slot_id) = 0;
    virtual void request_stop() = 0;
    virtual void shutdown() = 0;
};
```

`inject()` is called by the writer thread to push one token into the pipeline. `read_result()` is called by the reader thread and blocks until a result is available. `request_stop()` sets internal flags without touching sockets (threads may still hold references). `shutdown()` runs after all threads are joined and performs final device cleanup.

### MockPipeline

`MockPipeline` is a deterministic testing backend that models speculative decode behavior:

```cpp
class MockPipeline : public PipelineInterface {
    // Token model:
    //   actual_token  = token_id + 1 (accept)  or  token_id + 3 (reject)
    //   predicted_token = actual_token + 1
    //
    // Position model:
    //   actual_token_pos    = inject position + 1
    //   predicted_token_pos = inject position + 2
```

The `accept_rate` parameter controls speculation accuracy: at 1.0 (default) the mock always accepts, at 0.0 it always rejects. This allows unit tests to exercise both the match and mismatch code paths with full determinism.

The mock uses a mutex-guarded queue and condition variable internally. `inject()` immediately computes the result and pushes it to the queue. `read_result()` blocks on the condition variable. Optional latency simulation sleeps for a random duration in `[latency_min_us, latency_max_us]` on each `read_result()`.

Non-last prefill injects (where `prefill_token_id != EMPTY_TOKEN`) produce results that the reader skips via the `prefill_in_flight` counter -- the mock only populates `slot_id` for these.

The mock also records every `InjectDescriptor` in an `inject_log_` vector accessible via `get_inject_log()`, enabling tests to verify the exact sequence of injections.

### SocketPipeline

`SocketPipeline` connects to pre-created H2D/D2H sockets exported by a separate launcher process. Its implementation is straightforward:

```cpp
void SocketPipeline::inject(const InjectDescriptor& desc) {
    impl_->write_buf = serialize_inject(desc, use_deepseek_md_format_);
    impl_->h2d_socket->write(impl_->write_buf.data(), 1);
}

ResultDescriptor SocketPipeline::read_result() {
    while (!impl_->stop_requested.load(std::memory_order_acquire)) {
        if (impl_->d2h_socket->has_data()) {
            impl_->read_buf.fill(0);
            impl_->d2h_socket->read(impl_->read_buf.data(), 1);
            return deserialize_result(impl_->read_buf, use_deepseek_md_format_);
        }
    }
    return ResultDescriptor{.slot_id = INVALID_SLOT};
}
```

Both sockets use 64-byte pages. The `read_result()` spin-polls `has_data()` until data arrives or `request_stop()` is called. `shutdown()` sends a sentinel (`INVALID_SLOT`) through the pipeline and drains until it echoes back, then calls `barrier()` on both sockets.

The `use_deepseek_md_format` flag selects between two wire formats (detailed in section 8.17).

---

## 8.16 Data Structures

### 8.16.1 UserTable

The central per-user register file. All vectors are sized to `max_users` at construction. Parallel arrays indexed by `slot_id`:

```cpp
struct UserTable {
    std::vector<std::atomic<UserState>> state;
    std::vector<std::atomic<uint32_t>> current_position;
    std::vector<uint32_t> max_new_tokens;
    std::vector<uint32_t> tokens_generated;
    std::vector<std::atomic<uint32_t>> in_flight_count;
    std::vector<std::atomic<uint32_t>> prefill_in_flight;
    std::vector<std::atomic<uint32_t>> post_complete_in_flight;
    std::vector<uint32_t> prefill_pos;
    std::vector<uint32_t> prefill_start_pos;
    std::vector<uint32_t> prefill_chunk_remaining;
    std::vector<bool> spec_decode_enabled;
    std::vector<bool> ignore_eos;
    std::vector<float> temperature;
    std::vector<float> top_p;
    std::vector<int32_t> top_k;
    std::vector<uint32_t> spec_accepts;
    std::vector<uint32_t> spec_rejects;
};
```

Thread access pattern:

| Field | API thread | Writer thread | Reader thread |
|---|---|---|---|
| `state` | write (allocate/submit/cancel) | read | read/CAS (completion/cleanup) |
| `in_flight_count` | init to 0 | increment before inject | decrement on result |
| `current_position` | init | write (prefill advance) | write (decode completion) |
| `prefill_in_flight` | -- | increment (non-last prefill) | decrement (skip result) |
| `post_complete_in_flight` | -- | -- | increment/decrement (EOS writeback) |
| `tokens_generated` | init to 0 | -- | increment |
| `temperature/top_p/top_k` | write (submit/continue) | read (forward to pipeline) | -- |

Fields that are only accessed by one thread (like `prefill_pos`, only touched by the writer, and `tokens_generated`, only touched by the reader after initialization) do not need atomics. Fields shared between writer and reader (`in_flight_count`, `prefill_in_flight`) use `std::atomic<uint32_t>` with explicit memory orderings.

The `reset(uid)` method zeros all fields for a slot, called by the API thread on ALLOCATE.

### 8.16.2 PromptTable

Per-turn prompt token storage, decoupled from device-side KV positions:

```cpp
struct PromptTable {
    uint32_t max_users_;
    uint32_t max_seq_len_;
    std::vector<uint32_t> tokens_;      // max_users * max_seq_len flat array
    std::vector<uint32_t> lengths;

    void store(uint32_t uid, const uint32_t* prompt, uint32_t len);
    uint32_t get_token(uint32_t uid, uint32_t idx) const;
    uint32_t get_length(uint32_t uid) const;
    void clear(uint32_t uid);
};
```

Prompts are always stored starting at index 0 within their row. The writer maps prompt indices to device positions using `UserTable::prefill_start_pos`. For multi-turn CONTINUE, the new prompt tokens overwrite from index 0 while `prefill_start_pos` is set to `current_position` so device-side injection continues from where the previous turn left off.

### 8.16.3 DecodeStaging

Handoff buffer from the reader thread to the writer thread for decode loopback tokens:

```cpp
struct DecodeStagingEntry {
    uint32_t slot_id;
    uint32_t token_id;
    uint32_t position;
    uint32_t spec_token_id;     // speculative prediction
    uint32_t spec_position;
    uint32_t generation;        // for stale entry detection after cancel
    bool skip_spec;             // EOS writeback: BASE only, no SPEC pair
};

struct DecodeStaging {
    BoundedQueue<DecodeStagingEntry> fifo;
    std::vector<std::atomic<uint32_t>> generation;
    std::vector<std::atomic<uint32_t>> pending_count;
};
```

The `pending_count` per slot is a reference count covering both staged FIFO entries and in-progress reader iterations. Combined with `in_flight_count`, it forms a busy count that `maybe_finalize_cleanup` gates on -- the slot is only safe to free when both reach zero. This closes the cancel/recycle race where the API thread could otherwise finalize a slot during the gap between the reader staging an entry and the writer picking it up.

The `stage()` method increments `pending_count` *before* pushing to the FIFO. This ensures a concurrent CANCEL cannot observe `(pending=0, in_flight=0)` between staging and the writer's pop:

```cpp
void stage(uint32_t uid, uint32_t tok, uint32_t pos,
           uint32_t spec_tok, uint32_t spec_pos, bool skip_spec = false) {
    pending_count[uid].fetch_add(1, std::memory_order_acq_rel);
    uint32_t gen = generation[uid].load(std::memory_order_relaxed);
    DecodeStagingEntry entry{uid, tok, pos, spec_tok, spec_pos, gen, skip_spec};
    if (!fifo.try_push(entry)) {
        pending_count[uid].fetch_sub(1, std::memory_order_acq_rel);
        // FATAL: token dropped
    }
}
```

### 8.16.4 FreeIdPool

Multi-word bitmap allocator for user slot IDs:

```cpp
struct FreeIdPool {
    static constexpr uint32_t BITS_PER_WORD = 64;
    std::vector<std::atomic<uint64_t>> bitmap;

    uint32_t allocate();   // CTZ scan + CAS
    void free(uint32_t uid);       // atomic OR
    uint32_t count_free() const;   // popcount across all words
};
```

Allocation scans each word for any set bit using `__builtin_ctzll`, then attempts a CAS to clear it. If the CAS fails (another thread allocated the same bit), it retries with the updated word. Free sets the bit via `fetch_or`. For 64 users, this is a single atomic word; for larger deployments, it scales to `ceil(max_users / 64)` words.

### 8.16.5 PrefillQueue

A mutex-guarded deque for users in PREFILL state:

```cpp
struct PrefillQueue {
    std::deque<uint32_t> queue;
    std::mutex mtx;

    void push(uint32_t uid);
    bool try_front(uint32_t& uid);
    void pop_front();
    void rotate();            // move front to back
    void remove(uint32_t uid);
};
```

The mutex is only contended on `push()` (submit/continue from the API thread, which is rare). The writer's hot-path calls `try_front()` which also takes the lock but contention is minimal since the API thread pushes infrequently relative to the writer's per-tick polling rate.

The `rotate()` method implements chunked round-robin: after the writer exhausts a user's prefill chunk, it moves that user to the back of the deque so other users get a turn.

### 8.16.6 BoundedQueue

Generic mutex-based bounded ring buffer used for the three IS-PM queues:

```cpp
template <typename T>
struct BoundedQueue {
    std::vector<T> buffer;
    int capacity_, head, tail, count;
    mutable std::mutex mtx;

    bool try_push(const T& item);
    bool try_pop(T& item);
};
```

No dynamic allocation after construction. Sized at `2 * max_users` for request/response queues and `256 * max_users` for the output queue. The output queue is larger because each user can produce many output tokens between API interactions.

### 8.16.7 CancelBitmap

Multi-word atomic bitmap for tracking pending cancellations:

```cpp
struct CancelBitmap {
    std::vector<std::atomic<uint64_t>> bits;

    void mark(uint32_t uid);     // fetch_or with release
    void clear(uint32_t uid);    // fetch_and with release
    bool is_set(uint32_t uid) const;  // load with acquire
};
```

The release/acquire ordering on `mark`/`is_set` forms a synchronization pair: when any thread observes `is_set(uid) == true`, it is guaranteed to see all stores made by the API thread before the `mark()` call. This is used to safely propagate the `cancel_request_id` without requiring an atomic on the request ID itself.

---

## 8.17 Wire Format

The `wire_format.hpp` header defines the 64-byte page serialization for communication between the host and the device pipeline.

```cpp
static constexpr uint32_t PAGE_SIZE_BYTES = 64;
static constexpr uint32_t PAGE_SIZE_WORDS = 16;
using PageBuffer = std::array<uint32_t, PAGE_SIZE_WORDS>;
```

Two wire formats are supported, selected by the `use_deepseek_md_format` flag:

### Default Format (H2D -- host to device)

```
Word  Field
[0]   slot_id
[1]   token_id
[2]   position
[3]   prefill_token_id  (EMPTY_TOKEN => decode; otherwise non-last prefill)
[4]   spec_flag         (0 or 1)
[5]   temperature       (reinterpret_cast<float>)
[6]   top_p             (reinterpret_cast<float>)
[7]   top_k
[8-15] reserved (zero)
```

### Default Format (D2H -- device to host)

```
Word  Field
[0]   slot_id
[1]   actual_token
[2]   predicted_token
[3-5] reserved
[6]   actual_token_pos
[7-15] reserved
```

### DeepSeek MD Format (H2D)

```
Word  Field
[1]   token_type (0=BASE, 1=SPEC)
[2]   position
[6]   slot_id
[7]   token_id
[8]   position
[9]   prefill_token_id
```

### DeepSeek MD Format (D2H)

```
Word  Field
[0]   actual_token
[1]   actual_token_type
[2]   actual_token_pos
[3]   predicted_token
[4]   predicted_token_type
[5]   predicted_token_pos
[6]   slot_id
```

The DeepSeek MD format carries full type and position information for both actual and predicted tokens, which is needed for the speculative decode verification path. The default format is simpler but lacks per-token type fields.

Serialization and deserialization are inline functions:

```cpp
inline PageBuffer serialize_inject(const InjectDescriptor& desc, bool use_deepseek_md_format);
inline ResultDescriptor deserialize_result(const PageBuffer& page, bool use_deepseek_md_format);
```

Float fields (`temperature`, `top_p`) are stored via `std::memcpy` to avoid strict-aliasing violations.

---

## 8.18 Thread Model

The `PipelineManager::Impl` spawns three threads on `start()`:

```
Pipeline Manager Process
  |
  +-- api_thread      drains request_queue, handles ALLOCATE/SUBMIT/CONTINUE/CANCEL
  |
  +-- writer_thread   hot path: inject tokens into pipeline (decode priority, then prefill)
  |
  +-- reader_thread   hot path: read pipeline results, loopback, spec verify, completion
```

### Thread Interaction

```
                    request_queue
   Inference --------MPSC-------> api_thread
   Server                            |
       ^                        alloc/submit/cancel
       |                             |
       |   response_queue            v
       +-----SPSC<----------  FreeIdPool, UserTable
       |                       PromptTable, PrefillQueue
       |
       |   output_queue              
       +-----SPSC<----------  reader_thread
                                     |
                              DecodeStaging
                                     |
                              writer_thread
                                     |
                              PipelineInterface
                               inject / read_result
```

The writer and reader threads are the hot path. They communicate through `DecodeStaging` (reader stages tokens, writer pops them) and atomics in `UserTable` (shared `in_flight_count`, `state`). No mutex is held on the per-tick hot path.

The API thread handles requests infrequently (one per HTTP request, millisecond timescale) and touches `FreeIdPool`, `UserTable`, `PromptTable`, and `PrefillQueue`. The PrefillQueue mutex is the only shared lock between the API thread and the writer thread.

### Shutdown Sequence

`stop()` proceeds in careful order:

1. Set `running = false` (release store).
2. Call `pipeline_->request_stop()` -- sets internal flags only, no socket I/O.
3. Join `api_thread` (drains remaining requests).
4. Join `writer_thread` (exits when `running` is false).
5. Join `reader_thread` (exits when `read_result()` returns the sentinel).
6. Call `pipeline_->shutdown()` -- safe to use sockets now, exclusive ownership guaranteed. Sends sentinel through the pipeline and drains until it echoes back, then calls barrier on both sockets.

---

## 8.19 Message Types

### ISRequest (IS to PM)

```cpp
struct ISRequest {
    RequestType type;       // ALLOCATE, SUBMIT, CONTINUE, CANCEL
    uint32_t request_id;    // correlation ID echoed in PMResponse
    uint32_t slot_id;       // user slot (set for SUBMIT/CONTINUE/CANCEL)
    std::vector<uint32_t> tokens;
    GenerationParams gen;   // max_new_tokens, spec_decode, temperature, ...
};
```

### PMResponse (PM to IS)

```cpp
struct PMResponse {
    uint32_t request_id;    // echoed from ISRequest
    uint32_t slot_id;       // allocated slot (for ALLOCATE) or confirmed slot (for CANCEL)
    int32_t error_code;     // 0 = success, 1 = no capacity
};
```

### OutputMessage (PM to IS)

```cpp
struct OutputMessage {
    uint32_t slot_id;
    uint32_t token_id;
    bool is_complete;        // EOS, max_new_tokens, or context exhausted
    bool ctx_exhausted;      // context window full
    uint32_t tokens_generated;
    uint32_t generation;     // monotonic counter for stale detection
};
```

The `generation` field is incremented on every CANCEL. When the IS receives an `OutputMessage`, it can check if the generation matches its expected value; stale messages from a cancelled-and-reused slot will have an older generation.

### InjectDescriptor and ResultDescriptor

These are the pipeline-side descriptors exchanged between the manager and `PipelineInterface`:

```cpp
struct InjectDescriptor {
    uint32_t slot_id;
    uint32_t token_id;
    uint32_t prefill_token_id;   // EMPTY_TOKEN for decode
    uint32_t position;
    TokenType token_type;        // BASE or SPEC
    bool spec_flag;
    float temperature, top_p;
    int32_t top_k;
};

struct ResultDescriptor {
    uint32_t slot_id;
    uint32_t actual_token;
    TokenType actual_token_type;
    uint32_t actual_token_pos;
    uint32_t predicted_token;
    TokenType predicted_token_type;
    uint32_t predicted_token_pos;
};
```

---

## 8.20 Backpressure Semantics

| Queue | Capacity | If full |
|---|---|---|
| `request_queue` | `2 * max_users` | IS gets `false` from `push_request()`, returns HTTP 503 |
| `response_queue` | `2 * max_users` | PM spins with `_mm_pause()` (allocation is rare) |
| `output_queue` | `256 * max_users` | PM spins with `_mm_pause()` (IS should drain promptly) |

The `_mm_pause()` intrinsic hints the CPU to reduce power and avoid pipeline stalls during spin-waits. These spins are transient: the response queue drains within microseconds as the IS polls it, and the output queue only fills under severe IS lag.

---

[<< Previous: Pipeline Builder](01_pipeline_builder.md) | [Next: Scheduling Algorithm >>](03_scheduling_algorithm.md)
