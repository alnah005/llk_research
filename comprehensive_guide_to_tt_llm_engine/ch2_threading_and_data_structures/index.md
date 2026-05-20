# Chapter 2: Threading and Data Structures

The decode scheduler achieves microsecond-scale scheduling through a deliberately constrained design: three dedicated threads, fixed-size data structures pre-allocated at startup, and lock-free communication on the hot path. No dynamic memory allocation occurs after construction. Every data structure is sized at construction time from `SchedulerParams` and indexed by `slot_id`, enabling a direct mapping to the hardware deployment path described in Chapter 1.

This chapter covers the concrete implementation: which threads exist, what each thread touches, how concurrency is managed without mutexes on the hot path, and the precise layout and semantics of every shared data structure.

## Thread Topology

```
                          Inference Server
                               |
                  push_request(ISRequest)
                               |
                               v
              +=====================================+
              |        DecodeScheduler::Impl        |
              |         (central state holder)       |
              |                                      |
              |  +-------------+  +-------------+   |
              |  | request_q   |  | response_q  |   |
              |  | (IS -> PM)  |  | (PM -> IS)  |   |
              |  +------+------+  +------^------+   |
              |         |                |           |
              |         v                |           |
              |  +------+---------------+------+    |
              |  |        API Thread            |    |
              |  | _mm_pause() spin-wait loop   |    |
              |  | drains request_queue         |    |
              |  | ALLOCATE/SUBMIT/CONTINUE/    |    |
              |  | CANCEL processing            |    |
              |  +--+------+-------+-----------+    |
              |     |      |       |                 |
              |     v      v       v                 |
              |  FreeIdPool  UserTable  PrefillQueue  |
              |  PromptTable CancelBitmap             |
              |     |               |                 |
              |     v               v                 |
              |  +--------+   +----------+            |
              |  | Writer |   |  Reader  |            |
              |  | Thread |   |  Thread  |            |
              |  +---+----+   +----+-----+            |
              |      |             |                   |
              +======|=============|===================+
                     |             |
            inject() |             | read_result()
                     v             ^
              +========================+
              |      Pipeline          |
              | (Mock/Simulator/Socket)|
              +========================+
                                      |
                                      v
                              +-------------+
                              | output_q    |
                              | (PM -> IS)  |
                              +------+------+
                                     |
                                     v
                           Inference Server
                          (output loop thread)
```

The Writer and Reader form a closed loop through the hardware pipeline. The API thread is the sole entry point for external requests. All three threads are pinned to distinct CPU cores via `pthread_setaffinity_np`.

## Data Structure Map

Every data structure is pre-allocated and sized at construction. The table below shows where each structure fits in the pipeline from external request to device injection.

| Structure | Purpose | Sizing | Hot-Path Alloc |
|-----------|---------|--------|----------------|
| `FreeIdPool` | Slot allocation/deallocation | `ceil(max_users / 64)` atomic words | None |
| `UserTable` | Per-slot state registers | `max_users` per field (~20 fields) | None |
| `PromptTable` | Prompt token storage | `max_users * max_seq_len` flat array | None |
| `CancelBitmap` | Pending cancellation flags | `ceil(max_users / 64)` atomic words | None |
| `DecodeStaging` | Reader-to-Writer token handoff | `max_users * 4` entry FIFO + per-slot atomics | None |
| `PrefillQueue` | Users awaiting prefill | `std::deque` with mutex | Rare (SUBMIT/CONTINUE) |
| `SpecDecodeState` | Per-slot speculation tracking | `max_users` per field (6 fields) | None |
| `BoundedQueue<ISRequest>` | IS-to-PM request channel | `2 * max_users` entries | None |
| `BoundedQueue<SchedulerResponse>` | PM-to-IS allocation replies | `2 * max_users` entries | None |
| `BoundedQueue<OutputMessage>` | PM-to-IS token output | `256 * max_users` entries | None |

## Data Structure to Thread Mapping

For the complete thread-by-thread access matrix with memory ordering annotations, see the [Thread Interaction Matrix](./threading_model.md#thread-interaction-matrix) in the Threading Model section.

## Constants

These constants from `decode_types.hpp` and `pipeline_types.hpp` size the entire system:

| Constant | Value | Definition | Purpose |
|----------|-------|------------|---------|
| `DEFAULT_MAX_USERS` | 64 | `decode_types.hpp` | Default user slot count, sizes all per-user arrays |
| `DEFAULT_MAX_SEQ_LEN` | 131072 (128K) | `decode_types.hpp` | Maximum sequence length per user |
| `DEFAULT_CHUNK_SIZE` | 24 | `decode_types.hpp` | Prefill tokens per chunk before round-robin rotation |
| `DEFAULT_EOS_TOKEN` | 1 | `decode_types.hpp` | Default end-of-sequence token ID |
| `INVALID_SLOT` | `uint32_t max` (0xFFFFFFFF) | `pipeline_types.hpp` | Sentinel for "no slot allocated" |
| `EMPTY_TOKEN` | `uint32_t max` (0xFFFFFFFF) | `pipeline_types.hpp` | Sentinel for "no token present" |

`SchedulerParams` allows runtime override of `max_users`, `chunk_size`, `max_seq_len`, and `eos_token`. The constants provide defaults; actual sizes are set at `DecodeScheduler` construction time. `INVALID_SLOT` and `EMPTY_TOKEN` are defined as `std::numeric_limits<uint32_t>::max()` in `pipeline_types.hpp` and re-exported into the `scheduler::decode` namespace by `decode_types.hpp`.

## Queue Capacities

The three inter-component queues (`request_queue`, `response_queue`, `output_queue`) are sized as multiples of `max_users`. See [PrefillQueue and BoundedQueue](./prefill_queue_and_bounded_queue.md) for capacity formulas, computed values, and backpressure analysis.

## Hardware Deployment Path Mapping

Every structure in this chapter maps to a hardware primitive. The code is written so that the C++ scheduler can be replaced by co-processor firmware with only a transport-layer change.

| Software Structure | Hardware Equivalent | Why |
|-------------------|-------------------|-----|
| `FreeIdPool` bitmap | 64-bit register + CTZ instruction | Single-cycle alloc/free |
| `UserTable` parallel arrays | Register file / SRAM banks | Per-field atomicity without locking |
| `PromptTable` flat token array | DRAM-backed token store | Bulk storage, sequential access |
| `CancelBitmap` atomic words | Single register with bitwise ops | O(1) set/test/clear |
| `DecodeStaging` FIFO + pending_count | Hardware FIFO with ref-count tag | Bounded handoff with generation tracking |
| `SpecDecodeState` arrays | SRAM arrays | Reader-private, no cross-thread sync |
| `BoundedQueue<T>` (IS queues) | DMA FIFO / hardware mailbox | Fixed-size message passing |

## Reading Roadmap

| File | Content |
|------|---------|
| [`threading_model.md`](./threading_model.md) | Three-thread architecture, thread interaction matrix, lifecycle |
| [`cpu_affinity.md`](./cpu_affinity.md) | CPU pinning, AUTO_CPU resolution, production tuning |
| [`free_id_pool.md`](./free_id_pool.md) | Multi-word bitmap allocator, CAS allocation, atomic OR free |
| [`user_table.md`](./user_table.md) | Per-slot register file, field catalog with thread ownership, UserState FSM |
| [`prompt_table_and_cancel_bitmap.md`](./prompt_table_and_cancel_bitmap.md) | Prompt storage, cancel tracking bitmaps, memory orderings |
| [`decode_staging.md`](./decode_staging.md) | Reader-to-Writer handoff, pending_count reference counting, ReaderClaim |
| [`prefill_queue_and_bounded_queue.md`](./prefill_queue_and_bounded_queue.md) | PrefillQueue, BoundedQueue, queue capacities |
| [`spec_decode_state.md`](./spec_decode_state.md) | Per-user speculation tracking (Reader-only), ACCEPT/REJECT discrimination |
| [`lock_free_design.md`](./lock_free_design.md) | Lock-free hot path, memory ordering taxonomy, cancel/recycle race prevention |

---

**Next:** [Threading Model](./threading_model.md) | [Chapter 3](../ch3_scheduling_algorithm/index.md)
