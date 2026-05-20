# Final Synthesized Plan: Comprehensive Guide to tt-llm-engine

## Selection Rationale

This plan is a hybrid constructed from the strongest elements of all four candidate plans, guided by consensus findings across all four evaluations.

**Structural backbone: v4** -- Plan v4's chapter ordering and groupings were rated the most balanced. Its placement of IS integration alongside session lifecycle (ch6) was preferred over v3's placement in the tools chapter, and its merged threading+data-structures chapter (ch2) was unanimously praised across all evaluations as the strongest structural decision (eliminating circular forward-reference problems between data structures and threading).

**Granularity within chapters: v1/v2** -- While v3/v4's chapter-level merges are adopted, the per-file decomposition follows v1 and v2. Specifically, the merged ch2 keeps individual files for each data structure (FreeIdPool, UserTable, DecodeStaging, etc.) rather than v3's single monolithic `core_data_structures.md` file. This was explicitly recommended by the v3 evaluator.

**Traceability features: v1** -- Plan v1 was the only plan with an explicit question-to-chapter mapping table and three named reading paths. All evaluators noted these as significant advantages. Both are adopted.

**Detail level: v2** -- Plan v2 had the most detailed per-file content specifications, including specific function names, field names, and memory ordering annotations. These are incorporated throughout.

**Design decisions: v3** -- Plan v3's design-decisions section was the most comprehensive (14 distinct decisions including defer_complete, generation counters, pending_count dual reference counting). This is adopted and extended.

**Planned-vs-implemented flags: v3** -- Plan v3 was the only plan to explicitly flag KV cache tiering as "planned -- API and pseudocode specified in docs but not yet implemented in source." This transparency is adopted throughout.

**Key structural changes from any single plan:**
- KV tiering moved INTO ch6 (session lifecycle + IS integration), following the v4 evaluator's recommendation, creating a cohesive "everything about users and their KV state" chapter
- IS integration placed in ch6 (with lifecycle), not ch7 (with tools), per v3 and v4 evaluator feedback
- ch7 becomes a focused "Transport and Tools" chapter without IS integration or KV tiering
- CPU affinity gets a dedicated file in ch2, per v2/v4 evaluator feedback
- Lock-free design patterns get a dedicated file in ch2, per v3's strength

---

## Audience

**Primary readers:** Systems engineers and infrastructure developers who build, extend, or integrate with production LLM inference serving stacks on Tenstorrent hardware. Readers are expected to have:

- Working knowledge of C++20 (atomics, lock-free patterns, templates, RAII)
- Familiarity with multi-threaded systems programming (POSIX threads, lock-free data structures)
- Understanding of LLM inference concepts (prefill, decode, KV cache, speculative decoding, autoregressive generation)
- Basic understanding of host-device communication models (sockets, DMA, PCIe)
- Experience with CMake-based C++ build systems and Linux development environments
- No prior knowledge of the tt-llm-engine codebase or Tenstorrent-specific APIs (tt-metal, tt-metalium, H2D/D2H sockets) is assumed; all domain-specific concepts are introduced before use

**Secondary readers:** Inference server authors who need to integrate with the DecodeScheduler API (ALLOCATE/SUBMIT/CONTINUE/CANCEL lifecycle), DevOps engineers who build and deploy the engine, and hardware/firmware engineers who need to understand the host-device wire protocol for kernel-side implementations.

**Non-audience:** Model authors working exclusively in Python via tt-blaze or tt-metal's demo layer. This guide focuses on the host-side C++ control plane, not model graph construction.

---

## Chapter List

### Chapter 1: `ch1_architecture_and_layering`

**Description:** Establishes the engine's position in the Tenstorrent inference stack, defines the IS/PM ownership boundary, maps all major components, and introduces the key system invariants that govern every subsequent design decision.

**Files:**

- `index.md` -- Chapter introduction and reading roadmap. Forward references to all later chapters.

- `system_position.md`
  - Where tt-llm-engine sits: between Inference Server stacks (tt-inference-server, custom gRPC/HTTP) and persistent hardware workloads (tt-metal, tt-blaze)
  - The three-layer ASCII diagram: IS layer (HTTP/gRPC frontend) -> tt-llm-engine (control-plane scheduler) -> device-resident model (data-plane)
  - The "bridge" metaphor: the model is long-lived on device, the engine connects to pre-existing H2D/D2H sockets and drives token traffic without launching or tearing down the model
  - Semi-hardened deployment pattern: model launched once and stays resident
  - Repository URL transitional note (tt-asaigal -> tenstorrent)

- `ownership_contract.md`
  - IS-vs-PM (DS) responsibility matrix with explicit two-column table:
    - IS owns: HTTP/WebSocket endpoints, tokenization, chat history, session-to-user mapping, KV tier policy ("why/when"), user-facing wait/retry/timeout, context overflow pre-validation
    - PM owns: user_id allocation, prompt storage, prefill/decode scheduling, pipeline injection, in-flight tracking, KV cache lifecycle execution ("can/how"), safe deferred cancellation
  - What IS must never do (call PM functions directly, perform DMA)
  - What PM must never do (own HTTP sessions, maintain client-facing wait queues, own business-level fairness policy)
  - session_id vs user_id mapping: IS maintains the map, PM returns the allocated user_id

- `component_map.md`
  - Table of all engine components with implementation status: Decode scheduler (available), Pipeline transport (available), Mooncake transport (available), Prefill scheduler (planned), KV-cache migration scheduler (planned)
  - How future components slot in additively without restructuring
  - Two library artifacts: `libtt_llm_engine_core.so` (scheduling logic, MockPipeline, PipelineSimulator, no hardware deps) and `libtt_llm_engine.so` (adds SocketPipeline, requires tt-metalium)
  - Namespace hierarchy: `tt_llm_engine::scheduler::decode`, `tt_llm_engine::pipeline`, `tt_llm_engine::transport::mooncake_test`
  - Public headers mirror namespace (`include/tt_llm_engine/`), private sources mirror the same hierarchy (`src/`), plus `src/common/`
  - Directory layout map (include/, src/, tests/, tools/, kernels/, docs/, cmake/)

- `key_invariants.md`
  - The six key system invariants:
    1. PM never blocks on IS
    2. IS never calls PM functions directly; all interaction through bounded queues
    3. Pipeline never stalls due to IS latency
    4. User IDs not recycled until in_flight_count AND pending_count reach zero
    5. KV cache retained on generation completion (COMPLETE state) for multi-turn reuse
    6. PM does not own end-user waiting policy
  - Timescale separation: IS operates at millisecond granularity (HTTP, TLS, JSON, tokenization), PM at microsecond granularity (per-pipeline-tick injection)
  - Hardware deployment path: fixed-size pre-allocated data structures, bitmap allocators, bounded queues, no dynamic allocation on hot path, integer mode constants

---

### Chapter 2: `ch2_threading_and_data_structures`

**Description:** Details the decode scheduler's three-thread architecture, all shared data structures with their thread-access patterns and concurrency characteristics, and the lock-free design principles that enable microsecond-scale scheduling with zero dynamic allocation on the hot path.

**Files:**

- `index.md` -- Chapter overview. Thread topology diagram. How data structures map to the hardware deployment path. Constants: DEFAULT_MAX_USERS=64, DEFAULT_MAX_SEQ_LEN=131072, DEFAULT_CHUNK_SIZE=24, DEFAULT_EOS_TOKEN=1, INVALID_SLOT, EMPTY_TOKEN.

- `threading_model.md`
  - Three dedicated threads: Writer (hot path, tight inject loop), Reader (hot path, pipeline result processing), API handler (drains request queue via `_mm_pause()` spin-wait)
  - Thread interaction matrix: which thread (API/Writer/Reader) reads/writes each shared state field, at what memory ordering
  - DecodeScheduler::Impl as the central state holder with all data structures
  - Start/stop lifecycle: `running` atomic flag, pipeline->request_stop() then join threads, then pipeline->shutdown()
  - Why the API handler has a dedicated thread rather than piggy-backing on Reader

- `cpu_affinity.md`
  - SchedulerParams: writer_cpu, reader_cpu, api_cpu with AUTO_CPU / NOPIN_CPU sentinel values
  - resolve_cpu_assignments(): queries process affinity mask via sched_getaffinity; removes explicitly claimed cores; distributes remaining cores with stride = N/3 for cross-core spread
  - pin_to_cpu(): pthread_setaffinity_np on the thread's native handle
  - Production tuning guidance: pin Writer and Reader to distinct physical cores for cache isolation

- `free_id_pool.md`
  - FreeIdPool: multi-word bitmap-based O(1)-ish allocator
  - Backing: vector of `atomic<uint64_t>` words, sized as `ceil(max_users / 64)`
  - allocate(): word scan + `__builtin_ctzll` (count trailing zeros) + `compare_exchange_weak` (CAS claim)
  - free(): `fetch_or` (atomic OR to set bit) with release semantics
  - Thread safety: API allocates via CAS; Reader/cleanup-winner frees via atomic OR
  - Scales from single uint64_t (MAX_USERS <= 64) to arbitrary word counts
  - Hardware deployment path: maps directly to a single 64-bit register for small deployments
  - Source: `src/common/free_id_pool.hpp`

- `user_table.md`
  - UserTable: per-user register file, parallel vectors indexed by slot_id
  - Fields catalog with thread ownership annotations:
    - `state` (atomic<UserState>): [API: write on allocate/continue/cancel] [Writer: read] [Reader: write on completion/cleanup]
    - `current_position` (atomic<uint32_t>): [Writer: advance during prefill] [Reader: set on completion] [API: read on continue]
    - `in_flight_count` (atomic<uint32_t>): [Writer: increment before inject] [Reader: decrement on result]
    - `prefill_in_flight` (atomic<uint32_t>): [Writer: increment for non-last prefill] [Reader: decrement and skip]
    - `post_complete_in_flight` (atomic<uint32_t>): [Reader: increment on EOS write-back] [Reader: decrement on BASE result]
    - `in_thinking_phase` (atomic<bool>): [Reader: write after emit_token (release)] [Writer: read in device_sampling_params (acquire)]
    - Non-atomic fields: max_new_tokens, tokens_generated, prefill_pos, prefill_start_pos, prefill_chunk_remaining, spec_decode_enabled, ignore_eos, temperature, top_p, top_k, relaxed_acceptance_threshold, spec_accepts, spec_rejects
  - UserState FSM: INACTIVE -> PREFILL -> DECODE -> COMPLETE -> (CONTINUE back to PREFILL, or CANCEL to INACTIVE)
  - reset() semantics: zeroes all fields, used on ALLOCATE and cleanup
  - Source: `src/common/user_table.hpp`

- `prompt_table_and_cancel_bitmap.md`
  - PromptTable: fixed-size 2D array (max_users x max_seq_len). Prompts stored at index 0, decoupled from device KV position via prefill_start_pos. Writer maps prompt indices to device positions using prefill_start_pos offset. No concurrent-access hazards (API writes before PREFILL, Writer reads during PREFILL, never simultaneously).
  - CancelBitmap: multi-word atomic bitmap mirroring FreeIdPool structure. mark() with release semantics (fetch_or), clear() with relaxed (fetch_and), is_set() with acquire (load). Enables deferred cancellation without blocking the hot path.
  - Source: `src/common/user_table.hpp`, `include/tt_llm_engine/scheduler/decode/decode_types.hpp`

- `decode_staging.md`
  - DecodeStagingEntry struct: slot_id, token_id, position, spec_token_id, spec_position, generation counter, skip_spec flag
  - BoundedQueue-backed FIFO (capacity = max_users * 4): Reader stages entries, Writer pops
  - pending_count per slot: reference count covering FIFO entries + in-progress reader iterations. Combined with in_flight_count forms the busy count that gates maybe_finalize_cleanup
  - Generation counter per slot: Writer drops entries whose generation doesn't match current (handles stale entries from cancelled sessions)
  - stage(): claims pending_count BEFORE publishing entry (prevents cancel/recycle race)
  - release(): decrements pending_count (called by Writer after pop, and by Reader RAII claim destructor)
  - Source: `src/scheduler/decode/decode_staging.hpp`

- `prefill_queue_and_bounded_queue.md`
  - PrefillQueue: mutex-protected deque for users in PREFILL state. push (API on submit/continue), try_front/pop_front (Writer), rotate (Writer for chunked round-robin), remove (API on cancel). Mutex only contended on submit (rare, not on per-tick hot path).
  - BoundedQueue<T>: mutex-based bounded ring buffer for IS<->PM communication. Fixed capacity at construction, no dynamic allocation after init. try_push/try_pop return bool, never block. Used for request_queue (ISRequest, MPSC, 2*max_users), response_queue (SchedulerResponse, SPSC, 2*max_users), output_queue (OutputMessage, SPSC, 256*max_users). Not lock-free by design: acceptable for API path.
  - Why lock-free SPSC was not used for these queues (MPSC request_queue, rare contention makes mutex acceptable)
  - Source: `src/common/bounded_queue.hpp`, `src/scheduler/decode/prefill_queue.hpp`

- `spec_decode_state.md`
  - SpecDecodeState: per-user speculative decode tracking for the Reader thread only (no atomics needed, single-thread ownership)
  - Fields: unverified_token, has_unverified, has_verified, signal_to_exit, pending_complete (stashed OutputMessage), has_pending_complete
  - How these fields implement the state machine for ACCEPT/CONTINUE/REJECT/STALE discrimination
  - reset(): clears all tracking for a slot (called on ALLOCATE and cleanup)
  - Source: `src/scheduler/decode/spec_decode_state.hpp`

- `lock_free_design.md`
  - Hot path communication: Writer/Reader use atomic flags and SPSC staging -- no mutexes on Writer or Reader hot paths
  - DecodeStaging pending_count as dual-purpose reference count (FIFO entries + reader iteration)
  - ReaderClaim RAII: bumps pending_count on construction, releases + maybe_finalize_cleanup on destruction
  - Memory ordering discipline: release/acquire pairs on state transitions, cancel_pending, in_thinking_phase, prefill_in_flight, post_complete_in_flight; relaxed for counters where ordering is not critical (in_flight_count)
  - CAS pattern in maybe_finalize_cleanup: compare_exchange_strong on state -> INACTIVE; exactly one thread wins cleanup, losers see INACTIVE and return
  - Cancel/recycle race prevention: pending_count incremented BEFORE FIFO push; release-acquire on CancelBitmap propagates cancel_request_id
  - Hardware deployment path: all structures fixed-size, pre-allocated; parallel arrays map to register files/SRAM; bitmaps map to single machine words; bounded queues map to DMA FIFOs

---

### Chapter 3: `ch3_scheduling_algorithm`

**Description:** Walks through the complete scheduling flow for regular (non-speculative) decode: writer priority ordering, chunked prefill with round-robin interleaving, the PREFILL-to-DECODE transition, the decode loopback cycle, completion detection, and end-to-end flow walkthrough.

**Files:**

- `index.md` -- Chapter overview. The fundamental scheduling constraint: one token injected per pipeline tick, decode always has priority over prefill. Algorithmic design goals (minimize TPOT, prevent starvation, hardware-deployable).

- `writer_priority_and_decode.md`
  - Writer loop two-tier priority: (1) decode tokens from DecodeStaging FIFO, (2) prefill tokens from PrefillQueue
  - Decode path: try_pop from staging FIFO; check cancel_pending; compute do_spec flag; bump in_flight_count by (do_spec ? 2 : 1) BEFORE releasing staging claim (prevents momentary zero-crossing that could trigger premature cleanup); inject BASE token via pipeline_inject(); conditionally inject SPEC token back-to-back
  - device_sampling_params(): outside thinking phase forces k=1, top_p=1.0 (argmax on device); inside thinking phase passes user's top_p/top_k through (capped to [1,32]); temperature always passed through
  - No-work path: falls through to end of loop body (std::this_thread::yield on empty)
  - Rationale: decode users are actively generating output; every tick of latency is user-visible TPOT. Prefill is background KV build that tolerates delay

- `chunked_prefill.md`
  - Chunk size: configurable via SchedulerParams::chunk_size (default 24)
  - Round-robin: after exhausting a chunk, prefill_chunk_remaining resets to chunk_size, PrefillQueue rotates (move front to back)
  - Prevents starvation: a single long prompt (e.g., 128K tokens) does not starve decode users for thousands of ticks
  - Prefill position management: prefill_pos tracks current device position, prefill_start_pos bookmarks the multi-turn start, prompt_idx = device_pos - prefill_start_pos
  - Last prefill token detection: prompt_idx == prompt_len - 1. Injected with TokenMode::DECODE (triggers sampling on device) and prefill_token_id=EMPTY_TOKEN. On last token: pop from PrefillQueue, set state to DECODE, clear PromptTable
  - prefill_in_flight counter: non-last tokens increment this; Reader skips results that decrement it (they populated KV but produced no sampled output)
  - Context exhaustion during prefill: if device_pos >= max_seq_len, mark COMPLETE with ctx_exhausted=true, push OutputMessage, no output tokens produced
  - Interleaving example: two users in prefill, one in decode; tick-by-tick injection sequence

- `decode_loopback.md`
  - The core decode cycle: Reader receives pipeline result -> stages (token, position) in DecodeStaging -> Writer pops and re-injects -> pipeline processes -> Reader reads result -> cycle repeats every D ticks
  - Pipeline latency: D ticks (numStages, e.g., 64 for 16-device x 4-stage configuration) between inject and result
  - Steady-state: each user gets one output token per D ticks
  - current_position tracking: only updated on completion (set_position_on_complete) for multi-turn bookmark accuracy; intermediate positions tracked via staging entries
  - Generation tagging: stale entries from cancelled sessions filtered by Writer comparing entry generation to current user generation

- `completion_and_eos_writeback.md`
  - Three completion triggers: EOS token match (unless ignore_eos), max_new_tokens reached, context exhaustion (current_position >= max_seq_len)
  - emit_token lambda internals: thinking-phase toggle (release store on in_thinking_phase), tokens_generated increment, completion detection, OutputMessage publication
  - EOS write-back: on completion, stage_eos_writeback injects the completing token at actual_token_pos with skip_spec=true, writing the terminator into KV for next-turn context. post_complete_in_flight counter ensures Reader discards the loopback BASE result
  - State transition: DECODE -> COMPLETE; KV cache retained; current_position preserved for multi-turn
  - Generation counter on OutputMessage: enables IS to filter stale completions

- `regular_decode_flow.md`
  - End-to-end walkthrough with trace diagrams: ALLOCATE -> SUBMIT -> prefill chunked injection -> last-token DECODE mode -> Reader receives first sampled result -> decode loopback steady state -> EOS/max_new_tokens/ctx_exhausted -> COMPLETE
  - Multi-user interleaving diagram: two users in prefill interleaved with decode priority
  - Timing: each decode token traverses D pipeline stages; loopback re-injection happens on the next writer cycle after the reader stages it
  - Multi-turn CONTINUE: after COMPLETE, CONTINUE re-enters PREFILL from current_position with existing KV reused

---

### Chapter 4: `ch4_speculative_decode`

**Description:** Covers the MTP-based speculative decode protocol end-to-end: the four result cases, pair injection mechanics, throughput tradeoffs, KV correction on mismatch, the SpecDecodeState state machine, and thinking-phase tracking with relaxed acceptance.

**Files:**

- `index.md` -- Chapter overview. What MTP is and why it matters for per-user throughput. Pipeline bandwidth tradeoff (each spec user consumes 2 pipeline slots vs 1 for regular). Example: D=64, 20 spec + 24 regular = 64 slots.

- `result_classification.md`
  - The four result cases classified by (actual_token_type, has_unverified, has_verified, signal_to_exit):
    - ACCEPT: BASE result, actual matches unverified prediction (check_acceptance). 1 token to IS. Set has_verified=true. Defer completion if completing (stash OutputMessage via pending_complete).
    - CONTINUE: SPEC result after accept. Bonus output token. New prediction becomes unverified. Stage new pair. On signal_to_exit, suppress the bonus and flush deferred complete.
    - REJECT: BASE result, actual does not match prediction. 1 token to IS. New prediction becomes unverified. Stage pair with base at previous spec position (overwrites stale KV) + new spec at next position.
    - STALE: SPEC result after reject. Garbage from wrong KV. Discard entirely. On signal_to_exit, flush deferred complete.
  - INITIAL case: first sampled result after prefill, no prior speculation. Prediction stored as unverified; pair injected.
  - CONTINUE-suppress: when accept already completed the generation, drain stale SPEC and flush deferred output.
  - Per-round throughput summary: MATCH path = 2 outputs per 2 slots = 2x; MISMATCH path = 1 output per 2 slots + 1 wasted = 1x at 2x cost

- `pair_injection.md`
  - Writer mechanics: pop DecodeStagingEntry; check skip_spec flag and spec_decode_enabled; bump in_flight_count by 2 atomically BEFORE releasing staging claim; inject BASE then SPEC back-to-back via pipeline_->inject()
  - DecodeStagingEntry carries both base (token_id, position) and spec (spec_token_id, spec_position) in a single entry
  - stored_prediction/stored_prediction_pos handoff from Reader to Writer via SpecDecodeState
  - On REJECT: BASE goes to previous spec position (overwrite stale KV), SPEC goes to next position. Pipeline processes in order so SPEC sees corrected KV. No explicit rollback needed -- only one speculative token in pipeline per user
  - stage_eos_writeback: skip_spec=true ensures no phantom SPEC inject on completion

- `state_machine_detail.md`
  - SpecDecodeState fields and their transitions (reader-thread-only, no atomics):
    - unverified_token: the most recent prediction not yet verified by a BASE result
    - has_unverified: true when a prediction is outstanding
    - has_verified: true after ACCEPT, cleared after CONTINUE/STALE processes the paired result
    - signal_to_exit: set when ACCEPT detects completion (EOS/max_new_tokens); CONTINUE/STALE must drain before publishing completion
    - pending_complete: stashed OutputMessage for deferred completion publishing
    - has_pending_complete: gate for flush_pending_complete
  - INITIAL -> ACCEPT -> CONTINUE cycle (steady state match path)
  - INITIAL -> REJECT -> STALE cycle (mismatch path)
  - defer_complete mechanism: prevents client from seeing completion before stale SPEC/STALE result is consumed, avoiding race where CONTINUE arrives before stale result is drained

- `throughput_analysis.md`
  - Summary table: Regular vs Spec decode per-round comparison
  - Regular: 1 slot/user, 1 output token per D ticks
  - Spec match: 2 slots/user, 2 output tokens per D ticks (2x individual throughput)
  - Spec mismatch: 2 slots/user, 1 output token per D ticks + 1 wasted slot (1x throughput at 2x cost)
  - System-level: with N pipeline slots, N regular users produce N tok/D; M spec users produce at most 2M tok/D but consume 2M slots
  - When to enable spec decode: high accept rate (>80%) makes it worthwhile; low accept rate wastes aggregate concurrency
  - spec_accepts/spec_rejects counters per slot tracked in UserTable for diagnostics

- `thinking_phase_and_relaxed_acceptance.md`
  - Thinking-phase tracking: SchedulerParams::think_open_token_id and think_close_token_id bracket the reasoning phase (auto-detected from tokenizer via `<think>` / `</think>` single-token encoding)
  - Per-user in_thinking_phase atomic<bool>: Reader toggles on emit_token (release store), Writer reads in device_sampling_params (acquire load)
  - EMPTY_TOKEN sentinel on think_open/close disables the gate entirely (no token ever matches)
  - Outside thinking phase: force k=1, top_p=1.0 (argmax on device, saves device cycles); strict equality check in check_acceptance (result.actual_token == prev_spec_token_id)
  - Inside thinking phase: pass user's top_p/top_k through to device; relaxed acceptance via probability-delta check
  - check_acceptance(): scan p_indices[0..k) for prev_spec_token_id; compare (p_max - p_draft) <= relaxed_acceptance_threshold. Skip p_score==0 entries (top-P-filtered slots)
  - p_scores: raw bf16 bits (uint16_t); bf16_to_float conversion via bit shift
  - relaxed_acceptance_threshold from GenerationParams, applied per-user

---

### Chapter 5: `ch5_pipeline_and_wire_format`

**Description:** Covers the PipelineInterface abstraction, its three concrete implementations (MockPipeline, PipelineSimulator, SocketPipeline), the variant-based configuration system, and the host-device wire serialization format.

**Files:**

- `index.md` -- Chapter overview. The abstraction boundary: DecodeScheduler operates against PipelineInterface; concrete backend selected by PipelineConfig variant (MockConfig | SocketConfig | PipelineSimulatorConfig) via std::visit.

- `pipeline_interface.md`
  - Abstract base class with five virtual methods:
    - inject(): Writer thread calls, may block if pipeline full. Takes InjectDescriptor.
    - read_result(): Reader thread calls, blocking read. Returns ResultDescriptor with INVALID_SLOT sentinel on shutdown.
    - reset_kv(): cleanup, called during maybe_finalize_cleanup
    - request_stop(): sets internal flags, must not use sockets. Called before thread join.
    - shutdown(): called after all threads joined. Performs final device-side cleanup (exclusive ownership).
  - Two-phase shutdown protocol: request_stop() (flag only) -> join threads -> shutdown() (socket/device cleanup)
  - InjectDescriptor: slot_id, token_id, prefill_token_id, position, token_type (BASE/SPEC), spec_flag, temperature, top_p, top_k
  - ResultDescriptor: slot_id, actual_token, actual_token_type, actual_token_pos, predicted_token, predicted_token_type, predicted_token_pos, p_indices[32], p_scores[32]
  - Sentinels: INVALID_SLOT = UINT32_MAX, EMPTY_TOKEN = UINT32_MAX. TokenType enum: BASE=0, SPEC=1
  - Source: `include/tt_llm_engine/pipeline/pipeline_interface.hpp`, `include/tt_llm_engine/pipeline/pipeline_types.hpp`

- `mock_pipeline.md`
  - In-process queue-based implementation for unit testing without hardware
  - Token model (deterministic): actual = token_id + 1 (accept) or token_id + 3 (reject); predicted = actual + 1
  - Position model: actual_token_pos = position + 1, predicted_token_pos = position + 2
  - accept_rate parameter: 1.0 = always accept, random accept/reject for BASE tokens when < 1.0
  - Non-last prefill: produces result with only slot_id set (Reader skips via prefill_in_flight)
  - Optional latency injection: uniform random sleep in [latency_min_us, latency_max_us] on read_result()
  - Inject log: records all InjectDescriptors for test assertions
  - Shutdown: cv.notify_all, queue-based result delivery
  - Source: `include/tt_llm_engine/pipeline/mock_pipeline.hpp`

- `pipeline_simulator.md`
  - Timing-accurate systolic pipeline model with no tick thread
  - Three invariants: latency = numStages * stageDuration per token, throughput cap = 1/stageDuration, backpressure at numStages in-flight
  - Per-token timestamp design: inject() computes enter_time (rate-limited by lastEnter + stagePeriod) and exit_time = enter + totalLatency, pushes to monotonic FIFO
  - read_result(): busy-wait until head token's exit_time (sub-microsecond emit timing)
  - Why no tick thread: eliminates quantization padding and spin-loop overhead
  - Safe-vocab modular wrap: safeVocabBase + safeVocabModulus (must be 0 or >= 5) wraps token arithmetic to prevent long generations from drifting into stop-token IDs
  - Token model matches MockPipeline so spec-decode verification works identically
  - Configuration: PipelineSimulatorConfig with num_stages, stage_duration_us, decode_token_id, accept_rate, seed
  - Source: `include/tt_llm_engine/pipeline/pipeline_simulator.hpp`

- `socket_pipeline.md`
  - Real hardware backend via tt-metalium H2D/D2H sockets. Requires DS_HAS_METALIUM compile flag (auto-set when linking TtLlmEngine::Full)
  - PIMPL pattern (struct Impl; unique_ptr<Impl>) hiding metalium dependency
  - Constructor: H2DSocket::connect + D2HSocket::connect to pre-created sockets with configurable timeout
  - inject(): serialize_inject() into 256-byte PageBuffer, writes one page to H2D socket
  - read_result(): polls d2h_socket->has_data(), reads one page, deserialize_result()
  - request_stop(): sets atomic flag, breaks read_result poll loop
  - Shutdown: non-DeepSeek mode sends sentinel (INVALID_SLOT) via H2D and drains D2H until sentinel echoed back; DeepSeek mode skips sentinel exchange (kernel has its own lifecycle); both modes call barrier on both sockets
  - use_deepseek_md_format flag: selects between default and DeepSeek wire layouts
  - Source: `include/tt_llm_engine/pipeline/socket_pipeline.hpp`, `src/pipeline/socket_pipeline.cpp`

- `wire_format.md`
  - PAGE_SIZE_BYTES = 256 (matches deepseek_b1_ops::kMetadataTensorBytes). PAGE_SIZE_WORDS = 64. PageBuffer = std::array<uint32_t, 64>.
  - Default layout:
    - InjectPage: [0]=slot_id [1]=token_id [2]=position [3]=prefill_token_id [4]=spec_flag [5]=temperature [6]=top_p [7]=top_k
    - ResultPage: [0]=slot_id [1]=actual_token [2]=predicted_token [6]=actual_token_pos. First 64B meaningful.
  - DeepSeek MD layout (mirrors DeepseekMetadata struct):
    - InjectPage: [1]=token_type [2]=position [6]=slot_id [7]=token_id [8]=position_id [9]=prefill_token_id [10]=temperature [11]=k [12]=probability_mass_threshold
    - ResultPage: [0]=actual_token [1]=actual_token_type [2]=actual_token_pos [3]=predicted_token [4]=predicted_token_type [5]=predicted_token_pos [6]=slot_id [16-47]=p_indices[32] [48-63]=p_scores[32] (bf16, 2 values packed per uint32 word)
  - serialize_inject() and deserialize_result() functions with use_deepseek_md_format toggle
  - Source: `include/tt_llm_engine/pipeline/wire_format.hpp`

---

### Chapter 6: `ch6_lifecycle_is_integration_kv_tiering`

**Description:** Covers the complete user session state machine, multi-turn CONTINUE semantics, deferred cancellation safety, context exhaustion, the three-queue IS integration model, and KV cache tier management.

**Files:**

- `index.md` -- Chapter overview. The full user lifecycle from allocation to teardown. State machine diagram.

- `session_state_machine.md`
  - Full state diagram: INACTIVE -> (ALLOCATE) -> INACTIVE(allocated) -> (SUBMIT) -> PREFILL -> DECODE -> COMPLETE -> (CONTINUE back to PREFILL) or (CANCEL to INACTIVE)
  - State transitions with trigger, responsible thread, and preconditions
  - ALLOCATE: FreeIdPool.allocate(); reset UserTable + CancelBitmap + SpecDecodeState; push response with slot_id or error_code=1 (no free slots)
  - SUBMIT: store prompt in PromptTable; initialize all UserTable fields (position=0, prefill_pos=0, chunk_remaining=chunk_size, etc.); push into PrefillQueue
  - COMPLETE: KV cache retained, current_position preserved. EOS write-back stages completing token into KV for multi-turn context
  - Generation counter: advance_generation on CANCEL to invalidate stale DecodeStaging entries
  - ISRequest struct: type, request_id, slot_id, tokens vector, GenerationParams
  - SchedulerResponse: request_id echo, slot_id, error_code
  - OutputMessage: slot_id, token_id, is_complete, ctx_exhausted, tokens_generated, generation counter

- `multi_turn_continue.md`
  - handle_local_continue: stores new prompt tokens starting at index 0 in PromptTable, sets prefill_pos = prefill_start_pos = current_position (existing KV reused), re-enters PREFILL state
  - handle_disaggregated_continue: for disaggregated prefill/decode architectures. Sets state directly to DECODE, stages the migrated token into DecodeStaging. No local prefill phase
  - Multi-turn KV reuse: current_position persists through COMPLETE, next turn's prefill starts AFTER the EOS position (stage_eos_writeback writes terminator into KV)
  - GenerationParams: max_new_tokens, spec_decode, ignore_eos, temperature, top_p, top_k, disaggregated_decode, relaxed_acceptance_threshold

- `deferred_cancellation.md`
  - Why deferred: pipeline may have in-flight tokens for the cancelled user; immediate cleanup would corrupt pipeline state
  - Four-phase protocol:
    1. API: mark cancel_pending (release store); advance_generation (invalidates stale staging entries); remove from PrefillQueue; stash cancel_request_id BEFORE mark (release-acquire pair ensures cleanup winner sees it)
    2. Writer: skips cancelled users in both decode and prefill paths; releases pending_count for stale entries; calls maybe_finalize_cleanup
    3. Reader: decrements in_flight_count for stale results (discarded); clears prefill_in_flight and post_complete_in_flight; calls maybe_finalize_cleanup
    4. maybe_finalize_cleanup: gated on cancel_pending set AND in_flight_count==0 AND pending_count==0; CAS state -> INACTIVE (exactly one thread wins); pipeline_->reset_kv(); spec_state.reset(); free_ids.free(); push cancel_ack response
  - Cancel ack: SchedulerResponse with the original request_id; enables IS to know when the slot is truly free
  - Special case: CANCEL on allocated-but-never-submitted user (state==INACTIVE): handled directly by API thread (no in-flight to drain)
  - Pipelined teardown pattern in deepseek_inference_runner: batch CANCELs issued fire-and-forget, next batch's allocator drains via allocate_one_with_eviction

- `context_exhaustion.md`
  - Decode-side exhaustion: Reader detects current_position >= max_seq_len after decode result; sets is_complete=true + ctx_exhausted=true on OutputMessage; no loopback staged
  - Prefill-side exhaustion: IS is expected to reject overflow BEFORE submitting (get_current_position + prompt length check). Writer has defensive fallback: marks COMPLETE with no output tokens if device_pos >= max_seq_len
  - Configurable max_seq_len (default 128K)
  - Responsibility split table: PM reader detects decode exhaustion, IS detects prefill overflow before submit, IS owns evict/summarize/error policy, PM writer has defensive guard

- `is_integration_patterns.md`
  - Three-queue communication model:
    - request_queue (ISRequest): IS -> PM, MPSC, bounded (2*max_users capacity)
    - response_queue (SchedulerResponse): PM -> IS, SPSC, bounded (2*max_users capacity)
    - output_queue (OutputMessage): PM -> IS, SPSC, bounded (256*max_users capacity)
  - Backpressure semantics: request_queue full -> IS returns 503; response_queue full -> PM spins briefly (allocation rare); output_queue full -> PM spins with _mm_pause (design choice between drop and brief block)
  - Session-to-slot mapping: IS maintains session_id -> user_id and user_id -> session_id hash maps
  - IS-side pseudocode: allocate_and_submit, continue_conversation (with prefill overflow pre-check), cancel_session, drain_outputs loop with ctx_exhausted handling
  - OutputMessage generation counter: enables IS to match output tokens to the correct generation/turn
  - Independent deployment: same process, different process, different machine. Independent scaling (PM on accelerator co-processor, IS on host CPU). Independent testing (mock queues)

- `kv_cache_tiering.md`
  - **Status: Planned -- API contract and pseudocode are specified in docs but not yet implemented in source code.**
  - Three-tier model: T0-Hot (accelerator DRAM/HBM, lowest latency), T1-Warm (host DRAM, requires migration before decode), T2-Cold (SSD/object store, highest latency)
  - Strict IS/PM ownership: IS owns policy ("why/when" -- thresholds, SLA classes, idle-time demotion), PM owns execution ("can/how" -- tracks tier/capacity, executes migration, LRU eviction)
  - TierCommand API: TierOp enum (ENSURE_TIER, EVICT, PIN, UNPIN), TierCommand struct (user_id, target_tier, priority_class, deadline_ms, allow_auto_evict)
  - TierCommandResult with PMStatus codes: OK, NO_SPACE, BUSY, INVALID_USER, INVALID_STATE, IO_ERROR
  - tier_controller_tick: IS control-plane periodic loop at 100-1000ms cadence (not on token hot path). Checks idle time, issues demotion/spill commands.
  - Migration paths: T0->T1 (demotion), T1->T0 (promotion), T1->T2 (spill), T2->T1/T0 (restore)
  - PM execution mechanics: KVResidency per-user struct (tier, bytes_hot/warm/cold, last_access_ts, pinned, migration_in_flight). PMCapacity tracking (hot_capacity_bytes, hot_used_bytes). can_admit_decode() check.
  - Eviction: select_eviction_candidates via LRU sorted, filter out pinned/in-flight/active users. IS orchestrates final migration decision.
  - Admission failure: PM returns NO_SPACE immediately; IS decides wait/retry/timeout behavior

---

### Chapter 7: `ch7_transport_and_tools`

**Description:** Covers the Mooncake transport integration, the loopback microbenchmark (device kernel and host-side tools), and the DeepSeek inference runner.

**Files:**

- `index.md` -- Chapter overview. Practical tooling for testing, benchmarking, and running inference.

- `mooncake_transport.md`
  - Purpose: standalone transport testing for Transfer Engine (TCP, RDMA) and Mooncake Store. No decode scheduler dependency.
  - TestEngine RAII wrapper: centralizes TransferEngine init, transport installation (TCP or RDMA), buffer registration (posix_memalign, page-aligned), peer segment management
  - EngineConfig: local_server_name, metadata_uri (P2PHANDSHAKE default), mode (Tcp/Rdma), buffer_size, poll_interval, default_timeout
  - Transfer helpers: submit_write_and_wait, submit_read_and_wait. Poll-based completion with configurable timeout
  - RDMA handling: rdma_build_nic_priority_matrix() builds JSON from active libibverbs devices for Mooncake topology. Three gates: build-time flag (DS_MOONCAKE_WITH_RDMA), runtime known-broken-hardware check (bnxt_re* driver skip), active-device probe (IBV_PORT_ACTIVE). DS_MOONCAKE_FORCE_RDMA=1 bypasses runtime probes
  - Store tests: mooncake_master daemon lifecycle; Client::Put/Get via in-process master
  - Build gating: MOONCAKE=ON env var triggers DS_ENABLE_MOONCAKE. Mooncake built with USE_HTTP=OFF, WITH_STORE_RUST=OFF. Submodule pinned to tagged release.
  - Dependencies: libibverbs-dev required even for TCP-only. Boost/msgpack-cxx/zstd vendored.
  - Test binaries: test_mooncake_transport (smoke, TCP 64K/256K/8M, RDMA), test_mooncake_store (Put/Get)
  - run_mooncake_tests.sh: starts mooncake_master, runs ctest, kills master on exit
  - Source: `include/tt_llm_engine/transport/mooncake/mooncake_test_engine.hpp`, `tests/transport/mooncake/`, `docs/transport/mooncake.md`, `cmake/mooncake.cmake`

- `loopback_microbenchmark.md`
  - Purpose: exercises full DecodeScheduler stack against a device kernel that echoes tokens back; measures scheduling overhead without a real model
  - Two-process architecture:
    - dummy_pipeline_launcher: opens MeshDevice, creates H2D/D2H sockets, compiles and launches pipeline_loopback.cpp kernel, waits for kernel completion. CLI: --h2d-socket-id, --d2h-socket-id, --fifo-size
    - dummy_pipeline_connector: connects to sockets via SocketConfig, instantiates DecodeScheduler, allocates users, drives multi-user multi-turn traffic, reports comprehensive metrics (TTFT, ITL, TPOT, throughput, peak concurrent). CLI: --h2d/d2h-socket-id, --prompt-len, --max-decode, --max-turns, --num-users, --connect-timeout-ms
  - Launch methods: MPI launch (single mpirun command for convenience) or standalone (two terminals)
  - Environment: TT_METAL_RUNTIME_ROOT, TT_VISIBLE_DEVICES=0
  - Source: `tools/pipeline/dummy_pipeline_launcher.cpp`, `tools/pipeline/dummy_pipeline_connector.cpp`

- `loopback_kernel.md`
  - On-device kernel: `kernels/pipeline_loopback.cpp`
  - Reads InjectDescriptor pages from H2D socket; produces ResultDescriptor pages; writes to D2H socket
  - Token model: prefill -> actual_token = EMPTY_TOKEN (-1); decode -> actual_token = token_id + 1; predicted_token = EMPTY_TOKEN
  - Position model: actual_token_pos = position + 1
  - Sentinel protocol: exits when slot_id == INVALID_SLOT (0xFFFFFFFF), echoing it back
  - Wire format word indices matching wire_format.hpp (non-DeepSeek layout)
  - Device APIs used: socket_wait_for_pages, socket_reserve_pages, socket_push_pages, socket_notify_receiver/sender, noc_wwrite_with_state
  - Source: `kernels/pipeline_loopback.cpp`

- `deepseek_inference_runner.md`
  - Production-oriented runner for DeepSeek models. Always uses use_deepseek_md_format=true and DeepSeek chat template.
  - Two modes: synthetic (--prompt-len generates dummy token IDs for ISL-controlled benchmarking) and tokenizer (--tokenizer with --prompt or --prompts-file for real prompts)
  - tokenizers-cpp integration: wraps HuggingFace Rust-based tokenizers crate. Requires Rust >= 1.80 via rustup. Loads tokenizer.json.
  - DeepSeek chat template: BOS + User tag + content + Assistant tag (fullwidth bar U+FF5C and lower-one-eighth-block U+2581 Unicode)
  - EOS and thinking tokens (`<think>`/`</think>`) auto-detected from tokenizer (each must encode to exactly one token ID)
  - Prompts file format: flat JSON array (single-turn per user) or 2D JSON array (multi-turn per user). num_users/max_turns inferred from file.
  - Batching: --batch-size N processes users in batches. Pipelined teardown: prior batch's CANCELs issued fire-and-forget, next batch allocator drains via allocate_one_with_eviction for inter-batch slot recycling
  - Multi-turn: CONTINUE issued after each turn completes; existing KV reused
  - Comprehensive benchmark output: per-user summary (slot, turns, ISL, tokens, TTFT, TPOT, spec accept/reject stats), per-batch summary (when batching), aggregate serving benchmark (request throughput, output/total token throughput, peak concurrent, TTFT/ITL/TPOT mean/median/P99, spec accept rate)
  - Decoded output display in tokenizer mode (per user, per turn)
  - Model launch: --launch-only flag on model pipeline, exports deepseek_h2d/deepseek_d2h sockets, runner connects
  - Source: `tools/decode/deepseek_inference_runner.cpp`, `tools/decode/prompts.json`

---

### Chapter 8: `ch8_build_system_and_design_decisions`

**Description:** Documents the CMake build system, integration methods, runtime environment, and the key design decisions and tradeoffs that shaped the system.

**Files:**

- `index.md` -- Chapter overview.

- `build_system.md`
  - Two library variants:
    - libtt_llm_engine_core.so (TtLlmEngine::Core): scheduling logic only, MockPipeline, PipelineSimulator. Always built. Links Threads::Threads.
    - libtt_llm_engine.so (TtLlmEngine::Full): adds SocketPipeline. Requires TT::Metalium. Defines DS_HAS_METALIUM=1.
  - CMake structure: tt_llm_engine_core always built; tt_llm_engine only when TARGET TT::Metalium found (from parent add_subdirectory or find_package)
  - TT_METAL_SOURCE_DIR: points to tt-metal source for not-yet-exported socket headers (d2h_socket.hpp, h2d_socket.hpp)
  - Build scripts: build.sh (--standalone or --with-metal), setup.sh (combined tt-metal + tt-llm-engine)
  - Test targets: test_decode_scheduler (mock, always built), test_decode_scheduler_device (device, requires Metalium), test_mooncake_transport/store (requires DS_ENABLE_MOONCAKE). Google Test fetched via FetchContent.
  - Tool targets: dummy_pipeline_launcher (requires Metalium), dummy_pipeline_connector (links Full), deepseek_inference_runner (requires Metalium + Rust/cargo)
  - Tokenizer build: conditional on cargo presence; FetchContent for tokenizers-cpp pinned to known-good SHA; CMAKE_POLICY_VERSION_MINIMUM workaround for vendored msgpack
  - Mooncake build: DS_ENABLE_MOONCAKE, DS_MOONCAKE_WITH_RDMA options; cmake/mooncake.cmake inclusion; vendored deps (Boost, msgpack-cxx, zstd)
  - TSAN support: DS_ENABLE_TSAN, TSAN=ON shorthand for build.sh
  - CMake options summary table: DS_BUILD_TESTS, DS_ENABLE_TSAN, DS_ENABLE_MOONCAKE, DS_MOONCAKE_WITH_RDMA, DS_NO_TOKENIZER
  - Source: `CMakeLists.txt`, `cmake/mooncake.cmake`, `build.sh`, `setup.sh`

- `integration_methods.md`
  - Method 1: CMake add_subdirectory (recommended). Add tt-llm-engine as submodule. Pre-build tt-metal with build_metal.sh (clang-20 toolchain, Ninja generator). Pass TT_METAL_SOURCE_DIR and CMAKE_PREFIX_PATH.
  - Method 2: find_package (installed tt-llm-engine). cmake --install with CMAKE_INSTALL_PREFIX, consumer find_package(tt-llm-engine CONFIG).
  - Runtime environment: TT_METAL_RUNTIME_ROOT (required for device tests and Full library code outside tt-metal directory)
  - Install rules: CMake package config, export targets, include directory
  - Source: `cmake/tt-llm-engine-config.cmake.in`, `README.md`

- `design_decisions.md`
  - Configurable capacity (MAX_USERS): deployment-configurable via SchedulerParams, not hardcoded to 64. Supports wider pipelines, multi-chip pools, time-sliced admission.
  - Decode priority over prefill: minimizes TPOT. Decode users are actively waiting; prefill is background KV build that tolerates delay.
  - Chunked prefill (default 24): prevents single long prompt from starving decode users for thousands of ticks.
  - KV cache retention on COMPLETE: avoids re-prefilling for multi-turn. Cost: completed sessions consume slot and DRAM until explicit CANCEL.
  - Spec decode bandwidth cost: 2 pipeline slots per user per round. Up to 2x individual throughput at cost of roughly halved aggregate concurrency for fixed pipeline bandwidth.
  - FIFO vs priority scheduling: FIFO default for predictability. Pluggable SchedulerPolicy interface (FifoPolicy, PriorityPolicy) for SLA-aware scheduling.
  - Lock-free hot path: Writer/Reader use atomics + SPSC staging only. PrefillQueue mutex acceptable (rare contention on submit, not per-tick).
  - No KV rollback for spec mismatch: re-inject at same position overwrites stale entry. Only one speculative token in pipeline per user -- no downstream invalidation chain.
  - No PM waiting queue: returns admission failure promptly (NO_SPACE). IS owns wait/retry/timeout policy.
  - Fixed-size pre-allocated data structures: no dynamic allocation on hot path. Parallel arrays, bitmap allocators, bounded queues, integer mode constants (no string comparison in scheduling loops). Designed for eventual migration to hardware co-processor.
  - Generation counters for stale entry detection: Writer drops stale DecodeStaging entries by generation comparison. No timing dependencies between cancel and writer drain.
  - ReaderClaim RAII pattern: pending_count bump covers the full reader iteration, preventing premature finalization when cancel arrives mid-iteration.
  - defer_complete for spec decode: prevents client from seeing completion before stale SPEC/STALE result is consumed. Avoids race where CONTINUE arrives before stale result is drained.
  - pending_count dual reference counting: covers both FIFO entries and in-progress reader iterations. Closes cancel/recycle race between stage time and writer pick-up.
  - EOS write-back pattern: completing token staged back to pipeline with skip_spec=true so next turn sees it in KV context. post_complete_in_flight counter ensures write-back result is discarded.
  - Keep KV alive until pressure: completed sessions remain resident in T0. Eviction LRU-based under tier pressure. IS controls eviction policy.

---

## Conventions

### Terminology

| Term | Definition |
|------|-----------|
| **IS** | Inference Server -- the upstream HTTP/gRPC server stack (e.g., tt-inference-server) that owns client sessions and tokenization. |
| **PM** or **DS** | Pipeline Manager / Decode Scheduler -- the tt-llm-engine scheduling component. "PM" is the original name from internal docs and variable names; "DS" is the component-specific name. Used interchangeably in the codebase; this guide prefers "PM" when discussing the code and "Decode Scheduler" when discussing the architecture. |
| **Pipeline** | The hardware inference pipeline on Tenstorrent devices, modeled as D stages of latency with one-token-per-tick throughput. |
| **slot_id / user_id / uid** | Internal KV cache index allocated by PM. Range [0, max_users). All three names refer to the same integer. Opaque to the end user. |
| **session_id** | External client-facing identity managed by IS. Mapped to slot_id by IS. |
| **Turn** | One request-response cycle within a multi-turn conversation (SUBMIT or CONTINUE -> stream tokens -> COMPLETE). |
| **Tick** | One pipeline stage period; the minimum injection interval. |
| **D** | Number of pipeline stages (e.g., 64 for a 16-device x 4-stage configuration). |
| **TPOT** | Time Per Output Token -- per-user latency between successive output tokens. |
| **TTFT** | Time To First Token -- latency from request submission to the first output token. |
| **ITL** | Inter-Token Latency -- wall-clock time between consecutive output tokens across all users. |
| **ISL** | Input Sequence Length -- number of prompt tokens. |
| **MTP** | Multi-Token Prediction -- the speculative decode mechanism where the model produces a prediction alongside each sampled token. |
| **spec pair** | A (BASE, SPEC) token pair injected back-to-back for speculative decode. |
| **KV position** | The monotonically increasing write-head index into a user's KV cache on the device. Persists across turns. |
| **Generation** | A monotonically increasing counter per slot, advanced on CANCEL, used to detect stale DecodeStaging entries. |
| **in_flight_count** | Number of tokens for a given user currently inside the hardware pipeline. Must reach zero before slot can be freed. |
| **pending_count** | Per-slot reference count covering DecodeStaging FIFO entries and in-progress reader iterations. |
| **hot path** | The Writer and Reader thread loops that must run at pipeline tick frequency. |
| **wire format** | The serialized 256-byte page layout exchanged between host and device via H2D/D2H sockets. |
| **page** | A fixed-size buffer (256 bytes / 64 uint32 words) used as the unit of socket communication. |
| **EOS** | End-of-Sequence token. |
| **H2D / D2H** | Host-to-Device / Device-to-Host socket. |

### Notation

- C++ code listings use the exact types and names from the source. When the source uses `uint32_t`, the guide uses `uint32_t` (not `unsigned int`).
- Source file paths are always relative to the tt-llm-engine repository root (e.g., `src/scheduler/decode/decode_scheduler.cpp`).
- Struct/class names use C++ qualified notation where helpful: `tt_llm_engine::scheduler::decode::DecodeScheduler`.
- Memory ordering annotations appear explicitly in parentheses when discussing atomic operations: (relaxed), (acquire), (release), (acq_rel).
- Thread ownership is marked with brackets: [API], [Writer], [Reader], [any].
- Wire format word indices appear as `[N]` referencing uint32_t array positions.
- Pseudocode blocks are annotated with the thread they execute on.

### Formatting Rules

- Each `.md` file begins with a level-1 heading matching the file's topic, followed by a one-paragraph summary.
- Code blocks use triple-backtick fences with language identifiers (`cpp`, `cmake`, `bash`).
- Tables use GitHub-flavored Markdown pipe syntax.
- ASCII diagrams use box-drawing characters for architecture, state machines, and data flow. Mermaid is not used.
- Cross-references to other files within the guide use relative Markdown links (e.g., `[wire format](../ch5_pipeline_and_wire_format/wire_format.md)`).
- Key invariants and safety properties are called out in blockquotes prefixed with "**Invariant:**" or "**Safety:**".
- Section headers within files use `##` (H2) for major sections and `###` (H3) for subsections.
- Features that are designed but not yet implemented are marked with a prominent callout: "**Status: Planned** -- API contract specified in docs; not yet implemented in source."

---

## Cross-Chapter Dependencies

| Chapter | Depends On | Concepts Referenced |
|---------|-----------|---------------------|
| ch1 (Architecture & Layering) | -- | Foundation for all subsequent chapters. |
| ch2 (Threading & Data Structures) | ch1 | IS/PM ownership split; component boundaries define which structures belong to which subsystem; key invariants (no dynamic allocation, no blocking). |
| ch3 (Scheduling Algorithm) | ch2 | All data structures (UserTable, DecodeStaging, PrefillQueue, PromptTable, CancelBitmap). Threading model (Writer/Reader/API thread roles). Lock-free design patterns. |
| ch4 (Speculative Decode) | ch2, ch3 | SpecDecodeState, DecodeStagingEntry pair semantics. Writer priority ordering (pair injection). Reader loop (result classification). Decode loopback (as baseline). |
| ch5 (Pipeline & Wire Format) | ch1, ch2 | PipelineInterface role in layering (ch1). InjectDescriptor/ResultDescriptor types, TokenType, INVALID_SLOT/EMPTY_TOKEN sentinels (ch2 references these in staging entries). |
| ch6 (Lifecycle, IS Integration, KV Tiering) | ch2, ch3, ch4 | UserState FSM, CancelBitmap, in_flight_count/pending_count gating (ch2). API thread request handling (ch2). Prefill/decode flow (ch3). Spec decode completion deferral (ch4). IS/PM ownership split (ch1, reinforced). |
| ch7 (Transport & Tools) | ch1, ch5, ch6 | Architecture context (ch1). SocketPipeline for loopback kernel, wire format for page layout (ch5). ALLOCATE/SUBMIT/CONTINUE/CANCEL lifecycle for tools, multi-turn semantics for DeepSeek runner (ch6). |
| ch8 (Build System & Design Decisions) | all prior chapters | Design decisions reference specific mechanisms from all prior chapters: lock-free hot path (ch2), decode priority (ch3), spec decode tradeoffs (ch4), pipeline abstraction (ch5), lifecycle safety (ch6), tool dependencies (ch7). |

---

## Question-to-Chapter Mapping

| # | Question | Primary Chapter | Key Files |
|---|----------|----------------|-----------|
| Q1 | Architectural position, layering, IS/PM ownership | ch1 | system_position.md, ownership_contract.md, component_map.md, key_invariants.md |
| Q2 | Threading model (Writer/Reader/API, shared state, lock-free) | ch2 | threading_model.md, cpu_affinity.md, lock_free_design.md |
| Q3 | Core data structures | ch2 | free_id_pool.md, user_table.md, prompt_table_and_cancel_bitmap.md, decode_staging.md, prefill_queue_and_bounded_queue.md, spec_decode_state.md |
| Q4 | Scheduling algorithm (writer priority, chunked prefill, loopback) | ch3 | writer_priority_and_decode.md, chunked_prefill.md, decode_loopback.md, regular_decode_flow.md |
| Q5 | Speculative decode (MTP) | ch4 | result_classification.md, pair_injection.md, state_machine_detail.md, throughput_analysis.md |
| Q6 | PipelineInterface and implementations | ch5 | pipeline_interface.md, mock_pipeline.md, pipeline_simulator.md, socket_pipeline.md |
| Q7 | Wire format (page buffers, default vs DeepSeek layouts) | ch5 | wire_format.md |
| Q8 | User session lifecycle (state machine, multi-turn, deferred cancel) | ch6 | session_state_machine.md, multi_turn_continue.md, deferred_cancellation.md |
| Q9 | KV cache tier management | ch6 | kv_cache_tiering.md |
| Q10 | Mooncake transport | ch7 | mooncake_transport.md |
| Q11 | Loopback microbenchmark | ch7 | loopback_microbenchmark.md, loopback_kernel.md |
| Q12 | DeepSeek inference runner | ch7 | deepseek_inference_runner.md |
| Q13 | IS integration patterns (three-queue, backpressure, session mapping) | ch6 | is_integration_patterns.md, session_state_machine.md |
| Q14 | Build system (two libraries, CMake, integration) | ch8 | build_system.md, integration_methods.md |
| Q15 | Design decisions and tradeoffs | ch8 | design_decisions.md |
| Q16 | Thinking-phase tracking and relaxed acceptance | ch4 | thinking_phase_and_relaxed_acceptance.md |

---

## Reading Paths

### Quick Start Path (for IS integrators)

For developers who need to integrate with the DecodeScheduler API without understanding internal scheduling mechanics:

**ch1** (full chapter) -> **ch6** (session_state_machine.md, multi_turn_continue.md, is_integration_patterns.md) -> **ch8** (build_system.md, integration_methods.md)

This path covers the architectural context, the ALLOCATE/SUBMIT/CONTINUE/CANCEL lifecycle, the three-queue communication model, and how to build and link the library. ~5 files.

### Deep Dive Path (for scheduler developers)

For developers extending the scheduler, adding new scheduling policies, or debugging concurrency issues:

**ch1** -> **ch2** -> **ch3** -> **ch4** -> **ch5** -> **ch6** -> **ch7** -> **ch8**

Full sequential reading. Each chapter builds on the previous. This is the recommended order for first-time readers who need comprehensive understanding. ~40 files.

### Hardware/Transport Path (for kernel and transport developers)

For developers working on device kernels, wire format changes, or transport layer integration:

**ch1** (system_position.md, component_map.md) -> **ch5** (full chapter, especially wire_format.md, socket_pipeline.md) -> **ch7** (loopback_kernel.md, loopback_microbenchmark.md, mooncake_transport.md) -> **ch8** (build_system.md)

This path covers the host-device interface, page layout, socket protocol, the loopback kernel as a reference implementation, and how to build with hardware dependencies. ~10 files.
