# Final Plan: Comprehensive Understanding of the Pi0.5 Simulation and Benchmarking Framework

## Selection Rationale

This final plan is a hybrid that takes Plan v2's superior structural organization and merges in Plan v1's depth and coverage advantages. The key structural decisions are:

1. **Config-before-simulator ordering (from Plan v2).** The JSON configuration schema is placed in Chapter 2, immediately after the architecture overview. This follows the user's natural workflow: a developer encounters the config file before needing to understand the simulator internals. It also eliminates the circular dependency identified by evaluators where Plan v1's Chapter 2 (pipeline model) referenced config values not formally defined until Chapter 3.

2. **PipelineSimulator as a standalone chapter (from Plan v2).** Elevating the simulator to its own chapter (Ch3) gives dedicated space for the three invariants, the busy-wait trade-off, batch prefill behavior, and spec-decode token arithmetic. Plan v1 bundled this with the phase model, which risked simulator internals being overshadowed by higher-level pipeline discussion.

3. **Dedicated run_hybrid.sh orchestration file (from Plan v1, addressing Plan v2's gap).** Both evaluators noted that Plan v2 lacked a dedicated orchestration walkthrough. This plan includes a full run_hybrid.sh walkthrough in Chapter 7, covering the shell script line-by-line, environment variables, the IOMMU sleep rationale, and end-to-end startup/shutdown workflow.

4. **Dedicated race condition file in the bulk protocol chapter (from Plan v1).** Plan v1's `ch04_race_conditions.md` covering five distinct concurrency hazards was identified as a strength by both evaluators. This plan preserves a dedicated race condition analysis within the protocol chapter where the hazards are most relevant.

5. **Dedicated production kernel differences file (from Plan v2).** Both evaluators agreed this consolidation is superior to Plan v1's approach of scattering passthrough-vs-production concerns across Ch5 and Ch8.

6. **Broader audience definition (from Plan v1, addressing Plan v2's weakness).** Plan v1's audience definition does not assume prior Tenstorrent stack familiarity, making the guide accessible to a wider set of systems engineers. Chapter 1 introduces the necessary Tenstorrent concepts (MeshDevice, Tensix cores, NOC, circular buffers, H2D/D2H sockets) at the level needed for the guide.

7. **Explicit numerical config walkthroughs (from Plan v1, addressing Plan v2's weakness).** The reference configs file includes full numerical computation of total_stages, effective_stage_us, and total_latency_us for each of the three configs, matching Plan v1's pedagogical depth.

8. **Richer terminology table (merged from both plans).** Includes DEVICE_PULL, HOST_PUSH, and Bulk channel definitions from Plan v2 plus the Observation/Concern/Mitigation critique structure from Plan v1.

9. **pcie_noc_utils.h expanded coverage (addressing Plan v1 evaluator's concern).** The pcie_noc_utils file includes discussion of the NOC_MAX_BURST_SIZE value and its hardware rationale, not just the function signatures.

10. **pcie_bandwidth_gbps explicitly called out as informational-only (from Plan v2, addressing Plan v1 evaluator's concern).** This is noted in both the config schema chapter and the bulk transfer metrics file to prevent reader confusion.

---

## Audience

This guide targets **systems engineers and infrastructure developers** working on or adjacent to Tenstorrent's tt-llm-engine project. Readers are expected to:

- Be proficient in C++ (C++17/20), including templates, `std::variant`, atomics, condition variables, and POSIX process management (`fork`/`exec`/`waitpid`).
- Understand basic concepts of hardware pipelines (stages, latency vs. throughput, systolic execution).
- Have passing familiarity with PCIe data transfer concepts (DMA, TLB, page alignment).
- Know what LLM inference involves at a high level (prefill, decode, token generation, KV cache).
- Have heard of Pi0.5 / Physical Intelligence's vision-language-action model (SigLIP vision encoder, Gemma language backbone, flow-matching denoiser), but do NOT need to know its internals.
- Not necessarily have prior experience with Tenstorrent hardware, tt-metal APIs, the NOC, MeshDevice, Tensix cores, circular buffers, or the socket abstraction. These are introduced at the level needed within the guide.

The guide does NOT assume the reader has run the benchmark before. It should enable them to run, debug, extend, and critically evaluate the framework.

---

## Chapter List

### Chapter 1: Architecture Overview and Operating Modes

**Description:** Introduces the Pi0.5 benchmarking framework's purpose, the three-phase pipeline model (vision, text, denoise), the three operating modes (simulation, full socket, hybrid/bulk-only), and how all components compose to form the end-to-end system.

**Files:**

- `01_architecture_overview.md`
  - What Pi0.5 is (VLA robotics model: SigLIP vision encoder -> Gemma text backbone -> Euler flow-matching denoise loop) and why a dedicated benchmarking framework exists
  - Brief introduction to Tenstorrent concepts needed for this guide: MeshDevice (multi-chip topology manager), Tensix cores (RISC-V compute units), NOC (Network-on-Chip for data movement), circular buffers (L1 data staging), H2D/D2H sockets (host-device communication channels)
  - The three operating modes:
    - **Simulation-only:** PipelineSimulator backend inside DecodeScheduler, no hardware, no sockets, no device launcher; exercises timing model and scheduler logic only
    - **Full socket:** SocketPipeline for token path + BulkH2DChannel/BulkD2HChannel for pixel/action data + DeviceLauncher fork/exec of pi05_device_launcher; requires multi-chip topology with MPI (DistributedContext)
    - **Hybrid/bulk-only:** PipelineSimulator for token path + real BulkH2DChannel/BulkD2HChannel over PCIe; designed for disconnected N150s where multi-chip SocketPipeline is unavailable but single-chip PCIe works; DeviceLauncher runs with --bulk-only and TT_VISIBLE_DEVICES=0
  - Decision matrix: when to use each mode (development/CI vs integration testing vs single-chip hardware validation)
  - Annotated architecture diagram (textual) showing host process, device launcher, kernel, and data flow paths in each mode

- `02_three_phase_pipeline_model.md`
  - PhaseConfig struct: name, stage_durations_us vector, loop_count multiplier, and derived methods (num_stages, latency_us, effective_stages, effective_latency_us)
  - Vision phase: models SigLIP encoder; typically 4 stages at 1000/2200/2200/2200us; runs once per inference (loop_count=1)
  - Text phase: models Gemma language backbone; typically 20 stages at 591us each; runs once per inference (loop_count=1)
  - Denoise phase: models Euler flow-matching denoising steps; 6 stages at 757us each with loop_count=5 representing 5 denoising iterations (30 effective stages)
  - How effective_stages() and effective_latency_us() collapse the multi-phase model into a single (total_stages, effective_stage_us) pair for PipelineSimulator
  - Why the weighted-average flattening is necessary (PipelineSimulator models a uniform systolic pipeline) and what accuracy it sacrifices (per-phase timing structure is lost; the simulator cannot model the vision phase completing before text begins)
  - The bottleneck stage calculation: max_stage_duration_us vs max_non_vision_stage_us and how these feed corrected TTFT estimates

- `03_build_system_and_component_map.md`
  - Source file map: which files implement which components (pi05_pipeline_runner.cpp, pi05_device_launcher.cpp, pi05_bulk_passthrough.cpp, pcie_noc_utils.h)
  - CMakeLists.txt build targets: pi05_pipeline_runner (simulation-only, links tt_llm_engine_core), pi05_pipeline_runner_device (PI05_HAS_SOCKETS=1, links tt_llm_engine), pi05_device_launcher (links TT::Metalium)
  - Compile-time feature gating: PI05_HAS_SOCKETS preprocessor guard controls socket/bulk channel compilation; DS_HAS_METALIUM for SocketPipeline inside decode_scheduler.cpp
  - DS_KERNEL_DIR compile definition: how the device launcher locates kernel source files at runtime
  - Link dependencies and their implications: simulation-only builds avoid linking against Metalium, enabling development without hardware
  - Namespace map: pm = tt_llm_engine::scheduler::decode, pl = tt_llm_engine::pipeline

### Chapter 2: JSON Configuration Schema and Parsing

**Description:** Documents the custom JSON config format (PipelineConfig, SocketConfigParams, PixelPayloadConfig), the hand-rolled recursive-descent parser, the three reference config files with numerical walkthroughs, and CLI override precedence.

**Files:**

- `01_config_schema_reference.md`
  - PipelineConfig struct: phases (required array), input_tokens, output_tokens, num_users, max_turns (all optional uint32_t with defaults)
  - phases array: each element is an object with name (required string), stages (required array of uint32_t >= 1), loop_count (optional uint32_t >= 1, default 1)
  - socket block (optional, triggers use_sockets=true): h2d_token_socket_id, d2h_token_socket_id, h2d_bulk_socket_id, d2h_bulk_socket_id (all strings), h2d_mode ("HOST_PUSH" or "DEVICE_PULL", default "DEVICE_PULL"), connect_timeout_ms (uint32_t, default 30000), pcie_bandwidth_gbps (double, default 16.0), fifo_size (uint32_t, default 524288), mode ("full" or "bulk_only", default "full"), num_users_for_launch
  - PixelPayloadConfig struct: width, height, channels, frames (uint32_t), action_history_bytes (uint32_t, default 4096); pixel_bytes() = W*H*C*F; total_bytes() = pixel_bytes() + action_history_bytes
  - **NOTE:** pcie_bandwidth_gbps is informational-only -- it is NOT used in actual bandwidth calculation. Bandwidth is measured empirically from transfer timing. This field exists for documentation/logging purposes.
  - Default values for each field and what happens when they're omitted
  - CLI argument override priority: --num-users > config num_users > default 4; same pattern for --prompt-len, --max-decode, --max-turns
  - Validation rules: at least one phase, every phase needs a name and at least one stage >= 1us, loop_count >= 1, padded H2D payload must fit in fifo_size

- `02_custom_json_parser.md`
  - Why a custom parser exists (no external JSON dependency like nlohmann::json or rapidjson; minimal at ~120 lines; forward-compatible via skip_json_value)
  - Parser structure: recursive-descent with skip_ws, expect, peek, parse_string, parse_number, parse_float_val, skip_json_value
  - skip_json_value: handles strings, nested objects, arrays, booleans/null, and numbers recursively; enables forward compatibility by silently ignoring unknown keys at any nesting level
  - Known limitations and edge cases:
    - No escape sequence handling in parse_string (backslash-quote in values will break the parser)
    - No negative integer support in parse_number (only digits via stoul); a negative stage duration would throw
    - parse_float_val handles negative/scientific notation but shares ambiguity with parse_number at call sites
    - No trailing comma tolerance (strict JSON syntax required)
    - No duplicate key detection
    - No line/column error reporting (position-indexed only: reports expected vs actual character)
  - Error reporting: position-based error messages with expected vs actual character

- `03_reference_configs_walkthrough.md`
  - pi05_config.json: full socket mode, 8 users, 768 input tokens, 1 output token, 20 turns
    - Numerical walkthrough: vision 4 stages + text 20 stages + denoise 6*5=30 stages = 54 total effective stages
    - effective_stage_us = (1000+2200+2200+2200 + 20*591 + 30*757) / 54 = total_latency_us / 54
    - Compute total_latency_us for this config
  - pi05_config_hybrid.json: bulk_only mode, 1 user, 768 input, 1 output, 400 turns
    - Single-stage phases: 1+1+5=7 effective stages (single-stage per phase because single-chip cannot pipeline across devices)
    - pcie_bandwidth_gbps=25.0 (informational) vs full's 16.0; designed for sustained PCIe bandwidth measurement with 400 turns
  - pi05_config_sim.json: simulation-only (no socket block), 8 users, same phase layout as full config
    - Same effective stages and timing as full config, but runs entirely in software
  - Key insight: output_tokens=1 models Pi0.5's single-action-token output -- the model produces one action vector per inference (delivered via bulk D2H), not a token sequence; the single token triggers turn completion and the D2H read

### Chapter 3: PipelineSimulator Timing Model

**Description:** Deep dive into the per-token timestamp simulation engine that models systolic pipeline latency, throughput cap, and backpressure without a tick thread.

**Files:**

- `01_simulator_design.md`
  - PipelineSimulator class: the per-token timestamp model (no tick thread)
  - Three invariants:
    1. Latency: numStages * stagePeriod per token
    2. Throughput cap: at most 1/stagePeriod tokens per second
    3. Backpressure: at most numStages tokens in-flight simultaneously
  - inject() path: enter_time = max(now, lastEnter + stagePeriod) for rate limiting; exit_time = enter_time + totalLatency; pushed to FIFO
  - read_result() path: busy-waits until head token's exit_time, then pops; single-reader contract guarantees front element stability during spin
  - Why no tick thread: eliminates tick quantization error (up to stagePeriod of padding on cold sequences) and spin-loop overhead (~1-2us per tick)
  - Trade-off: busy-wait burns CPU but achieves sub-microsecond emit timing vs sleep_until's 5-50us jitter on non-RT kernels
  - How PipelineSimulatorConfig maps to PipelineSimulator constructor parameters (num_stages, stage_period_us, batch_prefill, accept_rate, etc.)

- `02_batch_prefill_behavior.md`
  - batchPrefill flag: when true, non-last prefill tokens (desc.prefill_token_id != EMPTY_TOKEN) complete immediately -- pushed to inflight queue with exit_time = clock::now(), bypassing rate limiting and pipeline latency
  - Why needed for Pi0.5: real hardware processes all prefill tokens in one batch operation, not sequentially through the pipeline stages; only the last prefill token (which triggers the transition to decode) actually traverses the pipeline
  - Impact on TTFT measurement: batch_prefill makes simulated TTFT match hardware TTFT by not penalizing prompt length
  - Interaction with backpressure: immediate-complete prefill tokens still occupy inflight slots momentarily but are consumed rapidly by the reader
  - How PipelineSimulatorConfig::batch_prefill = true propagates through DecodeScheduler to PipelineSimulator::batchPrefill

- `03_token_model_and_spec_decode.md`
  - Token generation model: accept path (actual = token_id + 1, predicted = actual + 1), reject path (actual = token_id + 3)
  - acceptRate: probability of accept vs reject per BASE token; 1.0 = always accept
  - safeVocabModulus: wraps token arithmetic within [safeVocabBase, safeVocabBase + modulus) to prevent drift into stop-token IDs during long generations; must be 0 (disabled) or >= 5
  - Fixed tokenId override: deterministic single-token mode, skips reject roll
  - Relevance to Pi0.5: Pi0.5 uses output_tokens=1 per turn and max_turns for iteration, so spec-decode is typically disabled but the mechanism is inherited from the shared scheduler framework
  - condition variable notification in read_result() for multi-threaded consumers

### Chapter 4: Bulk H2D/D2H Socket Protocol

**Description:** Details the wire protocol for pixel ingestion and action output, including header formats, page-aligned transfer, channel implementations, the sentinel shutdown mechanism, and concurrency hazards.

**Files:**

- `01_wire_protocol.md`
  - BulkH2DHeader (8 bytes): slot_id (uint32_t) + payload_length (uint32_t) -- identifies which user's pixel data follows
  - BulkD2HHeader (8 bytes): slot_id (uint32_t) + payload_length (uint32_t) -- identifies which user's action output follows
  - Page alignment: BULK_PAGE_SIZE=4096; total transfer = ceil((8 + payload_length) / 4096) * 4096 bytes; zero-padded to prevent stale data leakage
  - H2D payload composition: BulkH2DHeader + pixel_data (width * height * channels * frames bytes) + action_history (action_history_bytes)
    - Default config: 8 + 224*224*3*3 + 4096 = 455,176 bytes -> ceil(455176/4096) = 112 pages -> 458,752 bytes padded
  - D2H payload composition: BulkD2HHeader + ACTION_OUTPUT_BYTES (3200 = 50*32 bfloat16 action tensor)
    - Always fits in single page: 8 + 3200 = 3208 < 4096
  - Asymmetry: H2D is multi-page (variable-size pixel payloads), D2H is always single-page (fixed-size action output)
  - Sentinel protocol: slot_id = 0xFFFFFFFF with payload_length = 0, occupies a single page, used to signal kernel shutdown

- `02_bulk_channel_classes.md`
  - BulkH2DChannel: wraps H2DSocket::connect(); send() builds staging buffer with header + payload + zero-pad, calls socket_->write(staging, num_pages); returns transfer duration in microseconds
  - BulkD2HChannel: wraps D2HSocket::connect(); recv() spins on has_data(), reads one page, parses BulkD2HHeader, copies payload to output buffer; includes a 5-second spin-wait warning for detecting stuck kernels
  - send_sentinel(): writes single page with slot_id=0xFFFFFFFF, payload_length=0
  - recv_sentinel_echo(): reads one page, checks if first word is SENTINEL_SLOT_ID
  - Staging buffer reuse: staging_ vector grows to accommodate largest payload, never shrinks; avoids per-send allocation
  - Zero-fill security: staging buffer is memset to 0 before each send to prevent stale data leakage in padding bytes

- `03_shutdown_protocol.md`
  - Ordered shutdown sequence: (1) send sentinel via H2D, (2) barrier on H2D, (3) wait for D2H sentinel echo, (4) stop DecodeScheduler, (5) destroy bulk channels
  - Why ordering matters: sentinel must reach the kernel before DecodeScheduler::stop() tears down the token path; echo confirms kernel received and exited its main loop
  - Sentinel robustness concerns: single sentinel with no retry; if the sentinel page is corrupted or the kernel is stuck in a different wait, shutdown hangs; no timeout on recv_sentinel_echo()
  - Simulation mode teardown (contrast): EVICT all users -> wait for INACTIVE state (with timeout) -> stop scheduler -> normal exit; no sentinel needed

- `04_race_conditions.md`
  - Header/payload atomicity: the header and payload are written as a single multi-page socket write, so the device sees them atomically -- but only because socket_wait_for_pages(receiver, 1) waits for the first page (containing the header) before reading; the 8-byte header always starts at page offset 0, so cross-page header split cannot occur
  - The "pop-before-D2H" ordering in the passthrough kernel: safe because passthrough doesn't need H2D data, but a production kernel MUST defer pops until after data is fully consumed
  - D2H slot_id mismatch detection: the runner logs a WARNING but continues -- this could mask ordering bugs if multiple users complete simultaneously; in bulk_only mode with PipelineSimulator, D2H results arrive in FIFO order but DecodeScheduler completion order depends on simulation timing
  - Potential deadlock: D2H FIFO sized to BULK_PAGE_SIZE * num_users pages -- if the host doesn't read D2H results fast enough, the device kernel blocks on socket_reserve_pages; conversely, if num_users exceeds FIFO capacity, deadlock is guaranteed
  - Missing error propagation: send() and recv() don't check for socket errors or disconnection; a broken socket would cause indefinite blocking

### Chapter 5: Device Launcher and Kernel Architecture

**Description:** Covers the device launcher process lifecycle, MeshDevice setup, socket/kernel creation, the pi05_bulk_passthrough kernel's wire protocol implementation, and pcie_noc_utils.h internals.

**Files:**

- `01_device_launcher_host_side.md`
  - DeviceLauncher class (in pi05_pipeline_runner.cpp): fork() + prctl(PR_SET_PDEATHSIG, SIGTERM) + execl() pattern for orphan prevention
  - Child process: pi05_device_launcher binary; sets TT_VISIBLE_DEVICES=0 in bulk-only mode to prevent topology discovery crash on disconnected N150s
  - wait_for_descriptors(): polling for /dev/shm/tt_{h2d,d2h}_{socket_id}.bin files with 100ms interval until all appear or timeout; premature child exit detection via kill(pid, 0) without reaping
  - stop(): sends SIGTERM, waitpid() to reap; destructor calls stop()
  - is_alive(): kill(pid, 0) without reaping -- avoids premature waitpid that would change process state

- `02_device_launcher_internals.md`
  - pi05_device_launcher.cpp main(): parse args, create DistributedContext (full mode only), create MeshDevice (create_unit_mesh(0) for bulk-only, create(MeshShape{1,1}) for full)
  - Core assignments: token sockets on CoreCoord(0,0), bulk sockets on CoreCoord(1,0) in full mode; both on CoreCoord(0,0) in bulk-only mode
  - Socket creation: H2DSocket with BufferType::L1, configurable FIFO size and H2DMode (HOST_PUSH or DEVICE_PULL); D2HSocket with FIFO sized to BULK_PAGE_SIZE * num_users for deadlock prevention
  - CircularBuffer creation: TOKEN_PAGE_SIZE=256 for token core, BULK_PAGE_SIZE=4096 for bulk core
  - Kernel launch: pipeline_loopback kernel on token core (full mode only), pi05_bulk_passthrough on bulk core; compile_args pass socket config addresses, page size, CB index, and pull_from_host flag
  - MeshWorkload: single program dispatched via EnqueueMeshWorkload with blocking=false, followed by Finish() which blocks until kernel exits
  - _exit(EXIT_SUCCESS): bypasses atexit handlers to avoid ShmResourceTracker double-free race with the parent process; this means destructors don't run, which could leak shared memory in edge cases

- `03_bulk_passthrough_kernel.md`
  - Kernel entry: compile_time_arg_val for socket config addresses, page_size, output_cb_index, pull_from_host flag
  - SocketReceiverInterface and SocketSenderInterface setup, page size configuration
  - Main loop step-by-step:
    1. socket_wait_for_pages(receiver, 1) -- blocks until at least one H2D page arrives
    2. (DEVICE_PULL only) noc_read_page_chunked() pulls first page from host pinned RAM into L1 over PCIe
    3. Parse header: read slot_id and payload_length from first page
    4. Sentinel check: if slot_id == 0xFFFFFFFF, enter shutdown path
    5. Pop first page (safe for passthrough because data is discarded; PRODUCTION KERNEL MUST defer pops)
    6. Consume remaining pages: loop over num_pages-1, wait/pull/pop each
    7. Construct D2H response: write incrementing pattern (w + slot_id) as synthetic action output
    8. NOC write to D2H socket: noc_async_write + noc_async_write_barrier() BEFORE socket_push_pages() (prevents host from reading stale/partial data)
    9. Push D2H page + notify host
  - Sentinel handling: echo sentinel via D2H (write SENTINEL_SLOT_ID to output page, NOC write, barrier, push, notify), then pop the sentinel H2D page AFTER D2H echo is flushed (ordering matters for clean shutdown)
  - Cleanup: update_socket_config for both interfaces, socket_barrier, double noc barrier

- `04_pcie_noc_utils.md`
  - noc_write_page_chunked(): D2H direction, splits L1-to-PCIe write into NOC_MAX_BURST_SIZE chunks using noc_wwrite_with_state; initializes write state before loop
  - noc_read_page_chunked(): H2D direction, splits PCIe-to-L1 read into NOC_MAX_BURST_SIZE chunks using noc_read_with_state; caller must call noc_async_read_barrier() after all reads complete
  - Why chunking is needed: NOC has a maximum burst size for PCIe transactions; exceeding it causes undefined behavior or hardware faults. The NOC_MAX_BURST_SIZE constant defines the hardware-enforced ceiling per DMA transfer
  - NOC_MAX_BURST_SIZE: the value and its hardware rationale; how a developer would verify or update it for different hardware revisions or Tensix generations
  - WARMUP_ITERS constant: 5 iterations, used by benchmark_hd_sockets.cpp for warmup before timed measurement; not directly used by the pi05 kernel but present in the shared header

### Chapter 6: Decode Scheduler Integration

**Description:** Explains how Pi0.5 uses DecodeScheduler with batch_prefill and skip_eos_writeback, the ALLOCATE/SUBMIT/EVICT request lifecycle, the multi-turn fresh-context SUBMIT loop pattern, and user session management.

**Files:**

- `01_scheduler_request_lifecycle.md`
  - How pi05_pipeline_runner creates DecodeScheduler with PipelineSimulatorConfig (simulation/hybrid) or SocketConfig (full mode)
  - Request types: ALLOCATE (get a slot_id), SUBMIT (send prompt tokens + gen params), CONTINUE (multi-turn continuation in standard LLM), EVICT (free slot), STOP
  - Pi0.5 pattern: ALLOCATE all slots upfront -> SUBMIT with prompt tokens per turn -> on completion, SUBMIT again for next turn (fresh context each turn, NOT CONTINUE)
  - Why SUBMIT not CONTINUE: robotics VLA inference treats each camera frame as independent; there is no KV cache continuity between turns; each turn is a fresh prompt with new pixel data
  - SchedulerResponse: contains slot_id assignment and error_code; polled with timeout via poll_response_timed()
  - OutputMessage: contains slot_id, token_id (EMPTY_TOKEN for completion-only messages), is_complete flag, ctx_exhausted flag, tokens_generated count
  - SchedulerParams for Pi0.5: max_users from config, skip_eos_writeback=true always

- `02_batch_prefill_and_skip_eos.md`
  - batch_prefill in DecodeScheduler: passed through PipelineSimulatorConfig to PipelineSimulator constructor; makes non-last prefill tokens complete instantly (see Chapter 3, `02_batch_prefill_behavior.md` for simulator-level details)
  - skip_eos_writeback in SchedulerParams: when true, the stage_eos_writeback lambda in decode_scheduler.cpp returns immediately, suppressing the EOS token inject that normally follows generation completion
  - Why skip_eos_writeback for Pi0.5: standard LLM inference writes EOS back to KV so the next CONTINUE sees the terminator as context; Pi0.5 uses fresh SUBMIT each turn (no CONTINUE), so the EOS writeback is wasted work
  - What would break without skip_eos_writeback: the pipeline would inject an extra token after completion, consuming a pipeline slot and adding unnecessary latency; in the worst case, the stale EOS token could be read back and confuse the post-completion state machine
  - Interaction: with skip_eos_writeback=true and fresh SUBMIT loops, the slot's KV is implicitly reset by each new SUBMIT; the EOS writeback would write into a position that the next SUBMIT's prefill would overwrite anyway

- `03_user_session_management.md`
  - UserSession struct: slot_id, current_turn, total_tokens, ctx_exhausted, done flags; per-turn timing: turn_start, first_token_time, last_token_time, turn_token_count, first_token_received
  - slot_to_user map: maps slot_id back to user index for OutputMessage routing
  - Multi-turn loop: on completion, check if current_turn >= max_turns or ctx_exhausted; if not, generate new pixel data, send bulk H2D with action history from previous turn's D2H output, re-SUBMIT
  - action_history reuse: the runner uses the D2H action output from turn N as the action_history input for turn N+1, creating a temporal dependency chain; if D2H data is corrupted, the corruption propagates through subsequent turns
  - Stagger: initial submissions are staggered by stagger_us (default: total_latency_us / num_users) to avoid thundering herd on pipeline entry; with output_tokens=1, stagger mainly affects TTFT distribution rather than sustained throughput

### Chapter 7: Metrics, Reporting, and Benchmark Orchestration

**Description:** Documents all metrics computed by the framework, how each is measured, output formats, and how the run_hybrid.sh script orchestrates a complete benchmark run.

**Files:**

- `01_timing_metrics.md`
  - TTFT (Time To First Token): per-turn, milliseconds from turn_start (SUBMIT time) to first non-EMPTY_TOKEN OutputMessage; samples collected per turn across all users; reported as mean/median/p99
  - ITL (Inter-Token Latency): global, milliseconds between consecutive non-EMPTY_TOKEN outputs across ALL users; measures system-wide token emission cadence, not per-user decode
  - TPOT (Time Per Output Token): per-turn, (last_token_time - first_token_time) / (turn_token_count - 1); average per-user decode latency excluding TTFT; only computed when turn_token_count > 1
  - Output throughput (tok/s): grand_total_tokens / bench_duration_s; measures aggregate system throughput
  - Peak output TPS: 1-second sliding window maximum over token_timestamps (simulation mode only)
  - Corrected TTFT (simulation mode only): pipeline-only estimate excluding prefill contention
    - single = total_latency_us / 1000
    - mean = (total_latency_us + (num_users - 1) * max_non_vision_stage_us / 2) / 1000
    - Assumes uniform stagger and ignores vision-phase dominance -- an approximation
  - compute_stats(): sorts samples, computes mean, median (n/2 index), p99 (ceil(0.99*n)-1 index)
    - **GOTCHA:** The p99 computation for small sample sizes can equal the median index -- p99 becomes meaningless with fewer than ~100 samples

- `02_bulk_transfer_metrics.md`
  - BulkTransferMetrics struct: vectors of h2d_transfer_us and d2h_transfer_us, cumulative total_h2d_bytes and total_d2h_bytes
  - H2D bandwidth: h2d_payload_bytes / (mean_h2d_us * 1000) -> GB/s; h2d_payload includes padded total (header + pixel_bytes + action_history_bytes + zero-padding)
  - D2H bandwidth: (BulkD2HHeader + ACTION_OUTPUT_BYTES) / (mean_d2h_us * 1000) -> GB/s
  - **NOTE:** pcie_bandwidth_gbps config field is informational only -- it is NOT used in the bandwidth calculation. Bandwidth is measured empirically from actual transfer timing. This prevents confusion about whether reported bandwidth is measured or configured.
  - Socket vs hybrid mode differences: both modes report bulk metrics identically; the difference is whether the token path is real (SocketPipeline) or simulated (PipelineSimulator)

- `03_output_format.md`
  - Simulation mode output: full benchmark table with TTFT/ITL/TPOT mean/median/P99, throughput tok/s and req/s, peak TPS, peak concurrent users, corrected TTFT
  - Socket/hybrid mode output: condensed benchmark table with TTFT/ITL/TPOT plus bulk H2D/D2H timing and bandwidth
  - Per-user summary: total turns, total tokens, status (done, ctx_exhausted)
  - Live status line: Active/Done/Total tokens, refreshed every 100ms via redraw_status()

- `04_run_hybrid_orchestration.md`
  - run_hybrid.sh walkthrough line-by-line:
    - Kill existing processes: pkill -9 pi05_device_launcher and pi05_pipeline_runner_device to clean up stale processes from previous runs
    - Clean /dev/shm files: rm -f /dev/shm/tt_d2h_* /dev/shm/tt_h2d_* /dev/shm/tt_socket_manifest_* to remove stale socket descriptors
    - Set TT_METAL_RUNTIME_ROOT: tells tt-metal where to find its runtime libraries and firmware; required when running outside the tt-metal build tree
    - Set TT_VISIBLE_DEVICES: restricts which Tenstorrent devices are visible to the process; defaults to "1" in the script (second N150)
    - Invoke pi05_pipeline_runner_device with config file path
  - The 2-second sleep after device launcher start: "wait for device kernel init to avoid IOMMU mapping race" -- the IOMMU needs time to set up DMA mappings before the host attempts PCIe transfers; this is a heuristic with no detection/retry mechanism
  - End-to-end workflow: runner starts -> forks device launcher -> launcher creates MeshDevice + sockets + kernels -> exports descriptors to /dev/shm -> runner polls for descriptors -> runner connects to sockets -> benchmark loop -> sentinel shutdown -> teardown
  - Shutdown sequence (socket modes): send H2D sentinel -> H2D barrier -> wait for D2H sentinel echo -> stop DecodeScheduler -> destroy channels -> _exit()
  - Why _exit() instead of normal exit: both runner and launcher use _exit() to skip atexit handlers due to ShmResourceTracker double-free race -- a workaround for a tt-metal teardown bug, not a principled solution

### Chapter 8: Design Critique, Failure Modes, and Gotchas

**Description:** Critically evaluates the framework's design decisions, identifies race conditions and failure modes, catalogs production-vs-passthrough kernel differences, and documents practical operational gotchas.

**Files:**

- `01_design_critique.md`
  - Pipeline flattening loss: collapsing 3 phases with different stage counts/durations into a single (total_stages, avg_stage_us) pair loses phase-boundary timing structure; the simulator cannot model the vision phase completing before text begins; TTFT estimates in simulation mode do not reflect real phase-sequential behavior
  - Corrected TTFT formula: assumes uniform stagger and ignores vision-phase dominance; underestimates actual TTFT variation when the vision stage (2200us) is the bottleneck
  - Custom JSON parser vs standard library: lacks escape handling, negative numbers, floating-point in phase stages, duplicate key detection, and line/column error reporting; fragile for human-edited configs but adequate for a fixed schema with CI-managed config files
  - Single BulkH2DChannel/BulkD2HChannel: all users share one bulk socket pair, serializing pixel transfers; doesn't model real hardware where multiple users might have parallel PCIe paths
  - Fork/exec pattern concerns: tight coupling between runner and launcher binary paths; no IPC beyond /dev/shm descriptor files; no structured error reporting from child to parent (only exit status code is available if MeshDevice::create fails)
  - D2H slot_id mismatch is logged as WARNING but not treated as error: in a multi-user scenario with out-of-order completion, mismatches could indicate a fundamental protocol violation but the benchmark continues anyway

  *Each critique item uses the structure: **Observation** (what the code does), **Concern** (why it might be problematic), **Mitigation** (what prevents the problem in practice or what would fix it).*

- `02_failure_modes.md`
  - Zombie processes: if the runner crashes after fork() but before waitpid(), the device launcher child becomes a zombie; /dev/shm files persist; prctl(PR_SET_PDEATHSIG) mitigates but may not fire on all Linux kernel versions under SIGKILL
  - /dev/shm cleanup: socket descriptors (/dev/shm/tt_{h2d,d2h}_{id}.bin) and manifest files persist across runs; stale files cause "descriptor already exists" errors or wrong-socket connections; run_hybrid.sh manually cleans these before each run
  - IOMMU mapping race: the 2-second sleep is a heuristic; on slow systems or under heavy load, the IOMMU may not have completed DMA setup, causing PCIe transfer failures; no detection or retry mechanism exists
  - PCIe alignment violations: BULK_PAGE_SIZE=4096 matches PCIe TLP payload alignment; if page_size were changed without updating the kernel's circular buffer config, transfers would fail silently or corrupt data
  - TT_VISIBLE_DEVICES interaction: if not set in bulk_only mode, MetalContext attempts topology discovery across all N150s, which crashes on disconnected systems; the code sets it to "0" as a safety net, but this conflicts with run_hybrid.sh's default of "1"
  - Sentinel duplication: if the host sends two sentinels (e.g., due to a retry), the kernel exits after the first; the second sentinel remains in the H2D FIFO, potentially corrupting the next run's socket state
  - FIFO size underrun: if pixel payload grows (higher resolution, more frames), the padded size may exceed fifo_size; the runner validates this but the device launcher does not -- validation gap between host and device side
  - Device launcher premature exit: if MeshDevice::create fails, the child exits and the parent gets a premature-exit error, but the error message from the child is lost (only exit status code available)
  - D2H recv spin-wait: the 5-second timeout warning in BulkD2HChannel::recv() is informational only -- it doesn't abort; if the device kernel is truly stuck, the runner hangs indefinitely with no recovery path

- `03_production_kernel_differences.md`
  - H2D pop timing: passthrough kernel pops pages immediately after reading header; production kernel MUST defer pops until pixel data is fully consumed by the vision encoder, because popping advances the FIFO read pointer and allows the host to overwrite the data with the next user's pixels
  - Signal-wait for denoise: passthrough kernel writes D2H immediately after receiving H2D; production kernel must wait for the denoise loop (5 Euler steps on the action decoder) to complete before writing the actual action output; this requires a Tensix signal-wait mechanism between the data-movement RISC and the compute RISC
  - Synthetic vs real action output: passthrough writes an incrementing pattern (w + slot_id) for validation; production writes actual bfloat16 action tensor from the compute pipeline
  - Core assignment: passthrough uses one core for bulk data movement; production will likely need multiple cores for the full vision+text+denoise pipeline, with inter-core NOC communication

- `04_operational_gotchas.md`
  - /dev/shm socket files: MUST be cleaned before each run (rm -f /dev/shm/tt_d2h_* /dev/shm/tt_h2d_* /dev/shm/tt_socket_manifest_*); stale files cause "descriptor already exists" errors or wrong-socket connections
  - TT_VISIBLE_DEVICES: must be set to restrict which physical devices are visible; disconnected N150s require device 0 only in bulk-only mode; full mode with MPI requires appropriate device mapping per rank
  - TT_METAL_RUNTIME_ROOT: must point to the tt-metal source tree for kernel compilation at runtime; if unset or wrong, kernel compilation fails with cryptic errors about missing includes
  - PCIe alignment: all bulk transfers must be page-aligned (4096 bytes); violating this causes NOC transfer errors or data corruption; the framework enforces this via ceil(total/4096)*4096 padding
  - Process cleanup: run_hybrid.sh kills existing processes with SIGKILL before starting; sleep 2 allows kernel resources to be released; insufficient sleep can cause device initialization failures
  - output_tokens=1: Pi0.5 config uses 1 output token per turn because the action output is delivered via bulk D2H, not the token stream; the single token triggers completion and the D2H read; confusion arises when developers assume output_tokens controls action output size
  - D2H FIFO sizing: D2H FIFO is sized to BULK_PAGE_SIZE * num_users to prevent deadlock; if num_users exceeds FIFO capacity, the kernel blocks on socket_reserve_pages while the host blocks on recv(), creating a circular wait deadlock
  - stagger_us auto-calculation: defaults to total_latency_us / num_users, which spaces submissions to fill the pipeline; but with output_tokens=1, each user completes after a single decode iteration, so stagger mainly affects TTFT distribution rather than sustained throughput
  - action_history chain: D2H output from turn N feeds as action_history input for turn N+1; if D2H data is corrupted (e.g., by a kernel bug), the corruption propagates through all subsequent turns for that user

---

## Conventions

### Terminology

| Term | Definition |
|------|-----------|
| **Phase** | One of the three logical stages of Pi0.5 inference: vision (SigLIP encoder), text (Gemma backbone), or denoise (Euler flow-matching). Each phase has one or more pipeline stages. |
| **Stage** | A single pipeline tick within a phase, with a fixed duration in microseconds. Corresponds to one device in a multi-device pipeline. |
| **Loop count** | Number of times a phase's stages repeat. Used for the denoise phase to model multiple Euler denoising steps. |
| **Effective stages** | num_stages * loop_count for a phase; total_stages = sum of effective stages across all phases. |
| **Slot** | A numbered user context in the DecodeScheduler (0 to max_users-1). Holds KV cache state and generation progress. Allocated once per user, reused across turns. |
| **Turn** | One complete inference cycle for a user: SUBMIT prompt -> receive output tokens -> receive D2H action output. Pi0.5 runs max_turns turns per user session, each with fresh context. |
| **Sentinel** | Special slot_id value (0xFFFFFFFF) used to signal kernel shutdown via the bulk H2D channel. The kernel echoes it via D2H for confirmation. |
| **Passthrough** | The kernel behavior of receiving H2D data and producing synthetic D2H output without performing real computation. Used for benchmarking PCIe transfer performance. |
| **Bulk channel** | BulkH2DChannel or BulkD2HChannel; wraps H2DSocket/D2HSocket for large (multi-page) pixel/action transfers, as opposed to the 256-byte token socket pages. |
| **FIFO** | Circular buffer on the Tensix core used by sockets for H2D/D2H data staging. |
| **DEVICE_PULL** | H2D mode where the device kernel explicitly reads data from host pinned RAM via NOC read commands (noc_read_page_chunked). The kernel must read before pop for correct flow control. |
| **HOST_PUSH** | H2D mode where the host pushes data into device L1 memory via PCIe TLB writes. Data is available in L1 when socket_wait_for_pages returns. |

### Notation

- All durations in the codebase are in **microseconds** unless suffixed with `_ms` (milliseconds) or `_s` (seconds).
- Memory sizes use **bytes** unless otherwise noted; FIFO sizes and page sizes are always in bytes.
- Source code references use the format `filename.cpp:L123` for specific lines and `filename.cpp:func()` for functions.
- Struct fields are referenced as `StructName::field_name`.
- Wire format diagrams use `[field (size)]` notation, e.g., `[slot_id (4B)] [payload_length (4B)] [pixel_data (var)]`.
- File paths are relative to the tt-llm-engine repository root unless otherwise specified.

### Formatting Rules

- Each chapter directory contains numbered markdown files (01_, 02_, ...) read in order.
- Code snippets are wrapped in fenced code blocks with language annotation (```cpp) and annotated with the source file path and approximate line numbers.
- Critical warnings and gotchas are prefixed with **WARNING:** or **GOTCHA:** in bold.
- Design critique items are structured as: **Observation** (what the code does), **Concern** (why it might be problematic), **Mitigation** (what prevents the problem in practice or what would fix it).
- Cross-references to other chapters use the format "(see Chapter N, `filename.md`)".
- Struct definitions are shown as simplified C++ with field-level comments.

---

## Cross-Chapter Dependencies

| Chapter | Depends On | Reason |
|---------|-----------|--------|
| Ch2 (JSON Config) | Ch1 (Architecture) | Config fields (socket.mode, phases) only make sense after understanding the three operating modes and three phases |
| Ch3 (PipelineSimulator) | Ch1 (Architecture), Ch2 (Config) | Simulator parameters (num_stages, stage_period_us) are derived from phase config; batch_prefill is a mode flag from config |
| Ch4 (Bulk Protocol) | Ch1 (Architecture), Ch2 (Config) | Bulk channels only exist in socket modes (Ch1); protocol details reference PixelPayloadConfig dimensions and SocketConfigParams fields (Ch2) |
| Ch5 (Device Launcher/Kernel) | Ch4 (Bulk Protocol), Ch1 (Architecture) | Kernel implements the wire protocol from Ch4; launcher creates sockets and devices described in Ch1 |
| Ch6 (Scheduler Integration) | Ch3 (PipelineSimulator), Ch4 (Bulk Protocol) | Scheduler uses PipelineSimulator or SocketPipeline (Ch3 concepts) and coordinates with bulk channels (Ch4) for multi-turn operation |
| Ch7 (Metrics & Orchestration) | Ch3 (Simulator), Ch4 (Bulk Protocol), Ch6 (Scheduler) | TTFT/ITL/TPOT depend on pipeline timing (Ch3); bulk bandwidth uses channel measurements (Ch4); corrected TTFT depends on batch_prefill (Ch6) |
| Ch8 (Critique & Gotchas) | All prior chapters | Design critique and failure mode analysis requires understanding all components to identify cross-cutting concerns |

**Key specific cross-references:**

- **Ch3 -> Ch2:** PipelineSimulatorConfig values (num_stages, stage_period_us) are computed from PhaseConfig entries defined in Ch2.
- **Ch5 -> Ch4:** The passthrough kernel main loop implements the wire protocol (header parsing, sentinel handling, page management) defined in Ch4.
- **Ch6 -> Ch3:** batch_prefill modifies PipelineSimulator behavior (Ch3). skip_eos_writeback modifies DecodeScheduler's completion handling, which interacts with the pipeline's token injection model.
- **Ch7 -> Ch6:** Multi-turn metrics collection depends on the UserSession management and SUBMIT loop from Ch6.
- **Ch8 -> Ch5, Ch4:** Production kernel differences (Ch8) contrast against passthrough kernel (Ch5) and wire protocol (Ch4) specifics.
