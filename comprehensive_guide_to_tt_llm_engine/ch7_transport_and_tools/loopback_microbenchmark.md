# The Loopback Microbenchmark

[Prev: Loopback Kernel](loopback_kernel.md) | [Next: DeepSeek Inference Runner](deepseek_inference_runner.md)

---

The loopback microbenchmark exercises the full DecodeScheduler stack -- from user allocation to multi-turn completion -- against a Tensix core running a minimal echo kernel. It measures pure scheduling and transport overhead without any model computation. The system is split into two cooperating processes: a **launcher** that creates sockets and runs the on-device kernel, and a **connector** that drives multi-user multi-turn traffic through the DecodeScheduler.

**Source files:**
- `tools/pipeline/dummy_pipeline_launcher.cpp` -- Kernel launcher process
- `tools/pipeline/dummy_pipeline_connector.cpp` -- Scheduler connector process

---

## Two-Process Architecture

```
  +--------------------------+          +--------------------------+
  |       LAUNCHER           |          |       CONNECTOR          |
  | (dummy_pipeline_launcher)|          | (dummy_pipeline_connector)|
  +--------------------------+          +--------------------------+
  |                          |          |                          |
  |  1. Create MeshDevice    |          |  1. Parse CLI args       |
  |  2. Create H2D socket    |  export  |  2. Create DecodeScheduler|
  |  3. Create D2H socket    | -------> |     with SocketConfig    |
  |  4. Export descriptors    |  socket  |  3. mgr.start()          |
  |  5. Compile kernel       |   IDs    |  4. ALLOCATE users       |
  |  6. Launch kernel        |          |  5. SUBMIT initial reqs  |
  |  7. Finish() (block)     |          |  6. Completion loop:     |
  |                          |          |     - pop OutputMessage  |
  |  Kernel runs until       |          |     - track metrics      |
  |  sentinel received       |          |     - issue CONTINUE     |
  +-----------+--------------+          |  7. CANCEL all users     |
              |                         |  8. Report benchmark     |
              |     Tensix Core         +-----------+--------------+
              |  +------------------+               |
              +->| pipeline_loopback|<--------------+
                 |   H2D -> D2H    |   (via sockets)
                 +------------------+
```

The two-process split mirrors production deployment: the device-side workload and the host-side scheduler are independent executables coordinated only through exported socket descriptors.

---

## The Launcher Process

**Source**: `tools/pipeline/dummy_pipeline_launcher.cpp`

The launcher's job is minimal: create the hardware resources, launch the kernel, and block until it finishes. It does not run any scheduling logic.

### CLI Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--h2d-socket-id` | Yes | -- | String identifier for the H2D socket descriptor |
| `--d2h-socket-id` | Yes | -- | String identifier for the D2H socket descriptor |
| `--fifo-size` | No | 1024 | FIFO buffer size in bytes for both sockets |

### Initialization Sequence

```cpp
// Lines 68-71: Distributed context setup (required even for single-process)
DistributedContext::create(argc, argv);
const auto& world = DistributedContext::get_current_world();
auto local_ctx = world->split(Color(*world->rank()), Key(0));
DistributedContext::set_current_world(local_ctx);
```

The `DistributedContext` is initialized even for non-MPI launches. When launched standalone (without `mpirun`), the context falls back to a single-rank world. The `split` + `set_current_world` pattern creates a local context scoped to this process's rank, which is necessary for socket descriptor export to work correctly with the distributed metadata store.

### Socket and Kernel Setup

```cpp
// Lines 85-95: Device and socket creation
auto mesh_device = MeshDevice::create(MeshDeviceConfig{MeshShape{1, 1}});
const MeshCoreCoord socket_core = {MeshCoordinate(0, 0), CoreCoord(0, 0)};

auto h2d_socket = H2DSocket(mesh_device, socket_core, BufferType::L1,
                             fifo_size, H2DMode::HOST_PUSH);
h2d_socket.export_descriptor(h2d_socket_id);

auto d2h_socket = D2HSocket(mesh_device, socket_core, fifo_size);
d2h_socket.export_descriptor(d2h_socket_id);
```

Key details:
- **Single core**: The kernel runs on core (0,0) of device (0,0). This is sufficient because the loopback kernel processes one page at a time serially.
- **H2DMode::HOST_PUSH**: The host pushes data into the device's L1 memory. This matches SocketPipeline's inject path (see Ch5).
- **Export**: `export_descriptor()` publishes the socket's connection metadata under the given string ID. The connector process uses these IDs to connect.
- **Page size**: The constant `PAGE_SIZE_BYTES = 64` at line 48. This is the non-DeepSeek wire format page size (16 words at 4 bytes each).

### Kernel Compilation and Launch

```cpp
// Lines 98-121
auto program = CreateProgram();

auto output_cb_index = tt::CBIndex::c_0;
auto output_cb_config = CircularBufferConfig(PAGE_SIZE_BYTES,
    {{output_cb_index, tt::DataFormat::UInt32}})
    .set_page_size(output_cb_index, PAGE_SIZE_BYTES);
CreateCircularBuffer(program, socket_core.core_coord, output_cb_config);

CreateKernel(
    program,
    DS_KERNEL_DIR "/pipeline_loopback.cpp",
    socket_core.core_coord,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = {
            static_cast<uint32_t>(h2d_socket.get_config_buffer_address()),
            static_cast<uint32_t>(d2h_socket.get_config_buffer_address()),
            PAGE_SIZE_BYTES,
            output_cb_index,
        }});

auto mesh_workload = MeshWorkload();
mesh_workload.add_program(MeshCoordinateRange(socket_core.device_coord),
                          std::move(program));
EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);
```

A circular buffer of one page (64 bytes) is allocated for the kernel's output staging area. The kernel binary path uses the `DS_KERNEL_DIR` CMake-defined macro to locate `pipeline_loopback.cpp`. After enqueue, the launcher calls `Finish()` which blocks until the kernel exits. The kernel exits when it receives a sentinel page with `slot_id == 0xFFFFFFFF` (see [loopback_kernel.md](loopback_kernel.md)).

---

## The Connector Process

**Source**: `tools/pipeline/dummy_pipeline_connector.cpp`

The connector is the more complex of the two processes. It acts as a combined IS (Inference Server) and traffic generator, exercising the DecodeScheduler through its full public API. It is structurally simpler than the DeepSeek runner -- it uses synthetic prompts only, no batching, and synchronous CANCEL wait -- making it a good reference for understanding the basic IS request lifecycle.

### CLI Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--h2d-socket-id` | Yes | -- | Must match launcher's H2D socket ID |
| `--d2h-socket-id` | Yes | -- | Must match launcher's D2H socket ID |
| `--prompt-len` | Yes | -- | Number of tokens per prompt (synthetic) |
| `--max-decode` | Yes | -- | Maximum decode tokens per turn |
| `--max-turns` | Yes | -- | Number of conversation turns per user |
| `--num-users` | No | 1 | Number of concurrent users |
| `--connect-timeout-ms` | No | 30000 | Socket connection timeout |

### DecodeScheduler Instantiation

```cpp
// Lines 171-174
pm::DecodeScheduler mgr(
    pl::SocketConfig{h2d_id, d2h_id, connect_timeout_ms},
    pm::SchedulerParams{.max_users = num_users});
mgr.start();
```

Note the absence of `use_deepseek_md_format` in the SocketConfig. The loopback kernel implements the **default** (non-DeepSeek) wire format, so the connector uses the default format as well. The `SocketConfig` triggers creation of a `SocketPipeline` backend that connects to the exported socket descriptors. The `connect_timeout_ms` allows time for the launcher to finish creating and exporting sockets, which is essential in MPI launch mode where both processes start simultaneously.

### Phase 1 -- Allocation (lines 181-197)

```cpp
// Fire all ALLOCATE requests
for (uint32_t u = 0; u < num_users; u++) {
    pm::ISRequest req{};
    req.type = pm::RequestType::ALLOCATE;
    req.request_id = req_id++;
    push_request(mgr, req);
}
// Collect all responses
for (uint32_t u = 0; u < num_users; u++) {
    pm::SchedulerResponse resp{};
    if (!poll_response(mgr, resp) || resp.error_code != 0) {
        throw std::runtime_error("ALLOCATE failed for user " + std::to_string(u));
    }
    users[u].slot_id = resp.slot_id;
}
```

All users are allocated upfront in a single batch. Each ALLOCATE returns a `slot_id` via the response queue. The connector builds a `slot_to_user` map for routing OutputMessages back to the correct UserSession.

### Phase 2 -- Initial SUBMIT (lines 216-232)

```cpp
for (uint32_t u = 0; u < num_users; u++) {
    auto prompt = make_prompt(prompt_len, u * max_turns);
    pm::ISRequest req{};
    req.type = pm::RequestType::SUBMIT;
    req.request_id = req_id++;
    req.slot_id = users[u].slot_id;
    req.tokens = prompt;
    req.gen.max_new_tokens = max_decode;
    push_request(mgr, req);
    users[u].turn_start = Clock::now();
}
```

Prompts are synthetic: `make_prompt(len, seed)` generates `[seed+2, seed+3, ..., seed+len+1]`. The `+2` avoids token ID 0 and 1, which could be confused with special tokens.

### Phase 3 -- Completion Loop (lines 241-314)

This is the core event loop. It continuously pops `OutputMessage` structs from the scheduler's output queue and routes them to the correct user session:

```cpp
while (users_done < num_users) {
    pm::OutputMessage out;
    if (!mgr.try_pop_output(out)) {
        std::this_thread::yield();
        continue;
    }
    auto it = slot_to_user.find(out.slot_id);
    // ... metric collection ...
    if (!out.is_complete) continue;
    // Turn complete: issue CONTINUE or mark done
}
```

For each token received:
- If `token_id != EMPTY_TOKEN`, it is a real decode token: update TTFT, ITL, and token counts
- If `is_complete`, the turn has ended. Either issue a CONTINUE for the next turn or mark the user done

### Phase 4 -- Multi-turn CONTINUE (lines 299-313)

```cpp
auto prompt = make_prompt(prompt_len, it->second * max_turns + u.current_turn);
pm::ISRequest req{};
req.type = pm::RequestType::CONTINUE;
req.request_id = req_id++;
req.slot_id = u.slot_id;
req.tokens = prompt;
req.gen.max_new_tokens = max_decode;
push_request(mgr, req);
u.current_turn++;
```

The CONTINUE request reuses the existing KV cache (see [Ch6 multi_turn_continue.md](../ch6_lifecycle_is_integration_kv_tiering/multi_turn_continue.md)). The user's position in the context window accumulates across turns. If the context window is exhausted (`ctx_exhausted` flag on the OutputMessage), the user is marked done regardless of remaining turns.

### Phase 5 -- Cleanup and CANCEL (lines 319-338)

```cpp
for (auto& u : users) {
    pm::ISRequest req{};
    req.type = pm::RequestType::CANCEL;
    req.request_id = req_id++;
    req.slot_id = u.slot_id;
    push_request(mgr, req);
}
// Wait for all users to reach INACTIVE state
for (int i = 0; i < MAX_POLL_ITERATIONS; i++) {
    pm::OutputMessage out;
    while (mgr.try_pop_output(out)) {}
    bool all_inactive = true;
    for (auto& u : users) {
        if (mgr.get_user_state(u.slot_id) != pm::UserState::INACTIVE)
            all_inactive = false;
    }
    if (all_inactive) break;
    std::this_thread::yield();
}
```

The cleanup drains any remaining output messages and polls `get_user_state()` until all users reach INACTIVE. This ensures the deferred cancellation protocol ([Ch6](../ch6_lifecycle_is_integration_kv_tiering/deferred_cancellation.md)) has fully completed before the scheduler is stopped. The DeepSeek runner's pipelined approach (see [deepseek_inference_runner.md](deepseek_inference_runner.md)) avoids this synchronous wait by overlapping CANCEL drain with the next batch's allocation.

---

## Helper Functions

### push_request (line 68-71)

```cpp
void push_request(pm::DecodeScheduler& mgr, const pm::ISRequest& req) {
    while (!mgr.push_request(req)) {
        std::this_thread::yield();
    }
}
```

Spin-yields until the bounded request queue has space. In practice, the queue (capacity 2 * max_users, see Ch6) is rarely full because the scheduler drains it continuously.

### poll_response (line 74-79)

```cpp
bool poll_response(pm::DecodeScheduler& mgr, pm::SchedulerResponse& resp) {
    for (int i = 0; i < MAX_POLL_ITERATIONS; i++) {
        if (mgr.try_pop_response(resp)) return true;
        std::this_thread::yield();
    }
    return false;
}
```

Polls up to 100,000,000 iterations (line 53: `MAX_POLL_ITERATIONS = 100000000`). Returns false only on genuine deadlock or error.

### make_prompt (line 82-88)

```cpp
std::vector<uint32_t> make_prompt(uint32_t len, uint32_t seed) {
    std::vector<uint32_t> prompt(len);
    for (uint32_t i = 0; i < len; i++) {
        prompt[i] = seed + i + 2;
    }
    return prompt;
}
```

Generates a deterministic sequence of non-zero, non-special-token IDs. The seed changes per user and per turn so every prompt is unique.

### compute_stats (line 127-137)

```cpp
Stats compute_stats(std::vector<double>& samples) {
    if (samples.empty()) return {};
    std::sort(samples.begin(), samples.end());
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    size_t n = samples.size();
    return {
        .mean = sum / static_cast<double>(n),
        .median = samples[n / 2],
        .p99 = samples[std::min(n - 1,
            static_cast<size_t>(std::ceil(0.99 * n) - 1))],
    };
}
```

Computes mean, median, and P99 from a mutable sample vector (sorted in place). The P99 index uses `ceil(0.99 * n) - 1`.

---

## UserSession State Tracking

The `UserSession` struct (lines 90-108) tracks per-user state across turns:

| Field | Type | Purpose |
|---|---|---|
| `slot_id` | `uint32_t` | PM-assigned slot (from ALLOCATE response) |
| `current_turn` | `uint32_t` | Number of completed + in-progress turns |
| `total_tokens` | `uint32_t` | Cumulative decode tokens across all turns |
| `ctx_exhausted` | `bool` | Set when OutputMessage reports context overflow |
| `done` | `bool` | Set when all turns complete or context exhausted |
| `turn_start` | `TimePoint` | Clock::now() at SUBMIT/CONTINUE push time |
| `first_token_time` | `TimePoint` | First token arrival for current turn |
| `last_token_time` | `TimePoint` | Most recent token arrival for current turn |
| `turn_token_count` | `uint32_t` | Tokens received in current turn |
| `first_token_received` | `bool` | Reset per turn for TTFT measurement |

---

## Metrics Collection

The connector collects four categories of latency and throughput metrics:

### Time to First Token (TTFT)

Measured per-turn: the wall-clock time from when a SUBMIT or CONTINUE request is pushed to when the first non-EMPTY_TOKEN OutputMessage arrives for that user. TTFT captures the full pipeline latency: API queue transit + prefill scheduling + device prefill + first decode + D2H result delivery.

### Inter-Token Latency (ITL)

Measured globally across all users: the time between any two consecutive token arrivals on the output queue. With the loopback kernel (which has near-zero processing time), ITL reflects the pure scheduling overhead per token.

### Time per Output Token (TPOT)

Measured per-turn per-user: the average time between consecutive tokens within a single turn, computed from first to last token: `(last_token_time - first_token_time) / (turn_token_count - 1)`.

### Throughput Metrics

| Metric | Formula |
|---|---|
| Request throughput (req/s) | `grand_total_turns / bench_duration_s` |
| Output token throughput (tok/s) | `grand_total_tokens / bench_duration_s` |
| Total token throughput (tok/s) | `(total_input_tokens + grand_total_tokens) / bench_duration_s` |
| Peak output token throughput (tok/s) | Max tokens in any 1-second sliding window |

The peak output throughput is computed via a sliding window over the sorted `token_timestamps` vector (lines 376-383):

```cpp
size_t left = 0;
for (size_t right = 0; right < token_timestamps.size(); right++) {
    while (std::chrono::duration<double>(
               token_timestamps[right] - token_timestamps[left]).count() > 1.0) {
        left++;
    }
    peak_output_tps = std::max(peak_output_tps,
        static_cast<uint32_t>(right - left + 1));
}
```

---

## Launch Methods

### MPI Launch (Single Command)

Both processes launched as separate MPI ranks under a single `mpirun` command:

```bash
mpirun -np 1 decode_scheduler/build-full/dummy_pipeline_launcher \
    --h2d-socket-id pm_test_h2d --d2h-socket-id pm_test_d2h \
  : -np 1 decode_scheduler/build-full/dummy_pipeline_connector \
    --h2d-socket-id pm_test_h2d --d2h-socket-id pm_test_d2h \
    --prompt-len 128 --max-decode 64 --max-turns 5 --num-users 16
```

The colon (`:`) in `mpirun` separates the two rank specifications. Each gets `-np 1` (one rank). The `DistributedContext::create(argc, argv)` call in both processes initializes MPI if available, and the `world->split(Color(*world->rank()), Key(0))` isolates each process into its own communicator -- they do not exchange MPI messages; coordination happens only through the socket descriptor export/connect mechanism.

### Standalone Launch (Two Terminals)

Terminal 1 (launcher):
```bash
LD_LIBRARY_PATH=decode_scheduler/build-full \
  decode_scheduler/build-full/dummy_pipeline_launcher \
    --h2d-socket-id pm_test_h2d \
    --d2h-socket-id pm_test_d2h \
    --fifo-size 1024
```

Terminal 2 (connector, started after launcher prints "Kernel launched"):
```bash
LD_LIBRARY_PATH=decode_scheduler/build-full \
  decode_scheduler/build-full/dummy_pipeline_connector \
    --h2d-socket-id pm_test_h2d \
    --d2h-socket-id pm_test_d2h \
    --prompt-len 128 \
    --max-decode 64 \
    --max-turns 5 \
    --num-users 16
```

In standalone mode, the launcher must finish exporting socket descriptors before the connector tries to connect. The connector's `--connect-timeout-ms` (default 30s) provides the necessary grace period.

### Environment Requirements

| Variable | Purpose |
|---|---|
| `LD_LIBRARY_PATH` | Must include tt-metalium shared libraries |
| `TT_METAL_RUNTIME_ROOT` | Required for device initialization if running outside tt-metal build tree |
| `TT_VISIBLE_DEVICES=0` | Optional: restrict to device 0 for reproducible results |

---

## Structural Comparison with DeepSeek Runner

The loopback connector is the historical predecessor of the DeepSeek runner. Both share the same core structure:

```
  main()
    +-- Parse args
    +-- Create DecodeScheduler(SocketConfig, SchedulerParams)
    +-- mgr.start()
    +-- ALLOCATE loop
    +-- SUBMIT loop (initial turn)
    +-- Completion loop:
    |   +-- try_pop_output()
    |   +-- Record metrics
    |   +-- On complete: CONTINUE or mark done
    +-- CANCEL loop
    +-- Print per-user summary
    +-- Print benchmark stats
    +-- mgr.stop()
```

The DeepSeek runner adds: tokenizer integration, batched execution, pipelined teardown, spec-decode support, thinking-token detection, per-batch metrics, and decoded output display. See the comparison table in [deepseek_inference_runner.md](deepseek_inference_runner.md).

---

[Prev: Loopback Kernel](loopback_kernel.md) | [Next: DeepSeek Inference Runner](deepseek_inference_runner.md)
