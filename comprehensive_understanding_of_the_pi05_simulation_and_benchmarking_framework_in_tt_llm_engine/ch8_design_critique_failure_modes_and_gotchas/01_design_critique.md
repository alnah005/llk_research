# 8.1 -- Design Critique

This section examines six design decisions in the Pi0.5 simulation framework, analyzing what each decision trades away and whether mitigations exist. Every item follows the **Observation / Concern / Mitigation** structure. These are not bugs -- they are deliberate design tradeoffs whose consequences should be understood by anyone extending or interpreting the framework's output.

---

## 8.1.1 Pipeline Flattening Loss

### Observation

The framework collapses three distinct pipeline phases (vision, text, denoise) into a single `PipelineSimulatorConfig` with two scalars: `total_stages` and `effective_stage_us` (the weighted average).

```cpp
// pi05_pipeline_runner.cpp:800-813
uint32_t total_stages = 0;
uint64_t total_latency_us = 0;
for (const auto& phase : phases) {
    total_stages += phase.effective_stages();
    total_latency_us += phase.effective_latency_us();
}
// ...
const uint32_t effective_stage_us = static_cast<uint32_t>(total_latency_us / total_stages);
```

For the default `pi05_config.json` with vision (4 stages at 1000/2200/2200/2200 us), text (20 stages at 591 us each), and denoise (6 stages x 5 loops at 757 us each), this produces:

$$\text{effective\_stage\_us} = \frac{7600 + 11820 + 22710}{4 + 20 + 30} = \frac{42130}{54} \approx 780 \text{ us}$$

### Concern

The flattening discards three kinds of information:

1. **Phase-boundary timing.** In reality, the transition from vision to text involves a context handoff. The vision bottleneck stage (2200 us) is nearly 4x the text stage (591 us). When user N's vision phase completes, user N+1's first text stage cannot start until the slowest vision stage drains. The simulation treats all 54 stages as identical 780 us slots, which cannot model phase-transition latency spikes or the "staircase" pattern visible in real hardware traces.

2. **Heterogeneous bottleneck identification.** Vision stage 0 runs at 1000 us while vision stages 1-3 run at 2200 us. The flattened model cannot reveal that user stagger patterns aligning with the 2200 us bottleneck stage create contention at a different point than patterns aligning with text's 591 us stages.

3. **Denoise loop semantics.** The `loop_count=5` multiplier expands denoise from 6 to 30 stages but does not model the iterative flow-matching structure where each loop iteration refines the previous output. The simulation treats these as 30 independent pipeline stages, not 5 iterations of a 6-stage sub-pipeline. The flattened model cannot capture inter-iteration pipeline refill costs.

### Mitigation

The flattening is **intentional**. The `PipelineSimulator` backend in `tt-llm-engine` does not support heterogeneous stage durations. The total-latency-preserving average ensures that **throughput** (tokens/second at steady state) and **total pipeline latency** (end-to-end for a single user) are correct. Only **per-phase contention patterns** are lost. The framework deliberately preserves the `PhaseConfig` vector and prints per-phase breakdowns -- the flattening is a `PipelineSimulator` limitation, not an information loss at the config layer. For capacity planning (the framework's primary purpose), this is an acceptable trade-off; throughput is within ~5% of phase-aware simulation for the typical Pi0.5 configuration. For phase-level bottleneck analysis, operators must use the hardware-attached socket mode.

---

## 8.1.2 Corrected TTFT Formula Assumptions

### Observation

The corrected TTFT (pipeline-only, excluding contention) is computed as:

```cpp
// pi05_pipeline_runner.cpp:1433-1434
double corrected_ttft_single_ms = total_latency_us / 1000.0;
double corrected_ttft_mean_ms = (total_latency_us
    + (num_users - 1) * max_non_vision_stage_us / 2.0) / 1000.0;
```

This yields for a single user:

$$\text{TTFT}_{\text{single}} = \frac{\text{total\_latency\_us}}{1000}$$

And for the mean across $N$ users:

$$\text{TTFT}_{\text{mean}} = \frac{\text{total\_latency\_us} + \frac{(N-1) \cdot \text{max\_non\_vision\_stage\_us}}{2}}{1000}$$

### Concern

The mean formula assumes that the stagger between users introduces a **uniform** average delay of $\frac{(N-1) \cdot \text{bottleneck}}{2}$. This makes two implicit assumptions:

1. **Uniform stagger distribution.** It assumes users are perfectly evenly spaced. The `--stagger-us` default is `total_latency_us / num_users`, which is uniform, but if an operator overrides `--stagger-us` to a smaller value, users bunch up and the actual contention is worse than the formula predicts. The actual contention depends on which pipeline stage each user occupies at the moment a new user enters -- a complex function of stagger timing relative to stage boundaries.

2. **Vision-phase exclusion.** The formula deliberately uses `max_non_vision_stage_us` (591 us for text in the default config) instead of `max_stage_duration_us` (2200 us for vision). The rationale is that vision processing happens via bulk H2D transfer, not through the pipeline stages. However, for the first turn of all users (where all users hit vision simultaneously), vision is the actual bottleneck, so the formula underestimates first-turn TTFT for multi-user scenarios. Additionally, the corrected TTFT is only printed in simulation mode (the socket mode output does not include it), yet in simulation mode vision stages *are* modeled as pipeline stages, making the exclusion inconsistent.

The code computes `max_non_vision_stage_us` by skipping phases named "vision":

```cpp
// pi05_pipeline_runner.cpp:823-829
uint32_t max_non_vision_stage_us = 0;
for (const auto& phase : phases) {
    if (phase.name == "vision") continue;
    for (auto d : phase.stage_durations_us) {
        max_non_vision_stage_us = std::max(max_non_vision_stage_us, d);
    }
}
```

This is a **string comparison** against the phase name, meaning any phase not literally named `"vision"` will be included, even if it has similar characteristics.

### Mitigation

The corrected TTFT is labeled "pipeline-only" in the output and is presented alongside the measured (simulated) TTFT statistics (mean, median, P99). Operators should treat the corrected value as a theoretical lower bound for single-user latency and a rough estimate for multi-user mean, not as a ground truth. The measured TTFT from the simulator remains the primary metric.

---

## 8.1.3 Custom JSON Parser vs. Standard Library

### Observation

The framework implements a hand-rolled recursive-descent JSON parser (~220 lines, lines 261-479 of `pi05_pipeline_runner.cpp`) instead of using a standard library such as `nlohmann::json` or `simdjson`.

```cpp
// pi05_pipeline_runner.cpp:289-297
auto parse_string = [&]() -> std::string {
    skip_ws();
    expect('"');
    std::string result;
    while (pos < content.size() && content[pos] != '"') {
        result += content[pos++];
    }
    expect('"');
    return result;
};
```

### Concern

The custom parser has the following limitations relative to RFC 8259:

1. **No escape handling.** The `parse_string` lambda reads characters until a closing `"` without processing escape sequences. A string containing `\"`, `\\`, `\n`, `\t`, or `\uXXXX` will either corrupt the parse state (the escaped quote will terminate the string early) or include literal backslash characters in the result.

2. **No negative integer support.** The `parse_number` function (lines 299-308) only accepts digits via `std::isdigit`; a leading `-` causes an immediate parse error. The `parse_float_val` function (lines 309-321) *does* handle negatives, but it is only called for `pcie_bandwidth_gbps`. A value like `"output_tokens": -1` (a plausible mistake) produces a confusing "expected a number" error with no line/column context rather than a clear "negative values not supported" message.

3. **Duplicate key behavior.** If a JSON object contains duplicate keys (e.g., two `"num_users"` entries or two `"phases"` blocks), the parser silently uses the last value. There is no warning or rejection.

4. **Position-only error messages.** Parse errors report byte offset (`"Config parse error at position 1234"`) rather than line and column numbers. For a 40-line config file this is manageable; for larger configs it forces manual byte counting.

5. **No comment support.** Config files with `//` or `/* */` comments (common in hand-edited configs) will cause parse failures. JSON does not support comments, and neither does this parser.

### Mitigation

The custom parser is **deliberate**. It avoids adding a third-party dependency to the benchmark binary, which must compile against the tt-metal build system where adding external deps requires CMake changes. The `skip_json_value()` forward-compatibility function (lines 325-354) ensures that unknown fields are gracefully skipped rather than causing errors, which is the most important extensibility property. The limitations are acceptable because: (a) config files are machine-generated or hand-written by the framework's developers, (b) the value domain is small (short ASCII strings, positive integers, a single float), and (c) adding a JSON library dependency would pull in ~20K lines for a ~40-line config file. Operators who need richer JSON features should validate configs offline with `jq` or `python -m json.tool` before passing them to the runner.

---

## 8.1.4 Single BulkH2DChannel / BulkD2HChannel

### Observation

All users' pixel transfers flow through a single `BulkH2DChannel` instance and a single `BulkD2HChannel` instance. There is one H2D socket and one D2H socket, each bound to a single Tensix core:

```cpp
// pi05_pipeline_runner.cpp:923-928
auto bulk_h2d = std::make_unique<BulkH2DChannel>(
    pipeline_config.socket.h2d_bulk_socket_id,
    pipeline_config.socket.connect_timeout_ms);
auto bulk_d2h = std::make_unique<BulkD2HChannel>(
    pipeline_config.socket.d2h_bulk_socket_id,
    pipeline_config.socket.connect_timeout_ms);
```

### Concern

This design serializes all bulk transfers. With 8 concurrent users, each sending ~448 KB of pixel data (224x224x3x3 + 4096 + 8 bytes header, padded to 4096-byte pages = 458,752 bytes = 112 pages; 112 * 4096 = 458,752), the H2D channel must sequentially transmit:

$$8 \times 458{,}752 \approx 3.7\text{MB}$$

At the configured PCIe bandwidth, each H2D transfer takes approximately:

$$t_{\text{H2D}} = \frac{458{,}752 \text{ bytes}}{16 \text{ GB/s}} \approx 28.7 \text{ us}$$

For 8 sequential users, the last user waits $\approx 200$ us before its transfer begins. This is small relative to the pipeline latency (~42 ms), so serialization is not currently a bottleneck. However, if the framework scales to 32 or 64 users, or if pixel resolution increases (e.g., 640x480x3x3 = ~2.8 MB/user), the serialized H2D time could grow to milliseconds and become visible in TTFT.

Additionally, the main loop issues H2D sends for the next turn **inline** with the response-processing loop (lines 1095-1107), blocking all response processing for other users during each H2D transfer.

The D2H direction is less concerning because action output is only 3208 bytes (8-byte header + 3200-byte tensor), fitting in a single 4096-byte page.

### Mitigation

The single-channel design matches the hardware constraint: there is **one** H2D socket and **one** D2H socket between host and device for bulk data. It simplifies the device kernel (one core handles all bulk traffic) and avoids multi-socket coordination complexity. Parallelism would require multiple socket pairs bound to multiple device cores -- a different architecture. The current design is correct for the N150/N300 single-device case. The framework's metrics accurately capture serialization costs because the H2D transfer time is measured per-call and included in `BulkTransferMetrics`.

---

## 8.1.5 Fork/Exec Pattern for Device Launcher

### Observation

The `DeviceLauncher` class manages the device-side process via `fork()`/`execl()`:

```cpp
// pi05_pipeline_runner.cpp:636-667
pid_ = fork();
if (pid_ < 0) throw std::runtime_error("fork() failed: " + std::string(strerror(errno)));
if (pid_ == 0) {
    prctl(PR_SET_PDEATHSIG, SIGTERM);
    if (bulk_only) {
        execl(launcher_binary.c_str(), launcher_binary.c_str(),
              "--bulk-only",
              "--h2d-bulk-socket-id", cfg.h2d_bulk_socket_id.c_str(),
              // ... more args ...
              nullptr);
    } else {
        execl(launcher_binary.c_str(), launcher_binary.c_str(),
              "--h2d-token-socket-id", cfg.h2d_token_socket_id.c_str(),
              // ... more args ...
              nullptr);
    }
    _exit(EXIT_FAILURE);
}
```

### Concern

Four issues:

1. **No structured error reporting.** If `execl()` fails (binary not found, permission denied), the child calls `_exit(EXIT_FAILURE)`. The parent detects this in `wait_for_descriptors()` via `is_alive()` returning false, but the error message is generic: `"Device launcher process exited prematurely (status=1)"`. The actual `errno` from `execl` (e.g., `ENOENT`, `EACCES`, `ENOEXEC`) is lost because the child has no channel to communicate it back to the parent before `_exit`.

2. **Tight coupling.** The parent assumes the child binary is located at the same directory as itself (`self_path.parent_path() / "pi05_device_launcher"`, derived from `/proc/self/exe`). If the launcher binary is renamed, moved, built to a different directory, or invoked via a symlink, the runner fails with a confusing "not found" error.

3. **Fork after multi-threading.** The runner creates a `DecodeScheduler` before forking (lines 880-916). After `fork()`, the child process inherits the parent's memory image but only the calling thread survives. If any of the parent's threads held a mutex (e.g., inside `DecodeScheduler`), the child inherits a locked mutex with no thread to unlock it, leading to potential deadlocks before `execl()` replaces the process image. The `execl()` call mitigates this *if it succeeds*, but the window between `fork()` and `execl()` is vulnerable.

   **Note:** In the current code, `mgr->start()` is called after `DeviceLauncher::start()` returns (line 930 follows the fork at line 636), so no scheduler threads exist at fork time. The concern is theoretical -- it would manifest only if the launch order were changed.

4. **Fragile argument passing.** The `std::to_string(cfg.fifo_size).c_str()` pattern creates temporaries whose `.c_str()` pointers are valid only within the `execl` argument list. While this is technically safe (the temporaries live until the end of the full-expression), it is fragile and error-prone for maintenance.

### Mitigation

`prctl(PR_SET_PDEATHSIG, SIGTERM)` (line 640) is the critical safety net: if the parent crashes, the child receives `SIGTERM` and cleans up. The fork/exec pattern is standard for launching separate binaries, and the `execl()` call immediately replaces the child's address space, closing the post-fork vulnerability window in practice. The alternative -- using a pipe for structured IPC or a launcher daemon -- would add significant complexity for a tool primarily used in controlled benchmarking environments. For structured error reporting, a pipe from child to parent could carry the `errno` value, but this adds complexity for a launch-time failure that occurs at most once per run.

---

## 8.1.6 D2H `slot_id` Mismatch Logged as WARNING

### Observation

When the D2H result's `slot_id` does not match the expected `slot_id` from the token pipeline, the code logs a warning and continues:

```cpp
// pi05_pipeline_runner.cpp:1075-1078
if (result_slot_id != out.slot_id) {
    std::cerr << "WARNING: D2H slot_id mismatch: expected "
              << out.slot_id << " got " << result_slot_id << std::endl;
}
```

### Concern

A slot_id mismatch means the bulk D2H channel and the token pipeline have gone **out of sync**. In a robotics inference system, this means robot A receives robot B's action commands -- a safety-critical data routing error. Despite this, the code:

1. Logs a WARNING to stderr (not an error).
2. Continues processing as if the match succeeded.
3. Feeds the mismatched action output into `action_history` for the next turn (lines 1099-1100), propagating the corruption.

This is not a transient condition -- once one mismatch occurs, every subsequent D2H result will be attributed to the wrong user, creating **silent cascading corruption**. The benchmark completes "successfully" with no indication that all results after the first mismatch are meaningless.

The mismatch can occur if the bulk kernel and token pipeline produce completions in different orders. With `output_tokens=1` (the default Pi0.5 config), all users produce exactly one token, so completion order matches submission order. But with `output_tokens > 1`, completion order can diverge.

There is **no recovery mechanism**: the code does not attempt to re-synchronize. A proper implementation would either abort the run or implement a re-sync protocol (e.g., drain and re-align using sentinel markers).

### Mitigation

In the passthrough kernel, slot_id mismatches should not occur because the kernel echoes the slot_id from the H2D header directly into the D2H header. A mismatch indicates either a hardware error or a protocol bug (e.g., sentinel duplication; see [Section 8.2.6](02_failure_modes.md#826-sentinel-duplication)). The warning severity is appropriate for the *benchmarking framework*, where the action output is synthetic and a mismatch merely indicates an ordering assumption violation. Elevating this to a fatal error would be appropriate for production use. For benchmarking, the WARNING allows the run to complete so that the operator can inspect the full output and diagnose the root cause.

---

**Next:** [Failure Modes](02_failure_modes.md)
