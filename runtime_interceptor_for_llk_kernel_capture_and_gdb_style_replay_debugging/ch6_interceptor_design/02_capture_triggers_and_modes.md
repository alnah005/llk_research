# Capture Triggers and Modes

This file specifies the trigger evaluation logic, capture modes, environment variable interface, device-side `CAPTURE_BREAKPOINT()` macro, WatcherServer integration for failure-triggered capture, ring buffer design, and overhead analysis. All proposals are grounded in the TT-Metal codebase at commit `621b949`.

> **Cross-references:** Ch 2 (state inventory -- data sizes), Ch 3 (LightMetal global toggle), Ch 4 Sec 2 (H4 hook, ~13 KB per dispatch), Ch 5 Sec 1 (Watcher polling, profiler infrastructure, `PAUSE()` macro), File 01 of this chapter (interceptor architecture, `KernelSnapshotCaptureContext` class, primary hook at Site B).

---

## 1. Trigger Architecture Overview

**PROPOSED:** The capture system supports three trigger classes, each with increasing capture scope and runtime cost:

| Mode | Trigger | What is Captured | Overhead | Storage per Invocation |
|------|---------|-----------------|----------|----------------------|
| MetadataOnly | Always-on (when enabled) | Program ID, kernel names, core assignments, compile hashes | <1% | ~300 B |
| Selective | Pattern match on kernel name, program ID, or core coordinate | Full binaries + CB configs + runtime args + optional L1 data | 5-15 ms (no L1) to 75 ms (with L1) | 55 KB - 101 MB |
| FailureTriggered | Watcher assert, dispatch timeout, or device-side `CAPTURE_BREAKPOINT()` | Post-mortem L1 dump + ring buffer flush of recent metadata | Zero until trigger; then ~200 ms | Ring buffer + post-mortem dump |

These modes correspond to `KernelCaptureConfig::Mode` enum defined in File 01.

---

## 2. **PROPOSED:** Environment Variable Interface

Following the TT-Metal convention for debug feature configuration (see `tt_metal/llrt/rtoptions.hpp`, `RunTimeOptions` class, and the Watcher env var entries at lines 134-152 of `rtoptions.cpp`), the capture system is controlled by environment variables prefixed with `TT_METAL_KERNEL_CAPTURE_`. The parsing follows the established pattern in `RunTimeOptions::ParseWatcherEnv()`.

### 2.1 Environment Variable Definitions

```
TT_METAL_KERNEL_CAPTURE         Master enable. Values:
                                  "0" or unset   -> Disabled
                                  "metadata"     -> MetadataOnly mode
                                  "selective"    -> Selective mode
                                  "on-assert"    -> FailureTriggered mode
                                  "1"            -> Alias for "selective"

TT_METAL_KERNEL_CAPTURE_PATTERN Kernel name glob pattern for selective mode.
                                  Default: "*" (capture all)
                                  Example: "matmul*", "*eltwise*", "reader_*"

TT_METAL_KERNEL_CAPTURE_PROGRAM Program ID filter for selective mode.
                                  Default: unset (capture all programs)
                                  Example: "42" (capture only program ID 42)

TT_METAL_KERNEL_CAPTURE_CORE    Target core coordinate (logical) for selective mode.
                                  Syntax: "x,y" (e.g., "3,4")
                                  Default: unset (capture all cores)

TT_METAL_KERNEL_CAPTURE_L1      Include L1 memory dump in capture.
                                  "0" or unset -> no L1 dump
                                  "1"          -> dump CB data regions
                                  "full"       -> dump entire L1

TT_METAL_KERNEL_CAPTURE_MAX     Maximum number of captures per session.
                                  Default: "100"

TT_METAL_KERNEL_CAPTURE_RING    Ring buffer capacity (number of snapshots retained).
                                  Default: "16"

TT_METAL_KERNEL_CAPTURE_DIR     Output directory for .ttksnap files.
                                  Default: "$TT_METAL_HOME/generated/kernel_captures/"
```

### 2.2 **PROPOSED:** Parsing Implementation

The parsing code integrates into the existing `RunTimeOptions` initialization pipeline. Following the pattern where `ParseWatcherEnv()` handles the `TT_METAL_WATCHER` family of variables:

```cpp
// PROPOSED: addition to RunTimeOptions class (rtoptions.hpp)
// New member:
KernelCaptureConfig kernel_capture_config_;

// PROPOSED: addition to RunTimeOptions constructor
// (rtoptions.cpp, following the ParseWatcherEnv() call)

void RunTimeOptions::ParseKernelCaptureEnv() {
    // Master enable
    const char* capture_env = std::getenv("TT_METAL_KERNEL_CAPTURE");
    if (!capture_env || std::string(capture_env) == "0") {
        kernel_capture_config_.mode = KernelCaptureConfig::Mode::Disabled;
        return;
    }

    std::string mode_str(capture_env);
    if (mode_str == "metadata") {
        kernel_capture_config_.mode = KernelCaptureConfig::Mode::MetadataOnly;
    } else if (mode_str == "selective" || mode_str == "1") {
        kernel_capture_config_.mode = KernelCaptureConfig::Mode::Selective;
    } else if (mode_str == "on-assert") {
        kernel_capture_config_.mode = KernelCaptureConfig::Mode::FailureTriggered;
    } else {
        log_warning(
            tt::LogMetal,
            "TT_METAL_KERNEL_CAPTURE: unrecognized value '{}', disabling capture",
            mode_str);
        kernel_capture_config_.mode = KernelCaptureConfig::Mode::Disabled;
        return;
    }

    // Kernel name pattern
    if (const char* pattern = std::getenv("TT_METAL_KERNEL_CAPTURE_PATTERN")) {
        kernel_capture_config_.kernel_name_pattern = pattern;
    } else {
        kernel_capture_config_.kernel_name_pattern = "*";
    }

    // Program ID filter
    if (const char* prog_id = std::getenv("TT_METAL_KERNEL_CAPTURE_PROGRAM")) {
        try {
            kernel_capture_config_.program_id = std::stoull(prog_id);
        } catch (...) {
            log_warning(tt::LogMetal,
                "TT_METAL_KERNEL_CAPTURE_PROGRAM: invalid value '{}', ignoring",
                prog_id);
        }
    }

    // Core coordinate filter (syntax: "x,y")
    if (const char* core_str = std::getenv("TT_METAL_KERNEL_CAPTURE_CORE")) {
        std::string cs(core_str);
        auto comma = cs.find(',');
        if (comma != std::string::npos) {
            try {
                uint32_t x = std::stoul(cs.substr(0, comma));
                uint32_t y = std::stoul(cs.substr(comma + 1));
                kernel_capture_config_.target_core = CoreCoord{x, y};
            } catch (...) {
                log_warning(tt::LogMetal,
                    "TT_METAL_KERNEL_CAPTURE_CORE: invalid value '{}', ignoring",
                    core_str);
            }
        }
    }

    // L1 dump flag
    if (const char* l1_env = std::getenv("TT_METAL_KERNEL_CAPTURE_L1")) {
        std::string l1_str(l1_env);
        kernel_capture_config_.include_l1_dump =
            (l1_str == "1" || l1_str == "full");
    }

    // Max captures
    if (const char* max_env = std::getenv("TT_METAL_KERNEL_CAPTURE_MAX")) {
        try {
            kernel_capture_config_.max_captures = std::stoul(max_env);
        } catch (...) {
            kernel_capture_config_.max_captures = 100;
        }
    }

    // Ring buffer capacity
    if (const char* ring_env = std::getenv("TT_METAL_KERNEL_CAPTURE_RING")) {
        try {
            kernel_capture_config_.ring_buffer_capacity = std::stoul(ring_env);
        } catch (...) {
            kernel_capture_config_.ring_buffer_capacity = 16;
        }
    }

    // Output directory
    if (const char* dir_env = std::getenv("TT_METAL_KERNEL_CAPTURE_DIR")) {
        kernel_capture_config_.output_dir = dir_env;
    } else {
        kernel_capture_config_.output_dir =
            get_root_dir() + "/generated/kernel_captures/";
    }

    log_info(
        tt::LogMetal,
        "Kernel capture enabled: mode={}, pattern='{}', max={}, ring={}",
        static_cast<int>(kernel_capture_config_.mode),
        kernel_capture_config_.kernel_name_pattern,
        kernel_capture_config_.max_captures,
        kernel_capture_config_.ring_buffer_capacity);
}
```

### 2.3 Usage Examples

```bash
# Capture all kernel invocations with full state (selective mode, all kernels)
export TT_METAL_KERNEL_CAPTURE=selective
export TT_METAL_KERNEL_CAPTURE_L1=1

# Capture only matmul kernels on core (3,4) with L1 dump
export TT_METAL_KERNEL_CAPTURE=selective
export TT_METAL_KERNEL_CAPTURE_PATTERN="matmul*"
export TT_METAL_KERNEL_CAPTURE_CORE=3,4
export TT_METAL_KERNEL_CAPTURE_L1=1

# Failure-triggered mode: maintain rolling buffer, dump on assert
export TT_METAL_KERNEL_CAPTURE=on-assert
export TT_METAL_KERNEL_CAPTURE_RING=32
export TT_METAL_WATCHER=1

# Metadata-only mode for lightweight tracing
export TT_METAL_KERNEL_CAPTURE=metadata
export TT_METAL_KERNEL_CAPTURE_MAX=10000
```

---

## 3. **PROPOSED:** Trigger Evaluation Logic

The `should_capture()` method evaluates whether a specific program dispatch matches the configured trigger conditions. This is called on every dispatch when the interceptor is active, so it must be fast.

```cpp
bool KernelSnapshotCaptureContext::should_capture(
    const detail::ProgramImpl& program,
    const std::vector<std::shared_ptr<KernelGroup>>& kernel_groups) const {

    // MetadataOnly always captures (lightweight, no filtering)
    if (config_.mode == KernelCaptureConfig::Mode::MetadataOnly) {
        return true;
    }

    // FailureTriggered mode: always capture metadata into ring buffer
    // (the full capture happens on failure trigger, not here)
    if (config_.mode == KernelCaptureConfig::Mode::FailureTriggered) {
        return true;
    }

    // Selective mode: evaluate filters
    if (config_.mode != KernelCaptureConfig::Mode::Selective) {
        return false;
    }

    // Program ID filter
    if (config_.program_id.has_value() &&
        program.get_id() != config_.program_id.value()) {
        return false;
    }

    // Kernel name pattern filter (glob matching)
    if (!config_.kernel_name_pattern.empty() &&
        config_.kernel_name_pattern != "*") {
        bool name_match = false;
        for (const auto& kg : kernel_groups) {
            for (KernelHandle kh : kg->kernel_ids) {
                auto kernel_ptr = program.get_kernel(kh);
                if (kernel_ptr &&
                    glob_match(config_.kernel_name_pattern, kernel_ptr->name())) {
                    name_match = true;
                    break;
                }
            }
            if (name_match) break;
        }
        if (!name_match) return false;
    }

    // Core coordinate filter
    if (config_.target_core.has_value()) {
        bool core_match = false;
        const CoreCoord& target = config_.target_core.value();
        for (const auto& kg : kernel_groups) {
            for (const auto& cr : kg->core_ranges.ranges()) {
                if (target.x >= cr.start_coord.x && target.x <= cr.end_coord.x &&
                    target.y >= cr.start_coord.y && target.y <= cr.end_coord.y) {
                    core_match = true;
                    break;
                }
            }
            if (core_match) break;
        }
        if (!core_match) return false;
    }

    return true;
}
```

**PROPOSED:** Simple glob matcher supporting `*` and `?` wildcards:

```cpp
static bool glob_match(const std::string& pattern, const std::string& text) {
    size_t pi = 0, ti = 0;
    size_t star_pi = std::string::npos, star_ti = 0;

    while (ti < text.size()) {
        if (pi < pattern.size() && (pattern[pi] == text[ti] || pattern[pi] == '?')) {
            ++pi; ++ti;
        } else if (pi < pattern.size() && pattern[pi] == '*') {
            star_pi = pi++; star_ti = ti;
        } else if (star_pi != std::string::npos) {
            pi = star_pi + 1; ti = ++star_ti;
        } else {
            return false;
        }
    }
    while (pi < pattern.size() && pattern[pi] == '*') ++pi;
    return pi == pattern.size();
}
```

### 3.1 Performance of Trigger Evaluation

The trigger evaluation path for the common case (pattern = `"*"`, no program ID or core filter) is three comparisons and a return. For pattern matching, kernel groups typically contain 1-3 kernels with names like `"reader_unary"`, `"compute_eltwise"`, `"writer_unary"` -- the glob match on short strings is negligible.

**Overhead estimate**: <100 ns per dispatch for pattern matching. The `program.get_kernel(kh)` call is a hash map lookup in `kernels_` (typically 1-5 entries). Total trigger evaluation: <500 ns.

---

## 4. **PROPOSED:** Device-Side `CAPTURE_BREAKPOINT()` Macro

For interactive debugging, a developer can insert `CAPTURE_BREAKPOINT()` in kernel source code. When execution reaches this point, the kernel signals the host to capture state and waits for a resume signal. This follows the existing `PAUSE()` macro pattern from `tt_metal/hw/inc/api/debug/pause.h` (see Ch 5, File 01).

### 4.1 L1 Mailbox Protocol

The mailbox protocol uses a dedicated word in the debug ring buffer region of L1. The existing `debug_ring_buf_msg_t` (see `dev_msgs.h`) provides 32 `uint32_t` entries per core. **PROPOSED:** Reserve entry index 31 for the capture breakpoint protocol:

```
L1 Mailbox Layout (debug_ring_buf_msg_t.data[31]):

  Bits [31:24]: Magic = 0xCA  ("CApture")
  Bits [23:16]: Breakpoint ID (user-specified, 0-255)
  Bits [15:8]:  RISC ID (0=BRISC, 1=NCRISC, 2=TRISC0, 3=TRISC1, 4=TRISC2)
  Bits [7:0]:   Status:
                  0x00 = Idle (no breakpoint active)
                  0x01 = Breakpoint hit (kernel waiting for host)
                  0x02 = Host acknowledged (capture in progress)
                  0x03 = Resume (host signals kernel to continue)
```

Note: This uses the Watcher-polled L1 mailbox mechanism rather than hardware debug registers. This is essential because Wormhole B0 does **not** have `RISC_DBG_CNTL` registers -- only Blackhole and Quasar have the Debug Module. The mailbox-polling approach works on all architectures (WH, BH, Quasar).

### 4.2 Device-Side Macro Implementation

```cpp
// PROPOSED: tt_metal/hw/inc/api/debug/capture_breakpoint.h

#pragma once

#include "hostdevcommon/common_values.hpp"

#ifdef KERNEL_BUILD

// Offset of capture breakpoint word within debug_ring_buf_msg_t
// debug_ring_buf_msg_t has 32 uint32_t entries; we use the last one
#define CAPTURE_BKPT_MAILBOX_OFFSET (31 * sizeof(uint32_t))

// Magic byte identifying a capture breakpoint signal
#define CAPTURE_BKPT_MAGIC 0xCA

// Status values
#define CAPTURE_BKPT_IDLE    0x00
#define CAPTURE_BKPT_HIT     0x01
#define CAPTURE_BKPT_ACK     0x02
#define CAPTURE_BKPT_RESUME  0x03

// Compute the mailbox address (within the debug ring buffer region)
#define CAPTURE_BKPT_ADDR \
    ((volatile uint32_t*)(GET_MAILBOX_ADDRESS_DEV(debug_ring_buf) \
                          + CAPTURE_BKPT_MAILBOX_OFFSET))

// CAPTURE_BREAKPOINT(id):
//   1. Write the "breakpoint hit" signal to the mailbox
//   2. Spin until the host writes "resume"
//   3. Clear the mailbox
//
// Usage in kernel code:
//   CAPTURE_BREAKPOINT(0);   // Breakpoint #0
//   CAPTURE_BREAKPOINT(1);   // Breakpoint #1

#define CAPTURE_BREAKPOINT(id) do {                                          \
    uint8_t risc_id = MY_RISC_ID;  /* Defined per-processor in firmware */  \
                                                                             \
    /* Construct the signal word */                                          \
    uint32_t signal = (CAPTURE_BKPT_MAGIC << 24) |                          \
                      ((uint8_t)(id) << 16) |                                \
                      (risc_id << 8) |                                       \
                      CAPTURE_BKPT_HIT;                                      \
                                                                             \
    /* Write signal to mailbox */                                            \
    volatile uint32_t* mbox = CAPTURE_BKPT_ADDR;                            \
    *mbox = signal;                                                          \
                                                                             \
    /* Memory barrier to ensure write is visible before polling */            \
    asm volatile("fence ow, ir" ::: "memory");                               \
                                                                             \
    /* Spin until host writes RESUME */                                      \
    while ((*mbox & 0xFF) != CAPTURE_BKPT_RESUME) {                         \
        asm volatile("nop; nop; nop; nop;" :::);                             \
    }                                                                        \
                                                                             \
    /* Clear the mailbox for next use */                                      \
    *mbox = 0;                                                               \
    asm volatile("fence ow, ir" ::: "memory");                               \
} while(0)

#endif // KERNEL_BUILD
```

### 4.3 Host-Side Detection (WatcherServer Integration)

The Watcher polling thread already reads debug mailboxes from each core. **PROPOSED:** Extend the per-core polling loop to detect the capture breakpoint magic:

```cpp
// PROPOSED: addition to watcher_device_reader.cpp
// Inside the per-core polling loop where debug_ring_buf is read:

void check_capture_breakpoint(
    IDevice* device,
    const CoreCoord& physical_core,
    const debug_ring_buf_msg_t& ring_buf) {

    uint32_t bkpt_word = ring_buf.data[31];
    uint8_t magic = (bkpt_word >> 24) & 0xFF;

    if (magic != 0xCA) return;  // No capture breakpoint

    uint8_t bkpt_id = (bkpt_word >> 16) & 0xFF;
    uint8_t risc_id = (bkpt_word >> 8) & 0xFF;
    uint8_t status  = bkpt_word & 0xFF;

    if (status != 0x01) return;  // Not in "hit" state

    log_info(tt::LogWatcher,
        "CAPTURE_BREAKPOINT({}) hit on core ({},{}) RISC {}",
        bkpt_id, physical_core.x, physical_core.y, risc_id);

    // Acknowledge: write ACK status to prevent re-triggering during capture
    uint32_t ack_word = (0xCA << 24) | (bkpt_id << 16) | (risc_id << 8) | 0x02;
    device->write_to_device(
        &ack_word, sizeof(uint32_t),
        physical_core,
        ring_buf_base_addr + 31 * sizeof(uint32_t),
        CoreType::WORKER);

    // Trigger capture -- kernel is halted, so L1 is stable
    auto& snap_ctx = KernelSnapshotCaptureContext::get();
    if (snap_ctx.is_active()) {
        snap_ctx.capture_on_failure(
            device, physical_core,
            fmt::format("CAPTURE_BREAKPOINT({}) on RISC {}", bkpt_id, risc_id));
    }

    // Resume the kernel
    uint32_t resume_word =
        (0xCA << 24) | (bkpt_id << 16) | (risc_id << 8) | 0x03;
    device->write_to_device(
        &resume_word, sizeof(uint32_t),
        physical_core,
        ring_buf_base_addr + 31 * sizeof(uint32_t),
        CoreType::WORKER);
}
```

### 4.4 Comparison with Existing `PAUSE()` Macro

| Aspect | `PAUSE()` | `CAPTURE_BREAKPOINT()` |
|--------|-----------|----------------------|
| Host detection | Watcher polls pause flag | Watcher polls `debug_ring_buf_msg_t.data[31]` |
| Host action | Log and optionally auto-unpause | Capture L1 state, then resume |
| Kernel behavior | Spin on pause flag | Spin on mailbox status word |
| State captured | None (just halts) | Full L1 snapshot at halt point |
| Mailbox location | Dedicated pause flag | `debug_ring_buf_msg_t.data[31]` |
| Guard flag | `TT_METAL_WATCHER` + pause enabled | `TT_METAL_KERNEL_CAPTURE=on-assert` |
| Architecture support | All (WH, BH, Quasar) | All (mailbox-based, no RISC_DBG_CNTL needed) |

### 4.5 Usage Example in Kernel Code

```cpp
// In a compute kernel
void MAIN {
    acquire_dst();
    unpack_AB_hw_configure_disaggregated(cb_in0, cb_in1);
    copy_tile_to_dst(cb_in0, 0, 0);
    copy_tile_to_dst(cb_in1, 0, 1);

    CAPTURE_BREAKPOINT(0);  // Host captures L1 state here

    // Math operation
    matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);

    // Pack result
    pack_tile(0, cb_out0);
    release_dst();
}
```

When the breakpoint fires, L1 contains the unpacked input tiles in SRCA/SRCB, the CB contents with input data, and the pre-math state -- exactly what a developer needs to debug a math operation.

---

## 5. **PROPOSED:** Ring Buffer Design

### 5.1 Purpose and Eviction Policy

The ring buffer maintains a bounded set of recent capture snapshots in host memory. In `FailureTriggered` mode, the ring buffer is populated with metadata-only snapshots on every dispatch. When a failure occurs, the entire ring buffer is flushed to disk, providing a "time-travel" view of the N most recent kernel invocations leading up to the failure.

**Eviction policy**: FIFO (oldest evicted first). When the buffer reaches capacity, `push_back` + `pop_front` on the `std::deque`. This is O(1) for both operations.

### 5.2 Memory Budget Calculations

**MetadataOnly mode** (ring buffer of metadata-only snapshots):
- Per invocation: ~300 B (program ID, kernel names, core assignments, compile hashes)
- Ring buffer of 16 entries: ~5 KB
- Ring buffer of 256 entries: ~75 KB

**Selective mode** (ring buffer of full snapshots, no L1 dump):
- Per invocation (single-core compute): ~55 KB
- Ring buffer of 16 entries: ~880 KB
- Per invocation (64-core matmul): ~5 MB (binaries only, no L1)
- Ring buffer of 16 entries: ~80 MB

**Full L1 dump mode** (ring buffer with L1 contents):
- Per invocation (single core, 1 MB L1 on WH): ~1 MB
- Ring buffer of 16 entries: ~16 MB
- Per invocation (64-core, 1.5 MB each on BH): ~96 MB
- Ring buffer of 16 entries: ~1.5 GB (excessive; reduce capacity or disable L1)

### 5.3 **PROPOSED:** Adaptive Ring Buffer with Memory Limit

To prevent excessive memory use, the ring buffer dynamically adjusts its effective capacity based on per-invocation size:

```cpp
void KernelSnapshotCaptureContext::push_to_ring_buffer(
    CapturedInvocation&& invocation) {

    std::lock_guard<std::mutex> lock(mutex_);

    // Calculate approximate size of this invocation
    uint64_t inv_size = sizeof(CapturedInvocation);
    for (const auto& cs : invocation.core_snapshots) {
        for (const auto& b : cs.binaries) {
            inv_size += b.elf_data.size() * sizeof(uint32_t);
        }
        for (const auto& cb : cs.circular_buffers) {
            inv_size += cb.data.size();
        }
        for (const auto& l1 : cs.l1_regions) {
            inv_size += l1.data.size();
        }
    }

    ring_buffer_.push_back(std::move(invocation));
    ring_buffer_total_bytes_ += inv_size;

    // Hard memory limit: 512 MB for ring buffer
    static constexpr uint64_t kMaxRingBufferBytes = 512ULL * 1024 * 1024;

    // Evict oldest entries until under both count and memory limits
    while (ring_buffer_.size() > config_.ring_buffer_capacity ||
           ring_buffer_total_bytes_ > kMaxRingBufferBytes) {
        if (ring_buffer_.empty()) break;

        uint64_t oldest_size = estimate_invocation_size(ring_buffer_.front());
        ring_buffer_.pop_front();
        ring_buffer_total_bytes_ -= oldest_size;
    }
}
```

The dual eviction policy (count-based and memory-based) ensures that the ring buffer stays within bounds even when individual invocations are large (e.g., with L1 data).

---

## 6. **PROPOSED:** WatcherServer Failure-Triggered Capture

### 6.1 Integration Architecture

The Watcher polling thread (see Ch 5, File 01) already detects device-side asserts, NOC sanitization violations, and kernel hangs. The failure-triggered capture extends this with two additions:

1. **Pre-failure state**: The ring buffer contents captured at previous dispatches
2. **Post-failure state**: L1 memory read from the halted core(s)

```
Dispatch Thread                    Watcher Thread
     |                                  |
     v                                  |
 EnqueueProgram()                       |
     |                                  |
     v                                  |
 assemble_device_commands()             |
     |                                  |
     +-- maybe_capture() ------------>  |  (pushes metadata to ring buffer)
     |                                  |
     v                                  |
 HWCommandQueue::enqueue()              |
     |                                  |
     v                                  v
 [kernel executes on device]        poll_device_mailboxes()
     |                                  |
     v                                  v
 [ASSERT fires on core (3,4)]      detect_assert(core=(3,4))
                                        |
                                        v
                                    killed_due_to_error_ = true
                                        |
                                        v
                                    flush ring buffer to disk
                                        |
                                        v
                                    capture_on_failure(device, core=(3,4))
                                        |
                                        v
                                    read L1 via UMD, serialize to .ttksnap
```

### 6.2 `capture_on_failure()` Implementation

The key integration point is `WatcherServer::killed_due_to_error()` (declared in `tt_metal/impl/debug/watcher_server.hpp` line 34) and `exception_message()` (line 36). When the Watcher detects an assert:

```cpp
void KernelSnapshotCaptureContext::capture_on_failure(
    IDevice* device,
    const CoreCoord& failing_core,
    const std::string& failure_reason) {

    log_info(tt::LogMetal,
        "Kernel capture: failure-triggered capture on core ({},{}) reason: {}",
        failing_core.x, failing_core.y, failure_reason);

    CapturedInvocation inv;
    inv.program_id = 0;  // Unknown in post-mortem
    inv.capture_timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    inv.device_id = device->id();

    CapturedInvocation::CoreSnapshot core_snap;
    core_snap.physical_core = failing_core;
    core_snap.logical_core = failing_core;  // May not be recoverable in post-mortem

    // Read the entire L1 memory from the halted core.
    // The core has asserted/halted, so L1 reads are safe.
    //
    // L1 sizes by architecture (from Ch 2, File 03):
    //   Wormhole B0: 1 MB (0x100000 bytes)
    //   Blackhole:   1.5 MB (0x180000 bytes)
    uint32_t l1_size = device->l1_size_per_core();

    CapturedInvocation::CoreSnapshot::L1Region full_l1;
    full_l1.start_address = 0;
    full_l1.size = l1_size;
    full_l1.data.resize(l1_size);

    // Read L1 in 4 KB pages to avoid overwhelming the NOC
    static constexpr uint32_t kPageSize = 4096;
    for (uint32_t offset = 0; offset < l1_size; offset += kPageSize) {
        uint32_t chunk_size = std::min(kPageSize, l1_size - offset);
        device->read_from_device(
            full_l1.data.data() + offset,
            failing_core,
            offset,
            chunk_size,
            CoreType::WORKER);
    }

    core_snap.l1_regions.push_back(std::move(full_l1));
    inv.core_snapshots.push_back(std::move(core_snap));

    // Serialize and write to disk
    std::string path = flush_to_disk(inv);
    log_info(tt::LogMetal,
        "Kernel capture: post-mortem snapshot written to {}", path);
}
```

### 6.3 L1 Readback Timing

Reading L1 from the host occurs over PCIe via the UMD layer (`Cluster::read_from_device()` at `build_Release/include/umd/device/cluster.hpp` line 360):

| Region Size | Read Time | Notes |
|------------|-----------|-------|
| 4 KB (1 page) | ~10 us | Single NOC read transaction |
| 64 KB (16 pages) | ~100 us | Sequential reads |
| 1 MB (full L1 on WH) | ~2.5 ms | Bulk sequential read |
| 1.5 MB (full L1 on BH) | ~3.8 ms | Bulk sequential read |
| 64 x 1.5 MB (64-core full dump) | ~240 ms | Sequential per-core reads |

These times are acceptable for post-mortem capture (the device has already halted). For pre-dispatch L1 capture in selective mode, the 2.5-3.8 ms per-core overhead may be significant for latency-sensitive workloads but is acceptable for debugging scenarios.

---

## 7. **PROPOSED:** Programmatic API

In addition to environment variables, capture can be controlled programmatically from host-side C++ code:

```cpp
// PROPOSED: addition to tt-metalium/experimental/host_api.hpp

namespace tt::tt_metal::experimental {

// Enable selective capture for a specific program
void EnableKernelCapture(Program& program, bool include_l1 = false);

// Enable capture for the next N dispatches globally
void EnableKernelCaptureForNextDispatches(uint32_t count, bool include_l1 = false);

// Disable capture
void DisableKernelCapture();

// Retrieve paths to captured snapshots
std::vector<std::string> GetCapturedSnapshotPaths();

}  // namespace tt::tt_metal::experimental
```

This enables test frameworks to capture specific invocations without environment variable configuration.

---

## 8. **PROPOSED:** Integration with `data_collection.hpp`

The existing `data_collection.hpp` (`tt_metal/impl/dispatch/data_collection.hpp`) provides `RecordDispatchData` (line 35), `RecordKernelGroup` (line 43), and `RecordProgramRun` (line 49). **PROPOSED:** Add a `DISPATCH_DATA_KERNEL_CAPTURE` enum value to the `data_collector_t` enum (lines 19-24) to integrate capture overhead into the existing data collection pipeline:

```cpp
// PROPOSED: addition to data_collector_t enum in data_collection.hpp
// Existing values follow DISPATCH_DATA_* naming convention
enum class data_collector_t : uint8_t {
    DISPATCH_DATA_CB_CONFIG = 0,
    DISPATCH_DATA_SEMAPHORE = 1,
    DISPATCH_DATA_RTARGS = 2,
    DISPATCH_DATA_BINARY = 3,
    DISPATCH_DATA_KERNEL_CAPTURE = 4,   // PROPOSED: New entry
};
```

Capture events also emit Tracy zones (`ZoneScopedN("KernelCapture::capture_dispatch")`) for profiler visibility. This enables unified performance analysis where capture overhead appears alongside dispatch overhead in the same trace timeline.

---

## 9. Scenario Walkthroughs

### 9.1 Scenario: "Capture on Assert"

**Setup:**
```bash
export TT_METAL_WATCHER=100             # Enable Watcher, 100ms poll interval
export TT_METAL_KERNEL_CAPTURE=on-assert
export TT_METAL_KERNEL_CAPTURE_L1=full
```

**What happens:**

1. Workload starts. Interceptor enters `MetadataOnly` mode. Each dispatch records ~300 B of identification data into the ring buffer (default 16 slots).

2. After 47 dispatches, TRISC1 on core (3,4) hits an `ASSERT`. The assert macro writes `line_num` and `tripped` to the `debug_assert_msg_t` mailbox, then spins.

3. Watcher's polling thread detects the assert within 100 ms. `killed_due_to_error()` returns `true`.

4. The interceptor detects the error flag and calls `capture_on_failure()`.

5. Post-mortem capture reads full L1 (1.5 MB for Blackhole) from the hung core via UMD `read_from_device()`. The core is still spinning in the assert handler, so L1 is stable. Time: ~3.8 ms.

6. The ring buffer's last 16 metadata entries are flushed alongside the deep snapshot, providing dispatch history leading up to the failure.

7. Output:
   ```
   [CAPTURE] Assert detected: TRISC1 on core (3,4)
   [CAPTURE] Post-mortem L1 snapshot: 1,572,864 bytes captured
   [CAPTURE] Written: kernel_captures/000047_postmortem_core3_4.ttksnap
   [CAPTURE] Ring buffer flushed: 16 dispatch metadata entries
   ```

### 9.2 Scenario: "Capture Specific Kernel by Name"

**Setup:**
```bash
export TT_METAL_KERNEL_CAPTURE=selective
export TT_METAL_KERNEL_CAPTURE_PATTERN="matmul*"
export TT_METAL_KERNEL_CAPTURE_CORE=3,4
export TT_METAL_KERNEL_CAPTURE_L1=1
```

**What happens:**

1. Every dispatch at Site B, the trigger evaluates if any kernel in the program has `kernel_full_name()` matching the glob `"matmul*"`.

2. For the data movement reader kernel (`reader_matmul_block_sharded`): no match. Cost: ~100 ns for glob evaluation.

3. For the compute kernel (`matmul_compute_<hash>`): match. The interceptor captures the full program state for core (3,4) only (single-core isolation). Binaries (~84 KB), runtime args, CB configs, and L1 CB data regions (~128 KB for 4 CBs).

4. Snapshot is written: ~212 KB uncompressed, ~85 KB with LZ4 compression.

### 9.3 Scenario: "CI Automatic Failure Capture"

**Setup (GitHub Actions):**
```yaml
# PROPOSED: CI pipeline configuration
- name: Run kernel tests
  env:
    TT_METAL_WATCHER: 100
    TT_METAL_KERNEL_CAPTURE: on-assert
    TT_METAL_KERNEL_CAPTURE_L1: 1
    TT_METAL_KERNEL_CAPTURE_DIR: /artifacts/kernel_captures
  run: |
    pytest tests/tt_metal/test_matmul.py -v

- name: Upload kernel captures on failure
  if: failure()
  uses: actions/upload-artifact@v4
  with:
    name: kernel-captures
    path: /artifacts/kernel_captures/
    retention-days: 30
```

This enables developers to download and inspect snapshots from failed CI runs without needing to reproduce the failure locally.

---

## 10. Interaction with Existing Debug Features

### 10.1 Watcher Coexistence

The Watcher disables DMA because the DMA library is not thread-safe (see Ch 5, Sec 1.1). The interceptor's L1 readback also uses non-DMA UMD reads, so both systems can coexist. **PROPOSED:** The interceptor posts a capture request to an async queue when Watcher detects an error; the capture runs on the interceptor's own thread, avoiding CQ mutex + WatcherServer mutex deadlock.

### 10.2 Inspector Coordination

When both Inspector and the interceptor are active, compile events flow through both systems. The Inspector's `program_kernel_compile_finished` callback (`inspector.hpp` line 39) fires first (existing behavior), then the interceptor's binary cache hook (Hook A from File 01) processes the same event. The two systems share no mutable state and require no synchronization.

### 10.3 Profiler Integration

Capture events emit Tracy zones (`ZoneScopedN("KernelCapture::...")`) so that capture overhead is visible in the profiler timeline alongside normal dispatch timing. The `RecordDispatchData` integration (Section 8) feeds the same data collection pipeline.

---

## 11. Performance Overhead Summary

| Mode | Per-Dispatch Overhead | Memory Overhead | Disk I/O |
|------|---------------------|-----------------|----------|
| Disabled | ~5 ns (singleton check + atomic load) | 0 | 0 |
| MetadataOnly | ~1 us (collect names + hashes) | ~300 B/invocation | None (ring buffer only) |
| Selective (no L1) | ~15-50 ms (iterate kernels, copy binaries) | ~55 KB - 5 MB/invocation | Per-capture write |
| Selective (with L1) | ~20-80 ms (add L1 readback) | ~1.5 MB - 100 MB/invocation | Per-capture write |
| FailureTriggered (no fault) | ~1 us per dispatch (metadata) | Ring buffer bounded | None until failure |
| FailureTriggered (on fault) | ~2.5-240 ms (L1 readback) | Post-mortem snapshot | On failure only |

The critical-path overhead for the disabled case is a single function-local-static access (guaranteed thread-safe by C++11) plus an atomic load of the `mode` field. This compiles to approximately two memory loads.

**Mitigation strategies:**
1. **Binary deduplication**: All cores in a `KernelGroup` share the same binary. Store one copy and reference it by hash (64x reduction for 64-core matmul). See File 03.
2. **Shadow dispatch**: Copy the ~13 KB `ProgramCommandSequence` inside the mutex (~2 us), defer serialization to a background thread. The `shared_ptr<Kernel>` refcount prevents ELF lifetime issues.
3. **Lazy L1 readback**: Defer L1 reads to flush time; the structured state from the hook is often sufficient.

---

## Key Takeaways

- **Three capture modes** (MetadataOnly, Selective, FailureTriggered) provide a graduated trade-off between overhead and capture depth. `FailureTriggered` is the recommended default for development -- zero steady-state cost beyond ~1 us metadata capture per dispatch, with automatic deep capture on failure.

- **The device-side `CAPTURE_BREAKPOINT()` macro** uses `debug_ring_buf_msg_t.data[31]` with a 4-byte protocol (magic 0xCA + breakpoint ID + RISC ID + status) to signal the host Watcher thread, which then reads L1 and resumes the kernel. This extends the existing `PAUSE()` mechanism and works on all architectures including Wormhole B0 (which lacks `RISC_DBG_CNTL` registers).

- **The ring buffer** provides "time-travel" debugging by maintaining the N most recent capture snapshots with both count-based and memory-based eviction (capped at 512 MB). When a failure triggers deep capture, the ring buffer is flushed alongside the post-mortem L1 dump.

- **Environment variable parsing** follows the established `TT_METAL_WATCHER` pattern in `RunTimeOptions` (`rtoptions.hpp`), providing a familiar interface for TT-Metal developers. The programmatic C++ API enables test framework integration.

- **WatcherServer integration** uses `killed_due_to_error()` (line 34) as the failure signal. The interceptor posts to an async queue from the Watcher thread, avoiding deadlock between the WatcherServer mutex and CQ mutex.

- **Per-dispatch overhead in the disabled case is approximately 5 ns** (one singleton access + one atomic load); in the MetadataOnly case approximately 1 us. Full capture overhead of 15-80 ms is acceptable for debugging but not for production workloads.
