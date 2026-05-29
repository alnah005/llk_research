# 7.4 -- `run_hybrid.sh` Orchestration

The `run_hybrid.sh` script is the top-level entry point for running a Pi0.5 benchmark with real hardware. It handles process cleanup, environment setup, and launching the device-aware runner binary. Despite being only ~15 lines, every line exists for a specific reason tied to tt-metal's runtime constraints.

**Source:** `examples/run_hybrid.sh`

---

## 7.4.1 Complete Script

```bash
# examples/run_hybrid.sh:1-15
#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"

pkill -9 -f pi05_device_launcher 2>/dev/null || true
pkill -9 -f pi05_pipeline_runner_device 2>/dev/null || true
sleep 2
rm -f /dev/shm/tt_d2h_* /dev/shm/tt_h2d_* /dev/shm/tt_socket_manifest_* /dev/shm/pi05_*

TT_METAL_RUNTIME_ROOT=/localdev/salnahari/testing_dir/tt-metal \
TT_VISIBLE_DEVICES="${TT_VISIBLE_DEVICES:-1}" \
  "$DIR/build-full/pi05_pipeline_runner_device" \
  --config "$DIR/examples/pi05_config_hybrid.json"
```

---

## 7.4.2 Line-by-Line Walkthrough

### Shell Options (line 2)

```bash
set -euo pipefail
```

| Flag | Meaning |
|------|---------|
| `-e` | Exit immediately on any non-zero return code |
| `-u` | Treat unset variables as errors (prevents silent `$UNDEFINED` expansion) |
| `-o pipefail` | A pipeline fails if any command in it fails (not just the last) |

These are strict-mode defaults that prevent silent failures.

### Working Directory (line 4)

```bash
DIR="$(cd "$(dirname "$0")/.." && pwd)"
```

Resolves `DIR` to the repository root (one level above the `examples/` directory where the script lives). `$0` is the script path, `dirname "$0"` gives `examples/`, and `..` navigates up to the repo root. The result is an absolute path used to locate both the binary and config file, making the script location-independent.

### Kill Existing Processes (lines 6--7)

```bash
pkill -9 -f pi05_device_launcher 2>/dev/null || true
pkill -9 -f pi05_pipeline_runner_device 2>/dev/null || true
```

**Why:** tt-metal's device runtime acquires exclusive access to Tenstorrent PCIe devices. If a previous run crashed or was interrupted, stale processes may still hold device handles, preventing new runs from initializing. The `-9` flag sends `SIGKILL` (unblockable) because a hung process waiting on IOMMU or socket I/O may not respond to `SIGTERM`.

- `-f` matches the full command line, not just the process name
- `2>/dev/null` suppresses "no process found" errors
- `|| true` prevents the script from exiting (due to `set -e`) when no matching processes exist

**Both binaries are killed:**

- `pi05_device_launcher` -- the device-side fork that creates sockets and runs kernels
- `pi05_pipeline_runner_device` -- the host-side benchmark runner compiled with `PI05_HAS_SOCKETS=1`

### Sleep for Cleanup (line 8)

```bash
sleep 2
```

**Why:** After killing previous processes with `SIGKILL`, the kernel needs time to:

1. Release PCIe device file descriptors held by the killed processes
2. Tear down IOMMU mappings associated with those processes
3. Clean up shared memory segments that were memory-mapped

The 2-second sleep provides a buffer for the kernel to fully reclaim these resources before the new process attempts to acquire them. Without this delay, the new process may encounter `EBUSY` or stale IOMMU entries. The value is empirically determined.

### Clean Shared Memory (line 9)

```bash
rm -f /dev/shm/tt_d2h_* /dev/shm/tt_h2d_* /dev/shm/tt_socket_manifest_* /dev/shm/pi05_*
```

tt-metal's socket infrastructure uses POSIX shared memory files as the communication substrate. Four categories of files are cleaned:

| Pattern | Purpose |
|---------|---------|
| `tt_d2h_*` | Device-to-Host socket descriptor files (exported by `export_descriptor()`) |
| `tt_h2d_*` | Host-to-Device socket descriptor files |
| `tt_socket_manifest_*` | Socket manifest files used by the distributed runtime |
| `pi05_*` | Any Pi0.5-specific shared state |

**Why explicit cleanup:** `SIGKILL` does not invoke `atexit()` handlers or destructors. tt-metal's `ShmResourceTracker` normally cleans these up on graceful exit, but after a forced kill, stale descriptor files remain. If the new process finds these files, it may attempt to connect to non-existent sockets, causing hangs or crashes. The `-f` flag prevents errors when no matching files exist.

### Environment Variables and Launch (lines 11--14)

```bash
TT_METAL_RUNTIME_ROOT=/localdev/salnahari/testing_dir/tt-metal \
TT_VISIBLE_DEVICES="${TT_VISIBLE_DEVICES:-1}" \
  "$DIR/build-full/pi05_pipeline_runner_device" \
  --config "$DIR/examples/pi05_config_hybrid.json"
```

**Environment variables:**

| Variable | Value | Purpose |
|----------|-------|---------|
| `TT_METAL_RUNTIME_ROOT` | `/localdev/.../tt-metal` | Tells tt-metal where to find firmware blobs, kernel binaries, and device configuration. Required for any tt-metal operation. |
| `TT_VISIBLE_DEVICES` | `${TT_VISIBLE_DEVICES:-1}` | Controls which Tenstorrent devices are visible, analogous to `CUDA_VISIBLE_DEVICES`. Default `1` selects device index 1. The `${..:-1}` syntax allows the caller to override by exporting the variable before invoking the script. |

**Binary:** `pi05_pipeline_runner_device` is the same source (`pi05_pipeline_runner.cpp`) compiled with `PI05_HAS_SOCKETS=1`, enabling the socket/hybrid code paths. It lives in `build-full/` (the full-feature build directory, as opposed to `build/` for simulation-only).

**Config:** `pi05_config_hybrid.json` is a pipeline configuration file with a `"socket"` block and `"mode": "bulk_only"`, activating hybrid mode (simulated decode + real bulk transfers).

---

## 7.4.3 End-to-End Workflow

The following diagram traces the complete lifecycle from `run_hybrid.sh` invocation through benchmark completion:

```
run_hybrid.sh
    |
    v
[1] Kill stale processes (pkill -9)
[2] Sleep 2s (IOMMU cleanup)
[3] Remove stale /dev/shm descriptors
    |
    v
pi05_pipeline_runner_device --config pi05_config_hybrid.json
    |
    +---> [4] parse_pipeline_config()
    |         reads JSON, populates PipelineConfig + SocketConfigParams
    |
    +---> [5] DecodeScheduler(PipelineSimulatorConfig)   [hybrid: simulated decode]
    |
    +---> [6] DeviceLauncher::start()
    |         |
    |         +---> fork()
    |         |     |
    |         |     child: prctl(PR_SET_PDEATHSIG, SIGTERM)
    |         |     child: execl("pi05_device_launcher", --bulk-only, ...)
    |         |             |
    |         |             +--> create H2D/D2H sockets
    |         |             +--> create & enqueue kernels
    |         |             +--> export_descriptor() --> /dev/shm/tt_h2d_*.bin
    |         |             +--> export_descriptor() --> /dev/shm/tt_d2h_*.bin
    |         |             +--> spin-wait for host to send sentinel
    |         |
    |         parent: poll /dev/shm/ for descriptor files (100ms interval)
    |         parent: return when all descriptors exist (30s timeout)
    |
    +---> [7] sleep(2000ms)    [IOMMU mapping race guard]
    |
    +---> [8] BulkH2DChannel::connect(socket_id)
    +---> [9] BulkD2HChannel::connect(socket_id)
    |
    +---> [10] mgr.start()     [start DecodeScheduler thread]
    |
    +---> [11] ALLOCATE slots for all users
    |
    +---> [12] for each user: bulk_h2d->send() + SUBMIT tokens
    |          (staggered by stagger_us between users)
    |
    +---> [13] main event loop:
    |          while (users_done < num_users):
    |            try_pop_output() --> measure TTFT/ITL/TPOT
    |            on turn complete: bulk_d2h->recv()
    |            if more turns: bulk_h2d->send() + re-SUBMIT
    |            redraw_status() every 100ms
    |
    +---> [14] shutdown:
    |          bulk_h2d->send_sentinel()
    |          bulk_h2d->barrier()
    |          bulk_d2h->recv_sentinel_echo()
    |          mgr.stop()
    |          destroy bulk channels
    |
    +---> [15] print results table
    |
    +---> [16] _exit(EXIT_SUCCESS)
```

---

## 7.4.4 The IOMMU Mapping Race (2-second sleep)

After `DeviceLauncher::start()` returns (indicating descriptor files exist), the runner sleeps for 2 seconds:

```cpp
// pi05_pipeline_runner.cpp:920-921
// Wait for device kernel init to avoid IOMMU mapping race.
std::this_thread::sleep_for(std::chrono::milliseconds(2000));
```

**Why:** The device launcher exports socket descriptors as soon as sockets are created, but the device kernel may still be initializing IOMMU page table entries for those sockets. If the host-side `connect()` call arrives before IOMMU mappings are fully established, the first `write()` to the socket can trigger an IOMMU fault (typically manifesting as a PCIe error or silent data corruption). The 2-second sleep is a conservative empirical guard.

This is the same 2-second window as the `sleep 2` in `run_hybrid.sh` line 8, but for a different purpose: the script's sleep guards against stale state from killed processes, while the C++ sleep guards against premature connection to freshly-created sockets.

---

## 7.4.5 The DeviceLauncher Fork/Exec Pattern

The `DeviceLauncher` class (lines 632--725) manages the device-side process:

**Fork and death signal:** The parent process forks, and the child immediately calls `prctl(PR_SET_PDEATHSIG, SIGTERM)` to ensure it dies when the parent exits:

```cpp
// pi05_pipeline_runner.cpp:639-640
// Child: die when parent exits (prevents orphan device launchers)
prctl(PR_SET_PDEATHSIG, SIGTERM);
```

This prevents orphan device processes that hold IOMMU mappings. Without this, a crashed parent would leave an orphan device launcher holding exclusive device handles, requiring manual `pkill` (which is exactly what `run_hybrid.sh` lines 6--7 handle as a fallback).

**Exec:** The child `execl()`s `pi05_device_launcher` with socket IDs and configuration. For `bulk_only` mode:

```cpp
// pi05_pipeline_runner.cpp:647-654
execl(launcher_binary.c_str(), launcher_binary.c_str(),
      "--bulk-only",
      "--h2d-bulk-socket-id", cfg.h2d_bulk_socket_id.c_str(),
      "--d2h-bulk-socket-id", cfg.d2h_bulk_socket_id.c_str(),
      "--h2d-mode", cfg.h2d_mode.c_str(),
      "--fifo-size", std::to_string(cfg.fifo_size).c_str(),
      "--num-users", std::to_string(cfg.num_users_for_launch).c_str(),
      nullptr);
```

**Descriptor polling:** The parent polls for descriptor files in `/dev/shm/` with a 100ms interval and a `connect_timeout_ms` deadline (default 30 seconds). It also checks `is_alive()` on each iteration to detect premature child death:

```cpp
// pi05_pipeline_runner.cpp:696-721
while (Clock::now() < deadline) {
    if (!is_alive()) {
        int status;
        waitpid(pid_, &status, 0);
        ...
        throw std::runtime_error("Device launcher process exited prematurely ...");
    }
    bool all_present = std::all_of(descriptors.begin(), descriptors.end(),
        [](const auto& d) { return std::filesystem::exists(d); });
    if (all_present) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}
```

---

## 7.4.6 The Sentinel Shutdown Protocol

After the benchmark loop completes, a structured shutdown sequence ensures both processes exit cleanly:

1. **H2D sentinel send:** The host sends a page with `slot_id = 0xFFFFFFFF` and `payload_length = 0`:

```cpp
// pi05_pipeline_runner.cpp:1132-1134
bulk_h2d->send_sentinel();
bulk_h2d->barrier();
```

2. **D2H sentinel echo:** The host waits for the device kernel to echo back a sentinel:

```cpp
// pi05_pipeline_runner.cpp:1137-1142
bool echo_ok = bulk_d2h->recv_sentinel_echo();
```

3. **DecodeScheduler stop:** `mgr->stop()` halts the simulated (or real) pipeline.

4. **Channel destruction:** Bulk channels are explicitly reset before exit.

---

## 7.4.7 Why `_exit()` Instead of Normal Exit

The socket/hybrid code path terminates with `_exit(EXIT_SUCCESS)` instead of returning from `main()`:

```cpp
// pi05_pipeline_runner.cpp:1194-1196
// Use _exit() to skip atexit handlers -- tt-metal's ShmResourceTracker
// cleanup races with socket teardown causing double-free.
_exit(EXIT_SUCCESS);
```

**The problem:** tt-metal registers `atexit` handlers through its `ShmResourceTracker` that clean up shared memory segments. When sockets have already been destroyed (step 4 above), these handlers attempt to double-free the same shared memory, causing segfaults or corruption.

**The solution:** `_exit()` (from `<cstdlib>`) bypasses all `atexit` handlers, C++ static destructors, and `stdio` buffer flushing. The OS reclaims all resources (file descriptors, memory mappings, shared memory) when the process terminates. This is safe because:

- All meaningful output has been flushed (results table printed with `std::endl`)
- The sentinel protocol has already ensured the device side is shutting down
- The `DeviceLauncher` destructor (which calls `stop()` -> `kill` + `waitpid`) will NOT run, but the child process has `PR_SET_PDEATHSIG` set, so it receives `SIGTERM` when the parent's process table entry is cleaned up

**Simulation mode does not use `_exit()`** -- it returns normally from `main()` (line 1458, `return EXIT_SUCCESS`) because there are no socket resources to conflict with.

---

## 7.4.8 Summary of Safety Mechanisms

| Mechanism | Location | Guards Against |
|-----------|----------|---------------|
| `pkill -9` | `run_hybrid.sh:6-7` | Stale processes from previous crashed runs |
| `sleep 2` | `run_hybrid.sh:8` | Kernel resource release lag after SIGKILL |
| `rm /dev/shm/tt_*` | `run_hybrid.sh:9` | Stale socket descriptors causing connect-to-nowhere |
| `prctl(PR_SET_PDEATHSIG)` | `pi05_pipeline_runner.cpp:639` | Orphan device launcher after parent crash |
| `sleep(2000ms)` | `pi05_pipeline_runner.cpp:920` | IOMMU mapping race on fresh sockets |
| `_exit()` | `pi05_pipeline_runner.cpp:1196` | Double-free in ShmResourceTracker vs. socket teardown |

---

**Next:** [Chapter 8 -- Design Critique, Failure Modes, and Gotchas](../ch8_design_critique_failure_modes_and_gotchas/index.md)
