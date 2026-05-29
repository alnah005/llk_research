# 5.1 DeviceLauncher: Host-Side Process Management

> **Source file:**
> - `examples/pi05_pipeline_runner.cpp` (lines 632--725, `DeviceLauncher` class)

The `DeviceLauncher` class lives inside `pi05_pipeline_runner.cpp` (guarded by `#ifdef PI05_HAS_SOCKETS`) and manages the lifecycle of a child process that owns all Tenstorrent hardware resources. The design addresses a fundamental constraint: tt-metal's `MeshDevice` and socket infrastructure are process-scoped singletons that cannot safely coexist with the host-side `DecodeScheduler` in the same address space when socket descriptors must be exported via `/dev/shm`. The solution is `fork()`-and-`exec()`: the runner spawns a child that creates sockets, exports descriptors, launches kernels, and blocks until the kernels finish.

---

## 5.1.1 Class Overview

`DeviceLauncher` has a minimal interface with three public methods and a destructor:

```cpp
// examples/pi05_pipeline_runner.cpp:632-725
class DeviceLauncher {
public:
    void start(const std::string& launcher_binary, const SocketConfigParams& cfg);
    void stop();
    bool is_alive() const;
    ~DeviceLauncher() { stop(); }

private:
    void wait_for_descriptors(const SocketConfigParams& cfg);
    pid_t pid_ = -1;
    bool launched_ = false;
};
```

The class tracks only two pieces of state: the child PID and a `launched_` flag. The `launched_` flag prevents double-stop and double-wait scenarios. The destructor unconditionally calls `stop()`, ensuring the device process is terminated even if the host exits abnormally (e.g., via an uncaught exception).

---

## 5.1.2 start(): Fork, Orphan Prevention, and Exec

The `start()` method creates the child process and dispatches it into one of two exec paths depending on the socket mode:

```cpp
// examples/pi05_pipeline_runner.cpp:634-671
void start(const std::string& launcher_binary, const SocketConfigParams& cfg) {
    bool bulk_only = (cfg.mode == "bulk_only");
    pid_ = fork();
    if (pid_ < 0) throw std::runtime_error("fork() failed: " + std::string(strerror(errno)));
    if (pid_ == 0) {
        // Child: die when parent exits (prevents orphan device launchers)
        prctl(PR_SET_PDEATHSIG, SIGTERM);
        // ...
    }
    // Parent: poll for descriptor files
    launched_ = true;
    wait_for_descriptors(cfg);
}
```

### Orphan Prevention with prctl

The `prctl(PR_SET_PDEATHSIG, SIGTERM)` call is the first action in the child process. This Linux-specific system call instructs the kernel to deliver SIGTERM to the child if the parent process dies for any reason -- crash, `kill -9`, or normal exit. Without this guard, a crash in the host runner would leave the device launcher process running indefinitely, holding the Tenstorrent device open and preventing subsequent runs.

The call must happen **before** `execl()` because `prctl(PR_SET_PDEATHSIG)` is preserved across exec (a Linux kernel guarantee). The signal choice of SIGTERM (rather than SIGKILL) gives the device launcher a chance to run its own cleanup, though in practice `pi05_device_launcher` calls `_exit()` which skips atexit handlers anyway.

**Ordering subtlety:** There is a TOCTOU window between `fork()` and `prctl()` -- if the parent dies during this nanosecond-scale interval, the child will not receive the death signal. In practice, the parent is not doing anything dangerous during this interval.

Combined with the destructor calling `stop()`, this provides two layers of orphan prevention:

1. **Normal teardown**: Destructor calls `stop()` which sends `SIGTERM`.
2. **Parent crash**: Kernel delivers `SIGTERM` to child via the death signal.

### Environment Setup for Bulk-Only Mode

In bulk-only mode, the child sets `TT_VISIBLE_DEVICES=0` if not already set:

```cpp
// examples/pi05_pipeline_runner.cpp:643-646
if (bulk_only) {
    // Restrict to single N150 to prevent topology discovery crash.
    if (!getenv("TT_VISIBLE_DEVICES")) {
        setenv("TT_VISIBLE_DEVICES", "0", 1);
    }
```

This environment variable restricts the UMD (User-Mode Driver) to device index 0 only. On systems with multiple disconnected N150 accelerators, the MetalContext/ControlPlane would otherwise attempt topology discovery across all devices, which crashes because disconnected chips cannot form a valid mesh topology. The `getenv()` guard allows an operator to override this default from the environment.

In full mode, `TT_VISIBLE_DEVICES` is not set because the `DistributedContext` + MPI environment handles multi-device coordination.

### execl() Dispatch: Full vs. Bulk-Only

The child process calls `execl()` with different argument lists depending on the mode:

**Bulk-only mode** (2 sockets, no token path):
```cpp
// examples/pi05_pipeline_runner.cpp:647-654
execl(launcher_binary.c_str(), launcher_binary.c_str(),
      "--bulk-only",
      "--h2d-bulk-socket-id", cfg.h2d_bulk_socket_id.c_str(),
      "--d2h-bulk-socket-id", cfg.d2h_bulk_socket_id.c_str(),
      "--h2d-mode", cfg.h2d_mode.c_str(),
      "--fifo-size", std::to_string(cfg.fifo_size).c_str(),
      "--num-users", std::to_string(cfg.num_users_for_launch).c_str(),
      nullptr);
```

**Full mode** (4 sockets: 2 token + 2 bulk):
```cpp
// examples/pi05_pipeline_runner.cpp:656-665
execl(launcher_binary.c_str(), launcher_binary.c_str(),
      "--h2d-token-socket-id", cfg.h2d_token_socket_id.c_str(),
      "--d2h-token-socket-id", cfg.d2h_token_socket_id.c_str(),
      "--h2d-bulk-socket-id", cfg.h2d_bulk_socket_id.c_str(),
      "--d2h-bulk-socket-id", cfg.d2h_bulk_socket_id.c_str(),
      "--h2d-mode", cfg.h2d_mode.c_str(),
      "--fifo-size", std::to_string(cfg.fifo_size).c_str(),
      "--num-users", std::to_string(cfg.num_users_for_launch).c_str(),
      nullptr);
```

The difference is that full mode passes four socket IDs (two token, two bulk) while bulk-only passes two (bulk only) plus the `--bulk-only` flag. Both modes forward `--h2d-mode`, `--fifo-size`, and `--num-users`.

If `execl()` fails (e.g., binary not found), control falls through to `_exit(EXIT_FAILURE)` on line 667. Using `_exit()` instead of `exit()` avoids running `atexit` handlers registered by the parent's address space (which the child inherited via `fork()`).

### Launcher Binary Discovery

The runner locates the device launcher binary relative to its own executable:

```cpp
// examples/pi05_pipeline_runner.cpp:907-913
auto self_path = std::filesystem::read_symlink("/proc/self/exe");
auto launcher_path = self_path.parent_path() / "pi05_device_launcher";
if (!std::filesystem::exists(launcher_path)) {
    throw std::runtime_error(
        "Device launcher not found at " + launcher_path.string() +
        ". Build it first (cmake --build ... --target pi05_device_launcher).");
}
```

This uses `/proc/self/exe` rather than `argv[0]` because `argv[0]` may be a relative path that becomes invalid after a `chdir()`. The launcher must be in the same directory as the runner binary.

---

## 5.1.3 wait_for_descriptors(): Polling for Socket Readiness

After forking, the parent process must wait for the device launcher to create its sockets and export their descriptors. The tt-metal socket layer writes descriptor files to `/dev/shm/` following the naming convention:

$$\texttt{/dev/shm/tt\_\{type\}\_\{socket\_id\}.bin}$$

where `{type}` is either `h2d` or `d2h`, and `{socket_id}` is the user-supplied string.

```cpp
// examples/pi05_pipeline_runner.cpp:693-721
void wait_for_descriptors(const SocketConfigParams& cfg) {
    auto deadline = Clock::now() + std::chrono::milliseconds(cfg.connect_timeout_ms);
    std::vector<std::string> descriptors;
    // tt-metal export_descriptor writes to /dev/shm/tt_{type}_{socket_id}.bin
    if (cfg.mode != "bulk_only") {
        descriptors.push_back("/dev/shm/tt_h2d_" + cfg.h2d_token_socket_id + ".bin");
        descriptors.push_back("/dev/shm/tt_d2h_" + cfg.d2h_token_socket_id + ".bin");
    }
    descriptors.push_back("/dev/shm/tt_h2d_" + cfg.h2d_bulk_socket_id + ".bin");
    descriptors.push_back("/dev/shm/tt_d2h_" + cfg.d2h_bulk_socket_id + ".bin");
```

The polling loop checks for two conditions every 100ms:

1. **Premature child exit**: Calls `is_alive()` to detect if the device launcher crashed during initialization.
2. **All descriptors present**: Uses `std::filesystem::exists()` on each expected `.bin` file.

```cpp
// examples/pi05_pipeline_runner.cpp:703-718
while (Clock::now() < deadline) {
    if (!is_alive()) {
        int status;
        waitpid(pid_, &status, 0);
        pid_ = -1;
        launched_ = false;
        throw std::runtime_error(
            "Device launcher process exited prematurely (status=" +
            std::to_string(WEXITSTATUS(status)) + ")");
    }
    bool all_present = std::all_of(descriptors.begin(), descriptors.end(),
        [](const auto& d) { return std::filesystem::exists(d); });
    if (all_present) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}
stop();
throw std::runtime_error("Timed out waiting for socket descriptors");
```

The premature-exit detection is critical: without it, the host would spin for the full `connect_timeout_ms` (default 30 seconds) before reporting failure, even though the child crashed in the first millisecond. When premature exit is detected, the method reaps the child via `waitpid()` and extracts the exit status with `WEXITSTATUS()` for the error message.

The 100ms poll interval is a pragmatic choice: fast enough to detect descriptor arrival within ~100ms of creation, slow enough to avoid burning CPU cycles. In practice, device initialization (MeshDevice creation, socket allocation, kernel compilation) takes 5-15 seconds, so the first several iterations of the loop will find no descriptors.

If the timeout expires without all descriptors appearing, `stop()` is called to kill the child, then an exception is thrown.

### Descriptor Count by Mode

| Mode | Descriptors waited for | Count |
|------|----------------------|-------|
| `"full"` | `tt_h2d_{token}`, `tt_d2h_{token}`, `tt_h2d_{bulk}`, `tt_d2h_{bulk}` | 4 |
| `"bulk_only"` | `tt_h2d_{bulk}`, `tt_d2h_{bulk}` | 2 |

---

## 5.1.4 stop(): SIGTERM and Reaping

The `stop()` method sends SIGTERM to the child and blocks until it exits:

```cpp
// examples/pi05_pipeline_runner.cpp:673-680
void stop() {
    if (launched_ && pid_ > 0) {
        kill(pid_, SIGTERM);
        int status;
        waitpid(pid_, &status, 0);
        pid_ = -1;
        launched_ = false;
    }
}
```

Key design choices:

- **SIGTERM, not SIGKILL**: Allows the device launcher to handle the signal gracefully, though `pi05_device_launcher` does not install a signal handler (it relies on the default SIGTERM behavior which terminates the process).
- **Blocking waitpid**: The parent blocks until the child is fully reaped. This prevents zombie processes and ensures the device is fully released before the host continues.
- **Idempotency**: The `launched_ && pid_ > 0` guard makes `stop()` safe to call multiple times, which is important because the destructor calls it unconditionally.
- **No SIGKILL escalation**: If the child ignores or blocks SIGTERM, the parent will block indefinitely. This is acceptable because `pi05_device_launcher` either exits promptly from its `Finish()` call or calls `_exit()` in its exception handler.

---

## 5.1.5 is_alive(): Non-Reaping Liveness Check

```cpp
// examples/pi05_pipeline_runner.cpp:685-688
bool is_alive() const {
    if (!launched_ || pid_ <= 0) return false;
    return kill(pid_, 0) == 0;
}
```

The `kill(pid, 0)` idiom sends signal 0 (the null signal) to the process. It does not actually deliver a signal; the kernel simply checks whether the process exists and the caller has permission to signal it. This is deliberately chosen over `waitpid(pid, &status, WNOHANG)` because `waitpid` *reaps* the child, consuming its exit status. If `is_alive()` reaped the child, the subsequent `waitpid()` in `wait_for_descriptors()` would fail with `ECHILD`, and the exit status would be lost.

The `kill(pid, 0)` approach has one subtle limitation: it returns `true` for zombie processes (children that have exited but not been reaped). In this codebase, that is not a problem because the only caller is `wait_for_descriptors()`, which calls `waitpid()` immediately after detecting a dead child.

---

## 5.1.6 Post-Launch Synchronization

After `DeviceLauncher::start()` returns (meaning all descriptor files are present), the runner adds a 2-second sleep before connecting the bulk channels:

```cpp
// examples/pi05_pipeline_runner.cpp:920-921
// Wait for device kernel init to avoid IOMMU mapping race.
std::this_thread::sleep_for(std::chrono::milliseconds(2000));
```

This delay works around an IOMMU mapping race: the socket descriptors may be exported before the device kernel's PCIe IOMMU mappings are fully established. If the host connects and begins DMA transfers before IOMMU setup completes, the transfers will fault. The 2-second sleep is empirical -- long enough for IOMMU setup to complete on observed hardware configurations, but a pragmatic workaround rather than a proper synchronization barrier.

---

**Next:** [Device Launcher Internals](02_device_launcher_internals.md)
