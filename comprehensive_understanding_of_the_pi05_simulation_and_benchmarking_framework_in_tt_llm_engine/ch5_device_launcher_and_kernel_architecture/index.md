# Chapter 5: Device Launcher and Kernel Architecture

The Pi0.5 simulation framework splits execution across two OS processes: a **host-side runner** (`pi05_pipeline_runner`) that drives user sessions, scheduling, and bulk channel I/O, and a **device-side launcher** (`pi05_device_launcher`) that creates the MeshDevice, allocates sockets, and launches Tensix kernels. The host cannot talk to Tenstorrent hardware directly; it needs a dedicated child process because tt-metal's `MeshDevice` and socket infrastructure are process-scoped singletons that cannot safely coexist with the host-side `DecodeScheduler` in the same address space. This chapter traces the full lifecycle from `fork()` through kernel execution to `_exit()`, covering the host-side `DeviceLauncher` class, the device launcher binary internals, the `pi05_bulk_passthrough` kernel's data-movement protocol, and the low-level NOC chunking utilities that bridge L1 and PCIe memory.

## Sections

1. [DeviceLauncher: Host-Side Process Management](01_device_launcher_host_side.md)
   `fork()` + `prctl(PR_SET_PDEATHSIG)` orphan prevention, `execl()` dispatch for full vs. bulk-only modes, descriptor polling with premature-exit detection, `stop()` / `is_alive()` semantics, destructor safety.

2. [Device Launcher Internals](02_device_launcher_internals.md)
   `pi05_device_launcher.cpp` argument parsing, `DistributedContext` setup (full mode only), `MeshDevice` creation, core assignments, socket creation with FIFO sizing, `CircularBuffer` configuration, kernel launch via `MeshWorkload`, and `_exit()` for atexit-handler avoidance.

3. [Bulk Passthrough Kernel](03_bulk_passthrough_kernel.md)
   Compile-time argument unpacking, `SocketReceiverInterface` / `SocketSenderInterface` setup, the page-by-page receive-pop-construct-write-push main loop, sentinel echo handling, and cleanup barriers.

4. [PCIe NOC Utilities](04_pcie_noc_utils.md)
   `noc_write_page_chunked()` and `noc_read_page_chunked()` implementations, `NOC_MAX_BURST_SIZE` chunking rationale, barrier responsibilities, and `WARMUP_ITERS`.

---

**Previous:** [Chapter 4 -- Bulk H2D/D2H Socket Protocol](../ch4_bulk_h2d_d2h_socket_protocol/index.md)

**Next:** [Chapter 6 -- Decode Scheduler Integration](../ch6_decode_scheduler_integration/index.md)
