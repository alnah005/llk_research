# 5.2 Device Launcher Internals

> **Source file:**
> - `examples/pi05_device_launcher.cpp` (lines 1--234)

The `pi05_device_launcher` binary is the device-side counterpart to the host runner. It is a standalone executable that creates a `MeshDevice`, allocates sockets on specific Tensix cores, configures circular buffers, launches kernels via a MeshWorkload, and blocks until the kernels complete. The design is driven by two constraints: (1) tt-metal resources are process-scoped and must be created in the process that will own them, and (2) the launcher must work in two distinct hardware topologies (full multi-chip and bulk-only single-chip). This section walks through the binary from `main()` entry to `_exit()`.

---

## 5.2.1 Argument Parsing and Mode Detection

The launcher accepts command-line arguments passed by the host runner's `execl()` call. Mode detection happens first:

```cpp
// examples/pi05_device_launcher.cpp:77
bool bulk_only = has_flag(argc, argv, "--bulk-only");
```

The remaining arguments are parsed with defaults that match the host-side `SocketConfigParams`:

```cpp
// examples/pi05_device_launcher.cpp:101-107
std::string h2d_token_id = get_arg(argc, argv, "--h2d-token-socket-id", "pi05_h2d_token");
std::string d2h_token_id = get_arg(argc, argv, "--d2h-token-socket-id", "pi05_d2h_token");
std::string h2d_bulk_id  = get_arg(argc, argv, "--h2d-bulk-socket-id", "pi05_h2d_bulk");
std::string d2h_bulk_id  = get_arg(argc, argv, "--d2h-bulk-socket-id", "pi05_d2h_bulk");
std::string h2d_mode_str = get_arg(argc, argv, "--h2d-mode", "DEVICE_PULL");
uint32_t fifo_size       = static_cast<uint32_t>(std::stoul(
    get_arg(argc, argv, "--fifo-size", "524288")));
uint32_t num_users       = static_cast<uint32_t>(std::stoul(
    get_arg(argc, argv, "--num-users", "8")));
```

Note that the token socket IDs are parsed even in bulk-only mode (they get their defaults), but they are never used. This keeps the argument-parsing code simple.

---

## 5.2.2 TT_VISIBLE_DEVICES and Topology Guard

In bulk-only mode, the launcher restricts UMD device visibility as a safety net:

```cpp
// examples/pi05_device_launcher.cpp:85-89
if (bulk_only) {
    const char* visible = std::getenv("TT_VISIBLE_DEVICES");
    if (!visible) {
        setenv("TT_VISIBLE_DEVICES", "0", 1);
    }
}
```

This duplicates the same guard that the host runner sets before `execl()`. The duplication exists because the device launcher can also be invoked standalone (e.g., under `mpirun` for full mode, or directly for debugging). Without `TT_VISIBLE_DEVICES=0`, MetalContext's ControlPlane attempts topology discovery across all devices on the system. On a machine with 8 disconnected N150 accelerators, these devices cannot form a valid mesh topology, and the discovery process crashes with an `std::out_of_range` exception. The `getenv` check ensures a user-specified value takes precedence.

---

## 5.2.3 DistributedContext: Full Mode Only

Multi-host coordination is only needed when running under MPI for the full (SocketPipeline + bulk) mode:

```cpp
// examples/pi05_device_launcher.cpp:95-99
if (!bulk_only) {
    DistributedContext::create(argc, argv);
    const auto& world = DistributedContext::get_current_world();
    auto local_ctx = world->split(Color(*world->rank()), Key(0));
    DistributedContext::set_current_world(local_ctx);
}
```

The `DistributedContext::create()` initializes the MPI communicator. The `world->split(Color(*world->rank()), Key(0))` call creates a per-rank communicator where each rank is in its own color group -- effectively isolating each rank's device operations. This is the standard tt-metal pattern for isolating per-device communication within a multi-host environment: the SocketPipeline requires MPI for cross-chip communication, but each device launcher manages only its local device.

In bulk-only mode, the launcher runs as a plain forked child with no MPI environment (the child was forked, not launched via `mpirun`), so `DistributedContext` is skipped entirely.

---

## 5.2.4 MeshDevice Creation

The MeshDevice is created differently for each mode:

```cpp
// examples/pi05_device_launcher.cpp:120-128
std::shared_ptr<MeshDevice> mesh_device;
try {
    mesh_device = bulk_only
        ? MeshDevice::create_unit_mesh(0)
        : MeshDevice::create(MeshDeviceConfig{MeshShape{1, 1}});
} catch (const std::out_of_range& e) {
    std::cerr << "\nMeshDevice::create failed with out_of_range: " << e.what() << std::endl;
    throw;
}
```

| Mode | Factory method | Semantics |
|------|---------------|-----------|
| `bulk_only` | `MeshDevice::create_unit_mesh(0)` | Creates a mesh with a single device (device index 0); no topology discovery |
| `full` | `MeshDevice::create(MeshDeviceConfig{MeshShape{1, 1}})` | Creates a 1x1 mesh through the full ControlPlane topology discovery path |

The `create_unit_mesh(0)` path is specifically designed for scenarios where multi-chip topology discovery would fail -- such as 8 disconnected N150s where only one is accessible. The `out_of_range` catch block provides a targeted error message for topology failures, which are the most common crash mode on disconnected multi-device systems.

---

## 5.2.5 Core Assignments

The launcher assigns kernels to specific Tensix cores:

```cpp
// examples/pi05_device_launcher.cpp:132-134
const CoreCoord token_core(0, 0);
const CoreCoord bulk_core(bulk_only ? CoreCoord(0, 0) : CoreCoord(1, 0));
const MeshCoordinate device_coord(0, 0);
```

| Kernel | Full Mode Core | Bulk-Only Core | Rationale |
|--------|---------------|----------------|-----------|
| Token loopback (`pipeline_loopback.cpp`) | `CoreCoord(0, 0)` | Not launched | Latency-critical token path |
| Bulk passthrough (`pi05_bulk_passthrough.cpp`) | `CoreCoord(1, 0)` | `CoreCoord(0, 0)` | Throughput-critical bulk path |

**Why separate cores in full mode?** Each Tensix core has a single set of socket configuration registers. The token path and bulk path have fundamentally different page sizes (256 vs. 4096 bytes), different transfer patterns (single-page token inject/result vs. multi-page pixel frames), and different latency requirements (token path is latency-critical, bulk is throughput-critical). Running them on separate cores eliminates any contention for L1 bandwidth, circular buffer space, and NOC resources -- if they shared a core, they would contend for the core's single NOC write path.

**Why the same core in bulk-only mode?** When the token loopback kernel is not launched (no SocketPipeline), the token core is unused. The bulk kernel runs on `CoreCoord(0, 0)` instead of `(1, 0)` because `create_unit_mesh(0)` may expose a different core mapping than the full topology path. Using `(0, 0)` is the safest choice -- it is always the first valid Tensix core.

---

## 5.2.6 Socket Creation and FIFO Sizing

### Token Sockets (Full Mode Only)

```cpp
// examples/pi05_device_launcher.cpp:137-148
std::optional<H2DSocket> token_h2d;
std::optional<D2HSocket> token_d2h;
if (!bulk_only) {
    token_h2d.emplace(mesh_device,
        MeshCoreCoord{device_coord, token_core},
        BufferType::L1, 1024, H2DMode::HOST_PUSH);
    token_h2d->export_descriptor(h2d_token_id);

    token_d2h.emplace(mesh_device,
        MeshCoreCoord{device_coord, token_core}, 1024);
    token_d2h->export_descriptor(d2h_token_id);
}
```

Token sockets use `std::optional` wrappers so they can be conditionally created. Key parameters:

| Parameter | H2D Token | D2H Token |
|-----------|-----------|-----------|
| Core | `token_core` = (0,0) | `token_core` = (0,0) |
| `BufferType` | `L1` | (implicit L1) |
| FIFO size | 1024 bytes = 4 pages of 256B | 1024 bytes = 4 pages of 256B |
| `H2DMode` | `HOST_PUSH` | N/A (D2H is always device-push) |

The `HOST_PUSH` mode for token H2D means the host writes token pages directly into device L1 via PCIe MMIO. This is appropriate for the token path because token pages are small (256 bytes = 1 PCIe TLP) and latency matters more than throughput.

### Bulk Sockets

```cpp
// examples/pi05_device_launcher.cpp:152-163
H2DMode bulk_mode = (h2d_mode_str == "DEVICE_PULL")
    ? H2DMode::DEVICE_PULL : H2DMode::HOST_PUSH;
auto bulk_h2d = H2DSocket(mesh_device,
    MeshCoreCoord{device_coord, bulk_core},
    BufferType::L1, fifo_size, bulk_mode);
bulk_h2d.export_descriptor(h2d_bulk_id);

// D2H FIFO sized to num_users pages (deadlock prevention)
uint32_t d2h_fifo_size = BULK_PAGE_SIZE * num_users;
auto bulk_d2h = D2HSocket(mesh_device,
    MeshCoreCoord{device_coord, bulk_core}, d2h_fifo_size);
bulk_d2h.export_descriptor(d2h_bulk_id);
```

Key design decisions for bulk sockets:

- **H2D FIFO size**: Configurable via `--fifo-size`, default 524288 bytes (512 KB). For the default pixel configuration ($224 \times 224 \times 3 \times 3 = 451{,}584$ bytes pixel data + 4096 bytes action history + 8 bytes header = 455,688 bytes, padded to $\lceil 455{,}688 / 4096 \rceil \times 4096 = 458{,}752$ bytes), the 512 KB FIFO has room for exactly one transfer with ~67 KB headroom.
- **D2H FIFO size**: Set to `BULK_PAGE_SIZE * num_users` -- one page per user. This sizing is a deadlock prevention measure: if all users complete their pipeline execution simultaneously, the device kernel must be able to write all $N_{\text{users}}$ D2H result pages without blocking. If the FIFO were smaller, the kernel could block on `socket_reserve_pages()` while the host is blocked waiting for a D2H read -- a classic circular-wait deadlock.
- **H2D mode**: Defaults to `DEVICE_PULL`, where the device kernel initiates the PCIe read. This achieves higher throughput for large transfers because the device's NOC read engine can saturate the PCIe link with optimal burst sizes, whereas host pushes are limited by PCIe MMIO write coalescing.

### D2H FIFO Sizing Formula

For $N$ users and a page size of $P$ bytes:

$$\text{D2H FIFO size} = P \times N$$

With the default $P = 4096$ and $N = 8$:

$$\text{D2H FIFO size} = 4096 \times 8 = 32768 \text{ bytes} = 32 \text{ KB}$$

The D2H response is always exactly 1 page (8-byte header + 3200-byte action output = 3208 bytes < 4096 bytes), so one page per user is the minimum that guarantees no back-pressure deadlock.

### export_descriptor()

After creating each socket, `export_descriptor()` writes the socket's connection metadata to `/dev/shm/tt_{type}_{id}.bin`. These are the files that the host runner's `wait_for_descriptors()` polls for. The descriptor contains the FIFO base address, size, PCIe BAR mapping information, and flow-control pointers that the remote side needs to connect.

---

## 5.2.7 CircularBuffer Configuration

Each kernel needs a circular buffer (CB) for staging output data before NOC writes:

```cpp
// examples/pi05_device_launcher.cpp:175-178 (token kernel CB)
auto token_cb_config = CircularBufferConfig(
    TOKEN_PAGE_SIZE, {{0, tt::DataFormat::UInt32}})
    .set_page_size(0, TOKEN_PAGE_SIZE);
CreateCircularBuffer(program, token_core, token_cb_config);
```

```cpp
// examples/pi05_device_launcher.cpp:193-196 (bulk kernel CB)
auto bulk_cb_config = CircularBufferConfig(
    BULK_PAGE_SIZE, {{0, tt::DataFormat::UInt32}})
    .set_page_size(0, BULK_PAGE_SIZE);
CreateCircularBuffer(program, bulk_core, bulk_cb_config);
```

The page size constants are defined at the top of the launcher:

```cpp
// examples/pi05_device_launcher.cpp:50-51
static constexpr uint32_t TOKEN_PAGE_SIZE = 256;   // wire_format.hpp PAGE_SIZE_BYTES
static constexpr uint32_t BULK_PAGE_SIZE  = 4096;  // PCIe-aligned bulk transfer pages
```

| CB | Buffer size | Page size | Data format | Index |
|----|-------------|-----------|-------------|-------|
| Token | 256 B | 256 B (1 page) | `UInt32` | 0 |
| Bulk | 4096 B | 4096 B (1 page) | `UInt32` | 0 |

Both CBs hold exactly one page. This is sufficient because both kernels process pages sequentially -- they never need to buffer more than one output page at a time. The `UInt32` data format is used because the kernels manipulate raw 32-bit words (headers, slot IDs, payload data) -- there is no compute involved, only data movement.

---

## 5.2.8 Kernel Creation and Compile-Time Arguments

### Token Loopback Kernel (Full Mode Only)

```cpp
// examples/pi05_device_launcher.cpp:179-189
CreateKernel(program, std::string(DS_KERNEL_DIR) + "/pipeline_loopback.cpp",
    token_core, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = std::vector<uint32_t>{
            token_h2d->get_config_buffer_address(),
            token_d2h->get_config_buffer_address(),
            TOKEN_PAGE_SIZE,
            0,  // output_cb_index
        },
    });
```

### Bulk Passthrough Kernel

```cpp
// examples/pi05_device_launcher.cpp:197-208
CreateKernel(program, std::string(DS_KERNEL_DIR) + "/pi05_bulk_passthrough.cpp",
    bulk_core, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = std::vector<uint32_t>{
            bulk_h2d.get_config_buffer_address(),
            bulk_d2h.get_config_buffer_address(),
            BULK_PAGE_SIZE,
            0,  // output_cb_index
            static_cast<uint32_t>(bulk_mode == H2DMode::DEVICE_PULL),
        },
    });
```

Both kernels use `DataMovementProcessor::RISCV_0` with the default NOC (NOC 0). The compile-time arguments are positional:

| Index | Token Loopback | Bulk Passthrough | Purpose |
|-------|---------------|------------------|---------|
| 0 | H2D config address | H2D config address | `SocketReceiverInterface` init |
| 1 | D2H config address | D2H config address | `SocketSenderInterface` init |
| 2 | 256 | 4096 | Page size |
| 3 | 0 | 0 | Output CB index |
| 4 | N/A | 0 or 1 | `pull_from_host` flag |

The `pull_from_host` flag is the critical difference: it controls whether the bulk kernel performs NOC reads from PCIe (`DEVICE_PULL`) or expects data to already be in L1 (`HOST_PUSH`). The token loopback kernel always uses `HOST_PUSH` mode and does not have this argument.

---

## 5.2.9 MeshWorkload Launch and Blocking

```cpp
// examples/pi05_device_launcher.cpp:213-219
auto mesh_workload = MeshWorkload();
mesh_workload.add_program(
    MeshCoordinateRange(device_coord), std::move(program));
EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);
std::cout << "Program running. Waiting for completion..." << std::endl;

Finish(mesh_device->mesh_command_queue());
```

The `MeshWorkload` abstraction allows launching the same program across multiple devices in a mesh. Here, with a 1x1 mesh and `MeshCoordinateRange(device_coord)` targeting `(0, 0)`, it launches on a single device. The `false` argument to `EnqueueMeshWorkload()` means "do not block" -- the call enqueues the workload and returns immediately.

`Finish()` then blocks until all enqueued work completes. Since both kernels run infinite loops (exiting only on sentinel), this call blocks for the entire duration of the benchmark. When the host sends a sentinel through the H2D sockets, the kernels exit, `Finish()` returns, and the launcher proceeds to teardown.

---

## 5.2.10 Teardown and _exit()

```cpp
// examples/pi05_device_launcher.cpp:222-228
token_h2d.reset();
token_d2h.reset();
} // sockets destroyed before mesh_device->close()
mesh_device->close();
// Use _exit() to skip atexit handlers -- tt-metal's ShmResourceTracker
// cleanup races with socket/device teardown causing double-free.
_exit(EXIT_SUCCESS);
```

The teardown order is carefully structured:

1. **Token sockets destroyed** (`reset()` on the optionals): Releases shared memory mappings for token descriptors.
2. **Bulk sockets destroyed** (scope exit): The bulk sockets are stack-allocated inside the inner block, so they are destroyed when the block exits -- *before* `mesh_device->close()`.
3. **MeshDevice closed** (`mesh_device->close()`): Releases the hardware device.
4. **Process terminated** via `_exit(EXIT_SUCCESS)`.

The use of `_exit()` instead of `return 0` or `exit(0)` is deliberate. tt-metal registers cleanup handlers via `atexit()` (specifically, `ShmResourceTracker` for shared memory segments). These handlers assume they are the first to clean up shared memory. But the explicit `reset()` and `close()` calls above have already cleaned up some of the same resources. If `atexit` handlers then run, they attempt to free already-freed shared memory, causing a double-free crash. The `_exit()` system call terminates the process immediately without running C++ destructors for static/global objects, calling `atexit()` handlers, or flushing stdio buffers.

The host runner's main function uses the same `_exit()` pattern for the same reason (line 1196).

The exception handler also uses `_exit()`:

```cpp
// examples/pi05_device_launcher.cpp:230-233
} catch (const std::exception& e) {
    std::cerr << "pi05_device_launcher: " << e.what() << std::endl;
    _exit(EXIT_FAILURE);
}
```

Even the error path uses `_exit()` to avoid the same double-free race. The exit status `EXIT_FAILURE` is what the parent's `wait_for_descriptors()` will report via `WEXITSTATUS(status)`.

---

## 5.2.11 Complete Initialization Timeline

The following diagram shows the temporal relationship between the parent runner and the child device launcher:

```
Parent (pi05_pipeline_runner)         Child (pi05_device_launcher)
    |                                      |
    +-- fork() --------------------------->|
    |                                      +-- prctl(PR_SET_PDEATHSIG)
    |                                      +-- execl("pi05_device_launcher")
    |                                      |
    +-- wait_for_descriptors() loop        +-- parse args
    |   (polling /dev/shm every 100ms)     +-- DistributedContext::create() [full only]
    |                                      +-- MeshDevice::create()        [~5-15s]
    |                                      +-- H2DSocket / D2HSocket ctor
    |                                      +-- export_descriptor()
    |   <---- descriptors appear ----------+-- [descriptors now in /dev/shm]
    |                                      +-- CreateCircularBuffer()
    +-- sleep(2000ms) [IOMMU settle]       +-- CreateKernel() x 2
    |                                      +-- EnqueueMeshWorkload()
    +-- BulkH2DChannel::connect()          |   [kernels now running]
    +-- BulkD2HChannel::connect()          +-- Finish()  [blocking]
    |                                      |
    |    ... data transfer loop ...        |   [kernels processing pages]
    |                                      |
    +-- send_sentinel()                    |
    +-- recv_sentinel_echo()               |
    +-- mgr->stop()                        |   [kernels exit main loop]
    +-- bulk channels destroyed            +-- Finish() returns
    |                                      +-- sockets destroyed
    |                                      +-- mesh_device->close()
    |                                      +-- _exit(EXIT_SUCCESS)
    +-- DeviceLauncher::stop()
    |   +-- kill(pid, SIGTERM)
    |   +-- waitpid()  [child already exited]
    +-- _exit(EXIT_SUCCESS)
```

The key synchronization points are:

1. **Descriptor files**: Parent blocks until all `/dev/shm/*.bin` files exist.
2. **IOMMU settle**: Parent sleeps 2 seconds before connecting channels.
3. **Sentinel handshake**: Host sends sentinel via H2D, kernel echoes via D2H, confirming orderly shutdown.
4. **`Finish()`**: Device launcher blocks until kernels complete their cleanup sequence.

---

**Next:** [Bulk Passthrough Kernel](03_bulk_passthrough_kernel.md)
