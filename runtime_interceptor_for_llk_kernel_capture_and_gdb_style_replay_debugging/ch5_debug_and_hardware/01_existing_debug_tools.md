# Chapter 5, File 1 -- Existing Debug Tools in TT-Metal and TT-LLK

> **Scope.** This file catalogues every debug mechanism currently shipping in
> TT-Metal and TT-LLK, documenting the mechanism, data format, activation,
> limitations, and relevance to the proposed runtime interceptor.  All citations
> reference commit `621b949` unless noted otherwise.

---

## 1. Watcher System

### 1.1 Architecture

The Watcher is TT-Metal's primary always-available debug subsystem.  A host-side
polling thread (`WatcherServer::Impl::poll_watcher_data`, `watcher_server.cpp`
line 505) reads L1 mailbox regions from every attached core at a configurable
interval and writes a human-readable log.

| Component | Path |
|-----------|------|
| Host server header | `tt_metal/impl/debug/watcher_server.hpp` (lines 14-50) |
| Host server impl   | `tt_metal/impl/debug/watcher_server.cpp` (lines 44-590) |
| Device reader       | `tt_metal/impl/debug/watcher_device_reader.hpp` / `.cpp` |
| Device waypoint API | `tt_metal/hw/inc/api/debug/waypoint.h` |
| Device assert API   | `tt_metal/hw/inc/api/debug/assert.h` |
| Device pause API    | `tt_metal/hw/inc/api/debug/pause.h` |
| Device ring buffer  | `tt_metal/hw/inc/api/debug/ring_buffer.h` |
| Runtime options      | `tt_metal/llrt/rtoptions.hpp` (lines 81-92, 341-377) |

The `WatcherServer::Impl` class (`watcher_server.cpp` lines 44-100) spawns a
background thread and holds a `watch_mutex_` to serialize reads against dispatch
operations.  Each device gets a `WatcherDeviceReader` that performs L1 reads
and decodes mailbox contents.  Three output files are maintained: `watcher.log`,
`kernel_names.txt`, and `kernel_elf_paths.txt`.

**Key constraint:** The watcher disables DMA (`rtoptions.set_disable_dma_ops(true)`,
`watcher_server.cpp` line 128) because the DMA library is not thread-safe.  Any
interceptor coexisting with watcher must account for this.

### 1.2 Environment Variables

| Env Var | Effect |
|---------|--------|
| `TT_METAL_WATCHER` | Master enable; set to poll interval in ms |
| `TT_METAL_WATCHER_APPEND` | Append to log instead of overwrite |
| `TT_METAL_WATCHER_DUMP_ALL` | Dump all cores, not just those with kernels |
| `TT_METAL_WATCHER_AUTO_UNPAUSE` | Automatically resume paused cores |
| `TT_METAL_WATCHER_NOINLINE` | Compile kernels with `noinline` for better stack info |
| `TT_METAL_WATCHER_DISABLED_FEATURES` | Comma list of sub-features to disable |
| `TT_METAL_WATCHER_DEBUG_DELAY` | Insert N cycle delays on NOC read/write/atomic |

Feature gating at compile time is controlled by `WATCHER_ENABLED`, with individual
sub-features disabled via `WATCHER_DISABLE_PAUSE`, `WATCHER_DISABLE_ASSERT`,
`WATCHER_DISABLE_WAYPOINT`, `WATCHER_DISABLE_RING_BUFFER`, and `FORCE_WATCHER_OFF`.

### 1.3 Waypoints

```c
// waypoint.h -- device side
WAYPOINT("NRW");   // 4-char tag packed into uint32_t, written to L1 mailbox
```

Each RISC-V hart has a 4-byte `debug_waypoint` slot in the watcher mailbox
(`watcher_server.cpp` line 361).  The encoding is a constexpr fold of up to
4 ASCII characters into a `uint32_t` (`waypoint.h` lines 24-33).  The host
reads these and prints a comma-separated line per core per dump cycle.

**Limitation:** Only 4 characters per hart; no timestamp; only the *latest*
value survives each poll cycle.

### 1.4 Assert

When `WATCHER_ENABLED` is defined and the condition is false, `assert_and_hang`
(`assert.h` lines 12-39) writes the `__LINE__` number, assert type, and hart
index into the `debug_assert_msg_t` mailbox and enters an infinite spin.
For ERISC, it performs an early exit via `disable_erisc_app()` instead.

```
struct debug_assert_msg_t {
    uint16_t line_num;
    uint8_t  tripped;     // DebugAssertOK vs DebugAssertTripped
    uint8_t  which;       // hart index
};
```

The watcher detects assert types including `DebugAssertTripped`,
`DebugAssertNCriscNOCReadsFlushedTripped`,
`DebugAssertNCriscNOCNonpostedWritesSentTripped`,
`DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped`,
`DebugAssertNCriscNOCPostedWritesSentTripped`,
`DebugAssertRtaOutOfBounds`,
and `DebugAssertCrtaOutOfBounds` (`debug_helpers.hpp` lines 101-124).

### 1.5 Pause

```c
// pause.h
PAUSE();   // halt this hart until host clears flag
```

The device writes `1` into `pause_msg->flags[hw_thread_idx]` and enters a spin
loop checking for the host to clear it (`pause.h` lines 13-27).  Waypoints
`PASW` / `PASD` bracket the wait.

**Limitation:** Requires the watcher host thread to be running; without it
the core hangs permanently.  `TT_METAL_WATCHER_AUTO_UNPAUSE` can auto-clear.
PAUSE is the closest existing analog to a GDB breakpoint, but pause points
must be compiled in -- they cannot be set dynamically at arbitrary PCs.

### 1.6 Debug Ring Buffer

```c
// ring_buffer.h
WATCHER_RING_BUFFER_PUSH(my_uint32);
```

A small circular buffer in L1 (`DEBUG_RING_BUFFER_ELEMENTS` entries) per core,
initialized to sentinel index `-1` (`int16_t`, `ring_buffer.h` line 10).  The
host reads it during watcher dumps.  Index wraps with a `wrapped` flag
(`ring_buffer.h` lines 19-30).

**Limitation:** Fixed-width `uint32_t` entries only.  No type tagging.
No timestamps.  Typically 32 entries.

### 1.7 NOC Sanitization and Debug Delays

The watcher initializes per-core `sanitize` mailbox entries with sentinel values.
Device firmware writes NOC address/length/type when a transaction is issued; the
watcher host thread checks for illegal addresses (e.g. out-of-range L1, misaligned
multicast).  The `sanitize.h` module (`tt_metal/hw/inc/internal/debug/sanitize.h`)
validates NOC addresses against known valid coordinate ranges.

Debug delays (`TT_METAL_WATCHER_DEBUG_DELAY`) insert configurable cycle waits
on NOC reads, writes, and atomics.  Processor masks control which RISCs are
affected (`watcher_server.cpp` lines 390-468).

### 1.8 Stack Usage Watermarking

Firmware paints the stack with `0xBABABABA`; `measure_stack_usage()` walks
upward to find the high-water mark (`tt_metal/hw/inc/internal/debug/stack_usage.h`
lines 17-53).  Results are logged per-RISC during watcher dumps.

---

## 2. DPRINT System

### 2.1 Architecture

DPRINT provides `printf`-style output from device kernels.  A device-side
`DebugPrinter` object writes typed, serialized tokens into a per-hart L1 ring
buffer.  A host-side `DPrintServer` polling thread drains the buffers and
reassembles human-readable text.

| Component | Path |
|-----------|------|
| Device API | `tt_metal/hw/inc/api/debug/dprint.h` (484 lines) |
| Host server | `tt_metal/impl/debug/dprint_server.hpp` / `.cpp` |
| Shared constants | `hostdevcommon/dprint_common.h` |
| Buffer address | `tt_metal/impl/debug/debug_helpers.hpp` (lines 76-80) |

### 2.2 Wire Format

Each print token is serialized as `[ type_code : 1B ][ payload_size : 1B ][ payload : NB ]`.
`DPRINT_BUFFER_SIZE` is 204 bytes per thread (`dprint_common.h` line 27).  The
`DebugPrintMemLayout` struct packs `wpos`, `rpos`, `core_x`, `core_y` into a
12-byte `Aux` header, leaving 192 bytes for data.

Type codes are enumerated in `DPrintTypeID` (`dprint_common.h` lines 55-62):
`DPrintCSTR`, `DPrintENDL`, `DPrintSETW`, `DPrintUINT8` through `DPrintUINT64`,
`DPrintINT8` through `DPrintINT64`, `DPrintFLOAT32`, `DPrintCHAR`,
`DPrintBFLOAT16`, `DPrintTILESLICE`, `DPrintU32_ARRAY`, `DPrintTYPED_U32_ARRAY`,
plus formatting controls (`HEX`, `OCT`, `DEC`, `FIXED`, `DEFAULTFLOAT`,
`SETPRECISION`).

### 2.3 Device API

```c
DPRINT << SETW(2) << 0 << 0.1f << "hello" << ENDL();
DPRINT_MATH(x)     // only in math TRISC
DPRINT_UNPACK(x)   // only in unpack TRISC
DPRINT_PACK(x)     // only in pack TRISC
```

The `DPRINT` macro gates on `DEBUG_PRINT_ENABLED` and `!FORCE_DPRINT_OFF`.
Per-TRISC variants use `UCK_CHLKC_*` compile defines.

### 2.4 Flow Control and TileSlice

When the buffer is full, the device spins waiting for the host to advance `rpos`
(`dprint.h` lines 356-366, waypoint `DPW`/`DPD`).  The host detects stalls via
`DPrintServer::hang_detected()`.

When compiled with `KERNEL_BUILD`, `dprint_tile.h` adds tile-printing support
with `SliceRange`.  Return codes include `DPrintOK`, `DPrintErrorBadTileIdx`,
`DPrintErrorBadPointer`, `DPrintErrorUnsupportedFormat`.

**Key limitations:**
- **192-byte buffer per thread** -- large prints stall the device.
- **Disabled during profiling** -- `PROFILE_KERNEL` suppresses DPRINT.
- **No timestamps** -- printed data has no timing information.

---

## 3. Assert / Pause Mechanisms (Three Tiers)

### 3.1 Tier 1: Watcher Asserts (ASSERT)

Covered in Section 1.4 above.  Full mailbox metadata including line number, assert
type, and RISC index.

### 3.2 Tier 2: Lightweight Kernel Asserts

When `LIGHTWEIGHT_KERNEL_ASSERTS` is defined (and watcher is disabled), the
`ASSERT()` macro compiles to a bare `asm volatile("ebreak")` (`assert.h` line 60),
triggering the RISC-V breakpoint exception.  Provides no mailbox metadata; the
host detects halts by scanning `RISC_DBG_STATUS_0` across cores.  Post-mortem
analysis is handled by `tt-triage`.

### 3.3 Tier 3: LLK Assert

```c
// tt-llk/tt_llk_wormhole_b0/llk_lib/llk_assert.h
LLK_ASSERT(condition, "message");
```

Two modes of operation (`llk_assert.h` lines 7-37):

1. **`ENV_LLK_INFRA` (standalone LLK test):** Issues `ebreak` directly (line 18).
2. **TT-Metal integration:** Delegates to `ASSERT(condition)` (line 28).
3. **Disabled:** `(void)sizeof((condition))` -- zero overhead (line 36).

Controlled by `TT_METAL_LLK_ASSERTS=1` / `ENABLE_LLK_ASSERT`.  Validates tensor
dimensions, data formats, hardware config parameters, and matrix multiplication
constraints within the LLK compute stack.

---

## 4. Inspector

The Inspector is a Cap'n Proto RPC-based introspection system that tracks program
lifecycle at the TTNN/TT-Metal operation granularity.

| Component | Path |
|-----------|------|
| Public API | `tt_metal/impl/debug/inspector/inspector.hpp` |
| Implementation | `tt_metal/impl/debug/inspector/inspector.cpp` |
| Data storage | `tt_metal/impl/debug/inspector/data.hpp` (lines 14-75) |
| RPC server | `tt_metal/impl/debug/inspector/rpc_server_controller.hpp` |
| Types | `tt_metal/impl/debug/inspector/types.hpp` |

The `inspector::Data` class maintains `programs_data` (map of program_id to
lifecycle/compilation/kernel data), `mesh_devices_data`, `runtime_ids` (capped
at 10,000 entries), dispatch core assignments, and `fw_compile_hash`.

RPC methods include `rpc_get_programs()`, `rpc_get_kernel(params)`,
`rpc_get_mesh_devices()`, `rpc_get_all_dispatch_core_infos()`, and
`rpc_get_all_build_envs()` (`data.hpp` lines 23-33).

When Inspector is enabled, `TT_METAL_RISCV_DEBUG_INFO` defaults to `true`,
causing kernel ELF files to be compiled with DWARF debug info
(`rtoptions.cpp` lines 1110-1122, 1332-1334).  This is the sole mechanism
that enables source-level GDB debugging of device kernels.

**PROPOSED:** The interceptor could register as an Inspector observer, receiving
callbacks at `program_compile_finished` to extract ELF binaries and at
`mesh_workload_created` to capture dispatch topology, rather than reimplementing
this tracking.

---

## 5. Kernel Profiler

### 5.1 TT-Metal Profiler

The TT-Metal kernel profiler (`tt_metal/tools/profiler/kernel_profiler.hpp`)
writes timestamped markers into per-RISC L1 buffers.  It supports zone-based
profiling, dispatch core profiling, and sum counters.  A Tracy backend
(`tt_metal_tracy.hpp`) exports to Tracy visualization.

Timestamps are read from Tensix wall clock registers
(`RISCV_DEBUG_REG_WALL_CLOCK_L` at `0xFFB121F0`, `RISCV_DEBUG_REG_WALL_CLOCK_H`
at `0xFFB121F8`).

**Mutually exclusive with DPRINT**: `PROFILE_KERNEL` disables DPRINT
(`dprint.h` line 437).

### 5.2 LLK Profiler (tt-llk)

The LLK profiler (`tests/helpers/include/profiler.h`) uses FNV-1a hashing
of `"LLK_PROFILER:file:line:marker"` to produce 16-bit zone IDs.

Buffer layout: each TRISC gets `BUFFER_LENGTH = 0x400` (1024) entries.  On
3 TRISCs share buffer space ending at `0x16E000` (`profiler.h` lines 77-80).
No Quasar-specific profiler configuration was found in the codebase at this
commit; the Quasar profiler buffer layout requires verification.

Entry format (8 bytes per entry, `profiler.h` lines 140-150):

```
Word 0: [31:28] = EntryType | [27:12] = id16 | [11:0] = timestamp_high
Word 1: [31:0]  = timestamp_low
```

Entry types (`profiler.h` lines 58-64): `TIMESTAMP` (0b1000),
`TIMESTAMP_DATA` (0b1001), `ZONE_START` (0b1010), `ZONE_END` (0b1011).

Marker metadata is stored in an ELF `.profiler_meta` section for host
post-processing.  Thread synchronization uses a barrier array at `BARRIER_START`
where each TRISC writes `1` and waits for all others (`sync_threads()`).

---

## 6. NOC Debugging / Event Profiler

### 6.1 NOC Debug State Machine

```cpp
// tt_metal/impl/debug/noc_debugging.hpp
struct NocWriteEvent { src_addr, dst_addr, num_bytes, counter_snapshot, ... };
struct NocReadEvent  { src_addr, dst_addr, num_bytes, counter_snapshot, ... };
```

`NOCDebugState` (`noc_debugging.hpp` lines 156-260) accumulates timestamped
NOC events, sorts them, and detects protocol violations:

| Issue Type | Description |
|-----------|-------------|
| `WRITE_FLUSH_BARRIER` | Write followed by barrier without flush |
| `READ_BARRIER` | Read barrier issued with outstanding reads |
| `UNFLUSHED_WRITE_AT_END` | Kernel exits with pending writes |
| `WRITE_TO_LOCKED_CORE_LOCAL_MEM` | Write into a locked memory region |
| `WRITE_TO_LOCKED_CB` | Write into a locked circular buffer |

Events use the variant type `NOCDebugEvent` (`noc_debugging.hpp` line 87)
covering `NocWriteEvent`, `NocReadEvent`, `NocReadBarrierEvent`,
`NocWriteBarrierEvent`, `NocWriteFlushEvent`, and `ScopedLockEvent`.

### 6.2 NOC Data Logging

Separate from the debugging state machine, NOC logging (`impl/debug/noc_logging.cpp`)
repurposes the DPrint buffer infrastructure to collect per-RISC NOC transaction
size histograms (32 buckets, each counting transactions of size $[2^i, 2^{i+1})$).

---

## 7. Data Collection (Dispatch Statistics)

```cpp
// tt_metal/impl/dispatch/data_collection.hpp
void RecordDispatchData(program_id, type, transaction_size, processor);
void RecordKernelGroup(program, core_type, kernel_group);
void RecordProgramRun(program_id);
```

Records per-program dispatch statistics: CB config size, semaphore config size,
runtime args size, binary size.  Types: `DISPATCH_DATA_CB_CONFIG`,
`DISPATCH_DATA_SEMAPHORE`, `DISPATCH_DATA_RTARGS`, `DISPATCH_DATA_BINARY`.

**Limitation:** Statistics/analytics only -- no per-instruction granularity.

---

## 8. tt-exalens -- External Debug and Simulation Bridge

The `ExalensServer` (`tests/python_tests/helpers/exalens_server.py` lines 37-93)
manages a `tt-exalens` subprocess bridging either real hardware or a
cycle-accurate simulator:

```python
class ExalensServer:
    READY_PATTERN = "[4B MODE]"
    READY_TIMEOUT_S = 600

    def start(self) -> None:
        self._process = subprocess.Popen([
            "tt-exalens", f"--port={self._port}",
            "--server", "-s", self._simulator_path,
        ], ...)
```

The Python device interface imports `ttexalens.context.Context`,
`ttexalens.debug_tensix.TensixDebug`, `ttexalens.hardware.risc_debug.CallstackEntry`,
and read/write functions (`tests/python_tests/helpers/device.py` lines 14-29).

| Capability | Silicon | Simulator |
|-----------|---------|-----------|
| Read/write L1 memory | Yes | Yes |
| Load ELF binaries | Yes | Yes |
| Read RISC-V registers | Yes | Yes |
| Hardware breakpoints | Yes | Yes |
| Callstack unwinding | Yes | Yes |
| Cycle-accurate timing | No | Yes |

---

## 9. Firmware Debug Stubs

The firmware debug header (`tt_metal/hw/inc/internal/debug/fw_debug.h` lines 5-9)
defines stub macros (`FWASSERT`, `FWLOG0-2`) that are compile-time disabled in
production builds.  Historically used for early firmware bring-up.

---

## 10. Gap Summary: What Existing Tools Cannot Do

| Capability | Watcher | DPRINT | Inspector | Profiler | NOC Debug | tt-exalens |
|-----------|---------|--------|-----------|----------|-----------|------------|
| **Register-level state capture** | No | No | No | No | No | Partial |
| **Per-instruction stepping** | No | No | No | No | No | Yes (HW) |
| **Breakpoint (HW/SW)** | PAUSE only | No | No | No | No | Yes |
| **Tile data (SRCA/SRCB/DEST)** | No | TileSlice | No | No | No | Yes |
| **Coprocessor config dump** | No | No | No | No | No | Yes |
| **Deterministic replay** | No | No | No | No | No | No |
| **Offline replay on host** | No | No | No | No | No | No |
| **Source-level debug** | No | No | Enables DWARF | No | No | With ELF |
| **Timestamps** | No (poll) | No | No | Yes | Yes | Yes |

None of the existing tools provide:

1. **State snapshots** of Tensix registers (SRCA, SRCB, DEST, config regs)
   at arbitrary points during LLK execution.
2. **Deterministic replay** of a captured kernel on a host emulator
   (see Ch5 File 3 for emulation assessment).
3. **GDB-style stepping** through LLK compute instructions -- the hardware
   breakpoint infrastructure exists but has no host-side GDB integration
   (see Ch5 File 2).
4. **Interceptor-based capture** of LLK API calls for offline replay.

This gap motivates the runtime interceptor design (Ch6) and the replay
debugger design (Ch7).

---

## Infrastructure Reuse Summary for the Interceptor

| Existing Infrastructure | Reuse for Interceptor |
|------------------------|----------------------|
| `GetDprintBufAddr()` address calculation | Allocate capture buffers at known L1 offsets |
| Watcher `WatcherDeviceReader` | Read L1 state (mailboxes, ring buffer, semaphores) |
| Inspector lifecycle callbacks | Trigger capture at program dispatch boundaries |
| LLK profiler thread barrier (`sync_threads()`) | Synchronize TRISCs before host reads state |
| Profiler zone mechanism | Mark LLK call boundaries for state capture |
| Watcher ring buffer | Device-to-host signaling for capture-ready notifications |
| `rtoptions` `TargetSelection` | Specify which chips, cores, RISCs to capture |
| tt-exalens unified API | Bridge captured state to both silicon and simulator |

---

## Key Source Files

| File | Relevance |
|------|-----------|
| `tt_metal/impl/debug/watcher_server.hpp/.cpp` | Host polling server |
| `tt_metal/impl/debug/watcher_device_reader.hpp/.cpp` | Device state reader |
| `tt_metal/hw/inc/api/debug/waypoint.h` | Waypoint macros |
| `tt_metal/hw/inc/api/debug/assert.h` | Assert + `assert_and_hang` |
| `tt_metal/hw/inc/api/debug/pause.h` | PAUSE mechanism |
| `tt_metal/hw/inc/api/debug/ring_buffer.h` | Debug ring buffer |
| `tt_metal/hw/inc/api/debug/dprint.h` | DPRINT device API |
| `hostdevcommon/dprint_common.h` | Shared DPRINT constants |
| `tt_metal/impl/debug/dprint_server.hpp/.cpp` | DPrint host server |
| `tt_metal/impl/debug/inspector/inspector.hpp` | Inspector RPC API |
| `tt_metal/impl/debug/inspector/data.hpp` | Inspector tracked state |
| `tt_metal/impl/debug/noc_debugging.hpp` | NOC debug state machine |
| `tt_metal/impl/dispatch/data_collection.hpp` | Dispatch statistics |
| `tt_metal/tools/profiler/kernel_profiler.hpp` | TT-Metal profiler |
| `tt-llk/tests/helpers/include/profiler.h` | LLK profiler |
| `tt-llk/tt_llk_wormhole_b0/llk_lib/llk_assert.h` | LLK_ASSERT macro |
| `tt_metal/hw/inc/internal/debug/stack_usage.h` | Stack watermarking |
| `tt_metal/hw/inc/internal/debug/fw_debug.h` | Firmware debug stubs |
| `tests/python_tests/helpers/exalens_server.py` | tt-exalens server |
