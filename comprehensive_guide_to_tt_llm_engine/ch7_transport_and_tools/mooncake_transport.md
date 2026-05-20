# Mooncake Transport Integration

[Prev: DeepSeek Inference Runner](deepseek_inference_runner.md) | [Next: Ch8 -- Build System and Design Decisions](../ch8_build_system_and_design_decisions/index.md)

---

The Mooncake transport layer provides standalone testing infrastructure for Mooncake's Transfer Engine (TCP, RDMA) and Mooncake Store (distributed KV object store). This subsystem is entirely independent of the decode scheduler -- it exists to validate the data-movement substrate that future KV cache migration will build on.

All source lives in the `tt_llm_engine::transport::mooncake_test` namespace. The code is test-only today; the header comment in `mooncake_test_engine.hpp` notes that "later plans will move pieces of this into a real transport library" (line 11).

**Source files:**
- `include/tt_llm_engine/transport/mooncake/mooncake_test_engine.hpp` -- TestEngine RAII wrapper, RDMA utilities
- `tests/transport/mooncake/test_mooncake_transport.cpp` -- Transport test suite
- `tests/transport/mooncake/test_mooncake_store.cpp` -- Store test suite
- `tests/transport/mooncake/mooncake_test_fixture.hpp` / `.cpp` -- Shared fixture
- `tests/transport/mooncake/mooncake_test_port_util.hpp` -- Ephemeral endpoint generator
- `tests/transport/mooncake/mooncake_transport_test_policies.hpp` -- TCP/RDMA policy traits
- `cmake/mooncake.cmake` -- Build system integration
- `docs/transport/mooncake.md` -- User-facing setup docs
- `run_mooncake_tests.sh` -- Test runner script

---

## TestEngine: The RAII Foundation

`TestEngine` is the central abstraction. It wraps Mooncake's `TransferEngine` in a single RAII object that handles allocation, initialization, transport installation, buffer registration, peer management, and transfer execution. The design goal (stated in the file header, line 6-8) is that "bumping the Mooncake submodule requires changing exactly one file, not every test."

### EngineConfig

Configuration is captured in a simple struct:

```cpp
// mooncake_test_engine.hpp, lines 45-57
struct EngineConfig {
    std::string local_server_name;        // e.g. "initiator@127.0.0.1:14001"
    std::string metadata_uri{"P2PHANDSHAKE"};
    TransportMode mode = TransportMode::Tcp;

    std::size_t buffer_size = 16 * 1024 * 1024;   // 16 MiB
    std::chrono::microseconds poll_interval{50};   // 50 us
    std::chrono::milliseconds default_timeout{5000};  // 5 s
};
```

| Field | Default | Purpose |
|-------|---------|---------|
| `local_server_name` | (required) | Unique identity for this engine within the P2P handshake |
| `metadata_uri` | `"P2PHANDSHAKE"` | Metadata exchange mode; P2PHANDSHAKE = direct peer-to-peer, no external metadata service |
| `mode` | `Tcp` | `TransportMode::Tcp` or `TransportMode::Rdma` |
| `buffer_size` | 16 MiB | Size of the page-aligned DMA buffer |
| `poll_interval` | 50 us | Sleep between transfer status polls |
| `default_timeout` | 5 s | Maximum wait for any single transfer |

The defaults are "calibrated for the Phase-A payload range (64 KiB -- 8 MiB)" (line 52). Tests that need different sizes override via the `ConfigTweak` callback.

### Construction Sequence

The constructor (`mooncake_test_engine.hpp`, lines 235-310) performs four steps in strict order:

```
1. posix_memalign() -- page-aligned buffer
2. TransferEngine::init() -- P2P handshake setup
3. installTransport("tcp"|"rdma") -- with RDMA NIC matrix if applicable
4. registerLocalMemory() -- at "cpu:0" NUMA location
```

**Buffer allocation** uses `posix_memalign` with 4096-byte alignment (line 240) because some transport backends require page-aligned buffers. The buffer is wrapped in `std::unique_ptr<void, decltype(&std::free)>` for automatic deallocation.

**TransferEngine construction** passes `auto_discover=false` to the constructor (line 249) so transport installation is explicit. The `init()` call follows separately (line 254). The `local_server_name` embeds an address for P2PHANDSHAKE mode, but Mooncake ignores the embedded port and allocates a fresh one. The `server_name()` accessor (line 343) reconstructs the real reachable name from Mooncake's internal RPC metadata:

```cpp
// mooncake_test_engine.hpp, lines 343-346
std::string server_name() const {
    const auto& rpc = engine_->getMetadata()->localRpcMeta();
    return rpc.ip_or_host_name + ":" + std::to_string(rpc.rpc_port);
}
```

This is critical because callers (e.g., `make_pair` calling `openSegment`) must connect to the port the P2P daemon actually listens on, not the port embedded in the original `local_server_name`.

**Transport installation** branches on mode (lines 264-293):

```
            +-----------+
            |   mode?   |
            +-----+-----+
                  |
          +-------+-------+
          |               |
        TCP             RDMA
          |               |
  installTransport   rdma_build_nic_priority_matrix()
  ("tcp", nullptr)       |
                   +------+------+
                   |             |
              matrix found   empty
                   |             |
            installTransport  discover({})
            ("rdma", args)   installTransport
                              ("rdma", nullptr)
```

For RDMA, the NIC priority matrix is built from active libibverbs devices (see RDMA section below). If no devices are found, it falls back to Mooncake's auto-discover path.

**Memory registration** calls `registerLocalMemory` with location `"cpu:0"` (NUMA node 0), `remote_accessible=true`, and `update_metadata=true` (lines 301-304).

### Destruction Sequence

The destructor (lines 312-322) unregisters the local memory before the `engine_` destructor runs:

```cpp
~TestEngine() {
    if (engine_ && buffer_) {
        const int rc = engine_->unregisterLocalMemory(buffer_.get(),
                                                      /*update_metadata=*/true);
        // ...error logging only, no throw...
    }
    // engine_ destructor calls freeEngine internally.
}
```

### Transfer Helpers

`submit_write_and_wait` and `submit_read_and_wait` (lines 380-529) share a common `submit_and_wait` implementation. The flow:

1. Look up the remote buffer base address via `peer_buffer_base()` (lines 368-373)
2. Build a `TransferRequest` with the absolute virtual address (TCP transport sends raw virtual addresses over the wire -- line 435)
3. Allocate a `BatchID` from Mooncake
4. Submit the transfer
5. Poll `getTransferStatus` until COMPLETED, FAILED, CANCELED, TIMEOUT, or wall-clock deadline
6. Free the `BatchID`

The polling loop sleeps for `cfg_.poll_interval` between checks (line 525). Terminal states are detected explicitly, and human-readable status names are logged on failure via `transfer_status_name()` (lines 405-415).

> **Note:** The TCP transport has a subtle completion semantic: it marks a transfer COMPLETED when the initiator has finished *sending*, but the target's worker thread may not yet have written bytes to memory. The `TcpPolicy::verify_match` accounts for this (see Test Policies section).

---

## RDMA Device Probing and the Three-Gate Safety Model

Three functions in `mooncake_test_engine.hpp` handle RDMA hardware discovery. Together with the build-time flag, they form the three-gate RDMA safety model.

### Gate 1: Build-Time (`DS_MOONCAKE_WITH_RDMA`)

`cmake/mooncake.cmake` lines 80-84 set `USE_RDMA ON|OFF` based on the CMake option. When disabled, the `MOONCAKE_HAS_RDMA` define is not set and all RDMA code is excluded at compile time. The `AllTransports` typelist conditionally includes `RdmaPolicy`:

```cpp
// mooncake_transport_test_policies.hpp, lines 114-118
#if defined(MOONCAKE_HAS_RDMA) && MOONCAKE_HAS_RDMA
using AllTransports = ::testing::Types<TcpPolicy, RdmaPolicy>;
#else
using AllTransports = ::testing::Types<TcpPolicy>;
#endif
```

### Gate 2: Known-Broken Hardware (Runtime)

The function `rdma_hardware_known_broken_for_loopback()` (lines 184-199) flags hosts where the only RDMA hardware is Broadcom `bnxt_re*`, which fails `ibv_create_qp` with EFAULT in self-loopback mode with Mooncake v0.3.10.post2:

```cpp
inline bool rdma_hardware_known_broken_for_loopback() {
    if (is_rdma_force_enabled()) return false;
    const auto devs = get_active_rdma_devices();
    if (devs.empty()) return false;
    for (const auto& name : devs) {
        // If ANY device is not bnxt_re*, treat host as not-known-broken
        if (name.rfind("bnxt_re", 0) != 0) return false;
    }
    return true;  // all active devices are bnxt_re*
}
```

### Gate 3: Active-Device Probe (Runtime)

The internal enumeration function `enumerate_rdma_devices_internal()` (lines 77-141) uses libibverbs to discover active RDMA devices:

```cpp
// Simplified flow
ibv_device** devices = ibv_get_device_list(&num_devices);
for (int i = 0; i < num_devices; ++i) {
    // force_rdma: accept without port state query
    if (force_rdma) { active.push_back(name); continue; }
    // Normal path: open device, query port 1
    ibv_context* ctx = ibv_open_device(devices[i]);
    ibv_query_port(ctx, /*port_num=*/1, &attr);
    if (attr.state == IBV_PORT_ACTIVE) {
        active.push_back(name);
    }
    ibv_close_device(ctx);
}
```

Only UP-state ports (`IBV_PORT_ACTIVE`) are included. This prevents DOWN-port NICs from entering Mooncake's `context_list_`, avoiding an out-of-bounds access during the handshake callback.

The `DS_MOONCAKE_FORCE_RDMA=1` environment variable bypasses both runtime gates for debugging.

### Summary Table

| Gate | Level | Mechanism | Effect |
|------|-------|-----------|--------|
| Build-time | `DS_MOONCAKE_WITH_RDMA=ON` | CMake option, sets `MOONCAKE_HAS_RDMA` define | RDMA test targets and `RdmaPolicy` only compiled when enabled |
| Known-broken HW | Runtime | `rdma_hardware_known_broken_for_loopback()` | GTEST_SKIP on hosts with only bnxt_re* NICs |
| Active-device probe | Runtime | `get_active_rdma_devices()` | GTEST_SKIP when no UP-state RDMA hardware found |

### NIC Priority Matrix

`rdma_build_nic_priority_matrix()` (lines 216-227) produces the JSON string that Mooncake's `Topology::parse()` expects:

```cpp
inline std::string rdma_build_nic_priority_matrix() {
    const auto devs = get_active_rdma_devices();
    if (devs.empty()) return {};
    std::string dev_list;
    for (const auto& name : devs) {
        if (!dev_list.empty()) dev_list += ',';
        dev_list += '"';
        dev_list += name;
        dev_list += '"';
    }
    return "{\"cpu:0\": [[" + dev_list + "], []]}";
}
```

Output format: `{"cpu:0": [["mlx5_0","mlx5_1"], []]}`. This maps CPU NUMA node 0 to the active RDMA devices. The critical property: **only active devices are included**. DOWN-port NICs are excluded to prevent an index mismatch in Mooncake's `context_list_`.

---

## Test Infrastructure

### MooncakeTransferFixture

The shared test fixture (`mooncake_test_fixture.hpp`, lines 46-88) provides a paired-engine setup for transfer tests:

```cpp
class MooncakeTransferFixture : public ::testing::Test {
protected:
    using ConfigTweak = std::function<void(EngineConfig&)>;
    void make_pair(TransportMode mode, const ConfigTweak& tweak = {});

    std::unique_ptr<TestEngine> initiator_;
    std::unique_ptr<TestEngine> target_;
    std::uint64_t target_handle_from_initiator_ = 0;
};
```

The `make_pair` implementation (`mooncake_test_fixture.cpp`, lines 24-52) constructs the engines in a specific order:

```cpp
void MooncakeTransferFixture::make_pair(TransportMode mode,
                                         const ConfigTweak& tweak) {
    EngineConfig target_cfg{};
    target_cfg.local_server_name = ephemeral_loopback_endpoint();
    target_cfg.mode = mode;
    if (tweak) tweak(target_cfg);

    EngineConfig initiator_cfg{};
    initiator_cfg.local_server_name = ephemeral_loopback_endpoint();
    initiator_cfg.mode = mode;
    if (tweak) tweak(initiator_cfg);

    // Target must be ready before initiator tries to openSegment.
    target_    = std::make_unique<TestEngine>(target_cfg);
    initiator_ = std::make_unique<TestEngine>(initiator_cfg);
    target_handle_from_initiator_ =
        initiator_->open_peer_segment(target_->server_name());

    ASSERT_NE(target_handle_from_initiator_,
              std::numeric_limits<std::uint64_t>::max());
}
```

Key design decisions:
- **Target created first**: The target engine must be fully initialized before the initiator calls `openSegment()`, because the peer handshake requires the target to be listening.
- **Ephemeral endpoints**: Each engine gets a unique `127.0.0.1:<port>` from the atomic counter. In P2PHANDSHAKE mode Mooncake allocates its own port anyway, so these ports just need to be unique strings.
- **ConfigTweak callback**: Tests can override `buffer_size`, `poll_interval`, etc. without subclassing the fixture.

The fixture also provides `do_write_and_verify<Policy>()` (lines 72-87), a template method that seeds the initiator buffer with a deterministic pattern, submits a write, and verifies the target received the data using the policy's `verify_match`:

```cpp
template <typename Policy>
::testing::AssertionResult do_write_and_verify(
    std::size_t bytes, std::chrono::milliseconds timeout,
    std::uint64_t seed = 0xDEADBEEFCAFEBABEull) {
    seed_initiator_pattern(initiator_->buffer(), target_->buffer(), bytes, seed);
    if (!initiator_->submit_write_and_wait(
            target_handle_from_initiator_, 0, 0, bytes, timeout)) {
        return ::testing::AssertionFailure() << "submit_write_and_wait failed...";
    }
    if (!Policy::verify_match(*initiator_, *target_, bytes)) {
        return ::testing::AssertionFailure() << "verify_match failed...";
    }
    return ::testing::AssertionSuccess();
}
```

### Ephemeral Endpoint Generator

`mooncake_test_port_util.hpp` provides a simple collision-free endpoint factory:

```cpp
// mooncake_test_port_util.hpp, lines 22-27
inline std::string ephemeral_loopback_endpoint() {
    static std::atomic<std::uint32_t> counter{30000};
    const std::uint32_t port = counter.fetch_add(1, std::memory_order_relaxed);
    return "127.0.0.1:" + std::to_string(port);
}
```

The counter starts at 30000 (above the IANA registered range, below the common ephemeral range 32768-60999). Since P2PHANDSHAKE mode ignores the embedded port anyway, this only needs to produce a unique *string*, not an actually-free TCP port. The atomic counter is safe for concurrent calls from multiple test threads.

### Transport Test Policies

The policy-traits pattern (`mooncake_transport_test_policies.hpp`) enables writing one test body that runs over multiple transports. Each policy provides the transport mode, a hardware gate, and a completion verification method.

**TcpPolicy** (lines 45-73):

```cpp
struct TcpPolicy {
    static constexpr TransportMode mode = TransportMode::Tcp;
    static void setup_gate(::testing::Test*) {}  // TCP never needs hardware gating

    static bool verify_match(TestEngine& initiator, TestEngine& target,
                             std::size_t bytes,
                             std::chrono::milliseconds deadline = 200ms) {
        const auto stop = std::chrono::steady_clock::now() + deadline;
        while (true) {
            if (std::memcmp(initiator.buffer(), target.buffer(), bytes) == 0) return true;
            if (std::chrono::steady_clock::now() >= stop) return false;
            std::this_thread::sleep_for(std::chrono::microseconds{500});
        }
    }
};
```

`TcpPolicy::verify_match` polls with a 200ms deadline because the TCP transport completion race means data may not be in the target's memory when `submit_write_and_wait` returns. The source documents this race at line 55: the TCP transport marks the transfer COMPLETED when the initiator has finished *sending*, but the target's worker thread may not yet have written the received bytes to memory.

**RdmaPolicy** (lines 79-102) uses an immediate `memcmp` because RDMA transfers are truly synchronous by the time `submit_write_and_wait` returns -- the data is already in the target's registered buffer via one-sided RDMA write:

```cpp
struct RdmaPolicy {
    static constexpr TransportMode mode = TransportMode::Rdma;
    static void setup_gate(::testing::Test*) {
        if (rdma_hardware_known_broken_for_loopback()) GTEST_SKIP() << ...;
        const auto devs = get_active_rdma_devices();
        if (devs.empty()) GTEST_SKIP() << ...;
    }
    static bool verify_match(TestEngine& initiator, TestEngine& target,
                             std::size_t bytes) {
        return std::memcmp(initiator.buffer(), target.buffer(), bytes) == 0;
    }
};
```

---

## Test Suite Structure

The test binary (`test_mooncake_transport.cpp`) contains four tiers of tests:

### 1. Smoke Tests (lines 36-63)

```cpp
TEST(MooncakeSmoke, EngineConstructsAndDestructsCleanly) {
    mc::EngineConfig cfg{};
    cfg.local_server_name = mc::ephemeral_loopback_endpoint();
    cfg.mode = mc::TransportMode::Tcp;
    EXPECT_NO_THROW({
        mc::TestEngine engine(cfg);
        EXPECT_NE(engine.buffer(), nullptr);
        EXPECT_GT(engine.buffer_size(), 0u);
    });
}
```

Validates the TestEngine lifecycle (construct/destruct) and custom buffer sizes.

### 2. TCP Tunable Demo (lines 73-84)

Exercises the `ConfigTweak` callback pattern:

```cpp
TEST_F(MooncakeTcpTransportTunable, TightPollIntervalOverride) {
    make_pair(mc::TransportMode::Tcp, [](mc::EngineConfig& c) {
        c.buffer_size = 2 * 1024 * 1024;
        c.poll_interval = std::chrono::microseconds{5};
    });
    EXPECT_EQ(initiator_->poll_interval(), std::chrono::microseconds{5});
    EXPECT_TRUE(do_write_and_verify<mc::TcpPolicy>(1 * 1024 * 1024, 5s, 0xABCDEFull));
}
```

### 3. TYPED_TEST Round-Trip Suite (lines 92-131)

Three parameterized tests execute once per policy in `AllTransports`:

| Test | Payload | Timeout | What it tests |
|------|---------|---------|---------------|
| `WriteSmallPayloadEndToEnd` | 64 KiB | 5 s | Minimum useful transfer size |
| `WriteLargePayloadEndToEnd` | 8 MiB | 30 s | Large transfer near buffer capacity |
| `WriteThenReadRoundTrip` | 256 KiB | 5 s | Bidirectional: write then read-back |

`WriteThenReadRoundTrip` demonstrates the full bidirectional path: write initiator to target, then overwrite the initiator buffer with `0xAB`, read target back to initiator, and verify the read data matches the original write data.

### 4. Single-Port RDMA Demo (lines 141-162)

Compiled only when `MOONCAKE_HAS_RDMA` is set. Demonstrates the explicit-count hardware-gating pattern using `require_active_rdma_devices(1)`:

```cpp
#if defined(MOONCAKE_HAS_RDMA) && MOONCAKE_HAS_RDMA
class MooncakeRdmaTransportSinglePort : public mc::MooncakeTransferFixture {
protected:
    void SetUp() override {
        if (mc::rdma_hardware_known_broken_for_loopback()) {
            GTEST_SKIP() << mc::kBnxtReSkipMessage;
        }
        const auto devs = mc::require_active_rdma_devices(1);
        if (!devs) {
            GTEST_SKIP() << "Test requires exactly 1 active RDMA port";
        }
        make_pair(mc::TransportMode::Rdma);
    }
};
#endif
```

---

## Mooncake Store Tests

The store test suite (`test_mooncake_store.cpp`) exercises the `mooncake::Client` Put/Get round-trip through a running `mooncake_master` daemon.

### MooncakeStoreFixture

The fixture (lines 32-112) creates a producer/consumer client pair:

```
+------------------+          +-------------------+
|    producer_     |          |    consumer_      |
|  MountSegment    |          | RegisterLocalMem  |
|  (64 MB buffer)  |          | (16 MB buffer)    |
|  Put(key, data)  |          | Get(key, buf)     |
+--------+---------+          +---------+---------+
         |                              |
         +---------- master -----------+
                  (port 50051)
```

**Producer setup** (lines 52-68):
1. Create a `mooncake::Client` with an ephemeral loopback endpoint
2. Allocate a 64 MB segment buffer
3. Call `MountSegment` to register it as available storage in the master

**Consumer setup** (lines 70-89):
1. Create a separate `mooncake::Client`
2. Allocate a 16 MB local buffer
3. Call `RegisterLocalMemory` with `remote_accessible=false` and `update_metadata=false` (read-side registration for zero-copy reads)

The master address defaults to `127.0.0.1:50051` but is overridable via the `MOONCAKE_MASTER_PORT` environment variable (lines 43-47).

### PutThenGetRoundTrip Test

The test (lines 114-160) performs a 1 MB round-trip:

1. Fill 1 MB source buffer with a deterministic pattern (seed `0xC0FFEE0123456789`)
2. `Put` the data under key `"mooncake_store_smoke_key_v1"` with `replica_num=1`
3. `Get` the data back into the registered consumer buffer
4. Verify byte-for-byte integrity via `memcmp`
5. `Remove` the key to avoid `OBJECT_ALREADY_EXISTS` on re-runs

**TearDown** (lines 92-106) calls `UnmountSegment` on the producer. The code logs but does not fail on unmount errors -- cleanup can fail legitimately when the master is already torn down.

---

## Build System Gating

The Mooncake build integration (`cmake/mooncake.cmake`) is the most complex CMake file in the project. It manages the Mooncake submodule and six vendored dependencies (glog, gflags, yaml-cpp, zstd, Boost, msgpack-cxx) plus Mooncake's own yalantinglibs.

### Flag Hierarchy

```
DS_ENABLE_MOONCAKE (CMake option)
    |
    +-- Controls whether cmake/mooncake.cmake is included at all
    |
    +-- DS_MOONCAKE_WITH_RDMA (CMake option)
            |
            +-- Sets USE_RDMA ON/OFF inside Mooncake
            +-- Defines MOONCAKE_HAS_RDMA for test compilation
```

The entry guard (line 19-21):

```cmake
if(NOT DS_ENABLE_MOONCAKE)
    return()
endif()
```

### Submodule Validation

Lines 24-56 validate that all required submodules are initialized. Each check is a separate `FATAL_ERROR` with a specific `git submodule update` command, so developers get precise instructions:

```cmake
if(NOT EXISTS "${_MOONCAKE_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR
        "DS_ENABLE_MOONCAKE=ON but the submodule at ${_MOONCAKE_ROOT} is not "
        "initialized.  Run: git submodule update --init --recursive")
endif()
```

Checked submodules: `mooncake`, `glog`, `gflags`, `yaml-cpp`, `zstd`, `boost`, `msgpack`, `yalantinglibs`.

### Mooncake Feature Flags

Lines 63-73 disable unnecessary Mooncake features:

| Mooncake Flag | Setting | Why |
|---------------|---------|-----|
| `USE_ETCD` | OFF | Drops etcd client dependency |
| `USE_HTTP` | OFF | Drops libcurl |
| `WITH_STORE_RUST` | OFF | Drops Rust binding build |
| `WITH_P2P_STORE` | OFF | Not needed |
| `WITH_METRICS` | OFF | No metrics collection |
| `BUILD_EXAMPLES` | OFF | No example binaries |
| `BUILD_UNIT_TESTS` | OFF | Build our own tests instead |
| `WITH_STORE` | ON | Needed for Store tests |
| `WITH_TE` | ON | Transfer Engine always needed |

### Vendored Dependency Wiring

The vendored dependencies use CMake 3.24's `OVERRIDE_FIND_PACKAGE` mechanism (lines 88-141). Each `FetchContent_Declare` with `OVERRIDE_FIND_PACKAGE` makes subsequent `find_package()` calls inside Mooncake's CMake resolve from the vendored source:

```cmake
FetchContent_Declare(gflags
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/gflags"
    OVERRIDE_FIND_PACKAGE)
```

Special handling:
- **zstd**: CMake entrypoint is at `build/cmake/`, not the repo root (line 132)
- **gflags**: Pins `cmake_minimum_required(VERSION 3.0.2)` which is below CMake 4.x floor; `CMAKE_POLICY_VERSION_MINIMUM=3.5` workaround (line 139)
- **xxhash**: Not vendored as a submodule; uses zstd's bundled `xxhash.h` for headers and the system `libxxhash.so.0` for the library (lines 157-184)
- **ASIO**: Not a system install; uses yalantinglibs' bundled copy at `include/ylt/thirdparty/` (lines 188-204)

### Post-Build Target Patching

After `add_subdirectory` (line 249), several Mooncake targets need include-directory and compile-option patches:

1. **Vendor includes** are injected PRIVATE to each Mooncake target individually (lines 258-268) -- not directory-scoped, to avoid polluting non-Mooncake targets
2. **glog build dir** is injected AFTER `add_subdirectory` and only to specific targets (`tcp_transport`, `base`) to prevent glog's `config.h` from shadowing Mooncake's own `config.h` (lines 274-279)
3. **gflags force-include** is applied to `mooncake_store` which uses `DEFINE_bool`/`DEFINE_int32` without including the header (lines 297-300)
4. **gflags namespace compat shim** is generated for `mooncake_master` which uses `google::` namespace names that the vendored gflags exposes as `gflags::` (lines 305-323)

### Stable Aliases

Lines 343-346 create stable CMake aliases:

```cmake
add_library(TtLlmEngine::Mooncake::TransferEngine ALIAS transfer_engine)
add_library(TtLlmEngine::Mooncake::Store ALIAS mooncake_store)
```

Tests link against these aliases. If Mooncake renames its internal targets, only `mooncake.cmake` needs to change.

---

## Test Runner Script

`run_mooncake_tests.sh` orchestrates the full test suite:

### Flow

```
1. Validate build exists (test_mooncake_smoke binary)
2. If RUN_STORE=ON:
   a. Find mooncake_master binary
   b. Start master on MOONCAKE_MASTER_PORT (default 50051)
   c. Poll /dev/tcp until master is ready (up to 10s)
   d. Check master hasn't crashed during startup
3. cd to build dir
4. Run: ctest -L mooncake [--exclude RDMA if RUN_RDMA=OFF]
5. Cleanup: kill master on exit (trap EXIT)
```

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `BUILD_DIR` | `./build-standalone` | Build output directory |
| `RUN_RDMA` | `AUTO` | `AUTO`/`ON`/`OFF` -- controls RDMA test inclusion |
| `RUN_STORE` | `ON` | Whether to start master and run store tests |
| `MOONCAKE_MASTER_PORT` | `50051` | Master daemon port |

### Master Lifecycle

The master is started in the background (lines 56-57):

```bash
"${MASTER_BIN}" --port="${MOONCAKE_MASTER_PORT}" --enable_metric_reporting=false \
    >"${BUILD_DIR}/mooncake_master.log" 2>&1 &
MASTER_PID=$!
```

Readiness is polled via bash's `/dev/tcp` pseudo-device (lines 62-82). The cleanup trap (lines 28-43) also detects master crashes that occurred during the test run:

```bash
cleanup() {
    if [[ -n "${MASTER_PID}" ]]; then
        if ! kill -0 "${MASTER_PID}" 2>/dev/null; then
            echo "ERROR: mooncake_master crashed during the test run." >&2
            tail -n 30 "${BUILD_DIR}/mooncake_master.log" >&2
            exit 1
        fi
        kill "${MASTER_PID}" 2>/dev/null || true
    fi
}
```

---

## Worked Example: TCP Write Round-Trip

Here is the concrete sequence of events when `WriteSmallPayloadEndToEnd` runs with `TcpPolicy`:

```
1. MooncakeTransportTyped<TcpPolicy>::SetUp()
   - TcpPolicy::setup_gate() -- no-op for TCP
   - make_pair(TransportMode::Tcp)
     - target_cfg.local_server_name = "127.0.0.1:30000"
     - initiator_cfg.local_server_name = "127.0.0.1:30001"
     - target_ = TestEngine(target_cfg)
       - posix_memalign(16 MiB, 4096-aligned)
       - TransferEngine::init("P2PHANDSHAKE", "127.0.0.1:30000", "", 0)
       - installTransport("tcp", nullptr)
       - registerLocalMemory(buffer, 16 MiB, "cpu:0")
     - initiator_ = TestEngine(initiator_cfg)  [same sequence]
     - target_handle_ = initiator_->open_peer_segment(target_->server_name())

2. TYPED_TEST body: do_write_and_verify<TcpPolicy>(64 KiB, 5s)
   - seed_initiator_pattern(initiator buffer, target buffer, 64K, 0xDEADBEEF...)
     - Fill initiator with mt19937_64 pseudo-random bytes
     - Zero target buffer
   - initiator_->submit_write_and_wait(target_handle_, 0, 0, 64K, 5s)
     - peer_buffer_base(target_handle_) -> remote vaddr
     - allocateBatchID(1)
     - submitTransfer(WRITE request: 64K from local[0] to remote[0+base])
     - Poll getTransferStatus every 50us until COMPLETED
   - TcpPolicy::verify_match(initiator, target, 64K, 200ms)
     - Poll memcmp(initiator.buffer, target.buffer, 64K) every 500us
     - Returns true when they match (TCP worker writes lag transfer completion)

3. TearDown()
   - initiator_.reset() -> ~TestEngine -> unregisterLocalMemory -> ~TransferEngine
   - target_.reset()   -> same
```

---

## Submodule Pinning Policy

From `docs/transport/mooncake.md`:

> The Mooncake submodule is pinned to a tagged release. Upstream pushes breaking changes (CMake renames, API churn) without notice -- do **not** float to `main`. Bump intentionally via PR and update both this doc and `mooncake_test_engine.hpp` to match any API changes.

This policy reflects the reality that Mooncake is an actively-developed external dependency with frequent breaking changes. The `TestEngine` wrapper centralizes all API calls to minimize the blast radius of upstream changes.

---

## Cross-References

- The Mooncake Store's Put/Get model is the planned substrate for **KV cache migration** workflows described in [Ch6 kv_cache_tiering.md](../ch6_lifecycle_is_integration_kv_tiering/kv_cache_tiering.md) (planned, not yet implemented)
- The `SocketPipeline` transport layer (distinct from Mooncake) is covered in [Ch5 socket_pipeline.md](../ch5_pipeline_and_wire_format/socket_pipeline.md)
- Build system details beyond Mooncake are in [Ch8 build_system.md](../ch8_build_system_and_design_decisions/build_system.md)

---

[Prev: DeepSeek Inference Runner](deepseek_inference_runner.md) | [Next: Ch8 -- Build System and Design Decisions](../ch8_build_system_and_design_decisions/index.md)
