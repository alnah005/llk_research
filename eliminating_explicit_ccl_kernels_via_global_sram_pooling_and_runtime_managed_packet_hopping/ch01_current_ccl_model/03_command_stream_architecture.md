# 03 -- Command Stream Architecture

## Context

The previous file traced how a CCL operation flows from the TTNN API through device operation creation to per-device program building. At the core of each program is a **command stream** -- a sequence of encoded instructions that tell device-side worker kernels exactly what data to read, where to send it, and how to synchronize. This file documents the CCL command-stream abstraction in detail: the opcode set (`CclCommandCode`, `CclCommandArgCode`), the argument encoding format, the host-side builders that generate command streams from tensor slice descriptors, the lowering pipeline that converts high-level commands to NOC-level instructions, the device-side command processor that interprets the encoded stream at runtime, and a concrete worked example showing how these pieces fit together. The reader should understand how human-readable "read this tensor slice and send it to device N" gets encoded into a compact binary stream packed into kernel runtime arguments.

---

## Architecture Overview

The command stream system has three layers, each with distinct host-side and device-side components:

```
 Host Side                              Device Side
 --------                              -----------

 Tensor Slice Builders                  Command Processor
  ccl_command_stream_builders.hpp        command_processor.hpp
       |                                      ^
       v                                      |
 Micro-Op Factory (uops)                Runtime Args
  ccl_host_commands.hpp                 (packed uint32_t[])
       |                                      ^
       v                                      |
 Command Lowering + Encoding            Kernel RT Args
  command_lowering.hpp
  ccl_worker_builder.hpp
```

---

## 1. Command Codes: `CclCommandCode`

```cpp
// ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp
enum class CclCommandCode : uint8_t {
    STREAM_TENSOR_TO_EDM = 0,  // Also: STREAM_TENSOR_TO_CB
    STREAM_TENSOR_TO_CB = 0,   // Read tensor slice into circular buffer
    STREAM_CB_TO_TENSOR = 1,   // Write circular buffer contents to tensor
    STREAM_EDM_TO_TENSOR = 2,  // Write from EDM to tensor (legacy alias)

    WAIT_VALUE = 3,            // Wait for semaphore to reach target value

    ATOMIC_INC = 4,            // Atomically increment a semaphore

    RAW_INLINE_WRITE_BYTES = 5, // Write a small inline value

    NOC_READ_BURST = 6,         // Burst of NOC reads from arbitrary addresses
    NOC_WRITE_BURST = 7,        // Burst of NOC writes to arbitrary addresses

    FLOW_CONTROLLED_NOC_READ_BURST = 8, // NOC reads with per-read semaphore waits
    NOC_WRITE_AND_ATOMIC_INC = 9,       // Combined write + atomic increment

    INVALID = 10
};
```

The command set divides into three categories:

### Tensor Data Movement Commands

| Command | Semantics |
|---------|-----------|
| `STREAM_TENSOR_TO_CB` (0) | Read pages from a tensor (via address generator) into a circular buffer. The source tensor location is described by the current `CclCommandTensor` state. |
| `STREAM_CB_TO_TENSOR` (1) | Write pages from a circular buffer into a tensor. The destination tensor location is described by the current `CclCommandTensor` state. |

These are the workhorse commands for all-gather and reduce-scatter. They operate on tensor **slices** -- 4D sub-regions of the full tensor described by shape, offset, and worker assignment.

### Synchronization Commands

| Command | Semantics |
|---------|-----------|
| `WAIT_VALUE` (3) | Spin-wait until a semaphore at the specified address reaches the target value. Used for inter-device synchronization. |
| `ATOMIC_INC` (4) | Atomically increment a semaphore. Can target local, unicast (remote chip), or multicast destinations. |

### NOC Burst Commands

| Command | Semantics |
|---------|-----------|
| `NOC_READ_BURST` (6) | Execute a burst of NOC read transactions, densely packing transfers into CB packets. Used for gathering non-contiguous data. |
| `NOC_WRITE_BURST` (7) | Execute a burst of NOC write transactions. Used for scattering data to non-contiguous locations. |
| `FLOW_CONTROLLED_NOC_READ_BURST` (8) | Like `NOC_READ_BURST` but each read waits for a semaphore signal first. Provides producer-consumer flow control. |
| `NOC_WRITE_AND_ATOMIC_INC` (9) | Combined write + atomic increment in a single command. Reduces synchronization overhead. |

---

## 2. Command Header Encoding

Every command is encoded into a single `uint32_t` header word:

```cpp
struct CclCommandHeader {
    CclCommandCode code : 6;        // Bits [0:5] - command opcode
    CclCommandDestType dest_type : 2; // Bits [6:7] - destination type
    uint8_t arg_count : 4;           // Bits [8:11] - number of argument blocks
    union {
        UnicastCommandDestArgs unicast;   // distance_in_hops + is_forward_direction
        MulticastCommandDestArgs multicast; // num_targets_forward + num_targets_backward
        LocalOnlyCommandDestArgs local_only;
    } command_dest_args;              // Bits [16:31] - destination-specific args (bits [12:15] are reserved/unused)
};
static_assert(sizeof(CclCommandHeader) == sizeof(uint32_t));
```

The destination type field (`CclCommandDestType`) determines routing:

```cpp
enum CclCommandDestType : uint8_t {
    CHIP_UNICAST = 0,      // Send to one specific remote device
    CHIP_MULTICAST = 1,    // Send to multiple devices
    CHIP_LOCAL_ONLY = 2    // No fabric transfer, local operation only
};
```

For unicast destinations, the header embeds the hop distance and direction:

```cpp
struct UnicastCommandDestArgs {
    uint8_t distance_in_hops;     // How many fabric hops to the target
    bool is_forward_direction;    // Forward or backward along the ring/line
};
```

For multicast destinations:

```cpp
struct MulticastCommandDestArgs {
    uint8_t num_targets_forward_direction;   // Devices to reach going forward
    uint8_t num_targets_backward_direction;  // Devices to reach going backward
};
```

> **Key Insight:** The destination encoding uses hop distance rather than absolute device addresses. This is a consequence of TT-Fabric's source-routing model: the packet header carries per-hop forwarding decisions, and the CCL command only needs to know "how far" rather than "exactly where." This is the explicit programming model that a transparent GSRP would need to eliminate -- the programmer (or host-side builder) must know the hop count and direction for every data movement.

---

## 3. Command Argument Codes: `CclCommandArgCode`

```cpp
enum class CclCommandArgCode : uint8_t {
    SET_TENSOR_SHAPE_IN_PAGES = 0,
    SET_TENSOR_SLICE_SHAPE_IN_PAGES = 1,
    SET_TENSOR_SLICE_OFFSET_IN_PAGES = 2,
    SET_WORKER_START_OFFSET_IN_SLICE_IN_PAGES = 3,
    SET_WORKER_PAGES_PER_SLICE = 4,
    SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES = 5,
    SET_TARGET_VALUE = 6,
    SET_ATOMIC_INC_VALUE = 7,
    SET_ADDRESS_INFO = 8,
    SET_CORE_DESCRIPTOR_INFO = 9,
    SET_NOC_TRANSFER_BURST_START_INFO = 10,
    SET_NOC_TRANSFER_BURST_SIZE_PER_PACKET = 11,
    INVALID = 0xFF
};
```

### Tensor Slice Descriptors (Codes 0-5)

These arguments collectively describe a `CclCommandTensor`, the core data structure for tensor-based commands:

```cpp
struct CclCommandTensor {
    Shape4D<uint32_t> tensor_shape;            // Full tensor shape in pages
    Shape4D<uint32_t> tensor_slice_shape;       // Shape of this device's slice
    Shape4D<uint32_t> tensor_slice_offset;      // Offset of the slice within the tensor
    Shape4D<uint32_t> worker_start_offset_in_slice; // This worker's offset within the slice
    uint32_t worker_pages_per_slice;            // Number of pages this worker processes
};
```

All shapes are expressed in **pages** (not bytes), using the `Shape4D<uint32_t>` type with dimensions `(w, z, y, x)` where `x` is the innermost (fastest-varying) dimension. The hierarchical decomposition is:

$$\text{Global Tensor} \supset \text{Tensor Slice (per-device)} \supset \text{Worker Slice (per-worker)}$$

The `SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES` (code 5) argument sets all five fields at once, consuming $4 \times 4 + 1 = 17$ uint32 words. Subsequent commands typically use individual setters to update only the fields that change (delta encoding), reducing the total size of the runtime argument array.

### Argument Header Encoding

Each argument block starts with a `CclCommandArgHeader`:

```cpp
struct CclCommandArgHeader {
    CclCommandArgCode code = CclCommandArgCode::INVALID; // Byte 0
    uint8_t inline_value0 = 0;  // Byte 1 - inline parameter
    uint8_t inline_value1 = 0;  // Byte 2 - inline parameter
    uint8_t inline_value2 = 0;  // Byte 3 - inline parameter
};
static_assert(sizeof(CclCommandArgHeader) == sizeof(uint32_t));
```

The three inline bytes after the code can carry small parameters (e.g., source/dest type, address type, core descriptor type) without consuming additional words.

---

## 4. Address Types and Core Descriptors

Commands reference source and destination locations through two orthogonal type systems:

### Address Types (`CclCommandAddrType`)

```cpp
enum class CclCommandAddrType : uint8_t {
    SEMAPHORE_ID,          // Address resolved from semaphore ID
    CIRCULAR_BUFFER_ID,    // Address resolved from CB ID
    ABSOLUTE_ADDRESS,      // Raw L1/DRAM address
    RELATIVE_ADDRESS,      // Offset from a base address
    NONE                   // No address (for inline commands)
};
```

### Core Descriptor Types (`CclCommandCoreDescriptorType`)

```cpp
enum class CclCommandCoreDescriptorType : uint8_t {
    ADDRGEN = 0,    // Core determined by address generator (tensor layout)
    LOCAL = 1,      // This core (self)
    NOC_XY = 2,     // Explicit NOC (x,y) coordinate
    RECTANGLE = 3,  // Multicast rectangle (start_x, start_y, end_x, end_y)
    NONE = 4        // No core specification (embedded in NOC burst)
};
```

The `ADDRGEN` type is the most common for tensor operations -- the target core is implicitly determined by the page index and the tensor's memory layout (interleaved banking or sharding pattern).

The `RECTANGLE` type (`CclCommandCoreDescriptorTypeMcast`) packs four 8-bit coordinates into a single `uint32_t`:

```cpp
struct CclCommandCoreDescriptorTypeMcast {
    uint8_t noc0_start_x;
    uint8_t noc0_start_y;
    uint8_t noc0_end_x;
    uint8_t noc0_end_y;
};
```

---

## 5. Host-Side Command Builders

The host-side code generates command streams through a layered builder system.

### 5.1 High-Level Tensor Slice Builders (`ccl_command_stream_builders.hpp`)

These functions operate on `TensorSlice` descriptors and produce per-worker command sequences:

```cpp
namespace ttnn::ccl::cmd::builder {
    // Split a tensor into N slices along a dimension
    std::vector<TensorSlice> generate_tensor_slices(
        size_t num_slices, const Tensor& tensor, size_t split_dim);

    // Page-aligned slicing (respects tile boundaries)
    std::vector<TensorSlice> compute_page_aligned_slices(
        size_t num_slices, const Tensor& input_tensor, size_t split_dim);

    // Distribute slices across workers with page alignment
    std::vector<std::vector<TensorSlice>> split_tensor_slices_across_workers_page_aligned(
        size_t num_workers, const std::vector<TensorSlice>& tensor_slices);

    // Complete worker slice generation
    std::vector<std::vector<TensorSlice>> generate_worker_tensor_slices(
        size_t num_slices, const Tensor& tensor, size_t num_workers, size_t split_dim);
}
```

### 5.2 Low-Level Command Micro-Ops (`ccl_host_commands.hpp`)

These functions produce individual `CclHostLowLevelWorkerCommand` objects -- the intermediate representation before encoding into runtime arguments:

```cpp
struct CclHostLowLevelWorkerCommand {
    CclCommandCode command_code;
    CclCommandArgs command_args;               // Variant of slice/wait/atomicInc/...
    CclCommandAddrType source_addr_type;
    CclCommandAddrArgs source_addr_args;
    CclCommandAddrType dest_addr_type;
    CclCommandAddrArgs dest_addr_args;
    CclCommandCoreDescriptorType core_desc_type;
    CclCommandCoreDescriptorArgs core_desc_args;
    CclCommandDestType fabric_transfer_type;   // unicast/multicast/local
    CclCommandDestArgs fabric_transfer_args;
};
```

Key factory functions in the `uops` namespace:

| Factory Function | What It Produces |
|-----------------|------------------|
| `read_tensor_slice_to_cb(slice, cb_id)` | Read from tensor -> circular buffer (local) |
| `read_tensor_slice_to_cb_for_eventual_fabric_write(slice, cb_id)` | Read from tensor -> CB for fabric forwarding |
| `local_write_cb_to_tensor_slice(slice, cb_id)` | Write from CB -> local tensor |
| `fabric_write_cb_to_tensor_slice(slice, cb_id, dest_args)` | Write from CB -> remote tensor via fabric |
| `local_semaphore_wait(sem_id, value)` | Wait for semaphore to reach value |
| `fabric_unicast_semaphore_inc(sem, inc, x, y, dest)` | Unicast semaphore increment via fabric |
| `fabric_multicast_semaphore_inc(sem, inc, x, y, dest)` | Multicast semaphore increment via fabric |
| `local_noc_read_burst_to_cb(addr, transfers, cb_size, cb_id)` | Burst of NOC reads into CB |

### 5.3 Command Encoding (`ccl_worker_builder.hpp`)

The `generate_ccl_command_stream_to_kernel_args` function handles the final encoding step -- converting the `CclHostLowLevelWorkerCommand` sequence into packed `uint32_t` runtime argument arrays:

```cpp
void generate_ccl_command_stream_to_kernel_args(
    std::vector<CclHostLowLevelWorkerCommand> const& ccl_command_stream,
    std::optional<size_t> tensor_index,
    std::optional<std::vector<size_t>> const& tensor_indices,
    tensor_address_runtime_args_overrider *rt_args_overrider_out,
    std::vector<uint32_t>& rt_args_out);
```

This function also populates a `tensor_address_runtime_args_overrider` that records which positions in the runtime argument array correspond to tensor base addresses, enabling efficient address updates on program cache hits without re-encoding the entire command stream.

---

## 6. Command Lowering Pipeline

The `command_lowering.hpp` module provides a transformation from tensor-slice-level commands to NOC-level burst commands:

```cpp
std::vector<CclHostLowLevelWorkerCommand> tensor_slice_commands_to_noc_commands(
    const std::vector<CclHostLowLevelWorkerCommand>& command_stream,
    const Tensor& tensor,
    size_t packet_size_bytes);
```

This lowering pass takes a command stream containing `STREAM_TENSOR_TO_CB` / `STREAM_CB_TO_TENSOR` commands and converts them to `NOC_READ_BURST` / `NOC_WRITE_BURST` commands that specify exact NOC addresses and transfer sizes. The lowered commands:

- Pre-compute all NOC addresses using the tensor's address generator
- Group transfers into packets respecting `packet_size_bytes`
- Eliminate the need for the device-side command processor to perform address generation at runtime

The `can_command_stream_be_lowered_to_noc_commands` predicate checks whether the lowering is applicable (it works for interleaved tensors but may have constraints for certain sharded configurations).

---

## 7. Device-Side Command Processor

On the device side, the command processor reads the encoded runtime arguments and interprets them:

```cpp
CclCommandHeader update_command_tensor(std::size_t &arg_idx, CclCommandTensor &cmd_tensor);
```

The processor maintains a `CclCommandTensor` state that accumulates updates across commands. For each command:

1. Read the `CclCommandHeader` from the runtime arguments
2. For each of the `arg_count` argument blocks:
   a. Read the `CclCommandArgHeader`
   b. Dispatch on the argument code via a `switch` statement
   c. Unpack the argument payload into the `CclCommandTensor` state
3. Execute the command using the current state

The unpacking uses `volatile` pointer casts to support potential future streaming of commands from DRAM or L1 by another core:

```cpp
CclCommandArg<CclCommandArgCode::SET_TENSOR_SHAPE_IN_PAGES>::unpack(
    reinterpret_cast<volatile uint32_t*>(get_arg_addr(arg_idx)),
    cmd_tensor.tensor_shape);
```

### Address Generator Dispatch

The command processor uses template-based selection of the correct address generator:

```cpp
template <TensorMemoryLayout layout, BufferType buf_type, Layout page_layout>
struct source_tensor_addrgen { /* ... */ };
```

Specializations map to:
- `InterleavedAddrGen<is_dram>` for interleaved layouts
- `InterleavedAddrGenFast<is_dram>` for tile-layout interleaved (faster path)
- `DefaultVirtualCoordWidthShardedAddressGenerator` for width-sharded
- `DefaultVirtualCoordHeightShardedAddressGenerator` for height-sharded
- `DefaultVirtualCoordBlockShardedAddressGenerator` for block-sharded

This compile-time dispatch eliminates runtime branching in the hot path of address calculation.

---

## 8. Sharding Address Generation

For sharded tensors, the address generation is handled by specialized address generators defined in `kernel_common/sharding_addrgen.hpp`:

```cpp
template <uint32_t SHARD_TYPE, uint32_t NUMBER_OF_CORES, uint32_t PAGE_SIZE_JUMP,
          uint32_t PAGES_PER_TENSOR_ROW, uint32_t CONTIGUITY,
          uint32_t PAGES_PER_SHARD_WIDTH, uint32_t ROWS_PER_SHARD_HEIGHT>
struct ShardedInfo { ... };
```

The address generator maps a global page index to a `ShardCoordInfo` triple:

```cpp
struct ShardCoordInfo {
    uint32_t core_num;              // Which core owns this page
    uint32_t page_num;              // Page offset within the core's shard
    uint32_t num_contiguous_pages;  // How many pages can be read/written contiguously
};
```

The contiguity information is critical for performance: it enables burst-mode NOC transfers when consecutive pages reside in contiguous L1 memory on the same core.

### Width-Sharded

Given `columns_per_shard` and `total_pages_last_dim`:

```math
\text{core num} = \lfloor \text{page col} / \text{columns per shard} \rfloor
```

```math
\text{local page} = \text{page row} \times \text{columns per shard} + \text{col offset within shard}
```

Contiguous pages span to the end of either the shard width or the tensor row, whichever comes first.

### Height-Sharded

Given `rows_per_shard` and `total_pages_last_dim`:

```math
\text{pages per core} = \text{total pages last dim} \times \text{rows per shard}
```

```math
\text{core num} = \lfloor \text{page num} / \text{pages per core} \rfloor
```

Contiguous pages span to the end of the shard.

### Block-Sharded

Given `columns_per_shard`, `rows_per_shard`, and `total_pages_last_dim`:

```math
\text{w core id} = \lfloor \text{page col} / \text{columns per shard} \rfloor
```

```math
\text{h core id} = \lfloor \text{page row} / \text{rows per shard} \rfloor
```

The 2D core ID is linearized via a mapping table (`mapping_table_t`) that accounts for the physical core grid layout.

---

## 9. Worked Example: Ring All-Gather Command Stream

To make the abstract architecture tangible, here is the concrete command stream for **one worker on one device** in a 4-device ring all-gather. The tensor has shape `(1, 1, 32, 64)` in pages, so each device's shard is `(1, 1, 32, 16)`.

### Worker 0, Forward Direction, Device 1 (ring index 1)

```
Command 0: STREAM_TENSOR_TO_CB
  dest_type: CHIP_UNICAST {distance=1, forward=true}
  args: SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES {
    tensor_shape: (1, 1, 32, 64),     -- full output tensor in pages
    slice_shape: (1, 1, 32, 16),      -- one device's chunk (1/4 of tensor)
    slice_offset: (0, 0, 0, 16),      -- device 1's slice starts at page col 16
    worker_offset: (0, 0, 0, 0),      -- worker 0 starts at beginning of slice
    worker_pages: 512                  -- worker 0 handles all 512 pages in slice
  }
  source_addr: CIRCULAR_BUFFER_ID {cb_id=0}
  core_desc: ADDRGEN
```

This command reads device 1's local tensor shard into a circular buffer. The `CHIP_UNICAST` destination with `distance=1, forward=true` instructs the fabric to forward the CB contents one hop forward (to device 2). The `ADDRGEN` core descriptor means the source pages are addressed by the tensor's memory layout address generator.

```
Command 1: STREAM_CB_TO_TENSOR
  dest_type: CHIP_LOCAL_ONLY
  args: SET_TENSOR_SLICE_OFFSET_IN_PAGES {
    slice_offset: (0, 0, 0, 16)       -- only the offset field is updated (delta encoding)
  }
  dest_addr: CIRCULAR_BUFFER_ID {cb_id=0}
  core_desc: ADDRGEN
```

This command writes the received data from the CB into the local output tensor at device 1's slice position. Note the delta encoding: only `SET_TENSOR_SLICE_OFFSET_IN_PAGES` is emitted because the tensor shape, slice shape, worker offset, and worker pages are unchanged from Command 0.

```
Command 2: STREAM_TENSOR_TO_CB
  dest_type: CHIP_UNICAST {distance=1, forward=true}
  args: SET_TENSOR_SLICE_OFFSET_IN_PAGES {
    slice_offset: (0, 0, 0, 0)        -- now forwarding device 0's slice
  }
  source_addr: CIRCULAR_BUFFER_ID {cb_id=0}
  core_desc: ADDRGEN
```

Ring step 2: device 1 reads the data received from device 0 (which arrived in the previous ring step) and forwards it to device 2.

```
Command 3: WAIT_VALUE
  dest_type: CHIP_LOCAL_ONLY
  args: SET_TARGET_VALUE {value=3}
  source_addr: SEMAPHORE_ID {sem_id=0}
```

Wait for all three ring steps to complete (3 signals expected from the forward direction).

```
Command 4: ATOMIC_INC
  dest_type: CHIP_UNICAST {distance=1, forward=true}
  args: SET_ATOMIC_INC_VALUE {value=1}
  dest_addr: SEMAPHORE_ID {sem_id=0}
  core_desc: NOC_XY {x=1, y=1}
```

Signal the next device that this device has completed its portion of the all-gather.

This 5-command sequence illustrates the key patterns: tensor slice reads with fabric forwarding, local writes, delta-encoded slice offset updates, and semaphore-based synchronization. A full all-gather would have additional commands for the backward direction and for each ring step.

---

## 10. The Complete Encoding Pipeline

Putting it all together, the full pipeline from tensor operation to device execution is:

```
Host Side:
  1. generate_tensor_slices() -> TensorSlice descriptors
  2. split_tensor_slices_across_workers_page_aligned() -> per-worker slices
  3. uops::read_tensor_slice_to_cb() / uops::fabric_write_cb_to_tensor_slice()
     -> CclHostLowLevelWorkerCommand sequence
  4. [optional] tensor_slice_commands_to_noc_commands()
     -> Lowered NOC burst commands
  5. generate_ccl_command_stream_to_kernel_args()
     -> Packed uint32_t runtime arguments

Device Side:
  6. update_command_tensor() reads and decodes runtime args
  7. Address generator (interleaved or sharded) resolves page -> NOC address
  8. Kernel executes NOC reads/writes and fabric sends
```

---

## Key Takeaways

- **The command stream is a custom bytecode:** CCL operations are not directly expressed as NOC API calls. Instead, they are encoded into a compact command stream with stateful argument updates (delta encoding) that device kernels interpret at runtime. This indirection enables program caching and runtime argument override.
- **Routing information is explicit in every command:** Each command header embeds the destination type (unicast/multicast/local) and hop distance. The host-side builder must compute these routing parameters using topology knowledge. This is the fundamental "explicitness" that a transparent model would eliminate.
- **The lowering pipeline bridges tensor semantics and NOC semantics:** The `tensor_slice_commands_to_noc_commands` function converts page-level tensor operations into concrete NOC address-level transfers, handling the complexities of interleaved banking and shard mapping. This is the layer where memory layout knowledge is consumed.
- **Sharding address generation is compile-time-parameterized:** The `ShardedInfo` template carries all shard layout information as compile-time constants, enabling the device-side address resolution to be fully inlined and optimized. This zero-overhead address generation is a performance advantage of the explicit model that a transparent alternative must preserve.
- **The encoding is dense and extensible:** Command headers (4 bytes), argument headers (4 bytes), and payloads (4-17+ words) are packed contiguously into the kernel runtime arguments. The opcode space has room for growth (codes 0-10 of 256), and the `NOC_READ_BURST` / `NOC_WRITE_BURST` commands represent a recent addition that shifts work from runtime interpretation to host-side precomputation.

## Source Code References

- `ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp` -- `CclCommandCode`, `CclCommandArgCode`, `CclCommandHeader`, `CclCommandArgHeader`, `CclCommandTensor`, argument types and variants
- `ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command_device.hpp` -- Device-side `update_command_tensor` and `build_from_args` specializations
- `ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_host_commands.hpp` -- `CclHostLowLevelWorkerCommand`, micro-op builder functions
- `ttnn/cpp/ttnn/operations/ccl/common/uops/command_lowering.hpp` -- `tensor_slice_commands_to_noc_commands` lowering pipeline
- `ttnn/cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp` -- Tensor slice generation and worker distribution
- `ttnn/cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp` -- `CCLWorkerArgBuilder`, `generate_ccl_command_stream_to_kernel_args`, runtime args encoding
- `ttnn/cpp/ttnn/operations/ccl/common/kernels/command_processor.hpp` -- Device-side command processor, `source_tensor_addrgen` specializations
- `ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp` -- `ShardedAddrGen`, `ShardedInfo`, `ShardCoordInfo`, shard coordinate computation
- `ttnn/cpp/ttnn/operations/ccl/sharding_addrgen_helper.hpp` -- Host-side shard argument generation utilities
- `ttnn/cpp/ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp` -- Runtime args override tracking

---

**Next:** [`04_host_orchestration_and_topology.md`](./04_host_orchestration_and_topology.md)
