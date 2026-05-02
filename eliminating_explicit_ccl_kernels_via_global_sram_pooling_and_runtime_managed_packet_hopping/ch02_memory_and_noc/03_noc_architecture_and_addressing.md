# 03 -- NOC Architecture and Addressing

## Context

The previous two files documented the L1 SRAM layout on Tensix and Ethernet cores -- what memory is available and where. This file covers how data moves between those cores: the dual-NOC (Network-on-Chip) system that connects all Tensix cores, DRAM controllers, PCIe endpoints, and Ethernet cores within a single chip. The NOC is the only mechanism for intra-chip data movement, and its address encoding, transaction types, and coordinate system define the hardware interface that any cross-chip abstraction must eventually translate to. The fundamental constraint that motivates the entire GSRP investigation is documented here: **the NOC is chip-local and does not natively cross chip boundaries.** Every inter-chip transfer must be mediated by software running on Ethernet cores.

---

## 1. Dual-NOC System

Each Tenstorrent chip has two independent NOC instances, NOC0 and NOC1, operating in parallel:

| Parameter | Wormhole | Blackhole |
|-----------|:--------:|:---------:|
| Number of NOCs | 2 | 2 |
| NOC data width | 256 bits (32 bytes) | 512 bits (64 bytes) |
| NOC word size | 32 bytes | 64 bytes |
| Maximum burst size | 8,192 bytes (256 x 32 B) | 16,384 bytes (256 x 64 B) |
| Virtual channels | 16 | 16 |
| Broadcast VC start | 4 | 4 |
| Router ports | 3 (NIU, X, Y) | 3 (NIU, X, Y) |

### 1.1 NOC Topology and Routing

The NOC is a 2D mesh interconnect connecting all cores on the chip. Each NOC node (at each core) has three ports:

```cpp
#define NOC_ROUTER_PORTS 3
#define NOC_PORT_NIU 0    // Network Interface Unit (local core connection)
#define NOC_PORT_X   1    // X-direction link (horizontal)
#define NOC_PORT_Y   2    // Y-direction link (vertical)
```

The NIU port connects the router to the local core's L1 and processor. The X and Y ports connect to adjacent routers in the mesh. The NOC uses **XY dimension-ordered routing** (route in X direction first, then Y direction), which is deadlock-free for unicast traffic within the 2D mesh.

Connected endpoints include:
- **Tensix cores**: 80 on WH, 140 on BH
- **DRAM controllers**: Connected to off-chip DRAM channels
- **Ethernet cores**: Connected to inter-chip Ethernet links
- **PCIe controller**: Connected to the host

### 1.2 Deadlock Avoidance Convention

The two NOCs are used to avoid deadlocks in bidirectional communication patterns. By convention:
- **NOC0** is used for writes in one direction and reads in the opposite direction
- **NOC1** is used for the reverse pattern

This separation prevents circular dependencies where core A is writing to core B while core B is simultaneously writing to core A, both blocking on the same NOC resources. The `noc_index` compile-time constant determines which NOC a given kernel processor uses by default, and the `DM_DYNAMIC_NOC` mode allows runtime NOC selection.

---

## 2. Register Space Layout

The NOC is controlled via memory-mapped registers in a dedicated address range:

```cpp
// Both WH and BH
#define NOC_REG_SPACE_START_ADDR 0xFF000000
#define NOC_REGS_START_ADDR      0xFFB20000
```

Key register groups:

| Register | Offset | Description |
|----------|:------:|-------------|
| `NOC_TARG_ADDR_LO/MID/HI` | `+0x0/0x4/0x8` | Target address (64-bit, split across three registers) |
| `NOC_RET_ADDR_LO/MID/HI` | `+0xC/0x10/0x14` | Return address (for reads and atomics) |
| `NOC_PACKET_TAG` | `+0x18` | Transaction ID and header store control |
| `NOC_CTRL` | `+0x1C` | Control register (send request) |
| `NOC_AT_LEN_BE` | `+0x20` | Atomic length and byte enable |
| `NOC_AT_DATA` | `+0x24` (WH) / `+0x28` (BH) | Atomic data value |
| `NOC_CMD_CTRL` | `+0x28` (WH) / `+0x40` (BH) | Command control (ready status) |
| `NOC_NODE_ID` | `+0x2C` (WH) / `+0x44` (BH) | This node's coordinates |

The command buffer offset differs between architectures:

| Parameter | Wormhole | Blackhole |
|-----------|:--------:|:---------:|
| `NOC_CMD_BUF_OFFSET` | `0x400` (10-bit shift) | `0x800` (11-bit shift) |
| `NOC_INSTANCE_OFFSET` | `0x10000` (16-bit shift) | `0x10000` (16-bit shift) |

Multiple command buffers allow pipelining: a processor can enqueue a new transaction in one command buffer while a previous transaction from another buffer is still in flight.

### Status Counters

Each NOC tracks outstanding transactions via status counters at the NIU port. There is no single `NIU_TRANS_STATUS` define; instead, outstanding transaction tracking uses the `NOC_STATUS` register indexed by `NIU_MST_REQS_OUTSTANDING_ID(id)`:

```cpp
#define NIU_MST_REQS_OUTSTANDING_ID(id)  (0x10 + (id))
// Read outstanding count: NOC_STATUS(NIU_MST_REQS_OUTSTANDING_ID(id))
```

Kernels poll `NOC_STATUS(NIU_MST_REQS_OUTSTANDING_ID(id))` to implement barriers -- waiting until all outstanding reads or writes for a given transaction ID have completed.

---

## 3. Coordinate Virtualization

Both WH and BH use coordinate virtualization to map physical core positions to a contiguous virtual address space:

| Parameter | Wormhole | Blackhole |
|-----------|:--------:|:---------:|
| `VIRTUAL_TENSIX_START_X` | 18 | 1 |
| `VIRTUAL_TENSIX_START_Y` | 18 | 2 |
| `COORDINATE_VIRTUALIZATION_ENABLED` | 1 | 1 |

On Wormhole, the virtual Tensix grid starts at coordinate `(18, 18)`. This unusual offset exists because WH's physical die has non-Tensix cores (DRAM controllers, Ethernet cores, PCIe) at lower coordinates, and the virtualization maps only the Tensix grid into a contiguous virtual space starting from `(18, 18)`.

On Blackhole, the virtual grid starts at `(1, 2)`, reflecting a different physical layout with Tensix cores closer to the origin.

### Translation Tables

Coordinate translation is implemented in hardware via lookup tables stored in NOC configuration registers:

```cpp
// Wormhole: 4-bit entries, 32 entries per table
#define NOC_X_ID_TRANSLATE_TABLE_0  0x6   // entries 0-7
#define NOC_X_ID_TRANSLATE_TABLE_1  0x7   // entries 8-15
#define NOC_X_ID_TRANSLATE_TABLE_2  0x8   // entries 16-23
#define NOC_X_ID_TRANSLATE_TABLE_3  0x9   // entries 24-31

// Blackhole: 5-bit entries, 32 entries per table
#define NOC_TRANSLATE_ID_WIDTH      5
#define NOC_X_ID_TRANSLATE_TABLE_0  0x6   // entries 0-5
// ... up to TABLE_5 for entries 30-31
```

Wormhole uses 4-bit entries (16 possible physical coordinates per dimension), while Blackhole uses 5-bit entries (32 possible coordinates). Translation is enabled per-NOC via `NIU_CFG_0[14]` (`NOC_ID_TRANSLATE_EN`). When enabled, the NOC hardware looks up the virtual coordinate in the translation table and replaces it with the physical coordinate before routing the packet.

Blackhole adds additional translation features:
- `NOC_ID_TRANSLATE_COL_MASK` / `NOC_ID_TRANSLATE_ROW_MASK`: Masks indicating which columns/rows bypass translation
- `DDR_COORD_TRANSLATE_TABLE_*`: Separate translation tables for DRAM coordinates
- `DDR_COORD_TRANSLATE_COL_SWAP`: Column swap for DRAM access routing

> **Key Insight:** The existing hardware coordinate translation tables on BH are a potential extension point for GSRP. Currently they map virtual-to-physical coordinates within a chip. They could potentially be repurposed or extended to flag "out-of-chip" coordinates that should be intercepted and redirected to an Ethernet core for fabric transmission. This is explored further in Chapter 6.

---

## 4. 64-Bit NOC Address Encoding

Every NOC address is a 64-bit value that encodes both the target core coordinates and the local memory offset:

### 4.1 Unicast Address Format

```
Bits [35:0]   = Local address (36 bits, addressing up to 64 GB of local address space)
Bits [41:36]  = X coordinate (6 bits, up to 64 nodes)
Bits [47:42]  = Y coordinate (6 bits, up to 64 nodes)
```

The address is constructed via:

```cpp
#define NOC_XY_ADDR(x, y, addr) \
    ((((uint64_t)(y)) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) | \
     (((uint64_t)(x)) << NOC_ADDR_LOCAL_BITS) | \
     ((uint64_t)(addr)))
```

Where `NOC_ADDR_LOCAL_BITS = 36` and `NOC_ADDR_NODE_ID_BITS = 6` on both architectures.

### 4.2 Multicast Address Format

For multicast (writing to a rectangular grid of cores), the address encodes four coordinates:

```
Bits [35:0]   = Local address
Bits [41:36]  = End X coordinate
Bits [47:42]  = End Y coordinate
Bits [53:48]  = Start X coordinate
Bits [59:54]  = Start Y coordinate
```

```cpp
#define NOC_MULTICAST_ADDR(x_start, y_start, x_end, y_end, addr) \
    ((((uint64_t)(x_start)) << (NOC_ADDR_LOCAL_BITS + 2 * NOC_ADDR_NODE_ID_BITS)) | \
     (((uint64_t)(y_start)) << (NOC_ADDR_LOCAL_BITS + 3 * NOC_ADDR_NODE_ID_BITS)) | \
     (((uint64_t)(x_end)) << NOC_ADDR_LOCAL_BITS) | \
     (((uint64_t)(y_end)) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) | \
     ((uint64_t)(addr)))
```

### 4.3 Coordinate-Only Encoding

For efficiency, coordinates can be pre-computed and combined with addresses later:

```cpp
// Unicast coordinate encoding (WH)
#define NOC_XY_ENCODING(x, y) \
    ((((uint32_t)(y)) << ((NOC_ADDR_LOCAL_BITS % 32) + NOC_ADDR_NODE_ID_BITS)) | \
     (((uint32_t)(x)) << (NOC_ADDR_LOCAL_BITS % 32)))

// BH uses a simpler encoding since coordinates are at lower bits
#define NOC_XY_ENCODING(x, y) \
    ((((uint32_t)(y)) << (NOC_ADDR_NODE_ID_BITS)) | (((uint32_t)(x))))
```

### 4.4 Blackhole 64-Bit Address Space

Blackhole's NOC hardware supports a full 64-bit address space, but for backward compatibility, the API uses the same 36-bit address encoding as WH. The `NOC_LOCAL_ADDR` macro on BH masks the address to the local bits while preserving bit 60 if it is already set:

```cpp
// BH: preserves bit 60 if already set (does NOT set it)
#define NOC_LOCAL_ADDR(addr) ((addr) & 0x1000000FFFFFFFFF)
```

Note that `NOC_LOCAL_ADDR` only preserves bit 60 -- it does not activate PCIe mode. PCIe base address encoding is performed by `NOC_XY_PCIE_ENCODING`, which sets the appropriate coordinate fields to indicate a PCIe transaction. Bit 60 in an address indicates a PCIe transaction on BH, but it must be set upstream (e.g., via `NOC_XY_PCIE_ENCODING`), not by `NOC_LOCAL_ADDR`. This distinction matters for system memory access but not for L1-to-L1 transfers.

---

## 5. NOC Transaction Types

The NOC supports several transaction types, controlled by flags in the `NOC_CTRL` register:

### 5.1 Unicast Write

The most common transaction: write data from local L1 to a specific remote core's L1 or DRAM.

| Flag | Value | Meaning |
|------|:-----:|---------|
| `NOC_CMD_WR` | `0x1 << 1` | Write operation |
| `NOC_CMD_RESP_MARKED` | `0x1 << 4` | Non-posted (request acknowledgment) |

Posted writes (`NOC_CMD_WR` without `NOC_CMD_RESP_MARKED`) fire-and-forget: the sender proceeds immediately. Non-posted writes wait for an acknowledgment from the receiver, ensuring the write is complete before the sender continues.

### 5.2 Unicast Read

Read data from a remote core's L1 or DRAM into local L1.

| Flag | Value | Meaning |
|------|:-----:|---------|
| `NOC_CMD_RD` | `0x0 << 1` | Read operation (bit 1 = 0) |
| `NOC_CMD_RESP_MARKED` | `0x1 << 4` | Track completion |

Reads are always non-posted: the initiating core must wait for the data to arrive. The return address registers (`NOC_RET_ADDR_LO/MID/HI`) specify where the read data should be written in local L1.

### 5.3 Multicast Write

Write the same data to all cores in a rectangular grid simultaneously.

| Flag | Value | Meaning |
|------|:-----:|---------|
| `NOC_CMD_BRCST_PACKET` | `0x1 << 5` | Broadcast/multicast packet |
| `NOC_CMD_BRCST_SRC_INCLUDE` | `0x1 << 17` | Include the source node in the multicast |

The multicast rectangle is defined by the start and end coordinates in the NOC address. The `NOC_CMD_BRCST_XY(y)` field specifies the Y-dimension extent for broadcast column selection.

### 5.4 Atomic Operations

Atomic operations perform read-modify-write operations at the target address:

| Flag | Value | Meaning |
|------|:-----:|---------|
| `NOC_CMD_AT` | `0x1 << 0` | Atomic operation |
| `NOC_AT_INS(ins)` | `ins << 12` | Atomic instruction code |

Supported atomic instructions:

| Code | Name | Operation |
|:----:|------|-----------|
| 0 | `NOP` | No operation |
| 1 | `INCR_GET` | Increment and return previous value |
| 2 | `INCR_GET_PTR` | Increment pointer with wrap |
| 3 | `SWAP` | Swap value |
| 4 | `CAS` | Compare-and-swap |
| 5 | `GET_TILE_MAP` | Tile map lookup |
| 6 | `STORE_IND` | Indirect store |
| 7 | `SWAP_4B` | 4-byte swap |
| 9 | `ACC` | Accumulate (BH only) |

> **Key Insight:** BH's hardware atomic accumulate instruction (`NOC_AT_INS_ACC`) is significant for GSRP: it enables in-network reduction for operations like all-reduce. Where WH requires a Tensix compute kernel to perform the reduction, BH could potentially perform partial reductions at the NOC level during data transit, reducing the number of round-trips needed.

The `ACC` instruction on BH supports multiple data formats:

```cpp
// BH-specific accumulate formats
#define NOC_AT_ACC_FP32        0x0
#define NOC_AT_ACC_FP16_A      0x1
#define NOC_AT_ACC_FP16_B      0x2
#define NOC_AT_ACC_INT32       0x3
#define NOC_AT_ACC_INT32_COMPL 0x4
#define NOC_AT_ACC_INT32_UNS   0x5
#define NOC_AT_ACC_INT8        0x6
```

### 5.5 Transaction IDs

Both architectures support 16 transaction IDs (`NOC_MAX_TRANSACTION_ID = 0xF`), each with up to 255 outstanding transactions (`NOC_MAX_TRANSACTION_ID_COUNT = 255`):

```cpp
#define NOC_PACKET_TAG_TRANSACTION_ID(id) ((id) << 10)
```

Transaction IDs enable fine-grained barrier synchronization: a kernel can wait for all transactions with a specific ID to complete without blocking transactions with other IDs. This is used extensively in the `experimental::Noc` class for overlapping independent data movements.

---

## 6. NOC Alignment Requirements

All NOC transfers must respect alignment constraints:

| Memory Type | Read Alignment | Write Alignment |
|:-----------:|:------------:|:-------------:|
| L1 (both) | 16 bytes | 16 bytes |
| DRAM (WH) | 32 bytes | 16 bytes |
| DRAM (BH) | 64 bytes | 16 bytes |
| PCIe (WH) | 32 bytes | 16 bytes |
| PCIe (BH) | 64 bytes | 16 bytes |

The L1 alignment of 16 bytes is the fundamental constraint for buffer allocation and data placement. The Ethernet word size is also 16 bytes, which means L1-aligned data is automatically Ethernet-aligned. DRAM read alignment is stricter (32 B on WH, 64 B on BH) because DRAM controllers operate on wider access granularities.

---

## 7. The `experimental::Noc` Class

The `experimental::Noc` class (defined in `tt_metal/hw/inc/experimental/noc.h`) provides a modern, type-safe C++ interface for NOC operations on the device side:

```cpp
// tt_metal/hw/inc/experimental/noc.h
class Noc {
public:
    enum class AddressType { NOC, LOCAL_L1 };
    enum class TxnIdMode { ENABLED, DISABLED };
    enum class ResponseMode { NON_POSTED, POSTED };
    enum class BarrierMode { TXN_ID, FULL };
    enum class McastMode { INCLUDE_SRC, EXCLUDE_SRC };
    enum class VcSelection { DEFAULT, CUSTOM };

    static constexpr uint32_t INVALID_TXN_ID = 0xFFFFFFFF;

    Noc();                        // Uses default noc_index
    explicit Noc(uint8_t noc_id); // Explicit NOC selection

    // Core operations
    template <TxnIdMode, uint32_t max_page_size, bool enable_noc_tracing, typename Src, typename Dst>
    void async_read(const Src& src, const Dst& dst, uint32_t size_bytes,
                    const src_args_t<Src>&, const dst_args_t<Dst>&, ...);

    template <TxnIdMode, ResponseMode, uint32_t max_page_size, bool enable_noc_tracing, typename Src, typename Dst>
    void async_write(const Src& src, const Dst& dst, uint32_t size_bytes,
                     const src_args_t<Src>&, const dst_args_t<Dst>&, ...);

    template <McastMode, TxnIdMode, ResponseMode, uint32_t max_page_size, bool enable_noc_tracing, typename Src, typename Dst>
    void async_write_multicast(const Src& src, const Dst& dst, uint32_t size_bytes,
                               uint32_t num_dsts, const src_args_t<Src>&, const dst_args_mcast_t<Dst>&, ...);

    template <TxnIdMode, InlineWriteDst, ResponseMode, typename Dst>
    void inline_dw_write(const Dst& dst, uint32_t val, const dst_args_t<Dst>&, ...);

    // Stateful operations (set state once, then issue multiple transfers)
    template <VcSelection, uint32_t max_page_size, typename Src>
    void set_async_read_state(const Src& src, uint32_t size_bytes, const src_args_t<Src>&, ...);

    template <VcSelection, uint32_t max_page_size, typename Src, typename Dst>
    void async_read_with_state(const Src& src, const Dst& dst, uint32_t size_bytes,
                               const src_args_t<Src>&, const dst_args_t<Dst>&, ...);

    // Barriers
    template <BarrierMode> void async_read_barrier(uint32_t trid = INVALID_TXN_ID);
    template <BarrierMode> void async_write_barrier(uint32_t trid = INVALID_TXN_ID);
    template <ResponseMode, BarrierMode> void async_writes_flushed(uint32_t trid = INVALID_TXN_ID);
    void async_atomic_barrier();
    void async_full_barrier();

    // Utilities
    bool is_local_bank(uint32_t virtual_x, uint32_t virtual_y) const;
    bool is_local_addr(const uint64_t noc_addr) const;
    uint8_t get_noc_id() const;
};
```

### Key Design Patterns

**Type-erased source/destination:** The `Noc` class uses `noc_traits_t<T>` specializations to resolve addresses for different source and destination types (L1 addresses, circular buffers, tensor accessors). This enables the same `async_write` call to work with raw addresses, circular buffer handles, or higher-level tensor abstractions. The `addr_underlying_t` type alias resolves the actual address type for each source/destination kind.

**Stateful transfers:** The `set_async_read_state` / `async_read_with_state` pattern programs NOC registers once and then issues multiple transfers that reuse the same configuration, saving register writes. This is critical for high-throughput data movement where register setup overhead would otherwise dominate.

**Template-driven optimization:** Parameters like `max_page_size`, `TxnIdMode`, and `ResponseMode` are compile-time template parameters. When `max_page_size <= NOC_MAX_BURST_SIZE`, the implementation uses a single-packet fast path that avoids loop overhead. This zero-cost abstraction is essential for maintaining performance in the hot path.

**Software-managed consistency (`commit()`):** The `RemoteCircularBuffer` (file 04) exposes a `commit()` method that writes cached pointer state back to L1. This pattern reflects a broader design choice: the `Noc` class and its users cache state in registers or local variables and only "commit" to L1 when synchronization is needed. This reduces L1 access overhead but requires explicit flush points -- a consideration for any GSRP design that adds cross-chip consistency requirements.

> **Key Insight:** The `experimental::Noc` class provides the right abstraction level for GSRP extension. Its template-based type system, where source and destination types are resolved through traits, could be extended with a `RemoteChipEndpoint` type that transparently handles cross-chip address resolution and fabric packet construction.

---

## 8. The Chip-Local Constraint

The most important fact about the NOC for the GSRP proposal:

> **The NOC is chip-local.** It connects cores within a single die. There is no hardware mechanism to route a NOC transaction to a core on a different chip.

When a Tensix core issues `noc_async_write(src_addr, noc_addr, size)`, the `(x, y)` coordinates in `noc_addr` must refer to a core on the same chip. Writing to a coordinate that does not exist on the local die is undefined behavior -- there is no error or trap, the packet simply has nowhere to go.

This constraint is fundamental: any cross-chip data movement must be explicitly mediated through software running on Ethernet cores. The current fabric model (Chapter 3) achieves this by having worker kernels construct fabric packet headers and send them to local Ethernet cores via the NOC. The Ethernet core's firmware then transmits the packet across the Ethernet link to the destination chip, where another Ethernet core delivers it to the target Tensix core via that chip's local NOC.

A GSRP would need to make this mediation transparent -- the worker kernel would issue what looks like a local NOC write to a "global address," and some layer (firmware, hardware, or compiler-inserted code) would intercept it and route it through the fabric. The candidate architectures for this interception are explored in Chapter 6.

---

## 9. BH Security Fence Registers

Blackhole introduces a `NOC_SEC_FENCE_RANGE` register array (32 entries at `NOC_REGS_START_ADDR + 0x400`) and associated `NOC_SEC_FENCE_ATTRIBUTE` registers. These provide hardware address-range matching with configurable access policies:

```cpp
#define NOC_SEC_FENCE_RANGE(cnt)     (NOC_REGS_START_ADDR + 0x400 + ((cnt) * 4))  // 32 entries
#define NOC_SEC_FENCE_ATTRIBUTE(cnt) (NOC_REGS_START_ADDR + 0x480 + ((cnt) * 4))  // 8 entries
#define NOC_SEC_FENCE_MASTER_LEVEL   (NOC_REGS_START_ADDR + 0x4A0)
```

While these registers are designed for security fencing (preventing unauthorized memory access), they could potentially be repurposed as a lightweight address interception mechanism: configuring a fence range to match "remote chip" addresses and triggering a firmware trap or redirect when a NOC transaction targets that range. This is speculative and would require hardware validation, but it represents an existing BH hardware feature that could reduce the firmware overhead of address translation.

---

## Key Takeaways

- **The NOC is chip-local -- this is the fundamental constraint.** There is no hardware path for a NOC transaction to cross chip boundaries. All inter-chip communication must go through Ethernet cores running fabric router firmware. A transparent global address space requires an interception layer between the NOC and the fabric.
- **64-bit NOC addresses encode core coordinates and local offset.** The `{x, y, addr}` triple uniquely identifies any byte within a chip. Extending this to a cross-chip address requires adding `{mesh_id, chip_id}` components, expanding the address space beyond 48 bits.
- **Dual NOCs prevent deadlocks in bidirectional communication.** NOC0 and NOC1 operate independently with a 3-port router (NIU, X, Y) at each node using XY dimension-ordered routing. Any GSRP design must preserve this deadlock avoidance property when mixing local and remote traffic on the same NOCs.
- **BH has hardware capabilities that WH lacks for GSRP.** The atomic accumulate instruction (`NOC_AT_INS_ACC`), wider NOC data path (512 vs 256 bits), 5-bit coordinate translation tables, and security fence registers all provide hooks that could support more efficient cross-chip communication.
- **The `experimental::Noc` class provides the right abstraction level for GSRP extension.** Its template-based type system, where source and destination types are resolved through traits, could be extended with a `RemoteChipEndpoint` type that transparently handles cross-chip address resolution and fabric packet construction.

## Source Code References

- `tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc_parameters.h` -- Wormhole NOC register definitions, address encoding macros, alignment constants
- `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h` -- Blackhole NOC parameters, security fence registers, accumulate atomics
- `tt_metal/hw/inc/experimental/noc.h` -- `experimental::Noc` class with type-safe async read/write/multicast/barrier API

---

**Next:** [`04_buffer_types_and_circular_buffers.md`](./04_buffer_types_and_circular_buffers.md)
