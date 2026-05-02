# 6.2 Address Space Design

## Context

Section 6.1 defined the GSRP's three requirements -- addressability, accessibility, and
performance visibility -- and identified Level B (runtime-assisted) as the primary design
target for Blackhole hardware. This section tackles the first requirement:
**addressability**. Every byte of L1 SRAM across a multi-chip mesh must be uniquely
identifiable by a single 64-bit address. The design must accommodate the existing NOC
64-bit address format (Chapter 2, Section 3), the `FabricNodeId` namespace of
`(mesh_id, chip_id)` (Chapter 3, Section 5), the alignment constraints imposed by L1,
the NOC, and Ethernet word sizes, and the `MeshBuffer` consistent-address property
(Chapter 4, Section 4).

---

## 6.2.1 Requirements for a Global Address

A global address must encode four pieces of information:

1. **Mesh ID** -- which mesh (in a multi-mesh deployment) the target resides in.
2. **Chip ID** -- which chip within that mesh.
3. **Core coordinates** -- the (x, y) NOC coordinates of the target core on that chip.
4. **L1 offset** -- the byte offset within that core's L1 SRAM.

### Address Space Sizing

| Field | Description | Max value (BH) | Bits needed |
|-------|-------------|---------------:|------------:|
| `mesh_id` | Mesh within multi-mesh topology | 255 | 8 |
| `chip_id` | Chip within the mesh | 255 | 8 |
| `noc_x` | NOC virtual x-coordinate | 63 | 6 |
| `noc_y` | NOC virtual y-coordinate | 63 | 6 |
| `l1_offset` | Byte offset in core's L1 | 1,572,863 | 21 |
| **Total** | | | **49 bits** |

A 64-bit encoding leaves 15 spare bits for flags, versioning, or future expansion. This
is fortunate because the NOC already uses 64-bit addresses, so the global address coexists
with the local NOC address format without widening any buses.

### Alignment Constraints

Three alignment requirements constrain the low-order bits of any global address:

| Alignment source | Granularity | Notes |
|-----------------|:-----------:|-------|
| NOC word | 16 B | `PACKET_WORD_SIZE_BYTES = 16`; all NOC transactions must be 16-byte aligned |
| Ethernet minimum transfer | 16 B | EDM packet payload alignment matches NOC word |
| L1 allocator | 32 B (default) | `L1BankingAllocator` default alignment |
| Circular buffer page | 32 B (typical) | `RemoteCircularBuffer` page alignment |

For GSRP purposes, the binding constraint is Ethernet's 16-byte minimum. Sub-16-byte
remote accesses are not supported at the fabric level (Chapter 3, Section 1). The
recommended approach is to enforce 16-byte alignment at the API level and document it
as a GSRP constraint.

---

## 6.2.2 The 64-Bit GlobalAddress Encoding

We propose a flat 64-bit encoding that packs all fields into a single register-width
integer. The design draws from V4's clean field sizing, V3's struct-based type with
`FORCE_INLINE` accessors, and V1's cycle-counted encode/decode analysis.

```
Bit layout (MSB to LSB):
 63    56 55    48 47  42 41  36 35           21 20            0
+--------+--------+------+------+-------------+--------------+
| mesh_id| chip_id| noc_x| noc_y| flags/rsvd  | l1_offset    |
| 8 bits | 8 bits |6 bits|6 bits|  15 bits    | 21 bits      |
+--------+--------+------+------+-------------+--------------+
```

### Field Widths

| Field | Bits | Position | Range | Notes |
|-------|:----:|:--------:|:-----:|-------|
| `mesh_id` | 8 | 63:56 | 0--255 | Matches `MAX_NUM_MESHES`; current use: 0--7 |
| `chip_id` | 8 | 55:48 | 0--255 | Sufficient for 32-chip Galaxy; $8\times$ headroom |
| `noc_x` | 6 | 47:42 | 0--63 | BH virtual: max ~18; $3\times$ headroom |
| `noc_y` | 6 | 41:36 | 0--63 | BH virtual: max ~12; $5\times$ headroom |
| Flags / reserved | 15 | 35:21 | -- | Per-access metadata (see below) |
| `l1_offset` | 21 | 20:0 | 0--2,097,151 | Covers full 1,536 KB BH L1 with headroom |

### Reserved Flag Bits

The 15 flag bits encode per-access GSRP metadata, allowing the runtime to make routing
and ordering decisions without additional API parameters:

| Bit(s) | Purpose | Values |
|--------|---------|--------|
| 35 | Access type | 0 = write, 1 = read |
| 34 | Ordering | 0 = relaxed, 1 = release/acquire |
| 33 | Atomic | 0 = normal, 1 = atomic operation |
| 32--30 | Priority | 0--7 (QoS / traffic class) |
| 29 | Consistent | 1 = address from `MeshBuffer` consistent allocation |
| 28 | Ephemeral | 1 = address valid only for current epoch |
| 27--21 | Reserved | Future use |

### The GlobalAddress Type

```cpp
struct GlobalAddress {
    uint64_t raw;

    // Construction -- 5 shifts + 5 masks + 4 ORs = ~14 RISC-V instructions (~14 ns at 1 GHz)
    static FORCE_INLINE GlobalAddress from_components(
        uint8_t mesh_id, uint8_t chip_id,
        uint8_t noc_x, uint8_t noc_y,
        uint32_t l1_offset) {
        return GlobalAddress{
            .raw = ((uint64_t)mesh_id << 56) |
                   ((uint64_t)chip_id << 48) |
                   ((uint64_t)(noc_x & 0x3F) << 42) |
                   ((uint64_t)(noc_y & 0x3F) << 36) |
                   ((uint64_t)(l1_offset & 0x1FFFFF))
        };
    }

    // Decomposition -- each accessor is 2 instructions (shift + mask)
    FORCE_INLINE uint8_t  mesh_id()   const { return (raw >> 56) & 0xFF; }
    FORCE_INLINE uint8_t  chip_id()   const { return (raw >> 48) & 0xFF; }
    FORCE_INLINE uint8_t  noc_x()     const { return (raw >> 42) & 0x3F; }
    FORCE_INLINE uint8_t  noc_y()     const { return (raw >> 36) & 0x3F; }
    FORCE_INLINE uint16_t flags()     const { return (raw >> 21) & 0x7FFF; }
    FORCE_INLINE uint32_t l1_offset() const { return raw & 0x1FFFFF; }

    // Construct with flags
    FORCE_INLINE GlobalAddress with_flags(uint16_t f) const {
        return GlobalAddress{ .raw = (raw & ~(0x7FFFULL << 21)) | ((uint64_t)(f & 0x7FFF) << 21) };
    }

    // Decompose into FabricNodeId + NOC address for fabric API
    FORCE_INLINE uint64_t noc_addr() const {
        return ((uint64_t)noc_x() << 30) | ((uint64_t)noc_y() << 24) | l1_offset();
    }
};
```

### Encode/Decode Cost Analysis

$$\text{Encode cost} = 5\ \text{shifts} + 5\ \text{masks} + 4\ \text{ORs} \approx 14\ \text{RISC-V instructions}$$

$$\text{Decode cost} = 1\ \text{shift} + 1\ \text{mask} \approx 2\ \text{RISC-V instructions per field}$$

At ~1 GHz Tensix clock (Chapter 2), full encode is ~14 ns and single-field decode is ~2 ns
-- negligible compared to fabric latency ($5$--$20\ \mu\text{s}$ per hop).

---

## 6.2.3 Locality Detection

A critical optimization is classifying whether a global address refers to local or remote
L1. Local accesses bypass the fabric entirely and use direct NOC transactions.

```cpp
enum class gsrp_locality {
    LOCAL_CORE,    // Same core: direct L1 pointer dereference       (~1 ns)
    LOCAL_CHIP,    // Same chip, different core: NOC transaction      (~100-500 ns)
    REMOTE_CHIP,   // Different chip, same mesh: fabric packet        (~5-20 us/hop)
    REMOTE_MESH    // Different mesh: multi-hop fabric packet         (~20-100 us)
};

FORCE_INLINE gsrp_locality classify(
    GlobalAddress addr,
    uint8_t my_mesh, uint8_t my_chip,
    uint8_t my_x, uint8_t my_y)
{
    if (addr.mesh_id() != my_mesh) return gsrp_locality::REMOTE_MESH;
    if (addr.chip_id() != my_chip) return gsrp_locality::REMOTE_CHIP;
    if (addr.noc_x() != my_x || addr.noc_y() != my_y) return gsrp_locality::LOCAL_CHIP;
    return gsrp_locality::LOCAL_CORE;
}
```

The classification cost is ~8 instructions (4 field extractions + 4 comparisons). For
each locality class, the GSRP runtime dispatches to the appropriate transport:

| Locality | Transport | Latency |
|----------|-----------|---------|
| `LOCAL_CORE` | Direct L1 pointer dereference | ~1 ns |
| `LOCAL_CHIP` | NOC unicast read/write | ~100--500 ns |
| `REMOTE_CHIP` | Fabric unicast via ERISC | ~5--20 $\mu$s/hop |
| `REMOTE_MESH` | Multi-hop fabric via mesh gateway | ~20--100 $\mu$s |

This four-level dispatch is the foundation of the Level B runtime (Section 6.4,
Architecture 1: Fat Fabric API).

---

## 6.2.4 Encoding Strategy A: Extended NOC Address

The most direct approach reuses the existing 64-bit NOC address layout and repurposes its
unused upper bits for cross-chip identification.

**Mechanism:** Pack mesh and chip IDs into the currently-reserved upper bits of the NOC
address. For local addresses (`mesh_id == 0`, `chip_id == local_chip`), the upper bits are
zero and the lower bits form a valid chip-local NOC address.

**Advantages:** Zero encoding overhead; no table lookups. Local accesses require only
masking bits 63:48 to zero. Direct mapping to `FabricNodeId`.

**Disadvantages:** Physical topology is baked into the address -- if the mesh is
reconfigured, all addresses change. No support for address-space virtualization.

**Verdict:** Works only if hardware is modified to recognize the extended format. Suitable
as the **wire format** (the encoding that appears in fabric packet headers) and as the
Level A (next-gen silicon) path.

---

## 6.2.5 Encoding Strategy B: Software Virtual-to-Physical Table

Instead of encoding physical coordinates in the address, Strategy B uses a region ID that
is resolved through a per-core translation table in L1.

```cpp
// Translation table entry (16 bytes, naturally aligned)
struct alignas(16) GlobalRegionEntry {
    uint16_t mesh_id;           // Target mesh
    uint16_t chip_id;           // Target chip
    uint8_t  noc_x;             // Target core x
    uint8_t  noc_y;             // Target core y
    uint16_t flags;             // {valid, read-only, ...}
    uint32_t remote_base;       // L1 base offset on target core
    uint32_t region_size;       // Bounds check (0 = unbounded)
};  // 16 bytes per entry
```

### L1 Budget

| Table size (entries) | Memory cost | % of usable BH L1 |
|---------------------:|------------:|-------------------:|
| 16 | 256 B | 0.02% |
| 64 | 1,024 B | 0.07% |
| 256 | 4,096 B | 0.27% |

**Advantages:** Maximum flexibility. Supports topology-independent kernel code (same
binary runs on any mesh by loading different tables). Natural `MeshBuffer` integration --
each shard maps to a region entry. Enables bounds checking and per-entry permissions.
Host can update mappings between epochs for dynamic reconfiguration.

**Disadvantages:** Per-access lookup cost of ~10--20 cycles for L1 read + pointer
arithmetic. Table must be populated by the host during program setup.

**Verdict:** The most flexible strategy. **Achievable on current WH/BH hardware.**
Recommended as the kernel-facing API mechanism for Level B.

---

## 6.2.6 Encoding Strategy C: Segment Registers ("Remote Windows")

Each core maintains 4--8 segment registers that map contiguous local L1 address ranges to
remote targets. Writing to a local window triggers a fabric write to the corresponding
remote location.

```
Segment Descriptor File (per core, 8 entries):
+-----+------------+--------+--------+------+-----------+
| Idx | local_base | mesh_id| chip_id| xy   | remote_off|
+-----+------------+--------+--------+------+-----------+
|  0  | 0x150000   |   0    |   3    | 5,7  | 0x020000  |
|  1  | 0x160000   |   0    |   5    | 8,2  | 0x030000  |
| ... |   ...      |  ...   |  ...   | ...  |   ...     |
|  7  |  (unused)  |        |        |      |           |
+-----+------------+--------+--------+------+-----------+
 Each entry: 16 bytes.  Total: 8 x 16 = 128 bytes.
```

**Advantages:** Tiny L1 footprint (128 bytes). Natural fit for producer-consumer and
double-buffering patterns where a core alternates between 2--4 fixed remote targets.

**Disadvantages:** Hard limit of 8 simultaneous remote windows. Linear scan adds
~40--60 cycles per access in the worst case. Cannot represent non-contiguous mappings.
An MoE layer with 64 experts across 32 chips cannot be served by 8 segments.

**Verdict:** Too restrictive for general use. Attractive as a **hot-path fast-path**
within a Strategy B system -- the translation table handles the general case, while
segment registers accelerate latency-critical patterns.

---

## 6.2.7 Strategy Comparison

| Criterion | A: Extended NOC | B: V-to-P Table | C: Segments |
|-----------|:--------------:|:---------------:|:-----------:|
| L1 overhead per core | 0 B | 4,096 B (typical) | 128 B |
| Max simultaneous targets | Unlimited | 256 (scalable) | 8 (fixed) |
| Lookup cost (cycles) | 0 | ~10--20 | ~40--60 |
| Topology independence | No | Yes | Partial |
| Permission control | None | Per-entry | Per-window |
| Epoch reconfiguration | N/A | Host rewrites table | Host rewrites regs |
| MeshBuffer integration | Manual | Natural | Manual |
| Fits current silicon | No (needs HW) | **Yes** | Partial (needs FW) |
| Best-fit transparency level | A (HW-Transparent) | B (RT-Assisted) | C (Library) |

**Recommendation:** A hybrid design combining all three:

1. **Wire format (Strategy A):** The 64-bit `GlobalAddress` that appears in fabric packet
   headers encodes physical coordinates directly. No per-packet table lookups on the
   critical data path.
2. **Kernel-facing API (Strategy B):** Kernels reference remote data through region IDs
   resolved via per-core L1 translation tables, populated by the host during
   `MeshWorkload::apply()`.
3. **Hot-path optimization (Strategy C):** For kernels with a small, fixed set of remote
   targets, segment registers provide a faster path than the full translation table.

---

## 6.2.8 Relationship to Existing Addressing Primitives

The `GlobalAddress` extends but does not replace existing naming schemes:

| Identifier | Scope | GSRP relation |
|-----------|-------|---------------|
| NOC address (64-bit) | Single chip | GSRP superset (local case) |
| `FabricNodeId(mesh_id, chip_id)` | Multi-mesh | Maps to `mesh_id:chip_id` fields |
| `dst_start_node_id` | Packet header | Derivable from `GlobalAddress` |
| `MeshCoordinate(x, y)` | Single mesh | Maps to `chip_id` via lookup |
| `MeshCoreCoord` | Multi-chip | Equivalent to `(chip_id, noc_x, noc_y)` |

Bidirectional conversions are zero-cost:

```cpp
// GlobalAddress -> existing fabric API fields
FORCE_INLINE FabricNodeId to_fabric_node_id(GlobalAddress ga) {
    return FabricNodeId{ .mesh_id = ga.mesh_id(), .chip_id = ga.chip_id() };
}

// GlobalAddress -> chip-local NOC address
FORCE_INLINE uint64_t to_noc_addr(GlobalAddress ga) {
    return ga.noc_addr();  // Shifts already match NOC layout
}

// MeshCoreCoord + l1_offset -> GlobalAddress
FORCE_INLINE GlobalAddress from_mesh_core(
    uint8_t mesh_id, uint8_t chip_id,
    uint8_t core_x, uint8_t core_y, uint32_t offset) {
    return GlobalAddress::from_components(mesh_id, chip_id, core_x, core_y, offset);
}
```

---

## 6.2.9 The MeshBuffer Stepping Stone

`MeshBuffer::create()` allocates buffers at a **consistent L1 address** across all
devices in the mesh (Chapter 4, Section 4). This property dramatically simplifies GSRP
addressing for `MeshBuffer`-backed data.

### REPLICATED Layout

Every chip holds the same data at the same L1 offset. The `GlobalAddress` for any
chip's copy differs only in the `chip_id` field:

```
MeshBuffer "weights" (REPLICATED), allocated at L1 0x20000, core (4,3):
  Chip 0: GlobalAddress::from_components(0, 0, 4, 3, 0x20000)
  Chip 5: GlobalAddress::from_components(0, 5, 4, 3, 0x20000)
  Only chip_id varies; noc_xy and l1_offset are constant.
```

### SHARDED Layout

Each chip holds a different shard at the same L1 base:

```
MeshBuffer "activations" (SHARDED), shard_size = 16 KB, 8 chips:
  Shard 0 -> Chip 0: GlobalAddress::from_components(0, 0, 4, 3, 0x20000)
  Shard 3 -> Chip 3: GlobalAddress::from_components(0, 3, 4, 3, 0x20000)
  Only chip_id varies -- l1_offset is identical across all shards.
```

### Pre-Computed Address Maps

The host computes a complete address map at `MeshWorkload` setup time:

$$\text{Address table size} = N_{\text{remote shards}} \times 8\ \text{B (per GlobalAddress)}$$

| Scenario | Shards | Map size |
|----------|:------:|:--------:|
| 32-chip BH, 1 shard/chip | 32 | 256 bytes |
| Typical workload (8--64 shards) | 8--64 | 64 B -- 512 B |
| Per-core sharding (4,480 cores) | 4,480 | ~35 KB (use 2-level) |

For per-chip sharding (the common case), the address map fits comfortably in L1. For
per-core sharding, a two-level scheme is needed:

```
Level 1 (per-chip):  32 entries x 16 bytes  =    512 bytes
  shard_group_id -> (mesh_id, chip_id, L1_table_base)

Level 2 (per-core):  140 entries x 16 bytes = 2,240 bytes
  core_id -> (noc_x, noc_y, buffer_base)

Total: 512 + 2,240 = 2,752 bytes  (within 16 KB metadata budget)
```

The Level 2 table is identical on every core within a chip (all cores share the same map
of that chip's core layout). Only the Level 1 table differs between chips.

> **Key Insight:** The gap between the current mesh API and a Level B global address API
> is narrower than it appears. The primary missing piece is not routing (already solved
> by `ControlPlane`) but the address-to-`FabricNodeId` extraction and the single-call API
> wrapper that performs it automatically.

---

## 6.2.10 Address Space Partitioning

Not all L1 should be globally addressable. We partition each core's L1 into four regions:

```
Core L1 Layout (BH, 1,536 KB):
 Address 0x0
 +---------------------------+
 | System reserved (~40 KB)  |  Firmware, mailbox, routing tables
 +---------------------------+
 | GSRP metadata (4--16 KB)  |  Translation table, route cache,
 |                           |  segment registers, response buffers
 +---------------------------+  <- GSRP_GLOBAL_BASE
 | Globally-visible L1       |  Addressable via GSRP
 | (configurable: 25--50%    |  Application tensors, KV caches,
 |  of remaining L1)         |  activations, expert routing tables
 +---------------------------+  <- GSRP_PRIVATE_BASE
 | Core-private L1           |  Local CBs, kernel scratch, stack
 +---------------------------+
 Address 0x180000
```

### Zero-Overhead When Unused

Programs that do not use cross-chip communication set `gsrp_global_size = 0`. The GSRP
metadata region is not allocated, and the entire L1 above `MEM_MAP_END` is available as
core-private memory. This follows the existing `AllocatorConfig` model (Chapter 2,
Section 1) where optional features have zero L1 cost when disabled.

---

## 6.2.11 Scalability

With 8-bit `mesh_id` and 8-bit `chip_id`, the encoding supports up to
$256 \times 256 = 65{,}536$ chips. For the largest currently envisioned system (4 meshes
$\times$ 64 chips = 256 chips):

$$\text{Total addressable cores} = 256 \times 140 = 35{,}840\ \text{Tensix cores}$$

$$\text{Total addressable L1} = 35{,}840 \times 1{,}536\ \text{KB} \approx 52.5\ \text{GB}$$

The 64-bit encoding comfortably accommodates this. For single-mesh deployments, `mesh_id`
defaults to 0, reducing the translation table to a simple chip-indexed array.

---

## Key Takeaways

- **A flat 64-bit `GlobalAddress` encoding `{mesh_id(8), chip_id(8), noc_x(6), noc_y(6),
  flags(15), l1_offset(21)}` fits within the existing NOC address width** with 15 flag
  bits for per-access metadata. Addresses are self-describing, requiring zero L1 metadata
  for construction. Full encode is ~14 RISC-V instructions (~14 ns); single-field decode
  is 2 instructions (~2 ns).

- **Three encoding strategies trade off generality against overhead:** Strategy A (flat
  physical encoding, 0 cycles, no portability) serves as the wire format; Strategy B
  (L1 translation table, ~10--20 cycles, 4 KB, full portability) serves as the
  kernel-facing API; Strategy C (segment registers, 128 B, 8 targets) serves as a
  hot-path fast-path. The recommended design is a hybrid of all three.

- **`MeshBuffer`'s consistent-address property reduces the GSRP address problem for most
  TTNN workloads to chip identification only** -- since `noc_xy` and `l1_offset` are
  identical across all devices, the host pre-computes a bounded address table (256 B for
  32-chip per-chip sharding) that eliminates runtime address computation entirely.

## GSRP Implications

The address space design resolves the first fundamental question of the GSRP: how to name
a remote byte. The `GlobalAddress` type provides a portable, efficient naming scheme that
maps directly onto the existing fabric infrastructure. The flat encoding (Strategy A) serves
as the wire format in fabric packet headers, while the translation table (Strategy B)
provides topology-independent kernel-facing access. The flag bits carry per-access ordering
and atomicity hints that enable relaxed consistency enforcement without additional metadata.
The `MeshBuffer` bridge constrains the initial GSRP prototype to a bounded, pre-known
address set, sidestepping the general address validation problem.

---

**Previous:** [`01_problem_statement_and_transparency_levels.md`](./01_problem_statement_and_transparency_levels.md) | **Next:** [`03_coherence_and_consistency_models.md`](./03_coherence_and_consistency_models.md)
