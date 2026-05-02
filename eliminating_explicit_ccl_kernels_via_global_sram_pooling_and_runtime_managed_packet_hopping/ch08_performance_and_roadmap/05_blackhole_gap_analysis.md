# 8.5 Blackhole Gap Analysis

## Context

Section 8.4 analyzed Wormhole's hardware gaps for GSRP and identified WH as viable for
Phase 1 (write-only) but constrained for read-heavy workloads and Galaxy-scale deployments.
This section performs the corresponding analysis for **Blackhole (BH)**, which offers
several architectural advantages: larger L1 (1,536 KB per Tensix), dual RISC-V on Ethernet
cores, native 2D routing support, and -- critically for the GSRP vision -- the
`NOC_SEC_FENCE_RANGE` register that could serve as a hardware-assisted address interception
mechanism. The reader should be familiar with BH's memory layout (Chapter 2, Sections 1--2),
the 2D routing model (Chapter 3, Sections 3--4), the `FabricConfig` 2D variants (Chapter 3,
Section 5), and the candidate GSRP architectures (Chapter 6, Section 4). This section also
examines what the Quasar architecture (`tt-2xx`) may provide for the GSRP long-term vision.

---

## 8.5.1 BH Architectural Advantages for GSRP

### Side-by-Side Comparison

| Feature | Wormhole | Blackhole | GSRP Impact |
|---------|:--------:|:---------:|-------------|
| Tensix cores per chip | 80 | 140 | 75% more cores to serve |
| Tensix L1 per core | 1,464 KB | 1,536 KB | +72 KB per core for GSRP metadata |
| ERISC L1 per core | 256 KB | 512 KB | 2x headroom for read handler |
| ERISC processors | Single RISC-V | Dual RISC-V | Dedicated GSRP handler core |
| Active Ethernet cores | 16 | 14 | Slightly fewer but higher BW each |
| Routing model | Primarily 1D | Native 2D (mesh/torus) | Lower hop count for 2D topologies |
| `NOC_SEC_FENCE_RANGE` | Not available | Available | HW address range detection |
| `noc_address_translation_table` | Not available | **Not available** (Quasar only) | N/A for BH |
| Coordinate virtualization | Limited | Full (`noc_xy` remapping) | Simplifies global addressing |
| Per-link bandwidth | ~10.8 B/c (4096 B) | ~12.8 B/c (4096 B) | +19% per-link throughput |

> **Key Insight:** BH closes or narrows every WH gap identified in Section 8.4, making it
> the natural target architecture for GSRP adoption. The one feature BH does **not** have
> is hardware address translation tables -- that capability arrives with Quasar (`tt-2xx`).

---

## 8.5.2 Dual RISC-V on Ethernet Cores

BH's most significant architectural advantage for GSRP is the **dual RISC-V** on each
Ethernet core:

```
BH Ethernet Core Architecture:

+-----------------------------------------------+
|  Primary RISC-V (ERISC-0)                     |
|    - Runs EDM router firmware (unchanged)      |
|    - Handles packet forwarding, flow control   |
|    - Same role as WH single RISC-V             |
+-----------------------------------------------+
|  Secondary RISC-V (ERISC-1)                   |
|    - Available for GSRP read handler           |
|    - Can run address translation firmware      |
|    - Independent instruction stream            |
+-----------------------------------------------+
|  512 KB Shared L1 SRAM                        |
|    - ~359 KB usable (with routing FW)          |
|    - Sufficient for 4 outstanding reads        |
+-----------------------------------------------+
```

The dual-RISC-V architecture enables **zero-overhead GSRP processing** on the router
critical path: the primary RISC-V continues handling store-and-forward packet routing
without interruption, while the secondary RISC-V handles GSRP-specific operations
(read requests, address translation, export validation) on a parallel instruction stream.

### ERISC L1 Budget (BH)

```
BH ERISC L1 Budget:

Total:        512 KB
FW reserved: ~153 KB
Available:   ~359 KB
GSRP (4 outstanding reads): ~20.2 KB (5.6%)
Remaining for EDM buffers:  ~339 KB
```

GSRP ERISC allocations total **~20.2 KB** with 4 outstanding reads (16 KB staging +
4.2 KB metadata). Chapter 7, Section 4.9 documents the single-read baseline of ~7.8 KB;
the 4-read configuration is the recommended production target, enabled by BH's 512 KB
ERISC L1. This leaves ~339 KB for EDM buffers -- **17x more headroom** than WH's ~144 KB.

---

## 8.5.3 Native 2D Routing

BH supports multiple 2D fabric configurations natively:

```cpp
// tt_metal/api/tt-metalium/experimental/fabric/control_plane.hpp
enum class FabricConfig {
    DISABLED,
    FABRIC_1D_NEIGHBOR_EXCHANGE,
    FABRIC_1D,
    FABRIC_1D_RING,
    FABRIC_2D,              // BH: 2D mesh, no wrap
    FABRIC_2D_TORUS_X,      // BH: torus in X dimension
    FABRIC_2D_TORUS_Y,      // BH: torus in Y dimension
    FABRIC_2D_TORUS_XY,     // BH: torus in both dimensions
    CUSTOM,
};
```

### 2D Routing Impact on Latency

For a 32-chip Galaxy (8x4):

```
Hop Count Comparison (8x4 Galaxy):

Path                     1D Ring Hops    2D Mesh Hops    Improvement
---------------------------------------------------------------------
Adjacent (same row)            1               1            0%
Row end (3 cols away)          3               3            0%
Adjacent row same col         ~8               1           87%
Diagonal (7,3)               ~15              10           33%
Farthest corner              ~15              10           33%
```

The 2D routing advantage is most pronounced for **column-traversing paths**, where 1D ring
routing must traverse the entire ring to reach an adjacent row. BH's worst-case for a
32-chip 8x4 mesh is 10 hops (7 cols + 3 rows, Manhattan distance), compared to WH's 15
hops on a 1D ring emulation of the same topology.

### `HybridMeshPacketHeader` 2D Route Encoding

```
2D Route Buffer (BH):

+---+---+---+---+---+---+---+---+---+---+---+---+...
| H0| H1| H2| H3| H4| H5| H6| H7| H8| H9|H10|H11|
+---+---+---+---+---+---+---+---+---+---+---+---+...
  4-bit per hop: EAST(0)/WEST(1)/NORTH(2)/SOUTH(3)/WRITE(4)/FWD(5)/...

Default route buffer: 35 bytes = 70 half-byte hops (max 67 used)
Extended tiers: up to 128 bytes for long routes
```

---

## 8.5.4 `NOC_SEC_FENCE_RANGE` and Address Interception

BH's `NOC_SEC_FENCE_RANGE` register is a hardware mechanism that defines address ranges
for NOC security fencing:

```cpp
// tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h
#define NOC_SEC_FENCE_RANGE(cnt) (NOC_REGS_START_ADDR + 0x400 + ((cnt) * 4))  // 32 inst
```

The register supports 32 range entries, each defining a protected address range. While
designed for security isolation, this mechanism could be repurposed for GSRP:

```
Proposed NOC_SEC_FENCE_RANGE Use for GSRP:

1. Configure fence range to cover "remote address" space
   (e.g., addresses with bits [63:40] set to non-zero)
2. When a NOC transaction targets a fenced range:
   - Hardware generates an exception/interrupt
   - Secondary ERISC-1 handles the redirect
   - Transaction is converted to a fabric packet

Benefits:
   - Hardware detects remote addresses without software checking
   - Eliminates the ~3-cycle locality check from the injection shim
   - Enables partial hardware-assisted transparency (Phase 2.5)

Limitations:
   - The fence registers are security-oriented, not designed for redirect
   - Interrupt latency may exceed software-check latency
   - Requires firmware to handle the exception path
   - Current register count (32) may be insufficient for fine-grained ranges
```

> **Recommendation:** Investigate `NOC_SEC_FENCE_RANGE` as a Phase 2.5 optimization on BH.
> The software shim (Phase 1--2) remains the primary path, with hardware-assisted
> interception as an acceleration opportunity.

---

## 8.5.5 Quasar Architecture (`tt-2xx`) Forward Look

The Quasar architecture introduces **hardware address translation tables** that provide
native support for the GSRP vision. This is a Quasar-specific feature -- it is **not
available on Blackhole**.

```cpp
// tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_address_translation_tables.hpp

// Enable address translation and dynamic routing
void noc_address_translation_table_en(
    bool en_address_translation,
    bool en_dynamic_routing);

// Mask table: match MSB bits of address, extract endpoint ID
void noc_address_translation_table_mask_table_entry(
    uint32_t entry,      // 0--15 (16 entries)
    uint32_t mask,       // MSB bits to compare
    uint64_t compare,    // Expected value
    uint32_t ep_id_idx,  // Bit offset for endpoint ID
    uint32_t ep_id_size, // Size of endpoint ID field
    uint32_t table_offset);

// Endpoint table: map endpoint ID to physical NOC coordinates
void noc_address_translation_table_endpoint_table_entry(
    uint32_t entry,      // 0--1023 (1024 entries)
    uint32_t x,
    uint32_t y);

// Routing table: 32 entries mapping to 16-element routing lists
void noc_address_translation_table_routing_table_entry(
    uint32_t entry,      // 0--31
    uint32_t compare,
    uint32_t routing[]); // 16-element routing list

// Address rebase: remap base address for translated accesses
void noc_address_translation_table_mask_table_entry_rebase(
    uint32_t entry,
    uint32_t mask,
    uint64_t compare,
    uint32_t ep_id_idx,
    uint32_t ep_id_size,
    uint64_t base,       // New base address for offset calculation
    uint32_t table_offset);
```

### How Quasar Enables Hardware GSRP

```
Quasar Address Translation Flow:

1. Tensix core issues NOC write to GlobalAddress (64-bit)
2. NOC hardware matches MSBs against mask table (16 entries)
3. On match: extract endpoint ID from address bits
4. Lookup endpoint table: endpoint ID -> physical (x, y) coordinates
5. If (x, y) is remote: consult routing table for fabric path
6. Hardware constructs fabric packet and injects into ERISC
7. Address rebase: translate global offset to local L1 offset

All in hardware. Zero software overhead per access.
```

| Quasar Feature | GSRP Phase Enabled | Impact |
|:---------------|:------------------:|:-------|
| Mask table (16 entries) | Phase 3 | Hardware address matching |
| Endpoint table (1024 entries) | Phase 3 | Hardware coordinate resolution |
| Routing table (32 entries) | Phase 3 | Hardware route selection |
| Address rebase | Phase 3 | Hardware offset translation |
| Dynamic routing enable | Phase 3 | Runtime route updates |

> **Key Insight:** Quasar's address translation tables provide the exact hardware mechanism
> needed for Phase 3 ("Hardware-Assisted GSRP"). The entire GSRP injection pipeline
> (Chapter 7, Section 1) could be replaced by hardware, reducing per-access overhead from
> ~13 cycles (software cache-hit path) to near-zero.

---

## 8.5.6 BH GSRP Feature Matrix by Phase

| Feature | Phase 0 (Today) | Phase 1 | Phase 2 | Phase 3 (Quasar) |
|---------|:---------------:|:-------:|:-------:|:----------------:|
| Address translation | Manual | SW shim | SW shim + fence | HW tables |
| Route resolution | Manual | L1 lookup | L1 + ERISC-1 | HW routing table |
| Connection mgmt | Per-kernel | Persistent | Persistent | HW-managed |
| Cross-chip write | Explicit | `gsrp_write()` | `gsrp_write()` | HW NOC extension |
| Cross-chip read | N/A | N/A | `gsrp_read()` (ERISC-1) | HW NOC read |
| 2D routing | Available | Used | Optimized | HW dynamic |
| Outstanding reads | N/A | N/A | 4 (per ERISC) | HW-managed |
| Deadlock avoidance | Dateline+VC | Dateline+VC | Epoch+VC | HW deadlock-free |

---

## 8.5.7 BH Performance Ceiling for GSRP

Based on Section 8.1 measurements and 2D routing advantages:

```
BH Galaxy (32-chip) GSRP Performance Envelope:

                          Writes        Reads
Peak BW (adjacent):      ~12.8 B/c     ~6.4 B/c
Peak BW (5-hop 2D):      ~8.0 B/c      ~4.0 B/c
Peak BW (10-hop 2D):     ~5.5 B/c      ~2.8 B/c
GSRP BW (adjacent):     ~12.5 B/c     ~6.3 B/c  (-2.3%)
GSRP BW (5-hop 2D):      ~7.8 B/c     ~3.9 B/c  (-2.5%)
GSRP BW (10-hop 2D):     ~5.4 B/c     ~2.7 B/c  (-1.8%)

Aggregate L1 pool: 140 cores x 32 chips x 1,536 KB = ~6.7 GB
GSRP-usable pool (after overhead): ~6.4 GB
```

### BH vs. WH Performance Comparison for GSRP

| Metric | WH (T3K, 8-chip) | BH (Galaxy, 32-chip) | BH Advantage |
|--------|:-----------------:|:--------------------:|:------------:|
| Peak adjacent BW | 10.8 B/c | 12.8 B/c | +19% |
| Worst-case BW | 5.2 B/c (7 hop) | 5.5 B/c (10 hop) | +6% |
| ERISC outstanding reads | 1 | 4 | 4x |
| Aggregate L1 pool | ~914 MB | ~6.7 GB | 7.3x |
| Max 2D routing benefit | N/A (1D only) | 33--87% fewer hops | Significant |

---

## 8.5.8 BH-Specific GSRP Deployment Configuration

### Recommended Phase 1 Configuration (Software-Only)

```
BH Galaxy GSRP Phase 1 Configuration:

Fabric mode:        FABRIC_2D (or FABRIC_2D_TORUS_X for ring semantics)
Routing planes:     2 (BH default)
  - Plane 0: GSRP persistent connections
  - Plane 1: Explicit CCL operations
MUX cores:          4 per chip (dedicating 4 of 140 Tensix = 2.9%)
Connection pool:    8 entries per Tensix core
Route cache:        32 entries (covers all chips in Galaxy)
GSRP export region: Configurable per workload (default: top 256 KB of L1)
```

### Coordinate Virtualization for Global Addressing

BH's coordinate virtualization simplifies the GSRP address space:

```
BH Virtual Coordinate Space:

Physical chip coordinates vary by board wiring.
Virtual coordinates are remapped to a canonical grid:

  Virtual Tensix grid starts at (1, 2)
  Consistent across all BH chips

Global address = {mesh_id, chip_id, virtual_x, virtual_y, l1_offset}

The virtual coordinate system ensures that the same GlobalAddress
encoding works regardless of physical board topology.
```

---

## 8.5.9 System-Level Resource Accounting (BH)

### Per-Core L1 Budget

Per-core L1 overhead: existing fabric 6,864 B + GSRP Phase 2 5,288 B = **12,152 B**
(0.77% of 1,536 KB), leaving ~1,524 KB available. See Chapter 7, Section 4.4 for
the itemized breakdown.

### 32-Chip Galaxy System Summary

| Resource | Per-Chip | 32-Chip Galaxy |
|:---------|:--------:|:--------------:|
| Tensix L1 for GSRP (Phase 2) | 5,288 B x 140 = 723 KB | 22.6 MB |
| ERISC L1 for GSRP handler | 20.2 KB x 14 = 283 KB | 8.8 MB |
| MUX cores (4 per chip) | 4 cores | 128 cores |
| **Total memory overhead** | ~1,006 KB | ~31.4 MB |
| **Memory overhead fraction** | 0.47% | 0.47% |
| Initialization time | -- | ~1.8 ms |

---

## Key Takeaways

- **BH closes every WH gap:** larger ERISC L1 (512 KB vs. 256 KB), dual RISC-V (dedicated
  GSRP handler), native 2D routing, `NOC_SEC_FENCE_RANGE`, and +19% per-link bandwidth.

- **BH ERISC can support 4 outstanding reads** within the L1 budget (16 KB staging + 4.2 KB
  metadata = 20.2 KB, 5.6% of available), compared to WH's practical limit of 1.

- **Quasar's `noc_address_translation_table` is the Phase 3 enabler** -- this feature is
  available on Quasar (`tt-2xx`) only, NOT on Blackhole. It provides 16 mask entries,
  1024 endpoint entries, and 32 routing entries for hardware-native global address
  resolution, replacing the entire software injection shim.

- **BH 2D routing achieves 33--87% hop reduction** for column-traversing paths, making
  Galaxy-scale GSRP practical at 5.5 B/c worst-case (10-hop diagonal).

- **System-wide GSRP overhead is 0.47%** of total L1 for a 32-chip Galaxy (~31.4 MB out
  of ~6.7 GB), confirming that the <16 KB per-core target from Chapter 6, Section 1 is
  met with significant margin.

## GSRP Implications

BH is the optimal target for GSRP adoption. The dual ERISC enables dedicated GSRP
processing without impacting router throughput. The 2D routing reduces worst-case latency
by 33--87% compared to WH 1D ring. The `NOC_SEC_FENCE_RANGE` provides a potential
hardware-assist path for Phase 2.5, while Quasar's full address translation tables enable
complete hardware GSRP in Phase 3. The aggregate L1 pool of ~6.7 GB (32-chip Galaxy) makes
the "Global SRAM Pool" name meaningful -- this is a substantial memory resource worth
exposing as a unified address space. BH should be the primary platform for Phase 1
production deployment, with WH T3K serving as the validation platform.

## Source Code References

| File | Key Symbols |
|------|-------------|
| `tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h` | `MEM_L1_SIZE` (1536 KB), `MEM_MAP_END`, `MEM_TENSIX_ROUTING_TABLE_BASE` |
| `tt_metal/hw/inc/internal/tt-1xx/blackhole/eth_l1_address_map.h` | ERISC L1 layout (512 KB), `ERISC_L1_UNRESERVED_BASE` |
| `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h` | `NOC_SEC_FENCE_RANGE(cnt)`, 32 range entries |
| `tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_address_translation_tables.hpp` | Full HW address translation API (Quasar only) |
| `tt_metal/hw/inc/internal/tt-2xx/quasar/noc/registers/noc_address_translation_table_a_reg.h` | Register definitions for Quasar address translation hardware |
| `tt_metal/api/tt-metalium/experimental/fabric/control_plane.hpp` | `FabricConfig::FABRIC_2D`, `FABRIC_2D_TORUS_*` variants |
| `tt_metal/fabric/fabric_edm_packet_header.hpp` | `HybridMeshPacketHeaderT`, 4-bit per-hop 2D encoding |

---

**Previous:** [`04_wormhole_gap_analysis.md`](./04_wormhole_gap_analysis.md) | **Next:** [`06_workload_suitability.md`](./06_workload_suitability.md)
