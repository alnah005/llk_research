# 8.4 Wormhole Gap Analysis

## Context

Sections 8.1--8.3 established the quantitative performance envelope and the tradeoff
landscape between explicit CCL and GSRP. This section assesses what the Wormhole (WH)
architecture already supports versus what is missing for a GSRP deployment. The reader
should understand the WH Tensix L1 layout (Chapter 2, Section 1), the WH Ethernet core
specifications (Chapter 3, Section 1), and the 1D routing model with extension words
(Chapter 3, Section 3). WH is the current-generation Tenstorrent chip (`tt-1xx` family),
and any GSRP deployment that delivers near-term value must work on WH hardware -- firmware
and software changes only, no silicon modifications.

---

## 8.4.1 Wormhole Hardware Summary

| Parameter | Value | Source |
|-----------|:-----:|:------:|
| Tensix cores per chip | 80 | `dev_mem_map.h` |
| L1 SRAM per Tensix core | 1,464 KB | `MEM_L1_SIZE = 1464 * 1024` |
| Ethernet cores per chip | 16 | Chapter 3, Section 1 |
| ERISC processor | Single RISC-V | Chapter 3, Section 1 |
| ERISC L1 per core | 256 KB (minus 32 B barrier) | `eth_l1_address_map.h` |
| Usable ERISC L1 (with routing FW) | ~151 KB | Chapter 3, Section 1 |
| NOC | Dual (NOC0, NOC1) | Chapter 2, Section 3 |
| Routing planes | 4 | Chapter 3, Section 4 |
| Fabric topology support | 1D: Linear, Ring, Neighbor Exchange | `FabricConfig` enum |
| Max hops (1D) | 63 | Chapter 3, Section 5 |

### WH Ethernet Link Topology

```
WH 8-chip T3K Ring (16 Ethernet cores per chip, 4 links per direction):

  Chip 0 ---[4 links]--- Chip 1 ---[4 links]--- Chip 2 --- ... --- Chip 7
    |                                                                  |
    +---------------------------[4 links]------------------------------+
                            (ring closure)

  Each link: ~12.5 GB/s per direction
  Per-neighbor aggregate: ~50 GB/s per direction (4 links x 12.5 GB/s)
  Total bisection BW (8-chip ring): ~100 GB/s (2 cut edges x 50 GB/s)
```

---

## 8.4.2 What WH Already Supports

The following GSRP-required capabilities exist in WH hardware and firmware:

| Capability | WH Support | Implementation |
|-----------|:----------:|:---------------|
| Programmable Ethernet firmware | Yes | ERISC runs EDM router firmware |
| Source routing (per-hop fields) | Yes | `LowLatencyPacketHeader` 2-bit per-hop |
| Unicast write across chips | Yes | `fabric_unicast_noc_unicast_write()` |
| Multicast write across chips | Yes | `WRITE_AND_FORWARD` mode |
| Atomic increment across chips | Yes | `fabric_unicast_noc_unicast_atomic_inc()` |
| Flow control (credit-based) | Yes | Stream register-based free slot tracking |
| Routing tables in L1 | Yes | `MEM_TENSIX_ROUTING_TABLE_BASE` |
| Connection manager | Yes | `FabricConnectionManager` (Chapter 4, Section 2) |
| Speedy-path optimization | Yes | `super_speedy_mode` (~9 cycles saved) |
| Packet header pool | Yes | Pre-allocated in L1 |
| Watcher + DPrint debugging | Yes | Conditional compilation on ERISC/Tensix |
| Fabric telemetry | Yes | Bandwidth counters, router state |
| GlobalSemaphore | Yes | Cross-chip atomic synchronization |
| MUX mode (Tensix aggregation) | Yes | `FabricTensixConfig::MUX` |

> **Key Insight:** WH already provides the fundamental building blocks for the GSRP
> Fat Fabric API (Chapter 6, Section 4.1). The write path, atomic operations, flow control,
> and debugging infrastructure are production-ready. The gap is in higher-level abstractions
> and address translation, not in low-level mechanisms.

---

## 8.4.3 What WH Is Missing

### Gap 1: Hardware Address Translation

WH has no hardware mechanism to intercept a NOC write targeting an out-of-range address and
redirect it to the fabric. The NOC address space is chip-local: coordinates `(x, y)` + L1
offset always refer to a core on the same chip.

| Missing Mechanism | Impact | Mitigation |
|-------------------|--------|:----------:|
| NOC address translation (MMU/TLB) | Cannot transparently intercept remote writes | Software-only: explicit `gsrp_write()` API |
| `noc_address_translation_table` | Quasar (`tt-2xx`) only -- not available on WH | N/A for WH |
| `NOC_SEC_FENCE_RANGE` | BH only -- not available on WH | N/A for WH |

**WH mitigation:** The Fat Fabric API approach (Chapter 6, Section 4.1) avoids the need
for hardware address translation by making `gsrp_write()` an explicit call. This is Level B
transparency (runtime-assisted with explicit calls, implicit routing) rather than Level A
(fully hardware-transparent).

### Gap 2: Limited 2D Routing

WH's fabric configuration supports only 1D topologies:

| FabricConfig | WH Support | BH Support |
|:-------------|:----------:|:----------:|
| FABRIC_1D | Yes | Yes |
| FABRIC_1D_RING | Yes | Yes |
| FABRIC_1D_NEIGHBOR_EXCHANGE | Yes | Yes |
| FABRIC_2D | **No** | Yes |
| FABRIC_2D_TORUS_X | **No** | Yes |
| FABRIC_2D_TORUS_Y | **No** | Yes |
| FABRIC_2D_TORUS_XY | **No** | Yes |

For a T3K (2x4 mesh), WH treats the 8 chips as a 1D ring. 2D mesh routing must be
emulated by the host-side `ControlPlane` computing 1D paths.

```
WH T3K Routing: 1D Ring vs. Ideal 2D Mesh

1D Ring (WH actual):
  Chip (0,0) -> Chip (1,1): up to 3 hops through the ring
  Worst case: 7 hops (half ring of 8 chips)

Ideal 2D Mesh (not available on WH):
  Chip (0,0) -> Chip (1,1): 2 hops (one East + one North)
  Worst case: 4 hops in a 2x4 mesh (1 row + 3 cols)
```

### Gap 3: Single RISC-V per ERISC

WH Ethernet cores have a single RISC-V processor running the EDM router firmware. This
leaves no headroom for additional GSRP processing without impacting router throughput.

| Processing Task | Cycles Required | WH ERISC Headroom |
|:----------------|:---------------:|:-----------------:|
| EDM router main loop | ~40--80/packet | Primary task |
| GSRP address translation | ~15--30/packet | **Conflicts** with router |
| GSRP read request handling | ~50--100/request | **Conflicts** with router |
| GSRP export validation | ~5--10/packet | Marginal fit |

> **Recommendation:** On WH, the GSRP should rely on the **Fat Fabric API** (Candidate 1
> from Chapter 6) rather than the **Address-Translation Shim** (Candidate 2). The single
> RISC-V ERISC cannot afford the shim processing overhead without degrading router
> performance. Address translation must happen on the Tensix side.

### Gap 4: ERISC L1 Constraints

WH ERISC L1 is 256 KB total, with ~151 KB usable after routing firmware reservation. The
GSRP read handler (Chapter 7, Section 4.9) requires:

| GSRP ERISC Component | Size | WH ERISC Budget Impact |
|:---------------------|:----:|:----------------------:|
| Read handler firmware | ~2 KB | 1.3% of usable |
| Read staging buffer | 4,096 B | 2.7% of usable |
| GSRP telemetry | 32 B | Negligible |
| Export config cache (80 cores) | ~960 B | 0.6% of usable |
| **Total** | **~7.1 KB** | **4.7% of usable ERISC L1** |

This is feasible but leaves less margin than BH (which has 512 KB ERISC L1). Multiple
outstanding reads (requiring additional staging buffers) may not be practical on WH.

### Gap 5: No Cross-Chip Read Protocol

The `NOC_UNICAST_READ` send type exists in the `NocSendType` enum
(`tt_metal/fabric/fabric_edm_packet_header.hpp`), indicating hardware awareness of read
operations. However, there is no standard cross-chip read protocol:

- No firmware-level read request/response handling on ERISC.
- No response routing mechanism (the request packet must carry a return path).
- No read completion signaling to the requesting Tensix core.

**WH mitigation:** Phase 1 GSRP on WH should be **write-only**. Phase 2 can add read
support with firmware extensions from Chapter 7, Section 1.5.

### Gap 6: Connection State Persistence

Current WH fabric connections are per-operation: opened at program start, closed at program
end. The GSRP connection pool requires persistent connections that survive across multiple
program invocations. A firmware extension is needed to maintain connection state in ERISC L1
across program launches.

---

## 8.4.4 WH GSRP Capability Matrix

```
WH Capability Assessment for GSRP Phases:

  Capability                    Phase 1   Phase 2   Phase 3   Phase 4
  ---------------------------------------------------------------
  gsrp_write() (posted)         [READY]   [READY]   [READY]   [READY]
  gsrp_write_release()          [READY]   [READY]   [READY]   [READY]
  gsrp_atomic_inc()             [READY]   [READY]   [READY]   [READY]
  gsrp_read()                   [----]    [FW MOD]  [FW MOD]  [FW MOD]
  gsrp_epoch_barrier()          [READY]   [READY]   [READY]   [READY]
  Transparent injection (HW)    [----]    [----]    [----]    [NO HW]
  Address translation (HW)      [----]    [----]    [----]    [NO HW]
  2D mesh routing               [----]    [----]    [----]    [NO HW]
  Compiler-inserted comm        [----]    [----]    [SOFT]    [SOFT]
  Persistent connection pool    [FW MOD]  [FW MOD]  [FW MOD]  [FW MOD]

  Legend: [READY] = existing HW/FW supports it
          [FW MOD] = firmware modification needed
          [SOFT] = software-only implementation
          [NO HW] = WH hardware cannot support
          [----] = not targeted for this phase
```

---

## 8.4.5 Minimum Firmware Changes for WH

### Phase 1 Firmware Changes

Phase 1 requires ~8--10 engineering-weeks of firmware work (see Section 8.7.3 for the
detailed effort breakdown and timeline). The key WH-specific changes:

- Persistent connection pool state in ERISC L1 (new L1 layout)
- Export region validation in ERISC firmware
- Route table extension for GSRP addressing in host driver

Phase 1 delivers: `gsrp_write()`, `gsrp_write_release()`, `gsrp_atomic_inc()`,
`gsrp_epoch_barrier()`, and telemetry -- all with write-only semantics on WH hardware.

### Phase 2 Effort (~12--16 weeks)

| Change | Scope | Risk | Effort |
|:-------|:-----:|:----:|:------:|
| Read request handler on ERISC | ERISC firmware (major) | Moderate | 4--6 weeks |
| Read response routing | ERISC firmware + packet format | Moderate | 3--4 weeks |
| Read completion signaling | Tensix API + ERISC | Low | 2 weeks |
| Dynamic route caching | Tensix firmware | Low | 2--3 weeks |
| **Total Phase 2** | | **Moderate overall** | **~12--16 weeks** |

---

## 8.4.6 WH System-Level Impact

### L1 Budget for 80-Core WH Chip

Per-core L1 overhead: existing fabric 6,864 B + GSRP Phase 2 5,288 B = ~12.2 KB (0.83%
of 1,464 KB). See Chapter 7, Section 4.4 for the itemized breakdown.

| Metric | Phase 1 | Phase 2 |
|:-------|:-------:|:-------:|
| GSRP per core | 1,016 B | 5,288 B |
| 80-core total | ~79 KB | ~413 KB |
| Available L1 per core | ~1,456 KB | ~1,452 KB |

### 8-Chip T3K System Summary

| Resource | Per-Chip | 8-Chip T3K |
|:---------|:--------:|:----------:|
| Tensix L1 for GSRP (Phase 2) | 5,288 B x 80 = 413 KB | 3.2 MB |
| ERISC L1 for GSRP handler | 7.1 KB x 16 = 114 KB | 907 KB |
| MUX cores (if 4 per chip) | 4 cores | 32 cores |
| **Total memory overhead** | ~527 KB | ~4.1 MB |
| **Memory overhead fraction** | 0.46% | 0.46% |

---

## Key Takeaways

- **WH provides all low-level primitives** needed for Phase 1 GSRP: unicast/multicast
  write, atomic increment, flow control, routing tables, debugging. The Fat Fabric API can
  be implemented with ~8--10 weeks of firmware work.

- **Three hardware limitations constrain WH GSRP:** (1) no hardware address translation
  (forces explicit `gsrp_write()` calls), (2) 1D-only routing (longer paths in T3K), and
  (3) single RISC-V per ERISC (no headroom for address-translation shim). These are
  architectural constraints, not bugs -- they define the ceiling for WH GSRP.

- **The single ERISC RISC-V is the binding constraint.** It precludes the
  Address-Translation Shim on WH and limits read handler complexity. Phase 2 read support
  requires careful firmware integration to avoid degrading router throughput.

- **WH GSRP should target Phase 1 (write-only) + Phase 2 (read with FW changes).**
  Phases 3 and 4 (hardware-assisted translation, compiler integration) cannot be fully
  realized on WH due to the lack of `noc_address_translation_table` (Quasar-only) and
  `NOC_SEC_FENCE_RANGE` (BH-only) hardware.

- **L1 impact is negligible:** GSRP Phase 2 adds ~5.3 KB per Tensix core (0.36% of WH L1),
  with system-wide overhead of ~4.1 MB for an 8-chip T3K (0.46% of total L1).

## GSRP Implications

WH is the right platform for GSRP Phase 1 validation. The Fat Fabric API's write-path-only
subset can be implemented and tested on existing T3K hardware, providing early feedback on
programmability gains and throughput overhead before investing in the more complex read path
and firmware extensions required for Phase 2. The 1D routing limitation means T3K GSRP
performance will be worst-case (up to 7 hops) compared to BH's 2D routing (max ~4 hops for
8 chips), but this is acceptable for validation purposes.

## Source Code References

| File | Relevance |
|------|-----------|
| `tt_metal/hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h` | `MEM_L1_SIZE = 1464 * 1024`, `MEM_MAP_END`, routing table bases |
| `tt_metal/hw/inc/internal/tt-1xx/wormhole/eth_l1_address_map.h` | ERISC L1 layout, `ERISC_L1_UNRESERVED_BASE`, 256 KB total |
| `tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc_parameters.h` | WH NOC registers (no `NOC_SEC_FENCE_RANGE`) |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp` | Speedy-path constraints (single sender channel) |
| `tt_metal/fabric/fabric_edm_packet_header.hpp` | `LowLatencyPacketHeader` (1D), `NocSendType::NOC_UNICAST_READ` |
| `tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp` | `RoutingPlaneConnectionManager` for up to 4 planes on WH |

---

**Previous:** [`03_explicit_vs_implicit_tradeoff_matrix.md`](./03_explicit_vs_implicit_tradeoff_matrix.md) | **Next:** [`05_blackhole_gap_analysis.md`](./05_blackhole_gap_analysis.md)
