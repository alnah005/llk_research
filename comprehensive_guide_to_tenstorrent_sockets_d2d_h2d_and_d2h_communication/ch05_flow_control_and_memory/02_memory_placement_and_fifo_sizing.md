# 5.2 Memory Placement and FIFO Sizing

The `SocketMemoryConfig` struct controls where a socket's data buffer lives and how large it is. These two decisions -- storage type and FIFO size -- have the largest impact on socket throughput and on the resources available for compute. This section provides decision frameworks for both, along with alignment requirements, sub-device binding, and socket reuse patterns.

---

## 5.2.1 The SocketMemoryConfig Fields

Every D2D `MeshSocket` is configured with a `SocketMemoryConfig`:

```cpp
struct SocketMemoryConfig {
    BufferType socket_storage_type = BufferType::L1;  // L1 or DRAM
    uint32_t   fifo_size;                              // FIFO capacity in bytes
    optional<SubDeviceId> sender_sub_device;           // bind sender to sub-device
    optional<SubDeviceId> receiver_sub_device;         // bind receiver to sub-device
};
```

H2D and D2H sockets specify `buffer_type` and `fifo_size` directly in their constructors, but the same trade-offs apply. The `socket_storage_type` controls where the data FIFO is allocated. The config buffer (counters, metadata) always resides in L1 regardless.

---

## 5.2.2 L1 vs DRAM: The Storage Decision Framework

| Property | L1 SRAM | DRAM |
|----------|---------|------|
| Capacity per core | ~1464 KB (Blackhole Galaxy) | 12 GB shared across device |
| Access latency | Single-cycle load/store (core-local) | Higher, potential bank contention |
| Contention | None (private per core) | Bank conflicts with other cores |
| Impact on compute | Reduces L1 for circular buffers | No L1 impact |

Each Tensix core's ~1464 KB L1 is shared between compute circular buffers, socket FIFOs, and config buffers. Every byte allocated to a socket FIFO is unavailable for compute.

### Decision Tree: "Should I Use L1 or DRAM?"

```
Is the FIFO small (< 64 KB)?
  |
  +-- Yes --> USE L1  (lowest latency, no contention)
  |
  +-- No (>= 64 KB)
        |
        Does the core also run compute with large CBs?
          +-- Yes --> USE DRAM  (preserve L1 for compute)
          +-- No (dedicated I/O core)
                |
                Will the FIFO exceed ~256 KB?
                  +-- Yes --> USE DRAM
                  +-- No  --> L1 is viable
```

### "What Breaks" -- Wrong Buffer Type

**DRAM on a latency-sensitive path:** Under bank contention, individual pages incur variable latency. The host sees jitter on D2H reads. Switch to L1 to eliminate contention.

**L1 on a memory-constrained core:** A 64 KB FIFO on a core running transformer kernels may fail at allocation time -- the allocator cannot fit both FIFO and compute CBs within 1464 KB. Fix: move to DRAM, reduce FIFO size, or use a dedicated I/O core.

**Warning:** There is no runtime error when L1 is oversubscribed. Allocation fails at socket creation or at compute kernel dispatch. Plan L1 budgets before deploying.

---

## 5.2.3 FIFO Sizing Methodology

The FIFO must cover the bandwidth-delay product of the transfer path so the producer does not stall waiting for acknowledgements.

### The Sizing Formula

```
fifo_size = page_size * pipeline_depth
pipeline_depth >= ceil(round_trip_latency / transfer_time_per_page) + 1
```

### Step 1: Choose the Page Size

| Data Pattern | Recommended Page Size |
|-------------|----------------------|
| Single tile row | Row size (e.g., 64 B for BF16 x 32) |
| Full tile (32x32 BF16) | 2048 B |
| Tensor slice | Tens of KB |
| Token embedding | Model-dependent (4 KB - 16 KB) |

Constraints: minimum 64 bytes (NOC word width), multiple of 16 bytes (SRAM alignment). For HOST_PUSH, ideally a multiple of PCIe TLB page size (4 KB or 16 KB).

### Step 2: Determine Pipeline Depth

```
Transfer path                         Recommended depth
D2D intra-mesh (few hops)             2-4 pages
D2D inter-mesh (MGD gateway)          4-8 pages
H2D/D2H on high-BW port (Gen4 x8)    4-8 pages   (~16 GB/s)
H2D/D2H on low-BW port  (Gen4 x1)    8-16 pages  (~2 GB/s)
Latency-sensitive loopback            2 pages (double-buffer)
```

**PCIe bandwidth asymmetry:** Blackhole has 4 high-bandwidth Gen4 x8 ports (ASIC 6, ~16 GB/s each) and 28 low-bandwidth Gen4 x1 ports (~2 GB/s each). Low-bandwidth ports need deeper FIFOs.

For throughput-oriented paths, apply a 2x-4x multiplier: `fifo_size = (2 to 4) * page_size * ceil(round_trip / transfer_time_per_page)`.

### Validation Checks

```
fifo_size >= 2 * page_size     (double-buffer minimum)
fifo_size % page_size == 0     (exact multiple -- no partial pages)
fifo_size <= available memory  (L1 budget or DRAM)
page_size >= 64                (NOC minimum)
page_size % 16 == 0            (SRAM alignment)
```

### "What Breaks" -- Sizing Errors

| Mistake | Symptom | Fix |
|---------|---------|-----|
| `fifo_size == page_size` | No overlap; lock-step throughput | Use >= 2 pages |
| `fifo_size < page_size` | Cannot fit one page; creation fails | `fifo_size >= 2 * page_size` |
| `page_size < 64` bytes | Silent corruption; NOC reads beyond boundary | Use 64 B minimum |
| FIFO too large for L1 | Compute CB allocation fails at dispatch | Switch to DRAM or reduce |
| FIFO too small for D2D | Sender idles between bursts; throughput cliff | Increase to cover round-trip |

---

## 5.2.4 Sub-Device Binding

The `sender_sub_device` and `receiver_sub_device` fields optionally bind socket endpoints to sub-device scheduling domains.

```
Does your pipeline reload sub-devices during execution?
  |
  +-- No --> Binding optional. Unbound sockets persist in global pool.
  |
  +-- Yes
        +-- Socket should persist across reloads? --> DO NOT BIND.
        +-- Socket lifetime matches sub-device?   --> BIND.
```

**DeepSeek V3 pattern:** Sockets created once during initialization, reused across all iterations without sub-device binding ([Chapter 7](../ch07_production_usage/index.md)).

**"What breaks":** A bound socket in use when its sub-device unloads has its backing memory freed. Subsequent access causes stale data, corruption, or hard fault.

---

## 5.2.5 Alignment Requirements

Three alignment constraints apply. Violations produce incorrect behavior that may not trigger runtime errors.

| Constraint | Minimum | Violation effect |
|-----------|---------|-----------------|
| Page size | 64 B (NOC word width) | Silent corruption; NOC reads/writes beyond boundary |
| SRAM alignment | 16 B | Hard fault or silent partial write |
| PCIe TLB | 4 KB / 16 KB | Straddled TLB entries; doubled per-page latency (HOST_PUSH) |
| FIFO divisibility | `fifo_size % page_size == 0` | Pages wrap mid-buffer causing corruption |

### Alignment Decision Tree

```
D2D?        --> page_size >= 64 B, multiple of 16 B
HOST_PUSH?  --> Above PLUS page_size multiple of 4 KB or 16 KB (TLB)
DEVICE_PULL/D2H? --> Above PLUS pinned buffer aligned to 4 KB
```

**vIOMMU requirement:** All H2D and D2H modes require vIOMMU on the host. Even HOST_PUSH needs it for the device's NOC write of `bytes_acked` to host pinned memory. Without vIOMMU, host-device sockets fail at creation.

**Debugging tip:** Intermittent, position-dependent corruption (only certain pages, only under sustained load) is the signature of alignment violation. Check page_size, fifo_size divisibility, and FIFO base address alignment first.

---

## 5.2.6 Socket Reuse and Recreation

Sockets are designed to be created once and reused across pipeline iterations via the socket pool pattern.

### Why Reuse Matters

| Operation | Saved by reuse |
|-----------|----------------|
| Device-side buffer allocation (L1 or DRAM) | Yes |
| Host memory pinning (H2D/D2H) | Yes |
| TT-Fabric route establishment (D2D) | Yes |
| Descriptor export (cross-process) | Yes |
| Config buffer initialization | Yes |

### The Socket Pool Pattern

```python
# Initialization (once)
h2d = ttnn.H2DSocket(mesh_device, core, BufferType.L1, fifo_size,
                     H2DMode.HOST_PUSH)
d2h = ttnn.D2HSocket(mesh_device, core, fifo_size)

# Iteration loop (many times)
for batch in batches:
    h2d.write(batch.data, batch.num_pages)
    d2h.read(output.data, output.num_pages)
    h2d.barrier()
    d2h.barrier()
```

After `barrier()`, the FIFO is logically empty (`bytes_sent == bytes_acked`) and ready for reuse. Counters continue monotonically -- they are not reset.

### Decision Tree: "Can I Reuse This Socket?"

```
Has any of the following changed?
  +-- FIFO size      --> MUST RECREATE
  +-- Target core    --> MUST RECREATE
  +-- Storage type   --> MUST RECREATE
  +-- Device reset   --> MUST RECREATE
  +-- Page size only --> CAN REUSE (call set_page_size())
  +-- None of above  --> REUSE (barrier(), then proceed)
```

**Warm-up latency:** The first transfer may be slower (lazy TLB mapping, uncached fabric routes, unmapped pinned pages). In DeepSeek V3, this cost is absorbed on the first inference request. For latency-sensitive workloads, issue a dummy transfer after creation.

---

## 5.2.7 Configuration Walkthrough

**D2D tile transfer** (BF16 32x32 = 2048 B/tile, between meshes): At ~12.5 GB/s per link (~0.16 us/page) and ~2 us round-trip, depth = `ceil(2.0/0.16) = 13`. Rounding up to 28 pages (~2x safety plus headroom). `fifo_size = 57344 B` (~56 KB, 3.8% of L1).

```python
mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 57344)
```

**H2D token injection** (4 KB embeddings, HOST_PUSH, Gen4 x8): 8 pages for practical double-buffering with headroom. `fifo_size = 32768 B`. L1 for latency.

```python
h2d = ttnn.H2DSocket(mesh_device, MeshCoreCoord(...),
                     BufferType.L1, 32768, H2DMode.HOST_PUSH)
```

**Post-deployment validation:** Sender never blocks = FIFO may be oversized (reduce to save L1). Sender frequently blocks at `fifo_size` = increase FIFO or investigate consumer bottleneck.

---

## Key Takeaways

- Choose L1 for small, latency-sensitive FIFOs; DRAM for large, throughput-oriented FIFOs. The config buffer always lives in L1.
- Size the FIFO as `page_size * pipeline_depth` to cover the bandwidth-delay product. Use 2-4 pages for D2D, 4-8 for high-BW PCIe, 8-16 for low-BW PCIe. Apply 2x-4x safety for throughput paths.
- Three alignment levels: 64 B NOC minimum, 16 B SRAM alignment, 4/16 KB PCIe TLB. `fifo_size` must be an exact multiple of `page_size`. Violations cause silent corruption.
- All H2D and D2H modes require vIOMMU on the host system.
- Bind sockets to sub-devices only when lifetime should match. Leave unbound for persistent sockets.
- Reuse sockets across iterations via `barrier()`. The first transfer has warm-up cost; subsequent transfers run at steady state.

---

*Previous: [5.1 Unified Circular FIFO Model](01_unified_circular_fifo_model.md) | Next: [Chapter 6 -- Cross-Process Socket Sharing](../ch06_cross_process/index.md) | Up: [Chapter 5 Index](index.md)*
