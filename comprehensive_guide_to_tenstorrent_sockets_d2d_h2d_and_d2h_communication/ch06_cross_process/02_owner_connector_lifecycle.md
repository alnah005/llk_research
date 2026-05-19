# 6.2 -- Owner and Connector Lifecycle

Every cross-process socket has exactly one owner and typically one connector. Multiple connectors can call `connect()` on the same descriptor, but only one should be active at a time (see Invariant 3 and Failure 5 below). The owner creates the socket, allocates device resources, exports the descriptor, and controls teardown. The connector attaches via `connect()`, operates on the shared FIFO, and must detach before the owner tears down. These roles are permanent -- assigned at construction time and never transferable.

---

## 6.2.1 Role Definitions

| Responsibility | Owner | Connector |
|---------------|-------|-----------|
| Create socket (allocate FIFO, pin memory) | Yes | No |
| `set_page_size()` | Yes (before export) | No (inherited) |
| `export_descriptor()` | Yes | No |
| `write()` (H2D) / `read()` (D2H) | Yes | Yes |
| `barrier()` | Yes | Yes |
| `discard_pending_pages()` (D2H) | Yes | Yes |
| Deallocate device memory, unpin host memory | Yes | No |
| Unlink `/dev/shm` file on destroy | Yes | No |
| Requires MetalContext | Yes | No (UMD only) |

---

## 6.2.2 What the Connector Cannot Do

| Forbidden Operation | Why |
|---------------------|-----|
| Allocate device FIFO | Requires MetalContext device allocator |
| Change FIFO size | Requires deallocation and reallocation of device L1 |
| Change page size | Baked into the flatbuffer at export time |
| Reconfigure TLB mappings | Requires MetalContext TLB manager |
| Dispatch device kernels | Requires MetalContext command queue |
| Export the socket | Only the original owner can serialize state |
| Survive owner destruction | TLB mappings and pinned memory become invalid |

> **What breaks:** If a connector attempts `set_page_size()`, the call either silently no-ops or raises an error. The connector operates with the page size baked into the flatbuffer. Code that assumes reconfiguration after connecting will produce misaligned data.

### Decision Tree: Can My Process Do This Operation?

```
  Is this process the owner (created the socket with constructor)?
  |
  +-- YES --> Full access to all socket operations.
  |
  +-- NO (connector via connect()) -->
        +-- write()/read()/barrier()/discard_pending_pages()? --> Allowed.
        +-- set_page_size()?     --> NO. Fixed at export time.
        +-- Requires MetalContext? --> NO. Connector cannot perform it.
```

---

## 6.2.3 Three Lifecycle Invariants

Every failure mode in Section 6.2.5 traces back to violating one of these.

**Invariant 1: Owner creates and exports before connector connects.** The connector's `connect()` polls for the `/dev/shm` file and will time out if the owner has not yet exported.

**Invariant 2: Connector finishes before owner destroys.** If the owner destroys first, the physical addresses the connector uses become invalid -- device L1 is deallocated, host memory is unpinned.

**Invariant 3: At most one active writer (or reader) per socket.** The FIFO has a single write pointer (`bytes_sent`) and no locking. Concurrent writes corrupt the pointer. For D2H, concurrent readers advancing `bytes_acked` cause non-monotonic acknowledgement values.

---

## 6.2.4 Complete Lifecycle Sequence

```
  TIME  OWNER PROCESS                    CONNECTOR PROCESS
  ----  -------------                    -----------------
   t0   h2d = H2DSocket(mesh_device,
            recv_core, buffer_type,
            fifo_size, h2d_mode)
   t1   h2d.set_page_size(4096)
   t2   h2d.export_descriptor("tok_in")
        --> writes /dev/shm/tok_in
   t3   (signals connector)              (waits for signal/timeout)
   t4                                    h2d_r = H2DSocket.connect(
                                             "tok_in", timeout_ms=5000)
   t5   (lifecycle only; no writes)      h2d_r.write(data, n_pages)
   t6                                    h2d_r.barrier()
                                         del h2d_r --> unmaps shared memory
   t7   h2d.barrier()
        del h2d --> unpin, unlink, dealloc
```

| Constraint | Violation consequence |
|------------|----------------------|
| Owner must create before export | Cannot export non-existent socket |
| Owner must set_page_size before export | Connector inherits page_size=0; writes fail |
| Owner must export before connector connects | connect() times out |
| Connector must destroy before owner | Undefined behavior from freed memory |
| Both must call barrier() before destroy if data in flight | Device kernel hang |

---

## 6.2.5 Failure Mode Catalog

### Failure 1: Owner Dies Before Connector

**Trigger:** Owner crashes (SIGKILL, OOM) while connector is active.
**Symptom:** Connector writes to deallocated L1; segfault, silent corruption, or device hang.
**Root cause:** Owner deallocation frees L1 FIFO; connector's PCIe writes target freed memory that may be reallocated for another socket.
**Resolution:** Process supervision; connector detects owner death via PID monitoring and exits.

> **What breaks:** If the L1 has been reallocated, the connector writes to memory belonging to a different socket. No error is raised -- silent data corruption of unrelated data.

### Failure 2: Connector Dies Before Completing (D2H)

**Trigger:** D2H connector crashes mid-read without acknowledging consumed pages.
**Symptom:** Owner's `barrier()` spins indefinitely waiting for `bytes_acked` to catch up.
**Root cause:** Connector crashed without writing `bytes_acked` to device L1 via TLB write.
**Resolution:** Use `discard_pending_pages()` to rebase `bytes_acked` and unblock the device kernel. In PipelineBlock, HostInterface catches process-exit signals for this purpose.

### Failure 3: Racing Writers or Readers

**Trigger:** Both owner and connector call `write()` (H2D) or `read()` (D2H) concurrently.
**Symptom:** Counter corruption; device kernel enters undefined behavior.
**Root cause:** Flow-control counters updated non-atomically by two processes.
**Resolution:** Designate a single writer (H2D) or single reader (D2H). No built-in locking exists.

### Failure 4: Stale Descriptor File

**Trigger:** Owner crashes without running destructor; descriptor persists in `/dev/shm`.
**Symptom:** Connector connects successfully but operates on freed device memory.
**Resolution:** Clean `/dev/shm` at deployment startup; embed PIDs or timestamps in socket IDs.

### Failure 5: Multiple Connectors Without Coordination

**Trigger:** Two connector processes both call `connect()` on the same `socket_id`.
**Symptom:** H2D: concurrent writers (Failure 3). D2H: each connector sees alternating pages.
**Resolution:** Enforce one connector per socket via naming conventions or IPC.

---

## 6.2.6 Production Pattern: Three-Barrier Shutdown

The DeepSeek V3 PipelineBlock uses distributed context barriers to enforce lifecycle ordering:

```python
rank = int(ttnn.distributed_context_get_rank())

# Phase 1-2: Create and configure (owners only)
if rank == 0:
    h2d = ttnn.H2DSocket(mesh_device, recv_core, BufferType.L1, fifo_size, H2DMode.HOST_PUSH)
    h2d.set_page_size(page_size)
ttnn.distributed_context_barrier()   # All sockets created

# Phase 3: Export/connect
if rank == 0:
    h2d.export_descriptor("stage0_h2d")
ttnn.distributed_context_barrier()   # All descriptors published
if rank == 1:
    h2d_remote = ttnn.H2DSocket.connect("stage0_h2d", timeout_ms=5000)
ttnn.distributed_context_barrier()   # All connections established

# Phase 4: Active data transfer (connector writes, owner manages lifecycle)

# Phase 5: Three-barrier shutdown
ttnn.distributed_context_barrier()   # Barrier 1: All transfers complete
if rank == 1:
    del h2d_remote                   # Connector destructs first
ttnn.distributed_context_barrier()   # Barrier 2: All connectors gone
if rank == 0:
    del h2d                          # Owner destructs last
ttnn.distributed_context_barrier()   # Barrier 3: All sockets cleaned up
ttnn.close_mesh_device(mesh_device)
```

The three-barrier shutdown guarantees connector-before-owner ordering even when processes run at different speeds. The owner never calls `write()` -- it only manages lifecycle. The connector handles all data-plane operations, eliminating concurrency hazards.

For independent processes (not SPMD), rely on `timeout_ms` and application-level IPC or file-based readiness signals:

```python
# Pattern: File-based readiness for multi-socket setups
# Owner:
for name, socket in sockets.items():
    socket.export_descriptor(name)
Path("/dev/shm/all_sockets_ready").touch()

# Connector:
while not Path("/dev/shm/all_sockets_ready").exists():
    time.sleep(0.1)
for name in expected_sockets:
    handles[name] = H2DSocket.connect(name)
```

---

## 6.2.7 Lifecycle Comparison Table

| Phase | Owner (H2D/D2H) | Connector (H2D/D2H) |
|-------|-----------------|---------------------|
| Create | Constructor (MetalContext, MeshDevice) | `connect()` (UMD only, no MeshDevice) |
| Configure | `set_page_size()` | Inherited from flatbuffer |
| Share | `export_descriptor()` | N/A |
| Data transfer | `write()`/`read()` | `write()` via PCIeCoreWriter (H2D) or `read()` via mmap (D2H) |
| Sync | `barrier()` | `barrier()` |
| Destroy | Wait for ack, unpin memory, release TLB, unlink `/dev/shm`, dealloc FIFO | Wait for ack, unmap shared memory (no device deallocation) |

---

## Key Takeaways

- Owner holds MetalContext and controls all device resources; connector operates through UMD-mapped PCIe only.
- Roles are permanent and non-transferable -- assigned at construction time for the socket's lifetime.
- The connector cannot change page size, resize FIFOs, reconfigure TLBs, or dispatch kernels.
- FIFOs are single-producer, single-consumer with no locking; concurrent access corrupts flow-control counters.
- Three invariants govern correctness: export before connect, connector finishes before owner destroys, one active writer per socket.
- Five failure modes exist (owner death, connector death, racing writers, stale descriptors, multiple connectors); most stem from ordering violations.
- The three-barrier shutdown pattern (DeepSeek V3 PipelineBlock) guarantees correct teardown across processes running at different speeds.

---

**Previous:** [6.1 Export/Connect Protocol](./01_export_connect_protocol.md) | **Next:** [6.3 Security and Operational Considerations](./03_security_and_operational_considerations.md) | **Up:** [Chapter Index](./index.md)
