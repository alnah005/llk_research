# 6.3 -- Security and Operational Considerations

The export/connect protocol transfers raw physical addresses and PCIe BAR offsets through `/dev/shm` flatbuffer files. Any process that can read those files can map device memory and issue PCIe transactions. This section covers the security implications, vIOMMU in cross-process operation, tmpfs lifecycle hazards, multi-host limitations, and debugging techniques.

---

## 6.3.1 The /dev/shm Security Surface

When `export_descriptor()` writes a flatbuffer to `/dev/shm/<socket_id>`, the file contains physical base addresses of device L1 FIFO regions, host-pinned memory addresses, PCIe BAR offsets, TLB window parameters, and flow-control counter locations. **Any process with read access can reconstruct a socket handle and perform raw PCIe transactions against the device.**

### Threat Model

| Threat | Impact |
|--------|--------|
| Unauthorized data injection | Attacker writes malicious data to device FIFO; model processes attacker-controlled input |
| Data exfiltration | Attacker reads D2H output from host-pinned memory |
| Descriptor tampering | Connector maps wrong TLB/memory; writes land in arbitrary device L1 |
| Denial of service | Corrupted FIFO pointers cause device kernel hang |

The protocol provides no authentication, authorization, encryption, integrity checking, replay protection, or audit logging.

### Hardening Recommendations

```
  Single-tenant deployment?
  |
  +-- YES --> umask 0077 before export; chmod 0600 on descriptor.
  |
  +-- NO (multi-tenant) --> Do NOT use export/connect across trust boundaries.
                            Isolate tenants via containers or MAC policies.
```

**Restrict UMD device access** (primary gate): Without `/dev/tenstorrent/*` access, possessing the descriptor is insufficient.

```
# /etc/udev/rules.d/99-tenstorrent.rules
SUBSYSTEM=="tenstorrent", GROUP="tt_users", MODE="0660"
```

**Container isolation:** Mount a private tmpfs for `/dev/shm` per container group so descriptors are invisible across isolation boundaries.

> **What breaks:** In Kubernetes, the default `/dev/shm` is shared across all containers in a pod. If two workloads export descriptors with colliding `socket_id` names, one overwrites the other. Use unique, workload-prefixed socket IDs.

**Early descriptor cleanup:** Once all connectors have attached, unlink the descriptor to reduce exposure:

```python
h2d.export_descriptor(socket_id)
ttnn.distributed_context_barrier()   # All connectors connected
os.unlink(f"/dev/shm/{socket_id}")   # Socket remains functional
```

---

## 6.3.2 vIOMMU in Cross-Process Context

All H2D and D2H sockets require vIOMMU regardless of cross-process sharing. The cross-process case adds a nuance: the connector maps PCIe resources set up by the owner's vIOMMU context.

The vIOMMU translation table is per-device, not per-process. When the owner pins memory, the IOMMU mapping is registered. The connector accesses the same device through the same translations:

```
  Owner: pin memory --> IOMMU mapping: dev addr 0xABCD -> host phys 0x1234
  Export: flatbuffer contains dev addr 0xABCD
  Connector: connect() reads 0xABCD, device NOC uses same IOMMU translation
```

The mapping persists as long as the owner's pinned memory is active. If the owner unpins (during destruction), the IOMMU mapping is torn down and device transactions from the connector trigger an IOMMU fault.

> **What breaks:** The vIOMMU does not enforce per-process access control to device memory. Any process that can open `/dev/tenstorrent/<N>` can reach any L1 address on that device. vIOMMU protects host memory from device DMA but does not prevent two host processes from writing to the same device L1 through different TLB windows.

### IOMMU Fault Symptoms

```bash
# Typical dmesg output when IOMMU mapping is invalidated:
[12345.678] AMD-Vi: Event logged [IO_PAGE_FAULT device=65:00.0 ...]
[12345.678] DMAR: DRHD: handling fault status reg 3
```

This typically causes a device hang -- the NOC engine stalls on the faulted transaction.

### Verification

```bash
dmesg | grep -i iommu                       # Should show IOMMU enabled
ls /sys/kernel/iommu_groups/*/devices/       # Should list Tenstorrent PCI IDs
dmesg | grep -i "dmar.*fault"               # Should be empty (no faults)
```

---

## 6.3.3 tmpfs Lifecycle Hazards

| Event | Effect on `/dev/shm` files |
|-------|---------------------------|
| Owner exits cleanly | Destructor unlinks the file |
| Owner receives SIGTERM | Destructor runs; file is unlinked |
| Owner receives SIGKILL / OOM | **Destructor does NOT run; file persists** |
| System reboot | All `/dev/shm` contents cleared |

### The Stale Descriptor Problem

The most dangerous hazard: if the owner receives SIGKILL, the destructor does not run, the file persists, but device memory is freed. A connector calling `connect()` succeeds but operates on freed device memory -- undefined behavior that can corrupt other sockets.

> **What breaks:** Orphaned descriptors accumulate across crashes. A new owner may export with the same `socket_id`, but a stale connector from a previous run reads the new descriptor and connects to the wrong owner's FIFO -- an unintended cross-generation connection.

### Mitigation Strategies

| Strategy | Mechanism |
|----------|-----------|
| Pre-start cleanup | `rm -f /dev/shm/<known_ids>` before launch |
| PID-embedded IDs | `<name>_<pid>`; connector checks PID liveness |
| Freshness validation | Reject descriptors with mtime older than startup |
| File locking | Owner holds `flock()`; connector checks lock |
| Namespace cleanup | Clean entire `<service>/<instance>/` prefix on restart |

```bash
# Diagnostic commands
ls -la /dev/shm/                   # List descriptors
fuser /dev/shm/my_socket_id       # Empty = stale
rm -f /dev/shm/pipeline_*         # Pattern cleanup
```

---

## 6.3.4 Multi-Host Limitations

The export/connect protocol is fundamentally a **single-host mechanism**. `/dev/shm` is local tmpfs and PCIe BAR mappings are local.

| Scenario | Mechanism |
|----------|-----------|
| Same host, same device | export/connect via `/dev/shm` |
| Same host, different devices | Separate socket instances per device |
| Different hosts, D2D | DistributedContext + MeshSocket + ranks over TT-Fabric |
| Different hosts, H2D/D2H | Not supported. Network data to target host, then local socket. |

### Multi-Host Architecture (DeepSeek V3)

Each host creates local H2D/D2H sockets. Cross-host data movement uses D2D MeshSockets. The DistributedContext (`tt::distributed::open_distributed_context()`) auto-initializes on device open:

```python
mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 4))
assert ttnn.distributed_context_is_initialized()
rank = int(ttnn.distributed_context_get_rank())
```

```
  Host A                                Host B
 +-------------------------+          +-------------------------+
 | H2D/D2H (local)         |          | H2D/D2H (local)         |
 | MeshSocket (D2D) -------+-- Fabric -->  MeshSocket (D2D)     |
 +-------------------------+          +-------------------------+
         +-- DistributedContext (ranks + barriers) --+
```

> **What breaks:** If a process opens a mesh device and then forks, the child inherits distributed context state but not the communication channels. Inter-host operations deadlock because the transport was not re-initialized in the child.

---

## 6.3.5 Debugging Cross-Process Socket Issues

### Diagnostic Flowchart

```
  connect() times out?
    +-> ls /dev/shm/<socket_id>
    |     Missing? -> Owner has not exported yet
    |     Present? -> Check permissions, check file size > 0
  connect() succeeds but data is wrong?
    +-> stat /dev/shm/<id> -- old mtime? -> Stale descriptor
    +-> Multiple processes calling write()? -> FIFO corruption
    +-> Check page size match between owner and connector
```

### Common Errors

| Symptom | Likely Cause | Investigation |
|---------|-------------|---------------|
| `connect()` timeout | Owner not exported, wrong socket_id | `ls /dev/shm/<id>` |
| Segfault in `write()` | Stale descriptor; owner dead | `kill -0 <owner_pid>` |
| Device kernel hang | Wrong page size or concurrent writers | Verify single-writer; compare page sizes |
| `barrier()` hangs on owner | Connector crashed without acking | `discard_pending_pages()` |
| Permission denied | File/device permissions | `ls -la /dev/shm/<id>; ls -la /dev/tenstorrent*` |
| DMAR faults in dmesg | vIOMMU misconfigured or owner destroyed | `dmesg \| grep -i dmar` |

### Flow Control Counter Inspection

```
  bytes_sent == bytes_acked        FIFO drained, no data in flight
  bytes_sent >> bytes_acked        Consumer stuck (not reading/acking)
  bytes_sent == FIFO capacity      Producer stuck (FIFO full, no acks)
```

For H2D: both `bytes_sent` and `bytes_acked` reside in device L1. Host writes `bytes_sent` via TLB write. Device writes `bytes_acked` after consuming pages; in HOST_PUSH mode, device also mirrors `bytes_acked` to host-pinned memory via NOC write so the host can poll locally.

For D2H: `bytes_sent` is written by device to host-pinned memory via NOC write (host reads locally). `bytes_acked` is written by host to device L1 via TLB write (device reads via cache invalidation).

---

## Key Takeaways

- `/dev/shm` descriptors contain raw physical addresses; restrict access via umask 0077, udev rules on `/dev/tenstorrent/*`, and container isolation.
- vIOMMU is required for all H2D/D2H modes; it provides address translation but not per-process device memory isolation.
- Stale descriptors from crashed owners are the most dangerous operational hazard; clean `/dev/shm` at deployment startup.
- Export/connect is single-host only; multi-host uses DistributedContext with rank-based MeshSocket addressing over TT-Fabric.
- DistributedContext auto-initializes on device open but is not fork-safe.
- Debug flow-control stalls by inspecting `bytes_sent` vs `bytes_acked` and checking process liveness.

---

**Previous:** [6.2 Owner and Connector Lifecycle](./02_owner_connector_lifecycle.md) | **Next:** [Chapter 7](../ch07/index.md) | **Up:** [Chapter Index](./index.md)
