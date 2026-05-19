# 7.3 -- End-to-End Inference Walkthrough

This section traces a DeepSeek V3 inference request from process initialization through socket setup, prefill, decode, and result extraction. Each phase opens with the failure that occurs when a step is skipped or misordered. A token tracing table shows exactly which socket type, FIFO location, and flow-control counters are active at each hop.

---

## 7.3.1 Reference Deployment: 4 Meshes, 2 Hosts

The walkthrough uses this concrete topology:

```
  Host 0 (rank 0)                    Host 1 (rank 1)
  +-------------+  +-------------+   +-------------+  +-------------+
  | Mesh 0      |  | Mesh 1      |   | Mesh 2      |  | Mesh 3      |
  | Stage: FIRST|->| Stage: MID  |-->| Stage: MID  |->| Stage: LAST |
  | Layers 0-14 |  | Layers 15-29|   | Layers 30-44|  |  +LOOPBACK  |
  +-------------+  +-------------+   +-------------+  | Layers 45-59|
        ^                                              +-------------+
        |                                                     |
        +------------------D2D loopback-----------------------+
        |                                                     |
      H2D (tokens in)                                  D2H (logits out)
```

### Stage Assignment Table

| Mesh | Host (rank) | Stage Type | Layers | H2D | D2H | Upstream D2D | Downstream D2D | Loopback D2D |
|------|-------------|------------|--------|-----|-----|--------------|----------------|--------------|
| 0 | 0 | First+D2D | 0-14 | Yes | No | No | Yes | Recv (from Mesh 3) |
| 1 | 0 | Middle | 15-29 | No | No | Yes | Yes | No |
| 2 | 1 | Middle | 30-44 | No | No | Yes | Yes | No |
| 3 | 1 | Last+Loopback | 45-59 | No | Yes | Yes | No | Send (to Mesh 0) |

---

## 7.3.2 Phase 1: Process Initialization

### What Breaks: Single-Process Launch

Launching only one host process for a multi-mesh Galaxy causes an indefinite hang when the first cross-mesh socket tries to connect to a non-existent peer.

### What Breaks: Barrier Omission After Device Open

The distributed context auto-initializes on `open_mesh_device()`, but if Host 0 proceeds to create PipelineBlocks before Host 1 finishes device open, the cross-mesh `MeshSocket` constructor gets stale peer information.

**Symptom**: Constructor hangs or throws "rank not found."

### Correct Initialization Sequence

```python
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 4))
# Distributed context auto-initialized -- no explicit init needed
my_rank = int(ttnn.distributed_context_get_rank())
ttnn.distributed_context_barrier()  # Both hosts must reach this point
```

### Initialization Phase Comparison

| Phase | What happens | Failure if skipped |
|-------|-------------|-------------------|
| Fabric config | TT-Fabric topology set to 2D | D2D sockets fail; no fabric routing |
| Device open | Mesh device opened, distributed context auto-init | All socket APIs fail; no device handle |
| Barrier | Cross-host sync | Race: data sent before receiver socket exists |
| Pipeline config | Stage types assigned, layer partitioning computed | Wrong stage types; sockets misconfigured |
| Socket creation | FIFOs allocated, connections established | Transfers hang on missing endpoints |
| Page size set | Transfer granularity configured | First transfer fails; alignment undefined |

---

## 7.3.3 Phase 2: Socket Setup

### What Breaks: Asymmetric Socket Creation Order

If hosts create sockets in different orders, cross-mesh connections deadlock. Host 0 blocks on Stage1-downstream (waiting for Stage2-upstream on Host 1), while Host 1 blocks on a different socket that Host 0 has not created yet.

### Socket Wiring (Annotated)

```python
# Intra-host D2D (same Galaxy): use mesh_ids
link_0_1 = ttnn.SocketConfig(connections, mem_config,
    sender_mesh_id=mesh_id_0, receiver_mesh_id=mesh_id_1)

# Inter-host D2D (cross-Galaxy): use ranks
link_1_2 = ttnn.SocketConfig(connections, mem_config,
    sender_rank=0, receiver_rank=1)  # DistributedContext handles routing
```

### D2D Connection Classification

| Connection | Source | Destination | Same host? | Transport |
|------------|--------|-------------|------------|-----------|
| Forward: Mesh 0 -> Mesh 1 | Host 0 | Host 0 | Yes | TT-Fabric (intra-Galaxy) |
| Forward: Mesh 1 -> Mesh 2 | Host 0 | Host 1 | No | TT-Fabric (cross-host) |
| Forward: Mesh 2 -> Mesh 3 | Host 1 | Host 1 | Yes | TT-Fabric (intra-Galaxy) |
| Loopback: Mesh 3 -> Mesh 0 | Host 1 | Host 0 | No | TT-Fabric (cross-host) |

Cross-host D2D connections use `sender_rank`/`receiver_rank`; intra-host connections can use `sender_mesh_id`/`receiver_mesh_id`.

---

## 7.3.4 Phase 3: Prefill

### What Breaks: Undersized FIFOs for Prefill Activations

Prefill processes the entire prompt in one forward pass, producing activations much larger than single-token decode activations. If FIFOs were sized for decode-phase tensors, the sender blocks in `socket_reserve_pages()` while the receiver cannot catch up.

**Fix**: Size FIFOs for worst-case (prefill) tensor size, or chunk prefill into FIFO-sized segments.

### Prefill Data Flow (Step by Step)

| Step | Mesh 0 (First) | Mesh 1 (Middle) | Mesh 2 (Middle) | Mesh 3 (Last+LB) |
|------|-----------------|------------------|------------------|-------------------|
| 1 | H2D: host writes token embeddings | -- (waiting) | -- (waiting) | -- (waiting) |
| 2 | Compute layers 0-14 | -- (waiting) | -- (waiting) | -- (waiting) |
| 3 | D2D send to Mesh 1 | D2D recv from Mesh 0 | -- (waiting) | -- (waiting) |
| 4 | -- | Compute layers 15-29 | -- (waiting) | -- (waiting) |
| 5 | -- | D2D send to Mesh 2 | D2D recv from Mesh 1 | -- (waiting) |
| 6 | -- | -- | Compute layers 30-44 | -- (waiting) |
| 7 | -- | -- | D2D send to Mesh 3 | D2D recv from Mesh 2 |
| 8 | -- | -- | -- | Compute layers 45-59 |
| 9 | -- | -- | -- | D2H: device writes logits to host |

Total socket hops: 1 H2D + 3 D2D + 1 D2H = 5 transfers.

### What Breaks: Page Size Not Set Before Transfer

Both H2DSocket and D2HSocket require `set_page_size()` before the first transfer. For D2HSocket, this call may block: "If alignment causes read pointer to wrap, waits for device to send enough data to cover the alignment adjustment."

---

## 7.3.5 Phase 4: Autoregressive Decode

### Token Trace: One Decode Iteration

The following table traces a single token through the full pipeline, identifying the exact socket type, FIFO location, and counter locations at each hop:

```
  Hop  From           To             Socket   FIFO Location    Counter Locations
  ---  ----           --             ------   -------------    -----------------
  0    Host CPU       Mesh 0 core    H2D      Device L1        Both in device L1 (host writes bytes_sent via TLB; device updates bytes_acked locally)
  1    Mesh 0 exit    Mesh 1 entry   D2D      Device L1        Internal (MeshSocket)
  2    Mesh 1 exit    Mesh 2 entry   D2D      Device L1        Internal (MeshSocket)
  3    Mesh 2 exit    Mesh 3 entry   D2D      Device L1        Internal (MeshSocket)
  4    Mesh 3 exit    Host CPU       D2H      Host pinned mem  Host pinned memory (bytes_sent and bytes_acked live in host RAM; device reads bytes_acked via L1 cache invalidation)
  LB   Mesh 3 exit    Mesh 0 entry   D2D      Device L1        Internal (MeshSocket)
```

A token enters via H2D (HOST_PUSH, counters in device L1), traverses stages via D2D (MeshSocket, counters managed internally), and exits via D2H (bytes_sent in host-pinned memory; bytes_acked written by host to device L1, read by device via cache invalidation). The loopback D2D fires simultaneously with the D2H write on the last stage.

### Decode Loop

```python
# fabric_loopback mode: loopback D2D feeds Stage 0 after step 0; host only reads D2H
for step in range(max_new_tokens):
    if step == 0:
        h2d_socket.write(seed_token_data, 1)  # First iteration only: H2D seed
    # Pipeline stages execute internally: recv -> compute -> send
    # After step 0, Stage 0 receives input from loopback D2D, not H2D
    d2h_socket.read(logit_buffer, num_logit_pages)
    next_token = sample(logit_buffer)
    if next_token == eos_token:
        break
# For host_loopback mode, h2d_socket.write() must be called every iteration
# (host reads D2H, samples, writes next token back via H2D)
```

### Prefill vs Decode Comparison

| Dimension | Prefill | Decode |
|-----------|---------|--------|
| Input size | Full prompt (many tokens) | Single token per iteration |
| Number of pipeline passes | 1 | N (one per generated token) |
| Loopback active | No | Yes (fabric_loopback) |
| Mesh 0 ingress | H2D | H2D (first iter) / Loopback D2D (subsequent) |
| Mesh 3 egress | D2H only | D2H + Loopback D2D |
| Dominant bottleneck | Compute (large activations) | Communication (small payloads, many round-trips) |

### What Breaks: D2H Read Without Notify

`D2HSocket.read()` defaults to `notify_sender=True`. Setting it to `False` for batching but forgetting to ever notify causes the device to block permanently in `socket_reserve_pages()`:

```python
# RIGHT -- batch reads, notify on the last one
for i in range(batch_size - 1):
    d2h_socket.read(buf, pages, notify_sender=False)
d2h_socket.read(buf, pages, notify_sender=True)  # Frees device buffer
```

### What Breaks: Decode Iteration Desynchronization

All stages must process the same token in lockstep. If one stage falls behind, FIFO backpressure prevents corruption but causes increasing latency as the pipeline stalls.

---

## 7.3.6 Phase 5: Teardown

### What Breaks: Teardown Before Pipeline Drain

Closing the mesh device while the last iteration is in flight kills device kernels mid-execution, losing D2H data in transit and corrupting socket state.

### What Breaks: D2HSocket Destructor Race

The D2HSocket destructor waits for device acknowledgement. If `close_mesh_device` runs first (e.g., Python GC ordering), the destructor hangs or throws.

### Correct Teardown Sequence

```python
# 1. Read final results
final_logits = d2h_socket.read(buf, pages)
# 2. Synchronize hosts
ttnn.distributed_context_barrier()
# 3. Delete sockets (destructors communicate with device)
del pipeline_blocks
# 4. Final barrier
ttnn.distributed_context_barrier()
# 5. Close device
ttnn.close_mesh_device(mesh_device)
```

**Rule**: Sockets must be destroyed before the mesh device is closed. Use explicit `del` rather than relying on garbage collection ordering.

---

## 7.3.7 Complete Timeline

Both hosts execute the same phases in lockstep: fabric config, mesh open, barrier, socket creation, prefill, decode loop, barrier, close. Each phase transition is a synchronization point. The D2D blocking recv provides implicit inter-host synchronization during steady-state inference -- no mid-pipeline barriers are needed.

---

## Key Takeaways

1. Both hosts must call `distributed_context_barrier()` after device open and before socket creation to ensure the distributed context is fully ready.

2. Socket creation order must be deterministic across all hosts. PipelineBlock enforces this via stage-index ordering to prevent cross-host deadlocks.

3. Prefill activations are much larger than decode activations. FIFOs must accommodate the worst case, or prefill must be chunked.

4. A token enters via H2D (counters in device L1), traverses D2D hops (counters internal to MeshSocket), and exits via D2H (counters in host-pinned memory) -- the token trace table is the debugging reference for identifying which counter to inspect at each hop.

5. `d2h_socket.read()` with `notify_sender=False` must eventually be followed by a notifying read, or the device blocks permanently.

6. Teardown order: drain pipeline, read results, barrier, delete sockets, barrier, close device. Never close the device with live sockets.

7. Cross-host D2D connections require rank-based addressing in SocketConfig; intra-host connections can use mesh IDs alone. The D2D blocking recv provides implicit synchronization during steady-state execution.

---

**Previous:** [7.2 Socket Interface and Host Interface](./02_socket_interface_and_host_interface.md) | **Next:** [7.4 Troubleshooting and Debugging](./04_troubleshooting_and_debugging.md) | **Up:** [Chapter 7 Index](./index.md)
