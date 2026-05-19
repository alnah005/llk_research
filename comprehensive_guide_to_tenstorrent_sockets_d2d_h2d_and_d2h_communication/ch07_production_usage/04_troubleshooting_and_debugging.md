# 7.4 -- Troubleshooting and Debugging

Pipeline failures in a multi-mesh, multi-host DeepSeek V3 deployment are difficult to diagnose because symptoms on one mesh often originate from a different mesh, host, or socket type. This section provides a master symptom-cause-fix table, structured decision trees for systematic diagnosis, comparison matrices for distinguishing similar failures, and the CCL comparison for developers coming from collective communication backgrounds.

---

## 7.4.1 Master Symptom-Cause-Fix Table

| # | Symptom | Likely Cause | Root Location | Fix |
|---|---------|-------------|---------------|-----|
| 1 | Pipeline hangs, no output | Deadlock: two stages both blocking on send | D2D socket pair | Verify recv-before-send dispatch order in both stages |
| 2 | Pipeline hangs after first token | Loopback D2D not created or misconfigured | First or Last+LB stage | Check LoopbackConfig is fabric_loopback; verify loopback sockets exist |
| 3 | Pipeline hangs at initialization | Barrier reached by only one host | distributed_context_barrier() | Ensure both hosts reach the barrier; check network |
| 4 | Corrupted output logits | Page size mismatch between sender and receiver | Any socket pair | Verify set_page_size() called with identical values on both endpoints |
| 5 | Corrupted output, intermittent | FIFO misalignment: fifo_size % page_size != 0 | D2D or D2H FIFO | Ensure fifo_size is a multiple of page_size |
| 6 | Connection timeout on D2D | Peer mesh not yet initialized | Cross-host D2D socket | Add distributed_context_barrier() before socket creation |
| 7 | Connection timeout on H2D/D2H | export_descriptor() not called before connect() | H2D or D2H socket | Ensure export before connect (Chapter 6 ordering) |
| 8 | Fabric init failure | FABRIC_2D not set before device open | ttnn.set_fabric_config() | Call set_fabric_config(FABRIC_2D) before open_mesh_device() |
| 9 | Rank mismatch error | Wrong rank in StageMetadata | Pipeline config | Verify StageMetadata rank matches distributed_context_get_rank() |
| 10 | D2H read returns zeros | Device kernel did not call socket_push_pages() | D2H sender kernel | Verify kernel calls socket_push_pages() + socket_notify_receiver() |
| 11 | H2D write hangs | Device kernel not consuming pages | H2D receiver kernel | Verify kernel calls socket_wait_for_pages() + socket_pop_pages() |
| 12 | Correct but slow results | PCIe x1 chip used for H2D/D2H | HostIoPlacement | Move H2D/D2H cores to Gen4 x8 chips (ASIC 6) |
| 13 | Teardown crash | Sockets destroyed while data in flight | Teardown sequence | Call barrier() on all sockets before destruction |
| 14 | Stale /dev/shm descriptor | Previous run crashed without cleanup | /dev/shm/<socket_id> | Remove stale files; implement startup cleanup script |
| 15 | vIOMMU error on H2D/D2H | vIOMMU not enabled in BIOS/kernel | System configuration | Enable vIOMMU; all H2D and D2H modes require it |

---

## 7.4.2 Hang Diagnosis Decision Tree

Pipeline hangs are the most common failure mode because socket operations block with no default timeout.

```
  Pipeline is hung (no output, no crash).
  |
  +-- Did the pipeline ever produce output?
  |   |
  |   +-- NO, hung from the start
  |   |   |
  |   |   +-- Is distributed_context_barrier() completing?
  |   |   |   |
  |   |   |   +-- NO --> One host is not reaching the barrier.
  |   |   |   |          Check: network, process launch, fabric config.
  |   |   |   |
  |   |   |   +-- YES --> Are D2D sockets created on both sides?
  |   |   |               |
  |   |   |               +-- NO --> Socket creation failed.
  |   |   |               |          Check: mesh_id, rank in SocketConfig.
  |   |   |               |
  |   |   |               +-- YES --> Is page_size set on all sockets?
  |   |   |                           |
  |   |   |                           +-- NO --> Call set_page_size().
  |   |   |                           +-- YES --> Check dispatch order.
  |   |   |                                       Is recv issued before send?
  |   |
  |   +-- YES, hung after N tokens
  |       |
  |       +-- N == 1 (hung after first token)
  |       |   |
  |       |   +-- Loopback path issue. Check:
  |       |       - LoopbackConfig == fabric_loopback?
  |       |       - Loopback D2D sockets created on both ends?
  |       |       - Loopback D2D page_size set?
  |       |
  |       +-- N > 1 (hung after several tokens)
  |           |
  |           +-- Likely FIFO backpressure or slow consumer.
  |               Check: FIFO utilization, D2H notify_sender usage,
  |               whether bytes_acked is drifting behind bytes_sent.
```

### Diagnosis Techniques

**Flow control counters**: `bytes_sent == bytes_acked` means buffer empty (look upstream). `bytes_sent - bytes_acked == fifo_size` means buffer full (look downstream). A gap between zero and fifo_size means the receiver is falling behind.

**Barrier isolation**: Insert `distributed_context_barrier()` calls between stages with print statements. The last message before the hang identifies the stuck stage. Remove barriers after diagnosis -- they change timing and may mask races.

---

## 7.4.3 Data Corruption Diagnosis

### Corruption Source Comparison

| Indicator | Socket-Level Corruption | Model/Compute Error | Numerical Precision |
|-----------|------------------------|--------------------|--------------------|
| Reproducible with same input | Sometimes (timing-dependent) | Always | Always |
| Output contains NaN/Inf | Rare (usually garbage bytes) | Possible | Common |
| Corruption pattern | Random bytes, shifted data | Structured but wrong values | Small deviations |
| Affected by FIFO size change | Yes | No | No |
| Affected by page size change | Yes | No | No |
| Occurs on single-mesh (Combined) | Only if H2D/D2H issue | Yes | Yes |
| Occurs only on multi-mesh | Likely D2D issue | Unlikely | Unlikely |

### Socket-Level Corruption Causes

| Cause | Mechanism | Diagnosis | Fix |
|-------|-----------|-----------|-----|
| Page size mismatch | Sender writes N-byte pages, receiver reads M-byte pages | Compare set_page_size() on both endpoints | Use identical page sizes |
| FIFO wraparound misalignment | fifo_size % page_size != 0 | Check alignment | Ensure fifo_size % page_size == 0 |
| Alignment violation | Page size not multiple of 64 bytes | Check page_size % 64 | Round up to 64-byte multiple |
| Core reuse conflict | Two sockets share a core, FIFOs overlap | Check HostIoPlacement for duplicates | Assign distinct cores |
| H2D write before kernel ready | Data lands in L1 before kernel init | Random first output | Synchronize kernel launch with first write |
| vIOMMU disabled | Memory mapping failure | Check dmesg for IOMMU faults | Enable vIOMMU |
| Missing embedding tensor | No embedding lookup | All-same-token output | Verify embedding_tensor in PipelineBlock |

**Checksum comparison**: Print `tensor.sum().item()` on sender and receiver sides. Divergent checksums indicate transport corruption. Matching checksums with wrong output indicate a compute stage bug.

---

## 7.4.4 Connection and Fabric Errors

### Connection Error Comparison

| Error Type | Socket Type | Cause | Symptom | Fix |
|------------|-------------|-------|---------|-----|
| D2D connection timeout | MeshSocket | Peer not created yet | create_socket_pair() hangs | Barrier before socket creation |
| D2D fabric route failure | MeshSocket | Fabric not initialized | Exception on first send | set_fabric_config() before device open |
| H2D connect timeout | H2DSocket | No export_descriptor() | connect() raises after timeout_ms | Fix owner/connector ordering |
| D2H connect timeout | D2HSocket | No export_descriptor() | connect() raises after timeout_ms | Fix owner/connector ordering |
| Rank mismatch | MeshSocket | Wrong rank in SocketConfig | Exception on creation | Match ranks to distributed_context_get_rank() |
| Mesh ID mismatch | MeshSocket | Wrong mesh_id in StageMetadata | Socket created but no data flows | Verify mesh_id matches physical mesh |

### Fabric Initialization Checklist

Verify in order: (1) `set_fabric_config(FABRIC_2D)` called before device open, (2) `open_mesh_device(MeshShape(4,4))` returns valid handle, (3) `distributed_context_is_initialized()` returns True, (4) `distributed_context_get_size()` matches host count, (5) physical Ethernet links are up.

---

## 7.4.5 Performance Diagnosis

### Performance Bottleneck Comparison Matrix

| Bottleneck | Symptom | Diagnosis | Fix |
|------------|---------|-----------|-----|
| PCIe bandwidth (H2D/D2H) | Prefill slow, decode acceptable | H2D/D2H on x1 PCIe chip | Move to x8 PCIe chip (ASIC 6) |
| D2D fabric bandwidth | All stages slow uniformly | Cross-host D2D link saturated | Check inter-Galaxy Ethernet utilization |
| FIFO too small | Intermittent stalls, bursty throughput | Producer blocks on full FIFO | Increase fifo_size (2x-4x activation tensor) |
| Pipeline bubble | Decode latency scales linearly with stages | Last stage idle while first computes | Rebalance layer distribution |
| Compute imbalance | One stage consistently slower | Profile per-stage compute time | Redistribute layers across stages |
| Loopback latency | Decode throughput limited by round-trip | Loopback D2D crosses host boundaries | Minimize loopback distance |
| Small page size overhead | High per-page overhead | Profile with different page sizes | Increase page size |

### PCIe Bandwidth Reference (Blackhole Galaxy)

| Chip Position | PCIe Config | Approximate BW | Suitable for H2D/D2H? |
|---------------|-------------|----------------|----------------------|
| ASIC 6 (4 chips per Galaxy) | Gen4 x8 | ~12.8 GB/s | Yes (preferred) |
| Other ASICs (28 chips per Galaxy) | Gen4 x1 | ~1.6 GB/s | Avoid for latency-sensitive I/O |

**FIFO sizing rules**: minimum = activation_tensor_bytes, recommended = 2x. Constraints: `fifo_size % page_size == 0`, `page_size >= 64` (NOC word), `page_size % 16 == 0` (SRAM alignment).

---

## 7.4.6 Teardown and Cleanup

### Teardown Problem Comparison

| Problem | Cause | Symptom | Fix |
|---------|-------|---------|-----|
| Hang on socket destroy | Data in flight | Process blocks in destructor | barrier() on all sockets first |
| Kernel panic on device close | Fabric connections not severed | System crash | Destroy all sockets before close |
| Stale /dev/shm files | Owner crashed | Next run connects to dead descriptor | rm /dev/shm/<socket_id> |
| Leaked pinned memory | D2H destroyed without barrier | Host memory not returned | Barrier + proper destructor |
| Cross-host teardown race | Host 0 destroys before Host 1 | Host 1 D2D send fails | distributed_context_barrier() first |

### Correct Teardown Sequence

```
  WRONG:                            RIGHT:
  1. close_mesh_device()            1. distributed_context_barrier()
  2. del pipeline_blocks            2. del pipeline_blocks
     (HANG -- device gone)          3. distributed_context_barrier()
                                    4. close_mesh_device()
```

> **Note:** The "RIGHT" sequence assumes the final `d2h_socket.read()` has already completed (see Section 7.3.6 for the full end-of-inference teardown that includes the read step).

### Abnormal Termination

If aborting mid-iteration, `read()` would block on data that will never arrive. Use `discard_pending_pages()` instead:

```python
d2h_socket.discard_pending_pages()  # Rebases bytes_acked to bytes_sent
del d2h_socket                      # Safe to destroy
```

### /dev/shm Management

Stale descriptor files from crashed runs are a persistent operational hazard. Run `rm /dev/shm/pipeline_socket_*` before each pipeline launch. Use a naming convention that includes the pipeline instance ID to avoid collisions when multiple pipelines share a host.

---

## 7.4.7 Sockets vs CCL: When to Use Which

For developers familiar with collective communication libraries, this comparison maps CCL concepts to the socket-based pipeline approach.

| Dimension | Sockets (MeshSocket, H2D, D2H) | CCL (all_gather, reduce_scatter) |
|-----------|--------------------------------|----------------------------------|
| Pattern | Point-to-point, 1:1 | Collective (all-reduce, all-gather) |
| Topology | Explicit (user wires each connection) | Implicit (library chooses rings/trees) |
| Flow control | FIFO backpressure per link | Barrier-synchronized |
| Deadlock risk | Medium (user ensures recv-before-send) | Low (library manages ordering) |
| Flexibility | Arbitrary topologies including loopback | Fixed collective patterns |
| Pipeline stages | Native (PipelineBlock, StageMetadata) | Not native (user builds on top) |
| Cross-process | export_descriptor/connect (H2D/D2H only) | Not typical |
| Debug visibility | Transparent (counter values, FIFO state) | Opaque (library internals) |

**What breaks using sockets for all-reduce**: O(N) pairs, manual reduction, slower than CCL's ring/tree algorithms, one failure breaks everything.

**What breaks using CCL for pipeline parallelism**: Barrier synchronization forces all stages to the same token, destroying pipeline throughput.

**Rule**: Sockets for pipeline parallelism. CCL for tensor/data parallelism. Mixing them in the wrong roles causes either poor performance or deadlocks.

---

## 7.4.8 Quick Reference: Symptom-to-Fix

| Symptom | Most Likely Cause | Section |
|---------|-------------------|---------|
| Hang on first iteration | Socket creation order mismatch | 7.4.2 |
| Hang after N iterations | D2H read without notify | 7.4.2 |
| Garbage output from start | H2D write before kernel ready | 7.4.3 |
| Output degrades over time | fifo_size % page_size != 0 | 7.4.3 |
| MeshSocket constructor timeout | Peer not running | 7.4.4 |
| Core reuse validation error | Duplicate cores in SocketConfig | 7.4.4 |
| 2-3x expected latency | PCIe slot bandwidth mismatch | 7.4.5 |
| Process hangs on exit | Device closed before sockets | 7.4.6 |

---

## Key Takeaways

1. The master symptom-cause-fix table (Section 7.4.1) covers the 15 most common pipeline failures. Start diagnosis there before deep-diving into decision trees.

2. Hangs divide into two categories: never-produced-output (init/config issues) and hung-after-N-tokens (loopback or backpressure issues). Flow control counters reveal whether the sender or receiver side is stuck.

3. Data corruption from socket misconfiguration is distinguishable from model errors by sensitivity to FIFO/page size changes. Use the corruption source comparison matrix to isolate the cause.

4. Cross-host failures almost always trace back to rank mismatch, missing barriers, or fabric initialization ordering.

5. PCIe bandwidth varies 8x across chip positions in a Galaxy (x8 vs x1); `HostIoPlacement` on the wrong chip is the most common performance issue.

6. Teardown must follow reverse-init order with barriers at every step. `discard_pending_pages()` is the safe path for abnormal termination.

7. Sockets = pipeline parallelism. CCL = tensor/data parallelism. The comparison matrix maps the tradeoffs for developers transitioning between paradigms.

---

**Previous:** [7.3 End-to-End Inference Walkthrough](./03_end_to_end_inference_walkthrough.md) | **Up:** [Chapter 7 Index](./index.md)
