# 3.2 Data Movement Micro-Ops

Data movement micro-ops shuttle tiles between cores, between L1 and DRAM, and between logical tensor shapes and physical memory layouts. Their TRISC is either a no-op or performs only tilize/untilize -- no FPU arithmetic. What makes them architecturally interesting is the NOC programming: multicast addressing, semaphore choreography, dual-NOC routing optimization, and DRAM bank assignment.

This section covers five data movement micro-ops: **gather** (with the **gather_reduce** variant), **mcast**, **create_q_heads**, **kv_cache_update**, and the deep-dive treatment of **dram_streaming_matmul**.

---

## 3.2.1 gather (+ gather_reduce variant)

**Source:** `micro_ops/gather/op.py` | **Kernel headers:** `unified_kernels/gather.hpp`, `unified_kernels/gather_reduce.hpp`

### Purpose

Collects sharded data from $N$ sender cores onto a single receiver core. The output tensor on the receiver is the concatenation of all input shards. The inverse of mcast (Section 3.2.2).

### Computation

$$\text{output}_{\text{receiver}} = \text{concat}(\text{shard}_0, \text{shard}_1, \ldots, \text{shard}_{N-1})$$

### Data Flow Diagram (Dual-NOC Routing)

```
  NOC_0 senders                        NOC_1 senders
  (shorter hop via NOC 0)              (shorter hop via NOC 1)
  +------+  +------+  +------+        +------+  +------+  +------+
  |Core 0|  |Core 1|  |Core 2|        |Core 5|  |Core 6|  |Core 7|
  +--+---+  +--+---+  +--+---+        +--+---+  +--+---+  +--+---+
     |         |         |                |         |         |
     | NCRISC  | NCRISC  | NCRISC        | NCRISC  | NCRISC  | NCRISC
     | write   | write   | write         | write   | write   | write
     |         |         |                |         |         |
     v         v         v                v         v         v
     +----noc_async_write----+            +----noc_async_write----+
     | to receiver L1 addr   |            | to receiver L1 addr   |
     | at sender_idx offset  |            | at sender_idx offset  |
     +-----------+-----------+            +-----------+-----------+
                 |                                    |
                 | semaphore_inc(noc0_sem)             | semaphore_inc(noc1_sem)
                 v                                    v
              +------------------------------------------+
              |         Receiver Core (BRISC)            |
              | wait(noc0_sem, noc0_count)               |
              | wait(noc1_sem, noc1_count)               |
              | cb_push_back(dst_cb, total_pages)        |
              +------------------------------------------+
```

### CB Layout

| CB | Index | Role | Backed by |
|----|-------|------|-----------|
| src (senders) | 0 | Input shard data | Sharded tensor |
| dst (receiver) | 0 | Output concatenated data | Sharded tensor (output) |

### Compile-Time Args

| Arg Name | RISC | Type | Description |
|----------|------|------|-------------|
| `gather_src_cb` | NCRISC | `uint32_t` | Source CB index on senders |
| `gather_dst_cb` | BRISC | `uint32_t` | Destination CB index on receiver |
| `gather_src_num_pages` | NCRISC | `uint32_t` | Pages per shard |
| `gather_data_size_bytes` | NCRISC | `uint32_t` | Bytes per shard |
| `gather_noc0_num_senders` | BRISC | `uint32_t` | Senders using NOC 0 |
| `gather_noc1_num_senders` | BRISC | `uint32_t` | Senders using NOC 1 |
| `gather_sender_grid_bounds` | NCRISC | `uint32_t` | Packed (start_x, start_y, end_x, end_y) for grid mode |

**Per-core (scattered mode only):** `gather_sender_idx` -- unique index per sender, set via `PerCoreCompileTimeDescriptor`.

### Runtime Args

| RISC | Core | Args | Description |
|------|------|------|-------------|
| NCRISC | Senders | `[receiver_data_addr, receiver_sem_addr, receiver_noc_x/y]` | Receiver's L1 address and semaphore |

### RISC Roles

| Processor | Role | Core Set |
|-----------|------|----------|
| NCRISC | **Sender.** Waits for source CB, writes data to receiver's L1 at `receiver_data_addr + sender_idx * data_size`, increments receiver semaphore. | All sender cores |
| BRISC | **Receiver.** Waits for both NOC semaphores to reach expected counts, resets them, pushes output CB. | Receiver core only |
| TRISC | No-op. | All cores |

### Key Implementation Detail: Dual-NOC Routing Optimization

When `noc=None` (the default), the op automatically splits senders between NOC_0 and NOC_1 based on hop distance:

```python
for core in sender_cores:
    noc0_hop = device.get_worker_noc_hop_distance(core, gather_core, NOC_0)
    noc1_hop = device.get_worker_noc_hop_distance(core, gather_core, NOC_1)
    if noc0_hop <= noc1_hop:
        noc0_cores.append(core)
    else:
        noc1_cores.append(core)
```

Two separate semaphores (IDs 0 and 1) prevent race conditions: senders on NOC_0 increment semaphore 0, senders on NOC_1 increment semaphore 1. The receiver waits for both independently. The receiver's NOC is chosen to avoid conflicting with senders.

### Two Modes: Rectangular Grid vs. Scattered Cores

**Grid mode** (`op()`): Senders form a rectangular grid. Each sender computes its index from grid coordinates: `sender_idx = (core_y - start_y) * grid_width + (core_x - start_x)`.

**Scattered mode** (`op_scattered()`): Senders are at arbitrary positions. Each sender gets an explicit index via `PerCoreCompileTimeDescriptor`. More expensive in compilation but handles arbitrary core topologies.

### Receiver Write Address

A critical subtlety: the `receiver_data_addr` is passed as a **runtime arg** (`output_tensor.buffer_address()`), not through the CB, because the destination CB does not exist on sender cores. Senders write directly to the L1 address on the receiver core.

### gather_reduce Variant

`GatherReduceSingleCore` extends the basic gather by adding an element-wise reduction (typically summation) on the receiver core after all shards arrive. Instead of a simple concatenation, the receiver uses TRISC to accumulate all incoming tiles into a single output. This is used in the pre_sdpa pipeline where kn_sliced_matmul partial products must be gathered **and** summed.

### Composed Into

- **pre_sdpa** -- Gather after kn_sliced_matmul: 8 partial products gathered and reduced on a single core. Uses the gather_reduce variant.
- **post_sdpa** -- Gather kv_b2 outputs to assemble full KV projection.
- **moe** -- Gather after routed expert computation: partial results collected from expert cores.
- **shared_expert** -- Gather after shared expert gate/up matmul.
- **gated_local_reduce_down_proj** -- Input gather stage collecting partial results from multiple cores.
- **down_proj** -- Output gather for down projection results.
- **lm_head_sampling** -- Gather scores from matmul cores to final core for argmax.

---

## 3.2.2 mcast

**Source:** `micro_ops/mcast/op.py` | **Kernel header:** `unified_kernels/mcast.hpp`

### Purpose

Replicates data from a single sender core to multiple receiver cores using hardware multicast. The inverse of gather.

### Computation

$$\text{output}_d = \text{input}_{\text{sender}} \quad \forall \; d \in \text{receiver\_grid}$$

### CB Layout

| CB | Index | Role | Backed by |
|----|-------|------|-----------|
| src (sender) | 0 | Source data | Sharded tensor |
| dst (receivers) | 0 | Replicated data | Sharded tensor (output) |

### Compile-Time Args

| Arg Name | RISC | Type | Description |
|----------|------|------|-------------|
| `mcast_src_cb` | NCRISC, BRISC | `uint32_t` | Source CB index |
| `mcast_dst_cb` | NCRISC | `uint32_t` | Destination CB index |
| `mcast_num_cores` | NCRISC, BRISC | `uint32_t` | Number of receiver cores |
| `mcast_dest_noc_start_x/y` | BRISC | `uint32_t` | Multicast rectangle start (NOC coords) |
| `mcast_dest_noc_end_x/y` | BRISC | `uint32_t` | Multicast rectangle end (NOC coords) |
| `mcast_data_sender_semaphore_id` | NCRISC, BRISC | `uint32_t` | Readiness semaphore (receivers -> sender) |
| `mcast_data_receiver_semaphore_id` | NCRISC, BRISC | `uint32_t` | Data-ready semaphore (sender -> receivers) |
| `is_part_of_receiver_grid` | BRISC | `uint32_t` | Sender is also a receiver (self-include) |

### Runtime Args

None.

### RISC Roles

| Processor | Role |
|-----------|------|
| NCRISC | **Receiver.** Signals sender that it is ready via `mcast_data_sender_semaphore`. Waits for `mcast_data_receiver_semaphore`, then pushes destination CB. Also sets up source CB on sender core. |
| BRISC | **Sender.** Waits for all receivers to signal readiness. Performs `noc_async_write_multicast` from source CB to all receivers' L1 addresses. Signals all receivers via `noc_semaphore_set_multicast`. |
| TRISC | No-op. |

### Key Implementation Detail

The hardware multicast facility writes to all cores in a NOC coordinate rectangle simultaneously -- a single NOC transaction replaces $N$ individual writes. The multicast destination is defined by physical (NOC) coordinates:

```python
mcast_dest_noc_start_core = device.worker_core_from_logical_core(mcast_grid.start)
mcast_dest_noc_end_core   = device.worker_core_from_logical_core(mcast_grid.end)
```

If the sender core is within the multicast grid (`is_part_of_receiver_grid`), the multicast includes a loopback write to itself.

### Composed Into

- **pre_sdpa** -- Broadcast activation to kn_sliced_matmul cores.
- **post_sdpa** -- Broadcast gathered output to o_proj matmul cores.
- **moe** -- Broadcast activation from rmsnorm core to expert cores.
- **shared_expert** -- Broadcast activation to shared expert gate/up cores.
- **down_proj** -- Broadcast activation to down projection cores. Broadcast residual to output cores.
- **gated_local_reduce_down_proj** -- Broadcast reduced output to down proj cores.
- **lm_head_sampling** -- Broadcast rmsnorm output to vocab projection cores.

---

## 3.2.3 create_q_heads

**Source:** `micro_ops/create_q_heads/op.py` | **Kernel header:** `unified_kernels/create_q_heads.hpp`

### Purpose

Assembles Q heads from two source tensors (QNOPE and QROPE) using a three-phase write protocol. The most structurally complex single-device data movement op.

### Computation

Each of 8 receiver cores assembles data from 12 senders (8 QNOPE + 4 QROPE) to produce `[8, 576]` tiled output per receiver, where $576 = 512 (\text{NOPE}) + 64 (\text{ROPE})$.

### Data Flow Diagram (Three-Phase Write Protocol)

```
  QNOPE senders (8x8 grid)            QROPE senders (4x8 grid)
  +---+---+---+---+---+---+---+---+  +---+---+---+---+
  |c00|c01|c02|c03|c04|c05|c06|c07|  |c00|c01|c02|c03|
  +---+---+---+---+---+---+---+---+  +---+---+---+---+
  |c10|c11|... shard [1, 512] each |  |c10|... [2, 64]|
  +---+---+---+---+---+---+---+---+  +---+---+---+---+
  |...                             |  |...            |
  +--------------------------------+  +---------------+

  Phase 1: QNOPE cols 0-7 write first 256 elements each
           -> receiver's interm at offset col * 256 * elem_size
           -> signal nope_phase1_semaphore (ID=2)

  Phase 2: QNOPE cols 0-7 write second 256 elements each
           -> receiver's interm at offset (8*256 + col*256) * elem_size
           -> signal nope_phase2_semaphore (ID=3)

  Phase 3: QROPE cols 0-3 write 128 elements each (2 heads * 64)
           -> receiver's interm at offset (8*512 + 2*col*64) * elem_size
           -> signal rope_semaphore (ID=0)

  Receiver: wait for each phase semaphore, then tilize:
    Phase 1: [8, 256] -> 8 tiles
    Phase 2: [8, 256] -> 8 tiles
    Phase 3: [8, 64]  -> 2 tiles
    Total: 18 tiles of [8, 32] per receiver
```

### CB Layout

| CB | Index | Role | Backed by |
|----|-------|------|-----------|
| src (senders) | 0 | QNOPE or QROPE shard | Sharded tensor |
| qrope_src (senders) | 1 | QROPE sender CB | Sharded tensor |
| interm (receivers) | 2 | Row-major intermediate buffer | `interm_tensor` |
| out (receivers) | 3 | Tiled output | Sharded tensor |

### Compile-Time Args

| Arg Name | RISC | Type | Description |
|----------|------|------|-------------|
| `sender_grid_start_x/y` | NCRISC | `uint32_t` | QNOPE grid origin |
| `qrope_grid_start_x/y` | NCRISC | `uint32_t` | QROPE grid origin |
| `nope_phase1_semaphore_id` | NCRISC, BRISC | `uint32_t` | Semaphore ID 2 |
| `nope_phase2_semaphore_id` | NCRISC, BRISC | `uint32_t` | Semaphore ID 3 |
| `rope_semaphore_id` | NCRISC, BRISC | `uint32_t` | Semaphore ID 0 |
| `target_noc_x_y_packed` | NCRISC | `uint32_t` | Packed target coordinates (`noc_x | (noc_y << 16)`) |

### Runtime Args

| RISC | Core | Args | Description |
|------|------|------|-------------|
| NCRISC | Senders | `[interm_addr, sem_addr, noc_x, noc_y]` | Receiver's intermediate buffer address |

### RISC Roles

| Processor | Role | Core Set |
|-----------|------|----------|
| NCRISC | **Sender** (all 96 sender cores). Writes QNOPE/QROPE data to target receiver's intermediate tensor address, signals phase semaphores. | 64 QNOPE + 32 QROPE |
| BRISC | **Receiver** (8 receiver cores). Waits for phase semaphores from all senders per phase, then triggers tilization. | 8 receivers (4x2) |
| TRISC | **Compute on receivers only.** Runs `tilize_block` for each phase. No-op on sender-only cores. | 8 receivers |

### Key Implementation Detail

**Row-to-core mapping:** Each sender row maps to a specific receiver core:

```python
SENDER_ROW_TO_TARGET_CORE = {
    0: CoreCoord(0, 1),  1: CoreCoord(1, 1),
    2: CoreCoord(2, 1),  3: CoreCoord(3, 1),
    4: CoreCoord(0, 2),  5: CoreCoord(1, 2),
    6: CoreCoord(2, 2),  7: CoreCoord(3, 2),
}
```

**Three separate semaphores** prevent race conditions between phases: QNOPE senders signal after first half (sem 2), after second half (sem 3), and QROPE senders signal after completion (sem 0). The receiver waits for the appropriate count before beginning each tilization phase.

**NOC coordinate packing:** To avoid 16 separate compile-time args, target NOC coordinates are packed into a single `uint32`: `packed_coords = noc_x | (noc_y << 16)`. The kernel unpacks with bitwise operations.

### Composed Into

- **pre_sdpa** -- Assembles Q heads from QNOPE and QROPE outputs before SDPA computation.

---

## 3.2.4 kv_cache_update

**Source:** `micro_ops/kv_cache_update/op.py` | **Kernel header:** `unified_kernels/kv_cache_update.hpp`

### Purpose

Reads 16 BFP8 tiles from a DRAM-resident KV cache tensor, untilizes them (BFP8 to bfloat16), merges with new KV data from the current decode step, retilizes (bfloat16 to BFP8), and writes back to DRAM.

### Data Format Pipeline

```
DRAM (BFP8, 32x32 tiles)
    |  NCRISC read (TensorAccessorArgs)
    v
CB 31 (kv_cache_input, BFP8, 16 tiles)
    |  TRISC untilize
    v
CB 29 (kv_cache_intermed, bfloat16, 17 tiles)
    |  BRISC: merge new data at position_id slot
    v
CB 29 (updated intermediate)
    |  TRISC tilize
    v
CB 30 (kv_cache_output, BFP8, 16 tiles)
    |  BRISC write back to DRAM
    v
DRAM (BFP8, 32x32 tiles)
```

### CB Layout

| CB | Index | Format | Size | Purpose |
|----|-------|--------|------|---------|
| `kv_cache_input` | 31 | BFP8, 32x32 | 16 tiles | DRAM read buffer |
| `kv_cache_output` | 30 | BFP8, 32x32 | 16 tiles | DRAM write buffer |
| `kv_cache_intermed` | 29 | bfloat16, 32x32 | 17 tiles (extra for double-buffering) | Merge workspace |
| `kv_rmsnorm_output` | 26 | Tensor-backed | Varies | New NOPE data |
| `krope_output` | 27 | Tensor-backed | Varies | New ROPE data |

### Compile-Time Args

| Arg Name | RISC | Type | Description |
|----------|------|------|-------------|
| `is_nope_core` | All | `uint32_t` | 1 on NOPE cores, 0 otherwise |
| `is_rope_core` | All | `uint32_t` | 1 on ROPE cores, 0 otherwise |
| `kv_cache_input_cb` | NCRISC, TRISC | `uint32_t` | CB 31 |
| `kv_cache_output_cb` | BRISC, TRISC | `uint32_t` | CB 30 |
| `kv_cache_intermed_cb` | BRISC, TRISC | `uint32_t` | CB 29 |
| `kv_rmsnorm_output_cb` | NCRISC | `uint32_t` | CB 26 |
| `krope_output_cb` | NCRISC | `uint32_t` | CB 27 |
| `position_ids_tensor_address` | BRISC | `uint32_t` | L1 address of position IDs |
| TensorAccessorArgs | NCRISC | `uint32_t[]` | DRAM address computation |

### Runtime Args

None (position is read dynamically from `position_ids_tensor` in L1).

### RISC Roles

| Processor | Role |
|-----------|------|
| NCRISC | Reads DRAM KV cache tiles into CB31 using TensorAccessor. Provides addresses for kv_rmsnorm_output and krope_output CBs. |
| BRISC | Reads `position_id` from position_ids tensor. Manages the untilize-retilize pipeline. After update, multicasts `kv_cache_cur_pos_ready_semaphore` to the **entire device grid** so SDPA knows the cache is valid. |
| TRISC | Performs untilize (BFP8 to bfloat16) and tilize (bfloat16 to BFP8). |

### Key Implementation Detail

**NOC mode:** Uses `DM_DYNAMIC_NOC`, allowing both BRISC and NCRISC to dynamically select which NOC to use per transfer.

**Full-grid position ready multicast:** After writing the cache, BRISC multicasts a semaphore signal to `CoreRange(CoreCoord(0,0), CoreCoord(device_grid_size.x-1, device_grid_size.y-1))`, notifying the fused SDPA op (which may already be running on other cores) that the cache at the new position is valid.

**Core grid:** The op runs on two types of cores: NOPE cores (512-element nope portion) and ROPE cores (64-element rope portion), differentiated by `is_nope_core` / `is_rope_core` compile-time flags via `UnifiedCompileTimeCoreDescriptor`.

### Composed Into

- **kv_cache_branch** -- The fused kv_cache_branch kernel includes KV cache update as its final stage, writing the normalized and RoPE-processed KV data back to DRAM.

---

## 3.2.5 dram_streaming_matmul -- Deep Dive

**Source:** `micro_ops/dram_streaming_matmul/op.py` | **Kernel header:** `unified_kernels/dram_streaming_matmul.hpp`

`DRAMStreamingMatmul` is the single most complex micro-op in the entire DeepSeek V3 B1 codebase. It implements a multicore matmul where weights are streamed from DRAM bank-by-bank through triple-buffered circular buffers, with optional fused SiLU activation, element-wise multiplication, and scalar scaling. It is the workhorse for MoE expert computation, where 256 experts' weights live in DRAM and cannot fit in L1 simultaneously.

### 3.2.5.1 High-Level Architecture

- **Input A (`in0`):** REPLICATED on all compute cores. Each core has the full $[M, K]$ tensor in L1. For decode, $M = 1$.
- **Input B (`in1`):** WIDTH_SHARDED in DRAM as $[K, N]$. Each DRAM bank holds $[K, \text{per\_core\_N}]$ tiles, **pre-shuffled from row-major to column-major order**.
- **Output:** WIDTH_SHARDED on compute cores, $[M, \text{per\_core\_N}]$.

### 3.2.5.2 Column-Major Tile Reorder

The pre-shuffle is the critical insight. In the original row-major layout, tiles for different $N$ columns of the same $K$ row are contiguous, which is wrong for the streaming pattern. After column-major reorder, all $K$ tiles for a single $N$ column are contiguous in physical DRAM, enabling burst reads of entire K-sticks:

```
Standard row-major layout for a [K, N] tile grid:
  tile(0,0), tile(0,1), tile(0,2), ...  // row 0
  tile(1,0), tile(1,1), tile(1,2), ...  // row 1

Column-major reordering per shard:
  tile(0,0), tile(1,0), tile(2,0), ...  // column 0 (all K tiles)
  tile(0,1), tile(1,1), tile(2,1), ...  // column 1
```

Reading one column of $K_t$ tiles becomes a single contiguous DRAM burst, maximizing DRAM bandwidth utilization.

### 3.2.5.3 Triple Buffering Strategy

CB 1 (the DRAM read buffer) holds three sub-buffers, each containing `subblock_k` tiles:

```python
num_in1_buffers = 3
in1_CB_tiles = subblock_k * num_in1_buffers
in1_CB_size = in1_CB_tiles * in1_tile_size
```

The pipeline stages overlap to hide both DRAM read latency and compute latency:

```
  Cycle t | Buffer 0         | Buffer 1         | Buffer 2
  --------+------------------+------------------+------------------
  0       | READ K[0..s)     |                  |
  1       | COMPUTE K[0..s)  | READ K[s..2s)    |
  2       | FREE             | COMPUTE K[s..2s) | READ K[2s..3s)
  3       | READ K[3s..4s)   | FREE             | COMPUTE K[2s..3s)
```

### 3.2.5.4 NOC Burst Optimization: `get_max_page_size_and_num_pages()`

```python
def get_max_page_size_and_num_pages(device, num_tiles, tile_size):
    total_size = num_tiles * tile_size
    # Wormhole: 8192 bytes max, Blackhole: 16384 bytes max
    noc_max_page_size = 8192 if arch == WORMHOLE_B0 else 16384
    page_size = (noc_max_page_size // tile_size) * tile_size
    while total_size % page_size != 0 and page_size >= tile_size:
        page_size -= tile_size
    return page_size, total_size // page_size
```

This packs as many tiles as possible into each NOC read, up to the hardware burst limit. Larger pages mean fewer NOC transactions and higher effective bandwidth.

### 3.2.5.5 Subblock-K Tiling

When $K$ is large, the full $K$ dimension is split into `num_subblocks_k` blocks of `subblock_k` tiles:

```python
subblock_k = 4  # tiles per subblock (configurable)
num_subblocks_k = Kt // subblock_k
```

The kernel's inner loop:

```
for n in per_core_N:
    for k_sub in num_subblocks_k:
        wait(CB1, subblock_k tiles)
        for k in subblock_k:
            matmul_tiles(in0[k_sub * subblock_k + k], in1[k], out[n])
    pack out[n]
```

Smaller `subblock_k` reduces the triple-buffer size but increases loop overhead. The sweet spot depends on L1 capacity vs. DRAM latency.

### 3.2.5.6 Subblock-W and Dest Register Sizing

The output width is tiled into subblocks constrained by the number of destination registers:

```python
if fp32_dest_acc_en:
    max_dest = 4   # half-sync FP32
else:
    max_dest = 8   # half-sync BF16
subblock_w = max_dest
while subblock_w > 1 and per_core_N % subblock_w != 0:
    subblock_w -= 1
```

FP32 accumulation halves the available dest registers (4 vs. 8), reducing the output subblock width.

### 3.2.5.7 Expert Indexing

When `index_tensor` is provided, the kernel reads expert indices from a HEIGHT_SHARDED tensor (one index per core, stored as $[1, 16]$ tiles). The DRAM read offset becomes:

$$\text{offset} = \text{expert\_idx} \times K_t \times \text{in1\_block\_size\_bytes}$$

This enables the same kernel to serve any of 256 experts by simply changing the index tensor, without recompiling.

### 3.2.5.8 Fused SiLU + Multiply + Scalar Chain

The MoE gated expert computation requires:

$$\text{output} = \text{SiLU}(x \cdot W_{\text{gate}}) \odot (x \cdot W_{\text{up}}) \cdot s$$

`DRAMStreamingMatmul` fuses this into a single kernel invocation:

1. **SiLU** (`fuse_silu = 1`): applied to the matmul output in TRISC before writing to output CB.
2. **Element-wise multiply** (`enable_mul`): matmul output (CB 8, $1 \times 32$ tiles) reinterpreted as $16 \times 16$ tiles (CB 7), multiplied with `mul_tensor` (CB 6, also $16 \times 16$), written to final output (CB 4).
3. **Scalar multiply** (`enable_scalar_mul`): scalar from `scalar_tensor` (CB 10) broadcast-multiplied into the result (CB 9 as working buffer).

The tile format change ($1 \times 32 \to 16 \times 16$) is a view change only -- no data is moved. The number of $16 \times 16$ output tiles:

$$\text{mul\_num\_tiles} = \left\lceil \frac{M \times \text{per\_core\_N} \times \text{tile\_width}}{256} \right\rceil$$

### 3.2.5.9 Per-Core Bank ID and VC Assignment

Each core is assigned to a DRAM bank and virtual channel:

```python
for idx, core in enumerate(all_worker_cores):
    bank_id = idx % num_dram_banks
    vc = bank_id & 0x3
    # VC conflict resolution: avoid NOC contention for cores on same row
    for j in range(idx):
        if prev_core.y == core.y and (bank_ids[j] & 0x3) == (bank_id & 0x3):
            vc = (vc + 1) & 0x3
            break
```

When two cores in the same row would share the same VC (because their bank IDs share the low 2 bits), one is bumped to the next VC. These are passed as per-core compile-time args:

```python
PerCoreCompileTimeDescriptor("dram_mm_bank_id", bank_id_core_values, other_value=0)
PerCoreCompileTimeDescriptor("dram_mm_vc", vc_core_values, other_value=0)
```

### 3.2.5.10 Working Buffer and Loop Iteration

When `num_loop_iters > 1` (for testing CB boundary wrapping), a `working_buf_tensor` must be provided. Its buffer address is compiled into both NCRISC and BRISC so the kernel knows the CB 1 base address for wrap-around arithmetic. Without this, CB 1 is allocated as a plain L1 buffer (`cb_in1_buf_addr = 0`).

### 3.2.5.11 Complete CB Map

| CB ID | Name | Backing | Tile Format | Purpose |
|-------|------|---------|-------------|---------|
| 0 | in0 | input_a tensor | native | Full activation (replicated) |
| 1 | in1 | working_buf or L1 | native | Triple-buffered DRAM read |
| 4 | out (no mul) | output_tensor | native | Final output |
| 4 | out (with mul) | output_tensor | $16 \times 16$ | Final output after mul |
| 5 | index | index_tensor | native | Expert index (optional) |
| 6 | mul_in1 | mul_tensor | $16 \times 16$ | Multiply operand |
| 7 | mul_in0 | mm_out_tensor | $16 \times 16$ | Matmul output viewed as 16x16 |
| 8 | mm_out (with mul) | mm_out_tensor | native | Intermediate matmul output |
| 9 | scalar | L1 | $16 \times 16$ | Scalar working buffer |
| 10 | scalar_src | scalar_tensor | native | Scalar source value |

### 3.2.5.12 RISC Roles

| Processor | Role |
|-----------|------|
| NCRISC | Streams weight tiles from DRAM into CB1 using bank-direct addressing. Manages expert indexing (reads at `expert_idx * K_tiles` offset). Handles mul tensor CB setup. |
| BRISC | When mul is enabled: reads scalar from CB10, broadcasts to CB9 in 16x16 format. Otherwise no-op. |
| TRISC | Performs matmul with K-subblocking, optional fused SiLU, optional element-wise mul and scalar mul. |

### 3.2.5.13 Golden Reference

```python
result = input_a @ input_b
if fused_activation == "silu":
    result = F.silu(result)
if mul_tensor is not None:
    result = result * mul_tensor
if scalar_tensor is not None:
    result = result * scalar_tensor.flatten()[0]
```

### Composed Into

- **moe_routed_expert** -- Expert gate_proj and up_proj matmuls: DRAM-streamed matmul with expert indexing. The SiLU + mul + scalar chain is fused for the gated expert path.
- **shared_expert** -- Shared expert weight matmuls with DRAM streaming.
- **down_proj** -- Down projection with DRAM-streamed weights.
- **lm_head_sampling** -- Vocab projection with DRAM-streamed embedding weights.

---

## Summary Table

| Micro-Op | Direction | Cores | Key Mechanism |
|----------|-----------|-------|---------------|
| `gather` | Many $\to$ 1 | $N + 1$ | Dual-NOC routing, per-core sender index, separate semaphores |
| `gather_reduce` | Many $\to$ 1 (with sum) | $N + 1$ | Gather + TRISC reduction on receiver |
| `mcast` | 1 $\to$ Many | $1 + N$ | Hardware multicast rectangle, 2-semaphore protocol |
| `create_q_heads` | $96 \to 8$ | 104 | 3-phase semaphore, packed NOC coords, in-receiver tilize |
| `kv_cache_update` | L1 $\leftrightarrow$ DRAM | varies | BFP8 untilize/tilize, TensorAccessor, full-grid mcast semaphore |
| `dram_streaming_matmul` | DRAM $\to$ compute | bank-mapped | Column-major reorder, triple buffer, fused SiLU/mul/scalar, expert indexing, NOC burst opt, per-core bank_id/vc |

---

**Next:** [`03_communication_and_infrastructure_micro_ops.md`](./03_communication_and_infrastructure_micro_ops.md)
