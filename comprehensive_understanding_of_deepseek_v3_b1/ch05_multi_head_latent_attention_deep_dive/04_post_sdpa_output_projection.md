# 5.4 Post-SDPA Output Projection

This section traces the final stage of MLA: how the 512-dimensional attention output per head is projected back to the model's hidden dimension via a two-stage matmul chain (`kv_b2_proj` then `o_proj`), with gather, mcast, and CCL all-reduce operations connecting them. The `post_sdpa` fused op chains six stages -- Matmul4 + Gather2 + Mcast3 + Matmul5 + Gather3 + CCL All-Reduce -- into a single kernel dispatch across 130+ cores. Under TP=2, each device holds partial results for its 64 heads; the all-reduce sums these partial contributions to reconstruct the full attention output.

Source: `fused_ops/post_sdpa/op.py`, `blitz_decode_weights.py`

---

## 5.4.1 The Mathematical Chain

After FlashMLA produces attention output $A \in \mathbb{R}^{64 \times 512}$ (64 heads, each with 512-element V output) on each device, the output projection reconstructs the hidden state:

$$\text{Per-head:} \quad h^{(i)} = A^{(i)} \cdot W_{kv\_b2}^{(i)} \qquad W_{kv\_b2}^{(i)} \in \mathbb{R}^{512 \times 128}$$

$$\text{Concatenated:} \quad H = [h^{(0)} \| h^{(1)} \| \cdots \| h^{(63)}] \in \mathbb{R}^{1 \times 8192}$$

$$\text{Output projection:} \quad y_{local} = H \cdot W_{o\_proj} \qquad W_{o\_proj} \in \mathbb{R}^{8192 \times 7168}$$

$$\text{All-reduce:} \quad y = \sum_{d=0}^{TP-1} y_{local}^{(d)} + x_{residual}$$

The intermediate dimension of 8192 comes from $64 \text{ heads} \times 128 \text{ head\_dim}$ -- this is the decompressed per-head output dimension before the final projection back to hidden size. For TP=2:
- Device 0 computes $y_{local}^{(0)}$ from heads 0-63.
- Device 1 computes $y_{local}^{(1)}$ from heads 64-127.
- The all-reduce sums both partial products and adds the residual connection.

---

## 5.4.2 End-to-End Shape Trace

```
Stage 0 -- FlashMLA Output (from Section 5.3)
  Shape: [1, 1, 64, 512]     (64 heads, 512-dim compressed output)
  Per S1 core: [8, 512] = 8 heads x 512 dim
  Tile: [8x32] tiny tiles -> scattered to [1x32] for matmul4

     |  Scatter: S1 cores extract rows from [8x32] tiles into [1x32] tiles
     |  Route to matmul4 cores (64 cores, one per head)
     v

Stage 1 -- Per-Head Input to kv_b2 Matmul
  Shape: [1, 512] per core    (64 cores, one head each)
  Tiles: 16 x [1x32]          (512 / 32 = 16 tiles)
  Layout: L1, matmul4_in0_cb (CB 0), sharded on kv_b2 grid

     |  Matmul4: [1, 512] x [512, 128] -> [1, 128]
     |  W_kv_b2_proj: HEIGHT_SHARDED on 64 cores (tile-rearranged to [8192, 512])
     |    Core grid: (8,0)-(12,7) [40 cores] + (0,8)-(11,9) [24 cores] = 64
     |    Shard: [128, 512] = 4x16 x [32x32] tiles per core
     v

Stage 2 -- Per-Head Decompressed Output
  Shape: [1, 128] per core    (64 cores)
  Tiles: 4 x [1x32]           (128 / 32 = 4 tiles per core)
  Layout: L1, matmul4_out_cb (CB 2)

     |  Gather2: 64 cores -> single core (12, 9)
     |  64 cores x 4 tiles = 256 tiles = [1, 8192]
     v

Stage 3 -- Concatenated Head Output
  Shape: [1, 8192]            (64 heads x 128 dim)
  Tiles: 256 x [1x32]         (8192 / 32 = 256 tiles)
  Layout: L1, gather2_dst_cb (CB 3), on core (12, 9)

     |  Mcast3: (12, 9) -> 13x10 grid = 130 cores
     |  Data size: 256 tiles x 64 bytes = 16,384 bytes
     v

Stage 4 -- Broadcast to o_proj Grid
  Shape: [1, 8192] per core   (256 x [1x32] tiles, replicated)
  Layout: L1, matmul5_in0_cb (CB 4), on all 130 mcast3 grid cores

     |  Matmul5: [1, 8192] x [8192, 64] -> [1, 64]
     |  W_o_proj: WIDTH_SHARDED on 112 active cores
     |    Core grid: (0,0)-(11,7) [96 cores] + (1,8)-(8,9) [16 cores] = 112
     |    Shard: [8192, 64] = 256x2 x [32x32] tiles per core
     |    (18 inactive mcast3 cores skip matmul5: is_matmul5_core=false)
     v

Stage 5 -- Per-Core o_proj Output
  Shape: [1, 64] per core     (112 active cores)
  Tiles: 2 x [1x32]           (64 / 32 = 2 tiles per core)
  Layout: L1, matmul5_out_cb (CB 6)

     |  Gather3: 112 cores -> single core (12, 9)
     |  112 cores x 2 tiles = 224 tiles = [1, 7168]
     v

Stage 6 -- Full Hidden State (Per-Device)
  Shape: [1, 7168]            (hidden_dim, matches model width)
  Tiles: 224 x [1x32]         (7168 / 32 = 224 tiles)
  Layout: L1, gather3_dst_cb (CB 7), on core (12, 9)

     |  CCL All-Reduce: sum [1, 7168] across all devices in mesh
     |  Plus optional residual addition (skip connection from input)
     v

Stage 7 -- Reduced Hidden State
  Shape: [1, 7168]            (attention output + residual)
  Layout: L1, ccl_output_cb (CB 12), on core (12, 9)
  This becomes the input to the FFN/MoE block
```

---

## 5.4.3 Complete Core Grid Diagram

The post_sdpa op uses four distinct core regions within the 13x10 grid. The kv_b2 (Matmul4) grid is (8,0)-(12,7) + (0,8)-(11,9) = 64 cores. The o_proj (Matmul5) grid is (0,0)-(11,7) + (1,8)-(8,9) = 112 cores. These overlap on 48 cores.

```
     Col 0   1   2   3   4   5   6   7   8   9  10  11  12
     ───────────────────────────────────────────────────────
R0  | M5  | M5  | M5  | M5  | M5  | M5  | M5  | M5  |M45  |M45  |M45  |M45  | M4  |
R1  | M5  | M5  | M5  | M5  | M5  | M5  | M5  | M5  |M45  |M45  |M45  |M45  | M4  |
R2  | M5  | M5  | M5  | M5  | M5  | M5  | M5  | M5  |M45  |M45  |M45  |M45  | M4  |
R3  | M5  | M5  | M5  | M5  | M5  | M5  | M5  | M5  |M45  |M45  |M45  |M45  | M4  |
R4  | M5  | M5  | M5  | M5  | M5  | M5  | M5  | M5  |M45  |M45  |M45  |M45  | M4  |
R5  | M5  | M5  | M5  | M5  | M5  | M5  | M5  | M5  |M45  |M45  |M45  |M45  | M4  |
R6  | M5  | M5  | M5  | M5  | M5  | M5  | M5  | M5  |M45  |M45  |M45  |M45  | M4  |
R7  | M5  | M5  | M5  | M5  | M5  | M5  | M5  | M5  |M45  |M45  |M45  |M45  | M4  |
R8  | M4  |M45  |M45  |M45  |M45  |M45  |M45  |M45  |M45  | M4  | M4  | M4  | mc  |
R9  | M4  |M45  |M45  |M45  |M45  |M45  |M45  |M45  |M45  | M4  | M4  | CS  | GC  |
     ───────────────────────────────────────────────────────

  Legend:
    M5   = Matmul5 (o_proj) core only
           Cols 0-7, Rows 0-7 = 64 cores
    M45  = Both Matmul4 (kv_b2) AND Matmul5 (o_proj) core
           Cols 8-11, Rows 0-7 = 32 cores
           Cols 1-8, Rows 8-9 = 16 cores
           Total: 48 cores
    M4   = Matmul4 (kv_b2) core only
           Col 12, Rows 0-7 = 8 cores
           Col 0, Rows 8-9 = 2 cores
           Cols 9-11, Rows 8-9 = 6 cores
           Total: 16 cores
    mc   = Mcast3-only core (12, 8): part of 13x10 grid, no matmul
    CS   = CCL Sender core (11, 9)
    GC   = Gather/CCL Receiver core (12, 9)

  Matmul4 total:  48 (M45) + 16 (M4) = 64 cores (kv_b2 grid)
  Matmul5 total:  48 (M45) + 64 (M5) = 112 cores (o_proj grid)
  Mcast3 grid:    13x10 = 130 cores (rectangular superset)
  Inactive mcast: 18 cores (receive mcast but skip matmul5; 16 of these are M4-only cores)
```

### Core Role Summary

| Role | Grid | Core Count | CB Participation |
|------|------|-----------|-----------------|
| Matmul4 (kv_b2) | (8,0)-(12,7) + (0,8)-(11,9) | 64 | CB 0-2 |
| Gather2 sender | Same as Matmul4 | 64 | CB 2 (source) |
| Gather2 receiver + Mcast3 sender | Core (12, 9) | 1 | CB 3, CB 4 |
| Mcast3 receiver | 13x10 full grid | 130 | CB 4 |
| Matmul5 (o_proj) | (0,0)-(11,7) + (1,8)-(8,9) | 112 | CB 4-6 |
| Gather3 sender | Same as Matmul5 | 112 | CB 6 (source) |
| Gather3 receiver | Core (12, 9) | 1 | CB 7 |
| CCL Sender | Core (11, 9) | 1 | CB 8, CB 13 |
| CCL Receiver | Core (12, 9) | 1 | CB 7, CB 9-13 |

### Why 130 Cores Instead of 112?

The mcast3 grid must be **rectangular** for efficient hardware multicast. The o_proj grid has 112 cores in a non-rectangular layout ($12 \times 8 + 8 \times 2$). Embedding this in the smallest rectangle gives $13 \times 10 = 130$. The 18 extra cores receive the mcast data (unavoidable with rectangular mcast) but are tagged `is_matmul5_core=false` to skip the matmul. This wastes $18 \times 16$ KB $= 288$ KB of aggregate L1, but the simplicity of a single rectangular mcast is worth this cost.

---

## 5.4.4 Stage 1: Matmul4 -- kv_b2 Decompression

The FlashMLA output $[1, 512]$ per head is the input to Matmul4. Each of the 64 kv_b2 cores holds one head's $W_{kv\_b2}^{(h)}$ weights:

$$\text{Matmul4: } [1, 512] \times [512, 128] = [1, 128] \text{ per core}$$

### Weight Layout

$W_{kv\_b2\_proj}$ has raw shape $(512, 8192)$, which is tile-rearranged via `shuffle_kv_b2()` to $(8192, 512)$ for the height-sharded layout. From `KVB12_PROJ_SingleDeviceOverlapSpec`:

```
  Logical shape: [512, 8192] -> tile-rearranged to [8192, 512]
  Sharding: HEIGHT_SHARDED on 64 cores
  Shard: [128, 512] per core    (128 rows = one head's v_head_dim slice)
  Core grid: (8,0)-(12,7) [5x8=40] + (0,8)-(11,9) [12x2=24] = 64 cores
  Dtype: BFP8 (tile size = 1088 bytes per 32x32 tile)
  Tiles per shard: (128/32) x (512/32) = 4 x 16 = 64 tiles
  Bytes per shard: 64 x 1088 = 69,632 bytes ~ 68 KB
```

### Per-Core CB Layout

| CB | Index | Tile | Count | Per-Core Size | Content |
|----|-------|------|-------|---------------|---------|
| `matmul4_in0_cb` | 0 | $1 \times 32$ | 16 | 1,024 B | SDPA output ($1 \times 512$) |
| `matmul4_in1_cb` | 1 | BFP8 $32 \times 32$ | 64 | 69,632 B | $W_{kv\_b2}$ shard (OverlappedTensor) |
| `matmul4_out_cb` | 2 | $1 \times 32$ | 4 | 256 B | Result ($1 \times 128$) |

**L1 budget per Matmul4 core:** $69,632 + 1,024 + 256 = 70,912$ bytes $\approx 69$ KB.

### Non-Rectangular Grid Handling

The 64 kv_b2 cores form a non-rectangular grid (two rectangular ranges). Each core gets a contiguous sender index for the subsequent gather via `gather2_sender_idx_per_core`, pre-computed at program construction time:

```python
matmul4_cores = ttnn.corerange_to_cores(matmul4_core_grid, row_wise=True)
gather2_sender_idx_per_core = [(core, idx) for idx, core in enumerate(matmul4_cores)]
```

### TP=2 Weight Distribution

For TP=2, the full $W_{kv\_b2}$ has shape $(512, 16384)$ across all 128 heads. Each device receives its 64-head slice:

```python
per_device_b2_w = cfg.kv_b2_proj_shape[1]  # 8192
b2_slice = kv_b2_proj_weights[:, tp_idx * per_device_b2_w : (tp_idx + 1) * per_device_b2_w]
```

---

## 5.4.5 Stage 2: Gather2 -- 64 Cores to Single Core

After matmul4, 64 per-head results ($[1, 128]$ each) must be concatenated into $[1, 8192]$:

```
Gather2 topology:
  Senders: 64 kv_b2 cores
  Receiver: core (12, 9)
  Transport: NOC_0 (all senders), BRISC on receiver

Per-sender:
  Source CB: matmul4_out_cb (CB 2)
  Tiles: matmul4_out_w_per_core = 4 tiles
  Data size: 4 tiles x 64 bytes (1x32 BF16 tile) = 256 bytes
  Write offset = sender_idx * 256

Receiver:
  Dest CB: gather2_dst_cb (CB 3)
  Total tiles: 64 cores x 4 tiles = 256 tiles
  Total size: 256 x 64 = 16,384 bytes = 16 KB
  Semaphore: gather2_noc0_receiver_semaphore (ID 0)
  Wait condition: noc0_num_senders = 64

Result in gather2_dst_cb:
  [sender_0: 128 elems | sender_1: 128 elems | ... | sender_63: 128 elems]
  = [1, 8192] in row-major order (head 0 through head 63)
```

---

## 5.4.6 Stage 3: Mcast3 -- Broadcast to 13x10 Grid

The concatenated $[1, 8192]$ is broadcast from the gather core to all 130 cores of the 13x10 grid:

```
Mcast3 parameters:
  Source: core (12, 9)
  Destination grid: (0,0)-(12,9) = 13 columns x 10 rows = 130 cores
  Source CB: gather2_dst_cb (CB 3)
  Dest CB: matmul5_in0_cb (CB 4)

  Data size: 256 tiles x 64 bytes = 16,384 bytes = 16 KB
  Num pages: 256 (1x32 tiles)
  Semaphores: mcast3_data_sender (ID 2), mcast3_data_receiver (ID 3)
  Loopback: mcast3_is_part_of_receiver_grid = true (core (12,9) is in mcast grid)
```

After mcast3, all 130 cores have $[1, 8192]$ = 256 tiles in CB 4. Only 112 of these are active matmul5 cores (`is_matmul5_core=true`); the remaining 18 receive the data but skip the matmul.

---

## 5.4.7 Stage 4: Matmul5 -- o_proj Output Projection

The o_proj matmul projects the 8192-dim concatenated head output back to the 7168-dim hidden state. This is the most compute-intensive stage in the post-SDPA pipeline.

$$\text{Matmul5: } [1, 8192] \times [8192, 64] = [1, 64] \text{ per core (112 active cores)}$$

### Weight Layout

From `O_PROJ_GATE_MM_RMSNORM_GAMMA_SingleDeviceOverlapSpec`:

```
  Logical shape: [8192, 7168]     (num_heads*v_head_dim x hidden_dim)
  Sharding: WIDTH_SHARDED on 112 cores
  Shard: [8192, 64] per core     (64 output columns per core)
  Core grid: (0,0)-(11,7) [96] + (1,8)-(8,9) [16] = 112 cores
  Dtype: BFP8 (tile size = 1088 bytes per 32x32 tile)
  Tiles per shard: (8192/32) x (64/32) = 256 x 2 = 512 tiles
  Bytes per shard: 512 x 1088 = 557,056 bytes ~ 544 KB per core

Total o_proj weight: 112 cores x 557,056 = 62.4 MB in L1
  (This is the largest single weight in the attention block)
```

**The o_proj weight shard is the largest single L1 consumer in the entire model** at $544$ KB per core. In BF16, each shard would be $512 \times 2048 = 1,048,576$ bytes $\approx 1$ MB, which would exceed the L1 budget when combined with activation CBs. The BFP8 format is essential.

### Per-Core CB Layout

| CB | Index | Tile | Count | Per-Core Size | Content |
|----|-------|------|-------|---------------|---------|
| `matmul5_in0_cb` | 4 | $1 \times 32$ | 256 | 16,384 B | Mcast3 result ($1 \times 8192$) |
| `matmul5_in1_cb` | 5 | BFP8 $32 \times 32$ | 512 | 557,056 B | $W_{o\_proj}$ shard (OverlappedTensor) |
| `matmul5_out_cb` | 6 | $1 \times 32$ | 2 | 128 B | Result ($1 \times 64$) |

**L1 budget per Matmul5 core:** $557,056 + 16,384 + 128 = 573,568$ bytes $\approx 560$ KB. This is the tightest L1 budget in the model. The weight shard alone consumes over one-third of the typical $\sim 1.5$ MB L1 per core.

### TP=2 Weight Distribution

For TP=2, the full $W_{o\_proj}$ has shape $(16384, 7168)$. Each device holds the rows corresponding to its 64 heads:

$$W_{o\_proj}^{(d)} \in \mathbb{R}^{8192 \times 7168}$$

Since the TP split is along the *input* dimension (heads), each device's matmul5 computes a partial sum of the output projection. The full result requires summing across TP devices -- this is the purpose of the CCL all-reduce.

---

## 5.4.8 Stage 5: Gather3 -- 112 Cores to Single Core

All 112 o_proj cores send their 2-tile results to the gather core:

```
Gather3 topology:
  Senders: 112 active matmul5 cores
  Receiver: core (12, 9)
  Transport: NOC_0, BRISC on receiver

Per-sender:
  Source CB: matmul5_out_cb (CB 6)
  Tiles: matmul5_out_w_per_core = 2 tiles
  Data size: 2 tiles x 64 bytes = 128 bytes
  Write offset = sender_idx * 128

Receiver:
  Dest CB: gather3_dst_cb (CB 7)
  Total tiles: 112 x 2 = 224 tiles
  Total size: 224 x 64 = 14,336 bytes = 14 KB
  Semaphore: gather3_noc0_receiver_semaphore (ID 4)
  Wait condition: noc0_num_senders = 112

  After completion, signals gather3_completion_semaphore (ID 6)
  to notify the CCL sender core that local data is ready.
```

The `gather3_completion_semaphore` (ID 6) is a critical handoff point: BRISC on the receiver core $(12, 9)$ increments the semaphore on the sender core $(11, 9)$, which is waiting in its NCRISC reader for this signal before beginning the fabric transmission.

---

## 5.4.9 Stage 6: CCL All-Reduce -- Cross-Device Summation

On multi-device configurations, each device produces a partial $[1, 7168]$ vector. CCL all-reduce sums these across the mesh and optionally adds the residual connection.

### Two-Core Architecture

| Core | Logical | Role |
|------|---------|------|
| CCL Receiver | $(12, 9)$ | Already holds local data; receives remote data; performs reduction |
| CCL Sender | $(11, 9)$ | Reads local data from receiver's CB; sends via fabric to neighbor |

### Sender Pipeline (Core 11, 9)

**NCRISC (reader):**
1. Waits on `gather3_completion_semaphore_id` (ID 6) -- ensures gather3 has finished writing.
2. Reads 224 tiles of $1 \times 32$ from the receiver core's CB 7 via NOC (using `ccl_sender_data_noc_x/y`).
3. Writes to CB 8 (`ccl_sender_in_cb`).

**BRISC (writer):**
1. Builds a fabric packet header in CB 13 (`ccl_packet_header_cb`).
2. Sends the payload (`ccl_payload_size_bytes = 14,336` bytes) to the neighbor device's receiver core.
3. Destination specified by `ccl_sender_link` (0 for first chip, 1 for second chip) and `ccl_sender_dst_num_hops = 1`.

### Receiver Pipeline (Core 12, 9)

**NCRISC (reader):**
1. Waits for the remote packet to arrive at `ccl_receiver_semaphore_addr` (global semaphore).
2. Writes received data to CB 9 (`ccl_remote_data_cb`), backed by `intermediate_tensor`.
3. The `ccl_receiver_skip_local_push = 1` flag avoids re-reading local data -- it is already in CB 7 from Gather3.
4. If residual is enabled (`has_residual = 1`), reads the residual tensor into CB 10 (`ccl_residual_cb`).

**TRISC (compute):**

Performs element-wise reduction across all 224 tiles:

$$y[i] = \text{local}[i] + \text{remote}[i] + \text{residual}[i]$$

Using CBs:
- CB 7 (`gather3_dst_cb`): Local data
- CB 9 (`ccl_remote_data_cb`): Remote data from fabric
- CB 10 (`ccl_residual_cb`): Residual (optional, when `has_residual = 1`)
- CB 11 (`ccl_temp_cb`): Scratch for 3-input computation
- CB 12 (`ccl_output_cb`): Final output, backed by `output_tensor`

### Ring Topology for TP=2

For a mesh with `cluster_axis = 0`, the two TP peers form a ring:

- **Device 0** (`ring_index = 0`, `is_first_chip = True`):
  - Sender link 0 (forward), sends to Device 1
  - Receiver link 1 (backward), receives from Device 1
  - Sender semaphore: `ccl_semaphore1_addr`
  - Receiver semaphore: `ccl_semaphore2_addr`

- **Device 1** (`ring_index = 1`, `is_first_chip = False`):
  - Sender link 1 (backward), sends to Device 0
  - Receiver link 0 (forward), receives from Device 0
  - Sender semaphore: `ccl_semaphore2_addr`
  - Receiver semaphore: `ccl_semaphore1_addr`

The semaphore swap ensures that both devices use different semaphores for sending vs. receiving, preventing deadlocks.

### Residual Addition

The residual tensor $x_{residual}$ is the original hidden state from the beginning of the attention block (before RMSNorm1). It is added during the all-reduce, not as a separate op:

$$y = \sum_d y_{local}^{(d)} + x_{residual}$$

This fusion eliminates a separate eltwise-add dispatch. When `has_residual = 0`, the residual path is skipped and the reduction is a simple 2-input sum.

---

## 5.4.10 End-to-End NOC Data Flow Diagram

```
  Complete Post-SDPA Pipeline on a Single Device:

  SDPA output cores (8 S1 cores)
       |
       |  Scatter: 8 cores -> 64 kv_b2 cores
       |  [8 x (8, 512)] scattered to form [64 x (1, 512)] activation
       v
  +------------------------------------------------------+
  |  MATMUL4: 64 kv_b2 cores                            |
  |  [1, 512] x [512, 128] = [1, 128] per core          |
  |  K=16 tiles, W=4 tiles output                        |
  +------+-----------------------------------------------+
         |
         |  GATHER2: 64 cores --NOC0--> Core (12,9)
         |  Concatenate: 64 x 4 tiles = 256 tiles
         v
  +--------------------------+
  |  Core (12, 9)            |
  |  [1, 8192] = 256 tiles   |
  +------+-------------------+
         |
         |  MCAST3: Core (12,9) --NOC1--> 13x10 grid
         |  Broadcast 16,384 bytes to 130 cores
         v
  +------------------------------------------------------+
  |  MATMUL5: 112 o_proj cores                           |
  |  [1, 8192] x [8192, 64] = [1, 64] per core          |
  |  K=256 tiles, W=2 tiles output                       |
  +------+-----------------------------------------------+
         |
         |  GATHER3: 112 cores --NOC0--> Core (12,9)
         |  Concatenate: 112 x 2 tiles = 224 tiles
         v
  +--------------------------+
  |  Core (12, 9)            |         Core (11, 9)
  |  [1, 7168] = 224 tiles   |<--SEM--|  CCL Sender
  |  Gather3 receiver +      |  ID 6  |  Reads via NOC0
  |  CCL Receiver             |        |  Sends via Fabric
  +------+-------------------+        +------+-----------+
         |                                    |
         |  CCL: receive remote               |  CCL: send local
         |  data from partner device          |  data to partner
         |                                    |
         v                                    v
  +------------------------------------------------------+
  |  TRISC REDUCE on Core (12, 9):                       |
  |  output = local_data + remote_data [+ residual]      |
  |  224 tiles of 1x32, result in CB 12                  |
  |  Final: [1, 7168] attention output                   |
  +------------------------------------------------------+
```

---

## 5.4.11 Weight Layout: OverlappedTensor Buffer Sharing

The post-SDPA weights participate in two fused buffers that pack multiple sub-tensors per core to simplify weight management:

### kv_b12 Fused Buffer

The `KVB12_PROJ_SingleDeviceOverlapSpec` fuses $W_{kv\_b1\_proj}$ and $W_{kv\_b2\_proj}$:

| Sub-Tensor | Shape | Format | Core Grid | Per-Core Shard | Shard Bytes |
|-----------|-------|--------|-----------|----------------|-------------|
| $W_{kv\_b1\_proj}$ | $(8192, 512)$ | BFP8 | $8 \times 8$ (64 cores) | $(128, 512) = 64$ tiles | 69,632 B |
| $W_{kv\_b2\_proj}$ | $(8192, 512)$ | BFP8 | $5 \times 8 + 12 \times 2$ (64 cores) | $(128, 512) = 64$ tiles | 69,632 B |

The two sub-tensors reside on disjoint core sets (kv_b1 on the Qnope grid, kv_b2 on the remaining 64 cores), so they share the same fused buffer address without physical overlap.

### o_proj Fused Buffer

The `O_PROJ_GATE_MM_RMSNORM_GAMMA_SingleDeviceOverlapSpec` fuses $W_{o\_proj}$, the MoE gate matrix, and four gamma vectors:

| Sub-Tensor | Shape | Format | Core Grid | Per-Core Shard | Shard Bytes |
|-----------|-------|--------|-----------|----------------|-------------|
| $W_{o\_proj}$ | $(8192, 7168)$ | BFP8 | 112 cores | $(8192, 64) = 512$ tiles | 557,056 B |
| $W_{\text{gate\_mm}}$ | $(7168, 256)$ | BF16 | 8 cores (col 12, rows 0-7) | $(7168, 32) = 224$ tiles | 458,752 B |
| $\gamma_{\text{attn\_norm}}$ | $(1, 7168)$ | BF16 | core $(12, 9)$ | $(1, 7168)$ | 14,336 B |
| $\gamma_{q\_norm}$ | $(1, 1536)$ | BF16 | core $(12, 9)$ | $(1, 1536)$ | 3,072 B |
| $\gamma_{\text{kv\_norm}}$ | $(1, 512)$ | BF16 | core $(0, 8)$ | $(1, 512)$ | 1,024 B |
| $\gamma_{\text{ffn\_norm}}$ | $(1, 7168)$ | BF16 | core $(12, 9)$ | $(1, 7168)$ | 14,336 B |

---

## 5.4.12 CB Overlap with SDPA KV Cache Buffer

When `sdpa_kv_cache_buffer` is available (the typical fused case), several post_sdpa temporary CBs are overlapped into the KV cache's L1 allocation to reduce L1 pressure:

```
  SDPA KV Cache Buffer Overlap Layout (per core):

  +--------------------------------------------------+
  |  Offset 0:    matmul4_out (CB 2)                  |
  |               4 tiles x 64 bytes = 256 bytes      |
  +--------------------------------------------------+
  |  Offset 256:  matmul5_in0 (CB 4)                  |
  |               256 tiles x 64 bytes = 16,384 bytes  |
  +--------------------------------------------------+
  |  Offset 16640: matmul5_out (CB 6)                 |
  |               2 tiles x 64 bytes = 128 bytes      |
  +--------------------------------------------------+
  |  Offset 16768: ccl_sender_in (CB 8)               |
  |               224 tiles x 64 bytes = 14,336 bytes  |
  +--------------------------------------------------+
```

This overlap is safe because these CBs are used in strictly sequential stages: CB 2 is written in Stage 1 and read in Stage 2, CB 4 is written in Stage 3 and read in Stage 4, CB 6 is written in Stage 4 and read in Stage 5, and CB 8 is written in Stage 5 and read in Stage 6. No two overlapping CBs are live simultaneously.

---

## 5.4.13 Complete CB Layout for post_sdpa

| CB | Index | Tile | Core Grid | Size (tiles) | Purpose |
|----|-------|------|-----------|-------------|---------|
| `matmul4_in0` | 0 | $1 \times 32$ | kv_b2 (64 cores) | 16 | SDPA output per head |
| `matmul4_in1` | 1 | $32 \times 32$ BFP8 | kv_b2 (64 cores) | 64 (sharded) | kv_b2 weights |
| `matmul4_out` | 2 | $1 \times 32$ | kv_b2 (64 cores) | 4 | Per-head decompressed output |
| `gather2_dst` | 3 | $1 \times 32$ | (12,9) | 256 | Concatenated heads $[1, 8192]$ |
| `matmul5_in0` | 4 | $1 \times 32$ | 13x10 (130 cores) | 256 | Mcast'd activation |
| `matmul5_in1` | 5 | $32 \times 32$ BFP8 | o_proj (112 cores) | 512 (sharded) | o_proj weights |
| `matmul5_out` | 6 | $1 \times 32$ | o_proj (112 cores) | 2 | Per-core o_proj output |
| `gather3_dst` | 7 | $1 \times 32$ | (12,9) | 224 | Full hidden state $[1, 7168]$ |
| `ccl_sender_in` | 8 | $1 \times 32$ | (11,9) | 224 | CCL sender staging |
| `ccl_remote_data` | 9 | $1 \times 32$ | (12,9) | 224 | Received remote data |
| `ccl_residual` | 10 | $1 \times 32$ | (12,9) | 224 | Residual connection |
| `ccl_temp` | 11 | $1 \times 32$ | (12,9) | 224 | Compute scratch |
| `ccl_output` | 12 | $1 \times 32$ | (12,9) | 224 | Final reduced output |
| `ccl_packet_header` | 13 | Raw | (11,9)+(12,9) | varies | Fabric packet headers |

When SDPA reduce-to-all is fused in, 11 additional CBs (indices 14-24) are added for the SDPA worker and forwarder buffers.

---

## 5.4.14 Semaphore Budget

The full post_sdpa program (with SDPA and CCL) uses:

| ID | Name | Signaler | Waiter |
|----|------|----------|--------|
| 0 | `gather2_noc0_receiver_semaphore` | 64 kv_b2 cores (NCRISC) | Core (12,9) BRISC |
| 1 | `gather2_noc1_receiver_semaphore` | Reserved (noc1 path, unused) | -- |
| 2 | `mcast3_data_sender_semaphore` | Core (12,9) BRISC | All 130 mcast3 cores |
| 3 | `mcast3_data_receiver_semaphore` | Mcast3 receiver cores | Core (12,9) BRISC |
| 4 | `gather3_noc0_receiver_semaphore` | 112 o_proj cores (NCRISC) | Core (12,9) BRISC |
| 5 | `gather3_noc1_receiver_semaphore` | Reserved (unused) | -- |
| 6 | `gather3_completion_semaphore` | Core (12,9) BRISC | Core (11,9) NCRISC |
| 7 | `scatter_arrival_semaphore` | SDPA scatter | Matmul4 cores |
| 8-11 | `sdpa_fwd_r1/r2_sem`, `sdpa_bwd_r1/r2_sem` | SDPA forwarder slot management | Forwarder cores |
| Global | `ccl_semaphore1`, `ccl_semaphore2` | CCL fabric endpoint coordination | Inter-device sync |
| Global | `sdpa_semaphore1`, `sdpa_semaphore2` | SDPA fabric endpoint coordination | Inter-device sync |

The critical synchronization chain is:

$$\text{Matmul4} \xrightarrow{\text{sem 0}} \text{Gather2} \xrightarrow{\text{CB 3}} \text{Mcast3} \xrightarrow{\text{sem 2/3}} \text{Matmul5} \xrightarrow{\text{sem 4}} \text{Gather3} \xrightarrow{\text{sem 6}} \text{CCL Send/Recv} \to \text{Output}$$

---

## 5.4.15 Data Volume Summary

Tracking bytes through the post-SDPA pipeline for a single decode step:

| Stage | Data | Bytes | Cores | Bandwidth Pattern |
|-------|------|-------|-------|-------------------|
| SDPA scatter | $[8, 512] \to 64 \times [1, 512]$ | 65,536 | 8 -> 64 | Unicast (row extraction) |
| Matmul4 compute | $[1, 512] \times [512, 128]$ | Read: 70,656 per core | 64 | Local compute |
| Gather2 | $64 \times [1, 128] \to [1, 8192]$ | 16,384 total | 64 -> 1 | NOC_0 unicast |
| Mcast3 | $[1, 8192]$ to 130 cores | 16,384 x 130 = 2.13 MB | 1 -> 130 | NOC multicast |
| Matmul5 compute | $[1, 8192] \times [8192, 64]$ | Read: 573,440 per core | 112 | Local compute |
| Gather3 | $112 \times [1, 64] \to [1, 7168]$ | 14,336 total | 112 -> 1 | NOC_0 unicast |
| CCL All-Reduce | $[1, 7168]$ ring exchange | 14,336 per direction | 2 cores | Fabric links |

The entire post-SDPA path is **compute-bound and NOC-bound**, not DRAM-bound. All weights are pre-loaded in L1 via OverlappedTensors, with zero per-step DRAM weight reads. This contrasts sharply with Flash MLA (Section 5.3), which is DRAM-bandwidth-bound.

The critical path latency is dominated by Matmul5 (256 K-loop iterations over 512 weight tiles per core) and the CCL fabric latency for the all-reduce.

---

## 5.4.16 L1 Budget by Core Type

| Core Type | Count | Weights | Act. CBs | Total L1 | Notes |
|-----------|------:|--------:|---------:|---------:|-------|
| M5-only core | 64 | 544 KB | 16.5 KB | ~561 KB | o_proj weight shard |
| M45 overlap core | 48 | 68 + 544 KB | 17 KB | ~629 KB | Both weight shards |
| M4-only core | 15 | 68 KB | 1.3 KB | ~69 KB | kv_b2 weight shard |
| CCL sender (11,9) | 1 | 68 KB | 15 KB | ~83 KB | M4 core + CCL sender CB |
| Gather/CCL receiver (12,9) | 1 | 0 | ~133 KB | ~133 KB | All gather + CCL CBs + gammas |
| Mcast-only (12,8) | 1 | 0 | 16 KB | ~16 KB | Receives mcast, no matmul |

**Maximum per-core L1:** The M45 overlap cores reach $\sim 629$ KB, representing about $42\%$ of a typical $\sim 1.5$ MB L1. These cores are the tightest in the system when all fused ops are active.

---

## 5.4.17 Fused SDPA ReduceToAll Integration

In production, `SDPAReduceToAll` is fused into the `post_sdpa` kernel rather than running as a separate dispatch. The integration points:

1. **SDPA worker cores** (8 cores) overlap with matmul4 input cores. The SDPA output scatter writes directly to CB 0 (`matmul4_in0`), providing the matmul4 input without an intermediate tensor.

2. **SDPA forwarder cores** (2 cores) are added to the unified kernel's core set via `full_grid.merge(sdpa_forwarder_grid)`.

3. **Scatter arrival semaphore** (ID 7): Matmul4 cores wait on this semaphore before beginning their matmul, ensuring the scatter from SDPAReduceToAll has completed.

4. **Timing diagram** for the fused pipeline:

```
SDPAReduceToAll:  [Exchange L/M/S] [Reduce] [Scatter to matmul4 input]
                                              |
Matmul4:                                      [Wait sem 7] [Compute]
                                                              |
Gather2:                                                      [Collect]
                                                                |
Mcast3:                                                         [Broadcast]
                                                                   |
Matmul5:                                                           [Compute]
                                                                      |
Gather3:                                                              [Collect]
                                                                         |
CCL All-Reduce:                                                          [Exchange + Reduce]
```

---

**Prev:** [Flash MLA Decode](03_flash_mla_decode.md) | **Next:** [Chapter 6 -- Mixture-of-Experts Deep Dive](../ch06_mixture_of_experts_deep_dive/index.md)
