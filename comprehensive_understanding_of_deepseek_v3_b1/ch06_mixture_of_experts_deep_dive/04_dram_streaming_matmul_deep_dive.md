# 6.4 DRAM Streaming Matmul Deep Dive

This section dissects the DRAM streaming matmul -- the micro-op that makes MoE computation possible on Tenstorrent hardware. We trace the data from DRAM banks through the NOC to compute cores, examining the column-major tile reordering, triple-buffering scheme, page size optimization, indexed expert access, the weight upload pipeline, and the complete set of optional features.

> **Cross-reference:** Chapter 3 (Section 3.1) catalogs the `dram_streaming_matmul` micro-op interface. Section 6.2 shows how this op is used three times in the routed expert pipeline (gate_proj, up_proj, down_proj). Chapter 4 (Section 4.3) provides the overlap infrastructure context.

---

## 6.4.1 The Problem: Expert Weights That Cannot Fit in L1

Each routed expert has three weight matrices. Taking gate_proj as an example:

$$W_{\text{gate}} \in \mathbb{R}^{7168 \times 2048}$$

At `bfloat4_b` precision, each 32x32 tile occupies 576 bytes. The total weight size per expert projection:

$$\frac{7168}{32} \times \frac{2048}{32} \times 576 = 224 \times 64 \times 576 = 8{,}257{,}536 \text{ bytes} \approx 7.9 \text{ MB}$$

With 256 experts and 3 projections, the total weight footprint is approximately $256 \times 3 \times 7.9 \approx 6.1$ GB per MoE layer. Even a single expert's projection (~7.9 MB) exceeds the L1 SRAM capacity of a Tensix core (~1.5 MB usable).

The DRAM streaming matmul solves this by keeping input A in L1 (it is small: $[1, 7168]$ or $[1, 2048]$) and streaming weight tiles from DRAM during computation. The key insight is that with $M = 1$ (single-token decode), we need only one row of the output, so we can process one N-column at a time, streaming only the K tiles needed for that column.

---

## 6.4.2 Architecture Overview

```
  L1 (replicated)          DRAM (WIDTH_SHARDED)        L1 (sharded)
  +-----------+           +------------------+        +-----------+
  | input_a   |           | input_b          |        | output    |
  | [1, K]    |           | [K, N]           |        | [1, N/C]  |
  | all cores |           | C banks/shards   |        | per core  |
  +-----+-----+           +--------+---------+        +-----+-----+
        |                          |                        ^
        v                          v                        |
    CB 0 (in0)              CB 1 (in1)                 CB 4 (out)
    tensor-backed           triple-buffered            tensor-backed
    [K tiles]               [subblock_k x 3 tiles]     [N/C tiles]
        |                          |                        ^
        |                          |                        |
        |       +------- NCRISC ---+                        |
        |       |  NOC_0 DRAM reads:                        |
        |       |    for each N column:                     |
        |       |      for each K subblock:                 |
        |       |        async_read(dram_addr, cb1_wr)      |
        |       |        push_back(cb1, subblock_k)         |
        |       |                                           |
        +-------+------ TRISC ---------+--------------------+
                |  Compute loop:       |
                |    for n in 0..per_core_N:
                |      for k_sub in 0..num_subblocks_k:
                |        wait(cb1, subblock_k)
                |        matmul_tiles(cb0, cb1, dst)
                |        pop(cb1, subblock_k)
                |      pack(dst -> cb4)
```

Each of the $C$ compute cores (typically 8, one per DRAM bank) processes $N/C$ output columns. Within each core, the entire K dimension of input_a is available in CB 0, and K-tile sticks of input_b are streamed from DRAM into CB 1 via triple-buffering.

The three RISC processors execute different parts of this pipeline concurrently:
- **NCRISC:** DRAM reads (NOC operations), expert index lookup, scalar source setup
- **TRISC:** Matmul tile operations, optional SiLU activation, optional element-wise multiply
- **BRISC:** Scalar broadcast (when element-wise scalar multiply is enabled)

---

## 6.4.3 Column-Major Tile Reordering

The `_shuffle_dram_tiles()` algorithm transposes each per-bank shard from row-major $[K_t, N_t]$ to column-major $[N_t, K_t]$ tile order, making all K tiles for a given output column contiguous in physical memory. See [Chapter 3 Section 3.2.5.2](../ch03_the_micro_op_library/02_data_movement_micro_ops.md) for the full algorithm derivation and permutation formula.

### Concrete Example

For gate_proj with $K = 7168$, $N = 2048$, 8 DRAM banks, and tile size 32:

- $K_t = 7168 / 32 = 224$ tiles
- $\text{per\_N} = 2048 / 8 = 256$
- $\text{per\_N\_tiles} = 256 / 32 = 8$ tiles
- Tiles per shard: $224 \times 8 = 1792$

Before shuffle: tile 0 is $(K_0, N_0)$, tile 1 is $(K_0, N_1)$, ..., tile 8 is $(K_1, N_0)$.

After shuffle: tile 0 is $(K_0, N_0)$, tile 1 is $(K_1, N_0)$, ..., tile 224 is $(K_0, N_1)$.

The first 224 tiles now form a contiguous K-stick for output column $N_0$. The kernel can read all 224 tiles (or `subblock_k` tiles at a time) as a sequential DRAM burst.

### Impact on Performance

Without column-major reordering, each K tile for a given N column would require a separate DRAM read at a different row offset. With sequential access, the DRAM controller can prefetch and pipeline reads. The difference is substantial -- column-major layout enables approximately 90% DRAM bandwidth utilization vs. 30--40% for scattered row-major reads.

---

## 6.4.4 Triple-Buffered DRAM Reads

The NCRISC reader uses triple buffering (3 buffer slots in CB 1) to hide DRAM read latency behind compute. See [Chapter 3 Section 3.2.5.3](../ch03_the_micro_op_library/02_data_movement_micro_ops.md) for the buffering mechanism and pipeline timing.

For the MoE projections, the concrete CB 1 sizes are:
- **gate_proj/up_proj:** $56 \times 3 = 168$ tiles ($\times 576$ bytes/tile = ~97 KB)
- **down_proj:** $32 \times 3 = 96$ tiles ($\times 576$ bytes/tile = ~55 KB)

The NOC supports at most 15 concurrent outstanding transactions (enforced at `moe_routed_expert/op.py`, line 208), well above the 3-buffer requirement.

---

## 6.4.5 The `subblock_k` Parameter

When the full K dimension is too large to hold in a triple buffer, the K dimension is split into subblocks:

$$\text{subblock\_k} = K_t / \text{num\_subblocks\_k}$$

Each subblock represents a partial K accumulation. The compute loop processes one subblock at a time:

```
for n in 0..per_core_N:
    zero_accumulator(dst)
    for k_sub in 0..num_subblocks_k:
        wait(cb1, subblock_k tiles)
        for k in 0..subblock_k:
            matmul_tiles(cb0[k_sub*subblock_k + k], cb1[k], dst[n])
        pop(cb1, subblock_k)
    pack(dst[n] -> cb_out)
```

**Values used in the MoE pipeline:**

| Projection | $K_t$ | `num_subblocks_k` | `subblock_k` | CB1 Size (tiles) | CB1 Size (bytes) |
|-----------|--------|-------------------|-------------|-----------------|-----------------|
| gate_proj | 224 ($7168/32$) | 4 | 56 | $56 \times 3 = 168$ | ~97 KB |
| up_proj | 224 | 4 | 56 | $56 \times 3 = 168$ | ~97 KB |
| down_proj | 64 ($2048/32$) | 2 | 32 | $32 \times 3 = 96$ | ~55 KB |

The constraint is $K_t \bmod \text{num\_subblocks\_k} = 0$ (asserted at line 198, `dram_streaming_matmul/op.py`).

The choice of `num_subblocks_k` balances three concerns:
1. **DRAM burst efficiency:** Larger `subblock_k` means longer contiguous reads, better utilizing DRAM row buffer locality.
2. **CB1 memory:** CB1 holds $\text{subblock\_k} \times 3$ tiles. Larger `subblock_k` requires more L1.
3. **Accumulation precision:** Intermediate partial products are accumulated in destination registers.

---

## 6.4.6 The `subblock_w` Parameter

Within each N output tile computation, the destination register file is used to accumulate partial results. The `subblock_w` parameter controls how many N tiles are accumulated before packing to the output CB:

```python
# dram_streaming_matmul/op.py, lines 204-219
if dst_full_sync_en and fp32_dest_acc_en:
    max_dest = 8
elif dst_full_sync_en and not fp32_dest_acc_en:
    max_dest = 16
elif not dst_full_sync_en and fp32_dest_acc_en:
    max_dest = 4
elif not dst_full_sync_en and not fp32_dest_acc_en:
    max_dest = 8
max_subblock_w = min(max_dest, per_core_N)

subblock_w = max_subblock_w
while subblock_w > 1 and per_core_N % subblock_w != 0:
    subblock_w -= 1
```

The logic finds the largest `subblock_w` that:
1. Does not exceed the available dest registers (constrained by FP32 mode and sync mode)
2. Evenly divides `per_core_N`

For the MoE pipeline with `fp32_dest_acc_en=False` and `dst_full_sync_en=False`, `max_dest = 8`. Since `per_core_N=8` for gate_proj, `subblock_w=8` -- the entire per-core output fits in the accumulator registers, enabling full-width accumulation without partial writes.

---

## 6.4.7 Page Size Optimization

The `get_max_page_size_and_num_pages` function (`dram_streaming_matmul/op.py`, lines 45--76) calculates optimal NOC burst sizes for DRAM reads: 8192 bytes max on Wormhole, 16384 bytes max on Blackhole. See [Chapter 3 Section 3.2.5.4](../ch03_the_micro_op_library/02_data_movement_micro_ops.md) for the full function and algorithm.

### Worked Example: Wormhole

For gate_proj with bfloat4_b tiles: `tile_size = 576` bytes, `subblock_k = 56`:
- Total: $56 \times 576 = 32{,}256$ bytes
- Initial `page_size = \lfloor 8192 / 576 \rfloor \times 576 = 14 \times 576 = 8064$
- Check: $32{,}256 \bmod 8064 = 0$? $32{,}256 / 8064 = 4$. Yes.
- Result: `page_size = 8064`, `num_pages = 4`

Each K-subblock DRAM read is broken into 4 NOC transactions of 8064 bytes each.

### Worked Example: Blackhole

For gate_proj with bfloat4_b tiles on Blackhole:
- `noc_max_page_size = 16384`
- Initial `page_size = \lfloor 16384 / 576 \rfloor \times 576 = 28 \times 576 = 16{,}128`
- Check: $32{,}256 \bmod 16{,}128 = 0$? $32{,}256 / 16{,}128 = 2$. Yes.
- Result: `page_size = 16128`, `num_pages = 2`

On Blackhole, only 2 NOC transactions per subblock instead of 4 on Wormhole -- a 2x reduction in NOC transaction overhead.

---

## 6.4.8 Indexed Expert Access

When `enable_indexing=True`, the DRAM streaming matmul reads weights for a specific expert rather than a fixed location. The mechanism:

1. **Index tensor:** A $[1, 16]$ tile (HEIGHT_SHARDED, one per DRAM core) contains the expert index at position 0. This is received via mcast from the gate.

2. **DRAM offset calculation:** The kernel reads the expert index and computes:

$$\text{dram\_offset} = \text{expert\_idx} \times K_t \times \text{tile\_size}$$

This offset is added to the base DRAM address before each column read.

3. **Compile-time args for indexing** (`dram_streaming_matmul/op.py`, lines 445--448):

```python
("dram_mm_enable_indexing", 1 if enable_indexing else 0),
("dram_mm_cb_index", cb_id_index if enable_indexing else 0),
("dram_mm_index_offset", 0),  # Configurable offset into index tensor
```

In multi-device mode, `dram_mm_index_offset` is set to `mesh_chip_id` (from `moe/op.py`, line 1530). This means device 0 reads `top8_indices[0]`, device 1 reads `top8_indices[1]`, and so on.

### How Expert Indexing Works with Separate Tensors

In the fused MoE kernel, each expert is stored as a **separate tensor** with its own buffer address. The first expert's tensor is passed as the weight tensor to the matmul setup. At runtime, the kernel uses the expert index to compute an address offset relative to this base tensor:

```
expert_address = base_address + expert_idx * expert_stride
```

where `expert_stride = K_tiles * tile_size` for the shard on each DRAM bank. Since all expert tensors are uploaded contiguously via the same upload loop in `blitz_decode_weights.py` (lines 1398--1418), this arithmetic maps directly to the correct expert's weights. All expert tensors must be kept alive to prevent deallocation.

---

## 6.4.9 Bank ID and Virtual Channel Conflict Resolution

Each DRAM streaming core is assigned a unique bank ID and virtual channel (VC) to avoid NOC contention. The algorithm — optimal bank assignment via `get_optimal_dram_bank_to_logical_worker_assignment()`, followed by VC conflict resolution where same-row cores increment their VC to avoid sharing — is documented in [Chapter 3 Section 3.2.5.9](../ch03_the_micro_op_library/02_data_movement_micro_ops.md).

In the fused MoE context (`moe/op.py`, lines 1302--1323), the same bank_id/vc values are computed once and reused for all three projections (gate_proj, up_proj, down_proj), since they all run on the same 8 DRAM cores. The values are passed as per-core compile-time arguments `dram_mm_bank_id` and `dram_mm_vc`.

---

## 6.4.10 Optional Fused Operations

The DRAM streaming matmul supports three optional fused operations, all controlled by compile-time flags.

### Fused SiLU Activation

When `fuse_silu=1`, SiLU is applied to each output tile after K accumulation is complete:

$$\text{out}[n] = \text{SiLU}\Bigl(\sum_k A[k] \times B[k, n]\Bigr)$$

This is used for gate_proj in the MoE pipeline. The TRISC compile-time arg `dram_mm_fuse_silu` controls this. Fusion eliminates a separate SiLU kernel dispatch, saving one full pass over the output tensor.

### Element-Wise Multiply with Gate Scores

When `enable_mul=True`, after the matmul (and optional SiLU), the result is element-wise multiplied with a `mul_tensor` and optionally a scalar:

$$\text{final}[n] = \text{matmul\_out}[n] \times \text{mul\_tensor}[n] \times \text{scalar}$$

This mode uses CB aliasing to view the $1 \times 32$ matmul output as $16 \times 16$ tiles for the multiply operation. The intermediate matmul output goes to `cb_id_mm_out` (CB 8), which is then aliased as `cb_id_mul_in0` (CB 7) with $16 \times 16$ tile format.

**CB layout with mul enabled:**

| CB | Index | Tile Format | Contents |
|----|-------|-------------|----------|
| `cb_id_mm_out` | 8 | $1 \times 32$ | Matmul intermediate (tensor-backed) |
| `cb_id_mul_in0` | 7 | $16 \times 16$ | Same memory as CB8, viewed as 16x16 |
| `cb_id_mul_in1` | 6 | $16 \times 16$ | mul_tensor viewed as 16x16 |
| `cb_id_out` | 4 | $16 \times 16$ | Final output (tensor-backed) |
| `cb_id_scalar` | 9 | $16 \times 16$ | Scalar working buffer (BRISC writes here) |
| `cb_id_scalar_src` | 10 | $1 \times 16$ | Scalar source (mcasted expert scale) |

The scalar multiply uses FP32 destination accumulation (`dram_mm_mul_fp32_dest_acc_en = 1`, line 495) for numerical precision, even when the matmul itself may use BF16 accumulation.

### Multi-Iteration Looping

When `num_loop_iters > 1`, the kernel repeats the entire streaming matmul computation multiple times. This is primarily a testing feature for verifying CB boundary wrapping. A `working_buf_tensor` must be provided to give the kernel a known CB 1 base address for address wrapping.

---

## 6.4.11 Weight Upload Pipeline

The weight upload pipeline transforms PyTorch tensors into DRAM-resident, column-major-reordered `ttnn.Tensor` objects. This is implemented in `get_tt_moe_routed_expert_weights()` (lines 1339--1421 of `blitz_decode_weights.py`).

### The End-to-End Flow

```
PyTorch weights (num_experts, K, N)
    |
    v
[Pad N to align with num_banks * tile_w]
    |
    v
[_shuffle_dram_tiles(): reorder from row-major to column-major]
    |
    v
[Reshape to (1, 1, K, N_padded)]
    |
    v
[ttnn.from_torch(): convert to bfloat4_b, TILE_LAYOUT, WIDTH_SHARDED in DRAM]
    |
    v
Device-resident ttnn.Tensor (one per expert per projection)
```

**Padding:** $N_{\text{padded}} = \lceil N / (\text{num\_banks} \times \text{tile\_w}) \rceil \times (\text{num\_banks} \times \text{tile\_w})$. This ensures that $N_{\text{padded}}$ is evenly divisible by $\text{num\_banks} \times 32$, so each bank receives an integer number of tiles.

**Shard specification:**

```python
shard_spec = ttnn.ShardSpec(dram_grid, [K, per_core_N], ShardOrientation.ROW_MAJOR)
mem_config = ttnn.MemoryConfig(WIDTH_SHARDED, BufferType.DRAM, shard_spec)
```

**Replication:** The `ReplicateTensorToMesh` mapper ensures every device in the $4 \times 2$ mesh has an identical copy of all 256 experts. Each device independently selects which expert to compute based on its `mesh_chip_id`.

**Memory management:** All 256 expert tensors per projection must be kept alive (not garbage collected) for the duration of inference. The first tensor's buffer address serves as the base for indexed access; the others' addresses are computed as offsets.

**Progress logging:** The upload logs progress every 32 experts (line 1417):

```python
if (i + 1) % 32 == 0:
    logger.info(f"  Uploaded {i + 1}/{num_experts} experts")
```

With 256 experts and 3 projections, the total upload involves 768 tensor conversions and DRAM writes -- a significant initialization cost.

---

## 6.4.12 The Complete DRAM Read Sequence

Tracing the full DRAM read sequence for one core processing gate_proj with `per_core_N = 8` output columns and `num_subblocks_k = 4`:

```
NCRISC execution on one DRAM matmul core:

# Setup phase
read expert_idx from cb_index (if indexing enabled)
compute expert_offset = expert_idx * Kt * tile_size

for n in 0..7:                    # per_core_N output columns
    column_base = n * Kt * tile_size + expert_offset

    for k_sub in 0..3:            # num_subblocks_k
        subblock_base = column_base + k_sub * subblock_k * tile_size

        # Issue async DRAM read: subblock_k tiles
        for page in 0..num_pages-1:
            noc_async_read(
                src = dram_bank[bank_id] + subblock_base + page * page_size,
                dst = cb1_write_ptr + page * page_size,
                size = page_size,
                vc = vc
            )
        noc_async_read_barrier()

        # Signal compute: subblock_k tiles ready
        cb_push_back(cb1, subblock_k)

        # (TRISC is computing on a previously-pushed buffer)
        # (Triple buffering: next push goes to next slot)
```

The triple buffering is implicit in the CB: with `total_size = 3 * subblock_k * tile_size`, the write pointer wraps around after three pushes, overwriting the buffer that TRISC has already consumed.

---

## 6.4.13 The Complete Compute Sequence

TRISC's computation loop mirrors the NCRISC read sequence:

```
TRISC execution on one DRAM matmul core:

for n in 0..per_core_N-1:
    # Initialize accumulator for this output column

    for k_sub in 0..num_subblocks_k-1:
        # Wait for NCRISC to deliver subblock_k tiles
        cb_wait_front(cb1, subblock_k)

        for k in 0..subblock_k-1:
            # in0: input_a[k_sub * subblock_k + k]  (from CB 0)
            # in1: weight[k]                         (from CB 1 read head)
            matmul_tiles(cb0, k_sub * subblock_k + k,
                        cb1, k,
                        dst, n % subblock_w)

        cb_pop_front(cb1, subblock_k)

    # Optional: fused SiLU
    if fuse_silu:
        apply_silu(dst, n % subblock_w)

    # Pack result to output CB
    pack_tile(dst, n % subblock_w, cb_out)
```

The `subblock_w` parameter controls how many N tiles share the dst register file before being packed. When `subblock_w = 8` and `per_core_N = 8`, the entire row of outputs can accumulate in dst before a single pack phase.

---

## 6.4.14 The Unified Kernel Descriptor

The DRAM streaming matmul uses the `UnifiedKernelDescriptor` pattern (lines 502--533 of `dram_streaming_matmul/op.py`), which packages the three-RISC kernel into a single program descriptor:

```python
unified_kernel = UnifiedKernelDescriptor(
    kernel_source=KERNEL_PATH,
    core_ranges=compute_cores,
    ncrisc_named_compile_time_args=[...],
    brisc_named_compile_time_args=[...],
    trisc_named_compile_time_args=[...],
    trisc_compute_config=ttnn.ComputeConfigDescriptor(...),
    unified_compile_time_core_descriptors=[...],
    per_core_compile_time_descriptors=[...],
)
```

The `unified_compile_time_core_descriptors` field sets `dram_mm_is_active_core = 1` for compute cores and `0` for all others. The `per_core_compile_time_descriptors` field carries the per-core `dram_mm_bank_id` and `dram_mm_vc` values.

### IO Tensor Management

The `ttnn.generic_op()` call receives a variable-length list of IO tensors (lines 545--555):

```python
io_tensors = [input_a, input_b, output_tensor]
if working_buf_tensor is not None: io_tensors.append(working_buf_tensor)
if enable_indexing: io_tensors.append(index_tensor)
if enable_mul: io_tensors.append(mul_tensor); io_tensors.append(mm_out_tensor)
if enable_scalar_mul: io_tensors.append(scalar_tensor)
```

All tensors that back CBs must be included in the IO tensor list to prevent the runtime from deallocating them during execution.

---

## 6.4.15 DRAM Streaming in the Fused MoE Context

In the fused MoE operation, the DRAM streaming matmul is not invoked as a standalone op. Instead, its parameters are pre-computed during context setup and embedded into the fused kernel's compile-time arguments:

```python
# gate_proj DRAM matmul args embedded in fused kernel
("gate_proj_cb_in1", gate_proj_cb_in1),
("gate_proj_in1_tensor_addr", gate_proj_params["in1_tensor_addr"]),
("gate_proj_in1_page_size", gate_proj_params["in1_page_size"]),
("gate_proj_subblock_k", gate_proj_params["subblock_k"]),
("gate_proj_per_core_n", gate_proj_params["per_core_n"]),
("gate_proj_in1_block_size_bytes", gate_proj_params["in1_block_size_bytes"]),
("gate_proj_num_subblocks_k", gate_proj_params["num_subblocks_k"]),
("gate_proj_index_offset", mesh_chip_id),
```

The same pattern repeats for `up_proj` and `down_proj`. The DRAM streaming logic is compiled into the fused kernel's NCRISC/TRISC processors, executing as one phase of the larger fused operation.

The fused MoE op also handles the working buffer CB differently: in the fused context, the working buffer configuration is managed by the `_overlap_cbs_with_sdpa_buffer` mechanism, which reuses SDPA intermediate buffers as DRAM streaming working buffers to maximize L1 utilization (see Chapter 4 for the overlap infrastructure).

---

## 6.4.16 Performance Characteristics Summary

| Parameter | gate_proj/up_proj | down_proj |
|---|---|---|
| K dimension | 7168 | 2048 |
| K tiles | 224 | 64 |
| num_subblocks_k | 4 | 2 |
| subblock_k | 56 tiles | 32 tiles |
| per_core_N | 8 tiles | 28 tiles |
| subblock_w | 8 | 7 |
| Triple-buffer tiles | 168 | 96 |
| Triple-buffer bytes | ~97 KB | ~55 KB |
| NOC pages per subblock (WH) | 4 | depends on alignment |
| NOC pages per subblock (BH) | 2 | depends on alignment |
| Fused activation | SiLU (gate), None (up) | None |
| Expert indexing | Yes | Yes |
| Scalar multiply | Yes (in fused mul) | No |
| Expert weight size (bf4_b) | ~7.9 MB | ~7.9 MB |
| DRAM data per core per matmul | ~1 MB | varies |
| DRAM data across 8 cores | ~8 MB | varies |

The gate_proj and up_proj share identical streaming parameters (same weight dimensions) but differ in activation: gate_proj fuses SiLU, up_proj does not. The down_proj has smaller K (2048 vs 7168) but larger per_core_N (28 vs 8 tiles), reflecting the transposed dimension relationship.

### Memory Bandwidth

For gate_proj on one core:
- Data per column: $K_t \times \text{tile\_size} = 224 \times 576 = 128{,}896$ bytes ($\approx 126$ KB)
- Total per core: $\text{per\_core\_N} \times 128{,}896 = 8 \times 128{,}896 = 1{,}031{,}168$ bytes ($\approx 1$ MB)
- Total across 8 cores: $\approx 8$ MB

With DRAM bandwidth of $\sim$12 GB/s per bank on Wormhole, 1 MB per bank takes $\sim$83 microseconds. Triple buffering hides the latency of individual reads but cannot exceed the raw bandwidth. For the $M = 1$ decode case, the matmul is heavily memory-bound -- compute is fast (a few cycles per tile_mul), but DRAM reads take hundreds of cycles per page.

---

## 6.4.17 Complete Data Path: One Core, One N Column

```
                  DRAM Bank c
         +-------------------------+
         |  Expert e, Column n:    |
         |  [K=0] [K=1] ... [K=55]| <- subblock 0 (56 tiles, contiguous)
         |  [K=56]...[K=111]       | <- subblock 1
         |  [K=112]...[K=167]      | <- subblock 2
         |  [K=168]...[K=223]      | <- subblock 3
         +--------+----------------+
                  |
                  |  NCRISC: async_read (4 pages per subblock on WH)
                  |          triple-buffered into CB 1
                  v
         +------------------------+
         |  CB 1 (L1)             |
         |  [buf0: 56 tiles]      | <- NCRISC writes here
         |  [buf1: 56 tiles]      | <- while TRISC reads here
         |  [buf2: 56 tiles]      | <- and read of buf2 in flight
         +--------+---------------+
                  |
                  |  TRISC: matmul_tiles(CB0[k], CB1[k], dst)
                  |         for k in 0..55, then next subblock
                  v
         +------------------------+         +------------------------+
         |  CB 0 (L1, input_a)    |         |  After all 4 subblocks |
         |  [K=0] [K=1] ... [Kt]  |  --->   |  accumulated:          |
         |  Full input, replicated |         |  optional SiLU, pack   |
         +------------------------+         +----------+-------------+
                                                       |
                                                       v
                                            +------------------------+
                                            |  CB 4 (L1, output)     |
                                            |  [out_n]               |
                                            +------------------------+
```

This pattern repeats 8 times in parallel (one per DRAM bank core), producing the complete $[1, N_{\text{padded}}]$ output distributed across the 8 cores. The output CB accumulates all N tiles for this core, which are then available for the next pipeline stage (multiply, gather, or direct output).

---

[Previous: 6.3 Shared Expert and Combination](03_shared_expert_and_combination.md) | [Next: Chapter 7](../ch07/index.md)
