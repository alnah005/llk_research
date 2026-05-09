# 6.2 Routed Expert Computation

This section traces the complete routed expert pipeline from the point where expert indices and scores are available on all DRAM matmul cores, through the gate/up/down projections, to the final reduced output. We walk through the data flow step by step, focusing on how each stage hands off to the next.

> **Cross-reference:** Section 6.1 covers how the expert index and scale arrive on the compute cores. Section 6.4 provides a deep dive into the DRAM streaming matmul mechanics. Chapter 4 (Section 4.3.2) provides the stage table and CB map for `moe_routed_expert` as a catalog entry. This section goes deeper into the dataflow mechanics, expert parallelism model, and multi-device reduction.

---

## 6.2.1 The Routed Expert Pipeline at a Glance

The `MoeRoutedExpertOp` class (`moe/op.py`, lines 330--1900) orchestrates a 12-stage pipeline within a single kernel dispatch. The stages are temporally ordered -- each completes before the next begins, which enables extensive semaphore reuse and CB aliasing.

```
 Sender Core (12,9)              8 DRAM-Optimal Compute Cores
 ==================              ============================

 [1] RMSNorm (TRISC)
      |
 [2] Input Mcast ───────────────> [receive activation in gate_mm_input_cb]
      |                            |
 [3] <─── Gate Gather ─────────── Gate Matmul + Sigmoid (SRAM)
      |                            [1,7168]x[7168,32]-> [1,32] per core
 [4] DeepSeek Gate (TRISC)
      |  top-8 selection
 [5] Index Mcast ──────────────> [receive expert_index in gate_proj_cb_index]
 [6] Scale Mcast ──────────────> [receive expert_scale in mul_cb_scalar_src]
      |                            |
      |                     [7] gate_proj DRAM Streaming Matmul + SiLU
      |                            [1,7168]x[7168,N_e] indexed by expert_idx
      |                            |
      |                     [8] up_proj DRAM Streaming Matmul (no activation)
      |                            [1,7168]x[7168,N_e] indexed by expert_idx
      |                            |
      |                     [9] Fused Mul: SiLU(gate) x up x expert_scale
      |                            16x16 tile reinterpretation
      |                            |
 [10]<─── down_proj Gather ───── [send mul output to sender]
      |
 [11] down_proj Mcast ─────────> [receive fused output in down_proj_mcast_dst_cb]
      |                            |
      |                    [12] down_proj DRAM Streaming Matmul
      |                            [1,N_expert_padded]x[N_expert,K/8]
      |                            |
      |                    [13] Eltwise Add: down_proj + shared_expert_output
      |                            |
      |                    [14] ReduceToOne (optional, 4x2 mesh)
      |                            3-round tree: LEAF->inner, ROOT3->ROOT1/2, ROOT2->ROOT1
```

### Temporal Ordering and CB Reuse

The stages execute strictly sequentially on the same set of cores. This enables two key optimizations:

1. **Semaphore reuse:** `MCAST_SENDER` (ID 0) is reused for stages 2, 5, 6, and 11 because only one mcast is active at a time.

2. **CB aliasing:** `gate_proj_cb_in1` and `up_proj_cb_in1` share the same CB index (line 862 of `moe/op.py`: `up_proj_cb_in1 = gate_proj_cb_in1`). The gate_proj and up_proj DRAM streaming matmuls never overlap, so their working buffers can occupy the same L1 memory.

---

## 6.2.2 Stages 1--5: From Input to Expert Selection

Stages 1--5 (RMSNorm through index/scale mcast) are covered in [Section 6.1](01_moe_gating_and_routing.md). At the end of stage 5, every DRAM matmul core has:
- The full input activation $[1, 7168]$ in its gate_mm_input CB
- The selected expert index in its gate_proj_cb_index CB
- The expert scale in its mul_cb_scalar_src CB

---

## 6.2.3 Expert Weight Layout: 256 Experts in DRAM

Understanding how the 256 experts' weights are physically stored is essential for understanding the indexed DRAM reads.

Each of the 256 routed experts has three projection matrices:

| Projection | Shape per Expert | Storage | Size (bfloat4_b) |
|---|---|---|---|
| `gate_proj` | $(7168, N_\text{padded})$ | WIDTH_SHARDED in DRAM, bfloat4_b | ~7.9 MB |
| `up_proj` | $(7168, N_\text{padded})$ | WIDTH_SHARDED in DRAM, bfloat4_b | ~7.9 MB |
| `down_proj` | $(N_\text{expert}, K_\text{out\_padded})$ | WIDTH_SHARDED in DRAM, bfloat4_b | ~7.9 MB |

where $N_\text{padded}$ is the expert hidden dimension padded to be divisible by $(\text{num\_banks} \times \text{tile\_width})$. The per-expert weight size at bfloat4_b is:

$$\frac{7168}{32} \times \frac{2048}{32} \times 576 = 224 \times 64 \times 576 = 8{,}257{,}536 \text{ bytes} \approx 7.9 \text{ MB}$$

With 256 experts and 3 projections, the total weight footprint is approximately $256 \times 3 \times 7.9 \approx 6.1$ GB per MoE layer.

**Weight preparation** (`blitz_decode_weights.py`, lines 1339--1421): Each expert's projection matrix is uploaded as a separate `WIDTH_SHARDED` DRAM tensor with `bfloat4_b` quantization:

```python
def upload(expert_weights: torch.Tensor) -> list[ttnn.Tensor]:
    num_experts, K, N = expert_weights.shape
    N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    per_core_N = N_padded // num_banks

    shard_spec = ttnn.ShardSpec(dram_grid, [K, per_core_N], ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(WIDTH_SHARDED, BufferType.DRAM, shard_spec)

    for i in range(num_experts):
        w = expert_weights[i]
        if N_padded != N:
            w = torch.nn.functional.pad(w, (0, N_padded - N))
        w_shuffled = self._shuffle_dram_tiles(w.unsqueeze(0), tile_w, num_banks)
        tensors.append(ttnn.from_torch(w_shuffled, dtype=ttnn.bfloat4_b, ...))
```

### Column-Major Tile Reorder

The most critical step in weight preparation is the `_shuffle_dram_tiles` call (`blitz_decode_weights.py`, lines 1556--1604). Standard WIDTH_SHARDED layout stores tiles in row-major order within each shard, but the DRAM-streaming matmul kernel needs K tiles contiguous for each output column (column-major within each shard). The shuffle transposes tile order so that for each output column, all $K_t$ tiles are contiguous in physical DRAM memory:

```
Row-major (default):             Column-major (after shuffle):
  [k=0,n=0] [k=0,n=1] ...        [k=0,n=0] [k=1,n=0] [k=2,n=0] ...
  [k=1,n=0] [k=1,n=1] ...        [k=0,n=1] [k=1,n=1] [k=2,n=1] ...
  [k=2,n=0] [k=2,n=1] ...        ...
```

This is implemented via a permutation index (line 1599):

```python
i = torch.arange(num_tiles_per_shard)
source_idx = (i % K_tiles) * per_N_tiles + (i // K_tiles)
shuffled_tiles = tiles[:, source_idx, :, :]
```

The payoff: the kernel can issue a single contiguous DRAM read of $K_t$ tiles for each output column, maximizing NOC burst efficiency and enabling triple-buffered pipelining (see Section 6.4 for details).

### Expert Tensor Addressing

All 256 expert tensors for a given projection are separate ttnn.Tensors, each with its own buffer address in DRAM. The op receives the first expert's tensor as the base address and computes offsets for other experts using the expert index. The `gate_proj_index_offset` compile-time argument carries the mesh chip ID for multi-device partitioning (`moe/op.py`, line 1530):

```python
("gate_proj_index_offset", mesh_chip_id if ctx.enable_routing else 0),
```

In multi-device mode, device 0 uses `top8_indices[0]`, device 1 uses `top8_indices[1]`, and so on -- all 8 devices process different experts in parallel. All expert tensors must be kept alive to prevent deallocation; the `MoERoutedExpertWeights` dataclass holds the full lists.

---

## 6.2.4 Stage 6: gate_proj -- DRAM Streaming Matmul with SiLU

The gate projection computes:

$$\text{gate\_out}_{[1, N]} = \text{SiLU}\bigl(\text{input}_{[1, 7168]} \times W_{\text{gate\_proj}}^{[7168, N]}\bigr)$$

This is the first DRAM streaming matmul in the pipeline. The setup (`moe/op.py`, lines 1074--1082):

```python
gate_proj_params = MoeRoutedExpertOp.setup_dram_matmul(
    device=device,
    weights_tensor=gate_proj_weights_tensor,
    cb_in1_index=gate_proj_cb_in1,
    cb_out_index=gate_proj_cb_out,
    fp32_dest_acc_en=False,
    num_subblocks_k=4,    # K/4 tiles per subblock for partial accumulation
)
```

Key parameters:

| Parameter | Value | Derivation |
|-----------|-------|------------|
| $K_t$ | $7168 / 32 = 224$ tiles | From weight shard height / tile height |
| `subblock_k` | $224 / 4 = 56$ tiles | $K_t / \text{num\_subblocks\_k}$ |
| `per_core_n` | 8 tiles | $\text{shard\_width} / \text{tile\_width}$ |
| `fuse_silu` | 1 | SiLU activation fused into compute |
| `fp32_dest_acc_en` | False | BFloat16 accumulation |

The SiLU activation is fused into the matmul compute loop -- after accumulating all K subblocks for a given N output tile, SiLU is applied before writing to the output CB. The compile-time argument `gate_proj_fuse_silu=1` (line 1702) enables this.

**Core assignment:** The DRAM matmul runs on the "optimal DRAM bank worker" cores -- 8 cores assigned by `device.get_optimal_dram_bank_to_logical_worker_assignment(NOC_0)`. These are chosen for optimal DRAM bandwidth based on physical proximity to DRAM banks.

---

## 6.2.5 Stage 7: up_proj -- DRAM Streaming Matmul (No Activation)

The up projection computes:

$$\text{up\_out}_{[1, N]} = \text{input}_{[1, 7168]} \times W_{\text{up\_proj}}^{[7168, N]}$$

This uses the same DRAM streaming pattern as gate_proj, with the same cores, same expert indexing, and same subblock configuration. The only difference: no fused activation (`up_proj_fuse_silu=0`, line 1711).

Both gate_proj and up_proj read from the same input CB (`gate_mm_input_cb`) which holds the mcasted activation. They write to different output CBs:
- gate_proj writes to `gate_proj_cb_out`
- up_proj writes to `up_proj_cb_mm_out`

**CB aliasing:** The gate_proj weight CB and up_proj weight CB share the same CB index (`gate_proj_cb_in1 = up_proj_cb_in1` intentionally, line 862 of `moe/op.py`). This works because the two DRAM matmuls execute sequentially, never concurrently. This saves precious L1 memory equal to 168 tiles ($56 \times 3$ for triple-buffering) of weight working buffer.

---

## 6.2.6 Stage 8: Fused Element-Wise Multiply with Scaling

After both projections complete, the element-wise multiply fuses three operations into one pass:

$$\text{fused\_out}_{[1, N]} = \text{SiLU}(\text{gate\_out}) \odot \text{up\_out} \times s_e$$

where the SiLU was already applied in stage 6, and $s_e$ is the normalized routing score for the selected expert. The setup at `moe/op.py` lines 1100--1108:

```python
mul_params = MoeRoutedExpertOp.setup_eltwise_mul(
    cb_in0_index=mul_cb_in0,   # up_proj output aliased as 16x16
    cb_in1_index=mul_cb_in1,   # gate_proj output aliased as 16x16
    cb_out_index=mul_cb_out,   # fused output
    per_core_n=gate_proj_params["per_core_n"],
    cb_scalar_index=mul_cb_scalar,
    cb_scalar_src_index=mul_cb_scalar_src,
)
```

### CB Aliasing for 16x16 Multiply

The matmul outputs use $1 \times 32$ tiles, but the multiply kernel operates on $16 \times 16$ tiles. This is handled through **CB aliasing**: the same physical memory is viewed with different tile formats. The `setup_eltwise_mul` function (`moe_routed_expert/op.py`, lines 67--158) creates CB descriptors that overlay $16 \times 16$ tiles on top of the $1 \times 32$ matmul output memory:

```python
TILE_16x16 = ttnn.Tile((16, 16))
tile_16x16_size = TILE_16x16.get_tile_size(ttnn.bfloat16)

cb_in0_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_in0_index, in0_tensor)
cb_in0_descriptor.total_size = mul_num_tiles * tile_16x16_size
cb_in0_descriptor.format_descriptors[0].tile = tile_16x16_desc
cb_in0_descriptor.format_descriptors[0].page_size = tile_16x16_size
```

The number of $16 \times 16$ tiles is computed from the total element count:

$$\text{mul\_num\_tiles} = \left\lceil \frac{M \times N_\text{per\_core} \times 32}{256} \right\rceil$$

where 256 = 16 x 16. Each core processes its shard independently.

### Scalar Multiply Integration

The expert scale (routing score) is received via the expert scale mcast into `mul_cb_scalar_src`. BRISC reads this scalar value and fills a $16 \times 16$ working tile in `mul_cb_scalar` with the scalar at position [0,0]. TRISC then performs a three-way multiply: `out = in0 * in1 * scalar`. The `mul_scalar_index_offset` (line 1630) selects the correct score for the current device from the top-8 score tensor.

**CB layout for multiply (6 CBs on DRAM cores):**

| CB | Name | Tile | Backed By |
|----|------|------|-----------|
| `mul_cb_in0` | up_proj output aliased as 16x16 | $16 \times 16$ | `up_proj_cb_mm_out` memory |
| `mul_cb_in1` | gate_proj output aliased as 16x16 | $16 \times 16$ | `gate_proj_cb_out` memory |
| `mul_cb_out` | fused output | $16 \times 16$ | dedicated tensor |
| `mul_cb_scalar_src` | mcasted expert scale | $1 \times 16$ | tensor from mcast |
| `mul_cb_scalar` | scalar working buffer | $16 \times 16$ | allocated in L1 |

---

## 6.2.7 Stages 9--10: Gather and Re-Distribute for Down Projection

The fused output from stage 8 is width-sharded across 8 DRAM matmul cores. The down projection needs the full $[1, N_{\text{expert}}]$ vector as its input. This requires a gather-then-mcast pattern.

### Stage 9: down_proj Gather

Each DRAM matmul core sends its portion of the fused output to the sender core:

```
  DRAM Core 0          DRAM Core 1          ...     DRAM Core 7
  [fused_out            [fused_out                   [fused_out
   shard 0]              shard 1]                     shard 7]
     \                     |                            /
      -- NOC0 unicast --- gather to sender (12,9) -----
                           |
                           v
                    Sender Core
              +--------------------+
              | down_proj_gather   |
              | _dst_cb            |
              | [1, N_full]        |
              +--------------------+
```

The gather uses explicit sender indices (`use_explicit_sender_index=True`, line 1129 of `moe/op.py`) because the DRAM bank-optimal cores are scattered across the grid (they are not a contiguous rectangle). Each core's `down_proj_gather_sender_idx` is assigned in the per-core compile-time descriptors (line 1897).

### Stage 10: down_proj Mcast

The gathered result is multicast from the sender core to all 130 cores in the grid (`moe/op.py`, lines 1137--1149):

```python
down_proj_mcast_params = MoeOp.setup_mcast(
    sender_core=sender_core,
    mcast_grid=mcast_grid,
    src_cb=down_proj_gather_dst_cb,
    dst_cb=down_proj_mcast_dst_cb,
    sender_semaphore_addr=mcast_data_sender_semaphore_addr,
    receiver_semaphore_addr=down_proj_mcast_receiver_semaphore_addr,
    data_size_bytes=down_proj_mcast_data_size_bytes,
)
```

This gather-then-mcast pattern is a fundamental motif in the TT architecture: when a WIDTH_SHARDED result must become the replicated input for the next matmul, the sender core acts as a rendezvous point.

---

## 6.2.8 Stage 11: down_proj -- DRAM Streaming Matmul (Final Projection)

The down projection computes:

$$\text{down\_out}_{[1, K']} = \text{fused\_out}_{[1, N_{\text{expert}}]} \times W_{\text{down\_proj}}^{[N_{\text{expert}}, K_{\text{per\_core}}]}$$

This is the third DRAM streaming matmul. Its configuration differs from gate_proj/up_proj:

| Parameter | gate_proj/up_proj | down_proj |
|-----------|-------------------|-----------|
| `num_subblocks_k` | 4 | **2** |
| `subblock_k` | 56 tiles | **32** tiles |
| Input CB | `gate_mm_input_cb` | `down_proj_mcast_dst_cb` |
| Expert index CB | `gate_proj_cb_index` (reused) | `gate_proj_cb_index` (reused) |
| Fused activation | SiLU (gate) / None (up) | None |

The smaller `num_subblocks_k = 2` for down_proj reflects its smaller K dimension ($N_\text{expert} = 2048$, giving $K_t = 64$ tiles). The output is WIDTH_SHARDED across the 8 DRAM cores with each core producing $[1, K_\text{out}/8] = [1, 896]$.

---

## 6.2.9 Stage 12: Eltwise Add -- Combining with Shared Expert Output

The down projection output must be combined with the shared expert's output. This is done via element-wise addition:

$$\text{final\_out}[i] = \text{down\_proj\_out}[i] + \text{shared\_expert\_out}[i]$$

The add uses **32x32 tile aliasing** -- a different tile format from any prior stage. The `setup_eltwise_add` function (`moe/op.py`, lines 489--574) creates CB descriptors:

```python
add_params = MoeRoutedExpertOp.setup_eltwise_add(
    cb_in0_index=add_cb_in0,       # down_proj output (aliased 32x32)
    cb_in1_index=add_cb_in1,       # shared expert output (replicated)
    cb_out_index=add_cb_out,       # final per-device output
    width_per_core=down_proj_width_per_core,
    total_width=down_proj_total_width,
    core_ranges=gate_proj_core_ranges,
)
```

**Per-core indexing:** The shared expert output (`add_cb_in1`) is replicated across all cores as a full $[1, 7168]$ tensor. Each core uses its `add_sender_index` to select the correct $[1, K']$ slice:

$$\text{offset} = \text{sender\_index} \times \text{slice\_size\_bytes}$$

This is set up in `setup_eltwise_add` (`moe/op.py`, line 560):

```python
sender_index_core_values = [(core, idx) for idx, core in enumerate(compute_cores_list)]
```

---

## 6.2.10 The `_MoeRoutedExpertContext` Dataclass

The `_MoeRoutedExpertContext` dataclass (`moe/op.py`, lines 122--261) encapsulates all 80+ fields needed to configure the routed expert pipeline. Understanding its organization is key to navigating the code.

### Field Groups

| Group | Fields | Purpose |
|---|---|---|
| Device and mesh | `device`, `full_device_grid`, `device_grid_size`, `mesh_rows`, `mesh_cols` | Hardware topology |
| Core grids | `sender_core`, `input_core_grid`, `mcast_grid`, `gate_proj_core_ranges`, `num_gate_proj_cores` | Which cores do what |
| Data format | `data_format`, `tile_1x32_size`, `num_tiles_k` | Tile arithmetic |
| CB indices (shared) | `rmsnorm_output_cb`, `gate_mm_input_cb`, `gate_proj_cb_in1/out`, `mul_cb_*`, `down_proj_*`, `add_cb_*` | 16 CB slot assignments |
| Semaphore addresses | `mcast_data_sender_semaphore_addr`, `gather_noc0_receiver_semaphore_addr`, etc. | 4 global semaphore L1 addresses |
| Setup result dicts | `rmsnorm_mcast_params`, `gate_proj_params`, `up_proj_params`, `mul_params`, `down_proj_*_params`, `add_params` | 8 pre-computed parameter dicts |
| Per-core values | `bank_id_core_values`, `vc_core_values`, `sender_idx_core_values` | DRAM bank assignment for each core |
| Routing-only | `gate_mm_*`, `gate_*`, `gate_proj_cb_index`, `mul_cb_scalar*`, `use_hardcoded_expert_index` | 21+ fields, active only when `enable_routing=True` |
| ReduceToOne | `reduce_local_cb`, `reduce_received_cb_r*`, `reduce_output_cb`, `reduce_scratch_cb`, `reduce_packet_*_cb`, `reduce_params` | 11 fields for multi-device reduction |

### Why a Dataclass? Setup/Build Separation

The context pattern serves two purposes:

1. **Setup/build separation:** `_setup_dimensions()` computes everything once (including device queries, NOC coordinate lookups, and buffer address fetches). The `_build_compile_time_args()` and `_build_cb_descriptors()` methods then read from the context without any device communication, enabling fast per-device program construction.

2. **SDPA buffer overlap:** Many CB descriptors are initially `None` in the context and are filled in later by `_overlap_cbs_with_sdpa_buffer()`, which maps MoE intermediate buffers to the SDPA KV-cache memory (idle during MoE execution). The context provides stable CB index assignments that the overlap function can target.

The context is created once during model initialization and reused for every decode step. This amortizes the substantial setup cost (grid construction, semaphore address resolution, CB descriptor creation) across thousands of inference iterations.

---

## 6.2.11 Compile-Time Arg Propagation: From Setup to Kernel

Understanding how the pipeline configuration reaches the hardware kernel is essential for debugging. The compile-time args follow a three-level hierarchy.

### Level 1: Setup Functions

Each stage has a `setup_*` function that computes parameters and returns a dictionary. For example, `MoeRoutedExpertOp.setup_dram_matmul` (`moe/op.py`, line 347) returns:

```python
return {
    "per_core_n": per_core_n,       # N tiles per core
    "Kt": Kt,                       # Total K tiles
    "subblock_k": subblock_k,       # K tiles per subblock
    "subblock_w": subblock_w,       # N tiles per dest batch
    "in1_page_size": in1_page_size, # NOC page size
    "in1_num_pages": in1_num_pages, # Pages per subblock read
    "in1_block_size_bytes": in1_block_size_bytes,
    "in1_tensor_addr": weights_tensor.buffer_address(),
    "in1_buf_addr": in1_buf_addr,   # CB 1 base address
    "cb_in1_descriptor": cb_in1_descriptor,
    "cb_out_descriptor": cb_out_descriptor,
    ...
}
```

### Level 2: Context Dataclass

The setup results are stored as fields of `_MoeRoutedExpertContext`. For example, `ctx.gate_proj_params` holds the gate_proj setup dict.

### Level 3: Compile-Time Arg Lists

The `_build_compile_time_args` method (`moe/op.py`, lines 1443--1758) extracts values from the context and produces three lists of `(name, value)` tuples -- one for NCRISC, one for BRISC, one for TRISC. For gate_proj:

```python
# NCRISC args (lines 1520-1530)
("gate_proj_cb_in1", ctx.gate_proj_cb_in1),
("gate_proj_in1_tensor_addr", ctx.gate_proj_params["in1_tensor_addr"]),
("gate_proj_in1_page_size", ctx.gate_proj_params["in1_page_size"]),
("gate_proj_subblock_k", ctx.gate_proj_params["subblock_k"]),
("gate_proj_per_core_n", ctx.gate_proj_params["per_core_n"]),
("gate_proj_in1_block_size_bytes", ctx.gate_proj_params["in1_block_size_bytes"]),
("gate_proj_out_num_tiles", ctx.gate_proj_params["out_num_tiles"]),
("gate_proj_num_subblocks_k", ctx.gate_proj_params["num_subblocks_k"]),
("gate_proj_index_offset", mesh_chip_id if ctx.enable_routing else 0),

# TRISC args (lines 1693-1703)
("gate_proj_cb_in0", ctx.gate_mm_input_cb),
("gate_proj_cb_in1", ctx.gate_proj_cb_in1),
("gate_proj_cb_out", ctx.gate_proj_cb_out),
("gate_proj_subblock_k", ctx.gate_proj_params["subblock_k"]),
("gate_proj_per_core_n", ctx.gate_proj_params["per_core_n"]),
("gate_proj_subblock_w", ctx.gate_proj_params["subblock_w"]),
("gate_proj_num_subblocks_k", ctx.gate_proj_params["num_subblocks_k"]),
("gate_proj_tile_r_dim", ctx.gate_proj_params["tile_r_dim"]),
("gate_proj_fuse_silu", 1),
("gate_proj_fp32_dest_acc_en", 0),
```

Notice the naming convention: `gate_proj_*` prefix for all gate_proj-related args, `up_proj_*` for up_proj, `down_proj_*` for down_proj. This makes it easy to trace any kernel compile-time arg back to its setup origin.

### Per-Core Descriptors

Some args vary per-core. These use `PerCoreCompileTimeDescriptor` (lines 1866--1907):

```python
PerCoreCompileTimeDescriptor(
    named_compile_time_arg="gate_proj_bank_id",
    core_values=ctx.bank_id_core_values,  # [(core0, bank0), (core1, bank1), ...]
    other_value=0,  # Default for non-DRAM-matmul cores
)
```

Six per-core descriptors cover bank_id/vc for all three DRAM matmuls, plus `down_proj_gather_sender_idx` and `add_sender_index`.

### Unified Core Descriptors

Role flags are set via `UnifiedCompileTimeCoreDescriptor` (lines 1826--1864):

| Flag | Core Range | Value | Purpose |
|------|-----------|-------|---------|
| `is_sender_core` | sender_core | 1 | Enables mcast/gather/gate logic |
| `is_mcast_grid_core` | mcast_grid | 1 | Enables mcast receiver logic |
| `is_gate_mm_core` | gate_mm cores | 1 | Enables SRAM gate matmul |
| `is_gate_proj_core` | gate_proj cores | 1 | Enables DRAM matmul pipeline |
| `is_reduce_worker_core` | gate_proj cores | 1 | Enables ReduceToOne |
| `is_reduce_fabric_core` | fabric cores | 1 | Enables fabric forwarding |

These flags control `if constexpr` branches in the unified kernel, ensuring each core executes only its assigned stages.

---

## 6.2.12 Per-Core Configuration: Bank IDs and Virtual Channels

Each DRAM core needs unique bank ID and virtual channel (VC) compile-time parameters. The `_setup_dimensions` method (`moe/op.py`, lines 1303--1323) computes these using the optimal-bank-to-worker assignment and VC conflict resolution algorithm described in [Section 6.4.9](04_dram_streaming_matmul_deep_dive.md). The same bank_id/vc values are reused for all three projections since they share the same 8-core set.

---

## 6.2.13 ReduceToOne: Multi-Device Summation

When running on a 4x2 mesh (8 devices), each device processes a different expert. The results must be summed across all devices to produce the final output:

$$\mathbf{o}_\text{final} = \sum_{d=0}^{7} \mathbf{o}_\text{device}^{(d)}$$

This is implemented by the ReduceToOne integration, which uses a 3-level tree reduction across the mesh.

### Device Roles

The `get_reduce_device_role()` function (`moe_routed_expert/op.py`, lines 674--691) assigns each device one of four roles:

```python
MESH_LEAF  = 0   # Outer rows (rows 0, 3): send data only
MESH_ROOT3 = 1   # Inner row opposite to ROOT1: intermediate aggregator
MESH_ROOT2 = 2   # Same row as ROOT1, different column: intermediate aggregator
MESH_ROOT1 = 3   # The final destination: receives and sums all contributions
```

The role assignment for a 4x2 mesh (root at row 1, column 0):

```
        Col 0    Col 1
Row 0:  LEAF     LEAF       -- Outer row
Row 1:  ROOT1    ROOT2      -- Root row (ROOT1 is the final destination)
Row 2:  ROOT3    ROOT3      -- Inner row
Row 3:  LEAF     LEAF       -- Outer row
```

### 3-Level Tree Reduction

The reduction proceeds in three rounds:

**Round 1 (LEAFs -> adjacent inner row):** Each LEAF sends to the inner row in its same column. LEAF at row 0 sends to row 1 (ROOT1/ROOT2 row), LEAF at row 3 sends to row 2 (ROOT3 row). The routing logic (`moe_routed_expert/op.py`, lines 1843--1847): row 0 LEAFs use `dest = (row+1, col)`, row 3 LEAFs use `dest = (row-1, col)`. After Round 1: ROOT1(1,0) holds its own + LEAF(0,0); ROOT2(1,1) holds its own + LEAF(0,1); ROOT3(2,0) holds its own + LEAF(3,0); ROOT3(2,1) holds its own + LEAF(3,1).

**Round 2 (ROOT3 -> ROOT1/ROOT2):** Each ROOT3 sends to the root row in its same column (`dest = (root_row, col)`, line 1849). ROOT3(2,0) sends to ROOT1(1,0); ROOT3(2,1) sends to ROOT2(1,1). After Round 2: ROOT1(1,0) holds rows {0,1,2,3} of col 0; ROOT2(1,1) holds rows {0,1,2,3} of col 1.

**Round 3 (ROOT2 -> ROOT1):** ROOT2 sends its column sum to ROOT1 across the column boundary via fabric. ROOT1 performs the final accumulation: $R1 = R1 + R2$.

After all three rounds, ROOT1 holds the sum of all 8 partial results.

### Implementation Details

The reduce uses **fabric-based communication** for cross-device data transfer. Key parameters (computed at lines 1198--1298 of `moe/op.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `payload_size_bytes` | `shard_elements * 2` | bfloat16 payload per worker |
| `packet_header_size` | 96 bytes | Fabric packet header |
| `num_workers` | 8 (DRAM bank cores) | Worker cores performing reduce |

Each worker core handles its own width shard independently -- the reduction is per-shard, so the 8 DRAM cores can reduce their 8 shards in parallel.

### CB Layout for Reduce (8 CBs)

| CB | Variable | Purpose |
|----|----------|---------|
| `reduce_local_cb` | Alias of `add_cb_out` | Local partial result (from eltwise add) |
| `reduce_received_cb_r1` | Dedicated | Round 1 received data (LEAF -> inner row) |
| `reduce_received_cb_r2` | Dedicated | Round 2 received data (ROOT3 -> ROOT1/ROOT2) |
| `reduce_received_cb_r3` | Dedicated | Round 3 received data (ROOT2 -> ROOT1) |
| `reduce_output_cb` | Dedicated | Final reduced output |
| `reduce_scratch_cb` | Dedicated | Scratch for accumulation |
| `reduce_packet_cb` | Dedicated | Packet assembly for fabric send |
| `reduce_packet_header_cb` | Dedicated | Persistent 96-byte packet header |

Note that `reduce_local_cb` is intentionally aliased to `add_cb_out` -- the reduce reads directly from the eltwise add output without an additional copy.

### Synchronization

Four global semaphores coordinate the three rounds plus a final exit signal:

| Semaphore | Source | Purpose |
|-----------|--------|---------|
| `reduce_semaphores[0]` | From global semaphore pool | LEAF -> inner row handshake |
| `reduce_semaphores[1]` | | ROOT3 -> ROOT1/ROOT2 handshake |
| `reduce_semaphores[2]` | | ROOT2 -> ROOT1 handshake |
| `reduce_semaphores[3]` | | Completion signal |

### The Golden Reference

The `golden_reduce()` method (lines 797--815 of `moe_routed_expert/op.py`) provides the simple reference: stack all per-device outputs and sum along the device dimension:

```python
stacked = torch.stack([out for out in per_device_outputs], dim=0)
return torch.sum(stacked, dim=0)
```

---

## 6.2.14 CB Descriptor Construction

The `_build_cb_descriptors` method (`moe/op.py`, lines 1761--1821) assembles the full CB descriptor list from the context. The ordering matters -- CB descriptors are matched by their `buffer_index` field, not their position in the list. The method collects descriptors from:

1. **Pre-built descriptors** (from `_setup_dimensions`): `rmsnorm_output_cb_descriptor`, `gate_mm_input_cb_descriptor`
2. **Routing-only descriptors** (conditional on `ctx.enable_routing`): gate_mm weights, output, bias, indices, scores, output_indices
3. **Shared DRAM matmul descriptors**: `gate_proj_params["cb_in1_descriptor"]` (reused for up_proj)
4. **Mul descriptors**: 3 input/output CBs for element-wise multiply
5. **Routing scalar descriptors** (conditional): `mul_params["cb_scalar_src_descriptor"]`, `mul_params["cb_scalar_descriptor"]`
6. **Down-proj chain**: gather dst, mcast dst, weights, output
7. **Add chain**: 3 CBs for eltwise add
8. **Residual/RMSNorm**: residual mcast src, residual mcast dst, gamma
9. **Reduce CBs** (conditional on `ctx.enable_reduce_to_one`): received CBs r1--r3, output, scratch, packet, header

A typical fused MoE routed expert has 25--35 active CB descriptors.

---

## 6.2.15 The Standalone `moe_routed_expert` vs. Fused `moe`

The codebase contains two implementations of the routed expert pipeline:

1. **Standalone** (`fused_ops/moe_routed_expert/op.py`, 2150 lines): Used for unit testing. Allocates its own tensors, creates local semaphores, and dispatches a single `generic_op`. Uses hardcoded CB indices (0--32).

2. **Fused** (`fused_ops/moe/op.py`, 4442 lines): Used in production. Shares the sender core and mcast grid with the shared expert. Uses `CircularBufferIdManager` for dynamic CB assignment. Supports CB reconfiguration between routed and shared phases.

The key differences:

| Feature | Standalone | Fused |
|---------|-----------|-------|
| CB index assignment | Hardcoded (0--32) | Dynamic (`CircularBufferIdManager`) |
| Semaphores | Local, created per-op | Global (`MoeSem`, 17 semaphores) |
| RMSNorm | External (input pre-normalized) | Integrated (first stage) |
| Shared expert | Not included | Runs in same dispatch |
| CB reconfiguration | Not needed | `build_cb_reconfig_tensor` |
| Tensor allocation | Explicit per-tensor | SDPA buffer overlap |

Both implementations use the same setup functions (`setup_dram_matmul`, `setup_eltwise_mul`, etc.) and produce identical numerical results. The fused version is more complex but eliminates multiple dispatch overheads and enables buffer sharing.

---

## 6.2.16 Complete Data Flow Diagram

```
                         SENDER CORE (12,9)
                    +---------------------+
                    | input [1, 7168]     |
                    +----------+----------+
                               | mcast
                               v
    +----------------------------------------------+
    |              130-core grid                    |
    |                                              |
    |  8 Gate MM    8 DRAM Expert     ...           |
    |  Cores        Cores (bank-optimal)            |
    +----------------------------------------------+
                   | gate_mm -> gather -> gate
                   | -> index/scale mcast
                   v
    +----------------------------------------------+
    |        8 DRAM Expert Cores                   |
    |                                              |
    |  +--- gate_proj ---+  +--- up_proj ---+      |
    |  | DRAM stream      |  | DRAM stream    |    |
    |  | + SiLU           |  | (no act)       |    |
    |  +-------+----------+  +-------+--------+    |
    |          |                     |              |
    |          +---- eltwise_mul ----+              |
    |               x expert_scale                  |
    |                    |                          |
    |          fused_out [1, N] per core            |
    +--------------------+-------------------------+
                         | gather
                         v
                    SENDER CORE
                    [1, N_full]
                         | mcast
                         v
    +----------------------------------------------+
    |        8 DRAM Expert Cores                   |
    |                                              |
    |  +--- down_proj ---------------------+       |
    |  | DRAM stream [1,N_full]x[N,K']     |       |
    |  +---------------+-------------------+       |
    |                  |                           |
    |         +-- eltwise_add --+                  |
    |         | + shared expert |                  |
    |         |   output shard  |                  |
    |         +--------+--------+                  |
    |                  |                           |
    |       final_out [1, K'] per core             |
    +------------------+---------------------------+
                       |
                       | (if 4x2 mesh)
                       v
              +--------------------+
              |   ReduceToOne      |
              |   LEAF -> inner row |
              |   ROOT3 -> ROOT1/2 |
              |   ROOT2 -> ROOT1   |
              |   (sum across 8    |
              |    devices)        |
              +--------------------+
```

---

[Previous: 6.1 MoE Gating and Routing](01_moe_gating_and_routing.md) | [Next: 6.3 Shared Expert and Combination](03_shared_expert_and_combination.md)
