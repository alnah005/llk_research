# 06.03 -- Compute Ops

[<< Data Movement Ops](02_data_movement_ops.md) | [Next: Attention Ops >>](04_attention_ops.md)

Compute ops perform the mathematical operations -- matmuls, normalizations,
activations, element-wise operations. Each is a MicroOp backed by a C++ kernel
that runs on TRISC (the tensor compute RISC). This section covers the key
compute ops and traces how they compose into larger building blocks.


## Matmul: L1-Resident Weight Multiply

**Source**: `blaze/ops/matmul/op.py`

The basic matmul: activations from a CB times weights sharded in L1.

### Interface

```python
Matmul.emit(
    f: FusedProgram,
    in0,                  # CBHandle or Tensor (activations)
    in1,                  # Tensor (weights, WIDTH_SHARDED in L1)
    *,
    prefix: str,
    pop_in0: bool = True,
) -> CBHandle            # output on weight-shard cores
```

### Mechanics

1. Resolves `in0` to a CBHandle (activation input, same on all cores via Mcast)
2. Resolves `in1` weight tensor via `resolve_weight_direct_address()` -- the
   weight tensor lives in L1, sharded across compute cores
3. Computes `out_w_per_core = weight_pages / K_tiles` (output width per core)
4. Allocates output scratch CB on the active core ranges (matching weight sharding)
5. TRISC performs the matmul: `out[1, out_w] = act[1, K] x weight[K, out_w]`

### Core Mapping

Each active core computes one slice of the output. The active grid equals the
weight tensor's shard grid -- if weights are sharded across 8 cores, the matmul
produces 8 partial outputs.

```
Mcast activations        Weight shards (L1)       Output shards
  [1, K] on all   x   [K, out_w/8] per core  =  [1, out_w/8] per core
      cores                 8 cores                  8 cores
```

### Where Matmul Appears

- **PostSdpa**: `kv_b2_matmul` (per-head latent projection) and `o_proj_matmul`
  (output projection)
- **PreSDPA**: `matmul2` and `matmul3` (Q projection stages)
- **GlmFusedProj**: 4 parallel matmuls on disjoint core ranges

The `pop_in0` flag controls whether the activation CB is freed after this
matmul. When `False`, the same activation can feed multiple parallel matmuls
(fan-out pattern).


## DRAMStreamingMatmul: Expert Weight Multiply

**Source**: `blaze/ops/dram_streaming_matmul/op.py`

The workhorse for MoE: streams expert weights from DRAM, indexed by expert ID.

### Interface

```python
DRAMStreamingMatmul.emit(
    f: FusedProgram,
    act: CBHandle | Tensor,
    weights: Tensor,                 # DRAM-resident, bank-sharded
    index: CBHandle | Tensor | None, # expert index
    out: Tensor | None,
    *,
    prefix: str,
    fp32_dest_acc_en: bool,
    subblock_k: int | None,
    fused_activation: str | None,    # None, "silu", "clamped_silu_gate", "clamped_silu_up"
    activation_params: dict | None,  # {limit, alpha} for clamped activations
    index_offset: int,
    wait_for_out: bool,
    pop_index: bool,
    pop_act: bool,
) -> CBHandle
```

### Mechanics

1. Compute cores are assigned 1:1 to DRAM banks via
   `device.get_optimal_dram_bank_to_logical_worker_assignment()`
2. Each core's NCRISC reads weight tiles from its bank, using the expert `index`
   to select which expert's weight page to stream
3. Weights are triple-buffered in a scratch CB to overlap DRAM reads with compute
4. TRISC computes matmul in subblocks of K, with optional fused activation

### Index Offset for Multi-Device

In multi-device MoE, each device handles a subset of experts. The `index_offset`
adjusts expert indices so device 0 uses offset 0, device 1 uses offset 1, etc.:

```python
# From MoE.emit():
row, col = f.mesh_coord
rows, cols = f.mesh_shape
index_offset = row * cols + col + routed_index_offset
```

### Fused Activation Modes

DRAMStreamingMatmul supports four activation modes applied in-place after the
matmul accumulation:

| Mode | Constant | Use |
|------|----------|-----|
| None | `_FUSED_ACT_NONE` | Up-projection (no activation) |
| SiLU | `_FUSED_ACT_SILU` | Gate-projection in standard SwiGLU |
| Clamped gate | `_FUSED_ACT_CLAMPED_GATE` | Gate with clamped SiLU (alpha, limit) |
| Clamped up | `_FUSED_ACT_CLAMPED_UP` | Up with clamped SiLU (limit only) |

This eliminates a separate activation dispatch in the SwiGLU pipeline.


## MatmulFusedAct: Gate MM with Optional Sigmoid

**Source**: `blaze/ops/matmul_fused_act/op.py`

Used specifically for the MoE gate matmul. Computes `act x gate_weight` with
optional sigmoid activation. The output has face-tile format (16x16) for the
downstream gate selection kernel.

```python
# From MoERouter.emit():
gate_mm_handle = MatmulFusedAct.emit(f, act, weight,
    out_tensor=None,
    prefix=BlazeOp.child_prefix(prefix, "gate_mm"),
    use_silu=False, pop_act=pop_act)
```

The output is sharded across gate MM cores (one per expert group). A
subsequent Gather collects all gate logits to the sender core for routing.


## MatmulCB: Shared Output CB Matmul

**Source**: `blaze/ops/matmul_cb/op.py`

Matrix multiply that writes to a caller-provided output CB instead of allocating a
fresh scratch CB. Two MatmulCBs sharing one output CB produce contiguous pages --
this is how Q absorption `[1,512]` and Q_rope `[1,64]` are concatenated into
Q `[1,576]` without a host readback.

```python
MatmulCB.emit(f, in0, in1, *, out_cb, prefix, pop_in0=True) -> CBHandle
```

Sets `op_class = "Matmul"` to reuse the Matmul C++ kernel struct. Used in GLM Q
branch (absorption + rope concatenation into shared Q CB).


## RMSNorm: Root Mean Square Normalization

**Source**: `blaze/ops/rmsnorm/op.py`

### Interface

```python
RMSNorm.emit(
    f: FusedProgram,
    input_handle: CBHandle | Tensor,
    gamma_tensor: Tensor,
    *,
    prefix: str,
    cores: CoreRangeSet,
    epsilon: float = 1e-6,
    scalar: float = 1.0,
    pop_input: bool = True,
    notify_input: bool | None = None,
) -> CBHandle
```

### Mechanics

1. For row-tile inputs (height 1), auto-reinterprets to 32x32 tiles for compute
2. Auto-computes `scalar = 1/sqrt(width)` when input has row tiles
3. TRISC computes: `output = (input * scalar) * rsqrt(mean(x^2) + eps) * gamma`
4. Returns scratch CB on the specified cores

### Row Tile Reinterpretation

When the activation tensor uses 1x32 row tiles (common for 1-token decode), the
CB is aliased with a wider tile geometry (16x32 or 32x32). This lets the TRISC
LLK compute the norm reduction efficiently while the physical bytes stay the
same:

```python
if has_row_tiles(input_handle):
    width = input_handle.shape[1]
    itile, n_tiles = interpret_tile(width)  # e.g., 7168 -> Tile(16,32), 14 tiles
    scalar = 1.0 / math.sqrt(float(width))
    inp = f.cb_from_tensor(input_handle, tile=itile, page_size=page_sz)
```

### Where RMSNorm Appears

Every major FusedOp starts with RMSNorm:
- MoE/DenseMLP preamble: `emit_mlp_preamble()` -> RMSNorm -> Mcast
- PreSDPA: BroadcastRMSNorm -> RMSNorm
- PreSDPA Q path: `rmsnorm2` (second normalization for Q_a -> Q_b)
- KV branch: `kv_norm` (KV latent normalization)


## PaddedRMSNorm: Non-Tile-Aligned Normalization

**Source**: `blaze/ops/padded_rmsnorm/op.py`

RMS normalization for non-tile-aligned widths. Separated from RMSNorm so the CB
composition is fixed (5 CBs: input, gamma, padded_input, padded_gamma, output).
NCRISC zero-fills padded scratch CBs and copies original data; TRISC zeros garbage
padding rows in the last tile after squaring. Used by models with non-power-of-2
hidden dimensions.


## ClampedSilu: Clamped Activation

**Source**: `blaze/ops/clamped_silu/op.py`

Element-wise clamped activation for GPT-OSS clamped SwiGLU. Two modes:
- **gate**: `clamp(x, max=limit) * sigmoid(alpha * clamp(x, max=limit))`
- **up**: `clamp(x, -limit, limit) + 1.0`

Key kwargs: `mode` ("gate"/"up"), `alpha` (default 1.702), `limit` (default 7.0).
Also available as a fused activation mode in DRAMStreamingMatmul.


## EltwiseMul: Element-Wise Multiply

**Source**: `blaze/ops/eltwise_mul/op.py`

Element-wise multiplication of two CB inputs with optional scalar from a third CB.

```python
EltwiseMul.emit(
    f, up_out_handle, gate_out_handle, scalar,
    prefix=BlazeOp.child_prefix(prefix, "mul"),
    cores=compute_cores,
    in0_wait_cb=up_out_handle, in0_wait_tiles=out_num_tiles,
    pop_in0=True, pop_in1=True, pop_scalar=True)
```

The `in0_wait_cb` and `in0_wait_tiles` mechanism lets EltwiseMul wait for a
specific number of tiles in an upstream CB before starting -- this synchronizes
with the asynchronous DRAMStreamingMatmul that may still be pushing tiles.

Used in SwiGLU: `(x @ w_up) * silu(x @ w_gate)` where `silu()` is fused into
the gate matmul and the scalar carries the routing score.


## ResidualAdd: Element-Wise Addition

**Source**: `blaze/ops/residual_add/op.py`

Element-wise residual addition: `out = in0 + in1`. Supports optional `skip_add`
passthrough mode. Per-core `core_idx` enables correct residual indexing when `in1`
spans a wider grid than the active range. Used in DownProj residual connection and
MoE shared+routed merge.


## Rope: Rotary Position Embedding

**Source**: `blaze/ops/rope/op.py`

Applies rotary position encoding using position-dependent cos/sin values read from
DRAM. Allocates 5 internal CBs (input, cos, sin, output, scratch). For multi-row
inputs, `Ht` parameter controls the number of tile rows to process. Supports
pre-allocated CB sharing when the upstream op already produced the input CB.
Used in PreSDPA Q path and KV branch.


## SwiGLU: The Routed Expert MLP

**Source**: `blaze/ops/swiglu/op.py`

SwiGLU is the FusedOp that implements the core MLP for each routed expert.
It chains six MicroOps:

```
SwiGLU Composition Tree
├── DRAMStreamingMatmul (up_proj)^     -- x @ W_up
├── DRAMStreamingMatmul (gate_proj)^   -- silu(x @ W_gate)
├── EltwiseMul^                        -- up * gate (* score)
├── Gather^                            -- collect to sender
├── Mcast (post_gather_mcast)^         -- fan out to down cores
└── DRAMStreamingMatmul (down_proj)^   -- result @ W_down
```

### Data Flow Trace

```
act (CBHandle, all cores)
  |
  +----> DRAMStreamingMatmul(up_proj, pop_act=False)
  |        fused_activation=None or "clamped_silu_up"
  |        -> up_out_handle (per-DRAM-bank cores)
  |
  +----> DRAMStreamingMatmul(gate_proj, pop_act=pop_act)
  |        fused_activation="silu" or "clamped_silu_gate"
  |        -> gate_out_handle (same cores)
  |
  v
EltwiseMul(up_out, gate_out, scalar=routing_score)
  -> eltwise_out_handle (same cores)
  |
  v
Gather(eltwise_out -> sender core)
  -> gathered_out_handle (sender core, N*pages concatenated)
  |
  v
Mcast(gathered_out -> all compute cores)
  -> down_act_handle (all cores)
  |
  v
DRAMStreamingMatmul(down_proj)
  -> down_out_handle (per-DRAM-bank cores)
```

### Key Design Decisions

1. **Up and gate share activation**: `pop_act=False` on up_proj keeps the
   activation CB live for gate_proj. Gate proj consumes and optionally pops it.

2. **Index not popped until down_proj**: The expert `index` CB feeds all three
   DRAMStreamingMatmuls. `pop_index=False` on up and gate; `pop_index=True`
   on down_proj frees it after the final use.

3. **Gather + Mcast between stages**: After the element-wise multiply, partial
   results (width-sharded across DRAM-bank cores) must be gathered and re-
   distributed for the down projection. This is the Gather->Mcast pattern that
   appears everywhere in the library.

4. **Score fusion**: The `scalar` CB (from the MoE router) is consumed by
   EltwiseMul, multiplying each element by the routing score. This fuses the
   score scaling into the element-wise multiply rather than requiring a separate
   dispatch.


## KNMatmul: Shared Expert K-N Split Matmul

**Source**: `blaze/ops/kn_sliced_matmul/op.py`

KNMatmul is the L1-resident matmul variant for the shared expert. Unlike the
DRAM-streaming version (which indexes into DRAM weights), KNMatmul works with
weights that are fully resident in L1. It supports K-parallel (splitting the
reduction dimension across cores) and N-parallel (splitting the output width)
configurations.

Used in SharedExpert:

```python
# From SharedExpert.emit():
gu = KNMatmul.emit(f, activation, gate_up_weights,
    prefix=BlazeOp.child_prefix(prefix, "gu"),
    ab_coords=ab_coords, k_half_split=False,
    pop_act=pop_shared_act)
```

Returns a named tuple with `handle`, `k_parallel`, `n_parallel`, `a_grid`,
`b_grid`, `gate_handle`, `up_handle` -- providing grid metadata needed by
the downstream GatedReduce.


## GatedReduce: SiLU + Element-Wise + Reduction

**Source**: `blaze/ops/gated_reduce/op.py`

After KNMatmul produces gate and up partial products, GatedReduce:
1. Applies SiLU to the gate output
2. Performs element-wise multiply of gate * up
3. Reduces across K-parallel shards

This is the shared expert's equivalent of the SwiGLU EltwiseMul, but
optimized for L1-resident partial sums that need K-dimension reduction.


## GatedLocalReduce: Single-Core Gated Reduction

**Source**: `blaze/ops/gated_local_reduce/op.py`

Same math as GatedReduce (`SiLU(sum(group1)) * sum(group2)`) but operates entirely
on one core without multi-core gather. Used in dense MLP variants where gate/up
partial products are co-located on the same core.


## Layout Conversion

### Retilize

**Source**: `blaze/ops/retilize/op.py`

Convert (1,32) row tiles to (N,32) standard tiles using `tilize_block` LLK.
Primary use: convert per-head Q rows `[8x18 tiles of (1,32)]` into `[18 tiles of
(8,32)]` for FlashDecode consumption. Key kwargs: `num_tile_cols`, `num_tile_rows`,
`out_tile` (e.g., `Tile(8,32)` for Q), `skip_input_init`.

### Untilize

**Source**: `blaze/ops/untilize/op.py`

Reverse of Retilize -- convert (N,32) TILE_LAYOUT tiles to (1,32) ROW_MAJOR pages
using `pack_untilize` LLK. Output has `N * num_tile_cols` pages of contiguous row
data. Used post-SDPA before ScatterRaw distribution.

### TileRowConvert

**Source**: `blaze/ops/tile_row_convert/op.py`

Convert TopK position IDs to unique tile-row IDs via bitmask deduplication. Reads
position IDs, divides by 32, deduplicates via bitmask, writes unique tile-row IDs.
Runs on NCRISC on the coordinator core. Used in DSA pipeline.


## Sampling: Argmax

**Source**: `blaze/ops/argmax/op.py`

Find the index of the maximum bfloat16 score across cores and across devices.
Uses a two-stage multi-device reduction: (1) non-target-row devices send local
winners to target row via fabric, (2) target-row non-target-column sends to the
final target device. Each core performs local argmax, then a gather + semaphore
pattern collects per-core winners. Cross-device reduction uses fabric with
link-balanced routing. Used in LMHeadSampling.


## SharedExpert: The Complete Shared MLP

**Source**: `blaze/ops/shared_expert/op.py`

SharedExpert chains four stages into a single FusedOp:

```
SharedExpert Composition Tree
├── KNMatmul (gu)^       -- gate_up = act @ W_gate_up
├── GatedReduce^         -- reduced = silu(gate) * up, reduced across K
├── Mcast^               -- fan out reduced to down_proj cores
└── DownProj
    ├── Mcast (mcast2)^  -- optional second fan-out
    ├── ResidualAdd^     -- add pre-norm residual
    └── Gather^          -- collect to output
```

### Data Flow Trace

```python
# From SharedExpert.emit():
gu = KNMatmul.emit(f, activation, gate_up_weights, ...)
# gu.handle: partial products on AB grid

reduced = GatedReduce.emit(f, gu.handle, ...,
    k_parallel=gu.k_parallel, n_parallel=gu.n_parallel,
    gate_handle=gu.gate_handle, up_handle=gu.up_handle)
# reduced: single CBHandle with silu(gate)*up

down_act = Mcast.emit(f, reduced, ...)
# down_act: on all cores

return DownProj.emit(f, down_act, down_weights, bias, output, ...)
# DownProj performs the down projection, residual add, and gather
```

### Residual Connection

The `bias` input to SharedExpert is actually the pre-normalization activation.
DownProj adds it back as a residual: `output = (reduced @ W_down) + x_prenorm`.
This fuses the residual connection into the DownProj stage rather than requiring
a separate ResidualAdd dispatch.

In MoE with shared experts, the `skip_bias_add` flag is set for non-root
devices (since only the root device should add the residual after ReduceToOne).


## DenseMLP: Putting It Together Without Routing

**Source**: `blaze/ops/dense_mlp/op.py`

DenseMLP is the simplest complete MLP FusedOp -- no routing, no expert
selection. It is the shared expert path with a preamble:

```
DenseMLP Composition Tree
├── RMSNorm^              -- normalize activation
├── Mcast (act_mcast)^    -- broadcast to compute cores
└── SharedExpert
    ├── KNMatmul (gu)^
    ├── GatedReduce^
    ├── Mcast^
    └── DownProj
        ├── Mcast^
        ├── ResidualAdd^
        └── Gather^
```

12 MicroOps total. Used by models without MoE routing (Llama 3.1 8B, Qwen3 32B).

### DenseMLPDram Variant

For models that need DRAM-streamed weights (larger MLP dimensions), DenseMLPDram
uses DenseSwiGLU instead of SharedExpert:

```
DenseMLPDram Composition Tree
├── RMSNorm^
├── Mcast (act_mcast)^
└── DenseSwiGLU
    ├── DRAMStreamingMatmul (up)^
    ├── DRAMStreamingMatmul (gate)^
    ├── EltwiseMul^
    ├── Gather^
    ├── Mcast^
    └── DRAMStreamingMatmul (down)^
```

8 MicroOps total. No residual built-in (caller adds externally).


## Additional Fused Compositions

### DownProj

**Source**: `blaze/ops/down_proj/op.py`

Down-projection with residual add: `Mcast -> Matmul -> Mcast(bias) -> ResidualAdd
-> Gather`. The standard second-half of a gated MLP. Used by SharedExpert and
DenseMLP.

### MlpMixed

**Source**: `blaze/ops/mlp_mixed/op.py`

Mixed MLP that combines L1-sharded and DRAM-streaming matmul paths. Selects
between KNMatmul (L1) and DRAMStreamingMatmul (DRAM) based on weight size.

### LMHeadSampling

**Source**: `blaze/ops/lm_head_sampling/op.py`

Language model head sampling: `BroadcastRMSNorm -> Mcast -> Matmul -> Argmax`.
Composes the final token-generation pipeline from hidden states to a sampled
token index. Used as the final decode step in all models.

**SparseLayer** and **AllReduceMoE** are layer-level compositions that combine
attention and MoE -- see [Section 04 -- Composite Layer-Level Ops](04_attention_ops.md).

[<< Data Movement Ops](02_data_movement_ops.md) | [Next: Attention Ops >>](04_attention_ops.md)
