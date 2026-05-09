# 03 -- Layer Taxonomy and Decode Overview

This section describes the 61-layer model structure (3 dense + 58 MoE), the weight dataclasses that define each layer type, the weight fusion system, the pipeline stages of a decode step, the weight preparation pipeline, and the multi-host pipeline architecture. It concludes with a high-level end-to-end decode step walkthrough that serves as an orientation map for subsequent chapters.


## 3.1 Dense Layers vs. MoE Layers

DeepSeek V3's 61 decoder layers (indexed 0--60) are divided into two categories by the constant `FIRST_K_DENSE_REPLACE = 3` (`demo/cli.py`, line 35). Both layer types share the same MLA attention block but diverge only in the FFN portion:

```
                    +------------------+
                    |  RMSNorm (attn)  |
                    |       |          |
                    |  Multi-head      |
                    |  Latent Attention |
                    |  (MLA)           |
                    |       |          |
                    |  CCL All-Reduce  |
                    |  + Residual Add  |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
    +---------+---------+       +-----------+-----------+
    |   Dense Layer      |       |    MoE Layer           |
    |                    |       |                        |
    |  RMSNorm (ffn)     |       |  RMSNorm (ffn)        |
    |       |            |       |       |                |
    |  MLP (gate+up+down)|       |  Gate Network (256)    |
    |       |            |       |       |                |
    |  Residual Add      |       |  8 Routed Experts      |
    |                    |       |  + 1 Shared Expert     |
    |                    |       |       |                |
    |                    |       |  Residual Add          |
    +--------------------+       +------------------------+
```

| Component | Dense Layer (0--2) | MoE Layer (3--60) |
|---|---|---|
| Attention | MLA (identical) | MLA (identical) |
| Gate routing | None (`gate_mm = None`) | `gate_mm` weight + `gate_bias` |
| Routed experts | 1 per device (`ttnn.Tensor`) | 256 (`list[ttnn.Tensor]`) |
| Shared expert | Present | Present |
| Weight dataclass | `DeepSeekV3DenseLayerWeights` | `DeepSeekV3MoELayerWeights` |

The layer type is determined by index in `load_weights_from_cache()` (`demo/cli.py`, lines 69--95):

```python
layer_id = decoder_layer_id_from_mesh_id(mesh_id)
is_moe = layer_id >= FIRST_K_DENSE_REPLACE
```

### 3.1.1 Attention (Shared by Both Layer Types)

The attention block decomposes into two fused ops:

**PreSDPA** fuses the following into one kernel launch:
1. CCL Broadcast (multi-device input distribution)
2. RMSNorm ($\text{attn norm}$, applied to hidden state)
3. Mcast + Matmul ($W_{q\_a}$: $[1, 7168] \rightarrow [1, 1536]$) on 96 cores
4. Gather + RMSNorm ($\text{q norm}$)
5. Mcast + Matmul ($W_{q\_b}$: $[1, 1536] \rightarrow [1, 12288]$)
6. CreateQHeads: split into 64 Qnope ($[1, 128]$ each) and 64 Qrope ($[1, 64]$ each)
7. Matmul3: per-head $W_{kv\_b1}$ ($[1, 128] \times [128, 512] \rightarrow [1, 512]$) on 64 cores
8. RoPE on Qrope heads
9. KV latent projection: $W_{kv\_a}$ ($[1, 7168] \rightarrow [1, 576]$), RMSNorm ($\text{kv norm}$), on 18 cores
10. FlashMLA decode attention
11. KV cache update

**PostSDPA** fuses the output projection and cross-device reduction:
1. Matmul4: $W_{kv\_b2}$ ($[1, 512] \times [512, 128]$) on 64 cores (kv\_b2 grid)
2. Gather to core (12, 9): assemble $[1, 8192]$
3. Mcast to 130 cores
4. Matmul5: $W_o$ ($[1, 8192] \times [8192, 64]$) on 112 cores
5. Gather to core (12, 9): assemble $[1, 7168]$
6. CCL All-Reduce + Residual Add

### 3.1.2 Dense FFN

For dense layers, the FFN is a standard gated MLP implemented via `SharedExpertOp`:

$$
\text{FFN}(\mathbf{x}) = (\text{SiLU}(\mathbf{x} W_\text{gate}) \odot \mathbf{x} W_\text{up}) W_\text{down}
$$

Weight shapes (per device with $\text{moe tp} = 1$):
- $W_\text{gate}$: $(7168, 256)$ -- BFP4, block-sharded across 64 A cores
- $W_\text{up}$: $(7168, 256)$ -- BFP4, block-sharded across 64 B cores
- $W_\text{down}$: $(256, 7168)$ -- width-sharded across 112 cores

Dense layers also have a single routed expert per device alongside the shared expert. Both the "routed" expert (which acts as the dense MLP) and the shared expert execute.

### 3.1.3 MoE FFN

For MoE layers, the `MoeOp` fused op (`fused_ops/moe/op.py`) orchestrates:

**Routed Expert Pipeline** (`MoeRoutedExpertOp`):
1. Input mcast from core (12,9) to grid
2. Gate matmul: $[1, 7168] \times [7168, 256] \rightarrow [1, 256]$ (scores for 256 experts)
3. Gate selection: top-8 with bias correction (`e_score_correction_bias`)
4. Index + scale mcast to 130 cores
5. For each selected expert $e$:
   - DRAM-streaming `gate_proj` matmul: $[1, 7168] \times [7168, N_e]$
   - DRAM-streaming `up_proj` matmul: $[1, 7168] \times [7168, N_e]$
   - Fused SiLU-gated multiply
   - `down_proj` matmul + scaled accumulation
6. Eltwise add of routed result to output

**Shared Expert** (concurrent with routed):
- Same structure as the dense MLP but with smaller per-device shapes
- Gate/Up matmul on 128 compute cores (64 A + 64 B)
- Down-projection on 112 matmul cores
- Residual add

The routed and shared expert computations are composed into a single fused program using `CircularBufferIdManager` for CB allocation across phases and `build_cb_reconfig_tensor` to allow the shared-expert phase to reconfigure CBs left by the routed-expert phase. The op uses 17 global semaphores (`MoeSem.NUM_SEMAPHORES = 17`) to coordinate the many concurrent data flows.


## 3.2 Weight Dataclasses

The `prepare_weights.py` module defines typed dataclasses that package all weights for each layer type. These serve as the contract between weight preparation and kernel execution.

### Dense Layer Weights

```python
@dataclass
class DeepSeekV3DenseLayerWeights:
    # Attention fusion group: q_ab_kv_a
    q_a_proj: OverlappedTensor          # (7168, 1536), BFP8, fused
    q_b_proj: OverlappedTensor          # (1536, 12288), BFP8, fused
    kv_a_proj: OverlappedTensor         # (7168, 576), BFP8, fused

    # Attention fusion group: o_proj_gate_mm_norms
    o_proj: OverlappedTensor            # (8192, 7168), BFP8, fused
    attn_norm: OverlappedTensor         # (1, 7168), BF16, fused
    q_norm: OverlappedTensor            # (1, 1536), BF16, fused
    kv_norm: OverlappedTensor           # (1, 512), BF16, fused
    ffn_norm: OverlappedTensor          # (1, 7168), BF16, fused

    # Attention fusion group: kv_b12
    kv_b1_proj: OverlappedTensor        # (8192, 512), BFP8, fused
    kv_b2_proj: OverlappedTensor        # (512, 8192), BFP8, fused

    # MLP (shared expert structure)
    shared_gate_proj: OverlappedTensor  # (7168, 256), BFP4, fused
    shared_up_proj: OverlappedTensor    # (7168, 256), BFP4, fused
    shared_down_proj: ttnn.Tensor       # (256, 7168), standalone

    # MLP routed (dense has 1 "routed" expert)
    routed_gate_proj: ttnn.Tensor       # Single DRAM tensor
    routed_up_proj: ttnn.Tensor         # Single DRAM tensor
    routed_down_proj: ttnn.Tensor       # Single DRAM tensor
```

Note that `gate_mm` is absent from the dense layer dataclass -- dense layers have no MoE gating, so no gate weight matrix is needed.

### MoE Layer Weights

```python
@dataclass
class DeepSeekV3MoELayerWeights:
    # Same MLA attention fields as dense...
    q_a_proj: OverlappedTensor
    q_b_proj: OverlappedTensor
    kv_a_proj: OverlappedTensor
    o_proj: OverlappedTensor
    gate_mm: OverlappedTensor           # (7168, 256), BF16 -- MoE GATE ONLY
    attn_norm: OverlappedTensor
    q_norm: OverlappedTensor
    kv_norm: OverlappedTensor
    ffn_norm: OverlappedTensor
    gate_bias: ttnn.Tensor              # e_score_correction_bias, (16, 16) on sender core

    kv_b1_proj: OverlappedTensor
    kv_b2_proj: OverlappedTensor

    # Shared expert
    shared_gate_proj: OverlappedTensor
    shared_up_proj: OverlappedTensor
    shared_down_proj: ttnn.Tensor

    # 256 routed experts (lists of DRAM tensors)
    routed_gate_proj: list[ttnn.Tensor]  # 256 experts
    routed_up_proj: list[ttnn.Tensor]    # 256 experts
    routed_down_proj: list[ttnn.Tensor]  # 256 experts
```

The critical differences: MoE layers add `gate_mm` (the gating projection) and `gate_bias` (expert score correction), and have **lists** of routed expert weights (256 per projection) instead of single tensors.

### Per-Field Detail (MoE Layer)

The following tables provide per-field shape, precision, and core placement for each fusion group:

**Fusion group `q_ab_kv_a`** (fused into one WIDTH\_SHARDED buffer across 96 + 18 cores):

| Field | Logical Shape | Precision | Placement |
|---|---|---|---|
| `q_a_proj` | $(7168, 1536)$ | BFP8 | 96 cores ($8 \times 12$ grid) |
| `q_b_proj` | $(1536, 12288)$ | BFP8 | 96 cores (same grid, stacked below `q_a`) |
| `kv_a_proj` | $(7168, 576)$ | BFP8 | 18 cores (rows 8--9, cols 0--8) |

**Fusion group `o_proj_gate_mm_norms`** (fused into one buffer across 112 + 8 + 2 dedicated cores):

| Field | Logical Shape | Precision | Core Placement |
|---|---|---|---|
| `o_proj` | $(8192, 7168)$ | BFP8 | 112 cores |
| `gate_mm` | $(7168, 256)$ | BF16 | 8 cores (column 12, rows 0--7) |
| `attn_norm` | $(1, 7168)$ | BF16 | Core (12, 9) -- gamma core |
| `q_norm` | $(1, 1536)$ | BF16 | Core (12, 9) -- gamma core |
| `kv_norm` | $(1, 512)$ | BF16 | Core (0, 8) -- KV norm core |
| `ffn_norm` | $(1, 7168)$ | BF16 | Core (12, 9) -- gamma core |

**Fusion group `kv_b12`** (fused into one buffer across 64 + 64 cores):

| Field | Logical Shape | Precision | Core Placement |
|---|---|---|---|
| `kv_b1_proj` | $(8192, 512)$ | BFP8 | 64 cores ($8 \times 8$ Qnope grid), HEIGHT\_SHARDED |
| `kv_b2_proj` | $(512, 8192)$ | BFP8 | 64 cores (remaining), tile-rearranged |

**Fusion group `gate_up`** (fused into one buffer across 64 + 64 cores):

| Field | Logical Shape | Precision | Core Placement |
|---|---|---|---|
| `shared_gate_proj` | $(7168, 256)$ | BFP4 | 64 A-cores, block-sharded |
| `shared_up_proj` | $(7168, 256)$ | BFP4 | 64 B-cores, block-sharded |

**Standalone weights** (not fused):

| Field | Logical Shape | Precision | Placement |
|---|---|---|---|
| `shared_down_proj` | $(256, 7168)$ | BFP8 | 112 cores, WIDTH\_SHARDED |
| `gate_bias` | $(16, 16)$ | BF16 | HEIGHT\_SHARDED on core (10, 9) |
| `routed_*_proj` (256 each) | Expert-specific | BFP4/BFP8 | DRAM interleaved |

### The OverlappedTensor

Each sub-tensor within a fusion group is represented by an `OverlappedTensor` (defined in `blitz_decode_weights.py`, line 601):

```python
@dataclass
class OverlappedTensor:
    fused_tensor: ttnn.Tensor          # The shared physical buffer
    tensor_shape: tuple[int, int]      # Logical shape, e.g. (7168, 1536)
    shard_shape: tuple[int, int]       # Per-core shard, e.g. (3584, 32)
    core_range_set: ttnn.CoreRangeSet  # Which cores hold shards
    dtype: ttnn.DataType               # BFP8, BFP4, BFP16, etc.
    tile_shape: tuple[int, int]        # (32, 32) or (1, 32) for gammas
    byte_offset: int = 0              # Offset within fused buffer
    total_size: int = 0               # Byte size of this sub-tensor
```

Multiple `OverlappedTensor` instances share the same `fused_tensor` reference but have different `byte_offset` and `core_range_set` values. This is the mechanism by which the B1 system achieves near-zero L1 fragmentation: all weights for a fusion group occupy a single contiguous allocation.

### Additional Weight Types

```python
@dataclass
class DeepSeekV3EmbeddingLayerWeights:
    embedding: ttnn.Tensor    # (129280, 7168), BF16, ROW_MAJOR, DRAM

@dataclass
class DeepSeekV3LMHeadWeights:
    lm_head: ttnn.Tensor      # (7168, N_vocab_per_device), BFP8, WIDTH_SHARDED on 101 cores
    final_norm: ttnn.Tensor   # (1, 7168), BF16, HEIGHT_SHARDED on mcast core (10,9)
```

The `HostInterface` performs the embedding lookup on-device by reading from the embedding tensor using NOC addresses computed from the incoming token ID in a fused `h2d_receiver_embedding` kernel.


## 3.3 Weight Preparation Pipeline

Before any decode step can execute, weights must be loaded from the HuggingFace state dict, transformed, fused, and placed on device. The `prepare_weights.py` module manages this pipeline:

1. **Key mapping**: HuggingFace keys like `model.layers.{i}.self_attn.q_b_proj.weight` are mapped to B1 internal names.

2. **Transpose**: HuggingFace stores weights as $(out\_features, in\_features)$; B1 transposes to $(K, N) = (in\_features, out\_features)$ for the matmul convention used by TT-Blaze ops.

3. **KV\_B split**: `kv_b_proj` is split into `kv_b1` (K nope heads) and `kv_b2` (V heads) via `_split_kv_b_proj()`.

4. **TP slicing**: For the $4 \times 2$ mesh, attention weights are sliced per tensor-parallel rank (`mla_tp=2`); MoE weights are sliced 8-way (`moe_tp=8`).

5. **Shuffling**: Per-fusion-group shuffles reorder columns/shards for the physical core layout. For example, `shuffle_q_b` interleaves Qnope and Qrope heads by row groups; `shuffle_kv_a` reorders shards to match the KV cache branch core layout.

6. **Fusion**: `BlitzDecodeWeights` stitches multiple weight matrices into a single WIDTH\_SHARDED buffer per fusion group, producing `OverlappedTensor` views.

The key shape transformations from HF convention to TT-Blaze convention:

| Weight | HF Key | HF Shape | Transform | TT-Blaze Shape |
|---|---|---|---|---|
| `q_a_proj` | `self_attn.q_a_proj.weight` | $(1536, 7168)$ | `.T` | $(7168, 1536)$ |
| `q_b_proj` | `self_attn.q_b_proj.weight` | $(24576, 1536)$ | `.T` + TP slice | $(1536, 12288)$ |
| `o_proj` | `self_attn.o_proj.weight` | $(7168, 16384)$ | `.T` + TP slice | $(16384, 7168)$ |
| `kv_b_proj` | `self_attn.kv_b_proj.weight` | $(32768, 512)$ | split | `kv_b1` + `kv_b2` |
| `kv_a_proj` | `self_attn.kv_a_proj_with_mqa.weight` | $(576, 7168)$ | `.T` | $(7168, 576)$ |
| norms | `input_layernorm.weight`, etc. | $(7168,)$ | `.unsqueeze(0)` | $(1, 7168)$ |

The shapes above are full logical shapes for a $4 \times 2$ mesh (2-TP for MLA). Single-device mode uses full un-sliced dimensions.

Per-layer weight preparation is invoked through:

```python
attn_weights = prepare_attention_weights(bdw, state_dict, layer_idx, is_moe=True)
shared_weights = prepare_shared_expert_weights(bdw, state_dict, layer_idx, is_moe=True)
```

Weight preparation supports per-layer save/load for offline preparation via a manifest system with dtype serialization mapping.


## 3.4 Pipeline Stages

At the multi-host scale, the 61 decoder layers are partitioned across pipeline stages, each running on a separate T3K (Tenstorrent Galaxy) system.

### Pipeline Topology

| Mesh ID | Stage | Content |
|---|---|---|
| 0 | Embedding | Token ID lookup: int32 token ID to $[1, 7168]$ hidden state |
| 1--3 | Dense decoder layers | Layers 0--2 (attention + single routed expert + shared expert) |
| 4--61 | MoE decoder layers | Layers 3--60 (attention + 256-expert MoE + shared expert) |
| 62 | LM head + sampling | Final RMSNorm + vocabulary projection + argmax |

The mapping from mesh ID to layer ID:

```python
def decoder_layer_id_from_mesh_id(mesh_id: int) -> int:
    assert mesh_id > 0 and mesh_id <= 61
    return mesh_id - 1
```

The constants `SYSTEM_MESH_ID_EMBEDDING = 0` and `SYSTEM_MESH_ID_LM_HEAD = 62` (`demo/cli.py`, lines 65--66) anchor the two non-decoder stages.

### Multi-Host Pipeline Architecture

```
Host ---H2D---> [Stage 0: Embed + Layers 0..L0] ---D2D---> [Stage 1: Layers L0+1..L1]
  ^                                                                |
  |                                                                D2D
  |                                                                |
  +---D2H---< [Stage 0 loopback] <---D2D---< ... <---D2D---< [Stage N-1: LM Head]
```

Each `PipelineBlock` (`micro_ops/pipeline_block/op.py`) holds:
- An **entry `SocketInterface`**: receives data from the upstream stage
- An **exit `SocketInterface`**: sends data to the downstream stage
- (Stage 0 only) A `HostInterface` for H2D/D2H communication and an embedding tensor

The socket page sizes are configured to match the data being transferred:
- Upstream D2D (loopback to Stage 0): sized to the D2H socket page size
- Downstream D2D (entry to next stage): sized to the embedding dimension ($7168 \times 2 = 14336$ bytes in bfloat16)
- H2D token input: 64 bytes (one int32 token ID, PCIe-aligned)

Pipeline topology is configured through YAML files in `scaleout_configs/`. The `generate_blitz_decode_pipeline_configs.py` script maps physical ASIC locations to logical mesh coordinates in snake/zigzag order, producing configurations for single-pod (8 devices), 2-pod (16 devices), and superpod (32 devices) deployments.

### CLI Entry Point

The CLI (`demo/cli.py`) ties together weight loading and model creation:

1. Opens a $(4, 2)$ mesh device with 2D fabric configuration.
2. Calls `load_weights_from_cache()` which dispatches based on mesh ID:
   - Mesh ID 0: `load_embedding_weights()`
   - Mesh ID 1--3: `load_dense_decoder_layer()` (with `layer_id = mesh_id - 1`)
   - Mesh ID 4--61: `load_moe_decoder_layer()` (with preloaded routed experts via `load_moe_routed_experts()`)
   - Mesh ID 62: `load_lm_head_weights()`
3. Creates the `DeepSeekV3` model via `create_model()` from `demo/runtime.py`.
4. Calls `run_generation()` with the loaded tokenizer and model.

The demo requires slow dispatch mode (`TT_METAL_SLOW_DISPATCH_MODE=1`).


## 3.5 High-Level Decode Step Overview

Putting it all together, here is what happens during a single autoregressive decode step for the full 61-layer model across a multi-host pipeline:

### Phase 1: Token Ingestion (Stage 0)

1. Host writes a single int32 token ID (4 bytes, padded to 64 bytes) to the H2D socket.
2. The `HostInterface` on the entry device receives the token.
3. Embedding lookup: the token ID indexes into the $(129280, 7168)$ embedding table, producing a $[1, 7168]$ activation in bfloat16.
4. The activation is forwarded to the first decoder layer via the downstream D2D socket.

### Phase 2: Decoder Layers (Stages 0..N-1)

For each of the 61 layers, the activation $\mathbf{h} \in \mathbb{R}^{1 \times 7168}$ passes through:

**Attention block:**

5. **PreSDPA**: Executes the 11-stage pipeline described in Section 3.1.1, producing query/key/value projections and performing Flash MLA decode attention. The activation arrives at core (12,9) via D2D socket or CCL broadcast and exits as per-head $[1, 512]$ attention outputs.

6. **PostSDPA**: Executes the 6-stage output projection and cross-device reduction described in Section 3.1.1, producing the final attention output with residual add: $\mathbf{h} \leftarrow \mathbf{h} + \text{attn output}$.

**FFN block:**

7. **Dense layers**: `SharedExpertOp` performs the standard gated MLP. A single routed expert also fires alongside.

8. **MoE layers**: `MoeOp` performs:
   - RMSNorm ($\text{ffn norm}$)
   - Gate matmul + top-8 selection
   - 8 routed expert forward passes (DRAM-streaming matmuls, processed sequentially)
   - Shared expert forward pass (L1-resident matmuls, concurrent with routed)
   - `ReduceToOneB1` tree reduction across the $4 \times 2$ mesh
   - Accumulation and residual add: $\mathbf{h} \leftarrow \mathbf{h} + \text{moe output}$

9. The updated $\mathbf{h}$ is forwarded to the next layer (or next pipeline stage via D2D socket).

### Phase 3: LM Head + Sampling (Final Stage)

10. **LMHeadSampling** (fused op):
    - Final RMSNorm on the last hidden state
    - CCL Broadcast of the $[1, 7168]$ activation to all 8 devices
    - Per-device matmul: $[1, 7168] \times [7168, N_\text{per device}]$ across 101 cores
    - Fused argmax across all devices (two-stage mesh reduction)
    - The winning token index is written to the D2H socket (or forwarded via D2D socket for persistent mode)

### Phase 4: Output Return (Stage 0)

11. The sampled token ID travels back through the D2D loopback chain to Stage 0.
12. Stage 0 writes the token ID to the D2H socket.
13. The host reads the token ID and either feeds it back for the next step or terminates.

### Summary: From Token to Token

```
Token ID (int32) --> H2D Socket (HOST_PUSH)
    --> Embedding Lookup
    --> For layer_idx in 0..60:
        +-- PreSDPA: RMSNorm -> q_a -> q_b -> create_q_heads -> kv_a ->
        |            KV cache branch -> Flash MLA
        +-- PostSDPA: kv_b2 matmul -> gather -> mcast -> o_proj matmul ->
        |             gather -> CCL all-reduce + residual
        +-- [Dense layer]: SharedExpert + single RoutedExpert
        |   OR
        +-- [MoE layer]: Gate -> top-8 RoutedExperts (DRAM streaming) +
                         SharedExpert -> ReduceToOne + residual
    --> Final RMSNorm
    --> LM Head Matmul
    --> Sampling (argmax)
    --> D2H Socket --> Token ID (int32)
```

The entire path from token ingestion to next-token output runs as a sequence of `ttnn.generic_op` dispatches, each carrying a fully specified `ProgramDescriptor` with pre-compiled unified kernels and pre-configured circular buffers. There is no Python-level control flow between RISC instructions; all intra-op coordination happens through CB semaphores and NOC transfers specified at compile time.

### Timing Characteristics

The $B = 1$ constraint means this entire pipeline processes exactly one token per step. The critical path is dominated by:

- **Memory bandwidth**: Weight reads from L1 for the $[1, K] \times [K, N]$ matmuls (compute intensity $\approx 1$)
- **DRAM streaming**: Routed expert weights in MoE layers must be fetched from DRAM (8 experts $\times$ 3 projections per layer)
- **Cross-device communication**: CCL broadcasts, all-reduces, and D2D socket transfers at pipeline stage boundaries
- **Sequential expert execution**: Routed experts are processed one at a time (no expert-level parallelism within a layer)

The fused-op architecture minimizes the overhead between sequential operations: rather than returning to the host between each matmul, mcast, and gather, the entire attention block (PreSDPA) or FFN block (MoeOp/SharedExpertOp) executes as a single `ttnn.generic_op` call with all inter-core communication handled through CB handshakes and semaphores on-chip.

---

**Next:** [Chapter 2 -- The Unified Kernel System](../ch02_the_unified_kernel_system/index.md)
