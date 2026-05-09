# 01 -- DeepSeek V3 Architecture on Tenstorrent

This section maps the 671-billion-parameter DeepSeek V3 model onto its Tenstorrent implementation. It introduces the model's two defining architectural innovations (Multi-head Latent Attention and 256-expert MoE), shows how they decompose into a three-tier op hierarchy, walks through the directory layout, explains the batch-1 constraint and its consequences, describes the physical core grid, and introduces the multi-device topology.


## 1.1 The DeepSeek V3 Model at a Glance

DeepSeek V3 is a Mixture-of-Experts (MoE) language model whose architecture combines two innovations that make it simultaneously powerful and memory-hungry:

| Parameter | Value | Source in Code |
|---|---|---|
| Hidden dimension ($d_\text{model}$) | 7168 | `attn_norm_shape: (1, 7168)` in `blitz_decode_weights.py` |
| Decoder layers | 61 (layers 0--60) | `prepare_weights.py`, `demo/cli.py` |
| Attention heads (queries) | 128 (each head has both a nope and a rope component) | Full W_q_b HF shape (24576, 1536); per-TP=2 device: `num_qnope_heads=64, num_qrope_heads=64` |
| KV latent rank ($d_\text{kv lora}$) | 512 | `_KV_LORA_RANK = 512` in `prepare_weights.py` |
| Qnope head dimension | 128 | `_QK_NOPE_HEAD_DIM = 128` |
| Qrope head dimension | 64 | Implied by $24576 - 128 \times 128 = 128 \times 64$; full-model W_q_b output is 24576 |
| KV-rope dimension ($d_\text{rope}$) | 64 | 576 total latent minus 512 nope |
| Vocabulary size | 129,280 | `_LM_HEAD_VOCAB_SIZE = 129280` in `prepare_weights.py` |
| Routed experts per MoE layer | 256 | `NUM_ROUTED_EXPERTS = 256` in `prepare_weights.py` |
| Active routed experts per token | 8 | Top-8 gating in `deepseek_moe_gate` |
| Shared expert | 1 per MoE layer | Always fires alongside routed experts |
| Dense layers (0..2) | 3 | `FIRST_K_DENSE_REPLACE = 3` in `demo/cli.py` |
| MoE layers (3..60) | 58 | Remaining layers use 256-expert MoE |
| Total parameters | ~671 B | Dominated by 256-expert weight banks |
| Active parameters per token | ~37 B | 8 routed + 1 shared expert per MoE layer |


### 1.1.1 Multi-head Latent Attention (MLA)

Standard multi-head attention stores a KV cache of size $O(n_h \cdot d_k \cdot L)$ per layer, where $L$ is the sequence length. MLA replaces this with a two-stage projection through a low-rank bottleneck, reducing the KV cache to a 576-element compressed vector per layer per position (versus $128 \times 256 = 32{,}768$ elements with standard MHA).

The KV path projects the hidden state down to a latent space, normalizes, and then expands:

$$
\mathbf{c}_{kv} = W_{kv\_a} \cdot \mathbf{x} \quad \in \mathbb{R}^{1 \times 576}
$$

The 576-dim latent splits into a 512-dim nope component and a 64-dim rope component:

$$
[\mathbf{K}_\text{nope} \| \mathbf{V}] = W_{kv\_b} \cdot \text{RMSNorm}(\mathbf{c}_{kv}[:, :512])
$$

$$
\mathbf{K}_\text{rope} = \text{RoPE}(\mathbf{c}_{kv}[:, 512:])
$$

Only $\mathbf{c}_{kv}$ plus $\mathbf{K}_\text{rope}$ are cached (576 elements per layer per position).

The query path follows a similar two-stage structure:

$$
\mathbf{c}_q = W_{q\_a} \cdot \mathbf{x}, \quad
\hat{\mathbf{c}}_q = \text{RMSNorm}(\mathbf{c}_q), \quad
[\mathbf{Q}_\text{nope} \| \mathbf{Q}_\text{rope}] = W_{q\_b} \cdot \hat{\mathbf{c}}_q
$$

where $W_{q\_a}$: $(7168, 1536)$ and $W_{q\_b}$: $(1536, 24576) = (1536,\; 128 \times 128 + 128 \times 64)$.

The Tenstorrent implementation splits $W_{kv\_b}$ into two separate matrices at weight-preparation time (`_split_kv_b_proj` in `prepare_weights.py`, lines 210--224):

```python
def _split_kv_b_proj(kv_b_proj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = kv_b_proj.reshape(num_heads, _KV_B_PROJ_HEAD_DIM, _KV_LORA_RANK).contiguous()
    kv_b1 = w[:, :_QK_NOPE_HEAD_DIM, :].reshape(-1, _KV_LORA_RANK)   # (8192, 512)
    kv_b2 = w[:, _QK_NOPE_HEAD_DIM:, :].reshape(-1, _KV_LORA_RANK).T  # (512, 8192)
    return kv_b1, kv_b2
```

- **kv\_b1**: shape $(N_\text{heads} \cdot d_\text{qk nope}, d_\text{kv lora}) = (8192, 512)$ -- the key-nope projection
- **kv\_b2**: shape $(d_\text{kv lora}, N_\text{heads} \cdot d_v) = (512, 8192)$ -- the value projection (transposed for decode)

This split maps cleanly onto the two-stage matmul chain in the `PostSDPA` fused op (Matmul4 + Matmul5), where each stage operates on a different physical core grid.


### 1.1.2 256-Expert Mixture of Experts

In MoE layers (3--60), the FFN is replaced by a gating network that selects 8 out of 256 routed experts plus 1 shared expert. The gating operation in `deepseek_moe_gate` (`micro_ops/deepseek_moe_gate/op.py`) produces 8 normalized scores and 8 expert indices through:

1. **Sigmoid scoring**: $s_i = \sigma(\mathbf{x} \cdot W_\text{gate})_i$ (the unbiased score). Also compute a ranking value $r_i = s_i + \text{bias}_i$ used only for selection (the bias is `e_score_correction_bias`).
2. **Top-8 selection**: Sort all 256 experts by $r_i$ (the biased ranking value), select the top 8.
3. **Score normalization** (uses the unbiased scores $s_i$, not $r_i$):

```math
\hat{s}_i = \frac{s_i}{\sum_{j \in \text{top-8}} s_j + \epsilon} \cdot \alpha
```

where $\alpha = 2.5$ is the scaling factor and $\epsilon = 10^{-20}$.

Each routed expert has three weight matrices: `gate_proj`, `up_proj`, and `down_proj`. At 256 experts, these live in DRAM and are streamed to L1 on demand via `dram_streaming_matmul`. The shared expert uses the same gated MLP structure but its weights are fused into L1-resident overlapped tensors for low-latency access.

Key constants from `prepare_weights.py`:

```python
NUM_ROUTED_EXPERTS = 256
_QK_NOPE_HEAD_DIM = 128
_V_HEAD_DIM = 128
_KV_LORA_RANK = 512
_KV_B_PROJ_HEAD_DIM = _QK_NOPE_HEAD_DIM + _V_HEAD_DIM  # 256
```


## 1.2 The Three-Tier Op Hierarchy

The codebase organizes compute into three tiers that correspond to increasing levels of hardware fusion:

```
Tier 1: micro_ops/        -- Single mathematical primitives (one ProgramDescriptor each)
Tier 2: fused_ops/        -- Multi-primitive fusions sharing a ProgramDescriptor
Tier 3: model.py + demo/  -- Host-side orchestration (DeepSeekV3 class)
```

### Tier 1: Micro-Ops (`micro_ops/`)

Each micro-op implements a single hardware primitive -- one `ProgramDescriptor`, one kernel launch. There are 24 micro-op directories:

| Micro-Op | Purpose |
|---|---|
| `matmul` | Single-core $[1, K] \times [K, N]$ with optional fused activation (SiLU, sigmoid) |
| `rmsnorm` | Single-core RMS normalization |
| `rope` | Rotary positional embedding application |
| `mcast` | NOC multicast from one core to a rectangular grid |
| `gather` | Collect shards from a grid to a single core |
| `flash_mla` | Flash Multi-Latent Attention decode (shared KV tensor) |
| `sdpa` | Scaled dot-product attention |
| `sdpa_tail` | Post-SDPA output reshaping |
| `sdpa_reduce_to_all` | Cross-device SDPA reduction |
| `kv_cache_update` | BFP8 KV cache read-modify-write in DRAM |
| `create_q_heads` | Reshape/scatter Q projections to per-head layout |
| `local_reduce` | Per-core partial sum reduction with optional SiLU |
| `eltwise_add` | Per-core indexed addition |
| `kn_sliced_matmul` | K/N-sliced block-sharded matmul |
| `dram_streaming_matmul` | Multi-core matmul streaming weights from DRAM |
| `deepseek_moe_gate` | Sigmoid + bias + top-K selection for MoE routing |
| `sampling` | Token sampling (argmax / top-p) |
| `tilize_8x32` | Row-major to tiled format conversion ($8 \times 32$) |
| `host_io` | H2D/D2H socket interface with termination support |
| `d2d_exchange` | Device-to-device data forwarding via D2D sockets |
| `pipeline_block` | Multi-host pipeline stage lifecycle management |
| `ccl_broadcast` | Multi-device broadcast via fabric |
| `ccl_all_reduce` | Multi-device all-reduce via fabric |
| `reduce_to_one_b1` | 3-level reduction tree across $4 \times 2$ mesh |

Each micro-op follows the same structural pattern (detailed in [`02_tt_blaze_execution_pattern.md`](./02_tt_blaze_execution_pattern.md)): create CB descriptors, build a `UnifiedKernelDescriptor`, extract kernel descriptors, pack into a `ProgramDescriptor`, and execute via `ttnn.generic_op`.


### Tier 2: Fused Ops (`fused_ops/`)

Fused ops combine multiple micro-op-level computations into a single `ProgramDescriptor` launch. A single unified kernel source file targets all three RISC processors (NCRISC, BRISC, TRISC) with `#if defined(COMPILE_FOR_*)` guards, and `UnifiedCompileTimeCoreDescriptor` assigns roles to different cores. There are 11 fused-op directories:

| Fused-Op | What it fuses |
|---|---|
| `pre_sdpa` | CCL Broadcast + RMSNorm + Mcast + Matmul1 ($W_{q\_a}$) + Gather + RMSNorm2 + Mcast2 + Matmul2 ($W_{q\_b}$) + CreateQHeads + RoPE + Matmul3 ($W_{kv\_b1}$) + FlashMLA + KVCacheUpdate |
| `post_sdpa` | Matmul4 ($W_{kv\_b2}$) + Gather2 + Mcast3 + Matmul5 ($W_{o}$) + Gather3 + CCL All-Reduce |
| `shared_expert` | Activation Mcast + Gate/Up Matmul + Gather + GatedLocalReduce + Mcast + Down Proj Matmul + ResidualAdd + Output Gather |
| `moe` | Routed Expert pipeline + Shared Expert (full MoE layer) |
| `moe_routed_expert` | Input Mcast + Gate Matmul + Gate Selection + Index/Scale Mcast + Expert gate/up/down + Eltwise Add |
| `broadcast_rms` | CCL Broadcast + RMSNorm (building block) |
| `down_proj` | Down-projection matmul with residual add |
| `kv_cache_branch` | KV cache write + branch for attention |
| `gated_local_reduce` | SiLU-gated partial reduction |
| `gated_local_reduce_down_proj` | Gated reduce fused with down proj |
| `lm_head_sampling` | CCL Broadcast + RMSNorm + Mcast + Matmul + Argmax + Socket Output |

A fused op uses a `CircularBufferIdManager` (`circular_buffer_utils.py`) to allocate CB indices across its constituent stages. When one stage's output CB is the next stage's input CB, data stays in L1 with zero copies. The `cb_descriptor_from_overlapped_tensor` helper creates CB descriptors that alias into sub-regions of a fused weight tensor, so multiple logical weight matrices share a single physical L1 buffer.


### Tier 3: Model-Level Orchestration

The `DeepSeekV3` class in `model.py` drives the full decode sequence (prefill-by-decode, then autoregressive generation) by feeding tokens through H2D sockets and reading logits from D2H sockets. The runtime entry point in `demo/runtime.py` provides `create_model()` for instantiation and `TokenCodec` for encoding/decoding int32 token payloads. Weight preparation and layer orchestration live in `prepare_weights.py` and `blitz_decode_weights.py`.


## 1.3 Directory Layout

```
models/demos/deepseek_v3_b1/
    model.py                         # DeepSeekV3 class (host interface)
    unified_kernel_descriptor.py     # UnifiedKernelDescriptor, KernelGroup, etc.
    circular_buffer_utils.py         # CB ID management, reconfig tensor builder
    blitz_decode_weights.py          # OverlappedTensor, BlitzDecodeWeights, overlap specs
    prepare_weights.py               # Weight dataclasses, HF->device conversion
    utils.py                         # float_to_bfloat16_packed, float_to_uint32

    micro_ops/
        matmul/       op.py, kernels/    # Canonical example of the execution pattern
        rmsnorm/      op.py, kernels/
        rope/         op.py, kernels/
        mcast/        op.py, kernels/
        gather/       op.py, kernels/
        flash_mla/    op.py, kernels/
        host_io/      op.py, kernels/    # HostInterface: H2D receiver + D2H sender
        ...           (24 micro-op directories total)

    fused_ops/
        pre_sdpa/     op.py, kernels/
        post_sdpa/    op.py, kernels/
        shared_expert/  op.py, kernels/
        moe/          op.py  (+ __init__.py)
        moe_routed_expert/  op.py
        lm_head_sampling/   op.py, kernels/
        ...           (11 fused-op directories total)

    unified_kernels/                 # Shared .hpp headers for unified kernel components
        matmul.hpp
        rmsnorm.hpp
        mcast.hpp
        gather.hpp
        flash_mla.hpp
        argmax.hpp
        ...           (27 headers total)

    kernel_includes/                 # Additional shared C++ include files

    demo/
        runner.py                    # Generation loop (ModelLike protocol)
        cli.py                       # CLI entry point, FIRST_K_DENSE_REPLACE
        runtime.py                   # create_model(), TokenCodec

    scripts/
        generate_cache.py            # Offline weight preparation

    scaleout_configs/
        generate_blitz_decode_pipeline_configs.py  # Multi-host pipeline topology
        blitz_pipeline_config_single_pod.yaml
        blitz_pipeline_config_superpod.yaml

    tests/unit_tests/
        test_matmul.py, test_rmsnorm.py, ...       # Per-op unit tests
```

The `unified_kernels/` directory deserves special note: it contains the `.hpp` building blocks (e.g., `matmul.hpp`, `rmsnorm.hpp`, `mcast.hpp`, `gather.hpp`) that are `#include`-ed by fused-op kernel `.cpp` files. This is how `pre_sdpa` can compose RMSNorm, Mcast, Matmul, Gather, RoPE, FlashMLA, and KVCacheUpdate into a single kernel binary. The headers use `#if defined(COMPILE_FOR_NCRISC)` / `COMPILE_FOR_BRISC` / `COMPILE_FOR_TRISC` guards so a single source file compiles into three specialized kernels -- one for each RISC processor on every Tensix core.


## 1.4 Batch-1 Constraints

The name "B1" is literal: the entire implementation is built around $B = 1$. This is not a configuration parameter but a fundamental architectural constraint that shapes every buffer size and kernel configuration.

**Hard enforcement in `model.py` (line 102):**

```python
if batch_size != 1:
    raise ValueError(f"DeepSeekV3 currently supports only batch_size=1, got {batch_size}")
```

**Architectural consequences:**

1. **Activation shape.** Every decode step processes a single token, so the activation is always $[1, d_\text{model}] = [1, 7168]$. The `Matmul` micro-op is "hyper optimized for matmuls with shape $[1, K] \times [K, N]$ where $N$ is up to 4 tiles (128 elements)" (`micro_ops/matmul/op.py`, line 29). Many operations use $1 \times 32$ tiles (`ttnn.Tile([1, 32])`) for the activation dimension rather than the standard $32 \times 32$ tiles, eliminating wasted computation on padding rows.

2. **Memory-bandwidth-bound regime.** With $M = 1$, every matmul is effectively a matrix-vector multiply with compute intensity $\approx 1$. The decode step is dominated by weight-loading time rather than compute.

3. **Socket I/O sizing.** The H2D/D2H payload per step is exactly $B \times 4 = 4$ bytes (one `int32` token ID), padded to 64-byte PCIe alignment:

```math
\texttt{page\_size} = \left\lceil \frac{B \times 4}{64} \right\rceil \times 64 = 64 \text{ bytes for } B = 1
```

The `to_padded_input` function (`model.py`, lines 59--69) copies the 4-byte token ID into a 64-byte zero-padded buffer before sending it over the H2D socket.

4. **Prefill-by-decode.** Prefill feeds tokens one at a time through the decode path rather than using a separate prefill kernel -- the same decode kernel runs for both phases (see Section 2.3 for the full lifecycle).

5. **Weight sharding.** The overlapped-tensor system (`BlitzDecodeWeights`) exploits the fact that $B = 1$ activations are small enough to multicast to all cores, so weight matrices can be width-sharded or height-sharded across the full core grid without worrying about activation replication costs.

6. **Pipeline topology.** Multi-host pipelines (`PipelineBlock`) are designed around single-token forwarding: one H2D write and one D2H read per step, with D2D sockets sized to the embedding dimension ($7168 \times 2 = 14336$ bytes in bfloat16) rather than batched activations.

The $B = 1$ assumption permeates every abstraction boundary -- from the $1 \times 32$ tile shapes in the compute kernels, through the circular buffer sizing, to the socket page sizes. Extending to larger batch sizes would require changes at every tier.


## 1.5 Device Grid and Core Mapping

Tenstorrent Wormhole/Blackhole devices expose a $13 \times 10 = 130$-core compute grid. The B1 implementation partitions this grid into functional regions that vary by operation:

```
        Col 0   1   2   3   4   5   6   7   8   9  10  11  12
Row 0    [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [P]
    1    [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [P]
    2    [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [P]
    3    [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [P]
    4    [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [P]
    5    [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [P]
    6    [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [P]
    7    [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [P]
    8    [K] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [P]
    9    [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [S]

   M = general compute    P = phantom core (col 12, rows 0-8)
   S = mcast sender / gather receiver (12, 9)
   K = KV norm gamma core (0, 8)
```

The key partitions used across fused ops:

| Partition | Core Count | Coordinates | Used By |
|---|---|---|---|
| Full matmul grid | 112 | 130 minus 8 DRAM workers, 9 phantom (col 12, rows 0--8), 1 sender | $W_o$ matmul, down-proj |
| MLA Q-projection grid | 96 | (0,0)--(11,7), an $8 \times 12$ grid | $W_{q\_a}$, $W_{q\_b}$ in PreSDPA |
| KV-A projection grid | 18 | (0,8)--(8,9), rows 8--9 of the first 9 columns | $W_{kv\_a}$ in PreSDPA |
| Qnope / kv\_b1 grid | 64 | (0,0)--(7,7), an $8 \times 8$ grid | $W_{kv\_b1}$ per-head matmul |
| kv\_b2 grid | 64 | (8,0)--(12,7) union (0,8)--(11,9) | $W_{kv\_b2}$ in PostSDPA |
| SharedExpert A-cores | 64 | Gate matmul (block-sharded, columns 0--3 and 7--9) | Gate projection |
| SharedExpert B-cores | 64 | Up matmul (remaining columns) | Up projection |
| Shared expert down | 112 | Same as full matmul grid | Down projection |
| Mcast sender / Gather hub | 1 | Core (12, 9) | Central coordination point |
| KV norm gamma core | 1 | Core (0, 8) | $kv\_norm$ storage |
| Gate bias core | 1 | Core (10, 9) | `e_score_correction_bias` in MoE |

These assignments are encoded as `CoreRangeSet` objects in the `SingleDeviceOverlapSpec` dataclasses in `blitz_decode_weights.py` (e.g., `QAB_KVA_PROJ_SingleDeviceOverlapSpec`, `O_PROJ_GATE_MM_RMSNORM_GAMMA_SingleDeviceOverlapSpec`, `KVB12_PROJ_SingleDeviceOverlapSpec`, `GATE_UP_PROJ_SingleDeviceOverlapSpec`).


## 1.6 Multi-Device Topology

For production-scale deployment, the implementation targets a **$4 \times 2$ mesh** (8 devices):

```python
# blitz_decode_weights.py, BlitzDecodeWeights.__init__()
if num_devices == 1:
    self.mla_tp = 1
    self.moe_tp = 1
else:
    assert mesh_shape == (4, 2)
    self.mla_tp = 2   # Attention heads split across 2 columns
    self.moe_tp = 8   # Experts split across all 8 devices
```

- **MLA tensor parallelism**: `mla_tp = 2` (split across mesh columns). $W_{q\_b}$, $W_o$, $W_{kv\_b1}$, and $W_{kv\_b2}$ are column-parallel.
- **MoE tensor parallelism**: `moe_tp = 8` (all devices). Shared expert gate/up/down are 8-way parallel.
- **Multi-host pipeline**: Multiple $4 \times 2$ meshes form a pipeline ring across hosts, with each host processing a subset of the 61 layers.

Cross-device communication uses CCL (Collective Communication Library) primitives -- `ccl_broadcast` for distributing activations and `ccl_all_reduce` for aggregating partial results. The `ReduceToOneB1` micro op implements a tree reduction across the mesh with four roles:

| Role | Value | Description |
|---|---|---|
| `MESH_LEAF` | 0 | Sends partial result to parent |
| `MESH_ROOT3` | 1 | First reduction level |
| `MESH_ROOT2` | 2 | Second reduction level |
| `MESH_ROOT1` | 3 | Final root -- holds the reduced result |

---

**Next:** [`02_tt_blaze_execution_pattern.md`](./02_tt_blaze_execution_pattern.md)
