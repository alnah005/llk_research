# 4.1 Fusion Principles and Patterns

This section explains the engineering principles that motivate kernel fusion in DeepSeek V3 B1 and the concrete patterns used throughout the fused-op library. Readers should be familiar with the unified kernel system (Chapter 2) and individual micro-ops (Chapter 3) before proceeding.

---

## 4.1.1 Why Fuse

Every `ttnn.generic_op` dispatch carries overhead: host-side program construction, CB descriptor serialization, semaphore initialization, and kernel launch latency. For decode-mode inference (batch size 1), each micro-op processes very small tensors (often a single $[1, 7168]$ vector), so the ratio of useful compute to dispatch overhead becomes punishingly low.

Fusion eliminates these costs by packing an entire model sub-graph -- sometimes 12+ stages -- into a single kernel dispatch:

1. **Zero inter-op dispatch overhead.** A single `generic_op` call replaces what would be 8--15 separate calls. For `pre_sdpa`, this means CCL broadcast, two RMSNorms, three matmuls, a gather-reduce, RoPE, CreateQHeads, KV cache update, and FlashMLA all execute from one dispatch.

2. **Producer-consumer CB pipelining.** When stage $A$ writes to CB $k$ and stage $B$ reads from CB $k$, the data flows through L1 without ever touching DRAM. The CB `push_back` / `wait_front` protocol provides implicit synchronization. For example, in `pre_sdpa`, RMSNorm pushes to `rmsnorm_output_cb` (CB 2) and the mcast sender reads from it immediately.

3. **Aggressive L1 buffer overlap.** Fused ops carefully track the lifecycle of each CB and overlay buffers that are consumed before a later stage begins. In `pre_sdpa`, seventeen temporary CBs (matmul input, matmul output, RMSNorm2 buffers, QRoPE intermediates, CreateQHeads staging, DKV matmul output, KV RMSNorm buffers, K-RoPE output) are all overlapped with `sdpa_out_interm_buffer` at tracked byte offsets:

$$
\text{CB 5} \to \text{offset } 0 \text{ B}, \quad
\text{CB 4} \to \text{offset } 14336 \text{ B}, \quad
\text{CB 7} \to \text{offset } 14400 \text{ B}, \quad \ldots
$$

This technique reclaims tens of kilobytes of L1 per core that would otherwise require separate allocations.

4. **Semaphore reuse across stages.** Because stages execute sequentially, semaphore IDs used by early stages can be safely reused by later stages. In `pre_sdpa`, the CreateQHeads 3-phase semaphores reuse the gather and mcast semaphore addresses:

```python
nope_phase1_semaphore_addr = gather_noc0_receiver_semaphore_addr  # ID 2
nope_phase2_semaphore_addr = gather_noc1_receiver_semaphore_addr  # ID 3
rope_semaphore_addr = mcast_data_sender_semaphore_addr            # ID 0
```

Semaphore reuse is safe only when the operations they gate are **temporally non-overlapping**. For instance, `MCAST_SENDER` (ID 0) is reused for both the input activation mcast and the expert scale mcast in the shared expert path because those stages are separated by a full gather barrier.

5. **Overlapped execution via RISC specialization.** BRISC, NCRISC, and TRISC operate concurrently. While TRISC computes a matmul, NCRISC can prefetch the next stage's weights from DRAM, and BRISC can multicast the previous stage's output. The three-RISC architecture (see Chapter 2) allows true parallelism when CB handoffs are designed correctly.

---

## 4.1.2 CB Reconfig and `CircularBufferIdManager`

The Tensix architecture provides 64 circular buffer slots per core (indices 0--63). Simple fused ops like `broadcast_rms` (4 CBs) or `gated_local_reduce` (4 CBs) fit easily. But the fused `moe` op, which composes both routed and shared expert paths, requires far more than 64 logically distinct buffers -- the routed expert alone uses ~30 and the shared expert adds another 15.

### The Manager

`CircularBufferIdManager` (from `circular_buffer_utils.py`) tracks allocated CB IDs and their associated `(data_format, tile)` signatures:

```python
class CircularBufferIdManager:
    NUM_CIRCULAR_BUFFERS = 64

    def _allocate_id(self, data_format, tile, exclude):
        # Reuse existing ID if (data_format, tile) match and ID not in exclude set
        for cb_id, fmt_key in self._id_to_format.items():
            if fmt_key == (data_format, tile) and cb_id not in exclude:
                return cb_id
        # Otherwise allocate fresh
        ...
```

### Contexts

A `Context` represents one execution phase (e.g., the routed-expert pass or the shared-expert pass). **Within a single context**, every CB ID is unique. **Across contexts**, IDs can be shared when the data format and tile shape match:

```python
ctx_routed = manager.create_context()
ctx_shared = manager.create_context()
# These may return the same CB ID if data_format/tile match:
cb_a = ctx_routed.get_cb_id(ttnn.bfloat16, tile_1x32_desc)
cb_b = ctx_shared.get_cb_id(ttnn.bfloat16, tile_1x32_desc)
```

The MoE orchestrator (`moe/op.py`) uses this extensively: it creates separate contexts for the routing matmul stage, the gate stage, the expert matmul stage, and the down-projection stage, allowing CB IDs to be recycled across non-overlapping stages.

### Dummy CBs at Compile Time

When CB reconfig is active, the CBs declared in the `ProgramDescriptor` are **dummy descriptors** -- they carry the correct format (data format, tile descriptor) but use minimal sizing (`total_size = page_size`). The `CircularBufferIdManager.build_dummy_cb_descriptors()` method generates these. The real sizes and addresses are applied at runtime from the reconfig tensor.

### The Reconfig Tensor

At runtime, `build_cb_reconfig_tensor` serializes the CB metadata for a set of CBs into a host tensor that the kernel reads to reprogram CB registers between phases. The function `record_cb_metadata` captures the current state of each CB descriptor for later serialization:

```python
cb_metadata = record_cb_metadata(program.cbs)
reconfig_tensor = build_cb_reconfig_tensor(cb_metadata, full_device_grid, mesh_device)
```

The reconfig tensor is a `HEIGHT_SHARDED` L1 tensor with one shard per core. Each shard is 264 `uint32` words (1056 bytes):

| Words | Content |
|-------|---------|
| 0--255 | 64 CB configs, 4 words each: `[addr_bytes, size_bytes, num_pages, page_size_bytes]` |
| 256 | `cb_mask_low` (bitmask for CBs 0--31) |
| 257 | `cb_mask_high` (bitmask for CBs 32--63) |
| 258--259 | Cross-RISC sync semaphores (initialized to 0) |
| 260--263 | Reserved |

The fused kernel reads this tensor at startup and calls `setup_local_cb_read_write_interfaces()` to reconfigure all active CBs. This is the mechanism introduced in Chapter 2 -- here we see it deployed in production for the MoE fused op, where the routed-expert and shared-expert kernels must each configure their own CB layout despite running on the same physical cores.

---

## 4.1.3 Face-View Optimization

Several fused ops (`shared_expert`, `gated_local_reduce_down_proj`, and the shared expert path within `moe`) use **face-view** tiling to reduce the number of tiles processed during gather-reduce operations.

Standard tiles are $32 \times 32$ (1024 elements). A face is $16 \times 16$ (256 elements). When the activation is $[1, K]$ and the gathered tiles are small enough to pack into faces, the `can_use_face_view` function from `face_view_utils.py` returns `True`:

```python
FACE_HEIGHT = 16
FACE_WIDTH = 16
FACE_ELEMENTS = 256

def can_use_face_view(tile_h, tile_w, tiles_per_k, k_num_tiles):
    elements_per_tile = tile_h * tile_w
    if elements_per_tile >= FACE_ELEMENTS:
        return False
    if elements_per_tile * k_num_tiles != FACE_ELEMENTS:
        return False
    if tiles_per_k < 2 or tiles_per_k % 2 != 0:
        return False
    return True
```

The conditions require: (1) tiles smaller than a face, (2) each collection's `k_num_tiles` tiles exactly filling one face (e.g., 8 tiles of $1 \times 32 = 256$ elements = one face), (3) at least 2 faces for pairwise reduction, and (4) an even face count.

**Example:** With $1 \times 32$ tiles, `k_num_tiles = 8`, and `tiles_per_k = 8`: each tile has 32 elements, $8 \times 32 = 256$ fills one face. The reduce kernel processes 8 faces using pairwise face-level addition, rather than iterating over $8 \times 8 = 64$ individual tiles.

When face-view is active, the kernel uses $16 \times 16$ `face_tile_desc` instead of full tiles:

```python
if use_face_view:
    face_tile = ttnn.Tile([FACE_HEIGHT, FACE_WIDTH])
    face_tile_desc = ttnn.TileDescriptor(FACE_HEIGHT, FACE_WIDTH, False)
    kernel_k_num_tiles = 1         # All K positions packed into one face
    mcast_src_num_pages = 1        # Single face to mcast
    reduce_tile_size = face_tile_size
```

---

## 4.1.4 Compile-Time Roles

Every fused op partitions the device grid into **role groups** using `UnifiedCompileTimeCoreDescriptor` (see Chapter 2 for the mechanism). Each descriptor maps a named boolean compile-time argument to a `CoreRangeSet`:

```python
UnifiedCompileTimeCoreDescriptor(
    named_compile_time_arg="is_matmul_core",
    core_range=matmul_weights_core_grid,   # e.g., 96 cores
    value=1,
    other_value=0,
)
```

Inside the kernel, `is_matmul_core` is a compile-time constant. The kernel uses `if constexpr` to select the appropriate code path, and the compiler eliminates dead code for each core group, producing specialized binaries per role:

```cpp
if constexpr (is_matmul_core) {
    // execute matmul stage
}
if constexpr (is_qrope_core) {
    // execute RoPE stage
}
```

### Role Taxonomy Across Fused Ops

| Fused Op | # Roles | Key Role Flags |
|----------|---------|---------------|
| `broadcast_rms` | 3 | `is_sender` (per-device CCL sender/receiver/secondary) |
| `kv_cache_branch` | 4 | `is_dkv_matmul_core`, `is_kv_rmsnorm_core`, `is_knope_core`, `is_krope_core` |
| `down_proj` | 5 | `is_mcast_sender_core`, `is_mcast_receiver_core`, `is_matmul_core`, `is_gather_receiver_core`, `gather_use_per_core_sender_idx` |
| `shared_expert` | 8 | `is_gate_compute_core`, `is_up_compute_core`, `is_gated_reduce_core`, `is_mcast_sender_core`, `is_mcast_receiver_core`, `is_matmul_core`, `is_gather_receiver_core`, `gather_use_per_core_sender_idx` |
| `post_sdpa` | 8 | `is_matmul4_core`, `is_gather_receiver_core`, `is_matmul5_core`, `is_mcast3_receiver_core`, `is_ccl_sender_core`, `is_ccl_receiver_core`, `is_sdpa_worker_core`, `is_sdpa_forwarder_core` |
| `pre_sdpa` | 11+ | `is_input_core`, `is_matmul_core`, `is_matmul2_core`, `is_qnope_core`, `is_qrope_core`, `is_sdpa_input_core`, `is_dkv_matmul_core`, `is_kv_rmsnorm_core`, `is_knope_core`, `is_krope_core`, `is_mla_core` |
| `lm_head_sampling` | 7 | `is_input_core`, `is_mcast_receiver_core`, `is_matmul_core`, `is_rmsnorm_core`, `is_argmax_core`, `is_argmax_final_core`, `is_argmax_mesh_sender_core` |

A single core can hold multiple role flags set to 1 -- for example, the sender core at `(12, 9)` in `shared_expert` is simultaneously `is_gated_reduce_core`, `is_mcast_sender_core`, and `is_gather_receiver_core`.

### Per-Core Compile-Time Descriptors

Beyond boolean roles, `PerCoreCompileTimeDescriptor` assigns per-core **scalar values**. For example, `gather_sender_idx` assigns each core a unique integer index so it knows where in the destination buffer to write during a gather operation. In `moe_routed_expert`, every DRAM matmul core receives a unique `gate_proj_bank_id` and `gate_proj_vc` (virtual channel) for contention-free NOC access.

---

## 4.1.5 `OverlappedTensor` and Weight Packing

Multiple logical weight tensors can share a single physically-allocated L1 buffer when they occupy non-overlapping regions. The `OverlappedTensor` dataclass (defined in `blitz_decode_weights.py`, detailed in Chapter 7) carries:

```python
@dataclass
class OverlappedTensor:
    fused_tensor: ttnn.Tensor       # The actual device buffer (shared)
    tensor_shape: tuple[int, int]   # Logical shape of this sub-tensor
    shard_shape: tuple[int, int]    # Per-core shard dimensions
    core_range_set: ttnn.CoreRangeSet  # Which cores hold shards
    dtype: ttnn.DataType
    tile_shape: tuple[int, int]
    byte_offset: int = 0            # Starting byte within the fused buffer
    total_size: int = 0             # Size of this sub-tensor's region
```

The `byte_offset` and `total_size` fields are critical: they define the exact window within the fused buffer that this sub-tensor occupies. The helper `cb_descriptor_from_overlapped_tensor` creates a CB descriptor referencing the correct byte range:

```python
def cb_descriptor_from_overlapped_tensor(cb_index, overlapped, fused_tensor_device):
    cb_desc = ttnn.cb_descriptor_from_sharded_tensor(
        cb_index, fused_tensor_device,
        address_offset=overlapped.byte_offset,
        total_size=overlapped.total_size,
        core_ranges=overlapped.core_range_set,
    )
    tile = ttnn.Tile(overlapped.tile_shape)
    cb_desc.format_descriptors = [ttnn.CBFormatDescriptor(
        buffer_index=cb_index,
        data_format=overlapped.dtype,
        page_size=tile.get_tile_size(overlapped.dtype),
        tile=ttnn.TileDescriptor(tile),
    )]
    return cb_desc
```

This pattern is used extensively in `pre_sdpa` (gamma, q\_a\_proj weights, q\_norm gamma, q\_b\_proj weights, kv\_b1\_proj weights all share fused buffers) and `post_sdpa` (kv\_b2 and o\_proj weights share separate fused buffers).

---

## 4.1.6 The Canonical Construction Pattern

Every fused op follows a consistent construction pipeline. The variation lies in the specific stages composed, not in the overall skeleton:

### Simple Ops (broadcast_rms, gated_local_reduce)

```python
class BroadcastRMSNorm:
    @staticmethod
    def op(...):
        # 1. Compute dimensions, CB indices, tile sizes
        # 2. Per-device loop (for mesh ops):
        #    a. Determine CCL roles (sender/receiver/secondary)
        #    b. Build named compile-time args for NCRISC, BRISC, TRISC
        #    c. Build CB descriptors (tensor-backed + manual)
        #    d. Create UnifiedKernelDescriptor
        #    e. Create ProgramDescriptor
        #    f. Append fabric routing args if needed
        #    g. Add to MeshProgramDescriptor
        # 3. ttnn.generic_op(io_tensors, mesh_program_descriptor)
```

### Complex Ops (shared_expert, moe)

For ops with 100+ lines of setup, the construction is decomposed into a `@dataclass` context plus static build methods:

```
_setup_dimensions()  --> Context dataclass (grids, tile sizes, CB indices, semaphore IDs)
_build_compile_time_args(ctx)  --> NCRISC, BRISC, TRISC named arg lists
_build_cb_descriptors(ctx)  --> List[CBDescriptor]
_build_core_descriptors(ctx)  --> role flags + per-core values
UnifiedKernelDescriptor(...)  --> single fused .cpp kernel source, combined args
ProgramDescriptor(kernels=..., cbs=..., semaphores=...)  --> dispatch via generic_op
```

This decomposition is used by `SharedExpertOp`, `GatedLocalReduceDownProjOp`, `MoeRoutedExpertOp`, `MoeSharedExpertOp`, and the top-level `MoeOp`.

### The Fused MoE Meta-Pattern

The `moe` fused op adds one more level: it composes `MoeRoutedExpertOp` and `MoeSharedExpertOp` as sub-ops, each with their own context dataclasses (`_MoeRoutedExpertContext`, `_MoeSharedExpertContext`), wrapped by a top-level `MoeContext`. The `MoeOp.op()` method:

1. Calls `MoeRoutedExpertOp._setup_dimensions()` to build the routed context
2. Calls `MoeSharedExpertOp._setup_dimensions()` to build the shared context
3. Merges both into a `MoeContext`
4. Builds a single `UnifiedKernelDescriptor` spanning all cores
5. Issues one `ttnn.generic_op` call

This is the most complex single kernel launch in the entire DeepSeek V3 B1 implementation, coordinating routing, DRAM streaming matmul, gated reduce, shared expert matmul, cross-device reduction, and residual addition -- all within one program descriptor using `CircularBufferIdManager` to manage CB ID reuse across the routed and shared stages.

---

## Fused Op Inventory

The following table lists every fused op, its source location, and which section of this chapter covers it:

| Fused Op | Source | Section | Stages Composed |
|----------|--------|---------|----------------|
| `broadcast_rms` | `fused_ops/broadcast_rms/op.py` | 4.2.1 | CCL Broadcast + RMSNorm |
| `pre_sdpa` | `fused_ops/pre_sdpa/op.py` | 4.2.2 | CCL Bcast + RMSNorm + Mcast + Matmul + GatherReduce + RMSNorm2 + Mcast2 + Matmul2 + Matmul3/RoPE + CreateQHeads + KVCacheBranch + FlashMLA |
| `kv_cache_branch` | `fused_ops/kv_cache_branch/op.py` | 4.2.3 | DKV Matmul + Gather + RMSNorm + RoPE + KV Cache Write |
| `post_sdpa` | `fused_ops/post_sdpa/op.py` | 4.2.4 | Matmul4 + Gather2 + Mcast3 + Matmul5 + Gather3 + CCL AllReduce + SDPA ReduceToAll |
| `moe` | `fused_ops/moe/op.py` | 4.3.1 | Orchestrator for routed + shared expert |
| `moe_routed_expert` | `fused_ops/moe_routed_expert/op.py` | 4.3.2 | Mcast + Gate MM + Gather + Gate + Index/Scale Mcast + DRAM gate/up/down proj + Mul + Add + ReduceToOne |
| `shared_expert` | `fused_ops/shared_expert/op.py` | 4.3.3 | Act Mcast + Gate/Up KN-sliced Matmul + Gather + GatedLocalReduce + Mcast + Down Proj + ResidualAdd + Output Gather |
| `gated_local_reduce` | `fused_ops/gated_local_reduce/op.py` | 4.3.4 | Three-phase LocalReduceSilu |
| `gated_local_reduce_down_proj` | `fused_ops/gated_local_reduce_down_proj/op.py` | 4.3.5 | Input Gather + GatedLocalReduce + Mcast + Matmul + ResidualAdd + Output Gather |
| `down_proj` | `fused_ops/down_proj/op.py` | 4.3.6 | Mcast + Matmul + ResidualAdd + Gather |
| `lm_head_sampling` | `fused_ops/lm_head_sampling/op.py` | 4.4.1 | CCL Broadcast + RMSNorm + Mcast + Matmul + Argmax + Mesh Argmax |

---

## Semaphore Allocation Strategy

Fused ops use two semaphore mechanisms:

1. **Local semaphores** (`SemaphoreDescriptor` in `ProgramDescriptor`): L1-resident, initialized per-program. Used for intra-device coordination (mcast sender/receiver, gather arrival).

2. **Global semaphores** (via `ttnn.create_global_semaphore`): Persistent across dispatches, shared by address. Used for cross-device CCL synchronization and the MoE reduce-to-one protocol.

---

**Prev:** [Chapter 3 -- The Micro-Op Library](../ch03_the_micro_op_library/index.md) | **Next:** [Attention Fused Ops](02_attention_fused_ops.md)
