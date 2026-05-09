# 6.3 Shared Expert and Combination

This section covers the shared expert path, the `gated_local_reduce` operation, the final combination of routed and shared outputs, the semaphore coordination that keeps both pipelines synchronized, and CB reconfiguration between phases. The shared expert runs concurrently with the routed expert on a completely different set of cores, exploiting the spatial parallelism of the Tenstorrent grid to overlap two substantial computation pipelines within a single kernel dispatch.

> **Cross-reference:** Chapter 4 Section 4.3 catalogs the `shared_expert`, `gated_local_reduce`, `gated_local_reduce_down_proj`, and `down_proj` fused ops at the interface level. Section 6.2 covers the routed expert pipeline. This section focuses on the shared expert's unique architectural patterns and its interaction with the routed path.

---

## 6.3.1 Why a Shared Expert?

DeepSeek V3's MoE architecture pairs 256 routed experts with 1 shared expert per MoE layer. The shared expert always fires -- its output is added to the routed expert output regardless of which experts are selected by the gate. This provides a stable "baseline" representation that every token sees, while the routed experts specialize.

The shared expert uses the same gated MLP structure as each routed expert:

$$\text{shared\_out} = \bigl(\text{SiLU}(\text{input} \times W_{\text{gate}}^\text{shared}) \odot (\text{input} \times W_{\text{up}}^\text{shared})\bigr) \times W_{\text{down}}^\text{shared} + \text{bias}$$

The key difference from routed experts is where the weights live:
- **Routed experts:** Weights in DRAM, streamed during computation (~7.9 MB per projection per expert)
- **Shared expert:** Weights fused into L1-resident overlapped tensors for low-latency access

---

## 6.3.2 The 128-Core Dual Matmul Layout

The shared expert's gate and up projections run simultaneously on two disjoint sets of 64 cores, labeled "A cores" (gate) and "B cores" (up). The `build_ab_grids()` function (`shared_expert/op.py`, lines 168--198) constructs these grids:

```
         Col: 0  1  2  3  4  5  6  7  8  9 10 11 12
Row 0-3:      A  A  A  A  B  B  B  A  A  A  B  B  P/B
Row 4-7:      A  A  A  B  B  B  B  A  A  A  B  B  P/B
Row 8:        A  A  A  B  B  B  B  A  A  A  B  B  P(idle)
Row 9:        A  A  A  B  B  B  B  A  A  A  B  B  M(sender)

A = gate matmul core (64 total)
B = up matmul core (64 total)
P = phantom core (col 12, rows 0-8)
M = mcast/gather sender core (12,9)
```

Note: Column 12, rows 8 and 9 are excluded -- row 8 is an idle phantom core, and (12, 9) is the sender/gather core.

Each A core computes a shard of $\mathbf{x} W_\text{gate}^\text{shared}$; each B core computes a shard of $\mathbf{x} W_\text{up}^\text{shared}$. The weights are KN-sliced (split across both $K$ and $N$ dimensions) and distributed with `k_parallel` cores in the K direction and `n_parallel` cores in the N direction. With the standard configuration of 64 cores per group, the `_setup_dimensions` function (`shared_expert/op.py`, lines 276--279) establishes:

```python
num_compute_cores = 64
assert k_parallel * n_parallel == num_compute_cores
k_per_core = K_gate_tiles // k_parallel
K_down_tiles = n_parallel
```

### Why KN-Slicing, Not Just N-Slicing

An N-only split would give each core $[K, N/64]$ -- a tall, narrow shard requiring a large $K \times 32$ weight per core (28 tiles for bfloat4_b). With KN-slicing, the weight shard is only $[K/8, N/8]$ (approximately 3.5 tiles), fitting comfortably in L1 alongside the activation and output CBs. The tradeoff is needing a reduction across the K dimension, which the `gated_local_reduce` handles after gathering.

---

## 6.3.3 Shared Expert Weight Storage

Unlike routed expert weights (stored in DRAM as bfloat4_b), shared expert weights reside in **L1 SRAM** as `OverlappedTensor` objects. From `blitz_decode_weights.py` (lines 1170--1340):

| Weight | Shape | Storage | Format |
|---|---|---|---|
| `shared_gate_proj` | $(7168, 256)$ | Block-sharded across 64 A cores | bfloat4_b |
| `shared_up_proj` | $(7168, 256)$ | Block-sharded across 64 B cores | bfloat4_b |
| `shared_down_proj` | $(256, 7168)$ | WIDTH_SHARDED across 112 matmul cores | bfloat4_b |

The gate and up weights are part of a fusion group where they share a single L1 base address on each core via the `OverlappedTensor` infrastructure. The `get_tt_moe_shared_expert_weights` function (line 1170) handles the TP-sharded layout for multi-device configurations where `moe_tp > 1`.

The key advantage of SRAM storage: the shared expert's matmuls are pure compute -- no DRAM streaming overhead, no triple-buffering, no NOC contention with DRAM banks. The 128 cores can saturate their compute pipelines, making the shared expert significantly faster per-FLOP than the routed experts.

---

## 6.3.4 The Shared Expert Pipeline

The `shared_expert` fused op (`fused_ops/shared_expert/op.py`, 979 lines) implements a 9-stage pipeline on a 13x10 core grid:

```
Stage   Operation                   Core Set              Data Shape
-----   -------------------------   ------------------    ----------
  1     Activation mcast            (12,9) -> 130 cores   [1, K_gate]
  2     Gate matmul (KN-sliced)     64 A-cores            [1, K_gate] x [K, K_down/n_par]
  3     Up matmul (KN-sliced)       64 B-cores            [1, K_gate] x [K, K_down/n_par]
  4     Gate gather                 64 A-cores -> (12,9)  64 tiles -> [k_par, n_par]
  5     Up gather                   64 B-cores -> (12,9)  64 tiles -> [k_par, n_par]
  6     Gated local reduce          (12,9) TRISC          SiLU(sum(gate)) * sum(up)
  7     Mcast1 (down input)         (12,9) -> 130 cores   [1, K_down]
  7'    Mcast2 (residual/bias)      (12,9) -> 130 cores   [1, N_total]
  8     Down proj matmul            112 cores             [1, K_down] x [K_down, N/core]
  8'    Residual add                112 cores             matmul_out + bias shard
  9     Output gather               112 cores -> (12,9)   [1, N_per_core] -> [1, N]
```

### The Sender Core: (12, 9)

The sender core at logical coordinate (12, 9) is the orchestration hub. It is the mcast sender, gather receiver, and the core that runs the gated reduce. In the physical grid, (12, 9) sits at the bottom-right corner.

---

## 6.3.5 Stages 1--3: Activation Mcast and Dual Gate/Up Matmul

### Stage 1: Activation Mcast

The activation $[1, K_{\text{gate}}]$ (typically $[1, 7168]$) is broadcast from the sender core (12, 9) to all 130 cores in the $13 \times 10$ grid:

```python
# shared_expert/op.py, _build_compile_time_args, line 486
("act_mcast_src_cb", ctx.act_mcast_src_cb),
("act_mcast_src_num_pages", ctx.K_gate_tiles),
```

All 130 cores receive the activation, but only the 128 compute cores (A + B) will use it for matmul. The remaining 2 cores (DRAM workers and phantoms) receive the mcast semaphore signal but skip computation.

### Stages 2--3: KN-Sliced Gate and Up Matmul

Each A-core computes one tile of the gate projection; each B-core computes one tile of the up projection. The computation for core $(k, n)$ in either grid:

$$\text{output\_tile} = \text{activation}[k \cdot k_{\text{per\_core}} : (k+1) \cdot k_{\text{per\_core}}] \times W_{\text{shard}}[k_{\text{per\_core}}, 1]$$

The per-core K-offset is encoded as a compile-time arg (`gu_k_offset`, line 829, `shared_expert/op.py`):

```python
per_core_gu_k_offset = PerCoreCompileTimeDescriptor(
    named_compile_time_arg="gu_k_offset",
    core_values=[(core, (i // ctx.n_parallel) * ctx.k_per_core)
                 for i, core in enumerate(ctx.a_cores_list)]
              + [(core, (i // ctx.n_parallel) * ctx.k_per_core)
                 for i, core in enumerate(ctx.b_cores_list)],
    other_value=0,
)
```

Note that both A and B cores share the same `k_offset_core_values` array -- they read the same K-slices of the activation but apply them to different weight matrices (gate vs. up).

---

## 6.3.6 Stages 4--5: Gate and Up Gather

After the dual matmul, the 64 A-cores' output tiles must be gathered to the sender core, and similarly for the 64 B-cores.

**Gather pattern:** Each compute core sends its single output tile to the sender core at (12, 9). The sender receives 64 tiles for gate (into `group1_cb`) and 64 tiles for up (into `group2_cb`). The two gathers use separate semaphores to avoid interference:

| Gather | Semaphore | Source Cores | Destination CB |
|--------|-----------|-------------|----------------|
| Gate (A) | `ag_receiver_semaphore_id` (4) | 64 A-cores | CB 0 (`group1_cb`) |
| Up (B) | `bg_receiver_semaphore_id` (6) | 64 B-cores | CB 1 (`group2_cb`) |

**Sender index computation** -- each core's position in the gather determines where its tile lands in the destination buffer. For face-view mode:

```python
sender_idx = k_idx * n_parallel + n_idx
```

For non-face-view mode:

```python
sender_idx = n_idx * tiles_per_k + k_idx
```

The face-view optimization (Section 6.3.8) changes the tile ordering to enable efficient single-face processing.

---

## 6.3.7 Stage 6: The `gated_local_reduce` Operation

The `gated_local_reduce` op (`fused_ops/gated_local_reduce/op.py`, 176 lines) computes:

$$\text{output} = \text{SiLU}\Bigl(\sum_{i} \text{group1}[i]\Bigr) \odot \Bigl(\sum_{j} \text{group2}[j]\Bigr)$$

It runs on a single core -- the sender core at (12, 9) -- using TRISC.

### Three-Phase Computation

**Phase 1: Reduce group1 with SiLU**

Sum all gate tiles (from the 64 A-cores), then apply SiLU:

$$\text{intermed}[0] = \text{SiLU}\Bigl(\sum_{i=0}^{63} \text{group1\_tile}[i]\Bigr)$$

The reduction uses pairwise addition: read two tiles from `in0_cb`, add them, store result. Repeat until one tile remains. Then apply SiLU.

**Phase 2: Reduce group2 without SiLU**

Sum all up tiles (from the 64 B-cores):

$$\text{intermed}[1] = \sum_{j=0}^{63} \text{group2\_tile}[j]$$

Same pairwise reduction, but no activation function.

**Phase 3: Element-wise multiply**

$$\text{output} = \text{intermed}[0] \times \text{intermed}[1]$$

### Golden Reference

```python
@staticmethod
def golden(group1_inputs, group2_inputs):
    # Sum group 1 and apply SiLU
    group1_sum = group1_inputs[0]
    for inp in group1_inputs[1:]:
        group1_sum = group1_sum + inp
    group1_sum = torch.nn.functional.silu(group1_sum)

    # Sum group 2 (no SiLU)
    group2_sum = group2_inputs[0]
    for inp in group2_inputs[1:]:
        group2_sum = group2_sum + inp

    # Combine with element-wise multiply
    return group1_sum * group2_sum
```

### CB Layout (4 CBs)

| CB | Name | Size | Purpose |
|----|------|------|---------|
| 0 | `in0_cb` | 64+ tiles | Group 1 (gate) gather destination |
| 1 | `in1_cb` | 64+ tiles | Group 2 (up) gather destination |
| 2 | `intermed_cb` | 2 tiles | Holds phase 1 and phase 2 results |
| 3 | `out_cb` | 1 tile | Final multiply output |

**Constraints** (`gated_local_reduce/op.py`, lines 83--87):

```python
assert all_cores.num_cores() == 1, "Only single core is supported"
assert group1_num_tiles >= 2, "Need at least 2 tiles in group 1"
assert group1_num_tiles % 2 == 0, "Group 1 tile count must be even"
assert group2_num_tiles >= 2, "Need at least 2 tiles in group 2"
assert group2_num_tiles % 2 == 0, "Group 2 tile count must be even"
```

The even-tile requirement comes from the pairwise reduction pattern: tiles are consumed two at a time.

---

## 6.3.8 The Face-View Assertion and Its Mathematical Basis

The `setup_gated_reduce()` method (`moe/op.py`, lines 1972--2067) contains a critical assertion:

```python
assert can_use_face_view(tile_h, tile_w, tiles_per_k, k_num_tiles), (
    f"Face-view optimization is required for shared expert gated reduce "
    f"(tile={tile_h}x{tile_w}, tiles_per_k={tiles_per_k}, k_num_tiles={k_num_tiles})"
)
```

The `can_use_face_view()` function (`face_view_utils.py`) enforces four conditions:

1. **Tiles smaller than a face**: $\text{tile\_h} \times \text{tile\_w} < 256$ (a 16x16 face has 256 elements)
2. **K-tiles fill exactly one face**: $(\text{tile\_h} \times \text{tile\_w}) \times k_\text{num\_tiles} = 256$
3. **At least 2 reduction groups**: $\text{tiles\_per\_k} \geq 2$
4. **Even reduction count**: $\text{tiles\_per\_k} \bmod 2 = 0$

For DeepSeek V3 B1 decode with tile shape $[1, 32]$: each tile has 32 elements, so $32 \times k_\text{num\_tiles} = 256$ requires $k_\text{num\_tiles} = 8$. Since $k_\text{num\_tiles} = n_\text{parallel} = 8$ and $\text{tiles\_per\_k} = k_\text{parallel} = 8$ (which is $\geq 2$ and even), all four conditions are satisfied.

When active, face-view rewrites the CB descriptors to use 16x16 tiles instead of 1x32 tiles (lines 2034--2044):

```python
cb_group1_descriptor.format_descriptors[0].tile = face_tile_desc
cb_group1_descriptor.format_descriptors[0].page_size = face_tile_size
```

This packs 8 partial-sum tiles (each $1 \times 32 = 32$ elements) into a single $16 \times 16 = 256$ element face. The kernel then performs the reduce across `tiles_per_k=8` faces, with only `mcast_src_num_pages=1` output page. This eliminates per-tile overhead in the reduction loop and reduces the mcast data volume from $8 \times \text{tile\_size}$ to $1 \times \text{face\_tile\_size}$.

**Shared expert context fields for face-view** (`shared_expert/op.py`, lines 293--301):

```python
if use_face_view:
    face_tile = ttnn.Tile([FACE_HEIGHT, FACE_WIDTH])
    face_tile_desc = ttnn.TileDescriptor(FACE_HEIGHT, FACE_WIDTH, False)
    face_tile_size = face_tile.get_tile_size(data_format)
    kernel_tiles_per_k = tiles_per_k
    kernel_k_num_tiles = 1
    mcast_src_num_pages = 1
    reduce_tile_size = face_tile_size
```

---

## 6.3.9 Stages 7--9: Down Projection and Output

### Stage 7: Mcast1 (Down Projection Input)

The gated reduce output $[1, K_{\text{down}}]$ is broadcast from the sender core to all 130 cores in the grid. The mcast source is `mcast_src_cb` (CB 3), filled by TRISC during the gated reduce. The destination is `mcast_dst_cb` (CB 4), which also serves as the matmul input CB.

### Stage 7': Mcast2 (Residual/Bias)

A second mcast broadcasts the residual input $[1, N]$ from `residual_mcast_src_cb` (CB 10) to `residual_mcast_dst_cb` (CB 11). This is the bias that will be added after the down projection.

### Stage 8: Down Projection Matmul

The down projection runs on 112 matmul cores (the full 130-core grid minus 8 DRAM workers, 9 phantoms, and the sender):

$$\text{matmul\_out}_{[1, N_{\text{per\_core}}]} = \text{reduce\_out}_{[1, K_{\text{down}}]} \times W_{\text{down}}^{[K_{\text{down}}, N_{\text{per\_core}}]}$$

The 112 matmul cores are an SRAM matmul -- weights fit in L1. The grid excludes:
- 8 DRAM workers at `[(0,0), (0,3), (0,7), (0,9), (7,1), (7,4), (7,6), (7,9)]`
- 9 phantoms at column 12, rows 0--8
- 1 sender core at (12, 9)

The `UnifiedCompileTimeCoreDescriptor` sets `is_matmul_core=1` for the 112 active cores and `0` for the others.

### Stage 8': Residual Add

On the same 112 matmul cores:

$$\text{add\_out} = \text{matmul\_out} + \text{residual\_shard}$$

Each core adds its shard of the bias (from the mcast2 destination CB).

### Stage 9: Output Gather

The 112 cores send their $[1, N_{\text{per\_core}}]$ results to the sender core at (12, 9), where they are assembled into the full $[1, N]$ output.

---

## 6.3.10 Output Mcast Page Size Asymmetry

The `setup_output_mcast()` method (`moe/op.py`, lines 2143--2175) reveals an important asymmetry in the mcast that delivers the shared expert output to the routed expert's DRAM worker cores. The source and destination CBs use **different page sizes** even though the total byte count is identical:

- **Source CB** (on sender core 12,9): Uses $1 \times 32$ tile-sized pages. The total pages equal `gather_dst_num_pages = num_matmul_cores * out_w_per_core` -- e.g., $112 \times w$ tile pages.
- **Destination CB** (on DRAM cores): Uses slice-sized pages, where each page covers `width_per_core` elements. The page count equals `num_dram_worker_cores` (typically 8).

The relationship is:

$$\text{gather\_dst\_num\_pages} \times \text{tile\_size} = \text{dst\_num\_pages} \times \text{slice\_size}$$

This asymmetry exists because the DRAM cores need the data pre-sliced for their `add_cb_in1` (the routed expert's eltwise add input CB), which expects contiguous slices of the output vector rather than individual tiles. The mcast hardware streams the same raw bytes, but the sender reads them as tile-sized pages while each receiver writes them as a single large page covering its portion of the output.

---

## 6.3.11 How Routed and Shared Expert Outputs Combine

In the fused MoE kernel (`moe/op.py`), the routed and shared expert paths execute within the same program dispatch. The combination is straightforward:

$$\text{moe\_output} = \text{routed\_expert\_output} + \text{shared\_expert\_output}$$

This addition happens in stage 12 of the routed expert pipeline (Section 6.2.9). The shared expert output is stored as a HEIGHT_SHARDED tensor replicated on all DRAM matmul cores. The routed expert's eltwise add stage adds the shared expert's contribution.

**Data flow of the combination:**

```
  Shared Expert Path                    Routed Expert Path
  ------------------                    ------------------
  128-core gate/up matmul               8-core DRAM gate/up/down
       |                                      |
       v                                      v
  Gated reduce on (12,9)               down_proj output [1, K']
       |                                 on 8 DRAM cores
       v                                      |
  112-core down_proj                          |
       |                                      |
       v                                      |
  Output gather to (12,9)                     |
       |                                      |
       v                                      |
  shared_out [1, 7168]                        |
  (replicated to DRAM cores)                  |
       |                                      |
       +------------ eltwise_add -------------+
                          |
                          v
                  final_out [1, 7168]
                  (on 8 DRAM cores)
```

The shared expert output flows into `add_cb_in1` of the routed path, where each DRAM matmul core adds its corresponding slice. The `add_sender_index` per-core descriptor ensures each core reads the correct $[1, 896]$ slice from the replicated $[1, 7168]$ tensor.

---

## 6.3.12 MoE Semaphore Coordination: The 17 `MoeSem` Constants

The fused MoE kernel coordinates both paths using 17 global semaphores defined in `MoeSem` (`moe/op.py`, lines 50--71). Each semaphore mediates a specific producer-consumer relationship.

> **Cross-reference:** Chapter 4 Section 4.3.1 provides the semaphore catalog at the interface level. This section adds the temporal ordering analysis and reuse safety reasoning.

The full 17-semaphore allocation map (IDs 0--16) is documented in [Chapter 4 Section 4.3.1](../ch04_the_fused_op_library/03_moe_fused_ops.md). Here we focus on the temporal reuse safety analysis that determines why certain semaphores can be shared.

### Temporal Reuse Safety

Semaphore 0 (`MCAST_SENDER`) is reused across multiple mcast operations:
1. RMSNorm-normalized input mcast to 130-core grid
2. Index mcast to DRAM cores
3. Expert scale mcast to DRAM cores
4. down_proj input mcast to 130-core grid
5. Shared expert activation mcast
6. Shared expert down-proj input mcast

This is safe because each mcast is temporally ordered -- the sender waits for all receivers to acknowledge before initiating the next mcast. The receiver semaphores (1, 3, 5, 7, 9, 10, 11) are each dedicated to a specific mcast to prevent confusion.

### Semaphore Creation

All 17 semaphores are created by `MoeOp.create_semaphores()` and passed to the context setup. They are global semaphores (visible across the entire device grid) allocated with initial value 0:

```python
semaphores = [
    ttnn.create_global_semaphore(mesh_device, full_device_grid, 0)
    for _ in range(MoeSem.NUM_SEMAPHORES)
]
```

---

## 6.3.13 CB Reconfiguration Between Routed and Shared Expert Phases

The fused MoE kernel runs both the routed and shared expert paths in a single program dispatch. Since these paths use different core grids and CB layouts, the kernel must reconfigure circular buffers at runtime.

**The problem:** The routed expert uses CBs on the 8 DRAM-optimal cores (gate/up/down streaming matmul), while the shared expert uses CBs on the 128 A+B compute cores (KN-sliced matmul) and 112 matmul cores (down_proj). These overlapping CB assignments work because the two paths execute sequentially, but the CB hardware registers must be updated between phases.

**The solution:** `build_cb_reconfig_tensor` and `reconfig_cb_interfaces` (from `circular_buffer_utils.py`).

`build_cb_reconfig_tensor` pre-computes a CB reconfiguration payload: a tensor containing the new CB addresses, sizes, and tile formats for each core. At runtime, the kernel reads this tensor and writes the new CB configuration into the hardware registers, effectively "swapping" the CB layout without re-dispatching the program.

This is tracked in the `MoeContext` with the `reconfig_moe_cbs: bool` flag (line 119, `moe/op.py`). When True, the kernel includes CB reconfiguration logic between the routed and shared expert phases.

**What gets reconfigured:**
- CB indices that serve different purposes in each phase
- CB base addresses (switching between tensor-backed and manually-allocated buffers)
- CB tile formats (e.g., $1 \times 32$ for matmul vs. $16 \times 16$ for face-view operations)
- CB total sizes (different numbers of tiles in each phase)

When `reconfig_moe_cbs` is `False` (the typical production path), the CB layout is fixed at compile time, and all CBs are sized for the worst-case dimensions across both phases. When `True`, the kernel dynamically adjusts CB boundaries at the routed-to-shared transition point.

---

## 6.3.14 The `GatedLocalReduceDownProj` Variant

For contexts where the gated reduce feeds directly into a down projection (e.g., the standalone test path), the `gated_local_reduce_down_proj` fused op (`fused_ops/gated_local_reduce_down_proj/op.py`, 835 lines) combines the input gather, gated reduce, mcast, matmul, residual add, and output gather into a single dispatch.

**Its 7-stage pipeline:**

```
Stage   Operation               Core Set
-----   ---------------         ---------------
  1     Input gather            32 src -> (12,9)
  2     Gated reduce            (12,9) TRISC
  3     Mcast1 (down input)     (12,9) -> 130
  4     Mcast2 (residual)       (12,9) -> 130
  5     Down proj matmul        112 cores
  6     Residual add            112 cores
  7     Output gather           112 -> (12,9)
```

The input gather in stage 1 pulls tiles from 32 source cores (16 per group, from a prior KN-sliced matmul), collecting them into the sender core's group1/group2 CBs. This replaces the separate A-gather and B-gather of the standalone shared_expert op.

**CB layout (13 CBs):**

| CB | Name | Purpose |
|----|------|---------|
| 0 | `group1_cb` | Group 1 gather dest / reduce input |
| 1 | `group2_cb` | Group 2 gather dest / reduce input |
| 2 | `intermed_cb` | Intermediate (2 tiles, manual) |
| 3 | `mcast_src_cb` | Gated reduce output / mcast source |
| 4 | `mcast_dst_cb` | Mcast dest / matmul in0 |
| 5 | `matmul_in1_cb` | Down weights (tensor-backed) |
| 6 | `matmul_out_cb` | Matmul output (manual) |
| 7 | `gather_dst_cb` | Output gather dest (tensor-backed) |
| 8 | `input_src_g1_cb` | Group 1 source (tensor-backed) |
| 9 | `input_src_g2_cb` | Group 2 source (tensor-backed) |
| 10 | `residual_add_mcast_src_cb` | Bias mcast source |
| 11 | `residual_add_mcast_dst_cb` | Bias mcast dest |
| 12 | `residual_add_out_cb` | Residual add output |

The `_GatedReduceDownProjContext` dataclass mirrors `_SharedExpertContext` with additional fields for input gather source grids (`g1_source_grid`, `g2_source_grid`) and their semaphores (`ig_g1_receiver_semaphore_id`, `ig_g2_receiver_semaphore_id`).

---

## 6.3.15 Dense Layer Shared Expert Reuse

For dense layers (0--2), the same `SharedExpertOp` code handles the MLP. The difference is in weight preparation: dense layer MLP weights are the "first expert" of the full MLP weight tensor (which contains 9 experts of width 2048). The `get_tt_mlp_shared_expert_weights` function (`blitz_decode_weights.py`, line 1427) slices out the first 2048 columns for gate/up and first 2048 rows for down, then creates the same `OverlappedTensor` layout as the MoE shared expert.

This reuse means the dense MLP path is literally the same kernel as the shared expert -- a powerful example of the design philosophy of maximizing code reuse across layer types.

---

## 6.3.16 Timing and Parallelism Between Routed and Shared Expert Paths

In the fused MoE kernel, the routed and shared expert paths are not fully parallelized -- they share the sender core (12, 9) and the 130-core mcast grid. However, significant spatial parallelism exists:

```
Time ----------------------------------------------------->

Sender Core (12,9):
  RMSNorm -> Residual Mcast -> Input Mcast -> Gate MM/Gather/Gate
  -> Index/Scale Mcast -> [wait for down_proj_gather]
  -> down_proj Mcast -> [wait for output]
  ... then shared expert stages on sender ...

DRAM Cores (8):
  [receive mcast] -> gate_proj -> up_proj -> mul -> [gather send]
  -> [receive down_proj mcast] -> down_proj -> add -> [done]

A-Cores (64):
  [receive act mcast] -> gate matmul -> [gather send] -> [idle]

B-Cores (64):
  [receive act mcast] -> up matmul -> [gather send] -> [idle]

112 Matmul Cores:
  [receive mcast1/mcast2] -> down_proj -> residual_add -> [gather send]
```

The sender core's role as both mcast sender and gather receiver creates a natural serialization point. The shared expert's activation mcast can only begin after the routed expert's input mcast completes. However, once the shared expert's 128-core gate/up matmuls start, they can overlap with the routed expert's DRAM-streaming matmuls running on the 8 DRAM bank cores -- these are disjoint core sets. The temporal overlap of shared expert computation with routed expert DRAM streaming is the primary source of MoE layer throughput gain.

---

## 6.3.17 End-to-End MoE Layer Summary

Combining Sections 6.1 through 6.3, a complete MoE layer execution:

```
Input: hidden_state [1, 7168] on sender core

1. RMSNorm on sender core
2. Mcast normalized input to 130-core grid

-- Routed Expert Path ------------------------------------------------
3. Gate matmul (8 cores, SRAM) -> 256 scores
4. Gate gather to sender -> [16, 16]
5. DeepSeek MoE gate -> top-8 indices + scores
6. Index mcast + Scale mcast to 8 DRAM cores
7. gate_proj (DRAM streaming + SiLU, 8 cores)
8. up_proj (DRAM streaming, 8 cores)
9. eltwise_mul (SiLU(gate) x up x scale, 8 cores)
10. Gather fused output to sender
11. Mcast fused output to 8 DRAM cores
12. down_proj (DRAM streaming, 8 cores)

-- Shared Expert Path ------------------------------------------------
13. Activation mcast to 128 compute cores
14. Gate matmul (64 A-cores, KN-sliced SRAM)
15. Up matmul (64 B-cores, KN-sliced SRAM)
16. Gate gather + Up gather to sender
17. Gated local reduce on sender: SiLU(sum(gate)) x sum(up)
18. Mcast reduce output to 112 cores
19. Down proj matmul (112 cores, SRAM) + residual add
20. Output gather to sender

-- Combination -------------------------------------------------------
21. routed_down_proj_out + shared_expert_out (eltwise add on 8 DRAM cores)
22. [Optional] ReduceToOne across 4x2 mesh

Output: moe_output [1, 7168] on 8 DRAM cores (or ROOT1 after reduce)
```

---

[Previous: 6.2 Routed Expert Computation](02_routed_expert_computation.md) | [Next: 6.4 DRAM Streaming Matmul Deep Dive](04_dram_streaming_matmul_deep_dive.md)
