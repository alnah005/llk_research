# 3.1 Compute Micro-Ops

Compute micro-ops perform the numerical heavy lifting of the DeepSeek V3 model -- matrix multiplications, normalizations, activations, reductions, and token selection. Each runs primarily on TRISC (the tensor compute RISC-V) while NCRISC and BRISC handle sharded-buffer signaling or remain no-ops. All ops in this section follow the unified kernel pattern from Chapter 2: a single `.cpp` source file, `deepseek_b1_ops::` namespace structs, and `ttnn.generic_op` dispatch.

> **Stability note on "Composed Into" sections.** The CB index assignments listed in each op's "Composed Into" section reflect the codebase at the time of writing and are subject to change when fused ops are refactored. Treat them as navigational aids, not as stable API contracts.

---

## 3.1.1 matmul

**Source:** `micro_ops/matmul/op.py` | **Kernel header:** `unified_kernels/matmul.hpp`

### Purpose

Single-tile-height matrix multiplication, hyper-optimized for the decode path where $M = 1$ tile. Used for attention projections (Q/K/V) and small linear layers where weights fit in L1.

### Computation

$$\text{output}_{[1,N]} = A_{[1,K]} \times B_{[K,N]} \quad [\text{optional: } \sigma(\cdot) \text{ or SiLU}(\cdot)]$$

The `FusedActivation` enum (`NONE=0, SIGMOID=1, SILU=2`) controls post-matmul activation. The Python class mirrors this with `ACTIVATION_NONE/SIGMOID/SILU` constants.

### Data Flow Diagram

```
  L1 (sharded)         L1 (sharded)         L1 (sharded)
  +-----------+        +-----------+        +-----------+
  | input_a   |        | input_b   |        | output    |
  | [1, K]    |        | [K, N]    |        | [1, N]    |
  +-----+-----+        +-----+-----+        +-----+-----+
        |                     |                     ^
        v                     v                     |
  CB 0 (in0)           CB 1 (in1)            CB 2 (out)
  [aliased to           [aliased to           [aliased to
   tensor shard]         tensor shard]         tensor shard]
        |                     |                     ^
        +----------+----------+                     |
                   |                                |
            NCRISC / BRISC                          |
            No-op (sharded buffer                   |
            signaling handled by                    |
            generic_op framework)                   |
                   |                                |
                   v                                |
             TRISC (compute)                        |
             MOP-looped matmul:                     |
               for each of N output cols:           |
                 zero accum                         |
                 for k in 0..K_tiles:               |
                   matmul_tiles(CB0[k], CB1[k*N+n]) |
                 optional: sigmoid/silu(dst)        |
                 pack_tile -> CB2  -----------------+
```

### CB Layout

| CB | Index | Role | Backed by |
|----|-------|------|-----------|
| in0 | 0 | Activation $A$ -- $K$ tiles | Sharded tensor |
| in1 | 1 | Weights $B$ -- $K \times \text{out\_w}$ tiles | Sharded tensor |
| out | 2 | Output -- $\text{out\_w}$ tiles | Sharded tensor |

### Compile-Time Args

| Arg Name | RISC | Type | Description |
|----------|------|------|-------------|
| `matmul_in0` | NCRISC, TRISC | `uint32_t` | CB index for input A (0) |
| `matmul_in1` | NCRISC, TRISC | `uint32_t` | CB index for input B (1) |
| `matmul_out` | TRISC | `uint32_t` | CB index for output (2) |
| `matmul_k_num_tiles` | NCRISC, TRISC | `uint32_t` | Inner-dimension loop bound ($K / \text{tile\_width}$) |
| `matmul_out_w` | NCRISC, TRISC | `uint32_t` | Output width in tiles ($N_{\text{shard}} / \text{tile\_width}$) |
| `matmul_transpose` | TRISC | `uint32_t` | Transpose B before multiply (0 or 1) |
| `matmul_fused_activation` | TRISC | `uint32_t` | Post-matmul activation (0/1/2) |
| `is_active_core` | All | `uint32_t` | Guard for inactive cores |

### Runtime Args

None. All parameters are compile-time constants.

### RISC Roles

| Processor | Role |
|-----------|------|
| NCRISC | No-op. CB setup is handled externally via `setup_sharded_buffer`. |
| BRISC | No-op. |
| TRISC | MOP-based matmul inner loop along $K$, optional fused activation, pack to CB2. |

### Key Implementation Detail: MOP Inner Loop and Fused Activation on PACK Thread

The kernel uses `custom_mm_block`, an LLK that programs the Tensix MOP (Micro-Operation Processor) for the matmul inner loop:

```cpp
custom_mm_block_init_short<transpose, split_acc=true, dense_packing=true>(
    args.in0, args.in1, args.out, out_w);
custom_mm_block<finalize=true, read_transposed>(
    args.in0, args.in1, 0, 0, 0, args.k_num_tiles, out_w);
```

The `custom_mm_block` call programs a single MOP sequence that iterates over all $K$ tiles and all $N$ output columns internally, avoiding per-tile function call overhead. Template parameters:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `transpose` | Configurable | Whether to transpose in1 tiles |
| `split_acc` | `true` | Use half the dest registers per accumulation, enabling ping-pong between math and pack |
| `dense_packing` | `true` | Pack output tiles contiguously in dest registers |
| `finalize` | `true` | Apply dest accumulation finalization after the last $K$ tile |

**Without fused activation (batch path):** All `out_w` tiles are computed in a single `custom_mm_block` call, then packed in a tight loop -- one `tile_regs_acquire/release` pair for all tiles.

**With fused activation (per-tile path):** The output loop switches to per-tile processing. For each output tile:

1. `tile_regs_acquire` -- claim dest register.
2. `custom_mm_block` -- compute one output tile (all $K$ iterations).
3. `tile_regs_commit` -- signal math is done.
4. `TTI_SEMWAIT` -- stall PACK thread until MATH signals dest ready.
5. `PACK(llk_math_eltwise_unary_sfpu_silu<...>)` or `PACK(llk_math_eltwise_unary_sfpu_sigmoid<...>)` -- run SFPU activation **on the PACK thread**.
6. `pack_tile(0, args.out, w)` -- write activated tile to output CB.
7. `tile_regs_release` -- free dest register.

The key insight is step 5: the SFPU normally runs on the MATH thread, but `PACK(...)` forces it to execute on the PACK thread instead, overlapping with the next tile's MATH computation. For $1 \times 32$ tiny tiles, the SFPU processes 2 iterations (two 16-row faces).

### Compute Config

- Math fidelity: `LoFi` for maximum throughput
- FP32 accumulation: configurable via `fp32_dest_acc_en`
- `dst_full_sync_en` tracks `fp32_dest_acc_en`

### Composed Into

- **pre_sdpa** -- q_a_proj matmul (Stage 1): CB0 input, CB1 weights, CB2 output.
- **post_sdpa** -- kv_b2 matmul4 (Stage 1): CB0/CB1 input pair, CB2 output. o_proj matmul5 (Stage 3): CB4 input from mcast, CB5 weights, CB6 output.
- **lm_head_sampling** -- Vocab projection: CB1 mcast destination as matmul in0, CB2 vocab weights, CB16 output.
- **moe** -- Gate matmul: input from mcast CB as in0, gate weights as in1.
- **moe_routed_expert** -- Routing matmul with `matmul_fused_activation = ACTIVATION_SIGMOID`.
- **down_proj** -- Down projection: CB1 mcast destination as in0, CB2 weights, CB3 output.
- **shared_expert** -- Gate/Up matmul: CB9 activation mcast destination as in0, CB13 weights, CB14 output.
- **gated_local_reduce_down_proj** -- Down proj stage: CB4 mcast destination as in0, CB5 weights, CB6 output.

---

## 3.1.2 rmsnorm

**Source:** `micro_ops/rmsnorm/op.py` | **Kernel header:** `unified_kernels/rmsnorm.hpp`

### Purpose

Root-Mean-Square normalization, used at every transformer block boundary and before attention/FFN projections. Runs on a single core (`CoreCoord(0,0)`) because the full hidden dimension fits in one shard.

### Computation

$$\text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon}, \quad \text{output} = \frac{x}{\text{RMS}(x)} \cdot \gamma$$

The Python op pre-computes $\text{inv\_sqrt\_numel} = 1/\sqrt{n}$ and packs both it and $\epsilon$ as `float_to_uint32` runtime args passed to TRISC.

### Data Flow Diagram

```
   L1 (sharded)          L1 (sharded)          L1 (sharded)
   +-----------+         +-----------+         +-----------+
   |  input    |         |  gamma    |         |  output   |
   |  [1, D]   |         |  [1, D]   |         |  [1, D]   |
   +-----+-----+         +-----+-----+         +-----+-----+
         |                      |                      ^
         v                      v                      |
   CB 0 (input)           CB 1 (gamma)           CB 2 (output)
   [reinterpreted          [reinterpreted         [reinterpreted
    as 32x32 tiles]         as 32x32 tiles]        as 32x32 tiles]
         |                      |                      ^
         v                      v                      |
               TRISC (compute)                         |
               Phase 1: sum(x^2) via mul_reduce_scalar |
               Phase 2: rsqrt(mean + eps)              |
               Phase 3: broadcast multiply by 1/RMS    |
               Phase 4: multiply by gamma              |
               pack_tile -> CB2  ----------------------+
```

### Tile-Size Autodetection

The Python `op()` inspects `input_shape[1]`:

```
is_16x32_tile = (input_shape[1] // 32) % 32 != 0
```

- **7168 elements** (hidden dim): $7168 / 32 = 224$ tiles, $224 \% 32 = 0$ -- full 32x32 tiles (4 faces, 7 tiles).
- **1536 elements** (MLA compressed dim): $1536 / 32 = 48$, $48 \% 32 = 16 \neq 0$ -- half 16x32 tiles (2 faces, 3 tiles).
- **512 elements**: $512 / 32 = 16$, $16 \% 32 = 16 \neq 0$ -- half 16x32 tiles (2 faces, 1 tile).

CB `page_size` and `tile` descriptors are overwritten with the interpreted tile size and format.

### CB Layout

| CB | Index | Role | Backed by |
|----|-------|------|-----------|
| input | 0 | Input activations | Sharded tensor (tile descriptors overridden) |
| gamma | 1 | Weight (scale) tensor | Sharded tensor (tile descriptors overridden) |
| output | 2 | Normalized output | Sharded tensor (tile descriptors overridden) |

### Compile-Time Args

| Arg Name | RISC | Type | Description |
|----------|------|------|-------------|
| `rmsnorm_input_cb` | NCRISC, TRISC | `uint32_t` | Input CB index (0) |
| `rmsnorm_gamma_cb` | NCRISC, TRISC | `uint32_t` | Gamma CB index (1) |
| `rmsnorm_output_cb` | TRISC | `uint32_t` | Output CB index (2) |
| `rmsnorm_num_tiles` | NCRISC, TRISC | `uint32_t` | Number of interpreted tiles |
| `rmsnorm_num_faces` | NCRISC | `uint32_t` | Faces per tile (2 or 4) |
| `rmsnorm_fp32_acc` | TRISC | `uint32_t` | Enable FP32 accumulation |
| `rmsnorm_rsqrt_fast_approx` | TRISC | `uint32_t` | Use fast SFPU rsqrt path |

### Runtime Args

| RISC | Args | Description |
|------|------|-------------|
| TRISC | `[epsilon_packed, scalar_packed]` | $\epsilon$ and $1/\sqrt{n}$ as uint32 bit-patterns |

### RISC Roles

| Processor | Role |
|-----------|------|
| NCRISC | Signals sharded buffers ready for CB0 and CB1. |
| BRISC | No-op. |
| TRISC | Four-phase compute pipeline (see below), pack to CB2. |

### Key Implementation Detail: Four-Phase Compute Pipeline

The TRISC kernel executes four distinct phases. Between phases, tiles remain in the dest register -- no intermediate CB spills:

**Phase 1 -- Squared Reduce:**
```cpp
mul_reduce_scalar_tile<PoolType::SUM>(input_cb, input_cb, num_tiles, args.scalar);
```
Computes $\sum x_i^2 / n$ across all `num_tiles` in a single LLK call using `scalar = 1/\sqrt{n}$.

**Phase 2 -- Add Epsilon and Reciprocal Square Root:**
```cpp
add_rsqrt_tile<rsqrt_fast_approx, VectorMode::RC_custom, 1>(0, args.epsilon);
```
Adds $\epsilon$ and computes $1/\sqrt{\text{mean}(x^2) + \epsilon}$ in one operation. `rsqrt_fast_approx` trades 1 bit of precision for a faster SFPU path.

**Phase 3 -- Broadcast Multiply (Normalization):**
```cpp
rmsnorm_mul_bcast_scalar_reuse_tiles<num_tiles, true>(input_cb, 0, 0, 0);
```
Broadcasts the scalar $1/\text{RMS}(x)$ across all tiles while multiplying element-wise with input. Tiles remain in DST registers.

**Phase 4 -- Gamma Multiply:**
```cpp
binary_dest_reuse_tiles<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(gamma_cb, i, i);
```
Multiplies normalized values (in dest, moved to SRCA) by gamma (from `gamma_cb`). The `DEST_TO_SRCA` reuse type means dest contents serve as source A while gamma tiles serve as source B. Gamma is waited once and never popped -- it persists across all decode iterations.

### Compute Config

- Math fidelity: `LoFi`
- `rsqrt_fast_approx`: configurable from Python

### Composed Into

- **broadcast_rms** -- The entire fused op is "ccl_broadcast + rmsnorm." TRISC runs rmsnorm with CB0 input, CB1 gamma, CB2 output. First stage of every decoder layer.
- **pre_sdpa** -- Three RMSNorm instances: input layernorm (CB0/CB1/output, 32x32 tiles for 7168-wide), q_a_norm (CB7/CB6/CB9, 16x32 tiles for 1536-wide), KV RMSNorm (CB25/CB26/CB27 on kv_cache_branch core).
- **kv_cache_branch** -- KV RMSNorm: normalizing gathered KV projection before cache update.
- **moe** -- RMSNorm on sender core: normalizes residual input before mcast to expert cores.
- **lm_head_sampling** -- RMSNorm on sender core (CB0/CB7/CB8), output becomes mcast source for vocab projection.

---

## 3.1.3 rope

**Source:** `micro_ops/rope/op.py` | **Kernel header:** `unified_kernels/rope.hpp`

### Purpose

Rotary Position Embedding using Meta-style interleaving. Applied to Q and K heads before attention.

### Computation

$$\text{output} = (x \cdot \cos\theta) + (\text{rotate\_half}(x) \cdot \sin\theta)$$

where `rotate_half` for Meta-style interleaving transforms $[a_0, a_1, a_2, a_3, \ldots]$ into $[-a_1, a_0, -a_3, a_2, \ldots]$. On hardware, this is implemented as a matrix multiplication: $\text{rotate\_half}(x) = x \times T$ where $T$ is the `trans_mat` transformation matrix.

### Data Flow Diagram

```
   CB 0 (input)    CB 1 (cos)    CB 2 (sin)    CB 3 (trans_mat)
   [sharded,       [DRAM read    [DRAM read    [sharded,
    n_heads x       per pos_id]   per pos_id]   fixed matrix]
    head_dim]           |              |              |
        |               v              v              |
        |         NCRISC: reads cos/sin from DRAM     |
        |         using position_ids_tensor           |
        |         to index into cos/sin tables        |
        |               |              |              |
        v               v              v              v
                    TRISC (compute)
                    Step 1: rotated = input @ trans_mat -> CB 24
                    Step 2: sin_prod = rotated * sin   -> CB 26
                    Step 3: cos_prod = input * cos     -> CB 25
                    Step 4: out = cos_prod + sin_prod  -> CB 16 (output)
```

### CB Layout

| CB | Index | Role | Backed by | Lifetime |
|----|-------|------|-----------|----------|
| input | 0 | Input activations | Sharded tensor | Per-head (popped after step 3) |
| cos | 1 | Cosine values | L1 (NCRISC fills from DRAM) | Per-iteration (popped after all heads) |
| sin | 2 | Sine values | L1 (NCRISC fills from DRAM) | Per-iteration (popped after all heads) |
| trans_mat | 3 | Rotation matrix (1 tile) | Sharded tensor | Persistent (never popped) |
| output | 16 | RoPE result | Sharded tensor | Per-head |
| rotated_interm | 24 | Intermediate: $x \times T$ | L1 (not tensor-backed) | Per-head |
| cos_interm | 25 | Intermediate: $x \cdot \cos$ | L1 (not tensor-backed) | Per-head |
| sin_interm | 26 | Intermediate: rotated $\cdot \sin$ | L1 (not tensor-backed) | Per-head |

### Compile-Time Args

| Arg Name | RISC | Type | Description |
|----------|------|------|-------------|
| `in_cb` | NCRISC, TRISC | `uint32_t` | Input CB index (0) |
| `cos_cb`, `sin_cb` | NCRISC, TRISC | `uint32_t` | Cos/sin CB indices (1, 2) |
| `trans_mat_cb` | NCRISC, TRISC | `uint32_t` | Transformation matrix CB index (3) |
| `cos_tensor_address`, `sin_tensor_address` | NCRISC | `uint32_t` | DRAM base addresses |
| `position_ids_tensor_address` | NCRISC | `uint32_t` | Position ID tensor L1 address |
| `Wt`, `Ht`, `total_Wt` | NCRISC, TRISC | `uint32_t` | Width tiles, head tiles, full row width |
| `cos_sin_page_size` | NCRISC | `uint32_t` | Page size for DRAM reads |

**Per-core:** `start_tile_offset` -- each core reads its slice of cos/sin from DRAM. Set via `PerCoreCompileTimeDescriptor`.

### Runtime Args

None. Position is read dynamically from the `position_ids_tensor` in L1.

### RISC Roles

| Processor | Role |
|-----------|------|
| NCRISC | Reads cos and sin from DRAM interleaved tensors at `position_id * total_Wt + start_tile_offset`. |
| BRISC | No-op. |
| TRISC | Four-step compute per head: matmul(x, trans_mat), eltwise_mul(rotated, sin), eltwise_mul(x, cos), add. |

### Key Implementation Detail

- The `PerCoreCompileTimeDescriptor` for `start_tile_offset` distributes different width slices of the cos/sin DRAM tensor to different cores -- a pattern unique to RoPE.
- Tile dimensions are dynamically computed from `num_q_heads_per_core`: tile height = `shard_shape[0]` (the number of Q heads on this core), enabling tiles like `(12, 32)`.
- Cos and sin are shared across all heads (popped once after the $H_t$ loop), while input and intermediates cycle per head.

### Compute Config

- Math fidelity: `LoFi`
- FP32 accumulation: disabled

### Composed Into

- **pre_sdpa** -- QRoPE stage: applies RoPE to Q rope heads after matmul2 splits Q into Qnope and Qrope. Also applied to K rope heads in the kv_cache_branch path.
- **kv_cache_branch** -- K-RoPE: applies RoPE to K rope projection before KV cache update.

---

## 3.1.4 kn_sliced_matmul

**Source:** `micro_ops/kn_sliced_matmul/op.py` | **Kernel header:** `unified_kernels/kn_sliced_matmul.hpp`

### Purpose

K-sliced parallel matrix multiplication where multiple cores each compute a partial matmul using different K-ranges of a shared activation buffer. Used for the gate/up projections in the MoE FFN, where the activation is broadcast (via mcast) and each core holds a different K-slice of the weights.

### Computation

$$\text{output}_{[1,\text{out\_w}]} = A_{[k\_\text{offset}:k\_\text{offset}+k\_\text{per\_core}]} \times W_{[k\_\text{per\_core}, \text{out\_w}]}$$

The partial products from all cores must be reduced (via `local_reduce` or `gather_reduce`) to produce the final result.

### Data Flow Diagram

```
                     Core 0                          Core 1
              k_offset=0, k_per_core=K/2       k_offset=K/2, k_per_core=K/2
   +----------------------------------+  +----------------------------------+
   | CB 0 (act): full [1, K] sharded  |  | CB 0 (act): full [1, K] sharded  |
   | CB 1 (wt):  [K/2, N] sharded    |  | CB 1 (wt):  [K/2, N] sharded    |
   | CB 2 (out): [1, N] sharded      |  | CB 2 (out): [1, N] sharded      |
   +----------------------------------+  +----------------------------------+
          |                                       |
          v                                       v
     TRISC compute:                          TRISC compute:
     custom_mm_block(act, wt,                custom_mm_block(act, wt,
       k_offset=0, k_per_core=K/2)            k_offset=K/2, k_per_core=K/2)
```

### CB Layout

| CB | Index | Role | Backed by |
|----|-------|------|-----------|
| act | 0 | Full shared activation (all K tiles) | Sharded tensor (replicated) |
| weights | 1 | Per-core K-slice of weights | Sharded tensor |
| out | 2 | Partial product output | Sharded tensor |

### Compile-Time Args

| Arg Name | RISC | Type | Description |
|----------|------|------|-------------|
| `act_cb` | NCRISC, TRISC | `uint32_t` | Activation CB index (0) |
| `act_num_pages` | NCRISC | `uint32_t` | Total activation pages |
| `weights_cb` | NCRISC, TRISC | `uint32_t` | Weights CB index (1) |
| `weights_num_pages` | NCRISC | `uint32_t` | Weight pages |
| `out_cb` | TRISC | `uint32_t` | Output CB index (2) |
| `k_per_core` | TRISC | `uint32_t` | K tiles for this core |
| `act_total_tiles` | TRISC | `uint32_t` | Total activation tiles |
| `out_w` | TRISC | `uint32_t` | Output width in tiles |

**Per-core:** `k_offset` -- tile offset into the activation tensor for this core's K-slice, set via `PerCoreCompileTimeDescriptor`.

### Runtime Args

None -- all CB indices and dimensional parameters are compile-time args.

### RISC Roles

| Processor | Role |
|-----------|------|
| NCRISC | Signals sharded buffers ready. |
| BRISC | No-op. |
| TRISC | MOP-based matmul inner loop, indexing act at `k_offset`, pack partial result. |

### Key Implementation Detail

The critical innovation is the `k_offset` parameter:

```cpp
custom_mm_block<finalize, false>(
    args.act_cb, args.weights_cb,
    args.k_offset,  // per-core tile offset into act_cb
    0, 0,
    args.k_per_core, out_w);
```

All cores wait for the same total number of activation tiles (`act_total_tiles`), but each reads only its own K-slice. Properties: `transpose=false`, `split_acc=true`, `dense_packing=true`. `pop_act` defaults to `true`, `pop_weights` defaults to `false` -- enabling weight reuse across loop iterations.

### Compute Config

- Math fidelity: `LoFi`
- FP32 accumulation: configurable

### Composed Into

- **pre_sdpa** -- Matmul1 (q_a_proj): 8 cores each process `K/8` tiles with distinct `k_offset` values. CB0 holds full activation, CB1 per-core weight slices, CB2 partial output.
- **moe_routed_expert** -- Gate/Up KN-sliced matmul: 128 cores compute K-sliced gate/up projections.

---

## 3.1.5 local_reduce

**Source:** `micro_ops/local_reduce/op.py` | **Kernel header:** `unified_kernels/local_reduce.hpp`

### Purpose

Accumulates $N$ tiles from a single CB into one output tile, with optional SiLU activation. Used after K-sliced matmul to sum partial products across the K dimension.

### Computation

$$\text{output} = \begin{cases}
\text{SiLU}\left(\sum_{i=0}^{N-1} \text{in}[i]\right) & \text{if apply\_silu} \\
\sum_{i=0}^{N-1} \text{in}[i] & \text{otherwise}
\end{cases}$$

### Data Flow Diagram

```
   CB 0 (input): N tiles stacked
   +--------+--------+--------+-----+--------+
   | tile 0 | tile 1 | tile 2 | ... | tile N |
   +--------+--------+--------+-----+--------+
        |
        v
   TRISC (compute):
   DST register zeroed at startup (acc_to_dest)
   for i in 0..N step 2:
     add_tiles(CB0[i], CB0[i+1])   // pairwise add, accumulate in DST[0]
   if apply_silu:
     silu_tile(DST[0])
   pack_tile(DST) -> CB 1 (output)
        |
        v
   CB 1 (output): 1 tile
```

### CB Layout

| CB | Index | Role | Backed by |
|----|-------|------|-----------|
| in | 0 | $N$ input tiles to reduce | Sharded tensor |
| out | 1 | 1 output tile | Sharded tensor |

### Compile-Time Args

| Arg Name | RISC | Type | Description |
|----------|------|------|-------------|
| `local_reduce_in_cb` | NCRISC, TRISC | `uint32_t` | Input CB index (0) |
| `local_reduce_out_cb` | TRISC | `uint32_t` | Output CB index (1) |
| `local_reduce_num_tiles` | NCRISC, TRISC | `uint32_t` | Number of tiles to reduce |
| `local_reduce_apply_silu` | TRISC | `uint32_t` | Enable SiLU after reduction |

### Runtime Args

None.

### RISC Roles

| Processor | Role |
|-----------|------|
| NCRISC | Signals sharded buffer for CB0. |
| BRISC | No-op. |
| TRISC | Pairwise accumulation using `acc_to_dest`, optional SiLU via SFPU, pack. |

### Key Implementation Detail: `acc_to_dest` Accumulation and Face-View Optimization

The key optimization is `acc_to_dest = true`:

```cpp
add_tiles_init(args.in_cb, args.in_cb, true /* acc_to_dest */);
for (uint32_t i = 0; i < num_tiles; i += 2) {
    add_tiles(args.in_cb, args.in_cb, i, i + 1, 0);
}
```

With `acc_to_dest = true`, the operation is: $\text{DST}[0] = \text{A}[i] + \text{B}[i+1] + \text{DST}[0]$. After the loop, DST[0] contains $\sum_{i=0}^{N-1} \text{tile}[i]$. Input tiles are consumed in pairs (stride 2), requiring `num_tiles` to be even.

**Face-view optimization.** When input tiles are smaller than a face (e.g., 16 tiles of $1 \times 32$ = 512 elements):

```
total_elements = 16 * 32 = 512
512 / 256 = 2 faces of [16,16]
```

The Python `can_use_face_view()` checks: (a) tiles are sub-face sized, (b) evenly divide into faces, (c) total forms an even number of faces. When enabled, CB tile descriptors are overridden to $16 \times 16$ faces and `kernel_num_tiles` becomes the face count (2 instead of 16), dramatically reducing add-tiles loop iterations.

### Compute Config

- Math fidelity: `HiFi4` (higher precision for reduction)
- `math_approx_mode = apply_silu` (approximation only needed for SiLU)

### Composed Into

- **pre_sdpa** -- After kn_sliced_matmul (matmul1), 8 partial products are gathered and reduced.
- **gated_local_reduce** -- Two local_reduce calls (group1 with SiLU, group2 without) plus eltwise multiply.
- **shared_expert** -- Includes gather + gated_local_reduce (64 gate-path tiles reduced with SiLU, 64 up-path tiles reduced without, then multiplied).
- **gated_local_reduce_down_proj** -- Gated reduce stage (CB0/CB1 inputs, CB3 output).
- **moe_routed_expert** -- After K-sliced gate/up matmuls, gated local reduce: gate path with SiLU, up path without, then multiply.

---

## 3.1.6 eltwise_add

**Source:** `micro_ops/eltwise_add/op.py` | **Kernel header:** `unified_kernels/eltwise_add.hpp`

### Purpose

Element-wise addition with per-core indexing into a replicated tensor. Used after `down_proj` in MoE to add the fused residual.

### Computation

$$\text{output}_{\text{core}_i} = \text{down\_proj\_out}_{\text{core}_i} + \text{fused\_add}[i \cdot 896 : (i+1) \cdot 896]$$

### Data Flow Diagram

```
   Core i (sender_index = i)
   +--------------------------------------------------+
   |                                                  |
   | CB 0 (in0): down_proj_out, 896 elements          |
   | Backed by WIDTH_SHARDED tensor shard             |
   |                                                  |
   | CB 1 (in1): fused_add, 7168 elements (full)      |
   | Backed by HEIGHT_SHARDED (replicated) tensor     |
   | TRISC updates read ptr to offset = i * 1792 bytes|
   |                                                  |
   | CB 2 (out): result, viewed as 1 tile of 32x32    |
   +--------------------------------------------------+
```

### CB Layout (32x32 aliased view)

| CB | Index | Role | Backed by |
|----|-------|------|-----------|
| in0 | 0 | down_proj output (896 elements, viewed as 1 tile of 32x32) | WIDTH_SHARDED tensor |
| in1 | 1 | fused_add (full 7168 elements, replicated on each core) | HEIGHT_SHARDED tensor |
| out | 2 | output (32x32 tile, 2048 bytes) | Sharded tensor |

The 32x32 tile view is an aliasing trick: 896 bf16 elements = 1792 bytes, interpreted as one "tile" of 32x32 (2048 bytes) with the last 128 elements as garbage padding.

### Compile-Time Args

| Arg Name | RISC | Type | Description |
|----------|------|------|-------------|
| `add_cb_in0` | All | `uint32_t` | Down_proj output CB (0) |
| `add_cb_in1` | All | `uint32_t` | Fused_add CB (1) |
| `add_cb_out` | All | `uint32_t` | Output CB (2) |
| `add_num_tiles` | All | `uint32_t` | Tiles to add (1 in 32x32 view) |
| `add_slice_size_bytes` | All | `uint32_t` | 1792 (= 896 * 2) |
| `add_cb_in0_wait_tiles` | All | `uint32_t` | Wait count for in0 |
| `add_cb_in1_wait_tiles` | All | `uint32_t` | Wait count for in1 |

**Per-core:** `add_sender_index` -- core index for indexing into fused_add, set via `PerCoreCompileTimeDescriptor`.

### Runtime Args

None.

### RISC Roles

All three RISC processors receive the same compile-time args (unusual for a unified kernel):

| Processor | Role |
|-----------|------|
| NCRISC | Signals sharded buffers ready. |
| BRISC | Signals sharded buffers ready. |
| TRISC | Updates CB1 read pointer via `sender_index`, performs eltwise add, packs output. |

### Key Implementation Detail: CB Read Pointer Manipulation

Rather than copying the relevant 896-element slice of `fused_add`, the kernel manipulates the CB read pointer directly:

```cpp
constexpr uint32_t offset_aligned = sender_index * slice_size_bytes / L1_ALIGNMENT;
uint32_t cb_in1_base_rd_ptr = 0;
UNPACK(({
    cb_in1_base_rd_ptr = get_local_cb_rd_ptr(cb_in1);
    update_local_cb_rd_ptr(cb_in1, cb_in1_base_rd_ptr + offset_aligned);
}));
```

The `UNPACK(...)` wrapper ensures execution in the unpacker context (TRISC thread 0). After compute, the read pointer is restored:

```cpp
UNPACK(({ update_local_cb_rd_ptr(cb_in1, cb_in1_base_rd_ptr); }));
```

This zero-copy indexing avoids any data movement. The offset is 16-byte aligned (896 * 2 = 1792 bytes, which is 112 * 16). `PopInputs` and `PopOutput` are template booleans for controlling CB lifecycle in fused contexts.

### Compute Config

- Math fidelity: `HiFi4`

### Composed Into

- **moe** -- Final eltwise add stage (stage 12): each core adds down_proj result to corresponding slice of shared residual.
- **down_proj** -- Residual add stage: CB3 matmul output + shard of CB6 residual mcast destination, result to CB7.
- **gated_local_reduce_down_proj** -- Residual add stage: CB6 matmul output + shard of CB11 residual mcast destination, result to CB12.

---

## 3.1.7 sampling / argmax

**Source:** `micro_ops/sampling/op.py` | **Kernel header:** `unified_kernels/argmax.hpp`

### Purpose

Token selection at the end of the decode step. Currently implements $k = 1$ (argmax) with a multi-phase reduction that scales from a single device to an 8-device mesh.

### Computation

$$\text{selected\_index} = \arg\max_{i} \text{scores}_i \quad \text{(ties broken by lowest index)}$$

### Architecture (Single Device)

```
   Core 0           Core 1           ...          Core N-1
   [local argmax]   [local argmax]                [local argmax]
        \                |                            /
         \               |                           /
          \              v                          /
           +----> Final Core (semaphore wait) <----+
                  [global argmax over N winners]
                  [write output_index_tensor]
```

### CB Layout

| CB | Index | Role | Backed by |
|----|-------|------|-----------|
| winner | 0 (single) / 2 (mesh) | 8-byte winner payload (bf16 score + uint32 index) | L1 |
| gather | 1 (single) / 3 (mesh) | `winner_page_bytes * num_senders` -- gather buffer on final core | L1 |

### Compile-Time Args

| Arg Name | RISC | Type | Description |
|----------|------|------|-------------|
| `sampling_num_values` | NCRISC | `uint32_t` | Scores per core shard |
| `sampling_winner_page_bytes` | NCRISC | `uint32_t` | Winner payload size (8 bytes) |
| `sampling_num_senders` | NCRISC | `uint32_t` | Total sender cores |
| `sampling_expected_remote_incs` | NCRISC | `uint32_t` | Semaphore wait target |
| `sampling_winner_cb`, `sampling_gather_cb` | NCRISC | `uint32_t` | CB indices |
| `sampling_receiver_semaphore_id` | NCRISC | `uint32_t` | Semaphore ID |
| `sampling_mesh_mode` | NCRISC | `uint32_t` | 0 = single device, 1 = mesh |

**Per-core:** `sampling_sender_idx`, `sampling_is_active_core`, `sampling_is_final_core`, `sampling_mesh_sender_core` -- set via `PerCoreCompileTimeDescriptor`.

### Runtime Args

| RISC | Core | Args | Description |
|------|------|------|-------------|
| NCRISC | Non-final | `[scores_addr, indices_addr, output_addr]` | Shard addresses |
| NCRISC | Final | `[scores_addr, indices_addr, output_addr, gather_base_addr]` | Plus gather buffer |

### RISC Roles

| Processor | Role |
|-----------|------|
| NCRISC | Phase 1: per-core local argmax via linear scan. Phase 2 (final core only): global argmax over gathered winners. |
| BRISC | In mesh mode, handles fabric sends via `WorkerToFabricEdmSender`. In socket mode, sends token via D2H socket. |
| TRISC | No-op ($k=1$ is a comparison, not tile math). |

### Key Implementation Detail

**Per-core local argmax (Phase 1):**
```cpp
for (uint32_t i = 0; i < num_values; ++i) {
    if (bfloat16_greater(scores_ptr[i], best_score)) {
        best_score = scores_ptr[i];
        best_index = indices_ptr[i];
    }
}
```

The winner (score + index) is packed into 8 bytes and sent to the final core via `noc_async_write_one_packet`.

**Deterministic tie-breaking:**
```cpp
return bfloat16_greater(candidate_score, best_score) ||
       ((candidate_score == best_score) && (candidate_index < best_index));
```

**Multi-device mesh mode.** For an $(R, 2)$ mesh, a 2-stage hierarchical reduction: Stage 1 sends per-device winners along the x-axis to the target row (link selection by ring distance). Stage 2 sends within the target row to the target column. The `fabric_scratch_tensor` stores `stage1_num_slots + stage2_num_slots` winner payloads.

**Socket output modes:** `socket_mode = 0` (L1 only), `1` (L1 + D2H socket to host), `2` (L1 + D2D socket to next pipeline stage).

### Composed Into

- **lm_head_sampling** -- Fused argmax stage: after vocab matmul produces scores across all cores, argmax reduces them to a single output index. In multi-device mode, mesh argmax coordinates across all devices using fabric connections.

---

## 3.1.8 tilize_8x32

**Source:** `micro_ops/tilize_8x32/op.py`

### Purpose

Format conversion from row-major to tiled format for $[8, N]$ tensors. Used as a building block in data preparation.

### Computation

Identity (tilize + untilize = no-op). Reinterprets `[8, N]` row-major data as a sequence of `[8, 32]` tiles ($N/32$ tiles), executing `tilize_block` on each.

### CB Layout

| CB | Index | Role | Backed by |
|----|-------|------|-----------|
| in | 0 | Row-major input | HEIGHT_SHARDED tensor |
| out | 16 | Tiled output | HEIGHT_SHARDED tensor |

### Compile-Time Args

| Arg Name | RISC | Type | Description |
|----------|------|------|-------------|
| `in_cb` | TRISC | `uint32_t` | Input CB index (0) |
| `out_cb` | TRISC | `uint32_t` | Output CB index (16) |
| `block_size` | TRISC | `uint32_t` | Number of 8x32 tiles ($N / 32$) |

### Runtime Args

None.

### RISC Roles

| Processor | Role |
|-----------|------|
| NCRISC | No-op. |
| BRISC | No-op. |
| TRISC | Calls `tilize_block(in_cb, block_size, out_cb)`. |

### Key Implementation Detail

- Height must be exactly 8, width $N$ must be divisible by 32.
- Maximum width is 256 (8 tiles). For larger widths, the caller must loop.
- This is the simplest micro-op in the library.

### Compute Config

- Math fidelity: `LoFi` (pure data rearrangement, no arithmetic precision needed)
- `fp32_dest_acc_en = False`, `dst_full_sync_en = False`

### Composed Into

- **create_q_heads** -- Receiver cores perform three-phase tilization: Phase 1 tilizes 8 tiles of QNOPE first half, Phase 2 tilizes 8 tiles of QNOPE second half, Phase 3 tilizes 2 tiles of QROPE.
- **kv_cache_update** -- The retilize step (bfloat16 intermediate back to BFP8 for DRAM write-back) uses the same `tilize_block` primitive.

---

## Summary Table

| Micro-Op | Computation | Key Mechanism | TRISC Complexity |
|----------|-------------|---------------|------------------|
| `matmul` | $A \times B$ | MOP inner loop via `custom_mm_block`, fused SFPU on PACK thread | High |
| `rmsnorm` | $x / \text{RMS}(x) \cdot \gamma$ | 4-phase pipeline in dest: square-reduce, rsqrt, broadcast-mul, gamma-mul | High |
| `rope` | $x \cos\theta + Tx \sin\theta$ | Matrix-based rotate_half, position-dynamic DRAM reads | Medium |
| `kn_sliced_matmul` | Partial $A[K_i] \times B[K_i]$ | `k_offset` into shared activation CB | Medium |
| `local_reduce` | $\sum_i \text{tile}[i]$ | `acc_to_dest` pairwise accumulation, optional SiLU | Low |
| `eltwise_add` | $A + B[\text{index}]$ | CB read pointer manipulation, zero-copy indexing | Low |
| `sampling` | $\arg\max$ | 3-phase mesh reduction, fabric sends, deterministic tie-break | None (NCRISC/BRISC only) |
| `tilize_8x32` | Row-major to tiled | `tilize_block` LLK, max 256 width | Low |

---

**Next:** [`02_data_movement_micro_ops.md`](./02_data_movement_micro_ops.md)
