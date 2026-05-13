# 02 -- Dual-Expert Block FusedOp: Decomposition Options and CB Flow

## Context

Each of Pi0.5's 18 transformer layers processes **two token streams simultaneously**:
the VLM (PaliGemma) stream at width=2048 and the action expert stream at width=1024.
The reference implementation in `openpi/models/gemma.py` (class `Block`) shows how
these two streams interact:

1. **Pre-attention norms**: Each expert applies its own RMSNorm (VLM: standard,
   action: adaRMSNorm with timestep conditioning)
2. **Shared SDPA**: QKV projections from both experts are concatenated along the
   sequence dimension, a single self-attention is computed, then results are split
   back per-expert
3. **Output projections**: Each expert applies its own output einsum
4. **Gated residual add**: VLM uses standard residual, action uses gated residual
5. **Pre-FFN norms**: Same pattern as pre-attention (standard vs adaRMSNorm)
6. **FFNs**: Each expert runs its own gated FFN (GeLU-gated for both)
7. **Second gated residual add**

The question is: how to decompose this into FusedOps for TT-Blaze?

## Key Takeaways

1. **Option A (recommended)**: Three FusedOps per layer -- DualExpertPreAttn,
   SharedSDPA, DualExpertPostAttn. This matches natural data-dependency boundaries
   and lets SDPA use its existing optimized implementation.

2. **Option B (monolithic)**: One giant FusedOp per layer. Maximum fusion but
   impractical -- the CB count exceeds what FusedProgram can manage, and the
   kernel becomes unmaintainable.

3. **Option C (fully separate)**: Separate FusedOps per expert with explicit
   synchronization. Simpler per-op but requires host-side orchestration for the
   shared attention step, adding latency.

4. **Width mismatch** (2048 vs 1024) is resolved at the QKV projection boundary:
   both experts project to the same `head_dim=256`, so concatenation along the
   sequence dimension is valid regardless of input width.

5. The **None-expert pattern** (`[prefix_tokens, None]` or `[None, suffix_tokens]`)
   from the reference code maps to conditional CB allocation -- when an expert's
   input is None, its CBs are never allocated and its MicroOps are skipped.

## Option A: DualExpertPreAttn + SharedSDPA + DualExpertPostAttn (Recommended)

```
Layer N FusedOp Decomposition (Option A):

  VLM tokens [B,Sv,2048]    Action tokens [B,Sa,1024]    cond [B,1024]
       |                           |                          |
       v                           v                          v
  +------------------------------------------------------------------+
  |  DualExpertPreAttn FusedOp                                       |
  |                                                                  |
  |  [RMSNorm_0]              [AdaRMSNorm_1]                        |
  |       |                     |        |                           |
  |  [QKV_proj_0]          [QKV_proj_1]  gate_pre                   |
  |   q0,k0,v0              q1,k1,v1                                |
  |       |                     |                                    |
  |       +-------[concat]------+    (along sequence dim)            |
  |                  |                                               |
  |            q,k,v [B, Sv+Sa, head_dim]                            |
  +------------------------------------------------------------------+
       |                                          gate_pre held
       v                                               |
  +------------------------------------------------------------------+
  |  SharedSDPA FusedOp                                              |
  |                                                                  |
  |  [RoPE] -> [SDPA] -> attn_out [B, Sv+Sa, num_heads*head_dim]    |
  |                                                                  |
  +------------------------------------------------------------------+
       |
       v
  +------------------------------------------------------------------+
  |  DualExpertPostAttn FusedOp                                      |
  |                                                                  |
  |  [split attn_out by Sv, Sa]                                      |
  |       |                      |                                   |
  |  [O_proj_0]             [O_proj_1]                               |
  |       |                      |                                   |
  |  [ResidualAdd_0]        [GatedResidualAdd_1] <-- gate_pre        |
  |       |                      |                                   |
  |  [RMSNorm_0]            [AdaRMSNorm_1]                           |
  |       |                    |       |                              |
  |  [GeLU_FFN_0]          [GeLU_FFN_1] gate_post                    |
  |       |                      |                                   |
  |  [ResidualAdd_0]        [GatedResidualAdd_1] <-- gate_post       |
  |       |                      |                                   |
  +------------------------------------------------------------------+
       |                         |
  VLM out [B,Sv,2048]     Action out [B,Sa,1024]
```

### Why Option A works best

- **SDPA is already optimized**: Blaze has existing SDPA ops (`blaze/ops/sdpa/`,
  `blaze/ops/flash_mla/`) that handle KV-cache, RoPE, and attention masking.
  Wrapping SDPA in a monolithic FusedOp would duplicate this work.
- **Pre/Post FusedOps are naturally parallel**: The two experts' norms and projections
  are independent within each FusedOp, so the MicroOps can be scheduled on
  disjoint core ranges.
- **CB count is manageable**: Each FusedOp uses ~10-15 CBs, well within the 32-CB
  hardware limit per core.

## Width Mismatch Resolution

The VLM expert has width=2048; the action expert has width=1024. How do they share
a single SDPA?

From `openpi/models/gemma.py`, both experts project to the **same** QKV dimensions:

```
VLM (config index 0):      width=2048, num_heads=8, num_kv_heads=1, head_dim=256
Action (config index 1):   width=1024, num_heads=8, num_kv_heads=1, head_dim=256
```

The QKV projection contracts the width dimension:
- VLM: `[B, Sv, 2048] @ [2048, 256*8] -> [B, Sv, 2048]` (Q), similar for K/V
- Action: `[B, Sa, 1024] @ [1024, 256*8] -> [B, Sa, 2048]` (Q), similar for K/V

After projection, both streams have the same per-token dimension (num_heads *
head_dim = 8 * 256 = 2048). Concatenation along sequence dim is valid:
`Q = concat(Q_vlm, Q_action, dim=1)` gives `[B, Sv+Sa, 2048]`.

Post-attention, the concatenated output is split back:
- `attn_out[:, :Sv, :]` goes to VLM's O_proj: `[B, Sv, 2048] @ [2048, 2048]`
- `attn_out[:, Sv:, :]` goes to action's O_proj: `[B, Sa, 2048] @ [2048, 1024]`

The O_proj restores each expert's native width.

```
Width flow through one layer:

  VLM:    2048 --[QKV proj]--> 2048 --[SDPA]--> 2048 --[O proj]--> 2048
  Action: 1024 --[QKV proj]--> 2048 --[SDPA]--> 2048 --[O proj]--> 1024
                                  \      |      /
                                   [concatenated]
                                   [along seq dim]
```

## CB Flow for Option A: DualExpertPreAttn

```
CB Allocation for DualExpertPreAttn:

CB  | Owner        | Purpose                | Pages | Size/pg | Total
----|-------------|-------------------------|-------|---------|-------
 0  | VLM          | input activation       |   2   | 4096 B  |  8 KB
 1  | VLM          | gamma (norm scale)     |   1   | 4096 B  |  4 KB
 2  | VLM          | normed output          |   2   | 4096 B  |  8 KB
 3  | VLM          | Q projection out       |   2   | 4096 B  |  8 KB
 4  | VLM          | K projection out       |   2   |  512 B  |  1 KB
 5  | VLM          | V projection out       |   2   |  512 B  |  1 KB
----|-------------|-------------------------|-------|---------|-------
 8  | Action       | input activation       |   2   | 2048 B  |  4 KB
 9  | Action       | normed output          |   2   | 2048 B  |  4 KB
10  | Action       | cond input             |   2   | 2048 B  |  4 KB
11  | Action       | modulation output      |   2   | 2048 B  |  4 KB
12  | Action       | scale (from split)     |   1   | 2048 B  |  2 KB
13  | Action       | shift (from split)     |   1   | 2048 B  |  2 KB
14  | Action       | gate_pre (held)        |   1   | 2048 B  |  2 KB
15  | Action       | Q projection out       |   2   | 4096 B  |  8 KB
16  | Action       | K projection out       |   2   |  512 B  |  1 KB
17  | Action       | V projection out       |   2   |  512 B  |  1 KB
----|-------------|-------------------------|-------|---------|-------
20  | Shared       | concat Q output        |   2   | 4096 B  |  8 KB
21  | Shared       | concat K output        |   2   |  512 B  |  1 KB
22  | Shared       | concat V output        |   2   |  512 B  |  1 KB
----|-------------|-------------------------|-------|---------|-------
    | TOTAL        |                        |       |         | ~73 KB

Notes:
- VLM page sizes are 2x action's because width=2048 vs 1024
  (tile width 32, so 2048/32=64 tiles vs 1024/32=32 tiles per row)
- K/V have fewer heads (num_kv_heads=1), so their page sizes are small
- gate_pre CB (14) must persist until DualExpertPostAttn consumes it
- Page size = num_tiles_per_row * tile_size_bytes (2048 B per 32x32 bf16 tile)
```

## Option B: Monolithic Block FusedOp (Not Recommended)

A single FusedOp encompassing the entire transformer block would fuse all of:
pre-attention norms, QKV projections, SDPA, O projections, residual adds,
pre-FFN norms, FFNs, and final residual adds.

```
  +--------------------------------------------------------------+
  |  MonolithicBlock FusedOp                                     |
  |  [Everything from input to output in one kernel]             |
  |                                                              |
  |  Problems:                                                   |
  |  - 30+ CBs needed (exceeds practical limit)                  |
  |  - SDPA reimplemented inside the fused kernel                |
  |  - No reuse of existing SDPA ops                             |
  |  - Kernel source becomes 1000+ lines of C++                  |
  |  - Debugging is extremely difficult                          |
  |  - Any change to attention requires rebuilding everything    |
  +--------------------------------------------------------------+
```

This option is rejected because:
1. CB count exceeds the practical per-core limit
2. Reimplements SDPA instead of reusing validated ops
3. Maintenance cost is too high
4. No performance benefit over Option A (the DRAM round-trip between
   PreAttn and SDPA is minimal compared to compute)

## Option C: Separate Per-Expert Pipelines (Not Recommended)

Run each expert as its own FusedOp pipeline, synchronizing at the SDPA boundary
via host-side orchestration:

```
  Host orchestration:
    1. Run VLM_PreAttn FusedOp -> Q0, K0, V0
    2. Run Action_PreAttn FusedOp -> Q1, K1, V1
    3. Host concatenates Q, K, V
    4. Run SharedSDPA FusedOp
    5. Host splits attention output
    6. Run VLM_PostAttn FusedOp
    7. Run Action_PostAttn FusedOp
```

This option is rejected because:
1. Steps 3 and 5 require host-device round-trips (slow)
2. Six FusedOp launches per layer vs three in Option A
3. The two pre-attention FusedOps could run in parallel on disjoint cores
   (which Option A already achieves within a single FusedOp)

## The DualExpertPreAttn + SharedSDPA + DualExpertPostAttn Split

### DualExpertPreAttn

Inputs: `vlm_tokens`, `action_tokens` (or None), `cond` (or None)
Outputs: `Q`, `K`, `V` (concatenated), `gate_pre_attn`, `gate_pre_ffn` (held)

Internally:
1. VLM path: `RMSNorm.emit()` -> `QKV_Proj.emit()`
2. Action path: `AdaRMSNorm.emit()` -> `QKV_Proj.emit()`
3. Both Q/K/V concatenated along sequence dimension

When `action_tokens` is None (prefix-only inference), the action path MicroOps
are skipped and only VLM Q/K/V are produced. The FusedOp handles this via
conditional emit -- checking if the action input CBHandle is None before
emitting action-expert MicroOps.

### SharedSDPA

Inputs: `Q`, `K`, `V`, `positions`, `attn_mask`, optional `kv_cache`
Outputs: `attn_out`

This is the standard SDPA op from `blaze/ops/sdpa/`, unmodified. It does not
know about dual experts -- it just sees a concatenated Q/K/V sequence.

### DualExpertPostAttn

Inputs: `attn_out`, `vlm_residual`, `action_residual`, `gate_pre_attn`,
        `cond` (for post-FFN adaRMSNorm), `gate_pre_ffn`
Outputs: `vlm_out`, `action_out`

Internally:
1. Split `attn_out` by sequence lengths (Sv, Sa)
2. VLM path: `O_Proj.emit()` -> `ResidualAdd.emit()` -> `RMSNorm.emit()` -> `FFN.emit()` -> `ResidualAdd.emit()`
3. Action path: `O_Proj.emit()` -> `GatedResidualAdd.emit(gate_pre_attn)` -> `AdaRMSNorm.emit()` -> `FFN.emit()` -> `GatedResidualAdd.emit(gate_pre_ffn)`

## The None-Expert Pattern

The reference code passes `[prefix_tokens, None]` or `[None, suffix_tokens]` to
the dual-expert model. This maps to conditional execution:

```python
# In Pi0.sample_actions() -- prefix pass:
(prefix_out, suffix_out), kv_cache = self.PaliGemma.llm(
    [prefix_tokens, None],   # action expert is None
    mask=prefix_attn_mask,
    positions=positions,
)

# In Pi0.sample_actions() -- denoising step:
(prefix_out, suffix_out), _ = self.PaliGemma.llm(
    [None, suffix_tokens],   # VLM expert is None (uses KV cache)
    mask=full_attn_mask,
    positions=positions,
    kv_cache=kv_cache,
    adarms_cond=[None, adarms_cond],
)
```

In the Blaze FusedOp, this is implemented as:

```python
@staticmethod
def emit(f, vlm_input, action_input, cond, ...):
    # VLM path -- skip if vlm_input is None
    if vlm_input is not None:
        vlm_normed = RMSNorm.emit(f, vlm_input, vlm_gamma, ...)
        vlm_q, vlm_k, vlm_v = QKV_Proj.emit(f, vlm_normed, vlm_qkv_w, ...)
    else:
        vlm_q = vlm_k = vlm_v = None

    # Action path -- skip if action_input is None
    if action_input is not None:
        action_normed, gate = AdaRMSNorm.emit(f, action_input, ..., cond, ...)
        act_q, act_k, act_v = QKV_Proj.emit(f, action_normed, act_qkv_w, ...)
    else:
        act_q = act_k = act_v = None
        gate = None

    # Concatenate non-None Q/K/V
    q = concat_non_none(vlm_q, act_q)
    k = concat_non_none(vlm_k, act_k)
    v = concat_non_none(vlm_v, act_v)

    return q, k, v, gate
```

This pattern avoids allocating CBs for inactive experts, saving L1 space during
prefix-only or action-only inference phases.

## Source References

- `openpi/models/gemma.py:284-333` -- Block class with dual-expert norm/attn/FFN
- `openpi/models/gemma.py:158-249` -- Attention class with per-expert QKV projections
- `openpi/models/gemma.py:253-280` -- FeedForward (GeLU-gated FFN)
- `openpi/models/gemma.py:340-411` -- Module class with dual-expert forward pass
- `openpi/models/pi0.py:234-237` -- Prefix pass with `[prefix_tokens, None]`
- `openpi/models/pi0.py:261-268` -- Denoising pass with `[None, suffix_tokens]`
- `blaze/ops/sdpa/` -- Existing SDPA ops (reused unchanged)
- `blaze/ops/rmsnorm/op.py` -- RMSNorm MicroOp reused by VLM and adaRMSNorm
- `blaze/fused_program.py` -- FusedProgram CB allocation and management
