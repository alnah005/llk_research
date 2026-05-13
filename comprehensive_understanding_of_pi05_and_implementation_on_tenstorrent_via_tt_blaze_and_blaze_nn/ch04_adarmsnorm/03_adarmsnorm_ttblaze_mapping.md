# 03 -- adaRMSNorm TT-Blaze Mapping

## Context

Mapping adaRMSNorm onto Tenstorrent hardware via TT-Blaze surfaces a gap between
the model's requirements and the available op library. The existing `rmsnorm`
MicroOp in blaze-nn performs standard RMSNorm -- normalize, then scale by a
learned gamma -- but has no support for the adaptive modulation (scale, shift,
gate) that pi0.5's action expert requires. This section analyzes the gap,
proposes two decomposition strategies (composed vs fused), examines the
dual-output challenge that arises from needing to return both the normed output
and a gate tensor, and discusses memory layout considerations for the per-layer
modulation weights.

---

## 1. The Existing `rmsnorm` MicroOp

The current TT-Blaze functional API provides `rmsnorm` at
`blaze_nn/functional.py:46-48`:

```python
def rmsnorm(input: TensorProxy, gamma: Any, *, epsilon: float = 1e-6, **kwargs) -> TensorProxy:
    """Root mean square normalization."""
    return _dispatch("rmsnorm", input, gamma, epsilon=epsilon, **kwargs)
```

Registered in the op registry at `blaze_nn/_registry.py:36`:

```python
register_op("rmsnorm", OpMapping(blaze_op_name="rmsnorm", default_grid="all"))
```

This op computes:

```
output = (x * rsqrt(mean(x^2) + eps)) * gamma
```

Where `gamma` is a per-feature weight (typically initialized to ones in standard
implementations, but to `(1 + scale)` in pi0.5's formulation).

What the existing op provides vs what adaRMSNorm needs:

| Capability | `rmsnorm` MicroOp | adaRMSNorm requirement |
|-----------|-------------------|----------------------|
| RMS normalization | Yes | Yes |
| Static gamma scaling | Yes | Partially (the `(1+scale)` part) |
| Dynamic scale from cond | No | `normed * (1 + scale)` where scale is input-dependent |
| Shift (additive bias) | No | `+ shift` |
| Gate output | No | Must return gate as separate tensor |
| Modulation projection | No | `Dense(cond, D*3)` |
| Multiple outputs | No (single CBHandle) | Two outputs: (normed_output, gate) |

The gap is fundamental: `rmsnorm` is a pointwise unary op with a static weight,
while adaRMSNorm is a conditional multi-input, multi-output operation.

## 2. Comparison to `broadcast_rmsnorm`

Some Tenstorrent op libraries include a `broadcast_rmsnorm` variant designed for
cases where the normalization weight is broadcast across cores in a data-parallel
or tensor-parallel configuration. This op still fundamentally computes the same
`normed * gamma` formula -- it only changes the distribution strategy for the
gamma parameter, using multicast to replicate the weight to all cores before the
element-wise multiply.

`broadcast_rmsnorm` does NOT help with adaRMSNorm because:

1. It still lacks shift and gate outputs.
2. The "broadcast" refers to weight distribution topology, not to the
   broadcasting of a conditioning signal across sequence positions.
3. The modulation in adaRMSNorm is dynamic (computed from input), not static
   (loaded from parameters).

## 3. Proposed `AdaRMSNorm` FusedOp

To support adaRMSNorm on Tenstorrent hardware, a new FusedOp is needed. The
conceptual signature:

```
AdaRMSNorm(
    x:                  CBHandle [B, T, D],      # input activations
    cond:               CBHandle [B, D],          # conditioning vector (adarms_cond)
    modulation_weights: CBHandle [D, 3*D],        # Dense projection weights
    epsilon:            float = 1e-6
) -> (CBHandle [B, T, D], CBHandle [B, 1, D])    # (normed_output, gate)
```

This encapsulates the full adaptive normalization pipeline:

```
  Inputs: x [B,T,D], cond [B,D], W_mod [D, 3D]
       |
       +-- RMSNorm: normed = x * rsqrt(mean(x^2) + eps)
       |
       +-- Modulation: mod = cond @ W_mod  -> [B, 3D]
       |      split -> scale [B,1,D], shift [B,1,D], gate [B,1,D]
       |
       +-- Apply: output = normed * (1 + scale) + shift
       |
       +-- Return: (output [B,T,D], gate [B,1,D])
```

### Memory layout for modulation weights

Each adaRMSNorm instance stores a modulation weight matrix of shape
`[1024, 3072]`. With 37 instances across the model:

| Per instance | Shape | Bytes (BF16) |
|-------------|-------|--------------|
| `W_mod` | [1024, 3072] | 6,291,456 (6 MB) |
| **All 37 instances** | | **232,783,872 (222 MB)** |

In the scanned (layer-stacked) weight layout used by Flax `nn.scan`, the 36
per-layer weights (18 layers x 2 norms per layer) are stacked on an additional
leading axis: `[18, 1024, 3072]` for pre-attention norms and `[18, 1024, 3072]`
for pre-FFN norms. The final norm's weight is stored separately as `[1024, 3072]`.

For Tenstorrent DRAM placement, these weights should be laid out to minimize
page faults during sequential layer execution. Since all 18 layers are processed
in order and the same `cond` vector is reused, a contiguous allocation of
`[18, 1024, 3072]` per norm type allows sequential reads with predictable stride.

## 4. Decomposition Strategy A: Composed from Existing MicroOps

The first approach decomposes adaRMSNorm into existing ops plus a few new
element-wise primitives:

```
  Step 1: rmsnorm(x, ones)          # standard RMSNorm with gamma=1
          -> normed [B, T, D]

  Step 2: matmul(cond, W_mod)       # modulation projection
          -> mod [B, 3D]

  Step 3: split3(mod)               # split along last dim
          -> scale [B, D], shift [B, D], gate [B, D]

  Step 4: unsqueeze + broadcast     # add sequence dim
          -> scale [B, 1, D], shift [B, 1, D], gate [B, 1, D]

  Step 5: mul(normed, 1 + scale)    # apply scale
          -> scaled [B, T, D]

  Step 6: add(scaled, shift)        # apply shift
          -> output [B, T, D]

  Step 7: return (output, gate)     # two separate CBHandles
```

Mapped to existing blaze-nn ops:

| Step | blaze-nn op | Notes |
|------|------------|-------|
| 1 | `rmsnorm` | Use gamma = ones(D) |
| 2 | `matmul` (via `linear`) | Standard dense projection |
| 3 | (new) `split` | Not in current registry |
| 4 | (new) `broadcast_unsqueeze` | Implicit in CB tiling |
| 5 | `gated_reduce` or (new) `elementwise_mul` | Existing op uses activation |
| 6 | `residual_add` | Existing op |
| 7 | N/A | Graph-level routing |

```
  Composed decomposition -- op graph:

  x [B,T,D]          cond [B,D]       W_mod [D,3D]
      |                   |                |
  +---v---+          +----v----+           |
  |rmsnorm|          | matmul  |<----------+
  |(gamma=1)|        |         |
  +---+---+          +----+----+
      |                   |
      |              +----v----+
      |              | split3  |
      |              +--+--+---+
      |                 |  |  |
      |           scale | shift| gate
      |            [B,D] [B,D] [B,D]
      |                 |  |      |
      |           +-----+  |      |
      v           v        |      |
  +--------+  unsqueeze    |      |
  |normed  |  [B,1,D]     |      |
  |[B,T,D] |     |        |      |
  +---+----+     |        |      |
      |     +----v---+    |      |
      +---->| mul    |    |      |
            |(1+scl) |    |      |
            +---+----+    |      |
                |         |      |
           +----v----+    |      |
           |res_add  |<---+      |
           +----+----+ unsqueeze |
                |      [B,1,D]   |
                v                v
          output [B,T,D]    gate [B,1,D]
```

**Advantages**: Reuses existing verified MicroOps. Each step is independently
testable. No custom kernel development required.

**Disadvantages**: Multiple kernel launches per adaRMSNorm invocation. Each step
requires reading/writing intermediate results to circular buffers. For 37
instances, the launch overhead and intermediate memory traffic add up. The
`split3` and `broadcast_unsqueeze` ops do not exist in the current registry and
would need to be added.

## 5. Decomposition Strategy B: Custom Fused LLK SFPU Kernel

The second approach writes a single Low-Level Kernel (LLK) on the SFPU (Special
Function Processing Unit) that performs the entire adaRMSNorm computation in one
pass:

```
  Fused kernel pseudocode (per core, per tile):

  // Phase 1: compute RMS norm (reduction across D)
  variance = tile_reduce_mean_sq(x_tile)
  rsqrt_var = rsqrt(variance + epsilon)

  // Phase 2: compute modulation (matmul cond * W_mod, split)
  mod = tile_matmul(cond_tile, W_mod_tile)   // [tile_B, 3*tile_D]
  scale = mod[:, 0:tile_D]
  shift = mod[:, tile_D:2*tile_D]
  gate  = mod[:, 2*tile_D:3*tile_D]

  // Phase 3: apply and write two outputs
  normed = x_tile * rsqrt_var
  output = normed * (1 + scale) + shift      // -> output CB
  gate_out = gate                             // -> gate CB
```

**Advantages**: Single kernel launch per adaRMSNorm. No intermediate CB traffic.
The matmul in Phase 2 is small (`[B, 1024] x [1024, 3072]`) and can be tiled
efficiently. Maximum performance.

**Disadvantages**: Requires custom LLK development and verification. The kernel
mixes reduction (RMS), matmul, and element-wise ops, which is complex for SFPU
programming. The matmul step may better belong on the FPU/matrix engine rather
than SFPU, requiring a hybrid kernel or a two-phase approach where the matmul
runs on the systolic array and the element-wise ops run on SFPU.

### Hybrid variant

A practical middle ground: split the fused kernel into two phases that use
different compute units:

```
  Phase A (Matrix Engine / FPU):
    mod = matmul(cond, W_mod)         # [B, 3072]
    split -> scale, shift, gate

  Phase B (SFPU):
    normed = x * rsqrt(mean(x^2) + eps)
    output = normed * (1 + scale) + shift
    write output to CB_out
    write gate to CB_gate
```

This plays to each unit's strengths: the matrix engine handles the projection,
and the SFPU handles the normalization and element-wise modulation.

## 6. The Dual-Output Challenge

Standard FusedOps in TT-Blaze return a single `CBHandle`. adaRMSNorm must
return two: the normed output `[B, T, D]` and the gate `[B, 1, D]`.

Options:

### Option A: Concatenate and split

Pack both outputs into a single CB, then split downstream:

```
  combined = concat(output, gate_broadcast)   # [B, T+1, D] or similar
  output = combined[:, :T, :]
  gate   = combined[:, T:, :]
```

This wastes memory if `gate` needs to be broadcast to `[B, T, D]` (which it
does not -- the residual connection handles broadcasting). But it complicates
the CB layout and requires the consumer to know the split point.

### Option B: Two separate CBHandles via op tuple return

Extend the FusedOp interface to support returning a tuple of CBHandles:

```python
class AdaRMSNormOp(FusedOp):
    def __call__(self, x, cond, weights) -> tuple[CBHandle, CBHandle]:
        ...
        return (normed_output_cb, gate_cb)
```

This is cleaner but requires framework-level changes to how the tracing context
and graph builder handle multi-output ops. The `_dispatch` function in
`blaze_nn/functional.py:16-33` currently returns a single `TensorProxy`.

### Option C: Two sequential ops sharing state

Model adaRMSNorm as two ops that share intermediate state via a named CB:

```python
normed_output = F.adarmsnorm_forward(x, cond, weights)  # writes gate to named CB
gate = F.adarmsnorm_gate(x, cond, weights)               # reads gate from named CB
```

This avoids multi-output dispatch but introduces implicit state coupling between
ops, which complicates the graph optimizer.

### Recommendation

Option B (tuple return) is the cleanest long-term solution. The tracing
infrastructure changes are bounded: `_dispatch` needs to return
`tuple[TensorProxy, ...]` for multi-output ops, and the graph builder needs to
track multiple output CBs per node. This pattern will also be needed for other
multi-output operations (e.g., attention returning both output and updated KV
cache).

## 7. Instance-Level Mapping Summary

For the full pi0.5 model, the mapping must handle 37 adaRMSNorm instances
plus their PaliGemma counterparts:

```
  Layer k (k = 0..17):
  +--------------------------------------------------+
  |  Expert 0 (PaliGemma)    Expert 1 (Action)       |
  |                                                    |
  |  pre_attn_norm:          pre_attn_norm_1:         |
  |    rmsnorm(x, gamma)       AdaRMSNorm(x, cond, W) |
  |    -> (out, None)          -> (out, gate_attn)     |
  |                                                    |
  |  [shared attention]                                |
  |                                                    |
  |  gated_residual:         gated_residual:           |
  |    x + y                   x + y * gate_attn       |
  |                                                    |
  |  pre_ffw_norm:           pre_ffw_norm_1:           |
  |    rmsnorm(x, gamma)       AdaRMSNorm(x, cond, W) |
  |    -> (out, None)          -> (out, gate_ffw)      |
  |                                                    |
  |  [independent FFN]                                 |
  |                                                    |
  |  gated_residual:         gated_residual:           |
  |    x + y                   x + y * gate_ffw        |
  +--------------------------------------------------+

  Final:
  +--------------------------------------------------+
  |  final_norm:             final_norm_1:            |
  |    rmsnorm(x, gamma)       AdaRMSNorm(x, cond, W) |
  |    -> out                  -> out (gate discarded) |
  +--------------------------------------------------+
```

Op count per forward pass:

| Op type | Expert 0 | Expert 1 | Total |
|---------|----------|----------|-------|
| Standard `rmsnorm` | 37 | 0 | 37 |
| `AdaRMSNorm` (proposed) | 0 | 37 | 37 |
| `_gated_residual` (gate=None) | 36 | 0 | 36 |
| `_gated_residual` (gate!=None) | 0 | 36 | 36 |

## 8. Weight Prefetch and Scheduling Considerations

Because `adarms_cond` is the same across all layers, the modulation matmul
`cond @ W_mod` can be precomputed for all 37 instances before the layer scan
begins:

```
  Precompute opportunity:

  For each norm instance j in [0..36]:
    mod_j = adarms_cond @ W_mod_j        # [B, 3072]
    scale_j, shift_j, gate_j = split3(mod_j)

  Then during layer execution, each adaRMSNorm only needs:
    normed * (1 + scale_j) + shift_j     # element-wise only, no matmul
```

This converts 37 matmuls into a batch of matmuls that can be executed once,
followed by 37 pure element-wise operations. The precomputed modulations total
`37 * B * 3072 * 2 bytes = 227,328 * B` bytes in BF16 -- trivially small
compared to the activation memory.

The weight matrices for precomputation can be stacked:
`W_all = [W_mod_0, W_mod_1, ..., W_mod_36]` with shape `[37, 1024, 3072]`,
enabling a single batched matmul `adarms_cond @ W_all -> [37, B, 3072]`.

---

## Key Takeaways

- The existing `rmsnorm` MicroOp in blaze-nn handles only static-weight RMSNorm.
  It cannot support dynamic scale/shift from a conditioning signal or gate
  output.

- `broadcast_rmsnorm` solves a weight-distribution problem, not a conditional
  modulation problem. It does not close the adaRMSNorm gap.

- Strategy A (composed decomposition) chains `rmsnorm -> matmul -> split3 ->
  mul -> add` using mostly existing ops but incurs multi-launch overhead and
  needs new `split3` and `broadcast_unsqueeze` ops.

- Strategy B (fused LLK kernel) delivers maximum performance in a single
  launch but requires custom SFPU development. A hybrid variant splitting
  matmul (matrix engine) from element-wise ops (SFPU) is the practical path.

- The dual-output challenge (returning both `normed_output` and `gate`) does
  not fit the current single-`TensorProxy` return convention. Extending
  `_dispatch` to support tuple returns is the recommended solution.

- Precomputing all 37 modulation projections before the layer scan exploits
  the invariant that `adarms_cond` is constant across layers, converting
  per-layer matmuls into a single batched matmul plus per-layer element-wise
  ops.
