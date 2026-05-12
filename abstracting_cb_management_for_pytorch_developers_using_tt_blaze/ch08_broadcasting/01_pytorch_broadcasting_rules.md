# PyTorch Broadcasting Rules and Their LLM Workload Implications

Broadcasting is the mechanism by which PyTorch allows element-wise operations between tensors of different shapes without requiring the programmer to manually replicate data. A developer writes `a + b` where `a` has shape `[B, D]` and `b` has shape `[1, D]`, and PyTorch silently expands `b` along dimension 0 to match `a`. This convenience is so deeply woven into the PyTorch programming model that LLM code depends on it dozens of times per forward pass -- bias additions, attention masking, layer normalization, and gating operations all rely on broadcasting. When TT-Blaze must lower these operations to Tensix hardware, the broadcasting semantics must be preserved exactly, or numerical correctness breaks silently. This section formally defines PyTorch's broadcasting rules, catalogs the broadcasting patterns that appear in real LLM workloads (DeepSeek V3, LLaMA), explains what the developer expects from the abstraction, and identifies the gap that the remaining two sections of this chapter must fill.

> *Design Principle P1 (Separate Computation Intent from Execution Strategy):* Broadcasting is the canonical example of P1 in action. The developer's intent is "add a bias to every row" -- the execution strategy (whether to replicate data in L1, multicast via NOC, or have the compute kernel re-read a single tile) is an implementation detail the developer should never see.

**What you will learn:**

- The formal three-step broadcasting algorithm: right-align, expand dimensions of size 1, prepend missing dimensions
- Why PyTorch broadcasting is implicit and what guarantee the developer relies on
- Every broadcasting pattern that appears in LLM transformer blocks: bias addition, attention masking, layer norm scaling, MoE gating, RoPE frequency expansion
- Concrete shape examples from DeepSeek V3 (hidden dim 7168, 128 experts) and LLaMA (hidden dim 4096)
- The complete mapping from PyTorch broadcasting to the shape alignment questions TT-Blaze must answer
- Why broadcasting creates a three-way problem: logical shape expansion, physical data layout, and tile-level compute behavior

---

## 1. The Formal Broadcasting Algorithm

PyTorch's broadcasting rules are defined by [NumPy's broadcasting semantics](https://numpy.org/doc/stable/user/basics.broadcasting.html), which PyTorch adopted without modification. The algorithm operates on two input shapes and produces an output shape:

### Step 1: Right-Align Shapes

Given two tensors with shapes `A = (a_1, a_2, ..., a_n)` and `B = (b_1, b_2, ..., b_m)`, right-align the shape tuples so the trailing dimensions line up. If one shape has fewer dimensions (say `m < n`), conceptually prepend dimensions of size 1 until both shapes have the same rank.

```text
Example: A = [B, S, D], B = [D]

Right-align:
    A:  B   S   D
    B:          D        (prepend 1s)
    B': 1   1   D
```

### Step 2: Check Compatibility Per Dimension

For each aligned dimension pair `(a_i, b_i)`, the dimensions are compatible if:
- They are equal: `a_i == b_i`, or
- One of them is 1: `a_i == 1` or `b_i == 1`

If any dimension pair is incompatible (neither equal nor has a 1), the operation raises a `RuntimeError`.

### Step 3: Compute Output Shape

The output shape is the element-wise maximum of the aligned shapes:

```text
output_i = max(a_i, b_i)
```

When `a_i == 1` and `b_i > 1`, tensor A is "broadcast" (logically expanded) along dimension `i` to match B's size. No data is physically copied -- PyTorch sets the stride to 0 in that dimension, so the same memory is read repeatedly.

### The Complete Algorithm in Code

```python
# REFERENCE -- PyTorch broadcasting algorithm (equivalent to torch.broadcast_shapes)
def broadcast_shapes(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> tuple[int, ...]:
    """Compute the broadcast output shape for two input shapes.

    Raises ValueError if shapes are not broadcast-compatible.
    """
    # Step 1: Right-align by padding shorter shape with 1s
    max_rank = max(len(shape_a), len(shape_b))
    padded_a = (1,) * (max_rank - len(shape_a)) + shape_a
    padded_b = (1,) * (max_rank - len(shape_b)) + shape_b

    # Steps 2-3: Check compatibility and compute output
    output = []
    for i, (a, b) in enumerate(zip(padded_a, padded_b)):
        if a == b:
            output.append(a)
        elif a == 1:
            output.append(b)
        elif b == 1:
            output.append(a)
        else:
            raise ValueError(
                f"Shapes {shape_a} and {shape_b} are not broadcast-compatible: "
                f"dimension {i} has sizes {a} and {b}"
            )
    return tuple(output)
```

### Broadcasting Direction Classification

For the purposes of TT-Blaze lowering, we classify each broadcast dimension into one of three directions:

```text
Broadcast Direction Classification:

1. NO_BROADCAST:    a_i == b_i (no expansion needed)
2. LEFT_BROADCASTS: a_i == 1, b_i > 1 (left operand is expanded)
3. RIGHT_BROADCASTS: a_i > 1, b_i == 1 (right operand is expanded)
```

When only one operand has size 1 in a given dimension, that operand is the "broadcast source" -- its data is logically replicated to match the other operand. A single operation can broadcast in multiple dimensions simultaneously (e.g., `[1, 1, D]` broadcast against `[B, S, D]` broadcasts along both dimension 0 and dimension 1).

---

## 2. Broadcasting Patterns in LLM Workloads

LLM transformer architectures make heavy use of broadcasting. The patterns are not random -- they fall into a small number of recurring categories. Understanding these categories is essential for designing an automatic broadcast detection and insertion system.

### 2.1 Bias Addition: [1, D] + [B, D]

The most fundamental broadcast pattern. Every linear layer can optionally add a bias vector of shape `[D]` (or `[1, D]` after rank promotion) to an activation of shape `[B, D]`.

```python
# PyTorch developer code
x = linear(input)             # [B, D] = [B, D_in] @ [D_in, D] + [D]
# Under the hood:
# matmul_result: [B, D]
# bias:          [D] -> promoted to [1, D] -> broadcast to [B, D]
# output = matmul_result + bias
```

**Concrete dimensions:**

| Model | Activation Shape | Bias Shape | Broadcast Pattern |
|-------|-----------------|------------|-------------------|
| LLaMA-7B FFN up | `[1, 4096]` | `[4096]` -> `[1, 4096]` | No broadcast needed (B=1 in decode) |
| LLaMA-7B FFN up (prefill) | `[128, 4096]` | `[4096]` -> `[1, 4096]` | Broadcast bias across 128 tokens |
| DeepSeek V3 attention out | `[1, 7168]` | `[7168]` -> `[1, 7168]` | No broadcast needed (single-token decode) |
| DeepSeek V3 attention out (prefill) | `[128, 7168]` | `[7168]` -> `[1, 7168]` | Broadcast bias across 128 tokens |

**Tile-level implication:** The bias tensor occupies a single row of tiles. During broadcasting, the same row of tiles is read for every output row. In tile coordinates for `[128, 4096]` with 32x32 tiles, the bias is `[1, 128]` tiles and must be replicated across 4 tile rows.

### 2.2 Attention Mask: [1, 1, S, S] * [B, H, S, S]

Attention masking applies a causal or padding mask to the attention score matrix. The mask typically has shape `[1, 1, S, S]` (shared across batch and heads), while attention scores have shape `[B, H, S, S]`.

```python
# PyTorch developer code
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [B, H, S, S]
mask = causal_mask(S)                                              # [1, 1, S, S]
scores = scores + mask  # broadcast mask across B and H dimensions
```

**Concrete dimensions:**

| Model | Scores Shape | Mask Shape | Broadcast Dims |
|-------|-------------|------------|----------------|
| LLaMA-7B (S=128) | `[1, 32, 128, 128]` | `[1, 1, 128, 128]` | Dim 1: H=32 |
| DeepSeek V3 (S=128, MHA) | `[1, 128, 128, 128]` | `[1, 1, 128, 128]` | Dim 1: H=128 |
| Generic prefill (S=2048) | `[4, 64, 2048, 2048]` | `[1, 1, 2048, 2048]` | Dims 0,1: B=4, H=64 |

**Tile-level implication:** The mask is a 2D matrix of tiles that must be replicated across the batch and head dimensions. For the inner two dimensions, no broadcast occurs -- the mask is full-rank in S. The outer dimensions require either physical replication or kernel-level re-reading of the same tile data.

### 2.3 Layer Norm Scale/Shift: [D] * [B, S, D]

Layer normalization and RMS normalization apply learned scale (gamma) and shift (beta) parameters. The parameters have shape `[D]` while the activation has shape `[B, S, D]`.

```python
# PyTorch developer code -- RMSNorm
def rmsnorm(x, gamma, epsilon=1e-6):
    # x:     [B, S, D]
    # gamma: [D]  -> promoted to [1, 1, D] -> broadcast across B, S
    variance = x.pow(2).mean(-1, keepdim=True)  # [B, S, 1]
    x_normed = x * torch.rsqrt(variance + epsilon)  # [B, S, D]
    return x_normed * gamma  # broadcast gamma across B, S
```

This is the exact pattern implemented by `BroadcastRMSNorm` in TT-Blaze. The gamma tensor is a 1D vector that must be broadcast to match the activation's batch and sequence dimensions. The `BroadcastRMSNorm` op in TT-Blaze (see `blaze/ops/broadcast_rmsnorm/op.py`) handles this by first doing a CCL broadcast of the activation across devices, then applying RMSNorm with the gamma -- the RMSNorm LLK internally handles the gamma broadcast using `rmsnorm_mul_bcast_scalar_reuse_tiles` (line 140-141 of `rmsnorm/kernels/op.hpp`), which re-reads the scalar CB without needing physical data replication.

**Concrete dimensions:**

| Model | Activation | Gamma | Broadcast |
|-------|-----------|-------|-----------|
| LLaMA-7B | `[1, 4096]` or `[128, 4096]` | `[4096]` -> `[1, 4096]` | Dim 0: B (or B*S) |
| DeepSeek V3 | `[1, 7168]` or `[128, 7168]` | `[7168]` -> `[1, 7168]` | Dim 0: B (or B*S) |

### 2.4 Gating Operations: [B, D_expert] * [B, 1]

MoE (Mixture of Experts) gating multiplies each expert's output by its gating weight. The expert activation has shape `[B, D_expert]` and the gating weight has shape `[B, 1]`.

```python
# PyTorch developer code
gate_scores = router(x)              # [B, num_experts]
gate_weight = gate_scores[:, expert_idx:expert_idx+1]  # [B, 1]
expert_out = expert_mlp(x)           # [B, D_expert]
weighted = expert_out * gate_weight  # broadcast [B, 1] across D_expert
```

**Concrete dimensions (DeepSeek V3):**

| Tensor | Shape | Broadcast |
|--------|-------|-----------|
| Expert activation | `[1, 2048]` | None |
| Gate weight | `[1, 1]` -> scalar | Scalar broadcast across all tiles |
| Expert activation (prefill) | `[K, 2048]` | None |
| Gate weight (prefill) | `[K, 1]` | Broadcast across dim 1 |

**Tile-level implication:** When B=1 (single-token decode), the gate weight is a scalar that must be broadcast to every tile -- this is the classic `bcast_scalar` pattern. The `EltwiseMul` kernel in TT-Blaze handles this via `deepseek_mul_tiles_bcast_scalar` (line 144 of `eltwise_mul/kernels/op.hpp`), where a single scalar value in a dedicated `scalar_local` CB is multiplied into every tile.

### 2.5 RoPE Frequency Expansion: [1, D/2] * [S, D/2]

Rotary Position Embeddings apply position-dependent rotation matrices using precomputed cosine and sine frequencies. The frequencies have shape `[S, D/2]` while the input heads may need broadcasting.

```python
# PyTorch developer code
cos = cos_cached[:seq_len, :]   # [S, D/2]
sin = sin_cached[:seq_len, :]   # [S, D/2]
# For multi-head attention, cos/sin are broadcast across heads:
# q: [B, H, S, D/2]
# cos: [1, 1, S, D/2]  -- broadcast across B and H
q_embed = q * cos + rotate_half(q) * sin
```

The RoPE kernel in TT-Blaze (`rope/kernels/op.hpp`, line 154-157) implements this using `mul_tiles_bcast<BroadcastType::ROW>`, which broadcasts the cos/sin row tiles across multiple head tiles -- each head reads the same cos/sin values.

```cpp
// EXISTING -- from blaze/ops/rope/kernels/op.hpp (lines 154-157)
mul_bcast_rows_init_short(compute_cta::rotated_in_interm, compute_cta::cos_sin);
for (uint32_t j = 0; j < Wt; j++) {
    mul_tiles_bcast<BroadcastType::ROW>(
        compute_cta::rotated_in_interm, compute_cta::cos_sin, j, Wt + j, j);
}
```

### 2.6 Residual Addition: [B, S, D] + [B, S, D]

This is the degenerate case -- both operands have the same shape. No broadcasting is needed. It is listed here for completeness because the broadcast detection algorithm must correctly identify this as a no-broadcast case and avoid inserting unnecessary operations.

---

## 3. Formal Classification of Broadcasting Patterns

By analyzing the patterns above, we can classify all LLM broadcasting into a small taxonomy that directly maps to hardware primitives:

### Pattern 1: Row Broadcast (bcast_rows)

One operand has size 1 in the row dimension (second-to-last) and must be replicated across rows.

```text
Shape relationship:
  A: [*, R, C]
  B: [*, 1, C]
  Output: [*, R, C]

Tile-level: B's tiles are re-read R/tile_H times.

Examples:
  - Bias addition in prefill: [128, 4096] + [1, 4096]
  - Gamma broadcast in layer norm: [128, 4096] * [1, 4096]
  - cos/sin in RoPE: [H, S, D/2] * [1, S, D/2] (across heads)
```

The "row broadcast" name refers to the fact that the broadcast source has a single row of tiles that is replicated to fill multiple rows in the output. In the tile coordinate system ([Chapter 4](../ch04_tile_decomposition/)), if the activation has `R_tiles` rows and the bias has `1` tile row, each bias tile is read `R_tiles` times.

### Pattern 2: Column Broadcast (bcast_cols)

One operand has size 1 in the column dimension (last) and must be replicated across columns.

```text
Shape relationship:
  A: [*, R, C]
  B: [*, R, 1]
  Output: [*, R, C]

Tile-level: B's tiles are re-read C/tile_W times.

Examples:
  - Gate weight multiplication: [B, D_expert] * [B, 1]
  - Variance broadcast in layer norm: [B, S, D] * [B, S, 1]
  - Softmax denominator: [B, H, S, S] / [B, H, S, 1]
```

### Pattern 3: Scalar Broadcast (bcast_scalar)

One operand is a scalar (all dimensions are 1, or it is a 0-D tensor) and must be broadcast to every position.

```text
Shape relationship:
  A: [*, R, C]
  B: [1] or [1, 1] or scalar
  Output: [*, R, C]

Tile-level: B occupies a single tile; that tile is read for every output tile.

Examples:
  - Expert gating (decode, B=1): [1, 2048] * scalar
  - Attention scaling: scores / sqrt(d_k)  where sqrt(d_k) is a scalar
  - Epsilon addition in layer norm: variance + 1e-6
```

### Pattern 4: No Broadcast (identity)

Both operands have identical shapes. No expansion is needed.

```text
Shape relationship:
  A: [*, R, C]
  B: [*, R, C]
  Output: [*, R, C]

Examples:
  - Residual addition: [B, S, D] + [B, S, D]
  - Element-wise multiply: [B, S, D] * [B, S, D]
```

### Pattern 5: Multi-Dimensional Broadcast (compound)

Broadcasting occurs along multiple dimensions simultaneously. This cannot be handled by a single hardware primitive and must be decomposed.

```text
Shape relationship:
  A: [B, H, S, S]
  B: [1, 1, S, S]
  Output: [B, H, S, S]

Broadcast dims: 0 (B) and 1 (H)

This requires:
  1. First replicate B across dim 1 (H copies), then
  2. Replicate across dim 0 (B copies)
  OR:
  Treat the compound broadcast as a single "batch broadcast"
  where the inner [S, S] tile grid is shared across batch*heads.
```

### Summary Table

| Pattern | Broadcast Source Dims | Hardware Primitive | Example |
|---------|----------------------|-------------------|---------|
| Row broadcast | `dim[-2] == 1` | `bcast_rows` / tile re-read | Bias addition |
| Column broadcast | `dim[-1] == 1` | `bcast_cols` / tile re-read | Gate weight |
| Scalar broadcast | All dims == 1 | `bcast_scalar` / single-tile read | Scale factor |
| No broadcast | All dims match | Direct elementwise | Residual add |
| Multi-dim broadcast | Multiple dims == 1 | Decompose or batch | Attention mask |

---

## 4. What the Developer Expects

The fundamental contract that broadcasting provides to the PyTorch developer is:

**Write `a + b` and have it work, regardless of shape combinations, as long as shapes are broadcast-compatible.**

This expectation has several implications for the TT-Blaze abstraction layer:

### 4.1 Implicit Detection

The developer never writes "broadcast `b` along dimension 0." They write `a + b` and expect the framework to figure out that broadcasting is needed. Any system that requires the developer to explicitly specify broadcast dimensions has violated P1 (Separate Computation Intent from Execution Strategy).

> *Design Principle P7 (Transparent Interception That Preserves Identity):* The broadcast detection system runs within the `__torch_dispatch__` interception pipeline. The developer's `a + b` call is intercepted, analyzed for broadcasting requirements, and lowered to the appropriate hardware primitives -- all without the developer being aware that interception occurred. The result tensor's shape and dtype match exactly what PyTorch would produce.

### 4.2 Correct Output Shape

After `c = a + b` where `a` has shape `[128, 4096]` and `b` has shape `[1, 4096]`, the developer expects `c.shape == [128, 4096]`. The output must reflect the broadcast-expanded shape, not the shape of either input.

### 4.3 Numerical Equivalence

The broadcast result must be bit-for-bit identical (within floating-point precision) to the result of explicitly expanding the tensor with `b.expand_as(a)` and then performing element-wise addition. Any hardware optimization (tile re-reading, multicast) must produce the same numerical result.

### 4.4 No Performance Cliff

The developer expects broadcasting to be efficient. Physically replicating a `[1, 4096]` bias to `[128, 4096]` before adding it would waste 127x the memory and bandwidth. The hardware should exploit the structure of broadcasting to avoid unnecessary data movement.

### 4.5 Composability

Broadcasting must work in arbitrary combinations with other operations. If `a + b` broadcasts `b`, and the result feeds into a matmul, the matmul must receive the correct broadcast-expanded shape -- not `b`'s original shape. This means the shape tracking system (ShapeDescriptor from [Chapter 3](../ch03_tensor_adapter/)) must correctly propagate broadcast-expanded shapes through the entire op graph.

---

## 5. Broadcasting Frequency in LLM Architectures

To justify the investment in automatic broadcast handling, consider how many broadcast operations occur per transformer layer:

```text
DeepSeek V3 single decode layer (B=1, S=1, D=7168):

  Pre-attention RMSNorm:
    - gamma broadcast: [7168] -> [1, 7168]         (bcast_rows, trivial)
    - variance broadcast: [1, 1] -> [1, 7168]      (bcast_cols)
    Subtotal: 2 broadcasts

  Multi-Head Latent Attention:
    - Q/KV projection: no broadcast (matmul)
    - RoPE cos/sin: [1, 1, 1, D//2] -> [1, H, 1, D//2]  (batch broadcast)
    - Attention mask: [1, 1, 1, S_kv] -> [1, H, 1, S_kv] (batch broadcast)
    Subtotal: 2-3 broadcasts

  Post-attention RMSNorm:
    - Same as pre-attention: 2 broadcasts
    Subtotal: 2 broadcasts

  SwiGLU MLP:
    - Activation Mcast to compute cores: physical broadcast (Mcast)
    - Gate/up projections: no broadcast (matmul)
    - Result Mcast for down projection: physical broadcast (Mcast)
    Subtotal: 2 physical broadcasts (Mcast)

  Residual connections:
    - Residual Mcast to compute cores: physical broadcast
    Subtotal: 1 physical broadcast

  Total per layer: ~9-10 broadcast operations
  Total for 61 layers (DeepSeek V3): ~550-610 broadcast operations

  Of these:
    ~120 are tile-internal (bcast_rows/cols for norm scaling)
    ~120 are cross-core (Mcast for activation distribution)
    ~360 are trivial (same shape after tiling)
```

With 550+ broadcasts per model, any manual specification is a significant developer burden. Automating broadcast detection and insertion is not optional -- it is essential for developer productivity.

---

## 6. The Gap: From PyTorch Broadcasting to TT-Blaze Execution

PyTorch's broadcasting is implemented via stride manipulation -- setting the stride to 0 in broadcast dimensions so the same memory addresses are read repeatedly. This mechanism works because PyTorch operations access memory through stride-based indexing.

Tenstorrent hardware does not use stride-based memory access for tile operations. Instead, it uses circular buffers with explicit push/pop semantics and tile-based addressing. This creates a fundamental gap:

### 6.1 No Zero-Stride Trick

On Tensix, a CB has a fixed page count and page size. There is no "stride = 0" mechanism to make a CB appear larger than it is. If an operation needs to read the same tile 4 times (broadcasting a `[1, D]` vector across 4 rows), the tile must either:

1. **Stay in the CB** while being re-read 4 times (per-tile replication within the compute kernel), or
2. **Be physically replicated** into a larger CB with 4 copies, or
3. **Be multicast** from one core to multiple cores (cross-core replication via NOC)

### 6.2 Shape Information Is Lost After Tiling

Once a tensor is decomposed into tiles ([Chapter 4](../ch04_tile_decomposition/)), the tile grid only knows "I have N tiles." The distinction between "I have 1 row of 128 tiles that should be broadcast to 4 rows" and "I have 128 tiles in some arbitrary arrangement" is lost unless the ShapeDescriptor preserves it.

### 6.3 Cross-Core vs. Within-Core Broadcasting

PyTorch broadcasting is a single-device, single-thread operation. On Tenstorrent hardware, data is sharded across multiple cores ([Chapter 4](../ch04_tile_decomposition/)). Broadcasting may need to:

- **Within a core:** Have the compute kernel re-read tiles from the same CB position
- **Across cores:** Use Mcast to replicate data from one core to all cores
- **Across devices:** Use CCL Broadcast to replicate data from one device to all devices

These three levels of broadcasting correspond to different physical mechanisms (LLK compute intrinsics, NOC multicast, Ethernet fabric) and must be selected based on the data layout and sharding configuration.

### 6.4 Broadcasting and Padding Interaction

> *Design Principle P6 (Per-Operation Padding Semantics):* Broadcasting interacts with padding in a per-operation manner. A bias addition pads the broadcast source with zeros (additive identity), while a scale multiplication pads with ones (multiplicative identity). The padding fill value must match the operation's identity element, or the broadcast produces incorrect results in the padded region. The abstraction must coordinate padding fill values ([Chapter 5](../ch05_padding/)) with broadcast patterns to ensure correctness.

When dimensions are not tile-aligned, padding can absorb shape differences:

```text
Activation: [1, 37]  -> padded to [32, 64], tile grid [1, 2]
Bias:       [37]     -> padded to [64],      tile grid [1, 2]

The padding introduces 27 extra elements (37 -> 64) that must be zero
in the bias to avoid polluting the result. After padding, both have
tile grid [1, 2] -- no tile-level broadcast is needed, but only the
first 37 elements of each tile contain meaningful data.
```

### 6.5 The Three Questions TT-Blaze Must Answer

For every binary operation in the op graph, the broadcast system must answer:

1. **What kind of broadcast is needed?** (none, rows, cols, scalar, multi-dim)
2. **At what level does the broadcast occur?** (within-tile, within-core, cross-core, cross-device)
3. **How does the broadcast affect CB sizing?** (broadcast source needs fewer pages, output needs full pages)

The next two sections of this chapter address these questions: [Section 02](./02_mapping_to_blaze_broadcast_primitives.md) maps broadcast patterns to TT-Blaze primitives and the physical Mcast mechanism, and [Section 03](./03_broadcast_shape_tracking_and_validation.md) covers how broadcast shape tracking integrates with the TypedHandle and CB sizing systems.

---

## 7. Worked Example: DeepSeek V3 Transformer Block Broadcasting

To make the broadcasting patterns concrete, let us trace every broadcast that occurs in a single DeepSeek V3 transformer block during single-token decode (B=1). DeepSeek V3 uses a hidden dimension of 7168 and 128 attention heads.

```text
DeepSeek V3 Transformer Block -- Broadcasting Audit
====================================================

1. Input RMSNorm
   x:     [1, 7168]
   gamma: [7168] -> [1, 7168]
   Pattern: No broadcast (B=1, shapes match after rank promotion)
   TT-Blaze: BroadcastRMSNorm handles via rmsnorm_mul_bcast_scalar_reuse_tiles

2. Attention QKV Projection (no bias in DeepSeek V3)
   No broadcasting needed.

3. RoPE Application
   q_head:  [1, 1, 1, 128]  (per-head, single token)
   cos:     [1, 1, 1, 128]
   sin:     [1, 1, 1, 128]
   Pattern: No broadcast (shapes match for single token)
   TT-Blaze: RoPE kernel uses mul_tiles_bcast<BroadcastType::ROW>
             for the multi-head prefill case

4. Attention Score Computation
   q: [1, 128, 1, 128], k^T: [1, 128, 128, 1]
   scores = q @ k^T: [1, 128, 1, 128]
   No broadcasting in the matmul itself.

5. Attention Score Scaling
   scores:      [1, 128, 1, 128]
   scale_factor: scalar (1/sqrt(128))
   Pattern: Scalar broadcast
   TT-Blaze: bcast_scalar -- single tile read for all output tiles

6. Attention Mask Application
   scores: [1, 128, 1, 128]
   mask:   [1, 1, 1, 128]
   Pattern: Row broadcast (broadcast across H=128 heads)
   TT-Blaze: bcast_rows on the head dimension

7. Post-Attention RMSNorm
   Same as step 1.

8. MoE Gate
   gate_scores: [1, num_experts]
   No broadcasting in the gate itself.

9. Expert Gating Weight Application
   expert_out:  [1, 2048] (per expert)
   gate_weight: [1, 1] -> scalar
   Pattern: Scalar broadcast
   TT-Blaze: bcast_scalar via EltwiseMul with enable_scalar=True

10. Expert Output Summation (reduce across experts)
    No broadcasting -- this is a reduction, not an expansion.

11. Residual Addition
    residual: [1, 7168]
    output:   [1, 7168]
    Pattern: No broadcast (shapes match)
    TT-Blaze: Direct elementwise add
```

In this single-token decode trace, most broadcasts collapse to either no-broadcast (shapes already match because B=1) or scalar broadcast (single scale factors). The more complex row and column broadcast patterns emerge during **prefill** when the sequence dimension introduces shape mismatches.

---

## 8. Worked Example: LLaMA-7B Prefill Broadcasting

During prefill with sequence length S=128, the broadcasting patterns become more varied:

```text
LLaMA-7B Prefill (B=1, S=128, D=4096, H=32, D_h=128)
======================================================

1. Input RMSNorm
   x:     [128, 4096]  (S tokens flattened into batch dim)
   gamma: [4096] -> [1, 4096]
   Pattern: Row broadcast -- gamma replicated across 128 token rows
   Tile grid: x = [4, 128] tiles, gamma = [1, 128] tiles
   Broadcast: gamma row read 4 times

2. QKV Projection + Bias (if present)
   activation: [128, 4096]
   bias:       [1, 4096] -> broadcast across 128 rows
   Pattern: Row broadcast
   Tile grid: activation = [4, 128] tiles, bias = [1, 128] tiles

3. Attention Scores
   q: [1, 32, 128, 128], k^T: [1, 32, 128, 128]
   Pattern: No broadcast (shapes match)

4. Causal Mask
   scores: [1, 32, 128, 128]
   mask:   [1, 1, 128, 128]
   Pattern: Multi-dim broadcast -- mask replicated across H=32 heads
   This is a row broadcast in the "head" dimension

5. Softmax (internal broadcasts)
   max_val:     [1, 32, 128, 1] (row-wise max)
   scores:      [1, 32, 128, 128]
   Pattern: Column broadcast -- max replicated across S columns
   sum_exp:     [1, 32, 128, 1] (row-wise sum)
   Pattern: Column broadcast -- sum replicated across S columns

6. RoPE
   cos: [1, 1, 128, 64]
   q:   [1, 32, 128, 64]
   Pattern: Multi-dim broadcast -- cos/sin across H=32 heads
   TT-Blaze: mul_tiles_bcast<BroadcastType::ROW>

7. Expert Gate Weight
   gate_weight: [128, 1] (one weight per token, per expert)
   expert_out:  [128, 2048]
   Pattern: Column broadcast -- gate replicated across D_expert
   Tile grid: gate = [4, 1] tiles, expert = [4, 64] tiles
```

This trace reveals that prefill workloads exercise all four broadcast patterns (row, column, scalar, and multi-dimensional), whereas decode workloads primarily use scalar broadcast and no-broadcast.

---

## 9. The Developer's View vs. the Hardware's View

To summarize the gap that the rest of this chapter bridges:

```text
Developer writes:           Hardware needs:
-----------------           ------------------------------------
a + b                       1. Compare ShapeDescriptors of a, b
(shapes don't match)        2. Determine broadcast pattern (rows/cols/scalar/multi)
                            3. Decide broadcast level (tile/core/device)
                            4. If cross-core: emit Mcast op
                            5. If within-core: configure compute kernel (bcast_rows/etc.)
                            6. Set CB sizing (source: 1 page, dest: full pages)
                            7. Emit CT args (is_broadcast, bcast_type)
                            8. Track output shape as broadcast-expanded
                            9. Validate expanded shape fits tile model
```

The developer sees step 0 (write `a + b`). Steps 1-9 must be handled automatically by TT-Blaze. The next section shows how.

---

*Navigation:*

- Next in chapter: [02 -- Mapping to Blaze Broadcast Primitives](./02_mapping_to_blaze_broadcast_primitives.md)
- Next: [03 -- Broadcast Shape Tracking and Validation](./03_broadcast_shape_tracking_and_validation.md)
- Previous chapter: [Chapter 7 -- CB Sizing and L1 Budget Management](../ch07_cb_sizing/01_l1_memory_model.md)
- Next chapter: [Chapter 9 -- End-to-End Op Fusion Walkthrough](../ch09_fusion_walkthrough/)
