# Shape and Padding Across Fused Boundaries

When multiple ops are fused into a single `FusedOp`, each internal boundary between constituent `emit()` calls creates a shape/padding handoff point. The ShapeTracker must update its records at every such boundary, the padding pass must detect strategy transitions (ZERO to NEG_INF, for instance), and the `reconfig()` mechanism must checkpoint state so that subsequent phases start with correct metadata. This section traces these interactions through concrete fused op chains -- matmul+SiLU+RMSNorm, attention with softmax, and the full gated MLP -- to show exactly where shape descriptors change, where padding strategies transition, where re-padding is injected or skipped, and how `reconfig()` interacts with shape/padding state.

> *Design Principle P6 (Per-Operation Padding Semantics):* Within a fused op, the consumer-determines-padding principle applies at every internal emit() boundary, not just at op-level graph edges. When `DRAMStreamingMatmul.emit()` produces ZERO-padded output and the next call is `Softmax.emit()`, the padding system must insert a ZERO-to-NEG_INF re-padding step -- even though both calls are inside the same `compose()` method.

> *Design Principle P3 (Validate at Each Lowering Step, Not Just at the End):* Every emit() boundary within a fused op is a lowering step. The ShapeTracker validates that output shapes are consistent with input shapes at each boundary, the padding system verifies strategy compatibility, and the format system checks that data types match. Validation is not deferred to the final build() call.

**What you will learn:**

- How the ShapeTracker updates at each `emit()` call within a `FusedOp.compose()`, maintaining per-CB `ShapeDescriptor` records that track logical shape, padded shape, tile geometry, and padding strategy
- How padding strategies change at fused boundaries: the ZERO-to-NEG_INF transition at softmax, the ZERO preservation through matmul->SiLU chains, and the MASK injection for mean-reduction
- Seven optimization rules for skipping redundant re-padding within fused ops
- How `reconfig()` interacts with shape and padding state: what is checkpointed, what is invalidated, and what survives across phase boundaries
- End-to-end walkthroughs through attention (seq_len=7) and gated MLP (H=4096, I=11008) showing every shape/padding decision at every internal boundary
- The padding strategy state machine for fused ops

---

## ShapeTracker Updates Within compose()

### The Per-Emit Update Cycle

Every `emit()` call within a `FusedOp.compose()` triggers a four-step update in the ShapeTracker (introduced in [Chapter 3, File 02](../ch03_tensor_adapter/02_shape_metadata_system.md) and extended for fusion in [File 01](./01_blaze_fusion_model.md)):

```text
   +-----------------------+
   |  Before emit() call   |
   |  Lookup input descs   |
   |  from ShapeTracker     |
   +-----------+-----------+
               |
               v
   +-----------------------+
   |  emit() executes      |
   |  (unchanged -- P2)    |
   |  Returns CBHandle     |
   +-----------+-----------+
               |
               v
   +-----------------------+
   |  Infer output desc    |
   |  via ShapeEngine      |
   |  (op-specific rule)   |
   +-----------+-----------+
               |
               v
   +-----------------------+
   |  Record output desc   |
   |  in ShapeTracker      |
   |  Wrap as TypedHandle  |
   +-----------------------+
```

The critical invariant: **emit() itself is never modified** (Design Principle P2 -- Carry Metadata from Both Worlds Simultaneously). The ShapeTracker wraps around `emit()`, not inside it. The `FusionShapeManager.wrap_emit()` method (from File 01) performs the lookup-before and record-after steps.

### ShapeDescriptor Fields Tracked Per Boundary

At each internal boundary, the ShapeTracker records a complete `ShapeDescriptor`:

```python
# From Chapter 3, File 02 -- ShapeDescriptor
@dataclass(frozen=True)
class ShapeDescriptor:
    logical_shape: tuple[int, ...]    # Original PyTorch shape
    padded_shape: tuple[int, ...]     # Tile-aligned shape
    tile_shape: tuple[int, int]       # (32, 32) or (16, 32)
    grid: tuple[int, int]             # Tile grid dimensions
    format: str                        # "bfloat16", "bfloat8_b", etc.
    page_size: int                     # Bytes per tile
    padding_strategy: str              # "ZERO", "NEG_INF", etc.
    padding_fill: float                # Actual fill value
    blocking: dict                     # RT, CT, KT blocking factors
    op_hint: str                       # "matmul", "eltwise", etc.
```

The padding-relevant fields are `padding_strategy` and `padding_fill`. These propagate from producer to consumer at each internal boundary, and mismatches trigger re-padding.

### Shape Inference Rules at Fused Boundaries

Different op categories produce different output shapes. The `ShapeEngine` applies category-specific rules:

| Op Category | Output Shape Rule | Padding Rule |
|-------------|-------------------|--------------|
| **Matmul** | `(M_logical, N_logical)` from `in0[-2]` and `in1[-1]` | ZERO (zero-initialized accumulators) |
| **Elementwise** (SiLU, ReLU, add, mul) | Same as input | Inherits input strategy if $f(0) = 0$; otherwise requires re-evaluation |
| **Reduction** (RMSNorm, mean) | Reduced dims become 1 or removed | Strategy depends on reduction type (see [Ch5](../ch05_padding/01_padding_strategy_taxonomy.md)) |
| **Data movement** (mcast, gather) | Same as input | INHERIT (pass-through) |
| **Broadcast** | Expanded dims from [Ch8](../ch08_broadcasting/03_broadcast_shape_tracking_and_validation.md) | Inherits source strategy |

The critical invariant: **the logical shape is always the original user-visible shape and is never modified by padding**. Padding only affects `padded_shape`.

---

## Padding Strategy Transitions Within Fused Ops

### The Internal Padding Decision Tree

At each internal boundary between `emit()` calls, the padding system makes one of four decisions:

```text
Producer emit() output -> Consumer emit() input
         |
         v
   Is consumer NONE (tile-aligned)?
   YES -> No padding needed. Skip.
   NO  |
       v
   Does producer strategy match consumer requirement?
   YES -> Compatible. No PadOp needed.
   NO  |
       v
   Can producer handle it? (producer-handles-it optimization)
   YES -> Update producer metadata. No PadOp needed.
   NO  |
       v
   INSERT REPAD NODE between producer and consumer.
```

### The Seven "Producer-Handles-It" Scenarios

Within fused ops, the same seven scenarios from [Chapter 5, File 03](../ch05_padding/03_automatic_padding_insertion_and_propagation.md) apply. Within a fused op, these optimizations are even more valuable because each avoided PadOp saves one CB ID from the 64-slot limit.

| # | Scenario | Applicable Strategy |
|---|----------|-------------------|
| 1 | `cb_from_tensor` (tiled read) | ZERO (tilization zero-fills) |
| 2 | Matmul output | ZERO (accumulator init to 0) |
| 3 | SiLU / ReLU / GELU | ZERO ($f(0) = 0$) |
| 4 | PaddedRMSNorm | ZERO (explicit internal padding) |
| 5 | SDPA | All (internal masking) |
| 6 | Elementwise add/mul | ZERO ($0 + x = x$, $0 \times x = 0$) |
| 7 | Data movement (mcast/gather) | INHERIT (passes through) |

### Strategy Transition: ZERO -> NEG_INF at Softmax

The most critical padding transition in LLM inference occurs at the softmax boundary within a fused attention op:

```text
Example: seq_len=7, tile_width=32, head_dim=64

QK scores (one row):
  [s0, s1, s2, s3, s4, s5, s6, 0.0, 0.0, ..., 0.0]
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^
   7 real scores                 25 zero-padded positions

Without re-padding (WRONG):
  softmax denominator = exp(s0) + ... + exp(s6) + 25*exp(0)
                      = exp(s0) + ... + exp(s6) + 25*1.0
  Probability leak: 25/(7+25) = 78.1% of probability mass goes to padding

With re-padding to NEG_INF (CORRECT):
  softmax denominator = exp(s0) + ... + exp(s6) + 25*exp(-inf)
                      = exp(s0) + ... + exp(s6) + 25*0.0
                      = exp(s0) + ... + exp(s6)
  All probability mass stays on real positions.
```

The formal quantification of probability mass leak for `seq_len=5`:

$$
\text{Without re-padding: } \sum_{i=0}^{4} p_i = \frac{\sum_{i=0}^{4} e^{s_i}}{\sum_{i=0}^{4} e^{s_i} + 27 \cdot e^{0}} = \frac{\text{real\_sum}}{\text{real\_sum} + 27}
$$

For typical score magnitudes ($\text{real\_sum} \approx 12$), only $12 / (12 + 27) = 30.8\%$ of probability mass falls on real tokens. With correct NEG_INF re-padding:

$$
\text{With re-padding: } \sum_{i=0}^{4} p_i = \frac{\sum_{i=0}^{4} e^{s_i}}{\sum_{i=0}^{4} e^{s_i} + 27 \cdot e^{-\infty}} = \frac{\text{real\_sum}}{\text{real\_sum} + 0} = 1.0
$$

### Strategy Preservation: ZERO Through Matmul -> SiLU -> EltwiseMul

In the gated MLP path, padding strategy stays ZERO through the entire chain:

```text
Node: DRAMStreamingMatmul (gate projection)
  Input:  activation [1,4096] padded [32,4096], ZERO
  Output: gate_out [1,11008] padded [32,11008], ZERO
  Reason: matmul accumulator initialized to 0; padded rows produce all-zero output

Node: DRAMStreamingMatmul (up projection)
  Input:  activation [1,4096] padded [32,4096], ZERO
  Output: up_out [1,11008] padded [32,11008], ZERO

Node: EltwiseMul (SiLU(gate) * up)
  Input0: gate_out, ZERO
  Input1: up_out, ZERO
  Output: gated [1,11008] padded [32,11008], ZERO
  Reason: SiLU(0) = 0*sigmoid(0) = 0*0.5 = 0; 0 * up[i] = 0
          Even at padded positions where both inputs are 0:
          SiLU(0) * 0 = 0 * 0 = 0. ZERO preserved.

No PadOps needed. The entire chain is self-consistent.
```

---

## The PaddingChainTracker Within Fused Ops

### Tracking Padding Decisions Through compose()

The `PaddingChainTracker` (from [Chapter 5, File 03](../ch05_padding/03_automatic_padding_insertion_and_propagation.md)) records every padding decision within a fused op, enabling three capabilities:

1. **Redundant re-padding detection:** If two consecutive ops both require ZERO, no re-padding is needed
2. **Final unpadding specification:** The last entry's logical_shape determines the output unpadding
3. **Diagnostic trace:** Every decision is logged for debugging

```python
# PROPOSED -- PaddingChainTracker integration with FusionShapeManager
class FusionPaddingManager:
    """Manages padding tracking within a FusedOp.compose() scope."""

    def __init__(self, padding_registry, shape_tracker: ShapeTracker):
        self._registry = padding_registry
        self._shape_tracker = shape_tracker
        self._chain = PaddingChainTracker()
        self._repad_count = 0
        self._skip_count = 0

    def check_boundary(self, producer_handle, consumer_op_type: str,
                       consumer_port: str, prefix: str) -> tuple:
        """Check padding compatibility at an internal boundary.

        Returns (needs_repad: bool, target_strategy: PaddingSpec | None).
        """
        producer_desc = self._shape_tracker.lookup(producer_handle)
        if producer_desc is None:
            return (False, None)

        # Step 1: Tile-aligned -> no padding exists
        if producer_desc.logical_shape == producer_desc.padded_shape:
            self._skip_count += 1
            return (False, None)

        # Step 2: Resolve consumer requirement
        consumer_policy = self._registry.resolve_padding_for_port(
            op_type=consumer_op_type, port_name=consumer_port)

        # Step 3: Check compatibility
        producer_policy = PaddingSpec(
            policy=PaddingPolicy[producer_desc.padding_strategy],
            fill_value=producer_desc.padding_fill)

        if _policies_compatible(producer_policy, consumer_policy):
            self._skip_count += 1
            self._chain.record(node_id=f"{prefix}_boundary",
                               op_type=consumer_op_type,
                               desc=producer_desc, was_repadded=False)
            return (False, None)

        # Step 4: Check producer-handles-it
        if _producer_can_pad_strategy(producer_desc, consumer_policy):
            self._skip_count += 1
            return (False, None)

        # Step 5: Re-padding required
        self._repad_count += 1
        return (True, consumer_policy)

    def emit_repad_if_needed(self, f: FusedProgram, producer_handle: CBHandle,
                             consumer_op_type: str, consumer_port: str,
                             prefix: str, cores) -> CBHandle:
        """Check and optionally emit a re-padding op."""
        needs_repad, target_policy = self.check_boundary(
            producer_handle, consumer_op_type, consumer_port, prefix)

        if not needs_repad:
            return producer_handle

        producer_desc = self._shape_tracker.lookup(producer_handle)
        repadded = PadOp.emit(
            f, producer_handle, prefix=f"{prefix}_repad",
            cores=cores, fill_value=target_policy.fill_value,
            logical_width=producer_desc.logical_shape[-1],
            padded_width=producer_desc.padded_shape[-1],
            logical_height=producer_desc.logical_shape[-2],
            padded_height=producer_desc.padded_shape[-2],
            data_format=producer_desc.format)

        new_desc = producer_desc.with_padding(
            padding_strategy=target_policy.policy.value,
            padding_fill=target_policy.fill_value)
        self._shape_tracker.record(repadded, new_desc)
        return repadded
```

---

## Optimization: Seven Rules for Skipping Redundant Re-Padding

Not every boundary crossing requires an explicit PadOp. Seven optimization rules eliminate redundant re-padding within fused ops:

### Rule 1: Same Strategy Passthrough

When consecutive sub-ops require the same padding strategy, no re-padding is needed:

```text
matmul (ZERO) -> SiLU (ZERO) -> eltwise_mul (ZERO) -> matmul (ZERO)
  All ZERO-compatible. Zero PadOps inserted.
```

### Rule 2: f(0) = 0 Preservation

Activation functions where $f(0) = 0$ preserve ZERO padding without any action:

| Function | $f(0)$ | Preserves ZERO? |
|----------|--------|-----------------|
| SiLU | $0 \cdot \sigma(0) = 0$ | Yes |
| ReLU | $\max(0, 0) = 0$ | Yes |
| GELU | $0 \cdot \Phi(0) = 0$ | Yes |
| Tanh | $\tanh(0) = 0$ | Yes |
| Sigmoid | $\sigma(0) = 0.5$ | **No** -- re-padding required |

### Rule 3: Producer-Handles-It

Seven producer types produce correctly-padded output without a separate PadOp:
- `cb_from_tensor` (pre-padded during tilization)
- Matmul (zero-initialized accumulation dest)
- Activation functions where $f(0) = 0$
- PaddedRMSNorm (explicit internal padding)
- SDPA (internal masking)
- Elementwise add/mul with zero input
- Data movement ops (mcast, gather)

### Rule 4: NONE Eliminates All Padding Ops

When `padded_shape == logical_shape` (all dimensions tile-aligned), `padding_strategy = "NONE"` and no padding operations are needed at any boundary:

```text
LLaMA-7B gated MLP: hidden=4096, intermediate=11008
  4096 % 32 = 0, 11008 % 32 = 0
  All shapes tile-aligned -> padding_strategy = NONE everywhere
  Zero PadOps in the entire fused op.
```

### Rule 5: ZERO-to-MASK Compatibility

MASK-strategy ops (mean reduction, LayerNorm) accept ZERO-padded input -- they add count correction on top. No re-padding of the data itself is needed; only the mask tensor must be generated:

```python
# The data stays zero-padded. Only the mask CB is added.
if producer_strategy == "ZERO" and consumer_strategy == "MASK":
    # No data re-padding. Generate mask tensor for count correction.
    mask_handle = _generate_padding_mask(f, desc, prefix="mask")
    tracker.record(mask_handle, mask_desc)
```

### Rule 6: NEG_INF-to-IDENTITY(max) Compatibility

For max-reduction, the identity element is $-\infty$. If the producer already uses NEG_INF padding, no re-padding is needed because $\max(x, -\infty) = x$.

### Rule 7: Fan-Out With Mixed Requirements

When a single producer feeds multiple consumers with different padding requirements, the producer's output is not re-padded. Instead, per-consumer re-padding is inserted only for consumers that need it:

```text
matmul_out (ZERO)
  |
  +---> softmax (NEG_INF)  -> INSERT PadOp (ZERO -> NEG_INF)
  |
  +---> residual_add (ZERO) -> NO re-padding needed
```

### Optimization Impact

For production LLM workloads with tile-aligned dimensions:

| Model | Fused Op | Raw Boundary Crossings | PadOps After Optimization |
|-------|----------|----------------------|--------------------------|
| LLaMA-7B | Gated MLP (6 stages) | 5 | 0 (Rule 4: all NONE) |
| LLaMA-7B | Attention (8 stages) | 7 | 0 (NONE + internal SDPA masking) |
| DeepSeek V3 | MoE Expert (6 stages) | 5 | 0 (Rule 4) |
| Custom (hidden=4001) | Gated MLP (6 stages) | 5 | 0 (Rules 1-3: ZERO chain) |
| Custom (seq_len=5) | Attention (8 stages) | 7 | 1 (Rule fails at QK->softmax) |

The only common case requiring an explicit PadOp within a fused op is the QK-scores-to-softmax boundary when the sequence length is not a multiple of 32.

---

## reconfig() and Shape/Padding State

### What Happens at a reconfig() Boundary

The `reconfig()` call in a `FusedProgram` creates a phase boundary. The `CbReconfig` kernel reprograms CB registers via the reconfig tensor (264 uint32 words per core, as described in [Chapter 6, File 03](../ch06_data_formats/03_format_conversion_and_cb_reconfig.md)). After a `reconfig()` boundary:

- CB IDs may be reassigned (via `CircularBufferIdManager`)
- CB formats may change (different `data_format` and `page_size`)
- Scratch CB L1 addresses may move (the arena is repacked)

### Shape and Padding State Across reconfig()

**State that survives reconfig():**

| State Category | Survives? | Reason |
|---------------|-----------|--------|
| Tensor-backed CB ShapeDescriptors | YES | Fixed L1 addresses persist |
| Direct-address weight ShapeDescriptors | YES | Pre-loaded, descriptors remain valid |
| Padding strategy of surviving CBs | YES | Fill value is in the data, not CB metadata |
| Logical shape tracking | YES | Mathematical, not hardware-dependent |

**State that is invalidated by reconfig():**

| State Category | Invalidated? | Reason |
|---------------|-------------|--------|
| Scratch CB ShapeDescriptors | YES | Overwritten by new phase allocations |
| Scratch CB padding metadata | YES | New data has new padding properties |
| CB ID -> format mapping (scratch) | REMAPPED | Manager may assign different formats |
| Page count of scratch CBs | YES | New phase may allocate different page counts |

### Checkpointing and Bridging at reconfig()

```python
# PROPOSED -- ShapeTracker reconfig() handling
class ShapeTracker:
    def checkpoint(self, phase_name: str) -> None:
        """Snapshot current state before a reconfig() boundary."""
        self._checkpoints[phase_name] = dict(self._registry)

    def bridge_across_reconfig(
        self, old_handle: CBHandle, new_handle: CBHandle
    ) -> None:
        """Map a pre-reconfig CBHandle to its post-reconfig replacement.

        After reconfig(), a CB may get a new CB ID (via CircularBufferIdManager
        format-aware reuse) but still hold the same data. This method transfers
        the ShapeDescriptor from the old identity to the new identity.
        """
        desc = self._registry.get(id(old_handle))
        if desc is not None:
            self._registry[id(new_handle)] = desc
```

### Worked Example: reconfig() in a Three-Phase Gated MLP

```text
Phase 0: RMSNorm + Mcast
  CB 0 (act input):     logical=(1,4096), padded=(32,4096), ZERO, bfloat16
  CB 1 (gamma):         logical=(1,4096), padded=(32,4096), NONE, bfloat16
  CB 2 (rmsnorm out):   logical=(1,4096), padded=(32,4096), ZERO, bfloat16
  CB 3 (mcast out):     logical=(1,4096), padded=(32,4096), ZERO, bfloat16

  -- reconfig() boundary --
  ShapeTracker.checkpoint("phase_0")
  CB 3 survives: same data, same format -> bridge_across_reconfig()

Phase 1: Gate matmul + Up matmul + SiLU + EltwiseMul
  CB 3 (mcast act):     logical=(1,4096), padded=(32,4096), ZERO, bfloat16
    [bridged from Phase 0 -- ShapeDescriptor preserved]
  CB 4 (gate weight):   logical=(4096,11008), padded=(4096,11008), NONE, bfloat8_b
  CB 5 (up weight):     logical=(4096,11008), padded=(4096,11008), NONE, bfloat8_b
  CB 6 (gate output):   logical=(1,11008), padded=(32,11008), ZERO, bfloat16
    [inferred: matmul (1,4096)x(4096,11008) -> (1,11008)]
  CB 7 (up output):     logical=(1,11008), padded=(32,11008), ZERO, bfloat16
  CB 8 (silu*up out):   logical=(1,11008), padded=(32,11008), ZERO, bfloat16

  -- reconfig() boundary --
  ShapeTracker.checkpoint("phase_1")
  CB 8 survives

Phase 2: Down matmul
  CB 8 (silu*up in):    logical=(1,11008), padded=(32,11008), ZERO, bfloat16
    [bridged from Phase 1]
  CB 9 (down weight):   logical=(11008,4096), padded=(11008,4096), NONE, bfloat4_b
  CB 6 (down output):   logical=(1,4096), padded=(32,4096), ZERO, bfloat16
    [CB ID 6 reused: same (bfloat16, 32x32) format key]
```

At each boundary, the ShapeDescriptor for the surviving CB is identical on both sides -- `reconfig()` does not change the data, only the CB configuration metadata.

---

## End-to-End Walkthrough: Fused Attention (seq_len=7, head_dim=64)

This walkthrough traces every shape/padding decision through a fused attention op with a non-aligned sequence length.

```text
Configuration:
  seq_len = 7, head_dim = 64, tile = (32, 32)
  Padded seq = 32 (7 -> next multiple of 32)
  Q, K, V: logical [1, 7, 64], padded [1, 32, 64]
```

### Boundary 1: Input -> Q/K/V Projections

```text
Operation: Matmul (Q projection)
  Input:  [1, 7, 64] padded [1, 32, 64], ZERO (tilization)
  Weight: [64, 64] padded [64, 64], NONE (tile-aligned)

  Padding check:
    Producer: ZERO (cb_from_tensor, tilized)
    Consumer: ZERO (matmul.in0)
    Compatible? YES

  Shape inference:
    logical_out = (1, 7, 64), padded_out = (1, 32, 64), grid = (1, 2)

  No PadOp needed.
```

### Boundary 2: Q, K -> QK Scores

```text
Operation: Matmul (QK^T)
  in0 (Q):  [1,7,64] padded [1,32,64], ZERO
  in1 (K^T): [1,64,7] padded [1,64,32], ZERO

  Padding check: Both ZERO, matmul accepts ZERO -> compatible

  Shape inference:
    Contract K=64/32=2 tiles
    logical_out = (1, 7, 7), padded_out = (1, 32, 32), grid = (1, 1)

  CT args derived: k_num_tiles = 2, out_w_per_core = 1
  No PadOp needed.
```

### Boundary 3: QK Scores -> Softmax (CRITICAL TRANSITION)

```text
Operation: Softmax
  Input: scores [1,7,7] padded [1,32,32], ZERO

  Padding check:
    Producer: ZERO (matmul output)
    Consumer: NEG_INF (softmax.scores -- from padding registry)
    _policies_compatible(ZERO, NEG_INF)? NO
    _producer_can_pad(matmul, NEG_INF)? NO

  ** RE-PADDING REQUIRED: ZERO -> NEG_INF **

  RepadOp emitted:
    Input:  scores CB (1 tile, 2,048 bytes)
    Action: For each of 32 rows:
      Row 0-6 (real): copy cols 0-6 as-is, fill cols 7-31 with -inf
      Row 7-31 (padding): fill all 32 cols with -inf
    Output: repadded_scores CB (1 tile, 2,048 bytes)

  Fill value encoding (bfloat16):
    -inf = 0xFF80 (sign=1, exponent=0xFF, mantissa=0)

  Cost:
    +1 CB ID (now using CB ID N+1 for repadded output)
    +2,048 bytes L1 (one tile)
    +~0.5 us latency (NCRISC-only fill, zero SFPU cycles)
```

### Boundary 4: Softmax -> Probs @ V

```text
Softmax output:
  Real rows (0-6): proper probability distribution over cols 0-6
    exp(-inf) = 0 for cols 7-31, so probabilities sum to 1.0 correctly
  Padding rows (7-31): all-zero (exp(-inf)/sum(exp(-inf)) handled by kernel)

  padding_strategy of output: ZERO
    (padding positions have probability 0.0)

Operation: Matmul (probs @ V)
  in0: probs [1,7,7] padded [1,32,32], ZERO
  in1: V [1,7,64] padded [1,32,64], ZERO

  Padding check: Both ZERO, matmul accepts ZERO -> compatible
  No PadOp needed.

Note: Softmax output is ZERO because exp(-inf) = 0. This means only
1 re-padding is needed in the entire attention block (ZERO -> NEG_INF
before softmax), not 2 -- the softmax-to-matmul boundary is ZERO-to-ZERO.
```

### Boundary 5: Output Projection and Unpadding

```text
Operation: Matmul (output projection)
  in0: attn_out [1,7,64] padded [1,32,64], ZERO
  in1: W_o [64,64], NONE (tile-aligned)

  Padding check: ZERO compatible with matmul -> skip

  Final unpadding:
    PaddingChainTracker.get_final_unpad_spec() returns:
      {"needs_unpad": True,
       "logical_shape": (1, 7, 64),
       "padded_shape": (1, 32, 64),
       "slices": (slice(0,1), slice(0,7), slice(0,64))}
```

### Attention Summary

```text
Total internal boundaries:  5
Padding checks performed:   5
Re-pads inserted:           1 (ZERO -> NEG_INF at softmax)
Producer-handles-it:        3 (matmul x2, tilized read)
Tile-aligned skip:          1 (weight projection)
Extra CB IDs used:          1
Extra L1 used:              2,048 bytes (one tile)
```

---

## End-to-End Walkthrough: Gated MLP (H=4096, I=11008, 14 cores)

This walkthrough traces shape and padding through the 11-node gated MLP from File 01, using LLaMA-7B dimensions. All dimensions are tile-aligned.

### Phase 0: Activation Distribution and Projections

```text
Node 1: mcast(activation)
  Input:  [1, 4096] padded [32, 4096], ZERO
  Output: same shape, replicated to 14 cores
  Padding check: data movement -> INHERIT -> skip

Node 2: kn_sliced_matmul(act, gate_weights)
  in0: [1,4096] padded [32,4096], ZERO -- from mcast
  in1: [4096,11008] padded [4096,11008], NONE -- tile-aligned
  Padding check: ZERO/NONE -> skip
  Shape: logical=(1,11008), padded=(32,11008), grid=(1,344)
  CT args: k_num_tiles=128, out_w_per_core=25

Node 3: kn_sliced_matmul(act, up_weights)
  Same shape inference as Node 2

Node 4: gather(gate_partial)
  Shape unchanged, data movement only. Padding: INHERIT -> skip

Node 5: gather(up_partial)
  Same as Node 4

Node 6: gated_reduce(gate, up) = SiLU(gate) * up
  in0: gate [1,11008] padded [32,11008], ZERO
  in1: up [1,11008] padded [32,11008], ZERO
  Padding check:
    SiLU on in0: f(0) = 0 -> ZERO preserved (Rule 2, Scenario 3)
    Multiply: 0 * up[i] = 0 -> ZERO preserved (Rule 1, Scenario 6)
  Shape: (1,11008)/(32,11008), ZERO
```

### reconfig() Boundary

```text
f.reconfig()

FusionShapeManager.checkpoint_reconfig():
  Phase: 0 -> 1
  Surviving handles: mcast_act (tensor-backed), gate_wt, up_wt, down_wt
  Invalidated handles: gate_out scratch, up_out scratch, intermediates

  Special: gated_out from Node 6 survives because it feeds
  Node 7 (mcast). Preserved via CircularBufferIdManager.

  PaddingChainTracker: continues tracking across phases
```

### Phase 1: Down Projection and Residual

```text
Node 7: mcast(gated_out)
  Input: gated_out [1,11008] padded [32,11008], ZERO
  Padding check: data movement -> INHERIT -> skip

Node 8: matmul(mcast_gated, down_weights)
  in0: [1,11008] padded [32,11008], ZERO
  in1: [11008,4096] padded [11008,4096], NONE
  Shape: logical=(1,4096), padded=(32,4096), grid=(1,128)
  CT args: k_num_tiles=344, out_w_per_core=9

Node 9: mcast(bias)
  Shape: [1,4096] padded [32,4096], ZERO. INHERIT -> skip

Node 10: residual_add(down_out, bias)
  Padding check: eltwise add -> 0 + x = x -> ZERO compatible (Rule 1, Scenario 6)

Node 11: gather(result)
  Final ShapeDescriptor: (1,4096)/(32,4096), ZERO
```

### Gated MLP Summary

```text
Total internal boundaries:  10
Padding checks performed:   10
Re-pads inserted:           0
Producer-handles-it:        4 (2x matmul, 1x SiLU, 1x eltwise add)
Data movement skip:         4 (2x mcast, 2x gather)
Tile-aligned skip:          2 (weight inputs)
reconfig() checkpoints:     1
Extra CB IDs used:          0
Extra L1 used:              0
```

---

## Padding Strategy State Machine

The padding strategy at any internal boundary follows a state machine. The states are the five padding strategies; the transitions are triggered by ops:

```text
                     +------+
           +-------->| NONE |<--------+
           |         +------+         |
           |         (tile-aligned)   |
           |                          |
    tilized read                   exact dims
           |                          |
           v                          v
        +------+    matmul/SiLU    +------+
        | ZERO |<----------------->| ZERO |
        +------+    eltwise add    +------+
           |                          ^
           |                          |
       softmax input              softmax output
       (MUST repad)               (exp(-inf)=0)
           |                          |
           v                          |
       +---------+                    |
       | NEG_INF |--------------------+
       +---------+
           |
       SDPA internal
           |
           v
       +------+
       | MASK | (mean reduction, LayerNorm)
       +------+
           |
       identity reduction
           |
           v
       +----------+
       | IDENTITY | (max/min/product reduction)
       +----------+
```

**Key transitions:**

| From | To | Trigger | Automatic? |
|------|----|---------|-----------|
| ZERO | ZERO | matmul, SiLU, ReLU, GELU, eltwise add/mul | Yes (preserved) |
| ZERO | NEG_INF | softmax input boundary | Yes (RepadOp inserted) |
| NEG_INF | ZERO | softmax output | Yes (exp(-inf) = 0) |
| ZERO | MASK | mean reduction input | Yes (MASK adds count correction) |
| ZERO | IDENTITY | max/min reduction input | Depends on op |
| Any | NONE | tile-aligned output | Yes (no padding exists) |

---

## Error Taxonomy: Shape and Padding Errors Within Fused Ops

| Error | Root Cause | Symptom | Detection Point |
|-------|-----------|---------|----------------|
| **Stale ShapeDescriptor after reconfig** | Scratch CB invalidated but tracker holds old desc | Wrong CT args for Phase 1 ops | `checkpoint_reconfig()` clears stale entries |
| **Missing re-padding at softmax** | ZERO padding not converted to NEG_INF | 78.1% probability leak (seq_len=7) | `check_boundary()` returns `needs_repad=True` |
| **Redundant re-padding** | Re-pad inserted between two ZERO-compatible ops | Wasted CB ID and L1 memory | `_policies_compatible()` check prevents it |
| **Logical shape corruption** | Reshape changes padding dims without updating tracker | Wrong unpadding at output | `ShapeTracker.record()` validates consistency |
| **Cross-phase padding mismatch** | Phase 0 writes ZERO, Phase 1 expects NEG_INF via reused CB | Garbage padding values | `checkpoint_reconfig()` invalidates scratch state |
| **PadOp CB ID exhaustion** | Too many RepadOps in deeply fused op | `RuntimeError` from CircularBufferIdManager | Pre-fusion analysis estimates CB ID count |

---

## Integration with Design Principles

- **P1 (Separate Computation Intent from Execution Strategy):** The developer writes `compose_fused_attention(Q, K, V)`. The ShapeTracker and padding pass determine all boundary crossings, re-padding requirements, and CB sizing automatically. The developer never specifies padding strategies.

- **P2 (Carry Metadata from Both Worlds Simultaneously):** Each fused boundary carries three simultaneous representations: `logical_shape` (PyTorch world), `padded_shape` (hardware world), and `padding_strategy` (semantic world). Losing any one produces bugs.

- **P6 (Per-Operation Padding Semantics):** The consumer-determines-padding principle operates at every fused boundary, not just at graph edges. The padding registry is consulted for each sub-op's input ports.

---

## Key Takeaways

- The ShapeTracker updates at every `emit()` boundary within a `FusedOp.compose()`. The four-step cycle -- lookup inputs, call emit() unchanged, infer output shape, record result -- preserves Design Principle P2 while maintaining complete shape metadata through the fusion chain.
- Padding strategy transitions follow the **consumer-determines-padding** principle (P6) at every internal boundary. The most critical transition is ZERO-to-NEG_INF at softmax (1 PadOp); the gated MLP requires zero transitions because ZERO padding is preserved through matmul, SiLU, and elementwise ops.
- **Seven optimization rules** eliminate redundant re-padding: same-strategy passthrough, $f(0) = 0$ preservation, producer-handles-it (7 scenarios), NONE for tile-aligned shapes, ZERO-to-MASK compatibility, NEG_INF-to-IDENTITY(max) compatibility, and per-consumer fan-out re-padding. For production LLMs, these typically eliminate **all** PadOps within fused ops.
- The `reconfig()` boundary checkpoints shape/padding state: tensor-backed CB descriptors survive, scratch CB descriptors are invalidated, and the `PaddingChainTracker` continues tracking across phases. The `bridge_across_reconfig()` method transfers descriptors for surviving CBs.
- Softmax output is ZERO (exp(-inf) = 0), so only 1 re-padding is needed in fused attention (before softmax), not 2. The softmax-to-matmul boundary is ZERO-to-ZERO (compatible).

## Source Files

- `blaze/blaze_op.py` -- `MicroOp.emit()`, `FusedOp.compose()`: the emit/compose chain where internal boundaries occur
- `blaze/ops/padded_rmsnorm/op.py` -- `PaddedRMSNorm.emit()`: canonical internal padding example
- `blaze/ops/sdpa/op.py` -- `SDPADecode.emit()`: internal masking that handles NEG_INF padding
- `blaze/ops/swiglu/op.py` -- `SwigluOp.compose()`: five-MicroOp chain with reconfig() boundaries
- `blaze/_gated_mlp.py` -- `build_gated_mlp_graph()`: 11-node graph where all padding checks result in "skip"
- `blaze/fused_program.py` -- `FusedProgram.reconfig()`: phase boundary triggering state checkpointing
- `blaze/cb_reconfig.py` -- `CircularBufferIdManager`: CB ID management across reconfig boundaries

---

**Previous:** [01_blaze_fusion_model.md](./01_blaze_fusion_model.md) | **Next:** [03_format_transitions_within_fused_ops.md](./03_format_transitions_within_fused_ops.md)
