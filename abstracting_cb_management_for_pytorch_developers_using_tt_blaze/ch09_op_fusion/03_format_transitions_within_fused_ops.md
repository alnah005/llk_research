# Format Transitions Within Fused Ops

Within a fused op, consecutive kernel phases may require different data formats: a matmul accumulates in bfloat16 but the downstream RMSNorm needs bfloat16 for its reciprocal-square-root; gate weights are stored as bfloat8_b but the SiLU activation operates on bfloat16 values; the down-projection weights may use bfloat4_b for memory savings. Each of these transitions must be handled correctly -- either by the hardware unpacker (free), by a `cb_alias()` zero-cost reinterpretation, or by an explicit conversion op. This section traces format transitions through concrete fused op chains, explains the `FormatPolicy` negotiation at each internal edge, details the `CircularBufferIdManager`'s role in format-aware CB ID reuse, and shows how TensorAdapter automates what developers currently configure by hand.

> *Design Principle P5 (Operate on Graphs, Not Individual Tensors):* Format negotiation within a fused op requires visibility into all constituent emit() calls. A per-emit format decision may force an expensive conversion; a graph-level view of all emit() calls within compose() enables the negotiator to minimize total conversion cost.

> *Design Principle P4 (Provide Defaults with Per-Decision Overrides):* The precision profile sets the default format for all CBs within a fused op. Per-edge overrides allow targeting specific transitions -- like forcing HiFi4 + float32 accumulation for the softmax within an otherwise LoFi attention block.

**What you will learn:**

- How `cb_alias()` provides zero-cost format reinterpretation for FIFO CBs within fused ops, and why it cannot be used for direct-address CBs
- The `FormatPolicy` negotiation protocol applied to each internal edge of a fused op, including the three preference levels (preferred, tolerable, required) and the incompatible set
- How `CircularBufferIdManager` assigns CB IDs across format-differing phases, with the `(data_format, tile)` reuse key preventing format aliasing bugs
- A worked example of format transitions in a matmul (bfloat8_b weights) -> RMSNorm (bfloat16) -> matmul (bfloat4_b weights) chain
- How the three precision profiles (performance, balanced, accuracy) control format selection within fused ops, including the canonical math fidelity names (LoFi and HiFi4) and `fp32_dest_acc_en` settings
- Format consistency via CB reconfig at phase boundaries, referencing `cb_reconfig.py` and `cb_reconfig_builder.py`
- The complete manual vs. abstracted comparison for format management within a fused gated MLP

---

## cb_alias(): Zero-Cost Format Reinterpretation

### Mechanism

The `cb_alias()` method on `FusedProgram` creates a second CB ID that points to the same L1 memory as an existing CB, but with a different format descriptor. This is a metadata-only operation: no data is moved, no conversion is performed, and no extra L1 memory is consumed.

```python
# EXISTING -- from blaze/fused_program.py (simplified)
def cb_alias(self, source_handle: CBHandle, *,
             dtype: ttnn.DataType,
             page_size: int = None,
             tile: ttnn.Tile = None) -> CBHandle:
    """Create a zero-cost alias of source_handle with a different format.

    The alias shares the same L1 address as source_handle.
    The SFPU unpacker handles format conversion transparently.

    Constraints:
      - source_handle must be FIFO mode (not direct-address)
      - tile geometry must be compatible (same tile dimensions)
    """
    if source_handle.is_direct_address:
        raise ValueError(
            "cb_alias does not support direct-address CBs. "
            "Use tile_row_convert or retilize instead."
        )
    alias_id = self._cb_manager.get_cb_id(dtype, tile or source_handle.tile)
    alias_handle = CBHandle(
        cb_id=alias_id,
        l1_addr=source_handle.l1_addr,  # Same address!
        data_format=dtype,
        page_size=page_size or _compute_page_size(dtype, tile),
        num_pages=source_handle.num_pages,
        tile=tile or source_handle.tile,
        is_direct_address=False,
    )
    return alias_handle
```

### When cb_alias() Is Safe Within Fused Ops

The safety rule is directional: aliasing is safe only in the BFP-to-standard (unpack) direction, and only for FIFO-mode CBs:

| Source Format | Alias Format | Direction | Safe? | Mechanism |
|--------------|-------------|-----------|-------|-----------|
| bfloat8_b | bfloat16 | Unpack | Yes | Hardware unpacker expands BFP8 to bf16 |
| bfloat4_b | bfloat16 | Unpack | Yes | Hardware unpacker expands BFP4 to bf16 |
| bfloat16 | float32 | Accumulator | Yes | `fp32_dest_acc_en` widens dest register |
| bfloat16 | bfloat8_b | Pack | ~Free | Packer compresses during write |
| bfloat16 | bfloat8_b (read) | **Reverse** | **No** | Reading bf16 bytes as BFP block structure produces garbage |
| bfloat8_b | bfloat4_b (read) | **Cross-BFP** | **No** | Different BFP block layouts, shared exponent mismatch |
| Any | Any (direct-address) | N/A | **No** | DA CBs have fixed page addresses dependent on page_size |

The direct-address restriction exists because direct-address CBs have hardware-programmed page addresses that depend on the page size. Changing the format changes the page size, which invalidates the address computation. FIFO CBs use a circular pointer that is format-agnostic.

### cb_alias() vs. cb_from_view()

Both provide zero-copy format reinterpretation, but at different levels:

| Feature | cb_alias() | cb_from_view() |
|---------|-----------|---------------|
| Creates new CB ID | Yes | Yes |
| Shares L1 memory | Yes | Yes |
| Can change format | Yes | Yes |
| Can change tile geometry | No (same dims) | Yes (compatible dims) |
| Works on direct-address | No | No |
| Primary use case | Format alias within phase | Tile reinterpretation |

---

## FormatPolicy Negotiation Per Internal Edge

### Applying the Negotiation Protocol Within compose()

The format negotiation protocol from [Chapter 6, File 02](../ch06_data_formats/02_format_negotiation_protocol.md) operates at the graph level for the Graph API path. Within a `FusedOp.compose()`, the same protocol can be applied to each internal edge between `emit()` calls, treating each edge as a producer-consumer pair.

```text
Internal edge: DRAMStreamingMatmul.emit() output -> EltwiseMul.emit() input

Producer output preference:
  FormatPreference(
    preferred={bfloat16},         -- matmul naturally produces bfloat16
    tolerable={bfloat8_b},        -- packer can compress
    required=None,
    incompatible={bfloat4_b}      -- too much precision loss for activations
  )

Consumer input preference:
  FormatPreference(
    preferred={bfloat16},         -- eltwise mul native format
    tolerable={bfloat8_b},        -- unpacker expands
    required=None,
    incompatible=frozenset()
  )

Negotiation: best_common() -> bfloat16 (shared preferred)
Decision: No conversion needed. Both prefer bfloat16.
```

### The Fused-Internal Negotiation Pass

The negotiation algorithm extends the two-pass graph walk from [Chapter 6, File 02](../ch06_data_formats/02_format_negotiation_protocol.md) to internal edges within fused subgraphs, with additional considerations for CB ID pressure, phase awareness, and L1 sharing:

```python
# PROPOSED -- format negotiation for internal fused edges
class FusedFormatNegotiator:
    """Negotiate formats for internal edges within a fused op.

    Extends the graph-level negotiation from Chapter 6, File 02,
    with phase-awareness and CB ID pressure considerations.
    """

    def negotiate_fused_subgraph(
        self,
        ops: list[OpNode],
        edges: list[InternalEdge],
        phase_boundaries: list[int],
        profile: PrecisionProfile,
        l1_budget_remaining: int,
    ) -> FusedFormatPlan:
        """Negotiate formats for all internal edges.

        Steps:
        1. Forward pass: propagate format preferences through ops
        2. Backward pass: resolve conflicts, prefer free conversions
        3. Phase-aware pass: identify reconfig-eligible transitions
        4. L1 budget pass: estimate cost of each transition mechanism
        5. CB ID pressure pass: minimize format diversity

        Returns FusedFormatPlan with per-edge format assignments
        and the transition mechanism for each edge.
        """
        # Step 1-2: same as Chapter 6, File 02 algorithm
        assignments = self._two_pass_negotiate(ops, edges, profile)

        # Step 3: identify transitions at phase boundaries
        for edge in edges:
            if self._crosses_phase_boundary(edge, phase_boundaries):
                edge.transition = "reconfig"  # Free at phase boundaries
            elif self._can_alias(edge, assignments):
                edge.transition = "alias"     # Free via cb_alias
            elif self._is_free_unpack(edge, assignments):
                edge.transition = "unpack"    # Free via hardware unpacker
            else:
                edge.transition = "convert"   # Explicit conversion needed

        # Step 4: estimate L1 cost
        total_convert_cost = 0
        for edge in edges:
            if edge.transition == "convert":
                cost = self._conversion_l1_cost(edge, assignments)
                total_convert_cost += cost
                if total_convert_cost > l1_budget_remaining:
                    edge.transition = self._find_cheaper_alternative(
                        edge, assignments
                    )

        # Step 5: minimize CB ID pressure
        format_counts = Counter(assignments.values())
        if len(format_counts) > 3:  # More than 3 distinct formats
            assignments = self._unify_minority_formats(assignments, edges)

        return FusedFormatPlan(assignments=assignments, edges=edges)
```

### The Three-Zone Format Model Within Attention

A fused attention block contains three distinct format zones, each with different requirements:

```text
Zone 1: QKV Projections (matmul-dominated)
  Weight format:      bfloat8_b (performance profile) or bfloat16 (balanced)
  Activation format:  bfloat16
  Accumulation:       bfloat16 dest (LoFi) or float32 dest (accuracy)
  Math fidelity:      LoFi or HiFi4

Zone 2: Softmax
  Input format:       bfloat16 (REQUIRED -- bfloat8_b cannot represent -inf precisely)
  Accumulation:       float32 dest (REQUIRED -- exp() needs precision)
  Math fidelity:      HiFi4 (REQUIRED -- softmax safety override from Ch6 File 04)
  Output format:      bfloat16

Zone 3: Attention Value Matmul
  Weight format:      bfloat16 (V is activation, not compressed weight)
  Activation format:  bfloat16 (softmax output)
  Same as Zone 1 for accumulation/fidelity
```

The `required` constraint on Zone 2 is the softmax safety override described in [Chapter 6, File 04](../ch06_data_formats/04_precision_profiles_and_user_overrides.md). Even under the "performance" profile (which defaults to LoFi + bfloat8_b), softmax always uses HiFi4 + bfloat16 + float32 accumulation. This override is enforced by the format negotiation pass:

```python
# EXISTING -- softmax format override (from op registry)
SOFTMAX_FORMAT_PREFS = {
    "scores": FormatPreference(
        preferred={ttnn.bfloat16},
        tolerable=frozenset(),
        required={ttnn.bfloat16},
        incompatible={ttnn.bfloat8_b, ttnn.bfloat4_b},
    ),
}
```

### Per-Edge Format Decisions for the Gated MLP

The 11-node gated MLP has 12 internal edges. Under the "performance" precision profile, the format at each edge:

```text
Edge 1:  mcast -> gate_mm.in0        bfloat16 (activation)
Edge 2:  mcast -> up_mm.in0          bfloat16 (activation)
Edge 3:  gate_wt -> gate_mm.in1      bfloat8_b (weight, direct-address)
Edge 4:  up_wt -> up_mm.in1          bfloat8_b (weight, direct-address)
Edge 5:  gate_mm -> gather            bfloat16 (matmul output)
Edge 6:  up_mm -> gather              bfloat16 (matmul output)
Edge 7:  gather(gate) -> gated_reduce bfloat16 (data movement, format preserved)
Edge 8:  gather(up) -> gated_reduce   bfloat16 (data movement, format preserved)
Edge 9:  gated_reduce -> mcast        bfloat16 (eltwise output)
Edge 10: mcast -> down_mm.in0         bfloat16 (activation)
Edge 11: down_wt -> down_mm.in1       bfloat8_b (weight, direct-address)
Edge 12: down_mm -> residual_add      bfloat16 (matmul output)
```

**Format transitions count: 0 explicit conversions needed.**

All weight-to-matmul edges (3, 4, 11) are bfloat8_b-to-bfloat16 conversions handled entirely by the hardware unpacker. No PadOp-style conversion node is inserted. No `cb_alias()` is needed because the matmul kernel's unpack stage handles bfloat8_b natively.

---

## CircularBufferIdManager and Format-Aware CB ID Reuse

### The Format-Keyed Reuse Rule

As detailed in [Chapter 6, File 03](../ch06_data_formats/03_format_conversion_and_cb_reconfig.md), the `CircularBufferIdManager` reuses CB IDs only when the `(data_format, tile)` key matches:

```text
Reuse rule:
  CB ID X was used in Phase 0 with (bfloat16, 32x32)
  Phase 1 requests (bfloat16, 32x32)
  -> Reuse CB ID X (format matches)

No-reuse rule:
  CB ID Y was used in Phase 0 with (bfloat8_b, 32x32)
  Phase 1 requests (bfloat4_b, 32x32)
  -> Allocate new CB ID Z (format mismatch: bfloat8_b != bfloat4_b)
```

This rule prevents a subtle corruption: if Phase 0 writes bfloat8_b data (1,088 bytes/page) to CB ID Y and Phase 1 reads it as bfloat4_b (576 bytes/page), the page boundary computation produces wrong addresses. Even though the data was overwritten by Phase 1, the reconfig tensor programs the CB registers with page_size from the Phase 1 format -- and if Phase 0 and Phase 1 share a CB ID but with different page sizes, the reconfig tensor entry for that ID can only hold one page_size value.

### Walked Example: CB ID Allocation for Three-Format MLP

Consider a gated MLP variant where each weight matrix uses a different format:

```text
Gate weights: bfloat8_b  (aggressive compression for gate projection)
Up weights:   bfloat8_b  (same format as gate)
Down weights: bfloat4_b  (more aggressive compression for down projection)
```

**Phase 0 (Gate + Up + SiLU*Up):**

```python
# CB allocation trace within CircularBufferIdManager
ctx0 = manager.create_context()

# Activation (bfloat16, 32x32)
cb_act = ctx0.get_cb_id(bfloat16, TILE_32x32)      # -> ID 0 (new)

# Gate weight (bfloat8_b, 32x32, direct-address)
cb_gate_wt = ctx0.get_cb_id(bfloat8_b, TILE_32x32)  # -> ID 1 (new)

# Up weight (bfloat8_b, 32x32, direct-address)
cb_up_wt = ctx0.get_cb_id(bfloat8_b, TILE_32x32)    # -> ID 2 (ID 1 excluded)

# Gate output (bfloat16, 32x32, scratch)
cb_gate_out = ctx0.get_cb_id(bfloat16, TILE_32x32)  # -> ID 3 (ID 0 excluded)

# Up output (bfloat16, 32x32, scratch)
cb_up_out = ctx0.get_cb_id(bfloat16, TILE_32x32)    # -> ID 4 (IDs 0,3 excluded)

# SiLU*up output (bfloat16, 32x32, scratch)
cb_silu_up = ctx0.get_cb_id(bfloat16, TILE_32x32)   # -> ID 5 (IDs 0,3,4 excluded)

# Phase 0 total: 6 CB IDs (0-5)
```

**Manager state after Phase 0:**

```text
ID 0: (bfloat16, 32x32)
ID 1: (bfloat8_b, 32x32)
ID 2: (bfloat8_b, 32x32)
ID 3: (bfloat16, 32x32)
ID 4: (bfloat16, 32x32)
ID 5: (bfloat16, 32x32)
```

**Phase 1 (Down projection) after reconfig():**

```python
ctx1 = manager.create_context()

# SiLU*up result (bfloat16, 32x32) -- input to down matmul
cb_in = ctx1.get_cb_id(bfloat16, TILE_32x32)
# -> ID 0 (match: bfloat16/32x32, not in ctx1 exclude set)

# Down weight (bfloat4_b, 32x32, direct-address)
cb_down_wt = ctx1.get_cb_id(bfloat4_b, TILE_32x32)
# -> ID 6 (NEW! No bfloat4_b ID exists. Cannot reuse ID 1 or 2
#    because bfloat4_b != bfloat8_b)

# Down output (bfloat16, 32x32, scratch)
cb_down_out = ctx1.get_cb_id(bfloat16, TILE_32x32)
# -> ID 3 (match: bfloat16/32x32, ID 0 in exclude)

# Phase 1 total: 3 CB IDs (0, 3, 6)
```

**Final tally:**

```text
Total unique CB IDs: 7 (out of 64 maximum)
Reused across phases: 2 (IDs 0 and 3)
New in Phase 1:       1 (ID 6 for bfloat4_b weights)
Format-blocked reuse: 1 (bfloat4_b could not reuse bfloat8_b IDs 1,2)
```

### Why Format-Blocked Reuse Is Correct

If the manager had allowed bfloat4_b to reuse ID 1 (which was bfloat8_b in Phase 0), the reconfig tensor would contain conflicting metadata:

```text
WRONG (if reuse were allowed):
  Phase 0: CB 1 = {bfloat8_b, page_size=1088, ...}
  Phase 1: CB 1 = {bfloat4_b, page_size=576, ...}

  Reconfig tensor for CB 1 at phase boundary:
    Must write EITHER 1088 OR 576 as page_size
    Cannot have both -- reconfig tensor has one entry per CB ID

  If 576 is written: Phase 0 reads bfloat8_b with wrong page boundaries -> corruption
  If 1088 is written: Phase 1 reads bfloat4_b with wrong page boundaries -> corruption
```

By allocating a fresh ID (6) for bfloat4_b, each CB ID has a single, consistent format across all phases. The reconfig tensor can safely program each ID's metadata.

### L1 Impact of Format-Aware Allocation

The mixed-format approach provides concrete L1 savings for the three-format MLP:

| Weight CB | Uniform bfloat16 | Mixed bfloat8_b/4_b | Savings |
|-----------|------------------|---------------------|---------|
| Gate weight (96 tiles) | 96 x 2,048 = 196,608 B | 96 x 1,088 = 104,448 B | 92,160 B |
| Up weight (96 tiles) | 96 x 2,048 = 196,608 B | 96 x 1,088 = 104,448 B | 92,160 B |
| Down weight (96 tiles) | 96 x 2,048 = 196,608 B | 96 x 576 = 55,296 B | 141,312 B |
| **Total weights** | **589,824 B (576 KB)** | **264,192 B (258 KB)** | **325,632 B (55%)** |

This 318 KB saving is achieved entirely through free hardware-level format transitions. The `CircularBufferIdManager` uses 7 CB IDs instead of 6 (one extra for the bfloat4_b format key), a small price for 55% weight L1 reduction.

---

## Worked Example: Matmul -> RMSNorm -> Matmul Format Chain

This example traces format transitions through a common sub-pattern: a matmul producing bfloat16, feeding into RMSNorm (which needs bfloat16 for the reciprocal square root), then feeding into another matmul with bfloat8_b weights.

### Configuration

```text
Precision profile: performance
  Weight format: bfloat8_b
  Activation format: bfloat16
  Math fidelity: LoFi
  Accumulation: bfloat16 dest (fp32_dest_acc_en=False)

Shape: activation [1, 4096], first matmul weights [4096, 4096],
       RMSNorm gamma [4096], second matmul weights [4096, 11008]
```

### Step 1: First Matmul

```text
Operation: DRAMStreamingMatmul
  in0 CB: bfloat16, page_size=2048, activation
  in1 CB: bfloat8_b, page_size=1088, weight (direct-address)

  Hardware path:
    NCRISC streams bfloat8_b weight tiles from DRAM to weight CB
    TRISC0 unpacks bfloat8_b tiles -> bfloat16 in SRCB (FREE conversion)
    TRISC0 unpacks bfloat16 activation tiles -> bfloat16 in SRCA
    TRISC1 multiplies SRCA x SRCB, accumulates in dest (bfloat16)
    TRISC2 packs dest -> bfloat16 output CB

  Output CB: bfloat16, page_size=2048
  Format transitions: 1 (bfloat8_b -> bfloat16, free at unpack)

  FormatPolicy negotiation:
    Producer (matmul.out): preferred={bfloat16}
    Consumer (rmsnorm.input): preferred={bfloat16}, required={bfloat16}
    Result: bfloat16 (shared preferred, meets required)
```

### Step 2: RMSNorm

```text
Operation: PaddedRMSNorm
  input CB: bfloat16, page_size=2048 (from matmul output)
  gamma CB: bfloat16, page_size=2048 (direct-address, row-major)

  Internal computation:
    1. Square each element: x^2 (bfloat16)
    2. Sum of squares across width dimension
    3. Divide by element count (using logical count, not padded!)
    4. Reciprocal square root: 1/sqrt(mean(x^2))
       ** This requires bfloat16 precision for the rsqrt **
       ** bfloat8_b would introduce up to 0.8% relative error **
    5. Multiply: x * rsqrt * gamma

  Output CB: bfloat16, page_size=2048
  Format transitions: 0 (all bfloat16 throughout)

  FormatPolicy negotiation:
    Producer (rmsnorm.out): preferred={bfloat16}
    Consumer (matmul.in0): preferred={bfloat16}
    Result: bfloat16 (shared preferred)
```

### Step 3: Second Matmul (with bfloat8_b Weights)

```text
Operation: DRAMStreamingMatmul
  in0 CB: bfloat16, page_size=2048 (from RMSNorm output)
  in1 CB: bfloat8_b, page_size=1088 (weight, direct-address)

  Hardware path: identical to Step 1
    bfloat8_b weight unpacked to bfloat16 by TRISC0 (FREE)

  Output CB: bfloat16, page_size=2048
  Format transitions: 1 (bfloat8_b -> bfloat16, free at unpack)
```

### Summary of Format Decisions

```text
Total CBs:              7 (act_in, wt1, mm1_out, gamma, rms_out, wt2, mm2_out)
Explicit conversions:   0
Free conversions:       2 (bfloat8_b -> bfloat16 at each matmul unpack)
cb_alias() used:        0 (no FIFO-to-FIFO format change needed)
CB reconfig:            0 (single phase, no reconfig boundaries)
Page size values:       2 (2048 for bfloat16, 1088 for bfloat8_b)
L1 savings vs all-bf16: (1088 - 2048) * weight_tiles per weight CB
                        = 960 bytes saved per weight tile
```

### When cb_alias() Is Needed: Hypothetical Variant

If the matmul output were stored as bfloat8_b (to save L1) and the downstream RMSNorm requires bfloat16:

```text
Hypothetical: matmul output stored as bfloat8_b
  matmul output CB: 128 tiles * 1,088 B = 139,264 B (136 KB)
  vs bfloat16: 128 tiles * 2,048 B = 262,144 B (256 KB)
  Savings: 120 KB

Transition options:
  Option A: cb_alias (matmul output bf8_b -> alias as bf16)
    The hardware unpacker reads bfloat8_b bytes and expands to bf16 in SRCA
    Cost: 0 bytes L1, 0 cycles (hardware pipeline)
    Constraint: matmul output CB must be FIFO mode (not direct-address)
    Result: RMSNorm sees bf16 tiles via the alias CB ID

  Option B: Explicit conversion op
    Source: 128 tiles * 1,088 B = 136 KB
    Target: 128 tiles * 2,048 B = 256 KB (additional scratch CB)
    Cost: 256 KB L1 (negates the savings from bf8_b output)
    Not recommended: worse than using bf16 output in the first place

  Chosen: Option A (cb_alias) -- zero cost
```

---

## Format Transitions at reconfig() Boundaries

### What Changes at a reconfig() Boundary

The `reconfig()` call reprograms CB descriptors via the reconfig tensor (264 uint32 words per core, as described in [Chapter 6, File 03](../ch06_data_formats/03_format_conversion_and_cb_reconfig.md)). From the format perspective:

```text
Before reconfig():
  CB 4: {data_format=bfloat8_b, page_size=1088, num_pages=3200}
  CB 5: {data_format=bfloat8_b, page_size=1088, num_pages=3200}

Reconfig tensor writes:
  CB 4: [addr=0x1000, size=3481600, num_pages=3200, page_size=1088]
  CB 5: [addr=0x352400, size=3481600, num_pages=3200, page_size=1088]
  ... (all 64 slots programmed, unused slots zeroed)

After reconfig():
  CB 6: {data_format=bfloat4_b, page_size=576, num_pages=3096}
  (Phase 1 down weight, new CB ID because format differs)
```

### The CbReconfig Kernel

The `CbReconfig` MicroOp reads the reconfig tensor and reprograms all CB descriptors. It is automatically inserted at each `reconfig()` boundary by `cb_reconfig_builder.py`:

```python
# EXISTING -- CbReconfig emission (from cb_reconfig_builder.py)
class CbReconfig(MicroOp):
    """Reprograms CB descriptors from a reconfig tensor.

    Reads 264 uint32 values per core from a HEIGHT_SHARDED tensor.
    For each of the 64 CB slots, writes:
      - L1 address
      - Total size
      - Number of pages
      - Page size
    """
    @staticmethod
    def emit(f, reconfig_tensor_handle, *, prefix="cb_reconfig"):
        f.ncrisc_ct_args([
            (f"{prefix}.cb_config_l1_addr", 0),  # placeholder, patched later
        ])
        # The kernel iterates over all 64 CB slots and reprograms them
```

### Format Consistency Validation at reconfig()

TensorAdapter validates format consistency at each reconfig boundary:

```python
# PROPOSED -- format consistency check at reconfig()
def validate_reconfig_format_consistency(
    phase0_cbs: dict[int, CBMetadata],
    phase1_cbs: dict[int, CBMetadata],
) -> list[str]:
    """Validate that shared CB IDs have consistent formats across phases.

    A CB ID that appears in both phases must have the same (data_format, tile).
    If not, the reconfig tensor cannot program a consistent page_size.
    """
    errors = []
    shared_ids = set(phase0_cbs.keys()) & set(phase1_cbs.keys())
    for cb_id in shared_ids:
        meta0 = phase0_cbs[cb_id]
        meta1 = phase1_cbs[cb_id]
        if (meta0.data_format, meta0.tile) != (meta1.data_format, meta1.tile):
            errors.append(
                f"CB ID {cb_id} has inconsistent format across phases: "
                f"Phase 0 = ({meta0.data_format}, {meta0.tile}), "
                f"Phase 1 = ({meta1.data_format}, {meta1.tile}). "
                f"This will corrupt page boundary computation."
            )
    return errors
```

This validation catches the exact bug described in the "Why Format-Blocked Reuse Is Correct" section: if the `CircularBufferIdManager` were bypassed and a developer manually assigned conflicting formats to the same CB ID across phases, this check would flag it at compile time rather than causing silent data corruption at runtime.

### CB Reconfig vs. Separate Conversion Ops

Within a fused program, CB reconfig is always preferred over separate conversion ops for format transitions:

| Criterion | CB Reconfig | Separate Conversion Op |
|-----------|------------|----------------------|
| L1 cost | 1,056 B (reconfig tensor, shared) | num_tiles x page_size (per conversion) |
| CB slot cost | 0 (reuses existing ID) | +1 CB slot per conversion |
| Compute cost | ~5 us (CbReconfig kernel) | ~1 cycle/tile (SFPU pipeline) |
| Phase boundary required | Yes | No |
| Data movement | None (descriptor-only change) | Full tile copy through SFPU |

For a 128-tile conversion:
- CB reconfig: 1,056 B + 5 us
- Separate conversion: $128 \times 2{,}048 = 262{,}144$ B + 128 us

The reconfig approach is 248x cheaper in L1 and 25x faster.

---

## fp32_dest_acc_en: Precision Boundaries Within Fused Ops

### What fp32_dest_acc_en Does

The `fp32_dest_acc_en` flag widens the Tensix destination register from bfloat16 to float32 precision. When enabled, the FPU accumulates results in float32, which is then truncated to bfloat16 when packed to the output CB. This is a format transition that happens entirely within the SFPU pipeline -- no additional CBs, no data movement.

```text
Without fp32_dest_acc_en:
  SRCA (bf16) x SRCB (bf16) -> DEST (bf16, 16-bit accumulation)
  Risk: accumulation overflow for large K dimensions

With fp32_dest_acc_en:
  SRCA (bf16) x SRCB (bf16) -> DEST (fp32, 32-bit accumulation)
  -> Pack: DEST (fp32) -> output CB (bf16, truncated)
  Benefit: no overflow, higher precision for large K
  Cost: 0 additional L1, 0 additional CB slots
```

### compute_subblock_w() and Format Interaction

The `compute_subblock_w()` function from `blaze/ops/dram_streaming_matmul/op.py` determines the subblock width based on `fp32_dest_acc_en` and `dst_full_sync_en`:

```python
# EXISTING -- from blaze/ops/dram_streaming_matmul/op.py
def compute_subblock_w(per_core_N, fp32_dest_acc_en=False, dst_full_sync_en=False):
    """Compute subblock width for DRAMStreamingMatmul.

    The subblock width determines how many output tiles are accumulated
    in the dest register before flushing to the output CB.

    (fp32_dest_acc_en, dst_full_sync_en) -> max_subblock_w:
      (False, False) -> 8
      (False, True)  -> 16
      (True,  True)  -> 8
      (True,  False) -> 4
    """
```

When `fp32_dest_acc_en=True` and `dst_full_sync_en=False`, the maximum subblock width drops from 8 to 4 because each float32 accumulator occupies twice the register space. This affects the blocking strategy (fewer tiles per subblock = more iterations) but does not affect L1 (the dest register is not in L1).

---

## The Three Precision Profiles and Fused Op Formats

### How Profiles Map to Fused Op Format Decisions

The three precision profiles from [Chapter 6, File 04](../ch06_data_formats/04_precision_profiles_and_user_overrides.md) control every format decision within a fused op. Here is the complete mapping for the 11-node gated MLP:

**Performance Profile (bfloat8_b weights, LoFi math fidelity):**

```text
Node 1  mcast(act):        bfloat16 activation           [2,048 B/tile]
Node 2  gate_mm.in1:       bfloat8_b gate weight          [1,088 B/tile]
Node 2  gate_mm.out:       bfloat16 output                [2,048 B/tile]
Node 3  up_mm.in1:         bfloat8_b up weight            [1,088 B/tile]
Node 3  up_mm.out:         bfloat16 output                [2,048 B/tile]
Node 6  gated_reduce:      bfloat16 in/out                [2,048 B/tile]
Node 8  down_mm.in1:       bfloat8_b down weight          [1,088 B/tile]
Node 8  down_mm.out:       bfloat16 output                [2,048 B/tile]
Node 10 residual_add:      bfloat16 in/out                [2,048 B/tile]
Math fidelity:             LoFi (all matmuls)
Accumulation:              bfloat16 dest (fp32_dest_acc_en=False)
Free conversions:          3 (bfloat8_b -> bfloat16 at each matmul unpack)
```

**Balanced Profile (bfloat16 weights, HiFi4 math fidelity):**

```text
Node 2  gate_mm.in1:       bfloat16 gate weight           [2,048 B/tile]
Node 3  up_mm.in1:         bfloat16 up weight             [2,048 B/tile]
Node 8  down_mm.in1:       bfloat16 down weight           [2,048 B/tile]
All other nodes:           bfloat16                        [2,048 B/tile]
Math fidelity:             HiFi4 (all matmuls)
Accumulation:              bfloat16 dest (fp32_dest_acc_en=False, dst_full_sync_en=False)
Free conversions:          0 (all same format)
```

**Accuracy Profile (bfloat16 weights, HiFi4 math fidelity, float32 accumulation):**

```text
All nodes:                 bfloat16                        [2,048 B/tile]
Math fidelity:             HiFi4 (all matmuls)
Accumulation:              float32 dest (fp32_dest_acc_en=True)
  ** WARNING: float32 dest halves tile capacity **
  ** dest register: 16 tiles at bfloat16, 8 tiles at float32 **
  ** Must set dst_full_sync_en=True for correctness **
Free conversions:          0
Dest capacity impact:      compute_subblock_w must account for
                           halved dest register capacity
```

### Summary of Profile-to-Format Mapping

| Profile | Weight Format | Activation Format | Math Fidelity | fp32_dest_acc_en | dst_full_sync_en |
|---------|-------------|-------------------|---------------|-----------------|-----------------|
| Performance | bfloat8_b | bfloat16 | LoFi | False | False |
| Balanced | bfloat16 | bfloat16 | HiFi4 | False | False |
| Accuracy | bfloat16 | bfloat16 | HiFi4 | True | True |

Within a fused op, different sub-ops can have different `fp32_dest_acc_en` settings. The reconfig boundary between phases allows switching:

```text
Phase 0: RMSNorm
  fp32_dest_acc_en = True (norm computation needs precision)
  Math fidelity: HiFi4

  -- reconfig() --

Phase 1: DRAMStreamingMatmul (gate projection)
  fp32_dest_acc_en = False (performance mode for large matmul)
  Math fidelity: LoFi

  -- reconfig() --

Phase 2: DRAMStreamingMatmul (down projection)
  fp32_dest_acc_en = True (large K=11008, accumulation overflow risk)
  Math fidelity: HiFi4
```

The reconfig tensor programs the SFPU configuration registers alongside the CB descriptors. No additional L1 is needed for this precision transition.

### L1 Budget Impact Across Profiles

For the gated MLP with LLaMA-7B dimensions (H=4096, I=11008, 14 cores):

| Resource | Performance | Balanced | Accuracy |
|----------|------------|---------|----------|
| Gate weight shard (per core) | 3,200 × 1,088 = 3,400 KB | 3,200 × 2,048 = 6,400 KB | 6,400 KB |
| Up weight shard (per core) | 3,400 KB | 6,400 KB | 6,400 KB |
| Down weight shard (per core) | 3,096 × 1,088 = 3,290 KB | 3,096 × 2,048 = 6,192 KB | 6,192 KB |
| Total weight L1 | ~10,090 KB | ~18,992 KB | ~18,992 KB |
| Fits in ~1.5 MB L1? | No (sharded) | No (sharded) | No (sharded) |
| Scratch CB overhead | 154 KB | 154 KB | 154 KB |

> **Note:** Weight shards exceed L1 capacity (~1.5 MB total, ~1.3 MB usable for CBs) in all profiles. The weights are streamed from DRAM via direct-address CBs, which occupy their L1 shard allocation separately from the general L1. The bfloat8_b format in the performance profile nearly halves the DRAM bandwidth required for weight streaming (1,088 vs 2,048 bytes per tile).

---

## Worked Example: DeepSeek V3 MoE Expert Format Flow

This example traces format transitions through a complete gated MLP with different weight formats per phase (DeepSeek V3 configuration):

```text
Model: DeepSeek V3 MoE Expert
  hidden_dim = 7,168
  intermediate_dim = 2,048 per core
  Gate weights: bfloat8_b (default for shared experts)
  Up weights: bfloat8_b
  Down weights: bfloat4_b (more aggressive compression for large experts)
  Activations: bfloat16 throughout
```

### Phase 0: RMSNorm + Mcast

```text
CB 0: activation (bf16, 224 tiles, 2,048 B/tile)     = 458,752 B
CB 1: gamma (bf16, 7 tiles, 2,048 B/tile)            =  14,336 B
CB 2: rmsnorm output (bf16, 7 tiles, 2,048 B/tile)   =  14,336 B
CB 3: mcast dest (bf16, 7 tiles, 2,048 B/tile)       =  14,336 B

All CBs are bfloat16. No format transitions in Phase 0.
Total Phase 0 L1: 501,760 B (490 KB)
```

### Phase 1: Gate + Up Matmuls + EltwiseMul

```text
-- reconfig() boundary --

CB 3: mcast act (bf16, 7 tiles, reused from Phase 0)
  Format: bf16 in both phases -> reconfig reuses same ID at same format

CB 4: gate weights streaming (bf8_b, 96 tiles, 1,088 B/tile) = 104,448 B
CB 5: up weights streaming (bf8_b, 96 tiles, 1,088 B/tile)   = 104,448 B

Format transition at matmul boundary:
  gate_wt CB 4 (bf8_b) -> matmul compute (bf16)
  Mechanism: hardware unpacker (FREE)
  No cb_alias needed: the matmul kernel reads bf8_b and unpacks internally

CB 6: gate output (bf16, 64 tiles, 2,048 B/tile)     = 131,072 B
CB 7: up output (bf16, 64 tiles, 2,048 B/tile)       = 131,072 B
CB 8: silu_mul output (bf16, 64 tiles, 2,048 B/tile)  = 131,072 B

Format within Phase 1:
  Weights: bf8_b (stored in L1, unpacked to bf16 by hardware)
  Activations: bf16
  Outputs: bf16
  Transitions: 2 free (unpacker), 0 explicit

Total Phase 1 L1: 14,336 + 104,448 + 104,448 + 131,072 x 3 = 616,448 B (602 KB)
```

### Phase 2: Gather + Mcast + Down Matmul

```text
-- reconfig() boundary --

CB 8: silu_mul result (bf16, reused across boundary)
  Format match: bf16 -> bf16, reconfig reuses ID

CB 9: down weights streaming (bf4_b, 96 tiles, 576 B/tile)   = 55,296 B
  Note: bfloat4_b saves 49,152 B vs bf8_b (104,448 - 55,296)
  Format transition at matmul boundary:
    down_wt CB 9 (bf4_b) -> matmul compute (bf16)
    Mechanism: hardware unpacker (FREE)
    The unpacker handles bf4_b -> bf16 expansion identically to bf8_b -> bf16

CB 6: down output (bf16, 64 tiles, reused ID from Phase 1) = 131,072 B
  Format: bf16 in both phases -> reused

Total Phase 2 L1: 131,072 + 55,296 + 131,072 = 317,440 B (310 KB)
```

### Format Transition Summary

```text
Format transitions in the complete fused MLP:
  Phase boundary 0->1: CB 3 bf16 -> bf16 (reconfig, same format, free)
  Phase 1 internal: gate_wt bf8_b -> bf16 (unpacker, free)
  Phase 1 internal: up_wt bf8_b -> bf16 (unpacker, free)
  Phase boundary 1->2: CB 8 bf16 -> bf16 (reconfig, same format, free)
  Phase 2 internal: down_wt bf4_b -> bf16 (unpacker, free)

Total explicit conversion ops: 0
Total cb_alias insertions: 0
Total reconfig cost: 1,056 B (shared reconfig tensor)
Total format transitions: 5 (all free)

CB ID allocation via CircularBufferIdManager:
  Phase 0: IDs 0-3 (bf16 x4)
  Phase 1: ID 3 reused (bf16), IDs 4-5 new (bf8_b), IDs 6-8 new (bf16)
  Phase 2: ID 8 reused (bf16), ID 9 new (bf4_b), ID 6 reused (bf16)
  Total unique IDs: 10

  Without format-aware reuse: 15 IDs
  Savings: 5 IDs (33% reduction)
```

### L1 Savings Calculation

Using bfloat4_b for down weights saves significant L1 per expert. For DeepSeek V3 with 8 active experts per token:

$$
\text{Per-expert savings} = 96 \times (1{,}088 - 576) = 96 \times 512 = 49{,}152 \text{ bytes} \approx 48 \text{ KB}
$$

$$
\text{Per-token savings (8 experts)} = 8 \times 49{,}152 = 393{,}216 \text{ bytes} \approx 384 \text{ KB}
$$

This is critical for fitting MoE workloads within the ~1.5 MB per-core L1 budget.

---

## Manual vs. Abstracted Format Management

### The Manual Path (Today)

To configure formats within a fused gated MLP manually, the developer must:

```python
# EXISTING -- Manual format configuration for SwigluOp
f = FusedProgram(
    kernel=swiglu_kernel, device=device,
    math_fidelity=ttnn.MathFidelity.LoFi,     # Must choose
    math_approx_mode=True,                      # Must choose
    fp32_dest_acc_en=False,                     # Must choose
)

# Activation CB: developer must specify bfloat16
act = f.cb_from_tensor(act_tensor, data_format=ttnn.bfloat16)

# Gate weight CB: developer must specify bfloat8_b
# AND compute the correct page_size for bfloat8_b tiles
gate_wt = f.cb_from_tensor(
    gate_tensor, data_format=ttnn.bfloat8_b,
    is_direct_address=True,
    page_size=1088,  # Must know: 32*32 * (1 + 1/32) for BFP8
)

# Up weight CB: same format decisions
up_wt = f.cb_from_tensor(
    up_tensor, data_format=ttnn.bfloat8_b,
    is_direct_address=True, page_size=1088,
)

# Scratch CBs: developer must match activation format
gate_out = f.cb_scratch(
    name="gate_mm_out", data_format=ttnn.bfloat16,
    page_size=2048, num_pages=25, core_ranges=cores,
)

# ... 5 more CB format decisions ...

# After reconfig: down weight in potentially different format
f.reconfig()
down_wt = f.cb_from_tensor(
    down_tensor, data_format=ttnn.bfloat8_b,  # Or bfloat4_b
    is_direct_address=True, page_size=1088,    # Or 576 for bfloat4_b
)
```

**Manual decisions required:**

| Decision | Count | Error if Wrong |
|----------|-------|---------------|
| Data format per CB | 8+ | Unpack/pack format mismatch -> garbage output |
| Page size per CB | 8+ | Wrong page boundaries -> data corruption |
| Math fidelity | 1 | Incorrect precision for sensitive ops |
| fp32_dest_acc_en | 1 | Accumulation overflow or unnecessary perf cost |
| math_approx_mode | 1 | SiLU approximation error |
| Format consistency across reconfig | 1 | Silent corruption (as shown above) |
| **Total** | **~20** | |

### The Abstracted Path (Proposed)

```python
# PROPOSED -- Abstracted format management
adapter = TensorAdapter(device=device, precision="performance")

blaze_mlp = BlazeModule.from_torch(
    GatedMLP(hidden=4096, intermediate=11008),
    sample_input=torch.randn(1, 4096),
    device=device,
)

# All 20 format decisions are automatic:
#   - Precision profile sets: bfloat8_b weights, bfloat16 activations, LoFi
#   - Format negotiation resolves each edge (12 edges -> 0 explicit conversions)
#   - Page sizes computed from format: bfloat16->2048, bfloat8_b->1088
#   - Math fidelity from profile (LoFi) with softmax safety override (HiFi4)
#   - fp32_dest_acc_en from profile (False for performance)
#   - CB reconfig format consistency validated at compile time
```

### Comparison Table

| Aspect | Manual | Abstracted |
|--------|--------|-----------|
| Format decisions per fused op | ~20 | 1 (profile name) |
| Page size computations | 8+ manual calculations | 0 (auto from format) |
| Format consistency validation | None (runtime crash) | Compile-time check |
| Softmax safety override | Developer must remember | Always enforced |
| Weight format selection | Per-CB decision | Profile default + override |
| Activation format selection | Per-CB decision | Always bfloat16 |
| Math fidelity | Manual per FusedProgram | Profile default + per-op override |
| fp32_dest_acc_en coupling | Developer must set dst_full_sync_en | Auto-coupled |
| Lines of code (format config) | ~30 | ~1 |

---

## Error Taxonomy: Format Errors Within Fused Ops

| Error | Root Cause | Symptom | Detection Point |
|-------|-----------|---------|----------------|
| **Format mismatch at unpack** | CB declared as bfloat16 but contains bfloat8_b data | Garbage tiles (bytes misinterpreted) | Runtime: NaN/Inf in output. Abstracted: format negotiation prevents |
| **Page size mismatch** | page_size=2048 for bfloat8_b CB (should be 1088) | NCRISC reads past tile boundary | Runtime: data corruption. Abstracted: page_size auto-computed from format |
| **Cross-phase format conflict** | Same CB ID with different formats in Phase 0 and Phase 1 | Reconfig tensor programs wrong page_size | Runtime: silent corruption. Abstracted: CircularBufferIdManager prevents |
| **bfloat4_b at softmax** | Performance profile applies bfloat4_b to all ops | Cannot represent -inf; exp(-inf) != 0 | Abstracted: incompatible set blocks; Manual: developer must know |
| **Missing fp32 dest for accuracy** | fp32_dest_acc_en=False but computation needs float32 | Accumulation overflow in large reductions | Abstracted: accuracy profile auto-sets; Manual: developer must set |
| **dst_full_sync_en not set** | fp32_dest_acc_en=True but dst_full_sync_en=False | Race condition: compute reads stale dest values | Abstracted: auto-coupled; Manual: developer must know coupling |
| **cb_alias on direct-address CB** | Developer tries cb_alias() on weight CB | RuntimeError from FusedProgram | Both paths: explicit check in cb_alias() |

---

## Key Takeaways

- `cb_alias()` provides zero-cost format reinterpretation for FIFO CBs within fused ops. It creates a second CB ID pointing at the same L1 memory with a different format descriptor. The hardware unpacker handles the actual conversion. Direct-address CBs cannot use `cb_alias()` because their page addresses depend on page_size.
- The `FormatPolicy` negotiation from Chapter 6 applies to each of the 12 internal edges within a gated MLP `FusedOp.compose()`. Each edge resolves independently, with the three-level preference system (preferred, tolerable, required) plus the incompatible set ensuring correctness. The softmax safety override forces bfloat16 + HiFi4 + float32 accumulation regardless of the global precision profile.
- `CircularBufferIdManager` keys CB ID reuse on `(data_format, tile)` pairs. A bfloat4_b weight CB cannot reuse a bfloat8_b CB ID because the page_size differs and the reconfig tensor can only hold one page_size per CB ID. This prevents the subtle cross-phase corruption where the wrong format is programmed into CB registers.
- The three precision profiles map cleanly to format decisions: Performance uses bfloat8_b weights with LoFi math fidelity; Balanced uses bfloat16 weights with HiFi4; Accuracy uses bfloat16 weights with HiFi4 and `fp32_dest_acc_en=True` plus `dst_full_sync_en=True`. All three profiles use bfloat16 activations.
- For production LLMs under the performance profile, all weight-to-matmul format conversions (bfloat8_b -> bfloat16) are free at the hardware unpack stage. No explicit conversion ops, `cb_alias()` calls, or extra CBs are needed. The bfloat8_b format saves approximately 47% of DRAM bandwidth per weight tile ($1{,}088 / 2{,}048 \approx 0.53$).
- Mixed weight formats (bfloat8_b for gate/up, bfloat4_b for down) save 325,632 bytes (55%) in the DeepSeek V3 MoE expert, achieved entirely through free hardware transitions. For 8 active experts per token, this totals ~384 KB of savings -- critical for fitting within the ~1.5 MB per-core L1 budget.
- The abstracted path reduces ~20 manual format decisions (data format per CB, page size per CB, math fidelity, fp32_dest_acc_en, dst_full_sync_en) to a single precision profile name. All decisions are validated at compile time, and the softmax safety override is always enforced.

## Source Files

- `blaze/fused_program.py` -- `cb_alias()` (zero-cost format alias), `cb_from_tensor()` (format-specified CB creation), `reconfig()` (phase boundary)
- `blaze/cb_reconfig.py` -- `CircularBufferIdManager` (format-keyed CB ID reuse), `CBContext` (per-phase context), `_allocate_id()` (reuse logic keyed on `(data_format, tile)`)
- `blaze/cb_reconfig_builder.py` -- `prepare_for_build()` (multi-phase pipeline), `_build_multi_phase()` (reconfig tensor construction with 264 uint32 words per core)
- `blaze/blaze_op.py` -- `BlazeOp.math_fidelity`, `BlazeOp.math_approx_mode`: per-op fidelity fields
- `blaze/fused_program.py` -- `FusedProgram.math_fidelity`, `FusedProgram.fp32_dest_acc_en`: per-program fidelity fields
- `blaze/ops/swiglu/op.py` -- `SwigluOp.compose()`: the canonical multi-format fused op with reconfig boundary
- `blaze/utils.py` -- `interpret_tile()`, `interpret_tile_padded()`: tile geometry selection affecting format layout

---

**Previous:** [02_shape_and_padding_across_fused_boundaries.md](./02_shape_and_padding_across_fused_boundaries.md) | **Next:** [04_fusion_detection_and_limitations.md](./04_fusion_detection_and_limitations.md)
