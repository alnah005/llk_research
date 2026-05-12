# Format Negotiation Protocol: Automatic Cross-Op Format Selection

When a matmul produces `bfloat8_b` output but the downstream softmax needs `bfloat16` input with float32 accumulation, someone must decide where and how the format transition happens. Today, that someone is the developer -- manually configuring each CB's format and inserting conversion ops where needed. This section designs an automatic format negotiation protocol: each op declares its preferred and tolerable formats, a two-pass graph-walking algorithm propagates preferences and identifies mismatches, a cost model evaluates the expense of each conversion, and fallback policies ensure the system always makes a valid choice. The protocol draws on DeepSeek V3 MoE (where 256 experts must agree on format) and LLaMA attention (where three distinct format zones must coexist within a single fused op).

> *Design Principle P5 (Operate on Graphs, Not Individual Tensors):* Format negotiation is fundamentally a graph-level problem. Per-op format selection inserts conversions at every boundary. Graph-level analysis minimizes total conversions by finding globally optimal format assignments.

**What you will learn:**

- How to declare per-op format preferences (preferred set, tolerable set, incompatible set) at each input and output port
- The format negotiation algorithm: a two-pass graph walk (forward propagation + backward resolution) that propagates format preferences and finds minimum-cost assignments in O(V+E) time
- The cost model for format conversions: free conversions (SFPU-level unpack), cheap conversions (in-CB reconfig), and expensive conversions (explicit conversion ops)
- Fallback policy: when no preference is declared or no overlap exists, default to `bfloat16` (`DEFAULT_DTYPE`)
- Tile reinterpretation via `cb_from_view` and `cb_alias` in `FusedProgram`: zero-cost format aliasing when tile geometries are compatible
- Format unification pass: reducing format diversity to minimize CB ID pressure against the 64-slot limit
- Walkthrough: format negotiation through a LLaMA attention block and a DeepSeek V3 MoE layer

---

## Op Format Declarations

### The Three Preference Levels Plus Incompatible Set

Each op declares format preferences at three levels per port, plus an explicit incompatible set for safety:

```python
# PROPOSED -- Format preference declaration
@dataclass(frozen=True)
class FormatPreference:
    """Format preference for a single port of an op.

    The incompatible set (from V4's defensive design) categorically
    blocks dangerous combinations like bfloat4_b as softmax input.
    """
    preferred: frozenset[ttnn.DataType]
    """Best-performing formats for this port. The negotiator picks from
    this set when possible."""

    tolerable: frozenset[ttnn.DataType]
    """Formats that produce correct results but may not be optimal.
    The negotiator picks from this set only when preferred formats
    would require an expensive conversion."""

    required: frozenset[ttnn.DataType] | None = None
    """If set, the port MUST use one of these formats. A format outside
    this set produces incorrect results (e.g., softmax needs bfloat16
    input). None means any format is acceptable."""

    incompatible: frozenset[ttnn.DataType] = frozenset()
    """Formats that produce incorrect results or hardware errors at this
    port. A format in the incompatible set is never selected, even if it
    appears in the producer's preferred set. Example: bfloat4_b for
    softmax input (limited dynamic range cannot represent NEG_INF)."""

    def classify(self, fmt: ttnn.DataType) -> str:
        """Classify a format as preferred, tolerable, or incompatible."""
        if fmt in self.incompatible:
            return "incompatible"
        if self.required is not None and fmt not in self.required:
            return "incompatible"
        if fmt in self.preferred:
            return "preferred"
        if fmt in self.tolerable:
            return "tolerable"
        return "incompatible"

    def best_common(self, other: "FormatPreference") -> ttnn.DataType | None:
        """Find the best format that both preferences accept.

        Precedence: shared preferred > cross (one preferred, other
        tolerable) > shared tolerable > None.
        """
        blocked = self.incompatible | other.incompatible
        self_all = (self.preferred | self.tolerable) - blocked
        other_all = (other.preferred | other.tolerable) - blocked

        # Apply required constraints
        if self.required is not None:
            self_all &= self.required
        if other.required is not None:
            other_all &= other.required

        shared_pref = (self.preferred & other.preferred) - blocked
        if shared_pref:
            return _select_cheapest(shared_pref)

        cross = ((self.preferred & other.tolerable) |
                 (other.preferred & self.tolerable)) - blocked
        if cross:
            return _select_cheapest(cross)

        shared_tol = (self.tolerable & other.tolerable) - blocked
        if shared_tol:
            return _select_cheapest(shared_tol)

        return None
```

### Validation at Registration Time

```python
# PROPOSED -- validate preferences when ops are registered
def _validate_format_preference(pref: FormatPreference, op_name: str, port: str):
    """Validate a format preference at op registration time."""
    issues = []

    overlap = pref.preferred & pref.incompatible
    if overlap:
        issues.append(
            f"Op '{op_name}' port '{port}': formats {overlap} are both "
            f"preferred and incompatible -- contradictory."
        )

    if pref.required is not None and not (pref.preferred | pref.tolerable) & pref.required:
        issues.append(
            f"Op '{op_name}' port '{port}': required set {pref.required} does not "
            f"overlap with preferred or tolerable -- op would reject its own requirements."
        )

    if not pref.preferred and not pref.tolerable:
        issues.append(
            f"Op '{op_name}' port '{port}': no preferred or tolerable formats."
        )

    if issues:
        raise FormatRegistrationError("\n".join(issues))
```

### Format Preference Registry

Each Blaze op is associated with format preferences per input/output port:

```python
# PROPOSED -- Format preference registry
FORMAT_PREFERENCES: dict[str, dict[str, FormatPreference]] = {
    "matmul": {
        "in0": FormatPreference(
            preferred=frozenset({ttnn.bfloat16}),
            tolerable=frozenset({ttnn.float32}),
        ),
        "in1": FormatPreference(
            preferred=frozenset({ttnn.bfloat8_b, ttnn.bfloat4_b}),
            tolerable=frozenset({ttnn.bfloat16}),
        ),
        "out": FormatPreference(
            preferred=frozenset({ttnn.bfloat16}),
            tolerable=frozenset({ttnn.float32}),
        ),
    },
    "sdpa": {
        "q": FormatPreference(
            preferred=frozenset({ttnn.bfloat16}),
            tolerable=frozenset(),
            required=frozenset({ttnn.bfloat16}),
            incompatible=frozenset({ttnn.bfloat4_b, ttnn.bfloat8_b}),
        ),
        "k": FormatPreference(
            preferred=frozenset({ttnn.bfloat16}),
            tolerable=frozenset(),
            required=frozenset({ttnn.bfloat16}),
            incompatible=frozenset({ttnn.bfloat4_b, ttnn.bfloat8_b}),
        ),
        "v": FormatPreference(
            preferred=frozenset({ttnn.bfloat16}),
            tolerable=frozenset(),
            required=frozenset({ttnn.bfloat16}),
            incompatible=frozenset({ttnn.bfloat4_b, ttnn.bfloat8_b}),
        ),
        "out": FormatPreference(
            preferred=frozenset({ttnn.bfloat16}),
            tolerable=frozenset({ttnn.bfloat8_b}),
        ),
    },
    "rmsnorm": {
        "input": FormatPreference(
            preferred=frozenset({ttnn.bfloat16}),
            tolerable=frozenset({ttnn.float32}),
        ),
        "gamma": FormatPreference(
            preferred=frozenset({ttnn.bfloat16}),
            tolerable=frozenset(),
            required=frozenset({ttnn.bfloat16}),
        ),
        "out": FormatPreference(
            preferred=frozenset({ttnn.bfloat16}),
            tolerable=frozenset(),
        ),
    },
    "eltwise_mul": {
        "in0": FormatPreference(
            preferred=frozenset({ttnn.bfloat16}),
            tolerable=frozenset({ttnn.float32}),
        ),
        "in1": FormatPreference(
            preferred=frozenset({ttnn.bfloat16}),
            tolerable=frozenset({ttnn.float32}),
        ),
        "out": FormatPreference(
            preferred=frozenset({ttnn.bfloat16}),
            tolerable=frozenset({ttnn.float32}),
        ),
    },
    "mcast": {
        "in": FormatPreference(
            preferred=frozenset({ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b}),
            tolerable=frozenset({ttnn.float32}),
        ),
        "out": FormatPreference(
            preferred=frozenset({ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b}),
            tolerable=frozenset({ttnn.float32}),
        ),
    },
    "gather": {
        "in": FormatPreference(
            preferred=frozenset({ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b}),
            tolerable=frozenset({ttnn.float32}),
        ),
        "out": FormatPreference(
            preferred=frozenset({ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b}),
            tolerable=frozenset({ttnn.float32}),
        ),
    },
}
```

---

## The Format Negotiation Algorithm

### Overview: Two-Pass Graph Walk

The algorithm has two passes followed by an optional unification step:

1. **Forward pass:** Walk the graph in topological order. Propagate format constraints from inputs to outputs, narrowing candidate sets at each edge and assigning formats based on producer preferences and consumer requirements.
2. **Backward pass:** Walk the graph in reverse topological order. Where a consumer's preference could have influenced the producer's output choice to avoid a conversion, adjust the producer's format. Insert conversion nodes where producer and consumer formats are irreconcilable.
3. **Unification pass (optional):** Reduce format diversity across the graph to minimize CB ID pressure against the 64-slot limit.

The two-pass approach (adopted from V4's defensive design) handles cases that a single forward pass misses -- particularly fan-out scenarios where a producer feeds multiple consumers with conflicting preferences.

### Pass 1: Forward Propagation

Walk the graph in topological order. For each op, determine the output format based on the input formats and the op's compute requirements:

```python
# PROPOSED -- Pass 1: forward propagation
def forward_propagate(
    graph: BlazeGraph,
    profile: PrecisionProfile,
) -> dict[EdgeId, ttnn.DataType]:
    """Propagate format preferences forward through the graph."""
    assignments = {}

    for op in graph.topological_order():
        # Collect input format assignments
        input_formats = {}
        for in_edge in graph.input_edges(op):
            if in_edge.id in assignments:
                input_formats[in_edge.consumer_port] = assignments[in_edge.id]

        # Determine output format
        op_pref = get_format_pref(op.name, "out", profile)

        for out_edge in graph.output_edges(op):
            consumer_pref = get_format_pref(
                out_edge.consumer_op, out_edge.consumer_port, profile)

            # Compute candidate set (respecting incompatible sets)
            producer_set = (op_pref.preferred | op_pref.tolerable) - op_pref.incompatible
            consumer_set = (consumer_pref.preferred | consumer_pref.tolerable) - consumer_pref.incompatible

            if consumer_pref.required is not None:
                consumer_set &= consumer_pref.required
            if op_pref.required is not None:
                producer_set &= op_pref.required

            candidates = producer_set & consumer_set

            if not candidates:
                candidates = {DEFAULT_FORMAT}

            # Select: required > preferred > tolerable > fallback
            if op_pref.required and candidates & op_pref.required:
                chosen = _select_cheapest(candidates & op_pref.required)
            elif candidates & op_pref.preferred:
                chosen = _select_cheapest(candidates & op_pref.preferred)
            else:
                chosen = _select_cheapest(candidates)

            assignments[out_edge.id] = chosen

    return assignments
```

### Pass 2: Backward Resolution and Conversion Insertion

Walk the graph in reverse topological order. Identify edges where the forward-assigned format does not satisfy the consumer, and insert conversion nodes:

```python
# PROPOSED -- Pass 2: backward resolution
def backward_resolve(
    graph: BlazeGraph,
    assignments: dict[EdgeId, ttnn.DataType],
) -> list[ConversionNode]:
    """Walk backward, adjusting assignments and inserting conversions."""
    conversions = []

    for op in reversed(graph.topological_order()):
        for in_edge in graph.input_edges(op):
            assigned = assignments[in_edge.id]
            consumer_pref = get_format_pref(op.name, in_edge.consumer_port)

            # Check if assigned format is acceptable to consumer
            classification = consumer_pref.classify(assigned)

            if classification == "incompatible":
                # Must insert conversion
                if consumer_pref.required:
                    target = _select_cheapest(consumer_pref.required)
                else:
                    target = _select_cheapest(
                        consumer_pref.preferred or consumer_pref.tolerable)

                cost = conversion_cost(assigned, target)
                conversions.append(ConversionNode(
                    source_edge=in_edge,
                    source_format=assigned,
                    target_format=target,
                    cost=cost,
                ))

            elif classification == "tolerable":
                # Check if we can improve by adjusting the producer
                producer_pref = get_format_pref(
                    in_edge.producer_op, in_edge.output_port)
                better = consumer_pref.best_common(producer_pref)
                if better and better != assigned:
                    # Only adjust if it does not break other consumers
                    other_consumers = graph.other_consumers(in_edge)
                    if all(_format_acceptable(better, c) for c in other_consumers):
                        assignments[in_edge.id] = better

    return conversions
```

### Fan-Out Handling

When a single producer feeds multiple consumers with conflicting preferences, the backward pass determines whether to adjust the producer's format or insert per-consumer conversions. The decision minimizes total conversion cost:

```python
# PROPOSED -- fan-out resolution
def resolve_fan_out(
    producer_format: ttnn.DataType,
    consumers: list[tuple[str, FormatPreference]],
) -> dict[str, ttnn.DataType | None]:
    """Resolve format for a producer with multiple consumers.

    Returns per-consumer target format. None means no conversion needed.
    """
    results = {}
    total_convert_cost = 0
    for consumer_name, pref in consumers:
        if pref.classify(producer_format) != "incompatible":
            results[consumer_name] = None
        else:
            target = _select_cheapest(
                pref.required or pref.preferred or pref.tolerable)
            results[consumer_name] = target
            total_convert_cost += conversion_cost(producer_format, target)
    return results
```

### Unification Pass: Reducing Format Diversity

After the two-pass negotiation, the graph may have unnecessary format diversity -- edges using different formats where unification would reduce CB ID pressure. The unification pass identifies edges where converging to a common format reduces the number of distinct `(data_format, tile)` keys without increasing conversion cost:

```python
# PROPOSED -- format unification pass
def unify_formats(
    assignments: dict[EdgeId, ttnn.DataType],
    graph: BlazeGraph,
) -> dict[EdgeId, ttnn.DataType]:
    """Reduce format diversity to minimize CB ID consumption.

    The CircularBufferIdManager (cb_reconfig.py) keys reuse on
    (data_format, tile). Fewer distinct formats = fewer CB IDs = more
    headroom against the 64-slot limit.
    """
    format_counts = Counter(assignments.values())
    dominant_format = format_counts.most_common(1)[0][0]

    unified = dict(assignments)
    for edge_id, fmt in assignments.items():
        if fmt == dominant_format:
            continue
        edge = graph.get_edge(edge_id)
        consumer_pref = get_format_pref(edge.consumer_op, edge.consumer_port)
        producer_pref = get_format_pref(edge.producer_op, edge.output_port)

        # Only unify if the dominant format is acceptable to both sides
        if (consumer_pref.classify(dominant_format) != "incompatible" and
            producer_pref.classify(dominant_format) != "incompatible" and
            conversion_cost(fmt, dominant_format) == 0):
            unified[edge_id] = dominant_format

    return unified
```

---

## The Cost Model for Format Conversions

### Conversion Cost Matrix

| From \ To | float32 | bfloat16 | bfloat8_b | bfloat4_b |
|-----------|---------|----------|-----------|-----------|
| **float32** | 0 (identity) | FREE (packer) | CHEAP (packer) | CHEAP (packer) |
| **bfloat16** | FREE (unpacker) | 0 (identity) | CHEAP (packer) | CHEAP (packer) |
| **bfloat8_b** | FREE (unpacker) | FREE (unpacker) | 0 (identity) | CHEAP (repack) |
| **bfloat4_b** | FREE (unpacker) | FREE (unpacker) | CHEAP (repack) | 0 (identity) |

### Free Conversions (SFPU-Level Unpack)

When the Tensix unpacker reads a `bfloat8_b` or `bfloat4_b` tile from a CB, it automatically expands it to `bfloat16` in the source registers (SRCA/SRCB). This means:

- **bfloat8_b -> bfloat16 compute:** Free. The unpacker does the conversion.
- **bfloat4_b -> bfloat16 compute:** Free. Same mechanism.
- **bfloat16 -> float32 accumulation:** Free when `fp32_dest_acc_en=True`. The FPU accumulates in the dest register at float32 precision; no separate conversion.

This is the most common format transition pattern: read compressed weights in bfloat8_b, compute in bfloat16, accumulate in float32 (optionally). The entire chain is handled by hardware with zero overhead.

### Cheap Conversions (Packer/In-CB Reconfig)

When the output format differs from the compute format, the packer converts during the write:

- **float32 -> bfloat16:** The packer truncates from 32-bit to 16-bit during `cb_push_back`. One cycle per tile.
- **bfloat16 -> bfloat8_b:** The packer computes shared exponents per block and packs mantissas. A few cycles per tile.

### Expensive Conversions (Explicit Conversion Ops)

When a conversion cannot be handled by the unpacker/packer pipeline, an explicit conversion op is needed:

- **Retilize:** Changes tile geometry (e.g., 1x32 storage tiles to 32x32 compute tiles).
- **Cross-format with geometry change:** Two-step: retilize + reformat.

### Cost Constants

```python
# PROPOSED -- conversion cost model
CONVERSION_COSTS = {
    "free": 0,          # Handled by unpacker/packer hardware
    "pack_convert": 1,  # Packer does format conversion during write
    "reconfig": 5,      # CB reconfig between phases
    "retilize": 10,     # Explicit retilize micro-op
    "full_convert": 20, # Explicit format conversion micro-op
}

def conversion_cost(src: ttnn.DataType, dst: ttnn.DataType) -> int:
    """Estimate the cost of converting from src to dst format."""
    if src == dst:
        return 0

    # BFP -> standard: free (unpacker expansion)
    if src in (ttnn.bfloat8_b, ttnn.bfloat4_b) and dst in (ttnn.bfloat16, ttnn.float32):
        return CONVERSION_COSTS["free"]

    # Standard -> BFP: cheap (packer compression)
    if src in (ttnn.bfloat16, ttnn.float32) and dst in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        return CONVERSION_COSTS["pack_convert"]

    # float32 <-> bfloat16: free (unpacker/packer)
    if {src, dst} == {ttnn.float32, ttnn.bfloat16}:
        return CONVERSION_COSTS["free"]

    # BFP <-> BFP: requires unpack + repack via SFPU
    if src in (ttnn.bfloat8_b, ttnn.bfloat4_b) and dst in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        return CONVERSION_COSTS["pack_convert"]

    return CONVERSION_COSTS["full_convert"]
```

---

## Fallback Policy: DEFAULT_DTYPE

When no format preference is declared, or when the negotiation algorithm cannot find an overlap between producer and consumer preferences, the system falls back to `bfloat16`:

```python
# EXISTING -- from blaze/cb_engine.py
DEFAULT_DTYPE = "bfloat16"
```

# PROPOSED

```python
DEFAULT_FORMAT = ttnn.bfloat16

def apply_fallback(edge_candidates: set[ttnn.DataType]) -> ttnn.DataType:
    """When negotiation fails, fall back to DEFAULT_FORMAT.

    Rationale:
    - bfloat16 is supported by all ops (universal compatibility)
    - bfloat16 has good precision/memory trade-off
    - bfloat16 is the format the unpacker uses for BFP expansion
    - All existing Blaze kernels are tested with bfloat16
    """
    if DEFAULT_FORMAT in edge_candidates:
        return DEFAULT_FORMAT
    if ttnn.float32 in edge_candidates:
        return ttnn.float32
    return next(iter(edge_candidates))
```

---

## Tile Reinterpretation via cb_from_view and cb_alias

Not all format changes require data movement. TT-Blaze provides two zero-copy mechanisms for format reinterpretation.

### cb_from_view: Format Override on Overlapped Views

The `cb_from_view` method in FusedProgram accepts a `data_format` override that changes how the CB interprets the backing tensor's bytes without copying data:

```python
# EXISTING -- from blaze/fused_program.py (cb_from_view, line 1164)
def cb_from_view(self, view: "OverlappedView", **kwargs) -> CBHandle:
    """Allocate a CB from an OverlappedView.

    Accepts optional overrides for ``tile``, ``page_size``,
    ``core_ranges``, and ``data_format``.
    """
    eff_dtype = kwargs.get("data_format") or view.dtype
    eff_tile = kwargs.get("tile") or view.tile
    eff_page_size = kwargs.get("page_size") or view.page_size
```

**When this is safe:** Only when the target format is a strict "unpacking" of the source format (bfloat8_b -> bfloat16, bfloat4_b -> bfloat16). The underlying bytes remain the same; the CB's format descriptor tells the unpack hardware how to interpret the tile.

**When this is NOT safe:** Converting bfloat16 -> bfloat8_b via view reinterpretation would read bfloat16 bytes as if they were bfloat8_b blocks, producing garbage. BFP packing requires an explicit conversion op.

### cb_alias: Second CB ID on the Same Buffer

The `cb_alias` method creates a second CB ID that reads the same L1 bytes as an existing CB, but with a different format interpretation:

```python
# EXISTING -- from blaze/fused_program.py (cb_alias, line 1370)
def cb_alias(
    self,
    source: CBHandle,
    *,
    tile=None,
    page_size: int | None = None,
    total_size: int | None = None,
    dtype=None,
) -> CBHandle:
    """Second CB ID reading the same L1 bytes as *source* through a
    different format view."""
```

The key constraint:

```python
# EXISTING -- from blaze/fused_program.py
if source.is_direct_address:
    raise ValueError("cb_alias does not support direct-address CBHandles.")
```

Direct-address CBs (used for sharded weights) cannot be aliased because their address is fixed by the backing tensor. Aliased CBs must use FIFO mode.

### How the Negotiation Protocol Uses Aliasing

The negotiation protocol prefers `cb_alias` over explicit conversion when:

1. The conversion is unpack-direction (BFP -> bfloat16) -- the hardware handles it transparently.
2. Both the source and alias CBs are on the same core ranges.
3. The source CB is in FIFO mode.

```python
# PROPOSED -- alias-vs-conversion decision
def choose_conversion_strategy(
    src_format: ttnn.DataType,
    dst_format: ttnn.DataType,
    source_handle: CBHandle,
) -> str:
    """Decide whether to use cb_alias or a conversion op."""
    if src_format == dst_format:
        return "none"

    # Unpack-direction BFP -> standard: alias is safe
    if src_format in (ttnn.bfloat8_b, ttnn.bfloat4_b) and dst_format == ttnn.bfloat16:
        if source_handle.is_fifo:
            return "alias"

    # Standard -> BFP: requires packer, not a view reinterpretation
    if src_format in (ttnn.bfloat16, ttnn.float32) and dst_format in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        return "pack_convert"

    return "convert_op"
```

---

## Walkthrough: LLaMA Attention Format Negotiation

Consider a simplified LLaMA attention block with Q/K/V projections, QK^T scores, softmax, and attention output:

```text
Graph:
  act --[bfloat16]--> Q_proj(matmul) --> Q --> QK(matmul) --> scores
                                                                |
  act --[bfloat16]--> K_proj(matmul) --> K --> QK(matmul) -->   |
                                                                v
                                                        softmax(SDPA)
                                                                |
  act --[bfloat16]--> V_proj(matmul) --> V --> attn(matmul) <---+
                                                   |
                                                   v
                                           out_proj(matmul) --> output
```

### Forward Pass: Candidate Sets and Assignments

| Edge | Producer Pref | Consumer Pref | Candidates | Assigned |
|------|--------------|---------------|-----------|----------|
| act -> Q_proj.in0 | {bfloat16} | {bfloat16} | {bfloat16} | bfloat16 |
| W_q -> Q_proj.in1 | {bf8_b, bf4_b, bf16} | {bf8_b, bf4_b} | {bf8_b, bf4_b} | bfloat8_b |
| Q_proj.out -> QK.in0 | {bfloat16} | {bfloat16} | {bfloat16} | bfloat16 |
| QK.out -> softmax.in | {bfloat16} | {bfloat16} (required) | {bfloat16} | bfloat16 |
| softmax.out -> attn.in0 | {bfloat16} | {bfloat16} | {bfloat16} | bfloat16 |
| W_o -> out_proj.in1 | {bf8_b, bf4_b, bf16} | {bf8_b, bf4_b} | {bf8_b, bf4_b} | bfloat8_b |

### Backward Pass: Conversion Check

No adjustments needed. Every edge's assigned format satisfies both the producer and consumer preferences.

### Result: Zero Conversions

The LLaMA attention block has a natural format flow: bfloat8_b weights are unpacked for free by hardware, all intermediate tensors are bfloat16, and softmax handles float32 accumulation internally. The negotiation algorithm confirms this is already optimal.

The critical safety contribution of the negotiation algorithm here is *validation*: the `incompatible` set on the SDPA input ports prevents a developer from accidentally passing bfloat8_b or bfloat4_b scores to softmax, which would reduce precision from the shared exponent when scores have high dynamic range.

---

## Walkthrough: DeepSeek V3 MoE Format Negotiation

DeepSeek V3's MoE layer selects top-K experts per token, routes tokens to selected experts, and combines results.

### Format Assignments

| Op | Format | Rationale |
|----|--------|-----------|
| Router matmul weights | bfloat8_b | Small tensor (7168 x 256), needs precision for routing |
| Router output | bfloat16 | Softmax input; needs full precision for top-K selection |
| Expert gate weights | bfloat4_b | Large tensor (7168 x 2048 x 256 experts), maximum compression |
| Expert up weights | bfloat4_b | Same rationale |
| Expert down weights | bfloat4_b | Same rationale |
| Expert activations | bfloat16 | Compute precision for SiLU non-linearity |
| Combined output | bfloat16 | Result fed to residual add |

### Format Transitions

```text
Transition 1: Expert weights (bfloat4_b) -> matmul compute (bfloat16)
  Cost: FREE (unpacker expansion)

Transition 2: Gate output (bfloat16) -> SiLU activation (bfloat16)
  Cost: NONE (same format)
```

The MoE layer requires **zero explicit conversion ops** because all transitions are either same-format or handled by the unpacker. The BFP formats are specifically designed for weight storage, and the unpacker-to-bfloat16 conversion is the standard path.

### What Goes Wrong Without Negotiation

Without automatic negotiation, a developer might:
1. **Use bfloat16 for all expert weights:** 3.6x more L1 per expert, possibly exceeding budget
2. **Use bfloat4_b for the router weights:** Router scores need precision for top-K; bfloat4_b's 3-bit mantissa may corrupt routing decisions
3. **Use bfloat4_b for SiLU activation inputs:** The SiLU non-linearity `x * sigmoid(x)` amplifies precision loss near zero
4. **Use float32 everywhere "for safety":** 2x L1 vs bfloat16, 7x vs bfloat4_b -- does not fit

The negotiation algorithm prevents all four errors by applying the preference registry and incompatible-set constraints.

---

## Integration with CBHandle and FusedProgram

### The data_format Field on CBHandle

```python
# EXISTING -- from blaze/cb_handle.py
@dataclass(eq=False)
class CBHandle:
    cb_id: int
    num_pages: int
    page_size: int
    core_ranges: ttnn.CoreRangeSet
    data_format: object          # <-- this is the format
    tile_desc: object
    # ...
```

### CB ID Reuse Across Formats

The `CircularBufferIdManager` in `cb_reconfig.py` enables CB ID reuse when `(data_format, tile)` pairs match across contexts:

```python
# EXISTING -- from blaze/cb_reconfig.py
def _allocate_id(self, data_format, tile, exclude):
    key = (data_format, tile)
    for cb_id, fmt_key in self._id_to_format.items():
        if fmt_key == key and cb_id not in exclude:
            return cb_id
    # ... allocate new
```

Two CBs with the same format and tile geometry can share a CB ID across different program phases. The format negotiation algorithm can leverage this to minimize CB ID usage -- and the unification pass further reduces format diversity to maximize reuse opportunities.

### Disjoint-Core Format Sharing

```python
# EXISTING -- from blaze/fused_program.py
def _format_key(data_format, tile_desc, page_size=None) -> tuple | None:
    if data_format is None or tile_desc is None:
        return None
    h = getattr(tile_desc, "height", None)
    w = getattr(tile_desc, "width", None)
    if h is None or w is None:
        return None
    return (data_format, (h, w), page_size)
```

The format is part of the sharing key. Two CBs with different formats cannot share a CB ID even on disjoint cores.

---

## Negotiation Algorithm Complexity

| Phase | Time Complexity | What It Does |
|-------|----------------|-------------|
| Forward | O(V + E) | Topological walk with per-edge format selection |
| Backward | O(V + E) | Reverse walk for adjustment and conversion insertion |
| Unification | O(E) | One pass to converge formats |
| **Total** | **O(V + E)** | Linear in graph size |

For typical Blaze graphs (10-50 ops, 20-100 edges), negotiation completes in microseconds -- negligible compared to kernel compilation.

---

## Key Takeaways

- Each op declares format preferences at three levels per port: **preferred** (optimal performance), **tolerable** (correct but suboptimal), and **incompatible** (categorically blocked). SDPA requires `bfloat16` input and explicitly marks BFP formats as incompatible to prevent silent precision loss.
- The negotiation algorithm walks the graph in two passes (forward propagation + backward resolution) in O(V+E) time, followed by an optional unification pass that reduces format diversity to minimize CB ID pressure against the 64-slot hardware limit.
- Most format conversions in practice are **free** (handled by the unpacker hardware): `bfloat8_b` -> `bfloat16` and `bfloat4_b` -> `bfloat16` cost zero cycles. The negotiation algorithm exploits this by preferring BFP formats for weight storage while keeping activations in `bfloat16`.
- Two zero-copy mechanisms -- `cb_from_view` (format override on overlapped views) and `cb_alias` (second CB ID on same buffer, FIFO only) -- enable format reinterpretation without data movement. The negotiation protocol prefers these over explicit conversions when the direction is BFP-to-standard.
- When no preference is declared or no overlap exists, the system falls back to `bfloat16` (`DEFAULT_DTYPE`), which is universally supported.
- For LLaMA attention, the algorithm confirms zero conversions are needed. For DeepSeek V3 MoE, it assigns `bfloat4_b` to expert weights (3.6x L1 savings) while keeping activations in `bfloat16`.

## Source Files

- `blaze/cb_reconfig.py` -- `CircularBufferIdManager`: CB ID allocation with `(data_format, tile)` keyed reuse across contexts
- `blaze/cb_handle.py` -- `CBHandle.data_format`: the format field that negotiation populates
- `blaze/fused_program.py` -- `_format_key()`: disjoint-core sharing key includes format; `cb_from_view()`: tile reinterpretation; `cb_alias()`: second CB ID on same buffer
- `blaze/cb_engine.py` -- `DEFAULT_DTYPE = "bfloat16"`: the fallback format
- `blaze/blaze_op.py` -- `BlazeOp.math_fidelity`: per-op compute configuration that constrains format choices

---

← Previous: [Chapter 5 — Padding](../ch05_padding/) | Next: [Chapter 7 — Automatic CB Sizing and L1 Memory Budgeting](../ch07_cb_sizing/) →

< Previous: [File 01 -- Data Format Landscape](./01_data_format_landscape.md) | Next: [File 03 -- Format Conversion and CB Reconfig](./03_format_conversion_and_cb_reconfig.md) >
