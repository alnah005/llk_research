# Automatic Padding Insertion and Propagation

With the padding taxonomy (File 01) and registry (File 02) in place, the system can now automatically insert padding operations where the logical shape differs from the padded shape, select the correct fill value per op port, and propagate padding metadata through fused op chains. This section defines the padding pass algorithm, the PadOp MicroOp, optimizations that skip redundant padding, chain propagation with logical shape tracking, and re-padding at boundary crossings where consecutive ops require different padding strategies.

> *Design Principle P3 (Validate at Each Lowering Step):* The padding pass runs after shape inference (Chapter 4) and before CB allocation (Chapter 7). It validates that every graph edge has a compatible padding strategy between producer and consumer, inserting correction nodes where mismatches exist. No edge passes through unchecked.

**What you will learn:**

- The padding pass algorithm: walk the graph after shape inference, compare producer output padding with consumer input requirements from the registry, insert padding ops where they differ
- The PadOp MicroOp: a lightweight NCRISC-only kernel that writes fill values to padded tile positions, with zero-fill, neg-inf-fill, and identity-fill variants
- Seven optimization scenarios where PadOps are unnecessary, including the "producer-handles-it" optimization for matmul, tilized reads, and activation functions where `f(0) = 0`
- Chain propagation: how the logical shape `[7, 2048]` is tracked through matmul -> SiLU -> matmul chains so the final output can be unpadded back to `[7, 64]`
- Re-padding at boundary crossings: when a matmul produces ZERO-padded output but the downstream softmax requires NEG_INF, the pass inserts a re-padding step
- End-to-end walkthroughs through attention (seq_len=5) and MLP (hidden_dim=4000) patterns

---

## The Padding Pass

### Position in the Compilation Pipeline

```
Pipeline:
  1. Graph construction (blaze.fuse() / compose())
  2. Shape inference (Chapter 4: tile_decompose + infer_output_shape)
  3. ** Padding pass ** (this file)
  4. Format negotiation (Chapter 6)
  5. CB sizing (Chapter 7)
  6. CB ID assignment (CBEngine)
  7. Semaphore assignment (SemEngine)
  8. CT arg resolution (CTArgEngine)
  9. Compilation (BlazeCompiler)
```

The padding pass must run **after** shape inference (needs `logical_shape` and `padded_shape`) and **before** CB sizing (padding ops introduce new CBs that consume L1).

### The Algorithm

```python
# PROPOSED -- padding pass algorithm
def run_padding_pass(
    graph: BlazeGraph,
    shape_tracker: ShapeTracker,
) -> PaddingPassResult:
    """Walk the graph and insert/adjust padding at each edge.

    For each edge (producer_port -> consumer_port):
    1. Look up the consumer's required padding from the registry
    2. Look up the producer's output padding from the ShapeTracker
    3. If compatible, skip
    4. If producer can handle it directly, update producer metadata
    5. Otherwise, insert a PadOp or RepadOp node

    Returns PaddingPassResult with diagnostics and statistics.
    Mutates graph in-place: may add PadOp nodes and redirect edges.
    """
    diagnostics = []
    stats = PaddingPassStats()

    for edge in graph.edges:
        stats.total_edges += 1
        producer_desc = shape_tracker.lookup_by_op_port(
            edge.producer.id, edge.producer_port
        )

        # Step 1: Check if padding exists
        needs_padding = (
            producer_desc.logical_shape != producer_desc.padded_shape
        )
        if not needs_padding:
            stats.compatible_skip += 1
            continue

        stats.edges_needing_padding += 1

        # Step 2: Resolve consumer's required strategy
        consumer_policy = resolve_padding_for_port(
            op_type=edge.consumer.spec.name,
            port_name=edge.consumer_port,
            producer_spec=PaddingSpec(
                policy=PaddingPolicy[producer_desc.padding_strategy],
                fill_value=producer_desc.padding_fill,
            ),
        )

        # Step 3: Check compatibility
        producer_policy = PaddingSpec(
            policy=PaddingPolicy[producer_desc.padding_strategy],
            fill_value=producer_desc.padding_fill,
        )

        if _policies_compatible(producer_policy, consumer_policy):
            stats.compatible_skip += 1
            continue

        # Step 4: Try producer-handles-it optimization
        if _producer_can_pad(edge.producer, consumer_policy):
            _update_producer_padding(shape_tracker, edge.producer, consumer_policy)
            stats.producer_handled += 1
            diagnostics.append(PaddingMismatch(
                producer_op=edge.producer.spec.name,
                producer_port=edge.producer_port,
                producer_policy=producer_policy.policy,
                consumer_op=edge.consumer.spec.name,
                consumer_port=edge.consumer_port,
                consumer_policy=consumer_policy.policy,
                resolution="producer_handles",
            ))
            continue

        # Step 5: Insert re-padding node
        pad_node = _insert_repad_node(graph, edge, producer_desc, consumer_policy)
        shape_tracker.record_op(pad_node.id, "out", producer_desc.with_padding(
            padding_strategy=consumer_policy.policy.value,
            padding_fill=consumer_policy.fill_value,
        ))
        stats.repads_inserted += 1
        stats.extra_cbs_allocated += 1
        stats.extra_l1_bytes += producer_desc.total_tiles * producer_desc.page_size
        diagnostics.append(PaddingMismatch(
            producer_op=edge.producer.spec.name,
            producer_port=edge.producer_port,
            producer_policy=producer_policy.policy,
            consumer_op=edge.consumer.spec.name,
            consumer_port=edge.consumer_port,
            consumer_policy=consumer_policy.policy,
            resolution="repad_inserted",
        ))

    return PaddingPassResult(diagnostics=diagnostics, stats=stats)
```

### Policy Compatibility Check

```python
# PROPOSED -- padding policy compatibility
def _policies_compatible(producer: PaddingSpec, consumer: PaddingSpec) -> bool:
    """Check if producer's padding is compatible with consumer's requirements."""
    if consumer.policy == PaddingPolicy.NONE:
        return True
    if producer.policy == consumer.policy:
        if producer.policy == PaddingPolicy.IDENTITY:
            return producer.identity_op == consumer.identity_op
        return True
    if producer.policy == PaddingPolicy.NONE:
        return True  # No padded positions exist
    if (producer.policy == PaddingPolicy.ZERO
            and consumer.policy == PaddingPolicy.MASK):
        return True  # MASK adds count correction; data is already zero-padded
    if (producer.policy == PaddingPolicy.NEG_INF
            and consumer.policy == PaddingPolicy.IDENTITY
            and consumer.identity_op == "max"):
        return True  # -inf is the max identity
    return False
```

---

## The PadOp MicroOp

### Design

The PadOp is a lightweight MicroOp that writes fill values to padded tile positions. It runs on NCRISC (data mover), not TRISC (compute), meaning it costs zero SFPU cycles:

```python
# PROPOSED -- PadOp MicroOp
class PadOp(MicroOp):
    """Fills padded tile positions with a specified value.

    The NCRISC kernel copies logical_width * element_size bytes per row
    from the input CB, then fills (padded_width - logical_width) * element_size
    bytes with the encoded fill value. For row padding, entire rows are filled.

    CT args for NCRISC:
      - real_bytes_per_row:    logical_width * element_size
      - padded_bytes_per_row:  padded_width * element_size
      - logical_height:        number of real data rows
      - padded_height:         total rows (including padding)
      - fill_packed:           uint32 with two packed bfloat16 fill values
      - num_tiles:             total tile count
    """
    name: str = "pad_op"
    input: Input = Input()
    output: Output = Output()

    @staticmethod
    def emit(f, input_handle, *, prefix="pad", cores, fill_value=0.0,
             logical_width, padded_width, logical_height, padded_height,
             data_format="bfloat16", tile_desc=None) -> CBHandle:
        elem_sz = get_element_size_bytes(data_format)
        fill_packed = encode_fill_value(
            PaddingSpec(policy=_infer_policy(fill_value), fill_value=fill_value),
            data_format,
        )
        tile = tile_desc or input_handle.tile_desc
        out = f.cb_scratch(
            name=BlazeOp.cb_name(prefix, "padded_out"),
            num_pages=input_handle.num_pages, core_ranges=cores,
            data_format=data_format, tile=tile,
            page_size=input_handle.page_size,
        )
        f.ncrisc_ct_args([
            (f"{prefix}.real_bytes_per_row", logical_width * elem_sz),
            (f"{prefix}.padded_bytes_per_row", padded_width * elem_sz),
            (f"{prefix}.logical_height", logical_height),
            (f"{prefix}.padded_height", padded_height),
            (f"{prefix}.fill_packed", fill_packed),
            (f"{prefix}.num_tiles", input_handle.num_pages),
        ])
        return f.output("pad_op", out, core_ranges=cores, prefix=prefix,
                        input=input_handle)
```

### Existing Precedent: PaddedRMSNorm's Internal Padding

The PadOp generalizes a pattern already used in `PaddedRMSNorm.emit()`:

```python
# EXISTING -- from blaze/ops/padded_rmsnorm/op.py
# PaddedRMSNorm uses CT args to tell the NCRISC kernel where real data ends
ct_args = [
    ("input_bytes",  valid_rows * width * elem_sz),
    ("padded_bytes", padded_rows * width * elem_sz),
    ("pad_rows",     padded_rows - valid_rows),
    ("valid_rows",   valid_rows),
]
```

The proposed PadOp lifts this per-op pattern into a reusable MicroOp with standardized CT args (`real_bytes_per_row`, `padded_bytes_per_row`, `fill_packed`), eliminating the need for each op to implement its own padding logic.

### Cost Model

| Resource | Cost | Impact |
|----------|------|--------|
| L1 memory | 1 extra CB (output) | `num_pages * page_size` bytes |
| Compute | Zero SFPU cycles | NCRISC-only (data movement, no math) |
| Latency | NCRISC copy + fill | Proportional to padded size |
| CB IDs | 1 additional ID | Counts toward 64-ID limit |

> **Warning:** Each PadOp consumes one CB ID from the 64-ID hardware limit. In a complex fused op like a gated MLP with 11 nodes, adding PadOps at every boundary could consume 5-10 additional CB IDs. The optimization strategies below minimize insertions.

---

## Optimization: Producer-Handles-It

Many ops produce correctly-padded output without a separate PadOp. Seven scenarios where PadOps are unnecessary:

| # | Scenario | Mechanism | Applicable Policy |
|---|----------|-----------|------------------|
| 1 | `cb_from_tensor` (tiled tensor read) | Pre-padded in DRAM during tilization | ZERO |
| 2 | Matmul output | Zero-initialized accumulation dest | ZERO |
| 3 | SiLU / ReLU / GELU | `f(0) = 0` preserves zero padding | ZERO |
| 4 | PaddedRMSNorm | Explicit internal padding CBs | ZERO |
| 5 | SDPA | Internal masking mechanism | All (handled internally) |
| 6 | Elementwise add/mul | `0 + x = x`, `0 * x = 0` | ZERO |
| 7 | Data movement (mcast/gather) | Passes data through unchanged | INHERIT |

```python
# PROPOSED -- producer-can-pad check
def _producer_can_pad(producer_node: OpNode, required_policy: PaddingSpec) -> bool:
    op_type = producer_node.spec.name

    if op_type in ("matmul", "matmul_cb", "kn_sliced_matmul",
                    "dram_streaming_matmul"):
        return required_policy.policy == PaddingPolicy.ZERO

    if op_type in ("silu", "relu", "gelu", "clamped_silu"):
        return required_policy.policy == PaddingPolicy.ZERO

    if op_type == "padded_rmsnorm":
        return required_policy.policy == PaddingPolicy.ZERO

    if op_type == "sdpa_decode":
        return True  # All padding handled internally

    if op_type == "cb_from_tensor":
        return required_policy.policy == PaddingPolicy.ZERO

    return False
```

> **Warning:** The "tilization handles it" optimization assumes the host-side tilization step zero-fills padding positions. This is the default behavior of `ttnn.from_torch()`, but custom tilization paths may not guarantee this.

---

## Chain Propagation with Logical Shape Tracking

### The Problem

When padding is applied and the tensor flows through multiple ops, the **logical shape** must be tracked so the final output can be correctly unpadded:

```
Input:           [7, 2048]   logical
Padded:          [32, 2048]  padded (height from 7 to 32)

After matmul with weights [2048, 512]:
  Padded output: [32, 512]   <- hardware processes all 32 rows
  Logical output: [7, 512]   <- only first 7 rows are real

After SiLU:
  Padded output: [32, 512]   <- SiLU(0) = 0, padding stays zero
  Logical output: [7, 512]   <- unchanged

After another matmul with weights [512, 2048]:
  Padded output: [32, 2048]  <- 32 rows processed
  Logical output: [7, 2048]  <- only 7 valid

Final unpadding: output[0:7, 0:2048] = [7, 2048]
```

### How ShapeDescriptor Carries Padding Through Chains

Each op's `infer_output_shape` (Chapter 4, File 03) propagates logical shape alongside padded shape:

```python
# PROPOSED -- matmul shape inference preserves logical dimensions
class MatmulShapeRule(ShapeInferenceRule):
    @staticmethod
    def infer_output_shape(input_descriptors, op_params=None):
        in0 = input_descriptors["in0"]
        in1 = input_descriptors["in1"]
        M_logical = in0.logical_shape[-2]   # 7, not 32
        N_logical = in1.logical_shape[-1]
        M_padded = in0.padded_shape[-2]     # 32
        N_padded = in1.padded_shape[-1]
        return ShapeDescriptor(
            logical_shape=(M_logical, N_logical),
            padded_shape=(M_padded, N_padded),
            padding_strategy="ZERO",
            padding_fill=0.0,
            # ... other fields
        )
```

### The PaddingChainTracker

```python
# PROPOSED -- padding chain metadata
@dataclass(frozen=True)
class PaddingChainEntry:
    """Records padding decisions at each step in an op chain."""
    op_id: str
    op_type: str
    logical_shape: tuple[int, ...]
    padded_shape: tuple[int, ...]
    padding_strategy: PaddingPolicy
    padding_fill: float
    was_repadded: bool


class PaddingChainTracker:
    """Track padding decisions through an op chain.

    Records the padding state at each graph node so the final
    unpadding step knows exactly what to strip.
    """
    def __init__(self):
        self._chain: list[PaddingChainEntry] = []
        self._node_to_entry: dict[str, PaddingChainEntry] = {}

    def record(self, node_id, op_type, desc, was_repadded=False):
        entry = PaddingChainEntry(
            op_id=node_id, op_type=op_type,
            logical_shape=desc.logical_shape,
            padded_shape=desc.padded_shape,
            padding_strategy=PaddingPolicy[desc.padding_strategy],
            padding_fill=desc.padding_fill,
            was_repadded=was_repadded,
        )
        self._chain.append(entry)
        self._node_to_entry[node_id] = entry

    def get_final_unpad_spec(self) -> dict:
        if not self._chain:
            return {"needs_unpad": False}
        last = self._chain[-1]
        if last.logical_shape == last.padded_shape:
            return {"needs_unpad": False}
        return {
            "needs_unpad": True,
            "logical_shape": last.logical_shape,
            "padded_shape": last.padded_shape,
            "slices": tuple(slice(0, l) for l in last.logical_shape),
        }
```

---

## Re-Padding at Boundary Crossings

### When Re-Padding Is Required

The most common re-padding case in LLM models is the attention mechanism:

```
QK matmul output: ZERO padded (padded scores are 0.0)
Softmax input:    NEG_INF required (padded scores must be -inf)

Without re-padding:
  softmax([s0, s1, s2, 0.0, ..., 0.0]) -> WRONG (exp(0)=1 pollutes denominator)

With re-padding:
  repad([s0, s1, s2, 0.0, ..., 0.0], fill=-inf) -> [s0, s1, s2, -inf, ..., -inf]
  softmax([s0, s1, s2, -inf, ..., -inf]) -> CORRECT
```

### The Re-Padding Node

```python
# PROPOSED -- re-padding node insertion
def _insert_repad_node(graph, edge, producer_desc, target_policy):
    """Insert a re-padding node on a graph edge.

    Creates a PadOp that reads the producer's output, replaces the
    padding fill value, and writes to a new CB. Only padded positions
    are modified; real data passes through unchanged.
    """
    repad_node = OpNode(
        id=f"repad_{edge.producer.id}_{edge.consumer.id}",
        spec=OpSpec(name="repad", kernel_path="blaze/ops/pad/kernels/repad.hpp",
                    op_class="blaze::Repad",
                    input_ports=[TensorPort(name="input")],
                    output_ports=[TensorPort(name="output")]),
        kwargs={
            "logical_shape": producer_desc.logical_shape,
            "padded_shape": producer_desc.padded_shape,
            "old_fill": producer_desc.padding_fill,
            "new_fill": target_policy.fill_value,
        },
    )
    graph.insert_node_on_edge(edge, repad_node)
    return repad_node
```

### Re-Padding Cost

| Metric | Cost |
|--------|------|
| CB overhead | 1 additional CB (same size as input) |
| Compute | Zero SFPU cycles (NCRISC fill only) |
| CB ID | 1 additional from the 64-ID pool |
| Latency | Proportional to `padded_size - logical_size` |

For tile-aligned production models, re-padding is rare because NONE applies. Re-padding matters for non-standard sequence lengths and the softmax boundary.

---

## End-to-End Walkthrough: Attention Block (seq_len=5)

```
seq_len = 5, hidden_dim = 64, tile = (32, 32)
Padded activation: [1, 32, 64] (seq padded from 5 to 32)
```

**Step 1: Q/K/V projections (matmul -- ZERO correct)**
```
in0: activation [1,5,64] padded [1,32,64] -- cb_from_tensor pre-padded -> ok
in1: W_q [64,64] -- tile-aligned -> NONE
Output Q,K,V: logical [1,5,64], padded [1,32,64], ZERO
No PadOp needed (producer-handles-it: matmul + tilized read)
```

**Step 2: QK scores (matmul)**
```
Q [1,5,64] padded [1,32,64] @ K^T [1,64,5] padded [1,64,32]
Registry: ZERO for both -> compatible
Output: logical [1,5,5], padded [1,32,32], ZERO
Scores: real at [0:5, 0:5], zeros elsewhere
```

**Step 3: Softmax (CRITICAL BOUNDARY)**
```
Registry: softmax.scores -> NEG_INF
Producer: matmul -> ZERO
_policies_compatible(ZERO, NEG_INF) -> FALSE
_producer_can_pad(matmul, NEG_INF) -> FALSE

** INSERT REPAD: ZERO -> NEG_INF **
  Positions [0:5, 5:32]: 0.0 -> -inf
  Positions [5:32, :]:   0.0 -> -inf

Softmax output: [p00..p04, 0, ..., 0] for real rows; all-zero for padding rows
  padding_strategy = ZERO (exp(-inf) = 0)
```

**Step 4: Attention output (softmax_probs @ V)**
```
probs: ZERO (compatible with matmul.in0 -> ZERO)
V: ZERO (compatible with matmul.in1 -> ZERO)
No PadOp needed

Output: logical [1,5,64], padded [1,32,64], ZERO
```

**Step 5: Final unpadding**
```
output[0:1, 0:5, 0:64] -> [1, 5, 64]
```

**Summary: 1 RepadOp inserted (ZERO -> NEG_INF at softmax boundary)**

---

## End-to-End Walkthrough: MLP (hidden_dim=4000)

```
hidden_dim = 4000, intermediate_dim = 11008, tile = (32, 32)
Activation: [1, 4000] padded [32, 4000] (4000 is already tile-aligned; only height pads 1 -> 32)
```

**Step 1: Gate matmul ([1,4000] @ [4000,11008])**
```
in0: [1,4000] padded [32,4000], ZERO (tilization)
in1: [4000,11008] padded [4000,11008], ZERO (already tile-aligned)
K dimension: 4000 (tile-aligned, no extra zeros needed)
Output: logical [1,11008], padded [32,11008], ZERO
No PadOp needed
```

**Steps 2-4: Up matmul, SiLU(gate), gate * up**
```
All ZERO-compatible:
  silu(0) = 0 -> producer-handles-it
  0 * up_padding = 0, gate_padding * 0 = 0
No PadOps needed
```

**Step 5: Down matmul ([1,11008] @ [11008,4000])**
```
in0: [1,11008] padded [32,11008], ZERO
in1: [11008,4000] padded [11008,4000], ZERO (width tile-aligned)
Output: logical [1,4000], padded [32,4000], ZERO
```

**Step 6: Unpadding**
```
output[0:1, 0:4000] -> [1, 4000]
```

**Summary: ZERO PadOps for the entire non-aligned MLP**

Even with a non-aligned hidden dimension, no explicit PadOps are needed because tilization zero-fills the input, matmul produces zero for padded positions, and SiLU/elementwise ops preserve zero padding. The "producer-handles-it" optimization is highly effective for matmul + activation chains.

---

## Padding Pass Validation

```python
# PROPOSED -- post-pass validation
def validate_padding_pass(graph, shape_tracker):
    """Verify no unresolved mismatches remain after the pass."""
    errors = []
    for edge in graph.edges:
        producer_desc = shape_tracker.lookup_by_op_port(
            edge.producer.id, edge.producer_port)
        consumer_policy = resolve_padding_for_port(
            edge.consumer.spec.name, edge.consumer_port)
        producer_policy = PaddingSpec(
            policy=PaddingPolicy[producer_desc.padding_strategy])
        if not _policies_compatible(producer_policy, consumer_policy):
            errors.append(
                f"Unresolved: {edge.producer.id}.{edge.producer_port} "
                f"({producer_policy.policy.value}) -> "
                f"{edge.consumer.id}.{edge.consumer_port} "
                f"({consumer_policy.policy.value})")
    if errors:
        raise PaddingPassError(
            f"Padding pass left {len(errors)} unresolved mismatches:\n"
            + "\n".join(errors))
```

### Padding Pass Statistics

```python
# PROPOSED -- padding pass diagnostics
@dataclass
class PaddingPassStats:
    total_edges: int = 0
    edges_needing_padding: int = 0
    padops_inserted: int = 0
    repads_inserted: int = 0
    producer_handled: int = 0
    compatible_skip: int = 0
    extra_cbs_allocated: int = 0
    extra_l1_bytes: int = 0

@dataclass
class PaddingPassResult:
    diagnostics: list[PaddingMismatch]
    stats: PaddingPassStats
```

---

## Handling Reshape and Transpose

Reshape operations change the logical shape without moving data. The padding pass must handle these carefully:

```
Input:   logical=[1, 4096], padded=[32, 4096], grid=[1, 128]
         reshape to [32, 128]
Output:  logical=[32, 128], padded=[32, 128], grid=[1, 4]
```

Total element count must be preserved. When a reshape changes which dimensions are padded, the padding strategy may need to be re-evaluated against the downstream consumer.

Transpose swaps the last two dimensions:

```
Input:   logical=[4096, 11008], tile=(32, 32), grid=[128, 344]
Output:  logical=[11008, 4096], tile=(32, 32), grid=[344, 128]
```

> **Warning:** A transpose of a padded tensor changes which positions are "padding" and which are "real data." If the tensor is padded along the height dimension and then transposed, the padding is now along the width dimension. The PaddingChainTracker must update accordingly.

---

## Key Takeaways

- The padding pass runs after shape inference and before CB sizing. It walks every graph edge, compares the producer's output padding with the consumer's required padding from the registry, and inserts PadOp or RepadOp nodes where mismatches exist. The pass mutates the graph in-place and returns diagnostics.
- The PadOp MicroOp runs on NCRISC (data mover), not TRISC (compute), costing zero SFPU cycles. It copies real data and fills padding positions with the specified value. Each PadOp consumes one CB ID from the 64-ID limit.
- The "producer-handles-it" optimization eliminates PadOps for seven common scenarios: tilized reads (pre-padded), matmul (zero-initialized accumulators), activation functions where `f(0)=0`, PaddedRMSNorm (internal padding), SDPA (internal masking), elementwise ops, and data movement ops. For production LLMs with tile-aligned dimensions, this typically means zero PadOps are inserted.
- Chain propagation tracks the logical shape through every op via `ShapeDescriptor.logical_shape`. The critical invariant: the logical shape is the **original** shape before any padding and survives through all transformations unchanged. The PaddingChainTracker records the full chain for diagnostics and final unpadding.
- Re-padding at boundary crossings (ZERO -> NEG_INF at softmax) is the most common case where a PadOp is unavoidable. The attention walkthrough shows a single RepadOp suffices for the entire block.
- For production LLMs (LLaMA-7B, DeepSeek V3) with tile-aligned dimensions, the padding pass inserts zero PadOps. PadOps become important for non-standard sequence lengths, custom architectures, and the softmax boundary when the sequence length is not a multiple of 32.

## Source Files

- `blaze/ops/padded_rmsnorm/op.py` -- `PaddedRMSNorm.emit()`: canonical pattern for explicit padding with Internal CBs (NCRISC copy-and-fill + TRISC row-zeroing)
- `blaze/ops/sdpa/op.py` -- `SDPADecode.emit()`: implicit masking where padding is handled at the score level
- `blaze/utils.py` -- `interpret_tile_padded()`: computes padded tile geometry before the padding pass
- `blaze/graph.py` -- `Edge`, `OpNode`, `BlazeGraph`: graph structures the pass traverses and mutates
- `blaze/context.py` -- `FusionContext`: compilation context where the pass integrates

---

← Previous: [Chapter 4: Automatic Tile Decomposition](../ch04_tile_decomposition/) | Next: [Chapter 6: Data Format Selection](../ch06_data_formats/) →

[Previous file: 02 -- Op Padding Registry](./02_op_padding_registry.md) | [Next file: 04 -- Output Unpadding](./04_output_unpadding.md)
