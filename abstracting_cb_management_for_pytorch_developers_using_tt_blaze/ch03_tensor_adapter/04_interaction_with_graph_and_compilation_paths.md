# Interaction with Graph and Compilation Paths

The preceding files defined TensorAdapter's architecture (File 01), its shape metadata system (File 02), and its per-component integration classification (File 03). This file traces how TensorAdapter interacts with the two compilation paths in TT-Blaze -- the **Graph API** (`blaze.fuse()` context manager) and the **Composition API** (FusedProgram + `emit()` chains) -- and where shape metadata enters, propagates, and exits each path. It covers extending `FusionResult` and `ExternalTensor` to carry `ShapeDescriptor`, how `BlazeCompiler` uses shape metadata to auto-populate `user_args`, shadow graph recording, and graceful degradation when metadata is absent.

**What you will learn:**

- How `FusionContext` (`blaze/context.py`) and `blaze.fuse()` serve as the graph compilation path, and where `ShapeDescriptor` integrates
- How `FusionResult` is extended to carry `ShapeDescriptor` alongside `OpNode` + `port_name`
- How `ExternalTensor` is extended to carry `ShapeDescriptor` so shape metadata enters the graph from the very start
- How `BlazeCompiler.compile()` uses `ShapeDescriptor` to auto-populate `user_args`, with a graceful fallback when metadata is absent
- Shadow graph recording: how FusedProgram's `f.output()` calls record graph edges, and how TensorAdapter enriches this mechanism
- The three compilation paths: Graph API, Composition API, and the Hybrid path (mixed automatic and manual)
- Design decisions: why extend FusionResult (not a parallel graph), why extend ExternalTensor (not require pre-shaped tensors), why auto-derive at graph level

---

## The Two Compilation Paths

TT-Blaze supports two ways to build a program:

### Path 1: Graph API (blaze.fuse())

The developer declares ops and data dependencies within a `with blaze.fuse() as ctx:` block. The context manager captures op calls, builds a `BlazeGraph`, validates it, and hands it to `BlazeCompiler.compile()`.

```python
# EXISTING -- Graph API usage (from blaze/context.py)
with blaze.fuse() as ctx:
    act = ExternalTensor("activation", tensor=act_tensor)
    wt = ExternalTensor("weights", tensor=wt_tensor)
    mm_out = blaze.matmul({"in0": act, "in1": wt}, grid=core_grid)
    result = blaze.silu({"input": mm_out}, grid=core_grid)

compiler = BlazeCompiler(mesh_device)
program = compiler.compile(
    graph=ctx.graph,
    tensors={"activation": act_tensor, "weights": wt_tensor},
    output_tensor=output_tensor,
    user_args={"k_num_tiles": 128, "out_w_per_core": 16},  # Manual!
)
```

**The pain point:** The developer must manually compute and pass `user_args` because the graph has no shape metadata. TensorAdapter eliminates this by attaching `ShapeDescriptor` to graph edges and deriving `user_args` automatically.

### Path 2: Composition API (FusedProgram + emit())

```python
# EXISTING -- Composition API usage
f = FusedProgram(kernel=kernel_path, device=device, ...)
in0_handle = f.cb_from_tensor(act_tensor)
in1_handle, wt_addr = resolve_weight_direct_address(f, wt_tensor, ...)
out_handle = Matmul.emit(f, in0_handle, in1_handle, prefix="matmul")
silu_out = Silu.emit(f, out_handle, prefix="silu")
f.wire_output(silu_out, output_tensor)
result = f.build()
```

**The pain point:** Every CB parameter, CT arg, and blocking factor is computed manually. TensorAdapter wraps this by computing handles via `adapt()` before passing them to `emit()`.

### How TensorAdapter Supports Both Paths

```
                        TensorAdapter
                        +---------------------------+
                        |                           |
            Graph API   |   Composition API         |
            Path        |   Path                    |
            ----------  |   ----------------        |
                        |                           |
  with blaze.fuse():    |   f = FusedProgram(...)   |
    + ShapeDescriptor   |   handle = adapt(tensor)  |
    on ExternalTensor   |   out = Op.emit(f, ...)   |
    + ShapeDescriptor   |   tracker.record(out)     |
    on FusionResult     |                           |
                |       |           |               |
                v       |           v               |
            BlazeGraph  |   FusedProgram.build()    |
            + enriched  |   + shadow graph          |
            edges       |                           |
                |       |           |               |
                v       |           v               |
            BlazeCompiler.compile()                 |
            + auto user_args from ShapeDescriptor   |
                        +---------------------------+
```

| Criterion | Graph API | Composition API |
|-----------|-----------|-----------------|
| Entry point | `blaze.fuse()` context manager | `FusedProgram()` constructor |
| Shape carrier | `FusionResult.shape` -> `Edge.shape_descriptor` | `ShapeTracker` (id-based) -> shadow graph |
| CT arg derivation | `derive_user_args(ctx.graph)` before compile | `derive_user_args(f._shadow_graph)` after build |
| Best for | Multi-op graphs, automatic fusion | Fine-grained control, custom emit() patterns |

---

## Extending FusionResult

### Current State

```python
# EXISTING -- FusionResult (from blaze/context.py)
@dataclass
class FusionResult:
    """Handle returned by op calls inside a fuse() block."""
    node: OpNode
    port_name: str
```

`FusionResult` identifies *which op port* produced a value, but carries no information about *what shape* that value has.

### Proposed Extension

```python
# PROPOSED (illustrative, not prescriptive) -- Extended FusionResult
@dataclass
class FusionResult:
    """Handle returned by op calls inside a fuse() block.

    Now optionally carries ShapeDescriptor when TensorAdapter is active.
    """
    node: OpNode
    port_name: str
    shape: Optional[ShapeDescriptor] = None  # NEW: added by TensorAdapter

    @property
    def logical_shape(self) -> tuple[int, ...] | None:
        """Convenience: the output's logical shape, or None if unknown."""
        return self.shape.logical_shape if self.shape else None

    @property
    def tile_grid(self) -> tuple[int, ...] | None:
        """Convenience: tile counts, or None if unknown."""
        return self.shape.tile_grid_shape if self.shape else None
```

**Rationale:**

1. **Backward compatibility.** The `shape` field defaults to `None`. All existing code continues unchanged.
2. **Natural propagation.** When `FusionResult` is passed as an input to a subsequent op, `FusionContext.add_op()` can read `mm_out.shape` to infer the input shape for the downstream op.
3. **Derivation of user_args.** Edges carry `ShapeDescriptor` (set from `FusionResult.shape`), enabling automatic CT arg derivation.

### How Shape Enters FusionResult

When TensorAdapter is active, `FusionContext.add_op()` is enhanced to compute output shapes:

```python
# PROPOSED (illustrative, not prescriptive) -- Enhanced add_op()
def add_op(self, op_name, inputs, grid=None, **kwargs):
    spec = get_op_spec(op_name)
    node = OpNode(id=self._next_id(op_name), spec=spec, grid=grid, kwargs=kwargs)
    self._nodes.append(node)

    # --- Edge creation (shape propagation) ---
    input_shapes = {}
    for port_name, source in inputs.items():
        if isinstance(source, FusionResult):
            edge = Edge(
                producer=source.node, producer_port=source.port_name,
                consumer=node, consumer_port=port_name,
                shape_descriptor=source.shape,  # NEW: propagate onto edge
            )
            self._edges.append(edge)
            if source.shape:
                input_shapes[port_name] = source.shape
        elif isinstance(source, ExternalTensor):
            # ... existing handling ...
            if source.shape:
                input_shapes[port_name] = source.shape

    # --- NEW: Infer output shape from input shapes ---
    output_shape = None
    if self._shape_engine and input_shapes:
        output_shape = self._shape_engine.infer_output(op_name, input_shapes)

    return FusionResult(node=node, port_name=spec.output_ports[0].name,
                        shape=output_shape)
```

Shape inference inside `add_op()` is **optional** -- it only runs when a `_shape_engine` is attached to the `FusionContext`. Without TensorAdapter, `_shape_engine` is `None` and behavior is identical to today.

---

## Extending ExternalTensor

### Current State

```python
# EXISTING -- ExternalTensor (from blaze/context.py)
@dataclass
class ExternalTensor:
    """Placeholder for an external tensor input."""
    name: str
    tensor: Any = None
    metadata: dict = field(default_factory=dict)
```

### Proposed Extension

```python
# PROPOSED (illustrative, not prescriptive) -- Extended ExternalTensor
@dataclass
class ExternalTensor:
    """Placeholder for an external tensor input."""
    name: str
    tensor: Any = None
    metadata: dict = field(default_factory=dict)
    shape: Optional[ShapeDescriptor] = None  # NEW: added by TensorAdapter

    @classmethod
    def from_torch(cls, name, tensor, op_hint="", **kwargs):
        """Create an ExternalTensor with ShapeDescriptor from a PyTorch tensor."""
        desc = ShapeDescriptor.from_torch(tensor, op_hint=op_hint)
        return cls(name=name, tensor=None, shape=desc, **kwargs)

    @classmethod
    def from_ttnn(cls, name, tensor, **kwargs):
        """Create an ExternalTensor with ShapeDescriptor from a TTNN tensor."""
        desc = ShapeDescriptor.from_ttnn(tensor)
        return cls(name=name, tensor=tensor, shape=desc, **kwargs)
```

**Rationale:** Shape metadata must enter the graph at the boundary. If external tensors do not carry shapes, the first op has no input shape to reason about, and shape inference cannot begin. This is the same principle as XLA's shape tracking (DP12): shapes propagate from inputs through the graph.

**Design decision -- dedicated field vs. metadata dict:** The existing `metadata` dict could store the ShapeDescriptor (zero class modification). This was considered but rejected because: (1) a dedicated field is type-safe and self-documenting; (2) `ext.shape.logical_shape` is clearer than `ext.metadata.get("shape_descriptor").logical_shape`; (3) the `metadata` dict has no schema, making it easy to introduce typos. The additive field approach follows the same pattern as the Edge extension.

---

## Auto-Populating user_args from ShapeDescriptor

### Current State: Manual user_args

```python
# EXISTING -- Manual user_args (error-prone, verbose)
program = compiler.compile(
    graph=ctx.graph,
    tensors={"activation": act_tensor, "weights": wt_tensor},
    output_tensor=output_tensor,
    user_args={
        "k_num_tiles": 128,           # activation_width / tile_w = 4096 / 32
        "out_w_per_core": 16,         # weight_n_tiles / k_tiles
        "width": 128,                 # for rmsnorm: hidden_dim / 32
    },
)
```

### Proposed State: Automatic Derivation

```python
# PROPOSED (illustrative, not prescriptive) -- derive_user_args
def derive_user_args(graph: BlazeGraph) -> dict[str, dict]:
    """Walk graph edges and derive user_args from ShapeDescriptors.

    Returns: Mapping node_id -> {arg_name: value} for CTArgEngine consumption.

    When an edge lacks shape_descriptor (None), that edge is silently
    skipped. This enables graceful degradation: graphs built without
    TensorAdapter produce an empty dict, falling back to manual user_args.
    """
    user_args = {}
    for node in graph.topological_order():
        node_args = {}
        input_edges = graph.get_edges_to(node)

        if node.spec.name in ("matmul", "matmul_cb", "kn_sliced_matmul"):
            in0_edge = _find_edge(input_edges, "in0")
            in1_edge = _find_edge(input_edges, "in1")
            if in0_edge and in0_edge.shape_descriptor:
                k = in0_edge.shape_descriptor.tile_grid_shape[-1]
                node_args["k_num_tiles"] = k
            if in1_edge and in1_edge.shape_descriptor:
                total = in1_edge.shape_descriptor.total_tiles
                k = node_args.get("k_num_tiles", 1)
                node_args["out_w_per_core"] = total // k

        elif node.spec.name in ("rmsnorm", "padded_rmsnorm"):
            input_edge = _find_edge(input_edges, "input")
            if input_edge and input_edge.shape_descriptor:
                node_args["width"] = input_edge.shape_descriptor.tile_grid_shape[-1]

        elif node.spec.name in ("reduce", "gated_local_reduce"):
            input_edge = _find_edge(input_edges, "in0") or _find_edge(input_edges, "group1")
            if input_edge and input_edge.shape_descriptor:
                node_args["num_tiles"] = input_edge.shape_descriptor.total_tiles

        if node_args:
            user_args[node.id] = node_args

    return user_args
```

**Integration point:**

```python
# PROPOSED (illustrative, not prescriptive) -- Integration in compilation flow
with blaze.fuse(shape_engine=ShapeEngine()) as ctx:
    act = ExternalTensor.from_torch("activation", act_tensor, op_hint="matmul.in0")
    wt = ExternalTensor.from_ttnn("weights", wt_tensor)
    mm_out = blaze.matmul({"in0": act, "in1": wt}, grid=core_grid)
    result = blaze.silu({"input": mm_out}, grid=core_grid)

# Derive user_args automatically from graph shape metadata
auto_args = derive_user_args(ctx.graph)

compiler = BlazeCompiler(mesh_device)
program = compiler.compile(
    graph=ctx.graph,
    tensors={"activation": act_tensor, "weights": wt_tensor},
    output_tensor=output_tensor,
    user_args=auto_args,  # Automatic! Not manual.
)
```

**Override mechanism:** Manual `user_args` take precedence over auto-derived values. If the developer provides `user_args={"k_num_tiles": 64}`, this overrides the auto-derived value. This implements the P4 escape hatch at the compilation level.

When auto-derived values conflict with user-provided values, the system should emit a warning (cherry-picked from V4's enhanced `_build_user_overrides`):

```python
# PROPOSED (illustrative, not prescriptive) -- Conflict detection
def _build_user_overrides(self, graph, user_args, auto_args):
    merged = {}
    for node_id in set(list(user_args.keys()) + list(auto_args.keys())):
        manual = user_args.get(node_id, {})
        auto = auto_args.get(node_id, {})
        for key in set(list(manual.keys()) + list(auto.keys())):
            if key in manual and key in auto and manual[key] != auto[key]:
                warnings.warn(
                    f"Node {node_id}: user_args['{key}']={manual[key]} "
                    f"overrides shape-derived value {auto[key]}"
                )
            merged.setdefault(node_id, {})[key] = manual.get(key, auto.get(key))
    return merged
```

> **Warning:** The auto-derivation function must produce **exactly** the same integer values that CTArgEngine expects. During development, auto-derived values should be cross-checked against manual values for all existing tests.

---

## Shadow Graph Recording

### Current State

FusedProgram already records a "shadow graph" during composition-API usage. Each `cb_from_tensor()`, `cb_scratch()`, and `emit()` call creates graph nodes and edges internally. The shadow graph is used for visualization export (`BLAZE_EXPORT`) and debugging.

The mechanism works through `_node_id` and `_port_name` fields on `CBHandle`. When `Matmul.emit(f, in0, in1, prefix="matmul")` is called, FusedProgram stamps the output `CBHandle` with `_node_id = "matmul_1"` and `_port_name = "out"`, and creates shadow graph edges.

### How TensorAdapter Enriches the Shadow Graph

TensorAdapter attaches `ShapeDescriptor` to shadow graph edges, using the same `Edge.shape_descriptor` field from [File 03](./03_wrapping_vs_extending_vs_replacing.md):

```python
# PROPOSED (illustrative, not prescriptive) -- Shadow graph enrichment
def allocate_cb(self, f, desc, ...):
    handle = f.cb_scratch(...)
    # If FusedProgram has a shadow graph, attach shape to the edge
    if f._shadow_graph is not None:
        for edge in f._shadow_graph.edges:
            if (edge.producer.id == handle._node_id and
                edge.producer_port == handle._port_name):
                edge.shape_descriptor = desc
    return handle
```

This means the shadow graph, after composition, carries the same shape metadata as a Graph-API-constructed graph. The downstream compilation path can use `derive_user_args()` identically for both paths.

### ShapeTracker vs. Shadow Graph

Both propagate shape information through emit() chains. They serve different purposes:

| Mechanism | Scope | Lifetime | Purpose |
|-----------|-------|----------|---------|
| **ShapeTracker** | Within a single `adapt()` -> `emit()` -> `adapt()` chain | Discarded after `build()` | Enables the *next* `adapt()` call to recover the *previous* op's output shape |
| **Shadow Graph** | The complete FusedProgram composition | Persists with the compiled program | Enables `BlazeCompiler` to auto-derive `user_args`, visualization, and post-hoc analysis |

ShapeTracker is fine-grained (per-emit). Shadow graph is coarse-grained (full graph). Both are needed because composition is sequential (ShapeTracker sees one op at a time) while compilation is holistic (the compiler sees the whole graph).

---

## The Complete Compilation Flow with TensorAdapter

### Graph API Path

```python
# PROPOSED (illustrative, not prescriptive)
act = ExternalTensor.from_torch("activation", act_tensor, op_hint="matmul.in0")
wt = ExternalTensor.from_ttnn("weights", wt_tensor)

with blaze.fuse(shape_engine=ShapeEngine()) as ctx:
    mm_out = blaze.matmul({"in0": act, "in1": wt}, grid=core_grid)
    result = blaze.silu({"input": mm_out}, grid=core_grid)

auto_args = derive_user_args(ctx.graph)
program = compiler.compile(
    graph=ctx.graph,
    tensors={"activation": act_tensor, "weights": wt_tensor},
    output_tensor=output_tensor,
    user_args=auto_args,
)
output = program.run()
```

### Composition API Path

```python
# PROPOSED (illustrative, not prescriptive)
adapter = TensorAdapter(device=mesh_device)
tracker = ShapeTracker()
f = FusedProgram(kernel=kernel_path, device=device, ...)

act_typed = adapter.adapt(act_tensor, op_hint="matmul.in0", fused_program=f)
tracker.record(act_typed.handle, act_typed.shape)

wt_typed = adapter.adapt(wt_tensor, op_hint="matmul.in1", fused_program=f)
tracker.record(wt_typed.handle, wt_typed.shape)

mm_out = Matmul.emit(f, act_typed.handle, wt_typed.handle, prefix="matmul")
mm_shape = adapter.shape_engine.infer_matmul_output(act_typed.shape, wt_typed.shape)
tracker.record(mm_out, mm_shape)

silu_typed = adapter.adapt(mm_out, op_hint="silu.input", fused_program=f)
silu_out = Silu.emit(f, silu_typed.handle, prefix="silu")

f.wire_output(silu_out, output_tensor)
result = f.build()
# Shadow graph now carries ShapeDescriptor on edges
```

### The Hybrid Path (Mixed Manual and Automatic)

The escape hatch: within the same FusedProgram, some ops use TensorAdapter and others use manual configuration:

```python
# PROPOSED (illustrative, not prescriptive)
adapter = TensorAdapter(device=mesh_device)
f = FusedProgram(kernel=kernel_path, device=device, ...)

# Automatic: matmul via TensorAdapter
act_typed = adapter.adapt(act_tensor, op_hint="matmul.in0", fused_program=f)
wt_typed = adapter.adapt(wt_tensor, op_hint="matmul.in1", fused_program=f)
mm_out = Matmul.emit(f, act_typed.handle, wt_typed.handle, prefix="matmul")

# Manual: custom NOC pattern (TensorAdapter cannot handle this)
custom_cb = f.cb_scratch(
    name="custom_noc", num_pages=4, core_ranges=custom_grid,
    data_format=ttnn.bfloat16, tile=ttnn.Tile([32, 32]), page_size=2048,
)
CustomNocOp.emit(f, mm_out, custom_cb, prefix="custom")

# Both paths coexist in the same FusedProgram
result = f.build()
```

This hybrid model is essential for incremental adoption. Experienced Blaze developers use TensorAdapter for the 80% of ops that work with automatic configuration, retaining manual control for the 20% that need custom handling (paged KV cache, ring all-reduce, custom NOC patterns).

---

## Graceful Degradation

A critical design requirement is that the adapter's extensions never break existing code. The system degrades gracefully when shape metadata is absent:

| Scenario | What Happens | Developer Impact |
|----------|-------------|-----------------|
| ExternalTensor created without `from_torch()` | `shape=None` | No auto-derived CT args; developer provides `user_args` manually (same as today) |
| FusionResult lacks `shape` | Downstream ops receive `None` for shape inference | No shape validation at graph construction time; errors detected at runtime (same as today) |
| Edge in BlazeGraph lacks `shape_descriptor` | `derive_user_args()` returns empty dict for that edge | Compiler uses explicit `user_args` only (same as today) |
| Shadow graph edge lacks `shape_descriptor` | Fused op compilation proceeds without auto-derivation | Developer provides CT args via `user_args` (same as today) |

In every case, the fallback behavior is identical to the current TT-Blaze workflow. The adapter adds capabilities when metadata is available but does not require it.

### Validation of Graceful Degradation

```python
# Test: graph construction without shape metadata (backward compat)
with blaze.fuse() as ctx:
    act = ExternalTensor("activation")  # no shape
    w = ExternalTensor("weights")       # no shape
    out = blaze.matmul({"in0": act, "in1": w}, grid=cores)

assert out.shape is None                                    # graceful: no inference
assert all(e.shape_descriptor is None for e in ctx.graph.edges)  # no metadata
# compile() works as before with explicit user_args
```

```python
# Test: graph construction with shape metadata (new capability)
with blaze.fuse(shape_engine=ShapeEngine()) as ctx:
    act = ExternalTensor.from_torch("activation", torch.randn(1, 4096))
    w = ExternalTensor.from_torch("weights", torch.randn(4096, 11008))
    out = blaze.matmul({"in0": act, "in1": w}, grid=cores)

assert out.shape is not None
assert out.logical_shape == (1, 11008)
# compile() can auto-derive k_num_tiles=128, out_w_per_core from shapes
```

---

## Design Decisions

### Why Extend FusionResult (Not a Parallel Graph)?

An alternative considered was building a separate "shape graph" parallel to the BlazeGraph, with shape nodes mirroring op nodes. This was rejected because: (1) keeping two graphs in sync is error-prone; (2) shape metadata naturally belongs on the same edges as data flow; (3) the compilation path expects a single BlazeGraph, not two.

### Why Extend ExternalTensor (Not Require Pre-Shaped Tensors)?

An alternative was requiring developers to pre-compute ShapeDescriptors and pass them via a separate dict. This was rejected because: (1) it splits the information for a single tensor across two data structures; (2) factory methods (`from_torch`, `from_ttnn`) are more ergonomic; (3) the extension is backward-compatible.

### Why Auto-Derive at Graph Level (Not Per-emit)?

The `derive_user_args()` function operates on the complete graph, not on individual emit() calls. This is because: (1) some CT args depend on multiple ops' shapes (e.g., `out_w_per_core` depends on both in0 and in1); (2) graph-level derivation enables global optimization (e.g., consistent blocking across a fusion chain); (3) it works identically for both Graph API and Composition API (via the shadow graph).

---

## Forward References

The graph and compilation path extensions defined here are consumed by:

| Chapter | What It Uses |
|---------|-------------|
| [Chapter 4: Tile Decomposition](../ch04_tile_decomposition/01_three_shape_layers.md) | `ShapeDescriptor.from_torch()` computes the three shape layers |
| [Chapter 5: Padding](../ch05_padding/01_padding_strategy_taxonomy.md) | `padding_strategy` and `padding_fill` fields propagate through graph edges |
| [Chapter 6: Data Formats](../ch06_data_formats/01_data_format_landscape.md) | `data_format` field enables format negotiation across edges |
| [Chapter 7: CB Sizing](../ch07_cb_sizing/01_l1_memory_model.md) | `total_tiles` and `page_size` enable automatic CB sizing |
| [Chapter 9: Op Fusion](../ch09_op_fusion/01_blaze_fusion_model.md) | Shape propagation through fused op boundaries uses ShapeTracker + shadow graph |
| [Chapter 10: End-to-End](../ch10_end_to_end/01_the_blaze_module_api.md) | `BlazeModule.from_torch()` orchestrates the full pipeline |

---

## Key Takeaways

- TT-Blaze has **two compilation paths**: the Graph API (`blaze.fuse()` -> `BlazeCompiler.compile()`) and the Composition API (FusedProgram + `emit()` chains). TensorAdapter supports both, using shape-enriched `FusionResult` and `ExternalTensor` for the Graph path, and `ShapeTracker` for the Composition path.
- `FusionResult` and `ExternalTensor` are **extended** with optional `ShapeDescriptor` fields (defaulting to `None`). Shape enters the graph via `ExternalTensor.shape` and propagates via `FusionResult.shape`, flowing onto `Edge.shape_descriptor`.
- The `derive_user_args()` function walks graph edges to **automatically compute** shape-dependent CT args (`k_num_tiles`, `out_w_per_core`, `width`, `num_tiles`). Manual `user_args` override auto-derived values (P4 escape hatch). Conflicts produce warnings.
- The **shadow graph** is enriched with `ShapeDescriptor` by CBBackend, enabling the same `derive_user_args()` function to work for both compilation paths.
- The **hybrid model** allows mixing TensorAdapter-managed ops and manually-configured ops within the same `FusedProgram`, enabling incremental adoption.
- All extensions **degrade gracefully**: when shape metadata is absent, the system behaves identically to the current manual workflow. No existing code breaks.

## Source Files

- `blaze/context.py` -- `FusionContext`, `FusionResult` (extended: `shape` field), `ExternalTensor` (extended: `shape` field + factory methods), `fuse()` context manager
- `blaze/graph.py` -- `BlazeGraph`, `Edge` (extended: `shape_descriptor`), `OpNode`, `compile_engines()`
- `blaze/compiler.py` -- `BlazeCompiler.compile()`, `_compile_single()`, `_compile_fused_op()`
- `blaze/ct_args.py` -- `CTArgEngine.generate_tuples()`, `CTArgSchema`, `CTArgSpec`
- `blaze/fused_program.py` -- Shadow graph recording, `_node_id`/`_port_name` stamps
- `blaze/cb_handle.py` -- `CBHandle._node_id`, `CBHandle._port_name` (graph identity fields)

---

[Previous: 03 -- Wrapping vs. Extending vs. Replacing](./03_wrapping_vs_extending_vs_replacing.md) | [Next: Chapter 4 -- Automatic Tile Decomposition](../ch04_tile_decomposition/01_three_shape_layers.md)
