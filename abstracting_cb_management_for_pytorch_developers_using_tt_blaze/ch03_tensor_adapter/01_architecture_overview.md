# The TensorAdapter Architecture: Three Layers from PyTorch to Hardware

The TensorAdapter is the central abstraction proposed in this guide -- a shape-aware argument builder that bridges PyTorch's tensor-level thinking to TT-Blaze's CB-level execution model. This section defines its three-layer internal architecture, the `BlazeModule` class that wraps it for developer consumption, the `adapt()` method that is its primary entry point, the internal pipeline that converts a PyTorch tensor into a fully-configured `CBHandle`, and -- critically -- what existing infrastructure is left completely unchanged.

**What you will learn:**

- The three-layer architecture: TensorAdapter (PyTorch-facing) -> ShapeEngine (tiling/padding/format) -> CBBackend (Blaze-facing), and why this layering was chosen over alternatives (two layers or four)
- How `BlazeModule` extends `torch.nn.Module` to capture `forward()` calls and build `FusedProgram` invocations via TensorAdapter
- The `adapt()` method signature and its progressive-disclosure parameter design (Level 0: automatic, Level 1: configurable, Level 2: bypass)
- The six-step internal pipeline: `infer_shape -> select_tile -> compute_padding -> select_format -> allocate_cb -> return_handle`, with validation at each step
- A worked numerical example tracing `[1, 4096] x [4096, 11008]` through the complete pipeline
- The explicit list of components that are NOT changed: CBHandle, FusedProgram, CBEngine, SemEngine, CTArgEngine, the BlazeOp hierarchy, and the BlazeCompiler core

---

## Why Three Layers?

Chapter 2 established seven design principles from the framework survey. Two of them directly dictate TensorAdapter's internal structure:

- **P1 (Separate computation from execution strategy):** The developer's intent ("multiply these tensors") must be separable from the hardware strategy ("use 32x32 tiles, bfloat16, double-buffered CBs on a 4x8 core grid"). A single-class design would interleave both concerns, exactly replicating the problem that `emit()` has today.
- **P13 (Dialect layers from MLIR):** Progressive lowering through well-defined abstraction boundaries -- each layer adds hardware detail while preserving the information from the layer above. MLIR's `linalg -> memref -> gpu` pipeline maps directly to TensorAdapter's `Tensor -> Tile -> CB` pipeline.

The alternative considered -- a monolithic `TensorAdapter` class that directly converts a PyTorch tensor into a `CBHandle` -- was rejected because it collapses three distinct concerns:

1. **Shape reasoning** (what tile shape? what padding? what are the logical vs. padded vs. tiled dimensions?)
2. **Format and precision reasoning** (what data format? what math fidelity? where do we need fp32 accumulation?)
3. **CB construction** (how many pages? what page size? which `FusedProgram` method to call? how to wire CT args?)

Each of these concerns has different inputs, different validation rules, and different override mechanisms. Collapsing them prevents a developer from overriding tile shape without also reasoning about CB page counts.

### The Three Layers

```
Developer Code                TensorAdapter Layers              Existing Blaze
--------------                ----------------------              ---------------

                              LAYER 1: TensorAdapter
                              (PyTorch-facing)
                              - Accepts torch.Tensor + op_hint
y = blaze_matmul(             - Applies progressive disclosure
    activation,               - Routes to ShapeEngine
    weights                   - Returns TypedHandle to caller
)                                     |
                                      | ShapeDescriptor
                                      v
                              LAYER 2: ShapeEngine
                              (Tiling/Padding/Format)
                              - Computes logical -> padded -> tiled shapes
                              - Selects tile geometry per op
                              - Determines padding strategy + fill value
                              - Negotiates data format across graph edges
                              - Validates at each step (P3)
                                      |
                                      | Validated ShapeDescriptor
                                      |   + format + padding policy
                                      v
                              LAYER 3: CBBackend
                              (Blaze-facing)
                              - Computes num_pages, page_size
                              - Calls FusedProgram.cb_from_tensor()     --> FusedProgram
                              - Calls FusedProgram.cb_scratch()         --> CBHandle
                              - Derives CT arg values                   --> CTArgEngine
                              - Validates L1 budget                     --> l1_profile
                              - Returns CBHandle + ShapeDescriptor
                                      |
                                      v
                              TypedHandle (CBHandle + ShapeDescriptor)
```

Each layer maps to a conceptual transformation (drawn from V3's "Why Three Layers" rationale):

| Layer | Transformation | Input | Output |
|-------|---------------|-------|--------|
| TensorAdapter | Logical -> Analyzed | `torch.Tensor` + `op_hint` | ShapeDescriptor (logical shape + dtype) |
| ShapeEngine | Analyzed -> Sized | ShapeDescriptor (logical) | ShapeDescriptor (padded + tiled + formatted) |
| CBBackend | Sized -> Allocated | Validated ShapeDescriptor | CBHandle (hardware CB reference) |

### Rationale: Why Not Two Layers? Why Not Four?

**Two layers (Tensor -> CB)** would merge shape reasoning with format reasoning. This fails when the same shape analysis must produce different CB configurations under different precision profiles (e.g., a `[1, 4096]` activation tiled identically for "performance" mode with `bfloat8_b` and "accuracy" mode with `float32` -- same tile grid, different page sizes). Separating ShapeEngine from CBBackend allows the same `ShapeDescriptor` to be materialized into different CB configurations.

**Four layers** (adding a separate PaddingEngine between ShapeEngine and CBBackend) was considered but rejected because padding and tile shape selection are tightly coupled -- the padding strategy affects the padded shape, which affects the tile grid, which feeds back into padding. Forcing these into separate layers would require multiple round-trips between layers. Instead, ShapeEngine handles both tiling and padding as a single coherent pass (matching XLA's combined pad-and-tile approach from [Chapter 2, File 03](../ch02_design_patterns/03_triton_and_xla_patterns.md)).

---

## The BlazeModule Class

`BlazeModule` is the top-level developer-facing object. It extends `torch.nn.Module` so it participates naturally in PyTorch's module composition, serialization, and profiling. Its design draws directly from two patterns identified in Chapter 2:

- **DP1 (Module replacement from TT-Symbiote):** Like Symbiote's `TTNNModule`, `BlazeModule` wraps a PyTorch module and provides a hardware-accelerated `forward()`. Unlike Symbiote, it operates at the Blaze/LLK level (fused kernels on L1), not the TTNN op level.
- **DP4 (Four-phase lifecycle from TT-Symbiote):** `BlazeModule` follows the same analyze -> validate -> materialize -> execute phase ordering, with boolean guards to prevent redundant work.

```python
# PROPOSED (illustrative, not prescriptive) -- BlazeModule class structure
class BlazeModule(torch.nn.Module):
    """A PyTorch module that executes on Tenstorrent hardware via TT-Blaze.

    Wraps a standard nn.Module and handles all tile decomposition, CB allocation,
    padding, format selection, and L1 budgeting automatically.
    """

    def __init__(
        self,
        adapter: TensorAdapter,
        compiled_program: MeshCompiledProgram,
        weight_tensors: dict[str, ttnn.Tensor],
        shape_descriptors: dict[str, ShapeDescriptor],
    ):
        super().__init__()
        self._adapter = adapter
        self._compiled = compiled_program
        self._weights = weight_tensors
        self._shapes = shape_descriptors
        self._weights_on_device = False

    @classmethod
    def from_torch(
        cls,
        module: torch.nn.Module,
        sample_input: torch.Tensor,
        device,
        *,
        precision: str = "balanced",   # "performance" | "balanced" | "accuracy"
        format_hints: dict | None = None,
        grid: Any | None = None,
    ) -> "BlazeModule":
        """Convert a PyTorch module to a BlazeModule.

        Args:
            module: Standard PyTorch module (e.g., nn.Linear, nn.LayerNorm).
            sample_input: Representative input for shape inference.
            device: Tenstorrent mesh device.
            precision: Precision profile (see Chapter 6).
            format_hints: Per-parameter format overrides.
            grid: Core grid override (default: auto-select).
        """
        adapter = TensorAdapter(device=device, precision=precision)
        # Phase 1: Analyze -- trace module, infer shapes, select formats
        graph, shapes = adapter.analyze(module, sample_input)
        # Phase 2: Validate -- check L1 budget, format compatibility
        adapter.validate(graph, shapes)
        # Phase 3: Materialize -- preprocess weights, compile program
        weights = adapter.preprocess_weights(module, shapes, format_hints)
        compiled = adapter.compile(graph, weights, shapes, grid=grid)
        return cls(adapter, compiled, weights, shapes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Execute on Tenstorrent hardware. No manual CB/tile/format decisions."""
        if not self._weights_on_device:
            self._adapter.move_weights_to_device(self._weights)
            self._weights_on_device = True
        act = self._adapter.prepare_activation(input_tensor, self._shapes["input"])
        result = self._compiled.run()
        return self._adapter.extract_output(result, self._shapes["output"])
```

### Design Decisions in BlazeModule

**Why extend `torch.nn.Module`?** Unlike Symbiote's `TTNNModule`, which deliberately avoids subclassing `nn.Module` to prevent autograd interference, `BlazeModule` subclasses it because:

1. Module composition: a `BlazeModule` can be a child of a larger PyTorch model tree (e.g., `model.attention.q_proj` is a `BlazeModule` while `model.lm_head` remains on CPU).
2. `torch.save` / `torch.load` compatibility for checkpointing.
3. Hook integration: `register_forward_hook`, profiling, and debugging tools work unchanged.

The tradeoff is that autograd sees `BlazeModule` as a regular module. Since TT-Blaze targets inference (training is out of scope per [Chapter 1, File 04](../ch01_the_gap/04_abstraction_goals_and_scope.md)), this is acceptable.

**Why `from_torch()` as a classmethod?** The three-phase lifecycle (analyze -> validate -> materialize) must execute in order, and the constructor should receive already-validated objects. A classmethod factory enforces this ordering -- the developer cannot construct a `BlazeModule` from unvalidated shapes. This mirrors Symbiote's `TTNNModule.from_torch(old_module)` pattern (DP4).

---

## The adapt() Method: Progressive Disclosure in Action

The `adapt()` method is TensorAdapter's primary entry point. Its signature implements the three-tier control hierarchy established in [Chapter 2, File 05](../ch02_design_patterns/05_synthesized_design_principles.md) (Principle P4: Defaults with Per-Decision Overrides):

```python
# PROPOSED (illustrative, not prescriptive) -- adapt() method signature
class TensorAdapter:
    def adapt(
        self,
        tensor: torch.Tensor | ttnn.Tensor | CBHandle,
        *,
        op_hint: str | None = None,         # Level 0: automatic
        # --- Level 1: configurable (override specific decisions) ---
        tile_shape: tuple[int, int] | None = None,
        padding: PaddingPolicy | None = None,
        format: Any | None = None,           # ttnn.DataType
        num_pages: int | None = None,
        # --- Level 2: bypass (rare, expert use) ---
        raw_cb_params: dict | None = None,
    ) -> TypedHandle:
        """Convert a tensor into a CBHandle with attached shape metadata.

        Progressive disclosure:
          - Level 0: provide only tensor + op_hint. Everything automatic.
          - Level 1: override specific decisions (tile, padding, format, pages).
          - Level 2: provide raw_cb_params dict to bypass all inference.

        Args:
            tensor: PyTorch tensor, TTNN tensor, or existing CBHandle.
            op_hint: Operation context (e.g., "matmul.in0", "rmsnorm.input").
                Used to select tile shape, padding strategy, and format.
            tile_shape: Override tile geometry (default: inferred from op_hint).
            padding: Override padding strategy (default: from op registry).
            format: Override data format (default: from precision profile).
            num_pages: Override page count (default: from shape + blocking).
            raw_cb_params: Bypass all inference; pass dict directly to
                FusedProgram.cb_scratch(). Expert escape hatch.

        Returns:
            TypedHandle wrapping CBHandle + ShapeDescriptor.
        """
        ...
```

### Why This Signature?

The parameter design reflects the three-level developer journey from [Chapter 1, File 04](../ch01_the_gap/04_abstraction_goals_and_scope.md):

| Level | Developer | What They Provide | What the System Does |
|-------|-----------|-------------------|---------------------|
| **0: Automatic** | New to Tenstorrent | `adapt(tensor, op_hint="matmul.in0")` | Everything: tile shape, padding, format, page count, L1 budget |
| **1: Configurable** | Tuning performance | `adapt(tensor, op_hint="matmul.in0", tile_shape=(1, 32))` | Everything except the overridden parameter |
| **2: Bypass** | Expert, critical path | `adapt(tensor, raw_cb_params={...})` | Nothing -- developer takes full control |

Critically, overriding one parameter (e.g., `tile_shape`) does not require overriding all others. The system still infers padding, format, and page count from the overridden tile shape. This is the "orthogonal override" principle from Triton's `@autotune` (DP10 from [Chapter 2, File 03](../ch02_design_patterns/03_triton_and_xla_patterns.md)).

### The op_hint Parameter

The `op_hint` string encodes two pieces of information: the **operation type** and the **port role**:

| op_hint | Operation | Port | Effect |
|---------|-----------|------|--------|
| `"matmul.in0"` | Matmul | Activation input | 32x32 tiles, bfloat16, FIFO mode, zero-padding |
| `"matmul.in1"` | Matmul | Weight input | 32x32 tiles, bfloat8_b, DIRECT_ADDRESS mode, zero-padding |
| `"rmsnorm.input"` | RMSNorm | Input | 1x32 tiles, bfloat16, FIFO mode, zero-padding |
| `"softmax.input"` | Softmax | Input | 32x32 tiles, bfloat16, FIFO mode, neg-inf padding |

The alternative -- passing the operation object itself -- was considered but rejected because (a) it creates a circular dependency (the op needs the handle, but the handle needs the op to configure itself), and (b) string hints are serializable and debuggable. The `op_hint` is parsed into `(op_type, port_name)` internally, and the op registry provides the defaults.

---

## The Internal Pipeline

When `adapt()` is called with only a tensor and an `op_hint`, the following six-step pipeline executes:

```
     Step 1              Step 2              Step 3
   infer_shape()      select_tile()      compute_padding()
   +-----------+      +-----------+      +---------------+
   | Extract   |      | Choose    |      | Look up op's  |
   | logical   | ---> | tile_h,   | ---> | padding       |
   | shape     |      | tile_w    |      | strategy      |
   | + dtype   |      | from op   |      | + fill value  |
   | from      |      | hint +    |      | Compute       |
   | tensor    |      | shape     |      | padded_shape  |
   +-----------+      +-----------+      +---------------+
                                                |
                                                v
     Step 6              Step 5              Step 4
   return_handle()    allocate_cb()      select_format()
   +-----------+      +-----------+      +---------------+
   | Wrap      |      | Compute   |      | Apply         |
   | CBHandle  | <--- | num_pages | <--- | precision     |
   | + Shape   |      | page_size |      | profile       |
   | Descriptor|      | Call      |      | + per-op      |
   | into      |      | FusedProg |      | format prefs  |
   | TypedHndl |      | method    |      | Negotiate     |
   +-----------+      +-----------+      +---------------+
```

Each step is a separate method on the layer that owns it, enabling independent testing and override:

### Step 1: infer_shape() -- ShapeEngine

```python
# PROPOSED (illustrative, not prescriptive) -- ShapeEngine.infer_shape()
def infer_shape(self, tensor) -> ShapeDescriptor:
    """Extract logical shape and dtype from a PyTorch or TTNN tensor.

    Handles three input types:
    - torch.Tensor: read .shape and .dtype directly
    - ttnn.Tensor: read .shape and .dtype, accounting for tile layout
    - CBHandle: read from existing handle metadata (passthrough case)
    """
    if isinstance(tensor, torch.Tensor):
        return ShapeDescriptor(
            logical_shape=tuple(tensor.shape),
            dtype=torch_dtype_to_tt(tensor.dtype),
        )
    elif isinstance(tensor, CBHandle):
        # Passthrough: adapt() was called on an existing handle
        return ShapeDescriptor.from_cb_handle(tensor)
    ...
```

### Step 2: select_tile() -- ShapeEngine

```python
# PROPOSED (illustrative, not prescriptive) -- ShapeEngine.select_tile()
def select_tile(self, desc: ShapeDescriptor, op_hint: str) -> ShapeDescriptor:
    """Select tile geometry based on operation type and tensor shape.

    Rules (derived from existing Blaze ops):
    - matmul: (32, 32) for both in0 and in1
    - rmsnorm, layernorm: (1, 32) for row-vector processing
    - elementwise (add, mul, silu): match input tile shape
    - softmax/sdpa: (32, 32) for attention computations

    These match the patterns in blaze/utils.py (interpret_tile).
    """
    op_type, port = _parse_op_hint(op_hint)
    tile_shape = TILE_REGISTRY.get((op_type, port), DEFAULT_TILE_SHAPE)
    return desc.with_tile(tile_shape)
```

### Step 3: compute_padding() -- ShapeEngine

```python
# PROPOSED (illustrative, not prescriptive) -- ShapeEngine.compute_padding()
def compute_padding(self, desc: ShapeDescriptor, op_hint: str) -> ShapeDescriptor:
    """Compute padded shape and padding fill value.

    padded_H = ceil(logical_H / tile_H) * tile_H
    padded_W = ceil(logical_W / tile_W) * tile_W

    Fill value is op-specific (P6: per-operation padding semantics):
    - matmul: 0.0 (zero contributions to dot product)
    - softmax: -inf (zero probability after exp)
    - rmsnorm: 0.0 (no contribution to variance)
    """
    ...
```

### Step 4: select_format() -- ShapeEngine

Applies the precision profile and per-op format preferences. See [Chapter 6: Data Formats](../ch06_data_formats/01_data_format_landscape.md).

### Step 5: allocate_cb() -- CBBackend

```python
# PROPOSED (illustrative, not prescriptive) -- CBBackend.allocate_cb()
def allocate_cb(
    self, f: FusedProgram, desc: ShapeDescriptor, op_hint: str
) -> CBHandle:
    """Translate a validated ShapeDescriptor into a CBHandle.

    Calls the appropriate FusedProgram method:
    - f.cb_from_tensor() for tensor-backed CBs (external inputs/outputs)
    - f.cb_scratch() for intermediate CBs (inter-op data flow)

    Computes:
    - page_size: from tile_shape + data_format (see Warning below)
    - num_pages: determined by op blocking requirements
      (e.g., k_num_tiles for matmul in0, out_w_per_core for matmul out)
    """
    ...
```

> **Warning:** For standard formats (bfloat16, float32), `page_size = tile_h * tile_w * DTYPE_BYTES[format]`. However, for block floating-point formats (bfloat8_b, bfloat4_b), the page size includes shared exponent overhead and **cannot** be computed by simple multiplication. A bfloat8_b tile of 32x32 is approximately 1,088 bytes, not 1,024. Always use `tile.get_tile_size(data_format)` for packed formats. See [Chapter 1, File 02](../ch01_the_gap/02_tenstorrent_tile_cb_model.md) for details on the BFP tile layout.

### Step 6: return_handle() -- TensorAdapter

Wraps the `CBHandle` and `ShapeDescriptor` into a `TypedHandle` and returns it.

### Validation at Each Step (P3)

Following Principle P3 (validate at each lowering step), each pipeline step validates its output before passing to the next:

| Step | Validation | Error Raised |
|------|-----------|--------------|
| infer_shape | Dimensions positive, rank >= 1 | `ValueError("Cannot adapt scalar tensor")` |
| select_tile | op_hint recognized, tile shape is valid hardware geometry | `ValueError("Unknown op_hint: ...")` |
| compute_padding | Padded shape >= logical shape, fill value is finite (except -inf for softmax) | `ValueError("Padding produced smaller shape")` |
| select_format | Format supported by op, format-tile combination valid | `ValueError("bfloat4_b requires 32x32 tiles")` |
| allocate_cb | Total L1 per core <= budget, CB count < 64 | `L1BudgetError("Exceeds 1.5MB per core")` |

> **Warning:** In the current Blaze infrastructure, none of these validations exist at compile time. L1 overflow is detected only at runtime via silent data corruption or hardware hangs. The validation chain is one of TensorAdapter's primary value propositions -- it catches configuration errors before they reach the hardware. This directly addresses Architectural Invariant #3 (64-CB hard limit) and the L1 budget problem identified in [Chapter 2, File 05](../ch02_design_patterns/05_synthesized_design_principles.md).

### Error Pipeline Examples

When invalid inputs reach the pipeline, validation catches them early (cherry-picked from V4's error handling tables):

| Scenario | Fails At | Error | Recovery |
|----------|----------|-------|----------|
| `adapt(torch.tensor(3.14))` | Step 1 (infer_shape) | `ValueError("Cannot adapt scalar tensor; reshape to [1,1]")` | Reshape before calling adapt() |
| `adapt(torch.randn(0, 4096))` | Step 1 (infer_shape) | `ValueError("Zero-size dimension at index 0")` | Fix input shape |
| `adapt(tensor, op_hint="unknown_op.in0")` | Step 2 (select_tile) | `ValueError("Unknown op_hint: 'unknown_op'")` | Use a registered op hint or provide tile_shape override |
| `adapt(tensor, op_hint="matmul.in0", format="bfloat4_b", tile_shape=(1,32))` | Step 4 (select_format) | `ValueError("bfloat4_b requires 32x32 tiles, got (1,32)")` | Use (32,32) tiles or choose bfloat16 |
| Cumulative L1 > 1.5MB per core | Step 5 (allocate_cb) | `L1BudgetError` with per-CB breakdown | Reduce blocking factor, use smaller format, or split across more cores |

---

## Worked Example: [1, 4096] x [4096, 11008] Matmul

The following traces a complete matmul through TensorAdapter, from PyTorch tensors to hardware execution. This is the same nn.Linear used in [Chapter 1, File 03](../ch01_the_gap/03_the_manual_plumbing_burden.md), now handled automatically.

```
Developer writes:
    blaze_linear = BlazeModule.from_torch(nn.Linear(4096, 11008), ...)

1. adapter.adapt(weight_tensor, op_hint="matmul.in1")
   |
   +-> ShapeEngine:
   |     infer_shape:    logical = (4096, 11008)
   |     select_tile:    tile = (32, 32)        [matmul -> 32x32]
   |     compute_padding: padded = (4096, 11008) [already aligned: 4096/32=128, 11008/32=344]
   |     select_format:  bfloat8_b              [weight precision profile]
   |     -> ShapeDescriptor(logical=(4096, 11008), padded=(4096, 11008),
   |                        tile=(32,32), grid=(128, 344), total_tiles=44032,
   |                        page_size=1088, data_format="bfloat8_b")
   |
   +-> CBBackend.allocate():
   |     FusedProgram.cb_from_tensor(weight_tensor)           # EXISTING call
   |     -> CBHandle(cb_id=0, num_pages=344, page_size=1088,
   |                 access_mode=DIRECT_ADDRESS, ...)
   |
   +-> TypedHandle(handle=CBHandle, shape=ShapeDescriptor)

2. adapter.adapt(activation_tensor, op_hint="matmul.in0")
   |
   +-> ShapeEngine:
   |     infer_shape:    logical = (1, 4096)
   |     select_tile:    tile = (32, 32)
   |     compute_padding: padded = (32, 4096)    [1 -> 32 rows: ceil(1/32)*32 = 32]
   |     select_format:  bfloat16                [activation format]
   |     -> ShapeDescriptor(logical=(1, 4096), padded=(32, 4096),
   |                        tile=(32,32), grid=(1, 128), total_tiles=128,
   |                        page_size=2048, data_format="bfloat16")
   |
   +-> CBBackend.allocate():
   |     FusedProgram.cb_from_tensor(act_tensor)              # EXISTING call
   |     -> CBHandle(cb_id=1, num_pages=128, page_size=2048, ...)
   |
   +-> TypedHandle(handle=CBHandle, shape=ShapeDescriptor)

3. CBBackend.derive_ct_args():
       k_num_tiles = act_desc.tile_grid_shape[-1] = 128
       out_w_per_core = wt_desc.total_tiles / k_num_tiles = 44032 / 128 = 344
       -> FusedProgram.ncrisc_ct_args(...)                    # EXISTING call
       -> FusedProgram.trisc_ct_args(...)                     # EXISTING call

4. Matmul.emit(f, in0_handle, in1_handle, prefix="matmul")   # EXISTING call

5. FusedProgram.build() -> CompiledProgram                    # EXISTING call

6. CompiledProgram.run() -> output_tensor                     # EXISTING call
```

Note: the weight `page_size` is 1,088 bytes (not 1,024) because bfloat8_b uses block floating-point format with shared exponent bytes per 32x32 tile. This is computed via `tile.get_tile_size(ttnn.bfloat8_b)`, not by `32 * 32 * 1`.

### The CBParams Intermediate Type

Between ShapeEngine's output and CBBackend's CB allocation, the system uses an internal boundary type `CBParams` that bundles the parameters needed for a single CB allocation call. This separates shape-world concerns from CB-world concerns:

```python
# PROPOSED (illustrative, not prescriptive) -- CBParams (internal boundary type)
@dataclass
class CBParams:
    """Parameters for a single CB allocation.

    Computed from ShapeDescriptor by CBBackend, used to call
    FusedProgram.cb_from_tensor() or cb_scratch().
    """
    num_pages: int
    page_size: int
    data_format: object          # ttnn.DataType
    tile_desc: object            # ttnn.TileDescriptor
    core_ranges: object          # ttnn.CoreRangeSet
    is_tensor_backed: bool
    access_mode: CBAccessMode = CBAccessMode.FIFO
    scratch_name: str = ""

    @staticmethod
    def from_descriptor(desc: ShapeDescriptor, **overrides) -> "CBParams":
        """Derive CB allocation parameters from a ShapeDescriptor.

        For standard formats, page_size = tile_h * tile_w * dtype_bytes.
        For BFP formats (bfloat8_b, bfloat4_b), page_size must be
        obtained via tile.get_tile_size(data_format) to account for
        shared exponent overhead.
        """
        ...
```

`CBParams` is not exposed to the developer. It lives inside CBBackend as a clean handoff point between "what shape do I have?" (ShapeDescriptor) and "what FusedProgram call do I make?" (CBParams). This is unique to this design and adds a clean separation boundary.

---

## What Is NOT Changed

The compose-not-replace constraint from [Chapter 1, File 04](../ch01_the_gap/04_abstraction_goals_and_scope.md) is a hard architectural requirement. TensorAdapter is a **builder** that constructs the same `FusedProgram` calls a human developer would write. The following components are unchanged:

### CBHandle (blaze/cb_handle.py) -- UNCHANGED

The `CBHandle` dataclass remains exactly as defined: 12 fields (`cb_id`, `num_pages`, `page_size`, `core_ranges`, `data_format`, `tile_desc`, `byte_offset`, `backing_tensor`, `access_mode`, `_node_id`, `_port_name`, `scratch_name`), `eq=False`, `buffer_address()` method, and `require_fifo_handle()` utility. TensorAdapter creates `CBHandle` instances with computed parameters. The `_node_id` and `_port_name` fields for shadow graph recording are maintained by the CBBackend layer, preserving Architectural Invariant #6 (graph identity).

### FusedProgram (blaze/fused_program.py) -- UNCHANGED

FusedProgram gains **no new methods**. TensorAdapter calls the existing API: `cb_from_tensor()`, `cb_scratch()`, `cb_alias()`, `ncrisc_ct_args()`, `trisc_ct_args()`, `per_core_unified_ct_args()`, `output()`, `wire_output()`, `build()`, and `reconfig()`. The internal machinery -- disjoint-core CB sharing (`_format_key` dedup), scratch-mapped allocation, shadow graph recording (`_shadow_graph`), and `OverlappedView` management -- all operate exactly as today.

### CBEngine (blaze/cb_engine.py) -- UNCHANGED

The `CBEngine.assign()` algorithm -- topological walk, sequential CB ID allocation, fan-out handling via `_union_grids()`, shared tensor CB reuse, and `compact_cb_ids()` interval coloring -- is unchanged. TensorAdapter feeds pre-configured `TensorPort` metadata into the graph. The engine's 64-CB limit enforcement remains as-is.

### SemEngine (blaze/sem_engine.py) -- UNCHANGED

Semaphore assignment depends on inter-core topology (gather, mcast, CCL patterns), not tensor shapes. The `SemEngine` identifies `SyncPoint` objects from inter-core op schemas and assigns sequential global semaphore indices. TensorAdapter has no influence on this process.

### CTArgEngine (blaze/ct_args.py) -- UNCHANGED

`CTArgEngine.generate_tuples()` walks the graph, resolves values from CB assignments, semaphore assignments, grid context, and user overrides. TensorAdapter's role is to **compute the user_args** that CTArgEngine consumes -- the engine's resolution logic, prefix computation, and RISC grouping are all unchanged.

### BlazeOp Hierarchy (blaze/blaze_op.py) -- UNCHANGED

The `BlazeOp` base class, `Input`/`Output`/`Internal` port descriptors, `CompileTimeArg` declarations, `emit()`/`compose()`/`register()` methods, and the `FusedOpConfig` registry all remain as-is. TensorAdapter is an `emit()` **caller**, not an `emit()` replacer.

### BlazeCompiler (blaze/compiler.py) -- UNCHANGED

`BlazeCompiler.compile()` continues to accept a `BlazeGraph`, a tensor dict, an output tensor, and `user_args`. TensorAdapter pre-computes the `user_args` that would otherwise require manual calculation. The compiler's internal logic is not modified.

---

## The Adapter as emit() Caller

This is the conceptual model that ties everything together. TensorAdapter's primary job is to compute the arguments that `emit()` needs, then delegate to `emit()`:

```python
# PROPOSED (illustrative, not prescriptive) -- TensorAdapter calling Matmul.emit()
def _emit_matmul(self, f: FusedProgram, activation, weights, prefix="matmul"):
    # Step 1-4: Shape analysis via ShapeEngine
    act_desc = self.shape_engine.full_analysis(activation, "matmul.in0")
    wt_desc = self.shape_engine.full_analysis(weights, "matmul.in1")

    # Step 5: CB allocation via CBBackend
    in0_handle = self.cb_backend.allocate(f, act_desc)
    in1_handle = self.cb_backend.allocate_direct_address(f, wt_desc)

    # Step 6: Derive CT args from shape metadata
    k_num_tiles = act_desc.tile_grid_shape[-1]            # K tiles from activation width
    out_w_per_core = wt_desc.total_tiles // k_num_tiles

    # Delegate to existing emit() -- identical call to hand-written code
    out_handle = Matmul.emit(f, in0_handle, in1_handle, prefix=prefix)

    # Wrap with shape metadata for downstream ops
    out_desc = self.shape_engine.infer_matmul_output(act_desc, wt_desc)
    return TypedHandle(handle=out_handle, shape=out_desc)
```

Compare this with the manual approach from [Chapter 1, File 03](../ch01_the_gap/03_the_manual_plumbing_burden.md): the developer no longer makes the 12 manual decisions. But the downstream call -- `Matmul.emit(f, in0_handle, in1_handle, prefix=prefix)` -- is **identical**. The kernel binary produced is identical. The only difference is who computed the parameters.

### Compile-Time Only: Zero Runtime Overhead

TensorAdapter is a **compile-time abstraction**. All shape analysis, tile selection, padding computation, format negotiation, CB allocation, and CT arg derivation happen during graph construction and compilation. The `CompiledProgram` object that executes on hardware is byte-identical to one produced by hand-written code. There is no adapter in the execution path.

---

## Forward References

The three layers of TensorAdapter are elaborated in subsequent chapters:

| Layer | Component | Detailed In |
|-------|-----------|-------------|
| ShapeEngine | Tile decomposition algorithm | [Chapter 4: Tile Decomposition](../ch04_tile_decomposition/01_three_shape_layers.md) |
| ShapeEngine | Padding strategy registry | [Chapter 5: Padding](../ch05_padding/01_padding_strategy_taxonomy.md) |
| ShapeEngine | Format negotiation | [Chapter 6: Data Formats](../ch06_data_formats/01_data_format_landscape.md) |
| CBBackend | CB sizing and L1 budgeting | [Chapter 7: CB Sizing](../ch07_cb_sizing/01_l1_memory_model.md) |
| TensorAdapter | Broadcasting detection | [Chapter 8: Broadcasting](../ch08_broadcasting/01_pytorch_broadcasting_rules.md) |
| TensorAdapter | Op fusion support | [Chapter 9: Op Fusion](../ch09_op_fusion/01_blaze_fusion_model.md) |
| BlazeModule | End-to-end API and escape hatches | [Chapter 10: End-to-End](../ch10_end_to_end/01_the_blaze_module_api.md) |

---

## Key Takeaways

- TensorAdapter uses a **three-layer architecture** (TensorAdapter -> ShapeEngine -> CBBackend) because each layer addresses a distinct concern: developer interface, shape/format reasoning, and CB construction. This follows MLIR's dialect layering principle (P13) and avoids the monolithic design that would reproduce the problems of today's `emit()` methods.
- `BlazeModule` extends `torch.nn.Module` with a `from_torch()` factory that enforces the analyze -> validate -> materialize lifecycle ordering, drawing from Symbiote's `TTNNModule` lifecycle pattern (DP4).
- The `adapt()` method implements **progressive disclosure**: Level 0 (automatic) requires only a tensor and an `op_hint`; Level 1 (configurable) allows overriding individual decisions; Level 2 (bypass) passes raw parameters to `FusedProgram`. Overriding one decision does not require overriding all others.
- The six-step internal pipeline validates at each step, catching configuration errors before they reach hardware -- a fundamental improvement over the current runtime-corruption failure mode.
- A **worked example** traces a [1, 4096] x [4096, 11008] matmul through the complete pipeline: weight ShapeDescriptor with bfloat8_b (1,088 bytes/page), activation ShapeDescriptor with bfloat16 (2,048 bytes/page), automatic CT arg derivation (k_num_tiles=128, out_w_per_core=344), and delegation to the existing `Matmul.emit()`.
- **Nothing existing is replaced:** CBHandle, FusedProgram, CBEngine, SemEngine, CTArgEngine, the BlazeOp hierarchy, and BlazeCompiler all remain unchanged. TensorAdapter is an `emit()` caller that computes parameters automatically.

## Source Files

- `blaze/cb_handle.py` -- CBHandle dataclass (unchanged, wrapped by TensorAdapter)
- `blaze/fused_program.py` -- FusedProgram (unchanged, called by CBBackend layer)
- `blaze/cb_engine.py` -- CBEngine (unchanged, consumes TensorAdapter-enriched graph)
- `blaze/sem_engine.py` -- SemEngine (unchanged, shape-independent)
- `blaze/ct_args.py` -- CTArgEngine (unchanged, consumes TensorAdapter-computed user_args)
- `blaze/blaze_op.py` -- BlazeOp, MicroOp, FusedOp hierarchy (unchanged, emit() API preserved)
- `blaze/compiler.py` -- BlazeCompiler (unchanged, compile() API preserved)
- `blaze/context.py` -- FusionContext, FusionResult, ExternalTensor (extended in File 04)

---

[Previous: Chapter 2, File 05 -- Synthesized Design Principles](../ch02_design_patterns/05_synthesized_design_principles.md) | [Next: 02 -- Shape Metadata System](./02_shape_metadata_system.md)
