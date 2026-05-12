# The BlazeModule API: From PyTorch Module to Hardware Execution

This is the final chapter of the guide. It brings together every subsystem -- TensorAdapter, ShapeEngine, CBBackend, format negotiation, padding, CB sizing, broadcasting, and op fusion -- into a single developer-facing API that converts a standard PyTorch module into a hardware-accelerated program running on Tenstorrent silicon. The chapter presents the complete API, demonstrates it with worked examples, quantifies where automation may sacrifice performance, and documents the escape hatches available for power users.

**What you will learn:**

- The top-level entry point `blaze.from_pytorch()` and its relationship to `BlazeModule`
- The `BlazeModule.forward()` execution path and the four-phase lifecycle
- The under-the-hood pipeline: trace, BlazeGraph, engine passes, compile, execute
- Configuration knobs: `precision`, `format_hints`, `grid`, `overrides`, `fusion`
- Weight handling: preprocessing, format conversion, device placement, and lazy movement
- `BlazeModule` vs. `TTNNModule`: when to use each and how they interoperate
- How all seven design principles (P1-P7) govern the API and pipeline design

---

## The Top-Level Entry Point: blaze.from_pytorch()

The primary developer-facing function is `blaze.from_pytorch()`. It accepts a standard `torch.nn.Module`, a representative sample input, and a device handle, returning a `BlazeModule` that executes on Tenstorrent hardware:

```python
# PROPOSED (illustrative, not prescriptive) -- Top-level API
import blaze

model = torch.nn.Linear(4096, 11008)
blaze_model = blaze.from_pytorch(
    model,
    sample_input=torch.randn(1, 4096),
    device=mesh_device,
)
output = blaze_model(input_tensor)
```

Three lines of developer code replace the approximately 200 lines of manual CB configuration, tile decomposition, padding, format selection, CT arg computation, and L1 budgeting that the same operation requires today. The developer provides the tensors and the operation; the system derives everything else.

### The Function Signature

```python
# PROPOSED (illustrative, not prescriptive) -- blaze.from_pytorch() signature
def from_pytorch(
    module: torch.nn.Module,
    sample_input: torch.Tensor | tuple[torch.Tensor, ...],
    device,
    *,
    precision: str = "balanced",
    format_hints: dict[str, Any] | None = None,
    grid: Any | None = None,
    overrides: dict[str, dict] | None = None,
    fusion: bool = True,
    validate_l1: bool = True,
) -> BlazeModule:
    """Convert a PyTorch module to a BlazeModule for hardware execution.

    Args:
        module: Standard PyTorch module (nn.Linear, nn.LayerNorm, or custom).
        sample_input: Representative input tensor for shape inference. The
            system traces the module with this input to capture the computation
            graph. The batch dimension may vary at inference time if the module
            supports it.
        device: Tenstorrent mesh device (from ttnn.open_device or MeshDevice).
        precision: Precision profile name. One of:
            "performance" -- bfloat8_b weights, LoFi math, math_approx (Ch6)
            "balanced"    -- bfloat16 weights, HiFi4, exact math (default)
            "accuracy"    -- bfloat16 weights, HiFi4, fp32 accumulation
        format_hints: Per-parameter format overrides. Keys are parameter names
            (e.g., "weight", "gate_proj.weight"), values are ttnn.DataType.
        grid: Core grid override. Default: auto-select from device and weight
            shape (see Ch7).
        overrides: Per-op settings. Keys are op names or hints (e.g., "sdpa",
            "lm_head"), values are dicts of settings:
            {"math_fidelity": ..., "fp32_dest_acc_en": ..., "num_pages": ...}
        fusion: Whether to enable op fusion (default True). When False, each
            PyTorch op is compiled as a separate kernel dispatch.
        validate_l1: Whether to validate L1 budget at compile time (Ch7).

    Returns:
        BlazeModule wrapping the compiled hardware program.

    Raises:
        L1BudgetError: If validate_l1=True and the compiled program
            exceeds per-core L1 capacity (~1.5 MB).
        ValueError: If sample_input shape is incompatible with module.
    """
    adapter = TensorAdapter(device=device, precision=precision)
    graph, shapes = adapter.analyze(module, sample_input, fusion=fusion)
    adapter.validate(graph, shapes)
    weights = adapter.preprocess_weights(module, shapes, format_hints)
    compiled = adapter.compile(graph, weights, shapes, grid=grid,
                                overrides=overrides)
    return BlazeModule(adapter, compiled, weights, shapes)
```

### Parameter Design Rationale

Each parameter follows a specific design principle from the seven principles established in [Chapter 2, File 05](../ch02_design_patterns/05_synthesized_design_principles.md):

| Parameter | Design Principle | Rationale |
|-----------|-----------------|-----------|
| `module` | P1: Separate Computation Intent from Execution Strategy | The developer expresses intent through a standard PyTorch module; execution strategy is derived |
| `sample_input` | P2: Carry Metadata from Both Worlds Simultaneously | Provides the logical shape that grounds all downstream decisions in both PyTorch and hardware worlds |
| `precision` | P4: Provide Defaults with Per-Decision Overrides | One string configures format, fidelity, and accumulation for the entire model |
| `format_hints` | P4: Provide Defaults with Per-Decision Overrides | Surgical override for specific weight formats without affecting the rest |
| `grid` | P4: Provide Defaults with Per-Decision Overrides | Override auto-selected core grid for non-standard placements |
| `overrides` | P4: Provide Defaults with Per-Decision Overrides | Per-op settings that complement the global profile |
| `fusion` | P5: Operate on Graphs, Not Individual Tensors | Enables or disables graph-level fusion optimization |

---

## The Under-the-Hood Pipeline

When `blaze.from_pytorch()` is called, a six-stage pipeline executes. Each stage maps to a chapter of this guide and is governed by a specific design principle:

```text
Stage 1: TRACE                          (P7: Transparent Interception That Preserves Identity)
  Input:  torch.nn.Module + sample_input
  Action: Trace the module's forward() to capture the computation graph.
          Uses torch.fx or equivalent tracing mechanism.
  Output: Traced FX graph (PyTorch-level op sequence)
  Source: Chapter 2 (torch.compile patterns), Chapter 9 (fusion detection)

Stage 2: BUILD BLAZE GRAPH              (P5: Operate on Graphs, Not Individual Tensors)
  Input:  Traced FX graph
  Action: Map PyTorch ops to TT-Blaze ops via the op mapping registry
          (Chapter 9, File 04). Apply fusion analyzer to identify fusible
          subgraphs. Create BlazeGraph with ShapeDescriptor on every edge.
  Output: BlazeGraph with enriched edges (ShapeDescriptor, format, padding)
  Source: Chapter 3 (TensorAdapter architecture), Chapter 9 (fusion model)

Stage 3: ENGINE PASSES                  (P3: Validate at Each Lowering Step, Not Just at the End)
  Input:  BlazeGraph (enriched)
  Action: Run the three engine passes in sequence:
    3a. CBEngine.assign()   -- allocate CB IDs, enforce 64-slot limit
    3b. SemEngine.assign()  -- allocate semaphore indices
    3c. CTArgEngine.generate_tuples() -- resolve CT arg values from
        auto-derived user_args (Chapter 3, File 04)
  Output: Fully resolved BlazeGraph (CB IDs, semaphores, CT args assigned)
  Source: Chapter 7 (CB sizing), Chapter 3 (CT arg derivation)

Stage 4: COMPILE                        (P1: Separate Computation Intent from Execution Strategy)
  Input:  Resolved BlazeGraph + preprocessed weights + user overrides
  Action: BlazeCompiler.compile() produces a MeshCompiledProgram:
    4a. For each fused region: FusedProgram.build() -> kernel binary
    4b. For each standalone op: single-phase FusedProgram -> kernel binary
    4c. Link kernel binaries into a MeshCompiledProgram
  Output: MeshCompiledProgram (ready for hardware dispatch)
  Source: Chapter 3 (CBBackend.allocate()), Chapter 9 (fused composition)

Stage 5: WEIGHT PLACEMENT               (P2: Carry Metadata from Both Worlds Simultaneously)
  Input:  PyTorch state_dict + ShapeDescriptors + format_hints
  Action: Convert weights from PyTorch format to hardware format:
    5a. Transpose weights where needed (nn.Linear stores [out, in])
    5b. Apply format conversion (bfloat16 -> bfloat8_b per profile)
    5c. Pad to tile boundaries (Chapter 5)
    5d. Shard across cores (per weight ShapeDescriptor)
    5e. Move to device DRAM (lazy: deferred until first forward())
  Output: dict[str, ttnn.Tensor] on device
  Source: Chapter 4 (tile decomposition), Chapter 6 (format conversion)

Stage 6: EXECUTE                        (P7: Transparent Interception That Preserves Identity)
  Input:  MeshCompiledProgram + device tensors
  Action: BlazeModule.forward() dispatches the compiled program:
    6a. Prepare activation: adapt input tensor to hardware format
    6b. Dispatch compiled program to device
    6c. Extract output: unpad and convert result back to PyTorch tensor
  Output: torch.Tensor (logical shape, original dtype)
  Source: Chapter 5 (output unpadding), Chapter 6 (format extraction)
```

### Stage Timing Breakdown

| Stage | When It Runs | Typical Duration | Amortized Over |
|-------|-------------|-----------------|----------------|
| 1. Trace | `from_pytorch()` | 50-200 ms | Model lifetime |
| 2. Build Graph | `from_pytorch()` | 100-500 ms | Model lifetime |
| 3. Engine Passes | `from_pytorch()` | 10-50 ms | Model lifetime |
| 4. Compile | `from_pytorch()` | 200-2000 ms | Model lifetime |
| 5. Weight Placement | First `forward()` | 100-500 ms | Model lifetime |
| 6. Execute | Every `forward()` | 5-50 us | Per-inference |

Stages 1-4 execute once during `from_pytorch()`. Stage 5 executes on the first `forward()` call (lazy weight movement). Stage 6 executes on every subsequent `forward()` call. The compile-time cost (stages 1-4) is amortized over the model's inference lifetime; the per-inference cost (stage 6) is the only recurring overhead.

### Profile Cascade: From Intent to Hardware Parameters

A single `precision` string cascades through the entire pipeline, configuring dozens of low-level parameters. The cascade relationship (from [Chapter 6, File 04](../ch06_data_formats/04_precision_profiles_and_user_overrides.md)):

$$\text{profile} \xrightarrow{\text{format}} \text{page\_size} \xrightarrow{\text{sizing}} \text{num\_pages} \xrightarrow{\text{blocking}} (RT, CT, KT)$$

For example, switching from `"balanced"` to `"performance"` changes `weight_format` from bfloat16 (2,048 B/tile) to bfloat8\_b (1,088 B/tile), which reduces `page_size` by 47%, which allows larger `num_pages` within the same L1 budget, which enables larger blocking factors $(RT, CT, KT)$, which reduces NOC transaction count. The developer writes one string; the system derives the chain.

---

## BlazeModule: The Developer-Facing Object

`BlazeModule` extends `torch.nn.Module` so it participates naturally in PyTorch's module composition, serialization, and profiling infrastructure. Its design draws from two patterns identified in Chapter 2:

- **DP1 (Module replacement from TT-Symbiote):** Like Symbiote's `TTNNModule`, `BlazeModule` wraps a PyTorch module and provides hardware-accelerated `forward()`. Unlike Symbiote, it operates at the Blaze/LLK level (fused kernels on L1), not the TTNN op level.
- **DP4 (Four-phase lifecycle from TT-Symbiote):** `BlazeModule` follows analyze, validate, materialize, execute phase ordering with boolean guards to prevent redundant work.

```python
# PROPOSED (illustrative, not prescriptive) -- BlazeModule class
class BlazeModule(torch.nn.Module):
    """A PyTorch module that executes on Tenstorrent hardware via TT-Blaze.

    Created by blaze.from_pytorch(). Not intended for direct construction.
    """

    def __init__(self, adapter, compiled_program, weight_tensors, shape_descriptors):
        super().__init__()
        self._adapter = adapter
        self._compiled = compiled_program
        self._weights = weight_tensors
        self._shapes = shape_descriptors
        self._weights_on_device = False
        self._input_shape = shape_descriptors["input"]
        self._output_shape = shape_descriptors["output"]

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Execute on Tenstorrent hardware.

        Phase 1 (first call only): Move preprocessed weights to device.
        Phase 2: Prepare activation -- pad, format-convert, tilize.
        Phase 3: Dispatch compiled program.
        Phase 4: Extract output -- untilize, unpad, format-convert.
        """
        if not self._weights_on_device:
            self._adapter.move_weights_to_device(self._weights)
            self._weights_on_device = True

        act = self._adapter.prepare_activation(input_tensor, self._input_shape)
        result = self._compiled.run(activation=act)
        return self._adapter.extract_output(result, self._output_shape)

    @property
    def precision(self) -> str:
        """The active precision profile name."""
        return self._adapter.precision

    @property
    def fusion_report(self):
        """Diagnostic report from the fusion analyzer (Chapter 9)."""
        return self._compiled.fusion_report

    def l1_profile(self):
        """Print per-core L1 usage using blaze/l1_profile.py."""
        from blaze.l1_profile import print_cb_stats
        print_cb_stats(self._compiled.program)
```

### Why Extend torch.nn.Module?

Unlike Symbiote's `TTNNModule`, which deliberately avoids subclassing `nn.Module` to prevent autograd interference, `BlazeModule` subclasses it because:

1. **Module composition:** A `BlazeModule` can be a child of a larger PyTorch model tree (e.g., `model.attention.q_proj` is a `BlazeModule` while `model.lm_head` remains on CPU).
2. **Serialization:** `torch.save` / `torch.load` compatibility for checkpointing.
3. **Hook integration:** `register_forward_hook`, profiling, and debugging tools work unchanged.
4. **Familiar API:** PyTorch developers call `model(x)` regardless of the underlying backend.

Since TT-Blaze targets inference (training is out of scope per [Chapter 1](../ch01_the_gap/04_abstraction_goals_and_scope.md)), the autograd implications are acceptable.

---

## Configuration Knobs

### Precision Profiles

The `precision` parameter selects one of three built-in profiles defined in [Chapter 6, File 04](../ch06_data_formats/04_precision_profiles_and_user_overrides.md):

| Profile | Weight Format | Activation Format | Math Fidelity | fp32 Dest | Tile Size | Primary Use Case |
|---------|--------------|-------------------|---------------|-----------|-----------|-----------------|
| `"performance"` | bfloat8\_b | bfloat16 | LoFi | No | 1,088 B/tile (wt), 2,048 B/tile (act) | Maximum throughput, large models |
| `"balanced"` | bfloat16 | bfloat16 | HiFi4 | No | 2,048 B/tile | Default: good accuracy + good speed |
| `"accuracy"` | bfloat16 | bfloat16 | HiFi4 | Yes (`fp32_dest_acc_en=True`) | 2,048 B/tile | Debugging, quality-critical layers |

The profile configures the entire format and math-fidelity stack with a single string. The developer writes `precision="performance"` instead of manually setting `data_format`, `math_fidelity`, `math_approx_mode`, `fp32_dest_acc_en`, and `dst_full_sync_en` for each op.

**Tile size reference for all supported block floating-point formats:**

| Format | Tile Size (32x32) | Compression vs. bfloat16 |
|--------|-------------------|-------------------------|
| bfloat16 | 2,048 B/tile | 1.00x (baseline) |
| bfloat8\_b | 1,088 B/tile | 1.88x compression |
| bfloat4\_b | 576 B/tile | 3.56x compression |

The bfloat4\_b format provides the highest compression (576 bytes per 32x32 tile) and can be selected via `format_hints` for weight tensors where the quantization error is acceptable. All block floating-point formats (bfloat8\_b, bfloat4\_b) require 32x32 tile geometry because the shared exponent is computed over a 16-element face row within the 32x32 block.

### Per-Op Overrides

The `overrides` dict provides surgical control over specific ops without modifying the global profile:

```python
# PROPOSED -- per-op override usage
blaze_model = blaze.from_pytorch(
    llama_model,
    sample_input=torch.randn(1, 4096),
    device=device,
    precision="performance",
    overrides={
        "sdpa": {
            "math_fidelity": "HiFi4",
            "fp32_dest_acc_en": True,
        },
        "lm_head": {
            "fp32_dest_acc_en": True,
        },
    },
)
```

The override hierarchy (from [Chapter 6, File 04](../ch06_data_formats/04_precision_profiles_and_user_overrides.md)):

1. **Required constraints** (softmax safety, never violated)
2. **Per-op developer overrides** (the `overrides` dict)
3. **Precision profile defaults**
4. **System defaults** (bfloat16, HiFi4)

Overriding one setting does not require overriding all others (P4: Provide Defaults with Per-Decision Overrides).

---

## Weight Handling

Weight preprocessing bridges PyTorch's storage model (contiguous float32/bfloat16 parameters) to Tenstorrent's hardware model (tiled, sharded, format-converted device tensors).

### The Weight Pipeline

```text
PyTorch Parameter                Hardware-Ready Weight
[out_features, in_features]      [padded_K, padded_N] tiled, sharded, bfloat8_b
        |                                   ^
        | 1. Transpose                      |
        v                                   |
[in_features, out_features]      5. Move to device DRAM
        |                                   ^
        | 2. Format conversion              |
        v                                   |
bfloat8_b / bfloat16            4. Shard across cores
        |                                   ^
        | 3. Pad to tile boundary           |
        v                                   |
[padded_H, padded_W]            per-core shard tensors
```

Step 1 (transpose) is necessary because `nn.Linear` stores weights as `[out_features, in_features]`, but the matmul kernel expects `[K, N]` where $K = \text{in\_features}$.

Step 2 (format conversion) applies the precision profile's weight format. For the performance profile, this converts from `bfloat16` to `bfloat8_b`, reducing per-tile size from 2,048 to 1,088 bytes (a $\frac{2048}{1088} \approx 1.88\times$ compression ratio).

Step 3 (padding) ensures the weight dimensions are multiples of the tile shape (32x32 by default). The padding fill value is always ZERO for weights (Chapter 5), following P6 (Per-Operation Padding Semantics).

Step 4 (sharding) distributes the padded weight tensor across the core grid. For a [4096, 11008] weight on 14 cores, each core receives a shard of approximately [4096, 786].

Step 5 (device placement) moves the shard tensors to device DRAM. This is deferred until the first `forward()` call (lazy movement) to avoid occupying device memory for models that are constructed but not yet used.

### Lazy Weight Movement

```python
# PROPOSED (illustrative, not prescriptive) -- lazy weight movement
def forward(self, input_tensor):
    if not self._weights_on_device:
        # First call: move preprocessed weights to device DRAM
        self._adapter.move_weights_to_device(self._weights)
        self._weights_on_device = True
    # ... proceed with execution
```

This pattern matches Symbiote's lazy materialization (DP4): the analyze and validate phases complete during `from_pytorch()`, but the materialization phase (device transfer) is deferred until execution is imminent.

---

## BlazeModule vs. TTNNModule

Both `BlazeModule` and Symbiote's `TTNNModule` convert PyTorch modules to hardware execution, but they operate at different levels of the stack:

| Dimension | BlazeModule | TTNNModule (Symbiote) |
|-----------|-------------|----------------------|
| **Execution level** | Fused LLK kernels on L1 | TTNN ops on DRAM |
| **Intermediate storage** | L1 SRAM (~1.5 MB per core, no DRAM round-trips for fused ops) | DRAM (each op reads/writes DRAM) |
| **Fusion** | Multi-op fusion via FusedProgram | No fusion (each op is independent) |
| **CB control** | Explicit: tile shape, page count, format per CB | Implicit: TTNN manages CBs internally |
| **Performance ceiling** | Higher (direct hardware control) | Lower (TTNN abstraction overhead) |
| **Developer effort** | Lower with TensorAdapter (3 lines) | Already low (module replacement) |
| **Escape hatches** | Three levels (auto, configurable, manual) | Limited (exclude\_replacement set) |
| **Weight format** | Configurable per profile | Determined by TTNN op |
| **L1 budgeting** | Compile-time validation | No visibility |
| **torch.nn.Module subclass** | Yes | No (avoids autograd) |

### When to Use Each

- **Use BlazeModule** when performance is critical and the model uses standard patterns (matmul, MLP, attention, normalization) that the fusion analyzer recognizes.
- **Use TTNNModule** when rapid prototyping is needed, when the model uses ops not yet supported by Blaze, or when TTNN-level ops provide sufficient performance.
- **Use both** in a hybrid model: TTNNModule for unsupported ops, BlazeModule for performance-critical subgraphs. The `ExternalTensor` mechanism bridges the two worlds (see [File 06](./06_escape_hatches_for_power_users.md)).

---

## The Four-Phase Lifecycle

`BlazeModule` follows the four-phase lifecycle established in [Chapter 2, File 01](../ch02_design_patterns/01_tt_symbiote_module_replacement.md) (DP4 from TT-Symbiote):

```text
Phase 1: ANALYZE (during from_pytorch())
  - Trace the module's forward() to capture the computation graph
  - Map PyTorch ops to TT-Blaze ops
  - Infer shapes via ShapeEngine, producing ShapeDescriptor per edge
  - Detect fusible subgraphs via FusionAnalyzer
  Guard: _analyzed = True (prevents re-analysis)

Phase 2: VALIDATE (during from_pytorch())
  - Check L1 budget per core (~1.5 MB total, ~1.3 MB available after reserved)
  - Verify CB count <= 64 per phase
  - Validate format compatibility at all edges
  - Validate padding strategy transitions (P6: Per-Operation Padding Semantics)
  - Verify core grid alignment for fused regions
  Guard: _validated = True (prevents compilation of invalid graphs)

Phase 3: MATERIALIZE (during from_pytorch() + first forward())
  - Preprocess weights (transpose, format-convert, pad, shard)
  - Compile BlazeGraph to MeshCompiledProgram
  - [Deferred] Move weights to device (first forward())
  Guard: _compiled = True, _weights_on_device = True

Phase 4: EXECUTE (every forward() call)
  - Prepare activation tensor
  - Dispatch compiled program
  - Extract and return output tensor
  Guard: All previous guards must be True
```

Boolean guards ensure that phases execute in order. Calling `forward()` before `from_pytorch()` completes raises a clear error rather than producing undefined behavior.

---

## Compile-Time vs. Runtime Boundary

A critical architectural property of `BlazeModule` is that all shape analysis, tile selection, padding computation, format negotiation, CB allocation, and CT arg derivation happen at compile time (stages 1-4). The compiled program that executes on hardware is byte-identical to one produced by hand-written code. There is no adapter in the execution path.

```text
Compile-time (from_pytorch()):    Runtime (forward()):
  TensorAdapter                     CompiledProgram.run()
  ShapeEngine                       [no adapter code]
  PaddingPolicy                     [no shape inference]
  FormatPolicy                      [no format negotiation]
  CBSizer                           [no CB allocation]
  L1BudgetValidator                 [no L1 checking]
  CBEngine                          [no engine passes]
  CTArgEngine
  BlazeCompiler
         |
         v
  MeshCompiledProgram  -------->    Device execution
```

This separation means TensorAdapter adds zero runtime overhead. The per-inference latency is determined entirely by the compiled kernel, which is identical regardless of whether the parameters were computed manually or automatically.

---

## Error Handling and Diagnostics

When `from_pytorch()` fails, it provides actionable diagnostics:

```python
# L1 budget exceeded
try:
    blaze_model = blaze.from_pytorch(
        very_large_model,
        torch.randn(1, 65536),
        device,
        precision="accuracy",
    )
except L1BudgetError as e:
    # "CB 'weight_shard' requests 2,097,152 bytes (1024 pages x 2048
    #  bytes/page), but only ~1,300,000 bytes remain of ~1,500,000 total.
    #  Suggestion: Use precision='performance' (bfloat8_b weights) to
    #  reduce weight CB by 47%, or increase core count from 14 to 28."
    pass
```

---

## Key Takeaways

- `blaze.from_pytorch()` is the top-level entry point. It accepts a `torch.nn.Module`, a sample input, and a device, returning a `BlazeModule` that executes on hardware. Three lines of code replace approximately 200 lines of manual configuration.
- The under-the-hood pipeline has six stages: trace, build BlazeGraph, engine passes, compile, weight placement, and execute. Stages 1-4 execute once during `from_pytorch()`; stage 5 executes on the first `forward()` (lazy weight movement); stage 6 executes on every subsequent `forward()`.
- `BlazeModule` extends `torch.nn.Module` for seamless composition, serialization, and hook integration. Its four-phase lifecycle (analyze, validate, materialize, execute) with boolean guards prevents ordering violations.
- Configuration knobs (`precision`, `format_hints`, `grid`, `overrides`, `fusion`) implement P4 (Provide Defaults with Per-Decision Overrides). The developer can configure the entire model with a single `precision` string or surgically override individual ops.
- All TensorAdapter computation happens at compile time. The compiled program is byte-identical to hand-written code. There is zero runtime overhead from the abstraction.
- Three tile sizes for block floating-point formats: bfloat16 = 2,048 B/tile, bfloat8\_b = 1,088 B/tile, bfloat4\_b = 576 B/tile. All BFP formats require 32x32 tile geometry.

## Source Files

- `blaze/context.py` -- `FusionContext`, `ExternalTensor`, `fuse()` context manager
- `blaze/compiler.py` -- `BlazeCompiler.compile()`, `_get_compute_config()`
- `blaze/cb_engine.py` -- `CBEngine.assign()`, `compact_cb_ids()`
- `blaze/sem_engine.py` -- `SemEngine.assign()`
- `blaze/ct_args.py` -- `CTArgEngine.generate_tuples()`
- `blaze/fused_program.py` -- `FusedProgram.build()`
- `blaze/l1_profile.py` -- `print_cb_stats()`, `print_kernel_stats()`

---

**Next:** [`02_worked_example_linear_layer.md`](./02_worked_example_linear_layer.md)
