# MLIR and torch.compile: Dialect Layering and Inductor as Design Patterns

**What you will learn:**

- How MLIR's dialect system enables progressive lowering through well-defined abstraction boundaries
- How MLIR's bufferization pass (tensor to memref) maps to TensorAdapter's CB allocation step
- A hypothetical CB MLIR dialect that captures Tenstorrent-specific FIFO semantics missing from standard memref
- How `torch.compile` and Inductor capture PyTorch operations via FX graphs and lower them to hardware-specific kernels
- The fusion boundary problem: why boundaries are inevitable, their cost (~4us per tensor for DRAM round-trip), and their impact on single-token decode performance
- TT-Forge's MLIR-based pipeline and where TensorAdapter extends beyond it
- Four design patterns (DP13-DP16) for dialect-like layering, graph capture, pluggable backends, and compiler analysis feeding manual composition

---

## MLIR: Progressive Lowering Through Dialect Layers

### The MLIR Architecture

MLIR (Multi-Level Intermediate Representation) is a compiler infrastructure that solves a problem directly analogous to TensorAdapter's: bridging between high-level tensor operations and low-level hardware-specific execution. MLIR's core innovation is the **dialect system** -- a mechanism for defining multiple IRs at different abstraction levels within a single framework, with well-defined lowering passes between them.

A typical MLIR compilation pipeline for ML workloads:

```
torch dialect        -- PyTorch-level: torch.aten.mm, torch.aten.relu
    |  (lowering pass)
    v
linalg dialect       -- Structured ops: linalg.matmul, linalg.generic
    |  (lowering pass)
    v
affine/scf dialect   -- Loops and index math: affine.for, scf.for
    |  (lowering pass)
    v
memref dialect       -- Buffer management: memref.alloc, memref.load/store
    |  (lowering pass)
    v
llvm/gpu dialect     -- Target-specific: llvm.call, gpu.launch_kernel
```

Each dialect defines its own **types, operations, and transformations**. A lowering pass converts operations from one dialect to a lower dialect, progressively adding hardware-specific detail while removing high-level abstraction.

### The linalg Dialect: Algorithm Without Memory

The `linalg` dialect represents tensor computations as named operations on abstract tensors with no memory semantics:

```mlir
// EXISTING -- MLIR linalg dialect: matmul on tensors
func.func @matmul(%A: tensor<1024x512xf32>,
                   %B: tensor<512x768xf32>) -> tensor<1024x768xf32> {
  %C_init = tensor.empty() : tensor<1024x768xf32>
  %C = linalg.matmul
    ins(%A, %B : tensor<1024x512xf32>, tensor<512x768xf32>)
    outs(%C_init : tensor<1024x768xf32>) -> tensor<1024x768xf32>
  return %C : tensor<1024x768xf32>
}
```

This is purely declarative: `linalg.matmul` specifies the contraction without saying anything about tiles, buffers, or execution units. Compare to Blaze's `MicroOp` port descriptors, which are similarly declarative:

```python
# EXISTING -- Blaze port descriptors (declarative, like linalg)
class Matmul(MicroOp):
    in0: Input = Input()    # Algorithm: "I consume two inputs"
    in1: Input = Input()
    out: Output = Output()  # Algorithm: "I produce one output"
```

### Tiling Transform: From Algorithm to Tiled Algorithm

MLIR's tiling transform converts a single `linalg.matmul` into a tiled version wrapped in `scf.for` loops:

```mlir
// EXISTING -- MLIR tiling transform result
scf.for %i = 0 to 1024 step 32 {
  scf.for %j = 0 to 768 step 32 {
    %A_tile = tensor.extract_slice %A[%i, 0] [32, 512] [1, 1]
        : tensor<1024x512xf32> to tensor<32x512xf32>
    scf.for %k = 0 to 512 step 32 {
      %B_tile = tensor.extract_slice %B[%k, %j] [32, 32] [1, 1]
          : tensor<512x768xf32> to tensor<32x32xf32>
      %C_tile = linalg.matmul
          ins(%A_tile, %B_tile : tensor<32x512xf32>, tensor<32x32xf32>)
          outs(...) -> tensor<32x32xf32>
      // ... insert_slice back
    }
  }
}
```

The tile sizes (32x32 in this example) are parameters to the tiling pass, not embedded in the algorithm. This is TVM's schedule-algorithm separation (DP5 in [Section 02](./02_tvm_relay_tir_scheduling.md)) realized in an IR framework.

### Bufferization: Tensors to Memory References

The bufferization pass converts immutable tensors to mutable `memref` (memory reference) objects -- the MLIR equivalent of CB allocation:

```mlir
// EXISTING -- MLIR after bufferization
func.func @buffered_matmul(
    %A: memref<1024x512xf32>,
    %B: memref<512x768xf32>,
    %C: memref<1024x768xf32>) {
  // Allocate tile buffers -- analogous to cb_scratch()
  %A_buf = memref.alloc() : memref<32x512xf32, #gpu.address_space<workgroup>>
  %B_buf = memref.alloc() : memref<512x32xf32, #gpu.address_space<workgroup>>
  %C_buf = memref.alloc() : memref<32x32xf32, #gpu.address_space<private>>

  scf.for %i = 0 to 1024 step 32 {
    // Copy tile to workgroup memory -- analogous to cb_from_tensor()
    memref.copy %A_subview, %A_buf : memref<32x512xf32> to
        memref<32x512xf32, #gpu.address_space<workgroup>>
    scf.for %j = 0 to 768 step 32 {
      // Compute on tile buffers
      linalg.matmul ins(%A_buf, %B_buf) outs(%C_buf)
      // Write back -- analogous to wire_output()
      memref.copy %C_buf, %C_subview
    }
  }
  memref.dealloc %A_buf : memref<32x512xf32, #gpu.address_space<workgroup>>
}
```

The `#gpu.address_space<workgroup>` annotation is the memory scope -- directly analogous to TVM's `scope="shared"` (DP6 in [Section 02](./02_tvm_relay_tir_scheduling.md)) and Blaze's CB-in-L1:

| MLIR Address Space | GPU Analog | Tenstorrent Analog | Blaze Construct |
|-------------------|------------|-------------------|-----------------|
| `#gpu.address_space<global>` | DRAM/HBM | DRAM | `ttnn.Tensor` in DRAM |
| `#gpu.address_space<workgroup>` | Shared memory | L1 SRAM | `CBHandle(backing_tensor=...)` |
| `#gpu.address_space<private>` | Registers | Tile registers | DST accumulator (managed by LLK) |

The key insight: **bufferization is a compiler pass, not a programmer responsibility**. The algorithm author writes in `linalg`; the compiler inserts `memref.alloc` and `memref.copy` operations. TensorAdapter should perform an analogous transformation: the op author declares ports; the framework inserts `cb_from_tensor`, `cb_scratch`, and wiring.

---

## memref vs. Tenstorrent CBs: The Flow Control Gap

The `memref` abstraction is a good starting point but insufficient for Tenstorrent's CB model:

| Property | MLIR memref | Tenstorrent CB (CBHandle) |
|----------|-------------|---------------------------|
| Address mode | Random access (load/store) | FIFO (push/pop/wait) or direct-address |
| Size specification | Element counts per dimension | Page count * page size (tile-aligned) |
| Memory space | Attribute tag (integer/string) | CB slot index (0-63, hardware register) |
| Lifetime | Scope-based (alloc/dealloc) | Phase-based (can be reconfigured mid-kernel) |
| Producer-consumer | Data dependence analysis | Hardware flow control (cb_wait_front/cb_push_back) |
| Multi-occupant | Not supported | Disjoint-core sharing (same cb_id, different core_ranges) |

The key missing abstraction is **flow control**. A `memref` is a passive memory region; a CB is an active FIFO with hardware-managed synchronization. Writing to a full CB blocks the producer until the consumer pops; reading from an empty CB blocks the consumer until the producer pushes. This behavior is not expressible in `memref` semantics.

### A Hypothetical CB Dialect

If we were designing a Tenstorrent-specific MLIR dialect (which TT-Forge's MLIR integration partially does), it would need operations that capture the essential CB semantics that `memref` lacks:

```mlir
// PROPOSED -- hypothetical CB dialect operations (illustrative, not prescriptive)
%cb = cb.allocate {
    page_size = 2048 : i32,
    num_pages = 4 : i32,
    data_format = #tt.dtype<bfloat16>,
    tile_shape = [32, 32],
    access_mode = #cb.mode<fifo>,
    core_ranges = #tt.core_range<[0,0]-[7,7]>
} : !cb.handle

cb.push_back %cb, %tile : !cb.handle, !tt.tile<32x32xbf16>
cb.wait_front %cb, %n_pages : !cb.handle, i32
%tile = cb.pop_front %cb : !cb.handle -> !tt.tile<32x32xbf16>

cb.reconfigure %cb {
    // Mid-kernel: reprogram CB registers with new L1 addresses
    new_address = %addr : i32,
    new_total_size = %size : i32
}
```

This dialect would capture:

1. **Allocation with hardware constraints**: `cb.allocate` bundles page_size, num_pages, data_format, tile_shape, access_mode, and core_ranges -- the same six parameters a Blaze developer specifies manually per CB.
2. **FIFO flow control**: `cb.push_back` and `cb.wait_front` express the producer-consumer synchronization that `memref.store`/`memref.load` cannot.
3. **Mid-kernel reconfiguration**: `cb.reconfigure` captures the phase-boundary behavior where CB registers are reprogrammed without deallocation -- Blaze's `CircularBufferIdManager` context system.

TensorAdapter does not use MLIR directly, but its internal representation should capture these same concepts. The `CBHandle` type in Blaze is the closest existing construct to `!cb.handle`; TensorAdapter's contribution is automating the derivation of the six allocation parameters from tensor shape and op semantics.

---

## Design Pattern 13 (DP13): Dialect-Like Abstraction Layers

### EXISTING (MLIR)

The dialect pattern provides three guarantees at each level:

1. **Type safety**: Each dialect defines its types. A `tensor<32x32xf16>` in the linalg dialect is a different type from a `memref<32x32xf16>` in the memref dialect. The lowering pass that converts one to the other is explicit.
2. **Invariant preservation**: Each dialect has invariants. In the linalg dialect, matmul inputs must have compatible inner dimensions. In the memref dialect, buffer accesses must be within bounds. Each lowering pass preserves the target dialect's invariants.
3. **Partial lowering**: Not every operation must be lowered at once. A pass can lower `linalg.matmul` to `scf.for` loops while leaving `linalg.generic` unchanged for a later pass. This enables incremental migration.

### Mapping to TensorAdapter Layers

TensorAdapter should implement a three-layer architecture inspired by MLIR's dialect stacking:

```
# PROPOSED -- TensorAdapter dialect-like layers (illustrative, not prescriptive)

Layer 1: Tensor Layer (PyTorch dialect)
    Types: torch.Tensor with shape, dtype
    Operations: matmul, add, softmax, rmsnorm
    Invariant: shapes are compatible per PyTorch broadcasting rules

        |  ShapeEngine.decompose()
        v

Layer 2: Tile Layer (Blaze dialect)
    Types: ShapeDescriptor with logical_shape, padded_shape, tile_shape, tile_grid
    Operations: tiled_matmul, tiled_add, tiled_softmax
    Invariant: all shapes are tile-aligned; padding and format are specified

        |  CBSizer.allocate()
        v

Layer 3: CB Layer (Hardware dialect)
    Types: CBHandle with cb_id, num_pages, page_size, data_format, core_ranges
    Operations: FusedProgram.cb_from_tensor(), FusedProgram.cb_scratch()
    Invariant: L1 budget satisfied; CB IDs < 64; FIFO semantics respected
```

Each layer has explicit types, invariants, and a lowering function that converts the higher layer's types into the lower layer's types. The lowering functions are the automation points:

| Lowering | Input | Output | Automation |
|----------|-------|--------|------------|
| Tensor -> Tile | `torch.Tensor` + op hint | `ShapeDescriptor` | Tile shape, padding, format selection |
| Tile -> CB | `ShapeDescriptor` | `CBHandle` parameters | num_pages, page_size, core_ranges, CB ID |

### The Partial Lowering Advantage

MLIR's partial lowering is particularly relevant. A TensorAdapter pass might lower 8 of 10 ops automatically but leave 2 ops at the Tensor layer because their shapes are unusual. The developer then manually provides the missing ShapeDescriptors for those 2 ops and the pipeline continues. This is exactly the Level 1/Level 2 escape hatch from [Chapter 1: Abstraction Goals](../ch01_the_gap/04_abstraction_goals_and_scope.md) -- you do not have to lower everything automatically; you can mix automatic and manual lowering in the same pipeline.

---

## torch.compile and Inductor: Graph Capture and Backend Lowering

### The torch.compile Pipeline

`torch.compile` (introduced in PyTorch 2.0) provides a graph capture and compilation pipeline that transforms eager PyTorch code into optimized kernels:

```
@torch.compile
def fn(x, w):
    return torch.relu(x @ w)

fn(input)
    |
    v
TorchDynamo: traces Python bytecode, captures FX graph
    |
    v
FX Graph: graph of aten operations (aten.mm, aten.relu)
    |
    v
AOTAutograd: forward/backward decomposition (inference: identity pass)
    |
    v
Inductor: lowers FX graph to target code (Triton for GPU, C++ for CPU)
    |
    v
Compiled kernel
```

### Design Pattern 14 (DP14): FX Graph Capture for Operation Discovery

#### EXISTING (torch.compile)

TorchDynamo captures PyTorch operations into an FX graph -- a DAG of `aten` operations with explicit data dependencies. The FX graph provides exactly what TensorAdapter needs to make informed decisions:

1. **Operation types**: Each node declares its operation (`aten.mm`, `aten.add`, `aten._softmax`).
2. **Input/output shapes**: Shape propagation annotates each node with its output shape.
3. **Data flow**: Edges between nodes show which operation consumes which output.
4. **Metadata**: Nodes carry dtype, device, and other tensor properties.

```python
# EXISTING -- FX graph capture and inspection
import torch.fx

def fn(x, w):
    y = x @ w
    return torch.relu(y)

traced = torch.fx.symbolic_trace(fn)
for node in traced.graph.nodes:
    print(f"{node.name}: {node.op} -> {node.target}")
# matmul_1: call_function -> aten.mm
# relu_1: call_function -> aten.relu
```

#### Mapping to TensorAdapter

FX graph capture is the `torch.compile` equivalent of Blaze's graph context. Both intercept a computation and produce a DAG:

| Concept | torch.compile/FX | Blaze Graph API |
|---------|-----------------|-----------------|
| Capture mechanism | `torch._dynamo` tracing | `blaze.fuse()` context manager |
| Node representation | `fx.Node` with `target` function | `OpNode` with op name |
| Edge representation | Implicit (node input refs) | Explicit `Edge` objects |
| External inputs | `placeholder` nodes | `ExternalTensor` instances |

TensorAdapter should use FX graph capture (or BlazeGraph, which serves the same purpose) as the input to its adaptation pipeline. The graph provides the context TensorAdapter needs for cross-op decisions:

- **Format negotiation**: If op A produces bfloat8_b but op B needs bfloat16, insert a conversion CB at the boundary.
- **L1 budgeting**: Sum all CBs that are live simultaneously on the same core. If the total exceeds the budget, reduce num_pages or change formats.
- **CB reuse**: Two ops that are never live simultaneously can share a CB ID, reducing pressure on the 64-slot limit.

```python
# PROPOSED -- TensorAdapter consuming FX or BlazeGraph (illustrative, not prescriptive)
class TensorAdapter:
    @classmethod
    def from_fx_graph(cls, graph: torch.fx.Graph, device) -> "AdaptedProgram":
        """Lower an FX graph to Blaze FusedProgram with automatic CB management."""
        descriptors = {}
        for node in graph.nodes:
            if node.op == "call_function":
                op_hint = aten_to_blaze_hint(node.target)
                input_shapes = [descriptors[inp].padded_shape for inp in node.args
                                if inp in descriptors]
                descriptors[node] = ShapeEngine.decompose(
                    input_shapes, op_hint=op_hint, l1_budget=device.l1_size
                )
        return cls._build_fused_program(descriptors, device)
```

---

## The Fusion Boundary Problem

`torch.compile` encounters **graph breaks** when TorchDynamo cannot trace through certain Python constructs (data-dependent control flow, calls to unsupported libraries, dynamic shapes exceeding guard limits). Each graph break splits the compiled region into separate subgraphs, each compiled independently.

This is directly analogous to the fusion boundary problem in TensorAdapter. When a PyTorch computation cannot be fully captured as a single Blaze FusedProgram (due to dynamic shapes, unsupported operations, or CB count limits), it must be split into multiple fusion regions with explicit data handoff between them.

### Fusion Boundary Cost Analysis

The cost of a fusion boundary is a full L1-to-DRAM-to-L1 round-trip:

```python
# PROPOSED -- Fusion boundary cost model (illustrative, not prescriptive)
class FusionRegion:
    """A contiguous region of operations that can be compiled into
    a single Blaze FusedProgram.

    Boundaries occur when:
    - CB count exceeds 64 (triggering multi-phase reconfig instead)
    - An unsupported operation requires fallback to TTNN-level dispatch
    - Dynamic shapes prevent static CB allocation
    """

    def boundary_cost(self):
        """Cost in microseconds of materializing outputs to DRAM
        and re-loading as inputs to the next region.

        For a [1, 32, 4096] bfloat16 tensor:
          Write to DRAM: ~2us (L1 -> DRAM via NOC)
          Read from DRAM: ~2us (DRAM -> L1 via NOC)
          Total: ~4us per boundary tensor

        For single-token decode at ~10us per matmul,
        a boundary adds ~40% overhead per tensor crossing it.
        """
        pass
```

The three types of boundaries have very different costs:

| Boundary Type | Cost | Mechanism |
|--------------|------|-----------|
| Phase boundary (CB reconfiguration) | ~0us | CB registers reprogrammed in-place; no DRAM round-trip |
| DRAM materialization boundary | ~4us per tensor | L1 -> DRAM write + DRAM -> L1 read via NOC |
| Host round-trip boundary | ~100us+ | Device -> host -> device; Python dispatch overhead |

Phase boundaries within a FusedProgram (via Blaze's `CircularBufferIdManager`) are nearly free -- they reprogram CB registers without materializing data to DRAM. DRAM materialization boundaries occur when outputs must be visible to a separate FusedProgram. Host round-trip boundaries are the most expensive and occur when an unsupported op forces fallback to CPU execution.

The lesson: **fusion boundaries are inevitable, and the system must handle them gracefully**. Torch.compile inserts Python-level handoff code at graph breaks. TensorAdapter must insert DRAM-level handoff (materializing CB outputs to `ttnn.Tensor` objects) at fusion boundaries. Minimizing boundaries is a primary optimization goal, and the 64-CB slot limit is the most common trigger for boundaries that cannot be resolved through multi-phase reconfiguration.

---

## Design Pattern 15 (DP15): Backend-Specific Lowering with Shared Frontend

### EXISTING (torch.compile/Inductor)

Inductor is `torch.compile`'s default backend, but the architecture supports pluggable backends. A backend receives the FX graph and produces compiled code:

```python
# EXISTING -- torch.compile backend registration
@torch._dynamo.register_backend
def my_backend(gm: torch.fx.GraphModule, example_inputs):
    # Lower gm to target-specific code
    compiled_fn = lower_to_my_hardware(gm)
    return compiled_fn

model = torch.compile(model, backend="my_backend")
```

TT-Forge uses exactly this pattern -- it registers as a `torch.compile` backend that lowers FX graphs through its MLIR-based pipeline to TT-Metal programs. TensorAdapter should follow the same pattern: register as a compilation backend (or wrap a BlazeGraph consumer) that receives the graph and produces a FusedProgram with automatically-computed CB parameters.

### Inductor's Tile Size Heuristics

Inductor selects tile sizes for generated kernels using heuristics based on tensor shapes and hardware characteristics:

```python
# EXISTING -- Inductor tile size heuristics (simplified from triton_heuristics.py)
def select_tile_sizes(M, N, K, dtype):
    # Start with hardware-friendly defaults
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32

    # Adjust for small dimensions
    if M < 128:
        BLOCK_M = max(16, triton.next_power_of_2(M))
    if N < 128:
        BLOCK_N = max(16, triton.next_power_of_2(N))

    # Adjust for register pressure with larger dtypes
    if dtype == torch.float32:
        BLOCK_K = min(BLOCK_K, 32)
    elif dtype == torch.float16:
        BLOCK_K = min(BLOCK_K, 64)

    # Adjust for L2 cache size
    total_bytes = BLOCK_M * BLOCK_K * dtype.itemsize + BLOCK_K * BLOCK_N * dtype.itemsize
    while total_bytes > L2_CACHE_SIZE // 2 and BLOCK_K > 16:
        BLOCK_K //= 2
        total_bytes = BLOCK_M * BLOCK_K * dtype.itemsize + BLOCK_K * BLOCK_N * dtype.itemsize

    return BLOCK_M, BLOCK_N, BLOCK_K
```

These heuristics map to CB parameter selection:

| Inductor Heuristic | CB Parameter Analog |
|-------------------|-------------------|
| "Fit in shared memory" | `page_size * num_pages <= L1_budget` |
| "Powers of 2 tile sizes" | Tile dimensions aligned to 32 (Tenstorrent native) |
| "Balance parallelism vs. per-tile work" | Grid size vs. tiles-per-core tradeoff |
| "No tiling for small tensors" | Direct L1 residency without CB streaming |
| "Adjust for data type" | bfloat16 page_size=2048 vs. bfloat8_b page_size=1088 for 32x32 tiles |

Inductor also bounds fusion by hardware resources: intermediate buffers must fit in shared memory, and register pressure must stay within limits. This is the exact analog of Blaze's 64-CB limit as a fusion boundary.

### Inductor's Memory Planning

TorchInductor performs memory planning on the FX graph that is conceptually identical to Blaze's CB compaction:

```python
# EXISTING -- TorchInductor memory planning (simplified logic)
# Reuse buffers whose lifetimes don't overlap
for buf in sorted_by_size_descending:
    for candidate in free_buffers:
        if candidate.size >= buf.size and not overlaps(candidate.lifetime, buf.lifetime):
            buf.storage = candidate.storage  # Reuse
            break
    else:
        buf.storage = allocate_new(buf.size)
```

The difference: Inductor plans in bytes (any buffer size, any alignment); Blaze plans in CB slots (64-slot limit, page-aligned, format-constrained). But the algorithmic core is the same interval-based resource reuse that TVM's `StorageRewrite` and Blaze's `compact_cb_ids` both implement (see DP6 in [Section 02](./02_tvm_relay_tir_scheduling.md)).

---

## TT-Forge: MLIR-Inspired Compilation for Tenstorrent

### Where TT-Forge Fits

TT-Forge is Tenstorrent's graph compiler that sits at the same level as Inductor but targets TT-Metal instead of Triton/CUDA. Its pipeline:

```
PyTorch model
    |
    v
torch.compile capture -> FX graph
    |
    v
TT-Forge frontend: FX -> TTIR (Tenstorrent IR, an MLIR dialect)
    |
    v
TTIR optimizations: constant folding, op fusion, layout transforms
    |
    v
TTIR -> TTNN dialect (hardware-specific operations)
    |
    v
TTNN -> TT-Metal ProgramDescriptors
    |
    v
Dispatch to hardware
```

TT-Forge demonstrates that MLIR-style dialect lowering works for Tenstorrent hardware. However, TT-Forge is a **graph compiler** -- it generates kernels automatically. TT-Blaze is a **composition framework** -- developers write kernels manually. TensorAdapter bridges these worlds: it uses graph-compiler-like analysis (shape decomposition, format negotiation) but feeds the results into Blaze's manual composition API rather than generating kernels.

### Where TT-Forge Stops and TensorAdapter Begins

TT-Forge lowers to TTNN operations -- each TTNN op manages its own CBs internally. This is the same level as Symbiote (DP1-DP4 in [Section 01](./01_tt_symbiote_module_replacement.md)): per-op CB management with no cross-op visibility.

TensorAdapter's value proposition is lowering past TTNN to the **Blaze level** -- where CBs are explicitly managed across a fused pipeline of operations:

```
  TT-Forge output: ttnn.linear(x, w)  -->  Single TTNN dispatch
                                            CBs allocated internally
                                            No cross-op sharing

  TensorAdapter output: FusedProgram   -->  Single fused dispatch
    with explicit CBs:                      CBs managed across ops
      cb0 = cb_from_tensor(x)              Cross-op CB sharing
      cb1 = cb_from_tensor(w)              Interval coloring
      cb2 = cb_scratch("matmul_out")       Multi-phase reconfig
      Matmul.emit(f, cb0, cb1, cb2)
```

> **Warning:** TT-Forge and TensorAdapter are not competing systems -- they target different points on the automation-control spectrum. TT-Forge optimizes for breadth (many models, automatic compilation). TensorAdapter optimizes for depth (specific models, maximum hardware utilization through explicit CB management). In practice, a model might use TT-Forge for non-critical layers and TensorAdapter/Blaze for performance-critical fused kernels.

---

## Design Pattern 16 (DP16): Compiler Analysis Feeding Manual Composition

### PROPOSED

The insight from comparing TT-Forge and TT-Blaze is that compiler-style analysis and manual kernel composition are not mutually exclusive. TensorAdapter should:

1. **Analyze like a compiler**: Perform shape inference, format negotiation, L1 budgeting, and CB sizing across the whole graph -- the same analysis TT-Forge performs.
2. **Execute like Blaze**: Feed the analysis results into `FusedProgram.cb_from_tensor()`, `emit()` calls, and CT arg wiring -- the same execution path a human developer uses.

```python
# PROPOSED -- compiler analysis + manual composition (illustrative, not prescriptive)
analysis = TensorAdapter.analyze_graph(blaze_graph, device)
# analysis.descriptors: per-edge ShapeDescriptors
# analysis.format_assignments: per-edge data formats
# analysis.cb_sizing: per-CB num_pages and page_size
# analysis.l1_report: per-core memory usage

# Now feed into existing Blaze APIs
f = FusedProgram(kernel=None, device=device)
for op_node in blaze_graph.topological_order():
    input_descriptors = [analysis.descriptors[edge] for edge in op_node.input_edges]
    # TensorAdapter computes the parameters; Blaze executes the emit
    op_node.op_class.emit(f, *prepared_inputs, prefix=op_node.ct_prefix)
```

This pattern preserves Blaze's "C++ kernel developers are first-class citizens" philosophy while adding the shape-aware automation that PyTorch developers expect. The template-based nature of Blaze's `MicroOp` library makes this tractable: each op has known CB requirements (documented in its `OpSpec`), so CB allocation can be derived from the composition pattern rather than requiring per-instance specification.

> **Note:** The interface sketches above illustrate how surveyed principles might manifest in TensorAdapter. The actual design is defined in [Chapter 3: TensorAdapter Architecture](../ch03_tensor_adapter/index.md).

---

## The torch.compile/TT-Forge/TensorAdapter Comparison

| Dimension | torch.compile + Inductor | TT-Forge | TensorAdapter (Proposed) |
|-----------|--------------------------|----------|--------------------------|
| Input | PyTorch eager code | PyTorch model graph | PyTorch tensor + op hint (or BlazeGraph) |
| Graph capture | TorchDynamo (FX) | torch.compile (FX) | FX or BlazeGraph |
| Lowering target | Triton/C++ kernels | TT-Metal programs | FusedProgram (CB parameters) |
| Who writes kernels | Inductor (auto-generated) | TT-Forge (auto-generated) | Developer (hand-written C++) |
| Tiling decisions | Inductor heuristics | Forge MLIR passes | ShapeEngine rules |
| Memory management | Inductor scheduler | Forge layout transforms | CBSizer + L1 budget validator |
| CB-level control | N/A | N/A (internal to TTNN) | Full: per-op, per-graph, bypass |
| Escape hatch | Custom backend registration | Limited | Full: auto -> hints -> manual FusedProgram |

---

## Key Takeaways

- **DP13: MLIR's dialect layers** provide the architectural template for TensorAdapter: Tensor Layer (PyTorch shapes) -> Tile Layer (ShapeDescriptors) -> CB Layer (CBHandles). Each layer has explicit types, invariants, and lowering passes. Partial lowering lets TensorAdapter handle 80% of ops automatically while leaving 20% for manual specification.
- **The hypothetical CB dialect** (`cb.allocate`, `cb.push_back`, `cb.wait_front`, `cb.reconfigure`) captures Tenstorrent-specific FIFO semantics that MLIR's `memref` cannot express. TensorAdapter's internal representation should model these same concepts even without using MLIR directly.
- **DP14: FX graph capture** (used by torch.compile and TT-Forge) provides the graph-level context needed for cross-op decisions: format negotiation, L1 budgeting, CB reuse. TensorAdapter should consume BlazeGraph or FX graphs.
- **DP15: Pluggable backend architecture** lets TensorAdapter register alongside TT-Forge, receiving the same FX graph but lowering to Blaze FusedPrograms instead of TTNN dispatches.
- **Fusion boundaries are inevitable and expensive** (~4us per tensor for DRAM round-trip, ~40% overhead for single-token decode). Phase boundaries within a FusedProgram are nearly free; DRAM materialization boundaries are not. Minimizing boundaries is a primary optimization goal.
- **DP16: Compiler analysis feeding manual composition** is the key synthesis: analyze like TT-Forge, execute like Blaze. The adapter computes parameters; existing `emit()` calls consume them, preserving Blaze's kernel developer workflow.

## Source Files

- MLIR repository: `mlir/include/mlir/Dialect/Linalg/IR/LinalgOps.td` -- linalg op definitions
- MLIR repository: `mlir/lib/Dialect/Bufferization/Transforms/Bufferize.cpp` -- bufferization pass
- MLIR repository: `mlir/include/mlir/Dialect/MemRef/IR/MemRefOps.td` -- memref operations
- PyTorch repository: `torch/_dynamo/` -- TorchDynamo FX graph capture
- PyTorch repository: `torch/_inductor/codegen/triton.py` -- Inductor Triton codegen
- PyTorch repository: `torch/_inductor/select_algorithm.py` -- tile size selection heuristics
- TT-Forge repository: TTIR/TTNN dialect definitions (Tenstorrent internal)
- Blaze parallel: `blaze/graph.py` -- BlazeGraph (analogous to FX graph)
- Blaze parallel: `blaze/compiler.py` -- BlazeCompiler (analogous to Inductor backend)
- Blaze parallel: `blaze/fused_program.py` -- FusedProgram (the target of TensorAdapter's lowering)

---

**Next:** [05 -- Synthesized Design Principles](./05_synthesized_design_principles.md)
