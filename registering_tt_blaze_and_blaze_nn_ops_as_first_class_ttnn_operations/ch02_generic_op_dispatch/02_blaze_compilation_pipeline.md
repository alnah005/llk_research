# 2.2 Blaze Compilation Pipeline

## Opening Summary

Blaze's compilation pipeline transforms a high-level op composition (expressed through `emit()` calls on a `FusedProgram`) into a `MeshProgramDescriptor` that `ttnn.generic_op()` can execute. The pipeline has three distinct phases -- Composition, Compilation, and Execution -- anchored by two key abstractions: `BlazeProgram` (the descriptor builder) and `BlazeCompiler` (the mesh-aware orchestrator). This section traces each phase in detail, covering the `BlazeOp` hierarchy, the engine stack, kernel codegen, `ProgramDescriptor` assembly, and the metadata discard point where Blaze's rich Python-level semantics are collapsed into an opaque C++ descriptor.

---

## 2.2.1 Pipeline Overview

| Phase | Entry Point | Output | Owner |
|-------|------------|--------|-------|
| **1. Composition** | `MicroOp.emit(f, ...)` / `FusedOp.compose(f, ...)` | Populated `FusedProgram` with CBs, CT args, role flags | Op author |
| **2. Compilation** | `BlazeCompiler.compile(graph, tensors, ...)` | `MeshCompiledProgram` wrapping a `MeshProgramDescriptor` | Compiler |
| **3. Execution** | `MeshCompiledProgram.run()` | Output `ttnn.Tensor` | `ttnn.generic_op` |

```
Phase 1: Composition         Phase 2: Compilation              Phase 3: Execution
====================         =====================             ===================
BlazeOp.compose_fn(f, ...)   BlazeCompiler.compile()           MeshCompiledProgram.run()
  --> emit() calls             --> Engine pipeline               --> ttnn.generic_op()
  --> BlazeProgram built       --> ProgramDescriptor(s)          --> device dispatch
                               --> MeshProgramDescriptor
```

The pipeline is implemented across several files:

| Component | File | Role |
|-----------|------|------|
| `BlazeOp` | `blaze/blaze_op.py` | Op definition (ports, CT args, emit/compose) |
| `BlazeGraph` | `blaze/graph.py` | Dataflow IR (nodes, edges, tensor ports) |
| `FusedProgram` | `blaze/fused_program.py` | Composition context (emit calls build the graph and program) |
| `BlazeProgram` | `blaze/program.py` | Low-level program builder (CB, CT args, descriptors) |
| `BlazeCompiler` | `blaze/compiler.py` | Mesh-aware compilation orchestrator |
| `CBEngine` | `blaze/cb_engine.py` | CB slot assignment |
| `SemEngine` | `blaze/sem_engine.py` | Semaphore assignment |
| `CTArgEngine` | `blaze/ct_args.py` | Compile-time argument generation |
| `UnifiedKernelDescriptor` | `blaze/unified_kernel_descriptor.py` | Multi-RISC kernel descriptor builder |

---

## 2.2.2 Phase 1: Composition via `emit()`

### The `BlazeOp` Hierarchy

Every Blaze op inherits from `BlazeOp` and falls into one of two categories:

```
BlazeOp (abstract)
  +-- MicroOp    (backed by a C++ op struct; smallest unit of kernel logic)
  +-- FusedOp    (composes MicroOps via emit() chains)
```

The contract:

| Class | Must Override | Purpose |
|-------|-------------|---------|
| `MicroOp` | `emit(f, ...)`, `compose(cls, f, tensors, output, user_args)` | Atomic kernel operation; `emit()` adds CBs and CT args to a `FusedProgram` |
| `FusedOp` | `compose(cls, f, tensors, output, user_args)` | High-level op that chains `emit()` calls on constituent `MicroOp`s |

`BlazeOp.register()` auto-generates an `OpSpec` from port descriptors, a `CTArgSchema` from `ct_args` declarations (or auto-derived from the kernel `.hpp`), a `FusedOpConfig` if `compose()` is overridden, and `PhaseInfo` for kernel codegen. The class is stored in `BlazeOp._class_registry` keyed by `name`.

**Source file:** `blaze/blaze_op.py`

### What `emit()` Does

A `MicroOp.emit()` call on a `FusedProgram` `f` performs several actions:

1. **Allocates circular buffers** via `f.program.cb_from_tensor()`, `f.program.cb_scratch()`, or `f.program.cb_alias()`.
2. **Declares compile-time arguments** via `f.program.ncrisc_ct_args()`, `f.program.trisc_ct_args()`, etc.
3. **Sets per-core role flags** via `f.program.per_core_unified_ct_args()` (e.g., `is_active` flags for masking inactive cores).
4. **Allocates semaphores** via `f.semaphore(name)` for inter-core or inter-device synchronization.
5. **Records the op in the shadow graph** for kernel codegen and visualization.

A concrete example from `blaze/ops/copy/op.py`:

```python
class Copy(MicroOp):
    name: str = "copy"
    src: Input = Input()
    dst: Output = Output()

    @staticmethod
    def emit(f, src, output_tensor, *, prefix="copy", wait_src=False, ...):
        src_cb = f.cb_from_tensor(src) if not isinstance(src, CBHandle) else src
        dst_cb = f.cb_from_tensor(output_tensor)

        f.per_core_unified_ct_args([
            f.flag(f"{prefix}.is_active", src_cb.core_ranges),
        ])
        f.unified_ct_args([
            (f"{prefix}.src", src_cb),
            (f"{prefix}.dst", dst_cb),
            (f"{prefix}.num_pages", src_cb.num_pages),
            # ... additional CT args ...
        ])
        return f.output("copy", dst_cb, prefix=prefix, src=src_cb)
```

Each `emit()` call mutates the `FusedProgram` in-place, accumulating CB descriptors and CT arg tuples. Nothing runs on hardware until `run()` is called -- `emit()` is a builder pattern, not an execution.

### Shadow Graph Recording

Every `emit()` call implicitly records an `OpNode` in the `FusedProgram`'s shadow graph (`_shadow_graph: BlazeGraph`). This graph is used for:

1. **Kernel codegen** -- When `FusedProgram.build()` finds `self.program._kernel is None`, it generates a kernel from the shadow graph using `kernel_codegen.generate_kernel_file()`.
2. **Visualization** -- `blaze.export()` serializes the graph for interactive inspection.
3. **CB compaction** -- The shadow graph tracks CB lifetimes for temporal reuse.

**Source file:** `blaze/fused_program.py`

---

## 2.2.3 `BlazeProgram` -- The Descriptor Builder

`BlazeProgram` (`blaze/program.py`) is the core builder that accumulates all the data needed for a `ProgramDescriptor`. It is the mechanical heart of the pipeline.

### CB Allocation Methods

Every CB in a Blaze program is allocated through `BlazeProgram`:

| Method | Purpose | Tensor-backed? |
|--------|---------|----------------|
| `cb_from_tensor(tensor)` | Allocate CB backed by sharded tensor L1 | Yes |
| `cb_from_tensor_overlapped(tensor, offset, size, page_size)` | CB for a sub-region of a tensor's L1 | Yes |
| `cb_output(tensor)` | CB backed by tensor, marked as output | Yes |
| `cb_scratch(name, num_pages, like=...)` | Scratch CB, not tensor-backed | No |
| `cb_alias(source_cb_id, dtype=...)` | Alias format on existing CB | Shares source |

Each allocation increments `_next_cb_id` and appends a `ttnn.CBDescriptor` to `_cb_descriptors`. The CB ID counter is sequential and deterministic.

### Compile-Time and Runtime Arg Management

Args are accumulated in per-RISC lists:

```python
# Per-RISC CT arg lists
self._ncrisc_args: list[tuple[str, int]]  # Named CT args for NCRISC
self._brisc_args:  list[tuple[str, int]]  # Named CT args for BRISC
self._trisc_args:  list[tuple[str, int]]  # Named CT args for TRISC

# Per-RISC RT arg lists
self._ncrisc_named_rt_args: list[tuple[str, int]]
self._brisc_named_rt_args:  list[tuple[str, int]]
self._trisc_named_rt_args:  list[tuple[str, int]]

# Per-core CT args (role flags, per-core indices)
self._core_descriptors:     list[UnifiedCompileTimeCoreDescriptor]
self._per_core_descriptors: list[PerCoreCompileTimeDescriptor]
```

Helper methods route args to the appropriate lists:

```python
def unified_ct_args(self, args):    # -> all three RISC lists
def ncrisc_ct_args(self, args):     # -> NCRISC only
def unified_rt_args(self, args):    # -> all three RT lists
def ncrisc_rt_args(self, args):     # -> NCRISC RT only
```

### Semaphore Allocation

Two mechanisms are available:

1. **Global semaphores** -- `semaphore(device=...)` creates a `ttnn.GlobalSemaphore` on the mesh device and returns its L1 address. The object is stored in `_global_semaphores` to prevent GC.

2. **Program semaphores** -- `program_semaphore(core_ranges=...)` allocates a per-core slot ID (0..15), tracking usage in `_sem_ids_by_core` to allow reuse across disjoint core ranges.

### `build()` -- Assembling the `ProgramDescriptor`

```python
# Source: blaze/program.py
def build(self) -> ttnn.ProgramDescriptor:
    # 1. Resolve per-core RT args
    ncrisc_pc = self._resolve_per_core_rt_args(self._ncrisc_per_core_rt_args, all_cores)

    # 2. Build UnifiedKernelDescriptor
    ukd = UnifiedKernelDescriptor(
        kernel_source=self._kernel,
        core_ranges=self._core_ranges,
        ncrisc_named_compile_time_args=self._ncrisc_args,
        brisc_named_compile_time_args=self._brisc_args,
        trisc_named_compile_time_args=self._trisc_args,
        ncrisc_named_common_runtime_args=self._ncrisc_named_rt_args,
        # ... all other arg lists ...
        trisc_compute_config=self._compute_config,
        unified_compile_time_core_descriptors=self._core_descriptors,
        per_core_compile_time_descriptors=self._per_core_descriptors,
        defines=self._defines,
        noc_mode=self._noc_mode,
    )

    # 3. Get kernel descriptors (one per RISC)
    kernel_result = ukd.get_kernel_descriptors()

    # 4. Assemble ProgramDescriptor
    return ttnn.ProgramDescriptor(
        kernels=kernel_result.kernels,
        cbs=self._cb_descriptors,
        semaphores=self._semaphore_descriptors,
    )
```

The `UnifiedKernelDescriptor` is a Blaze abstraction that produces 1--3 `KernelDescriptor`s (one per active RISC processor: NCRISC for data movement, BRISC for data movement, TRISC for compute) from a single unified specification. It also generates C++ header files with `constexpr` accessors for named CT args, so kernels read `ct::my_arg` instead of positional `get_compile_time_arg_val(7)`.

### `run()` -- Direct Execution (Single-Device Shortcut)

```python
def run(self) -> ttnn.Tensor:
    sorted_tensors = [t for _, t in sorted(self._io_tensors, key=lambda x: x[0])]
    ttnn.generic_op(sorted_tensors, self.build())
    return self._output_tensors[0] if self._output_tensors else sorted_tensors[-1]
```

For single-device, single-op execution, `BlazeProgram.run()` builds the `ProgramDescriptor` and calls `ttnn.generic_op()` directly, without going through the compiler's mesh decomposition.

**Source file:** `blaze/program.py`

---

## 2.2.4 `FusedProgram` -- The Composition Context

`FusedProgram` (`blaze/fused_program.py`) wraps a `BlazeProgram` and provides the higher-level API that `emit()` functions use. It adds:

- **Device context** -- Grid config, NOC translation, worker core mapping
- **Shadow graph** -- Tracks the decomposition of fused ops into micro-ops for kernel codegen
- **Tensor view management** -- `OverlappedView` for sub-regions of L1 tensors
- **Convenience methods** -- `flag()`, `output()`, `cb_from_view()`, `named_tensor()`, `semaphore()`
- **Disjoint-share optimization** -- Reuses CB IDs across micro-ops on non-overlapping core ranges

When `emit()` calls like `Copy.emit(f, ...)` run, they call methods on the `FusedProgram` `f`, which delegates to `f.program` (a `BlazeProgram`) for the actual accumulation.

The `build()` method assembles the `ProgramDescriptor` via the underlying `BlazeProgram`:

```python
# Source: blaze/fused_program.py
def build(self, noc_mode=None) -> CompiledProgram:
    self._prepare_for_build()     # CB compaction, multi-phase reconfig
    if self.program._kernel is None:
        self.program.set_kernel_from_graph(self._shadow_graph, name=self._name)
    pd = self.program.build()
    io_tensors = [t for _, t in sorted(self.program._io_tensors, key=lambda x: x[0])]
    return CompiledProgram(pd, io_tensors, output_tensor=..., global_semaphores=...)
```

**Source file:** `blaze/fused_program.py`

---

## 2.2.5 `BlazeCompiler` -- Mesh-Aware Compilation

The `BlazeCompiler` (`blaze/compiler.py`) is the standard compilation entry point for production workloads. It handles multi-device (mesh) compilation.

### Step-by-Step Pipeline

```
BlazeCompiler.compile(graph, tensors, output_tensor, ...)
    |
    [1] Reset per-compile state (sem_dict, tensor_dict, internal_tensor_dict)
    |
    [2] Run device-agnostic engines:
    |     cb_assignments = CBEngine().assign(graph)
    |     sem_assignments = SemEngine().assign(graph)
    |
    [3] Auto-create global semaphores if op declares num_global_semaphores
    |
    [4] Split mesh tensors into per-device tensors:
    |     per_device_inputs = _split_tensors(tensors)
    |     per_device_outputs = ttnn.get_device_tensors(output_tensor)
    |
    [5] For each device (row, col):
    |     |
    |     +-- Merge user_args with per_device_overrides and mesh context
    |     |     merged_args = {mesh_coord, mesh_shape, mesh_device, ...}
    |     |
    |     +-- _compile_single(ctx, graph, dev_tensors, dev_output, ...)
    |     |     |
    |     |     +-- [fused op?] _compile_fused_op()
    |     |     |       +-- FusedProgram(kernel, device, ...)
    |     |     |       +-- config.compose_fn(f, tensors, output, args)
    |     |     |       +-- f.build()  --> CompiledProgram
    |     |     |
    |     |     +-- [engine path?] _compile_for_device()
    |     |           +-- Build CBs from engine assignments
    |     |           +-- Build CT args via CTArgEngine
    |     |           +-- Build UnifiedKernelDescriptor
    |     |           +-- ttnn.ProgramDescriptor(kernels, cbs)
    |     |
    |     +-- mesh_pd[MeshCoordinateRange(row,col)] = compiled.program_descriptor
    |
    [6] Assemble MeshCompiledProgram:
          MeshCompiledProgram(
              mesh_program_descriptor=mesh_pd,
              io_tensors=mesh_io_tensors,
              output_tensor=output_tensor,
              lifetime_tensors=...,
              lifetime_semaphores=...,
          )
```

### Dual Compilation Paths

`_compile_single()` dispatches between two paths:

| Path | Condition | Mechanism |
|------|-----------|-----------|
| **Fused Op** | Single-node graph with registered `FusedOpConfig` | `_compile_fused_op()`: creates `FusedProgram`, calls `compose()`, then `build()` |
| **Engine Path** | Multi-node graph or no `FusedOpConfig` | `_compile_for_device()`: runs CB/Sem/CT-arg engines, builds descriptors directly |

The fused-op path is dominant in production (most Blaze ops are single-node `FusedOp` or `MicroOp` classes with `compose()` registered). The engine path handles hand-crafted multi-node graphs.

**Source file:** `blaze/compiler.py`

### Engine Stack

The engine stack runs before per-device compilation and produces device-agnostic results:

| Engine | Input | Output | Purpose |
|--------|-------|--------|---------|
| `CBEngine` | `BlazeGraph` | `{key: CBAssignment}` | Assigns CB IDs to graph edges and internal buffers |
| `SemEngine` | `BlazeGraph` | `[SemAssignment]` | Maps sync protocols to semaphore IDs |
| `CTArgEngine` | Graph + CB map + sem map | `{risc: [(name, value)]}` | Generates named CT arg tuples per RISC |

#### `CBEngine`

Assigns monotonically increasing CB IDs to graph edges and external ports. The assignment categories:

- `ext_in_*` -- external input tensor ports (CB backed by sharded tensor L1)
- `intermed_*` -- intermediate buffers between fused ops (scratch allocations)
- `internal_*` -- op-declared internal scratch CBs
- `ext_out_*` -- external output tensor ports (CB backed by output tensor)

#### `SemEngine`

Assigns semaphore IDs to synchronization points in the graph. Each unique sync protocol gets a global semaphore, allocated on the mesh device.

#### `CTArgEngine`

Generates compile-time argument tuples for each RISC (NCRISC, BRISC, TRISC) by:

1. Walking the graph in topological order
2. For each node, consulting the `OpSpec`'s CT arg schema
3. Resolving CB references to actual CB IDs from `CBEngine`
4. Resolving semaphore references to L1 addresses from `SemEngine`
5. Applying user overrides

These engines implement the "graph API" compilation path, where the dataflow graph is the source of truth rather than imperative `emit()` calls.

---

## 2.2.6 IO Tensor Ordering

The `_build_io_tensors()` method constructs the tensor list that `generic_op` receives:

```python
# Source: blaze/compiler.py
def _build_io_tensors(self, graph, tensors, output_tensor):
    io_tensors = []
    added = set()
    # Input tensors in spec port order
    for node in graph.topological_order():
        for port in node.spec.input_ports:
            if (node.id, port.name) in graph.external_input_ports:
                for tensor_name, ports in graph.tensor_to_ports.items():
                    if (node.id, port.name) in ports and tensor_name not in added:
                        io_tensors.append(self._unwrap_tensor(tensors[tensor_name]))
                        added.add(tensor_name)
    # Output tensor last (generic_op convention)
    io_tensors.append(output_tensor)
    return io_tensors
```

This satisfies the invariant in `ttnn::prim::generic_op()` that `io_tensors.back()` is the output. The ordering convention is positional and fragile -- a mismatch between this ordering and the CB descriptors that reference the tensors would cause silent data corruption.

---

## 2.2.7 Execution via `ttnn.generic_op()`

Both `CompiledProgram.run()` and `MeshCompiledProgram.run()` converge to the same execution helper:

```python
# Source: blaze/compiler.py
def _run_program(descriptor, io_tensors, output_tensors, deallocate_tensors):
    ttnn.generic_op(io_tensors, descriptor)
    for t in deallocate_tensors:
        ttnn.deallocate(t)
    return output_tensors[0] if output_tensors else io_tensors[-1]
```

The `descriptor` is either a `ProgramDescriptor` (single-device) or a `MeshProgramDescriptor` (mesh). `ttnn.generic_op` resolves this via Python overload dispatch to the appropriate C++ `GenericOp::invoke()` overload (see [Section 2.1](./01_generic_op_internals.md)).

---

## 2.2.8 Lifetime Management

A critical concern in the Blaze pipeline is **tensor lifetime**. The `ProgramDescriptor` stores raw `Buffer*` pointers in `CBDescriptor.buffer`. If the Python-side `ttnn.Tensor` object is garbage-collected before the program executes, the pointer becomes dangling.

The `MeshCompiledProgram` addresses this with two pinning lists:

```python
class MeshCompiledProgram:
    _lifetime_tensors: list       # Per-device tensors whose Buffer* is in CBDescriptors
    _lifetime_semaphores: list     # GlobalSemaphore objects (PD holds only int addresses)
```

The compiler explicitly collects:
- All per-device `io_tensors` from each `_compile_single()` call
- Named tensors from `f.named_tensor()` (the `_tensor_dict`)
- Internal compiler tensors (the `_internal_tensor_dict`)
- All graph-engine and user-named semaphores

Without this pinning, use-after-free is a real risk -- it is documented in comments throughout `compiler.py`.

**Source file:** `blaze/compiler.py`

---

## 2.2.9 The Metadata Discard Point

The `ProgramDescriptor` construction is the **discard point** -- the boundary where Blaze's rich Python-level metadata is collapsed into an opaque C++ descriptor. The following table tracks what survives and what is lost:

| Metadata | Before Discard (Python) | After Discard (ProgramDescriptor) |
|----------|------------------------|-----------------------------------|
| **Op name** (e.g., "matmul", "rmsnorm") | Available via `OpNode.spec.name` | Lost -- all ops are `"ttnn::generic_op"` |
| **Graph topology** | Available via `BlazeGraph.edges` | Lost -- PD is a flat list of kernels/CBs |
| **Port names** (e.g., "input_weights") | Available via `OpSpec.input_ports` | Lost -- tensors are positional in `io_tensors` |
| **CB purpose** (e.g., "input weights CB") | Available via `CBAssignment.port_name` | Lost -- CBs are indexed by position |
| **Semaphore protocol** | Available via `SemAssignment.protocol` | Lost -- semaphores are integer addresses |
| **CT arg schema** | Available via `CTArgSchema` | Partially preserved in `named_compile_time_args` |
| **Per-op validation rules** | Available via `OpSpec` constraints | Lost -- `generic_op` validates only mesh coord uniqueness |

The named compile-time args partially survive (their names embed the op prefix, e.g., `"matmul.out_w"`), but the structured schema that validates their types, sources, and completeness is gone.

The information flow through the pipeline:

```
BlazeGraph                    Engine Pipeline               ProgramDescriptor
+------------------+          +-------------------+         +------------------+
| OpNode "matmul"  |   CBEngine                   |         | Kernels:
|   spec.name      |--------->| cb_id=0, port=in0 |-------->|   kernel_source
|   spec.kernel    |   SemEngine                   |         |   core_ranges
|   input_ports    |--------->| sem_id=0, proto=.. |-------->|   named_ct_args
|   output_ports   |   CTArgEngine                 |         | CBs:
|   kwargs         |--------->| matmul.out_w = 64  |-------->|   total_size
+------------------+          +-------------------+         |   format_descs
| OpNode "mcast"   |               |                        | Semaphores:
|   spec.is_inter  |               |                        |   core_ranges
|   sender kwarg   |               v                        |   initial_value
+------------------+          KernelCodegen                 +------------------+
| Edges            |               |                                |
|   matmul->mcast  |               v                                v
+------------------+          UnifiedKernelDescriptor        MeshProgramDescriptor
                                   |                                |
                              get_kernel_descriptors()               v
                                   |                         ttnn.generic_op()
                                   v                                |
                              KernelDescriptor x3                   v
                              (NCRISC, BRISC, TRISC)  GenericOpDeviceOperation.launch<>()
```

This discard boundary is the central architectural fact that motivates the first-class registration effort. Everything downstream of it -- profiling, graph tracing, cache hashing, validation -- operates on the impoverished `ProgramDescriptor` rather than the semantically rich `BlazeGraph`.

---

## 2.2.10 End-to-End Data Flow Diagram

The complete pipeline from user code to hardware execution:

```
User writes a Blaze op class (MicroOp / FusedOp)
    |
    | Op.register() -> OpSpec + CTArgSchema + FusedOpConfig
    v
User calls blaze.fuse() context or BlazeCompiler.compile()
    |
    v
BlazeCompiler.compile(graph, tensors, output_tensor, user_args)
    |
    |-- _split_tensors(tensors)          # mesh -> per-device
    |-- CBEngine().assign(graph)         # allocate CB IDs
    |-- SemEngine().assign(graph)        # allocate semaphore IDs
    |
    |-- FOR EACH device (row, col):
    |       |
    |       |-- _compile_single(ctx, graph, dev_tensors, dev_output, ...)
    |       |       |
    |       |       +-- Fused Op Path:
    |       |       |       FusedProgram(kernel, device, ...)
    |       |       |       config.compose_fn(f, tensors, output, args)
    |       |       |           |
    |       |       |           +-- MicroOp.emit(f, ...)   # CB alloc, CT args
    |       |       |           +-- MicroOp.emit(f, ...)   # more composition
    |       |       |           +-- ...
    |       |       |       f.build()
    |       |       |           |
    |       |       |           +-- BlazeProgram.build()
    |       |       |                   |
    |       |       |                   +-- UnifiedKernelDescriptor(...)
    |       |       |                   +-- .get_kernel_descriptors()
    |       |       |                   +-- ttnn.ProgramDescriptor(
    |       |       |                          kernels=..., cbs=..., semaphores=...)
    |       |       |
    |       |       +-- Engine Path:
    |       |               CTArgEngine().generate_tuples(graph, cb_for_ct, ...)
    |       |               UnifiedKernelDescriptor(...)
    |       |               ttnn.ProgramDescriptor(kernels=..., cbs=...)
    |       |
    |       +-- mesh_pd[MeshCoordinateRange(row,col)] = program_descriptor
    |
    +-- MeshCompiledProgram(mesh_pd, io_tensors, output_tensor,
    |       lifetime_tensors=..., lifetime_semaphores=...)
    |
    v
MeshCompiledProgram.run()
    |
    +-- ttnn.generic_op(io_tensors, mesh_program_descriptor)
            |
            +-- GenericOp::invoke()                [C++, generic_op.cpp]
            +-- ttnn::prim::generic_op()           [C++, generic_op_device_operation.cpp]
            +-- device_operation::launch<GenericOpDeviceOperation>(attrs, tensor_args)
            +-- GenericMeshProgramFactory::create_mesh_workload() or override_rt_args()
            +-- Program{ProgramDescriptor}         [tt-metalium]
            +-- Enqueue on device command queue
```

---

## 2.2.11 Kernel Codegen

When the `BlazeProgram` is initialized with a `graph` instead of a `kernel`, it triggers kernel codegen:

```python
if graph is not None and kernel is None:
    from .kernel_codegen import generate_kernel_file
    self._kernel = generate_kernel_file(graph)
```

The generated kernel file contains:
- Include directives for the CT arg header and RT arg header
- The op class instantiations from each `OpNode`'s `spec.op_class`
- A `run()` function that calls each op in topological order

The codegen path and the handwritten-kernel path both produce the same output: a `kernel_source` path that goes into `KernelDescriptor.kernel_source`.

---

## Key Takeaways

1. **Blaze owns the entire descriptor construction.** TTNN's `generic_op` is a pure dispatch mechanism with no semantic knowledge.

2. **The pipeline is fully Python-driven.** No C++ compilation logic runs until `Program{program_descriptor}` is constructed inside `GenericMeshProgramFactory::create_at()`.

3. **`emit()` is a builder pattern, not an execution.** Each `emit()` call mutates a `FusedProgram`/`BlazeProgram` in-place, accumulating descriptors. Nothing runs on hardware until `run()` is called.

4. **The compilation pipeline has two tiers:** Device-agnostic engines (CB/Sem assignment) run once; device-specific descriptor construction runs per mesh coordinate.

5. **The `ProgramDescriptor` is the discard boundary.** After construction, op names, graph topology, CB semantics, and validation rules are gone. `generic_op` receives only raw kernels, CBs, and semaphores. This discard is the fundamental motivation for first-class registration.

6. **Tensor lifetime management is manual.** `MeshCompiledProgram` must pin per-device tensors and `GlobalSemaphore` objects to prevent use-after-free between `compile()` and `run()`. This is a fragile pattern that first-class registration could automate.

---

| [Section 2.1: GenericOp Internals](01_generic_op_internals.md) | **Section 2.2** | [Section 2.3: Divergence Analysis](03_divergence_analysis.md) |
|:---|:---:|---:|
