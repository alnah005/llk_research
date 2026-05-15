# 3.2 Translation Challenges

Building on the lifecycle comparison in [Section 3.1](01_blaze_vs_ttnn_lifecycles.md), this section translates the structural mismatches into seven concrete engineering challenges that any registration strategy must solve. Each challenge is analyzed in four parts: the **problem** (what TTNN assumes vs what Blaze does), the **structural cause** (why the mismatch exists), a **failure-mode example** (what would break under a naive approach), and the **implication for the adapter**. The section concludes with the "100+ ops problem," a Blaze-to-TTNN concept mapping table, and a challenge-to-adapter-requirement summary.

---

## 3.2.1 Challenge 1: The Language Boundary

### Problem

Blaze's program-construction logic -- CB allocation, CT arg computation, semaphore assignment, kernel source generation, multi-phase composition -- is implemented entirely in Python. TTNN's `DeviceOperationConcept` expects this logic to live in C++ program factories. No mechanism exists to call Blaze's Python `emit()` from inside a C++ `ProgramFactory::create()` without embedding the Python interpreter.

### Structural Cause

TTNN's `device_operation::launch<>` template calls C++ factories in a context where the GIL is typically released. A factory cannot call `PyObject_CallFunction()` without reacquiring the GIL, importing the Blaze module, and invoking `emit()` -- a path that introduces interpreter re-entrancy, GIL contention, and a hard dependency on the Python runtime from deep inside the C++ dispatch path.

### What Would Break

```cpp
// BROKEN: Naive attempt to call Python emit() from C++ factory
CachedMeshWorkload<SharedVars> create_at(
    const MeshCoordinate& coord,
    const operation_attributes_t& attrs,
    const tensor_args_t& tensors) {

    py::gil_scoped_acquire gil;  // DEADLOCK: caller may already hold GIL
    py::object blaze_matmul = py::module_::import("blaze.ops.matmul").attr("Matmul");
    py::object fused_program = py::module_::import("blaze.fused_program")
        .attr("FusedProgram")(...);
    blaze_matmul.attr("emit")(fused_program, ...);  // Full Python emit() path
    // ... extract ProgramDescriptor from fused_program ...
}
```

This fails for three reasons:
1. **GIL deadlock/serialization.** The Python thread that called `ttnn.blaze.matmul(...)` may hold the GIL. Re-acquisition serializes all TTNN operations.
2. **Object lifetime.** The `ProgramDescriptor` contains raw `Buffer*` pointers. If Python-side `Tensor` objects are garbage-collected after the factory returns but before the program is enqueued, those pointers become dangling.
3. **Redundant recompilation.** On a cache miss, the factory re-executes `compose()` and the full engine pipeline. On a cache hit, it would need to patch runtime args without calling `compose()` -- but `compose()` has no "patch" mode.

### Adapter Requirement

The adapter accepts a *pre-built* `MeshProgramDescriptor` as its factory input -- never calling back into Python. Blaze continues to own program construction in Python; the adapter wraps the result for TTNN dispatch.

---

## 3.2.2 Challenge 2: Stateful vs Stateless Compilation

### Problem

Blaze's `FusedProgram` is a stateful accumulator. Each `emit()` call appends CB descriptors, CT arg tuples, semaphore allocations, and shadow graph nodes. The final program is the aggregate result of all `emit()` contributions. TTNN factories are stateless pure functions: they receive immutable `(operation_attributes_t, tensor_args_t)` and produce a `Program` in a single invocation.

### Structural Cause

The `emit()` pattern exists because Blaze FusedOps compose multiple MicroOps into a single fused kernel. Consider a fused RMSNorm + Linear op:

```python
class RMSNormLinear(FusedOp):
    @classmethod
    def compose(cls, f, tensors, output, user_args):
        # First emit: allocates CBs 0-2, CT args for RMSNorm phase
        normed = RMSNorm.emit(f, tensors["x"], ...)
        # Second emit: allocates CBs 3-5, reuses semaphore from phase 1
        result = Matmul.emit(f, normed, tensors["weight"], output, ...)
        return result
```

After both `emit()` calls, `f` contains 6 CBs, two sets of CT args, shared semaphores, and cross-references between phases. The `ProgramDescriptor` produced by `f.build()` is a flat snapshot of this accumulated state. A TTNN factory would need to reproduce this accumulation as a single function call -- which would require embedding the entire composition graph in `operation_attributes_t`.

### What Would Break

```cpp
// BROKEN: separate factories per phase -- cross-phase state is lost
auto phase1 = RMSNormFactory::create(attrs_phase1, args);  // CB 0-2
auto phase2 = MatmulFactory::create(attrs_phase2, args);    // CB 3-5, but how
                                                             // does it reference
                                                             // phase 1's output CB?
// The second factory has no way to reference CB handles from the first,
// because TTNN factories are independent, stateless invocations.
```

### Adapter Requirement

The adapter treats Blaze's assembled `ProgramDescriptor` as an opaque, atomic artifact. It cannot decompose the descriptor back into constituent MicroOp programs because CB IDs are shared, CT arg namespaces are merged, and the kernel source includes phase-switching logic for all phases.

---

## 3.2.3 Challenge 3: Output Tensor Management

### Problem

Blaze pre-allocates output tensors before compilation. The output tensor's `Buffer*` pointer is used during `emit()` to create tensor-backed CBs via `cb_from_tensor()`. By the time `generic_op` is called, the output tensor already exists and its address is baked into the `ProgramDescriptor`.

TTNN expects `create_output_tensors()` to handle allocation during `launch<>()`, *after* validation and output spec computation but *before* the factory runs.

### Structural Cause

Blaze needs raw `Buffer*` pointers at composition time because CB descriptors reference tensor memory directly. This creates a chicken-and-egg problem: Blaze needs the output address to build the program, but TTNN wants to build the program before allocating the output.

### What Would Break

```cpp
// BROKEN: Framework allocates output, but ProgramDescriptor references old address
tensor_return_value_t create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensors) {
    auto spec = compute_output_specs(attrs, tensors);
    return create_device_tensor(spec, ...);  // New tensor at address 0x7f0056780000
}

// Meanwhile, attrs.mesh_program_descriptor contains:
//   CBDescriptor { .buffer = 0x7f0012340000 }  <-- old pre-allocated address!
// The program would read from or write to the WRONG memory location.
```

This produces silent data corruption or a hardware fault. The `ProgramDescriptor` does not label which CBs are "output CBs" -- that information was lost at the metadata discard boundary.

### Adapter Requirement

The adapter preserves Blaze's pre-allocation model. Its `create_output_tensors()` returns the pre-allocated output tensor unchanged (matching `GenericOpDeviceOperation`'s current behavior). Its `compute_output_specs()` returns the spec of that pre-allocated tensor, enabling graph tracing to introspect the output shape without executing the op:

```cpp
// Adapter: preserves pre-allocation, but exposes specs for graph tracing
spec_return_value_t compute_output_specs(
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args) {
    return tensor_args.output_tensor.tensor_spec();  // meaningful spec
}
```

This means the adapter does not gain TTNN's automatic output management, but it avoids the descriptor-patching problem entirely. Full resolution of deferred output allocation requires architectural changes to Blaze's compilation model -- a topic for [Chapter 8](../ch08_build_migration_alternatives/index.md).

---

## 3.2.4 Challenge 4: Multi-Phase Programs

### Problem

A Blaze `FusedOp` produces a single `ProgramDescriptor` containing one unified kernel with multiple *phases*. Each phase corresponds to a constituent MicroOp's `emit()` call and shares CBs, semaphores, and CT arg namespaces with other phases. TTNN's factory model produces one `Program` per `create_at()` invocation, with no native concept of intra-program phases.

### Structural Cause

TTNN achieves multi-op execution through *graph-level composition*: dispatch `ttnn::rmsnorm`, then dispatch `ttnn::matmul` as separate programs. Blaze achieves it through *kernel-level fusion*: a single `Program` with one kernel binary containing all phases. The Blaze approach enables data to flow through L1 between phases without round-tripping through DRAM -- the core performance advantage of fused ops.

### What Would Break

If you tried to register each phase as a separate TTNN op:

```cpp
// BROKEN: Per-phase registration destroys fusion benefits
// Phase 1: read input into CB 0, compute partial result into CB 1
auto phase1_program = Phase1Factory::create_at(coord, attrs1, args);

// Phase 2: read new data into CB 0 (REUSE!), accumulate into CB 1
auto phase2_program = Phase2Factory::create_at(coord, attrs2, args);
// PROBLEM: CB 0 belongs to phase1_program's namespace.
//          phase2_program cannot reuse it -- it has its own CB 0.
// PROBLEM: Semaphore coordination between phases requires a shared
//          namespace. Separate programs have independent semaphore pools.
// PROBLEM: Ordering depends on command queue scheduler, not sequential code.
```

The phases are inherently coupled, and splitting them into separate ops destroys the resource sharing that makes fusion profitable.

### Adapter Requirement

The adapter registers at the **FusedOp granularity**. A `FusedOp` like RMSNorm becomes one registered operation (`"ttnn::blaze::rmsnorm"`), and its multi-phase kernel is the program produced by a single factory call. The factory receives the entire `ProgramDescriptor` -- containing all phases -- and wraps it into a `MeshWorkload`. TTNN sees one operation, unaware of internal phases.

---

## 3.2.5 Challenge 5: Dynamic Kernel Sources

### Problem

TTNN ops reference static kernel source files compiled into the binary or available at known filesystem paths. Blaze generates kernel C++ source *dynamically* at compile time via `kernel_codegen.py`. The generated source depends on the composition graph, the CT arg schema, the CB layout, and per-core role flags. Two invocations of the "same" FusedOp with different grid configurations may produce different kernel sources.

### Structural Cause

Blaze generates kernel source because its composition model is open-ended. The set of possible FusedOp compositions is combinatorial: any sequence of MicroOps can be fused, producing a unique kernel. Pre-writing static kernels for every possible composition is infeasible. Code generation is the only scalable approach.

### What Would Break

```cpp
// BROKEN: Kernel source does not exist at build time
struct GatedReduceFactory {
    static constexpr const char* KERNEL_PATH =
        "???";  // Unknown -- depends on runtime graph structure!

    static CachedMeshWorkload<T> create_at(...) {
        auto kernel = CreateKernel(program,
            "blaze/generated/gated_reduce_fused/kernel.cpp",  // Does not exist!
            core_range, config);
    }
};
```

The path does not exist at TT-Metal build time because the kernel is generated at Blaze compile time. Even if the generated kernel were pre-built, its content changes whenever the FusedOp's composition changes.

### Adapter Requirement

The adapter's factory does not generate kernel sources -- it uses whatever Blaze provided in the `ProgramDescriptor`. The `KernelDescriptor` already contains the JIT-compiled kernel path. However, the adapter's `compute_program_hash()` must account for kernel source identity to ensure correct cache behavior: two invocations with different compositions must produce different hashes, while two invocations with the same composition but different tensor addresses should be cache hits.

---

## 3.2.6 Challenge 6: Per-Core Specialization

### Problem

Blaze uses `PerCoreCompileTimeDescriptor` to vary compile-time arguments on a per-core basis. Different cores receive different CT arg values -- enabling `constexpr if` branches, compile-time loop unrolling, and dead-code elimination per core group. TTNN's standard pattern uses uniform compile-time args across all cores and achieves per-core variation through runtime arguments.

### Structural Cause

Blaze uses per-core CT args because they enable kernel specialization that runtime args cannot achieve:

- `constexpr` loop bounds enable full unrolling (vs dynamic loops with runtime bounds)
- `constexpr if` enables dead-code elimination for inactive phases on specific cores
- Template parameters enable type-level specialization

### What Would Break

```cpp
// BROKEN: CT args vary per core, but CreateKernel takes one compile_args vector
auto kernel = CreateKernel(program, kernel_path, all_cores,
    config{.compile_args = {/* which core's args? */}});

// Workaround: multiple CreateKernel calls with different CT args
auto kernel_active = CreateKernel(program, kernel_path, active_cores,
    config{.compile_args = {1, full_count}});      // is_active=1
auto kernel_inactive = CreateKernel(program, kernel_path, inactive_cores,
    config{.compile_args = {0, 0}});               // is_active=0
```

This is exactly what Blaze's `UnifiedKernelDescriptor` does internally -- it produces multiple `KernelDescriptor` entries when per-core CT arg variation exists. But a naive TTNN factory that tries to create a single kernel per RISC would fail.

### Adapter Requirement

The adapter delegates per-core CT arg handling to `ProgramBuilder`, which iterates all `KernelDescriptor` entries in the `ProgramDescriptor` and creates one kernel per unique CT arg configuration. This inherits `GenericMeshProgramFactory`'s existing behavior unchanged.

---

## 3.2.7 Challenge 7: The Engine Pipeline

### Problem

Blaze's `CBEngine`, `SemEngine`, and `CTArgEngine` form an integrated resource-allocation pipeline that operates on the `BlazeGraph` IR. The engines make globally-optimal decisions: `CBEngine` performs liveness analysis to identify temporal CB reuse opportunities; `SemEngine` assigns semaphore IDs globally across all synchronization points; `CTArgEngine` resolves cross-op references. TTNN has no equivalent orchestration layer -- each factory allocates resources inline.

### Structural Cause

The engine pipeline exists because the same engines serve all ops. `CBEngine`'s allocation algorithm works for matmul, RMSNorm, attention, and every other op -- it operates on graph topology, not op-specific logic. TTNN does not need this layer because each factory is self-contained and knows exactly what resources it requires.

### What Would Break

```cpp
// BROKEN: No graph-level optimization, no temporal reuse
CachedMeshWorkload<T> create_at(...) {
    Program program;

    // Phase 1: Copy
    auto cb_copy_in = CreateCircularBuffer(program, ...);   // CB 0
    auto cb_copy_out = CreateCircularBuffer(program, ...);  // CB 1

    // Phase 2: RMSNorm
    auto cb_norm_in = CreateCircularBuffer(program, ...);   // CB 2 -- could reuse CB 0!
    auto cb_norm_scratch = CreateCircularBuffer(program, ...); // CB 3
    auto cb_norm_out = CreateCircularBuffer(program, ...);  // CB 4

    // Phase 3: Matmul
    auto cb_mat_in = CreateCircularBuffer(program, ...);    // CB 5 -- could reuse CB 2!
    auto cb_mat_out = CreateCircularBuffer(program, ...);   // CB 6

    // Total: 7 CBs, using 7 * page_size bytes of L1
    // Blaze's CBEngine would produce: 4 CBs via temporal reuse
}
```

Without `CBEngine`'s liveness analysis, the factory allocates 7 CBs instead of 4. For L1-constrained operations (common on Tenstorrent hardware where L1 SRAM is 1.5 MB per core), this can be the difference between a program that fits and one that does not. Reproducing `CBEngine`'s behavior in C++ would require implementing a graph IR, liveness analysis, and the same compaction algorithm -- equivalent to porting the Blaze compiler to C++.

### Adapter Requirement

The adapter accepts the engine pipeline's output (the optimized CB/semaphore/CT-arg assignments already embedded in the `ProgramDescriptor`) rather than attempting to reproduce the pipeline in C++. However, this means the adapter cannot re-run engine logic on a cache hit with changed parameters. Shape changes always require a full recompilation in Python.

---

## 3.2.8 The "100+ Ops Problem"

The seven challenges above describe what must be solved for a *single* Blaze op. The problem compounds when considering the full Blaze op catalog.

### Scale

| Category | Count | Examples |
|----------|------:|---------|
| **MicroOps** | ~80+ | `Matmul`, `Copy`, `Reduce`, `Mcast`, `Gather`, `BinaryElementwise`, `Transpose`, `TopK` |
| **FusedOps** | ~30+ | `RMSNorm`, `SDPA`, `GatedMLP`, `RotaryEmbedding`, `MoEGate`, `FusedAttention` |
| **User-defined FusedOps** | unbounded | Custom compositions created by model authors at runtime |
| **Total** | **~112+** | Growing as new models require new ops |

### Per-Op C++ Cost

If each op required a hand-written C++ `DeviceOperation` struct:

| Component | Lines per MicroOp | Lines per FusedOp |
|-----------|------------------:|------------------:|
| `operation_attributes_t` + type aliases | 20--50 | 50--100 |
| `validate_on_program_cache_miss()` | 30--100 | 50--120 |
| `compute_output_specs()` | 10--30 | 30--60 |
| `create_output_tensors()` | 5--15 | 5--15 |
| `compute_program_hash()` | 10--20 | 30--60 |
| Factory `create_mesh_workload()` | 100--300 | 200--400 |
| `override_runtime_arguments()` | 20--50 | 30--60 |
| Nanobind bindings | 20--40 | 20--40 |
| **Total** | **~250--600** | **~400--850** |

### Aggregate Cost

At midpoint estimates:

$$80 \times 425 + 30 \times 625 \approx 34{,}000 + 18{,}750 = 52{,}750 \text{ lines of C++}$$

This does not include testing (each factory needs functional tests against the Python reference), maintenance (every `emit()` change requires a corresponding C++ update), or FusedOp composition complexity (FusedOps like `PagedAttention` involve 5--10 chained MicroOps with complex CB sharing).

### The Dynamic Composition Problem

FusedOps make the per-op approach even harder. A FusedOp's "attributes" are not a fixed struct -- they are the *composition graph itself*. Two invocations of `gated_reduce` with different internal configurations produce different programs. A dedicated C++ factory would need to encode the full composition logic. This is not a factory in the TTNN sense -- it is a compiler, precisely what Blaze already is, but in C++.

### The Adapter as Scaling Solution

The adapter pattern reduces per-op cost from hundreds of lines to approximately 5:

```cpp
// One C++ template handles ALL Blaze ops
template <typename OpTag>
struct BlazeDeviceOperationAdapter {
    // operation_attributes_t wraps MeshProgramDescriptor + op identity
    // tensor_args_t wraps io_tensors + output_tensor
    // Factory delegates to GenericMeshProgramFactory
    // Op-specific identity comes from OpTag::name
};

// Per-op cost: ONE tag struct + ONE registration
struct RMSNormTag { static constexpr const char* name = "blaze::rmsnorm"; };
constexpr auto blaze_rmsnorm =
    register_operation<"ttnn::blaze::rmsnorm",
                        BlazeDeviceOperationAdapter<RMSNormTag>>();
```

| Approach | Per-Op C++ | Total for 112 Ops | Maintenance |
|----------|-----------|-------------------|-------------|
| Dedicated `DeviceOperation` per op | 250--850 lines | ~53K lines | Every `emit()` change requires C++ update |
| Generic adapter + per-op tag | ~5 lines | ~600 lines + adapter template | Only adapter template needs maintenance |
| `generic_op` (status quo) | 0 lines | 0 lines | None (but no registration benefits) |

**Note on template instantiation:** Each `BlazeDeviceOperationAdapter<OpTag>` instantiation triggers a full `device_operation::launch<>` template expansion. With 112+ ops, this adds approximately 112 template instantiations to the TTNN build. Strategies to mitigate build-time impact (explicit template instantiation, unity builds, opaque-pointer indirection) are discussed in [Chapter 5](../ch05_op_registration/index.md).

---

## 3.2.9 Blaze-to-TTNN Concept Mapping Table

The following table maps every significant Blaze abstraction to its TTNN equivalent, identifies the gap, classifies which challenge is responsible, and notes whether the gap is bridgeable by the generic adapter.

| Blaze Concept | TTNN Equivalent | Gap | Challenge | Bridgeable? |
|---------------|-----------------|-----|:---------:|-------------|
| `BlazeOp` class (Python) | `DeviceOperation` struct (C++) | Language: Python vs C++; class vs concept conformance | 1 | Adapter wraps output, not class |
| `emit()` / `compose()` | `ProgramFactory::create()` | Stateful builder vs stateless function | 2 | Adapter receives pre-built descriptor |
| `FusedProgram` (stateful builder) | No equivalent | No cross-op accumulation in TTNN | 2 | Adapter treats final descriptor as atomic |
| `BlazeProgram.cb_from_tensor()` | `CreateCircularBuffer()` | Same operation, different timing and language | 1, 7 | Adapter inherits Blaze's CB assignments |
| `CBEngine.assign()` | Inline CB logic in factory | Graph-level optimization vs per-factory inline | 7 | Adapter inherits engine outputs |
| `SemEngine.assign()` | `CreateSemaphore()` | Graph-level assignment vs per-factory inline | 7 | Adapter inherits engine outputs |
| `CTArgEngine` | `ComputeConfig{.compile_args}` / `SetRuntimeArgs()` | Cross-op resolution vs direct positional args | 7 | Adapter inherits resolved CT args |
| `PerCoreCompileTimeDescriptor` | Uniform `compile_args` + per-core `SetRuntimeArgs()` | Per-core CT variation vs per-core RT variation | 6 | Adapter preserves per-core CT descriptors |
| `kernel_codegen.py` (dynamic) | Static `.cpp` kernel files | Runtime codegen vs build-time compilation | 5 | Adapter passes through kernel path |
| `UnifiedKernelDescriptor` | Per-RISC `CreateKernel()` calls | Unified abstraction vs explicit per-RISC calls | 5, 6 | Adapter passes through kernel descriptors |
| `ProgramDescriptor` | `Program` (built by factory) | Specification (data) vs artifact (live object) | 2 | `ProgramBuilder` converts |
| `MeshProgramDescriptor` | `MeshWorkload` (built by factory) | Pre-built vs factory-constructed | 2 | Adapter's factory wraps descriptor |
| Pre-allocated output tensor | `create_output_tensors()` | Caller-allocated vs framework-allocated | 3 | Adapter returns pre-allocated tensor |
| `compute_output_specs()` passthrough | `compute_output_specs()` inference | Returns existing spec vs derives from attributes | 3 | Adapter returns tensor spec |
| Multi-phase kernel (via `compose()`) | Single program per `create_at()` | No intra-program phase concept in TTNN | 4 | Adapter wraps entire FusedOp |
| `BlazeCompiler.compile()` | `device_operation::launch<>()` | Entire pipeline vs single dispatch call | 1, 7 | Adapter accepts pipeline output |
| `BlazeOp._class_registry` | `register_operation<>` (constexpr) | Python runtime dict vs C++ compile-time registration | 1 | Codegen script can bridge |
| `OpSpec` (ports, CT arg schema) | `operation_attributes_t` + `tensor_args_t` | Declarative metadata vs structural contracts | 1 | Per-op work for full semantic mapping |
| `custom_program_hash` (optional) | `compute_program_hash()` (per-op method) | Opt-in field vs required method | 2 | Adapter bridges via hash logic |
| `io_tensors` (flat vector) | `tensor_args_t` (typed struct) | Positional convention vs named fields | 3 | Adapter uses flat convention |
| `BlazeGraph` (dataflow IR) | No equivalent at op level | Per-op graph vs no internal graph | 7 | Not bridgeable at adapter level |
| `shadow_graph` (for codegen) | No equivalent | Runtime graph for code generation | 5, 7 | Adapter passes through generated kernel |
| `_lifetime_tensors` (GC pinning) | RAII via `CachedProgram<>` | Manual pinning vs framework-managed | 3 | Adapter must preserve pinning pattern |
| `FusedOp.compose()` (composition) | No equivalent | TTNN has no multi-phase composition | 4 | Adapter treats composed result as atomic |
| `BlazeProgram.validate()` | `validate_on_program_cache_miss()` | Python pre-dispatch vs C++ in-dispatch | 1 | Adapter can add C++ validation hooks |
| `MeshCompiledProgram` (reusable) | Program cache entry | Explicit reuse object vs implicit cache | 2 | Adapter uses cache |

---

## 3.2.10 Summary: Challenge Hierarchy and Adapter Requirements

The seven challenges are not independent. They form a dependency hierarchy:

```
Challenge 1 (Language Boundary)  <-- foundational
    |
    +---> Challenge 2 (Stateful vs Stateless)
    |         |
    |         +---> Challenge 4 (Multi-Phase Programs)
    |         |         |
    |         |         +---> Challenge 7 (Engine Pipeline)
    |         |
    |         +---> Challenge 3 (Output Tensor Management)  [partially independent]
    |
    +---> Challenge 5 (Dynamic Kernel Sources)
    |
    +---> Challenge 6 (Per-Core Specialization)
              |
              +---> Challenge 7 (Engine Pipeline)
```

Each challenge maps to a specific adapter requirement:

| # | Challenge | Adapter Requirement |
|---|-----------|-------------------|
| 1 | Language boundary | Factory accepts pre-built `MeshProgramDescriptor`; never calls back into Python |
| 2 | Stateful vs stateless | `operation_attributes_t` wraps the already-built descriptor; factory is a pass-through |
| 3 | Output tensor management | `create_output_tensors()` returns caller's tensor; `compute_output_specs()` reads spec from it |
| 4 | Multi-phase programs | Registers at FusedOp granularity; entire multi-phase descriptor is one factory output |
| 5 | Dynamic kernel sources | Factory passes kernel source strings through to `CreateKernel()`; semantic hash avoids $O(n)$ source hashing |
| 6 | Per-core specialization | Delegates per-core CT arg handling to `ProgramBuilder`, inheriting existing behavior |
| 7 | Engine pipeline | Engines continue to run in Python; adapter receives their output via the descriptor |
| 8 | 100+ ops scaling | Adapter is a parameterized template; per-op cost is one tag struct (~5 lines) |

These requirements are the specification for the `BlazeDeviceOperationAdapter<OpTag>` template designed in [Chapter 4](../ch04_adapter_pattern/index.md).

---

## Key Takeaways

- **Challenge 1 (language boundary) is foundational** — it eliminates any approach requiring Python callbacks from C++ factories. All other challenges are consequences of or compounded by this boundary.

- **Challenges 2 + 4 (statefulness + multi-phase) together force the adapter to treat `ProgramDescriptor` as atomic.** Decomposing a FusedOp's multi-phase program into separate TTNN factory calls would break CB and semaphore sharing.

- **Challenge 7 (engine pipeline) is the most expensive to solve natively.** Porting `CBEngine`, `SemEngine`, and `CTArgEngine` to C++ would be equivalent to rewriting the Blaze compiler.

- **At ~53,000 lines of C++ (midpoint), per-op porting is infeasible at scale.** Only a parameterized adapter template — ~5 lines per op — can cover 112+ ops without unsustainable maintenance burden.

---

| [Section 3.1: Blaze vs TTNN Lifecycles](01_blaze_vs_ttnn_lifecycles.md) | **Section 3.2** | [Chapter 4: The Adapter Pattern](../ch04_adapter_pattern/index.md) |
|:---|:---:|---:|
