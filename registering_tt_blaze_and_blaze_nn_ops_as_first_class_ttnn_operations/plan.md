# Final Plan: Registering TT-Blaze and blaze-nn Ops as First-Class TTNN Operations

---

## Selection Rationale

This final plan is a hybrid that draws from all five candidate plans, weighted by evaluator feedback:

- **Backbone from Plan v3.** Plan v3 was evaluated as the "strongest option among the five candidates" for its adapter-first progression, concrete end-to-end walkthroughs (matmul MicroOp, gated_reduce FusedOp), batch registration scalability discussion, and explicit open questions. Its chapter arc -- contract, current dispatch, adapter design, MicroOp registration, FusedOp registration, downstream integration, infrastructure, migration -- forms the structural skeleton of this final plan.

- **Gap analysis chapter from Plan v2.** All five evaluators identified the need for a dedicated gap analysis chapter. Plan v3's most significant weakness was jumping directly from "how generic_op works" to "adapter design" without systematically characterizing why the adapter is necessary. Plan v2's Chapter 3 ("The Compilation Model Gap") was rated as having the best analytical depth, enumerating seven specific translation challenges (language boundary, stateful vs stateless compilation, output tensor management, multi-phase programs, dynamic kernel sources, per-core specialization, engine pipeline). This is inserted as Chapter 3 in the final plan.

- **Precedents and existing patterns from Plans v1 and v5.** Plan v1 uniquely dedicated an entire chapter to precedents (experimental ops, moreh ops, Python-only registration, enhanced generic_op, plugin mechanisms). Plan v5 contributed real codebase references (`model_tracer/generic_ops_tracer.py`, `sweep_framework/load_ttnn_ops_data_v2.py`, moreh ops, CCL ops). These are consolidated into the migration chapter as a dedicated precedents file.

- **Question traceability from Plans v4/v5.** The "Addresses questions" annotation system was a unique contribution. This convention is adopted in the final plan, with the referenced questions embedded in the conventions section for standalone readability.

- **Phase 0 "no C++ changes" from Plan v5.** Plan v5's proposal to add `op_name` to `MeshProgramDescriptor` as a zero-adapter quick win was highlighted by multiple evaluators as the most pragmatic first step. It is promoted to the lead phase of the migration plan.

- **Open questions from Plan v3.** Plan v3 was the only plan with a dedicated open questions file, flagging repository ownership, versioning, thread safety, dynamic registration, and non-SPMD mesh heterogeneity. This is retained.

- **Reordering per evaluator feedback.** Evaluators for Plan v3 recommended swapping caching/tracing/profiling to come before blaze-nn integration, since the dependency graph shows the former feeds the latter. This reordering is applied.

- **Scope control.** Plan v1's 33 files caused "reader fatigue" concerns. Plan v4/v5's 21-22 files risked insufficient depth on critical topics. This plan targets 24 files across 8 chapters, matching the evaluated sweet spot.

---

## Audience

**Primary:** Tenstorrent engineers who work at the boundary between TT-Blaze (Python-based op authoring) and TT-Metal/TTNN (the C++ runtime and op framework). They are familiar with writing Blaze MicroOps and FusedOps, have used `ttnn.generic_op()` to dispatch programs, and understand TTNN at the Python-user level. They know what `ProgramDescriptor` and `MeshProgramDescriptor` are, and have encountered the limitations of generic_op (all ops appearing as one name in profiler, no structured program cache integration, invisible to graph capture).

**Secondary:** TTNN framework developers considering how to extend the registration system to accommodate externally-authored ops, and blaze-nn developers who want their functional API (`blaze_nn.F.linear`, `blaze_nn.F.rmsnorm`, etc.) to dispatch through registered `ttnn.*` callables instead of routing through `generic_op`. Also: technical leads evaluating the effort, risk, and migration strategy.

**Assumed knowledge:**
- Python fluency and basic C++ template literacy (concepts, `std::variant`, constexpr templates)
- Familiarity with TT-Metal concepts: `Program`, `ProgramDescriptor`, `MeshProgramDescriptor`, circular buffers, kernels, core ranges
- Experience writing or reading at least one Blaze op (understanding of `BlazeOp`, `MicroOp`, `FusedOp`, `emit()`, `compose()`)
- Basic understanding of TTNN tensor types, device placement, and `ttnn.generic_op()` usage
- Basic awareness of nanobind for Python-C++ interop

**Not assumed:**
- Detailed knowledge of TTNN's `DeviceOperationConcept` constraints, `register_operation<>` template mechanics, or `program_factory_t` variant structure
- Understanding of how `GenericOpDeviceOperation` works internally
- Knowledge of TTNN's graph tracing (`GraphTracker`), profiler integration, or program cache internals
- Experience with CMake-based build system integration for TTNN operations

---

## Chapter List

### Chapter 1: The TTNN Op Registration Contract

**Description:** Dissects how a native TTNN operation is defined, registered, dispatched, and exposed to Python, establishing the target specification that any Blaze adapter must satisfy.

**Directory:** `ch01_ttnn_registration_contract/`

**Files:**

- `01_device_operation_concept.md`
  - The `DeviceOperationConcept` C++ concept from `operation_concepts.hpp`: the five required type aliases (`operation_attributes_t`, `tensor_args_t`, `spec_return_value_t`, `tensor_return_value_t`, `program_factory_t`) and required static methods (`validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`)
  - Optional concept extensions: `DeviceOperationWithCustomProgramCacheConcept` (custom `compute_program_hash`), `HasValidateOnProgramCacheHit`, `HasSelectProgramFactory`, `HasSkipLaunch`
  - `ProgramFactoryConcept` vs `MeshWorkloadFactoryConcept`: the two factory paths (single-device `create()` vs mesh-aware `create_mesh_workload()` / per-coordinate `create_at()`)
  - `CachedProgram<T>`, `CachedMeshWorkload<T>`, `AdaptedCachedMeshWorkload<T>`: how factories store shared variables for runtime argument override
  - `AllFactoriesValid` concept constraint on `program_factory_t` variant
  - Walk through a concrete TTNN op struct (e.g., `UniformDeviceOperation` or from `examples/`) as a minimal reference implementation
  - Source references: `ttnn/api/ttnn/operation_concepts.hpp`, `ttnn/api/ttnn/device_operation.hpp`

- `02_registration_dispatch_and_bindings.md`
  - The `register_operation<"ttnn::name", Struct>()` template in `decorators.hpp`: `registered_operation_t` wrapper, compile-time name uniqueness via `operation_name_key_t` and `operation_key_t` friend injection, `invoke()` dispatch path splitting on `DeviceOperationConcept` vs composite
  - `MeshDeviceOperationAdapter<DeviceOperation>`: how it wraps single-device `ProgramFactoryConcept` factories into mesh workloads via `MeshWorkloadFactoryAdapter`
  - The `device_operation::launch<>` function: output creation, topology imputation, program cache lookup flow (`compute_mesh_workload_hash` -> hit/miss paths), `enqueue_mesh_workload`
  - Nanobind binding via `bind_registered_operation()`: creating a `nb::class_` from `registered_operation_t`, `nanobind_overload_t` for call signatures, `__ttnn_operation__` property, `mod.attr(operation.base_name())` module installation
  - Source references: `ttnn/api/ttnn/decorators.hpp`, `ttnn/cpp/ttnn-nanobind/decorators.hpp`

- `03_program_cache_graph_tracing_and_profiling.md`
  - Program cache: hash-based lookup in `program_cache.hpp`, default hash via `hash_objects_with_default_seed(type_hash<device_operation_t>, operation_attributes, tensor_args)`, custom hash via `compute_program_hash()`
  - Cache hit path: `validate_on_program_cache_hit` (lightweight re-check) -> `override_runtime_arguments()` (update tensor addresses without rebuilding program)
  - Cache miss path: `validate_on_program_cache_miss` -> factory `create_mesh_workload()` or `create_at()` -> cache insertion
  - Graph tracing: `GraphTracker::instance().track_function_start()` / `track_function_end()` called automatically in `device_operation::launch<>` with the operation type name
  - Tracy profiling: `TracyOpMeshWorkload` macro emits zone markers using the operation type name; `op_profiler.hpp` per-operation profiling data
  - Inspector integration: `emit_mesh_workload_annotation()` extracts operation name via `get_operation_name<>()`
  - Why these all matter: what generic_op loses by having all Blaze ops share the "GenericOpDeviceOperation" identity
  - Source references: `ttnn/api/ttnn/device_operation.hpp`, `tt-metalium/program_cache.hpp`

**Addresses questions:** Q1 (What is the TTNN registration contract?)

---

### Chapter 2: How generic_op Works as Blaze's Dispatch Mechanism

**Description:** Traces the complete path from a Blaze `emit()` call through `BlazeCompiler` to `ttnn.generic_op()`, identifying every point where Blaze ops diverge from native TTNN ops.

**Directory:** `ch02_generic_op_dispatch/`

**Files:**

- `01_generic_op_internals.md`
  - `GenericOp` struct in `generic_op.hpp`: its `invoke()` methods accepting `MeshProgramDescriptor` and `ProgramDescriptor` with `io_tensors`
  - Registration: `constexpr auto generic_op = ttnn::register_operation<"ttnn::generic_op", GenericOp>()` -- generic_op is itself a registered TTNN op
  - `GenericOpDeviceOperation`: `operation_attributes_t = MeshProgramDescriptor`, `tensor_args_t = {io_tensors, output_tensor}`, single-variant `program_factory_t = variant<GenericMeshProgramFactory>`
  - `create_output_tensors()` returns the pre-allocated output tensor unchanged; `compute_output_specs()` returns existing tensor spec
  - `compute_program_hash()`: iterates `mesh_programs`, hashes each `ProgramDescriptor` field-by-field (kernels, CBs, semaphores); supports `custom_program_hash` bypass
  - `GenericMeshProgramFactory`: `create_mesh_workload()` iterates per-coordinate programs; `create_at()` calls `ProgramBuilder` to construct `Program` from descriptors; `override_runtime_arguments()` updates CB addresses and runtime args
  - Source references: `ttnn/cpp/ttnn/operations/generic/generic_op.hpp`, `generic_op_device_operation.hpp`, `generic_op_program_factory.hpp`

- `02_blaze_compilation_pipeline.md`
  - The Blaze compilation pipeline: `BlazeGraph` construction via `blaze.fuse()` -> engine pipeline (`CBEngine.assign()`, `SemEngine.assign()`, `CTArgEngine`) -> kernel codegen via `kernel_codegen.py` -> JIT compilation -> `ProgramDescriptor` assembly -> `MeshProgramDescriptor`
  - `FusedProgram` as the per-device builder: `emit()` calls from MicroOps accumulate CB allocations, semaphore assignments, CT args, and phase ordering
  - `BlazeCompiler.compile()`: graph validation -> per-device iteration -> engine pipeline -> `UnifiedKernelDescriptor` -> `ProgramDescriptor` -> `MeshProgramDescriptor`
  - `CompiledProgram.run()` / `MeshCompiledProgram.run()` calls `_run_program()` which calls `ttnn.generic_op(io_tensors, descriptor)`
  - Key insight: at `_run_program()` time, all metadata (op name, input specs, output specs, hash-relevant attributes) exists in Python but is discarded into the opaque `MeshProgramDescriptor`
  - Source references: `blaze/compiler.py`, `blaze/cb_engine.py`, `blaze/sem_engine.py`, `blaze/ct_args.py`

- `03_divergence_analysis.md`
  - Systematic comparison table: what native TTNN ops have vs what Blaze gets through `generic_op`
  - Named identity: all Blaze ops appear as "ttnn::generic_op" in profiler traces, graph captures, and error messages
  - Program cache: `custom_program_hash` is user-controlled and optional; without it, the hash walks the entire descriptor on every dispatch
  - Output tensor management: Blaze pre-allocates; TTNN expects `create_output_tensors()` to handle allocation
  - Validation: only `verify_no_duplicate_mesh_coord_ranges`; no per-op shape/dtype/layout validation
  - Graph tracing: `GraphTracker` sees only `GenericOpDeviceOperation` starts/ends; cannot capture internal FusedOp subgraph structure
  - Composability: Blaze ops cannot be mixed with native TTNN ops in the same traced graph
  - `compute_output_specs`: Blaze ops lack the spec computation that graph tracing's NO_DISPATCH mode depends on

**Addresses questions:** Q2 (How does generic_op work?), Q3 (What does generic_op bypass?)

---

### Chapter 3: The Compilation Model Gap

**Description:** Systematically characterizes why Blaze ops cannot be trivially registered as TTNN device operations, analyzing the fundamental architectural mismatch between Blaze's Python-centric compilation and TTNN's C++-centric `DeviceOperationConcept`.

**Directory:** `ch03_compilation_model_gap/`

**Files:**

- `01_blaze_vs_ttnn_lifecycles.md`
  - Blaze lifecycle: Python op definition -> `emit()`/`compose()` populates `FusedProgram` -> engine pipeline (CBEngine, SemEngine, CTArgEngine) resolves resources in Python -> `kernel_codegen.py` generates kernel source -> JIT compilation -> `ProgramDescriptor` -> `ttnn.generic_op()` dispatch
  - TTNN lifecycle: Python call -> `registered_operation_t::operator()` -> `invoke()` -> `device_operation::launch<>()` -> hash -> cache lookup -> miss: C++ factory `create_mesh_workload()` builds `Program` via `CreateKernel()`, `CreateCircularBuffer()`, `CreateSemaphore()` -> hit: `override_runtime_arguments()` -> enqueue
  - Where decisions are made: Blaze makes CB allocation, semaphore assignment, CT arg computation, and kernel source generation decisions in Python; TTNN makes equivalent decisions in C++ program factories
  - Side-by-side comparison diagram showing the two pipelines and where they diverge

- `02_translation_challenges.md`
  - **Challenge 1: Language boundary.** Blaze's op logic (CB allocation, CT arg computation, phase composition) is in Python; TTNN expects C++ factories. No mechanism exists to call Blaze's Python `emit()` from inside a C++ `ProgramFactory::create()` without embedding the interpreter.
  - **Challenge 2: Stateful vs stateless.** Blaze's `FusedProgram` accumulates state across multiple `emit()` calls; TTNN factories are stateless functions that take attributes and produce programs.
  - **Challenge 3: Output tensor management.** Blaze pre-allocates output tensors and passes them through `io_tensors`; TTNN expects `create_output_tensors()` to handle allocation from `compute_output_specs()`.
  - **Challenge 4: Multi-phase programs.** A Blaze `FusedOp` produces a single kernel with multiple phases sharing CBs, semaphores, and CT arg namespaces; TTNN's factory model expects one program per invocation, with no native concept of intra-program phases.
  - **Challenge 5: Dynamic kernel sources.** Blaze generates kernel C++ source at compile time via `kernel_codegen.py`; TTNN ops typically have static kernel sources compiled into the binary.
  - **Challenge 6: Per-core specialization.** Blaze uses `PerCoreCompileTimeDescriptor` for per-core CT arg variation; TTNN's standard pattern uses uniform CT args with runtime args for per-core variation.
  - **Challenge 7: Engine pipeline.** Blaze's CBEngine/SemEngine/RoleEngine/CTArgEngine form an integrated pipeline with no equivalent orchestration layer in TTNN's factory model.
  - The "100+ ops problem": each Blaze MicroOp would need its own C++ `DeviceOperation` struct if done naively; FusedOps are even harder because their "attributes" are dynamically composed graphs
  - Mapping table: Blaze concept -> TTNN equivalent -> gap description

**Addresses questions:** Q4 (What is the structural gap between Blaze and TTNN?)

---

### Chapter 4: The Adapter Pattern -- Bridging Blaze Programs to DeviceOperationConcept

**Description:** Designs a generic parameterized adapter that wraps any Blaze op's `MeshProgramDescriptor` output into a `DeviceOperationConcept`-conforming struct without requiring per-op C++ factory code, then contrasts with per-op wrapping for high-value ops.

**Directory:** `ch04_adapter_pattern/`

**Files:**

- `01_design_goals_and_tradeoffs.md`
  - Generic adapter (one C++ template parameterized by op identity) vs per-op wrapping (dedicated C++ struct per Blaze op)
  - Tradeoff dimensions: development velocity, type safety, program cache granularity, profiler resolution, Python API ergonomics, maintenance cost
  - Why a hybrid approach is optimal: generic adapter for the common case with parameterized identity, individual structs only where custom `operation_attributes_t` or validation logic is needed
  - Decision matrix: when each approach is appropriate based on profiling needs, validation complexity, and maintenance cost
  - Per-op effort estimate: approximately 200-400 lines of C++ per MicroOp wrapper, 400-800 per FusedOp wrapper, plus nanobind bindings (vs ~0 per-op lines for the generic adapter)
  - Backwards compatibility requirement: existing `ttnn.generic_op()` paths must continue to work

- `02_blaze_device_operation_adapter.md`
  - Proposed `BlazeDeviceOperationAdapter<OpTag>` C++ struct template satisfying `DeviceOperationConcept`
  - `OpTag` struct pattern: `struct MatmulTag { static constexpr const char* name = "blaze::matmul"; }` -- minimal, stateless, used only for type identity and name
  - `operation_attributes_t`: wraps `MeshProgramDescriptor` plus an `op_name` string and optional typed metadata
  - `tensor_args_t`: wraps `io_tensors` vector plus explicit `output_tensor` reference (matching `GenericOpDeviceOperation` structure)
  - `validate_on_program_cache_miss`: delegates to `GenericOpDeviceOperation` validation plus optional per-op extension hook
  - `compute_output_specs`: derives specs from pre-allocated output tensor (preserving Blaze's pre-allocation model)
  - `create_output_tensors`: returns pre-allocated output from `tensor_args_t` (unchanged behavior)
  - `compute_program_hash`: wraps existing `GenericOpDeviceOperation::compute_program_hash` with `type_hash<BlazeDeviceOperationAdapter<OpTag>>` prefix for per-op cache namespace isolation
  - `program_factory_t`: reuses `GenericMeshProgramFactory` as the sole variant -- no new program factory code needed
  - Registration: `constexpr auto blaze_matmul = register_operation<"ttnn::blaze::matmul", BlazeDeviceOperationAdapter<MatmulTag>>()`
  - Diagram: Blaze Python compilation -> ProgramDescriptor -> BlazeDeviceOperationAdapter -> TTNN dispatch pipeline

- `03_python_dispatch_and_registration.md`
  - Modified `_run_program()` in `compiler.py`: dispatch through `ttnn.blaze.{op_name}(io_tensors, descriptor)` if a named binding exists, falling back to `ttnn.generic_op` otherwise
  - Python-side `ttnn.blaze` namespace: nanobind module exposing all registered Blaze op callables
  - Auto-generation from Blaze registry: a build-time or JIT Python script iterates `BlazeOp._class_registry` and `_FUSED_OP_CONFIGS`, emitting one tag struct per op plus the corresponding `register_operation<>` invocation
  - Alternative: macro-based approach (`REGISTER_BLAZE_OP("matmul")`) that expands to both the tag struct and registration
  - Nanobind binding: `bind_registered_operation(mod, ttnn::blaze::matmul, doc, overload...)` with overload signature `(io_tensors: list[Tensor], mesh_program_descriptor: MeshProgramDescriptor) -> Tensor`
  - Backward compatibility: `ttnn.generic_op()` path remains functional; adapter is opt-in

**Addresses questions:** Q5 (What adapter strategies exist?), Q6 (How do we register ops?)

---

### Chapter 5: Registering MicroOps and FusedOps

**Description:** Concretely instantiates the adapter pattern for both Blaze op categories, with end-to-end walkthroughs and a batch registration strategy for scaling to all 112+ ops.

**Directory:** `ch05_op_registration/`

**Files:**

- `01_microop_registration_walkthrough.md`
  - MicroOp characteristics: single kernel, declared ports (`Input`/`Output`/`Internal`), CT args from `cpp_parser.py`, a single `emit()` call producing one `ProgramDescriptor`
  - `cpp_parser.py` and `_auto_derive_from_kernel_hpp()`: how op metadata (CT arg types, RISC mapping, lifecycle methods) is extracted from `op.hpp` headers
  - End-to-end walkthrough: `Matmul` MicroOp from `blaze/ops/matmul/op.py` -> `OpTag` struct -> `BlazeDeviceOperationAdapter<MatmulTag>` -> `register_operation<"ttnn::blaze::matmul", ...>()` -> nanobind binding -> `ttnn.blaze.matmul(...)` callable in Python
  - Verifying program cache isolation: matmul adapter gets its own hash space separate from other Blaze ops
  - Profiler output comparison: before (all ops show as "generic_op") vs after (shows as "ttnn::blaze::matmul")

- `02_fusedop_registration_walkthrough.md`
  - FusedOp characteristics: `compose()` chains multiple MicroOp `emit()` calls on a shared `FusedProgram`, producing a single multi-phase `ProgramDescriptor`
  - Why this is fundamentally different from TTNN's per-op model: a FusedOp is one program, not a sequence of ops
  - The single-factory mapping: the adapter wraps the final assembled `MeshProgramDescriptor` the same way as for a MicroOp; the multi-phase nature is internal to the descriptor
  - `FusedOpConfig` integration: `compose_fn`, `compute_config`, optional handwritten kernel
  - Hash strategy for fused ops: hashing graph topology + user_args + tensor shapes rather than full descriptor
  - Edge case: FusedOps with dynamic graph structure (conditional composition) -- when caching breaks down
  - End-to-end walkthrough: `GatedReduce` FusedOp -> adapter struct -> registration -> Python callable -> profiler showing "ttnn::blaze::gated_reduce"

- `03_batch_registration_and_scaling.md`
  - Registering all 112+ Blaze ops without manual per-op boilerplate
  - Approach 1: C++ variadic template that iterates a compile-time list of tag structs
  - Approach 2: Python codegen script that reads `blaze.registry.list_ops()` and emits registration code
  - Approach 3: Macro-based registration table (one macro invocation per op)
  - Build system integration: where generated files live in the source tree, CMake rules for codegen, incremental rebuild strategy
  - Template instantiation cost: each `BlazeDeviceOperationAdapter<Tag>` instantiation generates a full `device_operation::launch<>` specialization; strategies to minimize compile time (explicit instantiation, unity builds)
  - Runtime discovery: a `ttnn.blaze.list_ops()` Python function that enumerates all registered Blaze ops
  - Handling the open set problem: new Blaze ops added without rebuilding TT-Metal; options include a finite "known ops" list vs a dynamic registration path

**Addresses questions:** Q5 (Adapter instantiation), Q7 (FusedOp challenges)

---

### Chapter 6: Program Caching, Graph Tracing, and Profiling Integration

**Description:** Details how first-class registration unlocks TTNN's program cache, graph capture, and profiling infrastructure for Blaze ops, and what adapter-level work is needed to make each feature function correctly.

**Directory:** `ch06_caching_tracing_profiling/`

**Files:**

- `01_program_cache_integration.md`
  - Current state: `GenericOpDeviceOperation.compute_program_hash()` uses `custom_program_hash` from descriptor or falls back to expensive field-by-field hashing of entire `ProgramDescriptor` (kernel source paths, all CB configs, all semaphore configs)
  - After registration: each Blaze op gets its own `type_hash<BlazeDeviceOperationAdapter<Tag>>` in the hash seed, isolating cache entries per op type
  - Cache key design: generic adapter key = `hash(type_hash<Adapter<Tag>>, custom_program_hash_or_descriptor_hash)` vs individual wrapper key = `hash(type_hash<OpStruct>, typed_attributes)`
  - Performance comparison: precomputed hash (O(1)) vs descriptor hash (O(kernels + CBs + sems)) vs typed attribute hash (O(attributes + tensor_args))
  - The `override_runtime_arguments` opportunity: on cache hit, only tensor addresses and runtime args need updating; Blaze currently rebuilds the entire `ProgramDescriptor` even when only runtime args change
  - Design sketch: the adapter's `shared_variables_t` stores CB handles and kernel handles; `override_runtime_arguments()` updates tensor buffer addresses via `UpdateCircularBufferAddress()` and runtime args via `SetRuntimeArgs()`
  - Cache invalidation: with descriptor hashing, any Blaze op change invalidates cache; with registration, only parameter-level changes cause invalidation

- `02_graph_tracing_and_inspector.md`
  - `GraphTracker::track_function_start(op_name, ...)` and `track_function_end(tensor_return_value)` -- called automatically by `device_operation::launch<>()`
  - Current state: all Blaze ops appear as one "generic_op" node in traced graphs
  - After registration: each Blaze op appears as a distinct named node ("ttnn::blaze::matmul", "ttnn::blaze::rmsnorm", etc.)
  - `NO_DISPATCH` mode: programs are not cached when graph hooks block; adapter must handle this correctly via existing `hook_blocks` check
  - `compute_output_specs` requirement: graph tracing uses this to determine output shapes without running the op; adapter provides valid implementation from pre-allocated tensor spec
  - Inspector integration: `emit_mesh_workload_annotation()` uses `get_operation_name<>()` which recurses through `MeshDeviceOperationAdapter` to the underlying adapter type; registered ops produce meaningful per-op annotations
  - Impact on graph-level optimization: named nodes enable future op-specific rewrite rules (matmul fusion, redundant op elimination)

- `03_tracy_profiling.md`
  - Tracy integration: `TracyOpMeshWorkload` macro in `enqueue_mesh_workload` extracts operation name from the `DeviceOperation` type; registered Blaze ops produce distinct Tracy zones
  - Before vs after: profiler flamegraphs showing "GenericOpDeviceOperation" dominating vs individual Blaze op names with distinct color coding
  - `op_profiler.hpp`: operation attributes logged in debug mode via `tt::stl::reflection::get_attributes()` -- making adapter's `operation_attributes_t` reflectable
  - Custom profiling metadata: per-op wrappers (for high-value ops) can inject typed attributes (M, K, N for matmul) into profiler annotations
  - Performance model hooks: `create_op_performance_model` in `MeshDeviceOperationAdapter` -- whether Blaze ops can provide custom performance models

**Addresses questions:** Q8 (Program cache implications), Q9 (Graph tracing), Q10 (Profiling)

---

### Chapter 7: Connecting blaze-nn to Registered Operations

**Description:** Rewires blaze-nn's functional API and module system to dispatch through registered `ttnn.blaze.*` callables instead of the `generic_op` escape hatch, covering dispatch changes, fallback strategy, and module-level implications.

**Directory:** `ch07_blaze_nn_integration/`

**Files:**

- `01_current_blaze_nn_dispatch.md`
  - blaze-nn's current dispatch: `F.linear()` -> `_dispatch("matmul", ...)` -> `_get_active_context().dispatch()` -> tracing context records a graph node or calls `blaze.matmul()` in compose mode
  - The `_REGISTRY` in `_registry.py`: `OpMapping` entries map functional names to Blaze op names with `default_grid` and optional kwargs transform functions
  - `TensorProxy`: lazy tensor type used during tracing; records operations without executing
  - `Parameter`: wraps weight tensors; resolved to device tensors at execution time
  - The tracing context: `GraphTracingContext` (graph mode) vs `ComposeTracingContext` (compose mode) dispatch paths
  - Current gap: blaze-nn has no awareness of TTNN op registration; everything routes through `generic_op`

- `02_rewired_dispatch_and_module_integration.md`
  - New dispatch path: `blaze_nn.F.linear()` -> registered `ttnn.blaze.matmul(...)` callable (when outside a tracing context)
  - Updating `_registry.py`: `OpMapping` gains a `ttnn_callable` field pointing to the registered operation
  - Fallback strategy: if a Blaze op is not yet registered as first-class, fall back to `generic_op` dispatch
  - Graph mode vs direct mode: in graph mode, blaze-nn still builds a `BlazeGraph` and compiles; in direct mode, it dispatches to the registered TTNN op directly
  - Module integration: `Module.forward()` traces through the functional API; registered ops appear in TTNN profiler with meaningful names
  - State dict and parameter management: unchanged (tensor ownership stays with blaze-nn)
  - Composability: mixing registered Blaze ops with native TTNN ops in the same model graph
  - End-to-end flow: `Module.forward()` -> `F.linear()` -> `ttnn.blaze.matmul()` -> TTNN dispatch pipeline -> device execution

**Addresses questions:** Q11 (blaze-nn integration)

---

### Chapter 8: Build System, Migration Path, and Open Questions

**Description:** Covers the practical engineering of integrating registered Blaze ops into the TT-Metal build system, a phased migration plan starting with a zero-C++-change quick win, existing precedent patterns, alternative strategies, and unresolved design questions.

**Directory:** `ch08_build_migration_alternatives/`

**Files:**

- `01_build_system_integration.md`
  - Current build model: Blaze ops are pure Python with JIT-compiled C++ kernels; TTNN ops are compiled via tt-metal CMake
  - Option A (in-tree): add `ttnn/cpp/ttnn/operations/blaze/` directory with adapter structs and registrations; integrated into main TTNN CMake target
  - Option B (out-of-tree plugin): separate shared library (`.so`) linked against tt-metal public headers and nanobind; loaded at runtime via `importlib` or `dlopen`
  - Option C (JIT registration): use nanobind's dynamic class creation to instantiate adapters at Python import time; avoids C++ compilation for new ops
  - Option D (hybrid): pre-compile adapters for known ops at build time; allow JIT registration for experimental/user-defined ops
  - CMake integration details: source file layout, include path management, dependency on `generic_op_device_operation.hpp` and `MeshProgramDescriptor`
  - Build dependency: adapter only needs op name string, not kernel code; no circular dependency between tt-blaze and tt-metal

- `02_migration_plan.md`
  - **Phase 0 (immediate, no C++ adapter changes):** Add `op_name` field to `MeshProgramDescriptor`; modify `GenericOpDeviceOperation::compute_program_hash()` to prefix with `op_name` hash; modify profiler hooks to extract and display `op_name`. Estimated scope: ~50 lines of C++ changes. Delivers immediate profiling visibility.
  - **Phase 1 (generic adapter):** Implement `BlazeDeviceOperationAdapter<OpTag>` template; register top 5-10 most profiled ops (matmul, rmsnorm, sdpa, rope, mcast) with hand-verified correctness; modify `_run_program()` to dispatch through named ops when available; `generic_op` remains as fallback
  - **Phase 2 (batch registration):** Auto-generate registrations for all MicroOps using codegen from Blaze registry; update blaze-nn `_registry.py` to prefer registered callables
  - **Phase 3 (FusedOp registration):** Register FusedOps using the same adapter; implement compile-time/runtime-arg separation for cache efficiency
  - **Phase 4 (blaze-nn integration):** Update `_dispatch()` to use registered ops; enable graph capture and optimization for blaze-nn models
  - Backwards compatibility at every phase: `ttnn.generic_op()` remains functional; `BlazeCompiler` and `FusedProgram` APIs unchanged; new `ttnn.blaze.*` namespace is purely additive
  - Testing strategy: parallel execution comparing generic_op path vs registered op path; numerical equivalence, program cache correctness, profiler trace validation
  - Risk assessment: C++ dependency coupling between tt-blaze and tt-metal adapter code; version management

- `03_existing_patterns_and_precedents.md`
  - TTNN's `experimental/` directory: experimental ops follow the full registration pattern (same `DeviceOperationConcept`, separate namespace); there is no "lightweight" registration path today
  - The moreh ops pattern (`ttnn/operations/moreh/`): externally developed ops following standard registration; demonstrates that external teams can register ops into TTNN
  - CCL operations (`all_gather`, `reduce_scatter`, `all_broadcast`): mesh-aware ops using `MeshWorkloadFactoryConcept` with `create_mesh_workload()` -- closest analog to Blaze's multi-device dispatch
  - `model_tracer/generic_ops_tracer.py` in TT-Blaze: traces generic_op calls and extracts op identity from Python call stack; an existing workaround that demonstrates the need
  - `sweep_framework/load_ttnn_ops_data_v2.py`: loads TTNN op metadata for sweep testing; registered Blaze ops would appear automatically
  - Alternative strategies considered:
    - **Enhanced generic_op only:** add `op_name` to `MeshProgramDescriptor` without any adapter (Phase 0 only). Pros: minimal change, immediate. Cons: no per-op cache isolation, no typed validation, no `compute_output_specs`
    - **Python-only wrappers:** register Python composite ops that call `generic_op` internally. Pros: no C++ compilation. Cons: no program cache integration, no Tracy profiling at device level
    - **Plugin/extension system:** a hypothetical `ttnn.register_external_op()` API. Pros: enables third-party ecosystems. Cons: requires fundamental changes to `register_operation<>` compile-time model
    - **Full C++ port of emit():** rewrite each op's program construction in C++ factories. Pros: maximum integration. Cons: impractical for 100+ ops and all FusedOps; loses Python JIT flexibility
  - Comparison table: each strategy rated on profiling visibility, program cache integration, graph tracing, development effort, maintenance burden

- `04_open_questions.md`
  - **Repository ownership:** Should `BlazeDeviceOperationAdapter` live in tt-metal or tt-blaze? In-tree vs out-of-tree tradeoffs for code review, release coupling, and build dependencies
  - **Non-SPMD mesh programs:** How to handle ops that span multiple mesh coordinates with heterogeneous `ProgramDescriptor`s per coordinate (different kernels on different devices)?
  - **Versioning and cache staleness:** When a Blaze op's kernel source changes (via `kernel_codegen.py`), how does the adapter detect that cached programs built from old descriptors are stale? Is `custom_program_hash` sufficient?
  - **Thread safety:** Concurrent registration of Blaze ops at Python import time vs TTNN's compile-time `constexpr` registration model. Is there a race condition if multiple threads import blaze-nn simultaneously?
  - **Dynamic registration:** Can new Blaze ops (created by users at runtime) be registered after TTNN module initialization? The `register_operation<>` template uses compile-time string parameters and friend injection -- runtime registration requires a fundamentally different mechanism
  - **Output tensor ownership transition:** If future Blaze ops want TTNN-managed allocation (via proper `compute_output_specs` + `create_output_tensors`), what is the migration path from pre-allocation? How does this interact with Blaze's `_lifetime_tensors` pattern?
  - **Performance regression risk:** The TTNN launch path adds topology imputation, graph tracking, and cache lookup overhead; for ultra-hot-path ops in autoregressive decoding, is this overhead acceptable?

**Addresses questions:** Q12 (Build system), Q13 (Migration strategy), Q14 (Alternative approaches)

---

## Conventions

### Terminology

| Term | Definition |
|------|-----------|
| **DeviceOperationConcept** | The C++ concept (from `operation_concepts.hpp`) that a struct must satisfy to be registered as a TTNN device operation. Requires type aliases for attributes, tensor args, return types, and factory; static methods for validation, output spec computation, output creation. |
| **registered operation** | A `constexpr auto` variable created by `ttnn::register_operation<>()` that wraps an operation struct in a `registered_operation_t` callable. |
| **generic_op** | `ttnn::generic_op` -- the registered TTNN operation that accepts raw `ProgramDescriptor`/`MeshProgramDescriptor` objects. The current dispatch target for all Blaze ops. |
| **GenericOpDeviceOperation** | The `DeviceOperationConcept` implementation for `generic_op`. Wraps `MeshProgramDescriptor` as `operation_attributes_t`. |
| **ProgramDescriptor** | A tt-metal struct describing a complete program: kernels (source, config, args), circular buffers, and semaphores. |
| **MeshProgramDescriptor** | A container mapping `MeshCoordinateRange` to `ProgramDescriptor`, enabling per-device program specialization in multi-device meshes. |
| **custom_program_hash** | An optional field on `ProgramDescriptor` that, when set, bypasses the default field-by-field hash computation and returns the pre-supplied hash value directly. |
| **program_factory_t** | A `std::variant` of factory types inside a DeviceOperation. Each variant must satisfy either `ProgramFactoryConcept` or `MeshWorkloadFactoryConcept`. |
| **BlazeOp** | The Python base class for all Blaze operations. |
| **MicroOp** | A Blaze op backed by a single C++ kernel header (`op.hpp`), representing one computational phase. Subclass of `BlazeOp` with `op_class` set. |
| **FusedOp** | A Blaze op composed of multiple MicroOp `emit()` calls, compiled into a single multi-phase program via `compose()`. |
| **emit()** | The Python method on BlazeOp subclasses that allocates CBs, assigns semaphores, declares CT args, and returns CBHandles. The core of Blaze's program construction logic. |
| **BlazeCompiler** | The Python class that compiles a `BlazeGraph` into a `MeshProgramDescriptor` by running the engine pipeline, generating kernel source, and assembling descriptors. |
| **blaze-nn** | A PyTorch-compatible layer on top of TT-Blaze providing `Module`, `functional`, and `Parameter` abstractions. Maps high-level ops to Blaze ops via `_REGISTRY`. |
| **adapter** | The proposed `BlazeDeviceOperationAdapter<OpTag>` C++ template that wraps pre-built Blaze `MeshProgramDescriptor` objects to satisfy `DeviceOperationConcept`. |
| **OpTag** | A minimal stateless C++ struct used as a template parameter to give each Blaze op a unique type identity and name within the generic adapter. |
| **deep integration** | The alternative path of rewriting Blaze MicroOp `emit()` logic as C++ program factories, making each op a fully native TTNN DeviceOperation. |
| **nanobind** | The Python-C++ binding library used by TTNN. `bind_registered_operation()` creates Python-callable wrappers for registered C++ operations. |

### Research Questions Referenced

The "Addresses questions" annotations reference the following research questions:

| # | Question |
|---|---------|
| Q1 | What is TTNN's complete operation registration contract (DeviceOperationConcept, register_operation, nanobind, cache, tracing)? |
| Q2 | How does `ttnn.generic_op()` currently work as Blaze's dispatch mechanism? |
| Q3 | What TTNN features does the generic_op path bypass? |
| Q4 | What are the structural mismatches between Blaze's Python-centric compilation and TTNN's C++-centric registration? |
| Q5 | What adapter strategies can bridge Blaze programs to DeviceOperationConcept? |
| Q6 | How would per-op and batch registration work in practice? |
| Q7 | What unique challenges do FusedOps present for registration? |
| Q8 | How does first-class registration change program caching behavior? |
| Q9 | How do registered ops appear in TTNN's graph capture system? |
| Q10 | How do registered ops appear in Tracy profiling and the inspector? |
| Q11 | How would blaze-nn's functional API connect to registered TTNN ops? |
| Q12 | What build system changes are needed (CMake, JIT, plugin)? |
| Q13 | What is a practical phased migration plan? |
| Q14 | What alternative approaches exist and how do they compare? |

### File Path Notation

- TTNN/tt-metal paths are relative to the tt-metal repository root: `ttnn/api/ttnn/decorators.hpp`
- TT-Blaze paths are relative to the tt-blaze repository root: `blaze/compiler.py`
- blaze-nn paths are relative to the blaze-nn repository root: `blaze_nn/_registry.py`

### Code Formatting

- C++ code blocks use `cpp` syntax highlighting; Python code blocks use `python`
- C++ concept and class names are written in `PascalCase` (e.g., `DeviceOperationConcept`)
- C++ template parameters use angle-bracket notation: `register_operation<"ttnn::x", XOp>()`
- TTNN operations use dotted Python notation: `ttnn.generic_op`, `ttnn.blaze.matmul`
- Blaze operations use dotted notation: `blaze.matmul`, `blaze.rmsnorm`
- Registered Blaze op names follow the pattern `"ttnn::blaze::op_name"`

### Formatting Rules

- Each file begins with a one-paragraph summary stating its purpose and what the reader will learn
- Each file ends with a "Key Takeaways" bulleted list
- Code listings that reference real files include a header comment with the file path and relevant line range
- Diagrams use ASCII art or Mermaid syntax for data flow and architecture illustrations
- Proposed designs are marked with `PROPOSED` labels; existing code is referenced by file path with `EXISTING` labels
- Trade-off discussions use a consistent format: Options listed -> Pro/Con analysis -> Recommendation with rationale
- Warning/caution callouts use blockquote format with `**Warning:**` prefix
- Cross-references use the format `[Chapter N, File M](../chNN_dir/MM_file.md)`
- Tables are used for feature comparisons and gap analyses; always include a header row

---

## Cross-Chapter Dependencies

```
Chapter 1 ──────> Chapter 2 ──────> Chapter 3 ──────> Chapter 4
(TTNN contract)   (generic_op)      (gap analysis)    (adapter design)
     |                                                     |
     |                                   +-----------------+
     |                                   |                 |
     |                                   v                 v
     |                             Chapter 5          Chapter 6
     |                             (MicroOp/FusedOp)  (cache/trace/profile)
     |                                   |                 |
     +-----------------------------------+---------+       |
                                                   v       v
                                             Chapter 7
                                             (blaze-nn integration)
                                                   |
                                                   v
                                             Chapter 8
                                             (build/migration/open questions)
```

**Detailed dependency edges:**

| Chapter | Depends On | Nature of Dependency |
|---------|-----------|---------------------|
| **Ch1** | (none) | Foundational. Defines all TTNN-side concepts referenced by every subsequent chapter. |
| **Ch2** | Ch1 | Uses Ch1's DeviceOperationConcept terminology to explain how GenericOpDeviceOperation satisfies the concept; references program cache and tracing mechanics from Ch1 to explain what generic_op bypasses. |
| **Ch3** | Ch1, Ch2 | Compares Blaze's pipeline (from Ch2) against TTNN's lifecycle (from Ch1). The translation challenges map Blaze concepts to TTNN concepts defined in Ch1. |
| **Ch4** | Ch1, Ch2, Ch3 | Adapter design directly addresses gaps identified in Ch3; the C++ struct must satisfy DeviceOperationConcept from Ch1; dispatch changes replace the generic_op path from Ch2. |
| **Ch5** | Ch4 | MicroOp and FusedOp registration are concrete instantiations of the generic adapter from Ch4. |
| **Ch6** | Ch1, Ch4 | Cache/tracing/profiling integration depends on Ch1's cache and tracing mechanics and Ch4's adapter `compute_program_hash` and `operation_attributes_t` design. |
| **Ch7** | Ch4, Ch5, Ch6 | blaze-nn integration requires that registered ops exist (Ch4, Ch5) and that caching/tracing works (Ch6). |
| **Ch8** | All prior | The migration plan synthesizes the gap analysis (Ch3), adapter design (Ch4), registration details (Ch5), infrastructure integration (Ch6-Ch7), and references to existing patterns. |

**Key cross-file references:**

- Ch1 File 3 introduces program cache and tracing mechanisms -> Ch2 File 3 shows these are underutilized by generic_op -> Ch6 File 1 proposes optimized caching strategy
- Ch2 File 3 (divergence analysis) produces the feature gap summary -> Ch3 File 2 formalizes into specific engineering challenges -> Ch4 File 1 addresses with design tradeoffs
- Ch3 File 2 (translation challenges) directly motivates Ch4's adapter approach (solving challenges without full C++ porting) and justifies why deep integration is impractical for most ops
- Ch4 File 2 defines the adapter struct -> Ch5 File 1 instantiates it for a MicroOp walkthrough -> Ch5 File 2 extends it for FusedOps
- Ch5 File 3 (batch registration) informs Ch8 File 1 (build system) regarding where generated files live and CMake integration
- Ch6 File 1 (program cache) and Ch6 File 3 (Tracy) describe benefits that motivate Ch8 File 2's phased migration timeline
- **Chapters 1-3 are independently readable** as a technical reference for the current state and gap analysis, even without the proposal chapters
- **Chapter 8 can be read after Chapter 3** for a high-level overview of options before diving into detailed adapter design in Chapters 4-7
