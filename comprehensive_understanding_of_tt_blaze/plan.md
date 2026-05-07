# Final Plan: Comprehensive Understanding of TT-Blaze

---

## Selection Rationale

### Base Plan

**Plan v1** serves as the structural backbone due to its unmatched granularity of file descriptions, the most thorough op library coverage of any plan, exemplary conventions section, and a clean foundational-to-advanced chapter ordering with no forward references. All six evaluators rated it highly on depth and completeness.

### Contributions from Each Plan

- **Plan v1 (base):** Chapter ordering (architecture -> op hierarchy -> compilation -> data flow -> op library -> production -> tooling -> models), op library organization with per-category files, comprehensive glossary and conventions, cross-chapter dependency matrix.

- **Plan v2:** The "API convergence" concept -- explicitly explaining how both APIs produce the same `BlazeGraph` IR. The "supporting modules" coverage of `DeviceContext`, `RoleEngine`, `BlazeProgram`, and infrastructure classes. The "higher-level compositions" concept (GQA attention config, gated MLP config graph builders).

- **Plan v3 (major contributor):** Dedicated per-RISC compilation coverage elevated to its own chapter, including the kernel header directory structure, LLK extensions, and build system. The standalone `OverlappedView` file. The standalone CB reconfig file. Clean cross-chapter reference documentation.

- **Plan v4 (major contributor):** Pipeline manager scheduling algorithm as a dedicated file. Speculative decode state machine as a dedicated file. KV tier management coverage. The separation of KV cache migration protocols from architecture.

- **Plan v5 (major contributor):** Dedicated FusedOp composition patterns chapter with worked examples (SharedExpert and MoE walkthroughs). Advanced composition patterns coverage (CB aliasing, scratch arena, noop prefixes, mesh-aware composition). Explicit learning outcomes in the audience section.

- **Plan v6:** Dedicated Graph API / BlazeGraph IR chapter separated from compilation. Op authoring walkthroughs (writing a MicroOp, writing a FusedOp) integrated into the op hierarchy chapter. Production deployment capstone file with environment variable reference.

### Evaluator Feedback Incorporated

1. **Per-RISC compilation needs dedicated depth** (flagged by evaluators of v1, v2, v4, v5, v6) -- Promoted to its own chapter (Ch 5), drawing from v3's dedicated chapter structure.

2. **FusedOp composition needs worked examples** (flagged by evaluators of v1, v2, v3, v4, v6) -- Dedicated chapter (Ch 7) with four files including SharedExpert and MoE walkthroughs, drawing from v5's Ch 6.

3. **Pipeline manager needs scheduling algorithm and speculative decode depth** (flagged by evaluators of v1, v2, v3, v5, v6) -- Three dedicated files for architecture, scheduling algorithm, and speculative decode, drawing from v4's Ch 6.

4. **Op library needs thorough per-op listings** (flagged by evaluators of v4, v6) -- Five files organized by category (data movement, compute, attention, MoE, utility/infrastructure), drawing from v1's Ch 5.

5. **Avoid overloaded catch-all chapters** (flagged by evaluators of v2, v3, v4, v5) -- No chapter combines more than 3 closely-related topics. Model integrations, testing, and tooling are separated. Pipeline manager and KV migration are in distinct chapters.

6. **Graph API / BlazeGraph IR needs explicit coverage** (flagged by evaluators of v3, v4, v5) -- The BlazeGraph IR and FusionContext are covered in a dedicated file within the compilation chapter, and the dual-API design gets a dedicated file in Ch 1.

7. **Model integrations should not be compressed into a single file** (flagged by evaluator of v5) -- DeepSeek V3 and GLM-5.1 each get their own file, with a separate production deployment file.

8. **Clean layered progression with no forward references** (universal feedback) -- Strict layered ordering verified: each chapter depends only on prior chapters.

---

## Audience

This guide is written for **C++ kernel developers, systems engineers, and ML infrastructure developers at Tenstorrent** (or adjacent teams) who need to understand, extend, debug, or deploy the TT-Blaze kernel composition framework for low-latency inference on Blackhole silicon.

**The reader is expected to already know:**

- The Tenstorrent Tensix architecture (BRISC/NCRISC/TRISC RISC-V processors, L1 SRAM, circular buffers, NOC interconnect, DRAM)
- TT-Metal's programming model at a conceptual level (ProgramDescriptor, CBDescriptor, KernelDescriptor, device dispatch via `ttnn.generic_op`)
- Python 3.10+ and C++20 (templates, constexpr, concepts, structured bindings)
- DAG-based dataflow programming concepts (topological ordering, producer-consumer edges)
- What transformer inference looks like at the op level (matmul, RMSNorm, attention, MoE routing, KV cache, prefill vs. decode)

**The reader does NOT need to know beforehand:** TT-Blaze internals, the Blaze compilation pipeline, the CBHandle abstraction, how `FusedProgram` works, the pipeline orchestration or token scheduling layers, or the specific DeepSeek V3 / GLM-5.1 model integrations.

**After reading, the reader should be able to:** write new MicroOps and FusedOps, understand the full compilation path from Python graph to dispatched kernel, navigate the complete op library, compose fused pipelines using advanced patterns, extend the pipeline orchestration, and reason about multi-host inference architecture.

---

## Chapter List

### Chapter 1 -- Architecture and Position in the Stack

**Description:** Establishes where TT-Blaze sits relative to TT-Metal, TT-LLK, and TT-Forge, explains the core design decisions, and provides a structural map of the entire repository.

**Directory:** `ch01_architecture/`

**Files:**

- `01_stack_position.md`
  - Where Blaze sits: above TT-Metal's dispatch layer, below model-level code; peer to TT-Forge but targeting handwritten C++ kernels rather than compiler-generated ones
  - Relationship to TT-LLK: Blaze kernel headers include LLK intrinsics via `kernel_includes/`; custom SFPU kernels for SDPA, RMSNorm, MoE gate topk
  - Relationship to TT-Forge: Blaze replaces the graph compiler for handwritten fused kernels; Forge targets automatic lowering from PyTorch
  - The `tt-metal` submodule pinned to `blaze-metal` branch: C++20 kernel compilation, named CT/RT arg infrastructure, `UnifiedKernelDescriptor`, `ttnn.generic_op()`
  - Why Blaze exists: the 14x code reduction claim (67-line FusedOp vs 979-line manual wiring); the "C++ kernel developers are first-class citizens" philosophy

- `02_repository_map.md`
  - Top-level directory structure: `blaze/` (core framework), `pipeline_builder/` (multi-host orchestration), `pipeline_manager/` (C++20 token scheduler), `disaggregation/migration/` (KV cache migration), `visualizer/` (interactive HTML), `tests/`, `docs/`
  - The `blaze/` package layout: core modules (`blaze_op.py`, `graph.py`, `cb_engine.py`, `sem_engine.py`, `ct_args.py`, `compiler.py`, `kernel_codegen.py`, `cpp_parser.py`, `fused_program.py`, `context.py`), ops library (`blaze/ops/`), models (`blaze/models/`), stages (`blaze/stages/`), kernel headers and LLK includes (`blaze/kernels/`)
  - Build and environment: `install.sh`, `build_blaze.sh`, `env.sh`, `pyproject.toml`
  - Key infrastructure modules: `DeviceContext` (grid size, NOC translation, harvested-core handling), `RoleEngine` / `GridConfig` (per-core role assignment), `BlazeProgram` (low-level program descriptor builder), `CBReconfig` (CB ID management for multi-phase programs), `L1Profile` (runtime CB statistics), `Barrier` / `CCL` (fabric synchronization and collective communication)
  - The auto-discovery mechanism in `blaze/__init__.py`: how `_OpHandle` proxies expose both `blaze.mcast(...)` (graph API) and `blaze.mcast.emit(f, ...)` (composition API)

- `03_two_api_design.md`
  - The graph API: `blaze.fuse()` context manager, `FusionContext`, `ExternalTensor`, `FusionResult`, how `_op_call` dispatches to the active context
  - The composition API: `FusedProgram`, `emit()` methods, `CBHandle` chaining
  - How the two APIs interact and converge: both produce a `BlazeGraph`; `BlazeCompiler` detects single-node fused ops and dispatches to `_compile_fused_op()` which calls `compose()` on the op class, building a `FusedProgram` internally
  - When to use each API: graph API for declarative multi-op pipelines compiled by engines; composition API for imperative kernel construction inside `compose()` with full control over CB sizes, semaphores, and per-core flags

---

### Chapter 2 -- The BlazeOp Class Hierarchy and Op Authoring

**Description:** Covers the type system that defines every op in Blaze -- from abstract base through MicroOp and FusedOp -- the C++ header parser, auto-registration, and practical step-by-step guides for writing new ops.

**Directory:** `ch02_blazeop_hierarchy/`

**Files:**

- `01_blazeop_base.md`
  - `BlazeOp` base class: `name`, `kernel`, `ct_args`, `phase`, `op_class` attributes
  - Port descriptors: `Input()`, `Output()`, `Internal()` as Python class-attribute descriptors with `__set_name__`
  - CT arg kind types: `CB`, `Sem`, `Grid`, `Derived`, `Param`, `PerCore`
  - `CompileTimeArg` dataclass: name, type (`Type.UINT32`, `Type.BOOL`), kind, riscs (`Risc.NCRISC | Risc.TRISC | Risc.BRISC`)
  - The `register()` method: auto-generates `OpSpec`, `CTArgSchema`, `PhaseInfo`, and `FusedOpConfig`
  - `graph_call()` for graph API path, `compose()` for compiler entry point

- `02_microop_and_fusedop.md`
  - `MicroOp(BlazeOp)` subclass: backed by a C++ `Op` struct in an `op.hpp` kernel header. Required overrides: `name`, `emit()`, `compose()`
  - Auto-derivation from kernel header: `_auto_derive_from_kernel_hpp()` calls `cpp_parser.parse_op_hpp()` to extract `struct_name`, CT args, RISC presence, lifecycle methods, pop flags
  - How `_find_kernel_hpp()` locates `kernels/op.hpp` relative to the op module
  - `FusedOp(BlazeOp)` subclass: compiles via composition (chains MicroOp `emit()` calls). Required override: `compose(cls, f, tensors, output, user_args)`. Optional: `emit()` (if the fused op is itself a reusable building block), `kernel` (handwritten path; if omitted, kernel is auto-generated)
  - The separation of `compose()` (compiler calling convention: tensor dict + user_args) vs `emit()` (free-form signature for embedding as a sub-stage)
  - Concrete walkthrough: `Matmul` (MicroOp) showing port descriptors, `emit()` with `f.cb_from_tensor`, `f.cb_scratch`, `f.per_core_unified_ct_args`, `f.ncrisc_ct_args`, `f.trisc_ct_args`, and `f.output()`

- `03_cpp_parser.md`
  - `cpp_parser.py`: the single source of truth philosophy -- "the C++ header IS the op specification"
  - How `parse_op_hpp()` works: regex-based extraction of `struct` names, inner `*CTArgs` structs, `static constexpr` field types (`CB`, `Semaphore`, `PerCore`, `Flag`, `uint32_t`, `bool`)
  - RISC mapping: `ComputeCTArgs` -> TRISC, `ReaderCTArgs` -> NCRISC, `WriterCTArgs` -> BRISC, `CoreCTArgs` -> all RISCs
  - Lifecycle detection: `init()` / `teardown()` presence and empty-body detection for codegen optimization
  - Pop flags: fields named `pop_*` in CoreCTArgs
  - The `_A_NS_TOKEN_RE` pattern for resolving passthrough vs aliased vs derived vs literal fields

- `04_auto_discovery_and_registration.md`
  - The `blaze/ops/__init__.py` `register_all()` function: `pkgutil.iter_modules` discovers op packages, imports `<pkg>.op`, finds `BlazeOp` subclasses, calls `register()`
  - The `blaze/__init__.py` `_OpHandle` proxy: wraps each registered class to expose `blaze.<op>(...)` (Graph API) and `blaze.<op>.emit(...)` (Composition API)
  - The op registry: `register_op()`, `get_op_spec()`, `list_ops()` -- maps op name to class for runtime lookup
  - The `_class_registry` dict and `FusedOpConfig` registry

- `05_writing_a_micro_op.md`
  - Step-by-step walkthrough following `docs/WRITING_A_MICRO_OP.md`: directory creation (`blaze/ops/<name>/`), Python class with `name` and ports, C++ kernel header (`kernels/op.hpp`), `emit()` implementation, `compose()` implementation, testing
  - Concrete example: the `Mcast` op -- CB allocation via `f.cb_from_tensor()` / `f.cb_scratch()`, CT arg declaration via `f.ncrisc_ct_args()` / `f.brisc_ct_args()`, role flags via `f.per_core_unified_ct_args()`, returning a `CBHandle`

- `06_writing_a_fused_op.md`
  - Step-by-step walkthrough following `docs/WRITING_A_FUSED_OP.md`: directory layout, pipeline composition, optional handwritten kernel
  - The `BlazeOp.child_prefix()` and `BlazeOp.cb_name()` methods for hierarchical prefix scoping
  - Preview: `SharedExpert` FusedOp (67 lines replacing 979 lines of manual wiring) -- detailed line-by-line walkthrough deferred to Ch 7

---

### Chapter 3 -- The Compilation Pipeline

**Description:** Traces the end-to-end compilation path from a `BlazeGraph` through all engines to dispatch, covering graph construction, CB assignment, semaphore allocation, CT arg generation, kernel codegen, and the `BlazeCompiler` assembly.

**Directory:** `ch03_compilation_pipeline/`

**Files:**

- `01_blaze_graph_and_fusion_context.md`
  - `BlazeGraph` data structure: `nodes: list[OpNode]`, `edges: list[Edge]`, `input_tensors`, `output_nodes`, `external_input_ports`, `tensor_to_ports`
  - `OpNode`: id, spec (`OpSpec`), grid, kwargs (including `ct_prefix`, `ct_values`)
  - `Edge`: producer/consumer OpNode + port names, `cb_id` (filled by CBEngine)
  - `FusionContext` class: `blaze.fuse()` context manager, `__enter__`/`__exit__`, `add_op()`, `_build_graph()`; thread-local active context via `_active_context`
  - `ExternalTensor` placeholders and `FusionResult` handles for wiring edges
  - Topological ordering via Kahn's algorithm; graph validation (duplicate IDs, port references, cycle detection)
  - `compile_engines()` orchestrator: CBEngine -> SemEngine -> CTArgEngine, returns unified `EngineResult`

- `02_cb_engine.md`
  - `CBEngine`: walks topological order assigning sequential CB IDs
  - Four categories: `ext_in` (external tensor inputs, tensor-backed), `internal` (scratch buffers from OpSpec), `intermed` (inter-op edges, not tensor-backed), `ext_out` (terminal outputs, tensor-backed)
  - Fan-out handling: one producer feeding multiple consumers shares a single CB ID with union of core_ranges
  - Shared tensor optimization: same ExternalTensor feeding multiple input ports reuses one CB ID
  - `CBAssignment` dataclass: cb_id, total_size, page_size, num_pages, core_ranges, is_tensor_backed, data_format, tile_shape, first_use/last_use positions
  - Lifetime tracking and interval coloring compaction (`compact_cb_ids`) for long pipelines approaching the Blackhole 64-CB hardware limit
  - `to_cb_id_manager()`: exports to `CircularBufferIdManager` for downstream allocation without ID collisions

- `03_sem_engine.md`
  - `SemEngine`: identifies inter-core synchronization points from CT arg schemas
  - Sync point discovery: walks topological order, checks for `Sem`-kind CT args
  - Protocol source_keys: `sender_semaphore`, `receiver_semaphore`, `noc0_receiver_semaphore`, `noc1_receiver_semaphore`
  - Mcast sender grouping: ops with same explicit `sender` kwarg share one semaphore
  - `SemAssignment`: sem_id (sequential index), protocol, node_id
  - Global semaphores: `ttnn.create_global_semaphore()`, no hardware-slot limit

- `04_ct_arg_engine.md`
  - `CTArgEngine`: generates scoped, type-validated compile-time arg tuples from graph context
  - `CTArgSchema` and `CTArgSpec`: per-op schema declaring name, risc, type, source (cb/sem/derived/per_core/user), source_key
  - Auto-prefixing by op instance: e.g., `gate_matmul_out_w` from prefix `gate_matmul` + arg `out_w`
  - Shared-prefix deduplication: nodes sharing an explicit `ct_prefix` reuse the same arg list
  - Value resolution: CB assignments -> cb_id, Sem assignments -> L1 addresses, grid context -> NOC coordinates, user overrides -> explicit values
  - Name collision detection (strict mode)
  - Grouping by RISC: `ct_tuples["ncrisc"]`, `ct_tuples["brisc"]`, `ct_tuples["trisc"]`

- `05_kernel_codegen.md`
  - `kernel_codegen.py`: generates complete C++ kernel `.cpp` from a `BlazeGraph`
  - `PhaseInfo` registry: `cpp_type`, `alias_suffix`, `has_init_teardown`, `setup_method`, `init_is_empty`, `teardown_is_empty`
  - `PhaseDecl`: alias, op_type, prefix, template_args
  - Type alias generation: `using ActMcast = blaze::Mcast::Op<ct_args::act_mcast>;`
  - Mcast grouping: leader/follower pattern for persistent sender state sharing (`init` once, `run_as<>` for followers, `teardown` after last)
  - `kernel_main()` structure: profiling via `DeviceZoneScopedN`, per-phase init/run/teardown lifecycle
  - Empty-body optimization: `init_is_empty`/`teardown_is_empty` (detected by `cpp_parser`) suppresses no-op calls
  - Noop kernel support: `DeviceZoneScopedN("..._NOOP")` for disabled phases
  - Debug DPRINT injection via `BLAZE_DEBUG_KERNELS` env var with per-RISC filtering
  - Content-hash file naming for caching: `{label}_{hash}.cpp` with atomic write-to-temp + rename

- `06_blaze_compiler.md`
  - `BlazeCompiler`: the central compiler class accepting a `mesh_device`
  - `compile()` method: mesh decomposition, per-device iteration, engine reuse, semaphore and tensor lifetime management
  - Two compilation paths: engine path (`_compile_for_device` for multi-node graphs) vs fused op path (`_compile_fused_op` for single-node compose()-driven ops)
  - Engine path steps: (1) run CBEngine + SemEngine, (2) build user overrides + grid context, (3) build CB descriptors + CT-engine CB mapping, (4) create global semaphores + build CT args, (5) build `UnifiedKernelDescriptor`, (6) assemble `ProgramDescriptor`
  - Fused op path: instantiate `FusedProgram`, call `config.compose_fn()`, auto-wire outputs, capture shadow graph, call `f.build()`
  - `MeshCompiledProgram` and `CompiledProgram`: wrapping `MeshProgramDescriptor` / `ProgramDescriptor`, io_tensors, lifetime-pinned tensors and semaphores (preventing use-after-free)
  - Execution via `ttnn.generic_op()`
  - Auto-export: `BLAZE_EXPORT=1` generates visualizer JSON

---

### Chapter 4 -- Data Flow: CBHandle Chain and FusedProgram

**Description:** Deep dive into how data flows between ops via circular buffers, how the CBHandle abstraction enables composable pipelines, how FusedProgram manages resource allocation, and how OverlappedView enables zero-copy weight sharing.

**Directory:** `ch04_data_flow/`

**Files:**

- `01_cbhandle_abstraction.md`
  - `CBHandle` dataclass: `cb_id`, `num_pages`, `page_size`, `core_ranges`, `data_format`, `tile_desc`, `byte_offset`, `backing_tensor`, `access_mode` (FIFO vs DIRECT_ADDRESS)
  - `CBAccessMode` enum: FIFO (standard wait/pop semantics) vs DIRECT_ADDRESS (read via `buffer_address()`)
  - `require_fifo_handle()` validator: ensures an op receives a FIFO handle
  - How CBHandle chaining works: one op's `emit()` returns a CBHandle, the next op's `emit()` accepts it as input -- the CB ID and metadata flow automatically
  - Integer coercion: `int(handle)` returns `cb_id` for use in CT-arg tuples
  - Shadow graph recording: `_node_id` and `_port_name` fields track provenance for the visualizer

- `02_fused_program.md`
  - `FusedProgram`: the composition context that ops' `emit()` methods call into
  - CB allocation APIs: `cb_from_tensor()` (tensor-backed input CB), `cb_scratch()` (intermediate scratch CB), `cb_alias()` (reuse CB ID with different metadata), `cb_from_shared_l1_views()` (overlapped view CB), `cb_output()` (register final output)
  - CT arg declaration: `ncrisc_ct_args()`, `trisc_ct_args()`, `brisc_ct_args()`, `unified_ct_args()` (all RISCs)
  - Per-core flags: `flag()` and `per_core_unified_ct_args()` for values like `is_active`, `pop_in0`
  - Semaphore management: `semaphore()` for user-named global semaphores
  - RT arg declaration: `ncrisc_rt_args()`, `trisc_rt_args()`, `brisc_rt_args()`
  - Named tensors: `named_tensor()` deduplicates tensor creation across per-device compile iterations
  - Grid helpers: `f.all_cores`, `f.matmul_cores`, `f.sender_grid`, `f.sender_core`, `f.phantom_cores` -- device-adaptive grids handling harvested Blackhole devices
  - Shadow graph recording: how `FusedProgram` builds a `BlazeGraph` behind the scenes for the visualizer
  - `build()` method: assembles `ProgramDescriptor` from accumulated CB descriptors, kernel descriptors, and io_tensors

- `03_overlapped_view.md`
  - `OverlappedView` dataclass: `tensor` (backing `ttnn.Tensor`), `shard_shape`, `core_ranges`, `dtype`, `tile`, `byte_offset`, `total_size`
  - Purpose: multiple logical tensors (e.g., gate_proj and up_proj weights) packed into one physical L1 allocation, differentiated by `byte_offset`
  - `cb_from_shared_l1_views()`: creates CBs for each view with disjoint-share optimization -- CBs on non-overlapping core ranges can share a CB ID
  - `_format_key()`: dedup key is `(data_format, tile_shape, page_size)` to prevent mismatched page sizes on shared IDs
  - How `BlazeCompiler._split_per_device()` preserves view metadata when decomposing mesh tensors

- `04_cb_reconfig.md`
  - `CircularBufferIdManager`: tracks allocated CB IDs per `(dtype, tile_desc)` to prevent collisions when downstream code allocates additional CBs
  - `cb_reconfig_builder.py`: runtime CB reconfiguration for ops that change tile format mid-pipeline (e.g., MoE routed experts switching between bf16 and bf8 tiles)
  - `CBContext` for scoped ID allocation within a phase
  - `build_cb_reconfig_tensor()` for constructing the reconfig tensor that tells the hardware to change CB format mid-program

---

### Chapter 5 -- Per-RISC Compilation and Kernel Infrastructure

**Description:** Explains how a single kernel source is compiled three times (once per RISC) and how the C++ kernel infrastructure, headers, LLK extensions, and build system support this model.

**Directory:** `ch05_risc_compilation/`

**Files:**

- `01_per_risc_model.md`
  - The three RISC processors per Tensix core: NCRISC (reader/NoC), BRISC (writer/NoC), TRISC (compute, further split into unpack/math/pack)
  - How Blaze compiles per-RISC: separate named CT arg lists for ncrisc, brisc, trisc
  - `UnifiedKernelDescriptor`: wraps kernel source, core_ranges, per-RISC CT args, RT args, compute config, per-core descriptors
  - Named compile-time args: each RISC gets its own list of `(name, value)` tuples; the kernel reads them via `get_named_compile_time_arg_val<T>("name")`
  - Per-core compile-time descriptors: `UnifiedCompileTimeCoreDescriptor` for flags like `is_active`, `PerCoreCompileTimeDescriptor` for values like `sender_idx`
  - Conditional compilation via `if constexpr` on role flags: cores outside an op's grid compile out that op's code
  - `COMPILE_FOR_BRISC` / `COMPILE_FOR_NCRISC` / `COMPILE_FOR_TRISC` preprocessor defines

- `02_kernel_headers.md`
  - The `blaze/kernels/` directory: `ops.hpp` (master include), `kernel_op_api.hpp` (Op lifecycle API), `kernel_utils.hpp` (utility macros), `ct_types.h` (compile-time type definitions), `rt_str.hpp` (runtime string utilities)
  - LLK extensions in `kernel_includes/`: custom SFPU kernels for SDPA, RMSNorm, MoE gate topk, and custom MM operations; both Blackhole and Wormhole B0 variants
  - Per-op kernel headers (`ops/<name>/kernels/op.hpp`): the C++ `Op<A>` struct template with `CoreCTArgs`, `ComputeCTArgs`, `ReaderCTArgs`, `WriterCTArgs` inner structs, and `init()`/`operator()()`/`teardown()` methods
  - The relationship between the Python `PhaseInfo` registry and the C++ `Op` struct types

- `03_build_system_and_jit.md`
  - `build_blaze.sh`: builds TT-Metal with Blaze kernel includes, syncs LLK extensions, builds pipeline_manager
  - JIT compilation: TT-Metal's kernel compiler JIT-compiles the generated `.cpp` for each RISC at dispatch time; content-hashed caching avoids recompilation
  - The `env.sh` setup: environment variables (`TT_METAL_HOME`, `PYTHONPATH`, `BLAZE_DEBUG_KERNELS`) and their effects on compilation and debug output
  - How the generated kernel `.cpp` flows through the JIT pipeline to produce three RISC binaries

---

### Chapter 6 -- The Op Library

**Description:** Catalogs and explains the complete op library -- data movement, compute, attention, MoE, and utility/infrastructure categories -- with thorough per-op listings and higher-level composition patterns.

**Directory:** `ch06_op_library/`

**Files:**

- `01_op_library_overview.md`
  - Op directory convention: `blaze/ops/<name>/` with `__init__.py`, `op.py`, and optional `kernels/op.hpp`
  - The 80+ ops in the library, organized by category
  - MicroOp vs FusedOp distinction: MicroOps have kernel headers and run on specific RISCs; FusedOps compose MicroOps via `emit()` chains
  - FusedOps as op trees: the MLA op tree (PreSDPA -> DistributedFlashMLA -> PostSdpa, each decomposing to 10+ MicroOps), the MoE op tree, the shared_expert pipeline

- `02_data_movement_ops.md`
  - `Mcast`: multicast sender -- broadcasts data from one source to all receiver cores; supports persistent sender state, `setup_src`/`run_as` for multi-mcast sequences
  - `Gather`: collects per-core results to a single receiver core; supports dual NOC, per-core `sender_idx`
  - `Scatter` / `ScatterRaw`: distributes data from one core to per-core destinations
  - `Copy` / `TileCopy`: intra-core data movement
  - `Shard2CB`: shard data from sender tensor to per-head CBs
  - `Embedding` / `EmbedMcast`: token embedding lookup and distribution
  - `AllReduce`: inter-device all-reduce via Fabric
  - `CclBroadcast`: cross-chip broadcast via Fabric
  - `BarrierSender`/`BarrierReceiver`: explicit inter-core synchronization
  - `PipelineStageSync`: inter-stage synchronization for pipeline parallelism
  - `Migration`: KV cache chunk migration between hosts

- `03_compute_ops.md`
  - `Matmul` / `MatmulCB`: matrix multiply with direct-address weights (DRAM) or CB-sourced weights
  - `MatmulFusedAct`: matmul with fused activation (sigmoid for GLM gate MMs)
  - `KNSlicedMatmul` (`KNMatmul`): K-parallel sliced matmul with per-core K-shard computation; gate/up dual-branch with shared prefix and branch suffixes
  - `DRAMStreamingMatmul`: streaming matmul reading weights tile-by-tile from DRAM
  - `RMSNorm` / `PaddedRMSNorm` / `BroadcastRMSNorm`: RMS normalization variants with custom LLK SFPU kernels; FP32 accumulation support
  - `ResidualAdd`: element-wise addition with residual
  - `EltwiseMul`: element-wise multiply
  - `Rope`: rotary position embedding
  - `ClampedSilu` / `SwiGLU`: activation functions
  - `ReduceToOne`: reduce multi-core results to a single core
  - `Retilize` / `Untilize` / `TileRowConvert` / `QIdxTilize`: data layout transformations
  - `Argmax`: argmax for sampling

- `04_attention_ops.md`
  - `MLA`: top-level Multi-Latent Attention FusedOp decomposing into PreSDPA, DistributedFlashMLA, PostSdpa
  - `PreSDPA`: RMSNorm, Q/KV projection, RoPE, KV cache update
  - `FlashMLA` / `SparseFlashMLA` / `DistributedFlashMLA`: flash attention decode with chunked SDPA and tree reduction; 64-core grid, DRAM KV cache streaming, 3-round tree reduction
  - `SDPA` / `SDPAReduce`: scaled dot-product attention + cross-core reduction with custom LLK
  - `PostSdpa`: scatter, KV_B2 matmul, output projection, all-reduce
  - `DsaPipeline`: Dynamic Sparse Attention selection (DsaIndexer -> DsaTopk -> DsaSparseGather -> DsaKvPrep)
  - `DsaSparseFlashDecode`: flash attention on gathered sparse KV subset
  - `CreateQHeads` / `QBranch` / `KVBranch` / `QHeads` / `QAProjection`: attention head manipulation ops
  - `KVCacheUpdate`: write new K/V to DRAM cache
  - GLM-5.1 specific attention: `GlmFusedProj`, `GlmQBranch`, `GlmQBranchFlash`, `GlmPostDsaWo`, `Glm5SdpaPostSdpa`
  - GQA graph builders: `GQAAttentionConfig`, `build_gqa_attention_graph()` -- parameterized graph builders for Grouped Query Attention

- `05_moe_ops.md`
  - MoE op taxonomy: 3 axes (gate style, expert count, shared expert) yielding 8 top-level FusedOps
  - Routers: `MoERouter` (DeepSeek group-based), `GLMMoERouter` (global top-k), `MoELargeRouter`/`GLMMoELargeRouter` (>256 experts, two-face routing)
  - Gates: `DeepseekMoeGate` (group-based top-4 groups -> top-8), `GLMMoeGate` (global top-8), `GLMMoeGateMerge` (two-face merge)
  - `RoutedExpert` variants: Router -> Mcast(indices,scores) -> SwigluOp
  - `SwigluOp`: DRAMStreamingMatmul(up) + DRAMStreamingMatmul(gate) + EltwiseMul + Gather + Mcast + DRAMStreamingMatmul(down)
  - `SharedExpert`: KNMatmul(gate+up) -> GatedReduce -> Mcast -> DownProj
  - `GatedReduce` / `GatedLocalReduce`: combine routed expert outputs with gating weights
  - `DistributedTopk` / `AllReduceMoe`: multi-device MoE communication
  - Large MoE variants: `large_moe`, `large_routed_expert`, `moe_large_router` for models with 256+ experts requiring 2-column phantom grids
  - Dispatch helpers: `moe_op()`, `glm_moe_op()` for automatic variant selection
  - Shared utilities: `moe_common.py`, `router_tensors.py`, `act_gather.py`, `bias_faces.py`, `core_split.py`
  - GLM MoE variants: `GlmMoe`, `GlmMoeGate`, `GlmMoeGateMerge`, `GlmMoeRouter`, `GlmRoutedExpert`, `GlmLargeMoe`, `GlmLargeRoutedExpert`
  - Higher-level compositions: `GatedMLPConfig`, `build_gated_mlp_graph()`, `build_shared_expert_mlp_graph()`, `DenseMLP`, `DenseMLPDram`, `DenseSwiGLU`, `MLPMixed`, `SparseLayer`, `LMHeadSampling`

---

### Chapter 7 -- FusedOp Composition Patterns

**Description:** Shows how to build fused ops by composing MicroOps, covering the composition primitives, advanced patterns, and complete worked examples from simple to complex.

**Directory:** `ch07_composition/`

**Files:**

- `01_composition_basics.md`
  - The `compose()` method contract: `(f: FusedProgram, tensors: dict, output: Tensor, user_args: dict) -> None`
  - Chaining `emit()` calls: each MicroOp returns a `CBHandle` that feeds into the next
  - Prefix namespacing: why every `emit()` call needs a unique `prefix=` argument; what happens when CT-arg names collide (the `CTArgEngine` validates uniqueness within each RISC)
  - Wiring outputs: `f.wire_output()`, `f.cb_output()`, and the auto-wiring logic in `BlazeCompiler._compile_fused_op()`

- `02_advanced_patterns.md`
  - CB aliasing: `f.cb_alias()` for reusing a CB ID across phases when the data format matches but the semantic content changes
  - OverlappedView pattern: packing multiple weight matrices into a single L1 allocation, then creating per-op views with `f.cb_from_shared_l1_views()`
  - Scratch arena: `f.scratch_tensors` and `f.scratch_mapping` for pre-allocated scratch space reused across multiple ops
  - Multi-output ops: ops that produce more than one output tensor (e.g., KV branch producing separate K and V)
  - Noop kernel prefixes: `noop_prefixes` for selectively disabling phases in a fused kernel without recompilation
  - Mesh-aware composition: `f.mesh_coord`, `f.mesh_shape`, `f.mesh_device` for per-device specialization in multi-device compilation
  - CB reconfiguration: `cb_reconfig` and `cb_reconfig_builder` for dynamically changing CB data formats between phases

- `03_worked_example_shared_expert.md`
  - Line-by-line walkthrough of `SharedExpert` compose(): RMSNorm -> Mcast -> KNSlicedMatmul (gate+up) -> GatedReduce -> Mcast -> DownProj -> Gather -> ReduceToOne
  - How the 67-line FusedOp replaces 979 lines of manual TT-Metal wiring
  - The resulting generated kernel: what `kernel_codegen.py` produces, and how it maps back to the compose() call sequence
  - CBHandle chain diagram showing data flow through every emit() call

- `04_worked_example_moe.md`
  - Walkthrough of the `MoE` fused op: dual-branch composition with shared expert and routed expert running in parallel from a single activation mcast
  - How MoE routing (gate, topk, scatter) feeds into the routed expert pipeline
  - The `create_act_gather_dst()` helper for CB allocation spanning sender + matmul core grids
  - How `ReduceToOne` and `ResidualAdd` merge the shared and routed branches
  - CB reconfiguration for MoE routed experts: switching between bf16 and bf8 tiles mid-pipeline

---

### Chapter 8 -- Multi-Host Pipeline Infrastructure

**Description:** Covers the pipeline_builder system for multi-host inference pipeline orchestration, the pipeline_manager C++20 token scheduling library with scheduling algorithm and speculative decode depth, and the KV cache migration library for disaggregated inference.

**Directory:** `ch08_pipeline_infra/`

**Files:**

- `01_pipeline_builder.md`
  - `pipeline_builder/` package: `PipelineGraph`, `Node`, `Edge`, `PipelineLayout`, `SubmeshPartition`
  - `PipelineGraph`: DAG of pipeline stages, each with a submesh shape and optional stage factory
  - Two build paths: `build()` (legacy, explicit chip IDs) vs `build_topology()` (auto-discovery via C++ control-plane APIs `resolve_graph_layout()`)
  - `PipelineConfigEntry`: `entry_node_coord` (chip receiving from previous stage), `exit_node_coord` (chip sending to next stage)
  - Stage order: topological sort on non-loopback edges via Kahn's algorithm
  - Loopback edges for circular pipelines (return from last stage to stage 0 carrying decode token)
  - `SubmeshPartition`: partitions a parent `MeshDevice` into submeshes for pipeline parallelism
  - Visualization: `serialize_layout()`, `write_json()` for pipeline topology export
  - Integration with `blaze/stages/`: `DecoderStage` creating `PipelineBlock` instances
  - `StageContext`, `StageKind`, `StageIO`: pipeline stage abstractions in `blaze/stages/stage_io.py`

- `02_pipeline_manager_architecture.md`
  - `pipeline_manager/` C++20 library: `PipelineManager`, `PipelineInterface`, `MockPipeline`, `SocketPipeline`
  - Motivation: separation of inference server (millisecond-scale HTTP/tokenization) from pipeline manager (microsecond-scale scheduling)
  - Thread model: writer thread (inject tokens per tick), reader thread (collect results), API handler (drain request queue)
  - Data structures: `UserTable` (parallel arrays indexed by user_id), `PromptTable` (fixed-size 2D token store), `PrefillQueue`, `DecodeStaging` (per-user staging registers + SPSC FIFO), `FreeIdPool` (bitmap-based O(1) allocator), `BoundedQueue` (lock-free SPSC queue)
  - `ISRequest` -> `PipelineManager.push_request()` -> scheduling -> `PipelineInterface.inject()` -> `PipelineInterface.read_result()` -> `OutputMessage`
  - Wire format: `wire_format.hpp` for host-device serialization (`InjectDescriptor`, `ResultDescriptor`)
  - `PipelineConfig` variant: `MockConfig` (latency simulation) or `SocketConfig` (real H2D/D2H socket transport)
  - `ManagerParams`: `max_users` (64), `chunk_size` (24), `max_seq_len` (128K), `eos_token`, CPU affinity for writer/reader/api threads
  - Design for hardware deployment: fixed-size pre-allocated structures, no dynamic allocation, O(1) bitmap alloc/free
  - Microbenchmark: `dummy_pipeline_launcher` + `dummy_pipeline_connector`; `deepseek_inference_runner` with real tokenizer support

- `03_scheduling_algorithm.md`
  - Writer thread loop: Priority 1 = decode tokens (minimize TPOT), Priority 2 = prefill tokens (chunked round-robin with configurable `chunk_size`). Blocks only on `pipeline_inject()` when pipeline is genuinely full
  - Reader thread loop: decrement in-flight, check cancel, skip non-sampled prefill, process regular decode (output + loopback staging) or speculative decode state machine
  - API request handling: ALLOCATE (bitmap pop), SUBMIT (store prompt, set PREFILL, push to prefill queue), CONTINUE (append tokens, resume from current_position), CANCEL (mark bitmap, defer cleanup until in-flight == 0)
  - Session lifecycle: INACTIVE -> PREFILL -> DECODE -> COMPLETE -> (CONTINUE back to PREFILL, or CANCEL to INACTIVE)
  - Speculative decode support: MTP (Multi-Token Prediction) flow -- inject base + predicted token back-to-back (2 pipeline slots per user per round)
  - Result page format: 64-byte `ResultDescriptor` carrying 0-2 tokens. Reader disambiguates via `token_0_type` x `num_tokens`: ACCEPT (BASE/1), REJECT (BASE/2), CONTINUE (SPEC/2), STALE (SPEC/0)
  - Match path: ACCEPT + CONTINUE = 2 output tokens per round (2x throughput). Mismatch path: REJECT + STALE = 1 output token + 1 wasted slot
  - No explicit KV rollback: base token re-injected at the previous spec position to overwrite stale KV
  - `SpecDecodeState`: tracks spec_accepts/spec_rejects counters

- `04_kv_cache_migration.md`
  - `disaggregation/migration/` C++ library: multi-host KV cache migration for disaggregated prefill/decode
  - 3-tier KV model: Tier-0 (Hot, accelerator DRAM), Tier-1 (Warm, host DRAM), Tier-2 (Cold, SSD/object store); IS owns policy (why/when), PM owns execution (can/how)
  - Architecture: `MigrationLayer` -> `MigrationFrontend` -> `SenderBackend`/`ReceiverBackend`
  - Transport abstraction: `EndpointTransport` with `RealMpiTransport` (MPI) and `LoopbackTransport` (testing)
  - DCN variants: `DcnSenderBackend`, `DcnReceiverBackend` for direct device-to-device transfer via Data Center Network
  - `TablePreprocessor` / `TableSplitter` / `DcnTableSplitter`: decompose KV chunk tables for multi-device routing
  - `DeviceReader` / `DeviceWriter` / `MultiDeviceReader` / `MultiDeviceWriter`: DRAM I/O operations
  - `SharedSPSCQueue` / `SPSCQueue`: lock-free inter-thread communication
  - `SameProducerConsumerCircularBuffer`: specialized buffer for host-side chunk staging
  - Migration message format: `MigrationMessage` with chunk metadata
  - Python bindings: `migration_ext.cpp` for test harness integration
  - Hardware configurations: textproto files for dual-host galaxy topologies; YAML rank binding files
  - Migration lifecycle: initialize mapping table -> connect to remote endpoint -> `migrate_slot()` / `block_until_migration_received()`

---

### Appendix A -- Visualization, Weight Providers, and Developer Tools

**Description:** Covers the interactive dataflow visualizer, the weight provider abstraction, L1 profiling, debugging tools, and Claude Code integration skills.

**Directory:** `appendix_a_tooling/`

**Files:**

- `01_visualizer.md`
  - `blaze.visualize(graph)`: generates interactive HTML visualization and opens in browser
  - `viz_export.py`: `export_viz_data()` generates JSON data structure with nodes, edges, CB assignments, semaphore assignments, tensor metadata
  - `visualizer/index.html`: standalone single-page app (D3.js-based) with dark/light themes, per-op color coding, CB/semaphore inspection panels, core grid overlays
  - `visualizer/schema.json`: JSON schema for the visualization data format
  - Auto-export: `BLAZE_EXPORT=1` in `BlazeCompiler` auto-dumps JSON after every compile
  - Pipeline topology embedding: merges `pipeline_config.json` from pipeline_builder for multi-stage visualization

- `02_weight_provider.md`
  - `WeightProvider` ABC: `get_mla_torch_weights(layer_idx)`, `get_moe_torch_weights(layer_idx)`
  - `SyntheticWeightProvider`: seeded random weights via caller-supplied generators, optional disk caching via `WeightCache`
  - `StateDictWeightProvider`: loads from HuggingFace safetensors, applies transforms (transpose, deinterleave Q_B_proj, split KV_B_proj, TP slicing, tile shuffling for routed experts, DRAM bank column-major reordering)
  - `BlitzCacheWeightProvider`: loads from Blitz tensorbin cache format with manifest.json; supports direct device tensor path via `load_overlapped_views()` (zero-copy fast path)
  - Factory: `get_weight_provider(source)` driven by `BLAZE_WEIGHT_SOURCE` env var (`synthetic`, `state_dict:<path>`, `blitz:<path>`)

- `03_developer_tools.md`
  - L1 profiling: `BLAZE_L1_PROFILE` env var triggers `l1_profile.py` to print CB memory stats
  - Debug kernels: `BLAZE_DEBUG_KERNELS` env var inserts DPRINT trace statements into generated kernels with per-RISC filtering (brisc, ncrisc, trisc, trisc0/1/2)
  - `DeviceZoneScopedN` profiling zones in generated kernels for Tracy integration
  - Claude Code skills (`.claude/skills/`): `bench-kernel`, `check-kernel-sync`, `dprint`, `fuse-l1-tensors`, `memory-audit`, `port-micro-op`, `triage-hang`
  - Barrier utilities: `barrier.py`, `injected_fabric_barrier_builder.py` for fabric-level synchronization
  - Environment variable reference: `TT_METAL_SLOW_DISPATCH_MODE`, `BLAZE_EXPORT`, `BLAZE_EXPORT_PATH`, `BLAZE_L1_PROFILE`, `BLAZE_DEBUG_KERNELS`, `BLAZE_WEIGHT_SOURCE`

---

### Appendix B -- Testing and Model Integrations

**Description:** Covers the test infrastructure, the DeepSeek V3/R1 and GLM-5.1 model integrations, and the production deployment workflow.

**Directory:** `appendix_b_testing_and_models/`

**Files:**

- `01_test_infrastructure.md`
  - Test organization: `tests/blaze/infra/` (engine unit tests), `tests/blaze/micro-ops/` (silicon tests per MicroOp), `tests/blaze/fused_ops/` (fused op silicon tests), `tests/blaze/backed/` (real-weight backed tests), `tests/blaze/generality/` (cross-model generality scoreboard), `tests/blaze/migration/` (KV cache migration tests), `tests/pipeline_builder/` (pipeline builder tests), `tests/blaze/glm5_1/` (GLM-5.1 tests)
  - `conftest.py` and `conftest_capture.py`: fixtures for device management, weight caching, output capture
  - `test_utils.py`: common test helpers for tensor creation, golden comparison
  - `torch_golden.py` / `sdpa_helpers.py`: PyTorch reference implementations for numerical verification
  - `weight_cache.py` and `device_weight_cache.py`: disk-cached weight generation for reproducible tests
  - The `backed/backed_mapping.json`: maps backed test names to weight configurations
  - Generality scoreboard: `generate_scoreboard.py` produces CSV tracking which ops pass across model variants (DeepSeek V3, GLM-5.1)
  - CI workflows: `pr-gate.yaml` (unit tests), `silicon-tests-bh-loudbox-impl.yaml` / `silicon-tests-p150-impl.yaml` (silicon tests), `p150-silicon-nightly.yaml` (nightly full suite)

- `02_deepseek_v3_integration.md`
  - `blaze/models/deepseek_v3_b1/`: `ModelPipeline`, `pipeline.py`, `cli.py`
  - How the MLA attention module is assembled from the op tree: `PreSDPA` -> `DistributedFlashMLA` -> `PostSdpa`, each composed of 10-20 MicroOps
  - How the MoE module is assembled: `Moe` FusedOp composing `RMSNorm` -> `Mcast` -> `DeepseekMoeGate` -> `RoutedExpert` -> `SharedExpert` -> `AllReduce` -> `ResidualAdd`
  - 4/16/64-process pipeline configurations via `create_pipeline_configuration_from_num_procs()`
  - DeepSeek decoder stages: MoE layers (layer 3+) and dense layers (layers 0-2)
  - Weight provider integration: `CacheWeightProvider`, `StateDictWeightProvider`, `SyntheticWeightProvider`
  - Helper modules: `_deepseek_grids.py` (grid configurations), `_gated_mlp.py` (GatedMLPConfig), `_gqa_attention.py` (GQAAttentionConfig)
  - Slow dispatch mode: `TT_METAL_SLOW_DISPATCH_MODE=1` requirement for socket-based pipeline

- `03_glm51_integration.md`
  - GLM-5.1 ops: dedicated MoE router, routed expert, shared expert, and MoE variants (`glm_moe`, `glm_moe_gate`, `glm_routed_expert`, `glm_large_moe`, `glm_moe_large_router`)
  - GLM-5.1 attention ops: `GlmFusedProj`, `GlmQBranch`, `GlmQBranchFlash`, `GlmPostDsaWo` for sparse attention
  - The DSA (Dynamic Sparse Attention) two-dispatch architecture for GLM-5.1
  - Differences from DeepSeek V3: wider MoE with more experts, different attention patterns, different weight formats
  - Tests: `tests/blaze/glm5_1/` with per-component silicon tests

- `04_production_deployment.md`
  - The full deployment stack end-to-end: `pipeline_builder` configures the graph, `BlazeCompiler` compiles per-stage programs, `pipeline_manager` schedules tokens, `migration` handles disaggregated KV cache transfer
  - Multi-host launch: MPI rank binding, Galaxy topology configuration, submesh partitioning
  - `pipeline_manager/run_inference.py`: Python entry point for running inference with the C++20 token scheduler
  - `ModelPipeline` class: end-to-end wrapper (weights_mode, cache_path, lm_head configuration, pipeline build)
  - Environment variables reference: `TT_METAL_SLOW_DISPATCH_MODE`, `BLAZE_EXPORT`, `BLAZE_EXPORT_PATH`, `BLAZE_L1_PROFILE`, `BLAZE_DEBUG_KERNELS`, `BLAZE_WEIGHT_SOURCE`

---

## Conventions

### Terminology

| Term | Definition |
|------|-----------|
| **MicroOp** | A single op backed by a C++ `Op` struct in an `op.hpp` kernel header. The atomic unit of composition. |
| **FusedOp** | A composite op built by chaining MicroOp `emit()` calls. Compiles into a single persistent mega-kernel. |
| **BlazeGraph** | A directed acyclic graph of `OpNode`s connected by `Edge`s. The intermediate representation for engine-based compilation. |
| **CBHandle** | A Python dataclass referencing a circular buffer, carrying metadata (cb_id, num_pages, page_size, core_ranges, data_format). The primary mechanism for inter-op data flow. Returned by `emit()`, consumed by the next `emit()`. |
| **CB** | Circular buffer -- L1 SRAM buffer on each Tensix core used for producer-consumer data flow. Blackhole hardware limit: 64 CB IDs per core. |
| **CT arg** | Compile-time argument -- baked into the kernel binary per RISC. Contrast with RT arg (runtime argument). |
| **RT arg** | Runtime argument -- written to device registers before dispatch. Can change without recompilation. |
| **RISC** | One of three RISC-V processors per Tensix core: NCRISC (reader/data movement via NOC0), BRISC (writer/data movement via NOC1), TRISC (compute, 3 sub-processors: unpack/math/pack). |
| **emit()** | The method on a BlazeOp that wires the op into a FusedProgram by declaring CBs, CT args, and role flags. Returns a CBHandle. |
| **compose()** | The compiler entry point on a BlazeOp that maps a tensor dict to `emit()` calls. Called by `BlazeCompiler`. |
| **FusedProgram** | The composition context that `emit()` methods operate on. Accumulates CB descriptors, CT/RT args, and builds a `ProgramDescriptor`. |
| **Engine** | One of the compilation passes (CBEngine, SemEngine, CTArgEngine) that processes a BlazeGraph. |
| **Phase** | One op's execution within the generated kernel. Maps to a C++ `Op` struct instantiation. |
| **Prefix** | A string scope (e.g., `shared_expert__gu`) that namespaces CT args and CB names within a fused kernel. |
| **Shadow graph** | A BlazeGraph recorded behind the scenes by FusedProgram during composition, used by the visualizer and codegen. |
| **OverlappedView** | A view into a shared L1 tensor at a specific byte offset, enabling multiple ops to share one physical allocation. |
| **PhaseInfo** | Codegen metadata mapping a Python op to its C++ `Op` struct for kernel generation: type name, lifecycle flags, setup method. |
| **Graph API** | `blaze.fuse()` context manager + `blaze.<op>(...)` calls: captures topology as a BlazeGraph for engine-driven compilation. |
| **Composition API** | `blaze.<op>.emit(f, ...)` calls inside a FusedProgram: full control over resource allocation and CT arg declaration. |
| **Pipeline stage** | One submesh in a multi-host inference pipeline, running a subset of model layers. |
| **PipelineManager** | The C++20 token scheduling library that manages user slots, prefill queuing, decode batching, and speculative decoding. |
| **Slot** | A user session in the pipeline_manager: tracks prefill/decode state, token generation, speculative decoding. |

### Notation

- File paths are given relative to the repository root (`/localdev/salnahari/testing_dir/tt-blaze/`) unless explicitly marked as absolute.
- Code examples use Python unless marked with `// C++` or enclosed in `.hpp`/`.cpp` fences.
- C++ types are shown in monospace with namespace qualifiers (e.g., `blaze::Matmul::Op`).
- Python class names use CamelCase; op `name` fields use snake_case.
- When referring to the three RISC processors, we use: NCRISC (reader), BRISC (writer), TRISC (compute). TRISC further decomposes into TRISC0 (unpack), TRISC1 (math), TRISC2 (pack).
- Grid coordinates are given as `(col, row)` following the tt-metal convention.
- CB IDs are 0-indexed integers with a hardware maximum of 64 on Blackhole.
- Semaphore IDs are 0-indexed sequential integers into the global semaphore list.

### Formatting Rules

- Each chapter opens with a one-paragraph summary and a list of key source files referenced.
- Each file begins with a one-paragraph summary and ends with a "Key Takeaways" bullet list.
- Diagrams use ASCII art or Mermaid syntax where appropriate.
- Code snippets are kept short (under 30 lines) and annotated with inline comments.
- Cross-references to other chapters use the format `[Chapter N: Title](../chNN_dir/file.md)`.
- Code listings include the source file path as a header comment.
- Environment variable references are formatted as `BLAZE_EXPORT=1` (inline code with equals sign).

---

## Cross-Chapter Dependencies

```
Chapter 1 (Architecture)
    -- foundational, no dependencies
    |
    v
Chapter 2 (BlazeOp Hierarchy)
    -- depends on Ch 1 for stack position, two-API design, repository layout
    |
    v
Chapter 3 (Compilation Pipeline)
    -- depends on Ch 1 for architecture overview, BlazeGraph concept
    -- depends on Ch 2 for BlazeOp, OpSpec, CTArgSchema, PhaseInfo, port descriptors
    |
    v
Chapter 4 (Data Flow and CBHandle)
    -- depends on Ch 2 for port descriptors and emit() pattern
    -- depends on Ch 3 for CBEngine assignments, BlazeCompiler, FusedProgram concept
    |
    v
Chapter 5 (Per-RISC Compilation)
    -- depends on Ch 3 for UnifiedKernelDescriptor and CT arg tuples
    -- depends on Ch 2 for C++ kernel header structure and PhaseInfo
    |
    v
Chapter 6 (Op Library)
    -- depends on Ch 2 for MicroOp/FusedOp distinction
    -- depends on Ch 3 for compilation pipeline and CT arg resolution
    -- depends on Ch 4 for CBHandle chaining and FusedProgram usage
    -- depends on Ch 5 for per-RISC model understanding kernel headers
    |
    v
Chapter 7 (FusedOp Composition Patterns)
    -- depends on Ch 4 for all composition primitives (CBHandle, FusedProgram, OverlappedView)
    -- depends on Ch 6 for specific ops used as building blocks
    |
    v
Chapter 8 (Pipeline Infrastructure)
    -- depends on Ch 1 for stack position
    -- depends on Ch 6 for op library (pipeline stages compose ops)
    -- depends on Ch 3 for BlazeCompiler for stage compilation
    -- pipeline_manager is self-contained C++; depends on Ch 1 for context only

Appendix A (Tooling)
    -- depends on Ch 3 for BlazeCompiler auto-export for visualizer
    -- depends on Ch 4 for OverlappedView for weight provider
    -- depends on Ch 6 for op library for understanding visualizer output

Appendix B (Testing and Models)
    -- depends on all prior chapters (Ch 2-7)
    -- Ch 8 for pipeline orchestration (DeepSeek V3 pipeline)
    -- Ch 6 for op library (test coverage)
    -- Ch 7 for composition patterns (backed tests)
    -- Appendix A for weight providers (backed tests)
```

| Chapter/Section | Depends on concepts from |
|----------------|------------------------|
| **Ch 1: Architecture** | (standalone -- foundational) |
| **Ch 2: BlazeOp Hierarchy** | Ch 1 (stack position, two-API design) |
| **Ch 3: Compilation Pipeline** | Ch 1 (architecture overview), Ch 2 (BlazeOp, MicroOp, FusedOp, ports, CT args) |
| **Ch 4: Data Flow and CBHandle** | Ch 2 (ports, emit()), Ch 3 (CBEngine, FusedProgram) |
| **Ch 5: Per-RISC Compilation** | Ch 3 (UnifiedKernelDescriptor, CT arg tuples), Ch 2 (kernel header structure) |
| **Ch 6: Op Library** | Ch 2 (MicroOp/FusedOp distinction), Ch 3 (compilation pipeline), Ch 4 (CBHandle chaining), Ch 5 (per-RISC model) |
| **Ch 7: Composition Patterns** | Ch 4 (FusedProgram, OverlappedView, CB aliasing), Ch 6 (ops used in examples) |
| **Ch 8: Pipeline Infrastructure** | Ch 1 (stack position), Ch 6 (op library for decoder stages), Ch 3 (BlazeCompiler) |
| **Appendix A: Tooling** | Ch 3 (BlazeCompiler, EngineResult), Ch 4 (OverlappedView), Ch 6 (op library) |
| **Appendix B: Testing and Models** | All prior chapters; Appendix A for weight providers |

The chapters are ordered so that each builds on the previous. A reader following chapters 1 through 8 in order will encounter no forward references to unexplained concepts. Appendices A and B can be read after completing Chapters 1-8 (or Chapters 1-7 for Appendix A). Chapters 5 and 8 can each be read independently of each other after their respective prerequisites are met.
