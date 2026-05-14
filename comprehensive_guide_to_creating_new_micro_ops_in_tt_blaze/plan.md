# Final Synthesized Plan: Comprehensive Guide to Creating New Micro-Ops in TT-Blaze

## Selection Rationale

This plan is a hybrid constructed from the strongest elements of all five candidate plans, guided by specific evaluator feedback:

**Structural backbone: Plan v1.** Plan v1 received the strongest overall evaluator rating for question coverage completeness (all 15 questions explicitly mapped), per-file content granularity (the most detailed bullet-point breakdowns), conventions rigor, and cross-chapter dependency documentation. It serves as the foundation.

**Inter-core ops as a dedicated chapter: from Plan v4.** Multiple evaluators (v1 eval, v2 eval, v3 eval, v4 eval) praised v4's dedicated Chapter 6 for inter-core operations. The v1 evaluator explicitly recommended: "Move inter-core ops from Ch8 to a dedicated chapter between Ch5 (Grids) and Ch6 (Composition)." The v3 evaluator noted: "consider splitting Chapter 6 into two chapters" (composition and inter-core). This plan adopts v4's approach and places inter-core ops as Chapter 6, before composition (Chapter 7).

**FusedOp composition before testing/debugging.** Multiple evaluators flagged that composition must precede testing/debugging to avoid forward references. The v4 evaluator stated: "FusedOp composition deferred to Chapter 8 ... creates a forward-reference problem." This plan places composition (Chapter 7) before testing/debugging (Chapter 8).

**Dedicated engine pipeline section: from Plans v2 and v4.** The v1 evaluator recommended: "Add a dedicated engine pipeline section covering CBEngine, SemEngine, RoleEngine, and CTArgEngine." The v3 evaluator concurred: "CBEngine, SemEngine, RoleEngine, CTArgEngine deserve their own walkthrough." This plan includes a dedicated engine pipeline file in the codegen chapter.

**Reading paths: from Plan v2.** The v1 evaluator recommended: "Add alternative reading paths following Plan v2's model." Plan v2's reading paths for different developer personas were praised by all evaluators as a unique and valuable addition.

**CbReconfig, balanced flag, scratch mapping: from Plan v3.** The v1 evaluator recommended: "Add mentions of CbReconfig, BLAZE_DISABLE_TENSOR_CB_SHARE, and all_scratch_mapped." Plan v3 is the only plan that covers all three topics. The v3 and v5 evaluators confirmed these are important advanced details.

**Performance/L1 as a dedicated file: from Plans v1, v3, v5.** The v2 and v4 evaluators both flagged that performance as a subsection is insufficient. This plan gives it a dedicated file following v1/v3/v5's approach, enriched with v3's CbReconfig and scratch mapping content.

**Pedagogical framing: from Plan v3.** The v3 evaluator praised the "What you will learn" / "Key takeaways" framing. This plan adopts that convention.

**Separation of hangs and gotchas: from Plans v1, v3, v5.** The v2 evaluator recommended: "Separate common hang causes from common gotchas into two files." This plan maintains that separation for better navigability during debugging.

**What was not adopted:**
- Plan v5's "anatomy first, architecture later" ordering was rejected by the v5 evaluator as "conceptually backwards" for newcomers.
- Plan v5's dedicated kernel header chapter (Ch2) was not adopted because it delays the processor model and fragments the anatomy; the same depth is achieved with a thorough file in the anatomy chapter (following v1's approach).
- Plan v4's file naming convention (chapter-number prefixes like `05_engine_pipeline.md`) was rejected as confusing (noted by the v4 evaluator).
- Plan v4's deferral of composition to Chapter 8 was rejected (noted by multiple evaluators as a forward-reference problem).

---

## Audience

This guide is written for **systems software engineers joining the TT-Blaze team or adjacent teams** who need to implement new micro-ops on Tenstorrent Blackhole hardware. The reader is expected to have:

- Working proficiency in Python and C++ (templates, constexpr, preprocessor guards)
- Basic understanding of RISC-V processor architectures and SIMD/tile-based compute
- Familiarity with the concept of circular buffers and producer/consumer synchronization
- Access to a Blackhole device (P150 harvested or P300 unharvested) or a multi-device mesh (Galaxy/T3K)
- No prior experience with TT-Blaze, tt-metal, or Tenstorrent's toolchain specifically

The guide takes the reader from zero knowledge of TT-Blaze to being able to independently design, implement, test, debug, and optimize a production micro-op, and to compose it into fused operations.

### Reading Paths

**New developer (no Blaze experience):** Read Chapters 1 through 8 sequentially.

**Experienced developer adding a compute-only op:** Skim Chapter 1, read Chapters 2-3, skip Chapters 4-5 (grids, codegen) and Chapter 6 (inter-core), read Chapter 8 section 4 (gotchas), then Chapter 9 (walkthrough).

**Developer debugging a hang:** Jump directly to Chapter 8 sections 2-3, referencing Chapter 5 section 3 for named_args inspection.

**Developer composing existing ops into a new fusion:** Read Chapter 7, referencing Chapter 3 for CBHandle chaining conventions and Chapter 6 for inter-core patterns.

**Developer building inter-core ops (after mastering compute):** Read Chapter 4 (grids) then Chapter 6 (inter-core), then Chapter 8 (debugging).

---

## Conventions

### Terminology

| Term | Definition |
|---|---|
| **Micro-op** | A single reusable operation (e.g., matmul, rmsnorm, mcast) defined by a C++ kernel header (`op.hpp`) and a Python class. The atomic building block of TT-Blaze. |
| **FusedOp** | A composition of multiple micro-ops into a single kernel dispatch. Data flows between ops through circular buffers in L1 without transiting DRAM. |
| **CB** | Circular buffer -- a FIFO queue in L1 SRAM used for producer/consumer synchronization between Tensix processors on the same core. |
| **CT arg** | Compile-time argument -- a `static constexpr` value baked into the kernel binary. Zero-cost at runtime. |
| **RT arg** | Runtime argument -- a value that can change per dispatch without recompilation. |
| **Prefix** | A string namespace (e.g., `"rmsnorm"`, `"mcast2"`) that scopes CT args to avoid collisions when composing multiple ops. Becomes a `ct_args::<prefix>::` struct in the generated header. |
| **CBHandle** | A Python dataclass carrying CB ID, page count, page size, data format, and core ranges. The mechanism by which ops pass data references to downstream ops. |
| **GridConfig** | Device-specific grid configuration (cols x rows, DRAM worker positions, phantom positions) that adapts to harvested/unharvested Blackhole variants. |
| **Phase** | One op's execution window within a fused kernel's `kernel_main()`. All processors run phases sequentially; overlap happens because empty processor code compiles out. |
| **Role flag** | A per-core boolean CT arg (e.g., `is_active`, `is_sender`) that uses `if constexpr` to compile out code on non-participating cores. |
| **DM0 / NCRISC** | Data-movement processor 0 -- handles NOC reads, DRAM fetches, and receive operations. |
| **DM1 / BRISC** | Data-movement processor 1 -- handles NOC writes, multicast, and send operations. |
| **TRISC** | Compute processor (actually three sub-cores: unpack, math, pack) -- handles tile-level compute. |
| **NOC** | Network on Chip -- the on-die interconnect for inter-core and core-to-DRAM data movement. |
| **Sender core** | The bottom-right core in the grid, typically used for gather output and external communication. |

### File Path Notation

- Paths relative to the tt-blaze repo root are written as `blaze/ops/my_op/op.py`
- Absolute source references point to `/localdev/salnahari/testing_dir/tt-blaze/`
- The notation `kernels/op.hpp` always means the file inside the op's `kernels/` subdirectory

### Code Formatting

- Python code blocks use `python` syntax highlighting
- C++ code blocks use `cpp` syntax highlighting
- Environment variables are written in `SCREAMING_SNAKE_CASE` (e.g., `BLAZE_DEBUG_KERNELS`)
- CT arg references use `ct_args::<prefix>::<field>` notation
- Processor-specific CT arg structs are referred to as `ReaderCTArgs` (DM0/NCRISC), `WriterCTArgs` (DM1/BRISC), `ComputeCTArgs` (TRISC), `CoreCTArgs` (all processors)
- `f` always refers to a `FusedProgram` instance inside `emit()` methods
- Grid configurations written as `NxM` (e.g., `13x10` means 13 columns, 10 rows)

### Diagrams

- Data flow diagrams use ASCII box-and-arrow notation
- Grid diagrams use `(col, row)` coordinate convention matching ttnn's `CoreCoord(x=col, y=row)`

### Pedagogical Framing

- Each chapter opens with a "What you will learn" list and closes with a "Key takeaways" summary
- Code examples are drawn from real ops in the codebase (rmsnorm, copy, mcast, matmul, gather) with explicit file path references
- Warning/gotcha callouts use blockquote format with a `**Warning:**` prefix

---

## Chapter List

### Chapter 1: Architecture and Mental Model

**Description:** Establishes the foundational mental model of a Tensix core, the three-processor execution model, and how TT-Blaze maps Python declarations to compiled device kernels.

**Directory:** `ch01_architecture/`

**Files:**

- `01_tensix_core_model.md`
  - The Tensix core: 5 RISC-V cores grouped into 3 logical processors (DM0/NCRISC, DM1/BRISC, TRISC with 3 sub-cores: unpack, math, pack)
  - Division of labor: DM0 handles NOC reads and data fetching; DM1 handles NOC writes, multicasts, and sends; TRISC handles unpack/math/pack compute pipelines
  - L1 SRAM as the shared workspace; circular buffers as the intra-core synchronization primitive; semaphores for inter-core coordination
  - Why all three processors run `kernel_main()` in parallel and what synchronization looks like (CB waits, semaphores)
  - How `#if defined(COMPILE_FOR_*)` guards enable per-processor code selection in a single `.hpp` file
  - Why "empty processors enable overlap": when your op has no DM1 logic, DM1 immediately advances to the next fused phase, overlapping with your TRISC compute

- `02_blaze_compilation_pipeline.md`
  - The three-phase lifecycle: graph construction (`blaze.fuse()`) -> engine pipeline (CBEngine, SemEngine, RoleEngine, CTArgEngine) -> JIT compilation and dispatch
  - How `emit()` declarations flow through engines to produce `named_args_generated.h`
  - The JIT: three compilations per kernel (one per processor), per-core specialization via CT args
  - Dispatch via `ttnn.generic_op()` and what the device receives
  - The two compilation modes: graph-based (BlazeCompiler with `blaze.fuse()`) and imperative (FusedProgram with direct `emit()` calls)
  - Reference: `blaze/compiler.py`, `blaze/cb_engine.py`, `blaze/sem_engine.py`, `blaze/role_engine.py`, `blaze/ct_args.py`

- `03_micro_op_vs_fused_op.md`
  - MicroOp: the atomic unit (C++ kernel header + Python class with `emit()`)
  - FusedOp: composition of MicroOps via chained `emit()` calls
  - How the `name` field drives auto-discovery and registration (`blaze.<name>()` and `blaze.<name>.emit()`)
  - When to write a MicroOp vs. when to compose existing ones into a FusedOp
  - The BlazeOp -> MicroOp/FusedOp class hierarchy (`blaze/blaze_op.py`)
  - The dual API surface: `blaze.<name>(...)` for graph construction inside `blaze.fuse()`, and `blaze.<name>.emit(f, ...)` for FusedProgram composition (via the `_OpHandle` proxy)

---

### Chapter 2: Anatomy of a Micro-Op

**Description:** Walks through every file in a micro-op directory, explaining the required structure, auto-discovery mechanism, and the relationship between the Python class and C++ kernel header.

**Directory:** `ch02_anatomy/`

**Files:**

- `01_directory_structure.md`
  - The canonical layout: `blaze/ops/my_op/{__init__.py, op.py, kernels/op.hpp}`
  - Auto-discovery: how `blaze/ops/__init__.py::register_all()` discovers BlazeOp subclasses via `pkgutil.iter_modules` + `inspect.getmembers`
  - The `_skip_subpackages` set (e.g., `utils`, `dsa`) and how to avoid it
  - Registration flow: `MicroOp.register()` -> `_auto_derive_from_kernel_hpp()` -> `parse_op_hpp()` -> `OpSpec`, `CTArgSchema`, `PhaseInfo`
  - The `name` field contract: becomes `blaze.<name>`, `ct_args::<prefix>::`, graph node type

- `02_python_class.md`
  - Subclassing `MicroOp` (not `BlazeOp` directly)
  - Port descriptors: `Input()`, `Output()`, `Internal(num_pages=N)` and how `__set_name__` captures the attribute name
  - Class-level configuration: `math_fidelity`, `math_approx_mode`, `op_class`, `fp32_dest_acc_en`
  - The `compose()` method: compiler entry point that bridges tensor dicts to `emit()` -- when it is required and when it can be omitted
  - The `emit()` method: the actual op definition (resource allocation + CT arg declaration); high-level overview of signature and purpose (full API reference deferred to Chapter 3)
  - Complete annotated walkthrough of `blaze/ops/rmsnorm/op.py` as a compute-only example
  - Complete annotated walkthrough of `blaze/ops/copy/op.py` as a data-movement example

- `03_cpp_kernel_header.md`
  - The `op.hpp` file structure: `namespace blaze`, outer struct, template CTArgs structs, Op class
  - CT arg struct conventions: `CoreCTArgs` (all processors), `ReaderCTArgs` (DM0), `WriterCTArgs` (DM1), `ComputeCTArgs` (TRISC)
  - The five typed field annotations and their semantics: `CB` (circular buffer ID), `Semaphore` (semaphore index), `PerCore` (per-core varying value), `Flag` (boolean role flag), `uint32_t`/`bool` (derived/computed values)
  - The `A::` template pattern: how `template <typename A> struct ReaderCTArgs` receives the generated `ct_args::<prefix>` struct as `A`
  - How `cpp_parser.py` extracts struct metadata: `_CT_TYPED_FIELD_RE`, `_PLAIN_FIELD_RE`, `_OUTER_STRUCT_RE`, `_INNER_STRUCT_RE`, `_A_NS_TOKEN_RE` regex patterns
  - The `Op` class: `init()`, `operator()()`, `teardown()` lifecycle
  - Preprocessor guards: `COMPILE_FOR_NCRISC`, `COMPILE_FOR_BRISC`, `COMPILE_FOR_TRISC`
  - Role flags and `if constexpr` for per-core code elimination
  - Annotated walkthrough of `blaze/ops/rmsnorm/kernels/op.hpp` (compute-only) and `blaze/ops/mcast/kernels/op.hpp` (inter-core)
  - Reference: `blaze/cpp_parser.py`

- `04_auto_derivation.md`
  - How `_auto_derive_from_kernel_hpp()` populates `op_class`, `ct_args`, `is_inter_core`, `has_init_teardown`, `supports_pop_control`, `pop_flags`, and `PhaseInfo` from the C++ header
  - The `ParsedKernel` and `ParsedCTArg` dataclasses and their fields
  - When to manually override auto-derived values vs. relying on the parser
  - Edge cases: aliased fields (`buffer_base = A::ncrisc_buffer_base`), derived expressions (`out_tiles = A::n * A::m`), internal constants (`MS_SEM_THRESHOLD = 1`)
  - Common mistake: using a raw `uint32_t` for what should be a `CB` type annotation

---

### Chapter 3: The emit() Method -- Bridging Python to C++

**Description:** Deep dive into the three-step structure of `emit()`: resource allocation, CT arg declaration, and output handle creation. This is the most critical method in the entire framework.

**Directory:** `ch03_emit/`

**Files:**

- `01_three_step_structure.md`
  - Step 1: Allocate resources (CBs, semaphores)
  - Step 2: Declare CT args per processor (`f.ncrisc_ct_args()`, `f.brisc_ct_args()`, `f.trisc_ct_args()`, `f.per_core_unified_ct_args()`)
  - Step 3: Return a `CBHandle` via `f.output()`
  - The `@staticmethod` convention and the `FusedProgram f` parameter
  - The `prefix` parameter: namespacing all CT args as `ct_args::<prefix>::<field>`, must be unique per op instance within a fusion
  - The `cores` parameter: which cores receive this op's CT args and role flags
  - How CT arg tuples `(f"{prefix}.field", value)` become `ct_args::<prefix>::field` in the generated header
  - The `CBHandle` chain: how `emit()` return values feed into the next op's inputs
  - Walk-through of RMSNorm.emit() showing all three steps with actual values

- `02_cb_allocation.md`
  - `f.cb_from_tensor(tensor)`: allocating a CB backed by a sharded tensor
  - `f.cb_scratch(name, num_pages, ...)`: allocating an internal scratch CB in L1
  - `f.cb_alias(source_cb_id)`: sharing L1 allocation with a different tile format (e.g., BFP8 weights as BFloat16)
  - `f.cb_from_view(view)`: allocating from an `OverlappedView` for fused weight buffers
  - `f.cb_from_shared_l1_views([...])`: multiple direct-address handles from one packed tensor
  - `f.cb_output(tensor)`: why it should NOT be used in emit() (output wiring is a composition concern)
  - Scratch CB naming convention: `f"{prefix}___<suffix>"` (three underscores) and why it matters for `scratch_mapping` lookup
  - CB page size derivation from tensor tile shape and dtype
  - The `balanced` flag on scratch CBs: when `True`, enables temporal CB reuse across phases; when `False` (e.g., mcast destinations), prevents reuse
  - CB ID limits: 64 max on Blackhole (`MAX_CB_ID` in `cb_engine.py`)
  - Disjoint-core CB sharing: same format + disjoint grids can reuse the same CB ID
  - Detailed comparison table: `cb_from_tensor` vs `cb_scratch` vs `cb_alias` vs `cb_output` -- allocation mechanics, L1 backing, disjoint-core sharing behavior
  - Reference: `blaze/fused_program.py`, `blaze/cb_engine.py`, `blaze/cb_handle.py`

- `03_role_flags_and_ct_args.md`
  - Compute ops: `f.flag(f"{prefix}.is_active", cores)` and optional pop flags
  - Inter-core ops: sender/receiver flags, NOC coordinate args
  - The contract: all flags must be emitted (even `False` ones) because the C++ Op faults on missing CT args
  - Per-RISC CT arg methods: `f.ncrisc_ct_args([(name, value)])`, `f.brisc_ct_args()`, `f.trisc_ct_args()`
  - `f.per_core_unified_ct_args()` vs. `f.per_core_ct_args()`: unified sends to all RISCs
  - How flags compile out: `if constexpr (core::is_active)` in C++ becomes a zero-cost branch when the flag is false
  - How the `CTArgEngine` auto-prefixes, type-validates, and groups CT args by RISC for the `UnifiedKernelDescriptor`
  - Type coercion: `CBHandle.__int__()` allows passing handles directly as CT arg values; `_float_to_bits()` for encoding floats as uint32_t
  - Semaphore allocation: `f.semaphore()` for global semaphores, `f.semaphore(program_semaphore=True)` for per-core program semaphores
  - Runtime args: `program.unified_rt_args()` and `program.ncrisc_per_core_rt_args()` for values that change per dispatch; `rt_args::get<>()` in C++
  - When to use CT vs RT args: the recompilation tradeoff
  - Reference: `blaze/ops/rt_args_demo/`

- `04_cbhandle_data_flow.md`
  - The `CBHandle` dataclass: `cb_id`, `num_pages`, `page_size`, `core_ranges`, `data_format`, `tile_desc`, `access_mode`
  - FIFO vs. direct-address access modes
  - `require_fifo_handle()`: guarding against accidental direct-address consumption
  - How handles chain: `Mcast.emit()` -> CBHandle -> `Matmul.emit()` -> CBHandle -> `Gather.emit()`
  - Why CBHandle enables zero-coupling between ops: each op reads `handle.cb_id` for its CT args without knowing the global CB ID numbering
  - `MultiOutput`: structured result from multi-output ops
  - How CBHandle carries metadata for downstream ops to allocate correctly (tile shape, data format, core ranges)
  - Reference: `blaze/cb_handle.py`

---

### Chapter 4: Device Grids and Multi-Device Configurations

**Description:** How to handle Blackhole's variable grid sizes, harvested devices, DRAM workers, and multi-device mesh topologies.

**Directory:** `ch04_grids/`

**Files:**

- `01_grid_config.md`
  - `GridConfig` dataclass: `grid_cols`, `grid_rows`, `dram_worker_positions`
  - Unharvested P300: 13x10 grid (130 cores), default DRAM worker positions at `(0,0), (0,3), (0,7), (0,9), (7,1), (7,4), (7,6), (7,9)`
  - Harvested P150: typically 11x10 grid (110 cores), variable DRAM positions
  - `GridConfig.from_device(device)`: querying actual hardware via `compute_with_storage_grid_size()` and `get_optimal_dram_bank_to_logical_worker_assignment()`
  - `GridConfig.default()`: the 13x10 fallback
  - Core categories and their accessor methods: `get_all_cores()`, `get_compute_cores()` (excludes DRAM workers + phantoms), `get_matmul_cores()` (excludes sender + DRAM + phantoms), `get_sender_core()`, `get_dram_worker_cores()`, `get_gate_mm_cores(num_cores)` for MoE phantom column cores
  - `build_matmul_core_grid()`, `build_mcast_receiver_grid()`, `build_ab_grids()` for CoreRangeSet construction
  - `FusedProgram` grid helpers: `f.all_cores`, `f.matmul_cores`, `f.sender_grid`, `f.mcast_receiver_grid`, `f.dram_workers`
  - `DeviceContext`: wraps device handle + `GridConfig` + `full_device_grid` CoreRangeSet; `worker_core_from_logical_core()` for NOC coordinate translation
  - How `(col, row)` tuples map to `ttnn.CoreCoord(x=col, y=row)`
  - Why grid-awareness matters: same fused op must work on both P150 (11x10) and P300 (13x10) without code changes
  - Reference: `blaze/role_engine.py`, `blaze/device_context.py`

- `02_multi_device_mesh.md`
  - MeshDevice: T3K (1x8 or 2x4), Galaxy (4x8 or 8x4) mesh topologies
  - `BlazeCompiler` mesh-aware compilation: per-device `DeviceContext`, `_split_tensors()` splits mesh tensors into per-device tensors, per-device programs
  - `MeshCompiledProgram`: wrapping per-device compiled programs
  - Submesh partitioning: `mesh_device.create_submesh()` for partial-mesh execution
  - `mesh_coord` and `mesh_device` on `FusedProgram`: how ops know which device they are on
  - CCL (Collective Communication Library) ops: `CclBroadcast`, `AllReduce` as examples
  - Fabric setup: `setup_fabric()` for inter-device NOC connections; allocates teardown + buffer_index semaphores per fabric connection
  - FabricNodeId addressing, link indices, and `ttnn.compute_fabric_connection_rt_args()`
  - Routing helpers: `compute_routing()` for broadcast patterns on 2D meshes
  - Reference: `blaze/ccl.py`, `blaze/ops/ccl_broadcast/op.py`, `blaze/ops/all_reduce/op.py`

---

### Chapter 5: Kernel Codegen Pipeline

**Description:** How the kernel `.cpp` file gets generated from the graph of `emit()` calls, the engine pipeline that processes declarations, and how to write a handwritten kernel when auto-codegen is insufficient.

**Directory:** `ch05_codegen/`

**Files:**

- `01_engine_pipeline.md`
  - The four engines and their execution order: CBEngine -> SemEngine -> RoleEngine -> CTArgEngine
  - **CBEngine**: assigns CB IDs to edges and external ports; disjoint-core format-key sharing for tensor-backed CBs with matching format keys; alias and scratch allocation; `CBAssignment` output; MAX_CB_ID=64 on Blackhole
  - **SemEngine**: identifies inter-core sync nodes (gather, mcast, CCL); assigns sequential semaphore indices; mcast sender grouping for persistent state sharing; `SemAssignment` output; GlobalSemaphore objects
  - **RoleEngine**: derives per-core boolean role flags from op grid placements; `f.flag()` tuples; `CoreRole` and `RoleGroup` outputs; how flags compose across multiple ops in a fusion
  - **CTArgEngine**: resolves final CT arg values from CB/sem/grid/user sources; produces `{processor: [(name, value)]}` dict; type validation and auto-prefixing
  - The shadow graph: how `FusedProgram.output()` records `OpNode`s during emit(), enabling codegen to reconstruct the phase sequence
  - `BlazeCompiler.compile()` as the orchestrator: graph validation, engine pipeline, kernel generation, JIT dispatch
  - Reference: `blaze/cb_engine.py`, `blaze/sem_engine.py`, `blaze/role_engine.py`, `blaze/ct_args.py`

- `02_auto_generated_kernels.md`
  - `kernel_codegen.py`: the core algorithm that turns a `BlazeGraph` into C++ source via `generate_kernel()`
  - `PhaseInfo` registry: `cpp_type`, `alias_suffix`, `has_init_teardown`, `setup_method`, `init_is_empty`, `teardown_is_empty`
  - Phase ordering: node insertion order in the graph matches `emit()` call order in Python; each `emit()` call creates one Phase
  - The `PhaseDecl` dataclass and grouping by `(prefix, op_type)`
  - Type alias generation: `_to_alias()` maps prefix + op type to C++ alias (e.g., `("act_mcast", "mcast", "Mcast")` -> `using ActMcast = blaze::Mcast::Op<ct_args::act_mcast>`)
  - `_to_ct_args_type()`: converts `prefix.subprefix` to `ct_args::prefix__subprefix`
  - `kernel_main()` generation: `DeviceZoneScopedN` profiler markers, init/run/teardown lifecycle
  - Mcast group optimization: leader/follower pattern for persistent sender state sharing across multiple mcast phases (`setup_src<>()` and `run_as<>()`)
  - Empty lifecycle elision: `init_is_empty` / `teardown_is_empty` from cpp_parser skip no-op calls
  - Content-hashed output files: `generated/kernels/<name>_<hash>.cpp`

- `03_named_args_generated_header.md`
  - The `named_args_generated.h` file: auto-generated, auto-included via `-include`
  - `ct_args::` namespace: per-prefix structs with `static constexpr uint32_t` fields
  - `rt_args::` namespace: runtime arg accessors via `rt_args::get<>()`
  - Per-core CT args: how `PerCore` fields produce different binaries per core (the `PerCoreCompileTimeDescriptor`)
  - Triple compilation: JIT compiles the kernel three times with `COMPILE_FOR_NCRISC`, `COMPILE_FOR_BRISC`, `COMPILE_FOR_TRISC`
  - Location: `~/.cache/tt-metal-cache/<kernel>/named_args_generated.h`
  - How to inspect: `program.generated_kernel`, `blaze.compile_engines(ctx.graph).ct_tuples`

- `04_handwritten_kernels.md`
  - When to handwrite: custom phase ordering, Tracy profiler markers, overlapping phases, non-standard lifecycle
  - The `kernel` class attribute on `FusedOp`: pointing to a `.cpp` file
  - Handwritten kernel structure: includes, role flag structs, type aliases from `ct_args::`, `kernel_main()` with lifecycle calls
  - How to maintain consistency between the handwritten kernel and `emit()` call sequence
  - How to mix handwritten kernels with codegen: using `program.generated_kernel` to inspect what codegen produces, then customizing
  - Example walkthrough: `blaze/ops/shared_expert/kernels/shared_expert_kernel.cpp`

---

### Chapter 6: Inter-Core Operations and NOC Communication

**Description:** The sender/receiver pattern for data movement across Tensix cores, NOC coordinates, semaphore protocols, and the specific implementation patterns for mcast, gather, scatter, and CCL ops.

**Directory:** `ch06_inter_core/`

**Files:**

- `01_sender_receiver_pattern.md`
  - The fundamental pattern: sender core reads data from L1 CB, writes to remote core(s) via NOC; receiver core(s) wait on semaphore, consume from local CB
  - NOC0 vs NOC1: two independent networks, avoiding congestion by splitting read/write traffic
  - Semaphore protocols: `SemProtocol.SENDER`, `SemProtocol.RECEIVER`, `SemProtocol.NOC0_RECEIVER`, `SemProtocol.NOC1_RECEIVER`
  - NOC coordinate translation: `f._ctx.worker_core_from_logical_core()` converting logical `(col, row)` to physical NOC addresses
  - How `is_sender` / `is_receiver` flags in emit() assign roles to different core subsets
  - NOC coordinates: `f.noc_start`, `f.noc_end`, `f.noc_sender` and how they derive from `DeviceContext`
  - How data_size_bytes flows through: `num_pages * page_size` from CBHandle, determining NOC transfer size

- `02_mcast_gather_scatter.md`
  - **Mcast** (one-to-many): sender multicasts data to all receiver cores via NOC multicast
    - Sender/receiver role flags: `f.flag(f"{prefix}.is_sender", f.sender_grid)`, `f.flag(f"{prefix}.is_receiver", f.mcast_receiver_grid)`
    - Sender semaphore and receiver semaphore allocation
    - DM1 (BRISC) handles the NOC mcast send; DM0 (NCRISC) handles receiver notification
    - The `setup_src<>()` and `run_as<>()` template methods for multi-phase mcast with persistent state
  - **Gather** (many-to-one): scattered cores send data to a single receiver (typically sender core)
    - NOC coordinate computation for the receiver
    - Per-sender semaphore pairs (noc0 + noc1) for flow control
    - Row-major vs column-major core ordering
  - **Scatter**: distributing data from one core to many; raw vs CB-mediated scatter
  - **Copy**: generic L1 data movement with configurable CB sync, processor selection, local vs remote copy
  - NOC congestion avoidance: linked/posted modes, data size considerations, noc0 vs noc1 channel selection
  - How inter-core ops compose with compute ops: Mcast feeds matmul (data arrives at all cores), Gather collects matmul results (data leaves from all cores to sender)

- `03_ccl_and_fabric.md`
  - CCL ops for multi-device: `ccl_broadcast` (fabric-based all-to-all), `all_reduce` (reduce across mesh), `scatter` (shard across devices)
  - Fabric connection setup: `ccl.setup_fabric()` allocating teardown and buffer-index semaphores, computing RT args via `ttnn.compute_fabric_connection_rt_args()`
  - FabricNodeId addressing and link indices
  - The `_fabric_cores_per_risc_per_ccl` tracking dict
  - Barrier ops: `barrier_sender` / `barrier_receiver` for mesh synchronization
  - Pipeline stage sync for multi-dispatch scenarios
  - Reference: `blaze/ccl.py`, `blaze/ops/ccl_broadcast/op.py`

---

### Chapter 7: Composing Micro-Ops into Fused Operations

**Description:** How to build FusedOps by chaining micro-op `emit()` calls, managing shared resources, and handling complex data-flow patterns.

**Directory:** `ch07_composition/`

**Files:**

- `01_fused_op_structure.md`
  - `FusedOp` class: subclass of `BlazeOp`, requires `compose()` override, optionally provides `emit()` for embedding as a sub-pipeline
  - Directory layout: `blaze/ops/my_fused_op/{__init__.py, op.py, kernels/}` (kernels/ optional)
  - The `kernel` attribute: selects handwritten vs codegen
  - `compose()` vs. `emit()`: why two methods exist (compiler convention vs. free-form reuse)
  - The `prefix` -> `ct_args::` -> phase mapping: how `emit()` call order becomes kernel phase order
  - The graph API path: `blaze.fuse()` context manager -> `blaze.<op>()` calls -> `BlazeGraph` DAG -> `BlazeCompiler.compile()` -> `compose()` -> `emit()`

- `02_prefix_namespaces_and_shared_resources.md`
  - Unique prefix assignment: two instances of the same MicroOp need different prefixes (e.g., `"mcast"` and `"mcast2"`, or `"act_mcast"` and `"mcast"`)
  - Child prefix convention: `BlazeOp.child_prefix(prefix, name)` using `__` delimiter (`CB_NAME_DELIMITER`)
  - Shared semaphores: allocating one `f.semaphore()` and passing it to multiple `emit()` calls (e.g., `sender_sem` shared across mcast phases)
  - Shared scratch CBs: reusing scratch buffers between non-overlapping phases
  - `ABGrid`: A/B core grid partitioning for KN-sliced matmul gate/up branches
  - Embedding sub-FusedOps: calling another FusedOp's `emit()` as a sub-pipeline (e.g., `DownProj.emit()` inside `SharedExpert.emit()`)

- `03_real_world_example.md`
  - Full annotated walkthrough of `SharedExpert` FusedOp: 5 micro-ops, 130 cores, 15 CBs
  - How the data flows: `Mcast -> KNMatmul -> GatedReduce -> Mcast -> DownProj`
  - Shared semaphore wiring across phases
  - Sub-FusedOp embedding: `DownProj.emit()` itself chains `matmul -> mcast -> residual_add -> gather`
  - Tracing CBHandle propagation end-to-end: `act_mcast::dst == matmul::in0` etc.
  - Reference: `blaze/ops/shared_expert/op.py`

---

### Chapter 8: Testing, Debugging, and Common Pitfalls

**Description:** Practical guide to writing tests, diagnosing hangs and correctness issues, using debugging tools, and avoiding the most common mistakes.

**Directory:** `ch08_testing_debugging/`

**Files:**

- `01_writing_tests.md`
  - Test file location: `tests/blaze/micro-ops/common/test_my_op.py` (common) or category subdirectories; `tests/blaze/fused_ops/` for fused ops
  - Two test paths: FusedProgram composition (direct `emit()`) vs. graph compilation (`BlazeCompiler`)
  - Graph-based test structure: `blaze.fuse()` -> `ExternalTensor` placeholders -> `BlazeCompiler.compile()` -> `program.run()` -> `comp_pcc()`
  - FusedProgram-based test anatomy: create `FusedProgram(kernel=..., device=device)`, call `emit()` directly, `f.build()` and `f.run()`
  - Setting up device tensors: `ttnn.from_torch()`, shard specs, memory configs, mesh mappers
  - Golden reference computation: PyTorch equivalents, `comp_pcc()` thresholds (0.999 for fp32, relaxed for bf16)
  - Test fixtures: `mesh_device` (1x1 mesh in tests), `conftest.py` setup, `_SILICON_PATHS` for CI filtering
  - Parametrized tests: widths, data formats, tile shapes, epsilon values, grid sizes
  - Reference: `tests/blaze/micro-ops/common/test_rmsnorm.py`, `tests/blaze/conftest.py`

- `02_debugging_tools.md`
  - `BLAZE_DEBUG_KERNELS`: injecting DPRINT markers at phase boundaries; supports RISC filtering (`BLAZE_DEBUG_KERNELS=ncrisc,trisc`); values `"1"`/`"all"` for all RISCs; requires `TT_METAL_DPRINT_CORES` to be set for output visibility
  - `TT_METAL_DPRINT_CORES`: enabling DPRINT output for specific cores
  - `BLAZE_L1_PROFILE`: reporting CB type, L1 address, dtype, tile shape on every `program.run()`; output prefixed with `[l1_profile]`; also callable as `print_cb_stats(program)` and `print_kernel_stats(program)` from `blaze/l1_profile.py`
  - `BLAZE_DISABLE_TENSOR_CB_SHARE`: bisection escape hatch for cross-tensor CB sharing bugs
  - Inspecting CT arg values: `blaze.compile_engines(ctx.graph).ct_tuples`
  - Inspecting generated kernels: `program.generated_kernel`, on-disk at `generated/kernels/`
  - Inspecting `named_args_generated.h`: location in `~/.cache/tt-metal-cache/`; verify CB IDs, semaphore indices, role flags
  - The visualizer: `blaze.visualize(ctx.graph)` for interactive HTML graph inspection with CB assignments, semaphore IDs, and core grid overlays
  - Tracy profiler integration: `DeviceZoneScopedN` markers in handwritten kernels

- `03_common_hang_causes.md`
  - CB deadlocks: producer pushes N pages but consumer waits for M > N (or vice versa); producer never pushes (missing `init()` push for sharded inputs)
  - Semaphore mismatches: sender increments but no receiver waits (or receiver waits but sender never signals); incorrect initial values; semaphore ID collision between phases
  - Missing role flags: an op's `init()` runs on a core where it shouldn't because `is_active` flag was not emitted
  - Phase ordering errors: a consumer phase runs before its producer phase completes
  - NOC congestion: too many concurrent multicasts or gathers saturating NOC bandwidth
  - DRAM worker conflicts: ops accidentally placed on DRAM worker cores
  - Grid mismatch: CT args declared for a core range that doesn't match the actual sharded tensor grid
  - How to diagnose: enable DPRINT, check which phase each core is stuck in, inspect CB state, verify CB push/pop balance

- `04_common_gotchas.md`
  - CT arg name mismatches: field name in C++ doesn't match `f"{prefix}.field"` in Python (produces a compile error, but can be confusing)
  - Missing flags: not emitting a `False` flag causes a CT arg fault (all flags must be declared, even disabled ones)
  - Wrong RISC struct: putting a CB field in `WriterCTArgs` when it should be in `ReaderCTArgs`; the value silently goes to the wrong processor
  - CB page mismatches: `num_pages` in `cb_scratch()` doesn't match what the kernel expects; `cb_from_tensor` infers page size from tensor tile while `cb_scratch` requires explicit `page_size`
  - Tile geometry mismatches: input uses `1x32` row tiles but the compute kernel expects `32x32` tiles (need `interpret_tile()`); mismatched tile descriptors between producer and consumer CBs
  - Scratch CB naming: forgetting the triple-underscore `___` separator in `cb_scratch` names
  - `cb_output()` in `emit()`: using `cb_output` instead of `cb_from_tensor` (output wiring is a composition concern)
  - Pop control: not wiring `pop_input` flags when the op supports pop control; leaving data in a CB that a later phase expects to reuse
  - `compose()` forgetting to pass `prefix` and `cores` from `user_args`
  - Direct-address handles passed to FIFO-expecting ops (need `require_fifo_handle()`)
  - Forgetting `compose()`: the op works via emit() in fused pipelines but fails when compiled via BlazeCompiler graph API

---

### Chapter 9: End-to-End Workflow and Performance

**Description:** A complete walkthrough of adding a new production micro-op from design to deployment, plus performance optimization techniques for L1 memory management.

**Directory:** `ch09_workflow_performance/`

**Files:**

- `01_end_to_end_walkthrough.md`
  - Choosing the operation: a hypothetical "ScaledAdd" op (output = alpha * input + beta * bias) that exercises compute, CB allocation, and role flags
  - Step 1: Creating the directory (`blaze/ops/scaled_add/`) and the three required files
  - Step 2: Writing the C++ kernel header -- designing CT arg structs for each processor, implementing lifecycle, using correct typed annotations
  - Step 3: Writing the Python class -- ports, `emit()`, `compose()`
  - Step 4: Verifying auto-discovery -- running `python -c "import blaze; print(blaze.list_ops())"` to confirm registration
  - Step 5: Running `cpp_parser.py` auto-derivation and verifying the extracted metadata
  - Step 6: Writing the first test -- graph-based compilation, simple input, verify PCC against PyTorch golden
  - Step 7: Debugging a deliberate bug -- wrong `num_tiles` causing a CB deadlock, diagnosed with `BLAZE_DEBUG_KERNELS` and DPRINT
  - Step 8: Composing into a FusedOp -- chaining with Mcast upstream and Gather downstream, assigning unique prefixes, verifying end-to-end
  - Step 9: Multi-core and multi-device testing -- verifying on full grids, harvested devices, using GridConfig-aware grid selection
  - Code review checklist: what reviewers look for

- `02_performance_and_l1_management.md`
  - CB scratch vs. CB from tensor: when to use each and the L1 cost implications
  - CB reuse across phases: how the allocator shares CB IDs for disjoint-core, same-format CBs
  - Temporal CB reuse: scratch CBs that can be repurposed across non-overlapping phases; conditions for temporal reuse
  - Direct-address mode: `CBAccessMode.DIRECT_ADDRESS` for weight tensors, avoiding FIFO overhead; `buffer_address()` + byte offset access
  - `OverlappedView` for fused weight buffers: packing multiple weight tensors into one L1 allocation
  - `CbReconfig` for multi-phase programs: how `CircularBufferIdManager` reprograms CB slots between phases to fit more ops in limited L1; when and why to use it
  - `all_scratch_mapped=True` compilation mode: where scratch CBs are backed by a pre-allocated fused buffer
  - `cb_alias()` for tile format optimization: reading weights in a different format without doubling L1 usage
  - L1 pressure estimation: `_tensor_shard_bytes()`, tile page size calculations
  - L1 profiling workflow: enable `BLAZE_L1_PROFILE=1`, run, parse `[l1_profile]` output, identify memory pressure points
  - Minimizing CB count: the 64 CB ID limit and strategies for staying under it
  - Phase overlap: how empty processor bodies enable natural pipelining between adjacent phases

---

## Cross-Chapter Dependencies

| Chapter | Depends On | Concepts Referenced |
|---|---|---|
| **Ch1: Architecture** | (none) | Foundational -- introduces all core concepts |
| **Ch2: Anatomy** | Ch1 | Tensix processor model, CB basics, compilation pipeline overview |
| **Ch3: emit()** | Ch1, Ch2 | CT arg struct conventions from Ch2; engine pipeline from Ch1; port descriptors from Ch2 |
| **Ch4: Grids** | Ch1, Ch3 | Grid helpers used in `emit()` from Ch3; core categories from Ch1; DeviceContext from Ch1 |
| **Ch5: Codegen** | Ch2, Ch3 | `emit()` call sequence from Ch3; PhaseInfo from Ch2; CT arg namespaces from Ch3 |
| **Ch6: Inter-Core** | Ch3, Ch4 | Semaphore allocation and flag patterns from Ch3; core grid layout and NOC coordinates from Ch4 |
| **Ch7: Composition** | Ch3, Ch5, Ch6 | `emit()` chaining from Ch3; codegen phases from Ch5; inter-core op semantics from Ch6 |
| **Ch8: Testing/Debugging** | Ch1-Ch7 | All prior concepts are needed to write tests and diagnose issues |
| **Ch9: Workflow/Performance** | Ch1-Ch8 | Integrates everything into a production workflow; debugging skills from Ch8; L1 concepts from Ch3 |

### Specific forward/backward references:

- Ch2 `03_cpp_kernel_header.md` references Ch1's Tensix processor model and introduces CT arg types used heavily in Ch3
- Ch3 `02_cb_allocation.md` introduces CB concepts used in Ch7's composition patterns and Ch9's L1 management
- Ch4 `01_grid_config.md` explains grid helpers that appear in every `emit()` example from Ch3 onward
- Ch4 `02_multi_device_mesh.md` builds on single-device concepts from Ch1-Ch3 and extends them to mesh topologies
- Ch5 `01_engine_pipeline.md` explains how the `emit()` declarations from Ch3 become compiled artifacts, connecting the Python layer to the codegen layer
- Ch6 `01_sender_receiver_pattern.md` requires understanding of semaphores (Ch3), grids (Ch4), and flags (Ch3)
- Ch6 `02_mcast_gather_scatter.md` builds on the sender/receiver concepts and provides the inter-core op semantics needed by Ch7's composition examples
- Ch7 `03_real_world_example.md` references mcast, matmul, gather, and residual_add ops, requiring inter-core knowledge from Ch6
- Ch8 `03_common_hang_causes.md` requires understanding of CBs (Ch1/Ch3), semaphores (Ch3/Ch6), and phase ordering (Ch5)
- Ch9 `02_performance_and_l1_management.md` builds on CB allocation methods from Ch3, engine pipeline from Ch5, and practical experience from Ch8
