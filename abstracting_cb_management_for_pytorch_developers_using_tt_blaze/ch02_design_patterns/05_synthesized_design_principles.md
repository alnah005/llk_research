# Synthesized Design Principles: A Decision Framework for TensorAdapter

**What you will learn:**

- Seven design principles extracted from the six frameworks surveyed in this chapter (TT-Symbiote, TVM, Triton, XLA, MLIR, torch.compile)
- A decision matrix mapping each of the 16 design patterns (DP1-DP16) to adopt, adapt, or reject for TensorAdapter
- A coverage matrix showing how each framework handles the 12 manual decisions from Chapter 1
- Six architectural invariants that any CB abstraction layer must preserve
- Five novel capabilities that must be invented (no existing framework provides them)
- The architectural blueprint connecting principles to concrete TensorAdapter components

---

## The Seven Principles

Each principle is grounded in evidence from multiple frameworks. The principle name encodes the actionable directive; the evidence section explains why.

---

### P1: Separate Computation Intent from Execution Strategy

**Source:** TVM (schedule vs. computation), MLIR (dialect layers), Triton (constexpr block sizes)

**Evidence:**

| Framework | How It Separates |
|-----------|-----------------|
| TVM | `te.compute` (algorithm) vs. `te.create_schedule` + `s.split/s.reorder` (schedule) |
| MLIR | `linalg.matmul` (algorithm) vs. `memref.alloc` + `memref.copy` (buffer management) |
| Triton | Algorithm written once; tile sizes varied by `@autotune` via `tl.constexpr` parameters |
| Blaze (current) | **Violates this principle**: `emit()` interleaves algorithm logic with `cb_scratch()`, CT arg wiring, and L1 calculations |

The developer's expression of *what* to compute must be separable from *how* to execute it on hardware. In current Blaze, changing a tile shape cascades into `page_size`, `num_pages`, CT args, and L1 calculations. TensorAdapter decouples these: the developer declares the computation, and the adapter applies tile decomposition, padding, format selection, and CB sizing as a separate pass. See [Chapter 3: TensorAdapter Architecture](../ch03_tensor_adapter/index.md).

---

### P2: Carry Metadata from Both Worlds Simultaneously

**Source:** TT-Symbiote (`TorchTTNNTensor` dual-backing), XLA (HLO shape tracking), MLIR (type preservation through lowering)

**Evidence:**

| Framework | Logical Representation | Physical Representation |
|-----------|----------------------|------------------------|
| TT-Symbiote | `.shape` via `DistributedTensorConfig.get_logical_shape()` | Per-device shard shape in `.ttnn_tensor.shape` |
| XLA | JAX tensor `.shape` | TPU-padded layout (128x128 for matmul) |
| MLIR | `tensor<33x65xf32>` | `memref<64x96xf32>` after bufferization |
| Triton | Kernel arguments M, N, K | `BLOCK_M x BLOCK_N` loaded blocks |
| torch.compile | FX node `.meta["val"].shape` | Inductor tile blocks |

The core data structure must carry PyTorch-world information (logical shape, dtype) and hardware-world information (tile shape, padded shape, data format, CB constraints) simultaneously. Symbiote's `TorchTTNNTensor` carries both `.elem` and `.ttnn_tensor`. TensorAdapter's `ShapeDescriptor` extends this to a richer dual record:

```python
# PROPOSED -- ShapeDescriptor (illustrative, not prescriptive)
@dataclass
class ShapeDescriptor:
    logical_shape: tuple[int, ...]     # PyTorch world: (1, 4096)
    padded_shape: tuple[int, ...]      # Hardware world: (32, 4096)
    tile_shape: tuple[int, int]        # (32, 32)
    data_format: ttnn.DataType         # ttnn.bfloat16
    tile_grid: tuple[int, int]         # (1, 128)
    page_size: int                     # 2048
    padding_fill: float                # 0.0
```

> **Warning:** The multi-level representation must be **lossless** -- each level must preserve all information from previous levels. If the padded shape discards the logical shape, the system cannot correctly slice results back to user-visible dimensions. MLIR enforces this through its type system; TensorAdapter must enforce it through the ShapeDescriptor's field structure.

See [Chapter 4: Tile Decomposition](../ch04_tile_decomposition/index.md).

---

### P3: Validate at Each Lowering Step, Not Just at the End

**Source:** TVM (multi-pass validation), MLIR (per-dialect invariants), XLA (shape checking at every HLO node)

**Evidence:**

| Framework | Lowering Levels | Where Validation Occurs |
|-----------|----------------|------------------------|
| TVM | Relay -> TIR -> target (3 levels) | Type inference, bounds checking, memory scope validation |
| MLIR | linalg -> memref -> scf -> gpu (4 levels) | Per-dialect invariant checking at each lowering pass |
| torch.compile | FX -> Inductor IR -> kernel (3 levels) | Shape propagation, memory planning, codegen validation |
| Blaze (current) | FusedProgram -> ProgramDescriptor (2 levels) | **No compile-time CB validation** -- L1 overflow detected via runtime corruption |

Each transformation must validate before passing to the next. TensorAdapter implements a validation chain:

1. **ShapeEngine.decompose** -- validates dimensions are positive, op_hint is recognized
2. **PaddingPolicy.apply** -- validates fill value correctness, tile alignment
3. **FormatPolicy.select** -- validates format is supported by the op
4. **CBSizer.compute** -- validates total L1 usage per core and CB ID count < 64
5. **FusedProgram.cb_from_tensor()** -- existing Blaze validation (unchanged)

The key principle: **validation happens at each lowering step, not just at the end**. TVM catches shape mismatches at the Relay level, buffer overflow at the TIR level, and target violations at codegen. TensorAdapter should catch tile incompatibilities before CB allocation, format mismatches before building the FusedProgram, and L1 overflows before kernel compilation.

See [Chapter 5: Padding](../ch05_padding/index.md) and [Chapter 7: CB Sizing](../ch07_cb_sizing/index.md).

---

### P4: Provide Defaults with Per-Decision Overrides

**Source:** TVM (auto-tune with manual schedule), Triton (constexpr override), XLA (no override -- the anti-pattern)

**Evidence:**

| Framework | Override Mechanism |
|-----------|-------------------|
| Triton | `@autotune` with `tl.constexpr`: framework searches, user can fix values via explicit `Config` |
| TVM | AutoTVM/Ansor search, but `s[C].split(axis, factor=user_value)` overrides |
| torch.compile | `torch._inductor.config` for manual tile sizes; `torch.compiler.allow_in_graph` |
| XLA | Limited: `jax.lax.with_sharding_constraint` for sharding, no tile override |
| TT-Symbiote | `exclude_replacement` set, `_bypass_tensor_wrapping`, pluggable dispatchers |
| Blaze (current) | **No defaults** -- all 12 decisions are required manually |

XLA demonstrates the failure mode: no escape hatch when the compiler makes a suboptimal tiling choice. TensorAdapter provides three tiers of control (matching the developer journey from [Chapter 1: Abstraction Goals](../ch01_the_gap/04_abstraction_goals_and_scope.md)):

| Tier | Style | Developer Experience | Analog |
|------|-------|---------------------|--------|
| **Level 0: Automatic** | XLA-style | Provide tensor + op hint; all CB decisions automatic | `TensorAdapter.adapt(tensor, "matmul.in0")` |
| **Level 1: Configurable** | Triton-style | Override specific parameters (tile_shape, num_pages) | `TensorAdapter.adapt(tensor, "matmul.in0", tile_shape=(16, 32))` |
| **Level 2: Manual** | Blaze-style | Full control via raw `FusedProgram.cb_scratch()` | Existing Blaze API, unchanged |

Critically, overriding one decision (tile shape) does not require overriding all others. See [Chapter 10: End-to-End Developer API](../ch10_end_to_end/index.md).

---

### P5: Operate on Graphs, Not Individual Tensors

**Source:** torch.compile/Inductor (FX graph), TT-Forge (whole-graph compilation), TVM (Relay fusion)

Cross-op decisions -- format negotiation, L1 budgeting, CB reuse -- require graph-level visibility. In a matmul -> softmax -> matmul chain, per-op format selection inserts conversions at every boundary. Graph-level analysis negotiates formats to minimize conversions.

The graph-level view enables decisions that single-op analysis cannot make:

- **Format negotiation**: If op A produces bfloat8_b but op B needs bfloat16, insert a conversion CB at the boundary.
- **L1 budgeting**: Sum all CBs that are live simultaneously on the same core. If the total exceeds the budget, reduce num_pages or change formats.
- **CB reuse**: Two ops that are never live simultaneously can share a CB ID, reducing pressure on the 64-slot limit.

See [Chapter 3](../ch03_tensor_adapter/index.md) and [Chapter 9: Op Fusion](../ch09_op_fusion/index.md).

---

### P6: Per-Operation Padding Semantics

**Source:** Triton (per-load `other=`), XLA (op-specific pad rules), Blaze PaddedRMSNorm

**Evidence:**

| Framework | How It Handles Padding Values |
|-----------|------------------------------|
| Triton | `other` parameter on `tl.load`: `0.0` for matmul, `-float('inf')` for softmax |
| XLA | Compiler's padding insertion pass selects fill value per HLO op type |
| TVM | Developer specifies fill value in `te.compute` with `if_then_else` |
| MLIR | Pad op carries explicit padding value as an attribute |
| torch.compile | Inductor handles padding during Triton codegen using mask/other |

Padding fill values are determined by the consuming operation, not the producing tensor:

| Operation | Correct Fill | Incorrect Fill Effect |
|-----------|-------------|----------------------|
| Matmul | 0.0 | Spurious dot-product contributions |
| Softmax | -inf | Non-zero probability for padding positions |
| RMSNorm | 0.0 | Inflated variance estimate |
| Max reduce | -inf | Bias toward zero when values are negative |
| Multiply | 1.0 | Zeroes out real values |

> **Warning:** Incorrect padding values are silent correctness bugs. A softmax with `0.0` padding instead of `-inf` padding will produce numerically wrong results without any error, warning, or crash. The padding strategy registry must be treated as a correctness-critical component with the same rigor as the kernel implementation itself.

TensorAdapter maintains a centralized, extensible `PADDING_REGISTRY` keyed by op type. The canonical padding value table and the full PaddingPolicy design are in [Chapter 5: Padding](../ch05_padding/index.md).

---

### P7: Transparent Interception That Preserves Identity

**Source:** TT-Symbiote (name-preserving module replacement), torch.compile (eager-identical semantics)

The abstraction layer must be invisible to the developer's code and debugging experience. Symbiote preserves module tree names via `override_children_module_names()`; torch.compile preserves eager semantics via `torch._dynamo`'s frame evaluation hook. The adoption barrier for a new hardware abstraction is directly proportional to how much code the developer must change.

TensorAdapter should integrate with existing interception mechanisms (Symbiote's `__torch_dispatch__` for eager-mode, torch.compile's FX capture for graph-mode) rather than requiring a separate entry point. See [Chapter 10: End-to-End Developer API](../ch10_end_to_end/index.md).

---

## Decision Matrix: Adopt, Adapt, or Reject

The 16 design patterns from Files 01-04 are classified by their applicability to TensorAdapter:

| Design Pattern | Source | Decision | Rationale |
|---------------|--------|----------|-----------|
| **DP1: Module replacement** | Symbiote | **Adapt** | Intercept at tensor-op level, not module level. Keep name preservation and structural context. |
| **DP2: Dual-backing tensor** | Symbiote | **Adopt** | ShapeDescriptor carries PyTorch + hardware metadata simultaneously. |
| **DP3: `__torch_dispatch__`** | Symbiote | **Adapt** | Proven routing mechanism. TensorAdapter adapts it as a secondary entry point for eager-mode integration, alongside its primary graph/compilation-time path. |
| **DP4: Four-phase lifecycle** | Symbiote | **Adopt** | analyze -> validate -> materialize with boolean guards prevents ordering violations. |
| **DP5: Schedule primitives** | TVM | **Adapt** | Deterministic rules replace full schedule language. Tenstorrent's constrained tile geometry makes search unnecessary. |
| **DP6: Buffer + memory scopes** | TVM | **Adapt** | Map DRAM/L1/CB scopes; Tenstorrent CBs are richer than TVM buffers (FIFO, pages, 64-slot limit). |
| **DP7: Configurable tile factors** | TVM | **Adopt** | Default tile shape with per-op override. The default-with-override pattern. |
| **DP8: Validation pipeline** | TVM | **Adopt** | Per-step validation: shape -> padding -> format -> CB -> L1 budget. |
| **DP9: Mask-based padding** | Triton | **Adapt** | No hardware masks on Tenstorrent; adopt the per-op fill value concept and co-location of fill value with the load. |
| **DP10: Constexpr block sizes** | Triton | **Adopt** | Map to CT args. Auto-compute tile counts from shape analysis. |
| **DP11: Auto padding + stripping** | XLA | **Adopt** | Auto-pad to tile boundary; track logical shape for output slicing. |
| **DP12: Shape tracking** | XLA | **Adopt** | Triple-shape invariant (logical, padded, tile_grid) at every pipeline node. |
| **DP13: Dialect layers** | MLIR | **Adopt** | Three-layer: Tensor -> Tile -> CB with explicit lowering passes and invariants. |
| **DP14: FX graph capture** | torch.compile | **Adapt** | BlazeGraph primary entry; FX as secondary entry point via torch.compile backend. |
| **DP15: Pluggable backend** | torch.compile | **Adopt** | TensorAdapter as pluggable analysis backend for BlazeCompiler or torch.compile. |
| **DP16: Compiler + manual** | TT-Forge + Blaze | **Adopt** | Analyze like a compiler, execute via existing emit() calls. Preserves kernel developer workflow. |

**Summary:** 11 adopted, 5 adapted, 0 rejected. The 5 adapted patterns operate at a level that does not map directly to Tenstorrent hardware (Triton's hardware masks, TVM's full schedule language, Symbiote's module-level granularity) but whose conceptual models are valuable when translated to CB semantics.

---

## Coverage Matrix: How Each Framework Handles the 12 Decisions

This table maps the 12 manual decisions identified in [Chapter 1: The Manual Plumbing Burden](../ch01_the_gap/03_the_manual_plumbing_burden.md) across all surveyed frameworks:

| Decision | Blaze (Current) | Symbiote | TVM | Triton | XLA | torch.compile | TensorAdapter (Proposed) |
|----------|-----------------|----------|-----|--------|-----|---------------|--------------------------|
| 1. Tile shape | Manual | N/A | Schedule | constexpr | Auto | Heuristic | ShapeEngine |
| 2. Padding | Manual variant ops | N/A | N/A | Mask | Auto pad+strip | Codegen | PaddingPolicy |
| 3. Fill value | Hardcoded per op | N/A | N/A | Per-load | Per-op rule | Per-op | Per-op registry |
| 4. Data format | Manual per CB | Auto | Config | dtype | Auto | Auto | FormatPolicy |
| 5. num_pages | Manual | N/A | Schedule | Block | Auto | Alloc | CBSizer |
| 6. page_size | Manual | N/A | Schedule | Block | Auto | Alloc | Derived from tile+dtype |
| 7. Core placement | Manual grid | Auto | bind() | Grid | Auto | N/A | GridPlanner |
| 8. CB ID | Sequential | N/A | Buffer ID | N/A | N/A | N/A | CBEngine (existing) |
| 9. Semaphores | SemEngine | N/A | N/A | N/A | N/A | N/A | SemEngine (existing) |
| 10. CT args | Manual | N/A | Config | constexpr | Compiler | Compiler | Auto-derive from shapes |
| 11. Broadcasting | Manual | Auto | Transform | Manual | Auto | Auto | BroadcastResolver |
| 12. L1 budget | No check | N/A | Cost model | Shared mem | HBM | Scheduler | L1BudgetValidator |

All 12 decisions move from "Manual" to automatic computation. CB ID and semaphore allocation are delegated to existing engines (CBEngine, SemEngine) which already provide automation.

---

## Architectural Invariants

Any CB abstraction layer (including TensorAdapter) must preserve these six invariants established by the hardware and existing Blaze infrastructure:

**1. CB ID uniqueness within a phase.** No two simultaneously-live CBs may share an ID on the same core range. This is enforced by CBEngine's interval coloring (`compact_cb_ids`) and must not be violated by auto-generated allocations.

**2. Page size formula.** `page_size = dtype_bytes_per_element x tile_h x tile_w` (with format-specific adjustments for block formats like bfloat8_b). This hardware invariant (from Tensix architecture) cannot be overridden. TensorAdapter must compute page sizes from dtype and tile geometry, never from user input.

**3. 64-CB hard limit per program phase.** Blackhole hardware enforces this. TensorAdapter must count allocations and insert `reconfig()` boundaries before exceeding this limit.

**4. FIFO vs. DIRECT_ADDRESS exclusivity.** A CB operates in exactly one access mode. TensorAdapter must track access modes and prevent mixing. The `require_fifo_handle` pattern from Blaze enforces this at the API level.

**5. Tensor-backed CBs require materialized device buffers.** A `CBHandle` with `is_tensor_backed=True` must reference a `ttnn.Tensor` with a valid `buffer_address()`. TensorAdapter cannot create tensor-backed CBs from unmaterialized tensors.

**6. Shadow graph edges require `_node_id` stamps.** For CBEngine and codegen to work, every CBHandle produced by an op must be stamped with its producing node ID. TensorAdapter must maintain this graph identity through auto-generated allocations. The `cb_alias` mechanism inherits `_node_id` from the source handle for this reason.

> **Warning:** Violating any of these invariants will not produce a compile-time error in the current Blaze infrastructure. Violations are detected at runtime via silent data corruption or hardware hangs. TensorAdapter's validation pipeline (P3) must enforce all six invariants at planning time.

---

## What Must Be Invented

The survey reveals that no existing framework solves the full TensorAdapter problem. Five capabilities must be built from scratch:

**1. CB-aware graph analysis.** No framework performs graph analysis with CB slot counts as a resource constraint. TVM and XLA analyze memory bytes; the 64-slot limit is a fundamentally different constraint (integer programming on slot indices, not linear programming on byte capacities).

**2. Automatic phase boundary insertion.** Blaze's `reconfig()` requires manual placement by the developer. TensorAdapter must automatically determine optimal phase boundaries from the op graph and CB lifetime analysis. This is a novel optimization problem: minimize the number of phase boundaries (each adds a reconfig kernel invocation) while keeping per-phase CB counts below 64.

**3. FIFO/direct-address mode inference.** No framework infers access mode from operation semantics. Blaze requires explicit specification. TensorAdapter should infer FIFO mode for streaming producer-consumer patterns and direct-address mode for random-access patterns (e.g., shared L1 views).

**4. Per-core heterogeneous CB planning.** No framework plans different CB configurations for different cores within a single kernel launch. Blaze supports this through disjoint-core sharing, but the developer must specify it manually. TensorAdapter should derive core-specific configurations from the operation's data flow and the core grid assignment.

**5. Cross-framework tensor bridging.** Converting between Symbiote's `TorchTTNNTensor` (with distributed config, logical shapes, mesh mappers) and Blaze's `CBHandle` (with page geometry, access modes, backing tensors) requires a bidirectional adapter that preserves both worlds' metadata. No existing system bridges two hardware abstraction layers at this level.

> **Warning:** The five capabilities listed above represent genuine unsolved problems, not incremental improvements on existing solutions. Attempting to build TensorAdapter by simply wrapping an existing framework (TVM backend, XLA plugin, Triton extension) will fail because none of these frameworks model the 64-CB slot constraint, FIFO access semantics, or multi-phase reconfiguration. TensorAdapter must be built on Blaze's primitives (`CBEngine`, `FusedProgram`, `CircularBufferIdManager`) with new planning logic on top.

---

## The Gap Summary

```
                    Framework Capabilities
                    ______________________
                   |                      |
  Symbiote:        |  Module replacement  |  <-- PyTorch integration (solved)
                   |  __torch_dispatch__  |
                   |  Distributed config  |
                   |______________________|
                           |
                   ________v___________
                  |                    |
  TVM/XLA/MLIR:  |  Graph lowering    |  <-- Multi-level compilation (solved)
                  |  Buffer lifetime   |
                  |  Memory planning   |
                  |____________________|
                           |
                   ________v___________
                  |                    |
  Triton:         |  Tile programming  |  <-- Developer experience (solved)
                  |  Implicit memory   |
                  |  Auto-tuning       |
                  |____________________|
                           |
                   ________v___________          __________________________
                  |                    |        |                          |
  THE GAP:        |  CB-aware planning |  <--   |  TensorAdapter's domain  |
                  |  64-slot constraint|        |  (to be built)           |
                  |  Phase allocation  |        |__________________________|
                  |  FIFO/DA inference |
                  |  Per-core planning |
                  |____________________|
                           |
                   ________v___________
                  |                    |
  Blaze:          |  FusedProgram      |  <-- CB composition (solved, manual)
                  |  CBEngine          |
                  |  CB reconfig       |
                  |  MicroOp library   |
                  |____________________|
```

TensorAdapter bridges the gap between framework-level tensor abstractions (Symbiote, torch.compile, TVM) and hardware-level CB composition (Blaze). The seven principles extracted from the framework survey define **how** to bridge this gap. The six architectural invariants define the **safety boundaries**. The five novel capabilities define **what must be invented**.

---

## Comprehensive Framework Comparison Table

| Dimension | TT-Symbiote | TVM | Triton | XLA/JAX | MLIR | torch.compile | TT-Blaze (manual) | TensorAdapter (proposed) |
|-----------|-------------|-----|--------|---------|------|---------------|-------------------|--------------------------|
| **Abstraction level** | nn.Module replacement | Multi-level IR | Tile-level kernel | HLO graph | Dialect hierarchy | FX graph | Per-core composition | PyTorch tensor -> CB binding |
| **Interception** | `__torch_dispatch__` | Frontend import | `@triton.jit` | `jax.jit` tracing | Source-level IR | `torch._dynamo` | Manual compose | dispatch + graph capture |
| **Algorithm-schedule sep.** | No | Yes | Partial | Yes (auto) | Yes | Partial | No | Yes (P1) |
| **Memory hierarchy** | DRAM (TTNN-managed) | Scoped buffers | SMEM/regs (implicit) | HBM/VMEM/regs | memref + spaces | `empty_strided` | L1 CB + DRAM | L1 CB + DRAM (auto) |
| **Buffer/CB allocation** | TTNN internal | StorageRewrite | Compiler | Automatic | `memref.alloc` | Inductor alloc | Manual `cb_*` | Automatic from shapes |
| **Tiling control** | None | Schedule | constexpr | Auto (128x128) | Tiling pass | Heuristic | Manual | ShapeEngine (P4 tiered) |
| **Lifetime/reuse** | TTNN GC | StorageRewrite | Compiler | Automatic | BufferDealloc | Buffer reuse | compact_cb_ids | Auto (inherits CBEngine) |
| **64-CB limit** | N/A (per-op) | N/A | N/A | N/A | N/A | N/A | Multi-phase reconfig | Auto phase insertion |
| **Access mode** | N/A | N/A | N/A | N/A | N/A | N/A | Manual FIFO/DA | Inferred from op |
| **L1 capacity check** | N/A | Memory scope | N/A | Memory space | N/A | Cache model | Manual | Auto per-core validation |
| **Multi-device** | DistributedTensorConfig | None | None | PartitionSpec | None | None | PipelineGraph | Mesh-aware CB allocation |
| **CB-level control** | None | None | None | None | Possible (dialect) | None | Full (explicit) | Full (automated) |
| **Developer effort (op)** | ~50 LOC | ~200 LOC | ~100 LOC | N/A | N/A | N/A | ~200 LOC | ~20 LOC (constraints) |

---

## Architectural Blueprint

The seven principles form the TensorAdapter pipeline defined in [Chapter 3](../ch03_tensor_adapter/index.md):

```
    PyTorch Tensor + Op Hint
            |
            | P1: Separate intent from strategy
            | P7: Transparent interception
            v
    [ShapeEngine]                        -- P2: Dual-world ShapeDescriptor
            |
            | P3: Validate tile decomposition
            v
    [PaddingPolicy]                      -- P6: Per-op fill values
            |
            | P3: Validate padding correctness
            v
    [FormatPolicy]                       -- P5: Graph-level negotiation
            |
            | P3: Validate format compatibility
            v
    [CBSizer + L1BudgetValidator]        -- P5: Graph-level L1 budget
            |
            | P3: Validate L1 budget + 64-CB limit
            v
    [CBHandle parameters]                -- P4: Override at any step
            |
            | P7: Transparent -- produces same emit() calls
            v
    [FusedProgram.emit()]                -- Existing Blaze (unchanged)
```

Each box is a component. Each arrow is a lowering pass with validation. The MLIR-inspired three-layer structure (Tensor -> Tile -> CB) maps to ShapeEngine -> PaddingPolicy/FormatPolicy -> CBSizer. The override hierarchy (P4) is orthogonal -- overrides inject at any layer.

---

## Principle-to-Chapter Mapping

| # | Principle | Primary Chapter | Key Component |
|---|-----------|----------------|---------------|
| 1 | Separate computation from execution strategy | Ch 3: TensorAdapter Core | Port descriptor -> CB inference |
| 2 | Carry dual-world metadata | Ch 4: Tile Decomposition | ShapeDescriptor dataclass |
| 3 | Validate at each lowering step | Ch 7: CB Sizing | Five-step validation chain |
| 4 | Defaults with per-decision overrides | Ch 3: TensorAdapter Core | Tiered API (auto/configurable/manual) |
| 5 | Graph-level analysis for cross-op decisions | Ch 9: Op Fusion | Format negotiation, L1 budgeting |
| 6 | Per-operation padding semantics | Ch 5: Padding | PADDING_REGISTRY |
| 7 | Transparent interception preserving identity | Ch 10: End-to-End API | `__torch_dispatch__` + module replacement |

---

## Key Takeaways

- **Seven principles** govern TensorAdapter's design: (P1) separate intent from strategy, (P2) carry dual-world metadata, (P3) validate at each step, (P4) defaults with per-decision overrides, (P5) graph-level analysis, (P6) per-op padding semantics, (P7) transparent interception.
- The **decision matrix** shows 11 of 16 patterns adopted and 5 adapted. No patterns are rejected outright -- even patterns that do not map directly to Tenstorrent hardware (Triton's masks, TVM's full schedule language) contribute conceptual models that inform TensorAdapter's design.
- The **coverage matrix** confirms TensorAdapter addresses all 12 manual decisions from Chapter 1, with CB ID and semaphore allocation delegated to existing engines.
- **Six architectural invariants** (CB uniqueness, page size formula, 64-CB limit, access mode exclusivity, materialized backing, graph identity) constrain the design space and must be enforced at planning time.
- **Five novel capabilities** must be invented: CB-aware graph analysis, automatic phase boundary insertion, FIFO/DA inference, per-core heterogeneous planning, and cross-framework tensor bridging.
- The **gap summary** shows TensorAdapter's precise position: above Blaze's manual CB composition, below existing framework abstractions, in the space no current tool addresses.

## Source Files

- This chapter synthesizes patterns from all sources referenced in Files 01-04:
  - File 01: `tt_symbiote/core/tensor.py`, `module.py`, `run_config.py`, `default_dispatcher.py`
  - File 02: TVM `src/te/schedule/schedule_ops.cc`, `src/tir/transforms/storage_rewrite.cc`; Blaze `cb_engine.py`
  - File 03: Triton `triton/language/core.py`, `triton/runtime/autotuner.py`; XLA `xla/service/buffer_assignment.cc`
  - File 04: MLIR `mlir/Dialect/Linalg/`, `mlir/Dialect/Bufferization/`; PyTorch `torch/_inductor/`, `torch/_dynamo/`
- Blaze core: `blaze/cb_handle.py`, `blaze/cb_engine.py`, `blaze/fused_program.py`, `blaze/cb_reconfig.py`

---

**Next:** [Chapter 3 -- TensorAdapter Architecture](../ch03_tensor_adapter/index.md)
