# Abstraction Goals and Scope

The preceding sections established a clear impedance mismatch: PyTorch developers think in tensors, shapes, and module composition, while TT-Blaze requires tiles, CBs, page sizes, core placement, and dozens of CT args. This section defines what an abstraction layer should do to bridge that gap, what it should leave alone, and how it fits into the existing TT-Blaze infrastructure.

**What you will learn:**

- The vision for a "TensorAdapter" layer that bridges PyTorch tensors to TT-Blaze CBs
- A comparison of what the developer specifies today versus what the system would infer, with escape hatches for every automated decision
- Scope boundaries: what the abstraction handles and what it does not
- How each capability maps to the research questions this guide answers (Q1-Q12)
- The design constraint: compose with existing infrastructure, do not replace it

---

## Vision: From Explicit Plumbing to Declarative Intent

The central idea is a **TensorAdapter** layer -- a shape-aware argument builder -- that sits between the PyTorch developer's tensor-level thinking and TT-Blaze's CB-level execution. The developer expresses intent -- "multiply these two tensors" -- and the adapter handles tile decomposition, CB allocation, padding, format selection, and L1 budgeting.

### Current State: Manual Pipeline

Today, converting a PyTorch `nn.Linear` to run on Tenstorrent hardware requires approximately this workflow:

```python
# EXISTING -- manual workflow (simplified)

# 1. Prepare weights: tilize, shard across cores, choose format
weight_tensor = ttnn.from_torch(
    linear.weight.T,                      # transpose for row-major matmul
    dtype=ttnn.bfloat8_b,                 # manual format choice
    layout=ttnn.TILE_LAYOUT,              # must be tiled
    device=device,
    memory_config=ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        shard_spec=ttnn.ShardSpec(
            core_range_set,               # manual core selection
            [K, N // num_cores],           # manual shard shape
            ...
        ),
    ),
    tile=ttnn.Tile([32, 32]),             # manual tile shape
)

# 2. Prepare activations: tilize, ensure tile alignment
act_tensor = ttnn.from_torch(
    activation,
    dtype=ttnn.bfloat16,                  # manual format choice
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=...,
    tile=ttnn.Tile([1, 32]),              # manual: row tiles for 1D input
)

# 3. Build FusedProgram and emit
f = FusedProgram(kernel=kernel_path, device=device, math_fidelity=ttnn.MathFidelity.LoFi)
result = Matmul.emit(f, act_tensor, weight_tensor, prefix="linear")
# ... wire output, build program, run ...
```

### Proposed State: Automated Pipeline

With the TensorAdapter, the equivalent workflow would be:

```python
# PROPOSED -- automated workflow

# Developer writes standard PyTorch
linear = torch.nn.Linear(4096, 11008)

# One-line conversion
blaze_linear = BlazeModule.from_torch(linear, sample_input=torch.randn(1, 4096), device=device)

# Inference -- no manual CB, tile, or format decisions
output = blaze_linear(input_tensor)
```

Under the hood, `BlazeModule.from_torch()` would:

1. Trace the module to identify the matmul operation
2. Call `TensorAdapter.adapt(weight, op_hint="matmul.in1")` to compute tile shape, padding, format, and CB sizing
3. Call `TensorAdapter.adapt(activation, op_hint="matmul.in0")` for the activation
4. Derive `k_num_tiles`, `out_w_per_core`, and all other CT args from the shape metadata
5. Build the FusedProgram with computed parameters
6. Validate L1 budget and compile

The developer provides the tensors and the operation; the system derives everything else.

### The Full Module-Level Vision

At the module level, the abstraction enables:

```python
# PROPOSED -- full module API
class GatedMLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(torch.silu(self.gate_proj(x)) * self.up_proj(x))

# Convert to Blaze execution
model = GatedMLP(4096, 11008)
blaze_model = blaze.from_pytorch(model, sample_input=torch.randn(1, 4096),
                                  device=mesh_device)
output = blaze_model(input_tensor)
```

Compare with the current approach, which requires the developer to manually build an 11-node graph (as shown in `blaze/_gated_mlp.py`), wire CT args for each op, allocate CBs, manage padding, and verify L1 budgets.

---

## What Changes: Developer vs. System Responsibility

The following table shows which decisions shift from developer to system with the proposed abstraction. The "Override Available?" column is a core design requirement -- every automatic decision has an escape hatch:

| Decision | Today (Manual) | With TensorAdapter | Override Available? |
|----------|---------------|-------------------|-------------------|
| **Tile shape** | Developer chooses `(32, 32)` or `(1, 32)` or `(16, 32)` | System selects based on tensor shape and op type | Yes -- `adapt(tensor, tile_shape=(16, 32))` |
| **Padded shape** | Developer computes `round_up(dim, tile_dim)` | System computes via ShapeDescriptor | Yes -- `adapt(tensor, padded_shape=...)` |
| **Padding fill value** | Developer knows zero for matmul, -inf for softmax | System looks up per-op padding registry | Yes -- `adapt(tensor, padding=PaddingPolicy.NEG_INF)` |
| **Data format** | Developer chooses `bfloat16`, `bfloat8_b`, etc. per CB | System negotiates across op chain based on format preferences | Yes -- `adapt(tensor, format=ttnn.bfloat4_b)` |
| **num_pages** | Developer computes from tile counts and blocking | System derives from shape metadata and op requirements | Yes -- via `cb_sizing={"in0": {"num_pages": 4}}` |
| **page_size** | Developer computes `tile_h * tile_w * dtype_bytes` | System computes from tile shape and format | Rarely needed |
| **Core placement** | Developer selects from device grid | System derives from weight shard spec or picks optimal grid | Yes -- explicit `grid=...` parameter |
| **CB ID** | Allocated sequentially by FusedProgram | Same allocation, same engine -- TensorAdapter just provides parameters | Via CBEngine configuration |
| **CT arg values** | Developer computes `k_num_tiles`, `out_w_per_core`, etc. | System derives from ShapeDescriptor | Yes -- `user_args` dict to `BlazeCompiler.compile()` |
| **Broadcasting pattern** | Developer selects `bcast_rows`, `bcast_cols`, or Mcast | System detects from shape comparison | Yes -- explicit broadcast mode |
| **Blocking strategy** | Developer chooses `RT_DIM`, `CT_DIM`, `KT_DIM` | System optimizes for L1 budget and throughput | Yes -- override blocking factors |
| **L1 budget check** | Developer hopes it fits (no compile-time check) | System validates total L1 usage before compilation | Warnings, not hard failures |

The pattern is consistent: the system makes the default decision; the developer can override any specific decision when the default is suboptimal.

---

## Scope Boundaries

### What the Abstraction Handles

The TensorAdapter abstraction is responsible for the translation from PyTorch's tensor model to TT-Blaze's tile/CB model:

| Capability | Description | Guide Chapter |
|-----------|-------------|---------------|
| **Shape decomposition** | Convert arbitrary PyTorch shapes into tile-aligned, padded shapes with tile grid metadata | [Chapter 4: Tile Decomposition](../ch04_tile_decomposition/index.md) |
| **Shape propagation** | Track logical, padded, and tiled shapes through op chains via ShapeDescriptor | [Chapter 4](../ch04_tile_decomposition/index.md) |
| **Padding management** | Select fill values, insert padding, propagate padding through op chains, strip on output | [Chapter 5: Padding](../ch05_padding/index.md) |
| **Format selection** | Choose data formats based on op preferences and precision profiles | [Chapter 6: Data Formats](../ch06_data_formats/index.md) |
| **Format negotiation** | Insert conversion ops when producers and consumers disagree on format | [Chapter 6](../ch06_data_formats/index.md) |
| **CB sizing** | Compute num_pages, page_size from shape metadata, format, and L1 budget | [Chapter 7: CB Sizing](../ch07_cb_sizing/index.md) |
| **L1 budgeting** | Validate that all CBs fit within per-core L1 before dispatch | [Chapter 7](../ch07_cb_sizing/index.md) |
| **Broadcasting detection** | Compare input shapes, detect broadcast dimensions, select bcast pattern or Mcast | [Chapter 8: Broadcasting](../ch08_broadcasting/index.md) |
| **CT arg derivation** | Compute shape-dependent CT args (k_num_tiles, out_w_per_core, etc.) from ShapeDescriptor | [Chapter 3: TensorAdapter Architecture](../ch03_tensor_adapter/index.md) |
| **Op fusion support** | Maintain correct shape/padding/format tracking across fused op boundaries | [Chapter 9: Op Fusion](../ch09_op_fusion/index.md) |

### What the Abstraction Does NOT Handle

Equally important is what the abstraction explicitly excludes from its scope:

| Exclusion | Rationale | Who Handles It |
|-----------|-----------|---------------|
| **Kernel authoring** | C++ kernel code (the `.hpp` files) is hardware-specific. The abstraction composes kernels via `emit()`, it does not generate them. | Kernel developers write C++ directly |
| **RISC-level programming** | NCRISC/TRISC/BRISC code structure is determined by the op's data movement pattern. | Kernel developers |
| **Custom NOC patterns** | Non-standard data movement (e.g., ring all-reduce, paged KV cache) requires topology-aware programming. | Op developers |
| **Semaphore protocols** | Semaphore assignment depends on inter-core topology, not tensor shapes. The SemEngine already handles this. | SemEngine (unchanged) |
| **CCL / multi-device** | Cross-device communication (CclBroadcast, fabric barriers) involves mesh topology, not tensor shapes. | CCL ops and MeshFusedProgram |
| **Dynamic shapes** | The abstraction assumes shapes are known at graph build time. Dynamic sequence lengths require separate handling. | Application-level logic or JIT recompilation |
| **Training / backward pass** | TT-Blaze targets inference. Autograd integration is out of scope. | Future work |

---

## Mapping Capabilities to Research Questions

This guide is structured around 12 research questions. Each question maps to a specific capability of the TensorAdapter and is addressed in a specific chapter:

| Question | Topic | Abstraction Component | Chapter |
|----------|-------|----------------------|---------|
| Q1 | How to decompose PyTorch shapes into tiles | ShapeDescriptor, ShapeEngine | [Chapter 4](../ch04_tile_decomposition/index.md) |
| Q2 | Design patterns from TVM/Triton/XLA/MLIR/torch.compile | Synthesized principles | [Chapter 2](../ch02_design_patterns/index.md) |
| Q3 | Automatic padding management | PaddingPolicy registry, padding pass | [Chapter 5](../ch05_padding/index.md) |
| Q4 | Data format selection and negotiation | FormatPolicy, negotiation algorithm | [Chapter 6](../ch06_data_formats/index.md) |
| Q5 | CB sizing and L1 budgeting | AutoBlocker, L1 budget validator | [Chapter 7](../ch07_cb_sizing/index.md) |
| Q6 | Broadcasting rule mapping | BroadcastResolver | [Chapter 8](../ch08_broadcasting/index.md) |
| Q7 | Op fusion with shape/format tracking | FusionAnalyzer, ShapeTracker | [Chapter 9](../ch09_op_fusion/index.md) |
| Q8 | Integration with CBEngine/CTArgEngine | Wrap/extend/replace analysis | [Chapter 3](../ch03_tensor_adapter/index.md) |
| Q9 | Integration with FusedProgram | CBBackend layer | [Chapter 3](../ch03_tensor_adapter/index.md) |
| Q10 | Integration with BlazeGraph/BlazeCompiler | Graph edge metadata extension | [Chapter 3](../ch03_tensor_adapter/index.md) |
| Q11 | Design patterns from TT-Symbiote | `__torch_dispatch__` and module replacement | [Chapter 2](../ch02_design_patterns/index.md) |
| Q12 | End-to-end developer API and escape hatches | BlazeModule, override hierarchy | [Chapter 10](../ch10_end_to_end/index.md) |

---

## Design Constraint: Compose, Do Not Replace

The most important architectural constraint is that the TensorAdapter must **compose with** the existing TT-Blaze infrastructure, not replace it. The existing components are battle-tested on production models (DeepSeek V3, Llama, Qwen) and represent hundreds of person-months of engineering. The abstraction layer is a **builder** that constructs the same FusedProgram calls a human developer would write.

### Components and Their Relationship to the Abstraction

| Component | File | Relationship | What Changes |
|-----------|------|-------------|-------------|
| **CBHandle** | `blaze/cb_handle.py` | WRAP | The abstraction creates CBHandles with computed parameters. The CBHandle dataclass is unchanged. |
| **FusedProgram** | `blaze/fused_program.py` | WRAP | The abstraction calls `cb_from_tensor()`, `cb_scratch()`, CT arg methods. FusedProgram gains no new methods. |
| **CBEngine** | `blaze/cb_engine.py` | WRAP | The abstraction feeds pre-configured TensorPorts into `CBEngine.assign()`. The engine's algorithm is unchanged. |
| **SemEngine** | `blaze/sem_engine.py` | UNCHANGED | Semaphore assignment depends on inter-core topology, not tensor shapes. |
| **CTArgEngine** | `blaze/ct_args.py` | WRAP | The abstraction computes user_args (tile counts, blocking factors) that CTArgEngine resolves into CT arg values. |
| **BlazeGraph** | `blaze/graph.py` | EXTEND | The abstraction attaches ShapeDescriptor metadata to graph edges. |
| **BlazeCompiler** | `blaze/compiler.py` | WRAP | The abstraction calls `BlazeCompiler.compile()` with pre-computed tensors and user_args. |
| **BlazeOp hierarchy** | `blaze/blaze_op.py` | UNCHANGED | MicroOp and FusedOp class hierarchy, `emit()` signatures, and `register()` all remain. |

### The Adapter as emit() Caller

The conceptual model is:

```
Developer code         TensorAdapter              Existing Blaze
--------------         -------------              ---------------
                       infer_shape()
y = blaze_matmul(     select_tile()
    activation,        compute_padding()
    weights            select_format()
)                      allocate_cb()
                       derive_ct_args()
                            |
                            v
                       Matmul.emit(f, in0_handle, in1_handle,
                                   prefix="matmul")
                            |
                            v
                       FusedProgram.cb_from_tensor()
                       FusedProgram.cb_scratch()
                       FusedProgram.ncrisc_ct_args()
                       FusedProgram.trisc_ct_args()
                       FusedProgram.output()
```

The TensorAdapter fills in the parameters that the developer currently specifies manually. The downstream FusedProgram calls are identical whether generated by the adapter or written by hand.

### Rationale: Why Compose Instead of Replace

#### Why Not Replace FusedProgram?

FusedProgram is the composition context that manages CB allocation, CT arg wiring, semaphore dedup, multi-phase reconfig, and shadow graph recording. It has been refined over many iterations to handle edge cases in production models. Replacing it would mean reimplementing:

- Disjoint-cores CB sharing (format-key dedup)
- Scratch-mapped CB allocation (pre-allocated arenas)
- Multi-phase CB reconfig (CircularBufferIdManager integration)
- Overlapped view management (shared fused tensors with byte-offset addressing)

All of these are complex, hardware-specific features that the abstraction does not need to reimplement. By wrapping FusedProgram, the abstraction inherits these capabilities for free.

#### Why Not Build a New Graph IR?

BlazeGraph already provides a validated DAG of OpNodes connected by Edges, with topological ordering, fan-out handling, and the `fuse()` context manager for graph construction. The CBEngine, SemEngine, and CTArgEngine already consume BlazeGraph and produce assignments. Building a parallel graph IR would fragment the toolchain and create maintenance burden. Instead, the abstraction extends BlazeGraph edges with ShapeDescriptor metadata and feeds the enriched graph to the same engines.

#### Why Not Just Improve Each Op?

One could argue that each op's `emit()` should be smarter -- inferring more parameters from its inputs. This has been tried (e.g., `PaddedRMSNorm.emit()` derives padding parameters from `width`). The problem is that cross-op concerns (shape propagation, format negotiation, L1 budgeting) cannot be solved within a single op. The abstraction provides a system-level view that individual ops cannot have.

---

## The Developer Journey

The abstraction enables a progression from complete novice to expert tuner:

```
Level 1: PyTorch developer (new to Tenstorrent)
  -> Uses BlazeModule.from_pytorch(model, sample_input)
  -> System handles everything automatically
  -> Performance: ~85-95% of hand-tuned

Level 2: Developer wants to tune performance
  -> Uses precision profiles: "performance", "balanced", "accuracy"
  -> Overrides specific format/CB decisions via context managers
  -> Performance: ~95-100% of hand-tuned for standard patterns

Level 3: Expert developer
  -> Drops down to raw FusedProgram API for critical ops
  -> Uses adapter for everything else
  -> Full control where needed, automation everywhere else
```

This progressive disclosure model ensures that the abstraction serves both audiences: PyTorch developers who want zero-friction hardware access, and experienced Blaze developers who want to automate the tedious parts while retaining control over the critical paths.

### The Escape Hatch Hierarchy

For every automatic decision the TensorAdapter makes, there is an explicit override:

| Level | Mechanism | Example |
|-------|-----------|---------|
| **Per-op** | Annotation or hint on the op call | `@blaze.override(cb_sizing={"in0": {"num_pages": 4}})` |
| **Per-graph** | Custom `user_args` dict to `BlazeCompiler.compile()` | Override shape-derived blocking factors |
| **Bypass** | Drop to raw `FusedProgram` API | Performance-critical ops use `emit()` directly |

The expected workflow is: start with the TensorAdapter for rapid prototyping, profile with `l1_profile.py`, identify bottlenecks, and surgically override specific decisions. The abstraction accelerates the common case (80% of ops work well with automatic configuration) while preserving full control for the edge cases (20% that need hand-tuning).

---

## What Success Looks Like

A successful TensorAdapter would:

1. **Reduce the decision surface** for standard ops from ~12 manual decisions to ~2 (choose the op, provide the tensors)
2. **Produce identical hardware programs** to hand-tuned code for common shapes (multiples of 32, standard LLM dimensions)
3. **Produce correct programs** for non-standard shapes (padding, non-tile-aligned dimensions) that would be error-prone to configure manually
4. **Add no runtime overhead** -- all computation happens at compile time (graph construction, engine passes); the generated kernel binary is identical to hand-written
5. **Preserve full manual control** via escape hatches at three levels: per-op overrides, per-graph overrides, and full API bypass

The subsequent chapters of this guide define how each capability is designed, how it integrates with existing code, and where the inevitable trade-offs between automation and performance emerge.

---

## Key Takeaways

- The TensorAdapter is a shape-aware argument builder that sits between PyTorch's tensor API and TT-Blaze's CB-level `emit()` calls.
- It automates tile decomposition, padding, format selection, CB sizing, broadcasting detection, and L1 budgeting -- while leaving kernel authoring, RISC programming, and semaphore management untouched.
- The abstraction wraps (does not replace) CBHandle, FusedProgram, CBEngine, CTArgEngine, and BlazeCompiler -- all existing APIs remain unchanged. The adapter is an `emit()` caller, not an `emit()` replacer.
- Every automated decision has an explicit override mechanism: per-op annotations, per-graph user_args, or full bypass to raw FusedProgram.
- The developer journey progresses from zero-config (Level 1) through profile-guided tuning (Level 2) to expert manual control (Level 3).
- Success means reducing ~81 manual decisions for a gated MLP to ~3 lines of developer code, with identical hardware performance for standard configurations.

## Source Files

- `blaze/cb_handle.py` -- CBHandle (wrapped by TensorAdapter)
- `blaze/fused_program.py` -- FusedProgram (extended, not replaced)
- `blaze/cb_engine.py` -- CBEngine (algorithm unchanged, inputs enriched)
- `blaze/context.py` -- FusionContext, ExternalTensor (graph entry point)
- `blaze/graph.py` -- BlazeGraph, OpNode, Edge (extended with ShapeDescriptor)
- `blaze/blaze_op.py` -- BlazeOp, MicroOp, FusedOp (class hierarchy unchanged)
- `blaze/_gated_mlp.py` -- build_gated_mlp_graph() (example of what the adapter automates)
- `blaze/l1_profile.py` -- L1 profiling utilities (used for validation)

---

**Next:** [Chapter 2 -- Design Patterns from Existing Tensor-to-Hardware Abstraction Layers](../ch02_design_patterns/index.md)
