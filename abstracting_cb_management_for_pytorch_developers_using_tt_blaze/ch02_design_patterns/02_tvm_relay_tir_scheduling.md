# TVM: Multi-Level IR and Scheduled Buffer Allocation as a Design Pattern

**What you will learn:**

- How TVM's two-level IR (Relay and TIR) separates high-level graph semantics from low-level buffer/loop decisions
- The scheduling primitive model: developer-guided transformations on a declarative computation definition
- TVM's buffer allocation and memory scope system as a precursor to CB management
- The StorageRewrite algorithm and its mathematical equivalence to Blaze's `compact_cb_ids` interval coloring
- Tiling as a first-class scheduling concept and its mapping to Tenstorrent tile decomposition
- Four design patterns (DP5-DP8) that TensorAdapter should adopt from TVM

---

## TVM's Architecture: The Two-Level IR

TVM addresses the same class of problem that TensorAdapter faces: bridging high-level tensor operations to hardware-specific memory management. TVM's solution is a **two-level intermediate representation** that cleanly separates concerns:

| Level | IR | Purpose | Analogy in Blaze |
|-------|------|---------|-----------------|
| High | Relay | Graph-level: op fusion, data flow, type inference | BlazeGraph (op DAG) |
| Low | TIR | Loop-level: tiling, buffer allocation, memory scopes | FusedProgram (CB wiring) |

Relay captures the computational graph -- `conv2d(x, w) + bias` -- without specifying how loops iterate, where data lives, or how tiles are sized. TIR takes a Relay subgraph and lowers it into explicit loops, buffer allocations, and memory-scope annotations that target specific hardware.

The separation is the key insight. A developer (or auto-tuner) can modify the TIR schedule without changing the Relay graph, and vice versa. This is the same separation TensorAdapter needs: the PyTorch developer's intent (matmul, RMSNorm, gated activation) should be separable from the hardware-specific decisions (tile shape, CB sizing, padding, format).

---

## Design Pattern 5 (DP5): Scheduling as Explicit Transformation

### EXISTING (TVM)

TVM's schedule primitives are explicit, composable transformations applied to a computation definition. A computation is declared once; the schedule decides how to execute it:

```python
# EXISTING -- TVM compute + schedule pattern
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
k = te.reduce_axis((0, K), name="k")
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

# Schedule (how to compute it)
s = te.create_schedule(C.op)
xo, xi = s[C].split(C.op.axis[0], factor=32)    # tile the M dimension
yo, yi = s[C].split(C.op.axis[1], factor=32)    # tile the N dimension
ko, ki = s[C].split(k, factor=32)                # tile the K dimension
s[C].reorder(xo, yo, ko, xi, ki, yi)             # reorder for locality
```

After scheduling, the loop nest looks like this (pseudocode):

```
for xo in range(M // 32):        # Outer M tiles
  for yo in range(N // 32):      # Outer N tiles
    for ko in range(K // 32):    # K tiles (streaming through reduction)
      for xi in range(32):       # Inner M elements
        for ki in range(32):     # Inner K elements
          for yi in range(32):   # Inner N elements
            C[xo*32+xi, yo*32+yi] += A[xo*32+xi, ko*32+ki] * B[ko*32+ki, yo*32+yi]
```

Each `split()` and `reorder()` call is a named, parameterized transformation. The computation definition (`te.compute`) remains unchanged throughout.

### Mapping to TT-Blaze Decisions

| TVM Schedule Primitive | Blaze Manual Decision | TensorAdapter Automation Opportunity |
|----------------------|----------------------|-------------------------------------|
| `split(axis, factor=32)` | Choose tile shape (32x32 vs 1x32) | `ShapeEngine.infer_tile_shape(tensor, op)` |
| `reorder(...)` | Choose iteration order / blocking strategy | `AutoBlocker.compute_blocking(shape, l1_budget)` |
| `cache_read(scope="shared")` | Allocate CB with specific memory scope | `CBSizer.allocate(descriptor, scope="l1")` |
| `compute_at(C, xo)` | Decide which ops share CBs vs. get fresh allocations | `FusionAnalyzer.detect_reuse_opportunities(graph)` |
| `bind(xo, "blockIdx")` | Core placement / `core_ranges` | `GridPlanner.assign_cores(op, device_grid)` |

### The Separation Principle

The critical lesson: **the computation definition and the execution strategy are separate concerns**. In TVM, you can change the tile factor from 32 to 16 without rewriting the computation. In current Blaze, changing the tile factor means rewriting `emit()` code, adjusting `num_pages`, recomputing `page_size`, updating CT args, and re-verifying L1 budgets.

TensorAdapter should implement this separation. The developer declares the computation (via PyTorch or via Blaze's graph API). The adapter applies "schedule-like" transformations -- tile decomposition, padding insertion, format selection -- as a separate pass, configurable without touching the op definition.

---

## Design Pattern 6 (DP6): Buffer Allocation with Memory Scopes

### EXISTING (TVM)

TVM's TIR lowers buffer allocations with explicit memory scopes:

```python
# EXISTING -- TIR buffer allocation with memory scope
A_shared = T.alloc_buffer([32, 32], dtype="float16", scope="shared")   # GPU shared memory
A_local  = T.alloc_buffer([1, 32], dtype="float16", scope="local")     # register file
```

The `scope` annotation tells the backend where to place the buffer. On GPU, `"shared"` maps to shared memory; `"local"` maps to registers. The size is derived from the tiling schedule -- a `split(factor=32)` produces 32x32 tiles, which directly determines buffer dimensions.

### Mapping to Tenstorrent CBs

| TVM Scope | Tenstorrent Equivalent | Blaze Construct |
|-----------|----------------------|-----------------|
| `"global"` | DRAM | `ttnn.Tensor` in DRAM |
| `"shared"` | L1 SRAM (per-core) | CB backed by L1 allocation |
| `"local"` | DST register file (per-Tensix) | Implicit -- managed by LLK |
| `"wmma.matrix_a"` | Unpacker SrcA register | Tile registers in math pipeline |

A Tenstorrent CB is essentially a TIR "shared" buffer with additional constraints:

- **Fixed page size**: Each page is one tile (e.g., 32x32 x sizeof(bfloat16) = 2048 bytes)
- **FIFO semantics**: Producers push pages, consumers pop pages, with hardware pointer tracking
- **64-slot limit**: Only 64 CBs per core, requiring careful ID allocation and lifetime management
- **Core-range binding**: Each CB is configured per core range, not globally

> **Warning:** TVM's predicated inner-loop access (conditional guards on out-of-bounds indices) does not translate to Tenstorrent tile hardware. Tensix cores process fixed-size tiles through the math pipeline without per-element predicate masks. TensorAdapter must use explicit padding, with the padding value determined by a per-op registry. See [Chapter 5: Padding](../ch05_padding/index.md).

TVM derives buffer size from the schedule. TensorAdapter should derive CB `num_pages` from the tile decomposition in exactly the same way: the tile shape determines `page_size`, and the blocking strategy determines `num_pages`.

---

## The StorageRewrite Algorithm: Interval Coloring for Buffer Reuse

TVM's `StorageRewrite` pass analyzes buffer lifetimes and merges non-overlapping buffers into shared memory allocations. This is mathematically identical to Blaze's CB compaction:

| Dimension | TVM StorageRewrite | Blaze CBEngine |
|-----------|-------------------|-------------------------------|
| Resource being allocated | Bytes in a memory region | CB slot indices (0-63) |
| Hard limit | Total memory size | 64 slots per Tensix core |
| Reuse constraint | Non-overlapping lifetimes | Non-overlapping lifetimes + matching data format (for multi-phase) |
| Granularity | Arbitrary byte ranges | Fixed-size pages (tile_h x tile_w x dtype_bytes) |
| Cross-phase support | N/A (single pass) | `CircularBufferIdManager` context system |

Both systems solve the same bin-packing problem -- minimum graph coloring of interval graphs:

```python
# EXISTING -- Blaze CB interval coloring (from cb_engine.py)
def compact_cb_ids(self, assignments):
    sorted_cbs = sorted(assignments.values(), key=lambda a: a.first_use)
    color_end: dict[int, int] = {}
    for cb in sorted_cbs:
        color = 0
        while color in color_end and color_end[color] >= cb.first_use:
            color += 1
        cb.cb_id = color
        color_end[color] = cb.last_use
    return assignments
```

TensorAdapter should expose this lifetime information automatically. When a developer chains operations in PyTorch, the system should infer that intermediate CB slots can be reused after their consumers complete, without requiring manual lifetime annotations.

---

## Design Pattern 7 (DP7): Automatic Tiling with Configurable Factors

### EXISTING (TVM)

TVM's auto-scheduler (Ansor) and AutoTVM search over tiling configurations to find optimal tile factors. The search space is constrained by hardware parameters, computation structure, and cost models. The developer can override the auto-tuner's choice with explicit tile factors:

```python
# TVM -- explicit tile factor override
s[C].split(axis, factor=user_specified_tile_size)
```

This is the same **default-with-override** pattern that TensorAdapter needs:

```python
# PROPOSED -- TensorAdapter tile shape with override (illustrative)
descriptor = TensorAdapter.adapt(tensor, op_hint="matmul.in0")
# Default: ShapeEngine selects (32, 32) based on tensor shape and op type
# Override:
descriptor = TensorAdapter.adapt(tensor, op_hint="matmul.in0", tile_shape=(16, 32))
```

TT-Blaze operates on a narrower hardware target than TVM (Tensix cores with fixed 32x32 tile geometry for most data formats), which means the search space is smaller. But the principle applies: CB sizing should use heuristics that automatically select `num_pages`, `page_size`, and double/triple buffering depth based on:

1. **Tensor shape**: A [1, 32768] activation needs different CB depth than a [128, 128] weight
2. **L1 budget**: Total available L1 per core constrains total CB size
3. **Operation type**: Matmul benefits from double-buffering (num_pages=2); elementwise ops may only need single-buffering
4. **Data format**: bfloat16 has page_size=2048 for 32x32 tiles; bfloat8_b has page_size=1088

### L1 Budget Calculation Example

```
# PROPOSED -- CB sizing cost function (illustrative)
total_l1_usage = sum(cb.num_pages * cb.page_size for cb in all_cbs_on_core)
constraint: total_l1_usage <= L1_BUDGET  # ~1.5 MB per core on Blackhole

# For a matmul with bfloat16 activations and bfloat8_b weights:
#   act_cb:    2 pages x 2048 bytes =  4,096 bytes
#   weight_cb: 1 page  x 1088 bytes =  1,088 bytes
#   output_cb: 2 pages x 2048 bytes =  4,096 bytes
#   scratch:   4 pages x 2048 bytes =  8,192 bytes
#   Total: 17,472 bytes per core (well within 1.5 MB)
```

---

## Design Pattern 8 (DP8): Lowering Pipeline with Validation Passes

### EXISTING (TVM)

TVM's compilation pipeline runs a sequence of transformation and validation passes:

```
Relay graph
    |  -- type inference pass
    |  -- operator fusion pass
    |  -- constant folding pass
    v
Fused Relay subgraphs
    |  -- schedule generation (auto-tune or manual)
    |  -- tiling + buffer allocation
    |  -- loop lowering
    v
TIR functions
    |  -- storage flatten pass
    |  -- memory scope validation
    |  -- bounds checking pass
    v
Target code (CUDA, LLVM, etc.)
```

Each pass validates its preconditions and postconditions.

### Lesson for TensorAdapter

TensorAdapter should implement a similar validation pipeline:

```
# PROPOSED -- TensorAdapter validation passes (illustrative)
PyTorch tensor + op hint
    |  -- tile decomposition pass (ShapeEngine)
    |  -- padding insertion pass (PaddingPolicy)
    |  -- format selection pass (FormatPolicy)
    v
ShapeDescriptors
    |  -- format compatibility check (producer format == consumer expectation)
    |  -- L1 budget validation (sum of all CBs per core <= budget)
    |  -- CB ID pressure check (total unique CBs <= 64)
    v
Validated descriptors --> CBHandle creation
```

The key principle: **validation happens at each lowering step, not just at the end**. TVM catches shape mismatches at the Relay level, buffer overflow at the TIR level, and target violations at codegen. TensorAdapter should catch tile incompatibilities before CB allocation, format mismatches before building the FusedProgram, and L1 overflows before kernel compilation.

This is in direct contrast to the current Blaze experience, where L1 overflow is detected only at runtime via silent corruption or hardware hangs (as documented in [Chapter 1: The Manual Plumbing Burden](../ch01_the_gap/03_the_manual_plumbing_burden.md)).

---

## TVM's Limitations for the Tenstorrent Model

TVM's IR and scheduling model were designed primarily for GPU and CPU targets. Several aspects do not map cleanly to Tenstorrent's execution model:

| TVM Assumption | Tenstorrent Reality |
|---------------|---------------------|
| Threads within a warp/block share memory implicitly | Tensix cores communicate via NOC; sharing is explicit |
| Buffer sizes are flexible (any power of 2) | CB pages must match tile geometry exactly |
| Single memory hierarchy (global/shared/local) | Two-level: DRAM + L1 with CB hardware FIFO layer |
| Auto-tuning searches over thousands of configurations | Tenstorrent tile shapes are constrained to a small set (32x32, 16x32, 1x32) |
| Compiler generates loop nests | Kernel developer writes explicit loop nests in C++ |
| Single static program per compilation | Blaze supports multi-phase CB reconfiguration mid-kernel |
| All compute units are identical | Different Tensix cores can have different CB configurations via disjoint-core sharing |

These differences mean TensorAdapter should **adopt TVM's separation principle and validation pipeline** but not attempt to replicate its auto-tuning search. The Tenstorrent decision space is small enough for deterministic rules (see [Chapter 4: Tile Decomposition](../ch04_tile_decomposition/index.md)) rather than learned cost models.

---

## Key Takeaways

- **DP5: Separation of computation and schedule** is TVM's foundational insight. TensorAdapter should separate "what operation" from "how to tile, pad, and allocate CBs" -- the same matmul can have different tile shapes and CB configurations without changing the op's kernel code.
- **DP6: Buffer allocation parameterized by tiling schedule** means CB sizing should be derived from tile decomposition, not specified independently. When tile shape changes, num_pages and page_size update automatically.
- **StorageRewrite and compact_cb_ids** solve mathematically equivalent problems (interval graph coloring) with different constraints (bytes vs. CB slots). TensorAdapter should make lifetime-based reuse automatic.
- **DP7: Default-with-override** for tile factors and buffer sizes is the right UX pattern. Auto-select for the common case, explicit override for expert tuning.
- **DP8: Validation at each lowering step** prevents errors from cascading. Catch tile incompatibilities before CB allocation, format mismatches before program build, L1 overflows before dispatch.

## Source Files

- TVM documentation: `tvm.apache.org/docs/tutorial/tensor_expr_get_started.html`
- TVM source: `src/te/schedule/schedule_ops.cc` -- schedule transform implementation
- TVM source: `src/tir/transforms/storage_rewrite.cc` -- buffer lifetime analysis and merging
- TVM auto-scheduler: `python/tvm/auto_scheduler/search_task.py` -- Ansor interface
- Blaze parallel: `blaze/fused_program.py` -- CB allocation methods (`cb_from_tensor`, `cb_scratch`)
- Blaze parallel: `blaze/cb_engine.py` -- `compact_cb_ids()` (interval coloring analog)
- Blaze parallel: `blaze/cb_reconfig.py` -- `CircularBufferIdManager` (multi-phase CB management)

---

**Next:** [03 -- Triton and XLA Patterns](./03_triton_and_xla_patterns.md)
