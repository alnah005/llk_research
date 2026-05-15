# 5.2 FusedOp Registration Walkthrough

[Section 5.1](01_microop_registration_walkthrough.md) showed that registering a MicroOp is mechanical: one X-macro entry, identical adapter instantiation, identical dispatch path. FusedOps follow the same registration steps, but the programs they produce are structurally different, and this difference surfaces in hashing, caching, and edge cases around dynamic composition. This section first characterizes what makes FusedOps fundamentally different from both MicroOps and native TTNN ops (addressing Challenge 4 from [Section 3.2](../ch03_compilation_model_gap/02_translation_challenges.md)), then walks through GatedReduce end-to-end using the adapter template from [Section 4.2](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md), and closes with the hash strategy, dynamic composition edge cases, and correctness considerations unique to fused operations.

---

## 5.2.1 FusedOp Characteristics

A FusedOp chains multiple MicroOp `emit()` calls on a shared `FusedProgram`, producing a single multi-phase `ProgramDescriptor`. Four characteristics distinguish FusedOps from MicroOps and from TTNN's per-op model:

### Multi-`emit()` Composition

A FusedOp's `compose()` method calls `emit()` on multiple constituent MicroOps:

```python
# Representative FusedOp composition pattern
class GatedReduce(FusedOp):
    name: str = "gated_reduce"
    x: Input = Input()
    gate_weights: Input = Input()
    reduce_weights: Input = Input()
    out: Output = Output()

    @staticmethod
    def emit(f, x, gate_weights, reduce_weights, output_tensor, *,
             prefix="gated_reduce", gate_activation="silu"):
        # Phase 1: Gate projection + activation
        gate_out = Matmul.emit(f, x, gate_weights, output_tensor=None,
                               prefix=f"{prefix}.gate")
        activated = SiLU.emit(f, gate_out, prefix=f"{prefix}.act") \
                    if gate_activation == "silu" \
                    else GeLU.emit(f, gate_out, prefix=f"{prefix}.act")

        # Phase 2: Reduction
        reduced = Matmul.emit(f, activated, reduce_weights, output_tensor,
                              prefix=f"{prefix}.reduce")
        return reduced

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        cls.emit(f, tensors["x"], tensors["gate_weights"],
                 tensors["reduce_weights"], output, **user_args)
```

Each `emit()` call appends CB allocations, CT args, and kernel phases to the shared `FusedProgram`. The result is a single `ProgramDescriptor` containing multiple phases with shared CB namespaces and semaphore pools.

### Single-Program, Multi-Phase Descriptor

The compiled `ProgramDescriptor` for GatedReduce contains all phases (gate matmul, activation, reduce matmul) in a single descriptor. This is fundamentally different from TTNN's per-op model, where each operation produces an independent `Program`:

| Aspect | TTNN Native Ops | Blaze FusedOp |
|--------|----------------|---------------|
| Programs per logical operation | One `Program` per op | One `ProgramDescriptor` for the entire fusion |
| CB lifetime | Per-op (allocated and freed per dispatch) | Shared across phases (CBEngine optimizes reuse) |
| Semaphore scope | Per-op | Shared across phases within the fusion |
| Cache granularity | Per-op cache entry | One cache entry for the entire fusion |
| Dispatch overhead | $N$ dispatches for $N$ ops | 1 dispatch for $N$ phases |

This is the performance advantage of Blaze's fusion model: inter-op data stays in L1 circular buffers without round-tripping through DRAM. The adapter preserves this by treating the entire multi-phase descriptor as a single `operation_attributes_t`.

The adapter wraps the *final assembled* `MeshProgramDescriptor` without inspecting the internal phase structure. Whether the descriptor contains 1 phase (MicroOp) or 5 phases (FusedOp), the dispatch path is identical: `invoke()` → `compute_program_hash()` → `BlazeAdapterMeshProgramFactory` → `GenericMeshProgramFactory` → `ProgramBuilder`. The comparison table in [Section 5.2.8](#528-microop-vs-fusedop-registration-comparison) confirms this: the "Adapter code differences" row is **None** for both categories.

---

## 5.2.2 Why FusedOps Cannot Be Decomposed into Per-Phase TTNN Ops

A natural question is whether each phase of a FusedOp could be registered as a separate TTNN operation (e.g., register `blaze::matmul` and `blaze::silu` separately, then dispatch them sequentially for GatedReduce). This would align with TTNN's per-op model. However, decomposition is impossible for three structural reasons:

**1. Shared CB IDs.** Phase 2's input CB is phase 1's output CB. The `CBEngine` assigns the same CB ID to both because their lifetimes overlap precisely -- the gate output is consumed immediately by the activation, and the activation output is consumed immediately by the reduce matmul. TTNN programs own their CB namespaces independently -- two separate `Program` objects cannot share a CB ID, so the zero-copy L1 handoff between phases would be lost.

**2. Single kernel binary.** The phases are compiled into one kernel with phase-switching logic by `kernel_codegen.py`. Splitting the kernel would require producing multiple independent kernels, losing the L1-resident data reuse that fusion enables. The unified kernel also enables instruction-level optimizations (pipeline interleaving) across phase boundaries that separate kernels cannot achieve.

**3. Shared semaphores.** Inter-phase synchronization (e.g., waiting for phase 1's data movement to complete before phase 2's compute begins) uses intra-program semaphores. Separate programs would require inter-program synchronization, which has different semantics and higher overhead.

The adapter's approach of wrapping the entire FusedOp's `MeshProgramDescriptor` as one TTNN operation is the only correct mapping. [Section 3.2.4](../ch03_compilation_model_gap/02_translation_challenges.md) established this principle; the GatedReduce walkthrough below demonstrates it concretely.

---

## 5.2.3 `FusedOpConfig` Integration

Blaze's `FusedOpConfig` dataclass controls compilation behavior for FusedOps:

```python
# Source: blaze/blaze_op.py (representative)
@dataclass
class FusedOpConfig:
    compose_fn: Callable           # The compose function to call
    compute_config: ComputeConfig  # Math fidelity, approx mode, etc.
    kernel: str | None = None      # Optional handwritten kernel path
    grid: GridConfig | None = None # Optional fixed grid
```

When `BlazeCompiler._compile_fused_op()` runs, it reads `FusedOpConfig` to determine:

- **`compose_fn`**: which composition function to invoke (usually `cls.compose` or `cls.emit`)
- **`compute_config`**: math fidelity and approximation settings passed to child `emit()` calls
- **`kernel`**: if set, the kernel codegen phase is skipped and the handwritten kernel is used directly
- **`grid`**: if set, overrides the default grid selection

None of these config fields affect the adapter. They are consumed during Phase 2 (Compilation) of the Blaze pipeline, producing a `MeshProgramDescriptor` that already embeds the configuration's effects. By the time the descriptor reaches the adapter, the `FusedOpConfig` has been fully consumed.

**One implication for hashing:** `FusedOpConfig` parameters (especially `compute_config`) can change the descriptor structure without changing the tensor shapes. Two calls with the same input tensors but different math fidelities produce different descriptors and should hash differently. The Tier 1 descriptor walk handles this automatically (different CB configs and CT args produce different hashes). For Tier 2 custom hashing, the hash must include `compute_config` parameters explicitly.

### FusedOps with Handwritten Kernels

Some FusedOps provide a handwritten kernel via `FusedOpConfig.kernel` instead of using the auto-generated kernel from the shadow graph. Examples include performance-critical attention ops where the auto-generated kernel does not achieve optimal instruction scheduling.

For the adapter, a handwritten kernel changes nothing. The kernel path is embedded in the `ProgramDescriptor`'s `KernelDescriptor` entries, and `ProgramBuilder` processes it identically whether the source was auto-generated or hand-written (this is how Challenge 5 -- dynamic kernel sources -- is resolved, as discussed in [Section 4.2.8](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md)). The adapter never inspects kernel source paths -- it delegates the entire program construction to `GenericMeshProgramFactory`.

The only implication is for Tier 2 hashing: if a FusedOp switches between auto-generated and handwritten kernels based on a configuration parameter, that parameter must be included in the custom hash. Otherwise, the hash may match a cached program that uses the wrong kernel variant.

---

## 5.2.4 Hash Strategy for FusedOps

FusedOp hashing is more nuanced than MicroOp hashing because FusedOps have two additional hash-relevant dimensions: graph topology and conditional composition.

### The Hash Challenge

For a MicroOp like Matmul, the descriptor is fully determined by input tensor shapes and user args ($M$, $K$, $N$, `transpose_b`). Two calls with the same shapes and args produce structurally identical descriptors.

For a FusedOp, the descriptor depends on:

1. **Input tensor shapes** -- as with MicroOps
2. **User args** -- configuration parameters passed through `user_args` dict
3. **Graph topology** -- which MicroOps are composed and in what order
4. **Conditional branches** -- `if`/`else` logic in `compose()` that selects different composition paths

Dimensions 3 and 4 can produce structurally different descriptors for the same tensor shapes. The Tier 1 descriptor walk handles this correctly (different descriptors produce different hashes), but at the cost of walking the entire multi-phase descriptor on every dispatch.

### Recommended: Tier 2 Topology-Aware Hash

For FusedOps, the recommended approach is a Tier 2 `custom_program_hash` that hashes the inputs to the composition, not the output:

```python
# PROPOSED: Hash computation in BlazeCompiler for FusedOps
def _compute_fusedop_hash(self, op_cls, tensors, user_args):
    """
    Compute a semantic hash for a FusedOp that captures:
    1. Tensor shapes (determines CB sizes and per-core config)
    2. User args (determines graph topology for conditional ops)
    3. Op class identity (distinguishes GatedReduce from SharedExpert)
    """
    import hashlib
    h = hashlib.sha256()

    # Op identity
    h.update(op_cls.name.encode())

    # Tensor shapes (sorted by port name for determinism)
    for port_name in sorted(tensors.keys()):
        t = tensors[port_name]
        h.update(port_name.encode())
        h.update(str(t.shape).encode())
        h.update(str(t.dtype).encode())
        h.update(str(t.layout).encode())

    # User args (sorted, serialized)
    for key in sorted(user_args.keys()):
        h.update(key.encode())
        h.update(str(user_args[key]).encode())

    # Compute config (affects descriptor structure -- math fidelity, approx mode)
    if hasattr(op_cls, '_fused_op_config') and op_cls._fused_op_config:
        cfg = op_cls._fused_op_config.compute_config
        if cfg is not None:
            h.update(str(cfg.math_fidelity).encode())
            h.update(str(cfg.approx_mode).encode())

    return int.from_bytes(h.digest()[:8], byteorder='little')
```

This hash is $O(1)$ in descriptor complexity (it does not walk the compiled descriptor) and correctly captures all dimensions that affect the descriptor structure. The hash is passed to the adapter via `BlazeAdapterAttributes.custom_program_hash`:

```python
# PROPOSED: In _run_program()
self._dispatch_fn(
    io_tensors,
    descriptor,
    custom_program_hash=self._fusedop_hash  # Tier 2
)
```

### Correctness Condition for Tier 2

The Tier 2 hash is correct if and only if:

$$
\forall (u_1, s_1), (u_2, s_2): \quad h(u_1, s_1) = h(u_2, s_2) \implies \text{PD}(u_1, s_1) = \text{PD}(u_2, s_2)
$$

where $u$ = user_args, $s$ = tensor shapes, and PD = the resulting `ProgramDescriptor`. This holds when:

1. The compose function is deterministic (same inputs produce same outputs).
2. All compilation-affecting parameters are included in the hash inputs.
3. No hidden state (global variables, random number generators) influences compilation.

### Hash Cost Comparison for FusedOps

FusedOps produce larger descriptors than MicroOps (more phases, more CBs, more CT args), making the Tier 1 descriptor walk proportionally more expensive. The following estimates are design projections based on descriptor complexity analysis:

| Op | Phases | Approx CB Count | Tier 1 Hash (est.) | Tier 2 Hash (est.) |
|----|--------|-----------------|-------------------|-------------------|
| Matmul (MicroOp) | 1 | 3--5 | ~2 $\mu$s | ~0.1 $\mu$s |
| GatedReduce | 3 | 8--12 | ~5--8 $\mu$s | ~0.1 $\mu$s |
| SharedExpert | 4--5 | 12--18 | ~8--12 $\mu$s | ~0.1 $\mu$s |
| FusedAttention | 5--8 | 15--25 | ~10--18 $\mu$s | ~0.1 $\mu$s |

For FusedOps dispatched at high frequency (e.g., SharedExpert called once per MoE layer, ~8 times per token), the Tier 2 hash saves an estimated ~80--140 $\mu$s per token. This is modest relative to kernel execution time (~1--10 ms per token), but becomes measurable at low batch sizes where dispatch overhead is a larger fraction of total time.

> **Note:** All hash time estimates in this section are design projections, not benchmarks. Actual values will depend on CPU speed, compiler optimizations, and descriptor structure. They are provided to illustrate the relative cost of Tier 1 vs Tier 2.

---

## 5.2.5 Edge Case: Dynamic Graph Structure

Some FusedOps contain conditional composition logic that produces structurally different descriptors based on runtime conditions:

```python
class ConditionalFusedOp(FusedOp):
    name = "conditional_fused"

    @staticmethod
    def emit(f, x, weights, output, *, use_bias=False, **kwargs):
        intermediate = Matmul.emit(f, x, weights, output_tensor=None, ...)

        if use_bias:
            # Adds an extra phase, extra CBs, extra CT args
            biased = ResidualAdd.emit(f, intermediate, kwargs["bias"], ...)
            return Reduce.emit(f, biased, output, ...)
        else:
            return Reduce.emit(f, intermediate, output, ...)
```

The `use_bias` parameter changes the number of phases in the descriptor. This creates two distinct program structures for the same op name.

### When Caching Works

The Tier 1 descriptor walk handles this correctly: different descriptors produce different hashes, so each branch gets its own cache entry. The Tier 2 topology hash also handles this correctly if `use_bias` is included in `user_args` and thus in the hash.

### When Caching Breaks Down

Caching breaks down when the condition depends on tensor *values* rather than tensor *shapes* or *user_args*:

```python
# PATHOLOGICAL: Conditional based on tensor content
def emit(f, x, weights, output, **kwargs):
    # This cannot be hashed without reading the tensor, which
    # would require a device-to-host transfer
    if x.shape[0] > 1 and kwargs.get("dynamic_routing"):
        # Route A: full graph
        ...
    else:
        # Route B: simplified graph
        ...
```

If the condition changes between dispatches with the same shapes and user_args, the Tier 2 hash produces the same value for different descriptors, leading to a cache hit that returns the wrong program. The Tier 1 descriptor walk still catches this (different descriptors hash differently), but at the cost of never achieving a cache hit when the condition alternates.

### Solution: `_graph_branch_keys` for Topology-Aware Hashing

For FusedOps with a small number of conditional branches (the common case), a `_graph_branch_keys` class attribute declares which `user_args` keys affect graph topology. The Blaze compiler uses these keys to partition the hash space:

```python
# PROPOSED: FusedOp author declares topology-affecting parameters
class ConditionalFusedOp(FusedOp):
    name = "conditional_fused"
    _graph_branch_keys = ["use_bias"]  # <-- declares topology-affecting params

    @staticmethod
    def emit(f, x, weights, output, *, use_bias=False, **kwargs):
        intermediate = Matmul.emit(f, x, weights, output_tensor=None, ...)
        if use_bias:
            biased = ResidualAdd.emit(f, intermediate, kwargs["bias"], ...)
            return Reduce.emit(f, biased, output, ...)
        else:
            return Reduce.emit(f, intermediate, output, ...)
```

The compiler's hash function includes these keys:

```python
# PROPOSED: In BlazeCompiler
def _compute_graph_id(self, op, user_args):
    """
    Compute a topology-sensitive graph ID.
    For static-graph FusedOps: return a constant.
    For dynamic-graph FusedOps: include branch-determining
    parameters in the ID.
    """
    base_id = hash(op.compose)
    branch_keys = getattr(op, '_graph_branch_keys', [])
    branch_values = tuple(user_args.get(k) for k in branch_keys)
    return hash((base_id, branch_values))
```

This approach ensures:
- **Static-graph FusedOps** (no `_graph_branch_keys`) produce a constant graph ID, and the `type_hash` prefix captures the op identity implicitly.
- **Dynamic-graph FusedOps** produce distinct graph IDs for distinct topologies, preventing cross-topology cache collisions.
- The Tier 2 correctness condition is preserved: the hash includes all inputs that determine the `ProgramDescriptor`.

> **Note:** `_graph_branch_keys` is a proposed design mechanism. In the current Blaze codebase, dynamic-graph FusedOps would need to be annotated with this attribute upon adoption.

### Prevalence of Dynamic-Graph FusedOps

In practice, dynamic-graph FusedOps are rare:

| Category | Count | Examples |
|----------|------:|---------|
| Static-graph FusedOps | ~35 | GatedReduce, FusedAttention, FusedRMSNormLinear |
| Dynamic-graph FusedOps | ~5 | ConditionalMoE, AdaptiveCompute, SwitchedExpert |
| MicroOps (no compose) | ~72 | Matmul, Copy, Reduce, Mcast, Transpose |
| **Total** | **~112** | |

### Summary of Hash Strategy by Op Category

| Category | Hash Strategy | Rationale |
|----------|--------------|-----------|
| MicroOp (standard) | Tier 1 (descriptor walk) | Simple descriptors; hash cost is low |
| MicroOp (high-frequency) | Tier 2 (semantic hash from shapes + user_args) | Saves an estimated ~2 $\mu$s per dispatch; zero C++ changes |
| FusedOp (static topology) | Tier 2 (shapes + user_args + op name) | Larger descriptors make Tier 1 expensive; topology is deterministic |
| FusedOp (dynamic topology) | Tier 2 with `_graph_branch_keys` | Include topology discriminant in hash for correctness |
| FusedOp (data-dependent topology) | Tier 1 (descriptor walk) | Correctness over performance; dynamic conditions require full descriptor comparison |
| FusedOp (Tier 3 candidate) | Tier 3 (OpTag `custom_hash` hook) | Only for ops where C++ semantic knowledge enables $O(1)$ hash with domain-specific invariants |

For FusedOps with a small number of conditional branches (2--4 variants), include the discriminant in the Tier 2 hash via `_graph_branch_keys`. For FusedOps with many variants or data-dependent topology, fall back to the Tier 1 descriptor walk. Register variants as separate ops only when they are semantically distinct enough to warrant separate profiling identities.

---

## 5.2.6 End-to-End GatedReduce Walkthrough

### Steps 1--3: Op Definition, X-Macro Entry, Build Artifacts

GatedReduce's definition (shown in Section 5.2.1 above) is already functional via `generic_op`. Registration follows the same three steps as Matmul ([Section 5.1.3](01_microop_registration_walkthrough.md)): (1) the op already exists, (2) add `X(gated_reduce, GatedReduce)` to `BLAZE_OP_LIST`, (3) the build system generates `GatedReduceTag`, `register_operation<"ttnn::blaze::gated_reduce", ...>()`, and the nanobind binding -- all structurally identical to the Matmul artifacts with only names changed. The adapter does not know or care that GatedReduce is a FusedOp.

### Step 4: Compilation Produces a Multi-Phase Descriptor

When `BlazeCompiler.compile()` runs GatedReduce, the `compose()` function chains three `emit()` calls on the shared `FusedProgram`. The resulting `ProgramDescriptor` contains:

```text
ProgramDescriptor for GatedReduce:
  kernels:
    - gate_matmul_kernel (BRISC + NCRISC + TRISC)
    - silu_kernel (TRISC only)
    - reduce_matmul_kernel (BRISC + NCRISC + TRISC)
  circular_buffers:
    - CB[0]: gate input activations (x)
    - CB[1]: gate weights
    - CB[2]: gate output / silu input    <-- shared across phases
    - CB[3]: silu output / reduce input  <-- shared across phases
    - CB[4]: reduce weights
    - CB[5]: reduce output (final)
  semaphores:
    - gate.sender, gate.receiver
    - reduce.sender, reduce.receiver
  ct_args:
    - gate.matmul.in0_w, gate.matmul.in1_w, gate.matmul.transpose_b
    - act.silu.in0, act.silu.out
    - reduce.matmul.in0_w, reduce.matmul.in1_w, reduce.matmul.transpose_b
    - per_core: gate.matmul.is_active, reduce.matmul.is_active
```

The key observation: CBs 2 and 3 are shared across phases. `CBEngine` assigned the same CB IDs to the gate output and silu input because their lifetimes overlap precisely -- the gate output is consumed immediately by silu, and the silu output is consumed immediately by the reduce matmul. This temporal reuse is what makes fusion efficient, and it is fully preserved in the `ProgramDescriptor`.

### Step 5: Dispatch Through the Adapter

```python
# At runtime:
compiled_gated_reduce.run([x, gate_weights, reduce_weights, output])
  -> _run_program(io_tensors, descriptor)
     -> ttnn.blaze.gated_reduce(io_tensors, descriptor)
```

The C++ dispatch path is identical to the MicroOp case:

```text
invoke(io_tensors, descriptor)
  -> BlazeAdapterAttributes{
       .mesh_program_descriptor = descriptor,  // multi-phase
       .op_name = "blaze::gated_reduce"}

launch<BlazeDeviceOperationAdapter<GatedReduceTag>>(...)
  -> compute_program_hash:
       type_seed = type_hash<Adapter<GatedReduceTag>>  // unique
       hash = hash(type_seed, descriptor_walk)
         // Descriptor walk covers all 3 phases, all 6 CBs,
         // all 4 semaphores, all CT args
  -> MISS: BlazeAdapterMeshProgramFactory::create_mesh_workload()
       -> GenericMeshProgramFactory -> ProgramBuilder
          -> builds ONE Program with 3 kernels, 6 CBs, 4 semaphores
  -> enqueue_mesh_workload()
       -> TracyOpMeshWorkload("ttnn::blaze::gated_reduce")
```

### Step 6: Profiler Output

```text
Tracy Timeline:
  |---------- ttnn::blaze::gated_reduce ----------|
  |                   (85 us)                      |

  (Internally: gate_matmul 35us + silu 8us + reduce_matmul 42us,
   but presented as ONE zone because it is ONE program dispatch)

Graph Trace:
  blaze::gated_reduce   (single node, not three nodes)

Profiler JSON:
  {
    "op_type": "BlazeDeviceOperationAdapter<GatedReduceTag>",
    "attributes": {"op_name": "blaze::gated_reduce", "num_mesh_programs": 1},
    "duration_ns": 85120
  }
```

The FusedOp appears as a single unit in all observability systems. The internal phase structure (gate, activation, reduce) is not visible at the TTNN dispatch level -- it is encapsulated within the single `Program` object. This is the correct behavior: the fusion is the unit of dispatch, and its internal structure is an implementation detail.

---

## 5.2.7 The `override_runtime_arguments` Subtlety for FusedOps

On a cache hit, the program cache returns a `CachedMeshWorkload` from a previous dispatch. The factory's `override_runtime_arguments()` patches `Buffer*` addresses and runtime args in place, without rebuilding the program. For MicroOps with a single kernel and a handful of CBs, this is straightforward. For FusedOps, the patch must cover all phases:

```text
Cache hit for GatedReduce:
  override_runtime_arguments(cached_workload, attrs, args, out)
    -> GenericMeshProgramFactory::override_runtime_arguments(...)
      -> for each runtime_arg in cached_workload:
           patch Buffer* addresses for new tensor allocations
      -> patches ALL 3 kernel runtime args
      -> patches ALL 6 CB buffer addresses
      -> cross-phase shared CBs (CB[2], CB[3]) are patched ONCE
         (same buffer backing both phases)
```

The `GenericMeshProgramFactory` handles this correctly because it iterates all runtime arguments in the cached workload, regardless of phase boundaries. The adapter inherits this behavior through `BlazeAdapterMeshProgramFactory`'s delegation ([Section 4.2.5](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md)).

**Why this matters:** If the factory only patched a subset of kernel runtime args (e.g., only the first kernel's), subsequent phases would reference stale `Buffer*` pointers, causing memory corruption. The existing `GenericMeshProgramFactory` implementation -- which iterates all kernels and all CBs uniformly -- naturally avoids this. The adapter inherits this complete-patch behavior by delegating to `GenericMeshProgramFactory` without modification.

This is an example of the adapter's "post-compilation, pre-built descriptor" design paying dividends: because the adapter does not decompose the `ProgramDescriptor` into per-phase components, the factory's patching logic operates on the complete, unified program, and cross-phase shared CBs are handled correctly without any phase-aware logic.

---

## 5.2.8 Comparison: MicroOp vs FusedOp Registration

| Dimension | MicroOp (Matmul) | FusedOp (GatedReduce) |
|-----------|-------------------|----------------------|
| X-macro entry | `X(matmul, Matmul)` | `X(gated_reduce, GatedReduce)` |
| Tag struct | `MatmulTag` | `GatedReduceTag` |
| Adapter instantiation | `BlazeDeviceOperationAdapter<MatmulTag>` | `BlazeDeviceOperationAdapter<GatedReduceTag>` |
| Registration string | `"ttnn::blaze::matmul"` | `"ttnn::blaze::gated_reduce"` |
| Python callable | `ttnn.blaze.matmul(...)` | `ttnn.blaze.gated_reduce(...)` |
| Phases in descriptor | 1 | 3 |
| CBs in descriptor | 3--5 | 8--12 |
| Tier 1 hash cost (est.) | ~2 $\mu$s | ~5--8 $\mu$s |
| Recommended hash tier | Tier 1 or 2 | Tier 2 (topology-aware) |
| Dynamic composition risk | None | Possible (conditional branches in `compose()`) |
| `override_runtime_arguments` | Patches 1 kernel, 3--5 CBs | Patches 3 kernels, 6+ CBs (including shared) |
| Adapter code differences | **None** | **None** |

The last row is the critical observation: the adapter code is identical for MicroOps and FusedOps. All differences are resolved in the Python compilation pipeline before the descriptor reaches C++.

---

| [Section 5.1: MicroOp Registration](./01_microop_registration_walkthrough.md) | [Section 5.3: Batch Registration](./03_batch_registration_and_scaling.md) |
|:---|---:|
