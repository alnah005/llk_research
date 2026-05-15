# 5.1 MicroOp Registration Walkthrough

This section traces the complete registration of a single Blaze MicroOp -- Matmul -- from its Python class definition through every C++ and Python layer to the point where `ttnn.blaze.matmul(...)` is callable, cache-isolated, and profiler-visible. The walkthrough is end-to-end: no step is omitted, and every code artifact produced along the way is shown. The goal is to demonstrate that the adapter pattern from [Chapter 4](../ch04_adapter_pattern/index.md) instantiates cleanly for real ops with zero ambiguity about what goes where.

---

## 5.1.1 MicroOp Characteristics and `cpp_parser.py`

A MicroOp is the atomic unit of computation in Blaze, defined by four properties: (1) a **single C++ kernel** (`kernels/op.hpp`) with compute logic, lifecycle methods, and compile-time argument structs; (2) **declared ports** (`Input`/`Output`/`Internal`) defining the tensor interface; (3) **auto-derived CT args** via `cpp_parser.py`, which parses the kernel header at import time (`_auto_derive_from_kernel_hpp()` → `parse_op_hpp()` → `ParsedKernel` dataclass containing `struct_name`, `ct_args`, RISC presence flags, lifecycle flags, and pop control); (4) a **single `emit()` call** producing one `ProgramDescriptor` phase. Contrast with FusedOps, which chain multiple `emit()` calls ([Section 5.2](02_fusedop_registration_walkthrough.md)).

For Matmul, auto-derivation produces: `op_class = "Matmul"`, `is_inter_core = True`, 8 CT args across three RISC groups, `has_init_teardown = True`, `pop_flags = ["pop_in0"]`. These are representative values; exact fields may vary with the kernel version.

The adapter does not need any of this metadata. It operates on the `MeshProgramDescriptor` that is the *output* of the compilation pipeline, which has already consumed the CT arg metadata, RISC mappings, and lifecycle flags. The auto-derivation pipeline runs before compilation; the adapter runs after. This separation is what makes registration zero-cost per op.

---

## 5.1.2 The Matmul MicroOp

Matmul is the most representative MicroOp for a walkthrough: it has multiple inputs, inter-core communication, per-core specialization, and is the single most performance-critical op in LLM inference.

### Python Definition (Existing -- No Changes)

```python
# Source: blaze/ops/matmul/op.py (simplified for walkthrough)
class Matmul(MicroOp):
    name: str = "matmul"
    math_fidelity: str = "LoFi"

    in0: Input = Input()     # activations
    in1: Input = Input()     # weights
    out: Output = Output()   # result

    @staticmethod
    def emit(f, in0, in1, output_tensor, *, transpose_b=False, prefix="matmul",
             math_fidelity=None, ...):
        # Step 1: Resolve inputs to CBHandles
        a_cb = f.cb_from_tensor(in0) if not isinstance(in0, CBHandle) else in0
        b_cb = f.cb_from_tensor(in1) if not isinstance(in1, CBHandle) else in1

        # Step 2: Compute dimensions
        k_num_tiles = a_cb.inner_dim
        out_w_per_core = b_cb.num_pages // k_num_tiles

        # Step 3: Allocate output CB
        out_cb = f.cb_scratch(
            name=BlazeOp.cb_name(prefix, "out"),
            num_pages=out_w_per_core, ...)

        # Step 4: Register CT args
        f.unified_ct_args([
            ("matmul.in0_w", a_cb.inner_dim),
            ("matmul.in1_w", b_cb.inner_dim),
            ("matmul.transpose_b", int(transpose_b)),
        ])
        f.per_core_unified_ct_args([
            f.flag("matmul.is_active", out_cb.core_ranges),
        ])

        # Step 5: Return output handle
        return f.output("matmul", out_cb, src=a_cb)

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        cls.emit(f, tensors["in0"], tensors["in1"], output, **user_args)
```

At this point, the op is fully functional via `ttnn.generic_op()`. The question is: what does it take to make it a first-class TTNN operation?

---

## 5.1.3 Step-by-Step Registration

### Step 1: Add to the X-Macro List (One Line)

The only per-op human action is adding Matmul to `BLAZE_OP_LIST`:

```cpp
// Source: (new) ttnn/cpp/ttnn/operations/blaze/blaze_op_list.hpp
#define BLAZE_OP_LIST(X)    \
    X(matmul, Matmul)       \   /* <-- this line */
    X(copy, Copy)           \
    X(reduce, Reduce)       \
    // ... see Section 5.3 for the full catalog
```

The first argument (`matmul`) is the op name used in Python and registration strings. The second argument (`Matmul`) is the tag struct prefix, producing `MatmulTag`.

### Step 2: Tag Struct Generation (Automated)

The `DEFINE_BLAZE_TAG` macro (from [Section 4.2.6](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md)) expands the X-macro entry into a tag struct:

```cpp
// Expansion of BLAZE_OP_LIST(DEFINE_BLAZE_TAG) for matmul:
namespace ttnn::operations::blaze {
struct MatmulTag {
    static constexpr const char* name = "blaze::matmul";
};
}  // namespace ttnn::operations::blaze
```

`MatmulTag` is a stateless type with no data members and no runtime cost. Its sole purpose is to differentiate `BlazeDeviceOperationAdapter<MatmulTag>` from every other adapter instantiation at the type level.

### Step 3: Adapter Instantiation and Registration (Automated)

The `REGISTER_BLAZE_OP_FROM_LIST` macro generates the `register_operation<>` call:

```cpp
// Expansion of BLAZE_OP_LIST(REGISTER_BLAZE_OP_FROM_LIST) for matmul:
namespace ttnn::blaze {
constexpr auto matmul = ::ttnn::register_operation<
    "ttnn::blaze::matmul",
    ::ttnn::operations::blaze::BlazeDeviceOperationAdapter<
        ::ttnn::operations::blaze::MatmulTag>>();
}  // namespace ttnn::blaze
```

This `constexpr auto` declaration triggers compile-time registration via `registered_operation_t` ([Section 1.2](../ch01_ttnn_registration_contract/02_registration_dispatch_and_bindings.md)):

- **Name registration:** `"ttnn::blaze::matmul"` is stored as the operation's compile-time name via `operation_name_key_t` friend injection. Name uniqueness is enforced at compile time -- a duplicate name causes a linker error.
- **Type registration:** `BlazeDeviceOperationAdapter<MatmulTag>` is the `DeviceOperation` type. Its `type_hash` is computed from the mangled type name, producing a value distinct from every other adapter instantiation and from `GenericOpDeviceOperation`.
- **Callable creation:** `registered_operation_t::operator()` routes to `BlazeDeviceOperationAdapter<MatmulTag>::invoke()`, which constructs `BlazeAdapterAttributes` and `BlazeAdapterTensorArgs` from the Python-supplied `io_tensors` and `MeshProgramDescriptor`.

### Step 4: Nanobind Binding (Automated)

The `BIND_BLAZE_OP_FROM_LIST` macro generates the Python binding:

```cpp
// Expansion of BLAZE_OP_LIST(BIND_BLAZE_OP_FROM_LIST) for matmul:
::ttnn::operations::blaze::bind_blaze_op(
    blaze_mod, ttnn::blaze::matmul, "Blaze Matmul");
```

This calls `bind_registered_operation()` ([Section 1.2](../ch01_ttnn_registration_contract/02_registration_dispatch_and_bindings.md)) with the overload signature `(io_tensors: list[Tensor], mesh_program_descriptor: MeshProgramDescriptor, custom_program_hash: int | None = None) -> Tensor`. The result is a Python callable at `ttnn.blaze.matmul` with the following properties:

```python
>>> ttnn.blaze.matmul.name
'matmul'
>>> ttnn.blaze.matmul.python_fully_qualified_name
'ttnn.blaze.matmul'
>>> ttnn.blaze.matmul.__ttnn_operation__
True
```

### Step 5: Python Dispatch Routing (Automated)

When `BlazeCompiler.compile()` creates a `MeshCompiledProgram`, it passes the op name:

```python
# Source: blaze/compiler.py (modified -- from Section 4.3.1)
compiled = MeshCompiledProgram(
    op_name="matmul",   # <-- from Matmul.name
    mesh_descriptor=mesh_descriptor,
    io_tensors=io_tensors, ...)
```

`MeshCompiledProgram.__init__` resolves the dispatch function once:

```python
self._dispatch_fn = self._resolve_dispatch("matmul")
# _resolve_dispatch returns ttnn.blaze.matmul (the registered callable)
```

Every subsequent `_run_program()` call invokes this pre-resolved callable directly -- no attribute lookup, no string matching, no fallback check on the hot path.

---

## 5.1.4 End-to-End Dispatch Trace

With registration complete, here is what happens when a model calls `compiled_matmul.run([a, b, output])`:

```text
Python:
  MeshCompiledProgram.run([a, b, output])
    -> _run_program(io_tensors=[a, b, output], descriptor=mesh_desc)
       -> self._dispatch_fn(io_tensors, descriptor)
          == ttnn.blaze.matmul(io_tensors, descriptor)

C++ (nanobind boundary):
  registered_operation_t<"ttnn::blaze::matmul",
      BlazeDeviceOperationAdapter<MatmulTag>>::operator()(io_tensors, descriptor)
    -> invoke(io_tensors, descriptor)
       -> returns:
          attrs = BlazeAdapterAttributes{
              .mesh_program_descriptor = descriptor,
              .op_name = "blaze::matmul",
              .custom_program_hash = nullopt}
          tensor_args = BlazeAdapterTensorArgs{
              .io_tensors = [a, b, output],
              .output_tensor = output}

  device_operation::launch<BlazeDeviceOperationAdapter<MatmulTag>>(attrs, args)
    -> GraphTracker::track_function_start("ttnn::blaze::matmul", ...)
    -> create_output_tensors(attrs, args)  =>  output  (pre-allocated passthrough)
    -> compute_program_hash(attrs, args):
         type_seed = type_hash<BlazeDeviceOperationAdapter<MatmulTag>>
         // No custom_hash hook on MatmulTag (Tier 1), no attrs.custom_program_hash
         // -> Tier 1 fallback: descriptor walk
         hash = hash(type_seed, descriptor_field_hash)
    -> program_cache.contains(hash)?

    CACHE MISS (first call):
      -> validate_on_program_cache_miss(attrs, args)
           -> verify no duplicate MeshCoordinateRanges
           -> verify !io_tensors.empty()
           -> (no OpTag::validate hook on minimal MatmulTag)
      -> BlazeAdapterMeshProgramFactory::create_mesh_workload(attrs, ...)
           -> extract attrs.mesh_program_descriptor
           -> delegate to GenericMeshProgramFactory::create_mesh_workload()
              -> for each (range, program_descriptor):
                    ProgramBuilder::build(program_descriptor) -> Program
           -> return CachedMeshWorkload
      -> program_cache.insert(hash, cached_workload)

    CACHE HIT (subsequent calls):
      -> BlazeAdapterMeshProgramFactory::override_runtime_arguments(...)
           -> delegate to GenericMeshProgramFactory::override_runtime_arguments()
              -> update Buffer* addresses, runtime args

    -> enqueue_mesh_workload(...)
         -> TracyOpMeshWorkload("ttnn::blaze::matmul")   // <-- NAMED zone
    -> GraphTracker::track_function_end(output)
    -> return output
```

---

## 5.1.5 Verifying Program Cache Isolation

Cache isolation is the most critical correctness property of the adapter. Two guarantees must hold:

**Guarantee 1: Adapter entries do not collide with generic_op entries.** The adapter's hash is seeded with `type_hash<BlazeDeviceOperationAdapter<MatmulTag>>`, while `GenericOpDeviceOperation`'s hash is seeded with `type_hash<GenericOpDeviceOperation>`. Since these are distinct C++ types, their `type_hash` values differ:

$$
h_{\text{adapter}}(\text{desc}) = \text{hash}\bigl(\texttt{type\_hash}\langle\texttt{Adapter}\langle\texttt{MatmulTag}\rangle\rangle,\; \text{desc\_hash}\bigr)
$$

$$
h_{\text{generic}}(\text{desc}) = \text{hash}\bigl(\texttt{type\_hash}\langle\texttt{GenericOpDeviceOperation}\rangle,\; \text{desc\_hash}\bigr)
$$

$$
h_{\text{adapter}}(\text{desc}) \neq h_{\text{generic}}(\text{desc}) \quad \forall\; \text{desc}
$$

This means the same descriptor dispatched through both paths (e.g., during migration) produces two separate cache entries. Both contain valid, functionally identical programs. The only cost is a duplicate entry -- not incorrect behavior.

**Guarantee 2: Adapter entries for different ops do not collide with each other.** The hash seed is per-`OpTag`:

$$
\texttt{type\_hash}\langle\texttt{Adapter}\langle\texttt{MatmulTag}\rangle\rangle \neq \texttt{type\_hash}\langle\texttt{Adapter}\langle\texttt{RMSNormTag}\rangle\rangle
$$

Even if a Matmul and an RMSNorm happen to produce structurally identical descriptors (hypothetically), their cache entries are in separate namespaces.

### Verification Test

```python
def test_matmul_cache_isolation():
    """Verify that ttnn.blaze.matmul creates cache entries
    separate from ttnn.generic_op and from other adapter ops."""
    desc = build_matmul_descriptor(M=1024, K=512, N=2048)
    io_tensors = [activation, weights, output]
    device = activation.device()

    cache = device.get_program_cache()
    initial = cache.num_entries()

    # Path A: generic_op
    ttnn.generic_op(io_tensors, desc)
    after_generic = cache.num_entries()
    assert after_generic == initial + 1

    # Path B: adapter (same descriptor)
    ttnn.blaze.matmul(io_tensors, desc)
    after_adapter = cache.num_entries()
    assert after_adapter == after_generic + 1  # NEW entry, not reused

    # Path C: same op, same descriptor -> cache HIT
    ttnn.blaze.matmul(io_tensors, desc)
    after_hit = cache.num_entries()
    assert after_hit == after_adapter  # No new entry
```

---

## 5.1.6 Profiler Output: Before vs After

Before registration, all Blaze ops appear as `"ttnn::generic_op"` in Tracy timelines, `"GenericOpDeviceOperation"` in graph traces, and share a common `op_type` in profiler JSON (see [Section 2.3](../ch02_generic_op_dispatch/03_divergence_analysis.md) for the full divergence analysis). There is no way to identify which Blaze op produced which profiler entry without correlating timestamps against Python-side logs.

### After Registration (Adapter Path)

With Matmul registered as `ttnn.blaze.matmul`, the same workload produces:

```text
Tracy Timeline:
  |--- ttnn::blaze::matmul ---|-- ttnn::blaze::rmsnorm --|---- ttnn::blaze::sdpa ----|
  |       (matmul, 45 us)     |     (rmsnorm, 12 us)     |      (sdpa, 180 us)       |

Graph Trace Nodes:
  blaze::matmul -> blaze::rmsnorm -> blaze::sdpa

Profiler JSON (op_profiler.hpp output):
  {
    "op_type": "BlazeDeviceOperationAdapter<MatmulTag>",
    "attributes": {"op_name": "blaze::matmul", "num_mesh_programs": 1},
    "duration_ns": 45230
  }
```

Each op now has a distinct name in the Tracy timeline, a distinct node in the graph trace, and a distinct `op_type` in profiler JSON. The `"attributes"` field reflects `BlazeAdapterAttributes::attribute_values()`, which exposes `op_name` and `num_mesh_programs` for Inspector serialization.

### Quantified Impact

| Metric | Before (generic_op) | After (Adapter) |
|--------|---------------------|-----------------|
| Unique op names in Tracy | 1 (`"ttnn::generic_op"`) | $N$ (one per registered op) |
| Unique nodes in graph trace | 1 (`"GenericOpDeviceOperation"`) | $N$ |
| Cache namespace isolation | None (all ops share one hash seed) | Per-op (each `OpTag` gets its own seed) |
| Profiler filtering by op type | Not possible | Standard Tracy/profiler filtering |
| Performance regression identification | Requires timestamp correlation | Direct: find `"blaze::matmul"` in timeline |

For a typical LLM inference pass with ~20 distinct Blaze ops dispatched per token, this transforms the profiler from an opaque wall of identical `generic_op` entries into a color-coded, filterable, per-op breakdown.

---

## 5.1.7 What Did NOT Change

The walkthrough above introduced exactly one line of per-op C++ (the X-macro entry). The following artifacts remained untouched:

| Artifact | Status |
|----------|--------|
| `blaze/ops/matmul/op.py` | Unchanged -- no modifications to `emit()`, `compose()`, ports, or CT args |
| `blaze/ops/matmul/kernels/op.hpp` | Unchanged -- no modifications to the kernel header |
| `blaze/cpp_parser.py` | Unchanged -- auto-derivation pipeline is unaffected |
| `blaze/compiler.py` (except `_resolve_dispatch`) | Unchanged -- the compilation pipeline produces the same `MeshProgramDescriptor` |
| `CBEngine`, `SemEngine`, `CTArgEngine` | Unchanged -- engine pipeline is unaffected |
| `GenericMeshProgramFactory` | Unchanged -- the adapter delegates to it through the factory wrapper |
| `GenericOpDeviceOperation` | Unchanged -- still handles unregistered ops via `generic_op` fallback |
| `MeshProgramDescriptor` schema | Unchanged -- same fields, same serialization |
| Hardware execution | Unchanged -- same `Program` objects, same `enqueue_mesh_workload` |

The adapter's power lies in this zero-modification property. It wraps the existing compilation output in a typed envelope without altering the output itself.

---

## 5.1.8 Extending to Other MicroOps

Registering additional MicroOps follows the identical five-step process. Here are representative examples across the op catalog:

```cpp
// blaze_op_list.hpp -- additional MicroOp entries
#define BLAZE_OP_LIST(X)    \
    X(matmul, Matmul)       \
    X(copy, Copy)           \
    X(reduce, Reduce)       \
    // ... ~60 MicroOps total; see Section 5.3 for the full catalog
```

Each entry generates the same artifacts: a tag struct, a `register_operation<>` call, and a nanobind binding. The per-op cost is approximately 5 lines of generated C++ (2 for the tag, 3 for the registration). For all ~60 MicroOps, the total generated code is approximately 300 lines.

### MicroOps with Custom Needs (Tier 2/3)

Most MicroOps are Tier 1 and need no per-op customization. However, a few high-value ops may benefit from Tier 2 (custom hash) or Tier 3 (custom validation) as described in [Section 4.1.4](../ch04_adapter_pattern/01_design_goals_and_tradeoffs.md):

| Op | Tier | Rationale |
|----|------|-----------|
| Matmul | 2 (custom hash) | Dominant in LLM workloads; hashing $M$, $K$, $N$, `transpose_b` instead of descriptor walk saves an estimated ~2--5 $\mu$s per dispatch |
| SDPA | 2 (custom hash) | Complex descriptor; semantic hash from `seq_len`, `head_dim`, `num_heads` is $O(1)$ vs $O(n)$ descriptor walk |
| Mcast | 1 (generic) | Low dispatch frequency; descriptor hash overhead is negligible |
| Gather | 1 (generic) | Standard pattern; no special validation needed |

Tier 2 promotion requires adding a `custom_program_hash` computation in the Blaze Python compilation path -- zero C++ changes per op. Tier 3 promotion (rare) requires adding `validate()` or `custom_hash()` hooks to the `OpTag`, replacing the macro-generated minimal tag with an extended tag (see [Section 4.2.1](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md)).

> **Note on hash time estimates:** The microsecond figures in this section (e.g., "~2--5 $\mu$s") are design estimates based on descriptor complexity analysis, not benchmarks from a running system. Actual values will depend on hardware, compiler optimizations, and descriptor size. They are provided to illustrate the relative cost of Tier 1 vs Tier 2, not as precise performance targets.

---

## 5.1.9 Edge Cases for MicroOp Registration

Three MicroOp categories require special consideration during registration.

### Multi-Device MicroOps

When a MicroOp runs on a multi-device mesh (e.g., T3K with 8 devices), the `MeshProgramDescriptor` contains one `(range, ProgramDescriptor)` entry per coordinate range. For SPMD execution, there is typically one entry with `MeshCoordinateRange(start=(0,0), end=(1,3))` covering all 8 devices. For MPMD execution, there may be multiple entries with distinct `ProgramDescriptor` objects per coordinate.

The adapter handles both cases identically: the factory wrapper delegates to `GenericMeshProgramFactory`, which iterates the ranges and calls `ProgramBuilder` once per coordinate. No adapter modifications are required for multi-device ops.

### MicroOps with Existing `custom_program_hash`

If a MicroOp author has already set `custom_program_hash` on the `MeshProgramDescriptor` (or passes it via the `BlazeAdapterAttributes.custom_program_hash` field), the adapter's three-tier hash cascade respects it at Priority 2 (see [Section 4.2.3](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md)). The existing hash is combined with the adapter's `type_hash` prefix, ensuring cache namespace isolation. No code changes are needed in the op definition -- the existing hash continues to work, now with per-op type isolation added.

### MicroOps with Zero Inputs (Source Ops)

Some MicroOps generate data without tensor inputs (e.g., `Arange`, `ConstantFill`). Their `io_tensors` contains only the output tensor (`io_tensors.size() == 1`). The adapter's `invoke()` currently requires `io_tensors.size() >= 2` for the general case (at least one input and one output), following `GenericOpDeviceOperation`'s validation logic. Source ops would need this precondition relaxed to `io_tensors.size() >= 1`. This is a minor adjustment: adding a conditional check in `validate_on_program_cache_miss()` that allows single-tensor `io_tensors` when the op's port declaration contains only outputs. Alternatively, the `OpTag` can declare a `static constexpr bool is_source_op = true;` flag that the adapter checks.

---

| [Section 5.1](./01_microop_registration_walkthrough.md) | [Section 5.2: FusedOp Registration](./02_fusedop_registration_walkthrough.md) |
|:---|---:|
