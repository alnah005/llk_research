# 6.1 Program Cache Integration

The program cache is TTNN's primary mechanism for avoiding redundant program compilation. A `uint64_t` hash maps to a `CachedProgramFactory` containing the compiled `MeshWorkload`; on a cache hit, `override_runtime_arguments` patches tensor buffer addresses and runtime args into the cached workload, skipping the entire `create_mesh_workload` path. The cache's effectiveness depends on two properties: (1) the hash correctly identifies equivalent programs (no false misses), and (2) the hash correctly distinguishes non-equivalent programs (no false hits). Registration via `BlazeDeviceOperationAdapter<OpTag>` changes how both properties are maintained for Blaze ops, without modifying the `ProgramCache` data structure itself.

This section traces the exact hash computation, cache lookup, and cache-hit fast path for Matmul (MicroOp) and GatedReduce (FusedOp), comparing `generic_op` to the adapter at every step. It includes quantitative cost models and benchmark estimates to ground the analysis in concrete numbers.

---

## 6.1.1 Hash Computation: Before and After

### Before: `GenericOpDeviceOperation::compute_program_hash`

When Blaze dispatches through `generic_op`, the hash is computed by `GenericOpDeviceOperation::compute_program_hash` (source: `generic_op_device_operation.cpp`, lines 48--119):

`GenericOpDeviceOperation::compute_program_hash` iterates `mesh_programs`, hashing each `(MeshCoordinateRange, ProgramDescriptor)` pair via `compute_program_descriptor_hash`. See [Section 2.1.5](../ch02_generic_op_dispatch/01_generic_op_internals.md) for the full implementation.

Two critical observations:

1. **No `type_hash` prefix.** The hash seed is `0`, not `type_hash<GenericOpDeviceOperation>`. Every `generic_op` invocation hashes from the same seed, so two structurally identical descriptors from different Blaze ops produce the same hash. For example, two elementwise unary ops that share the same kernel binary, CB layout, and runtime args (differing only in semantic identity) would collide. This is architecturally unsound — correct cache behavior depends on descriptors being structurally distinct, which is not guaranteed.

2. **`custom_program_hash` short-circuit.** Inside `compute_program_descriptor_hash`, if `program_descriptor.custom_program_hash` is set, the function returns it directly. Blaze can use this to avoid the $O(n)$ descriptor walk, but the user-supplied hash still lacks a type prefix.

### Cost Model: Descriptor Walk

The inner function `compute_program_descriptor_hash` walks every field of `ProgramDescriptor`. Let $K$ be the number of kernels, $C$ the number of CBs, $S$ the number of semaphores, and $A_k$ the average number of compile-time args per kernel. The descriptor-walk hash cost is:

$$T_{\text{desc}} = \sum_{d=1}^{D} \left[ c_K \cdot K_d + c_{\text{path}} \cdot K_d + c_A \cdot \sum_{k=1}^{K_d} A_{k,d} + c_C \cdot C_d + c_S \cdot S_d \right]$$

where $D$ is the number of mesh programs (typically 1 for single-device, up to 8 for T3K), and the $c$ constants represent per-element hash-combine cost:

| Operation | Symbol | Cost |
|---|---|---|
| `hash_combine` for scalar | $c_K$, $c_C$, $c_S$ | ~2 ns |
| `hash_combine` for string (kernel path) | $c_{\text{path}}$ | ~15--50 ns (length-dependent) |
| `hash_combine` for compile-time arg vector | $c_A$ | ~3 ns per element |
| `hash_combine` for `MeshCoordinateRange` | -- | ~4 ns |

### Back-of-Envelope Estimates: Representative Blaze Ops

The following table provides order-of-magnitude estimates for descriptor-walk hash time, derived from the per-element costs above (not empirical benchmarks). Single-device dispatch ($D = 1$):

| Blaze Op | Kernels ($K$) | CBs ($C$) | Sems ($S$) | CT Args ($\sum A_k$) | Estimated $T_{\text{desc}}$ |
|---|---|---|---|---|---|
| Elementwise (e.g., add) | 3 | 4 | 0 | 6 | ~250 ns |
| Matmul | 3 | 8 | 2 | 12 | ~450 ns |
| RMSNorm (fused) | 6 | 12 | 4 | 24 | ~850 ns |
| SDPA (fused attention) | 9 | 16 | 6 | 36 | ~1,400 ns |
| GatedMLP (fused) | 12 | 20 | 8 | 48 | ~2,100 ns |

For a model inference pass with 200 Blaze op invocations per token, the total hash computation time is:

$$T_{\text{hash,total}} \approx 200 \times 800 \text{ ns} = 160 \text{ us per token}$$

### After: `BlazeDeviceOperationAdapter<OpTag>::compute_program_hash`

When Blaze dispatches through the adapter (source: `blaze_device_operation_adapter.hpp`, [Section 4.2](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md)):

The adapter's `compute_program_hash` uses a three-tier cascade, with every tier seeded by `constexpr auto type_seed = tt::stl::hash::type_hash<BlazeDeviceOperationAdapter<OpTag>>` -- a compile-time constant unique to each `OpTag` instantiation. Priority 1 checks an `OpTag::custom_hash()` hook (Tier 3 ops, detected via SFINAE), Priority 2 checks an explicit `custom_program_hash` field passed from Python (Tier 2 ops), and Priority 3 falls back to the full descriptor walk (Tier 1 default). All tiers combine the result with `type_seed`, guaranteeing per-op cache namespace isolation. See [Section 4.2.3](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md) for the full implementation.

The structural difference is that **every tier begins with `type_seed = type_hash<BlazeDeviceOperationAdapter<OpTag>>`**, a compile-time constant unique to each `OpTag` instantiation.

The before/after hash computation details for specific ops (Matmul via Tier 1, GatedReduce via Tier 2) are shown in [Section 6.1.3](#613-the-three-tier-hash-cascade-in-practice), where each tier is explained with concrete examples.

---

## 6.1.2 Cache Key Design: Generic Adapter vs Individual Wrapper

The cache key has two conceptual components:

$$\text{cache\_key} = f(\text{type\_identity}, \text{program\_identity})$$

The following table compares how each component is derived under the three dispatch strategies:

| Component | `generic_op` | `BlazeDeviceOperationAdapter<OpTag>` (Tier 1) | Per-Op Wrapper (Tier 3) |
|-----------|-------------|-----------------------------------------------|------------------------|
| Type identity | None (seed `0`) | `type_hash<Adapter<OpTag>>` (per-op, compile-time) | `type_hash<MatmulDeviceOp>` (per-op, compile-time) |
| Program identity | `compute_program_descriptor_hash(pd)` | Same function, same walk | Semantic hash from typed attributes (e.g., `hash(M, K, N, fidelity)`) |
| Total hash cost | $O(k \cdot d)$ where $k$ = mesh programs, $d$ = descriptor size | Same as `generic_op` (Tier 1) or $O(1)$ (Tier 2) | $O(a)$ where $a$ = attribute count (typically 3--8) |
| Cross-op collision risk | Non-zero (shared type seed) | Zero (disjoint type seeds) | Zero (disjoint type seeds) |
| Cache namespace | Single flat namespace | Per-op namespace (via type prefix) | Per-op namespace (via type prefix) |

### Namespace Isolation Proof

For any two distinct tags `TagA` and `TagB`:

$$\text{type\_hash}(\texttt{Adapter<TagA>}) \neq \text{type\_hash}(\texttt{Adapter<TagB>})$$

because the mangled type name includes the tag struct's fully qualified name, and `TagA != TagB` implies distinct mangled names. The `type_hash` values are distinct with probability $1 - 2^{-64}$ (assuming a well-distributed hash function). This means:

- A Matmul cache entry can never be mistakenly retrieved by a GatedReduce lookup.
- A GatedReduce cache entry can never be mistakenly retrieved by a Matmul lookup.
- Both are disjoint from all `generic_op` entries (which use seed `0`).

---

## 6.1.3 The Three-Tier Hash Cascade in Practice

[Section 4.2](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md) defined three hash tiers abstractly. Here is how each tier applies to concrete ops with quantitative cost data:

### Tier 1: Full Descriptor Walk (Default)

**Applicable to:** most MicroOps (Matmul, Copy, Reduce, Mcast, Gather, Transpose, ...).

**How it works:** The adapter iterates `attrs.mesh_program_descriptor.mesh_programs`, hashing each `(MeshCoordinateRange, ProgramDescriptor)` pair via `compute_program_descriptor_hash`. The hash captures kernel count and source paths, CB configs, semaphore configs, and runtime arg vectors.

**Cost:** $O(k \cdot d)$ where $k$ is the number of mesh programs and $d$ is the total descriptor size. For a typical Matmul on a single device with 3 kernels, 8 CBs, and 32 cores: approximately 2--5 $\mu$s.

**Example (Matmul, single device) -- before/after comparison:**

Under `generic_op`, the hash seed is `0` (no type prefix), so the same descriptor walk produces a hash in a shared namespace. Under the adapter, the seed is `type_hash<Adapter<MatmulTag>>` (compile-time constant `0xA7F3...`), giving Matmul its own cache namespace. The descriptor walk itself is identical:

```text
type_seed = type_hash<Adapter<MatmulTag>>         // 0xA7F3...
hash = type_seed
hash_combine(hash, MeshCoordinateRange{{0,0},{0,0}})
hash_combine(hash, compute_program_descriptor_hash(pd))
  // pd has: 3 kernels, 8 CBs, 0 semaphores
  // hash captures: kernel paths, CB configs, RT args for 32 cores
  // cost: ~450 ns
return hash                                        // 0x82D1...
```

### Tier 2: Python-Computed Semantic Hash

**Applicable to:** FusedOps with dynamic composition (GatedReduce, FusedAttention, ...) and high-dispatch-frequency MicroOps where the $O(k \cdot d)$ descriptor walk is a measurable fraction of dispatch latency.

**How it works:** The Blaze compiler computes a semantic hash in Python from `(user_args, input_shapes, graph_topology_fingerprint)` and passes it as `custom_program_hash` in the `invoke()` call. The adapter's Priority 2 path combines this with `type_seed` and returns immediately.

**Cost:** $O(1)$ in C++ (one comparison + one hash combine, ~5 ns). The Python-side cost depends on the hash computation but is typically negligible compared to descriptor construction.

**Example (GatedReduce, T3K) -- before/after comparison:**

Under `generic_op`, the `custom_program_hash` is embedded per-`ProgramDescriptor` and processed inside `compute_program_descriptor_hash`, requiring iteration over all mesh programs. Under the adapter, the custom hash is a top-level attribute of `BlazeAdapterAttributes`, checked at Priority 2 before any descriptor walk begins. The adapter path is faster for multi-device ($D > 1$) and type-isolated via the `type_seed` prefix.

```python
# Python side (in BlazeCompiler._run_program):
semantic_hash = hash((
    tuple(sorted(fused_op.user_args.items())),
    tuple(t.shape for t in io_tensors),
    fused_op._graph_topology_fingerprint(),
))
ttnn.blaze.gated_reduce(io_tensors, descriptor, custom_program_hash=semantic_hash)
```

```text
# C++ side:
type_seed = type_hash<Adapter<GatedReduceTag>>     // 0xC4E1...
attrs.custom_program_hash = 0x5F2A...              // from Python
return hash_objects_with_default_seed(type_seed, 0x5F2A...)
                                                   // 0xE8B7...
// Total C++ cost: ~5 ns (no descriptor walk)
```

### Tier 3: OpTag Custom Hash Hook

**Applicable to:** the 3--5 highest-value ops where C++-side semantic hashing is justified (e.g., Matmul with `hash(M, K, N, fidelity, dtype)`).

**How it works:** The `OpTag` struct provides a `static custom_hash()` method detected via SFINAE. The adapter's Priority 1 path calls it before checking `custom_program_hash`.

**Example (hypothetical MatmulTagExtended):**

```cpp
struct MatmulTagExtended {
    static constexpr const char* name = "blaze::matmul";

    static std::optional<tt::stl::hash::hash_t> custom_hash(
        const MeshProgramDescriptor& desc,
        const std::vector<Tensor>& io_tensors) {
        const auto& a_shape = io_tensors[0].get_logical_shape();
        const auto& b_shape = io_tensors[1].get_logical_shape();
        auto M = a_shape[-2], K = a_shape[-1], N = b_shape[-1];
        return tt::stl::hash::hash_objects_with_default_seed(M, K, N);
    }
};
```

**Cost:** ~15--20 ns (shape extraction + 3 hash combines).

### Performance Comparison Table

| Hash Strategy | C++ Cost per Dispatch | Python Cost | Type Isolation | Use Case |
|:---|---:|---:|:---:|:---|
| `generic_op` (no type seed, descriptor walk) | 250--2,100 ns | None | No | Current default for all Blaze ops |
| `generic_op` with `custom_program_hash` | ~50 ns | Hash computation | No | Manual optimization, per-descriptor |
| Adapter Tier 1 (type seed + descriptor walk) | 250--2,100 ns | None | Yes | ~95% of ops (default after registration) |
| Adapter Tier 2 (type seed + Python hash) | ~5 ns | Hash computation | Yes | FusedOps, high-frequency MicroOps |
| Adapter Tier 3 (type seed + C++ semantic hash) | ~15--20 ns | None | Yes | 3--5 highest-value ops |
| Native TTNN op (typed attributes) | ~50--200 ns | N/A | Yes | Reference point (e.g., `BinaryDeviceOperation`) |

### Quantitative Improvement: Aggregate Hash Cost

| Scenario | Hash Time per Dispatch | Total (200 ops/token) | Speedup vs Current |
|---|---:|---:|:---:|
| All `generic_op` (current) | ~800 ns avg | ~160 us | -- |
| All adapter Tier 1 | ~800 ns avg | ~160 us | ~1x |
| 80% Tier 2, 20% Tier 1 | ~165 ns avg | ~33 us | **~5x** |
| 80% Tier 2, 15% Tier 3, 5% Tier 1 | ~47 ns avg | ~9.3 us | **~17x** |

Tier 1 is the zero-effort default. It has the same cost as `generic_op` but gains type isolation. Tiers 2 and 3 approach native TTNN op hash costs and are applied selectively.

---

## 6.1.4 The `override_runtime_arguments` Opportunity

On a cache hit, TTNN calls `override_runtime_arguments` instead of `create_mesh_workload`. This function patches tensor buffer addresses and runtime arguments into the cached `MeshWorkload`, avoiding the full program construction path. The cost savings are substantial: `create_mesh_workload` for a typical Matmul takes ~50--200 $\mu$s (kernel creation, CB allocation, semaphore setup), while `override_runtime_arguments` takes ~5--20 $\mu$s (memory copies into cached arg vectors).

### Before: `GenericMeshProgramFactory::override_runtime_arguments`

The generic factory's `override_runtime_arguments` (source: `generic_op_program_factory.cpp`, lines 52--113) operates on arbitrary descriptors:

`GenericMeshProgramFactory::override_program_runtime_arguments` iterates all kernel handles, all per-core runtime args, and all CB configs, copying updated values into the cached program -- even fields that did not change between invocations. See [Section 2.1.7](../ch02_generic_op_dispatch/01_generic_op_internals.md) for the full implementation.

This function must handle *any* descriptor structure because `generic_op` is type-erased. It iterates every kernel handle, every core's runtime args, and every CB config -- even fields that did not change between invocations.

**Problem:** Blaze currently rebuilds the entire `MeshProgramDescriptor` on every invocation, even when only tensor buffer addresses changed. The cost is dominated by the Python-side descriptor assembly (~50--200 $\mu$s for complex ops), not the C++ patching.

### After: Same Function, Same Cost (Tier 1)

For Tier 1 adapter registration, `BlazeAdapterMeshProgramFactory::override_runtime_arguments` delegates to the same `GenericMeshProgramFactory::override_runtime_arguments`:

```cpp
// PROPOSED: BlazeAdapterMeshProgramFactory::override_runtime_arguments
static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const operation_attributes_t& attrs,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {
    // Extract generic-compatible tensor args
    typename GenericOpDeviceOperation::tensor_args_t generic_args{
        .io_tensors = tensor_args.io_tensors,
        .output_tensor = tensor_args.output_tensor,
    };
    // Delegate to the same factory
    GenericMeshProgramFactory::override_runtime_arguments(
        cached_workload, attrs.mesh_program_descriptor,
        generic_args, tensor_return_value);
}
```

The cost is identical. The adapter does not improve `override_runtime_arguments` performance for Tier 1 -- it improves *cache hit rate* by providing type-isolated hashes, which means `override_runtime_arguments` is reached more reliably.

### After: The `shared_variables_t` Optimization Opportunity (Tier 3)

For Tier 3 per-op wrappers, a custom `shared_variables_t` can store kernel and CB handles at program creation time, enabling targeted patching on cache hit:

```cpp
// HYPOTHETICAL: Tier 3 MatmulMeshProgramFactory
struct MatmulMeshProgramFactory {
    struct shared_variables_t {
        KernelHandle reader_kernel;
        KernelHandle writer_kernel;
        KernelHandle compute_kernel;
        CBHandle cb_in0, cb_in1, cb_out;
    };

    static void override_runtime_arguments(
            CachedMeshWorkload<shared_variables_t>& cached,
            const BlazeAdapterAttributes& attrs,
            const BlazeAdapterTensorArgs& tensor_args,
            Tensor& output) {
        auto& sv = cached.shared_variables;
        auto& program = cached.workload.get_program_for(...);

        // Patch ONLY the buffer addresses that changed
        auto in0_addr = tensor_args.io_tensors[0].buffer()->address();
        auto in1_addr = tensor_args.io_tensors[1].buffer()->address();
        auto out_addr = output.buffer()->address();

        auto& reader_args = GetRuntimeArgs(program, sv.reader_kernel, CoreCoord{0,0});
        reader_args[0] = in0_addr;
        reader_args[1] = in1_addr;

        auto& writer_args = GetRuntimeArgs(program, sv.writer_kernel, CoreCoord{0,0});
        writer_args[0] = out_addr;

        // CB configs unchanged -- skip entirely
    }
};
```

This approach patches 3 values instead of iterating all kernels, all cores, and all CBs.

### `override_runtime_arguments` Cost Comparison

| Property | `GenericMeshProgramFactory` | `BlazeAdapterMeshProgramFactory` (Tier 1) | Custom Factory (Tier 3) |
|:---|:---|:---|:---|
| `shared_variables_t` contents | `num_kernel_handles` (count only) | Same (delegates) | Named handles: `reader_kernel`, `writer_kernel`, `compute_kernel`, `cb_in0`, ... |
| Override strategy | Iterate all kernels, all cores | Same (delegates) | Patch only changed addresses by handle |
| Cost per cache hit | ~5--20 $\mu$s | ~5--20 $\mu$s | ~0.5--1 $\mu$s |
| Per-op C++ required | None | None | ~50--100 lines |

### Full Cache-Hit Path Cost (Including Python Descriptor Rebuild)

The override function cost does not capture the full picture. On each cache hit, the current flow also rebuilds the `MeshProgramDescriptor` in Python:

| Component | Current (Full Rebuild) | Adapter (Tier 1, same override) | Adapter + Tier 3 override | Adapter + Tier 3 + Python skip (aspirational) |
|---|---:|---:|---:|---:|
| Python descriptor rebuild | 50--200 $\mu$s | 50--200 $\mu$s | 50--200 $\mu$s | **0** (requires Python-side changes) |
| Python/C++ marshaling | 5--20 $\mu$s | 5--20 $\mu$s | 5--20 $\mu$s | **0** (requires Python-side changes) |
| Hash computation | 250--2,100 ns | 250--2,100 ns (Tier 1) | 5 ns (Tier 2) | 5 ns (Tier 2) |
| `override_runtime_arguments` | 5--20 $\mu$s | 5--20 $\mu$s | ~0.5--1 $\mu$s | ~0.5--1 $\mu$s |
| **Total cache-hit latency** | **60--240 $\mu$s** | **60--240 $\mu$s** | **55--221 $\mu$s** | **~0.5--1 $\mu$s** |

The table separates four levels of optimization: (1) Tier 1 provides type isolation with no performance change; (2) Tier 3 override reduces the C++ override cost from ~5--20 $\mu$s to ~0.5--1 $\mu$s by patching only changed handles; (3) the aspirational last column additionally eliminates the Python descriptor rebuild, which requires the Blaze compiler to distinguish "first invocation" (full descriptor) from "repeat invocation" (tensor addresses only). This Python-side change is outside the adapter's scope and is recommended as a Phase 2 enhancement after basic registration is deployed.

---

## 6.1.5 Cache Invalidation Comparison

| Invalidation Trigger | `generic_op` Effect | Adapter Effect | Difference |
|:---|:---|:---|:---|
| `mesh_device.program_cache.clear()` | All `generic_op` entries lost | All adapter entries lost (same cache) | None -- both use the same `ProgramCache` instance |
| New device shape (mesh reconfiguration) | All entries lost (cache is per-device) | All entries lost (same mechanism) | None |
| `cache_misses_allowed = false` with new config | `TT_THROW` with message `"generic_op: ..."` | `TT_THROW` with message `"BlazeDeviceOperationAdapter<MatmulTag>: ..."` | Better error message identifies the op |
| Tensor shape change (same op, different shapes) | Descriptor hash changes -> miss, new entry | Same (Tier 1) or semantic hash changes -> miss (Tier 2) | Tier 2 may produce fewer false misses if semantic hash is shape-aware |
| `NO_DISPATCH` graph capture mode | Skip caching (buffer addresses invalid) | Skip caching (same mechanism, [Section 1.3.1](../ch01_ttnn_registration_contract/03_program_cache_graph_tracing_and_profiling.md)) | None |
| Bug fix in a single Blaze op's kernel | Descriptor hash changes -> automatic miss for affected op | Same: descriptor or semantic hash changes -> miss | Both paths get automatic invalidation; adapter's advantage is that the type-prefixed miss is unambiguous in profiler traces |
| Adding a new Blaze op | Zero invalidation (new descriptor in shared namespace) | Zero invalidation (new `type_hash` creates new namespace) | Same, but adapter provides structural isolation guarantee |

### Cache Entry Count Impact

With `generic_op`, all Blaze ops share one type-hash namespace. If a model uses 20 distinct Blaze ops, each with 3 input-shape configurations, the cache contains approximately 60 entries, all keyed under the same `type_hash<GenericOpDeviceOperation>`.

With the adapter, the same 60 entries are distributed across 20 per-op namespaces (3 entries each). The total cache size is the same, but:

1. **Lookup is no more expensive** -- `ProgramCache::contains` is an `unordered_map::find` on the full 64-bit hash, so namespace distribution is invisible to the lookup cost.
2. **Debugging is dramatically easier** -- when a cache miss occurs in production (`cache_misses_allowed = false`), the error message identifies the specific op and its hash.
3. **Collision probability drops** -- with $N$ ops sharing one seed vs. $N$ ops with disjoint seeds, the birthday-paradox collision probability decreases from $O(N^2 / 2^{64})$ within the shared namespace to $O(n_i^2 / 2^{64})$ within each per-op namespace (where $n_i \ll N$).

### Per-Op Invalidation Cost Model

Let $E$ be the number of cache entries and $T_{\text{rebuild}}$ the average program rebuild time:

| Scenario | Entries Invalidated | $T_{\text{rebuild}}$ | Total Cost |
|---|---:|---:|---:|
| `generic_op`, full cache clear | ~960 | ~300 $\mu$s | ~288 ms |
| Adapter, full cache clear | ~960 | ~300 $\mu$s | ~288 ms (same) |
| Adapter, single op invalidated | ~64 (one op's variants) | ~300 $\mu$s | ~19 ms |

---

## 6.1.6 What Changes, What Stays the Same

| Aspect | Changes? | Before (`generic_op`) | After (`Adapter<OpTag>`) |
|:---|:---:|:---|:---|
| `ProgramCache` data structure | No | `unordered_map<uint64_t, CachedProgramFactory>` | Same |
| Cache storage location | No | Per-`MeshDevice` | Same |
| Cache enable/disable API | No | `program_cache.enable()` / `.disable()` | Same |
| `CachedProgramFactory` type | No | `unique_any<4096, 32>` holding `CachedMeshWorkload` | Same |
| Hash seed | **Yes** | `0` (no type prefix) | `type_hash<Adapter<OpTag>>` (per-op) |
| Hash function for descriptor walk | No | `compute_program_descriptor_hash` | Same function |
| `custom_program_hash` support | **Yes** | Per-`ProgramDescriptor` field only | Per-`ProgramDescriptor` (Tier 1) + per-invoke attribute (Tier 2) + OpTag hook (Tier 3) |
| `override_runtime_arguments` logic | No | `GenericMeshProgramFactory::override_runtime_arguments` | Delegates to same function (Tier 1) |
| `override_runtime_arguments` cost | No (Tier 1); **Yes** (Tier 3) | ~5--20 $\mu$s | Same (Tier 1); ~0.5--1 $\mu$s (Tier 3) |
| `NO_DISPATCH` cache skip | No | `ProcessorHooks::get_block()` gates caching | Same mechanism |
| Cache miss error message | **Yes** | `"generic_op: program cache miss ..."` | `"BlazeDeviceOperationAdapter<MatmulTag>: program cache miss ..."` |
| Cross-op collision risk | **Yes** | Non-zero (shared seed) | Zero (disjoint seeds) |
| Total cache entry count | No | Same entries, same data | Same entries, same data (different hash keys) |

---

| **Section 6.1** | [Chapter Index](index.md) | [Next: Graph Tracing and Inspector](02_graph_tracing_and_inspector.md) |
|:---|:---:|---:|
