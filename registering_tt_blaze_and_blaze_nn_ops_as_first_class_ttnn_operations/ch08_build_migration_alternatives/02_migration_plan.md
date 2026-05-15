# 8.2 Migration Plan

This section presents a five-phase migration plan (Phase 0--4) for promoting Blaze ops from `generic_op` dispatch to first-class TTNN registrations. Each phase has explicit entry criteria, exit criteria, deliverables, effort estimates, risk assessment, and a "first-week scenario" illustrating what a team experiences when adopting that phase. The plan is designed so that each phase delivers standalone value and can be paused without breaking existing functionality.

---

## 8.2.1 Migration Overview

```text
Phase 0         Phase 1           Phase 2            Phase 3            Phase 4
Quick Win       Generic Adapter   Batch Registration  FusedOp Reg.       blaze-nn
(no adapter)    (top 10 ops)      (all MicroOps)      (all FusedOps)     Integration
    |               |                 |                   |                  |
    v               v                 v                   v                  v
+----------+   +------------+   +---------------+   +---------------+  +----------+
| op_name  |   | Adapter    |   | Codegen /     |   | FusedOp hash  |  | _dispatch|
| in MPD + |   | template + |   | X-macro for   |   | strategy +    |  | rewiring |
| profiler |   | 10 manual  |   | 60+ MicroOps  |   | 50+ FusedOps  |  | + module |
| hooks    |   | regs       |   | + CI          |   | + override_   |  | fallback |
+----------+   +------------+   +---------------+   | runtime_args  |  +----------+
                                                     +---------------+
```

### Cumulative Value Delivery

| Phase | Profiler Visibility | Cache Isolation | Graph Tracing | blaze-nn Integration |
|:-----:|:-------------------:|:---------------:|:-------------:|:--------------------:|
| 0 | Named ops in Tracy | Partial (hash prefix) | No | No |
| 1 | Named ops + per-op zones | Top 10 ops | Top 10 ops | No |
| 2 | All MicroOps named | All MicroOps | All MicroOps | No |
| 3 | All ops named | All ops | All ops | No |
| 4 | All ops named | All ops | All ops | Yes |

---

## 8.2.2 Phase 0: Immediate Quick Win (No C++ Adapter)

### Goal

Deliver profiler visibility for Blaze ops without any adapter code. Add an `op_name` field to `MeshProgramDescriptor` and propagate it through the existing `GenericOpDeviceOperation` path.

### Mechanism

```cpp
// PROPOSED: Change to mesh_program_descriptor.hpp (~5 lines)
struct MeshProgramDescriptor {
    // ... existing fields ...
    std::optional<std::string> op_name;  // NEW: Optional Blaze op identity
};
```

Modify `GenericOpDeviceOperation::compute_program_hash()` to incorporate `op_name`:

```cpp
// PROPOSED: Change to generic_op_device_operation.cpp (~10 lines)
tt::stl::hash::hash_t GenericOpDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs,
    const tensor_args_t&) {
    size_t hash = 0;
    // NEW: prefix with op_name hash for namespace isolation
    if (attrs.op_name.has_value()) {
        ttsl::hash::hash_combine(hash, attrs.op_name.value());
    }
    for (const auto& [range, desc] : attrs.mesh_programs) {
        ttsl::hash::hash_combine(hash, range);
        ttsl::hash::hash_combine(hash,
            compute_program_descriptor_hash(desc));
    }
    return hash;
}
```

Modify profiler/Tracy hooks to extract and display `op_name`:

```cpp
// PROPOSED: Change to op_profiler.hpp (~10 lines)
auto display_name = (!attrs.op_name.has_value())
    ? "GenericOpDeviceOperation"
    : "blaze::" + attrs.op_name.value();
```

On the Blaze side, `BlazeCompiler.compile()` sets `op_name`:

```python
# PROPOSED: Change to blaze/compiler.py (~3 lines)
mesh_descriptor = MeshProgramDescriptor(...)
mesh_descriptor.op_name = self._graph.root_op.name  # e.g., "matmul"
```

### First-Week Outcome

Tracy displays named ops (`[matmul] 45.2 us`, `[rmsnorm] 12.1 us`) instead of `[GenericOpDeviceOperation]`. Cache entries gain `op_name` prefixes, eliminating cross-op pollution. All existing tests pass unchanged. Graph tracing and `compute_output_specs` still require Phase 1.

### Quantitative Assessment

| Metric | Value |
|--------|-------|
| C++ lines changed | ~30 |
| Python lines changed | ~5 |
| Files modified | 4 (mesh_program_descriptor.hpp, generic_op_device_operation.cpp, op_profiler.hpp, compiler.py) |
| New files | 0 |
| Estimated effort | 1--2 engineering-days |
| Build impact | Zero (no new template instantiations) |

### What Phase 0 Does NOT Deliver

| Feature | Status | Requires |
|---------|:------:|----------|
| Per-op `type_hash` isolation | No | Phase 1 |
| Per-op Tracy zones | Partial (name in annotation, same zone type) | Phase 1 |
| Graph node identity | No | Phase 1 |
| `compute_output_specs` | No | Phase 1 |
| Per-op validation | No | Phase 1 |
| blaze-nn integration | No | Phase 4 |

### Entry Criteria

- None. Can be implemented immediately.

### Exit Criteria

- All Blaze ops show their name in Tracy flamegraphs
- Program cache hash includes `op_name` prefix when set
- `MeshProgramDescriptor.op_name` field exists and is populated by `BlazeCompiler`
- Existing tests pass (backward compatibility: absent `op_name` yields identical hash values)

---

## 8.2.3 Phase 1: Generic Adapter for Top Ops

### Goal

Implement `BlazeDeviceOperationAdapter<OpTag>` ([Section 4.2](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md)) and register the top 5--10 most performance-critical Blaze ops. Validate the adapter design end-to-end.

### Candidate Ops

Selection criteria: ops that dominate inference time in LLM workloads and benefit most from cache isolation and profiler identity.

| Rank | Op Name | Type | Rationale |
|:----:|---------|:----:|-----------|
| 1 | matmul | MicroOp | 40--60% of inference time; cache isolation critical |
| 2 | rmsnorm | MicroOp/Fused | Called per-layer; high invocation count |
| 3 | sdpa | MicroOp | Attention kernel; complex descriptor |
| 4 | sdpa_decode | MicroOp | Autoregressive decode hot path |
| 5 | rope | MicroOp | Called per-layer; moderate cost |
| 6 | mcast | MicroOp | Data movement; high frequency |
| 7 | gated_mlp | FusedOp | MLP block; validates FusedOp adapter path |
| 8 | copy | MicroOp | Most frequent data movement op |
| 9 | reduce | MicroOp | Used in normalization and attention |
| 10 | silu | MicroOp | Activation function; validates elementwise path |

### First-Week Outcome

Tracy shows distinct per-op zones (e.g., `[BlazeDeviceOperationAdapter<MatmulTag>]`), graph tracer displays named nodes, and parallel correctness tests confirm bit-exact equivalence. One-time cache miss occurs as entries migrate to new `type_hash` keys; see Section 8.2.9.

### Quantitative Assessment

| Metric | Value |
|--------|-------|
| New C++ lines | ~525 (template: ~350, registrations: ~80, bindings: ~80, CMake: ~15) |
| Modified Python lines | ~85 (compiler.py dispatch: ~60, test harness: ~25) |
| Test lines | ~200 (C++ adapter equivalence tests, cache isolation tests) |
| Estimated effort | 2--3 engineering-weeks |
| Build time impact | +3--5 seconds incremental |

### Testing Strategy

```python
# PROPOSED: Parallel execution test pattern
def test_adapter_equivalence(op_name, input_tensors, descriptor):
    """Run through both paths and compare outputs bit-exactly."""
    # Path A: generic_op (baseline)
    output_generic = ttnn.generic_op(input_tensors, descriptor)

    # Path B: registered adapter
    registered_fn = getattr(ttnn.blaze, op_name)
    output_registered = registered_fn(input_tensors, descriptor)

    # Numerical equivalence (bit-exact for deterministic ops)
    assert torch.equal(
        ttnn.to_torch(output_generic),
        ttnn.to_torch(output_registered)
    ), f"Numerical mismatch for {op_name}"

    # Cache key isolation: verify different type_hash seeds
    assert get_cache_key(output_generic) != get_cache_key(output_registered)
```

### Risk Assessment

| Risk | Severity (1-5) | Likelihood (1-5) | Mitigation |
|------|:-:|:-:|------------|
| Adapter template does not satisfy `DeviceOperationConcept` | 5 | 1 | Compile-time `static_assert` in template |
| `GenericMeshProgramFactory` wrapper introduces behavioral divergence | 4 | 2 | Bit-exact comparison tests on all 10 ops |
| `_resolve_dispatch()` fallback introduces import-order dependency | 3 | 3 | Lazy resolution with `_TTNN_BLAZE_CHECKED` flag ([Section 7.2](../ch07_blaze_nn_integration/02_rewired_dispatch_and_module_integration.md)) |
| Build breaks in downstream CI | 3 | 2 | Feature flag: `TTNN_ENABLE_BLAZE_ADAPTER=ON` in CMake |

### Entry Criteria

- Phase 0 complete (`op_name` in `MeshProgramDescriptor`)
- `BlazeDeviceOperationAdapter<OpTag>` design reviewed ([Chapter 4](../ch04_adapter_pattern/index.md))

### Exit Criteria

- 10 ops registered and passing numerical equivalence tests
- Tracy profiler shows distinct per-op names for all 10 ops
- Program cache entries isolated per-op (verified via cache dump)
- `generic_op` path still functional for all ops (feature flag off)
- No performance regression >2% on LLM inference benchmark

---

## 8.2.4 Phase 2: Batch Registration for All MicroOps

### Goal

Scale from 10 hand-registered ops to all ~62 MicroOps using the X-macro approach from [Section 5.3](../ch05_op_registration/03_batch_registration_and_scaling.md).

### Mechanism

1. Populate `blaze_op_list.hpp` with all MicroOp names
2. Macro expansions generate tag structs, registrations, and bindings
3. CI validation script checks that `blaze_op_list.hpp` entries match `BlazeOp._class_registry`
4. Sharded nanobind `.cpp` files for parallel compilation

### First-Week Outcome

Every MicroOp gains its own profiler identity; Tracy's `[GenericOp]` bucket contains only FusedOps. CI validation catches X-macro drift, and the sweep framework auto-discovers all registered MicroOps.

### Quantitative Assessment

| Metric | Value |
|--------|-------|
| New C++ lines | ~80 (50 X-macro entries + 30 CI script) |
| Test lines | ~330 (sweep-based equivalence tests for 60 MicroOps, CI validation scripts) |
| Modified files | 2 (blaze_op_list.hpp, CMakeLists.txt for shards) |
| Estimated effort | 1--2 engineering-weeks |
| Build time impact | +25--40 seconds (62 new instantiations), ~8--12s with sharding |

### Risk Assessment

| Risk | Severity (1-5) | Likelihood (1-5) | Mitigation |
|------|:-:|:-:|------------|
| X-macro list drift from Blaze registry | 3 | 4 | CI check: `diff(blaze_op_list, BlazeOp._class_registry)` |
| Template instantiation overwhelms build | 3 | 2 | Sharded compilation (6 files, ~10 ops each) |
| Untested ops have edge-case failures | 3 | 3 | Sweep test with configurable input shapes |

### Entry Criteria

- Phase 1 complete (adapter template proven on 10 ops)
- X-macro infrastructure validated

### Exit Criteria

- All ~62 MicroOps registered in `ttnn.blaze.*` namespace
- CI check enforces sync between X-macro list and Blaze registry
- Sweep test passes for all MicroOps with >95% shape coverage
- Build time increase <60 seconds with sharding

---

## 8.2.5 Phase 3: FusedOp Registration

### Goal

Register all ~50 FusedOps using the same adapter template. Address FusedOp-specific challenges: graph-topology-aware hashing, `_graph_branch_keys` for dynamic composition, and multi-phase `override_runtime_arguments` correctness (detailed in [Section 5.2](../ch05_op_registration/02_fusedop_registration_walkthrough.md)).

### FusedOp-Specific Considerations

| Concern | Description | Mitigation |
|---------|-------------|------------|
| **Hash strategy** | FusedOps with identical `compose()` graphs but different `user_args` need distinct cache entries. Hash must incorporate graph topology + user-supplied arguments + tensor shapes. | Tier 2 hashing with `custom_program_hash` including `_graph_branch_keys` |
| **Dynamic composition** | Some FusedOps use conditional logic in `compose()` producing structurally different descriptors. | `_graph_branch_keys` captures branch decisions for hash differentiation |
| **`override_runtime_arguments`** | Multi-phase programs share CBs. On cache hit, must update buffer addresses for ALL kernels. | `GenericMeshProgramFactory` already handles this; requires validation for complex FusedOps |

### First-Week Outcome

Tracy's `[GenericOp]` bucket disappears entirely. Conditional FusedOps use Tier 2 hashing with `_graph_branch_keys`. Cache entries per namespace drop from ~450 to ~4, and hit rate rises from 98.7% to 99.1% on Llama-70B.

### Quantitative Assessment

| Metric | Value |
|--------|-------|
| New C++ lines | ~60 (50 X-macro entries + 10 hash helper) |
| Modified Python lines | ~40 (hash computation in FusedOp compile path) |
| Test lines | ~300 (FusedOp equivalence tests, conditional composition coverage, hash collision tests) |
| Estimated effort | 2--3 engineering-weeks |
| Build time impact | +20--30 seconds (50 new instantiations) |

### Risk Assessment

| Risk | Severity (1-5) | Likelihood (1-5) | Mitigation |
|------|:-:|:-:|------------|
| Dynamic composition breaks cache correctness | 5 | 3 | `_graph_branch_keys` in hash; test matrix covers all branch combinations |
| Multi-phase `override_runtime_arguments` misses a kernel | 4 | 2 | Assertion: verify kernel count matches between cached and override paths |
| FusedOp hash collision due to topology aliasing | 3 | 2 | Include `compose_fn.__qualname__` in hash |

### Entry Criteria

- Phase 2 complete (all MicroOps registered)
- FusedOp hash strategy designed and code-reviewed

### Exit Criteria

- All ~50 FusedOps registered
- Dynamic composition ops tested with multiple branch combinations
- `override_runtime_arguments` verified for all multi-phase programs
- No cache hit/miss ratio regression on LLM inference benchmark

---

## 8.2.6 Phase 4: blaze-nn Integration

### Goal

Update blaze-nn's `_dispatch()` to route through registered `ttnn.blaze.*` callables, with three-level fallback and environment variable override. Detailed design in [Section 7.2](../ch07_blaze_nn_integration/02_rewired_dispatch_and_module_integration.md).

### Deliverables

1. Modified `OpMapping` with `ttnn_callable` lazy resolution
2. Three-level fallback: registered callable -> `generic_op` -> error
3. `BLAZE_NN_FORCE_GENERIC` environment variable escape hatch
4. Diagnostic API: `blaze_nn.debug.dispatch_report()` showing which ops use registered vs generic path
5. End-to-end model tests: transformer attention block, MoE layer

### First-Week Outcome

Tracy flamegraphs show named ops through the full model hierarchy (e.g., `LlamaModel.forward() > Attention > [ttnn::blaze::matmul]`). Graph capture displays structured named nodes. All existing blaze-nn tests pass unchanged.

### Quantitative Assessment

| Metric | Value |
|--------|-------|
| Modified Python lines | ~120 (_registry.py: ~80, tracing.py: ~20, diagnostic API: ~20) |
| Test lines | ~200 (end-to-end model tests, dispatch report validation) |
| Estimated effort | 1--2 engineering-weeks |

### Risk Assessment

| Risk | Severity (1-5) | Likelihood (1-5) | Mitigation |
|------|:-:|:-:|------------|
| Import-order dependency between blaze-nn and ttnn.blaze | 3 | 3 | Lazy resolution ([Section 7.2](../ch07_blaze_nn_integration/02_rewired_dispatch_and_module_integration.md)) |
| Tracing context interaction with registered dispatch | 3 | 2 | Both graph-mode and direct-mode tested end-to-end |
| `BLAZE_NN_FORCE_GENERIC` inadvertently left on in production | 2 | 2 | Warning log on startup when flag is set |

### Entry Criteria

- Phase 3 complete (all ops registered)
- blaze-nn `_registry.py` modifications code-reviewed

### Exit Criteria

- `blaze_nn.debug.dispatch_report()` shows >95% of ops using registered path
- End-to-end transformer inference produces identical outputs via both paths
- No performance regression on blaze-nn model benchmarks

---

## 8.2.7 Cumulative Effort Summary

| Phase | New/Modified LoC | Effort (eng-weeks) | Cumulative LoC | Cumulative Effort |
|:-----:|:-----------------:|:------------------:|:--------------:|:-----------------:|
| 0 | ~35 | 0.5 | 35 | 0.5 |
| 1 | ~810 | 2.5 | 845 | 3.0 |
| 2 | ~410 | 1.5 | 1,255 | 4.5 |
| 3 | ~400 | 2.5 | 1,655 | 7.0 |
| 4 | ~320 | 1.5 | 1,975 | 8.5 |

**Total estimated effort: 7--10 engineering-weeks** for a single engineer, or **4--6 calendar weeks** with two engineers overlapping on Phases 2/3.

### Comparison: Per-Op C++ Porting Alternative

For context, the "deep integration" alternative of rewriting each op's `emit()` logic as C++ program factories ([Section 3.2](../ch03_compilation_model_gap/02_translation_challenges.md)) would require:

| Metric | Adapter Migration (This Plan) | Deep Integration |
|--------|:-----------------------------:|:----------------:|
| Total LoC | ~2,000 | ~53,000 |
| Engineering effort | 7--10 weeks | 40--60 weeks |
| Ongoing maintenance | Low (adapter is generic) | High (dual Python + C++ per op) |
| Registration fidelity | Full (via adapter) | Full (native) |
| Performance optimization potential | Moderate (cache + profiling) | Maximum (custom factories) |

The adapter migration delivers 95% of the integration benefit at 4% of the effort.

### Migration Timeline

```text
Week 1-2:   Phase 0   (op_name in descriptor)
Week 3-4:   Phase 1   (top 10 MicroOps registered)
Week 5-6:   Phase 2   (all MicroOps registered)
Week 7-8:   Phase 3   (all FusedOps registered)
Week 9-10:  Phase 4   (blaze-nn integration)

Phases 2/3 are parallelizable. Phase 4 can begin once Phase 1 is stable.
```

---

## 8.2.8 Backward Compatibility Guarantees

Every phase preserves backward compatibility:

| Guarantee | Mechanism | Verified By |
|-----------|-----------|-------------|
| `ttnn.generic_op()` remains functional | Adapter is purely additive; `generic_op` is not modified | Regression test suite |
| `BlazeCompiler` API unchanged | `compile()` and `run()` signatures unchanged; `_resolve_dispatch()` is internal | API compatibility test |
| `MeshProgramDescriptor` unchanged for existing users | `op_name` field is `std::optional` with default `std::nullopt` (Phase 0) | Serialization test |
| Existing `custom_program_hash` honored | Adapter's hash cascade checks `custom_program_hash` before descriptor walk ([Section 6.1](../ch06_caching_tracing_profiling/01_program_cache_integration.md)) | Cache equivalence test |
| `BLAZE_NN_FORCE_GENERIC` override | Environment variable forces all dispatch through `generic_op` | Flag test |
| Hash disjointness | `type_hash<BlazeDeviceOperationAdapter<Tag>>` $\neq$ `type_hash<GenericOpDeviceOperation>` for all `Tag` types; cache entries cannot collide ([Section 4.1](../ch04_adapter_pattern/01_design_constraints_and_goals.md)) | `static_assert` + hash dump |
| Rollback is possible at every phase | Phase 0: revert PRs. Phase 1+: set `_BLAZE_ADAPTER_AVAILABLE = False`. Phase 4: set `BLAZE_NN_FORCE_GENERIC=1`. | Rollback test |

---

## 8.2.9 Risk Mitigation Playbooks

### Playbook: Cache Miss Storm at Upgrade

**Scenario:** After upgrading to Phase 1, a production serving cluster experiences elevated latency for the first ~100 requests as program caches warm under new hash keys.

**Mitigation:**
1. Run a warm-up script that dispatches representative workloads before accepting traffic.
2. If warm-up is not feasible, use `_BLAZE_ADAPTER_AVAILABLE = False` to revert to `generic_op` temporarily.
3. Once caches are warm, re-enable the adapter. Cache entries persist across requests within a device session.

### Playbook: FusedOp Hash Regression

**Scenario:** A FusedOp's hash (via the adapter's three-tier cascade) produces a different hash for semantically identical programs, causing excessive cache misses.

**Mitigation:**
1. Set `custom_program_hash` in Python for the affected FusedOp (Tier 2 hashing).
2. The hash should include only inputs that affect program structure: `_graph_branch_keys`, tensor shapes, `user_args`.
3. Monitor cache hit rates per op via the profiler.

---

## 8.2.10 Testing Matrix

| Test Category | Phase 0 | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---------------|:-------:|:-------:|:-------:|:-------:|:-------:|
| Existing TTNN unit tests | Pass | Pass | Pass | Pass | Pass |
| Existing Blaze unit tests | Pass | Pass | Pass | Pass | Pass |
| Parallel execution (MicroOps) | N/A | Top 10 | All 60 | All 60 | All 60 |
| Parallel execution (FusedOps) | N/A | N/A | N/A | All 50 | All 50 |
| Program cache correctness | Manual | Automated | Automated | Automated | Automated |
| Profiler trace validation | Manual | Automated | Automated | Automated | Automated |
| Graph capture validation | N/A | Top 10 | All 60 | All 112 | All 112 |
| blaze-nn end-to-end | N/A | N/A | N/A | N/A | Full model |
| Drift detection (CI) | N/A | Enabled | Enabled | Enabled | Enabled |

---

| [Section 8.1: Build System Integration](01_build_system_integration.md) | **Section 8.2** | [Section 8.3: Existing Patterns and Precedents](03_existing_patterns_and_precedents.md) |
|:---|:---:|---:|
