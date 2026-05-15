# 8.4 Open Questions

This section catalogs seven unresolved design questions that emerged during the analysis in Chapters 1--7. Each question is presented with a scenario that triggers it, analysis of options, evaluation criteria, and a recommended resolution path. A priority matrix (urgency x difficulty) ranks the questions to help teams decide what to tackle first. These questions are not blockers for Phase 0--1 of the migration ([Section 8.2](02_migration_plan.md)) but must be resolved before Phase 2+ reaches production stability.

---

## 8.4.1 Priority Matrix

Each question is scored on two axes:

- **Urgency** (1--3): How soon must this be resolved? 3 = must resolve before Phase 1 begins. 2 = should resolve during Phases 1--3. 1 = can defer past Phase 4.
- **Difficulty** (1--3): How hard is the solution? 1 = straightforward engineering. 2 = requires design decisions with significant tradeoffs. 3 = requires fundamental framework changes or cross-team coordination.

| # | Question | Urgency | Difficulty | Score | Recommended Phase |
|---|----------|:-------:|:----------:|:-----:|:-----------------:|
| 1 | Repository ownership | 3 | 2 | $3 \times 2 = 6$ | Before Phase 1 |
| 2 | Non-SPMD mesh programs | 2 | 2 | $2 \times 2 = 4$ | Phase 3 |
| 3 | Versioning and cache staleness | 2 | 2 | $2 \times 2 = 4$ | Before Phase 2 |
| 4 | Thread safety | 2 | 1 | $2 \times 1 = 2$ | Phase 1 |
| 5 | Dynamic registration | 1 | 3 | $1 \times 3 = 3$ | Post-Phase 4 |
| 6 | Output tensor ownership | 1 | 2 | $1 \times 2 = 2$ | Post-Phase 4 |
| 7 | Performance regression | 2 | 2 | $2 \times 2 = 4$ | Phase 1 |

### Priority Visualization

```text
              Difficulty
              Low (1)    Med (2)    High (3)
         +-----------+-----------+-----------+
High (3) |           |    [1]    |           |
Urgency  |           |           |           |
Med  (2) |    [4]    | [2][3][7] |           |
         |           |           |           |
Low  (1) |           |    [6]    |    [5]    |
         +-----------+-----------+-----------+
```

**Interpretation:**
- **Top-center (high urgency, medium difficulty):** Do first. Questions 1 (repository ownership) and 3 (versioning) are the highest priority items.
- **Top-left (high urgency, low difficulty):** Quick wins. No questions currently here.
- **Center (medium urgency, medium difficulty):** Do during implementation. Questions 2 (non-SPMD) and 7 (performance) should be resolved within their respective phases.
- **Center-left (medium urgency, low difficulty):** Question 4 (thread safety) -- easy to validate, low risk.
- **Bottom-right (low urgency, high difficulty):** Defer. Question 5 (dynamic registration) requires TTNN framework changes.

---

## 8.4.2 Open Question 1: Repository Ownership

### Scenario

Two engineers -- one from the Blaze team, one from TT-Metal infrastructure -- both have PRs open that modify `blaze_op_list.hpp`. The Blaze engineer adds a new op. The infra engineer refactors the X-macro expansion. Their PRs conflict. Who owns the file?

### Options

| Option | Location | Build Simplicity | Release Decoupling | CI Integration |
|--------|----------|:---:|:---:|:---:|
| **A: tt-metal in-tree** | `ttnn/cpp/ttnn/operations/blaze/` | 5 | 2 | 5 |
| **B: tt-blaze in-tree** | `blaze/adapter/` + separate build | 3 | 4 | 3 |
| **C: Dedicated repo** | `tt-blaze-ttnn-adapter/` | 2 | 5 | 2 |

### Ownership Models

**Model A: Blaze team owns entire directory.** Precedented by moreh ops ([Section 8.3.2](#832-precedent-moreh-operations)). Pro: alignment between op definitions and registrations. Con: Blaze team must follow TT-Metal CI/merge rules.

**Model B: Split ownership.** Adapter template (`@ttnn-infra-team`) + op list (`@blaze-team`). Pro: each team maintains what it knows best. Con: two owners on one directory increases merge friction.

### Evaluation Criteria

1. **Release cadence:** Same cadence -> Option A. Independent -> Option B or C.
2. **Op list change frequency:** Monthly -> Option A overhead is negligible. Weekly -> Option B/C reduces friction.
3. **CI ownership:** Which team's CI runs adapter tests?

### Recommendation

Start with **Option A (in-tree in tt-metal)** with **Model A ownership** (Blaze team owns the directory) for Phases 1--3. The adapter template (~200 lines) changes rarely. The moreh precedent validates this model. Reassess for Option B if release cadence divergence causes friction. The hybrid approach ensures new ops work immediately via `generic_op` fallback and are promoted to the X-macro in the next release cycle.

### Resolution Timeline

Before Phase 1. The choice affects where generated files live and which CI pipeline validates them.

---

## 8.4.3 Open Question 2: Non-SPMD Mesh Programs

### Scenario

A Blaze developer writes `ring_reduce` that uses different kernels on different devices: device 0 runs "gather," device 1 runs "reduce," device 2 runs "scatter." The `MeshProgramDescriptor` contains three different `ProgramDescriptor` objects. The developer registers it using the adapter.

**Question:** Is the cache hash correct for heterogeneous programs?

### Analysis

The adapter's `compute_program_hash` (via the three-tier cascade from [Section 4.2](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md)):

1. **Tier 1 (descriptor hashing):** Hashes the full `MeshProgramDescriptor`, including all per-coordinate `ProgramDescriptor` objects. Correct but expensive.
2. **Tier 2 (custom hash):** If `custom_program_hash` is set, the adapter uses it directly. Risk: custom hash may not account for per-coordinate variation.
3. **Tier 3 (per-op override):** A dedicated `OpTag` hash hook could implement efficient topology-aware hashing.

The concern is with Tier 2/3: user-provided hashes that do not account for per-coordinate variation could return stale programs for the wrong mesh coordinate.

### Recommendation

1. Add a runtime assertion in `BlazeCompiler`: if `MeshProgramDescriptor` has >1 distinct `ProgramDescriptor` and no `custom_program_hash` is set, require one to ensure correct per-coordinate caching.

```python
# PROPOSED: In BlazeCompiler.compile()
if len(mesh_descriptor.mesh_programs) > 1:
    if not all_programs_identical(mesh_descriptor):
        assert mesh_descriptor.custom_program_hash is not None, (
            f"Non-SPMD op '{op.name}' must set custom_program_hash "
            f"to avoid expensive per-dispatch descriptor hashing"
        )
```

2. Document the requirement: Tier 2/3 custom hashes for non-SPMD ops must incorporate mesh topology.
3. Add a test case: register a non-SPMD op, verify cache hit returns correct per-coordinate programs.

### Resolution Timeline

Before Phase 3 (FusedOp registration), since FusedOps are more likely to have non-SPMD behavior.

---

## 8.4.4 Open Question 3: Versioning and Cache Staleness

### Scenario

A developer modifies the matmul kernel source in `blaze/ops/matmul/kernels/op.hpp`. The change alters CT arg computation. The developer rebuilds TT-Blaze (Python) but does *not* rebuild TT-Metal (adapter is unchanged). The developer runs the model expecting the optimized kernel.

**Question:** Does the program cache serve a stale entry?

### Staleness Scenarios

| Scenario | Hash Changes? | Cache Behavior | Correctness |
|----------|:------------:|:-------------:|:-----------:|
| Kernel source text changes | Yes (descriptor includes kernel source path) | Miss (correct) | Correct |
| Kernel compile-time args change | Yes (CT args in descriptor) | Miss (correct) | Correct |
| CB layout changes | Yes (CB descriptors in descriptor) | Miss (correct) | Correct |
| Runtime arg semantics change (same structure) | No | Hit (incorrect!) | **Potentially incorrect** |
| `custom_program_hash` set but op semantics change | Depends on hash implementation | May hit | **Depends on hash quality** |

### Options

| Option | Mechanism | Correctness | Cost |
|--------|-----------|:-----------:|:----:|
| **A. Descriptor hash only** | Hash entire descriptor every dispatch | Covers scenarios A--C | $O(K + C + S)$ per dispatch |
| **B. Version stamp** | Add `version: uint64_t` to descriptor; increment on kernel change | All changes (if bumped correctly) | $O(1)$ but manual |
| **C. Kernel source hash** | Hash generated kernel C++ source, include in `custom_program_hash` | All kernel changes | Hashing overhead at compile time |
| **D. Combined (recommended)** | `custom_program_hash = hash(kernel_source_hash, user_args, tensor_shapes)` | Comprehensive | Computed once at Blaze compile time |

### Recommendation

**Two-pronged approach:**

1. **For development builds:** Set `BLAZE_DISABLE_CUSTOM_HASH=1` to force all ops to Tier 1 (descriptor hashing). Slow but guarantees correctness during kernel development.

2. **For production builds:** Require that `custom_program_hash` includes a kernel version token:

```python
# PROPOSED: In Blaze op compilation
kernel_version = hash(kernel_source_content)  # or CRC32
descriptor.custom_program_hash = hash((
    kernel_version,   # ensures cache invalidation on kernel change
    tuple(t.shape for t in inputs),
    tuple(user_args.items()),
))
```

Cost: ~10 ns per dispatch (CRC32 lookup, not recomputation).

### Resolution Timeline

Before Phase 2 (batch registration with custom hashes). Phase 1 ops can use Tier 1 (descriptor walk) exclusively.

---

## 8.4.5 Open Question 4: Thread Safety

### Scenario

A serving system imports `blaze_nn` in the main thread while a background thread also imports `blaze_nn` for a different model. Both threads call `_resolve_dispatch()` concurrently.

### Analysis

| Level | Concern | Status | Explanation |
|-------|---------|:------:|-------------|
| C++ `register_operation<>` | Compile-time constexpr | Thread-safe | Evaluated at compile time; no runtime mutation |
| Nanobind module init (`NB_MODULE`) | GIL-protected | Thread-safe | Python import lock serializes initialization |
| `_resolve_dispatch()` | `getattr(ttnn.blaze, op_name)` | Thread-safe | Protected by GIL in CPython |
| Program cache access | Device-level serialization | Thread-safe | TTNN's `ProgramCache` uses `std::unordered_map` with device serialization |

### Recommendation

**No action required for current CPython.** All paths are protected through compile-time evaluation, GIL, or device-level serialization.

For future GIL-free Python (PEP 703): wrap `_resolve_ttnn_callable()` in a `threading.Lock`. Cost: ~100 ns per first-access resolution, zero after caching.

```python
# PROPOSED: Thread-safe resolution for GIL-free Python
import threading
_resolve_lock = threading.Lock()

def _resolve_ttnn_callable_safe(blaze_op_name: str):
    with _resolve_lock:
        return _resolve_ttnn_callable(blaze_op_name)
```

### Resolution Timeline

Not blocking for any phase. Monitor PEP 703 adoption.

---

## 8.4.6 Open Question 5: Dynamic Registration

### Scenario

A researcher writes a custom Blaze MicroOp called `custom_attention`. The op is not in the main Blaze repository. The researcher wants it to appear as `ttnn.blaze.custom_attention` in the profiler without rebuilding TT-Metal.

### The Fundamental Blocker

```cpp
// register_operation<> requires a compile-time string literal:
constexpr auto op = register_operation<"ttnn::blaze::custom_attention", Adapter<Tag>>();
//                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                                     Must be a compile-time constant
```

### Options

| Option | Mechanism | Fidelity | Effort |
|--------|-----------|:--------:|:------:|
| **A: Pre-compiled slots** | Pre-compile N "spare" adapter slots; map runtime names | Partial (slot names in Tracy) | ~300 LoC |
| **B: Runtime registration API** | New `std::unordered_map<string, DynamicDispatcher>` parallel to `register_operation<>` | Limited (no `type_hash`, no friend injection) | ~1,500 LoC |
| **C: TTNN extension API** | Propose `ttnn.register_external_op()` as new framework feature | Full (with new infrastructure) | ~4,000 LoC |
| **D: Rebuild with codegen** | User runs `python generate_registrations.py && cmake --build .` | Full | ~200 LoC script |
| **E: Accept limitation** | Custom ops use `generic_op` with Phase 0 `op_name` | Partial (profiler annotation only) | Zero |

### Recommendation

**Phase 1--3: Option E (accept limitation).** Custom ops use `generic_op` with Phase 0's `op_name` field for profiler visibility. This covers ~95% of the use case.

**Phase 3+: Evaluate Option A (pre-allocated slots)** as an experimental feature if demand warrants. 32 slots cover realistic custom-op workloads.

**Long term: Advocate for Option C** as a TTNN framework improvement. If TTNN gains runtime registration, it benefits all external op authors (Blaze, moreh, third parties).

### Resolution Timeline

Not blocking for Phases 1--3. Evaluate after Phase 4 based on user feedback.

---

## 8.4.7 Open Question 6: Output Tensor Ownership Transition

### Scenario

The Blaze team wants a future `blaze.matmul` to let TTNN allocate the output tensor instead of pre-allocating it in Python, enabling TTNN's memory manager to optimize output placement.

### The Transition Path

```text
Current:  Blaze pre-allocates -> passes through adapter -> TTNN does not allocate
Future:   Blaze declares specs -> adapter computes specs -> TTNN allocates
```

The transition requires:

1. `compute_output_specs()` must compute shapes from op semantics (e.g., matmul: $(M, K) \times (K, N) \to (M, N)$), not from a pre-existing tensor.
2. Blaze's `FusedProgram` must stop pre-allocating output tensors.
3. Blaze's `_lifetime_tensors` pattern must be reconciled with TTNN-managed allocation.

### Recommendation

**Defer to post-Phase 4.** The current pre-allocation model works correctly with the adapter. The transition to TTNN-managed allocation is a separate project requiring changes to both Blaze and TTNN. If per-op override is needed in the future, add an optional `compute_output_specs` static method to `OpTag` (detected via SFINAE, as designed in [Section 4.2](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md)):

```cpp
struct MatmulTag {
    static constexpr const char* name = "blaze::matmul";

    // Optional: override compute_output_specs for TTNN-managed allocation
    static TensorSpec compute_output_specs(
        const BlazeAdapterAttributes& attrs,
        const BlazeAdapterTensorArgs& args) {
        auto [M, K1] = args.io_tensors[0].shape();
        auto [K2, N] = args.io_tensors[1].shape();
        return TensorSpec({M, N}, args.io_tensors[0].dtype(), ...);
    }
};
```

### Resolution Timeline

Post-Phase 4. Requires profiling data from production models to justify the investment.

---

## 8.4.8 Open Question 7: Performance Regression Risk

### Scenario

A production team runs autoregressive decoding of Llama-70B. Each step dispatches ~100--200 ops with an inter-op latency budget of ~5 microseconds. The team upgrades to Phase 1 and monitors per-op dispatch latency.

### Overhead Estimate

| Component | generic_op Path | Adapter Path | Delta |
|-----------|:-:|:-:|:-:|
| Topology imputation | ~100 ns | ~100 ns | 0 |
| Graph tracking | ~100 ns | ~150 ns (richer annotations) | +50 ns |
| Hash computation | ~800 ns (descriptor walk) | ~800 ns (Tier 1) or ~10 ns (Tier 2) | 0 to -790 ns |
| Inspector annotations | ~50 ns | ~80 ns (richer) | +30 ns |
| Tracy zones | ~20 ns | ~20 ns | 0 |
| **Total per dispatch** | **~1,070 ns** | **~1,150 ns (Tier 1)** or **~360 ns (Tier 2)** | **+80 ns** or **-710 ns** |

For Tier 1 (descriptor walk): +80 ns per dispatch. Over 200 ops per token: ~16 us. This is < 0.001% of a typical 20 ms token latency.

For Tier 2 (custom hash): **-710 ns per dispatch** because the adapter avoids the descriptor walk. Over 200 ops: **-142 us savings** -- a measurable improvement.

### The Autoregressive Decode Concern

In AR decode, single-token generation dispatches ~100--200 ops in <500 us of host overhead:

- **Tier 1:** 200 ops * 80 ns = 16 us additional. Fraction: 16/500 = **3.2%**. Marginal but noticeable.
- **Tier 2:** 200 ops * (-710 ns) = -142 us. Fraction: **-28% improvement**.

### Instruction Cache Pressure

With 112 `launch<>` template specializations at ~500--1,500 bytes each, the adapter adds ~55--165 KB of code. This fits in L2 but may not fit entirely in L1 icache (~32 KB typical). During AR decode hot loops where the same 10--30 ops are dispatched repeatedly, L1 icache pressure is bounded by the active working set, not the total registered set.

### Recommendation

1. **Phase 1:** Measure empirically on a representative LLM model. If overhead >5% on AR decode, prioritize Tier 2 hashing for top-10 ops.
2. **Phase 2+:** Ensure all MicroOps on the AR decode hot path use `custom_program_hash` (Tier 2). This makes the adapter path net-faster.
3. **Production mode:** Verify that Tracy and Inspector annotations compile to no-ops when profiling is disabled (`TRACY_ENABLE` not defined).

### Resolution Timeline

Phase 1. Empirical measurement is the first deliverable after the adapter is functional for the top-10 ops.

---

## 8.4.9 Resolution Roadmap

| Resolution Order | Question | Blocking Phase | Resolution | Effort |
|:-----:|----------|:-:|------------|:------:|
| 1 | Q1: Repository ownership | Phase 1 | Start in-tree (Option A); Blaze team owns directory | 0.5 weeks |
| 2 | Q3: Versioning / cache staleness | Phase 2 | Version stamp in `custom_program_hash`; env var for dev builds | 1 week |
| 3 | Q4: Thread safety | Phase 1 | No action (GIL); document GIL dependency; add lock for future | 0.5 days |
| 4 | Q7: Performance regression | Phase 1 | Empirical benchmark; Tier 2 for hot-path ops | 1 week |
| 5 | Q2: Non-SPMD mesh programs | Phase 3 | Runtime assertion + documentation + test case | 1 week |
| 6 | Q6: Output tensor ownership | Post-Phase 4 | Defer; current pre-allocation works | 4+ weeks (future) |
| 7 | Q5: Dynamic registration | Post-Phase 4 | Accept `generic_op` fallback; evaluate extension API later | 2+ weeks (future) |

**Total resolution effort for Phases 1--3 blocking questions:** ~3.5 engineering-weeks, overlapping with migration implementation.

### Cumulative Open Question Status by Phase

| Phase | Questions Resolved | Questions Remaining |
|:-----:|-------------------|-------------------|
| Before Phase 1 | Q1 (repository ownership) | Q2--Q7 |
| Phase 1 | Q3 (versioning), Q4 (thread safety), Q7 (performance benchmark) | Q2, Q5, Q6 |
| Phase 2--3 | Q2 (non-SPMD validation) | Q5, Q6 |
| Post-Phase 4 | Q5 (dynamic reg.), Q6 (output ownership) | None |

---

**End of guide.** Return to [Guide Index](../index.md)
