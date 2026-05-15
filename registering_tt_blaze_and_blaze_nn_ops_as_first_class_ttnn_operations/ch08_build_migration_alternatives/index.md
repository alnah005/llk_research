# Chapter 8: Build System, Migration Path, and Open Questions

The preceding seven chapters answered the technical "how" -- what the TTNN registration contract requires ([Chapter 1](../ch01_ttnn_registration_contract/index.md)), what `generic_op` provides and lacks ([Chapter 2](../ch02_generic_op_dispatch/index.md)), why the gap is structural ([Chapter 3](../ch03_compilation_model_gap/index.md)), how a parameterized adapter bridges it ([Chapter 4](../ch04_adapter_pattern/index.md)), what concrete registration looks like for MicroOps and FusedOps ([Chapter 5](../ch05_op_registration/index.md)), how caching/tracing/profiling infrastructure activates ([Chapter 6](../ch06_caching_tracing_profiling/index.md)), and how blaze-nn reconnects through registered callables ([Chapter 7](../ch07_blaze_nn_integration/index.md)). This chapter answers the engineering "when, where, and what else" -- the build system changes needed to ship the adapter, a phased migration plan from zero-change profiler visibility to full blaze-nn integration, existing patterns that validate or challenge the design, and seven open questions that require data-driven resolution.

---

## Chapter Contents

| # | File | Topic |
|---|------|-------|
| 1 | [`01_build_system_integration.md`](./01_build_system_integration.md) | Current build model (TT-Metal CMake + Blaze Python), four integration options (A: in-tree, B: out-of-tree plugin, C: JIT registration, D: hybrid), CMake integration details, 5-dimension weighted decision matrix, build dependency analysis proving no circular dependency, build-time cost projections and mitigation strategies. |
| 2 | [`02_migration_plan.md`](./02_migration_plan.md) | Phase 0 (op_name in MeshProgramDescriptor, ~35 LoC), Phase 1 (adapter template + top 10 ops, ~810 LoC), Phase 2 (batch MicroOp registration, ~410 LoC), Phase 3 (FusedOp registration, ~400 LoC), Phase 4 (blaze-nn dispatch rewiring, ~320 LoC). Per-phase entry/exit criteria, risk assessment, testing strategy, backward compatibility guarantees. Cumulative: ~2,000 LoC / 7--10 engineering-weeks. |
| 3 | [`03_existing_patterns_and_precedents.md`](./03_existing_patterns_and_precedents.md) | Five ecosystem precedents (`experimental/` ops, moreh ops, CCL operations, `model_tracer/generic_ops_tracer.py`, `sweep_framework`), four alternative strategies (enhanced generic_op, Python wrappers, plugin system, full C++ port), 7-dimension comparison table, cross-ecosystem patterns (PyTorch `torch.library`, XLA custom calls). |
| 4 | [`04_open_questions.md`](./04_open_questions.md) | Seven unresolved questions with priority matrix (urgency x difficulty): repository ownership, non-SPMD mesh programs, versioning/cache staleness, thread safety, dynamic registration, output tensor ownership, performance regression risk. Each with evaluation criteria and resolution timeline. |

---

## Key Takeaways

1. **Build integration is straightforward.** The adapter requires ~625 lines of new C++ with no TT-Blaze build dependency. Options A (in-tree) and B (out-of-tree) score nearly identically (4.05 weighted); the choice is organizational, not technical. Build-time impact is ~35--50 seconds for 112 ops, reducible to ~8--12 seconds with sharded compilation.

2. **The migration plan delivers value at every phase.** Phase 0 (profiler visibility) is achievable in 1--2 days with ~35 lines of code. The complete migration through Phase 4 requires ~2,000 total LoC and 7--10 engineering-weeks -- approximately 4% of the effort of the "deep integration" alternative (~53,000 LoC, 40--60 weeks).

3. **Ecosystem precedents validate the approach.** CCL operations prove mesh-aware factories work; moreh ops prove external registration works (but at impractical per-op cost); `experimental/` ops prove namespace-based staging works; the `model_tracer` confirms the identity problem exists. The adapter pattern scores highest (4.14/5.0) in the five-strategy comparison.

4. **Seven open questions require data-driven resolution, not design debate.** Each question has explicit evaluation criteria (release cadence metrics, empirical benchmarks, op prevalence counts) that determine the answer. Total resolution effort for Phase 1--3 blocking questions: ~3.5 engineering-weeks, overlapping with implementation.

5. **Performance regression is likely net-positive.** The adapter adds ~80 ns per dispatch for Tier 1 hashing (negligible at < 0.001% of token latency) and *saves* ~710 ns per dispatch for Tier 2 hashing. Empirical measurement in Phase 1 will confirm.

---

## Prerequisites

| Prerequisite | Why |
|---|---|
| [Chapter 1: The TTNN Op Registration Contract](../ch01_ttnn_registration_contract/index.md) | Build integration requires understanding of `register_operation<>`, `bind_registered_operation()`, and the compile-time registration model. |
| [Chapter 3: The Compilation Model Gap](../ch03_compilation_model_gap/index.md) | The migration plan addresses the translation challenges enumerated there. The effort estimates reference the "100+ ops problem" quantification. |
| [Chapter 4: The Adapter Pattern](../ch04_adapter_pattern/index.md) | The build system integrates the adapter template from [Section 4.2](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md). The migration phases instantiate and scale it. |
| [Chapter 5: Registering MicroOps and FusedOps](../ch05_op_registration/index.md) | Batch registration ([Section 5.3](../ch05_op_registration/03_batch_registration_and_scaling.md)) feeds directly into Phase 2--3 of the migration plan. |
| [Chapter 6: Program Caching, Graph Tracing, and Profiling](../ch06_caching_tracing_profiling/index.md) | The infrastructure benefits quantified there motivate the migration timeline and are cited in the performance regression analysis. |
| [Chapter 7: blaze-nn Integration](../ch07_blaze_nn_integration/index.md) | Phase 4 of the migration implements the dispatch rewiring designed in [Section 7.2](../ch07_blaze_nn_integration/02_rewired_dispatch_and_module_integration.md). |

---

## Cross-References and Research Questions

- **[Section 3.2](../ch03_compilation_model_gap/02_translation_challenges.md)** quantified the "100+ ops problem" (53,000 LoC for deep integration). Section 8.2 shows the adapter migration achieves 95% of the benefit at 4% of the effort. Addresses **Q13** (practical phased migration plan: five phases with per-phase effort, risk, testing, backward compatibility).
- **[Section 4.2](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md)** defined `BlazeDeviceOperationAdapter<OpTag>`. Section 8.1 specifies where it lives in the source tree and how it builds. Addresses **Q12** (build system changes: four build options with CMake integration, dependency analysis, build-time projections, decision matrix).
- **[Section 5.3](../ch05_op_registration/03_batch_registration_and_scaling.md)** designed X-macro and codegen approaches for batch registration. Section 8.2 (Phase 2) schedules their deployment.
- **[Section 6.1](../ch06_caching_tracing_profiling/01_program_cache_integration.md)** analyzed hash cost models. Section 8.4 uses those models to assess performance regression risk.
- **[Section 7.2](../ch07_blaze_nn_integration/02_rewired_dispatch_and_module_integration.md)** designed blaze-nn dispatch rewiring. Section 8.2 (Phase 4) schedules its deployment with entry/exit criteria. Addresses **Q14** (alternative approaches: five ecosystem precedents, four alternative strategies, 7-dimension comparison in Section 8.3).

---

## Source File Index

### TT-Metal (Existing, Build-Relevant)

| File | Role |
|------|------|
| `ttnn/CMakeLists.txt` | Top-level TTNN build; would gain `add_subdirectory(operations/blaze)` (Option A) |
| `ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.hpp` | `GenericOpDeviceOperation` -- adapter's dependency for factory types |
| `ttnn/cpp/ttnn/operations/generic/device/generic_op_program_factory.hpp` | `GenericMeshProgramFactory` -- adapter's sole factory |
| `tt_metal/api/tt-metalium/experimental/mesh_program_descriptor.hpp` | `MeshProgramDescriptor` -- modified in Phase 0 (op_name field) |
| `ttnn/api/ttnn/decorators.hpp` | `register_operation<>` -- adapter's registration mechanism |
| `ttnn/cpp/ttnn-nanobind/decorators.hpp` | `bind_registered_operation()` -- adapter's Python binding mechanism |

### Proposed New Files (Option A)

| File | Role | Phase |
|------|------|:-----:|
| `ttnn/cpp/ttnn/operations/blaze/blaze_device_operation_adapter.hpp` | Adapter template | 1 |
| `ttnn/cpp/ttnn/operations/blaze/blaze_op_list.hpp` | X-macro op list | 1 |
| `ttnn/cpp/ttnn/operations/blaze/blaze_op_registrations.hpp` | Tag structs + registrations | 1 |
| `ttnn/cpp/ttnn/operations/blaze/blaze_ops_nanobind.cpp` (sharded) | Nanobind bindings | 1--2 |
| `ttnn/cpp/ttnn/operations/blaze/CMakeLists.txt` | Build integration | 1 |

### TT-Blaze (Existing, Modified)

| File | Role | Phase |
|------|------|:-----:|
| `blaze/compiler.py` | `_run_program()` + `_resolve_dispatch()` | 0--1 |
| `blaze_nn/_registry.py` | `OpMapping.ttnn_callable` lazy resolution | 4 |

### Precedent Files (Referenced)

| File | Precedent |
|------|-----------|
| `ttnn/cpp/ttnn/operations/experimental/` | Experimental op namespace pattern |
| `ttnn/cpp/ttnn/operations/moreh/` | External team op registration |
| `ttnn/cpp/ttnn/operations/ccl/` | Mesh-aware factory pattern |
| `model_tracer/generic_ops_tracer.py` | Python-level identity extraction workaround |
| `sweep_framework/load_ttnn_ops_data_v2.py` | Registered op auto-discovery |

---

| [Chapter 7: blaze-nn Integration](../ch07_blaze_nn_integration/index.md) | **Chapter 8** | *End of Guide* |
|:---|:---:|---:|
