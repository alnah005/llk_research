# Chapter 3: The Compilation Model Gap

Chapter 2 documented *what* diverges between Blaze-via-`generic_op` and native TTNN operations -- fifteen dimensions of mismatch, classified by severity and priority. This chapter asks the deeper question: *why* do those divergences exist, and what makes them structurally difficult to resolve? The answer lies in the fundamentally different compilation models that the two systems embody. TTNN operations are C++ structs whose program factories build `Program` objects inside the framework's dispatch pipeline; Blaze operations are Python classes whose `emit()`/`compose()` methods build `ProgramDescriptor` objects outside that pipeline. The gap between these two models is not a matter of missing glue code -- it reflects different answers to the same engineering questions (where to size circular buffers, how to assign semaphores, whether kernel source is static or generated, whether program state is accumulated or computed fresh). This chapter maps those answers side by side, identifies seven concrete translation challenges that any bridge must solve, and quantifies the "100+ ops problem" that rules out naive per-op porting. Together, these analyses motivate the adapter pattern proposed in [Chapter 4](../ch04_adapter_pattern/index.md).

---

## Chapter Contents

| # | File | Topic |
|---|------|-------|
| 1 | [Blaze vs TTNN Lifecycles](01_blaze_vs_ttnn_lifecycles.md) | Side-by-side lifecycle comparison: where each system makes CB allocation, semaphore assignment, CT arg, kernel source, and program construction decisions. Decision locus table with labeled decision points (A--H), side-by-side pipeline diagram, temporal inversion analysis, and cache cost asymmetry. |
| 2 | [Translation Challenges](02_translation_challenges.md) | Seven concrete challenges preventing naive registration (language boundary, stateful vs stateless, output tensor management, multi-phase programs, dynamic kernel sources, per-core specialization, engine pipeline), each with failure-mode code examples. The "100+ ops problem" with cost estimates, a Blaze-to-TTNN concept mapping table with "Bridgeable?" annotations, and a challenge-to-adapter-requirement summary. |

---

## Key Takeaways

1. **The gap is architectural, not incidental.** Blaze and TTNN make the same categories of decisions (CB allocation, semaphore assignment, kernel selection, output allocation) but at fundamentally different points in the pipeline, in different languages, and with different statefulness assumptions.

2. **Seven translation challenges** block naive registration. Each arises from a concrete architectural mismatch: factories must be stateless C++ structs, output tensors must be allocated by the op, kernel sources must be static, and per-core variation must use runtime args. No single fix addresses all seven.

3. **The "100+ ops problem" makes per-op C++ porting infeasible.** Writing a dedicated `DeviceOperation` struct for each of Blaze's 112+ MicroOps and FusedOps would require approximately 53,000 lines of C++ boilerplate at midpoint estimates (range: 32,000--73,500) with ongoing dual-maintenance burden. The adapter pattern reduces per-op cost to ~5 lines.

4. **The challenges converge on a single design requirement:** a parameterized C++ template that accepts a pre-built `MeshProgramDescriptor` from Python, satisfies `DeviceOperationConcept` structurally, and injects op-specific identity without requiring per-op factory code. This is the `BlazeDeviceOperationAdapter<OpTag>` pattern developed in [Chapter 4](../ch04_adapter_pattern/index.md).

---

## Prerequisites

| Prerequisite | Why |
|---|---|
| [Chapter 1: The TTNN Op Registration Contract](../ch01_ttnn_registration_contract/index.md) | The TTNN lifecycle described here relies on `DeviceOperationConcept`, `register_operation<>`, `launch<>`, and the program cache / tracing / profiling hooks. |
| [Chapter 2: How generic_op Works](../ch02_generic_op_dispatch/index.md) | The Blaze lifecycle recapped here was documented in Sections 2.1--2.2. The divergence analysis in Section 2.3 provides the starting point for this chapter's translation challenges. |

---

## Cross-References

- **[Section 2.3](../ch02_generic_op_dispatch/03_divergence_analysis.md)** cataloged *what* is different. This chapter explains *why* those differences make registration non-trivial.
- **[Chapter 4](../ch04_adapter_pattern/index.md)** proposes the adapter design that resolves the translation challenges enumerated here.
- **[Section 1.2](../ch01_ttnn_registration_contract/02_registration_dispatch_and_bindings.md)** provides the detailed TTNN dispatch mechanics contrasted with Blaze's dispatch path in Section 3.1.

---

| [Chapter 2: How generic_op Works](../ch02_generic_op_dispatch/index.md) | **Chapter 3** | [Chapter 4: The Adapter Pattern](../ch04_adapter_pattern/index.md) |
|:---|:---:|---:|
