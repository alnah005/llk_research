# Chapter 7: Connecting blaze-nn to Registered Operations

blaze-nn is a PyTorch-style neural network library that provides a functional API (`blaze_nn.F.*`) and a module system (`blaze_nn.Module`) built on top of TT-Blaze's compilation pipeline. Every `F.linear()`, `F.rmsnorm()`, and `F.sdpa()` call eventually compiles a Blaze op and dispatches it through `MeshCompiledProgram._run_program()`. Today, that dispatch always terminates at `ttnn.generic_op()`. Chapters 4--6 designed, instantiated, and verified a registration pathway that replaces `generic_op` with per-op `ttnn.blaze.*` callables for individual Blaze ops. This chapter asks: **what does that registration mean for blaze-nn?**

The answer has two parts. First, we must understand blaze-nn's current dispatch architecture in detail -- the `_REGISTRY`, `OpMapping`, `TensorProxy`, `Parameter`, tracing contexts, and the functional-to-Blaze-op translation layer -- to see precisely where `generic_op` enters the picture and why blaze-nn has no awareness of TTNN op registration. Second, we must design the rewired dispatch path, the `_registry.py` modifications, the fallback strategy, and the module-level integration that allows registered ops to flow through blaze-nn transparently, with profiler visibility and graph-trace identity preserved.

This chapter takes a **scenario-driven approach**. Rather than describing changes in isolation, each section traces a concrete operation -- a transformer attention block using blaze-nn's `F.linear()`, `F.rmsnorm()`, and `F.sdpa()` -- through both the current and proposed dispatch paths, showing exactly what changes at each layer.

---

## Chapter Contents

| # | File | Topic |
|---|------|-------|
| 1 | [Current blaze-nn Dispatch Architecture](./01_current_blaze_nn_dispatch.md) | blaze-nn's dispatch internals: `_REGISTRY` and `OpMapping` in `_registry.py`, `TensorProxy` lazy tensor type, `Parameter` wrapping, `GraphTracingContext` vs `ComposeTracingContext`, the `F.linear()` -> `_dispatch("matmul", ...)` -> tracing/compose path. Scenario walkthrough: a transformer attention block traced through the current dispatch, showing how every call terminates at `generic_op` with no TTNN-visible identity. |
| 2 | [Rewired Dispatch and Module Integration](./02_rewired_dispatch_and_module_integration.md) | New dispatch path: `OpMapping` gains a lazily-resolved `ttnn_callable` field with `__ttnn_operation__` guard, three-level fallback strategy for unregistered ops, `BLAZE_NN_FORCE_GENERIC` environment variable escape hatch, graph mode vs direct mode dispatch, `Module.forward()` integration, state dict and parameter management (unchanged), composability with native TTNN ops, diagnostic introspection API, end-to-end scenario walkthrough of the same attention block through the proposed path, comprehensive before/after comparison at every layer, and six-step migration sequence. |

---

## Key Takeaways

1. **blaze-nn already routes through a single dispatch choke point.** Every functional API call passes through `_dispatch()` in `_registry.py`, which resolves the Blaze op name and delegates to the active tracing context or compose-mode execution. The `generic_op` call lives at the bottom of this stack, inside `MeshCompiledProgram._run_program()`. This single point of control means that the registration changes from [Section 4.3](../ch04_adapter_pattern/03_python_dispatch_and_registration.md) propagate to all blaze-nn consumers without per-function modifications.

2. **The `_REGISTRY` is the natural place to store `ttnn_callable` references.** Each `OpMapping` already maps a functional name to a Blaze op name with grid and kwargs transforms. Adding a lazily-resolved `ttnn_callable` property that caches the `ttnn.blaze.*` function on first access gives dispatch a zero-overhead path to the registered operation, while avoiding import-order dependencies.

3. **Two dispatch modes, one registration benefit.** In graph mode, blaze-nn builds a `BlazeGraph` and compiles it; the registered op name flows through compilation into the `MeshCompiledProgram`, where `_resolve_dispatch()` selects the `ttnn.blaze.*` callable. In direct mode (compose), blaze-nn calls `blaze.{op}.compose()` immediately; the same `_resolve_dispatch()` mechanism routes to the registered callable. Both modes gain profiler visibility and cache isolation.

4. **Module integration is transparent.** `Module.forward()` calls functional API methods, which call `_dispatch()`, which reaches the registered callable. No changes to `Module`, `Parameter`, `state_dict()`, or `load_state_dict()` are required.

5. **The fallback strategy is layered.** If `ttnn.blaze` is unavailable, all ops fall back to `generic_op`. If a specific op is not registered, that op falls back individually. An environment variable (`BLAZE_NN_FORCE_GENERIC=1`) provides a global override for debugging during migration.

---

## Prerequisites

| Prerequisite | Why |
|---|---|
| [Chapter 2: How generic_op Works](../ch02_generic_op_dispatch/index.md) | The "before" baseline for all dispatch comparisons: `generic_op` internals, `MeshProgramDescriptor` data model, and the divergence analysis that motivates registration. |
| [Chapter 4, Section 4.3: Python Dispatch and Registration](../ch04_adapter_pattern/03_python_dispatch_and_registration.md) | The `_run_program()` modification, `_resolve_dispatch()`, and `ttnn.blaze` namespace design that this chapter builds upon. |
| [Chapter 5: Registering MicroOps and FusedOps](../ch05_op_registration/index.md) | Concrete registration walkthroughs for Matmul and GatedReduce; demonstrates the per-op artifacts that blaze-nn's dispatch will route to. |
| [Chapter 6: Program Caching, Graph Tracing, and Profiling Integration](../ch06_caching_tracing_profiling/index.md) | The infrastructure behavior changes (cache isolation, named graph nodes, Tracy zones) that become visible through blaze-nn after registration. |

---

## Cross-References

- **[Section 4.3.1](../ch04_adapter_pattern/03_python_dispatch_and_registration.md)** designed the `_run_program()` dispatch modification. Section 7.1 shows how that modification interacts with blaze-nn's `_dispatch()` -> tracing context -> compose -> `_run_program()` pipeline.
- **[Section 4.3.7](../ch04_adapter_pattern/03_python_dispatch_and_registration.md)** previewed blaze-nn integration in a single paragraph. This chapter expands that preview into a full architectural treatment with scenario walkthroughs.
- **[Section 5.1](../ch05_op_registration/01_microop_registration_walkthrough.md)** registered Matmul as a concrete MicroOp. Section 7.1 and 7.2 trace `F.linear()` (which internally dispatches to Matmul) through the full blaze-nn stack.
- **[Sections 6.1--6.3](../ch06_caching_tracing_profiling/index.md)** describe cache, graph, and profiling behavior for individual registered ops. Section 7.2 shows how these behaviors compose when multiple registered ops execute within a single `Module.forward()` call.
- **[Chapter 8](../ch08_build_migration_alternatives/index.md)** covers build system integration and long-term migration alternatives. The migration sequence in Section 7.2 corresponds to Phase 4 of the overall plan.

---

## Addresses Research Questions

| Question | Coverage |
|----------|---------|
| **Q11:** How does blaze-nn integrate with registered TTNN operations? | Section 7.1 maps the current dispatch architecture and identifies the `generic_op` dependency. Section 7.2 designs the rewired dispatch path, fallback strategy, module integration, and composability with native TTNN ops. Both sections use end-to-end scenario walkthroughs of a transformer attention block. |

---

## Source File Index

### blaze-nn Source Files (Existing)

| File | Role | Modified? |
|------|------|:---------:|
| `blaze_nn/_registry.py` | `_REGISTRY` dict, `OpMapping` dataclass, `register_op()`, `_dispatch()` | **Yes** |
| `blaze_nn/functional.py` | `F.linear()`, `F.rmsnorm()`, `F.sdpa()`, and other functional API entry points | No |
| `blaze_nn/module.py` | `Module` base class, `Parameter`, `forward()`, `state_dict()`, `load_state_dict()` | No |
| `blaze_nn/tracing.py` | `GraphTracingContext`, `ComposeTracingContext`, `TensorProxy` | **Yes** (signature only) |
| `blaze_nn/graph.py` | `BlazeNNGraph`, graph compilation | No |

### Blaze Source Files (Existing, Modified)

| File | Role | Modified? |
|------|------|:---------:|
| `blaze/compiler.py` | `BlazeCompiler`, `MeshCompiledProgram`, `_run_program()`, `_resolve_dispatch()` | **Yes** (from [Section 4.3.1](../ch04_adapter_pattern/03_python_dispatch_and_registration.md)) |

---

| [Chapter 6: Program Caching, Graph Tracing, and Profiling](../ch06_caching_tracing_profiling/index.md) | **Chapter 7** | [Chapter 8: Build System, Migration, and Alternatives](../ch08_build_migration_alternatives/index.md) |
|:---|:---:|---:|
