# Chapter 9: End-to-End Workflow and Performance

This chapter brings together every concept from the preceding chapters into two practical sections. The first section walks through the complete process of adding a new micro-op to TT-Blaze from scratch, using a hypothetical "ScaledAdd" operation (`output = alpha * input + beta * bias`) as the running example. The second section covers performance optimization, L1 memory management, circular buffer strategies, and profiling.

## Organization

1. **[End-to-End Walkthrough](01_end_to_end_walkthrough.md)** -- A nine-step guide to implementing the ScaledAdd micro-op, covering directory layout, the C++ kernel header (`op.hpp`), the Python op class, the `emit()` method, `compose()`, registration, testing, and integration into a fused op.

2. **[Performance and L1 Management](02_performance_and_l1_management.md)** -- Deep coverage of CB allocation strategies (tensor-backed, scratch, aliased, direct-address), the `CbReconfig` multi-phase mechanism, scratch compaction, the `all_scratch_mapped` mode, L1 profiling with `BLAZE_L1_PROFILE`, and practical optimization patterns.

## Prerequisites

Readers should be familiar with:

- The `BlazeOp` / `MicroOp` / `FusedOp` class hierarchy (Chapter 3)
- Circular buffer fundamentals and `CBHandle` semantics (Chapter 4)
- The `FusedProgram` composition API (Chapter 5)
- CT arg declaration and the `cpp_parser` auto-derivation pipeline (Chapter 6)
- The `CBEngine` and `BlazeCompiler` compilation flow (Chapter 7)
- Kernel codegen and the shadow graph (Chapter 8)

## Key Source Files Referenced

| File | Role |
|------|------|
| `blaze/blaze_op.py` | `BlazeOp`, `MicroOp`, `FusedOp` base classes, `Input`/`Output`/`Internal` descriptors, `CompileTimeArg` |
| `blaze/fused_program.py` | `FusedProgram` context, `CBHandle`, `cb_scratch`, `cb_from_tensor`, `cb_alias`, `reconfig()`, `build()` |
| `blaze/cb_engine.py` | `CBEngine` -- graph-based CB ID assignment |
| `blaze/cb_handle.py` | `CBHandle` dataclass, `CBAccessMode` |
| `blaze/compiler.py` | `BlazeCompiler`, `CompiledProgram`, `MeshCompiledProgram` |
| `blaze/cb_reconfig_builder.py` | Multi-phase CB reconfig build pass, scratch compaction |
| `blaze/cpp_parser.py` | `parse_op_hpp` -- auto-derives CT args, lifecycle, flags from `op.hpp` |
| `blaze/l1_profile.py` | `print_cb_stats`, `print_kernel_stats` -- L1 profiling utilities |
| `blaze/ops/` | Reference micro-op implementations (`eltwise_mul`, `residual_add`, `copy`, `cb_reconfig`, etc.) |
