# Chapter 8: Testing, Debugging, and Common Pitfalls

## Overview

This chapter covers the full lifecycle of validating a new micro-op in TT-Blaze: writing tests at both the FusedProgram and graph-compiler levels, using Blaze's built-in debugging and profiling tools to diagnose hangs and correctness issues, and avoiding the most common mistakes that cause silent data corruption or device hangs.

## Section Map

| Section | File | Summary |
|---------|------|---------|
| 8.1 | [01_writing_tests.md](01_writing_tests.md) | Test infrastructure, FusedProgram vs graph-based test paths, device tensors, golden references, fixtures, and parametrization |
| 8.2 | [02_debugging_tools.md](02_debugging_tools.md) | `BLAZE_DEBUG_KERNELS`, `BLAZE_L1_PROFILE`, `BLAZE_DISABLE_TENSOR_CB_SHARE`, CT arg inspection, generated kernel inspection, visualizer, Tracy profiling, capture plugin |
| 8.3 | [03_common_hang_causes.md](03_common_hang_causes.md) | CB deadlocks, semaphore mismatches, phase ordering errors, NOC congestion, diagnosis workflow |
| 8.4 | [04_common_gotchas.md](04_common_gotchas.md) | CT arg name mismatches, missing flags, wrong RISC structs, CB page mismatches, scratch CB naming, and other frequent mistakes |

## Prerequisites

- Familiarity with Blaze graph construction (`blaze.fuse()`, `ExternalTensor`) and `FusedProgram` composition (`emit()`, `cb_from_tensor()`, `cb_scratch()`)
- Understanding of the CB lifecycle (push/pop semantics, tensor-backed vs scratch CBs)
- Access to a Tenstorrent device for silicon-level tests (unit/infra tests run CPU-only)

## Key Source File References

| File | Purpose |
|------|---------|
| `blaze/fused_program.py` | `FusedProgram` class: `build()`, `run()`, `cb_from_tensor()`, `cb_scratch()`, `cb_output()`, `BLAZE_DISABLE_TENSOR_CB_SHARE` escape hatch |
| `blaze/compiler.py` | `BlazeCompiler.compile()`, `CompiledProgram.run()`, `MeshCompiledProgram`, `BLAZE_L1_PROFILE` gate, `BLAZE_EXPORT` auto-export |
| `blaze/kernel_codegen.py` | `generate_kernel()`, `generate_kernel_file()`, `BLAZE_DEBUG_KERNELS` env var parsing, DPRINT RISC filtering via `_parse_debug_riscs()` |
| `blaze/l1_profile.py` | `print_cb_stats()`, `print_kernel_stats()`, CB address capture, overlap group analysis |
| `blaze/cb_engine.py` | `CBEngine.assign()` -- CB auto-assignment from graph, `CBAssignment` dataclass, `MAX_CB_ID = 64` |
| `blaze/cb_handle.py` | `CBHandle` dataclass, `CBAccessMode` enum (`FIFO` vs `DIRECT_ADDRESS`), `require_fifo_handle()` |
| `blaze/blaze_op.py` | `BlazeOp` base class, `CB_NAME_DELIMITER` (`___`), `CHILD_PREFIX_DELIMITER` (`__`), `register()`, `Risc` flag enum |
| `blaze/graph.py` | `compile_engines()`, `EngineResult` (with `.ct_tuples`), `BlazeGraph.validate()` |
| `blaze/visualizer.py` | `visualize()` -- interactive HTML graph visualization |
| `tests/blaze/conftest.py` | `_SILICON_PATHS` for CI gating, `collect_ignore` list |
| `tests/blaze/test_utils.py` | `verify_mesh_result()`, `comp_pcc()`, `tensor_bin_hash()` helpers |
| `tests/blaze/micro-ops/common/` | Canonical micro-op test examples (test_mcast.py, test_matmul.py, etc.) |
| `tests/blaze/fused_ops/` | Fused op test examples (test_dense_mlp.py, test_shared_expert.py, etc.) |
| `tests/blaze/infra/` | Infrastructure tests (test_cb_engine.py, test_ct_args.py, test_compiler.py, etc.) |
