# Comprehensive Guide to Creating New Micro-Ops in TT-Blaze

A deep-dive reference for engineers building new micro-ops and fused operations in the TT-Blaze framework. Each chapter was produced using a deep-planning research workflow (N=5 generators, N evaluators, 1 synthesizer, 1 verifier, Agent B critic, Agent C compression analyst) and verified against the TT-Blaze source code.

---

## Chapters

### [Chapter 1: Architecture and Mental Model](ch01_architecture/index.md)
The Blaze stack from PyTorch to silicon: FusedOps, MicroOps, the compilation pipeline, Tensix core architecture, circular buffers as the universal data-flow primitive, and the five RISC processors.

### [Chapter 2: Anatomy of a Micro-Op](ch02_anatomy/index.md)
Filesystem layout for auto-discovery, the Python class (ports, `compose()`, `emit()`), the C++ kernel header, and how auto-derivation bridges the two halves.

### [Chapter 3: The `emit()` Method](ch03_emit/index.md)
The boundary where Python becomes hardware: CB allocation (`cb_from_tensor`, `cb_scratch`, `cb_context`), compile-time arg registration per RISC, output handle creation, and the complete `emit()` contract.

### [Chapter 4: Device Grids and Multi-Device Configurations](ch04_grids/index.md)
GridConfig, DeviceContext, core classification (compute, sender, phantom, DRAM worker), multi-device mesh compilation, and CCL ops (CclBroadcast, AllReduce).

### [Chapter 5: Kernel Codegen Pipeline](ch05_codegen/index.md)
The engine pipeline (CBEngine, SemEngine, RoleEngine, CTArgEngine), auto-generated kernel files with content-hashed atomic writes, named CT args and the generated header, and handwritten kernel integration.

### [Chapter 6: Inter-Core and Inter-Device Operations](ch06_inter_core/index.md)
The sender/receiver pattern, Mcast (one-to-many), Gather (many-to-one), Scatter (one-to-many), Copy (generic), and CCL fabric operations with semaphore protocols.

### [Chapter 7: Composing Micro-Ops into Fused Operations](ch07_composition/index.md)
FusedOp structure and the compose/emit duality, prefix namespacing (`child_prefix`, `cb_name`), shared resources (CBHandle propagation, semaphore dedup, CbReconfig), and a full walkthrough of SharedExpert's 9-op pipeline.

### [Chapter 8: Testing, Debugging, and Common Pitfalls](ch08_testing_debugging/index.md)
Writing tests at both FusedProgram and graph-compiler levels, debugging tools (L1 profiling, generated kernel inspection, debug RISC guards), common hang causes, and common gotchas.

### [Chapter 9: End-to-End Workflow and Performance](ch09_workflow_performance/index.md)
Complete walkthrough of adding a new micro-op from scratch (ScaledAdd example), plus performance optimization: CB allocation strategies, CbReconfig, scratch compaction, L1 profiling, and the CB limit budget.

---

## Source Code

All claims in this guide were verified against:
- **TT-Blaze**: `/localdev/salnahari/testing_dir/tt-blaze/blaze/`
- **Key files**: `blaze_op.py`, `fused_program.py`, `compiler.py`, `kernel_codegen.py`, `cb_engine.py`, `sem_engine.py`, `ct_args.py`, `cb_handle.py`, `unified_kernel_descriptor.py`, `l1_profile.py`, `cb_reconfig_builder.py`, `role_engine.py`

## Quality Assurance

Each chapter underwent:
1. **5 independent generator candidates** with distinct research approaches
2. **5 evaluator scores** (accuracy, completeness, clarity, code quality, pedagogical value)
3. **1 synthesizer** merging the top-scored candidates
4. **1 verifier** checking 15-25+ claims against source code (all chapters PASS)
5. **Agent B** independent critic reviewing for factual errors (all critical issues fixed)
6. **Agent C** compression analyst checking for redundancy
