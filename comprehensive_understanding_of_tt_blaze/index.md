# Comprehensive Understanding of TT-Blaze

A deep technical guide to Tenstorrent's kernel composition framework for low-latency inference on Blackhole silicon.

---

## Chapter 1 -- Architecture and Design Philosophy

How Blaze sits in the TT-Metal stack, the repository layout, and the two-API design (MicroOp for kernel authors, FusedOp for model authors).

| Section | File |
|---------|------|
| 1.01 Stack Position | [ch01_architecture/01_stack_position.md](ch01_architecture/01_stack_position.md) |
| 1.02 Repository Map | [ch01_architecture/02_repository_map.md](ch01_architecture/02_repository_map.md) |
| 1.03 Two-API Design | [ch01_architecture/03_two_api_design.md](ch01_architecture/03_two_api_design.md) |

## Chapter 2 -- BlazeOp Hierarchy

The `BlazeOp` -> `MicroOp` -> `FusedOp` class hierarchy, the C++ header parser, auto-discovery and registration, and step-by-step guides for writing new ops.

| Section | File |
|---------|------|
| 2.01 BlazeOp Base | [ch02_blazeop_hierarchy/01_blazeop_base.md](ch02_blazeop_hierarchy/01_blazeop_base.md) |
| 2.02 MicroOp and FusedOp | [ch02_blazeop_hierarchy/02_microop_and_fusedop.md](ch02_blazeop_hierarchy/02_microop_and_fusedop.md) |
| 2.03 C++ Header Parser | [ch02_blazeop_hierarchy/03_cpp_parser.md](ch02_blazeop_hierarchy/03_cpp_parser.md) |
| 2.04 Auto-Discovery and Registration | [ch02_blazeop_hierarchy/04_auto_discovery_and_registration.md](ch02_blazeop_hierarchy/04_auto_discovery_and_registration.md) |
| 2.05 Writing a MicroOp | [ch02_blazeop_hierarchy/05_writing_a_micro_op.md](ch02_blazeop_hierarchy/05_writing_a_micro_op.md) |
| 2.06 Writing a FusedOp | [ch02_blazeop_hierarchy/06_writing_a_fused_op.md](ch02_blazeop_hierarchy/06_writing_a_fused_op.md) |

## Chapter 3 -- Compilation Pipeline

From `BlazeGraph` through CB/semaphore/CT-arg engines, kernel codegen, to the final `BlazeCompiler` orchestration.

| Section | File |
|---------|------|
| 3.01 BlazeGraph and FusionContext | [ch03_compilation_pipeline/01_blaze_graph_and_fusion_context.md](ch03_compilation_pipeline/01_blaze_graph_and_fusion_context.md) |
| 3.02 CB Engine | [ch03_compilation_pipeline/02_cb_engine.md](ch03_compilation_pipeline/02_cb_engine.md) |
| 3.03 Semaphore Engine | [ch03_compilation_pipeline/03_sem_engine.md](ch03_compilation_pipeline/03_sem_engine.md) |
| 3.04 CT Arg Engine | [ch03_compilation_pipeline/04_ct_arg_engine.md](ch03_compilation_pipeline/04_ct_arg_engine.md) |
| 3.05 Kernel Codegen | [ch03_compilation_pipeline/05_kernel_codegen.md](ch03_compilation_pipeline/05_kernel_codegen.md) |
| 3.06 BlazeCompiler | [ch03_compilation_pipeline/06_blaze_compiler.md](ch03_compilation_pipeline/06_blaze_compiler.md) |

## Chapter 4 -- Data Flow and L1 Management

CBHandle abstraction, FusedProgram state machine, OverlappedView for shared L1 views, and multi-phase CB reconfiguration.

| Section | File |
|---------|------|
| 4.01 CBHandle Abstraction | [ch04_data_flow/01_cbhandle_abstraction.md](ch04_data_flow/01_cbhandle_abstraction.md) |
| 4.02 FusedProgram | [ch04_data_flow/02_fused_program.md](ch04_data_flow/02_fused_program.md) |
| 4.03 OverlappedView and Shared L1 Views | [ch04_data_flow/03_overlapped_view.md](ch04_data_flow/03_overlapped_view.md) |
| 4.04 CB Reconfiguration | [ch04_data_flow/04_cb_reconfig.md](ch04_data_flow/04_cb_reconfig.md) |

## Chapter 5 -- RISC Compilation and Build System

Per-RISC processor model, kernel header conventions, and the JIT build system.

| Section | File |
|---------|------|
| 5.01 Per-RISC Model | [ch05_risc_compilation/01_per_risc_model.md](ch05_risc_compilation/01_per_risc_model.md) |
| 5.02 Kernel Headers | [ch05_risc_compilation/02_kernel_headers.md](ch05_risc_compilation/02_kernel_headers.md) |
| 5.03 Build System and JIT | [ch05_risc_compilation/03_build_system_and_jit.md](ch05_risc_compilation/03_build_system_and_jit.md) |

## Chapter 6 -- Op Library

Complete catalog: data movement ops, compute ops, attention (MLA) ops, and MoE ops.

| Section | File |
|---------|------|
| 6.01 Op Library Overview | [ch06_op_library/01_op_library_overview.md](ch06_op_library/01_op_library_overview.md) |
| 6.02 Data Movement Ops | [ch06_op_library/02_data_movement_ops.md](ch06_op_library/02_data_movement_ops.md) |
| 6.03 Compute Ops | [ch06_op_library/03_compute_ops.md](ch06_op_library/03_compute_ops.md) |
| 6.04 Attention Ops | [ch06_op_library/04_attention_ops.md](ch06_op_library/04_attention_ops.md) |
| 6.05 MoE Ops | [ch06_op_library/05_moe_ops.md](ch06_op_library/05_moe_ops.md) |

## Chapter 7 -- Composition

FusedOp composition basics, advanced patterns (CB aliasing, scratch arenas, mesh-aware composition), and two worked examples (SharedExpert, full MoE).

| Section | File |
|---------|------|
| 7.01 Composition Basics | [ch07_composition/01_composition_basics.md](ch07_composition/01_composition_basics.md) |
| 7.02 Advanced Patterns | [ch07_composition/02_advanced_patterns.md](ch07_composition/02_advanced_patterns.md) |
| 7.03 Worked Example: SharedExpert | [ch07_composition/03_worked_example_shared_expert.md](ch07_composition/03_worked_example_shared_expert.md) |
| 7.04 Worked Example: MoE | [ch07_composition/04_worked_example_moe.md](ch07_composition/04_worked_example_moe.md) |

## Chapter 8 -- Multi-Host Pipeline

Pipeline builder graph, PipelineManager C++20 token scheduler, scheduling algorithm (writer/reader loops, speculative decode), and KV cache migration.

| Section | File |
|---------|------|
| 8.01 Pipeline Builder | [ch08_multi_host_pipeline/01_pipeline_builder.md](ch08_multi_host_pipeline/01_pipeline_builder.md) |
| 8.02 Pipeline Manager Architecture | [ch08_multi_host_pipeline/02_pipeline_manager_architecture.md](ch08_multi_host_pipeline/02_pipeline_manager_architecture.md) |
| 8.03 Scheduling Algorithm | [ch08_multi_host_pipeline/03_scheduling_algorithm.md](ch08_multi_host_pipeline/03_scheduling_algorithm.md) |
| 8.04 KV Cache Migration | [ch08_multi_host_pipeline/04_kv_cache_migration.md](ch08_multi_host_pipeline/04_kv_cache_migration.md) |

---

## Appendix A -- Visualization, Weight Providers, and Developer Tools

| Section | File |
|---------|------|
| A.1 Interactive Visualizer | [appendix_a_tooling/01_visualizer.md](appendix_a_tooling/01_visualizer.md) |
| A.2 Weight Providers | [appendix_a_tooling/02_weight_provider.md](appendix_a_tooling/02_weight_provider.md) |
| A.3 Developer Tools | [appendix_a_tooling/03_developer_tools.md](appendix_a_tooling/03_developer_tools.md) |

## Appendix B -- Testing and Model Integrations

| Section | File |
|---------|------|
| B.1 Test Infrastructure | [appendix_b_testing_and_models/01_test_infrastructure.md](appendix_b_testing_and_models/01_test_infrastructure.md) |
| B.2 DeepSeek V3 Integration | [appendix_b_testing_and_models/02_deepseek_v3_integration.md](appendix_b_testing_and_models/02_deepseek_v3_integration.md) |
| B.3 GLM-5.1 Integration | [appendix_b_testing_and_models/03_glm51_integration.md](appendix_b_testing_and_models/03_glm51_integration.md) |
| B.4 Production Deployment | [appendix_b_testing_and_models/04_production_deployment.md](appendix_b_testing_and_models/04_production_deployment.md) |

---

## Quality Assurance

Every chapter and appendix was produced through a deep planning research flow:
- **N=6 generators** (N=3 for appendices) produced independent drafts from the source code
- **Evaluators** compared versions per-file and selected the best
- **Agent B (Correctness)** verified every code snippet, file path, class name, constant, and architectural claim against the actual `tt-blaze` source
- **Agent C (Compression)** identified cross-chapter duplication and replaced it with cross-references

All chapters passed both Agent B and Agent C review.
