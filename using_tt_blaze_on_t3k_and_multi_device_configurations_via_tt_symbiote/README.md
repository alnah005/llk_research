# Using TT-Blaze on T3K and Multi-Device Configurations via TT-Symbiote

A comprehensive guide to deploying models on Tenstorrent multi-device hardware using the TT-Symbiote (tensor parallelism) and TT-Blaze (pipeline parallelism) frameworks.

## Reading Paths

- **Symbiote-focused:** Ch1 -> Ch2 -> Ch3 -> Ch8
- **Blaze-focused:** Ch1 -> Ch4 -> Ch5 -> Ch6 (Part 2)
- **Integration-focused:** All chapters in order
- **"Just make it work on T3K":** Ch1 (skim) -> Ch8, back-reference Ch2-Ch3 as needed

## Table of Contents

### Chapter 1 -- Device Topologies and Mesh Initialization

- [01 Physical Topologies](ch01_device_topologies/01_physical_topologies.md)
- [02 Mesh Device Initialization](ch01_device_topologies/02_mesh_device_initialization.md)
- [03 Fabric and Routing](ch01_device_topologies/03_fabric_and_routing.md)

### Chapter 2 -- TT-Symbiote Core Architecture

- [01 Module Replacement Engine](ch02_symbiote_core/01_module_replacement_engine.md)
- [02 Dispatch and Tensor Wrapping](ch02_symbiote_core/02_dispatch_and_tensor_wrapping.md)
- [03 Run Modes](ch02_symbiote_core/03_run_modes.md)
- [04 End-to-End Model Flow](ch02_symbiote_core/04_end_to_end_model_flow.md)

### Chapter 3 -- TT-Symbiote Multi-Device Support

- [01 Distributed Config and Device Init](ch03_symbiote_multi_device/01_distributed_config_and_device_init.md)
- [02 Weight Sharding Strategies](ch03_symbiote_multi_device/02_weight_sharding_strategies.md)
- [03 CCL Operations](ch03_symbiote_multi_device/03_ccl_operations.md)
- [04 Distributed Modules](ch03_symbiote_multi_device/04_distributed_modules.md)

### Chapter 4 -- TT-Blaze Kernel Composition Framework

- [01 BlazeOp Architecture](ch04_blaze_kernel_composition/01_blaze_op_architecture.md)
- [02 CCL in Blaze](ch04_blaze_kernel_composition/02_ccl_in_blaze.md)
- [03 Ops Catalog Overview](ch04_blaze_kernel_composition/03_ops_catalog_overview.md)

### Chapter 5 -- TT-Blaze Pipeline System

- [01 Pipeline Graph and Layout](ch05_blaze_pipeline_system/01_pipeline_graph_and_layout.md)
- [02 Submesh Partition and Topology](ch05_blaze_pipeline_system/02_submesh_partition_and_topology.md)
- [03 Stage Kinds and Execution](ch05_blaze_pipeline_system/03_stage_kinds_and_execution.md)
- [04 Pipeline Manager (C++)](ch05_blaze_pipeline_system/04_pipeline_manager_cpp.md)
- [05 Hardware Configurations](ch05_blaze_pipeline_system/05_hardware_configurations.md)

### Chapter 6 -- Existing Multi-Device Model Implementations

- [01 Symbiote Tensor-Parallel Models](ch06_model_implementations/01_symbiote_tensor_parallel_models.md) (GLM-4.7, Gemma4, Ling, Qwen3.5)
- [02 Blaze DeepSeek V3 Pipeline](ch06_model_implementations/02_blaze_deepseek_v3_pipeline.md)
- [03 Reusable Patterns and Antipatterns](ch06_model_implementations/03_reusable_patterns_and_antipatterns.md)

### Chapter 7 -- Integration Boundaries and Parallelism Strategy Comparison

- [01 Framework Comparison](ch07_integration_and_comparison/01_framework_comparison.md)
- [02 Parallelism Strategies Compared](ch07_integration_and_comparison/02_parallelism_strategies_compared.md)
- [03 Integration Pathways](ch07_integration_and_comparison/03_integration_pathways.md)
- [04 Limitations, Gaps, and Roadmap](ch07_integration_and_comparison/04_limitations_gaps_and_roadmap.md)

### Chapter 8 -- Scaling a Single-Device Model to T3K and Beyond

- [01 Model Analysis and Strategy Selection](ch08_scaling_guide/01_model_analysis_and_strategy_selection.md)
- [02 Module Adaptation and Weight Distribution](ch08_scaling_guide/02_module_adaptation_and_weight_distribution.md)
- [03 Test Setup and Validation](ch08_scaling_guide/03_test_setup_and_validation.md)
- [04 Performance Optimization and Pipeline Escalation](ch08_scaling_guide/04_performance_optimization_and_pipeline_escalation.md)
