# Chapter 1: Architecture Overview and Model Mapping

This chapter establishes how the 671B-parameter DeepSeek V3 architecture maps to the TT-Blaze op library and Tenstorrent hardware, introduces the host-side model interface, and traces the execution model from Python to device.

## Reading Order

1. **[DeepSeek V3 Architecture on Tenstorrent](./01_deepseek_v3_architecture_on_tenstorrent.md)**
   The DeepSeek V3 model architecture (MLA, 256-expert MoE, 61 layers), the three-tier op hierarchy (micro-ops, fused ops, model layer), directory layout, batch-1 constraints, device grid partitioning, and multi-device topology.

2. **[TT-Blaze Execution Pattern](./02_tt_blaze_execution_pattern.md)**
   The five-step execution pipeline from CB descriptors to `ttnn.generic_op`, a step-by-step `Matmul.op()` walkthrough, the `DeepSeekV3` class interface and decode lifecycle, and how the pattern scales from simple micro-ops to complex fused operations.

3. **[Layer Taxonomy and Decode Overview](./03_layer_taxonomy_and_decode_overview.md)**
   The 61-layer structure (3 dense + 58 MoE), weight dataclasses and fusion groups, the weight preparation pipeline, pipeline stages, and a high-level end-to-end decode step walkthrough with timing characteristics.

## Key Source Files

| File | Role |
|---|---|
| [`model.py`](../../../models/demos/deepseek_v3_b1/model.py) | `DeepSeekV3` class: host-side interface |
| [`unified_kernel_descriptor.py`](../../../models/demos/deepseek_v3_b1/unified_kernel_descriptor.py) | `UnifiedKernelDescriptor` system |
| [`circular_buffer_utils.py`](../../../models/demos/deepseek_v3_b1/circular_buffer_utils.py) | CB management, reconfig tensors |
| [`blitz_decode_weights.py`](../../../models/demos/deepseek_v3_b1/blitz_decode_weights.py) | `OverlappedTensor`, weight fusion specs |
| [`prepare_weights.py`](../../../models/demos/deepseek_v3_b1/prepare_weights.py) | Weight dataclasses, HF-to-device conversion |
| [`demo/runner.py`](../../../models/demos/deepseek_v3_b1/demo/runner.py) | `ModelLike` protocol, `run_generation()` |
| [`demo/cli.py`](../../../models/demos/deepseek_v3_b1/demo/cli.py) | CLI entry point, pipeline stage dispatch |

## Questions Addressed

- **Question 1**: What is the DeepSeek V3 architecture and how does it map to Tenstorrent hardware?
- **Question 12 (overview)**: What does a complete decode step look like end-to-end?

## Navigation

- **Next chapter:** [Chapter 2 -- The Unified Kernel System](../ch02_the_unified_kernel_system/index.md)
