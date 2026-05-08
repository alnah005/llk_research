# 01 -- Framework Comparison: Symbiote vs Blaze

This chapter compares TT-Symbiote and TT-Blaze across every architectural dimension and synthesizes the integration boundaries, parallelism tradeoffs, and unresolved gaps that emerge only when the two frameworks are examined together. The goal is actionable guidance: which framework to choose for a given model and hardware scale, where the integration points and blockers lie, and what engineering tasks close the remaining gaps.

**Prerequisites:** [Ch2/01](../ch02_symbiote_core/01_module_replacement_engine.md) (TTNNModule lifecycle), [Ch2/02](../ch02_symbiote_core/02_dispatch_and_tensor_wrapping.md) (TorchTTNNTensor dispatch), [Ch3/01-03](../ch03_symbiote_multi_device/01_distributed_config_and_device_init.md) (Symbiote multi-device), [Ch4/01](../ch04_blaze_kernel_composition/01_blaze_op_architecture.md) (BlazeOp hierarchy, FusedProgram), [Ch5/01](../ch05_blaze_pipeline_system/01_pipeline_graph_and_layout.md) (PipelineGraph), [Ch6/01-03](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md) (model implementations).

---

## Hardware Context

Every design decision in both frameworks responds to the same chip constraints. The key resource limits:

| Resource | Wormhole (WH) | Blackhole (BH) | Design Impact |
|----------|---------------|----------------|---------------|
| DRAM per chip | ~12 GB | ~12 GB | Model weight capacity per device |
| L1 SRAM per core | 1.5 MB | 1.5 MB | Working set for activations + CBs |
| Compute cores | 8x8 (64) | Up to 13x10 (130) | Matmul parallelism per chip |
| Ethernet links (T3K) | 1 per hop | N/A | CCL bandwidth ceiling (~12.5 GB/s) |
| Ethernet links (Galaxy) | 4 vertical, 3 horizontal | 4 vertical, 3 horizontal | Asymmetric CCL bandwidth |
| CB limit per program | 32 | 64 | Fused kernel complexity ceiling |
| DRAM banks | 12 | 8 | Weight streaming bandwidth |

These constraints create a two-axis tension: models that fit in a single chip's DRAM avoid inter-device communication entirely, while models exceeding per-chip capacity must distribute weights and pay the ethernet bandwidth tax. Symbiote and Blaze resolve this tension in fundamentally different ways. The 64-CB Blackhole limit directly motivates Blaze's multi-phase CB reconfiguration (`FusedProgram.reconfig()` [fused_program.py:FusedProgram.reconfig, line 1876]); Symbiote never encounters this limit because each `ttnn.*` call manages its own CBs independently.

---

## Abstraction Level

The most fundamental difference is where each framework draws its boundary with user code.

```
                     User Code
                        |
  +-------- Symbiote ---|--- Blaze --------+
  |                     |                  |
  |  PyTorch model      |  BlazeOp class   |
  |  (nn.Module tree)   |  (emit/compose)  |
  |         |           |       |          |
  |  Module replacement |  FusedProgram    |
  |  (register_module_  |  (CB alloc,      |
  |   replacement_dict) |   CT args,       |
  |         |           |   semaphores)    |
  |  TorchTTNNTensor    |       |          |
  |  __torch_dispatch__ |  BlazeCompiler   |
  |         |           |  (graph -> MPD)  |
  |     ttnn.* ops      |       |          |
  |         |           |  ttnn.generic_op |
  |         v           |       v          |
  |     TT-Metal        |   TT-Metal      |
  +---------------------+------------------+
```

**Symbiote** operates at **PyTorch module granularity**. The user provides a mapping `{nn.Linear: TTNNLinearLLama}` and calls `register_module_replacement_dict()` [module_replacement.py:register_module_replacement_dict]. Each replaced module's `forward()` calls TTNN ops (`ttnn.linear`, `ttnn.reduce_scatter`). Between modules, PyTorch's `aten` operations on `TorchTTNNTensor` subclasses are intercepted by `__torch_dispatch__` and routed to TTNN handlers or CPU fallback [run_config.py:NormalRun.torch_dispatch].

**Blaze** operates at **per-core kernel composition**. The developer defines a `FusedOp` class with port descriptors and a `compose()` method that chains `MicroOp.emit()` calls [blaze_op.py:FusedOp, line 408]. Each `emit()` allocates circular buffers, sets compile-time arguments per RISC processor (NCRISC, BRISC, TRISC), and wires dataflow between ops. The result compiles into a single kernel dispatch via `ttnn.generic_op()` [compiler.py:_run_program, line 97-102].

| Dimension | Symbiote | Blaze |
|-----------|----------|-------|
| Granularity | nn.Module replacement | Per-core MicroOp composition |
| User interface | `from_torch()` + `forward()` | `emit()` + `compose()` |
| Op boundary | Each `ttnn.*` call is a separate dispatch | Entire FusedOp is a single dispatch |
| Control flow | PyTorch's module tree (HF `generate()` loop) | Explicit `PipelineGraph` + stage factories |
| Tensor type | `TorchTTNNTensor` (torch subclass) | Raw `ttnn.Tensor` with explicit CB backing |

---

## Dispatch Model

### Symbiote: Intercept-and-Route

Symbiote's dispatch has two layers, both documented in [Ch2/02](../ch02_symbiote_core/02_dispatch_and_tensor_wrapping.md):

1. **Module dispatch**: `TTNNModule.__call__()` delegates to `TENSOR_RUN_IMPLEMENTATION.module_run()` [module.py:TTNNModule.call, line 78-79], which handles input wrapping, weight preprocessing, and forward execution.

2. **Op dispatch**: `TorchTTNNTensor.__torch_dispatch__()` intercepts ~70 `aten` operations and routes them to TTNN handlers via `can_dispatch_to_ttnn()` [default_dispatcher.py:can_dispatch_to_ttnn]. Operations without handlers fall back to CPU via `dispatch_to_torch_wrapper()`.

This design means each TTNN op (matmul, reduce_scatter, all_gather) is dispatched individually to the device. For a single transformer layer in Gemma4, this produces approximately 9 separate CCL operations and dozens of individual TTNN op dispatches [Ch6/01, Section 6.3](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md).

### Blaze: Compile-Once, Execute-Fused

Blaze's dispatch is a single `ttnn.generic_op()` call per FusedOp [compiler.py:CompiledProgram.run, line 117-122]. The `BlazeCompiler.compile()` method [compiler.py:BlazeCompiler.compile, line 192] builds a `MeshProgramDescriptor` that encodes the entire fused computation. At execution time, a single dispatch sends the program to all devices in the mesh.

For a DeepSeek V3 MoE decoder layer, this means the entire sequence -- RMSNorm, attention (Q/K/V projections, RoPE, SDPA, O projection), residual add, RMSNorm, MoE gate, expert computation, shared expert, residual add -- executes as one fused dispatch [Ch6/02, Section 6.8](../ch06_model_implementations/02_blaze_deepseek_v3_pipeline.md).

### Concrete Evidence: Dispatch Count Per Decoder Layer

| Model | Framework | Dispatches/Layer | Source |
|-------|-----------|-----------------|--------|
| GLM-4.7 | Symbiote | ~7 (each linear is a dispatch; attention on CPU) | [Ch6/01, Section 6.2](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md) |
| Gemma4 | Symbiote | ~15 (matmuls + CCL + norms) | [Ch6/01, Section 6.3](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md) |
| Ling MoE | Symbiote | ~20+ (matmuls + CCL + MoE routing + all_to_all) | [Ch6/01, Section 6.4](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md) |
| DeepSeek V3 | Blaze | 1-3 (SparseLayer fused) | [Ch6/02, Section 6.8](../ch06_model_implementations/02_blaze_deepseek_v3_pipeline.md) |

> **Warning:** Symbiote's multiple dispatches each incur host-device synchronization overhead. On single-token decode where each matmul completes in microseconds, dispatch overhead can exceed compute time. Blaze's fused kernels eliminate this but require manual CB management and kernel composition -- a significantly higher development cost per model.

**Architectural consequence**: Symbiote pays dispatch overhead per TTNN op but gains flexibility (any TTNN op can be called from any `forward()`). Blaze amortizes dispatch to a single up-front cost but requires all computation to be expressed through the `emit()`/`compose()` pattern before execution begins.

---

## Compilation Model

### Symbiote: No Compilation Phase

Symbiote has no explicit compilation step. Module replacement is a Python-level swap (`model._modules[name] = new_module` [module_replacement.py:register_module_replacement_dict_with_module_names]). Weight preprocessing (`preprocess_weights_impl`) and device transfer (`move_weights_to_device_impl`) happen eagerly. At inference time, each `forward()` call directly invokes TTNN ops.

The closest analog to compilation is **trace capture** via `@trace_enabled`, which records a sequence of TTNN ops and replays them without re-dispatching. But this is optional and operates at the module boundary, not at the kernel level. Trace capture has significant constraints: no dynamic allocation, no `ttnn.all_reduce` (must decompose to `reduce_scatter + all_gather`), and fixed tensor shapes [Ch6/03, Antipattern 3](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md).

### Blaze: Graph-to-Kernel Compilation

Blaze has a full compilation pipeline [Ch4/01, Section 9](../ch04_blaze_kernel_composition/01_blaze_op_architecture.md):

```
Graph Construction  ->  Engine Pipeline  ->  Per-Device Compilation  ->  Mesh Assembly
(fuse() context)       (CB/Sem/CT args)      (FusedProgram.build())      (MeshProgramDescriptor)
```

Two compilation paths exist within `_compile_single()` [compiler.py:BlazeCompiler._compile_fused_op, line 905]:

1. **Fused Op Path**: Creates a `FusedProgram`, calls `compose()` which chains `MicroOp.emit()` calls, then calls `f.build()`. This is the common path for complex ops (MoE, MLA, DenseMLP).

2. **Engine Path**: Runs `CBEngine`, `SemEngine`, and `CTArgEngine` on the graph to auto-assign resources, then builds a `UnifiedKernelDescriptor`.

The compilation is mesh-aware: `BlazeCompiler` iterates over all devices in the mesh, merging `mesh_coord` and `per_device_overrides` into each device's compilation context [compiler.py:BlazeCompiler.compile, line 274-286].

---

## Memory Model

### Symbiote: TTNN-Managed Buffers

Symbiote delegates all memory management to TTNN. Tensors are created via `ttnn.from_torch()`, placed on device via `ttnn.to_device()`, and freed via `ttnn.deallocate()`. The `TorchTTNNTensor` wrapper maintains dual backing stores (`.elem` for CPU, `.ttnn_tensor` for device) with lazy conversion [Ch2/02, "TorchTTNNTensor"](../ch02_symbiote_core/02_dispatch_and_tensor_wrapping.md).

Weight memory is managed per-module through the `preprocess_weights_impl` / `move_weights_to_device_impl` lifecycle. The `@deallocate_weights_after` decorator frees weights after each forward call for memory-constrained scenarios [Ch2/01, "deallocate_weights()"](../ch02_symbiote_core/01_module_replacement_engine.md).

### Blaze: Explicit CB and L1 Management

Blaze manages memory at the circular buffer level. `FusedProgram` provides explicit allocation methods [fused_program.py:FusedProgram, line 391]:

- `cb_from_tensor(tensor)` -- tensor-backed CB
- `cb_scratch(name, ...)` -- intermediate scratch CB
- `cb_from_view(view)` -- sub-region of a shared physical buffer via `OverlappedView`

The `OverlappedView` mechanism [fused_program.py:OverlappedView, line 220] packs multiple weight matrices into a single L1 allocation with byte-offset views on disjoint cores. This level of control enables DeepSeek V3's MLA to pack Q/K/V/gate weights into one bfloat8 tensor [Ch4/01, Section 5.2](../ch04_blaze_kernel_composition/01_blaze_op_architecture.md).

CB lifetime tracking and interval coloring (`CBEngine.compact_cb_ids()` [cb_engine.py:CBEngine.compact_cb_ids]) keep the total CB count under the Blackhole hardware limit of 64. Multi-phase CB reconfiguration (`FusedProgram.reconfig()` [fused_program.py:FusedProgram.reconfig, line 1876]) enables ops with more than 64 logical CBs by reusing IDs across non-overlapping phases.

> **Warning:** The memory model gap is the primary barrier to integration. Symbiote's `TorchTTNNTensor` wraps `ttnn.Tensor` objects that live in TTNN-managed DRAM buffers. Blaze's `CBHandle` references L1-resident circular buffers with explicit page sizes and access modes (FIFO vs DIRECT_ADDRESS). Converting between these representations requires materializing the Blaze CB output back to a DRAM-resident `ttnn.Tensor` -- a round-trip that negates the fusion benefit.

---

## Multi-Device Model

### Symbiote: Mesh Mapper / Composer

Symbiote's multi-device model centers on `DistributedConfig` [run_config.py:DistributedConfig, line 64], which bundles a `DistributedTensorConfig` (mesh_mapper, mesh_composer, logical_shape_fn) and a `TT_CCL` manager. The activation gate is `device.get_num_devices() > 1` [Ch3/01](../ch03_symbiote_multi_device/01_distributed_config_and_device_init.md).

Every `TorchTTNNTensor` created during inference automatically receives the default `DistributedTensorConfig` via `get_default_distributed_tensor_config()`. CCL operations (`reduce_scatter`, `all_gather`) are called inline within module `forward()` methods using TTNN's CCL API with `topology=Ring` for trace compatibility [Ch3/02](../ch03_symbiote_multi_device/02_weight_sharding_strategies.md).

### Blaze: PipelineGraph + SubmeshPartition + Fabric CCL

Blaze's multi-device model has two tiers:

1. **Intra-submesh**: Within a stage's submesh (e.g., 4x2 = 8 chips), tensor parallelism is implemented via the Blaze CCL ops (AllReduce, Scatter). The `setup_fabric()` function [ccl.py:setup_fabric, line 18] allocates semaphores and computes fabric RT args for cross-chip communication within the submesh.

2. **Inter-submesh**: Between pipeline stages, data flows through D2D sockets managed by `PipelineBlock`. The `PipelineGraph` [pipeline_builder/graph.py:PipelineGraph, line 110] describes the logical stage DAG, and `build_topology()` auto-discovers physical chip connections via C++ control-plane APIs [Ch5/01](../ch05_blaze_pipeline_system/01_pipeline_graph_and_layout.md).

| Aspect | Symbiote | Blaze |
|--------|----------|-------|
| Multi-device abstraction | `DistributedConfig` + mesh mapper/composer | `PipelineGraph` + `SubmeshPartition` |
| CCL implementation | TTNN API (`ttnn.reduce_scatter`, `ttnn.all_gather`) | Blaze MicroOps (AllReduce, Scatter) + fabric setup |
| Cross-device data | Implicit (TTNN CCL on full mesh) | Explicit D2D sockets between submeshes |
| Topology discovery | `ttnn.open_mesh_device(mesh_shape)` | `build_topology()` + `resolve_graph_layout()` |
| Dispatch mode | Fast dispatch (default) | Slow dispatch required (`TT_METAL_SLOW_DISPATCH_MODE=1`) [model_pipeline.py:line 45-47] |

---

## The Tensor Interface Boundary

Both frameworks ultimately operate on `ttnn.Tensor` objects, but access them through incompatible abstractions:

```
Symbiote side:                         Blaze side:

TorchTTNNTensor                        CBHandle
  .elem (torch.Tensor, CPU)              .cb_id (int)
  .ttnn_tensor (ttnn.Tensor, device)     .num_pages, .page_size (int)
  .ttnn_distributed_tensor_config        .core_ranges (CoreRangeSet)
     .mesh_mapper / .mesh_composer       .backing_tensor (ttnn.Tensor|None)
     .logical_shape_fn                   .access_mode (FIFO|DIRECT_ADDRESS)
```

The shared type is `ttnn.Tensor`. A `TorchTTNNTensor` wraps one (`.ttnn_tensor`); a `CBHandle` may reference one (`.backing_tensor`). This shared substrate is the natural integration point for the pathways explored in [File 03](./03_integration_pathways.md).

> **Warning:** Passing a `TorchTTNNTensor` to a Blaze `FusedProgram.cb_from_tensor()` would fail -- it expects a raw `ttnn.Tensor` with a materialized device buffer and a valid `buffer_address()`. The `.ttnn_tensor` property can extract the raw tensor, but the distributed tensor config (mesh_mapper/mesh_composer) is lost, and the logical shape will mismatch the physical per-device shard. Bridging these worlds requires explicit conversion at every boundary.

---

## Strengths of Each Framework

### Symbiote Strengths

1. **Rapid model porting**: A new model can be running on device in hours by mapping 2-3 PyTorch classes to TTNN equivalents. GLM-4.7 requires only `{nn.Linear: TTNNLinearLLama, nn.SiLU: TTNNSilu}` [Ch6/01, Section 6.2](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md).

2. **HuggingFace compatibility**: The model's control flow, tokenizer, generation loop, and KV cache management all stay in PyTorch. `model.generate()` works unmodified after module replacement.

3. **Incremental acceleration**: Each module can be independently accelerated or left on CPU. The `exclude_replacement` set enables selective offloading [Ch2/01, "The exclude_replacement Set"](../ch02_symbiote_core/01_module_replacement_engine.md).

4. **Progressive validation**: The run mode progression (CPU -> NORMAL_WITH_FALLBACK -> SEL -> NORMAL -> TRACED) lets developers incrementally optimize and validate numerics without rewriting the model [Ch2/03](../ch02_symbiote_core/03_run_modes.md).

5. **Trace-compatible CCL**: The `reduce_scatter + all_gather` decomposition pattern and `Ring` topology enable multi-device modules to work under trace capture [Ch3/02](../ch03_symbiote_multi_device/02_weight_sharding_strategies.md).

### Blaze Strengths

1. **Kernel fusion**: An entire decoder layer (40+ tensor arguments, multiple MicroOps) compiles into a single dispatch. This eliminates inter-op synchronization and L1-to-DRAM round-trips between ops.

2. **Explicit L1 control**: `OverlappedView`, disjoint-core CB sharing, and multi-phase CB reconfig give the developer full control over L1 utilization. DeepSeek V3 packs all MLA weights into overlapped views to maximize L1 residency [Ch4/01, Section 5.2](../ch04_blaze_kernel_composition/01_blaze_op_architecture.md).

3. **Persistent mode execution**: Decoder stages run persistently on device, eliminating per-token dispatch overhead. A `persistent_next_iter_semaphore` gates iteration boundaries [Ch6/02, Section 6.12](../ch06_model_implementations/02_blaze_deepseek_v3_pipeline.md).

4. **Pipeline parallelism**: The `PipelineGraph` + D2D socket system enables scaling beyond a single mesh. DeepSeek V3 uses 64 stages across 4 galaxies (128 chips), with each submesh running assigned layers [Ch5/01](../ch05_blaze_pipeline_system/01_pipeline_graph_and_layout.md).

5. **Auto-derived kernel metadata**: MicroOps auto-derive CT args, phase info, and RISC annotations from C++ kernel headers, reducing the Python boilerplate to `name`, `emit()`, and ports [Ch4/01, Section 1.3](../ch04_blaze_kernel_composition/01_blaze_op_architecture.md).

### What Neither Framework Can Do

Neither framework supports data parallelism (independent batch replicas), sequence parallelism (sharding along the sequence dimension), or dynamic batch scheduling within a single forward pass. These gaps are analyzed in [File 04](./04_limitations_gaps_and_roadmap.md).

---

## Programming Model Summary

```
Symbiote developer workflow:          Blaze developer workflow:
================================      ================================
1. Define nn_to_ttnn mapping          1. Define MicroOp classes (C++ kernel + Python ports)
2. register_module_replacement_dict   2. Define FusedOp.compose() chaining MicroOp.emit()
3. set_device(model, mesh_device)     3. Build PipelineGraph with Node/Edge/factory
4. preprocess_weights + to_device     4. BlazeCompiler.compile(graph, tensors, ...)
5. model.generate()                   5. result.run() or pipeline.setup_and_run()

Iteration speed: hours                Iteration speed: days-weeks
Fusion granularity: none (per-op)     Fusion granularity: entire decoder layer
Hardware control: minimal             Hardware control: per-core, per-CB, per-RISC
```

---

## When to Choose Which

**Choose Symbiote when:** rapid prototyping is the priority; the model fits within a single T3K or TG mesh (up to ~35B on T3K, ~100B on TG with 32-way TP); HuggingFace integration is required; the team's expertise is in PyTorch; or the primary goal is numerical validation before production optimization.

**Choose Blaze when:** maximum decode throughput is the objective; the model exceeds single-mesh memory (>100B parameters requiring multi-galaxy PP); continuous batching with concurrent users is required (`PipelineManager` supports 64 slots); or the team can invest in custom kernel development for production deployment.

**The 35B-100B gap**: Models in this range are too large for Symbiote's TP-only approach on T3K but too small to justify the weeks of Blaze kernel development. This is the critical zone where integration pathways ([File 03](./03_integration_pathways.md)) and infrastructure improvements ([File 04](./04_limitations_gaps_and_roadmap.md)) provide the most value.

The fundamental insight is that Symbiote and Blaze are not competitors -- they occupy different points on the abstraction-performance tradeoff curve. Symbiote optimizes for developer velocity and model coverage; Blaze optimizes for execution efficiency and hardware utilization. The next files explore how this complementarity can be leveraged through concrete parallelism strategies and integration pathways.

---

| Previous | Up | Next |
|----------|-----|------|
| [Ch6: Model Implementations](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md) | [Table of Contents](../README.md) | [02_parallelism_strategies_compared.md](./02_parallelism_strategies_compared.md) |
