# 03 -- Integration Pathways

This file identifies three concrete pathways for integrating TT-Symbiote and TT-Blaze, analyzes the tensor interface boundary that each must cross, catalogs the current blockers, and provides engineering task breakdowns with effort estimates. The pathways emerge from the architectural differences established in [File 01](./01_framework_comparison.md): the dispatch model gap, the memory model gap, and the multi-device model gap.

**Prerequisites:** [File 01](./01_framework_comparison.md) (architectural comparison), [Ch2/01](../ch02_symbiote_core/01_module_replacement_engine.md) (TTNNModule lifecycle), [Ch4/01](../ch04_blaze_kernel_composition/01_blaze_op_architecture.md) (BlazeOp/FusedProgram), [Ch5/03](../ch05_blaze_pipeline_system/03_stage_kinds_and_execution.md) (StageKind).

---

## Why Integration Matters

A developer porting a 35B MoE model on T3K starts naturally with Symbiote -- the PyTorch integration gives fast iteration. But as they optimize, they hit performance walls: per-op dispatch overhead in attention blocks, inefficient MoE routing with separate `all_to_all` dispatches, and the lack of continuous batching. Blaze solves these but requires rewriting the entire model. The integration question: can the developer keep Symbiote's rapid prototyping for most of the model while delegating performance-critical blocks to Blaze?

---

## The Tensor Interface Boundary

Both frameworks operate on `ttnn.Tensor` objects. A `TorchTTNNTensor` wraps one (`.ttnn_tensor`) [Ch2/02](../ch02_symbiote_core/02_dispatch_and_tensor_wrapping.md), and a Blaze `FusedProgram` operates on raw `ttnn.Tensor` objects passed to `BlazeCompiler.compile()` [Ch4/01](../ch04_blaze_kernel_composition/01_blaze_op_architecture.md). The conversion is:

```python
# Symbiote -> Blaze: extract raw ttnn.Tensor
activation_ttnn = symbiote_output.ttnn_tensor
# (or, if _bypass_tensor_wrapping=True, forward() already receives raw ttnn.Tensor)

# Blaze -> Symbiote: wrap into TorchTTNNTensor
symbiote_input = TorchTTNNTensor(blaze_output)
# Must manually set DistributedTensorConfig for correct shape reporting and CCL
```

> **Warning:** The `DistributedTensorConfig` on `TorchTTNNTensor` controls how `ttnn.to_torch()` reassembles multi-device tensors. When creating a `TorchTTNNTensor` from a Blaze output, the developer must manually set the correct `mesh_mapper` and `mesh_composer` -- the automatic config from `get_default_distributed_tensor_config()` assumes Symbiote's default sharding strategy, which may not match Blaze's tensor layout [Ch3/01](../ch03_symbiote_multi_device/01_distributed_config_and_device_init.md).

---

## Pathway A: FusedOp Delegation (Blaze Kernels Inside Symbiote Modules)

**Concept**: A `TTNNModule.forward()` calls a pre-compiled Blaze `FusedOp` instead of issuing individual `ttnn.*` calls. The Symbiote module replacement infrastructure stays intact; only the per-module compute changes to Blaze.

### Architecture

```
  Symbiote Model Tree
  |
  +-- TTNNModule (RMSNorm)           -- Symbiote dispatch
  +-- BlazeBackedDecoderLayer        -- DELEGATED to Blaze FusedOp
  |     |
  |     +-- BlazeCompiler.compile()   (cached after first call)
  |     +-- compiled_program.run()    (single ttnn.generic_op dispatch)
  |     +-- Return raw ttnn.Tensor
  |
  +-- TTNNModule (MLP)               -- Symbiote dispatch
```

### Concrete Example: Gemma4 Attention via FusedOp

To fuse Gemma4's attention block into a single Blaze dispatch [Ch6/01, Section 6.3](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md), the FusedOp would compose: `RMSNorm.emit()` (Q/K/V norms), `Matmul.emit()` (fused QKV), `AllReduce.emit()` (reduce_scatter + all_gather), `RoPE.emit()`, `SDPA.emit()`, `Matmul.emit()` (O-projection), `ResidualAdd.emit()`. This is structurally similar to Blaze's existing `PreSDPA` + `PostSdpa` for DeepSeek V3 [Ch4/03, Section 3.1](../ch04_blaze_kernel_composition/03_ops_catalog_overview.md), but with Gemma4-specific differences (fused QKV, per-head norms, partial RoPE).

### The `TTNNBlazeDelegate` Base Class

A new `TTNNModule` subclass would bridge the two frameworks:

```python
class TTNNBlazeDelegate(TTNNModule):
    """Symbiote module that delegates compute to a Blaze FusedOp."""

    def move_weights_to_device_impl(self):
        # 1. Prepare weights in Blaze-compatible format
        # 2. Build the Blaze graph and compile
        self._compiled_program = BlazeCompiler(self.device).compile(
            graph=self._blaze_graph,
            tensors=self._blaze_tensors,
            output_tensor=self._output_buffer,
        )

    def forward(self, hidden_states):
        # Extract raw ttnn.Tensor (bypass wrapping for inner modules)
        raw_act = hidden_states if not isinstance(hidden_states, TorchTTNNTensor) \
                  else hidden_states.ttnn_tensor
        # Execute fused program (single dispatch)
        result = self._compiled_program.run()
        return result  # post_process_ttnn_module_output handles wrapping
```

### Engineering Tasks for Pathway A

1. **Create `TTNNBlazeDelegate` base class**: `TTNNModule` subclass whose `forward()` calls `compiled_program.run()`. Handles tensor wrapping/unwrapping based on `_bypass_tensor_wrapping`. (~1 week)

2. **Build `BlazeCompiler` cache**: Compile the Blaze program once during `move_weights_to_device()`, cache the `MeshCompiledProgram`, reuse on every `forward()`. Maps to `TracedRun`'s capture/replay pattern at the kernel level. (~1 week)

3. **Implement weight handoff**: Either (a) share `ttnn.Tensor` references between Symbiote's weight lifecycle and Blaze's tensor inputs, or (b) let Blaze's `WeightProvider` manage its own weights from the same HF checkpoint [Ch4/03, Section 10](../ch04_blaze_kernel_composition/03_ops_catalog_overview.md). (~1-2 weeks)

4. **Validate semaphore isolation**: Ensure `TT_CCL` semaphore indices [Ch3/03](../ch03_symbiote_multi_device/03_ccl_operations.md) and Blaze's `_alloc_mesh_semaphore()` named semaphores [Ch4/02](../ch04_blaze_kernel_composition/02_ccl_in_blaze.md) occupy disjoint address ranges. Use Blaze prefixes (e.g., `"blaze.all_reduce.out_sem"`) to avoid collisions. (~0.5 weeks)

5. **Handle dispatch mode conflict**: Blaze requires `TT_METAL_SLOW_DISPATCH_MODE=1` [model_pipeline.py:line 45-47]. Options: (a) run entire model in slow dispatch (acceptable if Blaze layers dominate), (b) use `ttnn.generic_op()` for non-persistent FusedOps (may work in fast dispatch but unvalidated). (~variable, depends on TTNN runtime)

### Blockers

| Blocker | Severity | Root Cause |
|---------|----------|------------|
| Dispatch mode conflict (fast vs slow) | Critical | `TT_METAL_SLOW_DISPATCH_MODE` is process-global [Ch6/02, Section 6.9](../ch06_model_implementations/02_blaze_deepseek_v3_pipeline.md) |
| Weight format divergence | High | Symbiote uses per-module `ttnn.Tensor`; Blaze uses `OverlappedView` packing [Ch4/01, Section 5.2](../ch04_blaze_kernel_composition/01_blaze_op_architecture.md) |
| CCL semantic gap | High | Symbiote CCL uses TTNN API; Blaze CCL uses MicroOp emissions with `setup_fabric()` [Ch4/02](../ch04_blaze_kernel_composition/02_ccl_in_blaze.md) |
| Compilation overhead | Medium | `BlazeCompiler.compile()` is expensive; must be cached, not run per-forward |

**Feasibility: HIGH** for a proof-of-concept (2-4 weeks). The tensor interface is bridgeable. The dispatch mode is the primary risk. A prototype could target a single attention block.

---

## Pathway B: Pipeline Orchestration (Symbiote Modules as Blaze Pipeline Stages)

**Concept**: Use Blaze's `PipelineGraph` and D2D socket infrastructure for multi-galaxy PP, but implement individual stages using Symbiote `TTNNModule` replacements. This gives Symbiote models pipeline parallelism without kernel-level rewrite.

### Architecture

```python
class SymbioteDecoderStage(StageKind):
    def __init__(self, hf_model_path, layer_ids, nn_to_ttnn):
        self.layer_ids = layer_ids
        self.nn_to_ttnn = nn_to_ttnn

    def create_pipeline_block(self, ctx):
        return PipelineBlock(
            ctx.mesh_device, PIPELINE_CORE_COORD,
            upstream_d2d_socket_page_size=ACTIVATION_PAGE_SIZE_BYTES,
            downstream_d2d_socket_page_size=ACTIVATION_PAGE_SIZE_BYTES, ...)

    def setup(self, ctx, pipeline_block):
        model_slice = load_layers(self.hf_model_path, self.layer_ids)
        register_module_replacement_dict(model_slice, self.nn_to_ttnn)
        set_device(model_slice, ctx.mesh_device)
        # preprocess + move weights for assigned layers only

    def launch_compute(self, ctx, pipeline_block):
        activation = pipeline_block.get_upstream_socket().read()
        output = self.model_slice(activation)
        pipeline_block.get_downstream_socket().write(output)
```

### Blockers

| Blocker | Severity | Root Cause |
|---------|----------|------------|
| Persistent mode incompatibility | Critical | Blaze stages loop on-device via semaphores; Symbiote re-dispatches per `forward()` [Ch5/03](../ch05_blaze_pipeline_system/03_stage_kinds_and_execution.md) |
| Mesh shape mismatch | High | Symbiote modules assume T3K `(1,8)`; pipeline submeshes are `(4,2)`. `@run_on_devices(DeviceArch.T3K)` fails on submesh shapes [Ch6/01](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md) |
| Slow dispatch requirement | High | Same process-global conflict as Pathway A |
| Socket page size parameterization | Medium | Blaze hardcodes `ACTIVATION_DIM=7168` for DeepSeek V3; other models need different page sizes [stage_io.py, lines 28-34] |

**Feasibility: MEDIUM**. Technically possible but performance-defeating: the host-driven dispatch model negates persistent-mode latency benefits. Best used as a prototyping step before full Blaze migration. Estimated effort: 4-8 weeks.

---

## Pathway C: Prototype-to-Production (Sequential Handoff)

**Concept**: The most practical pathway today. Use Symbiote for rapid model bring-up and numerical validation, then incrementally migrate to Blaze for production. No runtime integration required.

### Migration Stages

```
Stage 1: Symbiote Prototype (days)
  - register_module_replacement_dict with basic TTNNLinear
  - NORMAL_WITH_FALLBACK mode to find unsupported ops
  - SEL/DPL mode for numerical validation [Ch2/03](../ch02_symbiote_core/03_run_modes.md)
  - Export timing CSV to identify bottleneck modules

Stage 2: Bottleneck Identification (hours)
  - DispatchManager.save_stats_to_file("timing.csv") [Ch2/04](../ch02_symbiote_core/04_end_to_end_model_flow.md)
  - Identify top-3 modules by wall-clock time
  - Map CCL overhead per module from timing data

Stage 3: Targeted Blaze Replacement (weeks)
  - Write Blaze FusedOps for bottleneck modules
  - Wrap as TTNNBlazeDelegate (Pathway A) for validation
  - OR: begin full Blaze pipeline migration

Stage 4: Full Blaze Pipeline (months)
  - PipelineGraph with stage factories
  - WeightProvider for the model
  - PipelineManager for continuous batching
  - Persistent mode + production tuning
```

### Concrete Handoff Points

The developer should identify modules where the Symbiote-to-Blaze tensor boundary is cleanest:

| Module Type | Symbiote Implementation | Blaze Replacement | Handoff Tensor |
|------------|------------------------|-------------------|----------------|
| Attention block | `TTNNGemma4Attention` | `MLA` or `PreSDPA` FusedOp | `[B, S, hidden]` bfloat16 TILE |
| MoE block | `TTNNQwen3MoE` | `MoE` or `AllReduceMoE` FusedOp | `[B, S, hidden]` bfloat16 TILE |
| Full decoder | `TTNNGemma4DecoderLayer` | `SparseLayer` FusedOp | `[B, S, hidden]` bfloat16 TILE |
| LM head | `nn.Linear` (CPU or TTNN) | `LMHeadSampling` FusedOp | `[B, S, hidden]` -> token |

At each handoff, the tensor crossing the boundary is an activation in `ttnn.TILE_LAYOUT` at `bfloat16`. The shape and layout are identical in both frameworks.

### How Symbiote Enables the Migration

1. **Run mode progression** serves as the validation ladder [Ch2/03](../ch02_symbiote_core/03_run_modes.md): `CPU` -> `NORMAL_WITH_FALLBACK` -> `SEL` -> `NORMAL` -> `TRACED`.

2. **Timing data** from `DispatchManager` identifies exactly which modules consume the most time [Ch2/04](../ch02_symbiote_core/04_end_to_end_model_flow.md). The CSV distinguishes TTNN ops from PyTorch fallbacks.

3. **Numerical reference**: SEL mode produces golden outputs for every operation, serving as ground truth for Blaze kernel validation [Ch2/03](../ch02_symbiote_core/03_run_modes.md).

4. **Shared MicroOps**: Blaze's existing `Matmul`, `RMSNorm`, `RoPE`, `AllReduce`, `Mcast`, `Gather`, `Scatter` MicroOps [Ch4/03, Section 2](../ch04_blaze_kernel_composition/03_ops_catalog_overview.md) cover building blocks for most transformer architectures.

**Feasibility: HIGHEST**. This is the current de facto state. DeepSeek V3 and GLM-5.1 were developed this way implicitly. No architectural changes to either framework required.

---

## Pathway Comparison

| Aspect | Pathway A (Delegate) | Pathway B (Orchestrate) | Pathway C (Sequential) |
|--------|---------------------|------------------------|----------------------|
| Dispatch mode | Blocked (fast/slow conflict) | Blocked (slow only) | No conflict (separate phases) |
| Tensor conversion | Per-forward-call overhead | Socket bridge needed | None (separate processes) |
| Developer effort | Medium (2-4 weeks POC) | High (4-8 weeks POC) | Highest total (8-16 weeks/model) |
| Performance | Near-Blaze for fused layers | Pipeline parallelism, no fusion | Framework-specific optimal |
| HF compatibility | Preserved | Lost (`model.generate()` incompatible) | Symbiote: yes; Blaze: no |
| Current feasibility | High (with dispatch workaround) | Medium | Current state |

---

## Decision Framework

```
"I have a new model to deploy on Tenstorrent hardware."

Is model < 35B parameters?
  YES -> Symbiote only (T3K TP=8). Done.
  NO  -> Continue.

Is model < 100B and latency-sensitive?
  YES -> Symbiote on TG (TP=32), accept CCL overhead.
         If CCL overhead unacceptable -> Pathway C to Blaze.
  NO  -> Continue.

Is model > 100B?
  YES -> Blaze pipeline required (multi-galaxy PP+TP).
         Use Pathway C: prototype on Symbiote, rewrite in Blaze.

Is rapid iteration more important than peak performance?
  YES -> Stay on Symbiote. Use Pathway A to fuse hot layers.
  NO  -> Full Blaze implementation via Pathway C.
```

---

## What Would Make Integration Easier

1. **Per-program dispatch mode in TTNN**: Eliminates the process-global fast/slow conflict. Unblocks Pathways A and B.

2. **Shared compilation cache**: Blaze-compiled programs persisted to disk, invoked via lightweight `TTNNBlazeDelegate.forward()` without per-call compilation overhead.

3. **Submesh-aware `DistributedConfig`**: Allowing per-submesh `DistributedConfig` instances would enable hybrid PP+TP within Symbiote for the 35-100B model range.

4. **Standard tensor metadata**: A shared format on `ttnn.Tensor` encoding sharding strategy and logical shape, eliminating manual `DistributedTensorConfig` management at boundaries.

> **Warning:** The fast/slow dispatch mode conflict is a process-global setting (`TT_METAL_SLOW_DISPATCH_MODE` environment variable). Until TTNN supports per-program dispatch mode selection, any runtime integration requires committing the entire process to one dispatch mode. This is the single most important infrastructure blocker for Symbiote-Blaze integration.

---

| Previous | Up | Next |
|----------|-----|------|
| [02_parallelism_strategies_compared.md](./02_parallelism_strategies_compared.md) | [Table of Contents](../README.md) | [04_limitations_gaps_and_roadmap.md](./04_limitations_gaps_and_roadmap.md) |
