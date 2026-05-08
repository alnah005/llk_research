# 04 -- Limitations, Gaps, and Roadmap

This file catalogs the concrete limitations of each framework, identifies operations that resist parallelization, documents CCL and device-specific constraints, and proposes a 15-task roadmap with dependency graph. Every limitation is classified as **fundamental** (hardware-bound, requiring architectural change) or **solvable** (engineering effort on existing architecture), following the framing established in the preceding files.

**Prerequisites:** All prior files in this chapter; [Ch6/01-03](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md) (model implementations, patterns and antipatterns).

---

## Symbiote Limitations

### S1: Per-Op Dispatch Overhead (Partially Solvable)

Each `ttnn.*` call is a separate dispatch. A Gemma4 layer issues ~15 dispatches; across 60 layers, this is 900+ per forward pass. On single-token decode, dispatch overhead can exceed compute time [Ch6/01, Section 6.3](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md).

**Root cause:** Symbiote's intercept-and-route dispatch processes each `aten` operation individually [Ch2/02](../ch02_symbiote_core/02_dispatch_and_tensor_wrapping.md). **Mitigation:** `TracedRun` replays recorded op sequences but the trace is still a sequence of individual ops, not a fused kernel [Ch6/03, Pattern 6](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md).

### S2: CPU Fallback for Unsupported Operations (Solvable)

Operations without TTNN handlers fall back to CPU via `dispatch_to_torch_wrapper()`. This forces device-to-host transfer, CPU execution, and host-to-device transfer [Ch2/02](../ch02_symbiote_core/02_dispatch_and_tensor_wrapping.md). In GLM-4.7, attention runs entirely on CPU [Ch6/01, Section 6.2](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md).

### S3: No Pipeline Parallelism (Fundamental to Current Architecture)

`set_device()` propagates a single `MeshDevice` to all modules. No concept of submesh partitioning or inter-submesh data transfer [Ch3/01](../ch03_symbiote_multi_device/01_distributed_config_and_device_init.md). Maximum model size bounded by `N_devices * 12 GB` minus KV cache overhead. On T3K: ~35-70B. On TG: ~100B.

### S4: Trace Fragility (Solvable)

Dynamic allocations inside traced forwards cause crashes. `ttnn.all_reduce` must decompose to `reduce_scatter + all_gather`; `topology=Ring` is mandatory; `@deallocate_weights_after` is incompatible [Ch6/03, Antipattern 3](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md). Ring topology on T3K's physical linear chain requires N-1=7 hops instead of optimal ceil(N/2)=4, a 75% hop penalty.

### S5: No Continuous Batching or Multi-Batch (Solvable)

Symbiote relies on HuggingFace's `model.generate()` loop, processing one request at a time. No batching scheduler, no request queuing, no concurrent user support [Ch2/04](../ch02_symbiote_core/04_end_to_end_model_flow.md).

### S6: Silent Replication Fallback (Solvable)

When a tensor's dimensions are not divisible by the mesh shape, `get_tensor_config_for_tensor()` silently falls back to replication. Qwen3.5's `shared_expert_gate` (output_size=1) crashes when replaced with a TP linear [Ch6/03, Antipattern 2](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md).

---

## Blaze Limitations

### B1: High Authoring Cost Per Model (Solvable with Compiler Infrastructure)

Each model requires hand-authored `FusedOp.compose()` methods. The ops catalog has 114 entries, of which 34 are DeepSeek V3-specific and 18 are GLM-5.1-specific [Ch4/03, Section 1.1](../ch04_blaze_kernel_composition/03_ops_catalog_overview.md). No auto-fusion from a high-level graph.

### B2: Slow Dispatch Requirement (Fundamental to Persistent Kernel Design)

`TT_METAL_SLOW_DISPATCH_MODE=1` is mandatory [model_pipeline.py:line 45-47]. Fast dispatch's asynchronous command queue conflicts with the socket-based synchronization. This precludes using Blaze alongside any fast-dispatch code in the same process.

> **Warning:** Because `TT_METAL_SLOW_DISPATCH_MODE` is a process-global environment variable, there is no way to run Symbiote modules (fast dispatch) and Blaze FusedOps (slow dispatch) in the same process today. This is the single most important blocker for runtime integration pathways A and B described in [File 03](./03_integration_pathways.md). Task R6 addresses this.

### B3: Token-by-Token Prefill (Solvable)

Prefill processes one token at a time. For a 4096-token prompt on 64 stages, this means 4096 full pipeline traversals [Ch6/02, Section 6.13](../ch06_model_implementations/02_blaze_deepseek_v3_pipeline.md). This is the most critical production readiness gap.

### B4: Homogeneous Submesh Shape Requirement (Solvable)

`build_topology()` requires all pipeline nodes to have the same submesh shape [Ch5/01](../ch05_blaze_pipeline_system/01_pipeline_graph_and_layout.md). Stages like Embedding and LMHead waste a full submesh. `PassthroughStage` exists specifically to pad the pipeline [Ch5/03](../ch05_blaze_pipeline_system/03_stage_kinds_and_execution.md).

### B5: Hardcoded Hardware Constants (Solvable)

`MOE_SENDER_CORE = CoreCoord(12, 9)`, `PIPELINE_CORE_COORD = CoreCoord(11, 0)`, and NOC hardcodes assume the Blackhole 13x10 grid. Harvested devices (11x10, 12x10) or future architectures will fail silently [Ch6/03, Antipattern 7](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md). The fix (`GridConfig.from_device()`) is demonstrated in Gemma4 but not applied in Blaze.

### B6: No HuggingFace Integration (Partially Solvable)

No module replacement, no `__torch_dispatch__`, no `model.generate()`. Every model must be written from scratch. A Python wrapper implementing HF's `GenerationMixin` by delegating to `PipelineManager` could restore `model.generate()` compatibility for inference, but cannot support features requiring intermediate hidden states.

### B7: Experimental APIs (Solvable, Awaiting Stabilization)

`resolve_graph_layout()` and `generate_blitz_decode_pipeline()` live under `ttnn._ttnn.multi_device.experimental` [Ch6/02, Section 6.10](../ch06_model_implementations/02_blaze_deepseek_v3_pipeline.md). Changes without notice between TTNN versions will break the pipeline system.

---

## Non-Parallelizable Operations

| Operation | Why It Resists Parallelism | Workaround |
|-----------|---------------------------|------------|
| **Small gating linears** (output_size=1) | Cannot column-shard across N>1 devices | Exclude from replacement; use non-TP linear [Ch6/03, Antipattern 2](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md) |
| **TopK routing** (MoE, 256 experts, BF16) | Precision limit at ~0.004 magnitude | 3-pass centering TopK [Ch6/03, Pattern 7](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md) |
| **RMSNorm on col-sharded data** | Full hidden dim needed for statistics | 3-phase: pre_all_gather, all_gather, post_all_gather [Ch6/03, Pattern 1](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md) |
| **DeltaNet linear attention** | Recurrent state update not on TTNN | `TT_QWEN_CPU_LINEAR_ATTN` fallback [Ch6/01, Section 6.5](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md) |
| **Autoregressive sampling** | Token depends on previous; inherently sequential | HF generate loop (Symbiote); persistent semaphore loop (Blaze) |
| **Dynamic shapes** (variable seq_len) | TTNN trace requires fixed shapes | Power-of-2 padding [Ch6/03, Pattern 6](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md) |

---

## CCL Constraints

| Constraint | Framework | Source |
|------------|-----------|--------|
| `reduce_scatter + all_gather` required (not `all_reduce`) for trace | Symbiote | [Ch3/02](../ch03_symbiote_multi_device/02_weight_sharding_strategies.md) |
| Ring topology required for traced CCL; Linear uses dynamic allocation | Symbiote | [Ch3/03](../ch03_symbiote_multi_device/03_ccl_operations.md) |
| Single ethernet link per hop on T3K (~12.5 GB/s) | Symbiote | [Ch3/03](../ch03_symbiote_multi_device/03_ccl_operations.md) |
| `setup_fabric()` required before any CCL MicroOp | Blaze | [Ch4/02](../ch04_blaze_kernel_composition/02_ccl_in_blaze.md) |
| Cross-stage CCL not possible (only D2D sockets) | Blaze | [Ch5/03](../ch05_blaze_pipeline_system/03_stage_kinds_and_execution.md) |
| Adjacent stage page sizes must match (not validated) | Blaze | [Ch5/03](../ch05_blaze_pipeline_system/03_stage_kinds_and_execution.md) |

---

## Device-Specific Gaps

| Gap | What Is Missing | Impact |
|-----|-----------------|--------|
| **Symbiote on BH Galaxy** | `DeviceArch` enum includes `BHGLX` but no model tests target it [Ch2/01](../ch02_symbiote_core/01_module_replacement_engine.md) | Cannot validate Symbiote on BH hardware |
| **Blaze on Wormhole** | Grid constants assume BH 13x10; WH is 8x8. CB limit is 32 (vs 64) | Blaze cannot run on existing WH systems (N150, N300, T3K) |
| **Blaze single-chip** | Minimum submesh is 1x2 [Ch5/05](../ch05_blaze_pipeline_system/05_hardware_configurations.md) | No single-device path for dev/testing |
| **TG not exercised in Blaze** | Pipeline configs target SP4 with `(4,2)` submeshes | TG `(8,4)` shape not validated |

---

## What None of the Five Models Can Do Yet

These capabilities are absent from all five documented models (GLM-4.7, Gemma4, Ling, Qwen3.5, DeepSeek V3):

1. **Batch size > 1.** All five process a single sequence [Ch6/01](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md), [Ch6/02, Section 6.13](../ch06_model_implementations/02_blaze_deepseek_v3_pipeline.md).
2. **Chunked prefill.** Symbiote uses single-pass HF `model.generate()`; Blaze is token-by-token.
3. **Continuous batching.** No model supports adding/removing sequences from a running batch.
4. **Speculative decoding.** No draft-model + verification pipeline.
5. **Cross-framework numerical validation.** No mechanism to compare a Symbiote module against a Blaze FusedOp at the tensor level.
6. **Vision/multimodal input.** Gemma4's test is text-only (vision tower excluded) [Ch6/03, Antipattern 4](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md).
7. **Dynamic sequence length without padding.** Ling pads to power-of-2 for trace reuse [Ch6/03, Pattern 6](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md). Blaze uses fixed activation page sizes.
8. **Graceful degradation.** If one pipeline stage OOMs, the entire pipeline hangs. Symbiote's CPU fallback has no alerting beyond `DispatchManager` timing [Ch2/03](../ch02_symbiote_core/03_run_modes.md).

---

## Roadmap: 15 Tasks by Priority

### Tier 1: High Impact, Achievable Now (0-3 months)

| # | Task | Addresses | Framework | Effort |
|---|------|-----------|-----------|--------|
| R1 | **Formalize Pathway C handoff artifacts** | Integration | Both | 2-3 weeks |
|    | Tensor shape catalog from `DispatchManager` CSV, CCL pattern specs, golden numerical refs from DPL mode, weight transform docs. |
| R2 | **Blaze single-chip execution path** | B4, Device gap | Blaze | 2 weeks |
|    | 1x1 submesh support with CCL MicroOps as no-ops. Enables N150 development. |
| R3 | **Remove hardcoded core coordinates** | B5 | Blaze | 3-4 weeks |
|    | Replace `MOE_SENDER_CORE`, `PIPELINE_CORE_COORD`, etc. with `device.compute_with_storage_grid_size()`. |
| R4 | **Blaze chunked prefill (design)** | B3 | Blaze | 3-4 weeks |
|    | Design doc for multi-token prefill FusedOp. Variable activation page protocol. Single-stage prototype. |
| R5 | **Size-aware linear replacement** | S6 | Symbiote | 2 weeks |
|    | Auto-detect non-shardable linears (output_size < N_devices) and exclude from TP replacement. |

### Tier 2: High Impact, Requires Infrastructure (3-6 months)

| # | Task | Addresses | Framework | Effort | Dependencies |
|---|------|-----------|-----------|--------|-------------|
| R6 | **Per-program dispatch mode in TTNN** | B2, Integration | TTNN runtime | 4-8 weeks | TTNN team |
|    | API to select fast or slow dispatch per program. Unblocks Pathways A and B from [File 03](./03_integration_pathways.md). |
| R7 | **Blaze chunked prefill (implementation)** | B3 | Blaze | 6-8 weeks | R4 |
|    | Multi-token prefill for DeepSeek V3. KV cache bulk-write. Persistent kernel mode switching. |
| R8 | **Symbiote multi-batch scheduler** | S5 | Symbiote | 6-8 weeks | None |
|    | Request queue, batched forward, paged KV cache concurrent-sequence management. Builds on `TTNNPagedAttentionKVCache` from Ling/Qwen3.5 [Ch6/01](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md). |
| R9 | **TTNNBlazeDelegate proof-of-concept** | Pathway A | Both | 4-6 weeks | R6 (or workaround) |
|    | `TTNNModule` subclass calling a Blaze FusedOp. Demonstrated on one decoder layer. Validates tensor bridge, weight handoff, semaphore isolation. |
| R10 | **Symbiote on BH Galaxy** | Device gap | Symbiote | 3-4 weeks | BH hardware |
|     | Grid-adaptive `preprocess_weights()`, `MESH_DEVICE_MAP` updates, full test suite on BH. |

### Tier 3: Strategic, Long-Term (6-12 months)

| # | Task | Addresses | Framework | Effort | Dependencies |
|---|------|-----------|-----------|--------|-------------|
| R11 | **Blaze on Wormhole** | Device gap | Blaze | 6-10 weeks | R3 |
|     | Parameterized `GridConfig` for 8x8 WH. Retargeted DRAM workers, Mcast groups. Pipeline validated on T3K. |
| R12 | **Inference data parallelism** | DP gap | Both | 4-6 weeks | R8, R7 |
|     | Symbiote: model replication across submeshes. Blaze: multiple `Pipeline` instances with load-balanced `PipelineManager`. |
| R13 | **Automated op-trace-to-FusedOp compiler** | B1, Integration | Both | 12-24 weeks | R1, R9 |
|     | Tool that reads `DispatchManager` trace, maps `ttnn.*` calls to MicroOps, generates draft `compose()` code. Human review required for core grid and L1 layout. |
| R14 | **Heterogeneous submesh pipeline stages** | B4 | Blaze | 4-6 weeks | None |
|     | Constraint-satisfaction solver in `resolve_graph_layout()` for mixed submesh shapes. Enables smaller embedding/LMHead stages. |
| R15 | **Stabilize pipeline APIs** | B7 | Blaze | 2-3 weeks | None |
|     | Promote `resolve_graph_layout()` and `generate_blitz_decode_pipeline()` from `experimental` to stable with versioning. |

---

## Roadmap Dependency Graph

```
  R1 (handoff artifacts) -----> R13 (auto compiler)
                                  ^
  R3 (grid params) -----------> R11 (Blaze on WH)
                                  |
  R4 (prefill design) ---------> R7 (prefill impl) ---------> R12 (DP for Blaze)
                                                                  ^
  R8 (multi-batch Symbiote) --------------------------------------|
                                                                  |
  R6 (dispatch mode) ----------> R9 (TTNNBlazeDelegate) -------> R13 (auto compiler)

  Independent:
  R2 (Blaze single-chip), R5 (size-aware replacement),
  R10 (Symbiote on BH), R14 (heterogeneous submeshes), R15 (API stabilization)
```

---

## Limitation Classification Summary

| ID | Framework | Category | Fundamental or Solvable | Blocking For |
|----|-----------|----------|------------------------|-------------|
| S1 | Symbiote | Dispatch | Partially solvable (trace helps) | Decode latency |
| S2 | Symbiote | Op coverage | Solvable | New model porting |
| S3 | Symbiote | Scale | Fundamental (needs PP) | 100B+ models |
| S4 | Symbiote | Trace/CCL | Solvable (trace-aware allocator) | Developer experience |
| S5 | Symbiote | Serving | Solvable | Production deployment |
| S6 | Symbiote | Config | Solvable | MoE model correctness |
| B1 | Blaze | Usability | Solvable (compiler infra) | Model coverage |
| B2 | Blaze | Dispatch | Fundamental (persistent kernels) | Framework integration |
| B3 | Blaze | Prefill | Solvable | Production prefill latency |
| B4 | Blaze | Pipeline | Solvable | Resource efficiency |
| B5 | Blaze | Portability | Solvable | Hardware compatibility |
| B6 | Blaze | Usability | Partially solvable | Developer adoption |
| B7 | Blaze | API stability | Solvable | Long-term maintenance |

---

## The Developer's Gap Map

```
Model size:  8B      35B       100B       671B
             |        |         |          |
Symbiote:   [=======][========]
             works     works     blocked (S3: no PP)
                      well       on TG

Blaze:                         [=========][==========]
                                works      works
                                (B3: slow  (production
                                 prefill)   ready)

Integration:          [????????????????????]
gap                    35B-100B range lacks a good
                       single-framework solution
                       (R6, R9 would close this gap)
```

---

## State of Play

| Dimension | Today | Target (R1-R15 Complete) |
|-----------|-------|--------------------------|
| Models supported | Symbiote: 4 / Blaze: 2 | 10+ via Pathway C + R13 |
| Max model size | Symbiote: ~35B T3K / Blaze: 671B 4-galaxy | Same, faster onboarding |
| Hardware coverage | Symbiote: WH only / Blaze: BH only | Both architectures (R10, R11) |
| Prefill | Symbiote: batched (fast) / Blaze: token-by-token | Chunked for Blaze (R7) |
| Throughput scaling | Symbiote: no multi-batch / Blaze: 64 users | Multi-batch Symbiote (R8), DP (R12) |
| Integration | Zero coupling | Shared dispatch (R6), delegate bridge (R9) |
| Time to port new model | Hours (Symbiote) / weeks (Blaze) | Days (Symbiote) / days (R13) |

The highest-leverage tasks are R6 (dispatch mode unification), R7 (chunked prefill), and R1 (formalize handoff) -- together these close the three most impactful gaps with reasonable engineering effort. The long-term vision (R13: automated compiler) would transform the Symbiote-to-Blaze transition from a multi-week manual rewrite to a semi-automated process.

---

| Previous | Up |
|----------|-----|
| [03_integration_pathways.md](./03_integration_pathways.md) | [Table of Contents](../README.md) |
