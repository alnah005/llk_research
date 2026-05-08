# Chapter 6 -- Part 2: Blaze DeepSeek V3 Pipeline

This section traces a single token through the Blaze pipeline-parallel architecture for DeepSeek V3, covering the multi-process distributed topology, stage decomposition, fused decoder execution, and socket-based inter-stage communication. Unlike Symbiote's tensor-parallel approach (each device runs every layer with sharded weights), Blaze assigns entire decoder layers to separate galaxy/pod submeshes and streams activations between them via D2D sockets. Each component is assessed for production readiness.

**Prerequisites:** Chapter 4 (Blaze architecture overview), Chapter 5 (PipelineGraph, StageKind, PipelineConfiguration, CCL collectives).

---

## 6.7 Pipeline Topologies: 4, 16, and 64 Stages

`create_pipeline_configuration_from_num_procs` (`pipeline.py`, line 155) selects topology by process count. The complete stage layout tables for 4, 16, and 64 processes are in [Ch5/05](../ch05_blaze_pipeline_system/05_hardware_configurations.md). The 64-stage SP4 configuration maps DeepSeek V3's full 61-layer architecture: 3 dense + 58 MoE + bookend embed/LMHead/passthrough stages.

Each stage factory is a closure that captures a `WeightProvider` reference and a `layer_id`. Only the process assigned to that stage materializes its weights on its local mesh (`pipeline.py`, line 206):

```python
def build_pipeline(self, mesh_device):
    my_mesh_id = mesh_device.get_system_mesh_id()
    stage = self._stage_factories[my_mesh_id](mesh_device)
    return Pipeline(mesh_device, stage)
```

> **Warning:** `dense_layer_id_override` and `moe_layer_id_override` exist for testing only. Setting `moe_layer_id_override=3` makes all 58 MoE stages load the same layer-3 weights -- useful for pipeline plumbing validation but produces garbage output.

---

## 6.8 Token Trace: Complete Forward Pass (64-Stage SP4)

### Phase 1: Host Writes Token to Stage 0

`ModelPipeline.prefill_forward` (`model_pipeline.py`, line 96):
```
Host (mesh_id=0 only):
  token_id -> to_padded_input(torch.tensor([[tid]]))
  -> pipeline.write_token(token_tensor)
  -> PipelineBlock H2D socket writes to device DRAM
```

### Phase 2: Embedding (Stage 0)

`EmbeddingStage` (`stage_io.py`, line 59):
```
Stage 0 submesh (e.g., 4x2 = 8 chips):
  H2D socket receives token (TOKEN_PAGE_SIZE_BYTES = 64)
  -> Embedding lookup on device
  -> Activation tensor [1, 7168] (ACTIVATION_DIM = 7168)
  -> D2D socket sends downstream (ACTIVATION_PAGE_SIZE_BYTES = 14336)
       |
       v  [STAGE BOUNDARY -- D2D socket over ethernet]
```

### Phase 3: Dense Decoder Layers (Stages 1-3)

`DenseDecoderStage` (`deepseek_decoder.py`, line 295) extends `DecoderStage`:

```
Stage 1 submesh:
  1. D2D socket receives activation from stage 0
  2. Broadcast to all chips in submesh via entry_node_coord
  3. DecoderBlock.execute (fused op, ~40 tensor arguments):
     a. RMSNorm (gamma_overlapped)
     b. AttentionBlock:
        - Q/K/V projections (matmul_weights_overlapped)
        - RoPE (qrope_sin/cos, krope_sin/cos)
        - KV cache write (ttnn_kv_cache)
        - SDPA with position-based masking
        - O projection
        - WITHIN-SUBMESH CCL: sdpa_cluster_axis=0
     c. Residual add
     d. RMSNorm
     e. Dense MLP: gate_proj + silu, up_proj, down_proj
     f. Residual add
  4. Reduce-to-one via reduce_cluster_axis=1
     -> results consolidated to exit_node_coord
  5. D2D socket sends downstream
       |
       v  [STAGE BOUNDARY]
```

**Within-stage CCL:** The `DecoderBlock` uses two CCL axes. Attention scatters across `sdpa_cluster_axis=0` (rows), while the reduce-to-one gathers across `reduce_cluster_axis=1` (columns). On a 4x2 submesh, attention parallelizes across 4 rows and the final reduce consolidates across 2 columns.

### Phase 4: MoE Decoder Layers (Stages 4-61)

`MoEDecoderStage` (`deepseek_decoder.py`, line 256) adds MoE-specific tensors:

```
Stage N submesh (one of stages 4-61):
  1-2. Receive activation, broadcast within submesh
  3. DecoderBlock.execute (MoE path):
     a-c. RMSNorm + Attention (same as dense)
     d. MoE Gate: gate_mm_weights, gate_bias (e_score_correction),
        gate_indices, enable_routing=True
     e. Expert computation: num_routed_experts=256
        gate_proj/up_proj/down_proj weights on submesh
     f. Shared expert: shared_gate/up/down weights
     g. Combine routed + shared, residual add
  4. Reduce-to-one via MoE semaphores
  5. D2D socket sends downstream
```

**Key difference from Symbiote MoE:** In Blaze, expert weights are fully resident on the submesh. There is no cross-stage `all_to_all_dispatch` -- MoE routing and expert execution happen entirely within the submesh. Cross-submesh communication is limited to the pipeline socket.

### Phase 5: LM Head + Sampling (Stage 62)

`LMHeadStage` (`stage_io.py`, line 110):

```
Stage 62 submesh:
  1. D2D socket receives final hidden state
  2. Broadcast to LMHEAD_INPUT_CORE (10, 9)
  3. LMHeadSampling.op (fused):
     a. RMSNorm (final_norm weight)
     b. Matmul: [1, 7168] x [7168, 16160] = [1, 16160]
        - 101 cores, 160 outputs/core, tiles [1,32] x [32,32]
        - Weight sharded across mesh (ShardTensorToMesh dim=1)
     c. Distributed argmax:
        - Per-core local argmax
        - Gather to ARGMAX_FINAL_CORE (0, 0)
        - Cross-mesh argmax via global_semaphore
     d. Token ID (32 bits) -> exit socket
  4. D2D socket sends downstream (TOKEN_PAGE_SIZE_BYTES = 64)
```

### Phase 6: Token Passthrough and Loopback (Stage 63 -> Stage 0)

```
Stage 63: PassthroughStage(TOKEN)
  Receives token -> forwards unchanged -> loopback edge to stage 0

Stage 0 (loopback):
  Receives token via upstream D2D socket
  -> pipeline.read_output(output_tensor)
  -> Host reads token ID
```

`ModelPipeline.decode_forward` (line 116) then feeds this token back via `write_token` for the next decode step.

---

## 6.9 Pipeline Setup: The 4-Phase Protocol

The 4-phase lifecycle (`configure_block` → `setup` → `start_pipeline` → `start_compute`) is documented in [Ch5/03 "Pipeline Orchestrator: 4-Phase Lifecycle"](../ch05_blaze_pipeline_system/03_stage_kinds_and_execution.md). For DeepSeek V3, the `_build_decoder_program_context()` method (`deepseek_decoder.py`, lines 96-178) assembles all arguments for `DecoderBlock.get_program_context()`, including overlapped weight tensors, RoPE constants, KV cache, MoE gate/expert weights, CCL parameters (`reduce_cluster_axis=1`, `sdpa_cluster_axis=0`), and socket handles.

> **Warning:** The pipeline requires `TT_METAL_SLOW_DISPATCH_MODE=1` (`model_pipeline.py`, line 46). Fast dispatch is not compatible with the persistent kernel + socket model. Forgetting this flag produces a `RuntimeError` at initialization.

---

## 6.10 PipelineGraph: Topology Auto-Discovery

The two build paths (`build` legacy explicit vs `build_topology` auto-discovery) and the C++ `resolve_graph_layout()` solver are documented in [Ch5/01 "Build Path 2: Topology Auto-Discovery"](../ch05_blaze_pipeline_system/01_pipeline_graph_and_layout.md).

> **Warning:** `resolve_graph_layout()` and `generate_blitz_decode_pipeline()` (line 217) are under `ttnn._ttnn.multi_device.experimental` -- private and experimental. API changes in future TTNN versions will break this pipeline without notice.

---

## 6.11 Weight Distribution via WeightProvider

`ModelPipeline.__init__` (`model_pipeline.py`, line 56) selects a weight provider:

| Mode | Class | Usage |
|------|-------|-------|
| `"real"` | `CacheWeightProvider(cache_path)` | Production: pre-converted weight cache on disk |
| `"state_dict"` | `StateDictWeightProvider(model_path)` | Load from HF checkpoint, convert on-the-fly |
| `"synthetic"` | `SyntheticWeightProvider()` | Testing: random weights for pipeline validation |

Each provider exposes `load_embedding(device)`, `load_lm_head(device)`, `load_dense_layer(layer_id, device)`, and `load_moe_layer(layer_id, device)`. The stage factory closures capture these callables, so each host process loads only the weights for its assigned stage.

Weight transforms (`_state_dict_to_mla_tensors`, `_state_dict_to_moe_tensors`) handle HF-to-Blaze format conversion: transpose, deinterleave q_b_proj, split kv_b_proj, unsqueeze norms, TP sharding, and tile shuffling for DRAM bank alignment.

---

## 6.12 Persistent Mode Execution

When `persistent_mode=True` (default for all decoder stages), the `DecoderBlock` program remains resident on compute cores after `launch_compute`. A `persistent_next_iter_semaphore` gates iteration boundaries -- the host signals the semaphore to advance to the next token, and the program loops internally without re-dispatch overhead.

This is critical for decode throughput: re-dispatching a fused program with 30+ weight tensors and multiple CCL operations per token would dominate latency. Persistent mode amortizes dispatch to a single up-front cost.

---

## 6.13 Production Readiness Summary

| Component | Readiness | Blocking Issues |
|-----------|-----------|-----------------|
| Pipeline orchestration | High | Experimental `generate_blitz_decode_pipeline` API |
| Stage wiring (sockets) | High | Hardcoded core coords (`MOE_SENDER_CORE = CoreCoord(12,9)`) |
| Weight loading | High | Three validated modes |
| Decoder stages (MoE+Dense) | Medium | Test utility dependency (`create_decoder_block_tensors`), None parameters |
| LMHead + sampling | High | Persistent mode, distributed argmax |
| Prefill | Low | Token-by-token, not chunked (extremely slow for long prompts) |
| Multi-batch | Not implemented | Hardcoded `batch_size=1` |

---

| Previous | Up | Next |
|----------|-----|------|
| [Part 1: Symbiote TP Models](01_symbiote_tensor_parallel_models.md) | [Table of Contents](../README.md) | [Part 3: Patterns and Antipatterns](03_reusable_patterns_and_antipatterns.md) |
