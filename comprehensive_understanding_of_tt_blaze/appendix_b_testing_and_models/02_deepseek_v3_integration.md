# Appendix B.2 -- DeepSeek V3/R1 Model Integration

This appendix documents how the DeepSeek V3 (and its reasoning variant R1)
model is integrated into the TT-Blaze pipeline for multi-host inference on
Tenstorrent hardware.

## B.2.1 Module Layout

The DeepSeek V3 B1 integration lives at:

```
blaze/models/deepseek_v3_b1/
    __init__.py
    model_pipeline.py      # ModelPipeline: high-level inference API
    pipeline.py            # PipelineConfiguration, Pipeline, stage factories
    cli.py                 # CLI entry point, mesh device management
blaze/stages/
    __init__.py            # Re-exports all stage kinds
    stage_io.py            # StageKind ABC, Embedding, LMHead, Passthrough stages
    deepseek_decoder.py    # DecoderStage, MoEDecoderStage, DenseDecoderStage
```

External dependencies (in `tt-metal`):
- `models.demos.deepseek_v3_b1.fused_ops.decoder_block.op` -- `DecoderBlock`
- `models.demos.deepseek_v3_b1.fused_ops.attention_block.op` -- `AttentionBlock`
- `models.demos.deepseek_v3_b1.fused_ops.moe.op` -- `MoeOp`
- `models.demos.deepseek_v3_b1.fused_ops.lm_head_sampling.op` -- `LMHeadSampling`
- `models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op` -- `PipelineBlock`
- `models.demos.deepseek_v3_b1.demo.weight_provider` -- `WeightProvider` hierarchy

## B.2.2 Pipeline Configurations

The system supports three hardware topologies, auto-selected by
`create_pipeline_configuration_from_num_procs()`:

### 4-Process -- Single Galaxy

```python
def create_single_galaxy_pipeline_configuration(weight_provider, ...):
    """4-stage single-galaxy: Embed -> LMHead -> Token fwd -> Token fwd."""
```

| Stage | Kind | Factory |
|-------|------|---------|
| 0 | `EmbeddingStage` | `weight_provider.load_embedding(device)` |
| 1 | `LMHeadStage` | `weight_provider.load_lm_head(device)` |
| 2 | `PassthroughStage(TOKEN)` | -- |
| 3 | `PassthroughStage(TOKEN)` | -- |

This is a minimal topology used for testing the pipeline infrastructure
without decoder layers.

### 16-Process -- Single Pod

```python
def create_single_pod_pipeline_configuration(weight_provider, ...):
    """16-stage single-pod: Embed -> Dense(0,1,2) -> MoE(3..12) -> LMHead -> Token fwd."""
```

| Stage | Kind | Layer |
|-------|------|-------|
| 0 | `EmbeddingStage` | -- |
| 1-3 | `DenseDecoderStage` | Layers 0, 1, 2 |
| 4-13 | `MoEDecoderStage` | Layers 3-12 |
| 14 | `LMHeadStage` | -- |
| 15 | `PassthroughStage(TOKEN)` | -- |

### 64-Process -- Super-Pod (SP4)

```python
def create_sp4_pipeline_configuration(weight_provider, ...):
    """64-stage super-pod: Embed -> Dense(0,1,2) -> MoE(3..60) -> LMHead -> Token fwd."""
```

| Stage | Kind | Layer |
|-------|------|-------|
| 0 | `EmbeddingStage` | -- |
| 1-3 | `DenseDecoderStage` | Layers 0, 1, 2 |
| 4-61 | `MoEDecoderStage` | Layers 3-60 |
| 62 | `LMHeadStage` | -- |
| 63 | `PassthroughStage(TOKEN)` | -- |

Both the 16- and 64-process configurations accept `dense_layer_id_override`
and `moe_layer_id_override` to force all stages to use a single layer's
weights -- useful for single-layer correctness testing on full hardware.

## B.2.3 Stage Architecture

For the `StageKind` ABC, `StageContext` dataclass, and shared pipeline constants (`TOKEN_PAGE_SIZE_BYTES`, `ACTIVATION_DIM`, etc.), see [Ch. 8.01 -- Pipeline Builder, "Integration with blaze/stages/"](../ch08_multi_host_pipeline/01_pipeline_builder.md).

### EmbeddingStage

Stage 0: receives tokens from host (H2D), runs embedding lookup, sends
activation downstream.

```python
class EmbeddingStage(StageKind):
    def create_pipeline_block(self, ctx):
        return PipelineBlock(
            mesh_device, PIPELINE_CORE_COORD,
            upstream_d2d_socket_fifo_size=TOKEN_FIFO_SIZE,
            downstream_d2d_socket_fifo_size=ACTIVATION_FIFO_SIZE,
            upstream_d2d_socket_page_size=TOKEN_PAGE_SIZE_BYTES,
            downstream_d2d_socket_page_size=ACTIVATION_PAGE_SIZE_BYTES,
            h2d_socket_fifo_size=TOKEN_FIFO_SIZE,
            d2h_socket_fifo_size=TOKEN_FIFO_SIZE,
            d2h_socket_page_size=TOKEN_PAGE_SIZE_BYTES,
            embedding_tensor=self._weights.embedding,
        )
```

### LMHeadStage

Receives activation, runs RMSNorm + MatMul + argmax sampling, sends sampled
token downstream. Key parameters:

```python
class LMHeadStage(StageKind):
    M = 1
    K = ACTIVATION_DIM            # 7168
    NUM_MATMUL_CORES = 101
    N_PER_CORE = 160
    N_TOTAL = 16160               # 101 * 160
    A_TILE = ttnn.Tile([1, 32])
    B_TILE = ttnn.Tile([32, 32])
    OUT_TILE = ttnn.Tile([1, 32])
```

The setup phase allocates broadcast tensors, gamma weights, vocabulary
indices, score buffers, and scratch buffers for the distributed argmax.
`launch_compute` invokes `LMHeadSampling.op(...)` with socket I/O.

### DecoderStage (MoE and Dense Variants)

The `DecoderStage` base class implements the full decoder block pipeline
stage, supporting both MoE and dense configurations:

```python
class DecoderStage(StageKind):
    MOE_SENDER_CORE = ttnn.CoreCoord(12, 9)

    def __init__(self, state_dict, *, weights, layer_idx, position_id,
                 max_seq_len, persistent_mode, is_torus, is_moe,
                 num_routed_experts, use_hardcoded_expert_index,
                 enable_routing): ...
```

The `create_pipeline_block` method wires up the inter-stage sockets using the
pipeline config's entry/exit node coordinates.

The `setup` method:
1. Creates semaphores (`AttentionBlock.create_semaphores`,
   `MoeOp.create_semaphores`)
2. Allocates global reduce semaphores
3. Calls `create_decoder_block_tensors()` to set up all layer tensors
4. Gets socket handles from the `PipelineBlock`
5. Builds the decoder program context via `_build_decoder_program_context()`

The `launch_compute` method invokes `DecoderBlock.execute(...)`.

Two concrete subclasses:

```python
class MoEDecoderStage(DecoderStage):
    """Fused attention + MoE + reduce-to-one."""
    def __init__(self, *, weights=None, layer_idx=4,
                 num_routed_experts=256, ...): ...

class DenseDecoderStage(DecoderStage):
    """Fused attention + dense MLP + reduce-to-one."""
    def __init__(self, *, weights=None, layer_idx=0, ...): ...
```

## B.2.4 MLA Attention Assembly

The MLA (Multi-head Latent Attention) module in the decoder block is
assembled from the following tensors and operations (for the full MLA op tree, see [Ch. 6.04 -- Attention Ops](../ch06_op_library/04_attention_ops.md)):

1. **Input normalization**: RMSNorm with `gamma_overlapped` weights
2. **Q-branch**: `q_a_proj` matmul -> Q-norm -> `q_b_proj` matmul -> RoPE
3. **KV-branch**: `kv_a_proj` matmul -> KV-norm -> `kv_b1_proj` (produces
   compressed KV) + `kv_b2_proj` (produces full KV heads)
4. **KV cache update**: position-indexed write of compressed KV
5. **SDPA**: Scaled dot-product attention with KV cache
6. **Post-SDPA**: `o_proj` matmul -> reduce-to-one across mesh devices

All these are composed inside `DecoderBlock.get_program_context()` which
takes the full set of overlapped weight tensors, RoPE tables, KV cache
buffers, and SDPA scratch tensors as arguments (see the ~50-argument call in
`deepseek_decoder.py`).

## B.2.5 MoE Module Assembly

For MoE layers (layers 3+), the decoder block adds (for MoE composition details, see [Ch. 6.05 -- MoE Ops](../ch06_op_library/05_moe_ops.md) and [Ch. 7.04 -- Worked Example: MoE](../ch07_composition/04_worked_example_moe.md)):

1. **FFN normalization**: RMSNorm with `ffn_norm_overlapped` gamma
2. **Gate MM**: Matmul with `gate_mm_weights_tensor` + sigmoid activation
3. **MoE gate sort**: DeepSeek-style grouped top-k with bias correction
4. **Routed experts**: DRAM-streaming SwiGLU (8 selected experts per token)
5. **Shared expert**: L1-resident SwiGLU with gate/up parallel branches
6. **ReduceToOne**: Cross-device all-reduce of the combined MoE output

The gate/bias/indices/score output tensors are conditionally wired when
`self._is_moe` is True.

## B.2.6 Weight Provider Integration

The `ModelPipeline` constructor accepts a `weights_mode` parameter:

```python
class ModelPipeline:
    def __init__(self, mesh_device, *, weights_mode="real",
                 cache_path=None, model_path=None, ...):
```

| Mode | Provider | Source |
|------|----------|--------|
| `"real"` | `CacheWeightProvider(cache_path)` | Pre-generated tensorbin cache |
| `"state_dict"` | `StateDictWeightProvider(model_path)` | HuggingFace safetensors |
| `"synthetic"` | `SyntheticWeightProvider()` | Seeded random weights |

Each host loads **only** the weights for its pipeline stage. The provider's
`load_embedding()`, `load_lm_head()`, `load_dense_layer()`, and
`load_moe_layer()` methods return pre-formatted device tensors.

Additionally, TT-Blaze provides its own `WeightProvider` hierarchy in
`blaze/weight_provider.py` for the fused-op test path (see Appendix B.1.6).

## B.2.7 PipelineConfiguration and Pipeline

The `PipelineConfiguration` class maps stage IDs to lazy factory functions; each host builds only its own stage via `build_pipeline(mesh_device)`. The `Pipeline` class provides a four-phase orchestrator (`configure_block` -> `setup` -> `start_pipeline` -> `start_compute`). For the full API, see [Ch. 8.01 -- Pipeline Builder](../ch08_multi_host_pipeline/01_pipeline_builder.md).

The DeepSeek-specific entry point is `create_pipeline_configuration_from_num_procs(num_procs)`, which selects the 4/16/64-process topology and registers the corresponding stage factories.

## B.2.8 ModelPipeline -- Inference API

The top-level `ModelPipeline` class wraps the pipeline for end-to-end
inference:

```python
class ModelPipeline:
    def prefill_forward(self, tokens: list[int]) -> int: ...
    def decode_forward(self, input_token: int) -> int: ...
    def run_inference(self, prompt_token_ids, max_new_tokens,
                      on_token=None, eos_token_id=None,
                      return_generated_tokens=False) -> list[int] | None: ...
    def barrier(self): ...
    def terminate(self): ...
```

Key constraints:
- Requires `TT_METAL_SLOW_DISPATCH_MODE=1`
- Only mesh_id 0 (first host) invokes `prefill_forward`/`decode_forward`
- Process count must be 4, 16, or 64
- The `DeepSeekV3` host-side model interface is only instantiated on mesh_id 0

The `run_inference` method implements the standard prefill-then-decode loop:
1. Prefill: send each prompt token through the pipeline
2. Sample the first generated token from the last prefill output
3. Decode loop: feed each generated token back, sample next

## B.2.9 CLI Entry Point

`blaze/models/deepseek_v3_b1/cli.py` provides the production entry point:

```bash
# Launch with MPI (16-process single-pod example):
mpirun -np 16 python -m blaze.models.deepseek_v3_b1.cli \
    --prompt "Hello, world!" \
    --max-new-tokens 128 \
    --weights real \
    --cache-path /path/to/tensorbin/cache \
    --tokenizer deepseek-ai/DeepSeek-R1-0528
```

CLI arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | "Hello, world!" | Input prompt text |
| `--max-new-tokens` | 128 | Generation length |
| `--tokenizer` | deepseek-ai/DeepSeek-R1-0528 | HF tokenizer |
| `--weights` | real | `synthetic`, `real`, or `state_dict` |
| `--cache-path` | None | Tensorbin cache dir (required for `real`) |
| `--model-path` | None | HF model dir (required for `state_dict`) |
| `--fp32` / `--no-fp32` | True | FP32 dest accumulator for LM Head |
| `--persistent-mode` / `--no-persistent-mode` | True | Persistent kernel mode |
| `--dense-layer-id-override` | None | Force all dense stages to one layer |
| `--moe-layer-id-override` | None | Force all MoE stages to one layer |

The CLI auto-selects fabric config from process count:
- 4 processes: `FABRIC_2D`
- 16 processes: `FABRIC_2D_TORUS_Y`
- 64 processes: `FABRIC_2D_TORUS_Y`

Mesh device setup uses `bh_2d_mesh_device_context` which handles fabric
init/teardown, router config (`max_packet_payload_size_bytes=15232`), and
trace region allocation (`trace_region_size=573440`).
