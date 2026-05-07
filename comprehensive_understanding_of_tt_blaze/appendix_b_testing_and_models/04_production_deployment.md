# Appendix B.4 -- Production Deployment

This appendix documents the end-to-end deployment stack for running TT-Blaze
models on Tenstorrent hardware, covering multi-host launch, pipeline topology,
inference benchmarking, and environment configuration.

## B.4.1 End-to-End Deployment Stack

A production deployment involves the following layers:

```
User prompt
    |
    v
CLI (cli.py) or Inference Server
    |
    v
ModelPipeline (model_pipeline.py)
    |
    v
PipelineConfiguration (pipeline.py)
    |
    v
Pipeline (4-phase setup: configure_block -> setup -> start_pipeline -> start_compute)
    |
    v
StageKind instances per host:
    [EmbeddingStage] -> [DenseDecoderStage x3] -> [MoEDecoderStage xN] -> [LMHeadStage] -> [PassthroughStage]
    |
    v
PipelineBlock (socket-based inter-stage communication)
    |
    v
Blaze/Blitz fused kernels on Blackhole silicon
```

## B.4.2 Multi-Host Launch

### MPI Rank Binding

The pipeline uses `ttnn.distributed_context` for multi-process coordination.
Each MPI rank is bound to one pipeline stage (one Galaxy submesh).

```bash
# 16-process launch on a single pod:
TT_METAL_SLOW_DISPATCH_MODE=1 \
mpirun -np 16 \
    python -m blaze.models.deepseek_v3_b1.cli \
    --weights real \
    --cache-path /path/to/weights \
    --max-new-tokens 128
```

The `cli.py` entry point calls `ttnn.init_distributed_context()` to
initialize the MPI-based distributed runtime.

### Galaxy Topology

Hardware configurations are defined in the `pipeline_builder` module. Each
configuration specifies the mesh shape, submesh shape, and process count:

| Galaxies | Stage Shape | Stages | Mesh Shape | Processes |
|----------|-------------|--------|------------|-----------|
| 1 | 4x2 | 4 | 8x4 | 4 |
| 1 | 2x2 | 8 | 8x4 | 8 |
| 1 | 1x2 | 16 | 4x8 | 16 |
| 1 | 1x1 | 32 | 4x8 | 32 |
| 2 | 4x2 | 8 | 16x4 | 8 |
| 2 | 2x2 | 16 | 16x4 | 16 |
| 2 | 1x2 | 32 | 8x8 | 32 |
| 4 | 4x2 | 16 | 32x4 | 16 |
| 4 | 2x2 | 32 | 16x8 | 32 |
| 4 | 1x2 | 64 | 16x8 | 64 |

Each stage occupies `stage_rows * stage_cols` Blackhole chips. Tensor
parallelism (TP) equals the number of chips per stage.

### Fabric Configuration

Fabric topology auto-selection (`_fabric_config_for_num_procs`), router configuration (`max_packet_payload_size_bytes=15232`), and trace region sizing (`trace_region_size=573440`) follow the same rules documented in [B.2.9](#b29-cli-entry-point).

## B.4.3 Pipeline Builder -- Topology Discovery

The `pipeline_builder/` module provides both an explicit (hardcoded chip IDs) and auto-discovery (`build_topology()` via C++ resolver) path for mapping logical pipeline stages to physical submeshes. The `Node`, `Edge`, `PipelineGraph`, `SubmeshPartition`, and `PipelineLayout` APIs are documented in full in [Ch. 8.01 -- Pipeline Builder](../ch08_multi_host_pipeline/01_pipeline_builder.md).

The deployment-specific detail: `tests/pipeline_builder/test_model_pipelines.py` parametrizes topology tests across all supported configurations:

```python
MODEL_PIPELINE_SPECS = [
    ModelPipelineSpec("deepseek-v3-b1-1g-4x2", num_galaxies=1, stage_rows=4, stage_cols=2),
    ModelPipelineSpec("deepseek-v3-b1-1g-2x2", num_galaxies=1, stage_rows=2, stage_cols=2),
    # ... 10 total specs covering 1/2/4 galaxy configurations
]
```

Each test calls `PipelineGraph.build_topology(mesh_device)` and verifies the resolved stage count matches expectations.

## B.4.4 Inference Benchmarking

### Multi-User Benchmark (pipeline_manager/run_inference.py)

The `run_inference.py` script is a multi-user, multi-turn inference benchmark
client that communicates with a running inference server:

```bash
python pipeline_manager/run_inference.py \
    --url http://localhost:8000/v1/chat/completions \
    --prompts prompts.json \
    --batch-size 32 \
    --max-tokens 8192 \
    --log-file inference_log.txt
```

Features:
- Reads prompts from `prompts.json` (32 users x 8 turns)
- Submits concurrent user sessions in configurable batch sizes
- Tracks session IDs across turns for multi-turn conversations
- Collects streaming responses via SSE (Server-Sent Events)
- Measures per-turn TTFT (time to first token) and TPS (tokens per second)
- Writes per-user results and aggregate statistics to a log file

Key data structures:

```python
@dataclass
class TurnResult:
    turn_index: int
    prompt: str
    response_text: str
    session_id: str
    ttft_ms: float = 0.0
    tps: float = 0.0
    completion_tokens: int = 0
    total_tokens: int = 0

@dataclass
class UserSession:
    user_index: int
    prompts: list[str]
    turns: list[TurnResult]
    error: str | None = None
```

The benchmark produces three output sections:
1. Per-user detailed logs with all turns, prompts, responses, and stats
2. TPS and token count tables (per user, per turn)
3. Global summary (total tokens, avg TPS, avg TTFT, wall-clock time)

Security: all file paths are validated against the current working directory
via `_safe_path()` to prevent path traversal.

## B.4.5 ModelPipeline Class

The `ModelPipeline` class is the primary Python interface for running inference. For the full API (`prefill_forward`, `decode_forward`, `run_inference`, `barrier`, `terminate`) and weight provider selection, see [B.2.8](#b28-modelpipeline----inference-api).

## B.4.6 Environment Variables Reference

### Core Runtime

| Variable | Default | Description |
|----------|---------|-------------|
| `TT_METAL_SLOW_DISPATCH_MODE` | 0 | Required for pipeline mode. Set to `1`. |
| `TT_METAL_HOME` | -- | Path to tt-metal installation |
| `ARCH_NAME` | -- | Target architecture (e.g., `blackhole`) |
| `TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS` | `30000` | Fabric router sync timeout |

Weight management env vars (`BLAZE_WEIGHT_SOURCE`, `BLAZE_WEIGHT_CACHE`, `BLAZE_WEIGHT_CACHE_DIR`, `BLAZE_DEVICE_CACHE_DIR`): see [Appendix A.2](../appendix_a_tooling/02_weight_provider.md). Visualization export (`BLAZE_EXPORT`, `BLAZE_EXPORT_PATH`): see [Appendix A.1.5](../appendix_a_tooling/01_visualizer.md). CB blueprint capture (`BLAZE_CAPTURE_DIR`): see [B.1.4](#b14-cb-blueprint-capture). Full consolidated reference: [Appendix A.3.6](../appendix_a_tooling/03_developer_tools.md).

### CI Environment

| Variable | Value | Description |
|----------|-------|-------------|
| `ARCH_NAME` | `blackhole` | Architecture target |
| `TT_METAL_HOME` | `/work/tt-metal` | tt-metal path in CI container |
| `PYTHONPATH` | `/work:/work/tt-metal` | Python module search path |
| `LD_LIBRARY_PATH` | `/work/tt-metal/build_RelWithDebInfo/lib` | Shared library path |

## B.4.7 Deployment Checklist

1. **Hardware**: Ensure sufficient Blackhole chips for the target topology
   (4/8/16/32/64 chips depending on configuration)

2. **Software**: Install tt-metal from the `blaze-metal` branch; install
   tt-blaze via `pip install -e .`

3. **Weight preparation**: Either:
   - Generate tensorbin cache using Blitz's `prepare_weights.py`
   - Use `--weights state_dict` with HF safetensors directory
   - Use `--weights synthetic` for testing only

4. **Environment**:
   ```bash
   export TT_METAL_SLOW_DISPATCH_MODE=1
   export TT_METAL_HOME=/path/to/tt-metal
   ```

5. **Launch**:
   ```bash
   mpirun -np <NUM_PROCS> python -m blaze.models.deepseek_v3_b1.cli \
       --weights real \
       --cache-path /path/to/weights \
       --prompt "Your prompt here" \
       --max-new-tokens 256
   ```

6. **Monitoring**: Use the benchmark script for throughput measurement:
   ```bash
   python pipeline_manager/run_inference.py \
       --url http://localhost:8000/v1/chat/completions \
       --prompts prompts.json \
       --batch-size 32
   ```

## B.4.8 Pipeline Visualizer

The pipeline layout can be exported to JSON and viewed in the web-based
visualizer:

```python
layout = graph.build_topology(mesh_device)
if int(ttnn.distributed_context_get_rank()) == 0:
    layout.write_json("pipeline_layout.json")
```

When `BLAZE_EXPORT=1` is set, the layout JSON is also written to the
Blaze export directory, and a URL to the hosted visualizer is printed:

```
[BLAZE_EXPORT] generated/viz/pipeline_config.json
[BLAZE_EXPORT] https://super-robot-mv52q6p.pages.github.io/?data=...
```

The `pipeline_builder/visualize.py` module handles the JSON serialization,
converting submesh layouts, stage assignments, and fabric node IDs into a
format consumable by the web frontend hosted via GitHub Pages
(`.github/workflows/pages.yml`).
