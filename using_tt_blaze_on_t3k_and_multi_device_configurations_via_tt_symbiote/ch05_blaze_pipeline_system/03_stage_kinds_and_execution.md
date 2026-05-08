# 5.3 Stage Kinds and Execution

This section traces data flow through concrete pipeline stages -- from the `StageKind` abstract base through the `PipelineBlock` socket wiring to the `Pipeline` orchestrator's 4-phase lifecycle. The emphasis is on how tokens and activations physically move between stages via D2D sockets and FIFOs, and on the page-size consistency constraints that must hold across adjacent stages.

**Prerequisites:** Section 5.1-5.2 (PipelineGraph, PipelineLayout), Ch3 (CCL, weight sharding), Ch4 (FusedProgram, PipelineBlock basics).

---

## StageKind Abstract Base

Defined in `blaze/stages/stage_io.py` (line 45):

```python
class StageKind(ABC):
    @abstractmethod
    def create_pipeline_block(self, ctx: StageContext) -> PipelineBlock:
        """Create and return the PipelineBlock for this stage."""

    def setup(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        """Post-creation setup (tensor allocation, etc). Default: no-op."""

    def launch_compute(self, ctx: StageContext, pipeline_block: PipelineBlock) -> None:
        """Launch compute kernels after pipeline_block.run(). Default: no-op."""
```

Every stage must implement `create_pipeline_block()`. The other two methods are optional -- simple stages like `PassthroughStage` need only socket wiring, while decoder stages require weight loading and kernel launch.

### StageContext (line 37)

```python
@dataclass
class StageContext:
    mesh_device: ttnn.MeshDevice   # this stage's submesh
    pipeline_config: list          # list[PipelineConfigEntry]
    my_mesh_id: int                # this stage's index into pipeline_config
```

The context provides everything a stage needs to configure itself. `pipeline_config[my_mesh_id]` gives the entry/exit chip coordinates for this specific stage.

### StageKind Hierarchy

```
                  StageKind (ABC)
                  /     |     \
          Embedding  LMHead  Passthrough   DecoderStage
          Stage      Stage   Stage         /          \
                                     DenseDecoder  MoEDecoder
                                     Stage         Stage
```

---

## Socket and FIFO Configuration Constants

Defined at `stage_io.py` lines 28-34, these govern the socket plumbing:

| Constant | Value | Purpose |
|----------|-------|---------|
| `TOKEN_PAGE_SIZE_BYTES` | 64 | One token = 64 bytes (slot_id + token_id + metadata) |
| `TOKEN_FIFO_SIZE` | 1024 | Token socket FIFO depth (bytes) |
| `ACTIVATION_DIM` | 7168 | DeepSeek V3 hidden dimension |
| `ACTIVATION_PAGE_SIZE_BYTES` | 14336 | `7168 * 2` (bfloat16) |
| `ACTIVATION_FIFO_SIZE` | 57344 | `14336 * 4` (4 activation pages) |
| `PIPELINE_CORE_COORD` | `CoreCoord(11, 0)` | Dedicated pipeline management core |

Every stage boundary involves two D2D sockets -- upstream (receives from previous stage) and downstream (sends to next stage). The page size determines the granularity of socket transfers.

---

## Concrete Stage Kinds

### Data Flow Sequence Through a Full Pipeline

```
  H2D Socket                                                    D2H Socket
  (64B token)                                                   (64B token)
      |                                                              ^
      v                                                              |
  [EmbeddingStage]  ---(14KB activation)--->  [DecoderStage x N]     |
      ^                                           |                  |
      |                                      (14KB activation)       |
      |                                           v                  |
      |                                      [LMHeadStage]           |
      |                                           |                  |
      |                                      (64B token)             |
      |                                           v                  |
      +----(64B token, loopback)---- [PassthroughStage] -------------+
```

### EmbeddingStage (line 59)

Stage 0: receives a 64-byte token page from the host via H2D socket, performs embedding lookup, and sends a 14KB activation to the next stage.

```python
class EmbeddingStage(StageKind):
    def create_pipeline_block(self, ctx):
        return PipelineBlock(
            ctx.mesh_device, PIPELINE_CORE_COORD,
            upstream_d2d_socket_fifo_size=TOKEN_FIFO_SIZE,      # loopback receives tokens
            downstream_d2d_socket_fifo_size=ACTIVATION_FIFO_SIZE, # sends activations
            upstream_d2d_socket_page_size=TOKEN_PAGE_SIZE_BYTES,
            downstream_d2d_socket_page_size=ACTIVATION_PAGE_SIZE_BYTES,
            h2d_socket_fifo_size=TOKEN_FIFO_SIZE,
            d2h_socket_fifo_size=TOKEN_FIFO_SIZE,
            d2h_socket_page_size=TOKEN_PAGE_SIZE_BYTES,
            embedding_tensor=self._weights.embedding,
        )
```

Note the asymmetry: upstream uses TOKEN sizes (loopback returns tokens), downstream uses ACTIVATION sizes (sends embeddings to decoders). EmbeddingStage is the **only** stage that creates H2D and D2H sockets -- all other stages use only D2D.

### DenseDecoderStage / MoEDecoderStage (deepseek_decoder.py)

Both inherit from `DecoderStage` (line 32). They receive activations, run the fused `DecoderBlock` (attention + MLP/MoE + reduce), and send activations downstream.

Key socket wiring in `create_pipeline_block()` (line 69):
- Entry chip receives activation via `entry_node_downstream` -- a `MeshCoreCoord` pointing to `MOE_SENDER_CORE = CoreCoord(12, 9)`.
- Exit chip sends via `exit_node_upstream` -- a list of `MeshCoreCoord` at the reduce-root's shard cores.
- Both upstream and downstream use `ACTIVATION_FIFO_SIZE` / `ACTIVATION_PAGE_SIZE_BYTES`.

The two concrete subclasses differ only in constructor arguments:

| Subclass | `is_moe` | Weight Type | Routing |
|----------|----------|-------------|---------|
| `DenseDecoderStage` | False | `DeepSeekV3DenseLayerWeights` | No MoE gate |
| `MoEDecoderStage` | True | `DeepSeekV3MoELayerWeights` | Gate + top-k expert routing |

The `setup()` method (line 180) allocates all decoder tensors (KV cache, attention weights, MoE gate weights) and creates semaphores. The `launch_compute()` method (line 252) calls `DecoderBlock.execute()` with a pre-built program context, which runs the fused attention + MLP kernel in persistent mode.

### LMHeadStage (line 110)

Receives activations, runs fused LMHead + sampling, and sends the 64-byte token downstream.

Key details:
- Upstream page size: `ACTIVATION_PAGE_SIZE_BYTES` (receives 14KB activations).
- Downstream page size: `TOKEN_PAGE_SIZE_BYTES` (sends 64-byte tokens).
- Uses `LMHeadSampling.op()` which performs matmul (weight * activation), argmax across a 101-core grid, and CCL reduction to find the global winner token.
- Custom core assignments: `LMHEAD_INPUT_CORE = CoreCoord(10, 9)`, `ARGMAX_FINAL_CORE = CoreCoord(0, 0)`.

LMHead-specific layout constants:
- `M=1, K=7168` (single-token matmul), `NUM_MATMUL_CORES=101`, `N_PER_CORE=160`, `N_TOTAL=16160`.

> **Warning:** LMHead `setup()` creates multiple global semaphores. If `persistent_mode=True` but a semaphore creation fails (e.g., insufficient L1), the pipeline hangs on the first iteration with no error message -- the kernel waits forever on the missing semaphore.

### PassthroughStage (line 86)

Forward-only stage: receives data and passes it through unchanged. Configured for either `ACTIVATION` or `TOKEN` payload sizes via the `PassthroughPayload` enum. Used as the final stage before the loopback to pad the pipeline when there are more submeshes than compute stages.

---

## Stage Chain Consistency Table

Adjacent stages MUST have matching page sizes at their boundary. The pipeline framework does **not** validate this -- it is the developer's responsibility.

```
  Stage Type       upstream_page   downstream_page   Notes
  ---------------  -------------   ---------------   --------------------
  EmbeddingStage   TOKEN (64B)     ACTIVATION        H2D=TOKEN, D2H=TOKEN
  DecoderStage     ACTIVATION      ACTIVATION        Both directions same
  LMHeadStage      ACTIVATION      TOKEN             Receives act, sends tok
  PassthroughStage configurable    configurable      Matches ACTIVATION or TOKEN
```

> **Warning:** If page sizes mismatch between adjacent stages, the pipeline will silently read garbage. The socket layer does not validate page size compatibility. Always verify by tracing the chain: `EmbeddingStage.downstream = ACTIVATION_PAGE_SIZE` -> `DecoderStage.upstream = ACTIVATION_PAGE_SIZE` (match).

---

## PipelineBlock

Defined in `tt-metal/.../pipeline_block/op.py` (line 30), `PipelineBlock` manages the socket infrastructure for one stage. It creates:

- **Upstream D2D socket**: receives data from the previous stage.
- **Downstream D2D socket**: sends data to the next stage.
- **H2D/D2H sockets** (stage 0 only): host I/O for token injection and extraction.
- **HostInterface** (stage 0 only): manages embedding lookup.

Key methods:
- `run()`: starts the socket forwarding kernels on the pipeline core.
- `write_token(tensor)`: injects a token via H2D (stage 0 only).
- `read_output(tensor)`: reads a token from D2H (stage 0 only).
- `get_downstream_socket()` / `get_upstream_socket()`: returns socket handles for compute kernels to read/write directly.
- `terminate()`: shuts down socket forwarding kernels.

> **Warning:** If stage N's `create_pipeline_block()` fails (e.g., bad core coord), stage N-1's downstream socket never gets a peer. The socket `connect()` on stage N-1 blocks until the 30-second timeout, then crashes with an error pointing at the WRONG stage. Always check the failing rank's logs first.

---

## Pipeline Orchestrator: 4-Phase Lifecycle

Defined in `blaze/models/deepseek_v3_b1/pipeline.py` (line 210):

```python
class Pipeline:
    def __init__(self, mesh_device, stage_kind: StageKind):
        self._pipeline_config = ttnn._ttnn.multi_device.experimental \
            .generate_blitz_decode_pipeline(mesh_device)
        self._ctx = StageContext(mesh_device, self._pipeline_config, mesh_id)
```

### The Four Phases

```
  Phase 1: configure_block()    Phase 2: setup()
  ========================      ========================
  StageKind.create_pipeline_    StageKind.setup()
  _block(ctx)                   - Allocate tensors
  - Create sockets              - Load weights to device
  - Wire entry/exit chips       - Create semaphores

  Phase 3: start_pipeline()     Phase 4: start_compute()
  ========================      ========================
  PipelineBlock.run()           StageKind.launch_compute()
  - Start socket forwarding     - Launch persistent kernels
    kernels on CoreCoord(11,0)  - e.g. DecoderBlock.execute()
```

The `setup_and_run()` convenience method calls all four in sequence. This is the standard path for both tests and production deployment.

> **Warning:** The four phases must execute in strict order. Calling `setup()` before `configure_block()` raises `RuntimeError`. Calling `start_compute()` before `start_pipeline()` would attempt to use sockets that are not yet active, resulting in device hangs.

### Persistent Mode Execution

Decoder and LMHead stages run in **persistent mode**: the compute kernel stays resident on device and processes tokens continuously without re-launch overhead. This is controlled by the `persistent_mode` flag (default `True`) and a `persistent_next_iter_semaphore` that signals the kernel to process the next token.

The persistent kernel loop:
1. Wait on upstream socket for data arrival.
2. Execute compute (attention/MLP/matmul).
3. Write result to downstream socket.
4. Signal `persistent_next_iter_semaphore`.
5. Go to step 1.

Without persistent mode, each token requires a full kernel launch-execute-complete cycle. Persistent mode eliminates this overhead, keeping the pipeline's decode latency close to the raw compute time.

> **Warning:** If `persistent_mode=True` but the pipeline is not continuously fed, the persistent kernel blocks indefinitely on the upstream socket. Use `pipeline.terminate()` to send a sentinel that breaks the kernel out of its wait loop.

---

## Worked Example: 64-Stage SP4 Stage Assignment

The `create_sp4_pipeline_configuration()` function (`pipeline.py` lines 109-152) maps all 64 stages of a 4-galaxy deployment:

```
Stage  0:   EmbeddingStage       (token -> activation)
Stage  1:   DenseDecoderStage    (layer 0, dense MLP)
Stage  2:   DenseDecoderStage    (layer 1, dense MLP)
Stage  3:   DenseDecoderStage    (layer 2, dense MLP)
Stages 4-61: MoEDecoderStage    (layers 3-60, MoE MLP)
Stage 62:   LMHeadStage          (activation -> token)
Stage 63:   PassthroughStage     (TOKEN forward)
```

---

## Barrier and Termination

- `barrier()`: calls `ttnn.distributed_context_barrier()` -- synchronizes all MPI ranks. Used between pipeline phases to ensure all stages are ready before data flows.
- `terminate()`: sends a sentinel through the pipeline to shut down all persistent kernels and socket forwarding.

> **Warning:** Calling `terminate()` from only one rank without a preceding barrier can deadlock the pipeline. Always barrier before termination.

---

| Previous | Up | Next |
|----------|-----|------|
| [5.2 SubmeshPartition and Topology](02_submesh_partition_and_topology.md) | [Table of Contents](../README.md) | [5.4 Pipeline Manager (C++)](04_pipeline_manager_cpp.md) |
