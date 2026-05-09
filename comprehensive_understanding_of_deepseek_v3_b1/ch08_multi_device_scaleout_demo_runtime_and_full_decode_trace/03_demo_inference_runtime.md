# 8.3 Demo Inference Runtime

This section traces the complete demo runtime stack: the CLI entry point, the
model construction factory, and the generation loop. The demo is the top-level
orchestration layer that ties together weight loading (Chapter 7), pipeline
configuration (Section 8.1), and the model interface (which uses the
communication primitives from Section 8.2). A central design decision explored
here is **why slow dispatch is mandatory** and how the runtime manages the
lifecycle transitions between prefill and decode phases.

> **Cross-reference -> Chapter 1, Section 1.3**: DeepSeekV3 model class
> and decode lifecycle.

> **Cross-reference -> Chapter 7, Section 7.2**: Weight preparation, caching,
> and the BlitzDecodeWeights fusion system.

Source: `demo/cli.py` (211 lines), `demo/runner.py` (101 lines), `demo/runtime.py` (61 lines), `model.py` (229 lines)

---

## 8.3.1 Runtime Architecture Overview

```
 DEMO RUNTIME STACK
 ===================

 +------------------------------------------------------------------+
 |  cli.py (211 lines) -- Entry Point                               |
 |                                                                    |
 |  Responsibilities:                                                |
 |    - Parse CLI arguments (prompt, max_new_tokens, cache_path)     |
 |    - Open mesh device (4x2 validated)                             |
 |    - Load weights from cache (embedding/dense/MoE/LM head)       |
 |    - Initialize tokenizer (HuggingFace AutoTokenizer)             |
 |    - Orchestrate full demo lifecycle                              |
 +------------------------------------------------------------------+
          |
          | create_model(), TokenCodec
          v
 +------------------------------------------------------------------+
 |  runtime.py (61 lines) -- Model + Codec Factory                  |
 |                                                                    |
 |  Responsibilities:                                                |
 |    - Create H2D/D2H sockets and DeepSeekV3 model instance        |
 |    - TokenCodec: int token_id <-> padded ttnn.Tensor              |
 +------------------------------------------------------------------+
          |
          | model (DeepSeekV3), tokenizer, codec
          v
 +------------------------------------------------------------------+
 |  runner.py (101 lines) -- Generation Loop                        |
 |                                                                    |
 |  Responsibilities:                                                |
 |    - Tokenize prompt                                              |
 |    - Drive prefill-by-decode loop                                 |
 |    - Drive autoregressive generation loop                         |
 |    - Stream decoded text chunks to output                         |
 +------------------------------------------------------------------+
          |
          | start(), prefill(), decode_step(), stop()
          v
 +------------------------------------------------------------------+
 |  model.py (229 lines) -- DeepSeekV3 Host Interface               |
 |                                                                    |
 |  Responsibilities:                                                |
 |    - Manage prefill/decode H2D socket transitions                 |
 |    - Track sequence position                                      |
 |    - Interface with HostInterface for data exchange               |
 +------------------------------------------------------------------+
```

**Design rationale -- four-file decomposition:** Each file owns a single
responsibility. `cli.py` is the only file that knows about argument parsing,
environment variables, and mesh device lifecycle. `runtime.py` is the only file
that knows about socket construction. `runner.py` is the only file that knows
about tokenization and sampling. `model.py` is the only file that knows about
H2D/D2H protocol and position tracking. This decomposition means the model can
be tested without a tokenizer, the runner can test with a mock model, and the
CLI can be replaced with a server harness without changing model logic.

---

## 8.3.2 CLI Entry Point: `demo/cli.py`

The CLI module (211 lines) is the top-level entry point. It handles argument
parsing, device initialization, weight loading, and orchestrates the full
inference flow.

### Command-Line Arguments

> **Source**: `demo/cli.py`, lines 98-126

| Argument           | Default                   | Description                          |
|--------------------|---------------------------|--------------------------------------|
| `--prompt`         | `""`                      | Input text to generate from          |
| `--max-new-tokens` | `128`                     | Number of decode steps               |
| `--tokenizer`      | `deepseek-ai/DeepSeek-V3` | HuggingFace tokenizer ID or path     |
| `--loopback-mode`  | `True`                    | Use loopback HostInterface           |
| `--cache-path`     | (required)                | Weight cache directory               |
| `--layer-id-offset`| `0`                       | Layer ID offset for partial loading  |

The `--loopback-mode` flag is currently defaulted to `True` because the real
decoder pipeline (Pre-SDPA, Flash MLA, Post-SDPA, MoE composition) is still
being composed. In loopback mode, the HostInterface simply echoes token IDs
back without decoder computation.

The `--layer-id-offset` argument enables multi-host deployment where each host
runs the CLI with a different offset. The offset is added to the local mesh ID
to compute the global pipeline stage index.

### Mesh Device Initialization

> **Source**: `demo/cli.py`, lines 40-53:
> ```python
> def open_mesh_device():
>     if not os.environ.get("TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"):
>         os.environ["TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"] = "30000"
>     device_params = {"fabric_config": ttnn.FabricConfig.FABRIC_2D}
>     with bh_2d_mesh_device_context(device_params) as mesh_device:
>         mesh_shape = (mesh_device.shape[0], mesh_device.shape[1])
>         assert mesh_shape == EXPECTED_PIPELINE_STAGE_MESH_SHAPE, ...
>         yield mesh_device
> ```

Key details:
- Sets a 30-second fabric router sync timeout if not already configured
- Uses `FABRIC_2D` configuration for the 4x2 Blackhole mesh
- Validates the mesh shape is exactly `(4, 2)` -- fails fast on misconfigured
  hardware

### Slow Dispatch Requirement

> **Source**: `demo/cli.py`, lines 149-152:
> ```python
> if not is_slow_dispatch():
>     raise RuntimeError(
>         "DeepSeek V3 B1 demo requires slow dispatch mode. "
>         "Set TT_METAL_SLOW_DISPATCH_MODE=1 and rerun."
>     )
> ```

**Design rationale -- why slow dispatch is mandatory:** The B1 implementation
uses `ttnn.generic_op()` for all device programs -- D2D exchange kernels,
HostInterface kernels, embedding lookup, and (when composed) the decoder layer
kernels. `generic_op()` submits programs via the slow dispatch path, which
provides synchronous completion guarantees and simpler debugging. Fast dispatch
would offer lower per-op overhead but requires the command queue infrastructure
that `generic_op()` does not use. The explicit check prevents the confusing
failure mode where operations silently hang under fast dispatch.

> **Cross-reference -> Chapter 2, Section 2.1**: The generic_op execution
> model and ProgramDescriptor API.

### Weight Loading

The `load_weights_from_cache()` function dispatches to the correct weight
loader based on the mesh ID (source: `demo/cli.py`, lines 69-95).

> **Cross-reference -> Chapter 1, Section 3.4**: Full mesh-ID-to-loader dispatch
> table with the same 4-way mapping (embedding / dense / MoE / LM head).

Ch8 addition: MoE layers use **fast dispatch** exclusively for expert loading.
For MoE layers (mesh_id 4-61), the routed expert weights are preloaded using
fast dispatch before loading the rest of the layer:

> **Source**: `demo/cli.py`, lines 91-94:
> ```python
> if is_moe:
>     with ttnn.device.setup_fast_dispatch(mesh_device):
>         preloaded_experts = load_moe_routed_experts(cache_path, mesh_device, layer_id)
>     return load_moe_decoder_layer(cache_path, mesh_device, layer_id,
>                                    preloaded_routed_experts=preloaded_experts)
> ```

**Design rationale -- fast dispatch for expert loading only:** MoE layers have
256 routed experts, each a set of weight tensors. Loading them one-by-one
through slow dispatch would be prohibitively slow. The `setup_fast_dispatch()`
context temporarily enables fast dispatch specifically for bulk expert weight
loading. After experts are loaded, the context exits and slow dispatch resumes.

> **Cross-reference -> Chapter 7, Section 7.2**: Weight preparation, per-layer
> cache format, and the weight dataclasses.

### Current Demo Flow

```
 DEMO EXECUTION FLOW (cli.py::run_demo)
 ========================================

 1. Validate slow dispatch mode
 2. Load tokenizer (HuggingFace AutoTokenizer)
 3. Create TokenCodec (batch_size=1)
 4. Open mesh device (4x2 mesh, FABRIC_2D)
 5. Load weights from cache
    +-- Early return after weight load (current state) --+
    |   The code below is unreachable in current B1:     |
    6. Create DeepSeekV3 model (via runtime.create_model)|
    7. Run generation (prefill + decode loop)            |
    8. Return GenerationResult                           |
    +----------------------------------------------------+
```

Note: In the current codebase, `run_demo()` returns `None` immediately after
weight loading (line 171), with the model creation and generation code below
being unreachable. This reflects the active bring-up state where weight loading
is validated but the full decode pipeline is not yet connected.

> **Source**: `demo/cli.py`, lines 168-171:
> ```python
> weights = load_weights_from_cache(cache_path, mesh_device, layer_id_offset)
> logger.info("Weights loaded!")
> return None
> ```

---

## 8.3.3 Model Factory: `demo/runtime.py`

The runtime module (61 lines) provides two components: the model factory
function and the `TokenCodec` class.

### `create_model()`

Creates a `DeepSeekV3` instance with properly configured H2D/D2H sockets:

> **Source**: `demo/runtime.py`, lines 13-42

The function creates three sockets, all on core `(0,0)` of device `(0,0)`:

| Socket              | Mode          | Purpose                        |
|---------------------|---------------|--------------------------------|
| `h2d_socket_prefill`| DEVICE_PULL   | Prefill: device pulls from host|
| `h2d_socket_decode` | HOST_PUSH     | Decode: host pushes to device  |
| `d2h_socket`        | (D2H default) | Device-to-host output          |

All sockets use L1 buffer type with FIFO size computed from the batch size:

> **Source**: `demo/runtime.py`, line 19:
> ```python
> fifo_size = page_size_bytes(batch_size)  # 64 bytes for B=1
> ```

For batch_size=1: `page_size_bytes(1) = align_up(1 * 4, 64) = 64` bytes.

**Design rationale -- two separate H2D sockets:** Prefill and decode use
different H2D modes:

```
  Prefill Phase:                    Decode Phase:
  +------------------+              +------------------+
  | h2d_prefill      |              | h2d_decode       |
  | (DEVICE_PULL)    |              | (HOST_PUSH)      |
  |       |          |              |       |          |
  |       v          |              |       v          |
  | HostInterface    |              | HostInterface    |
  | (prefill)        |              | (decode)         |
  |       |          |              |       |          |
  |       v          |              |       v          |
  |   d2h_socket     |              |   d2h_socket     |
  |   (shared)       |              |   (shared)       |
  +------------------+              +------------------+
```

- **DEVICE_PULL** for prefill: The device asynchronously pulls the next token
  when ready. Appropriate because the host writes all prompt tokens upfront and
  the device processes them at its own pace.
- **HOST_PUSH** for decode: The host pushes each token as soon as sampling is
  complete. Appropriate because there is a strict dependency between the model
  output (logits) and the next input (sampled token).

The transition between modes happens once via `_switch_to_decode()`, avoiding
the overhead of mode negotiation on every step.

### `TokenCodec`

Handles the conversion between integer token IDs and PCIe-aligned ttnn tensors:

> **Source**: `demo/runtime.py`, lines 45-61

```
 TOKEN CODEC DATA FLOW
 ======================

 make_input(token_id=42):
   42  -->  torch.full((1,1), 42, int32)  -->  zero-padded to (1, 16)  -->  ttnn.Tensor
                                                     ^
                                                page_size_datums = 64/4 = 16

 extract_token_id(output_tensor):
   ttnn.Tensor  -->  ttnn.to_torch()  -->  reshape(-1)  -->  int(tensor[0])
```

The `page_size_datums` for batch_size=1 is `64 / 4 = 16` (64 bytes / 4 bytes
per int32), so the padded tensor has shape `(1, 16)` with the actual token ID
in position 0 and zeros elsewhere. The 64-byte alignment satisfies the PCIe
page alignment constraint (source: model.py, line 39).

---

## 8.3.4 Generation Loop: `demo/runner.py`

The runner module (101 lines) implements the model-agnostic generation loop.
It is decoupled from the model implementation through the `ModelLike` protocol
(4 methods: `start`, `prefill`, `decode_step`, `stop`).

> **Cross-reference -> Chapter 1, Section 2.3**: Full `ModelLike(Protocol)`
> definition and `run_generation()` usage example.

The Protocol enables the runner to work with any model that implements these
four methods -- the real `DeepSeekV3` class, a mock model for testing, or a
future multi-model ensemble.

### Tokenization

The `tokenize_prompt()` function supports both `encode()`-style and
`__call__()`-style tokenizers:

> **Source**: `demo/runner.py`, lines 32-40

If tokenization produces zero tokens, `ensure_non_empty_prompt_tokens()` falls
back to the BOS token ID:

> **Source**: `demo/runner.py`, lines 43-48

### Generation Flow

The `run_generation()` function implements the complete inference loop:

```
 GENERATION FLOW (runner.py::run_generation)
 ============================================

      [Tokenize]
          |
          v
 prompt_token_ids = [t0, t1, ..., tS-1]
          |
          v
 prefill_inputs = [make_input(t0), make_input(t1), ..., make_input(tS-1)]
          |
          v
      model.start()        <-- Launch device programs
          |
          v
 +--[Prefill Phase]-------+
 | for each token ti:     |
 |   send ti via H2D      |
 |   receive output D2H   |
 |   (discard if i < S-1) |
 | last_output = final    |
 +-----------+------------+
             |
             v
 next_token = extract_token_id(last_output)
             |
             v
 +--[Decode Phase]-----------------------+
 | for step in range(max_new_tokens):    |
 |   input = make_input(next_token)      |
 |   output = model.decode_step(input)   |
 |   next_token = extract_token_id(out)  |
 |   chunk = tokenizer.decode([token])   |
 |   write_text(chunk)  --> stdout       |
 +---------------------------------------+
             |
             v
      model.stop()         <-- Terminate device programs
             |
             v
 return GenerationResult(
     prompt_token_ids,
     generated_token_ids,
     generated_text
 )
```

> **Source**: `demo/runner.py`, lines 51-101

Key implementation details:

1. **Prefill-by-decode**: Each prompt token is processed one at a time through
   `decode_step()`-equivalent calls (not batched). Only the output from the
   last prompt token is used for sampling the first generated token.

2. **Streaming output**: The `write_text` callback is invoked after each decode
   step, enabling real-time token streaming. The callback in `cli.py` writes
   directly to stdout with `flush=True`.

3. **Clean shutdown**: The `try/finally` block ensures `model.stop()` is always
   called, even if generation raises an exception.

4. **Protocol decoupling**: `runner.py` never imports `ttnn` or any
   Tenstorrent-specific code. It communicates entirely through the `ModelLike`
   protocol, `make_input_tensor`, and `extract_token_id` callables.

**Design rationale -- prefill-by-decode:** The model processes the prompt one
token at a time (not chunked), using the same per-step interface as decode. This
avoids the need for a separate prefill kernel. The tradeoff is higher prefill
latency (S sequential steps instead of 1 batched step), but the advantage is
implementation simplicity -- the same HostInterface, socket setup, and D2D
exchange work for both phases.

---

## 8.3.5 DeepSeekV3 Model Class: Prefill/Decode State Machine

The `DeepSeekV3` class (`model.py`, 229 lines) manages the host-side state machine with two phases: PREFILL_ACTIVE (DEVICE_PULL H2D) and DECODE_ACTIVE (HOST_PUSH H2D).

> **Cross-reference → Chapter 1, Section 2.3**: The full state machine diagram (IDLE → PREFILL_ACTIVE → DECODE_ACTIVE → TERMINATED), `_switch_to_decode()` code (lines 152-161), and the `start()`/`prefill()`/`decode_step()`/`stop()` lifecycle.

The transition from prefill to decode is triggered by the first `decode_step()` call. The `sync_devices=True` flag in prefill termination ensures the device has fully stopped the prefill program before the decode program launches — without this, two programs could compete for the same socket resources.

### Position Tracking

The model tracks the current sequence position (number of tokens processed):

> **Source**: `model.py`, lines 214-219:
> ```python
> self._position += 1
> ...
> @property
> def position(self) -> int:
>     return self._position
> ```

This is maintained for future use when position IDs and KV cache write indices
are needed by the real decoder pipeline.

---

## 8.3.6 Termination Protocol (Detailed)

Clean shutdown is critical for multi-host pipelines where stale semaphores or
unflushed sockets cause hangs. The termination protocol operates at three levels.

### HostInterface Termination

The `HostInterface.terminate()` method follows a strict 4-step sequence:

> **Source**: `micro_ops/host_io/op.py`, lines 399-404:
> ```python
> def terminate(self, sync_devices):
>     self.h2d_socket.barrier()
>     self.d2h_socket.barrier()
>     ttnn.reset_global_semaphore_value(self.termination_semaphore, 1)
>     if sync_devices:
>         ttnn.synchronize_device(self.mesh_device)
> ```

The sequence is critical:

1. **H2D barrier**: Ensures all host-to-device writes have been received by the
   device. Without this, the termination signal could arrive before the last
   data write, causing the kernel to exit before processing all data.

2. **D2H barrier**: Ensures all device-to-host reads have been consumed by the
   host. Without this, the kernel could be terminated while data is still in the
   D2H pipeline, causing the host-side `read_tensor` to hang.

3. **Semaphore set**: Sets the termination semaphore to 1. Both the H2D receiver
   kernel and D2H sender kernel poll this semaphore during their blocking waits.

4. **Optional device sync**: When `sync_devices=True`, waits for the device to
   acknowledge that all programs have completed.

### HOST_PUSH vs DEVICE_PULL Termination Differences

**DEVICE_PULL mode (prefill):** The device initiates PCIe transfers. The key
risk is that the device may have initiated a pull that has not completed when
the semaphore is set. The `h2d_socket.barrier()` ensures this pull completes
before the semaphore is written.

**HOST_PUSH mode (decode):** The host pushes data to the device without the
device requesting it. The risk is that if the host pushes data but the kernel
has already exited its receive loop, the data sits in the socket buffer
unclaimed. The barrier prevents this by ensuring all pushed data is consumed
before termination.

### Loopback Mode Termination

In loopback mode, H2D and D2H kernels communicate via circular buffers on the
same core. The shared termination semaphore covers both kernels:

> **Source**: `micro_ops/host_io/op.py`, lines 72-74

Both kernels poll the same semaphore. Because they run on the same core as
reader/writer pairs, they terminate cooperatively: when the H2D receiver stops
pushing data into the CB, the D2H sender drains what remains and exits.

### Kernel-Level Termination Bound

The device-side kernels use polling loops with termination checks. Per the
source module docstring, clean shutdown completes within approximately 1000
device cycles (~1 us at ~1 GHz clock), avoiding indefinite blocking on
`socket_wait_for_pages()`.

> **Source**: `micro_ops/host_io/op.py`, lines 28-31

### Multi-Host Pipeline Shutdown Sequence

For the full multi-host pipeline, the shutdown follows a distributed protocol:

```
 MULTI-HOST SHUTDOWN PROTOCOL
 =============================

 All processes:
   1. ttnn.distributed_context_barrier()    <-- Wait for all stages

 Stage 0 (pipeline start):
   2. host_io.terminate(sync_devices=False)
   3. entry_socket_interface.terminate(sync_devices=False)
   4. exit_socket_interface.terminate(sync_devices=True)   <-- final sync

 Stages 1..N-1:
   2. entry_socket_interface.terminate(sync_devices=False)
   3. exit_socket_interface.terminate(sync_devices=True)   <-- final sync
```

Only the last interface terminated per stage uses `sync_devices=True`. This
ensures all device programs complete before the process exits.

### DeepSeekV3 Model Shutdown

The `stop()` method handles whichever mode is active:

> **Source**: `model.py`, lines 222-229:
> ```python
> def stop(self) -> None:
>     if self._prefill_active:
>         self._host_io_prefill.terminate(sync_devices=True)
>     if self._decode_active:
>         self._host_io_decode.terminate(sync_devices=True)
> ```

Both conditions are checked with `if` (not `elif`), which is defensive: in
normal operation only one can be active, but if an error occurred during mode
transition, both flags could theoretically be set.

---

## 8.3.7 Multi-Host Pipeline Setup (Production Path)

While the current demo uses single-host loopback mode, the PipelineBlock-based
multi-host path is designed for production deployment:

```
 MULTI-HOST PIPELINE SETUP
 ==========================

 1. tt-run launches N MPI ranks (one per pipeline stage)
    using rank_file and rank_binding.yaml

 2. Each rank opens its local mesh device (4x2 slice)

 3. Each rank creates a PipelineBlock:
    +-- Rank 0 (mesh_id=0):
    |     Creates HostInterface + embedding
    |     Creates exit SocketInterface -> stage 1
    |     Creates entry SocketInterface <- stage N-1 (loopback)
    |
    +-- Rank k (mesh_id=k, 0 < k < N-1):
    |     Creates entry SocketInterface <- stage k-1
    |     Creates exit SocketInterface  -> stage k+1
    |     Entry/exit may use fabric for cross-host hops
    |
    +-- Rank N-1 (mesh_id=N-1):
          Creates entry SocketInterface <- stage N-2
          Creates exit SocketInterface  -> stage 0 (loopback)

 4. All ranks call pipeline_block.run()
    (launches D2D exchange kernels on device)

 5. Rank 0 drives the generation loop:
    pipeline_block.write_token(token_tensor)
    pipeline_block.read_output(output_tensor)

 6. MPI barrier + pipeline_block.terminate()
```

> **Source**: `micro_ops/pipeline_block/op.py`, lines 112-157

**Design rationale -- independent host processes:** Each host runs the same
binary with different arguments rather than a single coordinator process. This
design avoids a single point of failure, simplifies debugging (each host's logs
are independent), and scales naturally -- adding a pod means adding more host
processes, not modifying the coordinator.

---

## 8.3.8 Environment Variables

| Variable                                | Purpose                                |
|-----------------------------------------|----------------------------------------|
| `TT_METAL_SLOW_DISPATCH_MODE=1`        | Required for current B1 demo           |
| `TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS`| Fabric sync timeout (default: 30000ms)|
| `TT_VISIBLE_DEVICES`                   | PCIe device IDs for this rank          |
| `TT_MESH_GRAPH_DESC_PATH`              | Path to mesh graph descriptor          |

---

## 8.3.9 Error Handling Summary

| Failure Point | Observable Symptom | Recovery |
|---|---|---|
| H2D write timeout | `write_tensor` hangs | Process kill |
| D2H read timeout | `read_tensor` hangs | Process kill |
| D2D socket hang | Pipeline stalls at stage | MPI timeout |
| Loopback fabric error | Token fails to return to stage 0 | Pipeline deadlock |
| Fast dispatch mode active | `RuntimeError` at startup | Set `TT_METAL_SLOW_DISPATCH_MODE=1` |
| Mesh shape mismatch | `AssertionError` at startup | Verify hardware config |
| Empty prompt | Falls back to BOS token | Automatic |

The most dangerous failure modes in the data path are silent corruptions
(e.g., embedding DRAM errors, incorrect expert computation) that produce
incorrect outputs without any error signal. The termination protocol handles
only clean shutdown; there is no error-triggered abort mechanism in the
current implementation.
