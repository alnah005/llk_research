# 01 -- Pipeline Stage Design

## Context

Pi0.5 inference has a distinctive execution pattern: one expensive prefix computation followed by many cheap denoising iterations that reuse the prefix's KV cache. This "prefix-once-then-iterate" pattern dictates how the model should be decomposed into pipeline stages on Tenstorrent hardware. This section defines the stage boundaries, analyzes three orchestration strategies, and compares with the DeepSeek V3 pipeline that already runs on TT-Blaze.

Source: `/tmp/openpi/src/openpi/models/pi0.py` (lines 224-279, `sample_actions` method).

## The Prefix-Once-Then-Iterate Pattern

The `sample_actions` method in the reference implementation reveals the execution structure:

```
1. embed_prefix(observation)        -- SigLIP + text embedding + prefix attention
2. PaliGemma.llm([prefix, None])    -- fill KV cache with prefix (one pass)
3. LOOP num_steps times:            -- default num_steps=10
   a. embed_suffix(obs, x_t, time) -- project noisy actions + timestep
   b. PaliGemma.llm([None, suffix]) -- attend suffix to cached prefix
   c. action_out_proj(suffix_out)   -- project to action space
   d. x_t = x_t + dt * v_t         -- Euler step
4. return x_0                       -- final denoised actions
```

Steps 1-2 run once per inference call. Step 3 runs 10 times (the default `num_steps`). Step 4 is trivial. This means the denoising loop body (step 3) accounts for roughly 10/(1+10) = ~91% of total wall-clock time at default settings, making it the critical optimization target.

## The Four Pipeline Stages

The natural decomposition follows the functional boundaries in the source code:

```
+-------------------------------------------------------------------+
|                     HOST (Python orchestrator)                     |
+-------------------------------------------------------------------+
      |                                                       ^
      v                                                       |
+-------------+    +----------------+    +----------+    +---------+
|  Stage 1:   |--->|   Stage 2:     |--->| Stage 3: |--->| Stage 4:|
|  SigLIP     |    |  Prefix + KV   |    | Denoise  |    | Action  |
|  Encode     |    |  Cache Fill    |    | (10x)    |    | Output  |
+-------------+    +----------------+    +----------+    +---------+
  (once)             (once)              (loop)           (once)
```

### Stage 1: SigLIP Vision Encoding

**What it does:** Processes up to 3 camera images (base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb) through the SigLIP So400m/14 vision encoder. Each 224x224 image is patchified (14x14 patches = 256 tokens) and processed through 27 transformer layers with width=1152.

**Inputs:** Up to 3 images of shape `[B, 224, 224, 3]` (float32 or normalized to [-1,1]).

**Outputs:** Image tokens of shape `[B, N_images * 256, 2048]` (projected to PaliGemma width via a `head` dense layer with `num_classes=paligemma_config.width`).

**Key ops:**
- `nn.Conv` with kernel_size=(14,14), stride=(14,14) -- patch extraction
- Sincos2D positional embedding
- 27x encoder blocks: LayerNorm, MultiHeadDotProductAttention (16 heads), MLP (width 1152, mlp_dim 4304)
- Final LayerNorm
- Linear projection 1152 -> 2048

**Source:** `/tmp/openpi/src/openpi/models/siglip.py` lines 207-290 (`_Module.__call__`).

**Why a separate stage:** SigLIP runs once per inference and uses standard bidirectional self-attention (no causal masking, no KV cache). It can be compiled and optimized independently. Its ~400M parameters are a distinct memory pool.

### Stage 2: Prefix Embedding + KV Cache Fill

**What it does:** Concatenates image tokens with embedded text tokens, constructs the prefix attention mask (bidirectional between image+text tokens), and runs the full 18-layer Gemma-2B backbone to populate the KV cache. Only the PaliGemma expert (width=2048) is active during this pass; the action expert is `None`.

**Inputs:**
- Image tokens from Stage 1: `[B, N_img * 256, 2048]`
- Tokenized prompt: `[B, max_token_len, 2048]` (after embedding)
- Image masks, prompt masks

**Outputs:** KV cache of shape `[18, B, prefix_len, 1, 256]` for both K and V (18 layers, 1 KV head, head_dim=256).

**Key ops:**
- Token embedding via `Embedder.encode` (vocab 257,152 x 2048)
- Attention mask construction: `make_attn_mask(input_mask, ar_mask)` with prefix-LM pattern
- 18x Gemma blocks with RMSNorm, GQA attention (8 query heads, 1 KV head, head_dim=256), gated MLP (width=2048, mlp_dim=16384)
- RoPE positional encoding

**Source:** `/tmp/openpi/src/openpi/models/pi0.py` lines 106-137 (`embed_prefix`), lines 233-237.

**Why a separate stage:** This is the most memory-intensive stage due to the embedding table (257K x 2048 = ~1 GB at bf16) and the large MLP weights (2048 x 16384 gating = ~67 MB per layer). It runs once and produces the KV cache that Stage 3 reuses.

### Stage 3: Denoising Loop (10 iterations)

**What it does:** Each iteration takes noisy actions `x_t`, projects them to action expert tokens, constructs the suffix attention mask, runs the dual-expert Gemma forward pass (suffix queries attend to cached prefix KV + each other), projects output to action space, and performs an Euler step.

**Inputs per iteration:**
- Noisy actions `x_t`: `[B, 50, 32]` (action_horizon=50, action_dim=32)
- Timestep `t`: scalar
- KV cache from Stage 2 (read-only)
- Observation (for state if pi0; for pi0.5, state is in prefix tokens)

**Outputs per iteration:** Updated `x_t` after Euler step.

**Key ops per iteration:**
- `action_in_proj`: Linear 32 -> 1024
- Timestep embedding: `posemb_sincos` + 2-layer MLP (for pi0.5 adaRMS path)
- Suffix attention mask construction
- 18x dual-expert Gemma blocks:
  - PaliGemma expert (width=2048): attends suffix to prefix KV cache. `None` input means only KV from cache.
  - Action expert (width=1024): processes action tokens with adaRMSNorm conditioning
  - Shared attention: Q/K/V projected separately per expert, concatenated for joint SDPA, output split back
- `action_out_proj`: Linear 1024 -> 32
- Euler step: `x_t = x_t + dt * v_t`

**Source:** `/tmp/openpi/src/openpi/models/pi0.py` lines 139-186 (`embed_suffix`), lines 239-278 (`step` function in `sample_actions`).

**Why the critical stage:** This runs 10 times. The dual-expert attention is architecturally unique (see Risk #2 in section 04). Every optimization here multiplies by the step count.

### Stage 4: Action Output

**What it does:** Extracts the final denoised actions `x_0` from the last denoising step. In practice this is just the output of the final Euler step -- no additional computation beyond what Stage 3 already produces.

**Why listed separately:** In a host-orchestrated pipeline, this is where the device-to-host transfer happens. In a device-loop pipeline, this is the exit condition check and the D2H path. It is a pipeline boundary, not a compute stage.

## Orchestration Strategies

### Strategy A: Host-Orchestrated Loop (Recommended First Approach)

```
HOST                           DEVICE
 |                               |
 |-- launch Stage 1 ----------->|-- SigLIP encode
 |<-- image tokens -------------|
 |                               |
 |-- launch Stage 2 ----------->|-- prefix + KV fill
 |<-- (KV cache stays on dev) --|
 |                               |
 |  for step in range(10):       |
 |    |-- launch Stage 3(x_t) ->|-- denoise iteration
 |    |<-- v_t -----------------|
 |    |   x_t += dt * v_t       |   (Euler step on host or device)
 |                               |
 |-- read x_0 from device ----->|
```

**Pros:**
- Simplest to implement and debug. Each stage is a separate TT-Blaze `FusedProgram` or blaze-nn traced module.
- No device-side looping infrastructure needed.
- Easy to profile individual stages.
- The Euler step is trivial and can run on host with negligible overhead.

**Cons:**
- 10 round-trips for the denoising loop. Each round-trip involves H2D transfer of `x_t` ([B, 50, 32] = 6.4 KB at bf16) and D2H transfer of `v_t` (same size). At PCIe Gen4 bandwidth (~25 GB/s), each transfer takes <1 us. The host-side Euler step is a few microseconds. Total loop overhead: ~10 * (1 us H2D + compute + 1 us D2H) -- the compute dominates by orders of magnitude, so the round-trip overhead is negligible.
- Cannot overlap consecutive denoising iterations (no double-buffering).

**Verdict:** Start here. The round-trip overhead is dwarfed by the ~1-2 ms per denoising iteration compute. Move to Strategy B only after proving the per-iteration compute time.

### Strategy B: Device-Side Loop

```
HOST                           DEVICE
 |                               |
 |-- launch Stage 1+2 --------->|-- SigLIP + prefix + KV
 |                               |
 |-- launch Stage 3 loop ------>|-- for step in range(10):
 |                               |    denoise iteration
 |                               |    Euler step
 |                               |
 |<-- read x_0 -----------------|
```

**Implementation on TT-Blaze:** The `PipelineGraph` in `pipeline_builder/graph.py` supports loopback edges:

```python
Edge("denoise", "denoise", is_loopback=True)
```

This creates a physical loopback path from the denoise stage's exit chip back to its entry chip. The `PipelineConfigEntry` at index `N` (where N = num_stages) encodes the loopback entry/exit coordinates. The device-side control flow would need a step counter and termination condition.

**Pros:**
- Eliminates all host round-trips during denoising.
- Enables pipelining: while iteration N is finishing its last layers, iteration N+1 can start its first layers.

**Cons:**
- Requires device-side conditional branching (step counter, comparison, jump). TT-Metal supports this via `jax.lax.while_loop`-style constructs in the dispatch path, but TT-Blaze does not yet have a high-level API for it.
- Debugging is harder: no intermediate `x_t` visibility without explicit trace buffers.
- The KV cache must remain pinned in DRAM across all iterations.

**When to pursue:** After Strategy A achieves correct results and per-iteration latency is characterized. The expected speedup is small (eliminating ~20 us of round-trip overhead from a ~15 ms per-iteration compute).

### Strategy C: Unrolled Pipeline

```
HOST                           DEVICE
 |                               |
 |-- launch full graph -------->|-- SigLIP
 |                               |-- prefix + KV
 |                               |-- denoise iter 0
 |                               |-- denoise iter 1
 |                               |-- ...
 |                               |-- denoise iter 9
 |                               |-- action output
 |<-- read x_0 -----------------|
```

**How it works:** Trace the full `sample_actions` function (including the while loop) into a single program. The loop is unrolled at trace time into 10 sequential copies of the denoising body.

**Pros:**
- Single dispatch, zero round-trips.
- Compiler can see the full graph and potentially optimize across iteration boundaries.

**Cons:**
- 10x weight duplication for the action expert if the compiler cannot deduplicate. At ~311 MB for the action expert, this would require ~3.1 GB just for the unrolled weights -- still within DRAM budget but tight.
- Fixed step count at compile time. Cannot do adaptive step counts.
- Very large graph: 18 layers x 10 iterations = 180 block instances.
- blaze-nn tracing of `jax.lax.while_loop` is an open question (see Risk #6 in section 04).

**When to pursue:** Only if device-side looping proves infeasible AND the fixed step count is acceptable. Weight deduplication is required.

## Comparison with DeepSeek V3 Pipeline on TT-Blaze

The DeepSeek V3 B1 pipeline running on TT-Blaze provides a concrete reference for what the infrastructure can handle today:

```
                 DeepSeek V3 B1                          Pi0.5
+-----------------------------------------+  +-----------------------------------------+
| Stage: Embedding                        |  | Stage 1: SigLIP Encode                  |
|   - Token embedding + RoPE              |  |   - Conv2d patch extraction              |
|   - Submesh: 4x2                        |  |   - 27-layer ViT                        |
+-----------------------------------------+  +-----------------------------------------+
| Stage: Attention layers (N stages)      |  | Stage 2: Prefix + KV Fill               |
|   - GQA + MoE + shared expert           |  |   - 18-layer Gemma-2B (single expert)   |
|   - Submesh: 4x2 each                   |  |   - Prefix-LM attention mask            |
+-----------------------------------------+  +-----------------------------------------+
| Stage: Output (LM head)                 |  | Stage 3: Denoising (10x loop)           |
|   - Final norm + linear                 |  |   - 18-layer dual-expert Gemma           |
|   - Submesh: 4x2                        |  |   - adaRMSNorm + shared SDPA            |
+-----------------------------------------+  +-----------------------------------------+
                                              | Stage 4: Action Output                  |
                                              |   - Linear 1024 -> 32                   |
                                              +-----------------------------------------+
```

### Key Structural Differences

| Aspect | DeepSeek V3 | Pi0.5 |
|---|---|---|
| **Pipeline topology** | Linear chain, no loopback | Linear with loopback for denoising loop |
| **Expert structure** | MoE (8-of-256 routing) + shared expert | Fixed dual-expert (PaliGemma + Action) |
| **Attention pattern** | Causal (autoregressive) | Prefix-LM (bidirectional prefix, causal suffix) |
| **KV cache usage** | Grows with sequence length | Fixed size, reused across denoising steps |
| **Vision encoder** | None | SigLIP So400m (Conv2d + 27-layer ViT) |
| **Looping** | None (single-pass per token) | 10-iteration denoising loop |
| **Width transitions** | Uniform width per stage | 2048 (PaliGemma) + 1024 (Action) in same attention |
| **Conditioning** | None | adaRMSNorm with timestep embedding |

### What Transfers Directly from DeepSeek

1. **GQA attention kernel:** DeepSeek uses GQA (8 query heads, 1 KV head on some configs). Pi0.5's Gemma backbone uses the same GQA layout (8 query heads, 1 KV head, head_dim=256). The `_gqa_attention.py` Blaze ops should work with configuration changes.

2. **Gated MLP:** Both models use gated MLPs with SiLU/GELU activation. DeepSeek's `SharedExpert` FusedOp (KNMatmul -> GatedReduce -> Mcast -> DownProj) maps directly to Gemma's `FeedForward` module (gating_einsum -> GELU -> linear).

3. **RMSNorm:** Both use pre-normalization with RMSNorm. DeepSeek's RMSNorm kernels transfer directly for the standard (non-adaptive) case.

4. **RoPE:** Both use rotary positional embeddings. The `rope` op in `blaze_nn/functional.py` handles this.

5. **Pipeline infrastructure:** `PipelineGraph`, `PipelineLayout`, submesh partitioning, and the `build_topology` auto-discovery path are all reusable.

### What Requires New Work

1. **adaRMSNorm:** Pi0.5's adaptive RMS normalization (scale + shift + gate from timestep conditioning) has no DeepSeek equivalent. This needs a new Blaze MicroOp or an extension to the existing RMSNorm op. See `/tmp/openpi/src/openpi/models/gemma.py` lines 113-131.

2. **Dual-expert width transition:** The attention layer concatenates Q/K/V from two experts with different widths (2048 and 1024), runs joint SDPA, then splits the output back. DeepSeek's MoE uses uniform-width experts. This width mismatch in the SDPA is new.

3. **Conv2d patch extraction:** SigLIP's first op is a 2D convolution (14x14 kernel, 14x14 stride, 3->1152 channels). DeepSeek has no convolution ops. This needs either a new Blaze kernel or a fallback to TTNN's conv2d.

4. **Prefix-LM attention masking:** The `make_attn_mask` function creates a hybrid mask where prefix tokens attend bidirectionally and suffix tokens attend causally. DeepSeek uses pure causal masking. The SDPA kernel needs to accept this non-standard mask pattern.

5. **KV cache management across loop iterations:** DeepSeek grows its KV cache token-by-token in autoregressive decoding. Pi0.5 fills the KV cache once (Stage 2) and reads it 10 times (Stage 3) with a fixed suffix length. The cache access pattern is different: no appending, just repeated reads with different suffix queries.

## Key Takeaways

- Pi0.5 decomposes into 4 stages. The denoising loop (Stage 3) runs 10 times and dominates latency at ~91% of total compute.
- Host-orchestrated looping (Strategy A) is the correct starting point. The ~20 us per-iteration round-trip overhead is negligible against ~15 ms per-iteration compute. Device-loop optimization can wait.
- ~60% of the op-level work (GQA, gated MLP, RMSNorm, RoPE, pipeline infrastructure) transfers from the existing DeepSeek V3 implementation. The new work centers on adaRMSNorm, Conv2d, dual-expert width handling, and prefix-LM masking.
- The loopback edge pattern in `PipelineGraph` already exists in the codebase, designed for exactly this kind of iterative pipeline. The infrastructure is ready; the stage implementations are the work.

---

**Next:** [`02_memory_and_latency_analysis.md`](./02_memory_and_latency_analysis.md)
