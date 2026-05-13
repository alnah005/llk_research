# 03 -- Complete Pi05Module Hierarchy and Forward Pass

## Context

TT-Blaze provides two layers of abstraction for model authors:

1. **blaze-nn (`blaze_nn/`)**: A PyTorch-like API where engineers write `Module`
   subclasses with `forward()` methods that call `F.*` functional ops. The tracing
   infrastructure converts these calls into either Graph API nodes or FusedProgram
   MicroOp chains, depending on whether the module is marked `_blaze_nn_compose`.

2. **Blaze ops (`blaze/ops/`)**: The underlying MicroOp/FusedOp implementations that
   handle CB allocation, CT args, and kernel dispatch.

For Pi0.5, we need a complete module hierarchy in blaze-nn that mirrors the reference
Pi0 model from `openpi/models/pi0.py` and `openpi/models/gemma.py`. This section maps
out every module, its parameters, and the functional ops it calls.

## Key Takeaways

1. The module hierarchy directly mirrors the reference Flax/JAX implementation:
   `Pi05Module` -> `GemmaBackbone` -> `Block` (x18) -> `{Attention, FeedForward,
   RMSNorm/AdaRMSNorm}`, plus top-level `SiglipEncoder`, `ActionInProj`,
   `TimeMLP`, `ActionOutProj`.

2. blaze-nn's `Module` base class already supports `_parameters` (weight tensors),
   `_modules` (sub-modules), `state_dict()` / `load_state_dict()`, and `to(device)`.
   No modifications to the base class are needed.

3. The existing `F.*` ops (`F.linear`, `F.rmsnorm`, `F.rope`, `F.gated_reduce`,
   `F.residual_add`, `F.mcast`, `F.gather`) cover most of the computation. New
   `F.*` ops are needed for: `F.ada_rmsnorm`, `F.sdpa`, `F.gated_residual_add`,
   `F.concat_seq`, `F.split_seq`, `F.silu`, `F.posemb_sincos`.

4. The `compose` decorator on `forward()` determines execution mode. Modules marked
   with `_blaze_nn_compose = True` use the `ComposeTracingContext` (manual FusedOp
   composition). Unmarked modules use the `GraphTracingContext` (auto-fused graph).

## Complete Module Hierarchy

```
Pi05Module
|
+-- siglip: SiglipEncoder
|   |-- (image encoding, produces image tokens at width=2048)
|   +-- weights: patch_embed, transformer blocks, final_proj
|
+-- embedder: Embedder
|   |-- input_embedding: Parameter [257152, 2048]
|   +-- (token embedding for language inputs)
|
+-- action_in_proj: Linear
|   |-- weight: Parameter [action_dim=32, 1024]
|   +-- (project noisy actions to action expert width)
|
+-- time_mlp_in: Linear
|   |-- weight: Parameter [1024, 1024]
|   +-- (first layer of timestep MLP)
|
+-- time_mlp_out: Linear
|   |-- weight: Parameter [1024, 1024]
|   +-- (second layer of timestep MLP)
|
+-- action_out_proj: Linear
|   |-- weight: Parameter [1024, action_dim=32]
|   +-- (project action expert output back to action space)
|
+-- backbone: GemmaBackbone
    |
    +-- layers: ModuleList[Block] (x18)
    |   |
    |   +-- Block[i]
    |       |
    |       +-- pre_attention_norm: RMSNorm (VLM, index 0)
    |       |   +-- scale: Parameter [2048]
    |       |
    |       +-- pre_attention_norm_1: AdaRMSNorm (Action, index 1)
    |       |   +-- w_mod: Parameter [1024, 3072]
    |       |   +-- (no learned scale -- modulation replaces it)
    |       |
    |       +-- attn: DualExpertAttention
    |       |   +-- q_einsum: Parameter [8, 2048, 256]   (VLM Q)
    |       |   +-- kv_einsum: Parameter [2, 1, 2048, 256] (VLM KV)
    |       |   +-- q_einsum_1: Parameter [8, 1024, 256] (Action Q)
    |       |   +-- kv_einsum_1: Parameter [2, 1, 1024, 256] (Action KV)
    |       |   +-- attn_vec_einsum: Parameter [8, 256, 2048] (VLM O)
    |       |   +-- attn_vec_einsum_1: Parameter [8, 256, 1024] (Action O)
    |       |
    |       +-- pre_ffw_norm: RMSNorm (VLM, index 0)
    |       |   +-- scale: Parameter [2048]
    |       |
    |       +-- pre_ffw_norm_1: AdaRMSNorm (Action, index 1)
    |       |   +-- w_mod: Parameter [1024, 3072]
    |       |
    |       +-- mlp: FeedForward (VLM)
    |       |   +-- gating_einsum: Parameter [2, 2048, 16384]
    |       |   +-- linear: Parameter [16384, 2048]
    |       |
    |       +-- mlp_1: FeedForward (Action)
    |           +-- gating_einsum: Parameter [2, 1024, 4096]
    |           +-- linear: Parameter [4096, 1024]
    |
    +-- final_norm: RMSNorm (VLM)
    |   +-- scale: Parameter [2048]
    |
    +-- final_norm_1: AdaRMSNorm (Action)
        +-- w_mod: Parameter [1024, 3072]
```

## Module Definitions in blaze-nn

### Pi05Module (top-level)

```python
class Pi05Module(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.siglip = SiglipEncoder(config.paligemma_config)
        self.embedder = Embedder(PALIGEMMA_VOCAB_SIZE, config.vlm_width)
        self.action_in_proj = Linear(config.action_dim, config.action_width)
        self.time_mlp_in = Linear(config.action_width, config.action_width)
        self.time_mlp_out = Linear(config.action_width, config.action_width)
        self.action_out_proj = Linear(config.action_width, config.action_dim)
        self.backbone = GemmaBackbone(config)

    def forward(self, ...):
        # See inference orchestration (chapter 05)
        ...
```

### Block

```python
class Block(Module):
    def __init__(self, vlm_config, action_config):
        super().__init__()
        self.pre_attention_norm = RMSNorm(vlm_config.width)
        self.pre_attention_norm_1 = AdaRMSNorm(action_config.width)
        self.attn = DualExpertAttention(vlm_config, action_config)
        self.pre_ffw_norm = RMSNorm(vlm_config.width)
        self.pre_ffw_norm_1 = AdaRMSNorm(action_config.width)
        self.mlp = FeedForward(vlm_config.width, vlm_config.mlp_dim)
        self.mlp_1 = FeedForward(action_config.width, action_config.mlp_dim)

    def forward(self, xs, kv_cache, positions, attn_mask, adarms_cond):
        # xs is [vlm_tokens, action_tokens] -- either can be None

        # Pre-attention norms
        vlm_normed = F.rmsnorm(xs[0], self.pre_attention_norm.scale)
        action_normed, gate_attn = F.ada_rmsnorm(
            xs[1], adarms_cond[1], self.pre_attention_norm_1.w_mod
        )

        # Shared attention
        attn_outs, kv_cache = self.attn(
            [vlm_normed, action_normed], positions, attn_mask, kv_cache
        )

        # Gated residual adds
        xs[0] = F.residual_add(xs[0], attn_outs[0])
        xs[1] = F.gated_residual_add(xs[1], attn_outs[1], gate_attn)

        # Pre-FFN norms
        vlm_ffn_in = F.rmsnorm(xs[0], self.pre_ffw_norm.scale)
        action_ffn_in, gate_ffn = F.ada_rmsnorm(
            xs[1], adarms_cond[1], self.pre_ffw_norm_1.w_mod
        )

        # FFNs
        vlm_ffn_out = self.mlp(vlm_ffn_in)
        action_ffn_out = self.mlp_1(action_ffn_in)

        # Final residual adds
        xs[0] = F.residual_add(xs[0], vlm_ffn_out)
        xs[1] = F.gated_residual_add(xs[1], action_ffn_out, gate_ffn)

        return xs, kv_cache
```

### FeedForward (GeLU-gated FFN)

```python
class FeedForward(Module):
    def __init__(self, width, mlp_dim):
        super().__init__()
        self.gating_einsum = Parameter()  # [2, width, mlp_dim]
        self.linear = Parameter()         # [mlp_dim, width]

    def forward(self, x):
        gate = F.sliced_matmul(x, self.gating_einsum, branch="gate")
        up   = F.sliced_matmul(x, self.gating_einsum, branch="up")
        activated = F.gated_reduce(gate, up, activation="gelu")
        return F.linear(activated, self.linear)
```

## Existing F.* Ops (Available in blaze_nn/functional.py)

| Op | Signature | Blaze Backend |
|----|-----------|---------------|
| `F.linear` | `(input, weight) -> output` | `blaze.matmul` |
| `F.rmsnorm` | `(input, gamma, epsilon=1e-6) -> output` | `blaze.rmsnorm` |
| `F.mcast` | `(input) -> output` | `blaze.mcast` |
| `F.gather` | `(input) -> output` | `blaze.gather` |
| `F.gated_reduce` | `(gate, up, activation="silu") -> output` | `blaze.gated_reduce` |
| `F.residual_add` | `(input, bias) -> output` | `blaze.residual_add` |
| `F.rope` | `(input, trans_mat=None) -> output` | `blaze.rope` |
| `F.sliced_matmul` | `(input, weight, branch="gate") -> output` | `blaze.kn_sliced_matmul` |

These eight ops handle the core compute primitives: matrix multiplication, normalization,
element-wise operations, and distributed communication. The dispatch mechanism in
`functional.py` resolves each call to either a Blaze graph node (via `blaze.fuse()`)
or a MicroOp `emit()` (via `FusedProgram`).

## New F.* Ops Needed for Pi0.5

| Op | Signature | Purpose | Blaze Backend |
|----|-----------|---------|---------------|
| `F.ada_rmsnorm` | `(input, cond, w_mod, epsilon=1e-6) -> (output, gate)` | Adaptive RMSNorm with timestep conditioning | New `AdaRMSNorm` FusedOp |
| `F.sdpa` | `(q, k, v, mask, positions) -> output` | Scaled dot-product attention | Existing `blaze.sdpa` or `flash_mla` |
| `F.gated_residual_add` | `(x, y, gate) -> output` | `x + y * gate` for adaRMS gating | New `GatedResidualAdd` MicroOp |
| `F.concat_seq` | `(a, b, dim=1) -> output` | Concatenate along sequence dimension | New `ConcatSeq` MicroOp |
| `F.split_seq` | `(x, lengths, dim=1) -> (a, b)` | Split along sequence dimension | New `SplitSeq` MicroOp |
| `F.silu` | `(input) -> output` | SiLU activation for timestep MLP | New `SiLU` MicroOp |
| `F.posemb_sincos` | `(positions, dim, min_period, max_period) -> output` | Sinusoidal positional encoding | Host-side (pre-computed) |
| `F.embedding` | `(tokens, table) -> output` | Token embedding lookup | `blaze.embedding` (exists) |
| `F.gelu` | `(input) -> output` | GeLU activation (for FFN gating) | Handled inside `F.gated_reduce(activation="gelu")` |

### Implementation notes for new ops

**`F.ada_rmsnorm`**: This is the FusedOp described in chapter 01. It returns a
tuple `(output, gate)` -- the tracing context must handle multi-output ops by
returning a tuple of `TensorProxy` objects. The current `_dispatch` returns a
single `TensorProxy`, so a `_dispatch_multi` variant is needed:

```python
def ada_rmsnorm(input, cond, w_mod, *, epsilon=1e-6, **kwargs):
    """Adaptive RMSNorm. Returns (normalized_output, gate)."""
    return _dispatch_multi("ada_rmsnorm", input, cond, w_mod,
                           epsilon=epsilon, **kwargs)
```

**`F.sdpa`**: Wraps the existing SDPA op from `blaze/ops/sdpa/`. The Q/K/V
concatenation and split happen outside SDPA -- it sees a single fused sequence.

**`F.gated_residual_add`**: A simple element-wise op: `x + y * gate`. Can be
implemented as a MicroOp with three inputs and one output, or composed from
existing `eltwise_mul` + `residual_add` MicroOps.

**`F.posemb_sincos`**: This is computed on host (CPU) before inference begins.
The timestep is a scalar per batch element; the sincos encoding produces a
`[B, width]` tensor that is transferred to device once per denoising step.
No device-side op is needed.

## Module Forward Pass: Tracing Path

When `Pi05Module.__call__()` is invoked:

```
Pi05Module.__call__(input)
    |
    +---> Is _blaze_nn_compose set on forward()? 
    |     NO -> _call_graph() path
    |     YES -> _call_compose() path
    |
    +---> _call_graph() path:
    |     1. Create GraphTracingContext
    |     2. Wrap inputs as TensorProxy(ExternalTensor)
    |     3. Execute forward() -- each F.* call records a graph node
    |     4. BlazeCompiler compiles the graph -> BlazeProgram
    |     5. program.run() -> result
    |
    +---> _call_compose() path:
          1. Create ComposeTracingContext
          2. Create FusedProgram
          3. Execute forward() -- each F.* call emits to FusedProgram
          4. fused_program.run() -> result
```

For the full Pi0.5 model, the recommended approach is:
- Use `_call_graph()` for the top-level orchestration (the graph auto-fuses
  adjacent compatible ops)
- Mark individual FusedOps (AdaRMSNorm, DualExpertPreAttn, etc.) with
  `_blaze_nn_compose = True` so they use manual composition internally

This gives the best of both worlds: automatic scheduling at the macro level,
manual control at the micro level.

## ASCII Diagram: Module Hierarchy

```
Pi05Module
+-----------+-----------------------------------------------------------+
|           |              |           |            |           |        |
SiglipEnc  Embedder   ActionInProj  TimeMLP    ActionOutProj  Backbone
                                   (in+out)                      |
                                                          GemmaBackbone
                                                                |
                                              +------ x18 ------+
                                              |
                                           Block[i]
                                              |
             +--------+--------+---------+----+----+---------+--------+
             |        |        |         |         |         |        |
         PreNorm  PreNorm1  Attn    PreFFNNorm PreFFNNorm1   MLP    MLP1
         (VLM)   (Action)  (Shared)  (VLM)    (Action)     (VLM)  (Action)
        RMSNorm  AdaRMS   DualAttn  RMSNorm   AdaRMS     GeLUFFN  GeLUFFN
         w=2048  w=1024              w=2048    w=1024    16384    4096
```

## Source References

- `blaze_nn/module.py:10-265` -- Module base class
- `blaze_nn/functional.py:1-81` -- Existing F.* ops and `_dispatch` mechanism
- `blaze_nn/_tracing.py:68-137` -- GraphTracingContext
- `blaze_nn/_tracing.py:139-203` -- ComposeTracingContext
- `blaze_nn/modules/_base.py:1-28` -- OpModule base for single-op modules
- `blaze_nn/parameter.py` -- Parameter class
- `openpi/models/gemma.py:44-53` -- Config dataclass (width, depth, mlp_dim, etc.)
- `openpi/models/gemma.py:58-109` -- get_config() for gemma_2b and gemma_300m
- `openpi/models/gemma.py:284-333` -- Block class
- `openpi/models/gemma.py:340-411` -- Module (backbone) class
- `openpi/models/pi0.py:66-101` -- Pi0.__init__ (top-level module structure)
