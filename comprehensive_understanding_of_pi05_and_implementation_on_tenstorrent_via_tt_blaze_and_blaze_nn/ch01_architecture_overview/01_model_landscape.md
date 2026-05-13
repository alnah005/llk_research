# 01 -- Model Landscape: What pi0.5 Is and Where It Comes From

## Context

pi0.5 is Physical Intelligence's second-generation vision-language-action (VLA) model for robotic manipulation. It consumes camera images, a natural-language instruction, and the robot's proprioceptive state, then outputs a trajectory of 50 action steps, each with 32 dimensions, covering roughly the next few seconds of motor control. This file introduces the model's purpose, compares it to its predecessor pi0, walks through the `Pi0Config` dataclass, and tallies the parameter budget across all three neural sub-networks.

---

## 1.1 Purpose: Vision-Language-Action for Robotic Control

At its core, pi0.5 solves the mapping:

$$
f_\theta: (\text{images},\; \text{language instruction},\; \text{robot state}) \;\longrightarrow\; \mathbf{a} \in \mathbb{R}^{50 \times 32}
$$

where each of the 50 rows is one future action step and each of the 32 columns encodes a joint-level command (positions, velocities, or gripper states, depending on the robot embodiment). The model is a *flow-matching* generative model: starting from Gaussian noise, it iteratively denoises for 10 Euler steps to produce the final action chunk.

Key properties:

- **Multi-camera input.** Three RGB images at $224 \times 224$: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`.
- **Language-conditioned.** A tokenized natural-language prompt (up to 200 tokens in pi0.5) specifies the task.
- **Continuous action output.** Unlike autoregressive token-prediction approaches, pi0.5 generates *continuous* float32 action vectors through a denoising process.

The model couples three pretrained subsystems in a single shared-attention transformer -- the PaliGemma backbone and the Action Expert share the same 18-layer attention computation but maintain separate projection weights (Q/K/V, FFN) per expert. This "mixture of experts with shared attention" design is the architectural signature of the pi0 family.

Source: `Pi0Config.inputs_spec()` in `/tmp/openpi/src/openpi/models/pi0_config.py` (lines 64-86).

---

## 1.2 pi0 vs pi0.5: What Changed

pi0.5 introduces three concrete architectural differences over the original pi0. All three are gated by a single boolean flag.

| Aspect | pi0 | pi0.5 |
|--------|-----|-------|
| **Timestep conditioning** | Concatenate time embedding with action tokens, pass through MLP (`action_time_mlp_in/out`) | Separate `time_mlp_in/out`, output feeds adaRMSNorm in every transformer layer |
| **State input** | Continuous vector projected by `state_proj` into one extra suffix token | Discrete: state is tokenized into language tokens and included in the prefix |
| **`max_token_len`** | 48 | 200 |
| **`state_proj` layer** | Present (`Linear(32, 1024)`) | Absent |
| **Suffix composition** | 1 state token + 50 action tokens = 51 tokens | 50 action tokens only |
| **Projection modules** | `state_proj`, `action_time_mlp_in`, `action_time_mlp_out` | `time_mlp_in`, `time_mlp_out` |

### 1.2.1 adaRMSNorm

In pi0, the flow-matching timestep $t$ was fused with the action tokens by concatenation and an MLP before entering the transformer. In pi0.5, the timestep instead goes through a dedicated two-layer MLP (`time_mlp_in` -> SiLU -> `time_mlp_out` -> SiLU) and the resulting vector conditions *every* RMSNorm in the Action Expert via adaptive scale, shift, and gate parameters:

```
# pi0.5 path in embed_suffix()
time_emb = posemb_sincos(timestep, width, 4e-3, 4.0)
time_emb = swish(self.time_mlp_in(time_emb))
time_emb = swish(self.time_mlp_out(time_emb))
adarms_cond = time_emb          # passed to every Block
```

Inside `RMSNorm` (in `gemma.py`, lines 113-131), when `cond is not None`, the norm produces:

$$
\text{modulation} = W_{\text{dense}} \cdot \text{cond} \in \mathbb{R}^{3d}
$$

$$
\hat{x} = \text{RMSNorm}(x) \cdot (1 + \text{scale}) + \text{shift}
$$

and also emits a gating vector $g$ used to scale the residual connection:

$$
x_{\text{out}} = x + g \cdot f(x)
$$

where $\text{scale}$, $\text{shift}$, and $g$ are all derived from a single `Dense(3 * width)` applied to `cond`. This mechanism allows every layer of the action expert to be conditioned on the denoising timestep without altering the input token sequence.

Source: `Pi0.__init__()` in `/tmp/openpi/src/openpi/models/pi0.py` (lines 93-99), `RMSNorm.__call__()` in `/tmp/openpi/src/openpi/models/gemma.py` (lines 113-131).

### 1.2.2 Discrete State Input

In pi0, the robot's proprioceptive state vector was projected through `state_proj` into a continuous token that became part of the *suffix* (alongside action tokens). In pi0.5, the state is instead represented as discrete language tokens and becomes part of the *prefix*, handled identically to language input. This is controlled by `Pi0Config.discrete_state_input`, which defaults to `True` when `pi05=True`.

This design change allows the state to participate in full bidirectional attention with image and language tokens in the prefix, rather than being isolated behind a causal boundary in the suffix.

Source: `Pi0.embed_suffix()` in `/tmp/openpi/src/openpi/models/pi0.py` (lines 151-154).

### 1.2.3 Longer Token Length

The maximum tokenized prompt length jumps from 48 to 200, reflecting the extra tokens needed for the discrete state representation and generally longer instructions.

```python
# pi0_config.py __post_init__
if self.max_token_len is None:
    object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
```

Source: `Pi0Config.__post_init__()` in `/tmp/openpi/src/openpi/models/pi0_config.py` (lines 37-41).

---

## 1.3 The `pi05: bool` Flag

All pi0.5-specific behavior is gated by a single field in the configuration dataclass:

```python
@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    pi05: bool = False
```

When `pi05=True`:

| Controlled behavior | `pi05=False` (pi0) | `pi05=True` (pi0.5) |
|---|---|---|
| `model_type` property | `ModelType.PI0` | `ModelType.PI05` |
| `max_token_len` default | 48 | 200 |
| `discrete_state_input` default | `False` | `True` |
| adaRMSNorm in Gemma Module | `adarms=False` | `adarms=True` |
| `use_adarms` per expert | `[False, False]` | `[False, True]` |
| State projection layer | `state_proj` (Linear) | not created |
| Time MLP layers | `action_time_mlp_in/out` | `time_mlp_in/out` |

Note that expert 0 (PaliGemma) *never* uses adaRMSNorm -- only expert 1 (Action Expert) does in pi0.5 mode. In the `Pi0` constructor:

```python
llm = nnx_bridge.ToNNX(
    _gemma.Module(
        configs=[paligemma_config, action_expert_config],
        embed_dtype=config.dtype,
        adarms=config.pi05,
    )
)
llm.lazy_init(rngs=rngs, method="init",
              use_adarms=[False, True] if config.pi05 else [False, False])
```

This flag propagates through the entire model: it changes prefix construction, suffix construction, transformer conditioning, and the `model_type` enum that downstream tooling uses to identify the model.

Source: `/tmp/openpi/src/openpi/models/pi0_config.py` (lines 18-56), `/tmp/openpi/src/openpi/models/pi0.py` (lines 67-100).

---

## 1.4 Parameter Count

pi0.5 comprises three neural sub-networks plus a handful of projection layers.

### SigLIP Vision Encoder (~400M parameters)

- Variant: `So400m/14` (SigLIP So400m with 14x14 patches)
- Width: 1152, depth: 27, MLP dim: 4304, heads: 16
- Pool type: `none` (all spatial tokens retained)
- `num_classes` is set to `paligemma_config.width` (2048), adding a final linear head that projects SigLIP's 1152-dim outputs to the PaliGemma embedding dimension

Source: `Pi0.__init__()` in `/tmp/openpi/src/openpi/models/pi0.py` (lines 81-90), `decode_variant()` in `/tmp/openpi/src/openpi/models/siglip.py` (lines 298-373).

### PaliGemma 2B Backbone (Expert 0)

- Variant: `gemma_2b`
- Width: 2048, depth: 18 layers, 8 heads, 1 KV head, head dim: 256
- MLP dimension: 16,384
- Includes the shared vocabulary embedder (vocab size 257,152 x 2048)

Source: `get_config("gemma_2b")` in `/tmp/openpi/src/openpi/models/gemma.py` (lines 79-87).

### Action Expert ~311M (Expert 1)

- Variant: `gemma_300m`
- Width: 1024, depth: 18 layers, 8 heads, 1 KV head, head dim: 256
- MLP dimension: 4,096
- Comment in source: "311M params"
- Does *not* have its own embedder; shares the PaliGemma embedder for expert 0

Source: `get_config("gemma_300m")` in `/tmp/openpi/src/openpi/models/gemma.py` (lines 69-77).

### Projection Layers

| Layer | Shape | Notes |
|-------|-------|-------|
| `action_in_proj` | $32 \to 1024$ | Projects raw actions to Action Expert width |
| `time_mlp_in` | $1024 \to 1024$ | pi0.5 only |
| `time_mlp_out` | $1024 \to 1024$ | pi0.5 only |
| `action_out_proj` | $1024 \to 32$ | Projects back to action space |

### Total

| Component | Variant | Approximate Parameters |
|---|---|---|
| SigLIP vision encoder | So400m/14 | ~400M |
| PaliGemma backbone (expert 0) | gemma_2b | ~2.0B |
| Action Expert (expert 1) | gemma_300m | ~311M |
| Projection layers | action_in/out_proj, time MLPs | ~2-3M |
| **Total** | | **~2.3B** |

Note: a significant fraction of the PaliGemma 2B parameter count is the shared vocabulary embedding table ($257{,}152 \times 2048 \approx 527M$ parameters), which is used only for the language modality. The "active" parameters during action generation are somewhat smaller than the full 2B figure.

---

## 1.5 Configuration Summary

The canonical pi0.5 configuration at a glance:

```python
Pi0Config(
    dtype                 = "bfloat16",
    paligemma_variant     = "gemma_2b",
    action_expert_variant = "gemma_300m",
    action_dim            = 32,
    action_horizon        = 50,
    max_token_len         = 200,        # set by __post_init__ when pi05=True
    pi05                  = True,
    discrete_state_input  = True,       # set by __post_init__ when pi05=True
)
```

| Parameter | Value | Notes |
|---|---|---|
| `action_dim` | 32 | Degrees of freedom per action step |
| `action_horizon` | 50 | Number of future steps predicted |
| `max_token_len` | 200 (pi0.5) / 48 (pi0) | Max language + state token length |
| `dtype` | `"bfloat16"` | Compute dtype for transformer layers |
| `paligemma_variant` | `"gemma_2b"` | Expert 0 configuration key |
| `action_expert_variant` | `"gemma_300m"` | Expert 1 configuration key |
| `pi05` | `True` | Enables adaRMSNorm and discrete state |
| `discrete_state_input` | `True` (derived) | State as language tokens |
| `pytorch_compile_mode` | `"max-autotune"` | For PyTorch backend compilation |
| Image resolution | $224 \times 224$ | `model.py IMAGE_RESOLUTION` |
| Camera views | 3 (base, left wrist, right wrist) | `model.py IMAGE_KEYS` |

All embedding-level computation uses `bfloat16` (controlled by `config.dtype` which is passed as `embed_dtype` to the Gemma `Module`). Higher-precision float32 is preserved where numerically necessary -- discussed in detail in [`03_data_types_and_observation.md`](./03_data_types_and_observation.md).

---

## Key Takeaways

- pi0.5 is a ~2.3B-parameter VLA model that maps 3 camera images + language + robot state to a continuous action trajectory of shape $[B, 50, 32]$ via flow-matching denoising.
- The `pi05: bool` flag in `Pi0Config` is the single toggle that activates all pi0.5-specific behavior: adaRMSNorm conditioning, discrete state input, and a 200-token prompt length.
- adaRMSNorm replaces the concatenation-based timestep fusion of pi0, injecting the denoising timestep as scale/shift/gate modulation into every transformer layer of the Action Expert.
- The architecture comprises three subsystems -- SigLIP So400m/14 (~400M), PaliGemma 2B, and the Action Expert (~311M) -- all wired through shared attention.
- All Gemma expert weights share depth=18 and the same attention geometry (8 heads, 1 KV head, head_dim=256), differing only in width and MLP dimension.

---

**Next:** [`02_component_map_and_data_flow.md`](./02_component_map_and_data_flow.md)
