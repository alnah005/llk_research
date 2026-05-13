# 03 -- Data Types and Observation Structure

## Context

Understanding the precise data structures that flow in and out of pi0.5 is essential for any implementation effort, whether in JAX, PyTorch, or on Tenstorrent hardware. This file documents the `Observation` dataclass, the `Actions` type alias, the differences in how pi0 and pi0.5 handle robot state, the detailed Gemma configurations for both experts, and the numerical precision strategy that mixes bfloat16 and float32 throughout the model.

---

## 3.1 The Observation Dataclass

The `Observation` struct is defined as a Flax `struct.dataclass` in `/tmp/openpi/src/openpi/models/model.py` (lines 82-108). It is generic over the array type (`ArrayT`) so it can hold JAX arrays, PyTorch tensors, or NumPy arrays.

```python
@struct.dataclass
class Observation(Generic[ArrayT]):
    images:               dict[str, Float[ArrayT, "*b h w c"]]
    image_masks:          dict[str, Bool[ArrayT, "*b"]]
    state:                Float[ArrayT, "*b s"]
    tokenized_prompt:     Int[ArrayT, "*b l"] | None
    tokenized_prompt_mask: Bool[ArrayT, "*b l"] | None
    token_ar_mask:        Int[ArrayT, "*b l"] | None    # pi0-fast only
    token_loss_mask:      Bool[ArrayT, "*b l"] | None   # pi0-fast only
```

### Field details

| Field | Dtype | Shape | Description |
|-------|-------|-------|-------------|
| `images` | float32 | `[B, 224, 224, 3]` per key | RGB images normalized to $[-1, 1]$. Three keys: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` |
| `image_masks` | bool | `[B]` per key | `True` if the camera image is valid for this batch element, `False` to mask out |
| `state` | float32 | `[B, 32]` | Low-dimensional robot state (action_dim=32). In pi0.5, this field still exists but the state is additionally tokenized into the language prompt |
| `tokenized_prompt` | int32 | `[B, 200]` | Token IDs from the PaliGemma tokenizer. Padded to `max_token_len` |
| `tokenized_prompt_mask` | bool | `[B, 200]` | `True` for real tokens, `False` for padding |

The `from_dict` class method handles conversion from the nested dictionary format produced by data transforms, including automatic conversion of `uint8` images to float32 in $[-1, 1]$.

Source: `Observation` in `/tmp/openpi/src/openpi/models/model.py` (lines 82-137).

---

## 3.2 The Actions Type

Actions are defined as a simple type alias:

```python
Actions = Float[ArrayT, "*b ah ad"]
```

where `ah` is `action_horizon` (50) and `ad` is `action_dim` (32). The concrete dtype is **float32** -- actions are always full-precision, even when the transformer backbone operates in bfloat16. The full shape during inference is:

$$
\texttt{actions} \in \mathbb{R}^{B \times 50 \times 32}, \quad \text{dtype} = \texttt{float32}
$$

During training, the loss is computed as:

$$
\mathcal{L} = \frac{1}{32} \sum_{d=1}^{32} (v_t^{(d)} - u_t^{(d)})^2
$$

averaged over the action dimension, yielding a per-step loss of shape `[B, 50]`.

Source: `Actions` type in `/tmp/openpi/src/openpi/models/model.py` (line 141), `compute_loss()` in `/tmp/openpi/src/openpi/models/pi0.py` (lines 188-214).

---

## 3.3 pi0 vs pi0.5: State Input Handling

This is one of the most consequential architectural differences between the two model versions.

### pi0: Continuous State Suffix Token

In pi0, the robot state vector `obs.state` of shape `[B, 32]` is projected through a linear layer (`state_proj: Linear(32, 1024)`) to produce a single continuous token of shape `[B, 1, 1024]`. This token is prepended to the action tokens in the *suffix*:

```python
# pi0 path in embed_suffix()
state_token = self.state_proj(obs.state)[:, None, :]   # [B, 1, 1024]
tokens.append(state_token)
ar_mask += [True]  # breaks causal boundary
```

The suffix therefore has 51 tokens (1 state + 50 actions).

### pi0.5: Discrete State Language Tokens

In pi0.5, `discrete_state_input=True` causes the state to be converted into text tokens upstream (by the data transforms, not inside the model). These tokens become part of the tokenized prompt and flow through the *prefix* alongside language tokens. The `embed_suffix()` method skips the state projection entirely:

```python
# pi0.5 path in embed_suffix()
if not self.pi05:
    # pi0: add state token here
    state_token = self.state_proj(obs.state)[:, None, :]
    ...
# pi0.5: state is already in the prefix via tokenized_prompt
```

The suffix in pi0.5 has exactly 50 tokens (actions only).

This design choice has important implications:

1. **Prefix length.** The pi0.5 prefix is longer because state information consumes some of the 200-token language budget.
2. **State representation.** Discretization means state values are rounded to text tokens, which may lose some precision but gains the benefit of using the pretrained language model's representations.
3. **Suffix simplicity.** The Action Expert processes only action tokens, with no heterogeneous state token mixed in.

Source: `Pi0.embed_suffix()` in `/tmp/openpi/src/openpi/models/pi0.py` (lines 139-186), `Pi0Config` in `/tmp/openpi/src/openpi/models/pi0_config.py` (lines 29-33).

---

## 3.4 Gemma Expert Configurations

Both experts are instances of the same `gemma.Module` class but with different `Config` parameters. They share the same depth, head count, KV-head count, and head dimension -- a requirement enforced by assertions in the `Attention` module.

```
+------------------------------------------------------+
|              Gemma Configuration Comparison           |
+-------------------+------------------+----------------+
| Parameter         | PaliGemma 2B     | Action Expert  |
|                   | (Expert 0)       | (Expert 1)     |
+-------------------+------------------+----------------+
| width             | 2048             | 1024           |
| depth             | 18               | 18             |
| mlp_dim           | 16384            | 4096           |
| num_heads         | 8                | 8              |
| num_kv_heads      | 1                | 1              |
| head_dim          | 256              | 256            |
| adaRMSNorm        | No               | Yes (pi0.5)    |
| ~Parameters       | ~2B              | ~311M          |
+-------------------+------------------+----------------+
```

### Shared constraints

The `Attention` class asserts (lines 165-167 of `gemma.py`):

```python
assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)
```

This guarantees that Q, K, V tensors from both experts can be concatenated along the sequence dimension for joint attention. The different `width` values only affect the input-side Q/K/V projection shapes and the FFN dimensions.

### Grouped Query Attention (GQA)

Both experts use `num_kv_heads=1` with `num_heads=8`, meaning 8 query heads share a single key-value head. This is classic grouped query attention with a group size of 8, which significantly reduces KV cache memory:

$$
\text{KV cache per layer} = 2 \times B \times S \times 1 \times 256 \times \text{sizeof(dtype)}
$$

where $S$ is the total sequence length (prefix + suffix).

### Naming convention

The first expert (PaliGemma) uses unsuffixed layer names (e.g., `attn`, `mlp`) to maintain compatibility with pretrained PaliGemma checkpoints. The second expert (Action Expert) uses suffixed names (e.g., `attn_1`, `mlp_1`), and its weights are initialized from scratch.

Source: `get_config()` in `/tmp/openpi/src/openpi/models/gemma.py` (lines 58-109), `_name()` helper (lines 443-450).

---

## 3.5 Numerical Precision Strategy

pi0.5 uses a deliberate mixed-precision strategy that balances throughput with numerical stability.

### bfloat16: the default compute dtype

The `Pi0Config.dtype` field is set to `"bfloat16"`. This propagates to:

- **Gemma `Module`**: `embed_dtype=config.dtype` -- all token embeddings and transformer layer computations run in bfloat16.
- **SigLIP**: `dtype_mm=config.dtype` -- attention, MLP, and layer norm intermediate computations use bfloat16 (but see exceptions below).
- **Projection layers** (`action_in_proj`, `time_mlp_in`, `time_mlp_out`, `action_out_proj`): NNX `Linear` layers that follow the module-level dtype.

### float32 preservation: normalization layers

RMSNorm (in `gemma.py`, lines 115-131) explicitly computes the variance in float32:

```python
var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
```

The scaling and shifting operations (`(1 + scale)`, `+ shift`) happen in float32 before the result is cast back to the input dtype. This prevents the loss of small gradients that would occur if RMSNorm ran entirely in bfloat16.

### float32 preservation: sinusoidal embeddings

The `posemb_sincos()` function (in `pi0.py`, lines 48-63) computes timestep embeddings in float32:

```python
fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)         # float32
period = min_period * (max_period / min_period) ** fraction      # float32
sinusoid_input = jnp.einsum("i,j->ij", pos, 1.0 / period * 2 * jnp.pi,
                            precision=jax.lax.Precision.HIGHEST)  # explicit highest precision
return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)
```

The `Precision.HIGHEST` flag ensures the einsum is not downcast. This is critical because the sinusoidal periods span three orders of magnitude ($4 \times 10^{-3}$ to $4.0$), and bfloat16's limited mantissa would alias high-frequency components.

### float32 preservation: SigLIP patch extraction and positional embeddings

SigLIP's `_Module.__call__()` (in `siglip.py`) performs patch extraction via `nn.Conv` in float32 and adds 2D sinusoidal positional embeddings in float32 before casting to `dtype_mm` for the transformer layers:

```python
image = jnp.asarray(image, jnp.float32)           # ensure float32
x = nn.Conv(..., dtype=jnp.float32)(image)         # patch embed in float32
x = x + get_posemb(..., jnp.float32)              # add posemb in float32
x = x.astype(self.dtype_mm)                        # then cast to bfloat16
```

### float32 preservation: attention logits

In the Gemma `Attention` class, the attention logit computation explicitly requests float32:

```python
logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)
```

After softmax (also computed on float32 logits), the probabilities are cast back to the input dtype for the value-weighted sum.

### Summary of precision boundaries

| Operation | Dtype | Reason |
|-----------|-------|--------|
| Token embeddings, FFN, projections | bfloat16 | Throughput |
| RMSNorm variance & normalization | float32 | Numerical stability |
| Sinusoidal timestep embeddings | float32 | Frequency precision |
| SigLIP patch embedding & posemb | float32 | Accuracy of input processing |
| Attention logits & softmax | float32 | Avoiding overflow/underflow |
| RoPE computation | float32 | Trigonometric precision |
| Euler-step update ($x_t + dt \cdot v_t$) | float32 | ODE integration accuracy over 10 steps |
| Actions (input & output) | float32 | Continuous control precision |

---

### Tenstorrent hardware implications

For a TT-Blaze implementation, these precision boundaries map directly to op-level decisions: the primary compute path uses bfloat16 tile format (natively supported on Tensix), while the seven float32 preservation points above each require either a float32 accumulator mode in the MicroOp, host-side computation (viable for sinusoidal embeddings and Euler steps), or explicit dtype promotion within FusedOps. The mixed-precision boundary between RMSNorm variance (float32) and the rest of the attention block (bfloat16) is particularly important for CB allocation -- the `rmsnorm` MicroOp must output float32 intermediates before recasting.

---

## Key Takeaways

- The `Observation` dataclass bundles images (float32 in $[-1,1]$), image masks (bool), state (float32), and tokenized prompt fields (int32 + bool mask), and is generic over the array backend.
- Actions are float32 tensors of shape `[B, 50, 32]` -- always full precision, even when the model backbone runs in bfloat16.
- pi0.5 moves the robot state from a continuous suffix token (pi0) to discrete language tokens in the prefix, simplifying the suffix to contain only action tokens.
- Both Gemma experts share identical attention geometry (depth=18, num_heads=8, num_kv_heads=1, head_dim=256) but differ in embedding width (2048 vs 1024) and MLP width (16384 vs 4096).
- The mixed-precision strategy preserves float32 at five critical boundaries -- normalization, sinusoidal embeddings, patch extraction, attention logits, and RoPE -- while running the bulk of computation in bfloat16 for throughput.

---

**Next:** [Chapter 2 -- SigLIP Vision Encoder](../ch02_siglip_encoder/index.md)
