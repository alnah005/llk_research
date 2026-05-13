# 01 -- adaRMSNorm Mechanics

## Context

Every transformer layer in pi0.5 applies normalization before both attention and
the feedforward network. The specific normalization variant depends on the expert:
expert 0 (PaliGemma) uses standard RMSNorm, while expert 1 (Action Expert) uses
adaptive RMSNorm (adaRMSNorm). Both variants are implemented in a single
`RMSNorm` class that branches on whether a conditioning signal is provided. The
difference is not just cosmetic -- adaRMSNorm produces three modulation signals
(scale, shift, gate) that give the action expert a way to adjust its behavior at
every normalization point based on the diffusion timestep. Understanding the
exact math, the initialization strategy, and how the gate feeds into residual
connections is essential for reasoning about both the training dynamics and the
hardware lowering.

---

## 1. Standard RMSNorm (Expert 0, `cond=None`)

When the conditioning signal `cond` is `None`, the class falls through to
standard RMSNorm. Source: `gemma.py:113-125`.

```
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        if cond is None:
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (1 + scale)
            return normed_inputs.astype(dtype), None
```

The math in closed form:

```
var_i      = mean(x_i^2)                       # per-token, across features
normed_i   = x_i * rsqrt(var_i + 1e-6)         # normalize
output_i   = normed_i * (1 + scale)             # affine rescale
```

Where `scale` is a learned parameter of shape `[D]` (1024 for action expert,
2048 for PaliGemma), initialized to all zeros via `nn.initializers.zeros_init()`.

Key properties:
- No mean subtraction (unlike LayerNorm). Only the root-mean-square is used.
- The variance computation is done in float32 regardless of input dtype.
- The `(1 + scale)` formulation means the effective weight starts at 1.0, not 0.0.
  At initialization, `scale = 0` so the layer is an identity-like normalization.
- Returns `(normed_output, None)` -- no gate for the standard path.

## 2. Adaptive RMSNorm (Expert 1, `cond != None`)

When `cond` is provided, the code enters the adaptive path. Source:
`gemma.py:127-131`.

```
        # adaptive RMSNorm
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift
        return normed_inputs.astype(dtype), gate
```

Step by step:

1. **Modulation projection**: A `Dense` layer maps `cond` from `[B, D]` to
   `[B, D*3]`. For the action expert with `D=1024`, this produces `[B, 3072]`.
   The kernel is zero-initialized via `nn.initializers.zeros`.

2. **Unsqueeze**: The `[:, None, :]` indexing reshapes `[B, 3072]` to
   `[B, 1, 3072]`, adding a sequence dimension for broadcasting.

3. **Split**: `jnp.split(..., 3, axis=-1)` divides the 3072-dim vector into
   three tensors of shape `[B, 1, 1024]` each: `scale`, `shift`, `gate`.

4. **Apply modulation**: `normed * (1 + scale) + shift`.

5. **Return**: `(modulated_output, gate)`. The gate is returned separately for
   use in the residual connection.

The math:

```
modulation = Dense_zero(cond)                   # [B, 3*D], zero-init kernel
scale, shift, gate = split3(modulation)         # each [B, 1, D]
output = normed * (1 + scale) + shift           # [B, T, D]
return (output, gate)                           # gate is [B, 1, D]
```

The full data flow as an ASCII diagram:

```
                        cond [B, 1024]
                             |
                    Dense (zero-init)
                     [1024, 3072]
                             |
                      [B, 3072]
                             |
                     unsqueeze dim=1
                             |
                     [B, 1, 3072]
                             |
                 +-----------+-----------+
                 |           |           |
            scale [B,1,D] shift [B,1,D] gate [B,1,D]
                 |           |           |
                 v           v           |
    x [B,T,D]   |           |           |
        |        |           |           |
     RMSNorm     |           |           |
        |        |           |           |
   normed [B,T,D]|           |           |
        |        |           |           |
   normed * (1 + scale)      |           |
              |              |           |
         + shift             |           |
              |              |           |
        output [B,T,D]     gate [B,1,D]  |
              |              +-----------+
              v                    |
       (to attention/FFN)     (to residual)
```

## 3. Zero Initialization: Why adaRMSNorm Starts as Standard RMSNorm

Both normalization paths use zero initialization, but for different reasons:

| Component | Init | Effect at t=0 |
|-----------|------|---------------|
| Standard `scale` | `zeros_init()` | `(1 + 0) = 1`, so output = normed input |
| Adaptive `Dense` kernel | `zeros` | All of scale, shift, gate = 0 |

At training initialization with `Dense` kernel = 0:
- `modulation = cond @ zeros = zeros`
- `scale = 0` --> `normed * (1 + 0) = normed` (identity scaling)
- `shift = 0` --> `normed + 0 = normed` (no shift)
- `gate = 0` --> residual becomes `x + y * 0 = x` (no contribution from sublayer)

This means the adaptive path behaves identically to a standard RMSNorm at
training start, with the additional property that the gate kills all sublayer
outputs. The network begins as if the transformer layers contribute nothing, and
gradually learns to open the gates. This is a well-known technique from the DiT
(Diffusion Transformer) literature for stabilizing training when conditioning
signals are introduced.

## 4. The Gate and Gated Residual Connections

The gate returned by adaRMSNorm is consumed by `_gated_residual` at
`gemma.py:453-459`:

```
def _gated_residual(x, y, gate):
    assert (x is None) == (y is None)
    if x is None:
        return None
    if gate is None:
        return x + y
    return x + y * gate
```

There are two uses per layer -- one for the attention sublayer and one for the
feedforward sublayer. From `Block.__call__` at `gemma.py:293-333`:

```
# Pre-attention norm (produces gate for attention residual)
for i, x in enumerate(xs):
    if x is not None:
        x, gate = RMSNorm(name=...)(x, adarms_cond[i])
    gates.append(gate ...)

# After attention: gated residual
xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates)]

# Pre-FFN norm (produces gate for FFN residual)
for i, (x, config) in enumerate(zip(xs, self.configs)):
    if x is not None:
        x, gate = RMSNorm(name=...)(x, adarms_cond[i])
        ...
    gates.append(gate ...)

# After FFN: gated residual
xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates)]
```

The full block flow with gates:

```
     xs[expert_1]
          |
    +-----+-----+
    |             |
    |     pre_attention_norm(x, cond)
    |         |          |
    |     normed_x     gate_attn [B,1,D]
    |         |
    |     attention
    |         |
    |     post_attn
    |         |
    +----> x + post_attn * gate_attn       <-- gated residual
              |
        +-----+-----+
        |             |
        |     pre_ffw_norm(x, cond)
        |         |          |
        |     normed_x     gate_ffn [B,1,D]
        |         |
        |       FFN
        |         |
        |       out
        |         |
        +----> x + out * gate_ffn          <-- gated residual
                  |
             xs[expert_1] (next layer)
```

For expert 0 (PaliGemma), `gate` is always `None`, so `_gated_residual` falls
through to the standard `x + y`.

## 5. Instance Counting

adaRMSNorm instances in the action expert:

| Location | Count per layer | Layers | Subtotal |
|----------|----------------|--------|----------|
| `pre_attention_norm_1` | 1 | 18 | 18 |
| `pre_ffw_norm_1` | 1 | 18 | 18 |
| `final_norm_1` | 1 | 1 | 1 |
| **Total** | | | **37** |

The suffix `_1` comes from the naming convention at `gemma.py:443-450`: expert 0
gets bare names (e.g., `pre_attention_norm`), expert 1 gets `_1` suffixes. The
`final_norm` instances are created at `gemma.py:382`:

```
self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]
```

And applied at `gemma.py:409-411`:

```
return [
    f(e, a)[0] if e is not None else e
    for f, e, a in zip(self.final_norms, embedded, adarms_cond)
], kv_cache
```

Note that the final norm also receives `adarms_cond`, so it uses the adaptive
path for expert 1. The `[0]` indexing discards the gate since there is no
residual connection after the final norm.

Each of the 37 adaRMSNorm instances has its own `Dense` projection with
independent weights of shape `[1024, 3072]`. That is `37 * 1024 * 3072 =
116,391,936` parameters dedicated solely to timestep modulation.

## 6. Expert Routing: `use_adarms` and `adarms_cond`

The decision of which expert gets adaptive normalization is controlled by two
mechanisms:

**At initialization** (`gemma.py:413-421`):

```
def init(self, use_adarms: Sequence[bool]):
    self(
        ...,
        adarms_cond=[jnp.zeros((1, c.width)) if u else None
                     for u, c in zip(use_adarms, self.configs)],
    )
```

With `use_adarms=[False, True]` for pi0.5, this passes `[None, zeros(1,1024)]`,
which triggers parameter creation for the `Dense` layer in expert 1's norms
while expert 0's norms only create the `scale` parameter.

**At runtime** (`pi0.py:210`):

```
(prefix_out, suffix_out), _ = self.PaliGemma.llm(
    [prefix_tokens, suffix_tokens], ..., adarms_cond=[None, adarms_cond]
)
```

The pattern `[None, adarms_cond]` is a list matching the expert count: `None`
for PaliGemma (standard RMSNorm), the actual conditioning vector for the action
expert (adaptive RMSNorm).

---

## Key Takeaways

- Standard RMSNorm uses `(1 + scale)` with zero-initialized `scale`, producing
  identity normalization at init. It returns `(output, None)`.

- Adaptive RMSNorm projects the conditioning vector through a zero-initialized
  Dense layer into scale, shift, and gate, each of shape `[B, 1, D]`. It
  returns `(output, gate)`.

- Zero initialization ensures both paths start identically: scaling by 1, no
  shift, gate = 0 (killing all sublayer contributions at init).

- The gate modulates residual connections via `x + y * gate`, giving each layer
  a learnable on/off switch for its own output.

- There are 37 adaRMSNorm instances in the action expert (2 per layer across 18
  layers + 1 final norm), each with an independent `[1024, 3072]` weight matrix
  totaling ~116M modulation parameters.
