# 5.1 Flow-Matching Theory

## Context

This file explains the mathematical framework behind pi0.5's action generation. Flow
matching is a generative modeling approach that learns a velocity field to transport
samples from a noise distribution to a data distribution along straight-line paths. For
Tenstorrent engineers, understanding this theory clarifies why inference uses exactly 10
Euler steps, why the time convention matters for numerical stability, and what each
intermediate tensor represents.

---

## The Core Idea: Noise-to-Data Transport

Flow matching defines a continuous path between two distributions:
- At t=1: pure Gaussian noise N(0, I)
- At t=0: clean action data from the training distribution

The model learns a velocity field v(x_t, t) that tells you which direction to move at any
point along this path. Starting from a noise sample at t=1, you follow the velocity field
backward in time to arrive at a clean action sample at t=0.

```
  t=1 (noise)                                          t=0 (clean)
  x_1 ~ N(0,I)                                         x_0 = actions
    |                                                      ^
    |    v(x_t, t)    v(x_t, t)    v(x_t, t)             |
    +-------->---------->---------->--------> ... --------+
         dt=-0.1      dt=-0.1      dt=-0.1
         step 1       step 2       step 3          step 10
```

### Time Convention Warning

The pi0.5 codebase uses the convention common in the diffusion literature, which is
**opposite** to the pi0 paper:

| Convention | t=0 means | t=1 means |
|------------|-----------|-----------|
| pi0 paper  | noise     | clean     |
| Codebase   | clean     | clean     |

The code includes an explicit comment about this:

> "note that we use the convention more common in diffusion literature, where t=1 is
> noise and t=0 is the target distribution. yes, this is the opposite of the pi0 paper,
> and I'm sorry."

Source: `pi0.py`, line 227-228.

---

## The Interpolation Path

During training, the model constructs noisy actions x_t by linearly interpolating between
the clean actions (a) and random noise (n):

```
  x_t = t * n + (1 - t) * a
```

At t=0: x_0 = a (pure clean actions).
At t=1: x_1 = n (pure noise).

The **ground-truth velocity** u_t is the tangent to this straight-line path:

```
  u_t = d(x_t)/dt = n - a
```

This is constant along the path -- the velocity does not depend on t. The model's job is to
predict this velocity from x_t and t alone (without seeing n or a separately).

Source: `pi0.py`, lines 196-200.

```python
# Training: construct interpolation and target velocity
noise = jax.random.normal(noise_rng, actions.shape)
time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
time_expanded = time[..., None, None]
x_t = time_expanded * noise + (1 - time_expanded) * actions     # interpolation
u_t = noise - actions                                             # target velocity
```

---

## Training Objective: MSE on Velocity

The training loss is the mean squared error between the predicted velocity v_t and the
ground-truth velocity u_t:

```
  L = E_t [ || v_t(x_t, t) - u_t ||^2 ]
```

In code, this is computed per-action-dimension and averaged:

```python
# pi0.py, line 214
return jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

The PyTorch implementation uses the same loss but through `F.mse_loss`:

```python
# pi0_pytorch.py, line 374
return F.mse_loss(u_t, v_t, reduction="none")
```

Note the argument order: PyTorch's `mse_loss(input, target)` takes prediction first, but
since MSE is symmetric this does not affect the gradient.

---

## Time Sampling: Beta(1.5, 1) Distribution

Training does not sample t uniformly from [0, 1]. Instead it uses a Beta(1.5, 1)
distribution, shifted to avoid the exact boundaries:

```
  t ~ Beta(1.5, 1) * 0.999 + 0.001
```

This produces a distribution biased toward larger t values (noisier samples):

```
  Beta(1.5, 1) PDF:

  P(t)
  2.0 |                                                  *
      |                                              ****
  1.5 |                                          ****
      |                                      ****
  1.0 |                                  ****
      |                           *******
  0.5 |                   ********
      |          *********
  0.0 +**********---------+--------+--------+--------+---
      0.0       0.2       0.4      0.6      0.8      1.0
                              t
```

The 0.999/0.001 scaling ensures t is never exactly 0 or 1, avoiding degenerate edge cases
where x_t would be pure noise or pure signal with zero gradient information.

**Why bias toward noise?** The noisier regime (t close to 1) is where the velocity field
needs to "steer" the most aggressively. Getting the high-noise predictions right is more
important for trajectory quality than fine-tuning the near-clean predictions.

Source: `pi0.py`, line 197; `pi0_pytorch.py`, lines 45-49, 183-185.

```python
# JAX version
time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001

# PyTorch version
def sample_time(self, bsize, device):
    time_beta = sample_beta(1.5, 1.0, bsize, device)
    time = time_beta * 0.999 + 0.001
    return time.to(dtype=torch.float32, device=device)
```

---

## Inference: 10-Step Euler Integration

At inference time, there is no ground truth -- the model must generate actions from pure
noise. It does this by numerically integrating the learned velocity field from t=1 to t=0
using Euler's method:

```
  x_{t+dt} = x_t + dt * v_t(x_t, t)
```

With num_steps=10 and dt=-0.1:

```
  Step  |  t_start  |  dt    |  t_end
  ------+-----------+--------+-------
    1   |   1.0     | -0.1   |  0.9
    2   |   0.9     | -0.1   |  0.8
    3   |   0.8     | -0.1   |  0.7
    4   |   0.7     | -0.1   |  0.6
    5   |   0.6     | -0.1   |  0.5
    6   |   0.5     | -0.1   |  0.4
    7   |   0.4     | -0.1   |  0.3
    8   |   0.3     | -0.1   |  0.2
    9   |   0.2     | -0.1   |  0.1
   10   |   0.1     | -0.1   |  0.0
```

The loop terminates when `time >= -dt/2`, i.e., when `time >= 0.05`. This is a
floating-point-safe check that ensures exactly 10 iterations execute.

Source: `pi0.py`, lines 228-278.

```python
# pi0.py, lines 228-278
dt = -1.0 / num_steps                              # dt = -0.1

def step(carry):
    x_t, time = carry
    # ... embed suffix, forward pass, get v_t ...
    return x_t + dt * v_t, time + dt                # Euler update

def cond(carry):
    x_t, time = carry
    return time >= -dt / 2                          # robust to float error

x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
```

The PyTorch implementation uses a Python `while` loop instead of `jax.lax.while_loop`:

```python
# pi0_pytorch.py, lines 402-419
dt = -1.0 / num_steps
dt = torch.tensor(dt, dtype=torch.float32, device=device)

x_t = noise
time = torch.tensor(1.0, dtype=torch.float32, device=device)
while time >= -dt / 2:
    expanded_time = time.expand(bsize)
    v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
    x_t = x_t + dt * v_t
    time += dt
return x_t
```

---

## Timestep Encoding: Sinusoidal Positional Embedding

The scalar timestep t must be projected into a high-dimensional embedding for the
transformer to condition on. pi0.5 uses sinusoidal positional encoding with log-spaced
periods tuned for the [0, 1] range:

```
  periods = min_period * (max_period / min_period) ^ linspace(0, 1, dim/2)
  sinusoid_input = t / period * 2 * pi
  time_emb = [sin(sinusoid_input), cos(sinusoid_input)]
```

Parameters: min_period=4e-3, max_period=4.0, dimension = action_expert_config.width (1024
for Gemma-300M).

This maps a single scalar t into a 1024-dimensional vector, which is then processed by
the time MLP for adaRMS conditioning.

Source: `pi0.py`, lines 48-63 (JAX); `pi0_pytorch.py`, lines 26-42 (PyTorch).

```
  Timestep Encoding Pipeline (pi0.5 / adaRMS path):

  t (scalar per batch)
  |
  v
  posemb_sincos(t, dim=1024, min_period=4e-3, max_period=4.0)
  |
  v  time_emb [b, 1024]
  |
  +-> time_mlp_in (Linear 1024 -> 1024)
  |
  +-> swish activation
  |
  +-> time_mlp_out (Linear 1024 -> 1024)
  |
  +-> swish activation
  |
  v  adarms_cond [b, 1024]
```

The adarms_cond vector modulates the RMSNorm in every transformer layer of the action
expert. This is how the model knows "how noisy" the current input is.

Source: `pi0.py`, lines 160-169; `gemma.py`, lines 113-131.

---

## Why 10 Steps Is Enough

Flow matching with straight-line interpolation paths produces a velocity field that is
nearly constant along each path. This means the Euler integration is nearly exact even
with few steps. Contrast this with DDPM-style diffusion models that require 50-1000 steps
because their trajectories are curved.

The 10-step budget translates to 10 transformer forward passes through the action expert
per observation. This is the primary compute bottleneck for real-time control and the
critical path for Tenstorrent optimization.

---

## Key Takeaways

1. **Flow matching learns a velocity field v(x_t, t) to transport noise to data along
   straight lines.** The training objective is MSE between predicted and ground-truth
   velocities: `L = ||v_t - (n - a)||^2`.

2. **Convention: t=1 is noise, t=0 is clean.** This is opposite to the pi0 paper but
   matches the codebase. All time-related code uses this convention.

3. **Time sampling uses Beta(1.5, 1) * 0.999 + 0.001.** This biases training toward
   noisier samples and avoids boundary degeneracies.

4. **Inference uses 10 Euler steps with dt=-0.1.** Each step requires a full forward pass
   through the action expert transformer, making this the compute-dominant loop.

5. **The timestep is encoded as a 1024-dim sinusoidal embedding**, then processed through
   a 2-layer MLP to produce the adaRMS conditioning signal that modulates every
   transformer layer.
