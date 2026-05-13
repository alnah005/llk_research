# Chapter 5: Flow-Matching Inference and the Denoising Loop

## Overview

This chapter explains how pi0.5 generates robot actions at inference time. The model uses
flow matching -- a continuous-time generative framework -- to transport Gaussian noise into
clean action trajectories through a learned velocity field. Unlike diffusion models that
require hundreds of steps, pi0.5 produces high-quality action sequences in just 10 Euler
integration steps, making it practical for real-time robotic control.

The inference loop is the hottest path in the entire system from a Tenstorrent deployment
perspective: the prefix (vision + language encoding) runs once per observation, but the
denoising loop runs 10 times with different suffix inputs each iteration. Understanding the
exact tensor shapes, caching strategy, and numerical details is essential for writing
correct and performant TT-Metal kernels.

## Chapter Contents

| File | Topic | Key Question Answered |
|------|-------|-----------------------|
| [01_flow_matching_theory.md](01_flow_matching_theory.md) | Flow-Matching Theory | What mathematical framework turns noise into actions? |
| [02_complete_inference_walkthrough.md](02_complete_inference_walkthrough.md) | Complete Inference Walkthrough | What happens tensor-by-tensor inside `sample_actions`? |
| [03_kv_cache_strategy.md](03_kv_cache_strategy.md) | KV Cache Strategy | How does prefix caching avoid redundant computation? |
| [04_pytorch_implementation_differences.md](04_pytorch_implementation_differences.md) | PyTorch Implementation Differences | What changes when moving from JAX to PyTorch/HuggingFace? |

## Reading Order

Read files 01 through 04 in order. File 01 establishes the mathematical foundation. File 02
walks through the actual code path tensor-by-tensor. File 03 zooms in on the KV cache
mechanism that makes the 10-step loop efficient. File 04 catalogs the differences between
the JAX reference and the PyTorch port, which is the starting point for any TT-Metal
implementation.

## Architecture at a Glance

```
                      pi0.5 Inference Pipeline
  ================================================================

  OBSERVATION (images + language prompt)
       |
       v
  +------------------+
  | embed_prefix()   |  <-- Runs ONCE
  | SigLIP + Embedder|
  +------------------+
       |
       v  prefix_tokens [b, ~968, 2048]
  +------------------+
  | Transformer      |  <-- Runs ONCE (prefix-only forward pass)
  | (18 layers)      |
  +------------------+
       |
       v  KV cache [18, b, ~968, 1, 256]  (frozen after this point)
       |
       |   +========== DENOISING LOOP (10 iterations) ==========+
       |   |                                                     |
       |   |  t = 1.0, 0.9, 0.8, ... 0.1                       |
       |   |                                                     |
       |   |  x_t (noisy actions) [b, 50, 32]                   |
       |   |       |                                             |
       |   |       v                                             |
       |   |  +------------------+                               |
       |   |  | embed_suffix()   |                               |
       |   |  | action_in_proj   |                               |
       |   |  | time_mlp (adaRMS)|                               |
       |   |  +------------------+                               |
       |   |       |                                             |
       |   |       v  suffix_tokens [b, 50, 1024]               |
       |   |  +------------------+                               |
       |   |  | Transformer      |  <-- reads prefix KV cache   |
       |   |  | (18 layers)      |      (no new cache written)   |
       |   |  +------------------+                               |
       |   |       |                                             |
       |   |       v  suffix_out [b, 50, 1024]                  |
       |   |  +------------------+                               |
       |   |  | action_out_proj  |                               |
       |   |  +------------------+                               |
       |   |       |                                             |
       |   |       v  v_t [b, 50, 32]                           |
       |   |                                                     |
       |   |  x_t = x_t + (-0.1) * v_t    (Euler step)         |
       |   |  t = t - 0.1                                        |
       |   |                                                     |
       |   +=====================================================+
       |
       v
  x_0 = clean actions [b, 50, 32]
```
