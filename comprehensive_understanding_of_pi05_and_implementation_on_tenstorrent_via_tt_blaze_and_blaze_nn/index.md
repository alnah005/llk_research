# Comprehensive Understanding of pi0.5 and Implementation on Tenstorrent via TT-Blaze and blaze-nn

## Overview

This guide provides a complete technical reference for implementing Physical Intelligence's [pi0.5](https://github.com/Physical-Intelligence/openpi) vision-language-action (VLA) model on Tenstorrent hardware using [TT-Blaze](https://github.com/tenstorrent/tt-blaze) and [blaze-nn](https://github.com/tenstorrent/blaze-nn).

pi0.5 is a ~2.3B parameter model that maps camera images, natural-language instructions, and robot state to continuous action trajectories of shape `[B, 50, 32]` via flow-matching denoising. It couples three pretrained subsystems -- SigLIP So400m/14 (~400M), PaliGemma 2B, and an Action Expert (~311M) -- through a dual-expert shared-attention transformer with adaptive RMSNorm conditioning.

## Audience

Tenstorrent ML infrastructure engineers and kernel developers. See [plan.md](plan.md) for detailed prerequisite knowledge and learning objectives.

## Chapters

| # | Chapter | Directory | Focus |
|---|---------|-----------|-------|
| 1 | [Architecture Overview](ch01_architecture_overview/index.md) | `ch01_architecture_overview/` | What pi0.5 is, pi0 vs pi0.5 differences, component map, data flow, data types, precision strategy |
| 2 | [SigLIP Vision Encoder](ch02_siglip_encoder/index.md) | `ch02_siglip_encoder/` | SigLIP So400m/14 internals, patch extraction, ViT layers, TT-Blaze mapping |
| 3 | [Dual-Expert Shared-Attention Transformer](ch03_dual_expert_transformer/index.md) | `ch03_dual_expert_transformer/` | Expert configs, shared Q/K/V attention, prefix-LM masking, GeGLU FFN, gated residuals |
| 4 | [Adaptive RMSNorm (adaRMSNorm)](ch04_adarmsnorm/index.md) | `ch04_adarmsnorm/` | adaRMSNorm mechanics, timestep embedding pipeline, TT-Blaze op design |
| 5 | [Flow-Matching Inference and Denoising Loop](ch05_flow_matching/index.md) | `ch05_flow_matching/` | Flow-matching theory, inference walkthrough, KV cache strategy, JAX vs PyTorch differences |
| 6 | [Mapping pi0.5 Ops to TT-Blaze](ch06_op_mapping/index.md) | `ch06_op_mapping/` | REUSE/ADAPT/NEW op classification, DeepSeek V3 and GLM-5.1 transferable patterns |
| 7 | [FusedOp Design and blaze-nn Expression](ch07_fusedop_and_blazenn/index.md) | `ch07_fusedop_and_blazenn/` | adaRMSNorm FusedOp, dual-expert block design, Module hierarchy, weight loading, inference orchestration |
| 8 | [Pipeline Architecture, Performance, and Roadmap](ch08_pipeline_performance_roadmap/index.md) | `ch08_pipeline_performance_roadmap/` | Pipeline stage design, memory/latency analysis, implementation roadmap, risks and open questions |

## Reading Order

Chapters are designed to be read sequentially. Chapters 1-5 cover pi0.5's architecture and inference mechanics. Chapters 6-8 address the Tenstorrent implementation, building on the architectural understanding from earlier chapters.

For readers who already understand pi0.5's architecture, Chapters 6-8 can be read standalone with occasional back-references to earlier chapters for specific details.

## Key Architectural Facts

- **Model size:** ~2.3B parameters (SigLIP ~400M + PaliGemma ~2B + Action Expert ~311M + projections ~2M)
- **Input:** 3 camera images (224x224), language instruction (up to 200 tokens), robot state (discretized into language tokens)
- **Output:** `[B, 50, 32]` float32 action trajectory via 10-step Euler denoising
- **Transformer:** 18 shared-attention layers, dual-expert (PaliGemma width=2048, Action Expert width=1024), shared geometry (8 heads, 1 KV head, head_dim=256)
- **Conditioning:** adaRMSNorm injects denoising timestep as scale/shift/gate modulation into every Action Expert layer
- **Precision:** bfloat16 compute with float32 at normalization, attention logits, sinusoidal embeddings, RoPE, and Euler integration
- **KV cache:** Prefix (~968 tokens) computed once and cached; suffix (50 action tokens) recomputed 10x during denoising

## Source Code References

- **OpenPI (pi0.5 reference):** [github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- **TT-Blaze:** `/localdev/salnahari/testing_dir/tt-blaze/`
- **blaze-nn:** `/localdev/salnahari/testing_dir/blaze-nn/`
