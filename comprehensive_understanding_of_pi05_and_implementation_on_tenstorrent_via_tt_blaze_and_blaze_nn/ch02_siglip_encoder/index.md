# Chapter 2: SigLIP Vision Encoder

## Overview

The SigLIP So400m/14 vision encoder is the visual front-end of the pi0.5 model. It processes raw camera images into dense spatial token sequences that the PaliGemma language model backbone consumes as prefix tokens. Three cameras (base, left wrist, right wrist) are each processed independently through a shared-weight SigLIP encoder, producing 768 total image tokens (3 cameras x 256 tokens each) that form the visual context for the flow-matching policy.

SigLIP is architecturally distinct from the Gemma transformer blocks used elsewhere in pi0.5. It uses LayerNorm (not RMSNorm), GELU activation (not SwiGLU), standard multi-head attention (not GQA), sinusoidal positional embeddings (not RoPE), and full bidirectional attention (not causal masking). These differences mean that the TT-Blaze operator set developed for Gemma inference cannot be reused directly -- several new ops or op adaptations are required.

This chapter covers the SigLIP architecture in detail and analyzes the engineering work needed to bring it onto Tenstorrent hardware via TT-Blaze.

## Contents

| File | Title | Description |
|------|-------|-------------|
| [01_siglip_architecture.md](01_siglip_architecture.md) | SigLIP Architecture Deep Dive | Configuration, patch embedding, positional encoding, encoder block structure, pooling, and projection head |
| [02_siglip_to_ttblaze.md](02_siglip_to_ttblaze.md) | Mapping SigLIP to TT-Blaze | Op gap analysis, decomposition strategies, pipeline staging options, and memory planning |

## Key Source Files

| File | Purpose |
|------|---------|
| `/tmp/openpi/src/openpi/models/siglip.py` | SigLIP ViT implementation (JAX/Flax Linen) |
| `/tmp/openpi/src/openpi/models/pi0.py` | Pi0 model class -- shows how SigLIP is instantiated and called |
| `/tmp/openpi/src/openpi/models/pi0_config.py` | Configuration showing variant selection and camera inputs |
| `/localdev/salnahari/testing_dir/blaze-nn/blaze_nn/functional.py` | Existing blaze-nn functional ops (gap analysis baseline) |
