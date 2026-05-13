# Chapter 1: pi0.5 Architecture Overview

This chapter provides a comprehensive introduction to Physical Intelligence's pi0.5 vision-language-action (VLA) model. pi0.5 transforms multi-camera images and natural-language instructions into continuous robotic action trajectories using a flow-matching denoiser conditioned via adaptive RMSNorm. We examine the model's purpose, its architectural subsystems, how sensor data flows through the network to produce continuous robot action trajectories, and the data type conventions critical for hardware porting.

The target audience is Tenstorrent ML infrastructure engineers planning to implement pi0.5 on Tenstorrent hardware via TT-Blaze and blaze-nn. Every section includes source-file references with line numbers from the OpenPI codebase to support direct code navigation.

## Contents

1. [**Model Landscape**](./01_model_landscape.md) -- What pi0.5 is, how it differs from pi0, the `pi05: bool` configuration flag, parameter counts across all subsystems (~2.3B total), and the complete `Pi0Config` summary.

2. [**Component Map and Data Flow**](./02_component_map_and_data_flow.md) -- The four major components (SigLIP So400m/14, PaliGemma Gemma-2B, Action Expert Gemma-300M, flow-matching denoiser), their roles, and the complete data flow from camera images and language prompts through prefix/suffix construction, 18 shared-attention transformer layers, and 10 Euler denoising steps to the final `[B, 50, 32]` action output.

3. [**Data Types and Observation**](./03_data_types_and_observation.md) -- The `Observation` dataclass, the `Actions` type alias, pi0.5 vs pi0 state handling, Gemma expert configurations side-by-side, and the bfloat16/float32 mixed-precision strategy with its critical float32 preservation points.

---

**Next:** [`01_model_landscape.md`](./01_model_landscape.md)
