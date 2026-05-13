# Chapter 7: FusedOp Design and blaze-nn Expression

## Context

Pi0.5 is a dual-expert transformer for robotic action generation. Its backbone is a
PaliGemma VLM (2B params, width=2048) paired with an action expert (300M params,
width=1024) that uses **adaptive RMSNorm (adaRMSNorm)** to inject diffusion timestep
conditioning. The two experts share a single SDPA (self-attention) block per layer,
then diverge for their respective FFNs. Porting this architecture to Tenstorrent
hardware via TT-Blaze requires expressing the dual-expert pattern as FusedOps --
fused kernel programs that chain MicroOps on-device without host round-trips.

This chapter walks through the complete FusedOp design for Pi0.5: from individual
ops like adaRMSNorm, through the dual-expert block decomposition, up to the
full module hierarchy, weight loading pipeline, and inference orchestration.

## Contents

| File | Topic |
|------|-------|
| [01_adarmsnorm_fusedop.md](01_adarmsnorm_fusedop.md) | adaRMSNorm FusedOp: composed vs fused kernel, CBHandle chain, L1 budget |
| [02_dual_expert_block_fusedop.md](02_dual_expert_block_fusedop.md) | Dual-expert block decomposition: Option A/B/C, CB flow, width mismatch |
| [03_module_hierarchy_and_forward_pass.md](03_module_hierarchy_and_forward_pass.md) | Complete Pi05Module hierarchy, existing and new F.* ops |
| [04_weight_loading_and_conversion.md](04_weight_loading_and_conversion.md) | HF safetensors to TT-Blaze: name mappings, transposes, WeightProvider |
| [05_inference_orchestration.md](05_inference_orchestration.md) | Two-phase inference: prefix-once + denoising loop, None-expert pattern |

## Key Takeaways

1. **adaRMSNorm is three outputs from one norm** -- scale, shift, gate -- produced by a
   Dense(width*3) modulation projection. The fused kernel chains RMSNorm + matmul +
   split + element-wise apply in a single device program, keeping all intermediates in
   L1 circular buffers.

2. **The dual-expert block splits naturally into three FusedOps**: DualExpertPreAttn
   (parallel pre-attention norms + QKV projections), SharedSDPA (single shared
   attention), and DualExpertPostAttn (parallel post-attention norms + FFNs +
   residual adds). This is Option A -- the recommended decomposition.

3. **Width mismatch (2048 vs 1024)** between the VLM and action expert is resolved at
   the SDPA boundary: QKV projections produce the same head_dim=256 regardless of
   input width, and the concatenated sequence is split back post-attention before
   entering width-specific output projections.

4. **blaze-nn's Module/functional API** mirrors PyTorch exactly -- engineers write
   `F.rmsnorm()`, `F.linear()`, `F.rope()` inside `forward()`, and the tracing
   context dispatches to either GraphTracingContext (auto-fused) or
   ComposeTracingContext (manual composition).

5. **Weight loading** uses a Pi05WeightProvider that extends the existing
   WeightProvider ABC. HuggingFace safetensor keys map to blaze parameter names via
   a deterministic naming convention derived from the `_name(name, i)` pattern in
   the reference Gemma implementation.

## Source References

- `blaze_nn/module.py` -- Module base class with `_call_graph` / `_call_compose` paths
- `blaze_nn/functional.py` -- F.linear, F.rmsnorm, F.rope, F.gated_reduce, etc.
- `blaze_nn/_tracing.py` -- GraphTracingContext / ComposeTracingContext
- `blaze/blaze_op.py` -- BlazeOp / FusedOp / MicroOp class hierarchy
- `blaze/fused_program.py` -- FusedProgram, CBHandle, OverlappedView
- `blaze/cb_handle.py` -- CBHandle with FIFO / DIRECT_ADDRESS access modes
- `blaze/weight_provider.py` -- WeightProvider ABC, StateDictWeightProvider
- `blaze/ops/rmsnorm/op.py` -- RMSNorm MicroOp (base for adaRMSNorm)
- `openpi/models/pi0.py` -- Reference Pi0/Pi0.5 implementation (JAX/Flax)
- `openpi/models/gemma.py` -- Dual-expert Gemma backbone with adaRMSNorm
- `openpi/models/pi0_config.py` -- Pi0Config with pi05 flag
