# Chapter 8: Pipeline Architecture, Performance, and Implementation Roadmap

## Context

Previous chapters established the pi0.5 model architecture (dual-expert Gemma backbone with adaRMSNorm, SigLIP vision encoder, flow-matching denoising loop) and the Tenstorrent toolchain (TT-Blaze kernel composition, blaze-nn tracing, TT-Metal dispatch). This chapter bridges the two: how to decompose pi0.5 into a concrete pipeline that runs on Tenstorrent hardware, what the memory and latency constraints look like, what order to build things in, and what can go wrong.

The audience is Tenstorrent ML engineers who need to plan and execute a pi0.5 bring-up. The chapter is structured as four sections that answer four questions:

1. **How do we split the model into pipeline stages?** -- Stage decomposition, orchestration strategies, and comparison with the DeepSeek pipeline already running on TT-Blaze.
2. **Does it fit, and is it fast enough?** -- Memory budget, compute breakdown, bandwidth analysis, and optimization levers.
3. **What do we build first?** -- A phased roadmap from standalone ops to full-model inference.
4. **What will bite us?** -- Technical risks and open questions ranked by likelihood and impact.

## Sections

| File | Title |
|---|---|
| [`01_pipeline_stage_design.md`](./01_pipeline_stage_design.md) | Pipeline Stage Design |
| [`02_memory_and_latency_analysis.md`](./02_memory_and_latency_analysis.md) | Memory and Latency Analysis |
| [`03_implementation_roadmap.md`](./03_implementation_roadmap.md) | Implementation Roadmap |
| [`04_technical_risks_and_open_questions.md`](./04_technical_risks_and_open_questions.md) | Technical Risks and Open Questions |

## Key Takeaways

- Pi0.5 fits on a single Wormhole/Blackhole device at bf16 (~5.6 GB params, ~12 GB DRAM available), but the denoising loop dominates latency and requires aggressive optimization (step reduction, weight caching, op fusion) to hit 10-50 Hz control rates.
- The natural pipeline decomposition is 4 stages: SigLIP encode, Prefix+KV fill, Denoising iterations, Action output. Host-orchestrated looping is the recommended first approach; device-loop and unrolled strategies are optimization paths.
- The implementation roadmap has 5 phases. Phase 1 (standalone ops) and Phase 2 (transformer blocks) derisk the hardest unknowns -- SDPA masking, width-mismatch matmul, and Conv2d patching -- before touching the full model graph.
- Eight technical risks are identified. The top three by combined likelihood and impact are: SDPA with prefix-LM masking on Tensix, the 2048-to-1024 width transition in dual-expert attention, and Conv2d patch extraction for SigLIP.

## Source References

| Path | What it contains |
|---|---|
| `/tmp/openpi/src/openpi/models/pi0.py` | Pi0/Pi0.5 model: `embed_prefix`, `embed_suffix`, `sample_actions` with KV-cache denoising loop |
| `/tmp/openpi/src/openpi/models/pi0_config.py` | Model config: `gemma_2b` + `gemma_300m` experts, action_dim=32, action_horizon=50 |
| `/tmp/openpi/src/openpi/models/gemma.py` | Dual-expert Gemma backbone: `Attention`, `Block`, `Module` with adaRMSNorm |
| `/tmp/openpi/src/openpi/models/siglip.py` | SigLIP So400m/14 vision encoder: 27 layers, width=1152, patch_size=14 |
| `/localdev/salnahari/testing_dir/tt-blaze/pipeline_builder/graph.py` | `PipelineGraph` DAG builder for multi-stage pipelines on TT hardware |
| `/localdev/salnahari/testing_dir/tt-blaze/blaze/compiler.py` | `BlazeCompiler`: graph-to-ProgramDescriptor compilation |
| `/localdev/salnahari/testing_dir/blaze-nn/blaze_nn/functional.py` | blaze-nn functional ops: `linear`, `rmsnorm`, `rope`, `gated_reduce` |
| `/localdev/salnahari/testing_dir/blaze-nn/blaze_nn/_tracing.py` | Tracing context for graph-mode / compose-mode dispatch |
