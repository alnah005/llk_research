# Chapter 1: Architecture Overview and Operating Modes

This chapter introduces the Pi0.5 benchmarking framework -- its purpose, its three-phase pipeline model (vision, text, denoise), the three operating modes (simulation, full socket, hybrid/bulk-only), and how all components compose to form the end-to-end system.

## Files

1. [**Architecture Overview**](./01_architecture_overview.md)
   - What Pi0.5 is: VLA model architecture (SigLIP, Gemma, Euler denoise), asymmetric data flow rationale
   - Essential Tenstorrent hardware concepts (MeshDevice, Tensix cores, NOC, circular buffers, sockets, DistributedContext)
   - The three operating modes with code excerpts from both host runner and device launcher
   - Decision matrix for mode selection (10 criteria)
   - Annotated architecture diagrams for all three modes
   - Mode selection flow in code (config + preprocessor two-axis logic)
   - End-to-end data flow summary with concrete byte counts

2. [**Three-Phase Pipeline Model**](./02_three_phase_pipeline_model.md)
   - `PhaseConfig` struct definition, fields, and derived methods
   - Vision phase: SigLIP encoder timing (4 stages, heterogeneous durations)
   - Text phase: Gemma backbone timing (20 uniform stages)
   - Denoise phase: Euler flow-matching loop (6 stages x 5 iterations)
   - Weighted-average flattening into `PipelineSimulatorConfig` with full numerical derivation
   - Accuracy tradeoffs of the uniform systolic approximation (quantified throughput overestimate)
   - Bottleneck stage calculation and corrected TTFT formulas
   - Hybrid configuration variant walkthrough (single-stage math)

3. [**Build System and Component Map**](./03_build_system_and_component_map.md)
   - Source file map: application code, config files, kernels, library headers, and `src/` implementation files
   - Three-tier CMake build targets (no hardware, Metalium available, direct Metalium)
   - Compile-time feature gating: `PI05_HAS_SOCKETS`, `DS_HAS_METALIUM`, `DS_KERNEL_DIR`
   - Target dependency graph with link libraries and compile definitions
   - Namespace map with PipelineConfig naming collision disambiguation
