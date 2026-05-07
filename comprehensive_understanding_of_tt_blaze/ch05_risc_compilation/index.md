# Chapter 5 -- Per-RISC Compilation and Kernel Infrastructure

This chapter covers TT-Blaze's per-RISC compilation model, kernel header infrastructure, and JIT build system. A single `.cpp` kernel source compiles into three distinct RISC-V binaries -- one per processor on each Tensix core. The chapter traces a concrete fused kernel (mcast -> matmul -> gather) through the entire pipeline from Python `emit()` to silicon execution.

## Sections

1. **[Per-RISC Compilation Model](01_per_risc_model.md)** -- The three-RISC hardware architecture, `UnifiedKernelDescriptor`, per-core kernel splitting, and the `BlazeProgram` builder. Includes a complete field reference and the worked example that threads through all sections.

2. **[Kernel Headers and LLK Extensions](02_kernel_headers.md)** -- The `blaze/kernels/` infrastructure layer: RISC detection, grid utilities, cross-RISC synchronization, CB reconfiguration, the per-op `Op` struct pattern, the `sfpu/` directory, and the `kernel_includes/` LLK extension hierarchy.

3. **[Build System and JIT Compilation](03_build_system_and_jit.md)** -- The outer build (`build_blaze.sh`, `env.sh`), JIT compilation from generated `.cpp` to RISC-V ELF binaries, content-hashed caching, per-RISC preprocessing views, and the end-to-end Python-to-silicon flow.

## Cross-Chapter References

- Ch2 S03 (`cpp_parser.py`): Parses kernel `op.hpp` headers to extract CT arg declarations for validation.
- Ch3 S05 (`kernel_codegen.py`): Generates the `.cpp` kernel source from a `BlazeGraph`.
- Ch3 S06 (`BlazeCompiler`): Orchestrates `ProgramDescriptor` assembly from `BlazeProgram.build()` output.
- Ch4 S02 (`FusedProgram`): Declares CT args from Python and wires CB allocations into the program.
