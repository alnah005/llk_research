# Chapter 5: Kernel Codegen Pipeline

## What you will learn

After completing this chapter you will be able to:

- Trace the full path from Python `emit()` calls through the shadow graph, four engines, and BlazeCompiler to a device-ready program descriptor.
- Explain how CBEngine, SemEngine, CTArgEngine, and the kernel codegen module each contribute to the final compiled kernel.
- Describe how `kernel_codegen.py` transforms a BlazeGraph into C++ source: PhaseInfo registration, PhaseDecl construction, type alias emission, mcast group formation, lifecycle elision, and content-hashed output.
- Read a generated kernel `.cpp` file and map every line back to the Python op that produced it.
- Understand the named-args generated header system: `ct_args` / `rt_args` namespaces, per-core CT arg descriptors, the triple compilation model, and how to inspect resolved values.
- Identify when and how to write a handwritten kernel, the `kernel` attribute on FusedOp, the required structure, and the shared_expert example.

## How the pieces fit together

```
emit() calls on FusedProgram
        |
        v
  Shadow Graph (BlazeGraph)
        |
        v
  +-----+-----+-----+-----+
  | CB  | Sem | Role| CT  |
  |Eng. |Eng. |Eng. |Arg  |
  |     |     |     |Eng. |
  +-----+-----+-----+-----+
        |
        v
  kernel_codegen.py
        |
        v
  generated_{hash}.cpp
        |
        v
  tt-metal JIT Compiler
        |
        v
  named_args_generated.h   (auto-included via -include)
  BRISC / NCRISC / TRISC binaries
```

When no handwritten kernel is set (`FusedOpConfig.kernel = None`), `FusedProgram.build()` calls `program.set_kernel_from_graph(self._shadow_graph)`, which invokes `generate_kernel_file()` from `kernel_codegen.py`. The generated `.cpp` is content-hashed so identical graphs produce the same file and skip regeneration. The tt-metal JIT compiler then produces `named_args_generated.h` and compiles the kernel three times -- once per RISC processor (NCRISC, BRISC, TRISC).

## Chapter contents

| # | File | Topic |
|---|------|-------|
| 1 | [`01_engine_pipeline.md`](./01_engine_pipeline.md) | The four engines (CBEngine, SemEngine, RoleEngine, CTArgEngine), the shadow graph, and `BlazeCompiler` orchestration |
| 2 | [`02_auto_generated_kernels.md`](./02_auto_generated_kernels.md) | `kernel_codegen.py`: PhaseInfo, PhaseDecl, type aliases, `kernel_main()`, mcast groups, lifecycle elision, hashed output |
| 3 | [`03_named_args_generated_header.md`](./03_named_args_generated_header.md) | The `named_args_generated.h` file: `ct_args::` / `rt_args::` namespaces, per-core CT arg specialization, triple compilation, inspection |
| 4 | [`04_handwritten_kernels.md`](./04_handwritten_kernels.md) | When and how to write a handwritten kernel: the `kernel` attribute, structure, consistency checklist, `shared_expert_kernel.cpp` walkthrough |

## Reading order

1. **Start with the engine pipeline.** Read [`01_engine_pipeline.md`](./01_engine_pipeline.md) to understand how the shadow graph captures every `emit()` call, how the four engines (CB, Semaphore, Role/Grid, CT Arg) process that graph in sequence, and how BlazeCompiler assembles their outputs into a `ProgramDescriptor`.

2. **Study auto-generated kernels.** Read [`02_auto_generated_kernels.md`](./02_auto_generated_kernels.md) to see how `kernel_codegen.py` converts a BlazeGraph into a complete C++ kernel source file. You will learn the PhaseInfo and PhaseDecl data structures, how type aliases are emitted, how mcast groups share persistent sender state via `setup_src<>()`/`run_as<>()`, how empty `init()`/`teardown()` bodies are elided, and how content hashing enables caching.

3. **Understand the named-args header.** Read [`03_named_args_generated_header.md`](./03_named_args_generated_header.md) to learn how compile-time and runtime arguments flow from Python to C++ through generated `ct_args::` and `rt_args::` namespaces. You will see the per-core CT arg system, the triple compilation model (one compile per RISC), and how to inspect resolved values for debugging.

4. **Learn handwritten kernels.** Read [`04_handwritten_kernels.md`](./04_handwritten_kernels.md) to understand when auto-generated kernels are insufficient, how FusedOps specify a handwritten kernel via the `kernel` attribute, the required structure of a handwritten kernel file, and a walk-through of the shared_expert kernel as a concrete example.

## Prerequisites

- Chapters 1-4 of this guide (Tensix core model, op anatomy, CB management, grid system)
- Familiarity with TT-Blaze's `FusedProgram`, `BlazeOp`, `MicroOp`, and `FusedOp` class hierarchy
- Basic C++ knowledge (templates, namespaces, preprocessor guards)

## Key source files

| File | Purpose |
|------|---------|
| `blaze/cb_engine.py` | CBEngine: assigns CB IDs and computes buffer sizes from graph edges |
| `blaze/sem_engine.py` | SemEngine: identifies inter-core sync points and assigns semaphore indices |
| `blaze/role_engine.py` | GridConfig and core layout utilities |
| `blaze/ct_args.py` | CTArgEngine: resolves and validates compile-time arg values |
| `blaze/kernel_codegen.py` | Auto-generates C++ kernel source from a BlazeGraph |
| `blaze/compiler.py` | BlazeCompiler: orchestrates the engine pipeline and JIT compilation |
| `blaze/fused_program.py` | FusedProgram: maintains the shadow graph during `emit()` composition |
