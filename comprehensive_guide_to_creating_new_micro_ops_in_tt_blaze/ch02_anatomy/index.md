# Chapter 2 -- Anatomy of a Micro-Op

## What you will learn

This chapter dissects the structure of a TT-Blaze micro-op from three perspectives: the filesystem layout that enables auto-discovery, the Python class that defines identity, ports, and composition logic, and the C++ kernel header that the hardware actually executes. A fourth section shows how auto-derivation bridges the two halves, eliminating redundant declarations.

By the end of this chapter you will be able to:

- Describe the canonical three-file layout (`__init__.py`, `op.py`, `kernels/op.hpp`) and explain why each file exists.
- Trace the full auto-discovery and registration pipeline from `register_all()` through the five global registries.
- Distinguish between `op_class` (C++ struct name) and `name` (Python identifier) and explain the precedence rules.
- Read and annotate any existing micro-op's Python class, identifying ports, config fields, `compose()`, and `emit()`.
- Read a C++ `op.hpp` file, identifying CT arg structs, typed fields, the `A::` template pattern, Op lifecycle methods, preprocessor guards, and role flags.
- Distinguish compute-only ops (RMSNorm), data-movement ops (Copy), and inter-core ops (Mcast) by examining their code.
- Explain how `_auto_derive_from_kernel_hpp()` populates class attributes from the header, and when manual overrides are needed.
- Diagnose the most common mistakes: using `uint32_t` instead of `CB`, missing port descriptors, wrong semaphore names, and misplaced kernel headers.

## Prerequisites

- Completion of Chapter 1, especially:
  - Section 1.1.4 (the Tensix core model and RISC processors)
  - Section 1.2.2 (the compilation pipeline from Python to device)
  - Section 1.3 (what a micro-op is and how it relates to FusedOps)

## Chapter structure

| Section | File | Topic |
|---------|------|-------|
| 2.1 | [`01_directory_structure.md`](./01_directory_structure.md) | Directory layout, auto-discovery, registration flow, `name` contract |
| 2.2 | [`02_python_class.md`](./02_python_class.md) | `MicroOp` subclassing, ports, config, `compose()`, `emit()`, walkthroughs |
| 2.3 | [`03_cpp_kernel_header.md`](./03_cpp_kernel_header.md) | `op.hpp` structure, CT arg structs, typed fields, `A::` pattern, lifecycle, guards, walkthroughs |
| 2.4 | [`04_auto_derivation.md`](./04_auto_derivation.md) | `_auto_derive_from_kernel_hpp()`, dataclasses, regex patterns, overrides, edge cases, common mistakes |

---

**Next:** [Section 2.1 -- Directory Structure and Auto-Discovery](./01_directory_structure.md)
