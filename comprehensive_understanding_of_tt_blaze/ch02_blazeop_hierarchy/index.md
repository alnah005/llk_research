# Chapter 2 -- The BlazeOp Class Hierarchy and Op Authoring

This chapter covers the type system that defines every operation in Blaze. It explains how the class hierarchy -- from the abstract `BlazeOp` base through `MicroOp` and `FusedOp` -- provides a unified framework for declaring, registering, and composing ops. It also walks through the C++ header parser, the auto-discovery mechanism, and practical guides for writing new ops of both kinds.

## Key Takeaways

- Every op in Blaze is a Python class. The class itself **is** the op definition: port descriptors declare the interface, `ct_args` declares compile-time arguments, `emit()` defines composition logic, and `compose()` bridges the graph API to the composition API.
- `MicroOp` subclasses are backed by a C++ `Op` struct in `kernels/op.hpp`. The Python class declares identity and ports; everything else -- CT args, phase metadata, lifecycle flags, pop flags -- is auto-derived from the C++ header at registration time.
- `FusedOp` subclasses compose multiple `MicroOp.emit()` calls into a pipeline. They produce no kernel of their own by default; the kernel is either auto-generated from the shadow graph or provided as a handwritten source file.
- The C++ parser (`cpp_parser.py`) is the single source of truth bridge: it reads the kernel header and extracts exactly the metadata that the Python side needs, eliminating redundant manual declarations.
- Auto-discovery (`register_all()`) makes adding a new op as simple as placing a `BlazeOp` subclass in `blaze/ops/<name>/op.py` -- no manual wiring required.

## Reading Order

1. [**BlazeOp Base Class**](./01_blazeop_base.md) -- The foundation: class attributes, port descriptors, CT arg kind types, the `register()` method, and the `graph_call()` / `compose()` entry points.
2. [**MicroOp and FusedOp**](./02_microop_and_fusedop.md) -- The two concrete subclasses, auto-derivation from kernel headers, the separation of `compose()` vs `emit()`, and a Matmul walkthrough.
3. [**The C++ Parser**](./03_cpp_parser.md) -- How `parse_op_hpp()` extracts CT args, RISC mapping, lifecycle detection, and pop flags from kernel headers.
4. [**Auto-Discovery and Registration**](./04_auto_discovery_and_registration.md) -- `register_all()`, the `_OpHandle` proxy, and the op registry.
5. [**Writing a MicroOp**](./05_writing_a_micro_op.md) -- Step-by-step walkthrough with the Mcast op as a concrete example.
6. [**Writing a FusedOp**](./06_writing_a_fused_op.md) -- Step-by-step walkthrough: directory layout, pipeline composition, `child_prefix()` and `cb_name()`, and a SharedExpert preview.

## Files in This Chapter

| File | Description |
|------|-------------|
| [`01_blazeop_base.md`](./01_blazeop_base.md) | BlazeOp base class: attributes, port descriptors, CT arg kinds, registration, and API entry points |
| [`02_microop_and_fusedop.md`](./02_microop_and_fusedop.md) | MicroOp and FusedOp subclasses, auto-derivation, compose/emit separation, Matmul walkthrough |
| [`03_cpp_parser.md`](./03_cpp_parser.md) | C++ header parser: single source of truth, RISC mapping, lifecycle detection, pop flags |
| [`04_auto_discovery_and_registration.md`](./04_auto_discovery_and_registration.md) | Auto-discovery via `register_all()`, `_OpHandle` proxy, global op registry |
| [`05_writing_a_micro_op.md`](./05_writing_a_micro_op.md) | Practical MicroOp authoring guide with the Mcast op as running example |
| [`06_writing_a_fused_op.md`](./06_writing_a_fused_op.md) | Practical FusedOp authoring guide with SharedExpert preview |

---
[Chapter 1 -- Architecture](../ch01_architecture/index.md) | **Index** | [01 -- BlazeOp Base Class](./01_blazeop_base.md) ->
