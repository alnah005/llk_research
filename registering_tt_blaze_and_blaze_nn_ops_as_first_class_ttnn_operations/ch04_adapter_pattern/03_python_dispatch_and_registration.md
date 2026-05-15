# 4.3 Python Dispatch and Registration

The C++ adapter template from [Section 4.2](02_blaze_device_operation_adapter.md) provides the type-level plumbing. But the adapter is inert until two things happen on the Python side: (1) Blaze's `_run_program()` must route dispatches to the per-op `ttnn.blaze.*` callables instead of `ttnn.generic_op`, and (2) the C++ registrations must be exposed to Python through nanobind bindings in a `ttnn.blaze` module. This section designs both pieces, explores three auto-generation approaches that keep per-op cost near zero on the Python side as well, addresses nanobind binding mechanics with `.pyi` type stubs for IDE support, specifies the backward-compatibility contract for every affected interface, provides concrete test functions, and closes with an end-to-end registration walkthrough.

---

## 4.3.1 Modified `_run_program()` Dispatch

Today, Blaze's `_run_program()` unconditionally dispatches through `generic_op`:

```python
# EXISTING: blaze/compiler.py (simplified)
class MeshCompiledProgram:
    def _run_program(self, io_tensors, mesh_program_descriptor):
        ttnn.generic_op(io_tensors, mesh_program_descriptor)
```

The modified version resolves the dispatch function **once at compile time** (in `MeshCompiledProgram.__init__`), not on every `_run_program()` call. This avoids repeated `getattr` lookups on the hot path:

```python
# PROPOSED: blaze/compiler.py (modified)
import ttnn

_BLAZE_ADAPTER_AVAILABLE = hasattr(ttnn, "blaze")

class MeshCompiledProgram:
    def __init__(self, op_name: str | None, mesh_program_descriptor, ...):
        self._op_name = op_name
        self._mesh_program_descriptor = mesh_program_descriptor
        self._dispatch_fn = self._resolve_dispatch(op_name)
        # ... existing initialization ...

    @staticmethod
    def _resolve_dispatch(op_name: str | None):
        """
        Resolve the dispatch function at compile time, not per-call.

        Returns the ttnn.blaze.{op_name} callable if available,
        or ttnn.generic_op as the fallback.
        """
        if _BLAZE_ADAPTER_AVAILABLE and op_name is not None:
            fn = getattr(ttnn.blaze, op_name, None)
            if fn is not None:
                return fn
        return ttnn.generic_op  # fallback: unchanged behavior

    def _run_program(self, io_tensors, mesh_program_descriptor):
        self._dispatch_fn(io_tensors, mesh_program_descriptor)
```

### Design Decisions

**Pre-resolution at compile time.** The dispatch function is stored as `self._dispatch_fn` during `MeshCompiledProgram.__init__`. Each subsequent `_run_program()` call is a direct function call with no lookup overhead. The cost of resolution is amortized across all invocations of the compiled program.

**The `op_name` comes from the Blaze op definition.** Each `BlazeOp` subclass already has a `name` class attribute (e.g., `Matmul.name = "matmul"`, `RMSNorm.name = "rmsnorm"`). The compiler passes this name through the compilation pipeline:

```python
# PROPOSED: blaze/compiler.py (in BlazeCompiler.compile())
class BlazeCompiler:
    def compile(self, op, tensors, output, user_args):
        # ... existing compilation pipeline ...
        mesh_descriptor = self._build_mesh_descriptor(...)
        return MeshCompiledProgram(
            op_name=op.name,      # <-- NEW: pass op name
            mesh_descriptor=mesh_descriptor,
            io_tensors=io_tensors,
            ...)
```

**Graceful degradation.** If `ttnn.blaze` does not exist (e.g., running against an older TT-Metal build without the adapter), `_BLAZE_ADAPTER_AVAILABLE` is `False` at import time. All ops use `generic_op` with zero overhead. No exception is raised.

### Call Signature Compatibility

The adapter's `invoke()` accepts the same `(io_tensors, mesh_program_descriptor)` base signature as `GenericOp::invoke()`, plus an optional `custom_program_hash` parameter. The Python dispatch path sends the same two positional arguments regardless of which backend is selected:

```python
# Both paths accept the same arguments:
ttnn.generic_op(io_tensors, descriptor)        # existing
ttnn.blaze.matmul(io_tensors, descriptor)      # new (identical args)
```

This means `_run_program()` can switch between backends by simply changing the callable -- no argument restructuring is needed.

---

## 4.3.2 The `ttnn.blaze` Namespace

Registered adapter ops are exposed through a `ttnn.blaze` nanobind submodule:

```python
# After import, the following are available:
ttnn.blaze.matmul(io_tensors, descriptor)
ttnn.blaze.copy(io_tensors, descriptor)
ttnn.blaze.reduce(io_tensors, descriptor)
ttnn.blaze.rmsnorm(io_tensors, descriptor)
ttnn.blaze.sdpa(io_tensors, descriptor)
# ... all registered ops ...
```

### Binding Helper Template

Each op binding follows the same pattern. A C++ helper function reduces per-op binding boilerplate:

```cpp
// PROPOSED: ttnn/cpp/ttnn/operations/blaze/blaze_nanobind.hpp
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include "ttnn/cpp/ttnn-nanobind/decorators.hpp"

namespace nb = nanobind;

namespace ttnn::operations::blaze {

template <typename RegisteredOp>
void bind_blaze_op(nb::module_& mod, const RegisteredOp& op,
                   const std::string& doc) {
    ttnn::bind_registered_operation(
        mod, op, doc,
        ttnn::nanobind_overload_t{
            [](const std::decay_t<RegisteredOp>& self,
               const std::vector<Tensor>& io_tensors,
               const tt::tt_metal::experimental::MeshProgramDescriptor&
                   mesh_program_descriptor,
               std::optional<tt::stl::hash::hash_t> custom_program_hash)
                -> Tensor {
                return self(io_tensors, mesh_program_descriptor,
                            custom_program_hash);
            },
            nb::arg("io_tensors"),
            nb::arg("mesh_program_descriptor"),
            nb::arg("custom_program_hash") = nb::none()});
}

}  // namespace ttnn::operations::blaze
```

The `custom_program_hash` parameter defaults to `None` in Python, mapping to `std::nullopt` in C++. Callers that do not need Tier 2 hashing pass only two arguments -- the same calling convention as `ttnn.generic_op`.

### Nanobind Module Registration

```cpp
// PROPOSED: ttnn/cpp/ttnn/operations/blaze/blaze_ops_nanobind.cpp
#include "blaze_nanobind.hpp"
#include "blaze_op_registrations.hpp"

namespace nb = nanobind;

void bind_blaze_operations(nb::module_& parent_mod) {
    auto blaze_mod = parent_mod.def_submodule("blaze",
        "Blaze operations registered as first-class TTNN ops");

    // Reuse the X-macro list for binding generation
    #define BIND_BLAZE_OP_FROM_LIST(op_name, tag_name) \
        ::ttnn::operations::blaze::bind_blaze_op(      \
            blaze_mod, ttnn::blaze::op_name, "Blaze " #tag_name);

    BLAZE_OP_LIST(BIND_BLAZE_OP_FROM_LIST)

    #undef BIND_BLAZE_OP_FROM_LIST
}
```

The `BLAZE_OP_LIST` X-macro (from [Section 4.2.6](02_blaze_device_operation_adapter.md)) drives nanobind bindings, tag structs, and registrations from a single source of truth.

### Properties Exposed to Python

Each bound operation automatically gets the properties defined by `bind_registered_operation()` ([Section 1.2](../ch01_ttnn_registration_contract/02_registration_dispatch_and_bindings.md)):

```python
>>> ttnn.blaze.matmul.name
'matmul'
>>> ttnn.blaze.matmul.python_fully_qualified_name
'ttnn.blaze.matmul'
>>> ttnn.blaze.matmul.__ttnn_operation__
True
```

The `__ttnn_operation__` property is critical: it allows framework code (graph tracing, automatic differentiation, op sweeps) to detect that `ttnn.blaze.matmul` is a registered TTNN operation, not an arbitrary Python callable.

---

## 4.3.3 Auto-Generation Approaches

Writing tag structs, registrations, and nanobind bindings by hand for 112+ ops is tedious and error-prone. Three strategies address this.

### Approach 1: Build-Time Python Codegen Script

A Python script runs at build time, iterates the Blaze op registry, and emits C++ source files:

```python
# PROPOSED: tools/generate_blaze_registrations.py
"""
Build-time script that generates:
  1. Tag structs + register_operation<> calls
  2. Nanobind binding calls
  3. .pyi type stubs
from the Blaze op registry.
"""
import sys

def generate(output_path: str):
    from blaze.blaze_op import BlazeOp

    ops = []
    for name, op_class in BlazeOp._class_registry.items():
        tag_name = name.replace("_", " ").title().replace(" ", "")
        ops.append({
            "op_name": name,
            "tag_name": tag_name,
            "class_name": op_class.__name__,
            "is_fused": hasattr(op_class, "compose"),
        })

    # Generate C++ header with tags + registrations
    with open(f"{output_path}/generated_blaze_registrations.hpp", "w") as f:
        f.write("#pragma once\n")
        f.write('#include "blaze_device_operation_adapter.hpp"\n')
        f.write('#include "ttnn/api/ttnn/decorators.hpp"\n\n')

        f.write("namespace ttnn::operations::blaze {\n")
        for op in ops:
            f.write(f'struct {op["tag_name"]}Tag {{\n')
            f.write(f'    static constexpr const char* name = '
                    f'"blaze::{op["op_name"]}";\n')
            f.write(f'}};\n')
        f.write("}  // namespace ttnn::operations::blaze\n\n")

        f.write("namespace ttnn::blaze {\n")
        for op in ops:
            f.write(
                f'constexpr auto {op["op_name"]} = '
                f'::ttnn::register_operation<\n'
                f'    "ttnn::blaze::{op["op_name"]}",\n'
                f'    ::ttnn::operations::blaze::BlazeDeviceOperationAdapter<\n'
                f'        ::ttnn::operations::blaze::{op["tag_name"]}Tag>>();\n'
            )
        f.write("}  // namespace ttnn::blaze\n")

    # Generate nanobind source
    with open(f"{output_path}/generated_blaze_nanobind.cpp", "w") as f:
        f.write('#include "generated_blaze_registrations.hpp"\n')
        f.write('#include "blaze_nanobind.hpp"\n\n')
        f.write("void bind_generated_blaze_ops(nb::module_& mod) {\n")
        for op in ops:
            f.write(f'    ::ttnn::operations::blaze::bind_blaze_op('
                    f'mod, ttnn::blaze::{op["op_name"]}, '
                    f'"Blaze {op["class_name"]}");\n')
        f.write("}\n")

if __name__ == "__main__":
    generate(sys.argv[1])
```

**CMake integration:**

```cmake
# PROPOSED: CMakeLists.txt addition
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated_blaze_registrations.hpp
           ${CMAKE_CURRENT_BINARY_DIR}/generated_blaze_nanobind.cpp
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_blaze_registrations.py
        ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS tools/generate_blaze_registrations.py
    COMMENT "Generating Blaze op registrations from registry"
)
```

**Advantages:** Fully automated; new ops added to Blaze are automatically registered at next build.

**Disadvantages:** Requires Blaze Python package importable at TT-Metal build time, creating a build-time dependency.

### Approach 2: C++ Preprocessor X-Macro List

Maintain a single list of op names as a macro, expanded in multiple contexts (as shown in [Section 4.2.6](02_blaze_device_operation_adapter.md)):

```cpp
// blaze_op_list.hpp -- single source of truth
#define BLAZE_OP_LIST(X)    \
    X(matmul, Matmul)       \
    X(copy, Copy)           \
    X(reduce, Reduce)       \
    // ... see Section 5.3 for the full catalog
```

**Advantages:** No external dependencies; pure C++; works in any build environment. Easy to review: adding an op is one line.

**Disadvantages:** Manual maintenance. New Blaze ops require a C++ change. Risk of silent drift from the Python registry.

### Approach 3: Hybrid (X-Macro + CI Validation)

Combine Approaches 1 and 2: maintain the X-macro list as the authoritative C++ source, but run a Python validation script in CI to detect drift:

```python
# PROPOSED: CI validation script
# tools/validate_blaze_registrations.py
import sys

def validate():
    from blaze.blaze_op import BlazeOp
    registered = parse_macro_list("blaze_op_list.hpp")
    in_registry = set(BlazeOp._class_registry.keys())
    missing = in_registry - registered
    extra = registered - in_registry
    if missing:
        print(f"ERROR: Blaze ops not registered in C++: {missing}")
        sys.exit(1)
    if extra:
        print(f"WARNING: C++ registrations for removed ops: {extra}")
    print(f"OK: {len(registered)} ops registered, "
          f"{len(in_registry)} in Blaze registry")

def parse_macro_list(path: str) -> set[str]:
    """Parse BLAZE_OP_LIST entries from the header file."""
    import re
    ops = set()
    with open(path) as f:
        for line in f:
            match = re.match(r'\s*X\((\w+),\s*\w+\)', line)
            if match:
                ops.add(match.group(1))
    return ops

if __name__ == "__main__":
    validate()
```

**Advantages:** Deterministic builds (no Python at build time); automated drift detection in CI.

**Disadvantages:** One line of manual C++ per new op (plus CI catches omissions).

### Comparison Table

| Factor | Build-Time Codegen | X-Macro List | Hybrid (Recommended) |
|--------|:-:|:-:|:-:|
| Per-op C++ changes | None | 1 line | 1 line |
| Build dependency on Blaze | Yes | No | No (CI only) |
| Drift detection | Inherent | Manual review | Automated CI |
| IDE support (C++ autocomplete) | Partial (generated files) | Full | Full |
| Complexity | Medium | Low | Medium |

**Recommendation:** The hybrid approach (Approach 3) is recommended for production. It provides deterministic builds, full IDE support, and automated drift detection. Start with the X-macro list (Approach 2) for initial deployment; add CI validation when the full catalog is registered.

---

## 4.3.4 Type Stubs for IDE Support

To provide IDE autocompletion and type checking for the `ttnn.blaze` module, generate a `.pyi` stub file alongside the bindings:

```python
# PROPOSED: ttnn/blaze/__init__.pyi (auto-generated or hand-written)
from ttnn import Tensor
from tt_metal.experimental import MeshProgramDescriptor

def matmul(io_tensors: list[Tensor],
           mesh_program_descriptor: MeshProgramDescriptor,
           custom_program_hash: int | None = None) -> Tensor: ...

def copy(io_tensors: list[Tensor],
         mesh_program_descriptor: MeshProgramDescriptor,
         custom_program_hash: int | None = None) -> Tensor: ...

def reduce(io_tensors: list[Tensor],
           mesh_program_descriptor: MeshProgramDescriptor,
           custom_program_hash: int | None = None) -> Tensor: ...

def rmsnorm(io_tensors: list[Tensor],
            mesh_program_descriptor: MeshProgramDescriptor,
            custom_program_hash: int | None = None) -> Tensor: ...

def sdpa(io_tensors: list[Tensor],
         mesh_program_descriptor: MeshProgramDescriptor,
         custom_program_hash: int | None = None) -> Tensor: ...

# ... all registered ops follow the same signature ...
```

The stub can be generated by the codegen script (Approach 1) or maintained manually alongside the X-macro list (Approach 3). For the hybrid approach, add a CI check that validates the stub against the macro list.

---

## 4.3.5 Backward Compatibility Specification

The following table specifies the backward-compatibility contract for every affected interface:

| Interface | Current Behavior | After Adapter | Breaking? |
|-----------|-----------------|---------------|:---------:|
| `ttnn.generic_op(io_tensors, descriptor)` | Dispatches through `GenericOpDeviceOperation` | Unchanged -- still works | No |
| `_run_program(io_tensors, descriptor)` | Calls `ttnn.generic_op` | Calls `ttnn.blaze.*` if available, else `ttnn.generic_op` | No |
| `MeshCompiledProgram.__init__` | No `op_name` parameter | Gains optional `op_name` parameter | No |
| `MeshProgramDescriptor` schema | `MeshPrograms` vector | Unchanged | No |
| `ProgramDescriptor` schema | Kernels, CBs, semaphores, optional `custom_program_hash` | Unchanged | No |
| `GenericMeshProgramFactory` | Used by `GenericOpDeviceOperation` | Also used by adapter (shared, not modified) | No |
| `GenericOpDeviceOperation` | Dispatches all Blaze ops | Unchanged; still dispatches unregistered ops | No |
| Program cache entries | Keyed by `type_hash<GenericOpDeviceOperation>` | Existing entries unchanged; new entries under `type_hash<Adapter<OpTag>>` | No |
| Tracy profiler output | All Blaze ops show as `"ttnn::generic_op"` | Registered ops show as `"ttnn::blaze::*"`; unregistered still `"ttnn::generic_op"` | No (additive) |
| Graph trace output | All Blaze ops show as `"GenericOpDeviceOperation"` | Registered ops show as adapter type; unregistered unchanged | No (additive) |
| `BlazeOp._class_registry` | Maps op names to Python classes | Unchanged | No |
| `blaze_nn.F.*` functions | Dispatch through `_run_program` | Automatically routed to `ttnn.blaze.*` | No |

---

## 4.3.6 Concrete Test Functions

Three test functions verify the adapter's correctness and backward compatibility:

### Test 1: Cache Isolation

```python
# PROPOSED: test_adapter_cache_isolation.py
def test_cache_isolation():
    """Verify that adapter and generic_op cache entries are disjoint."""
    descriptor = build_test_descriptor()
    io_tensors = build_test_tensors()
    device = io_tensors[0].device()

    cache = device.get_program_cache()
    initial_count = cache.num_entries()

    # Dispatch through generic_op
    ttnn.generic_op(io_tensors, descriptor)
    count_after_generic = cache.num_entries()
    assert count_after_generic == initial_count + 1, \
        "generic_op should create one cache entry"

    # Dispatch through adapter with same descriptor
    ttnn.blaze.matmul(io_tensors, descriptor)
    count_after_adapter = cache.num_entries()
    assert count_after_adapter == count_after_generic + 1, \
        "adapter should create a NEW cache entry (disjoint namespace)"
```

### Test 2: Behavioral Equivalence

```python
# PROPOSED: test_adapter_equivalence.py
def test_behavioral_equivalence():
    """Verify that adapter produces bit-identical results to generic_op."""
    descriptor = build_test_descriptor()

    # IMPORTANT: allocate SEPARATE output tensors for each path
    # to enable meaningful comparison
    io_tensors_generic = build_test_tensors()    # includes output tensor
    io_tensors_adapter = build_test_tensors()    # separate output tensor

    # Path A: generic_op
    result_generic = ttnn.generic_op(io_tensors_generic, descriptor)

    # Path B: adapter
    result_adapter = ttnn.blaze.matmul(io_tensors_adapter, descriptor)

    # Results must be bitwise identical
    assert torch.equal(
        result_generic.cpu().to_torch(),
        result_adapter.cpu().to_torch()), \
        "Adapter and generic_op must produce identical results"
```

### Test 3: Fallback to generic_op

```python
# PROPOSED: test_adapter_fallback.py
def test_fallback_to_generic_op():
    """Verify that unregistered ops fall back to generic_op."""
    descriptor = build_test_descriptor()
    io_tensors = build_test_tensors()

    # Use an op name that is NOT in ttnn.blaze
    compiled = MeshCompiledProgram(
        op_name="nonexistent_op_xyz",
        mesh_descriptor=descriptor,
        io_tensors=io_tensors)

    # Should not raise; should use generic_op fallback
    compiled.run(io_tensors)

    # Also test with op_name=None (legacy code path)
    compiled_legacy = MeshCompiledProgram(
        op_name=None,
        mesh_descriptor=descriptor,
        io_tensors=io_tensors)
    compiled_legacy.run(io_tensors)
```

---

## 4.3.7 blaze-nn Integration

The `blaze-nn` functional API (`blaze_nn.F.linear`, `blaze_nn.F.rmsnorm`, etc.) already compiles Blaze ops and dispatches them through `_run_program()`. Because the dispatch routing happens inside `MeshCompiledProgram._run_program()`, which is shared across all Blaze consumers, **blaze-nn benefits automatically without any code changes**:

```python
# EXISTING: blaze_nn/functional.py (simplified)
class BlazeNNFunction:
    def __call__(self, *args, **kwargs):
        compiled = self._compile(*args, **kwargs)
        compiled.run(io_tensors)
        # run() calls _run_program(), which now routes to ttnn.blaze.*

# blaze-nn benefits from named dispatch automatically without code changes.
# Chapter 7 describes optional enhancements (diagnostic API, early callable
# verification, env-var override) that add ~60 lines to blaze-nn for
# migration tooling and defense in depth.
```

The full dispatch chain becomes:

```
blaze_nn.F.linear(...)
  -> BlazeCompiler.compile() -> MeshCompiledProgram(op_name="matmul", ...)
    -> MeshCompiledProgram.run()
      -> _run_program()
        -> ttnn.blaze.matmul(io_tensors, descriptor)   [if registered]
        -> ttnn.generic_op(io_tensors, descriptor)      [fallback]
```

[Chapter 7](../ch07_blaze_nn_integration/index.md) covers the blaze-nn integration in full detail. The key point here is that the adapter's named dispatch is transparent to all callers of `MeshCompiledProgram`.

---

## 4.3.8 End-to-End Registration Walkthrough

To make the full picture concrete, here is the complete sequence for registering a new Blaze op, from Python op definition to profiler visibility:

**Step 1: Define the Blaze op (already exists -- no changes)**

```python
# blaze/ops/matmul/op.py (EXISTING)
class Matmul(MicroOp):
    name = "matmul"
    # ... emit(), ports, etc. ...
```

**Step 2: Add the op to the X-macro list (one line)**

```cpp
// blaze_op_list.hpp -- add one line
#define BLAZE_OP_LIST(X)    \
    X(matmul, Matmul)       \  // <-- this line
    // ...
```

Or, if using the codegen script (Approach 1), no C++ change is needed -- the script discovers `Matmul` in `BlazeOp._class_registry` automatically.

**Step 3: Build**

The build system expands the X-macro list, generating:
- `MatmulTag` struct with `name = "blaze::matmul"`
- `ttnn::blaze::matmul` registered operation
- `bind_blaze_op(mod, ttnn::blaze::matmul, "Blaze Matmul")` nanobind call

**Step 4: Run (no changes to model code)**

```python
# Model code (EXISTING -- no changes needed)
compiled_matmul = blaze.ops.Matmul.compile(a, b, output)
compiled_matmul.run([a, b, output])  # internally calls ttnn.blaze.matmul
```

**Step 5: Observe**

```
Tracy timeline (BEFORE):
  [ttnn::generic_op] [ttnn::generic_op] [ttnn::generic_op]

Tracy timeline (AFTER):
  [ttnn::blaze::matmul] [ttnn::blaze::rmsnorm] [ttnn::blaze::sdpa]
```

```
Graph trace (BEFORE):
  GenericOpDeviceOperation -> GenericOpDeviceOperation -> GenericOpDeviceOperation

Graph trace (AFTER):
  blaze::matmul -> blaze::rmsnorm -> blaze::sdpa
```

```
Program cache (BEFORE):
  hash(type_hash<GenericOpDeviceOperation>, descriptor_A) -> cached_program_A
  hash(type_hash<GenericOpDeviceOperation>, descriptor_B) -> cached_program_B

Program cache (AFTER):
  hash(type_hash<Adapter<MatmulTag>>, descriptor_A)  -> cached_program_A
  hash(type_hash<Adapter<RMSNormTag>>, descriptor_B) -> cached_program_B
```

Each op is now individually identifiable across all three observability systems. The programs that run on hardware are unchanged -- only the metadata envelope is different.

---

## Key Takeaways

- **`_run_program()` modification is minimal and backward-compatible.** One `_resolve_dispatch()` method, evaluated once at `MeshCompiledProgram.__init__` time, routes to `ttnn.blaze.*` when available and falls back to `ttnn.generic_op` otherwise. No per-call lookup overhead.

- **The `ttnn.blaze` namespace is a standard nanobind submodule.** Each op is bound using the same `bind_registered_operation` machinery as every other TTNN op. A `bind_blaze_op` template helper reduces per-op binding to one function call. The `__ttnn_operation__` property ensures framework tools recognize these as first-class TTNN operations.

- **Three auto-generation approaches** cover different deployment scenarios: build-time Python codegen (fully automated, requires Blaze at build time), C++ X-macro list (no external dependencies, one line per op), and a hybrid with CI validation (deterministic builds + automated drift detection). The hybrid approach is recommended for production.

- **`.pyi` type stubs** provide IDE autocompletion and type checking for the `ttnn.blaze` module, closing the ergonomics gap for Python callers.

- **blaze-nn benefits automatically.** Because dispatch routing happens inside `_run_program()`, which is shared across all Blaze consumers, `blaze-nn` functional API calls get named dispatch without any code changes.

- **Every interface is backward-compatible.** `generic_op` is unchanged, `MeshProgramDescriptor` is unchanged, program cache entries are disjoint, and the fallback path ensures graceful degradation against older TT-Metal builds.

- **Three concrete test functions** verify cache isolation (adapter creates separate entries), behavioral equivalence (bit-identical output with separate output tensors), and fallback behavior (unregistered ops use `generic_op`).

- **The full registration cost for a new op is one line of C++** (in the X-macro list) or zero lines (with the codegen script). The rest -- tag struct, `register_operation<>` call, nanobind binding, Python dispatch routing -- is automated.

---

**Next:** [Chapter 5 — Registering MicroOps and FusedOps](../ch05_op_registration/index.md)
