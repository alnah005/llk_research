# 02 -- MicroOp and FusedOp

`BlazeOp` is never instantiated directly. Two concrete subclasses define the two categories of ops in the system: `MicroOp` for ops backed by a C++ kernel header, and `FusedOp` for ops composed from other ops' `emit()` calls. Both live in `blaze/blaze_op.py`.

## The Inheritance Hierarchy

The hierarchy is flat -- there is no further subclassing depth:

```
BlazeOp                         (abstract base -- never instantiated)
    |
    +-- MicroOp                 (backed by C++ Op struct)
    |     +-- Matmul            (name="matmul")
    |     +-- Mcast             (name="mcast")
    |     +-- Gather            (name="gather")
    |     +-- RMSNorm           (name="rmsnorm")
    |     +-- ResidualAdd       (name="residual_add")
    |     +-- ... (~60 more)
    |
    +-- FusedOp                 (compiled via composition)
          +-- DownProj          (name="down_proj")
          +-- SharedExpert      (name="shared_expert")
          +-- DenseMLP          (name="dense_mlp")
          +-- DenseSwiGLU       (name="dense_swiglu")
          +-- ... (~20 more)
```

## MicroOp: The Kernel-Backed Op

A `MicroOp` is the atomic unit of computation in Blaze. It maps one-to-one to a C++ `Op` struct defined in a `kernels/op.hpp` file. The Python class provides identity and ports; the C++ header provides the kernel implementation and is the single source of truth for CT arg declarations, lifecycle methods, and RISC processor requirements.

### What a MicroOp Provides

| Responsibility | Source |
|---------------|--------|
| Op identity (`name`) | Python class attribute |
| Port declarations (`Input`, `Output`, `Internal`) | Python class descriptors |
| Composition logic (`emit()`) | Python static method |
| Graph API bridge (`compose()`) | Python classmethod |
| CT args, phase metadata, lifecycle flags | **Auto-derived from C++ header** |
| Kernel implementation (init, operator(), teardown) | C++ `Op` struct |

### `__init_subclass__` Enforcement

`MicroOp` uses `__init_subclass__` to validate that every concrete subclass meets the minimum contract:

1. If `name` is empty, the subclass is treated as an abstract intermediate and skipped.
2. If `op_class` is not set, it defaults to `cls.__name__` (e.g., class `Matmul` gets `op_class = "Matmul"`).
3. `emit()` must be overridden -- `TypeError` is raised otherwise.
4. `compose()` must be overridden -- `TypeError` is raised otherwise.

This enforcement means you cannot define a `MicroOp` subclass and forget to implement either method. The errors appear at class definition time, not at registration or runtime.

### Auto-Derivation from the Kernel Header

The key innovation of `MicroOp` is `_auto_derive_from_kernel_hpp()`. When a MicroOp's `ct_args` list is empty at registration time (the default), the registration process automatically locates and parses the C++ kernel header to populate everything:

**Locating the header.** `_find_kernel_hpp()` looks for `kernels/op.hpp` relative to the Python module file. The convention is that `blaze/ops/matmul/op.py` has its header at `blaze/ops/matmul/kernels/op.hpp`.

**Parsing the header.** Calls `parse_op_hpp()` (covered in [Section 03](./03_cpp_parser.md)) to extract a `ParsedKernel` object containing the struct name, CT arg declarations, lifecycle flags, and pop flags.

**Populating class attributes.** The parsed data is used to set:

- `cls.op_class` from the C++ outer struct name (e.g., `"Matmul"`)
- `cls.ct_args` from the parsed CT arg fields, converting each to a `CompileTimeArg` with the appropriate kind
- `cls.is_inter_core` set to `True` if both `has_brisc` and `has_ncrisc` are True (note: `CoreCTArgs` sets all three RISC flags, so any op with `CoreCTArgs` has `has_brisc = True`)
- `cls.has_init_teardown` from the presence of `init()` and `teardown()` methods
- `cls.supports_pop_control` and `cls.pop_flags` from `pop_*` fields in `CoreCTArgs`
- `cls.kernel` as a relative path from `$TT_METAL_HOME`
- `cls.phase` as a `PhaseInfo` for kernel codegen

**Kind resolution.** `_resolve_ct_arg_kind()` maps each parsed CT arg's source type to a Python kind object:

| Parsed Source | Kind Object | Notes |
|---------------|-------------|-------|
| `"cb"` | `CB(port)` | Port looked up from declared `Input`/`Output`/`Internal` descriptors |
| `"sem"` | `Sem(protocol)` | Protocol matched from the field name (e.g., `receiver_semaphore` maps to `SemProtocol.RECEIVER`) |
| `"derived"` | `Derived()` | Plain `uint32_t`/`bool` fields with `A::` dependencies |
| `"per_core"` | `PerCore()` | Fields declared as `PerCore` type in the header |
| `"flag"` | Skipped | Flag fields (`is_active`, `pop_*`) are role flags, not CT args in the schema sense |

### MicroOp `register()` Override

`MicroOp` overrides `register()` to inject the auto-derivation step before calling the base class:

```python
@classmethod
def register(cls):
    if not cls.ct_args:
        cls._auto_derive_from_kernel_hpp()
    else:
        cls._fill_phase_lifecycle_from_hpp()
    super().register()
```

If `ct_args` is manually declared (rare), `_fill_phase_lifecycle_from_hpp()` still reads the header to detect empty `init()`/`teardown()` bodies so that codegen can skip no-op lifecycle calls.

## FusedOp: The Composition Op

A `FusedOp` is compiled via composition -- it chains multiple `MicroOp.emit()` calls (and potentially other `FusedOp.emit()` calls) to build a multi-op pipeline within a single kernel dispatch. The `FusedOp` class itself owns no C++ kernel struct; its kernel is either auto-generated from the shadow graph or provided as a handwritten source.

### What a FusedOp Provides

| Responsibility | Source |
|---------------|--------|
| Op identity (`name`) | Python class attribute |
| Port declarations | Python class descriptors |
| Pipeline composition (`compose()`) | **Required** -- chains child `emit()` calls |
| Reusable building block (`emit()`) | **Optional** -- if the fused op is itself a composable unit |
| Handwritten kernel (`kernel`) | **Optional** -- if omitted, kernel is auto-generated |

### What a FusedOp Does Not Need

Unlike `MicroOp`, a `FusedOp` does not need:

| Attribute | Why not needed |
|---|---|
| `ct_args` | Composition handles CT arg registration through child `emit()` calls |
| `phase` | No single C++ struct; the codegen system walks the shadow graph |
| `op_class` | No single kernel struct identity |
| `kernel` (optional) | If omitted, auto-generated from the shadow graph |

This is the fundamental asymmetry: a `MicroOp` is a leaf node with its own CT arg schema; a `FusedOp` is an interior node that delegates to leaf nodes.

### `__init_subclass__` Enforcement

`FusedOp` enforces that `compose()` is overridden:

```python
def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    if _is_abstract_op(cls):
        return
    if cls.compose.__func__ is BlazeOp.compose.__func__:
        raise TypeError(f"{cls.__name__}: FusedOp must override compose()")
```

Note that `emit()` is **not** required for `FusedOp`. Many fused ops (e.g., `DenseMLP`, `DenseSwiGLU`) define both `compose()` and `emit()`, making them usable both from the graph API (via the compiler calling `compose()`) and as composable building blocks (via other ops calling `emit()`). But simpler fused ops that exist only as top-level dispatchers need only `compose()`.

## Separation of `compose()` vs `emit()`

The two methods serve distinct purposes:

| Method | Signature | Called By | Purpose |
|--------|-----------|-----------|---------|
| `compose(cls, f, tensors, output, user_args)` | classmethod | `BlazeCompiler` (graph API path) | Map a tensor dict to `emit()` calls |
| `emit(f, ...)` | staticmethod | Other ops, tests, direct usage | The actual composition logic |

`compose()` is a thin adapter. Its job is to unpack the `tensors` dict (keyed by port name), extract user-supplied configuration from `user_args`, and call `emit()`.

`emit()` is the actual composition logic. It receives a `FusedProgram` and concrete tensor/CBHandle arguments, allocates CBs, registers CT args, and returns a `CBHandle`. The `emit()` method is the reusable building block -- it can be called from any context, not just from the compiler.

For `FusedOp` subclasses, `compose()` often delegates to `emit()` with extra argument unpacking:

```python
class SharedExpert(FusedOp):
    @classmethod
    def compose(cls, f, tensors, output, user_args):
        cls.emit(
            f,
            tensors["activation"],
            tensors["gate_up_weights"],
            tensors["down_weights"],
            tensors["bias"],
            output,
            prefix=user_args.get("prefix", "shared_expert"),
            ab_coords=user_args.get("ab_grids"),
            down_coords=user_args.get("down_coords"),
        )
```

## Summary: MicroOp vs FusedOp

| Dimension | MicroOp | FusedOp |
|---|---|---|
| C++ kernel header | Required (`kernels/op.hpp`) | Optional (handwritten) or none |
| CT arg declaration | Auto-derived from C++ header | Not needed (child emit calls handle it) |
| `emit()` | Allocates CBs, registers CT args, returns `CBHandle` | Chains child `emit()` calls, returns final `CBHandle` |
| `compose()` | Thin adapter: unpack tensors, call `emit()` | May extract more user args, compute grids |
| `__init_subclass__` enforces | `emit()` + `compose()` | `compose()` only |
| Typical use | Single kernel operation | Multi-stage pipeline composition |

## Concrete Walkthrough: Matmul

The `Matmul` MicroOp (`blaze/ops/matmul/op.py`) illustrates the complete pattern. The C++ side lives in `blaze/ops/matmul/kernels/op.hpp`.

### Python Class Declaration

```python
class Matmul(MicroOp):
    name: str = "matmul"
    math_fidelity: str = "LoFi"

    in0: Input = Input()    # activations
    in1: Input = Input()    # weights
    out: Output = Output()  # result
```

Three attributes and three port descriptors -- that is the entire class-level declaration. Everything else comes from the header or the methods.

### Auto-Derivation Results

At registration time, `_auto_derive_from_kernel_hpp()` parses the Matmul kernel header (`kernels/op.hpp`) -- which declares `CoreCTArgs`, `ComputeCTArgs`, and `ReaderCTArgs` structs -- and produces: `op_class = "Matmul"`, `is_inter_core = True` (because `CoreCTArgs` sets both `has_brisc` and `has_ncrisc`), `has_init_teardown = True`, `pop_flags = ["pop_in0"]`, and 8 CT args across three RISC groups. For the full C++ header, parsed field-by-field output, and RISC mapping details, see [Section 03 -- "Worked Example: Matmul Header"](./03_cpp_parser.md).

### The `emit()` Method

Matmul's `emit()` follows the standard five-step pattern:

1. **Resolve inputs.** If `in0` is a `CBHandle`, use directly; if a tensor, allocate via `f.cb_from_tensor()`. Resolve `in1` weights via `resolve_weight_direct_address()`.

2. **Compute dimensions.** Derive `k_num_tiles` from the activation handle's page count; compute `out_w_per_core` from weight pages divided by $K$.

3. **Allocate output CB.** `f.cb_scratch()` creates a scratch CB for the intermediate output:

```python
out = f.cb_scratch(name=BlazeOp.cb_name(prefix, "out"),
                   num_pages=out_w_per_core, ...)
```

4. **Register CT args.** Per-core flags via `f.per_core_unified_ct_args()`, shared CBs via `f.unified_ct_args()`, NCRISC-specific via `f.ncrisc_ct_args()`, TRISC-specific via `f.trisc_ct_args()`, and TRISC runtime args via `f.trisc_rt_args()`.

5. **Return handle.** `f.output("matmul", out, ...)` records the node in the shadow graph and returns a `CBHandle` that downstream ops consume.

The return value is a `CBHandle` pointing to the output scratch buffer. Downstream ops receive this handle and can read its `num_pages`, `page_size`, `data_format`, and `tile_desc` to configure their own inputs -- this is the CBHandle chaining mechanism.

### The `compose()` Method

```python
@classmethod
def compose(cls, f, tensors, output, user_args):
    cls.emit(f, tensors["in0"], tensors["in1"],
             prefix=user_args.get("prefix", "matmul"))
```

A one-liner that unpacks the tensor dict and forwards to `emit()`.

## Concrete FusedOp: DownProj

`DownProj` (`blaze/ops/down_proj/op.py`) shows the `FusedOp` pattern. It chains four MicroOp building blocks into a linear pipeline:

$$\text{act} \xrightarrow{\text{matmul}} \text{mm} \xrightarrow[\text{bias}]{\text{residual\_add}} \text{added} \xrightarrow{\text{gather}} \text{output}$$

```python
class DownProj(FusedOp):
    name: str = "down_proj"
    math_fidelity: str = "LoFi"

    input: Input = Input()
    weights: Input = Input()
    add_input: Input = Input()
    output: Output = Output()
```

Each child `emit()` call uses `BlazeOp.child_prefix()` to create nested namespaces (`"down_proj__matmul"`, `"down_proj__mcast2"`, etc.) and returns a `CBHandle` that feeds into the next stage. The `FusedOp` has no CT args of its own -- they all come from the child MicroOps. For the full `emit()` walkthrough and composition details, see [Section 06 -- Writing a FusedOp](./06_writing_a_fused_op.md).

---
<- [01 -- BlazeOp Base Class](./01_blazeop_base.md) | [Index](index.md) | [03 -- The C++ Parser](./03_cpp_parser.md) ->
