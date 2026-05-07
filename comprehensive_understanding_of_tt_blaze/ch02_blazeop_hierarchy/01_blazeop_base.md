# 01 -- BlazeOp Base Class

## The Class-as-Definition Philosophy

Every operation in Blaze is a Python class that inherits from `BlazeOp`. The class itself **is** the op definition -- there is no separate configuration file, no schema YAML, no registry entry to maintain by hand. The class attributes declare identity and interface, the port descriptors declare data-flow connections, the `ct_args` list declares compile-time arguments, and the methods (`emit()`, `compose()`, `register()`) define behavior. This section covers the base class that makes all of this work.

The source lives at `blaze/blaze_op.py`. For the overall package layout, see Ch 1 Section 02 (Repository Map).

## Class Attributes

`BlazeOp` defines a set of class-level attributes that subclasses override. These attributes form three groups: **identity**, **compute configuration**, and **phase/lifecycle metadata**.

### Identity Attributes

| Attribute | Type | Purpose |
|-----------|------|---------|
| `name` | `str` | Op identifier. Used as the registry key, the graph node type, and the default CT arg prefix. An empty string marks abstract intermediates (not registered). |
| `kernel` | `str` | Path to the C++ kernel header or source file. For `MicroOp`, this is auto-derived from `kernels/op.hpp`. For `FusedOp`, this is either a handwritten kernel path or empty (auto-generated). |

### Compute Configuration

| Attribute | Type | Default | Purpose |
|-----------|------|---------|---------|
| `math_fidelity` | `str` | `"HiFi4"` | Math precision. Maps to `ttnn.MathFidelity` during registration. |
| `math_approx_mode` | `bool` | `False` | Enable approximate math on TRISC compute. |

### Phase and Lifecycle Metadata

| Attribute | Type | Default | Purpose |
|-----------|------|---------|---------|
| `op_class` | `str` | `""` | C++ struct name (e.g., `"Matmul"`, `"Mcast"`). Auto-derived for MicroOps. |
| `is_inter_core` | `bool` | `False` | True if the op uses both BRISC and NCRISC (data movement across cores). Auto-derived from the parser. |
| `has_init_teardown` | `bool` | `False` | True if the C++ Op struct has `init()` and `teardown()` lifecycle methods with non-trivial bodies. |
| `supports_pop_control` | `bool` | `False` | True if the op has `pop_*` flags that control CB consumption. |
| `pop_flags` | `list[str]` | `[]` | Names of pop-control flag fields (e.g., `["pop_in0"]`). |
| `ct_args` | `list` | `[]` | List of `CompileTimeArg` declarations. Empty for MicroOps (auto-derived from header). |
| `phase` | `PhaseInfo` | `None` | Codegen metadata. Auto-derived for MicroOps. |

### Delimiter Constants

Two class-level string constants control naming conventions in the composition API:

- `CHILD_PREFIX_DELIMITER = "__"` -- used by `child_prefix()` to nest op namespaces (e.g., `"shared_expert__gu"`)
- `CB_NAME_DELIMITER = "___"` -- used by `cb_name()` to construct CB identifiers (e.g., `"matmul___out"`)

These delimiters are chosen to be visually distinct and to avoid collisions with CT arg dot-notation (`"prefix.field"`).

## Port Descriptors

Ports declare the data-flow interface of an op. Blaze uses Python's descriptor protocol (`__set_name__`) so that each port knows its own name automatically:

```python
class Matmul(MicroOp):
    in0: Input = Input()   # in0.name == "in0"
    in1: Input = Input()   # in1.name == "in1"
    out: Output = Output() # out.name == "out"
```

There are three port types:

### `Input()`

Declares an input tensor port. At registration time, each `Input` descriptor is collected into the op's `input_ports` list in the `OpSpec`. In the graph API, positional arguments to `blaze.<op>(...)` are matched to input ports in declaration order.

### `Output()`

Declares an output tensor port. Most ops have a single output; multi-output ops (like `DeepseekMoeGate`) declare multiple. The `register()` method collects these into `output_ports`.

### `Internal(num_pages=1)`

Declares an internal scratch buffer that is neither an input nor an output -- it is used for intermediate storage within the op. The `num_pages` parameter specifies the default page count:

```python
class SomeOp(MicroOp):
    scratch: Internal = Internal(num_pages=4)
```

Internal CBs are registered as `InternalCB` entries in the `OpSpec` and are available to the CB engine for allocation.

All three descriptors use `__set_name__` to capture their attribute name:

```python
class Input:
    def __init__(self):
        self.name: str = ""

    def __set_name__(self, owner, name):
        self.name = name
```

This means the attribute name in the class body becomes the port name in the graph and the CB engine -- no separate string registration is needed.

## CT Arg Kind Types

Compile-time arguments are the primary mechanism for passing configuration from Python to C++ kernels. Each CT arg has a **kind** that determines where its value comes from at compile time. The six kind types are frozen dataclasses:

### `CB(port)`

The CT arg value is a circular buffer ID for the named port. The `port` parameter must reference one of the op's `Input`, `Output`, or `Internal` descriptors:

```python
CompileTimeArg("in0", Type.UINT32, CB(cls.in0), Risc.TRISC)
```

At resolution time, the CT arg engine looks up the CB ID assigned to the referenced port.

### `Sem(protocol)`

The CT arg value is a semaphore ID. The `protocol` parameter is a `SemProtocol` enum value:

```python
class SemProtocol(str, Enum):
    SENDER = "sender_semaphore"
    RECEIVER = "receiver_semaphore"
    NOC0_RECEIVER = "noc0_receiver_semaphore"
    NOC1_RECEIVER = "noc1_receiver_semaphore"
```

| Protocol | Meaning |
|----------|---------|
| `SemProtocol.SENDER` | Sender semaphore (`"sender_semaphore"`) |
| `SemProtocol.RECEIVER` | Receiver semaphore (`"receiver_semaphore"`) |
| `SemProtocol.NOC0_RECEIVER` | NOC0-specific receiver (`"noc0_receiver_semaphore"`) |
| `SemProtocol.NOC1_RECEIVER` | NOC1-specific receiver (`"noc1_receiver_semaphore"`) |

### `Grid()`

The CT arg value is a NOC or grid coordinate. The source key defaults to the arg name. Used for multicast destination coordinates:

```python
CompileTimeArg("dest_noc_start_x", Type.UINT32, Grid(), Risc.BRISC)
```

### `Derived()`

The CT arg value is computed at runtime from user arguments, grid context, or tensor properties. The source key defaults to the arg name. The CT arg engine resolves it from the merged user-args dict:

```python
CompileTimeArg("k_num_tiles", Type.UINT32, Derived(), Risc.TRISC)
```

### `Param()`

The CT arg value is a user-supplied parameter, passed explicitly via `user_args` at compile time. Functionally identical to `Derived()` at the resolution level, but semantically distinct: `Param` means "the caller provides this", while `Derived` means "computed from context".

### `PerCore()`

The CT arg value varies per core. It is not resolved by the `CTArgEngine`; instead, it is resolved later at assembly time by `PerCoreCompileTimeDescriptor`. Used for role flags and per-core index values:

```python
CompileTimeArg("is_active", Type.UINT32, PerCore(), Risc.TRISC | Risc.NCRISC | Risc.BRISC)
```

## The `CompileTimeArg` Dataclass

Each CT arg declaration is a frozen dataclass with four fields:

```python
@dataclass(frozen=True)
class CompileTimeArg:
    name: str          # Field name (e.g., "k_num_tiles")
    type: Type         # C++ type: Type.UINT32 or Type.BOOL
    kind: CB | Sem | Grid | Derived | Param | PerCore
    riscs: Risc        # Flag enum: Risc.TRISC | Risc.NCRISC etc.
```

The `riscs` field is a `Flag` enum that supports bitwise OR. A CT arg needed by both TRISC and NCRISC is declared as `Risc.TRISC | Risc.NCRISC`. During `_register_ct_schema()`, each RISC flag is expanded into a separate `CTArgSpec` entry, because the CT arg engine groups arguments per-RISC for the `UnifiedKernelDescriptor`.

The `Type` enum maps directly to C++ types:

| Python | C++ |
|--------|-----|
| `Type.UINT32` | `uint32_t` |
| `Type.BOOL` | `bool` |

## The `register()` Method

The `register()` classmethod is the central registration entry point. When called on a `BlazeOp` subclass, it performs the following steps:

**Step 1 -- OpSpec registration.** Collects port descriptors and metadata into an `OpSpec` and registers it in the global `_OP_REGISTRY`:

```python
register_op(OpSpec(
    name=cls.name,
    kernel_path=cls.kernel,
    op_class=cls.op_class,
    input_ports=[TensorPort(name=n) for n in input_names],
    output_ports=[TensorPort(name=n) for n in output_names],
    preferred_math_fidelity=cls.math_fidelity,
    # ... lifecycle flags, pop_flags, internal_cbs
))
```

**Step 2 -- Class registry.** Records the class in `BlazeOp._class_registry[cls.name] = cls` (see "The Class Registry" below).

**Step 3 -- CTArgSchema registration** (if `ct_args` is non-empty). Converts `CompileTimeArg` entries to `CTArgSpec` entries grouped by RISC, validates CB port references against declared ports, and registers a `CTArgSchema`.

**Step 4 -- PhaseInfo registration** (if `cls.phase` is not `None`). Auto-derives `cpp_type` from `op_class` if not explicitly set (e.g., `op_class="Matmul"` becomes `cpp_type="blaze::Matmul::Op"`), then registers with the phase registry for kernel codegen.

**Step 5 -- FusedOpConfig registration** (if `compose()` is overridden). Creates a `FusedOpConfig` linking the op's `compose()` method to its compute config. For `MicroOp` subclasses, `kernel` is set to `None` (codegen path). For `FusedOp` subclasses, `kernel` is set to `cls.kernel` (which may be a handwritten source file or `None` for auto-generated).

The detection of whether `compose()` is overridden uses function identity comparison:

```python
if cls.compose.__func__ is not BlazeOp.compose.__func__:
    register_fused_op(cls.name, FusedOpConfig(...))
```

The `FusedOpConfig` is a dataclass:

```python
@dataclass
class FusedOpConfig:
    """Configuration for compiling a fused op via composition."""
    compose_fn: Any      # (f, tensors, output, user_args) -> None
    compute_config: Any  # ttnn.ComputeConfigDescriptor
    kernel: str | None = None  # absolute kernel source path (None = codegen)
```

## `graph_call()` -- The Graph API Entry Point

When a user writes `blaze.matmul(activation, weights)` inside a `blaze.fuse()` block, the `_OpHandle.__call__` dispatches to `graph_call()`:

```python
@classmethod
def graph_call(cls, *args, grid=None, **kwargs):
    from .context import _op_call
    input_names = cls._get_input_names()
    inputs_dict = {name: arg for name, arg in zip(input_names, args)}
    return _op_call(cls.name, inputs_dict, grid=grid, **kwargs)
```

This method bridges positional arguments to named ports (using `_get_input_names()` to read the `Input()` descriptor order), then delegates to the `FusionContext` to record the op call as a graph node. For background on the graph API and `FusionContext`, see Ch 1 Section 03 (The Two-API Design).

## `compose()` -- The Compiler Entry Point

The `compose()` classmethod bridges the graph API to the composition API. It receives a `FusedProgram`, a dict of resolved tensors keyed by port name, an output tensor, and user arguments:

```python
@classmethod
def compose(cls, f, tensors, output, user_args):
    cls.emit(f, tensors["in0"], tensors["in1"],
             prefix=user_args.get("prefix", "matmul"))
```

The `BlazeCompiler` calls `compose()` for each node in topological order during compilation. The method's job is to translate the tensor dict into `emit()` calls. Both `MicroOp` and `FusedOp` enforce that subclasses override `compose()` at class definition time via `__init_subclass__`.

## Helper Methods

### `child_prefix(prefix, name)`

Builds a nested prefix for child ops within a fused composition:

$$\text{child\_prefix}(p, n) = \begin{cases} p \mathbin{\|} \text{``\_\_''} \mathbin{\|} n & \text{if } p \neq \text{``''} \\ n & \text{otherwise} \end{cases}$$

Example: `BlazeOp.child_prefix("shared_expert", "gu")` produces `"shared_expert__gu"`.

### `cb_name(prefix, name)`

Builds a CB identifier name:

$$\text{cb\_name}(p, n) = p \mathbin{\|} \text{``\_\_\_''} \mathbin{\|} n$$

Example: `BlazeOp.cb_name("matmul", "out")` produces `"matmul___out"`.

### `_get_input_names()` / `_get_output_names()` / `_get_internal_cbs()`

Introspection methods that walk the class `__dict__` collecting port descriptors. These are used by `register()` to build the `OpSpec` and by `graph_call()` to map positional arguments to port names.

### `_is_abstract_op(cls)`

Returns `True` if `cls.name` is empty, signalling that the class is an abstract intermediate (mixin or partial base) and should not be registered or validated.

## The Class Registry

`BlazeOp` maintains a class-level registry mapping op names to their class objects:

```python
_class_registry: dict[str, type] = {}
```

During `register()`, `BlazeOp._class_registry[cls.name] = cls` records the class. This registry is separate from the `_OP_REGISTRY` in `registry.py` (which stores `OpSpec` instances). The class registry enables lookup of the actual Python class for runtime dispatch and introspection.

---
<- [Index](index.md) | [Index](index.md) | [02 -- MicroOp and FusedOp](./02_microop_and_fusedop.md) ->
