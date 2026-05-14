# 7.1 FusedOp Class Structure

## The Class Hierarchy

Every Blaze op -- whether a hardware-backed micro-op or a composed pipeline -- inherits from `BlazeOp`. The class hierarchy has exactly two concrete branches, all defined in `blaze/blaze_op.py`:

```
BlazeOp                  (abstract base -- identity, ports, child_prefix, cb_name)
  |
  +-- MicroOp            (backed by a C++ Op struct and a kernel header)
  |
  +-- FusedOp            (composed from MicroOp emit() calls; no C++ struct)
```

**BlazeOp** is the root. It defines the common interface: `name`, `kernel`, port descriptors (`Input`, `Output`, `Internal`), `emit()`, `compose()`, `child_prefix()`, `cb_name()`, and auto-registration via `register()`. BlazeOp itself is never instantiated -- the class *is* the op definition.

**MicroOp** extends BlazeOp for ops backed by a single C++ `Op` struct. A MicroOp must override both `emit()` and `compose()`. Its `kernel` attribute points to the `.hpp` header that defines the C++ struct. Registration auto-derives `ct_args`, phase metadata, and lifecycle flags from that header. Examples: `Mcast`, `KNMatmul`, `GatedReduce`, `Gather`, `ResidualAdd`.

**FusedOp** extends BlazeOp for ops built entirely through composition. A FusedOp must override `compose()` and typically also provides a `@staticmethod emit()` so the op can itself be embedded as a building block inside even larger compositions. It has no C++ struct, no `op_class`, and no `ct_args` of its own. Examples: `SharedExpert`, `DownProj`, `DenseMLP`, `MoE`.


## The FusedOp Base Class

`FusedOp` is defined in `blaze/blaze_op.py` at line 408:

```python
class FusedOp(BlazeOp):
    """Fused op compiled via composition.

    Required:
        name                    -- identity
        Input()/Output() ports  -- interface
        compose()               -- compiler entry point (chains MicroOp emit() calls)

    Optional:
        emit()                  -- if the fused op is itself a reusable building block
        kernel                  -- handwritten kernel path; if omitted, the kernel
                                   is auto-generated from the shadow graph at build time

    Not needed: ct_args, phase, op_class (composition handles everything)
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if _is_abstract_op(cls):
            return
        if cls.compose.__func__ is BlazeOp.compose.__func__:
            raise TypeError(f"{cls.__name__}: FusedOp must override compose()")
```

The critical enforcement happens in `__init_subclass__`: every concrete FusedOp (one that sets `name`) must override `compose()`. If you forget, you get a `TypeError` at class definition time, before any test runs.

Compare this with `MicroOp`, which enforces both `emit()` and `compose()`:

```python
class MicroOp(BlazeOp):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if _is_abstract_op(cls):
            return
        if not cls.op_class:
            cls.op_class = cls.__name__
        if cls.emit is BlazeOp.emit:
            raise TypeError(
                f"{cls.__name__}: MicroOp must override emit()"
            )
        if cls.compose.__func__ is BlazeOp.compose.__func__:
            raise TypeError(
                f"{cls.__name__}: MicroOp must override compose() to support the Graph API path"
            )
```

A FusedOp does not require `emit()`. It requires `compose()`. If the FusedOp is also a reusable building block (called from another FusedOp's pipeline), it should additionally provide `emit()`, but this is optional. Conversely, a FusedOp never needs `ct_args`, `phase`, or `op_class` because composition handles compile-time argument registration through the child micro-ops.


## What a FusedOp Provides (and What It Does Not)

| Attribute | MicroOp | FusedOp | Notes |
|-----------|---------|---------|-------|
| `name` | Required | Required | Op identity string |
| `compose()` | Required | Required | Compiler entry point |
| `emit()` | Required | Optional (for embedding) | Building-block interface |
| `op_class` | Required (auto-derived) | Not needed | No C++ struct |
| `ct_args` | Auto-derived from kernel header | Not needed | Children handle CT args |
| `phase` | Auto-derived from kernel header | Not needed | Codegen derives phases |
| `is_inter_core` | Auto-derived | Not needed | MicroOp metadata |
| `has_init_teardown` | Auto-derived | Not needed | MicroOp metadata |
| `supports_pop_control` | Auto-derived | Not needed | MicroOp metadata |
| `pop_flags` | Auto-derived | Not needed | MicroOp metadata |
| `kernel` | Auto-derived (path to `kernels/op.hpp`) | Optional (handwritten kernel path) | Empty = auto-generated |
| `Input()`/`Output()` | Required | Required | Tensor interface |
| `math_fidelity` | Optional (default `"HiFi4"`) | Optional (default `"HiFi4"`) | Compute precision |
| `math_approx_mode` | Optional (default `False`) | Optional (default `False`) | SiLU/GELU approximation |

A FusedOp's compile-time arguments come entirely from its child micro-ops' `emit()` calls. Each child calls `f.unified_ct_args()`, `f.trisc_ct_args()`, etc., registering its own arguments on the shared `FusedProgram`. The FusedOp itself never touches CT arg registration.


## compose() vs emit(): Two Distinct Entry Points

### compose(): The Compiler Entry Point

`compose()` is a `@classmethod` called by the TT-Blaze compiler. Its signature is fixed:

```python
@classmethod
def compose(cls, f, tensors, output, user_args):
```

The four parameters are:

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | `FusedProgram` | The composition context. Holds device info, grid config, the underlying `BlazeProgram`, and all CB/semaphore/CT-arg state. |
| `tensors` | `dict[str, ttnn.Tensor]` | Maps port names (matching `Input()` descriptors) to device tensors. |
| `output` | `ttnn.Tensor` | The pre-allocated output tensor. |
| `user_args` | `dict` | Arbitrary keyword arguments passed from the model layer. |

The compiler resolves `tensors` from the model's tensor bindings and calls `compose()` with them. The FusedOp's job is to unpack those tensors, compute any derived values, and delegate to `emit()`.

The concrete invocation path runs through `BlazeCompiler._compile_fused_op()` in `blaze/compiler.py` (line 905). The key call at line 961 is:

```python
config.compose_fn(f, tensors, output_tensor, merged_args)
```

Where `config` is the `FusedOpConfig` registered for the op, and `merged_args` combines node-level kwargs with explicit `user_args`.

Here is `SharedExpert.compose()` from `blaze/ops/shared_expert/op.py`:

```python
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
        pop_shared_act=user_args.get("pop_shared_act", True),
    )
```

The pattern is consistent: `compose()` unpacks `tensors` by port name, extracts configuration from `user_args` with sensible defaults, and calls `cls.emit()`. It does not directly call `f.unified_ct_args()` or `f.cb_scratch()` -- those details belong to `emit()` and to the child micro-ops.

In more complex cases, `compose()` may do additional work before calling `emit()`. Here is `DownProj.compose()`:

```python
@classmethod
def compose(cls, f, tensors, output, user_args):
    act = Mcast.emit(f, tensors["input"], prefix="mcast")
    cls.emit(
        f,
        act,
        tensors["weights"],
        tensors["add_input"],
        output,
        mcast_sender_sem=0,
        core_coords=user_args.get("core_coords"),
    )
```

Notice that `DownProj.compose()` actually calls `Mcast.emit()` before calling `cls.emit()`. This is legal and sometimes necessary -- `compose()` can add preamble stages before the main pipeline. The `Mcast.emit()` call returns a `CBHandle`, which is then threaded into `DownProj.emit()` as the activation input. When `DownProj` is embedded inside `SharedExpert`, the Mcast has already happened, so `DownProj.emit()` is called directly (bypassing `compose()`'s Mcast).


### emit(): The Composition Building Block

`emit()` is a `@staticmethod` that takes explicit typed parameters rather than dictionaries. Its first parameter is always `f: FusedProgram`. Beyond that, the signature is op-specific:

```python
@staticmethod
def emit(
    f,
    activation,
    gate_up_weights,
    down_weights,
    bias,
    output,
    *,
    prefix: str = "shared_expert",
    ab_coords=None,
    down_coords=None,
    pop_shared_act=True,
    bias_dst_tile_info=None,
    bias_dst_num_pages=None,
    skip_bias_add: bool = False,
):
```

Key conventions:

1. **`f` is always first.** It is the `FusedProgram` that accumulates all CB allocations, CT args, and semaphores.

2. **Inputs follow `f`.** These are either `ttnn.Tensor` objects (for external tensors like weights) or `CBHandle` objects (for intermediate results from upstream stages).

3. **Keyword-only args after `*`.** The `prefix` parameter is always keyword-only and always has a default. Other configuration parameters (grid coordinates, flags, tile info overrides) are keyword-only with defaults.

4. **Return value is a `CBHandle` or structured result.** The return value is what the next stage consumes. `KNMatmul.emit()` returns a `KNMatmulOutput` dataclass wrapping a `CBHandle` plus grid metadata. `Mcast.emit()` returns a plain `CBHandle`. `SharedExpert.emit()` returns whatever `DownProj.emit()` returns.

Why is `emit()` a `@staticmethod` while `compose()` is a `@classmethod`? The `compose()` needs `cls` to call `cls.emit()` and to access class-level attributes like port descriptors. The `emit()` is intentionally decoupled from the class -- it is a pure function of `(f, inputs, config) -> CBHandle`. This means `emit()` can be called from any context: from the same class's `compose()`, from another FusedOp's `emit()`, from a test harness, or from a notebook.


### The Difference Between compose() and emit() -- Summary

| Aspect | `compose()` | `emit()` |
|--------|-------------|----------|
| **Decorator** | `@classmethod` | `@staticmethod` |
| **Signature** | `(cls, f, tensors, output, user_args)` | `(f, <op-specific args>, *, prefix, ...)` |
| **Called by** | The compiler / `BlazeCompiler._compile_fused_op()` | Other ops' `compose()` or `emit()` |
| **Purpose** | Bridge tensor-dict to emit() | Add CBs/CT-args/semaphores/graph nodes |
| **Input format** | `dict[str, Tensor]` | Individual tensors or CBHandles |
| **Returns** | Nothing (side effects on `f`) | CBHandle or structured result |

The mental model: `compose()` is the *public API* that the compiler calls. `emit()` is the *building block* that other ops compose with. When you embed one FusedOp inside another, you call its `emit()` directly -- never its `compose()`.


### When a FusedOp Has Both

A FusedOp with both `compose()` and `emit()` serves two roles:

- **As a standalone op** invoked by the compiler via `compose()`, which unpacks tensors and calls `emit()`.
- **As a reusable building block** called from another FusedOp's `emit()`, which passes `CBHandle` objects directly.

`SharedExpert` is an example. The compiler can invoke `SharedExpert.compose()` directly for a standalone shared-expert layer. But `DenseMLP.emit()` also calls `SharedExpert.emit()` as part of a larger pipeline:

```python
# From blaze/ops/dense_mlp/op.py
return SharedExpert.emit(
    f,
    act_mcast_handle,
    gate_up_weight,
    down_weight,
    act,
    shared_output,
    ab_coords=shared_ab_grids,
    down_coords=shared_down_coords,
    pop_shared_act=False,
)
```

Here, `act_mcast_handle` is a `CBHandle` from a prior `Mcast.emit()` call, not a raw tensor. The `SharedExpert.emit()` handles both cases:

```python
if isinstance(activation, ttnn.Tensor):
    ti = TileInfo.from_tensor(activation)
else:
    tile = ttnn.Tile([activation.tile_desc.height, activation.tile_desc.width])
    ti = TileInfo(
        tile=tile,
        data_format=activation.data_format,
        size=activation.page_size,
        desc=activation.tile_desc,
    )
```

This duality -- accepting both `ttnn.Tensor` and `CBHandle` -- is what makes an `emit()` function composable. It works at the boundary (tensor in from DRAM) and in the middle of a pipeline (CBHandle from prior stage).


## The kernel Attribute

The `kernel` attribute on `BlazeOp` has different semantics for `FusedOp` and `MicroOp`:

For a **MicroOp**, `kernel` is auto-derived from `kernels/op.hpp` relative to the op's Python module. It points to the C++ kernel header that defines the Op struct.

For a **FusedOp**, `kernel` is optional and has a different meaning:

- **If set**: points to a handwritten kernel source file. The compiler uses this source directly instead of auto-generating a kernel.
- **If empty (default)**: the kernel is auto-generated at build time. During `compose()`, each micro-op's `emit()` call records an `f.output()` node in the shadow graph. After composition completes, the codegen system walks this shadow graph to produce a unified kernel source (see `FusedProgram`, line 1951: `self.program.set_kernel_from_graph(self._shadow_graph, name=self._name)`).

The distinction is enforced in `BlazeOp.register()` (line 322-338):

```python
# Auto-register FusedOpConfig if compose() is overridden
if cls.compose.__func__ is not BlazeOp.compose.__func__:
    fidelity_map = {"HiFi4": ttnn.MathFidelity.HiFi4, "LoFi": ttnn.MathFidelity.LoFi}
    # MicroOps use codegen (kernel=None); their cls.kernel is the
    # .hpp header path, not a standalone kernel source.
    fused_kernel = cls.kernel if issubclass(cls, FusedOp) else None
    register_fused_op(
        cls.name,
        FusedOpConfig(
            compose_fn=cls.compose,
            kernel=fused_kernel,
            compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=fidelity_map.get(cls.math_fidelity, ttnn.MathFidelity.HiFi4),
                math_approx_mode=cls.math_approx_mode,
            ),
        ),
    )
```

Note the `issubclass(cls, FusedOp)` check. For `MicroOp` subclasses that also override `compose()`, `fused_kernel` is set to `None`, meaning their kernel path (the `.hpp` header) is not passed to `FusedOpConfig`. MicroOps use the codegen pipeline's phase-based kernel generation, not a standalone kernel source.

Most production FusedOps (including `SharedExpert`, `DownProj`, and `DenseMLP`) leave `kernel` empty, relying on auto-generation. Handwritten kernels are rare -- they exist only when the codegen pipeline cannot express a particular control-flow pattern.

> **Note on kernel auto-generation**: The codegen system reads the shadow graph populated by `f.output()` calls, identifies which micro-op phases are involved (via their registered `PhaseInfo`), and produces a C++ kernel source. The exact codegen mechanism is an internal implementation detail of `BlazeProgram.set_kernel_from_graph()` and may evolve. What is stable is the contract: if `FusedOpConfig.kernel` is `None` or empty, the framework handles kernel generation automatically.


## The FusedOpConfig and Registration

The `FusedOpConfig` dataclass captures the compilation configuration:

```python
@dataclass
class FusedOpConfig:
    """Configuration for compiling a fused op via composition.

    compose_fn(f: FusedProgram, tensors: dict, output: Tensor, user_args: dict) -> None
    """

    compose_fn: Any  # (f, tensors, output, user_args) -> None
    compute_config: Any  # ttnn.ComputeConfigDescriptor
    kernel: str | None = None  # absolute kernel source path (None = codegen)
```

Configs are stored in a module-level registry:

```python
_FUSED_OP_CONFIGS: dict[str, FusedOpConfig] = {}

def register_fused_op(name: str, config: FusedOpConfig) -> None:
    """Register a fused op configuration."""
    _FUSED_OP_CONFIGS[name] = config

def get_fused_op_config(name: str) -> FusedOpConfig | None:
    """Look up a fused op configuration by name."""
    return _FUSED_OP_CONFIGS.get(name)
```

When a FusedOp subclass calls `cls.register()` (inherited from `BlazeOp`), the registration system:

1. Collects `Input()` and `Output()` port descriptors to build an `OpSpec`.
2. Registers the `OpSpec` and stores the class in `BlazeOp._class_registry`.
3. Detects that `compose()` is overridden (via `cls.compose.__func__ is not BlazeOp.compose.__func__`).
4. Creates a `FusedOpConfig` and registers it in the global `_FUSED_OP_CONFIGS` dictionary.

The compiler retrieves this config at compilation time. In `BlazeCompiler._compile_fused_op()` (line 905 of `blaze/compiler.py`):

1. The `FusedOpConfig` is received as the `config` parameter.
2. A `FusedProgram` is constructed with device context, compute config, and optional kernel path (line 939).
3. `config.compose_fn(f, tensors, output_tensor, merged_args)` is called (line 961).
4. The resulting `FusedProgram` contains all CB descriptors, CT args, runtime args, and semaphores.
5. If `config.kernel` is a path, that source is compiled. If `None` or empty, the kernel is auto-generated from the shadow graph.


## Directory Layout

FusedOps follow the same directory convention as MicroOps:

```
blaze/ops/<op_name>/
    __init__.py          # re-exports the op class
    op.py                # the FusedOp class definition
    kernels/             # (optional) handwritten kernel if kernel= is set
```

For `SharedExpert`:

```
blaze/ops/shared_expert/
    __init__.py          # from .op import SharedExpert
    op.py                # SharedExpert(FusedOp)
```

There is no `kernels/` directory because `SharedExpert` uses auto-generated kernels. For `DownProj`:

```
blaze/ops/down_proj/
    __init__.py          # from .op import DownProj
    op.py                # DownProj(FusedOp) with compose(), emit(), and matmul()
```

Also no `kernels/` directory. The child micro-ops (`KNMatmul`, `Mcast`, `GatedReduce`, `Gather`, `ResidualAdd`) each have their own `kernels/op.hpp` in their respective directories, but the FusedOp itself does not.


## The Graph API Path

In addition to the compiler-driven `compose()` path, BlazeOp provides a `graph_call()` classmethod for the declarative Graph API:

```python
@classmethod
def graph_call(cls, *args, grid=None, **kwargs):
    """Graph API: create a node in a blaze.fuse() block."""
    from .context import _op_call

    input_names = cls._get_input_names()
    inputs_dict = {name: arg for name, arg in zip(input_names, args)}
    return _op_call(cls.name, inputs_dict, grid=grid, **kwargs)
```

This method is used inside `with blaze.fuse():` blocks:

```python
with blaze.fuse() as ctx:
    normed = BroadcastRmsNorm.graph_call(activation, gamma, grid=sender_grid)
    mcasted = Mcast.graph_call(normed, grid=all_cores)
    ...
```

The `graph_call()` method:

1. Collects `Input()` port names from the class via `_get_input_names()`.
2. Zips positional arguments with port names to build an `inputs_dict`.
3. Calls `_op_call()`, which dispatches to the active `FusionContext`.
4. The `FusionContext` creates an `OpNode` in the `BlazeGraph` and returns a `FusionResult`.

The returned `FusionResult` can be passed as input to subsequent `graph_call()` invocations, creating data dependency edges in the graph. The compiler then processes the graph: for each fused op node, it calls `_compile_fused_op()`, which invokes `compose()`.

For FusedOps, the Graph API path means the fused op appears as a single node in the outer graph. The internal structure (the micro-ops chained inside `compose()`) is not visible at the graph level -- it is expanded when the compiler calls `compose()`.

The graph API path is useful for:

- Dynamic fusion patterns where the op sequence is data-dependent
- Visualization (the `BlazeGraph` can be inspected or serialized)
- Optimization passes that operate on the graph before compilation
- Building model-level pipelines where multiple FusedOps are composed

For standalone op testing or when you need fine-grained control, calling `emit()` directly on a manually created `FusedProgram` is more straightforward.


## Compute Configuration

FusedOps set compute configuration at the class level:

```python
class SharedExpert(FusedOp):
    name: str = "shared_expert"
    math_fidelity: str = "LoFi"
    math_approx_mode: bool = True
```

These are captured in the `FusedOpConfig.compute_config` during registration:

```python
compute_config=ttnn.ComputeConfigDescriptor(
    math_fidelity=fidelity_map.get(cls.math_fidelity, ttnn.MathFidelity.HiFi4),
    math_approx_mode=cls.math_approx_mode,
)
```

The compute config applies to the entire fused kernel. All TRISC math operations within the fused kernel use the same fidelity and approximation mode. The two valid fidelity values are `"HiFi4"` (full precision, default) and `"LoFi"` (reduced precision, faster). Approximation mode controls whether SFP-unit approximation functions are enabled -- SiLU, GELU, and other transcendentals can use fast approximation when `math_approx_mode=True`.

This is a design constraint: if you need different fidelity for different stages (e.g., HiFi4 for attention but LoFi for MLP), you must split them into separate fused ops. When a FusedOp is embedded inside another FusedOp via `emit()`, the parent's compute config prevails because there is only one `FusedProgram` and one underlying `BlazeProgram`.


## The Prefix -> CT Args -> Phase Mapping

The order in which `emit()` calls appear in `compose()` (or in a parent fused op's `emit()`) determines the **kernel phase order**. Each `emit()` call adds its compile-time arguments and role flags to the `FusedProgram` in sequence. The kernel reads these arguments in the same order, executing one phase per micro-op.

Each `f.output()` call within an `emit()` marks a logical phase boundary in the shadow graph. The `FusedProgram` increments `_op_index` on every `output()` call (line 1789) and tracks CB lifetimes against this index. The codegen system uses the shadow graph to determine which phases to include in the generated kernel source.


## Minimal FusedOp Template

Putting it all together, the minimal viable FusedOp looks like:

```python
from blaze.blaze_op import BlazeOp, FusedOp, Input, Output

class MyFusedOp(FusedOp):
    name: str = "my_fused_op"
    math_fidelity: str = "LoFi"

    input_a: Input = Input()
    input_b: Input = Input()
    output: Output = Output()

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        cls.emit(
            f,
            tensors["input_a"],
            tensors["input_b"],
            output,
            prefix=user_args.get("prefix", "my_fused_op"),
        )

    @staticmethod
    def emit(f, input_a, input_b, output, *, prefix="my_fused_op"):
        # Stage 1: multicast input_a to all cores
        mcasted = Mcast.emit(f, input_a, prefix=BlazeOp.child_prefix(prefix, "mcast"))

        # Stage 2: some computation
        computed = SomeOp.emit(
            f, mcasted, input_b,
            prefix=BlazeOp.child_prefix(prefix, "compute"),
        )

        # Stage 3: gather results
        return Gather.emit(
            f, computed, output,
            prefix=BlazeOp.child_prefix(prefix, "gather"),
        )
```

The pattern is always the same:

1. Declare `Input` / `Output` ports for the compiler interface.
2. Implement `compose()` to bridge from `tensors` dict to `emit()`.
3. Implement `emit()` to chain micro-op calls, threading CBHandles from output to input, scoping each stage with `child_prefix()`.
4. Return the final `CBHandle` (or pass it to an output-writing op like Gather).
5. No `ct_args`, `phase`, or `op_class` is needed -- composition handles everything.

A minimal two-micro-op FusedOp (V2's `McastGather` example) demonstrates this even more concisely:

```python
class McastGather(FusedOp):
    """Minimal FusedOp: multicast input, then gather back to sender."""

    name: str = "mcast_gather"

    src: Input = Input()
    output: Output = Output()

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        dst = Mcast.emit(
            f, tensors["src"],
            prefix=BlazeOp.child_prefix("mg", "mcast"),
        )
        Gather.emit(
            f, dst, output,
            prefix=BlazeOp.child_prefix("mg", "gather"),
        )
```

After calling `McastGather.register()`, this op is available through both the direct-compile path and the Graph API. The compiler will auto-generate a kernel that sequences Mcast's BRISC/NCRISC phases followed by Gather's phases, with per-core `is_active` flags controlling which cores participate in each phase.


## Lifecycle of a FusedOp Invocation

When the compiler wants to execute a fused op, the lifecycle is:

1. **Lookup**: `get_fused_op_config("shared_expert")` retrieves the `FusedOpConfig`.
2. **FusedProgram creation**: A new `FusedProgram` is instantiated with the device, compute config, and optional kernel path (compiler line 939).
3. **Composition**: `config.compose_fn(f, tensors, output_tensor, merged_args)` is called (compiler line 961). This calls `SharedExpert.compose()`, which calls `SharedExpert.emit()`, which chains all child micro-ops.
4. **Program extraction**: `f.program` (the underlying `BlazeProgram`) now contains all CB descriptors, CT args, runtime args, and semaphores.
5. **Kernel compilation**: If `config.kernel` is a path, compile that source. If `None` or empty, auto-generate from the shadow graph via `set_kernel_from_graph()` (fused_program.py line 1951-1952).
6. **Execution**: The compiled kernel runs on the device grid.

The `FusedProgram` is a transient object -- created, populated during composition, and then its `program` is extracted for compilation. It is not persisted or reused across invocations (though resources like semaphores and named tensors may be shared via the `_sem_dict` and `_tensor_dict` injection points).


## Summary

| Concept | Detail |
|---------|--------|
| `FusedOp` inherits from | `BlazeOp` |
| Must override | `compose(cls, f, tensors, output, user_args)` |
| Optionally overrides | `emit()` for reusable composition |
| `compose()` signature | `@classmethod`, receives tensors as `dict`, output as `ttnn.Tensor`, config as `dict` |
| `emit()` signature | `@staticmethod`, receives explicit typed params, first arg is `FusedProgram` |
| `kernel` attribute | Optional path to handwritten kernel; empty means auto-generated from shadow graph |
| Registration | `FusedOpConfig` with `compose_fn`, `compute_config`, and optional `kernel` |
| Compiler invocation | `_compile_fused_op()` at compiler.py line 905; `compose_fn()` at line 961 |
| Directory layout | `blaze/ops/<name>/op.py`, optional `kernels/` |
| Graph API | `graph_call()` creates a single node in `blaze.fuse()` blocks |
| CT args | Handled entirely by child micro-ops' `emit()` calls |
| Phase order | Determined by `emit()` call order in `compose()` / `emit()` |
| Prefix scoping | `BlazeOp.child_prefix(prefix, name)` with `__` delimiter |
| CB name scoping | `BlazeOp.cb_name(prefix, name)` with `___` delimiter |
