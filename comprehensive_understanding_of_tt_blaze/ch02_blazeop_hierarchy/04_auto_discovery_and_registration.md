# 04 -- Auto-Discovery and Registration

## The Problem

A growing op library (currently over 100 ops) needs a zero-boilerplate registration mechanism. Requiring developers to manually add entries to a central registry file creates merge conflicts, stale entries, and an extra step that is easy to forget. Blaze solves this with auto-discovery: placing a `BlazeOp` subclass in `blaze/ops/<name>/op.py` with a non-empty `name` attribute is sufficient to make it available. For the high-level overview of this mechanism, see Ch 1 Section 02 (Repository Map); this section covers the implementation details.

## `register_all()` -- The Discovery Engine

The `register_all()` function lives in `blaze/ops/__init__.py`. It runs at import time (triggered by `blaze/__init__.py`) and performs a three-step scan:

### Step 1: Walk Sub-Packages

`pkgutil.iter_modules(__path__)` iterates over every immediate sub-package in `blaze/ops/`. Two special packages are excluded via `_skip_subpackages`:

```python
_skip_subpackages = frozenset({"utils", "dsa"})
```

- `utils/` -- shared helper modules (not ops)
- `dsa/` -- golden reference implementations (not ops)

For every other sub-package (e.g., `matmul`, `mcast`, `gather`, `rmsnorm`), the function imports `<package>.op`:

```python
mod = importlib.import_module(f".{info.name}.op", __package__)
```

### Step 2: Find BlazeOp Subclasses

Within each imported module, `inspect.getmembers(mod, inspect.isclass)` finds all classes. The filter criteria are:

1. `issubclass(obj, BlazeOp)` -- must inherit from the base
2. `obj is not BlazeOp` -- exclude the base class itself
3. `obj.name` -- the `name` attribute must be non-empty (abstract intermediates are excluded)
4. `obj.__module__ == mod.__name__` -- the class must be defined in this module (not imported from elsewhere)

The fourth check prevents double-registration when one op module imports another's class (e.g., `DownProj` imports `Gather` but should not re-register it).

### Step 3: Register and Export

Each discovered class calls `obj.register()`, which triggers the full registration cascade (as described in [Section 01](./01_blazeop_base.md)):

1. `MicroOp.register()` auto-derives from the C++ header, then delegates to `BlazeOp.register()`
2. `BlazeOp.register()` creates the `OpSpec`, `CTArgSchema`, `PhaseInfo`, and `FusedOpConfig`

After registration, the class is set as an attribute on the `blaze.ops` module:

```python
setattr(ops_module, obj.__name__, obj)
```

The function returns the list of all discovered classes so that `blaze/__init__.py` can create `_OpHandle` proxies without a second discovery pass.

## The Op Registry

The global op registry (`blaze/registry.py`) is a simple dict mapping op names to `OpSpec` instances:

```python
_OP_REGISTRY: dict[str, OpSpec] = {}

def register_op(spec: OpSpec) -> None:
    _OP_REGISTRY[spec.name] = spec

def get_op_spec(name: str) -> OpSpec:
    if name not in _OP_REGISTRY:
        raise KeyError(f"Op '{name}' not registered. ...")
    return _OP_REGISTRY[name]

def list_ops() -> list[str]:
    return list(_OP_REGISTRY.keys())
```

The registry is deliberately minimal -- it stores `OpSpec` objects and provides lookup. There are no inheritance hierarchies, no lazy loading, no plugin systems. The `KeyError` on lookup failure includes the list of available ops, which is useful for debugging typos.

### The Four Registries

During `register()`, metadata is distributed across four registries plus the class-level registry. For the detailed five-step registration walkthrough including code snippets, see [Section 01 -- "The `register()` Method"](./01_blazeop_base.md). In brief, the registries are: `_OP_REGISTRY` (OpSpec, in `registry.py`), `_CT_ARG_SCHEMAS` (CTArgSchema, in `ct_args.py`), `_PHASE_REGISTRY` (PhaseInfo, in `kernel_codegen.py`), `_FUSED_OP_CONFIGS` (FusedOpConfig, in `blaze_op.py`), and the class-level `_class_registry` (Python class object, on `BlazeOp`).

## `_OpHandle` -- The Dual-API Proxy

Once `register_all()` returns the discovered classes, `blaze/__init__.py` wraps each one in an `_OpHandle` and sets it as a module attribute:

```python
for _cls in _register_all():
    if hasattr(_blaze_module, _cls.name):
        raise ValueError(
            f"Op name {_cls.name!r} collides with existing blaze.{_cls.name}"
        )
    setattr(_blaze_module, _cls.name, _OpHandle(_cls))
    _op_names.append(_cls.name)
```

The collision check prevents an op from shadowing an existing module-level name (e.g., an op named `"fuse"` would collide with `blaze.fuse()`).

`_OpHandle` is a callable proxy that exposes both the graph API and the composition API through a single object:

```python
class _OpHandle:
    def __init__(self, op_cls):
        self._cls = op_cls
        self.emit = op_cls.emit    # composition API

    def __call__(self, *args, **kwargs):
        return self._cls.graph_call(*args, **kwargs)  # graph API

    def __repr__(self):
        return f"<blaze.{self._cls.name}>"
```

This design means:

- `blaze.matmul(act, wt)` calls `Matmul.graph_call(act, wt)` -- the graph API path, for use inside `blaze.fuse()` blocks
- `blaze.matmul.emit(f, act, wt)` calls `Matmul.emit(f, act, wt)` -- the composition API path, for use with `FusedProgram`
- `repr(blaze.matmul)` returns `<blaze.matmul>`

The `emit` attribute is assigned directly from the class's `emit` staticmethod, so there is no wrapper overhead on the hot path.

## The Complete Registration Flow

Putting it all together, here is the sequence that runs when Python encounters `import blaze`:

```
import blaze
    |
    v
blaze/__init__.py imports blaze.ops.register_all
    |
    v
register_all() walks blaze/ops/* sub-packages
    |
    v
For each sub-package, imports <package>.op module
    |
    v
For each BlazeOp subclass found:
    +-- MicroOp? --> _auto_derive_from_kernel_hpp()
    |                   -> parse_op_hpp() reads kernels/op.hpp
    |                   -> populates ct_args, phase, pop_flags
    +-- register()
            +-- register_op(OpSpec(...))
            +-- _class_registry[name] = cls
            +-- register_ct_schema(CTArgSchema(...))
            +-- register_phase_info(name, PhaseInfo(...))
            +-- register_fused_op(name, FusedOpConfig(...))
    |
    v
register_all() returns discovered classes
    |
    v
blaze/__init__.py wraps each in _OpHandle
    |
    v
setattr(blaze, name, _OpHandle(cls))
    |
    v
blaze.matmul, blaze.mcast, blaze.gather, ... available
```

## Error Handling During Discovery

The discovery process is designed to fail visibly:

- **Missing `blaze.ops` sub-module**: If a dependency inside an op fails to import (e.g., a missing NumPy), the original `ModuleNotFoundError` is re-raised. Only the root `blaze.ops` package itself failing is caught and logged as a warning.
- **Name collisions**: If two ops declare the same `name`, the second `register_op()` call silently overwrites the first in the registry. However, the `_OpHandle` collision check in `__init__.py` catches the case where an op name collides with an existing module attribute.
- **Invalid port references**: If a CT arg's `CB(port)` references a port that does not exist on the class, `_register_ct_schema()` raises a `ValueError` listing the valid ports.
- **Missing required methods**: `__init_subclass__` checks raise `TypeError` at class definition time if `emit()` or `compose()` is not overridden.

## What `register_all()` Does Not Do

For clarity on scope boundaries:

- It does **not** compile kernels. Compilation happens later, when `BlazeCompiler.compile()` or `FusedProgram.run()` is called.
- It does **not** allocate device resources. No tensors, CBs, or semaphores are allocated during registration.
- It does **not** validate op composition. Whether two ops can be chained is only checked when `emit()` is called at composition time.
- It does **not** recurse into nested packages. Only immediate sub-packages of `blaze/ops/` are scanned.

---
<- [03 -- The C++ Parser](./03_cpp_parser.md) | [Index](index.md) | [05 -- Writing a MicroOp](./05_writing_a_micro_op.md) ->
