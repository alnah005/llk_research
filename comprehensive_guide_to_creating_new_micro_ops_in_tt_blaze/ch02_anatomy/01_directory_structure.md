# 2.1 Directory Structure and Auto-Discovery

## What you will learn

- The canonical three-file layout for a micro-op package.
- How `register_all()` scans, filters, and registers every micro-op at framework startup.
- Which directories are excluded from scanning and why.
- The full registration pipeline from `MicroOp.register()` through auto-derivation to the five global registries.
- The distinction between `op_class` (C++ struct name) and `name` (Python identifier) and the precedence rules.
- A practical file checklist for adding a new micro-op.
- Common mistakes in directory layout and naming.

---

## 2.1.1 The canonical three-file layout

Every micro-op lives under `blaze/ops/<name>/` and follows the same three-file convention:

```
blaze/ops/<name>/
    __init__.py          # re-exports the MicroOp subclass for discovery
    op.py                # Python class: ports, config, compose(), emit()
    kernels/
        op.hpp           # C++ kernel header: CT arg structs, Op lifecycle
```

The convention is the configuration. There is no manifest file, no decorator registry, and no build-system entry. If the three files exist in the right structure, the framework discovers and registers the op automatically.

### Real-world examples

**RMSNorm** -- a compute-focused op with core, reader, and compute CT args:

```
blaze/ops/rmsnorm/
    __init__.py
    op.py                # class RMSNorm(MicroOp): Input, Output ports (no Internal)
    kernels/
        op.hpp           # CoreCTArgs, ReaderCTArgs, ComputeCTArgs, Op (init, operator(), teardown)
```

**Copy** -- a data-movement op with a single unified CT arg struct:

```
blaze/ops/copy/
    __init__.py
    op.py                # class Copy(MicroOp): src (Input), dst (Output) ports
    kernels/
        op.hpp           # CoreCTArgs only, Op (init, operator(), teardown)
```

**Mcast** -- an inter-core multicast op with reader, writer, and core CT args:

```
blaze/ops/mcast/
    __init__.py
    op.py                # class Mcast(MicroOp): src (Input), dst (Output) ports
    kernels/
        op.hpp           # CoreCTArgs, ReaderCTArgs, WriterCTArgs, Op (init, operator(), teardown)
```

These three examples illustrate the layout variations. RMSNorm uses three CT arg structs (core + reader + compute). Copy uses one (core). Mcast uses three (core + reader + writer) -- it has no ComputeCTArgs because it is purely a data-movement op. The parser classifies it as inter-core because both ReaderCTArgs and WriterCTArgs are present (plus CoreCTArgs sets all RISC flags).

### File-to-consumer mapping

| File | Primary consumer | What it provides |
|------|-----------------|------------------|
| `__init__.py` | `register_all()` | Makes the package importable; re-exports the `MicroOp` subclass |
| `op.py` | Framework runtime | Python class with identity, ports, config, `compose()`, `emit()` |
| `kernels/op.hpp` | `cpp_parser.py` (at registration) and the C++ compiler (at build) | CT arg structs, Op lifecycle methods, compile-time constants |

### Finding the kernel header: `_find_kernel_hpp()`

The framework locates the header relative to the Python module file:

```python
# blaze/blaze_op.py -- MicroOp._find_kernel_hpp()
@classmethod
def _find_kernel_hpp(cls) -> str | None:
    import inspect
    from pathlib import Path

    op_dir = Path(inspect.getfile(cls)).parent
    op_candidate = op_dir / "kernels" / "op.hpp"
    if op_candidate.is_file():
        return str(op_candidate)
    return None
```

The path is always `<op_dir>/kernels/op.hpp`. If the file is not found, `_find_kernel_hpp()` returns `None` and auto-derivation is silently skipped.

---

## 2.1.2 Auto-discovery with `register_all()`

At framework startup, `register_all()` scans every sub-package under `blaze/ops/` and registers every `MicroOp` subclass it finds. No manual registration is needed.

### The four-step discovery process

```python
# blaze/ops/__init__.py (simplified)
_skip_subpackages = frozenset({"utils", "dsa"})

def register_all():
    for info in pkgutil.iter_modules(__path__):
        if info.name in _skip_subpackages:
            continue
        mod = importlib.import_module(f".{info.name}.op", __package__)
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (issubclass(obj, BlazeOp)
                    and obj is not BlazeOp
                    and obj.name
                    and obj.__module__ == mod.__name__):
                obj.register()
```

Note the import path: `f".{info.name}.op"` imports directly into the `op.py` sub-module (e.g., `.rmsnorm.op`), not the package `__init__.py`. The base class check is against `BlazeOp`, not `MicroOp`, because `FusedOp` subclasses also live under `blaze/ops/`.

| Step | Action | Predicate | Effect |
|------|--------|-----------|--------|
| 1 | Scan sub-packages | `pkgutil.iter_modules(__path__)` | Yields every directory under `blaze/ops/` |
| 2 | Skip non-op directories | `info.name in _skip_subpackages` | Excludes `utils` and `dsa` |
| 3 | Import the op module | `importlib.import_module(f".{info.name}.op", ...)` | Imports `blaze/ops/<name>/op.py` directly |
| 4 | Filter and register | Four-condition check | Registers only concrete `BlazeOp` subclasses with a non-empty `name`, defined in the scanned module |

The four conditions in step 4 ensure:
- `issubclass(obj, BlazeOp)` -- only BlazeOp descendants, not unrelated classes.
- `obj is not BlazeOp` -- the base class itself is not registered.
- `obj.name` -- the class has a non-empty name (filters out abstract intermediaries).
- `obj.__module__ == mod.__name__` -- only classes defined in this module, not re-imported from elsewhere (prevents double registration).

### `_skip_subpackages`

The frozenset `{"utils", "dsa"}` excludes two directories that contain helper code, not ops:

- `blaze/ops/utils/` -- shared utilities used by multiple ops (e.g., tensor math helpers).
- `blaze/ops/dsa/` -- DSA (data structure abstraction) primitives used internally by the framework.

If you add a new non-op directory under `blaze/ops/`, add its name to `_skip_subpackages` to prevent import errors.

### Common mistakes in auto-discovery

1. **Wrong filename.** The Python class must be in `op.py` (or re-exported from `__init__.py`). A file named `my_op.py` will not be found by `inspect.getmembers(module)` unless `__init__.py` re-exports from it.

2. **Missing `__init__.py`.** Without `__init__.py`, the directory is not a Python package and `pkgutil.iter_modules` will not yield it.

3. **Empty `name` attribute.** `__init_subclass__` sets `op_class` from `cls.__name__` if not provided, but the `name` attribute must be explicitly set to a non-empty string. An empty name causes registration to fail silently or create an unusable registry entry.

4. **Class defined in the wrong module.** If the class is defined in a sub-module (e.g., `blaze/ops/my_op/helpers.py`) and only imported by `__init__.py`, the `obj.__module__ == module.__name__` check filters it out. Define the class in `op.py` and re-export from `__init__.py`.

---

## 2.1.3 The registration flow

When `obj.register()` is called for a `MicroOp` subclass, the following pipeline executes:

```
register_all()
   |
   v
MicroOp.register()
   |
   +-- cls.ct_args is empty?
   |     |
   |     +-- Yes: _auto_derive_from_kernel_hpp()
   |     |         +-- _find_kernel_hpp()          -> kernels/op.hpp path
   |     |         +-- parse_op_hpp(hpp_path)       -> ParsedKernel
   |     |         +-- Populates: op_class, ct_args, is_inter_core,
   |     |         |   has_init_teardown, supports_pop_control,
   |     |         |   pop_flags, kernel, phase
   |     |
   |     +-- No: _fill_phase_lifecycle_from_hpp()
   |               +-- Updates only init_is_empty, teardown_is_empty
   |
   v
BlazeOp.register()
   |
   +-- Populates 5 global registries
```

### The five global registries

| Registry | Module | Key | Value | Purpose |
|----------|--------|-----|-------|---------|
| `_OP_REGISTRY` | `blaze.registry` | `str` (name) | `OpSpec` | Maps op name to its OpSpec for lookup |
| `_class_registry` | `blaze.blaze_op` (on `BlazeOp`) | `str` (name) | `type` (BlazeOp subclass) | Name-to-class lookup for instantiation |
| `_CT_ARG_SCHEMAS` | `blaze.ct_args` | `str` (name) | `CTArgSchema` | CT arg names, types, kinds, and RISC targeting |
| `_PHASE_REGISTRY` | `blaze.kernel_codegen` | `str` (name) | `PhaseInfo` | Lifecycle metadata for code generation |
| `_FUSED_OP_CONFIGS` | `blaze.blaze_op` | `str` (name) | `FusedOpConfig` | Configuration for multi-op fusion |

### Concrete trace: RMSNorm through the pipeline

To make the pipeline concrete, here is what happens when `RMSNorm.register()` runs:

1. `cls.ct_args` is empty (no manual declaration).
2. `_find_kernel_hpp()` returns `blaze/ops/rmsnorm/kernels/op.hpp`.
3. `parse_op_hpp()` returns a `ParsedKernel` with:
   - `struct_name = ""` (empty -- see below)
   - `ct_args = [ParsedCTArg("input", "cb", NCRISC|TRISC), ParsedCTArg("gamma", "cb", NCRISC|TRISC), ...]`
   - `has_ncrisc = True` (ReaderCTArgs), `has_trisc = True` (ComputeCTArgs), `has_brisc = True` (CoreCTArgs sets all flags)
   - `has_init_teardown = True`, `teardown_is_empty = True` (the teardown body is `{}`)
4. `cls.op_class` is NOT overwritten by auto-derivation because `parsed.struct_name` is empty. This happens because `_OUTER_STRUCT_RE` uses `^struct` (column 0), but RMSNorm's outer struct is indented: `    struct RMSNorm {`. The `op_class` retains its `__init_subclass__` default of `"RMSNorm"`. (See [Section 2.4](./04_auto_derivation.md) for details on this parser limitation.)
5. `cls.ct_args` is set to the converted `CompileTimeArg` list, including fields from all three structs (CoreCTArgs, ReaderCTArgs, ComputeCTArgs).
6. `cls.is_inter_core` is set to `True` by auto-derivation (CoreCTArgs sets `has_brisc = True` and `has_ncrisc = True`). However, RMSNorm is not truly inter-core in practice -- this is a consequence of having `CoreCTArgs`. The class could override this if needed.
7. `BlazeOp.register()` populates all five registries.

---

## 2.1.4 The `name` field and `op_class` distinction

### `name`: the Python identity

Every `MicroOp` must declare a `name` attribute -- a non-empty string that serves as the op's unique identifier throughout the Python side of the framework:

| Usage | Where | Example |
|-------|-------|---------|
| Registry key | `_OP_REGISTRY[op_spec]`, `_class_registry[name]` | `"rmsnorm"` |
| CT arg prefix | `f"{name}.field_name"` in emit() | `"rmsnorm.input"` |
| OpSpec identity | `OpSpec(name="rmsnorm", ...)` | Lookup and matching |
| Error messages | Registration and compilation errors | `"rmsnorm: kernel header ct_cb field 'X' ..."` |
| Logging and debugging | Trace output, graph visualization | `"[rmsnorm] emit called"` |

### `op_class`: the C++ identity

`op_class` is the C++ struct name used for codegen type aliases and kernel instantiation:

```python
# codegen output example:
using Rmsnorm = blaze::RMSNorm::Op<ct_args::rmsnorm>;
```

Here `"RMSNorm"` is the `op_class` (C++ struct name), while `"rmsnorm"` is the `name` (Python identifier).

### The `op_class` precedence chain

The `op_class` value is determined by this priority:

1. **Auto-derived from the kernel header** (highest): If `_auto_derive_from_kernel_hpp()` runs and finds a `struct_name` in `op.hpp`, it **unconditionally** sets `cls.op_class = parsed.struct_name`. This overwrites any previous value, including one set by `__init_subclass__`.

2. **`__init_subclass__` default**: At class definition time, if `cls.op_class` is not already set (empty or falsy), `__init_subclass__` sets it to `cls.__name__` (the Python class name).

3. **Explicit class attribute** (only effective with manual `ct_args`): If you set `op_class` as a class attribute AND also provide manual `ct_args`, auto-derivation is skipped entirely, and your explicit value is preserved.

**Important:** Setting `op_class` explicitly on the class body does NOT prevent auto-derivation from overwriting it. The only way to preserve a manual `op_class` is to also declare manual `ct_args`, which causes `register()` to skip `_auto_derive_from_kernel_hpp()`.

---

## 2.1.5 File checklist for adding a new micro-op

When creating a new micro-op named `my_op`:

1. Create directory: `blaze/ops/my_op/`
2. Create `blaze/ops/my_op/__init__.py` with:
   ```python
   from .op import MyOp
   ```
3. Create `blaze/ops/my_op/op.py` with the `MicroOp` subclass (see [Section 2.2](./02_python_class.md)).
4. Create `blaze/ops/my_op/kernels/op.hpp` with the C++ kernel header (see [Section 2.3](./03_cpp_kernel_header.md)).
5. Add the kernel header to the aggregator include in `blaze/ops/ops.hpp`:
   ```cpp
   #include "blaze/ops/my_op/kernels/op.hpp"
   ```
   Without this include, the C++ compiler will not find the header during the metal build, even though the Python registration succeeds.

6. Verify registration by running the framework and checking that `my_op` appears in the op registry.

---

## Key takeaways

1. **Three files, one convention.** Every micro-op has `__init__.py`, `op.py`, and `kernels/op.hpp`. The framework discovers ops by scanning directories, not by reading a manifest.

2. **`register_all()` uses `pkgutil` + `inspect`.** It scans sub-packages, imports modules, and filters for concrete `MicroOp` subclasses defined in the scanned module. `_skip_subpackages` excludes `utils` and `dsa`.

3. **Five registries are populated.** `_OP_REGISTRY` (name-to-OpSpec in `blaze.registry`), `_class_registry` (name-to-class on `BlazeOp`), `_CT_ARG_SCHEMAS` (CT arg schemas in `blaze.ct_args`), `_PHASE_REGISTRY` (PhaseInfo in `blaze.kernel_codegen`), and `_FUSED_OP_CONFIGS` (FusedOpConfig in `blaze.blaze_op`).

4. **`name` and `op_class` serve different purposes.** `name` is the Python-side identifier (lowercase, used in registries and CT arg prefixes). `op_class` is the C++ struct name (used in codegen type aliases). Auto-derivation sets `op_class` from the header struct name when `_OUTER_STRUCT_RE` matches (column-0 `struct` declarations only — indented structs like RMSNorm's are not matched, so `op_class` retains its default).

5. **Do not forget the aggregator include.** Adding `ops.hpp` is the most commonly missed step -- the Python side registers fine, but the C++ build fails.

---

**Previous:** [Chapter 2 Index](./index.md) | **Next:** [Section 2.2 -- The Python Class](./02_python_class.md)
