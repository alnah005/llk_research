# 2.4 Auto-Derivation from the Kernel Header

## What you will learn

- The complete pipeline of `_auto_derive_from_kernel_hpp()` and every attribute it populates.
- The `ParsedKernel` and `ParsedCTArg` dataclasses that `parse_op_hpp()` produces.
- The regex patterns in `cpp_parser.py` and what each one matches.
- The `_resolve_ct_arg_kind()` method and how it converts parser output to CT arg kinds.
- RISC flag merging for fields that appear in multiple CT arg structs.
- When and why you might need to override auto-derived values.
- Edge cases: aliased fields, derived expressions, internal constants, pop_input asymmetry.
- The five most common mistakes and how to diagnose them.
- The `_fill_phase_lifecycle_from_hpp()` fallback for manually-declared CT args.

---

## 2.4.1 Why auto-derivation exists

Without auto-derivation, every micro-op would require the developer to declare CT args twice: once in the C++ header (as struct fields) and again in the Python class (as `CompileTimeArg` entries). The two declarations must be kept in sync -- same names, same RISC targeting, same types. This is tedious and error-prone.

Auto-derivation eliminates this redundancy. The C++ kernel header is the single source of truth. The Python class provides only what the header cannot express: identity (`name`), port descriptors (`Input`, `Output`, `Internal`), composition logic (`emit`, `compose`), and compute configuration (`math_fidelity`). Everything else is derived from the header at registration time.

---

## 2.4.2 The auto-derivation pipeline

When a `MicroOp` has no manually-declared `ct_args` (the common case), its `register()` method calls `_auto_derive_from_kernel_hpp()`:

```python
# blaze/blaze_op.py -- MicroOp.register()
@classmethod
def register(cls):
    if not cls.ct_args:
        cls._auto_derive_from_kernel_hpp()   # auto-derive everything
    else:
        cls._fill_phase_lifecycle_from_hpp()  # manual ct_args, still detect empty bodies
    super().register()                        # BlazeOp.register() -> 5 registries
```

### Step-by-step breakdown

```python
# blaze/blaze_op.py -- MicroOp._auto_derive_from_kernel_hpp()
@classmethod
def _auto_derive_from_kernel_hpp(cls):

    # Step 1: Locate the kernel header
    hpp_path = cls._find_kernel_hpp()
    if hpp_path is None:
        return  # no header found; skip silently

    from .cpp_parser import parse_op_hpp

    # Step 2: Parse the header into a ParsedKernel
    parsed = cls._parsed_phase = parse_op_hpp(hpp_path)

    # Step 3: Set op_class from the C++ outer struct name
    if parsed.struct_name:
        cls.op_class = parsed.struct_name  # UNCONDITIONAL overwrite

    # Step 4: Build port lookup for CB resolution
    port_map: dict[str, Input | Output | Internal] = {}
    for k, v in vars(cls).items():
        if isinstance(v, (Input, Output, Internal)):
            port_map[v.name] = v

    # Step 5: Convert ParsedCTArgs to CompileTimeArg list
    ct_args = []
    for arg in parsed.ct_args:
        kind = cls._resolve_ct_arg_kind(arg, port_map)
        if kind is None:
            continue  # flag fields are excluded
        ct_args.append(CompileTimeArg(arg.name, Type.UINT32, kind, arg.riscs))
    cls.ct_args = ct_args

    # Step 6: Auto-derive metadata (guarded assignments)
    if not cls.is_inter_core:
        cls.is_inter_core = parsed.has_brisc and parsed.has_ncrisc
    if not cls.has_init_teardown:
        cls.has_init_teardown = parsed.has_init_teardown
    if not cls.supports_pop_control and parsed.pop_flags:
        cls.supports_pop_control = True
        cls.pop_flags = parsed.pop_flags

    # Step 7: Auto-derive kernel path
    if not cls.kernel:
        cls.kernel = os.path.relpath(str(hpp_path), os.environ.get("TT_METAL_HOME", "."))

    # Step 8: Auto-derive PhaseInfo
    if cls.phase is None:
        cls.phase = PhaseInfo(
            cpp_type="",  # filled later by BlazeOp.register()
            alias_suffix=cls.op_class.split("::")[-1],
            has_init_teardown=parsed.has_init_teardown,
            setup_method=parsed.setup_method,
            init_is_empty=parsed.init_is_empty,
            teardown_is_empty=parsed.teardown_is_empty,
        )
```

### Key observations about the assignment guards

| Attribute | Guard | Implication |
|-----------|-------|-------------|
| `cls.op_class` | `if parsed.struct_name` | Auto-derivation overwrites op_class only if the parser found a struct name. If the outer struct is indented (like RMSNorm), `parsed.struct_name` is empty and op_class is NOT overwritten. Setting op_class explicitly on the class body is overwritten by auto-derivation only when the struct is at column 0 in the header. |
| `cls.ct_args` | **No guard** (always set) | Always overwritten with the parsed list. |
| `cls.is_inter_core` | `if not cls.is_inter_core` | A pre-set `True` is preserved. But you cannot force `False` when the header derives `True`. |
| `cls.has_init_teardown` | `if not cls.has_init_teardown` | A pre-set `True` is preserved. |
| `cls.supports_pop_control` | `if not cls.supports_pop_control and parsed.pop_flags` | A pre-set `True` is preserved. |
| `cls.kernel` | `if not cls.kernel` | A pre-set path is preserved. |
| `cls.phase` | `if cls.phase is None` | A pre-set `PhaseInfo` is preserved (but init_is_empty/teardown_is_empty are updated by the fallback path). |

This asymmetry is important: `op_class` and `ct_args` are always overwritten, while `is_inter_core`, `has_init_teardown`, `kernel`, and `phase` use guards that respect pre-set values.

---

## 2.4.3 The `ParsedKernel` and `ParsedCTArg` dataclasses

These are the output types of `parse_op_hpp()`, defined in `blaze/cpp_parser.py`.

### `ParsedCTArg`

```python
@dataclass
class ParsedCTArg:
    """A CT arg extracted from a kernel header."""
    name: str       # Field name (e.g., "src", "num_tiles")
    source: str     # "cb" | "sem" | "per_core" | "flag" | "derived"
    riscs: Risc     # Which RISC(s) read this arg (Flag enum, supports |)
```

| Field | Description | Example |
|-------|-------------|---------|
| `name` | The schema arg name (from `A::` token, not the LHS field name) | `"input"`, `"receiver_semaphore"`, `"num_tiles"` |
| `source` | Classification based on C++ type or RHS analysis | `"cb"` for `CB`, `"sem"` for `Semaphore`, `"derived"` for `uint32_t` with A:: refs |
| `riscs` | Bitwise-OR of `Risc` flags for processors needing this arg | `Risc.NCRISC \| Risc.TRISC` |

### `ParsedKernel`

```python
@dataclass
class ParsedKernel:
    """Complete metadata extracted from an op.hpp file."""
    struct_name: str = ""              # outer struct name (e.g., "RMSNorm")
    ct_args: list[ParsedCTArg] = field(default_factory=list)
    has_init_teardown: bool = False    # both init() and teardown() present?
    init_is_empty: bool = False        # init() body is whitespace/comments only?
    teardown_is_empty: bool = False    # teardown() body is whitespace/comments only?
    has_brisc: bool = False            # WriterCTArgs or CoreCTArgs present?
    has_ncrisc: bool = False           # ReaderCTArgs or CoreCTArgs present?
    has_trisc: bool = False            # ComputeCTArgs or CoreCTArgs present?
    pop_flags: list[str] = field(default_factory=list)  # All pop_* fields (any struct/type)
    setup_method: str | None = None    # setup_* static method name
```

| Field | How it is set | How it is used |
|-------|---------------|---------------|
| `struct_name` | First match of `_OUTER_STRUCT_RE` (may be empty -- see note below) | Sets `cls.op_class` only if non-empty |
| `ct_args` | Collected from all CT arg structs, merged by name | Converted to `CompileTimeArg` list via `_resolve_ct_arg_kind()` |
| `has_init_teardown` | Both `init()` and `teardown()` regex match | Sets `cls.has_init_teardown` |
| `init_is_empty` | `init()` body is whitespace/comments | Sets `PhaseInfo.init_is_empty` (codegen skips no-op call) |
| `teardown_is_empty` | `teardown()` body is whitespace/comments | Sets `PhaseInfo.teardown_is_empty` (codegen skips no-op call) |
| `has_brisc` | WriterCTArgs found OR CoreCTArgs found | Combined with has_ncrisc for `is_inter_core` |
| `has_ncrisc` | ReaderCTArgs found OR CoreCTArgs found | Combined with has_brisc for `is_inter_core` |
| `has_trisc` | ComputeCTArgs found OR CoreCTArgs found | Metadata (not directly used in auto-derivation logic) |
| `pop_flags` | All fields with `pop_*` name (any struct, any type) | Sets `cls.pop_flags`, `cls.supports_pop_control` |
| `setup_method` | Static method matching `static void setup_\w+()` (template methods like `template <typename A> void setup_src()` are NOT detected) | Sets `PhaseInfo.setup_method` for codegen |

### Parser limitation: `_OUTER_STRUCT_RE` and indented structs

The `_OUTER_STRUCT_RE` regex is `^struct\s+(\w+(?<!CTArgs))\s*\{` with `re.MULTILINE`. The `^` anchor requires the `struct` keyword to start at column 0. This means **indented outer structs are not detected**.

RMSNorm's header has the outer struct indented (inside the `blaze` namespace):

```cpp
namespace blaze {
    struct RMSNorm {    // <-- indented, NOT at column 0
```

For RMSNorm, `parsed.struct_name` is `""` (empty). Since the auto-derivation code guards the assignment with `if parsed.struct_name:`, `cls.op_class` is NOT overwritten, and the `__init_subclass__` default (`cls.__name__` = `"RMSNorm"`) survives. In this case, the end result is the same, but the mechanism is different from ops like Mcast where the struct is at column 0 and `op_class` IS set by auto-derivation.

In contrast, Mcast's header has the struct at column 0:

```cpp
struct Mcast {    // <-- column 0, detected by _OUTER_STRUCT_RE
```

This is an important context for debugging: if a new op's outer struct is indented, `parsed.struct_name` will be empty, and `op_class` will retain whatever `__init_subclass__` set. If the Python class name differs from the C++ struct name and the struct is indented, you must set `op_class` manually.

### RISC flag derivation from struct names

The parser sets RISC presence flags based on which CT arg structs are found:

```python
if struct_name == "ComputeCTArgs":
    result.has_trisc = True
elif struct_name == "ReaderCTArgs":
    result.has_ncrisc = True
elif struct_name == "WriterCTArgs":
    result.has_brisc = True
elif struct_name == "CoreCTArgs":
    result.has_brisc = True
    result.has_ncrisc = True
    result.has_trisc = True   # CoreCTArgs sets ALL three flags
```

**Important implication for `is_inter_core`:** Because `CoreCTArgs` sets all three RISC flags, any op with only `CoreCTArgs` (like Copy) will have `has_brisc = True` and `has_ncrisc = True`, causing `is_inter_core = True` in auto-derivation. This is technically correct from the parser's perspective (CoreCTArgs is dispatched to all RISCs), but may not match the developer's intent. If Copy does not actually coordinate between cores in the inter-core sense, the developer should set `is_inter_core = True` on the class (which prevents auto-derivation from overriding it -- but since the auto-derived value is also `True`, this is effectively a no-op for Copy), or accept the classification.

---

## 2.4.4 CT arg merging across structs

When the same field name appears in multiple CT arg structs, the parser merges their RISC flags rather than creating duplicates:

```python
# blaze/cpp_parser.py -- _merge_args()
def _merge_args(seen: dict[str, ParsedCTArg], new_args: list[ParsedCTArg]) -> None:
    for arg in new_args:
        if arg.name in seen:
            seen[arg.name].riscs |= arg.riscs  # union RISC flags
        else:
            seen[arg.name] = arg
```

Example: if `input` appears in both `ReaderCTArgs` (NCRISC) and `ComputeCTArgs` (TRISC):

```
ReaderCTArgs:  ParsedCTArg(name="input", source="cb", riscs=Risc.NCRISC)
ComputeCTArgs: ParsedCTArg(name="input", source="cb", riscs=Risc.TRISC)
    merged --> ParsedCTArg(name="input", source="cb", riscs=Risc.NCRISC | Risc.TRISC)
```

The merged entry becomes a single `CompileTimeArg` in Python, with RISC targeting `Risc.NCRISC | Risc.TRISC`. The `source` is taken from the first occurrence; subsequent occurrences only contribute their RISC flags.

---

## 2.4.5 The `_resolve_ct_arg_kind()` method

After `parse_op_hpp()` produces `ParsedCTArg` objects, each is converted to a `CompileTimeArg` kind:

```python
# blaze/blaze_op.py -- MicroOp._resolve_ct_arg_kind()
@classmethod
def _resolve_ct_arg_kind(cls, arg, port_map):
    if arg.source == "cb":
        port = port_map.get(arg.name)
        if port is None:
            raise ValueError(
                f"{cls.name}: kernel header ct_cb field '{arg.name}' "
                f"has no matching port. Ports: {sorted(port_map)}"
            )
        return CB(port)
    elif arg.source == "sem":
        protocol = cls._SEM_FIELD_TO_PROTOCOL.get(arg.name)
        if protocol is None:
            raise ValueError(
                f"{cls.name}: kernel header ct_sem field '{arg.name}' "
                f"is not a valid SemProtocol value. "
                f"Valid: {sorted(cls._SEM_FIELD_TO_PROTOCOL)}"
            )
        return Sem(protocol)
    elif arg.source == "flag":
        return None           # flags are NOT CT args
    elif arg.source in cls._SOURCE_TO_KIND:
        return cls._SOURCE_TO_KIND[arg.source]()  # Derived() or PerCore()
    else:
        raise ValueError(f"{cls.name}: unknown source '{arg.source}'")
```

| Source | Resolved kind | Validation | Error on failure |
|--------|--------------|------------|-----------------|
| `"cb"` | `CB(port)` | Port name must exist in `port_map` | `"ct_cb field 'X' has no matching port"` |
| `"sem"` | `Sem(protocol)` | Field name must be a valid `SemProtocol` value | `"ct_sem field 'X' is not a valid SemProtocol value"` |
| `"flag"` | `None` (excluded) | None | N/A |
| `"derived"` | `Derived()` | None | N/A |
| `"per_core"` | `PerCore()` | None | N/A |

The valid semaphore protocol names are: `sender_semaphore`, `receiver_semaphore`, `noc0_receiver_semaphore`, `noc1_receiver_semaphore`.

---

## 2.4.6 Concrete trace: Mcast through the pipeline

To make the pipeline concrete, here is what `_auto_derive_from_kernel_hpp()` produces for Mcast:

| Step | Action | Result |
|------|--------|--------|
| 1 | `_find_kernel_hpp()` | `blaze/ops/mcast/kernels/op.hpp` |
| 2 | `parse_op_hpp()` | `ParsedKernel(struct_name="Mcast", ct_args=[...], has_brisc=True, has_ncrisc=True, has_trisc=True, has_init_teardown=True, pop_flags=["pop_src"], setup_method=None)` |
| 3 | Set op_class | `cls.op_class = "Mcast"` |
| 4 | Build port_map | `{"src": Input(), "dst": Output(), ...}` |
| 5 | Convert ct_args | `[CompileTimeArg("receiver_semaphore", UINT32, Sem(...), NCRISC\|BRISC), CompileTimeArg("sender_semaphore", UINT32, Sem(...), BRISC), ...]` |
| 6 | Set is_inter_core | `True` (has_brisc AND has_ncrisc) |
| 7 | Set kernel path | `"blaze/ops/mcast/kernels/op.hpp"` |
| 8 | Create PhaseInfo | `PhaseInfo(cpp_type="", alias_suffix="Mcast", has_init_teardown=True, setup_method=None, init_is_empty=False, teardown_is_empty=False)` |

Note on `setup_method`: Mcast's `setup_src()` is a **template member method** (`template <typename A> void setup_src()`), NOT a static method (`static void setup_src()`). The `_SETUP_RE` regex (`static\s+void\s+(setup_\w+)\s*\(\s*\)`) does not match template methods, so `setup_method` is `None`. Note also that `pop_flags` contains `"pop_src"` from `CoreCTArgs` (the `pop_` prefix is detected regardless of type).

### PhaseInfo comparison: Mcast vs. RMSNorm

| PhaseInfo field | Mcast | RMSNorm |
|----------------|-------|---------|
| `cpp_type` | `"blaze::Mcast::Op"` (filled by register) | `"blaze::RMSNorm::Op"` (filled by register) |
| `alias_suffix` | `"Mcast"` | `"RMSNorm"` |
| `has_init_teardown` | `True` | `True` |
| `setup_method` | `None` (template method, not static -- not detected by `_SETUP_RE`) | `None` |
| `init_is_empty` | `False` | `False` |
| `teardown_is_empty` | `False` | `True` (RMSNorm's teardown is `void teardown() {}`) |

---

## 2.4.7 When to override auto-derived values

Auto-derivation handles the common case well. Manual overrides are needed only for edge cases.

### Scenario 1: Override `op_class`

**When:** The C++ struct name does not match the desired codegen type alias.

Auto-derivation overwrites `op_class` when the parser finds a struct name at column 0 (Step 3 above). If the outer struct is indented (like RMSNorm), `parsed.struct_name` is empty and `op_class` is not overwritten. However, if the struct IS at column 0, you cannot preserve a manual `op_class` unless you also provide manual `ct_args`. If you set `ct_args`, `_auto_derive_from_kernel_hpp()` is skipped entirely, and your explicit `op_class` survives:

```python
class MyOp(MicroOp):
    name = "my_op"
    op_class = "blaze::custom::MyOp"
    ct_args = [
        CompileTimeArg("input", Type.UINT32, CB(Input()), Risc.NCRISC),
        # ... manually declare all CT args
    ]
```

In practice, this override is rarely needed because the convention is for the C++ struct name to match the Python class name.

### Scenario 2: Override `ct_args` manually

**When:** The C++ header uses a non-standard pattern that the parser cannot handle (e.g., macro-generated fields, conditional compilation in struct bodies).

When `ct_args` is non-empty at registration time, `register()` skips `_auto_derive_from_kernel_hpp()` entirely and calls `_fill_phase_lifecycle_from_hpp()` instead (see [Section 2.4.10](#2410-the-_fill_phase_lifecycle_from_hpp-fallback)).

### Scenario 3: Override `is_inter_core`

**When:** The op needs to be classified as inter-core, but the parser does not detect it. Or: you want to force `is_inter_core = True` even though the header does not have both `ReaderCTArgs` and `WriterCTArgs`.

```python
class MyOp(MicroOp):
    name = "my_op"
    is_inter_core = True  # pre-set; auto-derivation will not overwrite True
```

**Note the asymmetry:** You can force `is_inter_core = True` (the guard `if not cls.is_inter_core` preserves a pre-set `True`), but you cannot force `is_inter_core = False` when the header derives `True`.

### Scenario 4: Override `phase`

**When:** You need custom `PhaseInfo` that the parser cannot infer (e.g., a custom setup method name, or non-standard lifecycle).

```python
class MyOp(MicroOp):
    name = "my_op"
    phase = PhaseInfo(
        cpp_type="blaze::MyOp::Op",
        alias_suffix="MyOp",
        has_init_teardown=True,
        setup_method="setup_noc",
    )
```

When `cls.phase is not None` before auto-derivation, the auto-derived PhaseInfo is skipped. The manual phase is preserved. However, `_fill_phase_lifecycle_from_hpp()` will still update `init_is_empty` and `teardown_is_empty` from the header if manually-declared ct_args trigger the fallback path.

---

## 2.4.8 Edge cases in field parsing

### Passthrough fields

The simplest pattern -- the field name matches the schema arg name:

```cpp
static constexpr uint32_t num_tiles = A::num_tiles;
// Schema arg: "num_tiles"
```

### Aliased fields

The C++ field name differs from the schema arg:

```cpp
static constexpr uint32_t buffer_base = A::ncrisc_buffer_base;
// Schema arg: "ncrisc_buffer_base" (the A:: token, not the LHS)
```

The `emit()` method must register the arg as `f"{prefix}.ncrisc_buffer_base"`, not `f"{prefix}.buffer_base"`.

### Derived expressions

Computed from multiple schema args:

```cpp
static constexpr uint32_t out_tiles = A::n * A::m;
// Schema args: "n" and "m" (both extracted from RHS)
```

Each A:: token creates a separate `ParsedCTArg`. The `emit()` must provide both.

### Internal constants

No A:: dependency:

```cpp
static constexpr uint32_t MS_SEM_THRESHOLD = 1;
// No schema arg (skipped entirely)
```

### The decision tree for `uint32_t`/`bool` fields

```
RHS contains A:: tokens?
  |
  +-- No  --> Internal constant. Skip (no ParsedCTArg created).
  |
  +-- Yes --> Extract all A::<name> tokens.
              For each unique <name>, create a ParsedCTArg
              with source="derived".
```

### The `pop_input` asymmetry

`pop_input` can appear in two different forms across ops. Both are detected as pop flags, but differ in CT arg treatment:

**In RMSNorm's `ComputeCTArgs`:**

```cpp
static constexpr bool pop_input = A::pop_input;
```

The parser sees `bool`, classifies it as `source="derived"`. `_resolve_ct_arg_kind()` returns `Derived()`, so it becomes a regular CT arg. Because the name starts with `pop_`, it is also added to `parsed.pop_flags`.

**In Matmul's `CoreCTArgs`:**

```cpp
static constexpr Flag pop_in0 = A::pop_in0;
```

The parser sees `Flag`, classifies it as `source="flag"`. `_resolve_ct_arg_kind()` returns `None`, excluding it from the CT arg list. Because the name starts with `pop_`, it is also added to `parsed.pop_flags`.

Both patterns are detected as pop flags (the `pop_*` prefix check runs on all `seen_args` regardless of source type). The asymmetry is in CT arg treatment: RMSNorm's `pop_input` appears in `cls.ct_args` as a `Derived()` arg, while Matmul's `pop_in0` does not (it is excluded because `Flag` source maps to `None`). Both patterns work correctly at runtime because `emit()` handles the registration explicitly.

---

## 2.4.9 The five most common mistakes

### Mistake 1: Using `uint32_t` instead of `CB` for a circular buffer field

```cpp
// WRONG: parser classifies as "derived"
static constexpr uint32_t input = A::input;

// CORRECT: parser classifies as "cb", links to port descriptor
static constexpr CB input = A::input;
```

**Problem:** The parser sees `uint32_t`, finds `A::input` in the RHS, and classifies the field as `source="derived"`. The CT arg schema registers it as `Derived()` instead of `CB(input_port)`.

**Effect:** Registration succeeds silently. But the engine cannot resolve CB IDs because the schema says the field is a derived value, not a circular buffer reference. The error surfaces later at compile time or runtime, far from the root cause.

**Fix:** Always use the `CB` type alias for circular buffer fields. Similarly, use `Semaphore` for semaphore addresses, `PerCore` for per-core values, and `Flag` for boolean flags.

**Diagnostic path:** If auto-derivation succeeds but the op fails at compile time with CB resolution errors, check that CB fields in the header use `CB` not `uint32_t`. If you see `_resolve_ct_arg_kind` producing `Derived()` where you expect `CB()`, this is the root cause.

### Mistake 2: Forgetting to add a port descriptor for a new CB field

If you add `static constexpr CB scratch = A::scratch;` to the C++ header but forget to add `scratch: Internal = Internal(...)` to the Python class:

```
ValueError: my_op: kernel header ct_cb field 'scratch' has no matching port.
Valid ports: ['input', 'output']
```

**Fix:** Add the corresponding port descriptor to the Python class.

### Mistake 3: Using a non-standard semaphore field name

If you name a semaphore field `my_sem` instead of a valid protocol name:

```
ValueError: my_op: kernel header ct_sem field 'my_sem' is not a valid
SemProtocol value. Valid: ['noc0_receiver_semaphore',
'noc1_receiver_semaphore', 'receiver_semaphore', 'sender_semaphore']
```

**Fix:** Use one of the four valid `SemProtocol` names as the field name.

### Mistake 4: Placing the kernel header in the wrong directory

If `op.hpp` is at `blaze/ops/my_op/op.hpp` instead of `blaze/ops/my_op/kernels/op.hpp`:

**Problem:** `_find_kernel_hpp()` returns `None`. The entire auto-derivation is silently skipped. The op registers with empty `ct_args`, no `PhaseInfo`, and `is_inter_core = False`.

**Effect:** Registration succeeds. But the op fails at compile time with missing CT args or incorrect codegen.

**Fix:** Ensure the header is at `<op_dir>/kernels/op.hpp`.

### Mistake 5: Forgetting the aggregator include in `ops.hpp`

**Problem:** The Python registration succeeds (it reads the header file directly). But the C++ build fails because the metal compiler does not include the header.

**Fix:** Add `#include "blaze/ops/my_op/kernels/op.hpp"` to `blaze/ops/ops.hpp`.

---

## 2.4.10 The `_fill_phase_lifecycle_from_hpp()` fallback

When `ct_args` is manually declared (non-empty), `_auto_derive_from_kernel_hpp()` is skipped. However, empty-body detection for `init()` and `teardown()` is still valuable for codegen optimization. The `_fill_phase_lifecycle_from_hpp()` method handles this:

```python
# blaze/blaze_op.py -- MicroOp._fill_phase_lifecycle_from_hpp()
@classmethod
def _fill_phase_lifecycle_from_hpp(cls):
    if cls.phase is None:
        return
    hpp_path = cls._find_kernel_hpp()
    if hpp_path is None:
        return
    from .cpp_parser import parse_op_hpp

    parsed = parse_op_hpp(hpp_path)
    p = cls.phase
    cls.phase = PhaseInfo(
        cpp_type=p.cpp_type,
        alias_suffix=p.alias_suffix,
        has_init_teardown=p.has_init_teardown,
        setup_method=p.setup_method,
        init_is_empty=parsed.init_is_empty,       # from header
        teardown_is_empty=parsed.teardown_is_empty, # from header
    )
```

This selectively updates only `init_is_empty` and `teardown_is_empty` from the parsed header, preserving all other manually-declared phase attributes. The codegen can then skip emitting no-op lifecycle calls even when the rest of the metadata is manually declared.

**Guard:** If `cls.phase is None` (no manual phase), this method does nothing -- the developer must provide a phase when using manual CT args. If the kernel header is not found, it also does nothing.

---

## 2.4.11 Summary: the auto-derivation data flow

```
kernels/op.hpp
     |
     v
parse_op_hpp()                           # blaze/cpp_parser.py
     |
     +-- _OUTER_STRUCT_RE               -> struct_name
     +-- _INNER_STRUCT_RE               -> struct names (CTArgs groups)
     |     +-- _CT_TYPED_FIELD_RE       -> CB, Semaphore, PerCore, Flag fields
     |     +-- _PLAIN_FIELD_RE          -> uint32_t, bool fields
     |           +-- _A_NS_TOKEN_RE     -> A:: dependencies
     +-- _INIT_RE, _TEARDOWN_RE         -> has_init_teardown
     +-- _method_body_is_empty()        -> init_is_empty, teardown_is_empty
     +-- _SETUP_RE                      -> setup_method
     +-- pop_* flag detection           -> pop_flags
     |
     v
ParsedKernel
     |
     v
_auto_derive_from_kernel_hpp()           # blaze/blaze_op.py
     |
     +-- cls.op_class                   <- parsed.struct_name (only if non-empty)
     +-- cls.ct_args                    <- [CompileTimeArg(...) for each ParsedCTArg]
     +-- cls.is_inter_core              <- parsed.has_brisc and parsed.has_ncrisc
     +-- cls.has_init_teardown          <- parsed.has_init_teardown
     +-- cls.supports_pop_control       <- bool(parsed.pop_flags)
     +-- cls.pop_flags                  <- parsed.pop_flags
     +-- cls.kernel                     <- relative path to op.hpp
     +-- cls.phase                      <- PhaseInfo(...)
     |
     v
BlazeOp.register()                       # populates 5 global registries
```

---

## Key takeaways

1. **Auto-derivation makes the kernel header the single source of truth.** The Python class needs only `name`, ports, `emit()`, and `compose()`. Everything else is derived from `op.hpp`.

2. **`ParsedKernel` captures everything.** Struct name, CT args with RISC targeting, lifecycle metadata, empty-body detection, pop flags, and setup methods.

3. **Field classification is type-driven.** `CB` -> `"cb"`, `Semaphore` -> `"sem"`, `PerCore` -> `"per_core"`, `Flag` -> `"flag"`. Plain `uint32_t`/`bool` with `A::` tokens -> `"derived"`. No `A::` tokens -> internal constant (skipped).

4. **Merging handles cross-struct duplicates.** Same-named fields in multiple structs get their RISC flags OR'd. The final list has unique entries with combined targeting.

5. **`op_class` is conditionally overwritten.** Auto-derivation overwrites `op_class` only when `parsed.struct_name` is non-empty. If the outer struct is indented (not at column 0), the regex fails to match and `op_class` retains its `__init_subclass__` default. To ensure a specific `op_class` with manual `ct_args`, provide both explicitly.

6. **`CoreCTArgs` sets all three RISC flags.** This means ops with only `CoreCTArgs` (like Copy) are classified as `is_inter_core = True`. You can force `is_inter_core = True` but not `False` via class attributes.

7. **Use the right type alias.** `CB` for buffer IDs, `Semaphore` for semaphore addresses, `PerCore` for per-core values, `Flag` for boolean flags. Using `uint32_t` for a CB field silently breaks port linking.

---

**Previous:** [Section 2.3 -- The C++ Kernel Header](./03_cpp_kernel_header.md) | **Next:** [Chapter 3 -- The emit() API](../ch03_emit/index.md)
