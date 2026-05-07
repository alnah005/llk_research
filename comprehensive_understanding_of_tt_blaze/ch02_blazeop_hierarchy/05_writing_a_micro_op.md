# 05 -- Writing a MicroOp

This section is a step-by-step guide for authoring a new MicroOp, using the Mcast op as a concrete example. Mcast multicasts data from a sender core to all receiver cores -- it exercises inter-core communication, sender/receiver role flags, semaphore management, and both NCRISC and BRISC processor logic, making it a pedagogically rich reference for the full pattern.

## Step 1: Directory Creation

Create the op package under `blaze/ops/`:

```
blaze/ops/mcast/
    __init__.py          # empty (or re-exports Mcast)
    op.py                # Python class
    kernels/
        op.hpp           # C++ kernel header
```

The directory name (`mcast`) is the package name. It does not need to match the op's `name` attribute, though by convention it usually does. The `__init__.py` can be empty -- it exists only to make the directory a Python package for `pkgutil.iter_modules` discovery.

Auto-discovery finds the class via `register_all()` (described in [Section 04](./04_auto_discovery_and_registration.md)). No other wiring is needed.

## Step 2: The Python Class

The Python class declares identity, ports, and the two required methods (`emit()` and `compose()`):

```python
from ...blaze_op import BlazeOp, Input, MicroOp, Output
from ...fused_program import CBHandle, FusedProgram, TileInfo

class Mcast(MicroOp):
    name: str = "mcast"

    src: Input = Input()
    dst: Output = Output()

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        cls.emit(f, tensors["src"],
                 prefix=user_args.get("prefix", "mcast"))

    @staticmethod
    def emit(f, src, *, prefix, sender_sem=None,
             dst_num_pages=None, dst_tile_info=None):
        # ... composition logic (see Step 5)
```

Key observations:

- **`name = "mcast"`** makes this a concrete op (not abstract). This is the registry key and the default CT arg prefix.
- **Two ports**: `src` (Input) and `dst` (Output). Port names match the CB fields in the C++ header.
- **No `ct_args`, `phase`, `op_class`, or lifecycle flags** are declared. All of these are auto-derived from the kernel header at registration time.
- **No `math_fidelity` override** -- Mcast is a data movement op, so the default `"HiFi4"` is fine (TRISC is not used for compute here).

### What You Need to Provide vs. What Is Auto-Derived

| You provide | Auto-derived from C++ header |
|---|---|
| `name` | `op_class` |
| `Input()` / `Output()` ports | `ct_args` (full list) |
| `emit()` method | `phase` (PhaseInfo) |
| `compose()` method | `is_inter_core` |
| | `has_init_teardown` |
| | `supports_pop_control`, `pop_flags` |
| | `kernel` (path) |

## Step 3: The C++ Kernel Header

The header at `blaze/ops/mcast/kernels/op.hpp` defines the C++ Op struct with three CTArgs structs and the full lifecycle.

### The CTArgs Structs

Each struct declares what that set of RISC processors reads:

**`CoreCTArgs`** (all three RISCs): role flags that control per-core behavior.

```cpp
template <typename A>
struct CoreCTArgs {
    static constexpr Flag is_sender = A::is_sender;
    static constexpr Flag is_receiver = A::is_receiver;
    static constexpr Flag pop_src = A::pop_src;
    static constexpr Flag init_src = A::init_src;
};
```

Flags are `bool` values resolved per-core. `is_sender` is `True` only on the sender core; `is_receiver` is `True` on all mcast receiver cores. The C++ code uses `if constexpr (core_cta::is_sender) { ... }` to compile different code paths for different cores.

**`ReaderCTArgs`** (NCRISC): data movement inputs.

```cpp
template <typename A>
struct ReaderCTArgs {
    static constexpr Semaphore receiver_semaphore = ...;
    static constexpr CB dst = A::dst;
    static constexpr uint32_t dst_num_pages = A::dst_num_pages;
    static constexpr CB src = A::src;
    static constexpr uint32_t src_num_pages = A::src_num_pages;
    static constexpr bool src_is_tensor_backed = ...;
};
```

**`WriterCTArgs`** (BRISC): NOC multicast parameters.

```cpp
template <typename A>
struct WriterCTArgs {
    static constexpr uint32_t dest_noc_start_x = ...;
    static constexpr uint32_t dest_noc_start_y = ...;
    static constexpr uint32_t dest_noc_end_x = ...;
    static constexpr uint32_t dest_noc_end_y = ...;
    static constexpr Semaphore sender_semaphore = ...;
    static constexpr Semaphore receiver_semaphore = ...;
    static constexpr uint32_t data_size_bytes = ...;
    static constexpr CB src = A::src;
    static constexpr uint32_t src_num_pages = A::src_num_pages;
    static constexpr CB dst = A::dst;
    static constexpr uint32_t num_cores = A::num_cores;
    static constexpr bool is_part_of_receiver_grid = ...;
    static constexpr bool linked = A::linked;
    static constexpr bool posted = A::posted;
    static constexpr bool loopback = A::loopback;
};
```

Key rules for writing CTArgs structs:

- **`Flag` fields** in `CoreCTArgs` become per-core boolean role flags. The parser detects these and excludes them from the CT arg schema (they are handled separately as `PerCore` descriptors in `emit()`).
- **`CB` fields** reference port descriptors. The field name must match a port on the Python class (`src`, `dst`).
- **`Semaphore` fields** reference semaphore protocols. The name must match a `SemProtocol` value.
- **`uint32_t`/`bool` fields** with `A::` RHS become `"derived"` args.
- Fields named `pop_*` are automatically detected as pop-control flags.

### The Op Class

The `Op` template class implements the three lifecycle methods:

- **`init()`**: On NCRISC, sets up the sharded buffer for the source tensor. On BRISC, initializes persistent sender state (NOC multicast addresses).
- **`operator()()`**: On BRISC (sender), waits for the source CB, sends data via NOC multicast, and signals receivers. On NCRISC (receivers), waits for the semaphore signal and pushes the received data into the destination CB.
- **`teardown()`**: On BRISC (sender), tears down the persistent sender state and issues a barrier.

Mcast also defines `setup_src()`, a template instance method for re-initializing the source buffer with different parameters. Because `setup_src()` is a template method (not `static`), the parser's `_SETUP_RE` regex does not match it — `setup_method` is `None` for Mcast. The codegen handles this method through a separate mechanism outside the parser.

## Step 4: The `compose()` Method

`compose()` is the bridge from the graph API to the composition API. For Mcast it is a one-liner:

```python
@classmethod
def compose(cls, f, tensors, output, user_args):
    cls.emit(f, tensors["src"],
             prefix=user_args.get("prefix", "mcast"))
```

The `tensors` dict is keyed by port name (matching the `Input()` descriptors). The `user_args` dict carries any extra configuration the caller passed through the graph API.

## Step 5: The `emit()` Method

The `emit()` method is the heart of the op. It takes a `FusedProgram` and concrete inputs, and wires the op into the pipeline. Walking through Mcast's `emit()`:

### Input Resolution

```python
if isinstance(src, CBHandle):
    src_handle = src
else:
    src_handle = f.cb_from_tensor(src)
```

Inputs can be either `CBHandle` (from an upstream op's `emit()`) or `ttnn.Tensor` (from the caller). This dual-input pattern is universal across all ops.

### Output CB Allocation

```python
dst = f.cb_scratch(
    name=BlazeOp.cb_name(prefix, "dst"),
    num_pages=_dst_num_pages,
    core_ranges=f.all_cores,
    data_format=_dst_data_format,
    tile=_dst_tile_desc,
    page_size=_dst_page_size,
    balanced=False,
)
```

`cb_scratch()` allocates a scratch circular buffer. The `name` uses `cb_name()` (triple underscore delimiter) to create a unique identifier. The `balanced=False` flag opts out of temporal reuse because some receiver cores push but never pop the CB.

### Semaphore Allocation

```python
if sender_sem is None:
    sender_sem = f.semaphore(f"{prefix}.sender")
receiver_sem = f.semaphore(f"{prefix}.receiver")
```

`f.semaphore()` allocates a global semaphore, deduplicating by name across iterations.

### Per-Core Role Flags

```python
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_sender", f.sender_grid),
    f.flag(f"{prefix}.is_receiver", f.mcast_receiver_grid),
    f.flag(f"{prefix}.pop_src", f.all_cores),
    f.flag(f"{prefix}.init_src", f.sender_grid, needs_init),
])
```

`f.flag()` creates a per-core boolean CT arg. The second argument is a `CoreRangeSet` specifying which cores get `True`; all others get `False`. This maps directly to the `Flag` fields in `CoreCTArgs`.

### RISC-Specific CT Args

NCRISC (reader) args:

```python
f.ncrisc_ct_args([
    (f"{prefix}.src", src_handle),
    (f"{prefix}.src_num_pages", num_pages),
    (f"{prefix}.src_is_tensor_backed", src_is_tensor_backed),
    (f"{prefix}.receiver_semaphore", receiver_sem),
    (f"{prefix}.dst", dst),
    (f"{prefix}.dst_num_pages", dst.num_pages),
])
```

BRISC (writer) args:

```python
f.brisc_ct_args([
    (f"{prefix}.dest_noc_start_x", f.noc_start.x),
    (f"{prefix}.dest_noc_start_y", f.noc_start.y),
    # ... more NOC coordinates
    (f"{prefix}.sender_semaphore", sender_sem),
    (f"{prefix}.receiver_semaphore", receiver_sem),
    (f"{prefix}.data_size_bytes", data_size_bytes),
    (f"{prefix}.src", src_handle),
    (f"{prefix}.src_num_pages", num_pages),
    (f"{prefix}.dst", dst),
    # ... flags: linked, posted, loopback
])
```

The CT arg names follow the `"prefix.field"` convention and must match the field names in the C++ CTArgs structs exactly.

### Return Value

```python
return f.output("mcast", dst, prefix=prefix, src=src)
```

`f.output()` records the node in the shadow graph and returns a `CBHandle` for the destination CB. Downstream ops receive this handle and use its properties (`num_pages`, `page_size`, etc.) to configure their own inputs.

## Step 6: Testing

A typical MicroOp test uses the graph API path -- build a symbolic graph, bind real tensors, compile, run, and verify:

```python
import blaze
from blaze import ExternalTensor
from blaze.compiler import BlazeCompiler

with blaze.fuse() as ctx:
    blaze.mcast(
        ExternalTensor("src"),
        grid=all_cores)

compiler = BlazeCompiler(mesh_device)
program = compiler.compile(
    graph=ctx.graph,
    tensors={"src": ttnn_input},
    output_tensor=ttnn_output,
)
program.run()
# Verify: all cores should have the same data
```

For direct composition API tests (without the graph API):

```python
f = FusedProgram(
    kernel=None,  # codegen
    device=device,
    math_fidelity=ttnn.MathFidelity.HiFi4,
)
src_tensor = create_sharded_tensor(...)
out = Mcast.emit(f, src_tensor, prefix="test_mcast")
result = f.run()
# validate result
```

For CPU-only unit tests (no device), test registration and CT arg generation:

```python
def test_mcast_registration():
    from blaze.registry import get_op_spec
    spec = get_op_spec("mcast")
    assert spec.is_inter_core is True
    assert spec.has_init_teardown is True
    assert "pop_src" in spec.pop_flags
```

### Debugging Tools

When something goes wrong in the compilation or execution of an op:

- **`BLAZE_L1_PROFILE=1`**: Environment variable that enables L1 profiling, showing per-core CB utilization and timing.
- **`BLAZE_DEBUG_KERNELS=1`**: Emits debug prints from the generated kernel code, useful for tracing CT arg values at runtime.
- **`program.generated_kernel`**: After compilation, inspect the auto-generated C++ kernel source to verify the codegen output.
- **`compile_engines(ctx.graph).ct_tuples`**: Inspect the resolved CT arg tuples before they are sent to the device.

## Common Patterns Across MicroOps

Having examined Mcast in detail, here are patterns that recur across the op library:

1. **Input resolution**: every `emit()` starts by checking `isinstance(input, CBHandle)` vs `ttnn.Tensor`.
2. **Prefixed naming**: all CT args, CB names, and flags use the `prefix` parameter consistently.
3. **Role flags via `f.flag()`**: per-core booleans that control `if constexpr` branches in the C++ kernel.
4. **RISC-specific CT args**: `f.ncrisc_ct_args()`, `f.trisc_ct_args()`, `f.brisc_ct_args()` mirror the CTArgs structs in the header.
5. **CBHandle return**: every `emit()` returns a CBHandle so downstream ops can chain.
6. **Thin `compose()`**: extracts tensors by port name and delegates to `emit()`.

## Checklist Summary

| Step | What to create | Key concerns |
|---|---|---|
| Directory | `blaze/ops/<name>/op.py` + `kernels/op.hpp` | Naming convention, `__init__.py` |
| Python class | `MicroOp` subclass with `name`, ports | Port names must match C++ CB field names |
| C++ header | CT arg structs + `Op` template class | Type annotations (CB, Semaphore, Flag, etc.) determine parse behavior |
| `emit()` | Allocate CBs, register CT args, return `CBHandle` | Prefix namespacing, flag registration, input type handling |
| `compose()` | Unpack tensor dict, call `emit()` | Thin bridge -- logic belongs in `emit()` |
| Test | Build graph, compile, run, verify | Use `BlazeCompiler` for graph path; `FusedProgram.run()` for direct composition |

---
<- [04 -- Auto-Discovery and Registration](./04_auto_discovery_and_registration.md) | [Index](index.md) | [06 -- Writing a FusedOp](./06_writing_a_fused_op.md) ->
