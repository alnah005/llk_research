# The Three-Step `emit()` Pattern

## The Pattern at a Glance

Every `emit()` method in TT-Blaze follows the same three-step structure:

```
Step 1: Allocate resources      (CBs, semaphores)
Step 2: Declare CT args         (per-RISC compile-time arguments)
Step 3: Return output handle    (via f.output())
```

This structure is not optional or advisory. The FusedProgram context, the shadow graph recorder, and the kernel codegen pipeline all depend on emit() calls following this order. Breaking the pattern does not cause a Python exception -- it causes silent miscompilation or runtime faults on hardware.

## The `@staticmethod` Convention and the `FusedProgram` `f` Parameter

As established in Chapter 2, every `emit()` is a `@staticmethod`. It does not use `self` or `cls` -- the op class is stateless. All mutable state lives in the `FusedProgram` object passed as the first argument.

The actual method signature from `BlazeOp` (in `blaze/blaze_op.py`):

```python
@staticmethod
def emit(f, *args, **kwargs) -> Any:
    """Add this op to a FusedProgram. THE op definition."""
    raise NotImplementedError
```

MicroOp enforces that subclasses override both `emit()` and `compose()` at class creation time:

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
                f"{cls.__name__}: MicroOp must override compose() "
                "to support the Graph API path"
            )
```

The `emit()` check ensures every MicroOp has an imperative implementation. The `compose()` check ensures the op also supports the declarative graph-API path (covered in Chapter 2). Failing to override either raises a `TypeError` at import time -- before any code runs.

The `FusedProgram` object (`f`) is a thin context around a `BlazeProgram`. It provides:

- **CB allocation**: `f.cb_from_tensor()`, `f.cb_scratch()`, `f.cb_alias()`, `f.cb_from_view()`, `f.cb_from_shared_l1_views()`
- **CT arg registration**: `f.ncrisc_ct_args()`, `f.brisc_ct_args()`, `f.trisc_ct_args()`, `f.unified_ct_args()`, `f.per_core_unified_ct_args()`
- **Role flags**: `f.flag()`
- **Semaphores**: `f.semaphore()`
- **Grid info**: `f.all_cores`, `f.sender_grid`, `f.mcast_receiver_grid`, `f.noc_start`, `f.noc_end`, etc.
- **Output registration**: `f.output()`, `f.multi_output()`
- **Device context**: `f.device`

## The `prefix` Parameter: Namespacing CT Args

The `prefix` parameter is present on every `emit()` method. It serves a critical namespacing function: all CT arg names emitted by this op instance are prefixed with `prefix.`, making them unique in the generated C++ header.

For example, when `prefix="rmsnorm"`:
- The tuple `(f"{prefix}.input", inp)` becomes the CT arg named `rmsnorm.input`
- In the generated C++ header, this maps to `ct_args::rmsnorm::input`

The prefix must be unique per op instance within a single fused program. If two RMSNorm ops are composed into one fused op, they must use different prefixes (e.g., `"pre_rmsnorm"` and `"post_rmsnorm"`). Duplicate prefixes cause a name collision detected by CTArgEngine's `_validate_no_collisions()` method.

The CB_NAME_DELIMITER (`"___"`, defined in `blaze/blaze_op.py`) is used to construct scratch CB names under the same prefix scope:

```python
class BlazeOp:
    CB_NAME_DELIMITER: str = "___"
    CHILD_PREFIX_DELIMITER: str = "__"

    @classmethod
    def cb_name(cls, prefix: str, name: str) -> str:
        """Build a CB name under this op's prefix scope."""
        return f"{prefix}{cls.CB_NAME_DELIMITER}{name}"

    @classmethod
    def child_prefix(cls, prefix: str, name: str) -> str:
        """Build a nested-op prefix under this op's prefix scope."""
        return f"{prefix}{cls.CHILD_PREFIX_DELIMITER}{name}" if prefix else name
```

So `BlazeOp.cb_name("rmsnorm", "out_cb")` produces `"rmsnorm___out_cb"`, which becomes the scratch CB's internal name for the scratch_mapping lookup system. The `child_prefix` method uses a double underscore for nesting sub-ops within a parent prefix scope.

## The `cores` Parameter: Which Cores Run This Op

The `cores` parameter (typically `cores: ttnn.CoreRangeSet`) specifies which cores receive this op's CT args and role flags. It controls:

1. **Role flags**: `f.flag(f"{prefix}.is_active", cores)` sets the flag to 1 on those cores, 0 elsewhere.
2. **Scratch CB placement**: `f.cb_scratch(..., core_ranges=cores, ...)` allocates the scratch buffer on those cores.
3. **Graph recording**: The shadow graph records which cores this op runs on.

Not all ops take an explicit `cores` parameter. Some derive it from inputs (e.g., from the input tensor's shard spec grid), and inter-core ops like Mcast use `f.all_cores`, `f.sender_grid`, or `f.mcast_receiver_grid` from the FusedProgram context.

## How CT Arg Tuples Become C++ Header Fields

Each CT arg tuple `(f"{prefix}.field", value)` follows a dot-separated naming convention. The first segment is the prefix (the op instance name), and the second is the field name. During kernel codegen, these are transformed into nested C++ namespaces:

```
Python:   (f"{prefix}.num_tiles", 4)
                    |
                    v
C++:      ct_args::<prefix>::num_tiles
```

The auto-generated kernel header defines a `ct_args` namespace with nested per-op namespaces:

```cpp
namespace ct_args {
namespace rmsnorm {
    constexpr uint32_t input = /* CB ID */;
    constexpr uint32_t gamma = /* CB ID */;
    constexpr uint32_t output = /* CB ID */;
    constexpr uint32_t num_tiles = /* tile count */;
    constexpr uint32_t epsilon = /* float bits */;
    constexpr uint32_t scalar = /* float bits */;
    // ...
}
}
```

The kernel reads these via the template struct pattern (the standard approach in all TT-Blaze op.hpp files):

```cpp
// From blaze/ops/rmsnorm/kernels/op.hpp
template <typename A>
struct ReaderCTArgs {
    static constexpr CB input = A::input;
    static constexpr CB gamma = A::gamma;
    static constexpr uint32_t num_tiles = A::num_tiles;
    static constexpr bool input_is_tensor_backed = A::input_is_tensor_backed;
};
```

Values in tuples undergo type coercion:
- `CBHandle` values are converted to `int` via `CBHandle.__int__()`, yielding the CB ID
- `float` values are converted via `_float_to_bits()` (IEEE 754 bit-pattern as uint32_t)
- `bool` values are passed as `int(True)` = 1 or `int(False)` = 0

## The CBHandle Chain: emit() Returns Feed Into Next Op's Inputs

The return value of `emit()` is always a `CBHandle` (or `MultiOutput` for multi-output ops). This handle is the sole interface between ops. The calling code passes one op's returned handle as input to the next op's `emit()`:

```python
# Mcast returns a CBHandle pointing to the multicast destination CB
act = Mcast.emit(f, input_tensor, prefix="act_mcast")

# That handle feeds directly into RMSNorm as its input
result = RMSNorm.emit(f, act, gamma, prefix="rmsnorm", cores=cores)

# Which feeds into Copy for write-back
output = Copy.emit(f, result, output_tensor, prefix="flush")
```

When RMSNorm receives `act` (a CBHandle), it skips the `f.cb_from_tensor()` path and uses the handle directly:

```python
if isinstance(input_handle, CBHandle):
    inp = input_handle  # no new CB allocated; reuse upstream's output CB
else:
    inp = f.cb_from_tensor(input_handle)
```

The CB ID from Mcast's output CB becomes RMSNorm's input CB. No data copy occurs -- both ops reference the same L1 circular buffer. This zero-coupling principle is the core of TT-Blaze's composition model. Section 04 explores this data flow in detail.

## Step 1: Allocate Resources

The first thing `emit()` does is allocate the circular buffers it needs. There are several allocation methods, each for a different source:

| Method | When to use |
|--------|-------------|
| `f.cb_from_tensor(tensor)` | Input data from a sharded tensor |
| `f.cb_scratch(name, ...)` | Internal working buffer in L1 |
| `f.cb_alias(source_cb)` | Reinterpret an existing CB with different tile format |
| `f.cb_from_view(view)` | Sub-region of a fused weight buffer |
| `f.cb_from_shared_l1_views([...])` | Multiple direct-address handles from one packed tensor |
| `f.semaphore(name)` | Cross-core synchronization point |

Section 02 covers each method in detail.

## Step 2: Declare CT Args Per Processor

After allocating resources, `emit()` declares which compile-time arguments each RISC processor should receive. The RISC processors correspond to the CT arg structs established in Chapter 2:

| Method | Target RISC | C++ struct |
|--------|-------------|------------|
| `f.ncrisc_ct_args([...])` | NCRISC (data movement reader) | ReaderCTArgs |
| `f.brisc_ct_args([...])` | BRISC (data movement writer) | WriterCTArgs |
| `f.trisc_ct_args([...])` | TRISC (compute) | ComputeCTArgs |
| `f.unified_ct_args([...])` | All three RISCs | Values appear in all three per-RISC structs |
| `f.per_core_unified_ct_args([...])` | All RISCs, per-core varying | Per-core flags |

Role flags are a special form of per-core CT args built with `f.flag()`:

```python
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_active", cores),        # 1 on active cores, 0 elsewhere
    f.flag(f"{prefix}.pop_input", cores, True),   # conditional flag
])
```

The rule: **all flags must be emitted, even when False**. The per-core mechanism needs to know which cores get 0 and which get 1. Omitting a flag entirely causes a kernel compile-time failure.

Section 03 covers the full CT arg registration system.

## Step 3: Return a CBHandle via `f.output()`

Every `emit()` ends with a call to `f.output()`, which does two things:

1. **Records the op in the shadow graph** -- wiring input edges, creating the node, and capturing CT arg values for codegen and debugging.
2. **Returns a CBHandle** -- the output handle that downstream ops receive.

The actual signature of `f.output()` (from `blaze/fused_program.py`):

```python
def output(
    self,
    op_name: str,
    handle: CBHandle | None = None,
    *,
    cb_id: int | None = None,
    num_pages: int | None = None,
    page_size: int | None = None,
    core_ranges=None,
    data_format=None,
    tile_desc=None,
    grid=None,
    does_produce_output=True,
    prefix: str = "",
    **port_sources,
) -> CBHandle:
```

Key parameters:
- `op_name`: The op's registered name (e.g., `"rmsnorm"`, `"mcast"`)
- `handle` or `cb_id`: The output CB -- either a CBHandle or a raw CB ID
- `prefix`: The CT arg prefix, used for graph recording
- `does_produce_output`: Set to `False` for synchronization-only ops that do not produce data output (default `True`)
- `**port_sources`: Input port-to-source mappings (CBHandle or tensor), used for wiring edges in the shadow graph

The method constructs a new CBHandle, attaches graph metadata (`_node_id`, `_port_name`) so downstream `f.output()` calls can wire edges, and advances the internal op index for CB lifetime tracking.

The typical call passes the output CB as a handle (or as `cb_id=`) and lists input sources as keyword arguments matching the op's `Input()` descriptor names:

```python
return f.output("rmsnorm", cb_id=out_cb, core_ranges=cores,
                prefix=prefix, input=inp, gamma=gamma_source)
```

The `**port_sources` kwargs (like `input=inp`, `gamma=gamma_source`) wire the shadow graph edges. If a source is a `CBHandle` with a `_node_id`, `f.output()` creates an edge from that producer node to this consumer node. If a source is an `ExternalTensor` or raw tensor, it becomes an external input port in the graph.

## Complete Walkthrough: RMSNorm.emit()

Below is the actual `RMSNorm.emit()` from `blaze/ops/rmsnorm/op.py`, annotated to show all three steps. This code is taken directly from the source, with annotations added.

```python
@staticmethod
def emit(
    f: FusedProgram,
    input_handle: CBHandle | ttnn.Tensor,
    gamma_tensor: ttnn.Tensor,
    *,
    prefix: str = "rmsnorm",
    cores: ttnn.CoreRangeSet,
    epsilon: float = 1e-6,
    scalar: float = 1.0,
    fp32_dest_acc_en: bool = False,
    rsqrt_fast_approx: bool = False,
    num_tiles: int | None = None,
    pop_input: bool = True,
    dst_cb: int | CBHandle | None = None,
    notify_input: bool | None = None,
    width: int | None = None,
) -> CBHandle:
```

**Signature notes:**
- `input_handle` accepts either a `CBHandle` (from an upstream op) or a raw `ttnn.Tensor` (for standalone use). This dual-input pattern is common -- the method handles both cases.
- `prefix` defaults to `"rmsnorm"` but must be overridden when composing multiple RMSNorm instances.
- `cores` has no default -- callers must specify which cores run this op.

### Step 1: Resource Allocation

```python
    # --- STEP 1: Allocate CBs ---

    # Input: either reuse upstream CBHandle or allocate from tensor
    if isinstance(input_handle, CBHandle):
        inp = input_handle
    else:
        if has_row_tiles(input_handle):
            # Row-tile tensors (1xN) need reinterpretation for compute
            if width is None:                              # guard: use explicit width if provided
                width = input_handle.shape[1]
            itile, n_tiles = interpret_tile(width)
            page_sz = itile.get_tile_size(ttnn.bfloat16)
            num_tiles = num_tiles or n_tiles
            if scalar == 1.0:                              # guard: only auto-compute if default
                scalar = 1.0 / math.sqrt(float(width))
            inp = f.cb_from_tensor(input_handle, tile=itile, page_size=page_sz)
        else:
            inp = f.cb_from_tensor(input_handle)

    # Gamma weights: allocate CB matching input tile geometry
    gamma_cb = (gamma_tensor if isinstance(gamma_tensor, CBHandle)
                else f.cb_from_tensor(gamma_tensor, tile=inp.tile_desc,
                                      page_size=inp.page_size))

    # Compute tile count
    compute_n = num_tiles if num_tiles is not None else inp.num_pages

    # NCRISC tile count (different derivation when input is a tensor)
    n = (inp.num_pages if not isinstance(input_handle, CBHandle)
         else (num_tiles if num_tiles is not None else inp.num_pages))

    # Output scratch CB (or caller-provided dst_cb)
    out_tile = inp.tile_desc
    out_page_sz = inp.page_size
    out_fmt = inp.data_format
    if dst_cb is not None:
        out_cb = dst_cb
    else:
        out_cb = f.cb_scratch(
            name=BlazeOp.cb_name(prefix, "out_cb"),  # e.g. "rmsnorm___out_cb"
            num_pages=compute_n,
            core_ranges=cores,
            data_format=out_fmt,
            tile=out_tile,
            page_size=out_page_sz,
        )
```

Three CBs allocated: input (from tensor or upstream handle), gamma (from tensor), and output (scratch or caller-provided). The scratch CB name uses `BlazeOp.cb_name()` to produce the triple-underscore-delimited form `"rmsnorm___out_cb"`.

### Step 2: CT Arg Declaration

```python
    # --- STEP 2: Declare CT args ---

    # Determine if NCRISC should push the input CB on init
    if notify_input is not None:
        input_is_tensor_backed = notify_input
    else:
        input_is_tensor_backed = not isinstance(input_handle, CBHandle)

    # Per-core flags (all RISCs)
    f.per_core_unified_ct_args([
        f.flag(f"{prefix}.is_active", cores),
        f.flag(f"{prefix}.pop_input", cores, pop_input),
    ])

    # NCRISC (reader) CT args
    f.ncrisc_ct_args(
        [
            (f"{prefix}.input", inp),                # CB ID via __int__()
            (f"{prefix}.gamma", gamma_cb),            # CB ID
            (f"{prefix}.num_tiles", n),               # tile count
            (f"{prefix}.input_is_tensor_backed",
             int(input_is_tensor_backed)),             # 0 or 1
        ]
    )

    # TRISC (compute) CT args
    f.trisc_ct_args(
        [
            (f"{prefix}.input", inp),                 # CB ID
            (f"{prefix}.gamma", gamma_cb),            # CB ID
            (f"{prefix}.output", out_cb),             # CB ID
            (f"{prefix}.num_tiles", compute_n),
            (f"{prefix}.fp32_dest_acc_en",
             1 if fp32_dest_acc_en else 0),
            (f"{prefix}.rsqrt_fast_approx",
             1 if rsqrt_fast_approx else 0),
            (f"{prefix}.epsilon", _float_to_bits(epsilon)),  # float -> uint32
            (f"{prefix}.scalar", _float_to_bits(scalar)),    # float -> uint32
        ]
    )
```

Note how different RISCs receive different arguments:
- **NCRISC** (reader) gets input/gamma CB IDs, tile count, and the tensor-backed flag (determines whether it needs to push the input CB).
- **TRISC** (compute) gets all CB IDs including the output, plus compute-specific parameters like epsilon and scalar (encoded as uint32 bit patterns via `_float_to_bits()`).
- **BRISC** (writer) gets no CT args from RMSNorm -- the output stays in a scratch CB. A downstream Copy or Gather op handles the write-back.

The per-core flags (`is_active`, `pop_input`) go to all RISCs via `per_core_unified_ct_args`. On cores outside `cores`, these flags are 0, and the kernel's `if constexpr` compiles them out to zero-cost no-ops.

### Step 3: Return Output Handle

```python
    # --- STEP 3: Return output CBHandle ---

    gamma_source = (gamma_tensor if isinstance(gamma_tensor, CBHandle)
                    else ExternalTensor(name=f"{prefix}.gamma",
                                        tensor=gamma_tensor))
    return f.output(
        "rmsnorm",
        cb_id=out_cb,
        core_ranges=cores,
        prefix=prefix,
        input=inp,           # port source: wires edge from upstream
        gamma=gamma_source,  # port source: external tensor or CBHandle
    )
```

The `f.output()` call:
1. Looks up the stored CB metadata for `out_cb` to fill in `num_pages`, `page_size`, `data_format`, and `tile_desc`.
2. Creates an `OpNode` in the shadow graph with op name `"rmsnorm"`.
3. Wires input edges: `input=inp` creates an edge from the upstream op's output to this node's `input` port. `gamma=gamma_source` either wires an edge (if gamma is a CBHandle from upstream) or marks it as an external tensor input.
4. Stamps the returned CBHandle with `_node_id` and `_port_name` so a downstream `f.output()` can wire an edge back to this node.
5. Returns the CBHandle that downstream ops will consume.

## Comparison: Compute Op vs. Inter-Core Op vs. Data-Movement Op

The three-step pattern is the same for all op types, but the resource allocation and CT arg distribution differ:

**Compute op (RMSNorm):**
- Allocates input CB, scratch output CB
- CT args go to NCRISC (reader) and TRISC (compute)
- Flags: `is_active`, `pop_input`

**Inter-core op (Mcast):**
- Allocates source CB, scratch destination CB, sender + receiver semaphores
- CT args go to NCRISC (multicast sender logic) and BRISC (NOC coordinates, semaphores)
- Flags: `is_sender`, `is_receiver`, `pop_src`, `init_src`
- BRISC gets NOC coordinates (`dest_noc_start_x/y`, `dest_noc_end_x/y`) from `f.noc_start` and `f.noc_end`

**Data-movement op (Copy):**
- Allocates source and destination CBs
- All CT args go to all RISCs via `f.unified_ct_args()`
- Flags: `is_active`
- Can run on either NCRISC or BRISC (selectable via `processor` parameter)

## The Contract: What `emit()` Must Do

Every `emit()` implementation must:

1. **Allocate all CBs it needs** -- either from tensors, scratch, views, or upstream handles.
2. **Emit ALL flags, even when False** -- the C++ Op struct reads every declared CT arg. A missing arg causes a runtime fault. `f.flag(name, cores, False)` emits 0 on all cores, which is different from not emitting the flag at all.
3. **Register CT args on the correct RISCs** -- reader args on NCRISC, compute args on TRISC, writer/NOC args on BRISC.
4. **Call `f.output()` exactly once** (or `f.multi_output()` for multi-output ops) -- this records the op in the shadow graph and returns the handle.
5. **Return a CBHandle** -- the output for downstream consumption.

Breaking any of these contracts does not raise a Python exception at composition time. It manifests as a kernel compilation failure (missing CT arg), a runtime hang (missing semaphore or flag), or silent data corruption (wrong CB ID passed to the wrong RISC).

## Summary: The Three Steps at a Glance

| Step | What Happens | Key Methods |
|------|-------------|-------------|
| 1. Allocate | CBs for input, scratch, output; semaphores | `f.cb_from_tensor()`, `f.cb_scratch()`, `f.semaphore()` |
| 2. Declare CT args | Per-RISC compile-time values; role flags | `f.ncrisc_ct_args()`, `f.trisc_ct_args()`, `f.brisc_ct_args()`, `f.per_core_unified_ct_args()` |
| 3. Return handle | CBHandle + shadow graph node | `f.output()` |

The next three sections dive into each step in detail.
