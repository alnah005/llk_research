# Role Flags and CT Arg Registration

## Overview

After allocating CBs (Step 1), `emit()` must tell each RISC processor what to do. This is done through compile-time argument (CT arg) registration -- a set of `(name, value)` tuples that become constants baked into the compiled kernel. This section covers the full CT arg registration system: role flags, per-RISC methods, the CTArgEngine, type coercion, semaphore allocation, runtime args (including C++ access patterns), and the contrast between CT and RT args.

## Role Flags: `f.flag()` and Per-Core CT Args

Role flags are the first CT args emitted in Step 2. They tell each core whether it participates in this op and, if so, what role it plays.

**The `f.flag()` method** (from `blaze/fused_program.py`):

```python
def flag(self, name, cores, enabled=True):
    """Build a per-core CT arg tuple for a boolean flag.

    Returns a (name, mapping) tuple suitable for passing to
    per_core_unified_ct_args() and variants. The flag is 1 on the given
    cores when enabled, 0 when disabled. Cores not in the range always
    get 0 (via other_value default).
    """
    return (name, {cores: int(enabled)})
```

The return value is a tuple `(name, {CoreRangeSet: int})` -- a per-core varying CT arg where cores in the range get the `enabled` value and all other cores get 0.

### Compute Op Flags

Compute ops typically emit two flags:

```python
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_active", cores),            # 1 on active cores
    f.flag(f"{prefix}.pop_input", cores, pop_input),  # conditional
])
```

- `is_active`: The master enable flag. On inactive cores (outside the op's core range), this is 0, and the C++ `if constexpr` compiles the entire op body to nothing.
- `pop_input`: Whether to pop the input CB after consuming it. This is True by default but can be set to False when a downstream op needs to read the same data again.

### Inter-Core Op Flags

Inter-core ops like Mcast have richer flag sets that distinguish sender from receiver roles:

```python
# From Mcast.emit()
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_sender", f.sender_grid),
    f.flag(f"{prefix}.is_receiver", f.mcast_receiver_grid),
    f.flag(f"{prefix}.pop_src", f.all_cores),
    f.flag(f"{prefix}.init_src", f.sender_grid, needs_init),
])
```

- `is_sender`: 1 only on the sender core. Controls whether the BRISC multicast path executes.
- `is_receiver`: 1 on all cores except the sender. Controls the receiver wait path.
- `pop_src`: 1 on all cores (both sender and receivers pop the source after consumption).
- `init_src`: 1 on the sender when the source is tensor-backed (needs `cb_reserve_back` + `cb_push_back` to unblock the first `cb_wait_front`).

### The Contract: All Flags Must Be Emitted

The C++ Op struct declares all its CT arg fields in a compile-time args struct. When the kernel reads a CT arg name that was never emitted from Python, it faults at program build time (missing named arg) or reads garbage. There is no mechanism for default values at the C++ level.

This means **every flag must be emitted, even when the value is False/0 on all cores**. The correct way to disable a flag everywhere is:

```python
f.flag(f"{prefix}.pop_src", f.all_cores, False)  # emits 0 on all cores
```

NOT:

```python
# WRONG: omitting the flag entirely causes a missing CT arg fault
# (no call to f.flag for pop_src)
```

## How Flags Compile Out: `if constexpr` Becomes Zero-Cost

In the generated C++ kernel, per-core flags are read as compile-time constants. From the actual `RMSNorm` kernel at `blaze/ops/rmsnorm/kernels/op.hpp`:

```cpp
template <typename A>
struct CoreCTArgs {
    static constexpr Flag is_active = A::is_active;
};

template <typename Args>
class Op {
    using core_cta = CoreCTArgs<Args>;
    using dm0_cta = ReaderCTArgs<Args>;
    using compute_cta = ComputeCTArgs<Args>;

    public:
        void init() {
#if defined(COMPILE_FOR_NCRISC)
            if constexpr (core_cta::is_active) {
                if constexpr (dm0_cta::input_is_tensor_backed) {
                    cb_reserve_back(dm0_cta::input, dm0_cta::num_tiles);
                    cb_push_back(dm0_cta::input, dm0_cta::num_tiles);
                }
                cb_reserve_back(dm0_cta::gamma, dm0_cta::num_tiles);
                cb_push_back(dm0_cta::gamma, dm0_cta::num_tiles);
            }
#endif
        }
        void operator()() {
#if defined(COMPILE_FOR_TRISC)
            if constexpr (core_cta::is_active) {
                constexpr uint32_t num_tiles = compute_cta::num_tiles;
                constexpr float scalar = __builtin_bit_cast(float, compute_cta::scalar);
                // ... entire compute body ...
            }
#endif
        }
};
```

Because `core_cta::is_active` is a compile-time constant, `if constexpr` eliminates the entire op body on cores where the flag is 0. The compiled kernel for inactive cores contains zero instructions for this op -- no branch, no comparison, nothing. This is why the role-flag pattern scales to fused kernels with dozens of ops: each core only executes the ops assigned to it.

Note the nested `if constexpr` usage: `input_is_tensor_backed` further eliminates the init push path when the input comes from an upstream op's CB rather than a tensor. These compile-time branches compose -- each one removes another code path at compile time.

Also note that the kernel uses `__builtin_bit_cast(float, compute_cta::scalar)` to decode the `_float_to_bits()`-encoded float back to its original value.

## Per-RISC CT Arg Methods

After flags, `emit()` registers the per-RISC CT args. Each method targets a specific RISC processor:

### `f.ncrisc_ct_args(args)`

Registers CT args for the NCRISC processor (data movement reader). These appear in the C++ `ReaderCTArgs` struct.

Typical NCRISC args: input CB IDs, tile counts, tensor-backed flags.

```python
# From RMSNorm.emit()
f.ncrisc_ct_args([
    (f"{prefix}.input", inp),          # CBHandle -> cb_id via __int__()
    (f"{prefix}.gamma", gamma_cb),
    (f"{prefix}.num_tiles", n),
    (f"{prefix}.input_is_tensor_backed", int(input_is_tensor_backed)),
])
```

Internally, each per-RISC method does three things:

```python
def ncrisc_ct_args(self, args, **kwargs):
    self._track_cb_arg_positions(args, ["ncrisc"])
    self._capture_ct_values(args)
    self.program.ncrisc_ct_args(args, **kwargs)
```

1. Tracks which arg positions hold CB IDs (for CB lifetime analysis).
2. Captures `(name, int(value))` pairs into `_pending_ct_values` (for the shadow graph).
3. Forwards the args to the underlying BlazeProgram for kernel descriptor assembly.

### `f.brisc_ct_args(args)`

Registers CT args for the BRISC processor (data movement writer). These appear in the C++ `WriterCTArgs` struct.

Typical BRISC args: NOC coordinates, semaphore addresses, data sizes for multicast.

```python
# From Mcast.emit()
f.brisc_ct_args([
    (f"{prefix}.dest_noc_start_x", f.noc_start.x),
    (f"{prefix}.dest_noc_start_y", f.noc_start.y),
    (f"{prefix}.dest_noc_end_x", f.noc_end.x),
    (f"{prefix}.dest_noc_end_y", f.noc_end.y),
    (f"{prefix}.num_cores", f.num_mcast_cores),
    (f"{prefix}.sender_semaphore", sender_sem),
    (f"{prefix}.receiver_semaphore", receiver_sem),
    (f"{prefix}.data_size_bytes", data_size_bytes),
    (f"{prefix}.src", src_handle),
    (f"{prefix}.src_num_pages", num_pages),
    (f"{prefix}.dst", dst),
    (f"{prefix}.is_part_of_receiver_grid",
     int(f.mcast_range.contains(f.sender))),
    (f"{prefix}.linked", 1),
    (f"{prefix}.posted", 1),
    (f"{prefix}.loopback", 0),
])
```

### `f.trisc_ct_args(args)`

Registers CT args for the TRISC processor (compute). These appear in the C++ `ComputeCTArgs` struct.

Typical TRISC args: all CB IDs (input, weight, output), compute parameters (epsilon, scalar, accumulation mode).

```python
# From RMSNorm.emit()
f.trisc_ct_args([
    (f"{prefix}.input", inp),
    (f"{prefix}.gamma", gamma_cb),
    (f"{prefix}.output", out_cb),
    (f"{prefix}.num_tiles", compute_n),
    (f"{prefix}.fp32_dest_acc_en", 1 if fp32_dest_acc_en else 0),
    (f"{prefix}.rsqrt_fast_approx", 1 if rsqrt_fast_approx else 0),
    (f"{prefix}.epsilon", _float_to_bits(epsilon)),
    (f"{prefix}.scalar", _float_to_bits(scalar)),
])
```

### `f.unified_ct_args(args)`

Registers CT args for ALL three RISCs simultaneously. Useful for data-movement ops where all processors need the same information.

```python
# From Copy.emit() — all 15 CT args broadcast to all RISCs
f.unified_ct_args([
    (f"{prefix}.src", src_cb),
    (f"{prefix}.dst", dst_cb),
    (f"{prefix}.num_pages", src_cb.num_pages),
    (f"{prefix}.num_bytes", num_bytes),
    (f"{prefix}.src_addr", src_addr if src_addr is not None else 0),
    (f"{prefix}.dst_addr", dst_addr if dst_addr is not None else 0),
    (f"{prefix}.dest_noc_x", dest_noc_x),
    (f"{prefix}.dest_noc_y", dest_noc_y),
    (f"{prefix}.wait_src", int(wait_src)),
    (f"{prefix}.pop_src", int(pop_src)),
    (f"{prefix}.push_dst", int(push_dst)),
    (f"{prefix}.src_is_cb", int(src_is_cb)),
    (f"{prefix}.dst_is_cb", int(dst_is_cb)),
    (f"{prefix}.is_remote", int(is_remote)),
    (f"{prefix}.use_ncrisc", int(processor == Risc.NCRISC)),
])
```

Internally, `unified_ct_args()` tracks CB arg positions for all three RISCs:

```python
def unified_ct_args(self, args, **kwargs):
    self._track_cb_arg_positions(args, ["ncrisc", "brisc", "trisc"])
    self._capture_ct_values(args)
    self.program.unified_ct_args(args, **kwargs)
```

## `f.per_core_unified_ct_args()` and Per-RISC Per-Core Variants

**`per_core_unified_ct_args(args)`:** Per-core varying CT args sent to all three RISCs. This is the most common variant, used for role flags. Each arg is a `(name, mapping)` tuple where `mapping` is a `{CoreRangeSet: value}` dict.

**Per-RISC variants:** For cases where per-core values should only go to one RISC:

```python
f.per_core_ncrisc_ct_args(args)   # NCRISC only
f.per_core_trisc_ct_args(args)    # TRISC only
f.per_core_brisc_ct_args(args)    # BRISC only
```

The underlying BlazeProgram implementation:

```python
def per_core_unified_ct_args(self, args, other_value=0):
    """Add per-core CT args to all RISCs."""
    self._per_core_ct_args_impl(args, other_value, ALL_RISCS)

def per_core_ncrisc_ct_args(self, args, other_value=0):
    self._per_core_ct_args_impl(args, other_value, (Risc.NCRISC,))
```

All per-core methods enforce name uniqueness via `_per_core_ct_args_check()`:

```python
def _per_core_ct_args_check(self, args):
    for name, mapping in args:
        if name in self._ct_arg_names_seen:
            raise ValueError(f"Duplicate per-core CT arg: {name!r}")
        self._ct_arg_names_seen.add(name)
```

## CTArgEngine: Auto-Prefixing, Type Validation, Grouping

The `CTArgEngine` (in `blaze/ct_args.py`) is the automated path for CT arg generation. It walks a `BlazeGraph`, looks up each op's `CTArgSchema`, and generates per-node, per-RISC `(name, value)` tuples. This engine operates on the graph-compilation path (via BlazeCompiler), not the imperative `emit()` path. When you call `f.ncrisc_ct_args()` directly in `emit()`, you bypass the engine. The engine's validation is applied separately when the shadow graph is built.

**CTArgSchema** declares what CT args an op needs:

```python
@dataclass
class CTArgSchema:
    op_name: str
    args: list[CTArgSpec] = field(default_factory=list)
```

Each `CTArgSpec` declares one arg:

```python
@dataclass
class CTArgSpec:
    name: str           # Base name before prefixing (e.g., "out_w")
    risc: str           # "ncrisc", "brisc", or "trisc"
    type: str           # C++ type ("uint32_t", "bool")
    source: str         # "cb", "sem", "derived", "per_core", "user"
    source_key: str     # Key into the relevant assignment dict
    description: str    # Human-readable doc
```

**Auto-prefixing:** The engine computes a prefix from the op's spec name or a `ct_prefix` kwarg override:

```python
def _compute_prefix(self, node, graph, merged_user=None) -> str:
    source = merged_user if merged_user is not None else node.kwargs
    prefix = source.get("ct_prefix", node.spec.name)
    if not prefix:
        return ""
    return f"{prefix}."
```

Each arg name becomes `f"{prefix}{spec.name}"`, e.g., `"matmul.out_w"`. Setting `ct_prefix=""` disables prefixing -- used for fused ops where arg names are already fully qualified.

**Shared prefix deduplication:** When multiple nodes share the same explicit `ct_prefix` (e.g., gate and up matmul both using `"gu"`), the engine generates args from the first node and reuses them for subsequent nodes:

```python
if explicit_prefix and prefix in shared_prefix_args:
    result[node.id] = shared_prefix_args[prefix]
    continue
```

**Collision detection:** `_validate_no_collisions()` ensures no two args share the same prefixed name within a RISC:

```python
def _validate_no_collisions(self, all_args):
    per_risc = {"ncrisc": {}, "brisc": {}, "trisc": {}}
    seen_ids = set()
    for node_id, args in all_args.items():
        args_id = id(args)
        if args_id in seen_ids:
            continue  # shared-prefix dedup
        seen_ids.add(args_id)
        for arg in args:
            existing = per_risc[arg.risc].get(arg.name)
            if existing is not None:
                raise ValueError(
                    f"CT arg name collision on {arg.risc}: '{arg.name}' "
                    f"defined by both '{existing}' and '{node_id}'"
                )
            per_risc[arg.risc][arg.name] = node_id
```

**Value resolution:** Values are resolved from different sources based on the `CTArgSpec.source` field:

| Source | Resolution |
|--------|-----------|
| `"cb"` | CB ID from CB assignments dict |
| `"sem"` | Semaphore ID from sem assignments dict |
| `"derived"` / `"user"` / `"grid"` | From merged user dict (kwargs + overrides) |
| `"per_core"` | Placeholder; resolved at assembly time |

Float values are auto-detected and converted:

```python
if isinstance(val, float):
    from .utils import _float_to_bits
    return _float_to_bits(val), True
return int(val), True
```

## Type Coercion: `CBHandle.__int__()` and `_float_to_bits()`

All CT arg values must be integers (uint32_t in C++). Two coercion mechanisms handle non-integer types.

### `CBHandle.__int__()`

When a `CBHandle` appears as a value in a CT arg tuple, FusedProgram's `_capture_ct_values()` converts it:

```python
def _capture_ct_values(self, args):
    """Coerce CBHandle values to int and record for shadow graph."""
    self._pending_ct_values.update({k: int(v) for k, v in args})
```

`int(handle)` invokes `CBHandle.__int__()`, which returns the `cb_id`:

```python
class CBHandle:
    def __int__(self) -> int:
        return self.cb_id

    def __index__(self) -> int:
        return self.cb_id
```

This allows writing `(f"{prefix}.input", inp)` instead of `(f"{prefix}.input", inp.cb_id)`. The `__index__()` method additionally supports using CBHandle in array indexing and `hex()` calls.

### `_float_to_bits()`

Floating-point parameters (epsilon, scalar, etc.) are encoded as their IEEE 754 bit pattern:

```python
def _float_to_bits(x: float) -> int:
    """Convert float to uint32_t bit representation for kernel CT args."""
    return struct.unpack("I", struct.pack("f", x))[0]
```

The kernel decodes them on the C++ side using `__builtin_bit_cast`:

```cpp
constexpr float scalar = __builtin_bit_cast(float, compute_cta::scalar);
```

This preserves exact floating-point values without any precision loss from string formatting.

Example from RMSNorm.emit():

```python
(f"{prefix}.epsilon", _float_to_bits(epsilon)),   # 1e-6 -> 0x358637BD
(f"{prefix}.scalar", _float_to_bits(scalar)),      # 1/sqrt(width) -> bits
```

### Boolean Coercion

Booleans are explicitly converted to 0 or 1:

```python
(f"{prefix}.fp32_dest_acc_en", 1 if fp32_dest_acc_en else 0),
(f"{prefix}.rsqrt_fast_approx", 1 if rsqrt_fast_approx else 0),
```

Or via `int(bool_value)`:

```python
(f"{prefix}.src_is_tensor_backed", int(not isinstance(src, CBHandle))),
```

## Semaphore Allocation

Semaphores are cross-core synchronization primitives. They are allocated via `f.semaphore()`:

**Signature** (from `blaze/fused_program.py`):

```python
def semaphore(
    self,
    name: str | None = None,
    *,
    program_semaphore: bool = False,
    initial_value: int = 0,
    core_ranges: ttnn.CoreRangeSet | None = None,
) -> int:
```

**Two allocation modes:**

1. **Named mesh-global semaphore** (`name` set, `program_semaphore=False`): Returns an L1 address. Deduped by name across FusedPrograms sharing a `_sem_dict`. Used for cross-device rendezvous in multi-device ops.

2. **Program semaphore** (`program_semaphore=True`): Per-device slot-based synchronization. Optional `core_ranges` restricts the semaphore to specific cores, allowing tt-metal to reuse slot IDs across disjoint cores.

**Naming convention:** Semaphore names follow the same prefix convention as CT args:

```python
sender_sem = f.semaphore(f"{prefix}.sender")
receiver_sem = f.semaphore(f"{prefix}.receiver")
```

**The returned value is an int** -- either an L1 address (for mesh-global semaphores) or a slot ID (for program semaphores). This integer is passed directly in CT arg tuples:

```python
f.brisc_ct_args([
    (f"{prefix}.sender_semaphore", sender_sem),
    (f"{prefix}.receiver_semaphore", receiver_sem),
])
```

**Duplicate detection:** Calling `f.semaphore()` with the same name twice within one op raises `RuntimeError`:

```python
if any(s is sem for s in self.program._global_semaphores):
    raise RuntimeError(
        f"f.semaphore({name!r}) called twice within one op -- "
        "give distinct sync points distinct names."
    )
```

**Example from Mcast.emit():**

```python
if sender_sem is None:
    sender_sem = f.semaphore(f"{prefix}.sender")
receiver_sem = f.semaphore(f"{prefix}.receiver")
```

## Runtime Args

Runtime args are values that can change between kernel invocations without recompilation. They are passed via per-RISC methods on FusedProgram.

### Per-RISC Runtime Arg Methods

**Signature examples** (from `blaze/fused_program.py`):

```python
def ncrisc_rt_args(self, args):
    """Add named common runtime args for NCRISC."""
    self.program.ncrisc_rt_args(args)

def brisc_rt_args(self, args):
    """Add named common runtime args for BRISC."""
    self.program.brisc_rt_args(args)

def trisc_rt_args(self, args):
    """Add named common runtime args for TRISC."""
    self.program.trisc_rt_args(args)
```

### Array and Per-Core Variants

```python
def ncrisc_rt_arg_arrays(self, args):
    """Add named common runtime arg arrays for NCRISC."""
    self.program.ncrisc_rt_arg_arrays(args)

def ncrisc_per_core_rt_arg_arrays(self, args):
    """Add per-core runtime arg arrays for NCRISC."""

def brisc_per_core_rt_arg_arrays(self, args):
    """Add per-core runtime arg arrays for BRISC."""

def trisc_per_core_rt_arg_arrays(self, args):
    """Add per-core runtime arg arrays for TRISC."""
```

There is also a lower-level per-core method for explicit core targeting:

```python
def _set_per_core_runtime_args(self, core, ncrisc=None, trisc=None, brisc=None):
    """Add per-core runtime args for a specific core."""
```

### Positional CT Args

For APIs like TensorAccessorArgs that use positional rather than named args:

```python
def ncrisc_positional_ct_args(self, args):
    """Add positional (unnamed) compile-time args for NCRISC."""

def brisc_positional_ct_args(self, args):
    """Add positional (unnamed) compile-time args for BRISC."""
```

### C++ Access Pattern: `rt_args::get<>()`

On the C++ side, runtime args are read via `rt_args::get<>()` using the CT arg descriptor as a template parameter. From the actual `rt_args_demo_kernel.cpp` at `blaze/ops/rt_args_demo/kernels/`:

```cpp
// Named runtime args -- rt_args::get<>() hides the dispatch type.
// The kernel doesn't need to know if it's common or per-core.
uint32_t num_tiles = rt_args::get<ct_args::my_kernel::num_tiles>();
uint32_t core_id = rt_args::get<ct_args::my_kernel::core_id>();
```

For array runtime args (from `array_rt_args_demo_kernel.cpp`):

```cpp
// Array named RT arg -- one mask value per tile
uint32_t copy_mask[max_tiles];
for (uint32_t i = 0; i < max_tiles; i++) {
    copy_mask[i] = rt_args::get<ct_args::my_kernel::copy_mask>(i);
}
```

The `rt_args::get<>()` template dispatches to either common or per-core runtime arg storage based on the descriptor metadata. The kernel code is agnostic to the dispatch type -- it uses the same `rt_args::get<>()` call regardless of whether the value was set via `ncrisc_rt_args()` (common) or `ncrisc_per_core_rt_arg_arrays()` (per-core).

**Reference:** The `blaze/ops/rt_args_demo/kernels/` directory contains two reference kernels:
- `rt_args_demo_kernel.cpp` -- Demonstrates scalar named runtime args (common and per-core)
- `array_rt_args_demo_kernel.cpp` -- Demonstrates array named runtime args

## CT vs. RT Args: The Recompilation Tradeoff

| Property | CT Args | RT Args |
|----------|---------|---------|
| When set | Compile time | Runtime (per invocation) |
| Cost to change | Recompilation | Zero (just update values) |
| Available in `if constexpr` | Yes | No |
| Dead-code elimination | Yes (zero-cost when flag=0) | No (branch evaluated at runtime) |
| C++ access | `ct_args::prefix::field` or `get_named_compile_time_arg_val()` | `rt_args::get<ct_args::prefix::field>()` |
| When to use | Op structure, CB IDs, flags, tile counts | Batch index, dynamic addresses, loop counters |

The design rule: **use CT args for anything structural**. CB IDs, tile counts, data formats, role flags, and semaphore addresses are all structural -- they define what the kernel does, not what data it processes. Changing any of these fundamentally changes the compiled kernel, so recompilation is unavoidable.

Use RT args for values that change between invocations of the same compiled kernel: batch indices, dynamic buffer addresses, or iteration counters. The kernel binary stays the same; only the arg values are updated.

In practice, most micro-ops use exclusively CT args. RT args appear primarily in higher-level fused ops that process variable-length sequences or dynamic batches, and in CCL (collective communication) and fabric ops where per-device or per-invocation addresses are involved.

## Putting It All Together: Complete CT Arg Flow for Mcast

The Mcast op provides a comprehensive example of a complex CT arg registration spanning all three RISCs. Here is the complete CT arg section from `Mcast.emit()`, annotated with the role of each arg:

```python
# Step 2a: Per-core role flags (unified = all RISCs)
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_sender", f.sender_grid),       # 1 on sender only
    f.flag(f"{prefix}.is_receiver", f.mcast_receiver_grid), # 1 on all non-sender
    f.flag(f"{prefix}.pop_src", f.all_cores),            # 1 everywhere
    f.flag(f"{prefix}.init_src", f.sender_grid, needs_init), # conditional
])

# Step 2b: NCRISC data-movement args
f.ncrisc_ct_args([
    (f"{prefix}.src", src_handle),                    # CB ID (via CBHandle.__int__)
    (f"{prefix}.src_num_pages", num_pages),            # plain int
    (f"{prefix}.src_is_tensor_backed", src_is_tensor_backed),
    (f"{prefix}.receiver_semaphore", receiver_sem),    # semaphore L1 address
    (f"{prefix}.dst", dst),                            # CB ID
    (f"{prefix}.dst_num_pages", dst.num_pages),
])

# Step 2c: BRISC multicast args
f.brisc_ct_args([
    (f"{prefix}.dest_noc_start_x", f.noc_start.x),    # NOC coordinates
    (f"{prefix}.dest_noc_start_y", f.noc_start.y),
    (f"{prefix}.dest_noc_end_x", f.noc_end.x),
    (f"{prefix}.dest_noc_end_y", f.noc_end.y),
    (f"{prefix}.num_cores", f.num_mcast_cores),
    (f"{prefix}.sender_semaphore", sender_sem),
    (f"{prefix}.receiver_semaphore", receiver_sem),
    (f"{prefix}.data_size_bytes", data_size_bytes),
    (f"{prefix}.src", src_handle),
    (f"{prefix}.src_num_pages", num_pages),
    (f"{prefix}.dst", dst),
    (f"{prefix}.is_part_of_receiver_grid",
        int(f.mcast_range.contains(f.sender))),
    (f"{prefix}.linked", 1),
    (f"{prefix}.posted", 1),
    (f"{prefix}.loopback", 0),
])
```

Notice the pattern:
- **NCRISC** gets data-movement args: source/destination CB IDs, page counts, semaphores for sync.
- **BRISC** gets multicast routing args: NOC coordinates for the destination range, semaphore addresses, data size, and multicast protocol flags (`linked`, `posted`, `loopback`).
- **Both share** some args (`src`, `dst`, semaphores) because both participate in the multicast protocol.
- **TRISC** gets no args from Mcast -- it is a pure data-movement op with no compute phase.

## Reference

- `blaze/fused_program.py` -- `flag()`, `per_core_unified_ct_args()`, per-RISC CT arg methods, `semaphore()`, runtime arg methods
- `blaze/ct_args.py` -- `CTArgSchema`, `CTArgSpec`, `CTArgEngine`
- `blaze/blaze_op.py` -- `CompileTimeArg`, `Risc` flag enum
- `blaze/utils.py` -- `_float_to_bits()`
- `blaze/ops/rmsnorm/kernels/op.hpp` -- `if constexpr` pattern with `is_active` flag
- `blaze/ops/rt_args_demo/kernels/` -- Runtime args demo kernels (`rt_args_demo_kernel.cpp`, `array_rt_args_demo_kernel.cpp`)
