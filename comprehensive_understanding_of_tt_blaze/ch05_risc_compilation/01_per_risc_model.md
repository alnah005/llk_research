# 01 -- Per-RISC Compilation Model

<- [Index](index.md) | [Kernel Headers and LLK Extensions](02_kernel_headers.md) ->

---

**Source files:** `blaze/unified_kernel_descriptor.py`, `blaze/program.py`, `blaze/blaze_op.py`, `blaze/kernels/kernel_op_api.hpp`

## Why Three Separate RISC Processors?

Every Tensix core on Blackhole silicon contains five RISC-V processors. TT-Blaze targets three of them: NCRISC (Network Controller RISC, `RISCV_1`), BRISC (Base RISC, `RISCV_0`), and TRISC (Tensor RISC, which further subdivides into three pipeline stages for unpack/math/pack). The hardware enforces this separation for a fundamental architectural reason: data movement and computation must overlap. If a single processor had to fetch tiles from DRAM, push them through the math pipeline, and write results back, the core would stall waiting for NOC transfers to complete. By splitting responsibilities across independent processors sharing L1 SRAM, TT-Blaze achieves full pipelining: NCRISC reads the next batch of tiles while TRISC computes on the current batch and BRISC writes finished results or multicasts data to other cores.

This is not a software convenience -- it reflects a hardware constraint. Each RISC has its own instruction memory and registers. The compiler must produce separate binaries for each, and each binary sees different hardware APIs.

| RISC   | Hardware Name | Role              | TT-Metal Processor Enum        | NOC Default            |
|--------|---------------|-------------------|---------------------------------|------------------------|
| NCRISC | RISCV_1       | Reader / data-in  | `DataMovementProcessor.RISCV_1` | `NOC.RISCV_0_default` |
| BRISC  | RISCV_0       | Writer / data-out | `DataMovementProcessor.RISCV_0` | `NOC.RISCV_1_default` |
| TRISC  | Compute pipeline (TRISC0/1/2) | Math / compute | `ComputeConfigDescriptor` | N/A |

NCRISC and BRISC include `dataflow_api.h` for NOC operations. TRISC includes the compute kernel API for matrix multiplies, reductions, and SFPU operations. These are mutually exclusive -- you cannot call `noc_async_write` from TRISC or `tile_regs_acquire` from BRISC.

---

## Per-RISC Compilation: One Source, Three Binaries

TT-Blaze writes a single `.cpp` file per fused kernel and compiles it three times -- once for each RISC. The `COMPILE_FOR_NCRISC`, `COMPILE_FOR_BRISC`, and `COMPILE_FOR_TRISC` preprocessor defines control which code paths survive each compilation pass. The Python-side `UnifiedKernelDescriptor` orchestrates this by creating three `KernelDescriptor` objects from one source path, each with its own compile-time args, runtime args, and processor configuration.

The design rationale for a single unified source (rather than three separate kernel files) is cohesion: an operation's reader, writer, and compute logic are tightly coupled. Splitting them across files creates maintenance burden (CB IDs must match across files, initialization order depends on all three agreeing) and obscures the dataflow contract.

`kernel_op_api.hpp` defines constexpr booleans from the `COMPILE_FOR_*` defines:

```cpp
// From blaze/kernels/kernel_op_api.hpp
#if defined(COMPILE_FOR_NCRISC)
inline constexpr bool is_ncrisc = true;
inline constexpr bool is_brisc = false;
inline constexpr bool is_trisc = false;
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#elif defined(COMPILE_FOR_BRISC)
inline constexpr bool is_ncrisc = false;
inline constexpr bool is_brisc = true;
inline constexpr bool is_trisc = false;
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#elif defined(COMPILE_FOR_TRISC)
inline constexpr bool is_ncrisc = false;
inline constexpr bool is_brisc = false;
inline constexpr bool is_trisc = true;
#include "tools/profiler/kernel_profiler.hpp"
#include "api/compute/blank.h"
#endif
```

And provides a type selector for RISC-specific type aliases without preprocessor guards:

```cpp
namespace unified_kernels {
template <typename Reader, typename Writer, typename Compute>
using SelectByRISCV = std::conditional_t<
    is_ncrisc, Reader,
    std::conditional_t<is_brisc, Writer, Compute>>;
}
```

---

## Worked Example: Mcast -> Matmul -> Gather

This section traces a concrete example -- a fused **mcast -> matmul -> gather** pipeline -- to show how every piece fits together. Consider a fused kernel where activations are multicast to all cores, each core performs a matrix multiply with its local weight shard, and results are gathered to a single receiver core.

```python
# Inside a FusedProgram composition function
act_handle = Mcast.emit(f, input_tensor, prefix="act_mcast")
mm_handle  = Matmul.emit(f, act_handle, weights_tensor, prefix="matmul")
Gather.emit(f, mm_handle, output_tensor, prefix="gather")
```

### Step 1: CT Args Land on Specific RISCs

Each `emit()` call registers named compile-time args for specific RISCs. Here is the per-RISC breakdown for our three ops:

**Mcast (prefix="act_mcast")**

| RISC   | CT Args                                                                   |
|--------|---------------------------------------------------------------------------|
| ALL    | `act_mcast.is_sender`, `act_mcast.is_receiver`, `act_mcast.pop_src`, `act_mcast.init_src` (per-core flags) |
| NCRISC | `act_mcast.src`, `act_mcast.src_num_pages`, `act_mcast.src_is_tensor_backed`, `act_mcast.receiver_semaphore`, `act_mcast.dst`, `act_mcast.dst_num_pages` |
| BRISC  | `act_mcast.dest_noc_start_x/y`, `act_mcast.dest_noc_end_x/y`, `act_mcast.num_cores`, `act_mcast.sender_semaphore`, `act_mcast.receiver_semaphore`, `act_mcast.data_size_bytes`, `act_mcast.src`, `act_mcast.dst` |

**Matmul (prefix="matmul")**

| RISC   | CT Args                                                               |
|--------|-----------------------------------------------------------------------|
| ALL    | `matmul.is_active`, `matmul.pop_in0` (per-core flags), `matmul.in0` (shared CB) |
| NCRISC | `matmul.k_num_tiles`, `matmul.in0_is_tensor_backed`                  |
| TRISC  | `matmul.in1`, `matmul.out`, `matmul.k_num_tiles`, `matmul.out_w_per_core` |

**Gather (prefix="gather")**

| RISC   | CT Args                                                               |
|--------|-----------------------------------------------------------------------|
| ALL    | `gather.is_sender`, `gather.is_receiver`, `gather.pop_src`, `gather.use_per_core`, `gather.sender_idx` (per-core flags/values) |
| NCRISC | `gather.dest_noc_x/y`, `gather.data_size_bytes`, `gather.noc0_receiver_semaphore`, `gather.src`, `gather.src_num_pages`, `gather.dst`, `gather.receiver_data_addr` |
| BRISC  | `gather.noc0_num_senders`, `gather.noc1_num_senders`, `gather.noc0_receiver_semaphore`, `gather.noc1_receiver_semaphore`, `gather.dst`, `gather.dst_num_pages` |

Notice the pattern: data-movement args (NOC coordinates, semaphore addresses, transfer sizes) go to NCRISC and BRISC. Compute args (tile counts, output widths, CB IDs for math) go to TRISC. Shared CBs that both data-movement and compute need are sent to ALL RISCs via `unified_ct_args`.

### Step 2: Per-Core Flags Create Kernel Splits

The `per_core_unified_ct_args` calls register `UnifiedCompileTimeCoreDescriptor` objects. For mcast:

```python
f.per_core_unified_ct_args([
    f.flag("act_mcast.is_sender", f.sender_grid),       # 1 on sender, 0 elsewhere
    f.flag("act_mcast.is_receiver", f.mcast_receiver_grid),  # 1 on receivers, 0 elsewhere
])
```

These per-core descriptors cause the `UnifiedKernelDescriptor._get_split_kernel_descriptors()` path to activate. The algorithm:

1. **Enumerate all cores** in the grid.
2. **Compute each core's complete set of named CT args** -- for every descriptor, check if the core is in its `core_range` (gets `value`) or not (gets `other_value`).
3. **Group cores by unique arg combinations** -- cores with identical frozen arg tuples share a kernel.
4. **Create one kernel set (NCRISC/BRISC/TRISC) per unique group.**

For our example with a 4x8 grid where core (0,0) is the sender:

- **Group A** (sender core): `is_sender=1, is_receiver=0, is_active=1, ...`
- **Group B** (receiver cores): `is_sender=0, is_receiver=1, is_active=1, ...`

This produces 6 kernel descriptors (2 groups x 3 RISCs) instead of 3.

### Step 3: UnifiedKernelDescriptor Assembly

The `BlazeProgram.build()` method assembles the `UnifiedKernelDescriptor`:

```python
ukd = UnifiedKernelDescriptor(
    kernel_source=self._kernel,           # path to generated .cpp
    core_ranges=self._core_ranges,
    ncrisc_named_compile_time_args=self._ncrisc_args,  # [("act_mcast.src", 0), ...]
    brisc_named_compile_time_args=self._brisc_args,
    trisc_named_compile_time_args=self._trisc_args,
    unified_compile_time_core_descriptors=self._core_descriptors,  # per-core flags
    per_core_compile_time_descriptors=self._per_core_descriptors,  # per-core values
    trisc_compute_config=self._compute_config,
    noc_mode=self._noc_mode,
    ...
)
```

`get_kernel_descriptors()` returns a `UnifiedKernelResult` containing:
- `kernels`: List of `KernelDescriptor` objects to pass to `ProgramDescriptor`
- `groups`: List of `KernelGroup` objects that record which kernel indices belong to which role

Each `KernelGroup` records the indices of its three RISC kernels:

```python
KernelGroup(
    core_range_set=sender_cores,
    compile_time_arg_values={"act_mcast.is_sender": 1, ...},
    ncrisc_kernel_index=0,
    brisc_kernel_index=1,
    trisc_kernel_index=2,
)
```

This allows callers to look up kernels by role:

```python
sender_group = result.get_group_by_arg("act_mcast.is_sender", 1)
sender_ncrisc_idx = sender_group.ncrisc_kernel_index
```

---

## Named CT Args: Why Not Positional?

TT-Blaze overwhelmingly uses named compile-time arguments rather than positional indices. The design rationale has three parts.

First, **positional args are fragile**. If you add a new compile-time argument between two existing ones on the Python side, every `get_compile_time_arg_val(index)` call on the C++ side silently reads the wrong value. Named args are order-independent -- the generated header binds names to values, so insertion order does not matter.

Second, **named args enable validation**. `BlazeProgram.validate()` parses the C++ header file (via `ct_parse`, see Ch2 S03) and cross-checks against the Python-supplied names. Missing args and extra args produce diagnostics before the kernel ever reaches the device:

```python
# From program.py validate()
missing = expected_names - all_supplied
extra = all_supplied - expected_names
if missing:
    errors.append(f"  {risc}: missing args {sorted(missing)}")
```

Third, **named args document intent**. A positional arg `get_compile_time_arg_val(7)` tells you nothing. A named arg `ct_args::matmul::k_num_tiles` tells you exactly what the value means. With 55 ops each having 10-30 compile-time arguments, this readability is essential.

Positional args still exist for specific cases where tt-metal APIs require them (e.g., `TensorAccessorArgs` that read sequential indices via `get_compile_time_arg_val(index)`), accessed through `ncrisc_positional_ct_args()` and `brisc_positional_ct_args()` on `BlazeProgram`. There is no TRISC equivalent on the builder API.

### The Prefix Convention

Named compile-time args use a `prefix.field` dotted naming convention. On the Python side:

```python
f.ncrisc_ct_args([
    ("act_mcast.src", src_handle),
    ("act_mcast.src_num_pages", num_pages),
])
```

The JIT build system converts these to a C++ header with `ct_args::` namespace structs (see Section 03). The prefix-to-namespace mapping replaces dots with double underscores: `"shared_expert.gu"` becomes `ct_args::shared_expert__gu`. The kernel reads them via template parameters:

```cpp
using ActMcast = blaze::Mcast::Op<ct_args::act_mcast>;
ActMcast act_mcast;
act_mcast.init();
act_mcast();
act_mcast.teardown();
```

---

## Per-Core Descriptors: Two Mechanisms

TT-Blaze provides two ways to vary CT args across cores.

### UnifiedCompileTimeCoreDescriptor (Range-Based)

For role flags where cores fall into distinct rectangular regions:

```python
UnifiedCompileTimeCoreDescriptor(
    named_compile_time_arg="act_mcast.is_sender",
    core_range=sender_core,     # CoreCoord or CoreRange or CoreRangeSet
    value=1,                    # value for cores in core_range
    other_value=0,              # value for all other cores
    riscs=ALL_RISCS,            # applies to all three RISCs
)
```

### PerCoreCompileTimeDescriptor (Individual Core Values)

For values that are unique to each core and cannot be expressed as rectangular ranges:

```python
PerCoreCompileTimeDescriptor(
    named_compile_time_arg="gather.sender_idx",
    core_values=[(core0, 0), (core1, 1), (core2, 2), ...],
    other_value=0,
    riscs=ALL_RISCS,
)
```

Because each unique value produces a separate compiled kernel, this should be used sparingly -- many unique values means many compilations.

### RISC Scoping

Both descriptor types accept a `riscs` parameter that controls which RISCs receive the per-core arg:

```python
# Only NCRISC gets the per-core bank_id
PerCoreCompileTimeDescriptor(
    named_compile_time_arg="bank_id",
    core_values=[(core, bank) for core, bank in bank_assignments],
    other_value=0,
    riscs=(Risc.NCRISC,),  # BRISC and TRISC do not see this arg
)
```

When `riscs` is scoped to a single processor, the arg only appears in that RISC's `named_compile_time_args` list. This prevents unnecessary kernel splits for args that only one RISC needs.

---

## UnifiedKernelDescriptor -- Complete Field Reference

**Source file:** `blaze/unified_kernel_descriptor.py`

`UnifiedKernelDescriptor` is a `@dataclass` with over 25 fields. All list fields use `field(default_factory=list)` for safe mutable defaults.

### Core Identity Fields

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `kernel_source` | `str` | (required) | Path to the unified `.cpp` kernel file |
| `core_ranges` | `ttnn.CoreRangeSet` | (required) | Set of cores where this kernel runs |
| `noc_mode` | `ttnn.NOC_MODE` | `DM_DEDICATED_NOC` | NOC arbitration mode for data-movement RISCs |

Note: `BlazeProgram.__init__` defaults `noc_mode` to `DM_DYNAMIC_NOC`, which overrides the UKD default when `build()` constructs the descriptor. In `DM_DYNAMIC_NOC` mode, both data-movement RISCs can share NOC resources dynamically; in `DM_DEDICATED_NOC` mode, each RISC owns its NOC exclusively.

### Per-RISC Compile-Time Args

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `ncrisc_compile_time_args` | `list` | `field(default_factory=list)` | Positional (unnamed) CT args for NCRISC |
| `brisc_compile_time_args` | `list` | `field(default_factory=list)` | Positional CT args for BRISC |
| `trisc_compile_time_args` | `list` | `field(default_factory=list)` | Positional CT args for TRISC |
| `ncrisc_named_compile_time_args` | `list` | `field(default_factory=list)` | Named CT args for NCRISC: `[(name, value), ...]` |
| `brisc_named_compile_time_args` | `list` | `field(default_factory=list)` | Named CT args for BRISC |
| `trisc_named_compile_time_args` | `list` | `field(default_factory=list)` | Named CT args for TRISC |

### Per-RISC Runtime Args (Named)

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `ncrisc_named_common_runtime_args` | `list` | `field(default_factory=list)` | Common RT args for NCRISC: `[(name, value), ...]` |
| `brisc_named_common_runtime_args` | `list` | `field(default_factory=list)` | Common RT args for BRISC |
| `trisc_named_common_runtime_args` | `list` | `field(default_factory=list)` | Common RT args for TRISC |
| `ncrisc_named_per_core_runtime_args` | `list` | `field(default_factory=list)` | Per-core RT args for NCRISC: `[(name, {CoreCoord: value}), ...]` |
| `brisc_named_per_core_runtime_args` | `list` | `field(default_factory=list)` | Per-core RT args for BRISC |
| `trisc_named_per_core_runtime_args` | `list` | `field(default_factory=list)` | Per-core RT args for TRISC |

### Per-RISC Runtime Arg Arrays (Named)

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `ncrisc_named_common_runtime_arg_arrays` | `list` | `field(default_factory=list)` | `[(name, [val0, val1, ...]), ...]` for NCRISC |
| `brisc_named_common_runtime_arg_arrays` | `list` | `field(default_factory=list)` | Same for BRISC |
| `trisc_named_common_runtime_arg_arrays` | `list` | `field(default_factory=list)` | Same for TRISC |
| `ncrisc_named_per_core_runtime_arg_arrays` | `list` | `field(default_factory=list)` | `[(name, {CoreCoord: [val0, ...]}), ...]` for NCRISC |
| `brisc_named_per_core_runtime_arg_arrays` | `list` | `field(default_factory=list)` | Same for BRISC |
| `trisc_named_per_core_runtime_arg_arrays` | `list` | `field(default_factory=list)` | Same for TRISC |

### Legacy Positional Runtime Args

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `ncrisc_common_runtime_args` | `list` | `field(default_factory=list)` | Positional common RT args for NCRISC |
| `brisc_common_runtime_args` | `list` | `field(default_factory=list)` | Same for BRISC |
| `trisc_common_runtime_args` | `list` | `field(default_factory=list)` | Same for TRISC |

### Per-Core Descriptors, Compute Config, and Defines

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `trisc_compute_config` | `Optional[ttnn.ComputeConfigDescriptor]` | `None` | Math fidelity, fp32 acc, approx mode for TRISC |
| `unified_compile_time_core_descriptors` | `list` | `field(default_factory=list)` | `UnifiedCompileTimeCoreDescriptor` list |
| `per_core_compile_time_descriptors` | `list` | `field(default_factory=list)` | `PerCoreCompileTimeDescriptor` list |
| `per_core_positional_cta_descriptor` | `Optional[PerCorePositionalCTADescriptor]` | `None` | Per-core, per-RISC positional CT arg arrays |
| `per_core_runtime_args_descriptor` | `Optional[PerCoreRuntimeArgsDescriptor]` | `None` | Legacy per-core RT args |
| `defines` | `list` | `field(default_factory=list)` | Preprocessor defines: `[(name, value), ...]` |

---

## The RISC Enum

**Source file:** `blaze/blaze_op.py`

```python
class Risc(Flag):
    NCRISC = auto()
    BRISC = auto()
    TRISC = auto()

ALL_RISCS: tuple[Risc, ...] = (Risc.NCRISC, Risc.TRISC, Risc.BRISC)
RISC_NAMES = {Risc.NCRISC: "ncrisc", Risc.TRISC: "trisc", Risc.BRISC: "brisc"}
```

`Risc` is a `Flag` enum, supporting bitwise combination. `ALL_RISCS` is the tuple `(NCRISC, TRISC, BRISC)` -- notably TRISC appears second, not third. `RISC_NAMES` maps each flag to the lowercase string used in validation and descriptor naming.

---

## BlazeProgram: The Builder Interface

**Source file:** `blaze/program.py`

`BlazeProgram` wraps `UnifiedKernelDescriptor` construction with a builder pattern, accumulating per-RISC arguments through explicit method calls and assembling everything in `build()`. The rationale for the builder is that `UnifiedKernelDescriptor` has over 25 fields, most with empty-list defaults. Constructing it directly would be error-prone.

### Constructor Parameters

```python
BlazeProgram(
    kernel: str | None = None,       # path to kernel .cpp
    core_ranges: CoreRangeSet | None = None,
    graph=None,                      # BlazeGraph for codegen path
    ct_prefix: str = "",             # prefix applied to named CT args
    ct_header: str | None = None,    # path for validation
    math_fidelity: MathFidelity = HiFi4,
    math_approx_mode: bool = False,
    fp32_dest_acc_en: bool = False,
    dst_full_sync_en: bool = False,
    noc_mode=NOC_MODE.DM_DYNAMIC_NOC,
)
```

### Named Compile-Time Arg Methods

| Method | Target RISCs |
|--------|--------------|
| `unified_ct_args(args, prefix=True)` | ALL (appends to NCRISC + BRISC + TRISC) |
| `ncrisc_ct_args(args, prefix=True)` | NCRISC |
| `brisc_ct_args(args, prefix=True)` | BRISC |
| `trisc_ct_args(args, prefix=True)` | TRISC |

### Positional CT Arg Methods

| Method | Target RISC | Returns |
|--------|-------------|---------|
| `ncrisc_positional_ct_args(args)` | NCRISC | Start index of appended args |
| `brisc_positional_ct_args(args)` | BRISC | Start index of appended args |

### Named Runtime Arg Methods (Common)

| Method | Target RISCs |
|--------|--------------|
| `unified_rt_args(args)` | ALL |
| `ncrisc_rt_args(args)` | NCRISC |
| `brisc_rt_args(args)` | BRISC |
| `trisc_rt_args(args)` | TRISC |

### Named Runtime Arg Methods (Per-Core)

Each entry is `(name, {CoreCoord: value})`. Cores not in the mapping get `other_value`.

| Method | Target RISCs |
|--------|--------------|
| `unified_per_core_rt_args(args, other_value=0)` | ALL |
| `ncrisc_per_core_rt_args(args, other_value=0)` | NCRISC |
| `brisc_per_core_rt_args(args, other_value=0)` | BRISC |
| `trisc_per_core_rt_args(args, other_value=0)` | TRISC |

### Named Runtime Arg Array Methods

Each entry is `(name, [val0, val1, ...])` -- occupies N contiguous RT arg slots. Per-core variants use `(name, {CoreCoord: [val0, ...]})`.

| Method | Target RISCs |
|--------|--------------|
| `unified_rt_arg_arrays(args)` | ALL |
| `ncrisc_rt_arg_arrays(args)` / `brisc_rt_arg_arrays` / `trisc_rt_arg_arrays` | Single RISC |
| `unified_per_core_rt_arg_arrays(args)` | ALL |
| `ncrisc_per_core_rt_arg_arrays(args)` / `brisc_per_core_rt_arg_arrays` / `trisc_per_core_rt_arg_arrays` | Single RISC |

### Per-Core CT Arg Methods

These create `UnifiedCompileTimeCoreDescriptor` or `PerCoreCompileTimeDescriptor` depending on whether the mapping keys are `CoreRangeSet`/`CoreRange` or `CoreCoord`.

| Method | Target RISCs |
|--------|--------------|
| `per_core_unified_ct_args(args, other_value=0)` | ALL |
| `per_core_ncrisc_ct_args(args, other_value=0)` | NCRISC |
| `per_core_brisc_ct_args(args, other_value=0)` | BRISC |
| `per_core_trisc_ct_args(args, other_value=0)` | TRISC |

### BlazeProgram-to-UKD Field Mapping

| BlazeProgram field | UKD field |
|---|---|
| `_kernel` | `kernel_source` |
| `_core_ranges` | `core_ranges` |
| `_ncrisc_positional_ct_args` | `ncrisc_compile_time_args` |
| `_brisc_positional_ct_args` | `brisc_compile_time_args` |
| `_ncrisc_args` | `ncrisc_named_compile_time_args` |
| `_brisc_args` | `brisc_named_compile_time_args` |
| `_trisc_args` | `trisc_named_compile_time_args` |
| `_ncrisc_named_rt_args` | `ncrisc_named_common_runtime_args` |
| `_core_descriptors` | `unified_compile_time_core_descriptors` |
| `_per_core_descriptors` | `per_core_compile_time_descriptors` |
| `_compute_config` | `trisc_compute_config` |
| `_defines` | `defines` |
| `_noc_mode` | `noc_mode` |

---

## Tracing Through: Descriptor to Hardware

Here is the full path from Python to silicon for our mcast -> matmul -> gather example:

```
Python emit() calls
    |
    v
BlazeProgram accumulates:
  _ncrisc_args:  [("act_mcast.src", 3), ("matmul.k_num_tiles", 4), ("gather.dest_noc_x", 1), ...]
  _brisc_args:   [("act_mcast.dest_noc_start_x", 18), ("gather.noc0_num_senders", 8), ...]
  _trisc_args:   [("matmul.in1", 2), ("matmul.out", 5), ("matmul.k_num_tiles", 4), ...]
  _core_descriptors: [is_sender=1 on sender, is_receiver=1 on receivers, ...]
    |
    v
BlazeProgram.build() creates UnifiedKernelDescriptor
    |
    v
UnifiedKernelDescriptor.get_kernel_descriptors() -> UnifiedKernelResult
  Splits into groups by per-core arg combinations:
    Group A (sender): ncrisc_idx=0, brisc_idx=1, trisc_idx=2
    Group B (receivers): ncrisc_idx=3, brisc_idx=4, trisc_idx=5
    |
    v
ProgramDescriptor(
    kernels=[desc0, desc1, desc2, desc3, desc4, desc5],
    cbs=[...],
    semaphores=[...],
)
    |
    v
ttnn.generic_op(tensors, program_descriptor)
  -> tt-metal JIT-compiles each kernel descriptor for its target RISC
  -> Downloads firmware to each core
  -> Launches program
```

The key insight is that a single unified `.cpp` source file, combined with per-RISC defines and per-core CT arg specialization, produces distinct firmware images for different (RISC, core-group) combinations. A 4x8 grid running this fused kernel might compile 6 distinct binaries from one source file.

---

<- [Index](index.md) | [Next: Kernel Headers and LLK Extensions](02_kernel_headers.md) ->
