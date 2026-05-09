# 2.1 The Unified Kernel Descriptor

The `UnifiedKernelDescriptor` is the Python-side engine that transforms a single kernel source file into the multiple `KernelDescriptor` objects required by `ttnn.ProgramDescriptor`. Rather than forcing each micro-op author to manually construct three kernel descriptors (one per RISC processor) and reason about their compile-time argument overlap, the unified descriptor accepts a single `kernel_source` path, per-RISC compile-time and runtime arguments, and optional per-core overrides, then generates all necessary descriptors through one call to `get_kernel_descriptors()`. This section traces the dataclass fields, the two generation paths (simple and split), provides a taxonomy of how different operations use each mechanism, and concludes with an end-to-end Gather trace from Python to hardware NOC write.

**Source file:** `models/demos/deepseek_v3_b1/unified_kernel_descriptor.py` (461 lines)

---

## 2.1.1 RISC Processor Roles and NOC Assignment

Before examining the descriptor itself, it is essential to understand the three RISC processors it targets, because the descriptor hard-codes their NOC assignments:

```
+------------------------------------------------------------------+
|                        Tensix Core                               |
|                                                                  |
|  NCRISC (RISCV_1)          BRISC (RISCV_0)         TRISC        |
|  +-----------------+       +-----------------+     +----------+  |
|  | Reader / DM     |       | Writer / DM     |     | Compute  |  |
|  | NOC_0 (default) |       | NOC_1 (default) |     | FPU/SFPU |  |
|  +-----------------+       +-----------------+     +----------+  |
|         |                         |                     |        |
|         +---------> L1 SRAM <-----+------> CBs <--------+        |
|                                                                  |
+------------------------------------------------------------------+
```

| RISC | Python Processor Enum | Default NOC | Primary Role |
|------|-----------------------|-------------|-------------|
| NCRISC | `DataMovementProcessor.RISCV_1` | `NOC.RISCV_0_default` (NOC_0) | **Reader** -- fetches data from DRAM or remote L1 into local circular buffers |
| BRISC | `DataMovementProcessor.RISCV_0` | `NOC.RISCV_1_default` (NOC_1) | **Writer** -- sends data from local circular buffers to other cores or DRAM |
| TRISC | N/A (uses `ComputeConfigDescriptor`) | N/A | **Compute** -- executes FPU/SFPU math on tiles in circular buffers |

In the `_get_simple_kernel_descriptors()` method (lines 250-314 of `unified_kernel_descriptor.py`), the mapping is explicit:

```python
# Reader kernel (NCRISC)
reader_descriptor = ttnn.KernelDescriptor(
    ...
    config=ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.RISCV_0_default,
        noc_mode=self.noc_mode,
    ),
)

# Writer kernel (BRISC)
writer_descriptor = ttnn.KernelDescriptor(
    ...
    config=ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0,
        noc=ttnn.NOC.RISCV_1_default,
        noc_mode=self.noc_mode,
    ),
)
```

The naming can be confusing: NCRISC is `RISCV_1` on the hardware but is the "reader" in the DeepSeek B1 convention; BRISC is `RISCV_0` but is the "writer." The NOC assignments (`RISCV_0_default` for NCRISC, `RISCV_1_default` for BRISC) are the default dedicated NOC channels. By giving each data-movement RISC a dedicated NOC, reader and writer can operate concurrently without contention in `DM_DEDICATED_NOC` mode. The `noc_mode` field (default `DM_DEDICATED_NOC`) on `UnifiedKernelDescriptor` allows switching to `DM_DYNAMIC_NOC` when both RISCs need access to both NOCs (used by Flash MLA).

---

## 2.1.2 The UnifiedKernelDescriptor Dataclass

The full dataclass (lines 127-172) has 17 fields. Here is the conceptual grouping:

**Core identity:**
- `kernel_source: str` -- path to the unified `.cpp` file
- `core_ranges: ttnn.CoreRangeSet` -- the full set of cores this kernel runs on
- `noc_mode: ttnn.NOC_MODE` -- `DM_DEDICATED_NOC` (default) or `DM_DYNAMIC_NOC`

**Per-RISC compile-time arguments:**
- `ncrisc_compile_time_args`, `brisc_compile_time_args`, `trisc_compile_time_args` -- positional (unnamed) args
- `ncrisc_named_compile_time_args`, `brisc_named_compile_time_args`, `trisc_named_compile_time_args` -- named args as `(name, value)` tuples

**Per-RISC runtime arguments:**
- `ncrisc_common_runtime_args`, `brisc_common_runtime_args`, `trisc_common_runtime_args` -- shared across all cores
- `per_core_runtime_args_descriptor: Optional[PerCoreRuntimeArgsDescriptor]` -- per-core, per-RISC runtime args

**Core specialization:**
- `unified_compile_time_core_descriptors: list` -- `UnifiedCompileTimeCoreDescriptor` objects for range-based compile-time overrides
- `per_core_compile_time_descriptors: list` -- `PerCoreCompileTimeDescriptor` objects for individual-core compile-time overrides

**Compute configuration:**
- `trisc_compute_config: Optional[ttnn.ComputeConfigDescriptor]` -- math fidelity, FP32 accumulation, etc.

**Preprocessor definitions:**
- `defines: list` -- `(name, value)` tuples passed to the compiler as `#define name value`

The key design insight is that `ncrisc_named_compile_time_args`, `brisc_named_compile_time_args`, and `trisc_named_compile_time_args` are *per-RISC* specializations, while `unified_compile_time_core_descriptors` and `per_core_compile_time_descriptors` are *shared across all three RISCs* -- they specialize by core location rather than by processor. The split path merges both sources.

---

## 2.1.3 The Simple Path: `_get_simple_kernel_descriptors()`

When neither `unified_compile_time_core_descriptors` nor `per_core_compile_time_descriptors` is populated, the `get_kernel_descriptors()` method (line 243) takes the simple path (lines 250-314). This is the common case for standalone micro-ops where all cores execute identical code with the same compile-time args.

```
get_kernel_descriptors()
  +--- _get_simple_kernel_descriptors()
         +---> KernelDescriptor[0]: NCRISC (RISCV_1, NOC_0)
         +---> KernelDescriptor[1]: BRISC  (RISCV_0, NOC_1)
         +---> KernelDescriptor[2]: TRISC  (ComputeConfig)
         +---> UnifiedKernelResult(kernels=[3 descs], groups=[1 group])
```

Each descriptor receives the same `kernel_source` and `core_ranges`, but RISC-specific `compile_time_args`, `named_compile_time_args`, `common_runtime_args`, and a RISC-specific config (`DataMovementConfigDescriptor` for NCRISC/BRISC, `ComputeConfigDescriptor` for TRISC). The result contains a single `KernelGroup` with `compile_time_arg_values={}` and indices `{ncrisc: 0, brisc: 1, trisc: 2}`.

**Example -- RMSNorm** (`micro_ops/rmsnorm/op.py`): RMSNorm is the canonical simple-path user. Every core does the same thing -- there is no sender/receiver distinction:

```python
UnifiedKernelDescriptor(
    kernel_source=KERNEL_PATH,
    core_ranges=compute_cores,
    ncrisc_named_compile_time_args=[("rmsnorm_input_cb", cb_id_in), ...],
    trisc_named_compile_time_args=[("rmsnorm_num_tiles", num_tiles), ...],
    trisc_compute_config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True),
)
```

Three descriptors, one group. No split.

---

## 2.1.4 The Split Path: `_get_split_kernel_descriptors()`

When any `UnifiedCompileTimeCoreDescriptor` or `PerCoreCompileTimeDescriptor` is specified, the split path activates (lines 316-461). This is fundamental to DeepSeek V3 B1's fused operations, where a single kernel binary serves cores playing different roles. The algorithm proceeds in four steps:

### Step 1: Enumerate all cores (line 330)

```python
all_cores = ttnn.corerange_to_cores(self.core_ranges)
```

### Step 2: Compute per-core compile-time arg maps (lines 346-365)

For each core, the method evaluates every `UnifiedCompileTimeCoreDescriptor` and `PerCoreCompileTimeDescriptor` to build a dict of `{arg_name: value}`. Range-based descriptors are pre-converted to `CoreRangeSet` objects (line 334), and per-core descriptors are pre-converted to `{(x,y): value}` lookup dicts (lines 339-343):

```python
for core_coord in all_cores:
    args = {}
    for desc, desc_core_range in zip(...):
        if desc_core_range.contains(core_coord):
            args[desc.named_compile_time_arg] = desc.value
        else:
            args[desc.named_compile_time_arg] = desc.other_value
    # ... also process per_core_compile_time_descriptors ...
    core_to_args[(core_coord.x, core_coord.y)] = tuple(sorted(args.items()))
```

### Step 3: Group cores by unique arg combinations (lines 368-371)

Cores with identical frozen arg tuples are grouped together:

```python
args_to_cores = {}
for core_tuple, frozen_args in core_to_args.items():
    args_to_cores.setdefault(frozen_args, []).append(core_tuple)
```

### Step 4: Emit 3 kernel descriptors per group (lines 376-461)

For each unique arg combination, NCRISC, BRISC, and TRISC `KernelDescriptor` objects are created. The per-RISC named args are *merged* with the unified args:

```python
ncrisc_named_compile_time_args_merged = list(self.ncrisc_named_compile_time_args) + unified_named_args
```

Each group's cores are converted to a `CoreRangeSet` of single-core ranges (line 388). Each group becomes a `KernelGroup` with the arg values dict and the three kernel indices.

**Output size:** If $G$ unique arg combinations exist, the result contains $3G$ kernel descriptors and $G$ kernel groups.

---

## 2.1.5 UnifiedCompileTimeCoreDescriptor: Range-Based Role Splitting

The `UnifiedCompileTimeCoreDescriptor` dataclass (lines 97-123) encodes a binary split: cores in `core_range` get `value`, all others get `other_value`. It is the primary mechanism for differentiating *roles* within a single kernel:

```python
@dataclass
class UnifiedCompileTimeCoreDescriptor:
    named_compile_time_arg: str              # e.g., "is_sender_core"
    core_range: Union[CoreCoord, CoreRange, CoreRangeSet]
    value: int                               # e.g., 1
    other_value: int                         # e.g., 0
```

### Taxonomy of Usage Across Operations

| Operation | Arg Name | `value` Cores | Purpose |
|-----------|----------|---------------|---------|
| **Mcast** | `is_sender_core` | Single sender core | Compile-time branch: sender executes BRISC mcast logic |
| **Mcast** | `is_receiver_core` | Output grid cores | Compile-time branch: receivers execute NCRISC wait+push |
| **DRAMStreamingMatmul** | `dram_mm_is_active_core` | All compute cores | Currently always 1; exists for future inactive-core support |
| **Fused MoE** | `is_sender_core` | Mcast sender | Differentiates input mcast sender from receivers |
| **Fused MoE** | `is_receiver_core` | Mcast receivers | Enables receiver-side CB reservation and semaphore waiting |
| **Fused MoE** | `is_gather_receiver_core` | Gather destination | Single core that aggregates gate/down_proj results |

### Mcast Example

From `micro_ops/mcast/op.py` (lines 181-194):

```python
unified_compile_time_core_descriptors=[
    UnifiedCompileTimeCoreDescriptor(
        named_compile_time_arg="is_sender_core",
        core_range=mcast_core,       # single CoreCoord
        value=1,
        other_value=0,
    ),
    UnifiedCompileTimeCoreDescriptor(
        named_compile_time_arg="is_receiver_core",
        core_range=output_core_grid,  # CoreRangeSet of receivers
        value=1,
        other_value=0,
    ),
]
```

With two descriptors, this creates up to $2^2 = 4$ unique arg combinations, though in practice the number of distinct groups depends on whether the sender overlaps the receiver grid. A typical Mcast produces 3 groups:

1. `{is_sender_core: 1, is_receiver_core: 1}` -- sender that is also in the receiver grid (loopback)
2. `{is_sender_core: 0, is_receiver_core: 1}` -- pure receiver cores
3. `{is_sender_core: 0, is_receiver_core: 0}` -- cores outside both ranges

Each group gets 3 kernel descriptors, so a Mcast with loopback produces **9 kernel descriptors** from a single `UnifiedKernelDescriptor`.

---

## 2.1.6 PerCoreCompileTimeDescriptor: Individual Core Values

While `UnifiedCompileTimeCoreDescriptor` assigns a binary value based on range membership, `PerCoreCompileTimeDescriptor` (lines 56-75) assigns *unique values to individual cores*:

```python
@dataclass
class PerCoreCompileTimeDescriptor:
    named_compile_time_arg: str       # e.g., "bank_id"
    core_values: list                 # [(CoreCoord, value), ...]
    other_value: int                  # Default for unlisted cores
```

This is critical when each core needs a hardware-specific constant that cannot be derived from grid position.

### DRAMStreamingMatmul: Combined Range + Per-Core Descriptors

DRAMStreamingMatmul (`micro_ops/dram_streaming_matmul/op.py`) uses *both* descriptor types simultaneously:

```python
unified_kernel = UnifiedKernelDescriptor(
    ...
    unified_compile_time_core_descriptors=[
        UnifiedCompileTimeCoreDescriptor(
            named_compile_time_arg="dram_mm_is_active_core",
            core_range=compute_cores,
            value=1,
            other_value=0,
        ),
    ],
    per_core_compile_time_descriptors=[
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="dram_mm_bank_id",
            core_values=bank_id_core_values,  # [(CoreCoord, bank_id), ...]
            other_value=0,
        ),
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="dram_mm_vc",
            core_values=vc_core_values,        # [(CoreCoord, vc), ...]
            other_value=0,
        ),
    ],
)
```

Every core gets a *unique combination* of `(dram_mm_is_active_core, dram_mm_bank_id, dram_mm_vc)`. If there are $N$ compute cores, the split path produces up to $N$ groups, yielding $3N$ kernel descriptors. This allows each NCRISC instance to have a compile-time `bank_id` and `vc` baked in as `constexpr` values.

### Gather: Scattered Core Indexing

`GatherSingleCore.op_scattered()` (`micro_ops/gather/op.py`) uses `PerCoreCompileTimeDescriptor` for a different purpose: assigning unique sender indices to non-rectangular core sets. When cores do not form a contiguous grid, the `linear_id_in_grid()` trick breaks down:

```python
per_core_desc = PerCoreCompileTimeDescriptor(
    named_compile_time_arg="gather_sender_idx",
    core_values=[(core, core_to_idx[(core.x, core.y)]) for core in cores],
    other_value=0,
)
```

Each sender core gets a unique compile-time `sender_idx`, enabling the C++ side to compute the correct offset into the receiver's output buffer.

---

## 2.1.7 PerCoreRuntimeArgsDescriptor

While compile-time args drive binary specialization, runtime args provide per-core values that vary at dispatch time. The `PerCoreRuntimeArgsDescriptor` (lines 78-93) supports per-RISC, per-core runtime arguments:

```python
@dataclass
class PerCoreRuntimeArgsDescriptor:
    ncrisc_args: list = field(default_factory=list)  # [(CoreCoord, args_list), ...]
    brisc_args: list = field(default_factory=list)
    trisc_args: list = field(default_factory=list)
```

This is used by **Flash MLA** (`micro_ops/flash_mla/op.py`), which has complex per-core roles determined at runtime by batch size, sequence length, and tree reduction topology. Each core receives different NOC coordinates, semaphore addresses, and chunk assignments for all three RISCs.

The `_build_runtime_args()` method (lines 184-226) converts this into a `ttnn.RuntimeArgs` object indexed by `(x, y)` coordinate:

```python
for core_coord in all_cores:
    core_key = (core_coord.x, core_coord.y)
    args = core_to_args.get(core_key, [])
    runtime_args[core_coord.x][core_coord.y] = list(args)
```

---

## 2.1.8 KernelGroup and UnifiedKernelResult

The `get_kernel_descriptors()` method returns a `UnifiedKernelResult` (lines 35-52) containing:

- `kernels: list` -- the flat list of `KernelDescriptor` objects to pass directly to `ProgramDescriptor`
- `groups: list[KernelGroup]` -- metadata for callers that need to identify kernels by role

Each `KernelGroup` (lines 19-31) records:

```python
@dataclass
class KernelGroup:
    core_range_set: ttnn.CoreRangeSet
    compile_time_arg_values: dict   # e.g., {"is_sender": 1}
    ncrisc_kernel_index: int
    brisc_kernel_index: int
    trisc_kernel_index: int
```

The `get_group_by_arg()` helper (lines 47-52) enables callers to find the kernel group for a specific role:

```python
result = unified_kernel.get_kernel_descriptors()
sender_group = result.get_group_by_arg("is_sender_core", 1)
sender_ncrisc_idx = sender_group.ncrisc_kernel_index
```

This is used in the fused MoE pipeline where subsequent operations need to set runtime args on specific kernel indices (e.g., setting the gather receiver's semaphore address on the sender kernel's runtime args).

---

## 2.1.9 Decision Tree: Which Path Does Your Op Need?

```
Does every core run identical code?
  +--- YES --> Simple path (no descriptors)
  |            Example: RMSNorm, Matmul (single core range)
  |            Result: 3 kernels, 1 group
  |
  +--- NO ---> Split path needed
               |
               +--- Do cores differ by ROLE (sender/receiver/compute)?
               |    +--- Use UnifiedCompileTimeCoreDescriptor
               |         Example: Mcast (is_sender_core, is_receiver_core)
               |         Result: 3G kernels where G = number of unique role combos
               |
               +--- Does each core need a UNIQUE constant (bank_id, vc, index)?
               |    +--- Use PerCoreCompileTimeDescriptor
               |         Example: DRAMStreamingMatmul (bank_id, vc)
               |         Result: up to 3N kernels where N = number of cores
               |
               +--- Both role AND unique constants?
                    +--- Use both descriptor types together
                         Example: DRAMStreamingMatmul (is_active_core + bank_id + vc)
                         Result: 3N kernels (each core is its own group)
```

---

## 2.1.10 The Split Path Algorithm Visualized

Consider a Mcast with 1 sender core at `(0,0)` and 4 receiver cores at `(0,1)-(1,2)`, where the sender is also in the receiver grid:

```
Step 2 - Per-core arg computation:
  (0,0): {is_sender_core: 1, is_receiver_core: 1}  <-- sender overlaps receiver grid
  (1,0): {is_sender_core: 0, is_receiver_core: 1}
  (0,1): {is_sender_core: 0, is_receiver_core: 1}
  (1,1): {is_sender_core: 0, is_receiver_core: 1}
  (0,2): {is_sender_core: 0, is_receiver_core: 0}  <-- outside both ranges

Step 3 - Grouping:
  Group A: {is_sender_core: 1, is_receiver_core: 1} --> [(0,0)]
  Group B: {is_sender_core: 0, is_receiver_core: 1} --> [(1,0), (0,1), (1,1)]
  Group C: {is_sender_core: 0, is_receiver_core: 0} --> [(0,2)]

Step 4 - Kernel generation:
  Group A --> 3 descriptors (NCRISC idx=0, BRISC idx=1, TRISC idx=2)
  Group B --> 3 descriptors (NCRISC idx=3, BRISC idx=4, TRISC idx=5)
  Group C --> 3 descriptors (NCRISC idx=6, BRISC idx=7, TRISC idx=8)

  Total: 9 KernelDescriptor objects, 3 KernelGroup objects
```

Each NCRISC descriptor for Group A will have `ncrisc_named_compile_time_args` merged with `[("is_sender_core", 1), ("is_receiver_core", 1)]`, enabling compile-time dead code elimination in the C++ kernel.

---

## 2.1.11 End-to-End Trace: Gather from Python to NOC Write

To make the abstract descriptor system concrete, let us trace the `GatherSingleCore.op()` call from Python through to the hardware NOC operation. Gather collects data from multiple sender cores onto a single receiver core.

### Step 1: Python -- Building Named Compile-Time Args

The gather builds `KernelDescriptor` objects directly. The key named compile-time args for a NOC0 sender kernel:

```python
named_args = [
    ("gather_dest_noc_x", gather_dest_noc_core.x),    # Physical NOC x of receiver
    ("gather_dest_noc_y", gather_dest_noc_core.y),    # Physical NOC y of receiver
    ("gather_data_size_bytes", total_size),             # Bytes per shard
    ("gather_receiver_semaphore_id", semaphore_id),    # Which semaphore to signal
    ("gather_src_cb", src_cb),                         # CB 0 (input data)
    ("gather_src_num_pages", src_num_pages),           # 1 page per sender
    ("gather_sender_grid_start_x", start_x),           # Grid bounds for index calc
    ("gather_sender_grid_start_y", start_y),
    ("gather_sender_grid_end_x", end_x),
    ("gather_sender_grid_end_y", end_y),
    ("gather_row_major", 1),
    ("gather_use_per_core_sender_idx", 0),             # Grid mode (not scattered)
    ("is_sender_core", 1),
    ("is_receiver_core", 0),
]
```

These 14 named args fully describe the sender's behavior at compile time.

### Step 2: C++ -- Kernel Reads Named Args as constexpr

The kernel file `micro_ops/gather/kernels/gather_kernel.cpp` includes the unified headers and reads args:

```cpp
#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/gather.hpp"

struct Core {
    static constexpr bool is_sender_core =
        get_named_compile_time_arg_val("is_sender_core") == 1;
    static constexpr bool is_receiver_core =
        get_named_compile_time_arg_val("is_receiver_core") == 1;
};
```

Every `get_named_compile_time_arg_val()` call is resolved at compile time. The compiler knows that for a sender kernel, `is_sender_core == true`.

### Step 3: C++ -- kernel_main() Populates the RTArgs Struct

The NCRISC section builds the `Gather::SenderArgs` struct:

```cpp
#if defined(COMPILE_FOR_NCRISC)
    if constexpr (Core::is_sender_core) {
        constexpr uint32_t src_cb = get_named_compile_time_arg_val("gather_src_cb");
        constexpr uint32_t src_num_pages = get_named_compile_time_arg_val("gather_src_num_pages");
        unified_kernels::setup_sharded_buffer(src_cb, src_num_pages);
    }

    uint32_t receiver_data_addr = get_common_arg_val<uint32_t>(0);  // Runtime arg

    Gather::SenderArgs gather_args{
        get_named_compile_time_arg_val("gather_dest_noc_x"),        // = 5 (physical)
        get_named_compile_time_arg_val("gather_dest_noc_y"),        // = 3 (physical)
        get_named_compile_time_arg_val("gather_data_size_bytes"),   // = 2048
        get_semaphore(get_named_compile_time_arg_val("gather_receiver_semaphore_id")),
        // ... remaining fields from named compile-time args ...
        receiver_data_addr,   // Runtime: output tensor buffer address
    };
```

Notice the hybrid: most fields are `constexpr` from named compile-time args, but `receiver_data_addr` comes from `get_common_arg_val<uint32_t>(0)` -- a runtime arg -- because the output tensor's L1 address may change between invocations.

### Step 4: C++ -- Gather::Op Executes the NOC Write

Inside `gather.hpp`, the NCRISC sender path:

```cpp
if constexpr (IsSenderCore) {
    // Compute linear index from grid position
    uint32_t core_index;
    if constexpr (UsePerCoreSenderIdx) {
        core_index = args.sender_idx;
    } else {
        core_index = unified_kernels::linear_id_in_grid<true>(
            args.sender_grid_start_x, args.sender_grid_start_y,
            args.sender_grid_end_x, args.sender_grid_end_y);
    }
    uint32_t offset = core_index * args.data_size_bytes;

    // Build 64-bit NOC address: physical (x,y) + L1 address
    const uint64_t dst_noc_coord = get_noc_addr(args.dest_noc_x, args.dest_noc_y, 0);
    uint64_t dst_data_noc_addr = dst_noc_coord | (uint64_t)(args.receiver_data_addr + offset);
    uint64_t dst_semaphore_noc_addr = dst_noc_coord | (uint64_t)args.receiver_semaphore_addr;

    // Wait for data in source CB
    cb_wait_front(args.src_cb, args.src_num_pages);

    // DMA the data directly into the receiver's L1
    uint32_t input_data_addr = get_read_ptr(args.src_cb);
    noc_async_write_one_packet<true, true>(
        input_data_addr, dst_data_noc_addr, args.data_size_bytes);

    // Signal the receiver via semaphore increment
    noc_semaphore_inc(dst_semaphore_noc_addr, 1);
    noc_async_posted_writes_flushed();

    if constexpr (pop_src) {
        cb_pop_front(args.src_cb, args.src_num_pages);
    }
    noc_async_atomic_barrier();
}
```

### The Complete Data Flow

```
Python: ("gather_dest_noc_x", 5)
    |
    v  [KernelDescriptor.named_compile_time_args]
    |
C++ kernel compile: get_named_compile_time_arg_val("gather_dest_noc_x") = 5
    |
    v  [constexpr resolution]
    |
Gather::SenderArgs{.dest_noc_x = 5, ...}
    |
    v  [passed to Op::impl()]
    |
get_noc_addr(5, 3, 0)  -->  64-bit NOC coordinate
    |
    v  [OR with L1 address]
    |
noc_async_write_one_packet(local_addr, noc_addr, size)
    |
    v  [hardware NOC DMA transfer]
    |
Data arrives in receiver core's L1 at (receiver_data_addr + offset)
```

On the receiver side (BRISC), the flow is symmetrical: the receiver waits on semaphores for all senders, then marks the output CB as populated. The receiver does not initiate any data movement -- the senders push data directly into its L1.

### Grid Mode vs. Scattered Mode

The gather supports two core indexing modes, selected by `gather_use_per_core_sender_idx`:

- **Grid mode** (`use_per_core_sender_idx=0`): The sender's linear index is computed at runtime from its position within a rectangular grid using `linear_id_in_grid<RowMajor>()`. Efficient ($O(1)$ kernel descriptors per NOC group) but requires a rectangular sender grid.
- **Scattered mode** (`use_per_core_sender_idx=1`): Each sender gets a unique `gather_sender_idx` via `PerCoreCompileTimeDescriptor`, creating $O(N)$ descriptors for $N$ senders. Supports arbitrary non-contiguous core sets.

---

## 2.1.12 The Python-to-C++ Bridge: A Summary

| Layer | Mechanism | Binding Time |
|-------|-----------|-------------|
| Python `named_compile_time_args` | `[("name", value)]` tuples | Program build |
| C++ `get_named_compile_time_arg_val("name")` | Returns `constexpr uint32_t` | Kernel compile |
| C++ `if constexpr (val == 1)` | Dead code elimination | Kernel compile |
| Python `common_runtime_args` | `[value, ...]` list | Program dispatch |
| C++ `get_common_arg_val<T>(index)` | Returns runtime value | Kernel execution |
| Python `per_core_runtime_args_descriptor` | `{(core_x, core_y): [args]}` | Program dispatch |
| C++ `get_arg_val<T>(index)` | Returns per-core runtime value | Kernel execution |

The descriptor system ensures that:
1. Role differentiation (sender/receiver/compute) is resolved at compile time via `constexpr` branching.
2. Topology information (grid bounds, NOC coordinates) is baked into the binary.
3. Only dynamic values (tensor addresses, fabric connection IDs) remain as runtime args.
4. The caller can look up kernel indices by role via `KernelGroup.get_group_by_arg()`.

---

**Next:** [`02_kernel_op_api_and_unified_headers.md`](./02_kernel_op_api_and_unified_headers.md)
