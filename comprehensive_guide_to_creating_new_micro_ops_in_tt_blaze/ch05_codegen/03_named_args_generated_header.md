# 5.3 The `named_args_generated.h` Header

Every TT-Blaze kernel -- whether auto-generated or handwritten -- is compiled with an auto-generated header file called `named_args_generated.h`. This file is **automatically included** in every kernel compilation via the compiler's `-include` flag -- no explicit `#include` is needed in the kernel source. The header provides two namespaces that the kernel reads at compile time:

- **`ct_args::`** -- static constexpr values for compile-time arguments
- **`rt_args::`** -- accessor templates for runtime arguments

Understanding how this header is structured and how to inspect it is essential for debugging CT arg mismatches and verifying that the codegen pipeline produced the expected values.

---

## 5.3.1 The CT arg flow: Python to C++

Compile-time arguments (CT args) are the primary mechanism for parameterizing generated kernels. The flow is:

```
emit() calls f.unified_ct_args() / f.ncrisc_ct_args() / etc.
    |
    v
FusedProgram captures (name, value) tuples in BlazeProgram._*_args lists
    |
    v
BlazeProgram.build() passes args to UnifiedKernelDescriptor
    |
    v
UKD passes named_compile_time_args to ttnn.KernelDescriptor per RISC
    |
    v
tt-metal JIT generates named_args_generated.h
    |
    v
C++ kernel reads values as static constexpr fields
```

The JIT compiler receives `(name, value)` tuples for each RISC and generates the header automatically. There is no manual step.

---

## 5.3.2 The ct_args:: namespace

### Structure

Each unique prefix in the CT arg tuples becomes a struct in the `ct_args` namespace. The struct contains `static constexpr uint32_t` fields for each arg:

```cpp
namespace ct_args {

struct act_mcast {
    static constexpr uint32_t src_cb = 8;
    static constexpr uint32_t dst_cb = 9;
    static constexpr uint32_t src_num_pages = 224;
    static constexpr uint32_t dst_num_pages = 224;
    static constexpr uint32_t data_size_bytes = 458752;
    static constexpr uint32_t dest_noc_start_x = 1;
    static constexpr uint32_t dest_noc_start_y = 1;
    static constexpr uint32_t dest_noc_end_x = 14;
    static constexpr uint32_t dest_noc_end_y = 10;
    static constexpr uint32_t num_cores = 130;
    static constexpr uint32_t data_sender_semaphore = 0x3FC00000;
    static constexpr uint32_t data_receiver_semaphore = 0x3FC00040;
    static constexpr uint32_t is_part_of_receiver_grid = 1;
};

struct shared_expert__gu {
    static constexpr uint32_t act_cb = 9;
    static constexpr uint32_t weights_cb = 13;
    static constexpr uint32_t out_cb = 14;
    static constexpr uint32_t k_offset = 0;
    static constexpr uint32_t k_per_core = 112;
    static constexpr uint32_t act_total_tiles = 224;
};

}  // namespace ct_args
```

This is zero-cost: the values are baked into the binary as immediate operands. There is no L1 read, no semaphore wait, no function call.

### Prefix-to-struct naming convention

The mapping from `emit()` prefix to struct name follows the `_to_ct_args_type()` function:

| Python prefix | CT arg name | C++ struct |
|---------------|------------|------------|
| `"act_mcast"` | `"act_mcast.src_cb"` | `ct_args::act_mcast` |
| `"shared_expert.gu"` | `"shared_expert.gu.act_cb"` | `ct_args::shared_expert__gu` |
| `"mcast2"` | `"mcast2.dst_cb"` | `ct_args::mcast2` |

Dots (`.`) become double underscores (`__`) and dashes (`-`) become single underscores (`_`) in the struct name, because C++ identifiers cannot contain dots or dashes.

### How auto-generated kernels use ct_args

In auto-generated kernels, the `ct_args::` struct is passed as a template argument to the Op:

```cpp
using ActMcast = blaze::Mcast::Op<ct_args::act_mcast>;
```

Inside the Op struct's C++ implementation, members are accessed via the `A::` template pattern:

```cpp
template <typename A>
struct ReaderCTArgs {
    static constexpr CB dst_cb = A::dst_cb;
    static constexpr uint32_t dst_num_pages = A::dst_num_pages;
    static constexpr Semaphore receiver_sem = A::data_receiver_semaphore;
};
```

The compiler resolves `A::dst_cb` to `ct_args::act_mcast::dst_cb` at compile time. Every field is `static constexpr`, so there is zero runtime cost.

### How handwritten kernels use CT args

Handwritten kernels (Section 5.4) can access the same values through the lower-level API:

```cpp
constexpr uint32_t src_cb = get_named_compile_time_arg_val("act_mcast_src_cb");
```

> **Warning:** Note the naming difference. In auto-generated kernels, the dotted prefix becomes a namespace separator: `ct_args::act_mcast::dst_cb`. In handwritten kernels using `get_named_compile_time_arg_val()`, the dot is replaced by an underscore: `"act_mcast_dst_cb"`. The `named_args_generated.h` header supports both access patterns. Mixing styles within a kernel causes compile errors.

---

## 5.3.3 Per-core CT args

Some compile-time args have **different values on different cores**. The most common example is role flags: `is_sender` is `1` on the sender core and `0` on all other cores. Another example is `sender_idx` for gather ops, where each sender core gets a unique index.

### UnifiedCompileTimeCoreDescriptor: boolean flags

The most common case is a role flag. For example, an mcast op sets `is_active=1` on all cores in the receiver grid and `is_active=0` on the sender:

```python
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_active", receiver_grid, enabled=True),
])
```

This creates a `UnifiedCompileTimeCoreDescriptor`:

```python
@dataclass
class UnifiedCompileTimeCoreDescriptor:
    named_compile_time_arg: str    # e.g. "mcast.is_active"
    core_range: CoreRangeSet       # cores where value applies
    value: int                     # value for cores in core_range (1)
    other_value: int               # value for all other cores (0)
```

### PerCoreCompileTimeDescriptor: per-core-varying values

For values that are unique to each core (e.g., a sender index in a gather op), `PerCoreCompileTimeDescriptor` provides an explicit per-core mapping:

```python
PerCoreCompileTimeDescriptor(
    named_compile_time_arg="gather.sender_idx",
    core_values=[(core, idx) for idx, core in enumerate(sender_cores)],
    other_value=0,
)
```

This creates one kernel compilation group per unique value. For a gather with 128 sender cores, there are 128 kernel groups (each with one core), plus one group for all non-sender cores with `sender_idx=0`. This is why gather ops with many senders can have long JIT times.

### How per-core specialization works

The `UnifiedKernelDescriptor.get_kernel_descriptors()` method groups cores by their unique compile-time arg combinations. Cores with identical values share a single kernel binary; cores with different values get separate compilations. This is how the same `.cpp` file compiles to different binaries for sender vs. receiver cores.

### In the generated header

Per-core CT args appear in `named_args_generated.h` with the value specific to that core's binary:

```cpp
// Binary for core (0,0): is_active = 1, is_sender = 0
namespace ct_args {
struct mcast {
    static constexpr uint32_t is_active = 1;
    static constexpr uint32_t is_sender = 0;
};
}

// Binary for core (12,9): is_active = 0, is_sender = 1
namespace ct_args {
struct mcast {
    static constexpr uint32_t is_active = 0;
    static constexpr uint32_t is_sender = 1;
};
}
```

The kernel source is identical for both compilations -- only the header values differ. This means `if constexpr (ct_args::mcast::is_sender)` compiles to true on core `(12,9)` and false on all others, with zero runtime branching.

---

## 5.3.4 Triple compilation (NCRISC, BRISC, TRISC)

The unified kernel `.cpp` file is compiled **three times** by tt-metal's JIT -- once for each RISC processor:

| Compilation | Define | RISC | Typical responsibilities |
|-------------|--------|------|--------------------------|
| 1 | `COMPILE_FOR_NCRISC` | DM1 (reader) | NOC reads, sharded buffer setup, gather send |
| 2 | `COMPILE_FOR_BRISC` | DM0 (writer) | NOC writes, mcast send, gather receive |
| 3 | `COMPILE_FOR_TRISC` (0, 1, 2) | Compute | Math ops (matmul, reduce, activation functions) |

Each compilation sees a **different** `named_args_generated.h` because the CTArgEngine assigns different args to different RISCs. For example, `act_mcast.src_cb` might appear in both NCRISC and BRISC headers (both need to know which CB to read/write), while `matmul.k_num_tiles` appears only in the TRISC header (only compute needs it).

Args declared via `f.unified_ct_args()` or `f.per_core_unified_ct_args()` appear in all three RISC headers.

The `UnifiedKernelDescriptor` manages this triple compilation:

```python
@dataclass
class UnifiedKernelDescriptor:
    kernel_source: str
    core_ranges: Any
    ncrisc_named_compile_time_args: list  # [(name, value), ...]
    brisc_named_compile_time_args: list
    trisc_named_compile_time_args: list
    # ...
```

Each list is passed to its respective `ttnn.KernelDescriptor` during build, which triggers the tt-metal JIT compiler to generate the RISC-specific `named_args_generated.h` and compile the kernel for that RISC.

### Per-core splitting

When per-core CT arg descriptors are present, the triple compilation becomes `3 * N` compilations where `N` is the number of unique core groups. The `UnifiedKernelDescriptor._get_split_kernel_descriptors()` method:

1. Enumerates all cores in `core_ranges`.
2. Computes each core's complete named CT arg set (base args + per-core overrides).
3. Groups cores by their unique arg combination (using frozen-dict hashing).
4. Creates one kernel descriptor triple (NCRISC, BRISC, TRISC) per unique group.

### TRISC sub-cores

Note that TRISC has three sub-compilations (`trisc0`, `trisc1`, `trisc2`) corresponding to the unpack, math, and pack stages of the compute pipeline. Each may receive different compile-time args. The kernel can use sub-core guards:

```cpp
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0
    // unpack-only code
#endif
```

---

## 5.3.5 The rt_args:: namespace

Runtime arguments follow a similar pattern but are resolved at dispatch time rather than compile time:

```python
f.trisc_rt_args([
    ("matmul.weights_l1_address", weight_tensor.buffer_address()),
])
```

On the device, runtime args are read via:

```cpp
uint32_t addr = rt_args::get<rt_args::ns::weights_l1_address>(0);
```

### When to use RT vs CT args

| Criterion | CT arg | RT arg |
|-----------|--------|--------|
| Changes between dispatches? | No (baked into binary) | Yes |
| Runtime cost | Zero (immediate operand) | Small L1 read per access |
| Requires recompilation? | Yes | No |
| Typical use | CB IDs, tile counts, NOC coords, flags | Weight L1 addresses, dynamic offsets |

The most common RT arg in TT-Blaze is `weights_l1_address` for matmul ops, where the L1 buffer address can change when the weight tensor is reallocated between dispatches. The compiler builds these automatically:

```python
def _build_direct_weight_rt_args(self, graph, tensors, merged_overrides):
    """Build TRISC runtime L1 addresses for direct-address matmul weights."""
    args = []
    for node in graph.topological_order():
        if node.spec.name in ("matmul", "matmul_cb"):
            port_name = "in1"
        elif node.spec.name == "kn_sliced_matmul":
            port_name = "weights"
        else:
            continue
        prefix = engine._compute_prefix(node, graph, ...)
        arg_name = f"{prefix}weights_l1_address"
        args.append((arg_name, self._direct_weight_l1_address(tensors[tensor_name])))
    return args
```

### RT arg variants

FusedProgram exposes several RT arg APIs:

| Method | Scope | Use case |
|--------|-------|----------|
| `f.ncrisc_rt_args(args)` | Common to all cores, NCRISC | L1 addresses, per-dispatch parameters |
| `f.brisc_rt_args(args)` | Common to all cores, BRISC | Same, for writer |
| `f.trisc_rt_args(args)` | Common to all cores, TRISC | Weight addresses |
| `f.ncrisc_rt_arg_arrays(args)` | Common array, NCRISC | Multi-value runtime data |
| `f.ncrisc_per_core_rt_arg_arrays(args)` | Per-core array, NCRISC | Values varying by core AND having multiple elements |

---

## 5.3.6 Location and cache structure

The JIT-compiled binaries and generated headers are cached at:

```
~/.cache/tt-metal-cache/
  <kernel_hash>/
    ncrisc/
      named_args_generated.h
      <binary>
    brisc/
      named_args_generated.h
      <binary>
    trisc0/
      named_args_generated.h
      <binary>
    trisc1/
      named_args_generated.h
      <binary>
    trisc2/
      named_args_generated.h
      <binary>
```

The kernel hash in the directory name is derived from the kernel source content plus all compile-time args. Changing any CT arg value produces a different hash and triggers recompilation.

---

## 5.3.7 Inspecting resolved values

### Via program.generated_kernel

The `BlazeProgram.generated_kernel` property returns the auto-generated kernel source (before JIT compilation):

```python
compiled = f.build()
source = f.program.generated_kernel  # Returns the .cpp source string or None
```

### Via compile_engines()

The `compile_engines()` function runs all four engines on a graph and returns an `EngineResult`:

```python
from blaze import compile_engines

result = compile_engines(graph, user_args={...})

# Inspect CT arg tuples per RISC:
for risc, tuples in result.ct_tuples.items():
    print(f"\n{risc}:")
    for name, value in tuples:
        print(f"  {name} = {value}")
```

### Via shadow graph CT values

Each `OpNode` in the shadow graph stores the CT values that were active when it was recorded:

```python
shadow = f._shadow_graph
for node in shadow.nodes:
    print(f"Op: {node.id}")
    for name, value in sorted(node.ct_values.items()):
        print(f"  {name} = {value}")
```

### Via BLAZE_EXPORT visualization

Setting `BLAZE_EXPORT=1` dumps a JSON file containing all CT args, CB assignments, and RT args. The interactive visualizer renders these as an annotated dataflow graph.

### Via engine validation (f.validate())

For comprehensive validation, `FusedProgram.validate()` runs all three engines on the shadow graph and diffs their outputs against the imperative allocations:

```python
result = f.validate()
if not result["ok"]:
    for diff_type, diffs in result.items():
        if diff_type != "ok":
            print(f"{diff_type}: {diffs}")
```

This compares engine-computed CB IDs, semaphore indices, and CT arg values against what `emit()` actually produced, catching discrepancies between the declarative schema and the imperative implementation.

### Using BLAZE_L1_PROFILE

Setting `BLAZE_L1_PROFILE=1` in the environment reports CB allocations with their L1 addresses, dtypes, tile shapes, and page sizes on every `program.run()`:

```
[l1_profile] CB  4: scratch  page_size=2048  num_pages=8  total=16384
[l1_profile] CB  5: tensor   page_size=2048  num_pages=16 total=32768
```

This is especially useful when debugging CB ID mismatches between the Python `emit()` declarations and the C++ kernel's expectations.

---

## 5.3.8 Common CT arg debugging patterns

### "CT arg not found" compile error

If the JIT fails with an error about a missing CT arg, the cause is usually:

1. **Name mismatch**: The C++ header declares `A::my_field` but `emit()` emits `f"{prefix}.my_feild"` (typo).
2. **Wrong RISC**: The field is in `WriterCTArgs` (BRISC) but `emit()` puts it in `f.ncrisc_ct_args()`.
3. **Missing emit**: The op's `emit()` method does not emit the arg at all -- perhaps it is conditionally skipped.

**Diagnosis**: Inspect the generated `named_args_generated.h` for the failing compilation. Search for the expected field name. If it is missing, trace back through the `emit()` call to find the gap.

### Wrong value at runtime

If the kernel compiles but produces incorrect results:

1. Print the CT arg tuples (see Section 5.3.7) and verify values against expected.
2. Check that CB IDs match the expected tensor binding order.
3. Verify semaphore addresses correspond to the correct global semaphores.
4. For per-core args, verify that the right cores get the right values.

### CB deadlock from wrong ID

If the kernel hangs:

1. Enable `BLAZE_DEBUG_KERNELS=1` and check which phase each core is stuck in.
2. Inspect the CB IDs in `named_args_generated.h` -- a producer writing to CB 4 while the consumer waits on CB 5 indicates a wiring bug in `emit()`.
3. Cross-reference with the shadow graph edges to verify data flow.

### Schema validation in FusedProgram

FusedProgram's `_validate_ct_args_against_schema()` method runs after every `_record_op()` call, comparing emitted CT args against the op's declared schema. It warns on missing args:

```
UserWarning: CT arg schema mismatch for 'mcast' (prefix='shared_expert__mcast'):
missing: ['dest_noc_end_x', 'dest_noc_end_y']
```

This validation catches the common mistake of forgetting to emit a CT arg in your `emit()` function before the error manifests as a kernel hang or incorrect results.

---

## Key takeaways

- `named_args_generated.h` is the auto-generated bridge between Python declarations and C++ kernel code. It contains `ct_args::` structs with `static constexpr` fields providing zero-cost access.
- The header is generated three times (once per RISC), with each version containing only the CT args declared for that processor.
- Per-core CT args (role flags, sender indices) cause the JIT to produce multiple kernel compilations per RISC, one per unique value set. This enables compile-time dead-code elimination for role-specific code paths.
- Runtime args (`rt_args::`) provide per-dispatch changing values at the cost of a small L1 read, avoiding recompilation.
- Triple compilation (NCRISC, BRISC, TRISC) partitions the kernel source via `#if defined(COMPILE_FOR_*)` guards, producing three independent binaries from one source file.
- Multiple inspection tools (shadow graph values, `compile_engines()`, `BLAZE_EXPORT`, `f.validate()`, strict mode, `BLAZE_L1_PROFILE`) help debug CT arg issues during op development.
