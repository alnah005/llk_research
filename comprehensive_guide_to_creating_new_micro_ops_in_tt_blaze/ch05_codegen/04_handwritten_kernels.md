# 5.4 Handwritten Kernels

Auto-generated kernels handle the vast majority of fused operations in TT-Blaze. But some operations need manual control over phase ordering, CT arg initialization, inter-phase resource sharing, or profiler scoping that the codegen cannot express. In these cases, a handwritten kernel is required.

---

## 5.4.1 When to handwrite

Auto-codegen is insufficient when you need:

1. **Custom phase interleaving.** The codegen emits phases in strict insertion order. If you need to interleave setup steps between phases (e.g., `setup_sharded_buffer` for down-proj weights between mcast1 and mcast2), you must handwrite.

2. **Shared init/teardown across non-adjacent phases.** The codegen's mcast group optimization handles sequential mcast followers, but if you need shared lifecycle management across non-mcast ops or non-sequential phases, handwriting is required.

3. **Manual CT arg construction.** Some ops need to construct `SenderArgs` or `ReceiverArgs` structs from CT args in ways that the template-driven auto-gen cannot express (e.g., conditional template parameters based on CT arg values, or hardcoded template parameters like `Loopback=false`).

4. **Overlapping phases with explicit synchronization.** When two processors can run concurrently across phase boundaries (e.g., NCRISC setting up the next mcast source while TRISC computes the current matmul), the codegen's strict sequential model cannot capture this.

5. **Custom profiler scoping.** The codegen wraps each phase in `DeviceZoneScopedN("PHASE_NAME")`. If you need finer-grained profiler markers (e.g., separating setup from execution within a phase), you must handwrite.

6. **Legacy op integration.** Ops that predate the auto-gen framework (e.g., the original `shared_expert`) use handwritten kernels with the older `deepseek_b1_ops::` namespace instead of `blaze::`.

In general, start with codegen. If the generated kernel does not produce correct results or does not meet performance targets, inspect the generated source (`f.program.generated_kernel`) and switch to handwriting only the parts that need customization.

---

## 5.4.2 The `kernel` class attribute

FusedOps specify a handwritten kernel via the `kernel` class attribute:

```python
class SharedExpert(FusedOp):
    name = "shared_expert"
    kernel = "blaze/ops/shared_expert/kernels/shared_expert_kernel.cpp"

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        # emit() calls that set up CBs, CT args, flags...
        # The handwritten kernel must be consistent with these
```

When `kernel` is set, `BlazeOp.register()` passes it through to `FusedOpConfig`:

```python
# In BlazeOp.register():
fused_kernel = cls.kernel if issubclass(cls, FusedOp) else None
register_fused_op(
    cls.name,
    FusedOpConfig(
        compose_fn=cls.compose,
        kernel=fused_kernel,  # None triggers codegen; path uses handwritten
    ),
)
```

During compilation, `BlazeCompiler._compile_fused_op()` passes this path to `FusedProgram`:

```python
if not config.kernel:
    ks = None      # codegen path
else:
    ks = config.kernel  # handwritten path
f = FusedProgram(kernel=ks, device=device, ...)
```

When `FusedProgram.build()` runs and `self.program._kernel` is already set (not `None`), the codegen step is skipped entirely.

> **Important:** For `MicroOp` subclasses, `cls.kernel` is the `.hpp` header path used for auto-derivation of CT args, not a standalone kernel source. Only `FusedOp` subclasses use `kernel` as a handwritten kernel path.

---

## 5.4.3 Handwritten kernel structure

A handwritten kernel follows a consistent structure:

### Includes

```cpp
#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/mcast.hpp"
#include "../../../unified_kernels/matmul.hpp"
#include "../../../unified_kernels/gather.hpp"
// ... one include per op type used in the fusion
```

The handwritten kernel includes the specific op headers it needs directly. The generated kernel uses generic `ops.hpp` / `kernel_op_api.hpp` / `kernel_utils.hpp`, which include all registered ops. The `named_args_generated.h` file is auto-included by the JIT via `-include` and does not need an explicit `#include` directive.

### Core role struct

The handwritten kernel defines a `Core` struct to hold per-core boolean flags:

```cpp
struct Core {
    static constexpr bool is_gate_compute_core =
        get_named_compile_time_arg_val("is_gate_compute_core") == 1;
    static constexpr bool is_up_compute_core =
        get_named_compile_time_arg_val("is_up_compute_core") == 1;
    static constexpr bool is_compute_core =
        is_gate_compute_core || is_up_compute_core;
    static constexpr bool is_gated_reduce_core =
        get_named_compile_time_arg_val("is_gated_reduce_core") == 1;
    static constexpr bool is_mcast_sender_core =
        get_named_compile_time_arg_val("is_mcast_sender_core") == 1;
    static constexpr bool is_mcast_receiver_core =
        get_named_compile_time_arg_val("is_mcast_receiver_core") == 1;
    static constexpr bool is_matmul_core =
        get_named_compile_time_arg_val("is_matmul_core") == 1;
    static constexpr bool is_gather_receiver_core =
        get_named_compile_time_arg_val("is_gather_receiver_core") == 1;
};
```

Note the derived flag `is_compute_core = is_gate_compute_core || is_up_compute_core`. This kind of flag composition is not possible with codegen -- it requires manual authoring. Derived flags reduce the number of CT args emitted by Python while enabling complex `if constexpr` guards in the kernel.

### Per-RISC argument setup

The kernel uses `#if defined(COMPILE_FOR_*)` guards to set up per-RISC argument structures:

```cpp
void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    // NCRISC: receiver args for mcasts, sender args for gathers
    using ActMcastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
    deepseek_b1_ops::Mcast::ReceiverArgs act_mcast_args{
        get_semaphore(get_named_compile_time_arg_val("act_mcast_data_receiver_semaphore")),
        get_named_compile_time_arg_val("act_mcast_dst_cb"),
        get_named_compile_time_arg_val("act_mcast_dst_num_pages"),
    };

#elif defined(COMPILE_FOR_BRISC)
    // BRISC: sender args for mcasts, receiver args for gathers
    using ActMcastCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("act_mcast_num_cores"),
        get_named_compile_time_arg_val("act_mcast_is_part_of_receiver_grid") == 1,
        /*Loopback=*/false>;

#elif defined(COMPILE_FOR_TRISC)
    // TRISC: compute args
    using ActMcastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
    deepseek_b1_ops::Mcast::ComputeArgs act_mcast_args{};
    deepseek_compute_kernel_init();
#endif
```

Each RISC constructs different `using` type aliases and argument structs. Note the `Loopback=false` template parameter -- this is hardcoded rather than derived from a CT arg, which is only possible in handwritten kernels.

### Phase execution with profiler zones

After arg construction, phases execute in order:

```cpp
    // Phase 1: Activation Mcast
    deepseek_b1_ops::Mcast::Op<ActMcastCTArgs, Core::is_mcast_sender_core,
        Core::is_mcast_receiver_core, Core::is_mcast_receiver_core,
        /*pop_src=*/true> act_mcast;
    act_mcast.init(act_mcast_args);
    {
        DeviceZoneScopedN("ACT_MCAST");
        act_mcast(act_mcast_args);
    }

    // Phase 2: Gate/Up Matmul
    {
        DeviceZoneScopedN("GATE_UP_MATMUL");
        deepseek_b1_ops::KNSlicedMatmul::Op<KNSlicedMatmulCTArgs,
            Core::is_compute_core, /*pop_act=*/true, /*pop_weights=*/false> gu_matmul;
        gu_matmul(gu_matmul_args);
    }
```

---

## 5.4.4 Interleaved buffer setup: the primary reason to handwrite

The critical advantage of the handwritten kernel is visible in the interleaved buffer setup:

```cpp
    // Mcast1 sends gated reduce output
    mcast(mcast_args);

    // Setup down proj weights AFTER mcast1 (L1 space freed by mcast1 send)
#if defined(COMPILE_FOR_NCRISC)
    if constexpr (Core::is_matmul_core) {
        unified_kernels::setup_sharded_buffer(matmul_in1,
            matmul_k_num_tiles * matmul_out_w_per_core);
    }
    if constexpr (Core::is_mcast_sender_core) {
        unified_kernels::setup_sharded_buffer(m2_src_cb, m2_src_pages);
    }
#endif

    // Mcast2 sends residual (after weights are set up)
    mcast(mcast2_args);
    act_mcast.teardown(act_mcast_args);
```

The buffer setup for `down_weights` is placed **between** mcast1 and mcast2 -- this is impossible with codegen, which does not emit `setup_sharded_buffer()` calls. The placement depends on application-level knowledge: which tensors are sharded, when they become available, and whether they share L1 with another buffer.

---

## 5.4.5 Walkthrough: `shared_expert_kernel.cpp`

**Source:** `tt-metal/models/demos/deepseek_v3_b1/fused_ops/shared_expert/kernels/shared_expert_kernel.cpp`

The shared expert kernel is the canonical example of a complex handwritten kernel. It fuses nine phases across 130 cores:

```
Phase 1: Act Mcast        -- sender (12,9) broadcasts activation to 130 cores
Phase 2: Gate/Up Matmul   -- 64 A cores (gate) + 64 B cores (up) do KN-sliced matmul
Phase 3: Gate Gather       -- 64 A cores send partials to sender core CB0
Phase 3b: Up Gather        -- 64 B cores send partials to sender core CB1
Phase 4: Gated Reduce      -- TRISC on sender core: SiLU(sum(gate)) * sum(up)
Phase 5: Mcast1            -- sender broadcasts gated reduce output to 130 cores
Phase 6: Mcast2            -- sender broadcasts residual to 130 cores (shared state with Mcast1)
Phase 7: Down Matmul       -- 112 matmul cores do down projection
Phase 8: Residual Add      -- 112 matmul cores add matmul output + residual shard
Phase 9: Output Gather     -- 112 matmul cores send results to sender core
```

### Why handwritten?

1. **Interleaved buffer setup:** Down proj weights are set up between mcast1 and mcast2 to optimize L1 memory usage. The weights need L1 space that is only available after mcast1 sends its data.
2. **Complex mcast sharing:** Three distinct mcast operations (act_mcast, mcast1, mcast2) share NOC command buffer state in a non-trivial pattern. Act_mcast has its own object; mcast1 and mcast2 share a second object.
3. **Per-RISC argument construction:** Each RISC has substantially different argument structures with explicit template parameter control (e.g., `Loopback=false`).
4. **Derived core flags:** The `Core::is_compute_core` flag is derived from `is_gate_compute_core || is_up_compute_core`, which cannot be expressed through the codegen's per-core flag system.

### Core roles

| Role | Cores | BRISC | NCRISC | TRISC |
|------|-------|-------|--------|-------|
| Sender | (12,9) | mcast send, gather recv | buffer setup, mcast recv | gated reduce |
| Gate compute (A) | 64 cores | - | buffer setup, mcast recv, gather send | gate matmul |
| Up compute (B) | 64 cores | - | buffer setup, mcast recv, gather send | up matmul |
| Matmul | 112 cores | - | buffer setup, mcast recv | down proj matmul, residual add |
| Idle | (12,8) | - | mcast recv (grid only) | - |

Each role is a composition of boolean flags that the C++ compiler uses for dead code elimination, ensuring each core runs only its assigned workload.

### CB layout

The handwritten kernel documents its CB layout in a header comment:

```
CB  0: A_gather_dst / reduce_g1 (sender, 64 face tiles, manual)
CB  1: B_gather_dst / reduce_g2 (sender, 64 face tiles, manual)
CB  2: intermediate (sender, 2 face tiles, manual)
CB  3: mcast1 source (sender, 1 face tile, manual, TRISC-filled)
CB  4: mcast1 dest / down_matmul_in0 (all 130 cores, manual)
CB  5: down_weights (112 matmul cores, tensor-backed)
CB  6: down_matmul_out (112 matmul cores, manual)
CB  7: output_gather_dst (sender, tensor-backed)
CB  8: act_mcast_src (sender, tensor-backed)
CB  9: act_mcast_recv (all 130 cores, manual)
CB 10: residual_mcast_src (sender, tensor-backed)
CB 11: residual_mcast_dst (all 130 cores, manual)
CB 12: residual_add_out (112 matmul cores, manual)
CB 13: gate_up_weights (128 compute cores, tensor-backed)
CB 14: gate_up_matmul_out (128 compute cores, 1 tile, manual)
```

This layout must match exactly what the `compose()` function allocates via `f.cb_from_tensor()`, `f.cb_scratch()`, and related calls. Any mismatch between the Python-side CB allocation and the kernel-side CB usage results in silent data corruption or hangs.

### Shared mcast state

Phases 5 and 6 (Mcast1 and Mcast2) demonstrate the leader/follower pattern:

```cpp
    // Phase 5: Mcast1 -- leader
    deepseek_b1_ops::Mcast::Op<McastCTArgs, ...> mcast;
    {
        DeviceZoneScopedN("MCAST1");
        mcast(mcast_args);      // Uses mcast_args (gated reduce output)
    }

    // ... inter-phase NCRISC setup for down weights and mcast2 source ...

    // Phase 6: Mcast2 -- follower (same Op object, different args)
    {
        DeviceZoneScopedN("MCAST2");
        mcast(mcast2_args);     // Same object, different payload
    }
    act_mcast.teardown(act_mcast_args);  // Teardown after last mcast phase
```

The same `mcast` object is called twice with different argument structs, sharing the NOC command buffer state. The `act_mcast.teardown()` at the end cleans up the activation mcast NOC state that was initialized at the beginning -- teardown happens after all mcast phases complete.

---

## 5.4.6 Maintaining consistency with emit()

The most critical requirement of a handwritten kernel is that it must be **perfectly consistent** with the `emit()` calls in `compose()`. Every CT arg name, CB ID, semaphore address, and role flag that the Python side sets up must be read by the kernel using exactly the same string names.

### Consistency checklist

When writing or modifying a handwritten kernel:

1. **CB ID order.** Verify that the CB allocation order in `emit()` (which determines sequential CB IDs) matches the CB layout documented in the kernel header.

2. **CT arg names.** For each `get_named_compile_time_arg_val("name")` in the kernel, verify that the corresponding `emit()` call produces that exact name string. Pay attention to prefix conventions: `emit()` uses dotted names (`"mcast.dst_cb"`) which become underscored in `get_named_compile_time_arg_val("mcast_dst_cb")`.

3. **Semaphore names.** Each `get_semaphore(get_named_compile_time_arg_val("sem_name"))` must correspond to a `f.semaphore("sem_name")` call in `emit()`.

4. **Role flags.** Each `Core::` flag must be emitted by `f.per_core_unified_ct_args()` or equivalent. Missing flags default to 0, which can cause incorrect `if constexpr` branch elimination.

5. **Phase count and order.** The number and order of phases in the kernel must match the `emit()` sequence. Extra or missing phases cause CB deadlocks.

---

## 5.4.7 Comparing auto-generated vs. handwritten

| Aspect | Auto-Generated | Handwritten |
|--------|---------------|-------------|
| Source | `kernel_codegen.py` generates `.cpp` at build time | Developer writes `.cpp` manually |
| Phase ordering | Strict insertion order from shadow graph | Developer controls arbitrary ordering |
| Mcast optimization | Leader/follower pattern via `McastGroup` | Developer manages shared Op variables |
| Setup interleaving | Not supported (no `setup_sharded_buffer` calls) | Developer can interleave setup between phases |
| Template parameters | Always from `ct_args::` structs | Can mix CT args with hardcoded values |
| Maintenance | Automatic: changes to `emit()` are reflected | Manual: kernel must be updated when `emit()` changes |
| Error risk | Low (codegen is correct by construction) | Higher (name mismatches, stale CB IDs, wrong RISC) |
| Performance tuning | Limited to what the codegen framework supports | Full control over phase overlap and resource timing |
| DPRINT injection | Automatic via `BLAZE_DEBUG_KERNELS` | Manual: add your own `DPRINT` statements |

---

## 5.4.8 How to write a handwritten kernel

### Step 1: Start from codegen output

Before writing from scratch, generate the auto-codegen output and use it as a starting point:

```python
f = FusedProgram(kernel=None, device=device, ...)
MyOp.emit(f, ...)
compiled = f.build()
print(f.program.generated_kernel)
```

Read the generated file to see the phase ordering, type aliases, and lifecycle calls that codegen produces. Then copy it and customize.

### Step 2: Define the Core struct

Create a `Core` struct with all role flags needed by the kernel. Derived flags (combinations of base flags) go here:

```cpp
struct Core {
    static constexpr bool is_active =
        get_named_compile_time_arg_val("is_active") == 1;
    static constexpr bool is_sender =
        get_named_compile_time_arg_val("is_sender") == 1;
    static constexpr bool is_receiver = is_active && !is_sender;
};
```

### Step 3: Add per-RISC argument setup

For each RISC, create the type aliases and argument structs inside `#if defined()` blocks. Use the Op's documented CT arg struct names (e.g., `blaze::Mcast::SenderCTArgs`, `blaze::Matmul::ComputeCTArgs`).

### Step 4: Write the phase sequence

Execute phases in the same order as the `emit()` calls. Wrap each phase in a `DeviceZoneScopedN` block for Tracy profiling. Insert `setup_sharded_buffer` calls between phases as needed.

### Step 5: Set the kernel attribute

```python
class MyFusedOp(FusedOp):
    kernel = "blaze/ops/my_op/kernels/my_kernel.cpp"
```

### Step 6: Test with BLAZE_EXPORT=1

Export the shadow graph to verify that CB IDs, semaphore indices, and CT arg values match your kernel's expectations. Run the fused op and verify correctness against a PyTorch golden reference.

### Step 7: Consider migrating to codegen later

If your custom control flow can be expressed as a sequence of standard MicroOps, refactor to use codegen. The performance is identical (the generated code calls the same Op struct methods), but the codegen path provides automatic lifecycle elision, mcast grouping, and profiling instrumentation.

---

## 5.4.9 The transition path: codegen to handwritten

In practice, most ops start as codegen-based compositions. The transition to handwritten happens when:

1. Performance profiling (via Tracy + `DeviceZoneScopedN` markers) reveals phase overlap opportunities.
2. L1 pressure requires careful setup interleaving to avoid simultaneous allocation peaks.
3. The fusion pattern requires non-standard lifecycle management (e.g., shared init across unrelated ops).

When transitioning:

1. Keep the `emit()` method unchanged -- it is the source of truth for CT arg wiring.
2. Start from the auto-generated kernel as a template.
3. Make surgical changes and verify with `BLAZE_DEBUG_KERNELS=1` that each phase still executes correctly.
4. Add the `kernel` attribute to the FusedOp class to activate the handwritten path.

The shadow graph is always built regardless of which path is used. Even with a handwritten kernel, `compose()` calls `emit()` which records the shadow graph. This graph is used for visualization (the Blaze visualizer), debug export (`BLAZE_EXPORT=1`), and CB lifetime tracking. The shadow graph is simply not used for codegen when a handwritten kernel is provided.

---

## Key takeaways

- **Handwrite only when codegen is insufficient.** Custom setup interleaving, shared lifecycle across non-adjacent phases, overlapping phases, and template parameter control are the primary reasons.
- **The kernel's CT arg reads must exactly match the emit() method's CT arg writes.** There is no automated consistency checker. Use the codegen output as a reference.
- The `Core` struct pattern collects all role flags in one place, enabling compound `constexpr` expressions (e.g., `is_compute_core = is_gate || is_up`).
- Per-RISC `#if defined(COMPILE_FOR_*)` blocks construct args from CT args. No-op ops use default-constructed args.
- The `shared_expert_kernel.cpp` walkthrough demonstrates interleaved setup, shared mcast lifecycle, explicit template parameters, and the CB layout documentation convention that handwritten kernels should follow.
- Always start from the auto-generated kernel. Copy it, customize it, and verify correctness before deploying.
- Start with codegen; graduate to handwritten only when profiling or correctness demands it.
