# 4.2 Interception Point Analysis: Hook Points, Thread Safety, and Serialization Costs

## Overview

This section identifies every viable interception point in the TT-Metal dispatch
pipeline, evaluates thread safety at each point, measures the data structures
available for capture, and estimates the serialization overhead. The analysis uses
a weighted scoring matrix to arrive at an evidence-based recommendation for which
hook points the runtime interceptor should instrument.

> **Cross-reference:** Section 4.1 defines the seven pipeline stages and their data
> flows. Ch2 (State Inventory) Section 1 catalogs the compiled binary state captured
> here. Ch3 (LightMetal Foundation) Section 1 describes the existing LightMetal
> capture system that this pipeline's interception extends.

---

## 4.2.1 Candidate Hook Points

| ID | Hook Point | Function | File |
|----|-----------|----------|------|
| H1 | Post-compile | `ProgramImpl::compile()` return | `tt_metal/impl/program/program.cpp:1527` |
| H2 | Post-finalize | `ProgramImpl::finalize_offsets()` return | `tt_metal/impl/program/program.cpp:1702` |
| H3 | Post-assemble | `assemble_device_commands()` return | `tt_metal/impl/program/dispatch.cpp:2037` |
| H4 | Post-update | `update_program_dispatch_commands()` return | `tt_metal/impl/program/dispatch.cpp:2299` |
| H5 | Pre-write | Before `write_program_command_sequence()` | `tt_metal/impl/program/dispatch.cpp:2481` |
| H6 | Post-enqueue | `enqueue_mesh_workload()` return | `tt_metal/distributed/fd_mesh_command_queue.cpp:453` |

---

## 4.2.2 Detailed Analysis of Each Hook Point

### H1: Post-Compile

**Available:** Kernel binaries (`kernel->binaries(build_key)`), metadata, core ranges,
kernel hash. **Not available:** RTAs (not yet set), dispatch commands, memory offsets,
`KernelGroup` launch messages. **Thread safety:** Safe -- `sync_build_steps()` ensures
all compilations complete before return. No locks held at function boundary.

**Cost:** ~100 KB for a 3-kernel program (~20 us memcpy). **Verdict:** Good for binary
caching (indexed by FNV1a hash), insufficient for dispatch capture.

### H2: Post-Finalize

**Available (adds to H1):** `ProgramConfig` per core type, `KernelGroup` structures,
`ProgramTransferInfo`. **Not available:** Final RTAs, dispatch commands, config buffer
slot addresses. **Cost:** ~2 KB (<1 us). **Verdict:** Captures L1 layout but not
dynamic state.

### H3: Post-Assemble

**Available (adds to H1, H2):** Complete `ProgramCommandSequence` with all sub-commands,
but with *initial* (template-stage) values. **Not available:** Final config buffer
addresses, stall counts, launch message ring buffer pointers, updated RTAs from
`update_program_dispatch_commands()`.

**Thread safety:** Called from `generate_dispatch_commands()`, safe to read.

**Serialization:** The `HostMemDeviceCommand` objects own backing memory via
`vector_aligned<uint32_t>`, accessible through `data()`/`size_bytes()`. The
`cb_configs_payloads` are raw pointers into the config buffer command, and
`launch_messages` contains `View` objects aliasing the launch message buffer.
Total: ~13 KB, ~3 us.

**Verdict:** Strong candidate but misses per-dispatch delta state. For re-dispatch
(cached path), several fields are not yet final.

### H4: Post-Update (RECOMMENDED PRIMARY HOOK POINT)

**Available:** Everything. The `ProgramCommandSequence` contains *final* values:
correct config buffer addresses, final stall counts, updated CB configs, all RTAs
(including cross-sequence copies via `rta_updates`), correct launch message ring
buffer write pointers, final go signal with dispatch core coordinates.

**Thread safety:** Called under `lock_api_function_()` in
`FDMeshCommandQueue::enqueue_mesh_workload()`:

```cpp
// fd_mesh_command_queue.cpp:258-259
void FDMeshCommandQueue::enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) {
    auto lock = lock_api_function_();  // std::unique_lock on CQ mutex
}
```

| Structure | Safety | Notes |
|-----------|--------|-------|
| `ProgramCommandSequence` | Safe | Under CQ mutex |
| `ProgramImpl` internals | Safe | Immutable after compile, under CQ mutex |
| `WorkerConfigBufferMgr` | Safe | Per-CQ, under CQ mutex |
| `LaunchMessageRingBufferState` | Safe | Per-CQ per-sub-device, under CQ mutex |

**Cost:** ~13 KB, ~1.5 us (identical structures to H3, just patched in-place).

**PROPOSED: H4 Interceptor Interface:**

```cpp
class DispatchInterceptor {
public:
    virtual void on_program_dispatch(
        const ProgramImpl& program,
        const ProgramCommandSequence& cmd_seq,
        const ProgramDispatchMetadata& dispatch_md,
        IDevice* device,
        SubDeviceId sub_device_id,
        uint32_t expected_num_workers_completed) = 0;
};
```

Arguments available at this point:
1. `program` -- full `ProgramImpl` with kernel groups, configs, kernel objects
2. `cmd_seq` -- finalized `ProgramCommandSequence` with all commands patched
3. `dispatch_md` -- config buffer addresses and stall information
4. `device` -- device handle for coordinate translations
5. `sub_device_id` -- which sub-device is being targeted
6. `expected_num_workers_completed` -- synchronization state

**Verdict:** Best single-point hook. Maximum data availability, strong thread safety.

### H5: Pre-Write

Functionally equivalent to H4 but on the critical path between update and hardware
write. `SystemMemoryManager::issue_queue_reserve()` may block. H4 preferred.

### H6: Post-Enqueue

Too late. `LaunchMessageRingBufferState` incremented (temporal state lost), lock
released (another thread may mutate program RTAs).

---

## 4.2.3 Comparison Matrix

| Criterion | H1 | H2 | H3 | H4 | H5 | H6 |
|-----------|----|----|----|----|----|----|
| Binaries available | Yes | Yes | Yes | Yes | Yes | Yes |
| RTAs final | No | No | Partial | **Yes** | **Yes** | Stale |
| CB configs final | No | No | Partial | **Yes** | **Yes** | Stale |
| Launch msgs final | No | No | Partial | **Yes** | **Yes** | Stale |
| Config buffer addrs | No | No | No | **Yes** | **Yes** | Yes |
| Thread safe | Yes | Yes | Yes | **Yes** | Yes | No |
| Latency impact | None | None | Low | **Low** | Medium | None |
| Sufficient for replay | No | No | Partial | **Yes** | **Yes** | Partial |

---

## 4.2.4 Weighted Scoring Matrix

### Criteria and Weights

| # | Criterion | Weight | Rationale |
|---|-----------|--------|-----------|
| 1 | Data completeness (single pass) | 25% | Fewer hooks = fewer failure modes |
| 2 | Data freshness (no stale RTAs) | 20% | Must capture actual dispatched state |
| 3 | Implementation complexity | 15% | Maintainability matters |
| 4 | Capture overhead (microseconds) | 10% | Must not regress hot-path perf |
| 5 | Thread safety | 10% | No races, no additional locks |
| 6 | LightMetal compatibility | 10% | Leverage existing infrastructure |
| 7 | Architecture portability (WH/BH/Quasar) | 5% | Same hook must work everywhere |
| 8 | Resilience to refactors | 5% | Stable API surface preferred |

### Scoring (1--5, higher is better) and Weighted Results

$$\text{Score}(H) = \sum_{i=1}^{8} w_i \cdot s_i(H)$$

| Criterion | H1 | H3 | H4 | H6 | H1+H4 |
|-----------|----|----|----|----|--------|
| 1. Data completeness | 2 | 4 | **5** | 3 | **5** |
| 2. Data freshness | 2 | 3 | **5** | 3 | **5** |
| 3. Impl. complexity | 4 | 4 | **5** | 3 | 4 |
| 4. Capture overhead | 4 | 4 | **4** | 3 | 3 |
| 5. Thread safety | 4 | 5 | **5** | 2 | **5** |
| 6. LightMetal compat | 4 | 4 | **5** | 3 | **5** |
| 7. Arch portability | 4 | 5 | **5** | 4 | **5** |
| 8. Refactor resilience | 3 | 3 | **4** | 3 | 4 |

| Hook | Weighted Score | Rank |
|------|---------------|------|
| H1 | **3.05** | 4 |
| H3 | **3.85** | 3 |
| **H4** | **4.85** | **1** |
| H6 | **2.95** | 5 |
| H1+H4 | **4.55** | **2** |

H4 scores 4.85/5.00, the clear winner. The dual-hook H1+H4 strategy scores
slightly lower (4.55) due to implementation complexity, but adds binary
deduplication.

---

## 4.2.5 The RuntimeArgsData Aliasing Problem

After `assemble_device_commands()`, the `rt_args_data` field in `RuntimeArgsData`
is retargeted to point into the `HostMemDeviceCommand` buffer (`dispatch.cpp:557-560`).
The kernel's `runtime_args_data()` now returns a view into the command buffer, not
the original `vector<uint32_t>`. When `SetRuntimeArgs()` is called, new values are
written directly into the command buffer.

**Capture implication:** At H4, RTAs are embedded in the `HostMemDeviceCommand`
buffers. No separate serialization of `RuntimeArgsData` objects is needed. The
`rta_updates` vector must be recorded for cross-sequence dependency replay.

**PROPOSED: RTA Serialization Protocol:**
1. Serialize all `runtime_args_command_sequences` (contains final unique RTAs)
2. Serialize `program_config_buffer_command_sequence` (contains common RTAs)
3. Record `rta_updates` as `{src_offset_in_seq, dst_offset_in_seq, size}` tuples
   for replay-time cross-referencing

This three-step protocol captures the complete RTA state without needing to
traverse the `RuntimeArgsData` pointer aliases.

---

## 4.2.6 Circular Buffer Config Update Flow

CB configs are another mutable component. The `cb_configs_payloads` vector stores
pointers into the `program_config_buffer_command_sequence` where CB data lives.
During `update_program_dispatch_commands()`, these are overwritten:

```cpp
// dispatch.cpp:2190-2213
for (const auto& cbs_on_core_range :
     cached_program_command_sequence.circular_buffers_on_core_ranges) {
    uint32_t* cb_config_payload = cached_program_command_sequence.cb_configs_payloads[i];
    for (const std::shared_ptr<CircularBufferImpl>& cb : cbs_on_core_range) {
        const uint32_t cb_address = cb->address();
        const uint32_t cb_size = cb->size();
        // ... write into cb_config_payload
    }
}
```

`CircularBufferImpl` objects may have addresses/sizes modified between dispatches
(via `UpdateCircularBufferTotalSize()`). At H4, final values are embedded in the
packed write payload. No special handling needed.

---

## 4.2.7 The `HostMemDeviceCommand` Serialization Contract

Each `HostMemDeviceCommand` is backed by either an internal
`vector_aligned<uint32_t>` (non-hugepage variant) or a raw pointer to a hugepage
region. For the `ProgramCommandSequence`, all commands use the non-hugepage variant,
with heap-allocated memory accessible via `data()` and `size_bytes()`.

At H4, `size_bytes() == write_offset_bytes()` for all commands (asserted in
`assemble_device_commands()`). Serialization is:

```
for each HostMemDeviceCommand cmd in ProgramCommandSequence:
    write(cmd.size_bytes())
    write(cmd.data(), cmd.size_bytes())
```

**Total:** ~13 KB at ~10 GB/s memcpy = ~1.3 us + ~0.2 us metadata = **~1.5 us**.

---

## 4.2.8 Multi-CQ and Multi-Device Considerations

H4 fires once per program in the workload's program map within
`enqueue_mesh_workload()`. For $N$ programs on $M$ sub-grids, H4 fires $N$ times.

**PROPOSED: Capture Correlation:**
1. `MeshWorkload::id` as correlation key
2. `device_range` (`MeshCoordinateRange`) per program
3. Monotonic dispatch sequence number per CQ
4. Global timestamp (`std::chrono::steady_clock`)
5. `SystemMemoryManager` event ID (`get_next_event()`)

Multiple CQs operate independently under separate `lock_api_function_()` mutexes.
Correlation metadata allows global ordering reconstruction.

---

## 4.2.9 Trace Mode and Existing Infrastructure

When in trace recording mode (`sysmem_manager.get_bypass_mode()`), the same
`update_program_dispatch_commands()` runs, so H4 works identically. The interceptor
should detect trace mode and annotate captures accordingly.

### Inspector Pattern Integration

TT-Metal's existing `Inspector` pattern provides compile-event callbacks. The
interceptor extends this with:

```cpp
// PROPOSED: New Inspector callback
Inspector::program_dispatch_captured(
    ProgramImpl* program, IDevice* device,
    const ProgramCommandSequence& cmd_seq,
    const ProgramDispatchMetadata& dispatch_md);
```

### LightMetal Extension

The existing LightMetal system (`LightMetalCaptureContext`) records API calls as
FlatBuffer commands, tracking objects via global-ID maps (`buffer_id_to_global_id_map_`,
`program_id_to_global_id_map_`, `kernel_to_global_id_map_`,
`cb_handle_to_global_id_map_`). It captures the *sequence of API calls* but not
the *resolved state* after finalization -- no compiled binaries, resolved L1
offsets, NOC coordinate mappings, or launch message contents.

**PROPOSED:** The interceptor extends LightMetal with an `API_PLUS_SNAPSHOT` capture
mode that serializes a `KernelSnapshot` FlatBuffer at H4, linked to programs via
`program_id_to_global_id_map_`. This requires no changes to the existing LightMetal
replay path -- snapshots are stored as an auxiliary section.

### Tracy Compatibility

The dispatch path already uses Tracy (`dispatch.cpp:13 #include <tracy/Tracy.hpp>`).
The interceptor's ~1.5 us overhead is comparable to a Tracy zone annotation.

### Watcher Compatibility

The watcher system instruments launch messages with `watcher_kernel_ids[]`
(`dev_msgs.h:159`). The interceptor can reuse these to identify which kernel is
running on each RISC, avoiding duplicate tracking.

### Architecture Portability

H4 sees `HalProgrammableCoreType` and branches on architecture. On Quasar, it
must capture NEO cluster assignment and compute processor indices from
`QuasarComputeKernel::compute_processors_`
(source: `tt_metal/impl/kernels/kernel.hpp:376-507`).

---

## 4.2.10 Risk Analysis

| Risk | Mitigation |
|------|-----------|
| `update_program_dispatch_commands` refactored | Single callback at function exit; easy to relocate |
| `ProgramCommandSequence` layout changes | Read via `data()`/`size_bytes()` API |
| Program caching bypasses assembly | H4 fires every dispatch (update always runs) |
| Multi-device / mesh dispatch | H4 fires per-program, captures per-device |
| Binary not available at H4 | H1 captures binaries; referenced by hash at H4 |

---

## 4.2.11 Recommended Interceptor Architecture

The recommended architecture uses a **dual-hook (H1 + H4)** strategy:

```
+-------------------+     +------------------+     +----------------+
| H1: Post-Compile  | --> | Binary Cache     | --> | Dedup Store    |
| (one-time per     |     | (hash -> binary) |     | (on disk)      |
|  kernel build)    |     +------------------+     +----------------+
+-------------------+

+-------------------+     +------------------+     +----------------+
| H4: Post-Update   | --> | Snapshot Buffer  | --> | Capture File   |
| (every dispatch)  |     | (ring of         |     | (streaming     |
|                   |     |  ProgramSnapshots|     |  write)        |
+-------------------+     +------------------+     +----------------+
```

**Why H1+H4 rather than H3 alone:** H3 captures data before
`update_program_dispatch_commands()` runs. On re-dispatch (the common case),
several fields are not yet final:

| Field | H3 State | H4 State |
|-------|----------|----------|
| Config buffer base addresses | Template values | **Final slot addresses** |
| Stall counts | Default | **Computed from sync state** |
| Launch message write pointers | Template | **Current ring buffer position** |
| Go signal dispatch coordinates | Placeholder | **Final dispatch core XY** |
| RTA cross-references | Not yet copied | **All `rta_updates` applied** |
| `host_assigned_id` | Not set | **Set to program runtime ID** |

**PROPOSED: H4 Implementation Sketch:**

```cpp
// PROPOSED: At end of update_program_dispatch_commands()
#ifdef TT_INTERCEPT_ENABLED
if (RuntimeInterceptor::is_capture_active()) {
    RuntimeInterceptor::capture_program_snapshot(
        program, cached_program_command_sequence, dispatch_md,
        device, sub_device_id, expected_num_workers_completed);
}
#endif
```

---

## Key Takeaways

1. **H4 (post-`update_program_dispatch_commands`) is the optimal primary hook,**
   scoring 4.85/5.00 in the weighted evaluation. It provides complete, final dispatch
   state under strong thread safety guarantees with ~1.5 us serialization overhead.

2. **The H1+H4 dual-hook architecture is recommended.** H1 captures immutable
   binaries once per compilation (amortized), H4 captures per-dispatch delta state
   (~13 KB steady-state).

3. **H3 (post-assemble) is insufficient for re-dispatch.** On cached dispatches,
   `update_program_dispatch_commands()` patches config buffer addresses, stall counts,
   launch message pointers, and go signal coordinates that H3 would miss.

4. **The `RuntimeArgsData` aliasing problem is self-resolving at H4.** RTAs are
   embedded in the `HostMemDeviceCommand` buffers, so serializing command buffers
   automatically captures final RTA values.

5. **Multi-CQ and multi-device capture requires correlation metadata** (CQ ID,
   timestamps, mesh workload ID) to reconstruct global dispatch ordering.

6. **The existing Inspector pattern provides a natural integration point** for
   interceptor hooks, allowing runtime enable/disable without modifying dispatch code.
