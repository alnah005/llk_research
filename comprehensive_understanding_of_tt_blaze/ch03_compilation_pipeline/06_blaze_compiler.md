# 06 -- BlazeCompiler

The `BlazeCompiler` is the final stage of the compilation pipeline. It accepts a `BlazeGraph` and real mesh tensors, orchestrates per-device compilation across the mesh, and produces a `MeshCompiledProgram` (or `CompiledProgram`) that can be dispatched via `ttnn.generic_op()`. It supports two compilation paths -- the engine path for multi-node graphs and the fused-op path for single-node graphs with registered composition functions.

**Source file:** `blaze/compiler.py`

---

## 1. Class Structure

```python
class BlazeCompiler:
    def __init__(self, mesh_device):
        self._mesh_device = mesh_device
        self._num_rows = mesh_device.shape[0]
        self._num_cols = mesh_device.shape[1]
        self._ctx = DeviceContext.from_device(mesh_device)
        self._sem_dict: dict = {}           # user-named semaphores
        self._graph_sem_pool: list = []     # engine-assigned global semaphores
        self._tensor_dict: dict = {}        # user-named tensors
        self._internal_tensor_dict: dict = {}  # compiler-internal tensors
```

The compiler takes a `mesh_device` -- even a 1x1 mesh. It creates a `DeviceContext` (wrapping grid size, NOC coordinate translation) and initializes containers for semaphores and tensors that persist across the per-device compilation loop.

---

## 2. The `compile()` Entry Point

```python
def compile(self, graph, tensors, output_tensor,
            user_args=None, kernel_source=None,
            user_overrides=None, per_device_overrides=None,
            scratch_tensors=None, scratch_mapping=None,
            select_prefix="", all_scratch_mapped=False,
            noop_prefixes=None, compact_scratch_cbs=True,
            noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC) -> MeshCompiledProgram:
```

### 2.1 Pre-Computation Phase

Before the per-device loop, the compiler performs device-agnostic work:

1. **Reset state**: Clears `_sem_dict`, `_graph_sem_pool`, `_tensor_dict`, and `_internal_tensor_dict` from any prior compilation.
2. **Pre-compute engine results**: Runs `CBEngine().assign(graph)` and `SemEngine().assign(graph)` once, storing results in an `_EngineResults` named tuple. These are reused across all devices since they depend only on graph structure, not device-specific state.
3. **Auto-create global semaphores**: If the op class declares `num_global_semaphores`, creates them upfront via `ttnn.create_global_semaphore()`.

```python
class _EngineResults(NamedTuple):
    """Pre-computed device-agnostic engine outputs, reusable across devices."""
    cb_assignments: dict
    sem_assignments: list

engine_results = _EngineResults(
    cb_assignments=CBEngine().assign(graph),
    sem_assignments=SemEngine().assign(graph),
)
```

### 2.2 Mesh Decomposition

The compiler decomposes mesh tensors into per-device tensors:

```python
per_device_inputs = _split_tensors(tensors)
per_device_outputs = ttnn.get_device_tensors(output_tensor)
```

`_split_tensors()` handles `OverlappedView` objects (views into a shared L1 buffer) by caching the per-device split of the underlying tensor and re-wrapping each per-device shard:

```python
def _split_tensors(tensors):
    per_mesh = {}
    def split_mesh(t):
        if id(t) not in per_mesh:
            per_mesh[id(t)] = ttnn.get_device_tensors(t)
        return per_mesh[id(t)]
    out = {}
    for name, t in tensors.items():
        if isinstance(t, OverlappedView):
            out[name] = [replace(t, tensor=dt) for dt in split_mesh(t.tensor)]
        else:
            out[name] = split_mesh(t)
    return out
```

The identity-based caching (`id(t)`) ensures that multiple `OverlappedView` names pointing at the same backing tensor share the same per-device split, preserving the tensor identity that downstream grouping relies on.

### 2.3 Per-Device Compilation Loop

```python
for idx in range(num_devices):
    row, col = divmod(idx, self._num_cols)
    dev_tensors = {name: per_dev[idx] for name, per_dev in per_device_inputs.items()}
    merged_args = {
        "mesh_coord": (row, col),
        "mesh_shape": (self._num_rows, self._num_cols),
        "mesh_device": self._mesh_device,
        **(user_args or {}),
        **dev_overrides,
    }
    compiled = self._compile_single(
        self._ctx, graph, dev_tensors, per_device_outputs[idx],
        user_args=merged_args, engine_results=engine_results, ...
    )
    mesh_pd[MeshCoordinate(row, col)] = compiled.program_descriptor
```

Each device gets its slice of the per-device input/output tensors, plus `mesh_coord`, `mesh_shape`, and `mesh_device` injected into user args for inter-device coordination.

### 2.4 Assembly

After the per-device loop:

```python
program = MeshCompiledProgram(
    mesh_program_descriptor=mesh_pd,
    io_tensors=mesh_io_tensors,
    output_tensor=output_tensor,
    shadow_graph=getattr(compiled, "shadow_graph", None),
    lifetime_tensors=all_per_device_io_tensors
        + list(self._tensor_dict.values())
        + list(self._internal_tensor_dict.values()),
    lifetime_semaphores=list(self._sem_dict.values()) + list(self._graph_sem_pool),
)
```

The `lifetime_tensors` and `lifetime_semaphores` lists pin Python objects whose raw pointers (buffer addresses, L1 addresses) are baked into the program descriptor. Without these pins, garbage collection would free the underlying L1 memory, causing use-after-free at dispatch time.

---

## 3. Two Compilation Paths

`_compile_single()` dispatches based on graph structure:

```python
def _compile_single(self, ctx, graph, tensors, output_tensor, ...):
    if len(graph.nodes) == 1:
        config = get_fused_op_config(graph.nodes[0].spec.name)
        if config is not None:
            return self._compile_fused_op(graph, tensors, output_tensor, config, ...)
    return self._compile_for_device(ctx, graph, tensors, output_tensor, ...)
```

### 3.1 Engine Path (`_compile_for_device`)

For multi-node graphs (or single-node graphs without a registered `FusedOpConfig`). This is the primary path for graph-API compiled pipelines. The six steps are detailed in Section 4 below.

### 3.2 Fused Op Path (`_compile_fused_op`)

For single-node graphs whose op has a registered `FusedOpConfig` (checked via `get_fused_op_config(...) is not None`). The config's `compose_fn` is then invoked during compilation. This path delegates to the composition API (see [Chapter 1 -- Two-API Design](../ch01_architecture/03_two_api_design.md)):

```python
def _compile_fused_op(self, graph, tensors, output_tensor, config, ...):
    f = FusedProgram(
        kernel=config.kernel, device=device,
        math_fidelity=..., name=node.spec.name,
        _sem_dict=self._sem_dict,
        _tensor_dict=self._tensor_dict,
        _internal_tensor_dict=self._internal_tensor_dict,
        ...
    )
    config.compose_fn(f, tensors, output_tensor, merged_args)
    # Handle noop prefixes
    if noop_prefixes and f._shadow_graph is not None:
        _tag_noop_nodes(f._shadow_graph, noop_prefixes)
    # Auto-wire outputs
    ...
    built = f.build()
    compiled = CompiledProgram(built.program_descriptor, built.io_tensors, ...)
    compiled.shadow_graph = f._shadow_graph
    return compiled
```

The fused op path:

1. Creates a `FusedProgram` with the same semaphore/tensor dicts as the compiler (shared across the per-device loop for deduplication).
2. Calls the op's `compose_fn(f, tensors, output_tensor, merged_args)`.
3. The compose function calls `emit()` on child ops, building CBs, CT args, and role flags.
4. Noop prefixes tag shadow graph nodes for empty-body codegen.
5. Output wiring supports multiple strategies: explicit `cb_output()`, tracked `_last_output_cb_ids`, or fallback `wire_output_from_graph()`.
6. `f.build()` produces the final `ProgramDescriptor`.

---

## 4. The Engine Path: Six Steps

`_compile_for_device()` implements the engine path:

### Step 1: Run CB and Sem Engines

```python
if engine_results is not None:
    cb_assignments, sem_assignments = engine_results
else:
    cb_assignments = CBEngine().assign(graph)
    sem_assignments = SemEngine().assign(graph)
```

Engine results are pre-computed once in `compile()` and reused across all devices -- they are device-agnostic.

### Step 2: Build User Overrides and Grid Context

```python
merged_overrides = self._build_user_overrides(graph, user_args or {})
grid_context = self._build_grid_context(graph, ctx)
```

NOC coordinate translation for mcast and gather ops:

- **Mcast**: `dest_noc_start_x/y`, `dest_noc_end_x/y`, `num_cores`, `is_part_of_receiver_grid`
- **Gather/moe_gather**: `dest_noc_x`, `dest_noc_y`, `sender_grid_start/end_x/y`

### Step 3: Build CB Descriptors

For each category of CB assignment:

- `ext_in`: `ttnn.cb_descriptor_from_sharded_tensor(cb_id, tensor)` -- backed by real tensor
- `intermed`/`internal`: `_build_cb_from_ref(cb_id, num_pages, core_ranges, ref_tensor)` -- tile/dtype from reference tensor
- `ext_out`: `ttnn.cb_descriptor_from_sharded_tensor(cb_id, output_tensor)`

This method also builds `cb_for_ct: dict[str, dict[str, int]]` -- the node-to-port CB ID mapping that the CTArgEngine needs.

### Step 4: Create Global Semaphores and Resolve Addresses

```python
global_sems, sem_addrs = self._create_global_semaphores(sem_assignments)
sem_for_ct = self._build_sem_for_ct(sem_assignments, sem_addrs)
```

`_create_global_semaphores()` creates `ttnn.GlobalSemaphore` objects for each unique `sem_id`, populating `self._graph_sem_pool`. `_build_sem_for_ct()` maps `(node_id, protocol)` to the integer L1 address:

```python
def _build_sem_for_ct(self, sem_assignments, sem_addrs):
    result = {}
    for sa in sem_assignments:
        result.setdefault(sa.node_id, {})[sa.protocol] = sem_addrs[sa.sem_id]
    return result
```

### Step 5: Generate CT Args and Build UnifiedKernelDescriptor

```python
ct_tuples = CTArgEngine().generate_tuples(
    graph, cb_for_ct, sem_for_ct,
    grid_context=grid_context,
    user_overrides=merged_overrides,
    strict=True,
)
```

Also builds direct-weight runtime args for matmul ops (L1 buffer addresses for weight tensors).

```python
ukd = UnifiedKernelDescriptor(
    kernel_source=kernel_source,
    core_ranges=all_cores,
    ncrisc_named_compile_time_args=ct_tuples["ncrisc"],
    brisc_named_compile_time_args=ct_tuples.get("brisc", []),
    trisc_named_compile_time_args=ct_tuples["trisc"],
    trisc_named_common_runtime_args=trisc_rt_args,
    trisc_compute_config=compute_config,
    unified_compile_time_core_descriptors=core_descriptors,
    per_core_compile_time_descriptors=per_core_descriptors,
)
kernel_result = ukd.get_kernel_descriptors()
```

The `UnifiedKernelDescriptor` creates NCRISC, BRISC, and TRISC `KernelDescriptor` objects. When per-core compile-time overrides exist (e.g., `is_sender` flags, per-core `sender_idx` values), it splits the core range into groups with different compile-time arg values and creates separate kernel descriptors per group.

### Step 6: Assemble ProgramDescriptor

```python
program_descriptor = ttnn.ProgramDescriptor(
    kernels=kernel_result.kernels,
    cbs=cb_descriptors,
)
```

---

## 5. CompiledProgram and MeshCompiledProgram

### 5.1 CompiledProgram

```python
class CompiledProgram:
    def __init__(self, program_descriptor, io_tensors,
                 output_tensor=None, deallocate_tensors=None,
                 global_semaphores=None):
        self.program_descriptor = program_descriptor
        self.io_tensors = io_tensors
        self.output_tensors = _normalize_output_tensors(output_tensor)
        self._deallocate_tensors = deallocate_tensors or []
        self._global_semaphores = global_semaphores or []
        self.shadow_graph = None

    def run(self):
        if os.environ.get("BLAZE_L1_PROFILE"):
            from blaze.l1_profile import print_cb_stats
            print_cb_stats(self)
        return _run_program(self.program_descriptor, self.io_tensors,
                           self.output_tensors, self._deallocate_tensors)
```

Used internally for per-device compilation results and by `FusedProgram.build()`.

### 5.2 MeshCompiledProgram

```python
class MeshCompiledProgram:
    def __init__(self, mesh_program_descriptor, io_tensors,
                 output_tensor=None, deallocate_tensors=None,
                 shadow_graph=None, lifetime_tensors=None,
                 lifetime_semaphores=None):
        self.mesh_program_descriptor = mesh_program_descriptor
        self.io_tensors = io_tensors
        self.output_tensors = _normalize_output_tensors(output_tensor)
        self._lifetime_tensors = lifetime_tensors or []
        self._lifetime_semaphores = lifetime_semaphores or []

    def run(self):
        if os.environ.get("BLAZE_L1_PROFILE"):
            from blaze.l1_profile import print_cb_stats
            print_cb_stats(self)
        return _run_program(self.mesh_program_descriptor, self.io_tensors,
                           self.output_tensors, self._deallocate_tensors)
```

The mesh variant holds:

- `mesh_program_descriptor`: A `ttnn.MeshProgramDescriptor` with per-coordinate `ProgramDescriptor` entries.
- `_lifetime_tensors`: Per-device io_tensors + named tensors + internal tensors -- kept alive to prevent UAF.
- `_lifetime_semaphores`: User-named + engine-assigned global semaphores -- same UAF prevention.
- `shadow_graph`: The decomposed `BlazeGraph` from the fused-op path's `FusedProgram`, used by the visualizer.

---

## 6. Dispatch via `ttnn.generic_op()`

Both `run()` methods call the shared `_run_program()`:

```python
def _run_program(descriptor, io_tensors, output_tensors, deallocate_tensors):
    ttnn.generic_op(io_tensors, descriptor)
    for t in deallocate_tensors:
        ttnn.deallocate(t)
    return output_tensors[0] if output_tensors else io_tensors[-1]
```

`ttnn.generic_op()` is the low-level dispatch entry point in TT-Metal. It takes the ordered `io_tensors` list (inputs followed by outputs) and the program descriptor, enqueues the program for execution on the device, and returns when complete. The `io_tensors` order must match the CB descriptor order in the program -- this is enforced by `_build_io_tensors()`.

---

## 7. IO Tensor Ordering

```python
def _build_io_tensors(self, graph, tensors, output_tensor):
    io_tensors = []
    added = set()
    # Inputs in spec port order (topological)
    for node in graph.topological_order():
        for port in node.spec.input_ports:
            if (node.id, port.name) not in graph.external_input_ports:
                continue
            for tensor_name, ports in graph.tensor_to_ports.items():
                if (node.id, port.name) in ports and tensor_name not in added:
                    io_tensors.append(self._unwrap_tensor(tensors[tensor_name]))
                    added.add(tensor_name)
    # Primary output
    io_tensors.append(output_tensor)
    # Additional output port tensors
    for node in graph.topological_order():
        for port in node.spec.output_ports:
            if port.name in tensors and port.name not in added:
                io_tensors.append(self._unwrap_tensor(tensors[port.name]))
                added.add(port.name)
    return io_tensors
```

The ordering is: external input tensors (in topological port order, deduplicated by name), then the primary output tensor, then any additional output port tensors. `OverlappedView` objects are unwrapped to their backing tensor since `ttnn.generic_op()` expects raw `ttnn.Tensor` objects.

This ordering is operationally important: the `io_tensors` list must match the CB descriptor order in the `ProgramDescriptor` for `ttnn.generic_op()` to correctly bind tensor-backed CBs to their underlying buffers.

---

## 8. Compute Configuration

```python
def _get_compute_config(self, graph, merged_overrides=None):
    fidelity_map = {"HiFi4": ttnn.MathFidelity.HiFi4, "LoFi": ttnn.MathFidelity.LoFi}
    fp32_dest_acc_en = False
    if merged_overrides:
        for node in graph.nodes:
            ov = merged_overrides.get(node.id, {})
            if "fp32_dest_acc_en" in ov:
                fp32_dest_acc_en = bool(ov["fp32_dest_acc_en"])
                break
    node = graph.nodes[0]
    spec = node.spec
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=fidelity_map.get(spec.preferred_math_fidelity, ttnn.MathFidelity.HiFi4),
        math_approx_mode=spec.math_approx_mode,
        fp32_dest_acc_en=fp32_dest_acc_en,
        dst_full_sync_en=fp32_dest_acc_en,
    )
    unpack_modes = self._unpack_to_dest_mode_from_overrides(graph, merged_overrides)
    if unpack_modes is not None:
        cfg.unpack_to_dest_mode = unpack_modes
    return cfg
```

The compute config is derived from the first node's `OpSpec` fields. User overrides can inject:

- `fp32_dest_acc_en`: Enables FP32 accumulation in the destination register (used by RMSNorm). When enabled, `dst_full_sync_en` is also set to `True`.
- `unpack_to_dest_mode`: Per-CB unpack-to-dest mode vector (64 entries).
- `unpack_to_dest_fp32_cb_indices`: Convenience -- specifies which CB indices should use `UnpackToDestFp32`.

---

## 9. Per-Device Overrides

The `compile()` method supports per-device specialization through the `per_device_overrides` callable:

```python
for idx in range(num_devices):
    row, col = divmod(idx, self._num_cols)
    dev_overrides = per_device_overrides(row, col) if per_device_overrides else {}
    merged_args = {
        "mesh_coord": (row, col),
        "mesh_shape": (self._num_rows, self._num_cols),
        "mesh_device": self._mesh_device,
        **(user_args or {}),
        **dev_overrides,
    }
```

The compiler automatically injects `mesh_coord`, `mesh_shape`, and `mesh_device` into every device's user args. The `per_device_overrides(row, col)` callback allows the caller to provide device-specific CT arg values (e.g., different tile counts for different chips in a multi-device topology).

---

## 10. BLAZE_EXPORT

When `BLAZE_EXPORT=1` is set, the compiler automatically exports a visualizer JSON file:

```python
def _auto_export(self, graph, program, tensors):
    from .viz_export import export_viz_data
    data = export_viz_data(export_graph, name=name, tensors=tensors, rt_args=rt_args)
    # Embed pipeline topology if pipeline_config.json exists
    if os.path.exists(pipeline_path):
        data["pipeline"] = json.load(f)
    json.dump(data, f, indent=2)
```

The export includes the graph structure, tensor metadata, runtime args, and optionally the pipeline topology. The output directory defaults to `generated/viz` but can be overridden via `BLAZE_EXPORT_PATH`. The file is named after the first node's op name (e.g., `shared_expert_viz.json`).

---

## 11. BLAZE_L1_PROFILE

When `BLAZE_L1_PROFILE` is set in the environment, both `CompiledProgram.run()` and `MeshCompiledProgram.run()` call `print_cb_stats()` from `blaze.l1_profile` before dispatching the program. This outputs a diagnostic summary of CB allocations and L1 memory usage, which is useful for debugging L1 pressure issues in large fused pipelines.

---

## 12. End-to-End Compilation Summary

```
User code:
    with blaze.fuse() as ctx:
        out = blaze.matmul({"in0": ext_act, "in1": ext_wt}, grid=grid)
        out = blaze.gelu({"input": out}, grid=grid)

    compiler = BlazeCompiler(mesh_device)
    result = compiler.compile(ctx.graph, tensors={"act": t_act, "wt": t_wt},
                              output_tensor=t_out, user_args={...})
    output = result.run()

Compilation flow:
    1. FusionContext.__exit__() builds BlazeGraph, validates
    2. BlazeCompiler.compile():
       a. Pre-compute: CBEngine.assign(graph), SemEngine.assign(graph) -> _EngineResults
       b. Auto-create global semaphores (if num_global_semaphores declared)
       c. Split mesh tensors per device
       d. For each device:
          i.   _compile_single() dispatches to engine or fused-op path
          ii.  Engine path:
               - Build merged user overrides + NOC grid context
               - Build CB descriptors from assignments + real tensors
               - Create global semaphores, resolve L1 addresses
               - CTArgEngine.generate_tuples() with strict=True
               - Build UnifiedKernelDescriptor -> KernelDescriptors
               - Assemble ProgramDescriptor(kernels, cbs)
          iii. Fused-op path:
               - Create FusedProgram, call compose_fn
               - Tag noop nodes, auto-wire outputs
               - Build via FusedProgram.build()
       e. Assemble MeshProgramDescriptor
       f. Pin lifetime tensors and semaphores
       g. Optional: auto-export visualizer JSON (BLAZE_EXPORT)
    3. result.run():
       - Optional: print_cb_stats (BLAZE_L1_PROFILE)
       - ttnn.generic_op(io_tensors, mesh_program_descriptor)
```

---

<- [05 -- Kernel Codegen](./05_kernel_codegen.md) | [Index](index.md) | Chapter 4 ->
