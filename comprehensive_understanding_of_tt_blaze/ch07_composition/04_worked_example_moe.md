# 07.04 -- Worked Example: MoE Fused Op

[<-- Worked Example -- SharedExpert](03_worked_example_shared_expert.md)

This section walks through the MoE (Mixture of Experts) fused op, the most complex composition in TT-Blaze. At 275 lines, it integrates RMSNorm, activation multicast, a routed expert branch, a shared expert branch, branch merging, and cross-device reduction into a single fused kernel. The source is `blaze/ops/moe/op.py`. This section builds on the MoE composition overview in [Ch. 6.05 -- MoE Ops](../ch06_op_library/05_moe_ops.md), which covers the 8-variant matrix, router architecture, and micro-op count. This section focuses on the composition mechanics: CBHandle chain, prefix hierarchy, mesh-aware logic, and CB reconfiguration.

---

## Architecture: Dual-Branch from Single Activation Mcast

The MoE op builds two parallel expert branches from a shared activation multicast:

```
                     act (input activation tensor)
                      |
                 [RMSNorm]
                      |
                 [Mcast] ----------> act_mcast_handle
                      |                     |
           +----------+----------+          |
           |                     |          |
    [RoutedExpert]        [SharedExpert]    |
           |                     |          |
           |              [Mcast output]    |
           |                     |          |
           +--- [ResidualAdd] ---+
                      |
                [ReduceToOne]
                      |
                   output
```

Both branches read from the same `act_mcast_handle` (a single scratch CB on all cores). This avoids duplicating the activation multicast -- one NOC transaction serves both branches.

---

## Class Declaration

```python
class MoE(FusedOp):
    name: str = "moe"

    act: Input = Input()
    rmsnorm_gamma: Input = Input()
    gate_weight: Input = Input()
    gate_bias: Input = Input()
    routed_swiglu_up_weights: Input = Input()
    routed_swiglu_gate_weights: Input = Input()
    routed_swiglu_down_weights: Input = Input()
    shared_gate_up_weight: Input = Input()
    shared_down_weight: Input = Input()
    red2one_intermediate: Input = Input()
    red2one_output: Input = Input()
    out: Output = Output()
```

Eleven input ports and one output. The input count reflects the number of weight matrices, routing tensors, and pre-allocated reduction buffers the MoE layer needs. Each becomes a key in the `tensors` dict passed to `compose()`.

---

## compose(): Unpacking User Args

```python
@classmethod
def compose(cls, f, tensors, output, user_args):
    ua = user_args or {}
    cls.emit(
        f,
        act=tensors["act"],
        rmsnorm_gamma=tensors["rmsnorm_gamma"],
        gate_weight=tensors["gate_weight"],
        gate_bias=tensors["gate_bias"],
        routed_swiglu_up_weights=tensors["routed_swiglu_up_weights"],
        routed_swiglu_gate_weights=tensors["routed_swiglu_gate_weights"],
        routed_swiglu_down_weights=tensors["routed_swiglu_down_weights"],
        shared_gate_up_weight=tensors["shared_gate_up_weight"],
        shared_down_weight=tensors["shared_down_weight"],
        red2one_intermediate=tensors["red2one_intermediate"],
        red2one_output=tensors["red2one_output"],
        prefix=ua.get("prefix", "moe"),
        use_silu=ua.get("use_silu", False),
        eps=ua.get("eps", 1e-20),
        routing_scaling_factor=ua.get("routing_scaling_factor", 2.5),
        routed_index_offset=ua.get("routed_index_offset", 0),
        routed_fp32_dest_acc_en=ua.get("routed_fp32_dest_acc_en", False),
        # ... (20+ kwargs for both expert branches)
        red2one_root_mesh_coord=ua["red2one_root_mesh_coord"],
        red2one_is_torus=ua.get("red2one_is_torus", False),
        output_tensor=output,
    )
```

Notice `red2one_root_mesh_coord` has no default -- it is a required user arg. The caller must specify which device in the mesh is the reduction root.

---

## emit(): Phase-by-Phase Walkthrough

### Phase 1: Input Preparation and RMSNorm

```python
@staticmethod
def emit(f, act, rmsnorm_gamma, gate_weight, gate_bias,
         routed_swiglu_up_weights, routed_swiglu_gate_weights,
         routed_swiglu_down_weights, red2one_intermediate, red2one_output,
         *, shared_gate_up_weight, shared_down_weight, prefix, ...):

    if isinstance(act, ttnn.Tensor):
        width = act.shape[-1]
        act_ti = TileInfo.from_tensor(act)
        act_mcast_pages = (act.shape[-2] // act_ti.tile.tile_shape[0]) * \
                          (act.shape[-1] // act_ti.tile.tile_shape[1])
        full_tile, _ = interpret_tile(width)
        full_page_sz = full_tile.get_tile_size(ttnn.bfloat16)
        rmsnorm_input = f.cb_from_tensor(act, tile=full_tile, page_size=full_page_sz)
        _rmsnorm_notify = True
    else:
        # act is a CBHandle from a parent op
        # ... derive tile info from CBHandle metadata
```

When `act` is a raw tensor, the op allocates a tensor-backed CB with full-width tile override (RMSNorm needs the full row). When `act` is a `CBHandle` (from a parent composition), the op may use `f.cb_alias()` to reinterpret the existing CB with a wider tile format:

```python
    else:
        # Face-view (1xW) activation -> full-width tile for RMSNorm
        if act.tile_desc.height == 1:
            itile, n_tiles = interpret_tile(width)
            page_sz = itile.get_tile_size(ttnn.bfloat16)
            rmsnorm_input = f.cb_alias(
                act, tile=itile, page_size=page_sz,
                total_size=page_sz * n_tiles,
            )
```

This is the CB aliasing pattern from [Ch. 7.02](02_advanced_patterns.md) in action. No data copy -- the alias CB reads the same L1 bytes through a different tile format.

### Phase 2: RMSNorm and Activation Mcast

```python
    rmsnormed_act = RMSNorm.emit(
        f,
        rmsnorm_input,
        rmsnorm_gamma,
        prefix=BlazeOp.child_prefix(prefix, "rmsnorm"),
        cores=f.sender_grid,
        epsilon=rmsnorm_epsilon,
        pop_input=False,
        scalar=_scalar,
        notify_input=_rmsnorm_notify,
    )
    act_mcast_handle = Mcast.emit(
        f,
        rmsnormed_act,
        prefix=BlazeOp.child_prefix(prefix, "act_mcast"),
        dst_num_pages=act_mcast_pages,
        dst_tile_info=act_ti,
    )
```

RMSNorm runs on the sender core only (`cores=f.sender_grid`). Its output is a scratch CB on the sender, which Mcast fans out to all cores. The `act_mcast_handle` is the branching point: both the routed expert and the shared expert read from this handle.

Note `pop_input=False` -- the original activation is not popped because the shared expert branch also needs the original (unmodified) activation as a residual/bias input.

### Phase 3: Routed Expert Branch

```python
    row, col = f.mesh_coord
    rows, cols = f.mesh_shape
    index_offset = row * cols + col + routed_index_offset

    routed_out = RoutedExpert.emit(
        f,
        act_mcast_handle,        # <-- same handle both branches read
        gate_weight,
        gate_bias,
        routed_swiglu_up_weights,
        routed_swiglu_gate_weights,
        routed_swiglu_down_weights,
        prefix=BlazeOp.child_prefix(prefix, "routed"),
        use_silu=use_silu,
        eps=eps,
        routing_scaling_factor=routing_scaling_factor,
        pop_router_act=False,     # Don't pop -- shared expert needs it too
        index_offset=index_offset,
        fp32_dest_acc_en=routed_fp32_dest_acc_en,
        fused_activation=routed_fused_activation,
        # ... subblock_k, wait_for_out flags
        pop_index=True,
        pop_swiglu_act=False,
    )
```

**Mesh-aware index offset**: The `index_offset` computation uses `f.mesh_coord` and `f.mesh_shape` to derive a per-device expert index. Device (0,0) in a 2x4 mesh gets offset 0; device (1,3) gets offset 7. This ensures each device routes to its assigned expert subset.

**RoutedExpert internals** (from `blaze/ops/routed_expert/op.py`): RoutedExpert is itself a FusedOp that chains:

1. **MoERouter.emit()** -- gate matmul + sigmoid + gather + DeepSeek MoE gate. Returns `(scores, indices)` as two CBHandles (via `multi_output()`).
2. **Mcast (indices)** -- multicast the top-K expert indices to all cores.
3. **Mcast (scores)** -- multicast the routing scores to all cores.
4. **SwigluOp.emit()** -- DRAM-streaming matmul (up) + DRAM-streaming matmul (gate) + eltwise mul + gather + mcast + DRAM-streaming matmul (down). Uses the mcast'd indices and scores for per-token expert selection and scaling.

The `pop_router_act=False` is critical: the shared expert branch also reads from `act_mcast_handle`, so the mcast destination CB must not be popped by the routed branch's router.

### Phase 4: Shared Expert Branch

```python
    if enable_shared_expert:
        shared_output = create_act_gather_dst(
            f, prefix=prefix,
            num_pages=act_mcast_pages, tile_info=act_ti,
        )
```

#### The create_act_gather_dst() Helper

This helper allocates a scratch CB visible to both matmul cores and the sender core:

```python
# blaze/ops/moe/op.py

def create_act_gather_dst(f, *, prefix, num_pages, tile_info):
    """Allocate CB-backed gather sink visible to sender and matmul cores."""
    dst_ranges = ttnn.CoreRangeSet(
        list(f.matmul_cores.ranges()) + list(f.sender_grid.ranges())
    )
    return f.cb_scratch(
        name=BlazeOp.cb_name(prefix, "gather_dst"),
        num_pages=num_pages,
        core_ranges=dst_ranges,
        data_format=tile_info.data_format,
        tile=tile_info.desc,
        page_size=tile_info.size,
    )
```

The union grid (`matmul_cores + sender_grid`) ensures the gather destination is accessible to both the gather's receiver address lookup and the downstream DownProj's matmul cores.

#### SharedExpert Invocation

```python
        is_reduce_root = (row, col) == tuple(red2one_root_mesh_coord)
        shared_out = SharedExpert.emit(
            f,
            act_mcast_handle,        # <-- same handle as routed branch
            shared_gate_up_weight,
            shared_down_weight,
            _shared_bias,
            shared_output,            # <-- scratch CB, not the final output
            prefix=BlazeOp.child_prefix(prefix, "shared"),
            ab_coords=shared_ab_grids,
            pop_shared_act=False,     # Don't pop -- routed branch may still need it
            bias_dst_tile_info=_bias_dst_ti,
            bias_dst_num_pages=_bias_dst_pages,
            skip_bias_add=not is_reduce_root,
        )
```

Key design decisions:

1. **`shared_output` is a scratch CB**, not the final output tensor. The SharedExpert's result goes through another Mcast and ResidualAdd before reaching the output.

2. **`skip_bias_add=not is_reduce_root`**: On non-root devices, the bias/residual addition is skipped. Only the reduction root device adds the residual, preventing double-counting across the mesh.

3. **`pop_shared_act=False`**: The activation mcast handle is preserved so both branches can read it without conflict.

### Phase 5: Merging Branches

```python
        shared_for_add = Mcast.emit(
            f,
            shared_out,
            prefix=BlazeOp.child_prefix(prefix, "shared_output_mcast"),
        )

        reduce_input = ResidualAdd.emit(
            f,
            routed_out,
            shared_for_add,
            prefix=BlazeOp.child_prefix(prefix, "merge"),
            cores=routed_out.core_ranges,
        )
```

The shared expert output is multicast to all cores (it lives only on the sender/gather core after the SharedExpert's internal gather). Then `ResidualAdd` element-wise adds the routed and shared expert outputs on the routed expert's core ranges. The result is a single `CBHandle` ready for cross-device reduction.

When `enable_shared_expert=False`, the merge is skipped and `reduce_input = routed_out` directly.

### Phase 6: Cross-Device Reduction

```python
    return ReduceToOne.emit(
        f,
        input=reduce_input,
        intermediate=red2one_intermediate,
        output=red2one_output,
        prefix=BlazeOp.child_prefix(prefix, "reduce_to_one"),
        root_mesh_coord=red2one_root_mesh_coord,
        is_torus=red2one_is_torus,
        sender_cores=ttnn.corerange_to_cores(f.sender_grid),
    )
```

`ReduceToOne` performs a tree-reduction across the mesh:
- **Leaf devices**: send their local result to the root (or an intermediate root).
- **Intermediate roots** (ROOT2, ROOT3): receive, accumulate, forward.
- **Root device**: receives from all stages, writes the final result into `red2one_output`.

The `intermediate` and `output` tensors are pre-allocated by the caller (typically the pipeline builder) and passed through as external tensors. The topology (number of stages, receive counts per device) is computed from `f.mesh_coord`, `f.mesh_shape`, and `root_mesh_coord` using the `get_reduction_topology()` helper in `blaze/ops/reduce_to_one/op.py`.

---

## CB Reconfiguration for Routed Experts

The routed expert branch (via SwigluOp) involves DRAM-streaming matmuls that use different CB configurations across their gate, up, and down phases. When the total number of logical CBs exceeds 64, or when the gate-up and down matmuls need incompatible formats at the same CB slot, CB reconfiguration is triggered.

The SwigluOp internally marks reconfig boundaries between the gate-up and down phases:

```
[gate_up DRAM matmul] -- reconfig boundary --> [down DRAM matmul]
```

At this boundary:
1. The gate-up phase's scratch CBs are freed.
2. New CB IDs are assigned for the down phase (reusing matching formats).
3. A reconfig tensor is built and the `CbReconfig` node patched with the real L1 address.

The routed expert's `fp32_dest_acc_en` flag also affects the `ComputeConfigDescriptor`. When enabled, the TRISC unpacker writes to FP32 accumulators, which changes tile alignment requirements. The reconfig boundary allows the down phase to use different unpack-to-dest modes than the gate-up phase.

---

## Full Prefix Tree

For a MoE with prefix `"moe"`:

```
moe__rmsnorm                                # RMSNorm on sender
moe__act_mcast                              # Activation mcast to all cores
moe__routed                                 # RoutedExpert
  moe__routed__router                       #   MoERouter
    moe__routed__router__gate_mm            #     Gate matmul
    moe__routed__router__gate_gather        #     Gate gather
    moe__routed__router__moe_gate           #     DeepSeek MoE gate
  moe__routed__index_mcast                  #   Index mcast
  moe__routed__score_mcast                  #   Score mcast
  moe__routed__swiglu                       #   SwigluOp
    moe__routed__swiglu__up_mm              #     Up DRAM matmul
    moe__routed__swiglu__gate_mm            #     Gate DRAM matmul
    moe__routed__swiglu__eltwise            #     Eltwise mul
    moe__routed__swiglu__gather             #     Gather
    moe__routed__swiglu__mcast              #     Mcast
    moe__routed__swiglu__down_mm            #     Down DRAM matmul
moe__shared                                 # SharedExpert
  moe__shared__gu                           #   KNMatmul (gate+up)
  moe__shared__gated_reduce                 #   GatedReduce
  moe__shared__mcast                        #   Intermediate mcast
  moe__shared__down_proj                    #   DownProj
    moe__shared__down_proj__matmul          #     Down matmul
    moe__shared__down_proj__mcast2          #     Bias mcast
    moe__shared__down_proj__residual_add    #     Residual add
    moe__shared__down_proj__gather          #     Gather
moe__shared_output_mcast                    # Shared output mcast
moe__merge                                  # ResidualAdd (merge branches)
moe__reduce_to_one                          # Cross-device reduction
```

Every leaf in this tree is a MicroOp with its own CT args, per-core flags, and CB allocations -- all namespaced to prevent collisions.

---

## Data Flow Summary

```
act -----> [RMSNorm] -> [Mcast] ------+-----> [RoutedExpert] ---> routed_out
                                       |                              |
                                       +-----> [SharedExpert] ---+    |
                                                                 |    |
                                                          [Mcast]+    |
                                                                 |    |
                                                         [ResidualAdd]+
                                                                 |
                                                          [ReduceToOne]
                                                                 |
                                                              output
```

The single `act_mcast_handle` fans out to both branches. The `ResidualAdd` collapses them. `ReduceToOne` handles the cross-device All-Reduce-to-root pattern.

---

## Key Takeaways

1. **Dual-branch composition**: Two FusedOps (RoutedExpert, SharedExpert) read the same mcast handle. The `pop_*=False` flags ensure neither branch invalidates the other's input.

2. **Mesh-aware logic**: `f.mesh_coord` drives per-device index offsets and root detection. The composition logic is identical on every device; only the CT-arg values differ.

3. **create_act_gather_dst()**: A reusable pattern for allocating union-grid scratch CBs that serve both gather receivers and downstream matmul consumers.

4. **Conditional composition**: `enable_shared_expert` and `skip_bias_add` demonstrate how composition kwargs control which phases are active and how they behave, without modifying the underlying MicroOps.

5. **CB reconfiguration**: The routed expert's DRAM-streaming matmuls may trigger CB reconfiguration across gate-up and down phases, managed transparently by the `cb_reconfig_builder` (see [Ch. 7.02](02_advanced_patterns.md)).

6. **Scale**: 275 lines of Python produce a fused kernel with approximately 20 phases, 40+ CB allocations, 100+ CT args, and cross-device fabric coordination -- all validated for name uniqueness, CB format consistency, and scratch-boundary compliance at compile time.

[<-- Worked Example -- SharedExpert](03_worked_example_shared_expert.md)
