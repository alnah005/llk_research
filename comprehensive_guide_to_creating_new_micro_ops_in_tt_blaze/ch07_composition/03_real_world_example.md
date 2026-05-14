# 7.3 Real-World Example: SharedExpert Annotated Walkthrough

## What SharedExpert Does

`SharedExpert` implements the shared-expert branch of a Mixture-of-Experts (MoE) transformer layer. In models like DeepSeek V3, the MoE layer has two branches: a set of routed experts selected per-token, and a single shared expert applied to every token. `SharedExpert` handles the shared branch.

The mathematical operation is:

```
output = W_down * ( SiLU(W_gate * x) * (W_up * x) ) + bias
```

This decomposes into four hardware stages:

1. **KNMatmul** -- Parallel gate/up matmul: compute `W_gate * x` on the A-branch cores and `W_up * x` on the B-branch cores simultaneously.
2. **GatedReduce** -- Gather the scattered matmul results, apply `SiLU(gate) * up`, reduce across K-parallel cores.
3. **Mcast** -- Multicast the reduced activation to all cores for the down-projection.
4. **DownProj** -- Down-projection matmul, bias addition, and gather to output.

`DownProj` is itself a sub-FusedOp containing four stages: matmul, bias mcast, residual add, and gather. The full pipeline spans 8+ micro-op phases, all executing in a single fused kernel without returning to the host.

**Reference file**: `blaze/ops/shared_expert/op.py`


## Class Declaration

```python
class SharedExpert(FusedOp):
    """SharedExpert: composed from KNMatmul + GatedReduce + DownProj building blocks."""

    name: str = "shared_expert"
    math_fidelity: str = "LoFi"
    math_approx_mode: bool = True

    activation: Input = Input()
    gate_up_weights: Input = Input()
    down_weights: Input = Input()
    bias: Input = Input()
    output: Output = Output()
```

Key observations:

- **`name = "shared_expert"`**: The op registry key. This is what the compiler uses to find the `FusedOpConfig` via `get_fused_op_config("shared_expert")`.

- **`math_fidelity = "LoFi"` and `math_approx_mode = True`**: The MLP uses low-fidelity math and approximation mode. All TRISC math operations in the fused kernel -- the gate/up matmul, the SiLU activation, the down-projection matmul -- all run with reduced-precision approximations. This is acceptable for inference in most transformer architectures and provides significant throughput gains.

- **Four Input ports**: `activation` (the hidden state), `gate_up_weights` (fused gate and up projection weights), `down_weights` (down projection weights), `bias` (the output bias added after the down projection).

- **One Output port**: `output` (the final result after DownProj + bias).

- **No `kernel` attribute**: SharedExpert relies on auto-generated kernel code. The codegen system stitches the micro-op phases into a single kernel file based on the shadow graph recorded during composition.

- **No `ct_args`, `phase`, or `op_class`**: Composition handles everything through the child micro-ops.


## compose(): The Compiler Entry Point

```python
@classmethod
def compose(cls, f, tensors, output, user_args):
    cls.emit(
        f,
        tensors["activation"],
        tensors["gate_up_weights"],
        tensors["down_weights"],
        tensors["bias"],
        output,
        prefix=user_args.get("prefix", "shared_expert"),
        ab_coords=user_args.get("ab_grids"),
        down_coords=user_args.get("down_coords"),
        pop_shared_act=user_args.get("pop_shared_act", True),
    )
```

This is a thin delegation layer. It:

1. Extracts the four input tensors from the `tensors` dictionary by port name.
2. Passes the pre-allocated `output` tensor through.
3. Extracts configuration from `user_args` with defaults:
   - `prefix`: defaults to `"shared_expert"`. Can be overridden when SharedExpert is nested inside a larger FusedOp (e.g., MoE passes `"moe__shared_expert"`).
   - `ab_grids`: the A/B core coordinate pairs for the gate/up matmul. `None` means auto-derive from the device grid.
   - `down_coords`: core coordinates for the down-projection matmul. `None` means use the default matmul grid.
   - `pop_shared_act`: whether the KNMatmul should pop (free) the activation CB after reading. Defaults to `True`, but `DenseMLP` passes `False` because the activation is needed later for the residual add.


## emit() Walkthrough

### Method Signature

```python
@staticmethod
def emit(
    f,
    activation,
    gate_up_weights,
    down_weights,
    bias,
    output,
    *,
    prefix: str = "shared_expert",
    ab_coords=None,
    down_coords=None,
    pop_shared_act=True,
    bias_dst_tile_info=None,
    bias_dst_num_pages=None,
    skip_bias_add: bool = False,
):
```

The keyword-only arguments (after `*`) provide composition configuration:

| Parameter | Purpose |
|-----------|---------|
| `prefix` | Namespace for all CT args and CB names |
| `ab_coords` | Optional `(a_coords, b_coords)` for KNMatmul core split |
| `down_coords` | Optional core grid for DownProj matmul |
| `pop_shared_act` | Whether to pop the activation CB after KNMatmul reads it |
| `bias_dst_tile_info` | Optional tile format override for bias mcast destination |
| `bias_dst_num_pages` | Optional page count override for bias mcast destination |
| `skip_bias_add` | Whether to skip bias addition in ResidualAdd |


### TileInfo Extraction

```python
if isinstance(activation, ttnn.Tensor):
    ti = TileInfo.from_tensor(activation)
else:
    tile = ttnn.Tile([activation.tile_desc.height, activation.tile_desc.width])
    ti = TileInfo(
        tile=tile,
        data_format=activation.data_format,
        size=activation.page_size,
        desc=activation.tile_desc,
    )
```

The first block extracts tile metadata from the activation input. Two cases:

- **`ttnn.Tensor`**: Called from `compose()` or as a standalone top-level call. `TileInfo.from_tensor()` reads the tile shape, data format, and page size directly from the tensor.
- **`CBHandle`**: Called from another FusedOp's `emit()` (e.g., `DenseMLP.emit()`). The metadata is extracted from the handle's `tile_desc`, `data_format`, and `page_size` fields.

The `TileInfo` object (`ti`) is threaded into downstream stages that need to know the activation tile geometry. `GatedReduce` uses it for scratch CB allocation and face-view optimization decisions. `Mcast` uses it for destination format overrides so the down-projection receives tiles in the correct geometry.


### Stage 1: KNMatmul (Gate/Up Matmul)

```python
gu = KNMatmul.emit(
    f,
    activation,
    gate_up_weights,
    prefix=BlazeOp.child_prefix(prefix, "gu"),
    ab_coords=ab_coords,
    k_half_split=False,
    pop_act=pop_shared_act,
)
```

KNMatmul splits the Tensix grid into two halves (A and B cores). The A-cores compute the gate projection, the B-cores compute the up projection. Both read the same activation but multiply by different weight slices.

**Parameters**:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `f` | The `FusedProgram` | All state accumulates here |
| `activation` | `ttnn.Tensor` or `CBHandle` | The input activations |
| `gate_up_weights` | `ttnn.Tensor` | The concatenated gate+up weight matrix |
| `prefix` | `BlazeOp.child_prefix(prefix, "gu")` | e.g., `"shared_expert__gu"` |
| `ab_coords` | `None` or `(a_coords, b_coords)` | Core assignment for A/B branches |
| `k_half_split` | `False` | Standard gate/up mode (not K-split) |
| `pop_act` | `pop_shared_act` | Whether to pop the activation CB after reading |

**What happens inside `KNMatmul.emit()`**:

1. **Input handling**: If `activation` is a `ttnn.Tensor`, it is registered as a CB via `f.cb_from_tensor()`. If it is already a `CBHandle`, it is validated via `require_fifo_handle()`:
   ```python
   if isinstance(act, CBHandle):
       act_handle = require_fifo_handle(act, "KNMatmul.act")
   else:
       act_handle = f.cb_from_tensor(act)
   ```

2. **ABGrid construction**: If `ab_coords` is provided, it builds an `ABGrid.from_coords()`. Otherwise, it uses `f.build_ab_grids()` and `_derive_kn_parallel()` to derive the partitioning from tile geometry and available cores.

3. **KN parallel derivation**: Computes `k_parallel`, `n_parallel`, and `num_per_branch` from the tile shape and available cores. For example, with 1x32 tiles and 64 cores per branch, `n_parallel` might be 8 and `k_parallel` might be 8. When explicit `ab_coords` are provided, `_derive_kn_parallel_from_weights()` derives the values from the weight shard geometry instead.

4. **Weight CB allocation**: `resolve_weight_direct_address()` creates a direct-address CB for the weight tensor and returns the L1 address.

5. **Output scratch CB**: `f.cb_scratch()` allocates a scratch CB for the matmul output on the compute grid:
   ```python
   gu_out = f.cb_scratch(
       name=BlazeOp.cb_name(prefix, "gu_out"),
       num_pages=1,
       core_ranges=ab.compute_grid,
       data_format=data_format,
       tile=input_tile_desc,
       page_size=input_tile_size,
   )
   ```
   CB name: `"shared_expert__gu___gu_out"`

6. **Per-core role flags**: Gate compute cores and up compute cores are flagged. Note that these flags are deliberately unprefixed because they represent a global core-role assignment referenced by both `KNMatmul` and `GatedReduce`:
   ```python
   f.per_core_unified_ct_args([
       f.flag("is_gate_compute_core", ab.a_grid),
       f.flag("is_up_compute_core", ab.b_grid),
   ])
   f.per_core_unified_ct_args([
       f.flag(f"{prefix}.is_active", ab.compute_grid),
       f.flag(f"{prefix}.pop_act", ab.compute_grid, pop_act),
   ])
   ```

7. **Unified CT args** for the activation:
   ```python
   f.unified_ct_args([
       (f"{prefix}.act", act_handle),
       (f"{prefix}.act_total_tiles", K_gate_tiles),
   ])
   ```
   Here, `(f"{prefix}.act", act_handle)` passes the `CBHandle` directly. The `FusedProgram.unified_ct_args()` method calls `int(act_handle)` to extract the CB ID and records the value for all three RISCs.

8. **RISC-specific CT args**:
   ```python
   f.ncrisc_ct_args([
       (f"{prefix}.act_is_tensor_backed", int(act_is_tensor_backed)),
   ])
   f.trisc_ct_args([
       (f"{prefix}.weights", gu_weights_cb),
       (f"{prefix}.out", gu_out),
       (f"{prefix}.k_per_core", k_per_core),
   ])
   f.trisc_rt_args([
       (f"{prefix}.weights_l1_address", weights_l1_address),
   ])
   ```

9. **Per-core sender indices**: Set for downstream gather. These use the bare `"ag"` and `"bg"` prefixes (not scoped under the KNMatmul prefix) because `GatedReduce` references them as a shared coordination namespace:
   ```python
   f.per_core_unified_ct_args([
       ("ag.sender_idx", a_indices),
       ("bg.sender_idx", b_indices),
       (f"{prefix}.k_offset", { ... }),
   ])
   ```

10. **Shadow graph recording**: Two `f.output()` calls record the gate and up branches as separate graph nodes:
    ```python
    gate_handle = f.output(
        "kn_sliced_matmul", gu_out,
        core_ranges=ab.a_grid, grid=ab.a_grid,
        prefix=f"{prefix}.gate",
        act=act_handle,
        weights=ExternalTensor(name=f"{prefix}.gate_weights"),
        branch="gate", role_name="gate_compute",
    )
    up_handle = f.output(
        "kn_sliced_matmul", gu_out,
        core_ranges=ab.b_grid, grid=ab.b_grid,
        prefix=f"{prefix}.up",
        act=act_handle,
        weights=ExternalTensor(name=f"{prefix}.up_weights"),
        branch="up", role_name="up_compute",
    )
    ```

**Return value**: A `KNMatmulOutput` dataclass:

```python
@dataclass
class KNMatmulOutput:
    handle: CBHandle        # the output CB (gu_out)
    k_parallel: int         # number of K-parallel cores per branch
    n_parallel: int         # number of N-parallel cores per branch
    num_per_branch: int     # total cores per branch
    a_grid: object          # CoreRangeSet for gate branch
    b_grid: object          # CoreRangeSet for up branch
    gate_handle: CBHandle | None  # shadow graph handle for gate
    up_handle: CBHandle | None    # shadow graph handle for up
```

This structured return is why `KNMatmulOutput` exists rather than a plain `CBHandle`. A plain `CBHandle` would not carry enough metadata for `GatedReduce` to know how to partition its gather and reduction. The `k_parallel`, `n_parallel`, `a_grid`, and `b_grid` fields encode the partitioning that `GatedReduce` needs.


### Stage 2: GatedReduce (SiLU(gate) * up with Reduction)

```python
reduced = GatedReduce.emit(
    f,
    gu.handle,
    prefix=BlazeOp.child_prefix(prefix, "gated_reduce"),
    k_parallel=gu.k_parallel,
    n_parallel=gu.n_parallel,
    tile_info=ti,
    a_grid=gu.a_grid,
    b_grid=gu.b_grid,
    gate_handle=gu.gate_handle,
    up_handle=gu.up_handle,
)
```

`GatedReduce` receives the `KNMatmulOutput` fields unpacked into explicit parameters:

| Parameter | Source | Meaning |
|-----------|--------|---------|
| `gu.handle` | The `gu_out` scratch CB | Contains scattered matmul results across A+B cores |
| `gu.k_parallel` | Derived in KNMatmul | Number of K-reduction steps |
| `gu.n_parallel` | Derived in KNMatmul | Number of N-parallel tiles per K-group |
| `ti` | From activation tile info | Controls scratch CB formats |
| `gu.a_grid` | Gate branch core grid | Tells GatedReduce where gate results live |
| `gu.b_grid` | Up branch core grid | Tells GatedReduce where up results live |
| `gu.gate_handle` | Shadow graph handle | For edge recording |
| `gu.up_handle` | Shadow graph handle | For edge recording |

**What happens inside `GatedReduce.emit()`**:

1. **Compute configuration derivation**: The number of compute cores per branch is derived from the input handle's core ranges (halved, since the compute grid spans both A and B):
   ```python
   num_compute_cores = len(ttnn.corerange_to_cores(gu_out_handle.core_ranges)) // 2
   tiles_per_k = k_parallel
   k_num_tiles = n_parallel
   ```

2. **Face-view optimization**: When tile dimensions allow it (e.g., 1x32 tiles with 8 K-parallel), the reduce operates on packed 16x16 face tiles for higher throughput:
   ```python
   use_face_view = can_use_face_view(
       input_tile.tile_shape[0], input_tile.tile_shape[1],
       tiles_per_k, k_num_tiles
   )
   ```

3. **Gather destination CBs**: Two scratch CBs (`group1` and `group2`) are allocated on the union of the compute grid and sender core. The receiver needs the CB on sender cores for `receiver_addr_from_dst_cb`, and senders need it on their cores:
   ```python
   gather_dst_ranges = ttnn.CoreRangeSet(
       list(gu_out_handle.core_ranges.ranges()) + list(f.sender_grid.ranges())
   )
   group1 = f.cb_scratch(
       name=BlazeOp.cb_name(prefix, "group1"),
       num_pages=group_num_pages,
       core_ranges=gather_dst_ranges,
       ...
   )
   group2 = f.cb_scratch(
       name=BlazeOp.cb_name(prefix, "group2"),
       num_pages=group_num_pages,
       core_ranges=gather_dst_ranges,
       ...
   )
   ```
   CB names: `"shared_expert__gated_reduce___group1"` and `"shared_expert__gated_reduce___group2"`

4. **Intermediate and output CBs**: `intermed` (2-page scratch for the reduction accumulator) and `mcast_src` (the output ready for downstream multicast) are allocated on the sender core only:
   ```python
   intermed = f.cb_scratch(
       name=BlazeOp.cb_name(prefix, "intermed"),
       num_pages=2,
       core_ranges=f.sender_grid,
       ...
   )
   mcast_src = f.cb_scratch(
       name=BlazeOp.cb_name(prefix, "mcast_src"),
       num_pages=mcast_src_num_pages,
       core_ranges=f.sender_grid,
       ...
   )
   ```

5. **Semaphore allocation**: Four semaphores for the two gather operations (one NOC0 and one NOC1 semaphore per gather):
   ```python
   ag_sem = f.semaphore(f"{ag}.sem")
   bg_sem = f.semaphore(f"{bg}.sem")
   ag_noc1_sem = f.semaphore(f"{ag}.noc1_sem")
   bg_noc1_sem = f.semaphore(f"{bg}.noc1_sem")
   ```
   Note that `ag` and `bg` are hardcoded to `"ag"` and `"bg"` within `GatedReduce.emit()`. These are NOT prefixed with the `gated_reduce` child prefix because they represent a cross-cutting concern: the gather sender indices (`ag.sender_idx`, `bg.sender_idx`) are set in `KNMatmul.emit()` and must match the gather receiver args set here. The shared unprefixed namespace is the coordination point between these two ops.

6. **NCRISC CT args**: Gather destination addresses, data sizes, semaphores, and source/destination CBs for both gate and up gathers:
   ```python
   f.ncrisc_ct_args([
       (f"{ag}.dest_noc_x", f.noc_sender.x),
       (f"{ag}.dest_noc_y", f.noc_sender.y),
       (f"{ag}.data_size_bytes", input_tile_size),
       (f"{ag}.noc0_receiver_semaphore", ag_sem),
       (f"{ag}.src", gu_out_cb),
       (f"{ag}.src_num_pages", 1),
       ...
       (f"{ag}.receiver_addr_from_dst_cb", True),
       (f"{ag}.dst", group1),
       # ... same pattern for bg ...
   ])
   ```

7. **BRISC CT args**: Receiver-side control for both gathers (number of senders, semaphore addresses, destination CB and page count):
   ```python
   f.brisc_ct_args([
       (f"{ag}.noc0_num_senders", num_compute_cores),
       (f"{ag}.noc1_num_senders", 0),
       (f"{ag}.noc0_receiver_semaphore", ag_sem),
       (f"{ag}.noc1_receiver_semaphore", ag_noc1_sem),
       (f"{ag}.dst", group1),
       (f"{ag}.dst_num_pages", ...),
       # ... same for bg ...
   ])
   ```

8. **Per-core role flags**: Gate compute cores are senders for `ag` gather; up compute cores are senders for `bg` gather. The sender core is the receiver for both:
   ```python
   f.per_core_unified_ct_args([
       f.flag(f"{ag}.is_sender", a_grid),
       f.flag(f"{ag}.pop_src", a_grid),
       f.flag(f"{ag}.use_per_core", a_grid),
   ])
   # ... same for bg with b_grid ...
   f.per_core_unified_ct_args([
       f.flag(f"{ag}.is_receiver", f.sender_grid),
       f.flag(f"{bg}.is_receiver", f.sender_grid),
   ])
   f.per_core_unified_ct_args([f.flag(f"{gr}.is_active", f.sender_grid)])
   ```

9. **TRISC CT args** for the gated_reduce compute:
   ```python
   f.trisc_ct_args([
       (f"{gr}.group1", group1),
       (f"{gr}.group2", group2),
       (f"{gr}.intermed", intermed),
       (f"{gr}.out", mcast_src),
       (f"{gr}.tiles_per_k", kernel_tiles_per_k),
       (f"{gr}.k_num_tiles", kernel_k_num_tiles),
       (f"{gr}.use_gelu", 1 if activation == "gelu" else 0),
   ])
   ```

10. **Shadow graph recording**: Three `f.output()` calls -- one for each gather and one for the gated_reduce itself:
    ```python
    ag_handle = f.output("gather", group1, ..., prefix=ag,
                         src=gate_handle if gate_handle is not None else gu_out_handle)
    bg_handle = f.output("gather", group2, ..., prefix=bg,
                         src=up_handle if up_handle is not None else gu_out_handle)
    return f.output("gated_reduce", mcast_src, grid=f.sender_grid,
                    prefix=gr, group1=ag_handle, group2=bg_handle)
    ```

**Return value**: A `CBHandle` pointing to the `mcast_src` scratch CB on the sender core, containing the reduced `SiLU(gate) * up` result.


### Stage 3: Mcast (Broadcast Reduced Activation)

```python
down_act = Mcast.emit(
    f,
    reduced,
    prefix=BlazeOp.child_prefix(prefix, "mcast"),
    sender_sem=None,
    dst_num_pages=gu.n_parallel,
    dst_tile_info=ti,
)
```

The `reduced` CBHandle from `GatedReduce` lives on the sender core only. `Mcast` broadcasts it to all cores so the down-projection matmul can proceed on the full compute grid.

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `reduced` | CBHandle on sender core | The `SiLU(gate) * up` result |
| `prefix` | `"shared_expert__mcast"` | Namespace for this multicast |
| `sender_sem` | `None` | Allocate a fresh semaphore |
| `dst_num_pages` | `gu.n_parallel` | Override destination page count to match matmul width |
| `dst_tile_info` | `ti` | Override destination tile format (potentially converting from face-view back to full tiles) |

**What happens inside `Mcast.emit()`**:

1. **Source CB**: Since `reduced` is a `CBHandle` (not a tensor), it is used directly. No new input CB is allocated:
   ```python
   if isinstance(src, CBHandle):
       src_handle = src
   else:
       src_handle = f.cb_from_tensor(src)
   ```

2. **Destination CB**: `f.cb_scratch()` allocates `dst` on `f.all_cores` with `balanced=False`. The imbalance flag reflects that every receiver core gets a push, but only the downstream matmul cores will pop:
   ```python
   dst = f.cb_scratch(
       name=BlazeOp.cb_name(prefix, "dst"),
       num_pages=_dst_num_pages,
       core_ranges=f.all_cores,
       ...
       balanced=False,
   )
   ```
   CB name: `"shared_expert__mcast___dst"`

3. **Semaphores**: Since `sender_sem=None` was passed, the mcast allocates its own sender and receiver semaphores:
   ```python
   sender_sem = f.semaphore(f"{prefix}.sender")
   receiver_sem = f.semaphore(f"{prefix}.receiver")
   ```
   Semaphore names: `"shared_expert__mcast.sender"` and `"shared_expert__mcast.receiver"`

4. **Per-core flags**: Sender, receiver, pop, and initialization flags:
   ```python
   f.per_core_unified_ct_args([
       f.flag(f"{prefix}.is_sender", f.sender_grid),
       f.flag(f"{prefix}.is_receiver", f.mcast_receiver_grid),
       f.flag(f"{prefix}.pop_src", f.all_cores),
       f.flag(f"{prefix}.init_src", f.sender_grid, needs_init),
   ])
   ```
   `needs_init` is `False` because the source is a `CBHandle` (not a tensor that needs initial CB setup).

5. **NCRISC and BRISC CT args**: Source CB, page counts, receiver semaphore, destination CB and page count (NCRISC). NOC coordinates for multicast range, core count, both semaphores, data sizes, linking/posting flags (BRISC).

6. **Shadow graph**: The return is a `CBHandle` via `f.output("mcast", dst, prefix=prefix, src=src)`.

**Return value**: A `CBHandle` to `dst`, now available on all cores.


### Stage 4: DownProj (Sub-FusedOp)

```python
return DownProj.emit(
    f,
    down_act,
    down_weights,
    bias,
    output,
    prefix=BlazeOp.child_prefix(prefix, "down_proj"),
    mcast_sender_sem=None,
    core_coords=down_coords,
    bias_dst_tile_info=bias_dst_tile_info,
    bias_dst_num_pages=bias_dst_num_pages,
    skip_bias_add=skip_bias_add,
)
```

This is where sub-FusedOp embedding happens. `DownProj` is itself a `FusedOp` with its own `emit()` that chains four micro-ops. The prefix `"shared_expert__down_proj"` scopes all of DownProj's internal state.

Note that `SharedExpert` calls `DownProj.emit()` -- **not** `DownProj.compose()`. The `compose()` method of `DownProj` would prepend an unnecessary `Mcast` stage (the activation is already multicast by SharedExpert's stage 3).


## DownProj: The Sub-FusedOp Pipeline

### DownProj Class and Standalone compose()

From `blaze/ops/down_proj/op.py`:

```python
class DownProj(FusedOp):
    """DownProj: chains Matmul -> Mcast(bias) -> ResidualAdd -> Gather."""

    name: str = "down_proj"
    math_fidelity: str = "LoFi"

    input: Input = Input()
    weights: Input = Input()
    add_input: Input = Input()
    output: Output = Output()

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        act = Mcast.emit(f, tensors["input"], prefix="mcast")
        cls.emit(
            f,
            act,
            tensors["weights"],
            tensors["add_input"],
            output,
            mcast_sender_sem=0,
            core_coords=user_args.get("core_coords"),
        )
```

`DownProj` has four ports: the activation input, weights, bias/residual input, and output. When used standalone, `compose()` adds an initial Mcast. When embedded in `SharedExpert`, the Mcast has already happened (stage 3 above), so `emit()` is called directly with the pre-multicast `CBHandle`.


### DownProj.emit() Internal Pipeline

```python
@staticmethod
def emit(
    f,
    act_handle,
    weights_tensor,
    bias_tensor,
    output,
    *,
    prefix="",
    mcast_sender_sem=None,
    core_coords=None,
    bias_dst_tile_info=None,
    bias_dst_num_pages=None,
    skip_bias_add: bool = False,
):
```

#### Step 4a: Core Grid Resolution

```python
if core_coords is None:
    matmul_cores = f.matmul_cores
elif isinstance(core_coords, tuple):
    matmul_cores = coords_to_core_range_set(list(core_coords))
else:
    matmul_cores = coords_to_core_range_set(core_coords)
```

DownProj supports three ways to specify which cores run the matmul:

1. **Default** (`core_coords=None`): Uses `f.matmul_cores`, which is the device grid minus DRAM workers, phantom cores, and the sender core.
2. **Tuple of coordinate lists**: Converted via `coords_to_core_range_set()`.
3. **List of coordinates**: Also converted via `coords_to_core_range_set()`.

This flexibility allows the caller (SharedExpert or MoE) to control which subset of the grid runs the down-projection, potentially reserving other cores for different operations.

#### Step 4b: DownProj.matmul (Width-Sharded Matmul)

```python
mm = DownProj.matmul(
    f,
    act_handle,
    weights_tensor,
    prefix=BlazeOp.child_prefix(prefix, "matmul"),
    cores=matmul_cores,
)
```

`DownProj.matmul()` is a `@staticmethod` on the `DownProj` class -- a private building block, not a separate `MicroOp`. It implements a WIDTH_SHARDED matrix multiply where each core gets the full input K dimension and a width-sharded slice of the weights (N_per_core columns), producing a complete output slice `[1, N_per_core]` with no cross-core reduction needed.

Prefix: `"shared_expert__down_proj__matmul"`

Inside `DownProj.matmul()`:

```python
@staticmethod
def matmul(f, in0_handle, weights_tensor, *, prefix="matmul", cores):
    in0_handle = require_fifo_handle(in0_handle, "DownProj.matmul.in0")
    k_num_tiles = in0_handle.num_pages
```

The `require_fifo_handle()` guard ensures the input is a standard FIFO CB (not a direct-address handle). This catches a common error where someone passes a direct-address weight CB as the activation input. The K dimension is read from the input's `num_pages`.

Weight resolution:

```python
in1_cb, weights_l1_address = resolve_weight_direct_address(
    f, weights_tensor, op_name="DownProj.matmul",
)
```

Weight validation -- the matmul validates that the weight page count is a positive multiple of the K dimension:

```python
if k_num_tiles <= 0 or in1_cb.num_pages % k_num_tiles:
    raise ValueError(
        f"DownProj.matmul weight pages ({in1_cb.num_pages}) must be "
        f"a positive multiple of K tiles ({k_num_tiles})."
    )
out_w_per_core = in1_cb.num_pages // k_num_tiles
```

This derives the per-core output width from the weight layout. If the weight tensor has `K * N_per_core` pages and the activation has `K` tiles, then each core produces `N_per_core = in1_cb.num_pages / k_num_tiles` output tiles.

Output CB allocation:

```python
out = f.cb_scratch(
    name=BlazeOp.cb_name(prefix, "out"),
    num_pages=out_w_per_core,
    core_ranges=cores,
    data_format=in0_handle.data_format,
    tile=out_tile_desc,
    page_size=out_page_size,
)
```

CB name: `"shared_expert__down_proj__matmul___out"`

CT arg registration:

```python
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_active", cores),
    f.flag(f"{prefix}.pop_in0", cores),
])
f.unified_ct_args([
    (f"{prefix}.in0", in0_handle),
])
f.ncrisc_ct_args([
    (f"{prefix}.k_num_tiles", k_num_tiles),
    (f"{prefix}.in0_is_tensor_backed", 0),
])
f.trisc_ct_args([
    (f"{prefix}.in1", in1_cb),
    (f"{prefix}.out", out),
    (f"{prefix}.k_num_tiles", k_num_tiles),
    (f"{prefix}.out_w_per_core", out_w_per_core),
])
f.trisc_rt_args([
    (f"{prefix}.weights_l1_address", weights_l1_address),
])
```

Note the use of `f.unified_ct_args()` for the `in0` CB handle. This is `f.unified_ct_args()`, NOT `f.all_core_ct_args` -- the latter does not exist in the TT-Blaze API. `unified_ct_args` pushes the argument to NCRISC, BRISC, and TRISC simultaneously.

The L1 address is registered as a runtime arg because it can change between iterations (when weights are re-allocated or swapped):

```python
f.trisc_rt_args([
    (f"{prefix}.weights_l1_address", weights_l1_address),
])
```

The matmul returns a `CBHandle` via `f.output()`:

```python
return f.output(
    "matmul", out,
    prefix=prefix,
    in0=in0_handle,
    in1=weights_tensor if isinstance(weights_tensor, CBHandle)
         else ExternalTensor(name=prefix + "_weights"),
)
```

#### Step 4c: Output Tensor Resolution

```python
if isinstance(output, CBHandle):
    output_tensor = None
    gather_dst = output
else:
    output_tensor = output
    gather_dst = f.cb_from_tensor(output_tensor)
```

This handles the duality of the output parameter. When `DownProj` is the terminal op (standalone or at the end of `SharedExpert`), `output` is a `ttnn.Tensor` and `cb_from_tensor()` allocates a tensor-backed output CB. When `DownProj` is embedded in a larger pipeline that needs the result as an intermediate (e.g., inside MoE), `output` can be a `CBHandle`.

#### Step 4d: Bias Mcast

```python
res = Mcast.emit(
    f, bias_tensor,
    prefix=BlazeOp.child_prefix(prefix, "mcast2"),
    sender_sem=mcast_sender_sem,
    dst_tile_info=bias_dst_tile_info,
    dst_num_pages=bias_dst_num_pages,
)
```

The bias tensor is multicast from DRAM to all cores. The prefix is `"shared_expert__down_proj__mcast2"` -- `mcast2` distinguishes it from the activation mcast in stage 3.

The `mcast_sender_sem` parameter allows the parent to inject a pre-allocated sender semaphore. When `None` (the case in `SharedExpert`), `Mcast` allocates its own. When `DownProj.compose()` is called standalone, it passes `mcast_sender_sem=0`, meaning the compose path reuses semaphore slot 0 (the first program semaphore).

#### Step 4e: ResidualAdd (Bias Addition)

```python
added = ResidualAdd.emit(
    f, mm, res,
    prefix=BlazeOp.child_prefix(prefix, "residual_add"),
    cores=matmul_cores,
    skip_add=skip_bias_add,
)
```

Prefix: `"shared_expert__down_proj__residual_add"`

Element-wise addition of the matmul output (`mm`) and the bias (`res`). The `cores` parameter restricts the computation to `matmul_cores` -- the same set used for the matmul itself.

Inside `ResidualAdd.emit()`:

```python
f.per_core_unified_ct_args([f.flag(f"{prefix}.is_active", cores)])
f.per_core_unified_ct_args([
    (f"{prefix}.core_idx", {core: idx for idx, core in enumerate(core_list)})
])
f.trisc_ct_args([
    (f"{prefix}.in0", in0_handle),
    (f"{prefix}.in1", in1_handle),
    (f"{prefix}.out", out),
    (f"{prefix}.out_w", out_w),
    (f"{prefix}.total_in1_tiles", total_in1_tiles),
    (f"{prefix}.skip_add", int(skip_add)),
])
```

The `skip_add` flag (from `skip_bias_add` in `SharedExpert.emit()`) allows bypassing the addition entirely, useful when the bias is zero or when the caller handles addition externally.

#### Step 4f: Gather (Output Collection)

```python
return Gather.emit(
    f, added,
    output_tensor=output_tensor,
    prefix=BlazeOp.child_prefix(prefix, "gather"),
    dst_cb=gather_dst
)
```

Prefix: `"shared_expert__down_proj__gather"`

The final stage gathers results from the width-sharded matmul cores onto the sender core (or into the output tensor's sharding pattern). The `dst_cb` parameter is the pre-allocated output CB from step 4c.

Inside `Gather.emit()`, the sender cores are derived from the input handle's `core_ranges`:

```python
sender_cores = ttnn.corerange_to_cores(src_handle.core_ranges, row_wise=row_major)
num_senders = len(sender_cores)
```

Per-core sender indices and role flags are registered:

```python
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_sender", src_handle.core_ranges),
    f.flag(f"{prefix}.is_receiver", recv_grid),
    f.flag(f"{prefix}.pop_src", src_handle.core_ranges),
    f.flag(f"{prefix}.use_per_core", src_handle.core_ranges),
])
f.per_core_unified_ct_args([
    (f"{prefix}.sender_idx",
     {core: idx for idx, core in enumerate(sender_cores)}),
])
```

Two semaphores are allocated for NOC0 and NOC1 coordination:

```python
noc0_sem = f.semaphore(f"{prefix}.noc0")
noc1_sem = f.semaphore(f"{prefix}.noc1")
```


## Complete CBHandle Flow Diagram

The data flow through `SharedExpert` can be summarized as a chain of CBHandle propagation. Each arrow represents a `CBHandle` being passed from one `emit()` call's return value to the next call's input parameter:

```
activation (ttnn.Tensor or CBHandle)
    |
    v
[KNMatmul.emit]
    | allocates: gu_out scratch CB on compute grid (A+B branches)
    | gu.handle = gu_out (CBHandle on ab.compute_grid)
    | gu.gate_handle, gu.up_handle = shadow graph handles
    |
    v
[GatedReduce.emit]
    | receives: gu.handle (gu_out on A+B cores)
    | allocates: group1, group2 (gather destinations on compute+sender)
    |            intermed (2-page scratch on sender)
    |            mcast_src (output on sender)
    | gathers: A-branch results into group1, B-branch into group2
    | computes: SiLU(group1) * group2 -> mcast_src
    | returns: mcast_src (CBHandle on sender_grid)
    |
    v
[Mcast.emit]
    | receives: mcast_src (on sender core only)
    | allocates: dst scratch CB on all_cores (balanced=False)
    | multicasts: sender -> all receivers
    | returns: dst (CBHandle on all_cores)
    |
    v
[DownProj.emit]
    |
    +--[DownProj.matmul]
    |    | receives: dst (from Mcast, on all_cores)
    |    | allocates: out scratch CB on matmul_cores
    |    | computes: WIDTH_SHARDED matmul (full K * sharded N)
    |    | returns: mm (CBHandle on matmul_cores)
    |
    +--[Mcast.emit (mcast2)]
    |    | receives: bias_tensor (ttnn.Tensor from DRAM)
    |    | allocates: dst scratch CB on all_cores
    |    | multicasts: bias from sender to all cores
    |    | returns: res (CBHandle on all_cores)
    |
    +--[ResidualAdd.emit]
    |    | receives: mm (matmul output), res (bias)
    |    | allocates: out scratch CB on matmul_cores
    |    | computes: mm + res element-wise
    |    | returns: added (CBHandle on matmul_cores)
    |
    +--[Gather.emit]
         | receives: added (on matmul_cores)
         | destination: output tensor CB (on sender/output grid)
         | gathers: matmul_cores -> receiver
         | returns: final CBHandle (tensor-backed output)
```

### Key Observations on the CBHandle Chain

1. **CBHandle carries geometry**: Every handle includes `num_pages`, `page_size`, `core_ranges`, `data_format`, and `tile_desc`. Downstream ops read these fields to configure their own CBs and CT args. For example, `DownProj.matmul()` reads `in0_handle.num_pages` to get the K tile count, and `ResidualAdd.emit()` reads `in0_handle.num_pages` to set `out_w`.

2. **No global CB name lookups**: Handles are passed explicitly from producer to consumer. There is no registry of "the matmul output CB" -- the handle itself is the reference.

3. **Tensor-backed vs scratch**: The first CB (activation input) and the last CB (output) are tensor-backed (created by `f.cb_from_tensor()`). All intermediate CBs are scratch (created by `f.cb_scratch()`). This matters for the compaction system: tensor-backed CBs are never reassigned; scratch CBs can be.

4. **Core ranges narrow and widen through the pipeline**: The activation starts on the input grid, KNMatmul produces on `compute_grid` (A+B cores), GatedReduce narrows to `sender_grid` (one core), Mcast widens back to `all_cores`, and DownProj matmul narrows to `matmul_cores`.


## Resource Inventory

### Circular Buffers (~14 total)

| CB Name | Stage | Type | Core Ranges | Purpose |
|---------|-------|------|-------------|---------|
| (activation) | KNMatmul | tensor-backed | All cores | Input activation |
| (gate_up_weights) | KNMatmul | direct-address | Compute grid | Weight matrix |
| `shared_expert__gu___gu_out` | KNMatmul | scratch | Compute grid (A+B) | Gate/up matmul output |
| `shared_expert__gated_reduce___group1` | GatedReduce | scratch | Compute + sender | Gate gather destination |
| `shared_expert__gated_reduce___group2` | GatedReduce | scratch | Compute + sender | Up gather destination |
| `shared_expert__gated_reduce___intermed` | GatedReduce | scratch | Sender only | Reduction accumulator |
| `shared_expert__gated_reduce___mcast_src` | GatedReduce | scratch | Sender only | Reduced output / mcast source |
| `shared_expert__mcast___dst` | Mcast | scratch | All cores | Broadcast destination |
| (down_weights) | DownProj.matmul | direct-address | Matmul cores | Down-projection weights |
| `shared_expert__down_proj__matmul___out` | DownProj.matmul | scratch | Matmul cores | Per-core matmul output |
| (bias) | DownProj.Mcast | tensor-backed | Sender | Bias tensor |
| `shared_expert__down_proj__mcast2___dst` | DownProj.Mcast | scratch | All cores | Bias broadcast destination |
| `shared_expert__down_proj__residual_add___out` | ResidualAdd | scratch | Matmul cores | Bias-added output |
| (output) | Gather | tensor-backed | Sender or output grid | Final output |

The exact count depends on whether CBs are shared via disjoint-core optimization (when two CBs have the same `(data_format, tile_shape, page_size)` and their core grids do not overlap, they can share a CB ID) and whether face-view packing is active (which changes tile descriptors and page sizes).

### Semaphores (10 total)

| Name | Stage | Purpose |
|------|-------|---------|
| `ag.sem` | GatedReduce | Gate gather NOC0 receiver |
| `bg.sem` | GatedReduce | Up gather NOC0 receiver |
| `ag.noc1_sem` | GatedReduce | Gate gather NOC1 receiver |
| `bg.noc1_sem` | GatedReduce | Up gather NOC1 receiver |
| `shared_expert__mcast.sender` | Mcast | Activation mcast sender handshake |
| `shared_expert__mcast.receiver` | Mcast | Activation mcast receiver handshake |
| `shared_expert__down_proj__mcast2.sender` | DownProj Mcast | Bias mcast sender handshake |
| `shared_expert__down_proj__mcast2.receiver` | DownProj Mcast | Bias mcast receiver handshake |
| `shared_expert__down_proj__gather.noc0` | Gather | Result gather NOC0 sync |
| `shared_expert__down_proj__gather.noc1` | Gather | Result gather NOC1 sync |

The important pattern is that `SharedExpert.emit()` itself does NOT allocate any semaphores. All semaphore allocation happens inside the child micro-op `emit()` calls. To share a semaphore across micro-ops (e.g., reusing a sender semaphore between two mcast phases), you would allocate it in the parent `emit()` and pass it via the `sender_sem` parameter. In this case, `SharedExpert` passes `sender_sem=None` and `mcast_sender_sem=None`, so each mcast allocates its own.

### Core Utilization (~130 cores on a 13x10 Blackhole grid)

| Role | Core Count | Which Cores |
|------|------------|-------------|
| A-cores (gate matmul) | ~64 | First half of compute grid |
| B-cores (up matmul) | ~64 | Second half of compute grid |
| Sender (Mcast/GatedReduce/Gather receiver) | 1 | Typically (12, 9) |
| Phantom/idle | 1 | Typically (12, 8) |
| DRAM workers | ~8 | Device-specific |
| Matmul cores (DownProj) | ~112 | All non-DRAM, non-phantom, non-sender |

Note: On a default 13x10 Blackhole grid, `build_ab_grids()` excludes the sender core and one phantom core, leaving 128 cores for the KNMatmul A/B split. These are divided evenly: 64 A-cores (gate) and 64 B-cores (up). The exact split depends on `_derive_kn_parallel()`, which may trim cores to satisfy the `k_parallel * n_parallel` constraint.


## CT Arg Namespace Map

For a `SharedExpert` invocation with `prefix="shared_expert"`, the full set of CT arg prefixes across all stages:

| CT Arg Prefix | Source Op | RISC(s) |
|---|---|---|
| `is_gate_compute_core` | KNMatmul | unified (per-core) |
| `is_up_compute_core` | KNMatmul | unified (per-core) |
| `shared_expert__gu.is_active` | KNMatmul | unified (per-core) |
| `shared_expert__gu.pop_act` | KNMatmul | unified (per-core) |
| `shared_expert__gu.act` | KNMatmul | unified |
| `shared_expert__gu.act_total_tiles` | KNMatmul | unified |
| `shared_expert__gu.act_is_tensor_backed` | KNMatmul | ncrisc |
| `shared_expert__gu.weights` | KNMatmul | trisc |
| `shared_expert__gu.out` | KNMatmul | trisc |
| `shared_expert__gu.k_per_core` | KNMatmul | trisc |
| `shared_expert__gu.k_offset` | KNMatmul | unified (per-core) |
| `ag.sender_idx` | KNMatmul | unified (per-core) |
| `bg.sender_idx` | KNMatmul | unified (per-core) |
| `ag.is_sender`, `ag.pop_src`, `ag.use_per_core` | GatedReduce | unified (per-core) |
| `bg.is_sender`, `bg.pop_src`, `bg.use_per_core` | GatedReduce | unified (per-core) |
| `ag.is_receiver`, `bg.is_receiver` | GatedReduce | unified (per-core) |
| `ag.dest_noc_x`, `ag.dest_noc_y`, `ag.data_size_bytes`, ... | GatedReduce | ncrisc |
| `bg.dest_noc_x`, `bg.dest_noc_y`, `bg.data_size_bytes`, ... | GatedReduce | ncrisc |
| `ag.noc0_num_senders`, `ag.noc1_num_senders`, ... | GatedReduce | brisc |
| `bg.noc0_num_senders`, `bg.noc1_num_senders`, ... | GatedReduce | brisc |
| `gated_reduce.is_active` | GatedReduce | unified (per-core) |
| `gated_reduce.group1`, `gated_reduce.group2`, `gated_reduce.intermed`, ... | GatedReduce | trisc |
| `shared_expert__mcast.is_sender`, `.is_receiver`, `.pop_src`, `.init_src` | Mcast | unified (per-core) |
| `shared_expert__mcast.src`, `.src_num_pages`, `.dst`, ... | Mcast | ncrisc |
| `shared_expert__mcast.dest_noc_start_x`, ... | Mcast | brisc |
| `shared_expert__down_proj__matmul.is_active`, `.pop_in0` | DownProj.matmul | unified (per-core) |
| `shared_expert__down_proj__matmul.in0` | DownProj.matmul | unified |
| `shared_expert__down_proj__matmul.k_num_tiles`, `.in0_is_tensor_backed` | DownProj.matmul | ncrisc |
| `shared_expert__down_proj__matmul.in1`, `.out`, `.out_w_per_core` | DownProj.matmul | trisc |
| `shared_expert__down_proj__mcast2.*` | Mcast (bias) | ncrisc, brisc, unified (per-core) |
| `shared_expert__down_proj__residual_add.is_active`, `.core_idx` | ResidualAdd | unified (per-core) |
| `shared_expert__down_proj__residual_add.in0`, `.in1`, `.out`, `.out_w`, ... | ResidualAdd | trisc, ncrisc, brisc |
| `shared_expert__down_proj__gather.is_sender`, `.is_receiver`, `.sender_idx`, ... | Gather | unified (per-core), ncrisc, brisc |

Note the asymmetry: `ag.*` and `bg.*` are not prefixed with the `gated_reduce` child prefix. This is because the gather sender indices (`ag.sender_idx`, `bg.sender_idx`) are set in `KNMatmul.emit()` and must match the gather receiver args set in `GatedReduce.emit()`. The shared namespace (`ag`, `bg`) is the coordination point between these two ops.


## Shadow Graph Structure

During composition, `FusedProgram` builds a shadow `BlazeGraph` that captures the complete op topology. For `SharedExpert`, the graph contains these nodes (in topological order):

1. `kn_sliced_matmul` (gate branch, prefix `shared_expert__gu.gate`)
2. `kn_sliced_matmul` (up branch, prefix `shared_expert__gu.up`)
3. `gather` (gate gather, prefix `ag`)
4. `gather` (up gather, prefix `bg`)
5. `gated_reduce` (prefix `gated_reduce`)
6. `mcast` (prefix `shared_expert__mcast`)
7. `matmul` (prefix `shared_expert__down_proj__matmul`)
8. `mcast` (bias, prefix `shared_expert__down_proj__mcast2`)
9. `residual_add` (prefix `shared_expert__down_proj__residual_add`)
10. `gather` (prefix `shared_expert__down_proj__gather`)

Edges connect:

- `kn_sliced_matmul.gate` -> `gather.ag` (gate results to gate gather)
- `kn_sliced_matmul.up` -> `gather.bg` (up results to up gather)
- `gather.ag` -> `gated_reduce.group1` (gathered gate tiles to reduce)
- `gather.bg` -> `gated_reduce.group2` (gathered up tiles to reduce)
- `gated_reduce` -> `mcast` (reduced result to multicast)
- `mcast` -> `matmul` (multicasted activation to down projection)
- `matmul` -> `residual_add.in0` (matmul output to residual add)
- `mcast(bias)` -> `residual_add.in1` (bias to residual add)
- `residual_add` -> `gather` (added result to final gather)

This graph is used by:

- **Kernel auto-generation**: The codegen pipeline walks the graph to determine which phases to include and their ordering in the generated kernel source.
- **CB lifetime tracking**: `f._mark_cb_use(cb_id)` is called in `output()`, updating `_cb_lifetimes` for temporal compaction.
- **Visualization and debugging**: The graph can be serialized for inspection.


## DenseMLP: Higher-Level Composition

`SharedExpert` is itself embedded inside larger FusedOps. `DenseMLP` in `blaze/ops/dense_mlp/op.py` provides a higher-level composition that adds RMSNorm and multicast before `SharedExpert`:

```python
class DenseMLP(FusedOp):
    name: str = "dense_mlp"

    act: Input = Input()
    rmsnorm_gamma: Input = Input()
    gate_up_weight: Input = Input()
    down_weight: Input = Input()

    out: Output = Output()

    @staticmethod
    def emit(
        f,
        act,
        rmsnorm_gamma,
        gate_up_weight,
        down_weight,
        *,
        prefix: str,
        shared_ab_grids=None,
        shared_down_coords=None,
        rmsnorm_epsilon: float = 1e-6,
        output_tensor=None,
    ):
        act_mcast_handle, act_ti, act_mcast_pages = emit_mlp_preamble(
            f, act, rmsnorm_gamma, prefix, rmsnorm_epsilon
        )

        if output_tensor is not None:
            shared_output = output_tensor
        else:
            shared_output = create_act_gather_dst(
                f,
                prefix=prefix,
                num_pages=act_mcast_pages,
                tile_info=act_ti,
            )

        # act (pre-norm) is passed as bias for residual: output = mlp(rmsnorm(x)) + x
        return SharedExpert.emit(
            f,
            act_mcast_handle,
            gate_up_weight,
            down_weight,
            act,
            shared_output,
            ab_coords=shared_ab_grids,
            down_coords=shared_down_coords,
            pop_shared_act=False,
        )
```

Three key observations about the `DenseMLP` embedding:

1. **The `bias` argument is the pre-norm activation.** `DenseMLP` passes `act` (the original pre-normalization activation) as the `bias` parameter to `SharedExpert.emit()`. This is because the residual connection in a dense MLP is `mlp(rmsnorm(x)) + x` -- the original `x` is added back after the down-projection. The `bias` input in `SharedExpert` flows through to `DownProj`, where `ResidualAdd` adds it to the matmul output.

2. **`pop_shared_act=False`** because `DenseMLP` passes the activation as a `CBHandle` (from `emit_mlp_preamble`'s mcast), not as a raw tensor. The activation CB must not be popped after KNMatmul reads it because the original (pre-norm) activation is passed separately for the residual path.

3. **The prefix tree has a gap at the SharedExpert boundary.** When `DenseMLP` is called with `prefix="dense_mlp"`, its preamble ops nest under `dense_mlp__`, but `DenseMLP.emit()` does not pass a `prefix=` keyword to `SharedExpert.emit()`. SharedExpert therefore uses its default prefix `"shared_expert"`, breaking the nesting chain:

```
dense_mlp
  +-- dense_mlp__rmsnorm              (RMSNorm from emit_mlp_preamble)
  +-- dense_mlp__act_mcast            (Mcast from emit_mlp_preamble)
  +-- shared_expert                    (SharedExpert -- NOT nested under dense_mlp)
        +-- shared_expert__gu
        +-- shared_expert__gated_reduce
        +-- shared_expert__mcast
        +-- shared_expert__down_proj
              +-- shared_expert__down_proj__matmul
              +-- shared_expert__down_proj__mcast2
              +-- shared_expert__down_proj__residual_add
              +-- shared_expert__down_proj__gather
```

This means `DenseMLP`'s internal CB and CT arg names are split across two unrelated prefix trees: `dense_mlp__*` for the preamble ops, and `shared_expert__*` for the core pipeline. When embedded inside MoE (which does pass an explicit prefix), the SharedExpert nesting would be `"moe__shared"` instead. The `DenseMLP` module docstring confirms the total: 12 micro-ops, all compiling to a single kernel dispatch.


## Design Principles Illustrated by SharedExpert

SharedExpert exemplifies several key design principles:

1. **Composition over inheritance.** SharedExpert does not subclass KNMatmul or GatedReduce. It composes them by calling their `emit()` methods. This keeps each micro-op independent and testable.

2. **`emit()` is the building block, `compose()` is the entry point.** SharedExpert provides both. `compose()` bridges from the compiler's tensor-dict interface. `emit()` is what DenseMLP and MoE call when embedding SharedExpert.

3. **Prefix namespacing prevents collisions.** Every stage gets a unique prefix via `child_prefix()`. The prefix tree can nest to arbitrary depth without collision.

4. **CBHandles are the data-flow edges.** Each stage returns a CBHandle that the next stage consumes. The handles carry enough metadata (page count, data format, tile descriptor, core ranges) for the downstream op to configure itself.

5. **Sub-FusedOps provide encapsulation.** DownProj hides its internal four-stage pipeline behind a single `emit()` call. SharedExpert does not need to know about DownProj.matmul, ResidualAdd, or Gather -- it just calls `DownProj.emit()` and gets back a CBHandle.

6. **Core partitioning is explicit.** KNMatmul uses `ABGrid` to split the grid into gate and up branches. DownProj uses `matmul_cores` for its width-sharded matmul. Mcast uses `all_cores` for broadcast. GatedReduce uses `sender_grid` for computation. Each stage declares exactly which cores it needs.

7. **Pop control enables resource sharing.** The `pop_shared_act` parameter lets the MoE composition share the activation between shared and routed expert paths without double-reading from DRAM.

8. **Flexible input types enable embedding.** The `isinstance(activation, ttnn.Tensor)` / `isinstance(output, CBHandle)` checks allow the same `emit()` to work at the boundary (tensor from DRAM) and in the middle of a pipeline (CBHandle from prior stage). This duality is what makes a FusedOp composable.


## Debugging Guidance

When something goes wrong in a fused pipeline, the prefix hierarchy is your roadmap:

### CT Arg Name Conflicts

The error message includes the full prefixed name:

```
ValueError: Duplicate per-core CT arg: 'shared_expert__mcast.is_sender'
```

This tells you exactly which stage has the collision. The most common cause is calling an `emit()` without a unique prefix, or calling the same op twice with the same prefix.

### CB ID Exhaustion

The `cb_scratch()` allocator assigns sequential CB IDs. If you hit the 32-CB hardware limit, check which stages allocate scratch CBs and whether disjoint-core sharing can merge any of them. CBs with the same `(data_format, tile_shape, page_size)` on disjoint core grids share IDs automatically via `_try_reuse_cb_id()`.

### Shadow Graph Inspection

The `f._shadow_graph` records every `f.output()` call with its prefix, op_name, and port sources. You can inspect it to verify the data-flow edges match your expectations.

### Per-Core Flag Verification

After composition, check `f.program._per_core_unified_ct_args` to verify that each core has the correct set of active flags. A common bug is forgetting to pass the correct core range to `f.flag()`, resulting in a micro-op being active on cores where it should not run.

### Semaphore Count

Each `f.semaphore()` call allocates either a mesh-global L1 address or a program-local slot. If you run out of semaphore slots, look for semaphores that can be shared (via the `sender_sem` parameter pattern) or converted to program-local with `program_semaphore=True`.

### Core Range Mismatches

If a matmul allocates its output on `matmul_cores` but the downstream ResidualAdd expects it on `all_cores`, the kernel will read uninitialized memory on cores outside the matmul set. Always verify that the `core_ranges` in each `CBHandle` match the expectations of the consuming op.


## Checklist for Creating a New FusedOp

Based on the SharedExpert example, here is a checklist for creating a new fused op:

### 1. Define the Class

```python
class MyOp(FusedOp):
    name: str = "my_op"
    math_fidelity: str = "LoFi"  # or "HiFi4"

    input_a: Input = Input()
    input_b: Input = Input()
    output: Output = Output()
```

### 2. Implement compose()

```python
@classmethod
def compose(cls, f, tensors, output, user_args):
    cls.emit(
        f,
        tensors["input_a"],
        tensors["input_b"],
        output,
        prefix=user_args.get("prefix", "my_op"),
    )
```

### 3. Implement emit() (if reusable)

```python
@staticmethod
def emit(f, input_a, input_b, output, *, prefix="my_op"):
    # Stage 1
    phase1_out = SomeMicroOp.emit(
        f, input_a,
        prefix=BlazeOp.child_prefix(prefix, "phase1"),
    )

    # Stage 2
    phase2_out = AnotherMicroOp.emit(
        f, phase1_out, input_b,
        prefix=BlazeOp.child_prefix(prefix, "phase2"),
    )

    # Stage 3 (sub-FusedOp if needed)
    return SubFusedOp.emit(
        f, phase2_out, output,
        prefix=BlazeOp.child_prefix(prefix, "phase3"),
    )
```

### 4. Verify the Composition

- Every micro-op call has a unique prefix via `BlazeOp.child_prefix()`.
- CBHandles flow from producer to consumer without global lookups.
- Semaphores are allocated where needed (inside micro-op `emit()` calls or explicitly in the fused op for sharing).
- Core grids are consistent (e.g., matmul output core range matches the gather source core range).
- The final `emit()` call returns a `CBHandle` that the caller can use.

### 5. Register the Op

```python
# In blaze/ops/my_op/__init__.py:
from .op import MyOp
MyOp.register()
```

### 6. Test

- Run the fused op standalone via `compose()`.
- Embed it inside another fused op via `emit()`.
- Verify the shadow graph structure via `f._shadow_graph`.
- Check CB count against hardware limits (32 CB IDs max per phase).


## Summary

The total micro-op count in a `SharedExpert` execution:

| Stage | Micro-ops | Details |
|-------|-----------|---------|
| KNMatmul | 1 | Gate/up matmul (2 shadow graph nodes for gate + up) |
| GatedReduce | 3 | 2 gathers + 1 gated reduce |
| Mcast | 1 | Activation fan-out |
| DownProj.matmul | 1 | Down-projection matmul |
| DownProj.Mcast | 1 | Bias fan-out |
| DownProj.ResidualAdd | 1 | Element-wise add |
| DownProj.Gather | 1 | Result collection |
| **Total** | **9 logical ops** | **10 shadow graph nodes** (KNMatmul produces 2) |

All 9 logical ops (producing 10 shadow graph nodes) run in a single kernel dispatch. No host round-trips. No DRAM intermediates. The only DRAM accesses are the initial weight and bias reads (via direct-address CBs and tensor-backed input CBs) and the final output write (via the tensor-backed output CB in Gather).

| Mechanism | How SharedExpert Uses It |
|-----------|------------------------|
| `FusedOp` subclass | `class SharedExpert(FusedOp)` with `name`, ports, `compose()`, `emit()` |
| `compose()` delegation | Unpacks tensor dict, forwards to `emit()` with user_args |
| `emit()` as building block | Chains KNMatmul -> GatedReduce -> Mcast -> DownProj |
| `child_prefix()` | Four child prefixes: `"gu"`, `"gated_reduce"`, `"mcast"`, `"down_proj"` |
| Sub-FusedOp embedding | `DownProj.emit()` called as a single step, expands to 4 internal phases |
| CBHandle propagation | Every phase returns a handle consumed by the next phase |
| ABGrid partitioning | `ab_coords` forwarded to `KNMatmul.emit()` for gate/up split |
| TileInfo propagation | Constructed from activation, passed to GatedReduce and Mcast |
| KNMatmulOutput | Structured return carrying handle + k/n_parallel + grids + sub-handles |
| Semaphore allocation | Delegated to each micro-op (10 total, none allocated by SharedExpert itself) |
| Shadow graph | Each `f.output()` call records a node with input/output edges (10 nodes) |
| Scratch CBs | ~9 scratch CBs for intermediates, ~5 tensor-backed/direct-address for input/output |
| Role flags | Per-core flags for gate/up cores, sender, receiver, active, pop |
