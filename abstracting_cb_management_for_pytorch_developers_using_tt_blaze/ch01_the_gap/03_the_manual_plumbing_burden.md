# The Manual Plumbing Burden

The previous two sections established what PyTorch gives developers for free and what Tenstorrent hardware demands in return. This section makes the gap concrete by walking through real Blaze code -- the `Matmul.emit()` method and the `DenseMLP` fused op -- to show every manual decision a developer must make. The goal is to count the decisions precisely so we can identify which ones an abstraction layer should automate.

**What you will learn:**

- Every manual decision in a real Blaze `Matmul.emit()` call, annotated line-by-line
- The 12 categories of manual decisions required per op
- How compound ops like DenseMLP multiply this complexity across 12 micro-ops
- A side-by-side comparison table mapping PyTorch concepts to TT-Blaze equivalents

---

## Walkthrough: Matmul.emit() -- Every Manual Decision

The `Matmul` class in `blaze/ops/matmul/op.py` is the canonical single-op example. Its `emit()` method configures three CBs, computes derived quantities, and wires CT args to three RISC processors. Here is the complete method with annotations marking each manual decision:

```python
# EXISTING -- from blaze/ops/matmul/op.py (complete emit method)
@staticmethod
def emit(
    f: FusedProgram,
    in0,                    # activation tensor or CBHandle
    in1,                    # weight tensor or OverlappedView
    *,
    prefix: str = "matmul",
    pop_in0: bool = True,
) -> CBHandle:
    # -- Decision 1: Resolve input CB (tile shape, format from tensor) ----
    if isinstance(in0, CBHandle):
        in0_handle = require_fifo_handle(in0, "Matmul.in0")
        in0_is_tensor_backed = 0
    else:
        in0_handle = f.cb_from_tensor(in0)      # <-- tile_shape, data_format derived from tensor
        in0_is_tensor_backed = 1

    # -- Decision 2: Extract tile count (K dimension) ---------------------
    k_num_tiles = in0_handle.num_pages           # developer must know this is the K dimension

    # -- Decision 3: Page size and format propagation ---------------------
    page_size = in0_handle.page_size
    data_format = in0_handle.data_format
    tile_desc = in0_handle.tile_desc

    # -- Decision 4: Weight CB with direct addressing ---------------------
    in1_cb, weights_l1_address = resolve_weight_direct_address(
        f, in1, op_name="Matmul",
    )

    # -- Decision 5: Core placement (from weight shard spec) --------------
    active_ranges = in1_cb.core_ranges

    # -- Decision 6: Validate K-dimension tiling --------------------------
    if k_num_tiles <= 0 or in1_cb.num_pages % k_num_tiles:
        raise ValueError(
            f"Matmul weight pages ({in1_cb.num_pages}) must be a "
            f"positive multiple of K tiles ({k_num_tiles})."
        )

    # -- Decision 7: Compute output width per core (blocking) ------------
    out_w_per_core = in1_cb.num_pages // k_num_tiles

    # -- Decision 8: Allocate output CB (scratch, page count, format) ----
    out = f.cb_scratch(
        name=BlazeOp.cb_name(prefix, "out"),
        num_pages=out_w_per_core,
        core_ranges=active_ranges,
        data_format=data_format,
        tile=tile_desc,
        page_size=page_size,
    )

    # -- Decision 9-10: Per-core flags (is_active, pop_in0) --------------
    f.per_core_unified_ct_args([
        f.flag(f"{prefix}.is_active", active_ranges),
        f.flag(f"{prefix}.pop_in0", active_ranges, pop_in0),
    ])
    f.unified_ct_args([
        (f"{prefix}.in0", in0_handle),
    ])

    # -- Decision 11: NCRISC CT args (tile count, tensor-backed flag) ----
    f.ncrisc_ct_args([
        (f"{prefix}.k_num_tiles", k_num_tiles),
        (f"{prefix}.in0_is_tensor_backed", in0_is_tensor_backed),
    ])

    # -- Decision 12: TRISC CT args (CB IDs, tile counts, blocking) ------
    f.trisc_ct_args([
        (f"{prefix}.in1", in1_cb),
        (f"{prefix}.out", out),
        (f"{prefix}.k_num_tiles", k_num_tiles),
        (f"{prefix}.out_w_per_core", out_w_per_core),
    ])
    f.trisc_rt_args([
        (f"{prefix}.weights_l1_address", weights_l1_address),
    ])

    return f.output("matmul", out, core_ranges=active_ranges,
                    prefix=prefix, in0=in0, in1=in1)
```

Compare this to the PyTorch equivalent:

```python
# PyTorch: the ENTIRE operation
y = x @ weight
```

One line of PyTorch becomes approximately 40 lines of Blaze composition code, with every line encoding a decision that PyTorch handles implicitly.

---

## The 12 Manual Decisions Per Op

Every TT-Blaze op requires the developer to make decisions in each of these categories:

| # | Decision Category | Matmul Example | PyTorch Equivalent |
|---|------------------|----------------|--------------------|
| 1 | **Tile shape** | Derived from input tensor tile geometry (32x32 or 1x32) | Not applicable -- no tiles |
| 2 | **Padding strategy** | Implicit -- assumes inputs are already tile-aligned | Not applicable -- any shape works |
| 3 | **Padding fill value** | Zero (for matmul correctness) | Not applicable |
| 4 | **Data format** | Propagated from `in0_handle.data_format` | Single dtype per tensor, auto-promoted |
| 5 | **num_pages** | `out_w_per_core` for output; `k_num_tiles` for input | Not applicable -- memory managed automatically |
| 6 | **page_size** | `in0_handle.page_size` (tile_h * tile_w * dtype_bytes) | Not applicable |
| 7 | **Core placement** | `active_ranges = in1_cb.core_ranges` -- which cores run this op | Not applicable -- framework decides parallelism |
| 8 | **CB ID** | Allocated by `f.cb_scratch()` or `f.cb_from_tensor()` -- must not exceed 64 per core | Not applicable |
| 9 | **Semaphore assignment** | Not needed for single-core matmul; required for Mcast, Gather | Not applicable |
| 10 | **CT arg wiring** | 8 explicit CT arg entries across NCRISC, TRISC, unified | Not applicable |
| 11 | **Broadcasting pattern** | Not needed for basic matmul; `bcast_rows`/`bcast_cols` for elementwise ops | Automatic broadcasting |
| 12 | **L1 budget check** | Implicit -- developer must ensure total CB allocation fits in ~1.5 MB | Not applicable -- OS manages memory |

Each of these decisions requires understanding of:
- The hardware's execution model (tiles, CBs, RISC roles)
- The specific op's kernel contract (which CB IDs it reads/writes)
- The data flow context (what format/shape the upstream op produced)
- The global resource state (which CB IDs are already used, how much L1 remains)

> **Warning:** Decision 12 (L1 budget) is particularly dangerous because there is no compile-time bounds check. If the combined CB allocations exceed L1 capacity (~1.5 MB per core), the kernel silently corrupts memory or crashes at runtime. The developer is responsible for mental arithmetic to ensure `sum(num_pages * page_size for all CBs on a core)` stays within budget.

---

## How Compound Ops Multiply the Complexity

### DenseMLP: 12 Micro-Ops, ~81 Decisions

A single matmul requires 12 categories of decisions. A real LLM inference layer like `DenseMLP` (from `blaze/ops/dense_mlp/op.py`) chains 12 micro-ops, each requiring its own set of decisions:

```python
# EXISTING -- from blaze/ops/dense_mlp/op.py
class DenseMLP(FusedOp):
    name: str = "dense_mlp"

    act: Input = Input()
    rmsnorm_gamma: Input = Input()
    gate_up_weight: Input = Input()
    down_weight: Input = Input()
    out: Output = Output()

    @staticmethod
    def emit(
        f, act, rmsnorm_gamma, gate_up_weight, down_weight,
        *, prefix, shared_ab_grids=None, shared_down_coords=None,
        rmsnorm_epsilon=1e-6, output_tensor=None,
    ):
        act_mcast_handle, act_ti, act_mcast_pages = emit_mlp_preamble(
            f, act, rmsnorm_gamma, prefix, rmsnorm_epsilon
        )
        # ... output tensor setup ...
        return SharedExpert.emit(
            f, act_mcast_handle, gate_up_weight, down_weight, act,
            shared_output, ab_coords=shared_ab_grids,
            down_coords=shared_down_coords, pop_shared_act=False,
        )
```

The `emit_mlp_preamble` invokes BroadcastRMSNorm and Mcast. `SharedExpert.emit` chains KNSlicedMatmul, GatedReduce, Mcast, DownProj, and more. The full execution involves approximately 12 micro-ops with this decision breakdown:

| Micro-op | CBs Allocated | CT Args | Flags | Total Decisions |
|----------|--------------|---------|-------|-----------------|
| BroadcastRMSNorm | 3-5 | ~10 | 2 | ~15 |
| Mcast (pre) | 2 | ~6 | 2 | ~10 |
| KNSlicedMatmul | 4-6 | ~12 | 4 | ~20 |
| GatedReduce | 3-4 | ~8 | 2 | ~12 |
| Mcast (post) | 2 | ~6 | 2 | ~10 |
| DownProj | 3-4 | ~10 | 2 | ~14 |
| **Total** | **~20** | **~52** | **~14** | **~81** |

For comparison, the same gated MLP in PyTorch:

```python
class SimpleMLP(torch.nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.norm = RMSNorm(hidden)
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        h = self.norm(x)
        return self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h)) + x
```

**Five lines of forward logic. Zero hardware decisions. Approximately 81 manual decisions in the Blaze equivalent.**

The `_gated_mlp.py` graph builder shows this complexity in graph API form:

```python
# EXISTING -- from blaze/_gated_mlp.py (abbreviated)
def build_gated_mlp_graph(config=None, grid_config=None, ab_grids=None):
    """Build an 11-node gated MLP dataflow graph."""
    gc = grid_config or GridConfig.default()
    sender = gc.sender_core
    all_cores = gc.get_all_cores()
    matmul_cores = gc.get_matmul_cores()
    a_cores, b_cores = gc.build_ab_grids()

    with fuse() as ctx:
        act = mcast(ExternalTensor(config.input_name),
                    grid=all_cores, sender=sender, ct_prefix="act_mcast")
        gate = kn_sliced_matmul(act, ExternalTensor(config.gate_up_weights_name),
                                grid=a_cores, ct_prefix="gu")
        up = kn_sliced_matmul(act, ExternalTensor(config.gate_up_weights_name),
                              grid=b_cores, ct_prefix="gu")
        gate_g = gather(gate, grid=a_cores, receiver=sender, ct_prefix="ag")
        up_g = gather(up, grid=b_cores, receiver=sender, ct_prefix="bg")
        red = gated_reduce(gate_g, up_g, grid=[sender],
                           activation=config.activation)
        down_act = mcast(red, grid=all_cores, sender=sender, ct_prefix="mcast")
        down = matmul(down_act, ExternalTensor(config.down_weights_name),
                      grid=matmul_cores)
        bias = mcast(ExternalTensor(config.bias_name),
                     grid=all_cores, sender=sender, ct_prefix="mcast2")
        added = residual_add(down, bias, grid=matmul_cores)
        gather(added, grid=matmul_cores, receiver=sender)

    return ctx.graph
```

11 op calls. Each one needs grid placement, CT arg prefix, and correct wiring of its inputs to the outputs of prior ops. The developer must also ensure that:

- All CBs across all 11 ops fit within each core's ~1.5 MB L1
- Data formats are compatible at each boundary (e.g., if gate matmul produces bfloat8_b but gated_reduce expects bfloat16)
- Padding is consistent through the chain
- Core placement is correct (gate on A-cores, up on B-cores, reduce on sender, mcast to all)
- CT arg names do not collide (hence the prefix system: `"gu"`, `"ag"`, `"bg"`, `"mcast"`, etc.)

### SwigluOp: An Even Larger Example

The `SwigluOp` fused op (from `blaze/ops/swiglu/op.py`) chains even more stages:

```
DRAMStreamingMatmul (up) -> DRAMStreamingMatmul (gate+SiLU) ->
EltwiseMul -> Gather -> Mcast -> DRAMStreamingMatmul (down)
```

Its `emit()` is 150+ lines and invokes six sub-op emits, each with their own CB allocations, CT args, and format handling. The total decision surface spans DRAM streaming matmul configurations, element-wise multiply with optional scalar, gather with explicit destination CB sizing, mcast with sender/receiver semaphore coordination, and down-projection with potential index-based expert selection. The complexity is not gratuitous -- each decision reflects a real hardware constraint -- but from the PyTorch developer's perspective, this is all `gate_proj * silu(up_proj) @ down_proj.T`.

### The PaddedRMSNorm Case: When Padding Adds Entire Ops

When a tensor's width is not tile-aligned, the developer cannot simply use RMSNorm -- they must use a separate `PaddedRMSNorm` op that allocates additional scratch CBs for the padded data:

```python
# EXISTING -- from blaze/ops/padded_rmsnorm/op.py (key excerpt)
class PaddedRMSNorm(MicroOp):
    name: str = "padded_rmsnorm"

    input: Input = Input()
    gamma: Input = Input()
    output: Output = Output()
    padded_input: Internal = Internal()    # Extra CB for padding!
    padded_gamma: Internal = Internal()    # Extra CB for padding!
```

The `emit()` method must:
1. Call `interpret_tile_padded(width)` to determine the padded tile geometry.
2. Allocate two extra scratch CBs (`padded_inp`, `padded_gam`) with the padded geometry.
3. Compute `input_bytes` and `padded_bytes` for the NCRISC to know how many bytes to copy vs. zero-fill.
4. Compute `sfpu_elems_per_row`, `valid_rows`, `pad_rows` for the TRISC to zero the garbage rows after squaring.
5. Wire 11 NCRISC CT args and 11 TRISC CT args, including the padding-specific parameters.

The total complexity: **5 CBs** (input, gamma, padded_input, padded_gamma, output) and **22+ CT args** for a single normalization operation. In PyTorch, RMSNorm works identically on aligned and non-aligned dimensions.

---

## Side-by-Side Comparison Table

| PyTorch Concept | TT-Blaze Equivalent | Who Decides |
|----------------|---------------------|-------------|
| `torch.Tensor` shape `[M, K]` | Tiles of `(32, 32)` with padding, `num_tiles = ceil(M/32) * ceil(K/32)` | Developer via `cb_from_tensor()` / `cb_scratch()` |
| `torch.float32` / `torch.bfloat16` | `ttnn.bfloat16` / `ttnn.bfloat8_b` / `ttnn.bfloat4_b` + `page_size` | Developer chooses per-CB |
| `y = x @ W.T` (shape inference) | `k_num_tiles`, `out_w_per_core` derived manually from input tile counts | Developer computes in `emit()` |
| `a + b` with broadcasting | Explicit `bcast_rows` or `bcast_cols` + `Mcast.emit()` for cross-core broadcast | Developer selects broadcast pattern |
| Memory allocation (`torch.empty(...)`) | `f.cb_scratch(name=..., num_pages=..., page_size=..., core_ranges=..., data_format=..., tile=...)` | Developer specifies all 6 parameters |
| No explicit buffer management | 64 CB slots per core, lifetime-tracked, compacted via `CBEngine.compact_cb_ids()` | Developer or CBEngine |
| Type promotion (bfloat16 + float32) | Developer sets `fp32_dest_acc_en=True` and passes as CT arg | Developer |
| No parallelism plumbing | `CoreRangeSet` placement + semaphore allocation + per-core flags | Developer via `emit()` |
| `nn.Module.forward()` composition | `FusedOp.compose()` + `MicroOp.emit()` chain with prefix management | Developer |
| `torch.compile()` / JIT | `blaze.fuse()` context manager + `BlazeCompiler.compile()` | Developer builds graph manually |
| `loss.backward()` (autograd) | Not applicable -- Blaze is inference-only | N/A |
| Module composition (`nn.Sequential`) | Chain `emit()` calls passing CBHandles between ops | Developer writes `compose()` |

### What the Comparison Reveals

The rightmost column is the key insight. In PyTorch, every row is either "automatic" or "framework handles it." In TT-Blaze, every row is "developer decides." The abstraction layer's job is to shift decisions from the right column to the framework, while preserving the ability for expert developers to override when needed.

---

## The Cost of Getting It Wrong

Manual plumbing is not just tedious -- errors are difficult to diagnose:

| Error Type | Symptom | Root Cause |
|-----------|---------|------------|
| Wrong tile shape | Silent numerical corruption | CB declared as 32x32 but kernel expects 1x32 |
| Wrong num_pages | Hardware hang or garbage output | CB too small for the data the kernel writes |
| Wrong page_size | Tile data misaligned in L1 | page_size does not match dtype * tile_h * tile_w |
| Wrong data_format | Unpacker produces garbage | CB format does not match the format the tensor was stored in |
| Padding error (zero where -inf needed) | Incorrect attention weights | Softmax gives non-zero weight to padding positions |
| L1 overflow | Silent corruption or hang | Total CB allocation exceeds ~1.5 MB per core |
| Wrong core_ranges | Some cores produce garbage | CB not configured on cores that the kernel runs on |
| CB ID collision | One op's data overwrites another's | Two ops sharing the same cb_id on overlapping core ranges |
| Missing CT arg | Compile error (best case) or wrong kernel behavior | A CT arg the kernel reads was not supplied |

None of these errors produce helpful error messages at the point of the mistake. The feedback loop is: write code, compile kernel, run on silicon, observe wrong numbers, debug by inspecting L1 contents. An abstraction layer that computes correct values for these parameters eliminates entire categories of bugs.

---

## Key Takeaways

- A single Blaze `Matmul.emit()` call requires 12 categories of manual decisions: tile shape, padding strategy, padding fill value, data format, num_pages, page_size, core placement, CB ID, semaphore assignment, CT arg wiring, broadcasting pattern, and L1 budget check.
- The PyTorch equivalent is `y = x @ w` -- one line with zero hardware decisions.
- Compound ops like DenseMLP chain 12 micro-ops, each with its own set of decisions. The per-micro-op breakdown yields approximately 81 total decisions versus 5 lines of PyTorch.
- Larger fused ops like SwigluOp (from `blaze/ops/swiglu/op.py`) extend this pattern further with 150+ lines of composition code.
- Non-tile-aligned dimensions require separate padded op variants (PaddedRMSNorm) with 5 CBs and 22+ CT args for what is conceptually "normalize this vector."
- Errors in any of these decisions produce silent numerical errors or hardware hangs -- there are no compile-time safety nets in the current API.
- The gap between "5 lines of PyTorch" and "81 manual hardware decisions" is the design space for the TensorAdapter abstraction.

## Source Files

- `blaze/ops/matmul/op.py` -- Matmul.emit() (the complete walkthrough example)
- `blaze/ops/dense_mlp/op.py` -- DenseMLP fused op composition (12 micro-ops)
- `blaze/ops/swiglu/op.py` -- SwigluOp: another compound op chaining DRAMStreamingMatmul, EltwiseMul, Gather, Mcast
- `blaze/ops/padded_rmsnorm/op.py` -- PaddedRMSNorm with explicit padding CBs (5 CBs, 22+ CT args)
- `blaze/_gated_mlp.py` -- build_gated_mlp_graph() 11-node graph builder
- `blaze/fused_program.py` -- FusedProgram (cb_from_tensor, cb_scratch, ct_args methods)

---

**Next:** [`04_abstraction_goals_and_scope.md`](./04_abstraction_goals_and_scope.md)
