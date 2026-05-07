# 06.05 -- MoE Ops

[<< Attention Ops](04_attention_ops.md) | [Back to Overview](01_op_library_overview.md)

Mixture-of-Experts ops form the largest and most complex family in the Blaze op
library. This section covers the routing, expert dispatch, the 8-variant MoE
matrix, and the >256 expert two-face path.


## MoE Architecture Overview

Every MoE block follows the same pipeline:

```
activation
  |
  v  RMSNorm (sender core)
  |
  v  Mcast (all compute cores)
  |
  +---------> Routed Expert Branch
  |           ├── Router (gate MM -> gather -> gate selection)
  |           ├── Mcast (indices, scores -> all cores)
  |           └── SwiGLU (indexed DRAM MMs -> eltwise -> gather -> mcast -> down MM)
  |
  +---------> Shared Expert Branch (optional)
  |           ├── KNMatmul (gate_up)
  |           ├── GatedReduce
  |           ├── Mcast
  |           └── DownProj (+ residual add)
  |
  v  ResidualAdd (merge routed + shared, if shared enabled)
  |
  v  ReduceToOne (cross-device tree reduction to root)
```

The shared/no-shared split determines whether the shared expert branch and
its merge ResidualAdd are allocated. When shared experts are absent, the
routed output feeds directly into ReduceToOne.


## The 8-Variant MoE Matrix

MoE ops are organized along three axes:

| Axis | Options | Reason |
|------|---------|--------|
| Gate style | DeepSeek (group-based top-k) vs GLM (global top-k) | Different routing algorithms |
| Expert count | Standard (<=256) vs Large (>256) | >256 requires two-face gate routing |
| Shared expert | With shared vs No shared | Different CB allocations |

### Full MoE FusedOps

| Experts | Shared | DeepSeek | GLM |
|---------|--------|----------|-----|
| <=256 | Yes | `MoE` | `GLMMoE` |
| <=256 | No | `MoENoShared` | `GLMMoENoShared` |
| >256 | Yes | `LargeMoE` | `GLMLargeMoE` |
| >256 | No | `LargeMoENoShared` | `GLMLargeMoENoShared` |

### Model-to-Op Mapping

| Model | Gate Type | Experts | Shared | Op |
|-------|-----------|---------|--------|----|
| DeepSeek V3 / R1 | DeepSeek | 256 | 1 | `MoE` |
| Ling 2.5 1T | DeepSeek | 256 | 1 | `MoE` |
| GLM-4.5 / 4.7 / 5 | C1 | 160-256 | 1 | `GLMMoE` |
| Mistral 3 Large | C1 | 128 | 1 | `GLMMoE` |
| Qwen3 30B / 235B | C1 | 128 | 0 | `GLMMoENoShared` |
| GPT-OSS 20B / 120B | B1 | 32-128 | 0 | `GLMMoENoShared` |
| Kimi K2 / K2.5 | C1 | 384 | 1 | `GLMLargeMoE` |
| Step 3.5 Flash | C1 | 288 | 1 | `GLMLargeMoE` |

### Decision Rules

```
1. gate_type = DeepSeek or C2  -->  DeepSeek variant
   gate_type = B1 or C1 or C0 -->  GLM variant
2. routed_experts_total > 256  -->  Large variant
3. shared_experts = 0          -->  NoShared variant
```

Or use the dispatch helper:

```python
from blaze.ops.utils.moe_dispatch import moe_op, glm_moe_op

op = moe_op(num_experts=256, enable_shared=True)       # -> blaze.moe
op = moe_op(num_experts=288, enable_shared=False)      # -> blaze.large_moe_no_shared
op = glm_moe_op(num_experts=128, enable_shared=False)  # -> blaze.glm_moe_no_shared
op = glm_moe_op(num_experts=384, enable_shared=True)   # -> blaze.glm_large_moe
```


## MoE.emit(): The Standard DeepSeek Path

**Source**: `blaze/ops/moe/op.py`

### Full Composition Tree

```
MoE
├── RMSNorm^
├── Mcast (act_mcast)^
├── RoutedExpert
│   ├── MoERouter
│   │   ├── MatmulFusedAct (gate_mm)^
│   │   ├── Gather (gate_gather)^
│   │   └── DeepseekMoeGate^ (group-based: top-4 groups -> top-8)
│   ├── Mcast (index_mcast)^
│   ├── Mcast (score_mcast)^
│   └── SwigluOp
│       ├── DRAMStreamingMatmul (up_proj)^
│       ├── DRAMStreamingMatmul (gate_proj)^
│       ├── EltwiseMul^
│       ├── Gather^
│       ├── Mcast (post_gather_mcast)^
│       └── DRAMStreamingMatmul (down_proj)^
├── SharedExpert
│   ├── KNMatmul (gu)^
│   ├── GatedReduce^
│   ├── Mcast^
│   └── DownProj
│       ├── Mcast (mcast2)^
│       ├── ResidualAdd^
│       └── Gather^
├── Mcast (shared_output_mcast)^
├── ResidualAdd (merge)^
└── ReduceToOne^
```

### Data Flow Trace

```python
# --- Preamble: normalize and broadcast ---
rmsnormed_act = RMSNorm.emit(f, act, rmsnorm_gamma,
    prefix="moe__rmsnorm", cores=f.sender_grid,
    epsilon=rmsnorm_epsilon, pop_input=False)

act_mcast_handle = Mcast.emit(f, rmsnormed_act,
    prefix="moe__act_mcast",
    dst_num_pages=act_mcast_pages, dst_tile_info=act_ti)
```

At this point, `act_mcast_handle` is live on all cores. Both branches read it
with `pop=False`.

```python
# --- Routed expert branch ---
routed_out = RoutedExpert.emit(f, act_mcast_handle,
    gate_weight, gate_bias,
    routed_swiglu_up_weights, routed_swiglu_gate_weights,
    routed_swiglu_down_weights,
    prefix="moe__routed",
    pop_router_act=False,    # keep act alive for shared branch
    pop_swiglu_act=False,    # keep act alive for shared branch
    index_offset=index_offset, ...)
```

RoutedExpert returns a CBHandle with the routed expert output on per-DRAM-bank
cores.

```python
# --- Shared expert branch ---
shared_out = SharedExpert.emit(f, act_mcast_handle,
    shared_gate_up_weight, shared_down_weight,
    _shared_bias, shared_output,
    prefix="moe__shared",
    pop_shared_act=False,
    skip_bias_add=not is_reduce_root)
```

SharedExpert returns a CBHandle with the shared expert output (see
[S03 -- SharedExpert](03_compute_ops.md#sharedexpert-the-complete-shared-mlp) for
the full internal pipeline). The residual add is skipped on non-root devices
(`skip_bias_add=True`) because the residual should only be added once after
cross-device reduction.

```python
# --- Merge and reduce ---
shared_for_add = Mcast.emit(f, shared_out,
    prefix="moe__shared_output_mcast")

reduce_input = ResidualAdd.emit(f, routed_out, shared_for_add,
    prefix="moe__merge",
    cores=routed_out.core_ranges)

return ReduceToOne.emit(f,
    input=reduce_input,
    intermediate=red2one_intermediate,
    output=red2one_output,
    prefix="moe__reduce_to_one",
    root_mesh_coord=red2one_root_mesh_coord, ...)
```


## MoERouter: Gate Selection

**Source**: `blaze/ops/moe_router/op.py`

The router determines which expert handles the current token. It outputs two
CBHandles: `gate_scores` and `gate_indices`.

### Standard Path (<=256 experts)

```
MoERouter
├── MatmulFusedAct (gate_mm)^    -- act x gate_weight -> logits per expert
├── Gather (gate_gather)^         -- collect all expert logits to sender
└── DeepseekMoeGate^              -- group-based top-k selection
```

The gate matmul uses face tiles (16x16) because the output has 256 elements
that pack into a single 16x16 tile. The Gather collects all gate MM core
outputs into one face tile on the sender core.

DeepseekMoeGate implements the DeepSeek V3 routing algorithm:
1. Group experts into groups (typically 8 groups of 32)
2. Select top-4 groups by summing intra-group logits
3. Within selected groups, select top-8 experts by individual logit
4. Apply softmax normalization to selected expert scores

### GLM Path (Global Top-K)

GLMMoERouter replaces DeepseekMoeGate with GLMMoeGate, which performs:
1. Optional sigmoid activation on gate logits
2. Global top-8 selection (no grouping)
3. Score normalization

```
GLMMoERouter
├── MatmulFusedAct (gate_mm, +sigmoid)^
├── Gather (gate_gather)^
└── GLMMoeGate^ (global top-8, no grouping)
```


## Two-Face Routing for >256 Experts

When a model has more than 256 routed experts, the gate logits cannot fit in a
single 16x16 tile face (which holds 256 elements). The solution: split the gate
MM output into two faces.

### Architecture

```
LargeRoutedExpert / GLMLargeRoutedExpert
├── MoELargeRouter / GLMMoELargeRouter
│   ├── MatmulFusedAct (gate_mm)^
│   ├── split_cb_handle_halves (face_a / face_b)
│   ├── Gather (gate_gather_a)^    -- first 256 experts
│   ├── Gather (gate_gather_b)^    -- experts 256+
│   └── GLMMoeGateMerge^           -- merge two faces -> global top-8
├── Mcast (index_mcast)^
├── Mcast (score_mcast)^
└── SwigluOp (same as standard)
```

### How Face Splitting Works

The gate MM cores are split in half:

```python
# From MoERouter.emit() when num_experts > 256:
all_cores = ttnn.corerange_to_cores(gate_mm_handle.core_ranges)
half = len(all_cores) // 2
face_a_cores_list = all_cores[:half]    # experts 0-255
face_b_cores_list = all_cores[half:]    # experts 256+
```

Each half gets its own CBHandle (wrapping the same underlying CB ID but with
different core ranges), its own Gather to the sender, and its own index set:

```python
# Static index tensors: face A starts at 0, face B starts at 256
"input_indices_a": _named_indices("input_indices_a", 0),
"input_indices_b": _named_indices("input_indices_b", 256),
```

### GLMMoeGateMerge

The merge kernel receives two face tiles on the sender core and produces:
1. Sort each face independently
2. Merge the two sorted lists
3. Select global top-8 from the merged result
4. Output `gate_indices` and `gate_scores` CBHandles

### Bias Faces

For large MoE, the gate bias tensor must also be split:

```python
# From GLMMoE.emit():
if num_experts > 256:
    gate_bias, gate_bias_b = prepare_bias_faces(f, gate_bias, num_experts)
    f.register_cleanup(gate_bias)
    f.register_cleanup(gate_bias_b)
```

The phantom column in `role_engine.py` dynamically expands from 1 to 2 columns
when >256 experts require more than 9 gate MM cores (grid_rows - 1).


## GLMMoE vs MoE: Structural Differences

The two MoE families are structurally identical except for the router sub-op:

```python
# MoE uses MoERouter -> DeepseekMoeGate (group-based)
routed_out = RoutedExpert.emit(f, ...)  # contains MoERouter

# GLMMoE uses GLMRoutedExpert -> GLMMoERouter -> GLMMoeGate (global top-k)
routed_out = GLMRoutedExpert.emit(f, ...)  # contains GLMMoERouter
```

The rest of the pipeline (SwiGLU, SharedExpert, ResidualAdd, ReduceToOne) is
identical. The `emit()` bodies of `MoE` and `GLMMoE` are structurally
parallel, differing only in which RoutedExpert variant they call and which
additional parameters they pass (e.g., `normalize`, `num_selected_experts`
for GLM).


## NoShared Variants

When `enable_shared_expert=False`, the composition tree simplifies dramatically:

```
MoENoShared / GLMMoENoShared
├── RMSNorm^
├── Mcast (act_mcast)^
├── RoutedExpert / GLMRoutedExpert
│   ├── Router (gate_mm -> gather -> gate)
│   ├── Mcast (index_mcast)^
│   ├── Mcast (score_mcast)^
│   └── SwigluOp (6 micro-ops)
└── ReduceToOne^
```

No SharedExpert, no merge ResidualAdd, no shared_output_mcast. The routed output
feeds directly into ReduceToOne. This saves ~7 micro-ops of CB allocation and
compute.

In code, this is a simple conditional in the shared `emit()`:

```python
if enable_shared_expert:
    shared_out = SharedExpert.emit(...)
    shared_for_add = Mcast.emit(...)
    reduce_input = ResidualAdd.emit(f, routed_out, shared_for_add, ...)
else:
    reduce_input = routed_out

return ReduceToOne.emit(f, input=reduce_input, ...)
```


## Shared Utilities

### emit_mlp_preamble (moe_common.py)

Factored preamble shared by MoE, DenseMLP, and DenseMLPDram:

```python
def emit_mlp_preamble(f, act, rmsnorm_gamma, prefix, rmsnorm_epsilon):
    rmsnormed_act = RMSNorm.emit(f, act, rmsnorm_gamma, ...)
    act_mcast_handle = Mcast.emit(f, rmsnormed_act, ...)
    return act_mcast_handle, act_ti, act_mcast_pages
```

### emit_moe_preamble (moe_common.py)

Extends the MLP preamble with the multi-device index offset:

```python
def emit_moe_preamble(f, act, rmsnorm_gamma, prefix, rmsnorm_epsilon, routed_index_offset):
    act_mcast_handle, _act_ti, _ = emit_mlp_preamble(...)
    index_offset = row * cols + col + routed_index_offset
    return act_mcast_handle, index_offset
```

### create_act_gather_dst (act_gather.py / moe.py)

Allocates a CB visible to both matmul cores and the sender core for gather
destinations:

```python
def create_act_gather_dst(f, *, prefix, num_pages, tile_info):
    dst_ranges = ttnn.CoreRangeSet(
        list(f.matmul_cores.ranges()) + list(f.sender_grid.ranges()))
    return f.cb_scratch(name=..., num_pages=num_pages,
        core_ranges=dst_ranges, ...)
```

### Router Static Tensors (router_tensors.py)

Pre-allocates named device tensors for gate input indices. For <=256 experts,
one 16x16 index tile. For >256, two index tiles plus merge scratch:

```python
if num_experts <= 256:
    return {"input_indices": _named_indices("input_indices", 0)}
else:
    return {
        "input_indices_a": _named_indices("input_indices_a", 0),
        "input_indices_b": _named_indices("input_indices_b", 256),
        "merge_act": ..., "merge_bias": ..., "merge_indices": ...,
    }
```


## Worked Example: Full MoE Micro-Op Count

For a standard `MoE` (DeepSeek, <=256 experts, shared expert enabled), the
total micro-op count in a single fused kernel is:

```
Preamble:       RMSNorm(1) + Mcast(1)                          = 2
Router:         MatmulFusedAct(1) + Gather(1) + DeepseekMoeGate(1)  = 3
Route Mcast:    Mcast(1) + Mcast(1)                             = 2
SwiGLU:         DRAMStreamingMatmul(3) + EltwiseMul(1) +
                Gather(1) + Mcast(1)                            = 6
SharedExpert:   KNMatmul(1) + GatedReduce(1) + Mcast(1) +
                DownProj(Mcast(1) + ResidualAdd(1) + Gather(1)) = 6
Merge:          Mcast(1) + ResidualAdd(1)                       = 2
Reduce:         ReduceToOne(1)                                  = 1
                                                         Total: 22
```

22 MicroOps, all compiled into a single kernel dispatch by BlazeCompiler
(see Ch4). Each MicroOp becomes a phase in the generated kernel's init-compute-
teardown loop, with CB allocations managed by the FusedProgram's scratch
compaction pass (see Ch5).


## Summary of Composition Patterns

The MoE family demonstrates several recurring patterns:

1. **Fan-out with pop_input=False**: Mcast output feeds both routed and shared
   branches without copying.

2. **Gather -> Mcast bridge**: Width-sharded partial results are gathered to the
   sender, then mcasted back to all cores for the next stage.

3. **Conditional composition**: `enable_shared_expert` gates an entire branch
   of the composition tree at `emit()` time.

4. **Index threading**: The expert `index` CB flows through three
   DRAMStreamingMatmuls, popped only at the last one.

5. **Hierarchical prefixes**: `moe__routed__router__gate_mm.out` traces the
   exact position in the composition tree.

6. **Dispatch helpers**: `moe_op()` / `glm_moe_op()` select the correct variant
   automatically, hiding the 8-variant matrix from model code.

[<< Attention Ops](04_attention_ops.md) | [Back to Overview](01_op_library_overview.md)
