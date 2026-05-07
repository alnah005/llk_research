# 06 -- The Op Library

[Next: Data Movement Ops -->](02_data_movement_ops.md)

This chapter catalogs the ~110 ops in `blaze/ops/`, organized by function. Each op lives in its own directory with an `op.py` that defines either a `MicroOp` or a `FusedOp` subclass. The chapter focuses on mechanical detail: class names, `emit()` signatures, CBHandle flow, and key kwargs. For the compilation pipeline that invokes these ops, see Ch4 (BlazeCompiler); for the FusedProgram and CBHandle abstractions they target, see Ch5 (FusedProgram Internals).


## 6.1 Directory Convention

Every op directory follows the same structure:

```
blaze/ops/<op_name>/
    __init__.py          # re-exports the op class
    op.py                # class definition (MicroOp or FusedOp)
    kernels/             # C++ kernel sources (MicroOps only)
        op.hpp           # unified kernel with COMPILE_FOR_NCRISC/BRISC/TRISC guards
```

The `__init__.py` re-exports the op class so callers write `from ..mcast import Mcast` rather than `from ..mcast.op import Mcast`.


## 6.2 MicroOp vs FusedOp

Both inherit from `BlazeOp` (see Ch2 S01-S02 for the full class hierarchy, attribute tables, `emit()`/`compose()` contracts, and the `__`/`___` prefix delimiter system). MicroOps map 1:1 to a C++ kernel phase and have their own `kernels/op.hpp`; FusedOps compose multiple `emit()` calls into a single dispatch. The sections below document each op's specific `emit()` signature, inputs, outputs, and kwargs.


## 6.3 CBHandle Flow Patterns

Ops communicate through CBHandles (see Ch4 S01 for the full abstraction and Ch4 S02 for the `cb_from_tensor()` API). An op receives either a `ttnn.Tensor` (first consumer, resolved via `f.cb_from_tensor()`) or a `CBHandle` (chained from upstream). Most ops accept either via an `isinstance` check, setting a CT arg flag (`in0_is_tensor_backed`) so the kernel knows whether `init()` should push the CB.


## 6.4 Op Census by Category

The ~110 op directories break down as follows:

| Category | Count | Examples |
|----------|-------|---------|
| **Data movement** | ~18 | Mcast, Gather, Scatter, Copy, Shard2CB, Embedding, EmbedMcast, AllReduce, CclBroadcast, BarrierSender, BarrierReceiver, Migration, PipelineStageSync, DmRiscHandshake |
| **Compute (core)** | ~18 | Matmul, MatmulCB, MatmulFusedAct, KNMatmul, DRAMStreamingMatmul, RMSNorm, PaddedRMSNorm, BroadcastRMSNorm, ResidualAdd, EltwiseMul, Rope, ClampedSilu, Retilize, Untilize, TileRowConvert, Argmax, LmHeadSampling |
| **Attention (MLA)** | ~20 | MLA, PreSDPA, QBranch, QAProjection, QHeads, CreateQHeads, KVBranch, KVCacheUpdate, FlashMLADecode, SdpaReduceToAll, DistributedFlashMLA, PostSdpa, SDPA, SparseFlashMLA |
| **Attention (DSA/GLM)** | ~18 | DsaPipeline, DsaIndexer, DsaTopk, DsaSparseGather, DsaSparseFlashDecode, DsaKvPrep, DsaRetilizeIndexer, DsaAttention, DsaFullAttention, DsaCacheUpdate, GlmFusedProj, GlmQBranch, SDPAPostSDPA, QSDPAPostSDPA, PreDsa, PostDsa |
| **Attention (GQA)** | ~2 | GQAPreSDPA, GatherReduce |
| **MoE** | ~30 | MoE, MoENoShared, GLMMoE, GLMMoENoShared, LargeMoE, LargeMoENoShared, GLMLargeMoE, GLMLargeMoENoShared, MoERouter, GLMMoERouter, MoELargeRouter, GLMMoELargeRouter, RoutedExpert, GLMRoutedExpert, LargeRoutedExpert, GLMLargeRoutedExpert, DeepseekMoeGate, GLMMoeGate, GLMMoeGateMerge, SwigluOp, SharedExpert, GatedReduce, GatedLocalReduce, DownProj, ReduceToOne, DenseMLP, DenseMLPDram, DenseSwiglu, AllReduceMoe |
| **CB utilities** | ~4 | CbFlush, CbReconfig, CbScratchReset, TileCopy |
| **Misc** | ~3 | RtArgsDemo, MlpMixed, DistributedTopk |


## 6.5 FusedOp Trees -- How Ops Compose

FusedOps form hierarchical trees. All MicroOp leaves in a tree share one compiled kernel dispatch -- the FusedProgram accumulates all their CB allocations, CT args, and semaphores into a single ProgramDescriptor.

The two largest trees in the library:
- **MLA**: PreSDPA -> DistributedFlashMLA -> PostSdpa (~20 MicroOp leaves). Full tree in [Section 04 -- Attention Ops](04_attention_ops.md).
- **MoE**: RMSNorm -> RoutedExpert -> SharedExpert -> ReduceToOne (~22 MicroOp leaves). Full tree in [Section 05 -- MoE Ops](05_moe_ops.md).


## 6.6 Shared Utilities (`blaze/ops/utils/`)

| Module | Key Functions | Used By |
|--------|--------------|---------|
| `moe_dispatch.py` | `moe_op()`, `glm_moe_op()`, `routed_expert_op()`, `glm_moe_router_op()` | Tests, model specs |
| `moe_common.py` | `emit_moe_preamble()`, `emit_shared_expert_and_reduce()` | All top-level MoE FusedOps |
| `router_tensors.py` | `create_single_face_static_tensors()`, `create_two_face_static_tensors()` | All router ops |
| `bias_faces.py` | `prepare_bias_faces()` | Large MoE ops (>256 experts) |
| `core_split.py` | `split_cb_handle_halves()` | Large routers (face_a/face_b) |
| `act_gather.py` | `create_act_gather_dst()` | SharedExpert path |
| `direct_address.py` | `resolve_weight_direct_address()` | Matmul, MatmulCB |
| `tensor_accessor.py` | `tensor_accessor_ct_args()` | Shard2CB, KVCacheUpdate, FlashMLADecode |
| `face_view_utils.py` | `can_use_face_view()`, `FACE_HEIGHT`, `FACE_WIDTH` | GatedReduce, KNMatmul |


## 6.7 Cross-Chapter References

- **Ch2** (Architecture): The Tensix core's BRISC/NCRISC/TRISC split that determines which RISC each CT arg targets.
- **Ch4** (BlazeCompiler): How `compose()` is invoked during compilation, and how the shadow graph is built from `f.output()` calls.
- **Ch5** (FusedProgram): The `CBHandle`, `cb_scratch()`, `cb_from_tensor()`, `cb_alias()`, semaphore allocation, and CT arg registration APIs that every `emit()` method calls.

---

[Next: Data Movement Ops -->](02_data_movement_ops.md)
