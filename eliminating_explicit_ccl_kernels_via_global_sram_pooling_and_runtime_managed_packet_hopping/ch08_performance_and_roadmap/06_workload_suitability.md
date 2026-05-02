# 8.6 Workload Suitability

## Context

Sections 8.1--8.5 established the performance envelope, the explicit-vs-implicit tradeoff
matrix, and the per-architecture gap analysis. This section assesses how six common
multi-chip parallelism patterns map onto the GSRP model, identifying which workloads
benefit most and which should remain on explicit CCL. The reader should understand the
per-collective mapping from Section 8.3.9, the programmability advantage analysis from
Section 8.3.4, and the latency/bandwidth characteristics from Sections 8.1--8.2. The
comparative lessons from Chapter 5 (Cerebras's dataflow model, Graphcore's BSP, NVIDIA's
persistent-explicit-CCL) inform the suitability assessment. This analysis guides the
phased adoption roadmap in Section 8.7 by identifying which workloads should be targeted
first.

---

## 8.6.1 Assessment Framework

Each parallelism pattern is scored on five criteria:

| Criterion | Scoring | GSRP-Favoring Indicators |
|-----------|---------|--------------------------|
| Communication regularity | Regular (1) to Irregular (5) | Irregular patterns benefit most |
| Transfer granularity | Bulk (1) to Small (5) | Small transfers avoid pipeline gap |
| Pattern dynamism | Static (1) to Data-dependent (5) | Dynamic patterns need runtime routing |
| Programmability burden | Low (1) to High (5) | High burden means largest GSRP win |
| Latency sensitivity | Tolerant (1) to Critical (5) | Critical needs fastest-path routing |

**Composite GSRP Fit Score:** Average of five criteria, scaled 1--5. Higher = better GSRP
fit. The inverse correlation between regularity and GSRP suitability reflects the core
insight: GSRP adds the most value where pre-computed CCL schedules are impractical.

---

## 8.6.2 Pattern 1: Data Parallelism (All-Reduce Dominated)

```
Data Parallel Training:

   Chip 0        Chip 1        Chip 2        Chip 3
+----------+  +----------+  +----------+  +----------+
| Forward  |  | Forward  |  | Forward  |  | Forward  |
| Backward |  | Backward |  | Backward |  | Backward |
| Gradients|  | Gradients|  | Gradients|  | Gradients|
+----+-----+  +----+-----+  +----+-----+  +----+-----+
     |             |             |             |
     +------- All-Reduce (ring-based) --------+
     |             |             |             |
+----+-----+  +----+-----+  +----+-----+  +----+-----+
| Update   |  | Update   |  | Update   |  | Update   |
+----------+  +----------+  +----------+  +----------+
```

### Assessment

| Criterion | Score | Rationale |
|-----------|:-----:|-----------|
| Regularity | 1 | Fixed ring pattern, same every iteration |
| Granularity | 1 | Large gradient tensors (MB--GB) |
| Dynamism | 1 | Static schedule, pre-computed routes |
| Programmability | 2 | Already a single TTNN API call |
| Latency sensitivity | 2 | Tolerant (pipeline hides latency) |
| **GSRP Fit Score** | **1.4** | |

> **Recommendation:** Keep on explicit CCL. All-reduce's ring pipeline achieves >99%
> efficiency for large tensors (Section 8.1). GSRP adds overhead without simplifying
> programming (already a single `ttnn.all_reduce()` call). This pattern represents the
> worst-case for GSRP value proposition.

---

## 8.6.3 Pattern 2: Tensor/Model Parallelism (All-Gather + Reduce-Scatter)

```
Tensor Parallel (Megatron-Style):

   Chip 0          Chip 1          Chip 2          Chip 3
+-----------+   +-----------+   +-----------+   +-----------+
| Col-split |   | Col-split |   | Col-split |   | Col-split |
| MatMul    |   | MatMul    |   | MatMul    |   | MatMul    |
+-----+-----+   +-----+-----+   +-----+-----+   +-----+-----+
      |               |               |               |
      +--------- All-Gather ---------+               |
      |               |               |               |
+-----+-----+   +-----+-----+   +-----+-----+   +-----+-----+
| Row-split |   | Row-split |   | Row-split |   | Row-split |
| MatMul    |   | MatMul    |   | MatMul    |   | MatMul    |
+-----+-----+   +-----+-----+   +-----+-----+   +-----+-----+
      |               |               |               |
      +------ Reduce-Scatter ---------+               |
```

### Assessment

| Criterion | Score | Rationale |
|-----------|:-----:|-----------|
| Regularity | 2 | Mostly regular, but shard sizes vary by layer |
| Granularity | 2 | Medium-to-large (activation slices, 100 KB--10 MB) |
| Dynamism | 1 | Static schedule per training step |
| Programmability | 3 | Moderate: topology selection, shard coordination |
| Latency sensitivity | 3 | Moderate: compute-comm overlap is critical |
| **GSRP Fit Score** | **2.2** | |

> **Recommendation:** Mixed. Retain explicit CCL for the all-gather and reduce-scatter
> collectives themselves (pipeline efficiency matters). Consider GSRP for the
> weight-update synchronization phase where smaller, irregular transfers occur.

---

## 8.6.4 Pattern 3: Pipeline Parallelism (Point-to-Point)

```
Pipeline Parallel:

   Chip 0          Chip 1          Chip 2          Chip 3
+-----------+   +-----------+   +-----------+   +-----------+
| Stage 0   |-->| Stage 1   |-->| Stage 2   |-->| Stage 3   |
| (Layers   |   | (Layers   |   | (Layers   |   | (Layers   |
|  0-7)     |   |  8-15)    |   |  16-23)   |   |  24-31)   |
+-----------+   +-----------+   +-----------+   +-----------+
                P2P send                P2P send
```

### Assessment

| Criterion | Score | Rationale |
|-----------|:-----:|-----------|
| Regularity | 2 | Fixed pipeline, but microbatch scheduling varies |
| Granularity | 2 | Medium (activation tensors per microbatch) |
| Dynamism | 2 | Static pipeline, but dynamic microbatch count |
| Programmability | 2 | `MeshSocket`/`FabricSocket` already abstracts well |
| Latency sensitivity | 4 | Pipeline bubble minimization is critical |
| **GSRP Fit Score** | **2.4** | |

> **Recommendation:** Either model works. The existing `MeshSocket` and `FabricSocket`
> APIs (Chapter 4, Section 3) already provide clean abstractions for point-to-point
> transfers. GSRP adds marginal value over sockets for fixed-pattern pipelines.

---

## 8.6.5 Pattern 4: Expert Parallelism / Mixture of Experts (MoE)

```
MoE All-to-All Pattern:

   Chip 0          Chip 1          Chip 2          Chip 3
+-----------+   +-----------+   +-----------+   +-----------+
| Token     |   | Token     |   | Token     |   | Token     |
| Router    |   | Router    |   | Router    |   | Router    |
| (per-token|   | (per-token|   | (per-token|   | (per-token|
|  expert)  |   |  expert)  |   |  expert)  |   |  expert)  |
+--+--+--+--+   +--+--+--+--+   +--+--+--+--+   +--+--+--+--+
   |  |  |  |      |  |  |  |      |  |  |  |      |  |  |  |
   +--+--+--+------+--+--+--+------+--+--+--+------+--+--+--+
              All-to-All Dispatch (data-dependent routing)
   +--+--+--+------+--+--+--+------+--+--+--+------+--+--+--+
   |  |  |  |      |  |  |  |      |  |  |  |      |  |  |  |
+--+--+--+--+   +--+--+--+--+   +--+--+--+--+   +--+--+--+--+
| Expert 0  |   | Expert 1  |   | Expert 2  |   | Expert 3  |
+-----------+   +-----------+   +-----------+   +-----------+
```

### Assessment

| Criterion | Score | Rationale |
|-----------|:-----:|-----------|
| Regularity | 5 | Highly irregular: expert routing is input-dependent |
| Granularity | 4 | Small-to-medium: per-token or per-chunk transfers |
| Dynamism | 5 | Fully data-dependent: which tokens go to which expert |
| Programmability | 5 | Very complex with explicit CCL: dynamic dispatch logic |
| Latency sensitivity | 3 | Moderate: expert computation dominates |
| **GSRP Fit Score** | **4.4** | |

> **Key Insight:** Expert parallelism / MoE is the **highest-scoring workload** for GSRP
> suitability. The data-dependent routing pattern makes pre-computed CCL schedules
> impractical -- each input batch generates a different all-to-all pattern.

```cpp
// MoE dispatch with GSRP (conceptual): ~10 lines
for (uint32_t t = 0; t < num_tokens; t++) {
    uint8_t expert_id = router_output[t];
    uint8_t dest_chip = expert_chip_map[expert_id];
    uint32_t dest_offset = expert_buffer_offset(expert_id, t);

    GlobalAddress dest = GlobalAddress::from_components(
        MY_MESH, dest_chip, EXPERT_CORE_X, EXPERT_CORE_Y, dest_offset);

    gsrp_write(dest, &token_data[t * token_size], token_size);
}
```

Compare to the explicit CCL approach, which requires pre-aggregating tokens by destination,
constructing per-destination packet batches, and managing connection lifecycles for each
destination chip -- easily 200+ lines of kernel code.

---

## 8.6.6 Pattern 5: Inference with KV-Cache Sharing

```
KV-Cache Sharing Across Chips:

   Chip 0          Chip 1          Chip 2          Chip 3
+-----------+   +-----------+   +-----------+   +-----------+
| KV-Cache  |   | KV-Cache  |   | KV-Cache  |   | KV-Cache  |
| (layers   |   | (layers   |   | (layers   |   | (layers   |
|  0-7)     |   |  8-15)    |   |  16-23)   |   |  24-31)   |
+-----------+   +-----------+   +-----------+   +-----------+
      ^               ^               ^               ^
      |               |               |               |
  +---+---+       +---+---+       +---+---+       +---+---+
  | Read  |       | Read  |       | Read  |       | Read  |
  | from  |<------| from  |<------| from  |<------| from  |
  | remote|       | remote|       | remote|       | remote|
  | KV    |       | KV    |       | KV    |       | KV    |
  +-------+       +-------+       +-------+       +-------+
```

### Assessment

| Criterion | Score | Rationale |
|-----------|:-----:|-----------|
| Regularity | 3 | Regular per-layer but variable per-request (batch) |
| Granularity | 4 | Small-to-medium: KV vectors per attention head |
| Dynamism | 4 | Batch-dependent: which KV entries are accessed |
| Programmability | 4 | Complex: cross-chip reads with consistency requirements |
| Latency sensitivity | 5 | Critical: KV access is on the inference critical path |
| **GSRP Fit Score** | **4.0** | |

> **Recommendation:** GSRP with prefetching for KV-cache sharing. The push model
> (writer pushes KV updates to all consumers via `gsrp_write()`) is preferred over
> the pull model (`gsrp_read()`) to avoid the ~2x read latency penalty (Section 8.2.8).

---

## 8.6.7 Pattern 6: Activation Checkpointing and Dynamic Routing

```
Activation Checkpointing:

   Forward Pass:                    Backward Pass:
   Chip 0 --> Chip 1 --> Chip 2    Chip 2 --> Chip 1 --> Chip 0
   [save ckpt to remote L1]        [read ckpt from remote L1]
   [pattern depends on memory       [recompute vs. read decision
    pressure and layer config]       is dynamic per layer]
```

### Assessment

| Criterion | Score | Rationale |
|-----------|:-----:|-----------|
| Regularity | 3 | Layer-regular but config-dependent |
| Granularity | 3 | Medium: activation tensors per layer |
| Dynamism | 5 | Dynamic: checkpoint-vs-recompute is runtime decision |
| Programmability | 4 | Complex: managing checkpoint locations across chips |
| Latency sensitivity | 3 | Moderate: overlapped with backward computation |
| **GSRP Fit Score** | **3.6** | |

> **Recommendation:** GSRP is well-suited. The dynamic decision of where to store
> checkpoints and whether to recompute or retrieve them maps naturally to `gsrp_write()`
> for storage and `gsrp_read()` for retrieval. The "named remote memory" concept (Phase 1)
> directly addresses the use case.

---

## 8.6.8 Summary: GSRP Suitability Ranking

```
GSRP Fit Score by Workload:

  5.0 |
  4.5 |  x Expert/MoE (4.4)
  4.0 |     x KV-Cache (4.0)
  3.5 |        x Activation Ckpt (3.6)
  3.0 |
  2.5 |           x Pipeline (2.4)
  2.0 |              x Tensor/Model (2.2)
  1.5 |                 x Data Parallel (1.4)
  1.0 |
      +----+----+----+----+----+----+--------
       Expert  KV   Ckpt  Pipe  TP   DP
```

| Rank | Workload | GSRP Fit | Recommendation |
|:----:|----------|:--------:|:---------------|
| 1 | Expert/MoE | 4.4 | **First adoption target** -- highest ROI |
| 2 | KV-Cache sharing | 4.0 | **Phase 2 target** -- requires read support |
| 3 | Activation checkpointing | 3.6 | **Phase 1--2 target** -- write-heavy, dynamic |
| 4 | Pipeline parallelism | 2.4 | Optional -- sockets already adequate |
| 5 | Tensor/model parallelism | 2.2 | Hybrid -- GSRP for sync, CCL for collectives |
| 6 | Data parallelism | 1.4 | **Keep on CCL** -- no GSRP benefit |

---

## 8.6.9 Production Model Analysis

### LLaMA-class Transformer (Training, 8-chip TP)

| Communication Event | Volume per Layer | Frequency | Recommendation |
|:-------------------|:----------------:|:---------:|:--------------:|
| TP all-gather (Q/K/V) | 8--32 MB | Every layer | Explicit CCL |
| TP reduce-scatter (output) | 8--32 MB | Every layer | Explicit CCL |
| DP all-reduce (gradients) | 100 MB--1 GB | Every backward pass | Explicit CCL |

**GSRP opportunity:** None for the core communication pattern. LLaMA training on 8 chips
is dominated by regular bulk collectives.

### Mixtral-class MoE (Training, 32-chip EP + TP)

| Communication Event | Volume per Layer | Frequency | Recommendation |
|:-------------------|:----------------:|:---------:|:--------------:|
| TP all-gather | 4--16 MB | Every layer | Explicit CCL |
| TP reduce-scatter | 4--16 MB | Every layer | Explicit CCL |
| MoE dispatch (EP) | Variable, 1--8 MB total | Every MoE layer | **GSRP** |
| MoE combine (EP) | Variable, 1--8 MB total | Every MoE layer | **GSRP** |
| DP all-reduce | 200 MB--2 GB | Every backward pass | Explicit CCL |

**GSRP opportunity:** Significant for MoE layers (dispatch + combine), which are the
hardest to optimize with explicit CCL due to data-dependent routing. The MoE dispatch
reduces from ~200+ lines of explicit CCL to ~10 lines of GSRP.

### LLaMA-class Inference (8-chip TP, Decode)

| Communication Event | Volume per Step | Frequency | Recommendation |
|:-------------------|:---------------:|:---------:|:--------------:|
| TP all-gather (per token) | 128 KB--2 MB | Every layer, every token | Explicit CCL |
| KV-cache read | 16--256 KB per head | Every layer, every token | **GSRP** |
| Activation streaming | 256 KB--4 MB | Every layer | Either |

**GSRP opportunity:** KV-cache sharing is a strong fit, especially for long-context
inference where the cache is distributed across chips.

---

## Key Takeaways

- **Expert parallelism (MoE) is the highest-value GSRP target** (fit score 4.4/5.0), with
  data-dependent routing that makes explicit CCL impractical and GSRP natural.

- **KV-cache sharing is the second-highest target** (4.0/5.0), requiring Phase 2 read
  support. The push model (`gsrp_write()`) is preferred over pull (`gsrp_read()`) to
  avoid the ~2x read latency penalty.

- **Data parallelism (all-reduce) should never use GSRP** (1.4/5.0). Ring pipeline
  efficiency and single-call TTNN API make explicit CCL permanently superior.

- **The hybrid CCL+GSRP model is validated by workload analysis:** different parallelism
  patterns within the same training run map to different communication models.

- **Production model analysis confirms selective deployment:** Mixtral-class MoE shows
  significant GSRP opportunity (dispatch + combine), while LLaMA-class training shows none.

## GSRP Implications

The workload suitability analysis reveals a clear adoption priority: MoE expert dispatch
first, then KV-cache sharing, then activation checkpointing. Phase 1 (write-only) is
sufficient for MoE dispatch and activation checkpointing. Phase 2 (read support) unlocks
KV-cache sharing. Data parallelism and tensor parallelism should remain on explicit CCL
permanently. The five-criteria scoring framework provides a systematic method for evaluating
future workload patterns as they emerge.

## Source Code References

| File | Key Symbols |
|------|-------------|
| `ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch/all_to_all_dispatch.hpp` | All-to-all dispatch for MoE: explicit CCL baseline |
| `ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/all_to_all_combine.hpp` | All-to-all combine: expert output collection |
| `ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather.hpp` | All-gather for tensor parallelism |
| `ttnn/cpp/ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp` | Reduce-scatter for tensor parallelism |
| `tt_metal/api/tt-metalium/experimental/sockets/mesh_socket.hpp` | `MeshSocket` for pipeline parallelism P2P |
| `ttnn/api/ttnn/distributed/fabric_socket.hpp` | `FabricSocket`, `BidirectionalFabricSocket` |
| `tt_metal/api/tt-metalium/mesh_buffer.hpp` | `MeshBuffer::create()` -- activation checkpoint allocation |

---

**Previous:** [`05_blackhole_gap_analysis.md`](./05_blackhole_gap_analysis.md) | **Next:** [`07_phased_adoption_roadmap.md`](./07_phased_adoption_roadmap.md)
