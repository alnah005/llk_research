# Final Plan: Using TT-Blaze on T3K and Multi-Device Configurations via TT-Symbiote

---

## Selection Rationale

### Base Plan

**Plan V3** serves as the structural backbone. Its dual-track reading path design (Symbiote track: Ch1-3; Blaze track: Ch1, 4-6; convergence at Ch7-8) is the strongest organizational idea across all five plans, allowing readers to follow only the framework they need. Its 8-chapter, ~23-file structure avoids the volume risk of V1 (28 files) while maintaining sufficient depth.

### Contributions from Each Plan

- **Plan V1 (major contributor):** The integration chapter structure -- V1's 4-file Chapter 7 (framework comparison, module-to-FusedOp delegation, pipeline integration proposals, gaps/roadmap) is the most thorough treatment of the Symbiote-Blaze integration boundary across all plans. The detailed source-code-level specificity in file-level bullet points is adopted throughout. V1's explicit cross-chapter dependency prose (explaining the "why" of each dependency) supplements the tabular format.

- **Plan V2 (contributor):** The upfront conventions-before-chapters placement is adopted -- readers encounter definitions before they see terms used in chapter descriptions. The table-format terminology section is cleaner and more scannable than bullet-list formats. The dedicated "reusable patterns" file concept (V2's Ch7-03) is incorporated into the model implementations chapter.

- **Plan V3 (base structure):** The dual-track reader path architecture with explicit skip-path guidance. The clean dependency table with a convergence diagram. The concrete T3K deployment walkthrough embedded within the Symbiote track so short-path readers get actionable content early. The warning callout conventions.

- **Plan V4 (major contributor):** The most detailed T3K walkthrough of any plan -- Ch6's 5-file structure (environment, analysis, implementation, module adaptation, validation) provides the granularity needed for Question 5. The dedicated limitations chapter with 4 files (current limitations, models needing multi-device, missing infrastructure, recommended scaling path) thoroughly addresses Questions 10 and 12. The clear audience statement that no MPI or C++ kernel authoring experience is assumed.

- **Plan V5 (major contributor):** The dedicated Symbiote core architecture chapter -- V5 is the only plan to give module replacement mechanics, TorchTTNNTensor dispatch, and traced execution their own chapter before multi-device topics. The `04_traced_execution.md` file is the most detailed treatment of trace capture/replay across all plans, operationally critical for T3K deployment. The model-specific ops catalog file (Ch4-04) surveying the 100+ ops in blaze/ops/.

### Evaluator Feedback Incorporated

1. **Separate Symbiote core architecture from multi-device** (flagged in evaluations of V2, V3): Following V4 and V5, Symbiote core (module replacement, dispatch, run modes, traced execution) gets its own chapter before multi-device specifics. This avoids the pedagogical problem of V2 (which mixes single-device architecture with multi-device in one chapter).

2. **Consolidate model implementations into a dedicated chapter** (flagged in evaluations of V1, V3, V4, V5): All five evaluations noted that model implementations were scattered across chapters. A dedicated chapter gathers GLM-4.7, DeepSeek V3, Gemma4, Qwen3, and reusable patterns into one place.

3. **Expand the integration/comparison chapter** (flagged in evaluations of V2, V3): V3's original Ch7 was overloaded with parallelism comparison, model implementations, and integration boundaries. These are now split: model implementations get their own chapter, and the integration chapter focuses on framework comparison, integration pathways, and limitations/gaps.

4. **Pipeline_manager deserves structural prominence** (flagged in evaluation of V1): While not elevated to its own chapter (to stay within the 8-chapter budget), the pipeline_manager gets a full dedicated file with detailed coverage of the three-thread architecture, speculative decoding, and socket backends.

5. **Place the practical scaling guide after integration analysis** (flagged in evaluation of V4): The T3K scaling guide comes after the reader has seen both frameworks' approaches and the integration boundaries, avoiding the risk of committing to a Symbiote-only path before understanding when Blaze pipeline parallelism is needed.

6. **Add performance optimization and pipeline escalation guidance** (flagged in evaluation of V3): The scaling guide includes a dedicated file for performance optimization, CCL profiling, and the decision point for escalating to Blaze pipeline parallelism.

---

## Audience

This guide is written for Tenstorrent software engineers and advanced users who:

- Have working experience writing PyTorch models and can read Python and C++20 code.
- Understand the TTNN runtime basics: tensor creation, device placement, `ttnn.linear`, layout conversion (TILE_LAYOUT vs ROW_MAJOR_LAYOUT).
- Are familiar with the concept of mesh devices (multiple chips addressed as a single logical device) but may not have used multi-device configurations in practice.
- Know what tensor parallelism, pipeline parallelism, and collective communication (all-reduce, all-gather, reduce-scatter) mean conceptually, but may not know how Tenstorrent's specific implementations realize them.
- Have access to multi-device Tenstorrent hardware (N300, T3K, TG/Galaxy, or P150-based configurations).

The reader is NOT expected to know the internals of TT-Symbiote or TT-Blaze before reading this guide. No prior experience with MPI-based distributed programming, fabric routing, or C++ kernel authoring is assumed -- the guide builds these concepts from the ground up.

After reading, the reader should be able to: deploy a TT-Symbiote model on T3K with tensor parallelism, understand TT-Blaze's pipeline parallelism system for multi-galaxy inference, compare parallelism strategies for a given model and hardware configuration, and identify the integration boundaries between the two frameworks.

---

## Conventions

### Terminology

| Term | Definition |
|------|-----------|
| **MeshDevice** | A ttnn object representing one or more physical Tenstorrent chips organized as a 2D grid (rows, cols). A single N150 is (1,1); T3K is (1,8); Galaxy/TG is (8,4). |
| **Submesh** | A rectangular partition carved from a MeshDevice via `mesh_device.create_submeshes()`. Each pipeline stage in TT-Blaze runs on one submesh. |
| **Mesh mapper** | A TTNN strategy object (e.g., `ShardTensor2dMesh`, `ReplicateTensorToMesh`) that determines how a host tensor is distributed across mesh devices during `ttnn.from_torch`. |
| **Mesh composer** | The inverse of mesh_mapper: reconstructs a single host tensor from per-device shards during `ttnn.to_torch` (e.g., `ConcatMesh2dToTensor`). |
| **CCL** | Collective Communication Library. Operations like all-gather, reduce-scatter, all-reduce, and broadcast that move data across devices within a mesh. |
| **Fabric** | The physical ethernet interconnect between Tenstorrent chips, managed by the control-plane routing layer. Configured via `FabricConfig`. |
| **Tensor parallelism (TP)** | Splitting weight matrices and activations across devices within a single layer; synchronization via CCL ops. |
| **Pipeline parallelism (PP)** | Assigning different model layers to different device groups (stages), passing activations between stages via fabric sockets. |
| **Data parallelism (DP)** | Running the same model on multiple devices with different input batches. |
| **Column-parallel** | Weight sharded on the output dimension (last dim); each device computes a slice of the output. No CCL needed for the matmul itself. |
| **Row-parallel** | Weight sharded on the input dimension (second-to-last dim); each device computes a partial sum that requires CCL reduction. |
| **TTNNModule** | The TT-Symbiote base class for TTNN-accelerated modules that replace PyTorch `nn.Module` instances. |
| **TorchTTNNTensor** | The `torch.Tensor` subclass in Symbiote that wraps both a PyTorch tensor and a ttnn.Tensor, enabling transparent dispatch. |
| **BlazeOp / FusedOp** | A TT-Blaze operation. FusedOps compose multiple MicroOps into a single multi-kernel device program. |
| **MicroOp** | A TT-Blaze operation backed by a single C++ kernel header with an Op struct. |
| **FusedProgram** | The Blaze compilation target that bundles multiple kernels, circular buffers, semaphores, and runtime arguments into a single device program. |
| **PipelineGraph** | A TT-Blaze DAG structure defining pipeline stages (Nodes) and inter-stage connections (Edges) for multi-device orchestration. |
| **PipelineLayout** | The resolved physical mapping from a PipelineGraph: which submesh runs which stage, entry/exit chip coordinates, MPI rank assignments. |
| **StageKind** | An abstract base class in TT-Blaze that defines how a pipeline stage creates its PipelineBlock, performs setup, and launches compute. |
| **pipeline_manager** | The C++20 token scheduling runtime in TT-Blaze that manages continuous-batching inference with writer/reader/API threads. |
| **Loopback edge** | In a pipeline graph, the edge from the last stage back to the first, carrying tokens for continuous generation. |
| **Trace** | A recorded sequence of device operations that can be replayed without host dispatch overhead. Captured via Symbiote's `TracedRun` system. |
| **Cluster axis** | The dimension (0 = vertical/rows, 1 = horizontal/cols) along which a CCL operation communicates. |

### Notation

- File paths are given relative to repository roots: `core/module.py` means `tt-metal/models/experimental/tt_symbiote/core/module.py`; `pipeline_builder/graph.py` means `tt-blaze/pipeline_builder/graph.py`.
- Code examples use Python unless explicitly noted as C++.
- Mesh shapes are written as `(rows, cols)` tuples matching the `ttnn.MeshShape` convention: `(1, 8)` for T3K, `(8, 4)` for TG/Galaxy.
- Device architecture names use the `DeviceArch` enum values: N150, N300, T3K, TG, P150, P300, P150x4, P150x8, BHGLX.
- Class names are written in `MonospacedCode`. Source file references include the function/class name and approximate line number for navigation.

### Formatting Rules

- Each file begins with a one-paragraph summary of its contents and prerequisites.
- Diagrams use ASCII art for device topologies, data flow, and pipeline stage arrangements.
- Warning callouts use `> **Warning:**` blockquote format for common pitfalls (e.g., trace-incompatible CCL ops, non-divisible tensor dimensions).
- Each file ends with a "Key Takeaways" section summarizing the 2-3 most important points.
- Cross-references to other files use `[Chapter N, File M](../chNN_name/MM_name.md)` relative links.
- Source code references use the format `[file.py:ClassName.method_name]` with line numbers where meaningful.
- "Deep-dive" sections within files are clearly marked with level-3 headers and can be skipped on first reading.

---

## Chapter List

### Chapter 1 -- Device Topologies and Mesh Initialization

**Description:** Establishes the physical and logical device landscape -- what T3K, TG, P150x4/x8, and BHGLX configurations look like, how chips are interconnected, and how the MESH_DEVICE environment variable drives the entire device configuration stack for both frameworks.

**Directory:** `ch01_device_topologies/`

**Files:**

- `01_physical_topologies.md`
  - Enumerates every supported configuration: N150 (1 Wormhole chip), N300 (1x2), T3K (1x8 ring of Wormhole chips), TG/Galaxy (8x4 torus of 32 chips), P150 (single Blackhole), P300 (1x2 BH), P150x4 (1x4 BH), P150x8 (1x8 BH), BHGLX (8x4 Blackhole Galaxy).
  - Describes the physical interconnect topology for each: ethernet links, ring vs torus, chip-to-chip connections, bandwidth characteristics.
  - Ethernet link counts per topology from `TT_CCL.get_num_links()` in `tt/ccl.py`: N300/T3K = 1 link, P150x4/P300 = 2 links, TG = 4/3 links (axis-dependent), BHGLX = 4/3 links.
  - Cluster axis semantics: axis 0 = vertical (North-South/rows), axis 1 = horizontal (East-West/cols).
  - Visual ASCII art diagrams of chip arrangements within each topology.
  - Maps the `DeviceArch` enum values from `core/module.py` to physical topologies via `MeshShapeToDeviceArch`.

- `02_mesh_device_initialization.md`
  - The `MESH_DEVICE` environment variable: how it selects the mesh shape (e.g., `T3K` -> `(1, 8)`), how conftest fixtures use it to create `ttnn.MeshDevice`.
  - The `device_params` fixture pattern: `trace_region_size`, `num_command_queues`, `fabric_config` (FABRIC_1D_RING vs FABRIC_2D), `fabric_router_config`.
  - Test parametrization across topologies: the `mesh_device` indirect parametrization using `{topology_name: (rows, cols)}` dict with `os.environ.get("MESH_DEVICE")` lookup.
  - Key APIs: `ttnn.MeshDevice`, `ttnn.MeshShape`, `ttnn.MeshCoordinate`, `get_num_devices()`, `create_submeshes()`, `get_fabric_node_id()`.
  - The `run_on_devices` decorator: how it reads `MESH_DEVICE` to gate `forward()` execution to specific architectures.
  - How TT-Blaze's conftest.py loads tt-metal's conftest as a pytest plugin to inherit device fixtures.

- `03_fabric_and_routing.md`
  - What the ethernet fabric is and how it enables inter-chip communication.
  - Fabric configuration: `FabricConfig.FABRIC_1D_RING` (used by Symbiote for T3K) vs `FabricConfig.FABRIC_2D` (used by Blaze for Galaxy topologies).
  - The distinction between `ttnn.Topology.Linear` vs `ttnn.Topology.Ring` and when each is used.
  - Implications for CCL operation performance: more links = higher aggregate bandwidth = faster all-reduce/all-gather.
  - How the C++ control-plane APIs (`get_forwarding_direction`, `get_chip_neighbors`, `resolve_graph_layout`) discover physical connectivity automatically.

---

### Chapter 2 -- TT-Symbiote Core Architecture

**Description:** Explains TT-Symbiote's transparent module replacement system, the TorchTTNNTensor dispatch mechanism, run modes, and the traced execution system -- the single-device foundation that must be understood before multi-device topics.

**Directory:** `ch02_symbiote_core/`

**Files:**

- `01_module_replacement_engine.md`
  - `register_module_replacement_dict()`: the top-level API that takes a PyTorch model and an `{old_class: new_class}` mapping, recursively walks the module tree, and calls `new_class.from_torch(old_module)` to create TTNN replacements.
  - The `TTNNModule` base class lifecycle: `from_torch()` -> `preprocess_weights()` -> `move_weights_to_device()` -> `forward()`.
  - Weight preprocessing pipeline: `preprocess_weights_impl` -> `move_weights_to_device_impl` -> ready for inference.
  - The `exclude_replacement` set for keeping specific modules on CPU.
  - Module name assignment via `named_modules()` and `override_children_module_names`.

- `02_dispatch_and_tensor_wrapping.md`
  - `TorchTTNNTensor`: the `torch.Tensor` subclass that wraps both `.elem` (torch.Tensor) and `.ttnn_tensor` (ttnn.Tensor), enabling transparent dispatch.
  - `__torch_dispatch__`: how `NormalRun.torch_dispatch` checks `can_dispatch_to_ttnn` and routes ops to either TTNN or PyTorch fallback.
  - `DistributedTensorConfig` stored per-tensor: `mesh_mapper`, `mesh_composer`, `logical_shape_fn`.
  - The `_bypass_tensor_wrapping` optimization: TTNN children of TTNN parents skip TorchTTNNTensor wrapping and work with raw `ttnn.Tensor` directly.
  - `fast_unwrap_to_device`: lightweight transform for bypass mode.

- `03_run_modes.md`
  - Catalogs all run modes: NORMAL, NORMAL_WITH_FALLBACK, SEL (side-by-side execution and comparison), DPL (dual-path with error propagation), TRACED (three-phase trace lifecycle), CPU, LIGHTWEIGHT.
  - How `TT_SYMBIOTE_RUN_MODE` env var selects the mode.
  - `TracedRun`: the three-phase lifecycle (warm-up run 1 -> capture run 2 -> replay run 3+). Cache key computation from module name + input tensor signatures. Persistent input buffers and `_copy_inputs_to_trace_buffer`.
  - `TTNNLayerStack`: trace-enabled wrapper that captures an entire layer sequence as a single trace, reducing host dispatch overhead from O(num_layers) to O(1).
  - `@trace_enabled` / `@trace_disabled` decorators and when to use them.
  - Ring topology requirement for traced execution: why trace-compatible CCL ops must use Ring instead of Linear (no dynamic intermediate allocation).
  - `pre_trace_execute` and `post_trace_execute` hooks for CCL-sensitive operations.

- `04_end_to_end_model_flow.md`
  - Walks through a complete single-device model acceleration example using the `test_glm_4_7.py` pattern: load model, define replacement dict, `register_module_replacement_dict()`, `set_device()`, preprocess weights, run inference.
  - Explains how HuggingFace `model.generate()` continues to work transparently because TorchTTNNTensor masquerades as a regular `torch.Tensor`.
  - Timing/profiling via `DispatchManager.save_stats_to_file()`.
  - The `compose_transforms` pipeline: `wrap_to_torch_ttnn_tensor` -> `to_ttnn_wrap` -> `set_device_wrap`.

---

### Chapter 3 -- TT-Symbiote Multi-Device Support

**Description:** Deep-dives into how TT-Symbiote handles multi-device scenarios -- the DistributedConfig system, device initialization flow, tensor sharding strategies, CCL operations, and the distributed module implementations that enable tensor parallelism.

**Directory:** `ch03_symbiote_multi_device/`

**Files:**

- `01_distributed_config_and_device_init.md`
  - `DistributedConfig` dataclass: `mesh_device`, `tensor_config` (DistributedTensorConfig), `ccl_manager` (TT_CCL). Auto-initializes when `get_num_devices() > 1`.
  - `DistributedTensorConfig`: `mesh_mapper`, `mesh_composer`, `logical_shape_fn` -- how these three pieces work together to describe how tensors are split and reassembled.
  - Default sharding: `ShardTensor2dMesh(mesh_device, mesh_device.shape, (0, -1))` shards batch dimension along rows and last dimension along columns. `ConcatMesh2dToTensor` is the inverse.
  - `logical_shape_for_batch_channel_sharding()`: how logical shapes are computed as `[batch * mesh_rows, ..., hidden * mesh_cols]`.
  - `get_tensor_config_for_tensor()`: automatic fallback to `ReplicateTensorToMesh` when tensor dimensions are not evenly divisible by mesh shape.
  - `DeviceInit` class: `DEVICE_TO_STATE_DICT` singleton, `init_state()` / `init_state_impl()` which creates `DistributedConfig` for each mesh device.
  - `set_device()`: recursive walk through `nn.Module` and `TTNNModule` trees, calling `_initialize_module_on_device()` which sets both the device and the distributed state.
  - The conditional: `if device.get_num_devices() > 1` as the multi-device activation gate.

- `02_weight_sharding_strategies.md`
  - Column-parallel: `TTNNLinearIReplicatedWColSharded` -- input replicated, weight sharded on last dim (output dimension), each device computes a slice of the output. No CCL needed for the matmul itself.
  - Row-parallel: `TTNNLinearIColShardedWRowSharded` -- input column-sharded, weight row-sharded (dim=-2), matmul produces partial sums, followed by `reduce_scatter` to combine.
  - All-reduce variant: `TTNNLinearIColShardedWAllReduced` -- decomposes all-reduce into `reduce_scatter` + `all_gather` for trace compatibility (TTNN trace requires stable buffer addresses).
  - `shard_tensor_to_mesh_mapper(device, dim=dim)`: how weights are distributed during `move_weights_to_device_impl()`.
  - bfloat8 variants for LLaMA: `TTNNLinearLLamaIColShardedWRowSharded`.
  - The activation flow pattern: replicated input -> column-parallel matmul -> sharded activations -> row-parallel matmul -> reduce_scatter/all_reduce.

- `03_ccl_operations.md`
  - `TT_CCL` class from `models/tt_transformers/tt/ccl.py`: initialization with mesh_device, sub_device core range set, semaphore management with double-buffering for barrier, all-gather, and reduce-scatter operations.
  - `get_num_links(mesh_device, cluster_axis)`: per-topology link count lookup.
  - `ttnn.reduce_scatter`: partial-sum reduction along a cluster axis.
  - `ttnn.all_gather`: replicating results across all devices along an axis.
  - `tt_all_reduce()`: the unified all-reduce that adapts to topology (N300/T3K uses reduce_scatter; TG uses all_gather + fast_reduce_nc).
  - Trace-compatible all-reduce decomposition: `reduce_scatter` + `all_gather` instead of monolithic `all_reduce` (avoids dynamic buffer allocation that breaks TTNN trace).
  - `cluster_axis` parameter: axis 0 for vertical, axis 1 for horizontal CCL operations.
  - Semaphore cycling (`get_and_cycle_*_semaphore_handles`) for double-buffered CCL.
  - Topology parameter: `ttnn.Topology.Ring` vs `ttnn.Topology.Linear`. Ring required during traced execution.

- `04_distributed_modules.md`
  - `TTNNDistributedRMSNorm` (`modules/normalization.py`): `rms_norm_pre_all_gather` -> `all_gather` (to collect statistics) -> `rms_norm_post_all_gather` (apply normalization with gathered statistics). Restricted to T3K via `@run_on_devices(DeviceArch.T3K)`.
  - `TTNNDistributedRotaryPositionEmbedding` (`modules/rope.py`): distributes cos/sin tables across the mesh; RoPE that works with sharded head dimensions.
  - `TTNNDistributedEmbedding`: shards embedding table on dim 1, gathers output via all_gather.
  - Gemma4 attention/MLP modules: fused QKV with column-parallel + all_reduce, gated MLP with sharded projections, O-projection with row-parallel + reduce_scatter.
  - MoE modules (`TTNNQwen3MoE`): expert parallelism with all_gather for routing + reduce_scatter for output aggregation.
  - `TTNNBailingMoEAttention` and `TTNNQwen3NextGatedAttention`: attention modules using `_maybe_all_gather()` to reconstruct KV heads for multi-device.
  - KV cache replication: how `TTNNPagedAttentionKVCache.to_device()` uses `ReplicateTensorToMesh` when `get_num_devices() > 1`.

---

### Chapter 4 -- TT-Blaze Kernel Composition Framework

**Description:** Introduces TT-Blaze's approach to building fused operations from low-level kernel compositions -- the BlazeOp system, FusedProgram compilation, and how CCL is integrated at the kernel level, providing the foundation for understanding how pipeline stages are implemented.

**Directory:** `ch04_blaze_kernel_composition/`

**Files:**

- `01_blaze_op_architecture.md`
  - The three-tier op hierarchy: `BlazeOp` (base), `FusedOp` (composition of MicroOps), `MicroOp` (backed by C++ kernel header).
  - Port descriptors: `Input`, `Output`, `Internal` class attributes that declare the op's tensor interface.
  - Compile-time arguments (`ct_args`): typed parameters with `Type.UINT32`, `Type.BOOL`; RISC processor targeting with `Risc.NCRISC`, `Risc.BRISC`, `Risc.TRISC`.
  - The op lifecycle: class definition -> `register()` auto-generates OpSpec + CTArgSchema + FusedOpConfig -> `compose()` / `emit()` for compilation.
  - `FusedProgram`: the container for multi-kernel programs. CB (circular buffer) management, semaphore allocation, role engine, and grid configuration.
  - `DeviceContext`: per-device compilation context with `GridConfig` and `full_device_grid`, enabling mesh-aware compilation via `DeviceContext.from_device()`.
  - The compilation pipeline: op definition -> graph construction -> program generation -> device execution via `ttnn.generic_op()`.

- `02_ccl_in_blaze.md`
  - `setup_fabric()` in `blaze/ccl.py`: how Blaze ops connect to the fabric for inter-device communication within FusedPrograms.
  - Fabric semaphore allocation: teardown and buffer_index semaphores per connection.
  - `ttnn.compute_fabric_connection_rt_args()`: computing fabric runtime arguments from node IDs and link indices.
  - Fabric kernel defines via `ttnn.get_fabric_kernel_defines("Linear")`.
  - How CCL operations (broadcast, reduce, all-reduce, scatter) are implemented at the kernel level in Blaze vs the ttnn API-level operations used in Symbiote.
  - The RISC processor model for CCL: only data-movement RISCs (NCRISC, BRISC) can connect to fabric.
  - The `injected_fabric_barrier_builder.py`: automatic fabric barrier injection for synchronization.

- `03_ops_catalog_overview.md`
  - Survey of the `blaze/ops/` directory: 100+ op implementations covering matmul, attention (flash_mla, distributed_flash_mla), MoE (large_moe, glm_moe, routed_expert, shared_expert), normalization (rmsnorm, broadcast_rmsnorm), CCL (all_reduce, scatter, gather, reduce_to_one, ccl_broadcast), pipeline infrastructure (pipeline_stage_sync, embed_mcast), and model-specific ops (deepseek decoder blocks, GLM-5.1 stages).
  - DeepSeek V3 ops: MLA (multi-latent attention), MoE gate, routed expert, shared expert, decoder block.
  - GLM 5.1 ops: glm5_fused_proj, glm5_q_branch, glm_moe, glm_routed_expert.
  - The FusedOp composition pattern: composing MicroOps (matmul, rmsnorm, rope, sdpa) into a single program.
  - How ops handle multi-device: `sender_coord`, `reduce_root_mesh_coord`, `cluster_axis` parameters.
  - The weight_provider system: `BLAZE_WEIGHT_SOURCE` env var selecting between synthetic, state_dict, and blitz weight providers.

---

### Chapter 5 -- TT-Blaze Pipeline System for Multi-Device Orchestration

**Description:** Covers the complete pipeline infrastructure that enables model execution across multiple submeshes -- PipelineGraph for logical layout, SubmeshPartition for physical mapping, topology-based auto-discovery, stage kinds, and the C++20 pipeline_manager for continuous batching.

**Directory:** `ch05_blaze_pipeline_system/`

**Files:**

- `01_pipeline_graph_and_layout.md`
  - `PipelineGraph`: nodes (ordered dict of name -> Node), edges (list of Edge), optional `h2d_entry_chip` and `d2h_exit_chip`.
  - `Node` dataclass: `shape` (ttnn.MeshShape), optional `factory` (callable that creates a StageKind), `name` (display label).
  - `Edge` dataclass: `src`, `dst`, optional `src_exit_chip`/`dst_entry_chip` (for legacy explicit path), `is_loopback` flag.
  - Stage ordering via topological sort (Kahn's algorithm on non-loopback edges).
  - Two build paths: `build()` (legacy explicit with hardcoded chip IDs) and `build_topology()` (auto-discovery with no hardcoded IDs).
  - Example: 4-stage linear pipeline with loopback: `{f0->f1, f1->f2, f2->f3, f3->f0[loopback]}`.

- `02_submesh_partition_and_topology.md`
  - `SubmeshPartition`: carves a parent MeshDevice into uniform submeshes via `mesh_device.create_submeshes(submesh_shape)`.
  - Local submesh detection: iterating submeshes to find which one returns `True` for `sub.get_view().is_local(MeshCoordinate(0,0))`.
  - `chip_to_coord_map()`: building a global-chip-ID to local-MeshCoordinate mapping across all submeshes.
  - The C++ topology resolver: `ttnn._ttnn.multi_device.experimental.resolve_graph_layout()` -- input (edge tuples + submesh chips) -> output (stage order, node-to-submesh mapping, resolved edges with entry/exit coordinates, H2D/D2H chips).
  - The backtracking search that assigns physical submeshes to logical nodes based on ethernet connectivity.
  - MPI rank binding: `distributed_context_allgather_int(my_stage_idx)` resolves which MPI rank owns each pipeline stage.
  - `PipelineLayout` dataclass: `my_stage_idx`, `my_submesh`, `submesh_to_stage`, `stages_metadata`, `pipeline_config`, `stage_factories`, `stage_labels`.
  - `PipelineConfigEntry`: `entry_node_coord` and `exit_node_coord` for each stage, plus the loopback entry at index N.

- `03_stage_kinds_and_execution.md`
  - `StageKind` abstract base: `create_pipeline_block()`, `setup()`, `launch_compute()`.
  - `StageContext` dataclass: `mesh_device`, `pipeline_config`, `my_mesh_id`.
  - Concrete stages: `EmbeddingStage` (H2D socket + embedding lookup + activation forwarding), `LMHeadStage` (activation receive + broadcast + matmul + argmax + CCL reduce + D2H socket), `DenseDecoderStage` (attention + dense MLP + reduce-to-one), `MoEDecoderStage` (attention + MoE routing + expert dispatch + reduce), `PassthroughStage` (token/activation forwarding).
  - `PipelineBlock`: per-stage socket/FIFO configuration (upstream/downstream page sizes, entry/exit coordinates, H2D/D2H sockets).
  - Constants: `ACTIVATION_DIM=7168`, `ACTIVATION_PAGE_SIZE_BYTES=14336`, `PIPELINE_CORE_COORD=CoreCoord(11,0)`.
  - The `Pipeline` orchestrator: 4-phase lifecycle (configure_block -> setup -> start_pipeline -> start_compute).
  - Persistent mode execution with `persistent_next_iter_semaphore` for continuous execution.

- `04_pipeline_manager_cpp.md`
  - The C++20 `PipelineManager::Impl`: three pinned threads -- `writer_thread` (token injection), `reader_thread` (result collection), `api_thread` (external request handling).
  - CPU affinity: `AUTO_CPU` assignment, `resolve_cpu_assignments()` with stride-based spreading, `pin_to_cpu()` for pthread affinity.
  - Writer loop priority: (1) decode tokens from `DecodeStaging`, (2) prefill tokens from `PrefillQueue` with chunked injection (`params.chunk_size` = 24).
  - `InjectDescriptor`: slot_id, token_id, position, token_type (BASE/SPEC), temperature/top_p/top_k sampling params.
  - Reader loop: processes `ResultDescriptor` from pipeline, handles decode loopback, completion detection (EOS), cancellation cleanup. `ReaderClaim` RAII for slot pinning.
  - API thread: handles ALLOCATE (slot allocation from `FreeIdPool`), SUBMIT (prompt store + prefill queue), CONTINUE (multi-turn), CANCEL (graceful cleanup with `CancelBitmap`).
  - Key data structures: `UserTable` (per-slot state, position tracking, generation counters), `PromptTable` (prompt token storage), `DecodeStaging` (FIFO + generation counter), `PrefillQueue` (round-robin chunked scheduling), `BoundedQueue<T>` for inter-thread communication.
  - Speculative decoding: dual injection (BASE + SPEC tokens), `SpecDecodeState` with `has_unverified`/`has_verified` flags, ACCEPT/REJECT/STALE state machine with deferred completion.
  - `PipelineConfig` variant: `MockConfig` (configurable latency/accept rate for testing) vs `SocketConfig` (real H2D/D2H socket communication with 64-byte page format).
  - `ManagerParams`: `max_users` (64), `max_seq_len` (128K), `chunk_size` (24), `eos_token`.

- `05_hardware_configurations.md`
  - `HardwareConfig` and `ModelPipelineSpec` system from `test_model_pipelines.py`.
  - Single galaxy (32 BH chips): 4x2 (4 stages), 2x2 (8 stages), 1x2 (16 stages), 1x1 (32 stages).
  - Dual galaxy (64 chips): 4x2 (8 stages), 2x2 (16 stages), 1x2 (32 stages).
  - Quad galaxy (128 chips): 4x2 (16 stages), 2x2 (32 stages), 1x2 (64 stages).
  - Mesh graph descriptors: textproto files that define the physical chip connectivity.
  - Rank binding YAML files: mapping MPI ranks to physical device assignments.
  - How these scale from Wormhole-based T3K (8 chips) to large Blackhole Galaxy clusters.

---

### Chapter 6 -- Existing Multi-Device Model Implementations

**Description:** Walks through concrete multi-device model implementations in both frameworks, extracting reusable patterns for weight sharding, CCL configuration, pipeline construction, and test parametrization.

**Directory:** `ch06_model_implementations/`

**Files:**

- `01_symbiote_tensor_parallel_models.md`
  - GLM-4.7 (`test_glm_4_7.py`): module replacement with `{nn.Linear: TTNNLinearLLama, nn.SiLU: TTNNSilu}`, mesh_device parametrization across all topologies (N150/N300/T3K/TG/P150/P300/P150x4/P150x8/BHGLX), single-device TP via basic TTNNLinear.
  - Qwen3-Coder-Next (`test_qwen3_coder_next.py`): fabric_config=FABRIC_1D_RING, placeholder for `TTNNQwen3MoE` and `TTNNLinearIColShardedWRowSharded` (currently commented out -- showing work in progress).
  - Gemma4 on T3K: full distributed attention with `TTNNDistributedRMSNorm`, `TTNNDistributedRotaryPositionEmbedding`, fused QKV with `TTNNLinearIColShardedWAllReduced`, O-projection with `TTNNLinearIColShardedWRowSharded`.
  - Ling-mini-2.0: `TTNNBailingMoEDecoderLayer` with distributed RMSNorm + attention + MoE on T3K.
  - The common pattern: load HF model -> define replacement dict -> `register_module_replacement_dict()` -> `set_device(model, mesh_device)` -> preprocess weights -> generate.
  - How `@run_on_devices(DeviceArch.T3K)` gates distributed forward paths so the same module can have both single-device and multi-device implementations.

- `02_blaze_deepseek_v3_pipeline.md`
  - The `ModelPipeline` class: initialization with `weights_mode` (synthetic/real/state_dict), distributed process count validation (4/16/64 procs).
  - `create_pipeline_configuration_from_num_procs()`: routes to single galaxy (4 stages), single pod (16 stages), or SP4 (64 stages) configurations.
  - The full pipeline topology for DeepSeek V3 671B: EmbeddingStage -> [DenseDecoderStage x 3] -> [MoEDecoderStage x 58] -> LMHeadStage (62 stages total).
  - `DecoderStage` implementation: `create_pipeline_block` sets up sockets, `setup` allocates tensors/semaphores/weights, `launch_compute` executes `DecoderBlock.execute()`.
  - `DecoderBlock` program context: overlapped weight tensors, RoPE tensors, KV cache, semaphores, socket handles.
  - Weight loading via `WeightProvider`: `load_moe_layer()`, `load_dense_layer()`, `load_embedding()`, `load_lm_head()` -- each host loads only its stage's weights.
  - Persistent mode execution with `persistent_next_iter_semaphore`.

- `03_reusable_patterns_and_antipatterns.md`
  - Pattern: multi-topology test parametrization -- how both frameworks use `MESH_DEVICE` env var with mesh shape dictionaries.
  - Pattern: weight sharding -- `shard_tensor_to_mesh_mapper(device, dim=N)` in Symbiote vs per-submesh weight distribution in Blaze.
  - Pattern: CCL operation selection -- Ring topology for 1D meshes (T3K), axis-specific `cluster_axis` for 2D meshes (TG/Galaxy).
  - Pattern: trace-compatible CCL -- decomposing all_reduce into reduce_scatter + all_gather in both frameworks.
  - Pattern: fabric config selection -- FABRIC_1D_RING for linear chains, FABRIC_2D for 2D meshes.
  - Pattern: factory-driven stage construction -- `Node.factory(submesh)` creates StageKind instances lazily.
  - Pattern: warmup-then-measure timing methodology.
  - Anti-pattern: tensor shape not evenly divisible by mesh dimensions (falls back to replication silently).
  - Anti-pattern: forgetting `fabric_config` in `device_params`.
  - Anti-pattern: using `all_reduce` inside trace capture (must decompose into reduce_scatter + all_gather).

---

### Chapter 7 -- Integration Boundaries and Parallelism Strategy Comparison

**Description:** Analyzes the architectural relationship between TT-Symbiote and TT-Blaze, compares their parallelism strategies, identifies concrete integration pathways, and catalogs current limitations and gaps.

**Directory:** `ch07_integration_and_comparison/`

**Files:**

- `01_framework_comparison.md`
  - Side-by-side comparison: Symbiote (PyTorch-transparent, module-replacement, ttnn API ops, TP-focused) vs Blaze (kernel-level, FusedOp composition, custom kernels, PP+TP-focused).
  - Abstraction level: Symbiote operates at PyTorch module granularity with TorchTTNNTensor dispatch; Blaze operates at per-core kernel composition with explicit buffer management.
  - Programming model: Symbiote uses `from_torch()` + `forward()` with automatic tensor conversion; Blaze uses `BlazeOp.emit()` + `FusedProgram` compilation.
  - Multi-device model: Symbiote uses DistributedConfig + mesh_mapper/composer with ttnn CCL API calls; Blaze uses PipelineGraph + SubmeshPartition + fabric-level CCL setup.
  - Strengths of each: Symbiote for rapid prototyping, HuggingFace ecosystem integration, transparent acceleration of arbitrary models; Blaze for fused kernel execution (lower dispatch overhead), pipeline parallelism, continuous batching, optimized CCL.

- `02_parallelism_strategies_compared.md`
  - Data parallelism (DP): batch sharding via `ShardTensor2dMesh(mesh, shape, (0, -1))` in Symbiote; Blaze's single-batch-per-pipeline design means DP is implicit at the pipeline level.
  - Tensor parallelism (TP): column-parallel + row-parallel linear layers in Symbiote; within-submesh TP in Blaze (e.g., 4x2 submesh = 8-way TP per stage).
  - Pipeline parallelism (PP): not supported in Symbiote; first-class in Blaze via PipelineGraph + pipeline_manager with continuous batching.
  - Expert parallelism: Symbiote has `TTNNQwen3MoE`; Blaze has deep MoE kernel support (large_moe, glm_moe, distributed_topk, routed_expert).
  - Hybrid PP+TP in Blaze: each pipeline stage runs TP across its submesh chips, and stages are connected via pipeline sockets.
  - Tradeoffs table: latency, throughput, memory efficiency, implementation complexity, trace compatibility.
  - Which strategy fits which model size and hardware configuration: TP alone for models fitting in 8-way sharded DRAM on T3K; PP+TP for models exceeding T3K aggregate memory; EP for MoE models.

- `03_integration_pathways.md`
  - Pathway A: Symbiote TTNNModule delegates to a Blaze-compiled FusedOp -- `forward()` calls a pre-compiled BlazeProgram instead of individual TTNN ops. `BlazeCompiler.compile()` produces a `MeshProgramDescriptor` executable via `ttnn.generic_op()`.
  - Pathway B: Symbiote provides model graph decomposition and module replacement; Blaze provides pipeline orchestration and continuous batching scheduling. Symbiote-replaced modules wrapped as Blaze `StageKind` implementations.
  - Pathway C: Symbiote as the single-device fast path for rapid prototyping on N150/N300; Blaze for production multi-device deployment on T3K/TG/Galaxy.
  - The tensor interface boundary: Symbiote's TorchTTNNTensor wraps ttnn.Tensor, which is what Blaze's compiler expects. TILE_LAYOUT + bfloat16 are compatible.
  - Current blockers: Blaze requires upfront program compilation with exact tensor shapes; Symbiote's dynamic dispatch discovers shapes at runtime. KV cache management divergence (Symbiote uses TTNNPagedAttentionKVCache, Blaze manages KV per-stage). No shared weight distribution protocol.

- `04_limitations_gaps_and_roadmap.md`
  - Symbiote multi-device limitations: only tensor parallelism (no pipeline parallelism), no continuous batching, no speculative decoding, limited to ttnn API-level ops (no kernel fusion), manual module class selection.
  - Blaze multi-device limitations: tightly coupled to specific model architectures (DeepSeek V3, GLM-5.1), requires model-specific stage implementations with C++/kernel-level expertise, no automatic module replacement.
  - Operations that do not parallelize well: attention KV cache updates (position-dependent), token embedding (small relative to communication overhead), MoE routing softmax across sharded tokens, final argmax (requires global reduction), non-divisible dimension handling (fallback to replication).
  - CCL constraints: Ring topology required for traced execution; single-link bandwidth on T3K limits TP scalability.
  - Models needing multi-device but lacking support: Qwen3-Coder-Next MoE (module replacement commented out), models exceeding ~12GB DRAM per Wormhole chip.
  - Device-specific gaps: P150x4/P150x8 parametrized but no Blackhole-specific sharding strategies in Symbiote; TG/Galaxy requires 2D mesh sharding partially supported via ShardTensor2dMesh but lacks tested end-to-end flows. `@run_on_devices` currently only has T3K-specific implementations for most distributed modules.
  - Missing infrastructure: no Symbiote StageKind adapter, no shared weight distribution protocol, no unified CCL semaphore management, no automatic sharding strategy inference, no dynamic shape support for traced execution.
  - Proposed roadmap: (1) Blaze FusedOp wrappers for key Symbiote modules, (2) SymbioteStage adapter for pipeline integration, (3) unified multi-device configuration system.

---

### Chapter 8 -- Scaling a Single-Device Model to T3K and Beyond

**Description:** Provides a concrete, step-by-step recipe for taking a working single-device TT-Symbiote model and scaling it to T3K (8 Wormhole chips) with tensor parallelism, covering every decision point from model analysis through validation, with guidance on when to escalate to Blaze pipeline parallelism.

**Directory:** `ch08_scaling_guide/`

**Files:**

- `01_model_analysis_and_strategy_selection.md`
  - Step 1: Profile the single-device model using `DispatchManager.save_stats_to_file()` -- identify compute-bound vs memory-bound layers, determine if the model fits in single-device DRAM.
  - Step 2: Analyze model architecture -- parameter count, hidden dimension, number of attention heads, MoE structure. Estimate per-device DRAM requirements for 8-way sharded tensor parallelism.
  - Step 3: Choose parallelism strategy -- decision tree:
    - Model fits on one device but is latency-bound? -> Data parallelism (batch sharding).
    - Weight matrices too large for one device, hidden_dim divisible by 8? -> Tensor parallelism via Symbiote on T3K (1,8).
    - Model has 60+ sequential layers exceeding T3K aggregate memory? -> Pipeline parallelism via Blaze.
    - MoE with many experts? -> Expert parallelism.
    - Mixed requirements? -> Hybrid TP within submesh + PP across submeshes.
  - Step 4: Identify which modules need distributed variants -- linear layers (column-parallel or row-parallel), normalization (distributed RMSNorm), attention (head-parallel QKV + distributed RoPE), MoE (expert distribution).

- `02_module_adaptation_and_weight_distribution.md`
  - Step-by-step replacement map for tensor parallelism on T3K (1x8):
    - `nn.Linear` for Q/K/V projections -> `TTNNLinearIReplicatedWColSharded` (weight column-sharded, each device holds 1/8 of heads).
    - `nn.Linear` for O projection -> `TTNNLinearIColShardedWRowSharded` (input column-sharded from attention output, weight row-sharded, `reduce_scatter` to combine).
    - `nn.Linear` for MLP gate/up -> `TTNNLinearIReplicatedWColSharded`.
    - `nn.Linear` for MLP down -> `TTNNLinearIColShardedWAllReduced` (reduce_scatter + all_gather for replicated output feeding residual add).
    - `RMSNorm` -> `TTNNDistributedRMSNorm` (pre-gather stats, all_gather, post-gather norm).
    - `RotaryEmbedding` -> `TTNNDistributedRotaryPositionEmbedding`.
  - Weight preprocessing: using `shard_tensor_to_mesh_mapper(device, dim=N)` with the appropriate dim, ensuring tensors are evenly divisible by the mesh shape.
  - Handling non-shardable tensors: the `get_tensor_config_for_tensor()` automatic replication fallback.
  - Adding `@run_on_devices(DeviceArch.T3K)` decorators to distributed forward methods.

- `03_test_setup_and_validation.md`
  - Environment setup: `MESH_DEVICE=T3K`, `device_params` with `fabric_config=ttnn.FabricConfig.FABRIC_1D_RING`, `trace_region_size` (50MB typical).
  - Test parametrization: the mesh_device fixture dict pattern `{"T3K": (1, 8)}` from `test_glm_4_7.py`.
  - Correctness validation: SEL run mode (`TT_SYMBIOTE_RUN_MODE=SEL`) for side-by-side TTNN vs PyTorch golden reference comparison at every op.
  - DPL mode for tracking error propagation through the network.
  - Verifying CCL operations: checking that composed output (via `mesh_composer`) against single-device reference.
  - Trace validation: running in TRACED mode and verifying that trace capture/replay produces identical results.
  - Per-module timing analysis: `DispatchManager.save_stats_to_file()` output format, identifying bottleneck modules.
  - Common failure modes: shape mismatches from non-divisible dimensions, CCL timeouts from misconfigured fabric, trace capture failures from dynamic buffer allocation in CCL ops, Ring vs Linear topology mismatches, fabric_config missing from device_params.

- `04_performance_optimization_and_pipeline_escalation.md`
  - Profiling multi-device execution: identifying CCL bottlenecks (all_gather and reduce_scatter durations), compute vs communication ratio.
  - Optimization strategies: reduce CCL operations per layer (fused QKV projections), use Ring topology for better bandwidth utilization, minimize host-device synchronization.
  - When to escalate to Blaze pipeline parallelism: model does not fit in 8-way TP memory budget, TP communication overhead dominates latency, need for continuous batching or speculative decoding.
  - The escalation path: define a `PipelineGraph` with one `Node` per stage, create `StageKind` implementations (using Blaze fused ops or wrapping Symbiote modules), use `build_topology()` for auto-discovery, connect via pipeline_manager for continuous batching.
  - Summary scaling roadmap: single-device validation (NORMAL/SEL) -> tensor parallelism on T3K (this chapter) -> pipeline parallelism via Blaze (when TP is insufficient) -> hybrid PP+TP for production deployment.

---

## Cross-Chapter Dependencies

| Chapter | Depends On | Concepts Referenced |
|---------|------------|-------------------|
| Ch 2 (Symbiote Core) | Ch 1 (Topologies) | MeshDevice creation, `MESH_DEVICE` env var, `DeviceArch` enum, `run_on_devices` decorator |
| Ch 3 (Symbiote Multi-Device) | Ch 1, Ch 2 | Mesh shapes and fabric config (Ch 1); TTNNModule lifecycle, module replacement, dispatch, traced execution (Ch 2) |
| Ch 4 (Blaze Kernel Composition) | Ch 1 | DeviceContext and mesh-aware compilation require understanding of device grids and mesh coordinates |
| Ch 5 (Blaze Pipeline System) | Ch 1, Ch 4 | Submesh partitioning and fabric topology (Ch 1); BlazeOp/FusedProgram compilation for stage compute (Ch 4) |
| Ch 6 (Model Implementations) | Ch 3, Ch 5 | Symbiote TP patterns (Ch 3) for Symbiote models; pipeline stage system (Ch 5) for DeepSeek V3 |
| Ch 7 (Integration and Comparison) | Ch 2, Ch 3, Ch 4, Ch 5 | Full understanding of both frameworks' architectures and multi-device approaches to identify meaningful integration boundaries |
| Ch 8 (Scaling Guide) | Ch 1, Ch 2, Ch 3, Ch 7 | Device topology selection (Ch 1); Symbiote architecture (Ch 2); multi-device TP adaptation (Ch 3); parallelism strategy comparison and escalation decision (Ch 7) |

The dependency graph forms two parallel tracks that converge at Chapter 7:

```
Symbiote track:  Ch 1 --> Ch 2 --> Ch 3 --+
                                           +--> Ch 6 --> Ch 7 --> Ch 8
Blaze track:     Ch 1 --> Ch 4 --> Ch 5 --+
```

**Reading paths:**
- **Symbiote-focused reader:** Chapters 1, 2, 3, 8. Read Chapter 7 selectively for parallelism comparison and escalation guidance.
- **Blaze-focused reader:** Chapters 1, 4, 5. Read Chapter 6 file 02 for DeepSeek V3. Read Chapter 7 selectively for integration context.
- **Integration-focused reader:** All chapters in order.
- **"Just make it work on T3K" reader:** Chapter 1 (skim), Chapter 8, then back-reference Chapters 2-3 as needed.
