# Final Synthesized Plan: Comprehensive Understanding of DeepSeek V3 B1

---

## Selection Rationale

### Base Plan

**Plan V1** serves as the primary structural base, with significant modifications drawn from Plans V2, V3, and V4. Plan V1 was selected because:

- It received the highest evaluation scores for depth, specificity, and question coverage across all evaluators.
- It follows a strict bottom-up dependency order: architecture overview -> unified kernel system -> micro-ops -> fused ops -> domain deep dives -> infrastructure -> integration. Every evaluator confirmed this ordering is pedagogically sound.
- Its per-file content descriptions are the most detailed of all six plans, reading almost like outlines for the actual written content. This dramatically reduces ambiguity for writers.
- It cleanly separates reference material (micro-op and fused-op catalogs) from narrative material (MLA and MoE deep dives), enabling both sequential reading and reference lookup.

### What Was Taken from Each Plan

**From Plan V1 (base):**
- The overall 8-chapter structure with the layered progression from foundational to advanced
- The dedicated MLA deep-dive chapter (Chapter 5) with 4 files covering query projection, compressed KV cache, Flash MLA decode, and post-SDPA output projection
- The dedicated MoE deep-dive chapter (Chapter 6) with 4 files including a DRAM-streaming matmul deep dive
- The detailed micro-op catalog organization (compute, data movement, communication/infrastructure)
- The comprehensive conventions section and cross-chapter dependency table

**From Plan V2:**
- Separation of weight preparation and scaleout into distinct chapters (V1 bundled them; all evaluators flagged this as a weakness)
- The intra-device parallelism discussion (TP/EP within a 4x2 mesh) for the scaleout chapter
- The audience definition balance (teaching both TT-Blaze and DeepSeek V3 specifics from first principles)

**From Plan V3:**
- The full decode step data flow placed as a capstone file after all subsystems have been explained (V3's evaluator praised this, and V1's evaluator recommended it). However, we also retain an early high-level overview in Chapter 1 for orientation.
- The fusion principles and patterns section as a dedicated opening file in the fused-op chapter
- The question coverage matrix
- The formatting rule that every golden function reference shows the mathematical formula it implements

**From Plan V4:**
- The dedicated TT-Blaze execution pattern walkthrough using `Matmul.op()` as the canonical minimal example (V4's evaluator praised this as a strong pedagogical entry point)
- The dedicated per-core runtime args treatment in the unified kernel system chapter
- The comprehensive terminology table entries (tensor-backed CB, face-view, B1)

**From Plan V5:**
- Multiple reading order recommendations (linear, attention-first, MoE-first, deployment)
- The MLA weight layout as a dedicated section within the MLA chapter
- The gate bias and special tensors coverage within weight preparation

**From Plan V6:**
- Forward-reference annotations in the cross-chapter dependency section
- The "CB reconfig" and "fusion group" terminology entries
- Reading order recommendations with 4+ paths

### How Evaluator Feedback Was Incorporated

1. **"Unified kernel system and CB management should come BEFORE the op library chapters" (all evaluators).** This is satisfied: Chapter 2 covers unified kernel system and CB management before Chapters 3 and 4 (micro-ops and fused ops). Plans V5 and V6 were specifically criticized for placing unified kernels at Chapter 4.

2. **"MLA and MoE need dedicated deep-dive chapters" (V2 evaluator's primary criticism).** This is satisfied: Chapters 5 (MLA) and 6 (MoE) are full deep-dive chapters with 4 files each. V2 was the only plan lacking these.

3. **"Full decode step data flow needs prominent placement, not buried" (V3 evaluator, V4 evaluator).** This is addressed with a dual approach: Chapter 1 contains a high-level decode step overview for orientation, and Chapter 7 (scaleout) contains the comprehensive annotated trace as a capstone after all subsystems are explained.

4. **"DRAM-streaming matmul needs proper coverage" (evaluators for V3, V4, V5, V6).** This is satisfied: DRAM-streaming matmul appears in the micro-op catalog (Chapter 3) and receives a dedicated deep-dive section in Chapter 6 (MoE), file 04, where it is most relevant.

5. **"Weight prep and scaleout should be properly separated" (V1 evaluator).** This is satisfied: weight preparation is Chapter 7, multi-device scaleout is in Chapter 8 alongside the demo runtime.

6. **"Chapter count should stay within 1-8, files per chapter 2-5" (constraint).** This is satisfied: 8 chapters, 2-4 files per chapter, 27 files total.

7. **"Every question must be covered by at least one chapter" (constraint).** Verified via the question coverage matrix below.

---

## Audience

This guide is written for **systems-level ML engineers and kernel developers** who work on deploying large language models on Tenstorrent hardware. Readers are expected to already understand:

- Transformer architecture fundamentals (attention, MLP/MoE, layer norms, embeddings, KV cache)
- Basic accelerator programming concepts (kernels, shared memory, tiling, data movement)
- The TT-Metal/TTNN ecosystem at an introductory level (devices, tensors, mesh devices, sharding, `ttnn.generic_op`)
- Tenstorrent hardware basics: Tensix cores, the three RISC-V processors (NCRISC, BRISC, TRISC), NOC, L1 SRAM, DRAM banks, circular buffers, tiles
- Python and C++ at a professional level

The guide assumes readers have **not** previously worked with:

- TT-Blaze and the unified kernel descriptor system
- The DeepSeek V3 architecture's specific design choices (Multi-Latent Attention, 256-expert MoE with shared experts)
- This production codebase

These are taught from first principles within the guide. Upon completion, readers will be able to trace a full decode step, modify individual ops, add new fusion patterns, and configure multi-host deployments.

---

## Chapter 1: Architecture Overview and Model Mapping

**Description:** Establishes how the 671B-parameter DeepSeek V3 architecture maps to the TT-Blaze op library and Tenstorrent hardware, introduces the host-side model interface, and traces the execution model from Python to device.

### Files

#### `01_deepseek_v3_architecture_on_tenstorrent.md`
- The DeepSeek V3 model architecture: 61 transformer layers (3 dense + 58 MoE), Multi-head Latent Attention (MLA) with compressed KV cache, 256-expert MoE with top-8 routing and 1 shared expert, 671B total parameters
- Three-tier op hierarchy: micro-ops (atomic compute primitives in `micro_ops/`) -> fused ops (multi-stage pipelines in `fused_ops/`) -> model layer (full transformer block)
- Directory layout walkthrough: `model.py`, `micro_ops/`, `fused_ops/`, `unified_kernels/`, `prepare_weights.py`, `blitz_decode_weights.py`, `scaleout_configs/`, `demo/`
- The batch-1 decode constraint: why B=1, what `page_size_bytes` alignment means for PCIe transfers, and how int32 token IDs flow over sockets
- Source files: `model.py`, `demo/runtime.py`

#### `02_tt_blaze_execution_pattern.md`
- The execution pattern shared by every op: (1) create CB descriptors from sharded tensors via `cb_descriptor_from_sharded_tensor`, (2) build a `UnifiedKernelDescriptor` with per-RISC compile-time and runtime args, (3) call `get_kernel_descriptors()` to produce NCRISC/BRISC/TRISC `KernelDescriptor` objects, (4) pack into `ProgramDescriptor`, (5) execute via `ttnn.generic_op(io_tensors, program_descriptor)`
- Walkthrough of `Matmul.op()` as the canonical minimal example demonstrating the full Python-to-device path
- The `DeepSeekV3` class in `model.py`: H2D/D2H socket protocol, prefill-by-decode strategy, DEVICE_PULL vs HOST_PUSH H2D modes, position tracking
- The decode lifecycle: `start()` -> `prefill()` -> `_switch_to_decode()` -> `decode_step()` -> `stop()`

#### `03_layer_taxonomy_and_decode_overview.md`
- The 61-layer model structure: layers 0-2 are dense (single routed expert per device), layers 3-60 are MoE (256 routed experts with top-8 gating plus shared expert). `FIRST_K_DENSE_REPLACE` constant (3)
- The pipeline stages: stage 0 = embedding, stages 1-61 = decoder layers, stage 62 = LM head + sampling
- The `DeepSeekV3DenseLayerWeights` vs `DeepSeekV3MoELayerWeights` dataclasses
- High-level decode step overview (preview): token ID -> H2D -> embedding -> N transformer layers (BroadcastRMS -> PreSDPA -> FlashMLA -> PostSDPA -> MoE/Dense MLP) -> LM head -> argmax -> D2H -> token ID. This serves as an orientation map; the fully annotated trace appears in Chapter 8.
- Source files: `model.py`, `demo/cli.py`, `demo/runner.py`

**Covers questions:** 1, 12 (overview)

---

## Chapter 2: The Unified Kernel System

**Description:** Explains the unified kernel descriptor infrastructure that enables a single C++ source file to compile for all three RISC processors on each Tensix core, the circular buffer management system, and how kernels are composed into programs.

### Files

#### `01_unified_kernel_descriptor.md`
- The `UnifiedKernelDescriptor` dataclass: how a single `kernel_source` path and per-RISC compile-time/runtime args produce 3+ `KernelDescriptor` objects
- RISC processor roles: NCRISC (reader/data movement via NOC_0), BRISC (writer/data movement via NOC_1), TRISC (compute via FPU/SFPU)
- The simple path (`_get_simple_kernel_descriptors`): one kernel set for all cores, 3 descriptors
- The split path (`_get_split_kernel_descriptors`): how `UnifiedCompileTimeCoreDescriptor` and `PerCoreCompileTimeDescriptor` create multiple kernel variants with different compile-time args for different core subsets (e.g., `is_sender` vs `is_receiver`, `bank_id` per core)
- `PerCoreRuntimeArgsDescriptor`: per-RISC, per-core runtime arguments for operations needing core-specific runtime parameters (e.g., per-core NOC addresses in gather/mcast, per-core bank IDs for DRAM streaming)
- `KernelGroup` and `UnifiedKernelResult`: how callers identify which kernel indices correspond to sender/receiver/compute roles
- Source file: `unified_kernel_descriptor.py`

#### `02_kernel_op_api_and_unified_headers.md`
- `kernel_op_api.hpp`: the `COMPILE_FOR_NCRISC`/`BRISC`/`TRISC` preprocessor guards, `is_ncrisc`/`is_brisc`/`is_trisc` constexpr bools, and the `SelectByRISCV<Reader, Writer, Compute>` template for type dispatch
- How a single .cpp/.hpp kernel file uses `#if defined(COMPILE_FOR_NCRISC)` sections to contain reader, writer, and compute code in one file
- The unified kernel pattern: a struct with nested `ReaderCTArgs`/`WriterCTArgs`/`ComputeCTArgs`, `ReaderArgs`/`WriterArgs`/`ComputeArgs`, and `reader()`/`writer()`/`compute()` methods
- How compile-time args become constexpr template parameters enabling dead-code elimination per core role
- `kernel_utils.hpp`: `linear_id_in_grid()` for core-to-index mapping, `get_split_half_core_info()` for A/B core group splitting, `setup_sharded_buffer()` for tensor-backed CB initialization, cross-RISC synchronization barriers (`sync_riscs_enter`/`sync_riscs_exit`), and `reconfig_cb_interfaces` for fused ops
- The unified_kernels/*.hpp headers: each header (matmul.hpp, rmsnorm.hpp, rope.hpp, gather.hpp, mcast.hpp, etc.) encapsulates one reusable op's logic as inline functions/templates callable from multiple kernel .cpp files
- Source files: `unified_kernels/kernel_op_api.hpp`, `unified_kernels/kernel_utils.hpp`, all `unified_kernels/*.hpp`

#### `03_circular_buffer_management.md`
- `CircularBufferIdManager`: how CB IDs (0-63) are allocated across multiple contexts with format-based reuse
- The Context pattern: within one context every CB ID is unique; across contexts, IDs can be reused if `data_format` and `tile` match
- `CBDescriptor` construction patterns: `cb_descriptor_from_sharded_tensor` (tensor-backed CBs where address/size come from the tensor) vs manual CBDescriptor (intermediate buffers with explicit sizing)
- `cb_descriptor_from_overlapped_tensor`: how overlapped/fused weight tensors expose sub-tensor views as CBs with `byte_offset`
- `build_dummy_cb_descriptors()`: generating minimal CB descriptors for all allocated IDs when the real configuration is applied at runtime via reconfig
- `build_cb_reconfig_tensor`: the 264-word-per-core L1 tensor that fused kernels read at startup to reconfigure CB read/write interfaces (addr, size, num_pages, page_size for all 64 CBs plus bitmask and sync semaphores)
- `record_cb_metadata()`: extracting per-CB address, size, page count for reconfig tensor construction
- Source file: `circular_buffer_utils.py`

**Covers questions:** 4, 9

---

## Chapter 3: The Micro-Op Library

**Description:** Provides a detailed reference for every micro-op, explaining its computation, tensor layout requirements, CB setup, and kernel implementation pattern. Organized by functional category.

### Files

#### `01_compute_micro_ops.md`
- **matmul**: Single-core `[1,K] x [K,N]` with MOP-based inner loop, optional fused sigmoid/SiLU activation; optimized for outer dim = 1 tile; uses `unified_kernels/matmul.hpp`
- **rmsnorm**: Single-core RMS normalization with tile-size autodetection (32x32 vs 16x32), epsilon and `inv_sqrt_numel` as runtime args, optional fast rsqrt approximation; uses `unified_kernels/rmsnorm.hpp`
- **rope**: Single-core rotary position embedding using Meta-style interleaving (`rotate_half` via `trans_mat` multiplication), per-core `start_tile_offset` for width-sliced cos/sin reads from DRAM; uses `unified_kernels/rope.hpp`
- **kn_sliced_matmul**: K-sliced matmul where each core computes a K-slice of shared activation against local weights via `k_offset`; used for parallel gate/up projections in shared expert; uses `unified_kernels/kn_sliced_matmul.hpp`
- **local_reduce**: Element-wise reduction of N tiles from a single CB with optional SiLU; face-view optimization groups small tiles (e.g., 16 [1,32] tiles -> 2 [16,16] faces); uses `unified_kernels/local_reduce.hpp`
- **eltwise_add**: Element-wise addition kernel with per-core `sender_index` offset into replicated tensor; uses `unified_kernels/eltwise_add.hpp` and standalone `eltwise_add_kernel.cpp`
- **sampling/argmax**: Multi-core sampling with local winners reduced on a final core; k=1 fast path for argmax; uses `unified_kernels/argmax.hpp`
- **tilize_8x32**: Format conversion for 8x32 tiles
- Source files: all `micro_ops/*/op.py` and their `kernels/` directories for compute ops

#### `02_data_movement_micro_ops.md`
- **gather**: Multi-core to single-core data collection with NOC routing optimization (auto-splits senders between NOC_0 and NOC_1 based on hop distance); supports both rectangular grid mode and scattered core mode (per-core sender indices); uses `unified_kernels/gather.hpp` and `gather_reduce.hpp`
- **mcast**: Single-core to multi-core broadcast using the unified kernel pattern; sender on BRISC, receivers on NCRISC; uses `unified_kernels/mcast.hpp`
- **create_q_heads**: Specialized gather from 12x8 grid to 4x2 grid with tilization; three-phase write: two QNOPE 8x256 halves + one QROPE 8x64; maps sender rows to specific receiver cores for Q head assembly; uses `unified_kernels/create_q_heads.hpp`
- **kv_cache_update**: Reads BFP8 tiles from DRAM, untilizes (BFP8->bfloat16), retilizes (bfloat16->BFP8) for KV cache slot updates; operates on nope cores (kv_rmsnorm) and rope cores (krope); uses `unified_kernels/kv_cache_update.hpp`
- **dram_streaming_matmul**: Multi-core matmul with DRAM-sharded weights; in0 replicated on all cores, in1 WIDTH_SHARDED in DRAM with column-major tile reordering for contiguous K-tile streaming; triple-buffered CB1 for DRAM read pipelining; optional expert indexing (reads K tiles at `expert_idx * K_tiles` offset), optional fused SiLU/mul/scalar chain; per-core `bank_id`/`vc` via `PerCoreCompileTimeDescriptor`; configurable `subblock_k` for K-dimension tiling; `get_max_page_size_and_num_pages()` for NOC burst optimization; uses `unified_kernels/dram_streaming_matmul.hpp`
- Source files: `micro_ops/gather/op.py`, `micro_ops/mcast/op.py`, `micro_ops/create_q_heads/op.py`, `micro_ops/kv_cache_update/op.py`, `micro_ops/dram_streaming_matmul/op.py`

#### `03_communication_and_infrastructure_micro_ops.md`
- **ccl_all_reduce**: Multi-device neighbor exchange and local sum reduction using fabric connections; supports residual tensor fusion; per-device program creation with fabric routing setup; uses `unified_kernels/all_reduce_sender.hpp` and `all_reduce_receiver.hpp`
- **ccl_broadcast**: Multi-device broadcast from sender to all receivers along mesh axis using fabric; supports single-axis and dual-axis (primary + secondary) broadcast on 2D meshes; supports torus and linear topologies; uses `unified_kernels/broadcast.hpp`
- **reduce_to_one_b1**: 3-level reduction tree for 4x2 mesh: LEAF->ROOT3->ROOT2->ROOT1 with cross-column final reduce; each level uses fabric send/receive; device roles (MESH_LEAF, MESH_ROOT3, MESH_ROOT2, MESH_ROOT1) based on mesh coordinate; uses `unified_kernels/reduce_to_one_b1.hpp`
- **sdpa_reduce_to_all**: Cross-device SDPA reduce on torus/ring topology for distributing partial attention statistics (L, S, M tensors) across devices with forwarder and worker kernel roles; uses `unified_kernels/sdpa_reduce_forwarder.hpp` and `sdpa_reduce_worker.hpp`
- **d2d_exchange**: Device-to-device bidirectional data exchange via D2D sockets and fabric connections; `SocketInterface` manages upstream/downstream socket pairs with termination semaphore
- **host_io**: Bidirectional host-device communication via H2D/D2H sockets with termination support; supports loopback mode (CB-based) and socket mode (D2D forwarding); optional fused embedding lookup on incoming tokens
- **pipeline_block**: Multi-host pipeline stage encapsulation; manages D2D socket interfaces for inter-stage forwarding; first stage additionally handles HostInterface with embedding
- **deepseek_moe_gate**: Single-core DeepSeek MoE gating on 16x16 tiles: sigmoid activation + bias addition + hierarchical top-8 selection (sort per group, top-2 sum ranking, final top-8) with score normalization and configurable scaling factor (2.5x); uses `unified_kernels/deepseek_moe_gate.hpp`
- Source files: all communication and infrastructure `micro_ops/*/op.py`

**Covers questions:** 2, 11 (catalog-level)

---

## Chapter 4: The Fused-Op Library

**Description:** Explains how fused ops compose multiple micro-ops into single programs that execute without kernel launch overhead between stages, detailing the fusion strategies, CB layouts, and core grid assignments.

### Files

#### `01_fusion_principles_and_patterns.md`
- Why fuse: eliminate host round-trips, keep data in L1, overlap compute with data movement
- The CB reconfig pattern: dummy CBs at compile time, runtime reconfiguration via `build_cb_reconfig_tensor` (introduced in Chapter 2)
- The `CircularBufferIdManager` context system for CB ID reuse across fused stages
- The face-view optimization (`face_view_utils.py`): converting 1x32 tiles to 16x16 face views for more efficient compute
- The compile-time role pattern: `is_sender`, `is_matmul_core`, `is_gather_core` via `UnifiedCompileTimeCoreDescriptor`
- The `OverlappedTensor` concept: multiple logical tensors sharing a single fused L1 allocation with byte offsets (mechanism detailed in Chapter 7)
- How every fused op follows the pattern: `_setup_dimensions()` -> `_build_compile_time_args()` -> `_build_cb_descriptors()` -> `_build_core_descriptors()` -> `UnifiedKernelDescriptor` -> `ProgramDescriptor` -> `ttnn.generic_op()`

#### `02_attention_fused_ops.md`
- **broadcast_rms**: Fuses CCL broadcast + RMSNorm into a single program; supports `skip_ccl` mode for single-device operation and socket-based input receive; per-device program creation with mesh iteration
- **pre_sdpa**: Fuses CCL Broadcast + RMSNorm + Mcast + Matmul (q_a_proj) + Gather + RMSNorm2 (q_norm) + Mcast2 + Matmul2 (q_b_proj) on a grid of cores; implements the full query projection pipeline producing interleaved QNOPE/QROPE output; 96-core q_ab grid + 18-core KV grid + dedicated sender core (12,9); 15+ CBs
- **kv_cache_branch**: Fuses DKV matmul + RMSNorm + RoPE for compressed KV cache update; produces the full KV cache entry (nope + rope concatenated) from the attention input
- **post_sdpa**: Fuses Matmul4 (kv_b2) + Gather2 + Mcast3 + Matmul5 (o_proj) + Gather3 + CCL All-Reduce; implements kv_b2 projection -> o_proj -> all-reduce with residual add; 13x10 mcast grid with 112 active matmul cores, 8 DRAM workers, 9 phantom cores; CCL receiver on (12,9), CCL sender on (11,9); 14 CBs
- Source files: `fused_ops/broadcast_rms/op.py`, `fused_ops/pre_sdpa/op.py`, `fused_ops/kv_cache_branch/op.py`, `fused_ops/post_sdpa/op.py`

#### `03_moe_fused_ops.md`
- **moe**: Top-level MoE orchestrator composing `MoeRoutedExpertOp` and `MoeSharedExpertOp`; manages `MoeContext` with routed and shared sub-contexts, semaphore creation (17 semaphores via `MoeSem`), IO tensor lists, and CB reconfig tensor for kernel fusion
- **moe_routed_expert**: Full routed expert pipeline as a single fused kernel: input mcast -> gate matmul -> gate gather -> DeepSeek gate (top-8 selection) -> index/scale mcast -> DRAM streaming gate_proj with SiLU -> up_proj -> fused SiLU*mul -> down_proj gather -> down_proj mcast -> down_proj matmul -> eltwise add -> ReduceToOne across devices; device roles (MESH_LEAF, ROOT1/2/3)
- **shared_expert**: Fuses activation mcast + gate/up KN-sliced matmul (128 cores, 64 A-group gate + 64 B-group up) + gather + gated local reduce (SiLU(sum(A)) * sum(B)) + mcast1 + mcast2 + down proj matmul + residual add + output gather; 15 CBs across 130-core grid
- **gated_local_reduce**: Composes three `LocalReduceSilu` phases for gated MLP: reduce(gate) with SiLU, reduce(up) without, then element-wise multiply
- **gated_local_reduce_down_proj**: Extends gated_local_reduce to also include input gather + down projection matmul + residual add + output gather; 13 CBs
- **down_proj**: Fuses Mcast1 + Mcast2 + Matmul + ResidualAdd + Gather on 112-core grid within 13x10 mcast grid; handles DRAM worker and phantom core masking; 8 CBs
- Source files: all MoE `fused_ops/*/op.py`

#### `04_output_fused_ops.md`
- **lm_head_sampling**: Fuses CCL broadcast + mcast + matmul for LM head vocab projection; in multi-device mode broadcasts input from sender device then multicasts from sender core to device grid; each matmul core computes `[1,K] x [K, N_per_core]` with width-sharded vocab weight on 101 cores; supports `skip_ccl` for single-device
- How argmax/sampling integrates with the LM head output for token selection
- Source files: `fused_ops/lm_head_sampling/op.py`, `micro_ops/sampling/op.py`

**Covers questions:** 3, 5 (catalog-level), 6 (catalog-level)

---

## Chapter 5: Multi-head Latent Attention Deep Dive

**Description:** Traces the complete MLA attention mechanism through all its constituent ops, from input normalization through compressed KV cache to attention output projection, providing end-to-end narrative with tensor shapes at each boundary.

### Files

#### `01_mla_query_projection_and_weight_layout.md`
- The MLA query path: input [1, 7168] -> RMSNorm -> q_a_proj [7168, 1536] -> gather -> q_a_norm -> q_b_proj [1536, 12288] -> split into QNOPE (64 heads x 128 dim) and QROPE (64 heads x 64 dim)
- How pre_sdpa implements this as a fused op with interleaved QNOPE/QROPE output layout for efficient SDPA consumption
- Weight shuffling: `shuffle_weights_for_interleaved_qnope_qrope` reorders q_b_proj columns so output is grouped by grid rows (8 QNOPE heads + 8 QROPE heads per row)
- The RoPE application on QROPE: cos/sin DRAM lookup indexed by `position_ids`, Meta-style `rotate_half` via `trans_mat` multiplication
- Q-path weights: q_a_proj (7168, 1536), q_b_proj (1536, 12288) with interleaved shuffle
- KV-path weights: kv_a_proj (7168, 576) with shard reorder, kv_b_proj split into kv_b1 (8192, 512) and kv_b2 (512, 8192) with tile rearrangement
- O-proj weights: (8192, 7168) BFP8 WIDTH_SHARDED on 112 cores
- Normalization gammas: attn_norm (1, 7168), q_norm (1, 1536), kv_norm (1, 512) on dedicated cores

#### `02_compressed_kv_cache.md`
- The MLA KV path: input -> kv_a_proj [7168, 576] -> kv_a_norm -> split nope (512 dim) / rope (64 dim) -> RoPE on rope portion -> concat [nope, rope] -> KV cache update
- How `kv_cache_branch` fuses the DKV matmul + RMSNorm + RoPE into a single program
- KV cache tensor layout: `[batch, 1, max_seq_len, 576]` ND-sharded in DRAM with `shard_height = k_chunk_size` (128)
- The `kv_cache_update` micro-op: BFP8 format conversion cycle (DRAM read -> untilize -> retilize -> DRAM write) for updating a single sequence position
- Optimal DRAM bank ordering: `FlashMLAOptimalGridNOC0` maps S blocks to DRAM banks for locality
- Position tracking via `cur_pos_tensor` (height-sharded, replicated per core) and the `kv_cache_cur_pos_ready` semaphore protocol

#### `03_flash_mla_decode.md`
- The Flash MLA decode algorithm: shared KV tensor where V occupies first `head_dim_v` elements; chunked Q*K^T with online softmax (max tracking, exp, rescaling) and V projection, all fused in a single kernel
- S-block grid architecture: 8 blocks of 8 cores each, laid out across columns 0-3 and 7-10 for optimal DRAM proximity
- Sequence-length parallelism: each Q shard (batch element) gets 8 cores (one per S block), each processing a subset of K chunks
- Tree reduction: 3-step log2(8) reduction across S blocks using semaphore-gated NOC transfers; per-step buffer isolation prevents data corruption from out-of-order completion
- Double-buffered K cache reads: DRAM streaming with 3 semaphores (mcast, receiver_ready, ncrisc_brisc_sync) for overlap between DRAM reads and compute
- `SDPAReduceToAll`: cross-device reduction of partial attention statistics (L, S, M tensors) via forwarder/worker patterns on torus/ring topology

#### `04_post_sdpa_output_projection.md`
- Post-attention: SDPA output [1, 128 per head] -> kv_b2 matmul [512, 128] on 64 cores (5x8 + 12x2 grid) -> gather to (12,9) producing [1, 8192] -> mcast to 130-core grid -> o_proj matmul [8192, 7168] on 112 active cores -> gather to (12,9) producing [1, 7168] -> CCL all-reduce with residual across 2 MLA-TP devices
- How post_sdpa fuses Matmul4 + Gather2 + Mcast3 + Matmul5 + Gather3 + CCL All-Reduce in one program
- The 13x10 mcast grid layout: 112 active matmul cores, 8 DRAM workers, 9 phantom cores at column 12
- CCL all-reduce integration: receiver on gather core (12,9), sender on adjacent core (11,9), fabric routing for cross-device reduction

**Covers questions:** 5

---

## Chapter 6: Mixture-of-Experts Deep Dive

**Description:** Traces the complete MoE subsystem from gating through expert computation to output, covering the 256-expert routing, shared expert fusion, DRAM-streaming matmul internals, and multi-device reduction.

### Files

#### `01_moe_gating_and_routing.md`
- The DeepSeek V3 gating mechanism: 256 experts organized as 16 groups x 16 experts, top-8 routing with bias-corrected scores
- How `deepseek_moe_gate` implements the hierarchical selection: sigmoid on scores, add bias (`e_score_correction_bias`), sort per 16-expert group, rank groups by top-2 sum, flatten top-4 groups to 64 candidates, final top-8 with normalized scores and configurable scaling factor (2.5x)
- The 16x16 tile layout: 256 experts mapped as 16 groups of 16, processed as a single face
- Gate matmul: [1, 7168] x [7168, 256] on 8 cores; gate gather to sender core; gate output: top-8 expert indices and normalized scores
- Dense layers (0-2) vs MoE layers (3-60): `first_k_dense_replace=3`

#### `02_routed_expert_computation.md`
- The routed expert pipeline within `moe_routed_expert`: input mcast -> gate matmul with sigmoid -> gate gather -> DeepSeek gate -> index/scale mcast -> DRAM streaming gate_proj with SiLU -> up_proj -> fused element-wise multiply with scaling -> down_proj gather -> down_proj mcast -> down_proj matmul -> eltwise add
- Expert parallelism: each DRAM core processes one expert's gate/up projection independently using indexed DRAM streaming
- Expert weight layout: 256 experts, each with gate_proj, up_proj, down_proj stored as individual DRAM-sharded tensors with column-major tile reorder for streaming
- The `_MoeRoutedExpertContext` dataclass with its 50+ fields covering CB indices, semaphore addresses, core grids, and setup result dicts
- The ReduceToOne integration: how partial results from 8 devices in the 4x2 mesh are reduced to ROOT1 via the 3-level tree (LEAF -> ROOT3 -> ROOT2 -> ROOT1) with cross-column final reduce via fabric

#### `03_shared_expert_and_combination.md`
- The shared expert path: runs on 128 cores after input mcast from sender core (12,9)
- How `shared_expert` fuses: activation mcast -> gate/up KN-sliced matmul (64 A-cores for gate, 64 B-cores for up) -> gather -> gated local reduce (SiLU(sum(gate)) * sum(up)) -> mcast -> down_proj matmul -> residual add -> output gather
- The `gated_local_reduce` op: three-phase reduction pattern (reduce gate with SiLU, reduce up without, multiply results); face-view optimization for small tiles
- How routed and shared expert outputs combine: element-wise addition of down_proj results on the final output
- MoE semaphore coordination: the 17 `MoeSem` constants and their roles in synchronizing mcast, gather, reduce, and fabric operations
- CB reconfig between routed and shared expert phases: how `build_cb_reconfig_tensor` and `reconfig_cb_interfaces` enable runtime CB reconfiguration

#### `04_dram_streaming_matmul_deep_dive.md`
- The problem DRAM streaming matmul solves: expert weight matrices (e.g., 7168x2048 per expert) are too large for L1; weights must stream from DRAM during computation
- Implementation: input A replicated on all compute cores, input B WIDTH_SHARDED in DRAM with tiles reordered to column-major for K-tile contiguity, triple-buffered DRAM reads with NOC page size optimization
- Column-major tile reordering: within each shard, tiles are pre-shuffled so that K tiles for each N column are contiguous in physical memory, enabling efficient DRAM streaming
- The `subblock_k` parameter: K subblock size for partial accumulation when full K does not fit in the triple-buffer
- Optional features: fused SiLU activation, indexed expert access (`expert_idx * K_tiles` offset), element-wise multiply with gate scores and scalar multiply, looping for multiple iterations
- Bank ID and VC conflict resolution: per-core compile-time args for DRAM bank assignment and virtual channel selection to avoid NOC contention
- Page size optimization: `get_max_page_size_and_num_pages()` calculates optimal NOC burst size (8192 bytes for Wormhole, 16384 bytes for Blackhole)

**Covers questions:** 6, 11

---

## Chapter 7: Weight Preparation and the Blitz Caching System

**Description:** Explains how HuggingFace DeepSeek V3 checkpoints are transformed into device-ready format, how the Blitz weight overlapping system fuses tensors for efficient L1 usage, and how the caching system enables fast model loading.

### Files

#### `01_weight_preparation_pipeline.md`
- `prepare_weights.py` overview: the transformation pipeline from HuggingFace state_dict to device-ready tensors
- Key mapping, transpose (HF `(out_features, in_features)` -> `(K, N)`), kv_b split into kv_b1/kv_b2, norms unsqueeze(0)
- Dataclass hierarchy: `AttentionWeights`, `SharedExpertWeights`, `DenseRoutedExpertWeights`, `MoERoutedExpertWeights` -> `DeepSeekV3DenseLayerWeights`, `DeepSeekV3MoELayerWeights`, `DeepSeekV3EmbeddingLayerWeights`, `DeepSeekV3LMHeadWeights`
- Fusion group mapping (`_FIELD_TO_FUSION_GROUP`): how individual weight fields are assigned to overlap groups (q_ab_kv_a, kv_b12, o_proj_gate_mm_norms, gate_up)
- TP slicing: `_slice_attention_weights_for_mla_tp()` and `_slice_shared_expert_weights_for_moe_tp()` for single-device vs multi-device
- Per-layer serialization: `save_decoder_layer` / `load_dense_decoder_layer` / `load_moe_decoder_layer` with manifest JSON for offline preparation and runtime load
- Expert weight handling: 256 experts per MoE layer with gate_proj, up_proj, down_proj stored as individual DRAM-sharded tensors; `load_moe_routed_experts()` with `setup_fast_dispatch`
- Source files: `prepare_weights.py`

#### `02_blitz_decode_weight_overlapping.md`
- The "overlapping" concept: fusing multiple weight tensors into a single WIDTH_SHARDED tensor sharing the same L1 base address, with sub-weights at known row offsets within the fused shard
- `OverlappedTensor` dataclass: `byte_offset`, `total_size`, `core_range_set`, `tile_shape`, `dtype` for locating sub-tensors within the fused container
- `BlitzDecodeWeights` class: the infrastructure that takes per-field weight specs, groups them by fusion group, and produces fused device tensors
- Overlap spec classes:
  - `QAB_KVA_PROJ_SingleDeviceOverlapSpec`: q_a (packed H/2 x 2W) + q_b (interleaved QNOPE/QROPE) + kv_a (shard reorder) on 96+18 cores
  - `O_PROJ_GATE_MM_RMSNORM_GAMMA_SingleDeviceOverlapSpec`: o_proj (BFP8 on 112 cores) + gate_mm (BFP16 on 8 cores) + 4 norm gammas on dedicated cores
  - `KVB12_PROJ_SingleDeviceOverlapSpec`: kv_b1 on 8x8 grid + kv_b2 tile-rearranged on 64 remaining cores
  - `GATE_UP_PROJ_SingleDeviceOverlapSpec`: BFP4 block-sharded gate/up with shard permutation across 64+64 cores
- Tile-reshape for mixed shard widths: when sub-tensors have different K dimensions, how the overlapping system reshapes tiles to maintain contiguous per-core buffers
- Weight shuffles: q_a pack, q_b interleaved head shuffle, kv_a shard reorder, kv_b2 tile rearrange
- Source files: `blitz_decode_weights.py`

#### `03_cache_serialization_and_special_tensors.md`
- Cache serialization: per-layer save/load with manifest JSON (`_MANIFEST_VERSION`, dtype mapping, per-field metadata including byte offsets and tile shapes)
- The `generate_cache.py` script: modes (dense, moe, experts, embedding, lm_head); uses `LazyStateDict` for memory-efficient HF safetensors loading; verification mode for sanity checks
- Embedding and LM head: `save_embedding_weights()` / `load_embedding_weights()`, `save_lm_head_weights()` / `load_lm_head_weights()` with vocab sharding via `ShardTensorToMesh(dim=1)`
- Gate bias tensor: `create_gate_bias_tensor()` -- `e_score_correction_bias` (256,) reshaped to (16,16), transposed, HEIGHT_SHARDED on sender core (10,9) with 16x16 tile
- Gate indices tensor: `create_gate_indices_tensor()` -- constant 0..255 indices in same layout
- Down proj spec: `DOWN_PROJ_SingleDeviceSpec` -- 112 matmul cores built by excluding DRAM workers and phantom cores from 13x10 grid
- LM head constants: `_LM_HEAD_MATMUL_CORE_GRID` (101 cores), `_LM_HEAD_MCAST_CORE` (10,9)
- Source files: `scripts/generate_cache.py`, `prepare_weights.py`, `blitz_decode_weights.py`

**Covers questions:** 8

---

## Chapter 8: Multi-Device Scaleout, Demo Runtime, and Full Decode Trace

**Description:** Covers how the model scales from a single 4x2 mesh to multi-host Galaxy pod deployments, the demo/inference runtime that ties everything together, and a comprehensive end-to-end decode step trace that synthesizes all previously explained subsystems.

### Files

#### `01_pipeline_parallelism_and_configuration.md`
- Galaxy pod topology: each pod contains 4 hosts, each host manages 4 slices (4x2 mesh each), giving 16 pipeline stages per pod
- Pipeline stage assignment: stage 0 = embedding, stages 1-3 = dense layers, stages 4-61 = MoE layers, stage 62 = LM head + sampling
- Pipeline config YAML files: `stage_to_slice_mapping` that maps pipeline stages to (host, slice) pairs
- Single pod (16 stages on 4 hosts x 4 slices), 2-pod (32 stages on 8 hosts), superpod (64 stages on 16 hosts) configurations
- The `generate_blitz_decode_pipeline_configs.py` script: canonical snake/zigzag host ordering via `sort_hosts_canonical()`, MPI-based physical discovery, rank file and rank binding generation
- Mesh graph descriptor textproto files: define the physical mesh topology for fabric routing
- Source files: `scaleout_configs/*.yaml`, `scaleout_configs/*.textproto`, `scaleout_configs/generate_blitz_decode_pipeline_configs.py`

#### `02_multi_device_communication_and_parallelism.md`
- D2D exchange pattern: upstream socket -> entry node -> processing -> exit node -> downstream socket; each pipeline stage is a `PipelineBlock`
- `PipelineBlock` class: D2D socket interfaces, HostInterface for first/last stage, embedding lookup, termination
- Fabric connections: `ttnn.setup_fabric_connection` and `ttnn.setup_routing_plane_connection` for cross-device data transfers
- The loopback pipeline topology: how `generate_blitz_decode_pipeline()` maps physical ASIC locations to logical mesh coordinates
- Intra-device parallelism within each 4x2 mesh:
  - MLA tensor parallelism (TP=2): q_b_proj and o_proj split across 2 columns; `ccl_all_reduce` sums partial results
  - MoE expert parallelism (EP=8): 256 experts distributed across 8 devices (32 per device); `reduce_to_one_b1` aggregates to ROOT1
  - Device role assignment for ReduceToOne: LEAF (rows 0,3), ROOT3 (row 2), ROOT2 (row 1 col 0), ROOT1 (row 1 col 1)
- Source files: `micro_ops/d2d_exchange/op.py`, `micro_ops/pipeline_block/op.py`, `micro_ops/host_io/op.py`

#### `03_demo_inference_runtime.md`
- `demo/cli.py`: command-line interface (--prompt, --max-new-tokens, --tokenizer, --cache-path, --loopback-mode, --layer-id-offset); `open_mesh_device()` context manager (fabric 2D config, 4x2 mesh shape assertion)
- `demo/runtime.py`: `create_model()` factory (socket setup on `MeshCoreCoord(0,0,0,0)`), `TokenCodec` for int32 token packing/extraction with PCIe alignment
- `demo/runner.py`: the `ModelLike` protocol (start/prefill/decode_step/stop), `run_generation()` orchestrating tokenize -> prefill -> decode loop with streaming text output, `GenerationResult` dataclass
- Multi-host pipeline setup: how `load_weights_from_cache` maps system mesh IDs to layer types (embedding, dense, MoE, LM head) and loads appropriate weight dataclasses; `decoder_layer_id_from_mesh_id()`
- The slow dispatch requirement (`TT_METAL_SLOW_DISPATCH_MODE=1`)
- Source files: `demo/cli.py`, `demo/runner.py`, `demo/runtime.py`

#### `04_full_decode_step_data_flow.md`
- End-to-end annotated trace of a single decode step across the full pipeline, synthesizing all subsystems from Chapters 1-7:
- **Stage 0 (embedding host):** H2D token ID `[B,1]` via HOST_PUSH -> embedding lookup -> D2D forward to stage 1
- **Stages 1-3 (dense layers):** BroadcastRMSNorm (attn_norm) -> PreSDPA (RMSNorm + q_a_proj + q_norm + q_b_proj + interleaved QNOPE/QROPE + KVCacheBranch + FlashMLA) -> PostSDPA (kv_b2 + o_proj + CCL all-reduce with residual) -> BroadcastRMSNorm (ffn_norm) -> shared expert (gate/up matmul + gated reduce + down_proj) -> D2D forward
- **Stages 4-61 (MoE layers):** Same attention path as dense + MoeOp (gate matmul + gate computation + routed expert with 256-expert gating + DRAM streaming gate_proj/up_proj + down_proj + ReduceToOne across 4x2 mesh + shared expert path) -> D2D forward
- **Stage 62 (LM head):** LMHeadSampling (CCL broadcast + mcast + DRAM-streaming vocab matmul + argmax) -> D2H token ID back to host
- Tensor shapes at each boundary (annotated)
- Residual connections: maintained via `eltwise_add`/`residual_add` fused into PostSDPA and MoE down_proj
- Position tracking: `cur_pos` incremented per step, propagated to FlashMLA and RoPE via height-sharded position tensor

**Covers questions:** 7, 10, 12

---

## Conventions

### Terminology

| Term | Definition |
|------|-----------|
| **Micro-op** | A single TT-Blaze operation implemented as a Python class (in `micro_ops/`) with `golden()` and `op()` methods, paired with one or more C++ kernel files. Micro-ops execute as standalone programs via `ttnn.generic_op`. |
| **Fused op** | A composite operation (in `fused_ops/`) that sequences multiple micro-op-level computations in a single program to eliminate kernel launch overhead. Fused ops still use the same `golden()`/`op()` pattern but construct more complex `ProgramDescriptor`s. |
| **Unified kernel** | A single `.hpp` or `.cpp` file compiled three times (for NCRISC, BRISC, TRISC) using `#if defined(COMPILE_FOR_*)` preprocessor guards, enabling all three RISC processors' code to live in one source file. |
| **CB** | Circular Buffer -- on-chip L1 SRAM FIFO used for producer-consumer data flow between RISC processors within a Tensix core. |
| **Tensor-backed CB** | A CB whose L1 address is derived from a pre-allocated sharded tensor (via `cb_descriptor_from_sharded_tensor`), as opposed to a dynamically allocated CB. |
| **CB reconfig** | Runtime reconfiguration of circular buffer addresses/sizes via a sharded L1 tensor, enabling fused ops to override CB layouts between phases. |
| **Overlapped tensor** | Multiple logical weight tensors fused into a single physical WIDTH_SHARDED tensor sharing one L1 base address per core; each sub-tensor at a known byte offset. The "Blitz" weight caching strategy. |
| **Fusion group** | A set of weight tensors that share a single fused/overlapped physical tensor (e.g., `q_ab_kv_a` contains q_a_proj + q_b_proj + kv_a_proj). |
| **DRAM streaming** | A matmul pattern where weight tiles are read directly from DRAM into L1 working buffers via triple-buffered NOC reads, avoiding full weight pre-load. |
| **S-block** | A group of 8 Tensix cores used for sequence-length parallelism in Flash MLA; the implementation uses 8 S blocks for 64 total cores. |
| **Face-view** | Optimization where multiple small tiles (e.g., 16 [1,32] tiles) are reinterpreted as fewer large tiles (e.g., 2 [16,16] faces) for more efficient compute. |
| **Sender core / Gather core** | Core (12, 9) by convention -- serves as the central hub for multicast source and gather destination in most fused ops. |
| **NOC** | Network-on-Chip -- the on-chip interconnect used for data transfers between cores. NOC_0 is used by NCRISC (reader), NOC_1 by BRISC (writer). |
| **Pipeline stage** | One 4x2 mesh device (8 Blackhole chips) in the multi-host pipeline, responsible for one or more transformer layers. |
| **Galaxy pod** | A Tenstorrent cluster unit; single_pod has 16 pipeline stages, 2_pod has 32, superpod has 64. |
| **B1** | Batch-1 -- the decode mode where batch size is fixed at 1 (one sequence being generated). |
| **MLA** | Multi-Latent Attention -- DeepSeek V3's attention variant using compressed KV cache with learned projections. Q is projected through a latent bottleneck (q_a then q_b), and K/V share a compressed representation (kv_a) expanded via kv_b1/kv_b2 projections. |
| **MoE** | Mixture of Experts -- 256-expert routing with top-8 selection and shared expert fusion. |
| **TP** | Tensor Parallelism -- splitting weight matrices across devices (TP=2 within each 4x2 mesh for MLA). |
| **EP** | Expert Parallelism -- distributing MoE experts across devices (EP=8 within each 4x2 mesh). |
| **CCL** | Collective Communication Library -- operations (broadcast, all-reduce, reduce-to-one) that coordinate data across multiple devices via fabric. |
| **BFP4/BFP8/BFP16** | Block floating-point formats: 4/8/16-bit mantissa with per-block shared exponent. |

### Notation

- Tensor shapes use bracket notation: `[dim0, dim1, ...]` with outer dimensions first. Matmul shapes: `[M, K] x [K, N]`.
- Tile dimensions use `(H, W)` convention: `(32, 32)` for standard tiles, `(16, 32)` for half tiles, `(1, 32)` for tiny tiles, `(16, 16)` for face tiles.
- Core coordinates use `(x, y)` logical coordinates (column, row) unless explicitly stated as physical/NOC coordinates.
- Core grids described as `AxB` (e.g., 12x8 = 96 cores, 13x10 = 130 cores).
- Tile counts use subscript-t convention: `Kt = K / tile_width`, `DHt = head_dim / tile_width`.
- CB indices are written as `CB N` (e.g., CB 0, CB 16).
- Data types: BFP4 = `bfloat4_b`, BFP8 = `bfloat8_b`, BFP16 = `bfloat16`, FP32 = `float32`.
- Source paths are relative to `models/demos/deepseek_v3_b1/` unless a full path is given.
- Semaphore IDs referenced by name from the relevant `MoeSem` class or inline constant.

### Formatting Rules

- Each chapter begins with a one-paragraph overview connecting it to the broader implementation.
- Each file begins with a one-paragraph summary of its scope.
- Code references cite specific source file paths and relevant function/class names. For specific locations: `filename:ClassName.method()` or `filename:line_range`.
- Diagrams use ASCII art for core grid layouts, data flow, and reduction trees.
- Core grid diagrams use letter codes: R=matmul, M=mcast sender, G=gather receiver, D=DRAM worker, P=phantom, A=gate compute, B=up compute.
- All numerical constants (tile sizes, buffer counts, semaphore IDs) are traced back to their defining source.
- Golden functions are used as algorithmic specifications: "the op computes the same result as the golden function." Every golden function reference shows the mathematical formula it implements.
- CB layouts are presented as tables mapping CB index to purpose, backing (tensor-backed vs. intermediate), tile format, and which cores use it.
- C++ kernel code quoted in fenced code blocks with `cpp` annotation; Python configuration code with `python` annotation.
- Weight shape transformations shown as pipeline tables: HF shape -> transform -> Blitz shape.

---

## Cross-Chapter Dependencies

| Chapter | Depends On | Nature of Dependency |
|---------|-----------|---------------------|
| Ch 2 (Unified Kernel System) | Ch 1 (Architecture) | Uses the op taxonomy and TT-Blaze execution pattern from Ch 1 to motivate the kernel system design |
| Ch 3 (Micro-Op Library) | Ch 2 (Unified Kernel System) | Every micro-op uses `UnifiedKernelDescriptor`, `kernel_op_api.hpp`, and CB patterns from Ch 2 |
| Ch 4 (Fused-Op Library) | Ch 2, Ch 3 | Fused ops compose micro-ops (Ch 3) using the kernel system (Ch 2); requires understanding of CB reconfig tensors from Ch 2 |
| Ch 5 (MLA Deep Dive) | Ch 3, Ch 4 | Traces through specific micro-ops (flash_mla, rope, rmsnorm, matmul, gather, mcast) and fused ops (pre_sdpa, post_sdpa, kv_cache_branch). Ch 4 provides catalog-level interface; Ch 5 provides end-to-end narrative with tensor shapes. |
| Ch 6 (MoE Deep Dive) | Ch 3, Ch 4 | Traces through micro-ops (deepseek_moe_gate, dram_streaming_matmul, reduce_to_one_b1) and fused ops (moe, moe_routed_expert, shared_expert, down_proj). Ch 4 provides catalog-level interface; Ch 6 provides end-to-end narrative. |
| Ch 7 (Weight Prep & Blitz) | Ch 2, Ch 4, Ch 5, Ch 6 | Weight preparation produces tensors consumed by ops in Ch 3-6; fusion groups map to fused op core grids (Ch 4); weight layout motivations come from MLA (Ch 5) and MoE (Ch 6) specifics. CB descriptor from overlapped tensor depends on CB management (Ch 2). |
| Ch 8 (Scaleout, Runtime, Decode Trace) | All previous | Integrates everything: model interface (Ch 1), ops (Ch 3-4), attention (Ch 5), MoE (Ch 6), weights (Ch 7), and the communication infrastructure. The full decode trace references all prior chapters. |

### Forward References

- Chapter 4 (Fused Ops) mentions `OverlappedTensor` for weight-backed CBs -- the full mechanism is detailed in Chapter 7 (Weight Prep).
- Chapters 5 and 6 both reference CCL All-Reduce -- introduced in Chapter 3 (micro-ops), used in Chapter 4 (post_sdpa and moe fused ops).
- Chapter 8 (Full Decode Trace) references all weight types from Chapter 7 and all fused ops from Chapters 4-6.

### Catalog-vs-Deep-Dive Boundary

Chapters 3-4 (micro-op and fused-op catalogs) and Chapters 5-6 (MLA and MoE deep dives) cover overlapping operations at different levels of abstraction:

- **Chapters 3-4** provide each op's **interface contract**: purpose, inputs/outputs, CB layout table, compile-time/runtime args, core grid, and a brief algorithmic description. These chapters serve as **reference lookups**.
- **Chapters 5-6** provide **end-to-end data flow narratives**: tracing tensor shapes through multiple ops with concrete dimensions, core-grid diagrams, and implementation rationale. These chapters explain **why** the ops are structured as they are and how they compose into complete subsystem pipelines.

### Reading Order Recommendations

1. **Sequential (recommended for first read):** Chapters 1 through 8 in order. Each chapter builds on concepts from earlier chapters.
2. **Attention deep-dive:** Ch 1 -> Ch 2 -> Ch 3 -> Ch 5 (skip fused-op catalog and MoE, come back later).
3. **MoE deep-dive:** Ch 1 -> Ch 2 -> Ch 3 -> Ch 6 (skip fused-op catalog and attention, come back later).
4. **Deployment focus:** Ch 1 -> Ch 7 -> Ch 8 (weight caching, pipeline setup, and runtime; return to kernel chapters as needed).
5. **Kernel engineering:** Ch 2 -> Ch 3 -> Ch 4 (bottom-up from infrastructure to micro-ops to fused ops; skip domain deep dives unless needed).

---

## Question Coverage Matrix

| # | Question | Primary Chapter(s) | Primary File(s) |
|---|----------|--------------------|-----------------|
| 1 | Overall architecture (model.py, forward pass, TT-Blaze ops) | Ch 1 | 01, 02, 03 |
| 2 | Micro-op library (purpose and Python/C++ structure) | Ch 3 | 01, 02, 03 |
| 3 | Fused-op library (fusion strategies, composition) | Ch 4 | 01, 02, 03, 04 |
| 4 | Unified kernel system (descriptor, headers, kernel_op_api.hpp) | Ch 2 | 01, 02 |
| 5 | Attention / MLA (compressed KV cache attention pipeline) | Ch 5 | 01, 02, 03, 04 |
| 6 | MoE subsystem (256-expert routing, shared experts, gating) | Ch 6 | 01, 02, 03 |
| 7 | Multi-device scaleout (pipeline configs, partitioning, CCL) | Ch 8 | 01, 02 |
| 8 | Weight preparation (HF conversion, Blitz caching) | Ch 7 | 01, 02, 03 |
| 9 | Circular buffer management | Ch 2 | 03 |
| 10 | Demo/inference runtime | Ch 8 | 03 |
| 11 | DRAM-streaming matmul | Ch 3 (catalog), Ch 6 (deep dive) | Ch3/02, Ch6/04 |
| 12 | Full decode step data flow | Ch 1 (overview), Ch 8 (comprehensive trace) | Ch1/03, Ch8/04 |
