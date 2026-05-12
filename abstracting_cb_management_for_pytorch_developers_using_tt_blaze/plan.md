# Final Plan: Abstracting CB Management for PyTorch Developers Using TT-Blaze

## Audience

**Primary audience:** PyTorch developers who are proficient with `torch.nn.Module`, `torch.Tensor`, autograd, broadcasting, and standard training/inference workflows, but who have no prior experience with Tenstorrent hardware concepts -- circular buffers, tile-based memory layout, L1 memory budgeting, RISC processor roles, face-order memory layout, or data format negotiation. They understand tensors, shapes, broadcasting, dtype casting, and the PyTorch module composition API. They may have passing familiarity with compilation frameworks (torch.compile, TorchScript, ONNX, CUDA) but have not written Blaze ops before.

**Secondary audience:** Experienced TT-Blaze or TT-Metal developers who currently write manual CB plumbing via FusedProgram.cb_from_tensor, FusedProgram.cb_scratch, and direct emit() calls, and who want to understand how an abstraction layer would formalize decisions they currently make by hand (CB sizing, padding strategy, data format negotiation) and where they could intervene when the abstraction makes suboptimal choices.

**Assumed prerequisite knowledge:**
- Python fluency, PyTorch nn.Module / torch.Tensor API, basic model composition
- Basic understanding of matrix multiplication, broadcasting, shape semantics, and mixed-precision concepts (float32, bfloat16)
- General awareness that hardware accelerators require data in specific layouts (no Tenstorrent specifics required)

**What this guide teaches (not assumed):**
- What circular buffers (CBs) are and why Tenstorrent hardware uses them
- Tile geometry (32x32 default, 1x32 row tiles, face-order layout), tile padding, and TileDescriptor
- L1 memory budgeting, shard specs, CoreRangeSet, and CB ID allocation
- Data format specifics (bfloat8_b, bfloat4_b) and their tile-size implications
- The BlazeOp/MicroOp/FusedOp class hierarchy, FusedProgram.emit(), CT args, and the CBEngine/SemEngine/CTArgEngine pipeline
- How TT-Symbiote's module replacement pattern bridges PyTorch and TTNN

---

## Chapter List

### Chapter 1: The Gap Between PyTorch Tensors and Tenstorrent Hardware

**Description:** Establishes the fundamental impedance mismatch between PyTorch's arbitrary-shape, type-flexible, implicit-broadcast tensor model and Tenstorrent's tile-fixed, CB-mediated, L1-budgeted execution model, motivating the need for an abstraction layer.

**Directory:** `ch01_the_gap/`

**Files:**

- `01_pytorch_tensor_model.md`
  - What a PyTorch tensor is: contiguous memory, arbitrary shapes, strides, dtypes
  - PyTorch's implicit contracts: shape inference, automatic broadcasting, type promotion, transparent memory management
  - How PyTorch developers think about computation: define module, call forward, shapes just work
  - Walk through `torch.nn.Linear(512, 1024)` -- what happens under the hood in standard PyTorch

- `02_tenstorrent_tile_cb_model.md`
  - Tenstorrent hardware at the conceptual level: cores, L1 memory, NOC fabric
  - What circular buffers are: ring-buffered L1 regions with `cb_reserve_back`/`cb_push_back`/`cb_wait_front`/`cb_pop_front` semantics
  - Tile geometry: 32x32 default tiles, 1x32 row tiles, face-order memory layout within tiles
  - Why tiles must be padded to 32-element boundaries (hardware datapaths operate on fixed tile dimensions)
  - CBHandle fields explained: `cb_id`, `num_pages`, `page_size`, `data_format`, `tile_desc`, `core_ranges`, `backing_tensor`, `access_mode` (FIFO vs DIRECT_ADDRESS)
  - Data formats: bfloat16, bfloat8_b, bfloat4_b, float32 -- packed vs unpacked tile sizes
  - Source references: `blaze/cb_handle.py` (CBHandle dataclass), `blaze/cb_engine.py` (DTYPE_BYTES, DEFAULT_TILE_SHAPE)

- `03_the_manual_plumbing_burden.md`
  - Walkthrough of a real Blaze matmul emit() call (from `blaze/ops/matmul/op.py`): every manual decision the developer must make
  - The 12 manual decisions per op: tile shape, padding strategy, padding fill value, data format, num_pages, page_size, core placement, CB ID, semaphore assignment, CT arg wiring, broadcasting pattern, L1 budget check
  - How a compound op like DenseMLP (`blaze/ops/dense_mlp/op.py`) multiplies this complexity across 12 micro-ops
  - Side-by-side comparison table: PyTorch concept vs TT-Blaze equivalent
  - Source references: `blaze/ops/matmul/op.py` (Matmul.emit), `blaze/fused_program.py` (FusedProgram)

- `04_abstraction_goals_and_scope.md`
  - Vision: a "TensorAdapter" layer where the developer writes `output = blaze_matmul(activation, weights)` and the system handles tile decomposition, CB allocation, padding, format selection, and L1 budgeting
  - Comparison table: what the developer specifies vs what the system infers, today and with the abstraction
  - Scope boundaries: what the abstraction handles (shape/tile/CB/padding/format/broadcasting) vs what it does not (kernel authoring, RISC-level programming, custom NOC patterns)
  - Mapping each capability to the specific questions this guide answers (Q1-Q12)
  - Design constraint: the abstraction must compose with existing FusedProgram/CBEngine/BlazeCompiler infrastructure, not replace it

---

### Chapter 2: Design Patterns from Existing Tensor-to-Hardware Abstraction Layers

**Description:** Surveys how TT-Symbiote, TVM, Triton, XLA, MLIR, and PyTorch FX/torch.compile solve the same class of problem, extracting reusable design principles that inform the TensorAdapter architecture defined in Chapter 3.

**Directory:** `ch02_design_patterns/`

**Files:**

- `01_tt_symbiote_module_replacement.md`
  - TT-Symbiote's architecture: `TorchTTNNTensor` as a `torch.Tensor` subclass with `__torch_dispatch__` override
  - Module replacement via `register_module_replacement_dict_with_module_names()`: walking `model._modules`, replacing `nn.Linear` with `TTNNLinear`
  - The dispatcher system: `can_dispatch_to_ttnn()` / `dispatch_to_ttnn()` for ATen op interception
  - `TTNNModule` lifecycle: `from_torch()` -> `preprocess_weights()` -> `move_weights_to_device()` -> `forward()`
  - Strengths: zero-change PyTorch code, transparent fallback. Limitations: operates at TTNN op level (not LLK/Blaze level), no CB-level control
  - Lesson for TensorAdapter: the module-replacement and `__torch_dispatch__` patterns provide the right entry point
  - Source references: `tt_symbiote/core/module.py`, `tt_symbiote/core/tensor.py`, `tt_symbiote/utils/module_replacement.py`

- `02_tvm_relay_tir_scheduling.md`
  - TVM's multi-level IR: Relay (graph-level) -> TIR (tensor-level) -> target code
  - How TVM handles tiling: `te.compute` + `s.tile()` schedule primitives, automatic padding insertion
  - Buffer allocation in TIR: `tir.allocate` with scope annotations, analogous to CB allocation
  - Memory scope annotations (shared, local, global) as a model for CB-level placement
  - Lesson: separate "what to compute" from "how to tile and buffer it" via scheduling

- `03_triton_and_xla_patterns.md`
  - Triton's tile-programming model: `tl.load`/`tl.store` with explicit block shapes, but auto-tuned block sizes
  - How Triton handles padding: `mask` parameters on loads, `other=0.0` or `other=-inf` for softmax
  - XLA/JAX's HLO lowering: shape inference, automatic padding for TPU tile alignment (128x128 for matmul, 8x128 for others)
  - XLA's implicit broadcasting: `BroadcastInDim` HLO op that maps 1-dim to N-dim
  - Lesson: auto-tuning tile/block dimensions based on shapes is proven; padding strategies should be op-specific

- `04_mlir_and_torch_compile.md`
  - MLIR's dialect layering: linalg -> memref -> gpu/scf, progressive lowering with shape metadata preserved at each level
  - torch.compile/Inductor: FX graph capture -> loop-level IR -> kernel codegen with tile size selection
  - How Inductor picks tile sizes: heuristics based on tensor shapes and hardware cache sizes
  - Connection to TT-Forge: TT-Forge's graph compiler does FX capture -> TTNN lowering; a Blaze backend would extend this to LLK-level kernels
  - Lesson: a multi-level representation (logical shape -> padded shape -> tiled shape -> CB layout) is the right architecture

- `05_synthesized_design_principles.md`
  - Seven principles extracted from the survey: (1) separate logical from physical shape, (2) op-specific padding strategies, (3) progressive lowering, (4) auto-tuning over manual specification, (5) escape hatches for power users, (6) transparent dispatch integration, (7) metadata propagation through op chains
  - How each principle maps to a specific TensorAdapter design decision in subsequent chapters
  - Summary comparison table: what each framework does well and where TT-Blaze's abstraction must go further

---

### Chapter 3: The TensorAdapter Architecture and Integration Surface

**Description:** Defines the TensorAdapter's class hierarchy, its three-layer internal architecture, and its precise relationship to TT-Blaze's existing CBHandle/FusedProgram/CBEngine/SemEngine/CTArgEngine/BlazeCompiler infrastructure, specifying what is wrapped, what is extended, and what is left unchanged.

**Directory:** `ch03_tensor_adapter/`

**Files:**

- `01_architecture_overview.md`
  - Three-layer architecture: TensorAdapter (PyTorch-facing) -> ShapeEngine (tiling/padding/format) -> CBBackend (Blaze-facing)
  - TensorAdapter wraps, does not replace, existing Blaze infrastructure: it is a builder that constructs FusedProgram calls
  - Class diagram: `BlazeModule` (extends `torch.nn.Module`) -> captures `forward()` calls -> builds `FusedProgram` via TensorAdapter
  - Core method: `adapt(tensor, *, op_hint=None, padding=None, format=None) -> CBHandle` with attached shape metadata
  - Internal pipeline: `infer_shape -> select_tile -> compute_padding -> select_format -> allocate_cb -> return_handle`
  - What is NOT changed: CBHandle, FusedProgram, CBEngine, SemEngine, CTArgEngine, BlazeOp hierarchy all remain intact

- `02_shape_metadata_system.md`
  - The `ShapeDescriptor` dataclass: `logical_shape`, `padded_shape`, `tile_shape`, `num_tiles_per_dim`, `total_tiles`, `padding_values`, `padding_strategy`
  - Static factory methods: `ShapeDescriptor.from_torch(tensor)` infers all fields from a PyTorch tensor; `ShapeDescriptor.from_ttnn(tensor)` extracts from an on-device tensor
  - The `ShapeTracker`: a companion object that propagates `ShapeDescriptor` through chains of op `emit()` calls
  - How shapes propagate: each op's TensorAdapter method returns a `TypedHandle` (CBHandle + ShapeDescriptor), and the next op reads the ShapeDescriptor to configure its own CBs
  - Shape inference rules per op category: matmul (contract K, produce [M, N]), elementwise (preserve shape), reduction (collapse dimension), broadcast (expand dimension)
  - Relationship to existing structures: TensorPort (has dtype, tile_shape but not logical_shape), CBHandle (has page_size, num_pages but not logical dimensions)

- `03_wrapping_vs_extending_vs_replacing.md`
  - CBHandle: WRAP -- TensorAdapter creates CBHandles with computed parameters; the CBHandle dataclass is unchanged
  - FusedProgram: EXTEND -- TensorAdapter calls FusedProgram.cb_from_tensor, cb_scratch, ncrisc_ct_args, trisc_ct_args; FusedProgram gains no new methods
  - CBEngine: WRAP -- TensorAdapter feeds shaped/typed TensorPorts into CBEngine.assign(); CBEngine's algorithm is unchanged
  - SemEngine: UNCHANGED -- semaphore assignment depends on inter-core topology, not tensor shapes
  - CTArgEngine: WRAP -- TensorAdapter computes user_args (tile counts, blocking factors, broadcast flags) that CTArgEngine resolves into CT arg values
  - BlazeGraph: EXTEND -- TensorAdapter attaches ShapeDescriptor metadata to graph edges
  - BlazeCompiler: WRAP -- TensorAdapter calls BlazeCompiler.compile() with pre-computed tensors and user_args
  - The "adapter as emit() caller" pattern: TensorAdapter's primary job is to compute the arguments that op.emit() needs, then delegate to emit()
  - Source references: `blaze/cb_handle.py`, `blaze/fused_program.py`, `blaze/cb_engine.py`, `blaze/sem_engine.py`, `blaze/ct_args.py`, `blaze/compiler.py`

- `04_interaction_with_graph_and_compilation_paths.md`
  - The FusionContext (`blaze/context.py`) and Graph API (`blaze.fuse()` context manager) as the compilation path
  - Extending FusionResult to carry ShapeDescriptor alongside OpNode + port_name
  - Extending ExternalTensor to carry ShapeDescriptor so shape metadata enters the graph from the start
  - How the BlazeCompiler can use ShapeDescriptor to auto-populate user_args that currently must be passed manually
  - Shadow graph recording: each `f.output()` call records graph edges, enabling the adapter to build the BlazeGraph automatically
  - Source references: `blaze/context.py` (FusionContext, FusionResult, ExternalTensor), `blaze/compiler.py` (BlazeCompiler.compile)

---

### Chapter 4: Automatic Tile Decomposition and Shape Propagation

**Description:** Defines the algorithm for automatically converting arbitrary PyTorch tensor shapes into tile-aligned representations, and designs the mechanism for propagating shape metadata (logical, padded, tiled) through op chains so downstream ops receive correct sizing information without manual specification.

**Directory:** `ch04_tile_decomposition/`

**Files:**

- `01_three_shape_layers.md`
  - Formal definition of the three shape levels: logical_shape (what the user wrote), padded_shape (after alignment to tile boundaries), tile_grid_shape (tile counts per dimension)
  - Worked examples: `[1, 37]` becomes padded `[32, 64]` becomes tile grid `[1, 2]` with 32x32 tiles; `[1, 768]` becomes 24 tiles of 1x32
  - Why all three must be tracked simultaneously: logical for user API, padded for correctness, tiled for CB allocation
  - How TT-Blaze currently computes these: `_tile_page_size()` in `cb_engine.py`, `DEFAULT_TILE_SHAPE = (32, 32)`
  - Source references: `blaze/cb_engine.py`, `blaze/graph.py` (TensorPort.tile_shape)

- `02_automatic_tile_decomposition_algorithm.md`
  - Step-by-step algorithm: given `[A, B, C, D]` and dtype, compute the tiled representation
  - Step 1: Identify tiled dimensions vs batch dimensions (last two dims are tile candidates)
  - Step 2: Select tile shape based on op type: 32x32 for matmul, 1x32 for row-vector ops like RMSNorm (mirrors `interpret_tile()` in `blaze/utils.py`)
  - Step 3: Compute padded dimensions: `padded_H = ceil(H / tile_H) * tile_H`
  - Step 4: Compute `num_pages` and `page_size` from padded dimensions and data format
  - Step 5: Handle higher-rank tensors: batch dimensions become loop bounds or core-mapping dimensions
  - Edge cases: scalar tensors, 1D tensors, tensors smaller than a single tile, non-standard tile shapes (16x32 for RMSNorm reinterpretation)
  - Source references: `blaze/cb_engine.py` (_tile_page_size), `blaze/utils.py` (UNPACKED_DTYPE_BYTES, interpret_tile)

- `03_shape_propagation_through_op_chains.md`
  - The shape propagation protocol: each op declares an `infer_output_shape(input_descriptors)` method
  - Rules for common ops: matmul output shape from input shapes, elementwise ops preserve shape, reduction ops collapse dimensions, broadcast ops expand dimensions
  - How the `TypedHandle` return type wraps `CBHandle` with `ShapeDescriptor` so each op output carries shape metadata
  - Edge cases: ops that change dimensionality (reduce), ops that reshape (view), ops that transpose
  - Handling shape-dependent CT args: `k_num_tiles`, `out_w_per_core`, `num_tiles` are derivable from ShapeDescriptor
  - Handling shape mismatches: when an op's input shape does not match the producer's output shape, the adapter inserts implicit reshape/pad nodes
  - Source references: `blaze/context.py` (FusionResult, FusionContext.add_op), `blaze/ct_args.py` (CTArgSpec)

- `04_shard_shape_and_core_mapping.md`
  - From global tile grid to per-core shard: how the tile grid is partitioned across cores
  - Relationship to `OverlappedView.shard_shape` and `OverlappedView.core_ranges` in `fused_program.py`
  - How `GridConfig` and `pick_matmul_cores()` from `utils.py` inform the shard-to-core mapping
  - The abstraction's role: given a logical shape and a core grid, automatically compute shard_shape, num_pages per core, and page_size
  - Source references: `blaze/fused_program.py` (OverlappedView), `blaze/utils.py` (pick_matmul_cores, GridConfig)

---

### Chapter 5: Transparent Padding Management

**Description:** Designs the automatic padding system that selects the correct padding strategy (zero-pad, negative-infinity pad, mask-based, identity-element) based on op semantics, removing the need for developers to specify padding per op while maintaining numerical correctness across fused op chains.

**Directory:** `ch05_padding/`

**Files:**

- `01_padding_strategy_taxonomy.md`
  - Zero-padding: the default for matmul and most compute ops -- padded elements contribute zero to the result
  - Negative-infinity padding: required for softmax/SDPA to ensure padded positions produce zero attention weights
  - Mask-based padding: for reductions where padded elements must be excluded from the count (e.g., mean reduction over a non-tile-aligned dimension)
  - Identity-element padding: 1.0 for multiplication reductions, 0.0 for addition reductions
  - No-padding: when tile alignment is already guaranteed by the model architecture
  - Current manual approach: PaddedRMSNorm op has explicit padded_input/padded_gamma Internal CBs; developers must create separate ops for padded vs non-padded variants
  - Source references: `blaze/ops/padded_rmsnorm/op.py`, `blaze/ops/sdpa/op.py` (causal masking constants)

- `02_op_padding_registry.md`
  - The `PaddingPolicy` enum: `ZERO`, `NEG_INF`, `MASK`, `IDENTITY`, `NONE`
  - Registry design: each Blaze op declares its required padding strategy per input port
  - Extending `TensorPort` (from `graph.py`) with a `padding_strategy` field
  - Default policies for all current Blaze ops, derived from examining each op's kernel semantics
  - Per-input-port padding: a matmul's in0 (activations) may need different padding than in1 (weights)
  - How the `OpSpec` registration in `blaze_op.py` can carry padding metadata alongside existing port metadata
  - Source references: `blaze/blaze_op.py` (OpSpec), `blaze/graph.py` (TensorPort)

- `03_automatic_padding_insertion_and_propagation.md`
  - The padding pass: after shape inference, walk the graph and insert padding ops where the logical shape does not match the padded shape
  - Padding op implementation: a lightweight MicroOp that writes padding values to the extra tile positions
  - Optimization: when the producer op can produce padded output directly (common for tiled tensor reads), skip the separate padding op
  - Chain propagation: if op A pads `[7, 2048]` to `[32, 2048]` and produces `[32, 64]`, op B sees padded input; the logical shape `[7, 64]` must be tracked for final unpadding
  - When a fused chain crosses padding boundaries (e.g., matmul output feeds softmax with different padding needs), the adapter inserts a repadding step
  - Source references: `blaze/ops/padded_rmsnorm/op.py` (existing padding pattern), `blaze/utils.py` (interpret_tile_padded)

- `04_output_unpadding.md`
  - When the final output returns to PyTorch, the abstraction must strip padding to restore the logical shape
  - Implementation: track the padding metadata through the op chain and apply a slice operation at the boundary
  - Performance consideration: unpadding is a host-side operation that should not add device overhead
  - Mask tensors: for attention patterns, the padding mask itself must be padded consistently with the data

---

### Chapter 6: Data Format Selection and Automatic Negotiation

**Description:** Explains TT-Blaze's data format landscape and designs an automatic format selection and cross-op negotiation system that picks optimal formats per op and inserts format conversion ops where producers and consumers disagree.

**Directory:** `ch06_data_formats/`

**Files:**

- `01_data_format_landscape.md`
  - PyTorch dtype world: torch.float32, torch.bfloat16, torch.float16, torch.int8
  - TT-Blaze format world: ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.uint8/16/32
  - Block-floating-point formats (bfp8_b, bfp4_b): what they are, why they matter for weight compression
  - Per-format trade-offs: precision, memory footprint, compute throughput
  - How format affects page_size: `DTYPE_BYTES` mapping from `cb_engine.py`, and `UNPACKED_DTYPE_BYTES` from `utils.py`
  - Which ops prefer which formats: matmul prefers BFloat8_b or BFloat4_b for weights; RMSNorm may want Float32 accumulation (`fp32_dest_acc_en`)
  - Source references: `blaze/cb_engine.py` (DTYPE_BYTES, _TTNN_DTYPE_MAP), `blaze/utils.py` (UNPACKED_DTYPE_BYTES)

- `02_format_negotiation_protocol.md`
  - Design: each op declares preferred input/output formats, plus a "tolerable" set
  - Negotiation algorithm: walk the graph, propagate preferred formats forward from producer to consumer, insert conversion nodes where no overlap exists
  - The cost model: format conversion has a latency cost; the negotiator minimizes total conversions while respecting op preferences
  - Fallback policy: when no preference is declared, default to bfloat16 (matching TT-Blaze's DEFAULT_DTYPE)
  - Tile reinterpretation: `cb_from_view` in FusedProgram supports tile override, enabling zero-cost format aliasing when tile geometries are compatible
  - Source references: `blaze/cb_reconfig.py` (CircularBufferIdManager), `blaze/cb_handle.py` (data_format field)

- `03_format_conversion_and_cb_reconfig.md`
  - How format conversion maps to existing Blaze primitives: `tile_row_convert` op, `retilize` op, or in-CB reconfig
  - The `CircularBufferIdManager` (`cb_reconfig.py`): how CB IDs can be reused across formats within different phases
  - `cb_reconfig_builder.py`: the existing infrastructure for reconfiguring CB formats between phases of a fused op
  - When conversion is free: reading BFloat8_b weights into a BFloat16 compute path happens at the SFPU level, no explicit conversion op needed
  - Source references: `blaze/cb_reconfig.py`, `blaze/cb_reconfig_builder.py`

- `04_precision_profiles_and_user_overrides.md`
  - Three built-in precision profiles: "performance" (Bfp8_b for weights, BFloat16 for activations, LoFi math), "balanced" (BFloat16 everywhere, HiFi4 math), "accuracy" (Float32 accumulation, fp32_dest_acc_en)
  - How profiles interact with per-op overrides: a developer can set a global profile but override specific ops
  - Escape hatch: `with blaze.format_hint(dtype=ttnn.bfloat8_b): ...` context manager to force a format at a boundary
  - Relationship to existing Blaze `math_fidelity` and `math_approx_mode` fields on BlazeOp
  - Source references: `blaze/blaze_op.py` (math_fidelity, math_approx_mode), `blaze/fused_program.py` (math_fidelity parameter)

---

### Chapter 7: Automatic CB Sizing and L1 Memory Budgeting

**Description:** Designs the system for automatically determining CB sizes (num_pages, page_size) and blocking strategies based on tensor shapes, op requirements, and available L1 memory, replacing the manual calculations that currently happen in each op's emit() method.

**Directory:** `ch07_cb_sizing/`

**Files:**

- `01_l1_memory_model.md`
  - L1 memory on Tenstorrent: per-core SRAM (~1.5 MB usable on Blackhole), shared among all CBs, scratch buffers, semaphores, and stack
  - The 64 CB slot hardware limit (`MAX_CB_ID = 64` in `cb_engine.py`)
  - What competes for L1: CB pages, scratch buffers, stack/firmware, semaphores, runtime metadata
  - How current Blaze ops budget L1: `FusedProgram.cb_scratch()` allocates `num_pages * page_size` bytes
  - Why L1 overcommit crashes silently: Metal's allocator does not bounds-check at compile time
  - The `l1_profile.py` module: existing tooling for CB address reporting and L1 utilization analysis
  - Source references: `blaze/l1_profile.py`, `blaze/cb_engine.py` (MAX_CB_ID)

- `02_automatic_page_count_and_page_size.md`
  - Page size determination: `page_size = tile_h * tile_w * dtype_bytes` (from `_tile_page_size()` in `cb_engine.py`)
  - num_pages for streaming ops: typically 1-2 pages for double-buffering
  - num_pages for matmul: K dimension determines how many pages to buffer -- `k_num_tiles` from `matmul/op.py`
  - num_pages for reduction: accumulation across tiles requires holding partial results
  - The sizing algorithm: (1) compute minimum num_pages per CB (1 for streaming, all for accumulation), (2) if total exceeds L1, reduce accumulation CB pages to fit, (3) if still exceeds, switch to multi-pass tiling
  - Double-buffering: allocating `2 * KT_DIM` pages hides data movement latency behind compute
  - Source references: `blaze/fused_program.py` (cb_from_tensor, cb_scratch), `blaze/cb_engine.py` (CBAssignment)

- `03_blocking_strategy_selection.md`
  - Matmul blocking dimensions: RT_DIM (row tiles), CT_DIM (column tiles), KT_DIM (accumulation tiles) define the tile-loop structure
  - For matmul: given M, K, N dimensions and available L1, choose the blocking factor that maximizes compute utilization while fitting in L1
  - `compute_subblock_w()` from `utils.py`: existing logic for choosing compute subblock dimensions
  - Trade-off: larger blocks = fewer NOC transactions but more L1 pressure; smaller blocks = more transactions but fit in tighter L1
  - Algorithm: enumerate feasible `(RT, CT, KT)` triples, maximize `RT * CT * KT` (throughput proxy) subject to L1 constraint
  - Extension to other ops: RMSNorm uses `num_tiles` for accumulation; softmax/SDPA have their own blocking patterns
  - Source references: `blaze/ops/matmul/op.py` (out_w_per_core = in1_cb.num_pages // k_num_tiles), `blaze/utils.py` (compute_subblock_w)

- `04_cb_compaction_and_temporal_reuse.md`
  - TT-Blaze's existing `CBEngine.compact_cb_ids()`: interval coloring to reuse CB IDs across non-overlapping lifetimes
  - `CBAssignment.first_use` and `last_use` fields for lifetime tracking
  - `CircularBufferIdManager`: format-aware reuse across multi-phase programs via `cb_reconfig.py`
  - The compact mode: `CBEngine(compact=True)` minimizes CB ID usage
  - How the ShapeTracker provides lifetime information automatically, reducing need for manual annotation
  - The 64 CB ID hardware limit: why compaction matters for large fused ops like MoE layers
  - Source references: `blaze/cb_engine.py` (compact_cb_ids), `blaze/cb_reconfig.py` (CircularBufferIdManager)

---

### Chapter 8: Broadcasting and Multi-Dimensional Shape Alignment

**Description:** Maps PyTorch's broadcasting rules to TT-Blaze's explicit bcast_rows/bcast_cols/bcast_scalar patterns and the physical multicast (Mcast) mechanism, with automatic detection and insertion.

**Directory:** `ch08_broadcasting/`

**Files:**

- `01_pytorch_broadcasting_rules.md`
  - PyTorch broadcasting semantics: right-align shapes, expand dimensions of size 1, missing dimensions are prepended as size 1
  - Examples relevant to LLM workloads: bias addition `[1, D] + [B, D]`, attention mask `[1, 1, S, S] * [B, H, S, S]`, layer norm scale `[D] * [B, S, D]`
  - What the developer expects: write `a + b` and have it work regardless of shape combinations

- `02_mapping_to_blaze_broadcast_primitives.md`
  - TT-Blaze broadcast primitives: `bcast_rows` (replicate across rows), `bcast_cols` (replicate across columns), `bcast_scalar` (single value to all positions)
  - Mapping algorithm: compare `ShapeDescriptor` of left and right operands, detect which dimensions are size 1, select the matching bcast pattern
  - When shapes are incompatible with any single bcast pattern: decompose into multiple ops
  - Mcast (multicast) as physical broadcast: when data must move from one core to all cores, `Mcast.emit()` handles the NOC transfer
  - The distinction between logical broadcast (shape expansion) and physical broadcast (data movement via Mcast/Gather)
  - When broadcasting requires an explicit mcast op (cross-core replication) vs when it can be handled within the compute kernel (per-tile replication)
  - Source references: `blaze/ops/mcast/op.py`, `blaze/ops/broadcast_rmsnorm/op.py`, `blaze/cb_engine.py` (_union_grids)

- `03_broadcast_shape_tracking_and_validation.md`
  - After broadcasting, the output ShapeDescriptor must reflect the expanded shape, not the original
  - How `TypedHandle` carries both the physical CB layout (unchanged) and the logical broadcast shape (expanded)
  - Optimization: when broadcast is along the tile's inner dimension, no extra CB is needed -- the compute kernel reads the same tile repeatedly
  - How broadcasting affects CB sizing: the broadcast source only needs 1 tile page (reused across iterations), while the non-broadcast input needs full pages
  - Validation: TensorAdapter rejects broadcasts that would require shapes not representable in the tile model
  - Integration with CT args: the resolver emits appropriate `is_broadcast`, `bcast_type` CT args that Blaze ops expect

---

### Chapter 9: Op Fusion with Shape, Padding, and Format Tracking

**Description:** Shows how the abstraction supports fusing multiple PyTorch ops into a single TT-Blaze FusedOp while maintaining correct shape, padding, and format tracking across the fused boundary, and demonstrates the fusion model with concrete examples.

**Directory:** `ch09_op_fusion/`

**Files:**

- `01_blaze_fusion_model.md`
  - TT-Blaze's two-level composition: `MicroOp` (single C++ kernel phase) vs `FusedOp` (composition of multiple MicroOps via `compose()` and `emit()` chains)
  - The `blaze.fuse()` context manager and `FusionContext`: captures op calls, builds `BlazeGraph`, runs validation
  - Graph API vs Composition API: `blaze.matmul(...)` (graph node creation) vs `Matmul.emit(f, ...)` (FusedProgram composition)
  - How `build_gated_mlp_graph()` in `_gated_mlp.py` composes 11 ops into a single fused graph -- the template for automatic fusion
  - How `SwigluOp.compose()` chains `DRAMStreamingMatmul`, `EltwiseMul`, `Gather`, `Mcast`, `DRAMStreamingMatmul`
  - Source references: `blaze/blaze_op.py` (FusedOp, MicroOp, compose), `blaze/_gated_mlp.py`, `blaze/ops/swiglu/op.py`

- `02_shape_and_padding_across_fused_boundaries.md`
  - Problem: when matmul + silu + rmsnorm are fused into a single FusedOp, padding changes at each boundary
  - The ShapeTracker must update at each emit() call and propagate to the next sub-op's adapt() call
  - Padding semantics at fused boundaries: if sub-op A produces zero-padded output but sub-op B needs neg-inf padding, the abstraction injects a pad-transform between them
  - Optimization: when the same padding strategy applies across consecutive ops, skip redundant re-padding
  - Handling `reconfig()` boundaries: when `FusedProgram.reconfig()` is called, the abstraction checkpoints its shape/format state and transitions to the next phase

- `03_format_transitions_within_fused_ops.md`
  - Within a fused chain, format changes happen via `cb_alias()` (reinterpret same buffer with different format descriptor) or by allocating a new scratch CB with conversion
  - The FormatPolicy negotiation applied per edge in the fused subgraph
  - Example: matmul produces `Bfp8_b`, RMSNorm requires `BFloat16`; the abstraction inserts a format-conversion micro-op or uses `fp32_dest_acc_en`
  - Format consistency: format conversions within a fused op use CB reconfig rather than separate ops
  - Source references: `blaze/cb_reconfig.py`, `blaze/cb_reconfig_builder.py`

- `04_fusion_detection_and_limitations.md`
  - Identifying fusible op sequences: the abstraction's "fusion analyzer" identifies patterns (e.g., linear + activation + residual_add) and maps them to existing Blaze fused ops (e.g., gated_reduce, dense_mlp, swiglu)
  - Limitations: not all PyTorch op sequences can be fused (control flow, dynamic shapes, unsupported ops); the adapter falls back to sequential single-op execution
  - Testing strategy: golden comparison against PyTorch CPU reference for every fused pattern
  - The op mapping registry: a table from PyTorch op names to Blaze op names with shape/format/padding rules

---

### Chapter 10: End-to-End Developer API, Performance, and Escape Hatches

**Description:** Presents the complete developer-facing API from PyTorch module to hardware execution, demonstrates it with worked examples, quantifies where automation may sacrifice performance, and documents the escape hatches available for power users.

**Directory:** `ch10_end_to_end/`

**Files:**

- `01_the_blaze_module_api.md`
  - Top-level API: `blaze.from_pytorch(module, sample_input, device=mesh_device) -> BlazeModule`
  - `BlazeModule.forward(input_tensor) -> output_tensor` with zero manual CB, padding, or tile management
  - Under the hood: the adapter traces the module, builds a BlazeGraph, runs CBEngine + SemEngine + CTArgEngine, compiles via BlazeCompiler, and executes
  - Configuration knobs: `blaze.from_pytorch(module, sample_input, math_fidelity="LoFi", target_format=ttnn.bfloat8_b)`
  - Weight handling: `preprocess_weights()` and `move_weights_to_device()` following the TTNNModule lifecycle pattern
  - How `BlazeModule` relates to TT-Symbiote's `TTNNModule`: BlazeModule operates at the Blaze/LLK level (fused kernels on L1), while TTNNModule operates at the TTNN op level

- `02_worked_example_linear_layer.md`
  - Starting point: `torch.nn.Linear(in_features=4096, out_features=11008)`
  - What the adapter does: decomposes `[1, 4096]` activation into tiles, selects BFloat16, allocates input CB with K tiles, allocates weight CB (direct address from sharded tensor), allocates output CB, sets blocking to optimal `(KT_DIM, CT_DIM)`
  - What the developer writes: `blaze_linear = BlazeLinear.from_torch(nn.Linear(...)); output = blaze_linear(input_tensor)`
  - Side-by-side: the 25 lines of manual `Matmul.emit()` code vs the 3-line `BlazeModule` call

- `03_worked_example_gated_mlp.md`
  - Starting point: a standard gated MLP with SiLU activation (gate/up projection, element-wise multiply, down projection)
  - What the adapter does: chains matmul, broadcast, element-wise, and reduction adapters; manages intermediate CBs automatically; handles gather/mcast for cross-core data movement
  - Comparison with `build_gated_mlp_graph()` in `blaze/_gated_mlp.py`: the 11-node graph is built identically but the developer specifies only tensor shapes
  - Performance comparison: adapter-generated vs hand-tuned `SwigluOp`

- `04_worked_example_attention.md`
  - Starting point: multi-head attention with Q/K/V projections, SDPA, output projection
  - Challenges: multiple interacting tensor shapes, broadcasting for attention masking, format transitions (high precision for softmax, low precision for matmuls)
  - How the adapter handles each challenge: ShapeTracker through Q/K/V projections, broadcast detection for mask, FormatPolicy for softmax boundary
  - The gap: attention patterns the current abstraction cannot fully automate (paged KV cache, dynamic sequence length masking) and how escape hatches address them

- `05_performance_implications_and_tradeoffs.md`
  - Where automation costs performance: (1) conservative CB sizing may leave L1 underutilized, (2) format negotiation may insert unnecessary conversions, (3) generic padding ops may be slower than op-specific padding, (4) broadcast inference may choose mcast when per-core replicate suffices
  - Quantifying the cost: framework overhead is compile-time (graph construction, engine passes), not runtime -- the generated kernel is identical to a hand-written one for correctly configured CBs
  - Where automation matches hand-tuned: CB ID assignment, semaphore allocation, core grid placement, CT arg derivation
  - Where hand-tuning wins: very tight L1 budgets (e.g., 60+ CBs in a MoE layer), non-standard tile geometries, custom NOC patterns
  - Expected performance profile: automation within 5-15% of hand-tuned for standard shapes
  - Using `l1_profile.py` output to validate automatic sizing stays within L1 bounds
  - Source references: `blaze/l1_profile.py` (print_cb_stats, print_kernel_stats)

- `06_escape_hatches_for_power_users.md`
  - Override level 1 (per-op): `@blaze.override(cb_sizing={"in0": {"num_pages": 4}}, format={"in1": ttnn.bfloat4_b})`
  - Override level 2 (per-graph): provide a custom user_args dict to BlazeCompiler.compile() that takes precedence over ShapeDescriptor-derived values
  - Override level 3 (bypass entirely): drop down to raw FusedProgram API for performance-critical ops, while using the abstraction for the rest
  - The hybrid model: mix automatic and manual ops within the same FusedProgram
  - The `ExternalTensor` escape hatch from `context.py`: pass pre-configured tensors to bypass automatic tile/format/padding decisions
  - The expert developer workflow: start with TensorAdapter for rapid prototyping, profile with l1_profile, identify bottlenecks, surgically override with escape hatches
  - Source references: `blaze/context.py` (ExternalTensor), `blaze/fused_program.py` (raw API)

- `07_migration_guide.md`
  - For existing Blaze users: how to incrementally adopt the adapter (mix manual and automatic ops in the same graph)
  - For PyTorch users new to Tenstorrent: the three-step flow: (1) define PyTorch model, (2) wrap with BlazeModule, (3) run
  - For TT-Symbiote users: mapping between TTNNModule patterns and BlazeModule patterns; when to use Symbiote (TTNN-level) vs Blaze abstraction (LLK-level)
  - Common pitfalls: shapes that do not tile evenly, formats that lose precision, L1 budget exhaustion on large models

---

## Conventions

### Terminology

| Term | Definition |
|------|-----------|
| **Logical shape** | The tensor shape as specified by the PyTorch developer, e.g., `[batch, seq_len, hidden_dim]`. No tile alignment assumed. |
| **Padded shape** | The logical shape after rounding the last two dimensions up to tile boundaries (multiples of tile_h and tile_w). |
| **Tile grid shape** | The number of tiles along each dimension: `(padded_H // tile_h, padded_W // tile_w)`. |
| **CB (Circular Buffer)** | A hardware-level FIFO in L1 SRAM identified by a cb_id (0-63), the mechanism by which data flows between RISC processors on a Tenstorrent core. |
| **CBHandle** | Python dataclass (`blaze/cb_handle.py`) referencing a CB with its metadata: cb_id, num_pages, page_size, core_ranges, data_format, tile_desc, backing_tensor, access_mode. |
| **Page** | One tile's worth of data in a CB. `page_size = tile_h * tile_w * dtype_bytes`. |
| **num_pages** | How many pages (tiles) a CB can hold simultaneously. Determines buffering depth. |
| **FusedProgram** | The composition context (`blaze/fused_program.py`) where ops are chained by passing CBHandles between emit() calls. |
| **BlazeGraph** | The dataflow IR (`blaze/graph.py`) consisting of OpNodes connected by Edges, with external tensor inputs. |
| **MicroOp** | An atomic op backed by a single C++ kernel header (`blaze/blaze_op.py`). The smallest unit of compute. |
| **FusedOp** | A composite op that chains MicroOps via compose() to produce a single kernel dispatch. |
| **TensorAdapter** | The proposed abstraction layer that converts PyTorch tensors and ops to Blaze's tile/CB model. |
| **ShapeDescriptor** | The proposed metadata object carrying logical_shape, padded_shape, tile_grid_shape, padding_strategy, and related fields through the op chain. |
| **ShapeTracker** | A companion object that propagates ShapeDescriptor through chains of op emit() calls within a FusedProgram. |
| **TypedHandle** | Proposed wrapper: CBHandle + ShapeDescriptor, returned by TensorAdapter methods to propagate shape metadata between ops. |
| **BlazeModule** | The proposed torch.nn.Module subclass that uses TensorAdapter internally to compile and execute on Tenstorrent hardware. |
| **Format negotiation** | The graph pass that determines data formats at each edge and inserts conversions where producer and consumer formats differ. |
| **CT arg** | Compile-time argument: a constant baked into the kernel binary, resolved by CTArgEngine from CB assignments, semaphore assignments, grid coordinates, or user parameters. |
| **L1** | Level-1 SRAM on each Tensix core (~1.5 MB on Blackhole), used for CBs, kernel text, and stack. |
| **Blocking** | The tile-loop structure of an op: which dimensions are looped over (streamed) vs accumulated in-place. `KT_DIM`, `CT_DIM`, `RT_DIM` for matmul. |
| **Shard** | Per-core slice of a tensor distributed across cores; `shard_shape` defines the local tile footprint. |

### Notation

- File paths reference the TT-Blaze repository at `/localdev/salnahari/testing_dir/tt-blaze/` unless otherwise noted. Paths are written repository-relative: `blaze/cb_handle.py`.
- TT-Metal references use `/localdev/salnahari/testing_dir/tt-metal/`.
- TT-Symbiote references use `/localdev/salnahari/testing_dir/tt-metal/models/experimental/tt_symbiote/`.
- PyTorch shapes are written in bracket notation: `[batch, seq, hidden]` or `[M, K]`.
- Tile shapes are written as `(H, W)` tuples: `(32, 32)`, `(1, 32)`.
- CB sizing is written as `num_pages x page_size` bytes: `4 x 2048B`.
- Data formats use their ttnn names: `bfloat16`, `bfloat8_b`, `float32`.
- Code snippets from the codebase are attributed with their source file path.
- Proposed API designs use Python type annotations and are marked as `# PROPOSED` to distinguish from existing code (`# EXISTING`).
- All dimension sizes are in elements unless explicitly stated as "bytes" or "tiles".

### Formatting Rules

- Each file begins with a level-1 heading (`#`) matching the file title.
- Each file opens with a one-paragraph summary and a "What you will learn" bullet list.
- Code examples use Python with type annotations; hardware-side code (C++ kernel headers) is shown only when essential.
- Diagrams are described in ASCII art within fenced code blocks.
- Design proposals include a "Current State" section (what exists today) and a "Proposed State" section (what the abstraction would add or change).
- Performance-sensitive sections include a "Cost Model" subsection explaining the runtime or compile-time overhead.
- Every design decision includes a "Rationale" paragraph explaining why this approach was chosen over alternatives.
- Each file ends with a "Key Takeaways" section (3-5 bullet points) and a "Source Files" section listing actual codebase paths referenced.
- Cross-references to other files use the format `[Chapter N: File Title](../chNN_dir/NN_file.md)`.
- Tables use GitHub-Flavored Markdown pipe syntax.
- Warning/caution blocks use blockquote with bold prefix: `> **Warning:** ...`.

---

## Cross-Chapter Dependencies

```
Chapter 1 (The Gap)
  |
  +---> Chapter 2 (Design Patterns) -----> Chapter 3 (TensorAdapter Architecture)
  |                                              |
  |                                              +---> Chapter 4 (Tile Decomposition)
  |                                              |         |
  |                                              |         +---> Chapter 5 (Padding)
  |                                              |
  |                                              +---> Chapter 6 (Data Formats)
  |                                              |         |
  |                                              +---> Chapter 7 (CB Sizing) <--- Ch4, Ch5, Ch6
  |                                              |
  |                                              +---> Chapter 8 (Broadcasting) <--- Ch4
  |                                                        |
  +---> Chapter 9 (Op Fusion) <--- depends on Ch3, Ch4, Ch5, Ch6, Ch7, Ch8
          |
          +---> Chapter 10 (End-to-End API) <--- depends on all prior chapters
```

**Detailed dependency notes:**

| Chapter | Depends On | Nature of Dependency |
|---------|-----------|---------------------|
| Chapter 2 (Design Patterns) | Chapter 1 (The Gap) | Chapter 1 establishes the problem space and vocabulary; Chapter 2 uses this to contextualize how other frameworks solved the same problems. |
| Chapter 3 (TensorAdapter) | Chapters 1, 2 | Chapter 1 provides the gap analysis; Chapter 2 provides the design principles that inform architectural decisions. Chapter 3 introduces ShapeDescriptor, TypedHandle, and the three-layer architecture that all subsequent chapters elaborate. |
| Chapter 4 (Tile Decomposition) | Chapters 1, 3 | Chapter 1 introduces tile geometry; Chapter 3 defines the ShapeDescriptor that Chapter 4 populates. |
| Chapter 5 (Padding) | Chapter 4 | Padding strategies operate on the difference between logical_shape and padded_shape as computed in Chapter 4. |
| Chapter 6 (Data Formats) | Chapters 3, 4 | Format selection affects page_size, which depends on tile dimensions (Ch4). Format negotiation uses the integration surface defined in Ch3. |
| Chapter 7 (CB Sizing) | Chapters 4, 5, 6 | CB sizing depends on tile_shape (Ch4), padding extent (Ch5), and data_format (Ch6). num_pages and page_size consume all three. |
| Chapter 8 (Broadcasting) | Chapters 3, 4 | Broadcast detection compares ShapeDescriptors (Ch3/Ch4). Broadcasting affects CB sizing through single-page reuse patterns (soft dependency on Ch7). |
| Chapter 9 (Op Fusion) | Chapters 3-8 | Fusion synthesizes all preceding subsystems: TensorAdapter architecture, shape tracking, padding transitions, format transitions, CB sizing, and broadcast handling within fused kernels. |
| Chapter 10 (End-to-End) | All prior chapters | The end-to-end API is the user-facing composition of all abstractions. Escape hatches reference specific components from each prior chapter. |

**Soft dependencies:**
- Chapter 5 (Padding) <-> Chapter 6 (Data Formats): Padding fill values must be representable in the chosen device format; -inf in BFloat16 vs Float32 uses different bit patterns.
- Chapter 7 (CB Sizing) <-> Chapter 8 (Broadcasting): Broadcast patterns affect CB num_pages (broadcast source uses 1 page vs full pages); the budget model must account for this.

### Reading Order Recommendations

- **PyTorch developers new to hardware acceleration:** Read Chapters 1 through 10 sequentially. Chapter 1 is essential context.
- **Developers familiar with TT-Blaze but wanting the abstraction:** Skim Chapter 1, read Chapter 2 for patterns, read Chapter 3 for the architecture, then the specific chapter for the subsystem of interest, then Chapter 10 for the API.
- **Contributors to the abstraction layer:** Read Chapter 3 first for the integration surface, then the specific chapter for the subsystem you are implementing.
- **Performance engineers:** Start with Chapter 10 files 05-06 (Performance Implications, Escape Hatches), then read Chapter 7 (CB Sizing) for the budget allocation algorithm.

---

## Selection Rationale

This final plan is a hybrid that draws on the strongest elements of all five candidate plans, informed by the evaluations' consistent feedback. Here is how each plan contributed:

### From Plan 2: Dedicated Early Framework Survey (Chapter 2)

All five evaluators identified the framework survey placement as a critical structural decision. Plan 2's dedicated Chapter 2 with five files covering TT-Symbiote, TVM, Triton, XLA, MLIR, and torch.compile -- plus a synthesized-principles file -- was consistently praised as the strongest treatment of Q2 and Q11. This structure is adopted directly. The evaluations of Plans 1, 3, 4, and 5 all flagged their framework survey placement as a weakness (scattered in integration chapters or buried at the end). Plan 2's approach of surveying patterns *before* defining the architecture ensures the reader has the comparative context needed to evaluate design choices as they encounter them.

### From Plan 2: Three-Layer TensorAdapter Architecture (Chapter 3)

Plan 2's three-layer architecture concept (TensorAdapter -> ShapeEngine -> CBBackend) provides clearer internal structure than the single-class TensorAdapter approach used by Plans 1, 3, and 4. This is combined with Plan 4's wrap/extend/replace integration analysis (Chapter 7, file 02), which was highlighted by the Plan 4 evaluator as "the most architecturally precise across all five plans." The result is Chapter 3, which defines the architecture *and* its integration surface in one place.

### From Plan 4: Explicit Wrap/Extend/Replace Analysis (Chapter 3, File 03)

Plan 4 uniquely provided a component-by-component decision (CBHandle: WRAP, FusedProgram: EXTEND, CBEngine: WRAP, SemEngine: UNCHANGED, CTArgEngine: WRAP, BlazeGraph: EXTEND, BlazeCompiler: WRAP). This granular analysis directly answers Q8 (integration surface) and was singled out by the evaluator. It is incorporated as a dedicated file in the TensorAdapter chapter.

### From Plans 1 and 3: Separate Chapters for Padding, Tile Decomposition, and Formats

Multiple evaluators flagged that Plans 2 and 5 merged tile decomposition with padding, and formats with CB sizing, resulting in insufficient depth. Plan 1 and Plan 3 both dedicated separate chapters to tile decomposition, padding, format negotiation, and CB sizing. This separation is adopted here (Chapters 4, 5, 6, 7), giving each topic the space it needs. Plan 1's four-file padding chapter structure (taxonomy, registry, insertion, unpadding) is the model for Chapter 5.

### From Plan 3: Precision Profiles Concept (Chapter 6, File 04)

Plan 3 introduced "precision profiles" (performance, balanced, accuracy) as a user-facing concept for format selection. The Plan 3 evaluator noted this was "unique but underspecified." This plan expands it into a full file within the data formats chapter, connecting profiles to existing `math_fidelity` and `math_approx_mode` fields.

### From Plan 5: Dedicated Op Fusion Chapter (Chapter 9)

Plan 5 was the only plan that gave op fusion its own full chapter with four files. Every evaluator for the other plans noted that fusion coverage was "thin" or "compressed." The Plan 5 evaluator confirmed this was a strength. This plan adopts the standalone fusion chapter structure, incorporating elements from Plan 1's correctness-under-fusion analysis (padding consistency, shape consistency, format consistency).

### From Plan 5: Worked Examples and Migration Guide (Chapter 10)

Plan 5's four worked examples (linear layer, gated MLP, attention, migration guide) were identified as the strongest end-to-end content across all plans. The Plan 5 evaluator noted this was "the most example-rich plan of the five," and the Plan 1 evaluator recommended adding similar examples. These are adopted as files in Chapter 10, providing concrete demonstrations of the full abstraction pipeline.

### From Plan 5: Named Implementation Classes

Plan 5's named classes (TileDecomposer, ShapeTracker, AutoBlocker, BroadcastResolver, FormatPolicy, BlazeModule) make the plan the most implementation-ready. Key naming choices (ShapeDescriptor, ShapeTracker, TypedHandle, BlazeModule) are adopted throughout.

### From Plan 1: Cross-Chapter Dependency Documentation

Plan 1's cross-chapter dependency table with "nature of dependency" descriptions was rated as "the best of all plans" by its evaluator. This plan adopts a similar detailed dependency table, augmented with soft-dependency annotations from Plan 3 and a visual dependency graph from Plans 2 and 4.

### From Plan 1: Reading Order Recommendations

Plan 1 uniquely provided four distinct reading-order recommendations for different audience profiles. This is adopted and extended for the 10-chapter structure.

### Structural Decision: 10 Chapters

All five evaluators recommended splitting overloaded chapters. The common modifications requested across all evaluations were: (1) separate padding from tile decomposition, (2) separate formats from CB sizing, (3) separate fusion from integration/API, (4) expand performance/escape hatches. Applying all of these to an 8-chapter base naturally yields 10 chapters, which provides adequate depth for each of the 12 research questions without excessive fragmentation. Each chapter remains focused on a single coherent theme.
