# Plan: Tiny Tiles — Sub-32-Row Tile Geometry from LLK to TT-Blaze

---

## Audience

This guide is written for **TT-Metal / TT-Blaze kernel authors, attention-op designers, LLK developers, and inference-stack engineers** who need to reason about sub-32-row tile geometry on Tenstorrent Tensix cores — when to use it, how to declare it, how the hardware actually handles it, and how it propagates through fused programs.

**Prerequisites the reader should have:**
- Working knowledge of the canonical Tensix 32x32 tile model (four 16x16 faces, row-major within each face). The [[introduction_to_tt_llk]] guide (`ch3_data_organization/tiles_and_faces.md`) is sufficient background.
- Familiarity with the LLK three-thread (unpack / math / pack) kernel structure and the role of MOP replay buffers — the level of the [[minimum_requirements_to_run_a_matmul_on_p150_using_pure_llk]] guide.
- Awareness of TT-Blaze's MicroOp / FusedOp / CBHandle / CBEngine model — the level of the [[comprehensive_guide_to_creating_new_micro_ops_in_tt_blaze]] guide. Readers unfamiliar with Blaze can read Chapters 1–5 standalone but will need that background for Chapter 5 onward.
- A conceptual understanding of attention (Q/K/V, multi-head, batch-1 decode) and online softmax. Detailed Flash MLA derivation is covered in [[comprehensive_understanding_of_deepseek_v3_b1]] (`ch05_multi_head_latent_attention_deep_dive/`).
- Comfort reading C++17 templates, preprocessor-define-driven configuration, and Python op-emit code.

**No prior experience with tiny tiles is assumed.** The guide builds the concept from the hardware constraints up.

---

## Chapter List

### Chapter 1: What Is a Tiny Tile

**Description:** Establishes the precise definition of a tiny tile, the legal geometries the Tensix data path supports, and the place of tiny tiles in the broader tile-format taxonomy. Anchors the rest of the guide.

**Directory:** `ch1_what_is_a_tiny_tile/`

**Files:**

- `01_canonical_tile_and_face_recap.md`
  - Recap of the canonical 32x32 tile composed of four 16x16 faces (F0–F3), stored row-major within each face.
  - Constants from `ckernel_defs.h`: `TILE_R_DIM=32`, `TILE_C_DIM=32`, `FACE_R_DIM=16`, `FACE_C_DIM=16`.
  - Why the hardware face is 16x16 (matrix-unit MVMUL granularity, source-register layout, packer stride logic).
  - Cross-reference: [[introduction_to_tt_llk]] ch3_data_organization/tiles_and_faces.md.

- `02_tiny_tile_definition_and_legal_geometries.md`
  - Definition: a **tiny tile** is a tile with `num_faces ∈ {1, 2, 4}` and `face_r_dim ∈ {1, 2, 4, 8, 16}`, where `face_c_dim` is fixed at 16. Effective tile heights: 1, 2, 4, 8, 16, or full 32 rows; effective widths: 16 or 32 columns.
  - The validator: `validate_tensor_shape_tile_dependent_ops_(const TensorShape&)` — full text and explanation.
  - Enumerate the practically observed shapes in the field: 1x32, 2x32, 4x32, 8x32, 16x32, and the rare 32x16 / 16x16 column-reduced variants.
  - Hardware vs API legality: which `(num_faces, face_r_dim)` combinations are physically representable in the unpacker/math/packer data path vs. which are merely API-legal and trap at runtime.
  - The "16x16 by 16x16 matmul is not supported" assertion — what it means and why it exists.

- `03_why_tiny_tiles_exist.md`
  - The motivation: batch-1 decode produces "tall-thin" activation shapes where the row dimension is naturally < 32 (e.g., 8 attention heads per core, 1 sequence position).
  - Padding to 32 rows wastes 75%+ of dest-register capacity, L1 footprint, and matmul cycles.
  - Hardware support for sub-32-row tiles is the right escape hatch — but it introduces geometry that must be threaded through every CB descriptor, kernel CT arg, and consumer op in the program.
  - Where tiny tiles are load-bearing in production: Flash MLA Q tile, RoPE shard mapping, sampling/argmax narrow outputs, gated local-reduce intermediates.

---

### Chapter 2: The LLK Data Path with Tiny Tiles

**Description:** Walks through what changes inside the unpacker, math engine, and packer when the tile is smaller than 32 rows. Documents the SFPU iteration count rule, dest-register occupancy, and the throttle / transpose / partial-face / DstSync interactions.

**Directory:** `ch2_llk_data_path/`

**Files:**

- `01_unpacker_with_tiny_tiles.md`
  - How the unpacker is told about the tile geometry: `face_r_dim_A`, `face_r_dim_B`, `num_faces_A`, `num_faces_B`, `partial_face_A`, `partial_face_B` template arguments to `_llk_unpack_AB_matmul_init_<>()`.
  - What changes in the unpack MOP when `face_r_dim < 16`: shorter inner loops, modified address-counter strides, `partial_face` mode activation.
  - The haloize-mode interaction with transpose and tiny tiles.
  - Source-register population pattern: how an 8x32 tile is laid out in SrcA / SrcB compared to a 32x32 tile.
  - References: `tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h`, `tests/sources/unpack_matmul_test.cpp`.

- `02_math_engine_with_tiny_tiles.md`
  - How `_llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>()` is parameterized by `tile_r_dim` / `tile_c_dim` and how this drives the MVMUL inner-loop count.
  - Reuse_a vs. reuse_b strategy under tiny tiles: which axis is reused changes the optimal MOP structure when ct_dim or rt_dim is small.
  - Fidelity-phase scaling: a HiFi4 tiny-tile matmul still runs 4 fidelity passes — what this means for cycle count.
  - Dest-register occupancy: an 8x32 tile occupies the same dest "slot" as a 32x32 tile (slots are tile-granular, not row-granular), so tiny tiles do NOT free additional dest slots for parallel tile production. This is a frequent source of misunderstanding.
  - References: `tt_llk_blackhole/llk_lib/llk_math_matmul.h`, `tests/sources/math_matmul_test.cpp`.

- `03_packer_with_tiny_tiles.md`
  - How the packer is configured for sub-32-row output: `_llk_pack_init_<...>(pack_dst_format, ...)` and the role of the third template parameter on Blackhole.
  - Stride and address-modifier programming for partial-row writes.
  - The L1 tile-size byte count: how `tile.get_tile_size(dtype)` shrinks for tiny tiles, and the alignment constraints on the resulting size (16-byte L1 granularity).
  - Pack-untilize / pack-tilize compatibility with tiny tiles — what's supported, what's not.
  - References: `tt_llk_blackhole/llk_lib/llk_pack.h`, `tests/sources/matmul_pack_untilize_test.cpp`.

- `04_sfpu_iteration_count_rule.md`
  - The rule: SFPU operations process the tile face-by-face, so the iteration argument N to `llk_math_eltwise_unary_sfpu_*<approx_mode, false, N>(...)` is the number of 16-row faces the tile contains.
  - Lookup table:
    - 1x32 → 2 faces (F0, F1) → N=2
    - 2x32 → 2 faces → N=2
    - 4x32 → 2 faces → N=2
    - 8x32 → 2 faces → N=2
    - 16x32 → 2 faces → N=2
    - 32x32 → 4 faces → N=4
    - 16x16 (1-face) → 1 face → N=1
  - Worked example: the `<approx_mode, false, 2>` template argument in `unified_kernels/matmul.hpp` for the fused `sigmoid` / `silu` on a 1x32 output — why 2, not 1.
  - The PACK-thread-runs-SFPU pattern (`PACK(ckernel::llk_math_eltwise_unary_sfpu_sigmoid<...>(0, (int)VectorMode::R))`) and why it composes with tiny tiles unchanged.

- `05_dstsync_dest_acc_and_throttle.md`
  - DstSync::SyncHalf vs. SyncFull with tiny tiles: total dest-slot count is unchanged (still 8/16 in 16-bit mode, 4/8 in fp32 mode), tile size in bytes per slot is unchanged at the hardware level (the slot is sized for a full tile and partially populated).
  - `is_fp32_dest_acc_en` (and `dst_full_sync_en` tracking it) — no special tiny-tile interaction, but the L1-vs-dest sizing arithmetic must use the **logical** tile size for L1 and the **hardware-tile** size for dest budgeting.
  - Throttle levels (0–5) and their interaction with sub-32 inner loops: lower-fidelity throttle becomes proportionally more impactful per cycle when the inner loop is short.
  - Stochastic rounding compatibility (Fpu, Pack, All modes) with tiny tiles — documented coverage from `test_unpack_matmul.py`.

---

### Chapter 3: LLK Test Infrastructure for Tiny Tiles

**Description:** Maps the LLK matmul test suite's tiny-tile coverage — which drivers exercise tiny tiles, what shapes they sweep, and where the coverage gaps are.

**Directory:** `ch3_llk_test_infrastructure/`

**Files:**

- `01_tiny_tile_test_matrix.md`
  - The five drivers that exercise tiny tiles:
    - `test_math_matmul.py` / `math_matmul_test.cpp` — multi-tile + tiny tiles, throttle 1–5, transpose, partial faces.
    - `test_unpack_matmul.py` / `unpack_matmul_test.cpp` — multi-tile + tiny tiles, stochastic rounding, face modes (1/2/4).
    - `perf_math_matmul.py` / `math_matmul_perf.cpp` — extended perf with tiny tiles and partial faces (LOOP_FACTOR=1024).
  - The two drivers that do NOT exercise tiny tiles and why: `test_matmul.py` (canonical baseline, 32x32 only) and `test_matmul_custom.py` (no-MOP experimental, 32x32 only).
  - Quasar coverage status (`quasar/matmul_quasar_test.cpp`).

- `02_generate_matmul_tiny_tiles_combinations.md`
  - Full walkthrough of the `generate_matmul_tiny_tiles_combinations()` helper.
  - The table of sweep configurations (in0_tile_r_dim ∈ {1, 2, 4, 8, 16} with in1 fixed at 32x32, face modes 2/4).
  - Why tiny tiles are restricted to **in0 only** in the matmul test matrix: in1 width is the dominant unpacker stride, and a tiny in1 changes the math-engine inner loop structure in ways that the current test drivers don't cover.
  - Comparison to `generate_matmul_dimension_combinations()` (standard 32x32 sweep): how the two helpers compose in the Python driver.

- `03_pcc_golden_comparison_for_tiny_tiles.md`
  - How `MatmulGolden` computes the reference output when the input shape is tiny.
  - Tilization of tiny inputs: `tilize_block` must know the `(face_r_dim, num_faces)` of the destination layout to produce a comparable golden.
  - PCC thresholds and known precision differences vs. full-tile matmul.
  - References: `tests/python_tests/test_math_matmul.py`, `tests/python_tests/helpers/matmul_sweep.py`.

- `04_coverage_gaps.md`
  - Operations that lack standalone tiny-tile test coverage in tt-llk:
    - Eltwise unary (SFPU) standalone tests at 8x32 / 16x32 — coverage is only via matmul-fused SFPU.
    - Reduce / transpose standalone tests at tiny geometry.
    - Tilize / untilize at tiny geometry.
    - Quasar tiny-tile matmul.
  - Recommended new test sources to close the gaps (with naming conventions and Python driver pattern).

---

### Chapter 4: Tiny Tiles in TT-Metal Tensor and CB Metadata

**Description:** Documents how tiny-tile geometry is declared, propagated, and validated through TT-Metal's tensor API, tile descriptors, and circular-buffer descriptors. This is the API surface kernel authors touch most often.

**Directory:** `ch4_tt_metal_metadata/`

**Files:**

- `01_ttnn_tile_and_tile_descriptor.md`
  - `ttnn.Tile((H, W))` — the host-side tile constructor, accepted (H, W) pairs.
  - `tile.get_tile_size(dtype)` — byte-size computation as a function of (H, W) and dtype, including BFP8 shared-exponent overhead.
  - `ttnn.TileDescriptor(tile)` — the CB-facing descriptor wrapper.
  - How `Tile.tile_shape[0]` (height) and `[1]` (width) feed into shard-shape math.
  - Worked example: deriving Flash MLA's `q_tiny_tile = ttnn.Tile((Q_TILE_HEIGHT, TILE_WIDTH))` where `Q_TILE_HEIGHT=8`.

- `02_cb_descriptor_from_sharded_tensor.md`
  - The `cb_descriptor_from_sharded_tensor(cb_id, tensor)` flow — how it reads the tensor's tile shape and shard shape to compute page count and page size.
  - The L1 byte-budget formula: `num_pages * tile.get_tile_size(dtype)` (plus alignment).
  - What changes when the input tensor's tile is tiny: the CB page size shrinks proportionally, but the **number of pages** is determined by the consumer's needs (double-buffering, accumulation depth), not by the tile size.
  - References: DeepSeek V3 B1 `micro_ops/flash_mla/op.py` lines ~537–555 and ~712–714.

- `03_tile_shape_propagation_through_a_program.md`
  - The propagation rule: a producer op's output tile geometry MUST match every downstream consumer's input tile geometry, modulo explicit re-tiling.
  - How TT-Blaze's CBHandle chain enforces this implicitly (the consumer reads the producer's `ttnn.Tile` from the shared CB descriptor).
  - Where the chain breaks down: ops that hand-construct tile descriptors (e.g., Flash MLA's intermediate stats CBs) must explicitly match the producer's geometry — this is a frequent source of bugs.
  - Decision tree: when an op outputs a tile of shape X, which CB descriptors downstream must be re-derived?

- `04_tile_shape_in_kernel_ct_args.md`
  - How tile geometry surfaces in kernel CT args: `TILE_HEIGHT`, `FACE_R_DIM`, `NUM_FACES`, often baked into `CoreCTArgs` or `ComputeCTArgs` structs in the C++ header.
  - The `cpp_parser.py` extraction path (TT-Blaze) — how the C++ header's CT arg fields are surfaced back to Python.
  - Worked example: Flash MLA's CT args carrying `Q_TILE_HEIGHT` and `K_TILE_HEIGHT` separately so the kernel can mix tiny-Q and standard-K geometry.

---

### Chapter 5: Tiny Tiles in DeepSeek V3 B1 — Flash MLA, RoPE, and Friends

**Description:** Deep dive into how tiny tiles are used in the canonical production reference. Centers on Flash MLA decode (the most complex tiny-tile user), then surveys RoPE, create_q_heads, KV-cache-branch, local-reduce, DRAM-streaming matmul, and the fused attention pipeline.

**Directory:** `ch5_deepseek_v3_b1_usage/`

**Files:**

- `01_flash_mla_q_tile_design.md`
  - Why Q is tiled as 8x32: each core handles `n_heads_per_core=8` heads, and the per-position activation row count is exactly 8 — treating each head as a "row" creates a natural 8x32 tile.
  - The `PNHt=1`, `Q_TILE_HEIGHT=8`, `DHt` parameter set and how it maps to model dimensions (head_dim=576 for DeepSeek MLA → DHt=18 BFP8 tiles).
  - K/V kept at standard 32x32 to maximize matmul throughput on the KV side (cache lines are 32-row aligned anyway).
  - The mixed-geometry matmul: `sdpa_custom_mm_block(Q[8,32]_tiles, K_chunk[32,32]_tiles, transpose_k=true)`.

- `02_flash_mla_cb_chain_walkthrough.md`
  - Full CB table for Flash MLA decode (from `ch05_multi_head_latent_attention_deep_dive/03_flash_mla_decode.md`):
    | CB | Tile geometry | Pages | L1 size | Role |
    |---|---|---|---|---|
    | cb_q_in | 8x32 (tiny) | 18 | 9,216 B | Q input |
    | cb_k_in | 32x32 BFP8 | 144 | 156,672 B | K chunks, double-buffered |
    | cb_mask | 8x32 (tiny) | 1 | 512 B | Causal mask |
    | cb_ms_in | 8x32 (tiny) | 3 | 1,536 B | Tree-reduction m/s stats |
    | cb_out_in | 8x32 (tiny) | 48 | 24,576 B | Tree-reduction O data |
    | cb_out_o | 8x32 (tiny) | 16 | 8,192 B | Compute output O |
    | cb_out_ms | 8x32 (tiny) | 1 | 512 B | Compute m/s stats |
    | cb_interm_out | 8x32 (tiny) | 16 | 8,192 B | Intermediate O (aliased) |
    | cb_interm_ms | 8x32 (tiny) | 1 | 512 B | Intermediate m/s (aliased) |
    | cb_out_final | 8x32 (tiny) | per spec | 8,192 B | Final sharded output |
  - The CB-aliasing pattern (`cb_out_o` and `cb_interm_out` share L1) and why aliasing only works because both consumers see the same tile geometry.
  - L1 budget comparison: 217 KB total with tiny Q vs. ~280 KB if Q were padded to 32x32 (everything in the Q chain inflates 4x). The mask, stats, and intermediate CBs are the biggest savings.

- `03_rope_tile_geometry.md`
  - How `RopeOp.emit()` derives the tile shape from the input tensor's shard shape:
    - `shard_shape[0] = num_q_heads_per_core` → tile height
    - `shard_shape[1] = head_dim` → tile width
    - `head_dim_per_core_t = shard_shape[1] // 32` → tile-column count
    - `tile = ttnn.Tile((num_q_heads_per_core, ttnn.TILE_SIZE))`
  - Intermediate buffer sizing: `num_interm_tiles = head_dim_per_core_t`.
  - Why RoPE's tile geometry must agree with Flash MLA's Q tile (RoPE feeds Q into MLA via the CB chain) — and what breaks if `num_q_heads_per_core != Q_TILE_HEIGHT`.
  - References: `micro_ops/rope/op.py` lines ~113–127, `tests/unit_tests/test_rope.py`.

- `04_create_q_heads_and_kv_cache_branch.md`
  - `create_q_heads`: produces the 8x32 Q sharded layout that Flash MLA consumes. Tile geometry handoff via the output CB.
  - `kv_cache_branch`: how the KV cache (stored as 32x32 tiles) is read into Flash MLA's K input. Tile geometry remains 32x32 here — the boundary between standard and tiny tiles is at the Q side, not the K side.
  - The `attention_block` fused op as the joining point that compiles both halves into one program.
  - References: `fused_ops/attention_block/op.py`, `fused_ops/pre_sdpa/op.py`, `tests/unit_tests/test_kv_cache_branch.py`.

- `05_local_reduce_and_gated_paths.md`
  - `local_reduce` / `reduce_to_one_b1` with tiny intermediates — when the reduction operates over 8-row tiles vs. 32-row tiles.
  - `gated_local_reduce` and `gated_local_reduce_down_proj`: how the gate path's narrow output uses tiny tiles to match the activation pipeline upstream.
  - The `down_proj` fused op and its mixed tile geometry.
  - References: `tests/unit_tests/test_local_reduce.py`, `fused_ops/pre_sdpa/op.py`.

- `06_dram_streaming_matmul_and_unified_kernels.md`
  - The `dram_streaming_matmul` micro-op: how it streams 32x32 weight tiles from DRAM but produces tiny-tile outputs when the LM-head or projection output is narrow.
  - The `unified_kernels/matmul.hpp` fused-activation path: the `<approx_mode, false, 2>` SFPU iteration count, what it means for a 1x32 output, and how it generalizes to other tiny output shapes.
  - References: `tests/unit_tests/test_dram_streaming_matmul.py`, `unified_kernels/matmul.hpp` lines ~166–180.

---

### Chapter 6: Trade-offs — Tiny Tiles vs. Pad-and-Mask

**Description:** Decision-making chapter. When is a tiny tile the right answer? When should you pad to 32x32 and accept the waste? Quantitative comparisons where possible, qualitative trade-offs where not.

**Directory:** `ch6_tradeoffs/`

**Files:**

- `01_decision_tree.md`
  - Decision criteria, in order of typical decisiveness:
    1. **L1 pressure** — if the activation chain has many double-buffered intermediates, tiny tiles can be the difference between fitting and not fitting in L1.
    2. **Effective-row count** — if the meaningful row count is ≥ 16 (e.g., 24-head attention), padding to 32 wastes only 25% and may be worth it to avoid geometry plumbing.
    3. **Downstream consumer geometry** — if the consumer already uses 32x32 (e.g., a downstream matmul against a full weight matrix), padding earlier may be simpler than re-tiling.
    4. **Dest-register reuse** — tiny tiles do NOT free dest slots, so a chain producing many tiny tiles may underutilize dest if not carefully scheduled.
    5. **Compute-bound vs. bandwidth-bound** — bandwidth-bound ops benefit most from tiny tiles (smaller L1 footprint → faster fills). Compute-bound ops see less benefit because MVMUL cycle counts don't shrink linearly with tile rows.
  - Flowchart converting the criteria into a recommendation.

- `02_l1_budget_quantitative_comparison.md`
  - Flash MLA L1 footprint computed two ways: with tiny Q (~217 KB) vs. padded Q (~280 KB). Per-CB breakdown.
  - Why the savings are not linear in the row ratio: K/V (the dominant CB) is unchanged because it's standard-tile, but every Q-side CB shrinks 4x.
  - L1 budget on Blackhole P150 (~1.3 MB usable) — fraction consumed in each configuration.
  - Cross-arch contrast (Wormhole B0 L1 size, if different).

- `03_perf_delta_measurements.md`
  - Methodology: how to compare tiny-tile vs. pad-and-mask perf using `math_matmul_perf.cpp` and the DeepSeek V3 B1 demo runner.
  - Expected results from the existing benchmarks (cycle counts for 1x32, 2x32, 4x32, 8x32, 16x32, 32x32 matmul at LoFi / HiFi2 / HiFi4).
  - The perf knee: at which row count does pad-and-mask become faster than the tiny-tile path? (Hypothesis: somewhere between 16 and 24 rows for compute-bound ops; tiny tiles always win for bandwidth-bound ops because L1 traffic scales linearly.)

- `04_failure_modes_and_misuse.md`
  - Symptoms and root causes for the common tiny-tile bugs:
    - **Hang on cb_wait_front**: CB page-size mismatch between producer (tiny) and consumer (full) — consumer waits forever for a page that never arrives.
    - **PCC mismatch with no hang**: SFPU iteration count wrong for the row count — SFPU processes the wrong number of faces, leaves garbage in unprocessed rows.
    - **Illegal-config trap**: `(num_faces, face_r_dim)` combination that the validator accepts but the hardware mishandles (rare; check `validate_tensor_shape_tile_dependent_ops_` precisely matches what the unpacker/packer actually support per arch).
    - **Silent corruption**: transpose + tiny tile + partial face combination, where the test matrix didn't cover the exact combo.
    - **Dest-slot starvation**: a chain producing many tiny tiles back-to-back fills dest slots that could have held full tiles; reorder or pad if needed.
  - Debugging checklist: BLAZE_DEBUG_KERNELS, BLAZE_L1_PROFILE, TT_METAL_DPRINT_CORES, named_args_generated.h inspection.

---

### Chapter 7: Cross-Cutting Concerns

**Description:** Concerns that span the stack and the architectures: CCL fabric alignment, host-side weight prep, multi-device propagation, per-arch differences (BH / WH-B0 / Quasar), and integration with the broader Blaze infrastructure.

**Directory:** `ch7_cross_cutting/`

**Files:**

- `01_cross_architecture_support.md`
  - Per-arch support matrix for tiny tiles:
    - Blackhole (P150 / P300): full support, all `(num_faces, face_r_dim)` combinations validated.
    - Wormhole B0: full support, minor pack-API differences (no third template parameter).
    - Quasar: status unclear — `quasar/matmul_quasar_test.cpp` does not exercise tiny tiles; this chapter audits what the Quasar LLK layer actually supports.
  - Source-code locations to inspect per arch: `tt_llk_blackhole/llk_lib/`, `tt_llk_wormhole_b0/llk_lib/`, `tt_llk_quasar/llk_lib/`.

- `02_multi_device_ccl_and_fabric.md`
  - When a tiny-tile activation crosses a chip boundary via `ccl_all_reduce`, `ccl_broadcast`, `d2d_exchange`, or `reduce_to_one_b1`: how the fabric packet size is computed (header + tile payload).
  - Alignment constraints: fabric packets have minimum-size and stride alignment requirements that may force padding even if the local tile is tiny.
  - Bandwidth implications: tiny tiles reduce per-packet payload, increasing header overhead ratio — when this matters and when it doesn't.
  - Cross-reference: [[comprehensive_guide_to_creating_new_micro_ops_in_tt_blaze]] ch06_inter_core/03_ccl_and_fabric.md.

- `03_host_side_weight_preparation.md`
  - When an op consumes a tiny-tile **activation** but its **weight** is a standard 32x32 tile: the activation tile geometry does NOT propagate to the weight layout. Weights stay in their canonical 32x32 DRAM layout.
  - When does `prepare_weights.py` need to know about tiny tiles? Almost never — only if the weight itself is structurally < 32 rows (rare; e.g., LM-head output projection where vocab dimension is the row axis).
  - Blitz cache implications: weights are cached at their on-device layout, which is independent of activation tile geometry.
  - References: DeepSeek V3 B1 `scripts/prepare_weights.py`, `scripts/blitz_decode_weights.py`.

- `04_blaze_pipeline_propagation.md`
  - Tile geometry across pipeline stages: a stage that produces tiny-tile outputs must have its socket / D2D-exchange protocol set up with the right page size (the bulk D2D-socket page size = tile bytes × pages per shard).
  - PipelineGraph metadata: does the BlazeGraph capture tile geometry per edge? (Yes via CBHandle's TileDescriptor.) How is it serialized for multi-host pipelines?
  - Cross-reference: [[comprehensive_guide_to_tenstorrent_sockets_d2d_h2d_and_d2h_communication]].

---

### Chapter 8: Developer Ergonomics, Tooling, and Next Steps

**Description:** Recommendations for making tiny tiles less error-prone, the gaps in tooling that should be closed, and the future direction of tiny-tile support.

**Directory:** `ch8_ergonomics_and_next_steps/`

**Files:**

- `01_developer_api_surface_audit.md`
  - The current authoring burden: a kernel author writing a tiny-tile op must declare the geometry in three places (Python `ttnn.Tile`, C++ kernel CT arg struct, kernel-body SFPU iteration count) and keep them in sync.
  - Concrete proposals to reduce burden:
    - Auto-derive SFPU iteration count from `face_r_dim` at compile time via a `constexpr` helper.
    - Promote `TileDescriptor` to a first-class field on CBHandle so consumers can read producer geometry without re-declaring it.
    - Lint check: validate that a kernel's declared CT args' `TILE_HEIGHT` matches the bound CB's `tile.tile_shape[0]`.
  - Documentation gaps: the `validate_tensor_shape_tile_dependent_ops_` validator is the authoritative spec but is not surfaced in any developer-facing doc.

- `02_debugging_tooling.md`
  - Existing tools and what they reveal about tiny tiles:
    - `BLAZE_L1_PROFILE` — dumps CB address, type, dtype, and page size; tile geometry is implicit in page size.
    - `BLAZE_DEBUG_KERNELS` — phase markers; useful for diagnosing CB wait deadlocks caused by geometry mismatch.
    - `named_args_generated.h` inspection — shows the resolved CT args including any `TILE_HEIGHT` fields.
    - `viz_export` — graph visualizer; shows CB edges but not tile geometry today.
  - Proposed additions:
    - Visualizer: show TileDescriptor on each CB edge.
    - L1 profiler: print "tile geometry" column explicitly.
    - Lint pass: flag CB-handle handoffs where producer / consumer tile geometries disagree.

- `03_test_infrastructure_gaps_and_recommendations.md`
  - Recap of Ch 3.04 gaps, plus concrete test-source proposals:
    - `tests/sources/eltwise_tiny_tile_test.cpp` — standalone SFPU at 8x32 / 16x32.
    - `tests/sources/reduce_tiny_tile_test.cpp` — standalone reduce at 8x32.
    - `tests/sources/quasar/matmul_quasar_tiny_test.cpp` — Quasar tiny matmul.
  - PCC golden helper that handles tiny tile output layout.

- `04_open_questions_and_future_work.md`
  - Open questions the research surfaces but does not answer:
    - Is there an arch-level reason `face_c_dim` is fixed at 16 (vs. allowing 1, 2, 4, 8 column-faces analogously to row-faces)?
    - Could the LLK API expose a `TinyTileDescriptor` type that statically guarantees the legality of `(num_faces, face_r_dim, face_c_dim)`?
    - What is the actual hardware cost of partial-face mode vs. tiny-tile mode — are they the same code path internally?
    - Quasar: does the new TDMA descriptor architecture make tiny-tile handling more uniform across arches, or does it introduce new constraints?
  - Suggested follow-on research topics: [[tiny_tiles_on_quasar]], [[automatic_tile_geometry_inference_in_blaze_nn]].

---

## Conventions

### Terminology
- **Tile:** A unit of data the Tensix data path moves as a single block. Default is 32x32 unless qualified.
- **Tiny tile:** A tile with `num_faces ∈ {1, 2, 4}` and `face_r_dim ∈ {1, 2, 4, 8, 16}`, with `face_c_dim` fixed at 16. Effective heights 1, 2, 4, 8, or 16 rows; effective widths 16 or 32 columns.
- **Face:** A 16x16 sub-block of a tile (the canonical 32x32 tile has 4 faces F0–F3, in row-major-of-faces order).
- **face_r_dim / face_c_dim:** Hardware-level face row/column dimension. `face_c_dim` is always 16 on tiles the Tensix data path consumes.
- **num_faces:** Total face count in the tile (1, 2, or 4).
- **TILE_HEIGHT:** Logical tile-row count = `face_r_dim * (num_faces_per_column)`. Used as the user-visible parameter.
- **PNHt:** Padded Number of Heads in tiles — the Flash MLA shorthand for Q's row-tile count (=1 when `n_heads_per_core ≤ 16`).
- **DHt:** Head-dimension in tiles (column-tile count).
- **Q_TILE_HEIGHT / K_TILE_HEIGHT:** Per-tensor logical tile height; Flash MLA uses different values per tensor in the same kernel.
- **CBHandle:** TT-Blaze's typed handle for a circular buffer; carries the TileDescriptor.
- **TileDescriptor:** TT-Metal wrapper around `ttnn.Tile` for use in CB descriptors.
- **MOP:** Micro-Operation Program — pre-programmed instruction sequence stored in a replay buffer.
- **DstSync:** Destination register synchronization mode (SyncHalf or SyncFull).
- **SFPU iteration count:** The `N` template argument to `llk_math_eltwise_unary_sfpu_*<approx_mode, false, N>(...)`; equals the number of 16-row faces in the tile.
- **Partial face:** A face with `face_r_dim < 16`, populated only in its top rows.
- **Pad-and-mask:** The alternative to tiny tiles — pad the activation to 32x32, run the standard data path, mask out the inactive rows in the consumer (or rely on zero padding).

### Notation
- LLK function names with leading/trailing underscores: `_llk_unpack_AB_matmul_init_<>()`.
- Template parameters in angle brackets: `<MATH_FIDELITY, THROTTLE_LEVEL>`.
- Tile shapes as RxC: 8x32, 32x32. When a tile geometry is given without a column dimension, 32 is implied.
- File paths relative to the repository root of the project they belong to (tt-llk, tt-metal, tt-blaze) unless qualified.
- Code examples are C++17 for kernel code and Python 3 for op-emit / host-side code.
- Cross-references between guides use `[[guide-slug]]` notation.

### Architecture-Specific Annotations
- `[BH]` for Blackhole-only statements.
- `[WH]` for Wormhole B0-only.
- `[Q]` for Quasar-only.
- Statements without a prefix apply to all three.

### Formatting Rules
- Each chapter file begins with a 2–4 sentence overview and a "Prerequisites" line listing earlier chapters / sections required.
- Tables for CB layouts and L1 budgets use the column set `(CB name, tile geometry, pages, L1 size, role)`.
- Code blocks include the file path and line range as a comment header when quoting from the source tree.

---

## Cross-Chapter Dependencies

| Chapter | Depends On | Reason |
|---------|-----------|--------|
| Ch 2 (LLK Data Path) | Ch 1 | The unpack/math/pack treatment requires the tile-geometry vocabulary from Ch 1. |
| Ch 3 (Test Infrastructure) | Ch 1, Ch 2 | Test sweeps and PCC comparison need the geometry definitions (Ch 1) and the data-path behavior (Ch 2). |
| Ch 4 (TT-Metal Metadata) | Ch 1 | The API surface mirrors the geometry concepts from Ch 1; data-path details from Ch 2 are not required for Ch 4. |
| Ch 5 (DeepSeek V3 B1 Usage) | Ch 1, Ch 2, Ch 4 | Production usage requires the geometry vocabulary, the data-path rules (esp. SFPU iteration count), and the TT-Metal API. |
| Ch 6 (Trade-offs) | Ch 2, Ch 4, Ch 5 | Trade-off analysis needs the cycle-count behavior (Ch 2), the L1 budget plumbing (Ch 4), and the production examples (Ch 5). |
| Ch 7 (Cross-Cutting) | Ch 1, Ch 5 | Per-arch and CCL concerns are layered on the core concepts (Ch 1) and the propagation rules learned from Ch 5. |
| Ch 8 (Ergonomics) | All prior chapters | Recommendations are grounded in the friction points surfaced throughout Ch 1–7. |

---

## Source Roots

- **TT-LLK:** `/localdev/salnahari/testing_dir/tt-llk`
- **TT-Metal:** `/localdev/salnahari/testing_dir/tt-metal`
- **TT-Blaze:** `/localdev/salnahari/testing_dir/tt-blaze`
- **DeepSeek V3 B1 reference implementation:** `/localdev/salnahari/testing_dir/tt-metal/models/demos/deepseek_v3_b1`

---

## Related Research

- [[introduction_to_tt_llk]] — `ch3_data_organization/tiles_and_faces.md` contains the canonical "Tiny Tiles (Non-32x32)" subsection that motivates this guide.
- [[minimum_requirements_to_run_a_matmul_on_p150_using_pure_llk]] — `ch8_test_infrastructure/01_matmul_test_survey.md` enumerates the tiny-tile test drivers leveraged in Ch 3.
- [[comprehensive_understanding_of_deepseek_v3_b1]] — `ch05_multi_head_latent_attention_deep_dive/03_flash_mla_decode.md` is the deep-dive source for Ch 5.
- [[comprehensive_guide_to_creating_new_micro_ops_in_tt_blaze]] — `ch06_inter_core/03_ccl_and_fabric.md` for the CCL alignment material referenced in Ch 7.
- [[comprehensive_guide_to_tenstorrent_sockets_d2d_h2d_and_d2h_communication]] — for the cross-device propagation discussion in Ch 7.
