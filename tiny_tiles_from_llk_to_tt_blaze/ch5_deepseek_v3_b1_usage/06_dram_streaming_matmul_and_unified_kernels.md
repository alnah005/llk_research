# DRAM-Streaming Matmul and the Unified-Kernel Activation Path

The DRAM-streaming matmul is where tiny tiles meet bandwidth-bound projections: weights live in DRAM, the activation is a single decode token replicated per core, and the output is a narrow strip of one to a handful of rows wide. This file walks the `dram_streaming_matmul.hpp` micro-op and the closely related `unified_kernels/matmul.hpp` to show how a single compile-time `tile_r_dim` (or, in the unified-kernels path, a hardcoded SFPU iteration count) lets one matmul kernel template cover output heights from `1x32` all the way to `32x32`.

**Prerequisites:** Chapter 2 (LLK tile/face geometry, SFPU face iteration), Chapter 3 (tt-metal `Tile` / `CB` descriptor plumbing for tiny tiles), Chapter 5 sections on Flash MLA decode and `create_q_heads`, which establish the tiny-tile producers whose outputs feed this matmul.

## 1. Why the LM-head / projection matmul wants tiny outputs

In DeepSeek V3 B1 decode, the activation feeding a per-core matmul is one decode token (M=1) replicated on every compute core, with weights `[K, N]` width-sharded across DRAM banks. The output is a width-sharded strip `[1, per_core_N]` in L1. If we pad M up to 32 the entire `[32, K]` activation slab inflates by 32x, and the matmul kernel produces 31 padding rows that get pack-discarded — a pure waste of unpack bandwidth, dest registers, and pack cycles.

The test `tests/unit_tests/test_dram_streaming_matmul.py` parametrizes `m ∈ {1, 4, 8}` (line 122) and binds `tile_h = m` straight into the `ttnn.Tile` constructor:

```python
# tests/unit_tests/test_dram_streaming_matmul.py:139-144
num_loop_iters = 100 if m == 1 else 1
tile_h = m  # Tile height matches m (1 for tiny tiles, 32 for standard)
tile_w = 32

# Create tile object for tiny tiles when m=1
in0_tile = ttnn.Tile([tile_h, tile_w])
out_tile = ttnn.Tile([tile_h, tile_w])
```

Both `in0` and the output use the same `[tile_h, 32]` tiny tile (`tile=in0_tile` on line 188, `tile=out_tile` on line 221), while the DRAM-resident `in1` weight tensor always uses `[32, 32]` tiles — see line 233:

```python
# tests/unit_tests/test_dram_streaming_matmul.py:233
in1_tile = ttnn.Tile([tile_w, tile_w])  # in1 uses 32x32 tiles
```

This is the mixed-geometry contract the kernel sees: `in0` is `[m, K]` with `[m, 32]` tiles, `in1` is `[K, N]` with `[32, 32]` tiles, output is `[m, N]` with `[m, 32]` tiles. The matmul itself is shape-correct (`[m, K] @ [K, N] = [m, N]`) but the unpacker, math, and packer must agree on the per-tile row dimension.

## 2. The `tile_r_dim` compile-time switch

The whole point of `DRAMStreamingMatmul::ComputeCTArgs` is to surface this row dimension as a template parameter so the kernel can route it into every face-aware LLK call. From `models/demos/deepseek_v3_b1/unified_kernels/dram_streaming_matmul.hpp:91-113`:

```cpp
// models/demos/deepseek_v3_b1/unified_kernels/dram_streaming_matmul.hpp:91-113
// Compute CTArgs (TRISC)
template <
    uint32_t cb_in0_,
    uint32_t cb_in1_,
    uint32_t cb_out_,
    uint32_t subblock_k_,
    uint32_t per_core_n_,
    uint32_t subblock_w_,
    uint32_t num_subblocks_k_,
    uint32_t tile_r_dim_,
    uint32_t fuse_silu_,
    uint32_t fp32_dest_acc_en_ = 0>
struct ComputeCTArgs {
    static constexpr uint32_t cb_in0 = cb_in0_;
    static constexpr uint32_t cb_in1 = cb_in1_;
    static constexpr uint32_t cb_out = cb_out_;
    static constexpr uint32_t subblock_k = subblock_k_;
    static constexpr uint32_t per_core_n = per_core_n_;
    static constexpr uint32_t subblock_w = subblock_w_;
    static constexpr uint32_t num_subblocks_k = num_subblocks_k_;
    static constexpr uint32_t tile_r_dim = tile_r_dim_;
    static constexpr bool fuse_silu = fuse_silu_ == 1;
    static constexpr bool fp32_dest_acc_en = fp32_dest_acc_en_ == 1;
};
```

`tile_r_dim` is the row dimension of the `in0` / output tile in *rows*, not in tiles. When the host op emits this kernel it sets `tile_r_dim = m` (so `1`, `4`, `8`, ... depending on the test parametrization or production call site). Two other knobs work in concert:

- `per_core_n`: number of output *tiles* per core in the N direction. For an LM head producing `[1, 2048]` width-sharded across 8 DRAM banks (line 158 of the test computes `per_core_N = n_padded // num_banks`), this is `2048 / 8 / 32 = 8` tiles per core.
- `subblock_w`: how many of those `per_core_n` columns get computed per inner subblock — this drives the per-tile loop in section 3.
- `fuse_silu`: when set, the kernel runs SiLU through the SFPU on the PACK thread between math commit and pack — see section 4.

Critically, the kernel never assumes 32 rows. Every face-iteration count is keyed off `tile_r_dim`.

## 3. Wait/reserve/compute structure on TRISC

Stripped of the indexing and pipelining detail, the TRISC body of `DRAMStreamingMatmul::Op::impl` is a two-level loop: an outer loop over `num_subblocks_n = per_core_n / subblock_w` and an inner loop over the `subblock_w` output columns per subblock. From `models/demos/deepseek_v3_b1/unified_kernels/dram_streaming_matmul.hpp:266-291`:

```cpp
// models/demos/deepseek_v3_b1/unified_kernels/dram_streaming_matmul.hpp:266-291
constexpr uint32_t num_subblocks_n = CTArgs::per_core_n / CTArgs::subblock_w;
constexpr uint32_t num_tiles_k = CTArgs::subblock_k * CTArgs::num_subblocks_k;
constexpr bool transpose = false;
constexpr bool split_acc = true;
constexpr bool dense_packing = false;

if constexpr (CTArgs::fp32_dest_acc_en != DST_ACCUM_MODE) {
    custom_mm_block_init<transpose, split_acc, dense_packing, CTArgs::fp32_dest_acc_en>(
        CTArgs::cb_in0, CTArgs::cb_in1, CTArgs::cb_out);
} else {
    reconfig_data_format<false, true>(CTArgs::cb_in1, CTArgs::cb_in0);
    pack_reconfig_data_format<true>(CTArgs::cb_out);
    custom_mm_block_init_short<transpose, split_acc, dense_packing>(
        CTArgs::cb_in0, CTArgs::cb_in1, CTArgs::cb_out);
}

if constexpr (CTArgs::fuse_silu) {
    PACK((llk_math_eltwise_unary_sfpu_silu_init<true>()));
} else {
    pack_block_contiguous_init(CTArgs::cb_out);
}

cb_wait_front(CTArgs::cb_in0, num_tiles_k);

for (uint32_t sb_n = 0; sb_n < num_subblocks_n; sb_n++) {
    cb_reserve_back(CTArgs::cb_out, CTArgs::subblock_w);
    ...
```

A few load-bearing observations:

1. **`cb_wait_front(cb_in0, num_tiles_k)` (line 288)** is done once, outside the N loop. `in0` is replicated per-core (the test height-shards `[m, K]` per core, lines 178–189), so it stays resident across every output column. With `m = 1` the entire `in0` shard is `1 * K = 7168` elements — a single `[1, 32]` tile lane has `K/32 = 224` tiles, each 32 bytes in bfp4 (the test in1 dtype on line 203). With `m = 32` the same K but row-padded tiles, you'd hold 32x as many bytes; this is the L1 savings argument expressed at the host-CB level.

2. **`cb_reserve_back(cb_out, subblock_w)` (line 291)** reserves at the tile granularity of the `cb_out` descriptor. If the op binds an `[m, 32]` tile to `cb_out`, that reserve allocates `subblock_w` tiles of `m * 32` bfloat16 elements each — exactly the narrow output the consumer expects.

3. **`num_tiles_k = subblock_k * num_subblocks_k` (line 267)** is the K-tile count for one column. The inner loop in section 4 walks this in chunks of `subblock_k` so the DRAM stream and the math thread can be double/triple-buffered (the NCRISC half of the kernel, lines 145–250, manages exactly that).

## 4. Fused SiLU and the `tile_r_dim`-aware SFPU iteration count

When `fuse_silu` is set, the inner loop processes one output tile at a time. The matmul block computes the full `[m, 32]` output into dest reg 0, then SiLU runs on the PACK thread before the pack. From `models/demos/deepseek_v3_b1/unified_kernels/dram_streaming_matmul.hpp:293-337`:

```cpp
// models/demos/deepseek_v3_b1/unified_kernels/dram_streaming_matmul.hpp:293-337
if constexpr (CTArgs::fuse_silu) {
    // Per-tile pipelining with SFPU overlap
    for (uint32_t w = 0; w < CTArgs::subblock_w; w++) {
        tile_regs_acquire();

        // Intermediate subblocks: finalize=false (partial accumulation)
        for (uint32_t sb_k = 0; sb_k < CTArgs::num_subblocks_k - 1; sb_k++) {
            cb_wait_front(CTArgs::cb_in1, CTArgs::subblock_k);
            custom_mm_block<false>(
                CTArgs::cb_in0, CTArgs::cb_in1, sb_k * CTArgs::subblock_k, 0, 0, CTArgs::subblock_k);
            cb_pop_front(CTArgs::cb_in1, CTArgs::subblock_k);
        }
        // Final subblock: finalize=true
        cb_wait_front(CTArgs::cb_in1, CTArgs::subblock_k);
        custom_mm_block<true>(
            CTArgs::cb_in0,
            CTArgs::cb_in1,
            (CTArgs::num_subblocks_k - 1) * CTArgs::subblock_k,
            0,
            0,
            CTArgs::subblock_k);
        cb_pop_front(CTArgs::cb_in1, CTArgs::subblock_k);

        tile_regs_commit();

        // Run SiLU on PACK thread
        PACK(TTI_SEMWAIT(
            p_stall::STALL_TDMA | p_stall::STALL_CFG,
            semaphore::t6_sem(semaphore::MATH_PACK),
            p_stall::STALL_ON_ZERO));
        PACK(TT_SETC16(
            DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));

        if constexpr (CTArgs::tile_r_dim <= 4) {
            PACK((llk_math_eltwise_unary_sfpu_silu<true, false, 2>(0, (int)VectorMode::R)));
        } else if constexpr (CTArgs::tile_r_dim == 8) {
            PACK((llk_math_eltwise_unary_sfpu_silu<true, false, 4>(0, (int)VectorMode::R)));
        } else {
            PACK((llk_math_eltwise_unary_sfpu_silu<true, false, 8>(0, (int)VectorMode::R)));
        }
```

The three branches selecting between SFPU iteration counts of `2`, `4`, and `8` are the heart of this section. The third template parameter to `_llk_math_eltwise_unary_sfpu_silu_<>` is the number of inner SFPU iterations the LLK runs over the tile's faces. The SFPU processes a tile face-by-face, and each iteration steps the SFPU across one full SIMD-pass worth of rows. The selection rule encoded here is:

- `tile_r_dim ∈ {1, 2, 4}` → 2 iterations
- `tile_r_dim == 8` → 4 iterations
- `tile_r_dim ∈ {16, 32}` → 8 iterations

Reading this against the LLK SFPU iteration model: each iteration walks the SFPU through 4 rows of one face. A standard `32x32` tile is `4 faces * 16 rows = 8 * 4`-row passes — hence `8` iterations cover it. An `8x32` tiny tile is `2 faces * 8 rows = 4 * 4`-row passes — `4` iterations. A `1x32`, `2x32`, or `4x32` tile still has the SFPU stepping at the face level but covers at most one effective face of dest data; `2` iterations is the safe minimum that finalizes correctly.

This is the cleanest closed-form mapping in DeepSeek V3 B1 between a Python-level `Tile([m, 32])` choice and the LLK template parameter on the math thread. Every other knob in the kernel (CB sizes, dest-reg layout, pack stride) flows from `tile_r_dim` via the data-format reconfig calls; only the SFPU loop needs an explicit branch because the LLK SFPU template count cannot itself be derived from the data-format alone.

### Table: SFPU iteration count vs. tile height

| tile rows | faces (16-row) | faces (8-row) | LLK iterations (DRAM-streaming) | notes |
|---|---|---|---|---|
| 1  | partial | partial | 2  | M=1 LM-head / decode projections |
| 2  | partial | partial | 2  | uncommon, still safe at 2 |
| 4  | partial | partial | 2  | RoPE-style 4-head shards |
| 8  | 1 (partial face row) | 1 full face | 4  | Flash MLA Q-tile geometry |
| 16 | 1 full face | 2 full faces | 8  | 16-head shards |
| 32 | 2 full faces | 4 full faces | 8  | standard tile |

The `dram_streaming_matmul.hpp` branch list (`<= 4`, `== 8`, else) is the production rule; the table generalizes it to the other tile heights that show up elsewhere in DeepSeek V3 B1 (Flash MLA's 8x32 in Section 03; RoPE's `[num_heads, 32]` in Section 02).

## 5. `unified_kernels/matmul.hpp`: the L1-resident sibling

`dram_streaming_matmul.hpp` is the variant where `in1` is streamed from DRAM. The sibling kernel `unified_kernels/matmul.hpp` is the simpler L1-resident matmul used when both `in0` and `in1` already live in L1 (e.g., the per-head Q/K/V projections after Flash MLA's KV cache branch). It also supports fused activation, and its iteration-count selection rule is even simpler. From `models/demos/deepseek_v3_b1/unified_kernels/matmul.hpp:61-73`:

```cpp
// models/demos/deepseek_v3_b1/unified_kernels/matmul.hpp:61-73
// Compute CTArgs (TRISC): out_w (output width in tiles), transpose, fused_activation
template <
    uint32_t out_w_,
    bool transpose_ = false,
    uint32_t fused_activation_ = 0,
    bool fused_activation_approx_mode_ = false>
struct ComputeCTArgs {
    static constexpr uint32_t out_w = out_w_;
    static constexpr bool transpose = transpose_;
    static constexpr FusedActivation fused_activation = static_cast<FusedActivation>(fused_activation_);
    static constexpr bool fuse_sigmoid = fused_activation == FusedActivation::SIGMOID;
    static constexpr bool fuse_silu = fused_activation == FusedActivation::SILU;
    static constexpr bool fused_activation_approx_mode = fused_activation_approx_mode_;
};
```

Note there is no `tile_r_dim` here. This is a deliberate scope narrowing: this kernel is only used for tiny-output paths where the tile height comes from the CB descriptor and the SFPU iteration count is *hardcoded to 2*. From `matmul.hpp:144-180`:

```cpp
// models/demos/deepseek_v3_b1/unified_kernels/matmul.hpp:144-180
// Reserve output tiles
cb_reserve_back(args.out, out_w);

if constexpr (fuse_activation) {
    // Initialize activation on PACK thread
    if constexpr (CTArgs::fuse_sigmoid) {
        PACK((ckernel::llk_math_eltwise_unary_sfpu_sigmoid_init<CTArgs::fused_activation_approx_mode>()));
    } else {
        PACK((ckernel::llk_math_eltwise_unary_sfpu_silu_init<CTArgs::fused_activation_approx_mode>()));
    }

    // Per-tile: matmul -> activation on PACK -> pack
    for (uint32_t w = 0; w < out_w; w++) {
        tile_regs_acquire();

        custom_mm_block<finalize, read_transposed>(args.in0, args.in1, 0, w * args.k_num_tiles, 0, args.k_num_tiles);

        tile_regs_commit();

        // Run activation on PACK thread
        PACK(TTI_SEMWAIT(
            p_stall::STALL_TDMA | p_stall::STALL_CFG,
            semaphore::t6_sem(semaphore::MATH_PACK),
            p_stall::STALL_ON_ZERO));
        PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));

        // Use 2 iterations for 1x32 tiny tiles
        if constexpr (CTArgs::fuse_sigmoid) {
            PACK((ckernel::
                      llk_math_eltwise_unary_sfpu_sigmoid<CTArgs::fused_activation_approx_mode, false, 2>(
                          0, (int)VectorMode::R)));
        } else {
            PACK((ckernel::llk_math_eltwise_unary_sfpu_silu<CTArgs::fused_activation_approx_mode, false, 2>(
                0, (int)VectorMode::R)));
        }

        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));

        pack_tile(0, args.out, w);
        tile_regs_release();
    }
}
```

The comment on line 170 is precise: **"Use 2 iterations for 1x32 tiny tiles"**. This kernel template is specialized for the narrow-output regime — `out_w` tiles where each tile is `1x32` (or any height that fits within one face's worth of dest rows). The `2` is the minimum count that finalizes the SFPU pipeline for a tile that occupies at most one face; any height up to and including `8x32` is still safe because the SFPU faces beyond the live rows process zero-padded dest registers, which are then discarded by the pack mask the `[tile_r_dim, 32]` CB descriptor installs.

This is *not* equivalent to the `<= 4 → 2` branch in `dram_streaming_matmul.hpp` — `matmul.hpp`'s `2` is a hard contract that says "if you bind a tile bigger than ~8 rows to this kernel, the upper rows won't be correctly SFPU-processed." The DRAM-streaming kernel needed the explicit branching because it gets reused for `m ∈ {1, 4, 8}` in tests and for larger `m` in production (e.g., prefill phases of the same projection). The unified-kernels matmul stays narrow on purpose: its consumers in the Flash MLA chain (Q/K-out projections after `create_q_heads`, gate-projection muls in MoE) all bind tiles of height ≤ 8.

### What `out_w` actually means here

In `matmul.hpp` `out_w` is the number of output tiles per call (line 67). With the test's `m=1`, `n=2048`, `num_cores=8`, each core packs `2048 / 8 / 32 = 8` tiles of width — so `out_w = 8`. Each loop iteration `w` runs one `custom_mm_block` that consumes the full `k_num_tiles` of K and produces one tile of output; the SFPU then processes that single tile in 2 iterations and the packer writes it to `cb_out` at offset `w`. The loop structure is:

```
for w in [0, out_w):
    acquire dest
    custom_mm_block(in0[K], in1[w*K : (w+1)*K]) -> dest[0]
    commit
    SFPU silu/sigmoid<approx, false, 2>(dest[0])
    pack_tile(dest[0], cb_out, w)
    release
```

This is the model: per-tile pipelining where MATH writes a fresh `[m, 32]` block into dest reg 0, PACK runs SFPU over it with 2 iterations, then PACK writes it to its slot in `cb_out`. Dest reuse keeps the working set at exactly one tile of dest, which is what lets the `m=1` case pack 8 tiles back-to-back with no register pressure.

## 6. Host-side flow: how `tile_r_dim` and the CB descriptors stay in sync

The host op (`micro_ops/dram_streaming_matmul/op.py`, called from `DRAMStreamingMatmul.op(...)` on line 261 of the test) is responsible for two contracts:

1. The `in0` / `out` CB descriptors are built from sharded tensors that carry `ttnn.Tile([m, 32])`. Tile-size in bytes flows automatically from `tile.get_tile_size(df)`.
2. The `tile_r_dim` template parameter the kernel sees is set to `m` so the SFPU branch picks the right iteration count.

These two paths are independent but must agree. The convention across DeepSeek V3 B1 is that the op extracts the row count from the input tensor's shard spec or tile object — there is no `tile_r_dim` argument the user passes; it is derived. See `flash_mla/op.py:411-462` (referenced in Chapter 5 Section 3) for the analogous pattern with `Q_TILE_HEIGHT`.

The DRAM-streaming weights tensor (`in1`) sidesteps the question entirely by always using `[32, 32]` tiles (test line 233). This is important: weights are quantized to bfp4, shuffled column-major per shard for streaming, and read 32 rows at a time by the NCRISC reader (lines 222–243 of the kernel). Making them `[m, 32]` would not give any L1 saving — they live in DRAM — and would complicate the page-size math the reader uses to pipeline `in1` across `num_buffers` slots (line 165).

## 7. Generalization to other tiny output shapes

The plan in this section claimed `<approx_mode, false, 2>` "generalizes to other tiny output shapes." The precise meaning, in light of the source: it generalizes to other shapes *whose effective face footprint fits in 2 SFPU iterations*. From the LLK iteration model:

- `1x32`, `2x32`, `4x32`: one partial face. 2 iterations process the live rows and discard the unused face columns at pack time. The kernel comment on `matmul.hpp:170` documents this exactly.
- `8x32`: one full 8-row face on Blackhole or two 8-row half-faces on Wormhole. `dram_streaming_matmul.hpp` upgrades to 4 iterations here for safety; `matmul.hpp` does not support this height with fused activation correctly and is not used for it in practice.
- `16x32`, `32x32`: standard cases; `dram_streaming_matmul.hpp` uses 8 iterations; `matmul.hpp` would be wrong here.

[BH] Note: on Blackhole the face row counts and SFPU iteration semantics align cleanly with the 4-iteration choice for `tile_r_dim == 8`. [WH] On Wormhole the same template count works because the LLK headers normalize iteration semantics across architectures (see Chapter 2, Section on SFPU iteration mapping). [Q] Quasar follows the same LLK contract; no separate branch is required.

## 8. Why this matters for the LM-head and projections

The LM-head matmul in DeepSeek V3 B1 decode produces a logits row of shape `[1, vocab]`. With vocab=128k width-sharded across the available DRAM banks, each core produces a width strip of one tile-row tall. Padding to `32x32` would 32x the dest-reg pressure, the pack volume, and the L1 footprint of the output CB — for outputs that get reduced/argmax'd downstream and the 31 padding rows are immediately discarded.

The `[m, 32]` tile + `tile_r_dim`-keyed SFPU branch is what makes this efficient: one matmul kernel template, one `Tile([m, 32])` host-side declaration, and the LLK math/pack threads see the exact face count the data needs. The CB descriptor carries the right page size, the SFPU iterates exactly enough times to cover the live rows, and the pack mask trims any face-aligned padding on the way out to L1.

References:
- Op header: `models/demos/deepseek_v3_b1/unified_kernels/dram_streaming_matmul.hpp:43-384`
- L1-resident sibling: `models/demos/deepseek_v3_b1/unified_kernels/matmul.hpp:49-212`
- Tests and parametrization: `models/demos/deepseek_v3_b1/tests/unit_tests/test_dram_streaming_matmul.py:117-291`
- Cross-reference: see Chapter 5 Section 3 (Flash MLA Q-tile geometry, 8x32 case) and Section 2 (RoPE `[num_heads, 32]` tile derivation); see Chapter 2 for the LLK SFPU iteration / face mapping the iteration counts here build on.
