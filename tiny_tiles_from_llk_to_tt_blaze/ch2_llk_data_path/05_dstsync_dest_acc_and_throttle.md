# DstSync, Dest Accumulation, and Throttle With Tiny Tiles

Dest register slots are allocated at *hardware tile* granularity (32x32), not at the
logical tile-row granularity that L1 buffers use. This file pins down the
arithmetic that follows from that fact: how `DstSync::SyncHalf` vs.
`DstSync::SyncFull` and `is_fp32_dest_acc_en` interact with tiny tiles, how to
size L1 versus dest in the same kernel, and why matmul throttling is disabled on
tiny tiles today.

**Prerequisites:** Chapter 1 (tile-shape definitions, in particular the
distinction between `face_r_dim` and `tile_r_dim`); Chapter 2 sections 01–04
(unpack/math/pack signatures, SFPU face iteration, and dest occupancy).

## 1. The dest slot is sized for the *hardware* tile, not the logical tile

The total number of dest slots available to a kernel is computed by
`get_dest_max_tiles<>()`:

```cpp
// tt_llk_blackhole/common/inc/ckernel.h:844-851
template <DstSync SYNC_MODE, bool ACCUM_MODE>
constexpr std::uint32_t get_dest_max_tiles() {
    constexpr std::uint32_t DEST_REGISTER_SIZE = SYNC_MODE == DstSync::SyncHalf
        ? (ACCUM_MODE ? DEST_REGISTER_HALF_SIZE >> 1 : DEST_REGISTER_HALF_SIZE)
        : (ACCUM_MODE ? DEST_REGISTER_FULL_SIZE >> 1 : DEST_REGISTER_FULL_SIZE);
    return DEST_REGISTER_SIZE >> DstTileSizeLog2[static_cast<int>(TILE_SHAPE)];
}
```

Three pieces of this formula interact with tiny tiles, and only one of them
changes anything:

1. **`SYNC_MODE`** selects between `DEST_REGISTER_HALF_SIZE` and
   `DEST_REGISTER_FULL_SIZE`. Half-sync exposes only one half of dest at a time
   to math while the other is being packed, doubling pipeline overlap at the
   cost of halving the visible budget. This is orthogonal to tile shape.
2. **`ACCUM_MODE`** (the C++ counterpart of `is_fp32_dest_acc_en`) shifts the
   visible budget right by one when fp32 accumulation is enabled. Each dest
   datum is twice as wide, so half as many tiles fit. Also orthogonal to tile
   shape.
3. **`DstTileSizeLog2[TILE_SHAPE]`** indexes into a fixed table keyed by the
   `DstTileShape` enum. The enum has exactly three entries — `Tile32x32`,
   `Tile32x16`, `Tile16x16` (`ckernel_defs.h:60-65`) — with shift values
   roughly 10, 9, 8 respectively. There is **no `Tile8x32`, no `Tile4x32`,
   no `Tile1x32`**.

The third point is the load-bearing one. An 8x32 tiny tile lives in a
`Tile32x32` dest slot. So does a 1x32 or 4x32 tile. The hardware does not have
"smaller dest slots" to give. It allocates a full 32x32-shaped slot and the
tiny tile partially populates it.

Consequence: `get_dest_max_tiles<SyncHalf, false>()` returns the same value
(4 for `Tile32x32` on Blackhole today) whether the kernel is doing
32x32 × 32x32 matmul or 8x32 × 32x32 matmul. The total dest-slot count is
**unchanged** by tiny tiles — still 8 (SyncHalf) or 16 (SyncFull) in 16-bit
mode and 4 (SyncHalf) or 8 (SyncFull) in fp32-dest-acc mode.

## 2. Flash MLA: dest budgeting in practice

Flash MLA uses 8x32 Q tiles against 32x32 K/V tiles. Its dest-size selection
makes the rule above concrete:

```python
# tt-blaze/.../micro_ops/flash_mla/op.py:513-518
if dst_full_sync_en:
    dst_size = 8 if fp32_dest_acc_en else 16  # SyncFull
else:
    dst_size = 4 if fp32_dest_acc_en else 8   # SyncHalf
```

Note what is *not* in this expression: the Q tile height. `Q_TILE_HEIGHT=8`
does not appear, nor does any geometry parameter from the tiny-tile descriptor
(`op.py:538`). The dest budget is a function of `dst_full_sync_en` and
`fp32_dest_acc_en` only. A Flash MLA kernel that emits a 32x32 output tile and
one that emits an 8x32 output tile both claim one dest slot per output.

This also means kernel authors should not "optimize" by assuming a 1x32 output
consumes 1/32 of a dest slot — it consumes a whole slot. If a kernel needs to
hold 16 output tiles concurrently in dest, that requires `SyncFull` plus
16-bit dest, regardless of whether the outputs are full or tiny.

## 3. L1 versus dest: two different units

The L1 circular-buffer math and the dest-slot math use different units, and
mixing them is a frequent source of bugs in tiny-tile kernels.

| Quantity | Sized by | Tiny-tile aware? |
| --- | --- | --- |
| CB page size in L1 | `tile.get_tile_size(dtype)` | **Yes** — uses logical rows |
| Dest slot size in regs | `DstTileSizeLog2[TILE_SHAPE]` | **No** — uses hardware shape |

A concrete comparison for the Flash MLA Q input (bfp8, 8x32):

```python
# tt-blaze/.../micro_ops/flash_mla/op.py:552-555
q_tile_size = q_tiny_tile.get_tile_size(q_df)   # ~280 B for 8x32 bfp8
# CB page for Q in L1 is sized by q_tile_size
# Dest slot consumed during compute is still 1024 B (Tile32x32 hardware slot)
```

The CB page for Q is roughly 280 bytes per tile (8 rows × 32 cols × 1 byte +
shared exponent for the bfp8 face). The dest slot, by contrast, is the full
1024-byte hardware Tile32x32 slot. When sizing the kernel:

- **L1 budget:** sum CB pages using the *logical* tile size returned by
  `get_tile_size`. Tiny tiles save L1.
- **Dest budget:** count slots using `get_dest_max_tiles<>()`. Tiny tiles do
  not save dest.

Forgetting the second rule is the classic failure mode: a kernel that fits in
L1 because of tiny inputs but cannot allocate enough dest slots because the
author assumed dest scales with row count.

## 4. Throttle levels and tiny tiles

Matmul throttle (`THROTTLE_LEVEL` 0–5) inserts NOPs between MVMUL instructions
to cap matmul throughput, trading performance for thermal/power headroom. The
target throughput percentages are documented at the top of the throttled
sequences:

```cpp
// tt_llk_blackhole/llk_lib/llk_math_matmul.h:507-512
// Level 1: ~73% of max throughput
// Level 2: ~67%
// Level 3: ~50%
// Level 4: ~40%
// Level 5: ~33%
```

The throttle paths (`run_throttled_sequence<N>()` at
`llk_math_matmul.h:422-500`) are dispatched by the math configurator at
`llk_math_matmul.h:506-535`, which carries an explicit assertion:

> "MM throttling only enabled for full 32x32 tile size"
> (`llk_math_matmul.h:534-535`)

For tiny tiles, only `throttle=0` is exercised. The Python test sweep mirrors
this:

```python
# tt_llk/.../test_math_matmul.py:67-93
# TINY_TILES_MATMUL_COMBINATIONS is enumerated with throttle=0 only
```

The reason the assert exists is not a hard hardware constraint — the throttle
sequences would still execute — but a fidelity/validation one. The MVMUL
inner-loop length varies with tile geometry (`replay_buf_len` ranges from 4 to
16 MVMULs depending on `face_r_dim`/`num_faces` combinations,
`llk_math_matmul.h:301-307`). A throttle sequence calibrated to dilute a 16-MVMUL
inner loop with N NOPs produces a *different* effective throughput when the
inner loop is only 4 MVMULs long: the same N NOPs now represent a larger
fraction of the cycle budget, pushing the effective rate further below the
documented target.

Without per-geometry calibration of the throttle tables, mixed-geometry
matmuls cannot make the published throughput guarantees, so the LLK simply
refuses to combine them. If a future workload needs throttled tiny-tile
matmul, the work item is to extend `run_throttled_sequence<>` with
inner-loop-aware NOP counts, then relax the assert at `llk_math_matmul.h:534`.

## 5. `is_fp32_dest_acc_en` is orthogonal to tile shape

The fp32-dest-accumulation flag changes the per-datum width in dest from 16 to
32 bits. Mechanically this is `ACCUM_MODE` in `get_dest_max_tiles<>()`: it
shifts the dest budget down by one (e.g. 8 tiles → 4 tiles in SyncHalf,
`ckernel.h:846-847`). There is no special tiny-tile case in this path. The
same flag flows through Flash MLA via `dst_full_sync_en` and
`fp32_dest_acc_en` (`op.py:513-518`) and through every other tt-blaze op that
chooses a dest budget.

The only thing the kernel author has to remember is the unit split from
Section 3: fp32 dest accumulation roughly doubles the *L1* size of any
intermediate that flows out of dest (because the packer writes wider data
back), while halving the *dest slot* count. Tiny tiles affect the L1 leg only.

## 6. Stochastic rounding

Stochastic rounding is a packer/unpacker feature, configured at unpack init:

```cpp
// tt_llk/.../tests/sources/unpack_matmul_test.cpp:39
_llk_unpack_configure_stoch_rnd_<STOCHASTIC_RND>();
```

The three modes (`Fpu`, `Pack`, `All`) apply rounding per-datum at fixed
pipeline stages. None of them inspect tile geometry — rounding decisions are
made on individual values, not on tiles, faces, or rows. There is therefore no
architectural interaction between stochastic rounding and tiny tiles.

Test coverage, however, is currently thin. The matmul sweep parameterizes
only `StochasticRounding.No`:

```python
# tt_llk/.../test_math_matmul.py:53
# parametrize uses StochasticRounding.No exclusively for the matmul tests
```

Tiny-tile stochastic-rounding coverage exists today only via matmul-fused
SFPU paths (see Chapter 2 Section 02 on the SFPU N parameter, and Chapter 3 on
test infrastructure), not as a standalone parameter axis. Anyone enabling
stochastic rounding with tiny-tile matmul should add a test variant — the
hardware should work, but the LLK-level evidence is currently a derivation,
not a measurement.

## Quick reference

- Dest slot count is a function of `DstSync` and `is_fp32_dest_acc_en`, not
  tile shape. Tiny tiles partially populate full Tile32x32 slots.
- Size L1 buffers from `tile.get_tile_size(dtype)`. Size dest budgets from
  `get_dest_max_tiles<>()`. Do not mix the two.
- Throttle (`THROTTLE_LEVEL` 1–5) is asserted off for tiny-tile matmul
  (`llk_math_matmul.h:534-535`); use `throttle=0`.
- Stochastic rounding is per-datum and has no tile-shape interaction, but the
  current test matrix does not exercise it with tiny-tile matmul.
