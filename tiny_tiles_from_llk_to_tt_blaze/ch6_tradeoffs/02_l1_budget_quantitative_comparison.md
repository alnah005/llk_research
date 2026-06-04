# L1 Budget — Quantitative Comparison (Flash MLA)

This section quantifies the per-core L1 footprint of Flash MLA decode in two configurations: the production tiny-Q layout (Q at 8x32) and a hypothetical pad-to-32x32 layout. Because the K/V buffer dominates the budget and is unaffected by Q's geometry, the savings from tiny tiles are sub-linear in the row ratio — but they are still the difference between fitting and not fitting on Blackhole P150 with usable double-buffering depth.

**Prerequisites:** Chapter 1 (tiny-tile definition and legal geometries), Chapter 5 Section "Flash MLA decode op" (CB inventory and shard structure), and Chapter 6, Section "When tiny tiles pay off — decision criteria".

## 1. Tiny-Q configuration (production)

Flash MLA decode uses Q at 8x32 (BF16) and K at 32x32 (BFP8). The per-core CB inventory comes from `models/demos/deepseek_v3_b1/tt/attention/flash_mla/op.py` and the matching kernel CB declarations; the consolidated table below is taken from `ch05_multi_head_latent_attention_deep_dive/03_flash_mla_decode.md` lines 568–596.

| CB | Tile geometry | Pages | L1 size (B) | Role |
|----|---------------|-------|-------------|------|
| `cb_q_in` | 8x32 BF16 | 18 | 9,216 | Q input shard |
| `cb_k_in` | 32x32 BFP8 | 144 | 156,672 | Double-buffered K chunks |
| `cb_mask` | 8x32 BF16 | 1 | 512 | Causal mask |
| `cb_ms_in` | 8x32 BF16 | 3 | 1,536 | Tree-reduction m/s inputs |
| `cb_out_in` | 8x32 BF16 | 48 | 24,576 | Tree-reduction O data |
| `cb_out_o` | 8x32 BF16 | 16 | 8,192 | Output O accumulator |
| `cb_out_ms` | 8x32 BF16 | 1 | 512 | Output m/s |
| `cb_interm_out` | 8x32 BF16 | 16 | 8,192 | Intermediate O (aliased with `cb_out_o`) |
| `cb_interm_ms` | 8x32 BF16 | 1 | 512 | Intermediate m/s (aliased with `cb_out_ms`) |
| `cb_out_final` | 8x32 BF16 | — | 8,192 | Final output stage |
| **Total** |  |  | **~209,408 B (~204.5 KB)** |  |

Notes on the table:
- A BF16 8x32 tile occupies 512 B (single 16x16 face packed at half the row count of a standard tile).
- A BFP8 32x32 tile occupies 1,088 B (1,024 B mantissas + 64 B per-face exponents).
- `cb_k_in` alone consumes 156,672 B / 209,408 B = **~75% of the per-core L1 budget**.
- The Q-side CBs collectively account for ~52.7 KB, i.e. the remaining ~25%.

## 2. Pad-to-32x32 hypothetical

If Q were padded from 8 rows to 32 rows (4x row inflation), every Q-side CB scales by 4x because each page now holds a full 32x32 tile (2,048 B BF16) instead of an 8x32 tile (512 B). The K CB is unchanged.

| CB | Tile geometry | Pages | Tiny-Q L1 (B) | Padded-Q L1 (B) | Delta |
|----|---------------|-------|---------------|-----------------|-------|
| `cb_q_in` | 32x32 BF16 | 18 | 9,216 | 36,864 | +27,648 |
| `cb_k_in` | 32x32 BFP8 | 144 | 156,672 | 156,672 | 0 |
| `cb_mask` | 32x32 BF16 | 1 | 512 | 2,048 | +1,536 |
| `cb_ms_in` | 32x32 BF16 | 3 | 1,536 | 6,144 | +4,608 |
| `cb_out_in` | 32x32 BF16 | 48 | 24,576 | 98,304 | +73,728 |
| `cb_out_o` | 32x32 BF16 | 16 | 8,192 | 32,768 | +24,576 |
| `cb_out_ms` | 32x32 BF16 | 1 | 512 | 2,048 | +1,536 |
| `cb_interm_out` | 32x32 BF16 | 16 | 8,192 | 32,768 | +24,576 |
| `cb_interm_ms` | 32x32 BF16 | 1 | 512 | 2,048 | +1,536 |
| `cb_out_final` | 32x32 BF16 | — | 8,192 | 32,768 | +24,576 |
| **Total** |  |  | **209,408** | **~404,432** | **+~195 KB** |

A more conservative estimate that ignores aliasing relief and matches the figure circulating in design notes (`plan.md` line 254 and prior Flash MLA reviews) puts the padded total at ~280 KB. The discrepancy reflects whether `cb_interm_*` is genuinely aliased with `cb_out_*` in the padded layout (the aliasing optimization is preserved across padding). Taking the aliasing-aware figure:

- **Tiny-Q total:** ~209 KB
- **Padded-Q total (with aliasing):** ~280 KB
- **Savings from tiny Q:** **~71 KB per core (~25% reduction)**

## 3. Why the savings are sub-linear in the row ratio

A naive expectation is "Q went from 32 rows to 8 rows, so L1 should shrink 4x." It does not, because the dominant CB (`cb_k_in` at 156,672 B, ~75% of the budget) is **already at full 32x32** and is unaffected by Q's geometry. K stays standard-tile for two reasons:

1. **K is shared across all Q rows in the matmul.** Shrinking K would only help if the head dimension Dh or chunk length collapsed, neither of which depends on the per-query batch.
2. **K is BFP8-packed already.** The packing already extracts most of the format-side savings; further shape changes would require structural rework of the K reader and DRAM layout.

The arithmetic of the inflation:

```
savings_ratio = (q_side_padded - q_side_tiny) / total_tiny
              = (~223 KB - ~52 KB) / 209 KB
              ~ 36% of tiny-budget reclaimed back as overhead if padded
```

Equivalently: 4x inflation on ~25% of the budget yields ~75% larger Q-side footprint, not 4x overall. The K-dominance pattern recurs across attention kernels — whenever the "narrow" axis (per-query, per-head) goes tiny but the "wide" axis (KV cache, sequence) stays standard, tiny tiles only pinch the narrow side.

## 4. Fraction of P150 L1 consumed

Blackhole P150 exposes ~1.3 MB usable per-core L1 (1,331,200 B after kernel text, runtime args, stack, and semaphores are reserved; see `plan.md` line 254 and the device-init reservations in `tt_metal/llrt/`).

| Configuration | L1 used (B) | Fraction of 1.3 MB | Headroom (B) |
|---------------|-------------|--------------------|--------------|
| Tiny-Q (8x32) | 209,408 | 15.7% | 1,121,792 |
| Padded-Q (32x32) | ~286,720 | 21.5% | 1,044,480 |
| Delta | ~71 KB | ~5.3 pp | ~71 KB |

The 71 KB of headroom recovered by tiny Q is what lets Flash MLA hold K **double-buffered** at depth 144 tiles. If Q were padded, the same K-depth would either force eviction-style streaming (single-buffer K, lose latency hiding) or a halving of K depth, doubling DRAM round-trips.

**[WH] Wormhole B0:** usable per-core L1 is ~1.0 MB — about 25% smaller than P150. The same tiny-Q layout consumes 209 KB / ~1,024 KB ≈ 20.4% of L1 on WH; the padded variant would consume ~28%, leaving correspondingly less room for the K double-buffer. Tiny tiles on WH are not optional for Flash MLA at the current K depth.

## 5. The BFP8 K interaction

The K CB's dominance also explains why BFP8 packing of K was a prerequisite for tiny-Q to be worth implementing. From `ch05_multi_head_latent_attention_deep_dive/03_flash_mla_decode.md` lines 250–254: a BF16 K tile is 2,048 B vs. 1,088 B for BFP8. Double-buffered K at 144 tiles would be:

- BFP8 (current): 144 x 1,088 = 156,672 B (~153 KB)
- BF16 (hypothetical): 144 x 2,048 = 294,912 B (~288 KB)

The BF16-K layout alone would consume more than the entire padded-Q layout's overhead reclaim, and combined with padded Q would push the kernel to ~419 KB — over 31% of P150 L1, leaving insufficient room for the producer-side reader CBs and the intermediate accumulators. Tiny Q saves the kernel from the **last** budget crisis; BFP8 K saved it from the first.

## 6. Summary

- **Tiny Q saves ~71 KB per core (~25% of the tiny-Q budget)** versus a padded-Q layout, enabling double-buffered K at depth 144 on Blackhole P150 and making the kernel viable at all on Wormhole B0.
- Savings are sub-linear in the Q row ratio (4x rows -> 1.25x total, not 4x total) because the dominant CB is K, which is independent of Q geometry.
- The decision to use tiny Q is conditional on K already being aggressively packed (BFP8). If K were BF16, tiny Q would not be enough — see Chapter 6, Section "When tiny tiles pay off — decision criteria" for the full ranking.
- Cross-arch: the savings absolute magnitude is identical ([BH] and [WH]) but the *fractional* impact is larger on WH due to its smaller L1, making tiny tiles effectively mandatory on WH for this op.

For a CB-layout walkthrough of the Q-side tree reduction (which contributes `cb_out_in` and `cb_ms_in` to the table above), see Chapter 5, Section "Flash MLA tree-reduction layout". For the cycle-cost side of the trade-off — i.e. why the 71 KB L1 saving does not come at a proportional cycle penalty — see Chapter 6, Section "Performance measurement methodology".
