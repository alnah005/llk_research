# 6.3 — Perf-Delta Measurements: Tiny Tiles vs. Pad-and-Mask

This section walks through the experimental methodology for putting numbers on the tiny-tile vs. pad-and-mask choice. We use the existing `math_matmul_perf.cpp` harness in tt-llk to sweep `in0_tile_r_dim ∈ {1, 2, 4, 8, 16, 32}` against a fixed 32x32 `in1`, collect cycle counts at each `MATH_FIDELITY`, and then reason about where the **perf knee** lies — the crossover point where the simpler pad-and-mask path overtakes a tiny-tile path despite the wasted L1.

**Prerequisites:** Chapter 2 (LLK plumbing — `face_r_dim`, `num_faces`, `PARTIAL_FACE` flags), Chapter 5 (Flash MLA decode case study), Chapter 6 Sections 1 and 2 (L1-budget arithmetic and decision-tree heuristics).

---

## 6.3.1 — Test infrastructure

The cycle-count comparison rides on two files:

1. `tests/sources/math_matmul_perf.cpp` — a parameterised C++ test that compiles into three TRISC images (unpack / math / pack). Each image is split into `INIT` and `TILE_LOOP` profiler zones via `ZONE_SCOPED()` and `PROFILER_SYNC()` so the harness can attribute cycles per thread per zone.
2. `tests/python_tests/perf_math_matmul.py` — the pytest driver that sweeps over `MATH_FIDELITIES`, the matmul format set, and the tiny-tiles tile-dimension list.

The harness has three measurement modes, selected by the `PERF_RUN_TYPE` constexpr threaded through all three kernels:

```cpp
// tests/sources/math_matmul_perf.cpp:158-189
if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)        { return; }
else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE) { _perf_math_matmul_mock(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM); return; }
else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
{
    for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
    {
        for (std::uint32_t j = 0; j < KT_DIM; j++)
        {
            _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(DST_INDEX, CT_DIM, RT_DIM);
        }
    }
}
else /* L1_TO_L1 or L1_CONGESTION — full unpack+math+pack */
```

`MATH_ISOLATE` strips out the unpack and pack threads (the math thread feeds itself a mock register stream) so the MVMUL latency is measured in isolation. `L1_TO_L1` is the full three-thread pipeline, where unpack-from-L1 and pack-to-L1 are on the critical path. The pair of modes lets us decompose total cycles into **compute** and **bandwidth** contributions — the heart of the tiny-tile decision.

The driver in `perf_math_matmul.py` constructs the test matrix:

```python
# tests/python_tests/perf_math_matmul.py:60-85
TINY_TILES_MATMUL_COMBINATIONS = sweep_tiny_tiles_matmul(
    MATMUL_FORMATS,
    DEST_ACC_MODES,
    STOCHASTIC_ROUNDING_MODES,
    DEST_SYNC_MODES,
    math_matmul=True,
)

ALL_TEST_PARAMS = list(
    chain(
        (
            (fidelity, combinations, 0)  # throttle=0 for all tiny-tile runs
            for fidelity, combinations in product(
                MATH_FIDELITIES, TINY_TILES_MATMUL_COMBINATIONS
            )
        ),
    )
)
```

`sweep_tiny_tiles_matmul()` delegates to `generate_matmul_tiny_tiles_combinations()` in `helpers/matmul_sweep.py:195`, which enumerates `in0_tile_r_dim ∈ {1, 2, 4, 8, 16}` with `in1` fixed at 32x32. The 32x32-on-32x32 baseline comes from the regular `sweep_matmul()` (commented out by default in `ALL_TEST_PARAMS` to keep CI runtime bounded, but easy to re-enable for a one-shot perf study).

The cycle counts themselves arrive through `pytest --perf`, which wires the device profiler dump into the `perf_report` fixture (`perf_math_matmul.py:89`). Per zone (`INIT`, `TILE_LOOP`) you get start/end timestamps per RISC-V thread, which are differenced to produce the cycle delta. The `LOOP_FACTOR` (defaulting to 1024 for the extended-perf runs) amortises one-off init costs so the per-iteration cycle figure has good signal-to-noise.

To reproduce:

```bash
# from tt-metal/tt_metal/tt-llk/tests/
CHIP_ARCH=blackhole pytest --compile-producer -n 8 -x python_tests/perf_math_matmul.py
CHIP_ARCH=blackhole pytest --compile-consumer    -x python_tests/perf_math_matmul.py --perf
```

The first pass JIT-compiles every (fidelity, format, dimension) variant; the second runs them on a single Tensix core and emits the per-variant cycle table.

---

## 6.3.2 — What the cycle counts encode

A 32x32 tile is composed of four 16x16 faces (`num_faces=4`, `face_r_dim=16`). A tiny tile reduces `num_faces` and/or `face_r_dim` to one of the legal combinations from `tensor_shape.h:87-94`:

| Effective rows | `num_faces` | `face_r_dim` | Faces touched |
|---:|---:|---:|---:|
| 1 | 1 | 1 | 1 |
| 2 | 1 | 2 | 1 |
| 4 | 1 | 4 | 1 |
| 8 | 1 | 8 | 1 |
| 16 | 1 | 16 | 1 |
| 16 | 2 | 8 | 2 (16x32 wide) |
| 32 | 4 | 16 | 4 (canonical) |

Inside `_llk_math_matmul_`, MVMUL operates on 16-element row vectors against a 32-wide weight stripe. The math thread iterates over rows of the in0 face — `face_r_dim` iterations per face — and over `num_faces` faces. This means **math cycles scale roughly linearly in the total row count touched**, but with a non-trivial fixed prologue per call (RWC programming, dst-pointer advance, fidelity-loop wrap for HiFi modes).

The implication for cycle-count interpretation is twofold:

1. In **MATH_ISOLATE** mode, a 1x32 matmul does *not* take 1/32 the cycles of a 32x32 matmul. Per-call fixed overhead and per-face fixed overhead are amortised over fewer rows, so cycles-per-output-row is *higher* for tiny tiles.
2. In **L1_TO_L1** mode, the unpack thread reads `tile_size = face_r_dim * num_faces * 16 * bpp` bytes from L1 instead of the full `32*32*bpp`. Unpack and pack cycles drop nearly linearly. The fixed math overhead becomes a smaller fraction of the total because the surrounding I/O shrank too.

The "win" for tiny tiles is therefore concentrated in the bandwidth-bound regime — and that is precisely the regime Flash MLA decode operates in (see Chapter 5).

---

## 6.3.3 — Expected cycle-count pattern

The exact cycles depend on architecture (BH vs. WH), `MATH_FIDELITY`, and `DestAccumulation`, but the qualitative shape is consistent. The table below shows the *relative* cycle cost (32x32 LoFi normalised to 1.00) that the perf suite is designed to surface. Values are rough estimates for an in0-tile-height sweep against a fixed 32x32 in1, K_DIM=1, on Blackhole at FP16-b:

| in0 rows | `num_faces`/`face_r_dim` | MATH_ISOLATE (LoFi) | MATH_ISOLATE (HiFi4) | L1_TO_L1 (LoFi) | L1_TO_L1 (HiFi4) | L1 traffic per call |
|---:|:---|---:|---:|---:|---:|---:|
|  1 | 1 / 1   | ~0.30 | ~0.55 | ~0.20 | ~0.45 |  64 B |
|  2 | 1 / 2   | ~0.32 | ~0.58 | ~0.22 | ~0.47 | 128 B |
|  4 | 1 / 4   | ~0.36 | ~0.62 | ~0.26 | ~0.51 | 256 B |
|  8 | 1 / 8   | ~0.45 | ~0.70 | ~0.34 | ~0.58 | 512 B |
| 16 | 1 / 16  | ~0.62 | ~0.82 | ~0.52 | ~0.72 |1024 B |
| 16 | 2 / 8   | ~0.62 | ~0.82 | ~0.52 | ~0.72 |1024 B |
| 24*| pad→32  | ~1.00 | ~1.00 | ~1.00 | ~1.00 |2048 B |
| 32 | 4 / 16  |  1.00 |  1.00 |  1.00 |  1.00 |2048 B |

\*24-row case is "pad-and-mask" — the kernel runs a full 32x32 matmul and masks the bottom 8 rows. Cycle count is by definition identical to the 32x32 baseline; the cost is the wasted L1 footprint (8 rows of useless output and 8 rows of useless intermediate) and downstream consumers having to either replicate the mask or tolerate the garbage.

These numbers are **estimates intended to set expectations**, not measured ground truth. The real values move with cache warm-up, NOC contention, dest-sync mode, throttle level, and per-arch MVMUL pipeline depth. Always rerun the suite on the target part before quoting numbers in a design doc.

What matters is the *shape*:

- **Math-isolate cycles grow sub-linearly in row count.** Doubling rows from 8 → 16 in MATH_ISOLATE costs ~38% more cycles, not 100%, because the per-call prologue dominates at small sizes.
- **L1-to-L1 cycles grow more steeply** because they include the linearly-scaling unpack/pack traffic on top of the math-isolate envelope.
- **Higher fidelity flattens the tiny-tile advantage.** At HiFi4 each MVMUL replays four times for `(weight, activation)` sub-products. The per-iteration cost is dominated by those replays, not by row count, so the relative cost of a 1x32 vs a 32x32 narrows considerably (~55% vs ~100%).

---

## 6.3.4 — The perf knee

The "knee" is the row count at which `cycles_tiny(rows) >= cycles_padded(32)`. Below the knee, tiny tiles are strictly faster *and* cheaper in L1. Above the knee, padding becomes faster (cycles-wise) and the only reason to keep using tiny tiles is L1 pressure.

From the table above, two cases emerge:

1. **Compute-bound (MATH_ISOLATE, HiFi4).** Tiny 16x32 already costs ~82% of the 32x32 baseline. The marginal MVMUL cost of going 16 → 32 rows is small because HiFi4 amortises overhead across four replay iterations. A 24x32 padded-to-32 case costs 100% — only 18% slower than tiny 16. **Knee sits around 16–20 effective rows.** Below that, tiny tiles still win on compute. Above, padding wins.
2. **Bandwidth-bound (L1_TO_L1, LoFi).** Tiny 16x32 costs ~52% of the 32x32 baseline. The unpack thread is reading half the bytes, and at LoFi the math thread can stay ahead of unpack with cycles to spare. **No knee** — tiny tiles win at every row count where they're legal, because the L1-to-dst traffic scales linearly with `face_r_dim * num_faces`.

The general rule: **tiny tiles always win for bandwidth-bound ops**, regardless of row count, because L1 traffic scales linearly with the tile footprint. For compute-bound ops, the knee is empirically around **16–24 rows**; below that, the per-call fixed overhead is small enough that tiny tiles still come out ahead, and above that the wasted math work in a padded 32x32 is less than the wasted MVMUL prologues in a tiny call.

---

## 6.3.5 — Flash MLA as a worked example

Flash MLA decode (Chapter 5, Section 3) is the canonical bandwidth-bound case. The kernel streams 32x32 BFP8 K-chunks from DRAM through `cb_k_in` and matmuls each chunk against the cached Q rows. With a tiny 8x32 Q layout:

- `cb_q_in` per-core size: 9,216 B (18 tiles × 512 B). Padding Q to 32x32 would balloon this to 36,864 B (4×).
- `cb_k_in` per-core size: 156,672 B (144 tiles × 1,088 B BFP8). K is unchanged by tiny Q.
- Total Q-side CB inflation if we padded: roughly **+65 KB per core, a 32% L1 overhead** on the 204.5 KB tiny-Q budget.

The matmul itself is `(8x32) × (32x32) → (8x32)`. From the table above, at L1_TO_L1 LoFi this costs ~0.34 of the 32x32 baseline. So **tiny Q saves both ~34% on compute *and* ~65 KB on L1**. Both savings compound — the L1 headroom enables deeper K double-buffering, which keeps the unpack thread busier, which keeps the math thread fed. That virtuous cycle is why every effort to "simplify" Flash MLA by re-padding Q to 32x32 has regressed both perf and L1.

The opposite regime — long-sequence prefill — would tilt the math balance. There, K is large enough that the matmul becomes compute-bound; the MVMUL prologue overhead of many 8x32 calls starts to add up. The decision tree (Chapter 6, Section 4) accounts for this: if sequence-length and head-dim cross a threshold, even MLA may prefer 32x32 Q.

---

## 6.3.6 — Caveats and gotchas

- **Cache warm-up.** The first iteration of any `LOOP_FACTOR` sweep pays cold-cache costs. With `LOOP_FACTOR=1024` this is amortised to <0.1% noise, but at smaller loop counts the warm-up shows up as a fixed adder.
- **NOC contention.** `L1_TO_L1` runs over a single Tensix core, so there is no inter-core NOC traffic. Real workloads (Flash MLA spans 64+ cores) see additional latency from NOC arbitration that the suite does not model. Use `L1_CONGESTION` mode (selectable via `PerfRunType`) to inject synthetic NOC traffic and see how it shifts the curves.
- **`DestSync::Half` vs. `Full`.** Half-sync exposes more parallelism between math and pack but halves the available dest slots. For tiny-tile chains, half-sync is usually a win because the slot fragmentation (see Chapter 6 Section 2) is less painful when you only need 4 slots active at a time.
- **Throttle level.** The tiny-tile sweep runs with `throttle=0` (`perf_math_matmul.py:79`). Increasing the throttle reduces math thread aggressiveness and tightens the compute envelope; rerun with `throttle ∈ {1..5}` if you are studying a thermal-constrained part.
- **Pad-and-mask is not free.** The "padded 32x32" baseline above assumes the downstream consumer either ignores or cheaply masks the bottom rows. If the consumer must apply an explicit SFPU mask (additional `_llk_math_eltwise_unary_sfpu_` call), add those cycles to the padded-row cost; that typically shifts the knee left by 4–8 rows.

---

## 6.3.7 — Recipe for a comparison study

1. Identify the candidate op (e.g., your new fused decode kernel).
2. Sketch its matmul as `(M x K) × (K x N) → (M x N)` and identify the dimension that would shrink under a tiny tile (usually M, the activation height).
3. Estimate K-side bandwidth: bytes per matmul × K_chunks_per_step. If K bandwidth dominates the matmul latency budget, you are bandwidth-bound and tiny tiles always win.
4. Run `pytest --perf perf_math_matmul.py` with `in0_tile_r_dim` swept over the candidate row count and the nearest power-of-two above it, at the fidelity you ship.
5. Take the L1_TO_L1 cycle delta and multiply by your op's expected per-step matmul count. Compare to the L1 savings (from Chapter 6 Section 1's budget table).
6. If both deltas favour tiny, ship tiny. If only L1 favours tiny and your part is L1-constrained, ship tiny. If only compute favours tiny and your part is compute-constrained, ship padded. If both favour padded, ship padded — you are above the knee.

The decision tree in the next section formalises this recipe.

---

**See also:** Chapter 6 Section 1 (L1-budget arithmetic from Flash MLA), Chapter 6 Section 2 (dest-register slot allocation), Chapter 6 Section 4 (the decision tree), Chapter 5 Section 3 (Flash MLA decode case study). For the underlying LLK plumbing of fidelity replay and partial-face flags, see [[introduction_to_tt_llk]].
