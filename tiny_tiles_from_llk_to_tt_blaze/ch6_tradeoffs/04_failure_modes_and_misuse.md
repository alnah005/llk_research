# 6.04 — Failure Modes and Misuse

Tiny tiles violate an unwritten assumption that pervades the LLK stack: that a "tile" is always 32x32. Most of the code is correct under this assumption, and the parts that aren't are guarded by `validate_tensor_shape_tile_dependent_ops_()`. The remaining failure surface is narrow but sharp — five recurring bug classes account for nearly every tiny-tile incident on Wormhole and Blackhole. This section catalogs each, walks through symptoms and root causes from real Flash MLA and matmul-perf code paths, and lists the debugging tools that actually work.

**Prerequisites**: Chapter 2 (LLK tiny-tile primitives, especially SFPU iteration count), Chapter 3 (CB descriptors and TileDescriptor plumbing), Chapter 5, Section "Flash MLA L1 budget" (CB table that drives the page-size discussion here), and Chapter 6, Section "Decision tree" (which sets up when you'd reach for tiny tiles in the first place).

---

## 1. Hang on `cb_wait_front` — CB page-size mismatch

**Symptom.** The kernel boots, the reader fires off NoC transactions, the writer never advances, and the device sits in `cb_wait_front(cb_id, num_tiles)` until the host watchdog times out. No assert. No DPRINT past the wait. tt-triage shows the math thread parked on a CB semaphore that never increments.

**Root cause.** A CB has a single page size baked into its `CBFormatDescriptor`. Producer and consumer must agree on that page size or the wait condition is structurally unsatisfiable. With tiny tiles this is easy to get wrong because the page size is computed from the tile descriptor, not the dtype alone:

- Producer pushes an 8x32 BF16 page = 8 * 32 * 2 = 512 B.
- Consumer was configured with the default 32x32 BF16 page = 32 * 32 * 2 = 2048 B.
- `cb_wait_front(cb, 1)` waits for 2048 B of valid data. The producer never delivers more than 512 B per push. Deadlock.

In TT-Blaze the CBHandle carries the `TileDescriptor` end-to-end, so this bug class collapses to a single misuse: constructing two different `CBFormatDescriptor`s for the same logical CB. In raw LLK on tt-metal, it's much easier to hit — the producer and consumer are usually separate compile units, and the tile geometry has to be threaded through CT args manually.

**Example.** Flash MLA's Q-input CB explicitly carries a tiny tile descriptor:

```python
# models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:536-543
q_tiny_tile = ttnn.Tile((Q_TILE_HEIGHT, TILE_WIDTH))  # (8, 32)
q_tile_descriptor = ttnn.TileDescriptor(q_tiny_tile)
cb_q_format = ttnn.CBFormatDescriptor(cb_q_in, q_df, q_tile_size, q_tile_descriptor)
```

Compare to `cb_k_in`, which is a full-tile BFP8 K-cache CB and uses the default descriptor:

```python
# models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:744
ttnn.CBFormatDescriptor(cb_k_in, k_df, k_tile_size)  # no tile_descriptor -> default 32x32
```

Mixing these — e.g. writing into `cb_q_in` from a kernel compiled with full-tile CT args — would produce 2048 B pushes against a 512 B page, which fails differently: the CB pointer arithmetic walks off the page boundary and the next reader sees stale data.

**Fix.** Treat the `TileDescriptor` as the single source of truth. In TT-Blaze, share the CBHandle. In hand-rolled LLK, route the same `(num_faces, face_r_dim, face_c_dim)` triple to every kernel that touches the CB via the CT-arg generator, and grep `named_args_generated.h` to confirm.

**Debugging.**
- Enable `BLAZE_DEBUG_KERNELS` and look for the last DPRINT before the stall — almost always immediately before a `cb_wait_front`.
- Run `BLAZE_L1_PROFILE` and diff the CB descriptor blocks: producer's and consumer's page sizes must be byte-identical.
- Inspect `build/.../named_args_generated.h` for both kernels and confirm `cb_X_tile_height` / `cb_X_face_r_dim` match.
- If the CB has multiple producers (rare but possible), every producer must push the same page size.

---

## 2. PCC mismatch with no hang — wrong SFPU iteration count

**Symptom.** The matmul output ahead of the SFPU is fine (PCC ~ 1.0 against torch golden). The output after the SFPU drops to PCC ~ 0.5 or lower. Looking at the raw output, the first 16 rows are correct and the last 16 rows are garbage — uninitialized dest data, the previous tile's residue, or zeros depending on what dest looked like on entry.

**Root cause.** SFPU operations are face-iterated. The template parameter `N` (often spelled `iterations`, `vector_mode`, or just an integer) tells the SFPU how many 16-row faces to process. A 32x32 tile has 4 faces; an 8x32 tile has 2 faces (both top-half F0 and F1, since `face_r_dim=8 < 16` still occupies two 16-wide column faces). The face count along the row axis collapses to one face row for any tile up to 16 rows tall, but there are still two faces across the 32-wide column dimension.

The lookup the kernel author needs:

| Tile shape | num_faces | face_r_dim | SFPU iterations N |
|------------|-----------|------------|-------------------|
| 1x32       | 2         | 1          | 2 |
| 2x32       | 2         | 2          | 2 |
| 4x32       | 2         | 4          | 2 |
| 8x32       | 2         | 8          | 2 |
| 16x32      | 2         | 16         | 2 |
| 32x32      | 4         | 16         | 4 |
| 16x16      | 1         | 16         | 1 |

Equivalently, `N = num_faces`. The bug is hardcoding `N=4` (because that's what 32x32 wanted) or `N=1` (because "the tile is only 1 row, surely one iteration is enough"). Both produce silent wrong-output.

**Example.** The fused-sigmoid path in the matmul micro-op uses an explicit 2 for tiny-tile geometries:

```cpp
// tt_llk/.../unified_kernels/matmul.hpp (excerpt)
_llk_math_eltwise_unary_sfpu_sigmoid<APPROX_MODE, /*is_fp32=*/false, /*iterations=*/2>();
```

For Flash MLA, the in-kernel exponentiation over the 8x32 attention scores uses `iterations=2` for the same reason (see `unified_kernels/flash_mla.hpp` around lines 166-180).

**Fix.** Compute N from the tile descriptor at compile time: `constexpr int N = (TILE_R + 15) / 16 * NUM_COL_FACES;` or just propagate `num_faces` directly from the CT args.

**Debugging.**
- Use `/cb-tap` to snapshot the CB immediately after the SFPU and before the packer. If rows 16-31 are garbage, the SFPU ran too few iterations.
- If rows 0-15 are also wrong, the bug is upstream (matmul) — not an SFPU iteration issue.
- DPRINT the SFPU template parameter at runtime via a static_assert message — confirms the kernel was compiled with the value you think it was.

---

## 3. Illegal-config trap — validator accepts, hardware mis-handles

**Symptom.** The kernel compiles, host-side validation passes, the dispatch goes through, and the result is either a hang in an unexpected place (often the unpacker or packer) or a corrupted output that doesn't match any of the standard failure signatures above. The combination is one the test matrix never exercised.

**Root cause.** `validate_tensor_shape_tile_dependent_ops_()` is the API-level gate:

```cpp
// tt_metal/tt-llk/common/tensor_shape.h:87-94
constexpr bool validate_tensor_shape_tile_dependent_ops_() {
    return (num_faces == 1 || num_faces == 2 || num_faces == 4)
        && (face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4
            || face_r_dim == 8 || face_r_dim == 16)
        && (face_c_dim == 16);
}
```

This is necessary but not sufficient. The combinatorial space `{1,2,4} x {1,2,4,8,16}` = 15 configs, but several combinations are only meaningful for certain ops. A `(num_faces=1, face_r_dim=1)` "tile" — a single 1x16 face — passes the validator but may exercise an untested unpacker stride on a particular arch.

**Mitigation.** Stick to the configs the in-tree test sweeps cover. `generate_matmul_tiny_tiles_combinations()` in `tt-llk/tests/python_tests/matmul_sweep.py` (line ~195) enumerates the *exercised* set, which is a strict subset of the *validated* set. Anything outside that subset should be treated as research-grade until you've written a unit test and a torch golden comparison.

**[Q] note.** Quasar's LLK layer (`tt_llk_quasar/llk_lib/`) is the least-audited of the three for tiny-tile support — grep for `face_r_dim` and `num_faces` guards in the unpacker, math, and packer headers before relying on a non-standard config there.

**Debugging.**
- Confirm the offending config passes `validate_tensor_shape_tile_dependent_ops_()` with a `static_assert`. If it doesn't, the validator caught it for you.
- Cross-reference the actual test sweep used in CI (`perf_math_matmul.py`, `matmul_sweep.py`). If your config isn't there, write a minimal repro and file a bug.
- Use `/port-kernel` or `/arch-lookup` to compare unpacker/packer behavior across architectures for your specific `(num_faces, face_r_dim)`.

---

## 4. Silent corruption — transpose + tiny tile + partial face

**Symptom.** Output matches golden for some input geometries and diverges for others. The failure is deterministic — same inputs produce the same wrong outputs — but the *shape* of the failure varies with tile dimensions. PCC may be 0.99 (close but not bit-exact) rather than catastrophic.

**Root cause.** The unpacker has two largely-orthogonal axes that interact poorly when both are exercised: (a) tile transpose, and (b) partial-face mode (when `face_r_dim < 16`). The test sweep covers:

- Tile transpose on full 32x32 tiles (well-tested).
- Tiny tiles without transpose, with `in0` varied and `in1` fixed at 32x32 (well-tested).
- Tiny tiles with transpose on `in0` only (partially tested).

The gap is **transpose on `in1` while `in1` is itself tiny** — the unpacker stride calculation and the math thread's face-iteration order both have to invert correctly, and that interaction is the least-covered code path. `generate_matmul_tiny_tiles_combinations()` intentionally fixes `in1` at 32x32 for exactly this reason: in1's width is the dominant unpacker stride, and varying it changes the inner loop in ways the test matrix can't fully cover.

**Mitigation.**
- Constrain tiny tiles to `in0` for matmul. This is the convention Flash MLA follows (Q is tiny, K is full).
- If you genuinely need a tiny `in1`, write a directed test that sweeps `(transpose, in1_height, partial_face)` and compares against a torch golden (tilize with matching geometry, then matmul, then untilize).
- Don't trust PCC alone — diff individual rows. Silent corruption often spares the top face and breaks the bottom one, which averages out to a deceptively healthy PCC.

**Debugging.**
- `/bisect-fused` to find the first CB where divergence appears.
- `/cb-tap` on the unpacker output (before math) to confirm whether the corruption is on the unpack side or the math side.
- Run with `transpose=false` first; if that fixes it, the bug is in the transpose-tiny interaction.

---

## 5. Dest-slot starvation

**Symptom.** A chain of tiny-tile operations runs slower than expected. Cycle count is super-linear in the chain length. The kernel is not L1-bound (`BLAZE_L1_PROFILE` shows headroom), not compute-bound (matmul utilization is low), and not bandwidth-bound (NoC traffic is light). Phase markers from `BLAZE_DEBUG_KERNELS` show repeated waits on dest acquire.

**Root cause.** Destination register slots are tile-granular, not row-granular. A 1x32 tile occupies the same dest slot as a 32x32 tile. The slot counts are fixed by mode:

- 16-bit `DstSync::Full`: 16 slots; `DstSync::Half`: 8 slots.
- 32-bit (FP32): 8 and 4 slots respectively.

A chain producing many tiny tiles back-to-back fills slots that *could* have held full tiles, but the math thread can't free a slot until the packer consumes it. If the packer is slower than the math thread (which is common with tiny tiles, because packing 8 rows takes proportionally less time than producing them on math, but the packer still has setup overhead), slots accumulate and the math thread stalls on `tile_regs_acquire`.

**Counterintuitive corollary.** Padding to 32x32 can be *faster* than tiny tiles when dest is the bottleneck, even though it does 4x the math work. The pad cost is amortized over the slot turnover.

**Mitigation.**
- Reorder the chain so a heavier op (e.g. a 32x32 reduce) sits between tiny-tile producers, giving the packer time to drain.
- Use `DstSync::Full` (16 slots) over `Half` (8 slots) when the dependency pattern allows.
- If the chain is short and L1 has headroom, just pad. The L1 cost of tiny tiles is only worth it when L1 is the binding constraint.

**Debugging.**
- Sweep `in0_tile_r_dim` in `{1, 2, 4, 8, 16, 32}` using `perf_math_matmul.py` and plot cycles vs. tile height. Linear is good; super-linear at small heights points to dest starvation.
- Look for repeated "wait on dest" phase markers under `BLAZE_DEBUG_KERNELS`.
- Compare against a padded baseline. If padded is faster, dest was the bottleneck.

---

## Summary debugging checklist

When a tiny-tile kernel misbehaves, run through this in order. Most issues fall out by step 3.

1. **Confirm CB geometry agreement.**
   - Grep `named_args_generated.h` for the suspect CB's tile dimensions in both producer and consumer.
   - Compare page sizes from `BLAZE_L1_PROFILE`.
   - In TT-Blaze, verify a single `CBHandle` is shared end-to-end.

2. **Verify SFPU iteration counts.**
   - Every SFPU template instantiation: `iterations` must equal `num_faces` for the operand tile.
   - `/cb-tap` after each SFPU to localize.

3. **Check the config is in the test sweep.**
   - `tt-llk/tests/python_tests/matmul_sweep.py` — `generate_matmul_tiny_tiles_combinations()`.
   - If not present, write a directed test.

4. **Rule out transpose + tiny + partial-face interactions.**
   - Disable transpose; if the bug vanishes, the unpacker is the culprit.
   - Prefer keeping `in1` at 32x32 for matmul.

5. **Profile for dest-slot starvation.**
   - Sweep tile height; look for super-linear cycle counts.
   - Compare against a padded baseline.

**Environment variables and tools to keep in your shell history.**

- `BLAZE_DEBUG_KERNELS=1` — emits phase markers on every CB wait, dest acquire, and SFPU entry/exit.
- `BLAZE_L1_PROFILE=1` — dumps the L1 allocator state and CB descriptors at kernel launch.
- `TT_METAL_DPRINT_CORES=0,0` — narrows DPRINT to a single core when the chain spans many cores.
- `build/.../named_args_generated.h` — the compiled CT args. The truth about what the kernel was actually built with.
- `/bisect-fused` — finds the first CB along a fused chain where output diverges from golden.
- `/cb-tap` — snapshots a single CB to host for direct comparison.
- `/inline-copy-fused` — flattens a production fused op into an inline test twin so taps can be inserted without polluting production code.

Most tiny-tile bugs reduce to one of the five above, and four of the five are CB-geometry or face-count bookkeeping errors at heart. Always validate `TileDescriptor` consistency between producer and consumer.
