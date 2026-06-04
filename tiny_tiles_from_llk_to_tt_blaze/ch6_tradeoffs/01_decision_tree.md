# Decision Tree: Tiny Tiles vs. Pad-and-Mask

Kernel authors face a recurring choice: accept the complexity of a non-32x32 tile path (tile-descriptor plumbing, SFPU iteration accounting, CB page-size synchronization), or pad the small dimension to 32 and discard the wasted rows after the fact. This file ranks the five criteria that drive that choice in order of decisiveness — L1 pressure first, dest-register slot accounting last — and ends with a recommendation matrix derived from Flash MLA's tiny-Q decode op.

**Prerequisites:** Chapter 1 (tiny-tile definition and legal geometries), Chapter 2 (LLK unpack/math/pack data path), Chapter 5 (Flash MLA decode usage of 8x32 tiles).

## The Five Criteria, Ranked

The ranking below reflects how each criterion behaves in practice: L1 pressure is binary (the kernel either fits or it doesn't), while compute-vs-bandwidth boundedness is a tuning concern that only matters once everything else is decided.

1. **L1 pressure** — does the kernel fit on the core at all?
2. **Effective-row count** — how much L1/dest does padding actually waste?
3. **Downstream consumer geometry** — does padding now mean re-tiling later?
4. **Dest-register reuse** — does the tiny-tile chain underutilize Dst?
5. **Compute-bound vs. bandwidth-bound** — does shrinking the tile shrink the cycle count?

### 1. L1 Pressure (decisive)

If the activation chain has many double-buffered intermediates and the kernel is already brushing the L1 budget, tiny tiles can be the difference between fitting and not fitting. This is a yes/no criterion: a kernel that doesn't fit doesn't run.

Flash MLA decode is the canonical example. With tiny Q (8x32), the per-core CB budget is approximately 204 KB:

| CB | Tile geometry | Pages | L1 size | Role |
|----|---------------|-------|---------|------|
| `cb_q_in` | 8x32 | 18 | 9,216 B | Q input |
| `cb_k_in` | 32x32 BFP8 | 144 | 156,672 B | Double-buffered K chunks |
| `cb_mask` | 8x32 | 1 | 512 B | Causal mask |
| `cb_ms_in` | 8x32 | 3 | 1,536 B | Tree-reduction stats |
| `cb_out_in` | 8x32 | 48 | 24,576 B | Tree-reduction O data |
| `cb_out_o` | 8x32 | 16 | 8,192 B | Output O |
| `cb_out_ms` | 8x32 | 1 | 512 B | Output m/s |
| `cb_interm_out` | 8x32 | 16 | 8,192 B | Intermediate O (aliased) |
| `cb_interm_ms` | 8x32 | 1 | 512 B | Intermediate m/s (aliased) |
| `cb_out_final` | 8x32 | per spec | 8,192 B | Final output |

Total: ~204.5 KB per core, with the BFP8 K chunks dominating at 75% of the footprint.

If every Q-side CB is inflated to 32x32, the same kernel needs roughly 280 KB per core — a 65 KB / 32% overhead. [BH] On P150 with ~1.3 MB usable L1 per core, that's 5% of the budget recovered, which is enough headroom to double-buffer an additional K chunk or carry an extra intermediate. The point is not the absolute saving but that it is structural: padding here costs you a doublings worth of pipeline depth.

**Rule:** if the padded variant blows the L1 budget (or eats the slack you need for pipeline depth), the L1 question alone forces tiny tiles.

### 2. Effective-Row Count (next decisive)

Once L1 fits, the next question is how much you are actually wasting by padding. The waste fraction is `1 - (effective_rows / 32)`:

| Effective rows | Waste | Verdict (default) |
|----------------|-------|-------------------|
| 1 | 97% | Tiny tiles |
| 2 | 94% | Tiny tiles |
| 4 | 88% | Tiny tiles |
| 8 | 75% | Tiny tiles (Flash MLA's case) |
| 16 | 50% | Borderline — tiny if downstream agrees |
| 24 | 25% | Pad, unless L1 already forces tiny |
| 32 | 0% | Full tile (always) |

The legal heights are constrained by `validate_tensor_shape_tile_dependent_ops_()`:

```cpp
// tt_metal/tt-llk/common/tensor_shape.h:87
constexpr bool validate_tensor_shape_tile_dependent_ops_(uint32_t num_faces,
                                                         uint32_t face_r_dim,
                                                         uint32_t face_c_dim) {
    return (num_faces == 1 || num_faces == 2 || num_faces == 4) &&
           (face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 ||
            face_r_dim == 8 || face_r_dim == 16) &&
           face_c_dim == 16;
}
```

Effective heights are therefore restricted to {1, 2, 4, 8, 16, 32} and widths to {16, 32}. Twenty-four heads per core can be padded to 32 but cannot be represented as a tiny tile directly — there is no 24x32 geometry in the validator.

**Rule:** below 16 effective rows, the waste is high enough that tiny tiles repay the plumbing cost. At exactly 16, it's a coin flip determined by downstream consumers and L1. At 24 or above, padding is usually cleaner.

### 3. Downstream Consumer Geometry

A tiny-tile producer feeding a 32x32 consumer pays the cost twice: once to plumb the tile descriptor on the producer side, and again to re-tile (or pad) on the consumer side. Conversely, when the immediate consumer is also tiny, the savings compound.

Flash MLA's `sdpa_custom_mm_block(Q[8,32]_tiles, K_chunk[32,32]_tiles, transpose_k=true)` is a mixed-geometry matmul: `_llk_unpack_AB_matmul_init_<>()` is configured so unpacker A pulls an 8x32 Q tile and unpacker B pulls a 32x32 K tile into the same FPU. Without tiny-tile support, Q would be padded to 32x32 and rows 9–32 would do useless MVMULs against K.

When the downstream is a full-width matmul against a fixed weight (e.g., an output projection), padding earlier may be simpler than re-tiling. The reverse case — a tiny SoftMax feeding a tiny re-weighting — is where the compounding savings show up.

**Rule:** match the producer geometry to the immediate consumer. If they disagree, pay the cost only at the boundary that is most expensive to inflate.

### 4. Dest-Register Slot Reuse

Tiny tiles do **not** free dest slots. The dest register is allocated tile-granular, not row-granular: every tile — full or tiny — occupies one full slot sized for 32x32.

```
DstSync::Full:                       DstSync::Half:
  16 slots (16-bit mode)               8 slots (16-bit mode)
   8 slots (FP32 mode)                 4 slots (FP32 mode)

  +------+ <-- one 8x32 tile          A chain of 12 tiny outputs
  | tile |     occupies one full       still needs 12 slots,
  | slot |     slot (rows 9..32        which exceeds Full=8 in
  | size |     are unused dest         FP32 mode.
  | 32x32|     space)
  +------+
```

Corollary: a chain producing many tiny-tile outputs back-to-back can stall waiting for dest to drain even though only a quarter of each slot carries data. Three full-tile outputs (3 slots) often pipeline better than twelve tiny-tile outputs (12 slots), even though the latter touches less data.

**Rule:** if you are producing many tiny tiles in flight (≥ 8 outstanding), check that DstSync mode and dest occupancy let them fit without back-pressuring the math pipeline.

### 5. Compute-Bound vs. Bandwidth-Bound

Matmul cycle counts are dominated by the K-dimension MVMUL loop, not by M. An 8x32 by 32x32 matmul issues roughly the same number of MVMULs as a 32x32 by 32x32 matmul for the same K (the MVMUL unit operates on 16-element vectors, and 8 rows still need 2 face-iterations on the M-side).

- **Bandwidth-bound ops** (e.g., a 1x32 projection from a streamed KV cache): L1 footprint dictates fill time. Tiny tiles shrink the activation CB, which directly shrinks the time to refill, which directly shrinks end-to-end latency. Flash MLA is in this regime — the K chunks are streamed from DRAM, and the 9,216 B vs. 36,864 B Q footprint translates to faster refills.
- **Compute-bound ops** (e.g., a 1x32 matmul against a 1024x32 weight): MVMUL iterates over 1024 K-positions regardless of M. Shrinking M from 32 to 1 saves ~25 KB of output L1 but does not reduce cycle count meaningfully.

**Rule:** tiny tiles help most when the bottleneck is L1 bandwidth or DRAM fill time. If profiling shows MVMUL utilization > 80%, expect smaller wins from tiny tiles.

## Flowchart

```
            +------------------------------+
            | Does padded variant fit L1?  |
            +------------------------------+
                 | no              | yes
                 v                 v
        +----------------+   +------------------------------+
        | TINY (forced)  |   | Effective rows ≥ 24?         |
        +----------------+   +------------------------------+
                                 | yes             | no
                                 v                 v
                        +----------------+  +----------------------+
                        | PAD (default)  |  | Downstream tiny too? |
                        +----------------+  +----------------------+
                                                 | yes      | no
                                                 v          v
                                        +----------+   +--------------------+
                                        | TINY     |   | > 8 tiny outputs   |
                                        +----------+   | outstanding?       |
                                                       +--------------------+
                                                          | yes      | no
                                                          v          v
                                                +--------------+  +--------+
                                                | Reconsider   |  | TINY   |
                                                | (dest stalls)|  |        |
                                                +--------------+  +--------+
```

## Recommendation Matrix

| Effective rows | L1 pressure | Downstream | Recommendation |
|----------------|-------------|------------|----------------|
| 1x32 | any | any | Tiny |
| 8x32 (Flash MLA) | tight | tiny matmul | Tiny |
| 16x32 | tight | tiny | Tiny |
| 16x32 | loose | full | Pad |
| 24 effective (no 24x32 exists) | any | any | Pad to 32x32 |
| 32x32 | any | any | Full tile |

## Common Failure Modes Tied to This Decision

Two failures dominate when authors choose tiny tiles without committing to the full plumbing:

- **CB page-size mismatch.** A producer writes an 8x32 page (512 B BF16) but the consumer waits on a 32x32 page (2,048 B). The CB never fills to the consumer's expected size and the kernel hangs. Root cause: the `ttnn.TileDescriptor(q_tiny_tile)` declared at the op layer (Flash MLA's `op.py:548`) was not propagated to every CB in the chain.
- **SFPU iteration count mismatch.** SFPU ops iterate face-by-face; the face count for any tile with face_r_dim ≤ 16 is N=2 (a single face row), while a 32x32 tile is N=4 (two face rows). Hard-coding N=4 in a SoftMax that runs on 8x32 tiles double-processes garbage in faces 2 and 3.

Both failures are symptoms of partial commitment. If criteria 1–3 say "tiny," commit fully and propagate the tile descriptor end-to-end; if criteria 4–5 give pause, pad. The intermediate path — tiny in some CBs and full in others without explicit re-tile boundaries — is the unstable equilibrium that produces these hangs.
