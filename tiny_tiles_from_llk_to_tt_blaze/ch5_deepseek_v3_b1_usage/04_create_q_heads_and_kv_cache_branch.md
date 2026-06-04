# Create Q Heads and the KV Cache Branch

Flash MLA decode reads Q and K from two CBs whose tile geometries do not match: Q is `8x32` (tiny) and K is `32x32` (standard). This section walks the two upstream producers that establish that asymmetry — `create_q_heads`, which gathers and tilizes 64 Q heads into `8x32` shards on a 4x2 receiver grid, and the `kv_cache_branch` fused op, which projects, normalizes, and rope-rotates the KV row so it can be appended to a `32x32`-tile DRAM KV cache. The `attention_block` fused op is where both halves meet inside a single program, with CB descriptors carrying tile geometry from producer to consumer.

**Prerequisites.** Chapter 3 (CB descriptors and tile geometry plumbing in tt-metal), Chapter 5 Section "Flash MLA Decode" (Q/K tile asymmetry inside the SDPA kernel), and Chapter 5 Section "RoPE" (the QRoPE producer that feeds `create_q_heads` Phase 3).

## 1. The Q side: tiling 64 heads into 8x32 shards

`create_q_heads` is the bridge between two very different sharding schemes. Its inputs are the outputs of two parallel matmul stages:

- **QNOPE matmul3 output**: block-sharded across an 8x8 grid, each core holds one `[1, 512]` shard (one head, NOPE half).
- **QRoPE output (after RoPE)**: block-sharded across an 8x4 grid, each core holds a `[2, 64]` shard (two heads, RoPE half).

These 96 sender cores feed into 8 receiver cores arranged as a 4x2 grid, where each receiver assembles 8 full Q heads (NOPE 512 + RoPE 64 = 576 elements per head) and tilizes the result on the spot. The row-to-receiver mapping is fixed:

```python
# models/demos/deepseek_v3_b1/micro_ops/create_q_heads/op.py:40-49
SENDER_ROW_TO_TARGET_CORE = {
    0: ttnn.CoreCoord(0, 1),
    1: ttnn.CoreCoord(1, 1),
    2: ttnn.CoreCoord(2, 1),
    3: ttnn.CoreCoord(3, 1),
    4: ttnn.CoreCoord(0, 2),
    5: ttnn.CoreCoord(1, 2),
    6: ttnn.CoreCoord(2, 2),
    7: ttnn.CoreCoord(3, 2),
}
```

So each receiver ends up with the 8 Q heads that originate from a single sender row. After tilization, each receiver holds `[8, 576]` row-major where row `i` corresponds to head `i` — exactly one `8x32` tile in the height dimension and `576 / 32 = 18` tiles in the width dimension. Eighteen `8x32` tiles per receiver, eight receivers, sixty-four heads total.

### Phase-based tilization

The kernel cannot just write 8 contiguous heads of 576 elements into the receiver CB and call `tilize_block` once, because `tilize_block` consumes pages of `face_w = 16` columns at a time and the hardware tilizer has a maximum width per call. The op splits the 576-element row into three phases:

```text
# create_q_heads/op.py:20-23
Memory layout for tilization (max tilize dim = 256):
  - Phase 1: First 8 halves of QNOPE - shape [8, 256] -> 8 tiles
  - Phase 2: Second 8 halves of QNOPE - shape [8, 256] -> 8 tiles
  - Phase 3: QROPE - shape [8, 64] -> 2 tiles
```

Each phase looks like a clean `[8, W]` block to the tilizer:

- Phase 1: `[8, 256]` of all head's first NOPE half (8 tiny tiles).
- Phase 2: `[8, 256]` of all head's second NOPE half (8 tiny tiles).
- Phase 3: `[8, 64]` of all head's RoPE half (2 tiny tiles).

After tilization, the 18 tiles for one head are laid out contiguously in CB pages 0..17 of the output CB. Crucially, the tile height is **8 rows**, not 32 — chosen because there are exactly 8 heads per receiver, and tiling them by row lets each Q head live in row `i` of an `8x32` tile without padding. The L1 cost of this choice is what the rest of Chapter 5 quantifies: an `8x32` tile is 512 B at BFP16 versus 2048 B for a padded `32x32` tile.

### Three-phase synchronization

To make the phases race-free without per-receiver bookkeeping, the op uses three separate semaphores rather than one. QNOPE senders signal twice (after each 256-element half), QRoPE senders signal once:

```python
# create_q_heads/op.py:240-242
nope_phase1_semaphore_id = 2  # QNOPE senders signal after first half
nope_phase2_semaphore_id = 3  # QNOPE senders signal after second half
rope_semaphore_id = 0         # QROPE senders signal after completion
```

These IDs deliberately alias the semaphores used by `pre_sdpa` (`mcast_data_sender_semaphore_id = 0`, `gather_noc0_receiver_semaphore_id = 2`, `gather_noc1_receiver_semaphore_id = 3`) — see the comment block at lines 235–242 of `op.py`. The reuse is safe because by the time `create_q_heads` starts, the earlier ops have already passed those wait points. All data movement runs on NCRISC (`RISCV_1`); BRISC is idle on every core (line 36 of the op docstring).

### CB binding and output tile geometry

The CB descriptors for `create_q_heads` are built directly from the sharded tensors, which carries tile geometry through automatically:

```python
# create_q_heads/op.py:412-421
qnope_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(qnope_cb, qnope_tensor)
qrope_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(qrope_cb, qrope_tensor)

# Receiver input CB: bound to interm_tensor's buffer where senders write row-major data.
# interm_tensor must be TILE_LAYOUT so page_size = tile_size (512 bytes for 8x32 bf16),
# giving 18 tile-sized pages that match the kernel's per-phase tilize_block calls.
receiver_in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(receiver_in_cb, interm_tensor)

# Output CB: bound to output_tensor's buffer for tilized output
out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(out_cb, output_tensor)
```

The intermediate tensor `interm_tensor` is sized so its page size equals the `8x32` tile size (512 B for BFP16): senders write row-major data, the receiver compute kernel reads it page-by-page and calls `tilize_block` three times. The output tensor's tile geometry is `(8, 32)` — and because Flash MLA's Q input is precisely this output tensor, the tile geometry propagates with no additional configuration on the consumer side.

## 2. The K side: KV cache stays 32x32

`kv_cache_branch` projects, normalizes, and RoPE-rotates the KV row that gets appended to the cache. Unlike Q, the K path operates entirely at standard `32x32` tile granularity, with one exception (the intermediate matmul output, see below). The op is a single fused kernel running on a 9x2 grid that subsumes four roles:

- **`is_dkv_matmul_core`** (9x2 = 18 cores): the `[1, 7168] @ [7168, 576]` DKV projection.
- **`is_kv_rmsnorm_core`** (single receiver core): RMSNorm over the 512-element NOPE half of the projection.
- **`is_knope_core`** (matmul cores minus krope): sender cores that gather their matmul output rows to the rmsnorm core.
- **`is_krope_core`** (2 cores): RoPE rotation of the 64-element ROPE half.

The role assignment is declared on a single `UnifiedKernelDescriptor` via `UnifiedCompileTimeCoreDescriptor` entries (lines 458–483 of `fused_ops/kv_cache_branch/op.py`). One kernel, four behaviors, selected at compile time by per-core boolean compile-time args.

### Tile geometry on the K path

The op uses four distinct tile shapes, each chosen to match the natural data width of its stage:

| CB name | tile geometry | role |
|---|---|---|
| `dkv_matmul_input_cb` | `1x32` | activation row (1 row, 7168 cols / 32 = 224 tiles) |
| `dkv_matmul_output_cb` | `1x32` | per-core matmul output |
| `dkv_matmul_weights_cb` | `32x32` | sharded weights (standard tile) |
| `kv_rmsnorm_input_cb` | `16x32` | gathered NOPE half (512 elements = 1 tile) |
| `kv_rmsnorm_gamma_cb` | `16x32` | normalization weights |
| `kv_rmsnorm_output_cb` | `16x32` | normalized NOPE half |
| `k_rope_output_cb` | `1x32` | rotated ROPE half (post-RoPE) |
| `kv_cache_tensor` (DRAM) | `32x32` | the cache itself |

```python
# fused_ops/kv_cache_branch/op.py:153-154
TILE_1x32 = ttnn.Tile((1, 32))
dkv_matmul_input_page_size = TILE_1x32.get_tile_size(input_tensor.dtype)
```

```python
# fused_ops/kv_cache_branch/op.py:320-322
TILE_16x32 = ttnn.Tile((16, 32))
kv_rmsnorm_tile_descriptor = ttnn.TileDescriptor(TILE_16x32)
kv_rmsnorm_page_size = TILE_16x32.get_tile_size(input_tensor.dtype)
```

The `1x32` and `16x32` tiles used internally inside `kv_cache_branch` are tiny-tile workspaces — but the **final** output, written to `kv_cache_tensor`, lands on `32x32` tile boundaries. The reason is concrete: a single Q decode step appends one row of `[1, 576]` to the cache (1 NOPE half of 512 + 1 ROPE half of 64). Across 32 decode steps, those rows fill the rows of a `32x32` tile; the cache is read by Flash MLA in chunks of `Sk_chunk_t * DHt` standard tiles. Storing the cache in `8x32` would waste 75 % of L1 read bandwidth, because the K side has no head-count asymmetry to exploit — every column of K is meaningful, and the SDPA inner loop sees 32 rows of K per `Sk_chunk_t` step.

So the tiny-tile boundary sits **inside the producer**, not at its output: the matmul produces `1x32` rows, the rmsnorm operates on `16x32` halves, but everything is repackaged back to `32x32` (in DRAM) before Flash MLA reads it.

### KV cache write

The cache write path is intentionally minimal. The op receives the kv cache buffer address and a metadata tensor (containing `position_id` and `slot_id`) as common runtime args, and the kernel computes the write tile id at runtime from those:

```python
# fused_ops/kv_cache_branch/op.py:128-132
kv_cache_buffer_addr = kv_cache_tensor.buffer_address()
metadata_tensor_addr = metadata_tensor.buffer_address()
kv_cache_tile = kv_cache_tensor.get_tile()
# Calculate starting tile ID based on write index
# KV cache shape is [1, 1, seq_len, kv_dim], tiles are [32, 32]
```

The "tiles are [32, 32]" comment is the load-bearing fact for downstream Flash MLA: the K input that Flash MLA reads, page-by-page, is `32x32` BFP8 (per the chapter overview's CB budget: 156,672 B of K is 144 standard tiles × 1088 B each, or ~75 % of the per-core CB budget on the SDPA workers).

## 3. The join: `attention_block` composes both halves

`fused_ops/attention_block/op.py` is the program-level composition point. It is a single `UnifiedKernelDescriptor` that emits the entire attention layer — pre-SDPA, Flash MLA, post-SDPA, CCL — into one program. Critically, it creates the CB IDs that *both* `create_q_heads` and `kv_cache_branch` write to, and that Flash MLA reads from, via a `CircularBufferIdManager`:

```python
# fused_ops/attention_block/op.py:887-933 (abridged)
TD_INTERP = ttnn.TileDescriptor(interpreted_tile)        # 32x32
TD_1x32   = ttnn.TileDescriptor(TILE_1x32)
TD_16x32  = ttnn.TileDescriptor(HALF_16x32_TILE)
TD_32x32  = ttnn.TileDescriptor(FULL_32x32_TILE)
TD_8x32   = ttnn.TileDescriptor(ttnn.Tile((8, 32)))
TD_SDPA   = ttnn.TileDescriptor(sdpa_tile)               # 8x32 again
TD_KV     = ttnn.TileDescriptor(kv_cache_tensor.get_tile())  # 32x32

# ...

create_q_heads_out_cb = cb_id_context.get_cb_id(
    data_format, TD_8x32
)  # Output CB for CreateQHeads (linked to output tensor on receiver cores)
```

The `cb_id_context` returns a stable, unique CB index per `(dtype, tile_descriptor)` pair. The `create_q_heads_out_cb` ID is the same ID that Flash MLA's Q input uses — that is how the `8x32` geometry "propagates" from the create_q_heads output to the SDPA compute kernel: not by passing tile shape through a function argument, but by sharing a CB descriptor whose tile field was set at descriptor-creation time and is read by every kernel that binds that CB index.

The K side works the same way, except `TD_KV` reflects the `32x32` of the DRAM cache. Both sides are visible to the SDPA worker compute kernel because the kernel reads tile shape off the tensor at op-emit time:

```python
# micro_ops/flash_mla/op.py:412-415
q_tile = input_tensor_q.get_tile()                  # 8x32
k_tile = input_tensor_k.get_tile()                  # 32x32
Q_TILE_HEIGHT = q_tile.tile_shape[0]                # 8
K_TILE_HEIGHT = k_tile.tile_shape[0]                # 32
```

Flash MLA's compute kernel then uses `Q_TILE_HEIGHT` and `K_TILE_HEIGHT` as compile-time args. The mixed-geometry matmul (`Q × K^T` with `Q` being `[PNHt, DHt]` of `8x32` tiles and `K` being `[Sk_chunk_t, DHt]` of `32x32` tiles, producing `[PNHt, Sk_chunk_t]` scores in DST) is the consequence of that asymmetry, discussed in detail in the Flash MLA section.

## 4. The data flow

```text
QNOPE matmul3 (8x8 grid) --+
                           |   create_q_heads     +--> [Q: 8x32 tiles]  --+
QRoPE + RoPE (8x4 grid) ---+   (12x8 -> 4x2,      |    18 tiles/head     |
                               3-phase tilize)    |    8 heads/receiver  |   +-- Flash MLA
                                                  |                      |   |    decode
input row [1, 7168]      --> kv_cache_branch  ----> [KV cache: 32x32]    +-->|
                              (9x2 grid:           |    DRAM, BFP8       |   |
                               DKV matmul +        |    one row appended +-->|
                               RMSNorm + KRoPE)    |    per decode step      |
                                                                             v
                                                                          scores in
                                                                          DST regs:
                                                                          [PNHt, Sk_chunk_t]
```

The Q side stays in L1 throughout (sharded across receiver cores), with tile geometry `8x32` preserved end-to-end from `create_q_heads`'s tilize through the SDPA Q reader. The K side round-trips through DRAM via the cache, with tile geometry `32x32` preserved end-to-end from `kv_cache_branch`'s final pack through the SDPA K reader. The two streams converge in the SDPA compute kernel's `sdpa_custom_mm_block(Q, K_chunk, transpose_k=true)` call, which compiles to a mixed-geometry matmul.

## 5. Test surfaces

`tests/unit_tests/test_kv_cache_branch.py` (486 lines) parametrizes over `position_id ∈ {0, 1, 5, 7}` and validates that the fused KV branch produces the right output for the rmsnorm-tested half and the rope-tested half, and that the correct row gets appended at the right tile offset in the cache. The test imports `FlashMLADecode` from `micro_ops/flash_mla/op.py` (line 17) — the unit test references it for golden tile-id math, even though it does not run SDPA. The `test_attention_block.py` test exercises the full join: it creates Q via `create_q_heads`, KV via `kv_cache_branch`, then runs Flash MLA on the combined CBs, and PCC-checks the SDPA output against the PyTorch golden in `AttentionBlock.golden` (lines 64–190 of `attention_block/op.py`).

## 6. Why this split matters

The tile-geometry boundary sits at the Q producer, not at the K producer, because **only Q has the head-count asymmetry that lets tiny tiles save L1**. A Q "row" is the activation for one decode token, replicated across heads — with `PNHt = num_q_heads_per_core / Q_TILE_HEIGHT = 8 / 8 = 1`, the kernel sees one `8x32` Q tile per head-dim column, exploiting the fact that there are only 8 Q heads per SDPA worker. K does not have this structure: each `32x32` K tile carries 32 distinct token positions' worth of cache, all needed simultaneously by the attention math. Forcing K to `8x32` would multiply the number of CB pages by 4 without reducing data volume.

This is why the chapter overview talks about the Flash MLA L1 budget as "75 % K, ~10 % Q, ~15 % stats and intermediates": the savings from Q's tiny tiles (9,216 B for `cb_q_in` vs. ~36,864 B if padded to `32x32`) are real but small in absolute terms. The architectural payoff is not bytes saved on the Q CB — it is **enabling the SDPA worker to fit at all** when both Q and K need to coexist with intermediate stats, mask, and output buffers in the same ~217 KB L1 budget. The Q side gets its tile geometry from `create_q_heads`; the K side gets its tile geometry from `kv_cache_branch`'s final pack into the DRAM cache; and `attention_block` is what binds the two into a single coherent program.

## References

- `models/demos/deepseek_v3_b1/micro_ops/create_q_heads/op.py` — 4x2 receiver tilize op, lines 11–49 (overview), 100–128 (phase-based tilization golden), 240–242 (semaphore aliasing), 412–421 (CB binding).
- `models/demos/deepseek_v3_b1/fused_ops/kv_cache_branch/op.py` — unified KV branch op, lines 138–148 (CB roles), 153–154 / 320–322 (tile shape declarations), 458–483 (per-core role assignment).
- `models/demos/deepseek_v3_b1/fused_ops/attention_block/op.py` — composition, lines 887–933 (`cb_id_context` and tile-descriptor reuse).
- `models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:412-415` — `get_tile()` extraction at the SDPA consumer.
- `models/demos/deepseek_v3_b1/tests/unit_tests/test_kv_cache_branch.py` and `test_attention_block.py` — coverage.

See Chapter 5, "Flash MLA Decode" for how Q's `8x32` and K's `32x32` are consumed inside the SDPA compute kernel, and Chapter 5, "RoPE" for the QRoPE producer that feeds `create_q_heads` Phase 3.
