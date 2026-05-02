## 4.4 Minimum Call Summary

This section consolidates the API call sequences from Sections 4.1--4.3 into concise reference tables, distinguishing between one-time initialization and per-iteration calls.

### 4.4.1 Complete Call Count by Thread

#### TRISC0 -- Unpack Thread

| # | Function | Category | Call Count | Template Parameters |
|---|---|---|---|---|
| 1 | `_llk_unpack_hw_configure_<is_fp32_dest_acc_en>(...)` | Init (once) | 1 | `is_fp32_dest_acc_en` |
| 2 | `_llk_unpack_AB_matmul_init_<>(...)` | Init (once) | 1 | `kernel_broadcast_a=0, kernel_broadcast_b=0` |
| 3 | `_llk_unpack_AB_matmul_<>(...)` | Per K-step | $KT\_DIM$ | `kernel_broadcast_a=0, kernel_broadcast_b=0` |

**Total calls:** $2 + KT\_DIM$

Each call to `_llk_unpack_AB_matmul_<>()` internally loops $t\_dim = \min(CT\_DIM, RT\_DIM)$ times, performing one double-buffered context-switch cycle per iteration.

#### TRISC1 -- Math Thread

| # | Function | Category | Call Count | Template Parameters |
|---|---|---|---|---|
| 1 | `_llk_math_matmul_init_<MATH_FIDELITY>(...)` | Init (once) | 1 | `MATH_FIDELITY, THROTTLE_LEVEL=0` |
| 2 | `_llk_math_pack_sync_init_<DstSync::SyncHalf, fp32_en>()` | Init (once) | 1 | `DstSync::SyncHalf, is_fp32_dest_acc_en` |
| 3 | `_llk_math_hw_configure_<fp32_en>(...)` | Init (once) | 1 | `is_fp32_dest_acc_en` |
| 4 | `_llk_math_wait_for_dest_available_<DstSync::SyncHalf>()` | Per block | 1 | `DstSync::SyncHalf` |
| 5 | `_llk_math_matmul_<MATH_FIDELITY>(...)` | Per K-step | $KT\_DIM$ | `MATH_FIDELITY, THROTTLE_LEVEL=0` |
| 6 | `_llk_math_dest_section_done_<DstSync::SyncHalf, fp32_en>()` | Per block | 1 | `DstSync::SyncHalf, is_fp32_dest_acc_en` |

**Total calls:** $5 + KT\_DIM$

Each call to `_llk_math_matmul_<>()` internally iterates over all $RT\_DIM \times CT\_DIM$ output tiles, running the MOP for each.

#### TRISC2 -- Pack Thread

| # | Function | Category | Call Count | Template Parameters |
|---|---|---|---|---|
| 1 | `_llk_pack_hw_configure_<fp32_en, false, false>(...)` | Init (once) | 1 | `is_fp32_dest_acc_en, untilize=false, tilize=false` |
| 2 | `_llk_pack_init_<false, false, false>(...)` | Init (once) | 1 | `untilize=false, zero_output=false, tilize=false` |
| 3 | `_llk_pack_dest_init_<DstSync::SyncHalf, fp32_en>()` | Init (once) | 1 | `DstSync::SyncHalf, is_fp32_dest_acc_en` |
| 4 | `_llk_packer_wait_for_math_done_()` | Per block | 1 | (none) |
| 5 | `_llk_pack_<DstSync::SyncHalf, fp32_en, false>(...)` | Per tile | $TILE\_CNT$ | `DstSync::SyncHalf, is_fp32_dest_acc_en, untilize=false` |
| 6 | `_llk_pack_dest_section_done_<DstSync::SyncHalf, fp32_en>()` | Per block | 1 | `DstSync::SyncHalf, is_fp32_dest_acc_en` |

**Total calls:** $5 + TILE\_CNT$

Where $TILE\_CNT = CT\_DIM \times RT\_DIM$.

### 4.4.2 Grand Total: Minimum API Calls for One MatMul Block

For a single output block of dimensions $RT\_DIM \times CT\_DIM$ with reduction dimension $KT\_DIM$:

| Thread | Init Calls | Per-Block Calls | Total |
|---|---|---|---|
| TRISC0 (Unpack) | 2 | $KT\_DIM$ | $2 + KT\_DIM$ |
| TRISC1 (Math) | 3 | $2 + KT\_DIM$ | $5 + KT\_DIM$ |
| TRISC2 (Pack) | 3 | $2 + TILE\_CNT$ | $5 + CT\_DIM \times RT\_DIM$ |
| **All Threads** | **8** | $4 + 2 \cdot KT\_DIM + CT\_DIM \times RT\_DIM$ | $12 + 2 \cdot KT\_DIM + CT\_DIM \times RT\_DIM$ |

**Concrete example -- $1 \times 1 \times 1$ matmul** (single tile, single K-step):
- TRISC0: 2 + 1 = **3 calls**
- TRISC1: 3 + 2 + 1 = **6 calls**
- TRISC2: 3 + 2 + 1 = **6 calls**
- Total: **15 calls**

**Concrete example -- $2 \times 2 \times 4$ matmul** (2x2 output block, 4 K-steps, 4 output tiles):
- TRISC0: 2 + 4 = **6 calls**
- TRISC1: 3 + 2 + 4 = **9 calls**
- TRISC2: 3 + 2 + 4 = **9 calls**
- Total: **24 calls**

> The "Per block" calls appear once in the single-block case shown here. In a multi-block pipeline (processing multiple output blocks in sequence), `_llk_math_wait_for_dest_available_`, `_llk_math_dest_section_done_`, `_llk_packer_wait_for_math_done_`, and `_llk_pack_dest_section_done_` would each repeat once per output block.

---

### 4.4.3 One-Time vs. Per-Iteration Call Classification

#### Init Calls (Called Once Before Any Compute)

These calls configure hardware state that persists for the duration of the matmul operation. They must be called before entering any loop.

| Thread | Call | Purpose |
|---|---|---|
| TRISC0 | `_llk_unpack_hw_configure_<>()` | Tile descriptors, format registers, strides, GPR tile sizes |
| TRISC0 | `_llk_unpack_AB_matmul_init_<>()` | MOP programming, transpose, address counter endpoints, kt_dim GPR |
| TRISC1 | `_llk_math_matmul_init_<>()` | Address modifiers, MOP (replay buffer + ckernel template) |
| TRISC1 | `_llk_math_pack_sync_init_<>()` | MATH_PACK semaphore init, DEST section base |
| TRISC1 | `_llk_math_hw_configure_<>()` | ALU config (FP32, INT8, ZEROACC mode) |
| TRISC2 | `_llk_pack_hw_configure_<>()` | Pack format registers, strides, edge masks, DEST read control |
| TRISC2 | `_llk_pack_init_<>()` | Pack address modifiers, MOP |
| TRISC2 | `_llk_pack_dest_init_<>()` | DEST offset registers, packer address counters |

#### Per-Block Calls (Once Per Output Block)

These calls manage the double-buffered DEST synchronization. In the simplest case (one block), they are called once. For multi-block matmul, they wrap around the K-loop and tile-pack loop.

| Thread | Call | Purpose |
|---|---|---|
| TRISC1 | `_llk_math_wait_for_dest_available_<>()` | Wait for DEST half to be free |
| TRISC1 | `_llk_math_dest_section_done_<>()` | Signal packer, flip DEST section |
| TRISC2 | `_llk_packer_wait_for_math_done_()` | Wait for math to complete a section |
| TRISC2 | `_llk_pack_dest_section_done_<>()` | Clear DEST, release to math, flip offset |

#### Per-Iteration Calls (Looped)

| Thread | Call | Loop Variable | Iterations |
|---|---|---|---|
| TRISC0 | `_llk_unpack_AB_matmul_<>()` | K-step | $KT\_DIM$ |
| TRISC1 | `_llk_math_matmul_<>()` | K-step | $KT\_DIM$ |
| TRISC2 | `_llk_pack_<>()` | Output tile | $CT\_DIM \times RT\_DIM$ |

---

### 4.4.4 Calling Order Constraints

The API calls must obey a strict partial order due to hardware dependencies.

#### TRISC0 (Unpack)

```
_llk_unpack_hw_configure_       MUST precede everything else (sets formats + tile descriptors)
    |
    v
_llk_unpack_AB_matmul_init_     MUST precede _llk_unpack_AB_matmul_ (programs MOP)
    |
    v
_llk_unpack_AB_matmul_          K-loop iterations (sequential; each uses double-buffered contexts)
```

#### TRISC1 (Math)

```
_llk_math_matmul_init_          \
_llk_math_pack_sync_init_        } All three must precede wait_for_dest and the K-loop
_llk_math_hw_configure_         /
    |
    v
_llk_math_wait_for_dest_available_   MUST precede _llk_math_matmul_ (acquires DEST half)
    |
    v
_llk_math_matmul_               K-loop iterations (sequential; accumulates into DEST)
    |
    v
_llk_math_dest_section_done_    MUST follow all _llk_math_matmul_ calls (signals packer)
```

The three init calls (`matmul_init`, `pack_sync_init`, `hw_configure`) are mutually independent and could theoretically be reordered, but the test kernel uses the order shown.

#### TRISC2 (Pack)

```
_llk_pack_hw_configure_         \
_llk_pack_init_                  } All three must precede wait_for_math_done
_llk_pack_dest_init_            /
    |
    v
_llk_packer_wait_for_math_done_     Blocks until TRISC1 calls dest_section_done
    |
    v
_llk_pack_                       Tile loop (sequential; one tile at a time)
    |
    v
_llk_pack_dest_section_done_     MUST follow all _llk_pack_ calls (clears DEST, releases to math)
```

#### Cross-Thread Dependencies

The inter-thread synchronization (detailed in Chapter 3) creates these ordering constraints:

1. **Unpack -> Math**: The unpacker posts `UNPACK_SYNC` semaphore and sets `SRCA_VLD`/`SRCB_VLD` data-valid signals. The math engine's `MVMUL` instructions implicitly stall until source data is valid. No explicit API call is needed -- this is hardware-mediated.

2. **Math -> Pack**: `_llk_math_dest_section_done_` posts `MATH_PACK` semaphore; `_llk_packer_wait_for_math_done_` stalls until this semaphore is non-zero.

3. **Pack -> Math**: `_llk_pack_dest_section_done_` decrements `MATH_PACK` semaphore via `_llk_packer_set_math_semaphore_`; `_llk_math_wait_for_dest_available_` stalls while the semaphore is at max.

---

### 4.4.5 Dimension Parameter Reference

The matmul computation $C[RT \times CT] = A[RT \times KT] \times B[KT \times CT]$ uses three tile dimensions:

| Symbol | Name | Description | Used By |
|---|---|---|---|
| $CT\_DIM$ | Column tile dimension | Number of output tile columns in the block | All threads |
| $RT\_DIM$ | Row tile dimension | Number of output tile rows in the block | All threads |
| $KT\_DIM$ | K tile dimension | Tiles along the reduction (inner) dimension | Unpack, Math |
| $TILE\_CNT$ | Tile count | Total output tiles = $CT\_DIM \times RT\_DIM$ | Pack |
| $FACE\_R\_DIM$ | Face row dimension | Rows per face (always 16 for standard tiles) | All threads |
| $FACE\_C\_DIM$ | Face column dimension | Columns per face (always 16) | All threads |
| $TILE\_R\_DIM$ | Tile row dimension | Rows per tile = $2 \times FACE\_R\_DIM$ = 32 | Math |
| $TILE\_C\_DIM$ | Tile column dimension | Columns per tile = $2 \times FACE\_C\_DIM$ = 32 | Math |

**Element-level dimensions** (for standard 32x32 tiles):

$$A \in \mathbb{R}^{(RT\_DIM \cdot 32) \times (KT\_DIM \cdot 32)}, \quad B \in \mathbb{R}^{(KT\_DIM \cdot 32) \times (CT\_DIM \cdot 32)}, \quad C \in \mathbb{R}^{(RT\_DIM \cdot 32) \times (CT\_DIM \cdot 32)}$$

**Reuse strategy:**

| Condition | Strategy | Held Register | Cycled Register | Inner Loop Dim |
|-----------|----------|---------------|-----------------|----------------|
| $CT\_DIM \geq RT\_DIM$ | `reuse_a` | SrcB (operand A) | SrcA (operand B) | $CT\_DIM$ |
| $CT\_DIM < RT\_DIM$ | `reuse_b` | SrcA (operand B) | SrcB (operand A) | $RT\_DIM$ |

---

### 4.4.6 DEST Register Capacity Constraint

Before entering the math loop, the canonical test asserts that the output tile block fits in one DEST half:

```cpp
LLK_ASSERT(
    (get_dest_max_matmul_tiles(0 /* DST_INDEX */, params.CT_DIM, params.RT_DIM) <
     get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
    "Block tile index exceeds maximum destination tiles for matmul");
```

`get_dest_max_matmul_tiles(0, CT, RT)` returns $CT \times RT - 1$ (the maximum tile index used). The maximum tiles in one DEST half depends on `is_fp32_dest_acc_en`:

| Mode | DEST Half Size (rows) | Tile Size (rows) | Max Tiles per Half |
|------|-----------------------|-------------------|--------------------|
| FP16/BF16 dest | `DEST_REGISTER_HALF_SIZE` | 64 | `DEST_REGISTER_HALF_SIZE / 64` |
| FP32 dest | `DEST_REGISTER_HALF_SIZE` | 64 | `DEST_REGISTER_HALF_SIZE / 64` (but each row is 32b, halving capacity) |

If $RT\_DIM \times CT\_DIM$ exceeds the DEST half capacity, the computation must be split into multiple passes with intermediate DEST flushes, which the canonical single-pass kernel does not handle.

---

### 4.4.7 Mandatory Global Variables

As documented in Chapter 3, the three global variables declared at the top of `matmul_test.cpp` are required for the LLK API to function:

```cpp
std::uint32_t unp_cfg_context          = 0;  // Tracks active unpacker config context (0 or 1)
std::uint32_t pack_sync_tile_dst_ptr   = 0;  // Packer tile tracking
std::uint32_t math_sync_tile_dst_index = 0;  // Math tile tracking
```

These are referenced by the LLK internals:
- `unp_cfg_context` is read and modified by `switch_config_context` and `_llk_unpack_configure_addresses_`.
- `pack_sync_tile_dst_ptr` is reset by `_llk_pack_dest_init_`.
- `math_sync_tile_dst_index` is reset by `_llk_math_dest_section_done_`.

Omitting any of these declarations causes link errors.

---

### 4.4.8 Data Flow Across Threads

The following diagram shows how data flows through the three threads for one output block. The vertical axis represents time; horizontal arrows represent semaphore signals.

```
TRISC0 (Unpack)          TRISC1 (Math)           TRISC2 (Pack)
================          ===============          ===============
hw_configure()           matmul_init()            pack_hw_configure()
AB_matmul_init()         pack_sync_init()         pack_init()
                         math_hw_configure()      pack_dest_init()

                         wait_for_dest()  <---+
                                              |
for j=0..KT_DIM-1:      for j=0..KT_DIM-1:  |   wait_for_math() -+
  AB_matmul(j)             math_matmul(j)     |                    |
  [fills SrcA/SrcB]        [MVMUL->DEST]      |                    |
                                               |                    |
                         dest_section_done() --+-> (MATH_PACK++)    |
                                               |                    |
                                               |   <-(MATH_PACK>0)-+
                                               |   for i=0..TILE_CNT-1:
                                               |     pack(i, L1_addr)
                                               |     [DEST -> L1]
                                               |
                                               +-- dest_section_done()
                                                   [ZEROACC + MATH_PACK--]
```

The `UNPACK_SYNC` semaphore (see Chapter 3) similarly mediates between unpack and math for source register contexts, with the double-buffered context mechanism allowing overlap between L1 reads and compute.

---

### 4.4.9 Blackhole-Specific Notes

See Section 4.3.8 for Blackhole-specific pack API differences.

---

**Next:** [Chapter 5 -- Compile-Time Configuration, Preprocessor Defines, and build.h](../ch5_compile_time_config/index.md)
