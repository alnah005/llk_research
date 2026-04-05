# Unpack Operations

## What Unpacking Does

Unpacking is the first stage of the compute pipeline. It performs a DMA transfer of tile data from **L1 memory** into the **Source A** and **Source B** register files, applying format conversion along the way. The unpack stage is controlled by **TRISC0** (the first of three RISC-V processors dedicated to compute).

The hardware contains two independent unpackers:

| Unpacker | Target Register | Notes |
|:---------|:---------------|:------|
| Unpacker 0 | Source A (also Destination register in `unpack_to_dest` mode) | Used for the primary operand in unary ops, or the left operand in binary ops |
| Unpacker 1 | Source B | Used for the second operand in binary operations |

Both unpackers read tile data from L1, decompress block floating point formats if needed, and convert data to the internal register format. This format conversion is handled by hardware gaskets (covered in [Chapter 3 -- Format Conversion](../ch3_data_organization/format_conversion.md)).

## LLK Unpack API Headers

The unpack API is split across several header files in `tt_llk_wormhole_b0/llk_lib/`, each targeting a specific unpack pattern:

### `llk_unpack_A.h` -- Single Tile to Source A

Used for **unary operations** where only one input tile is needed. Unpacker 0 loads a tile from L1 into Source A. In broadcast modes (`BroadcastType::COL`, `ROW`, `SCALAR`), Unpacker 1 is used instead to load the tile into Source B, while Source A receives a zero-source or the broadcast value.

Key function signatures:

```cpp
template <BroadcastType BType, bool acc_to_dest,
          EltwiseBinaryReuseDestType binary_reuse_dest, bool unpack_to_dest>
inline void _llk_unpack_A_init_(...);

template <BroadcastType BType, bool acc_to_dest,
          EltwiseBinaryReuseDestType binary_reuse_dest, bool unpack_to_dest>
inline void _llk_unpack_A_(const std::uint32_t address, ...);

template <BroadcastType BType>
inline void _llk_unpack_A_uninit_(const std::uint32_t face_r_dim);
```

### `llk_unpack_AB.h` -- Two Tiles to Source A and Source B

Used for **binary operations** (element-wise add, sub, mul) where two input tiles are needed. Unpacker 0 loads tile A into Source A and Unpacker 1 loads tile B into Source B.

```cpp
template <BroadcastType BType>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape,
                                  const std::uint32_t transpose = 0);

template <BroadcastType BType>
inline void _llk_unpack_AB_(const std::uint32_t address_a,
                             const std::uint32_t address_b);

inline void _llk_unpack_AB_uninit_(const ckernel::TensorShape unpA_tensor_shape,
                                    const ckernel::TensorShape unpB_tensor_shape);
```

The MOP configuration (`_llk_unpack_AB_mop_config_`) handles all four broadcast types, programming different outer/inner loop counts and different UNPACR instruction sequences depending on whether Source B is broadcast across columns, rows, or as a scalar.

### `llk_unpack_AB_matmul.h` -- Optimized Matmul Unpacking

Used for **matrix multiplication** where register reuse is critical for performance. The matmul unpacker implements a reuse strategy: when `ct_dim >= rt_dim`, Source B (which holds `in0`/`inA`) is reused while Source A (holding `in1`/`inB`) is cycled through via the MOP replay buffer; otherwise Source A (which holds `in1`/`inB`) is reused while Source B (holding `in0`/`inA`) is cycled. The `reuse_a` variable in the source code refers to reusing logical operand `inA`, which resides in the **SrcB** register (not the SrcA register), due to the swapped operand mapping.

Note the operand mapping for matmul is swapped relative to the naming:
- `in0` / `inA` is loaded to **Source B**
- `in1` / `inB` is loaded to **Source A**

```cpp
template <std::uint32_t kernel_broadcast_a, std::uint32_t kernel_broadcast_b>
inline void _llk_unpack_AB_matmul_init_(...);

template <std::uint32_t kernel_broadcast_a, std::uint32_t kernel_broadcast_b>
inline void _llk_unpack_AB_matmul_(
    const std::uint32_t base_address_a, const std::uint32_t base_address_b,
    const std::uint32_t tile_index_a,   const std::uint32_t tile_index_b,
    const std::uint32_t tile_size_a,    const std::uint32_t tile_size_b, ...);
```

The MOP uses the replay buffer (`lltt::record` / `lltt::replay_insn`) to program context-dependent unpack sequences. The `ckernel_unpack_template` is used here instead of `ckernel_template`, enabling conditional instruction execution based on a Z-mask and context ID.

### `llk_unpack_reduce.h` -- Tile to Source A + Scalar to Source B

Used for **reduction operations** (row, column, or scalar reduction). Unpacker 0 loads the input tile to Source A, while Unpacker 1 loads a scalar reduction coefficient from a fixed L1 buffer into Source B (with Z increment = 0, so the same scalar is reused across all faces).

```cpp
template <PoolType type, ReduceDim dim>
inline void _llk_unpack_reduce_init_(...);

template <PoolType type, ReduceDim dim>
inline void _llk_unpack_reduce_(const std::uint32_t address);
```

The init function configures Source B format registers to match the reduction scalar's format and sets the within-face transpose based on the reduction dimension (`REDUCE_ROW` requires transpose).

### `llk_unpack_tilize.h` -- Row-Major to Tile Format Conversion

Converts row-major data in L1 to tile-ordered format during unpack. This is used when input data arrives in natural row-major layout and needs to be reorganized into the face-based tile layout expected by the math engine.

### `llk_unpack_untilize.h` -- Tile to Row-Major Conversion

The inverse of tilize: converts tile-ordered data to row-major layout during unpack.

## Common API Pattern

All unpack API headers follow a consistent four-function pattern:

| Function | Purpose |
|:---------|:--------|
| `_llk_unpack_hw_configure_<>()` | One-time hardware configuration (format registers, data formats). Called during kernel setup. |
| `_llk_unpack_*_init_<>()` | Configure the MOP, set address counter limits (`SETADCXX`), apply transpose settings. Called before a batch of unpack operations. |
| `_llk_unpack_*_<>()` | Execute one unpack operation: set L1 address, wait for context, post semaphore, run MOP, release context. |
| `_llk_unpack_*_uninit_<>()` | Restore address counter defaults. Called after a batch to clean up state for the next operation type. |

## Key Template Parameters

The unpack APIs use template parameters to select compile-time behavior:

| Parameter | Type | Purpose |
|:----------|:-----|:--------|
| `BroadcastType` | `enum {NONE, COL, ROW, SCALAR}` | Controls how Source B data is broadcast. `NONE` = no broadcast; `COL` = broadcast first column; `ROW` = broadcast first row; `SCALAR` = broadcast single value. |
| `acc_to_dest` | `bool` | When `true`, the Destination register contents are read back as an input (accumulated results from a previous operation). |
| `EltwiseBinaryReuseDestType` | `enum {NONE, DEST_TO_SRCA, DEST_TO_SRCB}` | Selects whether to reuse the Destination register as Source A or Source B input. |
| `unpack_to_dest` | `bool` | When `true`, Unpacker 0 writes directly to the Destination register instead of Source A, bypassing the math stage. Used for 32-bit datacopy. |

## Context Switching and Double Buffering

The unpacker uses **double-buffered configuration contexts** to overlap TRISC0 programming with hardware execution. While the unpacker hardware is processing a tile using one context, TRISC0 can program the L1 address and configuration for the next tile in the alternate context.

The context switching sequence in every `_llk_unpack_*_()` function follows this pattern:

```cpp
// 1. Wait for the next context to be free
wait_for_next_context(2);

// 2. Program L1 base address into the current context's config register
const std::uint32_t upk0_reg = (unp_cfg_context == 0)
    ? THCON_SEC0_REG3_Base_address_ADDR32
    : THCON_SEC0_REG3_Base_cntx1_address_ADDR32;
cfg[upk0_reg] = address;

// 3. Signal that config is ready
semaphore_post(semaphore::UNPACK_SYNC);

// 4. Stall unpacker until TRISC config writes complete
TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

// 5. Execute the MOP
ckernel::ckernel_template::run();

// 6. Release the context
t6_semaphore_get(semaphore::UNPACK_SYNC);

// 7. Switch to the other context for next iteration
switch_config_context(unp_cfg_context);
```

The variable `unp_cfg_context` toggles between 0 and 1, and each context has its own set of base address registers (`Base_address` vs. `Base_cntx1_address`).

## Semaphore-Based Synchronization

The unpack stage uses the `UNPACK_SYNC` semaphore to coordinate with the hardware unpacker engine. For the full semaphore protocol, see [synchronization.md -- UNPACK_SYNC](./synchronization.md#unpack_sync----unpack-context-coordination).

---

**Next:** [`math_operations.md`](./math_operations.md)
