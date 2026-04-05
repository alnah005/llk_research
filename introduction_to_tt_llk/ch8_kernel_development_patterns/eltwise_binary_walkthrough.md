# Element-wise Binary Walkthrough

This section walks through `tests/sources/eltwise_binary_test.cpp` -- the reference kernel for element-wise binary operations (add, subtract, multiply). It demonstrates how the unpack/math/pack pipeline comes together in a real kernel, how MOP programming configures the FPU, and how the Python test harness parametrizes the full space of formats, broadcast types, tile dimensions, and fidelity levels.

## File Structure

Like all LLK test kernels, `eltwise_binary_test.cpp` contains three `run_kernel()` functions, each compiled separately for a different TRISC engine via `#ifdef` guards:

```cpp
#include "ckernel.h"
#include "llk_defs.h"
#include "tensor_shape.h"

// Global state -- required by every LLK kernel
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK
// ... unpack run_kernel ...
#endif

#ifdef LLK_TRISC_MATH
// ... math run_kernel ...
#endif

#ifdef LLK_TRISC_PACK
// ... pack run_kernel ...
#endif
```

The three TRISC engines execute concurrently. The math engine waits for unpacked data via `_llk_math_wait_for_dest_available_()`, and the packer waits for math results via `_llk_packer_wait_for_math_done_()`.

## UNPACK Section

The unpack section configures the unpacker hardware and then loops over tiles, feeding pairs of source tiles (A and B) into the source registers.

### Hardware Configuration

```cpp
_llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
    formats.unpack_A_src,       // Source A L1 format
    formats.unpack_B_src,       // Source B L1 format
    formats.unpack_A_dst,       // Source A destination (SRCA register) format
    formats.unpack_B_dst,       // Source B destination (SRCB register) format
    tensor_shape.face_r_dim,    // Face row dimension for A
    tensor_shape.face_r_dim,    // Face row dimension for B
    tensor_shape.total_num_faces(),  // Total faces for A
    tensor_shape.total_num_faces(),  // Total faces for B
    params.TILE_SIZE_UNPACK_A,  // Byte size of A tile in L1
    params.TILE_SIZE_UNPACK_B); // Byte size of B tile in L1
```

This configures the unpacker's format conversion pipeline -- it needs to know both the source format in L1 and the destination format in the source registers, because it performs on-the-fly format conversion (e.g., Bfp8 in L1 to Float16 in registers).

### Initialization

```cpp
_llk_unpack_AB_init_<BROADCAST_TYPE>(tensor_shape, transpose);
```

This sets up the unpack MOP template based on the broadcast type and tile shape. The `BROADCAST_TYPE` template parameter (from `build.h`) determines how the unpacker handles Source B -- whether it loads fresh B data for every tile or reuses it across rows/columns/all tiles.

### Main Loop

```cpp
for (std::uint32_t i = 0; i < num_total_tiles; ++i)
{
    _llk_unpack_AB_<BROADCAST_TYPE>(L1_ADDRESS(params.buffer_A[i]), L1_ADDRESS(params.buffer_B[i]));
}
```

Each call unpacks one tile from Source A and one from Source B. The `L1_ADDRESS()` macro converts buffer pointers to L1 physical addresses. The unpacker fires its MOP which handles face-level iteration internally.

## MATH Section

The math section is where the actual computation happens -- the FPU performs element-wise operations on the data that the unpacker has placed in the source registers.

### Synchronization and Hardware Init

```cpp
_llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
_llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
```

`_llk_math_pack_sync_init_()` sets up the half/full synchronization protocol between math and pack engines. `_llk_math_hw_configure_()` programs the math engine's format registers.

### FPU Initialization

```cpp
_llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BROADCAST_TYPE, MATH_FIDELITY, REUSE_DEST_TYPE>(
    tensor_shape, ACC_TO_DEST);
```

This is the critical initialization call. It does three things internally:

1. **Configures address modifiers** via `eltwise_binary_configure_addrmod()` -- sets up `ADDR_MOD_0` through `ADDR_MOD_3` to control how source and destination pointers advance through faces.
2. **Programs the MOP** via `eltwise_binary_configure_mop_standard()` -- constructs a `ckernel_template` with the correct loop counts and operation instructions.
3. **Resets counters** via `math::reset_counters()`.

The MOP structure for eltwise binary depends on the operation type:

- **ELWADD / ELWSUB**: A simple loop -- outer loop iterates over faces, inner loop iterates over 8-row chunks within each face. The loop body is a single `TT_OP_ELWADD` or `TT_OP_ELWSUB` instruction with `ADDR_MOD_0`. The end op clears source registers.
- **ELWMUL with LoFi**: Same structure as add/sub.
- **ELWMUL with HiFi**: The outer loop count becomes the fidelity level (2, 3, or 4). Each outer iteration re-reads the same source data with the fidelity phase counter incremented. The last inner iteration uses `ADDR_MOD_2` (which increments the fidelity counter) and the last outer iteration uses `ADDR_MOD_3` (which clears the fidelity counter and advances to the next face).

### Main Compute Loop

```cpp
for (std::uint32_t block = 0; block < num_blocks; block++)
{
    _llk_math_wait_for_dest_available_<dest_sync>();
    for (std::uint32_t n = 0; n < num_tiles_accumulations; n++)
    {
        for (std::uint32_t tile = 0; tile < tiles_in_block; tile++)
        {
            LLK_ASSERT(
                (static_cast<std::uint32_t>(tile) < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BROADCAST_TYPE, dest_sync, is_fp32_dest_acc_en, MATH_FIDELITY, REUSE_DEST_TYPE>(
                tensor_shape, tile /* dst_index */, false /* clear_fp32_dst_acc */);
        }
    }
    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}
```

The outer `block` loop processes blocks of tiles that fit in the destination register. Within each block:

1. **`_llk_math_wait_for_dest_available_()`** -- Blocks until the packer has finished reading the previous block from the destination register, freeing it for new results.
2. The inner loops process each tile. For each tile, `_llk_math_eltwise_binary_()` sets the destination write address and calls `ckernel_template::run()` to fire the pre-programmed MOP. For COL broadcast, the execute function loops over face rows calling `run()` multiple times.
3. **`_llk_math_dest_section_done_()`** -- Signals that this block of results is ready for the packer.

The `LLK_ASSERT` before each tile operation validates that the tile index does not exceed the destination register capacity -- a common source of bugs when tile counts are misconfigured.

## PACK Section

The pack section reads computed results from the destination register and writes them back to L1 memory.

### Hardware Configuration and Initialization

```cpp
_llk_pack_hw_configure_<is_fp32_dest_acc_en, false /* untilize */>(
    formats.pack_src, formats.pack_dst, tile_size,
    tensor_shape.face_r_dim, num_faces, partial_face, narrow_tile);

_llk_pack_init_<false /* untilize */, false /* zero_output */>(
    formats.pack_dst, tensor_shape.face_r_dim, num_faces, partial_face, narrow_tile);

_llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en, false /* untilize */>(
    tensor_shape.face_r_dim, narrow_tile);
```

Note the WH/BH divergence here -- Blackhole passes additional parameters (`total_col_dim()`) and omits `narrow_tile` from some calls, while Wormhole includes `narrow_tile` and `partial_face` throughout. This is controlled by `#ifdef ARCH_BLACKHOLE` / `#ifdef ARCH_WORMHOLE` guards.

### Main Pack Loop

```cpp
for (std::uint32_t block = 0; block < output_num_blocks; block++)
{
    _llk_packer_wait_for_math_done_();
    for (std::uint32_t tile = 0; tile < output_tiles_in_block; tile++)
    {
        std::uint32_t res_tile_idx = (block * output_tiles_in_block) + tile;
        LLK_ASSERT(
            (static_cast<std::uint32_t>(tile) < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
            "Block tile index exceeds maximum destination tiles");
        _llk_pack_<dest_sync, is_fp32_dest_acc_en, false /* untilize */>(
            tile, L1_ADDRESS(params.buffer_Res[res_tile_idx]));
    }
    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}
```

The structure mirrors the math loop:

1. **`_llk_packer_wait_for_math_done_()`** -- Blocks until the math engine signals that a block of results is ready.
2. Each tile is packed from its destination register slot to its L1 output address.
3. **`_llk_pack_dest_section_done_()`** -- Signals that the packer is done with this half of the destination register, freeing it for the math engine.

## Python Test Parametrization

The Python test `tests/python_tests/test_zzz_eltwise_binary.py` uses a `@parametrize` decorator to sweep the full combinatorial space. The key axes are:

### Format Parametrization

```python
dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
formats=lambda dest_acc: _get_valid_formats(dest_acc),
```

Formats are filtered by destination accumulation: when FP32 accumulation is enabled, the input must be Float32. The format space includes `Bfp4_b`, `Bfp8_b`, `Float16_b`, and `Float32`, and input/output formats can differ (`same=False`).

### Broadcast Type

```python
broadcast_type=[BroadcastType.None_, BroadcastType.Row, BroadcastType.Column, BroadcastType.Scalar],
```

Each broadcast type changes both the unpack MOP (how Source B is loaded) and the math MOP (how many times `ckernel_template::run()` is called per tile).

### Math Fidelity and Operation

```python
math_fidelity=lambda formats: _get_valid_math_fidelity(formats),
math_op=lambda math_fidelity: _get_valid_math_ops(math_fidelity),
```

Fidelity is constrained by format: Bfp4_b and Bfp8_b support only LoFi, Float16_b supports LoFi and HiFi2, Float32 supports HiFi3 and HiFi4. High fidelity is only valid for `Elwmul` because add/sub do not benefit from multi-phase computation.

### Tile Dimensions

```python
tile_dimensions=lambda transpose_srca, broadcast_type: _get_valid_tile_dimensions(transpose_srca, broadcast_type),
```

The supported tile sizes come from `SUPPORTED_TILE_SIZES` and include all eight entries: `16x16`, `1x32`, `2x32`, `4x32`, `8x32`, `16x32`, `32x32`, and `32x16`. Transpose only works with `32x32`. The `32x16` shape is not supported for COL or ROW broadcast.

### How `build.h` Connects Python to C++

Each test configuration generates a `build.h` file containing `constexpr` declarations. The Python `TemplateParameter` subclasses each emit one line:

| Python Class | Generated C++ |
|-------------|---------------|
| `BROADCAST_TYPE(BroadcastType.Row)` | `constexpr auto BROADCAST_TYPE = ckernel::BroadcastType::ROW;` |
| `MATH_OP(MathOperation.Elwmul)` | `constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::ELWMUL;` |
| `MATH_FIDELITY(MathFidelity.HiFi2)` | Generated as a constexpr MathFidelity value |
| `DEST_SYNC(DestSync.Half)` | `constexpr auto dest_sync = DstSync::SyncHalf;` |

The C++ kernel code uses these identifiers directly as template arguments, so the same source file compiles into different hardware configurations depending on which `build.h` is generated.

---

**Next:** [`matmul_walkthrough.md`](./matmul_walkthrough.md)
