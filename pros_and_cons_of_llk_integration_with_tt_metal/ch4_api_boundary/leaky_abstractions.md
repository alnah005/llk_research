# Leaky Abstractions

The three-layer API is intended to hide hardware details from op kernel authors. In practice, several hardware concepts leak through to the Layer 3 surface, requiring kernel writers to understand the underlying architecture.

## 1. The `UNPACK()`, `MATH()`, `PACK()` Macros

**Location:** Defined in [`hw/inc/api/compute/common_globals.h`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/hw/inc/api/compute/common_globals.h) (lines 15-37).

```cpp
#ifdef TRISC_MATH
#include "llk_math_common_api.h"
#define MATH(x) x
#define MAIN math_main()
#else
#define MATH(x)
#endif

#ifdef TRISC_PACK
#include "llk_pack_api.h"
#define PACK(x) x
#define MAIN pack_main()
#else
#define PACK(x)
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"
#define UNPACK(x) x
#define MAIN unpack_main()
#else
#define UNPACK(x)
#endif
```

**How they leak:** Every Layer 3 function uses these macros to annotate which of the three TRISC processors (Unpack, Math, Pack) should execute each call. For example, in [`api/compute/eltwise_binary.h`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/hw/inc/api/compute/eltwise_binary.h):

```cpp
ALWI void binary_op_init_common(uint32_t icb0, uint32_t icb1, uint32_t ocb, ...) {
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(icb0, icb1)));
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1)));
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb0, icb1)));
    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>()));
}
```

An op kernel author writing at Layer 3 must know:
- That Tensix has three separate TRISC processors.
- Which processor is responsible for unpacking, math, and packing.
- That the same source file is compiled three times with different `TRISC_*` defines, and the macros select which code survives each compilation.

This is a significant leak of the hardware threading model into the "user-facing" API.

## 2. `DST_SYNC_MODE` and `DST_ACCUM_MODE` as Compile-Time Globals

These two symbols appear throughout all three layers as template arguments and macro parameters. They are not passed explicitly by callers -- they are compile-time globals injected via build defines.

**Usage in Layer 2** ([`llk_api/llk_math_binary_api.h`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_binary_api.h)):

```cpp
template <...>
inline void llk_math_eltwise_binary(uint dst_index, const bool clear_fp32_dst_acc = true) {
    LLK_ASSERT((dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ...>()), "");
    _llk_math_eltwise_binary_<..., DST_SYNC_MODE, ...>(...);
}
```

**Usage in Layer 3** ([`api/compute/reg_api.h`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/hw/inc/api/compute/reg_api.h)):

```cpp
ALWI void tile_regs_commit() {
    MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
}

ALWI void tile_regs_release() {
    PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
}
```

**Why this is a problem:**
- The behavior of user-facing functions changes based on invisible compile-time state. A kernel author calling `tile_regs_commit()` cannot see from the call site that it behaves differently under `Full` vs. `Half` sync mode.
- Layer 1 functions accept these as template parameters, meaning the abstraction boundary does not hide the synchronization model -- it merely injects it automatically.
- Debugging requires knowing which build configuration was active, adding cognitive overhead.

## 3. Raw Data Format Integers Instead of Typed Enums

Layer 1 functions in LLK accept data formats as raw `std::uint32_t` values rather than typed enums.

**From [`llk_lib/llk_unpack_common.h`](https://github.com/tenstorrent/tt-llk/blob/main/tt_llk_wormhole_b0/llk_lib/llk_unpack_common.h):**

```cpp
inline void _llk_unpack_hw_configure_(
    const std::uint32_t unpA_src_format,
    const std::uint32_t unpB_src_format,
    const std::uint32_t unpA_dst_format,
    const std::uint32_t unpB_dst_format,
    ...
```

These format parameters are then compared against `DataFormat` enum values via casts:

```cpp
std::uint32_t unpack_ch1_x_stride =
    (std::uint32_t)(unpack_dst_format & 0x3) == (std::uint32_t)DataFormat::Float32   ? 4
    : (std::uint32_t)(unpack_dst_format & 0x3) == (std::uint32_t)DataFormat::Float16 ? 2
    : ...
```

**Impact:**
- No compile-time type checking prevents passing an invalid format integer.
- Callers (Layer 2) must cast enum values to `uint32_t` when forwarding, and the cast obscures the type information that was available at the call site.
- This makes the API error-prone: a transposed argument order in a `(src_format, dst_format)` pair would compile without complaint.

## 4. The `llk_io/` Directory: Blurred Ownership

**Location:** [`hw/ckernels/wormhole_b0/metal/llk_io/`](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/ckernels/wormhole_b0/metal/llk_io) (5 files, 337 lines total).

This directory contains:
- `llk_io.h` (8 lines)
- `llk_io_pack.h` (108 lines)
- `llk_io_unpack.h` (121 lines)
- `llk_operands.h` (57 lines)
- `llk_outputs.h` (43 lines)

These files handle circular buffer I/O -- reading operands from and writing results to circular buffers. They use the `llk_` prefix and live alongside the `llk_api/` directory under the same parent, but they are Metal-authored and Metal-owned.

**Why this blurs the boundary:**
- The `llk_` prefix suggests these files are part of the LLK abstraction layer, but they are not sourced from tt-llk.
- They exist per-architecture (Wormhole B0, Blackhole, Quasar) in the same structure as the LLK API wrappers, making it unclear whether they should be maintained by the LLK team or the Metal team.
- When new I/O patterns are needed (e.g., for new data movement schemes), it is ambiguous whether the code belongs in `llk_io/` or in the compute API layer above.

## Summary of Leaks

| Leak | Layer Affected | Consequence |
|------|---------------|-------------|
| `UNPACK`/`MATH`/`PACK` macros | Layer 3 (user-facing) | Kernel authors must know the TRISC threading model |
| `DST_SYNC_MODE` / `DST_ACCUM_MODE` | Layers 2 and 3 | Invisible compile-time state changes function behavior |
| Raw `uint32_t` data formats | Layer 1 | No type safety on format parameters |
| `llk_io/` naming and placement | Layer 2 boundary | Unclear whether Metal or LLK team owns I/O code |

---

**Next:** [`sfpu_boundary.md`](./sfpu_boundary.md)
