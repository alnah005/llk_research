# P150 Chip Identity and Its Mapping to Blackhole in TT-LLK

## The P150 Is a Blackhole Product

The Tenstorrent P150 accelerator card is built on the **Blackhole** chip architecture. Within the TT-LLK codebase, every architecture-specific header, instruction definition, and API implementation for the P150 lives under the directory `tt_llk_blackhole/`. Any time this guide refers to "targeting the P150," it is synonymous with targeting the Blackhole architecture in TT-LLK.

The TT-LLK repository supports three chip architectures side by side:

| Directory              | Architecture  | Product Family                |
|:-----------------------|:--------------|:------------------------------|
| `tt_llk_wormhole_b0/`  | Wormhole B0   | n150s, n300, Galaxy           |
| `tt_llk_blackhole/`    | Blackhole     | **P150**                      |
| `tt_llk_quasar/`       | Quasar        | Next-generation               |

All three directories share an identical internal layout. They are peers maintained in a single mono-repository (the legacy per-architecture repositories have been archived). For Blackhole, the subtree is:

```
tt_llk_blackhole/
    common/
        inc/              # Ckernel infrastructure headers
            ckernel.h             # Core runtime: semaphores, cfg access, sync
            ckernel_defs.h        # Enums, tile constants, data format helpers
            ckernel_ops.h         # Tensix instruction wrappers (TTI_* macros)
            ckernel_template.h    # MOP (Macro-Op) template engine
            ckernel_instr_params.h # Instruction parameter constants
            cmath_common.h        # FPU/math shared utilities
            cpack_common.h        # Packer shared utilities
            cunpack_common.h      # Unpacker shared utilities
            ckernel_globals.h     # Global state variables
            ...                   # Additional infrastructure headers
            sfpu/                 # SFPU operation headers (50+ files)
    instructions/
        assembly.yaml     # Tensix ISA instruction definitions for Blackhole
    llk_lib/              # LLK API headers (the primary public surface)
        llk_defs.h
        llk_math_matmul.h             # Matrix multiplication math kernel
        llk_math_common.h             # Shared math configuration
        llk_math_eltwise_binary.h     # Element-wise binary ops
        llk_math_eltwise_unary_datacopy.h
        llk_math_eltwise_unary_sfpu.h
        llk_math_reduce.h
        llk_unpack_AB_matmul.h        # MatMul-specific unpacker
        llk_unpack_A.h                # Unary unpacker
        llk_unpack_AB.h               # Binary unpacker
        llk_unpack_common.h           # Shared unpacker configuration
        llk_pack.h                    # Packer kernel
        llk_pack_common.h             # Shared packer configuration
        ...                           # Additional operation headers
        experimental/                 # Experimental / custom kernel variants
```

The `llk_lib/` directory is where kernel developers spend most of their time. Each file typically provides three function categories: `init` (one-time configuration), `hw_configure` (hardware register programming), and the runtime operation itself. For a matrix multiplication on the P150, the three headers that matter are:

| Thread              | Header                      |
|:--------------------|:----------------------------|
| TRISC0 (Unpack)     | `llk_unpack_AB_matmul.h`    |
| TRISC1 (Math)       | `llk_math_matmul.h`         |
| TRISC2 (Pack)       | `llk_pack.h`                |

The `common/inc/` directory provides the lower-level substrate: Tensix instruction macros, address modifier management, register definitions, and the ckernel runtime that translates C++ function calls into Tensix ISA instructions pushed to the coprocessor.

## The `ARCH_BLACKHOLE` Preprocessor Define

Architecture selection at compile time is driven by a single preprocessor macro. For Blackhole, that macro is:

```
-DARCH_BLACKHOLE
```

This define is injected into every RISC-V compilation command. The `TestConfig.setup_arch()` method in `tests/python_tests/helpers/test_config.py` sets this up:

```python
case ChipArchitecture.BLACKHOLE:
    TestConfig.ARCH_NON_COMPUTE = "-mcpu=tt-bh"
    TestConfig.ARCH_COMPUTE     = "-mcpu=tt-bh-tensix"
    TestConfig.ARCH_DEFINE      = "-DARCH_BLACKHOLE"
    TestConfig.ARCH_LLK_ROOT    = "tt_llk_blackhole"
```

The define is then embedded into the full compilation command line via `INITIAL_OPTIONS_COMPILE`:

```python
TestConfig.INITIAL_OPTIONS_COMPILE = (
    "-nostdlib -fno-use-cxa-atexit -Werror -Wall ... "
    f"-DTENSIX_FIRMWARE -DENV_LLK_INFRA -DENABLE_LLK_ASSERT {TestConfig.ARCH_DEFINE} "
    ...
)
```

Kernel code and infrastructure headers can then use `#ifdef ARCH_BLACKHOLE` to enable architecture-specific behavior. For example, `tests/helpers/src/brisc.cpp` uses the define to set the correct clock calibration constant:

```c
#ifdef ARCH_WORMHOLE
#define ARCH_CYCLE_MICRO_SECOND 1000
#endif
#ifdef ARCH_BLACKHOLE
#define ARCH_CYCLE_MICRO_SECOND 1350
#endif
```

## Architecture Detection at Runtime

Before compilation begins, the infrastructure must know which chip is present. Two detection paths exist:

1. **`tt-smi` hardware query.** The `setup_testing_env.sh` script calls `tt-smi -ls` and looks for the string `"Blackhole"` in the output. If found, it sets `CHIP_ARCH=blackhole` in the environment.

2. **`ttexalens` context query.** The Python helper `tests/python_tests/helpers/chip_architecture.py` defines a `ChipArchitecture` enum with three members -- `BLACKHOLE`, `WORMHOLE`, and `QUASAR`. The function `get_chip_architecture()` first checks the `CHIP_ARCH` environment variable. If unset, it queries the device through `ttexalens` and maps the result to the enum. The result is cached for the duration of the test session.

```python
class ChipArchitecture(Enum):
    BLACKHOLE = "blackhole"
    WORMHOLE  = "wormhole"
    QUASAR    = "quasar"
```

## Distinguishing Blackhole from Wormhole B0 and Quasar

While all three architectures share the Tensix Core model and the same LLK API surface, their implementations diverge in several important respects. These differences are visible in the codebase and must be understood to write correct Blackhole-targeted kernels.

### Compiler Target Flags

Each architecture requires a different `-mcpu` flag for the RISC-V cross-compiler:

| Architecture | Non-compute flag   | Compute flag          |
|:-------------|:-------------------|:----------------------|
| Wormhole B0  | `-mcpu=tt-wh`      | `-mcpu=tt-wh-tensix`  |
| Blackhole    | `-mcpu=tt-bh`      | `-mcpu=tt-bh-tensix`  |
| Quasar       | `-mcpu=tt-bh` (fallback) | `-mcpu=tt-bh-tensix` (fallback) |

The "non-compute" flag is used for BRISC and NCRISC (general RISC-V code), while the "compute" flag is used for TRISC0/1/2 code, enabling the Tensix ISA instruction extensions. Quasar currently falls back to Blackhole's compiler target until official compiler support is available.

### Clock Speed

Blackhole's Tensix cores run at a higher clock rate than Wormhole's:

- **Wormhole B0:** 1 GHz
- **Blackhole:** 1.35 GHz (35% faster than Wormhole)

This constant appears in `tests/helpers/src/brisc.cpp` and is used for timeout calibration in the BRISC firmware.

### The `ZEROACC` Instruction

The `ZEROACC` instruction clears the Destination register. Blackhole's version has additional parameters compared to Wormhole's:

**Wormhole B0** (3 parameters):
```c
// ZEROACC(clear_mode, AddrMode, dst)
#define TT_OP_ZEROACC(clear_mode, AddrMode, dst) \
    TT_OP(0x10, (((clear_mode) << 19) + ((AddrMode) << 15) + ((dst) << 0)))
```

**Blackhole** (5 parameters):
```c
// ZEROACC(clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where)
#define TT_OP_ZEROACC(clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where) \
    TT_OP(0x10, (((clear_mode) << 19) + ((use_32_bit_mode) << 18) + \
                  ((clear_zero_flags) << 17) + ((addr_mode) << 14) + ((where) << 0)))
```

The `use_32_bit_mode` flag tells the hardware whether the destination register is configured for 32-bit (FP32) accumulation. The `clear_zero_flags` field controls whether zero-detection flags are also cleared. These extra fields reflect Blackhole's richer accumulator architecture.

### The Pack API: `_llk_pack_hw_configure_`

The internal `_llk_pack_hw_configure_` function has a different template signature on Blackhole compared to Wormhole B0:

**Wormhole B0** (2 template parameters):
```cpp
template <bool is_fp32_dest_acc_en, bool untilize = false>
inline void _llk_pack_hw_configure_(...);
```

**Blackhole** (3 template parameters):
```cpp
template <bool is_fp32_dest_acc_en, bool untilize = false, bool tilize = false>
inline void _llk_pack_hw_configure_(...);
```

Blackhole adds a third `tilize` template parameter and also adds a `tile_c_dim` function parameter that Wormhole's version does not have. The `tilize` flag controls a Blackhole-specific packing mode where data is written to L1 in tile order with a custom row-swizzle pattern. When `tilize` is true, the packer skips unswizzling rows in the tile, which is the correct behavior when the source data has already been tilized.

### Single Packer vs. Four Packers

Blackhole uses a single packer (`NUM_PACKERS = 1`) while Wormhole B0 uses four packers (`NUM_PACKERS = 4`). The Wormhole `cpack_common.h` defines `PACK_CNT = 4` and a `PACK_SEL()` helper to select subsets of packers; Blackhole's packer configuration struct is also slightly smaller (3 words vs. 4 words, omitting the `downsample_mask` / `read_mode` / `exp_threshold` word that Wormhole's fourth word contains).

### ALU Format Inference

On Blackhole, the ALU format is **inferred** by hardware from the source register data formats. This means that functions such as `_llk_math_reconfig_data_format_srca_` do not need to program explicit `ALU_FORMAT_SPEC` registers (as noted in the code comment: "Following functions do not need to program ALU_FORMAT_SPEC_REG0_SrcA/ALU_FORMAT_SPEC_REG1_SrcB for blackhole since ALU format is inferred"). This simplifies the math-side configuration path compared to Wormhole.

### Destination Clock Gating Control Register

Blackhole introduces a hardware register, `RISCV_DEBUG_REG_DEST_CG_CTRL`, that controls clock gating for the destination register. During device setup in the LLK test infrastructure (`tests/helpers/include/boot.h`), this register is written to zero to disable destination clock gating:

```c
#if defined(ARCH_BLACKHOLE) && !defined(ARCH_QUASAR)
    ckernel::reg_write(RISCV_DEBUG_REG_DEST_CG_CTRL, 0);
#endif
```

The register has two fields: `DEST_CG_EN` (bits 8-31), which enables clock gating per destination register partition, and `DEST_CG_HYST` (bits 0-6), which controls the hysteresis delay before clock gating activates. Writing zero disables all clock gating, ensuring the destination register is always clocked during computation. This is a Blackhole-specific initialization step that Wormhole does not require.

### Quasar Structural Differences

Quasar diverges more significantly from both Wormhole and Blackhole. Its `KERNEL_COMPONENTS` list includes a fourth entry, `"sfpu"`, reflecting a hardware change that separates SFPU into its own thread driven by a fourth RISC-V core (`TRISC3`). Quasar also uses `BootMode.TRISC` instead of `BootMode.BRISC`, and has its own distinct set of memory-mapped addresses and linker scripts.

---

**Next:** [`02_tensix_core_and_dataflow.md`](./02_tensix_core_and_dataflow.md)
