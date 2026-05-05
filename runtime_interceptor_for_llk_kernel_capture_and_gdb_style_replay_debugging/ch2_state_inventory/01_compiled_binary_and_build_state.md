# Compiled Binary and Build State

This file inventories the first category of kernel capture state: the compiled binaries and all build-time information that produced them. A kernel invocation cannot be replayed without the exact ELF images that were loaded onto each RISC-V processor, and those images cannot be reproduced without the full set of compile-time parameters, preprocessor defines, and HLK descriptors that governed the JIT compilation.

---

## 1. The Kernel Class Hierarchy

The `Kernel` base class (`tt_metal/impl/kernels/kernel.hpp`) is the central abstraction for all device-side code. It derives from `JitBuildSettings` and carries the source, configuration, compiled binaries, runtime arguments, and core placement for a single kernel. The class hierarchy:

```
JitBuildSettings (abstract)
  |
  +-- Kernel (abstract)
        |
        +-- DataMovementKernel          (BRISC or NCRISC, Wormhole/Blackhole)
        +-- ComputeKernel               (TRISC0/1/2, Wormhole/Blackhole)
        +-- EthernetKernel              (active or idle ethernet cores)
        +-- experimental::quasar::QuasarDataMovementKernel  (Quasar DM cores)
        +-- experimental::quasar::QuasarComputeKernel       (Quasar compute processors)
```

Each concrete subclass carries a `Config` variant that determines its processor class, NOC assignment, math fidelity, and other parameters. The `Kernel::Config` type is defined as:

```cpp
using Config = std::variant<
    DataMovementConfig,
    EthernetConfig,
    ComputeConfig,
    experimental::quasar::QuasarDataMovementConfig,
    experimental::quasar::QuasarComputeConfig>;
```

### 1.1 `Kernel` Base Class Field Inventory

Every field below must be captured or its absence justified. Source: `tt_metal/impl/kernels/kernel.hpp`.

| Field | C++ Type | Size | Capture Role |
|---|---|---|---|
| `programmable_core_type_` | `HalProgrammableCoreType` | 4 B (enum) | Identifies target core type (TENSIX, ACTIVE_ETH, IDLE_ETH) |
| `processor_class_` | `HalProcessorClassType` | 4 B (enum) | DM vs. COMPUTE classification |
| `watcher_kernel_id_` | `int` | 4 B | Kernel ID assigned by Watcher; useful for correlating captures with Watcher logs |
| `kernel_src_` | `KernelSource` | variable | The source code or file path (see Section 2) |
| `kernel_full_name_` | `std::string` | variable | Canonical name including hash suffix; used as build cache key |
| `core_range_set_` | `CoreRangeSet` | variable | Set of core ranges this kernel is deployed to |
| `compile_time_args_` | `std::vector<uint32_t>` | $4n$ B | Positional compile-time arguments baked into binary via `hlk_defines_generated.h` |
| `named_compile_time_args_` | `std::unordered_map<std::string, uint32_t>` | variable | Named compile-time arguments; also baked into binary |
| `core_to_runtime_args_` | `std::vector<std::vector<std::vector<uint32_t>>>` | variable | 3D array: `[core_y][core_x][args]`; up to `max_runtime_args = 341` uint32 values per core |
| `common_runtime_args_` | `std::vector<uint32_t>` | $4m$ B | Runtime args shared across all cores |
| `defines_` | `std::map<std::string, std::string>` | variable | Preprocessor defines passed to compiler |
| `binaries_` | `std::unordered_map<uint64_t, std::vector<const ll_api::memory*>>` | variable | Build key to compiled ELF binary mapping |

---

## 2. Kernel Source: `KernelSource`

The `KernelSource` struct (`tt_metal/impl/kernels/kernel.hpp`) records where the kernel code comes from:

```cpp
struct KernelSource {
    enum SourceType { FILE_PATH, SOURCE_CODE };
    std::string source_;       // file path or inline source code
    SourceType source_type_;
    std::filesystem::path path_;  // resolved path (if FILE_PATH)
};
```

**Capture requirements:**

- For `FILE_PATH` kernels: the interceptor should capture the actual source content via `get_content()` rather than the path, since the path is machine-specific. The file may change between capture and replay; the snapshot must be self-contained.
- For `SOURCE_CODE` kernels: `source_` already contains the content.
- The `name()` method extracts the kernel name from the file path (e.g., `"matmul"` from `"kernels/compute/matmul.cpp"`). This name is used in the build key and must be preserved.

**Size estimate:** Kernel source files are typically 200--2000 bytes for LLK compute kernels, up to 5 KB for complex data movement kernels.

---

## 3. Config Variants: Field Inventory

### 3.1 `DataMovementConfig`

Source: `tt_metal/api/tt-metalium/kernel_types.hpp`.

| Field | C++ Type | Default | Size | Capture Role |
|---|---|---|---|---|
| `processor` | `DataMovementProcessor` | `RISCV_0` (BRISC) | 1 B | Selects BRISC (`RISCV_0`) or NCRISC (`RISCV_1`) |
| `noc` | `NOC` | `RISCV_0_default` | 1 B | NOC assignment (NOC_0 or NOC_1) |
| `noc_mode` | `NOC_MODE` | `DM_DEDICATED_NOC` | 1 B | Dedicated vs. dynamic NOC assignment |
| `compile_args` | `std::vector<uint32_t>` | empty | $4n$ B | Positional compile-time arguments |
| `defines` | `std::map<std::string, std::string>` | empty | variable | Preprocessor defines |
| `named_compile_args` | `std::unordered_map<std::string, uint32_t>` | empty | variable | Named compile-time arguments |
| `opt_level` | `KernelBuildOptLevel` | `O2` | 1 B | Compiler optimization level |

### 3.2 `ComputeConfig`

Source: `tt_metal/api/tt-metalium/kernel_types.hpp`.

| Field | C++ Type | Default | Size | Capture Role |
|---|---|---|---|---|
| `math_fidelity` | `MathFidelity` | `HiFi4` | 1 B | Controls FPU precision (LoFi, HiFi2, HiFi3, HiFi4) |
| `fp32_dest_acc_en` | `bool` | `false` | 1 B | Enable 32-bit accumulation in DEST register |
| `dst_full_sync_en` | `bool` | `false` | 1 B | Full synchronization on DEST register access |
| `unpack_to_dest_mode` | `std::vector<UnpackToDestMode>` | empty | variable | Per-operand unpack-to-dest configuration |
| `bfp8_pack_precise` | `bool` | `false` | 1 B | Higher-precision BFP8 packing |
| `math_approx_mode` | `bool` | `false` | 1 B | Enable approximation mode for transcendental functions |
| `compile_args` | `std::vector<uint32_t>` | empty | $4n$ B | Positional compile-time arguments |
| `defines` | `std::map<std::string, std::string>` | empty | variable | Preprocessor defines |
| `named_compile_args` | `std::unordered_map<std::string, uint32_t>` | empty | variable | Named compile-time arguments |
| `opt_level` | `KernelBuildOptLevel` | `O3` | 1 B | Compiler optimization level |

> **Critical for replay**: `math_fidelity`, `fp32_dest_acc_en`, and `math_approx_mode` directly affect numerical results. Capturing these is mandatory for bit-exact replay on hardware; emulator replay may not model all fidelity levels.

### 3.3 `EthernetConfig`

Source: `tt_metal/impl/kernels/kernel.hpp`.

| Field | C++ Type | Default | Size | Capture Role |
|---|---|---|---|---|
| `eth_mode` | `Eth` | `SENDER` | 1 B | SENDER, RECEIVER, or IDLE |
| `noc` | `NOC` | `NOC_0` | 1 B | NOC assignment |
| `processor` | `DataMovementProcessor` | `RISCV_0` | 1 B | Processor target |
| `compile_args` | `std::vector<uint32_t>` | empty | $4n$ B | Compile-time arguments |
| `defines` | `std::map<std::string, std::string>` | empty | variable | Preprocessor defines |
| `named_compile_args` | `std::unordered_map<std::string, uint32_t>` | empty | variable | Named compile-time arguments |
| `opt_level` | `KernelBuildOptLevel` | `Os` | 1 B | Optimization level |
| `noc_mode` | `NOC_MODE` | `DM_DEDICATED_NOC` | 1 B | NOC mode |

### 3.4 `QuasarComputeConfig`

Source: `tt_metal/api/tt-metalium/experimental/host_api.hpp`.

| Field | C++ Type | Default | Capture Notes |
|---|---|---|---|
| `num_threads_per_cluster` | `uint32_t` | `QUASAR_NUM_TENSIX_ENGINES_PER_CLUSTER` (4) | Number of Tensix engines to use per cluster |
| `math_fidelity` | `MathFidelity` | `HiFi4` | Same as `ComputeConfig` |
| `fp32_dest_acc_en` | `bool` | `false` | |
| `dst_full_sync_en` | `bool` | `false` | |
| `unpack_to_dest_mode` | `std::vector<UnpackToDestMode>` | empty | |
| `bfp8_pack_precise` | `bool` | `false` | |
| `math_approx_mode` | `bool` | `false` | |
| `compile_args` | `std::vector<uint32_t>` | empty | |
| `defines` | `std::map<std::string, std::string>` | empty | |
| `named_compile_args` | `std::unordered_map<std::string, uint32_t>` | empty | |
| `opt_level` | `KernelBuildOptLevel` | `O3` | |

### 3.5 `QuasarDataMovementConfig`

Source: `tt_metal/api/tt-metalium/experimental/host_api.hpp`.

| Field | C++ Type | Default | Capture Notes |
|---|---|---|---|
| `num_threads_per_cluster` | `uint32_t` | `QUASAR_NUM_DM_CORES_PER_CLUSTER` (8) | Number of DM cores to use per cluster |
| `compile_args` | `std::vector<uint32_t>` | empty | |
| `defines` | `std::map<std::string, std::string>` | empty | |
| `named_compile_args` | `std::unordered_map<std::string, uint32_t>` | empty | |
| `opt_level` | `KernelBuildOptLevel` | `O2` | |

> Every field in every config variant must be captured. The `math_fidelity`, `fp32_dest_acc_en`, and `unpack_to_dest_mode` fields directly affect the generated binary via preprocessor defines injected into the HLK compilation. Changing any one of them produces a different ELF.

---

## 4. Preprocessor Defines

Preprocessor defines are a critical but often overlooked part of build state. They are injected at multiple levels and collectively determine the exact binary produced.

| Define Category | Examples | Source |
|---|---|---|
| User-provided | Arbitrary key-value pairs from `Kernel::defines_` | `Config::defines` |
| Architecture | `ARCH_BLACKHOLE`, `ARCH_WORMHOLE_B0`, `ARCH_QUASAR` | JIT build environment |
| Math fidelity | `MATH_FIDELITY=n` (0=LoFi, 2=HiFi2, 3=HiFi3, 4=HiFi4) | `ComputeConfig::math_fidelity` |
| Dest accumulation | `is_fp32_dest_acc_en` | `ComputeConfig::fp32_dest_acc_en` |
| Data formats | Per-CB data format defines in `hlk_defines_generated.h` | `JitBuildOptions::hlk_desc` |
| Tile dimensions | Per-CB tile dimensions | `tt_hlk_desc` vectors |
| Device kernel defines | NOC address widths, L1 size constants | `JitBuildEnv::init()` via `device_kernel_defines` |

**Capture requirements:** All defines from all sources must be captured. The `Kernel::defines()` method returns the user-provided defines, but architecture and config-derived defines must be reconstructed from the config fields and architecture identifier. A complete snapshot should store the *effective* define set -- the union of all defines that were passed to the compiler.

**Size estimate:** Typically 50--200 key-value pairs, ~2--5 KB as serialized strings.

---

## 5. Compile-Time Arguments

Compile-time arguments are integer values baked into the kernel binary through `hlk_defines_generated.h`. They are stored in two forms on the `Kernel` object:

**Indexed compile-time args** (`Kernel::compile_time_args_`): a `std::vector<uint32_t>`, accessed by index in the kernel code via `get_compile_time_arg_val(0)`. These are emitted as `#define KERNEL_COMPILE_TIME_ARG_0 <value>` in the generated header. The subclass-specific virtual methods `process_compile_time_args()` and `process_named_compile_time_args()` may transform these before compilation.

**Named compile-time args** (`Kernel::named_compile_time_args_`): an `std::unordered_map<std::string, uint32_t>`, accessed by name via `get_compile_time_arg_val("block_w")`. These are emitted as `#define <name> <value>`.

**Capture requirements:** Both must be captured exactly. The indexed args determine the binary via their values at compile time; the named args also become preprocessor defines.

**Size estimate:** Typically 5--30 arguments, < 1 KB total.

---

## 6. The HLK Descriptor: `tt_hlk_desc`

The `tt_hlk_desc` class (`tt_metal/jit_build/hlk_desc.hpp`) is one of the most important pieces of build state. It describes the data formats and tile dimensions for every circular buffer, which directly determines the unpacker, math, and packer configurations compiled into the kernel:

| Field | Type | Size (per CB) | Description |
|---|---|---|---|
| `buf_dataformat_arr` | `std::vector<DataFormat>` | 1 byte | Data format for each CB |
| `buf_num_faces_arr` | `std::vector<uint32_t>` | 4 bytes | Number of faces per tile |
| `buf_partial_face_arr` | `std::vector<uint32_t>` | 4 bytes | Partial face flag |
| `buf_face_r_dim_arr` | `std::vector<uint32_t>` | 4 bytes | Face row dimension |
| `buf_narrow_tile_arr` | `std::vector<uint32_t>` | 4 bytes | Narrow tile flag |
| `buf_tile_r_dim_arr` | `std::vector<uint32_t>` | 4 bytes | Tile row dimension |
| `buf_tile_c_dim_arr` | `std::vector<uint32_t>` | 4 bytes | Tile column dimension |
| `buf_tile_size_arr` | `std::vector<uint32_t>` | 4 bytes | Tile size in bytes |
| `math_fidelity` | `MathFidelity` | 1 byte | Math fidelity setting |
| `approximation_mode` | `bool` | 1 byte | Approximation mode |
| `hlk_args` | `void*` | variable | User-defined HLK args (opaque blob) |
| `hlk_args_size` | `size_t` | 8 bytes | Size of HLK args blob |

Each vector is sized to `max_cbs` (32 on Wormhole, 64 on Blackhole/Quasar).

**Size estimate for `tt_hlk_desc`:**

$$\text{hlk desc size} = \text{max cbs} \times (1 + 7 \times 4) + 18 + \text{hlk args size}$$

- Wormhole (32 CBs): $32 \times 29 + 18 \approx 946$ bytes + HLK args
- Blackhole/Quasar (64 CBs): $64 \times 29 + 18 \approx 1874$ bytes + HLK args

---

## 7. Compiled ELF Binary Artifacts

The `Kernel::binaries_` field maps a build key (FNV1a hash of all compilation parameters) to a vector of `ll_api::memory*` objects, each representing a compiled ELF binary for one RISC-V processor.

| Kernel Type | Number of Binaries | Processor Targets |
|---|---|---|
| `DataMovementKernel` | 2 | BRISC, NCRISC (one active per config, both slots in the vector) |
| `ComputeKernel` | 3 | TRISC0 (unpack), TRISC1 (math), TRISC2 (pack) |
| `EthernetKernel` | 1--2 | ERISC (sender/receiver) |
| `QuasarDataMovementKernel` | up to 8 | DM0 through DM7 |
| `QuasarComputeKernel` | up to 16 | 4 engines x 4 compute processors |

Each binary is a RISC-V ELF compiled by `riscv-tt-elf-g++`. The `Kernel` class provides `get_binary_packed_size(device, index)` for the size as written to device and `get_binary_text_size(device, index)` for the text segment size.

**Architecture-specific constraint:** On Wormhole B0, NCRISC kernels run from IRAM (16 KB limit). On Blackhole and Quasar, there is no IRAM constraint and NCRISC kernels run from L1.

**Size estimates:**

| Binary Type | Size Without `-g` | Size With `-g` |
|---|---|---|
| BRISC/NCRISC DM kernel | 8--32 KB | 20--160 KB |
| TRISC unpack/math/pack | 4--32 KB | 10--160 KB |
| Total per ComputeKernel (3 binaries) | 12--96 KB | 30--480 KB |
| Total per DM kernel (2 binaries) | 16--64 KB | 40--320 KB |
| Total per QuasarComputeKernel (up to 16) | 64--512 KB | 160--2560 KB |

### 7.1 Build Cache and Binary Identification

The JIT build system uses FNV1a hashing (via `stable_hash_hlk_desc` in `hlk_desc.hpp`) to compute a build key from the kernel source hash, all compile-time arguments, preprocessor defines, `tt_hlk_desc` contents, architecture identifier, and optimization level. The function `jit_build_once()` ensures each unique configuration is compiled exactly once.

**Capture implications:** The build hash uniquely identifies the binary. If the snapshot records the build hash, a replay system can verify that it has the correct binary without recompilation. However, the binary blob itself must be included in the snapshot for portability since the build cache is machine-local.

---

## 8. Optimization Level

The `KernelBuildOptLevel` enum (`tt_metal/api/tt-metalium/kernel_types.hpp`):

```cpp
enum class KernelBuildOptLevel : uint8_t {
    O1, O2, O3, O0, Os, Ofast, Oz,
};
```

- Default for `DataMovementConfig`: `O2`
- Default for `ComputeConfig` / `QuasarComputeConfig`: `O3`
- For debugging: `O0` (combined with `-g` flag when `TT_METAL_RISCV_DEBUG_INFO` is set)

> The optimization level directly affects binary layout, register allocation, and function inlining. A snapshot captured at `O3` cannot be debugged with source-level fidelity because the compiler may have eliminated variables, reordered statements, and inlined functions. PROPOSED: the interceptor should support a dual-binary mode that captures both the production binary (for faithful reproduction) and a debug binary (for source-level replay), or trigger recompilation at `O0` when capture is requested.

---

## 9. Summary: Binary and Build State Inventory

| State Item | Source Location | Size Estimate | Capture Priority |
|---|---|---|---|
| ELF binaries (all processors) | `Kernel::binaries_[build_key]` | 12--512 KB | **Critical** -- without binaries, no replay |
| Kernel source code | `Kernel::kernel_src_` | 0.2--5 KB | High -- needed for recompilation and source-level debug |
| Kernel config variant | `Kernel::config()` | < 1 KB | **Critical** -- defines binary identity |
| Indexed compile-time args | `Kernel::compile_time_args_` | < 1 KB | **Critical** -- baked into binary |
| Named compile-time args | `Kernel::named_compile_time_args_` | < 1 KB | **Critical** -- baked into binary |
| User-provided defines | `Kernel::defines_` | 1--5 KB | **Critical** -- baked into binary |
| Effective defines (all sources) | Reconstructed from config + arch | 2--5 KB | High -- needed for recompilation |
| HLK descriptor | `JitBuildOptions::hlk_desc` | 1--2 KB | High -- needed for recompilation |
| Optimization level | Config variant `opt_level` field | 1 byte | Medium -- affects binary but embedded in build hash |
| Build state hash | `JitBuildState::build_state_hash_` | 8 bytes | High -- binary identity fingerprint |
| Core placement | `Kernel::core_range_set_` | < 1 KB | **Critical** -- which cores run this kernel |
| **Total** | | **~15--530 KB** | |

## Capture Feasibility Summary

| State Element | Host Accessible? | Timing Sensitivity | Omission Impact |
|---|---|---|---|
| ELF binaries | Yes -- `Kernel::binaries_` | None -- stable after compilation | **Fatal** -- replay impossible without binaries |
| Config variant fields | Yes -- `Kernel::config()` | None -- set at kernel creation | **Fatal** -- wrong config produces wrong binary |
| Compile-time args | Yes -- `Kernel::compile_time_args_` | None -- set at kernel creation | **Fatal** -- baked into binary identity |
| Preprocessor defines | Yes -- `Kernel::defines_` + system | None -- set at build time | **High** -- cannot recompile without defines |
| HLK descriptor | Yes -- `JitBuildOptions::hlk_desc` | None -- set at build time | **High** -- data format handling incorrect |
| Source code | Yes -- `KernelSource::get_content()` | None | Medium -- needed for recompilation, not for ELF replay |
| Build hash | Yes -- `JitBuildState::build_state_hash_` | None | Medium -- binary verification only |

> **Key source files:** `tt_metal/impl/kernels/kernel.hpp`, `tt_metal/api/tt-metalium/kernel_types.hpp`, `tt_metal/api/tt-metalium/experimental/host_api.hpp`, `tt_metal/jit_build/build.hpp`, `tt_metal/jit_build/jit_build_options.hpp`, `tt_metal/jit_build/hlk_desc.hpp`, `tt_metal/jit_build/jit_build_settings.hpp`

---

## Key Takeaways

- **The ELF binaries are the single most important piece of capture state.** Without them, replay requires recompilation from source, which demands the exact same build environment and is fragile. A self-contained snapshot must embed the binary blobs directly.

- **Five config variants exist** (`DataMovementConfig`, `ComputeConfig`, `EthernetConfig`, `QuasarDataMovementConfig`, `QuasarComputeConfig`), and every field in the active variant must be captured because each affects the generated binary.

- **The `tt_hlk_desc` descriptor encodes per-CB data formats and tile dimensions** that drive the unpack/math/pack code generation pipeline. Missing or incorrect `hlk_desc` fields will produce a binary that processes tile data incorrectly.

- **Quasar scales the binary count significantly**: a `QuasarComputeKernel` produces up to 16 binaries (4 engines x 4 processors at `QUASAR_NUM_COMPUTE_PROCESSORS_PER_TENSIX_ENGINE = 4`), compared to 3 for a standard `ComputeKernel`, multiplying the binary portion of the capture envelope by approximately 5x.

---

**Next:** [`02_runtime_configuration_state.md`](./02_runtime_configuration_state.md)
