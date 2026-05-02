# Chapter 5: Compile-Time Configuration, Preprocessor Defines, and `build.h`

## 5.2 `build.h` Generation and Constexpr Configuration

Every test variant produces a unique auto-generated `build.h` header that
bridges Python test parameters and C++ kernel code. This section traces the
generation pipeline and documents the key structures it produces.

---

### 5.2.1 Generation Flow and File Location

```
TestConfig.__init__()
    +-- formats_config = data_formats(...)     # Infer all 11 format fields
    +-- generate_runtime_args_struct()          # Build RuntimeParams C struct
TestConfig.run()
    +-- generate_variant_hash()                # SHA-256 of compile-time state
    +-- build_elfs()
         +-- generate_build_header()           # Produce build.h content
         +-- write build.h to VARIANT_DIR
         +-- compile unpack.elf, math.elf, pack.elf
```

The file is written to a per-variant directory:

```
/tmp/tt-llk-build/<test_name>/<variant_hash>/
    build.h, obj/, elf/{unpack,math,pack}.elf
```

The variant directory is added to the include path with `-I{VARIANT_DIR}`, and
`params.h` includes `build.h` unconditionally, so all three TRISC threads
share the same configuration constants.

### 5.2.2 The `generate_build_header()` Method -- Phase by Phase

**Phase 1: Fixed preamble**

```python
header_content: list[str] = [
    "// AUTO-GENERATED CONFIGURATION HEADER. DO NOT EDIT MANUALLY!",
    "#pragma once",
    '#include <array>',
    '#include <type_traits>',
    '#include "operand.h"',
    '#include "llk_defs.h"',
    '#include "llk_sfpu_types.h"',
    '#include "perf.h"',       # omitted on Quasar
    '#include "tensix_types.h"',
```

**Phase 2: RUNTIME_PARAMETERS macro and fixed constexpr values**

```python
    "#define RUNTIME_PARAMETERS  [[maybe_unused]] const struct RuntimeParams&",
    f"constexpr bool l1_acc_en = {self.l1_acc.value};",
    f"constexpr bool unpack_to_dest = {str(self.unpack_to_dest).lower()};",
```

The `RUNTIME_PARAMETERS` macro defines the type signature used in every
`run_kernel()` declaration. The `[[maybe_unused]]` attribute suppresses
warnings when SPEED_OF_LIGHT empties the struct.

**Phase 3: FormatConfig struct**

```python
] + (FORMATS_CONFIG_STRUCT_COMPILETIME if self.compile_time_formats
     else FORMATS_CONFIG_STRUCT_RUNTIME)
```

Emits either the runtime variant (mutable, zero-initialized) or the
compile-time variant (`const` fields, `constexpr` constructor). See Section
5.2.4.

**Phase 4: Destination accumulation flag**

```python
header_content.append(
    f"constexpr bool is_fp32_dest_acc_en = {self.dest_acc.cpp_enum_value};"
)
```

Consumed by nearly every LLK API as a template parameter. When true, DEST
accumulates in FP32, halving available tiles.

> **Cross-reference:** Chapter 2 covers `is_fp32_dest_acc_en` and its effect on DEST capacity.

**Phase 5: SPEED_OF_LIGHT tile sizes** -- only when SOL is enabled; otherwise
these live in `RuntimeParams`:

```python
if TestConfig.SPEED_OF_LIGHT:
    header_content.extend([
        f"constexpr std::uint32_t TILE_SIZE_PACK = {self.pack_size};",
        f"constexpr std::uint32_t TILE_SIZE_UNPACK_A = {self.unpack_size_a};",
        f"constexpr std::uint32_t TILE_SIZE_UNPACK_B = {self.unpack_size_b};",
    ])
```

**Phase 6: TemplateParameter emission**

```python
for parameter in self.templates:
    header_content.append(parameter.convert_to_cpp())
```

Each subclass implements `convert_to_cpp()` returning a `constexpr` declaration:

| Python class | Generated C++ |
|---|---|
| `MATH_FIDELITY(MathFidelity.HiFi4)` | `constexpr ckernel::MathFidelity MATH_FIDELITY = ckernel::MathFidelity::HiFi4;` |
| `DEST_SYNC(DestSync.Half)` | `constexpr auto dest_sync = ckernel::DstSync::SyncHalf;` |
| `THROTTLE_LEVEL(0)` | `constexpr int THROTTLE_LEVEL = 0;` |

A representative implementation showing how the Python enum maps to the C++
qualified name:

```python
@dataclass
class MATH_FIDELITY(TemplateParameter):
    math_fidelity: MathFidelity

    def convert_to_cpp(self) -> str:
        return f"constexpr ckernel::MathFidelity MATH_FIDELITY = {self.math_fidelity.cpp_enum_value};"
```

where `MathFidelity.cpp_enum_value` returns `"ckernel::MathFidelity::{name}"` to bridge the Python enum (LoFi=0, HiFi2=2, HiFi3=3, HiFi4=4) to the C++ enum string. See Section 5.3.7 for the full C++ enum definition and phase/accuracy table.

> **Cross-reference:** Chapter 2 covers MathFidelity values and DstSync modes; Chapter 3 covers the synchronization protocol.

**Phase 7: Compile-time data formats** -- when `compile_time_formats=True`,
emits `constexpr` values for all 11 format fields plus a `constexpr
FormatConfig` object. For fused multi-iteration tests, arrays are generated:

```cpp
constexpr std::uint32_t L1_to_L1_ITERATIONS = 2;
constexpr std::array<FormatConfig, L1_to_L1_ITERATIONS> formats_array = { ... };
```

**Phase 8: RuntimeParams struct** -- either the dynamically-generated struct
(see Section 5.2.5) or an empty `struct RuntimeParams {};` for SPEED_OF_LIGHT.

### 5.2.3 Essential Constexpr Values for Matmul

| Constant | Type | Typical Value | Source |
|---|---|---|---|
| `MATH_FIDELITY` | `ckernel::MathFidelity` | `HiFi4` | `MATH_FIDELITY` TemplateParameter |
| `is_fp32_dest_acc_en` | `bool` | `false` | `DestAccumulation` enum |
| `l1_acc_en` | `bool` | `false` | `L1Accumulation` enum |
| `unpack_to_dest` | `bool` | `false` | `TestConfig` constructor arg |
| `THROTTLE_LEVEL` | `int` | `0` | `THROTTLE_LEVEL` TemplateParameter |
| `dest_sync` | `ckernel::DstSync` | `SyncHalf` | `DEST_SYNC` TemplateParameter |

Note: `dest_sync` is not always emitted in `build.h`. The matmul test
hard-codes `DstSync::SyncHalf` in the kernel source, but other tests emit it
as a TemplateParameter.

### 5.2.4 The FormatConfig Struct

The `FormatConfig` struct holds 11 `uint32_t` fields describing the entire
unpack-math-pack format pipeline: `unpack_A_src`, `unpack_B_src`,
`unpack_S_src`, `unpack_A_dst`, `unpack_B_dst`, `unpack_S_dst`, `math`,
`pack_src`, `pack_dst`, `pack_S_src`, `pack_S_dst`.

**Runtime variant** (when `-DRUNTIME_FORMATS` is defined):

```cpp
struct FormatConfig
{
    std::uint32_t unpack_A_src = 0;
    std::uint32_t unpack_B_src = 0;
    // ... 9 more fields, all mutable, default-initialized to 0
};
```

The host writes actual format enum values into L1 at the `RuntimeParams`
address, and the kernel reads them at boot via `copy_runtimes_from_L1()`.

**Compile-time variant** (when `-DRUNTIME_FORMATS` is absent):

```cpp
struct FormatConfig
{
    const std::uint32_t unpack_A_src;
    const std::uint32_t unpack_B_src;
    // ... all 11 fields as const, with constexpr constructor
    constexpr FormatConfig(uint32_t unpack_A_src_, ...) : unpack_A_src(unpack_A_src_), ... {}
};
```

Python then emits the concrete values:

```cpp
constexpr auto UNPACK_A_IN = ckernel::to_underlying(DataFormat::Float16_b);
// ... all 11 format constants
constexpr FormatConfig formats = FormatConfig(UNPACK_A_IN, UNPACK_B_IN, ...);
```

The compile-time path enables the compiler to propagate format values as
constants, eliminating branches in format-dependent code paths.

### 5.2.5 The RuntimeParams Struct and L1 Serialization

`RuntimeParams` is generated by `generate_runtime_args_struct()` with a fixed
prefix followed by conditional fields:

```python
lines = ["struct RuntimeParams {",
         "std::uint32_t TILE_SIZE_PACK;",
         "std::uint32_t TILE_SIZE_UNPACK_A;",
         "std::uint32_t TILE_SIZE_UNPACK_B;"]
self.runtime_format = "@III"
```

If runtime formats are active, a `FormatConfig` field follows (adding
`"IIIIIIIIIII"` -- 11 `uint32_t` fields -- to the pack format). For
multi-iteration fused tests, an array `FormatConfig formats[N]` is used
instead. Then stimuli fields (if any), and per-`RuntimeParameter` fields:

| Python class | C++ fields | Pack format |
|---|---|---|
| `CRK_TILE_DIMM(c, r, k)` | `uint32_t CT_DIM; uint32_t RT_DIM; uint32_t KT_DIM;` | `"III"` |
| `NUM_FACES()` | `uint32_t num_faces; uint32_t num_faces_A; uint32_t num_faces_B;` | `"III"` |
| `TILE_COUNT(n)` | `uint32_t TILE_CNT;` | `"I"` |

**L1 serialization round-trip:** The host serializes values via `struct.pack`
with the accumulated format string and writes them to L1 at a fixed address:

```python
serialised_data = struct.pack(self.runtime_format, *argument_data)
write_to_device(location, RUNTIME_ADDRESS, serialised_data)
```

The device reads them at boot in `trisc.cpp`:

```cpp
void copy_runtimes_from_L1(struct RuntimeParams* temp_args)
{
    extern const volatile struct RuntimeParams __runtime_args_start[];
    ckernel::memcpy_blocking(temp_args, __runtime_args_start, sizeof(struct RuntimeParams));
}
```

The linker symbol `__runtime_args_start` is placed at `0x20000` (non-coverage)
or `0x6E000` (coverage). The `@` prefix in the format string selects native
byte order with native alignment, matching the RISC-V struct layout.

### 5.2.6 The TemplateParameter / RuntimeParameter Design Pattern

```
TemplateParameter (ABC)                RuntimeParameter (ABC)
    |                                      |
    +-- convert_to_cpp() -> str            +-- convert_to_cpp() -> str
                                           +-- convert_to_struct_fields() -> (str, str)
```

**`TemplateParameter`** emits `constexpr` declarations baked into the binary.
**`RuntimeParameter`** emits both a `constexpr` declaration (for SOL mode) and
a struct field + `struct.pack` format char (for normal mode).

Example (`CRK_TILE_DIMM`):

```python
@dataclass
class CRK_TILE_DIMM(RuntimeParameter):
    c_dimm: c_uint32 = 0
    r_dimm: c_uint32 = 0
    k_dimm: c_uint32 = 0

    def convert_to_cpp(self) -> str:
        return "\n".join([
            f"constexpr std::uint32_t RT_DIM = {self.r_dimm};",
            f"constexpr std::uint32_t CT_DIM = {self.c_dimm};",
            f"constexpr std::uint32_t KT_DIM = {self.k_dimm};",
        ])

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "\n".join([
            "std::uint32_t CT_DIM;",
            "std::uint32_t RT_DIM;",
            "std::uint32_t KT_DIM;",
        ]), "III"
```

SPEED_OF_LIGHT mode collapses the distinction:

```python
if TestConfig.SPEED_OF_LIGHT:
    templates += runtimes    # All runtimes become constexpr
    runtimes = []
    compile_time_formats = True
```

The `RuntimeParams` struct becomes `struct RuntimeParams {};` -- no L1 writes,
no `memcpy_blocking()`, maximum compiler optimization.

### 5.2.7 Variant Hashing and Build Caching

Each unique combination of build parameters produces a SHA-256 variant hash
that determines the build directory path:

```python
def generate_variant_hash(self):
    temp_str = [
        str(value)
        for field_name, value in self.__dict__.items()
        if field_name not in NON_COMPILATION_ARGUMENTS
    ]
    self.variant_id = sha256(str(" | ".join(temp_str)).encode()).hexdigest()
```

The `NON_COMPILATION_ARGUMENTS` set contains fields that affect only runtime
behavior: stimuli data, runtime parameter values (in non-SOL mode), format
config values (in runtime mode), and similar. These are excluded because they
do not affect the compiled binary. This means two test runs that differ only in
tile dimensions or input data share the same compiled ELF -- only the L1 data
blob changes. Recompilation is only triggered when a compile-time-affecting
parameter changes (format mode, math fidelity, accumulation flags, etc.).

A `.build_complete` marker file in the variant directory prevents redundant
recompilation when the same variant is rebuilt. The build system uses
`FileLock` to handle concurrent test runs safely -- multiple test processes
that need the same variant can race to build it, with only the first one
performing the actual compilation.

### 5.2.8 Complete `build.h` Example

A typical `build.h` for Float16_b matmul, HiFi4, runtime formats:

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// AUTO-GENERATED CONFIGURATION HEADER. DO NOT EDIT MANUALLY!

#pragma once
#include <array>
#include <type_traits>
#include "operand.h"
#include "llk_defs.h"
#include "llk_sfpu_types.h"
#include "perf.h"
#include "tensix_types.h"
#define RUNTIME_PARAMETERS  [[maybe_unused]] const struct RuntimeParams&
constexpr bool l1_acc_en = false;
constexpr bool unpack_to_dest = false;
struct FormatConfig
{
    std::uint32_t unpack_A_src = 0;
    std::uint32_t unpack_B_src = 0;
    std::uint32_t unpack_S_src = 0;
    std::uint32_t unpack_A_dst = 0;
    std::uint32_t unpack_B_dst = 0;
    std::uint32_t unpack_S_dst = 0;
    std::uint32_t math = 0;
    std::uint32_t pack_src = 0;
    std::uint32_t pack_dst = 0;
    std::uint32_t pack_S_src = 0;
    std::uint32_t pack_S_dst = 0;
};
constexpr bool is_fp32_dest_acc_en = false;
constexpr ckernel::MathFidelity MATH_FIDELITY = ckernel::MathFidelity::HiFi4;
struct RuntimeParams {
    std::uint32_t TILE_SIZE_PACK;
    std::uint32_t TILE_SIZE_UNPACK_A;
    std::uint32_t TILE_SIZE_UNPACK_B;
    FormatConfig formats;
    std::uint32_t num_faces;
    std::uint32_t num_faces_A;
    std::uint32_t num_faces_B;
    std::uint32_t TILE_CNT;
    std::uint32_t CT_DIM;
    std::uint32_t RT_DIM;
    std::uint32_t KT_DIM;
};
```

### 5.2.9 The Header Include Chain

The generated `build.h` is included indirectly through `params.h`, which is the
standard include in every test source:

```
test_source.cpp (e.g., matmul_test.cpp)
  |-- #include "params.h"
        |-- #include "build.h"    (auto-generated, in variant dir)
        |-- #include "ckernel_defs.h"
        |-- #include "ckernel_sfpu.h"
        |-- #include "tensix_types.h"
```

The variant directory is added to the include path via `-I<variant_dir>` so
that `#include "build.h"` resolves correctly. This per-variant include path
is what allows the same source to be compiled with different configurations
simply by pointing to different `build.h` files.

---

**Previous:** [`01_required_preprocessor_defines.md`](./01_required_preprocessor_defines.md) | **Next:** [`03_matmul_specific_config_structs.md`](./03_matmul_specific_config_structs.md)
