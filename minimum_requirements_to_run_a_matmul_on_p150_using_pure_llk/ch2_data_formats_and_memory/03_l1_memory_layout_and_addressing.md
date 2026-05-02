# L1 Memory Layout and Addressing

## The Blackhole L1 Memory Map

Each Tensix Core on the Blackhole chip has its own L1 SRAM. The total physical L1 size is 1.5 MB, defined in `tensix_types.h`:

```cpp
static constexpr std::uint32_t L1_SIZE = 0x180000;  // 1.5 MB (1,572,864 bytes)
```

However, the firmware-validated usable limit is smaller. The `dev_mem_map.h` header defines:

```cpp
#define MEM_L1_SIZE (1464 * 1024)  // 0x16E000 (~1.43 MB, 1,499,136 bytes)
```

The `is_valid_L1_address()` function in `llk_memory_checks.h` uses `MEM_L1_SIZE` (0x16E000) as the upper boundary, not `L1_SIZE` (0x180000). Any tile address at or above 0x16E000 will fail validation.

This memory holds everything the core needs to operate: firmware code for all RISC-V processors, initialized data, runtime parameters, input tiles, output tiles, and synchronization structures. The layout is defined by the linker script `tests/helpers/ld/memory.blackhole.ld`.

### L1 Region Layout (Performance Mode)

The performance layout is the default for LLK tests. The linker script uses relative addressing -- each region begins immediately after the previous one. The resolved addresses are:

| Region | Start | Size | End | Purpose |
|:-------|:------|:-----|:----|:--------|
| `BRISC_CODE` | 0x00000 | 16 KB | 0x03FFF | BRISC firmware instructions |
| `BRISC_LOADER_INIT_MEM` | 0x04000 | 8 KB | 0x05FFF | BRISC initialized data (copied to LOCAL_MEM at boot) |
| `TRISC0_CODE` | 0x06000 | 16 KB | 0x09FFF | Unpack thread firmware instructions |
| `TRISC0_LOADER_INIT_MEM` | 0x0A000 | 4 KB | 0x0AFFF | TRISC0 initialized data |
| `TRISC1_CODE` | 0x0B000 | 16 KB | 0x0EFFF | Math thread firmware instructions |
| `TRISC1_LOADER_INIT_MEM` | 0x0F000 | 4 KB | 0x0FFFF | TRISC1 initialized data |
| `TRISC2_CODE` | 0x10000 | 16 KB | 0x13FFF | Pack thread firmware instructions |
| `TRISC2_LOADER_INIT_MEM` | 0x14000 | 4 KB | 0x14FFF | TRISC2 initialized data |
| `RUNTIME_ARGS` | 0x20000 | 1 KB | 0x203FF | Runtime arguments structure |
| *Stimuli / Tile Data* | 0x21000 | ~1.30 MB | 0x16DFFF | Input tiles, output tiles, free space (validated limit: 0x16E000) |

The firmware regions are tightly packed: BRISC gets 16 KB for code plus 8 KB for loader data, and each TRISC gets 16 KB for code plus 4 KB for loader data. After the loader data for TRISC2 ends at 0x14FFF, there is a gap before `RUNTIME_ARGS` at 0x20000 and another before tile data begins at 0x21000.

### The Linker Script Structure

The linker setup uses a two-level structure:

1. **`memory.blackhole.ld`** defines the `MEMORY` regions -- where each core's code and data physically reside in L1, and where private memory is. It uses relative addressing so regions chain automatically:

    ```
    BRISC_CODE            : ORIGIN = L1_BASE, LENGTH = 16K
    BRISC_LOADER_INIT_MEM : ORIGIN = ORIGIN(BRISC_CODE) + LENGTH(BRISC_CODE),
                            LENGTH = BRISC_LOCAL_MEM_LENGTH
    TRISC0_CODE           : ORIGIN = ORIGIN(BRISC_LOADER_INIT_MEM)
                            + LENGTH(BRISC_LOADER_INIT_MEM), LENGTH = 16K
    ...
    ```

2. **Per-thread linker scripts** (`unpack.ld`, `math.ld`, `pack.ld`, `brisc.ld`) create `REGION_ALIAS` mappings that select which MEMORY regions from step 1 each thread uses:

    ```
    // unpack.ld
    REGION_ALIAS("REGION_CODE", TRISC0_CODE)
    REGION_ALIAS("REGION_DATA", TRISC0_LOCAL_DATA_MEM)
    REGION_ALIAS("REGION_STACK", TRISC0_STACK_MEM)
    REGION_ALIAS("REGION_LOADER_INIT", TRISC0_LOADER_INIT_MEM)
    ```

3. **`sections.ld`** is the shared sections script that uses the generic `REGION_*` aliases to lay out `.text`, `.data`, `.bss`, and other sections. It is included by all threads.

This design allows the same sections script to be reused across all three TRISC threads (and BRISC), with only the memory region assignments differing per thread. Each thread is compiled to a separate ELF binary that is loaded at its designated L1 code address.

### Performance vs. Debug Memory Layouts

The LLK test infrastructure provides two linker scripts for Blackhole:

- **`memory.blackhole.ld`** (performance mode): Compact firmware regions (16 KB code per core), stimuli data starts at 0x21000, maximizing space for tile buffers.
- **`memory.blackhole.debug.ld`** (debug mode): Larger firmware regions (64 KB code + 32 KB data per TRISC), with GCOV coverage support. Stimuli data starts at 0x70000, leaving less room for tile buffers but accommodating debug instrumentation.

The `StimuliConfig` class selects the starting address based on the mode:

```python
class StimuliConfig:
    STIMULI_L1_ADDRESS_PERF  = 0x21000
    STIMULI_L1_ADDRESS_DEBUG = 0x70000
```

For a minimal matmul in performance mode, tile data begins at 0x21000 and extends up to the firmware-validated L1 boundary at 0x16E000, giving approximately 1.30 MB for tile storage. (The physical L1 extends to 0x180000, but `is_valid_L1_address()` rejects addresses at or above 0x16E000.)

## Private Data Memory (LOCAL_MEM_BASE)

In addition to L1, each RISC-V core has access to a small block of **private data memory** that is separate from L1. This memory is defined in the linker script:

```
LOCAL_MEM_BASE = 0xFFB00000;
BRISC_LOCAL_MEM_LENGTH = 8K;
TRISC_LOCAL_MEM_LENGTH = 4K;
```

Each core sees its private memory at the same virtual address `0xFFB00000`, but each core's hardware maps it to a different physical memory region. A TRISC0 write to `0xFFB00000` goes to TRISC0's private SRAM; a TRISC1 write to the same address goes to TRISC1's separate private SRAM.

| Core | Data | Stack | Total |
|:-----|:-----|:------|:------|
| BRISC | 6 KB | 2 KB | 8 KB |
| TRISC0 | 3584 bytes (3.5 KB) | 512 bytes | 4 KB |
| TRISC1 | 3584 bytes (3.5 KB) | 512 bytes | 4 KB |
| TRISC2 | 3584 bytes (3.5 KB) | 512 bytes | 4 KB |

The `LOADER_INIT_MEM` sections in L1 hold the initial values of global variables. At boot time, the startup code copies these values from L1 to `LOCAL_MEM_BASE`, initializing each core's private working memory. This separation ensures that runtime variables for each core (like `dest_offset_id`, `unp_cfg_context`, and `math_sync_tile_dst_index`) live in fast private memory rather than in shared L1. The private memory is not visible to other cores and cannot be used for tile data or inter-thread communication.

## L1 Address Transformation: The L1_ADDRESS() Function

A critical detail for programming the hardware: the LLK functions do **not** use raw byte addresses when programming the unpackers and packers. Instead, they use a **transformed address** computed by the `L1_ADDRESS` function from `llk_memory_checks.h`:

```cpp
/**
 * Hardware L1 address fields use 16-byte granularity, not byte addresses.
 * The hardware expects addresses in units of 16-byte blocks.
 *
 * Tensix L1 address fields use an off-by-one convention: you program
 * (addr_16B - 1), and the hardware internally increments the value
 * before using it.
 */
constexpr inline std::uint32_t L1_ADDRESS(std::uint32_t buffer_address)
{
    return (buffer_address >> 4) - 1;
}
```

The transformation has two components:

1. **Division by 16** (`>> 4`): The hardware's address fields represent addresses in units of 16-byte blocks, not bytes. A physical byte address of 0x21000 becomes $0x21000 \gg 4 = 0x2100$ in hardware units.

2. **Decrement by 1** (`- 1`): The Tensix L1 address convention uses an off-by-one encoding. When you program an address value $N$, the hardware internally uses $N + 1$. This means you must program `(addr / 16) - 1` so the hardware reads $((addr / 16) - 1) + 1 = addr / 16$, yielding the correct 16-byte-aligned address.

For example, to point the unpacker at L1 byte address 0x21000:

$$L1\_ADDRESS(0x21000) = (0x21000 \gg 4) - 1 = 0x2100 - 1 = 0x20FF$$

This transformed address is what gets written into the THCON base address registers (`THCON_SEC0_REG3_Base_address`, etc.).

The 16-byte alignment requirement means that all tile buffer addresses must be placed at 16-byte-aligned boundaries. In practice, the format tile sizes (2048, 4096, 1088, etc.) are all multiples of 16, so consecutively placed tiles remain aligned.

There is a related function `unpack_16B_address()` in `cunpack_common.h` that performs a slightly different transformation used for FIFO-aligned addresses:

```cpp
inline std::uint32_t unpack_16B_address(const std::uint32_t addr)
{
    return (addr << FIFO_BASE_ADDRESS_ALIGN_BITS) >> 4;
}
```

Where `FIFO_BASE_ADDRESS_ALIGN_BITS = 9` (512-byte alignment). This is used when addresses have been pre-divided by the FIFO alignment and need to be converted to 16-byte units. It is distinct from `L1_ADDRESS()`, which is the primary transformation used by kernel code when passing tile addresses to unpack/pack APIs.

## Valid L1 Region for Tile Data

Not all of L1 is available for tile data. The `llk_memory_checks.h` file defines the valid region:

```cpp
constexpr std::uint32_t L1_REGION_START = L1_ADDRESS(MEM_MAP_END);
constexpr std::uint32_t L1_REGION_END   = L1_ADDRESS(MEM_L1_BASE + MEM_L1_SIZE);
```

Where `MEM_MAP_END` is the end of all reserved system memory (firmware, mailboxes, routing tables, fabric metadata), and `MEM_L1_BASE + MEM_L1_SIZE` is the total L1 extent. Every address passed to the unpack or pack APIs is validated:

```cpp
inline static bool is_valid_L1_address(const std::uint32_t address)
{
    return (address >= ckernel::L1_REGION_START && address < ckernel::L1_REGION_END);
}
```

This validation catches common errors: accidentally passing a byte address instead of a transformed address, or pointing into reserved system memory.

## The Operand Class

The `Operand` class from `tests/helpers/include/operand.h` provides a simple abstraction for computing L1 addresses of individual tiles within a buffer:

```cpp
class Operand
{
public:
    constexpr Operand(std::uint32_t base, std::uint32_t size) : base_addr(base), tile_size(size) {}

    [[nodiscard]] constexpr std::uint32_t operator[](std::uint32_t index) const noexcept
    {
        return base_addr + index * tile_size;
    }

private:
    std::uint32_t base_addr;
    std::uint32_t tile_size;
};
```

An `Operand` is constructed with a base L1 address and a tile size (in bytes). The subscript operator returns the L1 byte address of the $n$-th tile:

$$\text{operand}[n] = \text{base\_addr} + n \times \text{tile\_size}$$

The `Operand` class does **not** perform the L1_ADDRESS transformation -- the addresses it returns are physical byte addresses. The L1_ADDRESS transformation is applied when these addresses are passed to the unpacker/packer hardware configuration functions.

For example, with Float16_b tiles (2048 bytes each) starting at 0x21000:

```cpp
constexpr Operand buffer_A(0x21000, 2048);

buffer_A[0]  // = 0x21000  (first tile)
buffer_A[1]  // = 0x21800  (second tile)
buffer_A[2]  // = 0x22000  (third tile)
```

## The StimuliConfig Class

The Python-side `StimuliConfig` class (from `tests/python_tests/helpers/stimuli_config.py`) computes the L1 addresses for all three buffers and generates the firmware-side `Operand` declarations. The address computation follows a sequential layout -- buffers are packed contiguously in L1 with no gaps:

```python
self.buf_a_addr = StimuliConfig.STIMULI_L1_ADDRESS_PERF  # 0x21000

self.buf_b_addr = self.buf_a_addr + self.tile_size_A_bytes * self.tile_count_A

self.buf_res_addr = self.buf_b_addr + self.tile_size_B_bytes * self.tile_count_B
```

For a single-tile matmul with Float16_b (2048 bytes/tile):

| Buffer | Start Address | Size | End Address |
|:-------|:-------------|:-----|:------------|
| buffer_A (in0/inA) | 0x21000 | 2048 bytes | 0x21800 |
| buffer_B (in1/inB) | 0x21800 | 2048 bytes | 0x22000 |
| buffer_Res (output) | 0x22000 | 2048 bytes | 0x22800 |

The `generate_stimuli_header_addresses` method produces the C++ declarations that are written into `build.h`:

```python
def generate_stimuli_header_addresses(self) -> list[str]:
    return [
        f"constexpr Operand buffer_A({hex(self.buf_a_addr)}, {self.tile_size_A_bytes});",
        f"constexpr Operand buffer_B({hex(self.buf_b_addr)}, {self.tile_size_B_bytes});",
        f"constexpr Operand buffer_Res({hex(self.buf_res_addr)}, {self.buf_res_tile_size});",
    ]
```

For runtime-argument-based tests (where addresses are not compiled in), StimuliConfig generates a list of address/size pairs that are written to the RUNTIME_ARGS region at 0x20000:

```python
def generate_runtime_operands_values(self) -> list:
    return [
        self.buf_a_addr,        # Operand A base address
        self.tile_size_A_bytes, # Operand A tile size
        self.buf_b_addr,        # Operand B base address
        self.tile_size_B_bytes, # Operand B tile size
        self.buf_res_addr,      # Result base address
        self.buf_res_tile_size, # Result tile size
    ]
```

## MatMul Operand Mapping: The Counterintuitive Convention

A critical detail for matmul is the mapping between logical operands and hardware source registers. The FPU computes $D = B \times A$ (Source B is the left matrix, Source A is the right matrix). This leads to a counterintuitive convention:

| Logical Name | L1 Buffer | Unpacker | Source Register | Role in $D = B \times A$ |
|:-------------|:----------|:---------|:----------------|:-------------------------|
| in0 / inA | `buffer_A` | Unpacker 1 (SEC1) | **Source B** (left matrix) | $B$ |
| in1 / inB | `buffer_B` | Unpacker 0 (SEC0) | **Source A** (right matrix) | $A$ |
| result | `buffer_Res` | Packer | Destination | $D$ |

The comments in `llk_unpack_AB_matmul.h` confirm this:

```cpp
// in0/inA - loaded to SrcB
// in1/inB - loaded to SrcA
```

In the `_llk_unpack_AB_matmul_` function, the address assignment implements this swap:

```cpp
// Validate and configure addresses
// (note: address_b goes to SEC0, address_a to SEC1 for matmul)
_llk_unpack_configure_addresses_(address_b, address_a, cfg);
```

The `_llk_unpack_configure_addresses_` function writes these into the THCON registers:

```cpp
cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;  // SEC0 = Unpacker 0 -> SrcA
cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;  // SEC1 = Unpacker 1 -> SrcB
```

So `address_b` (the base address of in0/inA data, which goes to SrcB) is passed as the first argument to `_llk_unpack_configure_addresses_`, which routes it to SEC1. Meanwhile, `address_a` (for SrcA) goes to SEC0. The naming is confusing because the arguments are explicitly swapped at the matmul function level.

If you are computing $C = A \times B$ in the mathematical sense:
- Load matrix $A$ (the left matrix) into `buffer_A` (in0), which the unpacker loads into SrcB.
- Load matrix $B$ (the right matrix) into `buffer_B` (in1), which the unpacker loads into SrcA.
- The hardware then computes $D = SrcB \times SrcA = A \times B = C$.

Getting this backward is a silent error -- the matmul will execute without any hardware fault, but the result will be $B \times A$ instead of $A \times B$, which is incorrect for non-square non-symmetric matrices.

## Putting It All Together: L1 for a Minimal MatMul

For a single 32x32 tile matmul with Float16_b format, the complete L1 picture is:

```
L1 Address Space (Performance Mode)
====================================

0x00000 +-----------------------+
        | BRISC code (16 KB)    |
0x04000 +-----------------------+
        | BRISC loader (8 KB)   |
0x06000 +-----------------------+
        | TRISC0 code (16 KB)   |    Unpack thread firmware
0x0A000 +-----------------------+
        | TRISC0 loader (4 KB)  |
0x0B000 +-----------------------+
        | TRISC1 code (16 KB)   |    Math thread firmware
0x0F000 +-----------------------+
        | TRISC1 loader (4 KB)  |
0x10000 +-----------------------+
        | TRISC2 code (16 KB)   |    Pack thread firmware
0x14000 +-----------------------+
        | TRISC2 loader (4 KB)  |
0x15000 +-----------------------+
        | (reserved gap)        |
0x20000 +-----------------------+
        | Runtime args (1 KB)   |    FormatConfig, Operand addresses
0x20400 +-----------------------+
        | (gap)                 |
0x21000 +-----------------------+
        | buffer_A (2048 B)     |    in0/inA -> loaded to SrcB
0x21800 +-----------------------+
        | buffer_B (2048 B)     |    in1/inB -> loaded to SrcA
0x22000 +-----------------------+
        | buffer_Res (2048 B)   |    Output: D = B * A
0x22800 +-----------------------+
        | (remaining stimuli    |
        |  space, unused)       |
        |  ...                  |
0x16E000+-----------------------+    Firmware-validated L1 limit (MEM_L1_SIZE, ~1.43 MB)
        | (reserved, 72 KB)    |
0x180000+-----------------------+    Physical end of L1 (L1_SIZE, 1.5 MB)
```

The complete data flow for a minimal matmul test:

1. **Python generates tile data** in numpy arrays and packs them into the appropriate binary format (e.g., `pack_bfp16()` for Float16_b).

2. **StimuliConfig.write()** writes the packed data to L1 via `ttexalens` at the computed addresses (0x21000 for buffer_A, 0x21800 for buffer_B).

3. **StimuliConfig generates C++ Operand declarations** that are compiled into the firmware. The firmware knows where each buffer starts and how large each tile is.

4. **During execution**, the unpack kernel reads `buffer_A[tile_index]` and `buffer_B[tile_index]` to get byte addresses, transforms them via `L1_ADDRESS()`, and passes the transformed addresses to `_llk_unpack_AB_matmul_()`. The addresses are validated against `L1_REGION_START` and `L1_REGION_END` before programming the hardware.

5. **After computation**, the pack kernel writes the result tile to `buffer_Res[tile_index]`.

6. **Python reads back** the result from `buf_res_addr` via `ttexalens` and unpacks it for verification against the golden reference.

The address relationship is:

$$\text{HW register value} = L1\_ADDRESS(\text{Operand}[n]) = \left(\frac{\text{base\_addr} + n \times \text{tile\_size}}{16}\right) - 1$$

## Summary of Key Addresses

| Address | Symbol | Description | Source |
|:--------|:-------|:------------|:-------|
| `0x00000` | `L1_BASE` | Start of L1; BRISC code begins here | `memory.blackhole.ld` |
| `0x06000` | `TRISC0_CODE` | Unpack thread firmware | `memory.blackhole.ld` |
| `0x0B000` | `TRISC1_CODE` | Math thread firmware | `memory.blackhole.ld` |
| `0x10000` | `TRISC2_CODE` | Pack thread firmware | `memory.blackhole.ld` |
| `0x20000` | `RUNTIME_ARGS_START` | Runtime arguments region (1 KB) | `memory.blackhole.ld` |
| `0x21000` | `STIMULI_L1_ADDRESS_PERF` | First available address for tile data (perf mode) | `stimuli_config.py` |
| `0x70000` | `STIMULI_L1_ADDRESS_DEBUG` | First available address for tile data (debug mode) | `stimuli_config.py` |
| `0x16E000` | `MEM_L1_SIZE` | Firmware-validated L1 upper limit (~1.43 MB) | `dev_mem_map.h` |
| `0x180000` | `L1_SIZE` | Physical end of L1 (1.5 MB total) | `tensix_types.h` |
| `0xFFB00000` | `LOCAL_MEM_BASE` | Per-core private data memory | `memory.blackhole.ld` |
| `0xFFE40000` | `__instrn_buffer` | Instruction buffer base (provided by linker) | `memory.blackhole.ld` |

---

**Next:** [Chapter 3 — The Three-Thread Kernel Architecture](../ch3_kernel_architecture/index.md)
