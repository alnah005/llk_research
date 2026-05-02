# LLK's Position in the Software Stack

## The First Layer Above Hardware

Low-Level Kernels (LLKs) occupy the lowest rung of the Tenstorrent software stack. They sit directly above the Tensix hardware and provide the first programmable interface to the chip's computational capabilities. As described in the official LLK introduction (`docs/llk/l1/intro.md`):

> LLKs are essential software components of our AI ecosystem. These kernels enable AI model operation on Tensix cores at high performance. They act as the first software layer above the hardware and provide direct access to its computational capabilities.

LLKs control the Tensix Engine using custom Tensix ISA instructions, wrapping the raw instruction encodings in C++ functions that handle register configuration, synchronization, address modifier setup, and format conversion. They transform the complexity of the Tensix instruction set into a structured API organized around three operation categories: unpack, math, and pack.

## The Software Stack Hierarchy

The full Tenstorrent software stack has three layers, with LLK at the foundation:

```
+----------------------------------------------+
|  ML Frameworks (PyTorch, TensorFlow, etc.)   |
+----------------------------------------------+
|  TT-Forge  (compiler / graph optimizer)      |
+----------------------------------------------+
|  TT-Metalium  (runtime SDK / kernel dispatch)|
+----------------------------------------------+
|  TT-LLK  (low-level kernel headers)         |  <-- You are here
+----------------------------------------------+
|  Tensix Hardware  (P150 / Blackhole ASIC)    |
+----------------------------------------------+
```

- **TT-Forge** is a compiler that bridges machine learning frameworks (PyTorch, etc.) with Tenstorrent hardware. It takes high-level model definitions and lowers them into operations that run on Tensix cores. It uses TT-Metalium under the hood.

- **TT-Metalium** (TT-Metal) is an open-source SDK that provides a runtime environment for kernel development. It handles host-device communication, multi-core dispatch, memory management, and NOC data movement. TT-Metal consumes LLK headers to compile compute kernels for the TRISC processors.

- **TT-LLK** is the kernel library. It provides the C++ function definitions that a TRISC firmware binary calls to configure and execute operations on the Tensix Engine. LLK does not handle dispatch, memory allocation, or multi-core coordination -- those are TT-Metal's responsibility.

Each higher layer depends on the one below it. TT-Forge compiles through TT-Metal, and TT-Metal compiles against TT-LLK headers. You can use any layer independently: TT-Forge for maximum abstraction, TT-Metal for custom kernel development, or TT-LLK alone for bare-metal Tensix programming.

## TT-LLK Is a Header-Only Library

A critical architectural fact: **TT-LLK produces no standalone binary.** It is a header-only C++ library. The repository's README states:

> This repository contains header-only low-level kernels (LLK) for Tenstorrent AI chips, including Wormhole, and Blackhole. These kernels serve as foundational compute primitives, acting as building blocks for higher-level software stacks that implement machine learning (ML) operations.

The entire LLK codebase -- for any architecture -- consists of `.h` files. There is no `.cpp` compilation unit, no static library, no shared object. When a kernel is compiled, the RISC-V cross-compiler (`riscv-tt-elf-g++`) includes the relevant LLK headers, and the compiler inlines the LLK function bodies directly into the TRISC firmware binary.

This header-only design has several practical consequences:

1. **No ABI stability concern.** Since LLK functions are inlined at compile time, there is no binary interface to maintain. Changes to LLK internals are picked up on the next recompile.

2. **All configuration is compile-time or firmware-runtime.** Template parameters like `MathFidelity`, `is_fp32_dest_acc_en`, and `untilize` are compile-time constants that select code paths through `if constexpr` and template specialization. The resulting firmware binary contains only the specific instruction sequences needed for the configured operation, with no runtime overhead.

3. **Architecture selection via include paths.** The build system points the compiler at the correct architecture directory (`-I../tt_llk_blackhole/llk_lib`, `-I../tt_llk_blackhole/common/inc`, etc.). Since all architectures expose the same set of header file names with the same function signatures, swapping architectures is simply a matter of changing include paths and the `-DARCH_*` define. No runtime dispatch is needed.

For the P150, the headers are drawn from two locations:

- **`tt_llk_blackhole/llk_lib/`** -- The LLK API headers. These are the functions kernel code calls directly (e.g., `_llk_math_matmul_init_`, `_llk_unpack_AB_matmul_hw_configure_`).
- **`tt_llk_blackhole/common/inc/`** -- The ckernel infrastructure. These provide the Tensix instruction macros, register management, address modifier helpers, and other plumbing that the API headers build upon.

## What LLK Is Not

To set correct expectations:

- LLK programs a single Tensix core. It does **not** manage multi-core parallelism, graph-level optimization, or dispatch -- those are the domain of TT-Forge and TT-Metal.
- LLK assumes data is already in L1 memory. It does **not** handle host-device transfers, PCIe communication, or device drivers. In the test infrastructure, hardware access is provided by TTExalens.

LLK's scope is intentionally narrow: given data in L1 and a configured Tensix core, execute a specific compute operation (such as matrix multiplication) as efficiently as the hardware allows.

## "Pure LLK" vs. LLK Within TT-Metal

There are two distinct ways to use LLK, and this guide is concerned with the first:

### Pure LLK (Using the TT-LLK Test Infrastructure)

The TT-LLK repository includes its own test infrastructure that can compile and dispatch kernels to a single Tensix core **without** requiring TT-Metal.

In pure LLK mode:

- The SFPI cross-compiler (`riscv-tt-elf-g++`) is downloaded and set up by `tests/setup_testing_env.sh`.
- Hardware-specific headers (e.g., `tensix_types.h`, `cfg_defines.h`) are downloaded from the TT-Metal repository during setup into `tests/hw_specific/blackhole/inc/`.
- A Python test harness (`tests/python_tests/helpers/test_config.py`) orchestrates compilation and execution: it generates a `build.h` header with compile-time parameters, compiles separate ELF files for each TRISC thread (unpack, math, pack), and a fourth ELF (brisc.elf) for the BRISC core to manage boot and lifecycle.
- The compilation process feeds a synthetic compilation unit to the compiler via stdin: `#include <test_name>` followed by `#include <trisc.cpp>`. The same source is compiled three times, once for each TRISC thread, with a different `-DCOMPILE_FOR_TRISC=N` flag (0 for unpack, 1 for math, 2 for pack).
- The compiled ELFs are loaded to device memory via `ttexalens`, BRISC releases the TRISC cores from reset, and the test framework polls completion mailboxes.
- Data is written to and read from L1 via `ttexalens` device access functions.

The key benefit of pure LLK is isolation: you test exactly one Tensix core, exactly one operation, with full control over every input byte. There is no runtime to debug, no dispatch overhead, and no multi-core complexity.

### LLK Within TT-Metal

In production, LLK headers are consumed by TT-Metal's kernel compilation pipeline. TT-Metal provides its own build system, runtime, and dispatch infrastructure. When you write a "compute kernel" in TT-Metal, you `#include` LLK headers and call LLK functions, but TT-Metal handles everything surrounding the kernel:

| Aspect           | Pure LLK                                  | LLK in TT-Metal                              |
|:-----------------|:------------------------------------------|:----------------------------------------------|
| Scope            | Single Tensix core                        | Full chip (multi-core)                        |
| Data movement    | Manual L1 writes via `ttexalens`          | Runtime-managed circular buffers and NOC      |
| Kernel dispatch  | Direct ELF loading and reset              | Runtime kernel queue and dispatch             |
| Dependencies     | TT-LLK repo + SFPI compiler only         | Full TT-Metal SDK                             |
| Build system     | LLK test infrastructure (`TestConfig`)    | TT-Metal JIT compilation                      |

The LLK headers themselves are identical in both cases -- the same `llk_math_matmul.h` file is included whether you compile through TT-Metal or through the pure LLK test harness. What differs is everything surrounding the kernel.

## What This Means for a Minimal MatMul

To run a matrix multiplication on the P150 using pure LLK, we need to work within the TT-LLK test infrastructure. Concretely, this means:

1. **Setting up the toolchain** -- Running `setup_testing_env.sh` to obtain the SFPI compiler and hardware headers for Blackhole.
2. **Writing three kernel fragments** -- One each for the unpack (T0), math (T1), and pack (T2) threads, using functions from `llk_unpack_AB_matmul.h`, `llk_math_matmul.h`, and `llk_pack.h`.
3. **Compiling** -- Using the test infrastructure's compilation pipeline with `-DARCH_BLACKHOLE` and `-mcpu=tt-bh-tensix`.
4. **Providing input data** -- Writing tile-formatted input tensors to specific L1 addresses via `ttexalens`.
5. **Executing** -- Loading the ELF files onto a Tensix core and waiting for completion via mailbox polling.
6. **Reading results** -- Reading the output tile from L1 and verifying it against a golden reference.

The remaining chapters of this guide walk through each of these steps in detail.

---

**Next:** [Chapter 2 — Tile Formats, Data Formats, and Memory Layout](../ch2_data_formats_and_memory/index.md)
