# Software Stack Position

TT-LLK is the first software layer above Tenstorrent's Tensix hardware. It is a **header-only C++17 library** of compute primitives that directly drives the Tensix Engine using the custom Tensix instruction set (Tensix ISA). By encapsulating low-level hardware instructions into callable C++ functions, TT-LLK lets developers write kernels without working in raw assembly while still achieving peak hardware utilization.

## What TT-LLK Provides

Each Tenstorrent chip is a grid of Tensix Cores. Within every core, three RISC-V processors (TRISC0, TRISC1, TRISC2) issue Tensix ISA instructions to the Tensix Engine. TT-LLK wraps those instructions into three categories of C++ functions:

- **Unpack operations** (controlled by TRISC0) -- move data from L1 memory into source registers or the destination register, performing format conversion along the way.
- **Math operations** (controlled by TRISC1) -- execute matrix and vector computations on the FPU and SFPU.
- **Pack operations** (controlled by TRISC2) -- move results from the destination register back to L1 memory, with optional format conversion.

Because TT-LLK is header-only, these functions are compiled directly into the kernel binary that runs on each TRISC. There is no separate library to link against; you include the headers, and the compiler inlines the Tensix ISA sequences into your kernel.

## Position in the Software Stack

TT-LLK sits at the bottom of a layered software stack. The diagram below shows the flow from a high-level ML framework down to hardware:

```
+---------------------+
|   ML Framework      |   PyTorch, TensorFlow, ONNX, ...
+---------------------+
          |
          v
+---------------------+
|     TT-Forge        |   Compiler: translates ML graphs into optimized
|                     |   kernel programs for Tenstorrent hardware
+---------------------+
          |
          v
+---------------------+
|    TT-Metalium      |   Runtime SDK: compiles kernels, manages memory,
|    (TT-Metal)       |   dispatches work across the chip's core grid
+---------------------+
          |
          v
+---------------------+
|      TT-LLK         |   Header-only C++17 library of compute primitives
|                     |   (this repository)
+---------------------+
          |
          v
+---------------------+
|   Tensix Hardware    |   Tensix Cores, Tensix Engine, L1 SRAM,
|                     |   FPU, SFPU, Unpackers, Packers
+---------------------+
```

### TT-Forge

[TT-Forge](https://github.com/tenstorrent/tt-forge) is the compiler layer. It accepts models from ML frameworks, applies graph-level optimizations, and lowers operations into kernel programs that ultimately call TT-LLK functions. When a framework-level "matmul" or "softmax" reaches the hardware, it has been decomposed into sequences of LLK unpack, math, and pack calls.

### TT-Metalium (TT-Metal)

[TT-Metalium](https://github.com/tenstorrent/tt-metal) is the open-source runtime SDK. It handles:

- Compiling kernel source code (which includes TT-LLK headers) into binaries for each TRISC.
- Allocating L1 memory and DRAM buffers.
- Dispatching compiled kernels across the chip's grid of Tensix Cores.
- Managing data movement between cores via the Network on Chip (NOC).

TT-Metal is the primary consumer of TT-LLK. When you write a compute kernel in TT-Metal, you call LLK functions (such as `llk_unpack_AB`, `llk_math_eltwise_binary`, `llk_pack`) that are defined in the TT-LLK headers.

### TT-LLK as the Hardware Abstraction

A typical kernel sequence:

```cpp
// Unpack two tiles from L1 into Source A and Source B
llk_unpack_AB(...);

// Perform element-wise binary operation using the FPU
llk_math_eltwise_binary(...);

// Pack the result from the destination register back to L1
llk_pack(...);
```

## Standalone Development and Testing

A key property of TT-LLK is that it can be developed and tested **independently of TT-Metal**. The `tt-llk` repository contains its own test infrastructure (in the `tests/` directory) that validates LLK APIs by targeting a single Tensix Core on real hardware. This standalone capability means:

- LLK developers do not need a full TT-Metal build to iterate on kernel primitives.
- New operations or bug fixes can be validated at the LLK level before being integrated into the broader stack.
- The repository has its own CI pipeline, making it easier for contributors to verify changes in isolation.

The test environment uses the [SFPI compiler](https://github.com/tenstorrent/sfpi) for building SFPU-related tests and supports all current hardware platforms (see [Repository Structure](./repository_structure.md)).

---

**Next:** [`repository_structure.md`](./repository_structure.md)
