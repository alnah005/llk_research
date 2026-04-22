# Friction Points in the SFPU Extension Path

The six-step process described in the walkthrough is functional -- operations do get added, and the system works at runtime. But the developer experience has significant friction. This section analyzes each pain point, its root cause, and its practical impact.

---

## 1. Per-Architecture Duplication

**The problem:** Every new SFPU operation requires separate files in each architecture directory. In TT-LLK, the `common/inc/sfpu/` directories for Blackhole and Wormhole B0 each contain 52 files. In TT-Metal, the `llk_sfpu/` wrapper directories contain 159 (Blackhole) and 158 (Wormhole B0) files respectively.

**Why it matters:** For many operations -- `abs`, `negative`, `relu`, basic comparisons -- the SFPI code is identical across Wormhole B0 and Blackhole. The architectures share the same SFPI ISA for these operations. Yet each requires its own copy of the file.

The Quasar architecture highlights the problem from the opposite direction: it has only 14 SFPU files in TT-LLK, meaning the vast majority of operations available on Wormhole/Blackhole are simply missing for Quasar. Each one will need to be individually ported, even when the math is architecture-independent.

**Concrete cost:** Adding `abs` requires creating `ckernel_sfpu_abs.h` separately in:
- `tt_llk_blackhole/common/inc/sfpu/`
- `tt_llk_wormhole_b0/common/inc/sfpu/`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/` (two files: ckernel + llk_math wrapper)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/` (two files: ckernel + llk_math wrapper)

That is 6 files for the math of a single operation, even when 4 of them contain identical code.

**Root cause:** There is no shared/common SFPU directory in TT-LLK for architecture-independent operations. The `common/` directory at the repo root contains only `llk_assert.h` and `tensor_shape.h` -- it is not used for SFPU code. In TT-Metal, the `hw/ckernels/` directory structure mirrors this per-arch layout without a shared tier.

---

## 2. `sfpu_split_includes.h` Centralized Bottleneck

**The problem:** Every SFPU operation must register itself in a single file:

```
tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
```

This file currently contains 46 conditional include blocks of the form:

```cpp
#if SFPU_OP_ACTIVATIONS_INCLUDE
#include "api/compute/eltwise_unary/activations.h"
#endif
```

**Why it matters:**

1. **Merge conflicts.** Any two PRs that add SFPU operations will conflict on this file, even if the operations are completely unrelated. This creates a serialization point in the development workflow.

2. **Manual bookkeeping.** Forgetting to add the `#if` block here is a silent failure -- the operation will compile in isolation but will not be available at runtime because the include guard will never be set to 1.

3. **The design exists for a good reason** (instruction memory constraints on the RISC-V cores prevent including all headers unconditionally), but the implementation puts the burden entirely on the developer. There is no build-time check that verifies every registered `SFPU_OP_*_INCLUDE` macro has a corresponding entry in this file.

**Root cause:** The split-include mechanism is a manual solution to a real hardware constraint. An automated registration system (code generation, build-system enumeration, or a linker-level solution) could eliminate the manual step while preserving the selective inclusion property.

---

## 3. Macro-Based Registration Obscuring Call Chains

**The problem:** The connection between the host-side `UnaryOpType` enum and the device-side `<op>_tile()` function is mediated entirely through string-based preprocessor defines. The `get_block_defines()` function in `unary_op_utils.cpp` constructs strings like:

```cpp
block_defines["SFPU_OP_CHAIN_0"] = "SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0";
block_defines["SFPU_OP_CHAIN_0_INIT_0"] = "hardsigmoid_tile_init();";
block_defines["SFPU_OP_CHAIN_0_FUNC_0"] = "hardsigmoid_tile(0);";
```

These defines are passed as `-D` compiler flags. On the device side, `eltwise_sfpu.cpp` consumes them through `#ifdef SFPU_OP_CHAIN_0` / `SFPU_OP_CHAIN_0`.

**Why it matters:**

1. **No IDE support.** You cannot "Go to Definition" from `SFPU_OP_CHAIN_0` in the compute kernel to the function it calls. The macro is not defined in any header file.

2. **No compile-time type checking.** The init/func strings are assembled via `fmt::format`. A typo in the function name (e.g., `"hardsigmoid_tile_initt();"`) will only be caught when the device kernel is compiled, which happens at runtime during program creation.

3. **Debugging opacity.** When an SFPU operation produces wrong results, tracing from the failing kernel back to the math implementation requires manually following the string chain: `UnaryOpType::HARDSIGMOID` -> `get_op_init_and_func()` -> `"hardsigmoid_tile"` -> `activations.h` -> `llk_math_eltwise_unary_sfpu_hardsigmoid` -> `ckernel_sfpu_activations.h` -> `_calculate_hardsigmoid_()`. That is six hops with a string-to-symbol boundary in the middle.

4. **Two separate switch statements** in `unary_op_utils.cpp` must stay in sync: `get_macro_definition()` and `get_op_init_and_func()` (or its parameterized variant). A third function, `update_macro_defines()`, is a trivial two-line wrapper that delegates to `get_macro_definition()` and does not contain its own switch statement. Adding a new op requires updating both switch-based functions consistently.

**Root cause:** The code injection approach via `-D` flags is a consequence of the split compilation model (host compiles device kernels at runtime). There is no shared type system between the host C++ and the device C++ -- the preprocessor is the only bridge.

---

## 4. 54 Existing Headers in `eltwise_unary/`

**The problem:** The `tt_metal/hw/inc/api/compute/eltwise_unary/` directory contains 54 files. This is the public compute API surface.

**Why it matters:**

1. **Discoverability.** A developer trying to understand what SFPU operations are available must scan 53 header files. There is no generated index, and the naming is not always obvious (e.g., `comp.h` for comparison operations, `xielu.h` for SiLU/GeLU variants).

2. **Inconsistent grouping.** Some headers contain a single operation (`abs.h`, `cbrt.h`). Others contain families (`activations.h` has hardsigmoid + softsign + celu + softshrink, `trigonometry.h` has sin + cos + tan + sinh + cosh + etc.). The grouping criteria are not documented, and a new developer must decide whether their operation goes in an existing file or a new one.

3. **CMakeLists.txt maintenance.** Each new header must be added to `tt_metal/hw/CMakeLists.txt` for installation. This is another file that sees merge conflicts when multiple SFPU operations are added concurrently.

4. **Linear growth.** There is no namespace or subdirectory organization within `eltwise_unary/`. At the current rate of addition, this directory will continue to grow as new operations are needed.

---

## 5. No Lightweight Testing Harness

**The problem:** There is no way to test a new SFPU operation in isolation without building and running a full TT-Metal program on hardware (or a device simulator).

**Why it matters:**

1. **The TT-LLK test infrastructure** (`tests/` directory) does have Python-based test scripts and some helpers (`sfpu_stub.h`, `sfpu_operations.h`), but these are designed for integration testing on hardware, not for unit-testing a single SFPU function's math.

2. **Iteration speed.** The typical cycle for developing a new SFPU operation is: write the SFPI code, create all the wrapper files, build TT-Metal, write a test program, run on hardware, check results. If the math is wrong, you repeat the entire cycle. There is no way to run just the SFPU math function on a host CPU with simulated SFPI semantics.

3. **Math validation gap.** Many SFPU operations involve numerical approximations (polynomial evaluations, lookup tables, range reductions). Validating these approximations against reference implementations requires running on hardware. A CPU-side SFPI simulator could catch precision issues before they reach the hardware test stage.

**Root cause:** SFPI intrinsics compile to Tensix-specific instructions that have no x86/ARM equivalent. Building an SFPI emulation layer would require significant investment but would dramatically improve the development loop.

---

## 6. `SFPU_OP_CHAIN_0` Code Injection Difficulty

**The problem:** The `SFPU_OP_CHAIN_0` mechanism injects arbitrary C++ statements into the compute kernel via preprocessor defines. The macro expands inline in the kernel's hot loop:

```cpp
// eltwise_sfpu.cpp
copy_tile(tt::CBIndex::c_0, 0, 0);
#ifdef SFPU_OP_CHAIN_0
    SFPU_OP_CHAIN_0    // <-- expands to: hardsigmoid_tile_init(); hardsigmoid_tile(0);
#endif
tile_regs_commit();
```

**Why it matters:**

1. **Debugging is painful.** The preprocessed kernel source (which is what actually compiles) differs from the source file on disk. Error messages from the device compiler reference line numbers and symbols that do not appear in the original `.cpp` file.

2. **Limited composability.** The chain mechanism supports multiple operations (`SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0 SFPU_OP_CHAIN_0_INIT_1 SFPU_OP_CHAIN_0_FUNC_1 ...`), but because everything is string concatenation, there is no type safety or validation that the chained operations are compatible.

3. **No access to intermediate state.** If you need to inspect or transform values between chained operations, you cannot -- the chain is a flat sequence of statements. There is no hook point between operations in the chain.

4. **Scope for subtle bugs.** Because the init call and the tile call are separate strings assembled independently, it is possible to have a mismatch (e.g., initializing for one operation but calling another). The host-side code constructs these strings in `get_block_defines()` and the per-op `get_defines_impl()` functions, but there is no structural guarantee they match.

5. **Multiple consumption points.** The `SFPU_OP_CHAIN_0` macro is consumed in at least 7 different kernel files across TT-Metal and TTNN:
   - `tt_metal/kernels/compute/eltwise_sfpu.cpp`
   - `tt_metal/kernels/compute/eltwise_binary.cpp`
   - `ttnn/.../binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`
   - `ttnn/.../binary/device/kernels/compute/eltwise_binary_kernel.cpp`
   - `ttnn/.../unary/device/kernels/compute/eltwise_sfpu.cpp`
   - `ttnn/.../unary/device/kernels/compute/where_tss_kernel.cpp`
   - `ttnn/.../examples/example/device/kernels/compute/eltwise_sfpu.cpp`

   Each of these files independently uses `#ifdef SFPU_OP_CHAIN_0` with the assumption that the macro content is well-formed C++ statements. A change to the macro generation in `unary_op_utils.cpp` affects all of them.

**Root cause:** The code injection pattern exists because the compute kernel must be a single compilation unit (it runs on a bare-metal RISC-V core with no dynamic linking). The only way to parameterize which SFPU operation runs is at compile time, and `-D` preprocessor defines are the simplest mechanism for compile-time code selection. The alternative would be a function pointer or switch-based dispatch, but that would add runtime overhead on the performance-critical path.

---

## Summary

The six friction points share a common theme: the SFPU extension path trades developer experience for runtime performance and hardware compatibility. Each design choice has a defensible rationale -- per-arch files allow hardware-specific optimizations, split includes save instruction memory, macro injection avoids runtime dispatch overhead. But the cumulative effect is a process that requires touching 7-10+ files across two repositories to add a single mathematical function, with minimal tooling support and no lightweight validation path.

The highest-impact improvements would be:
1. A shared SFPU directory for architecture-independent operations (eliminates ~50% of file duplication).
2. Automated `sfpu_split_includes.h` generation from a registry (eliminates the merge conflict bottleneck).
3. An SFPI emulation layer for host-side testing (eliminates the hardware-in-the-loop requirement for math validation).

---

**Next:** [Chapter 8 -- Portability and Improvement Opportunities](../ch8_portability_and_improvement_opportunities/index.md)
