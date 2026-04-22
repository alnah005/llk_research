# Pros and Cons of Writing TT-Metal Kernels Using LLK: Guide Plan

## 1. Audience

**Primary audience:** Compute kernel authors and op developers working in TT-Metal who write or maintain device-side compute kernels that use the TT-LLK stack. Also relevant for API designers evaluating improvements to the kernel-authoring workflow.

**Assumed knowledge:**
- Intermediate C++ (templates, preprocessor macros, `constexpr`).
- Familiarity with tile-based accelerator programming concepts (circular buffers, destination registers, data movement).
- Basic understanding of the Tensix core architecture (three TRISCs: unpack, math, pack) and why a single kernel source is compiled three times.
- Awareness of the four-layer API stack (raw LLK -> LLK API wrappers -> Compute Kernel API -> kernel `.cpp`), as documented in the companion guide "TT-LLK Usage in TT-Metal."

**NOT assumed:**
- Deep knowledge of individual LLK instruction encodings or hardware register layouts.
- Experience writing SFPU operations from scratch.
- Familiarity with Quasar-specific build paths or TLS-style compilation.

---

## 2. Chapter List

---

### Chapter 1: `ch1_trisc_macro_system`

**Description:** Evaluates the advantages and pitfalls of the TRISC macro system (MATH/PACK/UNPACK) that gates code to specific processor pipelines within a single kernel source file.

**Files:**

#### `index.md`
- Chapter introduction and reading order.
- Brief recap of the TRISC compilation model: one `.cpp` file produces three binaries via `TRISC_MATH`, `TRISC_PACK`, `TRISC_UNPACK` defines.

#### `advantages.md`
- Single-source authoring: one file describes the complete compute pipeline, making it easy to see the full data flow (unpack -> compute -> pack) in one place.
- Compile-time elimination: the `MATH(x)`/`PACK(x)`/`UNPACK(x)` macros expand to nothing on non-matching TRISCs, producing zero overhead and tight binaries.
- Implicit pipeline parallelism: the three TRISCs run concurrently; the macro system lets the author write sequential-looking code that actually runs in parallel.
- `ALWI` (always-inline) attribute on API functions ensures no call overhead despite the layered abstractions.

#### `pitfalls.md`
- Invisible code elimination: developers cannot easily tell which lines run on which TRISC without memorizing the macro definitions. Silent no-ops when a `PACK(...)` call is placed in a section that only runs on TRISC_MATH.
- Debugging confusion: breakpoints and stack traces in generated `.cpp` files (`chlkc_math.cpp`, etc.) do not correspond to the original kernel source line numbers.
- Macro hygiene: the `MATH`/`PACK`/`UNPACK` macros use simple expansion, so complex expressions with commas require extra parentheses (e.g., `PACK((llk_pack_relu_config(ReluType::ZERO_RELU)))`). Missing parentheses cause cryptic compile errors.
- No cross-TRISC static analysis: there is no tooling to verify that the unpack/math/pack pipelines are correctly synchronized. Mismatched `cb_wait_front`/`cb_pop_front`/`cb_push_back` counts between TRISCs cause silent hangs.

---

### Chapter 2: `ch2_compute_kernel_api_abstraction`

**Description:** Assesses how well the user-facing Compute Kernel API (`hw/inc/api/compute/*.h`) abstracts away LLK complexity, and where the abstraction leaks.

**Files:**

#### `index.md`
- Chapter introduction summarizing the four-layer stack and the Compute API's role as the primary author-facing surface.

#### `abstraction_strengths.md`
- Descriptive function names (`binary_op_init_common`, `matmul_tiles`, `pack_tile`, `reduce_tile`) that map to domain concepts rather than hardware registers.
- Orchestration of all three pipelines in single calls: e.g., `binary_op_init_common()` internally calls `UNPACK(llk_unpack_hw_configure...)`, `MATH(llk_math_hw_configure...)`, `PACK(llk_pack_hw_configure...)`.
- `state_configure()` guard avoids redundant hardware reconfiguration when switching between operations -- the developer does not need to track configuration state manually.
- Well-documented doxygen-style tables for function parameters in API headers.

#### `abstraction_leaks.md`
- Manual `reconfig_data_format()` and `pack_reconfig_data_format()` calls: complex kernels (layernorm, groupnorm) require 20+ explicit reconfiguration calls when alternating between operations on different data formats. The API does not automate this.
- Circular buffer index management: developers must manually assign and track `tt::CBIndex::c_0` through `c_31` for all intermediate buffers. No abstraction over buffer allocation.
- Destination register management: the `tile_regs_acquire`/`tile_regs_commit`/`tile_regs_wait`/`tile_regs_release` protocol is fully manual. Forgetting a step or misordering causes hangs with no error message.
- Template parameters like `DST_ACCUM_MODE`, `MATH_FIDELITY`, `BroadcastType` must be known and threaded through correctly. Many are injected via preprocessor defines, creating implicit dependencies.
- The `SFPU_OP_CHAIN_0` macro pattern for fusing SFPU operations after binary/unary ops is a preprocessor-level code injection that is hard to read, debug, or compose.

---

### Chapter 3: `ch3_defines_as_configuration`

**Description:** Analyzes the pros and cons of using preprocessor defines (passed from host-side program factories to kernel compilation) as the primary kernel parameterization mechanism.

**Files:**

#### `index.md`
- Chapter introduction. Overview of the flow: `ComputeConfig.defines` map -> `defines_generated.h` -> `#ifdef` branches in kernel code.

#### `advantages_of_defines.md`
- Zero-cost abstraction: all configuration is resolved at compile time, so the resulting binary is specialized for the exact operation variant. No runtime branching overhead.
- Enables a single kernel source to serve many operation variants (e.g., `eltwise_binary.cpp` handles ADD, SUB, MUL, with/without SFPU fusion, with/without dest accumulation) via `ELTWISE_OP_TYPE`, `SFPU_OP_CHAIN_0`, `DST_ACCUM_MODE`, etc.
- Build cache deduplication: JIT build hashes include all defines, so identical configurations reuse cached binaries.
- Host-device decoupling: the host factory specifies behavior declaratively via string->string maps, keeping the kernel source free of host-side dependencies.

#### `disadvantages_of_defines.md`
- Readability: kernel `.cpp` files become forests of `#ifdef`/`#ifndef`/`#if defined(X) || defined(Y)` blocks. Example: `eltwise_binary.cpp` has 9 conditional compilation directives in 91 lines; `layernorm.cpp` has 17 in a much larger file.
- Implicit contract: the set of defines a kernel expects is not declared anywhere in the kernel file itself. It is only discoverable by reading the program factory source on the host side. There is no schema or validation.
- Combinatorial explosion: each combination of defines produces a distinct JIT-compiled binary. Complex ops can generate dozens of variants, increasing compile time and cache size.
- Error opacity: if a required define is missing, the kernel silently takes the wrong `#ifdef` branch rather than failing with a clear error. Type errors from wrong define values produce cryptic template error messages that reference deep LLK headers.
- Refactoring hazard: renaming or removing a define requires coordinating changes across the program factory (host C++), the kernel source (device C++), and potentially the Python op binding.

---

### Chapter 4: `ch4_tile_processing_pattern`

**Description:** Evaluates the ergonomics and error-proneness of the standard init/acquire/compute/commit/wait/pack/release tile processing pattern used in all compute kernels.

**Files:**

#### `index.md`
- Chapter introduction. Canonical pattern diagram:
  `cb_wait_front -> cb_reserve_back -> tile_regs_acquire -> [compute] -> tile_regs_commit -> tile_regs_wait -> [pack] -> tile_regs_release -> cb_pop_front -> cb_push_back`.

#### `pattern_ergonomics.md`
- The pattern makes the pipeline explicit: acquire/commit hands off from math to pack, wait/release synchronizes pack with math. This maps directly to the hardware pipeline.
- Predictable structure: nearly all kernels follow the same skeleton, making them easy to read once the pattern is learned.
- Helper aliases (e.g., `ACQ()` / `REL()` for `acquire_dst()` / `release_dst()` in layernorm kernels) show that developers create shorthand to reduce verbosity.
- Flexibility: the pattern allows interleaving multiple operations between acquire and commit (e.g., copy-to-dst then binary op then SFPU fusion), enabling complex fused kernels.

#### `common_mistakes.md`
- Protocol violations that cause silent hangs: forgetting `tile_regs_release`, mismatching `cb_reserve_back` count with `cb_push_back`, calling `pack_tile` before `tile_regs_wait`.
- No RAII or scoped guard for destination registers: the acquire/release protocol is entirely manual. A missed `tile_regs_release` in one loop iteration deadlocks the entire core.
- CB synchronization bugs: `cb_wait_front` and `cb_pop_front` counts must exactly match what the reader/writer kernels push/pop. Mismatches cause deadlocks that are extremely hard to diagnose because there are no error messages -- only hangs.
- Tile index management: developers must manually track which DST register index each tile occupies (e.g., `pack_tile(i, cb_out0)` where `i` is the DST index). Off-by-one errors silently corrupt data.
- Nested acquire/commit is undefined: the API provides no warning if `tile_regs_acquire` is called while registers are already acquired.

---

### Chapter 5: `ch5_template_heavy_api_design`

**Description:** Examines the trade-offs of the template-heavy API design in LLK and the Compute Kernel API for kernel authors.

**Files:**

#### `index.md`
- Chapter introduction. Scope of template usage: from `EltwiseBinaryType` enum template params to `DST_ACCUM_MODE`, `MATH_FIDELITY`, `BroadcastType`, and deeply nested `_llk_math_*` templates.

#### `flexibility_and_performance.md`
- Templates enable compile-time specialization: the compiler generates optimal code for each `(EltwiseBinaryType, BroadcastType, MathFidelity)` combination, with dead code eliminated at compile time.
- Template parameters on SFPU functions (e.g., `exp_tile<approx, fast_and_approx, scale_en, skip_positive_check, InputClamping, iterations>`) give precise control over algorithm variants without runtime overhead.
- `constexpr` compile-time arguments (`get_compile_time_arg_val(N)`) interact well with templates, allowing the host to specialize kernel behavior through the JIT pipeline.

#### `developer_friction.md`
- Compile-time costs: each template instantiation is a unique compilation path. The SFPU `exp_tile` function alone has 6 template parameters, each combination producing a distinct function body. JIT compilation of complex kernels can be slow.
- Error message quality: template errors propagate through 4+ layers of API headers. A type mismatch at the Compute API level produces error messages referencing internal `_llk_math_*` functions in `llk_lib/`, with deeply nested template instantiation traces.
- Discoverability: valid template parameter combinations are not always obvious from the API header alone. For example, `MATH_FIDELITY` is typically a preprocessor define rather than an explicit template argument, making it invisible in function signatures.
- Macro-template interaction: patterns like `SFPU_TEMPLATE_PARAMS_KERNEL_FN(calculate_exponential, approx, fast_and_approx, DST_ACCUM_MODE, ...)` mix preprocessor macros with templates, making it nearly impossible to navigate with IDE tools or understand from reading headers alone.

---

### Chapter 6: `ch6_debugging_tools`

**Description:** Evaluates the effectiveness of the current debugging tools available for kernel development: `LLK_ASSERT`, `fake_kernels_target`, generated file inspection, and watcher infrastructure.

**Files:**

#### `index.md`
- Chapter introduction and overview of the debugging toolkit.

#### `available_tools.md`
- `LLK_ASSERT(condition, message)`: conditional assertion in LLK code, enabled via `ENABLE_LLK_ASSERT` runtime option. Three modes: (1) TT-LLK infra mode triggers `ebreak` RISC-V instruction, (2) TT-Metal mode delegates to `ASSERT()` macro, (3) disabled mode compiles but does not execute the condition (preserving type checking via `sizeof`).
- `fake_kernels_target`: CMake target that compiles all kernel `.cpp` files (from `tt_metal/kernels/`, `ttnn/cpp/ttnn/operations/*/kernels/`) at build time against a fake JIT prelude, catching syntax errors and missing includes without requiring hardware.
- Generated file inspection: developers can examine `chlkc_math.cpp`, `chlkc_pack.cpp`, `chlkc_unpack.cpp`, `defines_generated.h`, and `chlkc_descriptors.h` in the JIT build cache to understand exactly what was compiled.
- Watcher infrastructure: `WATCHER_ENABLED` define activates runtime bounds checking on argument accesses and provides hang detection with core dump information.

#### `gaps_and_limitations.md`
- `LLK_ASSERT` is opt-in and off by default: most developers never enable it, so assertions are silent in normal development. The `ebreak` mechanism requires a debugger attached to be useful.
- `fake_kernels_target` catches compile errors but cannot validate runtime correctness, CB synchronization, or define consistency. It also compiles with a fixed set of defines that may not match the actual runtime configuration.
- No printf/logging from compute kernels: unlike data movement kernels which can use `DPRINT`, compute kernels running on TRISCs have extremely limited observability. Most debugging is done by inspecting output tensors.
- Hang diagnosis: when a kernel hangs due to CB synchronization mismatch or protocol errors, the watcher can identify the stalled core but cannot explain why. Developers must manually inspect all three TRISC binaries.
- No single-step debugging support for TRISC processors in common development setups.
- Error messages from JIT compilation failures point to generated files (e.g., line numbers in `chlkc_math.cpp`), not the original kernel source, requiring manual cross-referencing.

---

### Chapter 7: `ch7_sfpu_extension_path`

**Description:** Walks through the end-to-end experience of adding a new SFPU operation, identifying the friction points at each step of the multi-layer integration.

**Files:**

#### `index.md`
- Chapter introduction. Summary of the six-step SFPU extension path: (1) LLK SFPU implementation, (2) Metal llk_sfpu wrapper, (3) Compute API header, (4) sfpu_split_includes wiring, (5) compute kernel, (6) program factory.

#### `step_by_step_walkthrough.md`
- Step 1 -- TT-LLK SFPU implementation: write `ckernel_sfpu_<op>.h` in `tt_llk_<arch>/common/inc/sfpu/` using SFPI intrinsics. Must be done per-architecture if behavior differs across Blackhole/Wormhole/Quasar.
- Step 2 -- Metal SFPU wrapper: write `ckernel_sfpu_<op>.h` in `ckernels/<arch>/metal/llk_api/llk_sfpu/` for each supported architecture. This layer adds Metal-specific conversions and calls the LLK SFPU implementation.
- Step 3 -- Compute API header: write `hw/inc/api/compute/eltwise_unary/<op>.h` with user-facing `<op>_tile_init()` and `<op>_tile()` functions. Must use `SFPU_TEMPLATE_INIT_KERNEL` and `SFPU_TEMPLATE_PARAMS_KERNEL_FN` macros.
- Step 4 -- sfpu_split_includes wiring: add a new `#if SFPU_OP_<OP>_INCLUDE` guard in `sfpu_split_includes.h` and ensure the program factory emits the corresponding define.
- Step 5 -- Compute kernel: write or extend a `.cpp` kernel that calls the new operation.
- Step 6 -- Program factory: create the host-side factory that emits the correct defines, compile-time args, and runtime args.

#### `friction_points.md`
- Per-architecture duplication: the SFPU wrapper must be created separately for each architecture (`blackhole/`, `wormhole_b0/`). The implementations are often identical, but must be physically duplicated because the include paths are architecture-specific.
- The `sfpu_split_includes.h` mechanism requires touching a centralized file for every new SFPU op, creating merge conflicts and a growing 190-line file of `#if` guards.
- Macro-based registration (`SFPU_TEMPLATE_INIT_KERNEL`, `SFPU_TEMPLATE_PARAMS_KERNEL_FN`): these macros obscure the actual function call chain, making it hard to trace from the user-facing API to the SFPU implementation. IDE navigation fails across macro boundaries.
- The 53+ existing headers in `eltwise_unary/` and the growing `sfpu_split_includes.h` file suggest that the extension pattern does not scale well.
- Testing requires the full host-to-device pipeline: there is no lightweight harness for testing a new SFPU operation in isolation.
- The `SFPU_OP_CHAIN_0` mechanism for fusing operations is a code-injection pattern that works but makes debugging fused operations very difficult.

---

### Chapter 8: `ch8_portability_and_improvement_opportunities`

**Description:** Covers how per-architecture divergence (especially Quasar) affects kernel portability, and identifies the highest-impact improvements to the kernel-authoring workflow.

**Files:**

#### `index.md`
- Chapter introduction.

#### `architecture_divergence.md`
- Quasar exclusions: `compute_kernel_api.h` wraps large sections of includes in `#ifndef ARCH_QUASAR` -- binary ops, SFPU, reduce, tilize/untilize, and many tile_move_copy functions are absent on Quasar. This means most existing compute kernels cannot be compiled for Quasar without modification.
- Quasar-specific features: fourth TRISC file (`chlkc_isolate_sfpu.cpp`), `thread_local` qualifiers for RTA base pointers, TLS-style build path, separate linker scripts.
- CB API fragmentation: `cb_api.h` has Quasar-specific stubs for `get_tile_address` and `read_tile_value` that assert-fail at runtime. Kernels using these will compile but crash.
- Impact on kernel authors: writing a portable kernel requires either `#ifndef ARCH_QUASAR` guards throughout or maintaining separate kernel files per architecture. Neither approach scales.
- Matmul divergence: `matmul.h` has 10+ `#ifndef ARCH_QUASAR` guards excluding most matmul API functions, meaning matmul kernels need per-arch variants.

#### `improvement_opportunities.md`
- RAII-based destination register management: a scoped guard (e.g., `DstRegScope`) that automatically calls `tile_regs_release` on scope exit would eliminate an entire class of deadlock bugs.
- CB synchronization validation: compile-time or runtime tooling that verifies matching push/pop/wait/reserve counts across reader, writer, and compute kernels.
- Define schema/contract: a declarative mechanism (annotation or companion file) for kernels to declare which defines they expect, with validation at `CreateKernel` time.
- Automated data format reconfiguration: the API could track the current data format configuration and automatically insert `reconfig_data_format` calls, removing 20+ manual calls from complex kernels.
- Unified SFPU registration: replacing the `sfpu_split_includes.h` file with a more scalable mechanism (e.g., compile-time registration or code generation) to reduce merge conflicts and manual wiring.
- Architecture abstraction layer: a compatibility shim or feature-flag system that enables writing portable kernels without per-architecture `#ifdef` guards.
- Improved error diagnostics: source-map information in JIT-compiled binaries so that error messages reference original kernel source lines rather than generated files.
- Lightweight SFPU testing harness: a mechanism to test new SFPU operations without requiring the full host program pipeline.
- Reducing template depth: flattening some of the 4-layer template chains would improve compile times and error message quality.

---

## 3. Conventions

1. **File paths** are always relative to the repository root of the referenced codebase. Paths starting with `tt_metal/` or `ttnn/` refer to TT-Metal. Paths starting with `tt_llk_*` or `common/` refer to TT-LLK.
2. **Architecture names**: "Blackhole" or `blackhole`, "Wormhole B0" or `wormhole_b0`, "Quasar" or `quasar`. Directory names use lowercase forms.
3. **API layer names**: "Compute API" = `hw/inc/api/compute/*.h`. "LLK API" = `ckernels/<arch>/metal/llk_api/*.h`. "LLK lib" = `tt_llk_<arch>/llk_lib/*.h`.
4. **Pro/Con format**: Each chapter is structured around explicit advantages and disadvantages, supported by concrete code examples from real kernels. Advantages and disadvantages are in separate files for clarity.
5. **Code examples** show actual code from the codebase with file path annotations. When illustrating a problem, the example includes both the problematic pattern and a description of the failure mode.
6. **Kernel references**: The guide uses the following kernels as recurring examples:
   - `eltwise_binary.cpp` (simple, illustrates basic patterns and defines)
   - `eltwise_sfpu.cpp` (simple SFPU pattern)
   - `layernorm.cpp` (complex, illustrates reconfiguration burden and `#ifdef` density)
   - `sdpa.cpp` (very complex, illustrates compile-time arg explosion -- 30 compile-time args)
   - `bmm_large_block_zm_fused_bias_activation.cpp` (complex, illustrates template usage and fused patterns)
   - `groupnorm.cpp` (largest kernel at 823 lines, illustrates scalability challenges)
7. **Cross-repo references**: When a file exists in both TT-LLK (submodule) and TT-Metal, the path is qualified with "(TT-LLK)" or "(TT-Metal)".

---

## 4. Cross-Chapter Dependencies

| Chapter | Depends On | Reason |
|---------|-----------|--------|
| Ch1 (TRISC Macro System) | None | Foundational; introduces the compilation model that all other chapters reference. |
| Ch2 (Compute Kernel API Abstraction) | Ch1 | Needs understanding of the TRISC macro system to explain how the API orchestrates all three pipelines. |
| Ch3 (Defines as Configuration) | Ch1, Ch2 | Needs both the compilation model and the API layer to explain how defines flow from host to kernel behavior. |
| Ch4 (Tile Processing Pattern) | Ch1, Ch2 | The acquire/commit/wait/release pattern spans all three TRISCs and uses Compute API functions. |
| Ch5 (Template-Heavy API Design) | Ch2, Ch3 | Templates interact with both the API layer and the defines system (e.g., `MATH_FIDELITY` as a define used in templates). |
| Ch6 (Debugging Tools) | Ch1, Ch3, Ch4 | Debugging challenges are caused by the TRISC compilation model, define opacity, and protocol-based synchronization. |
| Ch7 (SFPU Extension Path) | Ch1, Ch2, Ch3, Ch5 | Adding a new SFPU op touches the macro system, API layers, define wiring, and template patterns. |
| Ch8 (Portability and Improvements) | All previous | Synthesizes findings from all chapters to assess portability issues and propose improvements. |
