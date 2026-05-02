# Chapter 8 -- Putting It All Together: Test Infrastructure and Alternatives

## 8.3 Alternatives to Pure LLK -- and Why There Is No LLK-Free Path

### 8.3.1 The Central Question

This section examines every conceivable approach to running a matrix multiplication on the Tenstorrent P150 (Blackhole) hardware and evaluates whether any of them bypass Low-Level Kernels. The conclusion, supported by architectural evidence from the hardware and every layer of the software stack, is that no LLK-free path to hardware matmul exists. Every route to the FPU's `MVMUL` instruction passes through LLK code or reimplements its logic.

### 8.3.2 The Hardware Constraint: Why LLK Is Inescapable

To understand why alternatives are bounded, we must revisit the hardware architecture. Five constraints dictate that any path to hardware matmul must, at minimum, configure the unpacker, configure the FPU address modes, issue MVMUL instructions, and configure the packer -- precisely what LLK functions do:

1. **The FPU is the only matrix multiply unit.** The Tensix Engine contains a single FPU with a grid of FPU cells that perform accumulated dot products. There is no DMA-based multiply, no SFPU matmul, no external matrix coprocessor. The `MVMUL` Tensix ISA instruction is the sole mechanism for triggering a matrix-vector multiply in the FPU.

2. **MVMUL operates on source registers, not L1.** Data must be moved from L1 into Source A and Source B registers by the unpacker (controlled by TRISC0) before the FPU can operate on it. Results land in the destination register and must be packed back to L1 by the packer (controlled by TRISC2). This three-phase pipeline is not optional -- it is dictated by the hardware's register file connectivity.

3. **The unpacker requires configuration.** Before `_llk_unpack_AB_matmul_` can move tiles from L1 into source registers, the unpacker hardware must be configured with data format gasket settings, face dimensions, tile layout, and address stride patterns. These configurations are performed by calling `_llk_unpack_hw_configure_` and `_llk_unpack_AB_matmul_init_`. Without them, the unpacker does not know how to interpret the L1 data.

4. **The FPU requires address-mode programming.** Before `MVMUL` instructions can iterate over a tile's faces correctly, the address modification registers (`ADDR_MOD_0` through `ADDR_MOD_6`) must be programmed to control source register pointer increments, destination register pointer increments, and fidelity phase cycling. The `matmul_configure_addrmod()` function in `llk_math_matmul.h` performs this programming. Without it, the MVMUL instruction would read from and write to incorrect register locations.

5. **The packer requires configuration.** After math completes, the packer must be configured with output format gasket settings, tile size, and untilize/tilize mode before it can transfer data from the destination register to L1. `_llk_pack_hw_configure_` and `_llk_pack_init_` perform this setup.

### 8.3.3 The Abstraction Stack

Every path to matmul on the P150 passes through the same layered stack. The higher the entry point, the more is hidden; the lower, the more must be reimplemented:

```
Layer 5:  TT-Forge / PyBuda       (model graph -> device execution)
Layer 4:  TT-Metal ops            (matmul_tiles() + CB management)
Layer 3:  TT-Metal compute kernel (acquire_dst/release_dst + pack_tile)
Layer 2:  LLK API                 (_llk_math_matmul_ + friends)     <-- This guide
Layer 1:  LLK internals           (replay buffers, addr_mods, MOP)
Layer 0:  TTI instructions        (TTI_MVMUL, TTI_SETRWC raw asm)
```

This guide operates at Layer 2, with occasional dips into Layer 1 (as in the no-MOP analysis in Section 8.1.6). The pure LLK approach provides the right balance of hardware visibility and development productivity for understanding how matmul actually executes on a Tensix core.

### 8.3.4 Alternative 1: TT-Metal / TTNN

**What it is.** TT-Metal (also called TT-Metalium) is Tenstorrent's open-source runtime SDK. TTNN is its high-level tensor library. A minimal TT-Metal matmul compute kernel would contain:

```cpp
// TT-Metal compute kernel (simplified)
void MAIN {
    acquire_dst();
    cb_wait_front(cb_in0, onetile);
    cb_wait_front(cb_in1, onetile);
    matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);
    cb_pop_front(cb_in0, onetile);
    cb_pop_front(cb_in1, onetile);
    pack_tile(0, cb_out0);
    cb_push_back(cb_out0, onetile);
    release_dst();
}
```

Under the hood, `matmul_tiles()` calls the same `_llk_math_matmul_init_` and `_llk_math_matmul_` functions documented in Chapters 3--5. The circular buffer (CB) abstraction manages the unpack/pack coordination that pure LLK requires manual handling of.

**Tradeoffs compared to pure LLK:**

| Aspect | TT-Metal | Pure LLK |
|--------|----------|----------|
| Multi-core | Automatic | Not available (single-core only) |
| Memory management | CB abstractions | Manual L1 address management |
| Compilation | Integrated build system | Manual RISC-V cross-compilation |
| Data movement | NOC APIs, reader/writer kernels | Host-side PCIe writes only |
| Performance tuning | Limited to knob parameters | Full hardware register access |
| Debugging visibility | Framework logging | Direct register/L1 inspection |
| Code size | 5--20 lines of compute kernel | 50--120 lines of C++ across 3 TRISCs |

TT-Metal adds significant value beyond LLK: it manages multi-core tiling, DRAM-to-L1 data movement via NOC, circular buffers for streaming pipelines, and host-device synchronization. But the per-core compute kernel is pure LLK. TT-Metal does not have an alternative math backend -- the LLK headers in `tt_llk_blackhole/llk_lib/` are the only path to the FPU.

**Verdict:** LLK is embedded within TT-Metal. Using TT-Metal does not avoid LLK; it wraps it.

### 8.3.5 Alternative 2: TT-Forge (Compiler Path)

**What it is.** TT-Forge is Tenstorrent's ML compiler that accepts models from PyTorch, ONNX, or TensorFlow and compiles them to run on Tenstorrent hardware. When a model contains a `torch.matmul` or `nn.Linear` layer, TT-Forge:

1. Extracts the operation graph.
2. Tiles the matrices according to the Tensix core grid dimensions.
3. Generates TT-Metal programs with appropriate data movement and compute kernels.
4. Compiles those kernels using the RISC-V cross-compiler with LLK headers.

The compilation chain is: `TT-Forge -> TT-Metal -> LLK -> Tensix hardware`. TT-Forge is a strictly higher-level abstraction than TT-Metal, which itself depends on LLK. There is no path through TT-Forge that avoids LLK.

**Verdict:** TT-Forge depends on TT-Metal depends on LLK. The dependency is transitive and unavoidable.

### 8.3.6 Alternative 3: Direct Tensix ISA (Bare-Metal Assembly)

**What it is.** A developer could theoretically bypass the LLK C++ headers entirely and write raw Tensix ISA instructions. The actual matmul instruction is `TTI_MVMUL`:

```cpp
TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B0A0
TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // B0A0
TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B0A1
TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);  // B0A1
// ... 16 instructions per tile in the replay buffer
```

Each `TTI_MVMUL` instruction triggers one 8x16 * 16x16 matrix-vector multiply in the FPU. A full 32x32 tile requires 16 such instructions (4 face-pairs x 4 sub-rows per face-pair).

**What the code would look like.** To perform a matmul without LLK, one would need to:

1. Program unpacker configuration registers (dozens of register writes to set format gaskets, tile dimensions, address strides, face counts).
2. Issue `TT_OP_UNPACR_AB_MATMUL` instructions with correct operand encoding.
3. Program address modification registers (ADDR_MOD_0 through ADDR_MOD_6) with the correct increment/clear/carry-reset patterns for matmul iteration.
4. Program the MOP replay buffer with the sequence of MVMUL instructions, or issue them inline.
5. Issue `TT_OP_MVMUL` instructions with the correct `p_setrwc::CLR_A` / `CLR_B` / `CLR_NONE` flags and address-mod selections.
6. Program packer configuration registers.
7. Issue pack instructions to move results to L1.

This is precisely what the LLK functions do. Consider `matmul_configure_addrmod()` in `llk_math_matmul.h` (676 lines total). This single function programs six address modification registers (ADDR_MOD_0 through ADDR_MOD_5) that control how source and destination register pointers advance during the MVMUL loop. Each register encodes:

- **Source A pointer increment** (advance to the next face within a tile)
- **Source B pointer increment** (advance to the next row-block within a face)
- **Destination register pointer increment** (accumulate into the correct output face)
- **Carry-reset behavior** (whether the pointer wraps or clears after a face boundary)
- **Fidelity-phase cycling** (for HiFi2/HiFi3/HiFi4 modes, re-read the same source data multiple times for higher precision accumulation)

These values vary based on whether the input is transposed, which fidelity level is selected, and whether the tile is half-face or full-face. Getting any single field wrong -- say, a source A increment of 2 instead of 4 when faces are 16x16 -- causes the FPU to read from the wrong source register location, producing silently incorrect results. There is no hardware exception for an incorrect address-mod value; the FPU simply multiplies whatever data happens to be at the wrong register offset.

Writing this in raw ISA would mean reimplementing every line of `matmul_configure_addrmod()`, `matmul_configure_mop()`, and `_llk_math_matmul_()` by hand -- and debugging without the benefit of any existing test suite or known-good reference values.

**Verdict:** Direct ISA programming does not avoid LLK -- it reimplements it. The LLK source is the definitive specification for what register values are correct.

### 8.3.7 Alternative 4: tt-exalens Instruction Injection

**What it is.** The tt-exalens Python library allows injecting individual Tensix instructions from the host without compiling or loading any firmware. Chapter 7, Section 1 documents this capability for device initialization.

**Why it does not scale to matmul.** A single 32x32 tile matmul at LoFi requires:
- ~10 unpacker configuration register writes
- 1 `_llk_unpack_AB_matmul_init_` call (multiple register writes and instruction sequences)
- 1 `_llk_unpack_AB_matmul_` call (issues UNPACR instructions in a loop over faces)
- 6 address modification register programs
- 16 MVMUL instructions per tile (in the MOP replay buffer) or 16+ inline MVMUL calls
- ~5 packer configuration register writes
- 1 pack instruction per output tile

Each `inject_instruction()` call is a host-to-device round trip. The latency of injecting hundreds of instructions individually would be orders of magnitude slower than loading an ELF with the compiled LLK code. Moreover, instruction injection does not handle the TRISC threading model: the three phases (unpack, math, pack) must run on their respective TRISC threads with semaphore-based synchronization.

**Verdict:** Instruction injection is useful for initialization and debugging but is not a viable path for matmul execution.

### 8.3.8 Alternative 5: SFPU-Based Matmul

**What it is.** The SFPU (Special FPU) is a SIMD engine within the Tensix Engine that operates on destination register data. Could matmul be implemented using SFPU instructions instead of the FPU?

**Why it cannot work:**

1. **No source register access.** SFPU reads from and writes to the destination register only. It cannot access Source A or Source B registers. This means it has no way to read two separate input tiles.

2. **No multiply-accumulate between two input tiles.** SFPU operations are unary (operating on one tile in the destination register). While it has multiply and add instructions, they operate on elements within a single tile or against scalar constants -- not between two separate input matrices.

3. **No dot product capability.** Matrix multiplication requires computing dot products between rows and columns. The SFPU processes 32 elements in parallel (one column of a face) but cannot perform the cross-element accumulation needed for dot products between two input operands.

4. **The FPU cell design is the key.** The FPU contains dedicated dot-product hardware: each FPU cell combines a multiplier array with an accumulator. The `MVMUL` instruction causes each cell to compute `D[8,16] = B[8,16] * A[16,16]`, performing 8x16x16 = 2048 multiply-accumulate operations in a pipelined fashion. The SFPU has no equivalent capability.

**Verdict:** SFPU cannot perform matmul. It is architecturally limited to unary operations on destination register contents.

### 8.3.9 Alternative 6: Software Matmul on RISC-V Cores

**What it is.** The Tensix core contains five RISC-V processors (BRISC, NCRISC, TRISC0/1/2). Could matmul be implemented in pure RISC-V software, using the RISC-V ALU for scalar multiply-accumulate?

**Why it is technically possible but absurd:**

1. **Performance.** The RISC-V cores are single-issue, in-order, ~1 GHz processors. A 32x32 tile matmul requires 32,768 multiply-accumulate operations. At one MAC per cycle, this takes ~33 microseconds per tile on a single core. The FPU performs the same operation in tens of nanoseconds (2048 MACs per MVMUL instruction, pipelined). The RISC-V approach would be roughly 1000x slower.

2. **Data format.** The RISC-V cores use standard IEEE floating point (via soft-float or the M extension for integer). They cannot natively process Bfp8_b, Float16_b, or the 19-bit source register format. All format conversions would need software implementation.

3. **No hardware parallelism.** The FPU operates concurrently with the unpacker and packer. The three-TRISC model overlaps data movement with computation. A software matmul on a single RISC-V core serializes everything.

4. **This is not what the hardware is for.** The entire Tensix architecture is designed around the FPU/SFPU pipeline controlled by LLK. Using the RISC-V cores for computation defeats the purpose of the accelerator.

**Verdict:** Technically possible but ~1000x slower and defeats the purpose of the hardware. Not a real alternative.

### 8.3.10 Alternative 7: Custom Kernel Without LLK Headers (Reimplementation)

**What it is.** A developer could write a C++ kernel that targets the Tensix ISA without `#include`-ing any LLK header, instead directly calling the Tensix instruction intrinsics and programming configuration registers through their memory-mapped addresses.

**What the evidence shows.** The existing `matmul_custom_test.cpp` in the repository partially demonstrates this approach. It uses `experimental/llk_math_matmul_custom_no_mop.h`, which replaces the standard MOP-based matmul with explicit inline MVMUL sequences. However, even this "custom" variant:

- Still includes `llk_unpack_AB_matmul.h` for the unpack phase
- Still includes `llk_pack.h` and `llk_pack_common.h` for the pack phase
- Still relies on `ckernel_include.h`, `ckernel_ops.h`, and `cmath_common.h` for register definitions, address-mode types, and hardware constants

The experimental no-MOP header itself still uses the `addr_mod_t` structure, the `ADDR_MOD_*` constants, the `TTI_MVMUL` macro, and the `math::set_dst_write_addr` function -- all defined in the LLK common headers.

To truly bypass all LLK headers, a developer would need to:
1. Define all hardware register addresses from the Tensix register specification
2. Define all instruction encodings (MVMUL opcode, operand field positions, etc.)
3. Reimplement address-mode register programming from scratch
4. Reimplement unpacker and packer configuration from scratch
5. Reimplement the MOP or provide an alternative instruction scheduling mechanism

This is a multi-thousand-line effort that reimplements LLK without using it. The result would be a private fork of LLK logic in a different code organization.

**Verdict:** Reimplementation, not avoidance. The hardware requires the same configuration sequence regardless of which software module provides it.

### 8.3.11 Alternative 8: Functional Simulator

**What it is.** The tt-llk test infrastructure supports a simulator mode that executes kernels on a software model of the Tensix core instead of real hardware. The `--run-simulator` pytest flag activates this path. Internally, `conftest.py` checks `test_target.run_simulator` and, when true, launches an `ExalensServer` instance:

```python
if test_target.run_simulator:
    simulator_path = os.environ.get("TT_UMD_SIMULATOR_PATH")
    _exalens_server = ExalensServer(
        simulator_path=simulator_path,
        port=test_target.simulator_port,
    )
```

The simulator models the Tensix pipeline at the register-transfer level: unpackers, the FPU, the SFPU, destination registers, source registers, and the packer all behave according to their hardware specification. The same compiled ELF binaries that would run on real silicon execute inside the simulator.

**What you get:**
- No physical P150 card required -- useful for CI/CD pipelines or development machines without hardware.
- Deterministic execution -- no thermal noise, no clock jitter, no intermittent silicon errata.
- Cycle-approximate timing information for performance estimation.
- Per-test simulator reset via `test_target.reset_simulator_per_test` for test isolation.

**What you lose:**
- Real hardware validation. Simulator bugs may mask or create issues that do not exist on silicon.
- Performance data is approximate, not measured on actual hardware.
- Execution is significantly slower than hardware (cycle-accurate simulation adds overhead of roughly 1000x).
- Does not exercise actual silicon behavior such as power-dependent timing margins or process-corner-specific effects.

**Relationship to LLK.** The simulator executes the same compiled ELF binaries as real hardware. The kernel code, LLK API calls, data formats, and three-TRISC synchronization are identical regardless of whether execution happens on silicon or in the simulator. The simulator is not an alternative to LLK -- it is an alternative execution target for LLK-based kernels. Nothing about the minimum requirements documented in Chapters 1-7 changes when using the simulator.

**Verdict:** The simulator is an execution target, not an alternative to LLK. The firmware it runs is identical to what runs on hardware.

### 8.3.12 Alternative 9: Pre-Built Op Library

**What it is.** Rather than writing kernels, a developer could call pre-compiled matmul operations from a library (analogous to cuBLAS on NVIDIA GPUs).

**Current state.** Tenstorrent does not ship a pre-compiled op library independent of TT-Metal. The TTNN library in TT-Metal provides `ttnn.matmul()` which is the closest equivalent, but it compiles kernels at runtime (JIT) using LLK headers. There is no binary-only matmul library that hides LLK internals.

If such a library existed, it would internally contain compiled LLK code. The user would not write LLK code, but LLK code would still execute on the hardware. TT-Metal handles the combinatorial space through JIT compilation: kernels are compiled at first use with the exact parameter combination needed, then cached. A static pre-compiled library would either enumerate all valid configurations at build time (producing a very large binary) or JIT-compile at runtime (reintroducing the LLK compilation dependency).

**Verdict:** A pre-built library would hide LLK from the user but not from the hardware. LLK code (or its equivalent) must run on the Tensix cores.

### 8.3.13 Summary: The LLK Dependency Is Architectural

| Alternative | Avoids writing LLK code? | Avoids LLK code running on hardware? | Viable for matmul? |
|---|---|---|---|
| TT-Metal / TTNN | Yes (user writes host code) | No (compute kernels include LLK) | Yes |
| TT-Forge | Yes (user provides ML model) | No (generates TT-Metal kernels with LLK) | Yes |
| Direct Tensix ISA | Yes (writes raw instructions) | No (reimplements LLK register programming) | Theoretically, not practically |
| tt-exalens injection | Yes (injects from host) | No (same instructions, serially injected) | No (too slow, no TRISC concurrency) |
| SFPU-based | N/A | N/A | No (wrong hardware unit) |
| Software on RISC-V | Yes (pure RISC-V code) | Yes | No (~1000x slower) |
| Custom no-LLK kernel | Yes (avoids #include) | No (reimplements LLK logic) | Theoretically, impractical |
| Pre-built op library | Yes (calls library function) | No (library contains compiled LLK) | Hypothetical (does not exist) |
| Functional simulator | No (same ELF binaries) | No (simulates same LLK instructions) | Yes (development only) |

The only alternative that truly avoids LLK code running on the hardware is software matmul on the RISC-V cores, which is approximately 1000x slower and defeats the purpose of the accelerator. The functional simulator is a viable development target but still executes LLK-based firmware. Every other alternative either uses LLK directly, wraps it, or reimplements it.

### 8.3.14 The Root Cause: LLK Is the Hardware Abstraction Layer

LLK is not merely a convenience library. It is the hardware abstraction layer (HAL) for the Tensix Engine. The relationship is analogous to:

- **x86 microcode** for x86 processors -- you cannot avoid it; it implements the ISA
- **GPU shader compiler** for GPUs -- even hand-written PTX ultimately executes SASS instructions that the compiler generates
- **Device driver** for a peripheral -- the OS kernel module mediates all access

The LLK headers define the software contract with the hardware: what register values produce correct behavior, what instruction sequences are valid, and what synchronization protocols the three TRISCs must follow. This contract is not documented anywhere else -- the LLK source code **is** the specification.

The `llk_math_matmul.h` header alone contains 676 lines of address-mode calculation, MOP programming, throttle-level replay buffer sequences, and fidelity-phase loop control. These are not boilerplate wrappers around simple instructions -- they encode complex hardware-specific knowledge about face traversal order, register banking, pipeline hazards, and double-buffer synchronization that would take significant reverse-engineering effort to reconstruct independently.

### 8.3.15 The Minimum LLK API Surface

Distilled from the test survey (Section 8.1) and Chapters 4--6, the irreducible set of LLK calls for a P150 matmul is:

**Unpack (3 calls):**
- `_llk_unpack_hw_configure_<is_fp32_dest_acc_en>(...)` -- format gaskets, tile sizes
- `_llk_unpack_AB_matmul_init_<>(...)` -- matmul-specific unpack configuration
- `_llk_unpack_AB_matmul_<>(...)` -- issue UNPACR to move tiles from L1 to source registers

**Math (6 calls):**
- `_llk_math_matmul_init_<MATH_FIDELITY>(...)` -- address-mod registers and MOP programming
- `_llk_math_pack_sync_init_<DstSync, is_fp32_dest_acc_en>()` -- dest register sync
- `_llk_math_hw_configure_<is_fp32_dest_acc_en>(...)` -- FPU format configuration
- `_llk_math_wait_for_dest_available_<DstSync>()` -- wait for packer to release dest bank
- `_llk_math_matmul_<MATH_FIDELITY>(...)` -- execute MVMUL sequence
- `_llk_math_dest_section_done_<DstSync, is_fp32_dest_acc_en>()` -- signal math complete

**Pack (6 calls):**
- `_llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(...)` -- packer format gaskets
- `_llk_pack_init_<false, false, false>(...)` -- packer strides and output format
- `_llk_pack_dest_init_<DstSync, is_fp32_dest_acc_en>()` -- dest register read-side sync
- `_llk_packer_wait_for_math_done_()` -- wait for TRISC1 completion signal
- `_llk_pack_<DstSync, is_fp32_dest_acc_en, false>(...)` -- pack tile from dest to L1
- `_llk_pack_dest_section_done_<DstSync, is_fp32_dest_acc_en>()` -- signal pack complete

This is 15 irreducible LLK calls. Remove any one and the matmul either cannot execute, cannot read its inputs, or cannot write its outputs.

### 8.3.16 Evidence from the Test Suite

The test survey in Section 8.1 provides additional empirical evidence. Every C++ matmul kernel source in the repository -- across all seven functional tests, two performance tests, and the Quasar variant -- includes LLK headers. Even the experimental `matmul_custom_test.cpp`, which deliberately avoids the standard MOP-based math path, still includes:

- `llk_unpack_AB_matmul.h` for the unpack phase
- `llk_pack.h` and `llk_pack_common.h` for the pack phase
- `experimental/llk_math_matmul_custom_no_mop.h` for the math phase (which itself depends on `ckernel_include.h`, `ckernel_ops.h`, `cmath_common.h`, and `llk_math_common.h`)

There is no test in the repository -- not even an experimental or draft test -- that performs matmul without LLK headers. This is not an oversight; it reflects the architectural reality that LLK is the only available interface to the Tensix Engine's FPU.

### 8.3.17 Conclusion

There is no LLK-free path to hardware-accelerated matrix multiplication on the Tenstorrent P150. The reasons are architectural: the FPU's `MVMUL` instruction requires unpacker configuration, address-mode register programming, and packer configuration that LLK functions provide. Higher-level frameworks (TT-Metal, TT-Forge, TTNN) all depend on LLK for their compute kernels. Lower-level approaches (direct ISA, instruction injection) require reimplementing LLK logic. The only truly LLK-free compute path is software emulation on the RISC-V cores, which is not a practical alternative.

For the purposes of this guide, the minimum requirements to run a matmul on P150 are therefore the minimum requirements to run an LLK-based matmul kernel, as documented in Chapters 1 through 7.

> **Cross-reference:** [Chapter 1](../ch1_p150_and_blackhole/) introduces the LLK architecture. [Chapter 2](../ch2_data_formats_and_memory/) covers the hardware data path. [Chapter 5](../ch5_compile_time_config/) details the Tensix ISA instructions used by LLK matmul. [Chapter 6](../ch6_compilation/) covers the compilation pipeline that transforms LLK C++ into Tensix ELFs. [Chapter 7](../ch7_host_execution/) describes the host-side execution workflow that loads and runs those ELFs.

---

**Previous:** [8.2 -- Python Test Harness](./02_python_test_harness.md)
