# Gaps for Kernel-Level Debugging

This section systematically compares the state that LightMetal captures (as inventoried in Section 1) against the full state inventory required for single-kernel replay debugging (as catalogued in Chapter 2). Each gap is described in terms of what is missing, why it matters for replay fidelity, and how large the missing data is in practice. The analysis concludes with a quantitative assessment of LightMetal's coverage and a dependency ordering for gap resolution.

All source references cite files in `/localdev/salnahari/testing_dir/tt-metal/`.

---

## 1. Gap 1: No Compiled Binary Capture

### 1.1 What Is Missing

As detailed in Section 1, 4.3.1, `CreateKernelCommand` stores only `file_name: string` -- a filesystem path, not the compiled binary (the line 79 comment confirms intent: `// Later replace with src, then binary`). What it omits:

- The compiled ELF binaries (one per RISC-V processor per kernel)
- The JIT build hash that identifies the specific compilation
- The `ll_api::memory` objects that hold the binary representations
- DWARF debug symbols (when `TT_METAL_RISCV_DEBUG_INFO` is enabled)
- The build key derived from device architecture and harvesting mask

### 1.2 Why It Matters

The replay handler (Section 1, 6.3) calls `CreateKernel()` with the stored file path, triggering a fresh JIT compilation. This has four consequences:

1. **The replay requires the full TT-Metal build environment**, including the RISC-V toolchain, kernel source tree, and all header dependencies. A standalone replay binary cannot function without them.
2. **The compiled binary may differ from the original** if any of the following have changed: compiler version, header file contents, environment variables, or build flags not captured in the `KernelConfig` (specifically `named_compile_args`, `opt_level`, and `hlk_desc`, as identified in Section 1 of this chapter).
3. **Kernels created from inline source code** (`KernelSource::SOURCE_CODE` via `CreateKernelFromString()`) are not captured at all -- the source string is not serialized, only `file_name` exists in the schema.
4. **For kernel-level debugging, the exact binary is the artifact under investigation.** Re-compiling produces a potentially different binary, defeating the purpose of deterministic replay.

Cross-reference: Ch2, Section 1, Items 1-7 define the full compiled binary and build state inventory that LightMetal omits.

### 1.3 Size of the Missing Data

The binary sizes documented in Ch2, Section 1, Item 7 range from 12-96 KB per compute kernel (optimized) to 30-480 KB (with debug symbols). A typical matmul program with 3 kernels (reader, writer, compute) would require approximately 36-544 KB of binary data, increasing 2-4x with `-g` debug symbols.

### 1.4 Architecture Considerations

On **Wormhole/Blackhole**, each Tensix core runs up to 5 RISC-V processors (BRISC, NCRISC, TRISC0, TRISC1, TRISC2). On **Quasar**, the processor layout is fundamentally different: `QuasarComputeKernel` supports up to 16 compute processors per cluster. The current FlatBuffer `KernelConfig` union does not include `QuasarDataMovementConfig` or `QuasarComputeConfig` variants (present in the C++ `Kernel::Config` variant at `kernel.hpp`, lines 114-119), making Quasar kernels entirely unrepresentable.

---

## 2. Gap 2: No L1 / Circular Buffer Data Snapshot

### 2.1 What Is Missing

As established in Section 1, 4.3.3, `CreateCircularBufferCommand` captures the full `CircularBufferConfig` structure but zero bytes of actual CB data. At the moment a kernel is dispatched, L1 memory contains:

- Input tile data in source CBs, placed there by a preceding data movement kernel or host write
- Partial results in intermediate CBs from previous compute phases
- Semaphore values in L1-mapped semaphore addresses
- Stack and local variable data for running firmware
- The `locally_allocated_address_` assigned by `ProgramImpl` during `finalize_offsets()` -- the actual L1 address where each CB resides (not serialized by the FlatBuffer `CircularBufferConfig`)

None of this is captured by LightMetal. The `EnqueueWriteBufferCommand` captures host-to-device DRAM writes, but CB data comes from data movement kernels (NOC reads from DRAM into L1), a path entirely invisible to LightMetal.

### 2.2 Why It Matters

For single-kernel replay, the compute kernel expects specific tile data to be present in its input CBs. Without this data, the kernel will either read uninitialized memory and produce garbage output, or hang waiting for a semaphore that signals data availability. The input data for a compute kernel is typically the *output* of a preceding data movement kernel, reformatted by the reader kernel (tilized, reblocked, transposed). Replaying the reader kernel would reconstruct the CB contents, but that requires replaying a full program -- not an individual kernel.

Cross-reference: Ch2, Section 3, Items 1-4 define the full L1 memory and tile data state. Ch2, Section 3, Subsection 3.1 identifies pre-dispatch capture as the ideal timing for compute kernel replay.

### 2.3 Size of the Missing Data

| Architecture | L1 Size Per Core | Capture Strategy | Data Per Core |
|-------------|-----------------|-----------------|---------------|
| Wormhole | 1,464 KB | Full L1 dump | 1,464 KB |
| Wormhole | -- | Targeted (active CBs + RT args) | 4-128 KB |
| Blackhole | 1,472 KB | Full L1 dump | 1,472 KB |
| Quasar | 4,096 KB | Full L1 dump | 4,096 KB |

For a matmul running on 64 cores, a full L1 snapshot would be $64 \times 1{,}464\text{ KB} \approx 91\text{ MB}$. However, for single-kernel debugging, typically only the input CBs need capturing. A targeted snapshot (2-8 CBs per core at 2-64 KB each) yields $64 \times 128\text{ KB} \approx 8\text{ MB}$ -- manageable for a debugging tool.

---

## 3. Gap 3: No Semaphore Runtime State

### 3.1 What Is Missing

LightMetal does not capture semaphore state at all. No command in `command.fbs` corresponds to `CreateSemaphore`, `SetSemaphoreValue`, or any semaphore-related API. The `data_collection.hpp` system tracks `DISPATCH_DATA_SEMAPHORE` sizes (line 21) but only for dispatch profiling, not for state capture.

### 3.2 Why It Matters

Tensix kernels use L1 semaphores for inter-kernel synchronization within a core (e.g., between BRISC and TRISC0) and for inter-core synchronization via NOC. A compute kernel typically waits on a semaphore posted by the reader kernel before processing each tile batch. Without the correct semaphore values:

- TRISC0 (unpack) may block waiting for a semaphore that BRISC should have set after writing tiles to the CB
- TRISC1 (math) may block waiting for TRISC0 to signal that unpacking is complete
- TRISC2 (pack) may block waiting for TRISC1 to signal that math is complete

For single-kernel replay, the interceptor must snapshot semaphore values at the exact moment of the target kernel's dispatch and restore them before replay.

Cross-reference: Ch2, Section 2, Item 4 defines the `Semaphore` class. Ch2, Section 4, Item 7 discusses inter-core semaphore state and the distributed semaphore problem.

### 3.3 Size of the Missing Data

Semaphores occupy 4 bytes each at well-known L1 addresses. `NUM_SEMAPHORES = 16` per core (`tt_metal/impl/buffers/semaphore.hpp`, line 15). For 64 cores:

$64 \times 16 \times 4\text{ B} = 4{,}096\text{ B} \approx 4\text{ KB}$

This is negligible in size but critical in correctness.

---

## 4. Gap 4: No NOC Transaction Log

### 4.1 What Is Missing

LightMetal has no representation of NOC (Network on Chip) transactions. The `NOC` and `NOC_MODE` enums in `base_types.fbs` (lines 21-29) define the NOC assignment for data movement kernels but do not capture any actual transactions.

### 4.2 Why It Matters

Data movement kernels issue NOC transactions to move data between DRAM, L1 memories of different cores, and host memory. For debugging data movement kernels, NOC transaction traces are essential for diagnosing incorrect addresses, missing transactions, or deadlock patterns.

### 4.3 Practical Assessment

NOC transaction logging is extremely high-bandwidth. The practical solution is not to capture NOC state but to **ensure the NOC is quiesced** before taking a kernel snapshot. This is achievable by inserting a `Finish()` barrier before the snapshot point, which drains all pending NOC transactions. The interceptor's strategy should be to capture the *result* of NOC activity (L1 snapshots) rather than the activity itself. NOC transaction logging may be useful as an optional diagnostic mode but should not be required for basic kernel replay.

Cross-reference: Ch2, Section 4, Item 8 discusses NOC transaction ordering and why in-flight state is hard to capture.

---

## 5. Gap 5: No Single-Kernel Dispatch Granularity

### 5.1 What Is Missing

LightMetal's capture granularity is the **Program**. As identified in Section 1, 3.4, no `EnqueueProgramCommand` exists in the FlatBuffer schema despite a forward declaration in `lightmetal_replay_impl.hpp` (line 37). The program dispatch event is not serialized -- LightMetal captures program *construction* but not execution. There is no mechanism to:

1. Capture a single kernel within a multi-kernel program in isolation.
2. Snapshot the device state *between* kernel executions within the same program.
3. Replay one kernel without replaying the entire program (and all programs that preceded it in the command sequence).

### 5.2 Why It Matters

The interceptor's core use case is debugging a specific kernel -- for example, "the TRISC0 compute kernel on core (3,4) is producing incorrect tile output at dispatch #47." This requires:

1. Identifying the specific kernel invocation within a specific program dispatch
2. Capturing the state visible to that kernel at that moment
3. Replaying that kernel in isolation, without the other kernels in the program

A kernel-level dispatch identifier would need to encode:

| Component | Description | Example |
|-----------|-------------|---------|
| Program global_id | Which program | 3 |
| Kernel global_id | Which kernel within the program | 5 |
| Dispatch ordinal | Which invocation of this program | 47 |
| Core coordinate | Which physical core (for per-core debugging) | (3, 4) |
| Processor type | Which RISC-V (for multi-processor kernels) | TRISC0 |

None of these are available in the current LightMetal schema. The `EnqueueProgram` call is the natural interception point for the kernel capture system -- the moment when all kernel state is finalized and the program is about to be dispatched to hardware.

Cross-reference: Ch1, Section 2 (the "Focus Model") describes how GPU debuggers use hierarchical identifiers (device, SM, warp, lane) -- the Tenstorrent equivalent is (device, core, processor, dispatch_ordinal).

---

## 6. Gap 6: No Hardware Register State

### 6.1 What Is Missing

The FlatBuffer schema has no representation of hardware register state. The following register categories, as inventoried in Ch2, Section 4, Items 3-5, are entirely absent:

| Register Category | Description | Approx. Size Per Core |
|-------------------|-------------|----------------------|
| Tensix config registers (HW_CFG) | ALU mode, data format, tile dimensions (187 registers) | ~748 B |
| SFPU register file | Special function processor register values | ~256 B |
| Destination register file (DEST) | Accumulator data (up to 16 rows) | 1-4 KB |
| Source register banks (SRCA, SRCB) | Unpacker output register banks | 1-2 KB each |
| RISC-V GPRs (x0-x31 per RISC) | General-purpose registers | 640 B (5 RISCs) |
| RISC-V CSRs | Control/status registers | ~400 B (5 RISCs) |

### 6.2 Why It Matters

For debugging math correctness issues, the state of compute register files (SRCA, SRCB, DEST) is essential -- the developer needs to inspect what the unpacker delivered, what the math pipeline produced, and what parameters the FPU was operating with.

However, at kernel *launch* time, hardware registers are generally at known reset values and are configured by the kernel's preamble code (generated from LLK `init` functions). Register capture becomes essential primarily for **mid-execution debugging** (capturing state at the moment of a failure). For replay from launch, if the kernel binary, L1 memory contents, and configuration are captured correctly, the register state should be reproducible.

Cross-reference: Ch1, Section 1.2 describes how GPU debuggers use side-channel register read paths. Ch2, Section 4, Items 3-5 inventory the Tensix register file state.

### 6.3 Architecture Differences

| Register | Wormhole | Blackhole | Quasar |
|----------|----------|-----------|--------|
| Dest register file depth | 16 rows | 16 rows | TBD (potentially 64 rows) |
| SFPU register width | 32-bit | 32-bit | TBD |
| Register read mechanism | Multi-step SRCA-via-DEST workaround (Ch1, Section 5.3) | Same | Direct `rvdbg_cmd::RD_REG`, `RD_FPREG`, `RD_VECREG` |

---

## 7. Gap 7: Missing Configuration Fields in `KernelConfig`

Section 1, 4.3.1 identified four configuration fields missing from the FlatBuffer schema across all three config types: `named_compile_args` (all three), `opt_level` (all three), `noc_mode` (`EthernetConfig` only -- `kernel.hpp`, line 54), and `hlk_desc` (Ch2, Section 1, Item 6). These omissions mean that replay recompilation may produce different binaries if any kernel uses non-default values for these fields. Each is a correctness issue, not a convenience issue.

Additionally, the FlatBuffer `KernelConfig` union has only three members (`DataMovementConfig`, `ComputeConfig`, `EthernetConfig`), while the C++ `Kernel::Config` variant (`kernel.hpp`, lines 114-119) includes five -- `QuasarDataMovementConfig` and `QuasarComputeConfig` are entirely unrepresentable in the current schema.

---

## 8. Gap 8: No Common Runtime Args Capture

The `Kernel` class distinguishes between per-core runtime args (`core_to_runtime_args_`) and common runtime args (`common_runtime_args_`). Common runtime args are shared across all cores running a kernel and are set via `SetCommonRuntimeArgs()`. There is no `CaptureSetCommonRuntimeArgs` helper in `host_api_capture_helpers.hpp`, and no corresponding FlatBuffer command type. If a kernel uses common runtime args, those args will not be captured or replayed.

Cross-reference: Ch2, Section 2, Subsection 1.2 defines common runtime args.

Size impact: typically 4-20 uint32 values ($20 \times 4 = 80$ bytes). Negligible in size, critical for correctness.

---

## 9. Gap 9: No Launch Message or Go Signal Capture

### 9.1 What Is Missing

The `launch_msg_t` and `go_msg_t` structures written to each core's mailbox during dispatch are not captured. These contain:

- `kernel_config_base` offsets (kernel text, RTA, CB, semaphore offsets)
- The `enables` bitmask (which processors are active)
- `noc_id`, `noc_mode`, `host_assigned_id`
- `dispatch_message_offset`, `master_x`, `master_y`, `signal` (in `go_msg_t`)

Cross-reference: Ch2, Section 2, Items 5-6 provide the full `launch_msg_t` and `go_msg_t` field inventories.

### 9.2 Why It Matters

These values encode the precise memory map for the kernel's execution context on a specific core. Without them, the replay environment must reconstruct the L1 layout from higher-level information, which is error-prone and architecture-dependent.

---

## 10. Gap 10: No Binary Metadata or Environment Versioning

The `LightMetalBinary` root table (`light_metal_binary.fbs`, line 33) has the following TODO:

```
// TODO (kmabee) - Git Hash, Versioning, SystemDesc, etc.
```

The binary contains no git commit hash, architecture identifier, HAL/firmware version, device descriptor, compiler version, or build timestamp. Without versioning, there is no way to detect architecture mismatches, firmware mismatches, or toolchain incompatibilities between capture and replay.

For the interceptor's kernel-level capture, versioning is even more critical because the captured state includes low-level details (register values, memory layouts) that are tightly coupled to specific hardware and firmware versions.

---

## 11. Quantitative Gap Assessment

### 11.1 Coverage by State Category

The following table maps the state categories from Chapter 2 to LightMetal's capture coverage, with cross-references to specific Ch2 sections:

| Ch2 State Category | Section | LightMetal Coverage | Coverage % | Notes |
|--------------------|---------|---------------------|-----------|-------|
| **Compiled binaries** | Ch2 S1, Items 1-7 | File path only | 5% | Path stored; no binary, build hash, or debug symbols |
| **Compile-time args** | Ch2 S1, Item 5 | Positional only | 70% | `compile_args` captured; `named_compile_args` missing |
| **Preprocessor defines** | Ch2 S1, Item 4 | Complete | 95% | `defines` map fully captured in all 3 config types |
| **HLK descriptor** | Ch2 S1, Item 6 | None | 0% | Not represented in FlatBuffer schema |
| **Runtime args** | Ch2 S2, Items 1.1-1.2 | Per-core complete | 80% | Missing common runtime args |
| **CB configuration** | Ch2 S2, Item 2 | Complete | 95% | Full `CircularBufferConfig` captured |
| **CB data contents** | Ch2 S3, Items 2-3 | None | 0% | Only host-to-device data captured |
| **L1 memory snapshot** | Ch2 S3, Item 1 | None | 0% | Not captured |
| **Semaphore state** | Ch2 S2, Item 4 / Ch2 S4, Item 7 | None | 0% | Not captured |
| **Launch/go messages** | Ch2 S2, Items 5-6 | None | 0% | Not captured |
| **NOC state** | Ch2 S4, Item 8 | NOC assignment only | 5% | `noc` field in config; no transaction data |
| **Hardware registers** | Ch2 S4, Items 3-5 | None | 0% | Not captured |
| **Firmware/dispatch state** | Ch2 S4, Items 1-2 | None | 0% | Not captured |
| **Core placement** | Ch2 S2, Item 7 | Complete | 100% | `core_spec` in `CreateKernelCommand` |
| **Buffer allocation** | Ch2 S2 | Complete | 100% | Full `BufferCreateCommand` |
| **Host data payloads** | Ch2 S3, Item 5 | Partial | 60% | Only uint32/uint16/void* types handled |
| **Architecture metadata** | Ch2 S4, Item 1 | None | 0% | No `Arch`, no versioning |

### 11.2 Weighted Assessment

Weighting by importance to single-kernel replay (weights sum to 100):

| Category | Weight | Coverage % | Weighted Score |
|----------|--------|-----------|----------------|
| Compiled binaries | 25 | 5% | 1.25 |
| Runtime args (all types) | 15 | 80% | 12.0 |
| CB config + data | 15 | 47.5% | 7.1 |
| L1 memory snapshot | 15 | 0% | 0.0 |
| Core placement + buffer alloc | 10 | 100% | 10.0 |
| Hardware/semaphore/NOC state | 10 | 1.7% | 0.17 |
| Architecture metadata | 5 | 0% | 0.0 |
| Compile-time config (full, including hlk_desc) | 5 | 66% | 3.3 |
| **Total** | **100** | -- | **33.8%** |

**Assessment: LightMetal provides approximately 34% of the capture infrastructure needed for single-kernel replay debugging.** The largest gaps are in compiled binary capture (contributing 1.25 of a possible 25 points) and runtime memory state (contributing 0 of a possible 15 points for L1 snapshots).

---

## 12. Gap Interactions and Dependency Ordering

The gaps identified above do not exist in isolation. Several interact to create compound effects:

**Binary + L1 interaction:** Without kernel binaries (Gap 1), the replay engine must recompile from source. But recompilation may produce a different binary if the build environment differs. Without L1 snapshots (Gap 2) to validate, there is no way to detect that the replayed kernel is operating on different code. The combination makes silent behavioral divergence likely.

**Semaphore + single-kernel interaction:** Semaphore state (Gap 3) is only meaningful if the interceptor can isolate a single kernel (Gap 5). Without kernel-level granularity, there is no defined moment at which semaphore values should be captured. With it, the "just before kernel launch" boundary becomes the natural capture point.

**Versioning + all other gaps:** Without versioning (Gap 10), every schema extension risks breaking existing binaries. This means the order in which gaps are filled matters: versioning should come first.

These interactions suggest a dependency ordering for gap resolution:

```
Gap 10 (Versioning) --> Gap 1 (Binary) --> Gap 5 (Single-kernel) --> Gap 2 (L1)
                                                                  --> Gap 3 (Semaphore)
                                                                  --> Gap 6 (Registers)
                                                                  --> Gap 4 (NOC log)
```

Versioning is prerequisite to everything. Binary capture is prerequisite to meaningful single-kernel replay. Single-kernel granularity is prerequisite to the device-state captures (L1, semaphore, register) that only make sense at kernel boundaries.

---

## 13. What LightMetal Provides That the Interceptor Should Reuse

Despite the gaps, LightMetal provides substantial infrastructure that should not be rebuilt from scratch:

1. **The FlatBuffer serialization framework**: schema definitions, builder patterns, and verification
2. **The object-to-global-id mapping system**: proven mechanism for serializing object references
3. **The `TraceScope` / `LIGHT_METAL_TRACE_FUNCTION_CALL` pattern**: re-entrancy guard for API-level capture (the interceptor will need its own `DeepTraceScope` at a different depth)
4. **The sequential command replay loop**: simple, correct, extensible
5. **Buffer and CB configuration capture**: complete and well-tested
6. **Runtime args capture** (minus common args): three variants covering the main use cases
7. **The `LightMetalBinary` container**: serialization/deserialization with FlatBuffer verification
8. **The `LightMetalCompare` verification pattern**: demonstrates how to embed golden data and validate replay output
9. **The `data_collection.hpp` dispatch hooks**: already execute at the right point in the dispatch pipeline for deep capture

The interceptor's strategy should be to extend this infrastructure rather than replace it, adding new command types and data payloads while preserving the existing architecture. This is the subject of Section 3.

---

## 14. Capture Budget: Current vs. Proposed

The following table estimates binary sizes for three representative workloads, comparing current LightMetal with a proposed kernel-snapshot extension. These estimates are derived from the per-component analysis in the preceding gaps and from the capture envelope sizes in Ch2, Section "Capture Envelope Size Summary."

### Single matmul (1024x1024 bf16, single core)

| Component | Current LightMetal | With Kernel Snapshot | Delta |
|-----------|-------------------|---------------------|-------|
| Command metadata | ~2 KB | ~2.5 KB | +500 B |
| Buffer write data (2 inputs) | ~4 MB | ~4 MB | 0 |
| Golden/compare data | ~2 MB | ~2 MB | 0 |
| Kernel binaries (3 kernels) | 0 | ~384 KB | +384 KB |
| L1 snapshot (1 core, CB regions) | 0 | ~64 KB | +64 KB |
| Semaphore state (1 core) | 0 | 64 B | +64 B |
| **Total** | **~6 MB** | **~6.5 MB** | **+449 KB (+7%)** |

### Multi-op pipeline (5 ops, 1024x1024, 64 cores)

| Component | Current LightMetal | With Kernel Snapshot | Delta |
|-----------|-------------------|---------------------|-------|
| Command metadata | ~10 KB | ~12 KB | +2 KB |
| Buffer write data | ~24 MB | ~24 MB | 0 |
| Kernel binaries (15 kernels, deduplicated) | 0 | ~1.9 MB | +1.9 MB |
| L1 snapshot (64 cores, targeted CB regions) | 0 | ~8 MB | +8 MB |
| Semaphore state (64 cores) | 0 | ~2 KB | +2 KB |
| **Total** | **~24 MB** | **~34 MB** | **+10 MB (+42%)** |

### Single LLK unit test (32x32 tile, 1 core, 1 compute kernel)

| Component | Current LightMetal | With Kernel Snapshot | Delta |
|-----------|-------------------|---------------------|-------|
| Command metadata | ~1 KB | ~1.2 KB | +200 B |
| Buffer write data | ~2 KB | ~2 KB | 0 |
| Kernel binaries (1 kernel) | 0 | ~128 KB | +128 KB |
| L1 snapshot (1 core, CB regions) | 0 | ~8 KB | +8 KB |
| Semaphore state (1 core) | 0 | 64 B | +64 B |
| **Total** | **~5 KB** | **~139 KB** | **+134 KB** |

**Key observations**: For large workloads, kernel snapshots add a manageable 7-42% to binary size. For small LLK tests, the snapshot dominates but absolute size remains trivial (~139 KB). Semaphore and register data are negligible in all cases. The dominant variable cost is L1 data, which scales with core count and CB sizes.
