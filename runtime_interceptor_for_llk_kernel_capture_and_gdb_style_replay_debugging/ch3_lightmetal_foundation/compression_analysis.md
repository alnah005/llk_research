# Compression Analysis: Chapter 3 Redundancies

## Intra-Chapter Redundancies (Within Ch3 files 01, 02, 03)

REDUNDANCY 1: CRUCIAL
Location: Ch3/02 Section 1.1 ("What Is Missing"), paragraphs describing CreateKernelCommand storing file_name
Overlaps with: Ch3/01 Section 4.3.1, which already provides the full CreateKernelCommand schema, the comment on line 79 ("Later replace with src, then binary"), and the field-level comparison tables
Content: "Ch3/02 re-states that CreateKernelCommand stores file_name: string (line 79 of command.fbs) and quotes the same TODO comment. Ch3/01 already covered this in detail in Section 4.3.1 with the full FlatBuffer schema and the 'Later replace with src, then binary' comment."
Recommendation: Replace Ch3/02 Section 1.1's first paragraph with a cross-reference: "As detailed in Section 1, 4.3.1, CreateKernelCommand stores only a file path (see the line 79 comment confirming intent to replace with source then binary). What it omits is:" -- then proceed directly to the bullet list of missing items. Saves ~120 words.

REDUNDANCY 2: CRUCIAL
Location: Ch3/02 Section 1.2, paragraph about replay calling CreateKernel with file_name triggering JIT recompilation
Overlaps with: Ch3/01 Section 6.3, which already shows the exact replay code for CreateKernelCommand including the call to CreateKernel(*program, cmd->file_name()->c_str(), core_spec, cfg) and explains the recompilation dependency
Content: "Ch3/02 Section 1.2 re-describes the replay handler calling CreateKernel with file_name triggering fresh JIT compilation. Ch3/01 Section 6.3 already provides the full code listing and explains this exact recompilation dependency."
Recommendation: Replace Ch3/02 Section 1.2's first paragraph with: "The replay handler (Section 1, 6.3) calls CreateKernel() with the stored file path, triggering a fresh JIT compilation. This has four consequences:" -- then proceed with the numbered consequences (which are new analysis). Saves ~100 words.

REDUNDANCY 3: CRUCIAL
Location: Ch3/02 Section 2.1, paragraph describing CreateCircularBufferCommand capturing config but no data
Overlaps with: Ch3/01 Section 4.3.3, which already states "This captures CB configuration comprehensively but captures zero CB data contents. The interceptor needs both."
Content: "Ch3/02 re-states that CreateCircularBufferCommand captures CircularBufferConfig structure but zero bytes of actual CB data. Ch3/01 Section 4.3.3 already made this exact observation."
Recommendation: Replace Ch3/02 Section 2.1's opening with: "As established in Section 1, 4.3.3, CreateCircularBufferCommand captures zero bytes of actual CB data. At the moment a kernel is dispatched, L1 memory contains:" -- then proceed to the bullet list. Saves ~80 words.

REDUNDANCY 4: MINOR
Location: Ch3/02 Section 3.1, stating "No command in command.fbs corresponds to CreateSemaphore"
Overlaps with: Ch3/01 Section 4.1, which provides the exhaustive command list (17 entries) that implicitly shows no semaphore commands exist
Content: "The complete command enumeration in Ch3/01 Section 4.1 already establishes the full vocabulary; Ch3/02 repeats that semaphore commands are absent."
Recommendation: Add brief cross-reference: "The complete command vocabulary (Section 1, 4.1) contains no semaphore-related commands." Trim by ~40 words.

REDUNDANCY 5: CRUCIAL
Location: Ch3/02 Section 5.1, paragraphs describing absence of EnqueueProgramCommand in the schema
Overlaps with: Ch3/01 Section 3.4 "Notable absences" paragraph, which already identifies that EnqueueProgram is not captured, and notes the forward declaration in lightmetal_replay_impl.hpp with no corresponding capture helper or FlatBuffer table
Content: "Ch3/02 Section 5.1 re-describes the absence of EnqueueProgramCommand from the CommandType union, the forward declaration in lightmetal_replay_impl.hpp, and the lack of a capture helper. Ch3/01 Section 3.4 already covered all of this."
Recommendation: Replace with: "As identified in Section 1, 3.4, no EnqueueProgramCommand exists in the FlatBuffer schema despite a forward declaration in the replay implementation. The program dispatch event is not serialized." Then proceed directly to the new analysis (why it matters). Saves ~150 words.

REDUNDANCY 6: CRUCIAL
Location: Ch3/02 Section 7 (all four subsections), repeating the missing KernelConfig fields
Overlaps with: Ch3/01 Section 4.3.1, which already provides detailed field-level comparison tables for all three config types showing exactly which fields are missing (named_compile_args, opt_level, noc_mode for Ethernet, hlk_desc) with full analysis
Content: "Ch3/02 Section 7 devotes ~500 words to re-describing the same missing fields (named_compile_args, opt_level, noc_mode, hlk_desc) that Ch3/01 Section 4.3.1 already inventoried in detail with per-config-type comparison tables."
Recommendation: Replace Ch3/02 Section 7 with a brief summary and cross-reference: "Section 1, 4.3.1 identified four configuration fields missing from the FlatBuffer schema: named_compile_args (all three config types), opt_level (all three), noc_mode (EthernetConfig only), and hlk_desc. These omissions mean that replay recompilation may produce different binaries if any kernel uses non-default values for these fields. Quasar config types (Section 1, 8.1) are also entirely missing from the KernelConfig union." Saves ~400 words.

REDUNDANCY 7: MINOR
Location: Ch3/02 Section 8, re-describing the absence of common runtime args capture
Overlaps with: Ch3/01 Section 4.3.4, which already states "there is no SetCommonRuntimeArgs capture. Common runtime args are distinct from per-core args. No CaptureSetCommonRuntimeArgs helper exists."
Content: "Ch3/02 Section 8 restates the common runtime args gap already identified in Ch3/01 Section 4.3.4."
Recommendation: Replace with: "As noted in Section 1, 4.3.4, no CaptureSetCommonRuntimeArgs helper exists." Then add only the new Ch2 cross-reference and size estimate. Saves ~60 words.

REDUNDANCY 8: MINOR
Location: Ch3/02 Section 10, restating the LightMetalBinary TODO about missing git hash, versioning, SystemDesc
Overlaps with: Ch3/01 Section 5.1, which quotes the same TODO and provides the same analysis
Content: "Both sections quote the same TODO comment from light_metal_binary.fbs line 33 and draw the same conclusion about missing versioning."
Recommendation: Replace with: "The versioning gap identified in Section 1, 5.1 (no git hash, architecture ID, or system descriptor in the binary) becomes even more critical for kernel-level snapshots." Saves ~80 words.

REDUNDANCY 9: MINOR
Location: Ch3/02 Section 13 "What LightMetal Provides That the Interceptor Should Reuse" -- 9-item list
Overlaps with: Ch3/01 Section 12 "Summary: What LightMetal Captures" table and general narrative throughout Ch3/01
Content: "The reuse recommendations in Ch3/02 Section 13 largely restate information already present in Ch3/01's summary table (Section 12) and narrative sections."
Recommendation: Keep this section as-is -- it serves a different purpose (forward-looking reuse guidance vs. backward-looking inventory), and the redundancy is minor. Add a brief note: "The following reuse list is derived from the detailed architecture analysis in Section 1."

REDUNDANCY 10: CRUCIAL
Location: Ch3/03 Section 1.1 (Union Member Limit), re-explaining that CommandType has 17 members and uint8 limits unions to 256
Overlaps with: Ch3/01 Section 4.2, which already explains: "FlatBuffers encodes union type tags as uint8, limiting a single union to 256 members. With 17 used, 239 slots remain."
Content: "Ch3/03 Section 1.1 restates the 256-member union limit, the current 17 members, and the remaining capacity."
Recommendation: Replace with: "The CommandType union has 17 members out of a maximum 256 (Section 1, 4.2). The proposal adds 5 new members, bringing the total to 22." Saves ~80 words.

REDUNDANCY 11: MINOR
Location: Ch3/03 Section 1.4, re-explaining UInt32Vector as a workaround for nested vectors
Overlaps with: Ch3/01 Section 4.3.4, which already describes SetRuntimeArgsUint32VecPerCoreCommand using [UInt32Vector] for vector-of-vectors
Content: "The UInt32Vector wrapper table pattern is already demonstrated in Ch3/01's runtime args analysis."
Recommendation: Add cross-reference: "The UInt32Vector wrapper pattern is already used in SetRuntimeArgsUint32VecPerCoreCommand (Section 1, 4.3.4)." Saves ~40 words.

REDUNDANCY 12: MINOR
Location: Ch3/03 Section 3.2, re-explaining that TraceScope fires at depth == 1 and quoting the depth check mechanism
Overlaps with: Ch3/01 Section 3 (full macro analysis including depth == 1 check, TraceScope definition, and code listing)
Content: "Ch3/03 re-explains the TraceScope depth mechanism that Ch3/01 already analyzed in full detail."
Recommendation: Replace the first paragraph with: "As analyzed in Section 1, 3.3, LightMetal's TraceScope fires at depth == 1 (the public API boundary). Deep capture must hook at a different level." Then proceed directly to the DeepTraceScope proposal. Saves ~60 words.

REDUNDANCY 13: MINOR
Location: Ch3/03 Section 4.1, re-listing the four object map types from lightmetal_capture.hpp
Overlaps with: Ch3/01 Section 2.3, which provides the complete object-to-global-ID mapping system with full code listing and analysis
Content: "Ch3/03 Section 4.1 re-states the four map types and shared next_global_id_ counter."
Recommendation: Replace with: "The current maps (Section 1, 2.3) cover Buffer, Program, Kernel, and CBHandle with a shared monotonic next_global_id_ counter." Saves ~40 words.

REDUNDANCY 14: MINOR
Location: Ch3/03 Section 6.1, re-describing data_collection.hpp functions and the DISPATCH_DATA_BINARY category
Overlaps with: Ch3/01 Section 9, which already provides the full data_collection.hpp analysis including the enum values and three recording functions
Content: "Ch3/03 Section 6.1 re-lists the three recording functions and repeats the observation about DISPATCH_DATA_BINARY tracking kernel binary sizes."
Recommendation: Replace with: "The data_collection.hpp integration point (Section 1, 9) already executes at the dispatch boundary and tracks kernel binary sizes via DISPATCH_DATA_BINARY." Then proceed directly to the new proposed hooks. Saves ~100 words.

---

## Cross-Chapter Redundancies (Ch3 vs. Ch1 and Ch2)

REDUNDANCY 15: CRUCIAL
Location: Ch3/01 Section 4.3.1, field-level comparison tables for DataMovementConfig, ComputeConfig, EthernetConfig
Overlaps with: Ch2/01 Section 3 (Config Variants: Field Inventory), which provides complete field inventories for all five config types with identical field names, types, defaults, and capture role analysis
Content: "Ch3/01 provides per-field comparison tables for the three config types (DataMovementConfig, ComputeConfig, EthernetConfig) showing C++ fields vs FlatBuffer fields. Ch2/01 Section 3 already provides the complete C++ field inventories for all five config types. The Ch3 tables add the 'Captured in FBS?' column but repeat all C++ field information."
Recommendation: Restructure Ch3/01 Section 4.3.1 tables to show only: Field name, FBS status (Yes/No), Gap impact. Add cross-reference: "For the complete C++ field inventory, see Ch2, Section 1, Item 3." This avoids duplicating the C++ type, default, and role columns. Saves ~300 words across the three tables.

REDUNDANCY 16: CRUCIAL
Location: Ch3/02 Section 1.3, table of kernel binary sizes per type (DataMovementKernel, ComputeKernel, EthernetKernel)
Overlaps with: Ch2/01 Section 7 (Compiled ELF Binary Artifacts), which provides the same per-kernel-type binary count and size ranges
Content: "Ch3/02 provides a size table for missing binary data (4-64 KB per processor, 4-128 KB per DM kernel, 24-384 KB per compute kernel). Ch2/01 Section 7 provides equivalent size estimates (8-32 KB per DM processor, 4-32 KB per TRISC, 12-96 KB per compute kernel without debug, 30-480 KB with debug)."
Recommendation: Replace Ch3/02 Section 1.3 table with: "The binary sizes documented in Ch2, Section 1, Item 7 range from 12-96 KB per compute kernel (optimized) to 30-480 KB (debug). A typical matmul program with 3 kernels would require approximately 36-544 KB of binary data." Saves ~150 words.

REDUNDANCY 17: MINOR
Location: Ch3/02 Section 1.4, note about Quasar processor layout and QuasarComputeKernel supporting up to 16 processors
Overlaps with: Ch2/01 Section 3.4 (QuasarComputeConfig) and Ch2/04 Section 10 (Quasar-Specific Capture Implications)
Content: "The Quasar processor count and architecture differences are covered in detail in Ch2/01 Section 3.4 and Ch2/04 Section 10."
Recommendation: Replace with a cross-reference: "Quasar's expanded processor layout (Ch2, Section 1, Item 3.4; Ch2, Section 4, Item 10) increases both the binary count (up to 16 compute ELFs) and the KernelConfig representation problem (no Quasar variants in the FlatBuffer union)." Saves ~60 words.

REDUNDANCY 18: MINOR
Location: Ch3/02 Section 2.2, sentence "For single-kernel replay, the compute kernel expects specific tile data to be present in its input CBs"
Overlaps with: Ch2/03 Section 3.1, which provides the same analysis about pre-dispatch capture being ideal because CB data is at rest
Content: "The pre-dispatch capture timing rationale appears in both Ch3/02 and Ch2/03."
Recommendation: Keep as brief context but add cross-reference to Ch2/03 Section 3.1 for the full timing analysis. No significant savings.

REDUNDANCY 19: MINOR
Location: Ch3/02 Section 2.3, L1 size table (Wormhole 1,464 KB, Blackhole 1,472 KB, Quasar 4,096 KB) and multi-core size estimates
Overlaps with: Ch2/03 Section 1.4 comparison table and Section 7 multi-core snapshot estimates
Content: "Ch3/02 provides L1 sizes and multi-core snapshot calculations that closely mirror Ch2/03's tables."
Recommendation: Replace the table with: "L1 sizes per architecture are documented in Ch2, Section 3, Item 1.4. A 64-core matmul full L1 snapshot is approximately 91 MB; a targeted CB-only snapshot is approximately 8 MB (Ch2, Section 3, Item 7)." Saves ~80 words.

REDUNDANCY 20: MINOR
Location: Ch3/02 Section 3.3, stating NUM_SEMAPHORES = 16, 4 bytes each, 64 bytes per core
Overlaps with: Ch2/02 Section 4.1 (Semaphore class: NUM_SEMAPHORES = 16, id 0-15, 4 bytes each) and Ch2/04 Section 7.2 (64 bytes per core for 16 semaphores)
Content: "Semaphore count and size calculation appears in Ch2/02, Ch2/04, and Ch3/02."
Recommendation: Replace with: "Semaphore state is 64 bytes per core (16 semaphores x 4 bytes; Ch2, Section 2, Item 4)." Saves ~40 words.

REDUNDANCY 21: MINOR
Location: Ch3/02 Section 6.1, table of register categories (Tensix config, SFPU, DEST, SRCA, SRCB, RISC-V GPRs, CSRs)
Overlaps with: Ch2/04 Sections 3-4 (Tensix Configuration Registers, SFPU Register File), which provide more detailed register analysis
Content: "The register category inventory in Ch3/02 Section 6.1 is a compressed version of Ch2/04 Sections 3-5."
Recommendation: Replace with: "The register categories identified in Ch2, Section 4, Items 3-5 total approximately 748 B (config) + 256 B (SFPU) + 1-4 KB (DEST) + 1-2 KB (SRCA/SRCB) + 640 B (GPRs) + 400 B (CSRs) per core." Saves ~100 words.

REDUNDANCY 22: MINOR
Location: Ch3/02 Section 5.2, describing the hierarchical kernel dispatch identifier (Program global_id, Kernel global_id, Dispatch ordinal, Core coordinate, Processor type)
Overlaps with: Ch1/01 Section 2 (Focus Model) and Ch1/02 Section 1.1 (Translation to Tenstorrent), which define the equivalent hierarchy as (device, core, RISC processor)
Content: "Ch3/02 proposes a dispatch identifier hierarchy that parallels the focus model from Ch1."
Recommendation: Add cross-reference: "This identifier hierarchy mirrors the Tenstorrent focus model established in Ch1, Section 2: (device, core(x,y), RISC processor), extended with program and dispatch ordinal dimensions for temporal identification." Saves ~40 words.

REDUNDANCY 23: CRUCIAL
Location: Ch3/01 Section 12, "Summary: What LightMetal Captures" table (31 rows)
Overlaps with: Ch3/02 Section 11.1, "Coverage by State Category" table (17 rows)
Content: "Both tables inventory LightMetal's capture coverage. The Section 12 table lists individual data items with Captured?/Where/Fidelity columns. The Section 11.1 table maps Ch2 state categories to coverage percentages. They express the same information from different angles, but the Section 11.1 table adds the coverage percentage and Ch2 cross-references."
Recommendation: Remove the Ch3/01 Section 12 table or compress it to 10 key rows (the ones with partial or no coverage). The Ch3/02 Section 11.1 table is the more valuable version because it includes coverage percentages and Ch2 cross-references. Add a note in Ch3/01: "For the quantitative coverage assessment, see Section 2, Item 11." Saves ~200 words from the Ch3/01 table.

REDUNDANCY 24: MINOR
Location: Ch3/02 Section 14, capture budget tables (single matmul, multi-op, single LLK test)
Overlaps with: Ch3/03 Section 2.3, size estimate tables for KernelSnapshotCommand
Content: "Both sections estimate snapshot sizes for similar workloads. Ch3/02 compares current LightMetal vs. proposed extension; Ch3/03 estimates per-instance sizes for the proposed FlatBuffer tables."
Recommendation: Keep both -- they serve different purposes (Ch3/02 is comparative, Ch3/03 is per-table). Add cross-reference in Ch3/03: "These estimates are consistent with the capture budget analysis in Section 2, Item 14." (This cross-reference already exists.)

REDUNDANCY 25: MINOR
Location: Ch3/03 Section 5.1, re-quoting the TODO from light_metal_binary.fbs about missing git hash/versioning/SystemDesc
Overlaps with: Ch3/01 Section 5.1 and Ch3/02 Section 10, both of which already quote this TODO
Content: "The same TODO comment is quoted in three places across Ch3."
Recommendation: Ch3/03 should reference: "The versioning gap (Section 1, 5.1; Section 2, Item 10) motivates the following proposed schema extension." Remove the direct TODO quote. Saves ~30 words.

REDUNDANCY 26: MINOR
Location: Ch3/01 Section 4.3.1, description of hlk_desc gap referencing "Ch2, Section 1, Item 6"
Overlaps with: Ch3/02 Section 7.4, description of hlk_desc gap also referencing "Ch2, Section 1, Item 6"
Content: "The hlk_desc gap is described in two separate Ch3 files with the same Ch2 cross-reference."
Recommendation: This is acceptable duplication -- Section 1 identifies the gap in the field comparison, Section 2 analyzes its impact. No change needed.

REDUNDANCY 27: MINOR
Location: Ch3/02 Section 6.2, stating "On Wormhole/Blackhole, this uses the multi-step SRCA-via-DEST workaround described in 01_gpu_debugger_architectures.md Section 5.3"
Overlaps with: Ch1/01 Section 5.3, which provides the full description of the SRCA-via-DEST debug array read path
Content: "Ch3/02 briefly references the debug array workaround from Ch1. This is a proper cross-reference, not redundancy."
Recommendation: Keep as-is -- this is the correct pattern (brief mention + cross-reference).

---

## Summary

| Category | Count | Estimated Word Savings |
|----------|-------|----------------------|
| CRUCIAL intra-chapter | 6 (R1, R2, R3, R5, R6, R10) | ~930 words |
| CRUCIAL cross-chapter | 3 (R15, R16, R23) | ~650 words |
| MINOR intra-chapter | 5 (R4, R7, R8, R11-R14) | ~370 words |
| MINOR cross-chapter | 8 (R17-R22, R24-R27) | ~390 words |
| **Total** | **22 redundancies** | **~2,340 words** |

The most impactful changes are:
1. **R6 (CRUCIAL)**: Ch3/02 Section 7 re-describes all missing KernelConfig fields already covered in Ch3/01 Section 4.3.1 -- replace with a 2-sentence summary and cross-reference (~400 words saved).
2. **R15 (CRUCIAL)**: Ch3/01 config comparison tables duplicate Ch2/01 Section 3's field inventories -- restructure to show only FBS coverage status (~300 words saved).
3. **R23 (CRUCIAL)**: Two overlapping summary tables in Ch3/01 Section 12 and Ch3/02 Section 11.1 -- compress the Ch3/01 table (~200 words saved).
4. **R5 (CRUCIAL)**: Ch3/02 re-describes the missing EnqueueProgramCommand already identified in Ch3/01 (~150 words saved).
5. **R16 (CRUCIAL)**: Ch3/02 binary size table duplicates Ch2/01 Section 7 (~150 words saved).

Crucial updates: yes
