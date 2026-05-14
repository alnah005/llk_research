# Agent C Compression Analysis: Chapter 2 -- Anatomy of a Micro-Op

## Pass 1: Structure Inventory

### Per-file line counts

| File | Lines | Role |
|------|-------|------|
| `index.md` | 36 | Chapter overview, learning objectives, nav table |
| `01_directory_structure.md` | 280 | Filesystem layout, auto-discovery, registration flow, name/op_class |
| `02_python_class.md` | 456 | MicroOp subclassing, ports, config, compose(), emit() overview, walkthroughs |
| `03_cpp_kernel_header.md` | 758 | op.hpp structure, CT arg structs, typed fields, A:: pattern, lifecycle, walkthroughs |
| `04_auto_derivation.md` | 577 | _auto_derive_from_kernel_hpp(), dataclasses, regex, overrides, edge cases, mistakes |
| **Total** | **2107** | |

### Section structure by file

**index.md (36 lines)**
- "What you will learn" (learning objectives list)
- Prerequisites
- Chapter structure nav table
- Minimal. No compression opportunity.

**01_directory_structure.md (280 lines)**
- 2.1.1 Canonical three-file layout (~60 lines): tree diagrams, 3 real-world examples, file-to-consumer table, `_find_kernel_hpp()` code
- 2.1.2 Auto-discovery with `register_all()` (~52 lines): code listing, 4-step table, _skip_subpackages, common mistakes
- 2.1.3 Registration flow (~52 lines): ASCII flow diagram, 5-registry table, RMSNorm trace
- 2.1.4 name vs. op_class (~40 lines): usage tables, precedence chain
- 2.1.5 File checklist (~20 lines): numbered steps for adding a new op
- Key takeaways (~15 lines)

**02_python_class.md (456 lines)**
- 2.2.1 __init_subclass__ (~30 lines): code + table
- 2.2.2 Port descriptors (~70 lines): declaration, __set_name__, port types table, port-name contract, OpSpec
- 2.2.3 Class config fields (~20 lines): table + example
- 2.2.4 compose() (~50 lines): signature table, 3 walkthroughs (RMSNorm, Copy, Matmul)
- 2.2.5 emit() overview (~80 lines): 5-step pattern, CB allocation table, CT arg registration table, prefix pattern, float-to-bits
- 2.2.6 RMSNorm walkthrough (~70 lines): full class structure code + 5 observations
- 2.2.7 Copy walkthrough (~55 lines): full class structure code + 5 observations
- 2.2.8 Contrast table (~20 lines): RMSNorm vs. Copy comparison
- Key takeaways (~15 lines)

**03_cpp_kernel_header.md (758 lines)**
- 2.3.1 Five-layer structure (~55 lines): annotated code + table + static constexpr explanation
- 2.3.2 CT arg structs and RISC mapping (~55 lines): struct-to-RISC table, execution effect prose, is_inter_core derivation code
- 2.3.3 Typed fields (~55 lines): 6-type table, type aliases code, _TYPE_TO_SOURCE mapping, Flag behavior
- 2.3.4 A:: template pattern (~35 lines): end-to-end flow diagram, 4 field patterns
- 2.3.5 Op lifecycle (~90 lines): init/operator()/teardown, lifecycle detection regex, empty body detection, RMSNorm lifecycle example
- 2.3.6 Preprocessor guards and role flags (~70 lines): guard table, role flags table, if constexpr examples, comparison table
- 2.3.7 cpp_parser.py regex patterns (~35 lines): 7-pattern table, parsing flow description
- 2.3.8 RMSNorm kernel walkthrough (~90 lines): CT arg struct code, parser output, merged CT args, Op class code, observations
- 2.3.9 Mcast kernel walkthrough (~140 lines): 4 CT arg struct code blocks, parser output, Op class with role flags, setup methods, observations, comparison table
- Key takeaways (~15 lines)

**04_auto_derivation.md (577 lines)**
- 2.4.1 Why auto-derivation exists (~10 lines)
- 2.4.2 Auto-derivation pipeline (~75 lines): step-by-step code + assignment guards table
- 2.4.3 ParsedKernel/ParsedCTArg dataclasses (~70 lines): dataclass definitions, field tables, RISC flag derivation code
- 2.4.4 CT arg merging (~25 lines): _merge_args code + example
- 2.4.5 _resolve_ct_arg_kind() (~40 lines): code + resolution table
- 2.4.6 Mcast concrete trace (~30 lines): step table + PhaseInfo comparison
- 2.4.7 When to override (~60 lines): 4 scenarios with code examples
- 2.4.8 Edge cases in field parsing (~55 lines): passthrough, aliased, derived, internal constants, decision tree, pop_input asymmetry
- 2.4.9 Five most common mistakes (~60 lines): 5 mistakes with code, problem, effect, fix
- 2.4.10 _fill_phase_lifecycle_from_hpp() fallback (~30 lines): code + explanation
- 2.4.11 Summary data flow (~35 lines): ASCII diagram
- Key takeaways (~20 lines)

### Largest sections (by approximate line count)

1. 2.3.9 Mcast kernel walkthrough (~140 lines)
2. 2.3.5 Op lifecycle (~90 lines)
3. 2.3.8 RMSNorm kernel walkthrough (~90 lines)
4. 2.2.5 emit() overview (~80 lines)
5. 2.4.2 Auto-derivation pipeline (~75 lines)

---

## Pass 2: Redundancy Analysis

### File: `01_directory_structure.md`

**Redundancy found: RISC flag derivation code block (lines ~193-202).**
The RMSNorm registration trace in 2.1.3 includes a description of how `has_ncrisc`, `has_trisc`, `has_brisc` are set from struct presence. This same logic is explained in full detail in `03_cpp_kernel_header.md` section 2.3.2 (lines 108-124) and `04_auto_derivation.md` section 2.4.3 (lines 169-184). However, the 2.1.3 version is a concrete trace (applied to RMSNorm specifically), not a general explanation, so it serves a different pedagogical purpose. **Not a compression candidate** -- it is an applied example, not duplicated explanation.

**Over-explanation: None.** The code blocks are structural (real source code), and the prose explains what the code does not show (e.g., ordering, error effects).

**Low-value content: None.** Every section serves someone building a new micro-op (the file checklist in 2.1.5 is directly actionable).

**Verdict: No compression opportunity.**

### File: `02_python_class.md`

**Redundancy found: compose() walkthroughs (lines 160-193).**
Three compose() walkthroughs (RMSNorm, Copy, Matmul) each show a 3-5 line method body. The RMSNorm and Copy compose() methods are shown again verbatim in the structural walkthroughs at 2.2.6 and 2.2.7 (embedded in the full class listing). The standalone walkthroughs in 2.2.4 and the embedded versions in 2.2.6/2.2.7 serve the same reader -- but the standalone ones are reached first and provide focused context, while the embedded ones show the method in its full class context. **Marginal redundancy** (~15 lines for the two duplicated compose() bodies), but removing either location would damage the reading flow.

**Redundancy found: CT arg registration method table (lines 243-251).**
This table listing `f.ncrisc_ct_args()`, `f.brisc_ct_args()`, etc. overlaps with the struct-to-RISC mapping table in `03_cpp_kernel_header.md` section 2.3.2 (lines 79-84). However, the 02 table focuses on the Python API methods, while the 03 table focuses on the C++ struct names. They present the same mapping from opposite sides. **Not redundant** -- they serve different contexts (Python emit() author vs. C++ header reader).

**Over-explanation: Float-to-bits conversion (lines 270-285).**
The prose explains `struct.pack`/`unpack` and shows both the Python and C++ sides. This is clear and useful for the target reader. **Not over-explained.**

**Low-value content: None.**

**Verdict: ~15 lines of marginal redundancy (duplicated compose() method bodies). Not worth cutting -- the structural walkthroughs in 2.2.6/2.2.7 need to show the full class, including compose().**

### File: `03_cpp_kernel_header.md`

**Redundancy found: RISC flag derivation from struct names (lines 108-124).**
This code block and explanation also appears in `04_auto_derivation.md` section 2.4.3 (lines 169-184). The two versions are nearly identical -- same code block, same explanatory comments. The 03 version explains it as part of the parser behavior; the 04 version explains it as part of auto-derivation data flow. **This is genuine cross-file redundancy.** Estimated: ~15 lines duplicated.

However, removing it from either location would create a gap. In 03, the reader needs to understand how the parser derives RISC flags to understand the `is_inter_core` explanation that follows. In 04, the reader needs it for the complete auto-derivation data flow. **Recommendation: Keep both, but the 04 version could add a brief cross-reference rather than repeating the code block.** Estimated savings: ~10 lines.

**Redundancy found: `is_inter_core` explanation from CoreCTArgs.**
The fact that CoreCTArgs sets all three RISC flags and thus makes `is_inter_core = True` is explained in:
- 03_cpp_kernel_header.md, section 2.3.2 (lines 106-124)
- 03_cpp_kernel_header.md, section 2.3.3 concept reiteration (N/A -- not duplicated within 03)
- 04_auto_derivation.md, section 2.4.3 (lines 186-187)
- 04_auto_derivation.md, section 2.4.7 Scenario 3 (lines 317-327)
- 02_python_class.md, section 2.2.7 observation 5 (lines 418)

This is explained 4 times across 3 files. However, each mention is in a different context (parser behavior, dataclass semantics, override scenario, Copy structural observation). Three of these are brief single-sentence references. Only the 03 and 04 versions have full code blocks. **Moderate redundancy but difficult to cut** because each context legitimately needs the reader to understand this fact.

**Over-explanation: "Why static constexpr?" (lines 66-72).**
Explains `static` and `constexpr` keywords. The target reader (someone building micro-ops) likely knows basic C++. However, this is only 6 lines and serves as a quick refresher. **Not worth cutting.**

**Low-value content: None.** The Mcast walkthrough (2.3.9) is the longest section but provides the only inter-core example in the chapter -- essential reference material.

**Verdict: ~10 lines compressible by cross-referencing the RISC flag derivation code block in 04 instead of repeating it. Otherwise clean.**

### File: `04_auto_derivation.md`

**Redundancy found: RISC flag derivation code block (lines 169-184).**
As noted above, this duplicates the code block in `03_cpp_kernel_header.md` section 2.3.2. **Could be replaced with a cross-reference + brief summary (~10 lines saved).**

**Redundancy found: Four field patterns repeated (lines 348-400).**
Section 2.4.8 "Edge cases in field parsing" re-explains the four field patterns (passthrough, aliased, derived, internal constant) that were already presented in `03_cpp_kernel_header.md` section 2.3.4 (lines 208-237). The code examples are identical. The 03 version introduces the patterns; the 04 version shows them as edge cases in the auto-derivation context. **This is genuine redundancy** -- the patterns are the same, the code examples are the same, and no new information is added in the 04 version except the decision tree (lines 392-400) which IS new. Estimated: ~30 lines of duplicated field-pattern examples. The decision tree and `pop_input` asymmetry sections are unique to 04.

**Recommendation:** Replace the repeated field-pattern examples in 2.4.8 with a cross-reference to 2.3.4, keeping only the decision tree and the pop_input asymmetry analysis. Estimated savings: ~25 lines.

**Redundancy found: pop_input asymmetry explanation.**
The pop_input asymmetry between RMSNorm and Matmul is explained in:
- 03_cpp_kernel_header.md, section 2.3.8 (lines 513-515): brief note
- 04_auto_derivation.md, section 2.4.8 (lines 402-422): full analysis with code

These are not redundant -- the 03 version is a brief observation, the 04 version is the full analysis. The 03 version correctly cross-references "Compare with Matmul" as context. **Not a compression candidate.**

**Redundancy found: Mistake 5 (aggregator include) in 2.4.9 (lines 479-483).**
This same mistake is mentioned in 01_directory_structure.md section 2.1.5 step 5 (lines 256-260) and the key takeaways (line 277). The 04 version adds no new information beyond what 01 already says. However, it is part of a "common mistakes" reference list, where completeness is a feature. **Marginal redundancy (~5 lines) but acceptable in a mistakes checklist.**

**Over-explanation: None.** The auto-derivation pipeline code walkthrough is detailed but necessary -- this is the most complex mechanism in the chapter.

**Low-value content: None.**

**Verdict: ~25 lines compressible by cross-referencing field patterns from 2.3.4 instead of repeating them. ~10 lines compressible by cross-referencing RISC flag derivation from 2.3.2.**

---

## Pass 3: Verdict

### 1. Total line count
**2107 lines** across 5 files.

### 2. Compression target: Could this be reduced by >15% without losing information?
**No.** The maximum identifiable compression is approximately 50-60 lines (2.4-2.8%), far below the 15% threshold (~316 lines). The chapter is well-structured with minimal genuine redundancy. The bulk of the content is code examples, tables, and walkthrough prose that serve the reference guide purpose.

Breakdown of compressible content:
- RISC flag derivation code duplication between 03 and 04: ~10 lines
- Four field patterns repeated between 03 section 2.3.4 and 04 section 2.4.8: ~25 lines
- Duplicated compose() method bodies between 2.2.4 and 2.2.6/2.2.7: ~15 lines (not recommended to cut)
- Aggregator include duplication between 01 and 04: ~5 lines (not recommended to cut)

**Total recommended cuts: ~35 lines (1.7%).**

### 3. Crucial updates needed
**No.** There are no contradictory duplicate explanations. All duplicated content is consistent across files. No factual conflicts were found. The duplications are of the "same thing explained in two places" variety, not "two contradictory explanations."

### 4. Recommended cuts

| Location | What to trim | Estimated savings | Priority |
|----------|-------------|-------------------|----------|
| `04_auto_derivation.md` section 2.4.8, lines 348-400 | Replace the four repeated field-pattern examples (passthrough, aliased, derived, internal constant) with a cross-reference to section 2.3.4. Keep the decision tree (lines 392-400) and pop_input asymmetry (lines 402-422). | ~25 lines | Low |
| `04_auto_derivation.md` section 2.4.3, lines 169-184 | Replace the RISC flag derivation code block with a cross-reference to section 2.3.2, retaining the "Important implication for is_inter_core" paragraph. | ~10 lines | Low |

**Total estimated savings: ~35 lines (1.7%).**

**Overall assessment:** Chapter 2 is tight. The content is well-partitioned across files with minimal redundancy. The walkthroughs, tables, and code examples all serve distinct purposes. The two recommended cuts are low-priority quality improvements (replacing repeated code blocks with cross-references), not compression necessities. No action is required.
