# Compression Analysis -- Chapter 5: Compute Kernel API

## Summary

| File | Lines | Estimated Reducible Lines |
|------|-------|--------------------------|
| `index.md` | 45 | ~0 (lean table-of-contents) |
| `api_headers.md` | 234 | ~25 |
| `operation_to_llk_mapping.md` | 398 | ~40 |
| `init_and_tile_pattern.md` | 322 | ~15 |
| **Total** | **999** | **~80 (~8%)** |

## CRUCIAL Suggestions

Crucial updates: no

## MINOR Suggestions

### M1. `add_tiles()` code snippet duplicated verbatim across two files

`api_headers.md` lines 202-208 and `operation_to_llk_mapping.md` lines 130-137 contain the identical `add_tiles()` code block and surrounding explanation. The copy in `api_headers.md` exists to illustrate wrapper structure; the copy in `operation_to_llk_mapping.md` exists to show the LLK mapping. **Suggestion:** Keep the full snippet in `operation_to_llk_mapping.md` (where the mapping detail is the point) and replace the one in `api_headers.md` with a forward reference: "See `add_tiles()` in [Operation-to-LLK Mapping](./operation_to_llk_mapping.md#2-element-wise-binary----eltwise_binaryh) for a concrete example."

### M2. Comprehensive mapping table at end of `operation_to_llk_mapping.md` restates every per-section table

Lines 370-394 of `operation_to_llk_mapping.md` present a 24-row summary table that recapitulates every LLK mapping already given in the per-operation sections above it. Every cell is a strict subset of information already present. **Suggestion:** Either (a) remove the summary table entirely and rely on the per-section tables, or (b) keep only the summary table and trim the per-section tables to prose-only. Option (a) saves ~30 lines with no information loss.

### M3. TRISC-thread-to-number legend repeated in three files

- `index.md` line 11: "TRISC0 = unpack, TRISC1 = math, TRISC2 = pack"
- `api_headers.md` lines 88-93: explains the same mapping in prose
- `operation_to_llk_mapping.md` lines 7-9: bullet list defining TRISC0/1/2

**Suggestion:** Define the legend once in `index.md` and have the other files use a brief back-reference ("TRISC thread numbering is defined in the [chapter introduction](./index.md)").

### M4. `ALWI` macro definition shown twice in `api_headers.md`

The `#define ALWI inline __attribute__((always_inline))` line appears in the large `compute_kernel_api.h` code block (line 41) and again as a standalone snippet (line 99). **Suggestion:** Remove the standalone re-quote at line 99 and reference the earlier block instead.

### M5. CB consumer/producer semantics explained in two files

`operation_to_llk_mapping.md` line 349 ("Consumer-side operations ... run on TRISC0 because the unpacker needs to read input data. Producer-side operations ... run on TRISC2 because the packer writes output data.") and `init_and_tile_pattern.md` lines 172-174 + 219 restate the same consumer/producer rationale. **Suggestion:** Keep the explanation in `init_and_tile_pattern.md` (where the full loop context makes it most useful) and trim the one in `operation_to_llk_mapping.md` to just the table.

### M6. DST synchronization protocol and deprecated API described twice

`operation_to_llk_mapping.md` lines 357-364 maps all four sync functions and notes the deprecated `acquire_dst`/`release_dst`. `init_and_tile_pattern.md` lines 189-236 covers the same four functions with more detail and also explains the deprecated API. **Suggestion:** In `operation_to_llk_mapping.md`, keep only the LLK-call table (which is the file's purpose) and drop the prose about the deprecated API, deferring to `init_and_tile_pattern.md` for the full protocol explanation.

### M7. Verbose hedging in `api_headers.md` line 175

"Pack operations only need the `PACK()` macro and the pack LLK APIs that are already included via `common_globals.h` (which includes `llk_pack_api.h` when `TRISC_PACK` is defined)." The parenthetical restates what the code block above already shows. **Suggestion:** Shorten to "Pack operations only need the `PACK()` macro, available via `common_globals.h`."

## Load-Bearing Evidence

- **`index.md`**: The "Key Source Files" table (lines 29-44) is the only place all 12 header files are listed with their purpose. No redundancy to cut.
- **`api_headers.md`**: The per-operation header include blocks (lines 110-188) are unique content not repeated elsewhere; they show which LLK headers each operation pulls in at the include level, distinct from the runtime-call mappings in `operation_to_llk_mapping.md`.
- **`operation_to_llk_mapping.md`**: The per-function code snippets with line-number citations (e.g., `mm_init()` at line 14, `reduce_init()` at line 169) are the authoritative mapping reference and are not duplicated elsewhere.
- **`init_and_tile_pattern.md`**: The complete `eltwise_binary.cpp` kernel example (lines 27-77) and the DST double-buffer ASCII diagram (lines 226-233) are unique to this file and essential for understanding the runtime flow.

## VERDICT

No crucial changes. The chapter is well-structured with clear separation of concerns across the four files. The redundancy is minor and localized -- primarily the `add_tiles()` snippet duplication (M1) and the comprehensive summary table (M2), which together account for roughly half the reducible lines. Applying all MINOR suggestions would cut approximately 80 lines (~8%) while preserving every fact.
