# Chapter 6 -- Agent B Factual Review (Pass 1)

## Verdict: CONDITIONAL APPROVAL -- minor findings below

Overall the three files are accurate. Code snippets, addresses, compiler flags, and linker script content closely match the actual tt-llk source. The findings listed below are low severity but should be corrected for precision.

---

### Finding 1: `.ldm_data` section ordering misrepresented in condensed snippet

**File:** `03_linker_scripts_and_memory_regions.md`, Section 6.3.4, lines ~247-253

**Error:** The condensed snippet places `.srodata .srodata.*` immediately after `.rodata .rodata.*`, before the `.init_array`/`.fini_array` blocks:

```
*(.rodata .rodata.* .gnu.linkonce.r.*)  *(.rodata1)  *(.srodata .srodata.*)
PROVIDE_HIDDEN(__init_array_start = .);
```

In the actual `tests/helpers/ld/sections.ld`, `.srodata` appears at line 106, **after** `.preinit_array`, `.init_array`, `.fini_array`, `.ctors`, `.dtors`, `.dynamic`, and `.data.rel.ro*`. The chapter's narrative also says "in order: read-only data (`.rodata`, `.srodata`), constructor/destructor arrays" which reverses the actual sequence.

**Fix:** Reorder the condensed snippet to match the actual layout, or add an explicit note that the snippet is reorganized for pedagogical clarity and does not match the literal ordering. Correct the prose to say "read-only data (`.rodata`), constructor/destructor arrays (`.init_array`, `.fini_array`), then small read-only data (`.srodata`), then initialized data (`.sdata`, `.data`)."

---

### Finding 2: `_start` attribute list incomplete for TRISC

**File:** `03_linker_scripts_and_memory_regions.md`, Section 6.3.6, lines ~449-455

**Error:** The chapter shows the `_start` signature as:

```cpp
extern "C" __attribute__((section(".init"), naked, noreturn)) std::uint32_t _start()
```

In the actual `tests/helpers/src/trisc.cpp` (line 99), the TRISC version is:

```cpp
extern "C" __attribute__((section(".init"), naked, noreturn, no_profile_instrument_function)) std::uint32_t _start()
```

The `no_profile_instrument_function` attribute is missing from the chapter. The `brisc.cpp` version (line 137) does match the chapter's snippet, so the chapter appears to show the BRISC version while implying it applies to both.

**Fix:** Either add `no_profile_instrument_function` to the example (since the context is TRISC), or note that the BRISC and TRISC signatures differ by this one attribute.

---

### Finding 3: `boot.h` location reference in Section 6.1.1 could be clearer

**File:** `01_toolchain_and_dependencies.md`, Section 6.1.1

**Error:** Not a factual error. The file correctly places `boot.h` under `tests/helpers/include/` in the table at Section 6.1.2. However, the earlier mention of `boot.h` in Section 6.1.6 (line 229) says "The `device_setup()` function in `boot.h`" without specifying the path, which could be confused with `tt_llk_blackhole/common/inc/boot.h` (which does not exist -- verified). Since no such file exists, no reader would actually be misled, so this is merely a style note.

**Fix:** No action required.

---

### Finding 4: `tmu-crt0.S` BSS clearing comment says "byte-by-byte" but uses `sw` (store word)

**File:** `03_linker_scripts_and_memory_regions.md`, Section 6.3.6, line ~405

**Error:** The chapter says: "Zeroes byte-by-byte (`addi t0, t0, 1`). The byte granularity handles non-word-aligned boundaries."

The actual code uses `sw zero, 0(t0)` (a 4-byte store) combined with `addi t0, t0, 1` (1-byte increment). This means each iteration stores 4 bytes at `t0` but advances only 1 byte, resulting in overlapping 4-byte writes. The chapter's description of "byte-by-byte" is accurate for the iteration step size but may mislead readers into thinking it stores one byte per iteration (which would require `sb` not `sw`). The source itself has a misleading comment ("Move to the next word (32-bit) address") for `addi t0, t0, 1`.

**Fix:** Clarify that the loop stores a 32-bit zero via `sw` but advances the pointer by only 1 byte per iteration, resulting in overlapping writes. This is slower than word-aligned clearing but handles arbitrary BSS boundaries.

---

### Finding 5: `/DISCARD/` section description omits some items, lists one that is also in `.ldm_data`

**File:** `03_linker_scripts_and_memory_regions.md`, Section 6.3.4, lines ~310-318

**Error:** The chapter says the `/DISCARD/` section strips "PLT/GOT entries." However, `.got.plt`, `.igot.plt`, `.got`, `.igot` also appear in the `.ldm_data` section (line 112 of `sections.ld`). The linker will place them in `.ldm_data` first; only the remaining unmatched input sections reach `/DISCARD/`. Listing "PLT/GOT entries" as discarded without this context is technically misleading -- they are only discarded if they were not already consumed by `.ldm_data`.

**Fix:** Remove "PLT/GOT entries" from the discarded list or note that `.got`/`.igot` entries are first claimed by `.ldm_data` and only residual ones are discarded.

---

### Finding 6: `.ldm_data` condensed snippet omits several section entries

**File:** `03_linker_scripts_and_memory_regions.md`, Section 6.3.4, lines ~247-263

**Error:** The condensed snippet omits these sections that exist in the actual `sections.ld`:
- `.preinit_array` (lines 83-85)
- `.data.rel.ro.local*`, `.data.rel.ro*` (line 105)
- `.sdata2 .sdata2.*` (line 108)
- `.sbss2 .sbss2.*` (line 116)
- `.dynsbss` (line 117)
- `.scommon` (line 119)
- `.dynbss` (line 121)

These omissions are acceptable for a condensed summary, but the prose then says "BSS (`.sbss`, `.bss`, `COMMON`)" without mentioning `.sbss2`, `.scommon`, `.dynsbss`, or `.dynbss`.

**Fix:** Either add a note that the condensed snippet omits minor subsections for brevity, or extend the BSS listing to include the other BSS-like subsections.

---

## Summary

No incorrect memory addresses, no wrong compiler flags, no mismatched linker script content, and no wrong function signatures were found. The SFPI version data, L1 memory map computations, L0 private memory sizes, `__instrn_buffer` address, and all cross-references are accurate. The six findings above are all minor precision issues in condensed/summarized code snippets rather than fundamental factual errors.

**APPROVED** with the caveat that findings 1 and 4 should be addressed to avoid misleading technically precise readers.

---

## Fixes Applied (2026-05-02)

- **Finding 1 fixed:** Corrected `.ldm_data` section ordering in prose -- now states "read-only data (`.rodata`), constructor/destructor arrays, small read-only data (`.srodata`), then initialized data". Added note that condensed snippet is reorganized for pedagogical clarity.
- **Finding 4 fixed:** Changed BSS clearing description from "Zeroes byte-by-byte" to accurately describe the `sw`+1-byte-increment overlapping write pattern.
- **Finding 5 fixed:** Clarified that `.got`/`.igot` entries are first claimed by `.ldm_data`; only residual ones reach `/DISCARD/`.
- **Agent C CRUCIAL-1 applied:** Replaced two preprocessor define tables in 6.2.4 with cross-reference to Ch5 Section 5.1.
- **Agent C CRUCIAL-2 applied:** Replaced per-thread define explanation with cross-reference to Ch5 Sections 5.1.2-5.1.3.
- **Agent C CRUCIAL-3 applied:** Replaced `do_crt0()` C++ reproduction with cross-reference to Ch3 Section 3.2; kept unique `tmu-crt0.S` assembly.
- **Agent C MINOR-2 applied:** Removed redundant 16-byte page size paragraph.
- **Agent C MINOR-4 applied:** Deduplicated `-nostartfiles` explanation.
- Total: 1295 → 1201 lines (7.3% reduction).

**FINAL: APPROVED**
