# Agent B Review -- Chapter 3 (Pass 1)

## Issue 1: "89 test source files" is significantly overstated

**File:** `coupling_and_synchronization.md`, near end of "Global Mutable State in LLK Tests" section  
**Claim:** "This appears across at least 89 test source files in `[tt-llk] tests/sources/`."  
**Actual:** `tt-llk/tests/sources/` contains 52 `.cpp` files (55 total entries including one subdirectory). The stated count of 89 overstates the actual number by ~70%.  
**Impact:** A reader citing this number would be materially wrong. Correct to "at least 50 test source files" or re-count and cite precisely.

## Issue 2: "204+ files" for `_llk_*` coupling surface is overstated

**File:** `coupling_and_synchronization.md`, end of "Implicit API Contracts" section  
**Claim:** "the `llk_api/` directory contains 204+ files across Wormhole and Blackhole that each directly invoke `_llk_*` functions"  
**Actual:** Grepping for `_llk_` across all `llk_api/` header files yields 183 files, not 204+. The total number of `.h` files in those directories is 371, but only 183 actually contain `_llk_` references.  
**Impact:** The number is inflated by ~11%. Not catastrophic, but a reader quoting "204+" as a verified count would be wrong. Correct to "180+" or verify the grep methodology.

## Issue 3: Compute kernels are compiled four times, not three

**File:** `coupling_and_synchronization.md`, "The `#ifdef TRISC_*` Conditional Compilation Pattern" section  
**Claim:** "Compute kernels are compiled three times -- once for each TRISC processor (unpack, math, pack)."  
**Actual:** `genfiles.cpp` line 227 explicitly states: "Generate the four TRISC source files (fourth only used on Quasar)." The code generates prologs and transformed sources for `TRISC_UNPACK`, `TRISC_MATH`, `TRISC_PACK`, and `TRISC_ISOLATE_SFPU` (lines 196-199, 216-220). The chapter quotes lines 196-198 but omits line 199 (`isolate_sfpu_prolog`).  
**Impact:** A developer implementing a build system change or adding a new TRISC-gated feature based on this chapter would miss the fourth compilation entirely. The `developer_ergonomics.md` also repeats this at line 84: "This happens three times -- once per TRISC processor" and shows only three `transform_to_legacy_syntax` calls while the actual code has four. Both locations should say "four times" and mention the Quasar-specific fourth processor.

## Issue 4: `DST_ACCUM_MODE` is a `constexpr bool`, not a preprocessor define

**File:** `developer_ergonomics.md`, "Macro-Heavy Kernel Authoring" section  
**Claim:** Grouped under "build-injected defines" alongside `ELTWISE_OP` and `PACK_RELU`, with the section heading "Macro-Heavy Kernel Authoring via build-injected defines."  
**Actual:** The chapter's own quoted code shows `constexpr bool DST_ACCUM_MODE = {};` (genfiles.cpp line 473). This is a typed C++ constant, not a `#define`. It goes through the type system, has scope, and would produce a compile error on type mismatch -- the exact opposite of the "no type safety" drawback attributed to it in the bullet list below.  
**Impact:** A reader could incorrectly conclude that `DST_ACCUM_MODE` bypasses the type system. Either remove this example from the "defines" discussion or note that it demonstrates an alternative pattern (constexpr generation) that avoids the type-safety problem of raw `#define`s.

---

**Items reviewed and found accurate (no flag needed):**

- `build.cpp` include list (lines 267-285), "TODO(pgk) this list is insane" comment, 10 base includes: all verified exact match.
- `.gitmodules` submodule declaration: verified exact match.
- `wh_hal.cpp` includes function (lines 101-120), 8 common + 2 compute entries: verified.
- `ckernel_globals.h` extern declarations: content matches (minor line range imprecision, not flagged as it wouldn't cause wrong implementation).
- `compute_kernel_api.h` TRISC ifdef pattern and MAIN macro redefinition: verified exact match.
- `genfiles.cpp` regex detection logic and `transform_to_legacy_syntax`: verified.
- SFPI toolchain three code paths (system, download, local build) and dual discovery: verified.
- `fake_kernels_target/CMakeLists.txt` TRISC_MATH=1 default: verified at line 123.
- 21 include path total arithmetic (10 + 8 + 2 + 1 SFPI): verified, though the SFPI add is at line 321, confirmed.

# Agent B Review: Chapter 3 — Disadvantages — Pass 3

**No feedback — chapter approved.**
