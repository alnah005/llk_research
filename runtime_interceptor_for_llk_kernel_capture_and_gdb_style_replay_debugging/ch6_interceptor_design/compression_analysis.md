# Ch6 Compression Analysis

## Current State

| File | Lines | Target | Strategy |
|------|------:|-------:|----------|
| 01 Interceptor Architecture | 781 | ~450 | Consolidate dispatch chain detail, trim class declarations |
| 02 Capture Triggers | 850 | ~400 | Reduce C++ code blocks, merge scenario walkthroughs |
| 03 Snapshot Schema | 1172 | ~500 | Compress FlatBuffer schema comments, trim serialization code |
| **Total** | **2803** | **~1350** | **52% reduction** |

## Compression Priority

1. **File 03** (1172 → 500): Largest file. FlatBuffer schema has verbose inline comments; serialization/deserialization code examples can be truncated to key functions.
2. **File 02** (850 → 400): Scenario walkthroughs and C++ trigger code are verbose; consolidate into compact tables.
3. **File 01** (781 → 450): Dispatch chain walkthrough and class declarations have moderate redundancy with Ch4.

## B-Review Issues to Address During Compression

- ProgramConfig capture code in File 01 shows 7/10 fields; File 03 has all 10 — reconcile
- Single-core size estimates vary (~22 KB, ~55 KB, ~95 KB) — add workload labels
- Selective mode overhead: File 01 says ~15-20 us, File 02 says ~15-50 ms — reconcile (likely ms is correct for full L1 readback, us is metadata-only)
- Clarify .ttlk (Ch4 proposed) vs .ttksnap (Ch6 proposed) relationship

## Decision

Compression deferred to post-Ch8 final pass. Files are functional at current length.
The B-review consistency issues (size estimates, overhead numbers) should be fixed
regardless of whether compression is applied.
