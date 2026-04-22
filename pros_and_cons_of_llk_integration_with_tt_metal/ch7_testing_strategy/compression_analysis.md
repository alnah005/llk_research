# Compression Analysis -- Chapter 7: Cross-Boundary Testing Strategy

**Agent:** C (Compressor) -- Pass 1
**Total lines surveyed:** 403 (index.md: 48, llk_standalone_testing.md: 122, metal_integration_testing.md: 118, testing_gaps.md: 115)

## VERDICT

Crucial updates: no

## Load-Bearing Evidence

- **index.md** -- The four "Key Observations" (lines 27-48) each preview content that is then fully developed in a dedicated sub-page. The observations themselves are concise and serve as a useful chapter summary. No structural flaw; content is load-bearing as a navigation hub.
- **llk_standalone_testing.md** -- Detailed description of the Python harness, C++ sources, hardware config, build toolchain, and performance testing. Each section introduces information not available elsewhere. The performance test file list (lines 109-113) is the only content that gets restated later in testing_gaps.md.
- **metal_integration_testing.md** -- The fake_kernels_target explanation (lines 76-104) and the "No Targeted Submodule-Update Tests" section (lines 106-114) carry unique detail (CMakeLists.txt glob behavior, fake_jit_prelude.h constants). However, the same conclusions appear in index.md and testing_gaps.md.
- **testing_gaps.md** -- Each of the five gaps is a distinct analytical claim. The gap descriptions are load-bearing. The redundancy is in the supporting evidence paragraphs, which re-cite files and mechanisms already explained in the other two sub-pages.

## Redundancy Findings

### R1: Submodule-update / fake_kernels_target -- stated 3 times (~30 lines removable)
- **index.md lines 39-43**: Observation 3 describes missing CI gate and fake_kernels_target catching only compile errors.
- **metal_integration_testing.md lines 76-114**: Full sections on fake_kernels_target and no targeted submodule-update tests.
- **testing_gaps.md lines 44-62 (Gap 3)**: Restates that there is no CI gate, re-explains fake_kernels_target with the same hardcoded-constants detail, re-cites `fake_jit_prelude.h` and `MATH_FIDELITY = 255`.

The canonical location for the fake_kernels_target mechanism should be metal_integration_testing.md. Gap 3 in testing_gaps.md should reference that section rather than re-explaining it. Index.md observation 3 can remain as a one-sentence summary but should drop the fake_kernels_target detail. Estimated saving: ~30 lines across the three files.

### R2: SFPU operations invisible to LLK -- stated 2 times (~10 lines removable)
- **index.md lines 46-48**: Observation 4 cites 158 SFPU headers, the directory path, and the fact that LLK tests never compile them.
- **testing_gaps.md lines 65-85 (Gap 4)**: Repeats the 158 count, the same directory path, the same `ckernel_sfpu_silu.h` / `ckernel_sfpu_cumsum.h` examples, and the same conclusion.

Index.md observation 4 could be trimmed to one sentence pointing to Gap 4.

### R3: "LLK tests call Layer 1 directly" -- stated 3 times (~12 lines removable)
- **index.md lines 27-31**: Observation 1 cites `eltwise_binary_test.cpp` and the specific `_llk_unpack_AB_init_`, `_llk_unpack_hw_configure_` functions.
- **llk_standalone_testing.md lines 51-64**: Code block and explanation of the same file and functions.
- **testing_gaps.md lines 11-14 (Gap 1)**: Again cites `eltwise_binary_test.cpp` and the same two function names.

The detailed code example belongs in llk_standalone_testing.md. The other two files should cross-reference rather than re-list the function names.

### R4: "Metal tests call Layer 3 / never isolate Layer 1-to-2 boundary" -- stated 3 times (~8 lines removable)
- **index.md lines 33-37**: Observation 2 names `eltwise_copy_block.cpp`, `copy_tile` / `pack_tile`, and the TTNN eltwise test directory.
- **metal_integration_testing.md lines 5-9**: Overview paragraph makes the same point.
- **testing_gaps.md lines 11-25 (Gap 1)**: Restates that Metal wrappers call the same `_llk_*` functions with their own conventions.

### R5: Performance test file list -- stated 2 times (~8 lines removable)
- **llk_standalone_testing.md lines 109-113**: Lists `perf_matmul.py`, `perf_eltwise_binary_fpu.py`, `perf_eltwise_unary_sfpu.py`, `perf_reduce.py`, `perf_unpack_tilize.py` with links.
- **testing_gaps.md lines 89-96**: Lists `perf_matmul.py`, `perf_eltwise_binary_fpu.py`, `perf_reduce.py` with the same links.

### R6: `test_zzz_eltwise_binary.py` citation -- stated 2 times (~4 lines removable)
- **llk_standalone_testing.md lines 36-38**: Cites the file with a description of what it sweeps.
- **testing_gaps.md lines 30-35**: Cites the same file with the same parameter list (data formats, tile dimensions, broadcast types, math fidelity).

## Estimated Savings

| Source | Current lines | Removable lines | Reduction |
|--------|--------------|-----------------|-----------|
| index.md | 48 | ~12 | 25% |
| llk_standalone_testing.md | 122 | ~4 | 3% |
| metal_integration_testing.md | 118 | ~6 | 5% |
| testing_gaps.md | 115 | ~50 | 43% |
| **Total** | **403** | **~72** | **18%** |

testing_gaps.md bears the heaviest redundancy because each gap re-explains mechanisms already covered in the other two sub-pages.

## MINOR Suggestions

1. **testing_gaps.md, all five gaps**: Replace re-explanations of mechanisms (fake_kernels_target, SFPU directory, perf file list) with cross-references like "As described in [metal_integration_testing.md](./metal_integration_testing.md#the-fake_kernels_target-compile-time-check), ..." and keep only the novel analytical claim in each gap.
2. **index.md Key Observations**: Trim each observation to one sentence plus a link to the relevant sub-page. Currently observations 1, 3, and 4 contain detail that is fully expanded in the sub-pages.
3. **llk_standalone_testing.md line 117-118**: The sentence "The results validate LLK in isolation but are never compared against Metal's end-to-end performance numbers" is restated almost verbatim in testing_gaps.md Gap 5. Keep it in one place.
4. **metal_integration_testing.md lines 7-9**: The sentence "Unlike LLK's standalone tests, Metal's test suite never isolates the Layer 1-to-Layer 2 boundary" appears as a near-duplicate of index.md observation 2's final clause. One location suffices.
