# Agent B Review: Chapter 1 — Pass 1

1. **File:** `03_build_system_and_component_map.md`, lines 72 and 256. **Error:** The text states that `pi05_pipeline_runner` (simulation-only) is "viable in CI without hardware on any Linux system with a C++17 compiler and pthreads." The project's `CMakeLists.txt` (lines 4-5) sets `CMAKE_CXX_STANDARD 20` with `CMAKE_CXX_STANDARD_REQUIRED ON`, meaning a C++17-only compiler will fail the build. **Fix:** Change "C++17 compiler" to "C++20 compiler" in both occurrences (lines 72 and 256).

# Agent B Review: Chapter 1 — Pass 2

Verified that the Pass 1 fix was correctly applied: `03_build_system_and_component_map.md` now contains "C++20 compiler" at both occurrences (lines 72 and 256), with zero remaining "C++17" references. This matches the source `CMakeLists.txt` (`CMAKE_CXX_STANDARD 20`).

**No feedback — chapter approved.**
