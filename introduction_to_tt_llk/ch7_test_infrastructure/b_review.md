# Agent B Review: Chapter 7 — Test Infrastructure — Pass 1

1. **performance_testing.md, "Fixtures" section**: The text states "Performance test functions must include **two** additional fixtures beyond `workers_tensix_coordinates`" but then lists only one fixture (`perf_report`). Cross-referencing actual perf tests (e.g., `perf_eltwise_binary_fpu.py`) and `conftest.py` confirms there is only one perf-specific fixture. The word "two" should be "one". A reader following this guidance would search fruitlessly for a second required fixture.

2. **test_architecture_overview.md, line 65**: The sentence "Each section includes `params.h`, which pulls in the auto-generated `build.h` header and defines the `RUNTIME_PARAMETERS` macro and `struct RuntimeParams`" is misleading. In the source, `params.h` includes `build.h`, `ckernel_defs.h`, `ckernel_sfpu.h`, and `tensix_types.h` -- it does not itself define `RUNTIME_PARAMETERS` or `struct RuntimeParams`. Those are generated into `build.h` by `test_config.py` (line 897: `#define RUNTIME_PARAMETERS ...`; line 538: `struct RuntimeParams {`). A developer inspecting `params.h` to find or modify the macro definition would be looking in the wrong file. The sentence should clarify that `build.h` (not `params.h`) defines these symbols.

# Agent B Review: Chapter 7 — Test Infrastructure — Pass 4

**No feedback — chapter approved.**
