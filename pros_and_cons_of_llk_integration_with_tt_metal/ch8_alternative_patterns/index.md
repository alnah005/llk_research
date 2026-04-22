# Chapter 8: Alternative Integration Patterns

## Summary

Chapters 2 through 7 documented the concrete trade-offs of the current submodule-based,
header-only integration between LLK and Metal. This final chapter explores three alternative
integration models that could address the identified weaknesses, evaluates them against the
same evidence, and provides an assessment of each.

No single alternative eliminates all friction. The current model's strengths -- full inlining,
cross-module LTO, zero ABI surface -- are direct consequences of its weaknesses (include path
complexity, tight coupling, submodule synchronization pain). Each alternative shifts the
trade-off curve rather than transcending it.

## Evaluation Criteria

Each alternative is assessed against five criteria drawn from earlier chapters:

| Criterion | Source Chapter | What It Measures |
|-----------|--------------|------------------|
| Runtime performance | Ch2 (header-only benefits), Ch6 (template trade-offs) | Whether the alternative preserves inlining and LTO across the LLK/Metal boundary |
| Build system complexity | Ch3 (build system complexity) | Whether it reduces the 14+ include path explosion documented in `build.cpp:268` |
| Development velocity | Ch3 (coupling and synchronization) | Whether submodule bumps, branch conflicts, and cross-repo PRs become easier |
| Maintenance burden | Ch5 (architecture duplication) | Whether per-architecture duplication (3 copies of ~6,000+ lines each) is reduced |
| Testing coverage | Ch7 (testing gaps) | Whether the five identified testing gaps are addressed or worsened |

## Sub-pages

| File | Topic |
|------|-------|
| [`compiled_library_model.md`](./compiled_library_model.md) | Pre-compiling LLK into static libraries: trade-off analysis |
| [`interface_stabilization.md`](./interface_stabilization.md) | Formalizing the Layer 1/Layer 2 boundary with a versioned API contract |
| [`monorepo_and_code_generation.md`](./monorepo_and_code_generation.md) | Monorepo absorption and template-based code generation |

## Key Findings

1. **The compiled library model is likely unsuitable.** The Tensix RISC-V cores have severe
   resource constraints (limited instruction memory, no virtual memory). Losing inlining and
   cross-module LTO -- the primary benefits documented in Ch2 -- would impose unacceptable
   performance and binary size penalties. The current header-only approach, while painful for
   build times, is the right trade-off for the target hardware.

2. **Interface stabilization offers the best incremental path.** Introducing a versioned
   `llk_api.h` contract at the Layer 1/Layer 2 boundary would address the implicit API
   coupling documented in Ch4 (where Metal wrappers call `_llk_*` functions that are not
   formally declared as stable) without changing the compilation model. This is the lowest-risk,
   highest-value change.

3. **Monorepo absorption solves synchronization but creates new problems.** Eliminating the
   submodule would remove the friction documented in Ch3 (submodule update conflicts, version
   drift) but would make LLK's independent testing infrastructure (Ch7) harder to maintain
   and would increase Metal's repository size.

4. **Code generation has the highest potential payoff but also the highest cost.** Reducing
   the per-architecture duplication documented in Ch5 (three copies of llk_lib with ~4,200-7,600
   lines each) through templated code generation would be transformative, but the upfront
   investment and ongoing tooling maintenance make it a long-term bet.
