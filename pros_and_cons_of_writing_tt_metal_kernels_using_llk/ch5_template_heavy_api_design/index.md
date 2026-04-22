# Chapter 5: Template-Heavy API Design

LLK exposes nearly all of its kernel-authoring surface through C++ templates. Every eltwise
binary operation, every SFPU function, and every matmul variant is parameterized at compile
time over enums like `EltwiseBinaryType`, `BroadcastType`, `MathFidelity`, and boolean
flags like `is_fp32_dest_acc_en`. The result is an API where the compiler, not the
programmer, resolves control flow -- producing lean machine code at the cost of a
distinctly heavy template footprint.

This chapter examines both sides of that trade-off:

- **[Flexibility and Performance](./flexibility_and_performance.md)** -- How templates
  enable compile-time specialization, dead-code elimination, and tight integration with
  the JIT pipeline.
- **[Developer Friction](./developer_friction.md)** -- The practical costs: compile-time
  explosion, opaque error messages, discoverability problems, and macro-template
  interactions that defeat tooling.

## Scope of template usage in LLK

Template parameters appear at every layer of the LLK stack:

| Layer | Example | Typical template parameters |
|---|---|---|
| Low-level SFPU | `_calculate_exponential_<APPROXIMATION_MODE, SCALE_EN, ITERATIONS, FAST_APPROX, SKIP_POSITIVE_CHECK, CLAMP_NEGATIVE>(...)` | 6 template params |
| LLK math | `_llk_math_eltwise_binary_<EltwiseBinaryType, BroadcastType, DstSync, is_fp32_dest_acc_en, MathFidelity, EltwiseBinaryReuseDestType>(...)` | 6 template params |
| LLK API wrappers | `llk_math_eltwise_binary<EltwiseBinaryType, BroadcastType, is_fp32_dest_acc_en, MathFidelity, EltwiseBinaryReuseDestType>(...)` | 5 template params |
| Compute API (tt-metal) | `binary_tiles_init<full_init, EltwiseBinaryType>(icb0, icb1)` | 2 template params (others resolved from `#define`s) |

The deepest functions carry the most parameters. As calls propagate upward, the tt-metal
compute API collapses several of these into pre-defined macros (`MATH_FIDELITY`,
`DST_ACCUM_MODE`), simplifying the surface that kernel authors touch directly -- but not
eliminating the underlying template machinery.
