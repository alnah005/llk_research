# Developer Friction

The template-heavy design that delivers lean runtime binaries creates real friction for
kernel developers. This section catalogs the concrete costs.

## Compile-time explosion

Each unique combination of template parameters generates a separate compilation path.
Consider the eltwise binary operation at the LLK math layer:

```cpp
// tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h

template <
    EltwiseBinaryType eltwise_binary_type,  // 5 values: ELWMUL, ELWDIV, ELWADD, ELWSUB, ELWLESS
    BroadcastType src_b_bcast_type,         // 4 values: NONE, COL, ROW, SCALAR
    DstSync Dst,                            // 2 values: SyncHalf, SyncFull
    bool is_fp32_dest_acc_en,               // 2 values
    MathFidelity math_fidelity,             // 4 values: LoFi, HiFi2, HiFi3, HiFi4
    EltwiseBinaryReuseDestType binary_reuse_dest>  // 3 values: NONE, DEST_TO_SRCA, DEST_TO_SRCB
inline void _llk_math_eltwise_binary_(...)
```

The theoretical combinatorial space is 5 x 4 x 2 x 2 x 4 x 3 = 960 instantiations. Not
all combinations are valid (e.g., `HiFi4` only works with `ELWMUL`), but the compiler
does not know that until it enters the function body and hits a `LLK_ASSERT`. Each
instantiation the compiler actually encounters triggers a full template expansion,
including all nested template calls to `eltwise_binary_configure_addrmod`,
`eltwise_binary_configure_mop_standard`, and `ckernel_template`.

For SFPU operations, the pattern is similar. The `_calculate_exponential_` function:

```cpp
template <bool APPROXIMATION_MODE, bool SCALE_EN, int ITERATIONS,
          bool FAST_APPROX, bool SKIP_POSITIVE_CHECK, bool CLAMP_NEGATIVE = true>
void _calculate_exponential_(const std::uint16_t exp_base_scale_factor)
```

has 2 x 2 x N x 2 x 2 x 2 = 32N possible instantiations (where N is the number of
distinct `ITERATIONS` values used). Each instantiation may itself call further templated
functions like `_calculate_exponential_piecewise_<APPROXIMATION_MODE, SCALE_EN,
SKIP_POSITIVE_CHECK>` and `_sfpu_reciprocal_<2>`, creating a tree of template
expansions.

The `call_sfpu_operation` test helper in `tests/helpers/include/sfpu_operations.h`
demonstrates the practical impact. It carries six template parameters of its own:

```cpp
template <bool APPROX_MODE, bool is_fp32_dest_acc_en, int ITERATIONS,
          bool FAST_MODE = false, bool STABLE_SORT = false,
          bool CLAMP_NEGATIVE = false>
void call_sfpu_operation(SfpuType operation, ...)
```

Inside, a `switch` statement dispatches to dozens of SFPU functions, each templated on
subsets of those parameters. A single instantiation of `call_sfpu_operation` pulls in
template expansions for `_calculate_abs_`, `_calculate_reciprocal_`, `_calculate_exp2_`,
`_calculate_elu_`, and many more -- each with their own nested template trees.

In practice, the JIT pipeline compiles only the instantiations actually needed for a given
program, which limits the blast radius. But during development -- when running test suites
or iterating on a new kernel -- compile times are a real bottleneck.

## Error message quality

When a template instantiation fails, the error propagates through multiple layers of the
API. Consider what happens if a kernel author passes an invalid `BroadcastType` to
`add_tiles`:

1. `add_tiles_init` calls `binary_tiles_init<true, ELWADD>` (compute API layer)
2. Which calls `llk_math_eltwise_binary_init_with_operands<ELWADD, NONE, MATH_FIDELITY>` (LLK API layer)
3. Which calls `_llk_math_eltwise_binary_init_<ELWADD, NONE, MATH_FIDELITY, NONE>` (LLK math layer)
4. Which calls `_llk_math_eltwise_binary_standard_init_<ELWADD, NONE, MATH_FIDELITY>` (LLK math internal)
5. Which calls `eltwise_binary_configure_addrmod<ELWADD, NONE, MATH_FIDELITY>` (address modifier setup)
6. Which calls `eltwise_binary_configure_mop_standard<ELWADD, NONE, MATH_FIDELITY>` (MOP configuration)

If a type mismatch or constraint violation occurs at layer 6, the compiler error message
includes the full template instantiation chain -- all six layers, each with their expanded
type names. The actual problem (e.g., an unsupported combination) is buried under pages of
`in instantiation of function template specialization` notes.

The situation is worse for SFPU errors because the nesting goes deeper. A failure inside
`_sfpu_reciprocal_<2>` called from within `_calculate_exponential_piecewise_` called
from `_calculate_exponential_` generates an error trace that spans three files and
includes fully-expanded `sfpi::vFloat` and `sfpi::vInt` types in every frame.

LLK mitigates this somewhat with `LLK_ASSERT` macros that catch invalid combinations at
the top of functions:

```cpp
LLK_ASSERT(math_fidelity == MathFidelity::LoFi || eltwise_binary_type == ELWMUL,
           "Math fidelity larger than LoFi only works with Eltwise multiply");
```

But these are runtime assertions (or `static_assert`-like checks), not type-system
constraints. The compiler still fully instantiates the invalid combination before the
assertion fires. A `static_assert` at the function entry would produce a cleaner error,
but the current `LLK_ASSERT` macro does not always use `static_assert`, meaning some
invalid combinations compile successfully and only fail at runtime on the device.

## Discoverability: valid parameter combinations

The headers declare template parameters with enums, but nothing in the type system
constrains which combinations are valid. Looking at the `_llk_math_eltwise_binary_`
signature, a developer sees:

- `EltwiseBinaryType`: 5 enum values defined in `llk_defs.h`
- `BroadcastType`: 4 enum values
- `MathFidelity`: 4 enum values
- `EltwiseBinaryReuseDestType`: 3 enum values

Which combinations are valid? The answer is scattered across `if constexpr` branches and
`LLK_ASSERT` calls deep inside the implementation:

- `MathFidelity > LoFi` only works with `ELWMUL`
  (asserted in `_llk_math_eltwise_binary_standard_init_`)
- `ELWDIV` and `ELWLESS` are defined in the enum but not handled in
  `eltwise_binary_func<>` -- that helper only covers `ELWADD`, `ELWSUB`, and `ELWMUL`
- `EltwiseBinaryReuseDestType::DEST_TO_SRCB` with `BroadcastType::COL` has specific
  face-clearing semantics that are non-obvious from the type alone

There is no single reference that enumerates the valid combinations. A developer must
read the implementation of each function to discover what works, or rely on compile errors
and runtime assertion failures as a discovery mechanism.

The `SfpuType` enum in `llk_sfpu_types.h` lists over 100 operation types. Each one maps
to a different SFPU calculation function with its own template parameter requirements.
Some need `APPROXIMATION_MODE`, some need `is_fp32_dest_acc_en`, some need both plus
additional parameters. The only way to know which parameters a given `SfpuType` requires
is to trace through the `switch` statement in `call_sfpu_operation` or find the
corresponding `ckernel_sfpu_*.h` file and read its template signature.

## Macro-template interaction

The tt-metal compute API collapses LLK's template parameters using preprocessor macros.
`MATH_FIDELITY` and `DST_ACCUM_MODE` are injected by the JIT build system as `-D`
defines. This works, but it creates a hybrid preprocessing-plus-templates layer that
defeats standard C++ tooling.

The `MATH(...)` and `UNPACK(...)` macros add another layer:

```cpp
// tt-metal/tt_metal/hw/inc/api/compute/eltwise_binary.h

MATH((llk_math_eltwise_binary_init_with_operands<
    eltwise_binary_type, NONE, MATH_FIDELITY>(icb0, icb1, acc_to_dest)));
```

Here `MATH_FIDELITY` is a macro expanding to a `MathFidelity` enum value, used as a
template argument inside another macro (`MATH`). IDEs and language servers (clangd,
IntelliSense) struggle with this pattern because:

1. `MATH_FIDELITY` is not defined in any header -- it is injected by the build system
   at compile time via `-DMATH_FIDELITY=...`.
2. The `MATH(...)` macro conditionally compiles its body based on `#ifdef TRISC_MATH`,
   which is also a build-system-injected define.
3. Go-to-definition on `MATH_FIDELITY` finds nothing. Go-to-definition on the function
   inside `MATH(...)` may not resolve because the preprocessor guard is not satisfied in
   the IDE's analysis context.

The `TRISC_MATH` / `TRISC_UNPACK` / `TRISC_PACK` guards mean that the same source file
is compiled three times with different preprocessor contexts for the three Tensix RISC-V
cores. An IDE analyzing the file without the correct defines sees large sections of the
API as grayed-out dead code.

## The compound effect

These frictions compound. A developer trying to write a new eltwise operation must:

1. Identify the correct template parameters from enum definitions in `llk_defs.h`
2. Determine which parameter combinations are valid by reading implementation code
3. Write the kernel, using macro-injected defines they cannot inspect locally
4. Wait for JIT compilation to discover type errors
5. Parse multi-layer template error messages to find the actual problem
6. Use IDE navigation that partially works (compute API) or barely works (LLK internals)

None of these problems are insurmountable, and the existing `add_tiles_init` /
`mul_tiles_init` convenience functions in the compute API go a long way toward hiding
the complexity for common cases. But for anyone working below the compute API layer --
adding new operations, debugging fidelity issues, or writing fused kernels -- the
template machinery is an unavoidable source of friction.

---

**Next:** [Chapter 6 -- Debugging Tools](../ch6_debugging_tools/index.md)
