# 2.3 The C++ Kernel Header

## What you will learn

- The five-layer structure of `op.hpp`: namespace, outer struct, CT arg structs, Op class, preprocessor guards.
- The four CT arg struct types and their RISC processor mapping.
- Typed fields (`CB`, `Semaphore`, `PerCore`, `Flag`) and plain fields (`uint32_t`, `bool`).
- The `A::` template pattern for zero-overhead compile-time argument binding.
- The `Op` lifecycle: `init()`, `operator()()`, `teardown()`.
- Preprocessor guards (`COMPILE_FOR_*`) and role flags (`is_active`, `is_sender`, `is_receiver`).
- The `cpp_parser.py` regex patterns that extract metadata from the header.
- Detailed walkthroughs of the RMSNorm and Mcast kernel headers.

---

## 2.3.1 The five-layer structure

Every `op.hpp` follows a layered structure:

```cpp
// Layer 1: Namespace
namespace blaze {

// Layer 2: Outer struct (the op's type identity)
struct RMSNorm {

    // Layer 3: CT arg structs (one per RISC group)
    template <typename A>
    struct ReaderCTArgs {
        static constexpr CB input = A::input;
        static constexpr uint32_t num_tiles = A::num_tiles;
        // ...
    };

    template <typename A>
    struct ComputeCTArgs {
        static constexpr CB input = A::input;
        static constexpr uint32_t epsilon = A::epsilon;
        // ...
    };

    // Layer 4: Op class template (lifecycle and execution)
    template <typename A>
    struct Op {
        void init() { /* NCRISC init logic */ }
        void operator()() { /* main execution per RISC */ }
        void teardown() { /* cleanup */ }
    };
};

// Layer 5: Preprocessor guards wrap per-RISC code within Op
//   #ifdef COMPILE_FOR_NCRISC ... #endif

} // namespace blaze
```

| Layer | Purpose | Parser relevance |
|-------|---------|-----------------|
| 1. Namespace | Groups all ops under `blaze::` | Not parsed; assumed `blaze` |
| 2. Outer struct | Provides the C++ type name (e.g., `RMSNorm`, `Mcast`) | Parsed by `_OUTER_STRUCT_RE`; becomes `cls.op_class` |
| 3. CT arg structs | Declare compile-time arguments per RISC group | Parsed by `_INNER_STRUCT_RE`; fields become `ParsedCTArg` entries |
| 4. Op class | Contains the executable lifecycle methods | Parsed for `init()`/`teardown()` detection and empty-body analysis |
| 5. Preprocessor guards | Compile different code for each RISC processor | Not parsed by `cpp_parser.py`; used by the C++ compiler |

### Why `static constexpr`?

Every field in a CT arg struct is `static constexpr`. This means:
- **`static`**: The field belongs to the struct type, not to any instance. CT arg structs are never instantiated.
- **`constexpr`**: The value is computed at compile time. The C++ compiler replaces field references with literal values, producing optimal machine code with no runtime overhead.

The `template <typename A>` parameter carries the concrete CT arg values from the Python-generated codegen into the C++ struct. When the compiler instantiates `ReaderCTArgs<ct_args::rmsnorm>`, every `A::field` is replaced with the actual compile-time constant.

---

## 2.3.2 CT arg structs and RISC mapping

Four CT arg struct names are recognized by the parser. Each maps to one or more RISC processors:

| Struct name | RISC target | Python registration method | Use case |
|-------------|-------------|---------------------------|----------|
| `CoreCTArgs` | BRISC + NCRISC + TRISC | `f.unified_ct_args()` / `f.per_core_unified_ct_args()` | Shared args and flags visible to all processors |
| `ReaderCTArgs` | NCRISC (DM0) | `f.ncrisc_ct_args()` | Reader-specific: CB IDs for input, tile counts |
| `WriterCTArgs` | BRISC (DM1) | `f.brisc_ct_args()` | Writer-specific: output addresses, NoC params |
| `ComputeCTArgs` | TRISC | `f.trisc_ct_args()` | Compute-specific: math params, accumulation flags |

The mapping is defined in `cpp_parser.py`:

```python
_STRUCT_TO_RISC = {
    "CoreCTArgs": Risc.BRISC | Risc.NCRISC | Risc.TRISC,  # all-RISC
    "ComputeCTArgs": Risc.TRISC,
    "ReaderCTArgs": Risc.NCRISC,
    "WriterCTArgs": Risc.BRISC,
}
```

### How RISC assignments affect execution

The RISC assignment determines which processor receives each CT arg value at runtime:

- **NCRISC (DM0)** handles data movement from DRAM/L1 into local circular buffers. Fields in `ReaderCTArgs` tell it which CBs to fill, how many tiles to read, and what addresses to use.
- **BRISC (DM1)** handles data movement from local circular buffers to DRAM/L1 or to other cores via NoC. Fields in `WriterCTArgs` specify output CBs, destination addresses, and multicast parameters.
- **TRISC** runs the math compute engine. Fields in `ComputeCTArgs` control compute behavior: which CBs to consume, mathematical precision, and accumulation modes.
- **CoreCTArgs** fields are available to all three processors. This is used when the same value (e.g., `num_tiles`) is needed by multiple RISCs, or when the op uses preprocessor guards to route different logic to different processors within a single struct.

### RISC flag derivation for `is_inter_core`

The parser tracks which structs are present and derives RISC presence flags:

```python
# In parse_op_hpp():
if struct_name == "ComputeCTArgs":
    result.has_trisc = True
elif struct_name == "ReaderCTArgs":
    result.has_ncrisc = True
elif struct_name == "WriterCTArgs":
    result.has_brisc = True
elif struct_name == "CoreCTArgs":
    result.has_brisc = True
    result.has_ncrisc = True
    result.has_trisc = True
```

The `is_inter_core` flag is set when both `has_brisc` and `has_ncrisc` are True, indicating the op coordinates between two data-movement processors. Note that `CoreCTArgs` alone sets all three flags, so an op with only `CoreCTArgs` (like Copy) will have `is_inter_core = True` by auto-derivation, even if it does not coordinate between cores in the traditional sense. See [Section 2.4.6](./04_auto_derivation.md) for override options.

---

## 2.3.3 Typed fields

The parser classifies fields in CT arg structs based on their C++ type. Six categories exist:

| C++ type | Parser source | `_resolve_ct_arg_kind()` result | Semantics |
|----------|-------------|--------------------------------|-----------|
| `CB` | `"cb"` | `CB(port)` | Circular buffer ID; linked to a port descriptor |
| `Semaphore` | `"sem"` | `Sem(protocol)` | Semaphore address; must match a `SemProtocol` name |
| `PerCore` | `"per_core"` | `PerCore()` | Value that varies per Tensix core |
| `Flag` | `"flag"` | `None` (excluded from CT args) | Boolean flag for pop control or core activation |
| `uint32_t` | `"derived"` (if `A::` refs) or skipped | `Derived()` | Computed/forwarded unsigned integer |
| `bool` | `"derived"` (if `A::` refs) or skipped | `Derived()` | Computed/forwarded boolean |

### The type aliases in `ct_types.h`

`CB`, `Semaphore`, `PerCore`, and `Flag` are all type aliases:

```cpp
// blaze/ct_types.h
using CB = uint32_t;
using Semaphore = uint32_t;
using PerCore = uint32_t;
using Flag = bool;
```

At the C++ level, they compile identically to `uint32_t` or `bool`. Their sole purpose is to carry semantic information for the Python parser. This is a zero-cost abstraction -- the type alias tells the parser what the field *means*, while the compiler sees only the underlying primitive type.

### The `_TYPE_TO_SOURCE` mapping

The parser uses this mapping to convert C++ type names to source classifications:

```python
_TYPE_TO_SOURCE = {
    "CB": "cb",
    "Semaphore": "sem",
    "PerCore": "per_core",
    "Flag": "flag",
}
```

Fields with types not in this mapping (i.e., `uint32_t` and `bool`) fall through to the plain-field regex (`_PLAIN_FIELD_RE`), where the RHS expression determines their classification.

### Flag field behavior

Flag fields receive special treatment:

1. The parser classifies them as `source = "flag"`.
2. `_resolve_ct_arg_kind()` returns `None` for flag sources.
3. The caller skips `None` results, so flag fields are excluded from `cls.ct_args`.
4. After all structs are processed, fields with names starting with `pop_` (e.g., `pop_input`, `pop_in0`) are collected into `parsed.pop_flags` regardless of their type (`Flag`, `bool`, etc.), which sets `cls.supports_pop_control = True`.
5. In `emit()`, flags are registered separately through `f.flag()` or `f.per_core_unified_ct_args()`, not through the standard CT arg registration methods.

---

## 2.3.4 The `A::` template pattern

The `template <typename A>` pattern is the mechanism that connects Python-registered CT arg values to C++ compile-time constants.

### End-to-end flow

```
Python emit():
    f.ncrisc_ct_args([("rmsnorm___0.input", cb_handle)])
         |
         v
Codegen produces:
    namespace ct_args {
        struct rmsnorm {
            static constexpr uint32_t input = 3;  // CB ID resolved from cb_handle
        };
    }
         |
         v
C++ instantiation:
    ReaderCTArgs<ct_args::rmsnorm>::input  -->  3  (compile-time constant)
```

The `A::` namespace in the header acts as an indirection layer. At template instantiation time, `A` is replaced by the concrete `ct_args::rmsnorm` struct, and every `A::field` becomes a literal constant.

### Four field patterns

**1. Passthrough** -- the field name matches the CT arg name:

```cpp
static constexpr uint32_t num_tiles = A::num_tiles;
// Schema arg: "num_tiles"
```

**2. Aliased** -- the local name differs from the schema arg:

```cpp
static constexpr uint32_t buffer_base = A::ncrisc_buffer_base;
// Schema arg: "ncrisc_buffer_base" (the A:: token, not the LHS)
```

**3. Derived expression** -- computed from multiple schema args:

```cpp
static constexpr uint32_t out_tiles = A::n * A::m;
// Schema args: "n" and "m" (both extracted from the RHS)
```

**4. Internal constant** -- no A:: dependency:

```cpp
static constexpr uint32_t MS_SEM_THRESHOLD = 1;
// No schema arg (skipped by parser)
```

The parser determines the pattern by examining the RHS of each `static constexpr` declaration. If the RHS contains `A::` tokens, those tokens name the schema args. If not, the field is an internal constant.

---

## 2.3.5 The `Op` lifecycle: `init()`, `operator()()`, `teardown()`

The `Op` class template contains the executable code. It supports three lifecycle methods:

```cpp
template <typename A>
struct Op {
    void init() {
        // Called once before the tile loop.
        // Typical: set up NoC configuration, prefetch semaphore addresses.
    }

    void operator()() {
        // Called once per tile batch.
        // This is the main execution body.
        // Reads from input CBs, computes, writes to output CBs.
    }

    void teardown() {
        // Called once after all tile batches complete.
        // Typical: write barriers, final synchronization.
    }
};
```

| Method | When called | Typical use | Required? |
|--------|------------|-------------|-----------|
| `init()` | Once, before the tile loop | NoC setup, semaphore init, prefetching | No; omit or leave empty |
| `operator()()` | Once per tile batch | Main tile processing loop | Yes (always present) |
| `teardown()` | Once, after all tile batches | Write barriers, final sync, cleanup | No; omit or leave empty |

### Lifecycle detection by the parser

The parser detects lifecycle methods using regex:

```python
_INIT_RE = re.compile(r"void\s+init\s*\(\s*\)")
_TEARDOWN_RE = re.compile(r"void\s+teardown\s*\(\s*\)")
```

- `has_init_teardown` is `True` only when **both** `init()` and `teardown()` are found. If only one is present, the flag is `False`.
- `init_is_empty` and `teardown_is_empty` are determined by stripping comments from the method body and checking if only whitespace remains.

### Empty body detection

The `_method_body_is_empty()` function:

1. Matches the method signature (`void init()` or `void teardown()`).
2. Finds the opening `{` after the signature.
3. Extracts the body between `{` and the matching `}` using brace-depth tracking.
4. Strips all line comments (`//...`) and block comments (`/* ... */`).
5. Returns `True` if the stripped body is all whitespace.

```python
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

def _method_body_is_empty(text, regex):
    m = regex.search(text)
    if not m:
        return False
    # ... locate opening brace, extract body ...
    body = _LINE_COMMENT_RE.sub("", body)
    body = _BLOCK_COMMENT_RE.sub("", body)
    return not body.strip()
```

This means all of the following detect as empty:
- `void teardown() {}`
- `void teardown() { /* nothing */ }`
- `void teardown() { // TODO: implement cleanup }`

And this detects as non-empty:
- `void teardown() { noc_async_write_barrier(); }`

The codegen uses these flags to skip emitting no-op lifecycle calls, which avoids unnecessary function call overhead on the device.

### Interaction between init/operator()/teardown

A typical pattern in RMSNorm (simplified from actual source):

```cpp
void init() {
    // NCRISC: push tensor-backed inputs so TRISC can wait on them
#if defined(COMPILE_FOR_NCRISC)
    if constexpr (core_cta::is_active) {
        if constexpr (dm0_cta::input_is_tensor_backed) {
            cb_reserve_back(dm0_cta::input, dm0_cta::num_tiles);
            cb_push_back(dm0_cta::input, dm0_cta::num_tiles);
        }
        cb_reserve_back(dm0_cta::gamma, dm0_cta::num_tiles);
        cb_push_back(dm0_cta::gamma, dm0_cta::num_tiles);
    }
#endif
}

void operator()() {
    // TRISC only: compute RMS normalization
#if defined(COMPILE_FOR_TRISC)
    if constexpr (core_cta::is_active) {
        cb_wait_front(compute_cta::input, num_tiles);
        cb_wait_front(compute_cta::gamma, num_tiles);
        // ... RMS computation (square, reduce, rsqrt, multiply) ...
        cb_push_back(compute_cta::output, num_tiles);
    }
#endif
}

void teardown() {}
```

The `init()` notifies TRISC that tensor-backed input data is ready by performing `cb_reserve_back`/`cb_push_back` (this "pushes" the CB so TRISC's `cb_wait_front` unblocks). `operator()()` contains ONLY TRISC compute code -- there is no NCRISC section. `teardown()` is empty. The preprocessor guards ensure each RISC only compiles its relevant code.

---

## 2.3.6 Preprocessor guards and role flags

### Preprocessor guards

The same `op.hpp` file is compiled three times -- once for each RISC processor. Preprocessor guards control which code each RISC sees:

| Guard | Active for | Use case |
|-------|-----------|----------|
| `COMPILE_FOR_NCRISC` | NCRISC (DM0) | Reader logic: NoC reads, CB pushes |
| `COMPILE_FOR_BRISC` | BRISC (DM1) | Writer logic: NoC writes, CB pops |
| `COMPILE_FOR_TRISC` | TRISC | Compute logic: math operations, CB wait/pop/push |

```cpp
void operator()() {
#if defined(COMPILE_FOR_NCRISC)
    // Only NCRISC compiles this: e.g. set up sharded buffers
    if constexpr (cta::use_ncrisc) { copy_impl(); }
#elif defined(COMPILE_FOR_BRISC)
    // Only BRISC compiles this: e.g. data movement
    if constexpr (!cta::use_ncrisc) { copy_impl(); }
#elif defined(COMPILE_FOR_TRISC)
    // Only TRISC compiles this: e.g. compute normalization
    cb_wait_front(compute_cta::input, compute_cta::num_tiles);
    // ... math operations ...
    cb_push_back(compute_cta::output, compute_cta::num_tiles);
#endif
}
```

### Role flags and `if constexpr`

For ops that use `CoreCTArgs`, boolean `Flag` fields control per-core behavior at compile time. Different ops use different flags:

| Flag | Used by | Type | Purpose |
|------|---------|------|---------|
| `is_active` | RMSNorm, Copy | `Flag` (`bool`) | Whether this core participates in the op at all |
| `is_sender` | Mcast | `Flag` (`bool`) | Whether this core is the multicast sender |
| `is_receiver` | Mcast | `Flag` (`bool`) | Whether this core is a multicast receiver |
| `pop_src` | Mcast | `Flag` (`bool`) | Whether this core pops the source CB after reading |
| `pop_input` | RMSNorm | `bool` (in `ComputeCTArgs`) | Whether this core pops the input CB after consuming |

These flags are typically used with `if constexpr` for compile-time dead code elimination:

```cpp
void operator()() {
    if constexpr (core_cta::is_sender) {
        // multicast send logic (BRISC)
        cb_wait_front(dm1_cta::src, dm1_cta::src_num_pages);
        mcast_detail::send_data_mcast<...>(...);
        if constexpr (core_cta::pop_src) {
            cb_pop_front(dm1_cta::src, dm1_cta::src_num_pages);
        }
    }
    // if !is_sender: this block is eliminated at compile time
}
```

### `if constexpr` vs. preprocessor guards

| Mechanism | Resolved at | Granularity | Use case |
|-----------|------------|-------------|----------|
| `#ifdef COMPILE_FOR_*` | Preprocessor time | Per-RISC | Different logic for different RISC processors |
| `if constexpr (A::flag)` | Template instantiation time | Per-core | Different logic for different Tensix cores within the same RISC |

Preprocessor guards partition code by RISC type (reader vs. writer vs. compute). Role flags partition code by core role within a RISC type (sender vs. receiver). Both eliminate dead code at compile time with zero runtime cost.

---

## 2.3.7 The `cpp_parser.py` regex patterns

The parser uses seven pre-compiled regex patterns to extract metadata from the header:

| # | Pattern | Regex | Matches | Example |
|---|---------|-------|---------|---------|
| 1 | `_OUTER_STRUCT_RE` | `^struct\s+(\w+(?<!CTArgs))\s*\{` | Outer op struct name | `struct RMSNorm {` |
| 2 | `_INNER_STRUCT_RE` | `(?:template\s*<[^>]*>\s*)?struct\s+(\w+CTArgs)\s*\{` | CT arg struct with optional template prefix | `template <typename A> struct CoreCTArgs {` |
| 3 | `_CT_TYPED_FIELD_RE` | `static\s+constexpr\s+(CB\|Semaphore\|PerCore\|Flag)\s+(\w+)\s*=` | Typed fields (CB, Semaphore, PerCore, Flag) | `static constexpr CB input = A::input;` |
| 4 | `_PLAIN_FIELD_RE` | `static\s+constexpr\s+(?:uint32_t\|bool)\s+(\w+)\s*=\s*([^;]+);` | Plain uint32_t/bool fields with RHS | `static constexpr uint32_t num_tiles = A::num_tiles;` |
| 5 | `_A_NS_TOKEN_RE` | `\bA::(\w+)` | A:: namespace tokens in RHS expressions | `A::num_tiles`, `A::n * A::m` |
| 6 | `_INIT_RE` / `_TEARDOWN_RE` | `void\s+init\s*\(\s*\)` / `void\s+teardown\s*\(\s*\)` | Lifecycle method signatures | `void init()`, `void teardown()` |
| 7 | `_SETUP_RE` | `static\s+void\s+(setup_\w+)\s*\(\s*\)` | Static setup methods | `static void setup_src()` |

### Parsing flow

The parser processes the header in this order:

1. **Outer struct**: `_OUTER_STRUCT_RE` extracts the op's C++ type name. The negative lookbehind `(?<!CTArgs)` prevents matching inner CT arg struct names.

2. **Inner structs**: For each `_INNER_STRUCT_RE` match, the parser:
   a. Looks up the struct name in `_STRUCT_TO_RISC` to get the RISC target.
   b. Extracts the struct body using brace-depth tracking.
   c. Scans for typed fields with `_CT_TYPED_FIELD_RE`.
   d. Scans for plain fields with `_PLAIN_FIELD_RE`, using `_A_NS_TOKEN_RE` to extract A:: dependencies from the RHS.
   e. Merges fields into the accumulated args dict (`_merge_args`), OR-ing RISC flags for duplicate names.

3. **Pop flags**: After all structs are processed, any field with a name starting with `pop_` is added to `pop_flags`, regardless of its source type (`flag`, `derived`, etc.).

4. **Lifecycle**: `_INIT_RE` and `_TEARDOWN_RE` detect init/teardown presence. `_method_body_is_empty()` checks for empty bodies.

5. **Setup methods**: `_SETUP_RE` finds static methods named `setup_*`.

---

## 2.3.8 Kernel walkthrough: RMSNorm

RMSNorm's kernel header illustrates the multi-struct pattern with core, reader, and compute phases. It has three CT arg structs, not two.

### CT arg structs

```cpp
namespace blaze {

    struct RMSNorm {

        template <typename A>
        struct CoreCTArgs {
            static constexpr Flag is_active = A::is_active;
        };

        template <typename A>
        struct ReaderCTArgs {
            static constexpr CB input = A::input;
            static constexpr CB gamma = A::gamma;
            static constexpr uint32_t num_tiles = A::num_tiles;
            static constexpr bool input_is_tensor_backed = A::input_is_tensor_backed;
        };

        template <typename A>
        struct ComputeCTArgs {
            static constexpr CB input = A::input;
            static constexpr CB gamma = A::gamma;
            static constexpr CB output = A::output;
            static constexpr uint32_t num_tiles = A::num_tiles;
            static constexpr bool fp32_dest_acc_en = A::fp32_dest_acc_en;
            static constexpr bool rsqrt_fast_approx = A::rsqrt_fast_approx;
            static constexpr bool pop_input = A::pop_input;
            static constexpr uint32_t epsilon = A::epsilon;
            static constexpr uint32_t scalar = A::scalar;
        };
```

Note that the outer struct is **indented** (`    struct RMSNorm {`), not at column 0. This is important for the parser -- see below.

**Parser output for these structs:**
- `struct_name = ""` (empty -- the `_OUTER_STRUCT_RE` regex uses `^struct` which requires column 0, but RMSNorm's struct is indented; see [Section 2.4](./04_auto_derivation.md) for details)
- `has_ncrisc = True` (ReaderCTArgs present + CoreCTArgs sets all)
- `has_trisc = True` (ComputeCTArgs present + CoreCTArgs sets all)
- `has_brisc = True` (CoreCTArgs sets all three flags)
- `is_inter_core = True` (has_brisc AND has_ncrisc both True due to CoreCTArgs)

**Merged CT args:**
- `is_active`: flag, all RISCs (from CoreCTArgs) -- excluded from ct_args
- `input`: CB, `Risc.NCRISC | Risc.TRISC` (merged from ReaderCTArgs and ComputeCTArgs, plus CoreCTArgs contribution)
- `gamma`: CB, `Risc.NCRISC | Risc.TRISC` (merged from ReaderCTArgs and ComputeCTArgs)
- `output`: CB, `Risc.TRISC` (ComputeCTArgs only)
- `num_tiles`: derived, `Risc.NCRISC | Risc.TRISC`
- `input_is_tensor_backed`: derived, `Risc.NCRISC`
- `fp32_dest_acc_en`: derived, `Risc.TRISC`
- `rsqrt_fast_approx`: derived, `Risc.TRISC`
- `pop_input`: derived, all RISCs (from CoreCTArgs as flag -- excluded; but also `bool` in ComputeCTArgs as derived, so it appears in ct_args for TRISC)
- `epsilon`: derived, `Risc.TRISC`
- `scalar`: derived, `Risc.TRISC`

Note on `pop_input`: In RMSNorm, `pop_input` appears as `bool` in `ComputeCTArgs`, making it `source="derived"` for that struct. However, it is also registered as a per-core flag via `f.per_core_unified_ct_args([f.flag(...)])` in the `emit()` method. In the `CoreCTArgs`, `is_active` and `pop_input` (via the flag path) are excluded from the CT arg list because `Flag` maps to `None` in `_resolve_ct_arg_kind()`.

### Op class

The actual RMSNorm Op class (simplified for the structural walkthrough -- the real compute body is significantly more complex):

```cpp
    template <typename Args>
    class Op {
        using core_cta = CoreCTArgs<Args>;
        using dm0_cta = ReaderCTArgs<Args>;
        using compute_cta = ComputeCTArgs<Args>;

    public:
        void init() {
#if defined(COMPILE_FOR_NCRISC)
            if constexpr (core_cta::is_active) {
                if constexpr (dm0_cta::input_is_tensor_backed) {
                    cb_reserve_back(dm0_cta::input, dm0_cta::num_tiles);
                    cb_push_back(dm0_cta::input, dm0_cta::num_tiles);
                }
                cb_reserve_back(dm0_cta::gamma, dm0_cta::num_tiles);
                cb_push_back(dm0_cta::gamma, dm0_cta::num_tiles);
            }
#endif
        }

        void operator()() {
#if defined(COMPILE_FOR_TRISC)
            if constexpr (core_cta::is_active) {
                // Wait for input and gamma tiles
                cb_wait_front(compute_cta::input, compute_cta::num_tiles);
                cb_wait_front(compute_cta::gamma, compute_cta::num_tiles);
                // ... RMS normalization math (square, reduce, rsqrt, mul) ...
                if constexpr (compute_cta::pop_input) {
                    cb_pop_front(compute_cta::input, compute_cta::num_tiles);
                }
                cb_reserve_back(compute_cta::output, compute_cta::num_tiles);
                // ... pack output tiles ...
                cb_push_back(compute_cta::output, compute_cta::num_tiles);
            }
#endif
        }

        void teardown() {}
    };
};
}  // namespace blaze
```

**Lifecycle detection:**
- Both `init()` and `teardown()` are present, so `has_init_teardown = True`.
- `init()` has a non-empty body (the `cb_reserve_back`/`cb_push_back` calls for tensor-backed inputs), so `init_is_empty = False`.
- `teardown()` has an **empty** body (`{}`), so `teardown_is_empty = True`. The codegen will skip emitting the teardown call for RMSNorm.

**Structural observations:**
1. **`operator()()` has ONLY a `COMPILE_FOR_TRISC` block.** There is NO NCRISC section in `operator()()`. The NCRISC work happens entirely in `init()` (pushing tensor-backed input and gamma CBs). This is a key difference from the generic pattern.
2. CT arg struct fields are accessed through type aliases: `core_cta::is_active`, `dm0_cta::input`, `compute_cta::num_tiles`. The Op uses `class Op` (not `struct Op`) and defines aliases at the top.
3. `pop_input` is used with `if constexpr` for compile-time conditional popping.
4. The `is_active` flag from `CoreCTArgs` gates all execution with `if constexpr`.

---

## 2.3.9 Kernel walkthrough: Mcast

Mcast is an inter-core multicast op that demonstrates the complexity of the kernel header: three CT arg structs (core, reader, writer -- NO ComputeCTArgs), role flags, non-trivial lifecycle, and template methods.

### CT arg structs (simplified from actual source)

```cpp
namespace blaze {

struct Mcast {

    template <typename A>
    struct CoreCTArgs {
        static constexpr Flag is_sender = A::is_sender;
        static constexpr Flag is_receiver = A::is_receiver;
        static constexpr Flag pop_src = A::pop_src;
        static constexpr Flag init_src = A::init_src;
    };

    template <typename A>
    struct ReaderCTArgs {
        static constexpr Semaphore receiver_semaphore = A::receiver_semaphore;
        static constexpr CB dst = A::dst;
        static constexpr uint32_t dst_num_pages = A::dst_num_pages;
        static constexpr CB src = A::src;
        static constexpr uint32_t src_num_pages = A::src_num_pages;
        static constexpr bool src_is_tensor_backed = A::src_is_tensor_backed;
    };

    template <typename A>
    struct WriterCTArgs {
        static constexpr uint32_t dest_noc_start_x = A::dest_noc_start_x;
        static constexpr uint32_t dest_noc_start_y = A::dest_noc_start_y;
        static constexpr uint32_t dest_noc_end_x = A::dest_noc_end_x;
        static constexpr uint32_t dest_noc_end_y = A::dest_noc_end_y;
        static constexpr Semaphore sender_semaphore = A::sender_semaphore;
        static constexpr Semaphore receiver_semaphore = A::receiver_semaphore;
        static constexpr uint32_t data_size_bytes = A::data_size_bytes;
        static constexpr CB src = A::src;
        static constexpr uint32_t src_num_pages = A::src_num_pages;
        static constexpr CB dst = A::dst;
        static constexpr uint32_t num_cores = A::num_cores;
        static constexpr bool is_part_of_receiver_grid = A::is_part_of_receiver_grid;
        static constexpr bool linked = A::linked;
        static constexpr bool posted = A::posted;
        static constexpr bool loopback = A::loopback;
    };
```

Note: Mcast has **NO `ComputeCTArgs`** -- it is purely a data-movement op. Only `CoreCTArgs`, `ReaderCTArgs`, and `WriterCTArgs` are present.

**Parser output:**
- `struct_name = "Mcast"` (the struct is at column 0, so `_OUTER_STRUCT_RE` matches)
- `has_brisc = True` (WriterCTArgs present + CoreCTArgs sets all flags)
- `has_ncrisc = True` (ReaderCTArgs present + CoreCTArgs sets all flags)
- `has_trisc = True` (CoreCTArgs sets all three flags, even though there is no ComputeCTArgs)
- `is_inter_core = True` (has_brisc AND has_ncrisc)

**CoreCTArgs contents (actual):**
- `is_sender`: Flag -- excluded from ct_args
- `is_receiver`: Flag -- excluded from ct_args
- `pop_src`: Flag -- excluded from ct_args, but added to `pop_flags` (name starts with `pop_`)
- `init_src`: Flag -- excluded from ct_args

Note: There is NO `is_active` flag in Mcast's `CoreCTArgs` (unlike RMSNorm). The semaphores (`sender_semaphore`, `receiver_semaphore`) live in `ReaderCTArgs` and `WriterCTArgs`, not in `CoreCTArgs`.

**Merged CT args (after _merge_args):**
- `receiver_semaphore`: sem, `Risc.NCRISC | Risc.BRISC` (merged from ReaderCTArgs and WriterCTArgs, plus CoreCTArgs contribution)
- `dst`: cb, `Risc.NCRISC | Risc.BRISC` (merged from ReaderCTArgs and WriterCTArgs)
- `src`: cb, `Risc.NCRISC | Risc.BRISC` (merged from ReaderCTArgs and WriterCTArgs)
- `sender_semaphore`: sem, `Risc.BRISC` (WriterCTArgs only)
- `dest_noc_start_x`, `dest_noc_start_y`, `dest_noc_end_x`, `dest_noc_end_y`: derived, `Risc.BRISC`
- `data_size_bytes`: derived, `Risc.BRISC`
- `num_cores`: derived, `Risc.BRISC`
- `dst_num_pages`, `src_num_pages`: derived, merged from Reader and Writer
- `src_is_tensor_backed`, `is_part_of_receiver_grid`, `linked`, `posted`, `loopback`: derived, various RISCs

### Op class with lifecycle and role flags

The actual Mcast Op class (simplified for the structural walkthrough):

```cpp
    template <typename Args>
    class Op {
        using core_cta = CoreCTArgs<Args>;
        using dm0_cta = ReaderCTArgs<Args>;
        using dm1_cta = WriterCTArgs<Args>;

    public:
        void init() {
#if defined(COMPILE_FOR_NCRISC)
            if constexpr (core_cta::is_sender && dm0_cta::src_is_tensor_backed) {
                // Sender on NCRISC: set up sharded buffer for source
                unified_kernels::setup_sharded_buffer(dm0_cta::src, dm0_cta::src_num_pages);
            }
#elif defined(COMPILE_FOR_BRISC)
            if constexpr (core_cta::is_sender) {
                // Sender on BRISC: configure persistent multicast state
                // (NOC multicast address, semaphore init, etc.)
                mcast_detail::init_persistent_sender<...>(...);
            }
#endif
        }

        void operator()() {
#if defined(COMPILE_FOR_BRISC)
            if constexpr (core_cta::is_sender) {
                // Sender on BRISC: wait for source data, multicast to receivers
                cb_wait_front(dm1_cta::src, dm1_cta::src_num_pages);
                mcast_detail::send_data_mcast<...>(...);
                // Signal receivers via semaphore multicast
                mcast_detail::send_with_state<...>(
                    dm1_cta::sender_semaphore, dm1_cta::receiver_semaphore, 4);
                if constexpr (core_cta::pop_src) {
                    cb_pop_front(dm1_cta::src, dm1_cta::src_num_pages);
                }
            }
#elif defined(COMPILE_FOR_NCRISC)
            if constexpr (core_cta::is_receiver) {
                // Receiver on NCRISC: wait for semaphore, push dst CB
                noc_semaphore_wait(dm0_cta::receiver_semaphore, VALID);
                noc_semaphore_set(dm0_cta::receiver_semaphore, INVALID);
                cb_push_back(dm0_cta::dst, dm0_cta::dst_num_pages);
            }
#endif
        }

        void teardown() {
#if defined(COMPILE_FOR_BRISC)
            if constexpr (core_cta::is_sender) {
                // Sender on BRISC: clean up persistent multicast state
                mcast_detail::teardown_persistent_sender<...>(dm1_cta::sender_semaphore);
            }
#endif
        }
    };
```

Note: There is **NO `COMPILE_FOR_TRISC` block** anywhere in Mcast's Op class. Mcast is purely a data-movement op -- BRISC handles the sender multicast, NCRISC handles the receiver semaphore wait. TRISC does nothing.

### Template methods: `run_as()` and `setup_src()`

Mcast defines two additional template methods beyond the standard lifecycle:

```cpp
    template <typename A>
    void run_as() {
        // Re-execute operator() logic with different CT args (e.g., a second mcast phase)
        // Used for multi-phase multicast patterns
    }

    template <typename A>
    void setup_src() {
        // Set up sharded buffer for a new source (called between phases)
    }
```

**Important:** These are **template member methods** (`template <typename A> void setup_src()`), NOT static methods (`static void setup_src()`). The parser's `_SETUP_RE` regex matches only `static void setup_\w+()`, so it does **NOT** detect Mcast's template `setup_src()`. As a result, `parsed.setup_method` is `None` for Mcast, not `"setup_src"`.

The `run_as()` and `setup_src()` methods are extensions specific to Mcast's multi-phase usage pattern (e.g., `m.init(); m(); m.run_as<ct_args::mcast>(); m.teardown();`). They are called explicitly in the generated kernel code, not auto-detected by the parser.

**Structural observations:**

1. **Three CT arg structs (not four).** Mcast uses `CoreCTArgs` (role flags), `ReaderCTArgs` (NCRISC receiver data), and `WriterCTArgs` (BRISC sender data). There is NO `ComputeCTArgs` -- Mcast is purely data-movement.
2. **CoreCTArgs contains only Flag fields.** The four fields (`is_sender`, `is_receiver`, `pop_src`, `init_src`) are all `Flag` type and are excluded from the CT arg list. Semaphores live in `ReaderCTArgs` (`receiver_semaphore`) and `WriterCTArgs` (`sender_semaphore`, `receiver_semaphore`).
3. **Two layers of code elimination.** Preprocessor guards eliminate code by RISC type; `if constexpr` with role flags eliminates code by core role. The combination means each core compiles only the code it actually executes.
4. **Persistent sender pattern.** BRISC `init()` calls `mcast_detail::init_persistent_sender<>()` to configure the NOC multicast state once. `operator()()` then reuses that state for the actual multicast. `teardown()` calls `mcast_detail::teardown_persistent_sender<>()` for cleanup.
5. **Non-trivial lifecycle.** `init()` sets up sharded buffers (NCRISC) and persistent multicast state (BRISC). `teardown()` cleans up the persistent sender. Both have non-empty bodies.

### Mcast vs. RMSNorm comparison

| Aspect | RMSNorm | Mcast |
|--------|---------|-------|
| CT arg struct count | 3 (Core, Reader, Compute) | 3 (Core, Reader, Writer) |
| is_inter_core | True (from CoreCTArgs) | True |
| Role flags in CoreCTArgs | `is_active` (Flag) | `is_sender`, `is_receiver`, `pop_src`, `init_src` (all Flag) |
| Semaphores | None | `receiver_semaphore` (ReaderCTArgs), `sender_semaphore` + `receiver_semaphore` (WriterCTArgs) |
| PerCore fields | None | None (num_cores is a plain uint32_t in WriterCTArgs) |
| Lifecycle | init: CB push for tensor-backed inputs (NCRISC); teardown: empty | init: sharded buffer setup (NCRISC) + persistent sender (BRISC); teardown: persistent sender cleanup (BRISC) |
| Code elimination | Preprocessor guards + `if constexpr (is_active)` | Preprocessor guards + `if constexpr` role flags |
| Parser struct_name | `""` (indented struct) | `"Mcast"` (column-0 struct) |

---

## Key takeaways

1. **Five layers.** Every `op.hpp` has a namespace, outer struct, CT arg structs, Op class, and preprocessor guards.

2. **Four CT arg structs map to RISCs.** `CoreCTArgs` (all), `ReaderCTArgs` (NCRISC), `WriterCTArgs` (BRISC), `ComputeCTArgs` (TRISC). The struct name determines which processor receives the values.

3. **Typed fields carry semantic meaning.** `CB`, `Semaphore`, `PerCore`, and `Flag` are `uint32_t`/`bool` aliases that the parser uses to classify fields. Always use the correct type alias.

4. **`A::` is zero-overhead indirection.** Template instantiation replaces `A::field` with compile-time constants. No runtime cost.

5. **The Op lifecycle is `init()`, `operator()()`, `teardown()`.** The parser detects presence and empty bodies. Codegen skips empty lifecycle calls.

6. **Preprocessor guards partition by RISC. Role flags partition by core.** Both eliminate dead code at compile time.

7. **Seven regex patterns extract all metadata.** Outer struct name, inner struct names, typed fields, plain fields, A:: tokens, lifecycle methods, and setup methods.

---

**Previous:** [Section 2.2 -- The Python Class](./02_python_class.md) | **Next:** [Section 2.4 -- Auto-Derivation](./04_auto_derivation.md)
