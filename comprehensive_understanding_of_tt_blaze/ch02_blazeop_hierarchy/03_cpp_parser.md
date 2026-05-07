# 03 -- The C++ Parser

## Single Source of Truth Philosophy

The C++ kernel header (`kernels/op.hpp`) is the single source of truth for every MicroOp's kernel interface. The Python class provides identity, ports, and composition logic; the C++ header defines what the kernel reads, on which RISC processor, and how it manages its lifecycle. The parser (`blaze/cpp_parser.py`) is the bridge: it reads the header once at registration time and extracts exactly the metadata that the Python side needs, so that no CT arg declarations, lifecycle flags, or RISC mappings need to be maintained in two places.

Before the parser existed, every MicroOp had to manually declare its CT args in Python -- a redundant, error-prone process that inevitably diverged from the header as ops evolved. For an op like Mcast with 20+ CT arg fields across three RISC-specific structs, this meant 20+ `CompileTimeArg` entries in Python that had to stay in sync with 20+ `static constexpr` lines in C++. The parser eliminated this entire class of bugs.

## What `parse_op_hpp()` Extracts

The parser's entry point is `parse_op_hpp(path)`, which reads a `.hpp` file and returns a `ParsedKernel` dataclass:

```python
@dataclass
class ParsedKernel:
    struct_name: str = ""           # Outer struct (e.g., "Matmul")
    ct_args: list[ParsedCTArg] = field(default_factory=list)
    has_init_teardown: bool = False  # init() AND teardown() both present
    init_is_empty: bool = False     # init() body is whitespace/comments
    teardown_is_empty: bool = False # teardown() body is whitespace/comments
    has_brisc: bool = False         # BRISC active (WriterCTArgs or CoreCTArgs present)
    has_ncrisc: bool = False        # NCRISC active (ReaderCTArgs or CoreCTArgs present)
    has_trisc: bool = False         # TRISC active (ComputeCTArgs or CoreCTArgs present)
    pop_flags: list[str] = field(default_factory=list)
    setup_method: str | None = None # Static setup_*() method name
```

Each `ParsedCTArg` carries the field name, its source category, and which RISC(s) read it:

```python
@dataclass
class ParsedCTArg:
    name: str       # e.g., "src", "k_num_tiles"
    source: str     # "cb" | "sem" | "per_core" | "flag" | "derived"
    riscs: Risc     # Flag enum: which RISC(s)
```

## How the Parser Works

The parser uses a cascade of regex patterns and structural analysis. There are no external parsing libraries -- the header format is constrained enough that regex-based extraction is reliable.

### Step 1: Extract the Outer Struct Name

The pattern `_OUTER_STRUCT_RE` matches top-level struct declarations inside the `blaze` namespace, excluding inner `CTArgs` structs:

```python
_OUTER_STRUCT_RE = re.compile(r"^struct\s+(\w+(?<!CTArgs))\s*\{", re.MULTILINE)
```

For Matmul's header, this matches `struct Matmul {` and yields `struct_name = "Matmul"`.

### Step 2: Parse Inner CTArgs Structs

The parser searches for inner struct declarations matching `_INNER_STRUCT_RE`:

```python
_INNER_STRUCT_RE = re.compile(
    r"(?:template\s*<[^>]*>\s*)?struct\s+(\w+CTArgs)\s*\{"
)
```

This captures template structs like `template <typename A> struct ComputeCTArgs { ... }`. The struct name determines the RISC mapping via `_STRUCT_TO_RISC`:

| Struct Name | RISC Mapping | RISC Presence Flags Set |
|-------------|--------------|-------------------------|
| `CoreCTArgs` | `Risc.BRISC \| Risc.NCRISC \| Risc.TRISC` | `has_brisc`, `has_ncrisc`, `has_trisc` -- all three |
| `ComputeCTArgs` | `Risc.TRISC` | `has_trisc` |
| `ReaderCTArgs` | `Risc.NCRISC` | `has_ncrisc` |
| `WriterCTArgs` | `Risc.BRISC` | `has_brisc` |

This mapping is the core of the RISC assignment system. In Tenstorrent's Tensix architecture, each core has three RISC processors: NCRISC handles data movement from DRAM/NOC to L1 (reader), BRISC handles data movement from L1 to DRAM/NOC (writer), and TRISC handles compute (FPU/SFPU). By placing CT args in the appropriate struct, the kernel author declares which processor(s) need each value.

The presence of each struct also sets the corresponding `has_*` flags on the parsed result. Critically, `CoreCTArgs` sets **all three** flags (`has_brisc`, `has_ncrisc`, `has_trisc`), which means any op with `CoreCTArgs` will have `is_inter_core = True` during auto-derivation (since `is_inter_core = has_brisc and has_ncrisc`).

### Step 3: Extract CT Args from Each Struct Body

For each inner CTArgs struct, `_extract_ct_args()` processes two categories of fields.

**Pass 1: Typed fields** matched by `_CT_TYPED_FIELD_RE`:

```python
_CT_TYPED_FIELD_RE = re.compile(
    r"static\s+constexpr\s+(CB|Semaphore|PerCore|Flag)\s+(\w+)\s*="
)
```

The type name maps to a source category:

| C++ Type | Source | Example |
|----------|--------|---------|
| `CB` | `"cb"` | `static constexpr CB in0 = A::in0;` |
| `Semaphore` | `"sem"` | `static constexpr Semaphore receiver_semaphore = ...;` |
| `PerCore` | `"per_core"` | `static constexpr PerCore idx = A::idx;` |
| `Flag` | `"flag"` | `static constexpr Flag is_active = A::is_active;` |

**Pass 2: Plain fields** (`uint32_t` / `bool`) matched by `_PLAIN_FIELD_RE`:

```python
_PLAIN_FIELD_RE = re.compile(
    r"static\s+constexpr\s+(?:uint32_t|bool)\s+(\w+)\s*=\s*([^;]+);"
)
```

The parser then examines the right-hand side (RHS) to classify the field. The `_A_NS_TOKEN_RE` pattern extracts all `A::name` tokens:

```python
_A_NS_TOKEN_RE = re.compile(r"\bA::(\w+)")
```

Based on the RHS analysis, the field is classified as one of four categories:

1. **Passthrough** (`X = A::X;`) -- a single `A::` token whose name matches the LHS. This is a schema arg with `source="derived"`.
2. **Aliased** (`buffer_base = A::ncrisc_buffer_base;`) -- a single `A::` token with a different name. The schema arg is named from the RHS name (`ncrisc_buffer_base`), not the LHS alias.
3. **Expression-derived** (`out_tiles = A::n * A::m;`) -- multiple `A::` tokens. Each referenced name (`n`, `m`) becomes a `"derived"` arg. These names should already be declared as standalone fields; the parser deduplicates via RISC-flag merging.
4. **Literal constant** (`MS_SEM_THRESHOLD = 1;`) -- no `A::` tokens. This is an internal kernel constant, not a CT arg. The parser **skips** it entirely.

### Step 4: Merge Across Structs

When the same field name appears in multiple CTArgs structs (e.g., `k_num_tiles` in both `ComputeCTArgs` and `ReaderCTArgs`), the parser merges them by OR-ing the RISC flags:

```python
def _merge_args(seen, new_args):
    for arg in new_args:
        if arg.name in seen:
            seen[arg.name].riscs |= arg.riscs
        else:
            seen[arg.name] = arg
```

This produces a single `ParsedCTArg` with `riscs = Risc.TRISC | Risc.NCRISC` when a field appears in both `ComputeCTArgs` and `ReaderCTArgs`.

### Step 5: Pop Flag Detection

After all CT args are collected, the parser scans for fields whose names start with `pop_`:

```python
for arg in seen_args.values():
    if arg.name.startswith("pop_"):
        result.pop_flags.append(arg.name)
```

These become the `pop_flags` list on the `ParsedKernel` and, through auto-derivation, the `pop_flags` and `supports_pop_control = True` on the MicroOp class. Pop flags give the composition layer control over when a CB is consumed. For example, `pop_in0` in Matmul controls whether the input CB is popped after the matmul completes. When Matmul's output feeds a downstream op that also needs the same input (e.g., a residual connection), `pop_in0` is set to `False` so the input survives for the second consumer.

### Step 6: Lifecycle Detection

The parser checks for `void init()` and `void teardown()` method signatures using `_INIT_RE` and `_TEARDOWN_RE`:

```
void\s+init\s*\(\s*\)
void\s+teardown\s*\(\s*\)
```

`has_init_teardown` is set to `True` only if **both** methods are present. This flag controls whether the codegen emits `op.init()` and `op.teardown()` calls in the generated kernel.

For each method found, `_method_body_is_empty()` extracts the body between braces, strips C and C++ comments, and checks whether anything remains. This enables the codegen to skip no-op lifecycle calls. For example, Matmul's `teardown()` has an empty body (`void teardown() {}`), so `teardown_is_empty = True`, and the generated kernel omits the `mm.teardown()` call.

### Step 7: Setup Methods

The parser searches for static methods matching `_SETUP_RE`:

```
static\s+void\s+(setup_\w+)\s*\(\s*\)
```

This captures static method names like `setup_weights`. The first match becomes `ParsedKernel.setup_method`. Note that the regex requires the `static` keyword — template instance methods like Mcast's `setup_src()` (declared as `template <typename A> void setup_src()`) do **not** match and result in `setup_method = None`. The codegen uses this field, when present, to emit setup calls at the appropriate point in the kernel.

## Worked Example: Matmul Header

Consider the Matmul header (`blaze/ops/matmul/kernels/op.hpp`):

```cpp
struct Matmul {
    template <typename A>
    struct CoreCTArgs {
        static constexpr Flag is_active = A::is_active;
        static constexpr Flag pop_in0 = A::pop_in0;
        static constexpr CB in0 = A::in0;
    };
    template <typename A>
    struct ComputeCTArgs {
        static constexpr CB in1 = A::in1;
        static constexpr CB out = A::out;
        static constexpr uint32_t k_num_tiles = A::k_num_tiles;
        static constexpr uint32_t out_w_per_core = A::out_w_per_core;
    };
    template <typename A>
    struct ReaderCTArgs {
        static constexpr uint32_t k_num_tiles = A::k_num_tiles;
        static constexpr uint32_t in0_is_tensor_backed = ...;
    };
    // Op class with init(), operator()(), teardown()
};
```

The parser extracts:

| Field | Source | Struct(s) | Final RISCs |
|-------|--------|-----------|-------------|
| `is_active` | `flag` | CoreCTArgs | BRISC\|NCRISC\|TRISC |
| `pop_in0` | `flag` | CoreCTArgs | BRISC\|NCRISC\|TRISC |
| `in0` | `cb` | CoreCTArgs | BRISC\|NCRISC\|TRISC |
| `in1` | `cb` | ComputeCTArgs | TRISC |
| `out` | `cb` | ComputeCTArgs | TRISC |
| `k_num_tiles` | `derived` | ComputeCTArgs + ReaderCTArgs | TRISC\|NCRISC (merged) |
| `out_w_per_core` | `derived` | ComputeCTArgs | TRISC |
| `in0_is_tensor_backed` | `derived` | ReaderCTArgs | NCRISC |

RISC presence: `has_trisc = True`, `has_ncrisc = True`, `has_brisc = True` (all from `CoreCTArgs`).
Pop flags: `["pop_in0"]`.
Lifecycle: `has_init_teardown = True`, `teardown_is_empty = True`.

## Complete Parse Flow

Summarizing the full flow from header text to `ParsedKernel`:

```
op.hpp text
  |
  +-- _OUTER_STRUCT_RE --> struct_name
  |
  +-- _INNER_STRUCT_RE (per struct) --> struct_name, RISC mapping
  |     |
  |     +-- _extract_struct_body --> body text
  |     |
  |     +-- _extract_ct_args(body, risc)
  |           |
  |           +-- _CT_TYPED_FIELD_RE --> typed fields (CB, Sem, PerCore, Flag)
  |           +-- _PLAIN_FIELD_RE + _A_NS_TOKEN_RE --> derived fields
  |
  +-- _merge_args --> deduplicated ct_args with merged RISC flags
  |
  +-- pop_* scan --> pop_flags
  |
  +-- _INIT_RE, _TEARDOWN_RE --> has_init_teardown
  |
  +-- _method_body_is_empty --> init_is_empty, teardown_is_empty
  |
  +-- _SETUP_RE --> setup_method
```

The result is a `ParsedKernel` that `MicroOp._auto_derive_from_kernel_hpp()` converts into the class attributes (`ct_args`, `op_class`, `phase`, etc.) needed by `BlazeOp.register()`.

## Parser Limitations

The regex-based parser handles the constrained format of Blaze op headers well, but it has known limitations:

1. **Struct nesting** -- the parser uses brace-depth tracking to extract struct bodies, but deeply nested or unconventional brace placement could confuse it.
2. **Preprocessor conditionals** -- `#if defined(COMPILE_FOR_TRISC)` blocks inside CTArgs structs are not evaluated; the parser processes the raw text. This is intentional: CTArgs structs should not contain conditional compilation directives (the RISC mapping comes from the struct name, not from preprocessor guards).
3. **Multiple outer structs** -- only the first non-CTArgs struct match is used as `struct_name`. This has not been an issue because each `op.hpp` declares exactly one outer struct.
4. **No full C++ parsing** -- the parser does not understand namespaces, includes, macros, or templates beyond the constrained `template <typename A>` pattern. It relies on the header conforming to the Blaze convention. Violations produce parse errors or silently incorrect metadata.

---
<- [02 -- MicroOp and FusedOp](./02_microop_and_fusedop.md) | [Index](index.md) | [04 -- Auto-Discovery and Registration](./04_auto_discovery_and_registration.md) ->
