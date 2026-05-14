# 9.1 End-to-End Walkthrough: Adding a ScaledAdd Micro-Op

This section walks through every step of adding a new micro-op to TT-Blaze, using a hypothetical **ScaledAdd** operation as the running example:

```
output = alpha * input + beta * bias
```

ScaledAdd accepts two tiled input tensors (`input` and `bias`), two scalar compile-time parameters (`alpha` and `beta`), and produces one tiled output. It runs purely on TRISC (compute), with NCRISC handling sharded-buffer setup for tensor-backed inputs.

> **Note on the example kernel**: The C++ kernel shown in Step 2 uses `add_tiles` as the core compute operation for clarity. This is a *simplified placeholder* -- it demonstrates the CB wait/pop/reserve/push protocol and register management patterns correctly, but does not implement the actual alpha/beta scaling. A production implementation would use `mul_tiles_bcast_scalar` or equivalent APIs to apply the scaling factors. The Python-side `emit()` and `compose()` code, registration flow, and composition patterns are fully accurate regardless of the kernel's internal compute logic.

---

## Step 1: Create the Directory Layout

Every micro-op lives under `blaze/ops/<op_name>/` with a fixed structure:

```
blaze/ops/scaled_add/
    __init__.py          # empty or re-exports the class
    op.py                # Python MicroOp class
    kernels/
        op.hpp           # C++ kernel header (single source of truth)
```

The `kernels/op.hpp` file is the authoritative interface definition. The Python class provides identity, ports, composition logic, and optional overrides. Everything else -- CT args, phase metadata, lifecycle flags, RISC mapping -- is auto-derived from the kernel header at registration time by `MicroOp._auto_derive_from_kernel_hpp()` (source: `blaze/blaze_op.py`, lines 518-577).

The `__init__.py` file typically re-exports the class:

```python
from .op import ScaledAdd  # noqa: F401
```

This layout is mandatory. The `MicroOp._find_kernel_hpp()` method (source: `blaze/blaze_op.py`, lines 475-484) locates the kernel header by looking for `kernels/op.hpp` relative to the op module file:

```python
# From blaze/blaze_op.py, MicroOp._find_kernel_hpp():
op_dir = Path(inspect.getfile(cls)).parent
op_candidate = op_dir / "kernels" / "op.hpp"
if op_candidate.is_file():
    return str(op_candidate)
return None
```

---

## Step 2: Write the C++ Kernel Header (`kernels/op.hpp`)

The kernel header follows a rigid struct convention parsed by `blaze/cpp_parser.py`. The parser (`parse_op_hpp`) uses regex to extract:

- The outer struct name (matched by `_OUTER_STRUCT_RE`: `^struct\s+(\w+(?<!CTArgs))\s*\{`)
- Inner CT arg structs (`_INNER_STRUCT_RE`: `(?:template\s*<[^>]*>\s*)?struct\s+(\w+CTArgs)\s*\{`)
- Typed fields: `CB`, `Semaphore`, `PerCore`, `Flag` (matched by `_CT_TYPED_FIELD_RE`)
- Plain `uint32_t`/`bool` fields with `A::name` references (matched by `_PLAIN_FIELD_RE` and `_A_NS_TOKEN_RE`)
- `init()` / `teardown()` lifecycle methods
- `setup_*` static methods

The struct name maps to RISC processors via `_STRUCT_TO_RISC` (source: `blaze/cpp_parser.py`, lines 106-111):

| Struct Name | RISC |
|---|---|
| `CoreCTArgs` | BRISC \| NCRISC \| TRISC |
| `ComputeCTArgs` | TRISC |
| `ReaderCTArgs` | NCRISC |
| `WriterCTArgs` | BRISC |

The typed field names map to CT arg sources via `_TYPE_TO_SOURCE` (source: `blaze/cpp_parser.py`, lines 98-103):

| C++ Type | Source |
|---|---|
| `CB` | `"cb"` |
| `Semaphore` | `"sem"` |
| `PerCore` | `"per_core"` |
| `Flag` | `"flag"` |

Here is the complete `op.hpp` for ScaledAdd:

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// ScaledAdd: output = alpha * input + beta * bias
// Compute-only op (TRISC), with NCRISC handling setup_sharded_buffer.
//
// NOTE: This is a simplified example. The operator()() body uses add_tiles
// as a placeholder for the actual alpha/beta scaling logic. A production
// kernel would use mul_tiles_bcast_scalar or equivalent APIs to apply
// the scaling factors before the addition.

#pragma once

#include "../../../kernels/ct_types.h"

#if defined(COMPILE_FOR_TRISC)
#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#endif

namespace blaze {

struct ScaledAdd {

    // ── CT Arg Interface ─────────────────────────────────────────
    // These structs are the single source of truth. The Python
    // class auto-derives ct_args from them via cpp_parser.

    template <typename A>
    struct CoreCTArgs {
        static constexpr Flag is_active = A::is_active;
    };

    template <typename A>
    struct ReaderCTArgs {
        // NCRISC needs CB ids for setup_sharded_buffer in init().
        static constexpr CB input = A::input;
        static constexpr CB bias = A::bias;
        static constexpr uint32_t num_tiles = A::num_tiles;
        static constexpr uint32_t bias_num_tiles = A::bias_num_tiles;
        static constexpr Flag input_is_tensor_backed = A::input_is_tensor_backed;
        static constexpr Flag bias_is_tensor_backed = A::bias_is_tensor_backed;
    };

    template <typename A>
    struct ComputeCTArgs {
        static constexpr CB input = A::input;
        static constexpr CB bias = A::bias;
        static constexpr CB out = A::out;
        static constexpr uint32_t num_tiles = A::num_tiles;
        static constexpr uint32_t alpha = A::alpha;
        static constexpr uint32_t beta = A::beta;
    };

    // ── Op ───────────────────────────────────────────────────────

    template <typename Args>
    class Op {
        using core_cta = CoreCTArgs<Args>;
        using reader_cta = ReaderCTArgs<Args>;
        using compute_cta = ComputeCTArgs<Args>;

    public:
        void init() {
#if defined(COMPILE_FOR_NCRISC)
            if constexpr (!core_cta::is_active) return;
            if constexpr (reader_cta::input_is_tensor_backed) {
                unified_kernels::setup_sharded_buffer(
                    reader_cta::input, reader_cta::num_tiles);
            }
            if constexpr (reader_cta::bias_is_tensor_backed) {
                unified_kernels::setup_sharded_buffer(
                    reader_cta::bias, reader_cta::bias_num_tiles);
            }
#endif
        }

        void operator()() {
            if constexpr (!core_cta::is_active) return;

#if defined(COMPILE_FOR_TRISC)
            constexpr uint32_t num_tiles = compute_cta::num_tiles;

            // NOTE: Simplified placeholder. Uses add_tiles for clarity,
            // but does NOT implement alpha/beta scaling. A real kernel
            // would use mul_tiles_bcast_scalar for the scaling step.

            reconfig_data_format<false, true>(
                compute_cta::input, compute_cta::bias);
            pack_reconfig_data_format<true>(compute_cta::out);

            cb_wait_front(compute_cta::input, num_tiles);
            cb_wait_front(compute_cta::bias, num_tiles);

            add_tiles_init(compute_cta::input, compute_cta::bias);

            constexpr uint32_t tiles_per_iter = 8;
            for (uint32_t base = 0; base < num_tiles;
                 base += tiles_per_iter)
            {
                const uint32_t chunk =
                    (base + tiles_per_iter <= num_tiles)
                        ? tiles_per_iter
                        : (num_tiles - base);

                cb_reserve_back(compute_cta::out, chunk);
                tile_regs_acquire();

                for (uint32_t j = 0; j < chunk; j++) {
                    add_tiles(compute_cta::input, compute_cta::bias,
                              j, j, j);
                }

                tile_regs_commit();
                tile_regs_wait();

                for (uint32_t j = 0; j < chunk; j++) {
                    pack_tile(j, compute_cta::out, j);
                }

                tile_regs_release();
                cb_pop_front(compute_cta::input, chunk);
                cb_pop_front(compute_cta::bias, chunk);
                cb_push_back(compute_cta::out, chunk);
            }
#endif
        }

        void teardown() {}
    };
};

}  // namespace blaze
```

Key design points from the kernel header:

1. **`CoreCTArgs` with `Flag is_active`**: Every op that may run on a subset of cores must include this. The `Flag` type maps to source `"flag"` in `cpp_parser.py`. Cores where `is_active=0` skip execution via `if constexpr (!core_cta::is_active) return;`.

2. **`ReaderCTArgs` for NCRISC**: Contains the CB ids needed by `setup_sharded_buffer()` in `init()`. The `Flag input_is_tensor_backed` fields let the init guard skip setup for non-tensor-backed CBs.

3. **`ComputeCTArgs` for TRISC**: Contains the CB ids (`input`, `bias`, `out`), the tile count, and the scalar parameters (`alpha`, `beta`). Plain `uint32_t` fields with `A::name` references become `"derived"` source CT args.

4. **Lifecycle**: The parser detects `init()` and `teardown()` via `_INIT_RE` and `_TEARDOWN_RE`. Since `teardown()` has an empty body, `_method_body_is_empty()` (source: `blaze/cpp_parser.py`, lines 177-199) returns `True`, and `parsed.teardown_is_empty` is set, allowing codegen to skip the teardown call. The function strips C/C++ comments before checking if the body is whitespace-only.

### Anatomy of Typed Field Aliases

The typed aliases used in kernel headers are defined in `blaze/kernels/ct_types.h`:

```cpp
using CB = uint32_t;           // circular buffer ID (port name)
using Semaphore = uint32_t;    // global semaphore L1 address
using PerCore = uint32_t;      // per-core varying value
using Flag = bool;             // boolean flag
```

The `cpp_parser.py` regex `_CT_TYPED_FIELD_RE` (line 58) matches `static constexpr CB|Semaphore|PerCore|Flag` to identify each field's source type. Plain `uint32_t` and `bool` fields are parsed by `_PLAIN_FIELD_RE` (line 71). The RHS is scanned for `A::name` tokens via `_A_NS_TOKEN_RE`. If no `A::` references are found (e.g., `static constexpr uint32_t CONSTANT = 42;`), the field is an internal constant and is skipped. This correctly excludes hardware constants while capturing all emit-provided values.

### Merge of Duplicate Fields

When a field name appears in multiple CTArgs structs (e.g., `input` in both `ReaderCTArgs` and `ComputeCTArgs`), the parser's `_merge_args()` function (source: `blaze/cpp_parser.py`, line 151) unions the RISC flags:

```python
def _merge_args(seen: dict[str, ParsedCTArg], new_args: list[ParsedCTArg]) -> None:
    for arg in new_args:
        if arg.name in seen:
            seen[arg.name].riscs |= arg.riscs
        else:
            seen[arg.name] = arg
```

For ScaledAdd, `input` appears in both `ReaderCTArgs` (NCRISC) and `ComputeCTArgs` (TRISC). The merged entry has `riscs = Risc.NCRISC | Risc.TRISC`.

---

## Step 3: Define the Python MicroOp Class (`op.py`)

The Python class provides identity, port descriptors, and composition logic. Here is the complete `op.py`:

```python
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""ScaledAdd micro-op: output = alpha * input + beta * bias."""

import ttnn

from ...blaze_op import BlazeOp, Input, MicroOp, Output
from ...fused_program import CBHandle, FusedProgram


class ScaledAdd(MicroOp):
    """Element-wise scaled addition: out = alpha * input + beta * bias."""

    name: str = "scaled_add"

    # ── Port descriptors ─────────────────────────────────────────
    input: Input = Input()
    bias: Input = Input()
    out: Output = Output()

    # ── Optional overrides ───────────────────────────────────────
    # math_fidelity defaults to "HiFi4" (inherited from BlazeOp)
    # math_approx_mode defaults to False

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        """Compiler entry point: map tensor dict -> emit()."""
        cls.emit(
            f,
            input=tensors["input"],
            bias=tensors["bias"],
            prefix=user_args.get("prefix", "scaled_add"),
            cores=user_args.get("cores"),
            alpha=user_args.get("alpha", 0x3F80),  # 1.0 in bf16
            beta=user_args.get("beta", 0x3F80),
            output_tensor=output,
        )

    @staticmethod
    def emit(
        f: FusedProgram,
        input: CBHandle | ttnn.Tensor,
        bias: CBHandle | ttnn.Tensor,
        *,
        prefix: str = "scaled_add",
        cores: ttnn.CoreRangeSet | None = None,
        alpha: int = 0x3F80,
        beta: int = 0x3F80,
        output_tensor: ttnn.Tensor | None = None,
    ) -> CBHandle:
        """Emit a ScaledAdd operation.

        Args:
            f: FusedProgram context.
            input: First input -- CBHandle from upstream or ttnn.Tensor.
            bias: Second input -- CBHandle or ttnn.Tensor.
            prefix: CT arg namespace prefix.
            cores: Core grid. Defaults to input's core_ranges.
            alpha: Scale factor for input (bf16 uint32 encoding).
            beta: Scale factor for bias (bf16 uint32 encoding).
            output_tensor: If provided, output is tensor-backed.

        Returns:
            CBHandle for the output.
        """
        # ── Step 3a: Resolve inputs ──────────────────────────────
        if isinstance(input, CBHandle):
            input_handle = input
        else:
            input_handle = f.cb_from_tensor(input)

        if isinstance(bias, CBHandle):
            bias_handle = bias
        else:
            bias_handle = f.cb_from_tensor(bias)

        num_tiles = input_handle.num_pages

        # ── Step 3b: Resolve core grid ───────────────────────────
        if cores is None:
            cores = input_handle.core_ranges

        # ── Step 3c: Allocate output CB ──────────────────────────
        if output_tensor is not None:
            out_handle = f.cb_from_tensor(output_tensor)
        else:
            out_handle = f.cb_scratch(
                name=BlazeOp.cb_name(prefix, "out"),
                num_pages=num_tiles,
                core_ranges=cores,
                data_format=input_handle.data_format,
                tile=input_handle.tile_desc,
                page_size=input_handle.page_size,
            )

        # ── Step 3d: Emit per-core flags ─────────────────────────
        f.per_core_unified_ct_args([
            f.flag(f"{prefix}.is_active", cores),
        ])

        # ── Step 3e: Emit unified CT args ────────────────────────
        f.unified_ct_args([
            (f"{prefix}.input", input_handle),
            (f"{prefix}.bias", bias_handle),
            (f"{prefix}.out", out_handle),
            (f"{prefix}.num_tiles", num_tiles),
            (f"{prefix}.alpha", alpha),
            (f"{prefix}.beta", beta),
        ])

        # ── Step 3f: Emit NCRISC-only CT args ────────────────────
        f.ncrisc_ct_args([
            (f"{prefix}.bias_num_tiles", bias_handle.num_pages),
            (f"{prefix}.input_is_tensor_backed",
             int(isinstance(input, ttnn.Tensor))),
            (f"{prefix}.bias_is_tensor_backed",
             int(isinstance(bias, ttnn.Tensor))),
        ])

        # ── Step 3g: Record output in shadow graph ───────────────
        return f.output(
            "scaled_add",
            out_handle,
            core_ranges=cores,
            prefix=prefix,
            input=input_handle,
            bias=bias_handle,
        )


ScaledAdd.register()
```

### Anatomy of the Class

**Class attributes** parsed by `MicroOp.__init_subclass__()` (source: `blaze/blaze_op.py`, lines 453-466):

- `name: str = "scaled_add"` -- Unique op identifier, used in the registry, graph nodes, and codegen.
- `Input()` / `Output()` descriptors -- `__set_name__` captures the attribute name (e.g., `"input"`, `"bias"`, `"out"`). These become `TensorPort` entries in the `OpSpec` at registration.
- `math_fidelity` / `math_approx_mode` -- Inherited defaults. Override if your op needs `"LoFi"` (as `EltwiseMul` does in `blaze/ops/eltwise_mul/op.py`).
- `op_class` -- Auto-set to `cls.__name__` by `MicroOp.__init_subclass__()` (line 457-458). Overridden by the C++ struct name at registration if the header is found.

**`MicroOp.__init_subclass__` validation** (lines 453-466):

The subclass hook enforces that every concrete `MicroOp`:
1. Has a non-empty `op_class` (auto-set from `cls.__name__` if not provided)
2. Overrides `emit()` (checked via `cls.emit is BlazeOp.emit`)
3. Overrides `compose()` (checked via `cls.compose.__func__ is BlazeOp.compose.__func__`)

The actual source code for these checks:

```python
class MicroOp(BlazeOp):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if _is_abstract_op(cls):
            return
        if not cls.op_class:
            cls.op_class = cls.__name__
        if cls.emit is BlazeOp.emit:
            raise TypeError(
                f"{cls.__name__}: MicroOp must override emit()"
            )
        if cls.compose.__func__ is BlazeOp.compose.__func__:
            raise TypeError(
                f"{cls.__name__}: MicroOp must override compose() to support the Graph API path"
            )
```

Both the `emit()` override check (`cls.emit is BlazeOp.emit`) and the `compose()` override check (`cls.compose.__func__ is BlazeOp.compose.__func__`) must pass for the class to be accepted. Abstract ops (those with an empty `name`) skip all checks via the `_is_abstract_op()` guard.

**Why `@staticmethod emit()` not `@classmethod`?** The `emit()` method is a static function that takes an explicit `FusedProgram` argument. This is the pattern established by every existing micro-op (`EltwiseMul`, `Copy`, `RMSNorm`, etc.). It allows emit to be called without instantiating the class, and makes the FusedProgram dependency explicit.

---

## Step 4: Understand the `emit()` Method

The `emit()` method is the core composition building block. It defines what happens when this op is embedded in a larger fused program. Every `emit()` follows a consistent pattern:

### 4a. Resolve Inputs

Inputs can arrive as either `CBHandle` (from an upstream op in a fused pipeline) or `ttnn.Tensor` (from an external tensor). The pattern is:

```python
if isinstance(input, CBHandle):
    input_handle = input
else:
    input_handle = f.cb_from_tensor(input)
```

`f.cb_from_tensor()` (source: `blaze/fused_program.py`, line 1050) allocates a new CB backed by the tensor's L1 buffer and returns a `CBHandle`. It:
1. Checks for OverlappedView inputs (delegates to `cb_from_view`)
2. Attempts disjoint-cores format-key reuse via `_try_reuse_tensor_cb()`
3. On miss, calls `self.program.cb_from_tensor()` to allocate a new CB ID
4. Tracks the allocation via `_track_cb_allocation()` and records format metadata for future reuse

### 4b. Allocate Scratch CBs

Intermediate buffers that exist only within the fused kernel are allocated via `f.cb_scratch()` (source: `blaze/fused_program.py`, line 1540). This method:

1. Checks `scratch_mapping` for a pre-allocated backing tensor (the `all_scratch_mapped` path)
2. If not mapped, allocates a new scratch CB via `self.program.cb_scratch()`
3. Returns a `CBHandle` with `backing_tensor=None` (scratch CBs have no tensor backing until the build pass materializes them into an L1 arena)

The `balanced` parameter (default `True`) controls whether the CB is eligible for temporal compaction. Set `balanced=False` when push/pop is not balanced across cores (e.g., mcast destination CBs).

**CB naming**: `BlazeOp.cb_name(prefix, "out")` (source: `blaze/blaze_op.py`, line 229) produces `"{prefix}___out"` using the `CB_NAME_DELIMITER = "___"`. This naming is used by `cb_scratch` for the scratch mapping lookup.

### 4c. Emit CT Args

CT args are emitted via three families of methods, each targeting different RISC combinations:

| Method | Target RISCs | Use Case |
|--------|-------------|----------|
| `f.unified_ct_args(args)` | NCRISC + BRISC + TRISC | CB IDs, shared parameters |
| `f.trisc_ct_args(args)` | TRISC only | Compute-specific params |
| `f.ncrisc_ct_args(args)` | NCRISC only | Reader-specific params |
| `f.brisc_ct_args(args)` | BRISC only | Writer-specific params |

All of these wrap the underlying `self.program.*_ct_args()` call, adding:
- **CB arg position tracking** via `_track_cb_arg_positions()` -- records `(risc, arg_index, cb_id)` tuples so multi-phase reconfig can remap CB IDs
- **CT value capture** via `_capture_ct_values()` -- stores `{name: int(value)}` for shadow graph annotation

**Why pass CBHandle objects (not `int(handle)`) to CT arg methods?** The `_track_cb_arg_positions()` method records the position of CBHandle-typed args so that `_remap_cb_ids()` in the build phase can rewrite CB IDs when multi-phase reconfig or compaction changes them. Passing raw `int` values bypasses this tracking and triggers a fallback warning (see `cb_reconfig_builder.py` `_warn_untracked_remaps()`).

Per-core flags use `f.per_core_unified_ct_args()` with `f.flag()`:

```python
f.per_core_unified_ct_args([f.flag(f"{prefix}.is_active", cores)])
```

`f.flag()` (source: `blaze/fused_program.py`, line 1595) returns a `(name, {cores: 1})` tuple. The `per_core_*` methods emit a per-core compile-time arg that is 1 on the specified cores and 0 on all other cores.

### 4d. Record the Output

The `f.output()` call (source: `blaze/fused_program.py`, line 1726) serves two purposes:

1. **Creates a CBHandle** with graph metadata attached (`_node_id`, `_port_name`)
2. **Records an OpNode** in the shadow graph via `_record_op()`

The `**port_sources` kwargs (e.g., `input=input_handle, bias=bias_handle`) define graph edges from the input ports to their source CBs, enabling dependency tracking and visualization.

---

## Step 5: Understand the `compose()` Method

The `compose()` classmethod is the compiler entry point. When `BlazeCompiler` encounters a single-node graph for a registered fused-op config, it calls `config.compose_fn(f, tensors, output, merged_args)`.

The `compose()` method bridges the tensor-dict interface to the `emit()` API:

```python
@classmethod
def compose(cls, f, tensors, output, user_args):
    cls.emit(
        f,
        input=tensors["input"],
        bias=tensors["bias"],
        prefix=user_args.get("prefix", "scaled_add"),
        cores=user_args.get("cores"),
        alpha=user_args.get("alpha", 0x3F80),
        beta=user_args.get("beta", 0x3F80),
        output_tensor=output,
    )
```

The `tensors` dict maps `ExternalTensor` names to `ttnn.Tensor` objects. The keys correspond to port names declared via `Input()` descriptors. The `output` is the pre-allocated output tensor. The `user_args` dict contains values from the graph node's `kwargs` merged with any per-device overrides.

Note that `compose()` passes `output_tensor=output` to `emit()`, which causes `emit()` to use `f.cb_from_tensor(output)` instead of `f.cb_scratch()` for the output CB. This is the standard pattern for making the output readable as a tensor after execution.

Every `MicroOp` must override `compose()` -- this is enforced by `MicroOp.__init_subclass__()` (line 463). When `compose()` is overridden, `BlazeOp.register()` (line 322) auto-registers a `FusedOpConfig`:

```python
if cls.compose.__func__ is not BlazeOp.compose.__func__:
    fused_kernel = cls.kernel if issubclass(cls, FusedOp) else None
    register_fused_op(
        cls.name,
        FusedOpConfig(
            compose_fn=cls.compose,
            kernel=fused_kernel,
            compute_config=ttnn.ComputeConfigDescriptor(...),
        ),
    )
```

For `MicroOp` subclasses, `fused_kernel` is `None` because the kernel is auto-generated from the shadow graph (codegen path). The `compose_fn` is what the compiler calls to build the program.

---

## Step 6: Registration

Registration is the final line in `op.py`:

```python
ScaledAdd.register()
```

`MicroOp.register()` (source: `blaze/blaze_op.py`, lines 580-591) orchestrates a multi-step process:

### 6a. Auto-derive from kernel header

When `cls.ct_args` is empty (the common case for new ops), `_auto_derive_from_kernel_hpp()` is called (lines 518-577):

1. **Locate `op.hpp`**: `_find_kernel_hpp()` looks for `kernels/op.hpp` relative to the op module file (lines 475-484).

2. **Parse the header**: `parse_op_hpp(hpp_path)` returns a `ParsedKernel` with:
   - `struct_name`: `"ScaledAdd"`
   - `ct_args`: A list of `ParsedCTArg` objects derived from the CT arg structs
   - `has_init_teardown`: `True` (both `init()` and `teardown()` are present)
   - `init_is_empty`: `False` (init has NCRISC code)
   - `teardown_is_empty`: `True` (teardown body is empty after comment stripping)
   - `has_ncrisc`: `True` (from `ReaderCTArgs`)
   - `has_trisc`: `True` (from `ComputeCTArgs`)
   - `has_brisc`: `True` (from `CoreCTArgs`, which maps to all three RISCs)

3. **Override `op_class`**: Set to the C++ struct name `"ScaledAdd"` from the parser.

4. **Build port lookup**: Collects `Input`, `Output`, `Internal` descriptors into a `port_map` dict keyed by name.

5. **Convert CT args**: For each `ParsedCTArg`, `_resolve_ct_arg_kind()` (lines 490-515) maps the source string to a kind object:
   - `"cb"` -> `CB(port)` where `port` is looked up from `port_map`
   - `"sem"` -> `Sem(protocol)`
   - `"flag"` -> Skipped (flags are not CT args; they're per-core)
   - `"derived"` -> `Derived()`
   - `"per_core"` -> `PerCore()`

6. **Auto-derive metadata**: Sets `is_inter_core` (True when both BRISC and NCRISC are present), `has_init_teardown`, and `pop_flags`.

7. **Auto-derive kernel path**: Stores the relative path to `op.hpp`.

8. **Auto-derive PhaseInfo**: Creates a `PhaseInfo` for codegen with the lifecycle flags.

### 6b. Delegate to `BlazeOp.register()`

The parent `register()` (lines 267-338):

1. **Builds `OpSpec`**: From ports, kernel path, op_class, metadata.
2. **Calls `register_op(spec)`**: Adds to the global op registry.
3. **Registers in `_class_registry`**: `BlazeOp._class_registry["scaled_add"] = ScaledAdd`.
4. **Registers `CTArgSchema`**: Via `_register_ct_schema()`, which converts `CompileTimeArg` list to `CTArgSpec` entries.
5. **Registers `PhaseInfo`**: For codegen's phase-aware kernel generation.
6. **Registers `FusedOpConfig`**: Since `compose()` is overridden, a `FusedOpConfig` is created with:
   - `compose_fn=cls.compose`
   - `kernel=None` (MicroOps use codegen, not standalone kernel sources; line 327)
   - `compute_config` from `math_fidelity` and `math_approx_mode`

### Verifying Registration

You can verify that the op registered correctly:

```python
from blaze.cpp_parser import parse_op_hpp

parsed = parse_op_hpp("blaze/ops/scaled_add/kernels/op.hpp")

# Verify struct name extraction
assert parsed.struct_name == "ScaledAdd"

# Verify RISC mapping for merged fields
from blaze.blaze_op import Risc
for arg in parsed.ct_args:
    if arg.name == "input" and arg.source == "cb":
        # Present in both ReaderCTArgs (NCRISC) and ComputeCTArgs (TRISC)
        assert Risc.NCRISC in arg.riscs

# Verify lifecycle
assert parsed.has_init_teardown is True
assert parsed.teardown_is_empty is True
assert parsed.init_is_empty is False
```

---

## Step 7: The Graph API Path

ScaledAdd can also be used via the graph API (`blaze.fuse()` context):

```python
with blaze.fuse() as ctx:
    out = blaze.scaled_add(input_ext, bias_ext, grid=[(0,0), (1,0)])
```

This works because `BlazeOp.graph_call()` (source: `blaze/blaze_op.py`, lines 239-245) extracts input port names from `Input()` descriptors via `_get_input_names()` and creates a graph node via `_op_call()`. The compiler then builds a `BlazeGraph`, runs `CBEngine().assign(graph)` for CB assignment, and invokes `compose()` through the fused-op compilation path.

The op module must be imported so that `register()` executes. This is done in `blaze/ops/__init__.py` or wherever the op is needed. The import triggers the module-level `ScaledAdd.register()` call, which populates all registries:

```python
import blaze.ops.scaled_add.op  # noqa: F401
```

---

## Step 8: Testing

Tests for micro-ops typically live under `tests/blaze/micro-ops/`. A test for ScaledAdd would follow the pattern seen in existing tests:

```python
import pytest
import torch
import ttnn

import blaze
from blaze import ExternalTensor
from blaze.compiler import BlazeCompiler


def test_scaled_add(bh_2d_mesh_device):
    mesh_device = bh_2d_mesh_device
    grid_size = mesh_device.compute_with_storage_grid_size()

    # 1. Create input tensors
    num_tiles = 4
    tile_h, tile_w = 32, 32
    shard_shape = (tile_h, num_tiles * tile_w)
    num_cores = grid_size.x * grid_size.y
    core_grid = ttnn.num_cores_to_corerangeset(
        num_cores, grid_size, row_wise=True)

    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, shard_shape,
                       ttnn.ShardOrientation.ROW_MAJOR),
    )

    torch_input = torch.randn(
        (num_cores, *shard_shape), dtype=torch.bfloat16)
    torch_bias = torch.randn(
        (num_cores, *shard_shape), dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device, memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_bias = ttnn.from_torch(
        torch_bias, dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device, memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_output = ttnn.from_torch(
        torch.zeros_like(torch_input), dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device, memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # 2. Build graph
    with blaze.fuse() as ctx:
        input_ext = ExternalTensor("input")
        bias_ext = ExternalTensor("bias")
        out = blaze.scaled_add(input_ext, bias_ext)

    # 3. Compile
    compiler = BlazeCompiler(mesh_device)
    result = compiler.compile(
        graph=ctx.graph,
        tensors={"input": tt_input, "bias": tt_bias},
        output_tensor=tt_output,
        user_args={"alpha": 0x3F80, "beta": 0x3F80},
    )

    # 4. Execute
    output = result.run()

    # 5. Validate
    # alpha=1.0, beta=1.0 => output = input + bias
    expected = torch_input + torch_bias
    actual = ttnn.to_torch(output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    assert torch.allclose(actual, expected, atol=0.1)
```

### Alternative: Direct FusedProgram Test

For unit-level testing without the graph API, use `FusedProgram` directly:

```python
from blaze.fused_program import FusedProgram
from blaze.ops.scaled_add import ScaledAdd

f = FusedProgram(kernel=None, device=device)
ScaledAdd.emit(
    f,
    input=tt_input,
    bias=tt_bias,
    prefix="scaled_add",
    cores=core_grid,
    alpha=0x3F80,
    beta=0x3F80,
    output_tensor=tt_output,
)
compiled = f.build()
compiled.run()
```

When `kernel=None`, the `build()` method (source: `blaze/fused_program.py`, lines 1937-1962) triggers deferred codegen:

```python
if self.program._kernel is None:
    self.program.set_kernel_from_graph(self._shadow_graph, name=self._name)
```

This generates the C++ kernel source from the shadow graph's recorded op sequence.

### Debugging: CB Deadlock from Wrong num_tiles

One of the most common bugs in micro-op development is a mismatch between the number of tiles pushed into a CB and the number popped. If `cb_scratch` is called with `num_pages=1` but the kernel calls `cb_reserve_back(cb, subblock_w)` with `subblock_w > 1`, the CB can never hold enough tiles and the device hangs indefinitely.

**Diagnosis workflow**:
1. Enable L1 profiling: `BLAZE_L1_PROFILE=1 pytest ...`
2. Inspect the CB stats output for unexpected `total_size_B` values
3. Cross-reference with the kernel's `cb_reserve_back`/`cb_wait_front` counts
4. Ensure `cb_scratch` `num_pages` matches what the kernel expects

---

## Step 9: Composing ScaledAdd into a Fused Op

The real power of micro-ops is composition. ScaledAdd can be chained with other ops in a `FusedOp`:

```python
from blaze.blaze_op import FusedOp, Input, Output
from blaze.ops.scaled_add import ScaledAdd
from blaze.ops.residual_add import ResidualAdd


class ScaledAddResidual(FusedOp):
    """ScaledAdd followed by residual addition."""

    name: str = "scaled_add_residual"

    input: Input = Input()
    bias: Input = Input()
    residual: Input = Input()
    out: Output = Output()

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        # Step 1: ScaledAdd
        scaled = ScaledAdd.emit(
            f,
            input=tensors["input"],
            bias=tensors["bias"],
            prefix="scaled_add",
            cores=f.all_cores,
            alpha=user_args.get("alpha", 0x3F80),
            beta=user_args.get("beta", 0x3F80),
        )

        # Step 2: ResidualAdd
        ResidualAdd.emit(
            f,
            in0_handle=scaled,
            in1_handle=f.cb_from_tensor(tensors["residual"]),
            prefix="residual_add",
            cores=f.all_cores,
            output_tensor=output,
        )
```

In this composition:

1. `ScaledAdd.emit()` runs without `output_tensor`, so it allocates a scratch CB for its output. The scratch CB is purely L1-internal.
2. `ResidualAdd.emit()` receives the `scaled` `CBHandle` as `in0_handle`. It reads from the scratch CB populated by ScaledAdd.
3. `ResidualAdd.emit()` receives `output_tensor=output`, so its output is tensor-backed and readable after execution.

The shadow graph records both ops as `OpNode` entries. Codegen generates a single unified kernel that executes both phases sequentially. The scratch CB between them is automatically materialized into an L1 arena by the build pass (`_materialize_scratch_cbs` in `blaze/cb_reconfig_builder.py`, lines 372-475).

When `ScaledAdd.emit()` receives a `CBHandle` as `input`, the `isinstance(input, CBHandle)` check returns the handle directly instead of calling `f.cb_from_tensor()`. This means no new CB is allocated for that input -- the upstream op's scratch CB ID becomes this op's input CB ID, providing zero-copy data flow between ops.

### Prefix Discipline

Every op uses a `prefix` parameter to namespace its CT args. This prevents collisions when multiple instances of the same op appear in one fused kernel. The standard pattern from `BlazeOp.cb_name()` (source: `blaze/blaze_op.py`, lines 228-231) is:

```python
f"{prefix}{cls.CB_NAME_DELIMITER}{name}"  # e.g., "scaled_add___out"
```

The `CB_NAME_DELIMITER` is `"___"` (three underscores), while the `CHILD_PREFIX_DELIMITER` for nested op prefixes is `"__"` (two underscores).

`BlazeOp.child_prefix("san", "add")` produces `"san__add"` using the `CHILD_PREFIX_DELIMITER`, which keeps CT arg names like `"san__add.input"` and `"san__norm.input"` distinct.

The `FusedOp.__init_subclass__` (source: `blaze/blaze_op.py`, lines 424-429) validates that `compose()` is overridden:

```python
class FusedOp(BlazeOp):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if _is_abstract_op(cls):
            return
        if cls.compose.__func__ is BlazeOp.compose.__func__:
            raise TypeError(f"{cls.__name__}: FusedOp must override compose()")
```

---

## Code Review Checklist

Before submitting a new micro-op for review, verify:

### Directory Structure
- [ ] Three files: `__init__.py`, `op.py`, `kernels/op.hpp`
- [ ] `__init__.py` re-exports the op class
- [ ] Package name matches the op's `name` attribute (lowercase, underscored)

### Kernel Header (`op.hpp`)
- [ ] `#pragma once` and `#include "../../../kernels/ct_types.h"`
- [ ] Outer struct name matches `op_class` (or Python class name)
- [ ] Every CB field uses the `CB` type alias (not raw `uint32_t`)
- [ ] Every per-core flag uses `Flag` (not raw `bool`)
- [ ] CTArgs struct names follow convention: `CoreCTArgs`, `ReaderCTArgs`, `WriterCTArgs`, `ComputeCTArgs`
- [ ] All `A::field` references have matching emit-side CT args
- [ ] `init()` and `teardown()` are both present (even if `teardown()` is empty)
- [ ] `COMPILE_FOR_BRISC`/`COMPILE_FOR_NCRISC`/`COMPILE_FOR_TRISC` guards are correct
- [ ] `cb_push_back` count matches `cb_reserve_back` count
- [ ] `cb_pop_front` count matches `cb_wait_front` count
- [ ] `tile_regs_acquire()` / `tile_regs_commit()` / `tile_regs_wait()` / `tile_regs_release()` are balanced

### Python Class (`op.py`)
- [ ] Inherits from `MicroOp` (not `BlazeOp` directly)
- [ ] `name` is set and non-empty
- [ ] Both `emit()` and `compose()` are overridden
- [ ] `Input()`, `Output()`, `Internal()` descriptors match kernel CB fields
- [ ] `compose()` maps `tensors`/`user_args` to `emit()` parameters
- [ ] `emit()` passes `CBHandle` objects (not `int`) to CT arg methods
- [ ] Scratch CB `num_pages` matches kernel's `cb_reserve_back`/`cb_wait_front` tile counts
- [ ] `f.output()` is called with correct port source kwargs for shadow graph edges
- [ ] CT arg names follow the `f"{prefix}.field"` convention
- [ ] `input_is_tensor_backed` flags are set for NCRISC `setup_sharded_buffer` calls

### Testing
- [ ] Structure tests (no device): import, name, ports, metadata
- [ ] Silicon test with golden comparison (PCC >= 0.999)
- [ ] Multi-core test (if applicable)
- [ ] Edge cases: single tile, large tile counts, different dtypes
- [ ] Composition test: used as intermediate in a FusedOp chain

---

## Summary: The Nine Steps

| Step | Action | Key Source |
|------|--------|-----------|
| 1 | Create directory layout | `blaze/ops/<name>/` convention |
| 2 | Write `kernels/op.hpp` | `blaze/cpp_parser.py` constraints |
| 3 | Define `MicroOp` class with `emit()` | `blaze/blaze_op.py` `MicroOp` |
| 4 | Implement `emit()` body | `blaze/fused_program.py` API |
| 5 | Implement `compose()` | `blaze/compiler.py` entry point |
| 6 | Call `register()` | `blaze/blaze_op.py` auto-registration |
| 7 | Use via graph API | `blaze/context.py` `_op_call` |
| 8 | Write tests | `tests/blaze/micro-ops/` |
| 9 | Compose into fused ops | `FusedOp` with chained `emit()` calls |
