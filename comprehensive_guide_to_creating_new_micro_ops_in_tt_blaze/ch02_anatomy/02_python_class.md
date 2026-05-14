# 2.2 The Python Class

## What you will learn

- How `MicroOp.__init_subclass__` enforces the class contract at import time.
- The port descriptor system (`Input`, `Output`, `Internal`) and the `__set_name__` protocol.
- Class configuration fields (`math_fidelity`, `math_approx_mode`, `op_class`, `fp32_dest_acc_en`).
- The `compose()` method signature and three walkthroughs (RMSNorm, Copy, Matmul).
- The `emit()` overview: signature pattern, CB allocation, CT arg registration.
- Detailed structural walkthroughs of RMSNorm and Copy as Python classes.
- The RMSNorm vs. Copy contrast table.

---

## 2.2.1 `MicroOp` subclassing and `__init_subclass__`

Every micro-op is a subclass of `MicroOp` (which inherits from `BlazeOp`). At class definition time -- before any instance is created -- Python's `__init_subclass__` hook runs and enforces three rules:

```python
# blaze/blaze_op.py -- MicroOp.__init_subclass__()
def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    if _is_abstract_op(cls):
        return                          # skip enforcement for intermediate base classes
    if not cls.op_class:
        cls.op_class = cls.__name__     # default: Python class name
    if cls.emit is BlazeOp.emit:
        raise TypeError(f"{cls.__name__}: MicroOp must override emit()")
    if cls.compose.__func__ is BlazeOp.compose.__func__:
        raise TypeError(f"{cls.__name__}: MicroOp must override compose() to support the Graph API path")
```

| Check | Rule | Failure |
|-------|------|---------|
| `cls.op_class` | If not already set, default to `cls.__name__` | No failure (sets default) |
| `cls.emit` | Must override `BlazeOp.emit` | `TypeError` at import time |
| `cls.compose` | Must override `BlazeOp.compose` | `TypeError` at import time |

The `_is_abstract_op()` helper returns `True` for intermediate classes that should not be registered directly (e.g., a `ComputeOp` base class that provides shared logic for multiple compute ops). These classes skip enforcement because they are not intended to be complete ops.

### What `__init_subclass__` does NOT check

- It does not check `name`. An empty name will pass `__init_subclass__` but cause problems at registration.
- It does not validate ports, CT args, or the kernel header. These are checked later during `register()`.

---

## 2.2.2 Port descriptors: `Input`, `Output`, `Internal`

Port descriptors declare the circular buffers (CBs) that the op reads from, writes to, or uses internally. They are class-level attributes using Python's descriptor protocol.

### Declaration

```python
class RMSNorm(MicroOp):
    name = "rmsnorm"

    input: Input = Input()
    gamma: Input = Input()
    output: Output = Output()
```

Note: RMSNorm has no `Internal` ports. It uses only `Input` and `Output` descriptors. Scratch buffers (when needed) are allocated via `f.cb_scratch()` in `emit()`, not via `Internal` port descriptors.

### The `__set_name__` protocol

When Python processes the class body, it calls `__set_name__` on each descriptor, passing the attribute name:

```python
class Input:
    def __set_name__(self, owner, name):
        self.name = name    # "input", "gamma", etc.
```

This means the port's name is always the Python attribute name. You do not set it manually.

### Port types

| Descriptor | C++ role | CB direction | Allocated by |
|------------|----------|-------------|--------------|
| `Input()` | Source CB | Read by reader and/or compute RISC | Caller (FusedOp allocates before calling `emit()`) |
| `Output()` | Destination CB | Written by writer RISC | Caller (FusedOp allocates before calling `emit()`) |
| `Internal(num_pages=N)` | Scratch/intermediate CB | Read and written within the op | The op itself (allocated in `emit()`) |

### Port-name contract

Port names create a three-way contract between the Python class, the C++ kernel header, and the `emit()` method:

1. **Python -> C++:** The port attribute name (e.g., `input`) must match the `CB` field name in the C++ CT arg struct (e.g., `static constexpr CB input = A::input;`).
2. **Python -> emit():** The port attribute is used in `emit()` to reference the CB handle (e.g., `self.input` resolves to a `CBHandle`).
3. **C++ -> runtime:** The CB field in the CT arg struct becomes a compile-time constant that the kernel uses to identify which circular buffer to read from or write to.

If the names do not match, registration fails with a clear error:

```
ValueError: rmsnorm: kernel header ct_cb field 'src' has no matching port.
Valid ports: ['gamma', 'input', 'output']
```

### `OpSpec` construction from ports

During registration, the ports are collected into an `OpSpec` that summarizes the op's identity:

```python
# Constructed during BlazeOp.register():
OpSpec(
    name="rmsnorm",
    inputs=["input", "gamma"],
    outputs=["output"],
    internals=[],
    is_inter_core=False,
)
```

This `OpSpec` is stored in `_OP_REGISTRY` keyed by name (`str`).

---

## 2.2.3 Class configuration fields

Beyond ports, the class can declare configuration attributes that affect code generation:

| Attribute | Type | Default | Purpose |
|-----------|------|---------|---------|
| `math_fidelity` | `str` | `"HiFi4"` | Precision level for math operations on TRISC |
| `math_approx_mode` | `bool` | `False` | Enable hardware math approximation |
| `op_class` | `str` | `cls.__name__` | C++ struct name (overridden by auto-derivation; see [Section 2.1.4](./01_directory_structure.md)) |
| `fp32_dest_acc_en` | `bool` | `False` | Enable FP32 accumulation in the destination register |

```python
class RMSNorm(MicroOp):
    name = "rmsnorm"
    math_fidelity: str = "LoFi"
    # ...ports, compose(), emit()...
```

Note that RMSNorm uses `"LoFi"` (low fidelity), not `"HiFi4"`. The `fp32_dest_acc_en` flag is `False` at the class level in RMSNorm -- it is passed as a runtime parameter through `emit()` and `compose()` rather than being a fixed class attribute.

These fields are not auto-derived from the kernel header. They must be set explicitly in the Python class.

---

## 2.2.4 The `compose()` method

`compose()` is the compiler's entry point for the op. It receives a dictionary of named tensors and returns the output tensor(s) by invoking `emit()` with the correct arguments.

### Signature

```python
@classmethod
def compose(cls, f, tensors, output, user_args):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | `FusedProgram` | The program builder; provides CB allocation, CT arg registration, and program structure |
| `tensors` | `dict[str, Tensor]` | Input tensors keyed by port name (e.g., `{"input": tensor_a, "gamma": tensor_g}`) |
| `output` | `Tensor` | The output tensor (pre-allocated by the framework) |
| `user_args` | `dict` | Additional parameters from the caller (e.g., `{"num_tiles": 4, "prefix": "rmsnorm"}`) |

### `compose()` walkthrough: RMSNorm

```python
@classmethod
def compose(cls, f, tensors, output, user_args):
    inp = tensors["input"]
    cores = user_args.get("cores") or inp.memory_config().shard_spec.grid
    cls.emit(
        f, inp, tensors["gamma"],
        prefix=user_args.get("prefix", "rmsnorm"),
        cores=cores,
        epsilon=user_args.get("epsilon", 1e-6),
        scalar=user_args.get("scalar", 1.0),
        fp32_dest_acc_en=user_args.get("fp32_dest_acc_en", False),
        # ...additional user_args...
    )
```

RMSNorm's compose extracts named tensors from `tensors`, pulls configuration from `user_args` with defaults, and forwards everything to `emit()`. Note the `prefix` parameter -- it is a string passed explicitly, not generated by a method on `f`.

### `compose()` walkthrough: Copy

```python
@classmethod
def compose(cls, f, tensors, output, user_args):
    cls.emit(
        f,
        tensors["src"],
        output,
        prefix=user_args.get("prefix", "copy"),
        wait_src=user_args.get("wait_src", False),
        pop_src=user_args.get("pop_src", True),
        push_dst=user_args.get("push_dst", True),
        processor=user_args.get("processor", Risc.NCRISC),
        # ...additional user_args...
    )
```

Copy uses ports named `src` and `dst` (not `input` and `output`). The `output` parameter (the pre-allocated output tensor) is passed directly to `emit()`.

### `compose()` walkthrough: Matmul (simplified)

```python
@classmethod
def compose(cls, f, tensors, output, user_args):
    cls.emit(f, tensors["in0"], tensors["in1"],
             prefix=user_args.get("prefix", "matmul"),
             cores=user_args.get("cores"),
             # ...dimension parameters from user_args...
             )
```

Matmul takes two input tensors and dimension parameters from `user_args`.

---

## 2.2.5 The `emit()` method overview

`emit()` is the composition building block. It allocates circular buffers, registers CT args for each RISC processor, and returns a handle to the output tensor. A full treatment of the `emit()` API is deferred to Chapter 3; this section provides the structural overview.

### General pattern

Every `emit()` method follows a five-step pattern. Note that `emit()` is a `@staticmethod`, not a `@classmethod` -- it does not receive `cls` as a parameter:

```python
@staticmethod
def emit(f: FusedProgram, *input_tensors, *, prefix: str = "opname",
         cores: ttnn.CoreRangeSet, **params) -> CBHandle:
    # Step 1: Resolve input CBs and allocate output/scratch CBs
    inp = f.cb_from_tensor(input_tensor)  # or receive a CBHandle from upstream
    out_cb = f.cb_scratch(name=BlazeOp.cb_name(prefix, "out_cb"),
                          num_pages=..., core_ranges=cores, ...)

    # Step 2: Register per-core flags
    f.per_core_unified_ct_args([
        f.flag(f"{prefix}.is_active", cores),
    ])

    # Step 3: Register CT args for each RISC
    f.ncrisc_ct_args([
        (f"{prefix}.input", inp),
        (f"{prefix}.num_tiles", num_tiles),
    ])
    f.trisc_ct_args([
        (f"{prefix}.input", inp),
        (f"{prefix}.epsilon", _float_to_bits(eps)),
    ])

    # Step 4: Return via f.output() (creates a shadow graph entry
    # for dependency tracking -- see Chapter 3)
    return f.output("opname", cb_id=out_cb, core_ranges=cores,
                    prefix=prefix, input=inp)
```

Key differences from a naive reading:
- **`prefix` is a parameter**, not generated by a `f.unique_op_prefix()` method. Each `emit()` accepts `prefix` as a keyword argument with a default value (e.g., `prefix="rmsnorm"`).
- **Scratch CBs use `f.cb_scratch()`**, not `f.allocate_cb()` or `f.internal_cb()`. The CB name is built with `BlazeOp.cb_name(prefix, "out_cb")` which uses the `CB_NAME_DELIMITER` (triple underscore `___`).
- **`f.output()` is the return path**, not a bare `return output_cb`. It registers the output in the shadow graph for dependency tracking.

### CB allocation methods

| Method | Purpose | When to use |
|--------|---------|-------------|
| `f.cb_from_tensor(tensor)` | Create a CB handle from a `ttnn.Tensor` | For input tensors backed by sharded memory |
| `f.cb_scratch(name, ...)` | Allocate a new scratch CB | For output or intermediate buffers (name built via `BlazeOp.cb_name(prefix, "out_cb")`) |
| `f.cb_output()` | Allocate an output CB from the output tensor | For output tensors (pre-allocated by the framework) |

### CT arg registration methods

| Method | RISC target | C++ struct | When to use |
|--------|-------------|------------|-------------|
| `f.ncrisc_ct_args([(name, val), ...])` | NCRISC (DM0) | `ReaderCTArgs` | Reader-specific CT args |
| `f.brisc_ct_args([(name, val), ...])` | BRISC (DM1) | `WriterCTArgs` | Writer-specific CT args |
| `f.trisc_ct_args([(name, val), ...])` | TRISC | `ComputeCTArgs` | Compute-specific CT args |
| `f.unified_ct_args([(name, val), ...])` | All three RISCs | `CoreCTArgs` | Shared CT args broadcast to all RISCs |
| `f.per_core_unified_ct_args([(flag, ...), ...])` | All three RISCs | `CoreCTArgs` | Per-core flags (e.g., `f.flag(...)` results) |

Note: There is no `f.all_core_ct_args()` method. The correct method name is `f.unified_ct_args()` for broadcasting CT args to all RISCs.

### The CT arg prefix and `BlazeOp.cb_name()`

CT arg names in `emit()` use a dot separator between the prefix and the field name:

```python
# In emit(), prefix is a parameter (e.g., prefix="rmsnorm"):
f.ncrisc_ct_args([
    (f"{prefix}.input", input_cb),
    # full name: "rmsnorm.input"
])
```

The dot `.` separates the prefix from the field name in CT arg names. When multiple instances of the same op appear in a fused program, the caller passes different prefix values (e.g., `"rmsnorm_0"`, `"rmsnorm_1"`).

CB names use a separate convention with the triple-underscore `CB_NAME_DELIMITER` (`___`):

```python
BlazeOp.cb_name(prefix, "out_cb")
# Result: "rmsnorm___out_cb"
```

These two naming patterns serve different purposes: CT arg names use the dot separator (`prefix.field`), while CB names use the triple-underscore delimiter (`prefix___cbname`) via `BlazeOp.cb_name()`.

### Float-to-bits conversion

When a CT arg is a floating-point value (e.g., `epsilon` in RMSNorm), it must be passed as a `uint32_t` bit pattern because CT args are always 32-bit unsigned integers:

```python
# Python side:
f.trisc_ct_args([(f"{prefix}.epsilon", _float_to_bits(epsilon))])

# _float_to_bits: struct.pack('f', value) -> struct.unpack('I', ...)[0]
```

On the C++ side, the kernel receives it as `uint32_t` and recovers the float:

```cpp
static constexpr uint32_t epsilon = A::epsilon;
// Use: float eps = __builtin_bit_cast(float, epsilon);
```

---

## 2.2.6 Structural walkthrough: RMSNorm

RMSNorm is a compute-focused op that normalizes activations using root mean square normalization. It serves as the canonical example of a micro-op with separate reader and compute phases.

### Class structure

The following shows the actual RMSNorm class structure (simplified for clarity but structurally accurate):

```python
class RMSNorm(MicroOp):
    name: str = "rmsnorm"
    math_fidelity: str = "LoFi"

    input: Input = Input()
    gamma: Input = Input()
    output: Output = Output()

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        inp = tensors["input"]
        cores = user_args.get("cores") or inp.memory_config().shard_spec.grid
        cls.emit(
            f, inp, tensors["gamma"],
            prefix=user_args.get("prefix", "rmsnorm"),
            cores=cores,
            epsilon=user_args.get("epsilon", 1e-6),
            # ...additional user_args...
        )

    @staticmethod
    def emit(
        f: FusedProgram,
        input_handle,
        gamma_tensor,
        *,
        prefix: str = "rmsnorm",
        cores: ttnn.CoreRangeSet,
        epsilon: float = 1e-6,
        scalar: float = 1.0,
        fp32_dest_acc_en: bool = False,
        # ...additional params...
    ) -> CBHandle:
        # Resolve input CBs
        inp = f.cb_from_tensor(input_handle)       # or receive CBHandle
        gamma_cb = f.cb_from_tensor(gamma_tensor)

        # Allocate output scratch CB
        out_cb = f.cb_scratch(
            name=BlazeOp.cb_name(prefix, "out_cb"),
            num_pages=..., core_ranges=cores, ...
        )

        # Register per-core flags via unified path
        f.per_core_unified_ct_args([
            f.flag(f"{prefix}.is_active", cores),
            f.flag(f"{prefix}.pop_input", cores, True),
        ])

        # Register NCRISC CT args (reader)
        f.ncrisc_ct_args([
            (f"{prefix}.input", inp),
            (f"{prefix}.gamma", gamma_cb),
            (f"{prefix}.num_tiles", n),
            (f"{prefix}.input_is_tensor_backed", int(input_is_tensor_backed)),
        ])

        # Register TRISC CT args (compute)
        f.trisc_ct_args([
            (f"{prefix}.input", inp),
            (f"{prefix}.gamma", gamma_cb),
            (f"{prefix}.output", out_cb),
            (f"{prefix}.num_tiles", compute_n),
            (f"{prefix}.fp32_dest_acc_en", 1 if fp32_dest_acc_en else 0),
            (f"{prefix}.epsilon", _float_to_bits(epsilon)),
            (f"{prefix}.scalar", _float_to_bits(scalar)),
        ])

        return f.output("rmsnorm", cb_id=out_cb, core_ranges=cores,
                        prefix=prefix, input=inp, gamma=...)
```

### Structural observations

1. **`emit()` is a `@staticmethod`, not a `@classmethod`.** It does not receive `cls`. This is consistent across all micro-ops in the codebase.
2. **`prefix` is a parameter.** It is passed as a keyword argument with a default value (`"rmsnorm"`), not generated by a method on `f`.
3. **Two RISC-specific registrations plus unified flags.** RMSNorm registers per-core flags via `f.per_core_unified_ct_args()` (which maps to `CoreCTArgs`), reader args via `f.ncrisc_ct_args()`, and compute args via `f.trisc_ct_args()`.
4. **No Internal ports or `f.internal_cb()` calls.** RMSNorm has only `Input` and `Output` ports. Scratch CBs are allocated via `f.cb_scratch()`.
5. **No `f.set_math_fidelity()` calls.** Math fidelity is set at the class level (`math_fidelity = "LoFi"`) and registered during `BlazeOp.register()`, not called in `emit()`.
6. **Float-to-bits.** The `epsilon` and `scalar` parameters are converted to `uint32_t` bit patterns via `_float_to_bits()` before passing as CT args.
7. **`f.output()` is the return path.** It registers the output in the shadow graph for dependency tracking.

---

## 2.2.7 Structural walkthrough: Copy

Copy is a data-movement op that transfers data between circular buffers. It is simpler than RMSNorm: a single CT arg struct, no compute phase, no math configuration.

### Class structure

The following shows the actual Copy class structure (simplified for clarity):

```python
class Copy(MicroOp):
    name: str = "copy"

    src: Input = Input()
    dst: Output = Output()

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        cls.emit(
            f,
            tensors["src"],
            output,
            prefix=user_args.get("prefix", "copy"),
            wait_src=user_args.get("wait_src", False),
            pop_src=user_args.get("pop_src", True),
            push_dst=user_args.get("push_dst", True),
            processor=user_args.get("processor", Risc.NCRISC),
            # ...additional user_args...
        )

    @staticmethod
    def emit(
        f: FusedProgram,
        src,
        output_tensor,
        *,
        prefix: str = "copy",
        wait_src: bool = False,
        pop_src: bool = True,
        push_dst: bool = True,
        processor: Risc = Risc.NCRISC,
        # ...additional params...
    ) -> CBHandle:
        src_cb = f.cb_from_tensor(src)
        dst_cb = f.cb_from_tensor(output_tensor)

        # Per-core flags
        f.per_core_unified_ct_args([
            f.flag(f"{prefix}.is_active", src_cb.core_ranges),
        ])

        # All data-movement args broadcast to all RISCs
        f.unified_ct_args([
            (f"{prefix}.src", src_cb),
            (f"{prefix}.dst", dst_cb),
            (f"{prefix}.num_pages", src_cb.num_pages),
            (f"{prefix}.wait_src", int(wait_src)),
            (f"{prefix}.pop_src", int(pop_src)),
            (f"{prefix}.push_dst", int(push_dst)),
            (f"{prefix}.use_ncrisc", int(processor == Risc.NCRISC)),
            # ...additional args...
        ])

        return f.output("copy", dst_cb, prefix=prefix, src=src_cb)
```

### Structural observations

1. **Ports are named `src` and `dst`**, not `input` and `output`. This matches the CB field names in the C++ `CoreCTArgs` struct (`static constexpr CB src = A::src;`).
2. **`emit()` is a `@staticmethod`.** Like all micro-ops, it does not receive `cls`.
3. **One unified registration call.** Copy uses `f.unified_ct_args()` (not `f.all_core_ct_args()`, which does not exist) because its C++ header has only `CoreCTArgs`. All three RISCs receive the same CT args.
4. **No math configuration.** Copy does not perform compute -- it only moves data.
5. **No internal CBs.** Copy reads from one buffer and writes to another with no intermediate storage.
6. **Processor selection in C++.** Copy uses preprocessor guards and a `use_ncrisc` flag in the kernel header to select which RISC actually performs work:

   ```cpp
   void operator()() {
   #if defined(COMPILE_FOR_NCRISC)
       if constexpr (cta::use_ncrisc) { copy_impl(); }
   #elif defined(COMPILE_FOR_BRISC)
       if constexpr (!cta::use_ncrisc) { copy_impl(); }
   #endif
   }
   ```

7. **CoreCTArgs and is_inter_core.** Because Copy uses `CoreCTArgs` (which sets `has_brisc`, `has_ncrisc`, and `has_trisc` all to `True` in the parser), the auto-derived `is_inter_core` will be `True`. This is a consequence of `CoreCTArgs` being classified as an all-RISC struct. See [Section 2.4](./04_auto_derivation.md) for details on this behavior and when to override it.

---

## 2.2.8 Contrast table: RMSNorm vs. Copy

| Dimension | RMSNorm | Copy |
|-----------|---------|------|
| Port names | `input`, `gamma` (Input), `output` (Output) | `src` (Input), `dst` (Output) |
| Active processors | NCRISC (reader) + TRISC (compute) | NCRISC or BRISC (selectable via `use_ncrisc` flag) |
| CT arg registration | `f.per_core_unified_ct_args()` + `f.ncrisc_ct_args()` + `f.trisc_ct_args()` | `f.per_core_unified_ct_args()` + `f.unified_ct_args()` |
| C++ CT arg structs | `CoreCTArgs` + `ReaderCTArgs` + `ComputeCTArgs` | `CoreCTArgs` only |
| Math configuration | `LoFi`, `fp32_dest_acc_en` passed as runtime param | None |
| Processor selection | Separate structs + preprocessor guards | `if constexpr (cta::use_ncrisc)` in `operator()()` |
| Internal CBs / scratch | Scratch via `f.cb_scratch()` (no Internal ports) | None |

This table highlights the two primary patterns for micro-ops:

- **Multi-struct pattern** (RMSNorm): Multiple CT arg structs target different RISCs (`CoreCTArgs` for shared flags, `ReaderCTArgs` for NCRISC, `ComputeCTArgs` for TRISC). The struct name determines which RISC receives which args.
- **Unified-struct pattern** (Copy): A single `CoreCTArgs` struct shared by all RISCs. Preprocessor guards and `if constexpr` flags in the Op class determine which RISC does what.

---

## Key takeaways

1. **`__init_subclass__` enforces the contract.** Every concrete `MicroOp` must override `emit()` and `compose()`. The `op_class` defaults to the Python class name if not set.

2. **Port descriptors use `__set_name__`.** The attribute name becomes the port name, which must match the CB field name in the C++ header. Mismatches produce clear error messages.

3. **`compose(cls, f, tensors, output, user_args)` maps tensors to `emit()`.** It extracts named tensors from `tensors`, pulls configuration from `user_args`, and forwards everything to `emit()`.

4. **`emit()` is a `@staticmethod`.** It does not receive `cls`. It accepts `prefix` as a keyword argument (not generated by a method on `f`), resolves CBs, registers per-RISC CT args, and returns via `f.output()`.

5. **CT arg registration targets specific RISCs.** `f.ncrisc_ct_args()` for readers, `f.trisc_ct_args()` for compute, `f.brisc_ct_args()` for writers, `f.unified_ct_args()` for all RISCs. Per-core flags use `f.per_core_unified_ct_args()` with `f.flag()`.

6. **Two patterns dominate.** Multi-struct ops (like RMSNorm) use different CT arg structs for different processors. Unified-struct ops (like Copy) use `CoreCTArgs` with preprocessor guards and `if constexpr` flags in C++.

---

**Previous:** [Section 2.1 -- Directory Structure](./01_directory_structure.md) | **Next:** [Section 2.3 -- The C++ Kernel Header](./03_cpp_kernel_header.md)
