# 07.01 -- Composition Basics

[Next: Advanced Patterns -->](02_advanced_patterns.md)

This chapter covers how TT-Blaze fused ops are built by composing MicroOp `emit()` calls inside a `compose()` function. The composition system is the heart of Blaze's productivity advantage: a single 60-line `compose()` can replace a thousand lines of manual TT-Metal wiring while remaining fully transparent to the compiler's CB allocation, CT-arg generation, and kernel codegen passes.

Readers should be familiar with the `FusedProgram` context (Ch. 4), the CB allocation model (Ch. 3), and the CT-arg engine (Ch. 5) before proceeding.

---

## The compose() Contract

Every `FusedOp` must override `compose(cls, f, tensors, output, user_args)` -- the compiler entry point that maps a tensor dict to `emit()` calls (see [Ch. 2.02 -- MicroOp and FusedOp](../ch02_blazeop_hierarchy/02_microop_and_fusedop.md) for the full contract and parameter table). The return value is `None`; side effects on `f` (the `FusedProgram`) are the entire point.

Here is a minimal real example -- the SharedExpert fused op's `compose()`:

```python
# blaze/ops/shared_expert/op.py

class SharedExpert(FusedOp):
    name: str = "shared_expert"

    activation: Input = Input()
    gate_up_weights: Input = Input()
    down_weights: Input = Input()
    bias: Input = Input()
    output: Output = Output()

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        cls.emit(
            f,
            tensors["activation"],
            tensors["gate_up_weights"],
            tensors["down_weights"],
            tensors["bias"],
            output,
            prefix=user_args.get("prefix", "shared_expert"),
            ab_coords=user_args.get("ab_grids"),
            down_coords=user_args.get("down_coords"),
            pop_shared_act=user_args.get("pop_shared_act", True),
        )
```

The pattern is deliberate: `compose()` unpacks the dict-based interface into the statically-typed `emit()` signature. This separation means `emit()` can also be called directly from a parent fused op (e.g., `MoE` calls `SharedExpert.emit(...)` without going through `compose()`).

---

## How the Compiler Invokes compose()

`BlazeCompiler._compile_fused_op()` creates a `FusedProgram`, calls the op's `compose_fn`, handles noop-prefix tagging, auto-wires outputs, and calls `f.build()` (see [Ch. 3.06 -- BlazeCompiler, Fused Op Path](../ch03_compilation_pipeline/06_blaze_compiler.md) for the full sequence).

Three details matter for composition authors:

1. **Mesh context injection** (step 2): The compiler automatically populates `f.mesh_coord`, `f.mesh_shape`, and `f.mesh_device` from the per-device compilation loop. Ops that need device-aware routing (MoE index offsets, ReduceToOne topology) read these fields directly.

2. **Auto-wiring** (step 5): If `compose()` does not explicitly call `f.wire_output()` or `f.cb_output()`, the compiler auto-wires the last emitted output CB to the output tensor. This is the common case for most ops.

3. **Build** (step 6): `f.build()` triggers `_prepare_for_build()`, which runs scratch-CB compaction and multi-phase reconfig (see [Ch. 7.02](02_advanced_patterns.md)), then either uses the provided kernel path or generates a kernel from the shadow graph.

---

## Chaining emit() Calls: The CBHandle Chain

The core composition pattern is a chain of `emit()` calls where each MicroOp's return value -- a `CBHandle` -- feeds into the next op as input. The CBHandle carries the CB ID, page geometry, data format, tile descriptor, and optional byte offset for overlapped views (see [Ch. 4.01 -- CBHandle Abstraction](../ch04_data_flow/01_cbhandle_abstraction.md) for the full field reference). Crucially, CBHandle supports integer coercion (`int(handle)` returns `cb_id`), so handles pass directly into CT-arg tuples:

```python
f.trisc_ct_args([
    (f"{prefix}.in0", in0_handle),   # Passes handle.cb_id as the arg value
    (f"{prefix}.out", out_handle),
    (f"{prefix}.k_num_tiles", k_num_tiles),
])
```

Here is how the chain works in practice. Consider Mcast.emit():

```python
# blaze/ops/mcast/op.py -- Mcast.emit() (simplified)

@staticmethod
def emit(f, src, *, prefix, sender_sem=None, dst_num_pages=None, dst_tile_info=None):
    # 1. Accept either a tensor (allocates input CB) or an existing CBHandle
    if isinstance(src, CBHandle):
        src_handle = src
    else:
        src_handle = f.cb_from_tensor(src)

    # 2. Allocate a scratch CB for the mcast destination
    dst = f.cb_scratch(
        name=BlazeOp.cb_name(prefix, "dst"),
        num_pages=_dst_num_pages,
        core_ranges=f.all_cores,
        data_format=_dst_data_format,
        tile=_dst_tile_desc,
        page_size=_dst_page_size,
        balanced=False,
    )

    # 3. Register semaphores
    sender_sem = f.semaphore(f"{prefix}.sender")
    receiver_sem = f.semaphore(f"{prefix}.receiver")

    # 4. Register CT args referencing both CBHandles
    f.ncrisc_ct_args([
        (f"{prefix}.src", src_handle),
        (f"{prefix}.dst", dst),
        ...
    ])

    # 5. Record shadow-graph node, return dst handle
    return f.output("mcast", dst, prefix=prefix, src=src)
```

The caller sees only the returned `CBHandle`. The next op in the chain consumes it:

```python
# In a compose() or parent emit():
act_mcast = Mcast.emit(f, activation, prefix="act_mcast", ...)
matmul_out = KNMatmul.emit(f, act_mcast, weights, prefix="gu", ...)
# act_mcast.cb_id feeds into KNMatmul's CT args as an input CB reference
```

This produces a data-dependency chain through the shadow graph that the kernel codegen uses to order phases.

---

## Prefix Namespacing

Every `emit()` call takes a `prefix` argument. Prefixes serve two purposes:

1. **CT-arg uniqueness**: All CT args emitted by a MicroOp are prefixed with `{prefix}.{arg_name}`. Since the unified kernel reads args by name across all phases, names must be unique per RISC. The `CTArgEngine._validate_no_collisions()` method enforces this:

```python
# blaze/ct_args.py -- _validate_no_collisions

def _validate_no_collisions(self, all_args):
    per_risc = {"ncrisc": {}, "brisc": {}, "trisc": {}}
    seen_ids = set()

    for node_id, args in all_args.items():
        args_id = id(args)
        if args_id in seen_ids:
            continue  # shared-prefix nodes reuse same args list
        seen_ids.add(args_id)

        for arg in args:
            existing = per_risc[arg.risc].get(arg.name)
            if existing is not None:
                raise ValueError(
                    f"CT arg name collision on {arg.risc}: '{arg.name}' "
                    f"defined by both '{existing}' and '{node_id}'"
                )
            per_risc[arg.risc][arg.name] = node_id
```

2. **CB name uniqueness**: Scratch CBs are named via `BlazeOp.cb_name(prefix, local_name)`, which joins with the triple-underscore delimiter:

```python
# blaze/blaze_op.py

class BlazeOp:
    CHILD_PREFIX_DELIMITER: str = "__"
    CB_NAME_DELIMITER: str = "___"

    @classmethod
    def child_prefix(cls, prefix: str, name: str) -> str:
        return f"{prefix}{cls.CHILD_PREFIX_DELIMITER}{name}" if prefix else name

    @classmethod
    def cb_name(cls, prefix: str, name: str) -> str:
        return f"{prefix}{cls.CB_NAME_DELIMITER}{name}"
```

### Hierarchical Prefixes

When a FusedOp composes other FusedOps, prefixes nest via `BlazeOp.child_prefix()`. The MoE op demonstrates this:

```
moe                                    # top-level prefix
  moe__rmsnorm                         # RMSNorm prefix
  moe__act_mcast                       # activation mcast prefix
  moe__routed                          # RoutedExpert prefix
    moe__routed__router                #   MoERouter prefix
    moe__routed__index_mcast           #   index mcast prefix
    moe__routed__score_mcast           #   score mcast prefix
    moe__routed__swiglu                #   SwiGLU prefix
  moe__shared                          # SharedExpert prefix
    moe__shared__gu                    #   KNMatmul prefix
    moe__shared__gated_reduce          #   GatedReduce prefix
    moe__shared__mcast                 #   intermediate mcast
    moe__shared__down_proj             #   DownProj prefix
  moe__shared_output_mcast            # shared output mcast
  moe__merge                           # ResidualAdd prefix
  moe__reduce_to_one                   # ReduceToOne prefix
```

Each level appends `__` (the child prefix delimiter), ensuring no two MicroOps within the same FusedProgram ever emit conflicting CT-arg names. A collision at any level triggers a `ValueError` from the CT-arg engine at compile time.

### What Happens When Names Collide

If two `emit()` calls accidentally use the same prefix, the per-core CT arg duplicate check fires first (in `FusedProgram._per_core_ct_args_check()`):

```python
def _per_core_ct_args_check(self, args):
    for name, mapping in args:
        if name in self._ct_arg_names_seen:
            raise ValueError(f"Duplicate per-core CT arg: {name!r}")
        self._ct_arg_names_seen.add(name)
```

For scalar CT args, the shadow-graph validation catches collisions during `CTArgEngine._validate_no_collisions()`. Either way, the error is deterministic and names the conflicting arg -- no silent corruption.

---

## Wiring Outputs

There are three ways to connect the final CB in a composition chain to the output tensor:

### 1. Explicit wire_output()

The op calls `f.wire_output(handle, output_tensor)` inside `emit()`. This converts a scratch CB into a tensor-backed output:

```python
# blaze/fused_program.py -- wire_output

def wire_output(self, handle: CBHandle, output_tensor) -> None:
    cb_id = handle.cb_id
    new_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_id, output_tensor)
    # Preserve tile and page_size from the existing CB
    for i, desc in enumerate(self.program._cb_descriptors):
        fmts = desc.format_descriptors
        if fmts and fmts[0].buffer_index == cb_id:
            for fi, old_fmt in enumerate(fmts):
                if fi < len(new_desc.format_descriptors):
                    new_desc.format_descriptors[fi].tile = old_fmt.tile
                    new_desc.format_descriptors[fi].page_size = old_fmt.page_size
            self.program._cb_descriptors[i] = new_desc
            break
    self.program._io_tensors.append((cb_id, output_tensor))
    self.program._output_tensors.append(output_tensor)
    self._tensor_backed_cbs.add(cb_id)
```

### 2. cb_output()

The op calls `f.cb_output(output_tensor)` to allocate a CB backed by the output tensor from the start:

```python
out = f.cb_output(output_tensor)
# ... kernel writes directly into output_tensor's L1 buffer
```

### 3. Auto-wiring in _compile_fused_op()

If compose() does neither, the compiler auto-wires using the last recorded output. The auto-wiring logic in `_compile_fused_op()` follows a priority chain:

```python
# blaze/compiler.py -- auto-wire sequence (simplified)

if f.program._output_tensors:
    pass  # Already wired explicitly
elif f._last_output_cb_ids:
    # f.output() recorded the last emitted handle
    for port_name, handle in f._last_output_cb_ids:
        tensor = all_tensors.get(port_name)
        if int(handle) in registered_cbs:
            f.program._output_tensors.append(tensor)
        else:
            f.wire_output(handle, tensor)
elif output_tensor is not None:
    # Fallback: wire from graph
    f.wire_output_from_graph(output_tensor)
```

In practice, most FusedOps rely on auto-wiring. The last MicroOp's `f.output()` call records its `CBHandle` in `_last_output_cb_ids`, and the compiler wires it to the output tensor after `compose()` returns.

---

## The FusedOp and MicroOp Class Hierarchy

The `BlazeOp -> FusedOp / MicroOp` hierarchy and the auto-registration of `FusedOpConfig` during `register()` are covered in [Ch. 2.02](../ch02_blazeop_hierarchy/02_microop_and_fusedop.md). The key points for composition: a `FusedOp` must override `compose()`; `emit()` is optional but recommended for reusability. A `MicroOp` must override both. A `FusedOp` subclass that overrides `compose()` gets auto-registered as a `FusedOpConfig`, storing the `compose_fn` that `_compile_fused_op()` will call. FusedOps that compose only MicroOps typically set `kernel=None`, letting the compiler's codegen generate the unified kernel from the shadow graph.

---

## Summary

| Concept | Mechanism |
|---------|-----------|
| **Entry point** | `compose(f, tensors, output, user_args) -> None` |
| **Chaining** | Each `emit()` returns a `CBHandle` consumed by the next |
| **Namespacing** | `prefix` on every `emit()` call; `child_prefix()` for nesting |
| **Collision detection** | `CTArgEngine._validate_no_collisions()` per RISC |
| **Output wiring** | Explicit `wire_output()`, `cb_output()`, or auto-wiring by compiler |
| **Registration** | `FusedOp.register()` auto-creates `FusedOpConfig` from `compose()` |

[Next: Advanced Patterns -->](02_advanced_patterns.md)
