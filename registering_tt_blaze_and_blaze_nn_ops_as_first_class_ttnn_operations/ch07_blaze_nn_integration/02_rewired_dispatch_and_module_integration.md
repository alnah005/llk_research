# 7.2 Rewired Dispatch and Module Integration

[Section 7.1](01_current_blaze_nn_dispatch.md) traced a transformer attention block through blaze-nn's current dispatch stack and showed that op identity is lost at Layer 6 when `_run_program()` unconditionally calls `ttnn.generic_op()`. This section rewires that dispatch path to route through registered `ttnn.blaze.*` callables, designs the fallback strategy for unregistered ops, addresses graph-mode and direct-mode differences, integrates with the module system, and shows how registered Blaze ops compose with native TTNN ops. We trace the **same attention block** through the proposed path, producing a direct layer-by-layer comparison with the current behavior.

---

## 7.2.1 Updating `_registry.py`: The `ttnn_callable` Field

The most natural place to store the registered TTNN callable is in the `OpMapping` itself. This keeps the dispatch resolution in the same data structure that already controls op-name resolution and kwargs transformation.

### Modified `OpMapping`

```python
# PROPOSED: blaze_nn/_registry.py (modified OpMapping)
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class OpMapping:
    """Maps a blaze-nn functional name to its Blaze op implementation.

    MODIFIED: Added lazily-resolved ttnn_callable for registered op dispatch.
    """
    blaze_op_name: str
    default_grid: Optional[Any]
    kwargs_transform: Optional[Callable] = None
    extra_defaults: Dict[str, Any] = field(default_factory=dict)

    # Lazy resolution state for the TTNN callable.
    # _resolved=False means "not yet checked".
    # _resolved=True, _ttnn_callable=<fn> means "registered callable found".
    # _resolved=True, _ttnn_callable=None means "checked and not available".
    _ttnn_callable: Optional[Any] = field(default=None, repr=False)
    _resolved: bool = field(default=False, repr=False)

    @property
    def ttnn_callable(self):
        """
        Lazily resolve the ttnn.blaze.* callable for this op.

        Returns the callable if registered, or None if not available.
        Resolution happens once per OpMapping instance; subsequent
        accesses return the cached result.
        """
        if not self._resolved:
            self._ttnn_callable = _resolve_ttnn_callable(self.blaze_op_name)
            self._resolved = True
        return self._ttnn_callable
```

### The Resolution Function

```python
# PROPOSED: blaze_nn/_registry.py (new functions)
import os

# Module-level flag: is ttnn.blaze available at all?
_TTNN_BLAZE_MODULE = None
_TTNN_BLAZE_CHECKED = False

def _get_ttnn_blaze_module():
    """
    Return the ttnn.blaze module if available, or None.
    Checked once at first access; cached thereafter.
    """
    global _TTNN_BLAZE_MODULE, _TTNN_BLAZE_CHECKED
    if not _TTNN_BLAZE_CHECKED:
        try:
            import ttnn
            _TTNN_BLAZE_MODULE = getattr(ttnn, "blaze", None)
            if _TTNN_BLAZE_MODULE is not None:
                logger.debug(
                    "ttnn.blaze module detected; registered ops will be "
                    "used for dispatch where available."
                )
        except ImportError:
            _TTNN_BLAZE_MODULE = None
        _TTNN_BLAZE_CHECKED = True
    return _TTNN_BLAZE_MODULE


def _resolve_ttnn_callable(blaze_op_name: str):
    """
    Attempt to resolve ttnn.blaze.{blaze_op_name}.

    Returns the callable if found and verified, or None if the op
    is not registered as a first-class TTNN operation.
    """
    if os.environ.get("BLAZE_NN_FORCE_GENERIC", "0") == "1":
        return None  # User explicitly requested generic_op for all ops

    blaze_mod = _get_ttnn_blaze_module()
    if blaze_mod is None:
        return None

    callable_obj = getattr(blaze_mod, blaze_op_name, None)
    if callable_obj is not None:
        # Verify it is a registered TTNN operation, not an arbitrary
        # attribute that happens to share the name. The __ttnn_operation__
        # property is set by bind_registered_operation() (Section 4.3.2).
        if getattr(callable_obj, "__ttnn_operation__", False):
            logger.debug(
                f"Resolved ttnn.blaze.{blaze_op_name} as registered "
                f"TTNN operation."
            )
            return callable_obj
        else:
            logger.warning(
                f"ttnn.blaze.{blaze_op_name} exists but is not a "
                f"registered TTNN operation; falling back to generic_op."
            )
    return None
```

### Design Decisions

**Lazy resolution.** The `ttnn_callable` property resolves on first access, not at module import time. This avoids import-order dependencies: `blaze_nn` can be imported before `ttnn` without error. The first `F.linear()` call triggers resolution; all subsequent calls use the cached result.

**Three-state caching.** `_resolved=False` means "not yet checked"; `_resolved=True, _ttnn_callable=<fn>` means "registered callable found"; `_resolved=True, _ttnn_callable=None` means "checked and not available." This avoids repeated `getattr` probes on the hot path.

**`__ttnn_operation__` guard.** The resolution function verifies that the attribute on `ttnn.blaze` is actually a registered TTNN operation (has the `__ttnn_operation__` property set by `bind_registered_operation()`, as documented in [Section 4.3.2](../ch04_adapter_pattern/03_python_dispatch_and_registration.md)). This guards against namespace collisions where a non-operation attribute happens to share a name with a Blaze op.

**`try/except` around `import ttnn`.** The `_get_ttnn_blaze_module()` function catches `ImportError` so that blaze-nn can be used in test environments where `ttnn` is not installed.

**`BLAZE_NN_FORCE_GENERIC` environment variable.** Setting `BLAZE_NN_FORCE_GENERIC=1` disables all registered dispatch, forcing every op through `generic_op`. The env var is read live on each resolution (not cached at import time), so it can be toggled at runtime and takes effect after `reset_dispatch_cache()`. This provides an immediate rollback mechanism during migration that requires no code changes -- only an environment variable.

**Per-op granularity.** Each `OpMapping` resolves its callable independently. If `ttnn.blaze.matmul` is registered but `ttnn.blaze.gelu` is not, `F.linear()` uses the registered path while `F.gelu()` falls back to `generic_op`. This enables incremental migration as described in Phase 2--4 of the migration plan ([Chapter 8](../ch08_build_migration_alternatives/index.md)).

> **Thread safety note.** The lazy resolution above is not thread-safe under concurrent first-access from multiple threads. In CPython, the GIL provides sufficient protection for the attribute assignment. For deployments using free-threaded Python or multi-threaded serving frameworks, a double-checked locking pattern with `threading.Lock` (as described in v3's evaluation) can be added to the property without changing the external API.

### Cache Invalidation

```python
# PROPOSED: blaze_nn/_registry.py
def reset_dispatch_cache():
    """
    Reset all cached ttnn_callable references.
    Call this after rebuilding TT-Metal with new registered ops,
    or after dynamic registration at runtime.
    """
    global _TTNN_BLAZE_CHECKED, _TTNN_BLAZE_MODULE
    _TTNN_BLAZE_CHECKED = False
    _TTNN_BLAZE_MODULE = None
    for mapping in _REGISTRY.values():
        mapping._resolved = False
        mapping._ttnn_callable = None
    logger.info("Dispatch cache reset; callables will be re-resolved on next use.")
```

---

## 7.2.2 The Dual-Resolution Architecture

The rewired dispatch has two independent resolution points. Understanding both is essential for seeing how blaze-nn connects to registered operations.

**Resolution Point 1: `_registry.py` (blaze-nn layer).** The `OpMapping.ttnn_callable` property tells blaze-nn whether a registered callable exists and caches the result. This resolution exists for the diagnostic/introspection API (`get_dispatch_info()`, `get_registration_report()`) and early validation logging. It does not gate dispatch.

**Resolution Point 2: `MeshCompiledProgram._resolve_dispatch()` (Blaze layer).** The actual callable switch happens in `_run_program()`, which is owned by `compiler.py` ([Section 4.3.1](../ch04_adapter_pattern/03_python_dispatch_and_registration.md)). The `_resolve_dispatch()` method resolves the dispatch function once at `MeshCompiledProgram` creation time via `getattr(ttnn.blaze, op_name, None)`. This is the sole dispatch decision maker.

**Why two resolution points?** blaze-nn and the Blaze compiler operate at different abstraction levels. blaze-nn knows the functional name and can verify the callable early (via `__ttnn_operation__`) for diagnostics and early validation logging. The Blaze compiler receives the op name, produces a `MeshCompiledProgram`, and independently resolves the callable. Both independently resolve, but Resolution Point 2 is authoritative -- it determines which function `_run_program()` actually calls. If they disagree (e.g., because `ttnn.blaze` was modified between module import and execution), `_resolve_dispatch()` governs runtime behavior.

```text
Import time (or first F.* call):
  _registry.py: OpMapping("matmul").ttnn_callable = ttnn.blaze.matmul  [Resolution Point 1]

Compile time (first call to F.linear):
  BlazeCompiler.compile(op_name="matmul", ...)
    -> MeshCompiledProgram(op_name="matmul", ...)
       -> _resolve_dispatch("matmul")
          -> getattr(ttnn.blaze, "matmul", None) -> ttnn.blaze.matmul  [Resolution Point 2]
          -> self._dispatch_fn = ttnn.blaze.matmul

Runtime (every subsequent call):
  _run_program(io_tensors, descriptor)
    -> self._dispatch_fn(io_tensors, descriptor)
    == ttnn.blaze.matmul(io_tensors, descriptor)     [direct call, no lookup]
```

---

## 7.2.3 Fallback Strategy

The fallback strategy is layered, with three levels of graceful degradation.

### Level 1: `ttnn.blaze` Namespace Absent

If the `ttnn` module does not have a `blaze` attribute (e.g., running against a TT-Metal build that predates the adapter), `_get_ttnn_blaze_module()` returns `None`. No `ttnn_callable` fields are populated. All ops dispatch through `generic_op`. This is the same behavior as the current system -- zero regression risk.

### Level 2: Specific Op Not Registered

If `ttnn.blaze` exists but a specific op is not yet in the X-macro list, `getattr(ttnn.blaze, op_name, None)` returns `None`. That op's `ttnn_callable` remains `None`, and its `_resolve_dispatch()` falls back to `ttnn.generic_op`.

### Level 3: Runtime Error Recovery

If a registered callable raises an exception at runtime (e.g., due to a version mismatch between the adapter and the descriptor schema), the error propagates to the caller. There is no automatic retry through `generic_op` -- this would mask bugs. The expectation is that registered callables and `generic_op` accept the same `(io_tensors, descriptor)` signature and should not diverge in behavior.

### Fallback Decision Table

| Condition | `OpMapping.ttnn_callable` | `_resolve_dispatch()` Result | Dispatch Target | Profiler Output |
|-----------|:-:|:-:|:-:|:-:|
| `ttnn.blaze` absent | `None` | `ttnn.generic_op` | `generic_op` | `"ttnn::generic_op"` |
| `ttnn.blaze` present, op registered | `ttnn.blaze.matmul` | `ttnn.blaze.matmul` | Adapter | `"ttnn::blaze::matmul"` |
| `ttnn.blaze` present, op NOT registered | `None` | `ttnn.generic_op` | `generic_op` | `"ttnn::generic_op"` |
| `ttnn` import fails (unlikely) | `None` | `ttnn.generic_op` | `generic_op` | `"ttnn::generic_op"` |
| `BLAZE_NN_FORCE_GENERIC=1` | `None` (forced) | `ttnn.generic_op` | `generic_op` | `"ttnn::generic_op"` |
| Callable raises at runtime | N/A | N/A | Exception propagated | N/A |

### Mixed-Registration Profiler Output

During migration, when some ops are registered and others are not, the profiler timeline reflects the mixed state:

```text
Tracy Timeline (partial migration):
  |--- ttnn::blaze::rmsnorm --|--- ttnn::blaze::matmul --|--- ttnn::generic_op ---|
  |  (rmsnorm registered)     |  (matmul registered)     | (rope NOT registered)  |
```

This mixed state is informative: it shows exactly which ops have been promoted and which still use the generic path.

---

## 7.2.4 Graph Mode vs Direct Mode

The rewired dispatch behaves differently in graph mode and direct mode, but both ultimately benefit from registration.

### Direct Mode (Compose)

In direct mode, each `F.*` call compiles and executes immediately. The `_resolve_dispatch()` call in `MeshCompiledProgram.__init__` selects the registered callable (or `generic_op` fallback). Execution proceeds as in [Section 4.3.1](../ch04_adapter_pattern/03_python_dispatch_and_registration.md).

```text
Direct mode:
  F.linear(h, wq) -> _dispatch("linear", ...) -> ComposeTracingContext.dispatch(...)
    -> BlazeCompiler.compile(Matmul, ...) -> MeshCompiledProgram(op_name="matmul")
       -> _resolve_dispatch("matmul") -> ttnn.blaze.matmul
    -> compiled.run()
       -> _run_program() -> ttnn.blaze.matmul(io_tensors, descriptor)
          -> BlazeDeviceOperationAdapter<MatmulTag>::invoke(...)
          -> launch<BlazeDeviceOperationAdapter<MatmulTag>>(...)
             -> TracyOpMeshWorkload("ttnn::blaze::matmul")  // NAMED
             -> compute_program_hash(type_hash<Adapter<MatmulTag>>, ...)  // ISOLATED
```

### Graph Mode

In graph mode, `F.linear()` records a graph node without executing. The op name is stored in the node. When the graph is compiled and executed:

```text
Graph mode:
  # Tracing phase (no hardware execution):
  F.linear(h_proxy, wq_proxy) -> GraphTracingContext.dispatch("matmul", ...)
    -> graph.add_node(op_name="matmul", ...)
    -> return TensorProxy(...)

  # Compilation phase:
  compiled_graph = blaze_nn.compile(graph)
    -> for each node in graph:
         BlazeCompiler.compile(node.op_class, ..., op_name=node.op_name)
         -> MeshCompiledProgram(op_name="matmul")
            -> _resolve_dispatch("matmul") -> ttnn.blaze.matmul

  # Execution phase:
  compiled_graph.run(real_tensors)
    -> for each compiled_program in sequence:
         compiled_program.run(tensors)
           -> _run_program() -> ttnn.blaze.matmul(io_tensors, descriptor)
              -> TracyOpMeshWorkload("ttnn::blaze::matmul")  // NAMED
```

The key difference is **when** `_resolve_dispatch()` runs: during the compilation phase, not the tracing phase. The tracing phase only records op names; the compilation phase resolves them to TTNN callables.

### Comparison Table

| Aspect | Graph Mode | Compose Mode |
|---|---|---|
| When trace occurs | Explicitly (inside `with GraphTracingContext():`) | Never (no tracing) |
| When hardware executes | Deferred (at `compile_and_run()`) | Immediately (at each `F.*()` call) |
| Where registration takes effect | In `_run_program()` at execution time | In `_run_program()` at each call |
| Graph capture benefit | Named nodes in executed graph | N/A (no graph) |
| Profiler benefit | Named Tracy zones per op | Named Tracy zones per op |
| Cache benefit | Per-op hash namespace isolation | Per-op hash namespace isolation |

---

## 7.2.5 Scenario Walkthrough: Attention Block Through Proposed Dispatch

We retrace the attention block from [Section 7.1.6](01_current_blaze_nn_dispatch.md) through the proposed dispatch path in direct mode. The following fully annotated trace of `F.linear(h, self.wq)` shows every change; lines marked `[CHANGED]` are the only differences from the current path.

### Annotated Call: `F.linear(h, self.wq)` (Q projection)

```text
F.linear(h, self.wq)
  -> _dispatch("linear", h, self.wq)
     -> mapping = _REGISTRY["linear"]
        -> OpMapping(blaze_op_name="matmul", default_grid="4x8",
                     kwargs_transform=_linear_kwargs_transform)
     -> mapping.ttnn_callable -> ttnn.blaze.matmul             [CHANGED: lazy resolve,
                                                                 __ttnn_operation__ verified,
                                                                 cached after first access]
     -> user_args = {"transpose_b": True}
     -> ctx = ComposeTracingContext
     -> ctx.dispatch("matmul", [h, wq_device], {"transpose_b": True}, "4x8")
        -> compiled = BlazeCompiler().compile(Matmul, tensors, output,
                                              {"transpose_b": True},
                                              op_name="matmul")  [CHANGED: op_name passed]
           -> MeshCompiledProgram(op_name="matmul", ...)
              -> _resolve_dispatch("matmul")
                 -> ttnn.blaze.matmul  [CHANGED: not generic_op]
        -> compiled.run(tensors)
           -> _run_program(io_tensors, mesh_program_descriptor)
              -> self._dispatch_fn(io_tensors, descriptor)
              == ttnn.blaze.matmul(io_tensors, descriptor)     [CHANGED: named callable]
                 -> BlazeDeviceOperationAdapter<MatmulTag>::invoke(...)
                 -> launch<BlazeDeviceOperationAdapter<MatmulTag>>(...)
                    -> TracyOpMeshWorkload("ttnn::blaze::matmul")  [CHANGED: named zone]
                    -> compute_program_hash(
                         type_hash<Adapter<MatmulTag>>, ...)       [CHANGED: isolated prefix]
                    -> [cache lookup, factory execution, enqueue]
     -> returns: q
```

All eight calls in the attention block follow the same pattern. The only change from the current dispatch path is at Layers 6--9, where `generic_op` is replaced by `ttnn.blaze.{op}`. The op name and grid configuration vary per call (`"rmsnorm"`/`"1x32"`, `"matmul"`/`"4x8"`, `"rope"`/`"1x32"`, `"sdpa"`/`"8x4"`), but the dispatch mechanics are identical to the annotated trace above.

---

## 7.2.6 Observability Impact: Per-Op Profiler Aggregation

After registration, each of the eight attention-block calls appears with a distinct name in Tracy, TTNN GraphTracker, and the program cache (see the before/after layer map in [Section 7.2.11](#7211-dispatch-layer-map-after-registration) for the per-layer comparison, and [Section 7.1.7](01_current_blaze_nn_dispatch.md) for the current opaque baseline).

### Per-Op Profiler Aggregation

With named dispatch, profiler post-processing can aggregate across the full 32-layer model:

```text
Op Type               | Invocations | Total Time | Avg Time | % of Forward Pass
----------------------|-------------|------------|----------|------------------
blaze::matmul         |         128 |    5.76 ms |  45.0 us |            48.0%
blaze::sdpa           |          32 |    5.76 ms | 180.0 us |            48.0%
blaze::rmsnorm        |          64 |    0.38 ms |   6.0 us |             3.2%
blaze::rope           |          64 |    0.10 ms |   1.5 us |             0.8%
```

This breakdown was impossible with `generic_op`. It is now a standard Tracy post-processing query.

---

## 7.2.7 Module Integration: What Changes, What Stays

### `Module.forward()`: No Changes

The `Module` base class calls `forward()`, which calls functional API methods. There is no dispatch logic in `Module` itself:

```python
# blaze_nn/module.py -- UNCHANGED
class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError
```

The registered dispatch is invisible to `Module` -- it happens inside the functional API calls that `forward()` makes.

### `Parameter` and State Dict: No Changes

```python
# blaze_nn/module.py -- UNCHANGED
class Parameter:
    # ... shape, dtype, _data, _device_tensor ...
    # No interaction with TTNN dispatch; resolved to device tensors
    # before the compilation pipeline runs.

class Module:
    def state_dict(self):
        # Returns {name: param._data for name, param in self._parameters.items()}
        # No interaction with dispatch.
        ...

    def load_state_dict(self, state_dict):
        # Copies data into Parameter._data fields.
        # No interaction with dispatch.
        ...
```

Parameter ownership, serialization, and device allocation are orthogonal to dispatch. The adapter operates on `MeshProgramDescriptor` objects, which are produced *after* parameters have been resolved to device tensors by the compilation pipeline. The ownership model remains:

- **blaze-nn owns the tensors** (via `Parameter` and `Module.state_dict()`).
- **Blaze compiler borrows the tensors** (via `io_tensors` in `MeshCompiledProgram`).
- **TTNN receives the tensors** (via the registered callable or `generic_op`).
- **The adapter does not copy or reallocate tensors** -- it passes the pre-allocated tensors through to the `GenericMeshProgramFactory`, which uses them as-is.

### `Module`-Level Profiling: New Capability

While `Module` itself is unchanged, the combination of registered dispatch and module structure enables a new profiling pattern. When profiling a `Module.forward()` call, the Tracy timeline shows the module's internal structure:

```text
Tracy Timeline for Attention.forward():
  |-- rmsnorm --|-- matmul(Q) --|-- matmul(K) --|-- matmul(V) --|
  |-- rope(Q) --|-- rope(K) ----|-- sdpa -------|-- matmul(O) --|
```

Tracy's zone nesting (if the caller wraps `forward()` in a Tracy zone) shows which ops belong to which module:

```text
[Attention.forward]
  [blaze::rmsnorm]     6 us
  [blaze::matmul]     45 us   <- Q projection
  [blaze::matmul]     45 us   <- K projection
  [blaze::matmul]     45 us   <- V projection
  [blaze::rope]        1.5 us <- Q rotation
  [blaze::rope]        1.5 us <- K rotation
  [blaze::sdpa]      180 us
  [blaze::matmul]     45 us   <- output projection
```

This nested view requires no changes to blaze-nn. It is a natural consequence of Tracy's zone stacking: the outer `[Attention.forward]` zone is created by the user or a profiling wrapper, and the inner `[blaze::*]` zones are created by the adapter's `TracyOpMeshWorkload` macro.

---

## 7.2.8 Composability: Mixing Registered Blaze Ops with Native TTNN Ops

A real model may mix blaze-nn operations with native TTNN operations. For example, embedding lookups or loss computation might use `ttnn.embedding()` or `ttnn.mse_loss()` while the transformer body uses blaze-nn's `F.linear()`, `F.rmsnorm()`, and `F.sdpa()`.

### Composability Scenario

```python
class TransformerWithEmbedding(Module):
    def forward(self, token_ids, freqs_cis, mask):
        # Native TTNN op: embedding lookup
        x = ttnn.embedding(token_ids, self.embedding_table)

        # blaze-nn ops: transformer layers
        for layer in self.layers:
            x = layer.attention(x, freqs_cis, mask)
            x = layer.mlp(x)

        # blaze-nn op: final norm
        x = F.rmsnorm(x, self.final_norm_weight)

        # Native TTNN op: output projection
        logits = ttnn.matmul(x, self.lm_head_weight)
        return logits
```

### Dispatch Path Comparison

```text
ttnn.embedding(token_ids, self.embedding_table)
  -> registered_operation_t<"ttnn::embedding", EmbeddingDeviceOperation>::invoke(...)
     -> TracyOpMeshWorkload("ttnn::embedding")
     -> [native TTNN dispatch: validate, cache, factory, enqueue]

F.rmsnorm(x, self.final_norm_weight)
  -> _dispatch("rmsnorm", ...) -> ... -> ttnn.blaze.rmsnorm(...)
     -> registered_operation_t<"ttnn::blaze::rmsnorm",
            BlazeDeviceOperationAdapter<RMSNormTag>>::invoke(...)
        -> TracyOpMeshWorkload("ttnn::blaze::rmsnorm")
        -> [adapter dispatch: validate, cache (adapter factory), enqueue]

ttnn.matmul(x, self.lm_head_weight)
  -> registered_operation_t<"ttnn::matmul", MatmulDeviceOperation>::invoke(...)
     -> TracyOpMeshWorkload("ttnn::matmul")
     -> [native TTNN dispatch: validate, cache, factory, enqueue]
```

### Combined Profiler Timeline

```text
|-- ttnn::embedding --|-- blaze::rmsnorm --|-- blaze::matmul --| ... (32 layers)
... |-- blaze::rmsnorm --|-- ttnn::matmul --|
```

All three types of operations -- native TTNN, registered Blaze adapter, and `generic_op` fallback -- appear with distinct names in the same profiler timeline.

### Tensor Compatibility

Native TTNN ops and registered Blaze ops both operate on `ttnn.Tensor` objects. There is no type barrier: the output of `ttnn.embedding()` is a valid input to `F.rmsnorm()`, and vice versa. The adapter wraps the dispatch path, not the data.

| Requirement | Status | Notes |
|---|---|---|
| **Memory layout** | Compatible | Both use tile layout by default. Mixed layouts require explicit `ttnn.to_layout()`. |
| **Memory config** | Compatible | Blaze ops may output to L1 or DRAM. Boundaries may need explicit memory config. |
| **Data type** | Compatible | Both produce standard dtypes (BFloat16, Float32). Explicit `ttnn.typecast()` if needed. |
| **Tensor shape** | Compatible | Blaze and TTNN use compatible shape representations. |
| **Program cache** | Isolated | Blaze adapter ops and native TTNN ops use different `type_hash` seeds. |
| **Graph trace** | Unified | Both appear as named nodes in `GraphTracker`. |
| **Tracy profiling** | Unified | Both produce named Tracy zones in the same timeline. |

---

## 7.2.9 Diagnostic Introspection API

The integration adds diagnostic capabilities that help developers understand the dispatch state at runtime:

```python
# PROPOSED: blaze_nn/_registry.py (new diagnostic functions)

def get_dispatch_info(functional_name: str) -> Dict[str, Any]:
    """Return dispatch information for a blaze-nn functional op.

    Parameters
    ----------
    functional_name : str
        The functional API name (e.g., "linear").

    Returns
    -------
    dict
        Dispatch information including functional_name, blaze_op_name,
        is_registered, dispatch_target, and ttnn_callable.

    Example
    -------
    >>> get_dispatch_info("linear")
    {
        'functional_name': 'linear',
        'blaze_op_name': 'matmul',
        'is_registered': True,
        'dispatch_target': 'ttnn.blaze.matmul',
        'ttnn_callable': <ttnn.blaze.matmul>
    }
    """
    mapping = _REGISTRY[functional_name]
    callable_ = mapping.ttnn_callable
    if callable_ is not None:
        target = f"ttnn.blaze.{mapping.blaze_op_name}"
    else:
        target = "ttnn.generic_op"

    return {
        "functional_name": functional_name,
        "blaze_op_name": mapping.blaze_op_name,
        "is_registered": callable_ is not None,
        "dispatch_target": target,
        "ttnn_callable": callable_,
    }


def get_registration_report() -> Dict[str, Dict[str, Any]]:
    """Return a complete report of all registered blaze-nn ops
    and their dispatch status.

    Returns
    -------
    dict
        Mapping from functional_name to dispatch info dict.
    """
    return {
        name: get_dispatch_info(name)
        for name in sorted(_REGISTRY.keys())
    }


def print_registration_report() -> None:
    """Print a human-readable registration report to stdout.

    Example output:
        blaze-nn Op Dispatch Report
        ===========================
          linear               -> ttnn.blaze.matmul               [REGISTERED]
          rmsnorm              -> ttnn.blaze.rmsnorm               [REGISTERED]
          sdpa                 -> ttnn.blaze.sdpa                  [REGISTERED]
          rope                 -> ttnn.blaze.rope                  [REGISTERED]
          exotic_op            -> ttnn.generic_op                  [GENERIC_OP]
        ---------------------------
        Total: 5 ops, 4 registered, 1 generic_op
    """
    report = get_registration_report()
    registered_count = sum(
        1 for info in report.values() if info["is_registered"]
    )
    generic_count = len(report) - registered_count

    print("blaze-nn Op Dispatch Report")
    print("===========================")
    for name, info in report.items():
        status = "REGISTERED" if info["is_registered"] else "GENERIC_OP"
        print(f"  {name:20s} -> {info['dispatch_target']:30s} [{status}]")
    print("---------------------------")
    print(f"Total: {len(report)} ops, {registered_count} registered, "
          f"{generic_count} generic_op")
```

These functions are useful for verifying which ops use registered dispatch during development, debugging unexpected fallback behavior, and generating reports for migration tracking.

---

## 7.2.10 End-to-End Flow Diagram

The complete flow from `Module.forward()` to device execution, with the proposed changes:

```text
Module.forward()
  -> F.linear(h, wq)
     -> _dispatch("linear", h, wq)
        -> _REGISTRY["linear"] -> OpMapping(blaze_op_name="matmul",
                                             ttnn_callable=ttnn.blaze.matmul)
     -> ComposeTracingContext.dispatch("matmul", [h, wq], {"transpose_b": True}, "4x8")
        -> BlazeCompiler.compile(Matmul, ..., op_name="matmul")
           -> MeshCompiledProgram(op_name="matmul")
              -> _resolve_dispatch("matmul") -> ttnn.blaze.matmul
        -> _run_program(io_tensors, descriptor)
           -> ttnn.blaze.matmul(io_tensors, descriptor)
              -> BlazeDeviceOperationAdapter<MatmulTag>::invoke(...)
                 [Adapter internals: see Chapters 4-5]
              -> launch<Adapter<MatmulTag>>(...)
                 -> GraphTracker::track("ttnn::blaze::matmul")
                 -> compute_program_hash(type_hash<Adapter<MatmulTag>>, ...)
                 -> program_cache lookup -> enqueue_mesh_workload
                 -> TracyOpMeshWorkload("ttnn::blaze::matmul")
  -> [Hardware execution on Tenstorrent device]
```

Op identity flows unbroken from the blaze-nn registry (`"matmul"`) through the Blaze compiler (`op_name="matmul"`) to the TTNN adapter (`"ttnn::blaze::matmul"`), and is preserved in the program cache, graph tracer, and profiler.

---

## 7.2.11 Dispatch Layer Map: After Registration

Reprising the layer map from [Section 7.1.10](01_current_blaze_nn_dispatch.md), now with registration:

| Layer | Function/Method | Op Identity (Before) | Op Identity (After) |
|-------|----------------|:---:|:---:|
| 1. Functional API | `F.linear(h, self.wq)` | Yes: `"linear"` | Yes: `"linear"` |
| 2. Registry Lookup | `_REGISTRY["linear"]` | Yes: `"matmul"` | Yes: `"matmul"` + `ttnn.blaze.matmul` |
| 3. Kwargs Transform | `_linear_kwargs_transform(...)` | N/A | N/A |
| 4. Context Dispatch | `ctx.dispatch("matmul", ...)` | Yes: `"matmul"` | Yes: `"matmul"` |
| 5. BlazeCompiler | `BlazeCompiler().compile(...)` | Yes: `Matmul` class | Yes: `Matmul` class + `op_name` |
| 6. `_run_program()` | `compiled.run(tensors)` | **No**: `generic_op` | **Yes**: `ttnn.blaze.matmul` |
| 7. TTNN Dispatch | Registered callable | **No**: `"ttnn::generic_op"` | **Yes**: `"ttnn::blaze::matmul"` |
| 8. Program Cache | `compute_program_hash(...)` | **No**: shared prefix | **Yes**: `type_hash<Adapter<MatmulTag>>` |
| 9. Tracy Profiler | `TracyOpMeshWorkload(...)` | **No**: `"ttnn::generic_op"` | **Yes**: `"ttnn::blaze::matmul"` |

Op identity now flows unbroken from Layer 1 through Layer 9. The gap at Layers 6--9 has been closed.

---

## 7.2.12 Change Summary

The dispatch mechanics changes are additive: `_REGISTRY` gains a lazily-resolved `ttnn_callable` field per `OpMapping`, `_dispatch()` resolves `ttnn_callable` for diagnostics, and `MeshCompiledProgram._run_program()` calls `self._dispatch_fn` (resolved at init) instead of `ttnn.generic_op` unconditionally. The call signature to TTNN is unchanged -- `(io_tensors, descriptor)` -- only the endpoint differs. The import-time dependency shifts from eager `import ttnn` to a lazy `getattr(ttnn, "blaze", None)` check.

All components not listed in the layer map above are unchanged -- functional API signatures, `kwargs_transform` functions, `TensorProxy`, `Parameter`, `Module`, `state_dict()`/`load_state_dict()`, `GraphTracingContext`, `BlazeNNGraph`, `BlazeOp._class_registry`, the `BlazeCompiler` pipeline engines, and `MeshProgramDescriptor` schema remain unmodified (see [Section 7.2.7](#727-module-integration-what-changes-what-stays) for details).

### Per-File Change Summary

| File | Change | Estimated Lines |
|------|--------|:-:|
| `blaze_nn/_registry.py` | `OpMapping` gains lazy `ttnn_callable`; `_resolve_ttnn_callable()`, `_get_ttnn_blaze_module()`, `reset_dispatch_cache()`, env var check; diagnostic API | ~60 |
| `blaze/compiler.py` | `MeshCompiledProgram.__init__` gains `op_name`; `_resolve_dispatch()` added; `_run_program()` calls `self._dispatch_fn` | ~15 (from [Section 4.3.1](../ch04_adapter_pattern/03_python_dispatch_and_registration.md)) |
| All other blaze-nn files | **Zero changes** (`functional.py`, `module.py`, `tracing.py`) | 0 |
| **Total blaze-nn changes** | | **~60** |

---

## 7.2.13 Testing Strategy

Three categories of tests verify the integration.

### Category 1: Registry-Level Tests

```python
def test_ttnn_callable_resolution():
    """Verify that ttnn_callable resolves correctly for registered ops."""
    from blaze_nn._registry import _REGISTRY
    mapping = _REGISTRY["linear"]
    callable_ = mapping.ttnn_callable
    if hasattr(ttnn, "blaze") and hasattr(ttnn.blaze, "matmul"):
        assert callable_ is not None
        assert getattr(callable_, "__ttnn_operation__", False)
    else:
        assert callable_ is None

def test_fallback_for_unregistered_op():
    """Verify that unregistered ops get ttnn_callable=None."""
    from blaze_nn._registry import OpMapping
    mapping = OpMapping(blaze_op_name="nonexistent_op_xyz_12345")
    assert mapping.ttnn_callable is None

def test_force_generic_env_var(monkeypatch):
    """Verify that BLAZE_NN_FORCE_GENERIC=1 disables all registered dispatch."""
    monkeypatch.setenv("BLAZE_NN_FORCE_GENERIC", "1")
    # Re-import or reset to pick up env var
    from blaze_nn._registry import _REGISTRY, reset_dispatch_cache
    reset_dispatch_cache()
    mapping = _REGISTRY["linear"]
    assert mapping.ttnn_callable is None

def test_cache_reset():
    """Verify that reset_dispatch_cache() clears all cached callables."""
    from blaze_nn._registry import _REGISTRY, reset_dispatch_cache
    mapping = _REGISTRY["linear"]
    _ = mapping.ttnn_callable  # trigger lazy resolution
    reset_dispatch_cache()
    assert not mapping._resolved
```

### Category 2: End-to-End Dispatch Tests

```python
def test_attention_block_dispatch():
    """Verify that all ops in the attention block route through
    registered callables."""
    attention = Attention(dim=4096, n_heads=32)
    attention.load_state_dict(test_weights)

    # Patch ttnn.blaze.* to record calls
    call_log = []
    original_matmul = ttnn.blaze.matmul
    def logging_matmul(*args, **kwargs):
        call_log.append("matmul")
        return original_matmul(*args, **kwargs)
    ttnn.blaze.matmul = logging_matmul

    try:
        output = attention(x, freqs_cis, mask)
        assert call_log.count("matmul") == 4, \
            f"Expected 4 matmul calls, got {call_log.count('matmul')}"
    finally:
        ttnn.blaze.matmul = original_matmul
```

### Category 3: Profiler Verification Tests

```python
def test_profiler_shows_named_ops():
    """Verify that Tracy zones have correct names after registration."""
    import ttnn.profiler as profiler
    profiler.start()

    attention = Attention(dim=4096, n_heads=32)
    attention.load_state_dict(test_weights)
    output = attention(x, freqs_cis, mask)

    events = profiler.stop_and_get_events()
    op_names = [e.name for e in events if e.name.startswith("ttnn::blaze::")]
    assert "ttnn::blaze::matmul" in op_names
    assert "ttnn::blaze::rmsnorm" in op_names
    assert "ttnn::blaze::sdpa" in op_names
    assert "ttnn::blaze::rope" in op_names
    assert "ttnn::generic_op" not in op_names, \
        "All attention ops should route through registered callables"
```

---

## 7.2.14 Migration Sequence

The blaze-nn changes should be deployed after the TTNN-side adapter and registration are in place (Chapters 4--6). The following sequence allows each step to be deployed and validated independently:

| Step | Action | Reversibility |
|:----:|--------|--------------|
| 1 | Add `_ttnn_callable`, `_resolved` fields to `OpMapping` (defaults to `None` / `False`) | Fully reversible -- `None` means no behavior change |
| 2 | Implement `_resolve_ttnn_callable()` and `_get_ttnn_blaze_module()` with `__ttnn_operation__` guard | Fully reversible -- returns `None` if `ttnn.blaze` does not exist |
| 3 | Add `ttnn_callable` property to `OpMapping` and `reset_dispatch_cache()` | Fully reversible -- not invoked unless step 4 is deployed |
| 4 | Wire `ttnn_callable` metadata from `_dispatch()` for diagnostic logging; `ComposeTracingContext.dispatch()` signature is unchanged | Reversible via `BLAZE_NN_FORCE_GENERIC=1` |
| 5 | Add diagnostic API (`get_dispatch_info()`, `get_registration_report()`, `print_registration_report()`) | No behavior change -- purely additive |
| 6 | Verify profiler, cache, and graph output for registered ops | N/A (validation step) |

If step 4 introduces a regression, setting `BLAZE_NN_FORCE_GENERIC=1` reverts all dispatch to `generic_op` without rolling back any code. Steps 1--3 are invisible to dispatch behavior. Step 5 is purely additive. This provides a safe, incremental migration path.

---

| [Current blaze-nn Dispatch Architecture](01_current_blaze_nn_dispatch.md) | [Chapter Index](index.md) | [Chapter 8: Build System, Migration, and Alternatives](../ch08_build_migration_alternatives/index.md) |
|:---|:---:|---:|
