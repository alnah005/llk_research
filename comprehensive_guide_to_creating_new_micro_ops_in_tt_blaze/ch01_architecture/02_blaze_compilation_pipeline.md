# 1.2 The TT-Blaze Compilation Pipeline

## What you will learn

- The three-phase lifecycle of a TT-Blaze program: graph construction, engine pipeline, and JIT compilation with dispatch.
- How `emit()` declarations flow through engines to produce `named_args_generated.h` and `CBDescriptor` lists.
- The JIT compilation model: three compilations per kernel (one per processor), with per-core specialization via CT args.
- The explicit Python-to-C++ CT arg flow -- how a value declared in `emit()` becomes a `constexpr` in the kernel binary.
- How dispatch works through `ttnn.generic_op()` and what the device receives.
- The two compilation modes: graph-based (`BlazeCompiler` with `blaze.fuse()`) and imperative (`FusedProgram` with direct `emit()` calls).

---

## 1.2.1 The three-phase lifecycle

Every TT-Blaze program follows three distinct phases:

```
Phase 1: Graph Construction      Phase 2: Engine Pipeline       Phase 3: JIT + Dispatch
  blaze.fuse() context              CBEngine                      3x compilation
  or FusedProgram.emit()            SemEngine                     (NCRISC, BRISC, TRISC)
                                    RoleEngine                    Kernel codegen
                                    CTArgEngine                   ttnn.generic_op()
        |                               |                               |
  Python declarations             Assign CB IDs             Compile & execute on device
  of ports, CBs, CT args          Assign semaphores
                                  Generate CT arg tuples
```

These phases separate concerns: Phase 1 is about *what* to compute (logical dataflow), Phase 2 is about *how* to map it to hardware resources (CB IDs, semaphore addresses, NOC coordinates), and Phase 3 is about *building and running* the actual kernel binaries.

### Phase 1: Graph construction

TT-Blaze provides two ways to build a program:

**Graph API** (`blaze.fuse()`): You describe a dataflow graph of operations. The `FusionContext` (`blaze/context.py`) captures each op call, creates `OpNode` objects, and wires `Edge` objects between them based on data dependencies:

```python
# blaze/context.py -- FusionContext captures op calls
with blaze.fuse() as ctx:
    act = blaze.mcast(ext_input, grid=sender_grid)
    out = blaze.matmul(act, ext_weights)
# ctx.graph is now a validated BlazeGraph with nodes and edges
```

When you call `blaze.mcast(...)`, it dispatches through `_OpHandle.__call__()` (defined in `blaze/__init__.py`), which delegates to the op class's `graph_call()` method:

```python
# blaze/__init__.py
class _OpHandle:
    def __init__(self, op_cls):
        self._cls = op_cls
        self.emit = op_cls.emit

    def __call__(self, *args, **kwargs):
        return self._cls.graph_call(*args, **kwargs)
```

**Imperative API** (`FusedProgram`): You directly call `emit()` on MicroOp classes, building up a program step by step. This is the more common path for writing new fused ops:

```python
# blaze/fused_program.py -- FusedProgram usage
f = FusedProgram(kernel=None, device=device, math_fidelity=...)
mcast_out = Mcast.emit(f, input_tensor, prefix="act_mcast")
matmul_out = Matmul.emit(f, mcast_out, weights, prefix="matmul")
result = f.build()
result.run()
```

Each `emit()` call directly registers CB descriptors, CT args, and semaphores on the `FusedProgram`. The `FusedProgram` also maintains a *shadow graph* that mirrors the emit sequence, enabling kernel codegen and visualization.

Both paths converge: the graph API ultimately calls `compose()`, which chains `emit()` calls on a `FusedProgram`.

### Phase 2: Engine pipeline

The engine pipeline transforms high-level declarations into concrete, hardware-ready parameters. Four engines process the graph or program state:

| Engine | Source file | Responsibility |
|---|---|---|
| **CBEngine** | `blaze/cb_engine.py` | Assigns sequential CB IDs (0..63) to all data paths: external tensor inputs, inter-op intermediates, internal scratch buffers, outputs. Computes buffer sizes from tensor metadata. |
| **SemEngine** | `blaze/sem_engine.py` | Identifies inter-core synchronization points (mcast, gather, CCL ops) by examining each node's CTArgSchema for `Sem`-kind entries. Assigns sequential global semaphore indices. |
| **RoleEngine** | `blaze/role_engine.py` | Resolves logical core coordinates to physical NOC coordinates, handles grid configuration for multicast ranges, and provides the `GridConfig` that maps logical positions to device topology. |
| **CTArgEngine** | `blaze/ct_args.py` | Walks the graph, looks up each op's `CTArgSchema`, auto-prefixes arg names, and resolves values from CB assignments, semaphore assignments, grid context, and user-supplied overrides. Groups results by RISC processor. |

In the graph-based path, these engines are invoked by `BlazeCompiler._compile_for_device()`:

```python
# blaze/compiler.py -- BlazeCompiler._compile_for_device() (simplified)

# Step 1: Run CB and Sem engines
cb_assignments = CBEngine().assign(graph)
sem_assignments = SemEngine().assign(graph)

# Step 2: Build grid context (NOC coordinate resolution)
grid_context = self._build_grid_context(graph, ctx)

# Step 3: Generate CT arg tuples per RISC
ct_tuples = CTArgEngine().generate_tuples(
    graph, cb_for_ct, sem_for_ct,
    grid_context=grid_context,
    user_overrides=merged_overrides,
    strict=True,
)
```

In the imperative `FusedProgram` path, the engines are not used directly. Instead, each `emit()` call invokes methods like `f.cb_from_tensor()`, `f.cb_scratch()`, `f.semaphore()`, `f.ncrisc_ct_args()`, `f.brisc_ct_args()`, and `f.trisc_ct_args()` to register resources and compile-time arguments. The `FusedProgram` accumulates these into its underlying `BlazeProgram` (`blaze/program.py`).

As a concrete walkthrough, the Mcast op's `emit()` (`blaze/ops/mcast/op.py`) exercises all resource types -- tensor-backed CBs, scratch CBs, semaphores, per-core flags, and both NCRISC and BRISC CT args:

```python
@staticmethod
def emit(f: FusedProgram, src, *, prefix: str, ...) -> CBHandle:
    # 1. Allocate CBs
    src_handle = f.cb_from_tensor(src)        # input CB from tensor
    dst = f.cb_scratch(                        # intermediate CB
        name=BlazeOp.cb_name(prefix, "dst"),
        num_pages=num_pages,
        core_ranges=f.all_cores,
        ...
    )

    # 2. Allocate semaphores
    sender_sem = f.semaphore(f"{prefix}.sender")
    receiver_sem = f.semaphore(f"{prefix}.receiver")

    # 3. Declare per-core flags
    f.per_core_unified_ct_args([
        f.flag(f"{prefix}.is_sender", f.sender_grid),
        f.flag(f"{prefix}.is_receiver", f.mcast_receiver_grid),
        f.flag(f"{prefix}.pop_src", f.all_cores),
        f.flag(f"{prefix}.init_src", f.sender_grid, needs_init),
    ])

    # 4. Declare NCRISC CT args
    f.ncrisc_ct_args([
        (f"{prefix}.src", src_handle),
        (f"{prefix}.receiver_semaphore", receiver_sem),
        (f"{prefix}.dst", dst),
        ...
    ])

    # 5. Declare BRISC CT args
    f.brisc_ct_args([
        (f"{prefix}.dest_noc_start_x", f.noc_start.x),
        (f"{prefix}.sender_semaphore", sender_sem),
        (f"{prefix}.receiver_semaphore", receiver_sem),
        (f"{prefix}.data_size_bytes", data_size_bytes),
        ...
    ])

    return f.output("mcast", dst, prefix=prefix, src=src)
```

When a `CBHandle` is passed as a CT arg value (e.g., `(f"{prefix}.src", src_handle)`), the `FusedProgram` automatically resolves it to the handle's numeric CB ID. Similarly, semaphore handles resolve to L1 addresses.

### Phase 3: JIT compilation and dispatch

The `UnifiedKernelDescriptor` (`blaze/unified_kernel_descriptor.py`) takes the CT arg tuples and kernel source path and constructs three `KernelDescriptor` objects -- one per processor:

```python
# blaze/unified_kernel_descriptor.py
"""Creates reader (NCRISC), writer (BRISC), and compute (TRISC) kernel
descriptors from a single unified kernel source file."""
```

The tt-metal runtime JIT-compiles each kernel descriptor: NCRISC gets compiled with `COMPILE_FOR_NCRISC` defined, BRISC with `COMPILE_FOR_BRISC`, and TRISC with `COMPILE_FOR_TRISC`. Named compile-time arguments become C++ `constexpr` values accessible via the generated `named_args_generated.h` header.

Dispatch happens through `ttnn.generic_op()`:

```python
# blaze/compiler.py -- _run_program()
ttnn.generic_op(io_tensors, descriptor)
```

This single call sends the compiled program to the device, binds the I/O tensors, and triggers execution across all Tensix cores in the specified grid.

## 1.2.2 The Python-to-C++ CT arg flow

Understanding how a CT arg value travels from Python `emit()` to C++ `constexpr` is crucial. Here is the complete chain, traced through a concrete example:

**Step 1: Python `emit()`**

```python
# Step 1: Python emit()
f.trisc_ct_args([
    ("rmsnorm.input", inp),        # CBHandle -> CB ID (e.g., 0)
    ("rmsnorm.num_tiles", 4),      # integer value
    ("rmsnorm.epsilon", 0x358637BD), # float bit-cast to uint32
])
```

**Step 2: Generated header (`named_args_generated.h`)**

```cpp
// Step 2: Generated header (named_args_generated.h)
namespace ct_args {
namespace rmsnorm {
    constexpr uint32_t input = 0;
    constexpr uint32_t num_tiles = 4;
    constexpr uint32_t epsilon = 0x358637BD;
}
}
```

**Step 3: C++ kernel header (`op.hpp`) reads via template parameter `A`**

```cpp
// Step 3: C++ kernel header (op.hpp) reads via template parameter A
template <typename A>
struct ComputeCTArgs {
    static constexpr CB input = A::input;           // 0
    static constexpr uint32_t num_tiles = A::num_tiles; // 4
    static constexpr uint32_t epsilon = A::epsilon;     // 0x358637BD
};
```

**Step 4: Generated kernel instantiation**

```cpp
// Step 4: Generated kernel instantiation
using Rmsnorm = blaze::RMSNorm::Op<ct_args::rmsnorm>;
```

Because these are `constexpr`, they are resolved at compile time. The compiler can unroll loops, evaluate `if constexpr` branches, and perform template specialization based on these values. This is why TT-Blaze kernels have zero runtime arg-parsing overhead.

## 1.2.3 The JIT: three compilations per kernel

The `UnifiedKernelDescriptor` creates three kernel descriptors from one source file:

| Compilation | Define | Processor | CT args it receives |
|---|---|---|---|
| 1 | `COMPILE_FOR_NCRISC` | DM0 (reader) | NCRISC args + unified args |
| 2 | `COMPILE_FOR_BRISC` | DM1 (writer) | BRISC args + unified args |
| 3 | `COMPILE_FOR_TRISC` | TRISC (all 3 sub-cores) | TRISC args + unified args + compute config |

Each compilation produces a separate binary. The `#if defined(COMPILE_FOR_*)` guards select which code paths survive, and the compiler dead-code-eliminates everything else. Because all CT args are `constexpr`, the compiler can:

- Dead-code-eliminate entire `if constexpr` blocks (e.g., `if constexpr (!cta::is_active) return;`).
- Unroll loops with known trip counts.
- Specialize NOC transfers for known sizes.
- Inline CB IDs, semaphore addresses, and tile counts as immediate constants.

### Per-core specialization

Some CT args vary per core. For example, the `is_active` flag is `1` on cores that participate in the op and `0` on all others. This is handled by `UnifiedCompileTimeCoreDescriptor`:

```python
# From blaze/fused_program.py
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_active", active_core_ranges),
])
```

This creates a descriptor that tells the JIT: "on cores in `active_core_ranges`, set `is_active = 1`; on all other cores, set `is_active = 0`." The JIT produces separate binaries for each group of cores (same source, different CT arg values), enabling fine-grained per-core behavior.

For example, if cores $(0,0)$ through $(11,9)$ have `is_active=1` and core $(12,9)$ (the sender) has `is_active=0`, two kernel descriptors are created for each processor -- one for the active region, one for the inactive region. This is how a single fused kernel can have different behavior on sender vs. receiver cores without any runtime branching.

## 1.2.4 Dispatch via `ttnn.generic_op()`

After JIT compilation, the program is dispatched to the device. The dispatch interface is:

```python
ttnn.generic_op(io_tensors, descriptor)
```

Where:
- `io_tensors` is an ordered list of `ttnn.Tensor` objects. Their L1 buffer addresses back the tensor-backed CBs.
- `descriptor` is a `ProgramDescriptor` (single device) or `MeshProgramDescriptor` (multi-device).

What the device receives:

1. **Kernel binaries**: Three compiled binaries (NCRISC, BRISC, TRISC) for each kernel in the program.
2. **CB descriptors**: Hardware register configurations for each circular buffer -- CB ID, L1 address, total size, page size, data format, tile shape, and core ranges.
3. **I/O tensor addresses**: L1 buffer addresses for input and output tensors, which back the tensor-backed CBs.
4. **Semaphore addresses**: L1 addresses for global semaphores used by inter-core ops.

For mesh (multi-device) execution, `BlazeCompiler.compile()` iterates over all devices in the mesh, compiles a per-device `ProgramDescriptor`, and assembles them into a `MeshProgramDescriptor`:

```python
# blaze/compiler.py -- per-device mesh compilation
mesh_pd = ttnn.MeshProgramDescriptor()
for idx in range(num_devices):
    row, col = divmod(idx, self._num_cols)
    merged_args = {
        "mesh_coord": (row, col),
        "mesh_shape": (self._num_rows, self._num_cols),
        **(user_args or {}),
    }
    compiled = self._compile_single(...)
    coord = ttnn.MeshCoordinate(row, col)
    mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = compiled.program_descriptor
```

Each device can have different CT arg values (e.g., different `mesh_coord` tuples, different NOC addresses), but they all share the same kernel source.

## 1.2.5 The two compilation modes

TT-Blaze supports two equivalent paths to arrive at a compiled program:

### Mode 1: Graph-based (`BlazeCompiler` with `blaze.fuse()`)

```python
# Phase 1: Build graph
with blaze.fuse() as ctx:
    act = blaze.mcast(ext_input, grid=sender_grid)
    result = blaze.matmul(act, ext_weights)

# Phase 2 + 3: Compile and dispatch
compiler = BlazeCompiler(mesh_device)
program = compiler.compile(
    graph=ctx.graph,
    tensors={"input": input_tensor, "weights": weight_tensor},
    output_tensor=output_tensor,
    user_args={"k_num_tiles": 4},
)
output = program.run()
```

The graph-based mode uses the full engine pipeline: `CBEngine` assigns CB IDs, `SemEngine` assigns semaphore indices, and `CTArgEngine` resolves all compile-time args. When the graph contains a single node whose op is registered as a `FusedOpConfig`, the compiler bypasses the engine pipeline and delegates to `_compile_fused_op()`, which internally creates a `FusedProgram` and calls the op's `compose()` method:

```python
# blaze/compiler.py -- _compile_single()
if len(graph.nodes) == 1:
    config = get_fused_op_config(graph.nodes[0].spec.name)
    if config is not None:
        return self._compile_fused_op(graph, tensors, output_tensor, config, ...)
```

### Mode 2: Imperative (`FusedProgram` with direct `emit()` calls)

```python
# Combined Phase 1 + 2
f = FusedProgram(kernel=None, device=device, math_fidelity=ttnn.MathFidelity.LoFi)
mcast_out = Mcast.emit(f, input_tensor, prefix="act_mcast")
matmul_out = Matmul.emit(f, mcast_out, weight_tensor, prefix="matmul")

# Phase 3
result = f.build()
output = result.run()
```

When `kernel=None` is passed, `FusedProgram.build()` uses the shadow graph to auto-generate the kernel source via `blaze/kernel_codegen.py`.

Both modes ultimately produce the same artifact: a `ProgramDescriptor` with three kernel binaries, CB descriptors, and an ordered tensor list for `ttnn.generic_op()`.

### When to use which mode

| Scenario | Recommended mode |
|---|---|
| Single MicroOp or simple FusedOp | Graph-based (`blaze.fuse()` + `BlazeCompiler`) |
| Complex composition with many MicroOps | Imperative (`FusedProgram` + chained `emit()` calls) |
| Multi-device mesh execution | Graph-based (BlazeCompiler handles mesh decomposition) |
| Inside a `FusedOp.compose()` method | Imperative (fine-grained CB control) |

In practice, most fused ops go through the `FusedProgram` path, even when invoked via the graph API. The graph API provides a cleaner user-facing interface, while `BlazeCompiler._compile_fused_op()` bridges to the imperative path internally.

## 1.2.6 Kernel codegen: from graph to C++

When a fused op does not provide a handwritten kernel source, TT-Blaze's kernel codegen system (`blaze/kernel_codegen.py`) generates the C++ kernel automatically from the `BlazeGraph` (or shadow graph).

The codegen walks the graph in topological order, looks up each op's `PhaseInfo` (registered during op registration), and emits:

1. **Include directives**: The aggregator header `ops.hpp` which includes all per-op kernel headers.
2. **Type aliases**: One `using` declaration per op phase, mapping the op's C++ struct to its CT arg namespace.
3. **`kernel_main()` body**: Sequential op invocations in topological (emit) order, each with `init()` / `operator()()` / `teardown()` lifecycle calls.

```cpp
// Auto-generated by blaze.kernel_codegen
#include "blaze/kernels/ops.hpp"
#include "blaze/kernels/kernel_op_api.hpp"
#include "blaze/kernels/kernel_utils.hpp"

using ActMcast = blaze::Mcast::Op<ct_args::act_mcast>;
using Matmul = blaze::Matmul::Op<ct_args::matmul>;
using Rmsnorm = blaze::RMSNorm::Op<ct_args::rmsnorm>;

void kernel_main() {
#if defined(COMPILE_FOR_TRISC)
    deepseek_compute_kernel_init();
#endif

    {
        DeviceZoneScopedN("ACT_MCAST");
        ActMcast act_mcast;
        act_mcast.init();
        act_mcast();
        act_mcast.teardown();
    }

    {
        DeviceZoneScopedN("MATMUL");
        Matmul matmul;
        matmul.init();
        matmul();
        matmul.teardown();
    }

    {
        DeviceZoneScopedN("RMSNORM");
        Rmsnorm rmsnorm;
        rmsnorm.init();
        rmsnorm();
        rmsnorm.teardown();
    }
}
```

The `PhaseInfo` metadata for each op controls codegen behavior:

```python
# blaze/kernel_codegen.py
@dataclass
class PhaseInfo:
    cpp_type: str          # e.g., "blaze::Mcast::Op"
    alias_suffix: str      # e.g., "Mcast" -> used in type alias
    has_init_teardown: bool # whether to emit init()/teardown() calls
    setup_method: str | None  # optional setup method name
    init_is_empty: bool    # skip init() if body is empty
    teardown_is_empty: bool # skip teardown() if body is empty
```

The `init_is_empty` and `teardown_is_empty` flags are auto-detected by the C++ parser (`blaze/cpp_parser.py`), which inspects the op header's `init()` and `teardown()` method bodies. If they contain only whitespace or comments, the codegen skips emitting those calls -- eliminating the function call overhead entirely.

Generated kernel files are named by content hash for caching:

```python
# blaze/kernel_codegen.py -- content-hashed output
out_path = out_dir / f"{label}_{content_hash}.cpp"
```

## Reference: key source files

| File | Purpose |
|------|---------|
| `blaze/compiler.py` | `BlazeCompiler` -- graph-based compilation, mesh decomposition, engine orchestration |
| `blaze/fused_program.py` | `FusedProgram` -- imperative composition context, CB/CT arg/semaphore registration |
| `blaze/cb_engine.py` | `CBEngine` -- auto-assigns CB IDs, computes sizes and lifetimes |
| `blaze/sem_engine.py` | `SemEngine` -- identifies inter-core sync points, assigns semaphore indices |
| `blaze/role_engine.py` | `GridConfig` -- device grid layout, core classification, NOC coordinate resolution |
| `blaze/ct_args.py` | `CTArgEngine` + `CTArgSchema` -- typed compile-time argument generation and validation |
| `blaze/kernel_codegen.py` | Generates C++ kernel source from `BlazeGraph`, manages `PhaseInfo` registry |
| `blaze/context.py` | `FusionContext` / `blaze.fuse()` -- graph construction context manager |
| `blaze/program.py` | `BlazeProgram` -- low-level program descriptor builder |
| `blaze/unified_kernel_descriptor.py` | Three-way kernel descriptor construction |
| `blaze/graph.py` | `BlazeGraph`, `OpSpec`, `OpNode`, `Edge` data structures |

## Key takeaways

1. **Three phases, cleanly separated:** Graph construction is pure Python. The engine pipeline assigns resources (CB IDs, semaphore addresses, CT args). The JIT compiles and dispatches. Each phase can be tested independently.

2. **Engines automate the tedious parts:** `CBEngine` assigns buffer IDs, `SemEngine` assigns semaphore indices, `CTArgEngine` generates scoped, typed compile-time arguments. You declare intent in `emit()`; the engines handle the plumbing.

3. **Three-way JIT compilation:** Every kernel is compiled three times (NCRISC, BRISC, TRISC). Compile-time arguments are `static constexpr`, enabling full dead-code elimination and constant folding per processor.

4. **`emit()` is the universal building block:** Whether you use the graph API or the imperative API, `emit()` is the function that defines what an op does. The graph API's `compose()` simply maps tensor names to `emit()` arguments.

5. **Kernel codegen is automatic:** When a fused op does not provide a handwritten kernel, `blaze/kernel_codegen.py` generates the C++ source from the graph, using `PhaseInfo` metadata registered by each op. Generated files are content-hash cached.

6. **`ttnn.generic_op()` is the final dispatch:** The compiled program descriptor and I/O tensors are sent to the device (or mesh) via this single API call.

---

**Next:** [`03_micro_op_vs_fused_op.md`](./03_micro_op_vs_fused_op.md)
