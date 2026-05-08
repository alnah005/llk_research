# 01 -- Blaze Op Architecture and Compilation Pipeline

This file covers the complete programming model for TT-Blaze kernel composition: the three-tier op hierarchy that structures every computation (BlazeOp, FusedOp, MicroOp), the port and compile-time argument systems that declare kernel interfaces, the FusedProgram container that orchestrates multi-op programs with circular buffer management and semaphore allocation, the DeviceContext that parameterizes per-device compilation, and the end-to-end pipeline from op definition through graph construction to device execution via `ttnn.generic_op()`. This is the lowest-level user-facing abstraction in TT-Blaze -- everything above (Symbiote module replacement from [Chapter 2](../ch02_symbiote_core/index.md), distributed sharding from [Chapter 3](../ch03_symbiote_multi_device/index.md)) ultimately generates programs through this layer.

---

## 1. The Three-Tier Op Hierarchy

Every computation in TT-Blaze is expressed as an op class. The class attributes ARE the op definition -- there are no separate configuration files or schema registrations. The hierarchy has three tiers:

```
                +------------------+
                |     BlazeOp      |   Base class: ports, ct_args,
                |  (abstract base) |   register(), emit(), compose()
                +--------+---------+
                         |
            +------------+------------+
            |                         |
    +-------+--------+      +--------+-------+
    |     FusedOp     |      |    MicroOp     |
    +----------------+      +----------------+
    | compose() chains|      | emit() wraps a |
    | MicroOp.emit()  |      | C++ kernel .hpp|
    | calls into one  |      | header; auto-  |
    | fused kernel    |      | derives ct_args|
    +----------------+      +----------------+

  Examples:                     Examples:
  - MLA (50+ micro ops)        - Matmul
  - MoE (15+ micro ops)        - Mcast
  - DenseMLP (12 micro ops)    - RMSNorm
  - GLMMoE                     - Gather
  - PostSdpa                   - AllReduce
```

[blaze/blaze_op.py:BlazeOp, lines 186-401]

### 1.1 BlazeOp: The Base Class

`BlazeOp` [blaze/blaze_op.py:BlazeOp, line 186] provides the foundation every op shares:

```python
class BlazeOp:
    # ---- Identity ----
    name: str = ""          # Op identifier, e.g. "matmul"
    kernel: str = ""        # Path to kernel source or .hpp header

    # ---- Compute config ----
    math_fidelity: str = "HiFi4"       # "HiFi4" or "LoFi"
    math_approx_mode: bool = False     # True for SiLU/approximation ops

    # ---- Phase metadata (MicroOps) ----
    op_class: str = ""                 # C++ struct name
    is_inter_core: bool = False        # True for gather, mcast, CCL
    has_init_teardown: bool = False     # True if kernel has init/teardown
    supports_pop_control: bool = False  # True if kernel supports pop flags

    # ---- Compile-time args ----
    ct_args: list = []     # List of CompileTimeArg declarations

    # ---- Core interface ----
    @staticmethod
    def emit(f, *args, **kwargs):
        """Add this op to a FusedProgram. THE op definition."""
        raise NotImplementedError

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        """Compiler entry point: map tensor dict -> emit(). Fused ops only."""
        raise NotImplementedError

    @classmethod
    def register(cls):
        """Auto-generate OpSpec + CTArgSchema + FusedOpConfig from class attrs."""
        ...
```

Key design principle: BlazeOp uses class-level attributes, not instance attributes. You never instantiate an op -- you call its class methods (`emit()`, `compose()`, `register()`).

The `register()` method [blaze/blaze_op.py:BlazeOp.register, line 267] is the bridge from class definition to runtime registration. It:

1. Extracts port descriptors (Input, Output, Internal) from class attributes
2. Builds an `OpSpec` and registers it in the global `_OP_REGISTRY`
3. Converts `ct_args` into a `CTArgSchema` and registers it
4. Auto-registers a `FusedOpConfig` if `compose()` is overridden
5. Registers `PhaseInfo` for kernel code generation if `phase` is set

### 1.2 FusedOp: Composition of MicroOps

`FusedOp` [blaze/blaze_op.py:FusedOp, line 408] represents a multi-stage kernel built by chaining MicroOp `emit()` calls. The developer writes `compose()` -- the compiler entry point that receives a `FusedProgram`, input tensors, and user arguments, then orchestrates the pipeline:

```python
class FusedOp(BlazeOp):
    """Fused op compiled via composition.

    Required:
        name                    -- identity
        Input()/Output() ports  -- interface
        compose()               -- compiler entry point

    Optional:
        emit()                  -- if the fused op is itself reusable
        kernel                  -- handwritten kernel path; if omitted,
                                   the kernel is auto-generated
    """
```

**Enforcement:** `FusedOp.__init_subclass__` raises `TypeError` if a concrete subclass (one with `name` set) fails to override `compose()`. This catches missing implementations at import time, not at runtime.

A typical FusedOp chains 5-15 MicroOps. For example, the MoE fused op [blaze/ops/moe/op.py:MoE] composes:

```
RMSNorm -> Mcast -> MoEGate -> RoutedExpert -> SharedExpert -> ResidualAdd -> ReduceToOne
```

Each MicroOp's `emit()` adds its CBs, CT args, and role flags to the shared FusedProgram. The entire chain compiles into a single kernel dispatch -- no inter-op synchronization overhead.

### 1.3 MicroOp: C++ Kernel-Backed Building Blocks

`MicroOp` [blaze/blaze_op.py:MicroOp, line 432] is the leaf tier, backed by a C++ kernel header (`kernels/op.hpp`). The Python class provides identity and ports; everything else -- ct_args, phase info, init/teardown flags -- can be auto-derived from the C++ header:

```python
class MicroOp(BlazeOp):
    """Micro op backed by a C++ Op struct.

    Required:
        name        -- op identifier
        emit()      -- composition building block
        compose()   -- compiler entry point

    Auto-derived from kernel header:
        op_class, ct_args, is_inter_core, has_init_teardown,
        supports_pop_control, pop_flags, phase
    """
```

**Auto-derivation flow** [blaze/blaze_op.py:MicroOp._auto_derive_from_kernel_hpp, line 518]:

```
MicroOp.register()
   |
   +-> _find_kernel_hpp()  -- locates kernels/op.hpp relative to op module
   |
   +-> parse_op_hpp(path)  -- C++ parser extracts struct name, CT fields,
   |                          RISC annotations, init/teardown signatures
   |
   +-> _resolve_ct_arg_kind()  -- maps parsed CT args to kind objects:
   |     CB(port), Sem(protocol), Grid(), Derived(), Param(), PerCore()
   |
   +-> Sets cls.ct_args, cls.op_class, cls.kernel, cls.phase
   |
   +-> BlazeOp.register()  -- standard OpSpec + schema registration
```

This means a MicroOp developer writes the C++ kernel header ONCE. The Python class needs only `name`, `emit()`, `compose()`, and port descriptors -- the rest is auto-extracted.

Concrete example -- the Matmul MicroOp:

```python
class Matmul(MicroOp):
    name: str = "matmul"
    math_fidelity: str = "LoFi"

    in0: Input = Input()       # activations
    in1: Input = Input()       # weights
    out: Output = Output()     # result

    @staticmethod
    def emit(f: FusedProgram, in0, in1, *, prefix="matmul", pop_in0=True) -> CBHandle:
        # Resolve activations (CBHandle or tensor)
        if isinstance(in0, CBHandle):
            in0_handle = require_fifo_handle(in0, "Matmul.in0")
        else:
            in0_handle = f.cb_from_tensor(in0)

        # Resolve weights with direct L1 addressing
        in1_cb, weights_l1_address = resolve_weight_direct_address(f, in1, ...)

        # Allocate output scratch CB
        out = f.cb_scratch(name=BlazeOp.cb_name(prefix, "out"), ...)

        # Set per-core flags
        f.per_core_unified_ct_args([
            f.flag(f"{prefix}.is_active", active_ranges),
            f.flag(f"{prefix}.pop_in0", active_ranges, pop_in0),
        ])

        # Set RISC-specific CT args
        f.ncrisc_ct_args([(f"{prefix}.k_num_tiles", k_num_tiles), ...])
        f.trisc_ct_args([(f"{prefix}.in1", in1_cb), (f"{prefix}.out", out), ...])

        return f.output("matmul", out, core_ranges=active_ranges, prefix=prefix)

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        cls.emit(f, tensors["in0"], tensors["in1"],
                 prefix=user_args.get("prefix", "matmul"))
```

---

## 2. Port Descriptors

Port descriptors declare the op's tensor interface. They are Python descriptor objects that use `__set_name__` to capture their attribute name automatically:

```python
class Input:
    """Input port descriptor."""
    def __init__(self):
        self.name: str = ""
    def __set_name__(self, owner, name):
        self.name = name

class Output:
    """Output port descriptor."""
    def __init__(self):
        self.name: str = ""
    def __set_name__(self, owner, name):
        self.name = name

class Internal:
    """Internal scratch buffer descriptor."""
    def __init__(self, num_pages: int = 1):
        self.name: str = ""
        self.num_pages = num_pages
    def __set_name__(self, owner, name):
        self.name = name
```

[blaze/blaze_op.py:Input, line 62; Output, line 73; Internal, line 82]

Usage in an op class:

```python
class AllReduce(MicroOp):
    name: str = "all_reduce"

    sender_cb_in0: Internal = Internal()    # scratch buffer
    receiver_cb_in0: Input = Input()        # intermediate input
    receiver_cb_in1: Input = Input()        # activation input
    receiver_cb_out0: Output = Output()     # reduced output
    receiver_cb_residual: Input = Input()   # residual connection
```

At registration time, `_get_input_names()`, `_get_output_names()`, and `_get_internal_cbs()` walk `vars(cls)` to collect all ports. These port names flow into:
- The `OpSpec` (for graph validation)
- The `CTArgSchema` (CB-kind CT args reference ports by name)
- The `CBEngine` (assigns CB IDs to each port)
- The `FusedProgram` (maps tensor names to CB handles)

> **Warning:** Port names are positional in `graph_call()` -- the order they appear in the class body determines argument order.

---

## 3. Compile-Time Arguments (ct_args)

Compile-time arguments are constants baked into the kernel at compile time. They tell each RISC processor where to find CBs, semaphores, grid coordinates, and user-supplied parameters. The system is declared via `CompileTimeArg` dataclasses:

```python
@dataclass(frozen=True)
class CompileTimeArg:
    name: str                           # e.g. "out_w"
    type: Type                          # Type.UINT32 or Type.BOOL
    kind: CB | Sem | Grid | Derived | Param | PerCore
    riscs: Risc                         # Flag enum, e.g. Risc.TRISC | Risc.NCRISC
```

[blaze/blaze_op.py:CompileTimeArg, line 142]

### 3.1 Type Enum

```python
class Type(str, Enum):
    UINT32 = "uint32_t"
    BOOL = "bool"
```

### 3.2 Kind Types -- Where Values Come From

Each kind specifies HOW the CT arg value is derived at compile time:

```
+-------------+--------------------------------------------+-------------------+
| Kind        | Resolved From                              | Example           |
+-------------+--------------------------------------------+-------------------+
| CB(port)    | CB engine assignment for the named port    | CB(cls.in0)       |
| Sem(proto)  | Semaphore engine assignment                | Sem(SENDER)       |
| Grid()      | NOC coordinate translation                 | dest_noc_x        |
| Derived()   | Runtime-computed value from user_args      | num_tiles         |
| Param()     | User-supplied parameter                    | epsilon           |
| PerCore()   | Per-core varying value (resolved at assembly)| sender_idx       |
+-------------+--------------------------------------------+-------------------+
```

### 3.3 RISC Processor Targeting

The `Risc` flag enum controls which RISC processors receive each CT arg:

```python
class Risc(Flag):
    NCRISC = auto()    # Network/DMA RISC (data movement)
    BRISC = auto()     # Base RISC (data movement, control)
    TRISC = auto()     # Tensor RISC (compute)
```

[blaze/blaze_op.py:Risc, line 47]

The three RISC processors have distinct roles on Tenstorrent hardware:

```
+----------+--------------------------------+---------------------------+
| RISC     | Primary Role                   | CT Arg Examples           |
+----------+--------------------------------+---------------------------+
| NCRISC   | NOC data movement, DRAM reads  | CB IDs, weight addresses  |
| BRISC    | NOC data movement, fabric I/O  | CB IDs, fabric RT args    |
| TRISC    | Compute (matmul, reduce, etc.) | CB IDs, tile counts       |
+----------+--------------------------------+---------------------------+
```

A single CT arg can target multiple RISCs via flag combination: `Risc.TRISC | Risc.NCRISC`. The `CTArgEngine` expands each multi-RISC declaration into per-RISC entries at schema registration time.

### 3.4 The CT Arg Resolution Pipeline

The `CTArgEngine` [blaze/ct_args.py:CTArgEngine, line 116] walks a `BlazeGraph` and for each op node:

1. Looks up the op's registered `CTArgSchema`.
2. Computes a prefix from the node's `ct_prefix` kwarg or the op spec name.
3. Resolves each arg value from CB assignments, semaphore assignments, grid context, or user overrides.
4. Validates no name collisions within a RISC (two args with the same prefixed name).
5. Groups results by RISC for `UnifiedKernelDescriptor` consumption.

```
  Graph Node "matmul_1"          CTArgSchema for "matmul"
  +-----------------------+      +-------------------------+
  | spec.name = "matmul"  |      | op_name = "matmul"      |
  | ct_prefix = "gate"    | ---> | args:                   |
  | cb_assignments = {...} |      |   in0: CB(in0), trisc   |
  +-----------------------+      |   k_num_tiles: Derived   |
                                 +-------------------------+
                                          |
                                          v
                                 Prefixed CT Args:
                                   gate.in0 = 3 (cb_id)
                                   gate.k_num_tiles = 16
```

The prefix mechanism is critical for multi-instance ops. When a fused op contains two matmuls (gate and up), each gets a distinct prefix. Shared prefixes (e.g., gate and up matmul sharing `"gu"` for merged weight access) are tracked by the engine and deduplicated [blaze/ct_args.py:CTArgEngine.generate, line 157].

---

## 4. The Op Lifecycle

### Phase 1: Class Definition and Registration

```python
# In blaze/ops/matmul/op.py:
class Matmul(MicroOp):
    name: str = "matmul"
    in0: Input = Input()
    in1: Input = Input()
    out: Output = Output()

    @staticmethod
    def emit(f, activation_cb, weight_tensor, *, prefix="matmul", ...):
        ...

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        cls.emit(f, tensors["in0"], tensors["in1"], ...)

# At import time:
Matmul.register()
```

`register()` [blaze/blaze_op.py, line 267] triggers:

```
  register()
      |
      +-- _get_input_names()   -> ["in0", "in1"]
      +-- _get_output_names()  -> ["out"]
      +-- _get_internal_cbs()  -> []
      |
      +-- register_op(OpSpec(
      |       name="matmul",
      |       kernel_path="blaze/ops/matmul/kernels/op.hpp",
      |       input_ports=[TensorPort("in0"), TensorPort("in1")],
      |       output_ports=[TensorPort("out")],
      |       preferred_math_fidelity="LoFi",
      |       ...
      |   ))
      |
      +-- BlazeOp._class_registry["matmul"] = Matmul
      |
      +-- (if ct_args declared)
      |       register_ct_schema(CTArgSchema(...))
      |
      +-- (if phase declared)
      |       register_phase_info(PhaseInfo(...))
      |
      +-- (if compose() overridden)
              register_fused_op(cls.name, FusedOpConfig(
                  compose_fn=cls.compose,
                  compute_config=ComputeConfigDescriptor(...),
                  kernel=cls.kernel or None,
              ))
```

For MicroOps, `register()` first calls `_auto_derive_from_kernel_hpp()` to parse the C++ header and populate ct_args, phase, and metadata before delegating to `BlazeOp.register()`.

**Auto-discovery:** In practice, ops are not registered one-by-one. The `register_all()` function in `blaze/ops/__init__.py` auto-discovers and registers every op at import time:

```python
def register_all():
    for info in pkgutil.iter_modules(__path__):
        mod = importlib.import_module(f".{info.name}.op", __package__)
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, BlazeOp) and obj.name:
                obj.register()
```

### Phase 2: Graph Construction

The `FusionContext` [blaze/context.py:FusionContext, line 72] captures op calls inside a `with blaze.fuse()` block:

```python
with blaze.fuse() as ctx:
    ext_act = ExternalTensor("activation")
    ext_weight = ExternalTensor("weight")
    out = Matmul.graph_call(ext_act, ext_weight, grid=[(0, 0)])

graph = ctx.graph  # BlazeGraph with nodes, edges, tensor_to_ports
```

Graph validation runs at context exit, checking for duplicate node IDs, invalid port references, and cycles (via topological sort).

### Phase 3: Composition and Execution

The two methods serve different callers:

- **`compose(cls, f, tensors, output, user_args)`** -- The compiler's entry point. Receives named tensors from the graph and forwards them to `emit()`.
- **`emit(f, ...)`** -- The actual composition logic. Allocates CBs, sets CT args, creates scratch buffers, and returns a `CBHandle` to its output.

For FusedOps, `compose()` chains multiple MicroOp `emit()` calls:

```python
class MoE(FusedOp):
    @classmethod
    def compose(cls, f, tensors, output, user_args):
        cls.emit(f, act=tensors["act"], ...)

    @staticmethod
    def emit(f, act, rmsnorm_gamma, gate_weight, ...):
        # Phase 1: Normalize
        norm_out = RMSNorm.emit(f, act, rmsnorm_gamma, ...)

        # Phase 2: Broadcast activation
        mcast_out = Mcast.emit(f, norm_out, ...)

        # Phase 3: Routed expert branch
        routed = RoutedExpert.emit(f, mcast_out, ...)

        # Phase 4: Shared expert branch
        shared = SharedExpert.emit(f, mcast_out, ...)

        # Phase 5: Combine
        combined = ReduceToOne.emit(f, routed, shared, ...)
        ResidualAdd.emit(f, combined, act, ...)
```

---

## 5. FusedProgram: The Composition Container

`FusedProgram` [blaze/fused_program.py:FusedProgram, line 391] is the central context object that MicroOp `emit()` calls interact with. It holds the device context, grid configuration, underlying `BlazeProgram`, and all state accumulated during composition.

### 5.1 Construction and Grid Setup

```python
f = FusedProgram(
    kernel="blaze/ops/moe/kernels/fused_moe.cpp",
    device=device,
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    dst_full_sync_en=False,
    name="moe",
    noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
)
```

On construction, FusedProgram:

1. Creates a `DeviceContext` from the device (grid size, NOC translation)
2. Computes grid layouts:
   - `f.sender` -- the sender core (last col, last row)
   - `f.all_cores` -- full grid as a single CoreRange (0,0)-(grid_cols-1, grid_rows-1), used for mcast
   - `f.matmul_cores` -- compute cores (excluding DRAM workers, phantoms, sender)
   - `f.mcast_receiver_grid` -- all cores minus sender
3. Pre-computes NOC coordinates for the sender and grid corners
4. Creates the underlying `BlazeProgram` with the kernel path and compute config
5. Initializes a shadow `BlazeGraph` for annotation capture
6. Initializes mesh context slots (`mesh_coord`, `mesh_shape`, `mesh_device`)

### 5.2 Circular Buffer Management

FusedProgram provides several CB allocation methods:

```
+---------------------------+----------------------------------------------+
| Method                    | Purpose                                      |
+---------------------------+----------------------------------------------+
| cb_from_tensor(tensor)    | Tensor-backed CB (L1 buffer from tensor)     |
| cb_from_view(view)        | CB from an OverlappedView sub-region         |
| cb_scratch(name, ...)     | Intermediate scratch CB (not tensor-backed)   |
| cb_output(tensor)         | Output tensor-backed CB                      |
| cb_from_tensor_overlapped | CB for a sub-region of a sharded tensor      |
+---------------------------+----------------------------------------------+
```

Each method returns a `CBHandle` [blaze/cb_handle.py:CBHandle, line 35]:

```python
@dataclass(eq=False)
class CBHandle:
    cb_id: int
    num_pages: int
    page_size: int
    core_ranges: ttnn.CoreRangeSet
    data_format: object
    tile_desc: object
    byte_offset: int = 0
    backing_tensor: object = None
    access_mode: CBAccessMode = CBAccessMode.FIFO
```

CBHandles support integer coercion (`int(handle)` returns `cb_id`), so they can be used directly in CT arg tuples.

**CB Access Modes:** FusedProgram supports two access modes:
- **FIFO** (default) -- Standard CB wait/pop semantics. Consumers call `cb_wait_front()` / `cb_pop_front()`.
- **DIRECT_ADDRESS** -- Consumers read via `buffer_address()`. Used for weight tensors that are read directly from L1 without CB flow control.

**Disjoint-core CB sharing:** When two ops run on non-overlapping core sets, FusedProgram can assign the same CB ID to both, reducing total CB count. The `_try_reuse_cb_id()` method [blaze/fused_program.py:FusedProgram._try_reuse_cb_id, line 727] indexes CBs by `(data_format, tile_shape, page_size)` and checks for disjoint grids. This is critical for staying under the Blackhole hardware limit of 64 CBs per program.

**CB lifetime tracking:** Every CB allocation and usage is timestamped by `_op_index`. The `_cb_lifetimes` dict maps `cb_id -> (first_use, last_use)`. This enables the multi-phase CB reconfig system (see Section 5.5).

**OverlappedView:** Multiple logical sub-tensors can share a single physical L1 buffer via `OverlappedView` [blaze/fused_program.py:OverlappedView, line 220]:

```python
@dataclass(frozen=True)
class OverlappedView:
    tensor: ttnn.Tensor        # fused backing tensor
    shard_shape: tuple[int, int]
    core_ranges: ttnn.CoreRangeSet
    dtype: ttnn.DataType
    tile: object
    byte_offset: int           # offset within each core's shard
    total_size: int            # bytes per shard for this view
```

Multiple OverlappedViews can point at the same fused tensor, each with different core_ranges and byte_offsets. This enables packing multiple weight matrices into a single L1 allocation:

```
  Fused L1 Buffer (per-core shard)
  +-------------------------------------------+
  | View A: q_a_proj weights                   |  byte_offset=0
  | (cores 0-59, 7168 x 1536 per shard)       |
  +-------------------------------------------+
  | View B: kv_a_proj weights                  |  byte_offset=7168*1536*2
  | (cores 60-89, 7168 x 576 per shard)       |
  +-------------------------------------------+
  | View C: gate weight                        |  byte_offset=...
  | (cores 90-98, 7168 x 256 per shard)       |
  +-------------------------------------------+
```

This is used by ops like GLM 5.1's fused projection to pack all four weight matrices into one bfloat8 tensor with byte-offset views on disjoint cores, eliminating redundant L1 usage.

### 5.3 Semaphore Allocation

FusedProgram provides two semaphore allocation mechanisms:

```python
# Named mesh-global semaphore (cross-device rendezvous)
sender_sem = f.semaphore(f"{prefix}.sender")

# Program-local semaphore (per-device slot-based sync)
local_sem = f.semaphore(program_semaphore=True, core_ranges=worker_grid)
```

[blaze/fused_program.py:FusedProgram.semaphore, line 560]

Named semaphores are deduplicated across FusedProgram instances sharing the same `_sem_dict` (injected by BlazeCompiler for cross-device consistency). Program semaphores are per-device slot-based and can be scoped to specific core ranges.

### 5.4 CT Arg Emission

`emit()` functions set CT args via FusedProgram wrapper methods:

```python
# All three RISCs
f.unified_ct_args([
    (f"{prefix}.in0_cb", in0_handle),       # CBHandle -> int(cb_id)
    (f"{prefix}.out_cb", out_handle),
    (f"{prefix}.k_num_tiles", k_num_tiles),
])

# NCRISC only
f.ncrisc_ct_args([
    (f"{prefix}.weights_l1_address", weights_l1_address),
])

# Per-core flags (CoreCTArgs)
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_active", active_ranges),
    f.flag(f"{prefix}.is_sender", f.sender_grid),
])
```

These wrapper methods forward to `BlazeProgram` AND capture values in a shadow graph for visualization and debugging.

### 5.5 Multi-Phase CB Reconfiguration

Complex fused ops may need more CBs than the hardware supports (64 on Blackhole). The `reconfig()` method [blaze/fused_program.py:FusedProgram.reconfig, line 1876] marks phase boundaries. CBs from different phases that have non-overlapping lifetimes can share the same hardware CB ID:

```
Phase 0: RMSNorm + Mcast (CBs 0-8)
   |
   reconfig()  <-- phase boundary
   |
Phase 1: SDPA + PostSdpa (CBs 0-5 reused, 9-12 new)
```

The `_build_multi_phase()` method applies interval coloring to remap CB IDs, and injects `CbReconfig` operations at phase boundaries to update the hardware CB configuration.

### 5.6 MultiOutput for Multi-Output Ops

Some ops produce multiple outputs (e.g., `DeepseekMoeGate` produces both scores and indices). `MultiOutput` [blaze/fused_program.py:MultiOutput, line 317] wraps a dict of CBHandles with attribute, dict, and positional access:

```python
class MultiOutput:
    def __init__(self, outputs: dict[str, CBHandle]):
        self._outputs = outputs

    # Supports:
    scores, indices = DeepseekMoeGate.emit(f, ...)   # positional
    result = DeepseekMoeGate.emit(f, ...)
    result.output                                     # by name
    result["output_indices"]                           # dict-style
```

### 5.7 Shadow Graph

FusedProgram always maintains a shadow `BlazeGraph` [blaze/fused_program.py:FusedProgram.__init__, line 493] that records every `output()` and `multi_output()` call during composition. Each MicroOp's `emit()` creates an `OpNode` in the shadow graph with captured CT arg values. This graph is NOT used for primary compilation -- it serves four purposes:

1. **Kernel code generation:** When no handwritten kernel is provided, the shadow graph is walked to emit a C++ kernel that chains the MicroOp phases in sequence.
2. **Visualization export:** When `BLAZE_EXPORT=1`, the shadow graph is serialized to JSON for the web-based visualizer.
3. **CB compaction:** The shadow graph provides lifetime information for interval coloring (temporal CB ID reuse across phases).
4. **Injected barrier building:** The fabric barrier system walks the shadow graph to insert inter-CCL synchronization nodes (see [Chapter 4, File 02](./02_ccl_in_blaze.md)).

### 5.8 The GridConfig and Role Engine

`GridConfig` [blaze/role_engine.py:GridConfig, line 17] encapsulates the device's compute grid:

```python
@dataclass(frozen=True)
class GridConfig:
    grid_cols: int                                     # e.g. 13
    grid_rows: int                                     # e.g. 10
    dram_worker_positions: frozenset[tuple[int, int]]  # 8 DRAM bank cores

    @property
    def sender_core(self) -> tuple[int, int]:  # (grid_cols-1, grid_rows-1)
    @property
    def total_cores(self) -> int:              # grid_cols * grid_rows
    @property
    def matmul_cores(self) -> int:             # total - DRAM - phantom - sender

    def get_matmul_cores(self) -> list[tuple[int, int]]
    def get_gate_mm_cores(self, num_cores) -> list[tuple[int, int]]
    def build_ab_grids(self) -> tuple[list, list]  # balanced A/B split
    def build_matmul_core_grid(self) -> CoreRangeSet
```

The default Blackhole grid is 13x10 (130 cores) with 8 DRAM worker positions. Harvested devices have fewer columns (11x10 or 12x10). `GridConfig.from_device()` queries the actual hardware.

**A/B grid splitting** for gated matmul (gate + up projections):

```
  Blackhole 13x10 Grid Layout
  +---+---+---+---+---+---+---+---+---+---+---+---+---+
  | D | A | A | A | B | B | B |   | A | A | B | B | P | row 0
  |   | A | A | A | B | B | B | D | A | A | B | B | P | row 1
  |   | A | A | A | B | B | B |   | A | A | B | B | P | row 2
  | D | A | A | A | B | B | B |   | A | A | B | B | P | row 3
  |   | A | A | A | B | B | B | D | A | A | B | B | P | row 4
  |   | A | A | A | B | B | B |   | A | A | B | B | P | row 5
  |   | A | A | A | B | B | B | D | A | A | B | B | P | row 6
  | D | A | A | A | B | B | B |   | A | A | B | B | P | row 7
  |   | A | A | A | B | B | B |   | A | A | B | B | P | row 8
  | D | A | A | A | B | B | B | D | A | A | B | B | S | row 9
  +---+---+---+---+---+---+---+---+---+---+---+---+---+
    0   1   2   3   4   5   6   7   8   9  10  11  12

  D = DRAM worker (8 total)  P = Phantom  S = Sender  A/B = Matmul cores
```

The A/B split algorithm lives in `GridConfig.build_ab_grids()` [blaze/role_engine.py:GridConfig.build_ab_grids, line 183], which partitions the matmul cores into two balanced sets. The result is stored in an `ABGrid` [blaze/fused_program.py:ABGrid, line 362] data structure:

```python
@dataclass
class ABGrid:
    a_cores: list       # gate branch cores
    b_cores: list       # up branch cores
    a_grid: ttnn.CoreRangeSet
    b_grid: ttnn.CoreRangeSet
    compute_grid: ttnn.CoreRangeSet
    num_per_branch: int
```

---

## 6. DeviceContext: Per-Device Compilation

`DeviceContext` [blaze/device_context.py:DeviceContext, line 22] encapsulates all device-dependent state:

```python
@dataclass
class DeviceContext:
    device: object                      # tt device handle
    grid_config: GridConfig             # grid dimensions and core layout
    full_device_grid: ttnn.CoreRangeSet # all compute cores

    def worker_core_from_logical_core(self, core) -> object:
        """Translate logical CoreCoord to physical (NOC) CoreCoord."""
        return self.device.worker_core_from_logical_core(core)

    @classmethod
    def from_device(cls, device) -> "DeviceContext":
        """Create from a real tt device."""
        grid_config = GridConfig.from_device(device)
        grid_size = device.compute_with_storage_grid_size()
        full_device_grid = ttnn.CoreRangeSet([
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1),
            )
        ])
        return cls(device=device, grid_config=grid_config,
                   full_device_grid=full_device_grid)
```

**Mesh-aware compilation:** The `BlazeCompiler` creates a single `DeviceContext` from the `MeshDevice` and reuses it for all devices in the mesh. Per-device specialization happens via `mesh_coord` and `per_device_overrides` injected into each iteration's `user_args`.

---

## 7. The BlazeGraph IR

The `BlazeGraph` [blaze/graph.py, line 101] is a DAG of `OpNode` objects connected by `Edge` objects:

```
  BlazeGraph
  +------------------------------------------+
  | nodes: list[OpNode]                      |
  | edges: list[Edge]                        |
  | input_tensors: dict[str, Any]            |
  | output_nodes: list[OpNode]               |
  | external_input_ports: set[(node, port)]  |
  | tensor_to_ports: dict[name -> [(n, p)]]  |
  +------------------------------------------+
       |
       v
  OpNode                     Edge
  +------------------+       +------------------+
  | id: str          |       | producer: OpNode |
  | spec: OpSpec     |       | producer_port    |
  | grid: Any        |       | consumer: OpNode |
  | kwargs: dict     |       | consumer_port    |
  | ct_values: dict  |       | cb_id: int|None  |
  +------------------+       +------------------+
```

Each `OpNode` carries:

```python
@dataclass
class OpNode:
    id: str                  # unique instance ID
    spec: OpSpec             # registered op specification
    grid: Any                # CoreRangeSet or grid specifier
    kwargs: dict             # user_args, ct_prefix, etc.
    ct_values: dict          # auto-captured CT arg values from emit()
```

Each `Edge` carries:

```python
@dataclass
class Edge:
    producer: OpNode
    producer_port: str       # output port name
    consumer: OpNode
    consumer_port: str       # input port name
    cb_id: int | None        # filled by CBEngine
```

Graphs are built in two ways:

1. **Graph API** (`blaze.fuse()` context manager):
   ```python
   with blaze.fuse() as ctx:
       x = ExternalTensor("input")
       norm = RMSNorm.graph_call(x, grid=all_cores, prefix="rn")
       out = Matmul.graph_call(norm, weight, prefix="mm")
   graph = ctx.graph
   ```

2. **Shadow graph** (automatic during FusedProgram composition):
   Each MicroOp `emit()` call appends an `OpNode` to `f._shadow_graph`.

The `topological_order()` method implements Kahn's algorithm for deterministic ordering. The `validate()` method checks structural integrity.

---

## 8. The CB and Semaphore Engines

### 8.1 CBEngine

The `CBEngine` [blaze/cb_engine.py:CBEngine, line 141] walks the graph in topological order and assigns sequential CB IDs to all data paths:

```
  CBEngine.assign(graph)
      |
      For each node in topological order:
      |
      +-- External input ports      -> "ext_in_{node}_{port}"
      |   (tensor-backed, sequential CB IDs)
      |
      +-- Internal CBs from OpSpec  -> "internal_{node}_{port}"
      |   (scratch buffers, NOT graph edges)
      |
      +-- Output ports:
          +-- No outgoing edges     -> "ext_out_{node}_{port}"
          |   (terminal output, tensor-backed)
          |
          +-- Has outgoing edges    -> "intermed_{node}_{port}"
              (intermediate CB, fan-out shares one ID)
              Core ranges = union(producer grid, all consumer grids)
      |
      v
  dict[key -> CBAssignment(cb_id, total_size, page_size, ...)]
```

Key features:
- **Fan-out sharing:** One output feeding N consumers gets one CB ID with core ranges as the union of all grids.
- **Lifetime tracking:** Each CBAssignment records `first_use` and `last_use` topological positions for compaction.
- **Compact mode:** When `compact=True`, the `compact_cb_ids()` method applies interval graph coloring to reuse CB IDs across non-overlapping lifetimes, critical for large pipelines that would otherwise exceed the 64-CB Blackhole limit:

```python
def compact_cb_ids(self, assignments):
    sorted_cbs = sorted(assignments.values(), key=lambda a: a.first_use)
    color_end: dict[int, int] = {}
    for cb in sorted_cbs:
        color = 0
        while color in color_end and color_end[color] >= cb.first_use:
            color += 1
        cb.cb_id = color
        color_end[color] = cb.last_use
    return assignments
```

This is analogous to register allocation in a compiler -- each CB lifetime is an interval, and the coloring minimizes total colors (CB IDs) used.

### 8.2 SemEngine

The `SemEngine` [blaze/sem_engine.py:SemEngine, line 94] identifies inter-core synchronization points and assigns global semaphore indices:

1. Walk topological order, identify inter-core sync nodes (gather, mcast, CCL ops).
2. Read each node's `CTArgSchema` to find `Sem`-kind entries (the `source_key` is the semaphore protocol).
3. Assign unique sequential indices. Mcast senders sharing the same explicit `sender` kwarg get the same index (persistent NOC command buffer sharing).
4. Return `SemAssignment` list, each mapping a `sem_id` to a protocol string.

At compile time, each `sem_id` maps to a `ttnn.GlobalSemaphore` created on the device. The L1 address of that semaphore is injected into CT args via the `CTArgEngine`.

---

## 9. The Full Compilation Pipeline

The complete pipeline from op definition to device execution:

```
 [1] Op Definition            [2] Graph Construction
 +------------------+         +---------------------+
 | class MyOp(FusedOp):  |   | with FusionContext():   |
 |   name = "my_op"      |   |   out = MyOp.graph_call |
 |   compose(f, t, o, a) |   |          (ext_a, ext_b) |
 |   MyOp.register()     |   |   graph = ctx.graph     |
 +------------------+         +---------------------+
                                       |
                                       v
 [3] BlazeCompiler.compile()
 +---------------------------------------------------------------+
 |                                                               |
 |  a) Reset semaphore/tensor dedup dicts                        |
 |  b) Pre-compute device-agnostic engine results:               |
 |     - CBEngine().assign(graph) -> cb_assignments              |
 |     - SemEngine().assign(graph) -> sem_assignments            |
 |  c) Split mesh tensors into per-device lists                  |
 |  d) For each device (row, col):                               |
 |     - Merge user_args + mesh_coord + per_device_overrides     |
 |     - _compile_single():                                      |
 |       * FusedOp path: create FusedProgram, call compose()     |
 |         OR                                                    |
 |       * Engine path: run CB/Sem/CT engines, build UKD         |
 |     - Add ProgramDescriptor to MeshProgramDescriptor          |
 |  e) Assemble MeshCompiledProgram                              |
 |                                                               |
 +---------------------------------------------------------------+
                     |
                     v
 [4] Execution: result.run()
 +------------------------------------------+
 | ttnn.generic_op(io_tensors, descriptor)  |
 | -> dispatches to each device             |
 | -> kernel runs on RISC processors        |
 | -> returns output tensor                 |
 +------------------------------------------+
```

[blaze/compiler.py:BlazeCompiler.compile, lines 192-341]

### 9.1 Two Compilation Paths

The compiler supports two compilation paths, selected automatically by `_compile_single()`:

**Fused Op Path** (single-node graph with registered `FusedOpConfig`):

```python
f = FusedProgram(kernel=..., device=device, ...)
f.mesh_coord = (row, col)
f.mesh_shape = (num_rows, num_cols)
f.mesh_device = mesh_device
config.compose_fn(f, tensors, output_tensor, merged_args)
compiled = f.build()
```

This is the common path for complex fused ops (MoE, MLA, etc.). The compose function chains MicroOp emit() calls on the FusedProgram.

**Engine Path** (multi-node graph, or no compose function):

```python
cb_assignments = CBEngine().assign(graph)
sem_assignments = SemEngine().assign(graph)
ct_tuples = CTArgEngine().generate_tuples(graph, cb_for_ct, sem_for_ct, ...)
ukd = UnifiedKernelDescriptor(
    kernel_source=...,
    core_ranges=...,
    ncrisc_named_compile_time_args=ct_tuples["ncrisc"],
    trisc_named_compile_time_args=ct_tuples["trisc"],
    ...
)
program_descriptor = ttnn.ProgramDescriptor(
    kernels=ukd.get_kernel_descriptors().kernels,
    cbs=cb_descriptors,
)
```

> **Warning:** The fused op path calls `compose()` which internally calls `emit()` methods that directly manipulate the FusedProgram's CB and CT arg state. The engine path instead derives everything from the graph structure and engine assignments. These are fundamentally different compilation strategies -- the fused op path is more flexible (supports arbitrary Python logic in compose), while the engine path is more structured (graph-driven, automatic CB assignment).

### 9.2 Mesh Assembly

Per-device `ProgramDescriptor` objects are inserted into a `MeshProgramDescriptor` keyed by `MeshCoordinate`:

```python
mesh_pd = ttnn.MeshProgramDescriptor()
for idx in range(num_devices):
    row, col = divmod(idx, num_cols)
    compiled = self._compile_single(ctx, graph, dev_tensors, ...)
    coord = ttnn.MeshCoordinate(row, col)
    mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = compiled.program_descriptor
```

[blaze/compiler.py:BlazeCompiler.compile, line 313]

> **Warning:** Per-device internal tensors whose raw `Buffer*` pointers are baked into CBDescriptors must be kept alive for the program's lifetime. The compiler pins these via `_lifetime_tensors` and `_lifetime_semaphores` on `MeshCompiledProgram` to prevent use-after-free when Python garbage collection runs.

### 9.3 Device Execution

Both paths terminate with `ttnn.generic_op()`:

```python
def _run_program(descriptor, io_tensors, output_tensors, deallocate_tensors):
    ttnn.generic_op(io_tensors, descriptor)
    for t in deallocate_tensors:
        ttnn.deallocate(t)
    return output_tensors[0] if output_tensors else io_tensors[-1]
```

[blaze/compiler.py:_run_program, line 98]

`ttnn.generic_op()` is the TT-Metal API that submits a program descriptor (or mesh program descriptor) to the device for execution. It handles hardware dispatch, CB allocation on L1, kernel binary loading, and RISC core scheduling.

---

## 10. The Op Registry

The registry [blaze/registry.py] is a simple global dict:

```python
_OP_REGISTRY: dict[str, OpSpec] = {}

def register_op(spec: OpSpec) -> None:
    _OP_REGISTRY[spec.name] = spec

def get_op_spec(name: str) -> OpSpec:
    if name not in _OP_REGISTRY:
        raise KeyError(f"Op '{name}' not registered.")
    return _OP_REGISTRY[name]
```

Four parallel registries coexist, all populated by a single `register()` call:

1. `_OP_REGISTRY` -- OpSpec by name (graph construction and validation)
2. `_CT_ARG_SCHEMAS` -- CTArgSchema by name (CT arg engine)
3. `_FUSED_OP_CONFIGS` -- FusedOpConfig by name (compiler dispatch)
4. `BlazeOp._class_registry` -- Python class by name (class lookup)

---

## 11. Putting It All Together: A Complete Example

Here is the complete lifecycle of the Matmul op, from class definition to device execution:

```python
# -- 1. Class Definition --
class Matmul(MicroOp):
    name = "matmul"
    math_fidelity = "LoFi"
    in0 = Input()    # activations
    in1 = Input()    # weights
    out = Output()   # result

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        cls.emit(f, tensors["in0"], tensors["in1"],
                 prefix=user_args.get("prefix", "matmul"))

    @staticmethod
    def emit(f, in0, in1, *, prefix="matmul", pop_in0=True):
        in0_handle = f.cb_from_tensor(in0) if not isinstance(in0, CBHandle) else in0
        in1_cb, weights_addr = resolve_weight_direct_address(f, in1)
        out = f.cb_scratch(name=BlazeOp.cb_name(prefix, "out"), ...)
        f.per_core_unified_ct_args([
            f.flag(f"{prefix}.is_active", active_ranges),
            f.flag(f"{prefix}.pop_in0", active_ranges, pop_in0),
        ])
        f.unified_ct_args([
            (f"{prefix}.in0_cb", in0_handle),
            (f"{prefix}.in1_cb", in1_cb),
            (f"{prefix}.out_cb", out),
            (f"{prefix}.k_num_tiles", k_num_tiles),
        ])
        return out

# -- 2. Register --
Matmul.register()    # -> OpSpec + CTArgSchema + FusedOpConfig in registries

# -- 3. Graph Construction --
with blaze.fuse() as ctx:
    acts = ExternalTensor("activations")
    wts = ExternalTensor("weights")
    out = Matmul.graph_call(acts, wts, prefix="mm")

# -- 4. Compile --
compiler = BlazeCompiler(mesh_device)
result = compiler.compile(
    graph=ctx.graph,
    tensors={"activations": ttnn_acts, "weights": ttnn_wts},
    output_tensor=ttnn_output,
    user_args={"prefix": "mm"},
)

# -- 5. Execute --
output = result.run()
# Internally: ttnn.generic_op(io_tensors, mesh_program_descriptor)
```

> **Warning:** CB IDs are assigned sequentially during composition. If you manually hardcode CB IDs instead of using `f.cb_from_tensor()` or `f.cb_scratch()`, they will conflict with the auto-assignment system. Always use the FusedProgram API.

---

## Summary

The Blaze kernel composition framework is built on four key abstractions:

1. **Op Hierarchy** -- BlazeOp/FusedOp/MicroOp provides a class-is-the-definition pattern where port descriptors, CT args, and composition logic are declared as class attributes.

2. **FusedProgram** -- The composition context that manages CB allocation (with disjoint-core sharing, temporal reuse, and OverlappedView packing), semaphore provisioning, CT arg emission, and shadow graph recording.

3. **Engine Pipeline** -- CBEngine, SemEngine, and CTArgEngine automatically assign resources from a BlazeGraph, enabling the Graph API compilation path.

4. **BlazeCompiler** -- The top-level compiler that handles mesh decomposition, per-device compilation (via either the fused-op or engine path), and assembly into MeshProgramDescriptors for `ttnn.generic_op()` execution.

---

## Key Source Files Reference

| File | Purpose |
|------|---------|
| `blaze/blaze_op.py` | BlazeOp, FusedOp, MicroOp base classes; port descriptors; CT arg types |
| `blaze/fused_program.py` | FusedProgram composition container; CB/sem allocation; shadow graph; OverlappedView; MultiOutput |
| `blaze/ct_args.py` | CTArgSchema, CTArgEngine; typed CT arg resolution |
| `blaze/device_context.py` | DeviceContext: per-device grid + NOC translation |
| `blaze/compiler.py` | BlazeCompiler: graph -> MeshProgramDescriptor -> execution |
| `blaze/cb_engine.py` | CBEngine: automatic CB ID assignment with interval coloring |
| `blaze/sem_engine.py` | SemEngine: semaphore auto-assignment |
| `blaze/graph.py` | BlazeGraph, OpNode, Edge, OpSpec data structures |
| `blaze/registry.py` | Global op registry |
| `blaze/role_engine.py` | GridConfig: device grid layout and core classification |
| `blaze/context.py` | FusionContext: graph construction via `fuse()` block |
| `blaze/cb_handle.py` | CBHandle: metadata for allocated circular buffers |

---

[**Previous:** Chapter 4 Index](./index.md) | [**Next:** `02_ccl_in_blaze.md`](./02_ccl_in_blaze.md)
