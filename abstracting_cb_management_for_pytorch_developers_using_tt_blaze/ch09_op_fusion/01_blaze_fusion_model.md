# TT-Blaze's Fusion Model: MicroOps, FusedOps, and the Composition Architecture

Op fusion is the mechanism by which multiple logically distinct PyTorch operations -- a matmul followed by SiLU followed by an elementwise multiply -- are compiled into a single TT-Blaze kernel that executes without returning to the host between steps. Fusion eliminates inter-op DRAM round-trips, keeps intermediate data in L1 SRAM, and enables cross-op optimizations (CB sharing, format continuity, pipeline overlap). But fusion also creates the most complex shape/padding/format tracking challenge in the entire system: when 11 operations are collapsed into one FusedOp, the abstraction must track shape transformations, padding strategy transitions, and format changes at every internal boundary -- all within a single `FusedProgram` scope. This section defines TT-Blaze's two-level composition architecture (MicroOp vs. FusedOp), the `blaze.fuse()` context manager and `FusionContext`, the Graph API vs. Composition API distinction, and traces a complete LLaMA-style gated MLP through both the manual fusion path and the proposed abstracted path.

> *Design Principle P5 (Operate on Graphs, Not Individual Tensors):* Fusion is fundamentally a graph-level operation. Identifying fusible sequences, validating cross-op constraints, and allocating shared resources (CB IDs, L1 memory, semaphores) all require visibility into the full subgraph, not just individual tensors.

**What you will learn:**

- The two-level composition architecture: `MicroOp` (single C++ kernel phase) vs. `FusedOp` (composition of multiple MicroOps via `compose()` and `emit()` chains)
- The `blaze.fuse()` context manager and `FusionContext`: how op calls are captured, how a `BlazeGraph` is built, and where validation occurs
- The FusionContext state machine with formal pre/post conditions for each transition
- Graph API vs. Composition API: `blaze.matmul(...)` (graph node creation) vs. `Matmul.emit(f, ...)` (FusedProgram composition), and when each is appropriate
- How `build_gated_mlp_graph()` in `_gated_mlp.py` composes 11 ops into a single fused graph -- the template for automatic fusion
- How `SwigluOp.compose()` chains `DRAMStreamingMatmul`, `EltwiseMul`, `Gather`, `Mcast`, and `DRAMStreamingMatmul`, with a complete CB flow trace
- Performance quantification: the DRAM round-trip tax, interval coloring for CB compaction, and L1 memory analysis
- A side-by-side comparison of manual vs. abstracted fusion approaches for a gated MLP

---

## Why Fusion Matters: The DRAM Round-Trip Tax

Before examining the fusion architecture, consider the cost of *not* fusing. A gated MLP layer in LLaMA-7B performs five operations. Without fusion, each is a separate kernel launch with DRAM round-trips for every intermediate:

```text
Without fusion (5 separate kernel launches, seq_len=1024):
  Intermediate output: (1024, 11008) = 32 x 344 = 11,008 tiles
  Final output: (1024, 4096) = 32 x 128 = 4,096 tiles

  1. gate_out = matmul(act, W_gate)    -- write 11,008 tiles to DRAM
  2. up_out   = matmul(act, W_up)      -- write 11,008 tiles to DRAM
  3. silu_out = silu(gate_out)          -- read 11,008, write 11,008 tiles
  4. mul_out  = silu_out * up_out       -- read 2 x 11,008, write 11,008 tiles
  5. down_out = matmul(mul_out, W_down) -- read 11,008, write 4,096 tiles

  Intermediate round-trips (write + later read-back):
    gate_out: 11,008 + 11,008 = 22,016 tile transfers
    up_out:   11,008 + 11,008 = 22,016 tile transfers
    silu_out: 11,008 + 11,008 = 22,016 tile transfers
    mul_out:  11,008 + 11,008 = 22,016 tile transfers
    Total: 88,064 tiles x 2,048 B/tile = ~172 MB of intermediate DRAM traffic

  Kernel launch overhead: 5 x ~20 us = ~100 us

With fusion (1 kernel launch):
  1-5. All ops execute in a single FusedProgram
       Intermediates stay in L1 CBs -- never touch DRAM

  Intermediate DRAM eliminated: ~172 MB
  Kernel launches saved: 4 x ~20 us = ~80 us
  Typical wall-clock speedup: 2-4x depending on batch size and weight format
```

The speedup comes entirely from eliminating intermediate DRAM round-trips. The compute is identical. This is why fusion is not optional for production performance -- it is the primary mechanism by which TT-Blaze achieves hardware utilization above 50%.

---

## The Two-Level Composition Architecture

### MicroOp: The Atomic Building Block

A `MicroOp` represents a single C++ kernel phase -- one invocation of the TRISC (compute), NCRISC (data movement), or BRISC (control) RISC processors on Tensix cores. Every Blaze op ultimately decomposes into one or more MicroOps. A MicroOp declares its ports (Input, Output, Internal), its compile-time arguments, and its `emit()` method.

```python
# EXISTING -- from blaze/blaze_op.py (simplified)
class MicroOp(BlazeOp):
    """A single kernel phase.

    MicroOps are the leaf nodes of the composition hierarchy.
    Each has a kernel path pointing to a C++ source file,
    input/output/internal port declarations, and CT args.

    Examples: Matmul, EltwiseMul, RMSNorm, Mcast, Gather, Silu.
    """
    kernel_path: str
    input: Input = Input()
    output: Output = Output()

    @classmethod
    def emit(cls, f: FusedProgram, *args, **kwargs) -> CBHandle:
        """Emit this MicroOp into a FusedProgram.

        Allocates CBs, wires CT args, and records the op node
        in the FusedProgram's shadow graph.
        """
        ...
```

A MicroOp's `emit()` method performs three actions:
1. **Allocate CBs** via `f.cb_scratch()` for internal/output buffers
2. **Wire CT args** via `f.ncrisc_ct_args()`, `f.trisc_ct_args()`, `f.per_core_unified_ct_args()`
3. **Record graph node** in `f._shadow_graph` with `_node_id` and `_port_name` stamps on the output CBHandle

Examples of MicroOps in the existing codebase:

| MicroOp | Kernel | Ports | Purpose |
|---------|--------|-------|---------|
| `Matmul` | `matmul.cpp` | in0, in1, out | Dense matrix multiply |
| `EltwiseMul` | `eltwise_binary_sfpu.cpp` | in0, in1, out | Element-wise multiply |
| `Silu` | `eltwise_unary_sfpu.cpp` | input, output | SiLU activation |
| `Mcast` | `mcast.cpp` | input, output | Multicast data to cores |
| `Gather` | `gather.cpp` | input, output | Gather data from cores |
| `RMSNorm` | `rmsnorm.cpp` | input, gamma, output | Root mean square normalization |
| `PaddedRMSNorm` | `padded_rmsnorm.cpp` | input, gamma, output, padded_input, padded_gamma | RMSNorm with explicit padding CBs |

### Architectural Invariant: emit() Is Pure

As established in [Chapter 3, File 03](../ch03_tensor_adapter/03_wrapping_vs_extending_vs_replacing.md), `emit()` methods are pure functions of their parameters. Given the same `FusedProgram`, input `CBHandle`s, and keyword arguments, `emit()` produces the same output. This invariant is what makes TensorAdapter's "adapter as `emit()` caller" pattern safe: the abstraction computes parameters, then delegates to `emit()` with confidence that the result is identical to hand-written code (Design Principle P2 -- Carry Metadata from Both Worlds Simultaneously).

### FusedOp: Composition of MicroOps

A `FusedOp` composes multiple MicroOps into a single compiled program. Its `compose()` method chains `emit()` calls on a shared `FusedProgram`, with `reconfig()` boundaries between phases that require different CB configurations:

```python
# EXISTING -- from blaze/blaze_op.py (simplified)
class FusedOp(BlazeOp):
    """A composition of multiple MicroOps sharing a single FusedProgram.

    FusedOps define their structure via compose(), which calls
    emit() on constituent MicroOps within a shared FusedProgram.
    All intermediate data stays in L1 -- no DRAM round-trips.
    """
    @classmethod
    def compose(cls, f: FusedProgram, *args, **kwargs):
        """Define the fused op's structure by chaining emit() calls.

        The compose() method receives a FusedProgram and the op's
        input handles. It calls emit() on each constituent MicroOp,
        threading output handles from one to the next.
        """
        ...
```

The critical distinction: **MicroOp.emit() stamps one kernel phase into an existing FusedProgram. FusedOp.compose() creates the FusedProgram and orchestrates multiple emit() calls.** The TensorAdapter operates at the FusedOp level -- it provides the shape/padding/format metadata that each emit() call needs, without the developer computing it manually.

### The Hierarchy in Practice

```text
BlazeOp (abstract base)
  |
  +-- MicroOp (single kernel phase)
  |     |
  |     +-- Matmul
  |     +-- RMSNorm / PaddedRMSNorm
  |     +-- Silu / ReLU / GeLU
  |     +-- EltwiseMul / EltwiseAdd
  |     +-- Gather / Mcast
  |     +-- DRAMStreamingMatmul
  |     +-- CbReconfig / CbScratchReset
  |
  +-- FusedOp (MicroOp composition)
        |
        +-- SwigluOp          (5 MicroOps, 2 reconfig boundaries)
        +-- DenseMLP           (12 MicroOps, 3 reconfig boundaries)
        +-- GatedMLP           (11 ops via graph API)
        +-- BroadcastRMSNorm   (2 MicroOps: CCL Broadcast + RMSNorm)
        +-- AttentionBlock     (Q/K/V proj + SDPA + output proj)
```

### Why Two Levels?

The two-level design mirrors the hardware: one level for the indivisible compute units (kernels), one level for their software-level composition (sequencing, data flow, phase boundaries). This maps directly to Principle P1 (Separate Computation Intent from Execution Strategy): the computation intent is expressed at the FusedOp level; the execution strategy is determined by the kernel implementations at the MicroOp level.

---

## The blaze.fuse() Context Manager and FusionContext

### The Graph API Entry Point

The `blaze.fuse()` context manager is the primary entry point for graph-level fusion. Within the context block, calls like `blaze.matmul(...)` create graph nodes rather than executing immediately:

```python
# EXISTING -- from blaze/context.py (simplified)
@contextlib.contextmanager
def fuse(*, shape_engine=None, **kwargs):
    """Create a FusionContext for graph construction.

    Within the with-block, calls like blaze.matmul(...) create
    graph nodes rather than executing immediately. The context
    builds a BlazeGraph that can be compiled via BlazeCompiler.
    """
    ctx = FusionContext(shape_engine=shape_engine, **kwargs)
    _FUSION_CONTEXT_STACK.append(ctx)
    try:
        yield ctx
    finally:
        _FUSION_CONTEXT_STACK.pop()
        ctx.finalize()
```

### FusionContext Lifecycle as a State Machine

The `FusionContext` progresses through six well-defined states, which we model as a formal state machine with pre/post conditions:

```text
FusionContext State Machine
===========================

State 0: EMPTY
  Pre:  Context just created
  Post: graph.nodes = [], graph.edges = []
  Invariant: No ops registered, no validation performed

  Transition: add_op() -> State 1 (BUILDING)

State 1: BUILDING
  Pre:  At least one op registered
  Post: graph.nodes.length >= 1, edges may exist
  Invariant: Every FusionResult references a valid node in graph.nodes
             Every edge connects valid (producer_port, consumer_port) pairs

  Transitions:
    add_op() -> State 1 (self-loop, add more ops)
    ctx.__exit__() -> State 2 (VALIDATING)

State 2: VALIDATING
  Pre:  Context block exited (with-statement ended)
  Post: ctx.validate() has run
  Invariant: Graph is acyclic (DAG)
             All external inputs have backing tensors
             No dangling edges (every consumer port is connected)
             No orphan nodes (every node has at least one input or output)

  Transitions:
    validate() succeeds -> State 3 (VALIDATED)
    validate() fails -> State 4 (ERROR)

State 3: VALIDATED
  Pre:  Validation passed
  Post: ctx.graph is a valid BlazeGraph ready for compilation
  Invariant: All State 2 invariants hold
             Shape metadata (if TensorAdapter is active) attached to all edges

  Transitions:
    BlazeCompiler.compile(ctx.graph, ...) -> State 5 (COMPILED)

State 4: ERROR
  Pre:  Validation failed
  Post: Error raised with diagnostic message
  Invariant: Graph is inconsistent; must not be compiled
  (terminal state)

State 5: COMPILED
  Pre:  BlazeCompiler.compile() succeeded
  Post: CompiledProgram or MeshCompiledProgram object available
  Invariant: CB IDs assigned, CT args resolved, kernel binaries linked

  Transitions:
    program.run() -> State 6 (EXECUTED)

State 6: EXECUTED
  Pre:  Program dispatched to device
  Post: Output tensor contains results
  Invariant: Output tensor shape matches terminal node's ShapeDescriptor.logical_shape
```

### Formal Pre/Post Conditions for Key Operations

```python
# PROPOSED -- formal pre/post conditions
class FusionContext:
    """State machine for graph construction and validation."""

    class State(Enum):
        EMPTY = 0
        BUILDING = 1
        VALIDATING = 2
        VALIDATED = 3
        ERROR = 4
        COMPILED = 5

    def add_op(self, op_spec: OpSpec, inputs: dict, grid, **kwargs) -> FusionResult:
        """Add an op node to the graph.

        Pre:  self._state in {EMPTY, BUILDING}
        Post: self._state == BUILDING
              len(self._graph.nodes) increased by 1
              For each input in inputs:
                  An edge exists from input.node/input.port to new_node
              If TensorAdapter is active:
                  ShapeDescriptor is inferred for each output port
        """
        assert self._state in (self.State.EMPTY, self.State.BUILDING)
        node = OpNode(spec=op_spec, grid=grid, **kwargs)
        self._graph.add_node(node)
        for port_name, source in inputs.items():
            if isinstance(source, FusionResult):
                self._graph.add_edge(source.node, source.port_name,
                                     node, port_name)
            elif isinstance(source, ExternalTensor):
                self._graph.add_external_input(source, node, port_name)
        self._state = self.State.BUILDING
        return FusionResult(node, op_spec.output_port_name)

    def validate(self) -> None:
        """Validate the graph after context exit.

        Pre:  self._state == BUILDING
        Post: self._state == VALIDATED (all checks pass)
              OR self._state == ERROR (any check fails)

        Checks:
          1. Acyclicity (topological sort succeeds)
          2. Port connectivity (no dangling inputs)
          3. Shape compatibility (ShapeDescriptor dims match at edges)
          4. Format compatibility (no incompatible format crossings)
          5. Grid consistency (all ops in chain use compatible grids)
        """
        assert self._state == self.State.BUILDING
        try:
            self._graph.topological_sort()
            self._validate_connectivity()
            self._validate_shapes()
            self._validate_formats()
            self._state = self.State.VALIDATED
        except Exception:
            self._state = self.State.ERROR
            raise
```

### State Transition Rules

| Current State | Event | Next State | What Happens |
|--------------|-------|-----------|-------------|
| EMPTY | `blaze.matmul(...)` called | BUILDING | First node added to graph |
| BUILDING | `blaze.matmul(...)` called | BUILDING | Additional node added (self-loop) |
| BUILDING | Exit `with` block | VALIDATING | `ctx.validate()` runs |
| VALIDATING | Validation passes | VALIDATED | Graph ready for compilation |
| VALIDATING | Validation fails | ERROR | Exception raised |
| VALIDATED | `compiler.compile()` | COMPILED | Program built |
| COMPILED | `program.run()` | EXECUTED | Output available |

### Error Transitions

| State | Error Condition | Failure |
|-------|----------------|---------|
| BUILDING | Op call with incompatible input types | `TypeError` at op call site |
| VALIDATING | Disconnected graph (unreachable nodes) | `GraphValidationError` |
| VALIDATING | Cycle detected | `GraphValidationError` |
| VALIDATED | L1 budget exceeded during compile | `L1BudgetError` |
| VALIDATED | CB count exceeds 64 | `CBLimitError` |

---

## Graph API vs. Composition API

TT-Blaze provides two distinct APIs for building fused programs, each suited to different use cases.

### Graph API: Declarative Node Creation

```python
# EXISTING -- Graph API usage
with blaze.fuse() as ctx:
    act = ExternalTensor("activation", tensor=act_tensor)
    wt_gate = ExternalTensor("gate_weights", tensor=gate_tensor)
    wt_up = ExternalTensor("up_weights", tensor=up_tensor)

    gate_out = blaze.matmul({"in0": act, "in1": wt_gate}, grid=cores)
    up_out = blaze.matmul({"in0": act, "in1": wt_up}, grid=cores)
    gate_silu = blaze.silu({"input": gate_out}, grid=cores)
    fused = blaze.mul({"in0": gate_silu, "in1": up_out}, grid=cores)

program = compiler.compile(graph=ctx.graph, tensors={...},
                           output_tensor=out, user_args={...})
```

The Graph API is declarative: describes *what* to compute, not *how*. Shape metadata propagates automatically through `FusionResult.shape` when TensorAdapter is active.

### Composition API: Imperative emit() Chains

```python
# EXISTING -- Composition API usage
f = FusedProgram(kernel=kernel_path, device=device, ...)
in0 = f.cb_from_tensor(act_tensor)
wt = f.cb_from_tensor(gate_tensor, is_direct_address=True)

gate_out = Matmul.emit(f, in0, wt, prefix="gate_mm")
silu_out = Silu.emit(f, gate_out, prefix="silu")
# ... more emit() calls ...

f.wire_output(final_out, output_tensor)
result = f.build()
```

The Composition API is imperative: developer manages every CB, CT arg, and reconfig boundary.

### When to Use Each

| Criterion | Graph API | Composition API |
|-----------|-----------|-----------------|
| Shape tracking | Automatic via `FusionResult.shape` | Manual or via `ShapeTracker` |
| CT arg derivation | `derive_user_args(ctx.graph)` | Manual computation |
| Format negotiation | Two-pass graph walk (Ch6) | Developer decides per CB |
| Padding management | Padding pass over graph (Ch5) | Developer decides per op |
| CB ID management | `CBEngine.assign()` | `CircularBufferIdManager` |
| Multi-phase reconfig | Automatic via `CBEngine` | Manual `f.reconfig()` calls |
| Developer expertise | PyTorch-level | Blaze-level |

The Graph API compiles down to the Composition API. `BlazeCompiler.compile()` converts a `BlazeGraph` into a sequence of `emit()` calls on a `FusedProgram`. TensorAdapter supports both: shape-enriched `FusionResult` for the Graph path, `ShapeTracker` for the Composition path (Design Principle P4 -- Provide Defaults with Per-Decision Overrides).

---

## Concrete Example: build_gated_mlp_graph() in _gated_mlp.py

The `build_gated_mlp_graph()` function in `blaze/_gated_mlp.py` is the canonical example of production-scale fusion. It composes 11 ops into a single fused graph that implements the gated MLP block from LLaMA and DeepSeek V3:

```text
Gated MLP: y = down_proj(SiLU(gate_proj(x)) * up_proj(x)) + residual

11-node graph:
  1. mcast(activation)           -- broadcast activation to all compute cores
  2. kn_sliced_matmul(act, gate) -- gate projection: [1, 4096] x [4096, 11008]
  3. kn_sliced_matmul(act, up)   -- up projection: [1, 4096] x [4096, 11008]
  4. gather(gate_partial)        -- gather partial gate results to sender core
  5. gather(up_partial)          -- gather partial up results to sender core
  6. gated_reduce(gate, up)      -- SiLU(gate) * up (fused activation + mul)
  7. mcast(reduced)              -- broadcast reduced result to all cores
  8. matmul(reduced, down)       -- down projection: [1, 11008] x [11008, 4096]
  9. mcast(bias)                 -- broadcast bias (if present)
  10. residual_add(mm_out, bias) -- add bias and/or residual
  11. gather(result)             -- gather final result to sender core
```

### The Graph Structure

```python
# EXISTING -- from blaze/_gated_mlp.py (simplified structure)
def build_gated_mlp_graph(config):
    g = BlazeGraph()

    # Phase 1: Activation distribution
    mcast_act = g.add_op("mcast", inputs={"in0": ext_activation}, ...)

    # Phase 2: Gate and Up projections (parallel)
    gate_mm = g.add_op("kn_sliced_matmul",
                        inputs={"in0": mcast_act, "in1": ext_gate_weights}, ...)
    up_mm = g.add_op("kn_sliced_matmul",
                      inputs={"in0": mcast_act, "in1": ext_up_weights}, ...)

    # Phase 3: Gather partial results
    gate_gathered = g.add_op("gather", inputs={"in0": gate_mm}, ...)
    up_gathered = g.add_op("gather", inputs={"in0": up_mm}, ...)

    # Phase 4: SiLU(gate) * up
    gated = g.add_op("gated_reduce",
                      inputs={"gate": gate_gathered, "up": up_gathered}, ...)

    # Phase 5: Down projection
    mcast_gated = g.add_op("mcast", inputs={"in0": gated}, ...)
    down_mm = g.add_op("matmul",
                        inputs={"in0": mcast_gated, "in1": ext_down_weights}, ...)

    # Phase 6: Residual add and output
    mcast_bias = g.add_op("mcast", inputs={"in0": ext_bias}, ...)
    residual = g.add_op("residual_add",
                         inputs={"in0": down_mm, "in1": mcast_bias}, ...)
    result = g.add_op("gather", inputs={"in0": residual}, ...)

    return g
```

### Performance Quantification: Fused vs. Unfused

For a LLaMA-7B gated MLP layer on a 4x8 Blackhole core grid (single token decode):

```text
Metric                          Unfused (11 ops)    Fused (1 op)    Improvement
-----------------------------------------------------------------------------
Kernel dispatches               11                  1               11x fewer
DRAM round-trips                10                  0 (L1 only)     Eliminated
DRAM traffic (intermediates)    ~200 KB             0 KB            200 KB saved
NOC transactions (inter-core)   22 (2 per op)       6 (mcast+gather) 3.7x fewer
Total latency (estimated)       ~45 us              ~12 us          3.75x
CB slots (peak simultaneous)    3-4 per op          ~18 total       Shared lifetime
L1 utilization (peak)           ~15% per op         ~45% combined   Better utilization
```

The latency improvement comes from three sources:
1. **Dispatch overhead elimination:** Each kernel dispatch costs approximately 2-3 us. Eliminating 10 dispatches saves 20-30 us.
2. **DRAM bandwidth elimination:** At 400 GB/s bandwidth, 200 KB takes approximately 0.5 us per direction. 10 intermediates = 10 us of DRAM traffic.
3. **NOC traffic reduction:** Fewer mcast/gather operations reduce NOC contention.

### Worked Shape Propagation Through the 11-Node Graph

Using LLaMA-7B dimensions ($H = 4096$, $I = 11008$, 14 compute cores):

```text
Node 1: mcast(activation)
  Input:  ShapeDescriptor(logical=(1, 4096), padded=(32, 4096),
          tile=(32, 32), grid=(1, 128), format=bfloat16)
  Output: Same shape; data replicated to all 14 cores.
  CB: 128 tiles * 2,048 B = 256 KB per core

Node 2: kn_sliced_matmul(act, gate_weights)
  in0:  ShapeDescriptor(logical=(1, 4096), tile=(32, 32), grid=(1, 128))
  in1:  ShapeDescriptor(logical=(4096, 11008), tile=(32, 32), grid=(128, 344),
        format=bfloat8_b)
  Contract K=4096/32=128 tiles, produce N=11008/32=344 tiles
  Per-core N: 344/14 = ~25 tiles
  Output: ShapeDescriptor(logical=(1, 11008), padded=(32, 11008),
          tile=(32, 32), grid=(1, 344), format=bfloat16)
  CT args derived: k_num_tiles=128, out_w_per_core=25

Node 3: kn_sliced_matmul(act, up_weights)
  Output: ShapeDescriptor(logical=(1, 11008), padded=(32, 11008), ...)
  [Same shape as Node 2]

Nodes 4-5: gather(gate), gather(up)
  Output shapes unchanged; data movement only.

Node 6: gated_reduce(gate, up) = SiLU(gate) * up
  in0: ShapeDescriptor(logical=(1, 11008), ...)
  in1: ShapeDescriptor(logical=(1, 11008), ...)
  Output: ShapeDescriptor(logical=(1, 11008), ...)
  [Elementwise: shape preserved]

Node 7: mcast(reduced)
  Output: ShapeDescriptor(logical=(1, 11008), ...)

Node 8: matmul(reduced, down_weights)
  in0: ShapeDescriptor(logical=(1, 11008), tile=(32, 32), grid=(1, 344))
  in1: ShapeDescriptor(logical=(11008, 4096), tile=(32, 32), grid=(344, 128),
       format=bfloat8_b)
  Contract K=11008/32=344 tiles, produce N=4096/32=128 tiles
  Output: ShapeDescriptor(logical=(1, 4096), padded=(32, 4096), ...)
  CT args derived: k_num_tiles=344, out_w_per_core=128/14=~9

Nodes 9-10: mcast(bias) + residual_add
  Output: ShapeDescriptor(logical=(1, 4096), ...)

Node 11: gather(result)
  Output: ShapeDescriptor(logical=(1, 4096), ...)
  Final output, ready for unpadding back to (1, 4096).
```

The ShapeTracker maintains all 11 descriptors, automatically deriving CT args at each step. Without the tracker, the developer computes `k_num_tiles`, `out_w_per_core`, and per-core tile counts by hand for each of the three matmul nodes.

---

## SwigluOp.compose(): A FusedOp Case Study

The `SwigluOp` in `blaze/ops/swiglu/op.py` demonstrates the Composition API fusion pattern. Its `compose()` method chains MicroOps across multiple phases:

```python
# EXISTING -- from blaze/ops/swiglu/op.py (simplified)
class SwigluOp(FusedOp):
    """SwiGLU: SiLU(gate_proj(x)) * up_proj(x) -> down_proj(result)"""

    @classmethod
    def compose(cls, f, activation, gate_weights, up_weights,
                down_weights, output_tensor, *, cores, config):
        # Phase 0: Gate and Up projections
        gate_out = DRAMStreamingMatmul.emit(
            f, activation, gate_weights, prefix="gate_mm",
            k_num_tiles=config.k_tiles, out_w=config.gate_w)
        up_out = DRAMStreamingMatmul.emit(
            f, activation, up_weights, prefix="up_mm",
            k_num_tiles=config.k_tiles, out_w=config.up_w)

        # Phase 0 continues: Fused SiLU + multiply
        gated = EltwiseMul.emit(
            f, gate_out, up_out, prefix="silu_mul",
            apply_silu=True)

        # Phase 0/1 boundary: Gather and redistribute
        gathered = Gather.emit(f, gated, prefix="gather", ...)

        f.reconfig()  # Phase boundary

        mcast_out = Mcast.emit(f, gathered, prefix="mcast", ...)

        # Phase 1: Down projection
        f.reconfig()  # Phase boundary
        down_out = DRAMStreamingMatmul.emit(
            f, mcast_out, down_weights, prefix="down_mm",
            k_num_tiles=config.down_k, out_w=config.down_w)

        f.wire_output(down_out, output_tensor)
```

### CB Flow Trace Through SwigluOp

The following traces all CBs through the 6 stages, showing format, access mode, and lifetime:

```text
CB allocation trace:

Stage 1 (gate_mm):
  CB A: activation input (FIFO, bfloat16, from upstream)
  CB B: gate_up_weight (DIRECT_ADDRESS, bfloat8_b)
  CB C: gate_output (scratch, FIFO, bfloat16)

Stage 2 (up_mm):
  CB A: activation input (reused -- same CB, still live)
  CB B: gate_up_weight (reused -- same backing tensor, different slice)
  CB D: up_output (scratch, FIFO, bfloat16)

Stage 3 (silu_mul):
  CB C: gate_output (consumed by eltwise_mul)
  CB D: up_output (consumed by eltwise_mul)
  CB E: mul_output (scratch, FIFO, bfloat16)

Stage 4 (gather):
  CB E: mul_output (consumed by gather)
  CB F: gather_output (scratch, on sender core)

Stage 5 (mcast):
  CB F: gather_output (consumed by mcast)
  CB G: mcast_output (scratch, on all cores)

Stage 6 (down_mm):
  CB G: mcast_output (consumed by down matmul)
  CB H: down_weight (DIRECT_ADDRESS, bfloat8_b)
  CB I: down_output (scratch, FIFO, bfloat16, final output)

Total unique CBs: 9 (A through I)
After compaction via compact_cb_ids(): ~5-6 CB IDs
  (Stages 1-2 and Stages 4-6 have non-overlapping lifetimes for scratch CBs)
```

### The reconfig() Boundaries

The `reconfig()` calls create three phases for CB management:

```text
Phase 0: Gate matmul + Up matmul + EltwiseMul
  CBs: act_cb, gate_wt, up_wt, gate_out, up_out, mul_out
  Active IDs: {0, 1, 2, 3, 4, 5, 6}  -- 7 CBs

Phase 1: Gather + Mcast
  CBs: gather_out, mcast_src, mcast_dst
  Active IDs: {0, 1, 2}               -- 3 CBs (IDs reused)

Phase 2: Down matmul
  CBs: mcast_out (reused from Phase 1), down_wt, down_out
  Active IDs: {0, 1, 2, 3}            -- 4 CBs (IDs reused)

Peak simultaneous: 7
Total without reconfig reuse: 14
Total with reconfig reuse: 7
```

Without `reconfig()`, all CBs from both phases would be alive simultaneously, potentially exceeding both the L1 budget (~1.5 MB total, ~1.3 MB usable for CBs per Tensix core) and the 64 CB ID limit.

---

## L1 Memory Analysis and Interval Coloring

### The Interval Coloring Problem

Fusion fundamentally changes the L1 memory problem described in [Chapter 7, File 04](../ch07_cb_sizing/04_cb_compaction_and_temporal_reuse.md). Without fusion, each op's CBs live for the duration of that op's dispatch and are then freed. With fusion, all CBs share a single program's lifetime space, but their individual lifetimes are shorter than the program's total duration.

This creates an interval coloring opportunity: CBs whose lifetimes do not overlap can share CB IDs via `compact_cb_ids()`. For the 11-node gated MLP:

```text
CB              first_use   last_use    Phase
act_input             0          1       Phase 0 (mcast)
mcast_dst             1          3       Phase 0-1 (mcast -> matmul)
gate_wt               2          2       Phase 1 (gate matmul)
up_wt                 3          3       Phase 1 (up matmul)
gate_out              2          4       Phase 1 (matmul -> gather)
up_out                3          5       Phase 1 (matmul -> gather)
gate_gather           4          6       Phase 1-2 (gather -> reduce)
up_gather             5          6       Phase 1-2 (gather -> reduce)
reduce_out            6          7       Phase 2 (reduce -> mcast)
mcast2_dst            7          8       Phase 2 (mcast -> down matmul)
down_wt               8          8       Phase 2 (down matmul)
down_out              8          9       Phase 2-3 (matmul -> residual)
bias                  9          10      Phase 3 (mcast bias)
residual_out          10         11      Phase 3 (residual add)
final_gather          11         11      Phase 3 (output gather)

Without compaction: 15 CB IDs
With compaction (interval coloring): 6 CB IDs
```

The formal relationship between fusion and CB compaction:

$$
\chi(\text{fused}) = \omega(\text{fused}) \leq \sum_{i} \omega(\text{op}_i)
$$

where $\chi$ is the chromatic number (CB IDs after compaction) and $\omega$ is the clique number (maximum simultaneously-alive CBs). The equality $\chi = \omega$ holds because CB lifetime intervals are contiguous, making the interval graph perfect. The inequality holds because fusion allows temporal sharing between ops that would be in separate dispatches otherwise.

### L1 Budget for the Fused Gated MLP

Using the performance precision profile (bfloat8_b weights, bfloat16 activations), per-core budget on Blackhole (~1.5 MB total L1, ~1.3 MB usable for CBs):

```text
Phase 0 (Nodes 1-6): mcast + gate_mm + up_mm + SiLU*up
  Activation CB (mcast): 2 tiles * 2,048 B = 4 KB (streaming)
  Gate weight shard: DRAM-streaming (2 pages double-buffered): 2 * 1,088 = 2 KB
  Up weight shard: same = 2 KB
  Gate output CB: 25 tiles * 2,048 B = 50 KB (scratch)
  Up output CB: 25 tiles * 2,048 B = 50 KB (scratch)
  SiLU*up output CB: 25 tiles * 2,048 B = 50 KB (scratch)
  Scratch total: 4 + 2 + 2 + 50 + 50 + 50 = 158 KB

Phase 1 (Nodes 7-11): mcast + down_mm + residual
  Mcast input: 2 tiles * 2,048 B = 4 KB
  Down weight shard: DRAM-streaming: 2 * 1,088 = 2 KB
  Down output CB: 9 tiles * 2,048 B = 18 KB
  Residual add output: 9 tiles * 2,048 B = 18 KB
  Scratch total: 4 + 2 + 18 + 18 = 42 KB

Peak across phases: ~158 KB
L1 utilization: 158 / 1,300 = 12.2%
CB IDs used: Phase 0 = ~8, Phase 1 = ~5 (with reuse) = ~10 total (of 64)
```

The fused gated MLP fits comfortably in L1. The `reconfig()` boundary between Phase 0 and Phase 1 is essential: Phase 0's scratch CBs (158 KB) are freed before Phase 1 allocates its scratch CBs (42 KB).

---

## Manual vs. Abstracted Fusion: Side-by-Side Comparison

### The Manual Path (Today)

```python
# EXISTING -- Manual fusion (simplified, showing only the gate matmul phase)
f = FusedProgram(kernel=swiglu_kernel, device=device,
                 math_fidelity=ttnn.MathFidelity.LoFi,
                 math_approx_mode=True)

# 1. Allocate activation CB
act_tile_info = TileInfo.from_tensor(act_tensor)
in0 = f.cb_from_tensor(act_tensor)

# 2. Allocate weight CBs (direct-address)
gate_wt, gate_addr = resolve_weight_direct_address(
    f, gate_tensor, num_cores=14, ...)
up_wt, up_addr = resolve_weight_direct_address(
    f, up_tensor, num_cores=14, ...)

# 3. Compute CT args (manually!)
k_tiles = 4096 // 32       # = 128
gate_n_tiles = 11008 // 32 # = 344
gate_w_per_core = gate_n_tiles // 14  # = ~25

# 4. Emit gate matmul
gate_out = DRAMStreamingMatmul.emit(
    f, in0, gate_wt, prefix="gate_mm",
    k_num_tiles=k_tiles,
    out_w_per_core=gate_w_per_core)

# 5. Emit up matmul (repeat for up projection)
# ... 6. Emit SiLU + multiply ...
# ... 7. Gather + mcast ...
# ... 8. reconfig() ...
# ... 9. Emit down matmul ...
# ... 10. Wire output ...
f.wire_output(final_out, output_tensor)
result = f.build()

# Manual decisions required: 30+ per fused MLP
```

### The Abstracted Path (Proposed)

```python
# PROPOSED -- Abstracted fusion via TensorAdapter (12 lines)
adapter = TensorAdapter(device=mesh_device, precision="performance")

with blaze.fuse(shape_engine=adapter.shape_engine) as ctx:
    act = ExternalTensor.from_torch("activation", act_tensor, op_hint="matmul.in0")
    gate_wt = ExternalTensor.from_ttnn("gate_weights", gate_weight_tensor)
    up_wt = ExternalTensor.from_ttnn("up_weights", up_weight_tensor)
    down_wt = ExternalTensor.from_ttnn("down_weights", down_weight_tensor)

    gate = blaze.matmul({"in0": act, "in1": gate_wt}, grid=core_grid)
    up = blaze.matmul({"in0": act, "in1": up_wt}, grid=core_grid)
    silu_gate = blaze.silu({"input": gate}, grid=core_grid)
    gated = blaze.eltwise_mul({"in0": silu_gate, "in1": up}, grid=core_grid)
    result = blaze.matmul({"in0": gated, "in1": down_wt}, grid=core_grid)

# All 30+ decisions automatic: shape, padding, format, CT args, reconfig
auto_args = derive_user_args(ctx.graph)
program = compiler.compile(graph=ctx.graph, tensors={...}, user_args=auto_args)
```

The developer writes 12 lines instead of 200+. The kernel binary produced is identical.

### Comparison Table

| Aspect | Manual | Abstracted | Reduction |
|--------|--------|-----------|-----------|
| Tile geometry decisions | 6+ per fused op | 0 (auto from op_hint) | 100% |
| Format decisions | 6+ per fused op | 1 (precision profile) | 83% |
| Page count decisions | 6+ per fused op | 0 (auto from shape + blocking) | 100% |
| CT arg computations | 10+ per fused op | 0 (auto from ShapeDescriptor) | 100% |
| Padding decisions | 0 (only if non-aligned) | 0 (auto from registry) | N/A |
| Reconfig placement | Manual | Auto at compose() boundaries | 100% |
| L1 budget validation | None (runtime crash) | Compile-time check | Qualitative |
| Lines of code (full MLP) | ~200 | ~12 | 94% |

---

## Shape Tracking Through the Fusion Model

### ShapeTracker in FusedOp.compose()

When TensorAdapter manages a fused op, the ShapeTracker (from [Chapter 3, File 02](../ch03_tensor_adapter/02_shape_metadata_system.md)) records the output `ShapeDescriptor` after each `emit()` call within `compose()`:

```python
# PROPOSED -- ShapeTracker integration with compose()
class FusionShapeManager:
    """Manages shape tracking within a FusedOp.compose() scope."""

    def __init__(self, adapter: TensorAdapter):
        self._adapter = adapter
        self._tracker = ShapeTracker()
        self._phase = 0

    def wrap_emit(self, op_class, f, *args, prefix="", **kwargs):
        """Wrap a MicroOp.emit() call with shape tracking.

        1. Look up input ShapeDescriptors from the tracker
        2. Call the existing emit() unchanged (P2)
        3. Infer the output ShapeDescriptor
        4. Record the output in the tracker
        5. Return a TypedHandle
        """
        input_descs = {}
        for i, arg in enumerate(args):
            if isinstance(arg, (CBHandle, TypedHandle)):
                desc = self._tracker.lookup(
                    arg.handle if isinstance(arg, TypedHandle) else arg)
                if desc:
                    input_descs[f"in{i}"] = desc

        out_handle = op_class.emit(f, *args, prefix=prefix, **kwargs)

        out_desc = self._adapter.shape_engine.infer_output(
            op_class.name, input_descs)
        self._tracker.record(out_handle, out_desc)

        return TypedHandle(handle=out_handle, shape=out_desc)

    def checkpoint_reconfig(self):
        """Called at f.reconfig() boundaries.

        Records which CBHandles are tensor-backed (survive reconfig)
        and which are scratch (invalidated by reconfig).
        """
        self._phase += 1
```

---

## Error Taxonomy: What Goes Wrong in Fusion

| Error | Root Cause | Symptom | When Caught |
|-------|-----------|---------|------------|
| **Shape mismatch at emit() boundary** | Incorrect `k_num_tiles` or `out_w_per_core` | Silent numerical corruption | Runtime (manual) / Compile-time (abstracted) |
| **Padding strategy disagreement** | ZERO padding fed to softmax within fused op | Probability mass leaked to padding positions | Runtime (manual) / Compile-time (abstracted) |
| **Format incompatibility at reconfig** | Phase 0 writes bfloat8_b, Phase 1 reads as bfloat16 via same CB ID | Garbage data | Runtime (manual) / Compile-time (abstracted) |
| **L1 overcommit in fused op** | Too many scratch CBs alive simultaneously | Silent data corruption or hang | Runtime crash (manual) / Compile-time (abstracted) |
| **CB ID exhaustion** | >64 unique CB IDs needed | `RuntimeError` from CircularBufferIdManager | Compile-time in both cases |
| **Stale ShapeDescriptor after reconfig** | Scratch CB invalidated but ShapeTracker holds old descriptor | Wrong shape for downstream op | Compile-time: `checkpoint_reconfig()` clears stale entries |

---

## Integration Points with Prior Chapters

| Chapter | Integration with Fusion |
|---------|----------------------|
| [Ch3: TensorAdapter](../ch03_tensor_adapter/01_architecture_overview.md) | `adapt()` wraps tensors before `emit()`; ShapeTracker propagates descriptors through chains |
| [Ch4: Tile Decomposition](../ch04_tile_decomposition/01_three_shape_layers.md) | Each sub-op selects its own tile shape; matmul uses (32,32), RMSNorm uses (1,32) |
| [Ch5: Padding](../ch05_padding/01_padding_strategy_taxonomy.md) | Consumer-determines-padding applies at each internal edge; re-padding inserted at mismatches |
| [Ch6: Data Formats](../ch06_data_formats/01_data_format_landscape.md) | Format negotiation runs per-edge within fused subgraph; `cb_alias` for zero-cost transitions |
| [Ch7: CB Sizing](../ch07_cb_sizing/01_l1_memory_model.md) | Phase boundaries enable L1 reuse; total L1 budget validated per phase |
| [Ch8: Broadcasting](../ch08_broadcasting/01_pytorch_broadcasting_rules.md) | Broadcast detection at fused boundaries; dual-shape TypedHandle carries physical vs. logical |

---

## Key Takeaways

- TT-Blaze's fusion model has two levels: **MicroOp** (single C++ kernel phase, `emit()`) and **FusedOp** (composition of MicroOps, `compose()`). The `compose()` method chains `emit()` calls within a shared `FusedProgram`, threading output handles as inputs to subsequent ops.
- The **`blaze.fuse()` context manager** creates a `FusionContext` that progresses through EMPTY -> BUILDING -> VALIDATING -> VALIDATED -> COMPILED -> EXECUTED, with formal pre/post conditions and validation at each transition (P3 -- Validate at Each Lowering Step, Not Just at the End).
- The **Graph API** (`blaze.matmul(...)`) creates graph nodes declaratively; the **Composition API** (`Matmul.emit(f, ...)`) composes imperatively. TensorAdapter supports both: shape-enriched `FusionResult` for the Graph path, `ShapeTracker` for the Composition path.
- The **11-node gated MLP** in `_gated_mlp.py` is the canonical fusion template, achieving a 3.75x latency improvement: 11 dispatches to 1, 200 KB DRAM traffic eliminated, NOC transactions reduced by 3.7x.
- **SwigluOp.compose()** chains 5 MicroOps through 6 stages (9 CBs total) with 2 reconfig boundaries, reducing peak CB IDs from 14 (uncompacted) to 7 (with reconfig reuse). Interval coloring achieves $\chi(\text{fused}) = \omega(\text{fused})$, the optimal CB ID assignment.
- The abstracted path reduces 30+ manual decisions per fused MLP to 12 lines of code. The kernel binary produced is identical to hand-written code.

## Source Files

- `blaze/blaze_op.py` -- `BlazeOp` (base), `MicroOp` (leaf), `FusedOp` (composite), `emit()`, `compose()`, `register()`, `Input`/`Output`/`Internal` port descriptors
- `blaze/_gated_mlp.py` -- `build_gated_mlp_graph()`: 11-node gated MLP graph construction, the template for automatic fusion
- `blaze/ops/swiglu/op.py` -- `SwigluOp.compose()`: chains DRAMStreamingMatmul, EltwiseMul, Gather, Mcast
- `blaze/context.py` -- `FusionContext`, `FusionResult`, `ExternalTensor`, `fuse()` context manager
- `blaze/fused_program.py` -- `FusedProgram`: `cb_from_tensor()`, `cb_scratch()`, `reconfig()`, `build()`
- `blaze/cb_reconfig.py` -- `CircularBufferIdManager`: format-aware CB ID reuse across phases
- `blaze/compiler.py` -- `BlazeCompiler.compile()`: Graph API to Composition API lowering

---

**Next:** [`02_shape_and_padding_across_fused_boundaries.md`](./02_shape_and_padding_across_fused_boundaries.md)
