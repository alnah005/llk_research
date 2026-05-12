# Fusion Detection and Limitations

The preceding three files described what happens *within* a fused op: shape tracking, padding transitions, and format management. This file addresses the boundary question: how does TT-Blaze decide *which* ops to fuse in the first place, what patterns does the fusion analyzer recognize, what limitations prevent certain op sequences from being fused, and what happens when fusion fails? The fusion analyzer must identify fusible patterns in the PyTorch graph, map them to known FusedOp templates, validate that constraints (L1 budget, CB count, format compatibility) are satisfiable, and fall back gracefully to sequential execution when fusion is not possible.

> *Design Principle P5 (Operate on Graphs, Not Individual Tensors):* Fusion detection is a graph-level pattern matching problem. The analyzer must see the full subgraph to identify fusible sequences, check that all intermediate shapes are compatible, and validate that the fused result fits within hardware constraints. Single-tensor analysis cannot determine fusibility.

> *Design Principle P3 (Validate at Each Lowering Step, Not Just at the End):* Before committing to a fusion decision, every constraint must be checked: L1 budget for the fused op, CB count within the 64-slot limit, format compatibility at each internal edge, and padding strategy transitions at each boundary. Validation failures cause graceful fallback, not crashes.

> *Design Principle P7 (Transparent Interception That Preserves Identity):* When fusion is not possible, the system falls back to sequential execution that produces identical results to unfused execution. The developer never sees different numerical results depending on whether fusion succeeded or failed. This transparency extends to the Level 0 developer experience: fusion happens behind the scenes with no visible API change.

**What you will learn:**

- The fusion analyzer's pattern detection algorithm: how PyTorch op sequences are matched to known FusedOp templates via a pattern registry
- The canonical fusion patterns for LLM inference: gated MLP (SwiGLU), dense MLP (GeLU), fused attention (QKV + softmax + AV), residual connection, RMSNorm + linear, and more
- Fusion limitations: control flow, dynamic shapes, unsupported ops, L1 budget overflow, CB ID exhaustion, and incompatible core grids
- The fallback strategy: graceful degradation from fused to sequential execution with per-op adaptation
- The fusion benefit model: a quantitative formula for estimating when fusion is worthwhile
- The testing methodology: golden comparison against PyTorch CPU, tolerance thresholds per precision profile, a systematic 90-test matrix, and parallel validation via TensorAdapterValidator
- The op mapping registry: how each PyTorch `torch.nn.Module` maps to one or more TT-Blaze `FusedOp` or `MicroOp` implementations
- The developer journey: how Level 0, Level 1, and Level 2 developers interact with the fusion system

---

## The Fusion Analyzer

### Architecture

The fusion analyzer sits between the PyTorch graph trace and the TT-Blaze compilation pipeline. It receives a traced graph (from `torch.fx` or a similar tracing mechanism), identifies subgraphs that match known fusion patterns, and replaces them with single FusedOp nodes. The analyzer operates in three phases:

```text
Phase 1: Pattern Detection
  Input: BlazeGraph (from blaze.fuse() or from TensorAdapter's graph builder)
  Scan for known fusible subsequences
  Output: list of FusionCandidate objects

Phase 2: Feasibility Validation
  For each candidate:
    Check L1 budget (will the fused op fit in ~1.5 MB per core?)
    Check CB ID budget (will total IDs stay under 64?)
    Check format compatibility (are all internal edges resolvable?)
    Check padding compatibility (are all internal boundaries compatible?)
    Check core grid alignment (do all sub-ops use compatible grids?)
  Output: list of validated FusionPlan objects

Phase 3: Fusion Application
  For each validated plan:
    Replace the subgraph with a FusedOp node
    Attach ShapeDescriptors to all edges
    Insert reconfig boundaries where needed
  Output: modified BlazeGraph with fused regions
```

### The Pattern Detection Algorithm

The fusion analyzer uses a two-phase approach: **candidate identification** followed by **constraint validation**. Pattern matching is template-based, with each pattern defined as a sequence of op types with connectivity constraints:

```python
# PROPOSED -- Fusion analyzer
class FusionAnalyzer:
    """Identifies fusible op sequences in a traced PyTorch graph.

    The analyzer maintains a registry of known fusion patterns.
    Each pattern specifies:
      - A sequence of PyTorch op types (the "trigger")
      - A corresponding FusedOp class (the "replacement")
      - Constraint functions that validate fusibility
      - Priority for overlapping patterns
    """

    def __init__(self, registry: FusionPatternRegistry,
                 device_config: DeviceConfig):
        self._registry = registry
        self._device_config = device_config
        self._stats = FusionStats()

    def analyze(self, fx_graph) -> FusionPlan:
        """Identify all fusible subgraphs in the traced graph.

        Phase 1: Candidate identification
          Walk the graph, matching op sequences against the registry.
          Longer patterns have higher priority (greedy matching).

        Phase 2: Constraint validation
          For each candidate, check:
            - L1 budget: will all intermediate CBs fit in ~1.5 MB?
            - CB count: will the fused op need <= 64 CB IDs?
            - Format compatibility: are all internal edges resolvable?
            - Padding compatibility: are all internal edges compatible
              or re-paddable?
            - Shape validity: are all intermediate shapes inferrable?
            - Core grid alignment: can all sub-ops share a grid?

        Phase 3: Conflict resolution
          When candidates overlap (share nodes), select the highest-
          priority candidate that passes all constraints.

        Returns a FusionPlan with the selected fusions and fallback
        nodes for rejected candidates.
        """
        candidates = self._identify_candidates(fx_graph)
        validated = self._validate_candidates(candidates)
        plan = self._resolve_conflicts(validated)
        return plan
```

### Subgraph Pattern Matching

The analyzer scans the BlazeGraph for known patterns using the template-based approach. Each pattern is defined as a sequence of op types with connectivity constraints. Template entries support exact names, alternation (`{silu|relu|gelu}`), and wildcards (`*`):

```python
# PROPOSED -- pattern matching with alternation and wildcard support
def _try_match(self, topo_order, start_idx, pattern, claimed):
    """Try to match a pattern starting at start_idx in topo order."""
    matched_nodes = []
    template_idx = 0
    search_idx = start_idx

    while template_idx < len(pattern.ops) and search_idx < len(topo_order):
        node = topo_order[search_idx]
        if node.id in claimed:
            search_idx += 1
            continue

        expected_op = pattern.ops[template_idx]
        if self._op_matches(node.spec.name, expected_op):
            matched_nodes.append(node)
            template_idx += 1

        search_idx += 1

    if template_idx == len(pattern.ops):
        edges = self._extract_internal_edges(matched_nodes)
        return MatchCandidate(nodes=matched_nodes, edges=edges)

    return None  # Incomplete match

def _op_matches(self, actual: str, expected: str) -> bool:
    """Check if an op name matches a template entry."""
    if expected == "*":
        return True
    if expected.startswith("{") and expected.endswith("}"):
        alternatives = expected[1:-1].split("|")
        return actual in alternatives
    return actual == expected
```

---

## The Canonical Fusion Patterns

### Pattern Registry Structure

```python
# PROPOSED -- Fusion pattern registry
@dataclass(frozen=True)
class FusionPattern:
    """A recognized fusible op sequence."""
    name: str
    ops: list[str]                    # Op type sequence (with alternation)
    connectivity: dict                # Connectivity constraints
    fused_op_class: type | str        # TT-Blaze FusedOp class
    priority: int                      # Higher = matched first
    min_ops: int                       # Minimum ops to trigger
    max_ops: int                       # Maximum ops in the pattern
    constraints: tuple                 # Additional constraint functions
```

### Pattern 1: Gated MLP (SwiGLU) -- Priority 100

```text
Pattern name:    gated_mlp / swiglu
Trigger ops:     (linear, linear, silu, mul, linear, add)
                 gate_proj  up_proj  silu  gate*up  down_proj  residual
FusedOp class:   SwigluOp or build_gated_mlp_graph()
Priority:        100 (highest -- most complex pattern)
Op count:        6 (minimum) to 11 (with mcast/gather)

PyTorch source:
    def forward(self, x, residual=None):
        gate = self.gate_proj(x)        # linear
        up = self.up_proj(x)            # linear
        gate = F.silu(gate)             # silu
        x = gate * up                   # mul
        x = self.down_proj(x)           # linear
        if residual is not None:
            x = x + residual            # add
        return x

TT-Blaze mapping (11-node Graph API):
    1. mcast(activation)             -> Mcast MicroOp
    2. kn_sliced_matmul(act, gate)   -> DRAMStreamingMatmul
    3. kn_sliced_matmul(act, up)     -> DRAMStreamingMatmul
    4. gather(gate_partial)          -> Gather MicroOp
    5. gather(up_partial)            -> Gather MicroOp
    6. gated_reduce(gate, up)        -> EltwiseMul (fused SiLU + mul)
    7. mcast(reduced)                -> Mcast MicroOp
    8. matmul(reduced, down)         -> DRAMStreamingMatmul
    9. mcast(bias)                   -> Mcast MicroOp (optional)
    10. residual_add(mm_out, res)    -> ResidualAdd MicroOp
    11. gather(result)               -> Gather MicroOp
```

### Pattern 2: Dense MLP (GeLU) -- Priority 90

```text
Pattern name:    dense_mlp
Trigger ops:     (linear, gelu, linear, add)
FusedOp class:   DenseMlpOp
Priority:        90

TT-Blaze mapping:
    1. mcast(activation)             -> Mcast
    2. matmul(act, up_weights)       -> DRAMStreamingMatmul
    3. gelu(mm_out)                  -> GeLU MicroOp
    4. mcast(gelu_out)               -> Mcast
    5. matmul(gelu_out, down_wt)     -> DRAMStreamingMatmul
    6. residual_add                  -> ResidualAdd
    7. gather(result)                -> Gather
```

### Pattern 3: Fused Attention (QKV + Softmax + AV) -- Priority 95

```text
Pattern name:    fused_attention
Trigger ops:     (linear, linear, linear, matmul, softmax, matmul, linear)
                 Q_proj  K_proj  V_proj  QK^T    softmax  AV      O_proj
FusedOp class:   FusedAttentionOp
Priority:        95

Constraints:
  - Softmax must use HiFi4 + bfloat16 + float32 accumulation
    (softmax safety override, regardless of precision profile)
  - Sequence length determines padding strategy (NEG_INF for non-aligned)
  - Head dimension must be tile-aligned (64, 128 -- standard for LLMs)
```

### Pattern 4: RMSNorm + Linear -- Priority 70

```text
Pattern name:    rmsnorm_linear
Trigger ops:     (rmsnorm, linear)
FusedOp class:   RMSNormLinearOp
Priority:        70

TT-Blaze mapping:
    1. PaddedRMSNorm(input, gamma)   -> PaddedRMSNorm MicroOp
    2. mcast(normed)                 -> Mcast
    3. matmul(normed, weights)       -> DRAMStreamingMatmul
```

### Pattern 5: Linear + Residual -- Priority 50

```text
Pattern name:    linear_residual
Trigger ops:     (linear, add)
FusedOp class:   LinearResidualOp
Priority:        50 (lower -- subsumed by larger patterns)
```

### Pattern 6: Linear + Activation -- Priority 40

```text
Pattern name:    linear_activation
Trigger ops:     (matmul, {silu|relu|gelu})
FusedOp class:   (generic composition)
Priority:        40
```

### Pattern Priority and Overlap Resolution

When multiple patterns match overlapping nodes, the analyzer selects based on priority using a greedy strategy:

```python
# PROPOSED -- Conflict resolution
def _resolve_conflicts(self, validated: list) -> FusionPlan:
    """Select non-overlapping candidates by priority.

    Greedy: process candidates by descending priority.
    Once a node is assigned to a fusion, it cannot be claimed
    by a lower-priority pattern.
    """
    validated.sort(key=lambda v: v.candidate.pattern.priority, reverse=True)
    claimed_nodes = set()
    fusions = []
    fallbacks = []

    for v in validated:
        nodes = set(v.candidate.matched_nodes)
        if nodes & claimed_nodes:
            fallbacks.append(FallbackDecision(
                pattern=v.candidate.pattern.name,
                reason="overlap_with_higher_priority",
                nodes=list(nodes - claimed_nodes),
            ))
            continue
        fusions.append(v)
        claimed_nodes.update(nodes)

    return FusionPlan(fusions=fusions, fallbacks=fallbacks)
```

---

## The Op Mapping Registry

### PyTorch to TT-Blaze Op Mapping

The op mapping registry defines how each PyTorch operation maps to TT-Blaze primitives. Each mapping specifies the shape inference rule, format preference key, padding strategy, and whether a padded op variant is needed for non-aligned dimensions:

| PyTorch Op | TT-Blaze MicroOp(s) | Shape Rule | Padding | Streaming | Fusible |
|-----------|---------------------|-----------|---------|-----------|---------|
| `torch.nn.Linear` | `DRAMStreamingMatmul` or `Matmul` | contract K | ZERO | Yes | Yes |
| `torch.nn.functional.silu` | `EltwiseMul` (apply_silu=True) | preserve | ZERO | N/A | Yes |
| `torch.nn.functional.gelu` | `GeLU` MicroOp | preserve | ZERO | N/A | Yes |
| `torch.nn.functional.relu` | `ReLU` MicroOp | preserve | ZERO | N/A | Yes |
| `torch.nn.functional.softmax` | `Softmax` MicroOp | preserve | NEG_INF | N/A | Yes |
| `torch.nn.RMSNorm` | `PaddedRMSNorm` | preserve | ZERO | N/A | Yes |
| `torch.nn.LayerNorm` | `PaddedLayerNorm` | preserve | MASK | N/A | Yes |
| `torch.matmul` | `Matmul` or `DRAMStreamingMatmul` | contract K | ZERO | Yes | Yes |
| `torch.add` / `+` | `ResidualAdd` | broadcast | ZERO | N/A | Yes |
| `torch.mul` / `*` | `EltwiseMul` | broadcast | ZERO | N/A | Yes |
| `torch.mean(dim=...)` | `ReduceMean` | collapse dim | MASK | N/A | Yes |
| `torch.sum(dim=...)` | `ReduceSum` | collapse dim | ZERO | N/A | Yes |
| `torch.max(dim=...)` | `ReduceMax` | collapse dim | IDENTITY(-inf) | N/A | Yes |
| `F.scaled_dot_product_attention` | `SDPADecode` | attention | NEG_INF | N/A | Yes |
| `torch.nn.Embedding` | `EmbeddingGather` | lookup | N/A | N/A | Yes |
| `torch.transpose` | Layout metadata change | passthrough | N/A | N/A | N/A |
| `torch.reshape` | Layout metadata change | passthrough | N/A | N/A | N/A |
| `torch.cat(dim=...)` | N/A | N/A | N/A | N/A | **No** |
| `torch.sort()` | N/A | N/A | N/A | N/A | **No** |
| `torch.unique()` | N/A | N/A | N/A | N/A | **No** |
| `torch.scatter_()` | N/A | N/A | N/A | N/A | **No** |

### Size-Dependent Op Selection

Some PyTorch ops map to different TT-Blaze implementations depending on tensor dimensions:

```python
# PROPOSED -- Size-dependent matmul selection
def select_matmul_op(M: int, K: int, N: int, num_cores: int) -> type:
    """Select the optimal matmul implementation based on dimensions.

    DRAMStreamingMatmul: weights streamed from DRAM, best for large K.
    Matmul (L1-resident): weights pre-loaded to L1, best for small K.
    KNSlicedMatmul: K and N sliced across cores, best for wide outputs.
    """
    k_tiles = K // 32
    n_tiles = N // 32
    weight_size_bytes = k_tiles * n_tiles * 1088  # bfloat8_b

    if weight_size_bytes <= 1_500_000:  # Fits in ~1.5 MB L1
        return Matmul  # L1-resident
    elif n_tiles > num_cores * 4:
        return KNSlicedMatmul  # Wide output, slice across cores
    else:
        return DRAMStreamingMatmul  # Stream from DRAM
```

### Shape Rules

Each shape rule corresponds to an inference function:

| Shape Rule | Description | Example |
|-----------|------------|---------|
| `contract_k` | Matmul: `[M, K] x [K, N] -> [M, N]` | `[1, 4096] x [4096, 11008] -> [1, 11008]` |
| `preserve` | Output shape = input shape | SiLU, RMSNorm, Softmax |
| `broadcast` | PyTorch broadcast rules applied | `[128, 4096] + [1, 4096] -> [128, 4096]` |
| `collapse_dim` | Reduce dimension to 1 (keepdim) | `sum([128, 4096], dim=1) -> [128, 1]` |
| `attention` | Q/K/V shapes -> attention output | Complex multi-head logic |

---

## Constraint Validation

Each candidate undergoes six validation checks before being accepted:

```python
# PROPOSED -- Constraint validation
def _validate_candidates(self, candidates) -> list:
    validated = []
    for cand in candidates:
        issues = []

        # Check 1: L1 budget
        if cand.estimated_l1 > self._device_config.usable_l1_bytes:
            issues.append(FusionIssue(
                type="L1_BUDGET",
                message=f"Estimated L1 usage {cand.estimated_l1:,} bytes "
                        f"exceeds budget {self._device_config.usable_l1_bytes:,}",
                severity="BLOCK",
            ))

        # Check 2: CB count
        if cand.estimated_cbs > 64:
            issues.append(FusionIssue(
                type="CB_EXHAUSTION",
                message=f"Estimated {cand.estimated_cbs} CBs exceeds 64-slot limit",
                severity="BLOCK",
            ))

        # Check 3: Format compatibility
        format_issues = self._check_format_compatibility(cand)
        issues.extend(format_issues)

        # Check 4: Padding compatibility
        padding_issues = self._check_padding_compatibility(cand)
        issues.extend(padding_issues)

        # Check 5: Shape validity
        shape_issues = self._check_shape_validity(cand)
        issues.extend(shape_issues)

        # Check 6: Core grid alignment
        grid_issues = self._check_core_grid_alignment(cand)
        issues.extend(grid_issues)

        blocking_issues = [i for i in issues if i.severity == "BLOCK"]
        if not blocking_issues:
            validated.append(ValidatedCandidate(
                candidate=cand,
                warnings=[i for i in issues if i.severity == "WARN"],
            ))
            self._stats.candidates_validated += 1
        else:
            self._stats.candidates_rejected += 1
            self._stats.rejection_reasons[blocking_issues[0].type] += 1

    return validated
```

---

## Fusion Limitations

### Limitation Taxonomy

Not all op sequences can be fused. The following taxonomy classifies the six categories:

| Category | Limitation | Example | Severity |
|----------|-----------|---------|----------|
| **Control flow** | Conditional branches within fused region | `if x.sum() > 0: path_a else: path_b` | BLOCK |
| **Dynamic shapes** | Tensor shapes not known at compile time | `x[:, :seq_len]` where `seq_len` varies | BLOCK |
| **Unsupported ops** | Op has no TT-Blaze MicroOp implementation | `torch.nn.functional.instance_norm` | BLOCK |
| **L1 budget** | Fused op intermediate CBs exceed ~1.5 MB | Very wide MLP (intermediate_dim > 65536) | BLOCK |
| **CB exhaustion** | Fused op needs > 64 CB IDs | Deeply fused op with many scratch CBs | BLOCK |
| **Incompatible core grids** | Consecutive ops require different core grids | 4x8 matmul followed by 1x32 reduction | BLOCK |

### Limitation 1: Control Flow

TT-Blaze's fusion model assumes a fixed dataflow graph at compile time. The `FusedProgram` programs NCRISC and TRISC with fixed CB addresses, page counts, and CT args. Control flow within the fused region would require conditional CB allocation, dynamic CT args, and variable phase counts -- none supported by the current hardware programming model.

```text
Example (NOT fusible):

def forward(self, x):
    gate = self.gate_proj(x)
    up = self.up_proj(x)
    if self.use_silu:        # <-- Control flow breaks fusion
        gate = F.silu(gate)
    else:
        gate = F.gelu(gate)
    return gate * up

Workaround: Create two separate fused ops (one with SiLU, one
with GeLU) and dispatch based on the condition before entering
the fused region.
```

### Limitation 2: Dynamic Shapes

Fusion requires fixed shapes at compile time because:

1. **CB page counts** are computed from shapes: `num_pages = padded_width / tile_width`
2. **CT args** like `k_num_tiles` and `out_w_per_core` are derived from shapes
3. **L1 budget** validation depends on exact byte counts
4. **Weight sharding** across cores depends on matrix dimensions

```text
Example (NOT fusible):

def forward(self, x, mask):
    # x: [batch, seq_len, hidden]  where seq_len varies per call
    attn = self.attention(x, mask)
    return attn

Workaround: Use "bucketed compilation" -- pre-compile fused ops
for common seq_len values (128, 256, 512, 1024, 2048) and select
at runtime. The padding system (from Chapter 5 and File 02)
handles the difference between seq_len and the bucketed value.
```

### Limitation 3: Unsupported Operations

If any op in the candidate sequence does not have a Blaze implementation, the sequence cannot be fused:

```text
Unsupported ops that break fusion:
  torch.sort()          -- no Blaze equivalent (complex data movement)
  torch.unique()        -- no Blaze equivalent (dynamic output size)
  torch.scatter_()      -- limited Blaze support (index-dependent writes)
  torch.nonzero()       -- dynamic output size
  Custom Python ops     -- no kernel binary
```

When an unsupported op appears in the middle of a fusible sequence, the analyzer splits the sequence into two groups: the ops before the unsupported op and the ops after.

### Limitation 4: L1 Budget Overflow

Even when the pattern is recognized and all ops are supported, the fused op may not fit in L1:

```text
Example: attempting to fuse an MoE layer with 8 experts

Per expert: ~300 KB L1 (from Chapter 7, File 01)
8 experts simultaneous: 2,400 KB > ~1.5 MB budget

Resolution:
  Cannot fuse all 8 experts into a single FusedOp.
  Fall back to sequential expert execution (each expert is a separate
  fused dispatch, or groups of 2-3 experts are fused).
```

### Limitation 5: CB ID Exhaustion

Large fused ops can approach the 64 CB ID limit even after compaction. The fusion analyzer includes a safety margin from CBEngine's warning threshold of `max_cb_id - 8 = 56`:

```text
Example: GLM-5.1 DSA + MoE (from Chapter 7, File 04)
  Raw CB count: 55-60
  After compaction: 28-35

  If additional profiling or debug CBs are needed:
    35 + 30 debug CBs = 65 (EXCEEDS LIMIT)
```

### Limitation 6: Incompatible Core Grids

If consecutive ops require different core grids (e.g., a 4x8 matmul followed by a 1x32 reduction), the data must be redistributed between ops. While `Gather` and `Mcast` can handle some grid transitions within a fused op, certain transitions require external redistribution that breaks fusion:

```text
Example:
  Op A: matmul on 4x8 grid (32 cores, each producing N/32 output columns)
  Op B: row reduction on 1x1 grid (single core needs all columns)

  This requires an all-gather across 32 cores before the reduction.
  The all-gather cannot execute within a single FusedProgram because
  it requires NOC transactions that span the entire device grid.

Workaround: Fuse ops within each grid configuration separately.
  Fused group 1: matmul (4x8 grid)
  Sequential: all-gather (host-mediated)
  Fused group 2: reduction (1x1 grid)
```

---

## The Fallback Strategy

### Graceful Degradation: Fused -> Sequential

When a fusion candidate fails validation, the analyzer falls back to sequential execution. Each op in the rejected candidate is compiled independently using `TensorAdapter.adapt()`. The critical invariant, enforced by P7 (Transparent Interception That Preserves Identity): **sequential execution produces numerically identical results to fused execution**, modulo floating-point accumulation order differences documented per op.

```text
Fusion attempt: gated_mlp (6 PyTorch ops -> 1 SwigluOp)
  |
  Constraint check: L1 budget exceeded (intermediate_dim = 65536)
  |
  FALLBACK: Execute as 6 separate adapted ops

Sequential execution:
  Op 1: gate_proj = adapt(Linear(4096, 65536))
    -> DRAMStreamingMatmul, compiled independently
    -> Output written to DRAM (inter-op round-trip)

  Op 2: up_proj = adapt(Linear(4096, 65536))
    -> DRAMStreamingMatmul, compiled independently
    -> Output written to DRAM

  Op 3: silu(gate_proj_output)
    -> SiLU MicroOp, compiled independently

  Op 4: gate * up
    -> EltwiseMul MicroOp, compiled independently

  Op 5: down_proj = adapt(Linear(65536, 4096))
    -> DRAMStreamingMatmul, compiled independently

  Op 6: residual_add
    -> ResidualAdd MicroOp, compiled independently
```

### Performance Cost of Fallback

| Metric | Fused (SwigluOp) | Sequential (6 ops) | Overhead |
|--------|-----------------|-------------------|----------|
| DRAM round-trips | 2 (input + output) | 12 (2 per op) | 6x |
| Kernel launches | 1 | 6 | 6x |
| Intermediate DRAM writes | 0 (all in L1) | 5 | N/A |
| L1 utilization | High (streaming) | Low (per-op) | ~3x waste |
| Latency (LLaMA-7B MLP) | ~12 us | ~45 us | ~3.75x |

### Partial Fusion

When a full pattern cannot be fused, the analyzer may find a partial match:

```text
Full pattern: gate_proj + up_proj + silu + mul + down_proj + add
Full fusion fails: L1 budget exceeded

Partial match 1: gate_proj + up_proj + silu + mul (4 ops -> gated_reduce)
  L1 check: passes (no down_proj weight shard needed)

Partial match 2: down_proj + add (2 ops -> linear_residual)
  L1 check: passes (single matmul fits)

Result: 2 fused ops + 0 sequential ops
  Fused 1: gated_reduce (4 ops fused)
  Fused 2: linear_residual (2 ops fused)
  Inter-fusion DRAM round-trip: 1 (gated_reduce output -> linear_residual input)
```

Partial fusion is better than no fusion: it reduces DRAM round-trips from 5 (fully sequential) to 1 (between the two fused groups).

---

## The Fusion Benefit Model

### Quantifying When Fusion Is Worthwhile

The benefit of fusion is modeled as:

$$
\text{benefit} = T_{\text{dispatch}} \times (N_{\text{ops}} - 1) + T_{\text{DRAM}} \times N_{\text{intermediates}} - T_{\text{overhead}}
$$

where:
- $T_{\text{dispatch}}$ is the per-dispatch overhead (~2-3 us on Blackhole)
- $N_{\text{ops}}$ is the number of ops fused
- $T_{\text{DRAM}}$ is the per-intermediate DRAM round-trip time
- $N_{\text{intermediates}}$ is the number of intermediate tensors eliminated
- $T_{\text{overhead}}$ is the fusion overhead (reconfig kernel invocations, L1 pressure)

### Quantified Examples

```text
LLaMA-7B Gated MLP (4096 -> 11008 -> 4096):
  N_ops = 11
  N_intermediates = 10
  T_dispatch = 2.5 us
  T_DRAM per intermediate:
    mcast_out [1, 4096]: 8 KB / 400 GB/s = 0.02 us (negligible)
    gate_result [1, 11008]: 22 KB / 400 GB/s = 0.055 us
  T_overhead: 2 reconfig invocations x 5 us = 10 us

  Benefit = 2.5 * 10 + 0.5 * 10 - 10 = 25 + 5 - 10 = 20 us
  Sequential: 45 us
  Fused: 45 - 20 = 25 us -> actual measured ~12 us
  (additional speedup from L1 locality not captured by model)

DeepSeek V3 MoE Expert (7168 -> 2048 -> 7168):
  N_ops = 6 (SwigluOp pattern)
  N_intermediates = 5
  T_dispatch = 2.5 us

  Benefit = 2.5 * 5 + 0.5 * 5 - 5 = 12.5 + 2.5 - 5 = 10 us
  Per-layer improvement: 10 us * 256 experts = 2,560 us (significant for MoE)
```

### When Fusion Is Not Worth It

For very small fusion groups with tiny tensors, the reconfig overhead can negate the benefit:

```text
Small fusion: Bias Addition + SiLU
  Input: [1, 128] (4 tiles)

  N_ops = 2, N_intermediates = 1
  T_DRAM: negligible (128 * 2 = 256 bytes)
  T_dispatch: 2.5 us saved
  T_overhead: 0 (single phase, no reconfig)

  Net benefit: 2.5 us -- worthwhile

  But if we needed a reconfig boundary:
  T_overhead: 5 us
  Net benefit: 2.5 - 5 = -2.5 us (fusion is SLOWER)
```

The fusion analyzer applies a minimum benefit threshold:

```python
# PROPOSED -- minimum benefit threshold
FUSION_BENEFIT_THRESHOLD_US = 2.0

def should_fuse(candidate: FusionCandidate) -> bool:
    estimated_benefit = estimate_fusion_benefit(candidate)
    return estimated_benefit >= FUSION_BENEFIT_THRESHOLD_US
```

---

## Testing Strategy

### Golden Comparison Against PyTorch CPU

The primary correctness validation for fused ops is comparison against PyTorch CPU execution:

```python
# PROPOSED -- Fusion correctness test framework
class FusionCorrectnessTest:
    """Test that fused execution produces the same result as sequential."""

    def __init__(self, precision_profile: str):
        self._profile = precision_profile
        self._tolerances = TOLERANCE_TABLE[precision_profile]

    def test_fused_vs_sequential(
        self,
        module: torch.nn.Module,
        sample_input: torch.Tensor,
    ) -> TestResult:
        """Compare fused and sequential execution.

        1. Run the PyTorch module on CPU -> reference output
        2. Run the same module with fusion enabled -> fused output
        3. Run the same module with fusion disabled -> sequential output
        4. Compare: fused vs reference, sequential vs reference
        """
        # Step 1: PyTorch CPU reference
        with torch.no_grad():
            ref_output = module(sample_input)

        # Step 2: Fused execution on device
        blaze_module_fused = BlazeModule.from_torch(
            module, sample_input=sample_input, device=self._device,
            fusion=True,
        )
        fused_output = blaze_module_fused(sample_input)

        # Step 3: Sequential execution on device
        blaze_module_seq = BlazeModule.from_torch(
            module, sample_input=sample_input, device=self._device,
            fusion=False,
        )
        seq_output = blaze_module_seq(sample_input)

        # Step 4: Compare
        fused_vs_ref = self._compare(fused_output, ref_output)
        seq_vs_ref = self._compare(seq_output, ref_output)
        fused_vs_seq = self._compare(fused_output, seq_output)

        return TestResult(
            fused_vs_ref=fused_vs_ref,
            seq_vs_ref=seq_vs_ref,
            fused_vs_seq=fused_vs_seq,
        )
```

### Tolerance Thresholds Per Precision Profile

| Profile | Max Absolute Error | Max Relative Error | Math Fidelity | Metric |
|---------|-------------------|-------------------|---------------|--------|
| performance | $5 \times 10^{-3}$ | $2 \times 10^{-2}$ | LoFi | Per-element |
| balanced | $1 \times 10^{-4}$ | $5 \times 10^{-4}$ | HiFi4 | Per-element |
| accuracy | $1 \times 10^{-6}$ | $1 \times 10^{-5}$ | HiFi4 | Per-element |

These thresholds account for:
- **performance:** LoFi math fidelity + bfloat8_b weight compression introduces quantization noise
- **balanced:** HiFi4 + bfloat16 is numerically close to PyTorch CPU float32
- **accuracy:** float32 accumulation matches CPU float32 to ~6 decimal places

### The 90-Test Matrix

Correctness is validated systematically across the cross product of patterns, shapes, and profiles:

```text
Test matrix:

Patterns (5):
  gated_mlp, swiglu, linear_activation, attention_qkv, sdpa_block

Shapes (6):
  - Aligned:     hidden=4096, intermediate=11008 (LLaMA-7B standard)
  - Non-aligned: hidden=4001, intermediate=10000 (padding required)
  - Small:       hidden=768, intermediate=3072 (GPT-2 scale)
  - Sub-tile:    hidden=7 (extreme padding case)
  - Batch:       batch=32, hidden=4096 (multi-batch)
  - Sequence:    seq=37, d_head=128, heads=32 (non-aligned attention)

Profiles (3):
  performance, balanced, accuracy

Total test cases: 5 patterns x 6 shapes x 3 profiles = 90 golden tests
```

For each combination:
1. **Fused vs. PyTorch CPU**: must be within profile-specific tolerance
2. **Fused vs. Sequential** (same device): must be bit-exact for same precision profile -- any difference indicates a fusion bug
3. **Sequential vs. PyTorch CPU**: must be within tolerance

### Parallel Validation via TensorAdapterValidator

During development, TensorAdapter validates its automatic decisions by running both manual and automatic derivation paths and comparing results:

```python
# PROPOSED -- parallel validation of automatic decisions
class TensorAdapterValidator:
    """Runs manual and automatic paths in parallel, asserts equality.

    Active during development and CI testing. Disabled in production
    for performance (the automatic path runs alone).
    """

    def validate_ct_args(
        self,
        op_name: str,
        manual_args: dict[str, int],
        auto_args: dict[str, int],
    ) -> list[str]:
        """Compare manually-computed CT args with auto-derived CT args."""
        mismatches = []
        all_keys = set(manual_args) | set(auto_args)
        for key in sorted(all_keys):
            manual_val = manual_args.get(key)
            auto_val = auto_args.get(key)
            if manual_val != auto_val:
                mismatches.append(
                    f"CT arg '{key}': manual={manual_val}, auto={auto_val}"
                )
        return mismatches

    def validate_cb_allocation(
        self,
        manual_cbs: list[CBHandle],
        auto_cbs: list[CBHandle],
    ) -> list[str]:
        """Compare manually-allocated CBs with auto-allocated CBs."""
        mismatches = []
        if len(manual_cbs) != len(auto_cbs):
            mismatches.append(
                f"CB count: manual={len(manual_cbs)}, auto={len(auto_cbs)}"
            )
            return mismatches

        for i, (manual, auto) in enumerate(zip(manual_cbs, auto_cbs)):
            if manual.data_format != auto.data_format:
                mismatches.append(
                    f"CB {i} format: manual={manual.data_format}, "
                    f"auto={auto.data_format}"
                )
            if manual.page_size != auto.page_size:
                mismatches.append(
                    f"CB {i} page_size: manual={manual.page_size}, "
                    f"auto={auto.page_size}"
                )
            if manual.num_pages != auto.num_pages:
                mismatches.append(
                    f"CB {i} num_pages: manual={manual.num_pages}, "
                    f"auto={auto.num_pages}"
                )
        return mismatches
```

**What gets validated in parallel:**

| Decision | Manual Source | Automatic Source | Comparison |
|----------|-------------|-----------------|------------|
| `k_num_tiles` | Developer computes from tensor shapes | ShapeDescriptor.tile_grid_shape[-1] | Exact integer match |
| `out_w_per_core` | Developer computes from weight shape / cores | ShapeEngine.infer_matmul_output() | Exact integer match |
| CB page_size | Developer looks up tile.get_tile_size() | ShapeDescriptor.page_size | Exact byte match |
| CB num_pages | Developer computes from blocking strategy | CBBackend.allocate() | Exact page count |
| data_format | Developer selects per tensor | FormatPolicy negotiation | Exact format match |
| padding_strategy | Developer selects per op | PaddingPolicy registry | Enum match |

A single mismatch in any of these values produces a test failure. The validator catches bugs before they produce silent numerical corruption.

### Failure Diagnosis

When a golden test fails, the system compares intermediate results at each sub-op boundary between standalone and fused execution, identifying the first boundary where the maximum absolute difference exceeds the tolerance. This narrows the root cause to a specific sub-op (e.g., "divergence at silu: max_diff=3.7e-02") rather than reporting only the final output mismatch.

---

## Fusion Diagnostics and Reporting

### The FusionReport

After analysis, the fusion analyzer produces a diagnostic report:

```python
# PROPOSED -- Fusion report
@dataclass
class FusionReport:
    """Diagnostic report from the fusion analyzer."""
    total_ops: int                  # Total PyTorch ops in the graph
    fused_ops: int                  # Ops consumed by fusion
    sequential_ops: int             # Ops executing sequentially
    fusions: list[FusionEntry]      # Details of each fusion
    fallbacks: list[FallbackEntry]  # Details of each fallback
    estimated_speedup: float        # vs. fully sequential execution
    dram_roundtrips_saved: int      # Inter-op DRAM accesses eliminated
```

### Example Report for LLaMA-7B Decoder Layer

```text
Fusion Report for LLaMA-7B Decoder Layer
=========================================
Total PyTorch ops:         14
Fused ops:                 12 (85.7%)
Sequential ops:            2 (14.3%)
FusedOps created:          2

Fusion 1: gated_mlp (SwigluOp)
  Pattern:        gate_proj + up_proj + silu + mul + down_proj + residual_add
  PyTorch ops:    6 -> 1 FusedOp (11 MicroOps)
  L1 estimate:    154 KB scratch (Phase 0), 40 KB scratch (Phase 1)
  CB IDs:         10 (of 64)
  Reconfig:       1 boundary (Phase 0 -> Phase 1)
  Padding:        0 RepadOps (all ZERO-compatible)
  Format:         bfloat8_b weights, bfloat16 activations (performance)
  Constraints:    All passed

Fusion 2: rmsnorm_linear (RMSNormLinearOp)
  Pattern:        rmsnorm + q_proj
  PyTorch ops:    2 -> 1 FusedOp (3 MicroOps)
  L1 estimate:    62 KB scratch
  CB IDs:         5 (of 64)
  Reconfig:       0 boundaries
  Padding:        0 RepadOps
  Format:         bfloat16 throughout
  Constraints:    All passed

Sequential ops:
  1. Attention (fused_attention pattern REJECTED)
     Reason: Custom attention implementation with control flow
     Fallback: 4 sequential ops (Q/K/V projections + SDPA)
  2. RMSNorm (standalone, not adjacent to fusible linear)

Estimated speedup: 2.8x (vs. fully sequential)
DRAM round-trips saved: 8 (gated_mlp: 5, rmsnorm_linear: 3)
```

---

## The Developer Journey

### Level 0: PyTorch Developer

The Level 0 developer writes standard PyTorch code and expects it to run on Tenstorrent hardware. P7 (Transparent Interception That Preserves Identity) governs this experience: fusion happens behind the scenes with no visible API change and no change in numerical results.

```python
# Level 0 -- standard PyTorch, no Blaze knowledge
class LLaMAMLP(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# Deploy to Tenstorrent
model = LLaMAMLP(4096, 11008)
compiled = blaze.compile(model, example_input=torch.randn(1, 4096))
# Fusion analyzer detects gated_mlp pattern -> SwigluOp
# TensorAdapter configures all CBs, formats, padding automatically
output = compiled(input_tensor)
```

The developer does not know that fusion happened. They do not know about CBs, tile shapes, or padding strategies.

### Level 1: Aware Developer

The Level 1 developer inspects fusion decisions and provides hints:

```python
# Level 1 -- inspecting and hinting
compiled = blaze.compile(
    model,
    example_input=torch.randn(1, 4096),
    precision_profile="balanced",      # Override: use balanced precision
    weight_format="bfloat8_b",         # Override: compress all weights
)

# Inspect fusion decisions
for candidate in compiled.fusion_report:
    print(f"Fused: {candidate.pattern_name} -> {candidate.target_blaze_op}")
    print(f"  CBs: {candidate.estimated_cb_count}, L1: {candidate.estimated_l1_bytes:,}")
    if not candidate.is_valid:
        print(f"  REJECTED: {candidate.rejection_reason}")
```

### Level 2: Expert Developer

The Level 2 developer bypasses fusion and writes fused ops directly using the Composition API, with TensorAdapter providing shape/format/padding assistance:

```python
# Level 2 -- direct composition with TensorAdapter
adapter = TensorAdapter(device=device, profile="accuracy")

f = FusedProgram(kernel=kernel_path, device=device)
act = adapter.adapt(act_tensor, op_hint="matmul.in0", fused_program=f)
wt = adapter.adapt(wt_tensor, op_hint="matmul.in1", fused_program=f)

# Manual composition with automatic shape/format/padding
gate_out = Matmul.emit(f, act, wt, prefix="gate_mm")
adapter.shape_tracker.record(gate_out, adapter.shape_engine.infer_matmul_output(
    adapter.shape_tracker.lookup(act), adapter.shape_tracker.lookup(wt)))

# Override format for specific CB
f.ct_arg("gate_mm.fp32_dest_acc_en", True)  # Force float32 accumulation

result = f.build()
```

---

## Integration with Prior Chapters

The fusion detection system integrates with every prior chapter:

| Chapter | Integration Point |
|---------|-------------------|
| [Ch 3: TensorAdapter](../ch03_tensor_adapter/) | `adapt()` configures input CBs; `ShapeTracker` propagates metadata through fused chains |
| [Ch 4: Tile Decomposition](../ch04_tile_decomposition/) | `tile_decompose()` determines padded shapes and tile grids for each fused sub-op |
| [Ch 5: Padding](../ch05_padding/) | Padding pass inserts PadOps at fused boundaries where strategies conflict |
| [Ch 6: Data Formats](../ch06_data_formats/) | Format negotiation selects per-edge formats within the fused subgraph |
| [Ch 7: CB Sizing](../ch07_cb_sizing/) | L1 budget validation determines whether the fused op fits; compaction reduces CB count |
| [Ch 8: Broadcasting](../ch08_broadcasting/) | Broadcast shape tracking validates multi-core data distribution within fused ops |
| [Ch 9, File 01](./01_blaze_fusion_model.md) | The MicroOp/FusedOp hierarchy provides the target ops for fusion mapping |
| [Ch 9, File 02](./02_shape_and_padding_across_fused_boundaries.md) | Shape and padding propagation across fused boundaries |
| [Ch 9, File 03](./03_format_transitions_within_fused_ops.md) | Format transitions at internal edges within fused ops |

---

## Error Taxonomy: Fusion Detection Errors

| Error | Root Cause | Symptom | Resolution |
|-------|-----------|---------|------------|
| **False positive** | Constraint check underestimates L1 | L1BudgetError at compile time | Improve L1 estimator conservatism |
| **False negative** | Constraint check overestimates CBs | Performance loss (sequential) | Tighten CB estimator for reuse |
| **Pattern not recognized** | Op sequence not in registry | No fusion attempted | Add pattern to registry |
| **Partial match missed** | Analyzer skips sub-patterns after rejection | Full sequential instead of partial | Enable partial fusion search |
| **Wrong FusedOp selected** | Priority conflict resolved incorrectly | Suboptimal fusion | Review priority assignments |
| **Dynamic shape at trace** | torch.fx captures symbolic shape | Cannot validate constraints | Use bucketed compilation |

---

## Key Takeaways

- The fusion analyzer identifies fusible op sequences via a pattern registry with priority-based conflict resolution. Six canonical patterns cover the majority of LLM inference: gated MLP (SwiGLU, priority 100), fused attention (priority 95), dense MLP (GeLU, priority 90), RMSNorm + linear (priority 70), linear + residual (priority 50), and linear + activation (priority 40). Greedy longest-match ensures the most beneficial fusion is selected when patterns overlap.
- Six categories of limitations block fusion: control flow (conditional branches require dynamic CT args), dynamic shapes (CB page counts and CT args must be fixed at compile time), unsupported ops (no TT-Blaze implementation), L1 budget overflow (intermediate CBs exceed ~1.5 MB per Tensix core), CB ID exhaustion (> 64 IDs needed), and incompatible core grids (different sub-ops require different multi-core configurations). Each is detected during constraint validation.
- The fallback strategy provides graceful degradation governed by P7 (Transparent Interception That Preserves Identity): rejected fusion candidates are compiled as sequential adapted ops via `TensorAdapter.adapt()`, producing numerically identical results. Partial fusion is supported: when the full pattern fails, sub-patterns may still be fusible.
- The fusion benefit model quantifies speedup as $T_{\text{dispatch}} \times (N-1) + T_{\text{DRAM}} \times N_{\text{intermediates}} - T_{\text{overhead}}$. A minimum threshold of 2.0 us prevents fusion when the overhead exceeds the savings. For LLaMA-7B gated MLP, fusion eliminates 10 DRAM round-trips and yields a 3.75x latency improvement.
- Correctness testing uses a systematic 90-test matrix (5 patterns x 6 shapes x 3 profiles) with golden comparison against PyTorch CPU. Per-profile tolerance thresholds: $5 \times 10^{-3}$ (performance/LoFi), $1 \times 10^{-4}$ (balanced/HiFi4), $1 \times 10^{-6}$ (accuracy/HiFi4+fp32). Fused vs. sequential on the same device must be bit-exact.
- Parallel validation during development (TensorAdapterValidator) runs both manual and automatic derivation paths for every CT arg, CB allocation, format, and padding decision. A single integer mismatch produces a test failure, catching silent correctness bugs before production.
- Three developer levels interact with fusion: Level 0 writes standard PyTorch and fusion happens transparently; Level 1 inspects decisions via `FusionReport` and provides precision/format hints; Level 2 uses the Composition API directly with TensorAdapter assistance.

## Source Files

- `blaze/blaze_op.py` -- `MicroOp`, `FusedOp`: the two composition levels that fusion patterns map onto
- `blaze/_gated_mlp.py` -- `build_gated_mlp_graph()`: the canonical 11-node gated MLP fusion template
- `blaze/ops/swiglu/op.py` -- `SwigluOp.compose()`: the FusedOp implementation of the gated MLP pattern
- `blaze/context.py` -- `FusionContext`, `fuse()`: the context manager that captures op calls during graph construction
- `blaze/fused_program.py` -- `FusedProgram`: the compilation target for fused ops
- `blaze/cb_reconfig.py` -- `CircularBufferIdManager`: CB ID management that enables fusion across multiple phases
- `blaze/cb_reconfig_builder.py` -- `prepare_for_build()`: the pipeline that finalizes multi-phase fused programs
- `blaze/graph.py` -- `BlazeGraph`, `OpNode`, `Edge`: graph structures that the analyzer operates on
- `blaze/cb_engine.py` -- `MAX_CB_ID = 64`, `compact_cb_ids()`: CB ID limit and compaction
- `blaze/l1_profile.py` -- `print_cb_stats()`: post-hoc verification of fused op L1 usage

---

**Previous:** [03_format_transitions_within_fused_ops.md](./03_format_transitions_within_fused_ops.md) | **Next:** [Chapter 10 -- End-to-End Developer API, Performance, and Escape Hatches](../ch10_end_to_end/index.md)
