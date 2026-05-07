# 04 -- CT Arg Engine

The CTArgEngine is the third and final engine in the compilation pipeline. It reads each op's registered `CTArgSchema`, resolves concrete values from CB assignments, semaphore assignments, user kwargs, and grid context, validates against name collisions, and produces per-RISC lists of `(name, value)` tuples that feed directly into the `UnifiedKernelDescriptor`.

**Source file:** `blaze/ct_args.py`

---

## 1. The Problem: Compile-Time Arg Explosion

A fused kernel on Blackhole can have 130--400+ compile-time arguments, manually strung together as positional tuples in the original approach. Each arg must land on the correct RISC (NCRISC, BRISC, or TRISC), reference the correct CB ID or semaphore address, and have a unique name within its RISC. The CTArgEngine replaces this manual process with schema-driven, auto-generated, type-validated args derived entirely from graph context.

---

## 2. CTArgSchema and CTArgSpec

### 2.1 CTArgSpec

Each compile-time arg is described by a `CTArgSpec`:

```python
@dataclass
class CTArgSpec:
    name: str              # base arg name (before prefixing), e.g. "out_w"
    risc: str              # "ncrisc", "brisc", or "trisc"
    type: str = "uint32_t" # C++ type
    source: str = "user"   # how the value is derived
    source_key: str = ""   # key into the relevant lookup dict
    description: str = ""  # human-readable documentation
```

The `source` field determines where the value comes from:

| Source | Resolution | Example |
|--------|-----------|---------|
| `"cb"` | CB ID from CB assignments (`cb_by_node[node_id][source_key]`) | `source_key="in0"` resolves to the CB ID for port `in0` |
| `"sem"` | Semaphore L1 address from sem assignments (`sem_by_node[node_id][source_key]`) | `source_key="noc0_receiver_semaphore"` resolves to the global sem address |
| `"derived"` | Computed from user kwargs, grid context, or op configuration | `source_key="num_tiles"` resolves from `user_args["num_tiles"]` |
| `"user"` | Direct user-supplied value (historical alias for `"derived"`) | Same resolution as `"derived"` in practice |
| `"per_core"` | Per-core varying value, resolved at assembly time by `PerCoreCompileTimeDescriptor` | `source_key="sender_idx"` -- each core gets a different value |

### 2.2 CTArgSchema

A schema aggregates all specs for one op type:

```python
@dataclass
class CTArgSchema:
    op_name: str
    args: list[CTArgSpec] = field(default_factory=list)

    def args_for_risc(self, risc: str) -> list[CTArgSpec]:
        return [a for a in self.args if a.risc == risc]
```

Schemas are registered globally via `register_ct_schema()` -- typically called during op registration in `blaze/ops/<name>/op.py` alongside the `OpSpec` and `PhaseInfo` registrations (see [Chapter 2 -- Auto-Discovery](../ch02_blazeop_hierarchy/04_auto_discovery_and_registration.md)). For `MicroOp` subclasses, the CT arg schema is auto-derived from the C++ kernel header by the C++ parser (see [Chapter 2 -- C++ Parser](../ch02_blazeop_hierarchy/03_cpp_parser.md)). For `FusedOp` subclasses, the schema is the union of all child ops' schemas.

### 2.3 Schema Registry

```python
_CT_ARG_SCHEMAS: dict[str, CTArgSchema] = {}

def register_ct_schema(schema: CTArgSchema) -> None:
    _CT_ARG_SCHEMAS[schema.op_name] = schema

def get_ct_schema(op_name: str) -> Optional[CTArgSchema]:
    return _CT_ARG_SCHEMAS.get(op_name)

def list_ct_schemas() -> list[str]:
    return list(_CT_ARG_SCHEMAS.keys())
```

The SemEngine also uses `get_ct_schema()` to discover `source="sem"` entries -- this is the bridge between semaphore assignment and CT arg resolution.

---

## 3. Auto-Prefixing

### 3.1 The Prefix Mechanism

When a fused kernel contains multiple ops of the same type (e.g., two matmuls), their CT arg names would collide. The engine prevents this by prefixing each arg name with the op instance's logical prefix:

```python
def _compute_prefix(self, node, graph, merged_user=None):
    source = merged_user if merged_user is not None else node.kwargs
    prefix = source.get("ct_prefix", node.spec.name)
    if not prefix:
        return ""
    return f"{prefix}."
```

Examples:

| `ct_prefix` kwarg | Op name | Prefix | Resulting arg name |
|-------------------|---------|--------|-------------------|
| (not set) | `matmul` | `matmul.` | `matmul.out_w` |
| `"gate_matmul"` | `matmul` | `gate_matmul.` | `gate_matmul.out_w` |
| `""` (empty string) | `matmul` | `""` | `out_w` (no prefix) |

Setting `ct_prefix=""` is used for fused ops where arg names are already fully qualified in the kernel header.

The dot-separated prefix matches the C++ kernel's `get_named_compile_time_arg_val()` string format. On the kernel side, the generated `ct_args` namespace struct uses `__` to encode the dot: `ct_args::gate_matmul` contains a field `out_w`.

### 3.2 Branch Suffix Handling

For branched ops like `kn_sliced_matmul` with gate/up variants, the `ct_prefix` may include a branch suffix (e.g., `gu.gate`, `gu.up`). The kernel codegen strips the branch suffix to get the parent prefix (e.g., `gu`), but the CT arg engine uses the full prefix to ensure arg name uniqueness.

---

## 4. Shared-Prefix Deduplication

When multiple nodes explicitly set the same `ct_prefix` kwarg (e.g., gate and up matmul both using `"gu"`), the kernel reads a single set of CT args. The engine generates args from the **first** node encountered and reuses them for subsequent nodes with the same explicit prefix:

```python
shared_prefix_args: dict[str, list[CTArgValue]] = {}

for node in graph.topological_order():
    ...
    explicit_prefix = "ct_prefix" in node.kwargs or "ct_prefix" in (user_overrides.get(node.id) or {})

    if explicit_prefix and prefix in shared_prefix_args:
        result[node.id] = shared_prefix_args[prefix]
        continue

    args = [...]  # generate args
    if explicit_prefix:
        shared_prefix_args[prefix] = args
    result[node.id] = args
```

This is a Python-level identity optimization: nodes sharing a prefix reference the same `list` object. The collision validator leverages this by deduplicating on `id(args)`.

Important: implicit prefixes (defaulting to `spec.name`) are **not** shared. Only explicitly set `ct_prefix` values participate in deduplication. Collisions between implicit prefixes indicate a graph construction error (two instances of the same op without distinct prefixes) and are caught by validation.

---

## 5. Value Resolution

The `_resolve_value()` method dispatches based on the `source` field:

```python
def _resolve_value(self, spec, cb, sem, user) -> tuple[int, bool]:
    if spec.source == "cb":
        if spec.source_key in cb:
            return cb[spec.source_key], True
        return 0, False
    elif spec.source == "sem":
        if spec.source_key in sem:
            return sem[spec.source_key], True
        return 0, False
    elif spec.source == "per_core":
        if spec.source_key in user:
            return int(user[spec.source_key]), True
        return 0, False
    else:
        # "derived", "user", "grid"
        if spec.source_key in user:
            val = user[spec.source_key]
            if isinstance(val, float):
                return _float_to_bits(val), True
            return int(val), True
        return 0, False
```

Each resolution returns a `(value, was_resolved)` tuple. When a key is missing, the value defaults to `0` and `was_resolved` is `False`. The engine tracks unresolved args and either warns or raises depending on `strict` mode.

### 5.1 Float-to-Bits Conversion

For derived float values (e.g., scaling factors), `_float_to_bits()` converts a Python `float` to its IEEE 754 `uint32_t` bit pattern. This is how floating-point constants are passed as compile-time args to the kernel.

### 5.2 Per-Core Args

`source="per_core"` args have different values per core and are resolved at assembly time by the `PerCoreCompileTimeDescriptor` (see [Section 06 -- BlazeCompiler](./06_blaze_compiler.md)). During CT arg generation, they receive a placeholder value. The `generate_tuples()` method skips them entirely:

```python
if arg.source == "per_core":
    continue  # handled by PerCoreCompileTimeDescriptor
```

### 5.3 CTArgValue

The intermediate representation between `generate()` and `generate_tuples()`:

```python
@dataclass
class CTArgValue:
    name: str        # prefixed name (e.g. "gate_matmul.out_w")
    risc: str        # "ncrisc", "brisc", or "trisc"
    value: int       # resolved concrete value
    source: str      # "cb" | "sem" | "derived" | "per_core"
    resolved: bool = True  # False if value came from default (key missing)
```

The `resolved` field enables diagnostics. After generation, the engine checks for unresolved args (excluding `per_core`):

```python
unresolved = [a for args in result.values() for a in args
              if not a.resolved and a.source != "per_core"]
if unresolved:
    if strict:
        raise ValueError(msg)
    else:
        warnings.warn(msg)
```

The warning groups unresolved args by source type, e.g.: `CTArgEngine: 3 unresolved args (defaulted to 0). cb: ['gate_matmul.input'], derived: ['num_tiles']`.

---

## 6. Name Collision Detection

The `_validate_no_collisions()` method ensures no two args share the same prefixed name within a RISC:

```python
def _validate_no_collisions(self, all_args):
    per_risc = {"ncrisc": {}, "brisc": {}, "trisc": {}}
    seen_ids: set[int] = set()

    for node_id, args in all_args.items():
        args_id = id(args)
        if args_id in seen_ids:
            continue  # shared-prefix nodes reference the same list
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

The `seen_ids` set enables shared-prefix deduplication: if two nodes share the same args list object (because they share an explicit `ct_prefix`), the validator only checks it once. This prevents false collision reports for multi-branch ops.

If a collision is detected in `strict=True` mode, a `ValueError` is raised. In the `compile_engines()` orchestrator, this is caught and CT arg generation is skipped gracefully (see [Section 01](./01_blaze_graph_and_fusion_context.md)).

---

## 7. RISC Grouping

The `generate_tuples()` method converts per-node args into per-RISC `(name, value)` tuples:

```python
def generate_tuples(self, graph, cb_assignments=None, sem_assignments=None,
                    grid_context=None, user_overrides=None, strict=False):
    all_args = self.generate(graph, cb_assignments, sem_assignments,
                             grid_context, user_overrides, strict=strict)

    grouped = {"ncrisc": [], "brisc": [], "trisc": []}
    seen = {"ncrisc": set(), "brisc": set(), "trisc": set()}

    for node in graph.topological_order():
        for arg in all_args.get(node.id, []):
            if arg.source == "per_core":
                continue
            if arg.name not in seen[arg.risc]:
                grouped[arg.risc].append((arg.name, arg.value))
                seen[arg.risc].add(arg.name)

    return grouped
```

The output is what the `UnifiedKernelDescriptor` consumes directly:

```python
ukd = UnifiedKernelDescriptor(
    ncrisc_named_compile_time_args=ct_tuples["ncrisc"],
    brisc_named_compile_time_args=ct_tuples.get("brisc", []),
    trisc_named_compile_time_args=ct_tuples["trisc"],
    ...
)
```

The traversal in topological order with a `seen` set ensures:

1. **Deterministic ordering**: Args appear in the same order as their owning nodes in the topological sort.
2. **Deduplication**: Shared-prefix args are emitted only once (first encounter wins).
3. **Per-core exclusion**: `per_core` args are filtered out -- they're handled by `PerCoreCompileTimeDescriptor`.

---

## 8. Worked Example: Two-Node Pipeline

Consider a graph: `gather_1` feeding `matmul_1`.

**CB Engine** assigns:
- `ext_in_gather_1_input` -> cb_id=0
- `intermed_gather_1_out` -> cb_id=1
- `ext_in_matmul_1_in1` -> cb_id=2
- `ext_out_matmul_1_out` -> cb_id=3

**Sem Engine** assigns (gather with `dual_noc=True`):
- `gather_1 / noc0_receiver_semaphore` -> sem_id=0
- `gather_1 / noc1_receiver_semaphore` -> sem_id=1

**CT Arg Engine** resolves:
- For gather_1, arg with `source="cb", source_key="input"`: resolves to cb_id=0
- For gather_1, arg with `source="sem", source_key="noc0_receiver_semaphore"`: resolves to the L1 address for sem_id=0
- For matmul_1, arg with `source="cb", source_key="in0"`: resolves to cb_id=1 (from the edge)
- For matmul_1, arg with `source="derived", source_key="out_w"`: resolves from user_args

Each resolved arg is prefixed (`gather.input`, `matmul.in0_cb_id`, etc.) and grouped by RISC into the three output lists.

---

## 9. Integration with the Pipeline

The CTArgEngine is the third and final engine in the `compile_engines()` pipeline (see [Section 01](./01_blaze_graph_and_fusion_context.md) for the full orchestrator and data flow diagram). It consumes `cb_by_node` from CBEngine and `sem_by_node` from SemEngine, and produces per-RISC `(name, value)` tuples that feed into the `UnifiedKernelDescriptor`.

In the `BlazeCompiler._compile_for_device()` method, the CTArgEngine is called with `strict=True` since all CB and semaphore assignments are guaranteed to be present. User overrides are merged with NOC grid context (logical-to-physical core coordinate translations for mcast and gather ops) before being passed as `user_overrides`.

---

<- [03 -- Sem Engine](./03_sem_engine.md) | [Index](index.md) | [05 -- Kernel Codegen](./05_kernel_codegen.md) ->
