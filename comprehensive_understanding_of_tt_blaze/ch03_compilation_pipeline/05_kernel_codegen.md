# 05 -- Kernel Codegen

The kernel codegen module generates complete C++ kernel source files from a `BlazeGraph`. It reads per-op `PhaseInfo` metadata, emits type aliases, assembles `kernel_main()` with profiling scopes, handles mcast persistent-state sharing, supports noop kernel optimization and debug DPRINT, and uses content-hash file caching for build efficiency.

**Source file:** `blaze/kernel_codegen.py`

---

## 1. Design Philosophy

The Python graph **is** the kernel spec. Traditional approaches require the developer to maintain a C++ kernel file and a Python description of that kernel in sync -- ensuring the prefix/field naming convention, the phase ordering, and the lifecycle calls all match. Blaze's codegen eliminates this synchronization burden: the generated kernel is derived entirely from:

1. The `BlazeGraph` node sequence (which ops, in what order).
2. The `PhaseInfo` metadata registered by each op (C++ type names, lifecycle flags).
3. The CT arg prefix structure (mapping Python op instances to C++ `ct_args::` struct namespaces).

The developer writes only the C++ `Op` struct and the Python `BlazeOp` subclass; the kernel file that wires them together is generated automatically. This also means the generated kernel depends on graph structure but not on specific CB IDs or semaphore addresses -- those are resolved at compile time via the CT arg mechanism, not baked into the kernel source.

---

## 2. PhaseInfo and PhaseDecl

### 2.1 PhaseInfo -- Per-Op Codegen Metadata

Each op type registers a `PhaseInfo` alongside its `OpSpec` and `CTArgSchema`:

```python
@dataclass
class PhaseInfo:
    cpp_type: str              # C++ Op struct, e.g. "blaze::Mcast::Op"
    alias_suffix: str = ""     # Short name for aliases, e.g. "Mcast", "Matmul"
    has_init_teardown: bool = False
    setup_method: str | None = None     # e.g. "setup_src", "setup_weights"
    init_is_empty: bool = False         # True if init() body is trivial
    teardown_is_empty: bool = False     # True if teardown() body is trivial
```

| Field | Purpose |
|-------|---------|
| `cpp_type` | The fully qualified C++ Op template, e.g. `"blaze::Mcast::Op"` |
| `alias_suffix` | Used to generate type aliases (e.g., `"Mcast"` produces `using ActMcast = blaze::Mcast::Op<ct_args::act_mcast>;`) |
| `has_init_teardown` | Whether the Op struct has `init()`/`teardown()` lifecycle methods |
| `setup_method` | Static method name for reconfiguring data sources (used by mcast followers to switch input CBs) |
| `init_is_empty` | If `True`, `init()` call is omitted (empty-body optimization) |
| `teardown_is_empty` | If `True`, `teardown()` call is omitted |

The empty-body flags (`init_is_empty`, `teardown_is_empty`) are detected by the C++ parser during op registration (see [Chapter 2 -- C++ Parser](../ch02_blazeop_hierarchy/03_cpp_parser.md)). When `init()` or `teardown()` contains only whitespace or comments, the parser marks the corresponding flag as `True`, and the codegen omits the call -- eliminating unnecessary function call overhead.

### 2.2 PhaseInfo Registry

```python
_PHASE_REGISTRY: dict[str, PhaseInfo] = {}

def register_phase_info(op_name: str, info: PhaseInfo) -> None:
    _PHASE_REGISTRY[op_name] = info

def get_phase_info(op_name: str) -> PhaseInfo | None:
    return _PHASE_REGISTRY.get(op_name)
```

### 2.3 PhaseDecl -- Per-Instance Codegen Plan

During codegen, each unique `(prefix, op_type)` pair produces a `PhaseDecl`:

```python
@dataclass
class PhaseDecl:
    alias: str                # C++ type alias name, e.g. "ActMcast"
    op_type: str              # e.g. "mcast"
    prefix: str               # e.g. "act_mcast"
    template_args: list[str]  # e.g. ["ct_args::act_mcast"]
    phase_info: PhaseInfo
```

---

## 3. Type Alias Generation

### 3.1 `_to_alias()` -- Naming Convention

The alias maps a logical prefix to a CamelCase C++ type name:

```python
def _to_alias(prefix, op_type, suffix):
    if prefix == op_type:
        return suffix  # "matmul" -> "Matmul"
    parts = re.split(r"[._]", prefix)
    alias = "".join(p.capitalize() for p in parts)
    if suffix.lower() not in alias.lower():
        alias += suffix
    return alias
```

Examples:

| prefix | op_type | suffix | alias |
|--------|---------|--------|-------|
| `"act_mcast"` | `mcast` | `"Mcast"` | `"ActMcast"` |
| `"gu"` | `kn_sliced_matmul` | `"Matmul"` | `"GuMatmul"` |
| `"matmul"` | `matmul` | `"Matmul"` | `"Matmul"` |
| `"mcast2"` | `mcast` | `"Mcast"` | `"Mcast2"` (suffix already present) |

### 3.2 `_to_ct_args_type()` -- CT Args Struct Name

Maps the logical prefix to the generated `ct_args::` struct namespace:

```python
def _to_ct_args_type(prefix):
    return prefix.replace(".", "__").replace("-", "_")
```

For example, `"shared_expert.gu"` becomes `"shared_expert__gu"`, yielding `ct_args::shared_expert__gu`.

### 3.3 Template Arg Construction

Each `PhaseDecl` gets a template argument list pointing to its CT args struct:

```python
template_args = [f"ct_args::{_to_ct_args_type(prefix)}"]
```

This connects the generated C++ type alias to the named compile-time arg struct that the `UnifiedKernelDescriptor` populates.

---

## 4. `generate_kernel()`: The Core Algorithm

```python
def generate_kernel(graph: BlazeGraph) -> str:
```

The function proceeds through four stages:

### Stage 1: Group Nodes by (prefix, op_type)

```python
node_order = graph.nodes  # insertion order, NOT topological
prefix_groups: dict[tuple[str, str], list[OpNode]] = defaultdict(list)
for node in node_order:
    prefix = _compute_prefix(node)
    prefix_groups[(prefix, node.spec.name)].append(node)
```

Nodes with the same `(prefix, op_type)` are grouped into a single `PhaseDecl`. Using insertion order (not topological) is intentional -- the user's emit sequence defines the correct execution sequence for overlapped data movement and compute phases.

### Stage 2: Build PhaseDecl List

For each `(prefix, op_type)` group, look up the `PhaseInfo` and construct a `PhaseDecl`:

```python
for (prefix, op_type), nodes in prefix_groups.items():
    info = get_phase_info(op_type)
    if info is None:
        continue
    template_args = [f"ct_args::{_to_ct_args_type(prefix)}"]
    alias = _to_alias(prefix, op_type, info.alias_suffix)
    phases.append(PhaseDecl(alias=alias, ...))
```

### Stage 3: Group Mcasts

All mcast phases are collected. The first becomes the leader; the rest are followers:

```python
mcast_phases = [p for p in phases if p.op_type == "mcast"]
if mcast_phases:
    leader = mcast_phases[0]
    followers = mcast_phases[1:]
    mcast_groups.append(McastGroup(leader=leader, followers=followers))
```

### Stage 4: Emit C++

The `_emit_kernel()` function assembles the final source from the analyzed components.

---

## 5. Generated Kernel Structure

A generated kernel follows this structure:

```cpp
// Auto-generated by blaze.kernel_codegen -- do not edit manually.
#include "blaze/kernels/ops.hpp"
#include "blaze/kernels/kernel_op_api.hpp"
#include "blaze/kernels/kernel_utils.hpp"

// Type aliases
using ActMcast = blaze::Mcast::Op<ct_args::act_mcast>;
using Matmul = blaze::Matmul::Op<ct_args::matmul>;
using Gelu = blaze::Gelu::Op<ct_args::gelu>;

void kernel_main() {
#if defined(COMPILE_FOR_TRISC)
    deepseek_compute_kernel_init();
#endif

    // Phase: ACT_MCAST
    {
        DeviceZoneScopedN("ACT_MCAST");
        ActMcast act_mcast;
        act_mcast.init();
        act_mcast();
        act_mcast.teardown();
    }

    // Phase: MATMUL
    {
        DeviceZoneScopedN("MATMUL");
        Matmul matmul;
        matmul();
    }

    // Phase: GELU
    {
        DeviceZoneScopedN("GELU");
        Gelu gelu;
        gelu();
    }
}
```

The `deepseek_compute_kernel_init()` call initializes the compute engine for TRISC cores, gated by `COMPILE_FOR_TRISC` so that NCRISC and BRISC compilations skip it. Each `Op<ct_args::prefix>` struct is a template specialization that reads its compile-time args from the generated `ct_args` namespace.

---

## 6. Mcast Persistent State Sharing

Mcast ops that share a persistent NOC command buffer (via the same `sender` kwarg) require special handling. The kernel codegen groups them into `McastGroup` structures:

```python
@dataclass
class McastGroup:
    leader: PhaseDecl
    followers: list[PhaseDecl] = field(default_factory=list)
```

The **leader** is the first mcast in the group. Its variable is declared at function scope (not block scope) so followers can reference it:

```cpp
// Leader: declared at function scope, init() called once
ActMcast act_mcast;
act_mcast.init();
{
    DeviceZoneScopedN("ACT_MCAST");
    act_mcast();
}

// Follower: setup_src + run_as on the leader's object
act_mcast.template setup_src<ct_args::wt_mcast>();
{
    DeviceZoneScopedN("WT_MCAST");
    act_mcast.template run_as<ct_args::wt_mcast>();
}

// Last follower: teardown the shared state
act_mcast.teardown();
```

Key mechanics:

1. `init()` is called only once on the leader.
2. Followers call `setup_src<FollowerCTArgs>()` to reconfigure the source CB, then `run_as<FollowerCTArgs>()` to execute with the follower's CT args.
3. `teardown()` is called only after the last follower completes.

The `setup_src` call is conditional: it's only emitted when the follower has an external input port (checked via `graph.external_input_ports`). This means the kernel needs to reconfigure the source CB pointer for the new tensor. If the follower reads from an intermediate (edge), no reconfiguration is needed.

---

## 7. Empty-Body Optimization

The `PhaseInfo.init_is_empty` and `teardown_is_empty` flags enable the codegen to skip trivial lifecycle calls:

```python
emit_init = not pdecl.phase_info.init_is_empty
emit_teardown = not pdecl.phase_info.teardown_is_empty
```

These flags are set by the C++ parser during op registration. When `init()` or `teardown()` contains only whitespace or comments, the corresponding call is omitted from the generated kernel. This reduces generated code size and avoids empty function call overhead.

---

## 8. Noop Kernels

When a node has `kwargs["noop_kernel"] = True` (set by `BlazeCompiler._tag_noop_nodes()` for ops that should be disabled at dispatch time), the codegen emits an empty profiling scope:

```cpp
{
    DeviceZoneScopedN("PHASE_NAME_NOOP");
}
```

This is useful for conditionally disabling parts of a fused pipeline (e.g., disabling attention phases during prefill or for A/B testing) without regenerating the kernel. The profiling marker still appears in traces, making it clear that the phase was intentionally skipped.

---

## 9. Debug DPRINT Support

When the environment variable `BLAZE_DEBUG_KERNELS` is set, the codegen inserts DPRINT statements around each phase:

```cpp
#if defined(COMPILE_FOR_NCRISC)
    DPRINT << "[PHASE] ACT_MCAST start" << ENDL();
#endif
    // ... phase body ...
#if defined(COMPILE_FOR_NCRISC)
    DPRINT << "[PHASE] ACT_MCAST done" << ENDL();
#endif
```

The variable supports fine-grained RISC filtering:

| Value | Behavior |
|-------|----------|
| `"0"` or unset | No debug output |
| `"1"` or `"all"` | DPRINT on all RISCs |
| `"ncrisc"` | DPRINT only on NCRISC |
| `"brisc,trisc"` | DPRINT on BRISC and TRISC |
| `"trisc0"` | DPRINT only on TRISC unpack lane (thread 0) |

The `_parse_debug_riscs()` function converts these to preprocessor guards:

```python
_RISC_GUARDS = {
    "brisc": "defined(COMPILE_FOR_BRISC)",
    "ncrisc": "defined(COMPILE_FOR_NCRISC)",
    "trisc": "defined(COMPILE_FOR_TRISC)",
    "trisc0": "defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0",
    "trisc1": "defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 1",
    "trisc2": "defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 2",
}
```

Multiple RISCs are combined with `||`. When debug is enabled, the generated file name includes a `_debug` suffix to keep debug and release kernels in separate cache slots.

---

## 10. Content-Hash File Caching

`generate_kernel_file()` writes the generated source to disk with content-hash naming:

```python
def generate_kernel_file(graph, output_dir=None, name=None) -> str:
    source = generate_kernel(graph)
    content_hash = hashlib.sha256(source.encode()).hexdigest()[:12]
    label = name or "generated"
    out_path = out_dir / f"{label}_{content_hash}.cpp"
```

The file path includes a 12-character SHA-256 prefix of the content. This means:

1. **Same graph produces same file**: If the graph hasn't changed, the hash matches and the file write is skipped (the function still returns the path).
2. **Different graphs produce different files**: Hash collisions are astronomically unlikely with 48 bits.
3. **Atomic writes**: The file is written to a `.tmp` file first, then atomically renamed via `os.rename()`, so concurrent processes never see a partial file.

Default output directory is `generated/kernels/` relative to the blaze package root.

---

## 11. Prefix Handling for Branched Ops

The `_compute_prefix()` helper in `kernel_codegen.py` handles branched ops:

```python
def _compute_prefix(node):
    prefix = node.kwargs.get("ct_prefix", node.spec.name)
    branch = node.kwargs.get("branch")
    if branch and prefix.endswith(f".{branch}"):
        prefix = prefix[:-len(f".{branch}")]
    return prefix
```

If a node has `ct_prefix="gu.gate"` and `branch="gate"`, the `.gate` suffix is stripped, yielding the parent prefix `"gu"`. This ensures the Op struct uses the shared parent namespace (`ct_args::gu`) while individual CT args retain their branch-specific names for uniqueness.

---

## 12. Codegen Data Flow Summary

```
BlazeGraph.nodes (insertion order)
    |
    v
Group by (prefix, op_type)
    |
    v
For each group:
    PhaseInfo lookup -> PhaseDecl
    |
    v
Identify McastGroups (leader + followers)
    |
    v
_emit_kernel():
    Includes (ops.hpp, kernel_op_api.hpp, ...)
    Type aliases (using ActMcast = ...)
    kernel_main():
        TRISC init
        For each node (insertion order):
            if noop -> empty scope
            if mcast leader -> init + run + (teardown if no followers)
            if mcast follower -> setup_src + run_as + (teardown if last)
            else -> scoped { init(); op(); teardown(); }
        Optional DPRINT markers
    |
    v
Content-hash file write -> "label_abc123def456.cpp"
```

---

<- [04 -- CT Arg Engine](./04_ct_arg_engine.md) | [Index](index.md) | [06 -- BlazeCompiler](./06_blaze_compiler.md) ->
