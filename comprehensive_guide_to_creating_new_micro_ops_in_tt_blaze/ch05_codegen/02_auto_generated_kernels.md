# 5.2 Auto-Generated Kernels: `kernel_codegen.py`

When a FusedProgram is built without an explicit `kernel` path, TT-Blaze auto-generates the complete C++ kernel from the shadow graph. The module `blaze/kernel_codegen.py` implements this generation. This section covers every stage of the algorithm, from the PhaseInfo registry through the final content-hashed output file.

---

## 5.2.1 PhaseInfo: the bridge between Python and C++

Every MicroOp that participates in kernel codegen registers a `PhaseInfo` dataclass. This is the metadata that tells the code generator how to emit C++ for that op type:

```python
@dataclass
class PhaseInfo:
    cpp_type: str                    # e.g. "blaze::Mcast::Op"
    alias_suffix: str = ""           # e.g. "Mcast", "Matmul"
    has_init_teardown: bool = False   # Op has init()/teardown() lifecycle
    setup_method: str | None = None  # e.g. "setup_src", "setup_weights"
    init_is_empty: bool = False      # init() body is whitespace/comments only
    teardown_is_empty: bool = False  # teardown() body is whitespace/comments only
```

The fields map directly to codegen decisions:

| Field | Codegen effect |
|-------|---------------|
| `cpp_type` | The fully-qualified C++ struct used in `using` declarations (e.g., `using ActMcast = blaze::Mcast::Op<ct_args::act_mcast>;`) |
| `alias_suffix` | The suffix appended to the humanized prefix to form the type alias name |
| `has_init_teardown` | Whether to emit `init()` / `teardown()` calls (Op-style lifecycle) |
| `setup_method` | Static setup method name for mcast re-targeting (e.g., `"setup_src"`) |
| `init_is_empty` | If `true`, the `init()` call is elided from the generated kernel |
| `teardown_is_empty` | If `true`, the `teardown()` call is elided from the generated kernel |

### Registration

PhaseInfo is registered through two paths:

1. **Automatic derivation from kernel header** (MicroOp): when a MicroOp subclass has no manually-declared `phase`, `_auto_derive_from_kernel_hpp()` parses the C++ header (`kernels/op.hpp`) and constructs PhaseInfo from the parsed struct metadata. The `init_is_empty` and `teardown_is_empty` flags are extracted by `cpp_parser.py` by inspecting the actual C++ source of the `init()` and `teardown()` methods.

2. **Manual declaration** (FusedOp or explicit MicroOp): the op class sets `phase = PhaseInfo(...)` directly as a class attribute. This is used by ops like `FlashMla`, `SdpaReduce`, `KvCacheUpdate`, and `DsaSparseFlashDecode`.

During `BlazeOp.register()`, if the `cpp_type` is empty (auto-derived case), it is filled in from `op_class`:

```python
if cls.phase is not None:
    phase = cls.phase
    if not phase.cpp_type and cls.op_class:
        class_name = cls.op_class.split("::")[-1]
        phase = PhaseInfo(
            cpp_type=f"blaze::{class_name}::Op",
            alias_suffix=phase.alias_suffix or class_name,
            has_init_teardown=phase.has_init_teardown,
            ...
        )
    register_phase_info(cls.name, phase)
```

The result is stored in the module-level `_PHASE_REGISTRY` dict, keyed by op name (e.g., `"mcast"`, `"matmul"`, `"gather"`). Only ops with a registered `PhaseInfo` participate in codegen. Ops without one (e.g., pure-Python utility ops) are silently skipped.

---

## 5.2.2 PhaseDecl: intermediate representation for codegen

Before emitting C++, the code generator constructs `PhaseDecl` objects -- one per unique `(prefix, op_type)` pair in the graph. PhaseDecl captures everything needed to emit a type alias and its execution block:

```python
@dataclass
class PhaseDecl:
    alias: str              # C++ type alias name, e.g. "ActMcast"
    op_type: str            # Python op name, e.g. "mcast"
    prefix: str             # CT arg prefix, e.g. "act_mcast"
    template_args: list[str]  # e.g. ["ct_args::act_mcast"]
    phase_info: PhaseInfo   # registered codegen metadata
```

### Prefix computation

The prefix is derived from the node's `ct_prefix` kwarg, with branch suffix stripping for ops like `KNSlicedMatmul` where gate/up branches share a parent prefix:

```python
def _compute_prefix(node: OpNode) -> str:
    prefix = node.kwargs.get("ct_prefix", node.spec.name)
    branch = node.kwargs.get("branch")
    if branch and prefix.endswith(f".{branch}"):
        prefix = prefix[: -len(f".{branch}")]
    return prefix
```

For example, if a KNSlicedMatmul has `ct_prefix="shared_expert__gu.gate"` and `branch="gate"`, the computed prefix is `"shared_expert__gu"` -- the parent scope that both gate and up branches share.

### Type alias generation

The `_to_alias()` function converts a prefix and op suffix into a C++ type alias name:

```python
def _to_alias(prefix: str, op_type: str, suffix: str) -> str:
    if prefix == op_type:
        return suffix  # "matmul" + "Matmul" -> "Matmul"
    parts = re.split(r"[._]", prefix)
    alias = "".join(p.capitalize() for p in parts)
    if suffix.lower() not in alias.lower():
        alias += suffix  # avoid "McastMcast"
    return alias
```

Examples:

| prefix | op_type | suffix | result |
|--------|---------|--------|--------|
| `"act_mcast"` | `"mcast"` | `"Mcast"` | `"ActMcast"` |
| `"gu"` | `"kn_sliced_matmul"` | `"Matmul"` | `"GuMatmul"` |
| `"matmul"` | `"matmul"` | `"Matmul"` | `"Matmul"` |
| `"mcast2"` | `"mcast"` | `"Mcast"` | `"Mcast2"` (suffix already present) |

### CT args type mapping

The `_to_ct_args_type()` function maps the logical prefix to the generated `ct_args::` struct name:

```python
def _to_ct_args_type(prefix: str) -> str:
    return prefix.replace(".", "__").replace("-", "_")
```

So a prefix like `"shared_expert.gu"` becomes `ct_args::shared_expert__gu`, matching the struct names in `named_args_generated.h`.

---

## 5.2.3 The generate_kernel() algorithm

The core of the codegen is `generate_kernel()`, which takes a `BlazeGraph` and returns a complete C++ source string. The algorithm has three stages:

### Stage 1: Group nodes by (prefix, op_type)

```python
def generate_kernel(graph: BlazeGraph) -> str:
    node_order = graph.nodes  # insertion order = emit() call order

    prefix_groups: dict[tuple[str, str], list[OpNode]] = defaultdict(list)
    for node in node_order:
        prefix = _compute_prefix(node)
        prefix_groups[(prefix, node.spec.name)].append(node)
```

Multiple nodes can share the same prefix (e.g., gate and up matmul branches both using `ct_prefix="gu"`). The code generator treats them as a single phase, emitting one type alias and one execution block.

### Stage 2: Build PhaseDecl objects

For each unique group, look up the registered PhaseInfo and construct a PhaseDecl:

```python
phases: list[PhaseDecl] = []
for (prefix, op_type), nodes in prefix_groups.items():
    info = get_phase_info(op_type)
    if info is None:
        continue
    template_args = [f"ct_args::{_to_ct_args_type(prefix)}"]
    alias = _to_alias(prefix, op_type, info.alias_suffix)
    phases.append(PhaseDecl(alias=alias, op_type=op_type, prefix=prefix,
                            template_args=template_args, phase_info=info))
```

### Stage 3: Group mcasts for persistent state sharing

All mcast phases are grouped into `McastGroup` objects. The first mcast is the **leader** (owns the persistent NOC command buffer state), and subsequent mcasts are **followers** (reuse the leader's state via `run_as<>()`):

```python
@dataclass
class McastGroup:
    leader: PhaseDecl
    followers: list[PhaseDecl] = field(default_factory=list)

mcast_phases = [p for p in phases if p.op_type == "mcast"]
mcast_groups: list[McastGroup] = []
if mcast_phases:
    leader = mcast_phases[0]
    followers = mcast_phases[1:]
    mcast_groups.append(McastGroup(leader=leader, followers=followers))
```

---

## 5.2.4 Kernel emission: _emit_kernel()

The `_emit_kernel()` function assembles the complete C++ file from the analyzed phases.

### File header and includes

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Auto-generated by blaze.kernel_codegen -- do not edit manually.

#include "blaze/kernels/ops.hpp"
#include "blaze/kernels/kernel_op_api.hpp"
#include "blaze/kernels/kernel_utils.hpp"
```

When `BLAZE_DEBUG_KERNELS` is set, a debug include is added (optionally guarded by RISC filters):

```cpp
#if defined(COMPILE_FOR_BRISC)
#include "api/debug/dprint.h"
#endif
```

### Phase type aliases

For each PhaseDecl, a `using` alias is emitted:

```cpp
using ActMcast = blaze::Mcast::Op<ct_args::act_mcast>;
using GuMatmul = blaze::KNSlicedMatmul::Op<ct_args::gu>;
using Matmul = blaze::Matmul::Op<ct_args::matmul>;
```

The template argument is always `ct_args::<sanitized_prefix>`, linking each phase to its named CT arg struct in the generated header.

### kernel_main() body

The function opens with compute kernel initialization:

```cpp
void kernel_main() {
#if defined(COMPILE_FOR_TRISC)
    deepseek_compute_kernel_init();
#endif
```

Then emits one block per phase in node insertion order (matching the `emit()` call sequence). Each block is wrapped in a `DeviceZoneScopedN` profiling zone named after the uppercased prefix.

### Non-mcast ops: scoped lifecycle

For non-mcast ops, the variable is scoped within the `DeviceZoneScopedN` block:

```cpp
    {
        DeviceZoneScopedN("MATMUL");
        Matmul matmul;
        matmul.init();      // emitted only if !init_is_empty
        matmul();
        matmul.teardown();  // emitted only if !teardown_is_empty
    }
```

The scoping ensures the Op struct's destructor runs at block exit, releasing any resources. When both `init_is_empty` and `teardown_is_empty` are `True`, the lifecycle calls are elided:

```cpp
    {
        DeviceZoneScopedN("COPY");
        Copy copy;
        copy();
    }
```

### Noop kernel support

When a node has `noop_kernel=True` in its kwargs (set by `_tag_noop_nodes()` during selective compilation), the phase is replaced with an empty Tracy scope:

```cpp
    {
        DeviceZoneScopedN("MATMUL_NOOP");
    }
```

This preserves the profiling trace while skipping all computation, useful for pipeline stages where one device has no work for that phase.

---

## 5.2.5 Mcast group optimization: leader/follower

Mcast phases receive special treatment because multiple mcast ops often share **persistent NOC command buffer state**. The codegen identifies a **leader** (the first mcast phase) and **followers** (subsequent mcasts).

### Leader emission

The leader is declared outside the profiling block so its lifetime persists across all followers. `init()` is called once:

```cpp
    ActMcast act_mcast;
    act_mcast.init();
    {
        DeviceZoneScopedN("ACT_MCAST");
        act_mcast();
    }
```

If there are no followers, `teardown()` is called immediately. If there are followers, teardown is deferred until after the last follower.

### Follower emission

Followers do not construct their own Op instance. Instead, they call `setup_src<>()` and `run_as<>()` on the leader's instance, using the follower's `ct_args::` struct as a template parameter:

```cpp
    act_mcast.template setup_src<ct_args::mcast2>();
    {
        DeviceZoneScopedN("MCAST2");
        act_mcast.template run_as<ct_args::mcast2>();
    }
```

Key behaviors:

- The `setup_src<>()` call re-targets the sender to a new source CB (if the follower consumes an external input). It is emitted only when the follower's mcast node is connected to an external input port (needs source buffer reconfiguration). The codegen checks this via `graph.external_input_ports`:

```python
is_external = any(nid == n.id for nid, _ in graph.external_input_ports)
if is_external:
    lines.append(f'    {mcast_leader_var}.template setup_src<ct_args::...>();')
```

- The `run_as<>()` call executes the multicast with the follower's CT arg parameters while reusing the leader's persistent NOC state.

### Deferred teardown

`teardown()` is emitted once after the **last follower** completes:

```cpp
    act_mcast.teardown();
```

If `teardown_is_empty` is `True`, it is elided entirely.

This optimization is essential for the shared expert kernel where activation mcast, mcast1 (down proj activation), and mcast2 (residual) share a sender core and NOC command buffers.

---

## 5.2.6 Lifecycle elision

A significant optimization is **lifecycle elision**: if the C++ parser determines that `init()` or `teardown()` have empty bodies (only whitespace or comments), the PhaseInfo records `init_is_empty=True` or `teardown_is_empty=True`. The code generator then skips those calls:

```python
emit_init = not pdecl.phase_info.init_is_empty
emit_teardown = not pdecl.phase_info.teardown_is_empty

if emit_init:
    lines.append(f"        {var_name}.init();")
lines.append(f"        {run_call}")
if emit_teardown:
    lines.append(f"        {var_name}.teardown();")
```

This matters because the JIT compiler cannot always optimize away empty virtual calls, and in a pipeline with many phases, the cumulative overhead is measurable. The Python-side analysis at registration time enables zero-cost elision without relying on compiler optimizations.

---

## 5.2.7 DPRINT debug injection

The `BLAZE_DEBUG_KERNELS` environment variable controls DPRINT instrumentation in generated kernels. The `_parse_debug_riscs()` function parses the variable:

| Value | Effect |
|-------|--------|
| `"0"` or unset | No debug output |
| `"1"` or `"all"` | DPRINT on all RISCs |
| `"brisc"` | DPRINT only on BRISC |
| `"ncrisc"` | DPRINT only on NCRISC |
| `"trisc"` | DPRINT on all compute sub-cores |
| `"trisc0"` | DPRINT only on unpack sub-core |
| `"trisc1"` | DPRINT only on math sub-core |
| `"trisc2"` | DPRINT only on pack sub-core |
| `"ncrisc,brisc"` | DPRINT on both data-movement processors |

When enabled, the generated kernel includes `#include "api/debug/dprint.h"` and emits phase start/end markers:

```cpp
#if defined(COMPILE_FOR_BRISC)
    DPRINT << "[PHASE] ACT_MCAST start" << ENDL();
#endif
    {
        DeviceZoneScopedN("ACT_MCAST");
        // ... phase body ...
    }
#if defined(COMPILE_FOR_BRISC)
    DPRINT << "[PHASE] ACT_MCAST done" << ENDL();
#endif
```

The RISC guard is a C preprocessor condition composed from the valid RISC names:

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

For multiple selected RISCs, the guards are combined with `||`:

```cpp
#if (defined(COMPILE_FOR_NCRISC)) || (defined(COMPILE_FOR_BRISC))
    DPRINT << "[PHASE] ... start" << ENDL();
#endif
```

> **Warning:** DPRINT output requires `TT_METAL_DPRINT_CORES` to be set in the environment as well. Setting only `BLAZE_DEBUG_KERNELS` injects the code but does not enable the runtime output channel.

---

## 5.2.8 Content-hashed output and atomic writes

`generate_kernel_file()` writes the generated C++ to disk with a content-hashed filename:

```python
def generate_kernel_file(graph, output_dir=None, name=None) -> str:
    source = generate_kernel(graph)
    content_hash = hashlib.sha256(source.encode()).hexdigest()[:12]
    label = name or "generated"
    out_path = out_dir / f"{label}_{content_hash}.cpp"

    if os.getenv("BLAZE_DEBUG_KERNELS", "0") != "0":
        out_path = out_dir / f"{label}_{content_hash}_debug.cpp"

    if not out_path.exists():
        fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".cpp.tmp")
        try:
            os.write(fd, source.encode())
            os.close(fd)
            os.rename(tmp, out_path)  # atomic on same filesystem
        except BaseException:
            os.close(fd)
            os.unlink(tmp)
            raise

    return str(out_path)
```

Properties of this approach:

1. **Idempotent:** The same graph always produces the same content hash, so the same file is generated. If the file already exists, generation is skipped entirely.
2. **Atomic writes:** The file is written to a temporary file first, then atomically renamed via `os.rename()`. This ensures concurrent processes never see a partial file.
3. **Debug variant:** When `BLAZE_DEBUG_KERNELS` is set, the output filename includes `_debug` to avoid cache collisions between debug and release builds.
4. **Default output location:** Files are written to `{BLAZE_ROOT}/../generated/kernels/` unless overridden.

### Triggering codegen

Codegen is triggered lazily at build time. In `FusedProgram.build()`:

```python
def build(self, noc_mode=None):
    self._prepare_for_build()
    if self.program._kernel is None:
        self.program.set_kernel_from_graph(self._shadow_graph, name=self._name)
    pd = self.program.build()
```

And in `BlazeProgram.set_kernel_from_graph()`:

```python
def set_kernel_from_graph(self, graph, name=None):
    from .kernel_codegen import generate_kernel_file
    self._kernel = generate_kernel_file(graph, name=name)
    self._generated = True
```

---

## 5.2.9 Complete generated kernel example

Given a shadow graph with nodes `[act_mcast (mcast), gu (kn_sliced_matmul), matmul (matmul)]`, the codegen produces:

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Auto-generated by blaze.kernel_codegen -- do not edit manually.

#include "blaze/kernels/ops.hpp"
#include "blaze/kernels/kernel_op_api.hpp"
#include "blaze/kernels/kernel_utils.hpp"

using ActMcast = blaze::Mcast::Op<ct_args::act_mcast>;
using GuMatmul = blaze::KNSlicedMatmul::Op<ct_args::gu>;
using Matmul = blaze::Matmul::Op<ct_args::matmul>;

void kernel_main() {
#if defined(COMPILE_FOR_TRISC)
    deepseek_compute_kernel_init();
#endif

    ActMcast act_mcast;
    act_mcast.init();
    {
        DeviceZoneScopedN("ACT_MCAST");
        act_mcast();
    }
    act_mcast.teardown();

    {
        DeviceZoneScopedN("GU");
        GuMatmul gu;
        gu.init();
        gu();
        gu.teardown();
    }

    {
        DeviceZoneScopedN("MATMUL");
        Matmul matmul;
        matmul.init();
        matmul();
        matmul.teardown();
    }

}
```

Key observations:
- `deepseek_compute_kernel_init()` is always emitted for TRISC.
- Phase order matches `emit()` call order.
- Mcast is declared outside the scope block for potential follower sharing.
- Each phase gets a Tracy profiling scope (`DeviceZoneScopedN`).
- Type aliases use `ct_args::` prefixed structs matching `named_args_generated.h`.
- Each processor (NCRISC, BRISC, TRISC) compiles this same source with different `COMPILE_FOR_*` defines. The `Op` struct's `operator()()` method uses `#if defined(...)` guards internally to select per-processor behavior.

### Inspecting generated kernels

To see what the codegen produces for a given FusedProgram, use the `generated_kernel` property:

```python
compiled = f.build()
source = f.program.generated_kernel  # Returns the .cpp source string or None
```

---

## Key takeaways

- **PhaseInfo** is the per-op metadata that maps Python ops to C++ Op structs. It captures the Op struct name, lifecycle properties, and setup methods.
- **PhaseDecl** groups graph nodes by `(prefix, op_type)`, collapsing shared-prefix branches into single phases.
- **`generate_kernel()`** walks nodes in insertion order (matching `emit()` call sequence) and emits one profiler-zoned block per phase.
- **Mcast groups** optimize multi-mcast fusions by sharing persistent sender state. The leader gets `init()`/`teardown()`; followers use `setup_src<>()`/`run_as<>()`.
- **Lifecycle elision** skips empty `init()` and `teardown()` calls based on `init_is_empty` / `teardown_is_empty` flags derived from the C++ parser at registration time.
- **Content-hashed filenames** provide automatic caching with atomic writes for concurrent safety.
- **`BLAZE_DEBUG_KERNELS`** provides fine-grained DPRINT injection with per-RISC and per-sub-core filtering, invaluable for diagnosing phase-ordering hangs.
