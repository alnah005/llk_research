# Appendix A.1 -- Interactive Dataflow Visualizer

The Blaze visualizer is a standalone toolchain that renders compiled fusion graphs as interactive HTML pages. It exposes every detail of the compiler's output -- node topology, circular buffer assignments, semaphore wiring, core grid mappings, RISC schedules, and compile-time argument resolution -- in a single browser tab.

## A.1.1 Entry Point: `blaze.visualize()`

**File**: `blaze/visualizer.py`

```python
def visualize(
    graph: BlazeGraph,
    engine_result: EngineResult | None = None,
    grid_config: GridConfig | None = None,
    *,
    name: str = "",
    tensors: dict | None = None,
    output_dir: str | pathlib.Path | None = None,
    open_browser: bool = True,
) -> pathlib.Path:
```

The function accepts a `BlazeGraph` produced by a `blaze.fuse()` context. If no `EngineResult` is supplied, it calls `compile_engines(graph)` automatically. The typical call site:

```python
with blaze.fuse() as ctx:
    out = blaze.matmul({"input": ext_a, "weights": ext_b})
blaze.visualize(ctx.graph, name="my_pipeline")
```

Internally, `visualize()` performs three steps:

1. Calls `export_viz_data()` to produce a JSON-serializable Python dict.
2. Locates the HTML template (first from `blaze/visualizer.html`, then `visualizer/index.html`).
3. Injects the JSON data as a `<script>` block before the `</body>` tag using `applyBlazeData(data, false)`, writes the self-contained HTML to disk, and optionally opens the browser.

The injection uses `json.dumps(data, separators=(",", ":"))` with `</` escaped to `<\/` for safe embedding inside `<script>` tags. The output path is either a caller-specified directory or a `tempfile.mkdtemp(prefix="blaze_viz_")` directory.

## A.1.2 Data Export: `viz_export.py`

**File**: `blaze/viz_export.py` (1176 lines)

This module converts compiled Blaze IR into the `BLAZE_DATA` JSON dict consumed by the visualizer. Two public functions are exposed:

```python
def export_viz_data(
    graph: BlazeGraph,
    result: EngineResult | None = None,
    grid_config: GridConfig | None = None,
    name: str = "",
    tensors: dict | None = None,
    rt_args: dict | None = None,
) -> dict:

def export_viz_json(
    graph: BlazeGraph, ..., path: str = "blaze_viz.json",
) -> None:
```

`export_viz_data()` is the core function. It runs the full graph analysis pipeline:

| Build step | Internal function | Output key |
|---|---|---|
| Op and tensor nodes | `_build_nodes()` | `nodes` |
| Dataflow edges with CB/sem info | `_build_edges()` | `edges` |
| DAG layout coordinates | `_build_node_positions()` | `nodePositions` |
| Topological execution phases | `_build_phases()` | `phases` |
| Gantt-style timeline bars | `_build_timeline()` | `timeline`, `totalTime` |
| Semaphore signal/wait events | `_build_sem_events()` | `semEvents` |
| Spatial chip grid | `_build_core_grid()` | `coreGrid` |
| Per-core L1 layout + RISC schedule | `_build_core_layouts()` | `coreLayouts` |
| Per-op CT arg schema + resolved values | `_build_op_args()` | `opArgs` |
| Full CB assignment details | `_build_cb_details()` | `cbDetails` |
| Full semaphore assignment details | `_build_sem_details()` | `semDetails` |
| Summary statistics | `_build_stats()` | `stats` |
| JIT cache header introspection | `_collect_jit_headers()` | `jitHeaders` |

### Op Color Palette

Every op type maps to a fixed hex color used throughout the visualizer:

```python
_OP_COLORS = {
    "mcast": "#f85149",        # red
    "matmul": "#2ea043",       # green
    "kn_sliced_matmul": "#2ea043",
    "gather": "#a371f7",       # purple
    "moe_gather": "#a371f7",
    "reduce": "#d29922",       # gold
    "gated_reduce": "#d29922",
    "gated_local_reduce": "#d29922",
    "residual_add": "#58a6ff", # blue
    "add": "#58a6ff",
    "ccl": "#db61a2",          # pink
}
_DEFAULT_COLOR = "#8b949e"     # gray
```

### Node Construction

`_build_nodes()` iterates the topological order and creates a node dict per op with fields: `id`, `type` (display category like `MATMUL`), `label`, `risc` (the RISC processors involved -- `TRISC` for compute ops, `BRISC/NCRISC` for data movement), `cores` (count), `opType`, `color`, and `topoIndex`.

External tensors are added as separate `INPUT` and `OUTPUT` nodes. For each, `_extract_tensor_metadata()` attempts three sources in priority order: the `ExternalTensor.metadata` dict, introspection of the `ExternalTensor.tensor` object, and lookup in the caller-supplied `tensors` dict. Introspected metadata includes `shape`, `dtype`, `layout`, `bufferType`, `memoryLayout`, `shardShape`, and `shardOrientation`.

### CB Detail Export

`_build_cb_details()` produces a list of every CB assignment with full metadata:

- `category`: one of `ext_in`, `intermed`, `ext_out`, `internal`
- `isTensorBacked`: whether the CB's address comes from a backing tensor buffer
- `addrOffset` / `addrEnd`: relative L1 offsets for allocator-managed CBs (null for tensor-backed)
- `firstUse`, `lastUse`, `lifetime`: temporal liveness interval
- `producer` / `consumer`: node IDs on each end

### Per-Core Layout and RISC Schedule

`_build_core_layouts()` iterates every core in the grid and builds a deduplicated layout map. Each unique layout contains:

- `cbs`: list of CB allocations on that core (id, name, size, pages, pageSize, format, tileShape)
- `sems`: semaphores active on that core (id, type hw/global, protocol, initial value)
- `risc`: per-RISC-lane Gantt bars (`BRISC`, `NCRISC`, `TRISC`) with idle gaps filled

Layouts are deduplicated by a fingerprint of their CB ID sets (plus a sender-core flag), so cores sharing identical configurations point to the same layout object. The `layoutMap` maps `"col,row"` strings to layout keys.

### CT Arg Resolution

`_build_op_args()` uses the `CTArgEngine` to resolve compile-time arguments for each op. The output for each node includes:

- `kernelPath`: the C++ kernel header path
- `schema`: the named CT arg interface (name, risc, type, source, C++ struct)
- `resolved`: per-RISC lists of resolved values with `baseName`, `value`, `source`, `cppStruct`, and `resolved` flag

The function also parses C++ kernel headers via `_parse_struct_names()` to map each CT arg back to its originating struct (`ReaderCTArgs`, `WriterCTArgs`, `ComputeCTArgs`, `CoreCTArgs`).

## A.1.3 Visualizer Frontend

**File**: `visualizer/index.html` (354 KB, single-page application)

The frontend is a self-contained HTML file with embedded CSS and JavaScript. It uses D3.js for the DAG rendering and provides multiple coordinated views.

### Themes

Two CSS themes are defined via CSS custom properties on `:root` and `[data-theme="light"]`:

| Property | Dark (default) | Light |
|---|---|---|
| `--bg` | `#0d1117` | `#f6f8fa` |
| `--panel-bg` | `#161b22` | `#ffffff` |
| `--text` | `#c9d1d9` | `#24292f` |

Op-specific colors (`--color-mcast`, `--color-matmul`, `--color-gather`, `--color-reduce`, `--color-add`) remain constant across themes.

### Views and Panels

The visualizer provides the following synchronized views:

1. **DAG View**: D3-rendered directed graph with nodes positioned by `nodePositions`. Edges display CB labels and semaphore markers. Clicking a node selects it.
2. **Core Grid Overlay**: A `cols x rows` spatial grid showing per-phase core assignments. Each cell is colored by the op running on it (from `phaseMap`). DRAM workers, phantom cores, and the sender core are visually distinguished.
3. **Timeline / Gantt View**: Horizontal bars per op showing estimated execution phases. Semaphore events (signal/wait) are overlaid.
4. **CB Inspection Panel**: On node selection, displays all CBs touching that node -- size, page count, data format, tile shape, address offset, producer/consumer chain.
5. **Semaphore Inspection Panel**: Lists all semaphores with protocol, initial value, connected nodes, and hw/global type.
6. **Op Args Panel**: Shows the CT arg schema and resolved values for the selected op, grouped by RISC, with C++ struct context.
7. **Per-Core L1 Layout Panel**: Clicking a core cell shows its CB stack, semaphores, and per-RISC schedule (BRISC/NCRISC/TRISC Gantt bars).

### Data Loading

The frontend accepts data via two mechanisms:
- **Injected**: `applyBlazeData(jsonObj, false)` called from an inline `<script>` (produced by `visualize()`).
- **URL parameter**: `?data=<url>` fetches an external JSON file, enabling a hosted viewer to load different pipeline exports.

A demo dataset is shipped as `visualizer/demo_data.json` (248 KB).

## A.1.4 JSON Schema

**File**: `visualizer/schema.json`

The formal JSON Schema (draft 2020-12) defines the `BLAZE_DATA` format. Required top-level keys:

```
nodes, edges, nodePositions, phases, timeline, totalTime, coreGrid
```

Optional keys: `metadata`, `semEvents`, `coreLayouts`, `opArgs`, `cbDetails`, `semDetails`, `generatedKernelPath`, `jitCacheDir`, `jitHeaders`, `stats`, `rtArgs`.

Key structural details from the schema:

- `cbDetails[].category` is an enum: `ext_in | intermed | ext_out | internal`
- `semDetails[].type` is an enum: `hw | global`
- `edges[].sem.type` is an enum: `hw | global`
- `nodes[].tensorMeta` is nullable and includes shard-level metadata
- `coreLayouts.layouts` is an object of deduplicated layout classes, referenced by `coreLayouts.layoutMap`

## A.1.5 Auto-Export via `BLAZE_EXPORT`

**File**: `blaze/compiler.py`, method `BlazeCompiler._auto_export()`

Setting `BLAZE_EXPORT=1` causes `BlazeCompiler.compile()` to automatically dump visualizer JSON after every compilation:

```python
# In BlazeCompiler.compile():
if os.environ.get("BLAZE_EXPORT"):
    self._auto_export(graph, program, tensors)
```

The auto-export behavior is controlled by two env vars:

| Env var | Default | Description |
|---|---|---|
| `BLAZE_EXPORT` | unset | Enable auto-export (any truthy value) |
| `BLAZE_EXPORT_PATH` | `generated/viz` | Output directory for JSON files |

The exported file is named `{op_name}_viz.json`. After writing, the compiler prints:
```
[BLAZE_EXPORT] generated/viz/matmul_viz.json
[BLAZE_EXPORT] https://super-robot-mv52q6p.pages.github.io/?data=...
```

The second line provides a direct URL to the hosted visualizer with the data file pre-loaded.

## A.1.6 Pipeline Topology Embedding

When the `pipeline_builder` has written a `pipeline_config.json` to the same export directory, the auto-export merges it into the visualization data under the `pipeline` key:

```python
pipeline_path = os.path.join(export_dir, "pipeline_config.json")
if os.path.exists(pipeline_path):
    with open(pipeline_path, encoding="utf-8") as f:
        data["pipeline"] = json.load(f)
```

The pipeline config is produced by `pipeline_builder/visualize.py::serialize_layout()`. The `PipelineLayout` dataclass and its per-chip/per-stage record structure are documented in [Ch. 8.01 -- Pipeline Builder](../ch08_multi_host_pipeline/01_pipeline_builder.md). The merged data allows the visualizer to render multi-stage pipeline topologies alongside the per-stage fusion graph.

## A.1.7 JIT Cache Header Introspection

`_collect_jit_headers()` scans `~/.cache/tt-metal-cache/` for `named_args_generated.h` files produced by tt-metal's JIT compiler. It groups them by RISC (brisc, ncrisc, trisc), deduplicates by content hash, and extracts the `static constexpr` fields into `{name: value}` dicts. This data is surfaced in the visualizer under `jitHeaders`, providing a cross-reference between Blaze's CT arg names and the values that actually reached the hardware compiler.
