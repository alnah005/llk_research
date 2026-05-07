# 02 -- Repository Map

## Top-Level Directory Structure

```
tt-blaze/
    blaze/                  # Core Python package -- ops, graph IR, engines, codegen
    pipeline_builder/       # Multi-stage pipeline graph partitioning and visualization
    pipeline_manager/       # C++ inference runtime with mesh management and CCL
    disaggregation/
        migration/          # C++ library + pybind11 module for device migration
    visualizer/             # Interactive HTML visualization for Blaze graphs
    tests/
        blaze/              # Unit tests (CPU) + silicon integration tests
        pipeline_builder/   # Pipeline partitioning tests
    docs/                   # Developer guides (MicroOp, FusedOp authoring)
    tt-metal/               # Git submodule -- pinned tt-metal commit on main
    install.sh              # One-shot setup: submodule init + tt-metal build + venv
    build_blaze.sh          # Incremental build: tt-metal and/or migration library
    env.sh                  # Environment activation (venv + PYTHONPATH + TT_METAL_HOME)
    pyproject.toml          # Package metadata (blaze-core, Python >= 3.10)
    conftest.py             # Shared pytest fixtures
```

### `pipeline_builder/`

Orchestrates multi-stage inference pipelines. Contains `graph.py` for pipeline graph construction, `submesh_partition.py` for device mesh partitioning, and `visualize.py` for pipeline visualization. The companion `pipeline_manager/` holds the C++ side -- a standalone CMake project with its own source tree, native kernels, and tools for runtime scheduling of pipeline stages.

### `disaggregation/migration/`

C++ library for tensor migration between devices in multi-host disaggregated inference, built via CMake. Includes pybind11 bindings (`_migration` module) and its own test suite. The `build_blaze.sh` script handles building this component, including resolving pybind11's cmake directory.

### `visualizer/`

Ships a self-contained interactive visualizer: `index.html` with `schema.json` for graph schema and `demo_data.json` for example data. The Python side (`blaze.visualize()` and `blaze.export()`) generates JSON that the HTML frontend renders.

### `tests/`

Two top-level test directories:

- `tests/blaze/` -- unit tests (no device), silicon integration tests, micro-op tests, fused-op tests, and generality tests. Subdirectories include `micro-ops/`, `fused_ops/`, `generality/`, `backed/`, `migration/`, and model-specific tests like `glm5_1/`.
- `tests/pipeline_builder/` -- tests for pipeline stage orchestration and model pipelines.

### `docs/`

Developer guides:

- `WRITING_A_MICRO_OP.md` -- how to write a C++ kernel header + Python `emit()` + tests
- `WRITING_A_FUSED_OP.md` -- how to compose MicroOps into fused pipelines
- `GQA_ATTENTION_MODULE.md`, `SDPA_CONTINUATION.md` -- module-specific documentation

## The `blaze/` Package Layout

The `blaze/` directory is the heart of the project. It contains the graph IR, the engine stack, the op library, kernel headers, code generation, and model-level compositions. The Python modules break into four groups:

### Core Modules (Graph IR, Compilation, Program Builders)

| Module | Purpose |
|---|---|
| `__init__.py` | Public API surface. Auto-discovers ops, creates `_OpHandle` proxies, exports all public symbols. |
| `graph.py` | Core data structures: `BlazeGraph`, `OpNode`, `Edge`, `OpSpec`, `TensorPort`, `InternalCB`, `EngineResult`. Houses `compile_engines()` which runs CB, Sem, and CTArg engines on a graph. |
| `context.py` | `FusionContext` and `fuse()` context manager for the graph API. Captures op calls, builds a validated `BlazeGraph`. |
| `blaze_op.py` | `BlazeOp` base class, `MicroOp` and `FusedOp` subclasses. Port descriptors (`Input`, `Output`, `Internal`), CT arg declarations (`CompileTimeArg`, `CB`, `Sem`, `Grid`, `Derived`, `Param`, `PerCore`), and `FusedOpConfig` registration. |
| `registry.py` | Generic op registry: `register_op()`, `get_op_spec()`, `list_ops()`. |
| `fused_program.py` | `FusedProgram` -- the composition API context. Wraps `BlazeProgram` with device context, grid info, CB allocation helpers, CT/RT arg methods, role flags, multi-output support, and shadow-graph recording. |
| `program.py` | `BlazeProgram` -- the low-level program builder. Handles CB descriptor construction, semaphore allocation, per-RISC CT/RT arg lists, `UnifiedKernelDescriptor` assembly, and `ttnn.generic_op()` dispatch. |
| `compiler.py` | `BlazeCompiler` -- assembles `MeshProgramDescriptor` from engine outputs, handles per-device tensor splitting for mesh execution. |
| `kernel_codegen.py` | Generates C++ kernel source from a `BlazeGraph`. Uses `PhaseInfo` metadata registered by each op. |
| `cpp_parser.py` | Parses C++ kernel headers to auto-derive CT arg schemas and phase metadata. |
| `unified_kernel_descriptor.py` | Bridges between Blaze's per-RISC arg model and TT-Metal's `UnifiedKernelDescriptor` API. |

### Engines and Resource Allocation

| Module | Purpose |
|---|---|
| `cb_engine.py` | `CBEngine` -- assigns CB IDs to graph edges and op ports. |
| `cb_handle.py` | `CBHandle` dataclass -- typed reference to a circular buffer. See [Section 03](./03_two_api_design.md) for the full field list and usage patterns. |
| `cb_reconfig.py` | `CircularBufferIdManager` -- manages CB ID reuse across multi-phase programs. CB IDs with matching `(data_format, tile)` can share an ID across phases. |
| `sem_engine.py` | `SemEngine` -- assigns semaphore IDs to inter-core synchronization points. |
| `ct_args.py` | `CTArgEngine` -- generates per-RISC compile-time arg tuples from graph structure, CB assignments, and semaphore assignments. |
| `role_engine.py` | `GridConfig` -- device-specific grid configuration (columns, rows, DRAM worker positions, phantom columns). Provides core classification: matmul cores, sender core, A/B branch grids, gate-MM phantom cores. |
| `device_context.py` | `DeviceContext` -- bundles device handle, `GridConfig`, and `full_device_grid` CoreRangeSet. Factory method `from_device()` queries real hardware. |

### Infrastructure and Utilities

| Module | Purpose |
|---|---|
| `barrier.py` | `Barrier` class -- cross-core synchronization primitive using global semaphores and sender/receiver ops. |
| `ccl.py` | `setup_fabric()` -- shared utilities for collective communication ops (fabric connections, teardown semaphores, RT args). |
| `l1_profile.py` | L1 memory profiling: `print_cb_stats()` and `print_kernel_stats()` for debugging CB address layouts and kernel assignments. |
| `weight_provider.py` | Weight loading abstraction: `WeightProvider`, `SyntheticWeightProvider`, `StateDictWeightProvider`, `BlitzCacheWeightProvider`. |
| `visualizer.py` | `visualize()` -- generates interactive HTML visualization of a `BlazeGraph`. |
| `viz_export.py` | `export_viz_json()` -- exports graph structure to JSON for the standalone visualizer. |
| `utils.py` | Shared utilities (dtype byte sizes, element size helpers). |
| `injected_fabric_barrier_builder.py` | Automatically injects fabric barrier ops for CCL teardown. |

### Sub-Directories (Ops, Kernels, Models, Stages)

```
  ops/                    # Op library (~110 op packages)
  kernels/                # Shared C++ headers, SFPU code, vendored LLK + Metal headers
  models/deepseek_v3_b1/  # DeepSeek V3 model pipeline composition
  stages/                 # Pipeline stage definitions (deepseek_decoder, stage_io)
```

**Higher-level convenience modules** also reside in `blaze/`:

- `_gated_mlp.py` -- parameterized graph builder for gated MLP patterns
- `_gqa_attention.py` -- parameterized graph builder for grouped-query attention
- `_deepseek_grids.py` -- grid configuration presets for DeepSeek V3 models

### Ops Library

The `blaze/ops/` directory contains over 100 registered ops, each in its own sub-package with an `op.py` file and a `kernels/` directory holding the C++ header. Representative op categories include:

- **Compute:** `matmul`, `kn_sliced_matmul`, `dram_streaming_matmul`, `eltwise_mul`, `swiglu`, `clamped_silu`
- **Data movement:** `mcast`, `gather`, `scatter`, `copy`, `shard2cb`, `embed_mcast`
- **Normalization:** `rmsnorm`, `broadcast_rmsnorm`, `padded_rmsnorm`
- **Attention:** `sdpa`, `flash_mla`, `rope`, `q_branch`, `kv_branch`, `post_sdpa`, `pre_sdpa`, `create_q_heads`
- **MoE:** `moe`, `moe_router`, `routed_expert`, `large_moe`, `deepseek_moe_gate`, `glm_moe`
- **Reduction:** `gated_reduce`, `gated_local_reduce`, `reduce_to_one`, `gather_reduce`
- **CCL/Barrier:** `all_reduce`, `ccl_broadcast`, `barrier_sender`, `barrier_receiver`
- **Infrastructure:** `cb_reconfig`, `cb_flush`, `cb_scratch_reset`, `dm_risc_handshake`, `tile_copy`, `retilize`, `untilize`
- **Fused compositions:** `shared_expert`, `dense_mlp`, `dense_swiglu`, `down_proj`, `dsa_attention`, `dsa_pipeline`

### Kernel Headers

The `blaze/kernels/` directory contains shared C++ infrastructure:

| Path | Content |
|---|---|
| `ops.hpp` | Aggregator include -- pulls in every op's C++ struct so generated kernels need only one `#include`. |
| `kernel_op_api.hpp` | RISC detection (`is_ncrisc`, `is_brisc`, `is_trisc`), `SelectByRISCV` template alias, and op lifecycle API. |
| `kernel_utils.hpp` | Shared C++ utilities for kernel code. |
| `ct_types.h` | Compile-time type definitions (`CB`, `Semaphore`, `PerCore`, `Flag`) shared between Python codegen and C++ kernels. |
| `rt_str.hpp` | Runtime string utilities for debug output. |
| `kernel_includes/tt_llk/` | Vendored LLK headers for Blackhole and Wormhole B0 architectures, plus custom LLK math/SFPU extensions (e.g., `ckernel_sfpu_sdpa_reduce_row.h`, `ckernel_sfpu_topk_xl.h`, `llk_math_rmsnorm_bcast_scalar_dest_reuse.h`). Both `tt_llk_blackhole/` and `tt_llk_wormhole_b0/` variants are present; Blackhole is far more extensive. |
| `kernel_includes/tt_metal/` | Vendored TT-Metal hardware headers, NOC utilities, and third-party dependencies. Includes high-level compute APIs (`sdpa.h`, `rmsnorm.h`, `custom_mm.h`) that wrap lower-level LLK calls. |
| `sfpu/` | Custom SFPU extensions: `clamped_silu_sfpu.hpp` (clamped SiLU for DeepSeek-style gated MLPs), `zero_pad_sfpu.hpp` (tile boundary alignment). |

**`ops.hpp`** is the aggregator include, so a generated fused kernel can instantiate any combination of ops with a single `#include`:

```cpp
#include "ops.hpp"
blaze::Mcast::Op<ct_args::act_mcast> mcast;
blaze::Matmul::Op<ct_args::matmul> matmul;
```

**`kernel_op_api.hpp`** demonstrates the C++20 pattern central to Blaze's "one kernel, all RISCs" compilation model. It uses `inline constexpr bool` variables (`is_ncrisc`, `is_brisc`, `is_trisc`) conditioned on preprocessor defines, enabling `if constexpr (is_ncrisc) { ... }` in kernel code. The `SelectByRISCV` template alias selects a type based on which RISC processor the code is compiled for:

```cpp
namespace unified_kernels {

template <typename Reader, typename Writer, typename Compute>
using SelectByRISCV = std::conditional_t<
    is_ncrisc, Reader,
    std::conditional_t<is_brisc, Writer, Compute>>;

} // namespace unified_kernels
```

**`ct_types.h`** defines the typed aliases used by named CT args, bridging the Python codegen layer (which emits CT arg tuples with these types) and the C++ kernel layer (which reads them as typed struct fields):

```cpp
using CB = uint32_t;        // circular buffer ID (port name)
using Semaphore = uint32_t; // global semaphore L1 address
using PerCore = uint32_t;   // per-core varying value
using Flag = bool;          // boolean flag (is_active, pop_src, etc.)
```

## Build and Environment

### `install.sh`

One-shot bootstrap. Initializes the `tt-metal` submodule, builds TT-Metal with `RelWithDebInfo`, and creates the Python virtual environment:

```bash
git submodule update --init --recursive
cd tt-metal && ./build_metal.sh --build-type RelWithDebInfo
./create_venv.sh
```

### `build_blaze.sh`

Incremental build script with target selection:

- `--migration-only` (default) -- builds only the C++ migration library under `disaggregation/migration/`.
- `--with-metal` -- builds TT-Metal first, then migration.
- `--metal-only` -- builds only TT-Metal.
- `--all` -- builds everything.

The migration build uses CMake, links against the TT-Metal build artifacts, and optionally compiles a pybind11 extension module (`_migration`) for Python access. Supports build type selection (`--release`, `--development`, `--debug`).

### `env.sh`

Activates the TT-Metal Python venv and sets the required environment variables:

```bash
source tt-metal/python_env/bin/activate
export TT_METAL_HOME="$SCRIPT_DIR/tt-metal"
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/tt-metal:...:$PYTHONPATH"
```

### `pyproject.toml`

The package is named `blaze-core` (version 0.1.0), requires Python >= 3.10, and has zero runtime dependencies beyond what TT-Metal provides. The only optional dependency is `pytest>=7.0` for development.

## Key Infrastructure Modules

### DeviceContext (`device_context.py`)

Bundles the raw device handle, its `GridConfig`, and a pre-computed `full_device_grid` CoreRangeSet. The classmethod `DeviceContext.from_device(device)` queries the real device for grid size and DRAM worker positions, then constructs the appropriate `GridConfig`. The `worker_core_from_logical_core()` method translates logical coordinates to physical NOC coordinates. This encapsulation means compilation code never queries the device directly -- it operates on the context.

### GridConfig / RoleEngine (`role_engine.py`)

`GridConfig` is a frozen dataclass that encodes the device's grid geometry: `grid_cols`, `grid_rows`, and `dram_worker_positions`. For an unharvested Blackhole, the default is 13 columns by 10 rows with 8 DRAM workers at specific positions. It supports harvested Blackhole devices with variable grid sizes (e.g., $11 \times 10$, $12 \times 10$, $13 \times 10$).

It provides core classification methods:

- `get_matmul_cores()` -- excludes DRAM workers, phantom columns, and the sender core
- `get_compute_cores()` -- excludes DRAM workers and phantom columns
- `get_gate_mm_cores(n)` -- allocates phantom column cores for MoE gate weight placement (up to 576 experts)
- `build_ab_grids()` -- splits the grid into balanced A (gate) / B (up) grids for KN-sliced matmul
- `build_matmul_core_grid()` / `build_mcast_receiver_grid()` -- construct `CoreRangeSet` objects for dispatch

### BlazeProgram (`program.py`)

The low-level program builder for a single unified kernel. Handles CB allocation, semaphore management, named CT/RT args for all three RISCs, per-core CT args, validation against kernel headers, and final `ProgramDescriptor` assembly and dispatch. See [Section 03](./03_two_api_design.md) "Key `FusedProgram` Methods" for the full method listing (since `FusedProgram` wraps `BlazeProgram`).

### CBReconfig (`cb_reconfig.py`)

Manages circular buffer ID allocation across multiple phases of a multi-phase program. The `CircularBufferIdManager` reuses CB IDs when `(data_format, tile)` matches across phases, and its `Context` class ensures uniqueness within a single phase. This is central to the temporal CB reuse optimization that lets multiple phases share the 64 available CB ID slots.

### L1Profile (`l1_profile.py`)

Diagnostic tool activated by `BLAZE_L1_PROFILE=1`. Reports CB type (tensor-backed, aliased, scratch), L1 addresses, data formats, page sizes, and tile descriptors. Output goes to both stdout and a timestamped log file.

### Barrier (`barrier.py`)

Multi-core synchronization primitive. Composes `BarrierSender` and `BarrierReceiver` ops with global semaphores. Tracks signaller count and receiver wait values, supporting multi-signal patterns where multiple groups of cores signal before receivers proceed.

### CCL (`ccl.py`)

Shared utilities for collective communication ops. `setup_fabric()` takes a `FusedProgram`, a worker core, destination fabric node IDs, and configures teardown and buffer-index semaphores, computes fabric connection runtime args, and adds the necessary kernel defines. This consolidates the fabric setup that individual CCL ops (`all_reduce`, `ccl_broadcast`, `scatter`, etc.) all need.

## The Auto-Discovery Mechanism

At import time, `blaze/__init__.py` orchestrates automatic op registration: `register_all()` walks every sub-package under `blaze/ops/`, discovers `BlazeOp` subclasses, calls `.register()` on each, then wraps each class in an `_OpHandle` proxy that exposes both `blaze.<op>(...)` (graph API) and `blaze.<op>.emit(f, ...)` (composition API) on a single module attribute. Adding a new op requires only creating a new directory under `blaze/ops/` with an `op.py` file -- no manual wiring in `__init__.py` is needed. For the full discovery loop, registry internals, and `_OpHandle` implementation, see [Ch 2 Section 04 -- Auto-Discovery and Registration](../ch02_blazeop_hierarchy/04_auto_discovery_and_registration.md).

---

**Next:** [`03_two_api_design.md`](./03_two_api_design.md)
