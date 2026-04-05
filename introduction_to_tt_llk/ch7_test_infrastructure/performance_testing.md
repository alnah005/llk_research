# Performance Testing

Performance tests measure kernel execution time in hardware clock cycles. They use the same two-file model as functional tests but replace `TestConfig` with `PerfConfig` and add C++ profiling macros to delineate timed regions. Separately, the infrastructure provides access to hardware performance counters for fine-grained pipeline analysis.

## Python Side: PerfConfig

### File Naming

Performance test Python files live in `tests/python_tests/` and follow the naming pattern `perf_<name>.py`.

### PerfConfig Object

`PerfConfig` (defined in `tests/python_tests/helpers/perf.py`) extends `TestConfig` and adds performance-specific functionality. Its constructor:

```python
PerfConfig(
    test_name: str,
    formats: FormatConfig = None,
    run_types: list[PerfRunType] = [],
    templates: list[TemplateParameter] = [],
    runtimes: list[RuntimeParameter] = [],
    variant_stimuli: StimuliConfig = None,
    unpack_to_dest=False,
    unpack_to_srcs=False,
    disable_format_inference=False,
    dest_acc=DestAccumulation.No,
    l1_acc=L1Accumulation.No,
    skip_build_header: bool = False,
    compile_time_formats: bool = False,
)
```

All arguments shared with `TestConfig` have the same behavior. The key addition is `run_types`.

### Run Types

The `run_types` argument takes a list of `PerfRunType` values (defined in `tests/python_tests/helpers/llk_params.py`). Each run type becomes a separate compiled variant by being appended to the `templates` list as a `PERF_RUN_TYPE` template parameter. The kernel's C++ code uses `PERF_RUN_TYPE` as a `constexpr` to conditionally enable or disable pipeline stages, allowing isolation of individual stages:

```python
run_types=[
    PerfRunType.L1_TO_L1,        # Full pipeline: unpack -> math -> pack
    PerfRunType.UNPACK_ISOLATE,  # Only unpack runs; math and pack return early
    PerfRunType.MATH_ISOLATE,    # Only math runs; unpack sets valid bits, pack returns
    PerfRunType.PACK_ISOLATE,    # Only pack runs; unpack and math return early
    PerfRunType.L1_CONGESTION,   # Measures L1 memory contention effects
]
```

### Fixtures

Performance test functions must include one additional fixture beyond `workers_tensix_coordinates`:

- `perf_report` -- A `PerfReport` object that accumulates timing data across variants and run types. It supports filtering, post-processing, and CSV export.

### Running a Performance Test

```python
configuration.run(perf_report, location=workers_tensix_coordinates)
```

Unlike functional tests (which return L1 data for assertion), performance tests write timing data into the `perf_report` fixture. No golden comparison is performed.

### Complete Python Example

```python
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16, DataFormat.Float16_b]),
    tile_count=16,
    math_fidelity=lambda formats: get_valid_math_fidelities(formats, PERF_RUN=True),
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
)
def test_perf_eltwise_binary(
    perf_report, formats, tile_count, math_fidelity, dest_acc, workers_tensix_coordinates
):
    configuration = PerfConfig(
        "sources/eltwise_binary_fpu_perf.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.UNPACK_ISOLATE,
                   PerfRunType.MATH_ISOLATE, PerfRunType.PACK_ISOLATE],
        templates=[MATH_FIDELITY(math_fidelity)],
        runtimes=[TILE_COUNT(tile_count)],
        variant_stimuli=StimuliConfig(
            None, formats.input_format,
            None, formats.input_format,
            formats.output_format,
            tile_count_A=tile_count, tile_count_B=tile_count, tile_count_res=tile_count,
        ),
        dest_acc=dest_acc,
    )
    configuration.run(perf_report, location=workers_tensix_coordinates)
```

## C++ Side: Profiling Macros

Performance test C++ files (named `*_perf.cpp` in `tests/sources/`) include the profiling headers:

```cpp
#include "perf.h"
#include "profiler.h"
```

### ZONE_SCOPED and PROFILER_SYNC

Two macros control the profiling instrumentation:

- **`ZONE_SCOPED("name")`** -- Opens a named timing zone. The name appears in the profiler output as a marker (e.g., `"INIT"`, `"TILE_LOOP"`). Zones are scoped to the enclosing `{}` block -- the zone ends when the block exits.
- **`PROFILER_SYNC()`** -- A synchronization barrier that ensures all three TRISC threads reach the same point before any thread proceeds. This is critical for accurate isolation measurements: without it, one thread might start its next zone while another is still in the current zone.

### Typical C++ Structure

```cpp
#ifdef LLK_TRISC_UNPACK
#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params) {
    {
        ZONE_SCOPED("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(...);
        _llk_unpack_AB_init_<>(DEFAULT_TENSOR_SHAPE);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE) {
            return;  // Skip unpack when isolating pack
        } else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE) {
            _perf_unpack_loop_set_valid<true, true>(TILE_CNT * TILE_NUM_FACES);
            return;  // Set valid bits for math without actual unpacking
        } else {
            for (std::uint32_t tile = 0; tile < TILE_CNT; tile++) {
                _llk_unpack_AB_<>(PERF_ADDRESS(PERF_INPUT_A, tile),
                                  PERF_ADDRESS(PERF_INPUT_B, tile));
            }
        }
        PROFILER_SYNC();
    }
}
#endif
```

Each `run_kernel` across all three TRISC sections follows the same pattern: an `"INIT"` zone for hardware configuration, and a `"TILE_LOOP"` zone for the main computation loop. The `PERF_RUN_TYPE` constexpr (set by `PerfConfig` for each run type) controls which stages are active.

### Profiler Data Model

The profiler system produces a `ProfilerData` object (from `tests/python_tests/helpers/profiler.py`) backed by a Pandas DataFrame. It provides two views:

- **Raw event view** (`ProfilerData.raw()`) -- Contains `TIMESTAMP`, `ZONE_START`, and `ZONE_END` entries. Data from each thread is concatenated (all UNPACK, then MATH, then PACK). Each `ZONE_START` is immediately followed by its `ZONE_END`.
- **Profiler view** (`ProfilerData.frame()`) -- Contains `TIMESTAMP` and `ZONE` entries ordered by timestamp across threads. `ZONE` entries include a `duration` column (end - start).

`ProfilerData` supports fluent filtering: `.unpack()`, `.math()`, `.pack()` filter by thread; `.zones()`, `.timestamps()` filter by entry type; `.marker("TILE_LOOP")` filters by zone name.

## Hardware Performance Counters

Beyond profiler-based zone timing, the infrastructure provides access to Tensix hardware performance counters for cycle-accurate pipeline analysis. These counters measure events like FPU instruction cycles, unpacker busy cycles, packer stalls, and L1 arbitration events.

### Counter Banks

Tensix cores contain five hardware performance counter banks:

| Bank | Description |
|------|-------------|
| `INSTRN_THREAD` | Instruction issue counts and stall reasons per thread (61 counters) |
| `FPU` | FPU and SFPU operation valid signals (3 counters) |
| `TDMA_UNPACK` | Unpacker busy signals and math pipeline status (11 counters) |
| `L1` | NoC ring transactions and L1 arbitration events (16 counters via mux) |
| `TDMA_PACK` | Packer busy signals and destination register access (3 counters) |

Each bank has control registers for mode selection, counter ID selection, and start/stop, plus two output registers: `OUT_L` (cycle count) and `OUT_H` (event count).

### Counter Modes

| Mode | Behavior |
|------|----------|
| 0 (Continuous) | Counter runs until explicitly stopped. Maintains reference cycle count in `OUT_L`. Use this for most cases. |
| 1 (Reference period) | Counter automatically stops after a specified number of cycles. |
| 2 (Continuous, no ref) | Like mode 0, but does not track elapsed cycles in `OUT_L`. |

### Shared Buffer Architecture

All TRISC threads share a single counter configuration and data buffer in L1, reducing memory usage. The synchronization protocol:

1. All threads call `llk_perf::start_perf_counters()`. The first thread to arrive initializes hardware.
2. All threads call `llk_perf::stop_perf_counters()`. The last thread to arrive reads hardware counters and writes results to the shared data buffer.

No mutex is needed -- each thread atomically sets its own bit in the sync control word.

### C++ Counter API

Include `counters.h` and call the start/stop functions in each TRISC section:

```cpp
#include "counters.h"

// In each of LLK_TRISC_UNPACK, LLK_TRISC_MATH, LLK_TRISC_PACK:
void run_kernel(RUNTIME_PARAMETERS params) {
    llk_perf::start_perf_counters();
    // ... kernel work ...
    llk_perf::stop_perf_counters();
}
```

On Quasar (4 TRISCs), the SFPU thread must also call start/stop.

### Python Counter API

The Python side (in `tests/python_tests/helpers/counters.py`) provides:

| Function | Description |
|----------|-------------|
| `configure_counters(location)` | Write counter configuration to L1. Configures all 94 counter definitions. Clears the data buffer and sync control word. |
| `read_counters(location)` | Read counter results. Returns a DataFrame with columns: `starter_thread`, `stopper_thread`, `bank`, `counter_name`, `counter_id`, `cycles`, `count`, `l1_mux`. Validates sync state. |
| `print_counters(results)` | Print counter results in human-readable format. |
| `export_counters(results, filename, test_params, worker_id)` | Export to CSV in `perf_data/` directory. |

`read_counters()` performs validation and reports specific errors, for example:

```
RuntimeError: Perf counters were not stopped by all threads.
Missing stop_perf_counters() call from: MATH. sync_ctrl=0x0000006c
```

### Derived Metrics

The metrics module (`tests/python_tests/helpers/metrics.py`) computes efficiency ratios from raw counter data. All metrics are ratios from 0.0 to 1.0 where higher is better:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Unpacker Write Efficiency | `SRCA_WRITE / UNPACK0_BUSY_THREAD0` (and SRCB equivalent) | Fraction of busy cycles spent writing data. Low values indicate L1 contention or data dependency stalls. |
| Packer Efficiency | `PACKER_DEST_READ_AVAILABLE / PACKER_BUSY` | Fraction of busy cycles with valid data available. Low values indicate math stage bottleneck. |
| FPU Execution Efficiency | `FPU_INSTRUCTION / FPU_INSTRN_AVAILABLE_1` | Fraction of available cycles where FPU actually executes. Distinguishes compute-bound from memory-bound workloads. |
| Math Pipeline Utilization (experimental) | `MATH_INSTRN_STARTED / MATH_INSTRN_AVAILABLE` | Measures instruction flow through the pipeline. |
| Math-to-Pack Handoff (experimental) | `AVAILABLE_MATH / PACKER_BUSY` | Pipeline balance between math and packer stages. |
| Unpacker-to-Math Data Flow (experimental) | `SRCA_WRITE_AVAILABLE / UNPACK0_BUSY_THREAD0` | Detects math backpressure causing unpacker stalls. |

Usage:

```python
from helpers.counters import configure_counters, read_counters
from helpers.metrics import compute_metrics, print_metrics

configure_counters(location="0,0")
# ... run kernel ...
results = read_counters(location="0,0")
metrics = compute_metrics(results)
print_metrics(metrics)
```

## Performance Report Generation

The `PerfReport` class (from `tests/python_tests/helpers/perf.py`) accumulates timing data across all variants and run types. It supports:

- **Filtering** -- `report.filter("column", value)` or `report.marker("TILE_LOOP")` for zone-specific data.
- **Post-processing** -- `report.post_process()` normalizes tile loop durations by loop factor and tile count, yielding per-tile cycle counts.
- **CSV export** -- `report.dump_csv("filename.csv")` writes to the `perf_data/` directory under the build artifacts path.
- **Scatter plots** -- The `dump_scatter()` function generates interactive HTML scatter plots using Plotly, showing cycles/tile across parameter sweeps.

The performance data DataFrame includes columns for run type, marker name, mean and standard deviation of zone durations, and sweep parameter values. Per-worker CSV files are written with naming patterns like `perf_my_test.gw0.csv` (worker 0), enabling post-hoc aggregation and comparison.

---

**Next:** [Chapter 8 -- Kernel Development Patterns](../ch8_kernel_development_patterns/index.md)
