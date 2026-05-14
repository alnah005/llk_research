# 01 -- Grid Configuration and Core Categories

Source: `blaze/role_engine.py`, `blaze/device_context.py`, `blaze/fused_program.py`

## The problem: variable grids

A Blackhole P300 exposes a 13x10 compute grid. A P150 exposes fewer columns (typically 11x10). Harvesting (manufacturing defects that disable silicon) further reduces the available grid. If your op hardcodes `grid_cols = 13`, it will produce incorrect CT args, allocate CBs on nonexistent cores, or simply crash on any device that does not match.

TT-Blaze centralizes all grid-dependent decisions in the `GridConfig` dataclass (defined in `blaze/role_engine.py`) and the `DeviceContext` wrapper (defined in `blaze/device_context.py`). Op authors interact with pre-computed grid fields on `FusedProgram` and should never query the device object directly.

---

## The GridConfig dataclass

`GridConfig` is a frozen (immutable) dataclass with three stored fields:

```python
@dataclass(frozen=True)
class GridConfig:
    grid_cols: int
    grid_rows: int
    dram_worker_positions: frozenset[tuple[int, int]]
```

All remaining properties are derived. Because `GridConfig` is frozen, it is safe to share across threads and to use as a dict key.

The three fields are:

| Field | Meaning |
|---|---|
| `grid_cols` | Number of usable columns in the compute grid. 13 on an unharvested P300, 11 on a P150, potentially 12 on a partially-harvested part. |
| `grid_rows` | Number of usable rows. Currently 10 on all Blackhole SKUs. |
| `dram_worker_positions` | Frozenset of `(col, row)` tuples identifying cores reserved by the hardware for DRAM bank access. These cores must be excluded from compute and matmul grids. |

Coordinates throughout TT-Blaze follow the convention `(col, row)`, which maps to `ttnn.CoreCoord(x=col, y=row)`.

### P300 vs P150 grids

| SKU | Typical Grid | Total Cores | Notes |
|-----|-------------|-------------|-------|
| P300 (unharvested) | 13x10 | 130 | Full grid |
| P300 (harvested) | 12x10 | 120 | One column harvested |
| P150 | 11x10 | 110 | Two columns harvested |

The `grid_cols` field captures this directly. All downstream calculations -- matmul core count, phantom column positions, sender core location -- derive from `grid_cols` and `grid_rows`, so they automatically adapt to any SKU.

### Construction

There are two construction paths:

**`GridConfig.from_device(device)`** -- the production path. Queries the device for its actual compute-with-storage grid size and DRAM worker positions:

```python
@classmethod
def from_device(cls, device) -> "GridConfig":
    import ttnn
    grid_size = device.compute_with_storage_grid_size()
    dram_workers = device.get_optimal_dram_bank_to_logical_worker_assignment(
        ttnn.NOC.NOC_0
    )
    return cls(
        grid_cols=grid_size.x,
        grid_rows=grid_size.y,
        dram_worker_positions=frozenset((c.x, c.y) for c in dram_workers),
    )
```

Two device APIs are used:
- `compute_with_storage_grid_size()` returns a `CoreCoord` whose `.x` and `.y` give the grid dimensions.
- `get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)` returns a list of `CoreCoord` objects identifying the DRAM worker cores.

This is the only way to get correct DRAM worker positions. Different Blackhole SKUs and harvesting patterns produce different positions.

**`GridConfig.default()`** -- a fixed 13x10 configuration for an unharvested Blackhole P300, with eight DRAM worker positions at known coordinates:

```python
@classmethod
def default(cls) -> "GridConfig":
    return cls(
        grid_cols=13,
        grid_rows=10,
        dram_worker_positions=frozenset([
            (0, 0), (0, 3), (0, 7), (0, 9),
            (7, 1), (7, 4), (7, 6), (7, 9),
        ]),
    )
```

The default layout has 8 DRAM worker cores, split between columns 0 and 7. Use `default()` for unit tests and offline code generation. Use `from_device()` for all production compilation paths.

---

## Core categories

Blackhole cores serve different roles. `GridConfig` partitions the grid into named categories with dedicated accessors. Understanding these categories is essential for placing ops correctly.

### Sender core

The sender core is always the bottom-right corner of the grid:

```python
@property
def sender_core(self) -> tuple[int, int]:
    return (self.grid_cols - 1, self.grid_rows - 1)
```

On a 13x10 grid, this is `(12, 9)`. On an 11x10 grid it shifts to `(10, 9)`. The sender core is responsible for multicast operations: it reads data from a CB and pushes it over the NOC to all receiver cores. It is excluded from matmul core sets. Ops that need a single core for multicast or gather operations should always use this property.

The module-level constant `SENDER_CORE = GridConfig.default().sender_core` (i.e. `(12, 9)`) is exported for backward compatibility, but new code should use `self.grid.sender_core` or `f.sender`.

### DRAM worker cores

DRAM workers are cores positioned next to DRAM banks. They service DRAM read/write traffic. Their positions are device-specific and stored in `dram_worker_positions`. On the default 13x10 grid, there are eight DRAM workers located in columns 0 and 7.

```python
def get_dram_worker_cores(self) -> list[tuple[int, int]]:
    return sorted(self.dram_worker_positions)
```

DRAM workers are excluded from compute core sets because they have reduced L1 availability (their L1 is partly consumed by DRAM bank buffers).

### Phantom (gate) cores

Phantom cores occupy the last column of the grid, rows 0 through `grid_rows - 2` (all rows except the sender row). They are used for gate-matrix-multiply weight placement in MoE (Mixture-of-Experts) models, where placing gate weights on a separate column prevents L1 pressure on the main A/B branch matmul cores.

```python
@property
def phantom_positions(self) -> frozenset[tuple[int, int]]:
    last_col = self.grid_cols - 1
    return frozenset((last_col, row) for row in range(self.grid_rows - 1))
```

The `get_gate_mm_cores(num_cores)` method returns the exact set of phantom cores needed for a given number of gate cores (where `num_cores = num_experts // tile_width`). It fills the last column first, then overflows to the second-to-last column for models needing more than `grid_rows - 1` gate cores:

```python
def get_gate_mm_cores(self, num_cores: int) -> list[tuple[int, int]]:
    rows_per_col = self.grid_rows - 1
    ncols = self.num_phantom_cols(num_cores)
    max_phantom = ncols * rows_per_col
    if num_cores > max_phantom:
        raise ValueError(...)
    coords = []
    for col_offset in range(ncols):
        col = self.grid_cols - 1 - col_offset
        for row in range(rows_per_col):
            if len(coords) >= num_cores:
                return coords
            coords.append((col, row))
    return coords
```

The `num_phantom_cols()` method decides whether 1 or 2 columns are needed:

```python
def num_phantom_cols(self, num_gate_cores: int = 0) -> int:
    rows_per_col = self.grid_rows - 1
    if num_gate_cores > rows_per_col:
        return 2
    return 1
```

Key details:
- On a 10-row grid, `rows_per_col = 9`. For up to 9 gate cores (corresponding to up to 256 experts at typical tile widths), one phantom column suffices.
- For more gate cores (e.g. DeepSeek V3 with >256 experts), a second phantom column is allocated.
- On a 13x10 grid, the maximum is `2 * 9 = 18` phantom cores across columns 12 and 11.

### All cores

```python
def get_all_cores(self) -> list[tuple[int, int]]:
    return [(col, row) for row in range(self.grid_rows)
                       for col in range(self.grid_cols)]
```

Returns every `(col, row)` coordinate in the grid, in row-major order. On a 13x10 grid this returns 130 coordinates.

### Compute cores

All cores that are neither DRAM workers nor phantoms:

```python
def get_compute_cores(self) -> list[tuple[int, int]]:
    excluded = self.dram_worker_positions | self.phantom_positions
    return [c for c in self.get_all_cores() if c not in excluded]
```

This is the broadest set of cores available for general computation. Note that compute cores include the sender core.

### Matmul cores

Compute cores minus the sender core. This is the set of cores available for matrix-multiply workloads:

```python
def get_matmul_cores(self) -> list[tuple[int, int]]:
    excluded = self.dram_worker_positions | self.phantom_positions | {self.sender_core}
    return [c for c in self.get_all_cores() if c not in excluded]
```

On a default 13x10 grid: `130 - 8 (DRAM) - 9 (phantom) - 1 (sender) = 112` matmul cores.

### Derived properties

| Property | Formula |
|---|---|
| `total_cores` | `grid_cols * grid_rows` |
| `matmul_cores` (count) | `total_cores - len(dram_worker_positions) - len(phantom_positions) - 1` |
| `phantom_positions` | Last column, rows 0..(grid_rows-2) |

---

## Build helpers

`GridConfig` provides methods that produce `ttnn.CoreRangeSet` objects ready for use in CB descriptors and kernel dispatch.

### build_matmul_core_grid()

Builds a `CoreRangeSet` for matmul cores, excluding DRAM workers and the entire last column (phantoms + sender). It produces row-contiguous `CoreRange` segments that skip over excluded positions:

```python
def build_matmul_core_grid(self):
    excluded = set(self.dram_worker_positions)
    for row in range(self.grid_rows):
        excluded.add((self.grid_cols - 1, row))
    # ... builds row-contiguous CoreRange segments
    return ttnn.CoreRangeSet(core_ranges)
```

The algorithm scans row by row, finds contiguous runs of non-excluded columns, and creates one `CoreRange` per contiguous run. This produces an efficient, minimal set of ranges -- important because tt-metal works more efficiently with coalesced ranges than with per-core ranges.

### build_mcast_receiver_grid()

Builds a `CoreRangeSet` containing every core except the sender. Used for multicast receiver configuration:

```python
def build_mcast_receiver_grid(self):
    sender = self.sender_core
    ranges = []
    for row in range(self.grid_rows):
        for col in range(self.grid_cols):
            if (col, row) == sender:
                continue
            ranges.append(ttnn.CoreRange(
                ttnn.CoreCoord(col, row), ttnn.CoreCoord(col, row)
            ))
    return ttnn.CoreRangeSet(ranges)
```

### build_ab_grids()

Splits the grid into balanced A (gate) and B (up) core sets for gated-linear-unit matmul patterns. This method excludes the sender core and one idle phantom `(last_col, grid_rows - 2)` to ensure an even total, then splits each row's remaining columns into left-half (A) and right-half (B), alternating which branch gets the extra core on odd-width rows to maintain global balance:

```python
def build_ab_grids(self):
    last_col = self.grid_cols - 1
    excluded = {self.sender_core, (last_col, self.grid_rows - 2)}

    a_cores = []
    b_cores = []
    a_surplus = 0  # Positive means A has more cores than B so far

    for row in range(self.grid_rows):
        cols = [col for col in range(self.grid_cols)
                if (col, row) not in excluded]
        n = len(cols)
        mid = n // 2

        if n % 2 == 1:
            if a_surplus <= 0:
                a_n = mid + 1
                a_surplus += 1
            else:
                a_n = mid
                a_surplus -= 1
        else:
            a_n = mid

        a_cores.extend((c, row) for c in cols[:a_n])
        b_cores.extend((c, row) for c in cols[a_n:])

    assert len(a_cores) == len(b_cores)
    return a_cores, b_cores
```

The assertion guarantees that both branches get exactly the same number of cores, which is critical for balanced workload distribution in gated matmul ops.

---

## DeviceContext

`DeviceContext` (defined in `blaze/device_context.py`) bundles a device handle, its `GridConfig`, and a pre-computed `full_device_grid` into a single object. It serves as the compilation-time interface to device topology:

```python
@dataclass
class DeviceContext:
    device: object
    grid_config: GridConfig
    full_device_grid: ttnn.CoreRangeSet

    def worker_core_from_logical_core(self, core) -> object:
        """Translate a logical CoreCoord to a physical (NOC) CoreCoord."""
        return self.device.worker_core_from_logical_core(core)

    @classmethod
    def from_device(cls, device) -> "DeviceContext":
        grid_config = GridConfig.from_device(device)
        grid_size = device.compute_with_storage_grid_size()
        full_device_grid = ttnn.CoreRangeSet([
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1),
            )
        ])
        return cls(
            device=device,
            grid_config=grid_config,
            full_device_grid=full_device_grid,
        )
```

### Why DeviceContext exists

`DeviceContext` serves two purposes:

1. **Isolation** -- Compilation code never touches the raw device directly. All grid queries go through `grid_config` and all NOC translations go through `worker_core_from_logical_core()`. This makes it possible to substitute a mock `DeviceContext` in tests.

2. **Mesh awareness** -- In mesh compilation, each device in the mesh gets its own `DeviceContext`. The `BlazeCompiler` creates one context from the `MeshDevice` (which exposes device-level APIs for any device in the mesh) and reuses it across all coordinates since Tenstorrent meshes are homogeneous.

### CoreCoord mapping: logical to physical NOC

The `worker_core_from_logical_core()` method translates **logical** coordinates (the `(col, row)` used in `GridConfig`) to **physical** NOC coordinates. This translation is device-specific because harvested rows/columns change the physical-to-logical mapping. Any op that needs to encode NOC addresses into CT args or RT args (multicast targets, gather destinations, fabric endpoints) must go through this translation:

```python
# In an op's emit() function
phys = mesh_device.worker_core_from_logical_core(worker_core)
core_noc_x, core_noc_y = phys.x, phys.y
```

In the compiler, `BlazeCompiler._build_grid_context()` uses this for mcast and gather ops:

```python
noc_start = ctx.worker_core_from_logical_core(start)
noc_end = ctx.worker_core_from_logical_core(end)
grid_context[node.id] = {
    "dest_noc_start_x": noc_start.x,
    "dest_noc_start_y": noc_start.y,
    "dest_noc_end_x": noc_end.x,
    "dest_noc_end_y": noc_end.y,
    ...
}
```

---

## FusedProgram grid helpers

When a `FusedProgram` is created, its `__init__` method calls `DeviceContext.from_device(device)` and pre-computes a complete set of grid-derived fields. These are the fields your op should use directly -- never call `device.compute_with_storage_grid_size()` from inside an `emit()` function.

Here is what `FusedProgram.__init__` sets up from the grid:

```python
self._ctx = DeviceContext.from_device(device)
self.grid = self._ctx.grid_config

# Precompute common grids
self.sender = ttnn.CoreCoord(*self.grid.sender_core)
self.sender_grid = ttnn.CoreRangeSet([ttnn.CoreRange(self.sender, self.sender)])
self.mcast_range = ttnn.CoreRange(
    ttnn.CoreCoord(0, 0),
    ttnn.CoreCoord(self.grid.grid_cols - 1, self.grid.grid_rows - 1),
)
self.all_cores = ttnn.CoreRangeSet([self.mcast_range])
self.num_mcast_cores = self.grid.total_cores

# Matmul core set
matmul_tuples = self.grid.get_matmul_cores()
self.num_matmul_cores = len(matmul_tuples)
self.matmul_cores_cc = [ttnn.CoreCoord(c, r) for c, r in matmul_tuples]
self.matmul_cores = ttnn.CoreRangeSet(
    [ttnn.CoreRange(c, c) for c in self.matmul_cores_cc]
)

# Mcast receiver grid (all cores except sender)
self.mcast_receiver_grid = ttnn.CoreRangeSet([
    ttnn.CoreRange(ttnn.CoreCoord(c, r), ttnn.CoreCoord(c, r))
    for r in range(self.grid.grid_rows)
    for c in range(self.grid.grid_cols)
    if (c, r) != self.grid.sender_core
])

# Full device grid for fallback
self.full_device_grid = self._ctx.full_device_grid

# NOC coordinates for sender and mcast range bounds
self.noc_sender = self._ctx.worker_core_from_logical_core(self.sender)
self.noc_start = self._ctx.worker_core_from_logical_core(self.mcast_range.start)
self.noc_end = self._ctx.worker_core_from_logical_core(self.mcast_range.end)
```

### Summary of FusedProgram grid attributes

| Attribute | Type | Description |
|---|---|---|
| `f.grid` | `GridConfig` | The device's grid configuration |
| `f.sender` | `CoreCoord` | Logical sender core |
| `f.sender_grid` | `CoreRangeSet` | Single-core range for the sender |
| `f.all_cores` | `CoreRangeSet` | Full device grid as a single range |
| `f.num_mcast_cores` | `int` | Total number of cores in the grid |
| `f.matmul_cores` | `CoreRangeSet` | Cores available for matmul (per-core ranges) |
| `f.matmul_cores_cc` | `list[CoreCoord]` | Same as above, as individual CoreCoords |
| `f.num_matmul_cores` | `int` | Count of matmul cores |
| `f.mcast_receiver_grid` | `CoreRangeSet` | All cores minus sender (for multicast) |
| `f.full_device_grid` | `CoreRangeSet` | Full grid from DeviceContext (bounding box) |
| `f.noc_sender` | `CoreCoord` | Physical NOC coord of sender |
| `f.noc_start` | `CoreCoord` | Physical NOC coord of grid origin (0,0) |
| `f.noc_end` | `CoreCoord` | Physical NOC coord of grid far corner |

### The flag() helper

The `flag()` method is a convenience for building per-core boolean CT args:

```python
def flag(self, name, cores, enabled=True):
    """Build a per-core CT arg tuple for a boolean flag.

    Returns a (name, mapping) tuple suitable for per_core_unified_ct_args().
    The flag is 1 on the given cores when enabled, 0 when disabled.
    Cores not in the range always get 0 (via other_value default).
    """
    return (name, {cores: int(enabled)})
```

This is the standard pattern for role flags like `is_sender_core`, `is_receiver_core`, and `is_active`:

```python
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_sender_core", sender_core_range_set),
    f.flag(f"{prefix}.is_receiver_core", receiver_core_range_set),
])
```

Cores not covered by the dict entries see 0 for all flags, effectively making them no-ops for the flagged kernel branch.

### ABGrid helper

For gated matmul ops that need balanced A/B core splits, `FusedProgram` provides `build_ab_grids()`:

```python
def build_ab_grids(self, ab_coords=None) -> ABGrid:
    if ab_coords is not None:
        a_coords, b_coords = ab_coords
    else:
        a_coords, b_coords = self.grid.build_ab_grids()
    return ABGrid.from_coords(a_coords, b_coords)
```

The `ABGrid` dataclass bundles everything a dual-branch matmul needs:

```python
@dataclass
class ABGrid:
    a_cores: list       # list[CoreCoord]
    b_cores: list       # list[CoreCoord]
    a_grid: CoreRangeSet
    b_grid: CoreRangeSet
    compute_grid: CoreRangeSet   # a_grid union b_grid
    num_per_branch: int          # len(a_cores) == len(b_cores)
```

You can pass explicit `ab_coords` for non-default splits, or let it fall back to the device-derived balanced split.

---

## Grid-awareness in practice

### Pattern: Using per-core flags with core range sets

The most common grid-aware pattern in a micro-op is setting per-core compile-time flags that tell the kernel which role each core plays. This requires building a `CoreRangeSet` for each role:

```python
@staticmethod
def emit(f: FusedProgram, ...):
    # Derive core sets from the grid
    sender_crs = f.sender_grid
    receiver_crs = f.mcast_receiver_grid

    # Set per-core flags
    f.per_core_unified_ct_args([
        f.flag(f"{prefix}.is_sender", f.sender_grid),
    ])
```

### Pattern: NOC coordinate injection

Ops that perform NOC transfers need physical coordinates. Always derive them from the device, never hardcode:

```python
worker_core = ttnn.CoreCoord(col, row)
phys = f.device.worker_core_from_logical_core(worker_core)
f.unified_ct_args([
    (f"{prefix}.core_noc_x", phys.x),
    (f"{prefix}.core_noc_y", phys.y),
])
```

### Pattern: Adapting to variable grid size

If an op needs to know how many cores are available for a computation, it should query the grid:

```python
num_compute = len(f.grid.get_compute_cores())
tiles_per_core = total_tiles // num_compute
```

Never assume `grid_cols == 13`. A harvested P150 has `grid_cols == 11`, meaning two fewer columns of compute cores.

### Common mistakes

**Hardcoded grid dimensions.** Never write `13` or `10` as grid constants. Use `self.grid.grid_cols` and `self.grid.grid_rows`.

**Hardcoded sender core.** `(12, 9)` is only correct on an unharvested 13x10 grid. Use `self.grid.sender_core` or `f.sender`.

**Forgetting NOC translation.** Logical `CoreCoord(5, 3)` is not physical `(5, 3)`. Any address passed to a kernel for NOC communication must go through `worker_core_from_logical_core()`.

**Assuming DRAM worker positions.** The 8 DRAM workers on a default grid are at specific positions, but harvested devices may relocate them. Always use `self.grid.get_dram_worker_cores()`.

### Correct pattern

```python
@staticmethod
def emit(f: FusedProgram, input_tensor, output_tensor, *, prefix="my_op"):
    # Use f.grid for all grid queries
    sender = f.sender
    sender_noc = f.noc_sender

    # Use grid-derived core sets for CB placement
    compute_cores = f.matmul_cores

    # Translate any core you need to address via NOC
    target = ttnn.CoreCoord(3, 2)
    target_noc = f.device.worker_core_from_logical_core(target)

    f.unified_ct_args([
        (f"{prefix}.target_noc_x", target_noc.x),
        (f"{prefix}.target_noc_y", target_noc.y),
    ])
```

The key principle: **let GridConfig and FusedProgram compute all grid-derived values from the device**. Never assume a specific grid size, core position, or NOC coordinate.

---

## Auxiliary utilities

### split_ab_cores

For ops that need custom A/B splits (not the balanced default), `role_engine.py` provides a `split_ab_cores()` utility:

```python
def split_ab_cores(
    a_cols_upper: set[int],
    a_cols_lower: set[int],
    upper_rows: range,
    lower_rows: range,
    *,
    exclude: Optional[set[tuple[int, int]]] = None,
    grid_cols: int = 13,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
```

This lets callers assign specific columns to the A or B branch in different row ranges, with an exclusion set for sender/phantom cores. It is used by model-specific grid layouts (e.g. `_deepseek_grids.py`) where the automatic split does not produce the desired mapping.

### compute_k_n_parallel

Also in `role_engine.py`, `compute_k_n_parallel()` finds a balanced factorization of the per-branch core count for KN-sliced matmul:

```python
def compute_k_n_parallel(num_per_branch, K_gate_tiles):
    """Find balanced k_parallel, n_parallel for gate/up matmul.

    Requirements:
        k_parallel * n_parallel == num_per_branch
        K_gate_tiles % k_parallel == 0
    Searches from sqrt(num_per_branch) downward for balanced factoring.
    """
    target = int(math.sqrt(num_per_branch))
    for k in range(target, 0, -1):
        if num_per_branch % k == 0 and K_gate_tiles % k == 0:
            return k, num_per_branch // k
    # searches upward if no downward factor found
```

This is grid-aware because `num_per_branch` comes from `ABGrid.num_per_branch`, which is itself derived from the device's actual grid dimensions.

---

## Key takeaways

1. **Never hardcode grid dimensions.** Use `GridConfig.from_device()` or rely on `FusedProgram.grid`.
2. **Exclude DRAM workers from compute grids.** The `get_compute_cores()` and `get_matmul_cores()` methods handle this automatically.
3. **Use `worker_core_from_logical_core()` for NOC addresses.** Logical-to-physical mapping is device-specific due to harvesting.
4. **The sender core is at `(grid_cols - 1, grid_rows - 1)`.** Its position shifts with grid size -- always access it via `grid.sender_core` or `f.sender`.
5. **Phantom columns are dynamic.** `num_phantom_cols()` returns 1 or 2 depending on the number of gate cores needed.
6. **`DeviceContext` is the compilation-time device abstraction.** Pass it around instead of raw device handles.
7. **`FusedProgram` pre-computes everything.** Use `f.sender`, `f.matmul_cores`, `f.noc_sender`, etc. rather than recomputing from the device.

| Concept | Source | Role |
|---|---|---|
| `GridConfig` | `blaze/role_engine.py` | Frozen dataclass holding grid dimensions, DRAM positions, core accessors |
| `GridConfig.from_device()` | Same | Constructs from real device query |
| `GridConfig.default()` | Same | Fixed 13x10 for offline testing |
| `DeviceContext` | `blaze/device_context.py` | Bundles device + `GridConfig` + `full_device_grid` |
| `DeviceContext.worker_core_from_logical_core()` | Same | Logical-to-physical NOC translation |
| `FusedProgram.grid` | `blaze/fused_program.py` | Alias to `GridConfig` on the current device |
| `FusedProgram.sender`, `.sender_grid`, `.all_cores`, etc. | Same | Pre-computed `CoreCoord`/`CoreRangeSet` fields |
| `FusedProgram.flag()` | Same | Convenience for per-core boolean CT args |
| `FusedProgram.build_ab_grids()` | Same | Balanced A/B split for gated matmul |
| `GridConfig.build_matmul_core_grid()` | `blaze/role_engine.py` | `CoreRangeSet` excluding DRAM workers and phantoms |
| `GridConfig.get_gate_mm_cores(n)` | Same | Phantom-column placement for gate weights |
| `split_ab_cores()` | Same | Custom column-set A/B split |
| `compute_k_n_parallel()` | Same | Balanced KN factorization for matmul |
| `SENDER_CORE` | Same | Module-level constant `(12, 9)` for backward compat |
