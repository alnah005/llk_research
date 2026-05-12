# L1 Memory Model: Per-Core SRAM Layout and Budget Constraints

Every automatic CB sizing decision depends on a single physical constraint: the amount of L1 SRAM available per Tensix core. On Blackhole silicon, each core has approximately 1.5 MB of usable L1, shared among all circular buffers, scratch buffers, firmware stack, semaphores, and runtime metadata. This section builds a precise model of how that memory is partitioned, what competes for it, how existing Blaze ops budget their allocations, why overcommitting crashes silently rather than failing at compile time, and how the existing `l1_profile.py` module provides visibility into the actual allocation. Understanding this model is the prerequisite for designing any automatic sizing system -- the budget ceiling determines every downstream decision about page counts, blocking factors, and multi-pass tiling strategies.

> *Design Principle P3 (Validate at Each Lowering Step, Not Just at the End):* The current L1 allocation path provides no compile-time bounds checking. Silent crashes from L1 overcommit are the single most common debugging nightmare for Blaze op developers. An automatic sizing system must validate the L1 budget at each allocation step, catching overcommit before it becomes a silent runtime crash.

**What you will learn:**

- The physical layout of L1 SRAM on a Blackhole Tensix core: total capacity, usable capacity, and what consumes the reserved portion
- The 64 CB slot hardware limit (`MAX_CB_ID = 64` in `cb_engine.py`) and why it constrains large fused ops
- Every category of L1 consumer: CB pages, scratch buffers, stack/firmware, semaphores, runtime metadata, and kernel text
- How current Blaze ops budget L1 through `FusedProgram.cb_scratch()` and `FusedProgram.cb_from_tensor()`
- Why L1 overcommit crashes silently and the four concrete failure modes on silicon
- How `l1_profile.py` reports CB addresses, page sizes, and utilization for post-hoc analysis
- The gap between what developers need (compile-time budget enforcement) and what exists today (runtime crash or post-hoc profiling)

---

## 1. Physical L1 Layout on Blackhole

Each Tensix core on Blackhole silicon contains a local SRAM bank called L1. The raw physical capacity is approximately 1.5 MB (1,572,864 bytes). Not all of this is available for application use -- firmware, the RISC-V stack, and runtime data structures consume a fixed portion at the bottom of the address space.

```text
L1 Address Space (per Tensix core, Blackhole)
+------------------------------------------------------+ 0x180000 (~1.5 MB)
|                                                      |
|          Application-Available Region                |
|                                                      |
|   CB pages, scratch buffers, tensor-backed buffers   |
|                                                      |
+------------------------------------------------------+ ~0x20000 (128 KB)
|       Reserved: firmware, stack, mailbox,             |
|       semaphores, runtime metadata, NOC state         |
+------------------------------------------------------+ 0x00000
```

The reserved region at the base of L1 includes:

| Component | Approximate Size | Purpose |
|-----------|-----------------|---------|
| Firmware code/data | ~48 KB | BRISC/NCRISC/TRISC firmware images |
| RISC-V stack | ~16 KB (shared across 3+3 RISCs) | Function call frames, local variables |
| Mailbox/sync | ~4 KB | Inter-RISC communication slots |
| Semaphores | ~2 KB (up to 8 global semaphores x 16 bytes) | Hardware synchronization primitives |
| NOC state | ~8 KB | NOC transaction descriptors, read/write buffers |
| Runtime metadata | ~16 KB | Kernel dispatch tables, CT arg storage |

After these fixed allocations, approximately **1.35-1.40 MB** remains for application use. This is the L1 budget that CB sizing must respect.

> **Warning:** The exact usable capacity varies between firmware versions and kernel configurations. The numbers above are representative of the Blackhole `blaze-metal` branch as of 2026. Using a hard-coded 1.5 MB budget in sizing calculations will silently overcommit by 100-150 KB.

### How L1 Addresses Are Assigned

Metal's L1 allocator assigns addresses from the top of the application region downward for tensor-backed CBs (which have pre-allocated `ttnn.Tensor` buffers), and from the bottom of the application region upward for scratch CBs (which are allocated by Metal at dispatch time). This bidirectional allocation strategy maximizes the usable space but creates a fragmentation risk when many small allocations are interleaved.

```text
Application L1 Region (~1.35 MB)
+------------------------------------------------------+ Top (high address)
|  Tensor-backed CB: weights (direct-address, largest) |
|  Tensor-backed CB: activation input                  |
|  Tensor-backed CB: output                            |
+------------------------------------------------------+
|                                                      |
|              Free / Fragmented Space                 |
|                                                      |
+------------------------------------------------------+
|  Scratch CB: matmul intermediate                     |
|  Scratch CB: rmsnorm intermediate                    |
|  Scratch CB: mcast destination                       |
+------------------------------------------------------+ Bottom (low address, above reserved)
```

---

## 2. The 64 CB Slot Hardware Limit

Blackhole hardware supports a maximum of 64 circular buffer IDs per core, numbered 0 through 63. This is encoded in `cb_engine.py`:

```python
# EXISTING -- from blaze/cb_engine.py (line 65)
MAX_CB_ID = 64
```

And in `cb_reconfig.py`:

```python
# EXISTING -- from blaze/cb_reconfig.py (line 26)
class CircularBufferIdManager:
    NUM_CIRCULAR_BUFFERS = 64
```

Each CB ID maps to a hardware register set that holds the buffer's base address, page size, number of pages, and read/write pointers. The hardware has exactly 64 such register sets -- there is no way to exceed this limit without CB ID reuse.

### Two Orthogonal Resource Constraints

L1 memory has two independent limits that must both be satisfied:

1. **Byte budget**: total CB bytes must fit within ~1.3 MB
2. **Slot count**: total active CB IDs must not exceed 64

These constraints are independent. A program could use only 2 CB slots but exhaust L1 bytes (two massive CBs), or it could use 64 CB slots but total only 128 KB (64 tiny CBs). Both constraints must be checked.

### Why 64 Is Not Enough for Large Fused Ops

A single fused op chain can consume CB IDs rapidly. Consider the SwigluOp from `blaze/ops/swiglu/op.py`, which chains six micro-ops:

```text
SwigluOp CB allocation:
  DRAMStreamingMatmul (up):    act_cb, weights_cb, out_cb, index_cb     = 4 CBs
  DRAMStreamingMatmul (gate):  (shares act_cb), weights_cb, out_cb      = 2 new CBs
  EltwiseMul:                  out_cb                                    = 1 new CB
  Gather:                      dst_cb                                    = 1 new CB
  Mcast:                       src_cb, dst_cb                            = 2 new CBs
  DRAMStreamingMatmul (down):  (shares mcast dst), weights_cb, out_cb   = 2 new CBs
  ─────────────────────────────────────────────────────────────
  Total without sharing:                                                 ~12 CBs
```

For a single SwigluOp, 12 CBs is manageable. But an MoE layer in DeepSeek V3 contains:

- 1 RMSNorm (3 CBs: input, gamma, output)
- 1 Mcast for activation broadcast (2 CBs)
- 1 MoE gate (3-4 CBs)
- 1 Routed expert SwigluOp (~12 CBs)
- 1 Shared expert pipeline (~15 CBs)
- 1 GatedReduce (3 CBs)
- 1 AllReduce (2 CBs)
- 1 ResidualAdd (3 CBs)

Without CB ID reuse, this totals approximately **43-45 CBs** -- approaching the 64-slot limit. For models with more complex MoE patterns (GLM-5.1 with dynamic sparse attention), the count can exceed 50 CBs before compaction.

The `CBEngine` warns when approaching the limit:

```python
# EXISTING -- from blaze/cb_engine.py (lines 377-383)
# Warn if approaching the hardware limit
if next_cb_id > self.max_cb_id - 8:
    warnings.warn(
        f"CB usage is high: {next_cb_id}/{self.max_cb_id} CBs assigned. "
        f"Approaching Blackhole hardware limit.",
        stacklevel=2,
    )
```

This warning threshold of `max_cb_id - 8` (i.e., 56 CBs) provides only 8 slots of headroom. An automatic sizing system must work within this limit, which is why CB compaction (File 04 of this chapter) is essential.

---

## 3. What Competes for L1

Every byte of L1 consumed by one allocation is a byte unavailable for others. The L1 consumers fall into five categories:

### Category 1: Tensor-Backed CB Pages

Tensor-backed CBs are the largest consumers. Their L1 footprint is determined by the backing tensor's shard size, which is computed during tensor preparation:

```python
# EXISTING -- from blaze/fused_program.py (lines 106-139)
def _tensor_shard_bytes(tensor) -> int | None:
    """Best-effort byte size of one shard of a tensor-backed L1 buffer."""
    try:
        spec = tensor.memory_config().shard_spec
    except Exception:
        return None
    if spec is None:
        return None

    shard_h, shard_w = spec.shape
    # ... tile size computation ...
    if shard_h % tile_h or shard_w % tile_w:
        return None
    return (shard_h // tile_h) * (shard_w // tile_w) * tile_size
```

For a matmul weight tensor sharded across 7 cores with K=4096 and N_per_core=1568 (DeepSeek V3 MoE expert weights in bfloat8_b):

```text
shard_shape = (4096, 1568)       # K rows x N_per_core columns
tile_shape  = (32, 32)           # standard 32x32 tiles
tiles       = (4096/32) * (1568/32) = 128 * 49 = 6,272 tiles
page_size   = 1,088 bytes        # bfloat8_b tile size
shard_bytes = 6,272 * 1,088      = 6,823,936 bytes (~6.5 MB)
```

This single weight shard exceeds the entire L1 capacity. That is why the Matmul op uses `DIRECT_ADDRESS` access mode -- the weight tensor resides in DRAM and is read tile-by-tile via NOC, never buffered entirely in L1. Only the working set (a few tiles at a time) occupies L1.

For DRAMStreamingMatmul, the weights CB is triple-buffered with `subblock_k` tiles per buffer:

```python
# EXISTING -- from blaze/ops/dram_streaming_matmul/op.py (lines 161-172)
num_weights_buffers = 3
weights_cb = f.cb_scratch(
    name=BlazeOp.cb_name(prefix, "weights_cb"),
    num_pages=subblock_k * num_weights_buffers,
    core_ranges=compute_cores,
    data_format=weights_dtype,
    tile=ttnn.TileDescriptor(weights_tile),
    page_size=weights_tile_size,
)
```

With `subblock_k=32` and `weights_tile_size=1,088` bytes (bfloat8_b):

```text
weights_cb L1 = 32 * 3 * 1,088 = 104,448 bytes (~102 KB)
```

This is a manageable 7% of the L1 budget.

### Category 2: Scratch CB Pages

Scratch CBs hold intermediate results between micro-ops within a fused kernel. Their size is `num_pages * page_size`, allocated via `FusedProgram.cb_scratch()`:

```python
# EXISTING -- from blaze/fused_program.py (lines 1540-1593)
def cb_scratch(
    self,
    name: str,
    *,
    num_pages,
    core_ranges,
    data_format,
    tile,
    page_size,
    balanced: bool = True,
) -> CBHandle:
    """Allocate a scratch CB."""
    # ... mapping and allocation logic ...
    cb_id = self.program.cb_scratch(
        name=name,
        num_pages=num_pages,
        core_ranges=core_ranges,
        data_format=data_format,
        tile=tile,
        page_size=page_size,
    )
```

A typical matmul output scratch CB with `out_w_per_core=49` tiles of bfloat16:

```text
scratch_cb L1 = 49 * 2,048 = 100,352 bytes (~98 KB)
```

### Category 3: Firmware, Stack, and Kernel Text

As described above, approximately 128 KB is consumed by fixed firmware allocations. Additionally, the compiled kernel binary for each RISC occupies L1 space. A complex fused kernel with many phases can have a code footprint of 20-40 KB per RISC (BRISC, NCRISC, TRISC), consuming up to 120 KB total.

### Category 4: Global Semaphores

Global semaphores are allocated via `ttnn.create_global_semaphore()` and occupy 16 bytes each in L1. A typical fused op uses 2-4 semaphores (mcast sender/receiver, gather sync), consuming 32-64 bytes -- negligible relative to CB allocations.

### Category 5: CB Reconfig Tensor

For multi-phase programs that use CB reconfiguration (via `cb_reconfig.py`), a per-core reconfig tensor is allocated in L1. The reconfig tensor encodes all 64 CB slots for each core:

```python
# EXISTING -- from blaze/cb_reconfig.py (lines 162-168)
# 264 uint32 per core: 64 CBs x 4 words [addr, size, num_pages, page_size],
# 2 mask words (which CBs are active), 2 sync semaphores, 4 reserved.
WORDS_PER_CORE = 264
```

The 264-word breakdown: `64 CBs * 4 words (addr, size, num_pages, page_size) = 256 words + 2 active-mask words + 2 sync semaphore words + 4 reserved words = 264 words`. This consumes `264 * 4 = 1,056 bytes` per core -- roughly 1 KB of L1 overhead for reconfigurable programs. The active mask is a 64-bit bitmask (two uint32 words at positions 256 and 257) indicating which CB IDs are configured in each phase; the kernel only reconfigures CBs whose bits are set.

### L1 Budget Summary

```text
L1 Budget Breakdown (typical MoE layer, Blackhole)
─────────────────────────────────────────────────────
Total L1:                           ~1,500 KB (1,536,000 bytes)
Reserved (firmware/stack/NOC):      ~  128 KB
Kernel text (3 RISCs):              ~   80 KB
CB reconfig tensor:                 ~    1 KB
Semaphores:                         ~    0.1 KB
─────────────────────────────────────────────────────
Available for CBs:                  ~1,291 KB (~1.26 MB)
─────────────────────────────────────────────────────

Typical CB allocation:
  Weight working buffer (3x32 bfp8 tiles):   ~102 KB
  Activation CB (128 bf16 tiles):            ~  262 KB
  Matmul output CB (49 bf16 tiles):          ~  100 KB
  RMSNorm CBs (input + gamma + output):      ~  150 KB
  Mcast CBs (source + destination):          ~   50 KB
  Intermediate scratch CBs (2-3):            ~  100 KB
─────────────────────────────────────────────────────
Total CB allocation:                         ~  764 KB
Headroom:                                    ~  527 KB (41%)
```

This 41% headroom for a single MoE expert is comfortable, but a fused MoE layer running shared and routed experts concurrently can push utilization above 90%, leaving almost no margin for error.

---

## 4. How Current Blaze Ops Budget L1

Current Blaze ops make all L1 budgeting decisions manually within their `emit()` methods. There is no centralized budget tracker -- each op independently computes its CB sizes based on local information.

### The Matmul Pattern

The Matmul op in `blaze/ops/matmul/op.py` derives CB sizes from its input handles:

```python
# EXISTING -- from blaze/ops/matmul/op.py (lines 48-74)
k_num_tiles = in0_handle.num_pages
page_size = in0_handle.page_size
data_format = in0_handle.data_format

# ... weight resolution ...
out_w_per_core = in1_cb.num_pages // k_num_tiles

out = f.cb_scratch(
    name=BlazeOp.cb_name(prefix, "out"),
    num_pages=out_w_per_core,
    core_ranges=active_ranges,
    data_format=data_format,
    tile=tile_desc,
    page_size=page_size,
)
```

The key sizing decisions:
1. **`k_num_tiles`** comes from the activation input's `num_pages` -- how many tiles of the K dimension are buffered
2. **`out_w_per_core`** is derived from the weight shard's total pages divided by K tiles -- how many output columns each core produces
3. **Output CB size** is `out_w_per_core` pages, holding one complete output row per core

None of these decisions consider whether the total L1 allocation across all CBs in the fused op fits within the available budget. Each op trusts that the developer has done the arithmetic correctly.

### The RMSNorm Pattern

RMSNorm in `blaze/ops/rmsnorm/op.py` computes `num_tiles` from the activation width and allocates accordingly:

```python
# EXISTING -- from blaze/ops/rmsnorm/op.py (lines 108-139)
if has_row_tiles(input_handle):
    if width is None:
        width = input_handle.shape[1]
    itile, n_tiles = interpret_tile(width)
    page_sz = itile.get_tile_size(ttnn.bfloat16)
    num_tiles = num_tiles or n_tiles
    # ...

out_cb = f.cb_scratch(
    name=BlazeOp.cb_name(prefix, "out_cb"),
    num_pages=compute_n,
    core_ranges=cores,
    data_format=out_fmt,
    tile=out_tile,
    page_size=out_page_sz,
)
```

The `interpret_tile()` function from `utils.py` selects between 32x32 and 16x32 tile geometries based on width:

```python
# EXISTING -- from blaze/utils.py (lines 121-133)
def interpret_tile(width: int) -> tuple[ttnn.Tile, int]:
    FULL = ttnn.Tile((32, 32))
    HALF = ttnn.Tile((16, 32))
    use_half = (width // FULL.tile_shape[1]) % FULL.tile_shape[0] != 0
    tile = HALF if use_half else FULL
    th, tw = tile.tile_shape
    return tile, width // (th * tw)
```

For DeepSeek V3's hidden dimension of 7,168 elements:

```text
width = 7168
FULL = (32, 32) -> 1024 elements/tile
7168 // 32 = 224 rows
224 % 32 = 0 -> use FULL tile
num_tiles = 7168 / 1024 = 7
```

RMSNorm allocates 7 tiles for input, 7 for gamma, and 7 for output = 21 tiles total. At 2,048 bytes per bfloat16 tile, that is 43,008 bytes (~42 KB) -- well within budget for a single op.

### The DRAMStreamingMatmul Pattern

DRAMStreamingMatmul shows the most explicit sizing calculations:

```python
# EXISTING -- from blaze/ops/dram_streaming_matmul/op.py (lines 126-172)
# Subblock K derivation
if subblock_k is None:
    subblock_k = max(1, Kt // 4)
while Kt % subblock_k != 0 and subblock_k > 1:
    subblock_k -= 1

# Subblock W (N output subblock)
subblock_w = compute_subblock_w(
    per_core_N,
    fp32_dest_acc_en=fp32_dest_acc_en,
    dst_full_sync_en=False,
)

# NOC page sizing
weights_page_size, weights_num_pages = _get_max_page_size_and_num_pages(
    f.device, subblock_k, weights_tile_size,
)

# Triple-buffered weights CB
num_weights_buffers = 3
weights_cb = f.cb_scratch(
    name=BlazeOp.cb_name(prefix, "weights_cb"),
    num_pages=subblock_k * num_weights_buffers,
    ...
)
```

The `subblock_k` derivation (`max(1, Kt // 4)`) is a heuristic that balances L1 usage against compute efficiency. Smaller `subblock_k` means fewer tiles buffered simultaneously (less L1), but more NOC transactions (more latency). This is exactly the kind of trade-off an automatic system should optimize.

---

## 5. Why L1 Overcommit Crashes Silently

The most dangerous property of the current L1 allocation model is the absence of compile-time bounds checking. When a fused op's total CB allocation exceeds the available L1, the failure manifests in one of four concrete modes:

### Failure Mode 1: Silent Data Corruption

Two CBs whose address ranges overlap write to the same L1 bytes. CB A's write overwrites CB B's data. CB B's consumer reads corrupted data. The matmul produces wrong results and the model outputs nonsensical tokens. No error is raised. No log message appears. Debugging requires inspecting L1 contents with a hardware debugger or running `l1_profile.py` to detect the address overlap post-hoc.

### Failure Mode 2: Hardware Hang

A CB's data region overlaps with firmware stack space. A `cb_push_back()` overwrites the RISC-V return address. The RISC-V processor jumps to an invalid address. The core hangs, the program times out, and the error message is "Device timeout" with no indication of root cause.

### Failure Mode 3: Host-Side SIGSEGV

The L1 address overflows the physical address space. Metal's device write function receives an invalid address. The host process crashes with SIGSEGV in the Metal runtime. The stack trace points to a Metal internal function, not user code.

### Failure Mode 4: NOC Write Corruption

A multicast operation writes to CB addresses on remote cores. If the source core's CB extends beyond L1, the multicast writes invalid data to all receiver cores simultaneously. All cores produce wrong results. The corruption propagates through the entire core grid.

### The Debugging Experience

The developer's experience across all four modes is: "my matmul produces wrong results" or "the device hangs after 10 seconds." Debugging requires:

1. Running `l1_profile.py` to dump CB addresses
2. Manually checking for overlapping address ranges
3. Tracing back to which `cb_scratch()` call allocated too many pages
4. Adjusting `num_pages` or `subblock_k` by trial and error

This debugging cycle typically takes 2-4 hours per incident. An automatic sizing system with compile-time budget validation (P6) would eliminate this class of bugs entirely.

### The `_tensor_shard_bytes()` Safety Check

The only existing L1 bounds check is `_tensor_shard_bytes()` in `fused_program.py`, which computes the per-core footprint of a tensor-backed CB:

```python
# EXISTING -- from blaze/fused_program.py (lines 106-139)
def _tensor_shard_bytes(tensor) -> int | None:
    """Best-effort byte size of one shard of a tensor-backed L1 buffer."""
    # ... extraction logic ...
    return (shard_h // tile_h) * (shard_w // tile_w) * tile_size
```

This function is informational -- it computes the size but does not compare it against a budget. No caller currently uses it for enforcement.

---

## 6. The l1_profile.py Module

TT-Blaze provides `blaze/l1_profile.py` as a post-hoc profiling tool for CB allocations. It reports CB addresses, sizes, types, and data formats after a program has been compiled and dispatched.

### Usage

```python
# EXISTING usage pattern
from blaze.l1_profile import print_cb_stats, print_kernel_stats

program = fp.build()          # CompiledProgram or MeshCompiledProgram
print_cb_stats(program)       # CB type, L1 addr, dtype, page_size, tile per descriptor
print_kernel_stats(program)   # kernel role, source stem, core_ranges
```

The module is activated by the `BLAZE_L1_PROFILE` environment variable, which triggers automatic profiling after every `generic_op` dispatch.

### Output Format

`print_cb_stats()` reports three categories of CBs:

```python
# EXISTING -- from blaze/l1_profile.py (lines 347-350)
if has_buf:
    cb_type = "aliased" if num_fmts > 1 else "tensor_backed"
else:
    cb_type = "scratch"
```

A typical output looks like:

```text
cb_stats: 12 CB descriptor(s)
  cb[0] type=tensor_backed buffer_index=0 total_size_B=14336 l1_addr=0x1a2000
    [dtype=DataType.BFLOAT16 page_size_B=2048 tile=32x32]
    core_ranges={(x=0,y=0) - (x=6,y=0)}
  cb[1] type=tensor_backed buffer_index=1 total_size_B=100352 l1_addr=0x78000
    [dtype=DataType.BFLOAT8_B page_size_B=1088 tile=32x32]
    core_ranges={(x=0,y=0) - (x=6,y=0)}
  cb[2] type=scratch buffer_index=2 total_size_B=100352 l1_addr=metal-managed
    [dtype=DataType.BFLOAT16 page_size_B=2048 tile=32x32]
    core_ranges={(x=0,y=0) - (x=6,y=0)}
```

### L1 Overlap Detection

The module includes `_build_overlap_groups()` (lines 410-479) which performs union-find on CB address ranges to detect overlapping allocations:

```python
# EXISTING -- from blaze/l1_profile.py (lines 410-479)
def _build_overlap_groups(cbs: dict[str, dict]) -> dict[str, list[dict]]:
    """Group CBs that share any L1 bytes (partial or full overlap)."""
    # ... union-find implementation ...
```

This is useful for diagnosing L1 overcommit after the fact, but it operates on the dispatched program -- the overlap has already occurred and potentially corrupted results.

### Blaze vs Blitz CB Address Capture

The `l1_profile.py` docstring (lines 22-53) documents two distinct address capture paths:

**Blaze path:** CB addresses come from Metal's L1 allocator at runtime. Tensor-backed CBs get `buffer->address() + address_offset`. Scratch CBs are assigned by Metal during dispatch.

**Blitz path:** CB addresses come from the CbReconfig tensor -- a pre-allocated `[n_cores, 264]` UINT32 L1 tensor that encodes all 64 CB slots for each core. This is the path used by multi-phase fused ops.

```python
# EXISTING -- from blaze/cb_reconfig.py (lines 162-168)
WORDS_PER_CORE = 264
config = torch.zeros((num_cores, WORDS_PER_CORE), dtype=torch.uint32)
```

---

## 7. The Gap: No Compile-Time Budget Enforcement

The fundamental gap that an automatic sizing system must address:

| Capability | Current State | Required State |
|-----------|--------------|---------------|
| Total L1 computation | Manual (developer calculates) | Automatic (system sums all CBs) |
| Budget comparison | None | Compare total vs available L1 |
| Overcommit detection | Post-hoc via `l1_profile.py` | Compile-time error with actionable message |
| Budget distribution | Manual (developer sizes each CB) | Automatic with priority-based allocation |
| Cross-op awareness | None (each op sizes independently) | Global view of all CBs in fused program |
| Reporting | Per-CB profiling after dispatch | Budget summary at compile time |

### What the Automatic System Needs from This Model

The L1 memory model provides three critical inputs to the automatic sizing algorithm:

1. **Available budget:** `total_l1 - reserved - kernel_text - reconfig_overhead - semaphores`
2. **Per-CB cost function:** `cost(cb) = cb.num_pages * cb.page_size`
3. **The 64-slot constraint:** Total unique CB IDs must not exceed `MAX_CB_ID`

The sizing algorithm (File 02) uses these to compute feasible `num_pages` per CB. The blocking strategy (File 03) uses the budget to select the optimal `(RT, CT, KT)` triple. And the compaction pass (File 04) ensures the 64-slot constraint is satisfied.

### Proposed L1 Budget Tracker

```python
# PROPOSED -- compile-time L1 budget enforcement
@dataclass
class L1Budget:
    """Tracks L1 consumption across all CBs in a FusedProgram."""

    total_l1_bytes: int             # Physical L1 capacity
    reserved_bytes: int             # Firmware + stack + NOC
    kernel_text_bytes: int          # Estimated RISC binary sizes
    reconfig_overhead_bytes: int    # CB reconfig tensor (if multi-phase)
    semaphore_bytes: int            # Global semaphore allocations

    _allocations: dict[str, int] = field(default_factory=dict)

    @property
    def available_bytes(self) -> int:
        return (self.total_l1_bytes - self.reserved_bytes
                - self.kernel_text_bytes - self.reconfig_overhead_bytes
                - self.semaphore_bytes)

    @property
    def consumed_bytes(self) -> int:
        return sum(self._allocations.values())

    @property
    def remaining_bytes(self) -> int:
        return self.available_bytes - self.consumed_bytes

    def allocate(self, name: str, num_pages: int, page_size: int) -> None:
        size = num_pages * page_size
        if self.consumed_bytes + size > self.available_bytes:
            raise L1BudgetExceeded(
                f"CB '{name}' requests {size:,} bytes ({num_pages} pages x "
                f"{page_size} bytes/page), but only {self.remaining_bytes:,} "
                f"bytes remain of {self.available_bytes:,} available.\n"
                f"Current allocations:\n"
                + "\n".join(f"  {k}: {v:,} B" for k, v in self._allocations.items())
            )
        self._allocations[name] = size

    def utilization(self) -> float:
        return self.consumed_bytes / self.available_bytes if self.available_bytes > 0 else 1.0
```

This tracker would be integrated into `FusedProgram`, called by every `cb_scratch()` and `cb_from_tensor()` invocation. The error message (P6) tells the developer exactly which CB caused the overcommit and how much L1 is consumed by each existing allocation.

### The Feedback Loop: L1 Budget Failures Propagate Upstream

When the L1 budget is exceeded, the automatic system does not simply fail -- it triggers a feedback loop that adjusts upstream decisions. This cross-chapter integration is critical for a self-consistent pipeline:

1. **Drop double-buffering** -- 10-20% throughput cost, no structural change
2. **Switch to DRAM streaming** -- weight CB shrinks dramatically, DRAM bandwidth becomes bottleneck
3. **Reduce format precision** (bfloat16 to bfloat8_b per [Chapter 6](../ch06_data_formats/) format negotiation) -- 47% page_size reduction, precision loss
4. **Increase core count** (per [Chapter 4](../ch04_tile_decomposition/) `GridConfig`) -- shard width halves, more all-gather overhead
5. **Fail with actionable error** (P6) -- developer must restructure

This priority ordering ensures the system exhausts transparent optimizations before requiring developer intervention. The format fallback (step 3) connects to Chapter 6's precision profiles; the core count adjustment (step 4) connects to Chapter 4's `pick_matmul_cores()`.

---

## 8. Worked Example: DeepSeek V3 MoE Layer L1 Budget

To make the L1 model concrete, let us trace through the L1 budget for a single DeepSeek V3 MoE expert layer running on 7 compute cores:

```text
Model parameters:
  hidden_dim = 7,168
  intermediate_dim = 18,432 (per expert)
  K = 7,168 (activation dimension)
  N = 18,432 / 7 cores = 2,633 cols/core (rounded to 2,624 = 82 tiles of 32)
  Data format: bfloat8_b for weights, bfloat16 for activations

Per-core L1 budget:
  Total L1:        ~1,500 KB
  Reserved:        ~  128 KB
  Kernel text:     ~   80 KB
  Reconfig:        ~    1 KB
  Semaphores:      ~    0.1 KB
  ─────────────────────────────
  Available:       ~1,291 KB

Per-core CB allocations (shared expert gate+up path):
  Activation CB (interpret_tile: 7 tiles x 2,048 B):           14,336 B  ( 14 KB)
  Gamma CB (RMSNorm: 7 tiles x 2,048 B):                      14,336 B  ( 14 KB)
  RMSNorm output CB (7 tiles x 2,048 B):                      14,336 B  ( 14 KB)
  Mcast destination CB (7 tiles x 2,048 B):                    14,336 B  ( 14 KB)
  KN-sliced matmul weights (direct-address, not in L1):             0 B  (  0 KB)
  KN-sliced matmul output CB (varies by k_parallel):          ~40,960 B  ( 40 KB)
  GatedReduce intermediate (gate * up):                        ~40,960 B  ( 40 KB)
  Mcast for down projection (varies):                          ~14,336 B  ( 14 KB)
  DRAMStreaming down weights (3 x subblock_k x 1,088 B):     ~104,448 B  (102 KB)
  DRAMStreaming down output CB:                                ~40,960 B  ( 40 KB)
  ─────────────────────────────────────────────────────────────────────
  Total:                                                      ~299,008 B  (~292 KB)
  Utilization:                                                        23%
```

This 23% utilization leaves substantial headroom. But when the routed expert runs concurrently with additional CBs for the MoE gate, index tensors, and scalar weights, utilization climbs to 50-60%. For models with larger intermediate dimensions or fp32 accumulation, the budget becomes tight.

---

## Key Takeaways

- Each Blackhole Tensix core has approximately 1.5 MB of L1 SRAM, of which **approximately 1.3 MB is available** for CB allocations after firmware, stack, kernel text, and runtime overhead are subtracted.
- The **64 CB slot hardware limit** (`MAX_CB_ID = 64`) constrains large fused ops. A DeepSeek V3 MoE layer uses 40-50 CBs before compaction; GLM-5.1 MoE layers can approach the limit.
- **L1 overcommit has no compile-time detection.** When total CB allocations exceed available L1, the result is one of four failure modes: silent data corruption, hardware hang, host SIGSEGV, or NOC write corruption -- never a helpful error message.
- Current ops budget L1 **independently and manually** within their `emit()` methods. Each op trusts the developer to have computed the total budget correctly. No centralized tracker exists.
- The `l1_profile.py` module provides **post-hoc visibility** into CB addresses and overlap detection, but operates after the damage has occurred. A compile-time budget tracker (P6) is needed to close this gap.
- An automatic sizing system must model three constraints: the **L1 byte budget** (determines num_pages per CB), the **64-slot limit** (determines need for compaction), and the **allocation pattern** (tensor-backed vs scratch, top-down vs bottom-up addressing).
- When L1 budget is exceeded, the system triggers a **feedback loop** that adjusts format (Chapter 6), core grid (Chapter 4), or streaming strategy before failing to the developer.

## Source Files

- `blaze/cb_engine.py` -- `MAX_CB_ID = 64` (line 65), `DTYPE_BYTES` (lines 20-27), `_tile_page_size()` (lines 73-79), `CBAssignment` dataclass (lines 122-138), `CBEngine.assign()` (lines 159-387), `compact_cb_ids()` (lines 422-437)
- `blaze/fused_program.py` -- `FusedProgram.cb_scratch()` (lines 1540-1593), `FusedProgram.cb_from_tensor()` (lines 1050-1065), `_tensor_shard_bytes()` (lines 106-139), `TileInfo.from_tensor()` (lines 41-49)
- `blaze/l1_profile.py` -- `print_cb_stats()` (lines 328-402), `print_kernel_stats()` (lines 311-325), `_build_overlap_groups()` (lines 410-479), `extract_cb_names()` (line 482)
- `blaze/cb_reconfig.py` -- `CircularBufferIdManager.NUM_CIRCULAR_BUFFERS = 64` (line 26), `WORDS_PER_CORE = 264` (line 175), `build_cb_reconfig_tensor()` (lines 162-218)
- `blaze/ops/matmul/op.py` -- Matmul.emit() CB sizing (lines 48-74)
- `blaze/ops/rmsnorm/op.py` -- RMSNorm.emit() CB sizing (lines 105-139)
- `blaze/ops/dram_streaming_matmul/op.py` -- DRAMStreamingMatmul.emit() subblock_k derivation and triple-buffered weights CB (lines 126-172)

---

< Previous: [Chapter 6 -- Data Format Selection and Automatic Negotiation](../ch06_data_formats/) | Next: [File 02 -- Automatic Page Count and Page Size](./02_automatic_page_count_and_page_size.md) >

← [Chapter 6 -- Data Format Selection and Automatic Negotiation](../ch06_data_formats/) | [Chapter 8 -- Broadcasting and Multi-Dimensional Shape Alignment](../ch08_broadcasting/) →
