# Tenstorrent Tile and Circular Buffer Model

Where PyTorch lets the developer think in arbitrary shapes and implicit memory management, Tenstorrent hardware requires data to be decomposed into fixed-geometry tiles, staged through explicitly sized circular buffers in per-core L1 SRAM, and formatted according to hardware datapaths. Every concept introduced here represents a decision that is invisible in PyTorch but mandatory in TT-Blaze.

**What you will learn:**

- Tenstorrent hardware at the conceptual level: cores, L1 memory, and the NOC fabric
- What circular buffers are: ring-buffered L1 regions with `cb_reserve_back`/`cb_push_back`/`cb_wait_front`/`cb_pop_front` semantics
- Tile geometry: 32x32 default tiles, 1x32 row tiles, and face-order memory layout within tiles
- Why tiles must be padded to 32-element boundaries
- The CBHandle dataclass field by field: `cb_id`, `num_pages`, `page_size`, `data_format`, `tile_desc`, `core_ranges`, `backing_tensor`, `access_mode`
- Data formats: bfloat16, bfloat8_b, bfloat4_b, float32 -- packed vs unpacked tile sizes

---

## Tenstorrent Hardware at the Conceptual Level

A Tenstorrent chip (Blackhole, for the current generation) contains a grid of **Tensix cores** connected by a Network-on-Chip (NOC). Each core is an independent processor with its own local memory and compute units:

```
+------------------------------------------------------------------+
|                      Tenstorrent Chip (Blackhole)                |
|                                                                  |
|   +----------+  +----------+  +----------+  +----------+        |
|   | Tensix   |  | Tensix   |  | Tensix   |  | Tensix   |  ...   |
|   | Core 0,0 |  | Core 1,0 |  | Core 2,0 |  | Core 3,0 |        |
|   |          |  |          |  |          |  |          |        |
|   | +------+ |  | +------+ |  | +------+ |  | +------+ |        |
|   | | L1   | |  | | L1   | |  | | L1   | |  | | L1   | |        |
|   | |~1.5MB| |  | |~1.5MB| |  | |~1.5MB| |  | |~1.5MB| |        |
|   | +------+ |  | +------+ |  | +------+ |  | +------+ |        |
|   | BRISC    |  | BRISC    |  | BRISC    |  | BRISC    |        |
|   | NCRISC   |  | NCRISC   |  | NCRISC   |  | NCRISC   |        |
|   | TRISC    |  | TRISC    |  | TRISC    |  | TRISC    |        |
|   | FPU/SFPU |  | FPU/SFPU |  | FPU/SFPU |  | FPU/SFPU |        |
|   +----------+  +----------+  +----------+  +----------+        |
|        |              |              |              |             |
|   ====== NOC0 / NOC1 (mesh interconnect) ========================|
|        |              |              |              |             |
|   +----------+  +----------+  +----------+  +----------+        |
|   | DRAM     |  | DRAM     |  | DRAM     |  | DRAM     |        |
|   | Bank 0   |  | Bank 1   |  | Bank 2   |  | Bank 3   |        |
|   +----------+  +----------+  +----------+  +----------+        |
+------------------------------------------------------------------+
```

Each Tensix core contains:

| Component | Role |
|-----------|------|
| **L1 SRAM** | ~1.5 MB per core on Blackhole. Holds circular buffers, kernel text, stack, and semaphores. This is the only memory the compute units can access directly. |
| **BRISC** | Baby RISC-V processor. Handles data movement coordination on NOC 0. |
| **NCRISC** | Network-connected RISC-V. Manages NOC reads/writes on NOC 1, DRAM streaming, and inter-core data transfers. |
| **TRISC** | Three-stage RISC-V pipeline: TRISC0 (unpack), TRISC1 (math), TRISC2 (pack). Configures and drives the math engine. |
| **FPU/SFPU** | The compute datapath. Operates on tiles in face-order format. Fixed tile dimensions are a hardware constraint of this unit. |
| **NOC** | Two independent networks (NOC0, NOC1) connecting all cores and DRAM banks. Used for multicast, unicast, gather, and DRAM streaming. |

The critical difference from a GPU is that **there is no shared global memory accessible by all cores simultaneously**. Each core has its own L1, and data must be explicitly moved between cores via the NOC. The circular buffer is the mechanism that mediates data flow between the RISC processors within a core and between cores.

---

## What Circular Buffers Are

A circular buffer (CB) is a ring-buffered region in a core's L1 SRAM. It is the fundamental data-passing mechanism on Tenstorrent hardware. The CB provides producer-consumer coordination between the RISC processors that share a core.

The typical data flow within a single core follows this pattern:

```
    NCRISC (producer)              TRISC (consumer)
    +-------------+               +-------------+
    | Read from   |               | Wait for    |
    | DRAM/NOC    |               | data ready  |
    |             |               |             |
    | cb_reserve  |               | cb_wait     |
    |  _back()    |               |  _front()   |
    |             |               |             |
    | Write data  |               | Process     |
    | to CB page  |               | tile data   |
    |             |               |             |
    | cb_push     |               | cb_pop      |
    |  _back()    |               |  _front()   |
    +------+------+               +------+------+
           |                              |
           v                              v
    +--------------------------------------+
    |       Circular Buffer in L1          |
    |                                      |
    |  +------+ +------+ +------+          |
    |  |Page 0| |Page 1| |Page 2| ...      |
    |  |(tile)| |(tile)| |(tile)|          |
    |  +------+ +------+ +------+          |
    |                                      |
    |  write_ptr -->        <-- read_ptr   |
    +--------------------------------------+
```

### The Four CB Operations

| Operation | Who Calls It | What It Does |
|-----------|-------------|--------------|
| `cb_reserve_back(cb_id, n)` | Producer (NCRISC or TRISC) | Block until `n` pages are available for writing. |
| `cb_push_back(cb_id, n)` | Producer | Advance the write pointer by `n` pages, signaling data is ready. |
| `cb_wait_front(cb_id, n)` | Consumer (TRISC or NCRISC) | Block until `n` pages are available for reading. |
| `cb_pop_front(cb_id, n)` | Consumer | Advance the read pointer by `n` pages, freeing space for the producer. |

This is a hardware-level FIFO protocol. The producer and consumer can operate at different rates -- the CB provides backpressure (reserve blocks when full) and flow control (wait blocks when empty). This enables **double-buffering**: while TRISC computes on one tile, NCRISC can be filling the next tile slot. The ring structure means no memory is copied -- the producer and consumer simply advance their read/write pointers.

Each CB's total L1 footprint is:

$$\text{CB size (bytes)} = \texttt{num\_pages} \times \texttt{page\_size}$$

Since all CBs for a core share the same ~1.5 MB of L1, the total of all CB sizes plus kernel text, stack, and semaphores must fit within L1. There is no virtual memory and no swapping -- overcommitting L1 causes silent data corruption or hardware hangs.

### CB Access Modes

TT-Blaze defines two access modes for CBs, captured in the `CBAccessMode` enum from `blaze/cb_handle.py`:

```python
# EXISTING -- from blaze/cb_handle.py
class CBAccessMode(str, Enum):
    FIFO = "fifo"
    DIRECT_ADDRESS = "direct_address"
```

- **FIFO mode** (`CBAccessMode.FIFO`): Standard ring-buffer semantics. The consumer calls `cb_wait_front`/`cb_pop_front`. This is the default for streaming data (activations flowing through an op chain).
- **DIRECT_ADDRESS mode** (`CBAccessMode.DIRECT_ADDRESS`): The CB points to a known L1 address (typically a pre-sharded weight tensor). The consumer reads directly from that address rather than using FIFO semantics. This avoids copying weight data into a separate CB region. Direct-address mode requires a `backing_tensor`.

---

## Tile Geometry

All data flowing through the Tensix compute engine is organized into **tiles** -- fixed-size, two-dimensional blocks of elements. The hardware's FPU and SFPU units are hardwired to process data in tile-sized chunks.

### The 32x32 Default Tile

The standard tile shape is **(32, 32)** -- 32 rows by 32 columns, containing 1,024 elements:

```python
# EXISTING -- from blaze/cb_engine.py
DEFAULT_TILE_SHAPE = (32, 32)
```

A 32x32 tile of bfloat16 data occupies:

$$\text{page\_size} = 32 \times 32 \times 2 = 2{,}048 \text{ bytes}$$

### Face-Order Memory Layout

Within a 32x32 tile, data is **not** stored in simple row-major order. The tile is subdivided into four **faces** -- 16x16 blocks of 256 elements each. The faces are stored sequentially in memory in what is called **face order**:

```
Tile layout in memory (face order):

         Col 0..15     Col 16..31
       +------------+------------+
Row    |            |            |
 0..15 |     F0     |     F1     |
       |            |            |
       +------------+------------+
Row    |            |            |
16..31 |     F2     |     F3     |
       |            |            |
       +------------+------------+

Memory: [F0: 256 elems] [F1: 256 elems] [F2: 256 elems] [F3: 256 elems]
```

Within each face, elements are stored row-major: all 16 elements of face-row 0, then face-row 1, through face-row 15. This means element (0, 16) in the tile is at memory offset 256 (start of F1), not at offset 16 as it would be in a standard row-major layout.

This face-order layout is a hardware constraint of the FPU and SFPU datapaths. The unpacker reads data in face order, the FPU computes on face-sized 16x16 operands, and the packer writes results in face order. Converting between PyTorch's row-major layout and the face-order tile layout is handled by the `ttnn.TILE_LAYOUT` transformation when data is moved to the device.

### Row Tiles (1x32)

For operations like RMSNorm that process vectors rather than matrices, TT-Blaze supports **(1, 32)** row tiles -- a single row of 32 elements. These are used when data arrives from memory in a flat 1D layout and must be reinterpreted as standard tile geometry for the FPU.

The `interpret_tile` utility in `blaze/utils.py` selects between a full 32x32 tile and a half 16x32 tile based on whether the width divides evenly by the full tile size:

```python
# EXISTING -- from blaze/utils.py
def interpret_tile(width: int) -> tuple[ttnn.Tile, int]:
    """Choose the interpreted tile geometry for a given element width.

    Data is stored in L1 as contiguous bfloat16 values (1x32 tiles), but the
    compute LLK (mul_reduce_scalar, etc.) requires standard tile geometry
    (faces of 16x16).  Returns (tile, num_tiles).
    """
    FULL = ttnn.Tile((32, 32))
    HALF = ttnn.Tile((16, 32))
    use_half = (width // FULL.tile_shape[1]) % FULL.tile_shape[0] != 0
    tile = HALF if use_half else FULL
    th, tw = tile.tile_shape
    return tile, width // (th * tw)
```

The function returns both the tile geometry and the number of tiles needed to represent the given width. The caller must use the returned tile shape when configuring CBs -- mismatches between the tile geometry declared on a CB and what the compute kernel expects will produce silent data corruption.

The general tile shapes supported by the hardware include:

| Tile Shape | Face Layout | Use Case |
|-----------|-------------|----------|
| (32, 32) | 2x2 of 16x16 | Matmul, standard compute |
| (16, 32) | 1x2 of 16x16 | RMSNorm reinterpretation for non-aligned widths |
| (1, 32) | Row storage | Weight/activation storage in L1 |

### Why Tiles Must Be Padded

The hardware datapaths operate on fixed tile dimensions. If a tensor has shape `[1, 37]` -- 37 elements -- it cannot be processed as-is. The last 5 elements do not fill a complete 32-element row, and the FPU has no mechanism to process partial tiles.

The solution is **padding**: the tensor is extended to the next tile boundary:

$$\text{padded\_dim} = \lceil \text{dim} / \text{tile\_dim} \rceil \times \text{tile\_dim}$$

For `[1, 37]` with 32x32 tiles:
- Height: $\lceil 1/32 \rceil \times 32 = 32$ (padded from 1 to 32)
- Width: $\lceil 37/32 \rceil \times 32 = 64$ (padded from 37 to 64)
- Result: 2 tiles of 32x32

The padded positions must be filled with appropriate values. The choice of padding fill value is **op-specific**, and getting it wrong produces silently incorrect results:

| Operation | Required padding value | Reason |
|-----------|----------------------|--------|
| Matmul | 0 | Padding elements contribute zero to the dot product |
| Softmax | $-\infty$ | Padding positions must produce zero attention weight after $e^x$ |
| RMSNorm | 0 (with correction) | Padding zeros affect the sum-of-squares; the scalar must use the original (unpadded) width |
| Mean reduction | Excluded via mask | Padding zeros would dilute the mean; a mask or corrected denominator is needed |

The `interpret_tile_padded` function in `blaze/utils.py` handles this padding calculation:

```python
# EXISTING -- from blaze/utils.py
def interpret_tile_padded(width: int) -> tuple[ttnn.Tile, int, int]:
    """Like interpret_tile but pads to the next tile-aligned boundary.

    Returns (tile, num_tiles, padded_width). When width is already
    tile-aligned, padded_width == width.
    """
    FULL = ttnn.Tile((32, 32))
    HALF = ttnn.Tile((16, 32))
    full_sz = FULL.tile_shape[0] * FULL.tile_shape[1]  # 1024
    half_sz = HALF.tile_shape[0] * HALF.tile_shape[1]  # 512
    MAX_TILES = 8  # mul_reduce_scalar_tile dest register limit

    if width % full_sz == 0 or width % half_sz == 0:
        tile, num_tiles = interpret_tile(width)
        return tile, num_tiles, width

    # Try HALF-tile padding first.
    padded_half = round_up(width, half_sz)
    if padded_half // half_sz <= MAX_TILES:
        return HALF, padded_half // half_sz, padded_half

    # Too many HALF tiles -- pad to next FULL-tile boundary.
    padded_full = round_up(width, full_sz)
    return FULL, padded_full // full_sz, padded_full
```

This function first tries half-tile (16x32) alignment, falling back to full-tile (32x32) when the number of half-tiles would exceed the hardware dest register limit of 8 tiles (a capacity constraint for specific SFPU reduction operations like `mul_reduce_scalar_tile`).

> **Warning:** PyTorch tensors have no concept of tile padding. A PyTorch tensor of shape `[1, 37]` is exactly 37 elements wide. When this tensor enters the Tenstorrent execution model, padding must be inserted explicitly, tracked through the op chain, and stripped on output. There is no automatic mechanism for this in the current TT-Blaze API.

---

## CBHandle: The Complete Field Reference

The `CBHandle` dataclass (from `blaze/cb_handle.py`) is the Python-side representation of a configured circular buffer. It is the return type of every `emit()` call and the input to every downstream op. Here is the complete dataclass:

```python
# EXISTING -- from blaze/cb_handle.py
@dataclass(eq=False)
class CBHandle:
    cb_id: int
    num_pages: int
    page_size: int
    core_ranges: ttnn.CoreRangeSet
    data_format: object
    tile_desc: object
    byte_offset: int = 0
    backing_tensor: object = None
    access_mode: CBAccessMode = CBAccessMode.FIFO
    _node_id: str = ""
    _port_name: str = "out"
    scratch_name: str = ""
```

The `@dataclass(eq=False)` is deliberate: two CBHandles with the same `cb_id` should not compare as equal if they represent different views (e.g., after disjoint-core sharing gives both callers handles with the same `cb_id` but different `core_ranges`). Identity semantics prevent accidental deduplication.

Each field maps to a specific hardware or composition concern:

### Primary Fields

| Field | Type | What It Controls |
|-------|------|-----------------|
| `cb_id` | `int` | Hardware CB slot index (0--63). Selects which register bank on the Tensix core is programmed. |
| `num_pages` | `int` | Number of tile-sized pages the CB can hold simultaneously. Controls buffering depth -- more pages enables double-buffering or full accumulation. |
| `page_size` | `int` | Bytes per page. Computed as `tile_h * tile_w * dtype_bytes` for unpacked formats. For a 32x32 bfloat16 tile: $32 \times 32 \times 2 = 2048$ bytes. |
| `data_format` | `ttnn.DataType` | How tile bytes are interpreted by the unpack hardware. Determines precision and byte layout. |
| `tile_desc` | `ttnn.TileDescriptor` | Encodes tile height and width (e.g., 32x32). The unpacker and packer use this to know face layout. |
| `core_ranges` | `ttnn.CoreRangeSet` | Which physical Tensix cores have this CB configured. Different ops may use different core subsets. |

### Backing and Access Fields

| Field | Default | Purpose |
|-------|---------|---------|
| `backing_tensor` | `None` | The `ttnn.Tensor` whose L1 buffer backs this CB. Set for tensor-backed CBs (external inputs, outputs). `None` for pure scratch CBs. Required for `DIRECT_ADDRESS` mode. |
| `byte_offset` | `0` | Offset within the backing tensor's L1 allocation. Non-zero when multiple logical CBs share one physical fused tensor buffer (overlapped views). |
| `access_mode` | `FIFO` | `FIFO` for standard `cb_wait_front`/`cb_pop_front` consumption. `DIRECT_ADDRESS` for cases where the consumer reads by explicit L1 address (used for shared weight buffers). |

### Metadata Fields

| Field | Default | Purpose |
|-------|---------|---------|
| `_node_id` | `""` | Shadow graph node ID that produced this handle. Used for automatic graph construction. |
| `_port_name` | `"out"` | Output port name on the producing node. |
| `scratch_name` | `""` | Symbolic name for scratch CBs, used for scratch-mapping lookup. |

The direct-address mode validates its requirements at construction time:

```python
# EXISTING -- from blaze/cb_handle.py
def __post_init__(self) -> None:
    if not isinstance(self.access_mode, CBAccessMode):
        self.access_mode = CBAccessMode(self.access_mode)
    if self.is_direct_address and self.backing_tensor is None:
        raise ValueError("Direct-address CBHandle requires a backing_tensor.")
```

CBHandle supports integer coercion -- `int(handle)` returns the `cb_id`. This allows handles to be passed directly into CT arg tuples:

```python
# EXISTING -- usage pattern from blaze/ops/matmul/op.py
f.trisc_ct_args([
    (f"{prefix}.in1", in1_cb),         # CBHandle auto-coerced to cb_id
    (f"{prefix}.out", out),            # CBHandle auto-coerced to cb_id
    (f"{prefix}.k_num_tiles", k_num_tiles),
    (f"{prefix}.out_w_per_core", out_w_per_core),
])
```

The total L1 memory consumed by a CB is:

$$\text{CB\_bytes} = \text{num\_pages} \times \text{page\_size}$$

For a matmul with $K = 128$ tiles of 32x32 bfloat16 data, one input CB consumes $128 \times 2048 = 262{,}144$ bytes (256 KB) -- a significant fraction of the ~1.5 MB available per core.

---

## Data Formats and Tile Sizes

Tenstorrent hardware supports several data formats that differ from standard PyTorch dtypes. The format determines both the byte size per element and the tile size in memory.

### Format Byte Sizes

TT-Blaze defines two byte-size mappings. The first is in `cb_engine.py` for graph-API compilation:

```python
# EXISTING -- from blaze/cb_engine.py
DTYPE_BYTES: dict[str, int] = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
}
```

The second is in `utils.py` for composition-API use with ttnn DataType objects:

```python
# EXISTING -- from blaze/utils.py
UNPACKED_DTYPE_BYTES: dict = {
    ttnn.bfloat16: 2,
    ttnn.float32: 4,
    ttnn.uint32: 4,
    ttnn.int32: 4,
}
```

The CBEngine also provides a format-string-to-ttnn-object mapping:

```python
# EXISTING -- from blaze/cb_engine.py
_TTNN_DTYPE_MAP = {
    "bfloat16": ttnn.bfloat16,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
    "float32": ttnn.float32,
    "uint8": ttnn.uint8,
    "uint16": ttnn.uint16,
    "uint32": ttnn.uint32,
}
```

### Per-Tile Sizes by Format

For a 32x32 tile, the page size varies by format:

| Format | Bytes/Element | Page Size (32x32) | Description |
|--------|--------------|-------------------|-------------|
| `float32` | 4 | 4,096 bytes | Full 32-bit IEEE float. Used for accumulation (`fp32_dest_acc_en`). |
| `bfloat16` | 2 | 2,048 bytes | Standard 16-bit brain float. The default format. |
| `bfloat8_b` | ~1.06 | 1,088 bytes | Block floating point: 8-bit mantissa with shared exponent per block. Excellent for weight compression. |
| `bfloat4_b` | ~0.56 | 576 bytes | Block floating point: 4-bit mantissa with shared exponent per block. Maximum weight compression. |
| `uint8` | 1 | 1,024 bytes | Unsigned 8-bit integer. |
| `uint16` | 2 | 2,048 bytes | Unsigned 16-bit integer. |
| `uint32` | 4 | 4,096 bytes | Unsigned 32-bit integer. |

The SDPA op in `blaze/ops/sdpa/op.py` explicitly defines these tile sizes as a reference:

```python
# EXISTING -- from blaze/ops/sdpa/op.py
_TILE_SIZES = {
    ttnn.bfloat16: 2048,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 576,
    ttnn.float32: 4096,
}
```

> **Warning:** `bfloat8_b` and `bfloat4_b` are **block floating point** formats, not simple reduced-precision types. A block of elements shares a single exponent, giving better dynamic range than naive 8-bit or 4-bit quantization. Their per-tile sizes include the shared exponent overhead, which is why `bfloat8_b` is 1,088 bytes (not $1024 \times 1 = 1024$) and `bfloat4_b` is 576 bytes (not $1024 \times 0.5 = 512$). The `page_size` for packed formats cannot be computed from `DTYPE_BYTES * tile_h * tile_w`; instead, use `tile.get_tile_size(data_format)`.

### Page Size Computation

The page size calculation is straightforward for unpacked formats:

```python
# EXISTING -- from blaze/cb_engine.py
def _tile_page_size(dtype: Optional[str], tile_shape: Optional[tuple[int, int]]) -> int:
    dt = dtype or DEFAULT_DTYPE
    ts = tile_shape or DEFAULT_TILE_SHAPE
    if dt not in DTYPE_BYTES:
        raise ValueError(f"Unsupported dtype '{dt}'. Supported: {list(DTYPE_BYTES.keys())}")
    return DTYPE_BYTES[dt] * ts[0] * ts[1]
```

For packed block floating-point formats (`bfloat8_b`, `bfloat4_b`), the page size is computed by `ttnn.Tile.get_tile_size(dtype)`, which accounts for the shared exponent overhead. This asymmetry -- some formats use a simple multiplication while others need special handling -- is one more detail that an abstraction layer should hide.

### Format Choice Impacts L1 Budget

The format choice has a direct multiplicative effect on CB memory consumption. For a matmul weight CB holding $K \times N / \text{num\_cores}$ tiles:

| Format | Page Size | CB for 128 tiles | L1 fraction (~1.5 MB) |
|--------|-----------|-------------------|----------------------|
| `float32` | 4,096 B | 512 KB | 34% |
| `bfloat16` | 2,048 B | 256 KB | 17% |
| `bfloat8_b` | 1,088 B | 136 KB | 9% |
| `bfloat4_b` | 576 B | 72 KB | 5% |

Using `bfloat8_b` instead of `bfloat16` for weights nearly halves the L1 footprint, allowing more tiles to be buffered (improving compute utilization) or freeing space for other CBs in a fused op.

### Format and Op Preferences

Different ops prefer different formats:

| Op Type | Preferred Input Format | Preferred Weight Format | Accumulation |
|---------|----------------------|----------------------|-------------|
| Matmul | bfloat16 | bfloat8_b or bfloat4_b | LoFi (bfloat16 dest) |
| RMSNorm | bfloat16 | bfloat16 | Optional float32 (`fp32_dest_acc_en`) |
| Softmax/SDPA | bfloat16 | N/A | HiFi4 (float32 dest) |
| Element-wise | Same as input | N/A | Same as input |

These format preferences are currently hardcoded per op -- there is no automatic negotiation when ops are chained.

---

## Putting It Together: A Single Tile Through the Pipeline

To illustrate the full data path, consider one tile of activation data flowing through a matmul:

1. **DRAM to L1**: NCRISC reads a 32x32 bfloat16 tile (2,048 bytes) from DRAM into CB slot 0 on core (0,0). Calls `cb_push_back(0, 1)`.

2. **L1 CB to Unpack**: TRISC0 calls `cb_wait_front(0, 1)`, reads the tile from CB 0, and unpacks it from bfloat16 to the FPU's internal format.

3. **Compute**: TRISC1 multiplies the activation tile with a weight tile (from CB 1, direct-address mode for sharded weights), accumulating into a dest register.

4. **Pack to Output**: TRISC2 packs the result from the dest register into bfloat16 and writes to CB 2 (the output CB). Calls `cb_push_back(2, 1)`.

5. **L1 to DRAM or downstream**: NCRISC calls `cb_wait_front(2, 1)` and either writes the result to DRAM or the data stays in L1 for the next fused op.

Every step requires that the CB be configured with the correct `cb_id`, `num_pages`, `page_size`, `data_format`, and `tile_desc`. Get any one wrong, and the hardware reads garbage, computes wrong results, or hangs waiting for a push/pop that never comes.

---

## The 64 CB Slot Hardware Limit

Each Tensix core supports a maximum of **64 circular buffer slots** (cb_id 0 through 63):

```python
# EXISTING -- from blaze/cb_engine.py
MAX_CB_ID = 64
```

A single fused op may need dozens of CBs: input activations, multiple weight tensors, intermediate scratch buffers, output buffers, and internal accumulators. For a complex fused op like a Mixture-of-Experts (MoE) layer, approaching or exceeding 64 CBs is a real concern.

TT-Blaze provides `CBEngine.compact_cb_ids()`, which uses interval coloring to reuse CB IDs across non-overlapping lifetimes, and `CircularBufferIdManager` (from `blaze/cb_reconfig.py`), which manages CB IDs across multi-phase programs:

```python
# EXISTING -- from blaze/cb_reconfig.py
class CircularBufferIdManager:
    NUM_CIRCULAR_BUFFERS = 64

    def __init__(self):
        self._id_to_format: dict[int, tuple] = {}
        self._next_id = 0
```

The hardware limit is hard: exceeding it produces a runtime crash, and Metal's allocator does not bounds-check at compile time. The CBEngine warns when approaching the limit:

```python
# EXISTING -- from blaze/cb_engine.py
if next_cb_id > self.max_cb_id - 8:
    warnings.warn(
        f"CB usage is high: {next_cb_id}/{self.max_cb_id} CBs assigned. "
        f"Approaching Blackhole hardware limit.",
        stacklevel=2,
    )
```

---

## Key Takeaways

- Tenstorrent hardware is a grid of independent Tensix cores, each with ~1.5 MB of L1 SRAM. There is no shared global memory; data moves between cores via the NOC.
- Circular buffers are ring-buffered L1 regions that provide flow control between RISC processors via `cb_reserve_back`/`cb_push_back`/`cb_wait_front`/`cb_pop_front`. Each core supports up to 64 CB slots.
- Tiles are the atomic unit of data: 32x32 elements by default, stored in face order (four 16x16 faces). The FPU operates on faces, not arbitrary-shaped arrays.
- Non-tile-aligned tensor dimensions must be padded to tile boundaries. The padding fill value is op-specific (zero for matmul, negative infinity for softmax), and getting it wrong corrupts results silently.
- CBHandle packages all CB metadata -- `cb_id`, `num_pages`, `page_size`, `data_format`, `tile_desc`, `core_ranges`, `access_mode` -- each representing a decision that PyTorch handles implicitly.
- Data formats include standard types (bfloat16, float32) and Tenstorrent-specific block floating point (bfloat8_b at 1,088 bytes/tile, bfloat4_b at 576 bytes/tile). Format affects tile size, compute precision, and L1 footprint. There is currently no automatic format selection or negotiation.

## Source Files

- `blaze/cb_handle.py` -- CBHandle dataclass, CBAccessMode enum, `require_fifo_handle()` guard
- `blaze/cb_engine.py` -- DTYPE_BYTES, DEFAULT_TILE_SHAPE, MAX_CB_ID, `_tile_page_size()`, `_TTNN_DTYPE_MAP`, CBEngine
- `blaze/cb_reconfig.py` -- CircularBufferIdManager, CB ID allocation across multi-phase programs
- `blaze/utils.py` -- UNPACKED_DTYPE_BYTES, `interpret_tile()`, `interpret_tile_padded()`, `round_up()`, `compute_subblock_w()`
- `blaze/ops/sdpa/op.py` -- `_TILE_SIZES` constant mapping formats to per-tile byte sizes

---

**Next:** [`03_the_manual_plumbing_burden.md`](./03_the_manual_plumbing_burden.md)
