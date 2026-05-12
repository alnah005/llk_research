# Data Format Landscape: From PyTorch Dtypes to Tenstorrent Block Floating Point

PyTorch developers think in two precisions: `float32` for training and `bfloat16` for inference. Tenstorrent hardware supports these, but also exposes block-floating-point formats -- `bfloat8_b` and `bfloat4_b` -- that compress weight data by 2x to 4x with minimal accuracy loss. These formats have no PyTorch equivalent. They affect tile page size (and therefore CB memory), compute throughput, and the precision of every downstream operation. This section maps the PyTorch dtype world to TT-Blaze's format world, explains the block-floating-point formats in detail, quantifies per-format trade-offs, shows how format choice cascades into page_size and L1 budgets, and identifies which ops prefer which formats -- using DeepSeek V3 MoE and LLaMA attention as concrete examples.

> *Design Principle P2 (Carry Metadata from Both Worlds Simultaneously):* The ShapeDescriptor must carry the data format alongside shape information, because format directly determines page_size and therefore CB allocation. A shape-only descriptor that ignores format produces wrong page_size values for bfloat8_b and bfloat4_b.

**What you will learn:**

- The complete mapping from PyTorch dtypes (`torch.float32`, `torch.bfloat16`, `torch.float16`, `torch.int8`) to TT-Blaze formats (`ttnn.float32`, `ttnn.bfloat16`, `ttnn.bfloat8_b`, `ttnn.bfloat4_b`, `ttnn.uint8/16/32`)
- What block-floating-point formats (bfp8_b, bfp4_b) actually are: shared exponents, per-block mantissa packing, and why they differ from naive quantization
- Quantified per-format trade-offs: precision (effective dynamic range), memory footprint (bytes per tile), and compute throughput (FPU pipeline width)
- How format affects page_size through `DTYPE_BYTES`, `UNPACKED_DTYPE_BYTES`, and `tile.get_tile_size()` -- and where the simple multiplication formula breaks down
- Which ops prefer which formats, traced through DeepSeek V3 MoE (bfloat4_b expert weights) and LLaMA attention (float32 softmax accumulation)
- Why automatic format selection is needed: the current codebase hardcodes format per op with no cross-op negotiation

---

## PyTorch Dtype World

PyTorch developers choose from a small set of well-understood dtypes:

| PyTorch Dtype | Bits | Exponent | Mantissa | Range | Use Case |
|---------------|------|----------|----------|-------|----------|
| `torch.float32` | 32 | 8 | 23 | ~1e-38 to ~3e38 | Training, accumulation |
| `torch.bfloat16` | 16 | 8 | 7 | ~1e-38 to ~3e38 | Inference activations |
| `torch.float16` | 16 | 5 | 10 | ~6e-8 to 65504 | Inference (limited range) |
| `torch.int8` | 8 | N/A | N/A | -128 to 127 | Quantized inference |
| `torch.int32` | 32 | N/A | N/A | -2B to 2B | Indices, counters |

The critical properties developers rely on:

1. **Per-element independence:** Every element has its own exponent and mantissa. A tensor of `[1e-30, 1e30]` is representable without loss in float32.
2. **IEEE semantics:** Arithmetic follows IEEE 754 rules. NaN, inf, denormals all work as expected.
3. **Uniform precision:** Every element in the tensor has the same number of mantissa bits.
4. **Transparent casting:** `tensor.to(torch.bfloat16)` converts in-place with well-defined rounding.

TT-Blaze's data format world breaks all four expectations: block-floating-point formats share exponents across groups of elements (violating per-element independence and uniform precision), denormals may not be preserved (violating IEEE semantics), and "casting" between BFP and IEEE formats requires a tile-level operation, not a per-element one (violating transparent casting).

---

## TT-Blaze Format World

TT-Blaze exposes a richer format space through two mapping dictionaries and a hardware-aware tile size function:

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

```python
# EXISTING -- from blaze/utils.py
UNPACKED_DTYPE_BYTES: dict = {
    ttnn.bfloat16: 2,
    ttnn.float32: 4,
    ttnn.uint32: 4,
    ttnn.int32: 4,
}
```

### The Full Format Table

| TT-Blaze Format | Bits/Elem | Page Size (32x32 tile) | PyTorch Nearest | Notes |
|-----------------|-----------|----------------------|-----------------|-------|
| `ttnn.float32` | 32 | 4,096 bytes | `torch.float32` | Exact IEEE match |
| `ttnn.bfloat16` | 16 | 2,048 bytes | `torch.bfloat16` | Exact IEEE match; DEFAULT_DTYPE |
| `ttnn.bfloat8_b` | ~8.5 | 1,088 bytes | No equivalent | Block floating point |
| `ttnn.bfloat4_b` | ~4.5 | 576 bytes | No equivalent | Block floating point |
| `ttnn.uint8` | 8 | 1,024 bytes | `torch.uint8` | Integer only |
| `ttnn.uint16` | 16 | 2,048 bytes | `torch.int16` (signed) | Integer only |
| `ttnn.uint32` | 32 | 4,096 bytes | `torch.int32` | Integer only |

> **Warning:** Notice that `bfloat8_b` and `bfloat4_b` are present in `_TTNN_DTYPE_MAP` but absent from both `DTYPE_BYTES` and `UNPACKED_DTYPE_BYTES`. This is because their per-tile sizes cannot be computed from a simple `bytes_per_element * tile_h * tile_w` formula -- the shared exponent overhead makes them non-uniform. The correct page size must be obtained via `tile.get_tile_size(data_format)`.

### The PyTorch-to-TT-Blaze Mapping

| PyTorch Dtype | Default TT-Blaze Format | Lossy? | When to Choose Alternative |
|---------------|------------------------|--------|---------------------------|
| `torch.float32` | `ttnn.float32` | No | Use `ttnn.bfloat16` for non-accumulation paths |
| `torch.bfloat16` | `ttnn.bfloat16` | No | Use `ttnn.bfloat8_b` for weight compression |
| `torch.float16` | `ttnn.bfloat16` | Slight (different mantissa/exponent split) | Direct mapping not supported |
| `torch.int8` | `ttnn.uint8` | Sign handling | Quantized weight storage |
| `torch.int32` | `ttnn.uint32` | No | Index tensors, masks |

The `TileInfo.from_tensor()` method in `fused_program.py` extracts format directly from a device tensor:

```python
# EXISTING -- from blaze/fused_program.py
@staticmethod
def from_tensor(tensor) -> "TileInfo":
    tile = tensor.get_tile()
    data_format = tensor.dtype
    return TileInfo(
        tile=tile,
        data_format=data_format,
        size=tile.get_tile_size(data_format),
        desc=ttnn.TileDescriptor(tile),
    )
```

This is the authoritative page size computation path -- it delegates to `tile.get_tile_size(data_format)`, which handles block floating-point formats correctly.

---

## Block-Floating-Point Formats: bfloat8_b and bfloat4_b

### What They Are

Block floating-point (BFP) formats share a single exponent across a block of mantissa values. Instead of each element carrying its own 8-bit exponent (as in bfloat16), a block of elements shares one exponent, and each element stores only a reduced-precision mantissa.

```text
Standard bfloat16 (per-element exponent):
  Element 0: [sign:1 | exp:8 | mantissa:7]  = 16 bits
  Element 1: [sign:1 | exp:8 | mantissa:7]  = 16 bits
  ...
  Element N: [sign:1 | exp:8 | mantissa:7]  = 16 bits
  Total: N * 16 bits

Block floating point bfloat8_b (shared exponent per block):
  Block header: [shared_exponent:8]           = 8 bits (one per block)
  Element 0: [sign:1 | mantissa:7]            = 8 bits
  Element 1: [sign:1 | mantissa:7]            = 8 bits
  ...
  Element B: [sign:1 | mantissa:7]            = 8 bits
  Total: 8 + B * 8 bits (for a block of B elements)

Block floating point bfloat4_b (shared exponent per block):
  Block header: [shared_exponent:8]           = 8 bits (one per block)
  Element 0: [sign:1 | mantissa:3]            = 4 bits
  Element 1: [sign:1 | mantissa:3]            = 4 bits
  ...
  Element B: [sign:1 | mantissa:3]            = 4 bits
  Total: 8 + B * 4 bits (for a block of B elements)
```

### The "_b" Suffix

The `_b` suffix in `bfloat8_b` and `bfloat4_b` denotes "Block" format, indicating that these are not simple 8-bit or 4-bit per-element formats. This is critical: a naive 8-bit format would have independent exponents per element (like FP8 E4M3 or E5M2). BFP formats sacrifice per-element dynamic range for higher mantissa precision per bit spent.

### Block Size and Tile Structure

On Tenstorrent hardware, the block boundary aligns with the face structure within a tile. A 32x32 tile contains four 16x16 faces. Each face row (16 elements) forms a block for BFP purposes. The shared exponent is the maximum exponent across the 16 elements in the block, and each element's mantissa is right-shifted to align with the shared exponent.

This face-row alignment means a 32x32 tile has 64 face rows (4 faces x 16 rows per face), and therefore 64 shared exponent bytes.

### The Swamping Effect

When elements within a block have very different magnitudes, the smaller elements lose mantissa bits because they are right-shifted to align with the block's shared (maximum) exponent. This is the "swamping effect":

```text
Example: Block with values [1.0, 0.001] in bfloat8_b
  Shared exponent: max_exp(1.0) = 0 (IEEE)
  1.0:    mantissa = 0b1000000 (7 bits)     -- full precision
  0.001:  mantissa = 0b0000000 (shifted out) -- ZERO effective precision

Contrast with bfloat16 (per-element exponent):
  1.0:    exp=127, mantissa=0b0000000         -- full precision
  0.001:  exp=117, mantissa=0b0000110         -- full 7-bit precision
```

In practice, trained neural network weights have relatively uniform magnitudes within each layer's weight matrix rows, so the block-level dynamic range limitation causes minimal accuracy loss -- typically less than 0.1% perplexity degradation for LLaMA-class models. The swamping effect is most dangerous for tensors with high dynamic range (e.g., attention scores after QK^T, logits before softmax), which is why these should remain in bfloat16 or float32.

### Page Size Computation for BFP Formats

The page sizes for bfloat8_b and bfloat4_b include the shared exponent overhead:

```python
# EXISTING -- from blaze/ops/sdpa/op.py
_TILE_SIZES = {
    ttnn.bfloat16: 2048,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 576,
    ttnn.float32: 4096,
}
```

The breakdown for a 32x32 tile:

| Format | Mantissa Bits | Blocks (face rows) | Mantissa Bytes | Exponent Bytes | Total |
|--------|--------------|--------------------|--------------------|----------------|-------|
| `bfloat16` | 16 per elem | N/A | 1024 * 2 = 2,048 | 0 (per-element) | 2,048 |
| `bfloat8_b` | 8 per elem | 64 face rows | 1024 * 1 = 1,024 | 64 * 1 = 64 | 1,088 |
| `bfloat4_b` | 4 per elem | 64 face rows | 1024 * 0.5 = 512 | 64 * 1 = 64 | 576 |
| `float32` | 32 per elem | N/A | 1024 * 4 = 4,096 | 0 (per-element) | 4,096 |

> **Warning:** The simple formula `page_size = DTYPE_BYTES[fmt] * tile_h * tile_w` from `_tile_page_size()` in `cb_engine.py` is correct only for standard formats (bfloat16, float32, uint types). For block-floating-point formats, the formula produces the wrong answer: it would give `1024 * 1 = 1024` for bfloat8_b (missing the 64-byte exponent overhead) and `1024 * 0.5 = 512` for bfloat4_b. The correct path is `tile.get_tile_size(data_format)`.

---

## Per-Format Trade-Offs: Precision, Memory, Throughput

### Precision

| Format | Effective Mantissa Bits | Dynamic Range | Relative Precision |
|--------|------------------------|---------------|-------------------|
| `float32` | 23 | Per-element | 1.0x (reference) |
| `bfloat16` | 7 | Per-element | 1/8192x of float32 |
| `bfloat8_b` | 7 | Per-block (shared exponent) | ~1/8192x, but block-limited |
| `bfloat4_b` | 3 | Per-block (shared exponent) | ~1/8x of bfloat8_b |

The per-block dynamic range limitation is the key distinction from per-element formats.

### Memory

The format choice has a direct multiplicative effect on CB sizing and L1 budget. Consider the weight CB for a DeepSeek V3 MoE expert with `[7168, 2048]` weights distributed across 14 cores:

| Format | Page Size | Tiles per Core | CB Size per Core | L1 Fraction (~1.5 MB) |
|--------|-----------|---------------|-------------------|----------------------|
| `float32` | 4,096 B | 1,024 | 4,096 KB | 266% (does not fit) |
| `bfloat16` | 2,048 B | 1,024 | 2,048 KB | 133% (does not fit) |
| `bfloat8_b` | 1,088 B | 1,024 | 1,088 KB | 71% (tight fit) |
| `bfloat4_b` | 576 B | 1,024 | 576 KB | 37% (comfortable) |

For DeepSeek V3's MoE layer with 256 experts, the format choice determines whether the expert weights fit in L1 at all. Using `bfloat4_b` (576 bytes/tile) instead of `bfloat16` (2,048 bytes/tile) reduces weight CB size by 3.6x -- the difference between a working and a failing kernel.

### Throughput

The FPU pipeline on Tenstorrent hardware processes tiles at a fixed rate per cycle regardless of format. The throughput advantage of smaller formats comes from reduced data movement:

| Format | DRAM Read BW per Tile | NOC Transfer per Tile | L1 Footprint per Tile |
|--------|----------------------|-----------------------|-----------------------|
| `float32` | 4,096 B | 4,096 B | 4,096 B |
| `bfloat16` | 2,048 B | 2,048 B | 2,048 B |
| `bfloat8_b` | 1,088 B | 1,088 B | 1,088 B |
| `bfloat4_b` | 576 B | 576 B | 576 B |

For DRAM-bandwidth-bound ops (large matmuls streaming weights from DRAM), using `bfloat8_b` provides ~47% bandwidth savings vs `bfloat16`, and `bfloat4_b` provides up to 3.6x more effective bandwidth than `bfloat16`. The unpacker hardware converts BFP formats to the FPU's internal representation automatically -- there is no explicit conversion step.

---

## How Format Affects page_size: The Three Computation Paths

### Path 1: DTYPE_BYTES (Standard Formats Only)

The `_tile_page_size()` function in `cb_engine.py` uses the simple multiplication:

```python
# EXISTING -- from blaze/cb_engine.py
def _tile_page_size(dtype: Optional[str], tile_shape: Optional[tuple[int, int]]) -> int:
    dt = dtype or DEFAULT_DTYPE
    ts = tile_shape or DEFAULT_TILE_SHAPE
    if dt not in DTYPE_BYTES:
        raise ValueError(f"Unsupported dtype '{dt}'. Supported: {list(DTYPE_BYTES.keys())}")
    return DTYPE_BYTES[dt] * ts[0] * ts[1]
```

This works for `bfloat16` (2 * 32 * 32 = 2048), `float32` (4 * 32 * 32 = 4096), and integer types. It raises `ValueError` for `bfloat8_b` and `bfloat4_b` because they are not in `DTYPE_BYTES`.

### Path 2: tile.get_tile_size() (All Formats Including BFP)

The `TileInfo.from_tensor()` method uses the hardware-aware path:

```python
# EXISTING -- from blaze/fused_program.py
size=tile.get_tile_size(data_format),
```

This call delegates to the ttnn library, which computes the correct page size for BFP formats (including exponent overhead). This is the path used for tensor-backed CBs created via `cb_from_tensor()`.

### Path 3: UNPACKED_DTYPE_BYTES (Compute-Side Expansion)

When BFP data is read into the FPU, the unpacker expands it to a wider format for computation. The `UNPACKED_DTYPE_BYTES` dictionary tracks the expanded size:

```python
# EXISTING -- from blaze/utils.py
UNPACKED_DTYPE_BYTES: dict = {
    ttnn.bfloat16: 2,
    ttnn.float32: 4,
    ttnn.uint32: 4,
    ttnn.int32: 4,
}
```

Note that `bfloat8_b` and `bfloat4_b` are absent -- their unpacked representation is `bfloat16` (2 bytes), because the unpacker converts them to bfloat16 in the SRCA/SRCB registers before the FPU operates on them. This means:

- **Storage page size:** 1,088 bytes (bfloat8_b) or 576 bytes (bfloat4_b) -- what matters for CB allocation and L1 budget
- **Compute precision:** bfloat16 (7-bit mantissa) or float32 (with `fp32_dest_acc_en`) -- what matters for numerical accuracy
- **Output format:** Whatever the packer is configured for -- typically bfloat16 for the result CB

# PROPOSED

TensorAdapter introduces a `FormatDescriptor` that captures storage, compute, and accumulation precision as a unified triple:

```python
# PROPOSED -- FormatDescriptor
@dataclass(frozen=True)
class FormatDescriptor:
    """A three-level format representation matching the hardware pipeline."""
    storage_format: ttnn.DataType    # What is stored in the CB (bfloat8_b)
    compute_format: ttnn.DataType    # What the FPU sees after unpack (bfloat16)
    accumulate_format: ttnn.DataType # Dest register format (bfloat16 or float32)
    page_size: int                   # Bytes per tile in storage_format
    unpacked_page_size: int          # Bytes per tile if expanded to compute_format
```

This triple (storage/compute/accumulate) captures the reality that a tensor may be stored in bfloat8_b, computed in bfloat16, and accumulated in float32. These three levels are configured independently: `storage_format` is determined by format negotiation, `compute_format` is determined by the unpacker hardware, and `accumulate_format` is determined by the `fp32_dest_acc_en` flag on the precision profile.

The `FormatDescriptor` integrates with `ShapeDescriptor.with_format()` (Chapter 3, File 02) to update `data_format` and `page_size` when format negotiation changes the storage format:

```python
# EXISTING -- from Chapter 3 ShapeDescriptor
def with_format(self, data_format: str) -> "ShapeDescriptor":
    """Return a new ShapeDescriptor with the given data format.
    Recomputes page_size (tile_shape unchanged).
    """
    # Returns new frozen instance with updated data_format and page_size
```

---

## Which Ops Prefer Which Formats

Different ops have different precision requirements and different sensitivities to format choice. The following table is derived from examining each op's emit() method and its interaction with `math_fidelity` and `fp32_dest_acc_en`:

```python
# EXISTING -- from blaze/blaze_op.py (BlazeOp base class)
math_fidelity: str = "HiFi4"
math_approx_mode: bool = False
```

| Op | Preferred Input Format | Preferred Weight Format | Accumulation | Math Fidelity |
|----|----------------------|----------------------|-------------|---------------|
| **Matmul** | bfloat16 | bfloat8_b or bfloat4_b | bfloat16 dest (LoFi) or float32 dest (HiFi) | LoFi for speed, HiFi4 for accuracy |
| **DRAMStreamingMatmul** | bfloat16 | bfloat8_b or bfloat4_b | bfloat16 dest | LoFi (common) |
| **RMSNorm** | bfloat16 | bfloat16 (gamma) | Optional float32 (`fp32_dest_acc_en`) | HiFi4 |
| **Softmax/SDPA** | bfloat16 | N/A | float32 dest (required) | HiFi4 (required for exp) |
| **EltwiseMul** | Same as input | N/A | Same as input | HiFi4 |
| **Mcast** | Pass-through | N/A | N/A | N/A (data movement) |
| **Gather** | Pass-through | N/A | N/A | N/A (data movement) |
| **GatedReduce** | bfloat16 | N/A | bfloat16 | HiFi4 |

### Walkthrough: DeepSeek V3 MoE Expert Format Selection

DeepSeek V3's MoE layer has 256 experts, each with gate/up/down projections. The format selection is critical because all experts share the same L1 budget:

```text
DeepSeek V3 MoE expert:
  hidden_dim = 7168
  expert_intermediate = 2048
  num_experts = 256

Gate/Up projection: [7168, 2048] per expert
Down projection: [2048, 7168] per expert

Format analysis for one expert's gate weights on 14 cores:
  Tiles: (7168/32) * (2048/32) = 224 * 64 = 14,336 total tiles
  Per core: 14,336 / 14 = 1,024 tiles

  bfloat16: 1,024 * 2,048 = 2,097,152 bytes = 2 MB   -- EXCEEDS L1
  bfloat8_b: 1,024 * 1,088 = 1,114,112 bytes = 1.06 MB -- FITS (barely)
  bfloat4_b: 1,024 * 576 = 589,824 bytes = 0.56 MB    -- COMFORTABLE
```

For DeepSeek V3 MoE, `bfloat4_b` is the natural choice for expert weights:
- 3.6x L1 savings vs bfloat16 enables fitting the weight CB plus activation and output CBs
- 4-bit mantissa precision is sufficient for linear projections (weights have bounded dynamic range)
- The activation tensor remains bfloat16 for full precision through the non-linear path

### Walkthrough: LLaMA Attention Format Selection

LLaMA attention has different precision requirements at each stage:

```text
LLaMA-7B attention:
  hidden_dim = 4096
  num_heads = 32
  head_dim = 128

Q/K/V projections (Matmul):
  Weights: bfloat8_b (compression for DRAM streaming)
  Activations: bfloat16
  Accumulation: bfloat16 (LoFi -- speed priority)

Softmax (SDPA):
  Input: bfloat16 scores
  Accumulation: MUST be float32 (fp32_dest_acc_en = True)
  Math: HiFi4 (required for exp() accuracy)
  Output: bfloat16

Output projection (Matmul):
  Weights: bfloat8_b
  Activations: bfloat16
```

The attention block requires three distinct format configurations:
1. **Q/K/V and output projections:** bfloat8_b weights, bfloat16 activations, LoFi math
2. **QK^T and attention-value multiply:** bfloat16 everywhere, LoFi math
3. **Softmax:** bfloat16 input, float32 accumulation, HiFi4 math

Without automatic format negotiation, the developer must manually configure these three configurations and ensure the format transitions between them are handled correctly.

---

## The Current State: Manual Format Selection

# EXISTING

Today, format selection is entirely manual. The developer chooses the format when preparing tensors for the device:

```python
# EXISTING -- typical weight preparation
weight_tensor = ttnn.from_torch(
    weight_data,
    dtype=ttnn.bfloat8_b,          # <-- manual format choice
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=shard_config,
    tile=ttnn.Tile([32, 32]),
)
```

And the format propagates through the `emit()` chain via `cb_from_tensor()`:

```python
# EXISTING -- format propagation in Matmul.emit()
in0_handle = f.cb_from_tensor(in0)    # format from in0's tensor dtype
page_size = in0_handle.page_size       # derived from format + tile
data_format = in0_handle.data_format   # carried forward to output CB
```

The output CB inherits the input's format by default. There is no mechanism to detect format mismatches between producer and consumer, and no way to automatically insert format conversions.

# PROPOSED

TensorAdapter introduces automatic format selection as Step 4 of the `adapt()` pipeline (Chapter 3, File 01), positioned after tile decomposition and before CB sizing. The system draws on three inputs:
1. **Op preference:** Each op declares preferred input/output formats
2. **Precision profile:** A global setting ("performance", "balanced", "accuracy")
3. **L1 budget:** Format affects CB size, which affects what fits in L1

The system is detailed in [File 02: Format Negotiation Protocol](./02_format_negotiation_protocol.md).

---

## Format Selection Impact on L1 Budget: A Quantified Comparison

To ground the discussion in real numbers, here is the L1 budget for a LLaMA-7B gated MLP layer under three format choices, assuming 14 active cores:

```text
Gate/Up matmul: [1, 4096] @ [4096, 11008] (fused gate+up)
  K tiles = 4096/32 = 128
  N tiles = 11008/32 = 344
  Per-core N tiles = 344/14 ~ 25 tiles (with padding)

Input CB (activation): 128 tiles (K dimension for streaming)
Weight CB (direct-address): 128 * 25 = 3,200 tiles
Output CB: 25 tiles
```

| Format Profile | Input CB | Weight CB | Output CB | Total per Core | Fits? |
|---------------|----------|-----------|-----------|----------------|-------|
| **All bfloat16** | 128 * 2,048 = 256 KB | 3,200 * 2,048 = 6,400 KB | 25 * 2,048 = 50 KB | 6,706 KB | No |
| **bfloat8_b weights** | 256 KB | 3,200 * 1,088 = 3,400 KB | 50 KB | 3,706 KB | No |
| **bfloat4_b weights** | 256 KB | 3,200 * 576 = 1,800 KB | 50 KB | 2,106 KB | No |
| **bfloat4_b + streaming** | 2 * 2,048 = 4 KB | Direct-address (shard) | 25 * 2,048 = 50 KB | 54 KB + shard | Yes |

The table illustrates two critical points:
1. Format alone does not solve L1 budget problems -- streaming (small num_pages) and sharding (direct-address weights) are also required
2. Format choice determines the shard size, which determines whether the shard fits alongside other CBs

---

## Interaction with Other Subsystems

### Format and Padding (Chapter 5)

Padding fill values must be representable in the target format. For NEG_INF padding:

| Format | Representation of -inf | Representable? |
|--------|----------------------|----------------|
| `float32` | `0xFF800000` | Exact |
| `bfloat16` | `0xFF80` | Exact |
| `bfloat8_b` | ~`-1e30` (max negative in block) | Approximate but sufficient for exp() |
| `bfloat4_b` | ~`-1e2` (limited range) | May not fully suppress `exp(fill)` |

> **Warning:** When using `bfloat4_b` with NEG_INF padding for softmax, the limited dynamic range (~-100 maximum negative) means `exp(-100) ~ 3.7e-44`, which is not zero. With many padding positions, the leaked probability can be significant. This is why the format negotiation algorithm (File 02) enforces `bfloat16` or wider for softmax inputs, regardless of the active precision profile.

### Format and Tile Decomposition (Chapter 4)

BFP formats require 32x32 tiles -- the face-row block structure assumes 16-element rows within 16x16 faces. Non-standard tile shapes (16x32, 1x32) are not compatible with BFP formats. The `ShapeDescriptor.with_format()` method (Chapter 3, File 02) enforces this by recomputing `page_size` through `tile.get_tile_size(data_format)`.

### Format and CB Sizing (Chapter 7)

CB memory is `num_pages * page_size`. Changing format from bfloat16 to bfloat8_b reduces page_size from 2,048 to 1,088, allowing more pages for the same L1 budget -- or the same number of pages in less L1.

---

## Key Takeaways

- TT-Blaze's format space extends beyond PyTorch's with block-floating-point formats (`bfloat8_b` at 1,088 bytes/tile, `bfloat4_b` at 576 bytes/tile) that share exponents across 16-element face rows within tile faces. These have no PyTorch equivalent.
- Block-floating-point formats provide 1.9x (`bfloat8_b`) to 3.6x (`bfloat4_b`) memory savings over `bfloat16`, enabling weight compression that determines whether large models (DeepSeek V3 MoE with 256 experts) fit in per-core L1 at all.
- Page size computation has three paths: the simple `DTYPE_BYTES * tile_h * tile_w` formula (correct only for standard formats), `tile.get_tile_size(data_format)` (correct for all formats including BFP), and `UNPACKED_DTYPE_BYTES` (compute-side register width, excludes BFP since they unpack to bfloat16). The `_tile_page_size()` function in `cb_engine.py` uses the first path and raises `ValueError` for BFP formats.
- Different ops require different formats: matmul prefers `bfloat8_b`/`bfloat4_b` weights with bfloat16 activations; softmax/SDPA requires float32 accumulation (`fp32_dest_acc_en`); RMSNorm may want float32 accumulation for stability. Without automatic negotiation, the developer must manually configure format at every boundary.
- The current codebase has no cross-op format negotiation. Format is chosen at tensor preparation time (`ttnn.from_torch(..., dtype=...)`) and propagated through `cb_from_tensor()` without validation that the consumer op accepts the producer's format.

## Source Files

- `blaze/cb_engine.py` -- `DTYPE_BYTES`, `_TTNN_DTYPE_MAP`, `_tile_page_size()`, `DEFAULT_DTYPE`
- `blaze/utils.py` -- `UNPACKED_DTYPE_BYTES`, `get_element_size_bytes()`, `compute_subblock_w()`
- `blaze/fused_program.py` -- `TileInfo.from_tensor()` (authoritative page size via `tile.get_tile_size()`), `FusedProgram.__init__()` with `math_fidelity`, `math_approx_mode`, `fp32_dest_acc_en`
- `blaze/ops/sdpa/op.py` -- `_TILE_SIZES` constant mapping formats to per-tile byte sizes
- `blaze/blaze_op.py` -- `BlazeOp.math_fidelity`, `BlazeOp.math_approx_mode`

---

← Previous: [Chapter 5 — Padding](../ch05_padding/) | Next: [Chapter 7 — Automatic CB Sizing and L1 Memory Budgeting](../ch07_cb_sizing/) →

< Previous: [Chapter 5](../ch05_padding/) | Next: [File 02 -- Format Negotiation Protocol](./02_format_negotiation_protocol.md) >
