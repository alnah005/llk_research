# Padding Strategy Taxonomy: Why Every Op Needs Its Own Fill Value

Padding is not a cosmetic detail -- it is a correctness requirement that determines whether a model produces the right answer or a subtly wrong one. When a tensor's logical shape does not align to tile boundaries, the extra positions must be filled with values that are mathematically neutral for the consuming operation. Zero is not always neutral. This section defines the five padding strategies required by TT-Blaze's op catalog, explains the mathematical reason each strategy exists, demonstrates the numerical consequences of choosing the wrong one, and traces the current manual approach through PaddedRMSNorm and SDPA to show why automation is needed.

> *Design Principle P6 (Per-Operation Padding Semantics):* Padding fill values are determined by the consuming operation, not by the producing tensor. A zero-padded activation is correct for matmul but catastrophically wrong for softmax. TensorAdapter's padding system must select fill values based on op semantics, not tensor properties.

**What you will learn:**

- The five padding strategies (ZERO, NEG_INF, MASK, IDENTITY, NONE) and the precise mathematical invariant each preserves
- Worked numerical examples showing the output corruption when the wrong strategy is used -- including quantified error rates for softmax (probability leak), mean reduction, and RMSNorm
- The "consumer-determines-padding" principle: why the consuming op determines the fill value, not the producing tensor
- How PaddedRMSNorm (`blaze/ops/padded_rmsnorm/op.py`) currently handles padding manually with explicit `padded_input`/`padded_gamma` Internal CBs, six CT args, and two RISC pipelines
- How SDPA (`blaze/ops/sdpa/op.py`) uses causal masking constants (`identity_scalar_packed = 0x3F803F80`, `zero_scalar_packed = 0x00000000`) that interact with padding
- Why a unified, automatic padding system eliminates an entire class of silent correctness bugs

---

## The Five Padding Strategies

### Overview Table

| Strategy | Enum Value | Fill Value | Mathematical Invariant | Primary Ops |
|----------|-----------|------------|----------------------|-------------|
| **Zero-padding** | `ZERO` | `0.0` | Additive identity: `x + 0 = x` | Matmul, linear, elementwise add, RMSNorm |
| **Negative-infinity padding** | `NEG_INF` | `-inf` | Softmax identity: `exp(-inf) = 0` | Softmax, SDPA, log-softmax |
| **Mask-based padding** | `MASK` | Tracked mask | Exclusion: padded elements excluded from aggregation | Mean reduction, LayerNorm, attention masking |
| **Identity-element padding** | `IDENTITY` | Op-dependent | Reduction identity: `op(x, e) = x` | Multiply-reduction (1.0), max-reduction (-inf), min-reduction (+inf) |
| **No padding** | `NONE` | N/A | Shape is already tile-aligned | Tile-aligned tensors (4096, 7168, 2048, etc.) |

> **Warning:** Four of these five strategies produce identical results when dimensions happen to be tile-aligned (the common case for production LLM shapes like 4096 and 7168). The bugs only surface when a developer uses a non-aligned dimension -- for example, a custom model with hidden_dim=768 (GPT-2) or an attention sequence length of 37 tokens. This makes padding bugs particularly insidious: they pass all tests on standard shapes and fail silently on edge cases.

---

### Strategy 1: Zero-Padding (ZERO)

**Fill value:** `0.0`

**Mathematical basis:** Zero is the additive identity. For any operation whose result depends on a sum of input elements, padding with zero ensures that padded positions contribute nothing:

- **Matmul:** `sum(a[i] * b[i])` is unaffected by positions where `a[i] = 0`, because `0 * b[i] = 0`.
- **RMSNorm denominator:** Padded positions contribute `0^2 = 0` to the sum of squares (correct), but the denominator must use the logical element count (not the padded count).
- **Elementwise add / residual add:** `x[i] + 0 = x[i]`.

```
Matmul walkthrough: [1, 37] @ [37, 64]

Logical activation:  [1, 37]   -- 37 real elements
Padded activation:   [32, 64]  -- tile-aligned with (32, 32) tiles
Fill value:          0.0

Tile layout (first row):
  [x0, x1, ..., x36, 0.0, 0.0, ..., 0.0]   <- 37 real + 27 zeros

Inner product for output[0][j]:
  sum(activation[0][k] * weight[k][j] for k in range(64))
  = sum(activation[0][k] * weight[k][j] for k in range(37))  -- real data
  + sum(0.0 * weight[k][j] for k in range(37, 64))           -- zeros
  = correct_result + 0.0
  = correct_result
```

**When zero-padding fails:**

| Operation | Zero-Padding Effect | Error Magnitude | Correct Strategy |
|-----------|--------------------|----------------|-----------------|
| Softmax | `exp(0) = 1` -- padded positions get nonzero probability | 71.2% probability leak (3 real + 29 padded) | NEG_INF |
| Mean reduction | Zero inflates denominator | 90.6% underestimate (divides by 32 instead of 3) | MASK |
| RMSNorm | Wrong element count in denominator | 31.5% RMS error, up to 153% per-element error | ZERO + count correction |
| Max reduction | `max(..., 0)` may be incorrect when all values are negative | Arbitrary (depends on data) | IDENTITY (-inf) |
| Multiply reduction | `prod(..., 0) = 0` -- zeroes out the entire product | 100% error (result is 0) | IDENTITY (1.0) |

# EXISTING

The current TT-Blaze codebase uses zero-padding as the implicit default. When `FusedProgram.cb_scratch()` allocates a CB, L1 memory is not initialized -- its content is whatever the previous program left behind. For tensor-backed CBs created via `FusedProgram.cb_from_tensor()`, the host pads the tensor to tile boundaries before sending it to the device. The fill value used by the host is typically zero, but this is not enforced at the Blaze level -- it depends on the TTNN tilize path.

# PROPOSED

TensorAdapter's `ShapeEngine.compute_padding()` step explicitly sets `padding_fill = 0.0` and `padding_strategy = "ZERO"` for all ops in the ZERO category. The fill value is recorded in the ShapeDescriptor and validated at CB allocation time.

---

### Strategy 2: Negative-Infinity Padding (NEG_INF)

**Fill value:** `-inf` (represented as `0xFF80` in bfloat16)

**Mathematical basis:** Softmax computes `softmax(x)[i] = exp(x[i]) / sum(exp(x[j]))`. For padded positions to contribute zero probability, their logits must be `-inf`, because `exp(-inf) = 0`.

```
Correct (neg-inf-padded softmax):
  Input:   [2.0, 1.0, 0.5, -inf, ..., -inf]  (3 real + 29 padded to tile width 32)
  exp:     [7.389, 2.718, 1.649, 0.0, ..., 0.0]
  sum:     11.756
  result:  [0.628, 0.231, 0.140, 0.000, ..., 0.000]
  sum(result[0:3]) = 0.999  (correct: real probabilities sum to ~1.0)

WRONG (zero-padded softmax -- silent correctness bug):
  Input:   [2.0, 1.0, 0.5, 0.0, 0.0, ..., 0.0]
  exp:     [7.389, 2.718, 1.649, 1.000, ..., 1.000]   <- 29 padding contribute 1.0 each
  sum:     7.389 + 2.718 + 1.649 + 29*1.0 = 40.756
  result:  [0.181, 0.067, 0.040, 0.025, ..., 0.025]
  sum(result[0:3]) = 0.288  <-- WRONG: 71.2% probability leaked to padding positions
```

> **Warning:** The zero-padded softmax produces results that are numerically plausible (all positive, sum to 1.0 across the full padded vector) but mathematically wrong. The attention weights for real token positions are diluted by a factor of ~3.5x in this example (only 28.8% of probability mass remains on real tokens). In a production model with 128 tokens and a non-multiple-of-32 sequence length, the model attends to "phantom" tokens that do not exist, degrading output quality in ways that are difficult to detect without explicit numerical validation.

**Representation in bfloat16:** The IEEE 754 negative infinity is `0xFF80` in bfloat16.

**Format-specific fill values:**

| Format | Representation of -inf | Notes |
|--------|----------------------|-------|
| **float32** | `0xFF800000` (IEEE -inf) | Exact |
| **bfloat16** | `0xFF80` (IEEE -inf) | Exact |
| **bfloat8_b** | ~`-1e30` (max negative) | Approximate; `exp(-1e30)` underflows to 0 |
| **bfloat4_b** | ~`-1e2` (max negative) | Small dynamic range; verify `exp(fill)` is effectively zero |

> **Warning:** Block floating-point formats (bfloat8_b, bfloat4_b) cannot represent true IEEE infinity because they use shared exponents within a 16x16 face. The maximum negative value in bfloat8_b is approximately `-1e30`, which is sufficient for softmax padding. However, bfloat4_b has a much smaller dynamic range and its maximum negative value may not fully suppress padded positions. For SDPA and softmax ops using bfloat4_b, promote to bfloat16 before softmax or verify the format-specific constant.

# EXISTING

SDPA in `blaze/ops/sdpa/op.py` handles the causal mask by applying a large negative value to future-token positions, effectively implementing NEG_INF padding for the attention mask. However, this is hardcoded within the SDPA kernel, not managed by a general padding system.

# PROPOSED

TensorAdapter registers softmax and SDPA inputs as NEG_INF-padded in the PaddingPolicy registry (File 02). Any tensor flowing into a softmax input port is automatically padded with the format-appropriate negative value. The developer never specifies padding for softmax.

---

### Strategy 3: Mask-Based Padding (MASK)

**Fill value:** A separate boolean mask tensor tracks which positions are real data vs. padding.

**Mathematical basis:** Some operations cannot be made padding-neutral with any single fill value. Mean reduction is the canonical example:

```
Mean reduction walkthrough: mean([3.0, 5.0, 7.0]) with tile width 32

Without mask (zero-padded):
  Padded: [3.0, 5.0, 7.0, 0.0, ..., 0.0]   -- 32 elements
  Sum:    15.0
  Count:  32                                  -- WRONG: counts padding
  Mean:   15.0 / 32 = 0.469                  -- WRONG: should be 5.0
  Error:  90.6% underestimate

With mask:
  Padded: [3.0, 5.0, 7.0, 0.0, ..., 0.0]
  Mask:   [1,   1,   1,   0,   ...,  0]
  Sum:    sum(x * mask) = 15.0
  Count:  sum(mask) = 3                       -- CORRECT
  Mean:   15.0 / 3 = 5.0                     -- CORRECT
```

Mask-based padding is more expensive than fill-value padding because it requires an additional CB to hold the mask tensor and modified kernel logic to apply the mask. For this reason, it is used only when no fill value can produce correct results.

**Applicable ops:** Mean reduction, variance computation, LayerNorm (which computes mean and variance internally), any custom reduction that divides by element count.

# EXISTING

TT-Blaze does not have a general mask-based padding mechanism. The causal mask in SDPA is the closest analog, but it tracks token-level validity, not tile-level padding. Developers who need masked reductions must implement the logic manually.

# PROPOSED

TensorAdapter provides mask-based padding as an option in the PaddingPolicy registry. When a mean-reduction op is registered with `MASK` strategy, the adapter generates a mask tensor at `adapt()` time and allocates an additional CB.

---

### Strategy 4: Identity-Element Padding (IDENTITY)

**Fill value:** Depends on the reduction operation:

| Operation | Identity Element | Fill Value | Invariant |
|-----------|-----------------|-----------|-----------|
| Addition (`+`) | `0.0` | `0x0000` (bf16) | `x + 0 = x` |
| Multiplication (`*`) | `1.0` | `0x3F80` (bf16) | `x * 1 = x` |
| Maximum (`max`) | `-inf` | `0xFF80` (bf16) | `max(x, -inf) = x` |
| Minimum (`min`) | `+inf` | `0x7F80` (bf16) | `min(x, +inf) = x` |
| Logical AND | `True` | `1` | `x AND True = x` |

```
Product reduction walkthrough: prod([2.0, 3.0, 4.0]) with tile width 32

Wrong (zero-padded product):
  Padded: [2.0, 3.0, 4.0, 0.0, 0.0, ..., 0.0]
  Product: 2 * 3 * 4 * 0 * ... * 0 = 0.0   -- 100% error

Correct (identity-padded product, fill=1.0):
  Padded: [2.0, 3.0, 4.0, 1.0, 1.0, ..., 1.0]
  Product: 2 * 3 * 4 * 1 * ... * 1 = 24.0  -- CORRECT
```

> **Warning:** The IDENTITY strategy subsumes ZERO for addition reductions and NEG_INF for max reductions. It is distinguished as a separate strategy because the fill value must be determined from the reduction operation, not just the op type. A "reduce" op might do addition, multiplication, or max depending on its configuration. The PaddingPolicy must inspect the reduction mode.

# EXISTING

Current Blaze reduction ops (e.g., `gated_reduce`, `gated_local_reduce`) assume zero-padding because they implement addition-based reductions.

# PROPOSED

The PaddingPolicy registry associates each reduction op with its reduction mode and derives the identity element automatically.

---

### Strategy 5: No Padding (NONE)

**Fill value:** N/A -- no padding is inserted.

**When applicable:** When the tensor's dimensions are already tile-aligned on both the height and width axes. No padded positions exist, so no fill value is needed.

| Model | Hidden Dim | Intermediate Dim | Both Tile-Aligned? |
|-------|-----------|-------------------|-------------------|
| LLaMA-7B | 4096 (128 tiles) | 11008 (344 tiles) | Yes |
| LLaMA-13B | 5120 (160 tiles) | 13824 (432 tiles) | Yes |
| DeepSeek V3 | 7168 (224 tiles) | 2048 (64 tiles) | Yes |
| GPT-2 Small | 768 (24 tiles) | 3072 (96 tiles) | Yes |
| Custom model | 37 | 100 | No |

Most production LLM architectures choose dimensions that are multiples of 32 (or even 128 or 256) precisely to avoid padding overhead. When the abstraction detects `padded_shape == logical_shape`, it sets `padding_strategy = "NONE"` and skips all padding logic.

> **Warning:** Even when a model's weight dimensions are tile-aligned, the **sequence length** or **batch size** during inference may not be. A single-token decode with `seq_len=1` requires padding height from 1 to 32 -- a 32x expansion. NONE applies only when ALL tiled dimensions are aligned.

---

## The Consumer-Determines-Padding Principle

A critical design insight -- established in Chapter 2, File 05 as Principle P6 and confirmed by Triton's per-load `other=` parameter pattern -- is that the **consuming operation** determines the fill value, not the producing operation.

Consider a tensor `X` that flows through a pipeline:

```
X = matmul(A, B)        -- produces X with logical [1, 4096]
Y = softmax(X)          -- consumes X, needs NEG_INF padding
Z = matmul(X, W)        -- also consumes X, needs ZERO padding
```

If X is padded at production time, which fill value should be used? Zero for the downstream matmul or `-inf` for the downstream softmax? Neither choice works for both consumers.

The solution: **padding is applied at the consumption boundary**, not at the production boundary. Each consumer declares its required padding strategy, and the adapter inserts padding (or re-padding) when the producer's strategy does not match:

```
                    Producer                        Consumers
                  +-----------+
                  | matmul    |
   A, B -------> | output: X |--+---> [re-pad to NEG_INF] --> softmax(X)
                  | (logical  |  |
                  |  shape)   |  +---> [pad with ZERO]     --> matmul(X, W)
                  +-----------+
```

This is why the padding strategy is recorded in the `ShapeDescriptor` -- it tells downstream ops what padding the tensor currently has, so they can decide whether re-padding is necessary.

---

## Current Manual Approach: PaddedRMSNorm Deep Dive

PaddedRMSNorm in `blaze/ops/padded_rmsnorm/op.py` is the canonical example of manual padding management. It demonstrates every aspect of the problem that the abstraction must automate.

### The CB Architecture

```
# EXISTING -- PaddedRMSNorm CB layout
class PaddedRMSNorm(MicroOp):
    input: Input = Input()          # Raw (unpadded) input from upstream
    gamma: Input = Input()          # Raw (unpadded) gamma weights
    output: Output = Output()       # Padded output
    padded_input: Internal = Internal()   # Scratch CB: zero-padded input
    padded_gamma: Internal = Internal()   # Scratch CB: zero-padded gamma
```

Five CBs for a single normalization op. The standard (non-padded) RMSNorm uses only three CBs (input, gamma, output). The two extra Internal CBs exist solely for padding.

### The Padding Pipeline

```
Step 1: NCRISC copies raw data into padded scratch CBs
  input (N tiles, storage layout) --> padded_input (padded_N tiles, interpreted tile)
  gamma (N tiles, storage layout) --> padded_gamma (padded_N tiles, interpreted tile)

Step 2: NCRISC zero-fills the padding positions
  Copies input_bytes of real data, then zeros out (padded_bytes - input_bytes)

Step 3: TRISC computes RMSNorm on padded data
  Uses padded_input and padded_gamma CBs
  Zeros garbage rows in the last tile after squaring (pad_rows / valid_rows)

Step 4: Output is in padded geometry
  Downstream ops must be aware of the padded shape
```

### CT Args for Padding

```python
# EXISTING -- PaddedRMSNorm padding-specific CT args
elem_sz = get_element_size_bytes(ttnn.bfloat16)
input_bytes = width * elem_sz           # Bytes of real data per row
padded_bytes = padded_w * elem_sz       # Bytes of padded row (including zeros)

# SFPU padding parameters
tile_elements = itile.tile_shape[0] * itile.tile_shape[1]
remainder = width % tile_elements
sfpu_elems_per_row = 32
total_rows = tile_elements // sfpu_elems_per_row
valid_rows = remainder // sfpu_elems_per_row if remainder else total_rows
pad_rows = total_rows - valid_rows
```

The developer must compute six padding-related CT args:
1. `input_bytes` -- byte count of real data
2. `padded_bytes` -- byte count including padding
3. `padded_num_tiles` -- tile count after padding
4. `pad_rows` -- number of SFPU rows to zero in the last tile
5. `valid_rows` -- number of SFPU rows with real data in the last tile
6. `total_rows` -- total SFPU rows per tile

> **Warning:** The `pad_rows` and `valid_rows` computation operates at the SFPU row level (32 elements per row), not at the element level. This means padding granularity is 32 elements. PaddedRMSNorm handles this by zero-filling at the byte level (NCRISC) and row-zeroing at the tile level (TRISC). Both levels must agree.

### Walkthrough: PaddedRMSNorm for width=37

```
width = 37 (non-tile-aligned)

Step 1: interpret_tile_padded(37)
  full_sz = 32 * 32 = 1024
  half_sz = 16 * 32 = 512
  37 % 1024 != 0 and 37 % 512 != 0 -> needs padding
  padded_half = round_up(37, 512) = 512
  512 / 512 = 1 tile, <= MAX_TILES (8) -> use HALF
  Result: tile = (16, 32), padded_n = 1, padded_w = 512

Step 2: Compute byte counts
  input_bytes = 37 * 2 = 74 bytes
  padded_bytes = 512 * 2 = 1024 bytes

Step 3: Compute SFPU row parameters
  tile_elements = 16 * 32 = 512
  remainder = 37 % 512 = 37
  sfpu_elems_per_row = 32
  total_rows = 512 / 32 = 16
  valid_rows = 37 / 32 = 1 (integer division)
  pad_rows = 16 - 1 = 15

Step 4: CB allocation
  padded_input: 1 tile of (16, 32) at page_size = 1024 bytes
  padded_gamma: 1 tile of (16, 32) at page_size = 1024 bytes
  Total padding overhead: 2 * 1024 = 2,048 bytes

Step 5: Scalar computation
  scalar = 1.0 / sqrt(37) = 0.1644 (NOT 1.0 / sqrt(512) = 0.0442)
  Using wrong scalar: 31.5% RMS error, up to 153% per-element error
```

### Walkthrough: PaddedRMSNorm for width=4000

```
width = 4000 (not a multiple of 512 or 1024)

Step 1: interpret_tile_padded(4000)
  padded_half = round_up(4000, 512) = 4096
  4096 / 512 = 8 tiles, == MAX_TILES (8) -> use HALF (borderline)
  Result: tile = (16, 32), padded_n = 8, padded_w = 4096

Step 2: Compute byte counts
  input_bytes = 4000 * 2 = 8,000 bytes
  padded_bytes = 4096 * 2 = 8,192 bytes
  Waste: 192 bytes (2.4% overhead)

Step 3: SFPU row parameters
  remainder = 4000 % 512 = 416
  valid_rows = 416 / 32 = 13
  pad_rows = 16 - 13 = 3

Step 4: CB allocation
  padded_input: 8 tiles at 1,024 bytes = 8,192 bytes
  padded_gamma: 8 tiles at 1,024 bytes = 8,192 bytes
  Total padding overhead: 16,384 bytes (16 KB)
```

The 96-element misalignment (4096 - 4000 = 96) costs 16 KB of L1 and 6 additional CT args.

# PROPOSED

TensorAdapter eliminates all six decisions. The adapter detects the non-aligned width, selects PaddedRMSNorm automatically, computes all CT args from ShapeDescriptor, and allocates padded CBs with zero-fill:

```python
# PROPOSED -- automatic PaddedRMSNorm selection
def _emit_rmsnorm(self, f, activation, gamma, prefix="rmsnorm"):
    act_desc = self.shape_engine.full_analysis(activation, "rmsnorm.input")
    needs_padding = act_desc.logical_shape != act_desc.padded_shape
    if needs_padding:
        input_bytes = act_desc.logical_shape[-1] * DTYPE_BYTES[act_desc.data_format]
        padded_bytes = act_desc.padded_shape[-1] * DTYPE_BYTES[act_desc.data_format]
        out = PaddedRMSNorm.emit(f, act_handle, gamma_handle,
                                  input_bytes=input_bytes,
                                  padded_bytes=padded_bytes, prefix=prefix)
    else:
        out = RMSNorm.emit(f, act_handle, gamma_handle, prefix=prefix)
    return TypedHandle(handle=out, shape=act_desc)
```

---

## SDPA Padding: Implicit Masking Strategy

SDPA's approach to padding differs fundamentally from PaddedRMSNorm's. Rather than zero-filling padded positions in the input, SDPA applies masking at the score level:

```python
# EXISTING -- from SDPA's CB allocation
cb_mask_in = _scratch("mask_in", qk_tiles)
cb_identity_scale_in = _scratch("identity_scale_in", 1)
cb_zero_in = _scratch("zero_in", 1)

# Packed scalar values
identity_scalar_packed = 0x3F803F80  # Two bfloat16 1.0 values
zero_scalar_packed = 0x00000000      # Two bfloat16 0.0 values
```

### The SDPA Masking Pipeline

```
Step 1: Compute QK^T scores (all positions, including padding)
  QK = Q @ K^T   ->  [PNHt, St] tile grid

Step 2: Apply causal mask (valid positions = 1.0, future/padding = 0.0)
  Scores = QK * mask  ->  future/padded positions multiplied by 0
  Scores[masked] = -inf

Step 3: Softmax over masked scores
  exp(-inf) = 0  ->  padded positions contribute zero weight

Step 4: Compute attention output
  Output = softmax(Scores) @ V  ->  padded V columns contribute nothing
```

This implicit masking is more efficient than explicit padding for SDPA because K and V tensors are not physically padded, masking happens in the score space, and the causal mask already exists for correctness.

---

## Walkthrough: Wrong Padding Through an Attention Block

To illustrate why per-op padding strategy matters, trace the consequences of naively zero-padding everything through a simplified attention block with `seq_len=3` and tile width 32:

**Steps 1-2: Q/K/V projections and QK scores (ZERO padding -- correct)**
```
Q, K, V = activation @ W_q, W_k, W_v
  Zero-padded activation: 29 zero rows produce zero Q/K/V vectors
  QK scores: [s00, s01, s02, 0, ..., 0] -- zero K -> zero scores
  Status: Correct so far -- zero-padding is appropriate for matmul
```

**Step 3: Softmax (REQUIRES NEG_INF -- ZERO is WRONG)**
```
WRONG (zero-padded scores into softmax):
  Row 0: [s00, s01, s02, 0.0, 0.0, ..., 0.0]   (3 real + 29 zeros)
  softmax denominator: exp(s00) + exp(s01) + exp(s02) + 29*exp(0)
                     = real_sum + 29*1.0 = real_sum + 29.0
  The 29 padded positions each contribute exp(0)=1.0, diluting attention
  weights across 29 phantom positions (~10x error for 3 real + 29 phantom).

CORRECT (neg-inf-padded scores):
  Row 0: [s00, s01, s02, -inf, ..., -inf]
  softmax denominator: real_sum + 29*exp(-inf) = real_sum + 0
```

**Step 4: Attention output**
```
WRONG: attention_weights @ V -- weights diluted by ~10x, wrong output
CORRECT: weights sum to 1.0 over 3 real positions, correct output
```

The error magnitude depends on the ratio of padding to real data. For `seq_len=3` in a 32-wide tile, the denominator inflates by `29/3 ~ 10x`.

---

## Padding Strategy Decision Tree

```
                 Does op contain softmax or log-sum-exp?
                        /                    \
                      YES                     NO
                       |                       |
                  NEG_INF               Is op a reduction?
                                        /              \
                                      YES               NO
                                       |                 |
                              What kind of reduction?    ZERO
                            /        |         \
                        sum/add    product     mean/variance
                           |         |              |
                         ZERO    IDENTITY(1.0)    MASK
                                                (ZERO + count correction)
```

This tree is what the PaddingPolicy registry (File 02) formalizes into an enum and per-op lookup.

---

## Padding Cost Model

| Strategy | CB Overhead | Compute Overhead | Memory Overhead |
|----------|------------|-----------------|----------------|
| ZERO | None (default L1 state) | None (zero contributions are free) | Padded positions occupy L1 space |
| NEG_INF | Fill kernel required if L1 not pre-filled | `exp(-inf)=0` is fast on hardware | Same as ZERO |
| MASK | +1 CB for mask tensor | Mask application per element | 2x CB memory (data + mask) |
| IDENTITY | Fill kernel required | None (identity contributions are free) | Same as ZERO |
| NONE | None | None | None |

For production LLM workloads, the vast majority of tensors use ZERO or NONE. NEG_INF applies only to attention/softmax paths. MASK and IDENTITY are rare but essential for correctness.

> *Design Principle P1 (Separate Computation from Execution Strategy):* The developer writes `output = softmax(input)`. The padding system automatically selects NEG_INF, computes the fill value for the tensor's data format, and inserts the padding op. The developer never specifies `-inf` or even knows that padding is happening.

---

## Interaction with Other Subsystems

### Padding and Tile Decomposition (Chapter 4)

Chapter 4's `tile_decompose()` sets a preliminary `padding_strategy = "ZERO"` when `logical_shape != padded_shape`. Chapter 5's padding pass refines this based on op semantics.

### Padding and Data Formats (Chapter 6)

The fill value must be representable in the target data format. For mixed-precision chains (bfloat8_b weights -> bfloat16 activations -> float32 accumulation), the padding fill value must be converted at each format boundary.

### Padding and CB Sizing (Chapter 7)

Extra padding CBs (like PaddedRMSNorm's `padded_input` and `padded_gamma`) consume L1 space. The L1 budget model must account for these.

### Padding and Broadcasting (Chapter 8)

When a broadcast source tensor requires padding, the padding fill value must be applied before broadcasting. Broadcasting a zero-padded tensor to 8 cores is correct for matmul but wrong for softmax.

---

## Proposed: Padding Strategy in ShapeDescriptor

```python
# PROPOSED -- padding fields in ShapeDescriptor (from Chapter 3)
@dataclass(frozen=True)
class ShapeDescriptor:
    # ... existing fields (logical_shape, padded_shape, tile_shape, etc.) ...

    padding_strategy: str = "ZERO"
    """Padding policy: 'ZERO', 'NEG_INF', 'MASK', 'IDENTITY', 'NONE'.
    Selected based on op semantics from the PaddingPolicy registry (File 02)."""

    padding_fill: float = 0.0
    """Numeric fill value for padded positions."""

    has_padding: bool = False
    """True when logical_shape != padded_shape. Computed in __post_init__."""
```

---

## Key Takeaways

- **Five padding strategies** cover all TT-Blaze op requirements: ZERO (additive identity for matmul/linear/add), NEG_INF (softmax suppression via `exp(-inf)=0`), MASK (mean/variance with correct denominator), IDENTITY (op-specific neutral element for multiply/max/min reductions), and NONE (tile-aligned tensors needing no padding).
- **The consumer determines the padding value**, not the producer. A tensor flowing to both softmax and matmul cannot have a single fill value that satisfies both. Padding is applied or re-applied at consumption boundaries based on the PaddingPolicy registry.
- **Wrong padding values are silent correctness bugs.** Zero-padding a softmax input leaks 71.2% of probability mass to padding positions (3 real + 29 padded). Zero-padding a mean reduction underestimates by 90.6%. Wrong RMSNorm scalar produces 31.5% RMS error. None of these produce errors, warnings, or crashes.
- **PaddedRMSNorm demonstrates the manual burden:** five CBs, six CT args, two RISC pipelines (NCRISC byte-copy + TRISC row-zeroing), and the developer must choose between RMSNorm and PaddedRMSNorm based on alignment. TensorAdapter automates all of this.
- **Block floating-point formats** (bfloat8_b ~1,088 bytes/tile, bfloat4_b ~576 bytes/tile) cannot represent true IEEE infinity. The padding system uses format-specific large-negative constants and must verify that `exp(fill_value)` is effectively zero for NEG_INF strategy.
- **Most production LLM dimensions are tile-aligned** (4096, 7168, 11008, 2048 are all multiples of 32), making NONE the most common strategy. Padding issues arise primarily at sequence and batch dimensions during inference.

## Source Files

- `blaze/ops/padded_rmsnorm/op.py` -- `PaddedRMSNorm.emit()`: explicit zero-padding with `padded_input`/`padded_gamma` Internal CBs, `input_bytes`/`padded_bytes`/`pad_rows`/`valid_rows` CT args
- `blaze/ops/sdpa/op.py` -- `SDPADecode.emit()`: implicit masking via `mask_in`, `identity_scale_in`, `zero_in` CBs; `identity_scalar_packed = 0x3F803F80`, `zero_scalar_packed = 0x00000000`
- `blaze/utils.py` -- `interpret_tile_padded()`: computes padded tile geometry and padded width, including MAX_TILES=8 constraint for HALF tiles
- `blaze/blaze_op.py` -- `Internal` class: descriptor for scratch CBs used by padding
- `blaze/cb_engine.py` -- `DTYPE_BYTES`: byte sizes per element, needed for `input_bytes`/`padded_bytes` computation

---

[Previous: Chapter 4 -- Automatic Tile Decomposition](../ch04_tile_decomposition/) | [Next: File 02 -- Op Padding Registry](./02_op_padding_registry.md)
