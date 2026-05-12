# Blocking Strategy Selection with Performance Modeling

The blocking strategy determines how a matmul's tile loops are structured: which tiles are loaded into L1 simultaneously, which are streamed from DRAM, and how partial results accumulate in the dest registers. This is the highest-impact sizing decision in the entire abstraction layer -- a suboptimal blocking configuration can leave 80% of the compute hardware idle while the NOC saturates, or conversely fill L1 to capacity while the compute units starve for data. This section provides a deep analysis of matmul blocking with detailed performance models, cycle-count estimates, and bandwidth calculations. It then extends the analysis to RMSNorm, softmax, and SDPA patterns.

For every blocking configuration, we estimate NOC transactions, compute cycles, pipeline stall probability, and L1 pressure -- giving the developer (or the automatic system) a principled basis for choosing among feasible configurations rather than relying on heuristics.

> *Design Principle P1 (Separate Computation Intent from Execution Strategy):* The developer specifies "matmul(A, B)" -- the system selects the blocking strategy. But for performance-critical ops, the developer needs to understand the trade-offs to evaluate or override the system's choice. This section provides that understanding.

> *Design Principle P5 (Operate on Graphs, Not Individual Tensors):* Blocking strategy selection requires a graph-level view — each op's blocking choice affects its neighbors' CB requirements, and the optimal configuration considers the entire fused program's L1 budget, not just one op in isolation. The search minimizes NOC traffic as a structural consequence of maximizing block volume subject to L1 constraints.

**What you will learn:**

- The three matmul blocking dimensions (RT, CT, KT) and how they define the tile-loop structure on Tenstorrent hardware
- A detailed performance model: NOC bandwidth cost, compute utilization, L1 pressure, and pipeline overlap as functions of (RT, CT, KT)
- Cycle-count estimates for DeepSeek V3 and LLaMA matmul configurations with concrete (RT, CT, KT) comparisons
- The `compute_subblock_w()` interaction with blocking: how dest register capacity constrains the inner loop
- The formal constraint problem (C1-C9) and enumeration algorithm for feasible blocking triples with L1-budget-constrained optimization
- Extension to non-matmul ops: RMSNorm accumulation blocking, softmax chunk sizing, SDPA tree reduction blocking
- How the proposed `AutoBlocker` integrates with the `AutoSizer` from File 02

---

## 1. Matmul Blocking Dimensions

A tiled matrix multiply `C[M, N] = A[M, K] * B[K, N]` on Tenstorrent hardware is parameterized by three blocking dimensions, each measured in tiles:

| Dimension | Symbol | Meaning | Loop Level |
|-----------|--------|---------|------------|
| Row tiles | RT | Number of output row tiles per core | Outer loop |
| Column tiles | CT | Number of output column tiles per core | Middle loop |
| K tiles | KT | Number of accumulation tiles loaded per inner-loop pass | Inner loop |

The kernel executes as a triple-nested loop:

```text
for rt in range(0, M_tiles, RT):           # outer: rows
    for ct in range(0, N_tiles_per_core, CT): # middle: columns
        for kt in range(0, K_tiles, KT):     # inner: accumulation
            load A[rt:rt+RT, kt:kt+KT] into in0_cb      # NCRISC: NOC read
            load B[kt:kt+KT, ct:ct+CT] from DRAM/L1      # NCRISC: NOC/DRAM read
            matmul_tiles(A_block, B_block, C_partial)      # TRISC: compute
        store C[rt:rt+RT, ct:ct+CT] from out_cb            # BRISC: NOC write
```

### Decode Mode vs Prefill Mode

In **decode mode** (single token, M=1), the activation is a single row vector. With 1x32 tiles:

```text
RT = 1 (always -- single token)
M_tiles = 1
```

The blocking reduces to two dimensions: CT and KT.

In **prefill mode** (multiple tokens, M>>1), the activation has multiple rows. With 32x32 tiles:

```text
RT varies from 1 to M_tiles
M_tiles = ceil(M / 32)
```

All three dimensions are active.

### How Current Blaze Ops Choose Blocking

The current codebase uses different blocking strategies depending on the matmul variant:

**Matmul (direct-address):** No explicit blocking -- the entire K dimension is buffered in `in0_cb` and the full N_per_core in the output CB. Effectively `KT = K_tiles, CT = N_per_core_tiles, RT = 1`:

```python
# EXISTING -- from blaze/ops/matmul/op.py (lines 48, 63)
k_num_tiles = in0_handle.num_pages      # KT = full K
out_w_per_core = in1_cb.num_pages // k_num_tiles  # CT = full N per core
```

**DRAMStreamingMatmul:** Explicit K blocking via `subblock_k`:

```python
# EXISTING -- from blaze/ops/dram_streaming_matmul/op.py (lines 128-132)
if subblock_k is None:
    subblock_k = max(1, Kt // 4)    # KT = Kt / 4 by default
while Kt % subblock_k != 0 and subblock_k > 1:
    subblock_k -= 1
```

**KNSlicedMatmul:** Both K and N are parallelized across cores:

```python
# EXISTING -- from blaze/ops/kn_sliced_matmul/op.py (lines 30-43)
def _derive_kn_parallel(tile_h, tile_w, max_cores_per_branch):
    elements_per_tile = tile_h * tile_w
    if elements_per_tile < 256 and 256 % elements_per_tile == 0:
        n_parallel = 256 // elements_per_tile
    else:
        n_parallel = 1
    num_per_branch = (max_cores_per_branch // n_parallel) * n_parallel
    k_parallel = num_per_branch // n_parallel
    return k_parallel, n_parallel, num_per_branch
```

None of these approaches systematically search the space of feasible blocking configurations. The automatic system should.

---

## 2. The Formal Constraint Problem

The blocking search can be stated as a formal optimization problem with nine constraints.

**Decision Variables:**
- RT: row blocking factor (tiles)
- CT: column blocking factor (tiles)
- KT: accumulation blocking factor (tiles)
- sb_h: subblock height (tiles)
- sb_w: subblock width (tiles)

**Constants:**
- M_t: total row tiles in output
- N_t: total column tiles per core
- K_t: total accumulation tiles
- s_A: page size for activation format (bytes)
- s_B: page size for weight format (bytes)
- s_C: page size for output format (bytes)
- L: available L1 budget for matmul CBs (bytes)
- D: max dest tiles (from compute_subblock_w logic)
- L_other: L1 consumed by non-matmul CBs in the fused program

**Constraints:**

```text
C1 (L1 Budget):       RT * KT * s_A + KT * CT * s_B + RT * CT * s_C  <=  L - L_other
C2 (CB Slot Limit):   3 + existing_cbs  <=  64
C3 (Row Divisibility): M_t % RT == 0
C4 (Col Divisibility): N_t % CT == 0
C5 (K Divisibility):   K_t % KT == 0
C6 (Dest Capacity):    sb_h * sb_w  <=  D
C7 (Subblock Div):     RT % sb_h == 0  AND  CT % sb_w == 0
C8 (Positivity):       RT >= 1,  CT >= 1,  KT >= 1,  sb_h >= 1,  sb_w >= 1
C9 (NOC Page Limit):   RT * KT * s_A <= NOC_MAX_PAGE_SIZE * max_transactions
```

**Objective:**

```text
MINIMIZE  estimated_execution_time(RT, CT, KT)
```

where execution time is `max(noc_cycles, compute_cycles)` with overlap adjustment, as derived in the next section. Note that maximizing the blocking product `RT * CT * KT` is a common heuristic proxy, but as the performance analysis below demonstrates, it is not always optimal.

---

## 3. Performance Model: Four Metrics

To compare blocking configurations, we need a performance model with four components:

### Metric 1: NOC Bandwidth Cost

Every tile that enters L1 from DRAM or from another core traverses the NOC. The NOC has finite bandwidth, and each transfer has both a latency component (setup time) and a throughput component (bytes/cycle).

**NOC parameters (Blackhole):**

| Parameter | Value | Notes |
|-----------|-------|-------|
| NOC bandwidth per link | 32 bytes/cycle | Bidirectional |
| NOC clock | ~1.2 GHz | Same as Tensix clock |
| NOC latency per transaction | ~80 cycles | Setup + arbitration |
| Max NOC page size | 16,384 bytes | Architecture limit |

The NOC cost of a blocking configuration is:

```text
NOC_cost = num_transactions * latency_per_transaction
         + total_bytes / bandwidth_per_cycle
```

For a matmul with blocking (RT, CT, KT):

```text
# Inner loop iterations
k_iters = K_tiles / KT

# Transactions per inner-loop iteration:
#   - Load activation block: RT * KT tiles
#   - Load weight block: KT * CT tiles (from DRAM)
transactions_per_k_iter = ceil(RT * KT * page_size_A / noc_max_page) +
                          ceil(KT * CT * page_size_B / noc_max_page)

# Total transactions for one (RT, CT) output block:
transactions_per_block = k_iters * transactions_per_k_iter

# Total blocks across full output:
num_blocks = (M_tiles / RT) * (N_tiles_per_core / CT)

# Total NOC cost (cycles):
total_noc_cycles = num_blocks * transactions_per_block * latency_per_transaction +
                   total_bytes_transferred / bandwidth_per_cycle
```

**Key insight:** Increasing KT reduces `k_iters` (fewer inner-loop iterations) but increases `RT * KT * page_size` per transaction (larger blocks). The amortized cost per tile decreases because the fixed latency per transaction is spread across more tiles.

### Metric 2: Compute Utilization

The Tensix compute unit (TRISC) processes tiles at a fixed rate determined by the matmul kernel's inner loop:

**Compute parameters (Blackhole):**

| Parameter | Value | Notes |
|-----------|-------|-------|
| FPU throughput | 1 tile-multiply per cycle | For 32x32 bfloat16 tiles |
| Accumulation | In-register (dest file) | No L1 round-trip for partial sums |
| Pipeline depth | ~8 cycles per tile-multiply | Including unpack + math + pack |
| Dest register tiles | 4-16 (depends on fp32/sync) | From `compute_subblock_w()` |

The compute cost of one (RT, CT, KT) output block:

```text
# Multiplications per output block:
muls_per_block = RT * CT * KT    # each output tile requires KT multiplications

# Compute cycles per block (assuming full pipeline utilization):
compute_cycles_per_block = muls_per_block * cycles_per_mul

# Total compute across all blocks:
total_compute_cycles = num_blocks * compute_cycles_per_block
```

Where `cycles_per_mul` depends on the math fidelity:

| Math Fidelity | Cycles per Tile Multiply | Notes |
|--------------|--------------------------|-------|
| LoFi | ~8 cycles | Single-precision approximate |
| HiFi2 | ~16 cycles | Higher precision, 2x slower |
| HiFi4 | ~32 cycles | Full precision, 4x slower |

**Compute utilization** is the fraction of total execution time spent in useful compute:

```text
utilization = total_compute_cycles / total_execution_cycles
```

Where `total_execution_cycles = max(total_noc_cycles, total_compute_cycles)` -- the bottleneck determines total execution time.

### Metric 3: L1 Pressure

The L1 footprint of a blocking configuration is the sum of all simultaneously live CBs:

```text
L1_pressure = activation_cb + weight_cb + output_cb + scratch_cbs

activation_cb = RT * KT * page_size_A                  # tiles in flight
weight_cb     = KT * CT * page_size_B * buffer_count    # multi-buffered
output_cb     = RT * CT * page_size_out                  # accumulation target
```

Where `buffer_count` is 1 for single-buffering, 2 for double-buffering, or 3 for triple-buffering (DRAMStreamingMatmul).

**L1 constraint (C1):** `L1_pressure <= L1_available`

### Metric 4: Pipeline Overlap Efficiency

The ideal execution overlaps NOC reads with compute: while TRISC multiplies the current tile block, NCRISC prefetches the next block. The overlap efficiency depends on whether the NOC and compute pipelines are balanced:

```text
noc_time_per_block = transactions_per_k_iter * latency + bytes_per_block / bandwidth
compute_time_per_block = muls_per_block * cycles_per_mul

overlap_ratio = min(noc_time_per_block, compute_time_per_block) /
                max(noc_time_per_block, compute_time_per_block)
```

When `overlap_ratio` is close to 1.0, the pipelines are balanced and double-buffering provides maximum benefit. When `overlap_ratio` is close to 0 (one pipeline is much slower), the faster pipeline stalls waiting for the slower one, and adding buffer pages provides diminishing returns.

---

## 4. Concrete Performance Analysis: DeepSeek V3 Matmul

Let us apply the model to a real workload: the Q projection matmul in DeepSeek V3 MLA attention, decode mode.

### Configuration

```text
Model: DeepSeek V3 MLA
Op: Q_A projection matmul
M = 1 (decode mode)
K = 7,168 (hidden dimension)
N = 1,536 per core (across 7 compute cores)
Activation: bfloat16, 1x32 tiles -> page_size_A = 64 bytes
Weights: bfloat8_b, 32x32 tiles -> page_size_B = 1,088 bytes (DIRECT_ADDRESS)
Output: bfloat16, 1x32 tiles -> page_size_out = 64 bytes
Math fidelity: LoFi (8 cycles/mul)
```

Derived values:
```text
K_tiles = 7168 / 32 = 224
N_tiles_per_core = 1536 / 32 = 48
RT = 1 (forced by M=1 decode mode)
```

### Configuration A: Full K, Full N (Current Matmul.emit() Default)

```text
RT=1, CT=48, KT=224
```

**L1 pressure:**
```text
activation_cb = 1 * 224 * 64       = 14,336 bytes (14 KB)
weight_cb     = 0 bytes              (DIRECT_ADDRESS, not in L1)
output_cb     = 1 * 48 * 64        = 3,072 bytes (3 KB)
Total L1:     17,408 bytes (17 KB)
```

**NOC cost:**
```text
Activation: single read of 224 tiles = 224 * 64 = 14,336 bytes
  Transactions: ceil(14,336 / 16,384) = 1 transaction
  Latency: 1 * 80 = 80 cycles + 14,336 / 32 = 448 bandwidth cycles = 528 cycles

Weights: read from DRAM per-tile via direct-address
  224 * 48 = 10,752 tile reads
  Each: 1,088 bytes -> ceil(1,088 / 16,384) = 1 transaction per tile
  Total: 10,752 * (80 + 1,088/32) = 10,752 * (80 + 34) = 10,752 * 114 = 1,225,728 cycles

Total NOC: ~1,226,256 cycles
```

**Compute cost:**
```text
Multiplications: 1 * 48 * 224 = 10,752 tile-muls
Compute: 10,752 * 8 = 86,016 cycles
```

**Analysis:**
```text
Bottleneck: NOC (1,226,256 >> 86,016 cycles)
Compute utilization: 86,016 / 1,226,256 = 7.0%
Overlap ratio: 86,016 / 1,226,256 = 0.07

This configuration is severely NOC-bound. The compute units are idle
93% of the time, waiting for weight data from DRAM.
```

### Configuration B: Subblocked K, Full N (DRAMStreamingMatmul Default)

```text
RT=1, CT=48, KT=56 (subblock_k = 224/4 = 56)
Triple-buffered weights
```

**L1 pressure:**
```text
activation_cb = 1 * 224 * 64          = 14,336 bytes (14 KB)
weight_cb     = 56 * 48 * 1,088 * 3   = 8,773,632 bytes -- EXCEEDS L1!
```

This configuration does not fit. Triple-buffering KT=56 columns of 48 output tiles requires 8.3 MB -- far beyond L1. DRAMStreamingMatmul does not buffer all CT columns simultaneously. Instead, it streams weights column-by-column:

**Corrected DRAMStreamingMatmul blocking:**
```text
RT=1, per-column streaming
Weight buffer: subblock_k * 3 (triple-buffered) tiles per column
  = 56 * 3 * 1,088 = 182,784 bytes (179 KB)
```

**NOC cost:**
```text
K iterations: 224 / 56 = 4 iterations per output column
Output columns: 48 per core

Weight reads per K iteration: 56 tiles * 1,088 bytes = 60,928 bytes
  Transactions per K iter: ceil(60,928 / 16,384) = 4 transactions
  Latency per K iter: 4 * 80 + 60,928 / 32 = 320 + 1,904 = 2,224 cycles

Total weight reads: 48 columns * 4 K-iters * 2,224 = 427,008 cycles
Activation reads: 4 K-iters * (56 * 64 = 3,584 bytes each)
  = 4 * (ceil(3,584/16,384) * 80 + 3,584/32) = 4 * (80 + 112) = 768 cycles

Total NOC: ~427,776 cycles
```

**Compute cost:**
```text
Multiplications: 48 * 224 = 10,752 tile-muls (same as Config A)
Compute: 10,752 * 8 = 86,016 cycles
```

**Analysis:**
```text
Bottleneck: NOC (427,776 >> 86,016)
Compute utilization: 86,016 / 427,776 = 20.1%

Improvement over Config A: 2.9x fewer NOC cycles due to triple-buffering
  overlap (while reading buffer 2 from DRAM, compute processes buffer 1,
  and buffer 0 is being filled for the next iteration).

But still NOC-bound: the DRAM bandwidth limit (32 bytes/cycle per link)
  constrains throughput regardless of blocking.
```

### Configuration C: Smaller Subblock, Tighter L1

```text
RT=1, per-column streaming
subblock_k = 28 (224/8)
Triple-buffered: 28 * 3 * 1,088 = 91,392 bytes (89 KB)
```

**NOC cost:**
```text
K iterations: 224 / 28 = 8 iterations per column
Weight reads per K iter: 28 * 1,088 = 30,464 bytes
  Transactions: ceil(30,464/16,384) = 2
  Latency: 2 * 80 + 30,464/32 = 160 + 952 = 1,112 cycles

Total weight reads: 48 * 8 * 1,112 = 427,008 cycles
  (Same total bytes, same total NOC cost!)
```

**Compute cost:** Unchanged at 86,016 cycles.

**Analysis:**
```text
Bottleneck: NOC (same)
Compute utilization: 20.1% (same)
L1 savings: 179 KB -> 89 KB (50% reduction)

Key insight: reducing subblock_k halves L1 pressure without changing
throughput, because the total bytes transferred over NOC is identical.
The only cost is more NOC transactions (8 vs 4 per column), each with
80-cycle latency overhead, but this is negligible relative to the
bandwidth-limited transfer time.
```

### Summary: DeepSeek V3 Q Projection

| Config | KT | L1 (weight CB) | NOC Cycles | Compute Cycles | Utilization | Bottleneck |
|--------|-----|----------------|------------|----------------|-------------|------------|
| A (full K) | 224 | 0 KB (DRAM direct) | 1,226,256 | 86,016 | 7.0% | NOC |
| B (subblock 56) | 56 | 179 KB | 427,776 | 86,016 | 20.1% | NOC |
| C (subblock 28) | 28 | 89 KB | 427,008 | 86,016 | 20.1% | NOC |
| D (subblock 8) | 8 | 26 KB | 473,088 | 86,016 | 18.2% | NOC |

The diminishing returns are clear: below `subblock_k ~ 56`, further reduction in L1 pressure has no throughput impact. The automatic system should select the **smallest subblock_k** that achieves near-maximum utilization, freeing L1 for other CBs in the fused op.

---

## 5. Concrete Performance Analysis: LLaMA Prefill Matmul

Prefill mode activates all three blocking dimensions. Consider a LLaMA 70B attention Q projection during prefill with sequence length 2,048:

### Configuration

```text
Model: LLaMA 70B
Op: Q projection matmul (prefill mode)
M = 2,048 (sequence length)
K = 8,192 (hidden dimension)
N = 1,024 per core (N_total=8,192 across 8 heads, 1 head per core)
Activation: bfloat16, 32x32 tiles -> page_size_A = 2,048 bytes
Weights: bfloat16, 32x32 tiles -> page_size_B = 2,048 bytes (L1 sharded)
Output: bfloat16, 32x32 tiles -> page_size_out = 2,048 bytes
Math fidelity: HiFi4 (32 cycles/mul)
```

Derived values:
```text
M_tiles = 2048 / 32 = 64
K_tiles = 8192 / 32 = 256
N_tiles_per_core = 1024 / 32 = 32
```

### Configuration P1: RT=4, CT=8, KT=32

```text
Blocks: (64/4) * (32/8) = 16 * 4 = 64 blocks
K iterations per block: 256 / 32 = 8
```

**L1 pressure:**
```text
activation_cb = 4 * 32 * 2,048    = 262,144 bytes (256 KB)
weight_cb     = 32 * 8 * 2,048    = 524,288 bytes (512 KB) -- weights in L1
output_cb     = 4 * 8 * 2,048     = 65,536 bytes (64 KB)
Total: 851,968 bytes (832 KB) -- 64% of available L1
```

**NOC cost:**
```text
Activation reads per block: 4 * 32 = 128 tiles = 262,144 bytes
  Per block: ceil(262,144/16,384) * 80 + 262,144/32 = 1,280 + 8,192 = 9,472 cycles

Weight reads per K iteration: 32 * 8 = 256 tiles = 524,288 bytes
  Per K iter: ceil(524,288/16,384) * 80 + 524,288/32 = 2,560 + 16,384 = 18,944 cycles

Per block NOC: 9,472 + 8 * 18,944 = 9,472 + 151,552 = 161,024 cycles
Total NOC: 64 * 161,024 = 10,305,536 cycles
```

**Compute cost:**
```text
Muls per block: 4 * 8 * 32 = 1,024 tile-muls
Per block: 1,024 * 32 = 32,768 cycles (HiFi4)
Total compute: 64 * 32,768 = 2,097,152 cycles
```

**Analysis:**
```text
Bottleneck: NOC (10.3M >> 2.1M)
Utilization: 2,097,152 / 10,305,536 = 20.3%
```

### Configuration P2: RT=8, CT=32, KT=8

Maximize output block size, minimize K iteration count:

```text
Blocks: (64/8) * (32/32) = 8 * 1 = 8 blocks
K iterations per block: 256 / 8 = 32
```

**L1 pressure:**
```text
activation_cb = 8 * 8 * 2,048       = 131,072 bytes (128 KB)
weight_cb     = 8 * 32 * 2,048      = 524,288 bytes (512 KB)
output_cb     = 8 * 32 * 2,048      = 524,288 bytes (512 KB)
Total: 1,179,648 bytes (1,152 KB) -- 89% of L1. Tight!
```

**NOC cost:**
```text
Activation reads per block: 8 * 8 = 64 tiles = 131,072 bytes
  Per block: ceil(131,072/16,384) * 80 + 131,072/32 = 640 + 4,096 = 4,736 cycles

Weight reads per K iter: 8 * 32 = 256 tiles = 524,288 bytes
  Per K iter: ceil(524,288/16,384) * 80 + 524,288/32 = 2,560 + 16,384 = 18,944 cycles

Per block NOC: 4,736 + 32 * 18,944 = 4,736 + 606,208 = 610,944 cycles
Total NOC: 8 * 610,944 = 4,887,552 cycles
```

**Compute cost:**
```text
Muls per block: 8 * 32 * 8 = 2,048 tile-muls
Per block: 2,048 * 32 = 65,536 cycles
Total compute: 8 * 65,536 = 524,288 cycles
```

**Analysis:**
```text
Bottleneck: NOC (4.9M >> 0.5M)
Utilization: 524,288 / 4,887,552 = 10.7%

Worse utilization than P1 despite larger blocks! Why?
  - More K iterations (32 vs 8) means more weight-read phases
  - Each K iteration reads the full weight block (512 KB)
  - 32 reads of 512 KB = 16 MB total weight data movement
```

### Configuration P3: RT=4, CT=4, KT=64 (Optimized)

Balance: large KT reduces K iterations, small CT reduces weight block size:

```text
Blocks: (64/4) * (32/4) = 16 * 8 = 128 blocks
K iterations per block: 256 / 64 = 4
```

**L1 pressure:**
```text
activation_cb = 4 * 64 * 2,048      = 524,288 bytes (512 KB)
weight_cb     = 64 * 4 * 2,048      = 524,288 bytes (512 KB)
output_cb     = 4 * 4 * 2,048       = 32,768 bytes (32 KB)
Total: 1,081,344 bytes (1,056 KB) -- 82% of L1
```

**NOC cost:**
```text
Activation per block: 4 * 64 = 256 tiles = 524,288 bytes
  Per block: ceil(524,288/16,384) * 80 + 524,288/32 = 2,560 + 16,384 = 18,944 cycles

Weight per K iter: 64 * 4 = 256 tiles = 524,288 bytes
  Per K iter: ceil(524,288/16,384) * 80 + 524,288/32 = 2,560 + 16,384 = 18,944 cycles

Per block NOC: 18,944 + 4 * 18,944 = 18,944 + 75,776 = 94,720 cycles
Total NOC: 128 * 94,720 = 12,124,160 cycles
```

**Compute cost:**
```text
Muls per block: 4 * 4 * 64 = 1,024
Per block: 1,024 * 32 = 32,768 cycles
Total: 128 * 32,768 = 4,194,304 cycles
```

**Analysis:**
```text
Bottleneck: NOC (12.1M >> 4.2M)
Utilization: 4,194,304 / 12,124,160 = 34.6%

Best utilization yet! Large KT amortizes K-loop overhead,
small CT keeps weight blocks manageable.
```

### LLaMA Prefill Summary

| Config | RT | CT | KT | L1 (KB) | NOC (M cyc) | Compute (M cyc) | Utilization |
|--------|-----|-----|-----|---------|-------------|-----------------|-------------|
| P1 | 4 | 8 | 32 | 832 | 10.3 | 2.1 | 20.3% |
| P2 | 8 | 32 | 8 | 1,152 | 4.9 | 0.5 | 10.7% |
| P3 | 4 | 4 | 64 | 1,056 | 12.1 | 4.2 | 34.6% |

**Key insight for the automatic system:** Maximizing `RT * CT * KT` (the throughput proxy) is **not** always optimal. P2 has the largest product (8*32*8=2,048) but the worst utilization (10.7%). P3 has a smaller product (4*4*64=1,024) but the best utilization (34.6%), because large KT amortizes the per-K-iteration NOC overhead.

The automatic system should optimize for **estimated execution time** (max of NOC and compute cycles), not the blocking product.

---

## 6. The Blocking Enumeration Algorithm

Given the performance model and constraints C1-C9, the automatic system enumerates feasible blocking triples and selects the one with minimum estimated execution time:

```python
# PROPOSED -- blocking strategy search
@dataclass
class BlockingConfig:
    """A candidate blocking configuration."""
    rt: int
    ct: int
    kt: int
    l1_bytes: int
    noc_cycles: int
    compute_cycles: int
    execution_cycles: int  # max(noc, compute) with overlap
    utilization: float

def find_optimal_blocking(
    M_tiles: int,
    K_tiles: int,
    N_tiles_per_core: int,
    page_size_A: int,
    page_size_B: int,
    page_size_out: int,
    l1_available: int,
    math_fidelity: str,
    buffer_count: int = 3,
    fp32_dest_acc_en: bool = False,
) -> BlockingConfig:
    """Enumerate feasible (RT, CT, KT) and select optimal.

    Strategy:
    1. Generate all divisor triples (RT | M_tiles, CT | N_tiles, KT | K_tiles)
    2. Filter by L1 constraint (C1)
    3. Filter by dest register constraint (C6, C7)
    4. Score by estimated execution time
    5. Return minimum-time configuration
    """
    cycles_per_mul = {"LoFi": 8, "HiFi2": 16, "HiFi4": 32}[math_fidelity]
    max_dest = compute_subblock_w(
        N_tiles_per_core,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )

    candidates: list[BlockingConfig] = []

    # Generate divisors (C3, C4, C5)
    rt_divs = [d for d in range(1, M_tiles + 1) if M_tiles % d == 0]
    ct_divs = [d for d in range(1, N_tiles_per_core + 1) if N_tiles_per_core % d == 0]
    kt_divs = [d for d in range(1, K_tiles + 1) if K_tiles % d == 0]

    for rt in rt_divs:
        for ct in ct_divs:
            # Check dest register constraint (C6, C7)
            if ct > max_dest and ct % max_dest != 0:
                continue

            for kt in kt_divs:
                # L1 constraint (C1)
                act_bytes = rt * kt * page_size_A
                wt_bytes = kt * ct * page_size_B * buffer_count
                out_bytes = rt * ct * page_size_out
                l1_total = act_bytes + wt_bytes + out_bytes

                if l1_total > l1_available:
                    continue

                # Performance estimate
                num_blocks = (M_tiles // rt) * (N_tiles_per_core // ct)
                k_iters = K_tiles // kt

                # NOC model (simplified)
                act_noc = _noc_cost(rt * kt * page_size_A)
                wt_noc = _noc_cost(kt * ct * page_size_B) * k_iters
                out_noc = _noc_cost(rt * ct * page_size_out)
                noc_per_block = act_noc + wt_noc + out_noc

                # Compute model
                muls_per_block = rt * ct * kt
                compute_per_block = muls_per_block * cycles_per_mul

                # Overlap: with multi-buffering, NOC and compute partially overlap
                overlap_factor = min(1.0, buffer_count / 2.0)
                exec_per_block = max(
                    noc_per_block * (1 - overlap_factor),
                    compute_per_block,
                ) + min(noc_per_block, compute_per_block) * overlap_factor

                total_exec = num_blocks * exec_per_block
                total_noc = num_blocks * noc_per_block
                total_compute = num_blocks * compute_per_block
                util = total_compute / max(total_exec, 1)

                candidates.append(BlockingConfig(
                    rt=rt, ct=ct, kt=kt,
                    l1_bytes=l1_total,
                    noc_cycles=int(total_noc),
                    compute_cycles=int(total_compute),
                    execution_cycles=int(total_exec),
                    utilization=util,
                ))

    if not candidates:
        raise L1BudgetExceeded(
            f"No feasible blocking for M={M_tiles}, K={K_tiles}, "
            f"N={N_tiles_per_core} with {l1_available} bytes available"
        )

    # Select minimum execution time
    return min(candidates, key=lambda c: c.execution_cycles)


def _noc_cost(total_bytes: int) -> int:
    """Estimated NOC cycles for transferring total_bytes."""
    NOC_LATENCY = 80
    NOC_BW = 32  # bytes/cycle
    NOC_MAX_PAGE = 16384
    num_transactions = max(1, (total_bytes + NOC_MAX_PAGE - 1) // NOC_MAX_PAGE)
    return num_transactions * NOC_LATENCY + total_bytes // NOC_BW
```

### Pruning the Search Space

For large matmuls, the divisor space can be large. For example, K_tiles=256 has 9 divisors, N_tiles=48 has 10, M_tiles=64 has 7 -- yielding 630 candidate triples. With L1 filtering, typically 50-100 survive.

For the automatic system, this is acceptable: evaluating 100 candidates with the performance model takes microseconds at compile time. No runtime cost.

For extremely large matmuls (K_tiles=1024, N_tiles=256), the divisor count grows, but L1 filtering eliminates most candidates early. A practical optimization is to generate divisors in descending order of KT (largest K blocks first) and prune entire KT values that exceed the L1 budget:

```python
# PROPOSED -- early pruning
for kt in reversed(kt_divs):  # largest KT first
    min_l1_for_kt = 1 * kt * page_size_A + kt * 1 * page_size_B * buffer_count + 1 * 1 * page_size_out
    if min_l1_for_kt > l1_available:
        continue  # even RT=1, CT=1 won't fit -- skip all larger KTs too
    # ... enumerate RT, CT for this KT ...
```

---

## 7. The compute_subblock_w() Interaction

The dest register constraint from `compute_subblock_w()` in `utils.py` limits how many output tiles can be processed in one compute invocation. This affects the CT dimension:

```python
# EXISTING -- from blaze/utils.py (lines 80-113)
def compute_subblock_w(num_tiles, *, fp32_dest_acc_en=False, dst_full_sync_en=None):
    # ... determines max_dest (4, 8, or 16 tiles) ...
    subblock_w = min(max_dest, num_tiles)
    while subblock_w > 1 and num_tiles % subblock_w != 0:
        subblock_w -= 1
    return subblock_w
```

The relationship: `CT` must be a multiple of `subblock_w`, or equal to `subblock_w`. If `CT = 48` and `max_dest = 8`:

```text
subblock_w = 8 (since 48 % 8 = 0)
Inner compute loop: 48 / 8 = 6 iterations of 8-wide tile multiplies
```

If `CT = 7` and `max_dest = 8`:

```text
subblock_w = 7 (since 7 < 8 and 7 % 7 = 0)
Inner compute loop: 1 iteration of 7-wide tile multiply (suboptimal -- only 7/8 dest utilization)
```

The blocking algorithm should prefer CT values that are multiples of `max_dest` for full dest utilization:

```python
# PROPOSED -- dest-aware CT filtering
def dest_efficient_ct_values(N_tiles_per_core: int, max_dest: int) -> list[int]:
    """Return CT divisors that use dest registers efficiently."""
    divs = [d for d in range(1, N_tiles_per_core + 1)
            if N_tiles_per_core % d == 0]
    # Prefer values that are multiples of max_dest
    efficient = [d for d in divs if d % max_dest == 0 or d <= max_dest]
    return efficient if efficient else divs
```

---

## 8. Extension to Non-Matmul Ops

### RMSNorm Blocking

RMSNorm does not have matmul-style blocking, but it does have an accumulation dimension. The `num_tiles` parameter determines how many tiles are accumulated in a single reduction:

```python
# EXISTING -- from blaze/ops/rmsnorm/op.py (line 121)
compute_n = num_tiles if num_tiles is not None else inp.num_pages
```

For wide hidden dimensions, reducing `num_tiles` to a fraction and performing multi-pass accumulation can reduce L1 pressure. However, the `mul_reduce_scalar_tile` LLK instruction has a hardware limit:

```python
# EXISTING -- from blaze/utils.py (lines 155-156)
MAX_TILES = 8  # mul_reduce_scalar_tile dest register limit
```

The blocking constraint for RMSNorm:
```text
num_tiles <= 8 (dest register limit for mul_reduce_scalar)
```

If hidden_dim requires more than 8 tiles (hidden_dim > 8 * 1024 = 8,192), `interpret_tile_padded()` switches to HALF tiles (16x32), doubling the tile count but halving each tile's footprint:

```python
# EXISTING -- from blaze/utils.py (lines 136-170)
def interpret_tile_padded(width: int) -> tuple[ttnn.Tile, int, int]:
    # ... Try HALF-tile padding first.
    padded_half = round_up(width, half_sz)
    if padded_half // half_sz <= MAX_TILES:
        return HALF, padded_half // half_sz, padded_half

    # Too many HALF tiles -- pad to next FULL-tile boundary instead.
    padded_full = round_up(width, full_sz)
    return FULL, padded_full // full_sz, padded_full
```

For DeepSeek V3 (hidden_dim=7,168):
```text
7168 / 1024 = 7 FULL tiles <= 8. Use FULL geometry.
CB sizing: 3 * 7 * 2,048 = 43,008 bytes (42 KB)
```

For a hypothetical model with hidden_dim=12,288:
```text
12288 / 1024 = 12 FULL tiles > 8. Try HALF:
12288 / 512 = 24 HALF tiles > 8. Still exceeds.
Fall back to FULL: 12 tiles, padded_width = 12288.
```

The automatic system would handle this by selecting the tile geometry that respects the dest register limit while minimizing padding overhead.

### Softmax Chunk Sizing

Softmax in SDPA processes attention scores in chunks. The chunk size determines how many KV sequence tiles are processed per pass:

```text
chunk_size = min(device_constrained_max, seq_tiles)
```

The L1 cost of softmax depends on chunk_size:
```text
QK_cb = chunk_size * page_size_float32  (QK scores in fp32)
softmax_scratch = chunk_size * page_size_float32  (exp, sum)
```

For long sequences (seq_len=128K), large chunk sizes reduce passes but increase L1:
```text
chunk_size=64:  64 * 4,096 = 262,144 bytes (256 KB) per CB
chunk_size=32:  32 * 4,096 = 131,072 bytes (128 KB) per CB
chunk_size=16:  16 * 4,096 =  65,536 bytes (64 KB) per CB
```

The automatic system should select chunk_size based on available L1 after accounting for Q, K, V, and output CBs.

### SDPA Tree Reduction Blocking

SDPA uses tree reduction across cores when cores_per_head > 1. The reduction blocking is determined by the tree depth:

```text
tree_rounds = log2(cores_per_head)
MAX_TREE_REDUCTION_ROUNDS = 6  (from sdpa/op.py)
```

Each tree round halves the active core count. The partial result CBs at each level are:

```text
partial_cb_per_round = 2 * dhead_tiles * page_size  (partial sum + max)
```

For 4 cores per head with dhead=128:
```text
tree_rounds = 2
dhead_tiles = 128 / 32 = 4
partial_cb_per_round = 2 * 4 * 2,048 = 16,384 bytes (16 KB)
total_tree_cb = 2 * 16,384 = 32,768 bytes (32 KB)
```

This is a modest L1 cost, but the automatic system must account for it in the total budget.

---

## 9. The Proposed AutoBlocker

The `AutoBlocker` combines the performance model with L1 budget awareness:

```python
# PROPOSED -- automatic blocking strategy selector
class AutoBlocker:
    """Selects optimal blocking strategy for matmul and related ops."""

    def __init__(self, l1_budget: L1Budget, device_params: DeviceParams):
        self._budget = l1_budget
        self._device = device_params

    def select_matmul_blocking(
        self,
        M: int, K: int, N_per_core: int,
        page_size_A: int, page_size_B: int, page_size_out: int,
        math_fidelity: str,
        fp32_dest_acc_en: bool = False,
        weight_in_l1: bool = False,
        buffer_count: int = 3,
    ) -> BlockingConfig:
        """Select optimal (RT, CT, KT) for a matmul.

        Args:
            M, K, N_per_core: dimensions in elements
            page_size_*: bytes per tile for each operand
            math_fidelity: "LoFi", "HiFi2", or "HiFi4"
            fp32_dest_acc_en: float32 accumulation mode
            weight_in_l1: True if weights are L1-sharded, False if DRAM-streamed
            buffer_count: number of weight buffer copies (1=single, 2=double, 3=triple)
        """
        tile_h_A, tile_w_A = self._infer_tile_shape(page_size_A)
        tile_h_B, tile_w_B = self._infer_tile_shape(page_size_B)

        M_tiles = M // tile_h_A if M > 1 else 1
        K_tiles = K // tile_w_A
        N_tiles = N_per_core // tile_w_B

        # Reserve L1 for non-matmul CBs in the fused op
        l1_for_matmul = self._budget.remaining_bytes

        config = find_optimal_blocking(
            M_tiles=M_tiles,
            K_tiles=K_tiles,
            N_tiles_per_core=N_tiles,
            page_size_A=page_size_A,
            page_size_B=page_size_B,
            page_size_out=page_size_out,
            l1_available=l1_for_matmul,
            math_fidelity=math_fidelity,
            buffer_count=buffer_count,
            fp32_dest_acc_en=fp32_dest_acc_en,
        )

        return config

    def select_rmsnorm_blocking(
        self, hidden_dim: int, data_format: ttnn.DataType,
    ) -> tuple[ttnn.Tile, int]:
        """Select tile geometry and num_tiles for RMSNorm."""
        return interpret_tile_padded(hidden_dim)

    def select_sdpa_chunk_size(
        self,
        seq_len: int,
        dhead: int,
        num_q_groups: int,
        k_format: ttnn.DataType,
        v_format: ttnn.DataType,
    ) -> int:
        """Select chunk_size for SDPA attention."""
        remaining = self._budget.remaining_bytes
        # Per-chunk L1: K_chunk + V_chunk + QK_scores + softmax_scratch
        per_chunk_tile_cost = (
            _TILE_SIZES[k_format] +    # K tiles
            _TILE_SIZES[v_format] +    # V tiles
            _TILE_SIZES[ttnn.float32] + # QK scores (always fp32)
            _TILE_SIZES[ttnn.float32]   # softmax scratch
        )
        max_chunk = remaining // per_chunk_tile_cost
        seq_tiles = seq_len // 32
        return min(max_chunk, seq_tiles, 64)  # cap at hardware tree limit
```

### Integration with AutoSizer

The `AutoBlocker` feeds into the `AutoSizer` from File 02:

```text
Pipeline:
  1. AutoBlocker.select_matmul_blocking() -> BlockingConfig
  2. BlockingConfig determines num_pages for activation, weight, output CBs
  3. AutoSizer.request() registers each CB with its minimum/preferred pages
  4. AutoSizer.solve() distributes remaining L1 budget for double-buffering
  5. FusedProgram.cb_scratch() uses the solved num_pages values
```

This two-phase approach separates the blocking decision (which is op-specific and performance-critical) from the budget allocation (which is global and correctness-critical).

---

## 10. Expert Override Mechanism

Following Design Principle P4 (Provide Defaults with Per-Decision Overrides), the automatic blocking system supports per-op overrides:

```python
# PROPOSED -- override syntax
# In user_args passed to compose():
user_args = {
    "blocking": {
        "rt": 4,
        "ct": 8,
        "kt": 64,
    },
    "subblock_k": 32,  # legacy override for DRAMStreamingMatmul
}

# Or via decorator on a BlazeModule:
@blaze.override(blocking={"rt": 4, "ct": 8, "kt": 64})
class MyMatmul(BlazeModule):
    pass
```

When an override is provided, the `AutoBlocker` skips the search and validates only that the override fits in L1. If it does not, a clear error is raised:

```text
L1BudgetExceeded: Manual blocking override (RT=4, CT=8, KT=64) requires
1,056 KB, but only 892 KB available. Current allocations:
  rmsnorm.input: 14 KB
  rmsnorm.gamma: 14 KB
  rmsnorm.out_cb: 14 KB
  mcast.dst: 14 KB
Suggestion: reduce KT to 32 (requires 528 KB) or free 164 KB from other CBs.
```

---

## Key Takeaways

- Matmul blocking is parameterized by three dimensions **(RT, CT, KT)** that determine L1 pressure, NOC bandwidth cost, and compute utilization. The optimal configuration depends on the specific operand sizes, data formats, and available L1 -- there is no universal "best" blocking.
- The blocking search is a formal constrained optimization problem with **nine constraints (C1-C9)** covering L1 budget, CB slot limit, divisibility, dest register capacity, subblock alignment, and positivity. Exhaustive enumeration of feasible divisor triples is practical at compile time.
- **Maximizing the blocking product RT * CT * KT is not always optimal.** The performance model shows that large KT (deep accumulation) amortizes per-iteration NOC overhead better than large CT (wide output), because each K iteration pays a fixed NOC latency cost.
- For DeepSeek V3 decode-mode matmuls (1x32 tiles), the matmul is **overwhelmingly NOC-bound** (7-20% compute utilization). Reducing `subblock_k` below the saturation point frees L1 without throughput impact -- the automatic system should exploit this.
- For LLaMA prefill-mode matmuls (32x32 tiles), all three dimensions interact. Configurations P1 through P3 show a **3x range in utilization** (10.7% to 34.6%) across feasible blockings for the same matmul.
- The `compute_subblock_w()` function constrains CT values based on dest register capacity (4-16 tiles). The automatic system should prefer CT values that are multiples of `max_dest` for efficient dest register utilization.
- Non-matmul ops have simpler blocking patterns: RMSNorm is constrained by the **8-tile dest limit** of `mul_reduce_scalar_tile`, softmax chunk size trades L1 for passes, and SDPA tree reduction adds modest L1 overhead per round.
- The **two-phase architecture** (AutoBlocker for op-specific blocking + AutoSizer for global budget allocation) separates performance optimization from correctness enforcement, with expert overrides (P7) available at every decision point.

## Source Files

- `blaze/ops/matmul/op.py` -- `k_num_tiles` from `in0_handle.num_pages` (line 48), `out_w_per_core` from weight shard (line 63), full blocking = KT=K, CT=N_per_core (implicit)
- `blaze/ops/dram_streaming_matmul/op.py` -- `subblock_k` heuristic (lines 128-132), `compute_subblock_w()` call (lines 136-140), triple-buffered weights (lines 161-172), `_get_max_page_size_and_num_pages()` NOC page sizing (lines 16-31)
- `blaze/ops/kn_sliced_matmul/op.py` -- `_derive_kn_parallel()` K/N splitting (lines 30-43), `_derive_kn_parallel_from_weights()` (lines 46-83)
- `blaze/utils.py` -- `compute_subblock_w()` dest register capacity (lines 80-113), `interpret_tile()` (lines 121-133), `interpret_tile_padded()` with MAX_TILES=8 (lines 136-170)
- `blaze/ops/rmsnorm/op.py` -- `num_tiles` derivation (line 121), `interpret_tile()` call (line 111)
- `blaze/ops/sdpa/op.py` -- `_TILE_SIZES` (lines 38-43), `MAX_TREE_REDUCTION_ROUNDS = 6` (line 33), `_compute_grid()` cores_per_head derivation (lines 68-89)
- `blaze/cb_engine.py` -- `MAX_CB_ID = 64` (line 65), `CBAssignment` with `first_use`/`last_use` (lines 137-138)

---

< Previous: [File 02 -- Automatic Page Count and Page Size](./02_automatic_page_count_and_page_size.md) | Next: [File 04 -- CB Compaction and Temporal Reuse](./04_cb_compaction_and_temporal_reuse.md) >

← [Chapter 6 -- Data Format Selection and Automatic Negotiation](../ch06_data_formats/) | [Chapter 8 -- Broadcasting and Multi-Dimensional Shape Alignment](../ch08_broadcasting/) →
