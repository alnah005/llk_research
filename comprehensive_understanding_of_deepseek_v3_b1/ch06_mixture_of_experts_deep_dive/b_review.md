# Agent B Review — Chapter 6

## Pass 1

**1. [01_moe_gating_and_routing.md, Section 6.1.8 — Gate MM weights incorrectly described as bfloat4_b]**

The chapter states: "They are stored in `bfloat4_b` format and block-sharded across the gate matmul cores."

The source code at `blitz_decode_weights.py` line 204 explicitly states: `gate_mm — BFP16, (7168, 256), WIDTH_SHARDED on 8 cores.` The `gate_mm_shard_bytes` property at line 278-279 computes bytes using `self.bfp16_tile_bytes` (2048 bytes per tile), not `bfloat4_b` (576 bytes per tile). The gate MM weights are BFP16 (bfloat16), not bfloat4_b. The table in Section 6.1.9 computes the correct size of 3.67 MB using bf16 math, which is internally consistent with BFP16 but contradicts the explicit "bfloat4_b" claim in Section 6.1.8.

**2. [01_moe_gating_and_routing.md, Section 6.1.8 — Incorrect dataclass name for weight config]**

The chapter states: "The `OProjGateMMFusedConfig` dataclass (`blitz_decode_weights.py`, lines 199--278) defines the layout."

The actual class name at line 198 of `blitz_decode_weights.py` is `O_PROJ_GATE_MM_RMSNORM_GAMMA_SingleDeviceOverlapSpec`, not `OProjGateMMFusedConfig`. The class exists at approximately the cited line range (lines 197-288), but the name is wrong.

**3. [01_moe_gating_and_routing.md, Section 6.1.3 — Incorrect line number for setup_sram_matmul]**

The chapter states: "The `setup_sram_matmul` function (`moe/op.py`, line 436) configures this..."

`setup_sram_matmul` is actually at line 2930 of `moe/op.py` (inside the `MoeOp` class). Line 436 is the docstring of `setup_eltwise_mul` (inside `MoeRoutedExpertOp`). The gate matmul setup call at line 1014 (`gate_mm_params = MoeOp.setup_sram_matmul(...)`) is correct, but the function definition line reference is wrong.

**4. [04_dram_streaming_matmul_deep_dive.md, Section 6.4.16 — Incorrect subblock_w for down_proj]**

The performance table in Section 6.4.16 states `subblock_w = 8` for down_proj. However, `per_core_N = 28` tiles for down_proj (7168 / 8 banks / 32 = 28). In `MoeRoutedExpertOp.setup_dram_matmul` (line 390-397 of `moe/op.py`), with `fp32_dest_acc_en=False`:

```python
max_subblock_w = 16 if per_core_n <= 16 else 8  # 28 > 16, so max_subblock_w = 8
subblock_w = max_subblock_w  # 8
while subblock_w > 1 and per_core_n % subblock_w != 0:
    subblock_w -= 1  # 28 % 8 = 4, decrement to 7; 28 % 7 = 0, stop
```

The actual `subblock_w` for down_proj is **7**, not 8. The same logic in `dram_streaming_matmul/op.py` (lines 217-220) confirms this: the loop decrements from 8 until it finds a divisor of 28, which is 7.

**5. [02_routed_expert_computation.md, Section 6.2.13 — Incorrect description of ReduceToOne Round 1 routing]**

The chapter states: "**Round 1 (LEAF -> ROOT3):** Each LEAF device sends its partial result to the ROOT3 device in the same column."

This is incorrect. The `get_reduce_device_role` function at lines 674-691 of `moe_routed_expert/op.py` assigns roles based on row distance from the root. With root at row 1:
- Row 0: LEAF
- Row 1: ROOT1 (col 0) and ROOT2 (col 1)
- Row 2: ROOT3
- Row 3: LEAF

But the destination routing at lines 1843-1847 of `moe_routed_expert/op.py` shows:

```python
if device_role == MESH_LEAF:
    if row == 0:
        dest_coord = ttnn.MeshCoordinate(row + 1, col)  # row 0 -> row 1 (ROOT1/ROOT2 row)
    else:  # row == 3
        dest_coord = ttnn.MeshCoordinate(row - 1, col)  # row 3 -> row 2 (ROOT3 row)
```

So LEAF at row 0 sends to **row 1** (the ROOT1/ROOT2 row), NOT to ROOT3 at row 2. Only LEAF at row 3 sends to ROOT3. Each LEAF sends to its adjacent inner row. The chapter's description ("Each LEAF device sends its partial result to the ROOT3 device in the same column") is only true for LEAFs at row 3, not for LEAFs at row 0.

Similarly, the Round 2 description ("ROOT3 sends to ROOT2") is also affected. ROOT3 at (2, col) sends to (root_row, col) per line 1849: `dest_coord = ttnn.MeshCoordinate(reduce_root_coord[0], col)`. For col 0 this is ROOT1, for col 1 this is ROOT2. So ROOT3 at col 0 sends to ROOT1 directly (not ROOT2), and ROOT3 at col 1 sends to ROOT2.

The actual reduction topology is:
- LEAF(0,0) -> ROOT1(1,0); LEAF(0,1) -> ROOT2(1,1)
- LEAF(3,0) -> ROOT3(2,0); LEAF(3,1) -> ROOT3(2,1)
- ROOT3(2,0) -> ROOT1(1,0); ROOT3(2,1) -> ROOT2(1,1)
- ROOT2(1,1) -> ROOT1(1,0)

This is a more nuanced tree than the simplified "LEAF->ROOT3->ROOT2->ROOT1" the chapter describes.

**No further issues flagged for this pass.**

---

## Pass 2

### Fix Verification

1. **Gate MM format (Issue 1):** VERIFIED — Section 6.1.8 now reads "stored in BFP16 (bfloat16) format and WIDTH_SHARDED across the gate matmul cores." This matches the source code at `blitz_decode_weights.py` line 204 (`gate_mm — BFP16, (7168, 256), WIDTH_SHARDED on 8 cores`) and line 278-279 where `gate_mm_shard_bytes` uses `self.bfp16_tile_bytes`. The table in Section 6.1.9 remains consistent (3.67 MB = 7168 * 256 * 2 bytes). Correct.

2. **Dataclass name (Issue 2):** VERIFIED — Section 6.1.8 now reads "The `O_PROJ_GATE_MM_RMSNORM_GAMMA_SingleDeviceOverlapSpec` class (`blitz_decode_weights.py`, lines 198--288)." The source code at line 198 shows `class O_PROJ_GATE_MM_RMSNORM_GAMMA_SingleDeviceOverlapSpec:` and the class body extends through the derived properties ending around line 288+. The name and line range match. Correct.

3. **Line number (Issue 3):** VERIFIED — Section 6.1.3 now reads "The `setup_sram_matmul` function (`moe/op.py`, line 2930)." The source code at `moe/op.py` line 2929-2930 shows the `@staticmethod` decorator followed by `def setup_sram_matmul(` at line 2930. Correct.

4. **subblock_w (Issue 4):** VERIFIED — The performance table in Section 6.4.16 now shows `subblock_w = 7` for down_proj. Confirmed via the `setup_dram_matmul` function in `moe/op.py` lines 390-397: with `fp32_dest_acc_en=False` and `per_core_n=28` (which is > 16), `max_subblock_w=8`; then the while loop finds `28 % 8 = 4` (not 0), decrements to 7, and `28 % 7 = 0`. The standalone `dram_streaming_matmul/op.py` lines 217-220 produces the same result via a slightly different path (`max_dest=8` for `not dst_full_sync_en and not fp32_dest_acc_en`). Both yield `subblock_w=7`. Correct.

5. **ReduceToOne topology (Issue 5):** VERIFIED — Section 6.2.13 now correctly describes the 3-round reduction. Round 1: "Each LEAF sends to the inner row in its same column. LEAF at row 0 sends to row 1 (ROOT1/ROOT2 row), LEAF at row 3 sends to row 2 (ROOT3 row)." This matches `moe_routed_expert/op.py` lines 1843-1847 where row 0 LEAFs use `dest = (row+1, col)` and row 3 LEAFs use `dest = (row-1, col)`. Round 2: "Each ROOT3 sends to the root row in its same column" matches line 1849 `dest = (reduce_root_coord[0], col)`. Round 3: "ROOT2 sends its column sum to ROOT1" matches line 1851 `dest = reduce_root_coord`. The device role assignment diagram, post-round state descriptions, and CB table are all consistent with the source. Correct.

### New Issues

**No new issues found.**

### Agent A Change Log — Pass 1

**Issue 1 fix (gate MM format) -- file 01:**
- Changed `bfloat4_b` to `BFP16 (bfloat16)` and `block-sharded` to `WIDTH_SHARDED` in Section 6.1.8.

**Issue 2 fix (dataclass name) -- file 01:**
- Changed `OProjGateMMFusedConfig` to `O_PROJ_GATE_MM_RMSNORM_GAMMA_SingleDeviceOverlapSpec` with correct line range (198--288).

**Issue 3 fix (line number) -- file 01:**
- Changed `setup_sram_matmul` line reference from 436 to 2930.

**Issue 4 fix (subblock_w) -- file 04:**
- Changed down_proj `subblock_w` from 8 to 7 in the performance table. `per_core_N=28`, and 28 % 8 != 0, so the loop decrements to 7.

**Issue 5 fix (ReduceToOne topology) -- file 02:**
- Rewrote the 3-round reduction description. Round 1: LEAFs send to adjacent inner row (row 0→row 1, row 3→row 2), not all to ROOT3. Round 2: ROOT3 sends to ROOT1/ROOT2 in root row, not to ROOT2. Round 3: ROOT2→ROOT1 unchanged. Updated diagram, CB table, and semaphore table to match.
