# Agent B Review — Chapter 3

## Pass 1

**1. [01_compute_micro_ops.md, Section 3.1.2 rmsnorm — Compute Config: incorrect math fidelity]**

The chapter states:

> Math fidelity: `HiFi4` (higher precision for reduction)

The actual source code at `micro_ops/rmsnorm/op.py` line 158 sets:

```python
math_fidelity=ttnn.MathFidelity.LoFi,
```

This is `LoFi`, not `HiFi4`. A downstream reader implementing or tuning rmsnorm would select the wrong fidelity, potentially impacting both performance (HiFi4 is slower) and their understanding of the precision trade-offs in the pipeline. Fix: change "HiFi4" to "LoFi" in the Compute Config section.

**2. [01_compute_micro_ops.md, Section 3.1.3 rope — Compute Config: incorrect math fidelity]**

The chapter states:

> Math fidelity: `HiFi4`
> FP32 accumulation: disabled

The actual source code at `micro_ops/rope/op.py` line 270 sets:

```python
math_fidelity=ttnn.MathFidelity.LoFi,
```

This is `LoFi`, not `HiFi4`. Same class of error as item 1. Fix: change "HiFi4" to "LoFi" in the Compute Config section.

**3. [01_compute_micro_ops.md, Section 3.1.3 rope — Data Flow Diagram: steps 2 and 3 are swapped]**

The Data Flow Diagram claims:

```
Step 1: rotated = input @ trans_mat -> CB 24
Step 2: cos_prod = input * cos     -> CB 25
Step 3: sin_prod = rotated * sin   -> CB 26
Step 4: out = cos_prod + sin_prod  -> CB 16 (output)
```

The actual kernel execution order in `unified_kernels/rope.hpp` (lines 196-245) is:

- Step 2 (actual): `sin_prod = rotated * sin -> CB 26` (lines 196-209, `mul_tiles_bcast<ROW>(rotated_in_interm_cb, sin_cb, ...)`)
- Step 3 (actual): `cos_prod = input * cos -> CB 25` (lines 214-225, `mul_tiles_bcast<ROW>(in_cb, cos_cb, ...)`)

The sin multiplication happens BEFORE the cos multiplication. The TRISC Roles row in the same section correctly states "matmul(x, trans_mat), eltwise_mul(rotated, sin), eltwise_mul(x, cos), add" — which matches the code — but the Data Flow Diagram contradicts it. A developer tracing execution through the diagram would get the pipeline ordering wrong. Fix: swap Steps 2 and 3 in the Data Flow Diagram to match the actual kernel order (and to be consistent with the TRISC Roles description in the same section).

### Agent A Change Log

**Issue 1 fix (rmsnorm math fidelity) -- `01_compute_micro_ops.md`:**
- Section 3.1.2 Compute Config: Changed `HiFi4` to `LoFi` per `rmsnorm/op.py` line 158.

**Issue 2 fix (rope math fidelity) -- `01_compute_micro_ops.md`:**
- Section 3.1.3 Compute Config: Changed `HiFi4` to `LoFi` per `rope/op.py` line 270.

**Issue 3 fix (rope diagram step order) -- `01_compute_micro_ops.md`:**
- Section 3.1.3 Data Flow Diagram: Swapped Steps 2 and 3 so sin_prod (rotated * sin → CB 26) comes before cos_prod (input * cos → CB 25), matching the actual kernel execution order in `rope.hpp` lines 196-225.

## Pass 2

**1. [02_data_movement_micro_ops.md, Section 3.2.3 create_q_heads — CB Layout table: incorrect output CB index]**

The chapter's CB Layout table states:

> | out (receivers) | 16 | Tiled output | Sharded tensor |

The actual source code at `micro_ops/create_q_heads/op.py` line 265 sets:

```python
out_cb = 3  # Output CB (tilized data)
```

The output CB is index 3, not 16. CB 16 is the output index used by `tilize_8x32` (a different micro-op), and the confusion likely stems from conflating the two. The same section also lists the intermediate receiver CB as `--` (no index), but the code assigns it `receiver_in_cb = 2` (line 264) and omits the separate QROPE sender CB at index 1 (`qrope_cb = 1`, line 263). A reader tracing data through the create_q_heads kernel using the CB Layout table would look at the wrong circular buffers. Fix: update the CB Layout table to show `out` at index 3, `interm` at index 2, and add a row for the QROPE sender CB at index 1.

**No other issues found -- chapter approved with the above correction.**

### Agent A Change Log — Pass 2

**Issue 1 fix (create_q_heads CB Layout) -- `02_data_movement_micro_ops.md`:**
- Section 3.2.3 CB Layout table: Changed output CB from index 16 to index 3, intermediate CB from `--` to index 2, and added QROPE sender CB row at index 1. Per `create_q_heads/op.py` lines 263-265.
