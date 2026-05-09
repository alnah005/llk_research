# Agent B Review — Chapter 1

## Pass 1

1. **File 01, Section 1.1.1 -- Standard MHA comparison uses wrong head count (off by 2x)**
   Location: The sentence "reducing the KV cache to a 576-element compressed vector per layer per position (versus $64 \times 256 = 16384$ elements with standard MHA)."
   Error: The full DeepSeek V3 model has 128 attention heads, not 64. The 64 value is the per-device count after TP=2 splitting. With 128 heads and `_KV_B_PROJ_HEAD_DIM = 256`, standard MHA would require $128 \times 256 = 32{,}768$ elements per position per layer. Using 16,384 understates the MLA compression ratio by a factor of 2 (actual: ~57x reduction, stated: ~28x reduction). Source: `prepare_weights.py` line 220 computes `num_heads = out_features // _KV_B_PROJ_HEAD_DIM` from the full HF shape `(32768, 512)`, yielding `num_heads = 128`.
   Fix: Change to "$128 \times 256 = 32{,}768$ elements with standard MHA".

2. **File 01, Section 1.1, parameter table -- Head count row conflates per-TP values with model-level parameters**
   Location: Row "Attention heads (queries) | 128 (64 Qnope + 64 Qrope)" with source `num_qnope_heads=64, num_qrope_heads=64 in PreSDPA.golden()`.
   Error: The model has 128 attention heads, each with both a nope component (dim 128) and a rope component (dim 64). The parenthetical "64 Qnope + 64 Qrope" and the cited code source are per-device values after TP=2 head splitting, not architectural parameters. As written, a reader would conclude there are two distinct head types (64 nope-only heads + 64 rope-only heads) that sum to 128, which is incorrect. The full-model W_q_b output dimension is 24576 = 128 * (128 + 64), not 12288.
   Fix: Change to "Attention heads (queries) | 128 | Full W_q_b HF shape (24576, 1536); per-TP=2 device: `num_qnope_heads=64, num_qrope_heads=64`" and add a clarifying note that each head has both a nope and a rope component.

3. **File 01, Section 1.5 -- ASCII grid diagram transposes the row and column axes**
   Location: The 13x10 ASCII grid diagram and its axis labels "Col 0..9" / "Row 0..12".
   Error: The actual device grid has 10 rows (y: 0-9) and 13 columns (x: 0-12), as confirmed by `GRID_X: int = 13`, `GRID_Y: int = 10` in `DOWN_PROJ_SingleDeviceSpec`, and the source code's own docstrings at `blitz_decode_weights.py` lines 439-440 ("Rows 0-3: A=cols{0-3,7-9}" and "Rows 4-9: A=cols{0-2,7-9}"). The diagram shows 13 "Rows" and 10 "Cols", swapping the axes. While the CoreCoord(x, y) references in the text are numerically correct (e.g., "Core (12, 9)" matches the code), the visual layout would mislead a reader into thinking the grid is tall-and-narrow (13 rows, 10 cols) when it is actually wide-and-short (10 rows, 13 cols). Anyone mapping the diagram to physical hardware or writing core-assignment code from the visual would get a transposed grid.
   Fix: Redraw the diagram with 10 rows (0-9) and 13 columns (0-12). Relabel accordingly. The [K] marker for kv_norm should appear at row 8, col 0 (not row 0, col 8), and the phantom cores should be in column 12 (not row 12).

4. **File 01, Section 1.1, parameter table -- Qrope head dimension row is derived from per-TP shape**
   Location: Row "Qrope head dimension | 64 | Implied by $12288 - 64 \times 128 = 64 \times 64$".
   Error: The derivation uses 12288, which is the per-device (TP=2) W_q_b output width, not the full model dimension (24576). While the final value of 64 is correct (qrope_head_dim is a per-head property and does not change with TP), the derivation is wrong as stated: using full-model values it should be $24576 - 128 \times 128 = 128 \times 64$, proving qrope_head_dim = 64. A reader checking the arithmetic with full-model dimensions would get a mismatch and doubt the result.
   Fix: Either derive from the full model ($24576 - 128 \times 128 = 128 \times 64$) or explicitly state the derivation uses per-TP=2 dimensions.

### Agent A Change Log

1. **Issue 1 (MHA comparison)**: Changed `$64 \times 256 = 16384$` to `$128 \times 256 = 32{,}768$` in Section 1.1.1.
2. **Issue 2 (Head count table row)**: Rewrote the "Attention heads (queries)" row to read "128 (each head has both a nope and a rope component)" with source "Full W_q_b HF shape (24576, 1536); per-TP=2 device: `num_qnope_heads=64, num_qrope_heads=64`".
3. **Issue 3 (ASCII grid diagram)**: Redrew the grid with 10 rows (y: 0-9) and 13 columns (x: 0-12). Phantom cores now appear in column 12 (rows 0-8), KV norm at row 8 col 0, mcast/gather at col 12 row 9. Legend updated with coordinate annotations.
4. **Issue 4 (Qrope derivation)**: Changed derivation from per-TP `$12288 - 64 \times 128 = 64 \times 64$` to full-model `$24576 - 128 \times 128 = 128 \times 64$` with note "full-model W_q_b output is 24576".

## Pass 2

1. **File 01, Section 1.1.1 -- $W_{q\_b}$ shape uses per-TP=2 dimensions in an architectural description**
   Location: The sentence "where $W_{q\_a}$: $(7168, 1536)$ and $W_{q\_b}$: $(1536, 12288) = (1536,\; 64 \times 128 + 64 \times 64)$."
   Error: Section 1.1.1 describes the MLA mechanism at the model-architecture level, yet $W_{q\_b}$ is given as $(1536, 12288)$ -- the per-device shape after TP=2 splitting. The full-model $W_{q\_b}$ is $(1536, 24576)$ (HF shape $(24576, 1536)$ transposed; source: `prepare_weights.py` line 253 comment and line 313). The decomposition uses 64 heads (per-device) instead of 128 (full model). Note that $W_{q\_a}$: $(7168, 1536)$ on the same line IS the full-model shape (it is not TP-split), so the two shapes are at inconsistent abstraction levels. Pass 1 already corrected the Qrope derivation from per-TP to full-model values, but this $W_{q\_b}$ shape was not updated to match.
   Fix: Change to "$W_{q\_b}$: $(1536, 24576) = (1536,\; 128 \times 128 + 128 \times 64)$" or explicitly note that the shape shown is per-device after TP=2 slicing.

2. **File 01, Section 1.1.2 -- MoE gate formula implies bias enters normalized scores; it does not**
   Location: Step 1 reads "$\text{score}_i = \sigma(\mathbf{x} \cdot W_\text{gate})_i + \text{bias}_i$", and Step 3 normalizes $\text{score}_i$.
   Error: The bias (`e_score_correction_bias`) is used only to rank experts for top-8 selection; it does NOT enter the normalized weights. The `deepseek_moe_gate` golden (lines 43-58 of `micro_ops/deepseek_moe_gate/op.py`) shows: (a) `bias_scores = scores + bias_tensor` is used for sorting and selection, but (b) the original unbiased `scores` are gathered for the selected experts and fed to the normalization denominator. As written, a reader implementing the gate from the document would add the bias to the scores before normalizing, producing wrong expert weights. The formula should separate the selection criterion (sigmoid + bias) from the normalization input (sigmoid only).
   Fix: Rewrite to clarify that Step 1 produces two quantities: $s_i = \sigma(\mathbf{x} \cdot W_\text{gate})_i$ (the score) and $r_i = s_i + \text{bias}_i$ (the ranking value). Step 2 selects the top-8 by $r_i$. Step 3 normalizes using $s_i$ (not $r_i$).

### Agent A Change Log — Pass 2

1. **Issue 1 (W_q_b shape)**: Changed `$W_{q\_b}$: $(1536, 12288) = (1536,\; 64 \times 128 + 64 \times 64)$` to `$W_{q\_b}$: $(1536, 24576) = (1536,\; 128 \times 128 + 128 \times 64)$` in Section 1.1.1, making it consistent with the full-model shape (matching W_q_a which was already full-model).
2. **Issue 2 (MoE gate bias in normalization)**: Rewrote Section 1.1.2 gating steps to separate the unbiased score $s_i = \sigma(\mathbf{x} \cdot W_\text{gate})_i$ from the ranking value $r_i = s_i + \text{bias}_i$. Step 2 now explicitly selects top-8 by $r_i$. Step 3 now explicitly normalizes using $s_i$ (not $r_i$), and the formula variable was changed from $\text{score}_i$ to $s_i$ / $\hat{s}_i$ accordingly. Verified against `micro_ops/deepseek_moe_gate/op.py` golden function (lines 43-58).
