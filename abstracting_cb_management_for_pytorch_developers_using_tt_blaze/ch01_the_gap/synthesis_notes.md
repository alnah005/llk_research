# Synthesis Notes: Chapter 1

## Base Version

**V5** was used as the consensus base (ranked #1 by 3/5 evaluators). The final chapter cherry-picks the strongest treatment of each section from across all 5 versions and applies all required technical fixes.

## File-by-File Source Mapping

### File 01: `01_pytorch_tensor_model.md`

| Section | Primary Source | Rationale |
|---------|---------------|-----------|
| Opening paragraph and "What you will learn" | V2 | Cleanest introduction with explicit "four contracts" framing |
| "What a PyTorch Tensor Is" | V1 + V3 | V1 has the best property table; V3 adds the stride offset formula |
| "Contract 1-4" framing | V2 | Evaluators praised V2's explicit numbering of contracts as "Contract 1-4" |
| Contract code examples | V1 + V2 | V1 has the richest code examples; V2 has cleaner structure |
| "How Developers Think" GatedMLP example | V1 | Best "notice what is absent" analysis |
| nn.Linear walkthrough | V5 + V1 | V5's step numbering is clearest; V1 adds the summary table of hidden decisions; V5's note about nn.Linear(513, 1023) is a strong pedagogical point |
| "Expectations That Transfer" table | V5 | Unique to V5; all evaluators noted this as a strong framing device |
| Key Takeaways | Synthesized | Combined best bullets from V1, V2, V5 |

### File 02: `02_tenstorrent_tile_cb_model.md`

| Section | Primary Source | Rationale |
|---------|---------------|-----------|
| Hardware architecture diagram | V1 | Most detailed ASCII diagram showing BRISC/NCRISC/TRISC within cores |
| Component table | V5 | V5 uniquely mentions five RISC processors (BRISC, NCRISC, TRISC0/1/2) |
| CB semantics | V1 + V5 | V1 has the best protocol diagram; V5 has the clearest four-operation table |
| CB access modes | V2 | Cleanest treatment of FIFO vs DIRECT_ADDRESS with CBAccessMode code |
| "Single Tile Through the Pipeline" | V5 | Unique to V5; all evaluators praised this as the standout pedagogical feature |
| Tile geometry and face order | V1 | Best face-order explanation connecting to FPU datapath |
| interpret_tile | V4 | Cleanest explanation without V5's confused narrative walkthrough |
| interpret_tile_padded | V1 | Best mathematical formulas (LaTeX) for padding |
| CBHandle fields | V4 | Best treatment including `eq=False` rationale and integer coercion |
| Data formats | V2 + V1 | V2 includes `_TILE_SIZES` from sdpa/op.py; V1 has correct ~1088/~576 sizes |
| BFP formats | V1 (corrected) | Fixed descriptions: "8-bit mantissa with shared block exponent" (not V5's "5-bit mantissa + 3-bit exponent") |
| _TTNN_DTYPE_MAP | V4 | Unique inclusion from cb_engine.py |
| 64 CB limit | V2 | Includes CircularBufferIdManager and CBEngine warning code |

**Technical fixes applied:**
1. bfloat8_b tile size corrected to ~1,088 bytes (V5 had ~1024)
2. bfloat4_b tile size corrected to ~576 bytes (V5 had ~512)
3. BFP bit layout fixed to "8-bit mantissa with shared block exponent" for bfloat8_b and "4-bit mantissa with shared block exponent" for bfloat4_b
4. interpret_tile(768) walkthrough removed (V5's confused narrative); replaced with V4's clean code-speaks-for-itself approach

### File 03: `03_the_manual_plumbing_burden.md`

| Section | Primary Source | Rationale |
|---------|---------------|-----------|
| Matmul.emit() walkthrough | V5 | Clearest annotation style with decision numbering |
| 12 decisions table | V5 + V1 | V5's table is most compact; V1 adds PyTorch equivalent column |
| DenseMLP analysis | V5 | Per-micro-op decision count table (81 total) is the most precise; evaluators noted V1's ~132 and V2's ~144 are less defensible |
| SwigluOp example | V5 | Unique to V5; provides valuable breadth |
| PaddedRMSNorm case study | V1 + V2 | V2 provides the specific 5 CBs / 22+ CT args numbers |
| Side-by-side table | V5 + V2 | V5's table with "What the Comparison Reveals" paragraph; V2 adds more rows |
| Error taxonomy | V4 | Structured table with Symptom and Root Cause columns |
| PyTorch comparison code | V5 | Most concise and effective 5-line SimpleMLP |

**Technical fix applied:**
- Removed unsubstantiated "Full attention block: ~200+" and "Transformer layer: ~400+" from V1's complexity multiplier table; kept only well-documented DenseMLP example

### File 04: `04_abstraction_goals_and_scope.md`

| Section | Primary Source | Rationale |
|---------|---------------|-----------|
| Vision code examples | V5 | Best manual workflow code (showing full weight preprocessing) contrasted with proposed one-liner |
| Comparison table | V5 | Unique "Override Available?" column |
| Scope boundaries | V4 + V5 | V4's structured tables + V5's chapter references |
| Q1-Q12 mapping | V1 + V5 | V1's abstraction component names + V5's concise format |
| Compose-not-replace | V1 + V4 | V1 has the most detailed rationale subsections ("Why Not Replace FusedProgram?" etc.); V4 has the cleanest WRAP/EXTEND/UNCHANGED table |
| Developer Journey | V2 | Three-level progressive disclosure (Level 1/2/3) unique to V2 |
| Escape Hatch Hierarchy | V5 | Three-level override table unique to V5 |
| "What Success Looks Like" | V5 | Five concrete success criteria unique to V5 |
| Boundary principle diagram | V2 + V5 | Combined for clarity |

## Cross-Version Quality Summary

| Dimension | Best Source |
|-----------|-----------|
| Mathematical precision | V1 (LaTeX formulas) |
| Pedagogical concreteness | V5 (tile pipeline walkthrough) |
| Code reference breadth | V2 (_TILE_SIZES, CircularBufferIdManager) |
| Error taxonomy | V4 (structured Symptom/Root Cause table) |
| Design rationale depth | V1 ("Why Not Replace?" subsections) |
| Developer journey framing | V2 (Level 1/2/3 progression) |
| Escape hatch design | V5 (three-level hierarchy) |
| Compound op breadth | V5 (SwigluOp addition) |
| Formatting compliance | All versions (equal) |

## Technical Corrections Applied

1. **bfloat8_b tile size**: Changed from ~1024 (V5) to ~1,088 bytes. Source: `_TILE_SIZES` from `blaze/ops/sdpa/op.py`.
2. **bfloat4_b tile size**: Changed from ~512 (V5) to ~576 bytes. Same source.
3. **BFP bit layout**: Changed from "5-bit mantissa + 3-bit shared exponent" (V5) to "8-bit mantissa with shared block exponent" for bfloat8_b; from "3-bit mantissa + 1-bit shared exponent" to "4-bit mantissa with shared block exponent" for bfloat4_b.
4. **interpret_tile(768) walkthrough**: Removed V5's confused narrative ("But actually...the actual logic checks..."). Used V4's clean approach of showing the function and one brief clarifying sentence.
5. **DenseMLP decision counts**: Used V5's per-micro-op table (81 total) rather than V1's rough "12 ops x 12 decisions = ~132" or V2's "~144". The per-micro-op breakdown is more defensible.
6. **Attention block / transformer layer estimates**: Removed V1's unsubstantiated "~200+" and "~400+" estimates. Kept DenseMLP as the primary substantiated example.
