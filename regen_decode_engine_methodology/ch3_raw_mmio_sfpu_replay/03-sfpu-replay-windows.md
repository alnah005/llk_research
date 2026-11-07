# SFPU Replay Windows

The SFPU is programmed end-to-end by an offline fragment compiler: [src/kernels.sfpu](../../../regen-glm5.3-flash/src/kernels.sfpu) holds Python-shaped *source that is never executed*; [tools/sfpu.py](../../../regen-glm5.3-flash/tools/sfpu.py) compiles each fragment to a static-inline C++ function of raw `TTI_*` SFPU instructions, emitting a generated include (`_out/kernels_sfpu.inc`) whose build rule's only inputs are `kernels.sfpu` and `sfpu.py` themselves [FACT, rules.py:102-103]; the brisc image `#include`s the result [FACT, src/brisc.cpp:10]. Fragments flagged `replay=True` additionally get an `<name>_instruction_count` constant — and a subset of them are *recorded* into the SFPU replay buffer and re-executed as **SFPU replay windows**: pre-compiled, count-pinned instruction sequences the interpreter replays into the SFPU pipe while it supplies only the runtime operands [FACT, tools/sfpu.py:28-32]. Why replay exists is issue economics: one REPLAY push re-issues a whole recorded body per loop iteration, where pushing the body's instructions word by word through the port would cost one push per instruction — the row-iterating recurrences are exactly the shape that rewards this [INFER, src/brisc.cpp:5660-5729]. This file walks source → compiler → include → replay semantics → consumers, and pins the correctness contract that makes replay safe.

**Scope:** covers Question 2, part 3 — the SFPU replay window mechanism and its consumers. Per-op performance (which window pays, where) is [Ch4](../ch4_dram_streaming_discipline/index.md)'s; the toolchain determinism argument built on the same pins is [Ch7](../ch7_build_determinism/index.md)'s. Terminology: "SFPU replay window" is the first of the guide's four pinned window senses ([chapter index](./index.md)); the `window` *field* of `TtGlmStepRecord` is unrelated and [Ch2 `04-read-ahead-rings.md`](../ch2_record_driven_execution/04-read-ahead-rings.md)'s.

## The source: fragments and their contracts

`kernels.sfpu` holds 123 decorated fragments: 116 `@sfpu(...)` entry points and 7 `@sfpu_helper()` inlined helpers (counts verified in-tree); 28 of the entry points declare `replay=True` [FACT, src/kernels.sfpu]. Each decorator is an *entry contract* — `inputs`/`outputs`/`preserve` bind vector values to live LRegs, `templates`/`offsets`/`addrmods`/`bits` declare parameters, `constants` name the exact bit patterns the caller must supply in read-only LRegs 11..14 [FACT, tools/sfpu.py:368-395]. Only LRegs 0..7 are allocated/writable; 9/10 are the architectural zero/one, 8 is a fixed constant, and 11..14 are caller-configured [FACT, tools/sfpu.py:12-13], with the compiler's fixed values $8=0x3f56594b$, $9=0$, $10=0x3f800000$ [FACT, tools/sfpu.py:386]. The ISA subset is enforced, not assumed: every call is checked against a fixed signature table — "unsupported SFPU instruction" — and unknown operations, arbitrary Python, loops, scheduling, fusion, and arithmetic reassociation are rejected at compile time [FACT, tools/sfpu.py:786-800, 870-887, 106-107]. Because entry contracts declare which broadcast constants are available, a fragment that needs an exp coefficient *names it in `constants`* rather than loading it [FACT, src/kernels.sfpu:420-421].

## The compiler: what it guarantees

`sfpu.py` is deliberately simple and deliberately strict (its own docstring: "This compiler does not prove hardware timing safety: callers still own engine hazards and configuration" [FACT, tools/sfpu.py:120-121]). What it does guarantee:

- **Range-checked instantiation.** Template parameters, AddrMod selectors, and every affine Dst-offset expression are wrapped in `static_assert` bounds checks (template < 8192, AddrMod < 4, offset in 0..8191) — an out-of-range instantiation fails the *build* [FACT, tools/sfpu.py:565-572].
- **Counted instructions.** Every emitted `TTI_*` line increments an instruction counter, including compiler-inserted copies and NOPs [FACT, tools/sfpu.py:574-579, 732-735].
- **The replay contract.** A `replay=True` body must be 1..32 compile-time TTI instructions with no runtime parameters and no scalar assignments — violations are compile-time errors ("replay bodies require 1..32 instructions") — and the compiler emits `static constexpr uint32_t <name>_instruction_count = ...;` for it [FACT, tools/sfpu.py:28-30, 732-735]. The caller "owns recording, replay storage, live LReg inputs, and RWC/configuration effects"; the compiler does not emit a REPLAY instruction [FACT, tools/sfpu.py:30-32].
- **BH silicon quirks lowered in the compiler, not the source.** A rotate or AND consuming a pending MAD result gets an inserted `SFPNOP` for Blackhole's missed dependency — "silicon reads the old value; the simulator does not" — and explicit `sfpnop()` in source is unsupported: "Mandatory instruction gaps belong in compiler lowering, not source code" [FACT, tools/sfpu.py:77-79, 641-643].
- **Deterministic rounding only.** `sfpround(fmt="bf16", rounding="nearest")` exposes SFPSTOCHRND's nearest conversion; "Stochastic rounding is intentionally unsupported, despite the hardware opcode's name" [FACT, tools/sfpu.py:70-73].

## The include and the pin

The generated include is a build product regenerated from exactly two inputs — `src/kernels.sfpu` + `tools/sfpu.py` [FACT, rules.py:102-103] — so regenerate-and-diff is the audit, and the binary embeds the fragments' instructions ([Ch7](../ch7_build_determinism/index.md) owns that chain). Two pin facts the mechanism needs: the fragments are compiled by the pinned toolchain `riscv-tt-elf-g++` from sfpi 7.76.0 (regen pin 7.76.0, GCC 15.1) [FACT, rules.py:12-13]; and the port separately re-verified that the instruction-count static_asserts still pass under sfpi 7.86.0 with the `xtttensixbh` march extension [cross-check, doc/research/regen_kernel_port.md (M0), per the briefing registry — a port-side result, clearly marked as such and not regen-internal evidence]. Why the pin is load-bearing (counts are compiler-version contracts) is [Ch7 `01-toolchain-and-recipes.md`](../ch7_build_determinism/01-toolchain-and-recipes.md)'s argument; this chapter only needs the fact of the pin.

## Replay semantics on device

The hardware primitive is one instruction: `TTI_REPLAY(start_idx, len, execute_while_loading, load_mode)` — opcode `0x04` [FACT, src/ckernel_ops.h:423-428]. The engine uses two modes:

- **Record (load):** `TTI_REPLAY(offset, count, 0u, 1u)` followed by the fragment's instructions captures `count` instructions into the replay buffer at `offset` without executing them — "record without executing or touching LRegs/Dst" [FACT, src/brisc.cpp:1104-1113 — `record_resident_replays()`].
- **Replay (execute):** `TTI_REPLAY(offset, count, 0u, 0u)` re-issues the recorded window — the same `count` instructions, byte-for-byte the recorded stream [FACT, src/brisc.cpp:2034, 5669 — execute-mode call sites].

A replay window is therefore a *loop body the hardware re-issues*: the interpreter supplies live-LReg operands (e.g. each state row's key and decay via runtime `SFPLOADI` pushes through the port) and pushes one REPLAY per iteration, with the fragment's `addrmod=1` store advancing Dst by four rows each pass [FACT, src/brisc.cpp:5695-5708; src/kernels.sfpu:50-60].

### The resident windows and the composition budget

At every image start, `hw_init` records three *resident* windows back-to-back at slot 0: attention-accumulate at offset 0, squares at `squares_replay`, scale-weight at `scale_weight_replay` — each offset the cumulative sum of the earlier bodies' counts — and establishes `gdn_replay` as the offset after them [FACT, src/brisc.cpp:62-64, 1104-1113, 1222-1223]. The composition is pinned by static_asserts [FACT, src/brisc.cpp:65-73]:

```c
static_assert(gdn_replay + sfpu_gdn_predict_body_instruction_count       <= 32u);
// ... five more of the same shape, one per asserted recurrence body ...
```

plus **one further `static_assert` whose four `&&` conditions cap the replayed KDA-chunk bodies at 32 each on their own** — scale, inverse, decay, gamma [FACT, src/brisc.cpp:71-73] (they record at slot 0, ad hoc, per use [FACT, src/brisc.cpp:8526, 8631, 8771]). The whole block states one invariant: the resident trio plus any one of the six asserted recurrence bodies fits the hardware's 32-entry replay budget — *enforced for exactly those bodies*; the compiler separately caps every replay body at 1..32 [FACT, tools/sfpu.py:732-735], so the two layers pin different things (per body vs per shared buffer) [INFER, src/brisc.cpp:62-73]. Two boundary facts the block does **not** cover: `sfpu_glm_moe_combine_body` records at `gdn_replay` (below) with **no entry among the 62-73 asserts**, and `sfpu_glm_hc_expand_body` records at slot 0 with its own local `static_assert(... <= 32u)` instead [FACT, src/brisc.cpp:18081, 16925] — the pin-coverage boundary is real and worth knowing when editing counts. One window even steps outside the SFPU fragment flow entirely: a flash-QSA matrix phase is hand-recorded at `gdn_replay` as a 7-instruction window over FPU/unpack instructions, with its own assert [FACT, src/brisc.cpp:9196-9197] — the replay buffer is not SFPU-only and not exclusively `sfpu.py`-generated.

### The borrower discipline

The replay buffer is a single shared resource. The Qwen images' exp and SiLU windows are *borrowers*: `qwen_replay_halves` "records a replay body of `length` instructions at slot 0" — overwriting the resident bodies — runs it over the halves of each 64-value Dst block, and the callers (`qwen_mix_replayed`, `qwen_silu_resident`, `qwen_silu_up_resident`) re-record the resident windows with `record_resident_replays()` before anything else uses them [FACT, src/brisc.cpp:6502-6520, 6536, 6551, 6569]. GLM decode does **not** borrow for SwiGLU: the GLM SwiGLU fragment runs inline, one invocation per half-block through the `glm_blocks` stepper, with no REPLAY at all [FACT, src/brisc.cpp:17333-17347; src/kernels.sfpu:1302-1315 — the fragment carries no `replay=True`].

## The 28 windows, by family

Counting `replay=True` in [src/kernels.sfpu](../../../regen-glm5.3-flash/src/kernels.sfpu) gives exactly 28 windows (of 123 fragments) — re-derived in-tree for this chapter [FACT, src/kernels.sfpu]. Every one of them, with its own line range:

| Family | Windows (fragment, lines) | What a replay computes |
| --- | --- | --- |
| GDN/KDA state bodies | `sfpu_gdn_predict_body` :5-15, `sfpu_gdn_update_body` :30-39, `sfpu_gdn_slice_predict_body` :43-50, `sfpu_kda_slice_predict_body` :53-61, `sfpu_gdn_slice_update_body` :76-85 | One recurrence step of a state row/block: decayed state, key/query mad, store with Dst advance [FACT, src/kernels.sfpu:5-85] |
| Resident decode windows | `sfpu_scale_weight_body` :165-176, `sfpu_squares_body` :206-213, `sfpu_attention_accumulate_body` :236-243 | Weight scaling, sum-of-squares accumulation, attention value accumulation — the three `hw_init`-recorded bodies [FACT, src/kernels.sfpu:165-176, 206-213, 236-243; src/brisc.cpp:1104-1113] |
| Qwen-image borrowers | `sfpu_qwen_exp_body` :459-462, `sfpu_qwen_mix_gate_body` :464-471, `sfpu_qwen_silu_gate_body` :477-485 | exp(−|x|) and sigmoid/SiLU gate passes, recorded into slot 0 "borrows the whole replay buffer" — after which the resident replays are re-recorded [FACT, src/kernels.sfpu:459-485; src/brisc.cpp:6502-6569] |
| MoE combine | `sfpu_glm_moe_combine_body` :553-557 | One 64-value combine block: "the slot at Dst rows 32 on times the gate in LReg3 plus the sum so far at rows 0 on"; "replayed, it advances Dst by four rows, so eight replays step to the next slot" [FACT, src/kernels.sfpu:550-557; src/brisc.cpp:18081-18086] |
| Qwen conv | `sfpu_qwen_conv_body` :628-637 | Four-tap convolution accumulation per block, replayed per block [FACT, src/kernels.sfpu:628-637] |
| Chunked GDN/KDA bodies | `sfpu_chunk_update_body` :738-741, `sfpu_chunk_split_body` :744-749, `sfpu_kda_chunk_scale_body` :766-775, `sfpu_kda_chunk_gamma_body` :778-781, `sfpu_kda_chunk_inverse_body` :805-816, `sfpu_kda_chunk_decay_body` :820-824, `sfpu_kda_chunk_scale_keep_body` :827-838, `sfpu_kda_chunk_zero_body` :841-844, `sfpu_kda_chunk_round_body` :847-858, `sfpu_kda_chunk_output_body` :951-961 | The chunked KDA recurrence: state decay-mad, BF16 rounding splits, cumulative gamma products, two-Newton reciprocals [FACT, src/kernels.sfpu:738-961] |
| KDA WY substitution | `sfpu_kda_wy_term_body` :913-920, `sfpu_kda_wy_delta_body` :922-933, `sfpu_kda_wy_finish_body` :935-945 | WY forward-substitution terms/deltas per token row, four LReg accumulators [FACT, src/kernels.sfpu:913-945] |
| GLM decode bodies | `sfpu_glm_dot_body` :1250-1258, `sfpu_glm_hc_expand_body` :1271-1286 | mHC stream dot products and expand blocks [FACT, src/kernels.sfpu:1250-1258, 1271-1286] |

Which decode ops consume which window at record level: the resident trio serves attention value accumulation, RMS-norm sum-of-squares (`glm_squares` replays `squares_replay` per block [FACT, src/brisc.cpp:2031-2034]), and per-element weight scaling in matvec paths (`sfpu_scale_to_bf16` replays `scale_weight_replay` [FACT, src/brisc.cpp:2060-2065]). The `gdn_replay` family serves the KDA/GDN recurrences — `gdn` records the predict body then replays it per state row, and the update body likewise ([`04-op-bodies.md`](./04-op-bodies.md) works one step end-to-end) [FACT, src/brisc.cpp:5785-5795] — the mHC dot (`glm_hc_dots` records and replays `sfpu_glm_dot_body` [FACT, src/brisc.cpp:16564-16572]), and the MoE combine (`glm_moe_combine_rows` records and replays `sfpu_glm_moe_combine_body` per slot, top-8 plus shared [FACT, src/brisc.cpp:18070-18090]). The borrowers serve only the Qwen command family; GLM decode does not route activations through them [FACT, src/brisc.cpp:6502-6569].

## The exp family and its consumers

One helper implements the exponential for nonpositive inputs — a range-restricted polynomial: $e^x = 2^n\, p(r)$ with $n$ from a magic-add round and $p$ a degree-5 polynomial in the remainder $r$, under an $x \le -20$ cutoff ("The integer predicate preserves the existing cutoff at x <= -20") [FACT, src/kernels.sfpu:394-418]. Its consumers are each pinned separately, because each inlines it at a different site:

| Consumer | What it computes | Anchor |
| --- | --- | --- |
| `sfpu_attention_exp_half` | softmax exponential: $e^{\text{score} - \text{maximum}}$ over a score half-tile | [FACT, src/kernels.sfpu:420-428] |
| `sfpu_qwen_sigmoid` | sigmoid: reciprocal of $1 + e^{-x}$ with a Newton correction | [FACT, src/kernels.sfpu:430-440 — call at 433] |
| `sfpu_kda_decay` | KDA forget-gate decay: $\exp(-5\,\mathrm{sigmoid}(\text{rate}(f + b)))$ | [FACT, src/kernels.sfpu:443-455 — calls at 447, 454] |
| `sfpu_qwen_exp_body` | $\exp(-|x|)$ stored per 32-value half (the replayed first pass of SiLU) | [FACT, src/kernels.sfpu:459-462 — call at 462] |
| `sfpu_swiglu_body` | Qwen-image SiLU of a gate clamped at $\pm 7$ (`0x40E00000`/`0xC0E00000`) | [FACT, src/kernels.sfpu:646-664 — call at 651] |
| `sfpu_glm_swiglu` | GLM decode SwiGLU: SiLU of the gate clamped above at 10, times the up value clamped to $[-10, 10]$ | [FACT, src/kernels.sfpu:1300-1315 — call at 1305] |

(The flash-attention path uses a *separate* base-2 helper `exp2_negative` — $2^y$ for nonpositive $y$ as a cubic, with caller-supplied coefficients "from the highest in LRegs 11, 12 and 14" [FACT, src/kernels.sfpu:969-981].) The exp consumers' coefficient registers are themselves managed: the firmware keeps "LReg11/12 = +/-7" — the Qwen-family SwiGLU clamps — "and LReg13/14 = log2(e)/ln(2)" as standing constants [FACT, src/brisc.cpp:902-908], and the attention softmax borrows the two clamp registers for its polynomial coefficients, restoring them afterwards (`sfpu_borrow_exp_constants` / `sfpu_restore_clamp_constants`) [FACT, src/brisc.cpp:4980-4993] — the LReg file is shared state, and the engine treats it with the same borrow/restore discipline as the replay buffer [INFER, src/brisc.cpp:4980-4993]. GLM decode's SwiGLU installs its own $\pm 10$ clamps in place of these ([`04-op-bodies.md`](./04-op-bodies.md)'s SwiGLU catalog entry owns the constants).

## Ownership fences

- Which windows a given op *uses* at record level is [`04-op-bodies.md`](./04-op-bodies.md)'s catalog; what they *cost* is [Ch4 `04-optimization-ledger.md`](../ch4_dram_streaming_discipline/04-optimization-ledger.md)'s.
- The count pins as *determinism gates* (why compiler drift must fail the build) are argued in [Ch7 `03-determinism-chain.md`](../ch7_build_determinism/03-determinism-chain.md).
- The startup re-recording of resident windows is one of the boot probes' dependencies; the probe set as a whole is [`05-correctness-discipline.md`](./05-correctness-discipline.md).

---

← [`02-instruction-port.md`](./02-instruction-port.md) | ↑ [chapter index](./index.md) | ↑ [guide root](../index.md)

**Next:** [`04-op-bodies.md`](./04-op-bodies.md)
