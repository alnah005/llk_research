# Open Questions and Future Work

This chapter surfaces four open questions that the research discovered but could not definitively answer, as well as two suggested follow-on research topics. Each question is grounded in a concrete code site or behavior gap; what is missing is either hardware-level documentation, an arch-team consultation, or a piece of dedicated research that exceeds the scope of this guide. The questions are deliberately phrased so they can be picked up later as standalone investigations.

**Prerequisites:** Chapter 1 (tiny-tile geometry and the `(num_faces, face_r_dim, face_c_dim)` triple), Chapter 2 (LLK data path and partial-face mode), Chapter 4 (TT-Metal API surface and `ttnn.Tile` / `ttnn.TileDescriptor`), Chapter 7 (trade-offs and per-arch differences), and Chapter 8 Sections 1–3 (the API audit, debugging tooling, and test-infrastructure gaps that motivate these questions).

---

## Q1: Why Is `face_c_dim` Fixed at 16?

Throughout the LLK and TT-Blaze codebase, the column dimension of a face is hard-coded to 16. The legal-geometry validator in `tensor_shape.h` makes this explicit:

```cpp
// tensor_shape.h:87-94
// validate_tensor_shape_tile_dependent_ops_:
// num_faces  in {1, 2, 4}
// face_r_dim in {1, 2, 4, 8, 16}
// face_c_dim == 16   // hard-coded, no other value accepted
```

The comment near `tensor_shape.h:356` reinforces this — all unpacker and packer implementations enforce `face_c_dim = 16`. The packer API signature does not parameterize the column dimension at all; the value 16 is baked into stride and address calculations downstream of `TensorShape`. By contrast, the row dimension is fully variable: `face_r_dim ∈ {1, 2, 4, 8, 16}` and `num_faces ∈ {1, 2, 4}` together generate the entire tiny-tile design space along the row axis.

This asymmetry is striking. The research literature and code consistently expose row-direction tiny tiles (8x32, 16x32, Flash MLA's `Q_TILE_HEIGHT=8`) but never column-direction tiny tiles. A natural question is whether the hardware *could* support `face_c_dim ∈ {1, 2, 4, 8, 16}` with the same generality, or whether there is a fundamental arch-level constraint pinning it at 16.

**Plausible explanations (none verified):**

1. **Source-register width alignment.** Tensix source registers (`SrcA`, `SrcB`) are 16 columns wide for FP16/BF16 datums (32-bit RISC-V lane × 16 columns = 512 bits per row). A column-tiny face would either waste source-register space or require a fundamentally different unpacker stride pattern.
2. **Unpacker stride logic.** The unpacker MOP configurations on Blackhole and Wormhole assume 16-column input granularity. Generalizing this would touch the entire unpack data path, not just the face-row counter.
3. **MVMUL FPU inner-loop granularity.** The matrix-vector multiply unit operates on 16-element vectors; faces are sized to match this. A face narrower than 16 would underfeed the FPU and waste cycles.
4. **Packer destination granularity.** L1 writes from the packer are 16-column-aligned; sub-16 column faces would need padding or a finer-grained write path.

None of these has been confirmed by inspecting hardware documentation or by arch-team correspondence. The fact that the constraint appears identically across `[BH]` and `[WH]` codebases — and that the validator asserts it with a comment rather than a citation — suggests it is structural rather than incidental, but the *reason* is opaque to the LLK layer.

**Conclusion.** This can only be answered by consulting the Tensix hardware team or the ISA documentation. If column-direction tiny tiles turned out to be cheap, RoPE-style and Flash MLA-style ops with narrow-column intermediates (e.g., the 128-dim KV head of MLA, which currently occupies 4 column-faces of 16) could see further L1 and compute savings. If they are expensive or impossible, the constraint should at least be *documented* in `tensor_shape.h` so future readers do not need to reverse-engineer it.

---

## Q2: Could We Have a Type-Safe `TinyTileDescriptor`?

The current legality check is purely runtime. `TensorShape` is a plain packed struct (`tensor_shape.h:44-74`) that exposes `constexpr` accessors:

```cpp
// tensor_shape.h:44-74 (abridged)
struct TensorShape {
    uint32_t num_faces;
    uint32_t face_r_dim;
    uint32_t face_c_dim;
    // ...
    constexpr uint32_t total_row_dim()     const;
    constexpr uint32_t total_col_dim()     const;
    constexpr uint32_t total_tensor_size() const;
    constexpr uint32_t total_num_faces()   const;
};
```

There is nothing in the type system that prevents constructing an illegal `TensorShape{num_faces=3, face_r_dim=5, face_c_dim=12}`. Legality is enforced by `validate_tensor_shape_tile_dependent_ops_` (`tensor_shape.h:87-94`), which fires at runtime — typically during kernel init, well after the host-side code has already shipped a CB descriptor referencing the bogus geometry. Symptoms in practice match the failure modes catalogued in Chapter 8 Section 1: `cb_wait_front` hangs from page-size mismatch, PCC divergence from wrong SFPU iteration counts, silent corruption from untested transpose + tiny + partial-face combinations.

A type-safe alternative would lift the legal-combination matrix into the template parameter list:

```cpp
// hypothetical
template <uint32_t NUM_FACES, uint32_t FACE_R_DIM>
struct TinyTileDescriptor {
    static_assert(NUM_FACES == 1 || NUM_FACES == 2 || NUM_FACES == 4);
    static_assert(FACE_R_DIM == 1  || FACE_R_DIM == 2  || FACE_R_DIM == 4 ||
                  FACE_R_DIM == 8  || FACE_R_DIM == 16);
    static constexpr uint32_t face_c_dim = 16;

    static constexpr uint32_t sfpu_iterations = NUM_FACES;  // derives from geometry
    static constexpr uint32_t total_rows      = NUM_FACES * FACE_R_DIM;
    static constexpr uint32_t tile_bytes      = NUM_FACES * FACE_R_DIM * 16 * sizeof(elem_t);
};
```

Benefits of such a type:

- **Illegal instantiations fail at compile time** rather than at kernel init.
- **SFPU iteration count is derived, not declared.** Today, Flash MLA's `<approx_mode, false, 2>` in `unified_kernels/matmul.hpp:173-176` must be kept in sync with the Python-side `Q_TILE_HEIGHT=8` by hand. With a `TinyTileDescriptor`, the iteration count becomes `Desc::sfpu_iterations` and the synchronization burden disappears.
- **CBHandle becomes a structural type.** Today `cb_handle.py:48` stores `tile_desc: object` as opaque pass-through metadata. If the descriptor were a typed template parameter, downstream consumers could `static_assert` against producer geometry, catching the producer/consumer mismatch class of bugs at compile/link time rather than discovering them via `cb_wait_front` hangs.

**Costs and caveats:**

- **API churn.** Every op that currently takes a `ttnn.Tile` and a `TensorShape` would need a templated counterpart. The transition is mechanical but touches a lot of files (every entry in `unified_kernels/`, every `op.py` that calls `ttnn.TileDescriptor`, every CB descriptor builder).
- **Template instantiation cost.** Each `(NUM_FACES, FACE_R_DIM)` pair generates a fresh template instantiation. The combinatorics are bounded (3 × 5 = 15 legal pairs) so this is tractable, but it does mean kernels become more aggressively monomorphized.
- **Python-side mirror.** TT-Blaze's `ttnn.TileDescriptor` would need a similar promotion — perhaps a `TinyTileDescriptor(num_faces, face_r_dim)` factory that returns a frozen, hashable type whose identity matches the C++ template instantiation. This is the natural follow-on to Chapter 8 Section 1's constexpr-helper proposal.

**Conclusion.** This is a long-term API evolution question, not a quick fix. The constexpr helpers proposed in Chapter 8 Section 1 are the minimum-viable-step toward this goal: they centralize the derivation of `sfpu_iterations` and tile-byte size from `(num_faces, face_r_dim)` without requiring a full template-type refactor. A `TinyTileDescriptor` type is the logical end state, but adopting it requires committing to a coordinated API migration across the LLK / TT-Metal / TT-Blaze stack.

---

## Q3: Are Partial-Face and Tiny-Tile the Same Hardware Path?

LLK exposes two distinct mechanisms for handling sub-16-row data:

1. **Tiny-tile mode** — the full tile has fewer than 4 faces (`num_faces ∈ {1, 2}`), with `face_r_dim = 16` per face.
2. **Partial-face mode** — the tile has the normal number of faces, but each face has `face_r_dim < 16` (e.g., `face_r_dim = 8` with `num_faces = 4` to express a 32-row "tall but short-face" layout).

Both manifest as flags in the unpack and pack MOP configuration. The LLK library treats them as variants of the same configuration knob, parameterizing the MOP loop bounds in either case. From the C++ API's perspective they are interchangeable representations of certain row counts — for example, a 16-row tile can be encoded as `num_faces=1, face_r_dim=16` (tiny) or `num_faces=2, face_r_dim=8` (partial-face with 2 faces).

**The open question:** is the underlying hardware execution path identical in these two cases, or are there measurable differences in cycle count, datapath utilization, or power?

Several reasons to suspect they might *not* be identical:

- **Face-iteration overhead.** The unpacker MOP iterates over faces. A 16-row payload spread across 2 partial faces incurs two face-setup transitions; the same payload as 1 full-row face incurs one. If face-transition cost is non-zero, the partial-face encoding pays a tax.
- **Source-register layout.** SrcA/SrcB are organized by face. Two half-height faces may occupy the same physical real estate as one full-height face but with different fill semantics — and downstream FPU/SFPU access patterns may differ.
- **Pack-path stride computation.** The packer computes destination strides from `(num_faces, face_r_dim)`. Two encodings of the same total row count *might* hit different code paths even if they produce the same L1 bytes.

This question is empirically answerable. `perf_math_matmul.py` already collects per-configuration cycle counts for the tiny-tile sweep generated by `sweep_tiny_tiles_matmul()` (`matmul_sweep.py:468-578`). Extending the harness to emit pairs of equivalent geometries — e.g., 16-row tile encoded as `(num_faces=1, face_r_dim=16)` versus `(num_faces=2, face_r_dim=8)` — and diffing the cycle counts would directly settle the question without needing arch-team input.

**Implications for the compiler / op author.** If the two paths are identical, the redundancy is harmless and one of the two encodings could be deprecated for clarity. If they are *not* identical, the compiler (BLAZE-NN tile geometry inference) should prefer whichever is cheaper, and ops that hand-author geometry (Flash MLA, RoPE) should be audited to ensure they have picked the cheaper encoding. Today this choice is made implicitly and without measurement.

**Conclusion.** This is the most tractable of the four open questions because it is answerable with infrastructure that already exists. It should be done as a one-off perf study before any further tiny-tile optimization work.

---

## Q4: Does Quasar's TDMA Make Tiny Tiles Uniform?

Quasar replaces the per-architecture unpack MOPs of `[BH]` and `[WH]` with TDMA descriptors (see `matmul_quasar_test.cpp` and `test_survey.md:136-152`). At a high level, the TDMA descriptor model unifies the description of how data is moved from L1 into source registers — rather than encoding the geometry into a chain of unpacker setup calls (`_llk_unpack_AB_matmul_init_<>()` and friends), the kernel programs a single descriptor that captures the row count, column count, face structure, and stride.

The open question is whether this descriptor model makes tiny-tile handling *more uniform* across architectures or whether it introduces new constraints specific to Quasar.

**Arguments for "more uniform":**

- A TDMA descriptor is data, not code. In principle, all legal `(num_faces, face_r_dim)` combinations can be expressed as descriptor field values without needing a separate kernel-init code path per geometry — the way `[BH]`/`[WH]` today require choosing among full-tile, partial-face, and tiny-tile init functions.
- If the descriptor format is general enough, the Python-side `ttnn.TileDescriptor` could map almost 1:1 to a TDMA descriptor, eliminating a layer of translation.

**Arguments for "new constraints":**

- TDMA descriptors have field widths and alignment requirements that are themselves new constraints. A descriptor field that is, say, 4 bits wide for `face_r_dim` only encodes 16 distinct values — possibly more restrictive, possibly identical to today's `face_r_dim ∈ {1, 2, 4, 8, 16}`.
- TDMA may impose new minimum-burst-size or alignment constraints that interact with tiny tiles in unexpected ways.
- Quasar tiny matmul is currently *untested* (Chapter 3 and Chapter 8 Section 3): the matmul sweep does not cover Quasar, and the standalone matmul test `matmul_quasar_test.cpp` does not exercise tiny geometries. So even if the descriptor model is general enough on paper, there is no empirical validation that it works.

The lack of Quasar tiny-tile test coverage is itself a finding: it means the question is genuinely open. Until Quasar tiny-tile tests exist (the proposed `test_math_matmul_quasar_tiny.cpp` from Chapter 8 Section 3), we cannot empirically say whether the TDMA model unifies behavior or introduces surprises.

**Conclusion.** This requires dedicated Quasar research. The most natural form is a guide modeled on this one but scoped to Quasar specifically — covering TDMA descriptor structure, mapping the `(num_faces, face_r_dim)` legal-combination matrix onto descriptor fields, porting the matmul sweep to Quasar, and documenting any geometry combinations that work on `[BH]`/`[WH]` but not on `[Q]` (or vice versa).

---

## Follow-On Research Topics

The four questions above are individually answerable but accumulate into two larger research directions worth pursuing as dedicated guides.

### [[tiny_tiles_on_quasar]]

**Scope:** comprehensive study of tiny-tile support in Quasar.

**Why it's needed.** Quasar is the next-generation architecture and the only one that uses TDMA descriptors instead of unpack MOPs. Every chapter of this guide is implicitly `[BH]`/`[WH]` because that is where the production usage lives today (Flash MLA, RoPE, DeepSeek V3 B1 ops). When tiny-tile patterns migrate to Quasar — which they will, since Flash MLA's `Q_TILE_HEIGHT=8` is too valuable to give up — there is no guide to consult.

**Suggested chapter outline (mirrors this guide):**

1. TDMA descriptor anatomy and how it encodes `(num_faces, face_r_dim, face_c_dim)`.
2. Legal-geometry validator on Quasar: is the legal set the same `{1, 2, 4} × {1, 2, 4, 8, 16}` as `[BH]`/`[WH]`, or does TDMA admit a different set?
3. Porting the matmul sweep (`matmul_sweep.py:468-578`) and SFPU/reduce/tilize tests to Quasar.
4. Mapping Flash MLA's 8x32 Q tile and RoPE's tiny intermediates to Quasar TDMA — what changes, what stays the same?
5. Per-geometry perf comparison: Quasar TDMA vs. `[BH]` unpack MOPs on equivalent workloads.

This guide would directly answer Q4 and supply the empirical baseline needed for any future Quasar-aware compiler work in BLAZE-NN.

### [[automatic_tile_geometry_inference_in_blaze_nn]]

**Scope:** eliminating the three-place declaration burden (Python `ttnn.Tile`, C++ CT args, kernel SFPU iteration count) by inferring tile geometry automatically from tensor shapes and compute constraints.

**Why it's needed.** Chapter 8 Section 1 catalogs the friction of authoring tiny-tile ops today: three declarations, manually kept in sync, with the validator (`tensor_shape.h:87-94`) only catching mistakes at runtime. The constexpr helpers and `TinyTileDescriptor` proposals from Q2 above are *type-level* solutions — they make manual declarations safer but do not eliminate them. The next step is a *compiler-level* solution: BLAZE-NN inspects tensor shapes, op-fusion boundaries, and L1 budgets, and emits the geometry automatically.

**Open sub-questions the guide would tackle:**

1. **The inference problem.** Given a fused op chain (e.g., Flash MLA) with input tensor shapes and L1 constraints, what is the optimal `(num_faces, face_r_dim)` for each intermediate CB? This is a constrained optimization problem with a small finite domain.
2. **Producer/consumer agreement.** Once a CB's geometry is inferred, it must be propagated to both the producer kernel and the consumer kernel. The CBHandle's `tile_desc` field (`cb_handle.py:48`) becomes the inference output, not user input.
3. **Override and pinning.** Some ops have hand-tuned geometry (Flash MLA's `Q_TILE_HEIGHT=8` was selected by analysis, not inference). The system needs a "pin this geometry" annotation for cases where author knowledge beats inference.
4. **Cost model.** Inference needs a cost model — and this is where Q3's partial-face vs. tiny-tile measurement directly feeds in. Without knowing the relative cost of `(1, 16)` vs. `(2, 8)`, the inference engine cannot rank alternatives.
5. **Validation.** The legal-combination matrix from Q1 / Q2 bounds the search space. If `face_c_dim` is ever generalized beyond 16 (Q1), the inference search space expands and the cost model has to evolve.

The dependency structure is worth noting: this research topic depends on Q1 (legal column geometries), Q2 (type-safe descriptors), and Q3 (cost-model data). Tackling automatic inference *before* answering Q3 would mean building an optimizer without a cost function. That ordering suggests the right sequence is: Q3 perf study first, then Q2 type-safe descriptors, then this guide.

---

## Closing

The open questions in this section are not gaps in this guide — they are gaps in the documented state of the system itself. Chapters 1 through 7 describe what is known and verifiable today. Chapter 8 Sections 1 through 3 describe what could be improved with the current understanding. This section describes what remains genuinely uncertain. Each question is annotated with a path to an answer (arch-team consultation, empirical perf study, or dedicated follow-on guide), and the two follow-on research topics provide a natural next-step scope.

The guide ends here. Future updates should either fold the answers to Q1–Q4 back into the relevant chapters once known, or replace the corresponding subsections with citations to the dedicated [[tiny_tiles_on_quasar]] and [[automatic_tile_geometry_inference_in_blaze_nn]] guides once those exist.
