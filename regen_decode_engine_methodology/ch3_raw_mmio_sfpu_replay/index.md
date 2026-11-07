# Chapter 3 — Raw-MMIO Compute and SFPU Replay

How the engine drives Tensix compute directly — cfg register writes at `0xFFEF0000`, the instruction port at `0xFFE40000`, and **[SFPU replay windows](./03-sfpu-replay-windows.md)** compiled offline by `tools/sfpu.py` — with no llk, no circular buffers, and the TRISCs never running programs of their own; plus the correctness disciplines that make programming the hardware this low-level safe. [Chapter 2's](../ch2_record_driven_execution/index.md) interpreter is the substrate all of this executes on; this chapter is what that interpreter actually *does* to the Tensix engines.

**Scope:** covers Question 2 (raw-MMIO compute and SFPU replay, with the correctness disciplines) of the guide's seven. On [Ch1's token walk](../ch1_engine_at_a_glance/02-anatomy-of-a-token.md) this chapter owns stage 2 — "op bodies over raw MMIO, SFPU replay windows": everything an **[op body](./04-op-bodies.md)** does between the interpreter's dispatch and its result. Primary source: the read-only regen tree at `/home/alex/mpi-shfs/tenstorrent/regen-glm5.3-flash` — all `src/...`, `notes/...`, and `README.md` citations are relative to that root; links resolve per the [Ch1 link convention](../ch1_engine_at_a_glance/index.md). Every line anchor in this chapter was re-verified in-tree before writing; fragment counts were re-derived by count.

## The thesis: the engine is its own runtime

Raw MMIO is a *methodology choice*, not a workaround: with one persistent interpreter per core ([Ch2 `02-interpreter-loop.md`](../ch2_record_driven_execution/02-interpreter-loop.md)), every configuration word and every instruction can be issued in-line by the interpreter itself — deterministic, dispatch-free, and free of circular-buffer machinery — which is exactly what record-driven execution needs [INFER, src/brisc.cpp]. The price is that everything llk and the ckernel runtime would otherwise guarantee (correct configuration sequences, instruction accounting, safe re-configuration) must be re-guaranteed by the engine itself. That re-guarantee is not aspirational: it is a build-time and boot-time discipline — instruction counts pinned by `static_assert`, window compositions pinned to the hardware's 32-instruction replay budget [FACT, src/brisc.cpp:62-73], and the raw-MMIO machine state probed before the record loop ever starts (the SrcA bank test and siblings, [`05-correctness-discipline.md`](./05-correctness-discipline.md)). Raw MMIO is only viable *with* those disciplines; without them it is silent silicon corruption.

## The fences this chapter keeps

- **Op MECHANISM lives here; op PERFORMANCE lives in Ch4.** [`04-op-bodies.md`](./04-op-bodies.md) catalogs what each command id computes and how it is composed; every per-op timing, the optimization ledger, and the DRAM-busy argument are [Ch4 `04-optimization-ledger.md`](../ch4_dram_streaming_discipline/04-optimization-ledger.md)'s and are never restated here.
- **The oracle's ROLE appears here** (as the correctness consumer in [`05-correctness-discipline.md`](./05-correctness-discipline.md)); its MECHANICS live once in [Ch6 `05-cpu-oracle.md`](../ch6_host_driver_weights_oracle/05-cpu-oracle.md).
- **NoC-ID budgeting, strip claims-ahead, and window rates** are [Ch4](../ch4_dram_streaming_discipline/index.md)'s; this chapter says only *that* op bodies issue NoC moves and that readers own their own synchronization.
- **The toolchain pin and image budgets** are [Ch7](../ch7_build_determinism/index.md)'s; this chapter states only the two pin facts its mechanism claims need and defers the determinism argument.
- **Port interpretation is [Ch8](../ch8_tt_metal_port/index.md)'s.** [`06-what-no-llk-means.md`](./06-what-no-llk-means.md) builds the engine-vs-llk trade table as a declared Ch8 input; it does not argue port decisions.

## Window terminology, pinned

"Window" is always qualified. The guide's three pinned window senses (SFPU replay, DRAM prefetch, PCIe BAR) are defined once in [Ch1's glossary](../ch1_engine_at_a_glance/index.md); this chapter owns the first. The one sense Ch1's table does not cover:

| Sense | What it is | Owned by |
| --- | --- | --- |
| Record `window` flag | A *field of `TtGlmStepRecord`* — the read-ahead restart flag of the record ring, a fourth, unrelated use of the word. | [Ch2 `04-read-ahead-rings.md`](../ch2_record_driven_execution/04-read-ahead-rings.md) |

## Files in this chapter

| File | What it gives you |
| --- | --- |
| [`01-mmio-register-layer.md`](./01-mmio-register-layer.md) | The register layer of [src/fw.h](../../../regen-glm5.3-flash/src/fw.h): bases, cfg-word writes, the per-op config sequencing, the live-config cache, and BAR0 exposure. |
| [`02-instruction-port.md`](./02-instruction-port.md) | Driving compute through `0xFFE40000`: the two emission paths, the completion handshakes, operand setup with plain L1 stores, "TRISCs idle" operationally, and why there are no CBs. |
| [`03-sfpu-replay-windows.md`](./03-sfpu-replay-windows.md) | The SFPU pipeline end-to-end: [src/kernels.sfpu](../../../regen-glm5.3-flash/src/kernels.sfpu) → [tools/sfpu.py](../../../regen-glm5.3-flash/tools/sfpu.py) → `kernels_sfpu.inc` → pinned, replayed windows, with the complete 28-window census. |
| [`04-op-bodies.md`](./04-op-bodies.md) | The op bodies as compositions of cfg writes + replay windows + NoC moves, with the command-id menu and three worked examples at record level. |
| [`05-correctness-discipline.md`](./05-correctness-discipline.md) | The safety net in lieu of llk: count pins, deterministic sizing, startup probes, the smoke gate, and the oracle's role — as a checklist. |
| [`06-what-no-llk-means.md`](./06-what-no-llk-means.md) | The full no-llk/no-CB contract as an argued trade, with the engine-vs-llk table Ch8 reuses. |

## Evidence conventions

Tag rules per [Ch1 `05-measured-envelope-and-evidence.md`](../ch1_engine_at_a_glance/05-measured-envelope-and-evidence.md) (one tag per claim, no hybrids, superlatives never [FACT]); this chapter is mostly source reading — `[FACT, src/...]` for structural facts, `[INFER, src/...]` for anything derived. The three [MEAS] claims: the smoke gate's yield ([README.md §4](../../../regen-glm5.3-flash/README.md)), the probe back-story ([notes/glm-stability.md](../../../regen-glm5.3-flash/notes/glm-stability.md)), and one diagnostic-cost figure from a source comment. Command ids follow [Ch2's frozen map](../ch2_record_driven_execution/01-step-records.md) and are restated here only as the op-body menu's index.

---

**Next:** [`01-mmio-register-layer.md`](./01-mmio-register-layer.md)
