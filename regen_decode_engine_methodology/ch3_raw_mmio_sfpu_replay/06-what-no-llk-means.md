# What No-llk Means: The Contract as a Trade

This chapter has shown the engine programming Tensix with nothing but cfg writes, instruction pushes, and replay windows — no llk, no ckernel runtime, no circular buffers. This file states that contract whole, as an argued trade: what regen gives up by leaving the llk abstraction, what it gains, and the price of each line. The result is one comparison table — the **engine-vs-llk table** — that [Ch8 `02-load-bearing-vs-incidental.md`](../ch8_tt_metal_port/02-load-bearing-vs-incidental.md) consumes verbatim as an input to the stays/comes/new analysis; this chapter builds the table but does not argue port decisions.

**Scope:** covers Question 2, part 6 — the no-llk/no-CB trade, mechanism-side. Op MECHANISM is [`04-op-bodies.md`](./04-op-bodies.md)'s and op PERFORMANCE is [Ch4](../ch4_dram_streaming_discipline/index.md)'s; nothing here cites a measured speedup as an argument for the trade — the ledger that shows where the design paid off is Ch4's, and the correctness net behind the trade is [`05-correctness-discipline.md`](./05-correctness-discipline.md)'s.

## What regen gives up

Leaving llk means regen owns, by hand, five things a conventional stack gets from the library:

- **CB flow control.** No CBs anywhere; the producer/consumer safety a CB constructs must be re-created by op-body waiting discipline — the mechanism and its evidence are the table's data-movement row below [INFER, src/brisc.cpp:317-329].
- **A maintained math library.** Every operation on data is either engine code or a fragment compiled by `sfpu.py`; there is no library of tested kernels to inherit, so every numerical idiom (an exp polynomial, a BF16 rounding rule) is regen's own and must be validated by regen's own oracle [INFER, src/kernels.sfpu; src/glm.cpp].
- **Portability across architectures and stacks.** The engine hardcodes Blackhole register layout, quirks (the missed SFPU MAD dependency gets an inserted NOP), and even per-image constants like `NOC1_BASE` [FACT, src/fw.h:16-21; src/brisc.cpp:694; tools/sfpu.py:77-79]. A different architecture means a new layer, not a config [INFER, src/fw.h].
- **TRISC issue capacity.** One BRISC issues all engine work; the other RISCs of a Tensix core execute nothing regen wrote, so per-core instruction issue is a single sequential stream — a resource a conventional stack would spend on parallel unpack/math/pack kernels; regen spends its parallelism on team helpers and NoC readers instead ([Ch2 `03`](../ch2_record_driven_execution/03-lead-helper-teams.md), [Ch4](../ch4_dram_streaming_discipline/index.md)) [INFER, src/brisc_entry.S:8-10; src/brisc.cpp].
- **Tested abstractions between engine and hardware.** The only things between a record and silicon are the disciplines of [`05-correctness-discipline.md`](./05-correctness-discipline.md); when one is violated, nothing type-checks the violation away [INFER, src/brisc.cpp:62-73].

## What regen gains

- **No reconfig churn.** The engine tracks its live unpack/math/pack configuration and skips rewriting it when back-to-back ops share it — "back-to-back matvecs skip rewriting it. Every setter of those registers forgets it" [FACT, src/brisc.cpp:141-149, 1410-1427]. Configuration is written exactly when an op needs a different one, not defensively per kernel [INFER, src/brisc.cpp:141-186].
- **No CB sync races — because no CBs.** The class of bug where a circular buffer overflows, a producer outruns slots, or a stale slot is consumed does not exist here; its replacement hazard — an op body ordering its own waits — is local to the code that owns the data [INFER, src/brisc.cpp:317-329].
- **Exact instruction budgets.** Because windows are count-pinned at build time and composition is assert-pinned to the hardware's 32-entry replay budget, the engine knows — compile-time — how many instructions every replayed loop body costs [FACT, src/brisc.cpp:62-73; tools/sfpu.py:732-735]. No instruction account depends on runtime scheduling [INFER, src/brisc.cpp:62-73].
- **Byte-stable windows.** A replay window re-issues the *recorded* instruction stream, so a hot loop body is bit-identical across every iteration and every build with the same pin [FACT, src/brisc.cpp:1104-1113]. The determinism argument that builds on this is [Ch7 `03-determinism-chain.md`](../ch7_build_determinism/03-determinism-chain.md)'s.
- **A single, auditable instruction account per core.** All Tensix instructions come from one interpreter thread ([`02-instruction-port.md`](./02-instruction-port.md)); there is no second schedule to reconstruct when debugging [INFER, src/brisc_entry.S:8-10; src/brisc.cpp:23614].
- **A fully static L1 map.** No device allocator and no CB allocation at launch: every buffer is an `_ADDR` constant whose non-overlap is static_asserted, so a build cannot produce a layout the run discovers too late [FACT, src/brisc.cpp:94-119; src/constants.h:2428].

## The engine-vs-llk table (a declared Ch8 input)

One honesty note binds the table: the llk column describes the conventional-stack baseline this guide's audience carries ([Ch1 `01-mental-model.md`](../ch1_engine_at_a_glance/01-mental-model.md)) — it is an [INFER, audience background], not regen evidence; the notes corpus records **no measured engine-vs-llk comparison** — a search finds only passing mentions of llk and reconfiguration [FACT, notes/qwen-hardware-history.md:305; notes/glm-bfp8.md:125] — so no row carries [MEAS]. The one measured datum in the neighborhood is a *diagnostic*-cost fact: the NoC-ID check's runtime flag "cost GLM decode about 50 us per token" and is therefore compiled out of silicon images [MEAS, src/brisc.cpp:344] — a reminder that even in-program probes are not free, which is why the engine's checks are build-gated rather than always-on. And the gives-up column is not a claim that llk could not achieve the gains — it is a claim that regen achieves them *by owning the layer llk would mediate* [INFER, src/brisc.cpp:141-149].

| Dimension | llk-based tt-metal stack [INFER — audience baseline] | regen engine (this chapter) | Established in |
| --- | --- | --- | --- |
| Program model | per-op kernel programs launched onto a grid; ckernel runtime + llk between engine and hardware [INFER] | one persistent interpreter per core; op bodies are in-process code; no launch, no runtime [FACT, src/brisc_entry.S:8-10] | [`02`](./02-instruction-port.md), [Ch2 `02`](../ch2_record_driven_execution/02-interpreter-loop.md) |
| Compute programming | llk init/config calls, math kernels from the library; TRISCs run kernel programs [INFER] | cfg writes + instruction pushes from BRISC; SFPU via compiled fragments and replay windows; TRISC threads hold no programs [FACT, src/fw.h:63-72; src/brisc.cpp:23614] | [`01`](./01-mmio-register-layer.md), [`03`](./03-sfpu-replay-windows.md) |
| Data movement & flow control | circular buffers with slot accounting; CB sync between producers/consumers [INFER] | plain L1 stores + unpack instructions; readers own all synchronization; no CBs [FACT, src/brisc.cpp:218-227, 920-922] | [`02`](./02-instruction-port.md) |
| Configuration discipline | library reconfigures per kernel/phase; defensive re-init is cheap and common [INFER] | live-config cache; config written only on change; base AddrMod table fixed once per image start [FACT, src/brisc.cpp:141-149, 1201-1220] | [`01`](./01-mmio-register-layer.md) |
| Instruction accounting | implicit (compiler + runtime); budgets discovered by profiling [INFER] | explicit and pinned: fragment counts emitted, compositions asserted ≤ 32, sizes `static constexpr` [FACT, src/brisc.cpp:62-73; tools/sfpu.py:732-735] | [`03`](./03-sfpu-replay-windows.md), [`05`](./05-correctness-discipline.md) |
| Correctness net | library tests, runtime asserts, watcher machinery [INFER] | build-time pins + boot probes + named-sentinel stops + smoke gate + CPU oracle [FACT, src/brisc.cpp:62-73, 23577-23611; README.md §4] | [`05`](./05-correctness-discipline.md) |
| Determinism | program-cache signatures, trace capture/replay [INFER] | pinned toolchain ⇒ pinned counts ⇒ byte-stable windows; the record list is its own trace [FACT, src/brisc.cpp:62-73, 1104-1113] | [`03`](./03-sfpu-replay-windows.md), [Ch7](../ch7_build_determinism/index.md) |
| Portability | llk targets WH/BH; device abstraction holds across stacks [INFER] | raw Blackhole MMIO; per-image constants; a new architecture is a new layer [FACT, src/fw.h:16-21; src/brisc.cpp:694] | [`01`](./01-mmio-register-layer.md) |
| Math provenance | maintained, shared, tested library kernels [INFER] | per-fragment implementations owned and validated by regen's oracle [FACT, src/kernels.sfpu; src/glm.cpp] | [`03`](./03-sfpu-replay-windows.md), [`04`](./04-op-bodies.md) |

Each row's "load-bearing vs incidental" classification — which side of each row a tt-metal port must re-derive and which is an artifact of regen's host — is [Ch8 `02`](../ch8_tt_metal_port/02-load-bearing-vs-incidental.md)'s, applied with its test ("does the roofline argument survive if tt-metal replaces this?"); this table is that analysis's input, declared here [INFER — the classification is a port-side judgment, out of this chapter's scope].

## The honest cost ledger

Three costs are worth stating plainly, because the gains above are real but not free:

1. **The net must grow with the catalog.** The disciplines of [`05-correctness-discipline.md`](./05-correctness-discipline.md) must grow whenever the op catalog grows — each new command family means new layout pins, new smoke checks [INFER, src/constants.h:2015-2018; src/tt_backend.cpp:37638-37657].
2. **Sequencing expertise lives in the op bodies.** Correct engine ordering (wait where the dependency is, record where the buffer is borrowed) is a per-op human discipline, grep-able but not machine-checked end to end [INFER, src/brisc.cpp].
3. **The trade was made for a specific workload.** Record-driven decode on one model, on one host family, with a pinned toolchain is where the engine's assumptions (closed op set, known config lattice, resident state) hold; the table describes what regen built, not what any workload should build [INFER — a scope judgment derived from the design's shape].

---

← [`05-correctness-discipline.md`](./05-correctness-discipline.md) | ↑ [chapter index](./index.md) | ↑ [guide root](../index.md)

**Next:** [Chapter 4 — DRAM Streaming Discipline](../ch4_dram_streaming_discipline/index.md)
