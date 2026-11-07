# Correctness Discipline

Programming Tensix at raw MMIO with no llk means the guarantees llk used to provide must come from somewhere else. In regen they come from a layered net: **build-time** (instruction-count static_asserts make codegen drift a build failure, not silent silicon corruption), **boot-time** (startup probes validate the raw-MMIO machine state before the record loop starts — the SrcA bank test and siblings), **run-time** (named sentinel stops), and **post-build** (the smoke gate as the post-build invariant, with the `glm.cpp` CPU oracle closing the loop end-to-end). This file presents the net as evidence, then compresses it into the checklist a port engineer runs when touching any layer of it.

**Scope:** covers Question 2, part 5 — the correctness disciplines. The oracle's *mechanics* (what `glm.cpp` computes, how comparisons are wired) are [Ch6 `05-cpu-oracle.md`](../ch6_host_driver_weights_oracle/05-cpu-oracle.md)'s; the *determinism chain* that protects these pins at build level is [Ch7](../ch7_build_determinism/index.md)'s argument.

## Layer 1: build-time — the pins

Four families of `static_assert` turn assumptions into build failures. Three of them — the fragment **range pins**, the window **count pins**, and the **composition pins** (with their boundary facts) — have their mechanisms and boundary facts established once in [`03-sfpu-replay-windows.md`](./03-sfpu-replay-windows.md), and are enforced here as build failures. The fourth family is this file's own: the **layout pins** — L1 operand buffers never overlap ("SWIGLU gate buffer overlaps up buffer" and siblings; RoPE bases 64-byte aligned) and record structs pinned to 256 bytes [FACT, src/brisc.cpp:94-119; src/constants.h:2428].

The causal claim that makes these load-bearing: because window lengths are `static constexpr` and every `TTI_REPLAY` length argument is a pinned count, *any* change that alters a fragment's instruction count — a compiler upgrade, a source edit, a lowering change — fails the composition assert at build time instead of replaying a wrong-length window into silicon [INFER, src/brisc.cpp:62-73; tools/sfpu.py:732-735]. That is the entire reason codegen drift is a build failure here [INFER, src/brisc.cpp:62-73].

## Layer 2: boot-time — the startup probes

Before any record runs, `brisc_main` establishes and *tests* the raw-MMIO substrate [FACT, src/brisc.cpp:23614-23626]. The probes, each of which records its outcome in L1 for the host:

1. **Data-cache off.** Blackhole RISC-Vs have "a small write-through L1 data cache that NoC writes leave stale", so the firmware disables it (CSR `0x7C0` bit 3) — "as tt-metal's firmware does" [FACT, src/brisc.cpp:1113-1118].
2. **Tensix thread empty.** `require_idle_thread()` polls `RISCV_DEBUG_REG_INSTRN_BUF_STATUS` for this core's thread bits; a prior client's halted run can leave the thread holding instructions that never issue, "which only a reset clears", so the core reports `TT_SRCA_THREAD_BUSY` and parks instead of pushing into a wedged thread [FACT, src/brisc.cpp:1121-1137; src/constants.h:47].
3. **The SrcA bank test.** A core halted between an unpack to SrcA and the `MOVA2D` that reads it keeps that bank valid through a BRISC reset, so "the math then reads one bank behind every unpack" [FACT, src/brisc.cpp:23595-23597 — the probe's own comment]. `srca_step()` runs a marker round-trip — write 64 marker words, unpack them to SrcA, route through Dst, store, and compare [FACT, src/brisc.cpp:23577-23593]. If the marker comes back intact, the bank is in step (`TT_SRCA_IN_STEP` = 1) and the loop proceeds; if stale, one `MOVA2D` + `CLEARDVALID` drains the extra bank and the probe re-runs, recording `TT_SRCA_DRAINED` (2) or `TT_SRCA_OUT_OF_STEP` (3) [FACT, src/brisc.cpp:23598-23611; src/constants.h:44-47]. This probe is the answer to a measured failure mode: the stale bank "hit chip 6 lead 2's mHC expand: wrong at every position, usually finite, +Inf at position 19, which layer 1's site partials spread to every chip" — wrong numbers, not a stop — and "on D this drained an injected lead and an injected helper without a reset, and the runs after them were clean" [MEAS, notes/glm-stability.md (lines 437-450)].
4. **Tensix unit resets pulsed.** Firmware start "pulses the Tensix unit resets (BH tensix.h's unpacker, packer, mover, search, glue, THCON, FPU, SrcA, SrcB and Dst bits) while every RISC is held", because a halt can also leave a *unit* waiting mid-op, which the next process's first op would wait on forever [MEAS, notes/glm-stability.md (lines 457-466)].
5. **Resident windows recorded.** `hw_init` ends by recording the resident replay bodies at slot 0, so every op body sees a known replay-buffer state [FACT, src/brisc.cpp:1104-1113, 1222-1223] — see [`03-sfpu-replay-windows.md`](./03-sfpu-replay-windows.md).

Beyond the boot probes, serving adds a step-0 run: one full decode step at position 0 whose per-lead stream slices must agree with the chip's lead-0 copy and with the host device proxy — "It costs about 37.5 s" [MEAS, notes/glm-stability.md (lines 451-455)].

## Layer 3: run-time — sentinels, not wrong numbers

An invariant violation stops a core with a *named* sentinel, it does not compute garbage. `stop_core(sentinel)` writes the sentinel word to `TT_L1_SENTINEL_ADDR` (and posts it to the lead, for helpers) and parks forever [FACT, src/brisc.cpp:668-676]; the vocabulary is `0xBAD`-class words encoding the cause: misaligned NoC reads (`0xBAD20000 | addr & 0xFFFF`) [FACT, src/brisc.cpp:677-679], a command the running image lacks (`0xBAD00000 | command`, posted at the top-level command loop and via `stop_core` inside `GLM_STEPS` together with `0xBAD90000 | its index` for a bad sync) [FACT, src/brisc.cpp:21105-21139, 23640-23645], NoC-ID budget overruns and role conflicts (`0xBADB0000 | NoC << 4 | ID` and siblings) [FACT, src/brisc.cpp:337-345], job-stamp disagreement (`0xBADE0000 | the job's kind`) [FACT, src/constants.h:2435-2436], and the batch-KDA row classes `0xBAD80E00`/`0xBAD80E01`/`0xBAD80E02` [FACT, src/constants.h:3550-3555]. A halted core looks exactly like a hang from outside; the sentinel word is what turns "stuck" into a named, addressable failure [INFER, src/brisc.cpp:668-676].

The honest caveat is the SrcA lesson of layer 2: a *stateful* hardware residue did once flow through as wrong numbers, which is exactly why the probe family exists [MEAS, notes/glm-stability.md (lines 439-449)]. The standing rule the corpus draws from it: "Tokens from any run after a mid-command halt and before a reset (or a build with these fixes) are confounded, even when they look clean" [MEAS, notes/glm-stability.md (lines 466-468)].

## Layer 4: the smoke gate as the post-build invariant

The smoke driver is [tests/tt_kernels.cpp](../../../regen-glm5.3-flash/tests/tt_kernels.cpp) — 45 lines of argument parsing that select sim or silicon and call `run_tt_kernel_check(..., smoke)` [FACT, tests/tt_kernels.cpp:8-40]. Inside, "`TT_LAB_CHECK` runs only the named checks, comma-separated. The smoke suite runs the checks marked for it, each in its short form; the real-weight runs cover the composed model" [FACT, src/tt_backend.cpp:37638-37640]; after each check the harness runs `require_no_stops` — "Throws if a lead or helper holds a stop sentinel", sweeping every lead and helper — before the next check lays out DRAM afresh, then prints its `check <name>: <n> s` line [FACT, src/tt_backend.cpp:37649-37653, 4546-4559]. A name the mesh does not run is a hard error, not a skip [FACT, src/tt_backend.cpp:38445].

The README's smoke invocation — quoted **verbatim** (all 28 named checks, exactly as written; the list is not glob-compressible, since a compressed form would silently change the check set the driver matches by literal name):

```sh
TT_LAB_CHECK=glm-kda,glm-kda-prefill,glm-prefill-tile,glm-prefill-matmul,glm-prefill-route,glm-prefill-swiglu,glm-prefill-combine,glm-mla-flash,glm-mla-cache,glm-mla-prefill-cache,glm-mla-prefill-flash,glm-mla-index,glm-mla-project,glm-mla-index-top,glm-mhc,glm-mhc-site,glm-mhc-restart,glm-router,glm-swiglu,glm-sum,glm-norm,glm-commands,glm-commands-cast,glm-hc-prefill,glm-batch-gather,glm-moe,glm-steps-index,glm-batch-steps \
    $K --silicon 64 smoke    # 70 PASS lines, about 2.5 min
```

[FACT, README.md §4 (lines 271-274) — the 28-name list at line 272 is quoted character-for-character]. The run observation on the same fence: 28 named checks producing 70 PASS lines in about 2.5 minutes on 64 silicon tiles [MEAS, README.md §4 (lines 272-273)]. The named checks map one-to-one onto the op families of [`04-op-bodies.md`](./04-op-bodies.md) — `glm-kda`, `glm-mla-*`, `glm-mhc*`, `glm-router`, `glm-swiglu`, `glm-moe`, `glm-norm`, `glm-sum`, plus `glm-commands`/`glm-commands-cast` for back-to-back command chaining — so the smoke pass is the op catalog's own invariant [INFER, README.md §4; src/tt_backend.cpp:37638-37657].

## Closing the loop: the oracle

The smoke checks' expected values are not hand-written: they come from the CPU reference model, [src/glm.cpp](../../../regen-glm5.3-flash/src/glm.cpp) (490 lines, with `glm.h` its 154-line interface) — the **[oracle](../ch1_engine_at_a_glance/index.md)** [FACT, src/glm.h; src/glm.cpp]. Its *role* in this chapter's net: device output is only ever judged against a reference computed independently of the device — per-op and end-to-end. The router check is the pattern: biased/zero/saturated logits written to L1, `TT_COMMAND_GLM_ROUTER` posted, routes read back, the same routes computed with the CPU reference, and *identical selection* required with weights within $2\times10^{-5}$, worst error printed on the PASS line [FACT, src/tt_backend.cpp:16514-16548]. So the gate tests divergence, not just absence of halts [INFER, src/tt_backend.cpp:16514-16548]. The oracle's *mechanics* — what the reference computes, fake-quantization parity, exactness/soak structure — are owned once by [Ch6 `05-cpu-oracle.md`](../ch6_host_driver_weights_oracle/05-cpu-oracle.md) and are not restated here.

## The checklist (for a port engineer touching this discipline)

1. **Changed a fragment or its compiler?** The `<name>_instruction_count` constants and every composition assert must still compile — a count change is a *design change*, not a rebuild [FACT, src/brisc.cpp:62-73].
2. **Changed a struct or an L1 `_ADDR`?** The layout pins own the overlap invariants; they will run at build time regardless [FACT, src/brisc.cpp:94-119; src/constants.h:2428].
3. **Changed startup assumptions?** Every probe outcome lands in L1 with a named sentinel or `TT_SRCA_*` code; never replace a probe halt with a best-effort continue [FACT, src/brisc.cpp:23577-23611; src/constants.h:43-47].
4. **Adding a check?** Register it in the smoke suite and keep it runnable under `TT_LAB_CHECK` selection — the 28-name list is an explicit contract, so a new check means a new README list, not a silently wider glob [FACT, src/tt_backend.cpp:37638-37657; README.md §4].
5. **Failure triage:** a parked core with a `0xBAD…` sentinel or a `TT_SRCA_*` probe code is the *expected* manifestation of a violated assumption; wrong-but-quiet numbers mean the net failed and the bug is upstream of it [INFER, src/brisc.cpp:668-676, 23577-23611].
6. **Post-halt rule:** treat any output from a run between a mid-command halt and a reset (or a fixed build) as confounded, even when it looks clean [MEAS, notes/glm-stability.md (lines 466-468)].

## Ownership fences

- Why the toolchain pin is what keeps layer 1 meaningful across rebuilds: [Ch7 `03-determinism-chain.md`](../ch7_build_determinism/03-determinism-chain.md).
- The oracle's mechanics: [Ch6 `05-cpu-oracle.md`](../ch6_host_driver_weights_oracle/05-cpu-oracle.md).
- What tt-metal's watcher/assert machinery can and cannot substitute for this net: [Ch8](../ch8_tt_metal_port/index.md)'s re-derivation material — this chapter states the invariants; it does not argue the port.

---

← [`04-op-bodies.md`](./04-op-bodies.md) | ↑ [chapter index](./index.md) | ↑ [guide root](../index.md)

**Next:** [`06-what-no-llk-means.md`](./06-what-no-llk-means.md)
