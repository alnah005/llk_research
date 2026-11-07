# The Interpreter Loop

The **[interpreter](./index.md)** is the chain `brisc_main` → `run_command` → `glm_steps` → `glm_step_list` → `glm_step` → `run_command`: one resident BRISC program per core that fetches a record, decodes its command id, rewrites its per-token arg words, dispatches to the op body, and advances — for every step of every token. This file walks that chain, states what persists across tokens, and covers the in-band trace/progress machinery that makes the loop observable without a capture pass.

**Scope:** covers Question 1, part 2 — the persistent per-lead loop. The record fields it decodes are [`01-step-records.md`](./01-step-records.md)'s; the teams its op bodies drive are [`03-lead-helper-teams.md`](./03-lead-helper-teams.md)'s; the read-ahead that feeds it is [`04-read-ahead-rings.md`](./04-read-ahead-rings.md)'s; what each op body *computes* is [Ch3 `04-op-bodies.md`](../ch3_raw_mmio_sfpu_replay/04-op-bodies.md)'s.

## Boot and role fork: `brisc_main`

`brisc_main` is the image's single entry point [FACT, src/brisc.cpp:23614]. After hardware init and the startup `SrcA` bank probe — a startup MMIO-validation step whose mechanism and failure codes are [Ch3 `05-correctness-discipline.md`](../ch3_raw_mmio_sfpu_replay/05-correctness-discipline.md)'s [FACT, src/brisc.cpp:23592-23610] — each core branches once on its team role and never returns [FACT, src/brisc.cpp:23616-23631]: Ethernet feeders and gather tiles run their dedicated loops [FACT, src/constants.h:1624, 3061], every helper enters `team_helper()` ([[`03-lead-helper-teams.md`](./03-lead-helper-teams.md)]'s subject) [FACT, src/brisc.cpp:3250], and the leads fall into the only remaining loop.

## The lead command loop

The lead loop is a spin on one word [FACT, src/brisc.cpp:23633-23663]:

1. Spin while `TT_L1_GO_ADDR` is clear, flushing the flight recorder in the gap [FACT, src/brisc.cpp:23634-23636].
2. Read the command id from `TT_L1_COMMAND_ADDR`, stamp the command-begin clock, start a progress record [FACT, src/brisc.cpp:23638-23641].
3. `run_command(command)` — for `GLM_STEPS`, the whole list walk below. A command the image lacks clears GO, writes `0xBAD00000 | command` to the sentinel, and continues serving [FACT, src/brisc.cpp:23643-23647].
4. Close: clear GO, meet the chip's leads in `allgather_barrier`, run the PCIe exchange, mark idle — then **shard 0 writes `TT_SENTINEL_TOKEN` (0x544F4B4E) to its sentinel word**, the completion marker [FACT, src/brisc.cpp:23647-23663; src/constants.h:1372].

The host closes the loop by polling that marker — `await_brisc_command` reads shard 0's `TT_L1_SENTINEL_ADDR` until it holds `TT_SENTINEL_TOKEN` or a `0xBAD...`-class word [FACT, src/tt_backend.cpp:6420-6440]. The sentinel's full stop-path discipline — the helper-to-lead post, load-time zeroing, and the absence of any startup write — is [`03-lead-helper-teams.md`](./03-lead-helper-teams.md)'s.

Note what the closing barrier in step 4 is *not*: it runs once per **command**, not per step or per token — the per-step fences are the records' own `sync` classes ([`01-step-records.md`](./01-step-records.md)), and the whole token loop of `glm_steps` sits inside one command [INFER, src/brisc.cpp:23633-23663]. The GO/COMMAND pair is the host's entire per-post interface [INFER, src/brisc.cpp:23633-23663].

## `GLM_STEPS`: one command, the device loops the tokens

`glm_steps` takes the `TtGlmSteps` descriptor — list address, count, start position, token count, row, profile flag [FACT, src/brisc.cpp:21182-21188] — and loops the list once per token, passing each token's `{position, row, rows, slot_offset}`:

```cpp
for (uint32_t t = 0; t < tokens; ++t) {
    glm_step_list(steps, count, {position + t, row + t, tokens, 0u}, profile != 0u);
}
```
[FACT, src/brisc.cpp:21189-21190]

So a single host command runs 1..N tokens end to end; the loop across tokens lives on the device, and the host is not consulted between them [INFER, src/brisc.cpp:21188-21190]. After the last token the lead restores the descriptor at `TT_LAYER_DESC_L1_ADDR` with the list's total cycles — the host's read-back of the command's cost [FACT, src/brisc.cpp:21193-21199]. A token count of 0 stops the lead with `0xBAD9FFFF` [FACT, src/brisc.cpp:21186-21188].

## The per-step chain

`glm_step_list` resolves the lead's own DRAM channel (`grid.dram[grid.worker]`) [FACT, src/brisc.cpp:21154-21155], keeps the ring's `ready`/`end` watermarks ([`04-read-ahead-rings.md`](./04-read-ahead-rings.md)), and calls `glm_step` per record [FACT, src/brisc.cpp:21173]. `glm_step` does, in order [FACT, src/brisc.cpp:21108-21148]:

1. **Name itself.** Read `command` and `sync`; mirror the running index and command to `TT_GLM_STEP_L1_ADDR` — "the index and command at `TT_GLM_STEP_L1_ADDR` name it for the host" [FACT, src/brisc.cpp:21105-21112].
2. **Post progress.** `progress(TT_PROGRESS_RUN, command)` — the observability hook (below) [FACT, src/brisc.cpp:21113-21115].
3. **Copy its arg words.** Exactly `index` words — "only the op's own arg words" — from the ring slot into the descriptor [FACT, src/brisc.cpp:21116-21117].
4. **Apply the per-token rewrites.** Position add/shift, `row`, `rows`, and the slot-word offsets — exactly the rules of [`01-step-records.md`](./01-step-records.md)'s contract, and nothing else [FACT, src/brisc.cpp:21119-21129].
5. **Validate sync.** `TT_GLM_SYNC_MET` or `TT_GLM_SYNC_CHIP`; anything else stops the lead with `0xBAD90000 | index` [FACT, src/brisc.cpp:21130-21131].
6. **Optionally broadcast the step input** (`input_bytes`) to every helper — the multicast mechanism, and the flag-cast ordering behind it, are [`03-lead-helper-teams.md`](./03-lead-helper-teams.md)'s [FACT, src/brisc.cpp:21133-21136].
7. **Dispatch.** `if (command == TT_COMMAND_GLM_STEPS || !run_command(command))` guards against a command this image lacks, stopping with `0xBAD00000 | command` [FACT, src/brisc.cpp:21137-21139]. The `run_command` table is a compile-time-specialized if-chain — "Runs a lead command of the GLM image ... false for commands it lacks" [FACT, src/brisc.cpp:23398, 21040-21042] — and recursion is fenced: a record whose command is itself `GLM_STEPS` stops the core; records never nest [INFER, src/brisc.cpp:21137-21139].
8. **Optionally post the async prefetch job** from the record's `prefetch` descriptor, which the lead does not wait for [FACT, src/brisc.cpp:21140-21145].
9. **Optionally meet the chip's leads** (`TT_GLM_SYNC_CHIP` → `allgather_barrier`) [FACT, src/brisc.cpp:21146-21148].

Advance is implicit: the loop index moves to the next ring slot; there is no queue, no callback, no completion interrupt, no scheduler, and no return to any loader between steps [INFER, src/brisc.cpp:21153-21179].

## What persists across tokens, and where

Nothing per step or per token is allocated. The interpreter's cross-token state is a set of L1 regions named by `_ADDR` symbols, established at image start and never reallocated [FACT, src/constants.h:2026-2028]:

| State | Home | Carries |
| --- | --- | --- |
| KDA recurrence history | `TT_GLM_KDA_HISTORY_ADDR` | 34 KDA layers' per-layer state (`TT_GLM_KDA_LAYERS` = 34) [FACT, src/constants.h:3332, 3335] |
| MLA latent history | `TT_GLM_MLA_HISTORY_ADDR` | 11 MLA layers × 2048 bytes (`TT_GLM_MLA_LAYERS` = 11) [FACT, src/constants.h:3333-3336] |
| Resident streams | `TT_GLM_STREAMS_ADDR` | The mHC stream slice each lead keeps, written per token by the host's writers ([`05-eliminating-the-runtime.md`](./05-eliminating-the-runtime.md)) [FACT, src/constants.h:3338] |
| mHC/MLA pending mixes | `TT_GLM_HC_PENDING_ADDR`, `TT_GLM_MLA_LATENT_PENDING_ADDR` | A lead's pending mix its first command finishes [FACT, src/constants.h:3340-3343] |
| Resident weight shares | `TT_GLM_RESIDENT_ADDR` | Helper L1 that prefetch jobs stage into, free between attention spans [FACT, src/constants.h:2443-2446] |
| The record ring | `TT_GLM_STEP_WINDOW_ADDR` | The top of the state region ([`04-read-ahead-rings.md`](./04-read-ahead-rings.md)) [FACT, src/constants.h:3344-3347] |

The L1 map's own static_asserts pin the whole layout — data, state, ring, and descriptors all below the control block [FACT, src/constants.h:3362-3377]. This residency is what makes "no phase boundaries" concrete: token $t+1$'s first record finds token $t$'s state exactly where it left it, in the same core's L1 [INFER, src/constants.h:3332-3343]. The per-user state inventory (KDA recurrence state, MLA latents, pooled indexer keys) as model facts is [Ch1 `03-workload-context.md`](../ch1_engine_at_a_glance/03-workload-context.md)'s.

## In-band observability: progress, flight, trace, profile

The loop is its own profiler; four mechanisms, all ordinary memory writes the host can read — no debugger, no capture pass [INFER, src/brisc.cpp]:

| Mechanism | What it records | Where | Anchor |
| --- | --- | --- | --- |
| Progress record | A 32-byte `TtProgress` — what it runs (a lead's command, a helper's job kind), phase, sequence, position — "Where a core is, for the hang dump" | `TT_L1_PROGRESS_ADDR` | [FACT, src/constants.h:59-66] |
| Flight recorder | Every progress change posted to a host-memory slot "once the last post has left, so the host keeps a chip's last phases after the chip drops off the link" | host buffer via `TT_L1_FLIGHT_ADDR` | [FACT, src/constants.h:68-71] |
| Event trace | 8-byte (tag, wall-clock) events appended to a ring of `TT_TRACE_EVENTS` = 4094 when `TT_L1_TRACE_ENABLE_ADDR` is set | `TT_TRACE_ADDR` = 0x158000 | [FACT, src/brisc.cpp:232-241; src/constants.h:117, 257-258] |
| Step profile | With `profile` set in `TtGlmSteps`, each of the first `TT_GLM_PROFILE_STEPS` = 1024 steps' cycles (op and round separately), stamped over the trace ring's storage, which GLM leads leave free | `TT_GLM_PROFILE_ADDR` = `TT_TRACE_ADDR` | [FACT, src/constants.h:2400-2411; src/brisc.cpp:21171-21177] |

The phase vocabulary is a fixed enum — idle, run, team, barrier, exchange, arrive — so a hang dump reads out as a readable state-machine trace [FACT, src/constants.h:74-80]. The host clears the trace enable on GLM images [FACT, src/brisc.cpp:242-244], and the profile's read-back prints "position, step, us, name" lines — the names come from the builder's step list, so the profile is directly an op-by-op timeline of the record list [FACT, src/tt_backend.cpp:24460-24468]. The roofline analysis was built from these mechanisms alone: "GLM_STEPS stamps per step on every chip (record fetched, op done, chip barrier done, mesh done), each op's own cycle results copied to a shadow record, and firmware event traces on every core for one token" [MEAS, notes/glm-decode-roofline.md (lines 82-88)].

## The record list IS the trace

A conventional stack captures a trace from a live run, then replays it; the capture and the program are different artifacts. Here the distinction does not exist [INFER, src/brisc.cpp]:

- the *program* the lead runs per step is the record's arg words — data;
- the *schedule* is the list's order plus its `sync` fields — data;
- the *audit trail* is the machinery above, which reads the same `TT_GLM_STEP_L1_ADDR` mirror the host reads on failure — "A failure names each lead's running step" from that mirror [FACT, src/tt_backend.cpp:24429-24441].

"Profiling" is the same post with `profile` set in `TtGlmSteps` — the host's `glm_run_steps_profiled` posts the identical list and reads back per-step cycle words [FACT, src/tt_backend.cpp:24463-24468]. An observed run and a production run differ in data fields, never in mechanism; there is no capture-versus-replay distinction anywhere in this engine [INFER, src/brisc.cpp:21153-21178].

## Where the per-step cost points

The per-step cost terms this loop produces — 0.34 us record fetch and copy, ~0.45 ms bookkeeping per 561-step token — feed [`05-eliminating-the-runtime.md`](./05-eliminating-the-runtime.md)'s elimination argument, which prices them and contrasts the interpreter with a dispatch-per-step engine; [Ch1 `01-mental-model.md`](../ch1_engine_at_a_glance/01-mental-model.md) owns the side-by-side contrast table.

---

← [`01-step-records.md`](./01-step-records.md) | ↑ [chapter index](./index.md) | ↑ [guide root](../index.md)

**Next:** [`03-lead-helper-teams.md`](./03-lead-helper-teams.md)
