# Step Records

A **[record](./index.md)** is one `TtGlmStepRecord` — a flat, 256-byte word-set that encodes one unit of device work: a command id, a sync class, a window of operand arg words, and the pacing/prefetch descriptors the interpreter consumes around it. A whole decode step is one flat per-core list of these. This file owns the format end to end: the struct, the arg-word contract, the command-id space, operand addressing, per-token parameterization, list replication, image-flag gating, and the argument for records-as-data over code-as-program.

**Scope:** covers Question 1, part 1 — the record **format** and the record/step struct family. The host builder that *produces* lists is [Ch6 `04-record-builder-and-serving.md`](../ch6_host_driver_weights_oracle/04-record-builder-and-serving.md)'s and is only pointed to; the interpreter that consumes these fields is [`02-interpreter-loop.md`](./02-interpreter-loop.md)'s; what each command id *computes* is [Ch3 `04-op-bodies.md`](../ch3_raw_mmio_sfpu_replay/04-op-bodies.md)'s.

## The frame: one command runs a whole list for many tokens

`TT_COMMAND_GLM_STEPS` = 124 is the command id that carries an entire decode step [FACT, src/constants.h:2403]. Its argument struct `TtGlmSteps` — `steps, count, position, tokens, row, profile`, plus a `cycles` result word the lead writes back [FACT, src/constants.h:2404-2407] — says: run `count` records from `steps`, for each of `tokens` (at least 1) positions from `position` [FACT, src/constants.h:2385-2387]. One post can therefore cover a whole multi-position row; the device, not the host, iterates the token count ([`02-interpreter-loop.md`](./02-interpreter-loop.md)).

Every lead runs the same `count` records from **its own DRAM channel — the lead's index** — at `steps` [FACT, src/constants.h:2385]; the host makes that true by placement, below.

## The record itself

`struct TtGlmStepRecord` is defined at [src/constants.h:2414-2424](../../../regen-glm5.3-flash/src/constants.h), with its size pinned at build time — `static_assert(sizeof(TtGlmStepRecord) == 256u && sizeof(TtGlmSteps) <= TT_LAYER_DESC_BYTES)` [FACT, src/constants.h:2428] — and size-parity with the prefill image's record, so one slot geometry serves both images [FACT, src/constants.h:2447]. Field by field:

| Field | Meaning | Anchor |
| --- | --- | --- |
| `args[TT_LAYER_DESC_WORDS]` | 48 arg words (192 bytes) of operand data — embedded L1 addresses and small immediates; `TT_LAYER_DESC_WORDS` = 48 [FACT, src/constants.h:1503-1504] | [src/constants.h:2415](../../../regen-glm5.3-flash/src/constants.h) |
| `command` | The op's command id — the dispatch key | [src/constants.h:2416](../../../regen-glm5.3-flash/src/constants.h) |
| `sync` | `TT_GLM_SYNC_MET` (0) or `TT_GLM_SYNC_CHIP` (1); anything else stops the lead (below) | [src/constants.h:2413, 2416](../../../regen-glm5.3-flash/src/constants.h) |
| `position` | Arg-word *index* that receives the running position (below) | [src/constants.h:2416](../../../regen-glm5.3-flash/src/constants.h) |
| `index` | Dual role: in DRAM, the count of arg words the lead copies (a multiple of 4); at `TT_GLM_STEP_L1_ADDR`, the running step index that names a failure to the host [FACT, src/constants.h:2417] | [src/constants.h:2417](../../../regen-glm5.3-flash/src/constants.h) |
| `position_add`, `position_shift` | The per-token position rewrite (below) | [src/constants.h:2418](../../../regen-glm5.3-flash/src/constants.h) |
| `window` | Read-ahead restart flag — the ring pacing contract ([`04-read-ahead-rings.md`](./04-read-ahead-rings.md)) | [src/constants.h:2419, 2388-2389](../../../regen-glm5.3-flash/src/constants.h) |
| `input`, `input_bytes` | Optional step input: `[input, input + input_bytes)` of the lead's L1 is multicast to every helper before the op ([`03-lead-helper-teams.md`](./03-lead-helper-teams.md)) | [src/constants.h:2419, 2389-2390](../../../regen-glm5.3-flash/src/constants.h) |
| `prefetch` (`TtGlmStrips`), `prefetch_halves` | Weight-prefetch descriptor posted as a resident-prefetch team job after the op (`TT_GLM_PREFETCH_LOWER`/`_UPPER`/`_STAGE` halves) [FACT, src/constants.h:2391-2395] | [src/constants.h:2420-2421](../../../regen-glm5.3-flash/src/constants.h) |
| `slot_words[2]`, `row`, `rows` | The per-token row/slot rewrites (below) | [src/constants.h:2422-2423](../../../regen-glm5.3-flash/src/constants.h) |

## The arg layer: word indices, not pointers

Records do not carry pointers; they carry *word indices* into the descriptor at `TT_LAYER_DESC_L1_ADDR` (0x171000 + 0x300), 48 words of 4 bytes [FACT, src/constants.h:1406, 1503-1504]. The `index` field says how many of those words are this op's own — "only the op's own arg words, which the record's `index` counts" are copied per step, never the whole struct [FACT, src/brisc.cpp:21116-21117]. Each command's C struct (`TtGlmSum`, `TtGlmNorm`, `TtGlmMatvec`, ...) is static_assert-pinned to fit the same 192-byte descriptor window, so one descriptor slot serves every command [FACT, src/constants.h:2362, 2428, 2488-2489].

Embedded operand addresses are offsets into the L1 memory map of `_ADDR` symbols — the resident mHC streams at `TT_GLM_STREAMS_ADDR` [FACT, src/constants.h:3338], the running-step copy at `TT_GLM_STEP_L1_ADDR` [FACT, src/constants.h:2441-2442], the helper matrix-share range at `TT_GLM_RESIDENT_ADDR` [FACT, src/constants.h:2443-2446] — all carved out of the control block rooted at `TT_L1_CONTROL_ADDR` = 0x171000 [FACT, src/constants.h:12]. A record is therefore self-contained: the interpreter never resolves a name or walks a pointer chain to find an operand — the address *is* the word [INFER, src/constants.h:2379-2483].

## Per-token parameterization: records rewrite their own args

One list serves every token of a post; three arg-word rewrite rules re-parameterize it per running token. The contract sentences, in the `TT_COMMAND_GLM_STEPS` comment:

> The arg word `position` takes (the running position + `position_add`) >> `position_shift`. The arg word `row` takes `row` plus the running token's index in the post, and `rows` the post's token count (GLM_BATCH: the running row and the row count rounded up to 16). The arg words `slot_words` address a user's DRAM caches; GLM_STEPS runs them as they are and GLM_BATCH adds its row's slot offset. [FACT, src/constants.h:2396-2399]

The interpreter carries the running context in a `GlmStepAt` (`position, row, rows, slot_offset`) [FACT, src/brisc.cpp:21060-21063] and applies exactly these rewrites per step with `glm_set_arg_word`, which silently ignores word indices of `TT_LAYER_DESC_WORDS` or more — "Word indices of `TT_LAYER_DESC_WORDS` name no word" [FACT, src/constants.h:2399-2400; src/brisc.cpp:21071-21074, 21119-21129]. That escape hatch is the format's "no rewrite" encoding, and the device side never does more: position, row/rows, and the slot-word offsets, nothing else [FACT, src/brisc.cpp:21119-21129]. This is why a single posted list is position- and user-correct without any host edit per token [INFER, src/brisc.cpp:21060-21129].

## The command-id space

GLM command ids are allocated per image, in bounded blocks. The decode GLM image's commands take IDs 64-127 — `TT_COMMAND_GLM_FIRST` 64 to `TT_COMMAND_GLM_LAST` 127 — fully sub-allocated: KDA 64-79, MLA 80-99, mHC and MoE 100-119, integration 120-127 [FACT, src/constants.h:2015-2017]. The glm-batch image's commands take a second block, 144-159 [FACT, src/constants.h:3382]. Three extension commands sit past that block: the GLM ops image's gather-tile pair `TT_COMMAND_GLM_GATHER_START` 160 and `TT_COMMAND_GLM_GATHER_WAIT` 161 [FACT, src/constants.h:3053-3060] (the gather-tile image is `glm_ops_image || bench_image` [FACT, src/brisc.cpp:59]), and `TT_COMMAND_GLM_BATCH_ETH_GATHER` 162, a glm-batch-image command since the batch-Ethernet image is built as a GLM batch image [FACT, src/constants.h:3520-3525; src/brisc.cpp:50]. `TT_COMMAND_GLM_GATHER` 147 sits in the batch block while also serving the GLM ops image — a GLM-family command id outside 64-127 [FACT, src/constants.h:3016]. (`TT_COMMAND_GLM_PREFILL_TILE` 128 / `_UNTILE` 129 also sit outside both blocks; prefill is out of guide scope [FACT, src/constants.h:2521-2528].) No `TT_COMMAND` id above 162 is defined, so the free space past the batch block starts at 163, not 160 [INFER, src/constants.h:2015-2017, 3059-3060, 3525]. The decode list's core commands, each defined with its full arg-struct contract:

| Id | Command | One record drives | Anchor |
| --- | ---: | --- | --- |
| 119 | `TT_COMMAND_GLM_TOP_LOGITS` | Per-chip top-logits fold for sampling/logprobs | [src/constants.h:2514](../../../regen-glm5.3-flash/src/constants.h) |
| 120 | `TT_COMMAND_GLM_MATVEC` | `repeats` synchronized helper passes over `strips` — the projection workhorse | [src/constants.h:2325](../../../regen-glm5.3-flash/src/constants.h) |
| 121 | `TT_COMMAND_GLM_SUM` | Weighted FP32 vector sums (split-K partials, MoE combine, router logits) | [src/constants.h:2369](../../../regen-glm5.3-flash/src/constants.h) |
| 122 | `TT_COMMAND_GLM_NORM` | The GLM RMSNorm | [src/constants.h:2379](../../../regen-glm5.3-flash/src/constants.h) |
| 123 | `TT_COMMAND_GLM_EXPERT_MATVEC` | Matvec over a routed-expert selection's virtual strips | [src/constants.h:2356](../../../regen-glm5.3-flash/src/constants.h) |
| 124 | `TT_COMMAND_GLM_STEPS` | Run a record list for `tokens` positions — the interpreter's entry command | [src/constants.h:2403](../../../regen-glm5.3-flash/src/constants.h) |
| 125 | `TT_COMMAND_GLM_MOVE` | Row copies between L1, team L1, DRAM, token rows, and host memory | [src/constants.h:2457](../../../regen-glm5.3-flash/src/constants.h) |
| 126 | `TT_COMMAND_GLM_ALLGATHER` | A cross-chip gather step | [src/constants.h:2473](../../../regen-glm5.3-flash/src/constants.h) |
| 127 | `TT_COMMAND_GLM_ARGMAX` | Chip-parallel argmax over the head's logits | [src/constants.h:2483](../../../regen-glm5.3-flash/src/constants.h) |

The same blocks carry the layer-kind families the walk interleaves — all defined in the command-id region of [src/constants.h](../../../regen-glm5.3-flash/src/constants.h):

- the KDA family from `TT_COMMAND_KDA` 64 [FACT, src/constants.h:2061] and the MLA/indexer family from `TT_COMMAND_GLM_MLA_FLASH` 80 / `TT_COMMAND_GLM_INDEX` 81 [FACT, src/constants.h:2555, 2596];
- the mHC/router/SwiGLU/MoE site commands 100-107 [FACT, src/constants.h:2604-2610, 2743];
- the batch and over-token block from `TT_COMMAND_GLM_BATCH` 144 [FACT, src/constants.h:3393], with `TT_COMMAND_GLM_GATHER` 147 [FACT, src/constants.h:3030], `TT_COMMAND_GLM_BATCH_EMBED` 153, and `TT_COMMAND_GLM_HOST_SYNC` 154 — "where the host joins a post" ([`05-eliminating-the-runtime.md`](./05-eliminating-the-runtime.md) §(c)) [FACT, src/constants.h:3469-3478, 3486-3492].

Their semantics and the walk they implement are [Ch1 `02-anatomy-of-a-token.md`](../ch1_engine_at_a_glance/02-anatomy-of-a-token.md)'s and [Ch3 `04-op-bodies.md`](../ch3_raw_mmio_sfpu_replay/04-op-bodies.md)'s; this file owns only the encoding.

## Sync classes: the only fences a record can request

The `sync` field is one of two values [FACT, src/constants.h:2412-2413]:

- `TT_GLM_SYNC_MET` (0) — none: "the op returns with the chip's leads met, as glm_chip_job leaves them" [FACT, src/constants.h:2412] — the chip-wide job pattern already met them ([`03-lead-helper-teams.md`](./03-lead-helper-teams.md)).
- `TT_GLM_SYNC_CHIP` (1) — "allgather_barrier: the chip's leads meet" after the step [FACT, src/constants.h:2413].

Anything else stops the lead with `0xBAD90000 | index` — sync class is validated, not defaulted [FACT, src/brisc.cpp:21130-21131]. This two-class design is the whole intra-list fence vocabulary: whether a step ends with a chip-wide barrier is a field, not code [INFER, src/constants.h:2412-2413].

## One flat list per core, replicated per channel

"Every lead runs `count` `TtGlmStepRecord` from its own DRAM channel (the lead's index) at `steps`, in order" [FACT, src/constants.h:2385]. The builder makes that true by construction: `glm_place_steps` writes the finished list "to every DRAM channel of every chip," rewriting per-chip arg words by a chip-indexed stride as it goes [FACT, src/tt_backend.cpp:24368-24382] — the per-chip patch mechanism is `GlmChipWord`: arg word `word` of step `step` takes its built value plus chip × `stride` [FACT, src/tt_backend.cpp:23273-23277]. This is the only per-chip variation; within a chip, all leads run identical records [INFER, src/tt_backend.cpp:24368-24382]. Eight leads streaming one logical list therefore produce eight independent single-channel DRAM read streams with no cross-talk [INFER, src/brisc.cpp:21154-21155]. The placement half of that contract is [Ch6 `03-channel-local-placement.md`](../ch6_host_driver_weights_oracle/03-channel-local-placement.md)'s; the fetch half is [`04-read-ahead-rings.md`](./04-read-ahead-rings.md)'s.

## Image flags gate record variants

One source tree builds several firmware images: `TT_FW_GLM`, `TT_FW_GLM_PREFILL`, `TT_FW_GLM_OPS`, `TT_FW_GLM_BATCH`, `TT_FW_GLM_BATCH_ETH` — compile-time constants with static_asserts enforcing at most one GLM variant per image [FACT, src/brisc.cpp:16-56] and exactly one of `TT_FW_DECODE`, `TT_FW_PREFILL`, `TT_FW_GLM` [FACT, src/brisc.cpp:16078-16079]. The flags gate which *commands* exist, not the record layout: the batch image "adds its batch and over-token commands and drops the decode MoE's and the index top-512" [FACT, src/brisc.cpp:21040-21041], and record-stream variants inside the Ethernet batch image are further gated by `TT_GLM_KDA2` and `TT_GLM_MTP` [FACT, src/brisc.cpp:21203-21212]. A record whose command the running image lacks stops its lead with `0xBAD00000 | command` — a build/serving mismatch fails loudly at the offending record, not silently [FACT, src/brisc.cpp:21137-21139]. Batch and MTP economics are [Ch4 `05-batch-and-mtp-economics.md`](../ch4_dram_streaming_discipline/05-batch-and-mtp-economics.md)'s; the `TtGlmPrefill*` family and prefill-phase design are out of guide scope.

## Why records-as-data beats code-as-program

The comparative argument, each benefit read off the structure that provides it [INFER, src/constants.h:2385-2447]:

- **Patchable.** Placed records are plain DRAM words; the host can rewrite an argument inside a placed list without rebuilding anything — e.g. `GlmStepper::set_inverse_temperature` patches the `inverse_temperature` word of the top-logits args directly in DRAM [FACT, src/tt_backend.cpp:31252-31266].
- **Host-auditable.** The running step's index and command are mirrored at `TT_GLM_STEP_L1_ADDR` "for the host" [FACT, src/brisc.cpp:21105-21112], and the host's post wrapper reads exactly that mirror to name the failing step per lead — "A failure names each lead's running step" [FACT, src/tt_backend.cpp:24429-24441].
- **No device recompiles.** New model shapes, routing changes, or per-user caches are new lists; the interpreter image does not change (the builder side is [Ch6 `04`](../ch6_host_driver_weights_oracle/04-record-builder-and-serving.md)'s) [INFER, src/brisc.cpp:23398].
- **Extensible.** A new op is a new command id in its image's id block — a new sub-range, or a fresh stretch of ids as the batch block itself demonstrates; today the genuinely free ids start at 163, past the taken 160-162 extension commands — plus one dispatch branch; old lists stay valid because the format's meaning is carried by the id, not by record shape [INFER, src/constants.h:2015-2017, 3382, 3059-3060, 3525].

The strongest evidence that the record abstraction carries real weight is that regen *tunes synchronization through record data alone* [INFER, src/tt_backend.cpp:23296-23343] — see [`05-eliminating-the-runtime.md`](./05-eliminating-the-runtime.md) Claim (a) for the mechanism (`glm_command_meets` / `glm_skip_barrier`) and its measured $-64$/$-71$ and $-33$ us/token wins.

---

← [`index.md`](./index.md) | ↑ [guide root](../index.md)

**Next:** [`02-interpreter-loop.md`](./02-interpreter-loop.md)
