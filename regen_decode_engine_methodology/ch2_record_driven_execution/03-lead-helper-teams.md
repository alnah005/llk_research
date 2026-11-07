# Lead/Helper Teams

Every Tensix core on a chip has a fixed role for its whole uptime: eight **[leads](./index.md)** — one per DRAM channel — run the interpreter, and a lead's **[helpers](./index.md)** are the cores the lead drives through posted **[team jobs](./index.md)**. Within a step, the op body the lead runs posts jobs; helpers take them, execute a share, and report done — a structure configured once at load that substitutes for per-op grid dispatch. The notes corpus is largely silent on this machinery, so this file leans on strict `[INFER, src/...]` tags for behavior read off the code and `[FACT, ...]` for what the code pins, with `[MEAS]` wherever a note does speak.

**Scope:** covers Question 1, part 3 — team roles and jobs, mailbox/relay handshakes, how the record stream reaches (or does not reach) helpers, and the grid-dispatch substitution. How a job's work splits into strip shares is [Ch4 `02-strip-claims-and-reader-placement.md`](../ch4_dram_streaming_discipline/02-strip-claims-and-reader-placement.md)'s; how leads carry cross-chip rounds is [Ch5](../ch5_cross_chip_transports/index.md)'s; the host's load-time team setup appears here only as the fixed input the device sees (backend mechanics are [Ch6 `01-backend-open-and-load.md`](../ch6_host_driver_weights_oracle/01-backend-open-and-load.md)'s).

## Roles

- **Leads.** Eight per chip — the chip-local peers, `TT_MAX_SHARDS` = 8 [FACT, src/constants.h:50] — "workers 0-7 ... helper[s] follow in lead order; worker 0 is the chip root" [FACT, src/tt_backend.cpp:4710-4711], each reading its records "from its own DRAM channel (the lead's index)" [FACT, src/constants.h:2385]. The production decode layout is "8 chips of 8 leads" [FACT, src/tt_backend.cpp:31169]. The eight leads barrier among themselves via `allgather_barrier`, an all-to-all exchange of NoC atomic arrivals over the chip's peer coordinates [FACT, src/brisc.cpp:2156-2165, 2128-2133].
- **Helpers.** "Tensix cores in the lead rows that run jobs released by their lead", at most `TT_MAX_HELPERS` = 16 per team [FACT, src/constants.h:123-124]. The chip-wide helper count must be a multiple of 8 and identical on every chip, because strip-shard arithmetic assumes it [FACT, src/tt_backend.cpp:13417-13425].
- **Special roles.** The Ethernet feeders (`TT_ROLE_ETH_FEEDER` = 2) and the GLM gather tiles (`TT_ROLE_GLM_GATHER` = 3) run their own dedicated loops before any team logic [FACT, src/constants.h:1624, 3061; src/brisc.cpp:23616-23624].

Which column leads is itself a placement decision with a DRAM rationale: decode reads each DRAM channel through the helpers dealt to it — half west of physical column 8, half east — so the lead column is chosen to split the helper columns as evenly as possible, keeping every channel's readers on its own side of the chip [FACT, src/tt_backend.cpp:1465-1472]. The streaming side of that contract is [Ch4 `02`](../ch4_dram_streaming_discipline/02-strip-claims-and-reader-placement.md)'s.

The host assigns roles once at load, before any command: for each lead a six-word team block `{0, —, —, helpers, base, base × helpers}` plus its helper coordinates [FACT, src/tt_backend.cpp:5052-5059]; for each helper `{1, lead coordinate, index within team, team size, base, base}` [FACT, src/tt_backend.cpp:5083-5087]; gather and feeder tiles get their role words [FACT, src/tt_backend.cpp:5130-5140]. Helpers' jobs start past the sequence `base`, "so none waits for a step input before the first is cast" [FACT, src/tt_backend.cpp:5069] — and "job sequences restart with the image, so no round RX, contribution or QSA count slot may keep an exact match" [FACT, src/tt_backend.cpp:5021]. Nothing about team structure changes per token [INFER, src/tt_backend.cpp:5050-5090].

## The team control map

All team coordination lives in one fixed L1 region [FACT, src/constants.h:126-132]:

| Word | Owner | Meaning |
| --- | --- | --- |
| `TT_L1_TEAM_ROLE_ADDR` | both | Nonzero on helpers [FACT, src/constants.h:126] |
| `TT_L1_TEAM_LEAD_ADDR` | helper | Its lead's coordinate [FACT, src/constants.h:127] |
| `TT_L1_TEAM_INDEX_ADDR` | helper | Its index within its team [FACT, src/constants.h:128] |
| `TT_L1_TEAM_COUNT_ADDR` | both | Team size [FACT, src/constants.h:129] |
| `TT_L1_TEAM_GO_ADDR` | helper | The job sequence its lead has posted [FACT, src/constants.h:130] |
| `TT_L1_TEAM_DONE_ADDR` | lead | Helper completions; helper side: first job − 1 [FACT, src/constants.h:131] |
| `TT_L1_TEAM_COORDS_ADDR` | lead | Up to `TT_MAX_HELPERS` helper coordinates [FACT, src/constants.h:132] |

## The job object and the kind space

A job is a small fixed descriptor the lead publishes: `TtTeamJob {kind, next, argument, row, mix}`, where `next` is "Job the lead posts after this one; helpers may prefetch its first strips" — the team-level read-ahead hook ([`04-read-ahead-rings.md`](./04-read-ahead-rings.md)'s family section) [FACT, src/constants.h:978-985]. Job kinds are "numbered in one list so kinds added on separate branches can't collide" [FACT, src/constants.h:171-172] — one flat enum from `TT_TEAM_JOB_BENCH` = 1 up through the GLM decode kinds (`TT_TEAM_JOB_GLM_MATVEC`, `_GLM_EXPERT_MATVEC`, `_GLM_MOE`, `_GLM_RESIDENT_PREFETCH`, the KDA/MLA/mHC families, ...) to the batch-image kinds [FACT, src/constants.h:173-237]. Which kinds an image serves is gated by the same image flags as commands ([`01-step-records.md`](./01-step-records.md)) [INFER, src/brisc.cpp:23499-23502].

## Posting a job: the lead's half

`team_post_with` is the entire lead-side protocol [FACT, src/brisc.cpp:3325-3350]:

1. **Wait for the previous job** to be fully done: `team_wait` spins on the done counter until it reaches `seq × team_size` — every helper of the sequence accounted [FACT, src/brisc.cpp:3321-3322].
2. **Run the args callback** — which may write words the helpers are about to read — then copy the `TtTeamJob` to `TT_L1_TEAM_JOB_ADDR`, "read with the descriptor" [FACT, src/brisc.cpp:3332-3334; src/constants.h:165].
3. On `noc_check` builds, **stamp** the descriptors with the job's sequence plus a checksum of the words its helpers read [FACT, src/brisc.cpp:3335-3337].
4. **Release**: bump its own GO, then write the same sequence word to *each helper's* `TT_L1_TEAM_GO_ADDR`, one NoC write per helper — "Every GO write differs only in its helper's coordinate" [FACT, src/brisc.cpp:3337-3348].

Variants: `team_post` (args-less) [FACT, src/brisc.cpp:3357-3359]; `team_post_copied`, which copies `words` words of args from `src` to a caller-given `dst` once the previous job is done [FACT, src/brisc.cpp:3361-3366]; and `team_post_async<kind>` for the five async kinds — the form the step loop uses for a record's `prefetch` field, targeting `TT_GLM_PREFETCH_ARGS_ADDR` in the second descriptor [FACT, src/brisc.cpp:3375-3378, 3187-3191].

## Running a job: the helper's half

`team_helper` is `[[noreturn]]` [FACT, src/brisc.cpp:3250]. Per job sequence it does exactly this [FACT, src/brisc.cpp:3260-3318]: read its lead, index, and first sequence from the control block and arm its atomic completion command buffer [FACT, src/brisc.cpp:3260-3267]; spin until the lead's GO sequence passes its own [FACT, src/brisc.cpp:3274-3280]; read the job plus both descriptors from the lead's L1 in one bulk NoC read [FACT, src/brisc.cpp:3281; src/constants.h:169]; on `noc_check` builds verify the stamp so "a helper whose copy disagrees stops with `0xBADE0000` | the job's kind" — the lead overwrote the descriptors before the helper read them [FACT, src/constants.h:2433-2437; src/brisc.cpp:3285-3287]; if the job reads a step input, wait for the input's flag first (below); dispatch through the helper job table [FACT, src/brisc.cpp:3303-3304]; then report done.

**Done is counted, not broadcast:** each helper's `allgather_arrive` posts one NoC atomic increment on its lead's `TT_L1_TEAM_DONE_ADDR`, and the lead's `team_wait` waits for the count to pass `seq × team_size` [FACT, src/brisc.cpp:2128-2133, 3321-3322; src/constants.h:130-131]. One refinement carries most of the decode prefetch design: "A decode prefetch job counts as done once this helper holds its args: its lead's next post need not wait for the reads, which the job that consumes them waits for on this helper, in job order" [FACT, src/brisc.cpp:3297-3298]. That early arrive is what lets a record's resident-prefetch overlap the next steps instead of serializing behind DRAM [INFER, src/brisc.cpp:3297-3314].

## Do helpers see the record stream?

No — records stay lead-side. Only the lead runs the record ring ([`04-read-ahead-rings.md`](./04-read-ahead-rings.md)); helpers never fetch or decode `TtGlmStepRecord`s. What crosses the lead→helper boundary is the job descriptor, the arg words the lead prepares, and — for step inputs — the multicast data [INFER, src/brisc.cpp:3250-3318]. Two consequences worth naming [INFER, src/brisc.cpp:3325-3349]:

- helpers never decode records, so record-stream bandwidth is spent once per team, not once per core;
- the release is unicast-per-helper but the *wait* is a single counter, so the lead's cost per job is O(team size) NoC writes plus one spin, not a handshake per helper.

## Step inputs: one multicast, not N unicasts

When a record carries `input_bytes`, the lead distributes the step input to every helper in one motion: the chip root — worker 0, "the only source of multicasts to the helpers" — casts the L1 range to the helper rectangles, then casts the reading job's sequence flag "on the same NoC and VC, so the flag lands behind the input"; helpers named by that sequence wait for the flag before running the job [FACT, src/brisc.cpp:21077-21103, 3290-3294]. The rectangles "also span the Ethernet feeders and idle tiles, whose L1 is free below `TT_ETH_FEEDER_ADDR`" [FACT, src/brisc.cpp:21081-21083]. Ordering by construction, not by a barrier [INFER, src/brisc.cpp:21077-21090]. `glm_team_write` is the same shape for a push: a range of the lead's L1 to every helper, looping over `TT_L1_TEAM_COORDS_ADDR` [FACT, src/brisc.cpp:17400-17406].

## Share math: how a step's work splits

Work-splitting is arithmetic on the job's descriptor, computed from the helper's `TT_L1_TEAM_INDEX_ADDR` inside the job body — the same code runs on every helper with a different index, which is why shares need no per-core programs [INFER, src/brisc.cpp:3261]. The canonical example, the routed-expert selection: a selection of at most 8 experts has $V = \texttt{count} \times \texttt{strips}$ virtual strips, "one share for the chip's W helpers that reads only local channels: team t of T = W / parts, in group g = t % G of G = 8 / parts, takes the virtual strips v = g + G k for a run of k, the T / G teams i = t / G of a group splitting its V / G strips by bytes" [FACT, src/constants.h:2341-2344]. Shares are thus fixed by the descriptor at post time; the per-helper strip-claim pacing behind them is [Ch4 `02`](../ch4_dram_streaming_discipline/02-strip-claims-and-reader-placement.md)'s, not the team's.

## Chip-wide jobs: the sync that comes free

Most decode ops run on *every* team of the chip at once: `glm_chip_job` posts the kind to each lead's team, the lead does idle work while helpers run, then each lead waits its team and the chip's leads meet [FACT, src/brisc.cpp:17379-17397]. That is precisely why a record can declare `TT_GLM_SYNC_MET` — the chip-wide job pattern has already met the chip's leads ([`01-step-records.md`](./01-step-records.md)'s sync-class section) — and why most steps need no barrier of their own.

## The sentinel: stop-path posts and the completion token — never a startup signal

The sentinel word at `TT_L1_SENTINEL_ADDR` is a core's stop status, written only on stop-or-complete paths:

- `stop_core` — "Stops this core with `sentinel`, also posted to a helper's lead" — writes the word, then, if the stopping core is a helper (nonzero team role), posts the same word to its lead's sentinel address so the failure surfaces where the host watches [FACT, src/brisc.cpp:667-675].
- A helper whose job kind the image lacks writes `0xBAD10000 | kind` and posts it to its lead — "The lead reports a job kind this image lacks; the helper stops" [FACT, src/brisc.cpp:3304-3310].
- A shard-0 lead writes `TT_SENTINEL_TOKEN` (0x544F4B4E) at command completion, the clean-finish marker the host polls [FACT, src/brisc.cpp:23661-23663; src/constants.h:1372].
- At load it is the host, not the device, that touches the sentinel words — it zeroes them, so a stale value can never masquerade as a report [FACT, src/tt_backend.cpp:5043-5046].

There is no sentinel write at startup, and helpers begin their loop by reading their lead, index, and first sequence — no signal involved [INFER, src/brisc.cpp:3260-3269]. Every other `stop_core` call site is an error path with a distinct `0xBAxD____` code naming what failed [INFER, src/brisc.cpp]. The stop codes are the team's contract with failure analysis: a wrong share, a lost job, or a misordered prefetch each halts a named core instead of corrupting a neighbor's result [INFER, src/brisc.cpp].

## What it costs, measured

The corpus is mostly silent on teams, but the roofline's per-op stamps price the handshake where it appears:

- An empty team post inside an mHC site "adds 0.6 us (helpers run an empty job)" [MEAS, notes/glm-decode-roofline.md lines 119-120].
- Waiting for the slowest team shows as measured skew: "the barrier waits 2 us for the slowest of the 8 teams" in a KDA layer [MEAS, notes/glm-decode-roofline.md (line 134)], 2.6 us and 3 us (combine phase) in a MoE layer [MEAS, notes/glm-decode-roofline.md (lines 148, 151)], and 3.2 us waiting at the barrier in an MLA layer [MEAS, notes/glm-decode-roofline.md (line 163)].

So team dispatch is real but small — sub-microsecond per post, low-single-digit microseconds of skew per layer — and it lives inside the roofline's "lead serial work, few-core stages, team dispatch and team skew" term, which [`05-eliminating-the-runtime.md`](./05-eliminating-the-runtime.md) returns to [MEAS, notes/glm-decode-roofline.md lines 177-179]. The ranked plan proposed claim-based or balanced shares to remove ~5 us of MoE team skew and ~2 us in KDA [EST, notes/glm-decode-roofline.md lines 204-205]. Team dispatch is therefore not free; it is cheap enough that its cost shows up inside the serial term, not as a separate category [INFER, notes/glm-decode-roofline.md].

## Why a fixed team replaces per-op grids

In a conventional stack, each op launch comes with a grid: the runtime picks cores, ships the program, and synchronizes the whole rectangle per op. Here the grid is fixed at load, the "program" is a 16-byte-class job descriptor, and the per-op synchronization is one GO write per helper plus one atomic arrival per helper [INFER, src/brisc.cpp:3250-3350]:

| Runtime concept (assumed background) | regen team equivalent |
| --- | --- |
| Launch a grid for an op | One `TtTeamJob` post: a word write per helper [FACT, src/brisc.cpp:3337-3348] |
| Per-core argument marshaling | Helpers copy the descriptor from the lead's L1 [FACT, src/brisc.cpp:3281] |
| Grid completion event | The DONE counter reaching `seq × team_size` [FACT, src/brisc.cpp:3321-3322] |
| Cross-op synchronization | The chip job's closing `allgather_barrier` (leads met) [FACT, src/brisc.cpp:17379-17384] |
| Runtime-asserted launch correctness | The job stamp check; failure halts with a named kind [FACT, src/constants.h:2433-2437] |

The trade is flexibility: work distribution can only change by rebuilding lists or images, not by a scheduler [INFER, src/brisc.cpp]; the port-side consequences of that trade are [Ch8 `02-load-bearing-vs-incidental.md`](../ch8_tt_metal_port/02-load-bearing-vs-incidental.md)'s.

## Teams and the cross-chip edge

The leads are the round participants: a step command like `GLM_ALLGATHER` runs a round through the lead column, and the chip root (worker 0) relays rounds between its chip and the peers [FACT, src/constants.h:154-161]. The transports those rounds ride — device-originated PCIe peer-BAR DMA and the ERISC service — and every measured latency are [Ch5](../ch5_cross_chip_transports/index.md)'s; this file stops at "the lead initiates and the root relays" [INFER, src/constants.h:154-161].

---

← [`02-interpreter-loop.md`](./02-interpreter-loop.md) | ↑ [chapter index](./index.md) | ↑ [guide root](../index.md)

**Next:** [`04-read-ahead-rings.md`](./04-read-ahead-rings.md)
