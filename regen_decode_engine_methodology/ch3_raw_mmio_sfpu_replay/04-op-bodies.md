# Op Bodies

An **[op body](./index.md)** is the BRISC code implementing one record command: a composition of cfg writes ([`01-mmio-register-layer.md`](./01-mmio-register-layer.md)), instruction pushes and SFPU replay windows ([`02-instruction-port.md`](./02-instruction-port.md), [`03-sfpu-replay-windows.md`](./03-sfpu-replay-windows.md)), and NoC moves. This file maps the command-id space to the op families, states the register/accumulator conventions they share, and walks three of them at record level — a norm, the router, and one KDA recurrence step. The op semantics (what KDA/MLA/mHC mean for the model) are [Ch1 `03-workload-context.md`](../ch1_engine_at_a_glance/03-workload-context.md)'s decoder ring, not restated here.

**Scope:** covers Question 2, part 4 — op MECHANISM only. Every per-op timing and the 9.3 → 6.05 ms/token ledger are [Ch4 `04-optimization-ledger.md`](../ch4_dram_streaming_discipline/04-optimization-ledger.md)'s; this file cites zero performance numbers by design.

## The auditable menu

Two artifacts make the composition auditable without reading 23,665 lines:

- [src/ckernel_ops.h](../../../regen-glm5.3-flash/src/ckernel_ops.h) (1,072 lines, auto-generated) — the complete Tensix opcode encoding table: every `TT_OP_*` the engine can push, hence the exact instruction vocabulary any op body may use [FACT, src/ckernel_ops.h:5-12].
- `run_command()` — the single dispatch chain from command id to op body, one static branch per command [FACT, src/brisc.cpp:23398-23496]. There is no function-pointer table and no code generation: the dispatch *is* the code, so the set of executable op bodies is closed and grep-able [INFER, src/brisc.cpp:23398-23496].

## The command-id map (decode block 64–127)

The GLM image's commands take IDs 64–127: "KDA 64-79, MLA 80-99, mHC and MoE 100-119, integration 120-127" [FACT, src/constants.h:2015-2018]. The ids this chapter's examples and cross-references use, matching [Ch2's frozen map](../ch2_record_driven_execution/01-step-records.md):

| Family | Ids (selection) | Meaning |
| --- | --- | --- |
| KDA | `TT_COMMAND_KDA` 64, `KDA_OUTPUT` 65, `KDA_LAYER` 66, `KDA_CHUNK` 67 | KDA decode recurrence per unit/token, its gated output norm, one layer's projections/slices/gated norm for the record list, the chunked (WY) prefill form [FACT, src/constants.h:2061, 2081, 3274, 2093] |
| MLA / indexer | `GLM_MLA_FLASH` 80, `MLA_CACHE` 82, `INDEX_TOP` 83, `MLA_QUERY` 84, `MLA_VALUE` 85 | absorbed attention and its cache/index/value stages [FACT, src/constants.h:2555, 3079, 3108, 3130, 3147] |
| mHC / MoE | `HC_PARTIALS` 100 … `HC_HEAD` 103, `GLM_ROUTER` 104, `GLM_SWIGLU` 105, `HC_SITE` 106, `GLM_MOE` 107 | mHC collapse/expand sites, the router, decode SwiGLU, MoE combine [FACT, src/constants.h:2604-2610, 2743] |
| integration | `TOP_LOGITS` 119, `MATVEC` 120, `SUM` 121, `NORM` 122, `EXPERT_MATVEC` 123, `GLM_STEPS` 124, `MOVE` 125, `ALLGATHER` 126, `ARGMAX` 127 | matvec, reductions, norms, the record interpreter itself, moves, cross-chip collectives, sampling [FACT, src/constants.h:2325-2483, 2514] |

Outside the decode block: prefill-image commands continue at 128+ (`GLM_PREFILL_TILE` 128, `GLM_PREFILL_UNTILE` 129) [FACT, src/constants.h:2527-2528], and the glm-batch image's commands take a second block from 144, including `TT_COMMAND_GLM_KDA_ROWS` 149 — the batch variant that runs one KDA layer's recurrence and gated norms for a post's rows [FACT, src/constants.h:3393, 3555]. `TT_COMMAND_GLM_STEPS` (124) is the record interpreter command — [Ch2 `02-interpreter-loop.md`](../ch2_record_driven_execution/02-interpreter-loop.md) owns it; from this chapter's side it is simply the op body that walks the list and calls the others.

## Shared conventions

Every op body is built from the same idiom set:

- **The 64-value Dst block stepper.** `glm_blocks(blocks, body)` runs `body(0)` and `body(2)` — the even and odd column halves of one 64-value Dst block — then advances the Dst row-write counter by four rows with `TTI_INCRWC`, resetting the counter at the end [FACT, src/brisc.cpp:15906-15913]. This is the unit of SFPU work: fragments are written per half-block, and their `addrmod=1` stores assume the four-row advance [FACT, src/kernels.sfpu:5-14, 50-60].
- **Unpacker as loader.** Values enter Dst as BF16 unpacks, 1024 at a time, with the unpacker's L1 context base pointing at the operand [FACT, src/brisc.cpp:15926-15933, 920-922].
- **Packer as storer.** Results leave Dst via `gdn_store`/`glm_store` pack instructions [FACT, src/brisc.cpp:15948, 17294-17295].
- **Live LReg contracts.** Per-op constants are installed with `sfpu_constant` into LRegs 11..14; fragment entry contracts name the LRegs they expect live [FACT, src/brisc.cpp:896-908; tools/sfpu.py:368-395].
- **Synchronize where the dependency is.** Op bodies wait on an engine with `wait_tensix`/`TTI_STALLWAIT` exactly at the data-dependency points, not defensively [FACT, src/brisc.cpp:317-329].
- **NoC moves by explicit request.** Where an op reads cross-chip data (e.g. `GLM_SUM` with `gather` set issues per-row `noc_read_to_l1_async` from the chip root [FACT, src/brisc.cpp:17355-17362]), the transport's *discipline* is [Ch5](../ch5_cross_chip_transports/index.md)'s.

## Worked example 1 — a norm (`TT_COMMAND_GLM_NORM`, 122)

`glm_latent_norm(input, weight, output, columns)` implements the GLM RMSNorm (eps $10^{-5}$) of 512 or 1536 BF16 values "as HF rounds it: the normalized value to BF16, then its product with the weight" [FACT, src/brisc.cpp:15933-15948; src/constants.h:2377-2379]:

1. Unpack input to Dst rows 0.. and weight to rows 256.., 1024 values per pass [FACT, src/brisc.cpp:15939-15940].
2. Sum of squares: `glm_squares` zeroes the SFPU accumulators, replays the **squares replay window** once per 64-value block, and reduces the partials on the SFPU to one scalar read back by BRISC [FACT, src/brisc.cpp:15917-15924, 2031-2034].
3. BRISC computes $\text{scale} = \text{rsqrt}(\text{squares}/n + 10^{-5})$ with a hardware reciprocal-square-root approximation and loads it into LReg6 [FACT, src/brisc.cpp:15943-15946].
4. Per half-block, replay-free inline fragment `sfpu_glm_norm`: value × scale, rounded to BF16, times the weight 256 rows on [FACT, src/kernels.sfpu:1288-1293].
5. Pack out [FACT, src/brisc.cpp:15948].

The pattern to notice: the *reduction* is a replayed window (row-iterating, accumulator-style), the *map* is an inline per-block fragment, and the scalar glue (rsqrt, scale) runs on the BRISC itself [INFER, src/brisc.cpp:15917-15948]. Dispatch: `run_command` maps command 122 straight to this body [FACT, src/brisc.cpp:23468-23470].

## Worked example 2 — the router (`TT_COMMAND_GLM_ROUTER`, 104)

`glm_router` computes the sigmoid scores and the top-8-of-288 expert choice [FACT, src/brisc.cpp:17285-17332; src/constants.h:2615-2616]:

1. Unpack the 5 blocks of logits and the correction bias into Dst [FACT, src/brisc.cpp:17288-17289].
2. Per half-block: `sfpu_qwen_sigmoid(offset, 0x3F800000u)` (scale one) then `sfpu_glm_choice` — score + bias, stored 128 rows on [FACT, src/brisc.cpp:17290-17293; src/kernels.sfpu:430-440, 1295-1298].
3. Pack scores and choices to L1 and `FENCE()` [FACT, src/brisc.cpp:17294-17297].
4. **The top-k scan runs on the BRISC**, not the SFPU: "Experts scan in ascending order, so an expert never displaces an earlier one of equal key: the ranks order by key then lowest expert as router_rank does. Groups of 8 loads issue together, and most choices fall to the bound" — first 8 placed, then a `#pragma GCC unroll 8` sweep over the remaining experts in groups of eight against a running bound [FACT, src/brisc.cpp:17297-17318].
5. Weights: sigmoid scores read back per chosen expert, normalized by a reciprocal of the sum, scaled by 2.5 [FACT, src/brisc.cpp:17320-17331 — `routes.weight[slot]` write at 17330].

Here the engine splits work by *shape*, not by engine preference: lane-parallel activation math goes to the SFPU, the sequential scan with tie-breaking goes to scalar BRISC code, and the two meet at an L1 arena behind a fence [INFER, src/brisc.cpp:17285-17332]. The router's downstream — which strips the routed experts trigger — is [Ch4](../ch4_dram_streaming_discipline/index.md)'s streaming story, not restated here.

## Worked example 3 — one KDA recurrence step (`TT_COMMAND_KDA`, 64)

`kda(args)` loops `units` × `tokens`, calling `kda_unit` per (unit, token) [FACT, src/brisc.cpp:16403-16413]. One unit/token, end to end [FACT, src/brisc.cpp:16369-16389]:

1. At position 0, zero the unit's state and conv history in L1 [FACT, src/brisc.cpp:16376-16378].
2. `kda_prepare`: the unit's convolution over the 3-slot history and the input adapter, then `kda_decays` — dt bias and rate unpacked, then `sfpu_kda_decay` per two rows computes $\exp(-5\,\mathrm{sigmoid}(\text{rate}(f+b)))$ per channel into the decays work area [FACT, src/brisc.cpp:16313-16326, 16357-16363, 16380-16381].
3. The recurrence itself: for a 32-column slice, `gdn_slice`; for a full half, `gdn` — both bottom out in the row loops of [`03-sfpu-replay-windows.md`](./03-sfpu-replay-windows.md). Inside `gdn` the predict and update passes are distinct [FACT, src/brisc.cpp:5768-5800]:
   - **Predict** records `sfpu_gdn_predict_body` at `gdn_replay` — "record without executing" — then replays it per state row with the row's **key and decay only, never the query**: with a uniform decay it is preloaded once via `load_bits32_to_lreg<3u>`; with per-row decays the operands arrive via `kda_rows`' runtime pushes ([`02-instruction-port.md`](./02-instruction-port.md)'s densest pattern), one REPLAY per row [FACT, src/brisc.cpp:5784-5791, 5695-5708].
   - **Update** re-zeros, records `sfpu_gdn_update_body`, and replays it per row with **key and query** — `gdn_rows(args.key, args.query, true, ...)` [FACT, src/brisc.cpp:5792-5795].
4. Pack the new state and the unit's output out [FACT, src/brisc.cpp:5796-5800].

This is the op [`03-sfpu-replay-windows.md`](./03-sfpu-replay-windows.md)'s replay economics was built for: 128 (or 64) iterations of one count-pinned body with per-row scalar operands [INFER, src/brisc.cpp:5768-5800]. The KDA layer command (`KDA_LAYER` 66) wraps the layer's projections, slices and gated norm for the record list — `TtKdaLayer`'s strip/parameter/state fields with per-phase prefetch bits [FACT, src/constants.h:3264-3274]; the batch image's `GLM_KDA_ROWS` 149 re-shapes it per post row [FACT, src/constants.h:3550-3555].

## The remaining families, one line each

- **Matvec** (`120`/`123`): strip-fed BFP matvecs — an unpack–`TTI_MVMUL` (opcode `0x26`)–pack loop whose mechanism is [`01`](./01-mmio-register-layer.md)'s config setters plus [Ch4](../ch4_dram_streaming_discipline/02-strip-claims-and-reader-placement.md)'s strip reader contract; BFP8 and BF16 strips run their two SrcA fidelity phases through the format's counter sequence (eight MVMULs per 64 columns) [FACT, src/brisc.cpp:1226-1231, 1396-1410; notes/glm-bfp8.md (lines 27-28)]; the scale-weight replay window scales inputs per element [FACT, src/brisc.cpp:2060-2065].
- **SwiGLU** (`105`): inline per-block `sfpu_glm_swiglu` — no replay — installing its $\pm 10$ clamps (`0x41200000`/`0xC1200000`) into LRegs 11/12 before the block loop and restoring the firmware constants after [FACT, src/brisc.cpp:17333-17347; src/kernels.sfpu:1300-1315].
- **MoE combine** (`107`): `glm_moe_combine_rows` records and replays the `sfpu_glm_moe_combine_body` window at `gdn_replay`, over nine slots of 512 rows (top-8 routed experts plus the shared expert) — each slot's blocks gated by its route weight, "eight replays step to the next slot" [FACT, src/brisc.cpp:18070-18090; src/kernels.sfpu:550-557].
- **mHC** (`100`-`103`, `106`): collapse/mean/expand fragments, pre/post weighting via live LRegs, the dot product replayed through `sfpu_glm_dot_body` at `gdn_replay` [FACT, src/brisc.cpp:16564-16572, 16923-16931; src/kernels.sfpu:1250-1258, 1271-1286].
- **MLA/indexer** (`80`-`85`): absorbed attention with `exp2_negative`-based flash rescaling, transposes, and the indexer's head sums [FACT, src/kernels.sfpu:969-981, 1138-1199].
- **Rounds**: `ALLGATHER` 126 and the round stages some ops trigger are cross-chip collectives — [Ch5 `03-rounds-and-transport-selection.md`](../ch5_cross_chip_transports/03-rounds-and-transport-selection.md) owns when each transport carries them.

---

← [`03-sfpu-replay-windows.md`](./03-sfpu-replay-windows.md) | ↑ [chapter index](./index.md) | ↑ [guide root](../index.md)

**Next:** [`05-correctness-discipline.md`](./05-correctness-discipline.md)
