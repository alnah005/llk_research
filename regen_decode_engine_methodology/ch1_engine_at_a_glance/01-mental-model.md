# The Mental Model

regen runs decode as data interpretation, not program execution — the one-sentence thesis is the [chapter index](./index.md)'s. This file states that inversion, contrasts it with a conventional tt-metal stack, and lists the five design principles the rest of the guide substantiates.

**Scope:** shared foundation (no question owned). The elimination argument itself is Ch2 `05`'s; this file only builds the intuition.

## The inversion

In the stack the audience knows, decode is a host-driven pipeline: for every token and every op, the host dispatches a program, the runtime launches kernels into grids, compute kernels exchange data through circular buffers (CBs), and the host re-enters between phases. Dispatch overhead is fought with traces and program caches; determinism is fought with the allocator.

regen inverts each of those:

- The unit of device work is a **[record](./index.md)** — one `TtGlmStepRecord`, 256 bytes with its size asserted at build time [FACT, src/constants.h:2414, 2428]; the host builds the per-core record lists once per batch configuration (Ch6 `04`).
- The only resident program per core is the **[interpreter](./index.md)**: `glm_step_list` walks the record list at `src/brisc.cpp:21153`, `glm_steps` loops it over tokens at `src/brisc.cpp:21182-21190` [FACT, src/brisc.cpp:21153-21190]. Nothing is launched per step: `TT_COMMAND_GLM_STEPS` = 124 runs an entire record list [FACT, src/constants.h:2403] — one command can cover a whole multi-position row because `glm_steps` iterates its `tokens` field on the device [FACT, src/brisc.cpp:21182-21190].
- Compute is raw MMIO — cfg-register writes plus an instruction stream through the Tensix instruction port, with SFPU fragments pre-compiled into count-pinned **[SFPU replay windows](../ch3_raw_mmio_sfpu_replay/03-sfpu-replay-windows.md)**; no llk, no CBs, TRISCs idle [INFER, src/brisc.cpp]. (Mechanism: Ch3.)
- Cross-chip collectives are device-initiated — PCIe **[peer-BAR DMA](../ch5_cross_chip_transports/01-pcie-peer-bar-dma.md)** and the ERISC **[message service](../ch5_cross_chip_transports/02-erisc-message-service.md)**, the latter the whole of `src/erisc.cpp` [INFER, src/erisc.cpp] — never a host CPU in the loop [INFER, src/brisc.cpp]. (Transports: Ch5.)
- The record list is the trace, the program, and the schedule at once — auditable and patchable as data, with no capture-vs-replay distinction [INFER, src/constants.h].

The host's role shrinks to setup and verification: open devices, quantize and load weights into DRAM, build and load the record lists, then read results and check them against the CPU **[oracle](../ch6_host_driver_weights_oracle/05-cpu-oracle.md)** (Ch6). The device runs the token loop alone.

## Side-by-side contrast

| Aspect | Conventional tt-metal decode (assumed audience background) | regen engine |
| --- | --- | --- |
| Resident program | Per-kernel programs, loaded/launched per op | One interpreter image per core, loaded once at firmware load |
| Per-step host cost | Dispatch + launch per op, per token | None — the host built the record list once per batch configuration |
| Compute interface | llk/ckernels over CBs | Raw MMIO cfg writes + instruction port; SFPU replay windows |
| TRISCs | Run the compute kernels | Idle — BRISC issues, Tensix executes |
| Data movement | CB flow control between kernels | Readers own all synchronization; weight **[strips](../ch4_dram_streaming_discipline/02-strip-claims-and-reader-placement.md)** claimed ahead |
| Cross-chip I/O | CCL ops on host-orchestrated meshes | Device-originated peer-BAR DMA + ERISC message service |
| Dispatch-overhead mitigation | Traces, program cache | Not needed — nothing is dispatched per step |
| The "trace" | Captured from execution, then replayed | The record list IS the trace; no capture phase exists |
| Determinism threats | Allocator, program cache, toolchain drift | Pinned toolchain, static_assert-pinned instruction counts, linker-asserted image budget (Ch7) |

The regen column is derived from reading the device source [INFER, src/brisc.cpp] except where separately cited; the conventional column is the background the guide assumes, not a claim about regen.

## The five design principles

1. **Records as data, not programs.** The model's control flow for a decode step is a data structure the host can audit, patch, and re-emit without touching device code; the device never recompiles. This eliminates phase boundaries and per-step dispatch in one move, and it is what makes the record list double as the trace. Substantiated by [Ch2 — The Record-Driven Execution Model](../ch2_record_driven_execution/index.md).
2. **The engine is its own runtime.** BRISC drives Tensix directly: configuration writes at the cfg base, the instruction port, and pre-compiled SFPU replay windows replace llk, CBs, and TRISC kernels (Ch3 `05`). Substantiated by [Ch3 — Raw-MMIO Compute and SFPU Replay](../ch3_raw_mmio_sfpu_replay/index.md).
3. **DRAM is the budget.** Decode on this model is bandwidth-bound, so the performance methodology is a streaming discipline: strip claims-ahead, two strips in flight, near-channel reader placement, DRAM prefetch windows, NoC-ID budgeting. The per-token bytes and the streaming-window rate are envelope rows 5-7 in [`05-measured-envelope-and-evidence.md`](./05-measured-envelope-and-evidence.md); the roofline exposition is [Ch4](../ch4_dram_streaming_discipline/index.md)'s. Substantiated by [Ch4 — DRAM Streaming Discipline](../ch4_dram_streaming_discipline/index.md).
4. **The device owns its cross-chip I/O.** A round is executed by the leads themselves — over peer-BAR DMA or the ERISC message service — with no host CPU on the path, keeping the no-host rule true for I/O as well as compute. Substantiated by [Ch5 — Cross-Chip Transports](../ch5_cross_chip_transports/index.md).
5. **Determinism and the oracle make the numbers citable.** A pinned toolchain (sfpi 7.76.0 [FACT, rules.py:12]) and static_assert-pinned instruction counts make builds instruction-level reproducible (Ch7), and the `glm.cpp` CPU oracle checks device output end-to-end (Ch6 `05`). Without this pair, neither the notes corpus nor this guide's envelope could be trusted. Substantiated by [Ch7 — Build Methodology and Determinism](../ch7_build_determinism/index.md) and [Ch6 — Host Driver, Weights Pipeline, and the CPU Oracle](../ch6_host_driver_weights_oracle/index.md).

## Why it pays — the headline numbers

The engine serves GLM-5.3-Flash single-user at 165.3 t/s (6.05 ms/token) near a hard DRAM floor, not a dispatch tax. Short-context rate, repeat stability, and context decay are envelope rows 1-4 in [`05-measured-envelope-and-evidence.md`](./05-measured-envelope-and-evidence.md); the roofline decomposition of the 6.05 ms is [Ch4](../ch4_dram_streaming_discipline/index.md)'s.

## What the host still does

The inversion removes the host from the dispatch and synchronization path, not from the token's edges: the host owns the embedding lookup before the step list and the final token read after it, with the measured edge costs stated on the walk they describe ([`02-anatomy-of-a-token.md`](./02-anatomy-of-a-token.md)'s step 0). The exact boundary is Ch2 `05`'s [elimination argument](../ch2_record_driven_execution/index.md); the builder and serving loop on the host side of it are Ch6's.

---

← [chapter index](./index.md) | ↑ [guide root](../index.md)

**Next:** [`02-anatomy-of-a-token.md`](./02-anatomy-of-a-token.md)
