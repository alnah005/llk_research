# Chapter 7: Replay Debugger Architecture -- Index

**Synthesized from 5 competing generator outputs (v1-v5), evaluated by independent reviewers.**

**Best scores:** v3 = 4.8/5.0 (best scoring matrices), v5 = 4.8/5.0 (best user workflows), v4 = 4.6/5.0 (best implementation code)

**Synthesis strategy:** v3 primary for Files 01 and 03, v5 primary for File 02, with corrections from all evaluations applied.

---

## Files

| # | File | Lines | Description |
|---|------|------:|-------------|
| 1 | `01_emulator_based_replay.md` | 719 | Spike emulator selection, snapshot loading, GDB RSP integration, SFPU handling, single-RISC and intra-Tensix modes, `tt-kernel-replay` CLI, fidelity levels (L0/L1/L2) |
| 2 | `02_on_hardware_and_host_model_replay.md` | 623 | On-hardware replay via RISC_DBG_CNTL (BH/Quasar) and ebreak+mailbox (WH), GDB RSP mapping, host functional model via `trisc.cpp`, BRISC-as-debug-agent, AddressSanitizer integration |
| 3 | `03_multi_core_replay_and_noc_coordination.md` | 558 | Multi-core replay strategies, NOC transaction modeling (instant/logged/synchronized), intra-Tensix 5-RISC coordination, dispatch emulation, practical scoping guidance |

**Total: 1,900 lines across 3 files**

---

## Key Findings

1. **Three replay targets with clear selection criteria**: Host model (fastest iteration, best GDB UX), emulator (no hardware needed, CB/semaphore modeling), on-hardware (exact fidelity, architecture-dependent)

2. **Architecture-dependent debug capabilities**:
   - **Wormhole**: No RISC_DBG_CNTL registers. Only ebreak+mailbox debugging. Prefer emulator/host model.
   - **Blackhole**: RISC_DBG_CNTL_0/1 at tensix.h:162-165 (raw registers only). Must port rvdbg_cmd from Quasar.
   - **Quasar**: Full rvdbg_cmd API (16 commands) + RISC-V Debug Module at 0x0300A000 + 8 hardware watchpoints

3. **`ckernel_riscv_debug.h` is Quasar-only** (under `tt_llk_quasar/common/inc/`). No equivalent exists for BH or WH.

4. **Host functional model scores highest** (90/130 weighted) for compute kernel debugging. Builds on existing `trisc.cpp` test infrastructure. AddressSanitizer catches L1 buffer overflows invisible on hardware.

5. **Single-TRISC replay covers ~70% of practical LLK bugs** because TRISCs never issue NOC transactions — all input data is in L1 CBs at kernel launch.

6. **Multi-core replay is tractable but expensive** (~3.2 GB for 64-core grid with 320 Spike instances). Practical scoping reduces to 2-4 cores for most bugs.

---

## Key Source Files Referenced

| File | Path | Relevance |
|------|------|-----------|
| `ckernel_riscv_debug.h` | `tt_llk_quasar/common/inc/` | rvdbg_cmd enum, RISC_DBG_CNTL protocol (Quasar only) |
| `tensix.h` (BH) | `tt-1xx/blackhole/tensix.h:162-165` | RISC_DBG_CNTL_0/1 register addresses |
| `tensix_functions.h` | `tt_metal/hw/inc/internal/tensix_functions.h:606-687` | WH/BH breakpoint API |
| `trisc.cc` | `tt_metal/hw/firmware/src/tt-1xx/trisc.cc` | Device firmware execution model |
| `trisc.cpp` | `tt_metal/third_party/tt_llk/tests/helpers/src/trisc.cpp` | Host functional model |
| `noc_debugging.hpp` | `tt_metal/impl/debug/noc_debugging.hpp` | NocWriteEvent/NocReadEvent structs |
| `overlay_reg_defines_debug.h` | `tt-2xx/quasar/overlay/meta/registers/` | Debug Module APB base address |
| `dev_msgs.h` | `tt_metal/hw/inc/hostdev/dev_msgs.h` | go_msg_t, launch_msg_t, sync protocol |

---

## Cross-Chapter Dependencies

- **Ch 2**: State inventory (regfile[64], config registers, L1, CBs, semaphores) defines what must be captured/replayed
- **Ch 4**: 5-RISC execution model and dispatch protocol define replay coordination requirements
- **Ch 5**: Debug hardware capabilities (by architecture) and E-taxonomy classify emulation fidelity
- **Ch 6**: `.ttksnap` schema and capture modes provide the input format for all replay targets
- **Ch 8**: Implementation phases (forward reference) — effort estimates rolled into Ch 8 roadmap
