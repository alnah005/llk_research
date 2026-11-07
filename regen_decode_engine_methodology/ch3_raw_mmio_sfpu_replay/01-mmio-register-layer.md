# The MMIO Register Layer

The entire register layer lives in [src/fw.h](../../../regen-glm5.3-flash/src/fw.h) — 84 lines: raw load/store macros, the hardware base addresses, the NoC register offsets, the stall/wait bit vocabulary, two small enums, and two emission helpers. Everything the engine knows about *where* Tensix state lives comes from this file; everything about *when* to write it lives in the op bodies ([`04-op-bodies.md`](./04-op-bodies.md)). This file walks the layer bottom-up, then shows how the engine sequences cfg writes per op — doing, by hand, the work llk/ckernel init code does in a conventional stack — and closes with how the host reaches the same space through BAR0 at load time.

**Scope:** covers Question 2, part 1 — the cfg register layer and its per-op sequencing. Mechanism only: the performance consequence of the live-config cache is measured in Ch4's ledger and is not restated here.

## The base addresses

| Symbol | Address | Defined | Used by |
| --- | --- | --- | --- |
| `NOC0_BASE` | `0xFFB20000u` | [FACT, src/fw.h:16] | NoC 0 command registers on this core |
| `NOC1_BASE` | `0xFFB30000u` | [FACT, src/brisc.cpp:694] (brisc image), [FACT, src/erisc.cpp:22] (erisc image) | NoC 1 command registers — defined **per image**, not in `fw.h` |
| `NOC_OVERLAY_BASE` | `0xFFB40000u` | [FACT, src/fw.h:17] | NoC overlay space |
| `TENSIX_REGFILE_BASE` | `0xFFE00000u` | [FACT, src/fw.h:18] | the RISC-V-visible data-movement register file (e.g. GPR63 completion handshake, [`02-instruction-port.md`](./02-instruction-port.md)) |
| `TENSIX_INST_BASE` | `0xFFE40000u` | [FACT, src/fw.h:19] | the instruction port ([`02-instruction-port.md`](./02-instruction-port.md)) |
| `TENSIX_CFG_BASE` | `0xFFEF0000u` | [FACT, src/fw.h:20] | the Tensix configuration register file — this file |
| `RISCV_DEBUG_REG_INSTRN_BUF_STATUS` | `0xFFB120A8u` | [FACT, src/fw.h:21] | per-thread instruction-buffer status (probe input, [`05-correctness-discipline.md`](./05-correctness-discipline.md)) |

Two details worth a porter's attention. First, `NOC1_BASE` is deliberately *not* in `fw.h`: each image defines it where it needs it, with the same value `0xFFB30000u` in both the brisc and erisc images [FACT, src/brisc.cpp:694; src/erisc.cpp:22] — an incidental-constant smell that [Ch8](../ch8_tt_metal_port/index.md) takes up. Second, the addresses are raw physical MMIO from the core's perspective: every access is a plain volatile store or load through the macros

```c
#define PHYS_RD32(addr) (*(volatile uint32_t *)(addr))
#define PHYS_WR32(addr, data) ...
#define FENCE() __asm__ volatile("fence rw, rw" ::: "memory")
```

[FACT, src/fw.h:9-15] — no abstraction, no address window, no runtime call between the engine and the hardware.

## What a cfg-word write is, mechanically

The whole configuration layer is one helper:

```c
static void cfg(uint32_t reg, uint32_t value) {
    PHYS_WR32(TENSIX_CFG_BASE + reg * sizeof(uint32_t), value);
}
```

[FACT, src/brisc.cpp:137-139]. A cfg register is addressed by *index* (the write scales it by 4 into the `0xFFEF0000` file), and one 32-bit word carries an op's worth of packed fields. The named setters show the pattern:

- `pack_config(read_32b, fp32_out, l1_accumulate)` writes the pack read mode (cfg 18), pack format control (cfg 70 — `0x00000001u` for FP32 out, `0x00000551u` otherwise), and the L1-accumulate bit (cfg 71) [FACT, src/brisc.cpp:157-163].
- `math_pack_config(src_a, src_b, pack_intermediate, fpu_fp32)` packs all four format selections into cfg 1's bit fields — SrcA format at bit 17, SrcB at 21, pack-intermediate at 25, FPU-fp32 at 29 [FACT, src/brisc.cpp:170-174].
- `unpack0_convert_config(...)` writes the unpacker 0 input format and dimensions (cfg 64/65), the output format plus transpose/unpack-to-Dst bits (cfg 72/73), and one `TTI_SETC16` SRCA-context override [FACT, src/brisc.cpp:178-186]; `unpack1_descriptor` writes cfg 112 [FACT, src/brisc.cpp:193-195]; `unpack0_destination` sets the Dst row extent in cfg 86 [FACT, src/brisc.cpp:196-199].

The `DataFormat` values written (FP32/TF32/BF16/BFP8/BFP4) come from `fw.h`'s own enum [FACT, src/fw.h:49-55], and the unpacker/pack context bases are named by the `L1Base` enum — Pack 69, Unpack0 76, Unpack1 124 [FACT, src/fw.h:57-61]. `set_base_cfg` computes a base register's value from an L1 address as `addr / 16 - 1`, pointing an unpacker at the buffer an op body just filled; a pipelined variant routes the same value through a GPR and a `TT_OP_WRCFG` instruction so the previous pack is not disturbed [FACT, src/brisc.cpp:862-869]. Here it is all engine code, visible in the op body's composition, with nothing between the setter and the MMIO write [INFER, src/brisc.cpp:157-199].

## How the engine sequences cfg writes per op

Two disciplines make hand-written cfg sequencing safe:

**1. Every setter invalidates the live-config cache.** The engine tracks what unpack/math/pack configuration is currently live — `LiveConfig`'s values are `Unknown`, `Bfp4Bf16`, `Bfp4Fp32`, `Bfp8Bf16`, `Bfp8Fp32`, `Bf16Fp32` — "so back-to-back matvecs skip rewriting it. Every setter of those registers forgets it; strip_config records it" [FACT, src/brisc.cpp:141-149 — the enum and its comment]. `config_changed()` is called by every config-changing setter [FACT, src/brisc.cpp:153-155], and `strip_config` returns early when the requested format pair is already live [FACT, src/brisc.cpp:1410-1427], so the next op re-establishes the full state it needs instead of inheriting half of it. The invariant this protects is *ops see a known configuration*, re-established lazily rather than defensively rewritten [INFER, src/brisc.cpp:141-186].

**2. The base configuration is established once per image start.** `hw_init()` writes the counter/context/format defaults with plain `cfg()` calls — cfg 2, 20, 24, 28, 50, 56, 59, 84, 113, 120, 121 [FACT, src/brisc.cpp:1140-1157] — and then, critically for the SFPU fragments, fixes the **address modifiers** in one `TTI_SETC16` table: AddrMod0 nop, AddrMod1 advances Dst by four rows, AddrMod2 `SrcA += 16, SrcB += 1`, AddrMod4 rewinds SrcA/SrcB and increments fidelity, AddrMod7 clears them, plus the pack-side AddrMods [FACT, src/brisc.cpp:1201-1220]. The `kernels.sfpu` fragments and replay windows reference these modifiers by number (`addrmod=1` everywhere a store advances Dst four rows); the fragments compile against a *fixed* AddrMod table, which is part of why window instruction counts are stable [INFER, src/brisc.cpp:1201-1220; src/kernels.sfpu]. Zeroing SrcA/SrcB banks and Dst zero-flags (`TTI_ZEROSRC`, `TTI_ZEROACC`) is also done here, once [FACT, src/brisc.cpp:1199-1200].

## The stall/wait vocabulary

Synchronization with the Tensix engines is register-flag arithmetic, defined in `fw.h`: `STALL_PACK/UNPACK/THCON/MATH/CFG/SFPU` bits to *hold* an engine, `WAIT_UNPACK0/PACK/FPU/SFPU` bits to *wait* for one [FACT, src/fw.h:38-47]. The engine's `wait_tensix(mask)` wraps the canonical handshake — stall THCON, push a completion flag into a reserved GPR in the regfile at `TENSIX_REGFILE_BASE`, and poll it back [FACT, src/brisc.cpp:317-329 — the full mechanism is [`02-instruction-port.md`](./02-instruction-port.md)'s]. Every op body that wants an engine quiesced issues one `TTI_STALLWAIT` / `wait_tensix` pair with exactly these bits; nothing else in the image touches engine progress [INFER, src/brisc.cpp:317-329].

## BAR0: how the host sees this space (brief)

The host never drives compute, but it loads the firmware and programs the windows the device later uses. In [src/tt_backend.cpp](../../../regen-glm5.3-flash/src/tt_backend.cpp), BAR0 is mapped at `0x100000000` and carved into 2 MB **BAR0 TLB windows** (the PCIe BAR window sense, [Ch5](../ch5_cross_chip_transports/01-pcie-peer-bar-dma.md)'s) — the leads' windows, the multicast window, the peer-round and lead-cast windows, and DRAM-channel windows — whose TLB configs live at BAR0 offset `0x1FC00000`, each pinned by `static_assert` against the device-side `_WINDOW` constants [FACT, src/tt_backend.cpp:69-87]. `silicon_bar_offset()` translates a tt-lab window number to the KMD id this process owns and *refuses* anything outside the engine's own windows — "silicon BAR0 access outside tt-lab's windows" throws; "BAR0 registers past the TLB configs stay where they are" [FACT, src/tt_backend.cpp:516-541]. The device-side `0xFFExxxxx` Tensix space above is *not* host-reachable through BAR0; it is the core-local half of the same discipline, and the host's involvement ends at firmware load ([Ch6 `01-backend-open-and-load.md`](../ch6_host_driver_weights_oracle/01-backend-open-and-load.md) owns the full open/load path).

## Ownership fences

- The NoC register *offsets* (`NOC_TARG_ADDR_*`, `NOC_RET_ADDR_*`, `NOC_PACKET_TAG`, `NOC_CTRL`, `NOC_ID_LOGICAL`; command-buffer stride `NOC_CMD_BUF_OFFSET` `0x800`) are listed in `fw.h` [FACT, src/fw.h:23-36], but their *use discipline* — IDs, budgets, pacing — is [Ch4](../ch4_dram_streaming_discipline/index.md)'s.
- The instruction port (`TENSIX_INST_BASE`) is the next file's whole subject.
- Why the image re-verifies its MMIO assumptions at start (the SrcA bank test) is [`05-correctness-discipline.md`](./05-correctness-discipline.md)'s.
- That the build pins this whole layer byte-for-byte (sfpi, linker) is [Ch7](../ch7_build_determinism/index.md)'s.

---

← [chapter index](./index.md) | ↑ [guide root](../index.md)

**Next:** [`02-instruction-port.md`](./02-instruction-port.md)
