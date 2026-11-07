# The Instruction Port

Compute is driven through a single 32-bit MMIO push: a write to `TENSIX_INST_BASE` at `0xFFE40000` places one Tensix instruction into the stream [FACT, src/fw.h:19]. The interpreter — the BRISC loop from [Ch2 `02-interpreter-loop.md`](../ch2_record_driven_execution/02-interpreter-loop.md) — pushes unpack, math, pack, SFPU, and data-movement instructions itself, as the op body's code runs. This file covers the two emission paths, how the interpreter reads completion back, how operands reach the engines without circular buffers, what "TRISCs idle" means operationally, and how the engine protects the port's own state.

**Scope:** covers Question 2, part 2 — the instruction port, operand setup, TRISC idling, and the no-CB contract's mechanism half. Synchronization of *data movement* (who waits for what stream) belongs to [Ch4](../ch4_dram_streaming_discipline/index.md); the engine-vs-llk *trade* argument is [`06-what-no-llk-means.md`](./06-what-no-llk-means.md).

## Two emission paths

Every Tensix instruction in the image is one 32-bit word, encoded by the `TT_OP(opcode, params)` convention — opcode in the top byte [FACT, src/ckernel_ops.h:11]. The word reaches the hardware by one of two paths, selected by the compiler:

```c
static inline void tensix_instruction(uint32_t instruction) {
    if (__builtin_constant_p(instruction)) {
        INSTRUCTION_WORD(instruction);
    } else {
        PHYS_WR32(TENSIX_INST_BASE, instruction);
    }
}
```

[FACT, src/fw.h:63-72 — the header's own comment marks the boundary: "Select after inlining/constant propagation; runtime operands use the instruction port", with an open upstream item noted for `__builtin_rvtt_submit` [FACT, src/fw.h:65-66]].

- **Compile-time instructions** go through `INSTRUCTION_WORD` — the `.ttinsn` assembler directive that "Swizzle[es] 32 bits into the instruction stream" [FACT, src/ckernel_ops.h:12]. All `TTI_*` macros in [src/ckernel_ops.h](../../../regen-glm5.3-flash/src/ckernel_ops.h) (1,072 lines, auto-generated — "Auto-generated file, do not modify!" [FACT, src/ckernel_ops.h:5-6]) lower to this; the emitted firmware binary *contains* the instruction stream inline. The table is the auditable menu: every instruction an op body can issue is enumerable here with its opcode — e.g. `TT_OP_MVMUL` (opcode `0x26`) [FACT, src/ckernel_ops.h:366-369], `TT_OP_STALLWAIT` (`0xa2`) [FACT, src/ckernel_ops.h:881], `TT_OP_SFPLOADI` (`0x71`) [FACT, src/ckernel_ops.h:678], and the SFPU replay instruction itself (`0x04`, [`03`](./03-sfpu-replay-windows.md)) [FACT, src/ckernel_ops.h:423-428].
- **Runtime instructions** — words whose operands only exist at run time — are pushed through the MMIO port by `tensix_instruction()`. The op bodies use this whenever an operand is data: e.g. `kda_rows` loads each state row's BF16 key and FP32 decay as `SFPLOADI` immediates computed from memory, pushing each word through the port [FACT, src/brisc.cpp:5695-5708].

For runtime halfword assembly the layer provides `pack_u16(low, high)`, which on non-constant operands uses a single packed instruction via inline assembly because "BH supports PACK, but not the full Zbkb extension required by GCC's pattern" [FACT, src/fw.h:74-84 — including the code comment on the BH limitation]. `kda_rows` uses it to build the second halfword of a two-part `SFPLOADI` [FACT, src/brisc.cpp:5705-5707].

The generated header also carries a `TT_`-prefix family that would store into `ckernel::instrn_buffer[0]` — a symbol nothing in this tree defines; the variants are vestige of the header's tt-metal ckernel-template origin, used only by the proxy/sim paths the guide's reader can skip [INFER, src/ckernel_ops.h:18-20].

## Reading completion back: the handshakes

Instructions execute asynchronously on the Tensix engines, so the interpreter needs a completion readback. `wait_tensix(wait_mask)` is the whole mechanism [FACT, src/brisc.cpp:317-329]: BRISC zeroes GPR63 in the Tensix regfile at `TENSIX_REGFILE_BASE + 63 × 4` — "GPR63 is reserved for completion; address construction uses GPR0" [FACT, src/brisc.cpp:319] — stalls its own ThCon pipeline until the requested engine flag is set (`TTI_STALLWAIT(STALL_THCON, wait_mask)`), posts a `SETDMAREG` write of 1 to GPR126, and spins until GPR63 reads nonzero. The code additionally drains the BRISC store buffer first, because "FENCE alone does not drain a BRISC store" [FACT, src/brisc.cpp:323]. The stall/wait mask vocabulary is [fw.h](../../../regen-glm5.3-flash/src/fw.h)'s [FACT, src/fw.h:38-47].

A second, finer handshake gates the unpacker: "After the unpacks issued so far finish, the Tensix writes the low bits of `seq` to GPR2" — `mvmul_post_unpacked(seq)` posts a sequence number through the data-movement regfile, and `mvmul_wait_unpacked` spins on GPR2 until the unpacker has caught up [FACT, src/brisc.cpp:1438-1451]. The strip matvec passes this as its `wait_chunk` callback: "`wait_chunk(c)` returns once chunk c has landed: unpacker 0 reads ahead within a chunk, so weights still arriving unpack stale" [FACT, src/brisc.cpp:1455-1460] — per-chunk gating inside one strip, finer than the per-op `wait_tensix` [INFER, src/brisc.cpp:1455-1468].

## How the interpreter streams instructions

An op body is a straight-line sequence of these pushes, interleaved with the cfg writes of [`01-mmio-register-layer.md`](./01-mmio-register-layer.md):

1. Configure: cfg writes set formats, dims, context bases (e.g. `mvmul_bfp8_setup()` [FACT, src/brisc.cpp:1226-1231]).
2. Push: unpack instructions (`TTI_UNPACR`) load operands from L1 into SrcA/SrcB or Dst; math instructions (`TTI_...`/replay) compute; pack instructions (`TTI_PACR`) store results [FACT, src/ckernel_ops.h — macro families; src/brisc.cpp op bodies].
3. Wait: `wait_tensix(mask)` stalls on the engine's flag set and completes via the GPR63 handshake [FACT, src/brisc.cpp:317-329].

There is no instruction queue the host fills and no dispatcher: the record's op body *is* the instruction stream's schedule, and instruction pushes from the interpreter and MMIO writes from `tensix_instruction()` interleave in plain program order. Unicast NoC writes stay in issue order on a static VC — "writes to one destination stay in issue order" [FACT, src/brisc.cpp:80-81 — the comment]; instruction pushes are sequential code, so the engine's own issue order holds by construction [INFER, src/brisc.cpp].

The densest sustained port-traffic pattern in the decode path is the loop-carried immediate stream [INFER, src/brisc.cpp:5660-5729]: `kda_rows` pushes three runtime words per state row (a key halfword OR-ed into an `SFPLOADI`, then the FP32 decay as two halfword loads) and only the fourth word is the window replay [FACT, src/brisc.cpp:5699-5707] — with `#pragma GCC unroll 4` keeping four rows of pushes in flight [FACT, src/brisc.cpp:5699] and the loads skipping `volatile` "so they pipeline ahead of the instruction pushes" [FACT, src/brisc.cpp:5657-5659].

## Operand setup with plain L1 stores — and why there are no CBs

Operands never travel through circular buffers. The engine writes descriptors and data into L1 with word stores (`l1_store_words` is the generic template [FACT, src/brisc.cpp:218-227]), points an unpacker context at the L1 address (`set_base_cfg(L1Base::Unpack0, addr)` [FACT, src/brisc.cpp:862-864, 920-922]), and pushes an unpack instruction that reads L1 directly. Data flow is L1/DRAM → unpacker → Dst/SrcA/SrcB → packer → L1, with every stage's ordering owned by *the code that issues it*: the op body waits (`TTI_STALLWAIT`, `wait_tensix`) exactly where a data dependency requires it [FACT, src/brisc.cpp:317-329; src/fw.h:38-47]. The staging buffers' addresses are fixed `_ADDR` constants whose non-overlap is build-pinned — e.g. "SWIGLU gate buffer overlaps up buffer" and its sibling layout asserts [FACT, src/brisc.cpp:94-119].

This inverts the CB contract. A circular buffer makes the *producer* safe against the consumer by construction; here there is no buffer to size, no slots to count, no CB sync races to have — but also no construction-time protection: readers own all synchronization, and the protection is the op body's ordering discipline [INFER, src/brisc.cpp]. Strip streaming (the weight side of the same "readers own it" rule) is [Ch4 `02-strip-claims-and-reader-placement.md`](../ch4_dram_streaming_discipline/02-strip-claims-and-reader-placement.md)'s.

## What "TRISCs idle" means operationally

The image defines exactly one entry point: `_start` sets the stack pointer to `0xFFB02000` (the 8 KB LDM) and calls `brisc_main`; there is no second thread's code in the image [FACT, src/brisc_entry.S:8-10]. `brisc_main` initializes the hardware, runs the startup probes, and enters the record loop [FACT, src/brisc.cpp:23614-23626]. So "the TRISCs are idle" does not mean the Tensix engines are off — unpack/math/pack threads execute every instruction the interpreter pushes — it means:

- **no per-thread programs are loaded or launched** — nothing exists to launch [FACT, src/brisc_entry.S:8-10; src/brisc.cpp:23614];
- **all Tensix instructions are issued from one place** — the BRISC interpreter, through `.ttinsn` constants or the port [FACT, src/fw.h:63-72];
- **the dispatch problem tt-metal solves with a per-kernel grid launch does not arise** — the persistent interpreter *is* the schedule [INFER, src/brisc.cpp; Ch2 `05-eliminating-the-runtime.md`](../ch2_record_driven_execution/05-eliminating-the-runtime.md).

This is a choice, not a limitation: it buys a single, auditable instruction account per core (which [`05-correctness-discipline.md`](./05-correctness-discipline.md) pins) at the cost of serializing all instruction issue through one RISC-V thread [INFER, src/brisc.cpp].

## Protecting the port's own state

Two disciplines guard the port against inherited state:

- **Thread-empty probe at boot.** `require_idle_thread()` polls `RISCV_DEBUG_REG_INSTRN_BUF_STATUS` (at `0xFFB120A8` [FACT, src/fw.h:21]) until this core's Tensix thread reports both bits empty — it "reads 0x77 with all three threads empty and idle, and bits 0 and 4 are this thread's" — with a ~1 ms timeout; an earlier client's run halted mid-kernel "can leave the thread holding instructions that never issue, which only a reset clears", in which case a push would block forever — the core reports `TT_SRCA_THREAD_BUSY` and parks instead [FACT, src/brisc.cpp:1121-1137]. The status register is unmodeled in `ttsim`, whose threads start empty [FACT, src/brisc.cpp:1124-1125 — the comment].
- **SFPU replay buffer ownership.** The replay buffer at slot 0 is *state*: `hw_init` records the resident replay bodies into it at every image start [FACT, src/brisc.cpp:1104-1113, 1222-1223], and any op that borrows the buffer must re-record them before the next op can use a resident replay window — the borrower discipline [`03-sfpu-replay-windows.md`](./03-sfpu-replay-windows.md) details.

## Ownership fences

- The NoC *instruction* path (data-movement register file, `TTI_SETDMAREG` etc.) is used by op bodies but its transaction discipline is [Ch4](../ch4_dram_streaming_discipline/index.md)'s.
- The SFPU half of the instruction stream — the replay windows — is [`03-sfpu-replay-windows.md`](./03-sfpu-replay-windows.md).
- What regen gives up and gains by having no CB abstraction at all is argued in [`06-what-no-llk-means.md`](./06-what-no-llk-means.md).

---

← [`01-mmio-register-layer.md`](./01-mmio-register-layer.md) | ↑ [chapter index](./index.md) | ↑ [guide root](../index.md)

**Next:** [`03-sfpu-replay-windows.md`](./03-sfpu-replay-windows.md)
