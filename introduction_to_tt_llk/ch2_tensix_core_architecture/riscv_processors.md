# RISC-V Processors

Each Tensix Core contains five 32-bit, in-order, single-issue RISC-V processor cores. They are identified in the codebase by short names and numeric thread IDs (defined in `tt_llk_wormhole_b0/common/inc/ckernel_defs.h`):

| Short Name | Thread ID | Full Name | LLK Relevant? |
|:-----------|:---------:|:----------|:-------------:|
| B (BRISC) | 0 | Board RISC-V | No |
| NC (NCRISC) | -- | NOC RISC-V | No |
| T0 (TRISC0) | 1 | Tensix RISC-V 0 | Yes |
| T1 (TRISC1) | 2 | Tensix RISC-V 1 | Yes |
| T2 (TRISC2) | 3 | Tensix RISC-V 2 | Yes |

## BRISC and NCRISC

BRISC and NCRISC handle NOC communication and board-level setup. They manage data transfers between L1 memory and DRAM, coordinate with other Tensix Cores across the chip, and perform initialization tasks. From the LLK developer's perspective, these two processors are outside the scope of compute kernels. Their work is done before LLK code runs (placing input data into L1) and after it completes (reading output data from L1).

## The Three TRISCs

The three TRISC processors are the ones that execute LLK code. Each TRISC controls a specific stage of the compute pipeline by issuing Tensix ISA instructions to its corresponding thread in the Tensix Engine:

```
TRISC0  ──(Tensix ISA instructions)──>  Thread 1 (UNPACK)
TRISC1  ──(Tensix ISA instructions)──>  Thread 2 (MATH)
TRISC2  ──(Tensix ISA instructions)──>  Thread 3 (PACK)
```

### TRISC0 -- Unpack Thread

TRISC0 controls the unpackers. Its LLK functions configure hardware DMA engines to transfer tile data from L1 memory into the Source A and Source B register files (or directly into the Destination register for certain SFPU workflows). During unpacking, hardware gaskets perform automatic data format conversion -- for example, converting a tile stored in BFloat16 in L1 into the internal register format.

Key LLK headers controlled by TRISC0 include:

- `llk_unpack_A.h` -- Unpack a single tile into Source A.
- `llk_unpack_AB.h` -- Unpack two tiles into Source A and Source B.
- `llk_unpack_AB_matmul.h` -- Optimized unpack for matrix multiplication with register reuse.
- `llk_unpack_tilize.h` -- Convert row-major data to tile format during unpack.

### TRISC1 -- Math Thread

TRISC1 controls both the FPU and the SFPU. It issues Tensix ISA instructions that trigger matrix and vector computations on data already present in the source or destination registers. The FPU reads from Source A and Source B and writes results to the Destination register. The SFPU reads from and writes to the Destination register.

Key LLK headers controlled by TRISC1 include:

- `llk_math_eltwise_binary.h` -- Element-wise add, subtract, or multiply via the FPU.
- `llk_math_matmul.h` -- Matrix multiplication via the FPU.
- `llk_math_eltwise_unary_sfpu.h` -- Unary operations (sigmoid, exp, reciprocal, etc.) via the SFPU.
- `llk_math_eltwise_unary_datacopy.h` -- Transfer a tile from Source A/B to the Destination register.

### TRISC2 -- Pack Thread

TRISC2 controls the packer. Its LLK functions configure a hardware DMA engine to transfer tile data from the Destination register back to L1 memory. Like the unpackers, the packer uses hardware gaskets for automatic data format conversion on the output path.

Key LLK headers controlled by TRISC2 include:

- `llk_pack.h` -- Pack a tile from the Destination register to L1 in tile format.
- `llk_pack_untilize.h` -- Pack a tile from the Destination register to L1, converting to row-major format.

## Two Instruction Sets

The TRISC processors execute RISC-V code that injects Tensix ISA instructions into the engine (see [Tensix Engine -- Tensix ISA vs. RISC-V ISA](./tensix_engine.md#tensix-isa-vs-risc-v-isa)).

## Concurrent Execution

All three TRISCs run concurrently. TRISC0 can be unpacking the next tile while TRISC1 is computing on the current tile and TRISC2 is packing the previous result. This overlap is essential for high throughput and is coordinated through hardware semaphores (e.g., `semaphore::UNPACK_SYNC`) that ensure a producer thread does not overwrite data before the consumer thread has finished reading it. The semaphore-based synchronization is handled within LLK functions, so kernel developers get correct pipelining by calling the standard LLK API.

---

**Next:** [`tensix_engine.md`](./tensix_engine.md)
