# Tensix Engine

The Tensix Engine is the hardware accelerator at the heart of each Tensix Core. It is a three-way threaded coprocessor with a custom instruction set (Tensix ISA), purpose-built for the matrix and vector operations that dominate AI/ML workloads. The three TRISC processors do not execute Tensix computations themselves -- they push Tensix ISA instructions into the engine, which decodes and executes them on specialized hardware units.

## Thread Model

The Tensix Engine runs three concurrent threads, each fed by one TRISC processor:

| Thread | Name | TRISC Source | Responsibility |
|:------:|:-----|:-------------|:---------------|
| T1 | UNPACK | TRISC0 | Controls unpackers -- moves data from L1 into source/destination registers |
| T2 | MATH | TRISC1 | Controls FPU and SFPU -- performs matrix and vector computations |
| T3 | PACK | TRISC2 | Controls packers -- moves results from the destination register to L1 |

Each thread has its own instruction stream. TRISC0 pushes unpack instructions to Thread 1, TRISC1 pushes math instructions to Thread 2, and TRISC2 pushes pack instructions to Thread 3. The threads run independently and overlap in time, creating a pipelined execution model.

## Architecture: Frontend and Backend

The Tensix Engine is divided into two logical halves:

```
          TRISC0            TRISC1            TRISC2
            |                 |                 |
            v                 v                 v
   +----------------+ +----------------+ +----------------+
   | Frontend (T1)  | | Frontend (T2)  | | Frontend (T3)  |
   | Instruction    | | Instruction    | | Instruction    |
   | decode &       | | decode &       | | decode &       |
   | dispatch       | | dispatch       | | dispatch       |
   +----------------+ +----------------+ +----------------+
            \                 |                 /
             \                |                /
              v               v               v
   +---------------------------------------------------+
   |              Shared Backend                        |
   |                                                    |
   |  Sync Unit      Unpackers       Matrix Unit (FPU)  |
   |  Packers        Vector Unit (SFPU)                 |
   |  Scalar Unit (ThCon)   Configuration Unit          |
   |  Mover          Miscellaneous Unit                 |
   +---------------------------------------------------+
```

### Independent Frontends

Each thread has its own frontend pipeline that handles instruction fetch, decode, and dispatch. The frontends operate in-order within each thread -- instructions from a single thread are decoded and dispatched in the order they were issued by the corresponding TRISC. This means that TRISC0's unpack instructions are processed sequentially, TRISC1's math instructions are processed sequentially, and TRISC2's pack instructions are processed sequentially.

### Shared Backend Execution Units

All three frontend pipelines dispatch instructions into a shared pool of backend execution units. The backend can execute instructions from different threads out of order with respect to each other -- for example, an unpack instruction from T1 and a math instruction from T2 can execute simultaneously on different hardware units. This is what enables the pipeline overlap that gives the Tensix Engine its throughput.

The shared backend contains the following execution units:

| Unit | Also Known As | Function |
|:-----|:--------------|:---------|
| **Sync Unit** | -- | Manages hardware semaphores for inter-thread synchronization. Ensures that producers and consumers do not conflict (e.g., unpack completes before math reads the data). |
| **Unpackers** | Unpacker 0, Unpacker 1 | DMA engines that transfer tile data from L1 into Source A (Unpacker 0), Source B (Unpacker 1), or the Destination register (Unpacker 0). Include hardware gaskets for data format conversion. |
| **Matrix Unit** | FPU | The primary computation engine. A matrix of FPU cells that perform dot products, element-wise additions, and element-wise multiplications on data from the source registers, writing results to the Destination register. |
| **Vector Unit** | SFPU | A SIMD engine for specialized operations (sigmoid, exponential, reciprocal, etc.). Reads from and writes to the Destination register. Instantiated within the FPU. |
| **Scalar Unit** | ThCon (Thread Configuration) | Handles per-thread configuration register reads and writes. Each thread can set configuration parameters that affect how its instructions are executed. |
| **Configuration Unit** | -- | Manages global configuration state for the Tensix Engine. |
| **Packers** | -- | DMA engines that transfer tile data from the Destination register back to L1 memory. Include hardware gaskets for data format conversion. |
| **Mover** | XMOV | Handles data movement between internal register files and configuration registers. |
| **Miscellaneous Unit** | -- | Handles operations that do not fit into the other units. |

## In-Order vs. Out-of-Order Execution

To summarize the execution model:

- **Within a single thread:** Instructions are dispatched in-order by the frontend. A TRISC's instruction sequence is respected.
- **Across threads:** The backend can execute instructions from different threads simultaneously and out of order with respect to each other.
- **Synchronization:** When ordering between threads matters (e.g., math must wait for unpack to finish writing source registers), the Sync Unit enforces it via hardware semaphores. LLK functions insert the necessary synchronization instructions automatically, using stall conditions such as `p_stall::STALL_UNPACK`, `p_stall::STALL_MATH`, and `p_stall::STALL_PACK` (defined in `tt_llk_wormhole_b0/common/inc/ckernel_instr_params.h`).

## Tensix ISA vs. RISC-V ISA

It is important not to confuse the two instruction sets at play in a Tensix Core:

| Property | RISC-V ISA | Tensix ISA |
|:---------|:-----------|:-----------|
| **Executed by** | The five RISC-V processors (BRISC, NCRISC, TRISC0/1/2) | The Tensix Engine |
| **Instruction width** | 32-bit standard RISC-V | 32-bit custom Tenstorrent encoding |
| **Purpose** | General-purpose control flow, memory access, function calls | Controlling unpackers, FPU, SFPU, packers, synchronization |
| **Issued how** | Fetched from L1 by the RISC-V core | Written into the Tensix Engine's instruction FIFO by a TRISC |

When you write LLK code in C++, the compiler produces RISC-V machine code that runs on a TRISC. Inside that code, LLK functions use inline macros (like `TTI_STALLWAIT`, `TTI_SETC16`) to inject Tensix ISA instructions into the engine's instruction stream. The RISC-V core acts as a programmable controller that decides which Tensix instructions to issue and in what order.

---

**Next:** [`data_flow.md`](./data_flow.md)
