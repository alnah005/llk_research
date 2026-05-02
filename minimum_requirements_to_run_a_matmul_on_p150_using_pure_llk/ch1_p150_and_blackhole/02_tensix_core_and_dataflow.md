# The Tensix Core and the MatMul Data Path

## Tensix Core Structure

Every Tenstorrent chip -- including the Blackhole chip inside the P150 -- is built from a grid of **Tensix Cores**. Each Tensix Core is a self-contained compute unit with its own memory, processors, and data movement hardware. When we run a matrix multiplication through LLK, all of the actual computation happens within a single Tensix Core. Understanding its internal structure is essential before writing any LLK kernel.

A Tensix Core consists of four major subsystems:

1. **L1 SRAM** -- Local scratchpad memory that stores input and output tensors, as well as the program code for all five RISC-V processors. All data entering or leaving the Tensix Engine passes through L1.

2. **Two Routers** -- Network-on-chip (NOC) interfaces that manage data movement between Tensix Cores and off-chip DRAM across the device.

3. **Five RISC-V Processors** -- 32-bit, in-order, single-issue cores:

   | Core       | Full Name        | Role                                                                   | LLK Category     |
   |:-----------|:-----------------|:-----------------------------------------------------------------------|:------------------|
   | **BRISC**  | Boot RISC        | Handles NOC communication, board-level setup, and TRISC lifecycle management | --                |
   | **NCRISC** | NOC RISC         | Manages NOC data transfers between cores and DRAM                      | --                |
   | **TRISC0** | Tensix RISC 0    | Controls the **Unpack** thread of the Tensix Engine                    | `llk_unpack_*`    |
   | **TRISC1** | Tensix RISC 1    | Controls the **Math** thread of the Tensix Engine                      | `llk_math_*`      |
   | **TRISC2** | Tensix RISC 2    | Controls the **Pack** thread of the Tensix Engine                      | `llk_pack_*`      |

4. **Tensix Engine** -- The hardware accelerator that performs all mathematical operations, controlled by the three TRISC processors.

BRISC and NCRISC handle data staging -- getting tiles into and out of L1 memory. The three TRISC cores are the ones that actually execute LLK code. Each TRISC core runs firmware compiled from C++ source that includes LLK API headers and emits Tensix ISA instructions to its corresponding coprocessor thread. Each TRISC core is compiled as a separate ELF binary (see Section 3 for the compilation model).

## The Tensix Engine: A Three-Threaded Coprocessor

The Tensix Engine is a multi-threaded, single-issue, in-order processor that uses the **Tensix ISA** -- a custom instruction set distinct from RISC-V. The three TRISC processors listed in the table above each drive one of the Tensix Engine's three concurrent threads (T0, T1, T2). Each TRISC core compiles and pushes Tensix ISA instructions to its corresponding thread. The three threads share a set of backend execution units, but each thread has an independent frontend pipeline that processes instructions in order. The backend can execute instructions from different threads in parallel, with re-ordering across threads when there are no data dependencies. This is how the unpack, math, and pack stages overlap and pipeline.

### Shared Backend Execution Units

The Tensix Engine's backend contains several specialized hardware blocks:

| Unit                    | Function                                                                |
|:------------------------|:------------------------------------------------------------------------|
| **Sync Unit**           | Thread synchronization via semaphores and mutexes                       |
| **Unpackers** (0 and 1) | DMA engines that move data from L1 to source registers, with format conversion |
| **Matrix Unit (FPU)**   | Matrix multiplication and element-wise operations on low-precision data |
| **Vector Unit (SFPU)**  | 32-lane SIMD engine for 32-bit float/int operations (sigmoid, exp, etc.) |
| **Packers**             | DMA engines that move data from Destination register to L1, with format conversion |
| **Scalar Unit (ThCon)** | General-purpose scalar operations and atomic L1 operations              |
| **Configuration Unit**  | Manages hardware configuration registers                                |
| **Mover**               | Bulk L1 data movement (DMA)                                            |
| **Miscellaneous Unit**  | Address counter management                                              |

For example, the Math thread (T1) must wait until the Unpack thread (T0) has finished loading data into the Source registers before it can begin computation. The details of the synchronization protocol are covered in Chapter 3.

The two unpackers have distinct connectivity that is important for understanding the matmul data path:

- **Unpacker 0** is connected to the **Source A** register and the **Destination** register.
- **Unpacker 1** is connected to the **Source B** register only.

This asymmetry means the two operands of a matrix multiplication follow different hardware paths.

## Register Files

The Tensix Engine has three categories of register files that sit along the data path:

### Source A and Source B Registers

The Source registers are the input operands for the FPU. They are two-dimensional structures, each capable of holding tile data with elements up to 19 bits wide (in current architectures).

- **Source A** is loaded by Unpacker 0.
- **Source B** is loaded by Unpacker 1.

Both Source A and Source B are **double-buffered**: each one contains two independent memory banks, each large enough to hold one tile. While the FPU reads from one bank, the Unpacker can simultaneously write new data into the other bank. When the current computation finishes, the banks are swapped (via the `SETRWC` instruction), and the FPU immediately begins working on the freshly loaded data without stalling.

This double-buffering mechanism is central to sustaining high throughput. It allows the Unpack and Math stages to overlap -- the Unpacker is preparing the next tile while the FPU processes the current one.

### Destination Register

The Destination register stores the output of FPU computations and serves as both the input and output for SFPU operations. It is larger and more flexible than the Source registers:

- It can store between 4 and 16 tiles depending on the data format and configuration. When `is_fp32_dest_acc_en` is true (32-bit accumulation mode), the register stores half as many tiles because each element occupies twice the space.
- It supports 32-bit data elements (unlike Source registers, which are limited to 19 bits), making it suitable for high-precision accumulation.
- It is double-buffered at the half level via `DstSync::SyncHalf` mode: while the Packer reads from one half, the Math thread can write to the other half. The `dest_offset_id` variable in the ckernel infrastructure tracks which half is currently active, toggling via `update_dest_offset_id()`.

For matrix multiplication, the Destination register serves as the accumulator. The FPU accumulates dot-product results into Destination rows across multiple inner-loop iterations.

## The Matrix Multiplication Data Path

A matrix multiplication on the Tensix Engine follows a precise data path through the hardware. Understanding this flow is prerequisite to programming the three TRISC threads correctly.

### Step-by-Step Flow

The end-to-end path for a matrix multiply of two tiles is:

$$
\text{L1} \xrightarrow{\text{Unpack}} \text{SrcA, SrcB} \xrightarrow{\text{FPU (MVMUL)}} \text{Dest} \xrightarrow{\text{Pack}} \text{L1}
$$

In detail:

1. **L1 to Source Registers (Unpack -- T0/TRISC0)**

   TRISC0 programs both Unpacker 0 and Unpacker 1 through the `llk_unpack_AB_matmul` API. During this transfer, the Unpackers perform hardware-accelerated data format conversion through specialized circuits called *gaskets* -- for example, converting BFP8 data stored in L1 into the 19-bit format required by the Source registers.

   **Operand mapping note:** The mapping from logical operands to source registers is counterintuitive for matmul. The first logical operand (`in0` / `inA` in the codebase) is loaded into **Source B**, while the second logical operand (`in1` / `inB`) is loaded into **Source A**. This is because the FPU computes D = B x A (Source B is the left matrix), so the "A matrix" of the matmul goes to SrcB and the "B matrix" goes to SrcA. The relevant comment in `llk_unpack_AB_matmul.h` confirms: `in0/inA - loaded to SrcB` and `in1/inB - loaded to SrcA`.

   For matmul specifically, the LLK uses `llk_unpack_AB_matmul.h`, which optimizes unpacking by supporting register reuse patterns where one operand tile can remain in a Source register while multiple tiles from the other operand are streamed in.

2. **Source Registers to FPU (Math -- T1/TRISC1)**

   TRISC1 issues `MVMUL` instructions to the Tensix Engine. The FPU executes:

   $$D[8, 16] = B[8, 16] \times A[16, 16]$$

   That is, it multiplies 8 rows of Source B against all 16 columns of a 16x16 face of Source A, producing 8 rows of output in the Destination register. The inner loop runs 4 times ($32 / 8 = 4$) to compute a full 32x16 face. The hardware automatically pairs columns of A with rows of B for the dot product.

   Each FPU cell combines a multiplier and adder, supporting an accumulated dot product operation. The result is accumulated directly into the Destination register, enabling multi-phase fidelity passes without moving data.

   The matmul kernel in `llk_math_matmul.h` programs this sequence into a **replay buffer** -- a small instruction cache in the Tensix Engine that can replay a recorded sequence of instructions without re-issuing them from the RISC-V core. The `ckernel_template` mechanism wraps the replay invocation with outer/inner loop counts and special end-of-loop instructions, forming the complete MOP (Macro Operation). The details of MOP construction are covered in Chapter 4.

3. **Destination Register to L1 (Pack -- T2/TRISC2)**

   Once the Math thread signals that computation is complete (via semaphore), TRISC2 issues Pack instructions that read the result tile from the Destination register, perform format conversion (e.g., from 32-bit accumulator format down to Float16_b for storage), and write the packed tile back to L1 memory. After packing is complete, the pack thread clears the used portion of the destination register (via the `ZEROACC` instruction) and signals the math thread that the destination is free for new results.

### Math Fidelity and Multi-Phase Computation

The Tensix FPU uses a limited number of bits for each multiplication. To achieve full precision for a given data format, multiple **fidelity phases** may be required. The `MathFidelity` setting controls this tradeoff:

- **LoFi** -- A single fidelity phase. Fastest, but lowest precision.
- **HiFi2** -- Two fidelity phases.
- **HiFi3** -- Three fidelity phases.
- **HiFi4** -- Four fidelity phases. Slowest, but highest precision.

The number of phases directly affects both accuracy and throughput. For matrix multiplication, the fidelity setting determines how many passes the FPU makes over the same source data before advancing to the next tile.

### FPU Operations Relevant to MatMul

The FPU cell supports four operations, but matrix multiplication uses only the first:

1. **Accumulated dot product** -- The core matmul operation. Each MVMUL instruction computes a partial dot product and accumulates the result into the Destination register.
2. Accumulated element-wise addition
3. Accumulated element-wise multiplication
4. Element-wise addition

The accumulated nature of the dot product is important: multiple MVMUL instructions can be issued in sequence, and their results are summed in the Destination register without requiring an explicit accumulate step.

### SFPU and MatMul

The SFPU (Special FPU) is **not** used during a basic matrix multiplication. It exists for operations that the FPU cannot perform -- transcendentals (sigmoid, exponential, reciprocal, square root), type conversions, and other element-wise unary functions. The SFPU is a 32-lane SIMD engine that reads from and writes to the Destination register.

The SFPU only becomes relevant to matmul in fused operations -- for example, computing a matrix multiply followed by a GELU activation, where the matmul result stays in the Destination register and the SFPU applies the activation in place before the Packer writes the final result to L1. Such fusion patterns are beyond the scope of a minimal matmul.

## Pipelining Through Double Buffering

The power of the three-thread architecture becomes clear when processing multiple tiles. The pipeline operates as follows:

```
Time -->
T0 (Unpack): [Tile 0] [Tile 1] [Tile 2] ...
T1 (Math):          [Tile 0] [Tile 1] [Tile 2] ...
T2 (Pack):                 [Tile 0] [Tile 1] ...
```

Because Source A and Source B are double-buffered, T0 can begin unpacking the next tile pair into one bank while T1 reads the current tile pair from the other bank. Synchronization between the unpack and math threads is managed through the Sync Unit's **semaphore** mechanism -- the unpack thread posts a semaphore when a bank is filled, and the math thread waits on that semaphore before reading. See Chapter 3 for the complete semaphore protocol.

Similarly, the Destination register can be double-buffered using `DstSync::SyncHalf` mode: the math thread writes results to one half while the pack thread reads from the other half.

This overlap means that, in steady state, all three threads are active simultaneously, each working on a different tile in the sequence. The throughput bottleneck becomes whichever stage (unpack, math, or pack) takes the longest.

---

**Next:** [`03_llk_software_position.md`](./03_llk_software_position.md)
