# Core Components

A Tensix Core is the fundamental computational unit in every Tenstorrent chip. Each chip contains a grid of these cores -- for example, Wormhole B0 has an 8x10 grid and Blackhole has a larger array -- all sharing the same internal architecture. Understanding the components of a single Tensix Core is the key to understanding how LLKs work, because every LLK targets exactly one core.

## Four Major Components

A Tensix Core consists of four major parts:

```
+---------------------------------------------------+
|                   Tensix Core                      |
|                                                    |
|  +---------------------------------------------+  |
|  |              L1 SRAM (Local Memory)          |  |
|  +---------------------------------------------+  |
|                                                    |
|  +------------------+    +------------------+      |
|  |   NoC Router 0   |    |   NoC Router 1   |     |
|  +------------------+    +------------------+      |
|                                                    |
|  +---------------------------------------------+  |
|  |         Five RISC-V Processors               |  |
|  |  [BRISC] [NCRISC] [TRISC0] [TRISC1] [TRISC2]| |
|  +---------------------------------------------+  |
|                                                    |
|  +---------------------------------------------+  |
|  |             Tensix Engine                    |  |
|  |  (Unpackers, FPU, SFPU, Packers, ...)       |  |
|  +---------------------------------------------+  |
+---------------------------------------------------+
```

### 1. L1 SRAM

L1 is the local memory within each Tensix Core. It serves two purposes:

- **Tensor storage** -- Input tiles are placed into L1 before computation, and output tiles are written back to L1 after computation. All data that the Tensix Engine operates on must reside in L1.
- **Program code** -- The firmware and kernel code for all five RISC-V processors is stored in L1.

L1 is the only memory directly accessible to the core's unpackers and packers. Data arrives in L1 from DRAM or other cores via the NoC routers, and LLK operations move data between L1 and the Tensix Engine's internal register files.

### 2. Two NoC Routers

Each Tensix Core contains two Network on Chip (NoC) routers that connect it to the rest of the chip. These routers handle:

- **Inter-core data movement** -- Transferring tiles between Tensix Cores.
- **DRAM access** -- Reading input data from and writing results to off-chip DRAM.

The NoC routers are managed by BRISC and NCRISC (see [RISC-V Processors](./riscv_processors.md)). From the LLK perspective, NoC communication is outside the scope of compute kernels -- LLKs assume that data has already been placed into L1 and that results will be read from L1 by a separate data movement kernel. This separation of concerns is a core design principle: LLKs focus exclusively on the compute pipeline within a single Tensix Core.

### 3. Five RISC-V Processors

The Tensix Core contains five 32-bit, in-order, single-issue RISC-V cores: BRISC, NCRISC, TRISC0, TRISC1, and TRISC2. The three TRISC processors execute LLK code, each pushing Tensix ISA instructions to its corresponding thread in the Tensix Engine. See [`riscv_processors.md`](./riscv_processors.md) for the full breakdown of each processor's role.

### 4. Tensix Engine

The Tensix Engine is the Tensix Core's hardware accelerator, described in detail in [`tensix_engine.md`](./tensix_engine.md).

## How LLKs Use These Components

LLKs orchestrate the unpack-math-pack pipeline: TRISC0 unpacks tiles from L1 into the source registers, TRISC1 performs computation via the FPU or SFPU, and TRISC2 packs results back to L1. The full pipeline and the register files that connect its stages are covered in [`data_flow.md`](./data_flow.md).

---

**Next:** [`riscv_processors.md`](./riscv_processors.md)
