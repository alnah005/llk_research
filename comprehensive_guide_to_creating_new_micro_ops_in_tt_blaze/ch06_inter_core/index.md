# Chapter 6: Inter-Core and Inter-Device Operations

## Overview

Tenstorrent Wormhole and Blackhole devices contain a grid of Tensix cores, each
with private L1 SRAM, five RISC-V processors (BRISC, NCRISC, and three TRISCs),
and two independent Network-on-Chip (NOC) interfaces (NOC0 and NOC1). There is
no shared memory between cores. When an operation like RMSNorm needs its input
replicated across every core, or a matmul scatter needs to distribute weight
rows, data must move through the NOC. For multi-device configurations (T3K,
Galaxy), data must also move between devices over the Tenstorrent fabric
interconnect.

TT-Blaze abstracts these data movements behind a small set of micro-ops that
share a common architectural pattern: the **sender/receiver pattern**.
Understanding this pattern is essential before diving into any specific
inter-core op.

Every inter-core op in TT-Blaze follows four principles:

1. **Role assignment** -- per-core flags (`is_sender`, `is_receiver`) route
   each core to its branch of the kernel at compile time via `if constexpr`.
2. **NOC coordinate translation** -- logical `CoreCoord` values are mapped to
   physical NOC coordinates via `worker_core_from_logical_core`.
3. **Semaphore-based synchronization** -- senders signal completion; receivers
   wait on L1 semaphores before consuming data.
4. **Processor specialization** -- sender logic typically runs on one
   data-movement RISC (BRISC or NCRISC) while receiver logic runs on the
   other, though some ops (Copy) deviate deliberately.

### Why inter-core ops exist

A single Tensix core has limited L1 SRAM (typically 1.5 MB). Many workloads
-- RMSNorm broadcast, attention head scatter/gather, all-reduce for tensor
parallelism -- need to distribute or collect data across multiple cores.
Inter-core micro-ops let the compiler express these data movements as
first-class graph nodes that compose with compute ops in a single fused
kernel, eliminating host round-trips and separate tt-metal programs.

---

## Section Map

### [Section 1 -- Sender/Receiver Pattern and NOC Fundamentals](01_sender_receiver_pattern.md)

The foundational data-movement pattern that underpins every inter-core op.
Covers the dual-NOC architecture, semaphore protocols (`SemProtocol` enum),
NOC coordinate translation in Python emit() code, role flags via `f.flag()`,
data-size computation from `CBHandle` properties, RISC processor assignments,
and the `FusedProgram` grid context.

### [Section 2 -- Mcast, Gather, Scatter, and Copy](02_mcast_gather_scatter.md)

Detailed walk-throughs of the four primitive inter-core micro-ops:

| Op | Pattern | Sender RISC | Receiver RISC | Key feature |
|----|---------|-------------|---------------|-------------|
| **Mcast** | one-to-all | BRISC | NCRISC | Persistent NOC state, linked/posted modes, setup_src/run_as phases |
| **Gather** | many-to-one | NCRISC | BRISC | Dual-semaphore (NOC0+NOC1), per-core sender index |
| **Scatter** | one-to-many | BRISC | NCRISC | Per-core runtime arg arrays for dest coordinates |
| **Copy** | point-to-point | configurable | configurable | Local or remote, CB or raw address |

Also covers ScatterRaw (row-level scatter with head replication) and how
these primitives compose with compute ops (e.g., EmbedMcast = Embedding +
Mcast).

### [Section 3 -- CCL and Fabric Operations](03_ccl_and_fabric.md)

Multi-device collective communication over the Tenstorrent fabric interconnect.
Covers `CclBroadcast` (dual-axis mesh broadcast with 3 semaphores),
`AllReduce` (2-core layout with pairwise exchange + sum reduction),
the `setup_fabric()` utility, `FabricNodeId` resolution, fabric kernel
defines, and barrier ops (`BarrierSender`/`BarrierReceiver`).

---

## Prerequisites

Readers should be familiar with:

- **Circular buffers (CBs)** -- `CBHandle`, `f.cb_from_tensor()`,
  `f.cb_scratch()`, `cb_wait_front` / `cb_pop_front` / `cb_reserve_back` /
  `cb_push_back` (Chapter 3).
- **Compile-time argument system** -- `f.unified_ct_args()`,
  `f.ncrisc_ct_args()`, `f.brisc_ct_args()`, `f.trisc_ct_args()`,
  `f.per_core_unified_ct_args()`, and the `f.flag()` helper (Chapter 4).
- **The emit() / compose() contract** -- `emit()` is a `@staticmethod`,
  `compose()` is a `@classmethod` with signature
  `def compose(cls, f, tensors, output, user_args)` (Chapter 2).
- **Semaphores** -- `f.semaphore()` allocating global L1 semaphores or
  per-program slot-based semaphores (Chapter 4).
- **The BRISC / NCRISC / TRISC RISC processor model** (Chapter 1).

## Conventions Used in This Chapter

- **BRISC** is also called "DM1" or "Writer" in kernel code; **NCRISC** is
  "DM0" or "Reader". Both are data-movement processors; TRISC handles
  compute.
- `f` always refers to a `FusedProgram` instance.
- `prefix` is the compile-time-arg namespace string that scopes all CT args
  for one op instance (e.g., `"act_mcast"`, `"gather"`, `"all_reduce"`).
- All code samples in this chapter are taken from the TT-Blaze source tree.

## Key Source Files

| File | Purpose |
|------|---------|
| `blaze/ops/mcast/op.py` | Mcast micro-op Python emitter |
| `blaze/ops/mcast/kernels/op.hpp` | Mcast C++ kernel with persistent sender state |
| `blaze/ops/gather/op.py` | Gather micro-op Python emitter |
| `blaze/ops/gather/kernels/op.hpp` | Gather C++ kernel (sender NOC write + receiver semaphore wait) |
| `blaze/ops/scatter/op.py` | Scatter micro-op Python emitter |
| `blaze/ops/scatter/kernels/op.hpp` | Scatter C++ kernel |
| `blaze/ops/scatter_raw/op.py` | ScatterRaw micro-op (row-level scatter with head replication) |
| `blaze/ops/copy/op.py` | Copy micro-op Python emitter |
| `blaze/ops/copy/kernels/op.hpp` | Copy C++ kernel (local + remote, CB + raw address) |
| `blaze/ops/ccl_broadcast/op.py` | CCL Broadcast micro-op Python emitter |
| `blaze/ops/ccl_broadcast/kernels/op.hpp` | CCL Broadcast C++ kernel (fabric multicast) |
| `blaze/ops/all_reduce/op.py` | AllReduce micro-op Python emitter |
| `blaze/ops/all_reduce/kernels/op.hpp` | AllReduce C++ kernel (fabric unicast + local sum) |
| `blaze/ops/barrier_sender/op.py` | BarrierSender micro-op Python emitter |
| `blaze/ops/barrier_receiver/op.py` | BarrierReceiver micro-op Python emitter |
| `blaze/ccl.py` | `setup_fabric()` -- shared fabric connection setup utility |
| `blaze/fused_program.py` | FusedProgram: `semaphore()`, `flag()`, NOC grid fields |
| `blaze/blaze_op.py` | `SemProtocol` enum, `Risc` flag enum, `MicroOp` base class |
