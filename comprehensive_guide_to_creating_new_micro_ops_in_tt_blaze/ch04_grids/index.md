# Chapter 4: Device Grids and Multi-Device Configurations

Every micro-op in TT-Blaze ultimately runs on a physical grid of Tensix cores. Blackhole devices come in different SKUs -- an unharvested P300 exposes a 13x10 grid while a harvested P150 exposes 11x10 -- and each device has certain cores permanently assigned to DRAM worker duty. On top of that, production deployments run across multi-device meshes (T3K with 8 devices, Galaxy with 32 devices) where each device in the mesh may need different compile-time arguments, different fabric routing, or different core assignments.

This chapter covers the two layers a micro-op author must understand:

1. **Single-device grid configuration** -- the `GridConfig` dataclass, the `DeviceContext` wrapper, and the core categorization APIs that let ops adapt to any Blackhole variant without hardcoded coordinates.

2. **Multi-device mesh compilation** -- how `BlazeCompiler`, `MeshFusedProgram`, and `MeshCompiledProgram` extend single-device compilation to mesh topologies, and how CCL ops use the fabric layer to move data between devices.

## Chapter contents

- [01 -- Grid Configuration and Core Categories](01_grid_config.md)
- [02 -- Multi-Device Mesh Compilation and CCL Ops](02_multi_device_mesh.md)

## Key principles

1. **Never hardcode grid dimensions.** Blackhole devices ship with 13x10, 12x10, or 11x10 grids depending on harvesting. `GridConfig.from_device()` queries the actual hardware; `GridConfig.default()` returns the unharvested 13x10 baseline for offline development.

2. **DRAM workers and phantom cores are not interchangeable with compute cores.** `GridConfig` tracks which positions are DRAM workers, which are phantom (gate MM) cores, and which is the sender core. Each accessor method (`get_compute_cores()`, `get_matmul_cores()`, etc.) applies the correct exclusion set.

3. **Multi-device programs are per-device compilations assembled into a mesh.** `BlazeCompiler.compile()` iterates over every device in the mesh, builds a per-device `ProgramDescriptor` with device-specific CT args (including `mesh_coord` and `mesh_shape`), and assembles them into a `MeshProgramDescriptor`. The single `ttnn.generic_op()` call dispatches the right program to each physical device.

4. **CCL ops own their routing logic.** Collective communication micro-ops (broadcast, all-reduce) compute per-device fabric routing from `mesh_coord` and `mesh_shape`, allocate semaphores via `FusedProgram.semaphore()`, and set up fabric connections through the shared `setup_fabric()` helper.

## Why grid-awareness matters

A micro-op that hardcodes grid dimensions (e.g. assuming 13 columns) will silently produce wrong results or crash on a harvested device with fewer columns. Similarly, an op that ignores DRAM worker positions may try to use a core that the hardware has reserved for DRAM traffic, corrupting memory. TT-Blaze solves this by requiring ops to query their grid through `GridConfig` rather than embedding constants, and by providing `DeviceContext` as the single point of device interaction during compilation.

For multi-device workloads the stakes are higher: each device in a mesh gets its own `DeviceContext` and its own set of compile-time arguments, but they share mesh-global resources like semaphores and fabric connections. The compiler handles this automatically for graph-compiled ops. For composition-driven ops (the FusedProgram path), authors interact with mesh state through `f.mesh_coord`, `f.mesh_shape`, and `f.mesh_device` -- attributes that the compiler injects before calling the op's `compose()` function.
