# Chapter 1 -- Device Topologies and Mesh Initialization

This chapter establishes the physical and logical device landscape for Tenstorrent hardware. It covers the full spectrum of supported configurations -- from single-chip N150 and P150 boards through multi-chip T3K and Galaxy systems -- explains how chips are interconnected via ethernet links, and details how the `MESH_DEVICE` environment variable drives the entire device configuration stack for both TT-Blaze and TT-Symbiote.

## Contents

1. [**`01_physical_topologies.md`**](./01_physical_topologies.md) -- Physical chip arrangements, interconnect topologies, ethernet link counts, cluster axis semantics, and the mapping from `DeviceArch` enum values to mesh shapes for every supported configuration.

2. [**`02_mesh_device_initialization.md`**](./02_mesh_device_initialization.md) -- How `MESH_DEVICE` selects the mesh shape, how conftest fixtures create `ttnn.MeshDevice` instances, test parametrization patterns, the `run_on_devices` decorator, key TTNN mesh APIs, and how TT-Blaze bootstraps TT-Metal's test infrastructure.

3. [**`03_fabric_and_routing.md`**](./03_fabric_and_routing.md) -- The ethernet fabric that enables inter-chip communication, fabric configuration modes (`FABRIC_1D_RING` vs `FABRIC_2D`), topology selection for CCL operations, bandwidth implications of link counts, and the C++ control-plane APIs that underpin pipeline auto-discovery.

## Prerequisites

- Familiarity with Python and pytest fixtures/parametrization.
- Access to a Tenstorrent system (N150 or larger) with tt-metal and tt-blaze installed.
- Basic understanding of distributed computing concepts (mesh, topology, collective communication).

---

**Next:** [`01_physical_topologies.md`](./01_physical_topologies.md)
