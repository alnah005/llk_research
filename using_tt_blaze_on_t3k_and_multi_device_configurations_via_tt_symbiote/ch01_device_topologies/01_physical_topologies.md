# 01 -- Physical Topologies

This file enumerates every supported Tenstorrent device configuration, from single-chip cards through 32-chip Galaxy clusters. It describes the physical ethernet interconnects, the number of communication links available per topology, cluster axis semantics, and how these physical topologies map to the `DeviceArch` enum in the TT-Symbiote/TT-Blaze software stack. Understanding these topologies is a prerequisite for every subsequent chapter, because the mesh shape dictates how tensors are distributed, how collective communication operations are routed, and what pipeline partitioning strategies are feasible.

## Supported Configurations

The Tenstorrent ecosystem spans two chip architectures -- **Wormhole B0** (WH) and **Blackhole** (BH) -- arranged in progressively larger configurations. The table below lists every configuration that the software stack recognizes.

| Name    | Architecture | Chips | Mesh Shape | `DeviceArch` Enum   | `MESH_DEVICE` Key | Description                                |
|---------|-------------|-------|------------|---------------------|-------------------|--------------------------------------------|
| N150    | Wormhole B0 | 1     | (1, 1)     | `DeviceArch.N150`   | `"N150"`          | Single WH chip, PCIe card                  |
| N300    | Wormhole B0 | 2     | (1, 2)     | `DeviceArch.N300`   | `"N300"`          | Dual WH chip, single card                  |
| N150x4  | Wormhole B0 | 4     | (1, 4)     | n/a                 | `"N150x4"`        | Four WH chips across two N300 cards        |
| T3K     | Wormhole B0 | 8     | (1, 8)     | `DeviceArch.T3K`    | `"T3K"`           | TT-Trillium, 4 N300 boards in a cluster   |
| TG      | Wormhole B0 | 32    | (8, 4)     | `DeviceArch.TG`     | `"TG"`            | TT-Galaxy, 4U shelf of 32 WH chips        |
| P150    | Blackhole   | 1     | (1, 1)     | `DeviceArch.P150`   | `"P150"`          | Single BH chip, PCIe card (8-DRAM-channel) |
| P300    | Blackhole   | 2     | (1, 2)     | `DeviceArch.P300`   | `"P300"`          | Dual BH chip, single card                  |
| P150x4  | Blackhole   | 4     | (1, 4)     | `DeviceArch.P150x4` | `"P150x4"`        | Four BH chips in a rack unit               |
| P150x8  | Blackhole   | 8     | (1, 8)     | `DeviceArch.P150x8` | `"P150x8"`        | Eight BH chips in a rack unit              |
| BHGLX   | Blackhole   | 32    | (8, 4)     | `DeviceArch.BHGLX`  | `"BHGLX"`         | Blackhole Galaxy, 32 BH chips             |

> **Warning:** P100 is a distinct Blackhole variant with a 7x1 DRAM grid (vs. 8x1 for P150). It appears in `determine_device_name()` in [model_config.py:determine_device_name] but is not present in the `DeviceArch` enum or `MeshShapeToDeviceArch` mapping, indicating it is not a first-class target for TT-Symbiote or TT-Blaze workloads. N150x4 is similarly absent from `DeviceArch` -- it is recognized by the runtime's `determine_device_name()` (and used for CCL link counting) but not by the `run_on_devices` gating mechanism.

### Architecture Detection at Runtime

The function `determine_device_name()` in [model_config.py:determine_device_name] resolves the physical configuration from two runtime signals: the chip architecture (Wormhole B0 vs Blackhole, detected via `is_blackhole()` / `is_wormhole_b0()`) and the number of devices in the mesh (`mesh_device.get_num_devices()`). For Blackhole single-chip systems, DRAM grid size further distinguishes P100 (7x1) from P150 (8x1).

```python
# Simplified from model_config.py:determine_device_name
if is_blackhole():
    dict_device_names = {
        1: "P100" if dram_grid_size.x == 7 else "P150",
        2: "P300", 4: "P150x4", 8: "P150x8", 32: "BHGLX",
    }
elif is_wormhole_b0():
    dict_device_names = {
        1: "N150", 2: "N300", 4: "N150x4", 8: "T3K", 32: "TG",
    }
```

This function is called by `get_num_links()` in [ccl.py:get_num_links] and is the sole authority for mapping a live `MeshDevice` to a configuration name.

## DeviceArch Enum and MeshShapeToDeviceArch Mapping

The `DeviceArch` enum in [module.py:DeviceArch] provides the canonical enumeration of all supported configurations. Each value is a string identifier used as the internal representation:

```python
# [core/module.py:DeviceArch]
class DeviceArch(Enum):
    N150   = "n150"
    N300   = "n300"
    T3K    = "t3k_wh"
    TG     = "gx_wh"
    P150   = "p150"
    P300   = "p300"
    P150x4 = "p150x4"
    P150x8 = "p150x8"
    BHGLX  = "bhglx"
```

The companion dictionary `MeshShapeToDeviceArch` maps the `MESH_DEVICE` environment variable string to the corresponding enum value:

```python
# [core/module.py:MeshShapeToDeviceArch]
MeshShapeToDeviceArch = {
    "N150":   DeviceArch.N150,
    "N300":   DeviceArch.N300,
    "T3K":    DeviceArch.T3K,
    "TG":     DeviceArch.TG,
    "P150":   DeviceArch.P150,
    "P300":   DeviceArch.P300,
    "P150x4": DeviceArch.P150x4,
    "P150x8": DeviceArch.P150x8,
    "BHGLX":  DeviceArch.BHGLX,
}
```

This mapping is the bridge between the environment-driven test configuration (`MESH_DEVICE=T3K`) and the runtime device-gating logic in `run_on_devices` (covered in [Chapter 1, File 2](./02_mesh_device_initialization.md)).

## Physical Interconnect Topologies

### N150 / P150 -- Single Chip (No Interconnect)

```
  +--------+
  | Chip 0 |
  | (0,0)  |
  +--------+
      |
    PCIe
```

A single chip has no ethernet links. All tensor operations are local. The mesh shape is (1, 1).

### N300 / P300 -- Dual Chip

```
  +--------+   ethernet    +--------+
  | Chip 0 |<============>| Chip 1 |
  | (0,0)  |               | (0,1)  |
  +--------+               +--------+
      |
    PCIe
```

Two chips connected by ethernet. The mesh shape is (1, 2), forming a linear topology along axis 1 (horizontal). Only Chip 0 is typically PCIe-attached; Chip 1 is reached exclusively through the ethernet link.

- N300: 1 ethernet link
- P300: 2 ethernet links

### N150x4 / P150x4 -- Four Chips, Linear

```
  +--------+   eth   +--------+   eth   +--------+   eth   +--------+
  | Chip 0 |<------->| Chip 1 |<------->| Chip 2 |<------->| Chip 3 |
  | (0,0)  |         | (0,1)  |         | (0,2)  |         | (0,3)  |
  +--------+         +--------+         +--------+         +--------+

  N150x4: 1 ethernet link per hop
  P150x4: 2 ethernet links per hop
```

Four chips in a (1, 4) mesh. Adjacent chips are connected by ethernet links.

### T3K -- Eight Wormhole Chips

T3K consists of four N300 boards daisy-chained into a line of 8 chips. The mesh shape is (1, 8).

```
  +--------+  +--------+  +--------+  +--------+  +--------+  +--------+  +--------+  +--------+
  | Chip 0 |--| Chip 1 |--| Chip 2 |--| Chip 3 |--| Chip 4 |--| Chip 5 |--| Chip 6 |--| Chip 7 |
  | (0,0)  |  | (0,1)  |  | (0,2)  |  | (0,3)  |  | (0,4)  |  | (0,5)  |  | (0,6)  |  | (0,7)  |
  +--------+  +--------+  +--------+  +--------+  +--------+  +--------+  +--------+  +--------+
      |                        |                        |                        |
    PCIe                     PCIe                     PCIe                     PCIe
   Board 0                  Board 1                  Board 2                  Board 3

  1 ethernet link per hop.
```

Each adjacent pair is connected by 1 ethernet link. All 8 chips lie along a single row (axis 1). When fabric is configured as `FABRIC_1D_RING`, the two endpoints can wrap around, forming a ring that halves the maximum hop distance to 4.

### P150x8 -- Eight Blackhole Chips

```
  +--------+  +--------+  +--------+  +--------+  +--------+  +--------+  +--------+  +--------+
  | Chip 0 |==| Chip 1 |==| Chip 2 |==| Chip 3 |==| Chip 4 |==| Chip 5 |==| Chip 6 |==| Chip 7 |
  | (0,0)  |  | (0,1)  |  | (0,2)  |  | (0,3)  |  | (0,4)  |  | (0,5)  |  | (0,6)  |  | (0,7)  |
  +--------+  +--------+  +--------+  +--------+  +--------+  +--------+  +--------+  +--------+

  == = 2 ethernet links per hop (double the bandwidth of T3K)
```

The Blackhole counterpart to T3K, with mesh shape (1, 8) and 2 ethernet links between each pair of adjacent chips.

### TG / BHGLX -- 32-Chip Galaxy (8x4 Grid)

The Galaxy configurations arrange 32 chips in an (8, 4) 2D mesh. This is the largest supported topology and the only one with a genuinely two-dimensional layout.

```
       Col 0     Col 1     Col 2     Col 3
      +-------+ +-------+ +-------+ +-------+
 R0   | (0,0) |-| (0,1) |-| (0,2) |-| (0,3) |
      +---+---+ +---+---+ +---+---+ +---+---+
          |          |          |          |
      +---+---+ +---+---+ +---+---+ +---+---+
 R1   | (1,0) |-| (1,1) |-| (1,2) |-| (1,3) |
      +---+---+ +---+---+ +---+---+ +---+---+
          |          |          |          |
      +---+---+ +---+---+ +---+---+ +---+---+
 R2   | (2,0) |-| (2,1) |-| (2,2) |-| (2,3) |
      +---+---+ +---+---+ +---+---+ +---+---+
          |          |          |          |
      +---+---+ +---+---+ +---+---+ +---+---+
 R3   | (3,0) |-| (3,1) |-| (3,2) |-| (3,3) |
      +---+---+ +---+---+ +---+---+ +---+---+
          |          |          |          |
      +---+---+ +---+---+ +---+---+ +---+---+
 R4   | (4,0) |-| (4,1) |-| (4,2) |-| (4,3) |
      +---+---+ +---+---+ +---+---+ +---+---+
          |          |          |          |
      +---+---+ +---+---+ +---+---+ +---+---+
 R5   | (5,0) |-| (5,1) |-| (5,2) |-| (5,3) |
      +---+---+ +---+---+ +---+---+ +---+---+
          |          |          |          |
      +---+---+ +---+---+ +---+---+ +---+---+
 R6   | (6,0) |-| (6,1) |-| (6,2) |-| (6,3) |
      +---+---+ +---+---+ +---+---+ +---+---+
          |          |          |          |
      +---+---+ +---+---+ +---+---+ +---+---+
 R7   | (7,0) |-| (7,1) |-| (7,2) |-| (7,3) |
      +-------+ +-------+ +-------+ +-------+

 Legend:  --- = horizontal ethernet link (axis 1): 3 links per chip-pair
          |  = vertical ethernet link (axis 0):   4 links per chip-pair
```

The link budget is asymmetric:

- **Axis 0 (vertical):** 4 ethernet links per chip-pair -- higher bandwidth
- **Axis 1 (horizontal):** 3 ethernet links per chip-pair

This asymmetry is critical for performance: vertical all-gather and reduce-scatter operations have 33% more bandwidth than horizontal ones. Both TG (Wormhole) and BHGLX (Blackhole) share this same asymmetric layout.

> **Warning:** On TG, the first 4 chip IDs (0-3) are gateway devices. The `first_available_tg_device()` function in `conftest.py` returns chip ID 4 as the first user-accessible device. When creating a single device on a TG system, the fixture code explicitly targets device ID 4 rather than 0.

## Ethernet Link Counts

The number of ethernet links available for CCL operations is critical for performance. The `get_num_links()` function in [ccl.py:get_num_links] returns the link count per topology, stored as `(axis_0_links, axis_1_links)` tuples:

```python
# [ccl.py:get_num_links] -- link_dict
link_dict = {
    "P100":   (0, 0),   # No inter-chip links
    "P150":   (0, 0),   # No inter-chip links
    "N150":   (0, 0),   # No inter-chip links
    "N300":   (1, 1),   # 1 link per axis
    "T3K":    (1, 1),   # 1 link per axis
    "P150x4": (2, 2),   # 2 links per axis
    "P150x8": (2, 2),   # 2 links per axis
    "P300":   (2, 2),   # 2 links per axis
    "BHGLX":  (4, 3),   # 4 vertical, 3 horizontal
    "TG":     (4, 3),   # 4 vertical, 3 horizontal
    "N150x4": (1, 1),   # 1 link per axis
}
```

### Link Count Query Semantics

The `get_num_links()` function accepts an optional `cluster_axis` parameter:

- **`cluster_axis=None`** (default): Returns `min(axis_0_links, axis_1_links)`. For TG/BHGLX, this returns 3.
- **`cluster_axis=0`**: Returns links along the **vertical** axis. For TG/BHGLX, returns 4.
- **`cluster_axis=1`**: Returns links along the **horizontal** axis. For TG/BHGLX, returns 3.

CCL operations use the link count to determine how many parallel data channels to open.

## Cluster Axis Semantics

The cluster axis convention is consistent across both frameworks and all APIs:

- **Axis 0 = vertical (rows):** Movement from row $i$ to row $i+1$. North-South direction. In an (8, 4) mesh, axis 0 spans 8 positions.
- **Axis 1 = horizontal (columns):** Movement from column $j$ to column $j+1$. East-West direction. In an (8, 4) mesh, axis 1 spans 4 positions.

This convention matches standard matrix indexing: `MeshShape(rows, cols)`, `MeshCoordinate(row, col)`, and the `cluster_axis` parameter in CCL operations all follow this convention.

For 1D topologies like T3K (1, 8), axis 0 has extent 1 (trivial -- no communication needed), and axis 1 has extent 8. The `tt_all_reduce()` function in [ccl.py:tt_all_reduce] skips CCL when the target axis has size 1.

```python
# Example: CCL operations with cluster_axis
tt_all_reduce(tensor, mesh_device, tt_ccl, cluster_axis=0)  # vertical (combine 8 rows)
tt_all_gather(tensor, mesh_device, tt_ccl, cluster_axis=1, dim=3)  # horizontal (combine 4 cols)
```

## Wormhole B0 vs Blackhole

Blackhole configurations mirror Wormhole at every scale (P150 ↔ N150, P300 ↔ N300, P150x4 ↔ N150x4, P150x8 ↔ T3K, BHGLX ↔ TG) but provide 2x the ethernet links per hop on sub-Galaxy systems — as visible in the `link_dict` above. The naming correspondence is shown in the Supported Configurations table. Both families share the same software API surface.

> **Warning:** Blackhole multi-chip systems (P300, P150x4, P150x8) have 2 links per direction — double the bandwidth of their Wormhole counterparts (N300, N150x4, T3K at 1 link).

---

**Next:** [`02_mesh_device_initialization.md`](./02_mesh_device_initialization.md)
