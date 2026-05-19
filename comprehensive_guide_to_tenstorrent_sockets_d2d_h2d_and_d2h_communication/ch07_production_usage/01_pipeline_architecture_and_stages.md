# 7.1 -- Pipeline Architecture and Stage Configuration

DeepSeek V3 distributes inference across multiple Galaxy systems. Each Galaxy contains two meshes of 4x4 Blackhole chips (32 chips per Galaxy). Model layers are partitioned into pipeline stages, each assigned to one mesh. The `PipelineBlock` abstraction in `pipeline_block/op.py` encodes five distinct stage configurations that arise from the position of a mesh in the pipeline, whether it touches the host, and whether the autoregressive decode loop requires a loopback path.

This section documents the Galaxy topology, the five stage types with decision tables for selecting the correct configuration, and the data structures that wire them together.

---

## 7.1.1 Galaxy Hardware Topology

### What Breaks: Mismatched Mesh Shape

Opening a mesh device with the wrong shape is the most common first error. If you request `MeshShape(8, 4)` expecting a flat 32-chip layout, device open fails because a Galaxy exposes two separate 4x4 meshes, not one contiguous 8x4 mesh.

```
# WRONG -- Galaxy is not a single 8x4 mesh
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
# RuntimeError: Cannot map requested mesh shape to available devices
```

### Correct Topology

A Galaxy system consists of two interconnected 4x4 meshes of Blackhole chips (32 chips total). Each mesh is opened independently, one per host process. The meshes are connected by Ethernet links through TT-Fabric.

Each host opens its local 4x4 mesh:

```python
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 4))
```

Key hardware parameters: AI clock 1.35 GHz, L1 per core 1464 KB, 4 high-BW PCIe links (Gen4 x8, ASIC 6) and 28 low-BW links (Gen4 x1) per mesh, NOC word width 64 bytes, SRAM write alignment 16 bytes.

### What Breaks: Fabric Config Omission

If `ttnn.set_fabric_config()` is not called before `open_mesh_device()`, inter-chip communication silently fails. D2D socket sends hang because the fabric routing tables are never populated.

---

## 7.1.2 The Five Stage Types

The pipeline runtime defines exactly five stage configurations. Every mesh in the pipeline maps to one of these five types based on three binary properties: (1) does it receive from the host, (2) does it send to the host, and (3) does it participate in autoregressive loopback.

### Stage Type Decision Table

| Property | First+D2D | Middle | Last+Loopback | Last | Combined |
|----------|-----------|--------|---------------|------|-------------------|
| Receives tokens from host (H2D) | Yes | No | No | No | Yes |
| Sends results to host (D2H) | Optional | No | Yes | Yes | Yes |
| Receives from upstream D2D | No | Yes | Yes | Yes | No |
| Sends to downstream D2D | Yes | Yes | No | No | No |
| Loopback path (decode cycle) | Optional | No | Yes (D2D) | No | No |
| Position in pipeline | Head | Interior | Tail (autoregressive) | Tail (single-pass) | Solo |

### Stage Type Selection Flowchart

```
  Is this mesh the only mesh in the pipeline?
  |
  +-- YES --> COMBINED (H2D + D2H, no D2D sockets)
  |
  +-- NO --> Is this the first mesh?
             |
             +-- YES --> FIRST+D2D (H2D + downstream D2D, optional loopback/D2H)
             |
             +-- NO --> Is this the last mesh?
                        |
                        +-- YES --> Does the model require autoregressive decode?
                        |           |
                        |           +-- YES --> LAST+LOOPBACK (upstream D2D + D2H + loopback D2D)
                        |           |
                        |           +-- NO --> LAST (upstream D2D + D2H)
                        |
                        +-- NO --> MIDDLE (upstream D2D + downstream D2D)
```

---

## 7.1.3 Socket Composition Per Stage

Each stage type uses a specific combination of socket types. The following matrix shows exactly which socket types are instantiated for each stage configuration.

### Socket Composition Matrix

| Socket Type | First+D2D | Middle | Last+Loopback | Last | Combined |
|-------------|-----------|--------|---------------|------|----------|
| H2D (HOST_PUSH) | 1 | 0 | 0 | 0 | 1 |
| D2H | 0 or 1 | 0 | 1 | 1 | 1 |
| Upstream D2D (MeshSocket recv) | 0 | 1 | 1 | 1 | 0 |
| Downstream D2D (MeshSocket send) | 1 | 1 | 0 | 0 | 0 |
| Loopback D2D (send) | 0 | 0 | 1 | 0 | 0 |
| Loopback D2D (recv) | 0 or 1 | 0 | 0 | 0 | 0 |

### What Breaks: Entry/Exit Coordinate Mismatch

If `entry_node_coord` for stage N does not align with the physical chip that the upstream D2D socket targets, data arrives at a chip with no kernel waiting for it. The symptom is a hang during the first decode iteration with no error message -- the receiver socket blocks in `socket_wait_for_pages()` forever.

---

## 7.1.4 Stage Configuration Data Structures

Three data structures encode the configuration of each stage.

### StageMetadata

```python
# From op.py -- describes one stage's identity in the pipeline
@dataclass
class StageMetadata:
    rank: int        # Host rank owning this stage (0, 1, 2, ...)
    mesh_id: int     # Which mesh (0 or 1) runs this stage
```

`rank` maps to a host in the distributed deployment. `mesh_id` identifies which of the two mesh devices within that host's Galaxy. Together, `(rank, mesh_id)` uniquely identifies a physical mesh device in the cluster.

### PipelineConfigEntry

```python
# From op.py -- maps a stage to entry/exit chip coordinates
@dataclass
class PipelineConfigEntry:
    entry_node_coord: MeshCoordinate  # Where D2D data enters this stage
    exit_node_coord: MeshCoordinate   # Where D2D data exits this stage
```

Entry and exit coordinates identify which chip in the 4x4 mesh grid handles D2D fabric traffic. These coordinates determine the `MeshCoreCoord` used to construct `SocketConnection` objects.

### HostIoPlacement

```python
# From op.py -- assigns each socket type to a separate core
@dataclass
class HostIoPlacement:
    h2d_core: MeshCoreCoord      # Core receiving host-to-device data
    d2h_core: MeshCoreCoord      # Core sending device-to-host data
    fwd_d2d_core: MeshCoreCoord  # Core handling forward D2D traffic
    lb_d2d_core: MeshCoreCoord   # Core handling loopback D2D traffic
```

All four cores must be distinct. Reusing a core for two sockets causes L1 resource conflicts because each socket allocates its own FIFO and config buffer in L1.

### What Breaks: Stage-to-Rank Assignment Errors

If `StageMetadata.rank` does not match the actual rank of the process running that stage, the distributed context routes socket connections to the wrong host. The failure appears as a timeout in `MeshSocket` construction when the peer process never calls the matching socket creation.

---

## 7.1.5 Five Stage Layout (2-Mesh Galaxy)

A concrete DeepSeek V3 deployment with five stages across two meshes:

```
Stage 0 (Mesh 0)  -->  Stage 1 (Mesh 0)  -->  Stage 2 (Mesh 1)
                                                      |
                                                      v
Stage 4 (Mesh 0)  <--  Stage 3 (Mesh 1)  <----------+
      |
      +--- Loopback to Stage 0 for next token
```

| Stage | Mesh | Rank | Role | Stage Type |
|-------|------|------|------|------------|
| 0 | 0 | 0 | Embedding + first transformer layers | First+D2D |
| 1 | 0 | 0 | Middle transformer layers | Middle |
| 2 | 1 | 1 | MoE expert routing + execution | Middle |
| 3 | 1 | 1 | Post-MoE layers | Middle |
| 4 | 0 | 0 | Final layers + LM head | Last+Loopback |

The five-stage configuration is generated by `generate_blitz_decode_pipeline()`, which populates both `StageMetadata` and `PipelineConfigEntry` for each stage.

---

## 7.1.6 The Autoregressive Loop

During autoregressive decoding, the output of the last pipeline stage must feed back to the first stage as input for the next token. This creates a circular data dependency across the linear pipeline.

```
  Prefill (single pass, no loopback):
  Host --H2D--> [First] --D2D--> [Middle] --D2D--> [Last] --D2H--> Host

  Decode (autoregressive, with loopback):
  Host --H2D--> [First] --D2D--> [Middle] --D2D--> [Last+LB] --D2H--> Host
                   ^                                    |
                   +------------- D2D loopback ---------+
```

Each autoregressive decode iteration:

1. Host pushes the current token via H2DSocket (HOST_PUSH mode) to Stage 0
2. Stage 0 computes embedding + early layers, sends activations via D2D to Stage 1
3. Stages 1-3 each compute their layers and forward via D2D
4. Stage 4 computes the LM head, sends logits via D2HSocket to Host
5. Stage 4 simultaneously sends the token embedding via loopback D2D to Stage 0
6. Host samples the next token, loops back to Step 1

### What Breaks: Cross-Mesh D2D Without Distributed Context

When a stage sends to another stage across mesh boundaries, the `MeshSocket` requires a valid `distributed_context`. The distributed context is auto-initialized when the mesh device is opened, but if a process opens and closes its mesh device before the peer process has opened its own, the context handshake fails.

**Symptom**: `MeshSocket` constructor throws a connection timeout.

**Fix**: Both hosts must open their mesh devices before either creates cross-mesh sockets. Use `ttnn.distributed_context_barrier()` after device open.

### What Breaks: FIFO Size Under-Provisioning

If `upstream_d2d_fifo_size` is smaller than the activation tensor for a single layer, the pipeline deadlocks. The sending stage fills the FIFO and blocks in `socket_reserve_pages()`, while the receiving stage has not yet consumed enough pages to free space. The minimum FIFO size must accommodate at least one full activation page to prevent circular dependencies.

---

## 7.1.7 Pipeline Auto-Configuration

The `generate_blitz_decode_pipeline()` function automatically determines stage assignments based on the number of meshes, the distributed context size, and the model configuration. It distributes layers evenly across meshes and assigns the correct stage type based on position: First+D2D for the head, Middle for interior, Last or Last+Loopback for the tail, and Combined for single-mesh deployments.

| Pipeline position | Meshes = 1 | Meshes = 2 | Meshes >= 3 |
|-------------------|------------|------------|-------------|
| Position 0 (head) | Combined | First+D2D | First+D2D |
| Position 1 | N/A | Last or Last+LB | Middle |
| Position N-1 (tail, N >= 3) | N/A | N/A | Last or Last+LB |
| Interior (1 < pos < N-1) | N/A | N/A | Middle |

---

## Key Takeaways

1. A Galaxy is two 4x4 meshes of Blackhole chips, not one flat mesh. Each host opens its local 4x4 mesh independently with `ttnn.open_mesh_device(mesh_shape=MeshShape(4, 4))`.

2. `ttnn.set_fabric_config(FabricConfig.FABRIC_2D)` must be called before opening the mesh device; omitting it causes silent D2D communication failure.

3. Every mesh maps to exactly one of five stage types (First+D2D, Middle, Last+Loopback, Last, Combined), determined by three binary properties: host ingress, host egress, and loopback participation.

4. `StageMetadata` (rank + mesh_id) locates a stage in the distributed cluster; `PipelineConfigEntry` provides D2D entry/exit coordinates; `HostIoPlacement` assigns four distinct cores to socket roles.

5. The socket composition matrix is the authoritative reference for which socket types each stage instantiates -- consult it before wiring any new topology.

6. `generate_blitz_decode_pipeline()` automates stage assignment based on mesh count, layer count, and distributed context metadata.

7. The autoregressive loop crosses mesh and host boundaries multiple times per decode iteration. Every crossing requires properly configured socket pairs, and FIFO sizes must accommodate at least one full activation tensor.

---

**Previous:** [Chapter 7 Index](./index.md) | **Next:** [7.2 Socket Interface and Host Interface](./02_socket_interface_and_host_interface.md) | **Up:** [Chapter 7 Index](./index.md)
