# 8.1 Pipeline Parallelism and Configuration

This section explains how the DeepSeek V3 B1 implementation maps 63 logical model stages onto the physical Galaxy pod topology using pipeline parallelism as the inter-host distribution strategy. Every configuration file, script, and naming convention is traced to source code. The central design rationale explored here is **why pipeline parallelism was chosen over tensor parallelism for inter-host scaleout**, and how the snake-order host traversal exploits the physical wiring of a Galaxy pod.

> **Cross-reference -> Chapter 1, Section 1.1**: DeepSeek V3 model architecture,
> 671B parameters, 61 decoder layers.

> **Cross-reference -> Chapter 7, Section 7.2**: Weight preparation and per-layer
> serialization -- the per-stage cache directories that this pipeline configuration
> ultimately targets.

Source: `models/demos/deepseek_v3_b1/scaleout_configs/` (YAML configs, textproto descriptors, `generate_blitz_decode_pipeline_configs.py`)

---

## 8.1.1 Galaxy Pod Topology

A single Galaxy pod consists of **4 hosts**, each connected to **4 Blackhole
slices** (each slice is a 4x2 mesh of 8 Blackhole chips). One pod therefore
contains **16 slices** (128 chips total). The slices within a pod are
interconnected via Tenstorrent fabric links, enabling device-to-device (D2D)
data transfer without host involvement.

```
 GALAXY POD TOPOLOGY: 4 Hosts x 4 Slices = 16 Pipeline Stages
 ===============================================================

  Host 0 (d05u08)         Host 1 (d05u02)         Host 2 (d06u02)         Host 3 (d06u08)
 +------------------+    +------------------+    +------------------+    +------------------+
 | Slice 0  [4x2]  |    | Slice 0  [4x2]  |    | Slice 0  [4x2]  |    | Slice 0  [4x2]  |
 |  Stage 0         |    |  Stage 4         |    |  Stage 8         |    |  Stage 12        |
 |  8 BH chips      |    |  8 BH chips      |    |  8 BH chips      |    |  8 BH chips      |
 +------------------+    +------------------+    +------------------+    +------------------+
 | Slice 1  [4x2]  |    | Slice 1  [4x2]  |    | Slice 1  [4x2]  |    | Slice 1  [4x2]  |
 |  Stage 1         |    |  Stage 5         |    |  Stage 9         |    |  Stage 13        |
 |  8 BH chips      |    |  8 BH chips      |    |  8 BH chips      |    |  8 BH chips      |
 +------------------+    +------------------+    +------------------+    +------------------+
 | Slice 2  [4x2]  |    | Slice 2  [4x2]  |    | Slice 2  [4x2]  |    | Slice 2  [4x2]  |
 |  Stage 2         |    |  Stage 6         |    |  Stage 10        |    |  Stage 14        |
 |  8 BH chips      |    |  8 BH chips      |    |  8 BH chips      |    |  8 BH chips      |
 +------------------+    +------------------+    +------------------+    +------------------+
 | Slice 3  [4x2]  |    | Slice 3  [4x2]  |    | Slice 3  [4x2]  |    | Slice 3  [4x2]  |
 |  Stage 3         |    |  Stage 7         |    |  Stage 11        |    |  Stage 15        |
 |  8 BH chips      |    |  8 BH chips      |    |  8 BH chips      |    |  8 BH chips      |
 +------------------+    +------------------+    +------------------+    +------------------+

       |                       |                       |                       |
       +--- Fabric Links ------+--- Fabric Links ------+--- Fabric Links ------+
               (8 channels per inter-slice connection)
```

Each slice is described in the mesh graph descriptor as a `4x2` mesh with
topology `[RING, LINE]` -- the row dimension (4 rows) forms a ring and the
column dimension (2 columns) is a line.

> **Source**: `blitz_decode_single_pod_mesh_graph_descriptor.textproto`, lines 3-8:
> `device_topology { dims: [ 4, 2 ] dim_types: [ RING, LINE ] }`

**Design rationale -- why 4x2 per slice:** The first dimension (rows=4) uses
RING topology, meaning row 0 connects to row 3 cyclically. The second dimension
(cols=2) uses LINE topology -- a direct pair. This 4x2 shape is not arbitrary:
it is the physical wiring of a single Blackhole slice. The RING dimension
exploits the chip-to-chip ethernet links that form a 4-chip ring within a
column, while the LINE dimension connects the two columns via a separate link
pair.

---

## 8.1.2 Pipeline Stage Assignment

DeepSeek V3 has **61 transformer layers**: 3 dense layers (layers 0-2) and 58
MoE layers (layers 3-60). With embedding at the front and LM head + sampling
at the back, the full pipeline has **63 logical stages** (indices 0-62):

> **Cross-reference → Chapter 1, Section 3.4**: The full 63-stage pipeline assignment table (stage 0 = embedding, stages 1-3 = dense, stages 4-61 = MoE, stage 62 = LM head), `decoder_layer_id_from_mesh_id()` function (`demo/cli.py`, lines 56-62), and boundary constants `SYSTEM_MESH_ID_EMBEDDING = 0` / `SYSTEM_MESH_ID_LM_HEAD = 62` (lines 65-66).

The first 3 decoder layers (mesh IDs 1-3, layer IDs 0-2) are dense, determined by `FIRST_K_DENSE_REPLACE = 3` (line 35).

**Design rationale -- one layer per stage:** Each pipeline stage gets exactly
one decoder layer (or one bookend). This 1:1 mapping simplifies weight loading
(each MPI rank loads exactly one layer's cache directory), simplifies the D2D
exchange protocol (fixed activation tensor size per stage), and avoids the
complexity of packing multiple layers into a single stage. The cost is a deeper
pipeline (63 stages), but this depth is acceptable because DeepSeek V3 decode
is memory-bandwidth-bound -- pipeline bubble overhead is small relative to
per-stage DRAM streaming time.

---

## 8.1.3 Pipeline Configuration YAML Files

Each deployment topology has a YAML configuration file that maps every pipeline
stage to a `(host, slice)` pair. The repository contains five configurations:

| Config File | Stages | Hosts | Slices | Purpose |
|---|---|---|---|---|
| `blitz_pipeline_config_single_pod.yaml` | 16 | 4 | 16 | Single-pod development |
| `blitz_pipeline_config_single_pod_ci.yaml` | 16 | 4 | 16 | CI (placeholder hosts) |
| `blitz_pipeline_config_single_pod_9_10.yaml` | 16 | 4 | 16 | Alternate pod (d09/d10) |
| `blitz_pipeline_config_2_pod.yaml` | 32 | 8 | 32 | 2-pod deployment |
| `blitz_pipeline_config_superpod.yaml` | 64 | 16 | 64 | Full superpod (4 pods) |

Each YAML file has three top-level keys:

```yaml
stage_to_slice_mapping:
  0:
    host: bh-glx-d05u08
    slice: 0
  1:
    host: bh-glx-d05u08
    slice: 1
  # ... one entry per stage

mesh_graph_desc_path: models/demos/.../blitz_decode_single_pod_mesh_graph_descriptor.textproto
rank_binding_file: blitz_decode_pipeline_rank_binding_single_pod.yaml
rank_file: blitz_decode_pipeline_rank_file_single_pod
```

### Single Pod (16 stages)

The single-pod configuration maps 16 stages across 4 hosts, using a
**snake/zigzag** traversal order.

> **Source**: `scaleout_configs/blitz_pipeline_config_single_pod.yaml`

| Stage | Host          | Slice | Role              |
|-------|---------------|-------|-------------------|
| 0     | bh-glx-d05u08 | 0     | Embedding         |
| 1     | bh-glx-d05u08 | 1     | Dense layer 0     |
| 2     | bh-glx-d05u08 | 2     | Dense layer 1     |
| 3     | bh-glx-d05u08 | 3     | Dense layer 2     |
| 4     | bh-glx-d05u02 | 0     | MoE layer 3       |
| ...   | ...           | ...   | ...               |
| 12    | bh-glx-d06u08 | 0     | MoE layer 11      |
| ...   | ...           | ...   | ...               |
| 15    | bh-glx-d06u08 | 3     | MoE layer 14      |

With only 16 stages, a single pod runs a partial model (stages 0-15 only, no
LM head at stage 62). It includes a loopback connection so that the output of
the last stage wraps back to stage 0 for testing.

### Two-Pod (32 stages)

The 2-pod configuration doubles capacity to 32 stages across 8 hosts (2 pods x
4 hosts). Stages 0-15 map to the first pod and stages 16-31 to the second pod.

> **Source**: `scaleout_configs/blitz_pipeline_config_2_pod.yaml`

The inter-pod link between stages 15 and 16 has reduced bandwidth (2 channels
versus 8 intra-pod):

> **Source**: `blitz_decode_mesh_graph_descriptor_2_pod.textproto`, lines 126-130:
> ```protobuf
> connections {
>   nodes { mesh { mesh_descriptor: "M0" mesh_id: 15 } }
>   nodes { mesh { mesh_descriptor: "M0" mesh_id: 16 } }
>   channels { count: 2 policy: RELAXED }
> }
> ```

The 2-pod descriptor does **not** include a loopback connection:

> **Source**: `blitz_decode_mesh_graph_descriptor_2_pod.textproto`, line 206:
> `# no loop-back connection, this is not currently supported for this configuration.`

### Superpod (64 stages)

The superpod configuration maps all 63 model stages (plus one loopback helper
at stage 63) across **4 pods = 16 hosts = 64 slices**:

> **Source**: `scaleout_configs/blitz_pipeline_config_superpod.yaml`

```
 SUPERPOD PIPELINE ROUTING (64 stages across 4 pods)
 ====================================================

 Pod 1 (d04, d03):  Stages  0-15   Embedding + Dense + MoE layers 3-14
    Stage  0 = d04u08:slice3  (Embedding)
    Stage  1 = d03u08:slice0  (Dense layer 0)
    ...
    Stage 15 = d04u08:slice2

 Pod 2 (d09, d10):  Stages 16-31   MoE layers 15-30
    Stage 16 = d09u08:slice1
    ...
    Stage 31 = d09u08:slice0

 Pod 3 (d08, d07):  Stages 32-47   MoE layers 31-46
    Stage 32 = d08u08:slice3
    ...
    Stage 47 = d08u08:slice2

 Pod 4 (d05, d06):  Stages 48-63   MoE layers 47-60 + LM head + loopback
    Stage 48 = d05u08:slice1
    ...
    Stage 62 = d06u08:slice3  (LM head)
    Stage 63 = d05u08:slice0  (Loopback helper)
```

The superpod is the only configuration that maps all 63 model stages (0-62)
plus a loopback helper (stage 63), enabling full end-to-end inference of the
671B parameter DeepSeek V3 model.

Notable features of the superpod config:

1. **Pod boundary stages use the "u08" host:** Every inter-pod transition exits
   from and enters to a u08 host. This is because the inter-pod ethernet links
   physically connect through the u08 hosts in the Galaxy topology.

2. **Non-sequential slice usage at boundaries:** Pod 1 starts at d04u08 slice 3
   (not slice 0). This positions the embedding stage on the slice closest to
   the inter-pod return link from the loopback stage.

3. **Stage 63 as explicit loopback:** Unlike single-pod where mesh 15 connects
   directly to mesh 0, the superpod dedicates a full stage (63) as a relay
   node. This stage does no model computation -- it simply forwards the LM head
   output back to the embedding stage across the pod boundary.

4. **Each pod uses exactly 16 slices:** 4 hosts x 4 slices = 16 slices per pod.
   The superpod's 64 stages exactly fill the 64 available slices with no waste.

### CI Configuration

A CI variant uses placeholder hostnames (`host-0` through `host-3`) that are
replaced at runtime via the `--hostfile` flag:

> **Source**: `scaleout_configs/blitz_pipeline_config_single_pod_ci.yaml`, lines 1-3

---

## 8.1.4 Snake-Order Host Traversal

Examining the single-pod config reveals a specific traversal pattern across the
four hosts:

```
Stage:  0  1  2  3 | 4  5  6  7 | 8  9 10 11 | 12 13 14 15
Host:  d05u08       | d05u02     | d06u02     | d06u08
Slice:  0  1  2  3 |  0  1  2  3 |  0  1  2  3 |  0  1  2  3
```

The host order is: `d05u08 -> d05u02 -> d06u02 -> d06u08`. This is a
**snake/zigzag** pattern that minimizes inter-host hops:

```
           Cabinet d05         Cabinet d06
           +--------+          +--------+
  u08  --> | Host 0 |          | Host 3 | <-- u08
           |   |    |          |   ^    |
           |   v    |          |   |    |
  u02      | Host 1 | -------> | Host 2 |     u02
           +--------+          +--------+

Traversal: Host0 -> Host1 -> Host2 -> Host3
           (u08)    (u02)    (u02)    (u08)
```

**Design rationale -- why snake order:** Within a cabinet, the u08 and u02
hosts share direct high-bandwidth interconnect. Between cabinets, the u02-to-u02
link provides the cross-cabinet connection. The snake order ensures that every
consecutive host-to-host transition uses a direct physical link. A naive
sequential order (u02, u08, u02, u08) would require cross-cabinet hops every
other transition, doubling latency for half the inter-host transfers.

The `sort_hosts_canonical()` function implements this ordering:

> **Source**: `generate_blitz_decode_pipeline_configs.py`, lines 27-54

Hostnames are parsed by `_parse_host()` (lines 17-24) using the regex
`(\d+)u(\d{2})$` to extract cabinet number and u-position. The sorting groups
hosts by cabinet number, then alternates sort direction: even-indexed cabinets
sort u-numbers descending (u08 first), odd-indexed cabinets sort ascending
(u02 first). This produces the zigzag pattern regardless of the input order.

---

## 8.1.5 Mesh Graph Descriptor Textproto Files

Each pipeline configuration references a mesh graph descriptor file (textproto
format) that defines the physical fabric topology for the TT runtime. These
files contain three key sections:

**1. Mesh Descriptors** -- Define the shape of each slice:

```protobuf
mesh_descriptors {
  name: "M0"
  arch: BLACKHOLE
  device_topology { dims: [ 4, 2 ] dim_types: [ RING, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2  policy: RELAXED }
}
```

**2. Graph Instances** -- One mesh instance per pipeline stage:

```protobuf
instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
...
```

**3. Connections** -- Fabric links between adjacent stages:

```protobuf
connections {
  nodes { mesh { mesh_descriptor: "M0" mesh_id: N } }
  nodes { mesh { mesh_descriptor: "M0" mesh_id: N+1 } }
  channels { count: 8 policy: RELAXED }
}
```

The single-pod descriptor includes a **loopback connection** from the last
stage back to stage 0, forming a ring:

> **Source**: `blitz_decode_single_pod_mesh_graph_descriptor.textproto`, lines 110-114:
> ```protobuf
> connections {
>   nodes { mesh { mesh_descriptor: "M0" mesh_id: 15 } }
>   nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
>   channels { count: 8 policy: RELAXED }
> }
> ```

### Bandwidth Tiers Across Configurations

| Connection Type | Channel Count | Notes |
|---|---|---|
| Intra-pod (same host or adjacent hosts) | 8 | Full bandwidth |
| Pod-boundary-adjacent (superpod) | 6 | 2 channels reserved for z-direction |
| Inter-pod (superpod) | 4 | `assign_z_direction: true` |
| Inter-pod (2-pod) | 2 | Reduced bandwidth |

The pod-boundary-adjacent connections (count: 6 instead of 8) exist because 2
of the 8 normal channels are reserved for the z-direction inter-pod links on
those slices. The inter-pod connections themselves use `assign_z_direction: true`,
which physically routes through the vertical ethernet cables connecting pods in
the Galaxy rack.

**Design rationale -- why a ring with loopback:** The loopback connection
eliminates a host round-trip for feeding the next-token embedding. Without it,
LM head output would traverse: device -> D2H -> host CPU -> H2D -> device.
With loopback, output flows: device(stage N) -> D2D fabric -> device(stage 0).
For decode where per-step latency is critical, eliminating the PCIe round-trip
saves approximately 10-20 us per token.

> **Cross-reference -> Chapter 3, Section 3.2**: D2D exchange micro-op
> kernel details.

---

## 8.1.6 Configuration Generator: `generate_blitz_decode_pipeline_configs.py`

This script (262 lines) automates the process of transforming a pipeline config
YAML into the rank bindings and rank files needed by `tt-run` (the multi-host
MPI launcher). It performs three operations:

### Step 1: Physical Device Discovery via MPI

The `generate_slice_to_pcie_device_mapping()` function (lines 57-121) runs a
physical discovery test across all hosts:

> **Source**: `generate_blitz_decode_pipeline_configs.py`, lines 57-120

```
 mpirun --np <num_hosts> --host <host_vector>
        --mca btl self,tcp --bind-to none --tag-output
        build/test/tt_metal/tt_fabric/test_physical_discovery
        --gtest_filter=*Generate2x4SliceToPCIeDeviceMapping*
```

This executes `test_physical_discovery` on each host via MPI, which probes the
PCIe bus to discover which physical device IDs correspond to each 4x2 slice.
The output is a `slice_to_pcie_device_mapping.yaml` file that maps
`[host][slice] -> [list of PCIe device IDs]`.

**Design rationale -- why runtime discovery:** PCIe device numbering is not
deterministic across reboots or across different host configurations. A
hardcoded mapping would break when devices are renumbered. The discovery test
probes the actual hardware topology and writes the canonical mapping.

### Step 2: Rank Bindings

The `generate_rank_bindings()` function (lines 123-163) creates a YAML file
mapping each MPI rank to its mesh ID and environment:

> **Source**: `generate_blitz_decode_pipeline_configs.py`, lines 123-163

```python
rank_bindings.append({
    "rank": stage,
    "mesh_id": stage,
    "mesh_host_rank": 0,
    "env_overrides": {
        "TT_VISIBLE_DEVICES": ",".join(map(str, devices_for_stage))
    },
})
```

The rank binding config also includes `mesh_graph_desc_path` pointing to the
textproto file. When running on remote workers with a different `tt-metal`
installation path, the `--worker-tt-metal-home` flag overrides paths via
`global_env` (lines 152-163).

### Step 3: MPI Rank File

The `generate_rank_file()` function (lines 166-172) produces an OpenMPI rank
file that pins each rank to its host:

```
rank 0=bh-glx-d05u08 slot=0-31
rank 1=bh-glx-d05u08 slot=0-31
...
rank 15=bh-glx-d06u08 slot=0-31
```

Each rank is bound to all 32 CPU slots on its host, allowing the process to
use any core for host-side work.

### Hostfile Override for CI

The `--hostfile` argument (lines 193-209) allows CI systems to provide
dynamically allocated hosts. The script reads the hostfile, sorts the allocated
hosts into canonical order using `sort_hosts_canonical()`, and remaps the
config's placeholder hostnames. If the hostfile provides a different number of
hosts than the pipeline config, the script fails immediately with a hard error.

---

## 8.1.7 End-to-End Configuration Generation Flow

```
 CONFIGURATION GENERATION PIPELINE
 ==================================

 [1] Author pipeline YAML               [2] Optional: provide hostfile
     (stage_to_slice_mapping)                 (one hostname per line)
              |                                       |
              v                                       v
 +---------------------------------------------------+
 | generate_blitz_decode_pipeline_configs.py          |
 |                                                     |
 |  1. sort_hosts_canonical()                         |
 |       Reorder hosts into snake/zigzag order        |
 |                                                     |
 |  2. generate_slice_to_pcie_device_mapping()        |
 |       mpirun test_physical_discovery across hosts  |
 |       Output: slice_to_pcie_device_mapping.yaml    |
 |                                                     |
 |  3. generate_rank_bindings()                       |
 |       Output: rank_binding.yaml                    |
 |       (rank -> mesh_id, TT_VISIBLE_DEVICES)        |
 |                                                     |
 |  4. generate_rank_file()                           |
 |       Output: rank_file (OpenMPI format)           |
 |       (rank N = hostname slot=0-31)                |
 +---------------------------------------------------+
              |
              v
 [3] tt-run (MPI launcher) reads:
     +-- rank_binding.yaml   (device assignment per rank)
     +-- rank_file           (host assignment per rank)
     +-- mesh_graph_desc     (fabric topology textproto)
```

---

## 8.1.8 Mesh Shape Validation

The demo requires each pipeline stage to use a `(4, 2)` mesh -- 4 rows and 2
columns of Blackhole chips. This is validated at startup:

> **Source**: `demo/cli.py`, lines 37 and 49-51:
> ```python
> EXPECTED_PIPELINE_STAGE_MESH_SHAPE = (4, 2)
> ...
> assert mesh_shape == EXPECTED_PIPELINE_STAGE_MESH_SHAPE, ...
> ```

This 4x2 shape is critical because the intra-device parallelism strategy
depends on it: MLA attention uses tensor parallelism across the 2 columns
(TP=2), and MoE uses expert parallelism across all 8 devices (EP=8).

> **Cross-reference -> Chapter 4, Section 4.3**: MLA attention TP strategy
> on the 4x2 mesh.

> **Cross-reference -> Chapter 5, Section 5.2**: MoE expert parallelism
> across the 4x2 mesh.

---

## 8.1.9 Operational Failure Mode Summary

| Failure Scenario | Detection | Impact | Mitigation |
|---|---|---|---|
| Unrecognized hostname in sort | Warning logged | Silent wrong ordering | Validate hostnames before deploy |
| Hostfile count mismatch | `sys.exit(1)` | Hard failure | Verify host allocation |
| Missing test_physical_discovery binary | `sys.exit(1)` | Hard failure | Run `build_metal.sh --build-tests` |
| MPI discovery returns non-zero | `sys.exit(returncode)` | Hard failure | Check network/SSH config |
| worker-tt-metal-home path wrong | Workers fail to find executable | Runtime crash | Verify path on workers |
| Fabric router sync timeout (30s) | Process exit | Pipeline fails to start | Increase via env var |
| 2-pod config with loopback decode loop | No loopback in textproto | Decode loop impossible | Use superpod config |
| Inter-pod bandwidth bottleneck | Degraded performance | Higher per-token latency | Profile pipeline balancing |

---

## 8.1.10 Configuration Summary

| Configuration | Stages | Hosts | Pods | Loopback | Layers Covered |
|---------------|--------|-------|------|----------|---------------------------|
| Single pod    | 16     | 4     | 1    | Yes      | Partial (stages 0-15)     |
| Single pod CI | 16     | 4     | 1    | Yes      | Partial (placeholder)     |
| Two pod       | 32     | 8     | 2    | No       | Partial (stages 0-31)     |
| Superpod      | 64     | 16    | 4    | Yes      | Full model (stages 0-63)  |

The superpod is the production configuration for full 671B inference. The
single-pod and 2-pod configurations serve as development and testing
stepping-stones, running partial pipelines that exercise the D2D exchange
and pipeline block infrastructure without requiring the full cluster.

---

## 8.1.11 Design Rationale: Pipeline vs. Tensor Parallelism for Inter-Host

The choice of pipeline parallelism for inter-host scaleout (rather than tensor
parallelism) is the most consequential architectural decision in the B1
implementation. The reasoning involves three factors:

**1. Communication volume comparison:**

| Strategy | Per-layer inter-host comm | 61 layers total |
|---|---|---|
| TP across hosts | AllReduce 14 KB x 2 (attn + FFN) per layer | ~1.7 MB |
| Pipeline parallel | D2D forward 14 KB x 1 per layer | ~0.86 MB |

TP requires two all-reduces per layer (one after attention, one after FFN), each
communicating the full hidden state. Pipeline parallelism requires one
point-to-point forward per layer. Although the total bytes are comparable, the
latency profile differs dramatically.

**2. Latency structure:**

- TP all-reduce latency: ~5-20 us per operation depending on host distance. With
  2 all-reduces per layer and 61 layers, this adds 610-2440 us (~0.6-2.4 ms)
  to the critical path.
- Pipeline D2D latency: ~0.2-5 us per hop. Each hop is overlapped with the
  previous stage's computation, so only one hop latency (the pipeline startup
  bubble) adds to the critical path.

**3. Memory efficiency:**

Pipeline parallelism places one full layer per stage (8 devices). Each device
holds 1/2 of the attention weights (TP=2) and 1/8 of the MoE expert weights
(EP=8). TP across hosts would require splitting weights more finely, reducing
per-device weight size but increasing the number of all-reduce operations and
fragmenting the DRAM streaming pattern.

The pipeline approach trades a one-time bubble (empty pipeline stages at startup
and drain) for per-step latency reduction. For decode workloads where throughput
is measured in tokens-per-second and each token depends on the previous, the
per-step latency dominates, making pipeline parallelism the clear choice.
