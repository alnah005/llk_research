# 5.5 Hardware Configurations

This section documents the `HardwareConfig` and `ModelPipelineSpec` dataclasses that map logical pipeline layouts to physical hardware, covering single-, dual-, and quad-galaxy configurations. It also describes the mesh graph descriptor files, rank binding, pipeline factory functions, scaling tradeoffs, and the complete end-to-end token lifecycle that ties all five sections together.

**Prerequisites:** Section 5.1-5.2 (PipelineGraph, SubmeshPartition, topology auto-discovery), Ch1 (galaxy physical layout).

---

## HardwareConfig

Defined in `tests/pipeline_builder/test_model_pipelines.py` (line 69):

```python
@dataclass(frozen=True)
class HardwareConfig:
    mesh_device_shape: tuple[int, int]   # (rows, cols) of the full MeshDevice
    mesh_graph_desc: str                 # mesh graph descriptor filename
    num_procs: int                       # expected MPI process count
    rank_binding: Optional[str] = None   # rank binding YAML (for tt-run)
```

The `mesh_graph_desc_path` property (line 80) resolves to the full path under `tests/pipeline_builder/mesh_graph_descriptors/`. The mesh graph descriptor is a textproto file that tells the runtime how to map physical chips to logical mesh coordinates. Different submesh shapes require different descriptors even on the same hardware because the tiling pattern changes.

---

## ModelPipelineSpec

Defined at line 85:

```python
@dataclass(frozen=True)
class ModelPipelineSpec:
    model_name: str       # e.g. "deepseek-v3-b1-1g-4x2"
    num_galaxies: int     # BH galaxies required
    stage_rows: int       # rows per stage submesh
    stage_cols: int       # cols per stage submesh

    @property
    def chips_per_stage(self) -> int:
        return self.stage_rows * self.stage_cols

    @property
    def num_stages(self) -> int:
        return (self.num_galaxies * 32) // self.chips_per_stage
```

The key derived property: `num_stages = (num_galaxies * 32) / chips_per_stage`. Each BH galaxy has 32 chips. Larger submesh shapes mean fewer stages (more tensor parallelism per stage); smaller shapes mean more stages (deeper pipeline parallelism).

---

## Configuration Lookup Table

The `_HARDWARE_CONFIGS` dict (line 122) maps `(num_galaxies, stage_rows, stage_cols)` to `HardwareConfig`. The full table:

### Single Galaxy (32 chips)

| Stage Shape | TP | Stages | Mesh Shape | Procs | Descriptor |
|------------|-----|--------|------------|-------|------------|
| 4x2 | 8 | 4 | (8, 4) | 4 | `single_bh_galaxy_4x2_...` |
| 2x2 | 4 | 8 | (8, 4) | 8 | `single_bh_galaxy_2x2_...` |
| 1x2 | 2 | 16 | (4, 8) | 16 | `single_bh_galaxy_1x2_...` |
| 1x1 | 1 | 32 | (4, 8) | 32 | `single_bh_galaxy_1x1_...` |

Note: the 1x2 and 1x1 configurations use a *transposed* mesh shape `(4, 8)` compared to the 4x2 and 2x2 `(8, 4)`. This is because the submesh tiling must evenly divide the parent mesh dimensions -- `MeshShape(1, 2)` tiles `(4, 8)` into 16 submeshes but cannot tile `(8, 4)` into uniform 1x2 blocks without leftover columns.

### Dual Galaxy (64 chips)

| Stage Shape | TP | Stages | Mesh Shape | Procs | Descriptor |
|------------|-----|--------|------------|-------|------------|
| 4x2 | 8 | 8 | (16, 4) | 8 | `dual_bh_galaxy_4x2_...` |
| 2x2 | 4 | 16 | (16, 4) | 16 | `dual_bh_galaxy_2x2_...` |
| 1x2 | 2 | 32 | (8, 8) | 32 | `dual_bh_galaxy_1x2_...` |

### Quad Galaxy (128 chips)

| Stage Shape | TP | Stages | Mesh Shape | Procs | Descriptor |
|------------|-----|--------|------------|-------|------------|
| 4x2 | 8 | 16 | (32, 4) | 16 | `quad_bh_galaxy_4x2_...` |
| 2x2 | 4 | 32 | (16, 8) | 32 | `quad_bh_galaxy_2x2_...` |
| 1x2 | 2 | 64 | (16, 8) | 64 | `quad_bh_galaxy_1x2_...` |

> **Warning:** The MeshDevice shape is NOT always `(total_rows, total_cols)` in the obvious way. Compare 1-galaxy 4x2 vs 1x2: the 4x2 config uses (8,4) but the 1x2 config transposes to (4,8). Using the wrong MeshDevice shape for a given stage shape causes `create_submeshes()` to produce the wrong number of submeshes.

---

## Physical Topology Visualizations

### Single Galaxy, 4x2 Stages (4 stages)

```
  MeshDevice (8, 4)
  +--------+--------+
  | stage0 | stage1 |     Each stage: 4x2 = 8 chips
  | (4x2)  | (4x2)  |     TP = 8 (tensor parallel)
  +--------+--------+
  | stage2 | stage3 |     4 hosts, 1 per stage
  | (4x2)  | (4x2)  |
  +--------+--------+

  Data flow: S0 -> S1 -> S2 -> S3 --loopback--> S0
```

### Quad Galaxy, 1x2 Stages (64 stages -- full DeepSeek V3)

```
  MeshDevice (16, 8)

  +--+--+--+--+--+--+--+--+
  |S0|S1|S2|S3|S4|S5|S6|S7|   row 0
  +--+--+--+--+--+--+--+--+
  |S8|  |  |  |  |  |  |15|   row 1
  +--+--+--+--+--+--+--+--+
  :  :  :  :  :  :  :  :  :   ...
  +--+--+--+--+--+--+--+--+
  |56|  |  |  |  |  |  |63|   row 15
  +--+--+--+--+--+--+--+--+

  64 stages x 2 chips each = 128 chips = 4 galaxies
  Pipeline: Embed -> Dense(0-2) -> MoE(3-60) -> LMHead -> Passthrough -> loopback
```

---

## Pipeline Factory Functions

`pipeline.py` provides factory functions that map stage counts to model layer assignments:

### Single Galaxy (4 stages, line 35)
```
  Stage 0: EmbeddingStage
  Stage 1: LMHeadStage
  Stage 2: PassthroughStage (token)
  Stage 3: PassthroughStage (token)
```

This is a test/validation configuration -- it does not include decoder layers.

### Single Pod (16 stages, line 63)
```
  Stage  0: EmbeddingStage
  Stage  1: DenseDecoderStage (layer 0)
  Stage  2: DenseDecoderStage (layer 1)
  Stage  3: DenseDecoderStage (layer 2)
  Stage  4-13: MoEDecoderStage (layers 3-12)
  Stage 14: LMHeadStage
  Stage 15: PassthroughStage (token)
```

### Super-Pod 4 (64 stages, line 109)
```
  Stage  0: EmbeddingStage
  Stage  1: DenseDecoderStage (layer 0)
  Stage  2: DenseDecoderStage (layer 1)
  Stage  3: DenseDecoderStage (layer 2)
  Stage  4-61: MoEDecoderStage (layers 3-60)
  Stage 62: LMHeadStage
  Stage 63: PassthroughStage (token)
```

The `create_pipeline_configuration_from_num_procs()` function (line 155) auto-selects the configuration based on the number of MPI processes: 4 -> single galaxy, 16 -> single pod, 64 -> SP4.

> **Warning:** Process counts of 8 and 32 are valid hardware configurations (see the matrix above) but have no automatic pipeline factory. For these, you must construct `PipelineConfiguration` manually with custom stage factories.

> **Warning:** Layer ID overrides (`dense_layer_id_override`, `moe_layer_id_override`) exist for testing purposes. When set, all dense or MoE stages use the same layer weights. Do not use in production -- each stage must load its own layer.

---

## Mesh Graph Descriptors

Each `HardwareConfig` references a `.textproto` file under `tests/pipeline_builder/mesh_graph_descriptors/`. These descriptors define:

1. **Chip IDs** -- the set of chips in the mesh and their physical coordinates.
2. **Ethernet links** -- which chips are connected by ethernet and the link direction.
3. **Mesh topology** -- row-major layout of chips in the grid.

The descriptor file is passed to `MeshDevice.open()` (or the test fixture) and determines how `create_submeshes()` tiles the mesh. The C++ topology resolver uses the fabric routing information derived from these descriptors to discover inter-submesh ethernet links.

> **Warning:** Using a descriptor that does not match the physical hardware causes silent topology resolution failures. The resolver may find no valid assignment or assign stages to disconnected submeshes. If galaxy ethernet links are physically recabled, the descriptor no longer matches reality and sockets open on non-existent links, hanging at connect time.

---

## Rank Binding

The rank binding YAML (referenced by `HardwareConfig.rank_binding`) maps MPI ranks to physical chip groups. It is passed to the launcher (`tt-run`) at deployment time, not read by the pipeline code.

The pipeline system does NOT assume `rank == stage_idx`. Instead, the topology resolver and MPI allgather (Section 5.2) handle this automatically -- the `stage_to_rank` mapping is always resolved dynamically.

> **Warning:** If the rank binding assigns two ranks to overlapping device sets, both ranks will open the same physical devices. The TTNN device driver does not detect this -- both ranks proceed, but hardware registers are corrupted by concurrent access. Symptoms: random data corruption, kernel hangs, or silent wrong results. Always validate rank binding with `print_submeshes()` before running real workloads.

---

## Scaling Tradeoffs

| Dimension | Fewer, Larger Stages (4x2) | More, Smaller Stages (1x1/1x2) |
|-----------|---------------------------|-------------------------------|
| TP parallelism | 8 chips per stage (high) | 1-2 chips per stage (low) |
| Pipeline depth | 4-16 stages (low latency per pass) | 32-64 stages (high throughput) |
| Weight memory | Each stage holds many layers | Each stage holds 1 layer |
| Ethernet hops | Fewer inter-stage transfers | More inter-stage transfers |
| Host count | 4-16 hosts (lower infra cost) | 32-64 hosts (higher infra cost) |

The 4x2 configuration maximizes tensor parallelism within each stage. The 1x2 configuration maximizes pipeline parallelism, enabling more concurrent tokens in flight. Production deployments typically use 1x2 (16-stage single galaxy or 64-stage quad galaxy) or 4x2 (16-stage quad galaxy) as the balance point.

---

## Configuration Selection Checklist

When deploying a new model pipeline, verify:

1. Determine `(num_galaxies, stage_rows, stage_cols)` from model requirements.
2. Look up HardwareConfig in `_HARDWARE_CONFIGS` -- confirm it exists.
3. Verify MeshDevice shape matches `hw.mesh_device_shape`.
4. Verify MPI process count matches `hw.num_procs`.
5. Verify mesh graph descriptor file exists and matches physical hardware.
6. Verify rank binding YAML assigns one rank per submesh with no overlap.
7. Run topology test to validate auto-discovery.
8. Run `print_submeshes()` from rank 0 to visually confirm stage assignment.
9. Check that stage chain page sizes are consistent (Section 5.3, consistency table).
10. Run end-to-end test before production deployment.

---

## Topology Test Framework

The test file `test_model_pipelines.py` contains parametrized tests for all 10 configurations. Each test:

1. Skips if the process count does not match `HardwareConfig.num_procs`.
2. Builds a linear loopback `PipelineGraph` with `num_stages` nodes.
3. Calls `graph.build_topology(mesh_device)`.
4. Asserts the discovered layout has the expected number of submeshes.

```python
def _build_linear_graph(spec: ModelPipelineSpec) -> PipelineGraph:
    n = spec.num_stages
    return PipelineGraph(
        nodes={f"s{i}": Node(shape=spec.stage_shape) for i in range(n)},
        edges=(
            [Edge(f"s{i}", f"s{i+1}") for i in range(n - 1)]
            + [Edge(f"s{n-1}", "s0", is_loopback=True)]
        ),
    )
```

This validates that the C++ topology resolver can find a valid physical layout for every supported hardware configuration without any hardcoded chip IDs.

> **Warning:** These tests validate topology discovery only -- they do not run actual compute. A passing topology test guarantees that submesh-to-stage assignment is valid and all ethernet links are found, but does NOT guarantee that data flows correctly through the pipeline. Use `test_deepseek_stages.py` for full validation.

---

## End-to-End Token Lifecycle Summary

Bringing all five sections together, here is the complete token lifecycle:

```
 1. Inference server pushes ISRequest to PipelineManager       [Sec 5.4]
 2. API thread: ALLOCATE slot, SUBMIT prompt                   [Sec 5.4]
 3. Writer thread: inject prefill tokens via H2D socket        [Sec 5.4]
 4. H2D socket delivers token to Stage 0 entry chip            [Sec 5.1]
 5. EmbeddingStage: token -> activation (14KB)                 [Sec 5.3]
 6. D2D socket: activation flows through decoder stages        [Sec 5.3]
 7. LMHeadStage: activation -> sampled token (64B)             [Sec 5.3]
 8. PassthroughStage: token -> loopback to Stage 0             [Sec 5.3]
 9. D2H socket delivers result token to host                   [Sec 5.1]
10. Reader thread: deserialize, emit OutputMessage             [Sec 5.4]
11. Writer thread: re-inject for next decode step              [Sec 5.4]
12. Repeat 4-11 until EOS or max_new_tokens                   [Sec 5.4]
```

The hardware configuration (Section 5.5) determines how many stages exist and how they map to physical chips. The `PipelineGraph` (Section 5.1) and `SubmeshPartition` (Section 5.2) abstract away the physical details so the same pipeline code runs on 4 stages or 64.

---

## Configuration-to-Topology Flow

```
  ModelPipelineSpec                HardwareConfig
  (model_name, galaxies,          (mesh_shape, descriptor,
   stage_rows, stage_cols)         num_procs, rank_binding)
        |                                |
        v                                v
  PipelineGraph                   MeshDevice.open(descriptor)
  (nodes, edges)                  mesh_device
        |                                |
        +---------- build_topology() ----+
                         |
                         v
                   PipelineLayout
                   (submesh assignments, pipeline_config,
                    stage metadata, factories)
                         |
              +----------+----------+
              |                     |
        Pipeline.setup_and_run()   PipelineManager.start()
```

---

| Previous | Up | Next |
|----------|-----|------|
| [5.4 Pipeline Manager (C++)](04_pipeline_manager_cpp.md) | [Table of Contents](../README.md) | [Ch6: Integration and Deployment](../ch6/) |
