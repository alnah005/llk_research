# Weight Sharding Strategies

Tensor parallelism in TT-Symbiote is implemented through a family of linear layer classes that control how weight matrices are split across devices and how the results are combined. Each class embodies a specific sharding strategy: column-parallel layers split the output dimension so each device computes a slice of the result, while row-parallel layers split the input dimension so each device computes a partial sum that requires collective reduction. This file documents every sharding variant in the linear module hierarchy, the weight distribution mechanism, bfloat8 variants for memory-constrained models, and the activation flow pattern through a complete transformer layer. Prerequisites: File 1 of this chapter (DistributedConfig, mesh mappers) and [Chapter 2, File 1](../ch02_symbiote_core/01_module_replacement_engine.md) (`TTNNModule` lifecycle, `preprocess_weights_impl`, `move_weights_to_device_impl`).

## The Linear Module Hierarchy

All distributed linear layers inherit from `TTNNLinear`, which provides the base forward pass [linear.py:TTNNLinear]:

```
TTNNLinear                                    (single-device, replicated)
  |
  +-- TTNNLinearInputShardedWeightSharded     (abstract base for sharded variants)
  |     |
  |     +-- TTNNLinearIColShardedWRowSharded  (row-parallel: reduce_scatter)
  |           |
  |           +-- TTNNLinearIColShardedWAllReduced  (all-reduce = RS + AG)
  |           +-- TTNNLinearLLamaIColShardedWRowSharded  (bfloat8)
  |
  +-- TTNNLinearInputReplicatedWeightSharded  (abstract base for col-sharded)
  |     |
  |     +-- TTNNLinearIReplicatedWColSharded  (column-parallel: no CCL)
  |
  +-- TTNNLinearLLama                         (bfloat8, single-device)
```

The naming convention encodes the sharding strategy: **I** = Input, **W** = Weight, **Replicated** = same data on all devices, **ColSharded** = sharded along last dim (columns), **RowSharded** = sharded along second-to-last dim (rows).

## Column-Parallel: TTNNLinearIReplicatedWColSharded

**Strategy:** Input is replicated (identical on all devices), weight is sharded along the output dimension (dim=-1). Each device computes a different slice of the output. No CCL operation is needed.

**When to use:** Q/K/V projections and MLP gate/up projections -- any layer where the output will be consumed by a subsequent row-parallel layer or by operations that work on sharded data.

### Weight Distribution

During `move_weights_to_device_impl` [linear.py:TTNNLinearInputReplicatedWeightSharded.move_weights_to_device_impl], the weight is sharded using `shard_tensor_to_mesh_mapper`:

```python
def move_weights_to_device_impl(self):
    if isinstance(self.tt_weight_host, torch.Tensor):
        self.tt_weight_host = preprocess_linear_weight(
            self.tt_weight_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(
                self.device, dim=self.weight_dim  # dim=-1
            ),
        )
```

For a weight `[4096, 4096]` on T3K (8 devices), each device gets `[4096, 512]`.

### Data Flow

```
  Device 0        Device 1       ...    Device 7
  +---------+    +---------+           +---------+
  | Input   |    | Input   |           | Input   |    <-- replicated
  | [B,S,H] |    | [B,S,H] |           | [B,S,H] |
  +----+----+    +----+----+           +----+----+
       |              |                     |
  W[:,0:H/8]    W[:,H/8:2H/8]        W[:,7H/8:H]    <-- weight shards
       |              |                     |
  +----v----+    +----v----+           +----v----+
  | Output  |    | Output  |           | Output  |    <-- sharded output
  | [B,S,   |    | [B,S,   |           | [B,S,   |
  |  H/8]   |    |  H/8]   |           |  H/8]   |
  +---------+    +---------+           +---------+
                 NO CCL NEEDED
```

### Forward Pass

```python
@run_on_devices(DeviceArch.T3K)
def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    # ... shape normalization to 4D ...
    tt_output = ttnn.linear(input_tensor, self.tt_weight,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if self.tt_bias is not None:
        tt_output += self.tt_bias
    # ... reshape back ...
    return tt_output
```

The `@run_on_devices(DeviceArch.T3K)` decorator gates this distributed forward path to T3K hardware; on single devices, the base class `TTNNLinear.forward()` runs instead.

## Row-Parallel: TTNNLinearIColShardedWRowSharded

**Strategy:** Input arrives column-sharded (each device holds a slice of dim=-1), weight is row-sharded (dim=-2, the input dimension). Each device computes a partial sum. A `reduce_scatter` combines the partial sums.

**When to use:** O-projection in attention, MLP down projection -- any layer that receives column-sharded activations and needs to produce a reduced result.

### Weight Distribution

The weight is sharded on dim=-2 [linear.py:TTNNLinearInputShardedWeightSharded.move_weights_to_device_impl]:

```python
weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(
    self.device, dim=self.weight_dim),  # dim=-2
```

For weight `[4096, 4096]` on T3K: each device gets `[512, 4096]` (input rows sharded).

### Data Flow

```
  Device 0        Device 1       ...    Device 7
  +---------+    +---------+           +---------+
  | Input   |    | Input   |           | Input   |    <-- col-sharded
  | [B,S,   |    | [B,S,   |           | [B,S,   |
  |  H/8]   |    |  H/8]   |           |  H/8]   |
  +----+----+    +----+----+           +----+----+
       |              |                     |
  W[:,0:H/8]    W[:,H/8:2H/8]        W[:,7H/8:H]     <-- weight shards
       |              |                     |
  +----v----+    +----v----+           +----v----+
  | Partial |    | Partial |           | Partial |    <-- partial sums
  | [B,S,O] |    | [B,S,O] |           | [B,S,O] |
  +----+----+    +----+----+           +----+----+
       |              |                     |
       +--------- reduce_scatter ----------+
       |              |                     |
  +----v----+    +----v----+           +----v----+
  | Output  |    | Output  |           | Output  |    <-- reduced shards
  | [B,S,   |    | [B,S,   |           | [B,S,   |
  |  O/8]   |    |  O/8]   |           |  O/8]   |
  +---------+    +---------+           +---------+
```

### Forward Pass

```python
@run_on_devices(DeviceArch.T3K)
def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    # ... shape normalization ...
    tt_output = ttnn.linear(input_tensor, self.tt_weight,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_output = ttnn.reduce_scatter(
        tt_output,
        dim=3,
        num_links=1,
        cluster_axis=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
    )
    if self.tt_bias is not None:
        tt_output += self.tt_bias
    return tt_output
```

The `reduce_scatter` sums each device's partial result and distributes the reduced output so each device holds `1/8` of the final result. The bias is added after `reduce_scatter` since it is already sharded to match.

> **Note:** Both `reduce_scatter` and `all_gather` in this chapter use `topology=ttnn.Topology.Ring`. Ring topology is required for trace compatibility -- see [File 3, Trace-Compatible All-Reduce](./03_ccl_operations.md) for the full explanation of why monolithic `ttnn.all_reduce` cannot be used in traced modules.

## All-Reduce Variant: TTNNLinearIColShardedWAllReduced

**Strategy:** Same as row-parallel, but instead of `reduce_scatter` alone, it decomposes the all-reduce into `reduce_scatter` + `all_gather`. The result is **replicated** on all devices rather than column-sharded.

**When to use:** When the output must be identical on all devices -- fused QKV projections in Gemma4 attention, fused gate-up projections in Gemma4 MLP, or any layer feeding operations that expect full tensors (e.g., per-head slicing, element-wise operations).

### Forward Pass

```python
@run_on_devices(DeviceArch.T3K)
def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    # ... shape normalization ...
    tt_output = ttnn.linear(input_tensor, self.tt_weight,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # Decompose all_reduce = reduce_scatter + all_gather
    # for trace compatibility
    tt_output = ttnn.reduce_scatter(
        tt_output, dim=3, num_links=1, cluster_axis=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
    )
    tt_output = ttnn.all_gather(
        tt_output, dim=3, num_links=1, cluster_axis=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
    )
    if self.tt_bias is not None:
        tt_output += self.tt_bias
    return tt_output
```

> **Note:** The `reduce_scatter` + `all_gather` decomposition is the trace-compatible form of all-reduce (see note in Row-Parallel section above).

### Data Flow

```
  Partial sums [B, S, O] per device (from matmul)
       |
  reduce_scatter(dim=3)     -->  [B, S, O/8]  (col-sharded reduced)
       |
  all_gather(dim=3)         -->  [B, S, O]    (replicated, full result)
       |
  Output: identical on all 8 devices
```

## shard_tensor_to_mesh_mapper

The `ttnn.shard_tensor_to_mesh_mapper(device, dim=N)` function creates a mapper that evenly slices a tensor along dimension `N` into `get_num_devices()` parts. During `move_weights_to_device_impl()`, this mapper is passed to `preprocess_linear_weight()`:

```python
self.tt_weight_host = preprocess_linear_weight(
    self.tt_weight_host,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(
        self.device, dim=self.weight_dim
    ),
)
```

For column-parallel (`dim=-1`): the weight's output dimension is divided. A `[4096, 4096]` weight becomes `[4096, 512]` per device.

For row-parallel (`dim=-2`): the weight's input dimension is divided. A `[4096, 4096]` weight becomes `[512, 4096]` per device (in TTNN's transposed weight format, this corresponds to the second-to-last dimension).

> **Warning:** The weight tensor dimensions must be evenly divisible by the number of devices along the sharding axis. For T3K `(1, 8)` with `dim=-1`, the output features must be divisible by 8. If not, `preprocess_linear_weight` will fail. There is no automatic padding.

## bfloat8 Variants for LLaMA

LLaMA-optimized variants use `bfloat8_b` precision for weight storage, roughly halving weight memory compared to `bfloat16` [linear.py:TTNNLinearLLama, TTNNLinearLLamaIColShardedWRowSharded]:

**TTNNLinearLLama** -- Single-device bfloat8 with `@deallocate_weights_after`:

```python
@trace_disabled
class TTNNLinearLLama(TTNNLinear):
    def preprocess_weights_impl(self):
        self.tt_weight_host = preprocess_linear_weight(
            self.weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )

    @deallocate_weights_after
    def forward(self, input_tensor):
        return super().forward(input_tensor)
```

**TTNNLinearLLamaIColShardedWRowSharded** -- Multi-device bfloat8 row-parallel:

```python
@trace_disabled
class TTNNLinearLLamaIColShardedWRowSharded(TTNNLinearIColShardedWRowSharded):
    def move_weights_to_device_impl(self):
        if isinstance(self.tt_weight_host, torch.Tensor):
            self.tt_weight_host = preprocess_linear_weight(
                self.tt_weight_host,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(
                    self.device, dim=self.weight_dim
                ),
            )

    @deallocate_weights_after
    @run_on_devices(DeviceArch.T3K)
    def forward(self, input_tensor):
        return super().forward(input_tensor)
```

Both LLaMA variants are `@trace_disabled` because the `@deallocate_weights_after` pattern (allocate on first call, deallocate after each call) creates dynamic allocation incompatible with trace replay (which requires stable buffer addresses). The decorator stacking on `TTNNLinearLLamaIColShardedWRowSharded.forward()` -- `@deallocate_weights_after` then `@run_on_devices(DeviceArch.T3K)` -- means the T3K device check happens first (inner decorator), and weight deallocation wraps the result (outer decorator).

> **Warning:** The `@deallocate_weights_after` decorator frees device weight memory after `forward()` returns. This saves DRAM between layers but prevents traced execution. Use the non-LLaMA variants when trace mode is required.

## Activation Flow Through a Transformer Layer

The sharding strategies form complementary pairs that chain together through a transformer layer. The exact pairing varies by model.

### Ling/Bailing Activation Pattern

Ling/Bailing uses a blanket replacement mapping all `nn.Linear` layers to `TTNNLinearIColShardedWRowSharded` (row-parallel). Every linear layer takes col-sharded input and produces col-sharded output via `reduce_scatter`. The distributed RMSNorm (`TTNNDistributedRMSNorm`) also produces col-sharded output (weights are sharded along dim=2), so the entire data path remains col-sharded throughout:

```
         COL-SHARDED (from DistributedRMSNorm)
  hidden_states [B, S, H/8]
         |
         v
  +-------------------+      reduce_scatter
  | Row-Parallel      | =====================>   COL-SHARDED
  | QKV projections   |                          Q,K,V [B, S, D/8]
  | IColShardedWRow   |                               |
  +-------------------+                          attention
                                                      |
                                                      v
                                              COL-SHARDED
                                            attn_out [B, S, H/8]
                                                      |
                                                      v
                                            +-------------------+
                                            | Row-Parallel      |  reduce_scatter
                                            | O projection      | =============>
                                            | IColShardedWRow   |
                                            +-------------------+
                                                      |
                                                      v
                                                COL-SHARDED
                                            output [B, S, H/8]
                                                      |
                                                residual add
                                          (col-sharded + col-sharded)
                                                      |
                                                      v
  +-------------------+      reduce_scatter
  | Row-Parallel      | =====================>   COL-SHARDED
  | MLP gate/up       |                         activations [B, S, I/8]
  | IColShardedWRow   |                               |
  +-------------------+                               v
                                            +-------------------+
                                            | Row-Parallel      |  reduce_scatter
                                            | MLP down          | =============>
                                            | IColShardedWRow   |
                                            +-------------------+
                                                      |
                                                      v
                                                COL-SHARDED
                                            output [B, S, H/8]
                                                      |
                                                residual add
                                          (col-sharded + col-sharded)
```

### Gemma4 Activation Pattern

Gemma4 uses a different strategy. Fused QKV uses `TTNNLinearIColShardedWAllReduced` (col-sharded input, `reduce_scatter` + `all_gather` producing replicated output). O-projection uses `TTNNLinearIReplicatedWColSharded` (column-parallel, replicated input, no CCL). MLP fused gate-up uses `TTNNLinearIColShardedWAllReduced`. MLP down uses `TTNNLinearIReplicatedWColSharded`. This alternating pattern means activations toggle between col-sharded and replicated at each layer boundary.

### Fused QKV Optimization

The Gemma4 attention module fuses Q, K, and V projections into a single matmul:

```
Before fusion: 3 column-parallel matmuls + 3 all_gathers = 3 matmuls + 6 CCL ops
After fusion:  1 all-reduce matmul (TTNNLinearIColShardedWAllReduced) = 1 matmul + 2 CCL ops
```

This reduces CCL count from 6 operations to 2, saving approximately 67% of the attention block's communication overhead. See [File 4](./04_distributed_modules.md) for the full implementation.

## Sharding Strategy Selection Summary

| Class | Weight Dim | Input State | Output State | CCL Op | Typical Use |
|-------|-----------|-------------|--------------|--------|-------------|
| `TTNNLinearIReplicatedWColSharded` | -1 (output) | Replicated | Col-sharded | None | Q/K/V proj, gate/up proj |
| `TTNNLinearIColShardedWRowSharded` | -2 (input) | Col-sharded | Col-sharded | `reduce_scatter` | O proj, down proj |
| `TTNNLinearIColShardedWAllReduced` | -2 (input) | Col-sharded | Replicated | `reduce_scatter` + `all_gather` | Fused QKV, fused gate-up |
| `TTNNLinearLLamaIColShardedWRowSharded` | -2 (input) | Col-sharded | Col-sharded | `reduce_scatter` | LLaMA row-parallel (bf8) |

---

**Next:** [`03_ccl_operations.md`](./03_ccl_operations.md)
