# 02 -- TT-Blaze Execution Pattern

This section traces the universal execution pattern shared by every operation in the DeepSeek V3 B1 codebase -- from the simplest single-core matmul to the most complex 130-core fused MoE pipeline. It walks through the five-step pipeline from Python descriptors to hardware execution, provides a detailed `Matmul.op()` walkthrough as the canonical minimal example, describes the `DeepSeekV3` class interface and decode lifecycle, and demonstrates how the pattern scales to complex fused operations.


## 2.1 The Execution Pipeline: From Descriptors to Hardware

The DeepSeek V3 B1 codebase does not use `ttnn` high-level ops (like `ttnn.matmul` or `ttnn.layer_norm`). Instead, every operation is constructed from first principles using a five-step pipeline:

```
CB Descriptors  -->  UnifiedKernelDescriptor  -->  get_kernel_descriptors()  -->  ProgramDescriptor  -->  ttnn.generic_op
```


### Step 1: Circular Buffer (CB) Descriptors

Circular buffers are the L1 SRAM communication channels between RISC processors on a Tensix core. Each CB has:

- A **buffer index** (0--63, managed by `CircularBufferIdManager`)
- A **data format** (bfloat16, bfloat8\_b, bfloat4\_b, uint32, etc.)
- A **tile descriptor** (dimensions like $32 \times 32$, $1 \times 32$, $16 \times 16$, etc.)
- A **page size** (bytes per tile in the given format)
- A **total size** (bytes of L1 allocated to the buffer)
- A **core range** (which cores own this CB)

There are two creation patterns:

**Tensor-backed CBs** point to an already-allocated sharded tensor's L1 memory:

```python
in0_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(in0_cb, input_a)
```

**Manual CBs** allocate fresh L1 space:

```python
mcast_dst_format = ttnn.CBFormatDescriptor(
    buffer_index=ctx.mcast_dst_cb,
    data_format=ctx.data_format,
    page_size=ctx.input_tile_size,
    tile=input_tile_desc,
)
mcast_dst_cb_descriptor = ttnn.CBDescriptor(
    total_size=ctx.K_down_tiles * ctx.input_tile_size,
    core_ranges=ctx.all_cores,
    format_descriptors=[mcast_dst_format],
)
```

#### CB Descriptors from Overlapped Tensors

The B1 weight system uses "overlapped tensors" -- multiple logical weight matrices fused into a single physical buffer. The `cb_descriptor_from_overlapped_tensor` function in `circular_buffer_utils.py` (lines 99--129) creates a CB descriptor from a sub-view of a fused tensor:

```python
def cb_descriptor_from_overlapped_tensor(
    cb_index: int,
    overlapped: OverlappedTensor,
    fused_tensor_device: ttnn.Tensor,
) -> ttnn.CBDescriptor:
    cb_desc = ttnn.cb_descriptor_from_sharded_tensor(
        cb_index, fused_tensor_device,
        address_offset=overlapped.byte_offset,
        total_size=overlapped.total_size,
        core_ranges=overlapped.core_range_set,
    )
    tile = ttnn.Tile(overlapped.tile_shape)
    cb_desc.format_descriptors = [
        ttnn.CBFormatDescriptor(
            buffer_index=cb_index,
            data_format=overlapped.dtype,
            page_size=tile.get_tile_size(overlapped.dtype),
            tile=ttnn.TileDescriptor(tile),
        )
    ]
    return cb_desc
```

The `address_offset` parameter shifts the CB's base address within the fused buffer, so each sub-tensor can be accessed at a known offset without any runtime indirection.

#### The CircularBufferIdManager

When fusing many ops into a single program, CB ID allocation becomes non-trivial: different stages may reuse the same CB slot if their data formats and tile shapes match, but within a single stage each CB ID must be unique. The `CircularBufferIdManager` (in `circular_buffer_utils.py`) solves this with a context-based allocator:

```python
manager = CircularBufferIdManager()
ctx_matmul = manager.create_context()
ctx_reduce = manager.create_context()

# Within each context, IDs are unique; across contexts, they can be reused
in0_id = ctx_matmul.get_cb_id(data_format=ttnn.bfloat16, tile=my_tile)
in1_id = ctx_reduce.get_cb_id(data_format=ttnn.bfloat16, tile=my_tile)
# in0_id == in1_id is possible (same format, different context)
```

The maximum is 64 CB IDs per core (hardware constraint enforced by `NUM_CIRCULAR_BUFFERS = 64`).

#### CB Reconfig Tensors

In deeply fused ops, the CB configuration left by a preceding stage may not match what the next stage needs. Rather than inserting barrier + reconfigure calls, the B1 system pre-computes a **reconfig tensor** -- an L1-sharded tensor containing the complete CB configuration for every core. The fused kernel reads this tensor at startup and calls `setup_local_cb_read_write_interfaces()` to reconfigure all CBs in one shot.

The layout per core is 264 uint32 words (1,056 bytes):

| Offset | Content |
|---|---|
| Words 0--255 | 64 CB configs, 4 words each: `[addr, total_size, num_pages, page_size]` |
| Word 256 | `cb_mask_low` (bits 0--31: which CBs 0--31 are active) |
| Word 257 | `cb_mask_high` (bits 32--63: which CBs 32--63 are active) |
| Words 258--259 | Cross-RISC sync semaphores (initialized to 0) |
| Words 260--263 | Reserved (zeros) |

This is built by `build_cb_reconfig_tensor()` in `circular_buffer_utils.py` and used heavily by `MoeOp`, where the shared-expert phase must reconfigure CBs left by the routed-expert phase.


### Step 2: UnifiedKernelDescriptor

The `UnifiedKernelDescriptor` (defined in `unified_kernel_descriptor.py`) is the central abstraction that maps a single unified `.cpp` kernel source to three RISC processors. The key insight is that one source file contains `#if defined(COMPILE_FOR_NCRISC)`, `#if defined(COMPILE_FOR_BRISC)`, and `#if defined(COMPILE_FOR_TRISC)` guards, and the descriptor produces separate `KernelDescriptor` objects for each.

```python
@dataclass
class UnifiedKernelDescriptor:
    kernel_source: str                       # Path to unified .cpp file
    core_ranges: ttnn.CoreRangeSet           # Where the kernel runs
    ncrisc_compile_time_args: list           # Reader (NCRISC) args
    brisc_compile_time_args: list            # Writer (BRISC) args
    trisc_compile_time_args: list            # Compute (TRISC) args
    ncrisc_named_compile_time_args: list     # Named args for NCRISC
    brisc_named_compile_time_args: list      # Named args for BRISC
    trisc_named_compile_time_args: list      # Named args for TRISC
    unified_compile_time_core_descriptors: list  # Per-core-range overrides
    per_core_compile_time_descriptors: list   # Per-individual-core overrides
    per_core_runtime_args_descriptor: Optional[PerCoreRuntimeArgsDescriptor]
    trisc_compute_config: Optional[ttnn.ComputeConfigDescriptor]
    defines: list                            # Preprocessor defines
    noc_mode: ttnn.NOC_MODE                  # NOC addressing mode
```

The RISC processor mapping is:

| Processor | Role | NOC | Example Responsibility |
|---|---|---|---|
| **NCRISC** (RISCV\_1) | Reader | NOC 0 | Setup sharded buffers, receive mcast data, read from DRAM |
| **BRISC** (RISCV\_0) | Writer | NOC 1 | Send mcast data, gather results, write to output buffers |
| **TRISC** | Compute | N/A | Matmul, RMSNorm, SiLU, element-wise ops, argmax |

The `UnifiedCompileTimeCoreDescriptor` is the mechanism for role assignment:

```python
UnifiedCompileTimeCoreDescriptor(
    named_compile_time_arg="is_mcast_sender_core",
    core_range=ctx.mcast_gather_core_grid,    # Only core (12,9)
    value=1,
    other_value=0,                             # All other cores get 0
)
```

Because these are compile-time args, the kernel uses `constexpr` and the compiler eliminates dead code paths -- a core compiled with `is_mcast_sender_core=0` has no sender code in its binary.


### Step 3: `get_kernel_descriptors()`

Calling `get_kernel_descriptors()` on a `UnifiedKernelDescriptor` produces the actual `KernelDescriptor` objects that map to hardware. Two paths exist:

**Simple path** (no per-core overrides): Returns exactly 3 `KernelDescriptor` objects (one per RISC), all covering the same `core_ranges`.

**Split path** (with `unified_compile_time_core_descriptors` or `per_core_compile_time_descriptors`): The method enumerates every core, computes its complete set of named compile-time args, groups cores by unique arg combinations, and produces $3 \times G$ `KernelDescriptor` objects where $G$ is the number of distinct groups. This enables compile-time dead-code elimination: a core compiled with `is_mcast_sender_core=1` compiles entirely different code from a core with `is_mcast_sender_core=0`, despite sharing the same source file.

The `UnifiedKernelResult` contains both the kernel list and a list of `KernelGroup` objects that map each group back to its kernel indices. This lets callers look up kernel indices by role:

```python
result = unified_kernel.get_kernel_descriptors()
sender_group = result.get_group_by_arg("is_sender", 1)
# sender_group.ncrisc_kernel_index, sender_group.brisc_kernel_index, ...
```


### Step 4: ProgramDescriptor

A `ProgramDescriptor` bundles everything needed for a single device-side program launch:

```python
program_descriptor = ttnn.ProgramDescriptor(
    kernels=unified_kernel.get_kernel_descriptors().kernels,
    cbs=[in0_cb_descriptor, in1_cb_descriptor, out_cb_descriptor],
    semaphores=semaphore_descriptors,  # Optional
)
```

For multi-device operations, a `MeshProgramDescriptor` wraps per-device `ProgramDescriptor` objects:

```python
mesh_program_descriptor = ttnn.MeshProgramDescriptor()
for row in range(mesh_rows):
    for col in range(mesh_cols):
        coord = ttnn.MeshCoordinate(row, col)
        # ... build per-device program ...
        mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program
```

This is used by `LMHeadSampling.op()` and `PostSDPA` where each device in a $4 \times 2$ mesh needs device-specific CCL routing, sender/receiver roles, and fabric connection setup.


### Step 5: `ttnn.generic_op`

The final step dispatches the program to hardware:

```python
io_tensors = [input_a, input_b, output_tensor]
output = ttnn.generic_op(io_tensors, program_descriptor)
```

`ttnn.generic_op` is the lowest-level Python API for launching custom programs. It resolves tensor addresses on device, configures CBs on the target cores, loads and launches the compiled kernels, and returns when all cores have completed. For mesh operations, the same call dispatches across all devices:

```python
result = ttnn.generic_op(io_tensors, mesh_program_descriptor)
```


## 2.2 Matmul.op() Walkthrough

The `Matmul` class in `micro_ops/matmul/op.py` is the cleanest example of the execution pattern. It implements a single-core matmul optimized for $[1, K] \times [K, N]$ with $N \leq 4$ tiles (128 elements). Walking through `Matmul.op()` step by step:

**Step 1 -- Shape validation and dimension extraction:**

```python
a_shape = input_a.shape
a_shard_shape = input_a.memory_config().shard_spec.shape
all_cores = input_a.memory_config().shard_spec.grid
num_tiles_k = a_shape[1] // in0_tile.tile_shape[1]
```

The method extracts the core grid directly from the input tensor's shard spec. It verifies that $M$ is exactly one tile height (the $B = 1$ constraint): `assert a_shard_shape[0] // in0_tile.tile_shape[0] == 1`. The inner-dimension tile count (`num_tiles_k`) becomes a compile-time argument.

**Step 2 -- CB descriptors (3 total):**

```python
in0_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(0, input_a)   # CB 0: Input A
in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(1, input_b)   # CB 1: Input B
out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(2, output_tensor)  # CB 2: Output
```

All three are tensor-backed -- no manual allocation needed.

**Step 3 -- Named compile-time args per RISC:**

```python
ncrisc_named_compile_time_args = [
    ("matmul_in0", 0),              # CB index for input A
    ("matmul_in1", 1),              # CB index for input B
    ("matmul_k_num_tiles", num_tiles_k),
    ("matmul_out_w", out_w),
]

trisc_named_compile_time_args = [
    ("matmul_in0", 0),
    ("matmul_in1", 1),
    ("matmul_out", 2),              # CB index for output
    ("matmul_k_num_tiles", num_tiles_k),
    ("matmul_out_w", out_w),
    ("matmul_transpose", transpose),
    ("matmul_fused_activation", fused_activation_val),  # 0=none, 1=sigmoid, 2=silu
]
```

NCRISC (reader) needs to know which CBs to set up and the loop bounds. TRISC (compute) additionally needs the output CB, transpose flag, and fused activation. BRISC (writer) gets no named args for this simple op.

**Step 4 -- UnifiedKernelDescriptor:**

```python
unified_kernel = UnifiedKernelDescriptor(
    kernel_source="models/demos/deepseek_v3_b1/micro_ops/matmul/kernels/matmul_kernel.cpp",
    core_ranges=all_cores,
    ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
    brisc_named_compile_time_args=[],
    trisc_named_compile_time_args=trisc_named_compile_time_args,
    trisc_compute_config=ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        dst_full_sync_en=fp32_dest_acc_en,
    ),
    unified_compile_time_core_descriptors=[
        UnifiedCompileTimeCoreDescriptor(
            named_compile_time_arg="is_active_core",
            core_range=all_cores,
            value=1,
            other_value=0,
        ),
    ],
)
```

Even this simple op uses a `UnifiedCompileTimeCoreDescriptor` to mark active cores (though here all cores in the range are active). The `trisc_compute_config` controls math fidelity: `LoFi` is used throughout the B1 implementation for maximum throughput. The `is_active_core` pattern is used pervasively in fused ops where some cores in a larger grid should skip the matmul.

**Step 5 -- ProgramDescriptor and dispatch:**

```python
program_descriptor = ttnn.ProgramDescriptor(
    kernels=unified_kernel.get_kernel_descriptors().kernels,
    cbs=[in0_cb_descriptor, in1_cb_descriptor, out_cb_descriptor],
)

io_tensors = [input_a, input_b, output_tensor]
output = ttnn.generic_op(io_tensors, program_descriptor)
```

The entire operation -- from shape validation to hardware execution -- fits in one `op()` method. No explicit memory management, no kernel compilation orchestration. The descriptor system handles everything.


## 2.3 The DeepSeekV3 Class Interface

The `DeepSeekV3` class in `model.py` (line 72) is the host-side entry point. It manages the socket lifecycle and position tracking for a single decode session.

### Constructor: Socket Setup

```python
class DeepSeekV3:
    def __init__(
        self,
        h2d_socket_prefill: ttnn.H2DSocket,   # DEVICE_PULL mode
        h2d_socket_decode: ttnn.H2DSocket,     # HOST_PUSH mode
        d2h_socket: ttnn.D2HSocket,            # Shared by both phases
        batch_size: int = 1,                    # Must be 1
        loopback_mode: bool = False,
    ) -> None:
```

Two separate H2D sockets serve prefill and decode because they use different transfer modes:

- **DEVICE\_PULL** (`h2d_socket_prefill`): The device-side NCRISC kernel calls `socket_wait_for_pages()` to pull data from the host FIFO. The device controls timing. This is preferred during prefill because the device is also managing KV cache updates between steps, and it processes at its own pace.

- **HOST\_PUSH** (`h2d_socket_decode`): The host calls `write_tensor()` and the data arrives in the device's L1 buffer via PCIe DMA without the device requesting it. Lower latency for the decode hot loop where the host already has the next token ready.

The constructor enforces these modes with runtime checks (lines 104--109):

```python
if h2d_socket_prefill.get_h2d_mode() != ttnn.H2DMode.DEVICE_PULL:
    raise ValueError(...)
if h2d_socket_decode.get_h2d_mode() != ttnn.H2DMode.HOST_PUSH:
    raise ValueError(...)
```

The `HostInterface` objects wrapping these sockets are created with FIFO sizes aligned to 64-byte PCIe boundaries:

```python
self._tensor_size_bytes: int = align_up(batch_size * TOKEN_ID_BYTES, PCIE_PAGE_ALIGNMENT_BYTES)
# For B=1: align_up(4, 64) = 64 bytes
```

### Lifecycle State Machine

The decode lifecycle follows a strict state machine:

```
     start()          prefill()           _switch_to_decode()        decode_step()
       |                  |                       |                       |
  [IDLE] -----> [PREFILL_ACTIVE] -----> [DECODE_ACTIVE] -----> [DECODE_ACTIVE] -----> ...
                                              ^                                   |
                                              +-----------------------------------+
```

1. **`start()`** (line 146) -- Launches the prefill `HostInterface` program on device. The DEVICE\_PULL H2D receiver kernel begins polling for data. Sets `_prefill_active = True`.

2. **`prefill(prompt_tokens)`** (line 164) -- Processes the prompt token-by-token using the "prefill-by-decode" strategy. For a prompt of $S$ tokens, the host writes each token ID and reads back the output $S$ times. Only the last output matters (it contains the logits for sampling the first generated token); outputs for positions 0 through $S - 2$ are discarded. The position counter increments with each step.

    ```python
    for token in prompt_tokens:
        self.h2d_socket_prefill.write_tensor(token)
        self.d2h_socket.read_tensor(self._output_buffer)
        self._position += 1
    ```

    This "prefill-by-decode" approach means the device runs the same decode kernel for both prefill and generation. There is no separate prefill kernel with parallel token processing -- each prompt token is processed sequentially, exactly as in autoregressive decode. This simplifies the device pipeline at the cost of prefill throughput.

3. **`_switch_to_decode()`** (line 152) -- Called automatically on the first `decode_step()`. Terminates the prefill program (which drains the DEVICE\_PULL socket and synchronizes devices), launches the decode program with HOST\_PUSH H2D. One-time transition; subsequent calls are no-ops.

4. **`decode_step(input_tensor)`** (line 192) -- Single autoregressive step: pad the input to PCIe alignment, write via the HOST\_PUSH socket, read the response. Each call increments `_position`.

5. **`stop()`** (line 222) -- Clean shutdown. Terminates whichever `HostInterface` is active by setting the termination semaphore to 1 and synchronizing devices. The device-side kernels poll this semaphore during their blocking socket/CB waits (the `HostInterface.terminate` method at `micro_ops/host_io/op.py`, line 399 calls `ttnn.reset_global_semaphore_value(self.termination_semaphore, 1)`). Both kernels exit their loops within approximately 1,000 device cycles.

### The HostInterface as a TT-Blaze Op

The `HostInterface` class (`micro_ops/host_io/op.py`, line 39) is itself a TT-Blaze op. Its `run()` method (line 290) follows the same five-step pattern:

1. Creates H2D and D2H kernel descriptors via `_create_h2d_kernel()` and `_create_d2h_kernel()`.
2. Creates CB descriptors via `_create_cb_descriptors()`.
3. Packs them into a `ProgramDescriptor` (or two, if H2D and D2H are on different devices).
4. Wraps into a `MeshProgramDescriptor` for multi-device dispatch.
5. Calls `ttnn.generic_op(io_tensors, mesh_program_descriptor)`.

In loopback mode, the H2D receiver and D2H sender communicate through a shared circular buffer on the same core (CB index 0). In socket mode, they connect to downstream/upstream cores via D2D socket pairs.

### The Runner Protocol

The `demo/runner.py` defines a `ModelLike` protocol that `DeepSeekV3` implements:

```python
class ModelLike(Protocol):
    def start(self) -> None: ...
    def prefill(self, prompt_tokens: list[Any]) -> Any: ...
    def decode_step(self, input_tensor: Any) -> Any: ...
    def stop(self) -> None: ...
```

The `run_generation()` function orchestrates the full generation loop:

```python
model.start()
last_prefill_output = model.prefill(prefill_inputs)
next_token_id = extract_token_id(last_prefill_output)

for _ in range(max_new_tokens):
    decode_input = make_input_tensor(next_token_id)
    decode_output = model.decode_step(decode_input)
    next_token_id = extract_token_id(decode_output)
```

The `TokenCodec` class in `demo/runtime.py` handles the encoding: `make_input()` creates a $(B, 1)$ int32 tensor padded to 64-byte alignment; `extract_token_id()` pulls the scalar token ID back from the device output tensor.

This clean separation between the model interface and the generation loop allows the same runner to work with both the loopback mock (for testing) and the real multi-host pipeline.


## 2.4 Scaling the Pattern: From Matmul to SharedExpertOp

The jump from `Matmul.op()` (3 CBs, 1 role) to `SharedExpertOp.op()` (15 CBs, 8 roles) illustrates how the same five-step pattern scales to complex fused operations without changing the fundamental structure.

`SharedExpertOp` computes:

$$
\text{SiLU}(\mathbf{x} \cdot W_\text{gate}) \odot (\mathbf{x} \cdot W_\text{up}) \cdot W_\text{down} + \mathbf{bias}
$$

It runs on 130 cores of the $13 \times 10$ device grid:

- **Core (12,9)**: Mcast sender, gather receiver, gated-reduce core
- **64 A cores**: Gate matmul (columns 0--3, 7--9 in rows 0--3; columns 0--2, 7--9 in rows 4--9)
- **64 B cores**: Up matmul (remaining compute columns)
- **112 matmul cores**: Down-projection matmul (subset excluding DRAM workers and column 12)

The 8 `UnifiedCompileTimeCoreDescriptor` entries create distinct compile-time roles:

```python
core_descs = [
    UnifiedCompileTimeCoreDescriptor("is_gate_compute_core", a_compute_grid, 1, 0),
    UnifiedCompileTimeCoreDescriptor("is_up_compute_core", b_compute_grid, 1, 0),
    UnifiedCompileTimeCoreDescriptor("is_gated_reduce_core", mcast_gather_core_grid, 1, 0),
    UnifiedCompileTimeCoreDescriptor("is_mcast_sender_core", mcast_gather_core_grid, 1, 0),
    UnifiedCompileTimeCoreDescriptor("is_mcast_receiver_core", mcast_receiver_grid, 1, 0),
    UnifiedCompileTimeCoreDescriptor("is_matmul_core", matmul_core_grid, 1, 0),
    UnifiedCompileTimeCoreDescriptor("is_gather_receiver_core", mcast_gather_core_grid, 1, 0),
    UnifiedCompileTimeCoreDescriptor("gather_use_per_core_sender_idx", matmul_core_grid, 1, 0),
]
```

Despite this complexity, the final dispatch is identical to `Matmul.op()`:

```python
program_descriptor = ttnn.ProgramDescriptor(
    kernels=unified_kernel.get_kernel_descriptors().kernels,
    cbs=cb_descriptors,
    semaphores=semaphore_descriptors,
)
ttnn.generic_op(io_tensors, program_descriptor)
```

The pattern is fractal: `MoeOp` composes `MoeRoutedExpertOp` and `MoeSharedExpertOp` contexts into an even larger fused program that orchestrates routed experts, gating, and shared-expert computation across all 130 cores with 17 semaphores (`MoeSem.NUM_SEMAPHORES = 17`) and CB-reconfig tensors for sharing CB IDs across sequential phases.

| Level | Example | CBs | Roles | Semaphores |
|---|---|---|---|---|
| Micro-op | `Matmul.op()` | 3 | 1 | 0 |
| Fused op | `SharedExpertOp.op()` | 15 | 8 | 4 |
| Compound fused | `MoeOp.op()` | 15+ (reused via reconfig) | 8+ | 17 |

---

**Next:** [`03_layer_taxonomy_and_decode_overview.md`](./03_layer_taxonomy_and_decode_overview.md)
