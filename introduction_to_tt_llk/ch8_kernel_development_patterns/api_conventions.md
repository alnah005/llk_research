# API Conventions

This section documents the naming patterns, parameter conventions, global state, and low-level programming mechanisms that every LLK kernel uses.

## Function Naming Convention

All LLK functions follow the pattern:

```
_llk_<stage>_<operation>_<variant>_<suffix>_()
```

The components are:

| Component   | Values / Examples | Description |
|-------------|-------------------|-------------|
| `stage`     | `unpack`, `math`, `pack` | Which TRISC engine runs this function. |
| `operation` | `AB`, `A`, `AB_matmul`, `eltwise_binary`, `matmul`, `pack` | The core operation being performed. |
| `variant`   | `standard`, `with_dest_reuse`, or omitted | A specialization of the operation. |
| `suffix`    | `hw_configure`, `init`, `uninit`, `dest_init`, `dest_section_done`, or omitted | The lifecycle phase. When omitted, this is the main "execute" call. |

Trailing underscores distinguish the low-level `_llk_` layer from any higher-level wrapper. Examples from the codebase:

```cpp
// Unpack stage
_llk_unpack_hw_configure_<is_fp32_dest_acc_en>(...)
_llk_unpack_AB_init_<BROADCAST_TYPE>(tensor_shape, transpose)
_llk_unpack_AB_<BROADCAST_TYPE>(addr_A, addr_B)
_llk_unpack_AB_matmul_init_<>(...)

// Math stage
_llk_math_hw_configure_<is_fp32_dest_acc_en>(...)
_llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>()
_llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BROADCAST_TYPE, MATH_FIDELITY, REUSE_DEST_TYPE>(...)
_llk_math_eltwise_binary_<...>(tensor_shape, dst_index, clear_fp32_dst_acc)
_llk_math_matmul_init_<MATH_FIDELITY>(...)
_llk_math_matmul_<MATH_FIDELITY>(dst_index, ct_dim, rt_dim)
_llk_math_matmul_uninit_()
_llk_math_wait_for_dest_available_<dest_sync>()
_llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>()

// Pack stage
_llk_pack_hw_configure_<is_fp32_dest_acc_en, untilize>(...)
_llk_pack_init_<untilize, zero_output>(...)
_llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>(...)
_llk_pack_<dest_sync, is_fp32_dest_acc_en, untilize>(tile_index, l1_addr)
_llk_packer_wait_for_math_done_()
_llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>()
```

The lifecycle for each stage always proceeds as: **hw_configure -> init -> (loop of execute calls) -> uninit (if any)**.

## Template Parameter Conventions

LLK functions make heavy use of C++ template parameters for compile-time configuration. These are supplied as `constexpr` values in the auto-generated `build.h` header (see [Chapter 7](../ch7_test_infrastructure/index.md)). The most common template parameters are:

| Parameter | Type | Typical Values | Purpose |
|-----------|------|----------------|---------|
| `is_fp32_dest_acc_en` | `bool` | `true`, `false` | Enables FP32 accumulation in the destination register. Halves the available tile slots. |
| `BroadcastType` | enum | `NONE`, `COL`, `ROW`, `SCALAR` | How Source B is broadcast during binary ops. |
| `EltwiseBinaryType` | enum | `ELWADD`, `ELWSUB`, `ELWMUL` | Which FPU binary operation to perform. |
| `MathFidelity` | enum | `LoFi`, `HiFi2`, `HiFi3`, `HiFi4` | Number of fidelity phases for multiply operations. Higher fidelity = more passes = better precision. |
| `DstSync` | enum | `SyncHalf`, `SyncFull` | Whether math and pack alternate on half or full destination register. |
| `EltwiseBinaryReuseDestType` | enum | `NONE`, `DEST_TO_SRCA`, `DEST_TO_SRCB` | Whether to feed destination back into a source register for accumulation. |

In the test C++ file, these appear as bare identifiers because they are defined in `build.h`:

```cpp
// In build.h (auto-generated):
constexpr auto BROADCAST_TYPE = ckernel::BroadcastType::NONE;
constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::ELWADD;
constexpr auto MATH_FIDELITY = MathFidelity::LoFi;
constexpr auto dest_sync = DstSync::SyncHalf;
constexpr bool is_fp32_dest_acc_en = false;
constexpr auto REUSE_DEST_TYPE = ckernel::EltwiseBinaryReuseDestType::NONE;
```

On the Python side, each template parameter is represented by a dataclass that inherits from `TemplateParameter` and implements `convert_to_cpp()`. For example, in `tests/python_tests/helpers/test_variant_parameters.py`:

```python
@dataclass
class BROADCAST_TYPE(TemplateParameter):
    broadcast_type: BroadcastType

    def convert_to_cpp(self) -> str:
        return f"constexpr auto BROADCAST_TYPE = ckernel::BroadcastType::{self.broadcast_type.value};"
```

## Runtime Parameter Conventions

Values that vary per-invocation (L1 buffer addresses, tile counts, face dimensions) are passed through a `params` struct at runtime. The struct is defined in the auto-generated `params.h` header, and kernel code accesses it as the argument to `run_kernel()`:

```cpp
void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint8_t face_r_dim = static_cast<std::uint8_t>(params.TEST_FACE_R_DIM);
    const std::uint32_t num_total_tiles = params.NUM_TILES_IN_BLOCK * params.NUM_BLOCKS;

    for (std::uint32_t i = 0; i < num_total_tiles; ++i)
    {
        _llk_unpack_AB_<BROADCAST_TYPE>(L1_ADDRESS(params.buffer_A[i]), L1_ADDRESS(params.buffer_B[i]));
    }
}
```

Common runtime parameters include:

- `params.buffer_A[i]`, `params.buffer_B[i]`, `params.buffer_Res[i]` -- L1 addresses of input/output tile buffers, wrapped with `L1_ADDRESS()`.
- `params.NUM_TILES_IN_BLOCK`, `params.NUM_BLOCKS` -- Tiling parameters for the outer loop.
- `params.TEST_FACE_R_DIM`, `params.TEST_FACE_C_DIM` -- Face dimensions.
- `params.num_faces_r_dim_A`, `params.num_faces_c_dim_A` -- Number of faces in each dimension.
- `params.TILE_SIZE_UNPACK_A`, `params.TILE_SIZE_UNPACK_B`, `params.TILE_SIZE_PACK` -- Tile sizes in bytes.
- `params.CT_DIM`, `params.RT_DIM`, `params.KT_DIM` -- Matmul blocking dimensions.
- `params.formats` -- A `FormatConfig` struct containing data format enums for unpack source/destination and pack source/destination.

## Global State Variables

Every LLK test kernel must declare three global variables at file scope:

```cpp
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;
```

These are shared mutable state used by the LLK library internally:

- **`unp_cfg_context`** -- Tracks which unpacker configuration context (0 or 1) is currently active, enabling double-buffered register banks for the unpacker.
- **`pack_sync_tile_dst_ptr`** -- The current destination pointer used by the packer for `SyncHalf`/`SyncFull` synchronization.
- **`math_sync_tile_dst_index`** -- The current destination index used by the math engine for synchronization.

These must be initialized to zero and are mutated by the `_llk_` functions during execution.

## The `ckernel` Namespace

All LLK types and utilities live in the `ckernel` namespace. Key types include:

- `ckernel::TensorShape` -- Describes tile geometry: `{face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim}`.
- `ckernel::BroadcastType`, `ckernel::EltwiseBinaryType`, `ckernel::EltwiseBinaryReuseDestType` -- Enums for operation configuration.
- `ckernel::DstSync`, `ckernel::DstTileShape` -- Enums for destination register management.
- `ckernel::ckernel_template`, `ckernel::ckernel_unpack_template` -- MOP programming classes (see below).
- `ckernel::math::` -- Sub-namespace for math utilities like `set_dst_write_addr()` and `reset_counters()`.

The math section of kernels typically opens with `using namespace ckernel;` for brevity.

## MOP Programming: `ckernel_template`

The **Macro Operation Pipe (MOP)** is a hardware mechanism that executes a sequence of Tensix instructions from a small set of programmable registers, avoiding repeated instruction fetches. The `ckernel_template` class (defined in `tt_llk_wormhole_b0/common/inc/ckernel_template.h`) provides a C++ interface to this mechanism.

### Math MOP: `ckernel_template`

The math MOP implements a double-nested loop:

```
LOOP_OUTER: <outer_loop_count>
  START_OP
  LOOP_INNER: <inner_loop_count>
    LOOP_OP0
    LOOP_OP1
  END_LOOP_INNER
  END_OP0
  END_OP1
END_LOOP_OUTER
```

Construction and usage:

```cpp
// Single inner-loop instruction
ckernel_template tmp(outer_loop_len, inner_loop_len, loop_op);

// Two inner-loop instructions
ckernel_template tmp(outer_loop_len, inner_loop_len, loop_op0, loop_op1);

// Configure special instructions
tmp.set_start_op(start_op0);          // Runs once at the start of each outer iteration
tmp.set_end_ops(end_op0, end_op1);    // Runs at the end of each outer iteration
tmp.set_last_inner_loop_instr(op);    // Replaces LOOP_OP1 on the last inner iteration
tmp.set_last_outer_loop_instr(op);    // Replaces LOOP_OP1 on the last outer iteration

// Program registers, then run
tmp.program();            // Writes to TENSIX_MOP_CFG_BASE registers
ckernel_template::run();  // Executes TTI_MOP(1, 0, 0)
```

The `program()` method writes 9 config registers and `run()` fires the MOP. This separation allows programming once and running multiple times (as seen in the eltwise binary execute path).

### Unpack MOP: `ckernel_unpack_template`

The unpacker has its own MOP template with zmask-based conditional execution:

```
LOOP:
  if (zmask[iteration]):
    UNPACR_A0, A1, A2, A3   (halo mode)
    UNPACR_B
  else:
    SKIP_A
    SKIP_B
```

Factory methods simplify construction:

```cpp
auto tpl = ckernel_unpack_template::lA(A_instr, skipA_instr);   // Source A only
auto tpl = ckernel_unpack_template::lBA(A_instr, skipA, B_instr, skipB); // Both sources
auto tpl = ckernel_unpack_template::lBhA(halo_mask);            // Halo mode with B

tpl.program_and_run(count, zmask);
```

## Address Modifier System (`addr_mod_t`)

Address modifiers control how source and destination register pointers advance after each FPU or MVMUL instruction. They are defined as `addr_mod_t` structs (in `tt_llk_wormhole_b0/common/inc/ckernel_addrmod.h`) and written to hardware slots `ADDR_MOD_0` through `ADDR_MOD_7`.

Each `addr_mod_t` contains sub-fields:

```cpp
struct addr_mod_t {
    struct addr_mod_src_t  { uint8_t incr, clr, cr; }  srca, srcb;
    struct addr_mod_dest_t { int16_t incr; uint8_t clr, cr, c_to_cr; } dest;
    struct addr_mod_fidelity_t { uint8_t incr, clr; } fidelity;
    struct addr_mod_bias_t     { uint8_t incr, clr; } bias;
};
```

- **`incr`** -- How many rows to advance the pointer after the instruction executes.
- **`clr`** -- Reset the pointer to zero.
- **`cr`** -- "Carry" -- add a carry amount (typically 16 for face transitions).
- **`c_to_cr`** -- Copy the counter value to the carry register.
- **`fidelity.incr`** -- Advance the fidelity phase counter (used in HiFi multiply).
- **`bias.incr`** -- Advance the bias counter.

Usage pattern:

```cpp
addr_mod_t {
    .srca = {.incr = MAX_FPU_ROWS},
    .srcb = {.incr = MAX_FPU_ROWS},
    .dest = {.incr = MAX_FPU_ROWS},
}.set(ADDR_MOD_0);
```

Each MOP instruction references an `ADDR_MOD_N` slot. The hardware applies the specified address modifications after executing the instruction, so the next instruction sees updated pointers. This mechanism eliminates explicit pointer arithmetic instructions from the inner loop.

## LLTT: Instruction Replay Buffers

LLTT (Low-Level Trace/replay) provides a mechanism for recording sequences of Tensix instructions into replay buffers and replaying them. This is used in kernels where the instruction sequence is too complex or irregular for the MOP template alone.

The API consists of two key functions:

- **`lltt::record(offset, length)`** -- Begin recording the next `length` instructions into the replay buffer starting at `offset`. The instructions are executed as they are recorded (unless the `lltt::NoExec` tag is used).
- **`lltt::record<lltt::NoExec>(offset, length)`** -- Record without executing.
- **`lltt::replay_insn(offset, length)`** -- Returns a Tensix instruction word that, when executed (e.g., as a MOP loop body), replays the `length` instructions starting at `offset` from the replay buffer.

The matmul kernel uses LLTT extensively. The initialization phase records a sequence of `TTI_MVMUL` instructions into the replay buffer, then the MOP loop body is simply `lltt::replay_insn(offset, len)`:

```cpp
// During init: record 16 MVMUL instructions into replay buffer
lltt::record(ckernel::math::replay_buf_offset, replay_buf_len);
TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // instruction 0
TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // instruction 1
// ... more MVMUL instructions ...

// MOP body replays the recorded sequence
ckernel_template tmp(1, inner_loops, lltt::replay_insn(ckernel::math::replay_buf_offset, replay_buf_len));
tmp.program();
```

This approach allows a single MOP iteration to execute a long sequence of instructions with varying address modifiers -- something the two-instruction MOP body could not express directly.

## LLK Assert System

The `LLK_ASSERT()` macro (defined in `common/llk_assert.h`) provides compile-time-optional runtime assertions:

```cpp
LLK_ASSERT(condition, "message");
```

When `ENABLE_LLK_ASSERT` is defined:

- **In the LLK test infrastructure** (`ENV_LLK_INFRA`): triggers a RISC-V `ebreak` instruction on failure, halting the core for debugging.
- **In tt-metal**: delegates to the tt-metal `ASSERT()` macro.

When `ENABLE_LLK_ASSERT` is not defined, the condition is compiled but not executed (`(void)sizeof((condition))`), ensuring type-checking without runtime cost.

Typical usage in kernel code:

```cpp
LLK_ASSERT(
    (static_cast<std::uint32_t>(tile) < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
    "Block tile index exceeds maximum destination tiles");
```

---

**Next:** [`eltwise_binary_walkthrough.md`](./eltwise_binary_walkthrough.md)
