# Chapter 4: The Tile Processing Pattern

Every TT-Metal compute kernel follows a distinctive tile processing protocol built on four pairs of coordinating calls. This chapter evaluates the ergonomics, predictability, and error-proneness of that protocol.

## The Canonical Pattern

The standard tile processing cycle coordinates two hardware threads -- MATH and PACK -- through a four-phase handshake over the shared DST (destination) register file:

```
    MATH thread                          PACK thread
    -----------                          -----------
    tile_regs_acquire()                      |
        |                                    |
    [compute into DST]                       |
        |                                    |
    tile_regs_commit()  ---- handoff ---> tile_regs_wait()
                                             |
                                        [pack from DST to CB]
                                             |
                                        tile_regs_release()
```

Surrounding the DST handshake, circular buffer (CB) calls manage data flow between the dataflow and compute engines:

```
cb_wait_front(cb_in, N)       -- wait for N input tiles from reader
cb_reserve_back(cb_out, N)    -- reserve space in output CB

tile_regs_acquire()           -- MATH acquires DST
  <math operations>           -- unpack from CB, compute into DST
tile_regs_commit()            -- MATH releases DST to PACK

tile_regs_wait()              -- PACK acquires DST
  pack_tile(dst_idx, cb_out)  -- pack from DST into output CB
tile_regs_release()           -- PACK releases DST back to MATH

cb_pop_front(cb_in, N)        -- free input CB slots
cb_push_back(cb_out, N)       -- signal output tiles to writer
```

This entire sequence is defined in [`tt_metal/hw/inc/api/compute/reg_api.h`](../../tt-metal/tt_metal/hw/inc/api/compute/reg_api.h).

## How Real Kernels Implement It

Nearly every compute kernel in the codebase follows this skeleton. The pattern is visible across all operation types:

- **Eltwise binary** ([`tt_metal/kernels/compute/eltwise_binary.cpp`](../../tt-metal/tt_metal/kernels/compute/eltwise_binary.cpp)) -- the simplest expression of the pattern
- **Layernorm** ([`layernorm_sharded.cpp`](../../tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded.cpp)) -- repeats the pattern many times for reduce, subtract, square, normalize, gamma, and beta stages
- **Groupnorm** ([`groupnorm_sharded_v2.cpp`](../../tt-metal/ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm_sharded_v2.cpp)) -- same four-phase cycle applied inside nested batch/group loops
- **BMM** ([`bmm_large_block_zm_fused_bias_activation.cpp`](../../tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp)) -- the pattern drives matmul accumulation, spill/reload, and fused bias
- **SDPA** ([`compute_common.hpp`](../../tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp)) -- uses `acquire_dst`/`release_dst` (the older, now-deprecated equivalent)

## Chapter Contents

- [**Pattern Ergonomics**](./pattern_ergonomics.md) -- Why the pattern works well: explicit pipeline, predictable structure, helper aliases, and flexibility.
- [**Common Mistakes**](./common_mistakes.md) -- Where the pattern fails developers: silent hangs, missing guards, CB synchronization bugs, and manual index tracking.
