# Agent B Review — Chapter 2

## Pass 1

**1. Section 2.3.6: `cb_addr_shift` is presented as a universal constant of 4, but it is RISC-dependent (material factual error)**

The chapter states: "The config tensor stores addresses and sizes in *bytes*, but the `LocalCBInterface` fields use *L1-alignment units* (each unit is 16 bytes, so `cb_addr_shift` is 4). The right-shift `>> cb_addr_shift` converts from bytes to alignment units."

This is incorrect as a blanket statement. In the actual firmware (`tt_metal/hw/inc/internal/circular_buffer_interface.h`, lines 117-121):

```cpp
#if defined(COMPILE_FOR_TRISC)
constexpr uint32_t cb_addr_shift = CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT;  // == 4
#else
constexpr uint32_t cb_addr_shift = 0;
#endif
```

For NCRISC and BRISC, `cb_addr_shift` is 0 -- the right-shift is a no-op and addresses remain in bytes. Only TRISC uses 16-byte alignment units. Since `reconfig_cbs_for_mask` is compiled separately per RISC, the shift behavior differs across compilations. The worked example in section 2.3.9 compounds this by showing `fifo_addr = 0x1A000 >> 4 = 0x1A00` as though it applies universally. A reader implementing or debugging custom CB reconfig code would compute wrong address values for the NCRISC and BRISC `LocalCBInterface` fields.

**2. Section 2.2.8: Fused MoE kernel claimed to include "13 micro-op headers" but actually includes 12 (factual error)**

The chapter says the fused MoE kernel "includes 13 micro-op headers." The actual file (`fused_ops/moe/moe_kernel.cpp`) contains 14 `#include` directives: 2 infrastructure headers (`kernel_op_api.hpp`, `kernel_utils.hpp`) and 12 operation headers (mcast, matmul, moe_gather, deepseek_moe_gate, rmsnorm, dram_streaming_matmul, eltwise_mul, eltwise_add, kn_sliced_matmul, gated_reduce, residual_add, reduce_to_one_b1). The count is 12 micro-op headers, not 13. A reader trying to identify which 13 headers are included would come up one short.

---

### Agent A Change Log

**Issue 1 fix (cb_addr_shift is RISC-dependent) -- `03_circular_buffer_management.md`:**
- Section 2.3.6: Replaced the paragraph claiming `cb_addr_shift` is always 4 with an explanation that the shift is RISC-dependent. Added the firmware code snippet from `circular_buffer_interface.h` showing TRISC uses `cb_addr_shift = 4` while NCRISC/BRISC use `cb_addr_shift = 0`. Noted that `reconfig_cbs_for_mask` compiles separately per RISC, producing different shift behavior.
- Section 2.3.9 (worked example, step 5): Expanded the single set of converted values into two subsections -- one for TRISC (showing the >> 4 conversion to alignment units) and one for NCRISC/BRISC (showing the >> 0 no-op that keeps byte values). Updated step 6 to clarify the shown values are TRISC-specific.

**Issue 2 fix (MoE header count is 12, not 13) -- `02_kernel_op_api_and_unified_headers.md`:**
- Section 2.2.8, paragraph 1: Changed "includes 13 micro-op headers" to "includes 12 micro-op headers (plus 2 infrastructure headers)".
- Section 2.2.8, dead-code elimination subsection: Changed "the full 13-op pipeline" to "the full 12-op pipeline".
- Section 2.2.8, dead-code elimination subsection: Changed "including 13 different operation implementations" to "including 12 different operation implementations".

---

## Pass 2

**1. Section 2.2.3 Category 3 table: `moe_gather.hpp` NCRISC/BRISC roles are swapped (factual error)**

The table in Category 3 (Sender/Receiver Pairs) lists `moe_gather.hpp` as NCRISC=Sender, BRISC=Receiver. The actual code in `moe_gather.hpp` defines:

```cpp
using RTArgs = unified_kernels::SelectByRISCV<ReceiverArgs, SenderArgs, ComputeArgs>;
```

Since `SelectByRISCV<R, W, C>` maps `R` to NCRISC and `W` to BRISC, this means NCRISC=Receiver and BRISC=Sender -- the same pattern as `mcast.hpp`, not `gather.hpp`. This is confirmed by `moe_kernel.cpp` where the NCRISC section constructs `MoeGather::ReceiverArgs` and the BRISC section constructs `MoeGather::SenderArgs`. The table entry should read NCRISC=Receiver, BRISC=Sender. A reader using this table to understand which NOC each role uses would wire up the wrong NOC assignments for MoeGather operations.

**No other issues found -- chapter approved with the above correction.**

### Agent A Change Log — Pass 2

**Issue 1 fix (moe_gather.hpp NCRISC/BRISC roles swapped) -- `02_kernel_op_api_and_unified_headers.md`:**
- Category 3 table: Changed `moe_gather.hpp` row from "Sender | Receiver" to "Receiver | Sender".
- Updated the code comment below the table to note that `moe_gather.hpp` follows the mcast pattern (NCRISC=Receiver, BRISC=Sender).
