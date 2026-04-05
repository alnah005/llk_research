# Data Flow

This section traces the path that data follows through the Tensix Core's compute pipeline. Every LLK operation -- unpack, math, or pack -- moves or transforms data along this path.

## The Core Compute Pipeline

The primary data flow through the Tensix Engine follows a linear path:

```
                          Tensix Core
  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │   L1 SRAM                                            │
  │   ┌──────────────────────────────────────────────┐   │
  │   │  Input Tiles          Output Tiles           │   │
  │   └──────┬───────────────────────▲───────────────┘   │
  │          │                       │                   │
  │          │ Unpack                │ Pack              │
  │          │ (TRISC0)              │ (TRISC2)          │
  │          ▼                       │                   │
  │   ┌──────────────┐    ┌─────────┴────────┐          │
  │   │  Source A     │    │  Destination     │          │
  │   │  Register     │───>│  Register        │          │
  │   ├──────────────┤    │                  │          │
  │   │  Source B     │───>│  (FPU output /   │          │
  │   │  Register     │    │   SFPU I/O)     │          │
  │   └──────────────┘    └──────────────────┘          │
  │          │                  ▲    │  ▲                │
  │          │       FPU        │    │  │                │
  │          └──────────────────┘    │  │                │
  │                                  │  │                │
  │                            SFPU  │  │  SFPU          │
  │                           (read) │  │ (write)        │
  │                                  └──┘                │
  │                          (TRISC1)                    │
  └──────────────────────────────────────────────────────┘
```

The pipeline stages are:

1. **L1 to Source Registers (Unpack)** -- TRISC0 configures the unpackers to DMA tile data from L1 memory into the Source A and Source B register files. Hardware gaskets convert the data format during transfer.
2. **Source Registers to Destination Register (FPU)** -- TRISC1 issues FPU instructions that read operands from Source A and Source B, perform computation (dot product, element-wise add/sub/mul), and write results to the Destination register.
3. **Destination Register to L1 (Pack)** -- TRISC2 configures the packer to DMA tile data from the Destination register back to L1 memory, with optional format conversion.

## Alternative SFPU Path

The SFPU does not read from the source registers. Instead, it operates entirely on the Destination register:

```
Destination Register  ──(load)──>  SFPU internal registers
                                        │
                                   (compute)
                                        │
SFPU internal registers  ──(store)──>  Destination Register
```

Data must first reach the Destination register before SFPU can process it. There are two ways to get data there:

- **Via the FPU** -- Unpack tiles into Source A/B, then use an FPU data copy operation (`llk_math_eltwise_unary_datacopy`) to transfer the data to the Destination register.
- **Direct unpack** -- Unpacker 0 can write directly to the Destination register from L1, bypassing the source registers entirely.

After the SFPU writes its results back to the Destination register, the packer moves them to L1 in the normal way.

## Register Files

### Source A and Source B Registers

The two source register files hold the input operands for FPU operations.

| Property | Source A | Source B |
|:---------|:---------|:---------|
| **Fed by** | Unpacker 0 | Unpacker 1 |
| **Structure** | Two-dimensional, double-buffered | Two-dimensional, double-buffered |
| **Capacity per bank** | One tile (32x32 datums) | One tile (32x32 datums) |
| **Number of banks** | 2 (double-buffered) | 2 (double-buffered) |
| **Element precision** | Up to 19-bit data elements | Up to 19-bit data elements |

**Double buffering** is a key throughput feature. While the FPU reads from one bank of Source A, the unpacker can simultaneously write the next tile into the other bank. When the FPU finishes, the banks swap roles. This hides unpacker latency behind compute and is managed automatically by the LLK synchronization logic.

The 19-bit element limit in current architectures (WH/BH) means that high-precision formats (e.g., Float32) are truncated when stored in the source registers. For operations requiring full 32-bit precision, the SFPU path through the Destination register is used instead.

### Destination Register (Dest)

The Destination register is the central accumulator of the Tensix Engine. It serves three roles:

1. **FPU output** -- Stores results of FPU computations (dot products, element-wise operations).
2. **SFPU input** -- Provides operands for SFPU operations.
3. **SFPU output** -- Stores results of SFPU operations.

| Property | Value |
|:---------|:------|
| **Capacity** | Configurable: 4 to 16 tiles, depending on data format and configuration |
| **Element precision** | Up to 32-bit data elements |
| **Read by** | FPU (as accumulator), SFPU (as operand source), Packer (for output) |
| **Written by** | FPU (computation results), SFPU (computation results), Unpacker 0 (direct unpack path) |

The Destination register's ability to hold 32-bit elements -- unlike the 19-bit source registers -- is what makes the SFPU path necessary for full-precision operations. The configurable tile capacity allows kernel developers (via configuration registers) to trade off between the number of tiles buffered and the precision of each element.

## Complete Data Flow Summary

Putting it all together, here is every data path through the Tensix Engine:

**Path 1 -- FPU operations:**
L1 --> Unpacker 0 --> Source A --> FPU --> Dest --> Packer --> L1
L1 --> Unpacker 1 --> Source B --> FPU --> Dest --> Packer --> L1

**Path 2 -- SFPU operations (via FPU transfer):**
L1 --> Unpacker 0/1 --> Source A/B --> FPU (datacopy) --> Dest --> SFPU --> Dest --> Packer --> L1

**Path 3 -- SFPU operations (direct unpack to Dest):**
L1 --> Unpacker 0 --> Dest --> SFPU --> Dest --> Packer --> L1

All three paths end the same way: the packer moves data from the Destination register to L1, and the output is ready for the next stage of processing or for transfer off-core via the NoC.

---

**Next:** [Chapter 3 -- Data Organization](../ch3_data_organization/index.md)
