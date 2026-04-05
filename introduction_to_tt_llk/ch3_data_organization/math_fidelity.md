# Math Fidelity

## The `MathFidelity` Enum

The Tensix FPU can only use a limited number of mantissa bits in a single multiplication pass. To utilize all the bits from wider data formats, the hardware can execute multiple multiplication **phases**, accumulating partial products to build up full precision. The `MathFidelity` enum (in `tt_llk_wormhole_b0/llk_lib/llk_defs.h`) controls how many phases are used:

```cpp
enum class MathFidelity : std::uint8_t
{
    LoFi  = 0,  // Lowest fidelity: fewest phases, fastest
    HiFi2 = 2,  // Two fidelity phases
    HiFi3 = 3,  // Three fidelity phases
    HiFi4 = 4   // Four fidelity phases: full precision, slowest
};
```

Higher fidelity means more multiplication phases, which produces more accurate results at the cost of additional cycles.

## How Fidelity Controls Multiplication Phases

The fidelity value encodes two pieces of information, extracted by helpers in `tt_llk_wormhole_b0/common/inc/cmath_common.h`:

```cpp
inline constexpr int get_math_num_fidelity_phases(const int math_fidelity_desc)
{
    return (math_fidelity_desc & 0x7);
}

inline constexpr int get_math_fidelity_increment(const int math_fidelity_desc)
{
    return ((math_fidelity_desc >> 3) & 0x1) + 1;
}
```

- **Number of phases** (`get_math_num_fidelity_phases`): the low 3 bits of the fidelity value. For `LoFi` this is 0 (meaning a single base pass with no additional fidelity phases), for `HiFi2` it is 2, for `HiFi3` it is 3, and for `HiFi4` it is 4.
- **Fidelity increment** (`get_math_fidelity_increment`): derived from bit 3. For all standard fidelity levels (values 0-4), bit 3 is 0, so the increment is 1. This value is used in address modifier configuration to step through the mantissa bits in each successive phase.

Each additional phase processes the next segment of mantissa bits and accumulates the partial product into the Destination register. The FPU's address modifier is configured with the fidelity increment so that each phase reads a different slice of the source operand bits.

## The `is_high_fidelity()` Helper

A simple predicate distinguishes LoFi from all higher-fidelity modes:

```cpp
inline constexpr bool is_high_fidelity(const MathFidelity math_fidelity_desc)
{
    return math_fidelity_desc != MathFidelity::LoFi;
}
```

This is used throughout the LLK codebase to conditionally execute additional fidelity phases. When `is_high_fidelity()` returns `false`, the kernel skips the extra multiplication passes entirely.

## Accuracy vs. Performance Trade-off

| Fidelity | Phases | Relative Speed | Use Case |
|:---------|:-------|:---------------|:---------|
| `LoFi` | 1 (base only) | Fastest | Inference with low-precision formats (BFP8, Float16_b) where slight accuracy loss is acceptable. |
| `HiFi2` | 2 | ~2x slower than LoFi | Good balance for most workloads; covers the most significant mantissa bits. |
| `HiFi3` | 3 | ~3x slower than LoFi | Higher precision for accumulation-heavy operations. |
| `HiFi4` | 4 | ~4x slower than LoFi | Full precision; required when using Float32 or TF32 inputs and exact results are needed. |

The performance impact is roughly linear in the number of phases because each phase requires a full pass through the FPU for every face of the tile. In practice, the overhead may be partially hidden by pipelining with unpack and pack operations, but the FPU-bound portion of execution time scales directly with the phase count.

Developers choose fidelity based on their model's sensitivity to numerical error. Training workloads that accumulate gradients over many iterations typically benefit from `HiFi3` or `HiFi4`. Inference workloads with quantized weights (e.g., BFP8) often use `LoFi` or `HiFi2` with no measurable accuracy loss.

---

**Next:** [Chapter 4 — The Unpack-Math-Pack Pipeline](../ch4_unpack_math_pack_pipeline/index.md)
