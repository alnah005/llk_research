# Wire Format: Host-Device Page Serialization

*Source: `wire_format.hpp`*

## The Page: 256 Bytes, 64 Words

```cpp
static constexpr uint32_t PAGE_SIZE_BYTES = 256;
static constexpr uint32_t PAGE_SIZE_WORDS = PAGE_SIZE_BYTES / sizeof(uint32_t);  // = 64
using PageBuffer = std::array<uint32_t, PAGE_SIZE_WORDS>;
```

Every page -- inject or result, default or DeepSeek -- is exactly 256 bytes (64 `uint32_t` words). The fixed size eliminates framing, length prefixes, and allocation: both sides pre-allocate a single `PageBuffer` and reuse it for every transaction. The 256-byte size aligns to a single Tenstorrent NoC tile transfer unit.

## Two Layout Families

The engine supports two page layouts, selected by `use_deepseek_md_format`. Each family defines an inject page (H2D) and a result page (D2H):

| Layout | When Used | Distinguishing Feature |
|--------|-----------|----------------------|
| **Default** | Standard decode models | Minimal fields; `slot_id` at word 0 |
| **DeepSeek** | DeepSeek MoE models with speculative decoding | Richer metadata; `token_type`, probability arrays |

## Default Inject Page

| Word | Field | Type | Notes |
|------|-------|------|-------|
| 0 | `slot_id` | `uint32_t` | `INVALID_SLOT` = shutdown sentinel |
| 1 | `token_id` | `uint32_t` | Token to decode |
| 2 | `position` | `uint32_t` | Sequence position |
| 3 | `prefill_token_id` | `uint32_t` | `EMPTY_TOKEN` if this is a decode step |
| 4 | `spec_flag` | `uint32_t` | 0 or 1 |
| 5 | `temperature` | `float` | IEEE 754 bits via `memcpy` |
| 6 | `top_p` | `float` | IEEE 754 bits via `memcpy` |
| 7 | `top_k` | `int32_t` | Cast to `uint32_t`; $-1$ = disabled |
| 8--63 | *(reserved)* | | Zero-filled |

## Default Result Page

| Word | Field | Type | Notes |
|------|-------|------|-------|
| 0 | `slot_id` | `uint32_t` | `INVALID_SLOT` = shutdown sentinel |
| 1 | `actual_token` | `uint32_t` | The token the model sampled |
| 2 | `predicted_token` | `uint32_t` | Speculative next-token prediction |
| 3--5 | *(reserved)* | | |
| 6 | `actual_token_pos` | `uint32_t` | Position of the actual token |
| 7--63 | *(reserved)* | | |

The default result layout is sparse -- most of the 256 bytes are unused. Token types and predicted position are inferred by the host from injection context (FIFO ordering).

## DeepSeek Inject Page

| Word | Field | Type | Notes |
|------|-------|------|-------|
| 0 | *(reserved)* | | |
| 1 | `token_type` | `uint32_t` | 0=BASE, 1=SPEC |
| 2 | `position` | `uint32_t` | Sequence position |
| 3--5 | *(reserved)* | | |
| 6 | `slot_id` | `uint32_t` | |
| 7 | `token_id` | `uint32_t` | |
| 8 | `position_id` | `uint32_t` | Set equal to `position` (`page[8] = desc.position`) |
| 9 | `prefill_token_id` | `uint32_t` | |
| 10 | `temperature` | `float` | `memcpy`'d |
| 11 | `top_k` | `int32_t` | Cast to `uint32_t` |
| 12 | `top_p` | `float` | `memcpy`'d (called `probability_mass_threshold` on device) |
| 13--63 | *(reserved)* | | Zero-filled |

Key differences from default: `slot_id` moves to word 6, `token_type` is explicit at word 1, and `position_id` at word 8 duplicates `position` at word 2 (currently identical, split for future extensibility in rotary position embeddings). `spec_flag` is absent -- DeepSeek firmware infers speculation from `token_type`.

## DeepSeek Result Page

The richest layout, using 48 of the 64 available words:

| Word(s) | Field | Type | Notes |
|---------|-------|------|-------|
| 0 | `actual_token` | `uint32_t` | |
| 1 | `actual_token_type` | `uint32_t` | `TokenType` enum |
| 2 | `actual_token_pos` | `uint32_t` | |
| 3 | `predicted_token` | `uint32_t` | |
| 4 | `predicted_token_type` | `uint32_t` | `TokenType` enum |
| 5 | `predicted_token_pos` | `uint32_t` | |
| 6 | `slot_id` | `uint32_t` | |
| 7--15 | *(reserved)* | | |
| 16--47 | `p_indices[0..31]` | `uint32_t[32]` | Top-32 token indices |
| 48--63 | `p_scores[0..31]` | `bf16[32]` | Packed 2 per word (16 words total) |

### bf16 Score Packing

The `p_scores` array stores bfloat16 values -- a 16-bit float preserving the exponent range of float32:

```
bf16:   [1 sign][8 exponent][7 mantissa]
float:  [1 sign][8 exponent][23 mantissa]
```

Two bf16 values pack into each `uint32_t` word:

$$\text{word}[i] = (\text{score}[2i+1] \ll 16) \;|\; \text{score}[2i]$$

Conversion to float32 is a left-shift: $\text{float32} = \text{bf16} \ll 16$. Together, $128 + 64 = 192$ bytes of the 256-byte page are consumed by probability data alone.

## Serialization: `serialize_inject()`

```cpp
inline PageBuffer serialize_inject(const InjectDescriptor& desc, bool use_deepseek_md_format) {
    PageBuffer page = {};  // zero-initialized
    if (use_deepseek_md_format) {
        page[1] = static_cast<uint32_t>(desc.token_type);
        page[2] = desc.position;
        page[6] = desc.slot_id;
        page[7] = desc.token_id;
        page[8] = desc.position;  // position_id = position
        page[9] = desc.prefill_token_id;
        std::memcpy(&page[10], &desc.temperature, sizeof(float));
        page[11] = static_cast<uint32_t>(desc.top_k);
        std::memcpy(&page[12], &desc.top_p, sizeof(float));
    } else {
        page[0] = desc.slot_id;
        page[1] = desc.token_id;
        page[2] = desc.position;
        page[3] = desc.prefill_token_id;
        page[4] = desc.spec_flag ? 1u : 0u;
        std::memcpy(&page[5], &desc.temperature, sizeof(float));
        std::memcpy(&page[6], &desc.top_p, sizeof(float));
        page[7] = static_cast<uint32_t>(desc.top_k);
    }
    return page;
}
```

Properties: zero-initialization guarantees all reserved words are zero; `std::memcpy` for floats avoids strict-aliasing violations (compilers optimize to a single store); no heap allocation (`PageBuffer` is stack-allocated); no endian conversion (host and device are both little-endian).

## Deserialization: `deserialize_result()`

```cpp
inline ResultDescriptor deserialize_result(const PageBuffer& page, bool use_deepseek_md_format) {
    ResultDescriptor r;
    if (use_deepseek_md_format) {
        r.actual_token         = page[0];
        r.actual_token_type    = static_cast<TokenType>(page[1]);
        r.actual_token_pos     = page[2];
        r.predicted_token      = page[3];
        r.predicted_token_type = static_cast<TokenType>(page[4]);
        r.predicted_token_pos  = page[5];
        r.slot_id              = page[6];
        std::memcpy(r.p_indices.data(), &page[16], 32 * sizeof(uint32_t));  // 128 bytes
        std::memcpy(r.p_scores.data(),  &page[48], 32 * sizeof(uint16_t));  //  64 bytes
    } else {
        r.slot_id          = page[0];
        r.actual_token     = page[1];
        r.predicted_token  = page[2];
        r.actual_token_pos = page[6];
    }
    return r;
}
```

For the default layout, `actual_token_type`, `predicted_token_type`, `predicted_token_pos`, `p_indices`, and `p_scores` remain at their default values -- they are not present in the page.

## Cross-Layout Comparison

| Field | Default Inject | DeepSeek Inject | Default Result | DeepSeek Result |
|-------|---------------|-----------------|---------------|-----------------|
| `slot_id` | word 0 | word 6 | word 0 | word 6 |
| `token_id` | word 1 | word 7 | -- | -- |
| `position` | word 2 | word 2 | -- | -- |
| `prefill_token_id` | word 3 | word 9 | -- | -- |
| `token_type` | -- | word 1 | -- | words 1, 4 |
| `spec_flag` | word 4 | -- | -- | -- |
| `temperature` | word 5 | word 10 | -- | -- |
| `top_p` | word 6 | word 12 | -- | -- |
| `top_k` | word 7 | word 11 | -- | -- |
| `actual_token` | -- | -- | word 1 | word 0 |
| `actual_token_pos` | -- | -- | word 6 | word 2 |
| `predicted_token` | -- | -- | word 2 | word 3 |
| `predicted_token_pos` | -- | -- | -- | word 5 |
| `p_indices[32]` | -- | -- | -- | words 16--47 |
| `p_scores[32]` | -- | -- | -- | words 48--63 |

## Sentinel Pages

The shutdown sentinel is identified by `slot_id == INVALID_SLOT` ($2^{32} - 1$). In the default format, this is word 0. In the DeepSeek format, this is word 6. The `serialize_inject()` function places `INVALID_SLOT` at the correct offset for whichever format is active, so the sentinel is always constructed through the serialization path (see [SocketPipeline](./socket_pipeline.md) shutdown code).

## Design Rationale

**Why fixed-size pages?** Variable-length messages would require framing, length prefixes, and error recovery. A fixed 256-byte page can be DMA'd in a single NoC transaction with no parsing on the device side.

**Why two layouts?** The DeepSeek layout was dictated by existing device firmware. Rather than version negotiation, the host selects the layout at construction time. This keeps the hot path branch-free.

**Why `memcpy` for floats?** C++ strict aliasing forbids writing a `float` through a `uint32_t*`. `std::memcpy` is the standard-compliant way to type-pun. Modern compilers optimize it to a single store instruction.

---

| | Navigation | |
|:---|:---:|---:|
| [SocketPipeline](./socket_pipeline.md) | [Table of Contents](../index.md) | [Chapter 6](../ch6_lifecycle_is_integration_kv_tiering/index.md) |
