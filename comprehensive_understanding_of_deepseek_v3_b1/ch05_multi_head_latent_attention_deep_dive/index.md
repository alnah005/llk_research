# Chapter 5 -- Multi-head Latent Attention Deep Dive

Chapter 4 cataloged the fused operations at a structural level -- what stages they contain, which cores they occupy, and how their circular buffers are wired. This chapter traces the complete Multi-head Latent Attention (MLA) mechanism end-to-end, following every tensor from the moment it enters the attention block to the moment the output projection merges it back into the residual stream. Where Chapter 4 described the *what*, this chapter explains the *why*: why each projection exists, why weights are shuffled the way they are, why the KV cache uses a particular DRAM layout, and why the attention kernel fuses online softmax with V projection.

DeepSeek V3 uses MLA to compress the KV cache from $O(n_h \cdot d_h)$ per token to $O(d_c)$ where $d_c \ll n_h \cdot d_h$. Standard multi-head attention stores separate K and V projections for all heads -- for 128 heads at 128+64 dimensions each, that is 24,576 elements per token. MLA instead stores a single compressed latent vector of 576 elements (512 nope + 64 rope), reducing the per-token KV cache to 576 elements -- a 42.7x compression ratio at the element level, or approximately 80x when combined with BFP8 storage versus BF16.

The narrative follows the data flow of a single decode step through four phases:

1. **Query projection and weight layout** -- how the hidden state becomes 64 query heads with both nope and rope components, and why the weight columns must be shuffled to match the grid topology.
2. **Compressed KV cache** -- how the same hidden state simultaneously produces the 576-element compressed KV entry, and how that entry is written to a DRAM cache designed for streaming reads.
3. **Flash MLA decode** -- how the fused attention kernel computes scaled dot-product attention across a variable-length KV cache using chunked online softmax, S-block parallelism, double-buffered DRAM streaming, and tree reduction.
4. **Post-SDPA output projection** -- how the attention output is expanded through the kv_b2 and o_proj matmuls and reduced across MLA-TP devices with an integrated CCL all-reduce.

Source root: `models/demos/deepseek_v3_b1/`

---

## Contents

### [5.1 MLA Query Projection and Weight Layout](./01_mla_query_projection_and_weight_layout.md)

| Topic | Key Concepts |
|-------|-------------|
| Q-path pipeline | RMSNorm $\to$ q\_a\_proj $\to$ gather-reduce $\to$ q\_a\_norm $\to$ q\_b\_proj $\to$ split $\to$ Matmul3/RoPE |
| Weight catalog | q\_a\_proj (7168, 1536), q\_b\_proj (1536, 12288), kv\_a\_proj (7168, 576), kv\_b1 (8192, 512), kv\_b2 (512, 8192), o\_proj (8192, 7168), normalization gammas |
| Weight shuffling | `shuffle_weights_for_interleaved_qnope_qrope` reorders q\_b\_proj columns by grid row |
| K-split packing | `shuffle_q_a` interleaves K-halves into packed (3584, 3072) layout |
| RoPE application | cos/sin DRAM lookup, `rotate_half` via `trans_mat`, position\_ids indexing |
| HF-to-hardware pipeline | Transpose, TP slice, shuffle, BFP8 convert, fuse into 3 overlap groups |
| TP=2 distribution | Per-device 64 heads, replicated $W_{q\_a}$, sharded $W_{q\_b}$ |

### [5.2 Compressed KV Cache](./02_compressed_kv_cache.md)

| Topic | Key Concepts |
|-------|-------------|
| KV-path pipeline | input $\to$ kv\_a\_proj $\to$ gather $\to$ kv\_a\_norm $\to$ RoPE $\to$ cache update |
| `kv_cache_branch` fusion | DKV matmul on 18 cores + gather to (0,8) + RMSNorm + RoPE in one program |
| Cache tensor layout | `[batch, 1, max_seq_len, 576]` ND-sharded, `shard_height = k_chunk_size` |
| BFP8 format | 1088 bytes/tile vs 2048 BF16; ~1.88x DRAM savings; read-modify-write update |
| DRAM bank ordering | `FlashMLAOptimalGridNOC0` maps S-blocks to banks for NOC locality |
| Position tracking | `cur_pos_tensor`, `kv_cache_cur_pos_ready` semaphore protocol (value=3) |
| Standalone vs fused CBs | CB index remapping between `kv_cache_branch` and `pre_sdpa` modes |

### [5.3 Flash MLA Decode](./03_flash_mla_decode.md)

| Topic | Key Concepts |
|-------|-------------|
| Algorithm | Shared KV tensor, chunked $QK^T$ with online softmax, fused V projection from K buffer |
| S-block architecture | 8 blocks of 8 cores each, columns 0--3 and 7--10 |
| Sequence-length parallelism | 8 cores per Q shard, each processing a subset of K chunks |
| Double-buffered K reads | DRAM streaming with page-level pipelining, 144-tile CB |
| TRISC pseudocode | Full online softmax compute loop with DST register layout |
| Tree reduction | 3-step $\log_2(8)$ reduction with per-step buffer isolation and bit-packed semaphores |
| `SDPAReduceToAll` | Cross-device partial attention reduction via forwarder/worker fabric, chunked L transfer |

### [5.4 Post-SDPA Output Projection](./04_post_sdpa_output_projection.md)

| Topic | Key Concepts |
|-------|-------------|
| Matmul chain | SDPA output $\to$ kv\_b2 $\to$ gather $\to$ mcast $\to$ o\_proj $\to$ gather $\to$ CCL all-reduce |
| `post_sdpa` fusion | Matmul4 + Gather2 + Mcast3 + Matmul5 + Gather3 + CCL AllReduce |
| 13$\times$10 mcast grid | 112 active matmul cores, 18 inactive mcast receivers |
| CCL integration | Receiver on (12,9), sender on (11,9), ring topology with fabric routing |
| Data volume tracking | Bytes through every pipeline stage from scatter to all-reduce |

---

**Previous:** [Chapter 4 -- The Fused-Op Library](../ch04_the_fused_op_library/index.md)
**Next:** [Chapter 6 -- Mixture-of-Experts Deep Dive](../ch06_mixture_of_experts_deep_dive/index.md)
