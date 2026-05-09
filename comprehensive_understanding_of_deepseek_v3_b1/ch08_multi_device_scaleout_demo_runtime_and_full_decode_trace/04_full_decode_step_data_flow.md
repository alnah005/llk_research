# 8.4 Full Decode Step Data Flow

This section provides an end-to-end annotated trace of a single decode step
through the entire DeepSeek V3 pipeline -- from the host writing a token ID to
receiving the next-token logits. The trace covers embedding lookup, dense
transformer layers, MoE transformer layers, and the LM head, with tensor shapes
and residual connections annotated at every boundary. This is the **capstone
section** of the guide, tying all chapters together: every pipeline stage
references the detailed explanation in its corresponding earlier chapter.

> **Cross-reference -> Chapter 2, Section 2.1**: Sub-block kernel architecture
> that executes within each pipeline stage.

> **Cross-reference -> Chapter 4, Section 4.3**: MLA attention pipeline
> (Pre-SDPA, Flash MLA, Post-SDPA).

> **Cross-reference -> Chapter 5, Section 5.2**: MoE gate, expert dispatch,
> and ReduceToOne reduction.

> **Cross-reference -> Chapter 6, Section 6.1**: Dense FFN kernel chain.

Source: `model.py`, `micro_ops/pipeline_block/op.py`, `micro_ops/d2d_exchange/op.py`, `micro_ops/host_io/op.py`, `micro_ops/ccl_all_reduce/op.py`, `micro_ops/ccl_broadcast/op.py`, `micro_ops/reduce_to_one_b1/op.py`

---

## 8.4.1 Decode Step Overview

A single decode step processes **one token per batch element** (B=1 in current
implementation). The token travels through 63 pipeline stages as a hidden state
vector, being transformed by each layer in sequence:

```
 FULL PIPELINE: 63 STAGES, ONE DECODE STEP
 ==========================================

 Host CPU                                           Host CPU
    |                                                  ^
    | write_tensor()                                   | read_tensor()
    | (64 bytes, int32 token ID)                       | (logits/token ID)
    v                                                  |
 +------+  +------+  +------+  +------+       +------+  +------+
 |Stg 0 |->|Stg 1 |->|Stg 2 |->|Stg 3 |->..->|Stg 61|->|Stg 62|
 |Embed  |  |Dense |  |Dense |  |Dense |  |   | MoE  |  |  LM  |
 |       |  |Lyr 0 |  |Lyr 1 |  |Lyr 2 |  |   |Lyr 60|  | Head |
 +------+  +------+  +------+  +------+  |   +------+  +------+
                                           |                |
            58 MoE layers (stages 4-61) ---+                |
                                                            v
                                                     Loopback to
                                                     Stage 0 (D2H)
```

All 63 processing stages operate concurrently in a pipelined fashion. After
pipeline warmup (63 steps to fill), each new token exits one step after it
enters. The pipeline latency for a single token is the sum of all stage
latencies; the throughput at steady state approaches the latency of the
slowest stage.

---

## 8.4.2 Visual Pipeline Diagram with Tensor Shapes

The following diagram traces the tensor shapes at every stage boundary. All
dimensions assume B=1 (batch size 1), d_model=7168 (DeepSeek V3 hidden
dimension), and V=129280 (vocabulary size).

```
 ANNOTATED DECODE PIPELINE WITH TENSOR SHAPES
 ==============================================

 HOST
  |
  |  token_id: int32, 64 bytes (PCIe-aligned)
  |  Actual payload: 4 bytes (1 x int32)
  v
 +================================================================+
 | STAGE 0: EMBEDDING (mesh_id=0)                                 |
 |                                                                 |
 |  Input:  token_id (int32)                                      |
 |  Op:     DRAM table lookup (fused in H2D receiver kernel)      |
 |          embedding_tensor[token_id] -> hidden state             |
 |  Output: (1, 7168) bfloat16 = 14,336 bytes                    |
 |                                                                 |
 |  Source: fused_h2d_receiver_embedding.cpp                      |
 +================================================================+
  |
  |  hidden: (1, 7168) bf16, 14336 bytes via D2D socket
  v
 +================================================================+
 | STAGE 1: DENSE LAYER 0 (mesh_id=1, layer_id=0)                |
 |                                                                 |
 |  [RMSNorm]  (1, 7168) -> (1, 7168)                            |
 |      Save residual = hidden                                    |
 |                                                                 |
 |  [MLA Attention -- TP=2 across columns]                        |
 |      Q_a projection: (1, 7168) -> (1, 1536)                   |
 |      Q_b projection: (1, 1536) -> (1, nqh*128)                |
 |      KV_a projection: (1, 7168) -> (1, 576)                   |
 |      KV_b split: (1, 576) -> K_nope(1, 128) + KV_pe(1, 448)  |
 |      Flash MLA:  Q, K_compressed, V_compressed -> attn_out    |
 |      O projection: attn_out -> (1, 7168)                      |
 |      AllReduce(TP=2): sum across 2 columns                    |
 |                                                                 |
 |  [Residual Add]  hidden = attn_out + residual                  |
 |                  (fused with AllReduce when possible)           |
 |                                                                 |
 |  [RMSNorm]  (1, 7168) -> (1, 7168)                            |
 |      Save residual = hidden                                    |
 |                                                                 |
 |  [Dense FFN -- SwiGLU]                                         |
 |      Gate proj:  (1, 7168) -> (1, 18432)                      |
 |      Up proj:    (1, 7168) -> (1, 18432)                      |
 |      SiLU(gate) * up -> (1, 18432)                             |
 |      Down proj:  (1, 18432) -> (1, 7168)                      |
 |                                                                 |
 |  [Residual Add]  hidden = ffn_out + residual                   |
 |                                                                 |
 |  Output: (1, 7168) bf16                                        |
 +================================================================+
  |
  |  hidden: (1, 7168) bf16 via D2D socket
  v
 +================================================================+
 | STAGES 2-3: DENSE LAYERS 1-2 (same structure as Stage 1)      |
 |  Each: RMSNorm -> MLA(TP=2) -> AllReduce -> Residual ->       |
 |        RMSNorm -> Dense FFN -> Residual                        |
 |  Output: (1, 7168) bf16                                        |
 +================================================================+
  |
  |  hidden: (1, 7168) bf16 via D2D socket
  v
 +================================================================+
 | STAGE 4: MoE LAYER 3 (mesh_id=4, layer_id=3)                  |
 |                                first MoE layer                 |
 |                                                                 |
 |  [RMSNorm]  (1, 7168) -> (1, 7168)                            |
 |      Save residual = hidden                                    |
 |                                                                 |
 |  [MLA Attention -- TP=2 across columns]                        |
 |      (same Q/KV projections as dense layer)                    |
 |      AllReduce(TP=2): sum across 2 columns                    |
 |                                                                 |
 |  [Residual Add]  hidden = attn_out + residual                  |
 |                                                                 |
 |  [RMSNorm]  (1, 7168) -> (1, 7168)                            |
 |      Save residual = hidden                                    |
 |                                                                 |
 |  [MoE Block -- EP=8 across all devices]                        |
 |      Gate:   (1, 7168) -> (1, 256) routing scores              |
 |      Broadcast gate scores to all 8 devices                    |
 |      Top-K selection (K=8 active experts per token)            |
 |      Expert dispatch: each device runs its 32 experts          |
 |        Per expert: gate_proj + up_proj + SiLU + down_proj      |
 |        (1, 7168) -> (1, 2048) -> (1, 7168) per expert          |
 |      ReduceToOne: 3-level tree aggregation to ROOT1            |
 |                                                                 |
 |  [Shared Expert]                                               |
 |      gate_proj: (1, 7168) -> (1, 2048)                        |
 |      up_proj:   (1, 7168) -> (1, 2048)                        |
 |      SiLU(gate) * up -> (1, 2048)                              |
 |      down_proj: (1, 2048) -> (1, 7168)                        |
 |                                                                 |
 |  [Combine]  moe_out = routed_expert_out + shared_expert_out   |
 |                                                                 |
 |  [Residual Add]  hidden = moe_out + residual                   |
 |                                                                 |
 |  Output: (1, 7168) bf16                                        |
 +================================================================+
  |
  |  hidden: (1, 7168) bf16 via D2D socket
  v
 +================================================================+
 | STAGES 5-61: MoE LAYERS 4-60 (same structure as Stage 4)      |
 |  Each: RMSNorm -> MLA(TP=2) -> AllReduce -> Residual ->       |
 |        RMSNorm -> MoE(EP=8) -> ReduceToOne -> Residual        |
 |  57 more MoE layers in sequence                                |
 |  Output: (1, 7168) bf16                                        |
 +================================================================+
  |
  |  hidden: (1, 7168) bf16 via D2D socket
  v
 +================================================================+
 | STAGE 62: LM HEAD + SAMPLING (mesh_id=62)                     |
 |                                                                 |
 |  [RMSNorm]  (1, 7168) -> (1, 7168)                            |
 |  [LM Head]  (1, 7168) -> (1, 129280) vocab logits             |
 |     W_lm: (7168, 129280), sharded across 8 devices             |
 |  [Sampling] argmax or top-p/top-k -> next_token_id (int32)    |
 |                                                                 |
 |  Output: token_id (int32) or logits                            |
 +================================================================+
  |
  |  Loopback via D2D socket chain -> Stage 0 -> D2H -> Host
  v
 HOST: read_tensor() -> extract_token_id() -> next decode step
```

---

## 8.4.3 Stage 0: Embedding (Fused H2D + DRAM Lookup)

The embedding stage is unique because it fuses the H2D receive operation with
an inline DRAM table lookup. The host writes a 64-byte packet containing the
token ID, and the fused kernel converts it to a 14,336-byte embedding vector.

```
 EMBEDDING STAGE DETAIL
 =======================

 Host            Core (0,0) on Entry Node        Core on Exit Node
  |               +-------------------------+     +------------------+
  | 64 bytes      | fused_h2d_receiver_     |     |                  |
  | int32 token   | embedding.cpp           |     | d2d_exchange.cpp |
  |-------------->|                         |     |                  |
  |               | 1. Receive 64B packet   |     |                  |
  |               | 2. Extract token_id     |     |                  |
  |               | 3. DRAM read:           |     |                  |
  |               |    embedding[token_id]  |     |                  |
  |               |    (1 x 7168 x bf16)    |     |                  |
  |               | 4. Forward via D2D      |---->| Forward to       |
  |               |    socket (14336 bytes) |     | Stage 1 entry    |
  |               +-------------------------+     +------------------+
```

The embedding tensor is stored in DRAM as an interleaved row-major tensor with
shape `(129280, 7168)` in bfloat16:

> **Source**: `prepare_weights.py`, line 666:
> `assert w.shape == (129280, 7168)`

The page size equals the inner dimension stride:

> **Source**: `micro_ops/host_io/op.py`, line 143:
> ```python
> self.embedding_page_size = self.embedding_tensor.shape[3] * dtype_size(
>     self.embedding_tensor.dtype)
> ```

For DeepSeek V3: `7168 * 2 = 14,336 bytes` per embedding row (bfloat16).

The downstream D2D socket page size must match:

> **Source**: `micro_ops/pipeline_block/op.py`, lines 79-82:
> ```python
> assert downstream_d2d_socket_page_size == embedding_size_bytes
> ```

> **Cross-reference -> Chapter 3, Section 3.2**: Embedding weight layout
> in DRAM.

---

## 8.4.4 Stages 1-3: Dense Transformer Layers

Each dense layer performs the same sequence of operations. The key distinction
from MoE layers is the FFN block: dense layers use a standard feed-forward
network with gate, up, and down projections (dimensions 7168 -> 18432 -> 7168).

### Residual Connection Tracking

Residual connections are critical for gradient flow and are tracked through each
layer:

```
 RESIDUAL CONNECTIONS IN A DENSE LAYER
 ======================================

 hidden_in ---------+
      |              |
      v              |  residual_1
  [RMSNorm]         |
      |              |
      v              |
  [MLA Attn]        |
      |              |
      v              |
  [AllReduce]       |
      |              |
      v              |
  [+ residual_1] <--+
      |
      v
 hidden_mid --------+
      |              |
      v              |  residual_2
  [RMSNorm]         |
      |              |
      v              |
  [Dense FFN]       |
      |              |
      v              |
  [+ residual_2] <--+
      |
      v
 hidden_out --> D2D to next stage
```

The AllReduce for MLA attention supports fused residual add, where the
reduction sum and residual addition happen in a single kernel pass:

> **Source**: `micro_ops/ccl_all_reduce/op.py`, lines 56-64:
> The `residual_tensor_mesh` parameter enables fused `result = sum(inputs) + residual`.

### MLA Attention Data Flow (TP=2)

Each device in the column pair processes its TP shard:

```
Device [r, 0]                           Device [r, 1]
+---------------------------+           +---------------------------+
| RMSNorm(hidden_state)     |           | RMSNorm(hidden_state)     |
| shape: (1, 7168) bf16     |           | shape: (1, 7168) bf16     |
|                            |           |                            |
| Q_a proj: (1,7168)->(1,1536)          | Q_a proj: (1,7168)->(1,1536)
| KV_a proj: (1,7168)->(1,576)          | KV_a proj: (1,7168)->(1,576)
|                            |           |                            |
| ... MLA attention ...      |           | ... MLA attention ...      |
| (see Ch4 for full trace)  |           | (see Ch4 for full trace)  |
|                            |           |                            |
| attn_out_partial           |           | attn_out_partial           |
| shape: (1, 3584) bf16     |           | shape: (1, 3584) bf16     |
+---------------------------+           +---------------------------+
             |                                       |
             v                                       v
        AllReduce (2-device, along column axis)
             |
             v
      attn_out shape: (1, 7168) bf16 on both devices
```

> **Cross-reference -> Chapter 4, Section 4.3**: Complete MLA computation
> including Pre-SDPA projections, Flash MLA, and Post-SDPA output projection.

### Dense FFN Data Flow

```
ffn_input = RMSNorm(hidden_state)         # (1, 7168) bf16

gate = ffn_input @ W_gate                  # (1, 7168) -> (1, 18432)
up   = ffn_input @ W_up                    # (1, 7168) -> (1, 18432)
ffn_mid = SiLU(gate) * up                  # (1, 18432)
ffn_out = ffn_mid @ W_down                 # (1, 18432) -> (1, 7168)

hidden_state = hidden_state + ffn_out      # Residual add
```

Dense layers repeat 3 times (stages 1-3 for decoder layers 0-2).

> **Cross-reference -> Chapter 6, Section 6.1**: Dense FFN kernel chain
> (gate/up projection, SwiGLU activation, down projection).

---

## 8.4.5 Stages 4-61: MoE Transformer Layers

MoE layers replace the dense FFN with a Mixture-of-Experts block. The attention
portion is identical to dense layers; the difference is entirely in the FFN
replacement.

### MoE Block Data Flow

```
 MoE BLOCK DATA FLOW (within one 4x2 mesh)
 ===========================================

           hidden (1, 7168) on all 8 devices
                     |
                     v
        +--[Gate Computation]--+
        |  On sender device    |
        |  (1,7168) -> (1,256) |
        |  + gate_bias         |
        +----------+-----------+
                   |
                   v
        +--[Broadcast]--+
        |  Gate scores  |
        |  to all 8 dev |
        +-------+-------+
                |
      +---------+---------+--------+--------+--------+--------+--------+
      |         |         |        |        |        |        |        |
      v         v         v        v        v        v        v        v
   Dev 0     Dev 1     Dev 2    Dev 3    Dev 4    Dev 5    Dev 6    Dev 7
   Exp 0-31  Exp 32-63 Exp 64-95 ...                              Exp 224-255
   top-K     top-K     top-K    top-K    top-K    top-K    top-K    top-K
   select    select    select   select   select   select   select   select
      |         |         |        |        |        |        |        |
      v         v         v        v        v        v        v        v
   Expert    Expert    Expert   Expert   Expert   Expert   Expert   Expert
   compute   compute   compute  compute  compute  compute  compute  compute
      |         |         |        |        |        |        |        |
      +----+----+----+----+---+----+---+----+---+----+---+----+---+----+
           |                                                        |
           v                                                        v
        +--[ReduceToOne: 3-level tree]-----------------------------+
        |  Level 1: Leaf rows -> inner rows                         |
        |  Level 2: ROOT3 -> ROOT2/ROOT1                            |
        |  Level 3: ROOT2 -> ROOT1 (cross-column)                   |
        +---------------------------+-------------------------------+
                                    |
                                    v
                           (1, 7168) at ROOT1
                                    |
                                    v
                          [+ shared_expert_out]
                                    |
                                    v
                          [+ residual] -> D2D to next stage
```

Each expert performs the same SwiGLU computation with a smaller intermediate
dimension:

```
Per expert (routed):
  gate_proj: (1, 7168) -> (1, 2048)
  up_proj:   (1, 7168) -> (1, 2048)
  SiLU(gate) * up -> (1, 2048)
  down_proj: (1, 2048) -> (1, 7168)
```

The shared expert runs in parallel on designated devices with the same
architecture:

```
Shared expert:
  gate_proj: (1, 7168) -> (1, 2048)
  up_proj:   (1, 7168) -> (1, 2048)
  SiLU(gate) * up -> (1, 2048)
  down_proj: (1, 2048) -> (1, 7168)
```

The gate bias (`e_score_correction_bias`) is applied during gate computation
for MoE layers only:

> **Source**: `prepare_weights.py` (line 78): `gate_bias` field in
> `AttentionWeights` dataclass, present only for MoE layers.

With 256 experts distributed across 8 devices (32 per device) and top-8
routing, each device typically computes 0-3 experts per token.

> **Cross-reference -> Chapter 5, Section 5.2**: MoE gate, top-K selection,
> and expert dispatch.

> **Cross-reference -> Chapter 5, Section 5.4**: ReduceToOne 3-level
> reduction tree.

### ReduceToOne Gather Detail

```
Level 1 (parallel, within columns):
  [0,0] LEAF -----> [1,0] ROOT2     (a0 -> a1)
  [0,1] LEAF -----> [1,1] ROOT1     (b0 -> b1)
  [3,0] LEAF -----> [2,0] ROOT3     (a3 -> a2)
  [3,1] LEAF -----> [2,1] ROOT3     (b3 -> b2)

Level 2 (parallel across columns):
  [2,0] ROOT3 ----> [1,0] ROOT2     (a2 -> a1, includes a2+a3)
  [2,1] ROOT3 ----> [1,1] ROOT1     (b2 -> b1, includes b2+b3)

Level 3 (cross-column):
  [1,0] ROOT2 ----> [1,1] ROOT1     (a1+a0+a2+a3 -> b1)

Result at ROOT1 [1,1]: sum of all 8 partial expert outputs
```

After reduction, the combined MoE output and residual add:

```
moe_out = sum(weighted_expert_outs) + shared_out
hidden_state = hidden_state + moe_out     # (1, 7168) bf16
```

This MoE layer sequence repeats 58 times (stages 4-61 for decoder layers 3-60).

---

## 8.4.6 Stage 62: LM Head and Sampling

The final computation stage converts the hidden state to vocabulary logits and
selects the next token.

```
 LM HEAD STAGE DETAIL
 =====================

 hidden: (1, 7168) bf16 from Stage 61
          |
          v
    [RMSNorm]
     (1, 7168) -> (1, 7168)
          |
          v
    [LM Head Weight Multiply]
     W_lm: (7168, 129280)
     (1, 7168) x (7168, 129280) -> (1, 129280)
     Result: logits over full vocabulary
          |
          v
    [Sampling]
     argmax(logits) -> token_id (int32)
          |
          v
    Output: token_id sent via D2D to loopback path
```

The LM head weight matrix has shape `(7168, 129280)`. On the 4x2 mesh, the
vocab dimension is sharded across 8 devices, giving each device `129280/8 =
16160` logit columns.

> **Source**: `prepare_weights.py`, line 717:
> `_LM_HEAD_VOCAB_SIZE = 129280`

> **Cross-reference -> Chapter 7, Section 7.3**: LM head weight preparation
> and vocab sharding.

---

## 8.4.7 Loopback Path: Stage 62 -> Stage 0 -> Host

After the LM head produces the output token, it must travel back to the host.
In the single-pod loopback topology, the last stage connects back to stage 0
via the fabric ring:

```
 LOOPBACK DATA PATH
 ===================

 Stage 62 (LM Head)
   Exit SocketInterface
       |
       | D2D exchange (fabric, wraps around ring)
       v
 Stage 0 (Embedding stage)
   Entry SocketInterface
       |
       | upstream_socket -> HostInterface upstream
       v
   D2H socket
       |
       | d2h_socket.read_tensor()
       v
 Host CPU: extract_token_id(output) -> next_token_id
```

The PipelineBlock for stage 0 creates the loopback entry socket by connecting
from `pipeline_config[num_procs - 1].exit_node_coord` to
`pipeline_config[num_procs].entry_node_coord`:

> **Source**: `micro_ops/pipeline_block/op.py`, lines 123-132

The PipelineBlock for the last stage sends its exit to mesh_id 0:

> **Source**: `micro_ops/pipeline_block/op.py`, line 156:
> ```python
> receiver_mesh=MeshWrapper(
>     mesh_id=self.my_mesh_id + 1 if self.my_mesh_id < num_procs - 1 else 0)
> ```

For the superpod, the loopback traverses stage 63 as an intermediate:
```
Stage 62 -> Stage 63 (loopback relay) -> Stage 0
            (d05u08:slice0)
```

---

## 8.4.8 End-to-End Annotated Trace Table

```
Step | Stage | Operation                | Tensor Shape        | Size     | Bound
-----|-------|--------------------------|---------------------|----------|--------
  1  | Host  | Pad token ID             | (1,16) int32        | 64 B     | CPU
  2  | Host  | H2D write (HOST_PUSH)    | (1,16) int32        | 64 B     | PCIe
  3  | Stg 0 | H2D receive              | (1,16) int32        | 64 B     | PCIe
  4  | Stg 0 | Embedding lookup (DRAM)  | (1,7168) bf16       | 14336 B  | Memory
  5  | Stg 0 | D2D exit to stage 1      | (1,7168) bf16       | 14336 B  | Fabric
  6  | Stg 1 | D2D entry receive        | (1,7168) bf16       | 14336 B  | Fabric
  7  | Stg 1 | RMSNorm                  | (1,7168) bf16       | 14336 B  | Compute
  8  | Stg 1 | MLA Attention (TP=2)     | (1,7168) bf16       | 14336 B  | Memory*
  9  | Stg 1 | AllReduce (2 cols)       | (1,7168) bf16       | 14336 B  | Latency
 10  | Stg 1 | Residual + RMSNorm       | (1,7168) bf16       | 14336 B  | Compute
 11  | Stg 1 | Dense MLP (SwiGLU)       | (1,7168) bf16       | 14336 B  | Memory
 12  | Stg 1 | Residual add             | (1,7168) bf16       | 14336 B  | Compute
 13  | Stg 1 | D2D exit to stage 2      | (1,7168) bf16       | 14336 B  | Fabric
     | ...   | (stages 2-3: same as 1)  |                     |          |
 14  | Stg 4 | D2D entry receive        | (1,7168) bf16       | 14336 B  | Fabric
 15  | Stg 4 | RMSNorm + MLA + Reduce   | (1,7168) bf16       | 14336 B  | Memory*
 16  | Stg 4 | Broadcast (1->8 devices) | (1,7168) bf16       | 14336 B  | Latency
 17  | Stg 4 | MoE Gate + TopK          | (1,256) bf16        | 512 B    | Compute
 18  | Stg 4 | Expert compute (EP=8)    | (1,7168) bf16 x8    | 14336 B  | Memory
 19  | Stg 4 | Shared expert (parallel) | (1,7168) bf16       | 14336 B  | Memory
 20  | Stg 4 | ReduceToOne (8->1)       | (1,7168) bf16       | 14336 B  | Latency
 21  | Stg 4 | Residual add             | (1,7168) bf16       | 14336 B  | Compute
 22  | Stg 4 | D2D exit to stage 5      | (1,7168) bf16       | 14336 B  | Fabric
     | ...   | (stages 5-61: same as 4) |                     |          |
 23  | Stg62 | D2D entry receive        | (1,7168) bf16       | 14336 B  | Fabric
 24  | Stg62 | CCL Broadcast (1->8)     | (1,7168) bf16       | 14336 B  | Latency
 25  | Stg62 | RMSNorm                  | (1,7168) bf16       | 14336 B  | Compute
 26  | Stg62 | Vocab matmul (sharded)   | (1,16160) bf16/dev  | varies   | Memory
 27  | Stg62 | Fused argmax             | (1,1) uint32        | 4 B      | Latency
 28  | Stg62 | Socket output            | (1,16) uint32       | 64 B     | Fabric
 29  | Loop  | D2D loopback to stage 0  | (1,16) uint32       | 64 B     | Fabric
 30  | Stg 0 | D2H send                 | (1,16) uint32       | 64 B     | PCIe
 31  | Host  | D2H read                 | (1,16) uint32       | 64 B     | PCIe
 32  | Host  | Extract token ID         | int scalar          | 4 B      | CPU
```

*Memory: SDPA KV cache reads dominate for long sequences.

---

## 8.4.9 Tensor Shape Summary Table

| Location                    | Tensor                | Shape/Size                | Type     |
|-----------------------------|-----------------------|---------------------------|----------|
| Host -> Stage 0             | token_id              | 64 bytes (aligned)        | int32    |
| Stage 0 output              | embedding             | (1, 7168)                 | bf16     |
| RMSNorm input/output        | hidden                | (1, 7168)                 | bf16     |
| Q_a projection output       | compressed_q          | (1, 1536)                 | bf16     |
| KV_a projection output      | compressed_kv         | (1, 576)                  | bf16     |
| MLA attention output        | attn_out              | (1, 7168)                 | bf16     |
| Dense FFN gate/up output    | intermediate          | (1, 18432)                | bf16     |
| Dense FFN down output       | ffn_out               | (1, 7168)                 | bf16     |
| MoE gate output             | gate_scores           | (1, 256)                  | bf16     |
| MoE expert intermediate     | expert_hidden         | (1, 2048) per expert      | bf16     |
| MoE expert output           | expert_out            | (1, 7168) per expert      | bf16     |
| ReduceToOne output          | reduced               | (1, 7168)                 | bf16     |
| Stage N -> Stage N+1        | hidden                | (1, 7168) = 14,336 bytes  | bf16     |
| LM head output              | logits                | (1, 129280)               | bf16     |
| LM head per-device shard    | logits_shard          | (1, 16160) per device     | bf16     |
| Sampling output             | token_id              | 4 bytes                   | int32    |
| Stage 62 -> Host            | output                | 64 bytes (PCIe-aligned)   | int32    |

---

## 8.4.10 Residual Connection Tracking

The residual stream passes through each layer unchanged except for two
additions. The complete algebraic trace through the pipeline:

```
x_0 = Embedding(token_id)                         # (1, 7168)

# Dense layer 0 (stage 1)
x_1 = x_0 + Attention_0(RMSNorm(x_0))             # First residual
x_2 = x_1 + FFN_0(RMSNorm(x_1))                   # Second residual

# Dense layer 1 (stage 2)
x_3 = x_2 + Attention_1(RMSNorm(x_2))
x_4 = x_3 + FFN_1(RMSNorm(x_3))

# Dense layer 2 (stage 3)
x_5 = x_4 + Attention_2(RMSNorm(x_4))
x_6 = x_5 + FFN_2(RMSNorm(x_5))

# MoE layer 3 (stage 4)
x_7 = x_6 + Attention_3(RMSNorm(x_6))
x_8 = x_7 + MoE_3(RMSNorm(x_7))                  # MoE replaces FFN

# ... layers 4-59 follow same pattern ...

# MoE layer 60 (stage 61)
x_121 = x_120 + Attention_60(RMSNorm(x_120))
x_122 = x_121 + MoE_60(RMSNorm(x_121))

# LM Head (stage 62)
logits = LMHead(RMSNorm(x_122))                   # (1, 129280)
token_id = sample(logits)
```

The hidden state shape remains `(1, 7168)` throughout all 61 layers -- the
same 14 KB tensor flows through every D2D exchange, every AllReduce, and every
residual add. This uniformity is why the pipeline can use fixed-size socket
FIFOs and page sizes at every stage.

**Design rationale -- fixed tensor size through the pipeline:** The DeepSeek V3
architecture maintains a constant hidden dimension throughout the layer stack.
This is exploited by the pipeline implementation to use identical D2D socket
configurations for every inter-stage link. No per-stage buffer sizing is needed,
no per-stage page size negotiation, and the same `PipelineBlock` code works for
stages 1 through 61.

The `DeepseekMinimalAllReduce` fuses the residual add with the all-reduce
operation (when `residual_tensor_mesh` is provided), saving one kernel launch
per residual add.

---

## 8.4.11 Position Tracking Through the Pipeline

The host-side `DeepSeekV3` class tracks the sequence position, which
increments by 1 for each prefill token and each decode step:

> **Source**: `model.py`, lines 188 and 214:
> ```python
> self._position += 1
> ```

This position is important for:
- **RoPE embeddings**: Each attention layer applies rotary position embeddings
  based on the current position in the sequence.
- **KV cache write index**: The position determines where new key-value pairs
  are written in the KV cache.
- **Causal mask**: For decode (single token), the causal mask is trivial but
  the position must be correct to index the precomputed mask table.

In the current loopback implementation, position tracking is maintained but not
yet used by device kernels. In the full pipeline, the position will be included
in the per-step metadata sent alongside the token ID (as noted in the model
docstring: "the engine tracks position for when the protocol is extended").

> **Cross-reference -> Chapter 4, Section 4.2**: RoPE application in the
> MLA attention pipeline.

---

## 8.4.12 Pipeline Bottleneck Analysis

### Per-Stage Latency Budget (B=1 Decode)

| Component | Estimated Latency | Notes |
|---|---|---|
| H2D (64 B) | ~1-2 us | PCIe fixed overhead |
| Embedding DRAM read | ~1-2 us | Single 14 KB read |
| D2D inter-stage hop | ~1-3 us | Per hop, depends on fabric load |
| Dense layer compute | ~50-100 us | Dominated by SDPA at long seq |
| MoE layer compute | ~100-200 us | Expert weight reads from DRAM |
| MoE communication | ~5-10 us | Broadcast + ReduceToOne |
| LM head matmul | ~20-50 us | Large vocab matmul |
| Fused argmax | ~5-10 us | Multi-level reduction |
| D2H (64 B) | ~1-2 us | PCIe fixed overhead |
| Loopback hop | ~1-3 us | Fabric ring return |

### Total Decode Step Estimate

```
  3 dense layers:   3 x ~75 us  = ~225 us
  58 MoE layers:    58 x ~150 us = ~8700 us
  1 LM head:        1 x ~60 us  = ~60 us
  62 D2D hops:      62 x ~2 us  = ~124 us
  H2D + D2H:        ~4 us
  Loopback:         ~3 us
  ---
  Total:            ~9100 us = ~9.1 ms per token (single-pod, B=1)
```

**Dominant bottleneck**: The 58 MoE layers account for approximately 96% of
the decode step latency. Within each MoE layer, the expert weight reads from
DRAM are the primary bottleneck at B=1.

### Scaling Considerations

- **Larger batch sizes** (B>1, not yet supported) would improve compute
  utilization by converting GEMVs into GEMMs, making the pipeline more
  compute-bound.
- **Inter-pod hops** (superpod) add extra latency at pod boundaries due to
  reduced channel count (4 vs 8 channels), potentially adding ~5-10 us per
  pod boundary crossing.
- **Sequence length growth** increases SDPA latency linearly (more KV cache
  pages to read), eventually making attention layers competitive with MoE
  layers as a bottleneck.

---

## 8.4.13 Data Size Budget Per Decode Step

For each decode step with B=1, the total data movement is:

| Transfer Type              | Count | Size Each  | Total         |
|----------------------------|-------|------------|---------------|
| H2D (token input)          | 1     | 64 B       | 64 B          |
| D2D inter-stage            | 62    | 14,336 B   | 888,832 B     |
| AllReduce intra-stage      | 61    | 14,336 B   | 874,496 B     |
| MoE Broadcast              | 58    | ~512 B     | ~29,696 B     |
| MoE ReduceToOne            | 58    | 14,336 B   | 831,488 B     |
| D2H (token output)         | 1     | 64 B       | 64 B          |
| Loopback (Stg62 -> Stg0)  | 1     | varies     | varies        |

The inter-stage D2D transfers dominate the communication budget, emphasizing
the importance of the fabric link bandwidth (8 channels intra-pod, 2 channels
inter-pod) discussed in Section 8.1.5.

---

## 8.4.14 Error Propagation During Decode

> **Cross-reference -> Section 8.3.9**: Full error/failure table covering H2D
> write timeout, D2H read timeout, D2D socket hang, loopback fabric error, and
> other runtime failure modes.

The decode data path adds three **compute-path** failure modes beyond the
communication errors in Section 8.3.9:

| Failure Point          | Observable Symptom                    | Recovery            |
|------------------------|---------------------------------------|---------------------|
| Embedding DRAM error   | Corrupted hidden states               | Silent corruption   |
| Expert compute error   | Incorrect reduction results           | Silent corruption   |
| Argmax error           | Wrong token ID produced               | Silent corruption   |

These are the most dangerous failure modes: silent corruptions produce incorrect
outputs without any error signal. The termination protocol (Section 8.3.6)
handles only clean shutdown; there is no error-triggered abort mechanism in the
current implementation.

---

## 8.4.15 Hardware-Software Co-Design Summary

| Software Decision | Hardware Constraint | Rationale |
|---|---|---|
| TP=2 for MLA | 4x2 mesh: 2 columns (LINE) | Column pair provides direct link; AllReduce is single exchange |
| EP=8 for MoE | 4x2 mesh: 8 devices total | Maximizes aggregate DRAM bandwidth for expert weight streaming |
| 3-level ReduceToOne tree | 4x2 mesh: 4-row RING + 2-col LINE | Exploits RING for vertical reduction, LINE for final horizontal |
| Pipeline parallelism | Limited inter-host bandwidth | Only 14 KB activation per hop vs. all-reduce at every layer |
| Snake-order host traversal | Galaxy pod physical wiring | Minimizes hops: u08<->u02 within cabinet, u02<->u02 across |
| Loopback ring topology | D2D fabric availability | Eliminates PCIe round-trip for next-token embedding (~10-20 us) |
| HOST_PUSH for decode | Per-token host-device dependency | Host pushes as soon as sample is ready; lowest latency for small payloads |
| DEVICE_PULL for prefill | Host writes all tokens upfront | Device pulls at own pace; no per-token synchronization |
| Slow dispatch mandatory | generic_op() interface | All kernels use generic_op; slow dispatch provides synchronous completion |
| Fused embedding on stage 0 | DRAM table on embedding device | Single kernel does H2D receive + DRAM lookup + D2D forward |

---

## 8.4.16 Guide Visual Map

This diagram serves as a visual index to the entire guide, showing which
chapter covers each component of the decode pipeline:

```
 DEEPSEEK V3 B1 ON TENSTORRENT: GUIDE MAP
 ==========================================

 +-- Ch 1: Architecture Overview ----------------------------------------+
 |   DeepSeek V3 model structure, 671B params, MLA + MoE design          |
 +-----------------------------------------------------------------------+

 +-- Ch 2: Kernel System & CB Management --------------------------------+
 |   UnifiedKernelDescriptor, CircularBufferIdManager, generic_op        |
 +-----------------------------------------------------------------------+

 +-- Ch 3: Micro-Op Catalog ---------------------------------------------+
 |   D2D exchange, AllReduce, Broadcast, ReduceToOne, HostInterface      |
 +-----------------------------------------------------------------------+

          HOST                                                HOST
           |                                                   ^
           v                                                   |
 +-- Ch 7: Host I/O (model.py, HostInterface) ---+  +-- Ch 7 -+
 |   H2D sockets, D2H sockets, prefill/decode    |  | D2H path|
 +------------------------------------------------+  +---------+
           |                                                   ^
           v                                                   |
 +-- Ch 3: Embedding Layer ---+  +-- Ch 8: Pipeline (D2D) -+  |
 |   DRAM lookup, fused H2D   |  | SocketInterface, fabric  |  |
 +-----------------------------+  +-------------------------+  |
           |                            |                      |
           v                            v                      |
 +-- Ch 4: MLA Attention ---------------------------------+    |
 |   Pre-SDPA: Q/KV projections, RoPE                     |    |
 |   Flash MLA: compressed KV cache attention              |    |
 |   Post-SDPA: O projection, AllReduce(TP=2)             |    |
 +---------------------------------------------------------+    |
           |                                                    |
           v                                                    |
 +-- Ch 5: MoE Layer (for stages 4-61) -------------------+    |
 |   Gate + Broadcast, Expert Dispatch, ReduceToOne        |    |
 +-- Ch 6: Dense FFN (for stages 1-3) --------------------+    |
 |   Gate/Up/Down projections                              |    |
 +---------------------------------------------------------+    |
           |                                                    |
           v                                                    |
 +-- Ch 3: LM Head --------+  +-- Ch 8: Loopback ---------+   |
 |   Final RMSNorm + linear |  | D2D ring back to Stage 0  |---+
 +---------------------------+  +---------------------------+

 +-- Ch 8: Scaleout Configuration ---------------------------------+
 |   Pipeline YAML, mesh graph textproto, rank bindings, MPI launch |
 +------------------------------------------------------------------+
```
