# 01 -- adaRMSNorm FusedOp: Composed vs Fused Kernel Design

## Context

Pi0.5's action expert uses **adaptive RMSNorm (adaRMSNorm)** instead of standard
RMSNorm. Where standard RMSNorm applies a learned per-channel scale, adaRMSNorm
derives scale, shift, and gate from a conditioning signal -- the diffusion timestep
embedding. This is the mechanism that lets the action expert modulate its behavior
at each denoising step.

The reference implementation in `openpi/models/gemma.py` (class `RMSNorm`) shows the
two paths clearly:

```python
# Standard path (cond is None):
normed = x * reciprocal(sqrt(var + eps))
normed = normed * (1 + scale)          # learned scale parameter

# Adaptive path (cond is not None):
normed = x * reciprocal(sqrt(var + eps))
modulation = Dense(width * 3)(cond)     # [B, width*3] from [B, width]
scale, shift, gate = split(modulation, 3, axis=-1)
normed = normed * (1 + scale) + shift   # apply scale and shift
# gate is returned separately, used for gated residual
```

Implementing this as a Blaze FusedOp requires chaining multiple MicroOps inside a
single device program -- keeping all intermediates in L1 circular buffers, never
touching DRAM between steps.

## Key Takeaways

1. adaRMSNorm is **not** a simple parameter swap on top of RMSNorm -- it adds a
   matmul (the modulation projection), a three-way split, and an element-wise
   scale+shift. The gate output is consumed downstream by the residual add.

2. The composed FusedOp approach chains existing MicroOps (`RMSNorm.emit()`,
   `Matmul.emit()`, element-wise ops) on the same `FusedProgram`. This is
   preferred over writing a monolithic kernel because it reuses validated
   building blocks.

3. For width=1024 (the action expert), the modulation weight matrix is
   `[1024, 3072]` (width to width*3). At bfloat16, this is 6 MB per layer --
   well within DRAM budget but must be streamed through L1 tile-by-tile.

4. The CBHandle chain is the critical design element: each MicroOp's output
   CBHandle becomes the next MicroOp's input. No host intervention occurs
   between steps.

## Composed vs Monolithic: Why Composition Wins

There are two ways to implement adaRMSNorm as a Blaze FusedOp:

**Option 1: Monolithic kernel (not recommended)**
Write a single C++ kernel that performs norm + matmul + split + apply. This requires
implementing all the tile math from scratch, including the matmul accumulation
pattern, RMSNorm's reciprocal-sqrt-reduce, and the three-way tensor split. Any bug
requires rebuilding the entire kernel.

**Option 2: Composed FusedOp (recommended)**
Chain existing MicroOps using `FusedProgram`:

```
                             FusedProgram boundary
    +----------------------------------------------------------------+
    |                                                                  |
    |  input ──> [RMSNorm] ──> normed_cb                              |
    |                                                                  |
    |  cond ──> [Matmul(W_mod)] ──> modulation_cb                     |
    |                                                                  |
    |  modulation_cb ──> [Split3] ──> scale_cb, shift_cb, gate_cb     |
    |                                                                  |
    |  normed_cb + scale_cb + shift_cb ──> [ScaleShift] ──> output_cb |
    |                                                                  |
    |  gate_cb ──> (returned as second output)                        |
    |                                                                  |
    +----------------------------------------------------------------+
```

Option 2 is preferred because:
- Each MicroOp is independently tested and validated
- The CB plumbing is handled by `FusedProgram`'s allocator
- New ops (e.g., `Split3`, `ScaleShift`) are small and reusable
- The composed kernel is auto-generated from the MicroOp chain

## CBHandle Chain in Detail

The `CBHandle` (defined in `blaze/cb_handle.py`) is a reference to a circular buffer
carrying metadata: `cb_id`, `num_pages`, `page_size`, `core_ranges`, `data_format`,
and `tile_desc`. When one MicroOp's `emit()` returns a `CBHandle`, the next MicroOp
receives it as input and knows exactly how to read the data.

```
Step-by-step CB flow for adaRMSNorm(input, gamma, cond, W_mod):

  1. RMSNorm.emit(f, input_cb, gamma_tensor)
     IN:  input_cb  (activation, FIFO)
     IN:  gamma_cb  (weight, tensor-backed)
     OUT: normed_cb (CB 2, FIFO, page_size=2048B for 32x32 bf16 tiles)

  2. Matmul.emit(f, cond_cb, w_mod_tensor)
     IN:  cond_cb   (timestep embedding, FIFO)
     IN:  w_mod_cb  (weight [1024,3072], tensor-backed / DRAM-streamed)
     OUT: mod_cb    (CB 4, FIFO, page_size=2048B, width=3072)

  3. Split3.emit(f, mod_cb)
     IN:  mod_cb    (width=3072)
     OUT: scale_cb  (CB 5, width=1024)
     OUT: shift_cb  (CB 6, width=1024)
     OUT: gate_cb   (CB 7, width=1024)

  4. ScaleShift.emit(f, normed_cb, scale_cb, shift_cb)
     IN:  normed_cb, scale_cb, shift_cb
     OUT: output_cb (CB 8, FIFO, width=1024)

  Return: (output_cb, gate_cb)
```

Each `CBHandle` in this chain satisfies the FIFO contract: the producing op pushes
tiles, the consuming op pops them. The `require_fifo_handle()` guard in
`cb_handle.py` ensures that ops expecting streaming data never accidentally receive
a direct-address handle.

## L1 Budget Analysis: Action Expert (width=1024)

The action expert has width=1024. With 32x32 bfloat16 tiles:
- One tile = 32 * 32 * 2 bytes = 2048 bytes (2 KB)
- Width 1024 = 1024/32 = 32 tiles per row

Per-layer L1 budget for the adaRMSNorm FusedOp:

```
CB Allocation Table (double-buffered where noted):

CB  | Purpose        | Pages | Page Size | Total   | Notes
----|----------------|-------|-----------|---------|------------------
 0  | input          |   2   |  2048 B   |   4 KB  | double-buffered
 1  | gamma          |   1   |  2048 B   |   2 KB  | tensor-backed
 2  | normed out     |   2   |  2048 B   |   4 KB  | double-buffered
 3  | cond input     |   2   |  2048 B   |   4 KB  | timestep embedding
 4  | mod out        |   2   |  2048 B   |   4 KB  | modulation [3072]
 5  | scale          |   1   |  2048 B   |   2 KB  | from split
 6  | shift          |   1   |  2048 B   |   2 KB  | from split
 7  | gate           |   1   |  2048 B   |   2 KB  | from split (held)
 8  | output         |   2   |  2048 B   |   4 KB  | final result
----|----------------|-------|-----------|---------|------------------
    | TOTAL CB L1    |       |           |  28 KB  |
```

**Modulation weight matrix `W_mod` [1024, 3072]:**
- Total size: 1024 * 3072 * 2 = 6,291,456 bytes (6 MB)
- This lives in DRAM. The matmul op streams it through L1 in tiles.
- Per-core weight shard (assuming 64 cores): 6 MB / 64 = 96 KB per core
- This fits comfortably; Wormhole L1 per core is ~1.5 MB

The adaRMSNorm FusedOp's CB overhead is modest (28 KB scratch) because the expensive
modulation weights are streamed from DRAM, not pre-loaded into L1.

## Integration with Standard RMSNorm

The standard RMSNorm case (VLM expert, `cond=None`) is simpler -- it is just the
existing `RMSNorm` MicroOp at `blaze/ops/rmsnorm/op.py`. The adaRMSNorm FusedOp
should be a separate class that *contains* an `RMSNorm.emit()` call as its first
step, sharing the same validated norm kernel:

```
class AdaRMSNorm(FusedOp):
    name = "ada_rmsnorm"

    input    = Input()    # activation
    gamma    = Input()    # norm scale (unused in ada path, but kept for API parity)
    cond     = Input()    # timestep conditioning [B, width]
    w_mod    = Input()    # modulation weights [width, width*3]
    output   = Output()   # normed + scale + shift result
    gate_out = Output()   # gate for gated residual

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        cls.emit(f, tensors["input"], tensors["gamma"],
                 tensors["cond"], tensors["w_mod"], ...)

    @staticmethod
    def emit(f, input_handle, gamma, cond, w_mod, *, prefix, cores, ...):
        normed = RMSNorm.emit(f, input_handle, gamma, prefix=..., cores=cores)
        mod    = Matmul.emit(f, cond, w_mod, prefix=..., cores=cores)
        scale, shift, gate = Split3.emit(f, mod, prefix=..., cores=cores)
        output = ScaleShift.emit(f, normed, scale, shift, prefix=..., cores=cores)
        return output, gate
```

This design means:
- The VLM expert calls `RMSNorm.emit()` directly (no adaRMS overhead)
- The action expert calls `AdaRMSNorm.emit()` which internally calls `RMSNorm.emit()`
- Both share the same tested norm kernel

## The Gate Output and Gated Residual

The gate tensor produced by adaRMSNorm is not consumed immediately. It is held in
its CB until the post-attention (or post-FFN) residual add. The reference code in
`gemma.py` shows:

```python
def _gated_residual(x, y, gate):
    if gate is None:
        return x + y       # VLM expert: standard residual
    return x + y * gate    # action expert: gated residual
```

In Blaze terms, the `gate_cb` from `AdaRMSNorm.emit()` is passed to a
`GatedResidualAdd.emit()` call later in the block. The CB must remain allocated
(not freed/reused) across the entire attention or FFN compute. This is a
scheduling constraint the FusedProgram must respect.

## ASCII Diagram: adaRMSNorm Data Flow

```
  timestep ──> [posemb_sincos] ──> [time_mlp_in] ──> [SiLU] ──> [time_mlp_out]
                                                                       |
                                                                   cond [B, 1024]
                                                                       |
                                                                       v
  input [B, S, 1024] ──> [RMSNorm] ──> normed          cond ──> [Matmul W_mod]
                              |                                        |
                              |                               modulation [B, 3072]
                              |                                        |
                              |                               [Split3 by width]
                              |                              /    |        \
                              |                         scale   shift     gate
                              |                          |       |          |
                              +--------> [ScaleShift] <--+-------+          |
                                              |                             |
                                         output [B, S, 1024]               |
                                              |                             |
                                              v                             v
                                        [Attention / FFN]          [GatedResidualAdd]
                                              |                             |
                                              +-----------------------------+
                                              |
                                         residual output
```

## Source References

- `openpi/models/gemma.py:113-131` -- RMSNorm class with standard/adaptive paths
- `openpi/models/gemma.py:453-459` -- `_gated_residual` function
- `openpi/models/gemma.py:293-333` -- Block class showing gate flow through attention + FFN
- `blaze/ops/rmsnorm/op.py` -- RMSNorm MicroOp with `emit()` and `compose()`
- `blaze/cb_handle.py:35-95` -- CBHandle dataclass with FIFO/DIRECT_ADDRESS modes
- `blaze/blaze_op.py:408-429` -- FusedOp base class requiring `compose()` override
- `blaze/fused_program.py:1-17` -- FusedProgram usage pattern
